// SPDX-License-Identifier: MIT
// GPU qualification consumes the private portable execution boundary.
// No Objective-C types, test-only dispatch implementation or CPU fallback.
#include <device/compute.hpp>
#include <simulation/geometry_storage.hpp>
#include <simulation/avbd_schedule.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace parallel_mater;
using namespace parallel_mater::simulation;
namespace compute = parallel_mater::device::compute;
using compute::BufferPtr;

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

struct Kernel {
    compute::KernelPtr native;
    std::uint32_t threads{};
    void check_pointee(const char *name, std::size_t size) const {
        const auto &field = native->parameter(name);
        require(field.pointer && field.pointee_size == size,
            std::string("host/native pointee size mismatch: ") + name);
    }
};

struct Arguments : compute::Arguments {
    const Kernel &kernel;
    explicit Arguments(const Kernel &value) : compute::Arguments(value.native), kernel(value) {}
    void buffer(const char *name, const BufferPtr &value) { bind(name, value); }
};

class Gpu {
public:
    std::unique_ptr<compute::Device> device;
    std::map<std::string, Kernel> kernels;
    explicit Gpu(std::unique_ptr<compute::Device> value) : device(std::move(value)) {}
    BufferPtr buffer(std::size_t size, const void *data = nullptr) const {
        auto result = device->allocate(size, true);
        if (data) std::memcpy(result->host_address(), data, size);
        else std::memset(result->host_address(), 0, size);
        return result;
    }
    template<class T> BufferPtr buffer(const std::vector<T> &data) const {
        return buffer(data.size() * sizeof(T), data.data());
    }
    void load(const std::string &manifest, bool execution_only) {
        std::ifstream input(manifest);
        require(input.good(), "cannot open Metal kernel manifest");
        std::string line;
        while (std::getline(input, line)) {
            const auto first = line.find('|');
            const auto second = line.find('|', first + 1);
            require(first != std::string::npos && second != std::string::npos,
                "invalid Metal kernel manifest record");
            const auto name = line.substr(first + 1, second - first - 1);
            const auto base = line.substr(second + 1);
            if (execution_only && name != "metal_device_contract" && name != "metal_record_contract" &&
                name != "pm_integrate_rigid_bodies" &&
                name != "deformable_project_links" &&
                name != "deformable_project_links_warp" &&
                name != "soft_body_damp_springs" &&
                name != "soft_body_damp_springs_warp" &&
                name != "soft_body_realtime_predict" &&
                name != "soft_body_realtime_project_links" &&
                name != "soft_body_realtime_apply_position_delta" &&
                name != "soft_body_realtime_project_shape" &&
                name != "soft_body_realtime_project_contacts" &&
                name != "soft_body_realtime_commit" &&
                name != "prepare_avbd_kernel" && name != "solve_avbd_kernel" &&
                name != "solve_avbd_stage_kernel") continue;
            auto native = device->load_kernel(base + ".metallib", base + ".json", name);
            const auto threads = native->threads.x;
            require(kernels.emplace(name, Kernel{std::move(native), threads}).second,
                name + ": duplicate kernel");
        }
        require(!kernels.empty(), "empty Metal kernel manifest");
        std::cout << "Native pipelines and argument layouts: " << kernels.size() << " passed\n";
    }
    void encode(const Arguments &arguments, std::uint32_t groups,
                const std::unique_ptr<compute::CommandList> &command) const {
        command->dispatch(arguments, {groups});
    }
    double complete(const std::unique_ptr<compute::CommandList> &command) const {
        auto completion = command->submit();
        completion->wait();
        const double elapsed = completion->elapsed_seconds();
        require(completion->ready() && elapsed >= 0,
            "invalid GPU completion/timestamps");
        return elapsed;
    }
    void run(const Arguments &arguments, std::uint32_t groups) const {
        auto command = device->commands();
        encode(arguments, groups, command);
        complete(command);
    }
};

void check_device_operations(Gpu &gpu) {
    constexpr std::uint32_t count = 8193;
    constexpr std::uint64_t addend = 0x1fffffff0ULL;
    const Kernel &kernel = gpu.kernels.at("metal_device_contract");
    const std::uint32_t groups = (count + kernel.threads - 1) / kernel.threads;
    for (const std::uint64_t initial : {0ULL, 0xfffffff0ULL, 0xfffffffffffffff0ULL}) {
        auto total = gpu.buffer(sizeof(initial), &initial);
        auto mask = gpu.buffer(sizeof(std::uint64_t));
        auto sum = gpu.buffer(sizeof(Vec3));
        auto indices = gpu.buffer(count * sizeof(std::uint32_t));
        auto maximum = gpu.buffer(sizeof(float));
        const float initial_minimum = std::numeric_limits<float>::max();
        auto minimum = gpu.buffer(sizeof(float), &initial_minimum);
        Arguments arguments(kernel);
        arguments.set("count", count);
        arguments.set("addend", addend);
        arguments.buffer("total", total);
        arguments.buffer("mask", mask);
        arguments.buffer("vector_sum", sum);
        arguments.buffer("indices", indices);
        arguments.buffer("maximum", maximum);
        arguments.buffer("minimum", minimum);
        gpu.run(arguments, groups);
        require(*static_cast<std::uint64_t *>(total->host_address()) == initial + addend * count,
            "64-bit reduction carry/wrap mismatch");
        require(*static_cast<std::uint64_t *>(mask->host_address()) == UINT64_MAX,
            "64-bit mask reduction mismatch");
        const auto vector = *static_cast<Vec3 *>(sum->host_address());
        require(vector.x == count && vector.y == 2 * count && vector.z == -float(count),
            "floating-point vector atomic mismatch");
        require(*static_cast<float *>(maximum->host_address()) == count &&
            *static_cast<float *>(minimum->host_address()) == 1.0f, "float min/max atomic mismatch");
        const auto *values = static_cast<std::uint32_t *>(indices->host_address());
        for (std::uint32_t index = 0; index < count; ++index)
            require(values[index] == index + groups * kernel.threads, "dispatch index mismatch");
    }
    std::cout << "GPU dispatch and atomic carry/wrap/contention tests passed\n";
}

template<class Function> void rejects(Function &&function, const char *message) {
    try { function(); }
    catch (const std::invalid_argument &) { return; }
    catch (const std::logic_error &) { return; }
    throw std::runtime_error(message);
}

void check_transfers_and_records(Gpu &gpu) {
    auto memory = gpu.device->allocate(64, false);
    require(memory->host_address() == nullptr, "device-only storage unexpectedly host-visible");
    auto readback = gpu.buffer(64);
    auto command = gpu.device->commands();
    std::array<std::byte, 64> source;
    source.fill(std::byte{37});
    command->upload(memory, 0, source);
    source.fill(std::byte{99}); // upload must already own a snapshot
    command->fill(memory, 16, 8, 0xab);
    command->copy(readback, 0, memory, 0, 64);
    rejects([&] { command->copy(memory, 0, memory, 4, 16); }, "overlapping copy accepted");
    rejects([&] { command->fill(memory, 60, 8, 0); }, "out-of-bounds fill accepted");
    rejects([&] { command->upload(memory, SIZE_MAX, source); }, "overflowing upload range accepted");
    auto completion = command->submit();
    rejects([&] { (void)command->submit(); }, "double submission accepted");
    rejects([&] { command->fill(memory, 0, 4, 0); }, "encoding after submission accepted");
    memory.reset();
    command.reset();
    completion->wait();
    const auto bytes = static_cast<const std::uint8_t *>(readback->host_address());
    for (unsigned i = 0; i < 64; ++i)
        require(bytes[i] == (i >= 16 && i < 24 ? 0xab : 37), "asynchronous transfer snapshot mismatch");

    constexpr std::uint32_t count = 257;
    struct RecordData {
        bool enabled{}, negate{};
        Vec3 gain{};
        float *inputs[2]{};
        float *output{};
        std::uint32_t count{};
    };
    const auto &kernel = gpu.kernels.at("metal_record_contract");
    kernel.check_pointee("records", sizeof(RecordData));
    const auto &layout = kernel.native->parameter("inline_record");
    require(layout.size == sizeof(RecordData) &&
        layout.fields.at("negate").offset == offsetof(RecordData, negate) &&
        layout.fields.at("inputs").offset == offsetof(RecordData, inputs) &&
        layout.fields.at("output").offset == offsetof(RecordData, output), "nested native record layout mismatch");
    auto output_a = gpu.buffer(count * sizeof(float));
    auto output_b = gpu.buffer(count * sizeof(float));
    {
        std::vector<float> values(count + 1);
        for (std::uint32_t i = 0; i <= count; ++i) values[i] = float(i);
        auto input = gpu.device->allocate(values.size() * sizeof(float), false);
        auto records = gpu.device->allocate(sizeof(RecordData), false);
        auto commands = gpu.device->commands();
        commands->upload(input, 0, std::as_bytes(std::span(values)));
        // Original record contains host pointers; relocation must not modify it.
        RecordData host{true, false, {2, 3, 0}, {values.data(), values.data()}, values.data(), count};
        compute::Record inline_record(layout, host);
        rejects([&] { (void)inline_record.bytes(); }, "unrelocated record accepted");
        rejects([&] { inline_record.bind("inputs[0]", input, 1); }, "misaligned record pointer accepted");
        inline_record.bind("inputs[0]", input);
        inline_record.bind("inputs[1]", input, sizeof(float));
        inline_record.bind("output", output_a);
        host.negate = true;
        compute::Record indirect(kernel.native->parameter("records"), host);
        indirect.bind("inputs[0]", input);
        indirect.bind("inputs[1]", input, sizeof(float));
        indirect.bind("output", output_b);
        commands->upload(records, 0, indirect.bytes());
        Arguments arguments(kernel);
        rejects([&] { arguments.set("inline_record", host); }, "raw host-pointer record accepted");
        arguments.record("inline_record", inline_record);
        arguments.bind("records", records);
        for (const auto &binding : indirect.bindings())
            arguments.reference("records[]." + binding.name, binding.range.buffer,
                binding.range.offset, binding.range.size, binding.range.access);
        gpu.encode(arguments, (count + kernel.threads - 1) / kernel.threads, commands);
        completion = commands->submit();
        require(host.inputs[0] == values.data() && host.inputs[1] == values.data(),
            "relocation mutated the host record");
        // Drop everything but output and completion before the GPU finishes.
    }
    completion->wait();
    auto a = static_cast<const float *>(output_a->host_address());
    auto b = static_cast<const float *>(output_b->host_address());
    for (std::uint32_t i = 0; i < count; ++i)
        require(a[i] == 5.0f * i + 3.0f && b[i] == -a[i], "GPU nested-pointer relocation mismatch");

    // Submission token destruction is not cancellation. Work remains ordered
    // and resources remain alive until a later submission completes.
    {
        auto commands = gpu.device->commands();
        auto temporary = gpu.device->allocate(64, false);
        commands->fill(temporary, 0, 64, 17);
        commands->copy(readback, 0, temporary, 0, 64);
        (void)commands->submit();
    }
    auto fence = gpu.device->commands()->submit();
    fence->wait();
    for (unsigned i = 0; i < 64; ++i) require(bytes[i] == 17, "dropped-token resource lifetime failure");
    std::cout << "Async uploads/copies, private memory, resource lifetimes and nested pointer records passed\n";
}

void check_device_teardown() {
    auto device = compute::create_metal_device();
    require(bool(device), "cannot create teardown test device");
    auto readback = device->allocate(64, true);
    auto survivor = device->commands();
    {
        auto temporary = device->allocate(64, false);
        auto command = device->commands();
        command->fill(temporary, 0, 64, 91);
        command->copy(readback, 0, temporary, 0, 64);
        (void)command->submit();
    }
    device.reset(); // Must drain even though no caller kept a completion token.
    const auto bytes = static_cast<const std::uint8_t *>(readback->host_address());
    for (unsigned i = 0; i < 64; ++i)
        require(bytes[i] == 91, "device teardown did not drain pending GPU transfer");
    rejects([&] { (void)survivor->submit(); }, "submission accepted after device teardown");
    std::cout << "Native device teardown drains dropped-token GPU work\n";
}

void check_rigid_integration(Gpu &gpu) {
    constexpr std::uint32_t count = 513;
    const auto &kernel = gpu.kernels.at("pm_integrate_rigid_bodies");
    kernel.check_pointee("parameters", sizeof(BodyParameters));
    kernel.check_pointee("accumulators", sizeof(BodyAccumulator));
    kernel.check_pointee("targets", sizeof(KinematicTarget));
    kernel.check_pointee("input", sizeof(RigidBodyState));
    std::vector<BodyParameters> parameters(count);
    std::vector<RigidBodyState> states(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        parameters[index].motion = index % 2 ? MotionType::dynamic : MotionType::static_body;
        parameters[index].inverse_mass = 1.0f;
        parameters[index].maximum_linear_speed = 1000.0f;
        parameters[index].maximum_angular_speed = 1000.0f;
        states[index].position = {float(index), 10.0f, 0.0f};
        states[index].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    }
    auto output = gpu.buffer(count * sizeof(RigidBodyState));
    Arguments arguments(kernel);
    arguments.buffer("parameters", gpu.buffer(parameters));
    arguments.buffer("input", gpu.buffer(states));
    arguments.buffer("output", output);
    arguments.buffer("accumulators", gpu.buffer(count * sizeof(BodyAccumulator)));
    arguments.buffer("targets", gpu.buffer(count * sizeof(KinematicTarget)));
    arguments.set("count", count);
    arguments.set("gravity", Vec3{0.0f, -10.0f, 0.0f});
    arguments.set("timestep", 0.1f);
    arguments.set("remaining_substeps", std::uint32_t{1});
    arguments.set("apply_impulses", false);
    gpu.run(arguments, (count + kernel.threads - 1) / kernel.threads);
    const auto *result = static_cast<RigidBodyState *>(output->host_address());
    for (std::uint32_t index = 0; index < count; ++index) {
        const float expected_y = index % 2 ? 9.9f : 10.0f;
        require(result[index].position.x == float(index) &&
            std::abs(result[index].position.y - expected_y) < 1.0e-5f &&
            result[index].linear_velocity.y == (index % 2 ? -1.0f : 0.0f),
            "shared rigid integration failed at body " + std::to_string(index));
    }
    std::cout << "Shared rigid integration: 513 bodies, multiple GPU threadgroups passed\n";
}

void check_soft_body_simd(Gpu &gpu) {
    struct Neighbor { std::uint32_t index; float rest, compliance; std::uint32_t bond; };
    struct SoftNeighbor { std::uint32_t index; float rest; };
    constexpr std::uint32_t count = 37;
    const std::uint32_t degrees[]{0, 1, 17, 33, 65};
    std::vector<Vec3> positions(count), velocities(count);
    std::vector<float> inverse_masses(count), minimum_lengths(count, 0.1f);
    std::vector<std::uint32_t> offsets;
    std::vector<Neighbor> neighbors;
    std::vector<SoftNeighbor> soft_neighbors;
    for (std::uint32_t node = 0; node < count; ++node) {
        positions[node] = {float(node) * 0.2f, float(node % 3) * 0.3f, 0};
        velocities[node] = {float(node % 5), -float(node % 7), 0.1f};
        inverse_masses[node] = node % 7 ? 1.0f : 0.0f;
        offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));
        for (std::uint32_t edge = 0; edge < degrees[node % 5]; ++edge) {
            const auto other = (node + edge + 1) % count;
            neighbors.push_back({other, 0.1f, 0.001f, 0});
            soft_neighbors.push_back({other, 0.1f});
        }
    }
    offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));
    const auto position_buffer = gpu.buffer(positions);
    const auto velocity_buffer = gpu.buffer(velocities);
    const auto mass_buffer = gpu.buffer(inverse_masses);
    const auto offset_buffer = gpu.buffer(offsets);
    const auto neighbor_buffer = gpu.buffer(neighbors);
    const auto soft_neighbor_buffer = gpu.buffer(soft_neighbors);
    const auto minimum_buffer = gpu.buffer(minimum_lengths);
    const std::uint8_t active = 1;
    const auto active_buffer = gpu.buffer(sizeof(active), &active);
    for (const bool damping : {false, true}) {
        const auto &scalar = gpu.kernels.at(damping ? "soft_body_damp_springs" : "deformable_project_links");
        const auto &simd = gpu.kernels.at(damping ? "soft_body_damp_springs_warp" : "deformable_project_links_warp");
        require(simd.native->simd_width == 32, "unsupported Metal SIMD width");
        scalar.check_pointee("neighbors", sizeof(Neighbor));
        simd.check_pointee("neighbors", sizeof(SoftNeighbor));
        const auto scalar_output = gpu.buffer(count * sizeof(Vec3));
        const auto simd_output = gpu.buffer(count * sizeof(Vec3));
        for (const bool wide : {false, true}) {
            Arguments arguments(wide ? simd : scalar);
            arguments.buffer("positions", position_buffer);
            arguments.buffer("inverse_masses", mass_buffer);
            arguments.buffer("offsets", offset_buffer);
            arguments.buffer("neighbors", wide ? soft_neighbor_buffer : neighbor_buffer);
            arguments.buffer(damping ? "output" : "scratch", wide ? simd_output : scalar_output);
            arguments.set("count", count);
            if (!wide) arguments.buffer("bond_active", active_buffer);
            if (damping) {
                arguments.buffer("velocities", velocity_buffer);
                arguments.set("damping", 0.2f);
                arguments.set("maximum_speed", 100.0f);
            } else {
                arguments.set("dt", 0.01f);
                arguments.set("maximum_projection_fraction", 0.1f);
                if (wide) {
                    arguments.set("compliance", 0.001f);
                    arguments.buffer("minimum_rest_lengths", minimum_buffer);
                }
            }
            gpu.run(arguments, wide ? (count + simd.threads / 32 - 1) / (simd.threads / 32)
                                    : (count + scalar.threads - 1) / scalar.threads);
        }
        const auto *expected = static_cast<Vec3 *>(scalar_output->host_address());
        const auto *actual = static_cast<Vec3 *>(simd_output->host_address());
        for (std::uint32_t node = 0; node < count; ++node)
            require(std::abs(expected[node].x - actual[node].x) < 1.0e-5f &&
                std::abs(expected[node].y - actual[node].y) < 1.0e-5f &&
                std::abs(expected[node].z - actual[node].z) < 1.0e-5f,
                "SIMD/scalar GPU soft-body mismatch at node " + std::to_string(node));
    }
    std::cout << "Soft-body SIMD tails, pinned nodes, variable degrees: projection/damping passed\n";
}

void check_soft_body_realtime(Gpu &gpu) {
    constexpr std::uint32_t count = 4;
    constexpr float dt = 1.0f / 60.0f;
    constexpr std::uint32_t profile = 1;
    constexpr std::uint32_t sweeps = 8;
    constexpr float radius = 0.005f;
    struct SoftNeighbor { std::uint32_t index; float rest; };
    struct ShapeMatrixHost { Vec3 columns[3]; };
    static_assert(sizeof(SoftNeighbor) == 8);
    static_assert(sizeof(ShapeMatrixHost) == 36);

    const auto subtract = [](Vec3 a, Vec3 b) {
        return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
    };
    const auto dot = [](Vec3 a, Vec3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    };
    const auto cross = [](Vec3 a, Vec3 b) {
        return Vec3{a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
    };
    const auto length = [&](Vec3 value) {
        return std::sqrt(dot(value, value));
    };
    const auto tetra_volume = [&](const Vec3 *positions) {
        return std::abs(dot(subtract(positions[1], positions[0]),
            cross(subtract(positions[2], positions[0]),
                subtract(positions[3], positions[0])))) / 6.0f;
    };

    std::vector<Vec3> rest{{0.0f, 0.15f, 0.0f},
        {0.1f, 0.15f, 0.0f}, {0.0f, 0.25f, 0.0f},
        {0.0f, 0.15f, 0.1f}};
    // Exercise the declared 100:1 interacting-mass limit on production Metal.
    std::vector<float> inverse_masses{50.0f, 0.5f, 50.0f, 0.5f};
    float movable_mass = 0.0f;
    Vec3 rest_center{};
    for (std::uint32_t node = 0; node < count; ++node) {
        const float mass = 1.0f / inverse_masses[node];
        movable_mass += mass;
        rest_center.x += mass * rest[node].x;
        rest_center.y += mass * rest[node].y;
        rest_center.z += mass * rest[node].z;
    }
    rest_center.x /= movable_mass;
    rest_center.y /= movable_mass;
    rest_center.z /= movable_mass;

    float covariance[3][3]{};
    for (std::uint32_t node = 0; node < count; ++node) {
        const float mass = 1.0f / inverse_masses[node];
        const Vec3 value = subtract(rest[node], rest_center);
        const float v[3]{value.x, value.y, value.z};
        for (unsigned row = 0; row < 3; ++row)
            for (unsigned column = 0; column < 3; ++column)
                covariance[row][column] += mass * v[row] * v[column];
    }
    float augmented[3][6]{};
    for (unsigned row = 0; row < 3; ++row)
        for (unsigned column = 0; column < 3; ++column) {
            augmented[row][column] = covariance[row][column];
            augmented[row][column + 3] = row == column ? 1.0f : 0.0f;
        }
    for (unsigned pivot = 0; pivot < 3; ++pivot) {
        unsigned best = pivot;
        for (unsigned row = pivot + 1; row < 3; ++row)
            if (std::abs(augmented[row][pivot]) >
                std::abs(augmented[best][pivot])) best = row;
        require(std::abs(augmented[best][pivot]) > 1.0e-12f,
            "soft-body rest covariance is singular");
        for (unsigned column = 0; column < 6; ++column)
            std::swap(augmented[pivot][column], augmented[best][column]);
        const float divisor = augmented[pivot][pivot];
        for (float &value : augmented[pivot]) value /= divisor;
        for (unsigned row = 0; row < 3; ++row) {
            if (row == pivot) continue;
            const float factor = augmented[row][pivot];
            for (unsigned column = 0; column < 6; ++column)
                augmented[row][column] -= factor * augmented[pivot][column];
        }
    }
    ShapeMatrixHost inverse_rest{};
    for (unsigned column = 0; column < 3; ++column) {
        inverse_rest.columns[column] = {augmented[0][column + 3],
            augmented[1][column + 3], augmented[2][column + 3]};
    }

    std::vector<std::uint32_t> offsets;
    std::vector<SoftNeighbor> neighbors;
    for (std::uint32_t node = 0; node < count; ++node) {
        offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));
        for (std::uint32_t other = 0; other < count; ++other) {
            if (other == node) continue;
            neighbors.push_back({other, length(subtract(rest[node], rest[other]))});
        }
    }
    offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));

    std::vector<Vec3> initial = rest;
    initial[1].x += 0.02f;
    initial[2].y += 0.015f;
    initial[3].z += 0.03f;
    std::vector<Vec3> velocities(count);
    std::vector<float> characteristic_lengths(count, 0.1f);
    std::vector<Vec3> normals(count, Vec3{0.0f, 1.0f, 0.0f});
    std::vector<float> plane_offsets(count, 0.0f);
    std::vector<std::uint8_t> contact_active(count, 1);
    const Quaternion identity{0.0f, 0.0f, 0.0f, 1.0f};
    auto positions = gpu.buffer(initial);
    auto velocity_buffer = gpu.buffer(velocities);
    auto previous = gpu.buffer(count * sizeof(Vec3));
    auto corrections = gpu.buffer(count * sizeof(Vec3));
    auto correction_used = gpu.buffer(count * sizeof(float));
    auto mass_buffer = gpu.buffer(inverse_masses);
    auto characteristic_buffer = gpu.buffer(characteristic_lengths);
    auto offset_buffer = gpu.buffer(offsets);
    auto neighbor_buffer = gpu.buffer(neighbors);
    auto rest_buffer = gpu.buffer(rest);
    auto orientation = gpu.buffer(sizeof(identity), &identity);
    auto normal_buffer = gpu.buffer(normals);
    auto plane_buffer = gpu.buffer(plane_offsets);
    auto active_buffer = gpu.buffer(contact_active);
    auto advances = gpu.buffer(sizeof(std::uint32_t));
    auto status = gpu.buffer(sizeof(std::uint32_t));
    for (const char *name : {"soft_body_realtime_predict",
             "soft_body_realtime_project_links",
             "soft_body_realtime_apply_position_delta",
             "soft_body_realtime_project_shape",
             "soft_body_realtime_project_contacts",
             "soft_body_realtime_commit"})
        require(gpu.kernels.contains(name), std::string("missing ") + name);
    gpu.kernels.at("soft_body_realtime_project_links").check_pointee(
        "neighbors", sizeof(SoftNeighbor));

    const auto encode_tick = [&](Vec3 gravity, std::uint32_t selected_profile) {
        auto command = gpu.device->commands();
        const auto groups = (count + 127U) / 128U;
        {
            Arguments args(gpu.kernels.at("soft_body_realtime_predict"));
            args.buffer("positions", positions);
            args.buffer("velocities", velocity_buffer);
            args.buffer("previous_positions", previous);
            args.buffer("correction_used", correction_used);
            args.buffer("inverse_masses", mass_buffer);
            args.buffer("characteristic_lengths", characteristic_buffer);
            args.set("count", count); args.set("capacity", count);
            args.set("gravity", gravity); args.set("velocity_damping", 5.0f);
            args.set("maximum_speed", 2.9f); args.set("dt", dt);
            args.set("profile_version", selected_profile);
            args.buffer("time_advances", advances); args.buffer("status", status);
            gpu.encode(args, groups, command);
        }
        for (std::uint32_t sweep = 0; sweep < sweeps; ++sweep) {
            {
                // Shape is first so an inverted input is rejected before any
                // bond projection can conceal the invalid orientation.
                Arguments args(gpu.kernels.at("soft_body_realtime_project_shape"));
                args.buffer("positions", positions); args.buffer("corrections", corrections);
                args.buffer("rest_positions", rest_buffer); args.buffer("inverse_masses", mass_buffer);
                args.buffer("characteristic_lengths", characteristic_buffer);
                args.buffer("correction_used", correction_used);
                args.set("count", count); args.set("capacity", count);
                args.set("movable_mass", movable_mass); args.set("rest_center", rest_center);
                args.set("inverse_rest", inverse_rest); args.buffer("stored_orientation", orientation);
                args.set("stiffness", 0.35f); args.set("dt", dt);
                args.set("profile_version", selected_profile); args.buffer("status", status);
                gpu.encode(args, 1, command);
            }
            {
                Arguments args(gpu.kernels.at("soft_body_realtime_project_links"));
                args.buffer("positions", positions); args.buffer("position_deltas", corrections);
                args.buffer("correction_used", correction_used);
                args.buffer("inverse_masses", mass_buffer);
                args.buffer("characteristic_lengths", characteristic_buffer);
                args.buffer("offsets", offset_buffer); args.buffer("neighbors", neighbor_buffer);
                args.set("count", count); args.set("capacity", count);
                args.set("compliance", 0.0f); args.set("dt", dt);
                args.set("profile_version", selected_profile); args.buffer("status", status);
                gpu.encode(args, groups, command);
            }
            {
                Arguments args(gpu.kernels.at("soft_body_realtime_apply_position_delta"));
                args.buffer("positions", positions); args.buffer("position_deltas", corrections);
                args.buffer("correction_used", correction_used);
                args.buffer("inverse_masses", mass_buffer);
                args.buffer("characteristic_lengths", characteristic_buffer);
                args.set("count", count); args.set("capacity", count);
                args.set("dt", dt); args.set("profile_version", selected_profile);
                args.buffer("status", status); gpu.encode(args, groups, command);
            }
            if ((sweep & 1U) != 0U) {
                Arguments args(gpu.kernels.at("soft_body_realtime_project_contacts"));
                args.buffer("positions", positions); args.buffer("previous_positions", previous);
                args.buffer("inverse_masses", mass_buffer);
                args.buffer("characteristic_lengths", characteristic_buffer);
                args.buffer("correction_used", correction_used); args.buffer("contact_normals", normal_buffer);
                args.buffer("contact_offsets", plane_buffer); args.buffer("contact_active", active_buffer);
                args.set("count", count); args.set("capacity", count);
                args.set("node_radius", radius); args.set("friction", 0.4f);
                args.set("dt", dt); args.set("profile_version", selected_profile);
                args.buffer("status", status); gpu.encode(args, groups, command);
            }
        }
        {
            Arguments args(gpu.kernels.at("soft_body_realtime_commit"));
            args.buffer("positions", positions); args.buffer("previous_positions", previous);
            args.buffer("velocities", velocity_buffer); args.buffer("inverse_masses", mass_buffer);
            args.buffer("characteristic_lengths", characteristic_buffer);
            args.buffer("contact_normals", normal_buffer); args.buffer("contact_offsets", plane_buffer);
            args.buffer("contact_active", active_buffer); args.set("count", count);
            args.set("capacity", count); args.set("node_radius", radius);
            args.set("friction", 0.4f); args.set("maximum_speed", 2.9f);
            args.set("dt", dt); args.set("profile_version", selected_profile);
            args.buffer("status", status); gpu.encode(args, groups, command);
        }
        return gpu.complete(command) * 1000.0;
    };

    const auto edge_error = [&](const Vec3 *values) {
        double squared{};
        double maximum{};
        std::size_t edge{};
        for (std::uint32_t node = 0; node < count; ++node)
            for (std::uint32_t other = node + 1; other < count; ++other) {
                const float target = length(subtract(rest[node], rest[other]));
                const float strain = length(subtract(values[node], values[other])) / target - 1.0f;
                squared += strain * strain;
                maximum = std::max(maximum, std::abs(double(strain)));
                ++edge;
            }
        return std::pair{std::sqrt(squared / double(edge)), maximum};
    };
    const auto shape_error = [&](const Vec3 *values, Quaternion rotation) {
        Vec3 center{};
        for (std::uint32_t node = 0; node < count; ++node) {
            const float mass = 1.0f / inverse_masses[node];
            center.x += mass * values[node].x;
            center.y += mass * values[node].y;
            center.z += mass * values[node].z;
        }
        center.x /= movable_mass; center.y /= movable_mass;
        center.z /= movable_mass;
        const auto rotate = [](Quaternion q, Vec3 value) {
            const Vec3 qv{q.x, q.y, q.z};
            const Vec3 t{2.0f * (qv.y * value.z - qv.z * value.y),
                2.0f * (qv.z * value.x - qv.x * value.z),
                2.0f * (qv.x * value.y - qv.y * value.x)};
            return Vec3{value.x + q.w * t.x + qv.y * t.z - qv.z * t.y,
                value.y + q.w * t.y + qv.z * t.x - qv.x * t.z,
                value.z + q.w * t.z + qv.x * t.y - qv.y * t.x};
        };
        double squared{};
        for (std::uint32_t node = 0; node < count; ++node) {
            const Vec3 relative = subtract(rest[node], rest_center);
            const Vec3 rotated = rotate(rotation, relative);
            const Vec3 difference = subtract(values[node],
                Vec3{center.x + rotated.x, center.y + rotated.y,
                    center.z + rotated.z});
            squared += dot(difference, difference);
        }
        return std::sqrt(squared / count) / 0.1;
    };
    const float target_volume = tetra_volume(rest.data());
    const float initial_volume_error = std::abs(tetra_volume(initial.data()) / target_volume - 1.0f);
    const auto initial_edges = edge_error(initial.data());
    encode_tick(Vec3{}, profile);
    require(*static_cast<std::uint32_t *>(advances->host_address()) == 1,
        "soft-body realtime did not advance exactly once");
    require(*static_cast<std::uint32_t *>(status->host_address()) == 0,
        "soft-body realtime quality tick rejected valid state");
    const auto *quality = static_cast<const Vec3 *>(positions->host_address());
    const float final_volume_error = std::abs(tetra_volume(quality) / target_volume - 1.0f);
    const auto final_edges = edge_error(quality);
    const double final_shape_error = shape_error(quality,
        *static_cast<const Quaternion *>(orientation->host_address()));
    require(final_volume_error < initial_volume_error &&
        final_edges.first < initial_edges.first,
        "soft-body shape/volume error did not improve");
    require(final_edges.first <= 0.04 && final_edges.second <= 0.12 &&
        final_volume_error <= 0.15 && final_shape_error <= 0.1,
        "soft-body one-step quality exceeds frozen profile gates");

    // Cold-start under gravity, impact and resting energy over 3,600 fixed ticks.
    std::memcpy(positions->host_address(), rest.data(), count * sizeof(Vec3));
    std::memset(velocity_buffer->host_address(), 0, count * sizeof(Vec3));
    std::memset(advances->host_address(), 0, sizeof(std::uint32_t));
    std::memset(status->host_address(), 0, sizeof(std::uint32_t));
    std::memcpy(orientation->host_address(), &identity, sizeof(identity));
    std::vector<double> timings;
    std::uint32_t tunneling_events{};
    for (std::uint32_t tick = 0; tick < 3600; ++tick) {
        const double elapsed = encode_tick({0.0f, -9.81f, 0.0f}, profile);
        if (tick >= 3000) timings.push_back(elapsed);
        const auto *tick_positions =
            static_cast<const Vec3 *>(positions->host_address());
        for (std::uint32_t node = 0; node < count; ++node) {
            require(std::isfinite(tick_positions[node].x) &&
                std::isfinite(tick_positions[node].y) &&
                std::isfinite(tick_positions[node].z),
                "soft-body long run produced non-finite state");
            if (tick_positions[node].y < radius - 1.0e-5f)
                ++tunneling_events;
        }
    }
    require(*static_cast<std::uint32_t *>(advances->host_address()) == 3600,
        "soft-body long run did not preserve one advance per tick");
    require(*static_cast<std::uint32_t *>(status->host_address()) == 0,
        "soft-body long run reported profile/material/contact failure");
    const auto *settled = static_cast<const Vec3 *>(positions->host_address());
    const auto *settled_velocity = static_cast<const Vec3 *>(velocity_buffer->host_address());
    double energy{};
    float minimum_height = std::numeric_limits<float>::max();
    for (std::uint32_t node = 0; node < count; ++node) {
        minimum_height = std::min(minimum_height, settled[node].y);
        energy += 0.5 * (1.0 / inverse_masses[node]) *
            dot(settled_velocity[node], settled_velocity[node]);
    }
    require(minimum_height >= radius - 1.0e-5f,
        "soft-body long run tunneled through contact plane");
    require(tunneling_events == 0,
        "soft-body long run had an intermediate tunneling event");
    require(std::isfinite(energy) && energy < 1.0e-4,
        "soft-body long run failed to settle");
    const auto settled_edges = edge_error(settled);
    const float settled_volume_error = std::abs(
        tetra_volume(settled) / target_volume - 1.0f);
    const double settled_shape_error = shape_error(settled,
        *static_cast<const Quaternion *>(orientation->host_address()));
    require(settled_edges.first <= 0.04 && settled_edges.second <= 0.12 &&
        settled_volume_error <= 0.15 && settled_shape_error <= 0.1,
        "soft-body settled state exceeds frozen profile gates");

    // An inverted rest mapping is rejected, never silently projected through.
    std::memcpy(positions->host_address(), rest.data(), count * sizeof(Vec3));
    static_cast<Vec3 *>(positions->host_address())[3].z = -0.1f;
    std::memset(correction_used->host_address(), 0, count * sizeof(float));
    std::memset(status->host_address(), 0, sizeof(std::uint32_t));
    {
        Arguments args(gpu.kernels.at("soft_body_realtime_project_shape"));
        args.buffer("positions", positions); args.buffer("corrections", corrections);
        args.buffer("rest_positions", rest_buffer); args.buffer("inverse_masses", mass_buffer);
        args.buffer("characteristic_lengths", characteristic_buffer);
        args.buffer("correction_used", correction_used); args.set("count", count);
        args.set("capacity", count); args.set("movable_mass", movable_mass);
        args.set("rest_center", rest_center); args.set("inverse_rest", inverse_rest);
        args.buffer("stored_orientation", orientation); args.set("stiffness", 0.35f);
        args.set("dt", dt); args.set("profile_version", profile);
        args.buffer("status", status); gpu.run(args, 1);
    }
    require((*static_cast<std::uint32_t *>(status->host_address()) & (1U << 22U)) != 0,
        "soft-body inversion policy did not reject inverted mapping");

    // Explicit negotiation rejects another profile without advancing time.
    std::memset(advances->host_address(), 0, sizeof(std::uint32_t));
    std::memset(status->host_address(), 0, sizeof(std::uint32_t));
    encode_tick(Vec3{}, profile + 1);
    require(*static_cast<std::uint32_t *>(advances->host_address()) == 0 &&
        (*static_cast<std::uint32_t *>(status->host_address()) & (1U << 17U)) != 0,
        "soft-body realtime accepted an unnegotiated profile");

    std::sort(timings.begin(), timings.end());
    const auto percentile = [&](double value) {
        return timings[std::min<std::size_t>(timings.size() - 1,
            static_cast<std::size_t>(value * timings.size()))];
    };
    std::cout << "F8_SOFT_BODY_METAL_QUALITY initial_volume_error="
              << initial_volume_error << " final_volume_error="
              << final_volume_error << " initial_edge_rms="
              << initial_edges.first << " final_edge_rms=" << final_edges.first
              << " final_edge_max=" << final_edges.second
              << " final_shape_rms_h=" << final_shape_error
              << " resting_energy_j=" << energy << " minimum_height_m="
              << minimum_height << " tunneling_events=" << tunneling_events
              << " settled_volume_error=" << settled_volume_error
              << " settled_edge_rms=" << settled_edges.first
              << " settled_edge_max=" << settled_edges.second
              << " settled_shape_rms_h=" << settled_shape_error
              << " p50_ms=" << percentile(0.50)
              << " p95_ms=" << percentile(0.95) << " p99_ms="
              << percentile(0.99) << " maximum_ms=" << timings.back() << '\n';
}

void check_avbd_schedule(Gpu &gpu) {
    // Edges deliberately connect bodies in different 32-lane workgroups.
    // The 545-body case also exercises prepare's device-memory cache tail.
    for (const std::uint32_t count : {97U, 545U}) {
        for (const bool impact : {false, true}) {
            constexpr float dt = 1.0f / 60.0f;
            constexpr std::uint32_t iterations = 3;
            std::vector<BodyParameters> parameters(count);
            std::vector<RigidBodyState> previous(count), predicted(count);
            std::vector<RigidBodyId> ids(count);
            for (std::uint32_t i = 0; i < count; ++i) {
                auto &body = parameters[i];
                body.motion = i == 0 ? MotionType::static_body : MotionType::dynamic;
                body.inverse_mass = i == 0 ? 0.0f : 1.0f / (1 + i % 4);
                body.inverse_inertia_local = {1, 1, 1};
                body.friction = 0.4f;
                body.restitution = 0.25f;
                body.collision_margin = 0.001f;
                previous[i].position = {float(i % 7), float(i) * 0.02f, 0};
                previous[i].linear_velocity = {0.03f * float(i % 3),
                    impact && i >= 33 ? -3.0f : 0.0f, 0};
                predicted[i] = previous[i];
                predicted[i].position.x += previous[i].linear_velocity.x * dt;
                predicted[i].position.y += previous[i].linear_velocity.y * dt;
                ids[i] = {i, 1};
            }
            std::vector<std::uint32_t> pairs;
            std::vector<ContactManifold> manifolds;
            for (std::uint32_t i = 0; i + 33 < count; ++i) {
                pairs.push_back(i * count + i + 33);
                ContactManifold manifold;
                manifold.count = 1;
                auto &contact = manifold.contacts[0];
                contact.normal = {0, -1, 0};
                contact.point = previous[i].position;
                contact.penetration = impact ? 0.01f : 0.0f;
                manifolds.push_back(manifold);
            }
            std::vector<RigidConstraintResource> joints(11);
            for (std::uint32_t i = 0; i < joints.size(); ++i) {
                auto &joint = joints[i];
                joint.alive = true;
                joint.options.type = static_cast<RigidConstraintType>(i % 8);
                joint.options.body_a = ids[i + 1];
                joint.options.body_b = ids[i + 65];
                joint.options.local_anchor_b = {
                    previous[i + 1].position.x - previous[i + 65].position.x,
                    previous[i + 1].position.y - previous[i + 65].position.y, 0};
                joint.options.motor.angular_enabled = i == 7;
                joint.options.motor.angular_target_velocity = 0.2f;
                joint.options.linear_limits.axes = rigid_constraint_all_axes;
                joint.options.angular_limits.axes = rigid_constraint_all_axes;
                joint.options.linear_springs.axes = rigid_constraint_axis_x;
                joint.options.linear_springs.stiffness.x = 10;
                joint.options.linear_springs.damping.x = 0.1f;
                joint.options.enabled = i != 8;
                joint.state.broken = i == 9;
                joint.options.solver_iterations = iterations;
                if (i == 10) {
                    joint.options.breaking_impulse_threshold = 1.0e-6f;
                    joint.options.local_anchor_b.x += 0.25f;
                }
            }
            const auto parameter_buffer = gpu.buffer(parameters);
            const auto previous_buffer = gpu.buffer(previous);
            const auto id_buffer = gpu.buffer(ids);
            const auto pair_buffer = gpu.buffer(pairs);
            const auto pair_count = static_cast<std::uint32_t>(pairs.size());
            const auto pair_count_buffer = gpu.buffer(sizeof(pair_count), &pair_count);
            struct Result {
                BufferPtr states, manifolds, joints, bodies, schedule;
            } results[2];
            for (std::uint32_t staged = 0; staged < 2; ++staged) {
                auto &result = results[staged];
                result.states = gpu.buffer(predicted);
                result.manifolds = gpu.buffer(manifolds);
                result.joints = gpu.buffer(joints);
                result.bodies = gpu.buffer(count * sizeof(GeometryConstraintBody));
                result.schedule = gpu.buffer(sizeof(ContactSchedule));
                const auto bind = [&](Arguments &arguments) {
                    arguments.buffer("parameters", parameter_buffer);
                    arguments.buffer("previous", previous_buffer);
                    arguments.buffer("states", result.states);
                    arguments.set("count", count);
                    arguments.buffer("manifolds", result.manifolds);
                    arguments.buffer("pairs", pair_buffer);
                    arguments.buffer("pair_count", pair_count_buffer);
                    arguments.buffer("joints", result.joints);
                    arguments.set("joint_capacity", static_cast<std::uint32_t>(joints.size()));
                    arguments.buffer("bodies", result.bodies);
                    arguments.buffer("schedule", result.schedule);
                    arguments.set("dt", dt);
                };
                const auto groups = staged ? (count + 31) / 32 : 1U;
                auto command = gpu.device->commands();
                Arguments prepare(gpu.kernels.at("prepare_avbd_kernel"));
                prepare.kernel.check_pointee("bodies", sizeof(GeometryConstraintBody));
                prepare.kernel.check_pointee("joints", sizeof(RigidConstraintResource));
                prepare.kernel.check_pointee("manifolds", sizeof(ContactManifold));
                bind(prepare);
                prepare.buffer("ids", id_buffer);
                prepare.set("gravity", Vec3{});
                prepare.set("base_iterations", iterations);
                prepare.set("requested_iterations", iterations);
                prepare.set("profile_version", 0U);
                prepare.set("grid_blocks", groups);
                gpu.encode(prepare, 1, command);
                Arguments solve(gpu.kernels.at(staged
                    ? "solve_avbd_stage_kernel" : "solve_avbd_kernel"));
                bind(solve);
                if (staged) {
                    // Deliberately over-dispatch bounds to check GPU guards.
                    // No schedule/color/impact readback between prepare/solve.
                    encode_avbd_schedule(iterations + 1, 4,
                        [&](AvbdStage operation, auto phase, auto iteration, auto color) {
                            solve.set("operation", static_cast<std::uint32_t>(operation));
                            solve.set("phase", phase);
                            solve.set("iteration", iteration);
                            solve.set("color", color);
                            gpu.encode(solve, groups, command);
                        });
                } else {
                    gpu.encode(solve, 1, command);
                }
                gpu.complete(command);
            }
            const auto &schedule = *static_cast<const ContactSchedule *>(results[1].schedule->host_address());
            require(schedule.remaining <= 4 && schedule.remaining > 1,
                "AVBD fixture color bound invalid");
            require((schedule.counts[0] != 0) == impact,
                "AVBD fixture failed to exercise selected impact phase: expected " +
                std::to_string(impact) + " actual " + std::to_string(schedule.counts[0]));
            require(schedule.maximum_passes == iterations * (impact ? 2 : 1),
                "AVBD phase budget mismatch");
            const auto *expected = static_cast<const RigidBodyState *>(results[0].states->host_address());
            const auto *actual = static_cast<const RigidBodyState *>(results[1].states->host_address());
            for (std::uint32_t i = 0; i < count; ++i) {
                const auto a = std::bit_cast<std::array<float, 13>>(actual[i]);
                const auto b = std::bit_cast<std::array<float, 13>>(expected[i]);
                static_assert(sizeof(RigidBodyState) == 13 * sizeof(float));
                for (std::size_t field = 0; field < 13; ++field)
                    require(std::isfinite(a[field]) && std::abs(a[field] - b[field]) < 2.0e-4f,
                        "AVBD staged/cooperative mismatch: body " + std::to_string(i) +
                        " field " + std::to_string(field) + " values " +
                        std::to_string(a[field]) + " / " + std::to_string(b[field]));
            }
            const auto *actual_joints = static_cast<const RigidConstraintResource *>(results[1].joints->host_address());
            const auto *expected_joints = static_cast<const RigidConstraintResource *>(results[0].joints->host_address());
            for (std::size_t i = 0; i < 8; ++i)
                require(actual_joints[i].avbd_count > 0, "AVBD fixture joint has no active rows");
            require(actual_joints[10].state.broken && !actual_joints[10].state.enabled,
                "AVBD fixture failed to exercise joint breaking");
            for (std::size_t i = 0; i < joints.size(); ++i)
                require(actual_joints[i].state.enabled == expected_joints[i].state.enabled &&
                    actual_joints[i].state.broken == expected_joints[i].state.broken &&
                    std::abs(actual_joints[i].state.applied_impulse -
                        expected_joints[i].state.applied_impulse) < 2.0e-4f,
                    "AVBD staged joint output mismatch");
            const auto *actual_contacts = static_cast<const ContactManifold *>(results[1].manifolds->host_address());
            const auto *expected_contacts = static_cast<const ContactManifold *>(results[0].manifolds->host_address());
            for (std::size_t i = 0; i < manifolds.size(); ++i) {
                const auto &a = actual_contacts[i].contacts[0];
                const auto &b = expected_contacts[i].contacts[0];
                require(std::isfinite(a.reported_normal_impulse) &&
                    std::abs(a.reported_normal_impulse - b.reported_normal_impulse) < 2.0e-4f &&
                    std::abs(a.reported_friction_impulse.x - b.reported_friction_impulse.x) < 2.0e-4f &&
                    std::abs(a.reported_friction_impulse.y - b.reported_friction_impulse.y) < 2.0e-4f &&
                    std::abs(a.reported_friction_impulse.z - b.reported_friction_impulse.z) < 2.0e-4f,
                    "AVBD staged contact output mismatch");
            }
        }
    }
    std::cout << "AVBD multi-dispatch: 97/545 bodies, contacts, all joint types, impact phases passed\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--execution-only"),
            "usage: metal-kernel-tests <manifest> [--execution-only]");
        auto device = compute::create_metal_device();
        if (!device) {
            std::cout << "SKIP: no supported native Metal device; no CPU fallback\n";
            return 77;
        }
        std::cout << "GPU: " << device->name() << '\n';
        Gpu gpu(std::move(device));
        gpu.load(argv[1], argc == 3);
        check_device_operations(gpu);
        check_transfers_and_records(gpu);
        check_device_teardown();
        check_rigid_integration(gpu);
        check_soft_body_simd(gpu);
        check_soft_body_realtime(gpu);
        check_avbd_schedule(gpu);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
