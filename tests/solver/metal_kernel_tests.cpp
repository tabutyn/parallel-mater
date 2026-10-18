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
    void complete(const std::unique_ptr<compute::CommandList> &command) const {
        auto completion = command->submit();
        completion->wait();
        require(completion->ready() && completion->elapsed_seconds() >= 0,
            "invalid GPU completion/timestamps");
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
        check_rigid_integration(gpu);
        check_soft_body_simd(gpu);
        check_avbd_schedule(gpu);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
