// SPDX-License-Identifier: MIT
// Native GPU qualification of generated Slang kernels. This is deliberately
// not a driver and never falls back to CPU execution.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <simulation/geometry_storage.hpp>

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

namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
NSString *native(const std::string &text) {
    return [NSString stringWithUTF8String:text.c_str()];
}
std::string description(NSError *error) {
    return error ? error.localizedDescription.UTF8String : "no native error detail";
}

void collect_structs(id value, NSMutableDictionary *types) {
    if ([value isKindOfClass:NSDictionary.class]) {
        NSDictionary *object = value;
        if ([object[@"kind"] isEqual:@"struct"] && object[@"name"] && object[@"fields"])
            types[object[@"name"]] = object;
        for (id child in object.allValues) collect_structs(child, types);
    } else if ([value isKindOfClass:NSArray.class]) {
        for (id child in value) collect_structs(child, types);
    }
}

struct Kernel {
    id<MTLComputePipelineState> pipeline;
    NSDictionary *entry;
    NSDictionary *types;
    std::size_t argument_size{};
    NSUInteger argument_index{};
    std::uint32_t threads{};

    NSDictionary *parameter(const char *name) const {
        for (NSDictionary *parameter in entry[@"parameters"])
            if ([parameter[@"name"] isEqualToString:@(name)]) return parameter;
        throw std::runtime_error(std::string("missing reflected parameter: ") + name);
    }
    void check_pointee(const char *name, std::size_t size) const {
        id value_type = parameter(name)[@"type"][@"valueType"];
        NSDictionary *type = [value_type isKindOfClass:NSString.class] ? types[value_type] : value_type;
        require([type isKindOfClass:NSDictionary.class], std::string("unresolved pointee: ") + name);
        bool matched = false;
        for (NSDictionary *layout in type[@"sizes"])
            if ([layout[@"kind"] isEqualToString:@"uniform"])
                matched = [layout[@"value"] unsignedLongLongValue] == size;
        require(matched, std::string("host/Slang pointee size mismatch: ") + name);
    }
};

struct Arguments {
    const Kernel &kernel;
    std::vector<std::byte> bytes;
    std::vector<id<MTLBuffer>> resources;
    explicit Arguments(const Kernel &value) : kernel(value), bytes(value.argument_size) {}

    template<class T> void set(const char *name, const T &value) {
        NSDictionary *binding = kernel.parameter(name)[@"binding"];
        const std::size_t offset = [binding[@"offset"] unsignedLongLongValue];
        require([binding[@"size"] unsignedLongLongValue] == sizeof(T),
            std::string("argument size mismatch: ") + name);
        require(offset + sizeof(T) <= bytes.size(), "argument outside constant buffer");
        std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }
    void buffer(const char *name, id<MTLBuffer> buffer) {
        require(buffer != nil, "buffer allocation failed");
        // CPU contents pointers are NEVER encoded as GPU device addresses.
        set(name, static_cast<std::uint64_t>(buffer.gpuAddress));
        resources.push_back(buffer);
    }
};

class Gpu {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::map<std::string, Kernel> kernels;

    explicit Gpu(id<MTLDevice> value) : device(value), queue([value newCommandQueue]) {
        require(queue != nil, "Metal command queue creation failed");
    }
    id<MTLBuffer> buffer(std::size_t size, const void *data = nullptr) const {
        id<MTLBuffer> result = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        require(result != nil, "Metal shared buffer allocation failed");
        if (data) std::memcpy(result.contents, data, size);
        else std::memset(result.contents, 0, size);
        return result;
    }
    template<class T> id<MTLBuffer> buffer(const std::vector<T> &data) const {
        return buffer(data.size() * sizeof(T), data.data());
    }
    void load(const std::string &manifest, bool execution_only) {
        std::ifstream input(manifest);
        require(input.good(), "cannot open Metal kernel manifest");
        std::string line;
        while (std::getline(input, line)) {
            @autoreleasepool {
                const auto first = line.find('|');
                const auto second = line.find('|', first + 1);
                require(first != std::string::npos && second != std::string::npos,
                    "invalid Metal kernel manifest record");
                const auto name = line.substr(first + 1, second - first - 1);
                const auto base = line.substr(second + 1);
                if (execution_only && name != "metal_device_contract" &&
                    name != "pm_integrate_rigid_bodies" &&
                    name != "deformable_project_links" &&
                    name != "deformable_project_links_warp" &&
                    name != "soft_body_damp_springs" &&
                    name != "soft_body_damp_springs_warp") continue;
                NSError *error = nil;
                id<MTLLibrary> library = [device newLibraryWithURL:
                    [NSURL fileURLWithPath:native(base + ".metallib")] error:&error];
                require(library != nil, name + ": " + description(error));
                id<MTLFunction> function = [library newFunctionWithName:native(name)];
                require(function != nil, name + ": entry missing from native library");
                MTLComputePipelineReflection *reflection = nil;
                MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
                descriptor.label = native(name);
                descriptor.computeFunction = function;
                id<MTLComputePipelineState> pipeline = [device
                    newComputePipelineStateWithDescriptor:descriptor
                    options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
                    reflection:&reflection error:&error];
                require(pipeline != nil, name + ": " + description(error));
                NSData *data = [NSData dataWithContentsOfFile:native(base + ".json")];
                require(data != nil, name + ": reflection file missing");
                NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
                require(json != nil, name + ": invalid reflection JSON");
                NSArray *entries = json[@"entryPoints"];
                require(entries.count == 1, name + ": expected single-entry Metal compilation");
                Kernel kernel;
                kernel.pipeline = pipeline;
                kernel.entry = entries[0];
                NSMutableDictionary *types = [NSMutableDictionary dictionary];
                collect_structs(json, types);
                kernel.types = types;
                require([kernel.entry[@"name"] isEqualToString:native(name)],
                    name + ": wrong reflected entry");
                kernel.threads = [kernel.entry[@"threadGroupSize"][0] unsignedIntValue];
                require(kernel.threads <= pipeline.maxTotalThreadsPerThreadgroup,
                    name + ": workgroup exceeds native pipeline limit");
                require(pipeline.staticThreadgroupMemoryLength <= device.maxThreadgroupMemoryLength,
                    name + ": threadgroup storage " + std::to_string(pipeline.staticThreadgroupMemoryLength) +
                    " exceeds device limit " + std::to_string(device.maxThreadgroupMemoryLength));
                bool found_arguments = false;
                for (id<MTLBinding> binding in reflection.bindings) {
                    if (binding.type != MTLBindingTypeBuffer ||
                        ![binding.name hasPrefix:@"entryPointParams"]) continue;
                    id<MTLBufferBinding> argument = (id<MTLBufferBinding>)binding;
                    found_arguments = true;
                    kernel.argument_index = argument.index;
                    kernel.argument_size = argument.bufferDataSize;
                    for (NSDictionary *field in kernel.entry[@"parameters"]) {
                        NSString *prefix = [field[@"name"] stringByAppendingString:@"_"];
                        bool found_field = false;
                        for (MTLStructMember *member in argument.bufferStructType.members) {
                            if (![member.name hasPrefix:prefix]) continue;
                            NSString *suffix = [member.name substringFromIndex:prefix.length];
                            if (suffix.length == 0 || [suffix rangeOfCharacterFromSet:
                                NSCharacterSet.decimalDigitCharacterSet.invertedSet].location != NSNotFound)
                                continue;
                            require(member.offset == [field[@"binding"][@"offset"] unsignedLongLongValue],
                                name + ": native/Slang argument offset mismatch");
                            found_field = true;
                            break;
                        }
                        require(found_field, name + ": native argument field missing");
                    }
                }
                require(found_arguments && kernel.argument_size > 0,
                    name + ": native argument buffer missing");
                require(kernels.emplace(name, kernel).second, name + ": duplicate kernel");
            }
        }
        require(!kernels.empty(), "empty Metal kernel manifest");
        std::cout << "Native pipelines and argument layouts: " << kernels.size() << " passed\n";
    }
    void run(Arguments &arguments, std::uint32_t groups) const {
        const auto &kernel = arguments.kernel;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        require(command != nil && encoder != nil, "Metal command encoding failed");
        [encoder setComputePipelineState:kernel.pipeline];
        const std::uint32_t dispatch[4]{groups, 1, 1, kernel.threads};
        [encoder setBytes:dispatch length:sizeof(dispatch) atIndex:0];
        [encoder setBytes:arguments.bytes.data() length:arguments.bytes.size()
            atIndex:kernel.argument_index];
        for (id<MTLBuffer> resource : arguments.resources)
            [encoder useResource:resource usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kernel.threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        // Explicit waits belong to the test, not the eventual asynchronous driver.
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
            "GPU execution failed: " + description(command.error));
        require(command.GPUEndTime >= command.GPUStartTime && command.GPUEndTime > 0,
            "missing native GPU execution timestamps");
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
        require(*static_cast<std::uint64_t *>(total.contents) == initial + addend * count,
            "64-bit reduction carry/wrap mismatch");
        require(*static_cast<std::uint64_t *>(mask.contents) == UINT64_MAX,
            "64-bit mask reduction mismatch");
        const auto vector = *static_cast<Vec3 *>(sum.contents);
        require(vector.x == count && vector.y == 2 * count && vector.z == -float(count),
            "floating-point vector atomic mismatch");
        require(*static_cast<float *>(maximum.contents) == count &&
            *static_cast<float *>(minimum.contents) == 1.0f, "float min/max atomic mismatch");
        const auto *values = static_cast<std::uint32_t *>(indices.contents);
        for (std::uint32_t index = 0; index < count; ++index)
            require(values[index] == index + groups * kernel.threads, "dispatch index mismatch");
    }
    std::cout << "GPU dispatch and atomic carry/wrap/contention tests passed\n";
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
    const auto *result = static_cast<RigidBodyState *>(output.contents);
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
        require(simd.pipeline.threadExecutionWidth == 32, "unsupported Metal SIMD width");
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
        const auto *expected = static_cast<Vec3 *>(scalar_output.contents);
        const auto *actual = static_cast<Vec3 *>(simd_output.contents);
        for (std::uint32_t node = 0; node < count; ++node)
            require(std::abs(expected[node].x - actual[node].x) < 1.0e-5f &&
                std::abs(expected[node].y - actual[node].y) < 1.0e-5f &&
                std::abs(expected[node].z - actual[node].z) < 1.0e-5f,
                "SIMD/scalar GPU soft-body mismatch at node " + std::to_string(node));
    }
    std::cout << "Soft-body SIMD tails, pinned nodes, variable degrees: projection/damping passed\n";
}
} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--execution-only"),
                "usage: metal-kernel-tests <manifest> [--execution-only]");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) {
                std::cout << "SKIP: no native Metal device; no CPU fallback\n";
                return 77;
            }
            if (![device supportsFamily:MTLGPUFamilyApple7]) {
                std::cout << "SKIP: native pointer tests require Apple GPU family 7 or newer\n";
                return 77;
            }
            std::cout << "GPU: " << device.name.UTF8String << '\n';
            Gpu gpu(device);
            gpu.load(argv[1], argc == 3);
            check_device_operations(gpu);
            check_rigid_integration(gpu);
            check_soft_body_simd(gpu);
            return 0;
        } catch (const std::exception &error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
