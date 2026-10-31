// SPDX-License-Identifier: MIT
#include "scene_builder.hpp"

#include <parallel_mater_scene/preparation.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace parallel_mater::gallery {
namespace {

constexpr PmFlags k_buffer_usage = PM_BUFFER_TRANSFER_SRC_BIT |
    PM_BUFFER_TRANSFER_DST_BIT | PM_BUFFER_GEOMETRY_INPUT_BIT |
    PM_BUFFER_PARTICLE_INPUT_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT;

std::atomic_uint64_t next_output_resource_generation{1U};

[[nodiscard]] const char *result_name(PmResult result) noexcept {
    switch (result) {
    case PM_SUCCESS: return "success";
    case PM_NOT_READY: return "not ready";
    case PM_TIMEOUT: return "timeout";
    case PM_INCOMPLETE: return "incomplete";
    case PM_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case PM_ERROR_INVALID_STATE: return "invalid state";
    case PM_ERROR_OUT_OF_HOST_MEMORY: return "out of host memory";
    case PM_ERROR_OUT_OF_DEVICE_MEMORY: return "out of device memory";
    case PM_ERROR_FEATURE_NOT_PRESENT: return "feature not present";
    case PM_ERROR_FORMAT_NOT_SUPPORTED: return "format not supported";
    case PM_ERROR_CAPACITY_EXCEEDED: return "capacity exceeded";
    case PM_ERROR_DEVICE_LOST: return "device lost";
    case PM_ERROR_INCOMPATIBLE_VERSION: return "incompatible version";
    case PM_ERROR_DRIVER_NOT_FOUND: return "driver not found";
    case PM_ERROR_LAYER_NOT_PRESENT: return "layer not present";
    case PM_ERROR_EXTENSION_NOT_PRESENT: return "extension not present";
    }
    return "unknown error";
}

[[nodiscard]] bool checked(PmResult result, std::string_view operation,
                           std::string &error) {
    if (result == PM_SUCCESS) return true;
    error.assign(operation);
    error.append(": ");
    error.append(result_name(result));
    return false;
}

[[nodiscard]] PmFloat3 pm(Vec3 value) noexcept {
    return {value.x, value.y, value.z};
}

void copy(float destination[3], Vec3 value) noexcept {
    destination[0] = value.x;
    destination[1] = value.y;
    destination[2] = value.z;
}

void copy(float destination[4], Quaternion value) noexcept {
    destination[0] = value.x;
    destination[1] = value.y;
    destination[2] = value.z;
    destination[3] = value.w;
}

[[nodiscard]] PmRigidState pm(RigidBodyState value) noexcept {
    PmRigidState result{};
    copy(result.position, value.position);
    copy(result.orientation, value.orientation);
    copy(result.linearVelocity, value.linear_velocity);
    copy(result.angularVelocity, value.angular_velocity);
    return result;
}

[[nodiscard]] PmMotionType pm(MotionType value) noexcept {
    switch (value) {
    case MotionType::static_body: return PM_MOTION_STATIC;
    case MotionType::kinematic: return PM_MOTION_KINEMATIC;
    case MotionType::dynamic: return PM_MOTION_DYNAMIC;
    }
    return PM_MOTION_DYNAMIC;
}

[[nodiscard]] PmConstraintType pm(RigidConstraintType value) noexcept {
    switch (value) {
    case RigidConstraintType::fixed: return PM_CONSTRAINT_FIXED;
    case RigidConstraintType::point: return PM_CONSTRAINT_POINT;
    case RigidConstraintType::hinge: return PM_CONSTRAINT_HINGE;
    case RigidConstraintType::slider: return PM_CONSTRAINT_SLIDER;
    case RigidConstraintType::piston: return PM_CONSTRAINT_PISTON;
    case RigidConstraintType::generic: return PM_CONSTRAINT_GENERIC;
    case RigidConstraintType::generic_spring: return PM_CONSTRAINT_GENERIC_SPRING;
    case RigidConstraintType::motor: return PM_CONSTRAINT_MOTOR;
    }
    return PM_CONSTRAINT_FIXED;
}

void copy_axes(PmAxisLimit destination[3],
               const RigidConstraintLimitOptions &source) noexcept {
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        destination[axis].enabled =
            (source.axes & (1U << axis)) != 0U ? PM_TRUE : PM_FALSE;
        destination[axis].lower = (&source.lower.x)[axis];
        destination[axis].upper = (&source.upper.x)[axis];
    }
}

void copy_axes(PmAxisSpring destination[3],
               const RigidConstraintSpringOptions &source) noexcept {
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        destination[axis].enabled =
            (source.axes & (1U << axis)) != 0U ? PM_TRUE : PM_FALSE;
        destination[axis].stiffness = (&source.stiffness.x)[axis];
        destination[axis].damping = (&source.damping.x)[axis];
    }
}

[[nodiscard]] PmRigidConstraintInfo pm(
    const RigidConstraintDefinition &definition) noexcept {
    const auto &source = definition.options;
    PmRigidConstraintInfo result{};
    result.active = source.enabled ? PM_TRUE : PM_FALSE;
    result.type = pm(source.type);
    result.bodyA = definition.body_a;
    result.bodyB = definition.body_b;
    copy(result.frameA.position, source.local_anchor_a);
    copy(result.frameA.orientation, source.local_orientation_a);
    copy(result.frameB.position, source.local_anchor_b);
    copy(result.frameB.orientation, source.local_orientation_b);
    copy_axes(result.linearLimits, source.linear_limits);
    copy_axes(result.angularLimits, source.angular_limits);
    copy_axes(result.linearSprings, source.linear_springs);
    copy_axes(result.angularSprings, source.angular_springs);
    result.linearMotors[0] = {
        source.motor.linear_enabled ? PM_TRUE : PM_FALSE,
        source.motor.linear_target_velocity,
        source.motor.linear_maximum_impulse};
    result.angularMotors[0] = {
        source.motor.angular_enabled ? PM_TRUE : PM_FALSE,
        source.motor.angular_target_velocity,
        source.motor.angular_maximum_impulse};
    result.collideConnected = source.disable_collisions ? PM_FALSE : PM_TRUE;
    result.breakingImpulseThreshold = source.breaking_impulse_threshold;
    result.solverIterations = source.solver_iterations;
    return result;
}

[[nodiscard]] std::string mesh_key(bool collision,
                                   const std::vector<std::uint32_t> &indices) {
    std::string result = collision ? "collision" : "render";
    for (std::uint32_t index : indices) {
        result.push_back(':');
        result.append(std::to_string(index));
    }
    return result;
}

[[nodiscard]] std::uint64_t grid_cells(const SmokeOptions &options) noexcept {
    if (options.grid_resolution == 0U) return 1U;
    return static_cast<std::uint64_t>(options.grid_resolution) *
        options.grid_resolution * options.grid_vertical_resolution;
}

} // namespace

struct CommandScene::Impl {
    struct Buffer {
        PmBuffer handle{};
        PmDeviceMemory memory{};
        void *mapped{};
        PmDeviceSize size{};
    };

    struct Geometry {
        Buffer input;
        PmGeometry handle{};
        PmDeviceMemory memory{};
    };

    struct Region {
        PmDeviceSize offset{};
        PmDeviceSize size{};
    };

    PmInstance instance{};
    PmPhysicalDevice physical{};
    PmDevice device{};
    PmQueue queue{};
    PmCommandPool command_pool{};
    PmCommandBuffer commands{};
    PmCommandBuffer reset_commands{};
    PmFence fence{};
    PmSimulation simulation{};
    PmDeviceMemory persistent_memory{};
    PmDeviceMemory scratch_memory{};
    PmPhysicsPipeline pipeline{};
    PmQueryPool statistics_query{};
    std::vector<Geometry> geometries;
    std::unordered_map<std::string, std::uint32_t> geometry_cache;
    Buffer particle_input;
    Buffer output;
    Region rigid_output;
    Region previous_rigid_output;
    Region fluid_output;
    std::vector<Region> cloth_outputs;
    std::vector<Region> cloth_surface_outputs;
    std::vector<Region> rope_outputs;
    std::vector<Region> soft_body_outputs;
    std::vector<Region> soft_body_surface_outputs;
    Region smoke_output;
    std::vector<Region> paint_outputs;
    Region constraint_output;
    Region contact_output;
    Region debug_output;
    std::vector<PmRigidConstraintInfo> constraints;
    std::vector<PmRigidConstraintInfo> initial_constraints;
    std::vector<PmRigidBodyInfo> rigid_bodies;
    std::vector<PmRigidBodyInfo> initial_rigid_bodies;
    std::vector<std::uint8_t> cloth_active;
    std::vector<std::uint8_t> rope_active;
    std::vector<std::uint8_t> soft_body_active;
    std::vector<PmFluidCapacity> fluid_capacities;
    std::vector<PmClothCapacity> cloth_capacities;
    std::vector<PmRopeCapacity> rope_capacities;
    std::vector<PmSoftBodyCapacity> soft_body_capacities;
    std::vector<PmSmokeCapacity> smoke_capacities;
    std::vector<PmPaintFieldCapacity> paint_capacities;
    std::uint32_t rigid_count{};
    std::uint32_t authored_constraint_count{};
    std::uint32_t contact_capacity{};
    std::uint32_t debug_capacity{};
    SceneDefinition original_scene;
    bool fence_submitted{};
    bool output_ready{};
    bool contacts_available{};
    bool smoke_active{};

    ~Impl() { release(); }

    void destroy_buffer(Buffer &buffer) noexcept {
        if (device != nullptr && buffer.memory != nullptr && buffer.mapped != nullptr)
            pmUnmapMemory(device, buffer.memory);
        if (device != nullptr && buffer.handle != nullptr)
            pmDestroyBuffer(device, buffer.handle);
        if (device != nullptr && buffer.memory != nullptr)
            pmFreeMemory(device, buffer.memory);
        buffer = {};
    }

    void release() noexcept {
        if (device != nullptr) (void)pmDeviceWaitIdle(device);
        if (device != nullptr && pipeline != nullptr)
            pmDestroyPhysicsPipeline(device, pipeline);
        if (device != nullptr && statistics_query != nullptr)
            pmDestroyQueryPool(device, statistics_query);
        if (device != nullptr && simulation != nullptr)
            pmDestroySimulation(device, simulation);
        simulation = nullptr;
        for (auto &geometry : geometries) {
            if (device != nullptr && geometry.handle != nullptr)
                pmDestroyGeometry(device, geometry.handle);
            destroy_buffer(geometry.input);
            if (device != nullptr && geometry.memory != nullptr)
                pmFreeMemory(device, geometry.memory);
        }
        geometries.clear();
        destroy_buffer(particle_input);
        destroy_buffer(output);
        if (device != nullptr && persistent_memory != nullptr)
            pmFreeMemory(device, persistent_memory);
        if (device != nullptr && scratch_memory != nullptr)
            pmFreeMemory(device, scratch_memory);
        if (device != nullptr && fence != nullptr) pmDestroyFence(device, fence);
        if (device != nullptr && command_pool != nullptr) {
            if (commands != nullptr)
                pmFreeCommandBuffers(device, command_pool, 1U, &commands);
            if (reset_commands != nullptr)
                pmFreeCommandBuffers(
                    device, command_pool, 1U, &reset_commands);
            pmDestroyCommandPool(device, command_pool);
        }
        if (device != nullptr) pmDestroyDevice(device);
        if (instance != nullptr) pmDestroyInstance(instance);
    }

    [[nodiscard]] bool create_buffer(PmDeviceSize size, Buffer &created,
                                     std::string &error,
                                     std::uint32_t memory_type = 1U) {
        if (size == 0U) size = 256U;
        PmBufferCreateInfo buffer_info{};
        buffer_info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = k_buffer_usage;
        if (!checked(pmCreateBuffer(device, &buffer_info, &created.handle),
                     "create buffer", error))
            return false;
        PmMemoryAllocateInfo allocation{};
        allocation.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = size;
        allocation.memoryTypeIndex = memory_type;
        if (!checked(pmAllocateMemory(device, &allocation, &created.memory),
                     "allocate buffer memory", error) ||
            !checked(pmBindBufferMemory(device, created.handle, created.memory, 0U),
                     "bind buffer memory", error))
            return false;
        created.size = size;
        if (memory_type == 1U &&
            !checked(pmMapMemory(device, created.memory, 0U, PM_WHOLE_SIZE,
                                 &created.mapped),
                     "map buffer memory", error))
            return false;
        return true;
    }

    [[nodiscard]] bool begin(std::string &error) {
        if (fence_submitted) {
            if (!checked(pmResetFences(device, 1U, &fence), "reset fence", error))
                return false;
            fence_submitted = false;
        }
        if (!checked(pmResetCommandBuffer(commands), "reset command buffer", error))
            return false;
        PmCommandBufferBeginInfo begin_info{};
        begin_info.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = PM_COMMAND_BUFFER_ONE_TIME_BIT;
        if (!checked(pmBeginCommandBuffer(commands, &begin_info),
                     "begin command buffer", error))
            return false;
        return pipeline == nullptr ||
            checked(pmCmdBindPhysicsPipeline(commands, pipeline),
                    "bind physics pipeline", error);
    }

    [[nodiscard]] bool submit(std::string &error,
                              PmResult expected_result = PM_SUCCESS) {
        if (!checked(pmEndCommandBuffer(commands), "end command buffer", error))
            return false;
        PmSubmitInfo submit_info{};
        submit_info.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1U;
        submit_info.pCommandBuffers = &commands;
        if (!checked(pmQueueSubmit(queue, &submit_info, fence), "submit commands",
                     error))
            return false;
        fence_submitted = true;
        const PmResult completion = pmWaitForFences(
            device, 1U, &fence, PM_TRUE, PM_TIMEOUT_INFINITE);
        if (completion != expected_result) {
            error = "wait for commands: expected ";
            error += result_name(expected_result);
            error += ", received ";
            error += result_name(completion);
            return false;
        }
        return true;
    }

    [[nodiscard]] bool submit_reset(std::string &error) {
        if (fence_submitted) {
            if (!checked(pmResetFences(device, 1U, &fence), "reset fence",
                         error))
                return false;
            fence_submitted = false;
        }
        const PmSubmitInfo submit_info{
            PM_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0U, nullptr, 1U,
            &reset_commands, 0U, nullptr};
        if (!checked(pmQueueSubmit(queue, &submit_info, fence),
                     "submit reset commands", error))
            return false;
        fence_submitted = true;
        return checked(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                                       PM_TIMEOUT_INFINITE),
                       "wait for reset commands", error);
    }

    [[nodiscard]] bool create_geometry(
        const std::vector<TriangleMesh> &source_meshes,
        const std::vector<std::uint32_t> &source_indices,
        std::uint32_t &output_index, std::string &error) {
        std::vector<PmFloat3> vertices;
        std::vector<std::uint32_t> indices;
        try {
            for (std::uint32_t mesh_index : source_indices) {
                if (mesh_index >= source_meshes.size()) {
                    error = "scene geometry references an invalid mesh";
                    return false;
                }
                const auto &mesh = source_meshes[mesh_index];
                const std::uint32_t base =
                    static_cast<std::uint32_t>(vertices.size());
                for (const Vertex &vertex : mesh.vertices)
                    vertices.push_back(pm(vertex.position));
                for (std::uint32_t index : mesh.indices)
                    indices.push_back(base + index);
            }
        } catch (const std::bad_alloc &) {
            error = "assemble geometry: out of host memory";
            return false;
        }
        if (vertices.empty() || indices.empty() || indices.size() % 3U != 0U) {
            error = "scene geometry is empty or not triangulated";
            return false;
        }
        Geometry geometry{};
        const PmDeviceSize vertex_bytes = vertices.size() * sizeof(PmFloat3);
        const PmDeviceSize index_bytes = indices.size() * sizeof(std::uint32_t);
        if (!create_buffer(vertex_bytes + index_bytes, geometry.input, error))
            return false;
        std::memcpy(geometry.input.mapped, vertices.data(), vertex_bytes);
        std::memcpy(static_cast<std::byte *>(geometry.input.mapped) + vertex_bytes,
                    indices.data(), index_bytes);
        PmGeometryCreateInfo create_info{};
        create_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
        create_info.vertices = {{geometry.input.handle, 0U, vertex_bytes},
                                PM_FORMAT_R32G32B32_SFLOAT,
                                static_cast<std::uint32_t>(vertices.size()),
                                sizeof(PmFloat3)};
        create_info.indices = {{geometry.input.handle, vertex_bytes, index_bytes},
                               PM_FORMAT_R32_UINT,
                               static_cast<std::uint32_t>(indices.size()),
                               sizeof(std::uint32_t)};
        if (!checked(pmCreateGeometry(device, &create_info, &geometry.handle),
                     "create geometry", error))
            return false;
        PmMemoryRequirements requirements{};
        pmGetGeometryMemoryRequirements(device, geometry.handle, &requirements);
        PmMemoryAllocateInfo allocation{};
        allocation.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex =
            (requirements.memoryTypeBits & 0x2U) != 0U ? 1U : 0U;
        if (!checked(pmAllocateMemory(device, &allocation, &geometry.memory),
                     "allocate geometry memory", error) ||
            !checked(pmBindGeometryMemory(device, geometry.handle, geometry.memory,
                                          0U),
                     "bind geometry memory", error))
            return false;
        try {
            geometries.push_back(std::move(geometry));
        } catch (const std::bad_alloc &) {
            error = "store geometry: out of host memory";
            return false;
        }
        output_index = static_cast<std::uint32_t>(geometries.size() - 1U);
        return true;
    }
};

CommandScene::CommandScene() noexcept = default;
CommandScene::~CommandScene() = default;
CommandScene::CommandScene(CommandScene &&) noexcept = default;
CommandScene &CommandScene::operator=(CommandScene &&) noexcept = default;

#include "scene_builder_create.cpp.inc"
#include "scene_builder_output.cpp.inc"
