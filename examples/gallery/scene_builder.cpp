// SPDX-License-Identifier: MIT
#include "scene_builder.hpp"

#include <algorithm>
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
    std::uint32_t debug_frame_capacity{};
    bool fence_submitted{};

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

    [[nodiscard]] bool submit(std::string &error) {
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
        if (!checked(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                                     PM_TIMEOUT_INFINITE),
                     "wait for commands", error))
            return false;
        return true;
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

bool CommandScene::create(const SceneDefinition &scene,
                          std::uint32_t debug_frame_capacity,
                          CommandScene &output,
                          CommandSceneInstance &instance,
                          std::string &error) {
    error.clear();
    output = {};
    instance = {};
    auto next = std::make_unique<Impl>();

    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    instance_info.pApplicationName = "parallel-mater-gallery";
    instance_info.enableValidation = PM_TRUE;
    if (!checked(pmCreateInstance(&instance_info, nullptr, &next->instance),
                 "create Parallel Mater instance", error))
        return false;
    std::uint32_t physical_count = 0U;
    if (!checked(pmEnumeratePhysicalDevices(next->instance, &physical_count,
                                            nullptr),
                 "enumerate physical devices", error) ||
        physical_count == 0U) {
        if (error.empty()) error = "no Parallel Mater physical device";
        return false;
    }
    std::vector<PmPhysicalDevice> physical_devices(physical_count);
    if (!checked(pmEnumeratePhysicalDevices(next->instance, &physical_count,
                                            physical_devices.data()),
                 "enumerate physical devices", error))
        return false;
    next->physical = physical_devices.front();
    PmPhysicalDeviceFeatures features{};
    pmGetPhysicalDeviceFeatures(next->physical, &features);
    const bool requires_fluid = scene.fluid_options.capacity != 0U;
    if (features.rigidBodies != PM_TRUE ||
        (!scene.rigid_constraints.empty() &&
         features.rigidConstraints != PM_TRUE) ||
        (requires_fluid && features.particleLiquids != PM_TRUE) ||
        (!scene.cloths.empty() && features.cloth != PM_TRUE) ||
        (!scene.ropes.empty() && features.rope != PM_TRUE) ||
        (!scene.soft_bodies.empty() && features.softBodies != PM_TRUE) ||
        (scene.has_smoke && features.smoke != PM_TRUE)) {
        error = "selected device does not expose every scene capability";
        return false;
    }
    float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures = features;
    if (!checked(pmCreateDevice(next->physical, &device_info, &next->device),
                 "create Parallel Mater device", error))
        return false;
    pmGetDeviceQueue(next->device, 0U, 0U, &next->queue);
    if (next->queue == nullptr) {
        error = "get Parallel Mater physics queue: invalid state";
        return false;
    }
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    if (!checked(pmCreateCommandPool(next->device, &pool_info,
                                     &next->command_pool),
                 "create command pool", error) ||
        !checked(pmAllocateCommandBuffers(next->device, next->command_pool, 1U,
                                          &next->commands),
                 "allocate command buffer", error) ||
        !checked(pmCreateFence(next->device, 0U, &next->fence),
                 "create queue fence", error))
        return false;

    std::vector<std::uint32_t> body_geometry(scene.rigid_bodies.size());
    try {
        next->geometry_cache.reserve(scene.rigid_bodies.size() + 16U);
        for (std::size_t body_index = 0U;
             body_index < scene.rigid_bodies.size(); ++body_index) {
            const auto &body = scene.rigid_bodies[body_index];
            const bool collision = !body.collision_mesh_indices.empty();
            const auto &mesh_indices = collision ? body.collision_mesh_indices
                                                 : body.mesh_indices;
            const std::string key = mesh_key(collision, mesh_indices);
            const auto cached = next->geometry_cache.find(key);
            if (cached != next->geometry_cache.end()) {
                body_geometry[body_index] = cached->second;
                continue;
            }
            std::uint32_t geometry_index = 0U;
            if (!next->create_geometry(
                    collision ? scene.collision_meshes : scene.meshes,
                    mesh_indices, geometry_index, error))
                return false;
            next->geometry_cache.emplace(key, geometry_index);
            body_geometry[body_index] = geometry_index;
        }
    } catch (const std::bad_alloc &) {
        error = "build scene geometry cache: out of host memory";
        return false;
    }

    struct PaintDefinition {
        std::uint32_t body{PM_INVALID_INDEX};
        std::uint32_t cloth{PM_INVALID_INDEX};
        std::uint32_t mesh{};
        std::uint32_t geometry{PM_INVALID_INDEX};
        std::uint32_t resolution{};
        std::uint32_t rigid_source{PM_INVALID_INDEX};
        float brush_radius{};
        bool liquid_source{};
    };
    std::vector<PaintDefinition> paints;
    try {
        for (std::uint32_t body_index = 0U;
             body_index < scene.rigid_bodies.size(); ++body_index) {
            const auto &body = scene.rigid_bodies[body_index];
            if (!body.paintable) continue;
            for (std::uint32_t mesh_index : body.mesh_indices) {
                const std::string key = "paint:" + std::to_string(mesh_index);
                auto found = next->geometry_cache.find(key);
                if (found == next->geometry_cache.end()) {
                    std::uint32_t geometry_index = 0U;
                    if (!next->create_geometry(scene.meshes, {mesh_index},
                                               geometry_index, error))
                        return false;
                    found = next->geometry_cache.emplace(key, geometry_index).first;
                }
                paints.push_back({body_index, PM_INVALID_INDEX, mesh_index,
                    found->second, body.paint_resolution, PM_INVALID_INDEX,
                    0.15F, true});
            }
        }
        for (std::uint32_t cloth_index = 0U;
             cloth_index < scene.cloths.size(); ++cloth_index) {
            const auto &cloth = scene.cloths[cloth_index];
            if (!cloth.paintable) continue;
            const auto source = std::find_if(
                scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
                [&](const RigidBodyDefinition &body) {
                    return body.source_name == cloth.paint_source;
                });
            if (source == scene.rigid_bodies.end()) {
                error = "cloth paint source is missing";
                return false;
            }
            paints.push_back({PM_INVALID_INDEX, cloth_index, cloth.mesh_index,
                PM_INVALID_INDEX, cloth.paint_resolution,
                static_cast<std::uint32_t>(source - scene.rigid_bodies.begin()),
                cloth.paint_brush_radius, false});
        }
    } catch (const std::bad_alloc &) {
        error = "build paint bindings: out of host memory";
        return false;
    }

    if (requires_fluid) {
        next->fluid_capacities.push_back({scene.fluid_options.capacity,
                                          scene.fluid_options.maximum_neighbors});
    }
    for (const auto &cloth : scene.cloths) {
        if (cloth.mesh_index >= scene.meshes.size()) {
            error = "cloth references an invalid mesh";
            return false;
        }
        const auto &mesh = scene.meshes[cloth.mesh_index];
        next->cloth_capacities.push_back({
            static_cast<std::uint32_t>(
                std::max(mesh.vertices.size(), mesh.indices.size())),
            static_cast<std::uint32_t>(mesh.indices.size()),
            static_cast<std::uint32_t>(
                std::max<std::size_t>(mesh.indices.size(), 1U) * 2U)});
    }
    std::vector<std::vector<Vec3>> sampled_ropes(scene.ropes.size());
    for (std::size_t index = 0U; index < scene.ropes.size(); ++index) {
        const auto &rope = scene.ropes[index];
        const Status sampled = sample_rope_centerline(
            {rope.centerline.data(), rope.centerline.size()},
            rope.options.node_spacing, sampled_ropes[index]);
        if (!sampled) {
            error = sampled.message != nullptr ? sampled.message
                                               : "sample rope centerline failed";
            return false;
        }
        next->rope_capacities.push_back(
            {static_cast<std::uint32_t>(sampled_ropes[index].size())});
    }
    for (const auto &body : scene.soft_bodies) {
        if (body.mesh_index >= scene.meshes.size()) {
            error = "soft body references an invalid mesh";
            return false;
        }
        const auto &mesh = scene.meshes[body.mesh_index];
        next->soft_body_capacities.push_back({
            static_cast<std::uint32_t>(body.nodes.size()),
            static_cast<std::uint32_t>(body.bonds.size()),
            static_cast<std::uint32_t>(mesh.vertices.size()),
            static_cast<std::uint32_t>(mesh.indices.size())});
    }
    if (scene.has_smoke) {
        const std::uint64_t cells = grid_cells(scene.smoke_options);
        if (cells > std::numeric_limits<std::uint32_t>::max()) {
            error = "smoke grid exceeds the public capacity range";
            return false;
        }
        next->smoke_capacities.push_back(
            {scene.smoke_options.capacity, static_cast<std::uint32_t>(cells)});
    }
    for (const PaintDefinition &paint : paints) {
        const std::size_t vertices = scene.meshes[paint.mesh].vertices.size();
        next->paint_capacities.push_back({
            static_cast<std::uint32_t>(vertices), paint.resolution,
            paint.resolution});
    }

    std::vector<PmCouplingState> couplings;
    const auto add_mechanical = [&](PmCouplingType type, std::uint32_t first,
                                    std::uint32_t second,
                                    PmMechanicalCouplingState parameters = {}) {
        PmCouplingState state{};
        state.type = type;
        state.first = first;
        state.second = second;
        state.parameters.mechanical = parameters;
        couplings.push_back(state);
    };
    try {
        if (requires_fluid)
            add_mechanical(PM_COUPLING_LIQUID_RIGID, 0U, 0U);
        for (std::uint32_t cloth = 0U; cloth < scene.cloths.size(); ++cloth) {
            if (requires_fluid && scene.cloths[cloth].contains_fluid) {
                const FluidClothCouplingOptions options{};
                add_mechanical(PM_COUPLING_LIQUID_CLOTH, 0U, cloth,
                    {options.contact_distance, options.stiffness,
                     options.damping, 0.0F, options.tangential_drag,
                     options.maximum_force, options.interaction_radius,
                     PM_TRUE});
            }
            for (std::uint32_t soft = 0U;
                 soft < scene.soft_bodies.size(); ++soft) {
                const SoftBodyClothCouplingOptions options{};
                add_mechanical(PM_COUPLING_SOFT_BODY_CLOTH, soft, cloth,
                    {options.contact_distance, 0.0F, 0.0F,
                     scene.cloths[cloth].contact_friction, 0.0F, 0.0F, 0.0F,
                     PM_TRUE});
            }
        }
        for (std::uint32_t soft = 0U; soft < scene.soft_bodies.size(); ++soft) {
            if (requires_fluid) {
                const FluidSoftBodyCouplingOptions options{};
                add_mechanical(PM_COUPLING_LIQUID_SOFT_BODY, 0U, soft,
                    {options.contact_distance, 0.0F, 0.0F,
                     options.friction, 0.0F, 0.0F, 0.0F, PM_TRUE});
            }
        }
        for (std::uint32_t rope = 0U; rope < scene.ropes.size(); ++rope) {
            if (requires_fluid) {
                const FluidRopeCouplingOptions options{};
                add_mechanical(PM_COUPLING_LIQUID_ROPE, 0U, rope,
                    {options.contact_distance, 0.0F, 0.0F,
                     options.friction, 0.0F, 0.0F,
                     options.maximum_rope_acceleration, PM_TRUE});
            }
            for (std::uint32_t soft = 0U;
                 soft < scene.soft_bodies.size(); ++soft) {
                const auto &definition = scene.ropes[rope];
                const RopeSoftBodyCouplingOptions options{};
                PmCouplingState state{};
                state.type = PM_COUPLING_ROPE_SOFT_BODY;
                state.first = rope;
                state.second = soft;
                state.parameters.mechanical.contactDistance =
                    options.contact_distance;
                state.parameters.mechanical.friction = options.friction;
                state.parameters.attachment = {
                    PM_INVALID_INDEX, PM_INVALID_INDEX,
                    0.0F, options.anchor_support_radius_scale,
                    options.anchor_contact_support_radius_scale,
                    options.maximum_soft_body_acceleration,
                    definition.first_soft_body == static_cast<int>(soft)
                        ? PM_TRUE : PM_FALSE,
                    definition.last_soft_body == static_cast<int>(soft)
                        ? PM_TRUE : PM_FALSE};
                couplings.push_back(state);
            }
            for (std::uint32_t cloth = 0U;
                 cloth < scene.cloths.size(); ++cloth) {
                const auto &definition = scene.ropes[rope];
                const bool first =
                    definition.first_cloth == static_cast<int>(cloth);
                const bool last =
                    definition.last_cloth == static_cast<int>(cloth);
                if (!first && !last) continue;
                const RopeClothCouplingOptions options{};
                PmCouplingState state{};
                state.type = PM_COUPLING_ROPE_CLOTH;
                state.first = rope;
                state.second = cloth;
                state.parameters.attachment = {
                    first ? definition.first_cloth_vertex : PM_INVALID_INDEX,
                    last ? definition.last_cloth_vertex : PM_INVALID_INDEX,
                    options.anchor_effective_mass, 0.0F, 0.0F,
                    options.maximum_cloth_acceleration, PM_FALSE, PM_FALSE};
                couplings.push_back(state);
            }
        }
        for (const auto &surface : scene.thermal_surfaces) {
            PmCouplingState state{};
            state.type = PM_COUPLING_LIQUID_SMOKE;
            state.first = 0U;
            state.second = 0U;
            copy(state.parameters.thermal.heaterCenter, surface.plane.center);
            copy(state.parameters.thermal.heaterOrientation,
                 surface.plane.orientation);
            state.parameters.thermal.heaterHalfExtents[0] =
                surface.plane.half_extents.x;
            state.parameters.thermal.heaterHalfExtents[1] =
                surface.plane.half_extents.y;
            state.parameters.thermal.heaterTemperature = surface.temperature;
            state.parameters.thermal.boilingTemperature = 100.0F;
            state.parameters.thermal.heatTransferRate =
                surface.heat_transfer_rate;
            state.parameters.thermal.windDrag = surface.smoke_drag;
            state.parameters.thermal.steamRiseSpeed = surface.steam_rise_speed;
            couplings.push_back(state);
        }
        if (scene.has_smoke) {
            for (std::uint32_t cloth = 0U; cloth < scene.cloths.size(); ++cloth) {
                const SmokeClothCouplingOptions options{};
                PmCouplingState state{};
                state.type = PM_COUPLING_SMOKE_CLOTH;
                state.first = 0U;
                state.second = cloth;
                state.parameters.smoke = {options.wind_drag,
                    options.maximum_wind_acceleration, 0.0F, 0.0F,
                    options.contact_distance, PM_FALSE};
                couplings.push_back(state);
            }
            for (std::uint32_t rope = 0U; rope < scene.ropes.size(); ++rope) {
                const SmokeRopeCouplingOptions options{};
                PmCouplingState state{};
                state.type = PM_COUPLING_SMOKE_ROPE;
                state.first = 0U;
                state.second = rope;
                state.parameters.smoke = {options.wind_drag,
                    options.maximum_wind_acceleration, 0.0F, 0.0F,
                    options.contact_distance, PM_FALSE};
                couplings.push_back(state);
            }
            for (std::uint32_t soft = 0U;
                 soft < scene.soft_bodies.size(); ++soft) {
                const SmokeSoftBodyCouplingOptions options{};
                PmCouplingState state{};
                state.type = PM_COUPLING_SMOKE_SOFT_BODY;
                state.first = 0U;
                state.second = soft;
                state.parameters.smoke = {options.wind_drag,
                    options.maximum_wind_acceleration, 0.0F, 0.0F,
                    options.contact_distance, PM_FALSE};
                couplings.push_back(state);
            }
            for (std::uint32_t body = 0U;
                 body < scene.rigid_bodies.size(); ++body) {
                const auto &definition = scene.rigid_bodies[body];
                const bool named_obstacle = !scene.smoke_obstacle_name.empty() &&
                    (definition.name == scene.smoke_obstacle_name ||
                     definition.source_name == scene.smoke_obstacle_name);
                if (!definition.smoke_collider && !named_obstacle &&
                    definition.options.motion != MotionType::dynamic)
                    continue;
                const SmokeRigidCouplingOptions options{};
                PmCouplingState state{};
                state.type = PM_COUPLING_SMOKE_RIGID;
                state.first = 0U;
                state.second = body;
                state.parameters.smoke = {0.0F, 0.0F, options.air_density,
                    options.drag_coefficient, options.contact_distance,
                    PM_TRUE};
                couplings.push_back(state);
            }
        }
    } catch (const std::bad_alloc &) {
        error = "build coupling pipeline: out of host memory";
        return false;
    }

    next->rigid_count = static_cast<std::uint32_t>(scene.rigid_bodies.size());
    next->contact_capacity = 65'536U;
    next->debug_capacity = debug_frame_capacity == 0U ? 0U :
        std::max<std::uint32_t>(1024U, next->rigid_count * 16U);
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = next->rigid_count;
    simulation_info.constraintCapacity = std::max<std::uint32_t>(
        static_cast<std::uint32_t>(scene.rigid_constraints.size()),
        next->rigid_count * 2U);
    simulation_info.contactCapacity = next->contact_capacity;
    simulation_info.fluidCount =
        static_cast<std::uint32_t>(next->fluid_capacities.size());
    simulation_info.pFluidCapacities = next->fluid_capacities.data();
    simulation_info.clothCount =
        static_cast<std::uint32_t>(next->cloth_capacities.size());
    simulation_info.pClothCapacities = next->cloth_capacities.data();
    simulation_info.ropeCount =
        static_cast<std::uint32_t>(next->rope_capacities.size());
    simulation_info.pRopeCapacities = next->rope_capacities.data();
    simulation_info.softBodyCount =
        static_cast<std::uint32_t>(next->soft_body_capacities.size());
    simulation_info.pSoftBodyCapacities = next->soft_body_capacities.data();
    simulation_info.smokeCount =
        static_cast<std::uint32_t>(next->smoke_capacities.size());
    simulation_info.pSmokeCapacities = next->smoke_capacities.data();
    simulation_info.couplingCapacity =
        static_cast<std::uint32_t>(couplings.size());
    simulation_info.emitterCapacity =
        static_cast<std::uint32_t>(scene.particle_sources.size());
    simulation_info.destroyerCapacity =
        static_cast<std::uint32_t>(scene.destroy_planes.size());
    simulation_info.paintFieldCount =
        static_cast<std::uint32_t>(next->paint_capacities.size());
    simulation_info.pPaintFieldCapacities = next->paint_capacities.data();
    simulation_info.paintRuleCapacity =
        static_cast<std::uint32_t>(paints.size());
    simulation_info.debugFrameCapacity = debug_frame_capacity;
    if (!checked(pmCreateSimulation(next->device, &simulation_info,
                                    &next->simulation),
                 "create simulation", error))
        return false;
    PmSimulationMemoryRequirements simulation_memory{};
    pmGetSimulationMemoryRequirements(next->device, next->simulation,
                                      &simulation_memory);
    PmMemoryAllocateInfo allocation{};
    allocation.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.memoryTypeIndex =
        (simulation_memory.persistent.memoryTypeBits & 0x2U) != 0U ? 1U : 0U;
    allocation.allocationSize = simulation_memory.persistent.size;
    if (!checked(pmAllocateMemory(next->device, &allocation,
                                  &next->persistent_memory),
                 "allocate persistent simulation memory", error))
        return false;
    allocation.memoryTypeIndex =
        (simulation_memory.scratch.memoryTypeBits & 0x2U) != 0U ? 1U : 0U;
    allocation.allocationSize = simulation_memory.scratch.size;
    if (!checked(pmAllocateMemory(next->device, &allocation,
                                  &next->scratch_memory),
                 "allocate scratch simulation memory", error))
        return false;
    const PmSimulationMemoryBinding binding{
        next->persistent_memory, 0U, next->scratch_memory, 0U};
    if (!checked(pmBindSimulationMemory(next->device, next->simulation, &binding),
                 "bind simulation memory", error))
        return false;

    PmRigidSolverState rigid_solver{};
    rigid_solver.rigidIterations = 10U;
    rigid_solver.enableSleeping = PM_FALSE;
    rigid_solver.sleepLinearThreshold = 0.02F;
    rigid_solver.sleepAngularThreshold = 0.02F;
    rigid_solver.sleepTimeSeconds = 0.5F;

    std::vector<PmLiquidMaterialState> fluid_materials;
    std::vector<PmClothMaterialState> cloth_materials;
    std::vector<PmRopeMaterialState> rope_materials;
    std::vector<PmSoftBodyMaterialState> soft_body_materials;
    std::vector<PmSmokeMaterialState> smoke_materials;
    try {
        if (requires_fluid) {
            const auto &source = scene.fluid_options;
            fluid_materials.push_back({
                source.rest_density, source.particle_radius,
                source.support_radius, source.rest_particle_volume,
                source.repulsion, source.viscosity, source.normal_damping,
                source.velocity_damping, source.maximum_speed,
                source.maximum_pair_acceleration, source.solver_iterations});
        }
        for (const auto &source : scene.cloths)
            cloth_materials.push_back({
                source.vertex_mass, source.thickness,
                source.stretch_compliance, 0.1F, source.velocity_damping,
                source.contact_friction, source.solver_iterations,
                source.break_strain, source.fracture_persistence_substeps,
                source.impact_break_impulse,
                source.preserve_volume ? PM_TRUE : PM_FALSE,
                source.target_volume, source.volume_compliance});
        for (const auto &source : scene.ropes) {
            const auto &options = source.options;
            rope_materials.push_back({
                options.node_spacing, options.radius, options.mass,
                options.stretch_compliance, options.velocity_damping,
                options.maximum_substep_timestep, options.friction,
                options.maximum_speed, options.solver_iterations,
                options.self_collision ? PM_TRUE : PM_FALSE});
        }
        for (const auto &source : scene.soft_bodies)
            soft_body_materials.push_back({
                source.node_mass, source.node_radius,
                source.stretch_compliance, source.velocity_damping,
                source.spring_damping, source.contact_friction,
                source.shape_matching_stiffness,
                source.maximum_projection_fraction,
                source.constraint_velocity_response, source.maximum_speed,
                source.solver_iterations});
        if (scene.has_smoke) {
            const auto &source = scene.smoke_options;
            smoke_materials.push_back({
                source.particle_radius, source.buoyancy, source.response,
                source.rest_number_density, source.pressure_stiffness,
                source.viscosity, source.vorticity_confinement,
                source.maximum_speed, source.grid_resolution,
                source.grid_vertical_resolution,
                source.grid_pressure_iterations,
                source.grid_kinematic_viscosity,
                source.grid_les_coefficient,
                source.grid_pressure_tolerance});
        }
    } catch (const std::bad_alloc &) {
        error = "assemble physics pipeline: out of host memory";
        return false;
    }
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid_solver;
    pipeline_info.fluidMaterialCount =
        static_cast<std::uint32_t>(fluid_materials.size());
    pipeline_info.pFluidMaterials = fluid_materials.data();
    pipeline_info.couplingCount = static_cast<std::uint32_t>(couplings.size());
    pipeline_info.pCouplings = couplings.data();
    pipeline_info.clothMaterialCount =
        static_cast<std::uint32_t>(cloth_materials.size());
    pipeline_info.pClothMaterials = cloth_materials.data();
    pipeline_info.ropeMaterialCount =
        static_cast<std::uint32_t>(rope_materials.size());
    pipeline_info.pRopeMaterials = rope_materials.data();
    pipeline_info.softBodyMaterialCount =
        static_cast<std::uint32_t>(soft_body_materials.size());
    pipeline_info.pSoftBodyMaterials = soft_body_materials.data();
    pipeline_info.smokeMaterialCount =
        static_cast<std::uint32_t>(smoke_materials.size());
    pipeline_info.pSmokeMaterials = smoke_materials.data();
    if (!checked(pmCreatePhysicsPipeline(next->device, &pipeline_info,
                                         &next->pipeline),
                 "create physics pipeline", error))
        return false;

    PmQueryPoolCreateInfo query_info{};
    query_info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = PM_QUERY_STEP_STATISTICS;
    query_info.queryCount = 1U;
    if (!checked(pmCreateQueryPool(next->device, &query_info,
                                   &next->statistics_query),
                 "create statistics query", error))
        return false;

    std::vector<PmRigidBodyInfo> rigid_bodies;
    std::vector<PmClothInfo> cloths;
    std::vector<std::vector<PmFloat3>> cloth_vertices;
    std::vector<PmRopeInfo> ropes;
    std::vector<std::vector<PmFloat3>> rope_centerlines;
    std::vector<PmSoftBodyInfo> soft_bodies;
    std::vector<std::vector<PmFloat3>> soft_nodes;
    std::vector<std::vector<PmFloat3>> soft_surface_vertices;
    std::vector<std::vector<PmSoftBodyBond>> soft_bonds;
    std::vector<std::vector<PmSoftBodySurfaceBinding>> soft_bindings;
    std::vector<PmParticleState> initial_particles;
    std::vector<PmParticleEmitterInfo> emitters;
    std::vector<std::vector<PmFloat3>> emitter_vertices;
    std::vector<PmParticleDestroyerInfo> destroyers;
    std::vector<PmPaintFieldInfo> paint_fields;
    std::vector<std::vector<PmFloat2>> paint_uvs;
    std::vector<PmPaintRuleInfo> paint_rules;
    try {
        rigid_bodies.resize(scene.rigid_bodies.size());
        for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
            const auto &source = scene.rigid_bodies[index].options;
            auto &target = rigid_bodies[index];
            target.active = PM_TRUE;
            target.motionType = pm(source.motion);
            target.geometry = next->geometries[body_geometry[index]].handle;
            target.state = pm(source.initial_state);
            target.mass = source.mass;
            copy(target.inertiaDiagonal, source.inertia_diagonal);
            target.friction = source.friction;
            target.restitution = source.restitution;
            target.linearDamping = source.linear_damping;
            target.angularDamping = source.angular_damping;
            target.maximumLinearSpeed = source.maximum_linear_speed;
            target.maximumAngularSpeed = source.maximum_angular_speed;
            target.collisionMargin = source.collision_margin;
            target.userData = source.user_data;
        }
        next->constraints.resize(simulation_info.constraintCapacity);
        next->authored_constraint_count = static_cast<std::uint32_t>(
            scene.rigid_constraints.size());
        for (std::size_t index = 0U;
             index < scene.rigid_constraints.size(); ++index)
            next->constraints[index] = pm(scene.rigid_constraints[index]);

        cloth_vertices.resize(scene.cloths.size());
        cloths.resize(scene.cloths.size());
        for (std::size_t index = 0U; index < scene.cloths.size(); ++index) {
            const auto &definition = scene.cloths[index];
            const auto &mesh = scene.meshes[definition.mesh_index];
            auto &vertices = cloth_vertices[index];
            vertices.reserve(mesh.vertices.size());
            for (const auto &vertex : mesh.vertices)
                vertices.push_back(pm(vertex.position));
            cloths[index] = {PM_TRUE, static_cast<std::uint32_t>(index),
                static_cast<std::uint32_t>(vertices.size()), vertices.data(),
                static_cast<std::uint32_t>(mesh.indices.size()),
                mesh.indices.data(), definition.inverse_masses.data()};
        }
        rope_centerlines.resize(scene.ropes.size());
        ropes.resize(scene.ropes.size());
        for (std::size_t index = 0U; index < scene.ropes.size(); ++index) {
            const auto &definition = scene.ropes[index];
            auto &points = rope_centerlines[index];
            points.reserve(definition.centerline.size());
            for (Vec3 point : definition.centerline) points.push_back(pm(point));
            auto &target = ropes[index];
            target.active = PM_TRUE;
            target.materialIndex = static_cast<std::uint32_t>(index);
            target.pointCount = static_cast<std::uint32_t>(points.size());
            target.pCenterline = points.data();
            if (definition.first_body >= 0) {
                target.first = {
                    static_cast<std::uint32_t>(definition.first_body),
                    pm(definition.options.first.local_anchor), PM_TRUE};
            }
            if (definition.last_body >= 0) {
                target.last = {
                    static_cast<std::uint32_t>(definition.last_body),
                    pm(definition.options.last.local_anchor), PM_TRUE};
            }
        }
        soft_nodes.resize(scene.soft_bodies.size());
        soft_surface_vertices.resize(scene.soft_bodies.size());
        soft_bonds.resize(scene.soft_bodies.size());
        soft_bindings.resize(scene.soft_bodies.size());
        soft_bodies.resize(scene.soft_bodies.size());
        for (std::size_t index = 0U; index < scene.soft_bodies.size(); ++index) {
            const auto &definition = scene.soft_bodies[index];
            const auto &mesh = scene.meshes[definition.mesh_index];
            for (Vec3 node : definition.nodes)
                soft_nodes[index].push_back(pm(node));
            for (const auto &bond : definition.bonds)
                soft_bonds[index].push_back(
                    {bond.first, bond.second, bond.rest_length});
            for (const auto &vertex : mesh.vertices)
                soft_surface_vertices[index].push_back(pm(vertex.position));
            for (const auto &binding_source : definition.surface_bindings) {
                PmSoftBodySurfaceBinding binding_target{};
                std::copy_n(binding_source.nodes, 4U, binding_target.nodes);
                std::copy_n(binding_source.weights, 4U, binding_target.weights);
                soft_bindings[index].push_back(binding_target);
            }
            soft_bodies[index] = {PM_TRUE, static_cast<std::uint32_t>(index),
                static_cast<std::uint32_t>(soft_nodes[index].size()),
                soft_nodes[index].data(),
                static_cast<std::uint32_t>(soft_bonds[index].size()),
                soft_bonds[index].data(), definition.inverse_masses.data(),
                static_cast<std::uint32_t>(
                    soft_surface_vertices[index].size()),
                soft_surface_vertices[index].data(),
                static_cast<std::uint32_t>(mesh.indices.size()),
                mesh.indices.data(), soft_bindings[index].data()};
        }

        if (requires_fluid) {
            const std::size_t requested = std::min<std::size_t>(
                scene.fluid_options.capacity, scene.initial_particles.size());
            initial_particles.reserve(requested);
            for (std::size_t index = 0U; index < requested; ++index) {
                const std::size_t source_index =
                    index * scene.initial_particles.size() / requested;
                const auto &source = scene.initial_particles[source_index];
                PmParticleState target{};
                copy(target.position, source.position);
                copy(target.velocity, source.velocity);
                target.temperatureCelsius = source.temperature;
                initial_particles.push_back(target);
            }
        }
        emitter_vertices.resize(scene.particle_sources.size());
        emitters.resize(scene.particle_sources.size());
        for (std::size_t index = 0U; index < scene.particle_sources.size(); ++index) {
            const auto &source = scene.particle_sources[index];
            for (Vec3 vertex : source.vertices)
                emitter_vertices[index].push_back(pm(vertex));
            emitters[index] = {source.options.enabled ? PM_TRUE : PM_FALSE, 0U,
                static_cast<std::uint32_t>(emitter_vertices[index].size()),
                emitter_vertices[index].data(),
                static_cast<std::uint32_t>(source.indices.size()),
                source.indices.data(), source.spacing,
                pm(source.options.initial_velocity),
                source.options.initial_temperature};
        }
        destroyers.resize(scene.destroy_planes.size());
        for (std::size_t index = 0U; index < scene.destroy_planes.size(); ++index) {
            const auto &source = scene.destroy_planes[index];
            auto &target = destroyers[index];
            target.active = source.enabled ? PM_TRUE : PM_FALSE;
            target.fluidIndex = 0U;
            target.center = pm(source.plane.center);
            copy(target.orientation, source.plane.orientation);
            target.halfExtents = {source.plane.half_extents.x,
                                  source.plane.half_extents.y};
            switch (source.crossing) {
            case CrossingDirection::along_normal:
                target.crossing = PM_CROSSING_ALONG_NORMAL;
                break;
            case CrossingDirection::against_normal:
                target.crossing = PM_CROSSING_AGAINST_NORMAL;
                break;
            case CrossingDirection::either:
                target.crossing = PM_CROSSING_EITHER;
                break;
            }
        }
        paint_uvs.resize(paints.size());
        paint_fields.resize(paints.size());
        paint_rules.resize(paints.size());
        for (std::size_t index = 0U; index < paints.size(); ++index) {
            const auto &paint = paints[index];
            const auto &mesh = scene.meshes[paint.mesh];
            for (const auto &vertex : mesh.vertices)
                paint_uvs[index].push_back({vertex.uv.x, vertex.uv.y});
            auto &field = paint_fields[index];
            field.active = PM_TRUE;
            field.targetType = paint.body != PM_INVALID_INDEX
                ? PM_PAINT_TARGET_RIGID : PM_PAINT_TARGET_CLOTH;
            field.targetIndex = paint.body != PM_INVALID_INDEX
                ? paint.body : paint.cloth;
            field.geometry = paint.geometry == PM_INVALID_INDEX
                ? nullptr : next->geometries[paint.geometry].handle;
            field.vertexCount =
                static_cast<std::uint32_t>(paint_uvs[index].size());
            field.pVertexUvs = paint_uvs[index].data();
            field.width = paint.resolution;
            field.height = paint.resolution;
            auto &rule = paint_rules[index];
            rule.active = PM_TRUE;
            rule.sourceType = paint.liquid_source
                ? PM_PAINT_SOURCE_LIQUID : PM_PAINT_SOURCE_RIGID;
            rule.sourceIndex = paint.liquid_source ? 0U : paint.rigid_source;
            rule.targetFieldIndex = static_cast<std::uint32_t>(index);
            rule.reach = 0.025F;
            rule.brushRadius = paint.brush_radius;
        }
        next->original_scene = scene;
        next->debug_frame_capacity = debug_frame_capacity;
    } catch (const std::bad_alloc &) {
        error = "assemble scene resources: out of host memory";
        return false;
    }

    if (requires_fluid) {
        const PmDeviceSize bytes = std::max<PmDeviceSize>(
            sizeof(PmParticleState),
            initial_particles.size() * sizeof(PmParticleState));
        if (!next->create_buffer(bytes, next->particle_input, error)) return false;
        if (!initial_particles.empty())
            std::memcpy(next->particle_input.mapped, initial_particles.data(),
                        initial_particles.size() * sizeof(PmParticleState));
    }

    if (!next->begin(error)) return false;
    for (const auto &geometry : next->geometries)
        if (!checked(pmCmdBuildGeometry(next->commands, geometry.handle),
                     "record geometry build", error))
            return false;
    if (!rigid_bodies.empty() &&
        !checked(pmCmdWriteRigidBodies(next->commands, next->simulation, 0U,
                                       static_cast<std::uint32_t>(
                                           rigid_bodies.size()),
                                       rigid_bodies.data()),
                 "record rigid bodies", error))
        return false;
    if (!next->constraints.empty() &&
        !checked(pmCmdWriteRigidConstraints(
                     next->commands, next->simulation, 0U,
                     static_cast<std::uint32_t>(next->constraints.size()),
                     next->constraints.data()),
                 "record rigid constraints", error))
        return false;
    if (requires_fluid) {
        const PmBufferView view{{next->particle_input.handle, 0U,
                                 next->particle_input.size},
                                PM_FORMAT_PARTICLE_STATE,
                                static_cast<std::uint32_t>(
                                    initial_particles.size()),
                                sizeof(PmParticleState)};
        if (!checked(pmCmdSetFluidParticles(next->commands, next->simulation,
                                            0U, &view),
                     "record fluid particles", error))
            return false;
    }
    if (!cloths.empty() &&
        !checked(pmCmdWriteCloths(next->commands, next->simulation, 0U,
                                  static_cast<std::uint32_t>(cloths.size()),
                                  cloths.data()),
                 "record cloths", error))
        return false;
    if (!ropes.empty() &&
        !checked(pmCmdWriteRopes(next->commands, next->simulation, 0U,
                                 static_cast<std::uint32_t>(ropes.size()),
                                 ropes.data()),
                 "record ropes", error))
        return false;
    if (!soft_bodies.empty() &&
        !checked(pmCmdWriteSoftBodies(
                     next->commands, next->simulation, 0U,
                     static_cast<std::uint32_t>(soft_bodies.size()),
                     soft_bodies.data()),
                 "record soft bodies", error))
        return false;
    if (scene.has_smoke) {
        const auto &source = scene.smoke_options;
        const PmSmokeInfo smoke{PM_TRUE, 0U, source.capacity,
            pm(source.emitter_center),
            {source.emitter_half_extents.x, source.emitter_half_extents.y},
            pm(source.initial_velocity), pm(source.wind),
            source.particles_per_second, source.lifetime,
            pm(source.grid_minimum), source.grid_edge_length};
        if (!checked(pmCmdWriteSmoke(next->commands, next->simulation, 0U, 1U,
                                     &smoke),
                     "record smoke", error))
            return false;
    }
    if (!emitters.empty() &&
        !checked(pmCmdWriteParticleEmitters(
                     next->commands, next->simulation, 0U,
                     static_cast<std::uint32_t>(emitters.size()),
                     emitters.data()),
                 "record particle emitters", error))
        return false;
    if (!destroyers.empty() &&
        !checked(pmCmdWriteParticleDestroyers(
                     next->commands, next->simulation, 0U,
                     static_cast<std::uint32_t>(destroyers.size()),
                     destroyers.data()),
                 "record particle destroyers", error))
        return false;
    if (!paint_fields.empty() &&
        (!checked(pmCmdWritePaintFields(
                      next->commands, next->simulation, 0U,
                      static_cast<std::uint32_t>(paint_fields.size()),
                      paint_fields.data()),
                  "record paint fields", error) ||
         !checked(pmCmdWritePaintRules(
                      next->commands, next->simulation, 0U,
                      static_cast<std::uint32_t>(paint_rules.size()),
                      paint_rules.data()),
                  "record paint rules", error)))
        return false;
    if (!next->submit(error)) return false;

    PmDeviceSize output_size = 0U;
    const auto append_region = [&](PmDeviceSize bytes, Impl::Region &region) {
        output_size = (output_size + 15U) & ~PmDeviceSize{15U};
        region = {output_size, bytes};
        output_size += bytes;
    };
    append_region(next->rigid_count * sizeof(PmRigidState), next->rigid_output);
    if (requires_fluid)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->fluid_capacities[0].particleCapacity) *
                sizeof(PmFluidParticleOutput), next->fluid_output);
    next->cloth_outputs.resize(next->cloth_capacities.size());
    next->cloth_surface_outputs.resize(next->cloth_capacities.size());
    for (std::size_t index = 0U; index < next->cloth_outputs.size(); ++index)
    {
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->cloth_capacities[index].vertexCapacity) *
                sizeof(PmClothVertexState), next->cloth_outputs[index]);
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(
                next->cloth_capacities[index].triangleIndexCapacity) *
                sizeof(PmFloat3), next->cloth_surface_outputs[index]);
    }
    next->rope_outputs.resize(next->rope_capacities.size());
    for (std::size_t index = 0U; index < next->rope_outputs.size(); ++index)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->rope_capacities[index].nodeCapacity) *
                sizeof(PmRopeNodeState), next->rope_outputs[index]);
    next->soft_body_outputs.resize(next->soft_body_capacities.size());
    next->soft_body_surface_outputs.resize(next->soft_body_capacities.size());
    for (std::size_t index = 0U; index < next->soft_body_outputs.size(); ++index)
    {
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->soft_body_capacities[index].nodeCapacity) *
                sizeof(PmSoftBodyNodeState), next->soft_body_outputs[index]);
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(
                next->soft_body_capacities[index].surfaceVertexCapacity) *
                sizeof(PmFloat3), next->soft_body_surface_outputs[index]);
    }
    if (scene.has_smoke)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->smoke_capacities[0].particleCapacity) *
                sizeof(PmSmokeParticleState), next->smoke_output);
    next->paint_outputs.resize(next->paint_capacities.size());
    for (std::size_t index = 0U; index < next->paint_outputs.size(); ++index)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->paint_capacities[index].width) *
                next->paint_capacities[index].height * sizeof(std::uint32_t),
            next->paint_outputs[index]);
    if (!next->constraints.empty())
        append_region(next->constraints.size() * sizeof(PmConstraintState),
                      next->constraint_output);
    append_region(sizeof(PmDomainOutputHeader) +
        static_cast<PmDeviceSize>(next->contact_capacity) * sizeof(PmContactEvent),
        next->contact_output);
    next->debug_capacity = next->rigid_count;
    if (requires_fluid)
        next->debug_capacity += next->fluid_capacities[0].particleCapacity;
    for (const auto &capacity : next->cloth_capacities)
        next->debug_capacity += capacity.vertexCapacity;
    for (const auto &capacity : next->rope_capacities)
        next->debug_capacity += capacity.nodeCapacity;
    for (const auto &capacity : next->soft_body_capacities)
        next->debug_capacity += capacity.nodeCapacity;
    if (debug_frame_capacity != 0U)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->debug_capacity) *
                sizeof(PmDebugRecord), next->debug_output);
    if (!next->create_buffer(output_size, next->output, error)) return false;

    instance.rigid_bodies.resize(scene.rigid_bodies.size());
    std::iota(instance.rigid_bodies.begin(), instance.rigid_bodies.end(), 0U);
    instance.rigid_constraints.resize(scene.rigid_constraints.size());
    std::iota(instance.rigid_constraints.begin(),
              instance.rigid_constraints.end(), 0U);
    instance.cloths.resize(scene.cloths.size());
    std::iota(instance.cloths.begin(), instance.cloths.end(), 0U);
    instance.ropes.resize(scene.ropes.size());
    std::iota(instance.ropes.begin(), instance.ropes.end(), 0U);
    instance.soft_bodies.resize(scene.soft_bodies.size());
    std::iota(instance.soft_bodies.begin(), instance.soft_bodies.end(), 0U);
    instance.fluid = requires_fluid ? 0U : PM_INVALID_INDEX;
    instance.smoke = scene.has_smoke ? 0U : PM_INVALID_INDEX;
    for (std::uint32_t index = 0U; index < paints.size(); ++index)
        instance.paint_bindings.push_back(
            {paints[index].body, paints[index].mesh, index});

    output.impl_ = std::move(next);
    return true;
}

namespace {

[[nodiscard]] PmBufferRange range(PmBuffer buffer,
                                  auto region) noexcept {
    return {buffer, region.offset, region.size};
}

template <typename T>
[[nodiscard]] bool read_domain(const std::byte *base,
                               auto region,
                               std::vector<T> &records,
                               std::uint64_t &revision,
                               std::string &error) {
    if (region.size < sizeof(PmDomainOutputHeader)) {
        records.clear();
        return true;
    }
    const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + region.offset);
    const PmDeviceSize needed = sizeof(*header) +
        static_cast<PmDeviceSize>(header->activeCount) * sizeof(T);
    if (header->activeCount > header->capacity || needed > region.size) {
        error = "simulation returned an invalid output header";
        return false;
    }
    try {
        const auto *begin = reinterpret_cast<const T *>(
            base + region.offset + sizeof(*header));
        records.assign(begin, begin + header->activeCount);
    } catch (const std::bad_alloc &) {
        error = "read simulation output: out of host memory";
        return false;
    }
    revision = std::max(revision, header->revision);
    return true;
}

} // namespace

bool CommandScene::step(float timestep, std::uint32_t substeps, Vec3 gravity,
                        CommandSceneFrame &frame, std::string &error,
                        bool collect_contacts) {
    error.clear();
    if (impl_ == nullptr || !(timestep > 0.0F) || substeps == 0U) {
        error = "step simulation: invalid argument";
        return false;
    }
    frame.previous_rigid_states = frame.rigid_states;
    if (!impl_->begin(error)) return false;
    if (!checked(pmCmdResetQueryPool(impl_->commands, impl_->statistics_query,
                                     0U, 1U),
                 "reset statistics query", error) ||
        !checked(pmCmdBeginStatisticsQuery(impl_->commands,
                                           impl_->statistics_query, 0U),
                 "begin statistics query", error))
        return false;
    PmStepInfo step_info{};
    step_info.timestep = timestep;
    step_info.substeps = substeps;
    copy(step_info.gravity, gravity);
    step_info.flags = collect_contacts
        ? static_cast<PmFlags>(PM_STEP_COLLECT_CONTACT_EVENTS_BIT) : 0U;
    if (!checked(pmCmdStepSimulation(impl_->commands, impl_->simulation,
                                     &step_info),
                 "record simulation step", error) ||
        !checked(pmCmdEndStatisticsQuery(impl_->commands,
                                         impl_->statistics_query, 0U),
                 "end statistics query", error))
        return false;
    const auto copy_range = [&](auto function, Impl::Region region,
                                std::string_view operation) {
        const PmBufferRange destination = range(impl_->output.handle, region);
        return checked(function(&destination), operation, error);
    };
    if (impl_->rigid_count != 0U &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyRigidStates(impl_->commands, impl_->simulation,
                    0U, impl_->rigid_count, destination);
            }, impl_->rigid_output, "copy rigid output"))
        return false;
    if (!impl_->fluid_capacities.empty() &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyFluidParticles(impl_->commands,
                    impl_->simulation, 0U, destination);
            }, impl_->fluid_output, "copy fluid output"))
        return false;
    for (std::uint32_t index = 0U; index < impl_->cloth_outputs.size(); ++index)
    {
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyClothState(impl_->commands, impl_->simulation,
                    index, destination);
            }, impl_->cloth_outputs[index], "copy cloth output"))
            return false;
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyClothSurface(impl_->commands,
                    impl_->simulation, index, destination);
            }, impl_->cloth_surface_outputs[index], "copy cloth surface"))
            return false;
    }
    for (std::uint32_t index = 0U; index < impl_->rope_outputs.size(); ++index)
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyRopeState(impl_->commands, impl_->simulation,
                    index, destination);
            }, impl_->rope_outputs[index], "copy rope output"))
            return false;
    for (std::uint32_t index = 0U;
         index < impl_->soft_body_outputs.size(); ++index)
    {
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopySoftBodyState(impl_->commands,
                    impl_->simulation, index, destination);
            }, impl_->soft_body_outputs[index], "copy soft body output"))
            return false;
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopySoftBodySurface(impl_->commands,
                    impl_->simulation, index, destination);
            }, impl_->soft_body_surface_outputs[index],
            "copy soft body surface"))
            return false;
    }
    if (!impl_->smoke_capacities.empty() &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopySmokeState(impl_->commands, impl_->simulation,
                    0U, destination);
            }, impl_->smoke_output, "copy smoke output"))
        return false;
    for (std::uint32_t index = 0U; index < impl_->paint_outputs.size(); ++index)
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyPaintField(impl_->commands, impl_->simulation,
                    index, destination);
            }, impl_->paint_outputs[index], "copy paint output"))
            return false;
    if (!impl_->constraints.empty() &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyConstraintStates(impl_->commands,
                    impl_->simulation, 0U,
                    static_cast<std::uint32_t>(impl_->constraints.size()),
                    destination);
            }, impl_->constraint_output, "copy constraint output"))
        return false;
    if (collect_contacts &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyContactEvents(impl_->commands,
                                              impl_->simulation, destination);
            }, impl_->contact_output, "copy contact events"))
        return false;
    if (impl_->debug_output.size != 0U &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyDebugData(impl_->commands, impl_->simulation,
                                          destination);
            }, impl_->debug_output, "copy debug output"))
        return false;
    if (!impl_->submit(error)) return false;

    const auto *base = static_cast<const std::byte *>(impl_->output.mapped);
    try {
        const auto *rigid = reinterpret_cast<const PmRigidState *>(
            base + impl_->rigid_output.offset);
        frame.rigid_states.assign(rigid, rigid + impl_->rigid_count);
        if (frame.previous_rigid_states.size() != frame.rigid_states.size())
            frame.previous_rigid_states = frame.rigid_states;
        frame.cloth_vertices.resize(impl_->cloth_outputs.size());
        frame.cloth_surface_vertices.resize(
            impl_->cloth_surface_outputs.size());
        frame.rope_nodes.resize(impl_->rope_outputs.size());
        frame.soft_body_nodes.resize(impl_->soft_body_outputs.size());
        frame.soft_body_surface_vertices.resize(
            impl_->soft_body_surface_outputs.size());
        frame.paint_fields.resize(impl_->paint_outputs.size());
    } catch (const std::bad_alloc &) {
        error = "read rigid simulation output: out of host memory";
        return false;
    }
    frame.revision = 0U;
    if (!impl_->fluid_capacities.empty() &&
        !read_domain(base, impl_->fluid_output, frame.fluid_particles,
                     frame.revision, error))
        return false;
    for (std::size_t index = 0U; index < impl_->cloth_outputs.size(); ++index)
    {
        if (!read_domain(base, impl_->cloth_outputs[index],
                         frame.cloth_vertices[index], frame.revision, error))
            return false;
        if (!read_domain(base, impl_->cloth_surface_outputs[index],
                         frame.cloth_surface_vertices[index],
                         frame.revision, error))
            return false;
    }
    for (std::size_t index = 0U; index < impl_->rope_outputs.size(); ++index)
        if (!read_domain(base, impl_->rope_outputs[index],
                         frame.rope_nodes[index], frame.revision, error))
            return false;
    for (std::size_t index = 0U;
         index < impl_->soft_body_outputs.size(); ++index)
    {
        if (!read_domain(base, impl_->soft_body_outputs[index],
                         frame.soft_body_nodes[index], frame.revision, error))
            return false;
        if (!read_domain(base, impl_->soft_body_surface_outputs[index],
                         frame.soft_body_surface_vertices[index],
                         frame.revision, error))
            return false;
    }
    if (!impl_->smoke_capacities.empty() &&
        !read_domain(base, impl_->smoke_output, frame.smoke_particles,
                     frame.revision, error))
        return false;
    for (std::size_t index = 0U; index < impl_->paint_outputs.size(); ++index)
        if (!read_domain(base, impl_->paint_outputs[index],
                         frame.paint_fields[index], frame.revision, error))
            return false;
    try {
        const auto *constraints = reinterpret_cast<const PmConstraintState *>(
            base + impl_->constraint_output.offset);
        frame.constraint_states.assign(
            constraints, constraints + impl_->constraints.size());
    } catch (const std::bad_alloc &) {
        error = "read constraint output: out of host memory";
        return false;
    }
    if (collect_contacts) {
        if (!read_domain(base, impl_->contact_output, frame.contacts,
                         frame.revision, error))
            return false;
    } else {
        frame.contacts.clear();
    }
    if (impl_->debug_output.size != 0U &&
        !read_domain(base, impl_->debug_output, frame.debug_records,
                     frame.revision, error))
        return false;
    const PmResult query_result = pmGetQueryPoolResults(
        impl_->device, impl_->statistics_query, 0U, 1U,
        sizeof(frame.statistics), &frame.statistics,
        sizeof(frame.statistics), PM_QUERY_RESULT_WAIT_BIT);
    return checked(query_result, "read statistics query", error);
}

bool CommandScene::set_rigid_state(std::uint32_t body,
                                   const RigidBodyState &state,
                                   std::string &error) {
    if (impl_ == nullptr || body >= impl_->rigid_count) {
        error = "set rigid state: invalid argument";
        return false;
    }
    const PmRigidState converted = pm(state);
    return impl_->begin(error) &&
        checked(pmCmdSetRigidStates(impl_->commands, impl_->simulation,
                                    body, 1U, &converted),
                "record rigid state", error) && impl_->submit(error);
}

bool CommandScene::set_kinematic_target(std::uint32_t body,
                                        const RigidBodyState &state,
                                        std::string &error) {
    if (impl_ == nullptr || body >= impl_->rigid_count) {
        error = "set kinematic target: invalid argument";
        return false;
    }
    const PmRigidState converted = pm(state);
    return impl_->begin(error) &&
        checked(pmCmdSetKinematicTargets(impl_->commands, impl_->simulation,
                                         body, 1U, &converted),
                "record kinematic target", error) && impl_->submit(error);
}

bool CommandScene::apply_rigid_inputs(const std::vector<PmRigidInput> &inputs,
                                      std::string &error) {
    if (impl_ == nullptr || inputs.empty()) return inputs.empty();
    return impl_->begin(error) &&
        checked(pmCmdApplyRigidInputs(impl_->commands, impl_->simulation,
                                      static_cast<std::uint32_t>(inputs.size()),
                                      inputs.data()),
                "record rigid inputs", error) && impl_->submit(error);
}

bool CommandScene::write_constraint(
    std::uint32_t constraint, const RigidConstraintDefinition &definition,
    std::string &error) {
    if (impl_ == nullptr || constraint >= impl_->constraints.size()) {
        error = "write rigid constraint: invalid argument";
        return false;
    }
    const PmRigidConstraintInfo converted = pm(definition);
    if (!impl_->begin(error) ||
        !checked(pmCmdWriteRigidConstraints(impl_->commands, impl_->simulation,
                                            constraint, 1U, &converted),
                 "record rigid constraint", error) ||
        !impl_->submit(error))
        return false;
    impl_->constraints[constraint] = converted;
    return true;
}

bool CommandScene::set_constraint_active(std::uint32_t constraint,
                                         bool active, std::string &error) {
    if (impl_ == nullptr || constraint >= impl_->constraints.size()) {
        error = "set rigid constraint active: invalid argument";
        return false;
    }
    PmRigidConstraintInfo converted = impl_->constraints[constraint];
    converted.active = active ? PM_TRUE : PM_FALSE;
    if (!impl_->begin(error) ||
        !checked(pmCmdWriteRigidConstraints(impl_->commands, impl_->simulation,
                                            constraint, 1U, &converted),
                 "record rigid constraint state", error) ||
        !impl_->submit(error))
        return false;
    impl_->constraints[constraint] = converted;
    return true;
}

bool CommandScene::reset(std::string &error) {
    if (impl_ == nullptr) {
        error = "reset simulation: invalid state";
        return false;
    }
    CommandScene replacement;
    CommandSceneInstance bindings;
    if (!create(impl_->original_scene, impl_->debug_frame_capacity,
                replacement, bindings, error))
        return false;
    impl_ = std::move(replacement.impl_);
    return true;
}

PmDevice CommandScene::device() const noexcept {
    return impl_ != nullptr ? impl_->device : nullptr;
}

PmSimulation CommandScene::simulation() const noexcept {
    return impl_ != nullptr ? impl_->simulation : nullptr;
}

PmQueue CommandScene::queue() const noexcept {
    return impl_ != nullptr ? impl_->queue : nullptr;
}

} // namespace parallel_mater::gallery
