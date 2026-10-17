// SPDX-License-Identifier: MIT
#include "internal.hpp"
#include "simulation/resource_plan.hpp"
#include "../simulation/exports.hpp"
#include "../simulation/geometry_constraints_dispatch.hpp"
#include "../simulation/geometry_storage.hpp"
#include "../simulation/primitives.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>

namespace {

using parallel_mater::Quaternion;
using parallel_mater::RigidBodyId;
using parallel_mater::RigidBodyOptions;
using parallel_mater::RigidBodyState;
using parallel_mater::RigidConstraintId;
using parallel_mater::RigidConstraintOptions;
using parallel_mater::RigidConstraintType;
using parallel_mater::TriangleMeshId;
using parallel_mater::Vec3;

[[nodiscard]] Vec3 vector3(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}

[[nodiscard]] Quaternion quaternion(const float value[4]) noexcept {
    return {value[0], value[1], value[2], value[3]};
}

[[nodiscard]] RigidBodyState rigid_state(const PmRigidState &value) noexcept {
    return {vector3(value.position), quaternion(value.orientation),
            vector3(value.linearVelocity), vector3(value.angularVelocity)};
}

[[nodiscard]] parallel_mater::MotionType motion_type(PmMotionType value) noexcept {
    switch (value) {
    case PM_MOTION_STATIC: return parallel_mater::MotionType::static_body;
    case PM_MOTION_KINEMATIC: return parallel_mater::MotionType::kinematic;
    case PM_MOTION_DYNAMIC: return parallel_mater::MotionType::dynamic;
    }
    return parallel_mater::MotionType::dynamic;
}

[[nodiscard]] bool finite(float value) noexcept { return std::isfinite(value); }

[[nodiscard]] bool finite(const PmRigidState &state) noexcept {
    for (float component : state.position) if (!finite(component)) return false;
    for (float component : state.orientation) if (!finite(component)) return false;
    for (float component : state.linearVelocity) if (!finite(component)) return false;
    for (float component : state.angularVelocity) if (!finite(component)) return false;
    return true;
}

[[nodiscard]] bool add_bytes(PmDeviceSize &total, PmDeviceSize count,
                             PmDeviceSize stride) noexcept {
    if (count != 0U && stride >
            std::numeric_limits<PmDeviceSize>::max() / count)
        return false;
    const PmDeviceSize bytes = count * stride;
    if (bytes > std::numeric_limits<PmDeviceSize>::max() - total) return false;
    total += bytes;
    return true;
}

[[nodiscard]] bool geometry_view_fits(PmDevice device,
                                      const PmBufferView &view,
                                      PmFormat format,
                                      PmDeviceSize element_size) noexcept {
    if (view.range.buffer == nullptr || view.range.buffer->device != device ||
        (view.range.buffer->usage & PM_BUFFER_GEOMETRY_INPUT_BIT) == 0U ||
        view.format != format || view.count == 0U ||
        view.stride != element_size ||
        view.count > std::numeric_limits<PmDeviceSize>::max() / element_size)
        return false;
    const PmDeviceSize required =
        static_cast<PmDeviceSize>(view.count) * element_size;
    return required <= view.range.size &&
        parallel_mater::api::range_fits(view.range.offset, view.range.size,
                                         view.range.buffer->size);
}

[[nodiscard]] PmResult geometry_requirements(
    const PmGeometryCreateInfo &info, PmMemoryRequirements &output) noexcept {
    parallel_mater::simulation::StorageRequirements plan{};
    const auto status = parallel_mater::simulation::geometry_storage_requirements(
        info.vertices.count, info.indices.count, plan);
    if (!status) return parallel_mater::api::from_status(status);
    // Geometry currently has no bound scratch storage. Keeping that class in
    // the shared plan makes a future addition fail here instead of silently
    // drifting away from the single-allocation public contract.
    if (plan.scratch != 0U) return PM_ERROR_INVALID_STATE;
    constexpr PmDeviceSize alignment = 256U;
    if (plan.persistent >
        std::numeric_limits<PmDeviceSize>::max() - (alignment - 1U))
        return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    const PmDeviceSize persistent =
        static_cast<PmDeviceSize>(plan.persistent);
    // Host BVH preparation reads this storage after upload. Keep the current
    // contract coherent and host-visible until building moves onto the GPU.
    output = {(persistent + alignment - 1U) & ~(alignment - 1U),
              alignment, 0x2U, PM_FALSE};
    return PM_SUCCESS;
}

[[nodiscard]] parallel_mater::SimulationOptions simulation_options(
    const PmSimulationCreateInfo &info) noexcept {
    parallel_mater::SimulationOptions options{};
    options.rigid_body_capacity = std::max(1U, info.rigidBodyCapacity);
    options.rigid_constraint_capacity = std::max(1U, info.constraintCapacity);
    options.triangle_mesh_capacity = std::max(
        16U, info.rigidBodyCapacity + info.paintFieldCount);
    options.contact_capacity = std::max(1U, info.contactCapacity);
    options.fluid_capacity = std::max(1U, info.fluidCount);
    options.cloth_capacity = std::max(1U, info.clothCount);
    options.rope_capacity = std::max(1U, info.ropeCount);
    options.soft_body_capacity = std::max(1U, info.softBodyCount);
    options.smoke_capacity = std::max(1U, info.smokeCount);
    options.particle_source_capacity = std::max(1U, info.emitterCapacity);
    options.particle_destroy_plane_capacity = std::max(1U, info.destroyerCapacity);
    options.paint_field_capacity = std::max(1U, info.paintFieldCount);
    options.paint_rule_capacity = std::max(1U, info.paintRuleCapacity);
    const std::uint32_t coupling_capacity = std::max(1U, info.couplingCapacity);
    options.fluid_smoke_coupling_capacity = coupling_capacity;
    options.smoke_soft_body_coupling_capacity = coupling_capacity;
    options.smoke_cloth_coupling_capacity = coupling_capacity;
    options.smoke_rope_coupling_capacity = coupling_capacity;
    options.smoke_rigid_coupling_capacity = coupling_capacity;
    options.fluid_cloth_coupling_capacity = coupling_capacity;
    options.soft_body_cloth_coupling_capacity = coupling_capacity;
    options.fluid_soft_body_coupling_capacity = coupling_capacity;
    options.fluid_rope_coupling_capacity = coupling_capacity;
    options.rope_soft_body_coupling_capacity = coupling_capacity;
    options.rope_cloth_coupling_capacity = coupling_capacity;
    options.physics_debug.frame_capacity = info.debugFrameCapacity;
    options.rigid_sleeping = false;
    return options;
}

[[nodiscard]] bool simulation_requirements(
    const PmSimulationCreateInfo &info,
    PmSimulationMemoryRequirements &output) noexcept {
    parallel_mater::simulation::StorageRequirements base{};
    if (!parallel_mater::SimulationRuntime::base_memory_requirements(
            simulation_options(info), base)) return false;
    PmDeviceSize persistent = base.persistent;
    PmDeviceSize scratch = base.scratch;
    const PmDeviceSize rigid = info.rigidBodyCapacity;
    for (const PmFluidCapacity capacity : std::span(info.pFluidCapacities, info.fluidCount)) {
        parallel_mater::simulation::StorageRequirements fluid{};
        if (!parallel_mater::SimulationRuntime::fluid_memory_requirements(capacity.particleCapacity, fluid) ||
            !add_bytes(persistent, 1, fluid.persistent) || !add_bytes(scratch, 1, fluid.scratch))
            return false;
    }
    for (const PmClothCapacity capacity : std::span(info.pClothCapacities, info.clothCount)) {
        parallel_mater::simulation::StorageRequirements plan{};
        if (!parallel_mater::SimulationRuntime::cloth_memory_requirements(capacity.vertexCapacity,
                capacity.triangleIndexCapacity, capacity.bondCapacity, std::max(1U, info.rigidBodyCapacity), plan) ||
            !add_bytes(persistent, 1, plan.persistent) || !add_bytes(scratch, 1, plan.scratch)) return false;
    }
    for (const PmRopeCapacity capacity : std::span(info.pRopeCapacities, info.ropeCount)) {
        parallel_mater::simulation::StorageRequirements plan{};
        if (!parallel_mater::SimulationRuntime::rope_memory_requirements(capacity.nodeCapacity,
                std::max(1U, info.rigidBodyCapacity), plan) ||
            !add_bytes(persistent, 1, plan.persistent) || !add_bytes(scratch, 1, plan.scratch)) return false;
    }
    for (const PmSoftBodyCapacity capacity : std::span(info.pSoftBodyCapacities, info.softBodyCount)) {
        parallel_mater::simulation::StorageRequirements plan{};
        if (!parallel_mater::SimulationRuntime::soft_body_memory_requirements(capacity.nodeCapacity,
                capacity.bondCapacity, capacity.surfaceVertexCapacity, capacity.surfaceTriangleIndexCapacity, plan) ||
            !add_bytes(persistent, 1, plan.persistent) || !add_bytes(scratch, 1, plan.scratch)) return false;
    }
    for (const PmSmokeCapacity capacity : std::span(info.pSmokeCapacities, info.smokeCount)) {
        parallel_mater::simulation::StorageRequirements plan{};
        if (!parallel_mater::SimulationRuntime::smoke_memory_requirements(capacity.particleCapacity,
                capacity.gridCellCapacity, plan) ||
            !add_bytes(persistent, 1, plan.persistent) || !add_bytes(scratch, 1, plan.scratch)) return false;
    }
    parallel_mater::simulation::CouplingStorageLimits limits{};
    for (auto capacity : std::span(info.pFluidCapacities, info.fluidCount))
        limits.fluid_particles = std::max(limits.fluid_particles, std::size_t(capacity.particleCapacity));
    for (auto capacity : std::span(info.pClothCapacities, info.clothCount))
        limits.cloth_vertices = std::max(limits.cloth_vertices, std::size_t(capacity.vertexCapacity));
    for (auto capacity : std::span(info.pRopeCapacities, info.ropeCount))
        limits.rope_nodes = std::max(limits.rope_nodes, std::size_t(capacity.nodeCapacity));
    for (auto capacity : std::span(info.pSoftBodyCapacities, info.softBodyCount)) {
        limits.soft_nodes = std::max(limits.soft_nodes, std::size_t(capacity.nodeCapacity));
        limits.soft_vertices = std::max(limits.soft_vertices, std::size_t(capacity.surfaceVertexCapacity));
        limits.soft_indices = std::max(limits.soft_indices, std::size_t(capacity.surfaceTriangleIndexCapacity));
    }
    parallel_mater::simulation::StorageRequirements coupling{}, emitter{};
    if (!parallel_mater::SimulationRuntime::coupling_memory_requirements(limits, coupling) ||
        !parallel_mater::SimulationRuntime::emitter_memory_requirements(emitter)) return false;
    if (!add_bytes(persistent, info.couplingCapacity, coupling.persistent) ||
        !add_bytes(persistent, info.emitterCapacity, emitter.persistent) ||
        !add_bytes(persistent, info.destroyerCapacity, 512U) ||
        !add_bytes(persistent, info.debugFrameCapacity,
                   std::max<PmDeviceSize>(1U, rigid) * 192U))
        return false;
    for (const PmPaintFieldCapacity capacity :
         std::span(info.pPaintFieldCapacities, info.paintFieldCount)) {
        parallel_mater::simulation::StorageRequirements plan{};
        if (!parallel_mater::SimulationRuntime::paint_field_memory_requirements(
                capacity.vertexCapacity, capacity.width, capacity.height,
                plan) ||
            !add_bytes(persistent, 1U, plan.persistent) ||
            !add_bytes(scratch, 1U, plan.scratch))
            return false;
    }
    constexpr PmDeviceSize alignment = 256U;
    if (persistent > std::numeric_limits<PmDeviceSize>::max() - (alignment - 1U) ||
        scratch > std::numeric_limits<PmDeviceSize>::max() - (alignment - 1U)) return false;
    persistent = (persistent + alignment - 1U) & ~(alignment - 1U);
    scratch = (scratch + alignment - 1U) & ~(alignment - 1U);
    output.persistent = {persistent, alignment, 0x2U, PM_FALSE};
    output.scratch = {scratch, alignment, 0x2U, PM_FALSE};
    return true;
}

[[nodiscard]] PmResult register_geometry_impl(
    PmSimulation simulation, PmGeometry geometry, parallel_mater::device::ExecutionQueue queue,
    TriangleMeshId &output) noexcept {
    if (geometry == nullptr || geometry->device != simulation->device ||
        !geometry->built || geometry->memory == nullptr)
        return PM_ERROR_INVALID_STATE;
    const auto found = simulation->geometries.find(geometry);
    if (found != simulation->geometries.end()) {
        output = found->second;
        return PM_SUCCESS;
    }
    const PmBufferView &vertices = geometry->info.vertices;
    const PmBufferView &indices = geometry->info.indices;
    if (parallel_mater::api::require_device_ownership(vertices.range.buffer) !=
            PM_SUCCESS ||
        parallel_mater::api::require_device_ownership(indices.range.buffer) !=
            PM_SUCCESS)
        return PM_ERROR_INVALID_STATE;
    if (vertices.format != PM_FORMAT_R32G32B32_SFLOAT ||
        indices.format != PM_FORMAT_R32_UINT ||
        vertices.stride != sizeof(Vec3) || indices.stride != sizeof(std::uint32_t))
        return PM_ERROR_FORMAT_NOT_SUPPORTED;
    auto *vertex_data = static_cast<Vec3 *>(
        parallel_mater::api::buffer_pointer(vertices.range.buffer,
                                             vertices.range.offset));
    auto *index_data = static_cast<std::uint32_t *>(
        parallel_mater::api::buffer_pointer(indices.range.buffer,
                                             indices.range.offset));
    if (vertex_data == nullptr || index_data == nullptr)
        return PM_ERROR_INVALID_STATE;
    TriangleMeshId created{};
    if (geometry->arena == nullptr) return PM_ERROR_INVALID_STATE;
    parallel_mater::simulation::ScopedMemoryArena geometry_scope(
        geometry->arena.get());
    const auto status = simulation->runtime->add_triangle_mesh(
        {vertex_data, vertices.count}, {index_data, indices.count}, created, queue);
    const PmResult result = parallel_mater::api::from_status(status);
    if (result != PM_SUCCESS) return result;
    try {
        simulation->geometries.emplace(geometry, created);
    } catch (const std::bad_alloc &) {
        (void)simulation->runtime->remove_triangle_mesh(created);
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    output = created;
    return PM_SUCCESS;
}

[[nodiscard]] PmResult retire_geometry_if_unused_impl(
    PmSimulation simulation, PmGeometry geometry) noexcept {
    if (geometry == nullptr ||
        std::find(simulation->rigid_body_geometries.begin(),
                  simulation->rigid_body_geometries.end(), geometry) !=
            simulation->rigid_body_geometries.end() ||
        std::find(simulation->paint_field_geometries.begin(),
                  simulation->paint_field_geometries.end(), geometry) !=
            simulation->paint_field_geometries.end())
        return PM_SUCCESS;
    const auto found = simulation->geometries.find(geometry);
    if (found == simulation->geometries.end()) return PM_SUCCESS;
    const parallel_mater::Status retired =
        simulation->runtime->remove_triangle_mesh(found->second);
    if (!retired) return parallel_mater::api::from_status(retired);
    simulation->geometries.erase(found);
    return PM_SUCCESS;
}

[[nodiscard]] PmResult validate_bound(PmSimulation simulation) noexcept {
    if (simulation == nullptr || simulation->runtime == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    return simulation->binding.persistentMemory != nullptr &&
            simulation->binding.scratchMemory != nullptr
        ? PM_SUCCESS : PM_ERROR_INVALID_STATE;
}

[[nodiscard]] RigidConstraintType constraint_type(PmConstraintType value) noexcept {
    switch (value) {
    case PM_CONSTRAINT_FIXED: return RigidConstraintType::fixed;
    case PM_CONSTRAINT_POINT: return RigidConstraintType::point;
    case PM_CONSTRAINT_HINGE: return RigidConstraintType::hinge;
    case PM_CONSTRAINT_SLIDER: return RigidConstraintType::slider;
    case PM_CONSTRAINT_PISTON: return RigidConstraintType::piston;
    case PM_CONSTRAINT_GENERIC: return RigidConstraintType::generic;
    case PM_CONSTRAINT_GENERIC_SPRING: return RigidConstraintType::generic_spring;
    case PM_CONSTRAINT_MOTOR: return RigidConstraintType::motor;
    }
    return RigidConstraintType::fixed;
}

void convert_axes(const PmAxisLimit source[3],
                  parallel_mater::RigidConstraintLimitOptions &destination) noexcept {
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        if (source[axis].enabled == PM_TRUE)
            destination.axes |= static_cast<std::uint8_t>(1U << axis);
        (&destination.lower.x)[axis] = source[axis].lower;
        (&destination.upper.x)[axis] = source[axis].upper;
    }
}

void convert_axes(const PmAxisSpring source[3],
                  parallel_mater::RigidConstraintSpringOptions &destination) noexcept {
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        if (source[axis].enabled == PM_TRUE)
            destination.axes |= static_cast<std::uint8_t>(1U << axis);
        (&destination.stiffness.x)[axis] = source[axis].stiffness;
        (&destination.damping.x)[axis] = source[axis].damping;
    }
}

} // namespace

namespace parallel_mater::api {

PmResult register_geometry(PmSimulation simulation, PmGeometry geometry,
                           parallel_mater::device::Services &services,
                           TriangleMeshId &output) noexcept {
    return register_geometry_impl(
        simulation, geometry, parallel_mater::device::execution_queue(services), output);
}

PmResult retire_geometry_if_unused(PmSimulation simulation,
                                   PmGeometry geometry) noexcept {
    return retire_geometry_if_unused_impl(simulation, geometry);
}

} // namespace parallel_mater::api

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateGeometry(
    PmDevice device, const PmGeometryCreateInfo *info, PmGeometry *geometry) {
    if (geometry == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *geometry = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO, info->pNext) ||
        info->queueFamilyIndex != 0U ||
        !geometry_view_fits(device, info->vertices,
                            PM_FORMAT_R32G32B32_SFLOAT, sizeof(Vec3)) ||
        !geometry_view_fits(device, info->indices, PM_FORMAT_R32_UINT,
                            sizeof(std::uint32_t)) ||
        info->indices.count % 3U != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    PmMemoryRequirements requirements{};
    const PmResult plan = geometry_requirements(*info, requirements);
    if (plan != PM_SUCCESS) return plan;
    auto *created = parallel_mater::api::make_object<PmGeometry_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->info = *info;
    created->requirements = requirements;
    *geometry = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyGeometry(
    PmDevice device, PmGeometry geometry) {
    if (device == nullptr || geometry == nullptr || geometry->device != device) return;
    (void)pmDeviceWaitIdle(device);
    parallel_mater::api::destroy_object(device, geometry);
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetGeometryMemoryRequirements(
    PmDevice device, PmGeometry geometry, PmMemoryRequirements *requirements) {
    if (requirements == nullptr) return;
    *requirements = {};
    if (device == nullptr || geometry == nullptr || geometry->device != device) return;
    *requirements = geometry->requirements;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmBindGeometryMemory(
    PmDevice device, PmGeometry geometry, PmDeviceMemory memory,
    PmDeviceSize offset) {
    if (device == nullptr || geometry == nullptr || memory == nullptr ||
        geometry->device != device || memory->device != device || offset % 256U != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    PmMemoryRequirements requirements{};
    pmGetGeometryMemoryRequirements(device, geometry, &requirements);
    if ((requirements.memoryTypeBits & (1U << memory->memory_type)) == 0U ||
        !parallel_mater::api::range_fits(offset, requirements.size, memory->size))
        return PM_ERROR_INVALID_ARGUMENT;
    if (geometry->memory != nullptr) return PM_ERROR_INVALID_STATE;
    try {
        geometry->arena =
            std::make_unique<parallel_mater::simulation::MemoryArena>(
                static_cast<std::byte *>(memory->allocation) + offset,
                static_cast<std::size_t>(requirements.size), nullptr, 0U,
                device->services.get());
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    geometry->memory = memory;
    geometry->memory_offset = offset;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdBuildGeometry(
    PmCommandBuffer command_buffer, PmGeometry geometry) {
    if (geometry == nullptr || geometry->device != command_buffer->device ||
        geometry->memory == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    return parallel_mater::api::record(
        command_buffer, [geometry](parallel_mater::device::Services &) noexcept {
            geometry->built = true;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateSimulation(
    PmDevice device, const PmSimulationCreateInfo *info,
    PmSimulation *simulation) {
    if (simulation == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *simulation = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO, info->pNext) ||
        info->queueFamilyIndex != 0U ||
        (info->rigidBodyCapacity == 0U && info->fluidCount == 0U &&
         info->clothCount == 0U && info->ropeCount == 0U &&
         info->softBodyCount == 0U && info->smokeCount == 0U) ||
        (info->fluidCount != 0U && info->pFluidCapacities == nullptr) ||
        (info->clothCount != 0U && info->pClothCapacities == nullptr) ||
        (info->ropeCount != 0U && info->pRopeCapacities == nullptr) ||
        (info->softBodyCount != 0U && info->pSoftBodyCapacities == nullptr) ||
        (info->smokeCount != 0U && info->pSmokeCapacities == nullptr) ||
        (info->paintFieldCount != 0U &&
         info->pPaintFieldCapacities == nullptr) ||
        static_cast<std::uint64_t>(info->rigidBodyCapacity) +
                info->paintFieldCount >
            std::numeric_limits<std::uint32_t>::max() ||
        (info->rigidBodyCapacity != 0U && device->enabled.rigidBodies == PM_FALSE) ||
        (info->constraintCapacity != 0U &&
         device->enabled.rigidConstraints == PM_FALSE) ||
        (info->fluidCount != 0U && device->enabled.particleLiquids == PM_FALSE))
        return PM_ERROR_INVALID_ARGUMENT;
    if ((info->clothCount != 0U && device->enabled.cloth == PM_FALSE) ||
        (info->ropeCount != 0U && device->enabled.rope == PM_FALSE) ||
        (info->softBodyCount != 0U && device->enabled.softBodies == PM_FALSE) ||
        (info->smokeCount != 0U && device->enabled.smoke == PM_FALSE) ||
        (info->emitterCapacity != 0U &&
         device->enabled.particleEmitters == PM_FALSE) ||
        ((info->paintFieldCount != 0U || info->paintRuleCapacity != 0U) &&
         device->enabled.paint == PM_FALSE))
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < info->fluidCount; ++index)
        if (info->pFluidCapacities[index].particleCapacity == 0U ||
            info->pFluidCapacities[index].neighborCapacityPerParticle == 0U)
            return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < info->clothCount; ++index)
        if (info->pClothCapacities[index].vertexCapacity == 0U ||
            info->pClothCapacities[index].triangleIndexCapacity == 0U ||
            info->pClothCapacities[index].bondCapacity == 0U)
            return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < info->ropeCount; ++index)
        if (info->pRopeCapacities[index].nodeCapacity < 2U)
            return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < info->softBodyCount; ++index) {
        const auto &capacity = info->pSoftBodyCapacities[index];
        if (capacity.nodeCapacity == 0U || capacity.bondCapacity == 0U ||
            capacity.surfaceVertexCapacity == 0U ||
            capacity.surfaceTriangleIndexCapacity == 0U)
            return PM_ERROR_INVALID_ARGUMENT;
    }
    for (std::uint32_t index = 0U; index < info->smokeCount; ++index)
        if (info->pSmokeCapacities[index].particleCapacity == 0U ||
            info->pSmokeCapacities[index].gridCellCapacity == 0U)
            return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < info->paintFieldCount; ++index) {
        const auto &capacity = info->pPaintFieldCapacities[index];
        if (capacity.vertexCapacity == 0U || capacity.width == 0U ||
            capacity.height == 0U || capacity.width > 4096U ||
            capacity.height > 4096U ||
            static_cast<std::uint64_t>(capacity.width) * capacity.height >
                16'777'216U)
            return PM_ERROR_INVALID_ARGUMENT;
    }
    PmSimulationMemoryRequirements memory_requirements{};
    if (!simulation_requirements(*info, memory_requirements))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmSimulation_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->info = *info;
    created->requirements = memory_requirements;
    try {
        if (info->fluidCount != 0U)
            created->fluids.assign(info->pFluidCapacities,
                                   info->pFluidCapacities + info->fluidCount);
        if (info->clothCount != 0U)
            created->cloth_capacities.assign(
                info->pClothCapacities,
                info->pClothCapacities + info->clothCount);
        if (info->ropeCount != 0U)
            created->rope_capacities.assign(
                info->pRopeCapacities,
                info->pRopeCapacities + info->ropeCount);
        if (info->softBodyCount != 0U)
            created->soft_body_capacities.assign(
                info->pSoftBodyCapacities,
                info->pSoftBodyCapacities + info->softBodyCount);
        if (info->smokeCount != 0U)
            created->smoke_capacities.assign(
                info->pSmokeCapacities,
                info->pSmokeCapacities + info->smokeCount);
        if (info->paintFieldCount != 0U)
            created->paint_field_capacities.assign(
                info->pPaintFieldCapacities,
                info->pPaintFieldCapacities + info->paintFieldCount);
        created->rigid_bodies.resize(info->rigidBodyCapacity);
        created->rigid_body_geometries.resize(info->rigidBodyCapacity);
        created->rigid_dense_indices.resize(info->rigidBodyCapacity,
                                            PM_INVALID_INDEX);
        created->constraints.resize(info->constraintCapacity);
        created->fluid_ids.resize(info->fluidCount);
        created->cloths.resize(info->clothCount);
        created->ropes.resize(info->ropeCount);
        created->soft_bodies.resize(info->softBodyCount);
        created->smokes.resize(info->smokeCount);
        created->emitters.resize(info->emitterCapacity);
        created->destroyers.resize(info->destroyerCapacity);
        created->paint_fields.resize(info->paintFieldCount);
        created->paint_field_geometries.resize(info->paintFieldCount);
        created->paint_rules.resize(info->paintRuleCapacity);
        created->runtime = std::make_unique<parallel_mater::SimulationRuntime>();
        created->info.pFluidCapacities = created->fluids.data();
        created->info.pClothCapacities = created->cloth_capacities.data();
        created->info.pRopeCapacities = created->rope_capacities.data();
        created->info.pSoftBodyCapacities =
            created->soft_body_capacities.data();
        created->info.pSmokeCapacities = created->smoke_capacities.data();
        created->info.pPaintFieldCapacities =
            created->paint_field_capacities.data();
    } catch (const std::bad_alloc &) {
        parallel_mater::api::destroy_object(device, created);
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    created->simulation_options = simulation_options(*info);
    *simulation = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroySimulation(
    PmDevice device, PmSimulation simulation) {
    if (device == nullptr || simulation == nullptr || simulation->device != device) return;
    (void)pmDeviceWaitIdle(device);
    {
        parallel_mater::simulation::ScopedMemoryArena scope(
            simulation->arena.get());
        simulation->runtime.reset();
    }
    simulation->arena.reset();
    parallel_mater::api::destroy_object(device, simulation);
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetSimulationMemoryRequirements(
    PmDevice device, PmSimulation simulation,
    PmSimulationMemoryRequirements *requirements) {
    if (requirements == nullptr) return;
    *requirements = {};
    if (device == nullptr || simulation == nullptr || simulation->device != device)
        return;
    *requirements = simulation->requirements;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmBindSimulationMemory(
    PmDevice device, PmSimulation simulation,
    const PmSimulationMemoryBinding *binding) {
    if (device == nullptr || simulation == nullptr || binding == nullptr ||
        simulation->device != device || binding->persistentMemory == nullptr ||
        binding->scratchMemory == nullptr ||
        binding->persistentMemory->device != device ||
        binding->scratchMemory->device != device ||
        binding->persistentOffset % 256U != 0U ||
        binding->persistentOffset % simulation->requirements.persistent.alignment != 0U ||
        binding->scratchOffset % simulation->requirements.scratch.alignment != 0U ||
        (simulation->requirements.persistent.memoryTypeBits &
            (1U << binding->persistentMemory->memory_type)) == 0U ||
        (simulation->requirements.scratch.memoryTypeBits &
            (1U << binding->scratchMemory->memory_type)) == 0U ||
        !parallel_mater::api::range_fits(
            binding->persistentOffset,
            simulation->requirements.persistent.size,
            binding->persistentMemory->size) ||
        !parallel_mater::api::range_fits(
            binding->scratchOffset, simulation->requirements.scratch.size,
            binding->scratchMemory->size))
        return PM_ERROR_INVALID_ARGUMENT;
    if (binding->persistentMemory == binding->scratchMemory) {
        const PmDeviceSize persistent_end = binding->persistentOffset +
            simulation->requirements.persistent.size;
        const PmDeviceSize scratch_end = binding->scratchOffset +
            simulation->requirements.scratch.size;
        if (binding->persistentOffset < scratch_end &&
            binding->scratchOffset < persistent_end)
            return PM_ERROR_INVALID_ARGUMENT;
    }
    if (simulation->binding.persistentMemory != nullptr)
        return PM_ERROR_INVALID_STATE;
    const PmResult current = device->services->activate();
    if (current != PM_SUCCESS) return current;
    PmDomainMemory domain_memory;
    try {
        using Plan = parallel_mater::simulation::StorageRequirements;
        struct Region { std::vector<std::unique_ptr<PmDomainMemory::Arena>> *owners; Plan plan; };
        std::vector<Region> regions;
        PmDeviceSize domain_persistent = 0, domain_scratch = 0;
        const auto append = [&](auto &owners, Plan plan) {
            domain_persistent += plan.persistent;
            domain_scratch += plan.scratch;
            regions.push_back({&owners, plan});
        };
        const auto rigid = simulation->simulation_options.rigid_body_capacity;
        for (const auto &capacity : simulation->fluids) {
            Plan plan{};
            if (!parallel_mater::SimulationRuntime::fluid_memory_requirements(capacity.particleCapacity, plan))
                return PM_ERROR_INVALID_ARGUMENT;
            append(domain_memory.fluids, plan);
        }
        for (const auto &capacity : simulation->cloth_capacities) {
            Plan plan{};
            if (!parallel_mater::SimulationRuntime::cloth_memory_requirements(capacity.vertexCapacity,
                    capacity.triangleIndexCapacity, capacity.bondCapacity, rigid, plan))
                return PM_ERROR_INVALID_ARGUMENT;
            append(domain_memory.cloths, plan);
        }
        for (const auto &capacity : simulation->rope_capacities) {
            Plan plan{};
            if (!parallel_mater::SimulationRuntime::rope_memory_requirements(capacity.nodeCapacity, rigid, plan))
                return PM_ERROR_INVALID_ARGUMENT;
            append(domain_memory.ropes, plan);
        }
        for (const auto &capacity : simulation->soft_body_capacities) {
            Plan plan{};
            if (!parallel_mater::SimulationRuntime::soft_body_memory_requirements(capacity.nodeCapacity,
                    capacity.bondCapacity, capacity.surfaceVertexCapacity, capacity.surfaceTriangleIndexCapacity, plan))
                return PM_ERROR_INVALID_ARGUMENT;
            append(domain_memory.soft_bodies, plan);
        }
        for (const auto &capacity : simulation->smoke_capacities) {
            Plan plan{};
            if (!parallel_mater::SimulationRuntime::smoke_memory_requirements(capacity.particleCapacity,
                    capacity.gridCellCapacity, plan)) return PM_ERROR_INVALID_ARGUMENT;
            append(domain_memory.smokes, plan);
        }
        const auto global_persistent = simulation->requirements.persistent.size - domain_persistent;
        const auto global_scratch = simulation->requirements.scratch.size - domain_scratch;
        auto *persistent = static_cast<std::byte *>(binding->persistentMemory->allocation) + binding->persistentOffset;
        auto *scratch = static_cast<std::byte *>(binding->scratchMemory->allocation) + binding->scratchOffset;
        auto *domain_persistent_begin = persistent + global_persistent;
        auto *domain_scratch_begin = scratch + global_scratch;
        // Disjoint per-slot pools keep resizing one population from fragmenting
        // the space promised to another. Global and child arenas never overlap.
        for (const auto &region : regions) {
            region.owners->push_back(std::make_unique<PmDomainMemory::Arena>(
                domain_persistent_begin, region.plan.persistent,
                domain_scratch_begin, region.plan.scratch, device->services.get()));
            domain_persistent_begin += region.plan.persistent;
            domain_scratch_begin += region.plan.scratch;
        }
        simulation->arena = std::make_unique<PmDomainMemory::Arena>(
            persistent, static_cast<std::size_t>(global_persistent),
            scratch, static_cast<std::size_t>(global_scratch), device->services.get());
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    parallel_mater::SimulationRuntime initialized;
    PmResult result = PM_SUCCESS;
    {
        parallel_mater::simulation::ScopedMemoryArena scope(
            simulation->arena.get());
        result = parallel_mater::api::from_status(
            parallel_mater::SimulationRuntime::create(
                simulation->simulation_options, initialized,
                parallel_mater::device::execution_queue(*device->services)));
    }
    if (result != PM_SUCCESS) {
        simulation->arena.reset();
        return result;
    }
    *simulation->runtime = std::move(initialized);
    simulation->domain_memory = std::move(domain_memory);
    simulation->binding = *binding;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWriteRigidBodies(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count, const PmRigidBodyInfo *bodies) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || bodies == nullptr || first > simulation->rigid_bodies.size() ||
        count > simulation->rigid_bodies.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmRigidBodyInfo> copied;
    try { copied.assign(bodies, bodies + count); }
    catch (const std::bad_alloc &) { return PM_ERROR_OUT_OF_HOST_MEMORY; }
    for (const auto &body : copied) {
        if (body.active != PM_FALSE &&
            (body.geometry == nullptr || !finite(body.state) ||
             body.motionType > PM_MOTION_DYNAMIC))
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, copied = std::move(copied)](
            parallel_mater::device::Services &services) noexcept {
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            PmResult result = validate_bound(simulation);
            if (result != PM_SUCCESS) return result;
            const auto detached = parallel_mater::api::clear_couplings(simulation);
            if (detached != PM_SUCCESS) return detached;
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                auto &slot = simulation->rigid_bodies[first + offset];
                auto &geometry_slot =
                    simulation->rigid_body_geometries[first + offset];
                const PmGeometry retired_geometry = geometry_slot;
                if (slot.has_value()) {
                    result = parallel_mater::api::from_status(
                        simulation->runtime->remove_rigid_body(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                geometry_slot = nullptr;
                const PmRigidBodyInfo &source = copied[offset];
                if (source.active == PM_FALSE) {
                    result = parallel_mater::api::retire_geometry_if_unused(
                        simulation, retired_geometry);
                    if (result != PM_SUCCESS) return result;
                    continue;
                }
                TriangleMeshId mesh{};
                result = parallel_mater::api::register_geometry(
                    simulation, source.geometry, services, mesh);
                if (result != PM_SUCCESS) {
                    const PmResult cleanup =
                        parallel_mater::api::retire_geometry_if_unused(
                            simulation, retired_geometry);
                    return cleanup == PM_SUCCESS ? result : cleanup;
                }
                RigidBodyOptions options{};
                options.motion = motion_type(source.motionType);
                options.mesh = mesh;
                options.initial_state = rigid_state(source.state);
                options.mass = source.mass;
                options.inertia_diagonal = vector3(source.inertiaDiagonal);
                options.friction = source.friction;
                options.restitution = source.restitution;
                options.linear_damping = source.linearDamping;
                options.angular_damping = source.angularDamping;
                options.maximum_linear_speed = source.maximumLinearSpeed;
                options.maximum_angular_speed = source.maximumAngularSpeed;
                options.collision_margin = source.collisionMargin;
                options.user_data = source.userData;
                RigidBodyId created{};
                result = parallel_mater::api::from_status(
                    simulation->runtime->add_rigid_body(options, created));
                if (result != PM_SUCCESS) {
                    PmResult cleanup =
                        parallel_mater::api::retire_geometry_if_unused(
                            simulation, source.geometry);
                    if (cleanup == PM_SUCCESS)
                        cleanup = parallel_mater::api::retire_geometry_if_unused(
                            simulation, retired_geometry);
                    return cleanup == PM_SUCCESS ? result : cleanup;
                }
                slot = created;
                geometry_slot = source.geometry;
                result = parallel_mater::api::retire_geometry_if_unused(
                    simulation, retired_geometry);
                if (result != PM_SUCCESS) return result;
            }
            simulation->rigid_dense_mapping_dirty = true;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdApplyRigidInputs(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t count, const PmRigidInput *inputs) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || inputs == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmRigidInput> copied;
    try { copied.assign(inputs, inputs + count); }
    catch (const std::bad_alloc &) { return PM_ERROR_OUT_OF_HOST_MEMORY; }
    for (const auto &input : copied)
        if (input.bodyIndex >= simulation->rigid_bodies.size())
            return PM_ERROR_INVALID_ARGUMENT;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, copied = std::move(copied)](
                            parallel_mater::device::Services &) noexcept {
            for (const auto &input : copied) {
                if (!simulation->rigid_bodies[input.bodyIndex].has_value())
                    return PM_ERROR_INVALID_STATE;
                const RigidBodyId id = *simulation->rigid_bodies[input.bodyIndex];
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->apply_rigid_input(
                        id, vector3(input.force), vector3(input.torque),
                        vector3(input.linearImpulse),
                        vector3(input.angularImpulse)));
                if (result != PM_SUCCESS) return result;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdSetRigidStates(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count, const PmRigidState *states) {
    const PmResult record_state = parallel_mater::api::require_recording(command_buffer);
    if (record_state != PM_SUCCESS) return record_state;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || states == nullptr || first > simulation->rigid_bodies.size() ||
        count > simulation->rigid_bodies.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmRigidState> copied;
    try { copied.assign(states, states + count); }
    catch (const std::bad_alloc &) { return PM_ERROR_OUT_OF_HOST_MEMORY; }
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, copied = std::move(copied)](
                            parallel_mater::device::Services &) noexcept {
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                const auto &slot = simulation->rigid_bodies[first + offset];
                if (!slot.has_value()) return PM_ERROR_INVALID_STATE;
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->set_rigid_body_state(
                        *slot, rigid_state(copied[offset])));
                if (result != PM_SUCCESS) return result;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdSetKinematicTargets(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count, const PmRigidState *targets) {
    const PmResult record_state =
        parallel_mater::api::require_recording(command_buffer);
    if (record_state != PM_SUCCESS) return record_state;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || targets == nullptr ||
        first > simulation->rigid_bodies.size() ||
        count > simulation->rigid_bodies.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmRigidState> copied;
    try { copied.assign(targets, targets + count); }
    catch (const std::bad_alloc &) { return PM_ERROR_OUT_OF_HOST_MEMORY; }
    for (const PmRigidState &target : copied)
        if (!finite(target)) return PM_ERROR_INVALID_ARGUMENT;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, copied = std::move(copied)](
                            parallel_mater::device::Services &) noexcept {
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                const auto &slot = simulation->rigid_bodies[first + offset];
                if (!slot.has_value()) return PM_ERROR_INVALID_STATE;
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->set_kinematic_target(
                        *slot, rigid_state(copied[offset])));
                if (result != PM_SUCCESS) return result;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWriteRigidConstraints(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count,
    const PmRigidConstraintInfo *constraints) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || constraints == nullptr ||
        first > simulation->constraints.size() ||
        count > simulation->constraints.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmRigidConstraintInfo> copied;
    try { copied.assign(constraints, constraints + count); }
    catch (const std::bad_alloc &) { return PM_ERROR_OUT_OF_HOST_MEMORY; }
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, copied = std::move(copied)](
                            parallel_mater::device::Services &) noexcept {
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                auto &slot = simulation->constraints[first + offset];
                if (slot.has_value()) {
                    const PmResult result = parallel_mater::api::from_status(
                        simulation->runtime->remove_rigid_constraint(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const auto &source = copied[offset];
                if (source.active == PM_FALSE) continue;
                if (source.bodyA >= simulation->rigid_bodies.size() ||
                    source.bodyB == PM_INVALID_INDEX ||
                    source.bodyB >= simulation->rigid_bodies.size() ||
                    !simulation->rigid_bodies[source.bodyA].has_value() ||
                    !simulation->rigid_bodies[source.bodyB].has_value())
                    return PM_ERROR_INVALID_ARGUMENT;
                RigidConstraintOptions options{};
                options.type = constraint_type(source.type);
                options.body_a = *simulation->rigid_bodies[source.bodyA];
                options.body_b = *simulation->rigid_bodies[source.bodyB];
                options.local_anchor_a = vector3(source.frameA.position);
                options.local_anchor_b = vector3(source.frameB.position);
                options.local_orientation_a = quaternion(source.frameA.orientation);
                options.local_orientation_b = quaternion(source.frameB.orientation);
                convert_axes(source.linearLimits, options.linear_limits);
                convert_axes(source.angularLimits, options.angular_limits);
                convert_axes(source.linearSprings, options.linear_springs);
                convert_axes(source.angularSprings, options.angular_springs);
                options.motor.linear_enabled = source.linearMotors[0].enabled == PM_TRUE;
                options.motor.linear_target_velocity =
                    source.linearMotors[0].targetVelocity;
                options.motor.linear_maximum_impulse =
                    source.linearMotors[0].maximumForce;
                options.motor.angular_enabled = source.angularMotors[0].enabled == PM_TRUE;
                options.motor.angular_target_velocity =
                    source.angularMotors[0].targetVelocity;
                options.motor.angular_maximum_impulse =
                    source.angularMotors[0].maximumForce;
                options.enabled = true;
                options.disable_collisions = source.collideConnected == PM_FALSE;
                options.breaking_impulse_threshold =
                    source.breakingImpulseThreshold;
                options.solver_iterations = source.solverIterations;
                RigidConstraintId created{};
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->add_rigid_constraint(options, created));
                if (result != PM_SUCCESS) return result;
                slot = created;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdResetSimulation(
    PmCommandBuffer command_buffer, PmSimulation simulation) {
    if (simulation == nullptr || simulation->device != command_buffer->device)
        return PM_ERROR_INVALID_ARGUMENT;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation](
                            parallel_mater::device::Services &services) noexcept {
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            PmResult result = validate_bound(simulation);
            if (result != PM_SUCCESS) return result;
            {
                parallel_mater::simulation::ScopedMemoryArena arena_scope(
                    simulation->arena.get());
                *simulation->runtime = parallel_mater::SimulationRuntime{};
            }
            // Runtime destruction releases its own geometry allocations. A
            // geometry arena may also contain another simulation's live mesh.
            simulation->arena->reset();
            parallel_mater::SimulationRuntime initialized;
            {
                parallel_mater::simulation::ScopedMemoryArena arena_scope(
                    simulation->arena.get());
                result = parallel_mater::api::from_status(
                    parallel_mater::SimulationRuntime::create(
                        simulation->simulation_options, initialized, queue));
            }
            if (result != PM_SUCCESS) {
                simulation->execution_result = result;
                return result;
            }
            *simulation->runtime = std::move(initialized);
            simulation->geometries.clear();
            for (auto &slot : simulation->rigid_bodies) slot.reset();
            std::fill(simulation->rigid_body_geometries.begin(),
                      simulation->rigid_body_geometries.end(), nullptr);
            simulation->rigid_dense_ids.clear();
            std::fill(simulation->rigid_dense_indices.begin(),
                      simulation->rigid_dense_indices.end(), PM_INVALID_INDEX);
            simulation->rigid_dense_mapping_dirty = true;
            for (auto &slot : simulation->constraints) slot.reset();
            for (auto &slot : simulation->fluid_ids) slot.reset();
            for (auto &slot : simulation->cloths) slot.reset();
            for (auto &slot : simulation->ropes) slot.reset();
            for (auto &slot : simulation->soft_bodies) slot.reset();
            for (auto &slot : simulation->smokes) slot.reset();
            for (auto &slot : simulation->emitters) slot.reset();
            for (auto &slot : simulation->destroyers) slot.reset();
            for (auto &slot : simulation->paint_fields) slot.reset();
            std::fill(simulation->paint_field_geometries.begin(),
                      simulation->paint_field_geometries.end(), nullptr);
            for (auto &slot : simulation->paint_rules) slot.reset();
            simulation->requested_couplings.clear();
            simulation->explicit_couplings = false;
            simulation->active_couplings.clear();
            simulation->coupling_pipeline_serial = 0;
            simulation->couplings_initialized = false;
            simulation->submitted_steps = 0U;
            simulation->last_substeps = 0U;
            simulation->execution_result = PM_SUCCESS;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdSetFluidParticles(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t fluid, const PmBufferView *particles) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        particles == nullptr || fluid >= simulation->fluids.size() ||
        command_buffer->bound_pipeline == nullptr ||
        fluid >= command_buffer->bound_pipeline->fluids.size() ||
        particles->count > simulation->fluids[fluid].particleCapacity)
        return PM_ERROR_INVALID_ARGUMENT;
    if (particles->count != 0U &&
        (particles->format != PM_FORMAT_PARTICLE_STATE ||
         particles->stride != sizeof(PmParticleState) ||
         particles->range.buffer == nullptr ||
         particles->range.buffer->device != command_buffer->device ||
         (particles->range.buffer->usage & PM_BUFFER_PARTICLE_INPUT_BIT) == 0U ||
         static_cast<PmDeviceSize>(particles->count) * particles->stride > particles->range.size ||
         !parallel_mater::api::range_fits(particles->range.offset, particles->range.size,
                                         particles->range.buffer->size)))
        return PM_ERROR_INVALID_ARGUMENT;
    const PmBufferView copied = *particles;
    const PmLiquidMaterialState material =
        command_buffer->bound_pipeline->fluids[fluid];
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, fluid, copied, material](
                            parallel_mater::device::Services &services) noexcept {
            const PmResult ownership = copied.count == 0U ? PM_SUCCESS :
                parallel_mater::api::require_device_ownership(copied.range.buffer);
            if (ownership != PM_SUCCESS) return ownership;
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            PmResult result = validate_bound(simulation);
            if (result != PM_SUCCESS) return result;
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->domain_memory.fluids[fluid].get());
            const auto detached = parallel_mater::api::clear_couplings(simulation);
            if (detached != PM_SUCCESS) return detached;
            auto &slot = simulation->fluid_ids[fluid];
            if (slot.has_value()) {
                result = parallel_mater::api::from_status(
                    simulation->runtime->remove_fluid(*slot, queue));
                if (result != PM_SUCCESS) return result;
                slot.reset();
            }
            parallel_mater::FluidOptions options{};
            options.capacity = simulation->fluids[fluid].particleCapacity;
            options.maximum_neighbors =
                simulation->fluids[fluid].neighborCapacityPerParticle;
            options.particle_radius = material.particleRadius;
            options.rest_density = material.restDensity;
            options.support_radius = material.supportRadius;
            options.rest_particle_volume = material.restParticleVolume;
            options.repulsion = material.pressureStiffness;
            options.viscosity = material.kinematicViscosity;
            options.normal_damping = material.normalDamping;
            options.velocity_damping = material.velocityDamping;
            options.maximum_speed = material.maximumSpeed;
            options.maximum_pair_acceleration =
                material.maximumPairAcceleration;
            options.solver_iterations = material.densityIterations;
            static_assert(sizeof(PmParticleState) ==
                          sizeof(parallel_mater::FluidParticle));
            const auto *source = static_cast<const parallel_mater::FluidParticle *>(
                copied.count == 0U ? nullptr :
                    parallel_mater::api::buffer_pointer(copied.range.buffer, copied.range.offset));
            parallel_mater::FluidId created{};
            result = parallel_mater::api::from_status(
                simulation->runtime->add_fluid(
                    options, {source, copied.count}, created, queue));
            if (result != PM_SUCCESS) return result;
            slot = created;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdStepSimulation(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    const PmStepInfo *step) {
    const PmResult record_state = parallel_mater::api::require_recording(command_buffer);
    if (record_state != PM_SUCCESS) return record_state;
    if (simulation == nullptr || step == nullptr || step->timestep <= 0.0F ||
        step->substeps == 0U || simulation->device != command_buffer->device ||
        command_buffer->bound_pipeline == nullptr ||
        (step->flags & ~PM_STEP_COLLECT_CONTACT_EVENTS_BIT) != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    const PmStepInfo copied = *step;
    const PmPhysicsPipeline pipeline = command_buffer->bound_pipeline;
    if (command_buffer->statistics_open)
        command_buffer->statistics_simulation = simulation;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, copied, pipeline](
                            parallel_mater::device::Services &services) noexcept {
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            PmResult result = validate_bound(simulation);
            if (result != PM_SUCCESS) return result;
            result = parallel_mater::api::initialize_couplings(simulation,
                                                                pipeline);
            if (result != PM_SUCCESS) return result;
            parallel_mater::StepOptions options{};
            options.timestep = copied.timestep;
            options.substeps = copied.substeps;
            options.gravity = vector3(copied.gravity);
            options.rigid_contact_iterations = pipeline->rigid.rigidIterations;
            const bool collect_contacts =
                (copied.flags & PM_STEP_COLLECT_CONTACT_EVENTS_BIT) != 0U;
            options.collect_rigid_contacts = collect_contacts;
            options.collect_fluid_contacts = collect_contacts;
            // This closure runs on the ordered submission worker. Retain the
            // frame through GPU completion and deferred overflow validation.
            parallel_mater::FrameToken frame;
            result = parallel_mater::api::from_status(
                simulation->runtime->step_async(options, frame, queue));
            if (result == PM_SUCCESS)
                result = parallel_mater::api::from_status(frame.wait());
            simulation->execution_result = result;
            if (result == PM_SUCCESS) {
                ++simulation->submitted_steps;
                simulation->last_substeps = copied.substeps;
            }
            return result;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyRigidStates(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count,
    const PmBufferRange *destination) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (simulation == nullptr || destination == nullptr ||
        simulation->device != command_buffer->device ||
        destination->buffer == nullptr ||
        destination->buffer->device != command_buffer->device ||
        first > simulation->rigid_bodies.size() ||
        count > simulation->rigid_bodies.size() - first ||
        destination->size < static_cast<PmDeviceSize>(count) *
                                sizeof(PmRigidState) ||
        !parallel_mater::api::range_fits(destination->offset,
                                         destination->size,
                                         destination->buffer->size))
        return PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, count, copied](
                            parallel_mater::device::Services &services) noexcept {
            const PmResult ownership =
                parallel_mater::api::require_device_ownership(copied.buffer);
            if (ownership != PM_SUCCESS) return ownership;
            parallel_mater::RigidBodyDeviceView view{};
            PmResult result = parallel_mater::api::from_status(
                simulation->runtime->rigid_body_view(view));
            if (result != PM_SUCCESS) return result;
            if (view.ids.size != view.states.size)
                return PM_ERROR_INVALID_STATE;
            if (simulation->rigid_dense_mapping_dirty) {
                try {
                    simulation->rigid_dense_ids.resize(view.ids.size);
                    simulation->rigid_dense_indices.assign(
                        simulation->rigid_bodies.size(), PM_INVALID_INDEX);
                } catch (const std::bad_alloc &) {
                    return PM_ERROR_OUT_OF_HOST_MEMORY;
                }
                auto &dense_ids = simulation->rigid_dense_ids;
                if (!dense_ids.empty()) {
                    const PmResult copied_ids = services.copy_and_wait(
                        dense_ids.data(), view.ids.data,
                        static_cast<PmDeviceSize>(dense_ids.size()) *
                            sizeof(dense_ids[0]),
                        parallel_mater::device::CopyDirection::device_to_host);
                    if (copied_ids != PM_SUCCESS) return copied_ids;
                }
                for (std::size_t slot_index = 0U;
                     slot_index < simulation->rigid_bodies.size();
                     ++slot_index) {
                    const auto &slot = simulation->rigid_bodies[slot_index];
                    if (!slot.has_value()) continue;
                    const auto dense = std::find(dense_ids.begin(),
                                                 dense_ids.end(), *slot);
                    if (dense == dense_ids.end())
                        return PM_ERROR_INVALID_STATE;
                    simulation->rigid_dense_indices[slot_index] =
                        static_cast<std::uint32_t>(dense - dense_ids.begin());
                }
                simulation->rigid_dense_mapping_dirty = false;
            }
            static_assert(sizeof(PmRigidState) == sizeof(RigidBodyState));
            auto *target = static_cast<std::byte *>(
                parallel_mater::api::buffer_pointer(copied.buffer,
                                                     copied.offset));
            for (std::uint32_t offset = 0U; offset < count;) {
                const std::uint32_t dense =
                    simulation->rigid_dense_indices[first + offset];
                if (dense == PM_INVALID_INDEX || dense >= view.states.size)
                    return PM_ERROR_INVALID_STATE;
                std::uint32_t run = 1U;
                while (offset + run < count &&
                       simulation->rigid_dense_indices[first + offset + run] ==
                           dense + run)
                    ++run;
                const PmResult state_copy = services.copy(
                    target + static_cast<std::size_t>(offset) *
                                 sizeof(PmRigidState),
                    view.states.data + dense,
                    static_cast<PmDeviceSize>(run) * sizeof(PmRigidState),
                    parallel_mater::device::CopyDirection::device_to_device);
                if (state_copy != PM_SUCCESS) return state_copy;
                offset += run;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyFluidParticles(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t fluid, const PmBufferRange *destination) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        fluid >= simulation->fluid_ids.size() ||
        destination == nullptr || destination->buffer == nullptr ||
        destination->buffer->device != command_buffer->device ||
        destination->size < sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(
                simulation->fluids[fluid].particleCapacity) *
                sizeof(PmFluidParticleOutput) ||
        !parallel_mater::api::range_fits(destination->offset, destination->size,
                                         destination->buffer->size))
        return PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, fluid, copied](
                            parallel_mater::device::Services &services) noexcept {
            const PmResult ownership =
                parallel_mater::api::require_device_ownership(copied.buffer);
            if (ownership != PM_SUCCESS) return ownership;
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            if (!simulation->fluid_ids[fluid].has_value())
                return PM_ERROR_INVALID_STATE;
            parallel_mater::FluidDeviceView view{};
            PmResult result = parallel_mater::api::from_status(
                simulation->runtime->fluid_view(*simulation->fluid_ids[fluid], view));
            if (result != PM_SUCCESS) return result;
            auto *target = static_cast<std::byte *>(
                parallel_mater::api::buffer_pointer(copied.buffer, copied.offset));
            return parallel_mater::simulation::export_fluid_particles(
                queue, view.positions.data, view.velocities.data,
                view.accelerations.data, view.temperatures.data, view.foam.data,
                view.stable_particle_ids.data, view.particle_count,
                simulation->fluids[fluid].particleCapacity, view.revision,
                reinterpret_cast<PmDomainOutputHeader *>(target),
                reinterpret_cast<PmFluidParticleOutput *>(
                    target + sizeof(PmDomainOutputHeader)));
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetSimulationStatus(
    PmDevice device, PmSimulation simulation, PmResult *execution) {
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *execution = PM_ERROR_FEATURE_NOT_PRESENT;
    if (device == nullptr || simulation == nullptr || simulation->device != device)
        return PM_ERROR_INVALID_ARGUMENT;
    if (simulation->pending_submissions.load(std::memory_order_acquire) != 0U)
        return PM_NOT_READY;
    *execution = simulation->execution_result.load();
    return PM_SUCCESS;
}
