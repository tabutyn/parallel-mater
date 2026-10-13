// SPDX-License-Identifier: MIT
#include "internal.hpp"
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

[[nodiscard]] bool simulation_requirements(
    const PmSimulationCreateInfo &info,
    PmSimulationMemoryRequirements &output) noexcept {
    PmDeviceSize persistent = 65536U;
    PmDeviceSize scratch = 4096U;
    const PmDeviceSize rigid = info.rigidBodyCapacity;
    const PmDeviceSize pairs = rigid * rigid;
    const PmDeviceSize manifolds = rigid > 1U ? rigid * (rigid - 1U) / 2U : 0U;
    const PmDeviceSize requested_leaf_pair_slots = std::max<PmDeviceSize>(
        parallel_mater::simulation::k_minimum_leaf_pair_cache_slots,
        rigid * parallel_mater::simulation::k_leaf_pair_cache_slots_per_body);
    const PmDeviceSize leaf_pair_slots =
        std::min(manifolds, requested_leaf_pair_slots);
    const PmDeviceSize leaf_pairs_per_slot =
        rigid <= parallel_mater::simulation::k_small_rigid_leaf_body_capacity
            ? parallel_mater::simulation::k_small_rigid_leaf_pair_capacity
            : parallel_mater::simulation::k_max_leaf_pairs_per_body_pair;
    if (leaf_pair_slots != 0U &&
        leaf_pairs_per_slot >
            std::numeric_limits<PmDeviceSize>::max() / leaf_pair_slots)
        return false;
    const PmDeviceSize leaf_pairs = leaf_pair_slots * leaf_pairs_per_slot;
    if (!add_bytes(persistent, rigid, 4096U) ||
        !add_bytes(persistent, pairs, 16U) ||
        !add_bytes(persistent, manifolds,
                   sizeof(parallel_mater::simulation::ContactManifold)) ||
        !add_bytes(persistent, leaf_pair_slots,
                   sizeof(parallel_mater::simulation::CachedContactPair)) ||
        !add_bytes(persistent, leaf_pairs,
                   sizeof(parallel_mater::simulation::LeafPair)) ||
        (rigid <= parallel_mater::simulation::k_small_rigid_leaf_body_capacity &&
         !add_bytes(persistent, leaf_pairs,
                    sizeof(parallel_mater::simulation::ContactManifold))) ||
        !add_bytes(persistent, std::max(1U, info.constraintCapacity),
                   sizeof(parallel_mater::simulation::RigidConstraintResource)) ||
        !add_bytes(persistent, info.contactCapacity, 128U) ||
        !add_bytes(scratch, pairs, 16U) ||
        !add_bytes(scratch, rigid, 2048U))
        return false;
    for (const PmFluidCapacity capacity :
         std::span(info.pFluidCapacities, info.fluidCount)) {
        if (!add_bytes(persistent, capacity.particleCapacity, 512U) ||
            !add_bytes(persistent,
                static_cast<PmDeviceSize>(capacity.particleCapacity) *
                    capacity.neighborCapacityPerParticle, 4U) ||
            !add_bytes(scratch, capacity.particleCapacity, 128U))
            return false;
    }
    for (const PmClothCapacity capacity :
         std::span(info.pClothCapacities, info.clothCount)) {
        if (!add_bytes(persistent, capacity.vertexCapacity, 768U) ||
            !add_bytes(persistent, capacity.triangleIndexCapacity, 32U) ||
            !add_bytes(persistent, capacity.bondCapacity, 96U) ||
            !add_bytes(scratch, capacity.vertexCapacity, 192U))
            return false;
    }
    for (const PmRopeCapacity capacity :
         std::span(info.pRopeCapacities, info.ropeCount)) {
        if (!add_bytes(persistent, capacity.nodeCapacity, 512U) ||
            !add_bytes(scratch, capacity.nodeCapacity, 128U))
            return false;
    }
    for (const PmSoftBodyCapacity capacity :
         std::span(info.pSoftBodyCapacities, info.softBodyCount)) {
        if (!add_bytes(persistent, capacity.nodeCapacity, 1024U) ||
            !add_bytes(persistent, capacity.bondCapacity, 128U) ||
            !add_bytes(persistent, capacity.surfaceVertexCapacity, 320U) ||
            !add_bytes(persistent, capacity.surfaceTriangleIndexCapacity, 32U) ||
            !add_bytes(scratch, capacity.nodeCapacity, 256U))
            return false;
    }
    for (const PmSmokeCapacity capacity :
         std::span(info.pSmokeCapacities, info.smokeCount)) {
        if (!add_bytes(persistent, capacity.particleCapacity, 320U) ||
            !add_bytes(persistent, capacity.gridCellCapacity, 320U) ||
            !add_bytes(scratch, capacity.particleCapacity, 128U) ||
            !add_bytes(scratch, capacity.gridCellCapacity, 96U))
            return false;
    }
    if (!add_bytes(persistent, info.couplingCapacity, 4096U) ||
        !add_bytes(persistent, info.emitterCapacity, 2048U) ||
        !add_bytes(persistent, info.destroyerCapacity, 512U) ||
        !add_bytes(persistent, info.paintRuleCapacity, 512U) ||
        !add_bytes(persistent, info.debugFrameCapacity,
                   std::max<PmDeviceSize>(1U, rigid) * 192U))
        return false;
    for (const PmPaintFieldCapacity capacity :
         std::span(info.pPaintFieldCapacities, info.paintFieldCount)) {
        if (!add_bytes(persistent, capacity.vertexCapacity, sizeof(PmFloat2)) ||
            !add_bytes(persistent,
                static_cast<PmDeviceSize>(capacity.width) * capacity.height,
                sizeof(std::uint32_t)))
            return false;
    }
    constexpr PmDeviceSize alignment = 256U;
    persistent = (persistent + alignment - 1U) & ~(alignment - 1U);
    scratch = (scratch + alignment - 1U) & ~(alignment - 1U);
    output.persistent = {persistent, alignment, 0x2U, PM_FALSE};
    output.scratch = {scratch, alignment, 0x2U, PM_FALSE};
    return true;
}

[[nodiscard]] PmResult register_geometry_impl(
    PmSimulation simulation, PmGeometry geometry, cudaStream_t stream,
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
        {vertex_data, vertices.count}, {index_data, indices.count}, created, stream);
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
                           cudaStream_t stream,
                           TriangleMeshId &output) noexcept {
    return register_geometry_impl(simulation, geometry, stream, output);
}

} // namespace parallel_mater::api

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateGeometry(
    PmDevice device, const PmGeometryCreateInfo *info, PmGeometry *geometry) {
    if (geometry == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *geometry = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO, info->pNext) ||
        info->queueFamilyIndex != 0U || info->vertices.range.buffer == nullptr ||
        info->indices.range.buffer == nullptr || info->vertices.count == 0U ||
        info->indices.count == 0U || info->indices.count % 3U != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmGeometry_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->info = *info;
    *geometry = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyGeometry(
    PmDevice device, PmGeometry geometry) {
    if (device == nullptr || geometry == nullptr || geometry->device != device) return;
    parallel_mater::api::destroy_object(device, geometry);
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetGeometryMemoryRequirements(
    PmDevice device, PmGeometry geometry, PmMemoryRequirements *requirements) {
    if (requirements == nullptr) return;
    *requirements = {};
    if (device == nullptr || geometry == nullptr || geometry->device != device) return;
    const PmDeviceSize triangles = geometry->info.indices.count / 3U;
    const PmDeviceSize vertices = geometry->info.vertices.count;
    // Vertices, reordered indices, a binary BVH, leaf table, closed-solid
    // planes, unique planes, shell normals, and alignment padding all live in
    // the geometry allocation. The conservative bound intentionally leaves
    // room for every triangle to contribute one unique plane.
    requirements->size =
        (vertices * 16U + triangles * 192U + 4095U) & ~PmDeviceSize{255U};
    requirements->alignment = 256U;
    // CUDA's current CPU BVH preparation reads the bound storage after the
    // input transfer. Advertise only coherent host-visible memory until that
    // preparation becomes a GPU build command; never accept device-local
    // memory and then dereference it on the host.
    requirements->memoryTypeBits = 0x2U;
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
                static_cast<std::size_t>(requirements.size), nullptr, 0U);
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
        command_buffer, [geometry](cudaStream_t) noexcept {
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
    parallel_mater::WorldOptions options{};
    options.rigid_body_capacity = std::max(1U, info->rigidBodyCapacity);
    options.rigid_constraint_capacity = std::max(1U, info->constraintCapacity);
    options.triangle_mesh_capacity = std::max(16U, info->rigidBodyCapacity);
    options.contact_capacity = std::max(1U, info->contactCapacity);
    options.fluid_capacity = std::max(1U, info->fluidCount);
    options.cloth_capacity = std::max(1U, info->clothCount);
    options.rope_capacity = std::max(1U, info->ropeCount);
    options.soft_body_capacity = std::max(1U, info->softBodyCount);
    options.smoke_capacity = std::max(1U, info->smokeCount);
    options.particle_source_capacity = std::max(1U, info->emitterCapacity);
    options.particle_destroy_plane_capacity = std::max(1U, info->destroyerCapacity);
    options.paint_field_capacity = std::max(1U, info->paintFieldCount);
    options.paint_rule_capacity = std::max(1U, info->paintRuleCapacity);
    const std::uint32_t coupling_capacity = std::max(1U, info->couplingCapacity);
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
    options.physics_debug.frame_capacity = info->debugFrameCapacity;
    options.rigid_sleeping = true;
    created->world_options = options;
    *simulation = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroySimulation(
    PmDevice device, PmSimulation simulation) {
    if (device == nullptr || simulation == nullptr || simulation->device != device) return;
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
    const PmResult current = parallel_mater::device::from_cuda(
        cudaSetDevice(device->services->ordinal()));
    if (current != PM_SUCCESS) return current;
    try {
        simulation->arena =
            std::make_unique<parallel_mater::simulation::MemoryArena>(
                static_cast<std::byte *>(
                    binding->persistentMemory->allocation) +
                    binding->persistentOffset,
                static_cast<std::size_t>(
                    simulation->requirements.persistent.size),
                static_cast<std::byte *>(binding->scratchMemory->allocation) +
                    binding->scratchOffset,
                static_cast<std::size_t>(simulation->requirements.scratch.size));
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
                simulation->world_options, initialized,
                device->services->stream()));
    }
    if (result != PM_SUCCESS) {
        simulation->arena.reset();
        return result;
    }
    *simulation->runtime = std::move(initialized);
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
    return parallel_mater::api::record(
        command_buffer,
        [simulation, first, copied = std::move(copied)](cudaStream_t stream) noexcept {
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            PmResult result = validate_bound(simulation);
            if (result != PM_SUCCESS) return result;
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                auto &slot = simulation->rigid_bodies[first + offset];
                if (slot.has_value()) {
                    result = parallel_mater::api::from_status(
                        simulation->runtime->remove_rigid_body(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const PmRigidBodyInfo &source = copied[offset];
                if (source.active == PM_FALSE) continue;
                TriangleMeshId mesh{};
                result = parallel_mater::api::register_geometry(
                    simulation, source.geometry, stream, mesh);
                if (result != PM_SUCCESS) return result;
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
                if (result != PM_SUCCESS) return result;
                slot = created;
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
    return parallel_mater::api::record(
        command_buffer, [simulation, copied = std::move(copied)](
                            cudaStream_t) noexcept {
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
    return parallel_mater::api::record(
        command_buffer, [simulation, first, copied = std::move(copied)](
                            cudaStream_t) noexcept {
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
    return parallel_mater::api::record(
        command_buffer, [simulation, first, copied = std::move(copied)](
                            cudaStream_t) noexcept {
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
    return parallel_mater::api::record(
        command_buffer, [simulation, first, copied = std::move(copied)](
                            cudaStream_t) noexcept {
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
    return parallel_mater::api::record(
        command_buffer, [simulation](cudaStream_t stream) noexcept {
            PmResult result = validate_bound(simulation);
            if (result != PM_SUCCESS) return result;
            {
                parallel_mater::simulation::ScopedMemoryArena arena_scope(
                    simulation->arena.get());
                *simulation->runtime = parallel_mater::SimulationRuntime{};
            }
            for (auto &[geometry, id] : simulation->geometries) {
                (void)id;
                if (geometry != nullptr && geometry->arena != nullptr)
                    geometry->arena->reset();
            }
            simulation->arena->reset();
            parallel_mater::SimulationRuntime initialized;
            {
                parallel_mater::simulation::ScopedMemoryArena arena_scope(
                    simulation->arena.get());
                result = parallel_mater::api::from_status(
                    parallel_mater::SimulationRuntime::create(
                        simulation->world_options, initialized, stream));
            }
            if (result != PM_SUCCESS) {
                simulation->execution_result = result;
                return result;
            }
            *simulation->runtime = std::move(initialized);
            simulation->geometries.clear();
            for (auto &slot : simulation->rigid_bodies) slot.reset();
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
            for (auto &slot : simulation->paint_rules) slot.reset();
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
        particles->format != PM_FORMAT_PARTICLE_STATE ||
        particles->stride != sizeof(PmParticleState) ||
        particles->count > simulation->fluids[fluid].particleCapacity ||
        particles->range.buffer == nullptr ||
        particles->range.buffer->device != command_buffer->device ||
        !parallel_mater::api::range_fits(
            particles->range.offset,
            static_cast<PmDeviceSize>(particles->count) * particles->stride,
            particles->range.buffer->size))
        return PM_ERROR_INVALID_ARGUMENT;
    const PmBufferView copied = *particles;
    const PmLiquidMaterialState material =
        command_buffer->bound_pipeline->fluids[fluid];
    return parallel_mater::api::record(
        command_buffer, [simulation, fluid, copied, material](
                            cudaStream_t stream) noexcept {
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            PmResult result = validate_bound(simulation);
            if (result != PM_SUCCESS) return result;
            auto &slot = simulation->fluid_ids[fluid];
            if (slot.has_value()) {
                result = parallel_mater::api::from_status(
                    simulation->runtime->remove_fluid(*slot, stream));
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
                parallel_mater::api::buffer_pointer(copied.range.buffer,
                                                     copied.range.offset));
            parallel_mater::FluidId created{};
            result = parallel_mater::api::from_status(
                simulation->runtime->add_fluid(
                    options, {source, copied.count}, created, stream));
            if (result != PM_SUCCESS) return result;
            slot = created;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdStepSimulation(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    const PmStepInfo *step) {
    if (simulation == nullptr || step == nullptr || step->timestep <= 0.0F ||
        step->substeps == 0U || simulation->device != command_buffer->device ||
        command_buffer->bound_pipeline == nullptr ||
        (step->flags & ~PM_STEP_COLLECT_CONTACT_EVENTS_BIT) != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    const PmStepInfo copied = *step;
    const PmPhysicsPipeline pipeline = command_buffer->bound_pipeline;
    if (command_buffer->statistics_open)
        command_buffer->statistics_simulation = simulation;
    return parallel_mater::api::record(
        command_buffer, [simulation, copied, pipeline](cudaStream_t stream) noexcept {
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
            result = parallel_mater::api::from_status(
                simulation->runtime->step(options, stream));
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
        destination->size < static_cast<PmDeviceSize>(count) * sizeof(PmRigidState))
        return PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record(
        command_buffer, [simulation, first, count, copied](cudaStream_t stream) noexcept {
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
                    cudaError_t copied_ids = cudaMemcpyAsync(
                        dense_ids.data(), view.ids.data,
                        dense_ids.size() * sizeof(dense_ids[0]),
                        cudaMemcpyDeviceToHost, stream);
                    if (copied_ids == cudaSuccess)
                        copied_ids = cudaStreamSynchronize(stream);
                    if (copied_ids != cudaSuccess)
                        return parallel_mater::device::from_cuda(copied_ids);
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
                const cudaError_t state_copy = cudaMemcpyAsync(
                    target + static_cast<std::size_t>(offset) *
                                 sizeof(PmRigidState),
                    view.states.data + dense,
                    static_cast<std::size_t>(run) * sizeof(PmRigidState),
                    cudaMemcpyDeviceToDevice, stream);
                if (state_copy != cudaSuccess)
                    return parallel_mater::device::from_cuda(state_copy);
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
    return parallel_mater::api::record(
        command_buffer, [simulation, fluid, copied](cudaStream_t stream) noexcept {
            if (!simulation->fluid_ids[fluid].has_value())
                return PM_ERROR_INVALID_STATE;
            parallel_mater::FluidDeviceView view{};
            PmResult result = parallel_mater::api::from_status(
                simulation->runtime->fluid_view(*simulation->fluid_ids[fluid], view));
            if (result != PM_SUCCESS) return result;
            auto *target = static_cast<std::byte *>(
                parallel_mater::api::buffer_pointer(copied.buffer, copied.offset));
            return parallel_mater::simulation::export_fluid_particles(
                stream, view.positions.data, view.velocities.data,
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
    *execution = simulation->execution_result;
    return PM_SUCCESS;
}
