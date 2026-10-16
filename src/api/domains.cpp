// SPDX-License-Identifier: MIT
#include "internal.hpp"
#include "../simulation/exports.hpp"
#include "../simulation/host_preparation.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace {

using parallel_mater::Vec2;
using parallel_mater::Vec3;

static_assert(sizeof(PmFloat2) == sizeof(Vec2));
static_assert(sizeof(PmFloat3) == sizeof(Vec3));

[[nodiscard]] Vec2 value(PmFloat2 source) noexcept {
    return {source.x, source.y};
}

[[nodiscard]] Vec3 value(PmFloat3 source) noexcept {
    return {source.x, source.y, source.z};
}

template <typename T>
[[nodiscard]] PmResult copy_array(const T *source, std::uint32_t count,
                                  std::vector<T> &destination) noexcept {
    if (count != 0U && source == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    try {
        if (count != 0U) destination.assign(source, source + count);
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    return PM_SUCCESS;
}

template <typename Id>
[[nodiscard]] PmResult require_slot(
    const std::vector<std::optional<Id>> &slots, std::uint32_t index,
    Id &output) noexcept {
    if (index >= slots.size() || !slots[index].has_value())
        return PM_ERROR_INVALID_STATE;
    output = *slots[index];
    return PM_SUCCESS;
}

struct RecordedCloth {
    PmBool32 active{};
    std::uint32_t material{};
    std::vector<PmFloat3> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<float> inverse_masses;
};

struct RecordedRope {
    PmBool32 active{};
    std::uint32_t material{};
    std::vector<PmFloat3> centerline;
    PmRopeAttachment first{};
    PmRopeAttachment last{};
};

struct RecordedSoftBody {
    PmBool32 active{};
    std::uint32_t material{};
    std::vector<PmFloat3> nodes;
    std::vector<PmSoftBodyBond> bonds;
    std::vector<float> inverse_masses;
    std::vector<PmFloat3> surface_vertices;
    std::vector<std::uint32_t> surface_indices;
    std::vector<PmSoftBodySurfaceBinding> surface_bindings;
};

struct RecordedEmitter {
    PmParticleEmitterInfo info{};
    std::vector<PmFloat3> vertices;
    std::vector<std::uint32_t> indices;
};

struct RecordedPaintField {
    PmPaintFieldInfo info{};
    std::vector<PmFloat2> uvs;
};

[[nodiscard]] PmResult validate_destination(
    PmCommandBuffer command_buffer, const PmBufferRange *destination,
    PmDeviceSize minimum) noexcept {
    if (destination == nullptr || destination->buffer == nullptr ||
        destination->buffer->device != command_buffer->device ||
        destination->size < minimum ||
        !parallel_mater::api::range_fits(destination->offset, destination->size,
                                         destination->buffer->size))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}

template <typename T>
[[nodiscard]] PmResult copy_device(
                                   parallel_mater::device::Services &services,
                                   const T *source,
                                   std::size_t count,
                                   std::vector<T> &destination) noexcept {
    try { destination.resize(count); }
    catch (const std::bad_alloc &) { return PM_ERROR_OUT_OF_HOST_MEMORY; }
    if (count == 0U) return PM_SUCCESS;
    if (source == nullptr) {
        std::fill(destination.begin(), destination.end(), T{});
        return PM_SUCCESS;
    }
    return services.copy_and_wait(destination.data(), source,
        static_cast<PmDeviceSize>(count) * sizeof(T),
        parallel_mater::device::CopyDirection::device_to_host);
}

template <typename T>
[[nodiscard]] PmResult upload_records(
                                      parallel_mater::device::Services &services,
                                      PmBufferRange destination,
                                      const std::vector<T> &records,
                                      std::uint32_t capacity,
                                      std::uint64_t revision) noexcept {
    const PmResult ownership =
        parallel_mater::api::require_device_ownership(destination.buffer);
    if (ownership != PM_SUCCESS) return ownership;
    const PmDeviceSize required = sizeof(PmDomainOutputHeader) +
        static_cast<PmDeviceSize>(records.size()) * sizeof(T);
    if (required > destination.size) return PM_ERROR_CAPACITY_EXCEEDED;
    const PmDomainOutputHeader header{
        static_cast<std::uint32_t>(records.size()), capacity, revision};
    auto *target = static_cast<std::byte *>(
        parallel_mater::api::buffer_pointer(destination.buffer,
                                             destination.offset));
    auto staging = std::unique_ptr<std::byte[]>(
        new (std::nothrow) std::byte[static_cast<std::size_t>(required)]);
    if (!staging) return PM_ERROR_OUT_OF_HOST_MEMORY;
    std::memcpy(staging.get(), &header, sizeof(header));
    if (!records.empty())
        std::memcpy(staging.get() + sizeof(header), records.data(),
                    records.size() * sizeof(T));
    return services.upload(target, std::move(staging), required);
}

[[nodiscard]] PmResult upload_bytes(
                                    parallel_mater::device::Services &services,
                                    PmBufferRange destination,
                                    const void *source,
                                    std::size_t size) noexcept {
    const PmResult ownership =
        parallel_mater::api::require_device_ownership(destination.buffer);
    if (ownership != PM_SUCCESS) return ownership;
    if (size == 0U) return PM_SUCCESS;
    auto staging = std::unique_ptr<std::byte[]>(
        new (std::nothrow) std::byte[size]);
    if (!staging) return PM_ERROR_OUT_OF_HOST_MEMORY;
    std::memcpy(staging.get(), source, size);
    return services.upload(
        parallel_mater::api::buffer_pointer(destination.buffer,
                                             destination.offset),
        std::move(staging), size);
}

[[nodiscard]] std::uint32_t rigid_slot(PmSimulation simulation,
                                       parallel_mater::RigidBodyId id) noexcept {
    for (std::uint32_t index = 0U; index < simulation->rigid_bodies.size(); ++index)
        if (simulation->rigid_bodies[index].has_value() &&
            *simulation->rigid_bodies[index] == id)
            return index;
    return PM_INVALID_INDEX;
}

} // namespace

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWriteCloths(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count, const PmClothInfo *cloths) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        command_buffer->bound_pipeline == nullptr || count == 0U || cloths == nullptr ||
        first > simulation->cloths.size() || count > simulation->cloths.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<RecordedCloth> copied(count);
    for (std::uint32_t index = 0U; index < count; ++index) {
        const PmClothInfo &source = cloths[index];
        auto &target = copied[index];
        target.active = source.active;
        target.material = source.materialIndex;
        if (source.active == PM_FALSE) continue;
        const PmClothCapacity capacity =
            simulation->cloth_capacities[first + index];
        if (source.materialIndex >= command_buffer->bound_pipeline->cloths.size() ||
            source.vertexCount == 0U || source.triangleIndexCount == 0U ||
            source.triangleIndexCount % 3U != 0U ||
            source.vertexCount > capacity.vertexCapacity ||
            source.triangleIndexCount > capacity.triangleIndexCapacity ||
            source.triangleIndexCount > capacity.bondCapacity)
            return PM_ERROR_INVALID_ARGUMENT;
        const auto &material = command_buffer->bound_pipeline->cloths[source.materialIndex];
        if ((material.breakStrain > 0 || material.impactBreakImpulse > 0) &&
            std::uint64_t(source.vertexCount) + source.triangleIndexCount > capacity.vertexCapacity)
            return PM_ERROR_CAPACITY_EXCEEDED;
        PmResult result = copy_array(source.pVertices, source.vertexCount,
                                     target.vertices);
        if (result == PM_SUCCESS)
            result = copy_array(source.pTriangleIndices,
                                source.triangleIndexCount, target.indices);
        if (result == PM_SUCCESS)
            result = copy_array(source.pInverseMasses, source.vertexCount,
                                target.inverse_masses);
        if (result != PM_SUCCESS) return result;
    }
    const PmPhysicsPipeline pipeline = command_buffer->bound_pipeline;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, pipeline, first, copied = std::move(copied)](
            parallel_mater::device::Services &) noexcept {
            if (simulation->binding.persistentMemory == nullptr) return PM_ERROR_INVALID_STATE;
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                parallel_mater::simulation::ScopedMemoryArena slot_scope(
                    simulation->domain_memory.cloths[first + offset].get());
                auto &slot = simulation->cloths[first + offset];
                if (slot.has_value()) {
                    PmResult result = parallel_mater::api::from_status(
                        simulation->runtime->remove_cloth(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const RecordedCloth &source = copied[offset];
                if (source.active == PM_FALSE) continue;
                const auto &material = pipeline->cloths[source.material];
                parallel_mater::ClothOptions options{};
                options.vertices = {reinterpret_cast<const Vec3 *>(
                    source.vertices.data()), source.vertices.size()};
                options.triangle_indices = {source.indices.data(), source.indices.size()};
                options.inverse_masses = {source.inverse_masses.data(),
                                          source.inverse_masses.size()};
                options.vertex_mass = material.vertexMass;
                options.thickness = material.thickness;
                options.stretch_compliance = material.stretchCompliance;
                options.bending_compliance = material.bendingCompliance;
                options.velocity_damping = material.velocityDamping;
                options.contact_friction = material.contactFriction;
                options.solver_iterations = material.solverIterations;
                options.break_strain = material.breakStrain;
                options.fracture_persistence_substeps =
                    material.fracturePersistenceSubsteps;
                options.impact_break_impulse = material.impactBreakImpulse;
                options.preserve_volume = material.preserveVolume == PM_TRUE;
                options.target_volume = material.targetVolume;
                options.volume_compliance = material.volumeCompliance;
                parallel_mater::ClothId created{};
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->add_cloth(options, created));
                if (result != PM_SUCCESS) return result;
                slot = created;
            }
            simulation->couplings_initialized = false;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWriteRopes(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count, const PmRopeInfo *ropes) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        command_buffer->bound_pipeline == nullptr || count == 0U || ropes == nullptr ||
        first > simulation->ropes.size() || count > simulation->ropes.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<RecordedRope> copied(count);
    for (std::uint32_t index = 0U; index < count; ++index) {
        const PmRopeInfo &source = ropes[index];
        auto &target = copied[index];
        target.active = source.active;
        target.material = source.materialIndex;
        target.first = source.first;
        target.last = source.last;
        if (source.active == PM_FALSE) continue;
        const PmRopeCapacity capacity =
            simulation->rope_capacities[first + index];
        if (source.materialIndex >= command_buffer->bound_pipeline->ropes.size() ||
            source.pointCount < 2U || source.pointCount > capacity.nodeCapacity)
            return PM_ERROR_INVALID_ARGUMENT;
        const PmResult result = copy_array(source.pCenterline, source.pointCount,
                                           target.centerline);
        if (result != PM_SUCCESS) return result;
        std::vector<Vec3> sampled;
        const auto prepared = parallel_mater::sample_rope_centerline(
            {reinterpret_cast<const Vec3 *>(target.centerline.data()), target.centerline.size()},
            command_buffer->bound_pipeline->ropes[source.materialIndex].nodeSpacing, sampled);
        if (!prepared) return parallel_mater::api::from_status(prepared);
        if (sampled.size() > capacity.nodeCapacity) return PM_ERROR_CAPACITY_EXCEEDED;
    }
    const PmPhysicsPipeline pipeline = command_buffer->bound_pipeline;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, pipeline, first, copied = std::move(copied)](
            parallel_mater::device::Services &) noexcept {
            if (simulation->binding.persistentMemory == nullptr) return PM_ERROR_INVALID_STATE;
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                parallel_mater::simulation::ScopedMemoryArena slot_scope(
                    simulation->domain_memory.ropes[first + offset].get());
                auto &slot = simulation->ropes[first + offset];
                if (slot.has_value()) {
                    PmResult result = parallel_mater::api::from_status(
                        simulation->runtime->remove_rope(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const RecordedRope &source = copied[offset];
                if (source.active == PM_FALSE) continue;
                const auto &material = pipeline->ropes[source.material];
                parallel_mater::RopeOptions options{};
                options.centerline = {reinterpret_cast<const Vec3 *>(
                    source.centerline.data()), source.centerline.size()};
                options.node_spacing = material.nodeSpacing;
                options.radius = material.radius;
                options.mass = material.mass;
                options.stretch_compliance = material.stretchCompliance;
                options.velocity_damping = material.velocityDamping;
                options.maximum_substep_timestep = material.maximumSubstepTimestep;
                options.friction = material.friction;
                options.maximum_speed = material.maximumSpeed;
                options.solver_iterations = material.solverIterations;
                options.self_collision = material.selfCollision == PM_TRUE;
                if (source.first.enabled == PM_TRUE) {
                    PmResult result = require_slot(simulation->rigid_bodies,
                        source.first.rigidBodyIndex, options.first.body);
                    if (result != PM_SUCCESS) return result;
                    options.first.local_anchor = value(source.first.localAnchor);
                    options.first.enabled = true;
                }
                if (source.last.enabled == PM_TRUE) {
                    PmResult result = require_slot(simulation->rigid_bodies,
                        source.last.rigidBodyIndex, options.last.body);
                    if (result != PM_SUCCESS) return result;
                    options.last.local_anchor = value(source.last.localAnchor);
                    options.last.enabled = true;
                }
                parallel_mater::RopeId created{};
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->add_rope(options, created));
                if (result != PM_SUCCESS) return result;
                slot = created;
            }
            simulation->couplings_initialized = false;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWriteSoftBodies(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count,
    const PmSoftBodyInfo *soft_bodies) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        command_buffer->bound_pipeline == nullptr || count == 0U ||
        soft_bodies == nullptr || first > simulation->soft_bodies.size() ||
        count > simulation->soft_bodies.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<RecordedSoftBody> copied(count);
    for (std::uint32_t index = 0U; index < count; ++index) {
        const PmSoftBodyInfo &source = soft_bodies[index];
        auto &target = copied[index];
        target.active = source.active;
        target.material = source.materialIndex;
        if (source.active == PM_FALSE) continue;
        const PmSoftBodyCapacity capacity =
            simulation->soft_body_capacities[first + index];
        if (source.materialIndex >= command_buffer->bound_pipeline->soft_bodies.size() ||
            source.nodeCount == 0U || source.bondCount == 0U ||
            source.surfaceVertexCount == 0U ||
            source.surfaceTriangleIndexCount == 0U ||
            source.surfaceTriangleIndexCount % 3U != 0U ||
            source.nodeCount > capacity.nodeCapacity ||
            source.bondCount > capacity.bondCapacity ||
            source.surfaceVertexCount > capacity.surfaceVertexCapacity ||
            source.surfaceTriangleIndexCount >
                capacity.surfaceTriangleIndexCapacity)
            return PM_ERROR_INVALID_ARGUMENT;
        PmResult result = copy_array(source.pNodes, source.nodeCount, target.nodes);
        if (result == PM_SUCCESS)
            result = copy_array(source.pBonds, source.bondCount, target.bonds);
        if (result == PM_SUCCESS)
            result = copy_array(source.pInverseMasses, source.nodeCount,
                                target.inverse_masses);
        if (result == PM_SUCCESS)
            result = copy_array(source.pSurfaceVertices,
                                source.surfaceVertexCount, target.surface_vertices);
        if (result == PM_SUCCESS)
            result = copy_array(source.pSurfaceTriangleIndices,
                                source.surfaceTriangleIndexCount,
                                target.surface_indices);
        if (result == PM_SUCCESS)
            result = copy_array(source.pSurfaceBindings,
                                source.surfaceVertexCount,
                                target.surface_bindings);
        if (result != PM_SUCCESS) return result;
    }
    const PmPhysicsPipeline pipeline = command_buffer->bound_pipeline;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, pipeline, first, copied = std::move(copied)](
            parallel_mater::device::Services &) noexcept {
            if (simulation->binding.persistentMemory == nullptr) return PM_ERROR_INVALID_STATE;
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                parallel_mater::simulation::ScopedMemoryArena slot_scope(
                    simulation->domain_memory.soft_bodies[first + offset].get());
                auto &slot = simulation->soft_bodies[first + offset];
                if (slot.has_value()) {
                    PmResult result = parallel_mater::api::from_status(
                        simulation->runtime->remove_soft_body(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const RecordedSoftBody &source = copied[offset];
                if (source.active == PM_FALSE) continue;
                const auto &material = pipeline->soft_bodies[source.material];
                std::vector<parallel_mater::SoftBodyBond> bonds(source.bonds.size());
                std::vector<parallel_mater::SoftBodySurfaceBinding> bindings(
                    source.surface_bindings.size());
                for (std::size_t i = 0; i < bonds.size(); ++i)
                    bonds[i] = {source.bonds[i].first, source.bonds[i].second,
                                source.bonds[i].restLength};
                for (std::size_t i = 0; i < bindings.size(); ++i) {
                    std::copy_n(source.surface_bindings[i].nodes, 4U,
                                bindings[i].nodes);
                    std::copy_n(source.surface_bindings[i].weights, 4U,
                                bindings[i].weights);
                }
                parallel_mater::SoftBodyOptions options{};
                options.nodes = {reinterpret_cast<const Vec3 *>(source.nodes.data()),
                                 source.nodes.size()};
                options.bonds = {bonds.data(), bonds.size()};
                options.inverse_masses = {source.inverse_masses.data(),
                                          source.inverse_masses.size()};
                options.surface_vertices = {reinterpret_cast<const Vec3 *>(
                    source.surface_vertices.data()), source.surface_vertices.size()};
                options.surface_triangle_indices = {source.surface_indices.data(),
                                                    source.surface_indices.size()};
                options.surface_bindings = {bindings.data(), bindings.size()};
                options.node_mass = material.nodeMass;
                options.node_radius = material.nodeRadius;
                options.stretch_compliance = material.stretchCompliance;
                options.velocity_damping = material.velocityDamping;
                options.spring_damping = material.springDamping;
                options.contact_friction = material.contactFriction;
                options.shape_matching_stiffness = material.shapeMatchingStiffness;
                options.maximum_projection_fraction =
                    material.maximumProjectionFraction;
                options.constraint_velocity_response =
                    material.constraintVelocityResponse;
                options.maximum_speed = material.maximumSpeed;
                options.solver_iterations = material.solverIterations;
                parallel_mater::SoftBodyId created{};
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->add_soft_body(options, created));
                if (result != PM_SUCCESS) return result;
                slot = created;
            }
            simulation->couplings_initialized = false;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWriteSmoke(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count, const PmSmokeInfo *smoke) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        command_buffer->bound_pipeline == nullptr || count == 0U || smoke == nullptr ||
        first > simulation->smokes.size() || count > simulation->smokes.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmSmokeInfo> copied(smoke, smoke + count);
    const PmPhysicsPipeline pipeline = command_buffer->bound_pipeline;
    for (std::uint32_t index = 0U; index < count; ++index) {
        if (copied[index].active == PM_FALSE) continue;
        if (copied[index].materialIndex >= pipeline->smokes.size())
            return PM_ERROR_INVALID_ARGUMENT;
        const auto &material = pipeline->smokes[copied[index].materialIndex];
        const std::uint64_t grid_cells =
            static_cast<std::uint64_t>(material.gridResolution) *
            material.gridResolution * material.gridVerticalResolution;
        const PmSmokeCapacity capacity =
            simulation->smoke_capacities[first + index];
        if (copied[index].particleCapacity == 0U ||
            copied[index].particleCapacity > capacity.particleCapacity ||
            grid_cells > capacity.gridCellCapacity)
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, pipeline, first, copied = std::move(copied)](
            parallel_mater::device::Services &) noexcept {
            if (simulation->binding.persistentMemory == nullptr) return PM_ERROR_INVALID_STATE;
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                parallel_mater::simulation::ScopedMemoryArena slot_scope(
                    simulation->domain_memory.smokes[first + offset].get());
                auto &slot = simulation->smokes[first + offset];
                if (slot.has_value()) {
                    PmResult result = parallel_mater::api::from_status(
                        simulation->runtime->remove_smoke(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const PmSmokeInfo &source = copied[offset];
                if (source.active == PM_FALSE) continue;
                if (source.materialIndex >= pipeline->smokes.size())
                    return PM_ERROR_INVALID_ARGUMENT;
                const auto &material = pipeline->smokes[source.materialIndex];
                parallel_mater::SmokeOptions options{};
                options.capacity = source.particleCapacity;
                options.emitter_center = value(source.emitterCenter);
                options.emitter_half_extents = value(source.emitterHalfExtents);
                options.initial_velocity = value(source.initialVelocity);
                options.wind = value(source.wind);
                options.particles_per_second = source.particlesPerSecond;
                options.lifetime = source.lifetime;
                options.particle_radius = material.particleRadius;
                options.buoyancy = material.buoyancy;
                options.response = material.response;
                options.rest_number_density = material.restNumberDensity;
                options.pressure_stiffness = material.pressureStiffness;
                options.viscosity = material.viscosity;
                options.vorticity_confinement = material.vorticityConfinement;
                options.maximum_speed = material.maximumSpeed;
                options.grid_resolution = material.gridResolution;
                options.grid_vertical_resolution = material.gridVerticalResolution;
                options.grid_pressure_iterations = material.gridPressureIterations;
                options.grid_kinematic_viscosity = material.gridKinematicViscosity;
                options.grid_les_coefficient = material.gridLesCoefficient;
                options.grid_pressure_tolerance = material.gridPressureTolerance;
                options.grid_minimum = value(source.gridMinimum);
                options.grid_edge_length = source.gridEdgeLength;
                parallel_mater::SmokeId created{};
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->add_smoke(options, created));
                if (result != PM_SUCCESS) return result;
                slot = created;
            }
            simulation->couplings_initialized = false;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWriteParticleEmitters(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count,
    const PmParticleEmitterInfo *emitters) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || emitters == nullptr || first > simulation->emitters.size() ||
        count > simulation->emitters.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<RecordedEmitter> copied(count);
    for (std::uint32_t index = 0U; index < count; ++index) {
        copied[index].info = emitters[index];
        if (emitters[index].active == PM_FALSE) continue;
        PmResult result = copy_array(emitters[index].pVertices,
            emitters[index].vertexCount, copied[index].vertices);
        if (result == PM_SUCCESS)
            result = copy_array(emitters[index].pTriangleIndices,
                emitters[index].triangleIndexCount, copied[index].indices);
        if (result != PM_SUCCESS) return result;
    }
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, copied = std::move(copied)](
            parallel_mater::device::Services &) noexcept {
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                auto &slot = simulation->emitters[first + offset];
                if (slot.has_value()) {
                    PmResult result = parallel_mater::api::from_status(
                        simulation->runtime->remove_particle_source(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const RecordedEmitter &source = copied[offset];
                if (source.info.active == PM_FALSE) continue;
                parallel_mater::FluidId fluid{};
                PmResult result = require_slot(simulation->fluid_ids,
                    source.info.fluidIndex, fluid);
                if (result != PM_SUCCESS) return result;
                parallel_mater::ParticleSourceMesh mesh{
                    {reinterpret_cast<const Vec3 *>(source.vertices.data()),
                     source.vertices.size()},
                    {source.indices.data(), source.indices.size()},
                    source.info.spacing};
                parallel_mater::ParticleSourceOptions options{};
                options.fluid = fluid;
                options.initial_velocity = value(source.info.initialVelocity);
                options.initial_temperature = source.info.initialTemperatureCelsius;
                options.enabled = true;
                parallel_mater::ParticleSourceId created{};
                result = parallel_mater::api::from_status(
                    simulation->runtime->add_particle_source(mesh, options, created));
                if (result != PM_SUCCESS) return result;
                slot = created;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWriteParticleDestroyers(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count,
    const PmParticleDestroyerInfo *destroyers) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || destroyers == nullptr ||
        first > simulation->destroyers.size() ||
        count > simulation->destroyers.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmParticleDestroyerInfo> copied(destroyers, destroyers + count);
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, copied = std::move(copied)](
            parallel_mater::device::Services &) noexcept {
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                auto &slot = simulation->destroyers[first + offset];
                if (slot.has_value()) {
                    PmResult result = parallel_mater::api::from_status(
                        simulation->runtime->remove_particle_destroy_plane(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const auto &source = copied[offset];
                if (source.active == PM_FALSE) continue;
                parallel_mater::FluidId fluid{};
                PmResult result = require_slot(simulation->fluid_ids,
                    source.fluidIndex, fluid);
                if (result != PM_SUCCESS) return result;
                parallel_mater::ParticleDestroyPlaneOptions options{};
                options.fluid = fluid;
                options.plane.center = value(source.center);
                options.plane.orientation = {source.orientation[0],
                    source.orientation[1], source.orientation[2],
                    source.orientation[3]};
                options.plane.half_extents = value(source.halfExtents);
                switch (source.crossing) {
                case PM_CROSSING_ALONG_NORMAL:
                    options.crossing = parallel_mater::CrossingDirection::along_normal;
                    break;
                case PM_CROSSING_AGAINST_NORMAL:
                    options.crossing = parallel_mater::CrossingDirection::against_normal;
                    break;
                case PM_CROSSING_EITHER:
                    options.crossing = parallel_mater::CrossingDirection::either;
                    break;
                }
                options.enabled = true;
                parallel_mater::ParticleDestroyPlaneId created{};
                result = parallel_mater::api::from_status(
                    simulation->runtime->add_particle_destroy_plane(options, created));
                if (result != PM_SUCCESS) return result;
                slot = created;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWritePaintFields(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count,
    const PmPaintFieldInfo *fields) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || fields == nullptr ||
        first > simulation->paint_fields.size() ||
        count > simulation->paint_fields.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<RecordedPaintField> copied(count);
    for (std::uint32_t index = 0U; index < count; ++index) {
        copied[index].info = fields[index];
        if (fields[index].active == PM_FALSE) continue;
        const PmPaintFieldCapacity capacity =
            simulation->paint_field_capacities[first + index];
        if (fields[index].vertexCount == 0U ||
            fields[index].vertexCount > capacity.vertexCapacity ||
            fields[index].width == 0U || fields[index].height == 0U ||
            fields[index].width > capacity.width ||
            fields[index].height > capacity.height ||
            (fields[index].targetType == PM_PAINT_TARGET_RIGID &&
             fields[index].geometry == nullptr) ||
            fields[index].targetType > PM_PAINT_TARGET_CLOTH)
            return PM_ERROR_CAPACITY_EXCEEDED;
        const PmResult result = copy_array(fields[index].pVertexUvs,
            fields[index].vertexCount, copied[index].uvs);
        if (result != PM_SUCCESS) return result;
    }
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, copied = std::move(copied)](
            parallel_mater::device::Services &services) noexcept {
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                auto &slot = simulation->paint_fields[first + offset];
                if (slot.has_value()) {
                    PmResult result = parallel_mater::api::from_status(
                        simulation->runtime->remove_paint_field(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const RecordedPaintField &source = copied[offset];
                if (source.info.active == PM_FALSE) continue;
                void *uv_allocation = nullptr;
                const PmDeviceSize uv_bytes =
                    static_cast<PmDeviceSize>(source.uvs.size()) * sizeof(Vec2);
                PmResult allocation_result =
                    services.allocate(uv_bytes, false, &uv_allocation);
                if (allocation_result != PM_SUCCESS) return allocation_result;
                auto *device_uvs = static_cast<Vec2 *>(uv_allocation);
                PmResult copy_result = services.copy_and_wait(
                    device_uvs, source.uvs.data(), uv_bytes,
                    parallel_mater::device::CopyDirection::host_to_device);
                if (copy_result != PM_SUCCESS) {
                    services.free(device_uvs);
                    return copy_result;
                }
                parallel_mater::PaintFieldOptions options{};
                if (source.info.targetType == PM_PAINT_TARGET_RIGID) {
                    PmResult result = require_slot(simulation->rigid_bodies,
                        source.info.targetIndex, options.body);
                    if (result != PM_SUCCESS) {
                        services.free(device_uvs);
                        return result;
                    }
                    PmResult geometry_result =
                        parallel_mater::api::register_geometry(
                            simulation, source.info.geometry, services,
                            options.mesh);
                    if (geometry_result != PM_SUCCESS) {
                        services.free(device_uvs);
                        return geometry_result;
                    }
                } else {
                    PmResult result = require_slot(simulation->cloths,
                        source.info.targetIndex, options.cloth);
                    if (result != PM_SUCCESS) {
                        services.free(device_uvs);
                        return result;
                    }
                }
                options.vertex_uvs = {device_uvs, source.uvs.size()};
                options.width = source.info.width;
                options.height = source.info.height;
                parallel_mater::PaintFieldId created{};
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->add_paint_field(options, created, queue));
                services.free(device_uvs);
                if (result != PM_SUCCESS) return result;
                slot = created;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWritePaintRules(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count, const PmPaintRuleInfo *rules) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || rules == nullptr || first > simulation->paint_rules.size() ||
        count > simulation->paint_rules.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmPaintRuleInfo> copied(rules, rules + count);
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, copied = std::move(copied)](
            parallel_mater::device::Services &) noexcept {
            parallel_mater::simulation::ScopedMemoryArena arena_scope(
                simulation->arena.get());
            for (std::uint32_t offset = 0U; offset < copied.size(); ++offset) {
                auto &slot = simulation->paint_rules[first + offset];
                if (slot.has_value()) {
                    PmResult result = parallel_mater::api::from_status(
                        simulation->runtime->remove_paint_rule(*slot));
                    if (result != PM_SUCCESS) return result;
                    slot.reset();
                }
                const auto &source = copied[offset];
                if (source.active == PM_FALSE) continue;
                parallel_mater::PaintRuleOptions options{};
                if (source.sourceType == PM_PAINT_SOURCE_LIQUID) {
                    PmResult result = require_slot(simulation->fluid_ids,
                        source.sourceIndex, options.source);
                    if (result != PM_SUCCESS) return result;
                } else {
                    PmResult result = require_slot(simulation->rigid_bodies,
                        source.sourceIndex, options.rigid_source);
                    if (result != PM_SUCCESS) return result;
                }
                PmResult result = require_slot(simulation->paint_fields,
                    source.targetFieldIndex, options.target);
                if (result != PM_SUCCESS) return result;
                options.reach = source.reach;
                options.brush_radius = source.brushRadius;
                options.enabled = true;
                parallel_mater::PaintRuleId created{};
                result = parallel_mater::api::from_status(
                    simulation->runtime->add_paint_rule(options, created));
                if (result != PM_SUCCESS) return result;
                slot = created;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdClearPaintFields(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    if (simulation == nullptr || simulation->device != command_buffer->device ||
        count == 0U || first > simulation->paint_fields.size() ||
        count > simulation->paint_fields.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, count](
            parallel_mater::device::Services &services) noexcept {
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            for (std::uint32_t offset = 0U; offset < count; ++offset) {
                const auto &slot = simulation->paint_fields[first + offset];
                if (!slot.has_value()) return PM_ERROR_INVALID_STATE;
                const PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->clear_paint_field(*slot, queue));
                if (result != PM_SUCCESS) return result;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyClothState(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t cloth, const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmResult result = validate_destination(command_buffer, destination,
                                           sizeof(PmDomainOutputHeader));
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device ||
        cloth >= simulation->cloths.size())
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, cloth, copied](
            parallel_mater::device::Services &services) noexcept {
            const PmResult ownership =
                parallel_mater::api::require_device_ownership(copied.buffer);
            if (ownership != PM_SUCCESS) return ownership;
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::ClothId id{};
            PmResult result = require_slot(simulation->cloths, cloth, id);
            if (result != PM_SUCCESS) return result;
            parallel_mater::ClothDeviceView view{};
            result = parallel_mater::api::from_status(
                simulation->runtime->cloth_view(id, view));
            if (result != PM_SUCCESS) return result;
            const std::uint32_t count = view.vertex_count;
            const PmDeviceSize required = sizeof(PmDomainOutputHeader) +
                static_cast<PmDeviceSize>(count) * sizeof(PmClothVertexState);
            if (required > copied.size) return PM_ERROR_CAPACITY_EXCEEDED;
            auto *target = static_cast<std::byte *>(
                parallel_mater::api::buffer_pointer(copied.buffer, copied.offset));
            return parallel_mater::simulation::export_cloth_vertices(
                queue, view.positions.data, view.velocities.data,
                view.rigid_contact_forces.data, view.fluid_contact_forces.data,
                view.soft_body_contact_forces.data,
                view.rope_contact_forces.data, count,
                simulation->cloth_capacities[cloth].vertexCapacity,
                simulation->submitted_steps,
                reinterpret_cast<PmDomainOutputHeader *>(target),
                reinterpret_cast<PmClothVertexState *>(
                    target + sizeof(PmDomainOutputHeader)));
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyClothSurface(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t cloth, const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmResult result = validate_destination(command_buffer, destination,
                                           sizeof(PmDomainOutputHeader));
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device ||
        cloth >= simulation->cloths.size())
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, cloth, copied](
            parallel_mater::device::Services &services) noexcept {
            const PmResult ownership =
                parallel_mater::api::require_device_ownership(copied.buffer);
            if (ownership != PM_SUCCESS) return ownership;
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::ClothId id{};
            PmResult result = require_slot(simulation->cloths, cloth, id);
            if (result != PM_SUCCESS) return result;
            parallel_mater::ClothDeviceView view{};
            result = parallel_mater::api::from_status(
                simulation->runtime->cloth_view(id, view));
            if (result != PM_SUCCESS) return result;
            const Vec3 *positions = view.surface_positions.size != 0U
                ? view.surface_positions.data : view.positions.data;
            const std::uint32_t count = static_cast<std::uint32_t>(
                view.surface_positions.size != 0U
                    ? view.surface_positions.size : view.positions.size);
            const PmDeviceSize required = sizeof(PmDomainOutputHeader) +
                static_cast<PmDeviceSize>(count) * sizeof(PmFloat3);
            if (required > copied.size) return PM_ERROR_CAPACITY_EXCEEDED;
            auto *target = static_cast<std::byte *>(
                parallel_mater::api::buffer_pointer(copied.buffer, copied.offset));
            return parallel_mater::simulation::export_surface_positions(
                queue, positions, count,
                simulation->cloth_capacities[cloth].triangleIndexCapacity,
                simulation->submitted_steps,
                reinterpret_cast<PmDomainOutputHeader *>(target),
                reinterpret_cast<PmFloat3 *>(
                    target + sizeof(PmDomainOutputHeader)));
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyRopeState(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t rope_index, const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmResult result = validate_destination(command_buffer, destination,
                                           sizeof(PmDomainOutputHeader));
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device ||
        rope_index >= simulation->ropes.size())
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, rope_index, copied](
            parallel_mater::device::Services &services) noexcept {
            const PmResult ownership =
                parallel_mater::api::require_device_ownership(copied.buffer);
            if (ownership != PM_SUCCESS) return ownership;
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::RopeId id{};
            PmResult result = require_slot(simulation->ropes, rope_index, id);
            if (result != PM_SUCCESS) return result;
            parallel_mater::RopeDeviceView view{};
            result = parallel_mater::api::from_status(
                simulation->runtime->rope_view(id, view));
            if (result != PM_SUCCESS) return result;
            const std::uint32_t count = static_cast<std::uint32_t>(
                view.positions.size);
            const PmDeviceSize required = sizeof(PmDomainOutputHeader) +
                static_cast<PmDeviceSize>(count) * sizeof(PmRopeNodeState);
            if (required > copied.size) return PM_ERROR_CAPACITY_EXCEEDED;
            auto *target = static_cast<std::byte *>(
                parallel_mater::api::buffer_pointer(copied.buffer, copied.offset));
            return parallel_mater::simulation::export_rope_nodes(
                queue, view.positions.data, view.velocities.data,
                view.constraint_forces.data, view.contact_forces.data,
                view.fluid_contact_forces.data,
                view.soft_body_contact_forces.data, count,
                simulation->rope_capacities[rope_index].nodeCapacity,
                simulation->submitted_steps,
                reinterpret_cast<PmDomainOutputHeader *>(target),
                reinterpret_cast<PmRopeNodeState *>(
                    target + sizeof(PmDomainOutputHeader)));
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopySoftBodyState(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t soft_body, const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmResult result = validate_destination(command_buffer, destination,
                                           sizeof(PmDomainOutputHeader));
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device ||
        soft_body >= simulation->soft_bodies.size())
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, soft_body, copied](
            parallel_mater::device::Services &services) noexcept {
            const PmResult ownership =
                parallel_mater::api::require_device_ownership(copied.buffer);
            if (ownership != PM_SUCCESS) return ownership;
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::SoftBodyId id{};
            PmResult result = require_slot(simulation->soft_bodies, soft_body, id);
            if (result != PM_SUCCESS) return result;
            parallel_mater::SoftBodyDeviceView view{};
            result = parallel_mater::api::from_status(
                simulation->runtime->soft_body_view(id, view));
            if (result != PM_SUCCESS) return result;
            const std::uint32_t count = view.node_count;
            const PmDeviceSize required = sizeof(PmDomainOutputHeader) +
                static_cast<PmDeviceSize>(count) * sizeof(PmSoftBodyNodeState);
            if (required > copied.size) return PM_ERROR_CAPACITY_EXCEEDED;
            auto *target = static_cast<std::byte *>(
                parallel_mater::api::buffer_pointer(copied.buffer, copied.offset));
            return parallel_mater::simulation::export_soft_body_nodes(
                queue, view.positions.data, view.velocities.data,
                view.rigid_contact_forces.data, view.cloth_contact_forces.data,
                view.fluid_contact_forces.data, view.rope_contact_forces.data,
                count, simulation->soft_body_capacities[soft_body].nodeCapacity,
                simulation->submitted_steps,
                reinterpret_cast<PmDomainOutputHeader *>(target),
                reinterpret_cast<PmSoftBodyNodeState *>(
                    target + sizeof(PmDomainOutputHeader)));
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopySoftBodySurface(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t soft_body, const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmResult result = validate_destination(command_buffer, destination,
                                           sizeof(PmDomainOutputHeader));
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device ||
        soft_body >= simulation->soft_bodies.size())
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, soft_body, copied](
            parallel_mater::device::Services &services) noexcept {
            const PmResult ownership =
                parallel_mater::api::require_device_ownership(copied.buffer);
            if (ownership != PM_SUCCESS) return ownership;
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::SoftBodyId id{};
            PmResult result = require_slot(simulation->soft_bodies, soft_body, id);
            if (result != PM_SUCCESS) return result;
            parallel_mater::SoftBodyDeviceView view{};
            result = parallel_mater::api::from_status(
                simulation->runtime->soft_body_view(id, view));
            if (result != PM_SUCCESS) return result;
            const std::uint32_t count = static_cast<std::uint32_t>(
                view.surface_positions.size);
            const PmDeviceSize required = sizeof(PmDomainOutputHeader) +
                static_cast<PmDeviceSize>(count) * sizeof(PmFloat3);
            if (required > copied.size) return PM_ERROR_CAPACITY_EXCEEDED;
            auto *target = static_cast<std::byte *>(
                parallel_mater::api::buffer_pointer(copied.buffer, copied.offset));
            return parallel_mater::simulation::export_surface_positions(
                queue, view.surface_positions.data, count,
                simulation->soft_body_capacities[soft_body].surfaceVertexCapacity,
                simulation->submitted_steps,
                reinterpret_cast<PmDomainOutputHeader *>(target),
                reinterpret_cast<PmFloat3 *>(
                    target + sizeof(PmDomainOutputHeader)));
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopySmokeState(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t smoke_index, const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmResult result = validate_destination(command_buffer, destination,
                                           sizeof(PmDomainOutputHeader));
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device ||
        smoke_index >= simulation->smokes.size())
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, smoke_index, copied](
            parallel_mater::device::Services &services) noexcept {
            const PmResult ownership =
                parallel_mater::api::require_device_ownership(copied.buffer);
            if (ownership != PM_SUCCESS) return ownership;
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::SmokeId id{};
            PmResult result = require_slot(simulation->smokes, smoke_index, id);
            if (result != PM_SUCCESS) return result;
            parallel_mater::SmokeDeviceView view{};
            result = parallel_mater::api::from_status(
                simulation->runtime->smoke_view(id, view));
            if (result != PM_SUCCESS) return result;
            const std::uint32_t count = view.particle_count;
            const PmDeviceSize required = sizeof(PmDomainOutputHeader) +
                static_cast<PmDeviceSize>(count) * sizeof(PmSmokeParticleState);
            if (required > copied.size) return PM_ERROR_CAPACITY_EXCEEDED;
            auto *target = static_cast<std::byte *>(
                parallel_mater::api::buffer_pointer(copied.buffer, copied.offset));
            return parallel_mater::simulation::export_smoke_particles(
                queue, view.positions.data, view.velocities.data,
                view.ages.data, view.number_densities.data,
                view.pressures.data, view.vorticities.data, count,
                simulation->smoke_capacities[smoke_index].particleCapacity,
                view.revision,
                reinterpret_cast<PmDomainOutputHeader *>(target),
                reinterpret_cast<PmSmokeParticleState *>(
                    target + sizeof(PmDomainOutputHeader)));
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyPaintField(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t field, const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmResult result = validate_destination(command_buffer, destination,
                                           sizeof(PmDomainOutputHeader));
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device ||
        field >= simulation->paint_fields.size())
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, field, copied](
            parallel_mater::device::Services &services) noexcept {
            parallel_mater::PaintFieldId id{};
            PmResult result = require_slot(simulation->paint_fields, field, id);
            if (result != PM_SUCCESS) return result;
            parallel_mater::PaintFieldDeviceView view{};
            result = parallel_mater::api::from_status(
                simulation->runtime->paint_field_view(id, view));
            if (result != PM_SUCCESS) return result;
            std::vector<std::uint32_t> pixels;
            result = copy_device(services, view.pixels.data,
                                 view.pixels.size, pixels);
            if (result != PM_SUCCESS) return result;
            return upload_records(services, copied, pixels,
                view.width * view.height, view.revision);
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyConstraintStates(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    std::uint32_t first, std::uint32_t count,
    const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    const PmDeviceSize bytes = static_cast<PmDeviceSize>(count) *
        sizeof(PmConstraintState);
    PmResult result = validate_destination(command_buffer, destination, bytes);
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device || count == 0U ||
        first > simulation->constraints.size() ||
        count > simulation->constraints.size() - first)
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, first, count, copied](
            parallel_mater::device::Services &services) noexcept {
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            std::vector<PmConstraintState> states(count);
            for (std::uint32_t offset = 0U; offset < count; ++offset) {
                const auto &slot = simulation->constraints[first + offset];
                if (!slot.has_value()) {
                    states[offset] = {PM_FALSE, PM_FALSE, 0.0F};
                    continue;
                }
                parallel_mater::RigidConstraintState state{};
                PmResult result = parallel_mater::api::from_status(
                    simulation->runtime->read_rigid_constraint_state(
                        *slot, state, queue));
                if (result != PM_SUCCESS) return result;
                states[offset] = {state.enabled ? PM_TRUE : PM_FALSE,
                    state.broken ? PM_TRUE : PM_FALSE, state.applied_impulse};
            }
            return upload_bytes(services, copied,
                states.data(), states.size() * sizeof(PmConstraintState));
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdQueryHitBox(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    const PmHitBox *box, const PmBufferRange *destination) {
    const PmResult recording =
        parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmDeviceSize maximum = sizeof(PmHitBoxOutputHeader) +
        (simulation != nullptr
             ? static_cast<PmDeviceSize>(simulation->rigid_bodies.size()) *
                   sizeof(std::uint32_t)
             : 0U);
    if (simulation != nullptr)
        for (const PmFluidCapacity capacity : simulation->fluids)
            maximum += static_cast<PmDeviceSize>(capacity.particleCapacity) *
                sizeof(PmHitBoxParticle);
    PmResult result = validate_destination(command_buffer, destination, maximum);
    if (result != PM_SUCCESS || simulation == nullptr || box == nullptr ||
        simulation->device != command_buffer->device)
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmHitBox copied_box = *box;
    const PmBufferRange copied_destination = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, copied_box, copied_destination](
            parallel_mater::device::Services &services) noexcept {
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::HitBox query{};
            query.center = {copied_box.center[0], copied_box.center[1],
                            copied_box.center[2]};
            query.orientation = {copied_box.orientation[0],
                copied_box.orientation[1], copied_box.orientation[2],
                copied_box.orientation[3]};
            query.half_extents = {copied_box.halfExtents[0],
                copied_box.halfExtents[1], copied_box.halfExtents[2]};
            parallel_mater::HitBoxResult matches{};
            PmResult result = parallel_mater::api::from_status(
                simulation->runtime->query_hit_box(query, matches, queue));
            if (result != PM_SUCCESS) return result;
            const PmDeviceSize bytes = sizeof(PmHitBoxOutputHeader) +
                matches.rigid_bodies.size() * sizeof(std::uint32_t) +
                matches.particles.size() * sizeof(PmHitBoxParticle);
            if (bytes > copied_destination.size)
                return PM_ERROR_CAPACITY_EXCEEDED;
            std::vector<std::byte> packed;
            try { packed.resize(static_cast<std::size_t>(bytes)); }
            catch (const std::bad_alloc &) { return PM_ERROR_OUT_OF_HOST_MEMORY; }
            const PmHitBoxOutputHeader header{
                static_cast<std::uint32_t>(matches.rigid_bodies.size()),
                static_cast<std::uint32_t>(matches.particles.size()),
                simulation->submitted_steps};
            std::memcpy(packed.data(), &header, sizeof(header));
            auto *rigid = reinterpret_cast<std::uint32_t *>(
                packed.data() + sizeof(header));
            for (std::size_t index = 0U; index < matches.rigid_bodies.size();
                 ++index)
                rigid[index] = rigid_slot(simulation,
                                          matches.rigid_bodies[index]);
            auto *particles = reinterpret_cast<PmHitBoxParticle *>(
                rigid + matches.rigid_bodies.size());
            for (std::size_t index = 0U; index < matches.particles.size();
                 ++index) {
                particles[index] = {matches.particles[index].fluid.index,
                                    matches.particles[index].stable_particle_id};
            }
            return upload_bytes(services, copied_destination,
                packed.data(), packed.size());
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyContactEvents(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmResult result = validate_destination(command_buffer, destination,
                                           sizeof(PmDomainOutputHeader));
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device)
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, copied](
            parallel_mater::device::Services &services) noexcept {
            const auto rigid_view = simulation->runtime->rigid_contacts();
            const auto liquid_view = simulation->runtime->contacts();
            std::vector<parallel_mater::RigidContactEvent> rigid;
            std::vector<parallel_mater::ContactEvent> liquid;
            PmResult result = copy_device(services, rigid_view.events.data,
                                          rigid_view.event_count, rigid);
            if (result == PM_SUCCESS)
                result = copy_device(services, liquid_view.events.data,
                                     liquid_view.event_count, liquid);
            if (result != PM_SUCCESS) return result;
            std::vector<PmContactEvent> records;
            try { records.reserve(rigid.size() + liquid.size()); }
            catch (const std::bad_alloc &) { return PM_ERROR_OUT_OF_HOST_MEMORY; }
            for (const auto &source : rigid) {
                PmContactEvent target{};
                target.type = PM_CONTACT_RIGID_RIGID;
                target.first = rigid_slot(simulation, source.body);
                target.second = rigid_slot(simulation, source.collider);
                std::memcpy(&target.position, &source.position, sizeof(Vec3));
                std::memcpy(&target.normal, &source.normal, sizeof(Vec3));
                target.penetration = source.penetration;
                target.normalImpulse = source.normal_impulse;
                std::memcpy(&target.frictionImpulse, &source.friction_impulse,
                            sizeof(Vec3));
                records.push_back(target);
            }
            for (const auto &source : liquid) {
                PmContactEvent target{};
                target.type = PM_CONTACT_LIQUID_RIGID;
                target.first = source.fluid.index;
                target.second = rigid_slot(simulation, source.rigid_body);
                std::memcpy(&target.position, &source.position, sizeof(Vec3));
                std::memcpy(&target.normal, &source.normal, sizeof(Vec3));
                target.normalImpulse = source.normal_impulse;
                records.push_back(target);
            }
            return upload_records(services, copied, records,
                simulation->info.contactCapacity,
                std::max(rigid_view.frame_index, liquid_view.frame_index));
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyDebugData(
    PmCommandBuffer command_buffer, PmSimulation simulation,
    const PmBufferRange *destination) {
    const PmResult recording = parallel_mater::api::require_recording(command_buffer);
    if (recording != PM_SUCCESS) return recording;
    PmResult result = validate_destination(command_buffer, destination,
                                           sizeof(PmDomainOutputHeader));
    if (result != PM_SUCCESS || simulation == nullptr ||
        simulation->device != command_buffer->device)
        return result != PM_SUCCESS ? result : PM_ERROR_INVALID_ARGUMENT;
    const PmBufferRange copied = *destination;
    return parallel_mater::api::record_simulation(
        command_buffer, simulation, [simulation, copied](
            parallel_mater::device::Services &services) noexcept {
            parallel_mater::PhysicsDebugFrameView frame{};
            PmResult result = parallel_mater::api::from_status(
                simulation->runtime->physics_debug_frame(frame));
            if (result != PM_SUCCESS) return result;
            std::vector<PmDebugRecord> records;
            try {
                records.reserve(frame.rigid_bodies.size +
                    frame.fluid_particles.size + frame.cloth_vertices.size +
                    frame.rope_nodes.size + frame.soft_body_nodes.size);
            } catch (const std::bad_alloc &) {
                return PM_ERROR_OUT_OF_HOST_MEMORY;
            }
            for (std::size_t i = 0U; i < frame.rigid_bodies.size; ++i) {
                const auto &source = frame.rigid_bodies.data[i];
                PmDebugRecord target{};
                target.type = PM_DEBUG_RIGID;
                target.objectIndex = rigid_slot(simulation, source.id);
                std::memcpy(&target.position, &source.state.position, sizeof(Vec3));
                std::memcpy(&target.velocity, &source.state.linear_velocity,
                            sizeof(Vec3));
                std::memcpy(&target.force, &source.applied_force, sizeof(Vec3));
                records.push_back(target);
            }
            for (std::size_t i = 0U; i < frame.fluid_particles.size; ++i) {
                const auto &source = frame.fluid_particles.data[i];
                PmDebugRecord target{};
                target.type = PM_DEBUG_LIQUID;
                target.objectIndex = source.fluid.index;
                target.elementIndex = source.stable_particle_id;
                std::memcpy(&target.position, &source.position, sizeof(Vec3));
                std::memcpy(&target.velocity, &source.velocity, sizeof(Vec3));
                std::memcpy(&target.force, &source.acceleration, sizeof(Vec3));
                records.push_back(target);
            }
            for (std::size_t i = 0U; i < frame.cloth_vertices.size; ++i) {
                const auto &source = frame.cloth_vertices.data[i];
                PmDebugRecord target{};
                target.type = PM_DEBUG_CLOTH;
                target.objectIndex = source.cloth.index;
                target.elementIndex = source.vertex;
                std::memcpy(&target.position, &source.position, sizeof(Vec3));
                std::memcpy(&target.velocity, &source.velocity, sizeof(Vec3));
                std::memcpy(&target.force, &source.rigid_contact_force, sizeof(Vec3));
                records.push_back(target);
            }
            for (std::size_t i = 0U; i < frame.rope_nodes.size; ++i) {
                const auto &source = frame.rope_nodes.data[i];
                PmDebugRecord target{};
                target.type = PM_DEBUG_ROPE;
                target.objectIndex = source.rope.index;
                target.elementIndex = source.node;
                std::memcpy(&target.position, &source.position, sizeof(Vec3));
                std::memcpy(&target.velocity, &source.velocity, sizeof(Vec3));
                std::memcpy(&target.force, &source.constraint_force, sizeof(Vec3));
                records.push_back(target);
            }
            for (std::size_t i = 0U; i < frame.soft_body_nodes.size; ++i) {
                const auto &source = frame.soft_body_nodes.data[i];
                PmDebugRecord target{};
                target.type = PM_DEBUG_SOFT_BODY;
                target.objectIndex = source.soft_body.index;
                target.elementIndex = source.node;
                std::memcpy(&target.position, &source.position, sizeof(Vec3));
                std::memcpy(&target.velocity, &source.velocity, sizeof(Vec3));
                std::memcpy(&target.force, &source.rigid_contact_force, sizeof(Vec3));
                records.push_back(target);
            }
            return upload_records(services, copied, records,
                static_cast<std::uint32_t>(records.size()), frame.frame_index);
        });
}

namespace parallel_mater::api {

PmResult initialize_couplings(PmSimulation simulation,
                              PmPhysicsPipeline pipeline) noexcept {
    if (simulation == nullptr || pipeline == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (simulation->couplings_initialized) return PM_SUCCESS;
    for (const PmCouplingState &source : pipeline->couplings) {
        const auto &mechanical = source.parameters.mechanical;
        const auto &smoke = source.parameters.smoke;
        const auto &attachment = source.parameters.attachment;
        const auto &thermal = source.parameters.thermal;
        PmResult result = PM_SUCCESS;
        switch (source.type) {
        case PM_COUPLING_LIQUID_RIGID:
            // Rigid/liquid contact is a core edge in the shared scheduler.
            break;
        case PM_COUPLING_LIQUID_LIQUID:
            return PM_ERROR_FEATURE_NOT_PRESENT;
        case PM_COUPLING_LIQUID_CLOTH: {
            FluidId fluid{}; ClothId cloth{};
            if ((result = require_slot(simulation->fluid_ids, source.first, fluid)) != PM_SUCCESS ||
                (result = require_slot(simulation->cloths, source.second, cloth)) != PM_SUCCESS)
                return result;
            FluidClothCouplingId created{};
            result = from_status(simulation->runtime->add_fluid_cloth_coupling({
                .fluid = fluid, .cloth = cloth,
                .contact_distance = mechanical.contactDistance,
                .interaction_radius = mechanical.maximumAcceleration,
                .stiffness = mechanical.stiffness,
                .damping = mechanical.damping,
                .tangential_drag = mechanical.tangentialDrag,
                .maximum_force = mechanical.maximumForce,
                .enabled = true}, created));
            break;
        }
        case PM_COUPLING_LIQUID_ROPE: {
            FluidId fluid{}; RopeId rope{};
            if ((result = require_slot(simulation->fluid_ids, source.first, fluid)) != PM_SUCCESS ||
                (result = require_slot(simulation->ropes, source.second, rope)) != PM_SUCCESS)
                return result;
            FluidRopeCouplingId created{};
            result = from_status(simulation->runtime->add_fluid_rope_coupling({
                .fluid = fluid, .rope = rope,
                .contact_distance = mechanical.contactDistance,
                .friction = mechanical.friction,
                .maximum_rope_acceleration = mechanical.maximumAcceleration,
                .enabled = true}, created));
            break;
        }
        case PM_COUPLING_LIQUID_SOFT_BODY: {
            FluidId fluid{}; SoftBodyId body{};
            if ((result = require_slot(simulation->fluid_ids, source.first, fluid)) != PM_SUCCESS ||
                (result = require_slot(simulation->soft_bodies, source.second, body)) != PM_SUCCESS)
                return result;
            FluidSoftBodyCouplingId created{};
            result = from_status(simulation->runtime->add_fluid_soft_body_coupling({
                .fluid = fluid, .soft_body = body,
                .contact_distance = mechanical.contactDistance,
                .friction = mechanical.friction,
                .solver_iterations = 4U, .enabled = true}, created));
            break;
        }
        case PM_COUPLING_LIQUID_SMOKE: {
            FluidId fluid{}; SmokeId smoke_id{};
            if ((result = require_slot(simulation->fluid_ids, source.first, fluid)) != PM_SUCCESS ||
                (result = require_slot(simulation->smokes, source.second, smoke_id)) != PM_SUCCESS)
                return result;
            FluidSmokeCouplingId created{};
            result = from_status(simulation->runtime->add_fluid_smoke_coupling({
                .fluid = fluid, .smoke = smoke_id,
                .heater = {
                    .center = {thermal.heaterCenter[0], thermal.heaterCenter[1],
                               thermal.heaterCenter[2]},
                    .orientation = {thermal.heaterOrientation[0],
                                    thermal.heaterOrientation[1],
                                    thermal.heaterOrientation[2],
                                    thermal.heaterOrientation[3]},
                    .half_extents = {thermal.heaterHalfExtents[0],
                                     thermal.heaterHalfExtents[1]}},
                .heater_temperature = thermal.heaterTemperature,
                .boiling_temperature = thermal.boilingTemperature,
                .heat_transfer_rate = thermal.heatTransferRate,
                .wind_drag = thermal.windDrag,
                .steam_rise_speed = thermal.steamRiseSpeed}, created));
            break;
        }
        case PM_COUPLING_SOFT_BODY_CLOTH: {
            SoftBodyId body{}; ClothId cloth{};
            if ((result = require_slot(simulation->soft_bodies, source.first, body)) != PM_SUCCESS ||
                (result = require_slot(simulation->cloths, source.second, cloth)) != PM_SUCCESS)
                return result;
            SoftBodyClothCouplingId created{};
            result = from_status(simulation->runtime->add_soft_body_cloth_coupling({
                .soft_body = body, .cloth = cloth,
                .contact_distance = mechanical.contactDistance,
                .friction = mechanical.friction,
                .solver_iterations = 4U, .enabled = true}, created));
            break;
        }
        case PM_COUPLING_ROPE_SOFT_BODY: {
            RopeId rope{}; SoftBodyId body{};
            if ((result = require_slot(simulation->ropes, source.first, rope)) != PM_SUCCESS ||
                (result = require_slot(simulation->soft_bodies, source.second, body)) != PM_SUCCESS)
                return result;
            RopeSoftBodyCouplingId created{};
            result = from_status(simulation->runtime->add_rope_soft_body_coupling({
                .rope = rope, .soft_body = body,
                .contact_distance = mechanical.contactDistance,
                .friction = mechanical.friction,
                .maximum_soft_body_acceleration = attachment.maximumAcceleration,
                .anchor_support_radius_scale = attachment.supportRadiusScale,
                .anchor_contact_support_radius_scale =
                    attachment.contactSupportRadiusScale,
                .attach_first = attachment.attachFirst == PM_TRUE,
                .attach_last = attachment.attachLast == PM_TRUE,
                .enabled = true}, created));
            break;
        }
        case PM_COUPLING_ROPE_CLOTH: {
            RopeId rope{}; ClothId cloth{};
            if ((result = require_slot(simulation->ropes, source.first, rope)) != PM_SUCCESS ||
                (result = require_slot(simulation->cloths, source.second, cloth)) != PM_SUCCESS)
                return result;
            RopeClothCouplingId created{};
            result = from_status(simulation->runtime->add_rope_cloth_coupling({
                .rope = rope, .cloth = cloth,
                .first_vertex = attachment.firstVertex,
                .last_vertex = attachment.lastVertex,
                .anchor_effective_mass = attachment.effectiveMass,
                .maximum_cloth_acceleration = attachment.maximumAcceleration,
                .enabled = true}, created));
            break;
        }
        case PM_COUPLING_SMOKE_RIGID: {
            SmokeId smoke_id{}; RigidBodyId body{};
            if ((result = require_slot(simulation->smokes, source.first, smoke_id)) != PM_SUCCESS ||
                (result = require_slot(simulation->rigid_bodies, source.second, body)) != PM_SUCCESS)
                return result;
            SmokeRigidCouplingId created{};
            result = from_status(simulation->runtime->add_smoke_rigid_coupling({
                .smoke = smoke_id, .body = body,
                .air_density = smoke.airDensity,
                .drag_coefficient = smoke.dragCoefficient,
                .contact_distance = smoke.contactDistance,
                .tracer_contact = smoke.tracerContact == PM_TRUE,
                .enabled = true}, created));
            break;
        }
        case PM_COUPLING_SMOKE_CLOTH: {
            SmokeId smoke_id{}; ClothId cloth{};
            if ((result = require_slot(simulation->smokes, source.first, smoke_id)) != PM_SUCCESS ||
                (result = require_slot(simulation->cloths, source.second, cloth)) != PM_SUCCESS)
                return result;
            SmokeClothCouplingId created{};
            result = from_status(simulation->runtime->add_smoke_cloth_coupling({
                .smoke = smoke_id, .cloth = cloth,
                .wind_drag = smoke.windDrag,
                .maximum_wind_acceleration = smoke.maximumWindAcceleration,
                .contact_distance = smoke.contactDistance,
                .enabled = true}, created));
            break;
        }
        case PM_COUPLING_SMOKE_ROPE: {
            SmokeId smoke_id{}; RopeId rope{};
            if ((result = require_slot(simulation->smokes, source.first, smoke_id)) != PM_SUCCESS ||
                (result = require_slot(simulation->ropes, source.second, rope)) != PM_SUCCESS)
                return result;
            SmokeRopeCouplingId created{};
            result = from_status(simulation->runtime->add_smoke_rope_coupling({
                .smoke = smoke_id, .rope = rope,
                .wind_drag = smoke.windDrag,
                .maximum_wind_acceleration = smoke.maximumWindAcceleration,
                .contact_distance = smoke.contactDistance,
                .enabled = true}, created));
            break;
        }
        case PM_COUPLING_SMOKE_SOFT_BODY: {
            SmokeId smoke_id{}; SoftBodyId body{};
            if ((result = require_slot(simulation->smokes, source.first, smoke_id)) != PM_SUCCESS ||
                (result = require_slot(simulation->soft_bodies, source.second, body)) != PM_SUCCESS)
                return result;
            SmokeSoftBodyCouplingId created{};
            result = from_status(simulation->runtime->add_smoke_soft_body_coupling({
                .smoke = smoke_id, .soft_body = body,
                .wind_drag = smoke.windDrag,
                .maximum_wind_acceleration = smoke.maximumWindAcceleration,
                .contact_distance = smoke.contactDistance,
                .enabled = true}, created));
            break;
        }
        }
        if (result != PM_SUCCESS) return result;
    }
    simulation->couplings_initialized = true;
    return PM_SUCCESS;
}

} // namespace parallel_mater::api
