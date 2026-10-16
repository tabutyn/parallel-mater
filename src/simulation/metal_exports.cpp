// SPDX-License-Identifier: MIT
#include "exports.hpp"
#include "driver/metal/schedule_context.hpp"
namespace parallel_mater::simulation {
PmResult query_rigid_hit_box(device::ExecutionQueue queue, const HitBox &box,
    const void *parameters, const void *states, const void *meshes,
    std::uint32_t count, std::uint8_t *matches) noexcept {
    auto *execution = metal::context(queue.native_handle);
    if (!execution) return PM_ERROR_INVALID_ARGUMENT;
    const auto result = execution->checked([&] {
        execution->meshes(meshes, parameters, count);
        const auto launched = metal::launch(queue.native_handle, "query_rigid_hit_box_kernel",
            (count + 255U) / 256U, box, parameters, states, meshes, count, matches);
        if (launched != cudaSuccess) throw std::runtime_error("Metal hit-box query failed");
        execution->wait();
    });
    return device::from_driver_error(result);
}
PmResult query_particle_hit_box(
    device::ExecutionQueue queue, const HitBox &box, const Vec3 *positions,
    std::uint32_t count, std::uint8_t *matches) noexcept {
    if (count == 0U) return PM_SUCCESS;
    if (!positions || !matches) return PM_ERROR_INVALID_ARGUMENT;
    return device::from_driver_error(metal::launch(queue.native_handle,
        "query_particle_hit_box_kernel", (count + 255U) / 256U,
        box, positions, count, matches));
}

PmResult export_fluid_particles(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *accelerations, const float *temperatures, const float *foam,
    const std::uint32_t *stable_ids, std::uint32_t count,
    std::uint32_t capacity, std::uint64_t revision, PmDomainOutputHeader *header,
    PmFluidParticleOutput *destination) noexcept {
    if (!header || (count && (!positions || !velocities || !accelerations ||
        !temperatures || !foam || !stable_ids || !destination))) return PM_ERROR_INVALID_ARGUMENT;
    return device::from_driver_error(metal::launch(queue.native_handle,
        "export_fluid_particles", (std::max(1U, count) + 255U) / 256U,
        positions, velocities, accelerations, temperatures, foam, stable_ids,
        count, capacity, revision, header, destination));
}

PmResult export_cloth_vertices(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *rigid_forces, const Vec3 *liquid_forces,
    const Vec3 *soft_body_forces, const Vec3 *rope_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmClothVertexState *destination) noexcept {
    if (!header || (count && (!positions || !velocities || !rigid_forces ||
        !liquid_forces || !soft_body_forces || !rope_forces || !destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    return device::from_driver_error(metal::launch(queue.native_handle,
        "export_cloth_vertices", (std::max(1U, count) + 255U) / 256U,
        positions, velocities, rigid_forces, liquid_forces, soft_body_forces, rope_forces,
        count, capacity, revision, header, destination));
}

PmResult export_surface_positions(
    device::ExecutionQueue queue, const Vec3 *positions,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmFloat3 *destination) noexcept {
    if (!header || (count && (!positions || !destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    return device::from_driver_error(metal::launch(queue.native_handle,
        "export_surface_positions", (std::max(1U, count) + 255U) / 256U,
        positions,
        count, capacity, revision, header, destination));
}

PmResult export_rope_nodes(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *constraint_forces, const Vec3 *contact_forces,
    const Vec3 *liquid_forces, const Vec3 *soft_body_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmRopeNodeState *destination) noexcept {
    if (!header || (count && (!positions || !velocities || !constraint_forces ||
        !contact_forces || !liquid_forces || !soft_body_forces || !destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    return device::from_driver_error(metal::launch(queue.native_handle,
        "export_rope_nodes", (std::max(1U, count) + 255U) / 256U,
        positions, velocities, constraint_forces, contact_forces, liquid_forces,
        soft_body_forces, count, capacity, revision, header, destination));
}

PmResult export_soft_body_nodes(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *rigid_forces, const Vec3 *cloth_forces,
    const Vec3 *liquid_forces, const Vec3 *rope_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmSoftBodyNodeState *destination) noexcept {
    if (!header || (count && (!positions || !velocities || !rigid_forces ||
        !cloth_forces || !liquid_forces || !rope_forces || !destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    return device::from_driver_error(metal::launch(queue.native_handle,
        "export_soft_body_nodes", (std::max(1U, count) + 255U) / 256U,
        positions, velocities, rigid_forces, cloth_forces, liquid_forces, rope_forces,
        count, capacity, revision, header, destination));
}

PmResult export_smoke_particles(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const float *ages, const float *number_densities, const float *pressures,
    const Vec3 *vorticities,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmSmokeParticleState *destination) noexcept {
    if (!header || (count && (!positions || !velocities || !ages ||
        !number_densities || !pressures || !vorticities || !destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    return device::from_driver_error(metal::launch(queue.native_handle,
        "export_smoke_particles", (std::max(1U, count) + 255U) / 256U,
        positions, velocities, ages, number_densities, pressures, vorticities,
        count, capacity, revision, header, destination));
}

} // namespace parallel_mater::simulation
