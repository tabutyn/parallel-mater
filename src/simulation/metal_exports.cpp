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
    device::ExecutionQueue, const HitBox &, const Vec3 *, std::uint32_t,
    std::uint8_t *) noexcept {
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

PmResult export_fluid_particles(
    device::ExecutionQueue, const Vec3 *, const Vec3 *, const Vec3 *,
    const float *, const float *, const std::uint32_t *, std::uint32_t,
    std::uint32_t, std::uint64_t, PmDomainOutputHeader *,
    PmFluidParticleOutput *) noexcept {
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

PmResult export_cloth_vertices(
    device::ExecutionQueue, const Vec3 *, const Vec3 *, const Vec3 *,
    const Vec3 *, const Vec3 *, const Vec3 *, std::uint32_t, std::uint32_t,
    std::uint64_t, PmDomainOutputHeader *, PmClothVertexState *) noexcept {
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

PmResult export_surface_positions(
    device::ExecutionQueue, const Vec3 *, std::uint32_t, std::uint32_t,
    std::uint64_t, PmDomainOutputHeader *, PmFloat3 *) noexcept {
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

PmResult export_rope_nodes(
    device::ExecutionQueue, const Vec3 *, const Vec3 *, const Vec3 *,
    const Vec3 *, const Vec3 *, const Vec3 *, std::uint32_t, std::uint32_t,
    std::uint64_t, PmDomainOutputHeader *, PmRopeNodeState *) noexcept {
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

PmResult export_soft_body_nodes(
    device::ExecutionQueue, const Vec3 *, const Vec3 *, const Vec3 *,
    const Vec3 *, const Vec3 *, const Vec3 *, std::uint32_t, std::uint32_t,
    std::uint64_t, PmDomainOutputHeader *, PmSoftBodyNodeState *) noexcept {
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

PmResult export_smoke_particles(
    device::ExecutionQueue, const Vec3 *, const Vec3 *, const float *,
    const float *, const float *, const Vec3 *, std::uint32_t, std::uint32_t,
    std::uint64_t, PmDomainOutputHeader *, PmSmokeParticleState *) noexcept {
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

} // namespace parallel_mater::simulation
