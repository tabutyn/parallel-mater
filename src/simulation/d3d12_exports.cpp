// SPDX-License-Identifier: MIT
#include "exports.hpp"

#include "driver/d3d12/schedule_context.hpp"

#include <array>
#include <cstring>

namespace parallel_mater::simulation {

namespace {
template <typename T>
void write(std::array<std::byte, 112> &bytes, std::size_t offset,
           const T &value) noexcept {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
} // namespace

PmResult query_rigid_hit_box(
    device::ExecutionQueue queue, const HitBox &box, const void *body_parameters,
    const void *body_states, const void *meshes, std::uint32_t body_count,
    std::uint8_t *matches) noexcept {
    auto *execution = d3d12::context(queue.native_handle);
    if (execution == nullptr || body_parameters == nullptr ||
        body_states == nullptr || meshes == nullptr || matches == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    std::array<std::byte, 112> constants{};
    write(constants, 0U, box.center);
    write(constants, 16U, box.orientation);
    write(constants, 32U, box.half_extents);
    write(constants, 48U, execution->address(body_parameters));
    write(constants, 64U, execution->address(body_states));
    write(constants, 80U, execution->translated_meshes(
                              meshes, body_parameters, body_count));
    write(constants, 92U, body_count);
    write(constants, 96U, execution->widened_flags(matches, body_count));
    cudaError_t result = execution->dispatch(
        "queries", "query_rigid_hit_box_kernel",
        dim3((body_count + 255U) / 256U), dim3(256U), constants);
    if (result == cudaSuccess) result = execution->synchronize();
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
