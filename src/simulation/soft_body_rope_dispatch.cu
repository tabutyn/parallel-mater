// SPDX-License-Identifier: MIT
#include "rope_coupling_dispatch.hpp"

#include "parallel_mater_soft_body_rope_runtime.cu"

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
Vec3_0 vector(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}

} // namespace

void sample_rope_soft_body_anchor(device::ExecutionQueue stream,
    void *anchor_positions, void *anchor_velocities, void *anchor_impulses,
    float *anchor_inverse_masses, std::uint32_t end, const void *surface,
    const void *velocities, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    const float offset[3]) noexcept {
    rope_soft_sample_anchor<<<1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(anchor_positions), as<Vec3_0>(anchor_velocities),
        as<Vec3_0>(anchor_impulses), anchor_inverse_masses, end,
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<std::uint32_t *>(indices),
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        triangle, vector(weights), vector(offset));
}

void scatter_rope_soft_body_anchor(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *anchor_positions,
    const void *anchor_impulses, std::uint32_t end,
    const void *node_positions, std::uint32_t node_count,
    float support_radius, float contact_support_radius,
    const std::uint32_t *contact_count, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    void *node_impulses) noexcept {
    rope_soft_scatter_anchor<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(anchor_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(anchor_impulses)), end,
        const_cast<Vec3_0 *>(as<Vec3_0>(node_positions)), node_count,
        support_radius, contact_support_radius,
        const_cast<std::uint32_t *>(contact_count),
        const_cast<std::uint32_t *>(indices),
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        triangle, vector(weights), as<Vec3_0>(node_impulses));
}

void apply_rope_soft_body_impulses(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, const void *impulses, std::uint32_t count,
    float maximum_acceleration, float maximum_speed, float timestep,
    float frame_inverse_timestep) noexcept {
    rope_soft_apply<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities), as<Vec3_0>(forces),
        const_cast<float *>(inverse_masses),
        const_cast<Vec3_0 *>(as<Vec3_0>(impulses)), count,
        maximum_acceleration, maximum_speed, timestep,
        frame_inverse_timestep);
}

} // namespace parallel_mater::simulation
