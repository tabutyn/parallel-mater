// SPDX-License-Identifier: MIT
#include "rope_coupling_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
Vec3 vector(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}

} // namespace

void sample_rope_soft_body_anchor(cudaStream_t stream,
    void *anchor_positions, void *anchor_velocities, void *anchor_impulses,
    float *anchor_inverse_masses, std::uint32_t end, const void *surface,
    const void *velocities, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    const float offset[3]) noexcept {
    (void)metal::launch(stream, "rope_soft_sample_anchor", 1U,
        anchor_positions, anchor_velocities,
        anchor_impulses, anchor_inverse_masses, end,
        surface,
        velocities,
        indices,
        bindings,
        triangle, vector(weights), vector(offset));
}

void scatter_rope_soft_body_anchor(dim3 grid, dim3 block,
    cudaStream_t stream, const void *anchor_positions,
    const void *anchor_impulses, std::uint32_t end,
    const void *node_positions, std::uint32_t node_count,
    float support_radius, float contact_support_radius,
    const std::uint32_t *contact_count, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    void *node_impulses) noexcept {
    (void)metal::launch(stream, "rope_soft_scatter_anchor", grid,
        anchor_positions,
        anchor_impulses, end,
        node_positions, node_count,
        support_radius, contact_support_radius,
        contact_count,
        indices,
        bindings,
        triangle, vector(weights), node_impulses);
}

void apply_rope_soft_body_impulses(dim3 grid, dim3 block,
    cudaStream_t stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, const void *impulses, std::uint32_t count,
    float maximum_acceleration, float maximum_speed, float timestep,
    float frame_inverse_timestep) noexcept {
    (void)metal::launch(stream, "rope_soft_apply", grid,
        positions, velocities, forces,
        inverse_masses,
        impulses, count,
        maximum_acceleration, maximum_speed, timestep,
        frame_inverse_timestep);
}

} // namespace parallel_mater::simulation
