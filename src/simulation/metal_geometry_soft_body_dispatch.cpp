// SPDX-License-Identifier: MIT
#include "geometry_soft_body_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"
#include <array>

namespace parallel_mater::simulation {
void collide_deformable_with_rigid_geometry(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *velocities, const void *previous, const float *inverse_masses,
    std::uint32_t count, float thickness, float timestep,
    const void *body_parameters, const void *previous_body_states,
    const void *body_states, const void *meshes, std::uint32_t body_count,
    void *impulses, void *contact_forces, void *contact_normals,
    void *contact_arms, void *contact_momentum_delta,
    float *accumulated_normal_delta,
    std::uint32_t *deformable_contact_count,
    std::uint32_t *dynamic_contact_flag, std::uint32_t *contact_flags,
    bool couple_dynamic, bool position_only, bool static_only,
    bool reconstruct_free_velocities, float maximum_speed,
    void *body_position_corrections) noexcept {
    if (auto *execution = metal::context(stream)) {
        if (execution->checked([&] { execution->meshes(meshes, body_parameters, body_count); }) != cudaSuccess) return;
    }
    (void)metal::launch(stream, "deformable_collide", grid,
        positions, velocities,
        previous,
        inverse_masses, count, thickness, timestep,
        body_parameters,
        previous_body_states,
        body_states,
        meshes,
        body_count, impulses,
        contact_forces, contact_normals,
        contact_arms, contact_momentum_delta,
        accumulated_normal_delta, deformable_contact_count,
        dynamic_contact_flag, contact_flags, couple_dynamic, position_only,
        static_only, reconstruct_free_velocities, maximum_speed,
        body_position_corrections);
}

void calculate_soft_body_surface_contacts(
    dim3 grid, dim3 block, cudaStream_t stream, const void *surface,
    const std::uint32_t *indices, std::uint32_t triangle_count,
    const void *body_parameters, const void *body_states,
    const void *meshes, std::uint32_t body_count, void *corrections) noexcept {
    if (auto *execution = metal::context(stream)) {
        if (execution->checked([&] { execution->meshes(meshes, body_parameters, body_count); }) != cudaSuccess) return;
    }
    (void)metal::launch(stream, "soft_body_surface_contacts", grid,
        surface,
        indices, triangle_count,
        body_parameters,
        body_states,
        meshes,
        body_count, corrections);
}

void apply_soft_body_surface_contacts(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    const void *corrections, const std::uint32_t *offsets,
    const void *influences, std::uint32_t node_count) noexcept {
    (void)metal::launch(stream, "soft_body_apply_surface_contacts", grid,
        positions,
        corrections,
        offsets,
        influences,
        node_count);
}

} // namespace parallel_mater::simulation
