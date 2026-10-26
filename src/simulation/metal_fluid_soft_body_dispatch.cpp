// SPDX-License-Identifier: MIT
#include "fluid_soft_body_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"
#include <array>

namespace parallel_mater::simulation {
void refit_fluid_soft_surface(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    const void *previous_surface, const std::uint32_t *surface_indices,
    const std::uint32_t *triangle_order, void *tree,
    const std::uint32_t *parents, std::uint32_t *ready,
    std::uint32_t tree_count, void *bounds) noexcept {
    (void)metal::launch(stream, "fluid_soft_refit", grid,
        surface,
        previous_surface,
        surface_indices,
        triangle_order, tree,
        parents, ready, tree_count,
        bounds);
}

void detect_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *previous, const void *velocities,
    const std::uint32_t *particle_count, const void *surface,
    const void *previous_surface, const std::uint32_t *surface_indices,
    std::uint32_t surface_index_count, const void *surface_bindings,
    const void *soft_velocities, float orientation, float distance,
    bool sweep, const std::uint32_t *triangle_generations,
    std::uint32_t soft_owner_slot, std::uint32_t soft_owner_generation,
    std::uint32_t soft_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    void *contacts,
    std::uint32_t *node_contact_counts, std::uint32_t *contact_count,
    float *maximum_penetration) noexcept {
    (void)metal::launch(stream, "fluid_soft_detect", grid,
        positions,
        previous,
        velocities,
        particle_count,
        surface,
        previous_surface,
        surface_indices, surface_index_count,
        surface_bindings,
        soft_velocities, orientation, distance, sweep, triangle_generations,
        soft_owner_slot, soft_owner_generation, soft_topology_revision,
        spatial_metadata, spatial_admission, spatial_leaf_capacity,
        candidate_counts, candidate_sources, candidates_per_query,
        spatial_status,
        contacts, node_contact_counts, contact_count,
        maximum_penetration);
}

void solve_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, void *accelerations, float *foam,
    const std::uint32_t *particle_count, const void *contacts,
    const std::uint32_t *node_contact_counts, const float *inverse_masses,
    const void *soft_velocities, float maximum_soft_speed,
    float inverse_particle_mass, float friction, float maximum_projection,
    float timestep, void *node_position_deltas, void *node_impulses) noexcept {
    (void)metal::launch(stream, "fluid_soft_solve", grid,
        positions, velocities,
        accelerations, foam,
        particle_count,
        contacts,
        node_contact_counts,
        inverse_masses,
        soft_velocities, maximum_soft_speed,
        inverse_particle_mass, friction, maximum_projection, timestep,
        node_position_deltas, node_impulses);
}

void apply_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const float *inverse_masses,
    const void *position_deltas, const void *impulses, void *forces,
    std::uint32_t node_count, float inverse_frame_timestep) noexcept {
    (void)metal::launch(stream, "fluid_soft_apply", grid,
        positions, velocities,
        inverse_masses,
        position_deltas,
        impulses, forces,
        node_count, inverse_frame_timestep);
}

void recover_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous, const std::uint32_t *particle_count,
    const void *surface, const void *previous_surface,
    const std::uint32_t *surface_indices, std::uint32_t surface_index_count,
    float orientation, float distance, bool sweep,
    const std::uint32_t *triangle_generations,
    std::uint32_t soft_owner_slot, std::uint32_t soft_owner_generation,
    std::uint32_t soft_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query,
    std::uint32_t *spatial_status) noexcept {
    (void)metal::launch(stream, "fluid_soft_recover", grid,
        positions, previous,
        particle_count,
        surface,
        previous_surface,
        surface_indices, surface_index_count, orientation, distance, sweep,
        triangle_generations, soft_owner_slot, soft_owner_generation,
        soft_topology_revision, spatial_metadata, spatial_admission,
        spatial_leaf_capacity, candidate_counts, candidate_sources,
        candidates_per_query, spatial_status);
}

} // namespace parallel_mater::simulation
