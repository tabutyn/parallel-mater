// SPDX-License-Identifier: MIT
#include "soft_body_cloth_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"
#include <array>

namespace parallel_mater::simulation {
void detect_soft_body_cloth_contacts(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions, const void *previous,
    const void *velocities, const float *inverse_masses,
    std::uint32_t count, const void *cloth_positions,
    const void *cloth_previous, const void *cloth_velocities,
    const float *cloth_inverse_masses, const std::uint32_t *indices,
    const void *surface, std::uint32_t triangle_count,
    const std::uint32_t *triangle_generations,
    std::uint32_t cloth_owner_slot, std::uint32_t cloth_owner_generation,
    std::uint32_t cloth_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission, std::uint32_t leaf_capacity,
    const std::uint32_t *query_leaf_sources,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t soft_owner_slot,
    std::uint32_t soft_owner_generation,
    std::uint32_t soft_topology_revision, std::uint32_t *status,
    float distance,
    float friction, void *contacts, std::uint32_t *contact_counts) noexcept {
    (void)metal::launch(stream, "soft_cloth_detect", grid,
        positions,
        previous,
        velocities,
        inverse_masses, count,
        cloth_positions,
        cloth_previous,
        cloth_velocities,
        cloth_inverse_masses,
        indices,
        surface, triangle_count, triangle_generations, cloth_owner_slot,
        cloth_owner_generation, cloth_topology_revision, spatial_metadata,
        spatial_admission, leaf_capacity, query_leaf_sources,
        candidate_counts, candidate_sources, candidates_per_query,
        soft_owner_slot, soft_owner_generation, soft_topology_revision,
        status, distance,
        friction, contacts, contact_counts);
}

void apply_soft_body_cloth_contacts_to_soft_body(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *previous_positions,
    void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    const std::uint32_t *contact_counts, float timestep,
    float maximum_speed) noexcept {
    (void)metal::launch(stream, "soft_cloth_apply_soft", grid,
        positions, previous_positions, velocities, forces,
        inverse_masses, count,
        contacts,
        contact_counts, timestep, maximum_speed);
}

void apply_soft_body_cloth_contacts_to_cloth(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *previous_positions,
    void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    std::uint32_t contact_count, const std::uint32_t *contact_counts,
    float timestep) noexcept {
    (void)metal::launch(stream, "soft_cloth_apply_cloth", grid,
        positions, previous_positions, velocities, forces,
        inverse_masses, count,
        contacts,
        contact_count, contact_counts, timestep);
}

} // namespace parallel_mater::simulation
