// SPDX-License-Identifier: MIT
#include "geometry_cloth_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"
#include <array>

namespace parallel_mater::simulation {
void constrain_rigid_bodies_with_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *cloth_positions,
    const void *cloth_velocities, const std::uint32_t *cloth_indices,
    const std::uint32_t *vertex_sources, const void *surface_positions,
    const float *inverse_masses, std::uint32_t vertex_count,
    std::uint32_t triangle_count, float thickness, float timestep,
    float contact_friction, const void *body_parameters,
    const void *previous_body_states, void *body_states, const void *meshes,
    const void *body_ids, std::uint32_t body_count,
    std::uint32_t cloth_index, std::uint32_t cloth_generation,
    std::uint32_t cloth_topology_revision,
    const std::uint32_t *triangle_generations,
    const void *spatial_metadata, const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    const void *paint_fields, std::uint32_t paint_field_capacity,
    const void *paint_rules, std::uint32_t paint_rule_capacity,
    void *corrections) noexcept {
    if (auto *execution = metal::context(stream)) {
        if (execution->checked_result([&] { execution->meshes(meshes, body_parameters, body_count);
            execution->paint_fields(paint_fields, paint_field_capacity, meshes); }) != PM_SUCCESS) return;
    }
    (void)metal::launch(stream, "cloth_constrain_bodies", grid,
        cloth_positions,
        cloth_velocities,
        cloth_indices,
        vertex_sources,
        surface_positions,
        inverse_masses, vertex_count, triangle_count,
        thickness, timestep, contact_friction,
        body_parameters,
        previous_body_states,
        body_states,
        meshes,
        body_ids, body_count,
        cloth_index, cloth_generation,
        cloth_topology_revision, triangle_generations, spatial_metadata,
        spatial_admission, spatial_leaf_capacity, candidate_counts,
        candidate_sources, candidates_per_query, spatial_status,
        paint_fields, paint_field_capacity, paint_rules, paint_rule_capacity, corrections);
}

void apply_cloth_body_corrections(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const float *inverse_masses, std::uint32_t vertex_count,
    const void *corrections, std::uint32_t body_count) noexcept {
    (void)metal::launch(stream, "cloth_apply_body_corrections", grid,
        positions, velocities,
        inverse_masses, vertex_count,
        corrections,
        body_count);
}

} // namespace parallel_mater::simulation
