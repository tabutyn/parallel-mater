// SPDX-License-Identifier: MIT
#include "geometry_soft_body_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"
#include <array>

namespace parallel_mater::simulation {
namespace {
Vec3 spatial_vector(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}
} // namespace

void collide_deformable_with_rigid_geometry(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const void *previous, const float *inverse_masses,
    std::uint32_t count, float thickness, float timestep,
    const void *body_parameters, const void *previous_body_states,
    const void *body_states, const void *meshes, std::uint32_t body_count,
    const void *body_ids, std::uint32_t body_topology_revision,
    const void *spatial_metadata, const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    std::uint32_t first_excluded_body, std::uint32_t first_excluded_count,
    std::uint32_t last_excluded_body, std::uint32_t last_excluded_count,
    void *impulses, void *contact_forces, void *contact_normals,
    void *contact_arms, void *contact_momentum_delta,
    float *accumulated_normal_delta,
    std::uint32_t *deformable_contact_count,
    std::uint32_t *dynamic_contact_flag, std::uint32_t *contact_flags,
    bool couple_dynamic, bool position_only, bool static_only,
    bool reconstruct_free_velocities, float maximum_speed,
    void *body_position_corrections) noexcept {
    if (auto *execution = metal::context(stream)) {
        if (execution->checked_result([&] { execution->meshes(meshes, body_parameters, body_count); }) != PM_SUCCESS) return;
    }
    (void)metal::launch(stream, "deformable_collide", grid,
        positions, velocities,
        previous,
        inverse_masses, count, thickness, timestep,
        body_parameters,
        previous_body_states,
        body_states,
        meshes,
        body_count, body_ids, body_topology_revision, spatial_metadata,
        spatial_admission, spatial_leaf_capacity, candidate_counts,
        candidate_sources, candidates_per_query, spatial_status,
        first_excluded_body, first_excluded_count,
        last_excluded_body, last_excluded_count, impulses,
        contact_forces, contact_normals,
        contact_arms, contact_momentum_delta,
        accumulated_normal_delta, deformable_contact_count,
        dynamic_contact_flag, contact_flags, couple_dynamic, position_only,
        static_only, reconstruct_free_velocities, maximum_speed,
        body_position_corrections);
}

void calculate_soft_body_surface_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    const std::uint32_t *indices, std::uint32_t triangle_count,
    const void *body_parameters, const void *body_states,
    const void *meshes, std::uint32_t body_count, void *corrections) noexcept {
    if (auto *execution = metal::context(stream)) {
        if (execution->checked_result([&] { execution->meshes(meshes, body_parameters, body_count); }) != PM_SUCCESS) return;
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
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *corrections, const std::uint32_t *offsets,
    const void *influences, std::uint32_t node_count) noexcept {
    (void)metal::launch(stream, "soft_body_apply_surface_contacts", grid,
        positions,
        corrections,
        offsets,
        influences,
        node_count);
}

void emit_deformable_spatial_primitives(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *previous_positions,
    const void *predicted_positions, std::uint32_t vertex_count,
    const std::uint32_t *indices,
    const std::uint32_t *primitive_generations,
    std::uint32_t primitive_count, std::uint32_t vertices_per_primitive,
    std::uint32_t family, std::uint32_t owner_slot,
    std::uint32_t owner_generation, std::uint32_t topology_revision,
    std::uint32_t primitive_kind, std::uint32_t leaf_offset,
    std::uint32_t leaf_capacity, std::uint32_t query_offset,
    std::uint32_t query_capacity, std::uint32_t sort_key_base,
    std::uint32_t query_mask, std::uint32_t admission_mask, float thickness,
    const float path_deviation[3], const float correction_budget[3],
    float epsilon, void *metadata, void *bounds, std::uint8_t *build_flags,
    void *query_bounds, std::uint32_t *query_masks,
    std::uint32_t *excluded_sort_keys, std::uint32_t *query_leaf_sources,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "pm_emit_deformable_spatial_primitives", grid,
        previous_positions, predicted_positions, vertex_count, indices,
        primitive_generations, primitive_count, vertices_per_primitive,
        family, owner_slot, owner_generation, topology_revision,
        primitive_kind, leaf_offset, leaf_capacity, query_offset,
        query_capacity, sort_key_base, query_mask, admission_mask, thickness,
        spatial_vector(path_deviation), spatial_vector(correction_budget),
        epsilon, metadata, bounds, build_flags, query_bounds, query_masks,
        excluded_sort_keys, query_leaf_sources, status);
}

void filter_deformable_spatial_candidates(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    const std::uint32_t *query_leaf_sources, std::uint32_t query_count,
    std::uint32_t query_family, std::uint32_t query_owner_slot,
    std::uint32_t query_owner_generation,
    std::uint32_t query_topology_revision,
    std::uint32_t query_primitive_kind,
    const std::uint32_t *query_primitive_generations,
    std::uint32_t query_primitive_count,
    const std::uint32_t *query_indices,
    std::uint32_t query_vertices_per_primitive,
    std::uint32_t target_family, std::uint32_t target_owner_slot,
    std::uint32_t target_owner_generation,
    std::uint32_t target_topology_revision,
    std::uint32_t target_primitive_kind,
    const std::uint32_t *target_primitive_generations,
    std::uint32_t target_primitive_count,
    const std::uint32_t *target_indices,
    std::uint32_t target_vertices_per_primitive,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *filtered_counts,
    std::uint32_t *filtered_sources, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "pm_filter_deformable_spatial_candidates",
        grid, metadata, admission, leaf_capacity, query_leaf_sources,
        query_count, query_family, query_owner_slot, query_owner_generation,
        query_topology_revision, query_primitive_kind,
        query_primitive_generations, query_primitive_count, query_indices,
        query_vertices_per_primitive, target_family, target_owner_slot,
        target_owner_generation, target_topology_revision,
        target_primitive_kind, target_primitive_generations,
        target_primitive_count, target_indices,
        target_vertices_per_primitive, candidate_counts, candidate_sources,
        candidates_per_query, filtered_counts, filtered_sources, status);
}

void emit_smoke_spatial_queries(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *cell_centers,
    std::uint32_t query_count, std::uint32_t query_offset,
    std::uint32_t query_capacity, std::uint32_t query_mask,
    const float half_extent[3], const float path_deviation[3],
    const float correction_budget[3], float epsilon, void *query_bounds,
    std::uint32_t *query_masks, std::uint32_t *excluded_sort_keys,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "pm_emit_smoke_spatial_queries", grid,
        cell_centers, query_count, query_offset, query_capacity, query_mask,
        spatial_vector(half_extent), spatial_vector(path_deviation),
        spatial_vector(correction_budget), epsilon, query_bounds, query_masks,
        excluded_sort_keys, status);
}

void filter_smoke_spatial_obstacles(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    std::uint32_t query_count, std::uint32_t accepted_family_mask,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *filtered_counts,
    std::uint32_t *filtered_sources, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "pm_filter_smoke_spatial_obstacles", grid,
        metadata, admission, leaf_capacity, query_count, accepted_family_mask,
        candidate_counts, candidate_sources, candidates_per_query,
        filtered_counts, filtered_sources, status);
}

} // namespace parallel_mater::simulation
