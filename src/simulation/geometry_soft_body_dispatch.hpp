// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

void collide_deformable_with_rigid_geometry(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
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
    void *body_position_corrections = nullptr) noexcept;

void calculate_soft_body_surface_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    const std::uint32_t *indices, std::uint32_t triangle_count,
    const void *body_parameters, const void *body_states,
    const void *meshes, std::uint32_t body_count, void *corrections) noexcept;

void apply_soft_body_surface_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *corrections, const std::uint32_t *offsets,
    const void *influences, std::uint32_t node_count) noexcept;

void emit_deformable_spatial_primitives(
    device::DispatchSize grid, device::DispatchSize block,
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
    std::uint32_t *status) noexcept;

void filter_deformable_spatial_candidates(
    device::DispatchSize grid, device::DispatchSize block,
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
    std::uint32_t *filtered_sources, std::uint32_t *status) noexcept;

void emit_smoke_spatial_queries(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *cell_centers,
    std::uint32_t query_count, std::uint32_t query_offset,
    std::uint32_t query_capacity, std::uint32_t query_mask,
    const float half_extent[3], const float path_deviation[3],
    const float correction_budget[3], float epsilon, void *query_bounds,
    std::uint32_t *query_masks, std::uint32_t *excluded_sort_keys,
    std::uint32_t *status) noexcept;

void filter_smoke_spatial_obstacles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    std::uint32_t query_count, std::uint32_t accepted_family_mask,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *filtered_counts,
    std::uint32_t *filtered_sources, std::uint32_t *status) noexcept;

} // namespace parallel_mater::simulation
