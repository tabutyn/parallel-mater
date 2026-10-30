// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

void apply_realtime_rope_soft_body_contacts(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *rope_positions,
    void *rope_previous_positions, void *rope_velocities,
    const float *rope_inverse_masses, std::uint32_t rope_count,
    float rope_maximum_speed, bool attach_first, bool attach_last,
    void *rope_forces, const void *surface, const void *previous_surface,
    const std::uint32_t *indices, std::uint32_t index_count,
    const void *bindings,
    const void *soft_velocities, const float *soft_inverse_masses,
    float orientation, float distance, float friction,
    const std::uint32_t *triangle_generations,
    std::uint32_t soft_owner_slot, std::uint32_t soft_owner_generation,
    std::uint32_t soft_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    void *node_impulses,
    std::uint32_t *contact_count, float *maximum_penetration,
    float timestep) noexcept;

void sample_rope_cloth_anchor(device::ExecutionQueue stream, void *anchor_positions,
    void *anchor_velocities, void *anchor_impulses,
    float *anchor_inverse_masses, std::uint32_t end, const void *positions,
    const void *velocities, const float *inverse_masses,
    std::uint32_t vertex, float effective_mass) noexcept;

void apply_rope_cloth_anchor(device::ExecutionQueue stream, void *rope_positions,
    void *rope_velocities, std::uint32_t rope_count, void *anchor_positions,
    void *anchor_velocities, const void *anchor_impulses,
    const void *rope_constraint_forces, std::uint32_t end, void *positions,
    void *velocities, void *forces, const float *inverse_masses,
    std::uint32_t vertex, float maximum_acceleration,
    float constraint_force_scale, float timestep,
    float frame_inverse_timestep) noexcept;

void accumulate_rope_cloth_reaction(device::ExecutionQueue stream,
    const void *previous_velocities, const void *current_velocities,
    const float *inverse_masses, std::uint32_t rope_count,
    std::uint32_t attached_end_mask, std::uint32_t attached_end_count,
    const float gravity[3], float timestep, void *cloth_forces,
    std::uint32_t cloth_vertex) noexcept;

void sample_rope_soft_body_anchor(device::ExecutionQueue stream,
    void *anchor_positions, void *anchor_velocities, void *anchor_impulses,
    float *anchor_inverse_masses, std::uint32_t end, const void *surface,
    const void *velocities, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    const float offset[3]) noexcept;

void scatter_rope_soft_body_anchor(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *anchor_positions,
    const void *anchor_impulses, std::uint32_t end,
    const void *node_positions, std::uint32_t node_count,
    float support_radius, float contact_support_radius,
    const std::uint32_t *contact_count, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    void *node_impulses) noexcept;

void apply_rope_soft_body_impulses(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, const void *impulses, std::uint32_t count,
    float maximum_acceleration, float maximum_speed, float force_scale,
    float timestep, float frame_inverse_timestep) noexcept;

void sync_rope_soft_body_attachment(device::ExecutionQueue stream,
    void *rope_positions, void *rope_previous_positions,
    void *rope_velocities, std::uint32_t rope_count,
    void *rope_anchor_targets,
    const void *anchor_positions, const void *anchor_velocities,
    std::uint32_t end, float timestep) noexcept;

void accumulate_rope_soft_body_attachment_reaction(
    device::ExecutionQueue stream,
    const void *previous_rope_velocities,
    const void *current_rope_velocities,
    const float *rope_inverse_masses, std::uint32_t rope_count,
    std::uint32_t attached_end_mask, std::uint32_t attached_end_count,
    const float gravity[3], float timestep, void *soft_forces,
    const std::uint32_t *indices, const void *bindings,
    std::uint32_t triangle, const float weights[3]) noexcept;

} // namespace parallel_mater::simulation
