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

void apply_realtime_rope_soft_body_contacts(
    device::DispatchSize grid, device::DispatchSize,
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
    float timestep) noexcept {
    (void)metal::launch(stream, "rope_soft_realtime_contact", grid,
        rope_positions, rope_previous_positions, rope_velocities,
        rope_inverse_masses, rope_count, rope_maximum_speed, attach_first,
        attach_last, rope_forces, surface, previous_surface, indices,
        index_count, bindings,
        soft_velocities, soft_inverse_masses, orientation, distance, friction,
        triangle_generations, soft_owner_slot, soft_owner_generation,
        soft_topology_revision, spatial_metadata, spatial_admission,
        spatial_leaf_capacity, candidate_counts, candidate_sources,
        candidates_per_query, spatial_status, node_impulses, contact_count,
        maximum_penetration, timestep);
}

void sample_rope_soft_body_anchor(device::ExecutionQueue stream,
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

void scatter_rope_soft_body_anchor(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *anchor_positions,
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

void apply_rope_soft_body_impulses(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, const void *impulses, std::uint32_t count,
    float maximum_acceleration, float maximum_speed, float force_scale,
    float timestep, float frame_inverse_timestep) noexcept {
    (void)metal::launch(stream, "rope_soft_apply", grid,
        positions, velocities, forces,
        inverse_masses,
        impulses, count,
        maximum_acceleration, maximum_speed, force_scale, timestep,
        frame_inverse_timestep);
}

void sync_rope_soft_body_attachment(device::ExecutionQueue stream,
    void *rope_positions, void *rope_previous_positions,
    void *rope_velocities, std::uint32_t rope_count,
    void *rope_anchor_targets,
    const void *anchor_positions, const void *anchor_velocities,
    std::uint32_t end, float timestep) noexcept {
    (void)metal::launch(stream, "rope_soft_sync_attachment", 1U,
        rope_positions, rope_previous_positions, rope_velocities, rope_count,
        rope_anchor_targets, anchor_positions, anchor_velocities, end,
        timestep);
}

void accumulate_rope_soft_body_attachment_reaction(
    device::ExecutionQueue stream,
    const void *previous_rope_velocities,
    const void *current_rope_velocities,
    const float *rope_inverse_masses, std::uint32_t rope_count,
    std::uint32_t attached_end_mask, std::uint32_t attached_end_count,
    const float gravity[3], float timestep, void *soft_forces,
    const std::uint32_t *indices, const void *bindings,
    std::uint32_t triangle, const float weights[3]) noexcept {
    (void)metal::launch(stream,
        "rope_soft_accumulate_attachment_reaction", 1U,
        previous_rope_velocities, current_rope_velocities,
        rope_inverse_masses, rope_count, attached_end_mask,
        attached_end_count, vector(gravity), timestep, soft_forces, indices,
        bindings, triangle, vector(weights));
}

} // namespace parallel_mater::simulation
