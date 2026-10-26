// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

void collide_fluid_with_static_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const void *previous, float *foam,
    const std::uint32_t *count, float radius, float spawn_clearance,
    std::uint32_t first_spawned, bool recover_spawn, const void *up,
    const void *body_parameters, const void *body_states, const void *meshes,
    const void *body_ids,
    std::uint32_t body_index, float particle_mass, bool collect_contacts,
    void *contact_samples, std::uint8_t *contact_flags,
    std::uint32_t fluid_index, std::uint32_t fluid_generation,
    bool apply_paint, const void *paint_fields,
    std::uint32_t paint_field_capacity, const void *paint_rules,
    std::uint32_t paint_rule_capacity) noexcept;

void calculate_fluid_body_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    const void *previous_states, const void *current_states,
    const void *meshes, std::uint32_t body_count, void *bounds) noexcept;

void index_fluid_body_cells(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    const void *bounds, std::uint32_t body_count, std::uint32_t words,
    float padding, void *masks, void *global_masks) noexcept;

void detect_fluid_moving_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *previous, const std::uint32_t *count, float radius,
    const void *body_parameters, const void *previous_states,
    const void *current_states, const void *meshes, const void *body_ids,
    std::uint32_t body_count, std::uint32_t body_topology_revision,
    const void *spatial_metadata, const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    bool first_iteration, void *contacts, std::uint32_t *contact_counts) noexcept;

void resolve_fluid_moving_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, float *foam, const std::uint32_t *count, float radius,
    float particle_mass, float timestep, float maximum_speed,
    const void *body_parameters, const void *body_states, const void *meshes,
    const void *contacts, const std::uint32_t *contact_counts, void *impulses,
    bool collect_contacts, void *contact_samples,
    std::uint8_t *particle_contact_flags, std::uint32_t fluid_index,
    std::uint32_t fluid_generation, const void *body_ids, bool apply_paint,
    const void *paint_fields, std::uint32_t paint_field_capacity,
    const void *paint_rules, std::uint32_t paint_rule_capacity) noexcept;

void reduce_fluid_body_impulses(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *impulses,
    const std::uint32_t *particle_count, const void *body_parameters,
    void *body_states, const std::uint32_t *contact_flags,
    std::uint32_t body_count, const void *position_corrections = nullptr) noexcept;

} // namespace parallel_mater::simulation
