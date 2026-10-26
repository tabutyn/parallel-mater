// SPDX-License-Identifier: MIT
#pragma once

#include "smoke_dispatch.hpp"

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

void apply_smoke_wind_to_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *velocities, const float *inverse_masses, std::uint32_t vertex_count,
    const SmokeDispatchOptions &smoke, const void *smoke_positions,
    const void *smoke_velocities, const std::uint64_t *keys,
    const std::uint32_t *indices, const SmokeGridDispatchField &smoke_grid,
    float drag, float maximum_acceleration, float timestep) noexcept;

void collide_smoke_with_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    float *pressures, std::uint32_t count,
    const SmokeDispatchOptions &smoke, const void *cloth_positions,
    const void *cloth_velocities, const std::uint32_t *cloth_indices,
    std::uint32_t index_count, const void *minimum, const void *maximum,
    float clearance, const std::uint32_t *triangle_generations,
    std::uint32_t cloth_owner_slot, std::uint32_t cloth_owner_generation,
    std::uint32_t cloth_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    const SmokeGridDispatchField &smoke_grid) noexcept;

void apply_smoke_wind_to_soft_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *velocities, const float *inverse_masses, std::uint32_t node_count,
    const SmokeDispatchOptions &smoke, const void *smoke_positions,
    const void *smoke_velocities, const std::uint64_t *keys,
    const std::uint32_t *indices, const SmokeGridDispatchField &smoke_grid,
    float drag, float maximum_acceleration, float timestep,
    float maximum_speed) noexcept;

void collide_smoke_with_soft_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    std::uint32_t count, const SmokeDispatchOptions &smoke,
    const void *surface_positions, const void *surface_bindings,
    const std::uint32_t *surface_indices, std::uint32_t index_count,
    const void *node_velocities, const void *minimum, const void *maximum,
    float clearance, const std::uint32_t *triangle_generations,
    std::uint32_t soft_owner_slot, std::uint32_t soft_owner_generation,
    std::uint32_t soft_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    const SmokeGridDispatchField &smoke_grid) noexcept;

void apply_smoke_wind_to_rope(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *rope_positions,
    void *rope_velocities, std::uint32_t rope_count,
    float rope_maximum_speed, bool first_attached, bool last_attached,
    bool soft_first, bool soft_last, const SmokeDispatchOptions &smoke,
    const void *smoke_positions, const void *smoke_velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    const SmokeGridDispatchField &smoke_grid, float drag,
    float maximum_acceleration, float timestep) noexcept;

void collide_smoke_with_rope(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    std::uint32_t count, const SmokeDispatchOptions &smoke,
    const void *rope_positions, const void *rope_velocities,
    std::uint32_t rope_count, const void *minimum, const void *maximum,
    float clearance, const std::uint32_t *segment_generations,
    std::uint32_t rope_owner_slot, std::uint32_t rope_owner_generation,
    std::uint32_t rope_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status) noexcept;

} // namespace parallel_mater::simulation
