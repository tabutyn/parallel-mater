// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

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
    float friction, void *contacts, std::uint32_t *contact_counts) noexcept;

void apply_soft_body_cloth_contacts_to_soft_body(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    const std::uint32_t *contact_counts, float timestep,
    float maximum_speed) noexcept;

void apply_soft_body_cloth_contacts_to_cloth(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    std::uint32_t contact_count, const std::uint32_t *contact_counts,
    float timestep) noexcept;

} // namespace parallel_mater::simulation
