// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

void detect_fluid_rope_contacts(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *particle_positions, void *particle_velocities,
    const std::uint32_t *particle_count, float particle_mass,
    const void *rope_positions, const void *rope_velocities,
    std::uint32_t rope_count, const std::uint32_t *segment_generations,
    std::uint32_t rope_owner_slot, std::uint32_t rope_owner_generation,
    std::uint32_t rope_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    float distance, float friction,
    std::uint32_t *contribution_nodes, void *contribution_impulses,
    std::uint32_t *contact_count,
    float *maximum_penetration, bool transfer_impulse) noexcept;

void apply_fluid_rope_impulses(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *rope_velocities, void *fluid_contact_forces,
    std::uint32_t rope_count, float rope_mass, float rope_maximum_speed,
    bool first_attached, bool last_attached,
    const std::uint32_t *particle_count,
    const std::uint32_t *contribution_nodes,
    const void *contribution_impulses,
    float maximum_acceleration, float timestep,
    float frame_inverse_timestep) noexcept;

} // namespace parallel_mater::simulation
