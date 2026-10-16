// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

void refit_fluid_soft_surface(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    const void *previous_surface, const std::uint32_t *surface_indices,
    const std::uint32_t *triangle_order, void *tree,
    const std::uint32_t *parents, std::uint32_t *ready,
    std::uint32_t tree_count, void *bounds) noexcept;

void detect_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *previous, const void *velocities,
    const std::uint32_t *particle_count, const void *surface,
    const void *previous_surface, const std::uint32_t *surface_indices,
    std::uint32_t surface_index_count, const void *surface_bindings,
    const void *soft_velocities, float orientation, float distance,
    const void *bounds, bool sweep, const void *tree,
    const std::uint32_t *triangle_order, void *contacts,
    std::uint32_t *node_contact_counts, std::uint32_t *contact_count,
    float *maximum_penetration) noexcept;

void solve_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, void *accelerations, float *foam,
    const std::uint32_t *particle_count, const void *contacts,
    const std::uint32_t *node_contact_counts, const float *inverse_masses,
    const void *soft_velocities, float maximum_soft_speed,
    float inverse_particle_mass, float friction, float maximum_projection,
    float timestep, void *node_position_deltas, void *node_impulses) noexcept;

void apply_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const float *inverse_masses,
    const void *position_deltas, const void *impulses, void *forces,
    std::uint32_t node_count, float inverse_frame_timestep) noexcept;

void recover_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous, const std::uint32_t *particle_count,
    const void *surface, const void *previous_surface,
    const std::uint32_t *surface_indices, std::uint32_t surface_index_count,
    float orientation, float distance, const void *bounds, bool sweep,
    const void *tree, const std::uint32_t *triangle_order) noexcept;

} // namespace parallel_mater::simulation
