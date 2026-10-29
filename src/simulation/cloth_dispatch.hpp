// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

void predict_cloth_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, const float gravity[3],
    float velocity_damping, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *time_advances,
    std::uint32_t *status) noexcept;

void project_cloth_realtime_links(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count,
    std::uint32_t capacity, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void apply_cloth_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void project_cloth_realtime_volume(
    device::ExecutionQueue stream, void *positions,
    const float *inverse_masses, const float *characteristic_lengths,
    float *correction_used, const std::uint32_t *indices,
    std::uint32_t vertex_count, std::uint32_t triangle_count,
    const std::uint32_t *corner_offsets,
    const std::uint32_t *corner_indices, void *gradients,
    float target_volume, float orientation, float compliance, bool enabled,
    float timestep, std::uint32_t profile_version, float *lambda,
    bool reset_lambda, std::uint32_t *status) noexcept;

void project_cloth_realtime_contacts(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *previous_positions, const float *inverse_masses,
    const float *characteristic_lengths, float *correction_used,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float thickness, float friction,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void commit_cloth_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const float *inverse_masses, const float *characteristic_lengths,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, const void *bonds,
    std::uint8_t *bond_active, std::uint8_t *bond_damage,
    std::uint32_t bond_count, float break_strain,
    std::uint32_t fracture_persistence, const void *contact_impulses,
    float impact_break_threshold, std::uint32_t count,
    std::uint32_t capacity, float thickness, float friction,
    float maximum_speed, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void project_cloth_volume(
    device::ExecutionQueue stream, void *positions, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t vertex_count,
    std::uint32_t triangle_count, const std::uint32_t *corner_offsets,
    const std::uint32_t *corner_indices, void *gradients, float target_volume,
    float orientation, float compliance, float timestep, float *lambda,
    bool reset_lambda) noexcept;

void break_cloth_bonds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *bonds, std::uint8_t *active, std::uint8_t *damage,
    std::uint32_t count, float break_strain, std::uint32_t persistence,
    const void *contact_impulses, float impact_threshold) noexcept;

void limit_cloth_strain(
    device::ExecutionQueue stream, void *positions, void *scratch,
    const float *inverse_masses, const std::uint32_t *offsets,
    const void *neighbors, const std::uint8_t *free_nodes,
    std::uint32_t count) noexcept;

void update_cloth_surface(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const std::uint32_t *source_indices, void *surface_positions,
    std::uint32_t triangle_count) noexcept;

} // namespace parallel_mater::simulation
