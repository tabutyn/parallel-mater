// SPDX-License-Identifier: MIT
#pragma once

#include "smoke_dispatch.hpp"

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

void emit_smoke_realtime_particles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, float *densities, float *pressures,
    const SmokeDomainDispatchOptions &options, std::uint32_t first_slot,
    std::uint32_t emit_count, std::uint64_t first_serial, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void predict_smoke_realtime_particles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const float gravity[3],
    float timestep, std::uint32_t profile_version,
    void *accumulated_corrections, std::uint32_t *time_advances,
    std::uint32_t *status) noexcept;

void solve_smoke_realtime_density_pressure(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions, const float *ages,
    std::uint32_t count, const SmokeDomainDispatchOptions &options,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_indices,
    std::uint32_t candidates_per_particle, float timestep,
    std::uint32_t profile_version, float *densities, float *pressures,
    void *position_deltas, std::uint32_t *status) noexcept;

void solve_smoke_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *velocities, const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_indices,
    std::uint32_t candidates_per_particle, const float *densities,
    const float *pressures, const void *accumulated_corrections,
    void *position_deltas, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void apply_smoke_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, void *accumulated_corrections,
    const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void commit_smoke_realtime_particles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void index_smoke_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, void *keys,
    std::uint32_t *indices) noexcept;

void calculate_smoke_density_pressure(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *keys,
    const std::uint32_t *indices, float *densities, float *pressures,
    float timestep) noexcept;

void calculate_smoke_vorticity(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *keys,
    const std::uint32_t *indices, const float *densities,
    void *vorticities, float *magnitudes) noexcept;

void calculate_smoke_pair_forces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *keys,
    const std::uint32_t *indices, const float *densities,
    const float *pressures, const void *vorticities,
    const float *vorticity_magnitudes, void *accelerations) noexcept;

void advect_smoke_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *accelerations,
    float timestep, const float gravity[3]) noexcept;

void emit_smoke_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, float *densities, float *pressures,
    void *vorticities, const SmokeDomainDispatchOptions &options,
    std::uint32_t first_slot, std::uint32_t count,
    std::uint64_t first_serial) noexcept;

void calculate_smoke_deformable_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    std::uint32_t count, void *minimum, void *maximum) noexcept;

void collide_smoke_with_rigid_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    float *pressures, std::uint32_t count, float lifetime,
    const void *meshes, std::uint32_t mesh_index,
    const void *body_parameters, void *body_states,
    const void *previous_body_states,
    std::uint32_t body, const void *body_ids,
    std::uint32_t body_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    float clearance, float particle_mass,
    float wall_drag, float pressure_stiffness, float timestep,
    float maximum_speed, const SmokeGridDispatchField &smoke_grid,
    void *impulses) noexcept;

void apply_smoke_rigid_body_impulses(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *impulses,
    std::uint32_t count, std::uint32_t body, const void *body_parameters,
    void *body_states) noexcept;

} // namespace parallel_mater::simulation
