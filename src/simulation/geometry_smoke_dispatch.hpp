// SPDX-License-Identifier: MIT
#pragma once

#include "smoke_dispatch.hpp"

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

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
    const void *mesh, void *body_states, const void *previous_body_states,
    std::uint32_t body, float clearance, float particle_mass,
    float wall_drag, float pressure_stiffness, float timestep,
    float maximum_speed, const SmokeGridDispatchField &smoke_grid,
    void *impulses) noexcept;

void apply_smoke_rigid_body_impulses(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *impulses,
    std::uint32_t count, std::uint32_t body, const void *body_parameters,
    void *body_states) noexcept;

} // namespace parallel_mater::simulation
