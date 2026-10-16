// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

struct FluidClothDispatchOptions {
    float contact_distance{};
    float interaction_radius{};
    float stiffness{};
    float damping{};
    float tangential_drag{};
    float maximum_force{};
};

void apply_fluid_cloth_forces(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *particle_positions, const void *particle_velocities,
    const std::uint32_t *particle_count, void *particle_accelerations,
    float *foam_source, float particle_mass, const void *cloth_positions,
    const void *cloth_velocities, const std::uint32_t *cloth_indices,
    std::uint32_t triangle_count, float cloth_orientation,
    const FluidClothDispatchOptions &options, void *cloth_forces) noexcept;

void apply_fluid_forces_to_cloth(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *positions, void *velocities, const float *inverse_masses,
    const void *forces, std::uint32_t count, float timestep) noexcept;

void project_fluid_inside_cloth(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *particle_positions, void *particle_velocities,
    const std::uint32_t *particle_count, float contact_distance,
    const void *cloth_positions, const void *cloth_velocities,
    const std::uint32_t *cloth_indices, std::uint32_t triangle_count,
    float cloth_orientation) noexcept;

} // namespace parallel_mater::simulation
