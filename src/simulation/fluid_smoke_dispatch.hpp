// SPDX-License-Identifier: MIT
#pragma once

#include "smoke_dispatch.hpp"

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

struct FluidSmokeDispatchOptions {
    SmokeDispatchOptions smoke{};
    float heater_center[3]{};
    float heater_orientation[4]{};
    float heater_half_extents[2]{};
    float heater_temperature{};
    float boiling_temperature{};
    float heat_transfer_rate{};
    float wind_drag{};
    float steam_rise_speed{};
};

void drag_fluid_with_smoke_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *velocities, const std::uint32_t *particle_count,
    const FluidSmokeDispatchOptions &options, const void *smoke_positions,
    const void *smoke_velocities, const std::uint64_t *smoke_keys,
    const std::uint32_t *smoke_indices, float timestep) noexcept;

void drag_fluid_with_smoke_grid(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *velocities, const std::uint32_t *particle_count,
    const FluidSmokeDispatchOptions &options,
    const SmokeGridDispatchField &smoke_grid, float timestep) noexcept;

void exchange_fluid_with_smoke(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *velocities, float *temperatures,
    const std::uint32_t *particle_count, float particle_radius,
    const FluidSmokeDispatchOptions &options, void *smoke_positions,
    void *smoke_velocities, float *smoke_ages, float *smoke_thermal_lift,
    float *smoke_densities, float *smoke_pressures,
    void *smoke_vorticities, std::uint32_t first_smoke_slot,
    std::uint32_t *converted, std::uint8_t *keep, float timestep,
    const float gravity[3]) noexcept;

void clear_inactive_fluid_keep(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, std::uint8_t *keep,
    const std::uint32_t *particle_count, std::uint32_t capacity) noexcept;

} // namespace parallel_mater::simulation
