// SPDX-License-Identifier: MIT
#include "fluid_smoke_dispatch.hpp"

#include "metal_dispatch_options.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
using metal_bindings::vector;

using metal_bindings::smoke_options;

FluidSmokeCouplingOptions coupling_options(
    const FluidSmokeDispatchOptions &value) noexcept {
    FluidSmokeCouplingOptions result{};
    result.heater.center = vector(value.heater_center);
    result.heater.orientation = {
        value.heater_orientation[0], value.heater_orientation[1],
        value.heater_orientation[2], value.heater_orientation[3]};
    result.heater.half_extents = {
        value.heater_half_extents[0], value.heater_half_extents[1]};
    result.heater_temperature = value.heater_temperature;
    result.boiling_temperature = value.boiling_temperature;
    result.heat_transfer_rate = value.heat_transfer_rate;
    result.wind_drag = value.wind_drag;
    result.steam_rise_speed = value.steam_rise_speed;
    return result;
}



} // namespace

void drag_fluid_with_smoke_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *velocities, const std::uint32_t *particle_count,
    const FluidSmokeDispatchOptions &options, const void *smoke_positions,
    const void *smoke_velocities, const std::uint64_t *smoke_keys,
    const std::uint32_t *smoke_indices, float timestep) noexcept {
    (void)metal::launch(stream, "fluid_smoke_drag", grid,
        positions, velocities,
        particle_count, coupling_options(options),
        smoke_options(options.smoke),
        smoke_positions,
        smoke_velocities,
        smoke_keys,
        smoke_indices, timestep);
}

void drag_fluid_with_smoke_realtime_particles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *current_velocities, void *velocity_deltas,
    const std::uint32_t *particle_count,
    const FluidSmokeDispatchOptions &options, const void *smoke_positions,
    const void *smoke_velocities, std::uint32_t smoke_owner_slot,
    std::uint32_t smoke_owner_generation,
    std::uint32_t smoke_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    const SmokeGridDispatchField &smoke_grid, float timestep) noexcept {
    (void)block;
    (void)metal::launch(stream, "fluid_smoke_realtime_drag", grid,
        positions, current_velocities, velocity_deltas, particle_count,
        coupling_options(options),
        smoke_options(options.smoke), smoke_positions, smoke_velocities,
        smoke_owner_slot, smoke_owner_generation, smoke_topology_revision,
        spatial_metadata, spatial_admission, spatial_leaf_capacity,
        candidate_counts, candidate_sources, candidates_per_query,
        spatial_status, metal::grid_record(smoke_grid), timestep);
}

void drag_fluid_with_smoke_grid(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *velocities, const std::uint32_t *particle_count,
    const FluidSmokeDispatchOptions &options,
    const SmokeGridDispatchField &smoke_grid, float timestep) noexcept {
    (void)metal::launch(stream, "fluid_smoke_grid_drag", grid,
        positions, velocities,
        particle_count, coupling_options(options),
        smoke_options(options.smoke), metal::grid_record(smoke_grid), timestep);
}

void exchange_fluid_with_smoke(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *base_velocities, const void *velocity_deltas,
    float *temperatures,
    const std::uint32_t *particle_count, float particle_radius,
    const FluidSmokeDispatchOptions &options, void *smoke_positions,
    void *smoke_velocities, float *smoke_ages, float *smoke_thermal_lift,
    float *smoke_densities, float *smoke_pressures,
    void *smoke_vorticities, std::uint32_t first_smoke_slot,
    std::uint32_t *converted, std::uint8_t *keep, float timestep,
    const float gravity[3]) noexcept {
    (void)metal::launch(stream, "fluid_smoke_exchange", grid,
        positions, base_velocities, velocity_deltas, temperatures,
        particle_count,
        particle_radius, coupling_options(options), smoke_options(options.smoke),
        smoke_positions, smoke_velocities, smoke_ages,
        smoke_thermal_lift, smoke_densities, smoke_pressures,
        smoke_vorticities, first_smoke_slot, converted, keep,
        timestep, vector(gravity));
}

void clear_inactive_fluid_keep(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, std::uint8_t *keep,
    const std::uint32_t *particle_count, std::uint32_t capacity) noexcept {
    (void)metal::launch(stream, "fluid_clear_inactive_keep", grid,
        keep, particle_count, capacity);
}

} // namespace parallel_mater::simulation
