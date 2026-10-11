// SPDX-License-Identifier: MIT
// Slang mirror of the device kernels in src/fluid_smoke.cuh.

import cuda_common;
import geometry_smoke;
import smoke_grid;

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_smoke_drag(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform uint* water_count, uniform FluidSmokeCouplingOptions coupling,
    uniform SmokeOptions smoke, uniform Vec3* smoke_positions,
    uniform Vec3* smoke_velocities, uniform uint64_t* smoke_keys,
    uniform uint* smoke_indices, uniform float dt)
{
    uint index = pmThreadIndex();
    if (index >= *water_count) return;
    SmokeFlowSample local = smoke_sample_flow(positions[index],
        smoke_positions, smoke_velocities, smoke_keys, smoke_indices, smoke);
    if (local.number_density <= 1.0e-6f) return;
    float occupancy = min(
        1.0f, local.number_density / smoke.rest_number_density);
    float response = 1.0f - exp(-coupling.wind_drag * occupancy * dt);
    Vec3 velocity = velocities[index];
    velocities[index] = clamp_length(add(velocity, multiply(
        subtract(local.velocity, velocity), response)),
        max(smoke.maximum_speed, sqrt(length_squared(velocity))));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_smoke_grid_drag(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform uint* water_count, uniform FluidSmokeCouplingOptions coupling,
    uniform SmokeOptions smoke, uniform SmokeGridField grid, uniform float dt)
{
    uint index = pmThreadIndex();
    if (index >= *water_count ||
        !smoke_grid_contains(positions[index], grid)) return;
    Vec3 air = {};
    float density = 0.0f;
    smoke_grid_sample(positions[index], grid, air, density);
    if (density <= 1.0e-6f) return;
    float response = 1.0f - exp(-coupling.wind_drag *
        clamp_scalar(density * smoke.rest_number_density, 0.0f, 1.0f) * dt);
    Vec3 velocity = velocities[index];
    velocities[index] = clamp_length(add(velocity, multiply(
        subtract(air, velocity), response)),
        max(smoke.maximum_speed, sqrt(length_squared(velocity))));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_smoke_exchange(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* temperatures, uniform uint* water_count,
    uniform float water_radius, uniform FluidSmokeCouplingOptions coupling,
    uniform SmokeOptions smoke, uniform Vec3* smoke_positions,
    uniform Vec3* smoke_velocities, uniform float* smoke_ages,
    uniform float* smoke_thermal_lift, uniform float* smoke_densities,
    uniform float* smoke_pressures, uniform Vec3* smoke_vorticities,
    uniform uint first_smoke_slot, uniform uint* converted,
    uniform uint8_t* keep, uniform float dt, uniform Vec3 gravity)
{
    uint index = pmThreadIndex();
    if (index >= *water_count) return;
    Vec3 position = positions[index];
    Vec3 velocity = velocities[index];
    Vec3 local = inverse_rotate(coupling.heater.orientation,
        subtract(position, coupling.heater.center));
    bool heated = abs(local.x) <= coupling.heater.half_extents.x &&
        abs(local.z) <= coupling.heater.half_extents.y &&
        local.y >= -water_radius * 0.5f &&
        local.y <= water_radius * 2.5f;
    float temperature = temperatures[index];
    if (heated && coupling.heater_temperature > temperature)
        temperature += (coupling.heater_temperature - temperature) *
            (1.0f - exp(-coupling.heat_transfer_rate * dt));
    temperatures[index] = temperature;
    keep[index] = 1u;
    if (temperature < coupling.boiling_temperature) return;
    uint ordinal = pmAtomicAdd(converted, 1u);
    if (ordinal >= smoke.capacity) return;
    uint slot = (first_smoke_slot + ordinal) % smoke.capacity;
    smoke_positions[slot] = position;
    smoke_velocities[slot] = add(velocity, multiply(
        normalized_or(multiply(gravity, -1.0f), {0.0f, 1.0f, 0.0f}),
        coupling.steam_rise_speed));
    smoke_ages[slot] = 0.0f;
    smoke_thermal_lift[slot] = coupling.steam_rise_speed;
    smoke_densities[slot] = 0.0f;
    smoke_pressures[slot] = 0.0f;
    smoke_vorticities[slot] = {};
    keep[index] = 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_clear_inactive_keep(
    uniform uint8_t* keep, uniform uint* count, uniform uint capacity)
{
    uint index = pmThreadIndex();
    if (index >= capacity) return;
    if (index >= *count) keep[index] = 0u;
}
