#include "dispatch_entry.slang"
#include "dispatch_shape.slangh"
#include "coupling_spatial.slangh"
// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_smoke.cuh. Rigid-contact entries follow the
// standalone particle solver and shared wind sampler below.

import types;
import math;
import device_intrinsics;
import fluid;
import geometry_fluid;
import smoke_grid;
import graphics_resources;

groupshared Vec3 smoke_impulse_linear[128];
groupshared Vec3 smoke_impulse_angular[128];

uint smoke_hash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

float smoke_random(uint seed)
{
    return float(smoke_hash(seed) & 0x00ffffffu) / 16777216.0f;
}

public struct SmokeFlowSample
{
    public Vec3 velocity;
    public float number_density;
};

public SmokeFlowSample smoke_sample_flow(
    Vec3 point, PMBuffer<Vec3> positions, PMBuffer<Vec3> velocities,
    PMBuffer<uint64_t> keys, PMBuffer<uint> indices, SmokeOptions options)
{
    SmokeFlowSample sample = {};
    float support = 3.0f * options.particle_radius;
    float inverse = 1.0f / support;
    int cx = int(floor(point.x * inverse));
    int cy = int(floor(point.y * inverse));
    int cz = int(floor(point.z * inverse));
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                uint64_t key = fluid_cell_key(cx + dx, cy + dy, cz + dz);
                for (uint item = fluid_lower_bound(
                         keys, options.capacity, key);
                     item < options.capacity && keys[item] == key; ++item)
                {
                    uint other = indices[item];
                    float distance = vector_length(
                        subtract(point, positions[other]));
                    if (distance >= support) continue;
                    float q = 1.0f - distance * inverse;
                    float weight = q * q * q;
                    sample.number_density += weight;
                    sample.velocity = add(sample.velocity,
                        multiply(velocities[other], weight));
                }
            }
    if (sample.number_density > 1.0e-6f)
        sample.velocity = multiply(
            sample.velocity, 1.0f / sample.number_density);
    return sample;
}

public Vec3 smoke_wind_delta(
    Vec3 position, Vec3 velocity, PMBuffer<Vec3> smoke_positions,
    PMBuffer<Vec3> smoke_velocities, PMBuffer<uint64_t> keys, PMBuffer<uint> indices,
    SmokeOptions smoke, SmokeGridField grid, float drag,
    float maximum_acceleration, float dt)
{
    if (grid.n != 0u && smoke_grid_contains(position, grid))
    {
        Vec3 air = {};
        float density = 0.0f;
        smoke_grid_sample(position, grid, air, density);
        float response = 1.0f - exp(-drag * clamp_scalar(
            density * smoke.rest_number_density, 0.0f, 1.0f) * dt);
        return clamp_length(multiply(subtract(air, velocity), response),
            maximum_acceleration * dt);
    }
    SmokeFlowSample local = smoke_sample_flow(position, smoke_positions,
        smoke_velocities, keys, indices, smoke);
    if (local.number_density <= 1.0e-6f) return {};
    float occupancy = min(
        1.0f, local.number_density / smoke.rest_number_density);
    float response = 1.0f - exp(-drag * occupancy * dt);
    return clamp_length(multiply(subtract(local.velocity, velocity), response),
        maximum_acceleration * dt);
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void smoke_emit_cells(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<float> ages, uniform uint count,
    uniform SmokeOptions options, uniform PMBuffer<uint64_t> keys,
    uniform PMBuffer<uint> indices PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= options.capacity) return;
    indices[particle] = particle;
    if (particle >= count || ages[particle] >= options.lifetime)
    {
        keys[particle] = 0xffffffffffffffffull;
        return;
    }
    float inverse = 1.0f / (3.0f * options.particle_radius);
    Vec3 point = positions[particle];
    keys[particle] = fluid_cell_key(
        int(floor(point.x * inverse)), int(floor(point.y * inverse)),
        int(floor(point.z * inverse)));
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void smoke_density_pressure(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<float> ages, uniform uint count,
    uniform SmokeOptions options, uniform PMBuffer<uint64_t> keys,
    uniform PMBuffer<uint> indices, uniform PMBuffer<float> densities,
    uniform PMBuffer<float> pressures, uniform float dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= count) return;
    if (ages[particle] >= options.lifetime)
    {
        densities[particle] = 0.0f;
        pressures[particle] = 0.0f;
        return;
    }
    SmokeFlowSample local = smoke_sample_flow(
        positions[particle], positions, positions, keys, indices, options);
    densities[particle] = local.number_density;
    float crowding = options.pressure_stiffness * max(
        0.0f, local.number_density / options.rest_number_density - 1.0f);
    pressures[particle] = min(
        100.0f, crowding + pressures[particle] * exp(-dt / 0.05f));
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void smoke_pair_forces(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> ages, uniform uint count, uniform SmokeOptions options,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> indices,
    uniform PMBuffer<float> densities, uniform PMBuffer<float> pressures,
    uniform PMBuffer<Vec3> vorticities, uniform PMBuffer<float> vorticity_magnitudes,
    uniform PMBuffer<Vec3> accelerations PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= count) return;
    accelerations[particle] = {};
    if (ages[particle] >= options.lifetime) return;
    Vec3 point = positions[particle];
    Vec3 velocity = velocities[particle];
    float support = 3.0f * options.particle_radius;
    float inverse = 1.0f / support;
    int cx = int(floor(point.x * inverse));
    int cy = int(floor(point.y * inverse));
    int cz = int(floor(point.z * inverse));
    Vec3 acceleration = {};
    Vec3 confinementGradient = {};
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                uint64_t key = fluid_cell_key(cx + dx, cy + dy, cz + dz);
                for (uint item = fluid_lower_bound(keys, options.capacity, key);
                     item < options.capacity && keys[item] == key; ++item)
                {
                    uint other = indices[item];
                    if (other == particle) continue;
                    Vec3 delta = subtract(point, positions[other]);
                    float distanceSquared = length_squared(delta);
                    if (distanceSquared >= support * support ||
                        distanceSquared < 1.0e-12f) continue;
                    float distance = sqrt(distanceSquared);
                    float q = 1.0f - distance * inverse;
                    float pairDensity = max(1.0f,
                        sqrt(densities[particle] * densities[other]));
                    float pressure =
                        (pressures[particle] + pressures[other]) * 0.5f;
                    acceleration = add(acceleration, multiply(delta,
                        3.0f * pressure * q * q /
                            (support * distance * pairDensity)));
                    acceleration = add(acceleration, multiply(
                        subtract(velocities[other], velocity),
                        options.viscosity * q * q / pairDensity));
                    if (options.vorticity_confinement > 0.0f)
                    {
                        Vec3 gradient = multiply(delta,
                            -3.0f * q * q /
                                (support * distance * pairDensity));
                        confinementGradient = add(confinementGradient,
                            multiply(gradient, vorticity_magnitudes[other] -
                                vorticity_magnitudes[particle]));
                    }
                }
            }
    if (length_squared(confinementGradient) > 1.0e-10f)
        acceleration = add(acceleration, multiply(cross(normalized_or(
            confinementGradient, {1.0f, 0.0f, 0.0f}),
            vorticities[particle]), options.vorticity_confinement * support));
    accelerations[particle] = clamp_length(acceleration, 50.0f);
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void smoke_compute_vorticity(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> ages, uniform uint count, uniform SmokeOptions options,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> indices,
    uniform PMBuffer<float> densities, uniform PMBuffer<Vec3> vorticities,
    uniform PMBuffer<float> magnitudes PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= count) return;
    vorticities[particle] = {};
    magnitudes[particle] = 0.0f;
    if (ages[particle] >= options.lifetime ||
        options.vorticity_confinement == 0.0f) return;
    Vec3 point = positions[particle];
    float support = 3.0f * options.particle_radius;
    float inverse = 1.0f / support;
    int cx = int(floor(point.x * inverse));
    int cy = int(floor(point.y * inverse));
    int cz = int(floor(point.z * inverse));
    Vec3 curl = {};
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                uint64_t key = fluid_cell_key(cx + dx, cy + dy, cz + dz);
                for (uint item = fluid_lower_bound(keys, options.capacity, key);
                     item < options.capacity && keys[item] == key; ++item)
                {
                    uint other = indices[item];
                    if (other == particle) continue;
                    Vec3 delta = subtract(point, positions[other]);
                    float distanceSquared = length_squared(delta);
                    if (distanceSquared >= support * support ||
                        distanceSquared < 1.0e-12f) continue;
                    float distance = sqrt(distanceSquared);
                    float q = 1.0f - distance * inverse;
                    float pairDensity = max(1.0f,
                        sqrt(densities[particle] * densities[other]));
                    Vec3 gradient = multiply(delta,
                        -3.0f * q * q /
                            (support * distance * pairDensity));
                    curl = add(curl, cross(subtract(
                        velocities[other], velocities[particle]), gradient));
                }
            }
    vorticities[particle] = curl;
    magnitudes[particle] = vector_length(curl);
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void smoke_advect(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> ages,
    uniform PMBuffer<float> thermal_lift, uniform uint count,
    uniform SmokeOptions options, uniform PMBuffer<Vec3> accelerations,
    uniform float dt, uniform Vec3 gravity PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index >= count || ages[index] >= options.lifetime) return;
    Vec3 point = positions[index];
    previous_positions[index] = point;
    Vec3 velocity = velocities[index];
    Vec3 lift = multiply(normalized_or(multiply(gravity, -1.0f),
        {0.0f, 1.0f, 0.0f}), options.buoyancy + thermal_lift[index]);
    velocity = add(velocity,
        multiply(add(accelerations[index], lift), dt));
    float response = 1.0f - exp(-options.response * dt);
    velocity = clamp_length(add(velocity,
        multiply(subtract(options.wind, velocity), response)),
        options.maximum_speed);
    positions[index] = add(point, multiply(velocity, dt));
    velocities[index] = velocity;
    ages[index] += dt;
    thermal_lift[index] *= exp(-dt / 3.0f);
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void smoke_emit(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> ages,
    uniform PMBuffer<float> thermal_lift, uniform PMBuffer<float> densities,
    uniform PMBuffer<float> pressures, uniform PMBuffer<Vec3> vorticities,
    uniform SmokeOptions options, uniform uint first_slot,
    uniform uint count, uniform uint64_t first_serial PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index >= count) return;
    uint slot = (first_slot + index) % options.capacity;
    uint serial = uint(first_serial + uint64_t(index));
    float y = 2.0f * smoke_random(serial ^ 0x132aef41u) - 1.0f;
    float z = 2.0f * smoke_random(serial ^ 0xa385c9d3u) - 1.0f;
    positions[slot] = add(options.emitter_center, {
        0.0f, y * options.emitter_half_extents.x,
        z * options.emitter_half_extents.y});
    previous_positions[slot] = positions[slot];
    velocities[slot] = options.initial_velocity;
    ages[slot] = 0.0f;
    thermal_lift[slot] = 0.0f;
    densities[slot] = 0.0f;
    pressures[slot] = 0.0f;
    vorticities[slot] = {};
}

// realtime-single-step v1 keeps particle smoke as a supported representation
// without reusing the legacy sorted-cell force/integrate loop. Candidate lists
// come from the one F8 spatial publication and remain fixed for six Jacobi
// pressure sweeps. The swept publication includes the correction allowance.
static const uint kSmokeRealtimeProfileVersion = 1u;
static const float kSmokeRealtimeTimestep = 1.0f / 60.0f;
static const float kSmokeRealtimeSolverDamping = 0.05f;
static const float kSmokeRealtimeMaximumCorrectionFraction = 0.25f;
static const uint kSmokeRealtimeInvalidProfile = 1u << 17u;
static const uint kSmokeRealtimeInvalidTimestep = 1u << 18u;
static const uint kSmokeRealtimeInvalidMaterial = 1u << 19u;
static const uint kSmokeRealtimeNonfiniteState = 1u << 20u;

bool smoke_realtime_finite_vec3(Vec3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

bool smoke_realtime_profile_valid(uint profile_version, float dt)
{
    return profile_version == kSmokeRealtimeProfileVersion && isfinite(dt) &&
        abs(dt - kSmokeRealtimeTimestep) <= 2.0e-9f;
}

float smoke_realtime_pressure_response(SmokeOptions options)
{
    return options.pressure_stiffness == 0.0f ? 0.0f :
        1.0f - exp(-options.pressure_stiffness);
}

bool smoke_realtime_particle_material_valid(SmokeOptions options)
{
    float support = 3.0f * options.particle_radius;
    float pressureResponse = smoke_realtime_pressure_response(options);
    return options.capacity > 0u && isfinite(options.lifetime) &&
        isfinite(options.particle_radius) && isfinite(options.buoyancy) &&
        isfinite(options.response) && isfinite(options.rest_number_density) &&
        isfinite(options.pressure_stiffness) && isfinite(options.viscosity) &&
        isfinite(options.maximum_speed) && options.lifetime > 0.0f &&
        support >= 0.005f && support <= 2.0f && options.buoyancy >= 0.0f &&
        options.response >= 0.0f && options.rest_number_density > 0.0f &&
        options.pressure_stiffness >= 0.0f && options.viscosity >= 0.0f &&
        (options.pressure_stiffness == 0.0f ||
            (pressureResponse >= 0.05f && pressureResponse <= 0.995f)) &&
        options.maximum_speed >= 0.0f && options.maximum_speed <= 20.0f &&
        options.maximum_speed * kSmokeRealtimeTimestep <= 0.5f * support &&
        options.buoyancy * kSmokeRealtimeTimestep *
            kSmokeRealtimeTimestep <= 0.25f * support;
}

void smoke_realtime_record_invalid(
    uint profile_version, float dt, SmokeOptions options,
    PMBuffer<uint> status)
{
    if (profile_version != kSmokeRealtimeProfileVersion)
        pmAtomicOr(status, kSmokeRealtimeInvalidProfile);
    if (!isfinite(dt) || abs(dt - kSmokeRealtimeTimestep) > 2.0e-9f)
        pmAtomicOr(status, kSmokeRealtimeInvalidTimestep);
    if (!smoke_realtime_particle_material_valid(options))
        pmAtomicOr(status, kSmokeRealtimeInvalidMaterial);
}

float smoke_realtime_particle_weight(float distance, float support)
{
    if (distance >= support) return 0.0f;
    float q = max(0.0f, 1.0f - distance / support);
    return q * q * q;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void smoke_realtime_particle_emit(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> ages,
    uniform PMBuffer<float> thermal_lift,
    uniform PMBuffer<float> densities, uniform PMBuffer<float> pressures,
    uniform SmokeOptions options, uniform uint first_slot,
    uniform uint emit_count, uniform uint64_t first_serial, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (!smoke_realtime_profile_valid(profile_version, dt) ||
        !smoke_realtime_particle_material_valid(options))
    {
        if (index == 0u)
            smoke_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (index >= emit_count) return;
    uint slot = (first_slot + index) % options.capacity;
    uint serial = uint(first_serial + uint64_t(index));
    float y = 2.0f * smoke_random(serial ^ 0x132aef41u) - 1.0f;
    float z = 2.0f * smoke_random(serial ^ 0xa385c9d3u) - 1.0f;
    positions[slot] = add(options.emitter_center, {
        0.0f, y * options.emitter_half_extents.x,
        z * options.emitter_half_extents.y});
    previous_positions[slot] = positions[slot];
    velocities[slot] = clamp_length(
        options.initial_velocity, options.maximum_speed);
    ages[slot] = 0.0f;
    thermal_lift[slot] = 0.0f;
    densities[slot] = 1.0f;
    pressures[slot] = 0.0f;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void smoke_realtime_particle_predict(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> ages,
    uniform PMBuffer<float> thermal_lift, uniform uint count,
    uniform SmokeOptions options, uniform Vec3 gravity, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<Vec3> accumulated_corrections,
    uniform PMBuffer<uint> time_advances,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (!smoke_realtime_profile_valid(profile_version, dt) ||
        !smoke_realtime_particle_material_valid(options))
    {
        if (particle == 0u)
            smoke_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (particle == 0u) ++time_advances[0];
    if (particle >= count || particle >= options.capacity ||
        ages[particle] >= options.lifetime) return;
    Vec3 point = positions[particle];
    Vec3 velocity = velocities[particle];
    previous_positions[particle] = point;
    accumulated_corrections[particle] = {};
    Vec3 up = normalized_or(multiply(gravity, -1.0f), {0.0f, 1.0f, 0.0f});
    velocity = add(velocity, multiply(up,
        (options.buoyancy + thermal_lift[particle]) * dt));
    float windResponse = 1.0f - exp(-options.response * dt);
    velocity = add(velocity,
        multiply(subtract(options.wind, velocity), windResponse));
    float support = 3.0f * options.particle_radius;
    velocity = clamp_length(velocity, min(options.maximum_speed,
        0.5f * support / dt));
    point = add(point, multiply(velocity, dt));
    if (!smoke_realtime_finite_vec3(point) ||
        !smoke_realtime_finite_vec3(velocity))
    {
        pmAtomicOr(status, kSmokeRealtimeNonfiniteState);
        return;
    }
    positions[particle] = point;
    velocities[particle] = velocity;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void smoke_realtime_particle_density_pressure(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<float> ages,
    uniform uint count, uniform SmokeOptions options,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_indices,
    uniform uint candidates_per_particle, uniform float dt,
    uniform uint profile_version, uniform PMBuffer<float> densities,
    uniform PMBuffer<float> pressures,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (!smoke_realtime_profile_valid(profile_version, dt) ||
        !smoke_realtime_particle_material_valid(options))
    {
        if (particle == 0u)
            smoke_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (particle >= count || particle >= options.capacity) return;
    if (ages[particle] >= options.lifetime)
    {
        densities[particle] = 0.0f;
        pressures[particle] = 0.0f;
        return;
    }
    uint candidateCount = candidate_counts[particle];
    if (candidateCount > candidates_per_particle)
    {
        pmAtomicOr(status, kSmokeRealtimeInvalidMaterial);
        return;
    }
    float support = 3.0f * options.particle_radius;
    float density = 1.0f;
    uint base = particle * candidates_per_particle;
    for (uint candidate = 0u; candidate < candidateCount; ++candidate)
    {
        uint other = candidate_indices[base + candidate];
        if (other >= count || other >= options.capacity || other == particle ||
            ages[other] >= options.lifetime) continue;
        density += smoke_realtime_particle_weight(vector_length(subtract(
            positions[particle], positions[other])), support);
    }
    float compression = max(
        density / options.rest_number_density - 1.0f, 0.0f);
    float pressure = smoke_realtime_pressure_response(options) * compression;
    if (!isfinite(density) || !isfinite(pressure))
    {
        pmAtomicOr(status, kSmokeRealtimeNonfiniteState);
        density = options.rest_number_density;
        pressure = 0.0f;
    }
    densities[particle] = density;
    pressures[particle] = pressure;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void smoke_realtime_particle_position_delta(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> ages, uniform uint count,
    uniform SmokeOptions options, uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_indices,
    uniform uint candidates_per_particle, uniform PMBuffer<float> densities,
    uniform PMBuffer<float> pressures,
    uniform PMBuffer<Vec3> accumulated_corrections,
    uniform PMBuffer<Vec3> position_deltas, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (!smoke_realtime_profile_valid(profile_version, dt) ||
        !smoke_realtime_particle_material_valid(options))
    {
        if (particle == 0u)
            smoke_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (particle >= count || particle >= options.capacity ||
        ages[particle] >= options.lifetime) return;
    uint candidateCount = candidate_counts[particle];
    if (candidateCount > candidates_per_particle)
    {
        pmAtomicOr(status, kSmokeRealtimeInvalidMaterial);
        return;
    }
    float support = 3.0f * options.particle_radius;
    Vec3 point = positions[particle];
    Vec3 delta = {};
    float weightSum = 0.0f;
    uint base = particle * candidates_per_particle;
    for (uint candidate = 0u; candidate < candidateCount; ++candidate)
    {
        uint other = candidate_indices[base + candidate];
        if (other >= count || other >= options.capacity || other == particle ||
            ages[other] >= options.lifetime) continue;
        Vec3 separation = subtract(point, positions[other]);
        float distanceSquared = length_squared(separation);
        if (distanceSquared >= support * support) continue;
        float distance = sqrt(max(distanceSquared, 1.0e-12f));
        Vec3 direction = distanceSquared > 1.0e-12f
            ? multiply(separation, 1.0f / distance)
            : (particle < other ? Vec3(-1.0f, 0.0f, 0.0f)
                                : Vec3(1.0f, 0.0f, 0.0f));
        float weight = smoke_realtime_particle_weight(distance, support);
        float pairPressure = 0.5f *
            (pressures[particle] + pressures[other]);
        delta = add(delta, multiply(direction,
            support * weight * pairPressure));
        // XSPH-like viscosity is a bounded displacement in the same Jacobi
        // proposal so the sweep count and correction budget remain exact.
        float viscosityResponse = 1.0f - exp(-options.viscosity * dt);
        delta = add(delta, multiply(subtract(
            velocities[other], velocities[particle]),
            dt * viscosityResponse * weight));
        weightSum += weight;
    }
    if (weightSum > 1.0e-6f) delta = multiply(delta, 1.0f / weightSum);
    float maximumCorrection =
        kSmokeRealtimeMaximumCorrectionFraction * support;
    float remaining = max(0.0f, maximumCorrection -
        vector_length(accumulated_corrections[particle]));
    delta = clamp_length(delta, remaining);
    if (!smoke_realtime_finite_vec3(delta))
    {
        pmAtomicOr(status, kSmokeRealtimeNonfiniteState);
        delta = {};
    }
    position_deltas[particle] = delta;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void smoke_realtime_particle_apply_position_delta(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<Vec3> accumulated_corrections,
    uniform PMBuffer<float> ages, uniform uint count,
    uniform SmokeOptions options, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (!smoke_realtime_profile_valid(profile_version, dt) ||
        !smoke_realtime_particle_material_valid(options))
    {
        if (particle == 0u)
            smoke_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (particle >= count || particle >= options.capacity ||
        ages[particle] >= options.lifetime) return;
    Vec3 point = add(positions[particle], position_deltas[particle]);
    Vec3 correction = add(accumulated_corrections[particle],
        position_deltas[particle]);
    if (!smoke_realtime_finite_vec3(point) ||
        !smoke_realtime_finite_vec3(correction))
    {
        pmAtomicOr(status, kSmokeRealtimeNonfiniteState);
        return;
    }
    positions[particle] = point;
    accumulated_corrections[particle] = correction;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void smoke_realtime_particle_commit(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> ages,
    uniform PMBuffer<float> thermal_lift, uniform uint count,
    uniform SmokeOptions options, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (!smoke_realtime_profile_valid(profile_version, dt) ||
        !smoke_realtime_particle_material_valid(options))
    {
        if (particle == 0u)
            smoke_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (particle >= count || particle >= options.capacity ||
        ages[particle] >= options.lifetime) return;
    Vec3 velocity = multiply(subtract(
        positions[particle], previous_positions[particle]), 1.0f / dt);
    velocity = clamp_length(multiply(velocity,
        1.0f - kSmokeRealtimeSolverDamping), options.maximum_speed);
    if (!smoke_realtime_finite_vec3(positions[particle]) ||
        !smoke_realtime_finite_vec3(velocity))
    {
        pmAtomicOr(status, kSmokeRealtimeNonfiniteState);
        positions[particle] = previous_positions[particle];
        velocities[particle] = {};
        return;
    }
    velocities[particle] = velocity;
    ages[particle] += dt;
    thermal_lift[particle] *= exp(-dt / 3.0f);
}

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void smoke_deformable_bounds(
    uniform PMBuffer<Vec3> surface, uniform uint count,
    uniform PMBuffer<Vec3> minimum, uniform PMBuffer<Vec3> maximum PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u || pmGroupThreadIndex().x != 0u) return;
    Vec3 low = surface[0];
    Vec3 high = surface[0];
    for (uint index = 1u; index < count; ++index)
    {
        low = component_min(low, surface[index]);
        high = component_max(high, surface[index]);
    }
    minimum[0] = low;
    maximum[0] = high;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void smoke_rigid_contact(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> ages, uniform PMBuffer<float> pressures,
    uniform uint count, uniform float lifetime,
    uniform PMBuffer<TriangleMeshResource> meshes, uniform uint mesh_index,
    uniform PMBuffer<BodyParameters> body_parameters,
    uniform PMBuffer<RigidBodyState> states,
    uniform PMBuffer<RigidBodyState> previous_states, uniform uint body,
    uniform PMBuffer<RigidBodyId> body_ids, uniform uint body_topology_revision,
    uniform PMBuffer<CouplingSpatialLeafMetadata> spatial_metadata,
    uniform PMByteBuffer spatial_admission, uniform uint spatial_leaf_capacity,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_sources,
    uniform uint candidates_per_query, uniform PMBuffer<uint> spatial_status,
    uniform float clearance, uniform float particle_mass,
    uniform float wall_drag, uniform float pressure_stiffness,
    uniform float dt, uniform float maximum_speed,
    uniform SmokeGridField grid, uniform PMBuffer<FluidBodyImpulse> impulses PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= count) return;
    impulses[particle] = {};
    if (ages[particle] >= lifetime) return;
    TriangleMeshResource mesh = meshes[mesh_index];
    uint candidateCount = candidate_counts[particle];
    if (candidateCount > candidates_per_query)
    {
        pmAtomicOr(spatial_status, kCouplingSpatialQueryOverflow);
        return;
    }
    bool bodyCandidate = false;
    uint candidateBase = particle * candidates_per_query;
    for (uint candidate = 0u; candidate < candidateCount; ++candidate)
    {
        uint source = candidate_sources[candidateBase + candidate];
        if (source >= spatial_leaf_capacity || spatial_admission[source] == 0u)
        {
            pmAtomicOr(spatial_status, kCouplingSpatialAdmissionInvalid);
            return;
        }
        CouplingSpatialLeafMetadata leaf = spatial_metadata[source];
        if (leaf.identity.family != kCouplingSpatialRigidFamily ||
            leaf.payload_ref != body) continue;
        if (leaf.identity.owner_slot != body_ids[body].index ||
            leaf.identity.owner_generation != body_ids[body].generation ||
            leaf.identity.topology_revision != body_topology_revision ||
            leaf.identity.primitive_kind != kCouplingSpatialBodyPrimitive ||
            leaf.identity.primitive_generation != mesh.generation)
        {
            pmAtomicOr(spatial_status, kCouplingSpatialStaleEpoch);
            return;
        }
        bodyCandidate = true;
        break;
    }
    if (!bodyCandidate) return;
    float boundary = 3.0f * clearance;
    RigidBodyState state = states[body];
    RigidBodyState previousState = previous_states[body];
    Vec3 before = inverse_rotate(previousState.orientation,
        subtract(previous_positions[particle], previousState.position));
    Vec3 point = inverse_rotate(state.orientation,
        subtract(positions[particle], state.position));
    if (max(point.x, before.x) < mesh.minimum.x - boundary ||
        min(point.x, before.x) > mesh.maximum.x + boundary ||
        max(point.y, before.y) < mesh.minimum.y - boundary ||
        min(point.y, before.y) > mesh.maximum.y + boundary ||
        max(point.z, before.z) < mesh.minimum.z - boundary ||
        min(point.z, before.z) > mesh.maximum.z + boundary) return;
    float nearestSquared = boundary * boundary;
    float earliest = 2.0f;
    bool found = false, swept = false;
    Vec3 nearest = {}, faceNormal = {};
    Vec3 path = subtract(point, before);
    uint stack[64] = {};
    int pending = mesh.bvh_node_count == 0u ? 0 : 1;
    while (pending != 0)
    {
        BvhNode node = mesh.bvh_nodes[stack[--pending]];
        if (!fluid_segment_bounds(before, point, node, boundary)) continue;
        if (node.triangle_count == 0u)
        {
            if (pending + 2 > 64) continue;
            stack[pending++] = node.right;
            stack[pending++] = node.left;
            continue;
        }
        for (uint item = 0u; item < node.triangle_count; ++item)
        {
            uint base = (node.first_triangle + item) * 3u;
            Vec3 a = mesh.vertices[mesh.indices[base]];
            Vec3 b = mesh.vertices[mesh.indices[base + 1u]];
            Vec3 c = mesh.vertices[mesh.indices[base + 2u]];
            Vec3 rawNormal = cross(subtract(b, a), subtract(c, a));
            if (length_squared(rawNormal) < 1.0e-12f) continue;
            float sideBefore = dot(subtract(before, a), rawNormal);
            float sideAfter = dot(subtract(point, a), rawNormal);
            if (sideBefore * sideAfter < 0.0f)
            {
                float fraction = sideBefore / (sideBefore - sideAfter);
                if (fraction < earliest)
                {
                    Vec3 hit = add(before, multiply(path, fraction));
                    Vec3 weights = {};
                    Vec3 onFace = fluid_closest_triangle_barycentric(
                        hit, a, b, c, weights);
                    if (length_squared(subtract(hit, onFace)) < 1.0e-8f)
                    {
                        earliest = fraction;
                        nearest = onFace;
                        faceNormal = normalized_or(
                            rawNormal, {1.0f, 0.0f, 0.0f});
                        swept = true;
                        found = true;
                    }
                }
            }
            if (swept) continue;
            Vec3 weights = {};
            Vec3 candidate = fluid_closest_triangle_barycentric(
                point, a, b, c, weights);
            float distanceSquared = length_squared(subtract(point, candidate));
            if (distanceSquared < nearestSquared)
            {
                nearestSquared = distanceSquared;
                nearest = candidate;
                faceNormal = normalized_or(rawNormal, {1.0f, 0.0f, 0.0f});
                found = true;
            }
        }
    }
    if (!found) return;
    float side = dot(subtract(before, nearest), faceNormal);
    if (abs(side) < 1.0e-5f)
        side = dot(subtract(point, nearest), faceNormal);
    Vec3 localNormal = side >= 0.0f ? faceNormal : multiply(faceNormal, -1.0f);
    Vec3 normal = rotate(state.orientation, localNormal);
    Vec3 arm = rotate(state.orientation, nearest);
    bool touching = swept || nearestSquared < clearance * clearance;
    Vec3 surfacePoint = add(state.position, arm);
    Vec3 tracePosition = add(surfacePoint, multiply(normal,
        grid.n != 0u ? 1.5f * grid.spacing : clearance));
    if (touching) positions[particle] = tracePosition;
    Vec3 surfaceVelocity = add(state.linear_velocity,
        cross(state.angular_velocity, arm));
    Vec3 oldVelocity = velocities[particle];
    Vec3 tracerVelocity = grid.n != 0u && touching &&
        smoke_grid_contains(tracePosition, grid)
        ? smoke_grid_sample_velocity(tracePosition, grid) : oldVelocity;
    Vec3 relative = subtract(tracerVelocity, surfaceVelocity);
    float normalInflow = max(0.0f, -dot(relative, normal));
    if (touching)
        relative = subtract(relative,
            multiply(normal, min(0.0f, dot(relative, normal))));
    if (grid.n != 0u && touching)
    {
        float sourceSpeed = max(vector_length(subtract(
            tracerVelocity, surfaceVelocity)), vector_length(subtract(
            oldVelocity, surfaceVelocity)));
        float tangentSpeed = vector_length(relative);
        if (tangentSpeed > 1.0e-6f && tangentSpeed < sourceSpeed)
            relative = multiply(relative, sourceSpeed / tangentSpeed);
    }
    float distance = touching ? clearance : sqrt(nearestSquared);
    float q = max(0.0f, 1.0f - distance / boundary);
    if (grid.n == 0u)
    {
        float response = 1.0f - exp(-wall_drag * q * q * dt);
        relative = multiply(relative, 1.0f - response);
    }
    if (touching && grid.n == 0u)
        pressures[particle] = max(pressures[particle],
            10.0f * pressure_stiffness * normalInflow * normalInflow);
    Vec3 newVelocity = clamp_length(
        add(surfaceVelocity, relative), maximum_speed);
    velocities[particle] = newVelocity;
    Vec3 reaction = multiply(subtract(oldVelocity, newVelocity), particle_mass);
    impulses[particle] = {reaction, cross(arm, reaction), body};
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void smoke_apply_rigid_impulses(
    uniform PMBuffer<FluidBodyImpulse> impulses, uniform uint count,
    uniform uint body, uniform PMBuffer<BodyParameters> parameters,
    uniform PMBuffer<RigidBodyState> states PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u ||
        parameters[body].motion != MotionType.dynamic) return;
    uint lane = pmGroupThreadIndex().x;
    Vec3 localLinear = {}, localAngular = {};
    for (uint particle = lane; particle < count;
         particle += pmGroupSize().x)
    {
        FluidBodyImpulse item = impulses[particle];
        if (item.body != body) continue;
        localLinear = add(localLinear, item.linear);
        localAngular = add(localAngular, item.angular);
    }
    smoke_impulse_linear[lane] = localLinear;
    smoke_impulse_angular[lane] = localAngular;
    pmGroupBarrier();
    for (uint stride = pmGroupSize().x / 2u; stride != 0u; stride /= 2u)
    {
        if (lane < stride)
        {
            smoke_impulse_linear[lane] = add(smoke_impulse_linear[lane],
                smoke_impulse_linear[lane + stride]);
            smoke_impulse_angular[lane] = add(smoke_impulse_angular[lane],
                smoke_impulse_angular[lane + stride]);
        }
        pmGroupBarrier();
    }
    if (lane != 0u) return;
    RigidBodyState state = states[body];
    BodyParameters settings = parameters[body];
    state.linear_velocity = clamp_length(add(state.linear_velocity,
        multiply(smoke_impulse_linear[0], settings.inverse_mass)),
        settings.maximum_linear_speed);
    state.angular_velocity = clamp_length(add(state.angular_velocity,
        inverse_inertia_world(settings, state, smoke_impulse_angular[0])),
        settings.maximum_angular_speed);
    states[body] = state;
}
