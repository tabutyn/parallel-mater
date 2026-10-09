// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_smoke.cuh. Rigid-contact entries follow the
// standalone particle solver and shared wind sampler below.

import cuda_common;
import fluid;
import smoke_grid;

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
    Vec3 point, Vec3* positions, Vec3* velocities,
    uint64_t* keys, uint* indices, SmokeOptions options)
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
    Vec3 position, Vec3 velocity, Vec3* smoke_positions,
    Vec3* smoke_velocities, uint64_t* keys, uint* indices,
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
[numthreads(256, 1, 1)]
void smoke_emit_cells(
    uniform Vec3* positions, uniform float* ages, uniform uint count,
    uniform SmokeOptions options, uniform uint64_t* keys,
    uniform uint* indices)
{
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
[numthreads(256, 1, 1)]
void smoke_density_pressure(
    uniform Vec3* positions, uniform float* ages, uniform uint count,
    uniform SmokeOptions options, uniform uint64_t* keys,
    uniform uint* indices, uniform float* densities,
    uniform float* pressures, uniform float dt)
{
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
[numthreads(256, 1, 1)]
void smoke_pair_forces(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* ages, uniform uint count, uniform SmokeOptions options,
    uniform uint64_t* keys, uniform uint* indices,
    uniform float* densities, uniform float* pressures,
    uniform Vec3* vorticities, uniform float* vorticity_magnitudes,
    uniform Vec3* accelerations)
{
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
[numthreads(256, 1, 1)]
void smoke_compute_vorticity(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* ages, uniform uint count, uniform SmokeOptions options,
    uniform uint64_t* keys, uniform uint* indices,
    uniform float* densities, uniform Vec3* vorticities,
    uniform float* magnitudes)
{
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
[numthreads(256, 1, 1)]
void smoke_advect(
    uniform Vec3* positions, uniform Vec3* previous_positions,
    uniform Vec3* velocities, uniform float* ages,
    uniform float* thermal_lift, uniform uint count,
    uniform SmokeOptions options, uniform Vec3* accelerations,
    uniform float dt, uniform Vec3 gravity)
{
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
[numthreads(256, 1, 1)]
void smoke_emit(
    uniform Vec3* positions, uniform Vec3* previous_positions,
    uniform Vec3* velocities, uniform float* ages,
    uniform float* thermal_lift, uniform float* densities,
    uniform float* pressures, uniform Vec3* vorticities,
    uniform SmokeOptions options, uniform uint first_slot,
    uniform uint count, uniform uint64_t first_serial)
{
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

[shader("compute")]
[numthreads(1, 1, 1)]
void smoke_deformable_bounds(
    uniform Vec3* surface, uniform uint count,
    uniform Vec3* minimum, uniform Vec3* maximum)
{
    if (cudaBlockIdx().x != 0u || cudaThreadIdx().x != 0u) return;
    Vec3 low = surface[0];
    Vec3 high = surface[0];
    for (uint index = 1u; index < count; ++index)
    {
        low = component_min(low, surface[index]);
        high = component_max(high, surface[index]);
    }
    *minimum = low;
    *maximum = high;
}
