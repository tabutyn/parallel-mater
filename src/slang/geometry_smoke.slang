// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_smoke.cuh. Rigid-contact entries follow the
// standalone particle solver and shared wind sampler below.

import cuda_common;
import fluid;
import geometry_fluid;
import smoke_grid;

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

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_rigid_contact(
    uniform Vec3* positions, uniform Vec3* previous_positions,
    uniform Vec3* velocities, uniform float* ages, uniform float* pressures,
    uniform uint count, uniform float lifetime,
    uniform TriangleMeshResource mesh, uniform RigidBodyState* states,
    uniform RigidBodyState* previous_states, uniform uint body,
    uniform float clearance, uniform float particle_mass,
    uniform float wall_drag, uniform float pressure_stiffness,
    uniform float dt, uniform float maximum_speed,
    uniform SmokeGridField grid, uniform FluidBodyImpulse* impulses)
{
    uint particle = pmThreadIndex();
    if (particle >= count) return;
    impulses[particle] = {};
    if (ages[particle] >= lifetime) return;
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
[numthreads(128, 1, 1)]
void smoke_apply_rigid_impulses(
    uniform FluidBodyImpulse* impulses, uniform uint count,
    uniform uint body, uniform BodyParameters* parameters,
    uniform RigidBodyState* states)
{
    if (cudaBlockIdx().x != 0u ||
        parameters[body].motion != MotionType.dynamic) return;
    uint lane = cudaThreadIdx().x;
    Vec3 localLinear = {}, localAngular = {};
    for (uint particle = lane; particle < count;
         particle += cudaBlockDim().x)
    {
        FluidBodyImpulse item = impulses[particle];
        if (item.body != body) continue;
        localLinear = add(localLinear, item.linear);
        localAngular = add(localAngular, item.angular);
    }
    smoke_impulse_linear[lane] = localLinear;
    smoke_impulse_angular[lane] = localAngular;
    pmGroupBarrier();
    for (uint stride = cudaBlockDim().x / 2u; stride != 0u; stride /= 2u)
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
