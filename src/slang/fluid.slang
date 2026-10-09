// SPDX-License-Identifier: MIT
// Slang mirror of src/fluid.cuh.

import cuda_common;

static const int kFluidCellBias = 1 << 20;
static const uint64_t kFluidEmptyCell = 0xffffffffffffffffull;

public uint64_t fluid_cell_key(int x, int y, int z)
{
    x = max(-kFluidCellBias, min(kFluidCellBias - 1, x));
    y = max(-kFluidCellBias, min(kFluidCellBias - 1, y));
    z = max(-kFluidCellBias, min(kFluidCellBias - 1, z));
    return (uint64_t(x + kFluidCellBias) << 42u) |
           (uint64_t(y + kFluidCellBias) << 21u) |
           uint64_t(z + kFluidCellBias);
}

public uint fluid_lower_bound(uint64_t* keys, uint size, uint64_t key)
{
    uint lo = 0u;
    uint hi = size;
    while (lo < hi)
    {
        uint middle = lo + (hi - lo) / 2u;
        if (keys[middle] < key) lo = middle + 1u;
        else hi = middle;
    }
    return lo;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_emit_cells(
    uniform Vec3* positions, uniform uint* count, uniform uint capacity,
    uniform float inverse_radius, uniform uint64_t* keys,
    uniform uint* indices)
{
    uint index = pmThreadIndex();
    if (index >= capacity) return;
    indices[index] = index;
    if (index >= *count)
    {
        keys[index] = kFluidEmptyCell;
        return;
    }
    Vec3 position = positions[index];
    keys[index] = fluid_cell_key(
        int(floor(position.x * inverse_radius)),
        int(floor(position.y * inverse_radius)),
        int(floor(position.z * inverse_radius)));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_compute_forces(
    uniform Vec3* positions, uniform Vec3* velocities, uniform uint* count,
    uniform uint64_t* keys, uniform uint* indices, uniform float* foam,
    uniform FluidOptions options, uniform Vec3 up, uniform Vec3* forces,
    uniform float* foam_source, uniform uint* overflow,
    uniform uint* maximum_neighbor_count)
{
    uint particle = pmThreadIndex();
    if (particle >= *count) return;
    Vec3 p = positions[particle];
    Vec3 v = velocities[particle];
    float inverseRadius = 1.0f / options.support_radius;
    int cx = int(floor(p.x * inverseRadius));
    int cy = int(floor(p.y * inverseRadius));
    int cz = int(floor(p.z * inverseRadius));
    Vec3 acceleration = {};
    Vec3 outward = {};
    float weight = 0.0f;
    float relativeSpeedSquared = 0.0f;
    float neighboringFoam = 0.0f;
    uint neighbors = 0u;
    for (int dz = -1; dz <= 1; ++dz)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                if (cx + dx < -kFluidCellBias || cx + dx >= kFluidCellBias ||
                    cy + dy < -kFluidCellBias || cy + dy >= kFluidCellBias ||
                    cz + dz < -kFluidCellBias || cz + dz >= kFluidCellBias)
                    continue;
                uint64_t key = fluid_cell_key(cx + dx, cy + dy, cz + dz);
                for (uint item = fluid_lower_bound(keys, options.capacity, key);
                     item < options.capacity && keys[item] == key; ++item)
                {
                    uint other = indices[item];
                    if (other == particle) continue;
                    Vec3 delta = subtract(p, positions[other]);
                    float squared = length_squared(delta);
                    if (squared >= options.support_radius * options.support_radius)
                        continue;
                    ++neighbors;
                    float distance = sqrt(max(squared, 1.0e-12f));
                    Vec3 negativeX = {-1.0f, 0.0f, 0.0f};
                    Vec3 positiveX = {1.0f, 0.0f, 0.0f};
                    Vec3 direction = squared > 1.0e-12f
                        ? multiply(delta, 1.0f / distance)
                        : (particle < other ? negativeX : positiveX);
                    float q = 1.0f - distance * inverseRadius;
                    outward = add(outward, multiply(direction, q));
                    weight += q;
                    Vec3 relativeVelocity = subtract(velocities[other], v);
                    relativeSpeedSquared += length_squared(relativeVelocity) * q;
                    neighboringFoam = max(neighboringFoam, foam[other] * q);
                    float radialSpeed = dot(relativeVelocity, direction);
                    acceleration = add(acceleration,
                        add(multiply(direction, options.repulsion *
                                (1000.0f / options.rest_density) * q * q +
                                options.normal_damping * radialSpeed),
                            multiply(relativeVelocity, options.viscosity * q)));
                }
            }
        }
    }
    if (options.maximum_pair_acceleration > 0.0f)
        acceleration = clamp_length(
            acceleration, options.maximum_pair_acceleration);
    forces[particle] = acceleration;
    float exposure = vector_length(outward) / max(weight, 1.0e-6f);
    float upward = max(0.0f, dot(normalized_or(outward, up), up));
    float agitation = sqrt(relativeSpeedSquared / max(weight, 1.0e-6f));
    foam_source[particle] = max(
        clamp_scalar((exposure - 0.12f) * 2.0f, 0.0f, 1.0f) * upward *
            clamp_scalar((agitation - 0.15f) * 1.5f, 0.0f, 1.0f),
        neighboringFoam * upward * 0.9f);
    pmAtomicMax(maximum_neighbor_count, neighbors);
    if (neighbors > options.maximum_neighbors) pmAtomicAdd(overflow, 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_integrate(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* previous, uniform float* foam, uniform Vec3* forces,
    uniform float* foam_source, uniform uint* count,
    uniform FluidOptions options, uniform Vec3 gravity, uniform float dt)
{
    uint index = pmThreadIndex();
    if (index >= *count) return;
    previous[index] = positions[index];
    Vec3 velocity = add(velocities[index],
        multiply(add(gravity, forces[index]), dt));
    velocity = clamp_length(
        multiply(velocity, exp(-options.velocity_damping * dt)),
        options.maximum_speed);
    Vec3 position = add(positions[index], multiply(velocity, dt));
    if (isfinite(position.x) && isfinite(position.y) && isfinite(position.z) &&
        isfinite(velocity.x) && isfinite(velocity.y) && isfinite(velocity.z))
    {
        positions[index] = position;
        velocities[index] = velocity;
    }
    foam[index] = max(max(0.0f, foam[index] - dt * 0.7f),
                      foam_source[index]);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void fluid_reserve_contact_events(
    uniform uint* selected_count, uniform uint* offset,
    uniform uint* world_count, uniform uint* overflow,
    uniform uint capacity)
{
    if (cudaBlockIdx().x != 0u || cudaThreadIdx().x != 0u) return;
    uint used = *world_count;
    uint available = capacity - used;
    uint retained = min(*selected_count, available);
    *offset = used;
    *world_count = used + retained;
    *overflow += *selected_count - retained;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_gather_contact_events(
    uniform uint* selected, uniform uint* selected_count,
    uniform uint* offset, uniform FluidContactSample* samples,
    uniform uint* stable_ids, uniform RigidBodyId* body_ids,
    uniform FluidId fluid, uniform ContactEvent* events,
    uniform uint capacity)
{
    uint item = pmThreadIndex();
    if (item >= *selected_count || *offset + item >= capacity) return;
    uint particle = selected[item];
    FluidContactSample sample = samples[particle];
    ContactEvent event;
    event.fluid = fluid;
    event.stable_particle_id = stable_ids[particle];
    event.rigid_body = body_ids[sample.body];
    event.position = sample.position;
    event.normal = sample.normal;
    event.normal_impulse = sample.normal_impulse;
    events[*offset + item] = event;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_source_vacancies(
    uniform Vec3* points, uniform uint amount, uniform float spacing,
    uniform Vec3* positions, uniform uint64_t* keys, uniform uint* indices,
    uniform uint capacity, uniform float inverse_cell_size,
    uniform uint8_t* vacant)
{
    uint item = pmThreadIndex();
    if (item >= amount) return;
    Vec3 p = points[item];
    int cx = max(-kFluidCellBias, min(kFluidCellBias - 1,
        int(floor(p.x * inverse_cell_size))));
    int cy = max(-kFluidCellBias, min(kFluidCellBias - 1,
        int(floor(p.y * inverse_cell_size))));
    int cz = max(-kFluidCellBias, min(kFluidCellBias - 1,
        int(floor(p.z * inverse_cell_size))));
    vacant[item] = 0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                uint64_t key = fluid_cell_key(cx + x, cy + y, cz + z);
                for (uint i = fluid_lower_bound(keys, capacity, key);
                     i < capacity && keys[i] == key; ++i)
                    if (length_squared(subtract(p, positions[indices[i]])) <
                        spacing * spacing) return;
            }
    vacant[item] = 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void fluid_source_emit(
    uniform Vec3* points, uniform uint8_t* vacant, uniform uint amount,
    uniform float spacing, uniform Vec3 velocity, uniform Vec3* positions,
    uniform Vec3* velocities, uniform uint* ids, uniform float* foam,
    uniform float* temperatures, uniform float initial_temperature,
    uniform uint* count, uniform uint capacity, uniform uint first_spawned,
    uniform uint first_id, uniform uint* misses)
{
    if (cudaThreadIdx().x != 0u || cudaBlockIdx().x != 0u) return;
    *misses = 0u;
    uint previousCount = *count;
    for (uint item = 0u; item < amount; ++item)
    {
        if (vacant[item] == 0u) continue;
        Vec3 p = points[item];
        bool occupied = false;
        for (uint i = first_spawned; i < previousCount; ++i)
        {
            if (length_squared(subtract(p, positions[i])) < spacing * spacing)
            {
                occupied = true;
                break;
            }
        }
        if (occupied) continue;
        if (*count == capacity)
        {
            ++*misses;
            continue;
        }
        uint index = (*count)++;
        positions[index] = p;
        velocities[index] = velocity;
        ids[index] = first_id + index - first_spawned;
        foam[index] = 0.0f;
        temperatures[index] = initial_temperature;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_destroy_flags(
    uniform Vec3* positions, uniform Vec3* previous, uniform uint* count,
    uniform uint capacity, uniform ParticleDestroyPlaneOptions plane,
    uniform bool combine, uniform uint8_t* keep)
{
    uint item = pmThreadIndex();
    if (item >= capacity) return;
    if (item >= *count)
    {
        keep[item] = 0u;
        return;
    }
    Vec3 before = inverse_rotate(plane.plane.orientation,
        subtract(previous[item], plane.plane.center));
    Vec3 after = inverse_rotate(plane.plane.orientation,
        subtract(positions[item], plane.plane.center));
    bool along = before.y < 0.0f && after.y >= 0.0f;
    bool against = before.y > 0.0f && after.y <= 0.0f;
    bool direction = plane.crossing == CrossingDirection.either
        ? along || against
        : plane.crossing == CrossingDirection.along_normal ? along : against;
    float fraction = before.y / (before.y - after.y + 1.0e-20f);
    Vec3 crossing = add(before, multiply(subtract(after, before), fraction));
    bool inBounds = abs(crossing.x) <= plane.plane.half_extents.x &&
                    abs(crossing.z) <= plane.plane.half_extents.y;
    uint8_t survives = uint8_t(!direction || !inBounds);
    keep[item] = combine ? keep[item] & survives : survives;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_gather(
    uniform Vec3* positions, uniform Vec3* velocities, uniform uint* ids,
    uniform float* foam, uniform float* temperatures,
    uniform FluidContactSample* contact_samples,
    uniform uint8_t* contact_flags, uniform bool copy_contacts,
    uniform uint* selected, uniform uint* count, uniform uint capacity,
    uniform Vec3* next_positions, uniform Vec3* next_velocities,
    uniform uint* next_ids, uniform float* next_foam,
    uniform float* next_temperatures,
    uniform FluidContactSample* next_contact_samples,
    uniform uint8_t* next_contact_flags)
{
    uint item = pmThreadIndex();
    if (item >= capacity || item >= *count) return;
    uint source = selected[item];
    next_positions[item] = positions[source];
    next_velocities[item] = velocities[source];
    next_ids[item] = ids[source];
    next_foam[item] = foam[source];
    next_temperatures[item] = temperatures[source];
    if (copy_contacts)
    {
        next_contact_samples[item] = contact_samples[source];
        next_contact_flags[item] = contact_flags[source];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_copy_initial(
    uniform FluidParticle* input, uniform uint count,
    uniform Vec3* positions, uniform Vec3* velocities, uniform uint* ids,
    uniform float* foam, uniform float* temperatures,
    uniform uint* invalid)
{
    uint index = pmThreadIndex();
    if (index >= count) return;
    FluidParticle particle = input[index];
    if (!isfinite(particle.position.x) ||
        !isfinite(particle.position.y) ||
        !isfinite(particle.position.z) ||
        !isfinite(particle.velocity.x) ||
        !isfinite(particle.velocity.y) ||
        !isfinite(particle.velocity.z) ||
        !isfinite(particle.temperature))
    {
        pmAtomicExchange(invalid, 1u);
        return;
    }
    positions[index] = particle.position;
    velocities[index] = particle.velocity;
    ids[index] = index;
    foam[index] = 0.0f;
    temperatures[index] = particle.temperature;
}
