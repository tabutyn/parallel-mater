#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
#include "dispatch_shape.slangh"
// Shared particle-liquid simulation kernels.

import types;
import math;
import device_intrinsics;
import graphics_resources;

#ifndef PM_TARGET_CUDA
#define PM_TARGET_CUDA 1
#endif

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

public uint fluid_lower_bound(PMBuffer<uint64_t> keys, uint size, uint64_t key)
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
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_emit_cells(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<uint> count, uniform uint capacity,
    uniform float inverse_radius, uniform PMBuffer<uint64_t> keys,
    uniform PMBuffer<uint> indices PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_compute_forces(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities, uniform PMBuffer<uint> count,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> indices, uniform PMBuffer<float> foam,
    uniform FluidOptions options, uniform Vec3 up, uniform PMBuffer<Vec3> forces,
    uniform PMBuffer<float> foam_source, uniform PMBuffer<uint> overflow,
    uniform PMBuffer<uint> maximum_neighbor_count PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_integrate(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> previous, uniform PMBuffer<float> foam, uniform PMBuffer<Vec3> forces,
    uniform PMBuffer<float> foam_source, uniform PMBuffer<uint> count,
    uniform FluidOptions options, uniform Vec3 gravity, uniform float dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
[numthreads(PM_THREADS(1), 1, 1)]
void fluid_reserve_contact_events(
    uniform PMBuffer<uint> selected_count, uniform PMBuffer<uint> offset,
    uniform PMBuffer<uint> world_count, uniform PMBuffer<uint> overflow,
    uniform uint capacity PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u || pmGroupThreadIndex().x != 0u) return;
    uint used = *world_count;
    uint available = capacity - used;
    uint retained = min(*selected_count, available);
    offset[0] = used;
    world_count[0] = used + retained;
    overflow[0] += *selected_count - retained;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_gather_contact_events(
    uniform PMBuffer<uint> selected, uniform PMBuffer<uint> selected_count,
    uniform PMBuffer<uint> offset, uniform PMBuffer<FluidContactSample> samples,
    uniform PMBuffer<uint> stable_ids, uniform PMBuffer<RigidBodyId> body_ids,
    uniform FluidId fluid, uniform PMBuffer<ContactEvent> events,
    uniform uint capacity PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_source_vacancies(
    uniform PMBuffer<Vec3> points, uniform uint amount, uniform float spacing,
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> indices,
    uniform uint capacity, uniform float inverse_cell_size,
    uniform PMByteBuffer vacant PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
[numthreads(PM_THREADS(1), 1, 1)]
void fluid_source_emit(
    uniform PMBuffer<Vec3> points, uniform PMByteBuffer vacant, uniform uint amount,
    uniform float spacing, uniform Vec3 velocity, uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<uint> ids, uniform PMBuffer<float> foam,
    uniform PMBuffer<float> temperatures, uniform float initial_temperature,
    uniform PMBuffer<uint> count, uniform uint capacity, uniform uint first_spawned,
    uniform uint first_id, uniform PMBuffer<uint> misses PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmGroupThreadIndex().x != 0u || pmBlockIndex().x != 0u) return;
    misses[0] = 0u;
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
            ++misses[0];
            continue;
        }
        uint index = count[0]++;
        positions[index] = p;
        velocities[index] = velocity;
        ids[index] = first_id + index - first_spawned;
        foam[index] = 0.0f;
        temperatures[index] = initial_temperature;
    }
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_destroy_flags(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> previous, uniform PMBuffer<uint> count,
    uniform uint capacity, uniform ParticleDestroyPlaneOptions plane,
    uniform bool combine, uniform PMByteBuffer keep PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
    PMResourceUint8 survives = PMResourceUint8(!direction || !inBounds);
    keep[item] = combine ? keep[item] & survives : survives;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_gather(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities, uniform PMBuffer<uint> ids,
    uniform PMBuffer<float> foam, uniform PMBuffer<float> temperatures,
    uniform PMBuffer<FluidContactSample> contact_samples,
    uniform PMByteBuffer contact_flags, uniform bool copy_contacts,
    uniform PMBuffer<uint> selected, uniform PMBuffer<uint> count, uniform uint capacity,
    uniform PMBuffer<Vec3> next_positions, uniform PMBuffer<Vec3> next_velocities,
    uniform PMBuffer<uint> next_ids, uniform PMBuffer<float> next_foam,
    uniform PMBuffer<float> next_temperatures,
    uniform PMBuffer<FluidContactSample> next_contact_samples,
    uniform PMByteBuffer next_contact_flags PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_copy_initial(
    uniform PMBuffer<FluidParticle> input, uniform uint count,
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities, uniform PMBuffer<uint> ids,
    uniform PMBuffer<float> foam, uniform PMBuffer<float> temperatures,
    uniform PMBuffer<uint> invalid PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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

static const uint kFluidSecondPopulation = 0x80000000u;
static const uint kFluidLocalIndexMask = 0x7fffffffu;

float fluid_particle_mass(FluidOptions options)
{
    float diameter = 2.0f * options.particle_radius;
    float volume = options.rest_particle_volume > 0.0f
        ? options.rest_particle_volume : diameter * diameter * diameter;
    return options.rest_density * volume;
}

// Three-dimensional poly6 sample. Writing the normalized form in terms of
// r^2 / h^2 avoids h^9 and keeps density values in kg/m^3.
float fluid_density_kernel(float squared_distance, float support_radius)
{
    float support_squared = support_radius * support_radius;
    if (squared_distance >= support_squared) return 0.0f;
    float q_squared = max(0.0f, 1.0f - squared_distance / support_squared);
    float support_cubed = support_radius * support_squared;
    return 1.56668147106f * q_squared * q_squared * q_squared /
        support_cubed;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_emit_pair_cells(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform float inverse_radius, uniform PMBuffer<uint64_t> keys,
    uniform PMBuffer<uint> identities PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint item = pmThreadIndex();
    uint totalCapacity = first_capacity + second_capacity;
    if (item >= totalCapacity) return;
    bool second = item >= first_capacity;
    uint local = second ? item - first_capacity : item;
    identities[item] = local | (second ? kFluidSecondPopulation : 0u);
    uint count = second ? *second_count : *first_count;
    if (local >= count)
    {
        keys[item] = kFluidEmptyCell;
        return;
    }
    Vec3 position = second ? second_positions[local] : first_positions[local];
    keys[item] = fluid_cell_key(
        int(floor(position.x * inverse_radius)),
        int(floor(position.y * inverse_radius)),
        int(floor(position.z * inverse_radius)));
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_count_pair_neighbors(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform uint first_maximum_neighbors,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform uint second_maximum_neighbors,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> identities,
    uniform uint total_capacity, uniform FluidOptions first_options,
    uniform FluidOptions second_options,
    uniform PMBuffer<uint> first_total_neighbors,
    uniform PMBuffer<uint> first_cross_neighbors,
    uniform PMBuffer<float> first_density_samples,
    uniform PMBuffer<uint> second_total_neighbors,
    uniform PMBuffer<uint> second_cross_neighbors,
    uniform PMBuffer<float> second_density_samples,
    uniform PMBuffer<uint> first_neighbor_overflow,
    uniform PMBuffer<uint> second_neighbor_overflow,
    uniform PMBuffer<uint> first_maximum_neighbor_count,
    uniform PMBuffer<uint> second_maximum_neighbor_count
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint item = pmThreadIndex();
    if (item >= total_capacity || item >= first_capacity + second_capacity)
        return;
    bool second = item >= first_capacity;
    uint local = second ? item - first_capacity : item;
    uint count = second ? *second_count : *first_count;
    if (second)
    {
        second_total_neighbors[local] = 0u;
        second_cross_neighbors[local] = 0u;
        second_density_samples[local] = 0.0f;
    }
    else
    {
        first_total_neighbors[local] = 0u;
        first_cross_neighbors[local] = 0u;
        first_density_samples[local] = 0.0f;
    }
    if (local >= count) return;

    Vec3 position = second ? second_positions[local] : first_positions[local];
    uint selfIdentity = local | (second ? kFluidSecondPopulation : 0u);
    FluidOptions localOptions = second ? second_options : first_options;
    float pairSupport = max(first_options.support_radius,
        second_options.support_radius);
    float inverseRadius = 1.0f / pairSupport;
    int cx = int(floor(position.x * inverseRadius));
    int cy = int(floor(position.y * inverseRadius));
    int cz = int(floor(position.z * inverseRadius));
    uint neighbors = 0u;
    uint crossNeighbors = 0u;
    float density = fluid_particle_mass(localOptions) *
        fluid_density_kernel(0.0f, localOptions.support_radius);
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
                for (uint sorted = fluid_lower_bound(keys, total_capacity, key);
                     sorted < total_capacity && keys[sorted] == key; ++sorted)
                {
                    uint otherIdentity = identities[sorted];
                    if (otherIdentity == selfIdentity) continue;
                    bool otherSecond =
                        (otherIdentity & kFluidSecondPopulation) != 0u;
                    uint other = otherIdentity & kFluidLocalIndexMask;
                    Vec3 otherPosition = otherSecond
                        ? second_positions[other] : first_positions[other];
                    bool crossPopulation = otherSecond != second;
                    float support = crossPopulation
                        ? pairSupport : localOptions.support_radius;
                    float squared = length_squared(
                        subtract(position, otherPosition));
                    if (squared >= support * support)
                        continue;
                    ++neighbors;
                    if (crossPopulation) ++crossNeighbors;
                    FluidOptions otherOptions = otherSecond
                        ? second_options : first_options;
                    density += fluid_particle_mass(otherOptions) *
                        fluid_density_kernel(squared, support);
                }
            }
        }
    }

    if (second)
    {
        second_total_neighbors[local] = neighbors;
        second_cross_neighbors[local] = crossNeighbors;
        second_density_samples[local] = density;
        pmAtomicMax(second_maximum_neighbor_count, neighbors);
        if (neighbors > second_maximum_neighbors)
            pmAtomicAdd(second_neighbor_overflow, 1u);
    }
    else
    {
        first_total_neighbors[local] = neighbors;
        first_cross_neighbors[local] = crossNeighbors;
        first_density_samples[local] = density;
        pmAtomicMax(first_maximum_neighbor_count, neighbors);
        if (neighbors > first_maximum_neighbors)
            pmAtomicAdd(first_neighbor_overflow, 1u);
    }
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_apply_pair_pressure_contact(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<Vec3> first_velocities,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform PMBuffer<Vec3> first_forces,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<Vec3> second_velocities,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform PMBuffer<Vec3> second_forces,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> identities,
    uniform uint total_capacity, uniform FluidOptions options
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint item = pmThreadIndex();
    if (item >= total_capacity || item >= first_capacity + second_capacity)
        return;
    bool second = item >= first_capacity;
    uint local = second ? item - first_capacity : item;
    uint count = second ? *second_count : *first_count;
    if (local >= count) return;

    Vec3 position = second ? second_positions[local] : first_positions[local];
    Vec3 velocity = second ? second_velocities[local] : first_velocities[local];
    uint selfIdentity = local | (second ? kFluidSecondPopulation : 0u);
    float inverseRadius = 1.0f / options.support_radius;
    int cx = int(floor(position.x * inverseRadius));
    int cy = int(floor(position.y * inverseRadius));
    int cz = int(floor(position.z * inverseRadius));
    float supportSquared = options.support_radius * options.support_radius;
    Vec3 acceleration = {};
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
                for (uint sorted = fluid_lower_bound(keys, total_capacity, key);
                     sorted < total_capacity && keys[sorted] == key; ++sorted)
                {
                    uint otherIdentity = identities[sorted];
                    bool otherSecond =
                        (otherIdentity & kFluidSecondPopulation) != 0u;
                    if (otherSecond == second) continue;
                    uint other = otherIdentity & kFluidLocalIndexMask;
                    Vec3 otherPosition = otherSecond
                        ? second_positions[other] : first_positions[other];
                    Vec3 delta = subtract(position, otherPosition);
                    float squared = length_squared(delta);
                    if (squared >= supportSquared) continue;
                    float distance = sqrt(max(squared, 1.0e-12f));
                    Vec3 negativeX = {-1.0f, 0.0f, 0.0f};
                    Vec3 positiveX = {1.0f, 0.0f, 0.0f};
                    Vec3 direction = squared > 1.0e-12f
                        ? multiply(delta, 1.0f / distance)
                        : (selfIdentity < otherIdentity ? negativeX : positiveX);
                    Vec3 otherVelocity = otherSecond
                        ? second_velocities[other] : first_velocities[other];
                    Vec3 relativeVelocity = subtract(otherVelocity, velocity);
                    float radialSpeed = dot(relativeVelocity, direction);
                    float q = 1.0f - distance * inverseRadius;
                    acceleration = add(acceleration,
                        multiply(direction, options.repulsion *
                            (1000.0f / options.rest_density) * q * q +
                            options.normal_damping * radialSpeed));
                }
            }
        }
    }
    if (second)
        second_forces[local] = add(second_forces[local], acceleration);
    else
        first_forces[local] = add(first_forces[local], acceleration);
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_apply_pair_viscosity(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<Vec3> first_velocities,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform PMBuffer<Vec3> first_forces,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<Vec3> second_velocities,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform PMBuffer<Vec3> second_forces,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> identities,
    uniform uint total_capacity, uniform FluidOptions options
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint item = pmThreadIndex();
    if (item >= total_capacity || item >= first_capacity + second_capacity)
        return;
    bool second = item >= first_capacity;
    uint local = second ? item - first_capacity : item;
    uint count = second ? *second_count : *first_count;
    if (local >= count) return;

    Vec3 position = second ? second_positions[local] : first_positions[local];
    Vec3 velocity = second ? second_velocities[local] : first_velocities[local];
    float inverseRadius = 1.0f / options.support_radius;
    int cx = int(floor(position.x * inverseRadius));
    int cy = int(floor(position.y * inverseRadius));
    int cz = int(floor(position.z * inverseRadius));
    float supportSquared = options.support_radius * options.support_radius;
    Vec3 acceleration = {};
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
                for (uint sorted = fluid_lower_bound(keys, total_capacity, key);
                     sorted < total_capacity && keys[sorted] == key; ++sorted)
                {
                    uint otherIdentity = identities[sorted];
                    bool otherSecond =
                        (otherIdentity & kFluidSecondPopulation) != 0u;
                    if (otherSecond == second) continue;
                    uint other = otherIdentity & kFluidLocalIndexMask;
                    Vec3 otherPosition = otherSecond
                        ? second_positions[other] : first_positions[other];
                    Vec3 delta = subtract(position, otherPosition);
                    float squared = length_squared(delta);
                    if (squared >= supportSquared) continue;
                    float distance = sqrt(max(squared, 1.0e-12f));
                    float q = 1.0f - distance * inverseRadius;
                    Vec3 otherVelocity = otherSecond
                        ? second_velocities[other] : first_velocities[other];
                    acceleration = add(acceleration,
                        multiply(subtract(otherVelocity, velocity),
                            options.viscosity * q));
                }
            }
        }
    }
    if (second)
        second_forces[local] = add(second_forces[local], acceleration);
    else
        first_forces[local] = add(first_forces[local], acceleration);
}

public float fluid_pair_pressure(
    FluidOptions options, float density)
{
    return options.repulsion * max(density - options.rest_density, 0.0f);
}

public float fluid_pressure_gradient_magnitude(
    float distance, float support_radius)
{
    float supportSquared = support_radius * support_radius;
    float qSquared = max(
        0.0f, 1.0f - distance * distance / supportSquared);
    return 9.40008882636f * distance * qSquared * qSquared /
        (supportSquared * supportSquared * support_radius);
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_apply_miscible_pair_pressure_contact(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<Vec3> first_velocities,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform PMBuffer<Vec3> first_forces,
    uniform PMBuffer<float> first_density_samples,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<Vec3> second_velocities,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform PMBuffer<Vec3> second_forces,
    uniform PMBuffer<float> second_density_samples,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> identities,
    uniform uint total_capacity,
    uniform FluidOptions first_options,
    uniform FluidOptions second_options
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint item = pmThreadIndex();
    if (item >= total_capacity || item >= first_capacity + second_capacity)
        return;
    bool second = item >= first_capacity;
    uint local = second ? item - first_capacity : item;
    uint count = second ? *second_count : *first_count;
    if (local >= count) return;

    Vec3 position = second ? second_positions[local] : first_positions[local];
    float pairSupport = max(
        first_options.support_radius, second_options.support_radius);
    float inverseSupport = 1.0f / pairSupport;
    int cx = int(floor(position.x * inverseSupport));
    int cy = int(floor(position.y * inverseSupport));
    int cz = int(floor(position.z * inverseSupport));
    float supportSquared = pairSupport * pairSupport;
    float firstMass = fluid_particle_mass(first_options);
    float secondMass = fluid_particle_mass(second_options);
    float pairMass = 2.0f * firstMass * secondMass /
        max(firstMass + secondMass, 1.0e-20f);
    float contactSeparation =
        first_options.particle_radius + second_options.particle_radius;
    float firstContactScale = first_options.repulsion *
        (1000.0f / first_options.rest_density);
    float secondContactScale = second_options.repulsion *
        (1000.0f / second_options.rest_density);
    Vec3 acceleration = {};

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
                for (uint sorted = fluid_lower_bound(keys, total_capacity, key);
                     sorted < total_capacity && keys[sorted] == key; ++sorted)
                {
                    uint otherIdentity = identities[sorted];
                    bool otherSecond =
                        (otherIdentity & kFluidSecondPopulation) != 0u;
                    if (otherSecond == second) continue;
                    uint other = otherIdentity & kFluidLocalIndexMask;

                    uint firstIndex = second ? other : local;
                    uint secondIndex = second ? local : other;
                    Vec3 firstPosition = first_positions[firstIndex];
                    Vec3 secondPosition = second_positions[secondIndex];
                    Vec3 delta = subtract(firstPosition, secondPosition);
                    float squared = length_squared(delta);
                    if (squared >= supportSquared) continue;

                    Vec3 firstVelocity = first_velocities[firstIndex];
                    Vec3 secondVelocity = second_velocities[secondIndex];
                    Vec3 relativeVelocity =
                        subtract(secondVelocity, firstVelocity);
                    float distance = sqrt(max(squared, 1.0e-12f));
                    Vec3 direction = squared > 1.0e-12f
                        ? multiply(delta, 1.0f / distance)
                        : normalized_or(relativeVelocity, Vec3(0.0f));
                    float firstDensity = max(
                        first_density_samples[firstIndex], 1.0e-6f);
                    float secondDensity = max(
                        second_density_samples[secondIndex], 1.0e-6f);
                    float firstPressure = fluid_pair_pressure(
                        first_options, firstDensity);
                    float secondPressure = fluid_pair_pressure(
                        second_options, secondDensity);
                    float pressureMagnitude = firstMass * secondMass *
                        (firstPressure / (firstDensity * firstDensity) +
                         secondPressure / (secondDensity * secondDensity)) *
                        fluid_pressure_gradient_magnitude(
                            distance, pairSupport);

                    float contactMagnitude = 0.0f;
                    if (distance < contactSeparation)
                    {
                        float q = 1.0f - distance / contactSeparation;
                        float radialSpeed = dot(relativeVelocity, direction);
                        float contactAcceleration =
                            0.5f * (firstContactScale + secondContactScale) *
                                q * q +
                            0.5f * (first_options.normal_damping +
                                second_options.normal_damping) * radialSpeed;
                        contactMagnitude = pairMass * contactAcceleration;
                    }

                    Vec3 commonForce = multiply(
                        direction, pressureMagnitude + contactMagnitude);
                    float forceLimit = 0.0f;
                    if (first_options.maximum_pair_acceleration > 0.0f)
                        forceLimit = firstMass *
                            first_options.maximum_pair_acceleration;
                    if (second_options.maximum_pair_acceleration > 0.0f)
                    {
                        float secondLimit = secondMass *
                            second_options.maximum_pair_acceleration;
                        forceLimit = forceLimit > 0.0f
                            ? min(forceLimit, secondLimit) : secondLimit;
                    }
                    if (forceLimit > 0.0f)
                        commonForce = clamp_length(commonForce, forceLimit);
                    acceleration = add(acceleration,
                        multiply(commonForce,
                            second ? -1.0f / secondMass : 1.0f / firstMass));
                }
            }
        }
    }

    if (second)
        second_forces[local] = add(second_forces[local], acceleration);
    else
        first_forces[local] = add(first_forces[local], acceleration);
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_apply_miscible_pair_viscosity(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<Vec3> first_velocities,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform PMBuffer<Vec3> first_forces,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<Vec3> second_velocities,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform PMBuffer<Vec3> second_forces,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> identities,
    uniform uint total_capacity,
    uniform FluidOptions first_options,
    uniform FluidOptions second_options
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint item = pmThreadIndex();
    if (item >= total_capacity || item >= first_capacity + second_capacity)
        return;
    bool second = item >= first_capacity;
    uint local = second ? item - first_capacity : item;
    uint count = second ? *second_count : *first_count;
    if (local >= count) return;

    Vec3 position = second ? second_positions[local] : first_positions[local];
    float pairSupport = max(
        first_options.support_radius, second_options.support_radius);
    float inverseSupport = 1.0f / pairSupport;
    int cx = int(floor(position.x * inverseSupport));
    int cy = int(floor(position.y * inverseSupport));
    int cz = int(floor(position.z * inverseSupport));
    float supportSquared = pairSupport * pairSupport;
    float firstMass = fluid_particle_mass(first_options);
    float secondMass = fluid_particle_mass(second_options);
    float firstVolume = firstMass / first_options.rest_density;
    float secondVolume = secondMass / second_options.rest_density;
    float pairVolume = 2.0f * firstVolume * secondVolume /
        max(firstVolume + secondVolume, 1.0e-20f);
    float firstDynamicViscosity =
        first_options.rest_density * first_options.viscosity;
    float secondDynamicViscosity =
        second_options.rest_density * second_options.viscosity;
    float viscositySum = firstDynamicViscosity + secondDynamicViscosity;
    float pairDynamicViscosity = viscositySum > 0.0f
        ? 2.0f * firstDynamicViscosity * secondDynamicViscosity /
            viscositySum
        : 0.0f;
    float forceLimit = 0.0f;
    if (first_options.maximum_pair_acceleration > 0.0f)
        forceLimit = firstMass * first_options.maximum_pair_acceleration;
    if (second_options.maximum_pair_acceleration > 0.0f)
    {
        float secondLimit = secondMass *
            second_options.maximum_pair_acceleration;
        forceLimit = forceLimit > 0.0f
            ? min(forceLimit, secondLimit) : secondLimit;
    }
    Vec3 acceleration = {};

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
                for (uint sorted = fluid_lower_bound(keys, total_capacity, key);
                     sorted < total_capacity && keys[sorted] == key; ++sorted)
                {
                    uint otherIdentity = identities[sorted];
                    bool otherSecond =
                        (otherIdentity & kFluidSecondPopulation) != 0u;
                    if (otherSecond == second) continue;
                    uint other = otherIdentity & kFluidLocalIndexMask;
                    uint firstIndex = second ? other : local;
                    uint secondIndex = second ? local : other;
                    float squared = length_squared(subtract(
                        first_positions[firstIndex],
                        second_positions[secondIndex]));
                    if (squared >= supportSquared) continue;
                    float distance = sqrt(max(squared, 0.0f));
                    float weight = 1.0f - distance * inverseSupport;
                    Vec3 commonForce = multiply(subtract(
                        second_velocities[secondIndex],
                        first_velocities[firstIndex]),
                        pairDynamicViscosity * pairVolume * weight);
                    if (forceLimit > 0.0f)
                        commonForce = clamp_length(commonForce, forceLimit);
                    acceleration = add(acceleration,
                        multiply(commonForce,
                            second ? -1.0f / secondMass : 1.0f / firstMass));
                }
            }
        }
    }

    if (second)
        second_forces[local] = add(second_forces[local], acceleration);
    else
        first_forces[local] = add(first_forces[local], acceleration);
}
