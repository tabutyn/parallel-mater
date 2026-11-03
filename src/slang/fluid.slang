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

// ABI mirrors PmSpatial* in primitives/spatial_build.slang. Fluid-domain
// kernels stage and consume leaves; the shared spatial service alone builds,
// admits and publishes the hierarchy.
struct FluidSpatialAabb
{
    float4 minimum;
    float4 maximum;
};

struct FluidSpatialIdentity
{
    uint family;
    uint owner_slot;
    uint owner_generation;
    uint topology_revision;
    uint primitive_kind;
    uint primitive_slot;
    uint primitive_generation;
};

struct FluidSpatialLeafMetadata
{
    FluidSpatialIdentity identity;
    uint sort_key;
    uint payload_ref;
    uint query_mask;
    uint admission_mask;
    uint flags;
    uint transfer_group;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static const uint kFluidSpatialFamily = 2u;
static const uint kFluidSpatialParticlePrimitive = 1u;
static const uint kFluidSpatialLeafRetained = 1u << 0u;
static const uint kFluidSpatialLeafProvisional = 1u << 1u;
static const uint kFluidSpatialLeafInactive = 1u << 4u;
static const uint kFluidSpatialInvalidBounds = 1u;
static const uint kFluidSpatialCountOverflow = 1u << 2u;
static const uint kFluidSpatialInvalidIdentity = 1u << 8u;
static const uint kFluidSpatialStaleEpoch = 1u << 9u;
static const uint kFluidSpatialQueryOverflow = 1u << 10u;
static const uint kFluidSpatialAdmissionInvalid = 1u << 12u;

bool fluid_spatial_finite_vec3(Vec3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

bool fluid_spatial_nonnegative_vec3(Vec3 value)
{
    return value.x >= 0.0f && value.y >= 0.0f && value.z >= 0.0f;
}

FluidSpatialAabb fluid_spatial_point_bounds(
    Vec3 start, Vec3 predicted, Vec3 pathDeviation, Vec3 correctionBudget,
    float radius)
{
    Vec3 expansion = add(add(pathDeviation, correctionBudget),
        Vec3(radius, radius, radius));
    Vec3 minimum = subtract(component_min(start, predicted), expansion);
    Vec3 maximum = add(component_max(start, predicted), expansion);
    FluidSpatialAabb result;
    result.minimum = float4(minimum.x, minimum.y, minimum.z, 0.0f);
    result.maximum = float4(maximum.x, maximum.y, maximum.z, 0.0f);
    return result;
}

bool fluid_spatial_valid_bounds(FluidSpatialAabb value)
{
    return all(isfinite(value.minimum.xyz)) &&
        all(isfinite(value.maximum.xyz)) &&
        all(value.minimum.xyz <= value.maximum.xyz);
}

bool fluid_spatial_identity_equal(
    FluidSpatialIdentity left, FluidSpatialIdentity right)
{
    return left.family == right.family &&
        left.owner_slot == right.owner_slot &&
        left.owner_generation == right.owner_generation &&
        left.topology_revision == right.topology_revision &&
        left.primitive_kind == right.primitive_kind &&
        left.primitive_slot == right.primitive_slot &&
        left.primitive_generation == right.primitive_generation;
}

bool fluid_spatial_current_particle(
    FluidSpatialLeafMetadata leaf, uint ownerSlot, uint ownerGeneration,
    PMBuffer<uint> stableIds, PMBuffer<uint> count, uint capacity)
{
    uint dense = leaf.payload_ref;
    return leaf.identity.family == kFluidSpatialFamily &&
        leaf.identity.primitive_kind == kFluidSpatialParticlePrimitive &&
        leaf.identity.owner_slot == ownerSlot &&
        leaf.identity.owner_generation == ownerGeneration &&
        dense < capacity && dense < count[0] &&
        stableIds[dense] == leaf.identity.primitive_slot;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_emit_spatial_particles(
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> predicted_positions,
    uniform PMBuffer<uint> stable_ids, uniform PMBuffer<uint> count,
    uniform uint particle_capacity, uniform uint owner_slot,
    uniform uint owner_generation, uniform uint material_revision,
    uniform uint leaf_offset, uniform uint leaf_capacity,
    uniform uint query_offset, uniform uint query_capacity,
    uniform uint sort_key_base, uniform uint query_mask,
    uniform uint admission_mask, uniform float admission_radius,
    uniform float query_radius, uniform Vec3 path_deviation,
    uniform Vec3 correction_budget, uniform float epsilon,
    uniform PMBuffer<FluidSpatialLeafMetadata> metadata,
    uniform PMBuffer<FluidSpatialAabb> bounds,
    uniform PMByteBuffer build_flags,
    uniform PMBuffer<FluidSpatialAabb> query_bounds,
    uniform PMBuffer<uint> query_masks,
    uniform PMBuffer<uint> excluded_sort_keys,
    uniform PMBuffer<uint> query_leaf_sources,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint dense = pmThreadIndex();
    if (dense >= particle_capacity) return;
    if (leaf_offset > leaf_capacity || dense >= leaf_capacity - leaf_offset ||
        query_offset > query_capacity || dense >= query_capacity - query_offset)
    {
        pmAtomicOr(status, kFluidSpatialCountOverflow);
        return;
    }
    uint source = leaf_offset + dense;
    uint query = query_offset + dense;
    FluidSpatialLeafMetadata leaf = {};
    leaf.flags = kFluidSpatialLeafInactive;
    metadata[source] = leaf;
    build_flags[source] = PMResourceUint8(0u);
    query_masks[query] = 0u;
    excluded_sort_keys[query] = 0u;
    query_leaf_sources[query] = source;
    uint live = count[0];
    if (live > particle_capacity)
    {
        if (dense == 0u) pmAtomicOr(status, kFluidSpatialCountOverflow);
        live = particle_capacity;
    }
    if (dense >= live) return;
    bool valid = owner_generation != 0u && material_revision != 0u &&
        query_mask != 0u && admission_mask != 0u &&
        isfinite(admission_radius) && admission_radius >= 0.0f &&
        isfinite(query_radius) && query_radius > 0.0f &&
        isfinite(epsilon) && epsilon > 0.0f &&
        fluid_spatial_finite_vec3(path_deviation) &&
        fluid_spatial_finite_vec3(correction_budget) &&
        fluid_spatial_nonnegative_vec3(path_deviation) &&
        fluid_spatial_nonnegative_vec3(correction_budget) &&
        fluid_spatial_finite_vec3(previous_positions[dense]) &&
        fluid_spatial_finite_vec3(predicted_positions[dense]) &&
        sort_key_base < 0xffffffffu - dense;
    if (!valid)
    {
        pmAtomicOr(status, kFluidSpatialInvalidIdentity);
        return;
    }
    float leafRadius = admission_radius + epsilon;
    float queryRadius = query_radius + epsilon;
    FluidSpatialAabb leafBounds = fluid_spatial_point_bounds(
        previous_positions[dense], predicted_positions[dense],
        path_deviation, correction_budget, leafRadius);
    FluidSpatialAabb queryBounds = fluid_spatial_point_bounds(
        previous_positions[dense], predicted_positions[dense],
        path_deviation, correction_budget, queryRadius);
    if (!fluid_spatial_valid_bounds(leafBounds) ||
        !fluid_spatial_valid_bounds(queryBounds))
    {
        pmAtomicOr(status, kFluidSpatialInvalidBounds);
        return;
    }
    uint sortKey = sort_key_base + dense + 1u;
    leaf.identity.family = kFluidSpatialFamily;
    leaf.identity.owner_slot = owner_slot;
    leaf.identity.owner_generation = owner_generation;
    leaf.identity.topology_revision = material_revision;
    leaf.identity.primitive_kind = kFluidSpatialParticlePrimitive;
    leaf.identity.primitive_slot = stable_ids[dense];
    leaf.identity.primitive_generation = 1u;
    leaf.sort_key = sortKey;
    leaf.payload_ref = dense;
    leaf.query_mask = query_mask;
    leaf.admission_mask = admission_mask;
    leaf.flags = kFluidSpatialLeafRetained;
    metadata[source] = leaf;
    bounds[source] = leafBounds;
    build_flags[source] = PMResourceUint8(1u);
    query_bounds[query] = queryBounds;
    query_masks[query] = query_mask;
    excluded_sort_keys[query] = sortKey;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_emit_spatial_source_proposals(
    uniform PMBuffer<Vec3> points, uniform uint amount,
    uniform float spacing, uniform PMBuffer<uint> count,
    uniform uint particle_capacity, uniform uint first_stable_id,
    uniform uint owner_slot, uniform uint owner_generation,
    uniform uint material_revision, uniform uint leaf_offset,
    uniform uint leaf_capacity, uniform uint query_offset,
    uniform uint query_capacity, uniform uint sort_key_base,
    uniform uint query_mask, uniform uint admission_mask,
    uniform float query_radius, uniform Vec3 path_deviation,
    uniform Vec3 correction_budget, uniform float epsilon,
    uniform PMBuffer<FluidSpatialLeafMetadata> metadata,
    uniform PMBuffer<FluidSpatialAabb> bounds,
    uniform PMByteBuffer build_flags,
    uniform PMBuffer<FluidSpatialAabb> query_bounds,
    uniform PMBuffer<uint> query_masks,
    uniform PMBuffer<uint> excluded_sort_keys,
    uniform PMBuffer<uint> query_leaf_sources,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint proposal = pmThreadIndex();
    if (proposal >= amount) return;
    if (leaf_offset > leaf_capacity ||
        proposal >= leaf_capacity - leaf_offset ||
        query_offset > query_capacity ||
        proposal >= query_capacity - query_offset)
    {
        pmAtomicOr(status, kFluidSpatialCountOverflow);
        return;
    }
    uint source = leaf_offset + proposal;
    uint query = query_offset + proposal;
    FluidSpatialLeafMetadata leaf = {};
    leaf.flags = kFluidSpatialLeafInactive;
    metadata[source] = leaf;
    build_flags[source] = PMResourceUint8(0u);
    query_masks[query] = 0u;
    excluded_sort_keys[query] = 0u;
    query_leaf_sources[query] = source;
    uint live = count[0];
    if (live > particle_capacity)
    {
        if (proposal == 0u) pmAtomicOr(status, kFluidSpatialCountOverflow);
        return;
    }
    if (proposal >= particle_capacity - live) return;
    bool valid = owner_generation != 0u && material_revision != 0u &&
        query_mask != 0u && admission_mask != 0u &&
        isfinite(spacing) && spacing > 0.0f &&
        isfinite(query_radius) && query_radius > 0.0f &&
        isfinite(epsilon) && epsilon > 0.0f &&
        fluid_spatial_finite_vec3(points[proposal]) &&
        fluid_spatial_finite_vec3(path_deviation) &&
        fluid_spatial_finite_vec3(correction_budget) &&
        fluid_spatial_nonnegative_vec3(path_deviation) &&
        fluid_spatial_nonnegative_vec3(correction_budget) &&
        sort_key_base < 0xffffffffu - proposal &&
        first_stable_id <= 0xffffffffu - proposal;
    if (!valid)
    {
        pmAtomicOr(status, kFluidSpatialInvalidIdentity);
        return;
    }
    // Half spacing on every fluid/proposal leaf makes AABB admission
    // conservative. Exact sphere tests still run in the neighbor consumer;
    // admission may intentionally underfill but can never overlap silently.
    float admissionRadius = 0.5f * spacing + epsilon;
    float queryRadius = query_radius + epsilon;
    FluidSpatialAabb leafBounds = fluid_spatial_point_bounds(points[proposal],
        points[proposal], path_deviation, correction_budget, admissionRadius);
    FluidSpatialAabb queryBounds = fluid_spatial_point_bounds(points[proposal],
        points[proposal], path_deviation, correction_budget, queryRadius);
    if (!fluid_spatial_valid_bounds(leafBounds) ||
        !fluid_spatial_valid_bounds(queryBounds))
    {
        pmAtomicOr(status, kFluidSpatialInvalidBounds);
        return;
    }
    uint sortKey = sort_key_base + proposal + 1u;
    leaf.identity.family = kFluidSpatialFamily;
    leaf.identity.owner_slot = owner_slot;
    leaf.identity.owner_generation = owner_generation;
    leaf.identity.topology_revision = material_revision;
    leaf.identity.primitive_kind = kFluidSpatialParticlePrimitive;
    leaf.identity.primitive_slot = first_stable_id + proposal;
    leaf.identity.primitive_generation = 1u;
    leaf.sort_key = sortKey;
    leaf.payload_ref = live + proposal;
    leaf.query_mask = query_mask;
    leaf.admission_mask = admission_mask;
    leaf.flags = kFluidSpatialLeafProvisional;
    metadata[source] = leaf;
    bounds[source] = leafBounds;
    build_flags[source] = PMResourceUint8(1u);
    query_bounds[query] = queryBounds;
    query_masks[query] = query_mask;
    excluded_sort_keys[query] = sortKey;
}

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void fluid_apply_spatial_source_admission(
    uniform PMBuffer<Vec3> points, uniform uint amount,
    uniform Vec3 velocity, uniform float initial_temperature,
    uniform uint proposal_leaf_offset, uniform uint leaf_capacity,
    uniform uint proposal_query_offset, uniform uint query_capacity,
    uniform PMBuffer<FluidSpatialLeafMetadata> metadata,
    uniform PMByteBuffer admission, uniform PMBuffer<uint> query_masks,
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<uint> stable_ids, uniform PMBuffer<float> foam,
    uniform PMBuffer<float> temperatures, uniform PMBuffer<uint> count,
    uniform uint particle_capacity, uniform PMBuffer<uint> accepted,
    uniform PMBuffer<uint> rejected, uniform PMBuffer<uint> capacity_misses,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmGroupThreadIndex().x != 0u || pmBlockIndex().x != 0u) return;
    accepted[0] = 0u;
    rejected[0] = 0u;
    capacity_misses[0] = 0u;
    if (proposal_leaf_offset > leaf_capacity ||
        amount > leaf_capacity - proposal_leaf_offset ||
        proposal_query_offset > query_capacity ||
        amount > query_capacity - proposal_query_offset ||
        count[0] > particle_capacity)
    {
        pmAtomicOr(status, kFluidSpatialCountOverflow);
        return;
    }
    for (uint proposal = 0u; proposal < amount; ++proposal)
    {
        uint source = proposal_leaf_offset + proposal;
        FluidSpatialLeafMetadata leaf = metadata[source];
        if ((leaf.flags & kFluidSpatialLeafInactive) != 0u)
        {
            ++capacity_misses[0];
            query_masks[proposal_query_offset + proposal] = 0u;
            continue;
        }
        if ((leaf.flags & kFluidSpatialLeafProvisional) == 0u)
        {
            pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
            return;
        }
        if (admission[source] == 0u)
        {
            ++rejected[0];
            leaf.flags = kFluidSpatialLeafInactive;
            metadata[source] = leaf;
            query_masks[proposal_query_offset + proposal] = 0u;
            continue;
        }
        if (count[0] == particle_capacity)
        {
            ++capacity_misses[0];
            leaf.flags = kFluidSpatialLeafInactive;
            metadata[source] = leaf;
            admission[source] = PMResourceUint8(0u);
            query_masks[proposal_query_offset + proposal] = 0u;
            continue;
        }
        uint dense = count[0]++;
        positions[dense] = points[proposal];
        velocities[dense] = velocity;
        stable_ids[dense] = leaf.identity.primitive_slot;
        foam[dense] = 0.0f;
        temperatures[dense] = initial_temperature;
        leaf.payload_ref = dense;
        leaf.flags = kFluidSpatialLeafRetained;
        metadata[source] = leaf;
        ++accepted[0];
    }
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_filter_spatial_candidates(
    uniform PMBuffer<FluidSpatialLeafMetadata> metadata,
    uniform PMByteBuffer admission, uniform uint leaf_capacity,
    uniform PMBuffer<uint> query_leaf_sources, uniform uint query_count,
    uniform uint query_owner_slot, uniform uint query_owner_generation,
    uniform PMBuffer<uint> query_stable_ids,
    uniform PMBuffer<uint> query_live_count, uniform uint query_capacity,
    uniform uint target_owner_slot, uniform uint target_owner_generation,
    uniform PMBuffer<uint> target_stable_ids,
    uniform PMBuffer<uint> target_live_count, uniform uint target_capacity,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_sources,
    uniform uint candidates_per_query, uniform uint output_payload_refs,
    uniform uint output_candidates_per_query,
    uniform PMBuffer<uint> filtered_counts,
    uniform PMBuffer<uint> filtered_sources,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint query = pmThreadIndex();
    if (query >= query_count) return;
    filtered_counts[query] = 0u;
    uint querySource = query_leaf_sources[query];
    if (querySource >= leaf_capacity)
    {
        pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
        return;
    }
    if (admission[querySource] == 0u) return;
    FluidSpatialLeafMetadata queryLeaf = metadata[querySource];
    if (!fluid_spatial_current_particle(queryLeaf, query_owner_slot,
        query_owner_generation, query_stable_ids, query_live_count,
        query_capacity))
    {
        pmAtomicOr(status, kFluidSpatialStaleEpoch);
        return;
    }
    uint count = candidate_counts[query];
    if (count > candidates_per_query)
    {
        pmAtomicOr(status, kFluidSpatialQueryOverflow);
        return;
    }
    uint written = 0u;
    uint inputBase = query * candidates_per_query;
    uint outputBase = query * output_candidates_per_query;
    for (uint candidate = 0u; candidate < count; ++candidate)
    {
        uint source = candidate_sources[inputBase + candidate];
        if (source >= leaf_capacity)
        {
            pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
            filtered_counts[query] = 0u;
            return;
        }
        if (admission[source] == 0u) continue;
        FluidSpatialLeafMetadata target = metadata[source];
        if (fluid_spatial_identity_equal(
                queryLeaf.identity, target.identity)) continue;
        if (target.identity.family != kFluidSpatialFamily ||
            target.identity.primitive_kind !=
                kFluidSpatialParticlePrimitive ||
            target.identity.owner_slot != target_owner_slot ||
            target.identity.owner_generation != target_owner_generation)
            continue;
        if (!fluid_spatial_current_particle(target, target_owner_slot,
            target_owner_generation, target_stable_ids, target_live_count,
            target_capacity))
        {
            pmAtomicOr(status, kFluidSpatialStaleEpoch);
            filtered_counts[query] = 0u;
            return;
        }
        if (written >= output_candidates_per_query)
        {
            pmAtomicOr(status, kFluidSpatialQueryOverflow);
            filtered_counts[query] = 0u;
            return;
        }
        filtered_sources[outputBase + written++] = output_payload_refs != 0u
            ? target.payload_ref : source;
    }
    filtered_counts[query] = written;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_count_spatial_neighbors(
    uniform PMBuffer<FluidSpatialLeafMetadata> metadata,
    uniform PMByteBuffer admission, uniform uint leaf_capacity,
    uniform PMBuffer<uint> query_leaf_sources, uniform uint query_count,
    uniform PMBuffer<Vec3> query_positions,
    uniform PMBuffer<uint> query_live_count, uniform uint query_capacity,
    uniform PMBuffer<Vec3> target_positions,
    uniform PMBuffer<uint> target_live_count, uniform uint target_capacity,
    uniform PMBuffer<uint> filtered_counts,
    uniform PMBuffer<uint> filtered_sources,
    uniform uint candidates_per_query, uniform float support_radius,
    uniform uint maximum_neighbors, uniform PMBuffer<uint> neighbor_counts,
    uniform PMBuffer<uint> neighbor_overflow,
    uniform PMBuffer<uint> maximum_neighbor_count,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint query = pmThreadIndex();
    if (query >= query_count) return;
    neighbor_counts[query] = 0u;
    if (!isfinite(support_radius) || support_radius <= 0.0f)
    {
        if (query == 0u) pmAtomicOr(status, kFluidSpatialInvalidBounds);
        return;
    }
    uint querySource = query_leaf_sources[query];
    if (querySource >= leaf_capacity)
    {
        pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
        return;
    }
    if (admission[querySource] == 0u) return;
    FluidSpatialLeafMetadata queryLeaf = metadata[querySource];
    if (queryLeaf.payload_ref >= query_capacity ||
        queryLeaf.payload_ref >= query_live_count[0])
    {
        pmAtomicOr(status, kFluidSpatialStaleEpoch);
        return;
    }
    uint candidateCount = filtered_counts[query];
    if (candidateCount > candidates_per_query)
    {
        pmAtomicOr(status, kFluidSpatialQueryOverflow);
        return;
    }
    Vec3 position = query_positions[queryLeaf.payload_ref];
    float supportSquared = support_radius * support_radius;
    uint neighbors = 0u;
    uint base = query * candidates_per_query;
    for (uint candidate = 0u; candidate < candidateCount; ++candidate)
    {
        uint source = filtered_sources[base + candidate];
        if (source >= leaf_capacity || admission[source] == 0u)
        {
            pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
            neighbor_counts[query] = 0u;
            return;
        }
        uint dense = metadata[source].payload_ref;
        if (dense >= target_capacity || dense >= target_live_count[0])
        {
            pmAtomicOr(status, kFluidSpatialStaleEpoch);
            neighbor_counts[query] = 0u;
            return;
        }
        float squared = length_squared(
            subtract(position, target_positions[dense]));
        if (squared < supportSquared) ++neighbors;
    }
    neighbor_counts[query] = neighbors;
    pmAtomicMax(maximum_neighbor_count, neighbors);
    if (neighbors > maximum_neighbors) pmAtomicAdd(neighbor_overflow, 1u);
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

// realtime-single-step v1 advances particle state once, then performs six
// Jacobi density sweeps against the owner-filtered F8 spatial snapshot. The
// legacy cell/sort/force/integrate path above remains profile 0.
static const uint kFluidRealtimeProfileVersion = 1u;
static const float kFluidRealtimeTimestep = 1.0f / 60.0f;
static const float kFluidRealtimeMaximumCorrectionFraction = 0.25f;
static const uint kFluidRealtimeInvalidProfile = 1u << 13u;
static const uint kFluidRealtimeInvalidTimestep = 1u << 14u;
static const uint kFluidRealtimeInvalidMaterial = 1u << 15u;
static const uint kFluidRealtimeNonfiniteState = 1u << 16u;

bool fluid_realtime_profile_valid(uint profile_version, float dt)
{
    return profile_version == kFluidRealtimeProfileVersion && isfinite(dt) &&
        abs(dt - kFluidRealtimeTimestep) <= 1.0e-7f;
}

bool fluid_realtime_material_valid(FluidOptions options)
{
    float densityResponse = options.repulsion == 0.0f ? 0.0f :
        1.0f - exp(-options.repulsion * kFluidRealtimeTimestep);
    bool responseValid = options.repulsion == 0.0f ||
        (densityResponse >= 0.05f && densityResponse <= 0.995f);
    return isfinite(options.particle_radius) &&
        isfinite(options.support_radius) &&
        isfinite(options.rest_density) &&
        isfinite(options.rest_particle_volume) &&
        isfinite(options.repulsion) && isfinite(options.viscosity) &&
        isfinite(options.normal_damping) &&
        isfinite(options.velocity_damping) &&
        isfinite(options.maximum_speed) && options.particle_radius > 0.0f &&
        options.support_radius >= 0.005f &&
        options.particle_radius <= 0.5f * options.support_radius &&
        options.rest_density > 0.0f && options.rest_particle_volume > 0.0f &&
        options.repulsion >= 0.0f && options.viscosity >= 0.0f &&
        options.normal_damping >= 0.0f && options.velocity_damping >= 0.0f &&
        responseValid &&
        options.maximum_speed >= 0.0f && options.maximum_speed <= 20.0f &&
        options.maximum_speed * kFluidRealtimeTimestep <=
            0.5f * options.support_radius;
}

void fluid_realtime_record_invalid(
    uint profile_version, float dt, FluidOptions options,
    PMBuffer<uint> status)
{
    if (profile_version != kFluidRealtimeProfileVersion)
        pmAtomicOr(status, kFluidRealtimeInvalidProfile);
    if (!isfinite(dt) || abs(dt - kFluidRealtimeTimestep) > 1.0e-7f)
        pmAtomicOr(status, kFluidRealtimeInvalidTimestep);
    if (!fluid_realtime_material_valid(options))
        pmAtomicOr(status, kFluidRealtimeInvalidMaterial);
}

Vec3 fluid_realtime_density_gradient(
    Vec3 delta, uint particle, uint other, float support_radius)
{
    float squared = length_squared(delta);
    if (squared >= support_radius * support_radius) return Vec3(0.0f);
    float distance = sqrt(max(squared, 1.0e-12f));
    Vec3 direction;
    if (squared > 1.0e-12f)
        direction = multiply(delta, 1.0f / distance);
    else
        direction = particle < other
            ? Vec3(-1.0f, 0.0f, 0.0f)
            : Vec3(1.0f, 0.0f, 0.0f);
    float remaining = max(0.0f, support_radius - distance);
    float h2 = support_radius * support_radius;
    float h6 = h2 * h2 * h2;
    // Gradient of the normalized three-dimensional spiky kernel.
    return multiply(direction,
        -14.3239448783f * remaining * remaining / h6);
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_realtime_predict(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<uint> count, uniform uint capacity,
    uniform FluidOptions options, uniform Vec3 gravity, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<Vec3> accumulated_corrections,
    uniform PMBuffer<uint> time_advances,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (!fluid_realtime_profile_valid(profile_version, dt) ||
        !fluid_realtime_material_valid(options))
    {
        if (particle == 0u)
            fluid_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (particle == 0u) ++time_advances[0];
    if (particle >= capacity || particle >= count[0]) return;
    Vec3 position = positions[particle];
    Vec3 velocity = velocities[particle];
    previous_positions[particle] = position;
    accumulated_corrections[particle] = Vec3(0.0f);
    velocity = add(velocity, multiply(gravity, dt));
    velocity = multiply(velocity, exp(-options.velocity_damping * dt));
    float travelSpeed = 0.5f * options.support_radius / dt;
    velocity = clamp_length(velocity,
        min(options.maximum_speed, travelSpeed));
    position = add(position, multiply(velocity, dt));
    if (!fluid_spatial_finite_vec3(position) ||
        !fluid_spatial_finite_vec3(velocity))
    {
        pmAtomicOr(status, kFluidRealtimeNonfiniteState);
        return;
    }
    positions[particle] = position;
    velocities[particle] = velocity;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_realtime_density_lambda(
    uniform PMBuffer<FluidSpatialLeafMetadata> metadata,
    uniform PMByteBuffer admission, uniform uint leaf_capacity,
    uniform PMBuffer<uint> query_leaf_sources, uniform uint query_count,
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<uint> count,
    uniform uint capacity, uniform PMBuffer<uint> filtered_counts,
    uniform PMBuffer<uint> filtered_sources,
    uniform uint candidates_per_query, uniform FluidOptions options,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<float> densities,
    uniform PMBuffer<float> lambdas,
    uniform uint record_neighbors,
    uniform PMBuffer<uint> neighbor_overflow,
    uniform PMBuffer<uint> maximum_neighbor_count,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint query = pmThreadIndex();
    if (!fluid_realtime_profile_valid(profile_version, dt) ||
        !fluid_realtime_material_valid(options))
    {
        if (query == 0u)
            fluid_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (query >= query_count) return;
    uint querySource = query_leaf_sources[query];
    if (querySource >= leaf_capacity || admission[querySource] == 0u)
    {
        if (querySource >= leaf_capacity)
            pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
        return;
    }
    uint particle = metadata[querySource].payload_ref;
    if (particle >= capacity || particle >= count[0])
    {
        pmAtomicOr(status, kFluidSpatialStaleEpoch);
        return;
    }
    uint candidateCount = filtered_counts[query];
    if (candidateCount > candidates_per_query)
    {
        pmAtomicOr(status, kFluidSpatialQueryOverflow);
        return;
    }
    float mass = fluid_particle_mass(options);
    float density = mass * fluid_density_kernel(
        0.0f, options.support_radius);
    Vec3 selfGradient = Vec3(0.0f);
    float gradientSquared = 0.0f;
    uint neighbors = 0u;
    uint base = query * candidates_per_query;
    Vec3 position = positions[particle];
    for (uint candidate = 0u; candidate < candidateCount; ++candidate)
    {
        uint source = filtered_sources[base + candidate];
        if (source >= leaf_capacity || admission[source] == 0u)
        {
            pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
            return;
        }
        uint other = metadata[source].payload_ref;
        if (other >= capacity || other >= count[0])
        {
            pmAtomicOr(status, kFluidSpatialStaleEpoch);
            return;
        }
        Vec3 delta = subtract(position, positions[other]);
        float squared = length_squared(delta);
        if (squared >= options.support_radius * options.support_radius)
            continue;
        ++neighbors;
        density += mass * fluid_density_kernel(
            squared, options.support_radius);
        Vec3 gradient = multiply(fluid_realtime_density_gradient(
            delta, particle, other, options.support_radius),
            mass / options.rest_density);
        selfGradient = add(selfGradient, gradient);
        gradientSquared += length_squared(gradient);
    }
    gradientSquared += length_squared(selfGradient);
    if (record_neighbors != 0u)
    {
        pmAtomicMax(maximum_neighbor_count, neighbors);
        if (neighbors > options.maximum_neighbors)
            pmAtomicAdd(neighbor_overflow, 1u);
    }
    float constraint = max(density / options.rest_density - 1.0f, 0.0f);
    float regularization = 1.0e-6f /
        (options.support_radius * options.support_radius);
    float lambda = -constraint / max(gradientSquared + regularization,
        1.0e-20f);
    if (!isfinite(density) || !isfinite(lambda))
    {
        pmAtomicOr(status, kFluidRealtimeNonfiniteState);
        density = options.rest_density;
        lambda = 0.0f;
    }
    densities[particle] = density;
    lambdas[particle] = lambda;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_realtime_position_delta(
    uniform PMBuffer<FluidSpatialLeafMetadata> metadata,
    uniform PMByteBuffer admission, uniform uint leaf_capacity,
    uniform PMBuffer<uint> query_leaf_sources, uniform uint query_count,
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<uint> count,
    uniform uint capacity, uniform PMBuffer<uint> filtered_counts,
    uniform PMBuffer<uint> filtered_sources,
    uniform uint candidates_per_query, uniform FluidOptions options,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<float> lambdas,
    uniform PMBuffer<Vec3> accumulated_corrections,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint query = pmThreadIndex();
    if (!fluid_realtime_profile_valid(profile_version, dt) ||
        !fluid_realtime_material_valid(options))
    {
        if (query == 0u)
            fluid_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (query >= query_count) return;
    uint querySource = query_leaf_sources[query];
    if (querySource >= leaf_capacity || admission[querySource] == 0u)
    {
        if (querySource >= leaf_capacity)
            pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
        return;
    }
    uint particle = metadata[querySource].payload_ref;
    if (particle >= capacity || particle >= count[0])
    {
        pmAtomicOr(status, kFluidSpatialStaleEpoch);
        return;
    }
    uint candidateCount = filtered_counts[query];
    if (candidateCount > candidates_per_query)
    {
        pmAtomicOr(status, kFluidSpatialQueryOverflow);
        return;
    }
    float mass = fluid_particle_mass(options);
    Vec3 deltaPosition = Vec3(0.0f);
    Vec3 contactPosition = Vec3(0.0f);
    uint base = query * candidates_per_query;
    Vec3 position = positions[particle];
    for (uint candidate = 0u; candidate < candidateCount; ++candidate)
    {
        uint source = filtered_sources[base + candidate];
        if (source >= leaf_capacity || admission[source] == 0u)
        {
            pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
            return;
        }
        uint other = metadata[source].payload_ref;
        if (other >= capacity || other >= count[0])
        {
            pmAtomicOr(status, kFluidSpatialStaleEpoch);
            return;
        }
        Vec3 separation = subtract(position, positions[other]);
        float squared = length_squared(separation);
        if (squared >= options.support_radius * options.support_radius)
            continue;
        Vec3 gradient = fluid_realtime_density_gradient(
            separation, particle, other, options.support_radius);
        deltaPosition = add(deltaPosition, multiply(gradient,
            (lambdas[particle] + lambdas[other]) * mass /
                options.rest_density));
        float distance = sqrt(max(squared, 1.0e-12f));
        float exclusion = 2.0f * options.particle_radius;
        if (distance < exclusion)
        {
            Vec3 negativeX = Vec3(-1.0f, 0.0f, 0.0f);
            Vec3 positiveX = Vec3(1.0f, 0.0f, 0.0f);
            Vec3 direction = squared > 1.0e-12f
                ? multiply(separation, 1.0f / distance)
                : (particle < other ? negativeX : positiveX);
            float q = max(0.0f,
                1.0f - distance / options.support_radius);
            contactPosition = add(contactPosition, multiply(direction,
                options.repulsion * (1000.0f / options.rest_density) *
                    q * q * dt * dt));
        }
    }
    float response = options.repulsion == 0.0f ? 0.0f :
        clamp_scalar(1.0f - exp(-options.repulsion * dt),
            0.05f, 0.995f);
    if (options.maximum_pair_acceleration > 0.0f)
        contactPosition = clamp_length(contactPosition,
            options.maximum_pair_acceleration * dt * dt);
    deltaPosition = add(multiply(deltaPosition, response),
        multiply(contactPosition, response * response));
    float maximumCorrection =
        kFluidRealtimeMaximumCorrectionFraction * options.support_radius;
    float remaining = max(0.0f, maximumCorrection -
        vector_length(accumulated_corrections[particle]));
    deltaPosition = clamp_length(deltaPosition, remaining);
    if (!fluid_spatial_finite_vec3(deltaPosition))
    {
        pmAtomicOr(status, kFluidRealtimeNonfiniteState);
        deltaPosition = Vec3(0.0f);
    }
    position_deltas[particle] = deltaPosition;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_realtime_apply_position_delta(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<Vec3> accumulated_corrections,
    uniform PMBuffer<uint> count, uniform uint capacity,
    uniform float dt, uniform uint profile_version,
    uniform FluidOptions options,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (!fluid_realtime_profile_valid(profile_version, dt) ||
        !fluid_realtime_material_valid(options))
    {
        if (particle == 0u)
            fluid_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (particle >= capacity || particle >= count[0]) return;
    Vec3 position = add(positions[particle], position_deltas[particle]);
    Vec3 correction = add(accumulated_corrections[particle],
        position_deltas[particle]);
    if (!fluid_spatial_finite_vec3(position) ||
        !fluid_spatial_finite_vec3(correction))
    {
        pmAtomicOr(status, kFluidRealtimeNonfiniteState);
        return;
    }
    positions[particle] = position;
    accumulated_corrections[particle] = correction;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_realtime_viscosity_delta(
    uniform PMBuffer<FluidSpatialLeafMetadata> metadata,
    uniform PMByteBuffer admission, uniform uint leaf_capacity,
    uniform PMBuffer<uint> query_leaf_sources, uniform uint query_count,
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<uint> count, uniform uint capacity,
    uniform PMBuffer<uint> filtered_counts,
    uniform PMBuffer<uint> filtered_sources,
    uniform uint candidates_per_query, uniform FluidOptions options,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<float> densities,
    uniform PMBuffer<Vec3> velocity_deltas,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint query = pmThreadIndex();
    if (!fluid_realtime_profile_valid(profile_version, dt) ||
        !fluid_realtime_material_valid(options))
    {
        if (query == 0u)
            fluid_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (query >= query_count) return;
    uint querySource = query_leaf_sources[query];
    if (querySource >= leaf_capacity || admission[querySource] == 0u)
    {
        if (querySource >= leaf_capacity)
            pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
        return;
    }
    uint particle = metadata[querySource].payload_ref;
    if (particle >= capacity || particle >= count[0])
    {
        pmAtomicOr(status, kFluidSpatialStaleEpoch);
        return;
    }
    uint candidateCount = filtered_counts[query];
    if (candidateCount > candidates_per_query)
    {
        pmAtomicOr(status, kFluidSpatialQueryOverflow);
        return;
    }
    Vec3 viscosityDelta = Vec3(0.0f);
    Vec3 normalDelta = Vec3(0.0f);
    Vec3 position = positions[particle];
    Vec3 velocity = velocities[particle];
    uint base = query * candidates_per_query;
    for (uint candidate = 0u; candidate < candidateCount; ++candidate)
    {
        uint source = filtered_sources[base + candidate];
        if (source >= leaf_capacity || admission[source] == 0u)
        {
            pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
            return;
        }
        uint other = metadata[source].payload_ref;
        if (other >= capacity || other >= count[0])
        {
            pmAtomicOr(status, kFluidSpatialStaleEpoch);
            return;
        }
        Vec3 separation = subtract(position, positions[other]);
        float squared = length_squared(separation);
        if (squared >= options.support_radius * options.support_radius)
            continue;
        float weight = max(0.0f,
            1.0f - sqrt(max(squared, 0.0f)) / options.support_radius);
        Vec3 relative = subtract(velocities[other], velocity);
        viscosityDelta = add(viscosityDelta,
            multiply(relative, weight));
        if (squared > 1.0e-12f)
        {
            Vec3 direction = multiply(separation, rsqrt(squared));
            normalDelta = add(normalDelta, multiply(direction,
                dot(relative, direction) * weight));
        }
    }
    float viscosityResponse = options.viscosity == 0.0f ? 0.0f :
        clamp_scalar(1.0f - exp(-options.viscosity * dt),
            0.0f, 0.995f);
    float normalResponse = options.normal_damping == 0.0f ? 0.0f :
        clamp_scalar(1.0f - exp(-options.normal_damping * dt),
            0.0f, 0.995f);
    Vec3 result = add(multiply(viscosityDelta, viscosityResponse),
        multiply(normalDelta, normalResponse));
    if (!fluid_spatial_finite_vec3(result))
    {
        pmAtomicOr(status, kFluidRealtimeNonfiniteState);
        result = Vec3(0.0f);
    }
    velocity_deltas[particle] = result;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_realtime_commit(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> velocity_deltas,
    uniform PMBuffer<uint> count, uniform uint capacity,
    uniform FluidOptions options, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (!fluid_realtime_profile_valid(profile_version, dt) ||
        !fluid_realtime_material_valid(options))
    {
        if (particle == 0u)
            fluid_realtime_record_invalid(profile_version, dt, options,
                status);
        return;
    }
    if (particle >= capacity || particle >= count[0]) return;
    Vec3 velocity = add(multiply(subtract(positions[particle],
        previous_positions[particle]), 1.0f / dt),
        velocity_deltas[particle]);
    // The host selects one scale for the simulation's configured liquid
    // population.  A truly single-particle simulation preserves authored
    // motion; every domain in a cross-fluid population receives the same
    // stabilizer so damping cannot create a momentum imbalance.
    velocity = multiply(velocity,
        options.realtime_commit_velocity_scale);
    velocity = clamp_length(velocity, options.maximum_speed);
    if (!fluid_spatial_finite_vec3(positions[particle]) ||
        !fluid_spatial_finite_vec3(velocity))
    {
        pmAtomicOr(status, kFluidRealtimeNonfiniteState);
        positions[particle] = previous_positions[particle];
        velocities[particle] = Vec3(0.0f);
        return;
    }
    velocities[particle] = velocity;
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
    uniform uint first_maximum_neighbors,
    uniform PMBuffer<Vec3> first_forces,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<Vec3> second_velocities,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform uint second_maximum_neighbors,
    uniform PMBuffer<Vec3> second_forces,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> identities,
    uniform uint total_capacity, uniform FluidOptions options,
    uniform PMBuffer<uint> neighbor_overflow,
    uniform PMBuffer<uint> maximum_neighbor_count,
    uniform uint count_neighbors
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
                    Vec3 delta = subtract(position, otherPosition);
                    float squared = length_squared(delta);
                    if (squared >= supportSquared) continue;
                    if (otherSecond == second) continue;
                    ++neighbors;
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
    if (count_neighbors != 0u)
    {
        pmAtomicMax(maximum_neighbor_count, neighbors);
        uint limit = second
            ? second_maximum_neighbors : first_maximum_neighbors;
        if (neighbors > limit) pmAtomicAdd(neighbor_overflow, 1u);
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

static const uint kFluidInterfaceSample = 1u;
static const uint kFluidInterfaceUnderResolved = 2u;
static const uint kFluidInterfaceDegenerate = 4u;
static const uint kFluidInterfaceCurvatureClamped = 8u;

// Invert the symmetric weighted position moment used by both reconstruction
// passes. The relative determinant threshold rejects collinear and coplanar
// neighborhoods without introducing a scale-dependent epsilon.
bool fluid_inverse_interface_moment(
    float xx, float xy, float xz, float yy, float yz, float zz,
    out float inverseXX, out float inverseXY, out float inverseXZ,
    out float inverseYY, out float inverseYZ, out float inverseZZ)
{
    float cofactorXX = yy * zz - yz * yz;
    float cofactorXY = xz * yz - xy * zz;
    float cofactorXZ = xy * yz - xz * yy;
    float cofactorYY = xx * zz - xz * xz;
    float cofactorYZ = xy * xz - xx * yz;
    float cofactorZZ = xx * yy - xy * xy;
    float determinant = xx * cofactorXX + xy * cofactorXY +
        xz * cofactorXZ;
    float trace = xx + yy + zz;
    float minimumDeterminant = max(1.0e-30f,
        1.0e-6f * trace * trace * trace);
    if (!isfinite(determinant) || determinant <= minimumDeterminant)
    {
        inverseXX = 0.0f;
        inverseXY = 0.0f;
        inverseXZ = 0.0f;
        inverseYY = 0.0f;
        inverseYZ = 0.0f;
        inverseZZ = 0.0f;
        return false;
    }
    float inverseDeterminant = 1.0f / determinant;
    inverseXX = cofactorXX * inverseDeterminant;
    inverseXY = cofactorXY * inverseDeterminant;
    inverseXZ = cofactorXZ * inverseDeterminant;
    inverseYY = cofactorYY * inverseDeterminant;
    inverseYZ = cofactorYZ * inverseDeterminant;
    inverseZZ = cofactorZZ * inverseDeterminant;
    return isfinite(inverseXX) && isfinite(inverseXY) &&
        isfinite(inverseXZ) && isfinite(inverseYY) &&
        isfinite(inverseYZ) && isfinite(inverseZZ);
}

Vec3 fluid_apply_interface_inverse(
    float inverseXX, float inverseXY, float inverseXZ,
    float inverseYY, float inverseYZ, float inverseZZ, Vec3 value)
{
    return {
        inverseXX * value.x + inverseXY * value.y + inverseXZ * value.z,
        inverseXY * value.x + inverseYY * value.y + inverseYZ * value.z,
        inverseXZ * value.x + inverseYZ * value.y + inverseZZ * value.z};
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_reconstruct_immiscible_interface_normals(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform PMBuffer<uint> first_total_neighbors,
    uniform PMBuffer<uint> first_cross_neighbors,
    uniform PMBuffer<float> first_phase_labels,
    uniform PMBuffer<Vec3> first_interface_normals,
    uniform PMBuffer<uint> first_interface_flags,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform PMBuffer<uint> second_total_neighbors,
    uniform PMBuffer<uint> second_cross_neighbors,
    uniform PMBuffer<float> second_phase_labels,
    uniform PMBuffer<Vec3> second_interface_normals,
    uniform PMBuffer<uint> second_interface_flags,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> identities,
    uniform uint total_capacity, uniform FluidOptions first_options,
    uniform FluidOptions second_options,
    uniform PMBuffer<uint> first_interface_sample_count,
    uniform PMBuffer<uint> second_interface_sample_count,
    uniform PMBuffer<uint> first_under_resolved_interface_count,
    uniform PMBuffer<uint> second_under_resolved_interface_count,
    uniform PMBuffer<uint> first_degenerate_interface_count,
    uniform PMBuffer<uint> second_degenerate_interface_count
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint item = pmThreadIndex();
    if (item >= total_capacity || item >= first_capacity + second_capacity)
        return;
    bool second = item >= first_capacity;
    uint local = second ? item - first_capacity : item;
    uint count = second ? *second_count : *first_count;
    Vec3 zero = {};
    if (second)
    {
        second_phase_labels[local] = local < count ? 1.0f : 0.0f;
        second_interface_normals[local] = zero;
        second_interface_flags[local] = 0u;
    }
    else
    {
        first_phase_labels[local] = local < count ? -1.0f : 0.0f;
        first_interface_normals[local] = zero;
        first_interface_flags[local] = 0u;
    }
    if (local >= count) return;

    uint totalNeighbors = second
        ? second_total_neighbors[local] : first_total_neighbors[local];
    uint crossNeighbors = second
        ? second_cross_neighbors[local] : first_cross_neighbors[local];
    if (crossNeighbors == 0u) return;
    if (second) pmAtomicAdd(second_interface_sample_count, 1u);
    else pmAtomicAdd(first_interface_sample_count, 1u);

    uint flags = kFluidInterfaceSample;
    if (totalNeighbors < 6u || crossNeighbors < 3u)
    {
        flags |= kFluidInterfaceUnderResolved;
        if (second) pmAtomicAdd(second_under_resolved_interface_count, 1u);
        else pmAtomicAdd(first_under_resolved_interface_count, 1u);
    }

    Vec3 position = second ? second_positions[local] : first_positions[local];
    float phase = second ? 1.0f : -1.0f;
    float pairSupport = max(
        first_options.support_radius, second_options.support_radius);
    float inverseSupport = 1.0f / pairSupport;
    float supportSquared = pairSupport * pairSupport;
    int cx = int(floor(position.x * inverseSupport));
    int cy = int(floor(position.y * inverseSupport));
    int cz = int(floor(position.z * inverseSupport));
    float xx = 0.0f;
    float xy = 0.0f;
    float xz = 0.0f;
    float yy = 0.0f;
    float yz = 0.0f;
    float zz = 0.0f;
    Vec3 phaseMoment = {};
    uint selfIdentity = local | (second ? kFluidSecondPopulation : 0u);
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
                    Vec3 delta = subtract(otherPosition, position);
                    float squared = length_squared(delta);
                    if (squared >= supportSquared) continue;
                    float distance = sqrt(max(squared, 1.0e-30f));
                    float q = 1.0f - distance * inverseSupport;
                    float weight = q * q;
                    xx += weight * delta.x * delta.x;
                    xy += weight * delta.x * delta.y;
                    xz += weight * delta.x * delta.z;
                    yy += weight * delta.y * delta.y;
                    yz += weight * delta.y * delta.z;
                    zz += weight * delta.z * delta.z;
                    float otherPhase = otherSecond ? 1.0f : -1.0f;
                    phaseMoment = add(phaseMoment,
                        multiply(delta, weight * (otherPhase - phase)));
                }
            }
        }
    }

    float inverseXX;
    float inverseXY;
    float inverseXZ;
    float inverseYY;
    float inverseYZ;
    float inverseZZ;
    bool invertible = fluid_inverse_interface_moment(
        xx, xy, xz, yy, yz, zz, inverseXX, inverseXY, inverseXZ,
        inverseYY, inverseYZ, inverseZZ);
    Vec3 gradient = invertible ? fluid_apply_interface_inverse(
        inverseXX, inverseXY, inverseXZ, inverseYY, inverseYZ, inverseZZ,
        phaseMoment) : zero;
    float gradientLength = vector_length(gradient);
    bool finiteGradient = isfinite(gradient.x) && isfinite(gradient.y) &&
        isfinite(gradient.z) && isfinite(gradientLength);
    if (!invertible || !finiteGradient ||
        gradientLength <= 1.0e-6f * inverseSupport)
    {
        flags |= kFluidInterfaceDegenerate;
        if (second) pmAtomicAdd(second_degenerate_interface_count, 1u);
        else pmAtomicAdd(first_degenerate_interface_count, 1u);
        gradient = zero;
    }
    else
        gradient = multiply(gradient, 1.0f / gradientLength);

    if (second)
    {
        second_interface_normals[local] = gradient;
        second_interface_flags[local] = flags;
    }
    else
    {
        first_interface_normals[local] = gradient;
        first_interface_flags[local] = flags;
    }
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_reconstruct_immiscible_interface_curvature(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform PMBuffer<Vec3> first_interface_normals,
    uniform PMBuffer<float> first_interface_curvatures,
    uniform PMBuffer<uint> first_interface_flags,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform PMBuffer<Vec3> second_interface_normals,
    uniform PMBuffer<float> second_interface_curvatures,
    uniform PMBuffer<uint> second_interface_flags,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> identities,
    uniform uint total_capacity, uniform FluidOptions first_options,
    uniform FluidOptions second_options,
    uniform PMBuffer<uint> first_under_resolved_interface_count,
    uniform PMBuffer<uint> second_under_resolved_interface_count,
    uniform PMBuffer<uint> first_degenerate_interface_count,
    uniform PMBuffer<uint> second_degenerate_interface_count,
    uniform PMBuffer<uint> first_curvature_clamp_count,
    uniform PMBuffer<uint> second_curvature_clamp_count
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint item = pmThreadIndex();
    if (item >= total_capacity || item >= first_capacity + second_capacity)
        return;
    bool second = item >= first_capacity;
    uint local = second ? item - first_capacity : item;
    uint count = second ? *second_count : *first_count;
    if (second) second_interface_curvatures[local] = 0.0f;
    else first_interface_curvatures[local] = 0.0f;
    if (local >= count) return;

    uint flags = second
        ? second_interface_flags[local] : first_interface_flags[local];
    if ((flags & kFluidInterfaceSample) == 0u ||
        (flags & kFluidInterfaceDegenerate) != 0u)
        return;
    Vec3 position = second ? second_positions[local] : first_positions[local];
    Vec3 normal = second
        ? second_interface_normals[local] : first_interface_normals[local];
    float pairSupport = max(
        first_options.support_radius, second_options.support_radius);
    float inverseSupport = 1.0f / pairSupport;
    float supportSquared = pairSupport * pairSupport;
    int cx = int(floor(position.x * inverseSupport));
    int cy = int(floor(position.y * inverseSupport));
    int cz = int(floor(position.z * inverseSupport));
    float xx = 0.0f;
    float xy = 0.0f;
    float xz = 0.0f;
    float yy = 0.0f;
    float yz = 0.0f;
    float zz = 0.0f;
    float bXX = 0.0f;
    float bXY = 0.0f;
    float bXZ = 0.0f;
    float bYX = 0.0f;
    float bYY = 0.0f;
    float bYZ = 0.0f;
    float bZX = 0.0f;
    float bZY = 0.0f;
    float bZZ = 0.0f;
    uint normalNeighbors = 0u;
    uint selfIdentity = local | (second ? kFluidSecondPopulation : 0u);
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
                    uint otherFlags = otherSecond
                        ? second_interface_flags[other]
                        : first_interface_flags[other];
                    if ((otherFlags & kFluidInterfaceSample) == 0u ||
                        (otherFlags & kFluidInterfaceDegenerate) != 0u)
                        continue;
                    Vec3 otherPosition = otherSecond
                        ? second_positions[other] : first_positions[other];
                    Vec3 delta = subtract(otherPosition, position);
                    float squared = length_squared(delta);
                    if (squared >= supportSquared) continue;
                    Vec3 otherNormal = otherSecond
                        ? second_interface_normals[other]
                        : first_interface_normals[other];
                    Vec3 deltaNormal = subtract(otherNormal, normal);
                    float distance = sqrt(max(squared, 1.0e-30f));
                    float q = 1.0f - distance * inverseSupport;
                    float weight = q * q;
                    xx += weight * delta.x * delta.x;
                    xy += weight * delta.x * delta.y;
                    xz += weight * delta.x * delta.z;
                    yy += weight * delta.y * delta.y;
                    yz += weight * delta.y * delta.z;
                    zz += weight * delta.z * delta.z;
                    bXX += weight * deltaNormal.x * delta.x;
                    bXY += weight * deltaNormal.x * delta.y;
                    bXZ += weight * deltaNormal.x * delta.z;
                    bYX += weight * deltaNormal.y * delta.x;
                    bYY += weight * deltaNormal.y * delta.y;
                    bYZ += weight * deltaNormal.y * delta.z;
                    bZX += weight * deltaNormal.z * delta.x;
                    bZY += weight * deltaNormal.z * delta.y;
                    bZZ += weight * deltaNormal.z * delta.z;
                    ++normalNeighbors;
                }
            }
        }
    }

    if (normalNeighbors < 6u &&
        (flags & kFluidInterfaceUnderResolved) == 0u)
    {
        flags |= kFluidInterfaceUnderResolved;
        if (second) pmAtomicAdd(second_under_resolved_interface_count, 1u);
        else pmAtomicAdd(first_under_resolved_interface_count, 1u);
    }
    float inverseXX;
    float inverseXY;
    float inverseXZ;
    float inverseYY;
    float inverseYZ;
    float inverseZZ;
    bool invertible = fluid_inverse_interface_moment(
        xx, xy, xz, yy, yz, zz, inverseXX, inverseXY, inverseXZ,
        inverseYY, inverseYZ, inverseZZ);
    float curvature = invertible
        ? bXX * inverseXX + bXY * inverseXY + bXZ * inverseXZ +
          bYX * inverseXY + bYY * inverseYY + bYZ * inverseYZ +
          bZX * inverseXZ + bZY * inverseYZ + bZZ * inverseZZ
        : 0.0f;
    if (!invertible || !isfinite(curvature))
    {
        if ((flags & kFluidInterfaceDegenerate) == 0u)
        {
            flags |= kFluidInterfaceDegenerate;
            if (second) pmAtomicAdd(second_degenerate_interface_count, 1u);
            else pmAtomicAdd(first_degenerate_interface_count, 1u);
        }
        curvature = 0.0f;
    }
    else
    {
        float minimumParticleRadius = max(1.0e-6f, min(
            first_options.particle_radius, second_options.particle_radius));
        float curvatureLimit = 2.0f / minimumParticleRadius;
        if (abs(curvature) > curvatureLimit)
        {
            flags |= kFluidInterfaceCurvatureClamped;
            curvature = clamp_scalar(
                curvature, -curvatureLimit, curvatureLimit);
            if (second) pmAtomicAdd(second_curvature_clamp_count, 1u);
            else pmAtomicAdd(first_curvature_clamp_count, 1u);
        }
    }
    if (second)
    {
        second_interface_curvatures[local] = curvature;
        second_interface_flags[local] = flags;
    }
    else
    {
        first_interface_curvatures[local] = curvature;
        first_interface_flags[local] = flags;
    }
}

public float fluid_immiscible_pressure_jump(
    float surface_tension, float first_curvature, float second_curvature)
{
    return surface_tension *
        0.5f * (first_curvature + second_curvature);
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void fluid_apply_immiscible_pair_pressure_capillarity(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<Vec3> first_velocities,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform PMBuffer<Vec3> first_forces,
    uniform PMBuffer<float> first_density_samples,
    uniform PMBuffer<Vec3> first_interface_normals,
    uniform PMBuffer<float> first_interface_curvatures,
    uniform PMBuffer<uint> first_interface_flags,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<Vec3> second_velocities,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform PMBuffer<Vec3> second_forces,
    uniform PMBuffer<float> second_density_samples,
    uniform PMBuffer<Vec3> second_interface_normals,
    uniform PMBuffer<float> second_interface_curvatures,
    uniform PMBuffer<uint> second_interface_flags,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> identities,
    uniform uint total_capacity,
    uniform FluidOptions first_options,
    uniform FluidOptions second_options,
    uniform float surface_tension
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
    float firstVolume = firstMass / first_options.rest_density;
    float secondVolume = secondMass / second_options.rest_density;
    float contactSeparation =
        first_options.particle_radius + second_options.particle_radius;
    float firstContactScale = first_options.repulsion *
        (1000.0f / first_options.rest_density);
    float secondContactScale = second_options.repulsion *
        (1000.0f / second_options.rest_density);
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
                    float gradientMagnitude =
                        fluid_pressure_gradient_magnitude(
                            distance, pairSupport);
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
                        gradientMagnitude;

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

                    float capillaryMagnitude = 0.0f;
                    uint firstFlags = first_interface_flags[firstIndex];
                    uint secondFlags = second_interface_flags[secondIndex];
                    bool validInterface = squared > 1.0e-12f &&
                        (firstFlags & kFluidInterfaceSample) != 0u &&
                        (secondFlags & kFluidInterfaceSample) != 0u &&
                        (firstFlags & kFluidInterfaceDegenerate) == 0u &&
                        (secondFlags & kFluidInterfaceDegenerate) == 0u;
                    if (surface_tension > 0.0f && validInterface)
                    {
                        Vec3 averageNormal = normalized_or(add(
                            first_interface_normals[firstIndex],
                            second_interface_normals[secondIndex]),
                            multiply(direction, -1.0f));
                        Vec3 firstToSecond = multiply(direction, -1.0f);
                        float alignment = max(
                            0.0f, dot(averageNormal, firstToSecond));
                        float pressureJump = fluid_immiscible_pressure_jump(
                            surface_tension,
                            first_interface_curvatures[firstIndex],
                            second_interface_curvatures[secondIndex]);
                        if (isfinite(pressureJump))
                            capillaryMagnitude = abs(pressureJump) *
                                firstVolume * secondVolume *
                                gradientMagnitude * alignment;
                    }

                    Vec3 commonForce = multiply(direction,
                        pressureMagnitude + contactMagnitude -
                            capillaryMagnitude);
                    if (forceLimit > 0.0f)
                        commonForce = clamp_length(commonForce, forceLimit);
                    acceleration = add(acceleration,
                        multiply(commonForce,
                            second ? -1.0f / secondMass :
                                1.0f / firstMass));
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
void fluid_apply_immiscible_pair_viscosity(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<Vec3> first_velocities,
    uniform PMBuffer<uint> first_count, uniform uint first_capacity,
    uniform PMBuffer<Vec3> first_forces,
    uniform PMBuffer<Vec3> first_interface_normals,
    uniform PMBuffer<uint> first_interface_flags,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<Vec3> second_velocities,
    uniform PMBuffer<uint> second_count, uniform uint second_capacity,
    uniform PMBuffer<Vec3> second_forces,
    uniform PMBuffer<Vec3> second_interface_normals,
    uniform PMBuffer<uint> second_interface_flags,
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

                    uint firstFlags = first_interface_flags[firstIndex];
                    uint secondFlags = second_interface_flags[secondIndex];
                    bool validInterface =
                        (firstFlags & kFluidInterfaceSample) != 0u &&
                        (secondFlags & kFluidInterfaceSample) != 0u &&
                        (firstFlags & kFluidInterfaceDegenerate) == 0u &&
                        (secondFlags & kFluidInterfaceDegenerate) == 0u;
                    Vec3 normalSum = add(
                        first_interface_normals[firstIndex],
                        second_interface_normals[secondIndex]);
                    float normalSquared = length_squared(normalSum);
                    if (!validInterface || normalSquared <= 1.0e-12f)
                        continue;

                    Vec3 interfaceNormal = multiply(
                        normalSum, 1.0f / sqrt(normalSquared));
                    Vec3 relativeVelocity = subtract(
                        second_velocities[secondIndex],
                        first_velocities[firstIndex]);
                    Vec3 tangentialVelocity = subtract(relativeVelocity,
                        multiply(interfaceNormal,
                            dot(relativeVelocity, interfaceNormal)));
                    float distance = sqrt(max(squared, 0.0f));
                    float weight = 1.0f - distance * inverseSupport;
                    Vec3 commonForce = multiply(tangentialVelocity,
                        pairDynamicViscosity * pairVolume * weight);
                    if (forceLimit > 0.0f)
                        commonForce = clamp_length(commonForce, forceLimit);
                    acceleration = add(acceleration,
                        multiply(commonForce,
                            second ? -1.0f / secondMass :
                                1.0f / firstMass));
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
void fluid_gather_spatial_payloads(
    uniform PMBuffer<uint> compacted_sources,
    uniform PMBuffer<uint> active_slot_count,
    uniform PMBuffer<FluidSpatialLeafMetadata> metadata,
    uniform uint leaf_capacity, uniform uint candidate_slot_capacity,
    uniform PMBuffer<uint> active_payloads,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint rank = pmThreadIndex();
    uint count = active_slot_count[0];
    if (count > candidate_slot_capacity)
    {
        if (rank == 0u) pmAtomicOr(status, kFluidSpatialCountOverflow);
        return;
    }
    if (rank >= count) return;
    uint source = compacted_sources[rank];
    if (source >= leaf_capacity)
    {
        pmAtomicOr(status, kFluidSpatialAdmissionInvalid);
        return;
    }
    active_payloads[rank] = metadata[source].payload_ref;
}
