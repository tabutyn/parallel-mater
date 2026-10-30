#include "dispatch_entry.slang"
#include "dispatch_shape.slangh"
// SPDX-License-Identifier: MIT
// Shared soft-body/cloth coupling kernels.

import types;
import math;
import device_intrinsics;
import geometry_fluid;
import graphics_resources;

// ABI mirror of the shared spatial leaf published by geometry_soft_body.
struct SoftClothSpatialIdentity
{
    uint family;
    uint owner_slot;
    uint owner_generation;
    uint topology_revision;
    uint primitive_kind;
    uint primitive_slot;
    uint primitive_generation;
};

struct SoftClothSpatialLeafMetadata
{
    SoftClothSpatialIdentity identity;
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

static const uint kSoftClothSpatialClothFamily = 3u;
static const uint kSoftClothSpatialSoftFamily = 5u;
static const uint kSoftClothSpatialTrianglePrimitive = 2u;
static const uint kSoftClothSpatialPointPrimitive = 3u;
static const uint kSoftClothSpatialStaleEpoch = 1u << 9u;
static const uint kSoftClothSpatialQueryOverflow = 1u << 10u;
static const uint kSoftClothSpatialAdmissionInvalid = 1u << 12u;

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void soft_cloth_detect(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> previous,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> inverse_masses,
    uniform uint count, uniform PMBuffer<Vec3> cloth_positions,
    uniform PMBuffer<Vec3> cloth_previous, uniform PMBuffer<Vec3> cloth_velocities,
    uniform PMBuffer<float> cloth_inverse_masses, uniform PMBuffer<uint> indices,
    uniform PMBuffer<Vec3> surface, uniform uint triangle_count,
    uniform PMBuffer<uint> triangle_generations,
    uniform uint cloth_owner_slot, uniform uint cloth_owner_generation,
    uniform uint cloth_topology_revision,
    uniform PMBuffer<SoftClothSpatialLeafMetadata> spatial_metadata,
    uniform PMByteBuffer spatial_admission, uniform uint leaf_capacity,
    uniform PMBuffer<uint> query_leaf_sources,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_sources,
    uniform uint candidates_per_query,
    uniform uint soft_owner_slot, uniform uint soft_owner_generation,
    uniform uint soft_topology_revision, uniform PMBuffer<uint> status,
    uniform float distance, uniform float friction,
    uniform PMBuffer<SoftClothContact> contacts, uniform PMBuffer<uint> contact_counts PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (node >= count) return;
    contacts[node] = {};
    if (inverse_masses[node] == 0.0f) return;
    uint querySource = query_leaf_sources[node];
    if (querySource >= leaf_capacity || spatial_admission[querySource] == 0u)
    {
        pmAtomicOr(status, kSoftClothSpatialAdmissionInvalid);
        return;
    }
    SoftClothSpatialLeafMetadata queryLeaf = spatial_metadata[querySource];
    if (queryLeaf.identity.family != kSoftClothSpatialSoftFamily ||
        queryLeaf.identity.owner_slot != soft_owner_slot ||
        queryLeaf.identity.owner_generation != soft_owner_generation ||
        queryLeaf.identity.topology_revision != soft_topology_revision ||
        queryLeaf.identity.primitive_kind != kSoftClothSpatialPointPrimitive ||
        queryLeaf.identity.primitive_slot != node ||
        queryLeaf.identity.primitive_generation != soft_owner_generation ||
        queryLeaf.payload_ref != node)
    {
        pmAtomicOr(status, kSoftClothSpatialStaleEpoch);
        return;
    }
    uint candidateCount = candidate_counts[node];
    if (candidateCount > candidates_per_query)
    {
        pmAtomicOr(status, kSoftClothSpatialQueryOverflow);
        return;
    }
    Vec3 point = positions[node];
    Vec3 start = previous[node];
    float bestSquared = 3.402823466e38f;
    Vec3 bestNormal = {};
    Vec3 bestWeights = {};
    Vec3 bestPoint = {};
    uint best = kInvalidDense;
    uint candidateBase = node * candidates_per_query;
    for (uint candidate = 0u; candidate < candidateCount; ++candidate)
    {
        uint source = candidate_sources[candidateBase + candidate];
        if (source >= leaf_capacity || spatial_admission[source] == 0u)
        {
            pmAtomicOr(status, kSoftClothSpatialAdmissionInvalid);
            return;
        }
        SoftClothSpatialLeafMetadata leaf = spatial_metadata[source];
        if (leaf.identity.family != kSoftClothSpatialClothFamily ||
            leaf.identity.owner_slot != cloth_owner_slot ||
            leaf.identity.owner_generation != cloth_owner_generation ||
            leaf.identity.topology_revision != cloth_topology_revision ||
            leaf.identity.primitive_kind !=
                kSoftClothSpatialTrianglePrimitive)
            continue;
        uint triangle = leaf.payload_ref;
        if (triangle >= triangle_count ||
            leaf.identity.primitive_slot != triangle ||
            triangle_generations[triangle] !=
                leaf.identity.primitive_generation)
        {
            pmAtomicOr(status, kSoftClothSpatialStaleEpoch);
            return;
        }
        uint base = triangle * 3u;
        Vec3 a = pmBufferPresent(surface) ? surface[base] :
            cloth_positions[indices[base]];
        Vec3 b = pmBufferPresent(surface) ? surface[base + 1u] :
            cloth_positions[indices[base + 1u]];
        Vec3 c = pmBufferPresent(surface) ? surface[base + 2u] :
            cloth_positions[indices[base + 2u]];
        Vec3 oldA = cloth_previous[indices[base]];
        Vec3 oldB = cloth_previous[indices[base + 1u]];
        Vec3 oldC = cloth_previous[indices[base + 2u]];
        Vec3 expansion = {distance, distance, distance};
        if (!bounds_overlap(component_min(point, start), component_max(point, start),
            subtract(component_min(component_min(a, component_min(b, c)),
                component_min(oldA, component_min(oldB, oldC))), expansion),
            add(component_max(component_max(a, component_max(b, c)),
                component_max(oldA, component_max(oldB, oldC))), expansion)))
            continue;
        Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) < 1.0e-14f) continue;
        Vec3 normal = normalized_or(face, {});
        Vec3 weights = {};
        Vec3 nearest = fluid_closest_triangle_barycentric(
            point, a, b, c, weights);
        Vec3 delta = subtract(point, nearest);
        float squared = length_squared(delta);
        if (squared >= bestSquared) continue;
        Vec3 oldNearest = add(multiply(
            cloth_previous[indices[base]], weights.x), add(multiply(
            cloth_previous[indices[base + 1u]], weights.y), multiply(
            cloth_previous[indices[base + 2u]], weights.z)));
        float before = dot(subtract(start, oldNearest), normal);
        float after = dot(delta, normal);
        float travel = vector_length(subtract(point, start)) +
            vector_length(subtract(nearest, oldNearest));
        bool crossed = before * after < 0.0f &&
            squared <= (travel + distance) * (travel + distance);
        if (!crossed && squared >= distance * distance) continue;
        best = triangle;
        bestSquared = squared;
        bestPoint = nearest;
        bestWeights = weights;
        bestNormal = crossed || squared < 1.0e-14f
            ? multiply(normal, before >= 0.0f ? 1.0f : -1.0f)
            : multiply(delta, rsqrt(squared));
    }
    if (best == kInvalidDense) return;
    float penetration = distance - dot(
        subtract(point, bestPoint), bestNormal);
    if (penetration <= 0.0f) return;
    SoftClothContact contact = {};
    contact.weights[0] = bestWeights.x;
    contact.weights[1] = bestWeights.y;
    contact.weights[2] = bestWeights.z;
    float denominator = inverse_masses[node];
    Vec3 clothVelocity = {};
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        uint vertex = indices[3u * best + corner];
        float weight = contact.weights[corner];
        contact.vertices[corner] = vertex;
        denominator += weight * weight * cloth_inverse_masses[vertex];
        clothVelocity = add(clothVelocity,
            multiply(cloth_velocities[vertex], weight));
        if (weight > 1.0e-6f && cloth_inverse_masses[vertex] > 0.0f)
            pmAtomicAdd(contact_counts + vertex, 1u);
    }
    contact.position_impulse = multiply(bestNormal,
        min(penetration, 2.0f * distance) / denominator);
    contact.soft_inverse_mass_fraction = inverse_masses[node] / denominator;
    Vec3 relative = subtract(velocities[node], clothVelocity);
    float normalSpeed = dot(relative, bestNormal);
    float normalImpulse = max(0.0f, -normalSpeed) / denominator;
    Vec3 tangent = subtract(relative, multiply(bestNormal, normalSpeed));
    contact.velocity_impulse = subtract(multiply(bestNormal, normalImpulse),
        clamp_length(multiply(tangent, 1.0f / denominator),
            friction * normalImpulse));
    contact.active = true;
    contacts[node] = contact;
}

float soft_cloth_relaxation(SoftClothContact contact, PMBuffer<uint> counts)
{
    uint degree = 1u;
    for (uint corner = 0u; corner < 3u; ++corner)
        if (contact.weights[corner] > 1.0e-6f)
            degree = max(degree, counts[contact.vertices[corner]]);
    float softFraction = contact.soft_inverse_mass_fraction;
    return 1.0f /
        (softFraction + float(degree) * (1.0f - softFraction));
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void soft_cloth_apply_soft(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> forces, uniform PMBuffer<float> inverse_masses,
    uniform uint count, uniform PMBuffer<SoftClothContact> contacts,
    uniform PMBuffer<uint> counts, uniform float dt, uniform float maximum_speed PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (node >= count || !contacts[node].active) return;
    SoftClothContact contact = contacts[node];
    float relaxation = soft_cloth_relaxation(contact, counts);
    Vec3 positionDelta = multiply(
        contact.position_impulse, relaxation * inverse_masses[node]);
    Vec3 impulse = multiply(contact.velocity_impulse, relaxation);
    Vec3 velocityDelta = multiply(impulse, inverse_masses[node]);
    positions[node] = add(add(positions[node], positionDelta),
        multiply(velocityDelta, dt));
    previous_positions[node] = add(previous_positions[node], positionDelta);
    velocities[node] = clamp_length(add(
        velocities[node], velocityDelta), maximum_speed);
    forces[node] = add(forces[node], multiply(impulse, 1.0f / dt));
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void soft_cloth_apply_cloth(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> forces, uniform PMBuffer<float> inverse_masses,
    uniform uint count, uniform PMBuffer<SoftClothContact> contacts,
    uniform uint contact_count, uniform PMBuffer<uint> counts, uniform float dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (vertex >= count) return;
    Vec3 positionImpulse = {};
    Vec3 velocityImpulse = {};
    for (uint node = 0u; node < contact_count; ++node)
    {
        SoftClothContact contact = contacts[node];
        if (!contact.active) continue;
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            if (contact.vertices[corner] != vertex) continue;
            float scale = -contact.weights[corner] *
                soft_cloth_relaxation(contact, counts);
            positionImpulse = add(positionImpulse,
                multiply(contact.position_impulse, scale));
            velocityImpulse = add(velocityImpulse,
                multiply(contact.velocity_impulse, scale));
        }
    }
    Vec3 positionDelta = multiply(
        positionImpulse, inverse_masses[vertex]);
    Vec3 velocityDelta = multiply(
        velocityImpulse, inverse_masses[vertex]);
    positions[vertex] = add(add(positions[vertex], positionDelta),
        multiply(velocityDelta, dt));
    previous_positions[vertex] = add(
        previous_positions[vertex], positionDelta);
    velocities[vertex] = clamp_length(add(
        velocities[vertex], velocityDelta), 20.0f);
    forces[vertex] = add(forces[vertex],
        multiply(velocityImpulse, 1.0f / dt));
}
