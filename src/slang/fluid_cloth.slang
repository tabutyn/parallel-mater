#include "dispatch_entry.slang"
#include "coupling_spatial.slangh"
// SPDX-License-Identifier: MIT
// Shared fluid and cloth coupling kernels.

import types;
import math;
import device_intrinsics;
import geometry_fluid;
import graphics_resources;

[shader("compute")]
[numthreads(128, 1, 1)]
void fluid_cloth_containment_forces(
    uniform PMBuffer<Vec3> particle_positions, uniform PMBuffer<Vec3> particle_velocities,
    uniform PMBuffer<uint> particle_count, uniform PMBuffer<Vec3> particle_accelerations,
    uniform PMBuffer<float> foam_source, uniform float particle_mass,
    uniform PMBuffer<Vec3> cloth_positions, uniform PMBuffer<Vec3> cloth_velocities,
    uniform PMBuffer<uint> cloth_indices, uniform uint triangle_count,
    uniform PMBuffer<uint> triangle_generations,
    uniform uint cloth_owner_slot, uniform uint cloth_owner_generation,
    uniform uint cloth_topology_revision,
    uniform PMBuffer<CouplingSpatialLeafMetadata> spatial_metadata,
    uniform PMByteBuffer spatial_admission, uniform uint spatial_leaf_capacity,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_sources,
    uniform uint candidates_per_query, uniform PMBuffer<uint> spatial_status,
    uniform float cloth_orientation,
    uniform FluidClothCouplingOptions options, uniform PMBuffer<Vec3> cloth_forces PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= *particle_count) return;
    Vec3 point = particle_positions[particle];
    float bestSquared = 3.402823466e38f;
    Vec3 bestPoint = {};
    Vec3 bestNormal = {};
    Vec3 bestWeights = {};
    uint bestTriangle = 0u;
    uint candidateCount = candidate_counts[particle];
    if (candidateCount > candidates_per_query)
    {
        pmAtomicOr(spatial_status, kCouplingSpatialQueryOverflow);
        return;
    }
    uint candidateBase = particle * candidates_per_query;
    for (uint candidateIndex = 0u;
         candidateIndex < candidateCount; ++candidateIndex)
    {
        uint source = candidate_sources[candidateBase + candidateIndex];
        if (source >= spatial_leaf_capacity || spatial_admission[source] == 0u)
        {
            pmAtomicOr(spatial_status, kCouplingSpatialAdmissionInvalid);
            return;
        }
        CouplingSpatialLeafMetadata leaf = spatial_metadata[source];
        if (leaf.identity.family != kCouplingSpatialClothFamily ||
            leaf.identity.owner_slot != cloth_owner_slot ||
            leaf.identity.owner_generation != cloth_owner_generation ||
            leaf.identity.topology_revision != cloth_topology_revision ||
            leaf.identity.primitive_kind != kCouplingSpatialTrianglePrimitive)
            continue;
        uint triangle = leaf.payload_ref;
        if (triangle >= triangle_count ||
            leaf.identity.primitive_slot != triangle ||
            leaf.identity.primitive_generation !=
                triangle_generations[triangle])
        {
            pmAtomicOr(spatial_status, kCouplingSpatialStaleEpoch);
            return;
        }
        uint aIndex = cloth_indices[3u * triangle];
        uint bIndex = cloth_indices[3u * triangle + 1u];
        uint cIndex = cloth_indices[3u * triangle + 2u];
        Vec3 a = cloth_positions[aIndex];
        Vec3 b = cloth_positions[bIndex];
        Vec3 c = cloth_positions[cIndex];
        Vec3 weights;
        Vec3 nearest = fluid_closest_triangle_barycentric(
            point, a, b, c, weights);
        float squared = length_squared(subtract(point, nearest));
        if (squared >= bestSquared) continue;
        Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) <= 1.0e-14f) continue;
        bestSquared = squared;
        bestPoint = nearest;
        Vec3 up = {0.0f, 1.0f, 0.0f};
        bestNormal = multiply(normalized_or(face, up), cloth_orientation);
        bestWeights = weights;
        bestTriangle = triangle;
    }
    if (bestSquared == 3.402823466e38f) return;
    float signedDistance = dot(subtract(point, bestPoint), bestNormal);
    float violation = options.contact_distance + signedDistance;
    bool outside = signedDistance > 0.0f;
    if (violation <= 0.0f || (!outside && bestSquared >
        options.interaction_radius * options.interaction_radius)) return;
    uint a = cloth_indices[3u * bestTriangle];
    uint b = cloth_indices[3u * bestTriangle + 1u];
    uint c = cloth_indices[3u * bestTriangle + 2u];
    Vec3 surfaceVelocity = add(
        multiply(cloth_velocities[a], bestWeights.x),
        add(multiply(cloth_velocities[b], bestWeights.y),
            multiply(cloth_velocities[c], bestWeights.z)));
    Vec3 relativeVelocity = subtract(
        particle_velocities[particle], surfaceVelocity);
    float normalSpeed = dot(relativeVelocity, bestNormal);
    float magnitude = min(options.maximum_force,
        max(0.0f, options.stiffness * violation +
                  options.damping * normalSpeed));
    Vec3 tangent = subtract(relativeVelocity,
        multiply(bestNormal, normalSpeed));
    Vec3 force = clamp_length(add(
        multiply(bestNormal, -magnitude),
        multiply(tangent, -options.tangential_drag)), options.maximum_force);
    particle_accelerations[particle] = add(
        particle_accelerations[particle],
        multiply(force, 1.0f / particle_mass));
    foam_source[particle] = max(foam_source[particle],
        clamp_scalar(magnitude / max(options.maximum_force, 1.0f),
                     0.0f, 1.0f));
    Vec3 reaction = multiply(force, -1.0f);
    atomic_add(cloth_forces + a, multiply(reaction, bestWeights.x));
    atomic_add(cloth_forces + b, multiply(reaction, bestWeights.y));
    atomic_add(cloth_forces + c, multiply(reaction, bestWeights.z));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_apply_fluid_forces(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> inverse_masses, uniform PMBuffer<Vec3> forces,
    uniform uint count, uniform float dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (vertex >= count || inverse_masses[vertex] == 0.0f) return;
    Vec3 velocityChange = clamp_length(
        multiply(forces[vertex], inverse_masses[vertex] * dt), 2.0f);
    velocities[vertex] = clamp_length(
        add(velocities[vertex], velocityChange), 20.0f);
    positions[vertex] = add(
        positions[vertex], multiply(velocityChange, dt));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fluid_project_inside_cloth(
    uniform PMBuffer<Vec3> particle_positions, uniform PMBuffer<Vec3> particle_velocities,
    uniform PMBuffer<uint> particle_count, uniform float contact_distance,
    uniform PMBuffer<Vec3> cloth_positions, uniform PMBuffer<Vec3> cloth_velocities,
    uniform PMBuffer<uint> cloth_indices, uniform uint triangle_count,
    uniform PMBuffer<uint> triangle_generations,
    uniform uint cloth_owner_slot, uniform uint cloth_owner_generation,
    uniform uint cloth_topology_revision,
    uniform PMBuffer<CouplingSpatialLeafMetadata> spatial_metadata,
    uniform PMByteBuffer spatial_admission, uniform uint spatial_leaf_capacity,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_sources,
    uniform uint candidates_per_query, uniform PMBuffer<uint> spatial_status,
    uniform float cloth_orientation PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= *particle_count) return;
    Vec3 point = particle_positions[particle];
    float bestSquared = 3.402823466e38f;
    Vec3 bestPoint = {};
    Vec3 bestNormal = {};
    Vec3 bestWeights = {};
    uint bestTriangle = 0u;
    uint candidateCount = candidate_counts[particle];
    if (candidateCount > candidates_per_query)
    {
        pmAtomicOr(spatial_status, kCouplingSpatialQueryOverflow);
        return;
    }
    uint candidateBase = particle * candidates_per_query;
    for (uint candidateIndex = 0u;
         candidateIndex < candidateCount; ++candidateIndex)
    {
        uint source = candidate_sources[candidateBase + candidateIndex];
        if (source >= spatial_leaf_capacity || spatial_admission[source] == 0u)
        {
            pmAtomicOr(spatial_status, kCouplingSpatialAdmissionInvalid);
            return;
        }
        CouplingSpatialLeafMetadata leaf = spatial_metadata[source];
        if (leaf.identity.family != kCouplingSpatialClothFamily ||
            leaf.identity.owner_slot != cloth_owner_slot ||
            leaf.identity.owner_generation != cloth_owner_generation ||
            leaf.identity.topology_revision != cloth_topology_revision ||
            leaf.identity.primitive_kind != kCouplingSpatialTrianglePrimitive)
            continue;
        uint triangle = leaf.payload_ref;
        if (triangle >= triangle_count ||
            leaf.identity.primitive_slot != triangle ||
            leaf.identity.primitive_generation !=
                triangle_generations[triangle])
        {
            pmAtomicOr(spatial_status, kCouplingSpatialStaleEpoch);
            return;
        }
        uint aIndex = cloth_indices[3u * triangle];
        uint bIndex = cloth_indices[3u * triangle + 1u];
        uint cIndex = cloth_indices[3u * triangle + 2u];
        Vec3 a = cloth_positions[aIndex];
        Vec3 b = cloth_positions[bIndex];
        Vec3 c = cloth_positions[cIndex];
        Vec3 weights;
        Vec3 nearest = fluid_closest_triangle_barycentric(
            point, a, b, c, weights);
        float squared = length_squared(subtract(point, nearest));
        if (squared >= bestSquared) continue;
        Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) <= 1.0e-14f) continue;
        bestSquared = squared;
        bestPoint = nearest;
        Vec3 up = {0.0f, 1.0f, 0.0f};
        bestNormal = multiply(normalized_or(face, up), cloth_orientation);
        bestWeights = weights;
        bestTriangle = triangle;
    }
    if (bestSquared == 3.402823466e38f) return;
    float signedDistance = dot(subtract(point, bestPoint), bestNormal);
    float violation = contact_distance + signedDistance;
    if (violation <= 0.0f) return;
    uint a = cloth_indices[3u * bestTriangle];
    uint b = cloth_indices[3u * bestTriangle + 1u];
    uint c = cloth_indices[3u * bestTriangle + 2u];
    particle_positions[particle] = subtract(
        particle_positions[particle], multiply(bestNormal, violation));
    Vec3 surfaceVelocity = add(
        multiply(cloth_velocities[a], bestWeights.x),
        add(multiply(cloth_velocities[b], bestWeights.y),
            multiply(cloth_velocities[c], bestWeights.z)));
    Vec3 relative = subtract(
        particle_velocities[particle], surfaceVelocity);
    float outwardSpeed = dot(relative, bestNormal);
    if (outwardSpeed > 0.0f)
        relative = subtract(relative,
            multiply(bestNormal, outwardSpeed));
    particle_velocities[particle] = add(surfaceVelocity, relative);
}
