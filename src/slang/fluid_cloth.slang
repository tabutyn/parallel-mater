// SPDX-License-Identifier: MIT
// Shared fluid and cloth coupling kernels.

#ifdef PM_TARGET_METAL
import metal_common;

struct MetalFluidClothConstants
{
    float timestep;
    uint surface_index_count;
    float contact_distance;
    float interaction_radius;
    float stiffness;
    float damping;
    float tangential_drag;
    float maximum_force;
    float particle_mass;
    float orientation;
    float maximum_particle_speed;
    uint surface_count;
    uint enabled;
    uint cloth_count;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidClothFluidPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidClothFluidVelocities : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<MetalParticleMetadata>
    metalFluidClothMetadata : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<float>
    metalFluidClothFoam : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidClothPositions : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidClothVelocities : register(u5);
[[vk::binding(6, 0)]] StructuredBuffer<float>
    metalFluidClothInverseMasses : register(t6);
[[vk::binding(7, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidClothSurfacePositions : register(u7);
[[vk::binding(8, 0)]] StructuredBuffer<uint>
    metalFluidClothSurfaceIndices : register(t8);
[[vk::binding(9, 0)]] StructuredBuffer<uint>
    metalFluidClothSurfaceSources : register(t9);
[[vk::binding(10, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidClothForces : register(u10);
[[vk::binding(11, 0)]] ConstantBuffer<MetalFluidClothConstants>
    metalFluidClothConstants : register(b11);
[[vk::binding(12, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidClothAccelerations : register(u12);
[[vk::binding(13, 0)]] RWStructuredBuffer<uint>
    metalFluidClothContributionNodes : register(u13);
[[vk::binding(14, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidClothContributionForces : register(u14);

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_fluid_cloth_forces(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalFluidClothConstants constants = metalFluidClothConstants;
    if (particle >= metalFluidClothMetadata[0].count ||
        constants.enabled == 0u) return;
    uint contribution = particle * 3u;
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        metalFluidClothContributionNodes[contribution + corner] = 0xffffffffu;
        metalFluidClothContributionForces[contribution + corner] =
            metalStore(float3(0.0f));
    }
    float3 point = metalLoad(metalFluidClothFluidPositions[particle]);
    float nearestSquared = 3.402823466e+38f;
    float3 nearest = float3(0.0f);
    float3 faceNormal = float3(0.0f);
    float3 weights = float3(0.0f);
    uint best = constants.surface_index_count;
    for (uint base = 0u; base + 2u < constants.surface_index_count;
         base += 3u)
    {
        float3 a = metalLoad(metalFluidClothSurfacePositions[
            metalFluidClothSurfaceIndices[base]]);
        float3 b = metalLoad(metalFluidClothSurfacePositions[
            metalFluidClothSurfaceIndices[base + 1u]]);
        float3 c = metalLoad(metalFluidClothSurfacePositions[
            metalFluidClothSurfaceIndices[base + 2u]]);
        float3 face = cross(b - a, c - a);
        float faceSquared = dot(face, face);
        if (faceSquared <= 1.0e-14f) continue;
        float3 candidate = metalClosestPointTriangle(point, a, b, c);
        float squared = dot(point - candidate, point - candidate);
        if (squared >= nearestSquared) continue;
        nearestSquared = squared;
        nearest = candidate;
        faceNormal = face * rsqrt(faceSquared) * constants.orientation;
        weights = metalTriangleWeights(candidate, a, b, c);
        best = base;
    }
    if (best == constants.surface_index_count) return;
    float signedDistance = dot(point - nearest, faceNormal);
    float violation = constants.contact_distance + signedDistance;
    bool outside = signedDistance > 0.0f;
    if (violation <= 0.0f ||
        (!outside && nearestSquared >
            constants.interaction_radius * constants.interaction_radius))
        return;
    uint surfaceVertices[3] =
    {
        metalFluidClothSurfaceIndices[best],
        metalFluidClothSurfaceIndices[best + 1u],
        metalFluidClothSurfaceIndices[best + 2u]
    };
    uint physicalVertices[3] =
    {
        metalFluidClothSurfaceSources[surfaceVertices[0]],
        metalFluidClothSurfaceSources[surfaceVertices[1]],
        metalFluidClothSurfaceSources[surfaceVertices[2]]
    };
    float cornerWeights[3] = {weights.x, weights.y, weights.z};
    float3 surfaceVelocity = float3(0.0f);
    for (uint corner = 0u; corner < 3u; ++corner)
        surfaceVelocity += metalLoad(
            metalFluidClothVelocities[physicalVertices[corner]]) *
            cornerWeights[corner];
    float3 velocity = metalLoad(metalFluidClothFluidVelocities[particle]);
    float3 relative = velocity - surfaceVelocity;
    float normalSpeed = dot(relative, faceNormal);
    float magnitude = min(constants.maximum_force,
        max(0.0f, constants.stiffness * violation +
            constants.damping * normalSpeed));
    float3 tangent = relative - faceNormal * normalSpeed;
    float3 force = metalLimit(
        -faceNormal * magnitude - tangent * constants.tangential_drag,
        constants.maximum_force);
    metalFluidClothAccelerations[particle] = metalStore(
        metalLoad(metalFluidClothAccelerations[particle]) +
        force / max(constants.particle_mass, 1.0e-12f));
    metalFluidClothFoam[particle] = max(
        metalFluidClothFoam[particle],
        clamp(magnitude / max(constants.maximum_force, 1.0f), 0.0f, 1.0f));
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        metalFluidClothContributionNodes[contribution + corner] =
            physicalVertices[corner];
        metalFluidClothContributionForces[contribution + corner] =
            metalStore(-force * cornerWeights[corner]);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_fluid_cloth_apply(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalFluidClothConstants constants = metalFluidClothConstants;
    if (node >= constants.cloth_count || constants.enabled == 0u) return;
    float3 force = float3(0.0f);
    for (uint particle = 0u;
         particle < metalFluidClothMetadata[0].count; ++particle)
    {
        uint contribution = particle * 3u;
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            uint item = contribution + corner;
            if (metalFluidClothContributionNodes[item] == node)
                force += metalLoad(metalFluidClothContributionForces[item]);
        }
    }
    metalFluidClothForces[node] = metalStore(
        metalLoad(metalFluidClothForces[node]) + force);
    float inverse = metalFluidClothInverseMasses[node];
    if (inverse <= 0.0f) return;
    float3 change = metalLimit(
        force * (inverse * constants.timestep), 2.0f);
    metalFluidClothVelocities[node] = metalStore(metalLimit(
        metalLoad(metalFluidClothVelocities[node]) + change, 20.0f));
    metalFluidClothPositions[node] = metalStore(
        metalLoad(metalFluidClothPositions[node]) +
        change * constants.timestep);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_fluid_cloth_surface_update(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint surface = dispatchThreadId.x;
    MetalFluidClothConstants constants = metalFluidClothConstants;
    if (surface >= constants.surface_count || constants.enabled == 0u) return;
    metalFluidClothSurfacePositions[surface] =
        metalFluidClothPositions[metalFluidClothSurfaceSources[surface]];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_fluid_cloth_project(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalFluidClothConstants constants = metalFluidClothConstants;
    if (particle >= metalFluidClothMetadata[0].count ||
        constants.enabled == 0u) return;
    float3 point = metalLoad(metalFluidClothFluidPositions[particle]);
    float nearestSquared = 3.402823466e+38f;
    float3 nearest = float3(0.0f);
    float3 normal = float3(0.0f);
    float3 weights = float3(0.0f);
    uint best = constants.surface_index_count;
    for (uint base = 0u; base + 2u < constants.surface_index_count;
         base += 3u)
    {
        float3 a = metalLoad(metalFluidClothSurfacePositions[
            metalFluidClothSurfaceIndices[base]]);
        float3 b = metalLoad(metalFluidClothSurfacePositions[
            metalFluidClothSurfaceIndices[base + 1u]]);
        float3 c = metalLoad(metalFluidClothSurfacePositions[
            metalFluidClothSurfaceIndices[base + 2u]]);
        float3 face = cross(b - a, c - a);
        float faceSquared = dot(face, face);
        if (faceSquared <= 1.0e-14f) continue;
        float3 candidate = metalClosestPointTriangle(point, a, b, c);
        float squared = dot(point - candidate, point - candidate);
        if (squared >= nearestSquared) continue;
        nearestSquared = squared;
        nearest = candidate;
        normal = face * rsqrt(faceSquared) * constants.orientation;
        weights = metalTriangleWeights(candidate, a, b, c);
        best = base;
    }
    if (best == constants.surface_index_count) return;
    float violation = constants.contact_distance + dot(point - nearest, normal);
    if (violation <= 0.0f) return;
    metalFluidClothFluidPositions[particle] =
        metalStore(point - normal * violation);
    uint physicalA = metalFluidClothSurfaceSources[
        metalFluidClothSurfaceIndices[best]];
    uint physicalB = metalFluidClothSurfaceSources[
        metalFluidClothSurfaceIndices[best + 1u]];
    uint physicalC = metalFluidClothSurfaceSources[
        metalFluidClothSurfaceIndices[best + 2u]];
    float3 surfaceVelocity =
        metalLoad(metalFluidClothVelocities[physicalA]) * weights.x +
        metalLoad(metalFluidClothVelocities[physicalB]) * weights.y +
        metalLoad(metalFluidClothVelocities[physicalC]) * weights.z;
    float3 relative =
        metalLoad(metalFluidClothFluidVelocities[particle]) - surfaceVelocity;
    float outwardSpeed = dot(relative, normal);
    if (outwardSpeed > 0.0f) relative -= normal * outwardSpeed;
    metalFluidClothFluidVelocities[particle] =
        metalStore(surfaceVelocity + relative);
}

#else
import types;
import math;
import device_intrinsics;
import geometry_fluid;

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_cloth_containment_forces(
    uniform Vec3* particle_positions, uniform Vec3* particle_velocities,
    uniform uint* particle_count, uniform Vec3* particle_accelerations,
    uniform float* foam_source, uniform float particle_mass,
    uniform Vec3* cloth_positions, uniform Vec3* cloth_velocities,
    uniform uint* cloth_indices, uniform uint triangle_count,
    uniform float cloth_orientation,
    uniform FluidClothCouplingOptions options, uniform Vec3* cloth_forces)
{
    uint particle = pmThreadIndex();
    if (particle >= *particle_count) return;
    Vec3 point = particle_positions[particle];
    float bestSquared = 3.402823466e38f;
    Vec3 bestPoint = {};
    Vec3 bestNormal = {};
    Vec3 bestWeights = {};
    uint bestTriangle = 0u;
    for (uint triangle = 0u; triangle < triangle_count; ++triangle)
    {
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
[numthreads(256, 1, 1)]
void cloth_apply_fluid_forces(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* inverse_masses, uniform Vec3* forces,
    uniform uint count, uniform float dt)
{
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
[numthreads(256, 1, 1)]
void fluid_project_inside_cloth(
    uniform Vec3* particle_positions, uniform Vec3* particle_velocities,
    uniform uint* particle_count, uniform float contact_distance,
    uniform Vec3* cloth_positions, uniform Vec3* cloth_velocities,
    uniform uint* cloth_indices, uniform uint triangle_count,
    uniform float cloth_orientation)
{
    uint particle = pmThreadIndex();
    if (particle >= *particle_count) return;
    Vec3 point = particle_positions[particle];
    float bestSquared = 3.402823466e38f;
    Vec3 bestPoint = {};
    Vec3 bestNormal = {};
    Vec3 bestWeights = {};
    uint bestTriangle = 0u;
    for (uint triangle = 0u; triangle < triangle_count; ++triangle)
    {
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

#endif
