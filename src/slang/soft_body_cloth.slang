// SPDX-License-Identifier: MIT
// Shared soft-body/cloth coupling for the CUDA mirror and Metal backend.

#ifdef PM_TARGET_METAL
import metal_common;

struct MetalSoftClothConstants
{
    float timestep;
    uint soft_count;
    uint soft_surface_count;
    uint cloth_count;
    uint cloth_surface_count;
    uint cloth_surface_index_count;
    float contact_distance;
    float friction;
    float maximum_soft_speed;
    uint solver_iterations;
    uint enabled;
};

struct MetalSoftClothContact
{
    uint vertices[3];
    float weights[3];
    float soft_inverse_mass_fraction;
    uint active;
    MetalPackedVec3 position_impulse;
    MetalPackedVec3 velocity_impulse;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftClothSoftPositions : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<MetalPackedVec3>
    metalSoftClothSoftPrevious : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftClothSoftVelocities : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<float>
    metalSoftClothSoftInverseMasses : register(t3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftClothSoftForces : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftClothClothPositions : register(u5);
[[vk::binding(6, 0)]] StructuredBuffer<MetalPackedVec3>
    metalSoftClothClothPrevious : register(t6);
[[vk::binding(7, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftClothClothVelocities : register(u7);
[[vk::binding(8, 0)]] StructuredBuffer<float>
    metalSoftClothClothInverseMasses : register(t8);
[[vk::binding(9, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftClothClothSurfacePositions : register(u9);
[[vk::binding(10, 0)]] StructuredBuffer<uint>
    metalSoftClothClothSurfaceIndices : register(t10);
[[vk::binding(11, 0)]] StructuredBuffer<uint>
    metalSoftClothClothSurfaceSources : register(t11);
[[vk::binding(12, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftClothClothForces : register(u12);
[[vk::binding(13, 0)]] ConstantBuffer<MetalSoftClothConstants>
    metalSoftClothConstants : register(b13);
[[vk::binding(14, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftClothSoftSurfacePositions : register(u14);
[[vk::binding(15, 0)]] StructuredBuffer<MetalSurfaceBinding>
    metalSoftClothSoftSurfaceBindings : register(t15);
[[vk::binding(16, 0)]] StructuredBuffer<MetalPackedVec3>
    metalSoftClothSoftRestPositions : register(t16);
[[vk::binding(17, 0)]] StructuredBuffer<MetalPackedVec3>
    metalSoftClothSoftSurfaceRestPositions : register(t17);
[[vk::binding(22, 0)]] RWStructuredBuffer<MetalSoftClothContact>
    metalSoftClothContacts : register(u22);
[[vk::binding(23, 0)]] RWStructuredBuffer<uint>
    metalSoftClothContactCounts : register(u23);

[ForceInline]
static float metalSoftClothRelaxation(MetalSoftClothContact contact)
{
    uint degree = 1u;
    for (uint corner = 0u; corner < 3u; ++corner)
        if (contact.weights[corner] > 1.0e-6f)
            degree = max(
                degree, metalSoftClothContactCounts[contact.vertices[corner]]);
    float softFraction = contact.soft_inverse_mass_fraction;
    return 1.0f /
        (softFraction + float(degree) * (1.0f - softFraction));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_cloth_contact(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint softNode = dispatchThreadId.x;
    MetalSoftClothConstants constants = metalSoftClothConstants;
    if (softNode >= constants.soft_count || constants.enabled == 0u) return;
    metalSoftClothContacts[softNode].active = 0u;
    float softInverse = metalSoftClothSoftInverseMasses[softNode];
    if (softInverse <= 0.0f) return;
    float3 point = metalLoad(metalSoftClothSoftPositions[softNode]);
    float3 start = metalLoad(metalSoftClothSoftPrevious[softNode]);
    float nearestSquared = 3.402823466e38f;
    float3 nearest = float3(0.0f);
    float3 bestNormal = float3(0.0f);
    float3 bestWeights = float3(0.0f);
    uint best = constants.cloth_surface_index_count;
    for (uint base = 0u;
         base + 2u < constants.cloth_surface_index_count; base += 3u)
    {
        uint surfaceA = metalSoftClothClothSurfaceIndices[base];
        uint surfaceB = metalSoftClothClothSurfaceIndices[base + 1u];
        uint surfaceC = metalSoftClothClothSurfaceIndices[base + 2u];
        float3 a = metalLoad(metalSoftClothClothSurfacePositions[surfaceA]);
        float3 b = metalLoad(metalSoftClothClothSurfacePositions[surfaceB]);
        float3 c = metalLoad(metalSoftClothClothSurfacePositions[surfaceC]);
        float3 face = cross(b - a, c - a);
        float faceSquared = dot(face, face);
        if (faceSquared <= 1.0e-14f) continue;
        float3 faceNormal = face * rsqrt(faceSquared);
        float3 candidate = metalClosestPointTriangle(point, a, b, c);
        float3 delta = point - candidate;
        float squared = dot(delta, delta);
        float3 weights = metalTriangleWeights(candidate, a, b, c);
        uint physicalA = metalSoftClothClothSurfaceSources[surfaceA];
        uint physicalB = metalSoftClothClothSurfaceSources[surfaceB];
        uint physicalC = metalSoftClothClothSurfaceSources[surfaceC];
        float3 oldNearest =
            metalLoad(metalSoftClothClothPrevious[physicalA]) * weights.x +
            metalLoad(metalSoftClothClothPrevious[physicalB]) * weights.y +
            metalLoad(metalSoftClothClothPrevious[physicalC]) * weights.z;
        float before = dot(start - oldNearest, faceNormal);
        float after = dot(delta, faceNormal);
        float travel = length(point - start) + length(candidate - oldNearest);
        bool crossed = before * after < 0.0f &&
            squared <= (travel + constants.contact_distance) *
                (travel + constants.contact_distance);
        if (!crossed && squared >=
            constants.contact_distance * constants.contact_distance)
            continue;
        if (squared >= nearestSquared) continue;
        nearestSquared = squared;
        nearest = candidate;
        bestWeights = weights;
        bestNormal = crossed || squared <= 1.0e-14f
            ? faceNormal * (before >= 0.0f ? 1.0f : -1.0f)
            : delta * rsqrt(squared);
        best = base;
    }
    if (best == constants.cloth_surface_index_count) return;
    float penetration =
        constants.contact_distance - dot(point - nearest, bestNormal);
    if (penetration <= 0.0f) return;
    uint surfaceVertices[3];
    surfaceVertices[0] = metalSoftClothClothSurfaceIndices[best];
    surfaceVertices[1] = metalSoftClothClothSurfaceIndices[best + 1u];
    surfaceVertices[2] = metalSoftClothClothSurfaceIndices[best + 2u];
    uint physicalVertices[3];
    physicalVertices[0] =
        metalSoftClothClothSurfaceSources[surfaceVertices[0]];
    physicalVertices[1] =
        metalSoftClothClothSurfaceSources[surfaceVertices[1]];
    physicalVertices[2] =
        metalSoftClothClothSurfaceSources[surfaceVertices[2]];
    float cornerWeights[3];
    cornerWeights[0] = bestWeights.x;
    cornerWeights[1] = bestWeights.y;
    cornerWeights[2] = bestWeights.z;
    float denominator = softInverse;
    float3 surfaceVelocity = float3(0.0f);
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        uint clothNode = physicalVertices[corner];
        float weight = cornerWeights[corner];
        denominator += weight * weight *
            metalSoftClothClothInverseMasses[clothNode];
        surfaceVelocity +=
            metalLoad(metalSoftClothClothVelocities[clothNode]) * weight;
    }
    if (denominator <= 1.0e-12f) return;
    MetalSoftClothContact contact;
    contact.position_impulse = metalStore(bestNormal *
        (min(penetration, 2.0f * constants.contact_distance) / denominator));
    contact.soft_inverse_mass_fraction = softInverse / denominator;
    float3 relative =
        metalLoad(metalSoftClothSoftVelocities[softNode]) - surfaceVelocity;
    float normalSpeed = dot(relative, bestNormal);
    float normalImpulse = max(0.0f, -normalSpeed) / denominator;
    float3 tangent = relative - bestNormal * normalSpeed;
    contact.velocity_impulse = metalStore(
        bestNormal * normalImpulse - metalLimit(
            tangent / denominator, constants.friction * normalImpulse));
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        uint clothNode = physicalVertices[corner];
        float weight = cornerWeights[corner];
        contact.vertices[corner] = clothNode;
        contact.weights[corner] = weight;
        if (weight > 1.0e-6f &&
            metalSoftClothClothInverseMasses[clothNode] > 0.0f)
            InterlockedAdd(metalSoftClothContactCounts[clothNode], 1u);
    }
    contact.active = 1u;
    metalSoftClothContacts[softNode] = contact;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_cloth_clear_counts(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint clothNode = dispatchThreadId.x;
    if (clothNode < metalSoftClothConstants.cloth_count)
        metalSoftClothContactCounts[clothNode] = 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_cloth_apply_soft(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint softNode = dispatchThreadId.x;
    MetalSoftClothConstants constants = metalSoftClothConstants;
    if (softNode >= constants.soft_count || constants.enabled == 0u ||
        metalSoftClothContacts[softNode].active == 0u)
        return;
    MetalSoftClothContact contact = metalSoftClothContacts[softNode];
    float relaxation = metalSoftClothRelaxation(contact);
    float inverse = metalSoftClothSoftInverseMasses[softNode];
    metalSoftClothSoftPositions[softNode] = metalStore(
        metalLoad(metalSoftClothSoftPositions[softNode]) +
        metalLoad(contact.position_impulse) * (relaxation * inverse));
    float3 impulse = metalLoad(contact.velocity_impulse) * relaxation;
    metalSoftClothSoftVelocities[softNode] = metalStore(metalLimit(
        metalLoad(metalSoftClothSoftVelocities[softNode]) + impulse * inverse,
        constants.maximum_soft_speed));
    metalSoftClothSoftForces[softNode] = metalStore(
        metalLoad(metalSoftClothSoftForces[softNode]) +
        impulse / max(constants.timestep, 1.0e-12f));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_cloth_apply(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint clothNode = dispatchThreadId.x;
    MetalSoftClothConstants constants = metalSoftClothConstants;
    if (clothNode >= constants.cloth_count || constants.enabled == 0u) return;
    float3 positionChange = float3(0.0f);
    float3 velocityChange = float3(0.0f);
    for (uint softNode = 0u; softNode < constants.soft_count; ++softNode)
    {
        MetalSoftClothContact contact = metalSoftClothContacts[softNode];
        if (contact.active == 0u) continue;
        float relaxation = metalSoftClothRelaxation(contact);
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            if (contact.vertices[corner] != clothNode) continue;
            float scale = -contact.weights[corner] * relaxation;
            positionChange += metalLoad(contact.position_impulse) * scale;
            velocityChange += metalLoad(contact.velocity_impulse) * scale;
        }
    }
    float inverse = metalSoftClothClothInverseMasses[clothNode];
    metalSoftClothClothPositions[clothNode] = metalStore(
        metalLoad(metalSoftClothClothPositions[clothNode]) +
        positionChange * inverse);
    metalSoftClothClothVelocities[clothNode] = metalStore(metalLimit(
        metalLoad(metalSoftClothClothVelocities[clothNode]) +
            velocityChange * inverse,
        20.0f));
    metalSoftClothClothForces[clothNode] = metalStore(
        metalLoad(metalSoftClothClothForces[clothNode]) +
        velocityChange / max(constants.timestep, 1.0e-12f));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_cloth_surface_update(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint surface = dispatchThreadId.x;
    MetalSoftClothConstants constants = metalSoftClothConstants;
    if (surface >= constants.cloth_surface_count || constants.enabled == 0u)
        return;
    metalSoftClothClothSurfacePositions[surface] =
        metalSoftClothClothPositions[
            metalSoftClothClothSurfaceSources[surface]];
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_cloth_soft_surface_update(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint surface = dispatchThreadId.x;
    MetalSoftClothConstants constants = metalSoftClothConstants;
    if (surface >= constants.soft_surface_count || constants.enabled == 0u)
        return;
    MetalSurfaceBinding binding =
        metalSoftClothSoftSurfaceBindings[surface];
    float3 point = metalLoad(
        metalSoftClothSoftSurfaceRestPositions[surface]);
    for (uint slot = 0u; slot < 4u; ++slot)
    {
        uint node = binding.nodes[slot];
        point += (metalLoad(metalSoftClothSoftPositions[node]) -
            metalLoad(metalSoftClothSoftRestPositions[node])) *
            binding.weights[slot];
    }
    metalSoftClothSoftSurfacePositions[surface] = metalStore(point);
}

#else
import types;
import math;
import device_intrinsics;
import geometry_fluid;

[shader("compute")]
[numthreads(256, 1, 1)]
void soft_cloth_detect(
    uniform Vec3* positions, uniform Vec3* previous,
    uniform Vec3* velocities, uniform float* inverse_masses,
    uniform uint count, uniform Vec3* cloth_positions,
    uniform Vec3* cloth_previous, uniform Vec3* cloth_velocities,
    uniform float* cloth_inverse_masses, uniform uint* indices,
    uniform Vec3* surface, uniform uint triangle_count,
    uniform float distance, uniform float friction,
    uniform SoftClothContact* contacts, uniform uint* contact_counts)
{
    uint node = pmThreadIndex();
    if (node >= count) return;
    contacts[node] = {};
    if (inverse_masses[node] == 0.0f) return;
    Vec3 point = positions[node];
    Vec3 start = previous[node];
    float bestSquared = 3.402823466e38f;
    Vec3 bestNormal = {};
    Vec3 bestWeights = {};
    Vec3 bestPoint = {};
    uint best = kInvalidDense;
    for (uint triangle = 0u; triangle < triangle_count; ++triangle)
    {
        uint base = triangle * 3u;
        Vec3 a = surface != nullptr ? surface[base] :
            cloth_positions[indices[base]];
        Vec3 b = surface != nullptr ? surface[base + 1u] :
            cloth_positions[indices[base + 1u]];
        Vec3 c = surface != nullptr ? surface[base + 2u] :
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
            pmAtomicAdd(&contact_counts[vertex], 1u);
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

float soft_cloth_relaxation(SoftClothContact contact, uint* counts)
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
[numthreads(256, 1, 1)]
void soft_cloth_apply_soft(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* forces, uniform float* inverse_masses,
    uniform uint count, uniform SoftClothContact* contacts,
    uniform uint* counts, uniform float dt, uniform float maximum_speed)
{
    uint node = pmThreadIndex();
    if (node >= count || !contacts[node].active) return;
    SoftClothContact contact = contacts[node];
    float relaxation = soft_cloth_relaxation(contact, counts);
    positions[node] = add(positions[node], multiply(
        contact.position_impulse, relaxation * inverse_masses[node]));
    Vec3 impulse = multiply(contact.velocity_impulse, relaxation);
    velocities[node] = clamp_length(add(velocities[node],
        multiply(impulse, inverse_masses[node])), maximum_speed);
    forces[node] = add(forces[node], multiply(impulse, 1.0f / dt));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void soft_cloth_apply_cloth(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* forces, uniform float* inverse_masses,
    uniform uint count, uniform SoftClothContact* contacts,
    uniform uint contact_count, uniform uint* counts, uniform float dt)
{
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
    positions[vertex] = add(positions[vertex],
        multiply(positionImpulse, inverse_masses[vertex]));
    velocities[vertex] = clamp_length(add(velocities[vertex],
        multiply(velocityImpulse, inverse_masses[vertex])), 20.0f);
    forces[vertex] = add(forces[vertex],
        multiply(velocityImpulse, 1.0f / dt));
}
#endif
