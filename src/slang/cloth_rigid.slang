// SPDX-License-Identifier: MIT
// Metal compatibility adapter for the serial cloth/rigid surface solve.

import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    clothRigidPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    clothRigidVelocities : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<float>
    clothRigidInverseMasses : register(t2);
[[vk::binding(3, 0)]] StructuredBuffer<uint>
    clothRigidTriangleIndices : register(t3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalRigidState>
    clothRigidStates : register(u4);
[[vk::binding(5, 0)]] StructuredBuffer<MetalRigidParameters>
    clothRigidParameters : register(t5);
[[vk::binding(6, 0)]] StructuredBuffer<MetalPackedVec3>
    clothRigidMeshVertices : register(t6);
[[vk::binding(7, 0)]] StructuredBuffer<MetalTriangleMeshInfo>
    clothRigidMeshes : register(t7);
[[vk::binding(8, 0)]] StructuredBuffer<MetalRigidState>
    clothRigidPreviousStates : register(t8);
[[vk::binding(9, 0)]] ConstantBuffer<MetalClothRigidConstants>
    clothRigidConstants : register(b9);
[[vk::binding(10, 0)]] RWStructuredBuffer<MetalClothBodyCorrection>
    clothRigidCorrections : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<MetalPackedVec3>
    clothRigidSurfacePositions : register(u11);
[[vk::binding(12, 0)]] StructuredBuffer<uint>
    clothRigidSurfacePhysicalIndices : register(t12);

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_cloth_rigid_surface_serial(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u) return;
    MetalClothRigidConstants constants = clothRigidConstants;

    for (uint bodyIndex = 0u; bodyIndex < constants.rigid_count;
         ++bodyIndex)
    {
        MetalClothBodyCorrection cleared;
        cleared.offset = metalStore(float3(0.0f));
        cleared.impulse = metalStore(float3(0.0f));
        cleared.contact = metalStore(float3(0.0f));
        cleared.support_radius = 0.0f;
        cleared.weight_sum = 0.0f;
        cleared.vertices[0] = 0xffffffffu;
        cleared.vertices[1] = 0xffffffffu;
        cleared.vertices[2] = 0xffffffffu;
        cleared.active = 0u;
        clothRigidCorrections[bodyIndex] = cleared;

        MetalRigidParameters body = clothRigidParameters[bodyIndex];
        if (body.motion != 2u || constants.fracture_enabled != 0u)
            continue;

        MetalRigidState state = clothRigidStates[bodyIndex];
        MetalTriangleMeshInfo mesh = clothRigidMeshes[body.mesh_index];
        float3 localCenter =
            (metalLoad(mesh.minimum) + metalLoad(mesh.maximum)) * 0.5f;
        float meshRadius = 0.0f;
        for (uint node = 0u; node < mesh.vertex_count; ++node)
        {
            float3 delta = metalLoad(
                clothRigidMeshVertices[mesh.vertex_offset + node]) -
                localCenter;
            meshRadius = max(meshRadius, length(delta));
        }
        float3 center = metalWorldPoint(state, localCenter);
        float3 previousCenter = metalWorldPoint(
            clothRigidPreviousStates[bodyIndex], localCenter);
        float radius = meshRadius + body.collision_margin +
            constants.thickness;

        float bestPenetration = 0.0f;
        float3 bestNormal = float3(0.0f);
        float3 bestContact = float3(0.0f);
        uint bestTriangle = 0u;
        for (uint base = 0u; base + 2u < constants.triangle_index_count;
             base += 3u)
        {
            uint ia = clothRigidTriangleIndices[base];
            uint ib = clothRigidTriangleIndices[base + 1u];
            uint ic = clothRigidTriangleIndices[base + 2u];
            if (ia >= constants.vertex_count ||
                ib >= constants.vertex_count ||
                ic >= constants.vertex_count)
                continue;
            float3 a = metalLoad(clothRigidPositions[ia]);
            float3 b = metalLoad(clothRigidPositions[ib]);
            float3 c = metalLoad(clothRigidPositions[ic]);
            float3 faceValue = cross(b - a, c - a);
            if (dot(faceValue, faceValue) < 1.0e-12f) continue;
            float3 face = normalize(faceValue);
            float3 nearest = metalClosestPointTriangle(center, a, b, c);
            float3 delta = center - nearest;
            float distanceValue = length(delta);
            float3 contactNormal = distanceValue > 1.0e-6f
                ? delta / distanceValue
                : face * (dot(previousCenter - a, face) >= 0.0f
                    ? 1.0f : -1.0f);
            float penetration = radius - distanceValue;
            float before = dot(previousCenter - a, face);
            float after = dot(center - a, face);
            if (before * after < 0.0f)
            {
                float fraction = before / (before - after);
                float3 crossing = previousCenter +
                    (center - previousCenter) * fraction;
                float3 crossingNearest = metalClosestPointTriangle(
                    crossing, a, b, c);
                if (dot(crossingNearest - crossing,
                        crossingNearest - crossing) < radius * radius)
                {
                    contactNormal = face * (before > 0.0f ? 1.0f : -1.0f);
                    penetration = max(penetration, radius + abs(after));
                }
            }
            else if (penetration > 0.0f && before * after > 0.0f)
                contactNormal = face * (before > 0.0f ? 1.0f : -1.0f);
            if (penetration > bestPenetration)
            {
                bestPenetration = penetration;
                bestNormal = contactNormal;
                bestContact = nearest;
                bestTriangle = base;
            }
        }
        if (bestPenetration <= 0.0f) continue;

        uint a = clothRigidTriangleIndices[bestTriangle];
        uint b = clothRigidTriangleIndices[bestTriangle + 1u];
        uint c = clothRigidTriangleIndices[bestTriangle + 2u];
        float freeFraction =
            ((clothRigidInverseMasses[a] > 0.0f ? 1.0f : 0.0f) +
             (clothRigidInverseMasses[b] > 0.0f ? 1.0f : 0.0f) +
             (clothRigidInverseMasses[c] > 0.0f ? 1.0f : 0.0f)) / 3.0f;
        float clothShift = 0.5f * bestPenetration;
        float3 arm = bestContact - metalLoad(state.position);
        float incoming = dot(metalLoad(state.linear_velocity), bestNormal);
        float3 normalAxis = cross(arm, bestNormal);
        float normalImpulse = incoming < 0.0f && body.inverse_mass > 0.0f
            ? -incoming / body.inverse_mass : 0.0f;
        float supportRadius = max(0.15f, meshRadius * 0.8f);
        float inverseSupportSquared = 1.0f /
            (supportRadius * supportRadius);
        float weightSum = 0.0f;
        float maximumWeightedInverseMass = 0.0f;
        float weightedInverseMassSquared = 0.0f;
        float3 weightedClothVelocity = float3(0.0f);
        for (uint node = 0u; node < constants.vertex_count; ++node)
        {
            float inverseMass = clothRigidInverseMasses[node];
            if (inverseMass <= 0.0f) continue;
            float3 delta = metalLoad(clothRigidPositions[node]) -
                bestContact;
            float weight = max(0.0f,
                1.0f - dot(delta, delta) * inverseSupportSquared);
            float weighted = weight * weight;
            weightSum += weighted;
            weightedInverseMassSquared +=
                inverseMass * weighted * weighted;
            weightedClothVelocity +=
                metalLoad(clothRigidVelocities[node]) * weighted;
            maximumWeightedInverseMass = max(
                maximumWeightedInverseMass, inverseMass * weighted);
        }
        float maximumClothImpulse = maximumWeightedInverseMass > 0.0f
            ? 0.1f * constants.thickness * weightSum /
                (constants.timestep * maximumWeightedInverseMass)
            : normalImpulse;
        float clothImpulse = min(normalImpulse, maximumClothImpulse);
        float3 tangentImpulse = float3(0.0f);
        if (weightSum > 1.0e-8f && constants.friction > 0.0f)
        {
            float3 clothVelocity = weightedClothVelocity / weightSum;
            float3 bodyVelocity = metalLoad(state.linear_velocity) +
                cross(metalLoad(state.angular_velocity), arm);
            float3 relative = bodyVelocity - clothVelocity;
            float separatingSpeed = dot(relative, bestNormal);
            float3 tangent = relative - bestNormal * separatingSpeed;
            float tangentSpeed = length(tangent);
            float releaseWeight =
                max(0.0f, 1.0f - max(incoming, 0.0f) / 0.5f);
            if (releaseWeight > 0.0f && separatingSpeed <= 0.0f &&
                tangentSpeed > 1.0e-6f)
            {
                float3 direction = tangent / tangentSpeed;
                float3 tangentAxis = cross(arm, direction);
                float clothInverseMass = weightedInverseMassSquared /
                    (weightSum * weightSum);
                float effectiveInverseMass = body.inverse_mass +
                    dot(cross(metalInverseInertiaMul(
                        body, state, tangentAxis), arm), direction) +
                    clothInverseMass;
                float correctionImpulse =
                    bestPenetration > 0.0f && body.inverse_mass > 0.0f
                    ? 0.2f * bestPenetration /
                        (constants.timestep * body.inverse_mass)
                    : 0.0f;
                float frictionLimit = constants.friction * releaseWeight *
                    max(normalImpulse, correctionImpulse);
                float magnitude = effectiveInverseMass > 1.0e-6f
                    ? min(tangentSpeed / effectiveInverseMass,
                        frictionLimit)
                    : 0.0f;
                tangentImpulse = direction * -magnitude;
            }
        }
        float3 clothTangentImpulse =
            metalLimit(tangentImpulse, maximumClothImpulse);
        MetalClothBodyCorrection correction;
        correction.offset = metalStore(bestNormal * -clothShift);
        correction.impulse = metalStore(
            bestNormal * -clothImpulse - clothTangentImpulse);
        correction.contact = metalStore(bestContact);
        correction.support_radius = supportRadius;
        correction.weight_sum = weightSum;
        correction.vertices[0] = a;
        correction.vertices[1] = b;
        correction.vertices[2] = c;
        correction.active = 1u;
        clothRigidCorrections[bodyIndex] = correction;

        state.position = metalStore(metalLoad(state.position) + bestNormal *
            (bestPenetration - clothShift * freeFraction));
        if (normalImpulse > 0.0f)
        {
            state.linear_velocity = metalStore(metalLimit(
                metalLoad(state.linear_velocity) + bestNormal *
                    (normalImpulse * body.inverse_mass),
                body.maximum_linear_speed));
            state.angular_velocity = metalStore(metalLimit(
                metalLoad(state.angular_velocity) +
                    metalInverseInertiaMul(
                        body, state, normalAxis * normalImpulse),
                body.maximum_angular_speed));
        }
        state.linear_velocity = metalStore(metalLimit(
            metalLoad(state.linear_velocity) +
                tangentImpulse * body.inverse_mass,
            body.maximum_linear_speed));
        state.angular_velocity = metalStore(metalLimit(
            metalLoad(state.angular_velocity) +
                metalInverseInertiaMul(
                    body, state, cross(arm, tangentImpulse)),
            body.maximum_angular_speed));
        clothRigidStates[bodyIndex] = state;
    }

    for (uint node = 0u; node < constants.vertex_count; ++node)
    {
        if (clothRigidInverseMasses[node] <= 0.0f) continue;
        float3 offset = float3(0.0f);
        float3 velocity = metalLoad(clothRigidVelocities[node]);
        for (uint body = 0u; body < constants.rigid_count; ++body)
        {
            MetalClothBodyCorrection correction =
                clothRigidCorrections[body];
            if (correction.active == 0u) continue;
            if (correction.vertices[0] == node ||
                correction.vertices[1] == node ||
                correction.vertices[2] == node)
                offset += metalLoad(correction.offset);
            if (correction.weight_sum > 1.0e-8f)
            {
                float3 delta = metalLoad(clothRigidPositions[node]) -
                    metalLoad(correction.contact);
                float supportSquared = correction.support_radius *
                    correction.support_radius;
                float weight = max(0.0f,
                    1.0f - dot(delta, delta) / supportSquared);
                velocity += metalLoad(correction.impulse) *
                    (clothRigidInverseMasses[node] * weight * weight /
                        correction.weight_sum);
            }
        }
        clothRigidPositions[node] = metalStore(
            metalLoad(clothRigidPositions[node]) + offset);
        clothRigidVelocities[node] = metalStore(metalLimit(velocity, 20.0f));
    }
    for (uint surface = 0u; surface < constants.surface_count; ++surface)
    {
        uint source = clothRigidSurfacePhysicalIndices[surface];
        if (source < constants.vertex_count)
            clothRigidSurfacePositions[surface] =
                clothRigidPositions[source];
    }
}
