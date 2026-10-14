// SPDX-License-Identifier: MIT
// Production serial particle-system coupling dispatcher for Metal.

import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    couplingPositionsA : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    couplingVelocitiesA : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<float>
    couplingInverseMassesA : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<MetalPackedVec3>
    couplingPositionsB : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalPackedVec3>
    couplingVelocitiesB : register(u4);
[[vk::binding(5, 0)]] StructuredBuffer<float>
    couplingInverseMassesB : register(t5);
[[vk::binding(6, 0)]] ConstantBuffer<MetalCouplingConstants>
    couplingConstants : register(b6);
[[vk::binding(7, 0)]] RWStructuredBuffer<MetalPackedVec3>
    couplingDiagnosticsA : register(u7);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalPackedVec3>
    couplingDiagnosticsB : register(u8);
[[vk::binding(9, 0)]] StructuredBuffer<MetalParticleMetadata>
    couplingParticleMetadata : register(t9);
[[vk::binding(10, 0)]] RWStructuredBuffer<uint>
    couplingContactCount : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<float>
    couplingMaximumPenetration : register(u11);

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_particle_coupling_serial(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u || couplingConstants.enabled == 0u) return;
    MetalCouplingConstants constants = couplingConstants;

    if (constants.mode == 1u)
    {
        for (uint selected = 0u; selected < 2u; ++selected)
        {
            uint clothNode = selected == 0u
                ? constants.first_vertex : constants.last_vertex;
            if (clothNode == 0xffffffffu || constants.count_a == 0u ||
                clothNode >= constants.count_b)
                continue;
            uint endpoint = selected == 0u ? 0u : constants.count_a - 1u;
            float3 clothPosition = metalLoad(couplingPositionsB[clothNode]);
            float3 clothVelocity = metalLoad(couplingVelocitiesB[clothNode]);
            if (couplingInverseMassesB[clothNode] > 0.0f)
            {
                float3 movement = metalLimit(
                    metalLoad(couplingPositionsA[endpoint]) - clothPosition,
                    constants.maximum_force * constants.timestep *
                        constants.timestep);
                float3 velocityChange =
                    movement / max(constants.timestep, 1.0e-12f);
                clothPosition += movement;
                clothVelocity += velocityChange;
                couplingPositionsB[clothNode] = metalStore(clothPosition);
                couplingVelocitiesB[clothNode] = metalStore(clothVelocity);
                couplingDiagnosticsB[clothNode] = metalStore(
                    metalLoad(couplingDiagnosticsB[clothNode]) +
                    velocityChange * (constants.damping /
                        couplingInverseMassesB[clothNode]));
            }
            couplingPositionsA[endpoint] = metalStore(clothPosition);
            couplingVelocitiesA[endpoint] = metalStore(clothVelocity);
        }
        return;
    }

    if (constants.mode == 2u)
    {
        float radiusSquared =
            constants.contact_distance * constants.contact_distance;
        for (uint target = 0u; target < constants.count_b; ++target)
        {
            if (couplingInverseMassesB[target] <= 0.0f) continue;
            float3 average = float3(0.0f);
            uint neighbors = 0u;
            for (uint source = 0u; source < constants.count_a; ++source)
            {
                float3 delta = metalLoad(couplingPositionsA[source]) -
                    metalLoad(couplingPositionsB[target]);
                if (dot(delta, delta) <= radiusSquared)
                {
                    average += metalLoad(couplingVelocitiesA[source]);
                    ++neighbors;
                }
            }
            if (neighbors == 0u) continue;
            average /= float(neighbors);
            float3 change =
                (average - metalLoad(couplingVelocitiesB[target])) *
                (constants.stiffness * constants.timestep);
            change = metalLimit(
                change, constants.maximum_force * constants.timestep);
            couplingVelocitiesB[target] = metalStore(
                metalLoad(couplingVelocitiesB[target]) + change);
        }
        return;
    }

    if (constants.mode == 4u)
    {
        float distanceLimit = max(constants.contact_distance, 1.0e-5f);
        uint contactCount = 0u;
        float maximumPenetration = 0.0f;
        for (uint particle = 0u;
             particle < couplingParticleMetadata[0].count; ++particle)
        {
            float3 particlePosition = metalLoad(couplingPositionsA[particle]);
            float nearestSquared = distanceLimit * distanceLimit;
            uint nearestSegment = constants.count_b;
            float nearestFraction = 0.0f;
            float3 nearestPoint = float3(0.0f);
            for (uint segment = 0u; segment + 1u < constants.count_b;
                 ++segment)
            {
                float3 first = metalLoad(couplingPositionsB[segment]);
                float3 second = metalLoad(couplingPositionsB[segment + 1u]);
                float3 axis = second - first;
                float fraction = clamp(
                    dot(particlePosition - first, axis) /
                        max(dot(axis, axis), 1.0e-12f),
                    0.0f, 1.0f);
                float3 point = first + axis * fraction;
                float squared = dot(
                    particlePosition - point, particlePosition - point);
                if (squared >= nearestSquared) continue;
                nearestSquared = squared;
                nearestSegment = segment;
                nearestFraction = fraction;
                nearestPoint = point;
            }
            if (nearestSegment == constants.count_b) continue;
            uint next = nearestSegment + 1u;
            float distanceValue = sqrt(max(nearestSquared, 0.0f));
            ++contactCount;
            maximumPenetration = max(
                maximumPenetration, distanceLimit - distanceValue);
            float3 ropeVelocity =
                metalLoad(couplingVelocitiesB[nearestSegment]) *
                    (1.0f - nearestFraction) +
                metalLoad(couplingVelocitiesB[next]) * nearestFraction;
            float3 relative =
                metalLoad(couplingVelocitiesA[particle]) - ropeVelocity;
            float3 normal;
            if (distanceValue > 1.0e-6f)
                normal = (particlePosition - nearestPoint) / distanceValue;
            else
            {
                float relativeSquared = dot(relative, relative);
                normal = relativeSquared > 1.0e-12f
                    ? -relative * rsqrt(relativeSquared)
                    : float3(0.0f, 1.0f, 0.0f);
            }
            couplingPositionsA[particle] = metalStore(
                particlePosition + normal * (distanceLimit - distanceValue));
            float approach = min(dot(relative, normal), 0.0f);
            float3 tangent = relative - normal * dot(relative, normal);
            float3 particleChange =
                -tangent * max(constants.friction, 0.0f) - normal * approach;
            couplingVelocitiesA[particle] = metalStore(
                metalLoad(couplingVelocitiesA[particle]) + particleChange);
            couplingDiagnosticsA[particle] = metalStore(
                metalLoad(couplingDiagnosticsA[particle]) +
                particleChange / max(constants.timestep, 1.0e-12f));

            float3 reactionImpulse =
                -particleChange * max(constants.stiffness, 0.0f);
            for (uint endpoint = 0u; endpoint < 2u; ++endpoint)
            {
                uint node = endpoint == 0u ? nearestSegment : next;
                float weight = endpoint == 0u
                    ? 1.0f - nearestFraction : nearestFraction;
                float nodeInverse = couplingInverseMassesB[node];
                if (nodeInverse <= 0.0f) continue;
                float3 nodeChange = metalLimit(
                    reactionImpulse * weight * nodeInverse,
                    constants.maximum_force * constants.timestep);
                couplingVelocitiesB[node] = metalStore(
                    metalLoad(couplingVelocitiesB[node]) + nodeChange);
                couplingDiagnosticsB[node] = metalStore(
                    metalLoad(couplingDiagnosticsB[node]) +
                    nodeChange * (constants.damping / nodeInverse));
            }
        }
        couplingContactCount[0] += contactCount;
        couplingMaximumPenetration[0] = max(
            couplingMaximumPenetration[0], maximumPenetration);
        return;
    }

    if (constants.mode == 5u)
    {
        float distanceLimit = max(constants.contact_distance, 1.0e-5f);
        float radiusSquared = distanceLimit * distanceLimit;
        for (uint node = 0u; node < constants.count_b; ++node)
        {
            if (couplingInverseMassesB[node] <= 0.0f) continue;
            float3 average = float3(0.0f);
            uint neighbors = 0u;
            for (uint particle = 0u; particle < constants.count_a; ++particle)
            {
                float3 delta = metalLoad(couplingPositionsA[particle]) -
                    metalLoad(couplingPositionsB[node]);
                if (dot(delta, delta) <= radiusSquared)
                {
                    average += metalLoad(couplingVelocitiesA[particle]);
                    ++neighbors;
                }
            }
            if (neighbors != 0u)
            {
                average /= float(neighbors);
                float3 change =
                    (average - metalLoad(couplingVelocitiesB[node])) *
                    (constants.stiffness * constants.timestep);
                change = metalLimit(
                    change, constants.maximum_force * constants.timestep);
                couplingVelocitiesB[node] = metalStore(
                    metalLoad(couplingVelocitiesB[node]) + change);
            }
        }
        for (uint particle = 0u; particle < constants.count_a; ++particle)
        {
            float nearestSquared = radiusSquared;
            uint nearestSegment = constants.count_b;
            float nearestFraction = 0.0f;
            float3 nearestPoint = float3(0.0f);
            float3 particlePosition = metalLoad(couplingPositionsA[particle]);
            for (uint segment = 0u; segment + 1u < constants.count_b;
                 ++segment)
            {
                float3 first = metalLoad(couplingPositionsB[segment]);
                float3 second = metalLoad(couplingPositionsB[segment + 1u]);
                float3 axis = second - first;
                float fraction = clamp(
                    dot(particlePosition - first, axis) /
                        max(dot(axis, axis), 1.0e-12f),
                    0.0f, 1.0f);
                float3 point = first + axis * fraction;
                float squared = dot(
                    particlePosition - point, particlePosition - point);
                if (squared >= nearestSquared) continue;
                nearestSquared = squared;
                nearestSegment = segment;
                nearestFraction = fraction;
                nearestPoint = point;
            }
            if (nearestSegment == constants.count_b) continue;
            uint next = nearestSegment + 1u;
            float3 ropeVelocity =
                metalLoad(couplingVelocitiesB[nearestSegment]) *
                    (1.0f - nearestFraction) +
                metalLoad(couplingVelocitiesB[next]) * nearestFraction;
            float3 relative =
                metalLoad(couplingVelocitiesA[particle]) - ropeVelocity;
            float separation = sqrt(max(nearestSquared, 0.0f));
            float3 normal;
            if (separation > 1.0e-6f)
                normal = (particlePosition - nearestPoint) / separation;
            else
            {
                float relativeSquared = dot(relative, relative);
                normal = relativeSquared > 1.0e-12f
                    ? -relative * rsqrt(relativeSquared)
                    : float3(0.0f, 1.0f, 0.0f);
            }
            couplingPositionsA[particle] =
                metalStore(nearestPoint + normal * distanceLimit);
            relative -= normal * min(dot(relative, normal), 0.0f);
            couplingVelocitiesA[particle] =
                metalStore(ropeVelocity + relative);
        }
        return;
    }

    if (constants.mode == 0u &&
        (constants.first_vertex != 0xffffffffu ||
         constants.last_vertex != 0xffffffffu))
    {
        for (uint selected = 0u; selected < 2u; ++selected)
        {
            uint endpoint = selected == 0u
                ? constants.first_vertex : constants.last_vertex;
            if (endpoint == 0xffffffffu || endpoint >= constants.count_a ||
                constants.count_b == 0u)
                continue;
            float3 ropePosition = metalLoad(couplingPositionsA[endpoint]);
            float nearestSquared = 3.402823466e38f;
            uint nearestNode = 0u;
            for (uint node = 0u; node < constants.count_b; ++node)
            {
                float3 delta =
                    metalLoad(couplingPositionsB[node]) - ropePosition;
                float squared = dot(delta, delta);
                if (squared < nearestSquared)
                {
                    nearestSquared = squared;
                    nearestNode = node;
                }
            }
            float ropeInverse = couplingInverseMassesA[endpoint];
            float softInverse = couplingInverseMassesB[nearestNode];
            float inverseSum = ropeInverse + softInverse;
            if (inverseSum <= 0.0f) continue;
            float3 softPosition = metalLoad(couplingPositionsB[nearestNode]);
            float3 separation = softPosition - ropePosition;
            float3 softChange = metalLimit(
                -separation * (softInverse / inverseSum),
                constants.maximum_force * constants.timestep *
                    constants.timestep);
            softPosition += softChange;
            float3 ropeVelocity = metalLoad(couplingVelocitiesA[endpoint]);
            float3 softVelocity = metalLoad(couplingVelocitiesB[nearestNode]);
            float3 softVelocityChange = metalLimit(
                (ropeVelocity - softVelocity) * (softInverse / inverseSum),
                constants.maximum_force * constants.timestep);
            softVelocity += softVelocityChange;
            couplingPositionsB[nearestNode] = metalStore(softPosition);
            couplingVelocitiesB[nearestNode] = metalStore(softVelocity);
            couplingPositionsA[endpoint] = metalStore(softPosition);
            couplingVelocitiesA[endpoint] = metalStore(softVelocity);
        }
    }

    float distanceLimit = max(constants.contact_distance, 1.0e-5f);
    for (uint first = 0u; first < constants.count_a; ++first)
    {
        if (constants.mode == 0u &&
            (first == constants.first_vertex || first == constants.last_vertex))
            continue;
        for (uint second = 0u; second < constants.count_b; ++second)
        {
            float3 a = metalLoad(couplingPositionsA[first]);
            float3 b = metalLoad(couplingPositionsB[second]);
            float3 delta = b - a;
            float squared = dot(delta, delta);
            if (squared >= distanceLimit * distanceLimit ||
                squared <= 1.0e-12f)
                continue;
            float distanceValue = sqrt(squared);
            float3 normal = delta / distanceValue;
            if (constants.mode == 3u)
            {
                float blend = min(
                    constants.damping * constants.timestep, 1.0f);
                couplingVelocitiesA[first] = metalStore(lerp(
                    metalLoad(couplingVelocitiesA[first]),
                    metalLoad(couplingVelocitiesB[second]), blend));
                continue;
            }
            float inverseA = couplingInverseMassesA[first];
            float inverseB = couplingInverseMassesB[second];
            float inverseSum = inverseA + inverseB;
            if (inverseSum <= 0.0f) continue;
            float penetration = distanceLimit - distanceValue;
            float correctionScale = min(
                penetration * max(constants.stiffness, 1.0f) *
                    constants.timestep,
                penetration);
            float3 correction = normal * correctionScale;
            a -= correction * (inverseA / inverseSum);
            b += correction * (inverseB / inverseSum);
            couplingPositionsA[first] = metalStore(a);
            couplingPositionsB[second] = metalStore(b);
            float3 relative = metalLoad(couplingVelocitiesB[second]) -
                metalLoad(couplingVelocitiesA[first]);
            float normalSpeed = dot(relative, normal);
            float impulse = -normalSpeed * max(constants.damping, 0.0f);
            if (constants.maximum_force > 0.0f)
                impulse = clamp(
                    impulse,
                    -constants.maximum_force * constants.timestep,
                    constants.maximum_force * constants.timestep);
            couplingVelocitiesA[first] = metalStore(
                metalLoad(couplingVelocitiesA[first]) -
                normal * impulse * (inverseA / inverseSum));
            couplingVelocitiesB[second] = metalStore(
                metalLoad(couplingVelocitiesB[second]) +
                normal * impulse * (inverseB / inverseSum));
        }
    }
}
