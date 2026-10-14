// SPDX-License-Identifier: MIT
// Shared fluid/rope coupling for the CUDA mirror and Metal backend.

#ifdef PM_TARGET_METAL
import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidRopeFluidPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidRopeFluidVelocities : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<float>
    metalFluidRopeFluidInverseMasses : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidRopeRopePositions : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidRopeRopeVelocities : register(u4);
[[vk::binding(5, 0)]] StructuredBuffer<float>
    metalFluidRopeRopeInverseMasses : register(t5);
[[vk::binding(6, 0)]] ConstantBuffer<MetalCouplingConstants>
    metalFluidRopeConstants : register(b6);
[[vk::binding(7, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidRopeFluidAccelerations : register(u7);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidRopeRopeForces : register(u8);
[[vk::binding(9, 0)]] StructuredBuffer<MetalParticleMetadata>
    metalFluidRopeFluidMetadata : register(t9);
[[vk::binding(10, 0)]] RWStructuredBuffer<uint>
    metalFluidRopeContactCount : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<uint>
    metalFluidRopeMaximumPenetration : register(u11);
[[vk::binding(13, 0)]] RWStructuredBuffer<uint>
    metalFluidRopeContributionNodes : register(u13);
[[vk::binding(14, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidRopeContributionChanges : register(u14);
[[vk::binding(15, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidRopeContributionForces : register(u15);

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_rope_contacts(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalCouplingConstants constants = metalFluidRopeConstants;
    if (particle >= metalFluidRopeFluidMetadata[0].count ||
        constants.enabled == 0u)
        return;
    uint contribution = particle * 2u;
    metalFluidRopeContributionNodes[contribution] = 0xffffffffu;
    metalFluidRopeContributionNodes[contribution + 1u] = 0xffffffffu;
    metalFluidRopeContributionChanges[contribution] =
        metalStore(float3(0.0f));
    metalFluidRopeContributionChanges[contribution + 1u] =
        metalStore(float3(0.0f));
    metalFluidRopeContributionForces[contribution] =
        metalStore(float3(0.0f));
    metalFluidRopeContributionForces[contribution + 1u] =
        metalStore(float3(0.0f));
    if (constants.count_b < 2u) return;

    float distanceLimit = max(constants.contact_distance, 1.0e-5f);
    float3 particlePosition =
        metalLoad(metalFluidRopeFluidPositions[particle]);
    float nearestSquared = distanceLimit * distanceLimit;
    uint nearestSegment = constants.count_b;
    float nearestFraction = 0.0f;
    float3 nearestPoint = float3(0.0f);
    for (uint segment = 0u; segment + 1u < constants.count_b; ++segment)
    {
        float3 first = metalLoad(metalFluidRopeRopePositions[segment]);
        float3 second = metalLoad(metalFluidRopeRopePositions[segment + 1u]);
        if (particlePosition.x < min(first.x, second.x) - distanceLimit ||
            particlePosition.x > max(first.x, second.x) + distanceLimit ||
            particlePosition.y < min(first.y, second.y) - distanceLimit ||
            particlePosition.y > max(first.y, second.y) + distanceLimit ||
            particlePosition.z < min(first.z, second.z) - distanceLimit ||
            particlePosition.z > max(first.z, second.z) + distanceLimit)
            continue;
        float3 axis = second - first;
        float fraction = clamp(
            dot(particlePosition - first, axis) /
                max(dot(axis, axis), 1.0e-12f),
            0.0f, 1.0f);
        float3 point = first + axis * fraction;
        float squared = dot(particlePosition - point,
            particlePosition - point);
        if (squared >= nearestSquared) continue;
        nearestSquared = squared;
        nearestSegment = segment;
        nearestFraction = fraction;
        nearestPoint = point;
    }
    if (nearestSegment == constants.count_b) return;

    uint next = nearestSegment + 1u;
    float distanceValue = sqrt(max(nearestSquared, 0.0f));
    float penetration = distanceLimit - distanceValue;
    float3 ropeVelocity =
        metalLoad(metalFluidRopeRopeVelocities[nearestSegment]) *
            (1.0f - nearestFraction) +
        metalLoad(metalFluidRopeRopeVelocities[next]) * nearestFraction;
    float3 relative =
        metalLoad(metalFluidRopeFluidVelocities[particle]) - ropeVelocity;
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
    metalFluidRopeFluidPositions[particle] = metalStore(
        particlePosition + normal * penetration);
    float approach = min(dot(relative, normal), 0.0f);
    float3 tangent = relative - normal * dot(relative, normal);
    float3 fluidChange =
        -tangent * max(constants.friction, 0.0f) - normal * approach;
    metalFluidRopeFluidVelocities[particle] = metalStore(
        metalLoad(metalFluidRopeFluidVelocities[particle]) + fluidChange);
    InterlockedAdd(metalFluidRopeContactCount[0], 1u);
    InterlockedMax(
        metalFluidRopeMaximumPenetration[0], asuint(penetration));

    float3 reactionImpulse =
        -fluidChange * max(constants.stiffness, 0.0f);
    for (uint endpoint = 0u; endpoint < 2u; ++endpoint)
    {
        uint node = endpoint == 0u ? nearestSegment : next;
        float weight = endpoint == 0u
            ? 1.0f - nearestFraction : nearestFraction;
        metalFluidRopeContributionNodes[contribution + endpoint] = node;
        if (node == constants.first_vertex || node == constants.last_vertex ||
            metalFluidRopeRopeInverseMasses[node] <= 0.0f)
            continue;
        metalFluidRopeContributionChanges[contribution + endpoint] =
            metalStore(reactionImpulse * weight);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_rope_apply(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalCouplingConstants constants = metalFluidRopeConstants;
    if (node >= constants.count_b || constants.enabled == 0u ||
        node == constants.first_vertex || node == constants.last_vertex)
        return;
    float3 reactionImpulse = float3(0.0f);
    for (uint particle = 0u;
         particle < metalFluidRopeFluidMetadata[0].count; ++particle)
    {
        uint contribution = particle * 2u;
        for (uint endpoint = 0u; endpoint < 2u; ++endpoint)
        {
            uint item = contribution + endpoint;
            if (metalFluidRopeContributionNodes[item] == node)
                reactionImpulse +=
                    metalLoad(metalFluidRopeContributionChanges[item]);
        }
    }
    float inverse = metalFluidRopeRopeInverseMasses[node];
    if (inverse <= 0.0f) return;
    float3 velocityChange = metalLimit(
        reactionImpulse * inverse,
        constants.maximum_force * constants.timestep);
    metalFluidRopeRopeVelocities[node] = metalStore(
        metalLoad(metalFluidRopeRopeVelocities[node]) + velocityChange);
    metalFluidRopeRopeForces[node] = metalStore(
        metalLoad(metalFluidRopeRopeForces[node]) +
        velocityChange * (constants.damping / inverse));
}

#else
import types;
import math;
import device_intrinsics;

public void atomic_add(Vec3* destination, Vec3 value)
{
    pmAtomicAdd(&destination.x, value.x);
    pmAtomicAdd(&destination.y, value.y);
    pmAtomicAdd(&destination.z, value.z);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_rope_contacts(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform uint* particle_count, uniform float particle_mass,
    uniform Vec3* rope_positions, uniform Vec3* rope_velocities,
    uniform uint rope_count, uniform float distance, uniform float friction,
    uniform Vec3* node_impulses, uniform uint* contact_count,
    uniform float* maximum_penetration, uniform bool transfer_impulse)
{
    uint particle = pmThreadIndex();
    if (particle >= *particle_count) return;
    Vec3 position = positions[particle];
    float nearest = distance * distance;
    uint segment = rope_count;
    float fraction = 0.0f;
    Vec3 closest = {};
    for (uint i = 0u; i + 1u < rope_count; ++i)
    {
        Vec3 a = rope_positions[i];
        Vec3 b = rope_positions[i + 1u];
        if (position.x < min(a.x, b.x) - distance ||
            position.x > max(a.x, b.x) + distance ||
            position.y < min(a.y, b.y) - distance ||
            position.y > max(a.y, b.y) + distance ||
            position.z < min(a.z, b.z) - distance ||
            position.z > max(a.z, b.z) + distance) continue;
        Vec3 ab = subtract(b, a);
        float t = clamp_scalar(dot(subtract(position, a), ab) /
            max(length_squared(ab), 1.0e-12f), 0.0f, 1.0f);
        Vec3 point = add(a, multiply(ab, t));
        float squared = length_squared(subtract(position, point));
        if (squared >= nearest) continue;
        nearest = squared;
        segment = i;
        fraction = t;
        closest = point;
    }
    if (segment == rope_count) return;
    float separation = sqrt(nearest);
    Vec3 ropeVelocity = add(
        multiply(rope_velocities[segment], 1.0f - fraction),
        multiply(rope_velocities[segment + 1u], fraction));
    Vec3 relative = subtract(velocities[particle], ropeVelocity);
    Vec3 up = {0.0f, 1.0f, 0.0f};
    Vec3 normal = separation > 1.0e-6f
        ? multiply(subtract(position, closest), 1.0f / separation)
        : normalized_or(multiply(relative, -1.0f), up);
    float penetration = distance - separation;
    positions[particle] = add(position, multiply(normal, penetration));
    float approach = min(dot(relative, normal), 0.0f);
    Vec3 tangent = subtract(relative, multiply(normal, dot(relative, normal)));
    Vec3 velocityChange = subtract(
        multiply(tangent, -friction), multiply(normal, approach));
    velocities[particle] = add(velocities[particle], velocityChange);
    if (!transfer_impulse) return;
    pmAtomicAdd(contact_count, 1u);
    uint* penetrationBits = reinterpret<uint*>(maximum_penetration);
    pmAtomicMax(penetrationBits, asuint(penetration));
    Vec3 reaction = multiply(velocityChange, -particle_mass);
    atomic_add(node_impulses + segment,
        multiply(reaction, 1.0f - fraction));
    atomic_add(node_impulses + segment + 1u,
        multiply(reaction, fraction));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_rope_apply(
    uniform Vec3* rope_velocities, uniform Vec3* fluid_contact_forces,
    uniform uint rope_count, uniform float rope_mass,
    uniform float rope_maximum_speed, uniform bool first_attached,
    uniform bool last_attached,
    uniform Vec3* node_impulses, uniform float maximum_acceleration,
    uniform float dt, uniform float frame_inverse_dt)
{
    uint node = pmThreadIndex();
    if (node >= rope_count || (node == 0u && first_attached) ||
        (node + 1u == rope_count && last_attached))
        return;
    float mass = rope_mass / float(rope_count);
    Vec3 velocityChange = clamp_length(
        multiply(node_impulses[node], 1.0f / mass),
        maximum_acceleration * dt);
    rope_velocities[node] = clamp_length(
        add(rope_velocities[node], velocityChange), rope_maximum_speed);
    fluid_contact_forces[node] = add(
        fluid_contact_forces[node],
        multiply(velocityChange, mass * frame_inverse_dt));
}
#endif
