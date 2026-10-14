// SPDX-License-Identifier: MIT
// Legacy Metal argument-table adapter for rigid contact-cache persistence.

import metal_common;

[[vk::binding(0, 0)]] StructuredBuffer<MetalRigidState>
    rigidCacheStates : register(t0);
[[vk::binding(4, 0)]] ConstantBuffer<MetalRigidStepConstants>
    rigidCacheStep : register(b4);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalContactManifold>
    rigidCacheManifolds : register(u8);
[[vk::binding(10, 0)]] StructuredBuffer<MetalHandle>
    rigidCacheIds : register(t10);
[[vk::binding(18, 0)]] StructuredBuffer<uint>
    rigidCacheActivePairs : register(t18);
[[vk::binding(19, 0)]] StructuredBuffer<uint>
    rigidCacheActivePairCount : register(t19);
[[vk::binding(20, 0)]] StructuredBuffer<uint>
    rigidCacheSubstepIndex : register(t20);
[[vk::binding(27, 0)]] RWStructuredBuffer<MetalCachedContactPair>
    rigidCachePairs : register(u27);
[[vk::binding(28, 0)]] StructuredBuffer<uint64_t>
    rigidCacheEpochBase : register(t28);

[ForceInline]
static uint rigidCachePairSlot(uint first, uint second, uint count)
{
    uint low = min(first, second);
    uint high = max(first, second);
    return low * (2u * count - low - 1u) / 2u + high - low - 1u;
}

static void rigidCacheLoad(
    inout MetalContactManifold manifold,
    MetalRigidState bodyState,
    MetalHandle bodyId,
    MetalHandle colliderId,
    MetalCachedContactPair saved,
    uint64_t epoch,
    float timestep)
{
    if (saved.epoch + 1ul != epoch ||
        saved.body_index != bodyId.index ||
        saved.body_generation != bodyId.generation ||
        saved.collider_index != colliderId.index ||
        saved.collider_generation != colliderId.generation ||
        abs(saved.timestep - timestep) > 1.0e-7f)
        return;
    manifold.color = saved.color;
    uint used = 0u;
    for (uint point = 0u; point < manifold.count; ++point)
    {
        MetalContactRecord contact = manifold.contacts[point];
        float3 local = metalRotate(
            metalQuaternionConjugate(bodyState.orientation),
            metalLoad(contact.point) - metalLoad(bodyState.position));
        float nearest = 0.02f * 0.02f;
        uint match = 8u;
        for (uint previous = 0u; previous < min(saved.count, 8u); ++previous)
        {
            float3 previousNormal = metalRotate(bodyState.orientation,
                metalLoad(saved.contacts[previous].local_normal));
            if ((used & (1u << previous)) != 0u ||
                dot(metalLoad(contact.normal), previousNormal) < 0.99f)
                continue;
            float3 delta =
                local - metalLoad(saved.contacts[previous].local_point);
            float distanceValue = dot(delta, delta);
            if (distanceValue < nearest)
            {
                nearest = distanceValue;
                match = previous;
            }
        }
        if (match == 8u) continue;
        used |= 1u << match;
        manifold.cached = 1u;
        MetalCachedContact previous = saved.contacts[match];
        contact.accumulated_normal_impulse = previous.normal_impulse;
        float3 friction = metalLoad(previous.friction_impulse);
        float3 normal = metalLoad(contact.normal);
        contact.accumulated_friction_impulse =
            metalStore(friction - normal * dot(friction, normal));
        for (uint axis = 0u; axis < 3u; ++axis)
            contact.avbd_dual[axis] = previous.avbd_dual[axis];
        contact.avbd_anchor_a = previous.avbd_anchor_a;
        contact.avbd_anchor_b = previous.avbd_anchor_b;
        contact.avbd_cached = 1u;
        float3 tangent0 = normalize(cross(normal,
            abs(normal.x) < 0.57735f
                ? float3(1.0f, 0.0f, 0.0f)
                : float3(0.0f, 1.0f, 0.0f)));
        float3 tangent1 = cross(normal, tangent0);
        contact.avbd_dual[1].lambda = dot(friction, tangent0) / timestep;
        contact.avbd_dual[2].lambda = dot(friction, tangent1) / timestep;
        manifold.contacts[point] = contact;
    }
}

static MetalCachedContactPair rigidCacheSave(
    MetalContactManifold manifold,
    MetalRigidState bodyState,
    MetalRigidState colliderState,
    MetalHandle bodyId,
    MetalHandle colliderId,
    MetalCachedContactPair saved,
    uint64_t epoch,
    float timestep)
{
    saved.body_index = bodyId.index;
    saved.body_generation = bodyId.generation;
    saved.collider_index = colliderId.index;
    saved.collider_generation = colliderId.generation;
    saved.epoch = epoch;
    saved.timestep = timestep;
    saved.count = manifold.count;
    saved.face_patch = manifold.face_patch;
    saved.color = manifold.color;
    for (uint point = 0u; point < manifold.count; ++point)
    {
        MetalContactRecord contact = manifold.contacts[point];
        MetalCachedContact output = saved.contacts[point];
        output.local_point = metalStore(metalRotate(
            metalQuaternionConjugate(bodyState.orientation),
            metalLoad(contact.point) - metalLoad(bodyState.position)));
        output.collider_local_point = metalStore(metalRotate(
            metalQuaternionConjugate(colliderState.orientation),
            metalLoad(contact.point) - metalLoad(colliderState.position)));
        output.local_normal = metalStore(metalRotate(
            metalQuaternionConjugate(bodyState.orientation),
            metalLoad(contact.normal)));
        output.penetration = contact.penetration - dot(
            (metalLoad(bodyState.position) -
             metalLoad(colliderState.position)) -
                metalLoad(manifold.initial_relative_position),
            metalLoad(contact.normal));
        output.normal_impulse = contact.accumulated_normal_impulse;
        output.friction_impulse = contact.accumulated_friction_impulse;
        for (uint axis = 0u; axis < 3u; ++axis)
            output.avbd_dual[axis] = contact.avbd_dual[axis];
        output.avbd_anchor_a = contact.avbd_anchor_a;
        output.avbd_anchor_b = contact.avbd_anchor_b;
        saved.contacts[point] = output;
    }
    return saved;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_contact_match_cache(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint active = dispatchThreadId.x;
    if (active >= rigidCacheActivePairCount[0]) return;
    MetalRigidStepConstants step = rigidCacheStep;
    uint pair = rigidCacheActivePairs[active];
    uint body = pair / step.body_count;
    uint collider = pair % step.body_count;
    uint slot = rigidCachePairSlot(body, collider, step.body_count);
    MetalContactManifold manifold = rigidCacheManifolds[active];
    rigidCacheLoad(manifold, rigidCacheStates[body], rigidCacheIds[body],
        rigidCacheIds[collider], rigidCachePairs[slot],
        rigidCacheEpochBase[0] + uint64_t(rigidCacheSubstepIndex[0]),
        step.timestep);
    rigidCacheManifolds[active] = manifold;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_advance_substep(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint active = dispatchThreadId.x;
    if (active >= rigidCacheActivePairCount[0]) return;
    MetalRigidStepConstants step = rigidCacheStep;
    uint pair = rigidCacheActivePairs[active];
    uint body = pair / step.body_count;
    uint collider = pair % step.body_count;
    uint slot = rigidCachePairSlot(body, collider, step.body_count);
    rigidCachePairs[slot] = rigidCacheSave(
        rigidCacheManifolds[active], rigidCacheStates[body],
        rigidCacheStates[collider], rigidCacheIds[body],
        rigidCacheIds[collider], rigidCachePairs[slot],
        rigidCacheEpochBase[0] + uint64_t(rigidCacheSubstepIndex[0]),
        step.timestep);
}
