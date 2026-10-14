// SPDX-License-Identifier: MIT
// Production rope prediction and attachment synchronization for Metal.

import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropePredictPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropePredictPrevious : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropePredictVelocities : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<float>
    ropePredictInverseMasses : register(t3);
[[vk::binding(5, 0)]] ConstantBuffer<MetalDeformableConstants>
    ropePredictConstants : register(b5);
[[vk::binding(6, 0)]] ConstantBuffer<MetalRopeAttachmentConstants>
    ropePredictAttachments : register(b6);
[[vk::binding(7, 0)]] StructuredBuffer<MetalHandle>
    ropePredictRigidIds : register(t7);
[[vk::binding(8, 0)]] StructuredBuffer<MetalRigidState>
    ropePredictRigidStates : register(t8);
[[vk::binding(17, 0)]] StructuredBuffer<MetalRopeAnchorState>
    ropePredictSoftAnchors : register(t17);
[[vk::binding(22, 0)]] StructuredBuffer<MetalRigidState>
    ropePredictOldRigidStates : register(t22);
[[vk::binding(23, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropePredictContactNormals : register(u23);
[[vk::binding(24, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropePredictContactNormals2 : register(u24);

[ForceInline]
static float3 ropePredictWorldPoint(
    MetalRigidState body, MetalPackedVec3 localPoint)
{
    return metalLoad(body.position) +
        metalRotate(body.orientation, metalLoad(localPoint));
}

[ForceInline]
static float3 ropePredictRigidPointVelocity(
    MetalRigidState body, MetalPackedVec3 localPoint)
{
    float3 arm = metalRotate(body.orientation, metalLoad(localPoint));
    return metalLoad(body.linear_velocity) +
        cross(metalLoad(body.angular_velocity), arm);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rope_predict(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalDeformableConstants constants = ropePredictConstants;
    if (index >= constants.count) return;
    MetalRopeAttachmentConstants attachments = ropePredictAttachments;
    uint firstBody = attachments.rigid_count;
    uint lastBody = attachments.rigid_count;
    for (uint body = 0u; body < attachments.rigid_count; ++body)
    {
        MetalHandle id = ropePredictRigidIds[body];
        if (id.index == attachments.first_body.index &&
            id.generation == attachments.first_body.generation)
            firstBody = body;
        if (id.index == attachments.last_body.index &&
            id.generation == attachments.last_body.generation)
            lastBody = body;
    }
    bool firstAttached = attachments.first_enabled != 0u &&
        firstBody < attachments.rigid_count;
    bool lastAttached = attachments.last_enabled != 0u &&
        lastBody < attachments.rigid_count;
    bool firstSoft = attachments.first_soft != 0u;
    bool lastSoft = attachments.last_soft != 0u;
    bool first = index == 0u;
    bool last = index + 1u == constants.count;

    if (first && firstAttached)
        ropePredictPositions[index] = metalStore(ropePredictWorldPoint(
            ropePredictOldRigidStates[firstBody], attachments.first_anchor));
    if (last && lastAttached)
        ropePredictPositions[index] = metalStore(ropePredictWorldPoint(
            ropePredictOldRigidStates[lastBody], attachments.last_anchor));
    if (first && firstSoft)
    {
        ropePredictPositions[index] = ropePredictSoftAnchors[0].position;
        ropePredictVelocities[index] = ropePredictSoftAnchors[0].velocity;
    }
    if (last && lastSoft)
    {
        ropePredictPositions[index] = ropePredictSoftAnchors[1].position;
        ropePredictVelocities[index] = ropePredictSoftAnchors[1].velocity;
    }
    ropePredictPrevious[index] = ropePredictPositions[index];
    ropePredictContactNormals[index] = metalStore(float3(0.0f));
    ropePredictContactNormals2[index] = metalStore(float3(0.0f));
    bool attached =
        (first && (firstAttached || firstSoft)) ||
        (last && (lastAttached || lastSoft));
    if (!attached && ropePredictInverseMasses[index] > 0.0f)
    {
        float3 velocity = metalLoad(ropePredictVelocities[index]) +
            metalLoad(constants.gravity) * constants.timestep;
        velocity *= exp(-constants.velocity_damping * constants.timestep);
        velocity = metalLimit(velocity, constants.maximum_speed);
        ropePredictVelocities[index] = metalStore(velocity);
        ropePredictPositions[index] = metalStore(
            metalLoad(ropePredictPositions[index]) +
            velocity * constants.timestep);
    }
    if (first && firstAttached)
    {
        ropePredictPositions[index] = metalStore(ropePredictWorldPoint(
            ropePredictRigidStates[firstBody], attachments.first_anchor));
        ropePredictVelocities[index] = metalStore(
            ropePredictRigidPointVelocity(
                ropePredictRigidStates[firstBody], attachments.first_anchor));
    }
    if (last && lastAttached)
    {
        ropePredictPositions[index] = metalStore(ropePredictWorldPoint(
            ropePredictRigidStates[lastBody], attachments.last_anchor));
        ropePredictVelocities[index] = metalStore(
            ropePredictRigidPointVelocity(
                ropePredictRigidStates[lastBody], attachments.last_anchor));
    }
    if (first && firstSoft)
    {
        ropePredictPositions[index] = ropePredictSoftAnchors[0].position;
        ropePredictVelocities[index] = ropePredictSoftAnchors[0].velocity;
    }
    if (last && lastSoft)
    {
        ropePredictPositions[index] = ropePredictSoftAnchors[1].position;
        ropePredictVelocities[index] = ropePredictSoftAnchors[1].velocity;
    }
}
