// SPDX-License-Identifier: MIT
// Legacy Metal argument-table adapter for shared rigid broad-phase behavior.

import metal_common;

struct MetalWorldAabb
{
    MetalPackedVec3 minimum;
    MetalPackedVec3 maximum;
};

struct MetalRigidCollisionGroup
{
    uint root;
    uint eligible;
};

[[vk::binding(0, 0)]] StructuredBuffer<MetalRigidState>
    rigidBroadphaseStates : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<MetalRigidParameters>
    rigidBroadphaseParameters : register(t1);
[[vk::binding(4, 0)]] ConstantBuffer<MetalRigidStepConstants>
    rigidBroadphaseStep : register(b4);
[[vk::binding(7, 0)]] StructuredBuffer<MetalTriangleMeshInfo>
    rigidBroadphaseMeshes : register(t7);
[[vk::binding(9, 0)]] StructuredBuffer<MetalRigidConstraintResource>
    rigidBroadphaseConstraints : register(t9);
[[vk::binding(14, 0)]] StructuredBuffer<MetalRigidState>
    rigidBroadphasePreviousStates : register(t14);
[[vk::binding(16, 0)]] RWStructuredBuffer<MetalWorldAabb>
    rigidBroadphaseWorldBounds : register(u16);
[[vk::binding(17, 0)]] RWStructuredBuffer<uint>
    rigidBroadphaseActiveFlags : register(u17);
[[vk::binding(18, 0)]] RWStructuredBuffer<uint>
    rigidBroadphaseActivePairs : register(u18);
[[vk::binding(19, 0)]] RWStructuredBuffer<uint>
    rigidBroadphaseActivePairCount : register(u19);
[[vk::binding(21, 0)]] RWStructuredBuffer<uint>
    rigidBroadphaseRowOffsets : register(u21);
[[vk::binding(25, 0)]] StructuredBuffer<MetalRigidCollisionGroup>
    rigidBroadphaseCompounds : register(t25);
// quiet_substeps, asleep
[[vk::binding(30, 0)]] RWStructuredBuffer<uint2>
    rigidBroadphaseSleepStates : register(u30);

[ForceInline]
static bool rigidBroadphaseBoundsOverlap(
    float3 minimumA, float3 maximumA, float3 minimumB, float3 maximumB)
{
    return all(minimumA <= maximumB) && all(maximumA >= minimumB);
}

static void rigidBroadphaseTransformedBounds(
    MetalRigidState state,
    float3 localMinimum,
    float3 localMaximum,
    float margin,
    out float3 minimum,
    out float3 maximum)
{
    float3 localCenter = (localMinimum + localMaximum) * 0.5f;
    float3 localHalf = (localMaximum - localMinimum) * 0.5f;
    float3 axisX = metalRotate(
        state.orientation, float3(1.0f, 0.0f, 0.0f));
    float3 axisY = metalRotate(
        state.orientation, float3(0.0f, 1.0f, 0.0f));
    float3 axisZ = metalRotate(
        state.orientation, float3(0.0f, 0.0f, 1.0f));
    float3 worldCenter = metalLoad(state.position) +
        axisX * localCenter.x + axisY * localCenter.y +
        axisZ * localCenter.z;
    float3 worldHalf = abs(axisX) * localHalf.x +
        abs(axisY) * localHalf.y + abs(axisZ) * localHalf.z + margin;
    minimum = worldCenter - worldHalf;
    maximum = worldCenter + worldHalf;
}

static void rigidBroadphaseTransformedMotionBounds(
    MetalRigidState previousState,
    MetalRigidState state,
    float3 localMinimum,
    float3 localMaximum,
    float margin,
    out float3 minimum,
    out float3 maximum)
{
    rigidBroadphaseTransformedBounds(
        state, localMinimum, localMaximum, margin, minimum, maximum);
    float3 previousMinimum;
    float3 previousMaximum;
    rigidBroadphaseTransformedBounds(previousState, localMinimum,
        localMaximum, margin, previousMinimum, previousMaximum);
    minimum = min(minimum, previousMinimum);
    maximum = max(maximum, previousMaximum);
}

[ForceInline]
static float rigidBroadphaseRotationalMotionBound(
    MetalRigidState previousState,
    MetalRigidState state,
    MetalTriangleMeshInfo mesh)
{
    float orientationDot = clamp(abs(
        previousState.orientation.x * state.orientation.x +
        previousState.orientation.y * state.orientation.y +
        previousState.orientation.z * state.orientation.z +
        previousState.orientation.w * state.orientation.w), 0.0f, 1.0f);
    float sineHalfAngle =
        sqrt(max(0.0f, 1.0f - orientationDot * orientationDot));
    float3 maximumAbsolute = max(
        abs(metalLoad(mesh.minimum)), abs(metalLoad(mesh.maximum)));
    return 2.0f * length(maximumAbsolute) * sineHalfAngle;
}

[ForceInline]
static bool rigidBroadphaseRequiresSweptContact(
    MetalRigidState previousState,
    MetalRigidState state,
    MetalTriangleMeshInfo mesh,
    float threshold)
{
    return length(metalLoad(state.position) -
        metalLoad(previousState.position)) +
        rigidBroadphaseRotationalMotionBound(previousState, state, mesh) >
        threshold;
}

static bool rigidBroadphaseBoundingSpheresMayContact(
    MetalRigidState previousBodyState,
    MetalRigidState bodyState,
    MetalTriangleMeshInfo bodyMesh,
    MetalRigidState previousColliderState,
    MetalRigidState colliderState,
    MetalTriangleMeshInfo colliderMesh,
    float margin)
{
    float3 previousRelative =
        metalLoad(previousBodyState.position) +
        metalRotate(previousBodyState.orientation,
            metalLoad(bodyMesh.bounding_center)) -
        metalLoad(previousColliderState.position) -
        metalRotate(previousColliderState.orientation,
            metalLoad(colliderMesh.bounding_center));
    float3 currentRelative =
        metalLoad(bodyState.position) +
        metalRotate(bodyState.orientation,
            metalLoad(bodyMesh.bounding_center)) -
        metalLoad(colliderState.position) -
        metalRotate(colliderState.orientation,
            metalLoad(colliderMesh.bounding_center));
    float3 movement = currentRelative - previousRelative;
    float squaredMovement = dot(movement, movement);
    float time = squaredMovement > 1.0e-12f
        ? clamp(-dot(previousRelative, movement) / squaredMovement,
            0.0f, 1.0f)
        : 0.0f;
    float3 nearest = previousRelative + movement * time;
    float radius = bodyMesh.radius + colliderMesh.radius + margin +
        1.0e-5f +
        rigidBroadphaseRotationalMotionBound(
            previousBodyState, bodyState, bodyMesh) +
        rigidBroadphaseRotationalMotionBound(
            previousColliderState, colliderState, colliderMesh);
    return dot(nearest, nearest) <= radius * radius;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_world_bounds(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint body = dispatchThreadId.x;
    MetalRigidStepConstants step = rigidBroadphaseStep;
    if (body >= step.body_count) return;
    MetalRigidParameters parameters = rigidBroadphaseParameters[body];
    MetalTriangleMeshInfo mesh =
        rigidBroadphaseMeshes[parameters.mesh_index];
    MetalRigidState state = rigidBroadphaseStates[body];
    MetalRigidState previousState = rigidBroadphasePreviousStates[body];
    float3 minimum;
    float3 maximum;
    if (rigidBroadphaseRequiresSweptContact(
        previousState, state, mesh, parameters.collision_margin))
        rigidBroadphaseTransformedMotionBounds(previousState, state,
            metalLoad(mesh.minimum), metalLoad(mesh.maximum), 0.0f,
            minimum, maximum);
    else
        rigidBroadphaseTransformedBounds(state, metalLoad(mesh.minimum),
            metalLoad(mesh.maximum), 0.0f, minimum, maximum);
    MetalWorldAabb bounds;
    bounds.minimum = metalStore(minimum);
    bounds.maximum = metalStore(maximum);
    rigidBroadphaseWorldBounds[body] = bounds;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_pair_filter(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint pair = dispatchThreadId.x;
    MetalRigidStepConstants step = rigidBroadphaseStep;
    uint pairCount = step.body_count * step.body_count;
    if (pair >= pairCount) return;
    uint body = pair / step.body_count;
    uint collider = pair % step.body_count;
    MetalRigidParameters bodyParameters = rigidBroadphaseParameters[body];
    MetalRigidParameters colliderParameters =
        rigidBroadphaseParameters[collider];
    bool active = bodyParameters.inverse_mass > 0.0f && body != collider;
    if (active && colliderParameters.inverse_mass > 0.0f &&
        collider < body)
        active = false;
    if (active && step.rigid_sleeping != 0u)
    {
        bool bodyAsleep = rigidBroadphaseSleepStates[body].y != 0u;
        bool colliderDynamic = colliderParameters.inverse_mass > 0.0f;
        bool colliderStatic = colliderParameters.motion == 0u;
        bool colliderAsleep = colliderDynamic &&
            rigidBroadphaseSleepStates[collider].y != 0u;
        if (bodyAsleep && (colliderStatic || colliderAsleep)) active = false;
    }
    MetalRigidCollisionGroup bodyGroup = rigidBroadphaseCompounds[body];
    MetalRigidCollisionGroup colliderGroup =
        rigidBroadphaseCompounds[collider];
    if (active && bodyGroup.eligible != 0u &&
        colliderGroup.eligible != 0u && bodyGroup.root == colliderGroup.root)
        active = false;
    if (active)
    {
        for (uint constraintIndex = 0u;
            constraintIndex < step.constraint_capacity; ++constraintIndex)
        {
            MetalRigidConstraintResource constraint =
                rigidBroadphaseConstraints[constraintIndex];
            if (constraint.alive != 0u && constraint.enabled != 0u &&
                constraint.broken == 0u &&
                constraint.disable_collisions != 0u &&
                ((constraint.body_a == body &&
                  constraint.body_b == collider) ||
                 (constraint.body_a == collider &&
                  constraint.body_b == body)))
            {
                active = false;
                break;
            }
        }
    }
    float margin = bodyParameters.collision_margin +
        colliderParameters.collision_margin;
    if (active)
    {
        MetalWorldAabb bodyBounds = rigidBroadphaseWorldBounds[body];
        MetalWorldAabb colliderBounds =
            rigidBroadphaseWorldBounds[collider];
        active = rigidBroadphaseBoundsOverlap(
            metalLoad(bodyBounds.minimum) - margin,
            metalLoad(bodyBounds.maximum) + margin,
            metalLoad(colliderBounds.minimum),
            metalLoad(colliderBounds.maximum));
    }
    if (active)
    {
        active = rigidBroadphaseBoundingSpheresMayContact(
            rigidBroadphasePreviousStates[body],
            rigidBroadphaseStates[body],
            rigidBroadphaseMeshes[bodyParameters.mesh_index],
            rigidBroadphasePreviousStates[collider],
            rigidBroadphaseStates[collider],
            rigidBroadphaseMeshes[colliderParameters.mesh_index], margin);
    }
    rigidBroadphaseActiveFlags[pair] = active ? 1u : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_pair_count_rows(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint body = dispatchThreadId.x;
    MetalRigidStepConstants step = rigidBroadphaseStep;
    if (body >= step.body_count) return;
    uint count = 0u;
    uint rowBegin = body * step.body_count;
    for (uint collider = 0u; collider < step.body_count; ++collider)
        count += rigidBroadphaseActiveFlags[rowBegin + collider] != 0u
            ? 1u : 0u;
    rigidBroadphaseRowOffsets[body + 1u] = count;
    if (body == 0u) rigidBroadphaseRowOffsets[0] = 0u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_rigid_pair_prefix_rows(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u) return;
    MetalRigidStepConstants step = rigidBroadphaseStep;
    uint cursor = 0u;
    for (uint body = 0u; body < step.body_count; ++body)
    {
        uint count = rigidBroadphaseRowOffsets[body + 1u];
        rigidBroadphaseRowOffsets[body] = cursor;
        cursor += count;
    }
    rigidBroadphaseRowOffsets[step.body_count] = cursor;
    rigidBroadphaseActivePairCount[0] = cursor;
    rigidBroadphaseActivePairCount[1] = (cursor + 63u) / 64u;
    rigidBroadphaseActivePairCount[2] = 1u;
    rigidBroadphaseActivePairCount[3] = 1u;
    rigidBroadphaseActivePairCount[4] = cursor;
    rigidBroadphaseActivePairCount[5] = 1u;
    rigidBroadphaseActivePairCount[6] = 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_pair_scatter_rows(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint body = dispatchThreadId.x;
    MetalRigidStepConstants step = rigidBroadphaseStep;
    if (body >= step.body_count) return;
    uint cursor = rigidBroadphaseRowOffsets[body];
    uint rowBegin = body * step.body_count;
    for (uint collider = 0u; collider < step.body_count; ++collider)
    {
        uint pair = rowBegin + collider;
        if (rigidBroadphaseActiveFlags[pair] == 0u) continue;
        rigidBroadphaseActivePairs[cursor] = pair;
        rigidBroadphaseActiveFlags[pair] = cursor + 1u;
        ++cursor;
    }
}
