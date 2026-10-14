// SPDX-License-Identifier: MIT
// Production rigid integration and substep bookkeeping for Metal.

import metal_common;

struct MetalRigidSleepState
{
    uint quiet_substeps;
    uint asleep;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalRigidState>
    rigidIntegrateStates : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalRigidParameters>
    rigidIntegrateParameters : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    rigidIntegrateForces : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<MetalPackedVec3>
    rigidIntegrateTorques : register(u3);
[[vk::binding(4, 0)]] ConstantBuffer<MetalRigidStepConstants>
    rigidIntegrateStep : register(b4);
[[vk::binding(14, 0)]] RWStructuredBuffer<MetalRigidState>
    rigidIntegratePreviousStates : register(u14);
[[vk::binding(20, 0)]] RWStructuredBuffer<uint>
    rigidIntegrateSubstepIndex : register(u20);
[[vk::binding(30, 0)]] StructuredBuffer<MetalRigidSleepState>
    rigidIntegrateSleepStates : register(t30);

[ForceInline]
static MetalQuaternion rigidIntegrateConjugate(MetalQuaternion value)
{
    value.x = -value.x;
    value.y = -value.y;
    value.z = -value.z;
    return value;
}

[ForceInline]
static MetalQuaternion rigidIntegrateNormalize(MetalQuaternion value)
{
    float inverse = rsqrt(max(value.x * value.x + value.y * value.y +
        value.z * value.z + value.w * value.w, 1.0e-12f));
    value.x *= inverse;
    value.y *= inverse;
    value.z *= inverse;
    value.w *= inverse;
    return value;
}

[ForceInline]
static float3 rigidIntegrateInverseInertiaMul(
    MetalRigidParameters body, MetalRigidState state, float3 value)
{
    float3 local = metalRotate(
        rigidIntegrateConjugate(state.orientation), value);
    return metalRotate(state.orientation,
        local * metalLoad(body.inverse_inertia));
}

[ForceInline]
static float3 rigidIntegrateQuaternionDeltaVelocity(
    MetalQuaternion from, MetalQuaternion to, float timestep)
{
    MetalQuaternion delta = metalQuaternionMultiply(
        to, rigidIntegrateConjugate(from));
    if (delta.w < 0.0f)
    {
        delta.x = -delta.x;
        delta.y = -delta.y;
        delta.z = -delta.z;
        delta.w = -delta.w;
    }
    delta = rigidIntegrateNormalize(delta);
    float3 vector = float3(delta.x, delta.y, delta.z);
    float vectorSize = length(vector);
    if (timestep <= 0.0f) return float3(0.0f);
    if (vectorSize <= 1.0e-6f) return vector * (2.0f / timestep);
    float angle = 2.0f * atan2(
        vectorSize, clamp(delta.w, -1.0f, 1.0f));
    return vector * (angle / (vectorSize * timestep));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_noop(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint unused = dispatchThreadId.x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_integrate(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint bodyIndex = dispatchThreadId.x;
    MetalRigidStepConstants step = rigidIntegrateStep;
    if (bodyIndex >= step.body_count) return;
    MetalRigidParameters body = rigidIntegrateParameters[bodyIndex];
    MetalRigidState state = rigidIntegrateStates[bodyIndex];
    rigidIntegratePreviousStates[bodyIndex] = state;
    if (step.rigid_sleeping != 0u && body.motion == 2u &&
        rigidIntegrateSleepStates[bodyIndex].asleep != 0u)
    {
        state.linear_velocity = metalStore(float3(0.0f));
        state.angular_velocity = metalStore(float3(0.0f));
        rigidIntegrateStates[bodyIndex] = state;
        return;
    }
    if (body.motion == 0u)
    {
        state.linear_velocity = metalStore(float3(0.0f));
        state.angular_velocity = metalStore(float3(0.0f));
    }
    else if (body.motion == 1u)
    {
        if (body.has_kinematic_target != 0u)
        {
            uint remainingSubsteps = max(
                1u, step.substeps - rigidIntegrateSubstepIndex[0]);
            float fraction = 1.0f / float(remainingSubsteps);
            float3 previousPosition = metalLoad(state.position);
            MetalQuaternion previousOrientation = state.orientation;
            state.position = metalStore(previousPosition +
                (metalLoad(body.kinematic_target.position) -
                    previousPosition) * fraction);
            MetalQuaternion target = body.kinematic_target.orientation;
            float orientationDot =
                previousOrientation.x * target.x +
                previousOrientation.y * target.y +
                previousOrientation.z * target.z +
                previousOrientation.w * target.w;
            if (orientationDot < 0.0f)
            {
                target.x = -target.x;
                target.y = -target.y;
                target.z = -target.z;
                target.w = -target.w;
            }
            MetalQuaternion orientation;
            orientation.x = previousOrientation.x +
                (target.x - previousOrientation.x) * fraction;
            orientation.y = previousOrientation.y +
                (target.y - previousOrientation.y) * fraction;
            orientation.z = previousOrientation.z +
                (target.z - previousOrientation.z) * fraction;
            orientation.w = previousOrientation.w +
                (target.w - previousOrientation.w) * fraction;
            state.orientation = rigidIntegrateNormalize(orientation);
            state.linear_velocity = metalStore(
                (metalLoad(state.position) - previousPosition) /
                    step.timestep);
            state.angular_velocity = metalStore(
                rigidIntegrateQuaternionDeltaVelocity(previousOrientation,
                    state.orientation, step.timestep));
            if (remainingSubsteps == 1u) body.has_kinematic_target = 0u;
        }
        else
        {
            state.linear_velocity = metalStore(float3(0.0f));
            state.angular_velocity = metalStore(float3(0.0f));
        }
    }
    else if (body.motion == 2u)
    {
        float dt = step.timestep;
        float3 linearVelocity = metalLoad(state.linear_velocity);
        linearVelocity += (metalLoad(step.gravity) +
            metalLoad(rigidIntegrateForces[bodyIndex]) * body.inverse_mass) *
            dt;
        linearVelocity /= 1.0f + body.linear_damping * dt;
        linearVelocity = metalLimit(
            linearVelocity, body.maximum_linear_speed);
        state.linear_velocity = metalStore(linearVelocity);

        float3 angularVelocity = metalLoad(state.angular_velocity);
        angularVelocity += rigidIntegrateInverseInertiaMul(body, state,
            metalLoad(rigidIntegrateTorques[bodyIndex])) * dt;
        angularVelocity /= 1.0f + body.angular_damping * dt;
        angularVelocity = metalLimit(
            angularVelocity, body.maximum_angular_speed);
        state.angular_velocity = metalStore(angularVelocity);
        state.position = metalStore(
            metalLoad(state.position) + linearVelocity * dt);
        MetalQuaternion angular;
        angular.x = angularVelocity.x;
        angular.y = angularVelocity.y;
        angular.z = angularVelocity.z;
        angular.w = 0.0f;
        MetalQuaternion derivative =
            metalQuaternionMultiply(angular, state.orientation);
        MetalQuaternion orientation;
        orientation.x = state.orientation.x + 0.5f * derivative.x * dt;
        orientation.y = state.orientation.y + 0.5f * derivative.y * dt;
        orientation.z = state.orientation.z + 0.5f * derivative.z * dt;
        orientation.w = state.orientation.w + 0.5f * derivative.w * dt;
        state.orientation = rigidIntegrateNormalize(orientation);
    }
    rigidIntegrateStates[bodyIndex] = state;
    rigidIntegrateParameters[bodyIndex] = body;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_increment_substep(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x == 0u) ++rigidIntegrateSubstepIndex[0];
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_clear_accumulators(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint bodyIndex = dispatchThreadId.x;
    if (bodyIndex >= rigidIntegrateStep.body_count) return;
    rigidIntegrateForces[bodyIndex] = metalStore(float3(0.0f));
    rigidIntegrateTorques[bodyIndex] = metalStore(float3(0.0f));
}
