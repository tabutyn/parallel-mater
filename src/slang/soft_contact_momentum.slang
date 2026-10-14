// SPDX-License-Identifier: MIT
// Production soft-body contact momentum maintenance for Metal.

import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    softMomentumPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    softMomentumVelocities : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<float>
    softMomentumInverseMasses : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<MetalRigidState>
    softMomentumRigidStates : register(u3);
[[vk::binding(4, 0)]] StructuredBuffer<MetalRigidParameters>
    softMomentumRigidParameters : register(t4);
[[vk::binding(8, 0)]] ConstantBuffer<MetalParticleRigidConstants>
    softMomentumConstants : register(b8);
[[vk::binding(9, 0)]] RWStructuredBuffer<MetalPackedVec3>
    softMomentumDiagnostics : register(u9);
[[vk::binding(18, 0)]] StructuredBuffer<MetalParticleRigidContact>
    softMomentumContacts : register(t18);
[[vk::binding(19, 0)]] StructuredBuffer<uint>
    softMomentumBodyContactCounts : register(t19);
[[vk::binding(24, 0)]] RWStructuredBuffer<MetalPackedVec3>
    softMomentumParticleLinearImpulses : register(u24);
[[vk::binding(25, 0)]] RWStructuredBuffer<MetalPackedVec3>
    softMomentumParticleAngularImpulses : register(u25);
[[vk::binding(27, 0)]] RWStructuredBuffer<MetalSoftContactAccumulator>
    softMomentumAccumulators : register(u27);
[[vk::binding(28, 0)]] RWStructuredBuffer<MetalSoftContactState>
    softMomentumState : register(u28);

groupshared float3 softMomentumActual[128];
groupshared float3 softMomentumContact[128];
groupshared float3 softMomentumCorrection;

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_contact_accumulators_clear(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalSoftContactAccumulator accumulator;
    accumulator.momentum_delta = metalStore(float3(0.0f));
    accumulator.friction_delta = metalStore(float3(0.0f));
    accumulator.arm = metalStore(float3(0.0f));
    accumulator.normal_delta = 0.0f;
    softMomentumAccumulators[node] = accumulator;
    if (node == 0u)
    {
        MetalSoftContactState state = softMomentumState[0];
        state.dynamic_contact_flag = 0u;
        softMomentumState[0] = state;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
public void pm_soft_body_measure_momentum(uint groupIndex : SV_GroupIndex)
{
    uint lane = groupIndex;
    MetalParticleRigidConstants constants = softMomentumConstants;
    float3 local = float3(0.0f);
    for (uint node = lane; node < constants.particle_count; node += 128u)
    {
        float inverse = softMomentumInverseMasses[node];
        if (inverse > 0.0f)
            local += metalLoad(softMomentumVelocities[node]) / inverse;
    }
    softMomentumActual[lane] = local;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 64u; stride != 0u; stride >>= 1u)
    {
        if (lane < stride)
            softMomentumActual[lane] += softMomentumActual[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
    {
        MetalSoftContactState state = softMomentumState[0];
        state.predicted_momentum = metalStore(softMomentumActual[0]);
        softMomentumState[0] = state;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
public void pm_soft_body_restore_momentum(uint groupIndex : SV_GroupIndex)
{
    if (softMomentumState[0].dynamic_contact_flag == 0u) return;
    uint lane = groupIndex;
    MetalParticleRigidConstants constants = softMomentumConstants;
    float3 actual = float3(0.0f);
    float3 contact = float3(0.0f);
    for (uint node = lane; node < constants.particle_count; node += 128u)
    {
        float inverse = softMomentumInverseMasses[node];
        if (inverse <= 0.0f) continue;
        actual += metalLoad(softMomentumVelocities[node]) / inverse;
        contact += metalLoad(softMomentumAccumulators[node].momentum_delta);
    }
    softMomentumActual[lane] = actual;
    softMomentumContact[lane] = contact;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 64u; stride != 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            softMomentumActual[lane] += softMomentumActual[lane + stride];
            softMomentumContact[lane] += softMomentumContact[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
    {
        float3 target = metalLoad(softMomentumState[0].predicted_momentum) +
            softMomentumContact[0];
        softMomentumCorrection = (target - softMomentumActual[0]) /
            max(constants.movable_mass, 1.0e-12f);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint node = lane; node < constants.particle_count; node += 128u)
    {
        if (softMomentumInverseMasses[node] > 0.0f)
            softMomentumVelocities[node] = metalStore(metalLimit(
                metalLoad(softMomentumVelocities[node]) +
                    softMomentumCorrection,
                constants.maximum_reaction_speed));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_contact_friction(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalParticleRigidConstants constants = softMomentumConstants;
    if (particle >= constants.particle_count) return;
    softMomentumParticleLinearImpulses[particle] =
        metalStore(float3(0.0f));
    softMomentumParticleAngularImpulses[particle] =
        metalStore(float3(0.0f));
    if (constants.fluid.generation != 0u ||
        constants.solid_contacts == 0u ||
        softMomentumInverseMasses[particle] <= 0.0f ||
        constants.friction <= 0.0f)
        return;
    MetalParticleRigidContact contact = softMomentumContacts[particle];
    if (contact.body == 0xffffffffu ||
        contact.body >= constants.rigid_count)
        return;
    uint supported = 0u;
    for (uint body = 0u; body < constants.rigid_count; ++body)
        supported += softMomentumBodyContactCounts[body];
    if (supported == 0u) return;
    float3 normal = normalize(metalLoad(contact.normal));
    float3 original = metalLoad(softMomentumVelocities[particle]);
    float3 tangent = original - normal * dot(original, normal);
    float inverse = softMomentumInverseMasses[particle];
    float3 force = metalLoad(softMomentumDiagnostics[particle]);
    float impulseAcceleration = dot(force, force) > 1.0e-10f
        ? length(force) * inverse : 0.0f;
    float supportedAcceleration =
        abs(dot(metalLoad(constants.gravity), normal)) *
        constants.movable_mass * inverse / float(supported);
    MetalSoftContactAccumulator accumulator =
        softMomentumAccumulators[particle];
    float constraintDelta = accumulator.normal_delta /
        max(constants.timestep, 1.0e-12f);
    MetalRigidParameters body =
        softMomentumRigidParameters[contact.body];
    float friction = body.motion == 2u
        ? sqrt(max(constants.friction * body.friction, 0.0f))
        : constants.friction;
    float maximumDelta = friction * max(
        max(impulseAcceleration, supportedAcceleration) * constants.timestep,
        constraintDelta);
    float3 accumulated = metalLoad(accumulator.friction_delta);
    accumulated -= normal * dot(accumulated, normal);
    float3 nextAccumulated = metalLimit(
        accumulated - tangent, maximumDelta);
    float3 change = nextAccumulated - accumulated;
    float3 velocity = metalLimit(
        original + change, constants.maximum_reaction_speed);
    softMomentumVelocities[particle] = metalStore(velocity);
    softMomentumPositions[particle] = metalStore(
        metalLoad(softMomentumPositions[particle]) +
        (velocity - original) * constants.timestep);
    accumulator.friction_delta = metalStore(nextAccumulated);
    float3 nodeImpulse = change / inverse;
    accumulator.momentum_delta = metalStore(
        metalLoad(accumulator.momentum_delta) + nodeImpulse);
    softMomentumAccumulators[particle] = accumulator;
    softMomentumDiagnostics[particle] = metalStore(
        metalLoad(softMomentumDiagnostics[particle]) +
        nodeImpulse / max(constants.timestep, 1.0e-12f));
    float3 reaction = -nodeImpulse;
    softMomentumParticleLinearImpulses[particle] = metalStore(reaction);
    softMomentumParticleAngularImpulses[particle] = metalStore(
        cross(metalLoad(accumulator.arm), reaction));
}

[shader("compute")]
[numthreads(128, 1, 1)]
public void pm_soft_contact_finish(uint groupIndex : SV_GroupIndex)
{
    MetalParticleRigidConstants constants = softMomentumConstants;
    for (uint body = groupIndex; body < constants.rigid_count; body += 128u)
    {
        MetalRigidParameters parameters =
            softMomentumRigidParameters[body];
        if (parameters.motion != 2u) continue;
        float3 linear = float3(0.0f);
        float3 angular = float3(0.0f);
        for (uint particle = 0u; particle < constants.particle_count;
             ++particle)
        {
            if (softMomentumContacts[particle].body != body) continue;
            linear += metalLoad(
                softMomentumParticleLinearImpulses[particle]);
            angular += metalLoad(
                softMomentumParticleAngularImpulses[particle]);
        }
        MetalRigidState state = softMomentumRigidStates[body];
        state.linear_velocity = metalStore(metalLimit(
            metalLoad(state.linear_velocity) +
                linear * parameters.inverse_mass,
            parameters.maximum_linear_speed));
        state.angular_velocity = metalStore(metalLimit(
            metalLoad(state.angular_velocity) +
                metalInverseInertiaMul(parameters, state, angular),
            parameters.maximum_angular_speed));
        softMomentumRigidStates[body] = state;
    }
}
