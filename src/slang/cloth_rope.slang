// SPDX-License-Identifier: MIT
// Shared cloth/rope coupling for the CUDA mirror and Metal backend.

#ifdef PM_TARGET_METAL
import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalRopeClothRopePositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalRopeClothRopeVelocities : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<float>
    metalRopeClothRopeInverseMasses : register(t2);
[[vk::binding(3, 0)]] StructuredBuffer<MetalPackedVec3>
    metalRopeClothClothPositions : register(t3);
[[vk::binding(4, 0)]] StructuredBuffer<MetalPackedVec3>
    metalRopeClothClothVelocities : register(t4);
[[vk::binding(5, 0)]] StructuredBuffer<float>
    metalRopeClothClothInverseMasses : register(t5);
[[vk::binding(6, 0)]] ConstantBuffer<MetalCouplingConstants>
    metalRopeClothConstants : register(b6);
[[vk::binding(12, 0)]] RWStructuredBuffer<MetalRopeAnchorState>
    metalRopeClothAnchors : register(u12);

[shader("compute")]
[numthreads(2, 1, 1)]
public void pm_rope_cloth_sample(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint end = dispatchThreadId.x;
    MetalCouplingConstants constants = metalRopeClothConstants;
    if (end >= 2u || constants.enabled == 0u) return;
    uint clothNode = end == 0u
        ? constants.first_vertex : constants.last_vertex;
    if (clothNode == 0xffffffffu || clothNode >= constants.count_b) return;
    MetalRopeAnchorState anchor;
    anchor.position = metalRopeClothClothPositions[clothNode];
    anchor.velocity = metalRopeClothClothVelocities[clothNode];
    anchor.impulse = metalStore(float3(0.0f));
    anchor.inverse_mass = min(
        metalRopeClothClothInverseMasses[clothNode],
        max(constants.stiffness, 0.0f));
    metalRopeClothAnchors[end] = anchor;
}

#else
import types;
import math;
import device_intrinsics;

[shader("compute")]
[numthreads(1, 1, 1)]
void rope_cloth_sample_anchor(
    uniform Vec3* soft_anchor_positions,
    uniform Vec3* soft_anchor_velocities,
    uniform Vec3* soft_anchor_impulses,
    uniform float* soft_anchor_inverse_masses, uniform uint end,
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* inverse_masses, uniform uint vertex,
    uniform float effective_mass)
{
    if (pmBlockIndex().x != 0u || pmGroupThreadIndex().x != 0u) return;
    soft_anchor_positions[end] = positions[vertex];
    soft_anchor_velocities[end] = velocities[vertex];
    soft_anchor_impulses[end] = {};
    soft_anchor_inverse_masses[end] =
        min(inverse_masses[vertex], 1.0f / effective_mass);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void rope_cloth_apply_anchor(
    uniform Vec3* rope_positions, uniform Vec3* rope_velocities,
    uniform uint rope_count, uniform Vec3* soft_anchor_positions,
    uniform Vec3* soft_anchor_velocities, uniform uint end,
    uniform Vec3* positions, uniform Vec3* velocities, uniform Vec3* forces,
    uniform float* inverse_masses, uniform uint vertex,
    uniform float maximum_acceleration, uniform float dt,
    uniform float frame_inverse_dt)
{
    if (pmBlockIndex().x != 0u || pmGroupThreadIndex().x != 0u) return;
    float inverseMass = inverse_masses[vertex];
    if (inverseMass <= 0.0f) return;
    Vec3 correction = clamp_length(
        subtract(soft_anchor_positions[end], positions[vertex]),
        maximum_acceleration * dt * dt);
    Vec3 deltaVelocity = multiply(correction, 1.0f / dt);
    velocities[vertex] = add(velocities[vertex], deltaVelocity);
    positions[vertex] = add(positions[vertex], correction);
    forces[vertex] = add(forces[vertex],
        multiply(deltaVelocity, frame_inverse_dt / inverseMass));
    uint node = end != 0u ? rope_count - 1u : 0u;
    rope_positions[node] = positions[vertex];
    soft_anchor_positions[end] = positions[vertex];
    rope_velocities[node] = velocities[vertex];
    soft_anchor_velocities[end] = velocities[vertex];
}
#endif
