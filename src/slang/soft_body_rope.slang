#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Shared soft-body and rope endpoint coupling kernels. Contact traversal is
// shared with rope.slang.

import types;
import math;
import device_intrinsics;
import graphics_resources;

groupshared float ropeSoftSums[256];

[shader("compute")]
[numthreads(1, 1, 1)]
void rope_soft_sample_anchor(
    uniform PMBuffer<Vec3> soft_anchor_positions,
    uniform PMBuffer<Vec3> soft_anchor_velocities,
    uniform PMBuffer<Vec3> soft_anchor_impulses,
    uniform PMBuffer<float> soft_anchor_inverse_masses, uniform uint end,
    uniform PMBuffer<Vec3> surface,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<uint> indices,
    uniform PMBuffer<SoftBodySurfaceBinding> bindings, uniform uint triangle,
    uniform Vec3 weights, uniform Vec3 offset PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmGroupThreadIndex().x != 0u || pmBlockIndex().x != 0u) return;
    uint base = 3u * triangle;
    float w[3];
    w[0] = weights.x;
    w[1] = weights.y;
    w[2] = weights.z;
    Vec3 position = offset;
    Vec3 velocity = {};
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        uint vertex = indices[base + corner];
        position = add(position, multiply(surface[vertex], w[corner]));
        SoftBodySurfaceBinding binding = bindings[vertex];
        for (uint slot = 0u; slot < 4u; ++slot)
            velocity = add(velocity, multiply(
                velocities[binding.nodes[slot]],
                w[corner] * binding.weights[slot]));
    }
    soft_anchor_positions[end] = position;
    soft_anchor_velocities[end] = velocity;
    soft_anchor_impulses[end] = {};
    soft_anchor_inverse_masses[end] = 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rope_soft_scatter_anchor(
    uniform PMBuffer<Vec3> soft_anchor_positions,
    uniform PMBuffer<Vec3> soft_anchor_impulses, uniform uint end,
    uniform PMBuffer<Vec3> node_positions, uniform uint node_count,
    uniform float support_radius, uniform float contact_support_radius,
    uniform PMBuffer<uint> contact_count, uniform PMBuffer<uint> indices,
    uniform PMBuffer<SoftBodySurfaceBinding> bindings, uniform uint triangle,
    uniform Vec3 weights, uniform PMBuffer<Vec3> node_impulses PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u) return;
    uint thread = pmGroupThreadIndex().x;
    Vec3 anchor = soft_anchor_positions[end];
    support_radius = contact_support_radius +
        (support_radius - contact_support_radius) /
        (1.0f + float(*contact_count) / 2.0f);
    float radiusSquared = support_radius * support_radius;
    float localSum = 0.0f;
    for (uint node = thread; node < node_count; node += pmGroupSize().x)
    {
        float squared = length_squared(subtract(node_positions[node], anchor));
        if (squared < radiusSquared)
        {
            float t = 1.0f - squared / radiusSquared;
            localSum += t * t;
        }
    }
    ropeSoftSums[thread] = localSum;
    pmGroupBarrier();
    for (uint stride = pmGroupSize().x / 2u; stride != 0u; stride /= 2u)
    {
        if (thread < stride)
            ropeSoftSums[thread] += ropeSoftSums[thread + stride];
        pmGroupBarrier();
    }
    if (ropeSoftSums[0] > 1.0e-8f)
    {
        Vec3 impulse = multiply(
            soft_anchor_impulses[end], 1.0f / ropeSoftSums[0]);
        for (uint node = thread; node < node_count; node += pmGroupSize().x)
        {
            float squared = length_squared(
                subtract(node_positions[node], anchor));
            if (squared >= radiusSquared) continue;
            float t = 1.0f - squared / radiusSquared;
            node_impulses[node] = add(
                node_impulses[node], multiply(impulse, t * t));
        }
        return;
    }
    if (thread != 0u) return;
    float w[3];
    w[0] = weights.x;
    w[1] = weights.y;
    w[2] = weights.z;
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        SoftBodySurfaceBinding binding =
            bindings[indices[3u * triangle + corner]];
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            float weight = w[corner] * binding.weights[slot];
            if (weight > 0.0f)
                node_impulses[binding.nodes[slot]] = add(
                    node_impulses[binding.nodes[slot]], multiply(
                        soft_anchor_impulses[end], weight));
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rope_soft_apply(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> forces, uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<Vec3> impulses, uniform uint count,
    uniform float maximum_acceleration, uniform float maximum_speed,
    uniform float dt, uniform float frame_inverse_dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (node >= count || inverse_masses[node] == 0.0f) return;
    Vec3 impulse = clamp_length(
        multiply(impulses[node], inverse_masses[node]),
        maximum_acceleration * dt);
    velocities[node] = clamp_length(
        add(velocities[node], impulse), maximum_speed);
    positions[node] = add(positions[node], multiply(impulse, dt));
    forces[node] = add(forces[node], multiply(
        impulse, frame_inverse_dt / inverse_masses[node]));
}
