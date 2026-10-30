#include "dispatch_entry.slang"
#include "dispatch_shape.slangh"
// SPDX-License-Identifier: MIT
// Shared cloth/rope coupling kernels.

import types;
import math;
import device_intrinsics;
import graphics_resources;

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void rope_cloth_sample_anchor(
    uniform PMBuffer<Vec3> soft_anchor_positions,
    uniform PMBuffer<Vec3> soft_anchor_velocities,
    uniform PMBuffer<Vec3> soft_anchor_impulses,
    uniform PMBuffer<float> soft_anchor_inverse_masses, uniform uint end,
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> inverse_masses, uniform uint vertex,
    uniform float effective_mass PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u || pmGroupThreadIndex().x != 0u) return;
    soft_anchor_positions[end] = positions[vertex];
    soft_anchor_velocities[end] = velocities[vertex];
    soft_anchor_impulses[end] = {};
    soft_anchor_inverse_masses[end] =
        min(inverse_masses[vertex], 1.0f / effective_mass);
}

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void rope_cloth_apply_anchor(
    uniform PMBuffer<Vec3> rope_positions, uniform PMBuffer<Vec3> rope_velocities,
    uniform uint rope_count, uniform PMBuffer<Vec3> soft_anchor_positions,
    uniform PMBuffer<Vec3> soft_anchor_velocities,
    uniform PMBuffer<Vec3> soft_anchor_impulses,
    uniform PMBuffer<Vec3> rope_constraint_forces, uniform uint end,
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities, uniform PMBuffer<Vec3> forces,
    uniform PMBuffer<float> inverse_masses, uniform uint vertex,
    uniform float maximum_acceleration, uniform float constraint_force_scale,
    uniform float dt,
    uniform float frame_inverse_dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u || pmGroupThreadIndex().x != 0u) return;
    float inverseMass = inverse_masses[vertex];
    if (inverseMass <= 0.0f) return;
    Vec3 correction = clamp_length(
        subtract(soft_anchor_positions[end], positions[vertex]),
        maximum_acceleration * dt * dt);
    Vec3 deltaVelocity = multiply(correction, 1.0f / dt);
    velocities[vertex] = add(velocities[vertex], deltaVelocity);
    positions[vertex] = add(positions[vertex], correction);
    uint node = end != 0u ? rope_count - 1u : 0u;
    rope_positions[node] = positions[vertex];
    soft_anchor_positions[end] = positions[vertex];
    rope_velocities[node] = velocities[vertex];
    soft_anchor_velocities[end] = velocities[vertex];
}

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void rope_cloth_accumulate_reaction(
    uniform PMBuffer<Vec3> previous_velocities,
    uniform PMBuffer<Vec3> current_velocities,
    uniform PMBuffer<float> inverse_masses, uniform uint rope_count,
    uniform uint attached_end_mask, uniform uint attached_end_count,
    uniform Vec3 gravity, uniform float dt,
    uniform PMBuffer<Vec3> cloth_forces,
    uniform uint cloth_vertex PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u || pmGroupThreadIndex().x != 0u ||
        attached_end_count == 0u || dt <= 0.0f) return;
    Vec3 externalImpulse = {};
    for (uint node = 0u; node < rope_count; ++node)
    {
        bool attached = (node == 0u && (attached_end_mask & 1u) != 0u) ||
            (node + 1u == rope_count && (attached_end_mask & 2u) != 0u);
        float inverseMass = inverse_masses[node];
        if (attached || inverseMass <= 0.0f) continue;
        float mass = 1.0f / inverseMass;
        externalImpulse = add(externalImpulse, subtract(
            multiply(subtract(current_velocities[node],
                previous_velocities[node]), mass),
            multiply(gravity, mass * dt)));
    }
    cloth_forces[cloth_vertex] = add(cloth_forces[cloth_vertex],
        multiply(externalImpulse,
            -1.0f / (dt * float(attached_end_count))));
}
