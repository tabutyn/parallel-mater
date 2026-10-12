// SPDX-License-Identifier: MIT
// Shared rope endpoint and cloth vertex coupling kernels.

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
    if (cudaBlockIdx().x != 0u || cudaThreadIdx().x != 0u) return;
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
    if (cudaBlockIdx().x != 0u || cudaThreadIdx().x != 0u) return;
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
