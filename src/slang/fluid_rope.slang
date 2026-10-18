#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Shared fluid/rope coupling kernels.

import types;
import math;
import device_intrinsics;
import graphics_resources;

public void atomic_add(PMBuffer<Vec3> destination, Vec3 value)
{
    pmAtomicAddVec3(destination, value);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fluid_rope_contacts(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<uint> particle_count, uniform float particle_mass,
    uniform PMBuffer<Vec3> rope_positions, uniform PMBuffer<Vec3> rope_velocities,
    uniform uint rope_count, uniform float distance, uniform float friction,
    uniform PMBuffer<Vec3> node_impulses, uniform PMBuffer<uint> contact_count,
    uniform PMBuffer<float> maximum_penetration, uniform bool transfer_impulse PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
    pmAtomicMaxPositiveFloat(maximum_penetration, penetration);
    Vec3 reaction = multiply(velocityChange, -particle_mass);
    atomic_add(node_impulses + segment,
        multiply(reaction, 1.0f - fraction));
    atomic_add(node_impulses + segment + 1u,
        multiply(reaction, fraction));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fluid_rope_apply(
    uniform PMBuffer<Vec3> rope_velocities, uniform PMBuffer<Vec3> fluid_contact_forces,
    uniform uint rope_count, uniform float rope_mass,
    uniform float rope_maximum_speed, uniform bool first_attached,
    uniform bool last_attached,
    uniform PMBuffer<Vec3> node_impulses, uniform float maximum_acceleration,
    uniform float dt, uniform float frame_inverse_dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
