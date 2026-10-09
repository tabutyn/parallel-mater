// SPDX-License-Identifier: MIT
// Slang mirror of src/rope_smoke.cuh.

import cuda_common;
import geometry_rope;
import geometry_smoke;

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_rope_wind(
    uniform RopeData rope, uniform SmokeOptions smoke,
    uniform Vec3* smoke_positions, uniform Vec3* smoke_velocities,
    uniform uint64_t* keys, uniform uint* indices,
    uniform SmokeGridField grid, uniform float drag,
    uniform float maximum_acceleration, uniform float dt,
    uniform int first, uniform int last)
{
    uint node = pmThreadIndex();
    if (node >= rope.count || rope_anchor_body(rope, node, first, last) >= 0 ||
        rope_soft_anchor(rope, node)) return;
    rope.velocities[node] = clamp_length(add(rope.velocities[node],
        smoke_wind_delta(rope.positions[node], rope.velocities[node],
            smoke_positions, smoke_velocities, keys, indices, smoke, grid,
            drag, maximum_acceleration, dt)), rope.options.maximum_speed);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_rope_contact(
    uniform Vec3* positions, uniform Vec3* previous_positions,
    uniform Vec3* velocities, uniform float* ages, uniform uint count,
    uniform float lifetime, uniform RopeData rope, uniform Vec3* minimum,
    uniform Vec3* maximum, uniform float clearance)
{
    uint particle = pmThreadIndex();
    if (particle >= count || ages[particle] >= lifetime) return;
    Vec3 point = positions[particle];
    Vec3 before = previous_positions[particle];
    Vec3 low = *minimum;
    Vec3 high = *maximum;
    if (max(point.x, before.x) < low.x - clearance ||
        min(point.x, before.x) > high.x + clearance ||
        max(point.y, before.y) < low.y - clearance ||
        min(point.y, before.y) > high.y + clearance ||
        max(point.z, before.z) < low.z - clearance ||
        min(point.z, before.z) > high.z + clearance) return;
    float earliest = 2.0f;
    float nearestSquared = clearance * clearance;
    uint segment = rope.count;
    Vec3 smokeHit = {};
    Vec3 ropeHit = {};
    Vec3 path = subtract(point, before);
    float pathSquared = length_squared(path);
    for (uint index = 0u; index + 1u < rope.count; ++index)
    {
        Vec3 a = rope.positions[index];
        Vec3 b = rope.positions[index + 1u];
        if (max(point.x, before.x) < min(a.x, b.x) - clearance ||
            min(point.x, before.x) > max(a.x, b.x) + clearance ||
            max(point.y, before.y) < min(a.y, b.y) - clearance ||
            min(point.y, before.y) > max(a.y, b.y) + clearance ||
            max(point.z, before.z) < min(a.z, b.z) - clearance ||
            min(point.z, before.z) > max(a.z, b.z) + clearance) continue;
        Vec3 onPath = {};
        Vec3 onRope = {};
        closest_segments(before, point, a, b, onPath, onRope);
        float distanceSquared = length_squared(subtract(onPath, onRope));
        if (distanceSquared >= clearance * clearance) continue;
        float fraction = pathSquared > 1.0e-12f
            ? clamp_scalar(dot(subtract(onPath, before), path) /
                pathSquared, 0.0f, 1.0f)
            : 0.0f;
        if (fraction > earliest + 1.0e-6f ||
            (abs(fraction - earliest) <= 1.0e-6f &&
             distanceSquared >= nearestSquared)) continue;
        earliest = fraction;
        nearestSquared = distanceSquared;
        segment = index;
        smokeHit = onPath;
        ropeHit = onRope;
    }
    if (segment == rope.count) return;
    Vec3 a = rope.positions[segment];
    Vec3 b = rope.positions[segment + 1u];
    Vec3 axis = subtract(b, a);
    float fraction = clamp_scalar(dot(subtract(ropeHit, a), axis) /
        max(length_squared(axis), 1.0e-12f), 0.0f, 1.0f);
    Vec3 ropeVelocity = add(
        multiply(rope.velocities[segment], 1.0f - fraction),
        multiply(rope.velocities[segment + 1u], fraction));
    Vec3 normal = subtract(smokeHit, ropeHit);
    if (length_squared(normal) < 1.0e-10f)
        normal = subtract(before, ropeHit);
    if (length_squared(normal) < 1.0e-10f)
    {
        Vec3 relativeVelocity = subtract(velocities[particle], ropeVelocity);
        normal = subtract(multiply(relativeVelocity, -1.0f),
            multiply(axis, -dot(relativeVelocity, axis) /
                max(length_squared(axis), 1.0e-12f)));
    }
    normal = normalized_or(normal, {0.0f, 1.0f, 0.0f});
    positions[particle] = add(ropeHit, multiply(normal, clearance));
    Vec3 relative = subtract(velocities[particle], ropeVelocity);
    relative = subtract(relative,
        multiply(normal, min(0.0f, dot(relative, normal))));
    velocities[particle] = add(ropeVelocity, relative);
}
