// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_cloth.cuh. The correction gather is independent
// of the rigid-mesh/paint query used by cloth_constrain_bodies.

import cuda_common;

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_apply_body_corrections(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* inverse_masses, uniform uint count,
    uniform ClothBodyCorrection* corrections, uniform uint body_count)
{
    uint vertex = pmThreadIndex();
    if (vertex >= count || inverse_masses[vertex] == 0.0f) return;
    Vec3 offset = {};
    Vec3 velocity = velocities[vertex];
    for (uint body = 0u; body < body_count; ++body)
    {
        ClothBodyCorrection correction = corrections[body];
        if (!correction.active) continue;
        if (correction.vertices[0] == vertex ||
            correction.vertices[1] == vertex ||
            correction.vertices[2] == vertex)
            offset = add(offset, correction.offset);
        if (correction.weight_sum > 1.0e-8f)
        {
            float squared = length_squared(subtract(
                positions[vertex], correction.contact));
            float supportSquared = correction.support_radius *
                correction.support_radius;
            float weight = max(0.0f, 1.0f - squared / supportSquared);
            velocity = add(velocity, multiply(correction.impulse,
                inverse_masses[vertex] * weight * weight /
                    correction.weight_sum));
        }
    }
    positions[vertex] = add(positions[vertex], offset);
    velocities[vertex] = clamp_length(velocity, 20.0f);
}
