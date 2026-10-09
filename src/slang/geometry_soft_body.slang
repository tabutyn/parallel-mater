// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_soft_body.cuh. The surface-contact gather is
// independent of the rigid-mesh query helpers used by the other two entries.

import cuda_common;

[shader("compute")]
[numthreads(256, 1, 1)]
void soft_body_apply_surface_contacts(
    uniform Vec3* positions, uniform Vec3* corrections,
    uniform uint* offsets, uniform SoftSurfaceInfluence* influences,
    uniform uint node_count)
{
    uint node = pmThreadIndex();
    if (node >= node_count) return;
    Vec3 correction = {};
    for (uint entry = offsets[node]; entry < offsets[node + 1u]; ++entry)
    {
        SoftSurfaceInfluence influence = influences[entry];
        Vec3 delta = multiply(
            corrections[influence.corner], influence.factor);
        float squared = length_squared(delta);
        if (squared > 1.0e-16f)
        {
            float remaining = max(
                0.0f, 1.0f - dot(correction, delta) / squared);
            correction = add(correction, multiply(delta, remaining));
        }
    }
    positions[node] = add(positions[node], correction);
}
