// SPDX-License-Identifier: MIT
// Slang mirror of the CUDA entry points declared directly in src/world.cu.

import cuda_common;

Vec3 hit_box_local_point(HitBox box, Vec3 point)
{
    return inverse_rotate(box.orientation, subtract(point, box.center));
}

bool hit_box_contains_point(HitBox box, Vec3 point)
{
    Vec3 local = hit_box_local_point(box, point);
    return abs(local.x) <= box.half_extents.x &&
           abs(local.y) <= box.half_extents.y &&
           abs(local.z) <= box.half_extents.z;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void query_particle_hit_box_kernel(
    uniform HitBox box, uniform Vec3* positions,
    uniform uint particle_count, uniform uint8_t* matches)
{
    uint particle = pmThreadIndex();
    if (particle >= particle_count) return;
    matches[particle] = hit_box_contains_point(box, positions[particle])
        ? uint8_t(1u) : uint8_t(0u);
}
