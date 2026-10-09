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

Vec3 hit_box_basis_axis(uint axis)
{
    if (axis == 0u) return {1.0f, 0.0f, 0.0f};
    if (axis == 1u) return {0.0f, 1.0f, 0.0f};
    return {0.0f, 0.0f, 1.0f};
}

bool hit_box_separates_triangle(
    Vec3 axis, Vec3 first, Vec3 second, Vec3 third, Vec3 half_extents)
{
    if (length_squared(axis) <= kEpsilon * kEpsilon) return false;
    float firstProjection = dot(axis, first);
    float secondProjection = dot(axis, second);
    float thirdProjection = dot(axis, third);
    float minimum = min(firstProjection, min(secondProjection, thirdProjection));
    float maximum = max(firstProjection, max(secondProjection, thirdProjection));
    float radius = abs(axis.x) * half_extents.x +
        abs(axis.y) * half_extents.y + abs(axis.z) * half_extents.z;
    return minimum > radius || maximum < -radius;
}

bool hit_box_overlaps_triangle(
    Vec3 first, Vec3 second, Vec3 third, Vec3 half_extents)
{
    Vec3 minimum = component_min(first, component_min(second, third));
    Vec3 maximum = component_max(first, component_max(second, third));
    if (minimum.x > half_extents.x || maximum.x < -half_extents.x ||
        minimum.y > half_extents.y || maximum.y < -half_extents.y ||
        minimum.z > half_extents.z || maximum.z < -half_extents.z) return false;
    Vec3 edges[3] = {
        subtract(second, first), subtract(third, second), subtract(first, third)};
    if (hit_box_separates_triangle(cross(edges[0], edges[1]),
        first, second, third, half_extents)) return false;
    for (uint edge = 0u; edge < 3u; ++edge)
        for (uint axis = 0u; axis < 3u; ++axis)
            if (hit_box_separates_triangle(cross(edges[edge],
                hit_box_basis_axis(axis)), first, second, third, half_extents))
                return false;
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void query_rigid_hit_box_kernel(
    uniform HitBox box, uniform BodyParameters* parameters,
    uniform RigidBodyState* states, uniform TriangleMeshResource* meshes,
    uniform uint body_count, uniform uint8_t* matches)
{
    uint body = pmThreadIndex();
    if (body >= body_count) return;
    TriangleMeshResource mesh = meshes[parameters[body].mesh.index];
    RigidBodyState state = states[body];
    bool overlap = false;
    for (uint index = 0u; index < mesh.index_count; index += 3u)
    {
        Vec3 first = hit_box_local_point(box,
            transform_point(state, mesh.vertices[mesh.indices[index]]));
        Vec3 second = hit_box_local_point(box,
            transform_point(state, mesh.vertices[mesh.indices[index + 1u]]));
        Vec3 third = hit_box_local_point(box,
            transform_point(state, mesh.vertices[mesh.indices[index + 2u]]));
        if (hit_box_overlaps_triangle(first, second, third, box.half_extents))
        {
            overlap = true;
            break;
        }
    }
    if (!overlap && mesh.solid_planes != nullptr)
    {
        Vec3 center = inverse_rotate(state.orientation,
            subtract(box.center, state.position));
        overlap = true;
        for (uint triangle = 0u; triangle < mesh.index_count / 3u; ++triangle)
        {
            CollisionPlane plane = mesh.solid_planes[triangle];
            if (dot(plane.normal, center) > plane.offset + kEpsilon)
            {
                overlap = false;
                break;
            }
        }
    }
    matches[body] = overlap ? uint8_t(1u) : uint8_t(0u);
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
