// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_soft_body.cuh. The surface-contact gather is
// independent of the rigid-mesh query helpers used by the other two entries.

import cuda_common;
import geometry_fluid;

[shader("compute")]
[numthreads(256, 1, 1)]
void deformable_collide(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* previous, uniform float* inverse_masses,
    uniform uint count, uniform float thickness, uniform float dt,
    uniform BodyParameters* parameters,
    uniform RigidBodyState* previous_states, uniform RigidBodyState* states,
    uniform TriangleMeshResource* meshes, uniform uint body_count,
    uniform FluidBodyImpulse* impulses, uniform Vec3* contact_forces,
    uniform Vec3* contact_normals, uniform Vec3* contact_arms,
    uniform Vec3* contact_momentum_delta,
    uniform float* accumulated_normal_delta,
    uniform uint* deformable_contact_count,
    uniform uint* dynamic_contact_flag, uniform uint* contact_flags,
    uniform bool static_only, uniform bool reconstruct_free_velocities,
    uniform float maximum_speed, uniform Vec3* body_position_corrections)
{
    uint vertex = pmThreadIndex();
    if (vertex >= count) return;
    if (body_position_corrections != nullptr)
        body_position_corrections[vertex] = {};
    impulses[vertex] = {};
    contact_forces[vertex] = {};
    if (contact_normals != nullptr) contact_normals[vertex] = {};
    if (contact_arms != nullptr) contact_arms[vertex] = {};
    if (inverse_masses[vertex] == 0.0f) return;
    Vec3 start = previous[vertex], end = positions[vertex];
    float bestPenetration = 0.0f;
    Vec3 bestNormal = {}, bestContact = {};
    uint bestBody = kInvalidDense;
    for (uint bodyIndex = 0u; bodyIndex < body_count; ++bodyIndex)
    {
        BodyParameters body = parameters[bodyIndex];
        if (static_only && body.motion != MotionType.static_body) continue;
        RigidBodyState state = states[bodyIndex];
        RigidBodyState previousState = previous_states != nullptr
            ? previous_states[bodyIndex] : state;
        TriangleMeshResource mesh = meshes[body.mesh.index];
        if (mesh.bvh_node_count == 0u) continue;
        Vec3 localStart = inverse_rotate(previousState.orientation,
            subtract(start, previousState.position));
        Vec3 localEnd = inverse_rotate(state.orientation,
            subtract(end, state.position));
        float margin = thickness + body.collision_margin;
        if (!fluid_segment_bounds(
            localStart, localEnd, mesh.bvh_nodes[0], margin)) continue;
        if (mesh.solid_planes != nullptr)
        {
            float nearestSide = -3.402823466e+38f;
            CollisionPlane nearestPlane = {};
            for (uint triangle = 0u; triangle < mesh.index_count / 3u;
                 ++triangle)
            {
                CollisionPlane plane = mesh.solid_planes[triangle];
                float side = dot(plane.normal, localEnd) - plane.offset;
                if (side > nearestSide)
                {
                    nearestSide = side;
                    nearestPlane = plane;
                }
                if (side > 0.0f) break;
            }
            if (nearestSide <= 0.0f)
            {
                float penetration = margin - nearestSide;
                if (penetration > bestPenetration)
                {
                    bestPenetration = penetration;
                    bestNormal = rotate(state.orientation, nearestPlane.normal);
                    bestContact = transform_point(state, subtract(localEnd,
                        multiply(nearestPlane.normal, nearestSide)));
                    bestBody = bodyIndex;
                }
                continue;
            }
        }
        uint stack[64] = {};
        int pending = 1;
        while (pending != 0)
        {
            BvhNode node = mesh.bvh_nodes[stack[--pending]];
            if (!fluid_segment_bounds(localStart, localEnd, node, margin))
                continue;
            if (node.triangle_count == 0u)
            {
                if (pending + 2 > 64) continue;
                stack[pending++] = node.right;
                stack[pending++] = node.left;
                continue;
            }
            for (uint item = 0u; item < node.triangle_count; ++item)
            {
                uint base = (node.first_triangle + item) * 3u;
                Vec3 a = mesh.vertices[mesh.indices[base]];
                Vec3 b = mesh.vertices[mesh.indices[base + 1u]];
                Vec3 c = mesh.vertices[mesh.indices[base + 2u]];
                Vec3 nearest = fluid_closest_triangle(localEnd, a, b, c);
                Vec3 delta = subtract(localEnd, nearest);
                float distance = vector_length(delta);
                bool solid = mesh.solid_planes != nullptr;
                Vec3 face = solid ? mesh.solid_planes[base / 3u].normal :
                    normalized_or(cross(subtract(b, a), subtract(c, a)),
                        {0.0f, 1.0f, 0.0f});
                Vec3 normal = distance > 1.0e-6f
                    ? multiply(delta, 1.0f / distance)
                    : multiply(face,
                        dot(subtract(localStart, a), face) >= 0.0f
                            ? 1.0f : -1.0f);
                float penetration = margin - distance;
                float before = dot(subtract(localStart, a), face);
                float after = dot(subtract(localEnd, a), face);
                if (before * after < 0.0f && (!solid || before > 0.0f))
                {
                    float fraction = before / (before - after);
                    Vec3 crossing = add(localStart,
                        multiply(subtract(localEnd, localStart), fraction));
                    if (length_squared(subtract(fluid_closest_triangle(
                        crossing, a, b, c), crossing)) < thickness * thickness)
                    {
                        normal = multiply(face, before > 0.0f ? 1.0f : -1.0f);
                        penetration = max(penetration, margin + abs(after));
                    }
                }
                if (penetration > bestPenetration)
                {
                    bestPenetration = penetration;
                    bestNormal = rotate(state.orientation, normal);
                    bestContact = transform_point(state, nearest);
                    bestBody = bodyIndex;
                }
            }
        }
    }
    if (bestBody == kInvalidDense)
    {
        if (reconstruct_free_velocities)
            velocities[vertex] = clamp_length(multiply(
                subtract(end, start), 1.0f / dt), maximum_speed);
        return;
    }
    BodyParameters body = parameters[bestBody];
    RigidBodyState state = states[bestBody];
    if (body.motion == MotionType.dynamic)
        pmAtomicExchange(dynamic_contact_flag, 1u);
    if (contact_normals != nullptr) contact_normals[vertex] = bestNormal;
    if (accumulated_normal_delta != nullptr)
        accumulated_normal_delta[vertex] += bestPenetration;
    if (deformable_contact_count != nullptr)
        pmAtomicAdd(deformable_contact_count, 1u);
    Vec3 velocity = multiply(subtract(end, start), 1.0f / dt);
    float nodeShare = inverse_masses[vertex] /
        (inverse_masses[vertex] + body.inverse_mass);
    end = add(end, multiply(bestNormal, bestPenetration * nodeShare));
    Vec3 arm = subtract(bestContact, state.position);
    impulses[vertex].body = bestBody;
    if (contact_arms != nullptr) contact_arms[vertex] = arm;
    pmAtomicExchange(&contact_flags[bestBody], 1u);
    Vec3 bodyVelocity = add(state.linear_velocity,
        cross(state.angular_velocity, arm));
    Vec3 relative = subtract(velocity, bodyVelocity);
    float incoming = dot(relative, bestNormal);
    if (incoming < 0.0f)
    {
        float mass = 1.0f / max(inverse_masses[vertex], 1.0e-6f);
        Vec3 armCross = cross(arm, bestNormal);
        float denominator = 1.0f / mass + body.inverse_mass + dot(cross(
            inverse_inertia_world(body, state, armCross), arm), bestNormal);
        if (denominator > kEpsilon)
        {
            Vec3 impulse = multiply(bestNormal,
                -(1.0f + body.restitution) * incoming / denominator);
            Vec3 velocityDelta = multiply(impulse, 1.0f / mass);
            velocity = add(velocity, velocityDelta);
            if (contact_momentum_delta != nullptr)
                contact_momentum_delta[vertex] = add(
                    contact_momentum_delta[vertex], impulse);
            if (body.motion == MotionType.dynamic)
                end = add(end, multiply(velocityDelta, dt));
            impulses[vertex] = {multiply(impulse, -1.0f),
                multiply(cross(arm, impulse), -1.0f), bestBody};
            contact_forces[vertex] = multiply(impulse, 1.0f / dt);
        }
    }
    if (body_position_corrections != nullptr)
        body_position_corrections[vertex] = multiply(bestNormal,
            -bestPenetration * body.inverse_mass /
            (inverse_masses[vertex] + body.inverse_mass));
    positions[vertex] = end;
    velocities[vertex] = clamp_length(velocity, maximum_speed);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void soft_body_surface_contacts(
    uniform Vec3* surface, uniform uint* indices,
    uniform uint triangle_count, uniform BodyParameters* parameters,
    uniform RigidBodyState* states, uniform TriangleMeshResource* meshes,
    uniform uint body_count, uniform Vec3* corrections)
{
    uint triangle = pmThreadIndex();
    if (triangle >= triangle_count) return;
    uint base = triangle * 3u;
    Vec3 world[3] = {surface[indices[base]], surface[indices[base + 1u]],
                     surface[indices[base + 2u]]};
    for (uint corner = 0u; corner < 3u; ++corner)
        corrections[base + corner] = {};
    float depths[2] = {};
    Vec3 normals[2] = {};
    float cornerDepths[2][3] = {};
    uint contactCount = 0u;
    for (uint body = 0u; body < body_count; ++body)
    {
        TriangleMeshResource mesh = meshes[parameters[body].mesh.index];
        if (mesh.solid_planes == nullptr) continue;
        RigidBodyState state = states[body];
        Vec3 point[3];
        for (uint corner = 0u; corner < 3u; ++corner)
            point[corner] = inverse_rotate(state.orientation,
                subtract(world[corner], state.position));
        float margin = parameters[body].collision_margin;
        Vec3 expansion = {margin, margin, margin};
        if (!bounds_overlap(subtract(component_min(point[0],
                component_min(point[1], point[2])), expansion),
            add(component_max(point[0], component_max(point[1], point[2])),
                expansion), mesh.minimum, mesh.maximum)) continue;
        float separation = -3.402823466e+38f;
        CollisionPlane support = {};
        for (uint face = 0u; face < mesh.index_count / 3u; ++face)
        {
            CollisionPlane plane = mesh.solid_planes[face];
            float minimumSide = min(dot(plane.normal, point[0]),
                min(dot(plane.normal, point[1]),
                    dot(plane.normal, point[2]))) - plane.offset;
            if (minimumSide > separation)
            {
                separation = minimumSide;
                support = plane;
            }
            if (separation >= margin) break;
        }
        if (separation >= margin ||
            (contactCount == 2u && margin - separation <= depths[1])) continue;
        bool contact = false;
        for (uint corner = 0u; corner < 3u && !contact; ++corner)
        {
            bool inside = true;
            for (uint face = 0u; face < mesh.index_count / 3u; ++face)
            {
                CollisionPlane plane = mesh.solid_planes[face];
                if (dot(plane.normal, point[corner]) > plane.offset)
                {
                    inside = false;
                    break;
                }
            }
            contact = inside;
        }
        for (uint face = 0u; face < mesh.index_count && !contact; face += 3u)
        {
            Vec3 a = mesh.vertices[mesh.indices[face]];
            Vec3 b = mesh.vertices[mesh.indices[face + 1u]];
            Vec3 c = mesh.vertices[mesh.indices[face + 2u]];
            for (uint corner = 0u; corner < 3u; ++corner)
                if (length_squared(subtract(point[corner],
                    fluid_closest_triangle(point[corner], a, b, c))) <
                    margin * margin) contact = true;
            if (!contact)
            {
                Vec3 softNearest = fluid_closest_triangle(a,
                    point[0], point[1], point[2]);
                if (length_squared(subtract(a, softNearest)) < margin * margin)
                    contact = true;
            }
        }
        if (!contact) continue;
        uint slot = contactCount == 0u ||
            margin - separation > depths[0] ? 0u : 1u;
        if (slot == 0u && contactCount != 0u)
        {
            depths[1] = depths[0];
            normals[1] = normals[0];
            for (uint corner = 0u; corner < 3u; ++corner)
                cornerDepths[1][corner] = cornerDepths[0][corner];
        }
        depths[slot] = margin - separation;
        normals[slot] = rotate(state.orientation, support.normal);
        for (uint corner = 0u; corner < 3u; ++corner)
            cornerDepths[slot][corner] = margin + support.offset -
                dot(support.normal, point[corner]);
        contactCount = contactCount == 0u ? 1u : 2u;
    }
    for (uint corner = 0u; corner < 3u && contactCount != 0u; ++corner)
    {
        Vec3 correction = multiply(normals[0],
            max(0.0f, cornerDepths[0][corner]));
        if (contactCount == 2u)
        {
            Vec3 second = multiply(normals[1],
                max(0.0f, cornerDepths[1][corner]));
            float squared = length_squared(second);
            if (squared > 1.0e-16f)
                correction = add(correction, multiply(second,
                    max(0.0f, 1.0f - dot(correction, second) / squared)));
        }
        corrections[base + corner] = correction;
    }
}

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
