// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_soft_body.cuh. The surface-contact gather is
// independent of the rigid-mesh query helpers used by the other two entries.

import types;
import math;
import device_intrinsics;
import geometry_fluid;
import geometry;

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
    uniform bool couple_dynamic, uniform bool position_only,
    uniform bool static_only, uniform bool reconstruct_free_velocities,
    uniform float maximum_speed, uniform Vec3* body_position_corrections)
{
    uint vertex = pmThreadIndex();
    if (vertex >= count) return;
    if (!position_only)
    {
        if (body_position_corrections != nullptr)
            body_position_corrections[vertex] = {};
        impulses[vertex] = {};
        contact_forces[vertex] = {};
        if (contact_normals != nullptr) contact_normals[vertex] = {};
        if (couple_dynamic && contact_arms != nullptr)
            contact_arms[vertex] = {};
    }
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
        if (couple_dynamic && mesh.solid_planes != nullptr)
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
                bool solid = couple_dynamic && mesh.solid_planes != nullptr;
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
    if (position_only)
    {
        positions[vertex] = add(end, multiply(bestNormal, bestPenetration));
        return;
    }
    BodyParameters body = parameters[bestBody];
    RigidBodyState state = states[bestBody];
    if (couple_dynamic && body.motion == MotionType.dynamic)
        pmAtomicExchange(dynamic_contact_flag, 1u);
    if (contact_normals != nullptr) contact_normals[vertex] = bestNormal;
    if (accumulated_normal_delta != nullptr)
        accumulated_normal_delta[vertex] += bestPenetration;
    if (deformable_contact_count != nullptr)
        pmAtomicAdd(deformable_contact_count, 1u);
    Vec3 velocity = multiply(subtract(end, start), 1.0f / dt);
    float nodeShare = couple_dynamic
        ? inverse_masses[vertex] /
            (inverse_masses[vertex] + body.inverse_mass)
        : 1.0f;
    end = add(end, multiply(bestNormal, bestPenetration * nodeShare));
    Vec3 arm = subtract(bestContact, state.position);
    if (couple_dynamic)
    {
        impulses[vertex].body = bestBody;
        if (contact_arms != nullptr) contact_arms[vertex] = arm;
        pmAtomicExchange(&contact_flags[bestBody], 1u);
    }
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
            if (couple_dynamic && body.motion == MotionType.dynamic)
                end = add(end, multiply(velocityDelta, dt));
            impulses[vertex] = {multiply(impulse, -1.0f),
                multiply(cross(arm, impulse), -1.0f), bestBody};
            contact_forces[vertex] = multiply(impulse, 1.0f / dt);
            if (!couple_dynamic)
                pmAtomicExchange(&contact_flags[bestBody], 1u);
        }
    }
    if (couple_dynamic && body_position_corrections != nullptr)
        body_position_corrections[vertex] = multiply(bestNormal,
            -bestPenetration * body.inverse_mass /
            (inverse_masses[vertex] + body.inverse_mass));
    positions[vertex] = end;
    velocities[vertex] = clamp_length(velocity, maximum_speed);
}

struct SoftTriangleHalfspace
{
    Vec3 normal;
    float depths[3];
};

struct SoftTriangleCorrections
{
    Vec3 values[3];
};

bool soft_halfspace_finite(float value)
{
    const float largest = 3.402823466e+38f;
    return value >= -largest && value <= largest;
}

bool soft_halfspace_finite_vector(Vec3 value)
{
    return soft_halfspace_finite(value.x) &&
        soft_halfspace_finite(value.y) &&
        soft_halfspace_finite(value.z);
}

bool soft_halfspace_contains(Vec3 normal, float depth, Vec3 displacement,
    float roundoff)
{
    float terms[4] = {normal.x * displacement.x,
        normal.y * displacement.y, normal.z * displacement.z, depth};
    float scale = 0.0f;
    for (uint term = 0u; term < 4u; ++term)
    {
        if (!soft_halfspace_finite(terms[term])) return false;
        scale = max(scale, abs(terms[term]));
    }
    if (scale == 0.0f) return true;
    float magnitude = 0.0f;
    for (uint term = 0u; term < 4u; ++term)
        magnitude += abs(terms[term]) / scale;
    return terms[0] / scale + terms[1] / scale + terms[2] / scale >=
        depth / scale - roundoff * magnitude;
}

bool soft_project_two_halfspaces(Vec3 n0, float depth0,
    Vec3 n1, float depth1, out Vec3 displacement)
{
    displacement = {};
    if (!soft_halfspace_finite_vector(n0) ||
        !soft_halfspace_finite_vector(n1) ||
        !soft_halfspace_finite(depth0) ||
        !soft_halfspace_finite(depth1)) return false;
    if (depth0 <= 0.0f && depth1 <= 0.0f) return true;
    float norm0 = length_squared(n0), norm1 = length_squared(n1);
    if (!(norm0 > 0.0f) || !(norm1 > 0.0f) ||
        !soft_halfspace_finite(norm0) || !soft_halfspace_finite(norm1))
        return false;
    float alignment = dot(n0, n1);
    if (depth0 >= 0.0f)
    {
        Vec3 candidate = multiply(n0, depth0 / norm0);
        if (soft_halfspace_finite_vector(candidate) &&
            soft_halfspace_contains(n0, depth0, candidate,
                8.0f * 1.1920928955078125e-7f) &&
            soft_halfspace_contains(n1, depth1, candidate,
                8.0f * 1.1920928955078125e-7f))
        {
            displacement = candidate;
            return true;
        }
    }
    if (depth1 >= 0.0f)
    {
        Vec3 candidate = multiply(n1, depth1 / norm1);
        if (soft_halfspace_finite_vector(candidate) &&
            soft_halfspace_contains(n0, depth0, candidate,
                8.0f * 1.1920928955078125e-7f) &&
            soft_halfspace_contains(n1, depth1, candidate,
                8.0f * 1.1920928955078125e-7f))
        {
            displacement = candidate;
            return true;
        }
    }
    float difference = 1.0f - alignment;
    float determinant = difference * (1.0f + alignment);
    if (!(determinant > 1.0e-6f)) return false;
    float lambda0 = alignment > 0.0f
        ? ((depth0 - depth1) + difference * depth1) / determinant
        : (depth0 - alignment * depth1) / determinant;
    float lambda1 = alignment > 0.0f
        ? ((depth1 - depth0) + difference * depth0) / determinant
        : (depth1 - alignment * depth0) / determinant;
    if (!(lambda0 >= 0.0f && lambda1 >= 0.0f) ||
        !soft_halfspace_finite(lambda0) ||
        !soft_halfspace_finite(lambda1)) return false;
    Vec3 candidate = add(multiply(n0, lambda0), multiply(n1, lambda1));
    if (!soft_halfspace_finite_vector(candidate)) return false;
    displacement = candidate;
    return true;
}

bool soft_triangle_halfspace_lower_bound(
    SoftTriangleHalfspace plane, out float cost)
{
    cost = 0.0f;
    if (!soft_halfspace_finite_vector(plane.normal)) return false;
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        float depth = plane.depths[corner];
        if (!soft_halfspace_finite(depth)) return false;
        if (depth > 0.0f) cost += depth * depth;
    }
    return soft_halfspace_finite(cost);
}

bool soft_project_triangle_halfspace_pair(
    SoftTriangleHalfspace first, SoftTriangleHalfspace second,
    out SoftTriangleCorrections corrections, out float cost)
{
    corrections = {};
    cost = 0.0f;
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        Vec3 candidate = {};
        if (!soft_project_two_halfspaces(first.normal, first.depths[corner],
            second.normal, second.depths[corner], candidate)) return false;
        const float roundoff = 64.0f * 1.1920928955078125e-7f;
        if (!soft_halfspace_contains(first.normal, first.depths[corner],
                candidate, roundoff) ||
            !soft_halfspace_contains(second.normal, second.depths[corner],
                candidate, roundoff)) return false;
        corrections.values[corner] = candidate;
        cost += length_squared(candidate);
    }
    return soft_halfspace_finite(cost);
}

SoftTriangleHalfspace soft_surface_halfspace(TriangleMeshResource mesh,
    RigidBodyState state, Vec3 world[3], float margin, uint face)
{
    CollisionPlane plane =
        mesh.solid_planes[mesh.index_count / 3u + face];
    SoftTriangleHalfspace result = {};
    result.normal = rotate(state.orientation, plane.normal);
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        Vec3 local = inverse_rotate(state.orientation,
            subtract(world[corner], state.position));
        result.depths[corner] = margin + plane.offset -
            dot(plane.normal, local);
    }
    return result;
}

bool soft_project_convex_pair_triangle(
    TriangleMeshResource firstMesh, RigidBodyState firstState,
    float firstMargin, TriangleMeshResource secondMesh,
    RigidBodyState secondState, float secondMargin, Vec3 world[3],
    SoftTriangleHalfspace seedFirst, SoftTriangleHalfspace seedSecond,
    out SoftTriangleCorrections corrections)
{
    corrections = {};
    float bestCost = 3.402823466e+38f;
    bool found = soft_project_triangle_halfspace_pair(
        seedFirst, seedSecond, corrections, bestCost);
    if (found && bestCost == 0.0f) return true;
    for (uint firstFace = 0u;
         firstFace < firstMesh.solid_unique_plane_count; ++firstFace)
    {
        SoftTriangleHalfspace first = soft_surface_halfspace(
            firstMesh, firstState, world, firstMargin, firstFace);
        float firstLower = 0.0f;
        if (!soft_triangle_halfspace_lower_bound(first, firstLower) ||
            (found && firstLower >= bestCost)) continue;
        for (uint secondFace = 0u;
             secondFace < secondMesh.solid_unique_plane_count; ++secondFace)
        {
            SoftTriangleHalfspace second = soft_surface_halfspace(
                secondMesh, secondState, world, secondMargin, secondFace);
            float secondLower = 0.0f;
            if (!soft_triangle_halfspace_lower_bound(second, secondLower) ||
                (found && secondLower >= bestCost)) continue;
            SoftTriangleCorrections candidate = {};
            float candidateCost = 0.0f;
            if (!soft_project_triangle_halfspace_pair(
                    first, second, candidate, candidateCost) ||
                (found && candidateCost >= bestCost)) continue;
            corrections = candidate;
            bestCost = candidateCost;
            found = true;
            if (bestCost == 0.0f) return true;
        }
    }
    return found;
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
    uint contactBodies[2] = {};
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
            if (!rigid_triangle_bounds_overlap(
                    point[0], point[1], point[2], a, b, c, margin))
                continue;
            Vec3 onSoft = {}, onRigid = {};
            rigid_closest_triangle_pair(point[0], point[1], point[2],
                a, b, c, onSoft, onRigid, false);
            contact = length_squared(subtract(onSoft, onRigid)) <
                margin * margin;
        }
        if (!contact) continue;
        uint slot = contactCount == 0u ||
            margin - separation > depths[0] ? 0u : 1u;
        if (slot == 0u && contactCount != 0u)
        {
            depths[1] = depths[0];
            normals[1] = normals[0];
            contactBodies[1] = contactBodies[0];
            for (uint corner = 0u; corner < 3u; ++corner)
                cornerDepths[1][corner] = cornerDepths[0][corner];
        }
        depths[slot] = margin - separation;
        contactBodies[slot] = body;
        normals[slot] = rotate(state.orientation, support.normal);
        for (uint corner = 0u; corner < 3u; ++corner)
            cornerDepths[slot][corner] = margin + support.offset -
                dot(support.normal, point[corner]);
        contactCount = contactCount == 0u ? 1u : 2u;
    }
    for (uint corner = 0u; corner < 3u && contactCount != 0u; ++corner)
        corrections[base + corner] = multiply(normals[0],
            max(0.0f, cornerDepths[0][corner]));
    if (contactCount == 2u)
    {
        uint firstBody = contactBodies[0], secondBody = contactBodies[1];
        TriangleMeshResource firstMesh =
            meshes[parameters[firstBody].mesh.index];
        TriangleMeshResource secondMesh =
            meshes[parameters[secondBody].mesh.index];
        RigidBodyState firstState = states[firstBody];
        RigidBodyState secondState = states[secondBody];
        SoftTriangleHalfspace firstSeed = {};
        SoftTriangleHalfspace secondSeed = {};
        firstSeed.normal = normals[0];
        secondSeed.normal = normals[1];
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            firstSeed.depths[corner] = cornerDepths[0][corner];
            secondSeed.depths[corner] = cornerDepths[1][corner];
        }
        SoftTriangleCorrections paired = {};
        if (soft_project_convex_pair_triangle(
            firstMesh, firstState, parameters[firstBody].collision_margin,
            secondMesh, secondState, parameters[secondBody].collision_margin,
            world, firstSeed, secondSeed, paired))
            for (uint corner = 0u; corner < 3u; ++corner)
                corrections[base + corner] = paired.values[corner];
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
