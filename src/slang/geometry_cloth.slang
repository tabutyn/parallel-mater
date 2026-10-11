// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_cloth.cuh. The correction gather is independent
// of the rigid-mesh/paint query used by cloth_constrain_bodies.

import cuda_common;
import geometry_fluid;

void stamp_rigid_cloth_paint(
    RigidBodyId rigid_id, ClothId cloth_id, Vec3 center, Vec3 contact,
    Vec3* positions, uint* indices, uint* vertex_sources,
    uint triangle_count, PaintFieldResource* fields, uint field_capacity,
    PaintRuleResource* rules, uint rule_capacity)
{
    for (uint ruleIndex = 0u; ruleIndex < rule_capacity; ++ruleIndex)
    {
        PaintRuleResource rule = rules[ruleIndex];
        if (!rule.alive || !rule.options.enabled ||
            rule.options.rigid_source.index != rigid_id.index ||
            rule.options.rigid_source.generation != rigid_id.generation ||
            rule.options.target.index >= field_capacity) continue;
        PaintFieldResource field = fields[rule.options.target.index];
        if (!field.alive || field.generation != rule.options.target.generation ||
            field.options.cloth.index != cloth_id.index ||
            field.options.cloth.generation != cloth_id.generation) continue;
        int width = int(field.options.width), height = int(field.options.height);
        float radiusSquared = rule.options.brush_radius *
            rule.options.brush_radius;
        for (uint triangle = 0u; triangle < triangle_count; ++triangle)
        {
            uint base = triangle * 3u;
            uint ia = indices[base], ib = indices[base + 1u];
            uint ic = indices[base + 2u];
            if (ia == ib) continue;
            Vec3 a = positions[ia], b = positions[ib], c = positions[ic];
            if (length_squared(subtract(contact,
                fluid_closest_triangle(contact, a, b, c))) > radiusSquared)
                continue;
            uint uaIndex = vertex_sources != nullptr ? vertex_sources[ia] : ia;
            uint ubIndex = vertex_sources != nullptr ? vertex_sources[ib] : ib;
            uint ucIndex = vertex_sources != nullptr ? vertex_sources[ic] : ic;
            Vec2 ua = field.uvs[uaIndex], ub = field.uvs[ubIndex];
            Vec2 uc = field.uvs[ucIndex];
            float e0x = ub.x - ua.x, e0y = ub.y - ua.y;
            float e1x = uc.x - ua.x, e1y = uc.y - ua.y;
            float determinant = e0x * e1y - e0y * e1x;
            if (abs(determinant) < 1.0e-10f) continue;
            int x0 = max(0, int(floor(min(ua.x, min(ub.x, uc.x)) * width)));
            int x1 = min(width - 1,
                int(floor(max(ua.x, max(ub.x, uc.x)) * width)));
            int y0 = max(0, int(floor(min(ua.y, min(ub.y, uc.y)) * height)));
            int y1 = min(height - 1,
                int(floor(max(ua.y, max(ub.y, uc.y)) * height)));
            uint side = dot(cross(subtract(b, a), subtract(c, a)),
                subtract(center, contact)) >= 0.0f ? 1u : 2u;
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                {
                    float qx = (float(x) + 0.5f) / float(width) - ua.x;
                    float qy = (float(y) + 0.5f) / float(height) - ua.y;
                    float v = (qx * e1y - qy * e1x) / determinant;
                    float w = (e0x * qy - e0y * qx) / determinant;
                    float u = 1.0f - v - w;
                    if (u < -1.0e-4f || v < -1.0e-4f || w < -1.0e-4f)
                        continue;
                    Vec3 point = add(multiply(a, u),
                        add(multiply(b, v), multiply(c, w)));
                    if (length_squared(subtract(point, contact)) <= radiusSquared)
                        pmAtomicOr(&field.pixels[y * width + x], side);
                }
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_constrain_bodies(
    uniform Vec3* cloth_positions, uniform Vec3* cloth_velocities,
    uniform uint* cloth_indices, uniform uint* vertex_sources,
    uniform Vec3* surface_positions, uniform float* cloth_inverse_masses,
    uniform uint vertex_count, uniform uint triangle_count,
    uniform float thickness, uniform float dt, uniform float contact_friction,
    uniform BodyParameters* parameters,
    uniform RigidBodyState* previous_states, uniform RigidBodyState* states,
    uniform TriangleMeshResource* meshes, uniform RigidBodyId* body_ids,
    uniform uint body_count, uniform ClothId cloth_id,
    uniform PaintFieldResource* paint_fields, uniform uint field_capacity,
    uniform PaintRuleResource* paint_rules, uniform uint rule_capacity,
    uniform ClothBodyCorrection* corrections)
{
    uint bodyIndex = pmThreadIndex();
    if (bodyIndex >= body_count) return;
    corrections[bodyIndex] = {};
    if (parameters[bodyIndex].motion != MotionType.dynamic) return;
    BodyParameters body = parameters[bodyIndex];
    RigidBodyState state = states[bodyIndex];
    TriangleMeshResource mesh = meshes[body.mesh.index];
    Vec3 center = transform_point(state, mesh.bounding_center);
    Vec3 previousCenter = transform_point(
        previous_states[bodyIndex], mesh.bounding_center);
    float radius = mesh.bounding_radius + body.collision_margin + thickness;
    float bestPenetration = 0.0f;
    Vec3 bestNormal = {}, bestContact = {};
    uint bestTriangle = 0u;
    for (uint triangle = 0u; triangle < triangle_count; ++triangle)
    {
        uint base = triangle * 3u;
        Vec3 a = surface_positions != nullptr ? surface_positions[base] :
            cloth_positions[cloth_indices[base]];
        Vec3 b = surface_positions != nullptr ? surface_positions[base + 1u] :
            cloth_positions[cloth_indices[base + 1u]];
        Vec3 c = surface_positions != nullptr ? surface_positions[base + 2u] :
            cloth_positions[cloth_indices[base + 2u]];
        Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) < 1.0e-12f) continue;
        Vec3 normal = normalized_or(face, {0.0f, 0.0f, 1.0f});
        Vec3 nearest = fluid_closest_triangle(center, a, b, c);
        Vec3 delta = subtract(center, nearest);
        float distance = vector_length(delta);
        Vec3 contactNormal = distance > 1.0e-6f
            ? multiply(delta, 1.0f / distance)
            : multiply(normal, dot(subtract(previousCenter, a), normal) >= 0.0f
                ? 1.0f : -1.0f);
        float penetration = radius - distance;
        float before = dot(subtract(previousCenter, a), normal);
        float after = dot(subtract(center, a), normal);
        if (before * after < 0.0f)
        {
            float fraction = before / (before - after);
            Vec3 crossing = add(previousCenter,
                multiply(subtract(center, previousCenter), fraction));
            if (length_squared(subtract(fluid_closest_triangle(
                crossing, a, b, c), crossing)) < radius * radius)
            {
                contactNormal = multiply(normal, before > 0.0f ? 1.0f : -1.0f);
                penetration = max(penetration, radius + abs(after));
            }
        }
        else if (penetration > 0.0f && before * after > 0.0f)
            contactNormal = multiply(normal, before > 0.0f ? 1.0f : -1.0f);
        if (penetration > bestPenetration)
        {
            bestPenetration = penetration;
            bestNormal = contactNormal;
            bestContact = nearest;
            bestTriangle = triangle;
        }
    }
    if (bestPenetration <= 0.0f) return;
    uint first = bestTriangle * 3u;
    uint a = cloth_indices[first], b = cloth_indices[first + 1u];
    uint c = cloth_indices[first + 2u];
    if (rule_capacity != 0u)
        stamp_rigid_cloth_paint(body_ids[bodyIndex], cloth_id, center,
            bestContact, cloth_positions, cloth_indices, vertex_sources,
            triangle_count, paint_fields, field_capacity,
            paint_rules, rule_capacity);
    if (surface_positions != nullptr) return;
    float freeFraction = ((cloth_inverse_masses[a] > 0.0f ? 1.0f : 0.0f) +
        (cloth_inverse_masses[b] > 0.0f ? 1.0f : 0.0f) +
        (cloth_inverse_masses[c] > 0.0f ? 1.0f : 0.0f)) / 3.0f;
    float clothShift = 0.5f * bestPenetration;
    Vec3 arm = subtract(bestContact, state.position);
    float incoming = dot(state.linear_velocity, bestNormal);
    Vec3 angularAxis = cross(arm, bestNormal);
    float normalImpulse = incoming < 0.0f && body.inverse_mass > 0.0f
        ? -incoming / body.inverse_mass : 0.0f;
    float supportRadius = max(0.15f, mesh.bounding_radius * 0.8f);
    float inverseSupportSquared = 1.0f / (supportRadius * supportRadius);
    float weightSum = 0.0f, maximumWeightedInverseMass = 0.0f;
    float weightedInverseMassSquared = 0.0f;
    Vec3 weightedClothVelocity = {};
    for (uint vertex = 0u; vertex < vertex_count; ++vertex)
    {
        float inverseMass = cloth_inverse_masses[vertex];
        if (inverseMass == 0.0f) continue;
        float squared = length_squared(subtract(
            cloth_positions[vertex], bestContact));
        float weight = max(0.0f, 1.0f - squared * inverseSupportSquared);
        float weighted = weight * weight;
        weightSum += weighted;
        weightedInverseMassSquared += inverseMass * weighted * weighted;
        weightedClothVelocity = add(weightedClothVelocity,
            multiply(cloth_velocities[vertex], weighted));
        maximumWeightedInverseMass = max(
            maximumWeightedInverseMass, inverseMass * weighted);
    }
    float maximumClothImpulse = maximumWeightedInverseMass > 0.0f
        ? 0.1f * thickness * weightSum /
            (dt * maximumWeightedInverseMass) : normalImpulse;
    float clothImpulse = min(normalImpulse, maximumClothImpulse);
    Vec3 tangentImpulse = {};
    if (weightSum > 1.0e-8f && contact_friction > 0.0f)
    {
        Vec3 clothVelocity = multiply(weightedClothVelocity, 1.0f / weightSum);
        Vec3 bodyVelocity = add(state.linear_velocity,
            cross(state.angular_velocity, arm));
        Vec3 relative = subtract(bodyVelocity, clothVelocity);
        float separatingSpeed = dot(relative, bestNormal);
        Vec3 tangent = subtract(relative,
            multiply(bestNormal, separatingSpeed));
        float tangentSpeed = vector_length(tangent);
        float releaseWeight = max(0.0f, 1.0f - max(incoming, 0.0f) / 0.5f);
        if (releaseWeight > 0.0f && separatingSpeed <= 0.0f &&
            tangentSpeed > 1.0e-6f)
        {
            Vec3 direction = multiply(tangent, 1.0f / tangentSpeed);
            Vec3 tangentAxis = cross(arm, direction);
            float clothInverseMass = weightedInverseMassSquared /
                (weightSum * weightSum);
            float effectiveInverseMass = body.inverse_mass + dot(cross(
                inverse_inertia_world(body, state, tangentAxis), arm),
                direction) + clothInverseMass;
            float correctionImpulse = bestPenetration > 0.0f &&
                body.inverse_mass > 0.0f
                ? 0.2f * bestPenetration / (dt * body.inverse_mass) : 0.0f;
            float frictionLimit = contact_friction * releaseWeight *
                max(normalImpulse, correctionImpulse);
            float magnitude = effectiveInverseMass > kEpsilon
                ? min(tangentSpeed / effectiveInverseMass, frictionLimit) : 0.0f;
            tangentImpulse = multiply(direction, -magnitude);
        }
    }
    Vec3 clothTangentImpulse = clamp_length(
        tangentImpulse, maximumClothImpulse);
    ClothBodyCorrection correction = {};
    correction.offset = multiply(bestNormal, -clothShift);
    correction.impulse = subtract(
        multiply(bestNormal, -clothImpulse), clothTangentImpulse);
    correction.contact = bestContact;
    correction.support_radius = supportRadius;
    correction.weight_sum = weightSum;
    correction.vertices[0] = a;
    correction.vertices[1] = b;
    correction.vertices[2] = c;
    correction.active = true;
    corrections[bodyIndex] = correction;
    state.position = add(state.position, multiply(bestNormal,
        bestPenetration - clothShift * freeFraction));
    if (normalImpulse > 0.0f)
    {
        state.linear_velocity = clamp_length(add(state.linear_velocity,
            multiply(bestNormal, normalImpulse * body.inverse_mass)),
            body.maximum_linear_speed);
        state.angular_velocity = clamp_length(add(state.angular_velocity,
            inverse_inertia_world(body, state,
                multiply(angularAxis, normalImpulse))),
            body.maximum_angular_speed);
    }
    state.linear_velocity = clamp_length(add(state.linear_velocity,
        multiply(tangentImpulse, body.inverse_mass)), body.maximum_linear_speed);
    state.angular_velocity = clamp_length(add(state.angular_velocity,
        inverse_inertia_world(body, state, cross(arm, tangentImpulse))),
        body.maximum_angular_speed);
    states[bodyIndex] = state;
}

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
