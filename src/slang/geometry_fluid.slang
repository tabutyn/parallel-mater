// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_fluid.cuh. Geometry contact kernels follow the
// shared closest-point helpers used by the coupling modules.

import cuda_common;

groupshared Vec3 fluid_impulse_linear[128];
groupshared Vec3 fluid_impulse_angular[128];
groupshared Vec3 fluid_impulse_position[128];

uint fluid_body_bucket(int x, int y, int z)
{
    uint hash = uint(x) * 73856093u ^ uint(y) * 19349663u ^
        uint(z) * 83492791u;
    return hash & 4095u;
}

public bool fluid_segment_bounds(
    Vec3 a, Vec3 b, BvhNode node, float radius)
{
    float lower = 0.0f, upper = 1.0f;
    float starts[3] = {a.x, a.y, a.z};
    float ends[3] = {b.x, b.y, b.z};
    float minima[3] = {node.minimum.x, node.minimum.y, node.minimum.z};
    float maxima[3] = {node.maximum.x, node.maximum.y, node.maximum.z};
    for (int axis = 0; axis < 3; ++axis)
    {
        float delta = ends[axis] - starts[axis];
        float minimum = minima[axis] - radius;
        float maximum = maxima[axis] + radius;
        if (abs(delta) < 1.0e-9f)
        {
            if (starts[axis] < minimum || starts[axis] > maximum) return false;
        }
        else
        {
            float first = (minimum - starts[axis]) / delta;
            float second = (maximum - starts[axis]) / delta;
            lower = max(lower, min(first, second));
            upper = min(upper, max(first, second));
            if (lower > upper) return false;
        }
    }
    return true;
}

public Vec3 fluid_closest_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c)
{
    Vec3 ab = subtract(b, a);
    Vec3 ac = subtract(c, a);
    Vec3 ap = subtract(p, a);
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    Vec3 bp = subtract(p, b);
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
        return add(a, multiply(ab, d1 / (d1 - d3)));
    Vec3 cp = subtract(p, c);
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
        return add(a, multiply(ac, d2 / (d2 - d6)));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f)
        return add(b, multiply(subtract(c, b),
            (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    float denominator = va + vb + vc;
    return denominator > 1.0e-12f
        ? add(a, add(multiply(ab, vb / denominator),
                     multiply(ac, vc / denominator)))
        : a;
}

public Vec3 fluid_closest_triangle_barycentric(
    Vec3 p, Vec3 a, Vec3 b, Vec3 c, out Vec3 weights)
{
    Vec3 ab = subtract(b, a);
    Vec3 ac = subtract(c, a);
    Vec3 ap = subtract(p, a);
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        weights = {1.0f, 0.0f, 0.0f};
        return a;
    }
    Vec3 bp = subtract(p, b);
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        weights = {0.0f, 1.0f, 0.0f};
        return b;
    }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        float v = d1 / (d1 - d3);
        weights = {1.0f - v, v, 0.0f};
        return add(a, multiply(ab, v));
    }
    Vec3 cp = subtract(p, c);
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        weights = {0.0f, 0.0f, 1.0f};
        return c;
    }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        float w = d2 / (d2 - d6);
        weights = {1.0f - w, 0.0f, w};
        return add(a, multiply(ac, w));
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f)
    {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        weights = {0.0f, 1.0f - w, w};
        return add(b, multiply(subtract(c, b), w));
    }
    float denominator = va + vb + vc;
    if (denominator <= 1.0e-12f)
    {
        weights = {1.0f, 0.0f, 0.0f};
        return a;
    }
    float inverse = 1.0f / denominator;
    float v = vb * inverse;
    float w = vc * inverse;
    weights = {1.0f - v - w, v, w};
    return add(a, add(multiply(ab, v), multiply(ac, w)));
}

public void atomic_add(Vec3* destination, Vec3 value)
{
    pmAtomicAdd(&destination.x, value.x);
    pmAtomicAdd(&destination.y, value.y);
    pmAtomicAdd(&destination.z, value.z);
}

void stamp_paint_at_contact(
    Vec3 local_particle, float particle_radius, FluidId source,
    RigidBodyId target, TriangleMeshResource* meshes,
    PaintFieldResource* fields, uint field_capacity,
    PaintRuleResource* rules, uint rule_capacity)
{
    for (uint ruleIndex = 0u; ruleIndex < rule_capacity; ++ruleIndex)
    {
        PaintRuleResource rule = rules[ruleIndex];
        if (!rule.alive || !rule.options.enabled ||
            rule.options.source.index != source.index ||
            rule.options.source.generation != source.generation ||
            rule.options.target.index >= field_capacity) continue;
        PaintFieldResource field = fields[rule.options.target.index];
        if (!field.alive || field.options.cloth.generation != 0u ||
            field.generation != rule.options.target.generation ||
            field.options.body.index != target.index ||
            field.options.body.generation != target.generation) continue;
        TriangleMeshResource mesh = meshes[field.options.mesh.index];
        float reach = particle_radius + rule.options.reach;
        float bestDistance = reach * reach;
        Vec2 bestUv = {};
        uint bestSide = 0u;
        uint stack[64] = {};
        int pending = mesh.bvh_node_count == 0u ? 0 : 1;
        while (pending != 0)
        {
            BvhNode node = mesh.bvh_nodes[stack[--pending]];
            if (!fluid_segment_bounds(
                local_particle, local_particle, node, reach)) continue;
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
                uint ia = mesh.indices[base], ib = mesh.indices[base + 1u];
                uint ic = mesh.indices[base + 2u];
                Vec3 a = mesh.vertices[ia], b = mesh.vertices[ib];
                Vec3 c = mesh.vertices[ic];
                Vec3 nearest = fluid_closest_triangle(local_particle, a, b, c);
                Vec3 delta = subtract(local_particle, nearest);
                float distance = length_squared(delta);
                if (distance >= bestDistance) continue;
                Vec3 ab = subtract(b, a), ac = subtract(c, a);
                Vec3 ap = subtract(nearest, a);
                float d00 = dot(ab, ab), d01 = dot(ab, ac);
                float d11 = dot(ac, ac), d20 = dot(ap, ab);
                float d21 = dot(ap, ac);
                float divisor = d00 * d11 - d01 * d01;
                if (divisor <= 1.0e-12f) continue;
                float v = (d11 * d20 - d01 * d21) / divisor;
                float w = (d00 * d21 - d01 * d20) / divisor;
                float u = 1.0f - v - w;
                bestUv = {
                    u * field.uvs[ia].x + v * field.uvs[ib].x +
                        w * field.uvs[ic].x,
                    u * field.uvs[ia].y + v * field.uvs[ib].y +
                        w * field.uvs[ic].y};
                bestSide = dot(cross(ab, ac), delta) >= 0.0f ? 1u : 2u;
                bestDistance = distance;
            }
        }
        if (bestSide == 0u) continue;
        int width = int(field.options.width);
        int height = int(field.options.height);
        int x = int(floor(bestUv.x * float(width))) % width;
        if (x < 0) x += width;
        int y = max(0, min(height - 1,
            int(floor(bestUv.y * float(height)))));
        pmAtomicOr(&field.pixels[y * width + x], bestSide);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_static_contacts(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* previous, uniform float* foam, uniform uint* count,
    uniform float radius, uniform float spawn_clearance,
    uniform uint first_spawned, uniform bool recover_spawn, uniform Vec3 up,
    uniform BodyParameters* parameters, uniform RigidBodyState* states,
    uniform TriangleMeshResource* meshes, uniform uint body_index,
    uniform float particle_mass, uniform bool collect_contacts,
    uniform FluidContactSample* samples, uniform uint8_t* contact_flags,
    uniform FluidId fluid_id, uniform RigidBodyId body_id,
    uniform bool apply_paint, uniform PaintFieldResource* paint_fields,
    uniform uint field_capacity, uniform PaintRuleResource* paint_rules,
    uniform uint rule_capacity)
{
    uint particle = pmThreadIndex();
    BodyParameters body = parameters[body_index];
    RigidBodyState state = states[body_index];
    TriangleMeshResource mesh = meshes[body.mesh.index];
    if (particle >= *count || mesh.bvh_node_count == 0u) return;
    Vec3 origin = inverse_rotate(state.orientation,
        subtract(previous[particle], state.position));
    Vec3 position = inverse_rotate(state.orientation,
        subtract(positions[particle], state.position));
    Vec3 velocity = inverse_rotate(state.orientation, velocities[particle]);
    bool newlySpawned = recover_spawn && particle >= first_spawned;
    float queryRadius = newlySpawned ? max(radius, spawn_clearance) : radius;
    Vec3 localUp = {};
    if (newlySpawned) localUp = inverse_rotate(state.orientation, up);
    float bestPenetration = 0.0f;
    Vec3 bestNormal = {}, bestContact = {};
    uint stack[64] = {};
    int pending = 1;
    while (pending != 0)
    {
        BvhNode node = mesh.bvh_nodes[stack[--pending]];
        if (!fluid_segment_bounds(origin, position, node, queryRadius)) continue;
        if (node.triangle_count == 0u)
        {
            if (pending + 2 > 64) continue;
            stack[pending++] = node.right;
            stack[pending++] = node.left;
            continue;
        }
        for (uint item = 0u; item < node.triangle_count; ++item)
        {
            uint triangle = (node.first_triangle + item) * 3u;
            Vec3 a = mesh.vertices[mesh.indices[triangle]];
            Vec3 b = mesh.vertices[mesh.indices[triangle + 1u]];
            Vec3 c = mesh.vertices[mesh.indices[triangle + 2u]];
            Vec3 face = normalized_or(cross(subtract(b, a), subtract(c, a)),
                {0.0f, 1.0f, 0.0f});
            Vec3 closest = fluid_closest_triangle(position, a, b, c);
            Vec3 delta = subtract(position, closest);
            float distance = vector_length(delta);
            Vec3 normal = distance > 1.0e-6f
                ? multiply(delta, 1.0f / distance)
                : multiply(face, dot(subtract(origin, a), face) >= 0.0f
                    ? 1.0f : -1.0f);
            float penetration = radius - distance;
            float before = dot(subtract(origin, a), face);
            float after = dot(subtract(position, a), face);
            if (before * after < 0.0f)
            {
                float fraction = before / (before - after);
                Vec3 crossing = add(origin,
                    multiply(subtract(position, origin), fraction));
                if (length_squared(subtract(fluid_closest_triangle(
                    crossing, a, b, c), crossing)) < radius * radius)
                {
                    normal = multiply(face, before > 0.0f ? 1.0f : -1.0f);
                    penetration = max(penetration, radius + abs(after));
                }
            }
            if (newlySpawned && abs(dot(face, localUp)) > 0.7f)
            {
                Vec3 floorNormal = dot(face, localUp) > 0.0f
                    ? face : multiply(face, -1.0f);
                float side = dot(subtract(position, a), floorNormal);
                if (side < radius && side > -spawn_clearance &&
                    distance < spawn_clearance)
                {
                    normal = floorNormal;
                    penetration = max(penetration, radius - side);
                }
            }
            if (penetration > bestPenetration)
            {
                bestPenetration = penetration;
                bestNormal = normal;
                bestContact = closest;
            }
        }
    }
    if (bestPenetration <= 0.0f) return;
    if (apply_paint && rule_capacity != 0u)
        stamp_paint_at_contact(position, radius, fluid_id, body_id,
            meshes, paint_fields, field_capacity, paint_rules, rule_capacity);
    position = add(position, multiply(bestNormal, bestPenetration));
    float incoming = dot(velocity, bestNormal);
    float normalImpulse = incoming < 0.0f
        ? -incoming * (1.0f + body.restitution) * particle_mass : 0.0f;
    if (incoming < 0.0f)
    {
        velocity = subtract(velocity,
            multiply(bestNormal, incoming * (1.0f + body.restitution)));
        Vec3 tangent = subtract(velocity,
            multiply(bestNormal, dot(velocity, bestNormal)));
        float tangentSpeed = vector_length(tangent);
        if (tangentSpeed > kEpsilon)
        {
            float frictionSpeed = min(tangentSpeed,
                body.friction * normalImpulse / particle_mass);
            velocity = subtract(velocity,
                multiply(tangent, frictionSpeed / tangentSpeed));
        }
        foam[particle] = max(foam[particle], min(1.0f, -incoming * 0.35f));
    }
    positions[particle] = add(state.position,
        rotate(state.orientation, position));
    velocities[particle] = rotate(state.orientation, velocity);
    if (collect_contacts && (contact_flags[particle] == 0u ||
        (contact_flags[particle] == 1u &&
         normalImpulse > samples[particle].normal_impulse)))
    {
        samples[particle] = {transform_point(state, bestContact),
            rotate(state.orientation, bestNormal), normalImpulse, body_index};
        contact_flags[particle] = 1u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_body_bounds_kernel(
    uniform BodyParameters* parameters, uniform RigidBodyState* previous,
    uniform RigidBodyState* current, uniform TriangleMeshResource* meshes,
    uniform uint count, uniform WorldAabb* bounds)
{
    uint body = pmThreadIndex();
    if (body >= count || parameters[body].motion == MotionType.static_body)
        return;
    TriangleMeshResource mesh = meshes[parameters[body].mesh.index];
    Vec3 minimum = {}, maximum = {};
    transformed_motion_bounds(mesh.minimum, mesh.maximum,
        bounds_transform(previous[body]), bounds_transform(current[body]),
        true, parameters[body].collision_margin, minimum, maximum);
    float rotationReach = rotational_motion_bound(
        previous[body], current[body], mesh);
    Vec3 expansion = {rotationReach, rotationReach, rotationReach};
    bounds[body] = {subtract(minimum, expansion), add(maximum, expansion)};
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_index_body_cells(
    uniform BodyParameters* parameters, uniform WorldAabb* bounds,
    uniform uint body_count, uniform uint words, uniform float padding,
    uniform uint64_t* masks, uniform uint64_t* global_masks)
{
    uint body = pmThreadIndex();
    if (body >= body_count ||
        parameters[body].motion == MotionType.static_body) return;
    WorldAabb box = bounds[body];
    int x0 = int(floor(box.minimum.x - padding));
    int y0 = int(floor(box.minimum.y - padding));
    int z0 = int(floor(box.minimum.z - padding));
    int x1 = int(floor(box.maximum.x + padding));
    int y1 = int(floor(box.maximum.y + padding));
    int z1 = int(floor(box.maximum.z + padding));
    uint64_t bit = 1ull << (body & 63u);
    uint word = body / 64u;
    if (int64_t(x1) - int64_t(x0) > 3 ||
        int64_t(y1) - int64_t(y0) > 3 ||
        int64_t(z1) - int64_t(z0) > 3)
    {
        pmAtomicOr(&global_masks[word], bit);
        return;
    }
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                pmAtomicOr(&masks[fluid_body_bucket(x, y, z) * words + word], bit);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_detect_moving_contacts(
    uniform Vec3* positions, uniform Vec3* previous, uniform uint* count,
    uniform float radius, uniform BodyParameters* parameters,
    uniform RigidBodyState* previous_states, uniform RigidBodyState* states,
    uniform TriangleMeshResource* meshes, uniform WorldAabb* bounds,
    uniform uint body_count, uniform uint64_t* masks,
    uniform uint64_t* global_masks, uniform uint words,
    uniform bool first_iteration, uniform FluidMovingContact* contacts,
    uniform uint* contact_counts)
{
    uint particle = pmThreadIndex();
    if (particle >= *count) return;
    contacts[particle] = {};
    Vec3 start = previous[particle], end = positions[particle];
    float bestPenetration = 0.0f;
    Vec3 bestNormal = {}, bestContact = {};
    uint bestBody = kInvalidDense;
    uint bucket = fluid_body_bucket(
        int(floor(end.x)), int(floor(end.y)), int(floor(end.z)));
    for (uint word = 0u; word < words; ++word)
    {
        uint64_t candidates = masks[bucket * words + word] | global_masks[word];
        for (uint bit = 0u; bit < 64u; ++bit)
        {
            if ((candidates & (1ull << bit)) == 0ull) continue;
            uint bodyIndex = word * 64u + bit;
            if (bodyIndex >= body_count) continue;
            BodyParameters body = parameters[bodyIndex];
            if (body.motion == MotionType.static_body) continue;
            WorldAabb box = bounds[bodyIndex];
            if (max(start.x, end.x) + radius < box.minimum.x ||
                min(start.x, end.x) - radius > box.maximum.x ||
                max(start.y, end.y) + radius < box.minimum.y ||
                min(start.y, end.y) - radius > box.maximum.y ||
                max(start.z, end.z) + radius < box.minimum.z ||
                min(start.z, end.z) - radius > box.maximum.z) continue;
            RigidBodyState state = states[bodyIndex];
            RigidBodyState old = first_iteration
                ? previous_states[bodyIndex] : state;
            Vec3 origin = inverse_rotate(old.orientation,
                subtract(start, old.position));
            Vec3 position = inverse_rotate(state.orientation,
                subtract(end, state.position));
            TriangleMeshResource mesh = meshes[body.mesh.index];
            if (mesh.bvh_node_count == 0u) continue;
            uint stack[64] = {};
            int pending = 1;
            while (pending != 0)
            {
                BvhNode node = mesh.bvh_nodes[stack[--pending]];
                if (!fluid_segment_bounds(origin, position, node, radius))
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
                    uint triangle = (node.first_triangle + item) * 3u;
                    Vec3 a = mesh.vertices[mesh.indices[triangle]];
                    Vec3 b = mesh.vertices[mesh.indices[triangle + 1u]];
                    Vec3 c = mesh.vertices[mesh.indices[triangle + 2u]];
                    Vec3 face = normalized_or(cross(subtract(b, a),
                        subtract(c, a)), {0.0f, 1.0f, 0.0f});
                    Vec3 closest = fluid_closest_triangle(position, a, b, c);
                    Vec3 delta = subtract(position, closest);
                    float distance = vector_length(delta);
                    Vec3 normal = distance > 1.0e-6f
                        ? multiply(delta, 1.0f / distance)
                        : multiply(face, dot(subtract(origin, a), face) >= 0.0f
                            ? 1.0f : -1.0f);
                    float penetration = radius - distance;
                    float before = dot(subtract(origin, a), face);
                    float after = dot(subtract(position, a), face);
                    if (before * after < 0.0f)
                    {
                        float fraction = before / (before - after);
                        Vec3 crossing = add(origin,
                            multiply(subtract(position, origin), fraction));
                        if (length_squared(subtract(fluid_closest_triangle(
                            crossing, a, b, c), crossing)) < radius * radius)
                        {
                            normal = multiply(face,
                                before > 0.0f ? 1.0f : -1.0f);
                            penetration = max(
                                penetration, radius + abs(after));
                        }
                    }
                    if (penetration > bestPenetration)
                    {
                        bestPenetration = penetration;
                        bestNormal = rotate(state.orientation, normal);
                        bestContact = transform_point(state, closest);
                        bestBody = bodyIndex;
                    }
                }
            }
        }
    }
    if (bestBody == kInvalidDense) return;
    contacts[particle] = {
        bestNormal, bestContact, bestPenetration, bestBody};
    pmAtomicAdd(&contact_counts[bestBody], 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_resolve_moving_contacts(
    uniform Vec3* positions, uniform Vec3* velocities, uniform float* foam,
    uniform uint* count, uniform float radius, uniform float particle_mass,
    uniform float timestep, uniform float maximum_speed,
    uniform BodyParameters* parameters, uniform RigidBodyState* states,
    uniform TriangleMeshResource* meshes,
    uniform FluidMovingContact* contacts, uniform uint* contact_counts,
    uniform FluidBodyImpulse* impulses, uniform bool collect_contacts,
    uniform FluidContactSample* samples,
    uniform uint8_t* particle_contact_flags, uniform FluidId fluid_id,
    uniform RigidBodyId* body_ids, uniform bool apply_paint,
    uniform PaintFieldResource* paint_fields, uniform uint field_capacity,
    uniform PaintRuleResource* paint_rules, uniform uint rule_capacity)
{
    uint particle = pmThreadIndex();
    if (particle >= *count) return;
    impulses[particle] = {};
    Vec3 end = positions[particle];
    FluidMovingContact contact = contacts[particle];
    Vec3 bestNormal = contact.normal;
    Vec3 bestContact = contact.point;
    float bestPenetration = contact.penetration;
    uint bestBody = contact.body;
    if (bestBody == kInvalidDense) return;
    BodyParameters body = parameters[bestBody];
    RigidBodyState state = states[bestBody];
    if (apply_paint && rule_capacity != 0u)
    {
        Vec3 localParticle = inverse_rotate(state.orientation,
            subtract(end, state.position));
        stamp_paint_at_contact(localParticle, radius, fluid_id,
            body_ids[bestBody], meshes, paint_fields, field_capacity,
            paint_rules, rule_capacity);
    }
    positions[particle] = add(end, multiply(bestNormal, bestPenetration));
    if (collect_contacts && particle_contact_flags[particle] != 2u)
    {
        samples[particle] = {bestContact, bestNormal, 0.0f, bestBody};
        particle_contact_flags[particle] = 2u;
    }
    Vec3 velocity = velocities[particle];
    Vec3 arm = subtract(bestContact, state.position);
    Vec3 bodyVelocity = add(state.linear_velocity,
        cross(state.angular_velocity, arm));
    Vec3 relative = subtract(velocity, bodyVelocity);
    float incoming = dot(relative, bestNormal);
    float recoverySpeed = min(maximum_speed,
        min(0.5f * radius, 0.2f * bestPenetration) / timestep);
    if (incoming >= recoverySpeed) return;
    Vec3 normalCross = cross(arm, bestNormal);
    float batchSize = 1.0f + float(contact_counts[bestBody] - 1u) *
        min(1.0f, 4.0f * particle_mass * body.inverse_mass);
    float normalDenominator = 1.0f / particle_mass + batchSize *
        (body.inverse_mass + dot(cross(inverse_inertia_world(
            body, state, normalCross), arm), bestNormal));
    if (normalDenominator <= kEpsilon) return;
    float normalImpulse = (recoverySpeed - incoming) / normalDenominator;
    if (collect_contacts && (particle_contact_flags[particle] != 2u ||
        normalImpulse > samples[particle].normal_impulse))
    {
        samples[particle] = {
            bestContact, bestNormal, normalImpulse, bestBody};
        particle_contact_flags[particle] = 2u;
    }
    Vec3 impulse = multiply(bestNormal, normalImpulse);
    velocity = add(velocity, multiply(impulse, 1.0f / particle_mass));
    Vec3 tangentVelocity = subtract(relative,
        multiply(bestNormal, incoming));
    float tangentSpeed = vector_length(tangentVelocity);
    if (tangentSpeed > kEpsilon)
    {
        Vec3 tangent = multiply(tangentVelocity, 1.0f / tangentSpeed);
        float tangentDenominator = 1.0f / particle_mass + batchSize *
            (body.inverse_mass + dot(cross(inverse_inertia_world(
                body, state, cross(arm, tangent)), arm), tangent));
        if (tangentDenominator > kEpsilon)
        {
            float tangentImpulse = min(tangentSpeed / tangentDenominator,
                body.friction * normalImpulse);
            Vec3 friction = multiply(tangent, -tangentImpulse);
            impulse = add(impulse, friction);
            velocity = add(velocity,
                multiply(friction, 1.0f / particle_mass));
        }
    }
    velocities[particle] = velocity;
    foam[particle] = max(foam[particle], min(1.0f, -incoming * 0.35f));
    impulses[particle] = {multiply(impulse, -1.0f),
        multiply(cross(arm, impulse), -1.0f), bestBody};
}

[shader("compute")]
[numthreads(128, 1, 1)]
void reduce_point_body_impulses(
    uniform FluidBodyImpulse* impulses, uniform uint* particle_count,
    uniform BodyParameters* parameters, uniform RigidBodyState* states,
    uniform uint* contact_flags, uniform uint body_count,
    uniform Vec3* position_corrections)
{
    uint body = cudaBlockIdx().x;
    if (body >= body_count || contact_flags[body] == 0u ||
        parameters[body].motion != MotionType.dynamic) return;
    uint lane = cudaThreadIdx().x;
    Vec3 localLinear = {}, localAngular = {}, localPosition = {};
    for (uint particle = lane; particle < *particle_count;
         particle += cudaBlockDim().x)
    {
        FluidBodyImpulse impulse = impulses[particle];
        if (impulse.body != body) continue;
        localLinear = add(localLinear, impulse.linear);
        localAngular = add(localAngular, impulse.angular);
        if (position_corrections != nullptr)
            localPosition = add(localPosition, position_corrections[particle]);
    }
    fluid_impulse_linear[lane] = localLinear;
    fluid_impulse_angular[lane] = localAngular;
    fluid_impulse_position[lane] = localPosition;
    pmGroupBarrier();
    for (uint stride = cudaBlockDim().x / 2u; stride != 0u; stride /= 2u)
    {
        if (lane < stride)
        {
            fluid_impulse_linear[lane] = add(fluid_impulse_linear[lane],
                fluid_impulse_linear[lane + stride]);
            fluid_impulse_angular[lane] = add(fluid_impulse_angular[lane],
                fluid_impulse_angular[lane + stride]);
            fluid_impulse_position[lane] = add(fluid_impulse_position[lane],
                fluid_impulse_position[lane + stride]);
        }
        pmGroupBarrier();
    }
    if (lane == 0u)
    {
        RigidBodyState state = states[body];
        BodyParameters options = parameters[body];
        state.position = add(state.position, fluid_impulse_position[0]);
        state.linear_velocity = clamp_length(add(state.linear_velocity,
            multiply(fluid_impulse_linear[0], options.inverse_mass)),
            options.maximum_linear_speed);
        state.angular_velocity = clamp_length(add(state.angular_velocity,
            inverse_inertia_world(options, state, fluid_impulse_angular[0])),
            options.maximum_angular_speed);
        states[body] = state;
    }
}
