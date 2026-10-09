// SPDX-License-Identifier: MIT
// Slang mirror of src/geometry_rope.cuh. Core geometry/solver kernels are
// defined below the shared rope ABI helpers as this migration advances.

import cuda_common;

public int rope_anchor_body(RopeData rope, uint node, int first, int last)
{
    return node == 0u ? first : (node + 1u == rope.count ? last : -1);
}

public bool rope_soft_anchor(RopeData rope, uint node)
{
    return (node == 0u && rope.soft_first) ||
           (node + 1u == rope.count && rope.soft_last);
}

public uint rope_soft_end(RopeData rope, uint node)
{
    return node == 0u ? 0u : 1u;
}

public Vec3 rope_anchor_local(RopeData rope, uint node)
{
    return node == 0u
        ? rope.options.first.local_anchor : rope.options.last.local_anchor;
}

float rope_node_weight(RopeData rope, uint node, int first, int last)
{
    if (rope_anchor_body(rope, node, first, last) >= 0) return 0.0f;
    if (rope_soft_anchor(rope, node))
        return rope.soft_anchor_inverse_masses[rope_soft_end(rope, node)];
    return float(rope.count) / rope.options.mass;
}

void rope_sync_anchors(
    RopeData rope, int first, int last, RigidBodyState* states)
{
    if (first >= 0)
        rope.positions[0] = transform_point(
            states[first], rope.options.first.local_anchor);
    if (last >= 0)
        rope.positions[rope.count - 1u] = transform_point(
            states[last], rope.options.last.local_anchor);
    if (rope.soft_first) rope.positions[0] = rope.soft_anchor_positions[0];
    if (rope.soft_last)
        rope.positions[rope.count - 1u] = rope.soft_anchor_positions[1];
}

public void closest_segments(
    Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2, out Vec3 c1, out Vec3 c2)
{
    Vec3 d1 = subtract(q1, p1);
    Vec3 d2 = subtract(q2, p2);
    Vec3 r = subtract(p1, p2);
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    float s = 0.0f;
    float t = 0.0f;
    float squaredEpsilon = kEpsilon * kEpsilon;
    if (a <= squaredEpsilon && e <= squaredEpsilon)
    {
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= squaredEpsilon)
        t = clamp_scalar(f / e, 0.0f, 1.0f);
    else
    {
        float c = dot(d1, r);
        if (e <= squaredEpsilon)
            s = clamp_scalar(-c / a, 0.0f, 1.0f);
        else
        {
            float b = dot(d1, d2);
            float denominator = a * e - b * b;
            if (abs(denominator) > kEpsilon * a * e)
                s = clamp_scalar((b * f - c * e) / denominator, 0.0f, 1.0f);
            t = (b * s + f) / e;
            if (t < 0.0f)
            {
                t = 0.0f;
                s = clamp_scalar(-c / a, 0.0f, 1.0f);
            }
            else if (t > 1.0f)
            {
                t = 1.0f;
                s = clamp_scalar((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    c1 = add(p1, multiply(d1, s));
    c2 = add(p2, multiply(d2, t));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rope_advance(
    uniform RopeData rope, uniform float dt, uniform Vec3 gravity,
    uniform int first, uniform int last, uniform BodyParameters* parameters,
    uniform RigidBodyState* states, uniform RigidBodyState* old_states,
    uniform TriangleMeshResource* meshes, uniform uint body_count,
    uniform bool first_substep, uniform RopeSoftTarget soft_a,
    uniform RopeSoftTarget soft_b)
{
    uint lane = cudaThreadIdx().x;
    if (lane == 0u)
        rope_sync_anchors(rope, first, last,
            old_states != nullptr ? old_states : states);
    pmGroupBarrier();
    for (uint node = lane; node < rope.count; node += cudaBlockDim().x)
    {
        rope.previous[node] = rope.positions[node];
        rope.normals[node] = {};
        rope.normals2[node] = {};
        if (first_substep)
        {
            rope.constraint_forces[node] = {};
            rope.contact_forces[node] = {};
            rope.fluid_contact_forces[node] = {};
            rope.soft_body_contact_forces[node] = {};
        }
        if (rope_anchor_body(rope, node, first, last) < 0 &&
            !rope_soft_anchor(rope, node))
        {
            rope.velocities[node] = clamp_length(multiply(add(
                rope.velocities[node], multiply(gravity, dt)),
                exp(-rope.options.velocity_damping * dt)),
                rope.options.maximum_speed);
            rope.positions[node] = add(rope.positions[node],
                multiply(rope.velocities[node], dt));
        }
        if (node + 1u < rope.count) rope.lambda[node] = 0.0f;
    }
    pmGroupBarrier();
    if (lane == 0u) rope_sync_anchors(rope, first, last, states);
    pmGroupBarrier();
    uint iterations = min(32u, max(1u, 4u * rope.options.solver_iterations));
    for (uint iteration = 0u; iteration < iterations; ++iteration)
    {
        for (uint phase = 0u; phase < 2u; ++phase)
        {
            for (uint edge = 2u * lane + phase; edge + 1u < rope.count;
                 edge += 2u * cudaBlockDim().x)
            {
                uint a = edge, b = edge + 1u;
                Vec3 delta = subtract(rope.positions[b], rope.positions[a]);
                float length = vector_length(delta);
                if (length <= 1.0e-8f) continue;
                float wa = rope_node_weight(rope, a, first, last);
                float wb = rope_node_weight(rope, b, first, last);
                float compliance = rope.options.stretch_compliance /
                    (dt * dt);
                float denominator = wa + wb + compliance;
                if (denominator <= 1.0e-12f) continue;
                float correction = (length - rope.rest[edge] +
                    compliance * rope.lambda[edge]) / denominator;
                rope.lambda[edge] -= correction;
                Vec3 impulse = multiply(delta, correction / length);
                if (wa > 0.0f)
                {
                    Vec3 movement = multiply(impulse, wa);
                    rope.positions[a] = add(rope.positions[a], movement);
                    rope.constraint_forces[a] = add(
                        rope.constraint_forces[a], multiply(movement, 1.0f /
                            (dt * dt * max(wa, 1.0e-12f))));
                }
                if (wb > 0.0f)
                {
                    Vec3 movement = multiply(impulse, -wb);
                    rope.positions[b] = add(rope.positions[b], movement);
                    rope.constraint_forces[b] = add(
                        rope.constraint_forces[b], multiply(movement, 1.0f /
                            (dt * dt * max(wb, 1.0e-12f))));
                }
            }
            pmGroupBarrier();
            if (lane == 0u) rope_sync_anchors(rope, first, last, states);
            pmGroupBarrier();
        }
        if (rope.options.self_collision)
        {
            for (uint node = lane; node < rope.count;
                 node += cudaBlockDim().x)
            {
                Vec3 correction = {};
                uint hits = 0u;
                float wi = rope_node_weight(rope, node, first, last);
                for (uint other = 0u; other < rope.count && wi > 0.0f;
                     ++other)
                {
                    if (abs(int(node) - int(other)) <= 2) continue;
                    Vec3 delta = subtract(
                        rope.positions[node], rope.positions[other]);
                    float length = vector_length(delta);
                    float wj = rope_node_weight(rope, other, first, last);
                    if (length >= 2.0f * rope.options.radius ||
                        wi + wj <= 0.0f) continue;
                    correction = add(correction, multiply(normalized_or(
                        delta, {1.0f, 0.0f, 0.0f}),
                        (2.0f * rope.options.radius - length) * wi /
                            (wi + wj)));
                    ++hits;
                }
                rope.scratch[node] = {};
                if (hits != 0u)
                    rope.scratch[node] = multiply(
                        correction, 1.0f / float(hits));
            }
            pmGroupBarrier();
            for (uint node = lane; node < rope.count;
                 node += cudaBlockDim().x)
                rope.positions[node] = add(
                    rope.positions[node], rope.scratch[node]);
            pmGroupBarrier();
        }
    }
    for (uint node = lane; node < rope.count; node += cudaBlockDim().x)
    {
        int body = rope_anchor_body(rope, node, first, last);
        if (rope_soft_anchor(rope, node))
        {
            uint end = rope_soft_end(rope, node);
            rope.velocities[node] = rope.soft_anchor_inverse_masses[end] > 0.0f
                ? clamp_length(multiply(subtract(
                    rope.positions[node], rope.previous[node]), 1.0f / dt),
                    rope.options.maximum_speed)
                : rope.soft_anchor_velocities[end];
        }
        else if (body < 0)
            rope.velocities[node] = clamp_length(multiply(subtract(
                rope.positions[node], rope.previous[node]), 1.0f / dt),
                rope.options.maximum_speed);
        else
        {
            Vec3 arm = rotate(states[body].orientation,
                rope_anchor_local(rope, node));
            rope.velocities[node] = add(states[body].linear_velocity,
                cross(states[body].angular_velocity, arm));
        }
    }
}
