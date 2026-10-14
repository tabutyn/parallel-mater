// SPDX-License-Identifier: MIT
// Slang-backed Metal compatibility adapter for the deterministic rope solve.

import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3> positions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3> previous : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3> velocities : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<float> inverse_masses : register(t3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalBond> bonds : register(u4);
[[vk::binding(5, 0)]] ConstantBuffer<MetalDeformableConstants> ropeStepConstants : register(b5);
[[vk::binding(6, 0)]] ConstantBuffer<MetalRopeAttachmentConstants> ropeStepAttachments : register(b6);
[[vk::binding(7, 0)]] StructuredBuffer<MetalHandle> rigid_ids : register(t7);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalRigidState> rigid_states : register(u8);
[[vk::binding(9, 0)]] RWStructuredBuffer<MetalPackedVec3> directions : register(u9);
[[vk::binding(10, 0)]] RWStructuredBuffer<float> diagonal : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<float> upper : register(u11);
[[vk::binding(12, 0)]] RWStructuredBuffer<float> rhs : register(u12);
[[vk::binding(13, 0)]] RWStructuredBuffer<float> lambdas : register(u13);
[[vk::binding(14, 0)]] RWStructuredBuffer<MetalPackedVec3> scratch : register(u14);
[[vk::binding(15, 0)]] RWStructuredBuffer<MetalPackedVec3> constraint_forces : register(u15);
[[vk::binding(16, 0)]] RWStructuredBuffer<MetalRigidParameters> rigid_parameters : register(u16);
[[vk::binding(17, 0)]] RWStructuredBuffer<MetalRopeAnchorState> soft_anchors : register(u17);
[[vk::binding(18, 0)]] RWStructuredBuffer<MetalPackedVec3> contact_forces : register(u18);
[[vk::binding(19, 0)]] RWStructuredBuffer<MetalPackedVec3> mesh_vertices : register(u19);
[[vk::binding(20, 0)]] RWStructuredBuffer<uint> mesh_indices : register(u20);
[[vk::binding(21, 0)]] StructuredBuffer<MetalTriangleMeshInfo> meshes : register(t21);
[[vk::binding(22, 0)]] RWStructuredBuffer<MetalRigidState> old_rigid_states : register(u22);
[[vk::binding(23, 0)]] RWStructuredBuffer<MetalPackedVec3> contact_normals : register(u23);
[[vk::binding(24, 0)]] RWStructuredBuffer<MetalPackedVec3> contact_normals2 : register(u24);
[[vk::binding(25, 0)]] RWStructuredBuffer<MetalPackedVec3> body_translation : register(u25);
[[vk::binding(26, 0)]] RWStructuredBuffer<uint> body_rotation : register(u26);
[[vk::binding(27, 0)]] RWStructuredBuffer<uint> soft_target_a : register(u27);
[[vk::binding(28, 0)]] RWStructuredBuffer<uint> soft_target_b : register(u28);
[[vk::binding(29, 0)]] RWStructuredBuffer<MetalPackedVec3> rope_soft_forces : register(u29);
[[vk::binding(30, 0)]] StructuredBuffer<MetalCollisionPlane> solid_planes : register(t30);

static float3 pm_rope_load_body_rotation(
    RWStructuredBuffer<uint> values, uint body) {
    uint base = 3u * body;
    return float3(asfloat(values[base]), asfloat(values[base + 1u]),
        asfloat(values[base + 2u]));
}

static void pm_rope_store_body_rotation(
    RWStructuredBuffer<uint> values, uint body, float3 rotation) {
    uint base = 3u * body;
    values[base] = asuint(rotation.x);
    values[base + 1u] = asuint(rotation.y);
    values[base + 2u] = asuint(rotation.z);
}

static float3 pm_triangle_weights(float3 point, float3 a, float3 b,
                                  float3 c) {
    const float3 first = b - a;
    const float3 second = c - a;
    const float3 offset = point - a;
    const float d00 = dot(first, first);
    const float d01 = dot(first, second);
    const float d11 = dot(second, second);
    const float d20 = dot(offset, first);
    const float d21 = dot(offset, second);
    const float denominator = d00 * d11 - d01 * d01;
    if (abs(denominator) <= 1.0e-14f) return float3(1.0f, 0.0f, 0.0f);
    const float second_weight = (d11 * d20 - d01 * d21) / denominator;
    const float third_weight = (d00 * d21 - d01 * d20) / denominator;
    float3 weights = max(
        float3(1.0f - second_weight - third_weight, second_weight,
               third_weight),
        0.0f);
    return weights / max(weights.x + weights.y + weights.z, 1.0e-12f);
}

static void pm_closest_segments(float3 first_a, float3 first_b,
                                float3 second_a, float3 second_b,
                                out float first_fraction,
                                out float second_fraction,
                                out float3 first_point,
                                out float3 second_point) {
    const float3 first_axis = first_b - first_a;
    const float3 second_axis = second_b - second_a;
    const float3 offset = first_a - second_a;
    const float first_size = dot(first_axis, first_axis);
    const float second_size = dot(second_axis, second_axis);
    const float second_offset = dot(second_axis, offset);
    if (first_size <= 1.0e-12f && second_size <= 1.0e-12f) {
        first_fraction = 0.0f;
        second_fraction = 0.0f;
    } else if (first_size <= 1.0e-12f) {
        first_fraction = 0.0f;
        second_fraction = clamp(second_offset / second_size, 0.0f, 1.0f);
    } else {
        const float first_offset = dot(first_axis, offset);
        if (second_size <= 1.0e-12f) {
            second_fraction = 0.0f;
            first_fraction = clamp(-first_offset / first_size, 0.0f, 1.0f);
        } else {
            const float axes = dot(first_axis, second_axis);
            const float denominator = first_size * second_size - axes * axes;
            first_fraction = abs(denominator) >
                                     1.0e-6f * first_size * second_size
                                 ? clamp((axes * second_offset -
                                          first_offset * second_size) /
                                             denominator,
                                         0.0f, 1.0f)
                                 : 0.0f;
            second_fraction =
                (axes * first_fraction + second_offset) / second_size;
            if (second_fraction < 0.0f) {
                second_fraction = 0.0f;
                first_fraction =
                    clamp(-first_offset / first_size, 0.0f, 1.0f);
            } else if (second_fraction > 1.0f) {
                second_fraction = 1.0f;
                first_fraction = clamp(
                    (axes - first_offset) / first_size, 0.0f, 1.0f);
            }
        }
    }
    first_point = first_a + first_axis * first_fraction;
    second_point = second_a + second_axis * second_fraction;
}

static void pm_closest_segment_triangle(
    float3 segment_a, float3 segment_b, float3 a, float3 b, float3 c,
    out float segment_fraction, out float3 segment_point,
    out float3 triangle_point, out float3 triangle_weights) {
    float best_squared = 3.402823466e+38f;
    const float3 face = cross(b - a, c - a);
    const float plane_denominator = dot(face, segment_b - segment_a);
    const float face_squared = dot(face, face);
    if (face_squared > 1.0e-12f &&
        abs(plane_denominator) > 1.0e-6f) {
        const float fraction =
            dot(face, a - segment_a) / plane_denominator;
        if (fraction >= 0.0f && fraction <= 1.0f) {
            const float3 hit = segment_a +
                               (segment_b - segment_a) * fraction;
            const float tolerance = -1.0e-5f * face_squared;
            const bool inside =
                dot(cross(b - a, hit - a), face) >= tolerance &&
                dot(cross(c - b, hit - b), face) >= tolerance &&
                dot(cross(a - c, hit - c), face) >= tolerance;
            if (inside) {
                segment_fraction = fraction;
                segment_point = hit;
                triangle_point = hit;
                triangle_weights = pm_triangle_weights(hit, a, b, c);
                return;
            }
        }
    }
    const float3 endpoints[2] = {segment_a, segment_b};
    for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
        const float3 nearest =
            metalClosestPointTriangle(endpoints[endpoint], a, b, c);
        const float squared =
            dot(endpoints[endpoint] - nearest, endpoints[endpoint] - nearest);
        if (squared >= best_squared) continue;
        best_squared = squared;
        segment_fraction = float(endpoint);
        segment_point = endpoints[endpoint];
        triangle_point = nearest;
        triangle_weights = pm_triangle_weights(nearest, a, b, c);
    }
    const float3 edge_a[3] = {a, b, c};
    const float3 edge_b[3] = {b, c, a};
    for (uint edge = 0u; edge < 3u; ++edge) {
        float first_fraction = 0.0f;
        float second_fraction = 0.0f;
        float3 first_point = 0.0f;
        float3 second_point = 0.0f;
        pm_closest_segments(segment_a, segment_b, edge_a[edge], edge_b[edge],
                            first_fraction, second_fraction, first_point,
                            second_point);
        const float squared =
            dot(first_point - second_point, first_point - second_point);
        if (squared >= best_squared) continue;
        best_squared = squared;
        segment_fraction = first_fraction;
        segment_point = first_point;
        triangle_point = second_point;
        triangle_weights = pm_triangle_weights(second_point, a, b, c);
    }
}
static float3 metalNormalizedOr(float3 value, float3 fallback) {
    const float squared = dot(value, value);
    return squared > 1.0e-12f ? value * rsqrt(squared) : fallback;
}

static float3 pm_rigid_point_velocity(
    MetalRigidState state, float3 local_anchor) {
    const float3 arm = metalRotate(state.orientation, local_anchor);
    return metalLoad(state.linear_velocity) +
           cross(metalLoad(state.angular_velocity), arm);
}

static float pm_rigid_direction_weight(
    MetalRigidParameters body,
    MetalRigidState state, float3 local_anchor,
    float3 direction) {
    const float3 arm = metalRotate(state.orientation, local_anchor);
    const float3 torque = cross(arm, direction);
    return body.inverse_mass +
           dot(torque, metalInverseInertiaMul(body, state, torque));
}

static float3 pm_rigid_impulse_movement(
    MetalRigidParameters body,
    MetalRigidState state, float3 local_anchor,
    float3 impulse) {
    const float3 arm = metalRotate(state.orientation, local_anchor);
    return impulse * body.inverse_mass +
           cross(metalInverseInertiaMul(body, state, cross(arm, impulse)),
                 arm);
}

static void pm_move_rigid_by_rope(
    inout MetalRigidParameters body, inout MetalRigidState state,
    float3 local_anchor, float3 impulse, float timestep) {
    if (body.inverse_mass <= 0.0f) return;
    const float3 arm = metalRotate(state.orientation, local_anchor);
    const float3 translation = impulse * body.inverse_mass;
    const float3 rotation = metalInverseInertiaMul(
        body, state, cross(arm, impulse));
    state.position = metalStore(metalLoad(state.position) + translation);
    const MetalQuaternion spin = metalQuaternionMultiply(
        {rotation.x, rotation.y, rotation.z, 0.0f}, state.orientation);
    state.orientation = metalQuaternionNormalize(
        {fma(0.5f, spin.x, state.orientation.x),
         fma(0.5f, spin.y, state.orientation.y),
         fma(0.5f, spin.z, state.orientation.z),
         fma(0.5f, spin.w, state.orientation.w)});
    state.linear_velocity = metalStore(
        metalLoad(state.linear_velocity) + translation / timestep);
    state.angular_velocity = metalStore(
        metalLoad(state.angular_velocity) + rotation / timestep);
}

static float pm_rope_node_inverse(
    uint node, uint count, bool first_attached, bool last_attached,
    bool first_soft, bool last_soft, StructuredBuffer<float> inverse_masses,
    RWStructuredBuffer<MetalRopeAnchorState> soft_anchors) {
    if ((node == 0u && first_attached) ||
        (node + 1u == count && last_attached))
        return 0.0f;
    if (node == 0u && first_soft) return soft_anchors[0].inverse_mass;
    if (node + 1u == count && last_soft)
        return soft_anchors[1].inverse_mass;
    return inverse_masses[node];
}

static float3 pm_rope_projected_mass(
    uint node, float3 value, float inverse_mass,
    RWStructuredBuffer<MetalPackedVec3> contact_normals,
    RWStructuredBuffer<MetalPackedVec3> contact_normals2) {
    const float3 first = metalLoad(contact_normals[node]);
    const float3 second = metalLoad(contact_normals2[node]);
    if (dot(first, first) > 0.5f) {
        const float3 edge = cross(first, second);
        if (dot(edge, edge) > 1.0e-4f) {
            const float3 tangent = normalize(edge);
            value = tangent * dot(tangent, value);
        } else {
            value -= first * dot(first, value);
        }
    } else if (dot(second, second) > 0.5f) {
        value -= second * dot(second, value);
    }
    return value * inverse_mass;
}

static float pm_rope_direction_weight(
    uint node, uint count, float3 direction, bool first_attached,
    bool last_attached, bool first_soft, bool last_soft,
    uint first_body, uint last_body,
    MetalRopeAttachmentConstants attachments,
    StructuredBuffer<float> inverse_masses,
    RWStructuredBuffer<MetalRopeAnchorState> soft_anchors,
    RWStructuredBuffer<MetalRigidParameters> rigid_parameters,
    RWStructuredBuffer<MetalRigidState> rigid_states,
    RWStructuredBuffer<MetalPackedVec3> contact_normals,
    RWStructuredBuffer<MetalPackedVec3> contact_normals2) {
    if (node == 0u && first_attached)
        return pm_rigid_direction_weight(
            rigid_parameters[first_body], rigid_states[first_body],
            metalLoad(attachments.first_anchor), direction);
    if (node + 1u == count && last_attached)
        return pm_rigid_direction_weight(
            rigid_parameters[last_body], rigid_states[last_body],
            metalLoad(attachments.last_anchor), direction);
    const float inverse = pm_rope_node_inverse(
        node, count, first_attached, last_attached, first_soft, last_soft,
        inverse_masses, soft_anchors);
    return dot(direction, pm_rope_projected_mass(
                              node, direction, inverse, contact_normals,
                              contact_normals2));
}

static float pm_rope_contact_weight(
    uint node, uint count, float3 direction, bool first_attached,
    bool last_attached, bool first_soft, bool last_soft,
    uint first_body, uint last_body,
    MetalRopeAttachmentConstants attachments,
    StructuredBuffer<float> inverse_masses,
    RWStructuredBuffer<MetalRopeAnchorState> soft_anchors,
    RWStructuredBuffer<MetalRigidParameters> rigid_parameters,
    RWStructuredBuffer<MetalRigidState> rigid_states) {
    if (node == 0u && first_attached)
        return pm_rigid_direction_weight(
            rigid_parameters[first_body], rigid_states[first_body],
            metalLoad(attachments.first_anchor), direction);
    if (node + 1u == count && last_attached)
        return pm_rigid_direction_weight(
            rigid_parameters[last_body], rigid_states[last_body],
            metalLoad(attachments.last_anchor), direction);
    return pm_rope_node_inverse(
        node, count, first_attached, last_attached, first_soft, last_soft,
        inverse_masses, soft_anchors);
}

// Positional contacts use a frozen PSD support projector in both effective
// mass and movement. Separating impulses release unilateral support; the
// full-mass velocity and soft-contact systems above remain unchanged.
static uint pm_rope_contact_support(
    float3 impulse, float3 first, float3 second) {
    return ((dot(first, first) > 0.5f && dot(impulse, first) <= 0.0f) ? 1u : 0u) |
           ((dot(second, second) > 0.5f && dot(impulse, second) <= 0.0f) ? 2u : 0u);
}

static float3 pm_rope_contact_project(
    float3 value, uint support, float3 first, float3 second) {
    if (support == 0u) return value;
    const float3 normal = (support & 1u) != 0u ? first :
        ((support & 2u) != 0u ? second : float3(0.0f));
    const float3 edge = support == 3u ? cross(normal, second) : float3(0.0f);
    if (dot(edge, edge) > 1.0e-4f) {
        const float3 tangent = normalize(edge);
        return tangent * dot(tangent, value);
    }
    return value - normal * dot(normal, value);
}

struct MetalRopeContactMass {
    uint body;
    float inverse;
    float3 arm;
    float3 first_support;
    float3 second_support;
};

struct MetalRopeFriction {
    float normal;
    float tangent;
};

static MetalRopeFriction pm_rope_coupled_contact_friction(
    float normalImpulse, float normalMass, float crossMass,
    float tangentMass, float tangentResidual, float friction) {
    MetalRopeFriction result = {normalImpulse, 0.0f};
    if (!(normalImpulse > 0.0f) || !(normalMass > 1.0e-12f) ||
        !(tangentResidual > 0.0f) || !(friction > 0.0f)) return result;
    float coupling = crossMass / normalMass;
    float schur = tangentMass - crossMass * coupling;
    if (!(schur > 1.0e-12f)) return result;
    float magnitude = tangentResidual / schur;
    float coneDenominator = 1.0f - friction * coupling;
    if (coneDenominator > 0.0f)
        magnitude = min(magnitude,
            friction * normalImpulse / coneDenominator);
    result.normal += coupling * magnitude;
    result.tangent = -magnitude;
    return result;
}
static float3 pm_rope_position_contact_response(
    MetalRopeContactMass mass, uint support, float3 direction,
    RWStructuredBuffer<MetalRigidParameters> parameters,
    RWStructuredBuffer<MetalRigidState> states) {
    if (mass.body == 0xffffffffu)
        return pm_rope_contact_project(direction, support, mass.first_support,
            mass.second_support) * mass.inverse;
    return direction * parameters[mass.body].inverse_mass + cross(
        metalInverseInertiaMul(parameters[mass.body], states[mass.body],
            cross(mass.arm, direction)), mass.arm);
}
static float3 pm_rope_contact_pair_response(
    MetalRopeContactMass first, MetalRopeContactMass second, MetalRopeContactMass collider,
    uint support_a, uint support_b, float a, float b, float3 direction,
    RWStructuredBuffer<MetalRigidParameters> parameters,
    RWStructuredBuffer<MetalRigidState> states) {
    float3 response = pm_rope_position_contact_response(first, support_a,
        direction, parameters, states) * (a * a);
    if (b > 0.0f) response += pm_rope_position_contact_response(second, support_b,
        direction, parameters, states) * (b * b);
    return response + pm_rope_position_contact_response(collider, 0u,
        direction, parameters, states);
}

struct MetalRopeRigidHit {
    float depth;
    float fraction;
    float3 normal;
    float3 point;
    uint body;
    bool found;
};

static MetalRopeRigidHit pm_rope_find_rigid_contact(
    RWStructuredBuffer<MetalPackedVec3> positions,
    RWStructuredBuffer<MetalPackedVec3> previous, uint count, uint node,
    bool segment, float rope_radius, uint first_body, uint last_body,
    MetalRopeAttachmentConstants attachments,
    RWStructuredBuffer<MetalRigidState> rigid_states,
    RWStructuredBuffer<MetalRigidState> old_states,
    RWStructuredBuffer<MetalRigidParameters> rigid_parameters,
    RWStructuredBuffer<MetalPackedVec3> vertices, RWStructuredBuffer<uint> indices,
    StructuredBuffer<MetalTriangleMeshInfo> meshes,
    StructuredBuffer<MetalCollisionPlane> solid_planes,
    RWStructuredBuffer<uint> body_rotation) {
    MetalRopeRigidHit best;
    best.depth = 0.0f;
    best.fraction = 0.0f;
    best.normal = float3(0.0f);
    best.point = float3(0.0f);
    best.body = attachments.rigid_count;
    best.found = false;
    const uint next = segment ? node + 1u : node;
    for (uint body = 0u; body < attachments.rigid_count; ++body) {
        if ((body == first_body &&
             node < attachments.first_contact_skip) ||
            (body == last_body &&
             next + attachments.last_contact_skip >= count))
            continue;
        MetalRigidState state = rigid_states[body];
        MetalRigidState old_state = old_states[body];
        MetalRigidParameters parameters = rigid_parameters[body];
        MetalTriangleMeshInfo mesh =
            meshes[parameters.mesh_index];
        const float radius = rope_radius + parameters.collision_margin;
        const MetalQuaternion inverse =
            metalQuaternionConjugate(state.orientation);
        const MetalQuaternion old_inverse =
            metalQuaternionConjugate(old_state.orientation);
        const float3 first = metalRotate(
            inverse, metalLoad(positions[node]) - metalLoad(state.position));
        const float3 second = metalRotate(
            inverse, metalLoad(positions[next]) - metalLoad(state.position));
        const float3 origin = metalRotate(
            old_inverse,
            metalLoad(previous[node]) - metalLoad(old_state.position));
        const float3 lower = min(first, segment ? second : origin) - radius;
        const float3 upper = max(first, segment ? second : origin) + radius;
        if (any(upper < metalLoad(mesh.minimum)) ||
            any(lower > metalLoad(mesh.maximum)))
            continue;
        const bool solid = mesh.solid_plane_count != 0u;
        if (solid) {
            uint hintIndex = 3u * attachments.rigid_capacity +
                node * attachments.rigid_capacity + body;
            uint hint = body_rotation[hintIndex];
            const bool stationary =
                metalLoad(state.position).x == metalLoad(old_state.position).x &&
                metalLoad(state.position).y == metalLoad(old_state.position).y &&
                metalLoad(state.position).z == metalLoad(old_state.position).z &&
                state.orientation.x == old_state.orientation.x &&
                state.orientation.y == old_state.orientation.y &&
                state.orientation.z == old_state.orientation.z &&
                state.orientation.w == old_state.orientation.w;
            bool outside = false;
            bool separated = false;
            float side = -3.402823466e+38f;
            float entry_time = -1.0f;
            float3 nearest_normal = 0.0f;
            float nearest_offset = 0.0f;
            float3 entry_normal = 0.0f;
            float entry_offset = 0.0f;
            if (hint < mesh.solid_plane_count) {
                MetalCollisionPlane plane =
                    solid_planes[mesh.solid_plane_offset + hint];
                const float3 plane_normal = metalLoad(plane.normal);
                const float hint_side =
                    dot(plane_normal, first) - plane.offset;
                if (hint_side > 0.0f) {
                    outside = true;
                    separated = stationary && hint_side > radius &&
                        (segment
                             ? dot(plane_normal, second) - plane.offset >
                                   radius
                             : dot(plane_normal, origin) - plane.offset >
                                   radius);
                }
            }
            for (uint local = 0u;
                 !outside && local < mesh.solid_plane_count; ++local) {
                MetalCollisionPlane plane =
                    solid_planes[mesh.solid_plane_offset + local];
                const float3 plane_normal = metalLoad(plane.normal);
                const float current_side =
                    dot(plane_normal, first) - plane.offset;
                if (current_side > side) {
                    side = current_side;
                    nearest_normal = plane_normal;
                    nearest_offset = plane.offset;
                }
                if (!segment) {
                    const float before =
                        dot(plane_normal, origin) - plane.offset;
                    if (before > 0.0f && current_side <= 0.0f) {
                        const float time =
                            before / (before - current_side);
                        if (time > entry_time) {
                            entry_time = time;
                            entry_normal = plane_normal;
                            entry_offset = plane.offset;
                        }
                    }
                }
                if (current_side > 0.0f) {
                    hint = local;
                    body_rotation[hintIndex] = hint;
                    outside = true;
                    separated = stationary && current_side > radius &&
                        (segment
                             ? dot(plane_normal, second) - plane.offset >
                                   radius
                             : dot(plane_normal, origin) - plane.offset >
                                   radius);
                }
            }
            if (separated) continue;
            if (!outside && !segment) {
                hint = 0xffffffffu;
                body_rotation[hintIndex] = hint;
                if (entry_time >= 0.0f) {
                    nearest_normal = entry_normal;
                    nearest_offset = entry_offset;
                    side = dot(entry_normal, first) - entry_offset;
                }
                if (radius - side > best.depth) {
                    best.depth = radius - side;
                    best.fraction = 0.0f;
                    best.normal = metalRotate(
                        state.orientation, nearest_normal);
                    best.point = metalWorldPoint(
                        state, first - nearest_normal * side);
                    best.body = body;
                    best.found = true;
                }
                continue;
            }
        }
        uint2 triangle_ranges[64];
        uint pending_range_count = 1u;
        triangle_ranges[0] = uint2(0u, mesh.index_count / 3u);
        while (pending_range_count != 0u) {
            const uint2 range = triangle_ranges[--pending_range_count];
            if (range.y - range.x > 4u) {
                const uint middle = range.x + (range.y - range.x) / 2u;
                if (pending_range_count + 2u <= 64u) {
                    triangle_ranges[pending_range_count++] =
                        uint2(range.x, middle);
                    triangle_ranges[pending_range_count++] =
                        uint2(middle, range.y);
                }
                continue;
            }
            for (uint triangle = range.x; triangle < range.y; ++triangle) {
                const uint local = 3u * triangle;
                const float3 a = metalLoad(vertices[
                    mesh.vertex_offset + indices[mesh.index_offset + local]]);
                const float3 b = metalLoad(vertices[
                    mesh.vertex_offset +
                    indices[mesh.index_offset + local + 1u]]);
                const float3 c = metalLoad(vertices[
                    mesh.vertex_offset +
                    indices[mesh.index_offset + local + 2u]]);
                if (any(upper < min(a, min(b, c))) ||
                    any(lower > max(a, max(b, c))))
                    continue;
                float fraction = 0.0f;
                float3 rope_point = first;
                float3 triangle_point = 0.0f;
                float3 triangle_weights = 0.0f;
                if (segment) {
                    pm_closest_segment_triangle(
                        first, second, a, b, c, fraction, rope_point,
                        triangle_point, triangle_weights);
                    const float3 rope_edge = second - first;
                    fraction = clamp(
                        dot(rope_point - first, rope_edge) /
                            max(dot(rope_edge, rope_edge), 1.0e-12f),
                        0.0f, 1.0f);
                } else {
                    triangle_point =
                        metalClosestPointTriangle(first, a, b, c);
                }
                const float3 delta = rope_point - triangle_point;
                const float squared = dot(delta, delta);
                const float3 face_value = cross(b - a, c - a);
                const float3 face = solid
                    ? metalLoad(solid_planes[
                          mesh.solid_plane_offset + local / 3u].normal)
                    : metalNormalizedOr(
                          face_value, float3(0.0f, 1.0f, 0.0f));
                const float distance_value = sqrt(max(squared, 0.0f));
                float depth = radius - distance_value;
                float3 normal = distance_value > 1.0e-7f
                                    ? delta / distance_value
                                    : face * (dot(origin - a, face) >= 0.0f
                                                  ? 1.0f
                                                  : -1.0f);
                if (segment && solid && distance_value < radius &&
                    dot(delta, face) <= 1.0e-7f) {
                    normal = face;
                    depth = radius + distance_value;
                }
                if (!segment) {
                    const float before = dot(origin - a, face);
                    const float after = dot(first - a, face);
                    if (before * after < 0.0f) {
                        const float3 crossing = origin + (first - origin) *
                            (before / (before - after));
                        const float3 nearest =
                            metalClosestPointTriangle(crossing, a, b, c);
                        if (dot(crossing - nearest, crossing - nearest) <=
                            radius * radius) {
                            normal =
                                face * (before >= 0.0f ? 1.0f : -1.0f);
                            depth = max(depth, radius + abs(after));
                        }
                    }
                }
                if (depth <= best.depth) continue;
                best.depth = depth;
                best.fraction = fraction;
                best.normal = metalRotate(state.orientation, normal);
                best.point = metalWorldPoint(state, triangle_point);
                best.body = body;
                best.found = true;
            }
        }
    }
    return best;
}

static void pm_rope_accumulate_rigid_movement(
    uint body, float3 impulse, float3 local_point,
    RWStructuredBuffer<MetalRigidParameters> rigid_parameters,
    RWStructuredBuffer<MetalRigidState> rigid_states,
    RWStructuredBuffer<MetalPackedVec3> body_translation,
    RWStructuredBuffer<uint> body_rotation) {
    const float3 arm = metalRotate(rigid_states[body].orientation, local_point);
    body_translation[body] = metalStore(
        metalLoad(body_translation[body]) +
        impulse * rigid_parameters[body].inverse_mass);
    pm_rope_store_body_rotation(body_rotation, body,
        pm_rope_load_body_rotation(body_rotation, body) +
        metalInverseInertiaMul(rigid_parameters[body], rigid_states[body],
            cross(arm, impulse)));
}

static void pm_rope_accumulate_rigid_movement_at_arm(
    uint body, float3 impulse, float3 arm,
    RWStructuredBuffer<MetalRigidParameters> rigid_parameters,
    RWStructuredBuffer<MetalRigidState> rigid_states,
    RWStructuredBuffer<MetalPackedVec3> body_translation,
    RWStructuredBuffer<uint> body_rotation) {
    body_translation[body] = metalStore(
        metalLoad(body_translation[body]) +
        impulse * rigid_parameters[body].inverse_mass);
    pm_rope_store_body_rotation(body_rotation, body,
        pm_rope_load_body_rotation(body_rotation, body) +
        metalInverseInertiaMul(rigid_parameters[body], rigid_states[body],
            cross(arm, impulse)));
}

static void pm_rope_contact_move(
    uint node, uint count, float3 impulse, bool first_attached,
    bool last_attached, bool first_soft, bool last_soft,
    uint first_body, uint last_body,
    MetalRopeAttachmentConstants attachments,
    RWStructuredBuffer<MetalPackedVec3> positions, StructuredBuffer<float> inverse_masses,
    RWStructuredBuffer<MetalRopeAnchorState> soft_anchors,
    RWStructuredBuffer<MetalRigidParameters> rigid_parameters,
    RWStructuredBuffer<MetalRigidState> rigid_states,
    RWStructuredBuffer<MetalPackedVec3> body_translation,
    RWStructuredBuffer<uint> body_rotation) {
    if (node == 0u && first_attached) {
        pm_rope_accumulate_rigid_movement(
            first_body, impulse, metalLoad(attachments.first_anchor),
            rigid_parameters, rigid_states, body_translation, body_rotation);
        return;
    }
    if (node + 1u == count && last_attached) {
        pm_rope_accumulate_rigid_movement(
            last_body, impulse, metalLoad(attachments.last_anchor),
            rigid_parameters, rigid_states, body_translation, body_rotation);
        return;
    }
    const float inverse = pm_rope_node_inverse(
        node, count, first_attached, last_attached, first_soft, last_soft,
        inverse_masses, soft_anchors);
    positions[node] = metalStore(metalLoad(positions[node]) + impulse * inverse);
}

static const uint pm_rope_soft_valid = 0u;
static const uint pm_rope_soft_index_count = 3u;
static const uint pm_rope_soft_surface_offset = 4u;
static const uint pm_rope_soft_previous_offset = 5u;
static const uint pm_rope_soft_velocity_offset = 6u;
static const uint pm_rope_soft_inverse_offset = 7u;
static const uint pm_rope_soft_index_offset = 8u;
static const uint pm_rope_soft_binding_offset = 9u;
static const uint pm_rope_soft_impulse_offset = 10u;
static const uint pm_rope_soft_contact_count = 11u;
static const uint pm_rope_soft_maximum_penetration = 12u;
static const uint pm_rope_soft_attach_first = 13u;
static const uint pm_rope_soft_attach_last = 14u;
static const uint pm_rope_soft_distance = 15u;
static const uint pm_rope_soft_friction = 16u;
static const uint pm_rope_soft_orientation = 17u;

static float pm_rope_soft_load_float(RWStructuredBuffer<uint> packed, uint word) {
    return asfloat(packed[word]);
}

static float3 pm_rope_soft_load_vec3(RWStructuredBuffer<uint> packed, uint word) {
    return float3(asfloat(packed[word]),
                  asfloat(packed[word + 1u]),
                  asfloat(packed[word + 2u]));
}

static void pm_rope_soft_store_vec3(RWStructuredBuffer<uint> packed, uint word,
                                    float3 value) {
    packed[word] = asuint(value.x);
    packed[word + 1u] = asuint(value.y);
    packed[word + 2u] = asuint(value.z);
}

static void pm_rope_soft_add_vec3(RWStructuredBuffer<uint> packed, uint word,
                                  float3 value) {
    pm_rope_soft_store_vec3(
        packed, word, pm_rope_soft_load_vec3(packed, word) + value);
}

static bool pm_rope_soft_inside_packed(
    float3 point, RWStructuredBuffer<uint> packed, uint surface_offset,
    uint index_offset, uint index_count) {
    const float3 direction = float3(1.0f, 0.371f, 0.173f);
    int winding = 0;
    bool ambiguous = false;
    for (uint base = 0u; base + 2u < index_count; base += 3u) {
        const uint ia = packed[index_offset + base];
        const uint ib = packed[index_offset + base + 1u];
        const uint ic = packed[index_offset + base + 2u];
        const float3 a = pm_rope_soft_load_vec3(
            packed, surface_offset + 3u * ia);
        const float3 edge1 = pm_rope_soft_load_vec3(
            packed, surface_offset + 3u * ib) - a;
        const float3 edge2 = pm_rope_soft_load_vec3(
            packed, surface_offset + 3u * ic) - a;
        const float3 h = cross(direction, edge2);
        const float3 s = point - a;
        const float determinant = dot(edge1, h);
        if (abs(determinant) < 1.0e-12f) continue;
        const float u = dot(s, h) / determinant;
        const float3 q = cross(s, edge1);
        const float v = dot(direction, q) / determinant;
        const float t = dot(edge2, q) / determinant;
        if (t < 0.0f || u < -1.0e-5f || v < -1.0e-5f ||
            u + v > 1.00001f)
            continue;
        if (u < 1.0e-5f || v < 1.0e-5f ||
            u + v > 0.99999f || t < 1.0e-7f)
            ambiguous = true;
        winding += determinant < 0.0f ? 1 : -1;
    }
    if (!ambiguous) return winding != 0;
    float angle = 0.0f;
    for (uint base = 0u; base + 2u < index_count; base += 3u) {
        const uint ia = packed[index_offset + base];
        const uint ib = packed[index_offset + base + 1u];
        const uint ic = packed[index_offset + base + 2u];
        const float3 a = pm_rope_soft_load_vec3(
            packed, surface_offset + 3u * ia) - point;
        const float3 b = pm_rope_soft_load_vec3(
            packed, surface_offset + 3u * ib) - point;
        const float3 c = pm_rope_soft_load_vec3(
            packed, surface_offset + 3u * ic) - point;
        const float la = length(a);
        const float lb = length(b);
        const float lc = length(c);
        angle += 2.0f * atan2(
            dot(a, cross(b, c)),
            la * lb * lc + dot(a, b) * lc + dot(b, c) * la +
                dot(c, a) * lb);
    }
    return abs(angle) > 6.2831853f;
}

// Match fluid_closest_triangle_barycentric in the CUDA backend. Returning
// the closest point and its weights from the same region tests avoids a
// second projection whose rounding can move reaction weight between soft
// nodes at triangle edges.
static float3 pm_closest_point_triangle_weights(
    float3 point, float3 a, float3 b, float3 c,
    out float3 weights) {
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = point - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        weights = float3(1.0f, 0.0f, 0.0f);
        return a;
    }
    const float3 bp = point - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        weights = float3(0.0f, 1.0f, 0.0f);
        return b;
    }
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        weights = float3(1.0f - v, v, 0.0f);
        return a + ab * v;
    }
    const float3 cp = point - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        weights = float3(0.0f, 0.0f, 1.0f);
        return c;
    }
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        weights = float3(1.0f - w, 0.0f, w);
        return a + ac * w;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        weights = float3(0.0f, 1.0f - w, w);
        return b + (c - b) * w;
    }
    const float denominator = va + vb + vc;
    if (denominator <= 1.0e-12f) {
        weights = float3(1.0f, 0.0f, 0.0f);
        return a;
    }
    const float inverse = 1.0f / denominator;
    const float v = vb * inverse;
    const float w = vc * inverse;
    weights = float3(1.0f - v - w, v, w);
    return a + ab * v + ac * w;
}

static void pm_rope_soft_contact_packed(
    RWStructuredBuffer<uint> packed, uint node, bool segment, uint rope_count,
    float timestep, bool first_attached, bool last_attached,
    bool first_soft, bool last_soft, uint first_body, uint last_body,
    MetalRopeAttachmentConstants attachments,
    RWStructuredBuffer<MetalPackedVec3> positions, RWStructuredBuffer<MetalPackedVec3> previous,
    StructuredBuffer<float> inverse_masses,
    RWStructuredBuffer<MetalRopeAnchorState> soft_anchors,
    RWStructuredBuffer<MetalRigidParameters> rigid_parameters,
    RWStructuredBuffer<MetalRigidState> rigid_states,
    RWStructuredBuffer<MetalPackedVec3> body_translation,
    RWStructuredBuffer<uint> body_rotation,
    RWStructuredBuffer<MetalPackedVec3> contact_normals,
    RWStructuredBuffer<MetalPackedVec3> contact_normals2,
    RWStructuredBuffer<MetalPackedVec3> rope_soft_forces) {
    if (packed[pm_rope_soft_valid] == 0u) return;
    if ((packed[pm_rope_soft_attach_first] != 0u && node == 0u) ||
        (packed[pm_rope_soft_attach_last] != 0u &&
         node + (segment ? 2u : 1u) == rope_count))
        return;
    const uint next = segment ? node + 1u : node;
    const uint surface_offset = packed[pm_rope_soft_surface_offset];
    const uint previous_offset = packed[pm_rope_soft_previous_offset];
    const uint velocity_offset = packed[pm_rope_soft_velocity_offset];
    const uint inverse_offset = packed[pm_rope_soft_inverse_offset];
    const uint index_offset = packed[pm_rope_soft_index_offset];
    const uint binding_offset = packed[pm_rope_soft_binding_offset];
    const uint impulse_offset = packed[pm_rope_soft_impulse_offset];
    const uint index_count = packed[pm_rope_soft_index_count];
    const float radius =
        pm_rope_soft_load_float(packed, pm_rope_soft_distance);
    const float orientation =
        pm_rope_soft_load_float(packed, pm_rope_soft_orientation);
    const float3 first = metalLoad(positions[node]);
    const float3 second = metalLoad(positions[next]);
    const float3 old = metalLoad(previous[node]);
    const float3 query_lower =
        min(first, segment ? second : old) - radius;
    const float3 query_upper =
        max(first, segment ? second : old) + radius;
    float best_depth = 0.0f;
    float best_fraction = 0.0f;
    float3 best_normal = 0.0f;
    float3 best_weights = 0.0f;
    float3 best_surface_point = 0.0f;
    float nearest_squared = 3.402823466e+38f;
    float earliest = 2.0f;
    uint best_base = index_count;
    for (uint base = 0u; base + 2u < index_count; base += 3u) {
        const uint ia = packed[index_offset + base];
        const uint ib = packed[index_offset + base + 1u];
        const uint ic = packed[index_offset + base + 2u];
        const float3 a = pm_rope_soft_load_vec3(
            packed, surface_offset + 3u * ia);
        const float3 b = pm_rope_soft_load_vec3(
            packed, surface_offset + 3u * ib);
        const float3 c = pm_rope_soft_load_vec3(
            packed, surface_offset + 3u * ic);
        if (any(query_upper < min(a, min(b, c))) ||
            any(query_lower > max(a, max(b, c))))
            continue;
        const float3 face_value = cross(b - a, c - a);
        if (dot(face_value, face_value) <= 1.0e-14f) continue;
        const float3 face = normalize(face_value) * orientation;
        float fraction = 0.0f;
        float3 rope_point = first;
        float3 surface_point = 0.0f;
        float3 weights = 0.0f;
        if (segment) {
            pm_closest_segment_triangle(first, second, a, b, c,
                                        fraction, rope_point,
                                        surface_point, weights);
            surface_point = pm_closest_point_triangle_weights(
                surface_point, a, b, c, weights);
            const float3 rope_edge = second - first;
            fraction = clamp(
                dot(rope_point - first, rope_edge) /
                    max(dot(rope_edge, rope_edge), 1.0e-12f),
                0.0f, 1.0f);
            const float3 delta = rope_point - surface_point;
            const float distance_value = length(delta);
            const float side = dot(delta, face);
            float3 normal = distance_value > 1.0e-7f
                                ? delta / distance_value
                                : face;
            float depth = radius - distance_value;
            if (side < -1.0e-7f) {
                normal = face;
                depth = radius + distance_value;
            }
            if (depth <= best_depth) continue;
            best_depth = depth;
            best_fraction = fraction;
            best_normal = normal;
            best_weights = weights;
            best_surface_point = surface_point;
            best_base = base;
            continue;
        }

        // Match CUDA's closed-skin nearest/swept query for rope nodes. A
        // maximum-penetration search can select the far wall of a thin body.
        surface_point = pm_closest_point_triangle_weights(
            first, a, b, c, weights);
        const float squared = dot(first - surface_point,
                                  first - surface_point);
        if (earliest > 1.0f && squared < nearest_squared) {
            nearest_squared = squared;
            best_depth = radius - dot(first - surface_point, face);
            best_fraction = 0.0f;
            best_normal = face;
            best_weights = weights;
            best_surface_point = surface_point;
            best_base = base;
        }
        const float3 old_surface_point =
            pm_rope_soft_load_vec3(packed, previous_offset + 3u * ia) *
                weights.x +
            pm_rope_soft_load_vec3(packed, previous_offset + 3u * ib) *
                weights.y +
            pm_rope_soft_load_vec3(packed, previous_offset + 3u * ic) *
                weights.z;
        const float3 transported_start =
            old + (surface_point - old_surface_point);
        const float before = dot(transported_start - a, face);
        const float after = dot(first - a, face);
        if (before < radius || after >= radius ||
            before - after < 1.0e-8f)
            continue;
        const float time = (before - radius) / (before - after);
        if (time >= earliest) continue;
        const float3 crossing =
            transported_start + (first - transported_start) * time -
            face * radius;
        float3 hit_weights = 0.0f;
        const float3 hit = pm_closest_point_triangle_weights(
            crossing, a, b, c, hit_weights);
        if (dot(crossing - hit, crossing - hit) > 1.0e-8f) continue;
        earliest = time;
        best_depth = radius - after;
        best_fraction = 0.0f;
        best_normal = face;
        best_weights = hit_weights;
        best_surface_point = hit;
        best_base = base;
    }
    if (best_base == index_count) return;
    if (!segment && earliest > 1.0f) {
        const float nearest = sqrt(nearest_squared);
        const bool face_interior = best_weights.x > 1.0e-4f &&
            best_weights.y > 1.0e-4f && best_weights.z > 1.0e-4f;
        const bool inside = face_interior
            ? dot(first - best_surface_point, best_normal) < 0.0f
            : pm_rope_soft_inside_packed(
                  first, packed, surface_offset, index_offset, index_count);
        if (!inside) {
            best_depth = radius - nearest;
            if (nearest > 1.0e-7f)
                best_normal = (first - best_surface_point) / nearest;
        } else {
            best_depth = radius + nearest;
            if (nearest > 1.0e-7f)
                best_normal = (best_surface_point - first) / nearest;
        }
    }
    if (best_depth <= 0.0f) return;
    const float first_fraction = 1.0f - best_fraction;
    const float second_fraction = best_fraction;
    const float first_weight = pm_rope_contact_weight(
        node, rope_count, best_normal, first_attached, last_attached,
        first_soft, last_soft, first_body, last_body, attachments,
        inverse_masses, soft_anchors, rigid_parameters, rigid_states);
    const float second_weight = segment
        ? pm_rope_contact_weight(
              next, rope_count, best_normal, first_attached, last_attached,
              first_soft, last_soft, first_body, last_body, attachments,
              inverse_masses, soft_anchors, rigid_parameters, rigid_states)
        : 0.0f;
    const float denominator = first_weight * first_fraction * first_fraction +
                              second_weight * second_fraction * second_fraction;
    if (denominator <= 1.0e-12f) return;
    float3 surface_velocity = 0.0f;
    float soft_weight = 0.0f;
    const float corner_weights[3] = {
        best_weights.x, best_weights.y, best_weights.z};
    for (uint corner = 0u; corner < 3u; ++corner) {
        const uint surface = packed[index_offset + best_base + corner];
        const uint binding = binding_offset + 8u * surface;
        for (uint slot = 0u; slot < 4u; ++slot) {
            const uint soft_node = packed[binding + slot];
            const float weight = corner_weights[corner] *
                pm_rope_soft_load_float(packed, binding + 4u + slot);
            const float inverse =
                pm_rope_soft_load_float(packed, inverse_offset + soft_node);
            soft_weight += weight * weight * inverse;
            surface_velocity += pm_rope_soft_load_vec3(
                packed, velocity_offset + 3u * soft_node) * weight;
        }
    }
    const float correction = best_depth / denominator;
    float3 impulse = best_normal * correction;
    const float3 rope_motion =
        (metalLoad(positions[node]) - metalLoad(previous[node])) *
            first_fraction +
        (metalLoad(positions[next]) - metalLoad(previous[next])) *
            second_fraction;
    const float3 relative_motion =
        rope_motion - surface_velocity * timestep;
    const float3 tangent = relative_motion -
        best_normal * dot(relative_motion, best_normal);
    const float tangent_length = length(tangent);
    if (tangent_length > 1.0e-8f) {
        const float3 direction = tangent / tangent_length;
        const float tangent_first = pm_rope_contact_weight(
            node, rope_count, direction, first_attached, last_attached,
            first_soft, last_soft, first_body, last_body, attachments,
            inverse_masses, soft_anchors, rigid_parameters, rigid_states);
        const float tangent_second = segment
            ? pm_rope_contact_weight(
                  next, rope_count, direction, first_attached,
                  last_attached, first_soft, last_soft, first_body,
                  last_body, attachments, inverse_masses, soft_anchors,
                  rigid_parameters, rigid_states)
            : 0.0f;
        const float tangent_denominator =
            tangent_first * first_fraction * first_fraction +
            tangent_second * second_fraction * second_fraction;
        const float friction =
            pm_rope_soft_load_float(packed, pm_rope_soft_friction);
        impulse -= direction * min(
            tangent_length / max(tangent_denominator, 1.0e-12f),
            friction * correction);
    }
    pm_rope_contact_move(
        node, rope_count, impulse * first_fraction, first_attached,
        last_attached, first_soft, last_soft, first_body, last_body,
        attachments, positions, inverse_masses, soft_anchors,
        rigid_parameters, rigid_states, body_translation, body_rotation);
    rope_soft_forces[node] = metalStore(
        metalLoad(rope_soft_forces[node]) +
        impulse * (first_fraction /
            max(timestep * timestep, 1.0e-12f)));
    if (segment) {
        pm_rope_contact_move(
            next, rope_count, impulse * second_fraction, first_attached,
            last_attached, first_soft, last_soft, first_body, last_body,
            attachments, positions, inverse_masses, soft_anchors,
            rigid_parameters, rigid_states, body_translation, body_rotation);
        rope_soft_forces[next] = metalStore(
            metalLoad(rope_soft_forces[next]) +
            impulse * (second_fraction /
                max(timestep * timestep, 1.0e-12f)));
    }
    if (soft_weight < 1.0e-9f) {
        for (uint item = node; item <= next; ++item) {
            const float3 first_normal = metalLoad(contact_normals[item]);
            if (dot(first_normal, first_normal) < 0.5f ||
                dot(first_normal, best_normal) > 0.95f)
                contact_normals[item] = metalStore(best_normal);
            else
                contact_normals2[item] = metalStore(best_normal);
        }
    }
    for (uint corner = 0u; corner < 3u; ++corner) {
        const uint surface = packed[index_offset + best_base + corner];
        const uint binding = binding_offset + 8u * surface;
        for (uint slot = 0u; slot < 4u; ++slot) {
            const uint soft_node = packed[binding + slot];
            const float weight = corner_weights[corner] *
                pm_rope_soft_load_float(packed, binding + 4u + slot);
            if (weight <= 0.0f) continue;
            pm_rope_soft_add_vec3(
                packed, impulse_offset + 3u * soft_node,
                -impulse * (weight / max(timestep, 1.0e-12f)));
        }
    }
    ++packed[pm_rope_soft_contact_count];
    const float maximum = pm_rope_soft_load_float(
        packed, pm_rope_soft_maximum_penetration);
    packed[pm_rope_soft_maximum_penetration] =
        asuint(max(maximum, best_depth));
}

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_rope_step_serial(
    uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint thread_index = dispatchThreadId.x;
    if (thread_index != 0u) return;
    MetalDeformableConstants constants = ropeStepConstants;
    MetalRopeAttachmentConstants attachments = ropeStepAttachments;
    uint first_body = attachments.rigid_count;
    uint last_body = attachments.rigid_count;
    for (uint index = 0; index < attachments.rigid_count; ++index) {
        if (rigid_ids[index].index == attachments.first_body.index &&
            rigid_ids[index].generation == attachments.first_body.generation)
            first_body = index;
        if (rigid_ids[index].index == attachments.last_body.index &&
            rigid_ids[index].generation == attachments.last_body.generation)
            last_body = index;
    }
    const bool first_attached = attachments.first_enabled != 0u &&
                                first_body < attachments.rigid_count;
    const bool last_attached = attachments.last_enabled != 0u &&
                               last_body < attachments.rigid_count;
    const bool first_soft = attachments.first_soft != 0u;
    const bool last_soft = attachments.last_soft != 0u;
    const uint edges = constants.count - 1u;
    float maximum_inverse_mass = 0.0f;
    for (uint index = 0u; index < constants.count; ++index)
        maximum_inverse_mass = max(maximum_inverse_mass,
                                   inverse_masses[index]);
    const float alpha = max(
        constants.compliance /
            max(constants.timestep * constants.timestep, 1.0e-12f),
        1.0e-5f * maximum_inverse_mass);
    for (uint edge = 0u; edge < edges; ++edge) lambdas[edge] = 0.0f;
    const uint maximum_iterations =
        min(32u, max(constants.solver_iterations,
                     4u * constants.solver_iterations));
    // CUDA spends eight contact-free stretch passes only when the nonlinear
    // contact budget is exhausted.  Keeping the recovery in this loop avoids
    // duplicating the direct solve while preserving that ordering exactly.
    for (uint iteration = 0u; iteration < maximum_iterations + 8u;
         ++iteration) {
        const bool recovery = iteration >= maximum_iterations;
        if (recovery && iteration == maximum_iterations) {
            for (uint node = 0u; node < constants.count; ++node) {
                contact_normals[node] = {0.0f, 0.0f, 0.0f};
                contact_normals2[node] = {0.0f, 0.0f, 0.0f};
            }
        }
        for (uint edge = 0u; edge < edges; ++edge) {
            const float3 delta = metalLoad(positions[edge + 1u]) -
                                 metalLoad(positions[edge]);
            directions[edge] = metalStore(
                dot(delta, delta) > 1.0e-12f
                    ? normalize(delta)
                    : float3(1.0f, 0.0f, 0.0f));
        }
        // Contact planes are unilateral constraints.  Rebuild the reduced
        // tridiagonal system when the tentative stretch impulse points away
        // from one, matching CUDA's active-set release before moving anything.
        for (uint active_pass = 0u; active_pass < 4u; ++active_pass) {
            for (uint edge = 0u; edge < edges; ++edge) {
                const float3 direction = metalLoad(directions[edge]);
                const float first_weight = pm_rope_direction_weight(
                    edge, constants.count, direction, first_attached,
                    last_attached, first_soft, last_soft, first_body,
                    last_body, attachments, inverse_masses, soft_anchors,
                    rigid_parameters, rigid_states, contact_normals,
                    contact_normals2);
                const float second_weight = pm_rope_direction_weight(
                    edge + 1u, constants.count, direction, first_attached,
                    last_attached, first_soft, last_soft, first_body,
                    last_body, attachments, inverse_masses, soft_anchors,
                    rigid_parameters, rigid_states, contact_normals,
                    contact_normals2);
                const float length_error =
                    distance(metalLoad(positions[edge + 1u]),
                             metalLoad(positions[edge])) -
                    bonds[edge].rest_length;
                rhs[edge] = -length_error - alpha * lambdas[edge];
                diagonal[edge] = first_weight + second_weight + alpha;
                if (edge + 1u < edges) {
                    upper[edge] = -dot(
                        metalLoad(directions[edge]),
                        pm_rope_projected_mass(
                            edge + 1u,
                            metalLoad(directions[edge + 1u]),
                            pm_rope_node_inverse(
                                edge + 1u, constants.count,
                                first_attached, last_attached, first_soft,
                                last_soft, inverse_masses, soft_anchors),
                            contact_normals, contact_normals2));
                }
            }
            for (uint edge = 1u; edge < edges; ++edge) {
                const float factor = upper[edge - 1u] /
                                     max(diagonal[edge - 1u], 1.0e-10f);
                diagonal[edge] -= factor * upper[edge - 1u];
                rhs[edge] -= factor * rhs[edge - 1u];
            }
            rhs[edges - 1u] /= max(diagonal[edges - 1u], 1.0e-10f);
            for (int edge = int(edges) - 2; edge >= 0; --edge) {
                rhs[uint(edge)] =
                    (rhs[uint(edge)] -
                     upper[uint(edge)] * rhs[uint(edge) + 1u]) /
                    max(diagonal[uint(edge)], 1.0e-10f);
            }
            bool released = false;
            for (uint node = 0u; node < constants.count; ++node) {
                float3 impulse = 0.0f;
                if (node != 0u)
                    impulse +=
                        metalLoad(directions[node - 1u]) * rhs[node - 1u];
                if (node < edges)
                    impulse -= metalLoad(directions[node]) * rhs[node];
                scratch[node] = metalStore(impulse);
                const bool rigid_attached =
                    (node == 0u && first_attached) ||
                    (node + 1u == constants.count && last_attached);
                if (!rigid_attached && active_pass < 3u) {
                    const float node_inverse = pm_rope_node_inverse(
                        node, constants.count, first_attached,
                        last_attached, first_soft, last_soft,
                        inverse_masses, soft_anchors);
                    const float release_impulse =
                        1.0e-6f / max(node_inverse, 1.0e-8f);
                    const float3 first_normal =
                        metalLoad(contact_normals[node]);
                    if (dot(impulse, first_normal) > release_impulse) {
                        contact_normals[node] = {0.0f, 0.0f, 0.0f};
                        released = true;
                    }
                    const float3 second_normal =
                        metalLoad(contact_normals2[node]);
                    if (dot(impulse, second_normal) > release_impulse) {
                        contact_normals2[node] = {0.0f, 0.0f, 0.0f};
                        released = true;
                    }
                }
            }
            if (!released) break;
        }
        float maximum_movement = 0.0f;
        for (uint node = 0u; node < constants.count; ++node) {
            const float3 impulse = metalLoad(scratch[node]);
            float node_inverse = inverse_masses[node];
            if (node == 0u && first_soft)
                node_inverse = soft_anchors[0].inverse_mass;
            if (node + 1u == constants.count && last_soft)
                node_inverse = soft_anchors[1].inverse_mass;
            float3 movement = pm_rope_projected_mass(
                node, impulse, node_inverse, contact_normals,
                contact_normals2);
            if (node == 0u && first_attached)
                movement = pm_rigid_impulse_movement(
                    rigid_parameters[first_body], rigid_states[first_body],
                    metalLoad(attachments.first_anchor), impulse);
            if (node + 1u == constants.count && last_attached)
                movement = pm_rigid_impulse_movement(
                    rigid_parameters[last_body], rigid_states[last_body],
                    metalLoad(attachments.last_anchor), impulse);
            maximum_movement = max(maximum_movement, length(movement));
        }
        const float scale = min(
            1.0f, constants.radius / max(maximum_movement, 1.0e-10f));
        for (uint edge = 0u; edge < edges; ++edge)
            lambdas[edge] += scale * rhs[edge];
        for (uint node = 0u; node < constants.count; ++node) {
            const float3 impulse = metalLoad(scratch[node]) * scale;
            constraint_forces[node] = metalStore(
                metalLoad(constraint_forces[node]) +
                impulse / max(constants.timestep * constants.timestep,
                              1.0e-12f));
            const bool rigid_attached =
                (node == 0u && first_attached) ||
                (node + 1u == constants.count && last_attached);
            float node_inverse = inverse_masses[node];
            int soft_end = -1;
            if (node == 0u && first_soft) {
                node_inverse = soft_anchors[0].inverse_mass;
                soft_end = 0;
            }
            if (node + 1u == constants.count && last_soft) {
                node_inverse = soft_anchors[1].inverse_mass;
                soft_end = 1;
            }
            if (!rigid_attached && node_inverse > 0.0f)
                positions[node] = metalStore(metalLoad(positions[node]) +
                    pm_rope_projected_mass(
                        node, impulse, node_inverse, contact_normals,
                        contact_normals2));
            if (soft_end >= 0) {
                soft_anchors[uint(soft_end)].impulse = metalStore(
                    metalLoad(soft_anchors[uint(soft_end)].impulse) +
                    impulse / max(constants.timestep, 1.0e-12f));
                if (node_inverse > 0.0f)
                    soft_anchors[uint(soft_end)].position = positions[node];
            }
        }
        if (first_attached)
            pm_move_rigid_by_rope(
                rigid_parameters[first_body], rigid_states[first_body],
                metalLoad(attachments.first_anchor), metalLoad(scratch[0]) * scale,
                constants.timestep);
        if (last_attached)
            pm_move_rigid_by_rope(
                rigid_parameters[last_body], rigid_states[last_body],
                metalLoad(attachments.last_anchor),
                metalLoad(scratch[constants.count - 1u]) * scale,
                constants.timestep);
        if (first_attached)
            positions[0] = metalStore(metalWorldPoint(
                rigid_states[first_body], metalLoad(attachments.first_anchor)));
        if (last_attached)
            positions[constants.count - 1u] = metalStore(metalWorldPoint(
                rigid_states[last_body], metalLoad(attachments.last_anchor)));
        if (first_soft) positions[0] = soft_anchors[0].position;
        if (last_soft)
            positions[constants.count - 1u] = soft_anchors[1].position;

        if (recovery) continue;

        // Match CUDA's nonlinear ordering: project stretch, then resolve
        // swept nodes and alternating capsule segments before measuring
        // convergence. Rigid reactions are accumulated per phase so all
        // contacts in a phase observe the same body pose.
        for (uint phase = 0u; phase < 3u; ++phase) {
            for (uint body = 0u; body < attachments.rigid_count; ++body) {
                body_translation[body] = {0.0f, 0.0f, 0.0f};
                pm_rope_store_body_rotation(
                    body_rotation, body, float3(0.0f));
            }
            const uint begin = phase == 0u ? 0u : phase - 1u;
            const uint stride = phase == 0u ? 1u : 2u;
            for (uint node = begin;
                 node < (phase == 0u ? constants.count : edges);
                 node += stride) {
                const bool segment = phase != 0u;
                const MetalRopeRigidHit hit = pm_rope_find_rigid_contact(
                    positions, previous, constants.count, node, segment,
                    constants.radius, first_body, last_body, attachments,
                    rigid_states, old_rigid_states, rigid_parameters,
                    mesh_vertices, mesh_indices, meshes, solid_planes,
                    body_rotation);
                if (hit.found) {
                const uint next = segment ? node + 1u : node;
                const float first_fraction = 1.0f - hit.fraction;
                const float second_fraction = hit.fraction;
                const bool first_free = !(node == 0u && first_attached) &&
                    !(node + 1u == constants.count && last_attached);
                const bool second_free = !(next == 0u && first_attached) &&
                    !(next + 1u == constants.count && last_attached);
                const float3 first_support = metalLoad(contact_normals[node]);
                const float3 first_support2 = metalLoad(contact_normals2[node]);
                const float3 second_support = metalLoad(contact_normals[next]);
                const float3 second_support2 = metalLoad(contact_normals2[next]);
                uint support_a = pm_rope_contact_support(hit.normal, first_support, first_support2);
                uint support_b = segment ? pm_rope_contact_support(hit.normal, second_support, second_support2) : 0u;
                float3 impulse = float3(0.0f);
                const float3 body_arm =
                    hit.point - metalLoad(rigid_states[hit.body].position);
                const float3 body_torque = cross(body_arm, hit.normal);
                const float3 rope_motion =
                    (metalLoad(positions[node]) - metalLoad(previous[node])) * first_fraction +
                    (metalLoad(positions[next]) - metalLoad(previous[next])) * second_fraction;
                const float3 body_velocity =
                    metalLoad(rigid_states[hit.body].linear_velocity) +
                    cross(metalLoad(rigid_states[hit.body].angular_velocity), body_arm);
                const float3 relative_motion = rope_motion - body_velocity * constants.timestep;
                const uint first_anchor_body = node == 0u ? first_body : last_body;
                const uint second_anchor_body = next == 0u ? first_body : last_body;
                MetalRopeContactMass mass_a;
                mass_a.body = first_free ? 0xffffffffu : first_anchor_body;
                mass_a.inverse = pm_rope_node_inverse(
                    node, constants.count, first_attached, last_attached,
                    first_soft, last_soft, inverse_masses, soft_anchors);
                mass_a.arm = first_free ? float3(0.0f) : metalRotate(
                    rigid_states[first_anchor_body].orientation,
                    metalLoad(node == 0u ? attachments.first_anchor :
                        attachments.last_anchor));
                mass_a.first_support = first_support;
                mass_a.second_support = first_support2;
                MetalRopeContactMass mass_b;
                mass_b.body = second_free ? 0xffffffffu : second_anchor_body;
                mass_b.inverse = pm_rope_node_inverse(
                    next, constants.count, first_attached, last_attached,
                    first_soft, last_soft, inverse_masses, soft_anchors);
                mass_b.arm = second_free ? float3(0.0f) : metalRotate(
                    rigid_states[second_anchor_body].orientation,
                    metalLoad(next == 0u ? attachments.first_anchor :
                        attachments.last_anchor));
                mass_b.first_support = second_support;
                mass_b.second_support = second_support2;
                MetalRopeContactMass mass_c;
                mass_c.body = hit.body;
                mass_c.inverse = 0.0f;
                mass_c.arm = body_arm;
                mass_c.first_support = float3(0.0f);
                mass_c.second_support = float3(0.0f);
                bool solvable = true;
                for (uint active_set = 0u; active_set < 5u; ++active_set) {
                const float first_weight = pm_rope_contact_weight(
                    node, constants.count, hit.normal, first_attached,
                    last_attached, first_soft, last_soft, first_body,
                    last_body, attachments, inverse_masses, soft_anchors,
                    rigid_parameters, rigid_states) * (first_free && support_a != 0u ? max(0.0f, dot(hit.normal,
                        pm_rope_contact_project(hit.normal, support_a, first_support, first_support2))) : 1.0f);
                const float second_weight = segment
                    ? pm_rope_contact_weight(
                          next, constants.count, hit.normal,
                          first_attached, last_attached, first_soft,
                          last_soft, first_body, last_body, attachments,
                          inverse_masses, soft_anchors, rigid_parameters,
                          rigid_states) * (second_free && support_b != 0u ? max(0.0f, dot(hit.normal,
                              pm_rope_contact_project(hit.normal, support_b, second_support, second_support2))) : 1.0f)
                    : 0.0f;
                const float body_weight =
                    rigid_parameters[hit.body].inverse_mass +
                    dot(body_torque,
                        metalInverseInertiaMul(
                            rigid_parameters[hit.body],
                            rigid_states[hit.body], body_torque));
                const float denominator =
                    first_weight * first_fraction * first_fraction +
                    second_weight * second_fraction * second_fraction +
                    body_weight;
                if (denominator <= 1.0e-12f) { solvable = false; break; }
                const float normal_impulse = hit.depth / denominator;
                impulse = hit.normal * normal_impulse;
                if (constants.spring_damping > 0.0f) {
                    const float3 normal_response = pm_rope_contact_pair_response(
                        mass_a, mass_b, mass_c, support_a, support_b, first_fraction,
                        second_fraction, hit.normal, rigid_parameters, rigid_states);
                    const float3 corrected_motion = relative_motion + normal_response * normal_impulse;
                    const float3 tangent = corrected_motion - hit.normal * dot(corrected_motion, hit.normal);
                    const float tangent_length = length(tangent);
                    if (tangent_length > 1.0e-8f) {
                        const float3 direction = tangent / tangent_length;
                        const float3 tangent_response = pm_rope_contact_pair_response(
                            mass_a, mass_b, mass_c, support_a, support_b, first_fraction,
                            second_fraction, direction, rigid_parameters, rigid_states);
                        MetalRopeFriction coupled =
                            pm_rope_coupled_contact_friction(
                            normal_impulse, denominator, dot(hit.normal, tangent_response),
                            dot(direction, tangent_response), tangent_length, constants.spring_damping);
                        impulse = hit.normal * coupled.normal + direction * coupled.tangent;
                    }
                }
                const uint next_a = pm_rope_contact_support(impulse, first_support, first_support2);
                const uint next_b = segment ? pm_rope_contact_support(impulse, second_support, second_support2) : 0u;
                if ((next_a == support_a && next_b == support_b) || active_set == 4u) break;
                // After a friction/support active-set cycle, retain known
                // planes for one conservative solve, not an inward move.
                support_a = active_set == 3u ? pm_rope_contact_support(
                    float3(0.0f), first_support, first_support2) : next_a;
                support_b = active_set == 3u && segment ? pm_rope_contact_support(
                    float3(0.0f), second_support, second_support2) : next_b;
                }
                if (solvable) {
                const float3 force_a = first_free ? pm_rope_contact_project(
                    impulse, support_a, first_support, first_support2) : impulse;
                const float3 force_b = second_free ? pm_rope_contact_project(
                    impulse, support_b, second_support, second_support2) : impulse;
                pm_rope_contact_move(
                    node, constants.count, force_a * first_fraction,
                    first_attached, last_attached, first_soft, last_soft,
                    first_body, last_body, attachments, positions,
                    inverse_masses, soft_anchors, rigid_parameters,
                    rigid_states, body_translation, body_rotation);
                if ((support_a & 1u) == 0u)
                    contact_normals[node] = metalStore(float3(0.0f));
                if ((support_a & 2u) == 0u)
                    contact_normals2[node] = metalStore(float3(0.0f));
                contact_forces[node] = metalStore(
                    metalLoad(contact_forces[node]) +
                    force_a * (first_fraction /
                        max(constants.timestep * constants.timestep,
                            1.0e-12f)));
                if (segment) {
                    pm_rope_contact_move(
                        next, constants.count, force_b * second_fraction,
                        first_attached, last_attached, first_soft, last_soft,
                        first_body, last_body, attachments, positions,
                        inverse_masses, soft_anchors, rigid_parameters,
                        rigid_states, body_translation, body_rotation);
                    if ((support_b & 1u) == 0u)
                        contact_normals[next] = metalStore(float3(0.0f));
                    if ((support_b & 2u) == 0u)
                        contact_normals2[next] = metalStore(float3(0.0f));
                    contact_forces[next] = metalStore(
                        metalLoad(contact_forces[next]) +
                        force_b * (second_fraction /
                            max(constants.timestep * constants.timestep,
                                1.0e-12f)));
                }
                if (rigid_parameters[hit.body].inverse_mass == 0.0f) {
                    for (uint item = node; item <= next; ++item) {
                        const float3 first_normal =
                            metalLoad(contact_normals[item]);
                        if (dot(first_normal, first_normal) < 0.5f ||
                            dot(first_normal, hit.normal) > 0.95f)
                            contact_normals[item] = metalStore(hit.normal);
                        else
                            contact_normals2[item] = metalStore(hit.normal);
                    }
                }
                pm_rope_accumulate_rigid_movement_at_arm(
                    hit.body, -impulse, body_arm, rigid_parameters,
                    rigid_states, body_translation, body_rotation);
                }
                }
                pm_rope_soft_contact_packed(
                    soft_target_a, node, segment, constants.count,
                    constants.timestep, first_attached, last_attached,
                    first_soft, last_soft, first_body, last_body,
                    attachments, positions, previous, inverse_masses,
                    soft_anchors, rigid_parameters, rigid_states,
                    body_translation, body_rotation, contact_normals,
                    contact_normals2, rope_soft_forces);
                pm_rope_soft_contact_packed(
                    soft_target_b, node, segment, constants.count,
                    constants.timestep, first_attached, last_attached,
                    first_soft, last_soft, first_body, last_body,
                    attachments, positions, previous, inverse_masses,
                    soft_anchors, rigid_parameters, rigid_states,
                    body_translation, body_rotation, contact_normals,
                    contact_normals2, rope_soft_forces);
            }
            for (uint body = 0u; body < attachments.rigid_count; ++body) {
                const float3 translation = metalLoad(body_translation[body]);
                const float3 rotation =
                    pm_rope_load_body_rotation(body_rotation, body);
                rigid_states[body].position = metalStore(
                    metalLoad(rigid_states[body].position) + translation);
                const MetalQuaternion spin = metalQuaternionMultiply(
                    {rotation.x, rotation.y, rotation.z, 0.0f},
                    rigid_states[body].orientation);
                rigid_states[body].orientation = metalQuaternionNormalize(
                    {fma(0.5f, spin.x,
                         rigid_states[body].orientation.x),
                     fma(0.5f, spin.y,
                         rigid_states[body].orientation.y),
                     fma(0.5f, spin.z,
                         rigid_states[body].orientation.z),
                     fma(0.5f, spin.w,
                         rigid_states[body].orientation.w)});
                rigid_states[body].linear_velocity = metalStore(
                    metalLoad(rigid_states[body].linear_velocity) +
                    translation / max(constants.timestep, 1.0e-12f));
                rigid_states[body].angular_velocity = metalStore(
                    metalLoad(rigid_states[body].angular_velocity) +
                    rotation / max(constants.timestep, 1.0e-12f));
            }
            if (first_attached)
                positions[0] = metalStore(metalWorldPoint(
                    rigid_states[first_body],
                    metalLoad(attachments.first_anchor)));
            if (last_attached)
                positions[constants.count - 1u] = metalStore(metalWorldPoint(
                    rigid_states[last_body],
                    metalLoad(attachments.last_anchor)));
            if (first_soft) positions[0] = soft_anchors[0].position;
            if (last_soft)
                positions[constants.count - 1u] = soft_anchors[1].position;
        }

        if (constants.self_collision != 0u) {
            const float diameter = 2.0f * constants.radius;
            for (uint first = 0u; first < constants.count; ++first) {
                const float first_inverse = pm_rope_node_inverse(
                    first, constants.count, first_attached, last_attached,
                    first_soft, last_soft, inverse_masses, soft_anchors);
                float3 correction = 0.0f;
                uint hits = 0u;
                for (uint second = 0u;
                     second < constants.count && first_inverse > 0.0f;
                     ++second) {
                    if (abs(int(first) - int(second)) <= 2) continue;
                    const float second_inverse = pm_rope_node_inverse(
                        second, constants.count, first_attached,
                        last_attached, first_soft, last_soft, inverse_masses,
                        soft_anchors);
                    const float3 delta = metalLoad(positions[first]) -
                                         metalLoad(positions[second]);
                    const float distance_value = length(delta);
                    const float inverse_sum = first_inverse + second_inverse;
                    if (distance_value >= diameter ||
                        inverse_sum <= 0.0f)
                        continue;
                    correction += metalNormalizedOr(
                        delta, float3(1.0f, 0.0f, 0.0f)) *
                        ((diameter - distance_value) * first_inverse /
                         inverse_sum);
                    ++hits;
                }
                scratch[first] = metalStore(
                    hits != 0u ? correction / float(hits) : 0.0f);
            }
            for (uint node = 0u; node < constants.count; ++node)
                positions[node] = metalStore(
                    metalLoad(positions[node]) + metalLoad(scratch[node]));
        }
        float strain = 0.0f;
        for (uint edge = 0u; edge < edges; ++edge)
            strain = max(
                strain,
                abs(distance(metalLoad(positions[edge + 1u]),
                             metalLoad(positions[edge])) /
                        bonds[edge].rest_length -
                    1.0f));
        if ((iteration >= 1u && strain < 1.0e-3f) ||
            (iteration + 1u >= constants.solver_iterations &&
             strain < 0.005f))
            break;
    }
    if (first_soft) soft_anchors[0].position = positions[0];
    if (last_soft)
        soft_anchors[1].position = positions[constants.count - 1u];
    for (uint node = 0u; node < constants.count; ++node) {
        if (node == 0u && first_attached) {
            velocities[node] = metalStore(pm_rigid_point_velocity(
                rigid_states[first_body], metalLoad(attachments.first_anchor)));
            continue;
        }
        if (node + 1u == constants.count && last_attached) {
            velocities[node] = metalStore(pm_rigid_point_velocity(
                rigid_states[last_body], metalLoad(attachments.last_anchor)));
            continue;
        }
        int soft_end = -1;
        if (node == 0u && first_soft) soft_end = 0;
        if (node + 1u == constants.count && last_soft) soft_end = 1;
        if (soft_end >= 0) {
            if (soft_anchors[uint(soft_end)].inverse_mass > 0.0f) {
                velocities[node] = metalStore(metalLimit(
                    (metalLoad(positions[node]) - metalLoad(previous[node])) /
                        constants.timestep,
                    constants.maximum_speed));
                soft_anchors[uint(soft_end)].velocity = velocities[node];
            } else {
                velocities[node] = soft_anchors[uint(soft_end)].velocity;
            }
            continue;
        }
        if (inverse_masses[node] <= 0.0f) {
            velocities[node] = {0.0f, 0.0f, 0.0f};
            continue;
        }
        velocities[node] = metalStore(metalLimit(
            (metalLoad(positions[node]) - metalLoad(previous[node])) /
                constants.timestep,
            constants.maximum_speed));
    }
    // Remove axial velocity using the same tridiagonal mass matrix so a taut
    // chain does not store residual stretch as kinetic energy.
    for (uint edge = 0u; edge < edges; ++edge) {
        const float3 delta = metalLoad(positions[edge + 1u]) -
                             metalLoad(positions[edge]);
        directions[edge] = metalStore(dot(delta, delta) > 1.0e-12f
                                        ? normalize(delta)
                                        : float3(1.0f, 0.0f, 0.0f));
        const float3 direction = metalLoad(directions[edge]);
        const float first_weight = edge == 0u && first_attached
            ? pm_rigid_direction_weight(
                  rigid_parameters[first_body], rigid_states[first_body],
                  metalLoad(attachments.first_anchor), direction)
            : (edge == 0u && first_soft
                   ? soft_anchors[0].inverse_mass
                   : inverse_masses[edge]);
        const float second_weight =
            (edge + 2u == constants.count && last_attached)
                ? pm_rigid_direction_weight(
                      rigid_parameters[last_body], rigid_states[last_body],
                      metalLoad(attachments.last_anchor), direction)
                : (edge + 2u == constants.count && last_soft
                       ? soft_anchors[1].inverse_mass
                       : inverse_masses[edge + 1u]);
        diagonal[edge] = first_weight + second_weight + alpha;
        rhs[edge] = -dot(metalLoad(directions[edge]),
                         metalLoad(velocities[edge + 1u]) -
                             metalLoad(velocities[edge]));
        if (edge + 1u < edges) {
            const float shared_weight = inverse_masses[edge + 1u];
            upper[edge] = -dot(metalLoad(directions[edge]),
                               metalLoad(directions[edge + 1u])) *
                          shared_weight;
        }
    }
    for (uint edge = 1u; edge < edges; ++edge) {
        const float factor = upper[edge - 1u] /
                             max(diagonal[edge - 1u], 1.0e-10f);
        diagonal[edge] -= factor * upper[edge - 1u];
        rhs[edge] -= factor * rhs[edge - 1u];
    }
    rhs[edges - 1u] /= max(diagonal[edges - 1u], 1.0e-10f);
    for (int edge = int(edges) - 2; edge >= 0; --edge)
        rhs[uint(edge)] =
            (rhs[uint(edge)] - upper[uint(edge)] * rhs[uint(edge) + 1u]) /
            max(diagonal[uint(edge)], 1.0e-10f);
    for (uint node = 0u; node < constants.count; ++node) {
        float3 impulse = 0.0f;
        if (node != 0u)
            impulse += metalLoad(directions[node - 1u]) * rhs[node - 1u];
        if (node < edges)
            impulse -= metalLoad(directions[node]) * rhs[node];
        const bool rigid_attached = (node == 0u && first_attached) ||
                                    (node + 1u == constants.count && last_attached);
        float node_inverse = inverse_masses[node];
        int soft_end = -1;
        if (node == 0u && first_soft) {
            node_inverse = soft_anchors[0].inverse_mass;
            soft_end = 0;
        }
        if (node + 1u == constants.count && last_soft) {
            node_inverse = soft_anchors[1].inverse_mass;
            soft_end = 1;
        }
        if (!rigid_attached && node_inverse > 0.0f)
            velocities[node] = metalStore(metalLimit(
                metalLoad(velocities[node]) + impulse * node_inverse,
                constants.maximum_speed));
        if (soft_end >= 0) {
            soft_anchors[uint(soft_end)].impulse = metalStore(
                metalLoad(soft_anchors[uint(soft_end)].impulse) + impulse);
            if (node_inverse > 0.0f)
                soft_anchors[uint(soft_end)].velocity = velocities[node];
            else
                velocities[node] = soft_anchors[uint(soft_end)].velocity;
        }
        constraint_forces[node] = metalStore(
            metalLoad(constraint_forces[node]) +
            impulse / max(constants.timestep, 1.0e-12f));
    }
    if (first_attached) {
        const float3 impulse =
            metalLoad(directions[0]) * -rhs[0];
        rigid_states[first_body].linear_velocity = metalStore(
            metalLoad(rigid_states[first_body].linear_velocity) +
            impulse * rigid_parameters[first_body].inverse_mass);
        const float3 arm = metalRotate(rigid_states[first_body].orientation,
                                     metalLoad(attachments.first_anchor));
        rigid_states[first_body].angular_velocity = metalStore(
            metalLoad(rigid_states[first_body].angular_velocity) +
            metalInverseInertiaMul(rigid_parameters[first_body],
                                   rigid_states[first_body],
                                   cross(arm, impulse)));
    }
    if (last_attached) {
        const float3 impulse =
            metalLoad(directions[edges - 1u]) * rhs[edges - 1u];
        rigid_states[last_body].linear_velocity = metalStore(
            metalLoad(rigid_states[last_body].linear_velocity) +
            impulse * rigid_parameters[last_body].inverse_mass);
        const float3 arm = metalRotate(rigid_states[last_body].orientation,
                                     metalLoad(attachments.last_anchor));
        rigid_states[last_body].angular_velocity = metalStore(
            metalLoad(rigid_states[last_body].angular_velocity) +
            metalInverseInertiaMul(rigid_parameters[last_body],
                                   rigid_states[last_body],
                                   cross(arm, impulse)));
    }
    if (first_attached) {
        positions[0] = metalStore(metalWorldPoint(
            rigid_states[first_body], metalLoad(attachments.first_anchor)));
        velocities[0] = metalStore(pm_rigid_point_velocity(
            rigid_states[first_body], metalLoad(attachments.first_anchor)));
    }
    if (last_attached) {
        positions[constants.count - 1u] = metalStore(metalWorldPoint(
            rigid_states[last_body], metalLoad(attachments.last_anchor)));
        velocities[constants.count - 1u] = metalStore(pm_rigid_point_velocity(
            rigid_states[last_body], metalLoad(attachments.last_anchor)));
    }
}
