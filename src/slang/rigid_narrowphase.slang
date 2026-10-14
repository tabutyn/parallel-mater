// SPDX-License-Identifier: MIT
// Slang-backed Metal compatibility adapter for rigid narrow-phase contacts.

// Keep the packed ABI private to this AIR object. Metal's linker does not
// coalesce Slang-generated aggregate initializers from separate modules.
struct MetalNpPackedVec3
{
    float x;
    float y;
    float z;
};

struct MetalNpQuaternion
{
    float x;
    float y;
    float z;
    float w;
};

struct MetalNpRigidState
{
    MetalNpPackedVec3 position;
    MetalNpQuaternion orientation;
    MetalNpPackedVec3 linear_velocity;
    MetalNpPackedVec3 angular_velocity;
};

struct MetalNpRigidParameters
{
    uint motion;
    float inverse_mass;
    MetalNpPackedVec3 inverse_inertia;
    float linear_damping;
    float angular_damping;
    float maximum_linear_speed;
    float maximum_angular_speed;
    uint has_kinematic_target;
    MetalNpRigidState kinematic_target;
    uint mesh_index;
    float friction;
    float restitution;
    float collision_margin;
};

struct MetalNpTriangleMeshInfo
{
    uint vertex_offset;
    uint vertex_count;
    uint index_offset;
    uint index_count;
    MetalNpPackedVec3 minimum;
    MetalNpPackedVec3 maximum;
    MetalNpPackedVec3 bounding_center;
    float radius;
    uint bvh_node_offset;
    uint bvh_node_count;
    uint solid_plane_offset;
    uint solid_plane_count;
};

struct MetalNpCollisionPlane
{
    MetalNpPackedVec3 normal;
    float offset;
};

struct MetalNpRigidStepConstants
{
    float timestep;
    MetalNpPackedVec3 gravity;
    uint body_count;
    uint constraint_capacity;
    uint collect_rigid_contacts;
    uint rigid_event_capacity;
    uint substeps;
    uint cooperative_contact_generation;
    uint rigid_sleeping;
    uint rigid_contact_pass_limit;
};

struct MetalNpAvbdDual
{
    float lambda;
    float penalty;
};

struct MetalNpContactRecord
{
    MetalNpPackedVec3 point;
    MetalNpPackedVec3 normal;
    float penetration;
    uint found;
    float accumulated_normal_impulse;
    MetalNpPackedVec3 accumulated_friction_impulse;
    float initial_normal_speed;
    float impact_fraction;
    uint persistent;
    uint warm_started;
    MetalNpAvbdDual avbd_dual[3];
    MetalNpPackedVec3 avbd_anchor_a;
    MetalNpPackedVec3 avbd_anchor_b;
    MetalNpPackedVec3 avbd_error;
    uint avbd_cached;
};

struct MetalNpContactManifold
{
    MetalNpContactRecord contacts[8];
    uint count;
    uint event_offset;
    uint color;
    MetalNpPackedVec3 initial_relative_position;
    uint face_patch;
    uint body_fixed_member;
    uint collider_fixed_member;
    uint cached;
    uint avbd_next_a;
    uint avbd_next_b;
};

struct MetalNpCachedContact
{
    MetalNpPackedVec3 local_point;
    MetalNpPackedVec3 collider_local_point;
    MetalNpPackedVec3 local_normal;
    float penetration;
    float normal_impulse;
    MetalNpPackedVec3 friction_impulse;
    MetalNpAvbdDual avbd_dual[3];
    MetalNpPackedVec3 avbd_anchor_a;
    MetalNpPackedVec3 avbd_anchor_b;
};

struct MetalNpCachedContactPair
{
    uint body_index;
    uint body_generation;
    uint collider_index;
    uint collider_generation;
    uint64_t epoch;
    float timestep;
    uint count;
    uint face_patch;
    uint color;
    MetalNpCachedContact contacts[8];
};

struct MetalNpRigidConstraintResource
{
    uint generation;
    uint alive;
    uint type;
    uint body_a;
    uint body_b;
    uint enabled;
    uint broken;
    MetalNpPackedVec3 local_anchor_a;
    MetalNpPackedVec3 local_anchor_b;
    MetalNpQuaternion local_orientation_a;
    MetalNpQuaternion local_orientation_b;
    float breaking_impulse_threshold;
    float applied_impulse;
    uint linear_limit_axes;
    MetalNpPackedVec3 linear_limit_lower;
    MetalNpPackedVec3 linear_limit_upper;
    uint angular_limit_axes;
    MetalNpPackedVec3 angular_limit_lower;
    MetalNpPackedVec3 angular_limit_upper;
    uint linear_spring_axes;
    MetalNpPackedVec3 linear_spring_stiffness;
    MetalNpPackedVec3 linear_spring_damping;
    uint angular_spring_axes;
    MetalNpPackedVec3 angular_spring_stiffness;
    MetalNpPackedVec3 angular_spring_damping;
    uint linear_motor_enabled;
    uint angular_motor_enabled;
    float linear_target_velocity;
    float linear_maximum_impulse;
    float angular_target_velocity;
    float angular_maximum_impulse;
    uint solver_iterations;
    uint disable_collisions;
    uint reserved[399];
};

struct MetalNpHandle
{
    uint index;
    uint generation;
};

struct MetalContactGeometry
{
    MetalNpPackedVec3 point;
    MetalNpPackedVec3 normal;
    float penetration;
    float impact_fraction;
};

struct MetalBvhNode
{
    MetalNpPackedVec3 minimum;
    MetalNpPackedVec3 maximum;
    uint left;
    uint right;
    uint first_triangle;
    uint triangle_count;
};

struct MetalMeshLeafInfo
{
    uint offset;
    uint count;
};

struct MetalRigidCollisionGroup
{
    uint root;
    uint eligible;
};

groupshared uint narrowSeparated;
groupshared MetalContactGeometry narrowScratchContacts[32][8];
groupshared uint narrowScratchCounts[32];

[[vk::binding(0, 0)]] StructuredBuffer<MetalNpRigidState> states : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<MetalNpRigidParameters> parameters : register(t1);
[[vk::binding(4, 0)]] ConstantBuffer<MetalNpRigidStepConstants> narrowStep : register(b4);
[[vk::binding(5, 0)]] StructuredBuffer<MetalNpPackedVec3> vertices : register(t5);
[[vk::binding(6, 0)]] StructuredBuffer<uint> indices : register(t6);
[[vk::binding(7, 0)]] StructuredBuffer<MetalNpTriangleMeshInfo> meshes : register(t7);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalNpContactManifold> manifolds : register(u8);
[[vk::binding(9, 0)]] StructuredBuffer<MetalNpRigidConstraintResource> constraints : register(t9);
[[vk::binding(10, 0)]] StructuredBuffer<MetalNpHandle> ids : register(t10);
[[vk::binding(13, 0)]] StructuredBuffer<MetalBvhNode> bvh_nodes : register(t13);
[[vk::binding(14, 0)]] StructuredBuffer<MetalNpRigidState> previous_states : register(t14);
[[vk::binding(18, 0)]] StructuredBuffer<uint> active_pairs : register(t18);
[[vk::binding(19, 0)]] StructuredBuffer<uint> activePairCountBuffer : register(t19);
[[vk::binding(20, 0)]] StructuredBuffer<uint> substepIndexBuffer : register(t20);
[[vk::binding(23, 0)]] StructuredBuffer<MetalMeshLeafInfo> mesh_leaf_infos : register(t23);
[[vk::binding(24, 0)]] StructuredBuffer<uint> bvh_leaves : register(t24);
[[vk::binding(26, 0)]] StructuredBuffer<MetalNpCollisionPlane> solid_planes : register(t26);
[[vk::binding(27, 0)]] StructuredBuffer<MetalNpCachedContactPair> cache : register(t27);
[[vk::binding(28, 0)]] StructuredBuffer<uint64_t> epochBaseBuffer : register(t28);
[[vk::binding(30, 0)]] RWStructuredBuffer<uint> sleep_states : register(u30);

static bool pm_rigid_is_asleep(RWStructuredBuffer<uint> values, uint body)
{
    return values[2u * body + 1u] != 0u;
}

static void pm_rigid_wake(RWStructuredBuffer<uint> values, uint body)
{
    InterlockedExchange(values[2u * body], 0u);
    InterlockedExchange(values[2u * body + 1u], 0u);
}

static uint pm_rigid_pair_slot(uint first, uint second, uint count)
{
    uint low = min(first, second);
    uint high = max(first, second);
    return low * (2u * count - low - 1u) / 2u + high - low - 1u;
}

static float3 pm_load(MetalNpPackedVec3 value) {
    return float3(value.x, value.y, value.z);
}

static MetalNpPackedVec3 pm_store(float3 value) {
    return {value.x, value.y, value.z};
}

static float3 pm_rotate(MetalNpQuaternion q, float3 point) {
    const float3 vector_part = float3(q.x, q.y, q.z);
    const float3 twice_cross = 2.0f * cross(vector_part, point);
    return point +
        (q.w * twice_cross + cross(vector_part, twice_cross));
}

static MetalNpQuaternion pm_quaternion_multiply(MetalNpQuaternion a, MetalNpQuaternion b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

static MetalNpQuaternion pm_quaternion_conjugate(MetalNpQuaternion value) {
    return {-value.x, -value.y, -value.z, value.w};
}

static MetalNpQuaternion pm_quaternion_normalize(MetalNpQuaternion value) {
    const float inverse = rsqrt(
        max(value.x * value.x + value.y * value.y +
                value.z * value.z + value.w * value.w,
            1.0e-12f));
    return {value.x * inverse, value.y * inverse, value.z * inverse,
            value.w * inverse};
}

static float3 pm_limit(float3 value, float maximum_length) {
    const float squared_length = dot(value, value);
    const float squared_maximum = maximum_length * maximum_length;
    return squared_length > squared_maximum
               ? value * (maximum_length * rsqrt(squared_length))
               : value;
}

static float3 pm_inverse_inertia_mul_orientation(
    MetalNpRigidParameters body, MetalNpQuaternion orientation,
    float3 value) {
    const float3 local = pm_rotate(
        pm_quaternion_conjugate(orientation), value);
    return pm_rotate(orientation,
                     local * pm_load(body.inverse_inertia));
}

static float3 pm_quaternion_delta_velocity(
    MetalNpQuaternion from, MetalNpQuaternion to, float timestep) {
    MetalNpQuaternion delta = pm_quaternion_multiply(
        to, pm_quaternion_conjugate(from));
    if (delta.w < 0.0f)
        delta = {-delta.x, -delta.y, -delta.z, -delta.w};
    delta = pm_quaternion_normalize(delta);
    const float vector_size = length(float3(delta.x, delta.y, delta.z));
    if (timestep <= 0.0f) return 0.0f;
    // Continuous quaternion-log limit; tiny corrections still carry torque.
    if (vector_size <= 1.0e-6f)
        return float3(delta.x, delta.y, delta.z) * (2.0f / timestep);
    const float angle =
        2.0f * atan2(vector_size, clamp(delta.w, -1.0f, 1.0f));
    return float3(delta.x, delta.y, delta.z) *
           (angle / (vector_size * timestep));
}

// Geometric joint metadata only: used for swept feature selection, never
// for velocity projection or a separate constraint solver.
struct MetalGuidedFrame {
    float3 anchor;
    float3 axis;
    float3 local_anchor;
    bool active;
    bool axial_rotation;
};

static float3 pm_guide_normalized_or(float3 value, float3 fallback) {
    const float squared = dot(value, value);
    return squared > 1.0e-12f ? value * rsqrt(squared) : fallback;
}

static MetalGuidedFrame pm_rigid_guided_frame(
    StructuredBuffer<MetalNpRigidParameters> parameters, uint body_count, uint body,
    StructuredBuffer<MetalNpRigidState> states,
    StructuredBuffer<MetalNpRigidConstraintResource> constraints,
    uint constraint_capacity, inout MetalNpQuaternion axial_orientation) {
    MetalGuidedFrame result = {};
    axial_orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    if (body >= body_count || parameters[body].inverse_mass <= 0.0f)
        return result;
    uint incident_count = 0u;
    for (uint index = 0u; index < constraint_capacity; ++index) {
        MetalNpRigidConstraintResource constraint =
            constraints[index];
        if (constraint.alive == 0u || constraint.enabled == 0u ||
            constraint.broken != 0u)
            continue;
        const bool is_a = constraint.body_a == body;
        const bool is_b = constraint.body_b == body;
        if (!is_a && !is_b) continue;
        ++incident_count;
        const bool axial = constraint.type == 3u || constraint.type == 4u;
        const uint other = is_a ? constraint.body_b : constraint.body_a;
        const bool static_anchor = other < body_count &&
                                   parameters[other].motion == 0u;
        if (result.active || !axial || !static_anchor ||
            constraint.breaking_impulse_threshold > 0.0f)
            continue;
        result.local_anchor = pm_load(
            is_a ? constraint.local_anchor_a : constraint.local_anchor_b);
        const MetalNpQuaternion local_orientation = is_a
            ? constraint.local_orientation_a
            : constraint.local_orientation_b;
        const float3 other_local_anchor = pm_load(
            is_a ? constraint.local_anchor_b : constraint.local_anchor_a);
        const MetalNpQuaternion other_local_orientation = is_a
            ? constraint.local_orientation_b
            : constraint.local_orientation_a;
        MetalNpRigidState other_state = states[other];
        axial_orientation = pm_quaternion_normalize(
            pm_quaternion_multiply(
                pm_quaternion_multiply(
                    other_state.orientation, other_local_orientation),
                pm_quaternion_conjugate(local_orientation)));
        result.anchor = pm_load(other_state.position) +
            pm_rotate(other_state.orientation, other_local_anchor);
        result.axis = pm_guide_normalized_or(
            pm_rotate(pm_quaternion_multiply(
                          other_state.orientation,
                          other_local_orientation),
                      float3(1.0f, 0.0f, 0.0f)),
            float3(1.0f, 0.0f, 0.0f));
        result.active = true;
        result.axial_rotation = constraint.type == 4u;
    }
    if (incident_count != 1u) result.active = false;
    return result;
}

// Contact-cache matching is generated from src/slang/rigid_contact_cache.slang.

static bool pm_restore_rigid_contact_geometry(
    inout MetalNpContactManifold manifold,
    MetalNpRigidState body_state,
    MetalNpRigidState collider_state,
    MetalNpHandle body_id,
    MetalNpHandle collider_id,
    MetalNpCachedContactPair saved,
    uint64_t epoch, float timestep) {
    if (saved.epoch + 1ul != epoch || saved.face_patch == 0u ||
        saved.count == 0u || saved.count > 8u ||
        saved.body_index != body_id.index ||
        saved.body_generation != body_id.generation ||
        saved.collider_index != collider_id.index ||
        saved.collider_generation != collider_id.generation ||
        abs(saved.timestep - timestep) > 1.0e-7f)
        return false;
    MetalNpContactManifold restored = {};
    restored.count = saved.count;
    restored.face_patch = 1u;
    restored.cached = 1u;
    restored.color = saved.color;
    for (uint point = 0u; point < saved.count; ++point) {
        MetalNpCachedContact cached = saved.contacts[point];
        const float3 body_point = pm_load(body_state.position) + pm_rotate(
            body_state.orientation, pm_load(cached.local_point));
        const float3 collider_point = pm_load(collider_state.position) +
            pm_rotate(collider_state.orientation,
                      pm_load(cached.collider_local_point));
        const float3 normal = normalize(pm_rotate(
            body_state.orientation, pm_load(cached.local_normal)));
        const float3 delta = collider_point - body_point;
        const float penetration = cached.penetration + dot(delta, normal);
        const float3 tangent = delta - normal * penetration;
        if (!isfinite(penetration) || dot(tangent, tangent) > 0.0001f ||
            abs(penetration) > 0.02f)
            return false;
        restored.contacts[point].point = pm_store(
            0.5f * (body_point + collider_point));
        restored.contacts[point].normal = pm_store(normal);
        restored.contacts[point].penetration = penetration;
        restored.contacts[point].found = 1u;
    }
    manifold = restored;
    return true;
}

// Contact-cache persistence is generated from src/slang/rigid_contact_cache.slang.

static uint pm_collision_group_root(
    StructuredBuffer<MetalRigidCollisionGroup> compounds, uint body) {
    while (compounds[body].root != body) body = compounds[body].root;
    return body;
}

static float3 pm_relative_rotation_vector(MetalNpQuaternion frame_a,
                                          MetalNpQuaternion frame_b) {
    MetalNpQuaternion relative = pm_quaternion_normalize(
        pm_quaternion_multiply(pm_quaternion_conjugate(frame_a), frame_b));
    if (relative.w < 0.0f)
        relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    const float size = sqrt(
        relative.x * relative.x + relative.y * relative.y +
        relative.z * relative.z);
    if (size <= 1.0e-6f)
        return 2.0f * float3(relative.x, relative.y, relative.z);
    const float angle = 2.0f * atan2(
        size, clamp(relative.w, -1.0f, 1.0f));
    return float3(relative.x, relative.y, relative.z) * (angle / size);
}

static float3 pm_normalized_or(float3 value, float3 fallback);

struct MetalHingeContactFrame {
    float3 anchor;
    float3 axis;
    float3 local_anchor;
    float inverse_mass;
    float inverse_moment;
    bool present;
    bool fixed;
    bool axial;
    bool axial_rotation;
    bool fixed_member;
    bool static_body;
};

static float pm_fixed_hinge_inverse_moment(
    MetalNpRigidParameters body,
    MetalNpRigidState state,
    MetalHingeContactFrame hinge) {
    if ((!hinge.fixed && !hinge.axial_rotation) ||
        body.inverse_mass <= 1.0e-6f)
        return 0.0f;
    const float3 local_axis = pm_rotate(
        pm_quaternion_conjugate(state.orientation), hinge.axis);
    const float3 inverse_inertia = pm_load(body.inverse_inertia);
    const float center_moment =
        local_axis.x * local_axis.x /
            max(inverse_inertia.x, 1.0e-6f) +
        local_axis.y * local_axis.y /
            max(inverse_inertia.y, 1.0e-6f) +
        local_axis.z * local_axis.z /
            max(inverse_inertia.z, 1.0e-6f);
    const float3 center_arm = pm_load(state.position) - hinge.anchor;
    const float3 perpendicular = center_arm -
        hinge.axis * dot(center_arm, hinge.axis);
    const float pivot_moment = center_moment +
        dot(perpendicular, perpendicular) / body.inverse_mass;
    return pivot_moment > 1.0e-6f ? 1.0f / pivot_moment : 0.0f;
}

static MetalHingeContactFrame pm_rigid_hinge_contact_frame(
    StructuredBuffer<MetalNpRigidParameters> parameters, uint body_count, uint body,
    StructuredBuffer<MetalNpRigidState> states,
    StructuredBuffer<MetalNpRigidConstraintResource> constraints,
    uint constraint_capacity, inout float3 reference) {
    MetalHingeContactFrame result = {};
    result.static_body = body < body_count && parameters[body].motion == 0u;
    if (body >= body_count || parameters[body].inverse_mass <= 0.0f) {
        reference = body < body_count
            ? pm_load(states[body].position) : float3(0.0f);
        return result;
    }
    uint incident_count = 0u;
    for (uint index = 0u; index < constraint_capacity; ++index) {
        MetalNpRigidConstraintResource constraint =
            constraints[index];
        if (constraint.alive == 0u || constraint.enabled == 0u ||
            constraint.broken != 0u)
            continue;
        const bool is_a = constraint.body_a == body;
        const bool is_b = constraint.body_b == body;
        if (!is_a && !is_b) continue;
        ++incident_count;
        if (constraint.type == 0u) result.fixed_member = true;
        const bool axial = constraint.type == 3u || constraint.type == 4u;
        const uint other = is_a ? constraint.body_b : constraint.body_a;
        const bool static_anchor = other < body_count &&
            parameters[other].motion == 0u;
        if (result.present ||
            (constraint.type != 2u &&
             !(axial && static_anchor &&
               constraint.breaking_impulse_threshold <= 0.0f)))
            continue;
        const float3 local_axis = axial
            ? float3(1.0f, 0.0f, 0.0f)
            : float3(0.0f, 0.0f, 1.0f);
        result.local_anchor = pm_load(
            is_a ? constraint.local_anchor_a : constraint.local_anchor_b);
        const MetalNpQuaternion local_orientation = is_a
            ? constraint.local_orientation_a
            : constraint.local_orientation_b;
        MetalNpRigidState state = states[body];
        result.anchor = pm_load(state.position) +
            pm_rotate(state.orientation, result.local_anchor);
        result.axis = pm_normalized_or(
            pm_rotate(pm_quaternion_multiply(
                          state.orientation, local_orientation),
                      local_axis),
            local_axis);
        result.present = true;
        result.fixed = static_anchor && !axial;
        result.axial = static_anchor && axial;
        result.axial_rotation = result.axial && constraint.type == 4u;
        if (static_anchor) {
            const float3 other_local_anchor = pm_load(
                is_a ? constraint.local_anchor_b : constraint.local_anchor_a);
            const MetalNpQuaternion other_local_orientation = is_a
                ? constraint.local_orientation_b
                : constraint.local_orientation_a;
            MetalNpRigidState other_state = states[other];
            result.anchor = pm_load(other_state.position) +
                pm_rotate(other_state.orientation, other_local_anchor);
            result.axis = pm_normalized_or(
                pm_rotate(pm_quaternion_multiply(
                              other_state.orientation,
                              other_local_orientation),
                          local_axis),
                result.axis);
        }
    }
    if (result.axial && incident_count != 1u) {
        result.axial = false;
        result.axial_rotation = false;
        result.present = false;
    }
    if (result.axial) {
        result.inverse_mass = parameters[body].inverse_mass;
        result.inverse_moment = pm_fixed_hinge_inverse_moment(
            parameters[body], states[body], result);
    }
    reference = result.present && !result.axial
        ? result.anchor : pm_load(states[body].position);
    return result;
}

static bool pm_guided_static_contact(
    MetalHingeContactFrame body,
    MetalHingeContactFrame collider);

static float3 pm_world_point(MetalNpRigidState state,
                             float3 local_point) {
    return pm_load(state.position) + pm_rotate(state.orientation, local_point);
}

static float3 pm_closest_point_triangle_mode(
    float3 point, float3 a, float3 b, float3 c, bool stable_face) {
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = point - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    const float3 bp = point - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        return a + (d1 / (d1 - d3)) * ab;
    }

    const float3 cp = point - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        return a + (d2 / (d2 - d6)) * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
    }

    if (stable_face) {
        const float3 normal = cross(ab, ac);
        return point - normal * (dot(ap, normal) / dot(normal, normal));
    }
    const float denominator = 1.0f / (va + vb + vc);
    return a +
        (ab * (vb * denominator) + ac * (vc * denominator));
}

static float3 pm_closest_point_triangle(float3 point, float3 a, float3 b,
                                        float3 c) {
    return pm_closest_point_triangle_mode(point, a, b, c, false);
}

static float3 pm_np_triangle_weights(float3 point, float3 a, float3 b,
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

static void pm_np_closest_segments(float3 first_a, float3 first_b,
                                float3 second_a, float3 second_b,
                                inout float first_fraction,
                                inout float second_fraction,
                                inout float3 first_point,
                                inout float3 second_point) {
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

static void pm_np_closest_segment_triangle(
    float3 segment_a, float3 segment_b, float3 a, float3 b, float3 c,
    inout float segment_fraction, inout float3 segment_point,
    inout float3 triangle_point, inout float3 triangle_weights) {
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
                triangle_weights = pm_np_triangle_weights(hit, a, b, c);
                return;
            }
        }
    }
    const float3 endpoints[2] = {segment_a, segment_b};
    for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
        const float3 nearest =
            pm_closest_point_triangle(endpoints[endpoint], a, b, c);
        const float squared =
            dot(endpoints[endpoint] - nearest, endpoints[endpoint] - nearest);
        if (squared >= best_squared) continue;
        best_squared = squared;
        segment_fraction = float(endpoint);
        segment_point = endpoints[endpoint];
        triangle_point = nearest;
        triangle_weights = pm_np_triangle_weights(nearest, a, b, c);
    }
    const float3 edge_a[3] = {a, b, c};
    const float3 edge_b[3] = {b, c, a};
    for (uint edge = 0u; edge < 3u; ++edge) {
        float first_fraction = 0.0f;
        float second_fraction = 0.0f;
        float3 first_point = 0.0f;
        float3 second_point = 0.0f;
        pm_np_closest_segments(segment_a, segment_b, edge_a[edge], edge_b[edge],
                            first_fraction, second_fraction, first_point,
                            second_point);
        const float squared =
            dot(first_point - second_point, first_point - second_point);
        if (squared >= best_squared) continue;
        best_squared = squared;
        segment_fraction = first_fraction;
        segment_point = first_point;
        triangle_point = second_point;
        triangle_weights = pm_np_triangle_weights(second_point, a, b, c);
    }
}

static float3 pm_normalized_or(float3 value, float3 fallback) {
    const float squared = dot(value, value);
    return squared > 1.0e-12f ? value * rsqrt(squared) : fallback;
}

static float3 pm_component_min(float3 first, float3 second) {
    return min(first, second);
}

static float3 pm_component_max(float3 first, float3 second) {
    return max(first, second);
}

static bool pm_bounds_overlap(float3 minimum_a, float3 maximum_a,
                              float3 minimum_b, float3 maximum_b) {
    return all(minimum_a <= maximum_b) && all(maximum_a >= minimum_b);
}

static void pm_transformed_bounds(
    MetalNpRigidState state, float3 local_minimum,
    float3 local_maximum, float margin, inout float3 minimum,
    inout float3 maximum) {
    const float3 local_center = (local_minimum + local_maximum) * 0.5f;
    const float3 local_half = (local_maximum - local_minimum) * 0.5f;
    const float3 axis_x =
        pm_rotate(state.orientation, float3(1.0f, 0.0f, 0.0f));
    const float3 axis_y =
        pm_rotate(state.orientation, float3(0.0f, 1.0f, 0.0f));
    const float3 axis_z =
        pm_rotate(state.orientation, float3(0.0f, 0.0f, 1.0f));
    const float3 world_center = pm_load(state.position) +
        axis_x * local_center.x + axis_y * local_center.y +
        axis_z * local_center.z;
    const float3 world_half =
        abs(axis_x) * local_half.x + abs(axis_y) * local_half.y +
        abs(axis_z) * local_half.z + margin;
    minimum = world_center - world_half;
    maximum = world_center + world_half;
}

static void pm_transformed_motion_bounds(
    MetalNpRigidState previous_state,
    MetalNpRigidState state, float3 local_minimum,
    float3 local_maximum, float margin, inout float3 minimum,
    inout float3 maximum) {
    pm_transformed_bounds(
        state, local_minimum, local_maximum, margin, minimum, maximum);
    float3 previous_minimum = 0.0f;
    float3 previous_maximum = 0.0f;
    pm_transformed_bounds(
        previous_state, local_minimum, local_maximum, margin,
        previous_minimum, previous_maximum);
    minimum = pm_component_min(minimum, previous_minimum);
    maximum = pm_component_max(maximum, previous_maximum);
}

static float pm_rotational_motion_bound(
    MetalNpRigidState previous_state,
    MetalNpRigidState state,
    MetalNpTriangleMeshInfo mesh) {
    const float orientation_dot = clamp(
        abs(previous_state.orientation.x * state.orientation.x +
            previous_state.orientation.y * state.orientation.y +
            previous_state.orientation.z * state.orientation.z +
            previous_state.orientation.w * state.orientation.w),
        0.0f, 1.0f);
    const float sine_half_angle =
        sqrt(max(0.0f, 1.0f - orientation_dot * orientation_dot));
    const float3 maximum_absolute = max(
        abs(pm_load(mesh.minimum)), abs(pm_load(mesh.maximum)));
    return 2.0f * length(maximum_absolute) * sine_half_angle;
}

static bool pm_requires_swept_pair_contact(
    MetalNpRigidState previous_body_state,
    MetalNpRigidState body_state,
    MetalNpTriangleMeshInfo body_mesh,
    MetalNpRigidState previous_collider_state,
    MetalNpRigidState collider_state,
    MetalNpTriangleMeshInfo collider_mesh, float threshold) {
    const float3 relative_translation =
        (pm_load(body_state.position) -
         pm_load(previous_body_state.position)) -
        (pm_load(collider_state.position) -
         pm_load(previous_collider_state.position));
    return length(relative_translation) +
               pm_rotational_motion_bound(
                   previous_body_state, body_state, body_mesh) +
               pm_rotational_motion_bound(
                   previous_collider_state, collider_state, collider_mesh) >
           threshold;
}

static bool pm_triangle_bounds_overlap(
    float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2,
    float margin) {
    const float3 minimum_a =
        pm_component_min(a0, pm_component_min(a1, a2)) - margin;
    const float3 maximum_a =
        pm_component_max(a0, pm_component_max(a1, a2)) + margin;
    const float3 minimum_b = pm_component_min(b0, pm_component_min(b1, b2));
    const float3 maximum_b = pm_component_max(b0, pm_component_max(b1, b2));
    return pm_bounds_overlap(minimum_a, maximum_a, minimum_b, maximum_b);
}

static bool pm_point_in_triangle(
    float3 point, float3 first, float3 second, float3 third,
    float3 normal) {
    float tolerance = -1.0e-5f * dot(normal, normal);
    return dot(cross(second - first, point - first), normal) >= tolerance &&
           dot(cross(third - second, point - second), normal) >= tolerance &&
           dot(cross(first - third, point - third), normal) >= tolerance;
}

static bool pm_segment_hits_triangle(
    float3 first, float3 second, float3 a, float3 b, float3 c,
    out float3 intersection) {
    intersection = float3(0.0f);
    float3 normal = cross(b - a, c - a);
    float3 direction = second - first;
    float denominator = dot(normal, direction);
    if (dot(normal, normal) <= 1.0e-12f ||
        abs(denominator) <= 1.0e-6f)
        return false;
    float amount = dot(normal, a - first) / denominator;
    if (amount < 0.0f || amount > 1.0f) return false;
    intersection = first + direction * amount;
    return pm_point_in_triangle(intersection, a, b, c, normal);
}

static void pm_closest_triangle_pair(
    float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2,
    inout float3 point_a, inout float3 point_b,
    bool stable_face = false) {
    const float3 a[3] = {a0, a1, a2};
    const float3 b[3] = {b0, b1, b2};

    float3 intersection = 0.0f;
    float best_squared = 3.402823466e+38f;
    for (uint edge = 0u; edge < 3u; ++edge) {
        if (pm_segment_hits_triangle(
                        a[edge], a[(edge + 1u) % 3u], b0, b1, b2,
                        intersection)) {
            point_a = intersection;
            point_b = intersection;
            return;
        }
    }
    for (uint edge = 0u; edge < 3u; ++edge) {
        if (pm_segment_hits_triangle(
                        b[edge], b[(edge + 1u) % 3u], a0, a1, a2,
                        intersection)) {
            point_a = intersection;
            point_b = intersection;
            return;
        }
    }
    for (uint vertex_index = 0u; vertex_index < 3u; ++vertex_index) {
        const float3 on_triangle =
            pm_closest_point_triangle_mode(
                a[vertex_index], b0, b1, b2, stable_face);
        const float squared = dot(a[vertex_index] - on_triangle,
                                  a[vertex_index] - on_triangle);
        if (squared < best_squared) {
            best_squared = squared;
            point_a = a[vertex_index];
            point_b = on_triangle;
        }
    }
    for (uint vertex_index = 0u; vertex_index < 3u; ++vertex_index) {
        const float3 on_triangle =
            pm_closest_point_triangle_mode(
                b[vertex_index], a0, a1, a2, stable_face);
        const float squared = dot(on_triangle - b[vertex_index],
                                  on_triangle - b[vertex_index]);
        if (squared < best_squared) {
            best_squared = squared;
            point_a = on_triangle;
            point_b = b[vertex_index];
        }
    }
    for (uint edge_a = 0u; edge_a < 3u; ++edge_a) {
        for (uint edge_b = 0u; edge_b < 3u; ++edge_b) {
            float first_fraction = 0.0f;
            float second_fraction = 0.0f;
            float3 on_a = 0.0f;
            float3 on_b = 0.0f;
            pm_np_closest_segments(
                a[edge_a], a[(edge_a + 1u) % 3u], b[edge_b],
                b[(edge_b + 1u) % 3u], first_fraction, second_fraction,
                on_a, on_b);
            (void)first_fraction;
            (void)second_fraction;
            const float squared = dot(on_a - on_b, on_a - on_b);
            if (squared >= best_squared) continue;
            best_squared = squared;
            point_a = on_a;
            point_b = on_b;
        }
    }
}

static void pm_closest_triangle_pair_swept(
    float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2,
    inout float3 point_a, inout float3 point_b) {
    const float3 a[3] = {a0, a1, a2};
    const float3 b[3] = {b0, b1, b2};
    float best_squared = 3.402823466e+38f;
    for (uint edge = 0u; edge < 3u; ++edge) {
        float fraction = 0.0f;
        float3 on_segment = 0.0f;
        float3 on_triangle = 0.0f;
        float3 weights = 0.0f;
        pm_np_closest_segment_triangle(
            a[edge], a[(edge + 1u) % 3u], b0, b1, b2, fraction,
            on_segment, on_triangle, weights);
        const float squared =
            dot(on_segment - on_triangle, on_segment - on_triangle);
        if (squared < best_squared) {
            best_squared = squared;
            point_a = on_segment;
            point_b = on_triangle;
        }
    }
    for (uint edge = 0u; edge < 3u; ++edge) {
        float fraction = 0.0f;
        float3 on_segment = 0.0f;
        float3 on_triangle = 0.0f;
        float3 weights = 0.0f;
        pm_np_closest_segment_triangle(
            b[edge], b[(edge + 1u) % 3u], a0, a1, a2, fraction,
            on_segment, on_triangle, weights);
        const float squared =
            dot(on_segment - on_triangle, on_segment - on_triangle);
        if (squared < best_squared) {
            best_squared = squared;
            point_a = on_triangle;
            point_b = on_segment;
        }
    }
}

static float3 pm_guided_triangle_normal(
    float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2,
    float3 point_a, float3 point_b, float3 fallback) {
    const float3 delta = point_a - point_b;
    float3 normal = pm_normalized_or(delta, fallback);
    const float3 faces[2] = {
        pm_normalized_or(cross(a1 - a0, a2 - a0), normal),
        pm_normalized_or(cross(b1 - b0, b2 - b0), normal)};
    // A coincident vertex/face hit can retain an obsolete edge separator.
    // Strict triangle-interior classification identifies its current face
    // without imposing a face normal at a genuine edge or corner.
    const float3 vertices[2][3] = {{a0, a1, a2}, {b0, b1, b2}};
    const float3 points[2] = {point_a, point_b};
    float face_alignment = -1.0f;
    float3 interior_normal = float3(0.0f);
    for (uint index = 0u; index < 2u; ++index) {
        const float3 raw_normal = cross(vertices[index][1] - vertices[index][0],
                                        vertices[index][2] - vertices[index][0]);
        const float tolerance = 1.0e-5f * dot(raw_normal, raw_normal);
        bool interior = tolerance > 0.0f;
        for (uint edge = 0u; edge < 3u; ++edge) {
            interior = interior && dot(cross(
                vertices[index][(edge + 1u) % 3u] - vertices[index][edge],
                points[index] - vertices[index][edge]), raw_normal) > tolerance;
        }
        const float alignment = abs(dot(faces[index], normal));
        if (interior && alignment > face_alignment) {
            face_alignment = alignment;
            const float projection = dot(delta, faces[index]);
            const float side = abs(projection) > 1.0e-6f
                ? projection : dot(fallback, faces[index]);
            interior_normal = side >= 0.0f ? faces[index] : -faces[index];
        }
    }
    if (face_alignment >= 0.0f) return interior_normal;
    float best_error = 4.0e-12f;
    if (dot(delta, delta) > 1.0e-12f) {
        for (uint face_index = 0u; face_index < 2u; ++face_index) {
            const float3 face = faces[face_index];
            const float projection = dot(delta, face);
            const float3 error_vector = delta - face * projection;
            const float error = dot(error_vector, error_vector);
            if (error < best_error) {
                best_error = error;
                normal = projection >= 0.0f ? face : -face;
            }
        }
    }
    return normal;
}

static bool pm_triangle_pair_face_contact(
    float3 a[3], float3 b[3], float3 separation) {
    const float3 normal = pm_normalized_or(separation, float3(0.0f));
    const float3 face_a = pm_normalized_or(
        cross(a[1] - a[0], a[2] - a[0]), float3(0.0f));
    const float3 face_b = pm_normalized_or(
        cross(b[1] - b[0], b[2] - b[0]), float3(0.0f));
    return abs(dot(normal, face_a)) > 0.9999f ||
           abs(dot(normal, face_b)) > 0.9999f;
}

static void pm_add_manifold_contact(
    inout MetalNpContactManifold manifold, MetalNpContactRecord candidate,
    float separation) {
    const float minimum_spacing_squared = separation * separation;
    for (uint index = 0u; index < manifold.count; ++index) {
        const float3 delta =
            pm_load(candidate.point) - pm_load(manifold.contacts[index].point);
        if (dot(delta, delta) < minimum_spacing_squared) {
            if (candidate.penetration >
                manifold.contacts[index].penetration)
                manifold.contacts[index] = candidate;
            return;
        }
    }
    if (manifold.count < 8u) {
        manifold.contacts[manifold.count++] = candidate;
        return;
    }
    uint shallowest = 0u;
    for (uint index = 1u; index < manifold.count; ++index)
        if (manifold.contacts[index].penetration <
            manifold.contacts[shallowest].penetration)
            shallowest = index;
    if (candidate.penetration >
        manifold.contacts[shallowest].penetration)
        manifold.contacts[shallowest] = candidate;
}

static float3 pm_guide_contact_direction(
    MetalNpContactRecord contact,
    MetalHingeContactFrame body_hinge,
    MetalHingeContactFrame collider_hinge) {
    MetalHingeContactFrame frame = body_hinge.axial
        ? body_hinge : collider_hinge;
    const float3 normal = pm_load(contact.normal);
    const float3 point = pm_load(contact.point);
    return float3(
        dot(normal, frame.axis) * sqrt(frame.inverse_mass),
        frame.axial_rotation
            ? dot(frame.axis, cross(point - frame.anchor, normal)) *
                  sqrt(frame.inverse_moment)
            : 0.0f,
        0.0f);
}

static void pm_add_pair_manifold_contact(
    inout MetalNpContactManifold manifold, MetalNpContactRecord candidate,
    float separation,
    MetalHingeContactFrame body_hinge,
    MetalHingeContactFrame collider_hinge) {
    if (!pm_guided_static_contact(body_hinge, collider_hinge)) {
        pm_add_manifold_contact(manifold, candidate, separation);
        return;
    }

    const float3 candidate_direction = pm_guide_contact_direction(
        candidate, body_hinge, collider_hinge);
    const float candidate_length = length(candidate_direction);
    if (candidate_length * candidate_length <= 1.0e-6f) return;
    if (candidate.impact_fraction > 0.0f) {
        float first_impact = 1.0f;
        for (uint row = 0u; row < manifold.count; ++row)
            if (manifold.contacts[row].impact_fraction > 0.0f)
                first_impact = min(
                    first_impact,
                    manifold.contacts[row].impact_fraction);
        const float simultaneous = 1.0e-4f;
        if (candidate.impact_fraction > first_impact + simultaneous) return;
        if (candidate.impact_fraction < first_impact - simultaneous) {
            uint kept = 0u;
            for (uint row = 0u; row < manifold.count; ++row)
                if (manifold.contacts[row].impact_fraction == 0.0f)
                    manifold.contacts[kept++] = manifold.contacts[row];
            manifold.count = kept;
        }
    }
    for (uint index = 0u; index < manifold.count; ++index) {
        const float3 direction = pm_guide_contact_direction(
            manifold.contacts[index], body_hinge, collider_hinge);
        const float size = length(direction);
        if (dot(candidate_direction, direction) >
            0.9999f * candidate_length * size) {
            if (candidate.penetration / candidate_length >
                manifold.contacts[index].penetration / size)
                manifold.contacts[index] = candidate;
            return;
        }
    }
    if (manifold.count < 8u) {
        manifold.contacts[manifold.count++] = candidate;
        return;
    }
    uint shallowest = 0u;
    for (uint index = 1u; index < manifold.count; ++index)
        if (manifold.contacts[index].penetration <
            manifold.contacts[shallowest].penetration)
            shallowest = index;
    if (candidate.penetration > manifold.contacts[shallowest].penetration)
        manifold.contacts[shallowest] = candidate;
}

static const float pm_rigid_surface_tolerance = 1.0e-5f;
static const float pm_rigid_maximum_rest_offset = 1.0e-3f;

static float pm_rigid_rest_offset(float margin) {
    return min(margin, pm_rigid_maximum_rest_offset);
}

static MetalNpContactRecord pm_make_contact_record(
    float3 point, float3 normal, float penetration,
    float impact_fraction) {
    MetalNpContactRecord contact = {};
    contact.point = pm_store(point);
    contact.normal = pm_store(normal);
    contact.penetration = penetration;
    contact.found = 1u;
    contact.impact_fraction = impact_fraction;
    return contact;
}

static float pm_contact_normal_speed(
    MetalNpRigidState body_state,
    MetalNpRigidState collider_state, float3 point,
    float3 normal) {
    const float3 body_velocity = pm_load(body_state.linear_velocity) +
        cross(pm_load(body_state.angular_velocity),
              point - pm_load(body_state.position));
    const float3 collider_velocity = pm_load(collider_state.linear_velocity) +
        cross(pm_load(collider_state.angular_velocity),
              point - pm_load(collider_state.position));
    return dot(body_velocity - collider_velocity, normal);
}

static bool pm_contact_reaches_rest_offset(
    MetalNpRigidState body_state,
    MetalNpRigidState collider_state, float3 point,
    float3 normal, float distance, float rest_offset, float timestep) {
    if (distance <= rest_offset + pm_rigid_surface_tolerance) return true;
    const float normal_speed = pm_contact_normal_speed(
        body_state, collider_state, point, normal);
    if (normal_speed > pm_rigid_surface_tolerance) return false;
    return -normal_speed * timestep + rest_offset +
               pm_rigid_surface_tolerance >=
           distance;
}

// Refine broad face contacts on small closed convex meshes. The bounded face
// count keeps clipping storage and per-pair work fixed; curved/concave meshes
// and edge impacts retain the triangle/BVH path.
static bool pm_convex_face_manifold(
    MetalNpRigidState body_state,
    MetalNpTriangleMeshInfo body_mesh,
    MetalNpRigidState collider_state,
    MetalNpTriangleMeshInfo collider_mesh,
    StructuredBuffer<MetalNpPackedVec3> vertices, StructuredBuffer<uint> indices,
    StructuredBuffer<MetalNpCollisionPlane> solid_planes, float margin,
    inout MetalNpContactManifold output) {
    const uint maximum_faces = 32u;
    if (body_mesh.solid_plane_count == 0u ||
        collider_mesh.solid_plane_count == 0u ||
        body_mesh.index_count > maximum_faces * 3u ||
        collider_mesh.index_count > maximum_faces * 3u)
        return false;

    float best_separation = -3.402823466e+38f;
    uint reference_face = 0u;
    bool reference_is_body = false;
    for (uint side = 0u; side < 2u; ++side) {
        MetalNpTriangleMeshInfo reference_mesh =
            side == 0u ? collider_mesh : body_mesh;
        MetalNpRigidState reference_state =
            side == 0u ? collider_state : body_state;
        MetalNpTriangleMeshInfo incident_mesh =
            side == 0u ? body_mesh : collider_mesh;
        MetalNpRigidState incident_state =
            side == 0u ? body_state : collider_state;
        for (uint face = 0u; face < reference_mesh.index_count / 3u;
             ++face) {
            const MetalNpCollisionPlane plane = solid_planes[
                reference_mesh.solid_plane_offset + face];
            const float3 normal = pm_rotate(
                reference_state.orientation, pm_load(plane.normal));
            const float3 incident_normal = pm_rotate(
                pm_quaternion_conjugate(incident_state.orientation),
                normal);
            float support = 3.402823466e+38f;
            for (uint vertex_index = 0u;
                 vertex_index < incident_mesh.vertex_count;
                 ++vertex_index)
                support = min(
                    support,
                    dot(incident_normal,
                        pm_load(vertices[incident_mesh.vertex_offset +
                                         vertex_index])));
            const float separation = support +
                dot(normal,
                    pm_load(incident_state.position) -
                        pm_load(reference_state.position)) -
                plane.offset;
            if (separation > best_separation) {
                best_separation = separation;
                reference_face = face;
                reference_is_body = side != 0u;
            }
        }
    }
    if (best_separation > margin) {
        output = {};
        return true;
    }

    MetalNpTriangleMeshInfo reference_mesh =
        reference_is_body ? body_mesh : collider_mesh;
    MetalNpRigidState reference_state =
        reference_is_body ? body_state : collider_state;
    MetalNpTriangleMeshInfo incident_mesh =
        reference_is_body ? collider_mesh : body_mesh;
    MetalNpRigidState incident_state =
        reference_is_body ? collider_state : body_state;
    const MetalNpCollisionPlane reference = solid_planes[
        reference_mesh.solid_plane_offset + reference_face];
    const float3 reference_normal = pm_load(reference.normal);
    const float3 outward = pm_rotate(
        reference_state.orientation, reference_normal);
    const float3 incident_axis = pm_rotate(
        pm_quaternion_conjugate(incident_state.orientation), outward);
    float alignment = 1.0f;
    uint incident_face = 0u;
    for (uint face = 0u; face < incident_mesh.index_count / 3u; ++face) {
        const MetalNpCollisionPlane plane = solid_planes[
            incident_mesh.solid_plane_offset + face];
        const float value = dot(incident_axis, pm_load(plane.normal));
        if (value < alignment) {
            alignment = value;
            incident_face = face;
        }
    }
    if (alignment > -0.98f) return false;

    MetalNpContactManifold manifold = {};
    const float3 normal = reference_is_body ? -outward : outward;
    const MetalNpCollisionPlane incident = solid_planes[
        incident_mesh.solid_plane_offset + incident_face];
    const float3 incident_normal = pm_load(incident.normal);
    for (uint triangle = 0u; triangle < incident_mesh.index_count / 3u;
         ++triangle) {
        const MetalNpCollisionPlane face = solid_planes[
            incident_mesh.solid_plane_offset + triangle];
        if (dot(pm_load(face.normal), incident_normal) < 0.99999f ||
            abs(face.offset - incident.offset) > pm_rigid_surface_tolerance)
            continue;
        float3 polygon[maximum_faces + 4u];
        float3 clipped[maximum_faces + 4u];
        uint count = 3u;
        for (uint corner = 0u; corner < 3u; ++corner) {
            const uint local_index = indices[
                incident_mesh.index_offset + triangle * 3u + corner];
            const float3 world = pm_world_point(
                incident_state,
                pm_load(vertices[incident_mesh.vertex_offset +
                                 local_index]));
            polygon[corner] = pm_rotate(
                pm_quaternion_conjugate(reference_state.orientation),
                world - pm_load(reference_state.position));
        }
        // Clip the incident triangle against the reference solid's side
        // faces. Only the supporting face is expanded for speculative
        // contacts.
        for (uint plane_index = 0u;
             plane_index < reference_mesh.index_count / 3u && count != 0u;
             ++plane_index) {
            const MetalNpCollisionPlane plane = solid_planes[
                reference_mesh.solid_plane_offset + plane_index];
            const float3 plane_normal = pm_load(plane.normal);
            const float offset = plane.offset +
                (dot(plane_normal, reference_normal) > 0.99999f
                     ? margin
                     : 0.0f);
            uint clipped_count = 0u;
            float3 previous = polygon[count - 1u];
            float previous_distance = dot(plane_normal, previous) - offset;
            for (uint vertex_index = 0u; vertex_index < count;
                 ++vertex_index) {
                const float3 current = polygon[vertex_index];
                const float distance = dot(plane_normal, current) - offset;
                if ((distance <= 0.0f) != (previous_distance <= 0.0f))
                    clipped[clipped_count++] = previous +
                        (current - previous) *
                            (previous_distance /
                             (previous_distance - distance));
                if (distance <= 0.0f) clipped[clipped_count++] = current;
                previous = current;
                previous_distance = distance;
            }
            count = clipped_count;
            for (uint vertex_index = 0u; vertex_index < count;
                 ++vertex_index)
                polygon[vertex_index] = clipped[vertex_index];
        }
        for (uint vertex_index = 0u; vertex_index < count; ++vertex_index) {
            const float distance =
                dot(reference_normal, polygon[vertex_index]) - reference.offset;
            const float3 local_point = polygon[vertex_index] -
                reference_normal * (distance * 0.5f);
            const float3 point = pm_world_point(
                reference_state, local_point);
            if (distance <= margin) {
                const MetalNpContactRecord contact = pm_make_contact_record(
                    point, normal, -distance, 1.0f);
                pm_add_manifold_contact(
                    manifold, contact, max(margin * 2.0f, 1.0e-4f));
            }
        }
    }
    // Parallel supporting faces with no overlap are narrowSeparated, including
    // adjacent corners that only coincide within floating-point roundoff.
    // Falling back to intersecting triangles invents penetration there.
    if (manifold.count == 0u && alignment > -0.999999f) return false;
    manifold.face_patch = 1u;
    output = manifold;
    return true;
}

static void pm_reduce_collinear_face_contacts(
    inout MetalNpContactManifold manifold) {
    uint kept = 0u;
    for (uint candidate = 0u; candidate < manifold.count; ++candidate) {
        const float3 point = pm_load(manifold.contacts[candidate].point);
        bool interior = false;
        for (uint first = 0u; first < manifold.count && !interior; ++first) {
            if (first == candidate) continue;
            const float3 start = pm_load(manifold.contacts[first].point);
            for (uint second = first + 1u; second < manifold.count;
                 ++second) {
                if (second == candidate) continue;
                const float3 edge =
                    pm_load(manifold.contacts[second].point) - start;
                const float squared_length = dot(edge, edge);
                if (squared_length <= 1.0e-10f) continue;
                const float fraction = dot(point - start, edge) /
                                       squared_length;
                if (fraction <= 1.0e-4f || fraction >= 0.9999f) continue;
                const float3 delta = point - (start + edge * fraction);
                if (dot(delta, delta) <=
                    max(1.0e-10f, squared_length * 1.0e-5f)) {
                    interior = true;
                    break;
                }
            }
        }
        if (!interior)
            manifold.contacts[kept++] = manifold.contacts[candidate];
    }
    manifold.count = kept;
}

static void pm_load_triangle(
    MetalNpRigidState state,
    MetalNpTriangleMeshInfo mesh,
    StructuredBuffer<MetalNpPackedVec3> vertices, StructuredBuffer<uint> indices,
    uint triangle, inout float3 a, inout float3 b, inout float3 c) {
    const uint base = mesh.index_offset + triangle * 3u;
    a = pm_world_point(
        state, pm_load(vertices[mesh.vertex_offset + indices[base]]));
    b = pm_world_point(
        state, pm_load(vertices[mesh.vertex_offset + indices[base + 1u]]));
    c = pm_world_point(
        state, pm_load(vertices[mesh.vertex_offset + indices[base + 2u]]));
}

static void pm_collide_triangle_ranges(
    MetalNpRigidState previous_body_state,
    MetalNpRigidState body_state,
    float3 body_reference,
    MetalNpTriangleMeshInfo body_mesh, uint body_first,
    uint body_count,
    MetalNpRigidState previous_collider_state,
    MetalNpRigidState collider_state,
    MetalNpTriangleMeshInfo collider_mesh, uint collider_first,
    uint collider_count, StructuredBuffer<MetalNpPackedVec3> vertices,
    StructuredBuffer<uint> indices, float margin, float timestep,
    bool robust_closest, bool fixed_cluster_contact,
    MetalHingeContactFrame body_hinge,
    MetalHingeContactFrame collider_hinge,
    inout MetalNpContactManifold manifold) {
    if (pm_guided_static_contact(body_hinge, collider_hinge)) return;
    const float rest_offset = pm_rigid_rest_offset(margin);
    for (uint body_triangle = body_first;
         body_triangle < body_first + body_count; ++body_triangle) {
        float3 a0, a1, a2;
        pm_load_triangle(body_state, body_mesh, vertices, indices,
                         body_triangle, a0, a1, a2);
        for (uint collider_triangle = collider_first;
             collider_triangle < collider_first + collider_count;
             ++collider_triangle) {
            float3 b0, b1, b2;
            pm_load_triangle(collider_state, collider_mesh, vertices, indices,
                             collider_triangle, b0, b1, b2);
            if (!pm_triangle_bounds_overlap(
                    a0, a1, a2, b0, b1, b2, margin))
                continue;
            float3 point_a = 0.0f;
            float3 point_b = 0.0f;
            if (robust_closest)
                pm_closest_triangle_pair_swept(
                    a0, a1, a2, b0, b1, b2, point_a, point_b);
            else
                pm_closest_triangle_pair(
                    a0, a1, a2, b0, b1, b2, point_a, point_b);
            const float3 delta = point_a - point_b;
            const float squared = dot(delta, delta);
            if (squared > margin * margin) continue;
            const float3 collider_normal = pm_normalized_or(
                cross(b1 - b0, b2 - b0), float3(0.0f, 1.0f, 0.0f));
            const float reference_side =
                dot(collider_normal, body_reference - point_b);
            const float3 fallback = reference_side >= 0.0f
                ? collider_normal
                : -collider_normal;
            const float distance = sqrt(max(squared, 0.0f));
            const float3 point = (point_a + point_b) * 0.5f;
            const float3 plane_projection = point_a - collider_normal *
                dot(point_a - b0, collider_normal);
            const float3 on_plane_face = pm_closest_point_triangle(
                plane_projection, b0, b1, b2);
            const bool on_triangle_face = dot(
                plane_projection - on_plane_face,
                plane_projection - on_plane_face) <=
                pm_rigid_surface_tolerance * pm_rigid_surface_tolerance;
            const bool small_convex = body_mesh.solid_plane_count != 0u &&
                                      body_mesh.index_count <= 96u;
            float3 normal =
                (distance <= pm_rigid_surface_tolerance ||
                 (small_convex && on_triangle_face))
                    ? fallback
                    : pm_normalized_or(delta, fallback);
            bool convex_surface_face = false;
            float face_separation = 0.0f;
            const bool convex_a = body_mesh.solid_plane_count != 0u;
            const bool convex_b = collider_mesh.solid_plane_count != 0u;
            if (convex_a != convex_b &&
                distance <= pm_rigid_surface_tolerance) {
                const float3 face0 = convex_a ? b0 : a0;
                const float3 face1 = convex_a ? b1 : a1;
                const float3 face2 = convex_a ? b2 : a2;
                MetalNpRigidState surface_state =
                    convex_a ? collider_state : body_state;
                MetalNpRigidState previous_surface =
                    convex_a ? previous_collider_state : previous_body_state;
                MetalNpRigidState previous_convex =
                    convex_a ? previous_body_state : previous_collider_state;
                float3 outward = pm_normalized_or(
                    cross(face1 - face0, face2 - face0), float3(0.0f));
                const float3 local_normal = pm_rotate(
                    pm_quaternion_conjugate(surface_state.orientation),
                    outward);
                const float3 old_normal = pm_rotate(
                    previous_surface.orientation, local_normal);
                const float3 local_point = pm_rotate(
                    pm_quaternion_conjugate(surface_state.orientation),
                    face0 - pm_load(surface_state.position));
                const float3 old_point =
                    pm_load(previous_surface.position) +
                    pm_rotate(previous_surface.orientation, local_point);
                if (dot(old_normal,
                        pm_load(previous_convex.position) - old_point) < 0.0f)
                    outward = -outward;
                const float3 incident_point = convex_a ? point_a : point_b;
                const float3 projection = incident_point - outward *
                    dot(incident_point - face0, outward);
                const float3 on_face = pm_closest_point_triangle(
                    projection, face0, face1, face2);
                convex_surface_face = dot(outward, outward) > 0.5f &&
                    dot(projection - on_face, projection - on_face) <=
                        pm_rigid_surface_tolerance *
                            pm_rigid_surface_tolerance;
                if (convex_surface_face) {
                    normal = convex_a ? outward : -outward;
                    const float3 convex0 = convex_a ? a0 : b0;
                    const float3 convex1 = convex_a ? a1 : b1;
                    const float3 convex2 = convex_a ? a2 : b2;
                    face_separation = min(
                        dot(convex0 - face0, outward),
                        min(dot(convex1 - face0, outward),
                            dot(convex2 - face0, outward)));
                }
            }
            if (!pm_contact_reaches_rest_offset(
                    body_state, collider_state, point, normal, distance,
                    rest_offset, timestep))
                continue;
            float penetration =
                rest_offset - distance + pm_rigid_surface_tolerance;
            if (convex_surface_face) {
                penetration = min(
                    margin, rest_offset - face_separation) +
                    pm_rigid_surface_tolerance;
            } else if (distance <= pm_rigid_surface_tolerance) {
                if (fixed_cluster_contact || small_convex) {
                    const float intersection_depth = max(
                        0.0f,
                        -min(dot(a0 - point_b, normal),
                             min(dot(a1 - point_b, normal),
                                 dot(a2 - point_b, normal))));
                    penetration = min(
                        margin, intersection_depth + rest_offset) +
                        pm_rigid_surface_tolerance;
                } else {
                    penetration = margin + pm_rigid_surface_tolerance;
                }
            }
            MetalNpContactRecord contact = pm_make_contact_record(
                point, normal, penetration, 1.0f);
            pm_add_pair_manifold_contact(
                manifold, contact, max(margin * 2.0f, 1.0e-4f),
                body_hinge, collider_hinge);
        }
    }
}

static void pm_collide_triangle_ranges_swept(
    MetalNpRigidState previous_body_state,
    MetalNpRigidState body_state,
    float3 previous_body_reference, float3 body_reference,
    MetalNpTriangleMeshInfo body_mesh, uint body_first,
    uint body_count,
    MetalNpRigidState previous_collider_state,
    MetalNpRigidState collider_state,
    MetalNpTriangleMeshInfo collider_mesh, uint collider_first,
    uint collider_count, StructuredBuffer<MetalNpPackedVec3> vertices,
    StructuredBuffer<uint> indices, float margin, float timestep,
    bool robust_closest,
    MetalHingeContactFrame body_hinge,
    MetalHingeContactFrame collider_hinge,
    inout MetalNpContactManifold manifold) {
    const bool guided = pm_guided_static_contact(
        body_hinge, collider_hinge);
    const float rest_offset = guided
        ? min(margin, 1.0e-4f) : pm_rigid_rest_offset(margin);
    for (uint body_triangle = body_first;
         body_triangle < body_first + body_count; ++body_triangle) {
        float3 previous_a[3];
        float3 current_a[3];
        pm_load_triangle(previous_body_state, body_mesh, vertices, indices,
                         body_triangle, previous_a[0], previous_a[1],
                         previous_a[2]);
        pm_load_triangle(body_state, body_mesh, vertices, indices,
                         body_triangle, current_a[0], current_a[1],
                         current_a[2]);
        float3 swept_a_minimum = pm_component_min(
            previous_a[0], current_a[0]);
        float3 swept_a_maximum = pm_component_max(
            previous_a[0], current_a[0]);
        for (uint vertex_index = 1u; vertex_index < 3u; ++vertex_index) {
            swept_a_minimum = pm_component_min(
                swept_a_minimum,
                pm_component_min(previous_a[vertex_index],
                                 current_a[vertex_index]));
            swept_a_maximum = pm_component_max(
                swept_a_maximum,
                pm_component_max(previous_a[vertex_index],
                                 current_a[vertex_index]));
        }
        for (uint collider_triangle = collider_first;
             collider_triangle < collider_first + collider_count;
             ++collider_triangle) {
            float3 previous_b[3];
            float3 current_b[3];
            pm_load_triangle(
                previous_collider_state, collider_mesh, vertices, indices,
                collider_triangle, previous_b[0], previous_b[1],
                previous_b[2]);
            pm_load_triangle(
                collider_state, collider_mesh, vertices, indices,
                collider_triangle, current_b[0], current_b[1],
                current_b[2]);
            float3 swept_b_minimum = pm_component_min(
                previous_b[0], current_b[0]);
            float3 swept_b_maximum = pm_component_max(
                previous_b[0], current_b[0]);
            for (uint vertex_index = 1u; vertex_index < 3u; ++vertex_index) {
                swept_b_minimum = pm_component_min(
                    swept_b_minimum,
                    pm_component_min(previous_b[vertex_index],
                                     current_b[vertex_index]));
                swept_b_maximum = pm_component_max(
                    swept_b_maximum,
                    pm_component_max(previous_b[vertex_index],
                                     current_b[vertex_index]));
            }
            if (!pm_bounds_overlap(
                    swept_a_minimum - margin,
                    swept_a_maximum + margin,
                    swept_b_minimum, swept_b_maximum))
                continue;

            float3 delta_a[3];
            float3 delta_b[3];
            float speed_bound = 0.0f;
            for (uint vertex_index = 0u; vertex_index < 3u;
                 ++vertex_index) {
                delta_a[vertex_index] =
                    current_a[vertex_index] - previous_a[vertex_index];
                delta_b[vertex_index] =
                    current_b[vertex_index] - previous_b[vertex_index];
                speed_bound = max(
                    speed_bound, length(delta_a[vertex_index]));
            }
            float collider_speed = 0.0f;
            for (uint vertex_index = 0u; vertex_index < 3u;
                 ++vertex_index)
                collider_speed = max(
                    collider_speed, length(delta_b[vertex_index]));
            speed_bound += collider_speed;
            const float3 common_motion = delta_b[0];
            float relative_body_speed = 0.0f;
            float relative_collider_speed = 0.0f;
            for (uint vertex_index = 0u; vertex_index < 3u;
                 ++vertex_index) {
                relative_body_speed = max(
                    relative_body_speed,
                    length(delta_a[vertex_index] - common_motion));
                relative_collider_speed = max(
                    relative_collider_speed,
                    length(delta_b[vertex_index] - common_motion));
            }
            speed_bound = min(
                speed_bound,
                relative_body_speed + relative_collider_speed);
            if (speed_bound <= 1.0e-6f) continue;

            float time = 0.0f;
            bool starts_near_contact = false;
            float3 separating_normal = 0.0f;
            bool have_separating_normal = false;
            for (uint iteration = 0u; iteration < 32u; ++iteration) {
                float3 a[3];
                float3 b[3];
                for (uint vertex_index = 0u; vertex_index < 3u;
                     ++vertex_index) {
                    a[vertex_index] = previous_a[vertex_index] +
                                      delta_a[vertex_index] * time;
                    b[vertex_index] = previous_b[vertex_index] +
                                      delta_b[vertex_index] * time;
                }
                float3 point_a = 0.0f;
                float3 point_b = 0.0f;
                if (guided)
                    pm_closest_triangle_pair(
                        a[0], a[1], a[2], b[0], b[1], b[2],
                        point_a, point_b, true);
                else if (robust_closest)
                    pm_closest_triangle_pair_swept(
                        a[0], a[1], a[2], b[0], b[1], b[2],
                        point_a, point_b);
                else
                    pm_closest_triangle_pair(
                        a[0], a[1], a[2], b[0], b[1], b[2],
                        point_a, point_b);
                const float3 delta = point_a - point_b;
                const float distance =
                    sqrt(max(0.0f, dot(delta, delta)));
                float contact_offset = rest_offset;
                if (guided &&
                    !pm_triangle_pair_face_contact(a, b, delta)) {
                    MetalHingeContactFrame guide = body_hinge.axial
                        ? body_hinge : collider_hinge;
                    const float3 direction = pm_normalized_or(delta, 0.0f);
                    const float axial = dot(direction, guide.axis);
                    const float angular = dot(
                        guide.axis,
                        cross(point_a - guide.anchor, direction));
                    if (abs(axial) > 1.0e-3f &&
                        abs(angular) > 1.0e-3f)
                        contact_offset = 0.0f;
                }
                if (iteration == 0u)
                    starts_near_contact = distance <=
                        contact_offset +
                            5.0f * pm_rigid_surface_tolerance;
                if (distance <=
                    contact_offset + pm_rigid_surface_tolerance) {
                    if (iteration == 0u && !guided) break;
                    const float3 collider_normal = pm_normalized_or(
                        cross(b[1] - b[0], b[2] - b[0]),
                        float3(0.0f, 1.0f, 0.0f));
                    float3 body_center =
                        pm_load(previous_body_state.position) +
                        (pm_load(body_state.position) -
                         pm_load(previous_body_state.position)) * time;
                    if (body_hinge.present)
                        body_center = previous_body_reference +
                            (body_reference - previous_body_reference) * time;
                    const float reference_side =
                        dot(collider_normal, body_center - point_b);
                    const float3 fallback = reference_side >= 0.0f
                        ? collider_normal
                        : -collider_normal;
                    // Keep swept interior-face normals identical to discrete
                    // normals instead of amplifying closest-point roundoff.
                    const float3 plane_projection = point_a - collider_normal *
                        dot(point_a - b[0], collider_normal);
                    const float3 face_delta = plane_projection -
                        pm_closest_point_triangle(plane_projection, b[0], b[1], b[2]);
                    const bool on_triangle_face = dot(face_delta, face_delta) <=
                        pm_rigid_surface_tolerance * pm_rigid_surface_tolerance;
                    const bool small_convex = body_mesh.solid_plane_count != 0u &&
                                              body_mesh.index_count <= 96u;
                    const float3 normal = guided
                                ? pm_guided_triangle_normal(
                                      a[0], a[1], a[2], b[0], b[1], b[2],
                                      point_a, point_b,
                                      have_separating_normal
                                          ? separating_normal : fallback)
                                : (distance <= pm_rigid_surface_tolerance ||
                                   (small_convex && on_triangle_face))
                                    ? fallback
                                    : pm_normalized_or(delta, fallback);
                    const float3 point = (point_a + point_b) * 0.5f;
                    const float normal_speed = pm_contact_normal_speed(
                        body_state, collider_state, point, normal);
                    if (normal_speed > pm_rigid_surface_tolerance) break;
                    const float remaining =
                        -normal_speed * timestep * (1.0f - time) - distance;
                    const float swept_penetration = guided
                        ? max(0.0f, remaining + contact_offset) +
                              pm_rigid_surface_tolerance
                        : max(0.0f, remaining) + rest_offset +
                              pm_rigid_surface_tolerance;
                    MetalNpContactRecord contact = pm_make_contact_record(
                        point, normal, swept_penetration,
                        starts_near_contact ? 0.0f : time);
                    pm_add_pair_manifold_contact(
                        manifold, contact,
                        max(margin * 2.0f, 1.0e-4f), body_hinge,
                        collider_hinge);
                    break;
                }
                separating_normal = pm_guided_triangle_normal(
                    a[0], a[1], a[2], b[0], b[1], b[2], point_a, point_b,
                    pm_normalized_or(
                        delta, float3(0.0f, 1.0f, 0.0f)));
                have_separating_normal = true;
                float advancement =
                    (distance - contact_offset) /
                    (speed_bound + 1.0e-6f) * 0.9f;
                if (guided) {
                    const float3 plane = pm_guided_triangle_normal(
                        a[0], a[1], a[2], b[0], b[1], b[2],
                        point_a, point_b, separating_normal);
                    float minimum_a = 3.402823466e+38f;
                    float maximum_b = -3.402823466e+38f;
                    float minimum_speed_a = 3.402823466e+38f;
                    float maximum_speed_b = -3.402823466e+38f;
                    for (uint vertex_index = 0u; vertex_index < 3u;
                         ++vertex_index) {
                        minimum_a = min(
                            minimum_a,
                            dot(a[vertex_index] - point_b, plane));
                        maximum_b = max(
                            maximum_b,
                            dot(b[vertex_index] - point_b, plane));
                        minimum_speed_a = min(
                            minimum_speed_a,
                            dot(delta_a[vertex_index], plane));
                        maximum_speed_b = max(
                            maximum_speed_b,
                            dot(delta_b[vertex_index], plane));
                    }
                    const float gap =
                        minimum_a - maximum_b - contact_offset;
                    const float closing_speed =
                        maximum_speed_b - minimum_speed_a;
                    if (gap > pm_rigid_surface_tolerance &&
                        closing_speed <= 0.0f)
                        break;
                    if (gap > 0.0f && closing_speed > 1.0e-6f)
                        advancement = max(
                            advancement, 0.9f * gap / closing_speed);
                }
                advancement = max(advancement, 1.0e-5f);
                time += advancement;
                if (time > 1.0f) break;
            }
        }
    }
}

static MetalNpContactManifold pm_collide_meshes(
    MetalNpRigidState previous_body_state,
    MetalNpRigidState body_state,
    float3 previous_body_reference, float3 body_reference,
    MetalNpTriangleMeshInfo body_mesh,
    MetalNpRigidState previous_collider_state,
    MetalNpRigidState collider_state,
    MetalNpTriangleMeshInfo collider_mesh,
    StructuredBuffer<MetalNpPackedVec3> vertices, StructuredBuffer<uint> indices,
    StructuredBuffer<MetalBvhNode> bvh_nodes,
    MetalMeshLeafInfo body_leaf_info,
    MetalMeshLeafInfo collider_leaf_info,
    StructuredBuffer<uint> bvh_leaves, float margin, float timestep,
    bool robust_closest, bool fixed_cluster_contact, bool swept_only,
    MetalHingeContactFrame body_hinge,
    MetalHingeContactFrame collider_hinge) {
    MetalNpContactManifold manifold = {};
    const bool swept = swept_only ||
        pm_guided_static_contact(body_hinge, collider_hinge) ||
        pm_requires_swept_pair_contact(
            previous_body_state, body_state, body_mesh,
            previous_collider_state, collider_state, collider_mesh, margin);
    const bool deep_sweep = robust_closest &&
        pm_requires_swept_pair_contact(
            previous_body_state, body_state, body_mesh,
            previous_collider_state, collider_state, collider_mesh,
            max(margin * 8.0f, 0.25f));
    float3 body_minimum = 0.0f;
    float3 body_maximum = 0.0f;
    float3 collider_minimum = 0.0f;
    float3 collider_maximum = 0.0f;
    if (swept) {
        pm_transformed_motion_bounds(
            previous_body_state, body_state, pm_load(body_mesh.minimum),
            pm_load(body_mesh.maximum), margin, body_minimum, body_maximum);
        pm_transformed_motion_bounds(
            previous_collider_state, collider_state,
            pm_load(collider_mesh.minimum), pm_load(collider_mesh.maximum),
            0.0f, collider_minimum, collider_maximum);
    } else {
        pm_transformed_bounds(
            body_state, pm_load(body_mesh.minimum), pm_load(body_mesh.maximum),
            margin, body_minimum, body_maximum);
        pm_transformed_bounds(
            collider_state, pm_load(collider_mesh.minimum),
            pm_load(collider_mesh.maximum), 0.0f, collider_minimum,
            collider_maximum);
    }
    if (!pm_bounds_overlap(
            body_minimum, body_maximum, collider_minimum, collider_maximum))
        return manifold;

    // CUDA compacts the Cartesian product into per-lane runs before contact
    // evaluation. Preserve that lane-grouped candidate order so symmetric
    // contacts select the same representative points.
    const uint body_leaf_count = body_leaf_info.count;
    const uint collider_leaf_count = collider_leaf_info.count;
    if (body_leaf_count != 0u && collider_leaf_count != 0u) {
        const uint candidate_count = body_leaf_count * collider_leaf_count;
        const float separation = max(margin * 2.0f, 1.0e-4f);
        uint overlapping_candidate_count = 0u;
        bool leaf_pair_overflow = false;
        for (uint lane = 0u; lane < 128u && !leaf_pair_overflow; ++lane) {
            for (uint candidate = lane; candidate < candidate_count;
                 candidate += 128u) {
                const uint body_leaf = candidate / collider_leaf_count;
                const uint collider_leaf = candidate -
                    body_leaf * collider_leaf_count;
                MetalBvhNode body_node =
                    bvh_nodes[bvh_leaves[
                        body_leaf_info.offset + body_leaf]];
                MetalBvhNode collider_node =
                    bvh_nodes[bvh_leaves[
                        collider_leaf_info.offset + collider_leaf]];
                if (swept) {
                    pm_transformed_motion_bounds(
                        previous_body_state, body_state,
                        pm_load(body_node.minimum),
                        pm_load(body_node.maximum), margin, body_minimum,
                        body_maximum);
                    pm_transformed_motion_bounds(
                        previous_collider_state, collider_state,
                        pm_load(collider_node.minimum),
                        pm_load(collider_node.maximum), 0.0f,
                        collider_minimum, collider_maximum);
                } else {
                    pm_transformed_bounds(
                        body_state, pm_load(body_node.minimum),
                        pm_load(body_node.maximum), margin, body_minimum,
                        body_maximum);
                    pm_transformed_bounds(
                        collider_state, pm_load(collider_node.minimum),
                        pm_load(collider_node.maximum), 0.0f,
                        collider_minimum, collider_maximum);
                }
                if (!pm_bounds_overlap(
                        body_minimum, body_maximum, collider_minimum,
                        collider_maximum))
                    continue;
                if (++overlapping_candidate_count > 512u) {
                    leaf_pair_overflow = true;
                    break;
                }
                MetalNpContactManifold local = {};
                if (!swept_only)
                    pm_collide_triangle_ranges(
                        previous_body_state, body_state, body_reference,
                        body_mesh,
                        body_node.first_triangle,
                        body_node.triangle_count, previous_collider_state,
                        collider_state,
                        collider_mesh, collider_node.first_triangle,
                        collider_node.triangle_count, vertices, indices,
                        margin, timestep, deep_sweep,
                        fixed_cluster_contact, body_hinge, collider_hinge,
                        local);
                if (swept) {
                    pm_collide_triangle_ranges_swept(
                        previous_body_state, body_state,
                        previous_body_reference, body_reference, body_mesh,
                        body_node.first_triangle, body_node.triangle_count,
                        previous_collider_state, collider_state,
                        collider_mesh, collider_node.first_triangle,
                        collider_node.triangle_count, vertices, indices,
                        margin, timestep, deep_sweep, body_hinge,
                        collider_hinge, local);
                }
                for (uint contact = 0u; contact < local.count; ++contact)
                    pm_add_pair_manifold_contact(
                        manifold, local.contacts[contact], separation,
                        body_hinge, collider_hinge);
            }
        }
        if (!leaf_pair_overflow) return manifold;
        // CUDA's bounded leaf-pair cache marks this pair as overflow and the
        // finalizer recomputes it with the serial BVH traversal below.
        manifold = {};
    }

    uint2 stack[256];
    uint stack_size = 0u;
    stack[0] = uint2(0u);
    bool overflow = body_mesh.bvh_node_count == 0u ||
                    collider_mesh.bvh_node_count == 0u;
    if (!overflow)
        stack[stack_size++] =
            uint2(body_mesh.bvh_node_offset, collider_mesh.bvh_node_offset);
    while (stack_size > 0u && !overflow) {
        const uint2 pair = stack[--stack_size];
        MetalBvhNode body_node = bvh_nodes[pair.x];
        MetalBvhNode collider_node = bvh_nodes[pair.y];
        if (swept) {
            pm_transformed_motion_bounds(
                previous_body_state, body_state, pm_load(body_node.minimum),
                pm_load(body_node.maximum), margin, body_minimum,
                body_maximum);
            pm_transformed_motion_bounds(
                previous_collider_state, collider_state,
                pm_load(collider_node.minimum),
                pm_load(collider_node.maximum), 0.0f, collider_minimum,
                collider_maximum);
        } else {
            pm_transformed_bounds(
                body_state, pm_load(body_node.minimum),
                pm_load(body_node.maximum), margin, body_minimum,
                body_maximum);
            pm_transformed_bounds(
                collider_state, pm_load(collider_node.minimum),
                pm_load(collider_node.maximum), 0.0f, collider_minimum,
                collider_maximum);
        }
        if (!pm_bounds_overlap(
                body_minimum, body_maximum, collider_minimum,
                collider_maximum))
            continue;
        const bool body_leaf = body_node.triangle_count != 0u;
        const bool collider_leaf = collider_node.triangle_count != 0u;
        if (body_leaf && collider_leaf) {
            if (!swept_only)
                pm_collide_triangle_ranges(
                    previous_body_state, body_state, body_reference,
                    body_mesh,
                    body_node.first_triangle,
                    body_node.triangle_count, previous_collider_state,
                    collider_state, collider_mesh,
                    collider_node.first_triangle,
                    collider_node.triangle_count, vertices, indices, margin,
                    timestep, deep_sweep, fixed_cluster_contact, body_hinge,
                    collider_hinge, manifold);
            if (swept) {
                pm_collide_triangle_ranges_swept(
                    previous_body_state, body_state,
                    previous_body_reference, body_reference, body_mesh,
                    body_node.first_triangle, body_node.triangle_count,
                    previous_collider_state, collider_state, collider_mesh,
                    collider_node.first_triangle,
                    collider_node.triangle_count, vertices, indices, margin,
                    timestep, deep_sweep, body_hinge, collider_hinge,
                    manifold);
            }
            continue;
        }
        const uint required = body_leaf || collider_leaf ? 2u : 4u;
        if (stack_size + required > 256u) {
            overflow = true;
            break;
        }
        if (body_leaf) {
            stack[stack_size++] = uint2(pair.x, collider_node.right);
            stack[stack_size++] = uint2(pair.x, collider_node.left);
        } else if (collider_leaf) {
            stack[stack_size++] = uint2(body_node.right, pair.y);
            stack[stack_size++] = uint2(body_node.left, pair.y);
        } else {
            stack[stack_size++] =
                uint2(body_node.right, collider_node.right);
            stack[stack_size++] =
                uint2(body_node.right, collider_node.left);
            stack[stack_size++] =
                uint2(body_node.left, collider_node.right);
            stack[stack_size++] =
                uint2(body_node.left, collider_node.left);
        }
    }
    if (overflow) {
        manifold = {};
        if (!swept_only)
            pm_collide_triangle_ranges(
                previous_body_state, body_state, body_reference,
                body_mesh, 0u,
                body_mesh.index_count / 3u,
                previous_collider_state, collider_state, collider_mesh, 0u,
                collider_mesh.index_count / 3u, vertices, indices, margin,
                timestep, deep_sweep, fixed_cluster_contact, body_hinge,
                collider_hinge, manifold);
        if (swept) {
            pm_collide_triangle_ranges_swept(
                previous_body_state, body_state,
                previous_body_reference, body_reference, body_mesh, 0u,
                body_mesh.index_count / 3u, previous_collider_state,
                collider_state, collider_mesh, 0u,
                collider_mesh.index_count / 3u, vertices, indices, margin,
                timestep, deep_sweep, body_hinge, collider_hinge, manifold);
        }
    }
    return manifold;
}

// Rigid broad phase is generated from src/slang/rigid_broadphase.slang.

static bool pm_guided_static_contact(
    MetalHingeContactFrame body,
    MetalHingeContactFrame collider) {
    return (body.axial && collider.static_body) ||
           (collider.axial && body.static_body);
}

static void pm_initialize_contact_solve(
    inout MetalNpContactManifold manifold,
    MetalNpRigidState body,
    MetalNpRigidState collider,
    MetalHingeContactFrame body_hinge,
    MetalHingeContactFrame collider_hinge,
    bool persistent_pair) {
    manifold.initial_relative_position = pm_store(
        pm_load(body.position) - pm_load(collider.position));
    manifold.body_fixed_member = body_hinge.fixed_member ? 1u : 0u;
    manifold.collider_fixed_member =
        collider_hinge.fixed_member ? 1u : 0u;
    const bool guided = pm_guided_static_contact(
        body_hinge, collider_hinge);
    for (uint point = 0u; point < manifold.count; ++point) {
        MetalNpContactRecord contact = manifold.contacts[point];
        contact.accumulated_normal_impulse = 0.0f;
        contact.accumulated_friction_impulse = {};
        contact.initial_normal_speed = pm_contact_normal_speed(
            body, collider, pm_load(contact.point), pm_load(contact.normal));
        contact.persistent = guided ||
            (persistent_pair && !body_hinge.present &&
             !collider_hinge.present && !body_hinge.fixed_member &&
             !collider_hinge.fixed_member);
        contact.warm_started = 0u;
        manifold.contacts[point] = contact;
    }
}

static bool pm_motor_constraint_member(
    uint body, StructuredBuffer<MetalNpRigidConstraintResource> constraints,
    uint constraint_capacity) {
    for (uint index = 0u; index < constraint_capacity; ++index) {
        MetalNpRigidConstraintResource constraint =
            constraints[index];
        if (constraint.alive != 0u && constraint.enabled != 0u &&
            constraint.broken == 0u && constraint.type == 7u &&
            (constraint.body_a == body || constraint.body_b == body))
            return true;
    }
    return false;
}

static void pm_stabilize_motor_collider_normals(
    inout MetalNpContactManifold manifold,
    MetalNpRigidState collider_state,
    MetalNpTriangleMeshInfo collider_mesh,
    StructuredBuffer<MetalNpCollisionPlane> solid_planes) {
    if (collider_mesh.solid_plane_count == 0u) return;
    for (uint point = 0u; point < manifold.count; ++point) {
        MetalNpContactRecord contact = manifold.contacts[point];
        const float3 normal = pm_load(contact.normal);
        float best_alignment = 0.99999f;
        float3 stable = normal;
        for (uint face = 0u; face < collider_mesh.solid_plane_count;
             ++face) {
            const float3 plane_normal = pm_rotate(
                collider_state.orientation,
                pm_load(solid_planes[
                    collider_mesh.solid_plane_offset + face].normal));
            const float alignment = dot(normal, plane_normal);
            if (abs(alignment) > best_alignment) {
                best_alignment = abs(alignment);
                stable = alignment >= 0.0f ? plane_normal : -plane_normal;
            }
        }
        contact.normal = pm_store(stable);
        manifold.contacts[point] = contact;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rigid_contact_generate(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint active_index = dispatchThreadId.x;
    MetalNpRigidStepConstants step = narrowStep;
    uint active_pair_count = activePairCountBuffer[0];
    uint substep_index = substepIndexBuffer[0];
    uint64_t epoch_base = epochBaseBuffer[0];
        if (active_index >= active_pair_count) return;
    const uint pair = active_pairs[active_index];
    MetalNpContactManifold output = {};
    const uint body = pair / step.body_count;
    const uint collider = pair % step.body_count;
    float3 previous_body_reference = 0.0f, body_reference = 0.0f, collider_reference = 0.0f;
    pm_rigid_hinge_contact_frame(parameters, step.body_count, body, previous_states,
        constraints, step.constraint_capacity, previous_body_reference);
    const MetalHingeContactFrame body_hinge = pm_rigid_hinge_contact_frame(parameters, step.body_count,
        body, states, constraints, step.constraint_capacity, body_reference);
    const MetalHingeContactFrame collider_hinge = pm_rigid_hinge_contact_frame(parameters, step.body_count,
        collider, states, constraints, step.constraint_capacity, collider_reference);
    MetalNpQuaternion body_axial_orientation = {};
    MetalNpQuaternion collider_axial_orientation = {};
    const MetalGuidedFrame body_guide = pm_rigid_guided_frame(parameters, step.body_count, body,
        states, constraints, step.constraint_capacity, body_axial_orientation);
    const MetalGuidedFrame collider_guide = pm_rigid_guided_frame(parameters, step.body_count, collider,
        states, constraints, step.constraint_capacity, collider_axial_orientation);
    const bool guided_static_pair = (body_guide.active && parameters[collider].motion == 0u)
        || (collider_guide.active && parameters[body].motion == 0u);
    const bool fixed_cluster_contact = body_hinge.fixed_member || collider_hinge.fixed_member;
    MetalNpTriangleMeshInfo body_mesh =
        meshes[parameters[body].mesh_index];
    MetalNpTriangleMeshInfo collider_mesh =
        meshes[parameters[collider].mesh_index];
    MetalMeshLeafInfo body_leaf_info =
        mesh_leaf_infos[parameters[body].mesh_index];
    MetalMeshLeafInfo collider_leaf_info =
        mesh_leaf_infos[parameters[collider].mesh_index];
    const float margin = parameters[body].collision_margin +
                         parameters[collider].collision_margin;
    const bool persistent_pair =
        body_mesh.solid_plane_count != 0u && body_mesh.index_count <= 96u &&
        (parameters[collider].motion != 2u ||
         (collider_mesh.solid_plane_count != 0u &&
          collider_mesh.index_count <= 96u));
    // A discrete face result (including an empty manifold) cannot replace a
    // sweep: narrowSeparated endpoints may still have crossed during this substep.
    const bool face_pair = !guided_static_pair &&
        body_mesh.solid_plane_count != 0u &&
        collider_mesh.solid_plane_count != 0u &&
        body_mesh.index_count <= 96u && collider_mesh.index_count <= 96u &&
        !pm_requires_swept_pair_contact(
            previous_states[body], states[body], body_mesh,
            previous_states[collider], states[collider], collider_mesh,
            margin);
    if (step.cooperative_contact_generation != 0u && face_pair) {
        const uint cache_slot =
            pm_rigid_pair_slot(body, collider, step.body_count);
        const uint64_t epoch = epoch_base + uint64_t(substep_index);
        if (pm_restore_rigid_contact_geometry(
                output, states[body], states[collider], ids[body],
                ids[collider], cache[cache_slot], epoch, step.timestep)) {
            pm_initialize_contact_solve(
                output, states[body], states[collider], body_hinge,
                collider_hinge, persistent_pair);
            output.cached = 1u;
            output.color = cache[cache_slot].color;
            if (step.rigid_sleeping != 0u && output.count != 0u &&
                pm_rigid_is_asleep(sleep_states, body) !=
                    pm_rigid_is_asleep(sleep_states, collider)) {
                pm_rigid_wake(sleep_states, body);
                if (parameters[collider].inverse_mass > 0.0f)
                    pm_rigid_wake(sleep_states, collider);
            }
            manifolds[active_index] = output;
            return;
        }
    }
    MetalNpContactManifold face_manifold = {};
    if (face_pair && pm_convex_face_manifold(
            states[body], body_mesh, states[collider], collider_mesh,
            vertices, indices, solid_planes, margin, face_manifold)) {
        pm_reduce_collinear_face_contacts(face_manifold);
        output = face_manifold;
        pm_initialize_contact_solve(
            output, states[body], states[collider], body_hinge,
            collider_hinge, persistent_pair);
        if (step.rigid_sleeping != 0u && output.count != 0u &&
            pm_rigid_is_asleep(sleep_states, body) !=
                pm_rigid_is_asleep(sleep_states, collider)) {
            pm_rigid_wake(sleep_states, body);
            if (parameters[collider].inverse_mass > 0.0f)
                pm_rigid_wake(sleep_states, collider);
        }
        manifolds[active_index] = output;
        return;
    }
    // Defer bounded unconstrained leaf products to one cooperative group.
    // Deep static sweeps retain the existing recovery/fallback path.
    const uint64_t leaf_product =
        uint64_t(body_leaf_info.count) * collider_leaf_info.count;
    const bool deep_static_sweep =
        (parameters[body].motion == 0u || parameters[collider].motion == 0u) &&
        pm_requires_swept_pair_contact(
            previous_states[body], states[body], body_mesh,
            previous_states[collider], states[collider], collider_mesh,
            max(margin * 8.0f, 0.25f));
    if (step.cooperative_contact_generation != 0u && leaf_product > 0u &&
        leaf_product <= 512u && !deep_static_sweep && !body_hinge.present &&
        !collider_hinge.present && !fixed_cluster_contact) {
        output.count = 0xffffffffu;
        manifolds[active_index] = output;
        return;
    }
    output = pm_collide_meshes(
        previous_states[body], states[body], previous_body_reference,
        body_reference, body_mesh,
        previous_states[collider], states[collider], collider_mesh, vertices,
        indices, bvh_nodes, body_leaf_info, collider_leaf_info, bvh_leaves,
        margin, step.timestep, false, fixed_cluster_contact, false,
        body_hinge, collider_hinge);
    bool has_approaching_contact = false;
    bool body_reference_crossed_contact = false;
    for (uint contact_index = 0u; contact_index < output.count;
         ++contact_index) {
        MetalNpContactRecord contact =
            output.contacts[contact_index];
        const float3 contact_normal = pm_load(contact.normal);
        const float previous_reference_side = dot(
            contact_normal,
            previous_body_reference - pm_load(contact.point));
        const float current_reference_side = dot(
            contact_normal, body_reference - pm_load(contact.point));
        body_reference_crossed_contact =
            body_reference_crossed_contact ||
            previous_reference_side * current_reference_side < 0.0f;
        has_approaching_contact = has_approaching_contact ||
            pm_contact_normal_speed(
                states[body], states[collider], pm_load(contact.point),
                contact_normal) < -pm_rigid_surface_tolerance;
    }
    // Keep the numerical recovery confined to extreme dynamic/static sweeps.
    // CUDA does not replace a valid ordinary dynamic-pair manifold.
    const bool deep_static_pair =
        (parameters[body].motion == 0u ||
         parameters[collider].motion == 0u) &&
        pm_requires_swept_pair_contact(
            previous_states[body], states[body], body_mesh,
            previous_states[collider], states[collider], collider_mesh,
            max(margin * 8.0f, 0.25f));
    const bool pair_requires_predictive_replacement =
        pm_requires_swept_pair_contact(
            previous_states[body], states[body], body_mesh,
            previous_states[collider], states[collider], collider_mesh,
            margin);
    const bool crossed_convex_surface =
        body_reference_crossed_contact &&
        !body_hinge.present && !collider_hinge.present &&
        ((body_mesh.solid_plane_count != 0u) !=
         (collider_mesh.solid_plane_count != 0u)) &&
        pair_requires_predictive_replacement;
    const bool needs_static_tunnel_recovery = deep_static_pair &&
        (output.count == 0u ||
         (!has_approaching_contact && body_reference_crossed_contact) ||
         crossed_convex_surface);
    if (needs_static_tunnel_recovery) {
        const bool swept_only =
            !fixed_cluster_contact && output.count != 0u &&
            pair_requires_predictive_replacement &&
            (!has_approaching_contact || crossed_convex_surface);
        MetalNpContactManifold fallback = pm_collide_meshes(
            previous_states[body], states[body], previous_body_reference,
            body_reference, body_mesh,
            previous_states[collider], states[collider], collider_mesh,
            vertices, indices, bvh_nodes, body_leaf_info,
            collider_leaf_info, bvh_leaves, margin, step.timestep, true,
            fixed_cluster_contact, swept_only, body_hinge,
            collider_hinge);
        if (swept_only) {
            uint retained = 0u;
            for (uint contact_index = 0u;
                 contact_index < fallback.count; ++contact_index) {
                const MetalNpContactRecord contact =
                    fallback.contacts[contact_index];
                if (pm_contact_normal_speed(
                        states[body], states[collider],
                        pm_load(contact.point), pm_load(contact.normal)) >
                    pm_rigid_surface_tolerance)
                    continue;
                fallback.contacts[retained++] = contact;
            }
            fallback.count = retained;
        }
        output = fallback;
    }
    if (pm_motor_constraint_member(
            body, constraints, step.constraint_capacity))
        pm_stabilize_motor_collider_normals(
            output, states[collider], collider_mesh, solid_planes);
    pm_initialize_contact_solve(
        output, states[body], states[collider], body_hinge,
        collider_hinge, persistent_pair);
    if (step.rigid_sleeping != 0u && output.count != 0u &&
        pm_rigid_is_asleep(sleep_states, body) !=
            pm_rigid_is_asleep(sleep_states, collider)) {
        pm_rigid_wake(sleep_states, body);
        if (parameters[collider].inverse_mass > 0.0f)
            pm_rigid_wake(sleep_states, collider);
    }
    manifolds[active_index] = output;
}

// The swept triangle path linearly interpolates endpoint vertices. The
// union of endpoint projections conservatively bounds that same motion.
static float2 pm_project_mesh_bounds(
    MetalNpRigidState state,
    MetalNpTriangleMeshInfo mesh, float3 axis) {
    const float3 center =
        (pm_load(mesh.minimum) + pm_load(mesh.maximum)) * 0.5f;
    const float3 extent =
        (pm_load(mesh.maximum) - pm_load(mesh.minimum)) * 0.5f;
    const float3 world_center =
        pm_load(state.position) + pm_rotate(state.orientation, center);
    const float3 local_axis =
        pm_rotate(pm_quaternion_conjugate(state.orientation), axis);
    const float projection = dot(world_center, axis);
    const float radius = dot(abs(local_axis), extent);
    return float2(projection - radius, projection + radius);
}

static float pm_bounds_coordinate_scale(
    MetalNpRigidState previous,
    MetalNpRigidState state,
    MetalNpTriangleMeshInfo mesh) {
    // Include unprojected coordinates: a projection can cancel large terms.
    const float3 scale = max(abs(pm_load(previous.position)),
                             abs(pm_load(state.position))) +
        max(abs(pm_load(mesh.minimum)), abs(pm_load(mesh.maximum)));
    return scale.x + scale.y + scale.z;
}

// Preserve CUDA's lane-grouped leaf order exactly: each lane evaluates one
// leaf pair independently, then lane zero merges the bounded manifolds in
// that same order. No atomically selected contact order or new CCD tolerance.
[shader("compute")]
[numthreads(32, 1, 1)]
public void pm_rigid_contact_generate_cooperative(
    uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    MetalNpRigidStepConstants step = narrowStep;
    uint active_pair_count = activePairCountBuffer[0];
    const uint active_index = group.x;
    if (active_index >= active_pair_count ||
        manifolds[active_index].count != 0xffffffffu) return;
    const uint pair = active_pairs[active_index];
    const uint body = pair / step.body_count;
    const uint collider = pair % step.body_count;
    MetalNpTriangleMeshInfo body_mesh =
        meshes[parameters[body].mesh_index];
    MetalNpTriangleMeshInfo collider_mesh =
        meshes[parameters[collider].mesh_index];
    MetalMeshLeafInfo body_leaves =
        mesh_leaf_infos[parameters[body].mesh_index];
    MetalMeshLeafInfo collider_leaves =
        mesh_leaf_infos[parameters[collider].mesh_index];
    const uint candidate_count = body_leaves.count * collider_leaves.count;
    const float margin = parameters[body].collision_margin +
                         parameters[collider].collision_margin;
    const float separation = max(margin * 2.0f, 1.0e-4f);
    const bool swept = pm_requires_swept_pair_contact(
        previous_states[body], states[body], body_mesh,
        previous_states[collider], states[collider], collider_mesh, margin);
    // Reject only narrowSeparated endpoint envelopes. This bounds both discrete
    // geometry and the existing linearly interpolated triangle sweep.

    if (lane == 0u)
        InterlockedExchange(narrowSeparated, 0u);
    GroupMemoryBarrierWithGroupSync();
    if (lane < 6u) {
        const uint side = lane / 3u;
        float3 local_axis = 0.0f;
        local_axis[lane % 3u] = 1.0f;
        const float3 axis = pm_rotate(
            states[side == 0u ? body : collider].orientation, local_axis);
        float2 first = pm_project_mesh_bounds(states[body], body_mesh, axis);
        float2 second = pm_project_mesh_bounds(states[collider], collider_mesh, axis);
        if (swept) {
            const float2 previous_first = pm_project_mesh_bounds(
                previous_states[body], body_mesh, axis);
            const float2 previous_second = pm_project_mesh_bounds(
                previous_states[collider], collider_mesh, axis);
            first = float2(min(first.x, previous_first.x),
                           max(first.y, previous_first.y));
            second = float2(min(second.x, previous_second.x),
                            max(second.y, previous_second.y));
        }
        const float scale = pm_bounds_coordinate_scale(
            previous_states[body], states[body], body_mesh) +
            pm_bounds_coordinate_scale(
                previous_states[collider], states[collider], collider_mesh);
        const float padding = margin * length(axis) + 1.0e-5f * (1.0f + scale);
        if (first.x > second.y + padding || second.x > first.y + padding)
            InterlockedExchange(narrowSeparated, 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (narrowSeparated != 0u) {
        if (lane == 0u) {
            MetalNpContactManifold empty = {};
            manifolds[active_index] = empty;
        }
        return;
    }
    MetalHingeContactFrame body_hinge = {};
    MetalHingeContactFrame collider_hinge = {};
    body_hinge.static_body = parameters[body].motion == 0u;
    collider_hinge.static_body = parameters[collider].motion == 0u;

    MetalNpContactManifold result = {};
    for (uint base = 0u; base < candidate_count; base += 32u) {
        const uint ordinal = base + lane;
        MetalNpContactManifold local = {};
        if (ordinal < candidate_count) {
            // Invert the original for(lane<128) for(candidate+=128) order.
            const uint runs = candidate_count / 128u;
            const uint extra = candidate_count % 128u;
            const uint long_end = extra * (runs + 1u);
            const uint source_lane = ordinal < long_end
                ? ordinal / (runs + 1u)
                : extra + (ordinal - long_end) / max(runs, 1u);
            const uint in_lane = ordinal < long_end
                ? ordinal % (runs + 1u)
                : (ordinal - long_end) % max(runs, 1u);
            const uint candidate = source_lane + in_lane * 128u;
            MetalBvhNode body_node = bvh_nodes[bvh_leaves[
                body_leaves.offset + candidate / collider_leaves.count]];
            MetalBvhNode collider_node = bvh_nodes[bvh_leaves[
                collider_leaves.offset + candidate % collider_leaves.count]];
            float3 body_minimum, body_maximum;
            float3 collider_minimum, collider_maximum;
            if (swept) {
                pm_transformed_motion_bounds(previous_states[body], states[body],
                    pm_load(body_node.minimum), pm_load(body_node.maximum), margin,
                    body_minimum, body_maximum);
                pm_transformed_motion_bounds(previous_states[collider], states[collider],
                    pm_load(collider_node.minimum), pm_load(collider_node.maximum), 0.0f,
                    collider_minimum, collider_maximum);
            } else {
                pm_transformed_bounds(states[body], pm_load(body_node.minimum),
                    pm_load(body_node.maximum), margin, body_minimum, body_maximum);
                pm_transformed_bounds(states[collider], pm_load(collider_node.minimum),
                    pm_load(collider_node.maximum), 0.0f, collider_minimum, collider_maximum);
            }
            if (pm_bounds_overlap(body_minimum, body_maximum,
                                  collider_minimum, collider_maximum)) {
                pm_collide_triangle_ranges(
                    previous_states[body], states[body], pm_load(states[body].position),
                    body_mesh, body_node.first_triangle, body_node.triangle_count,
                    previous_states[collider], states[collider], collider_mesh,
                    collider_node.first_triangle, collider_node.triangle_count,
                    vertices, indices, margin, step.timestep, false, false,
                    body_hinge, collider_hinge, local);
                if (swept)
                    pm_collide_triangle_ranges_swept(
                        previous_states[body], states[body],
                        pm_load(previous_states[body].position), pm_load(states[body].position),
                        body_mesh, body_node.first_triangle, body_node.triangle_count,
                        previous_states[collider], states[collider], collider_mesh,
                        collider_node.first_triangle, collider_node.triangle_count,
                        vertices, indices, margin, step.timestep, false,
                        body_hinge, collider_hinge, local);
            }
        }
        narrowScratchCounts[lane] = local.count;
        for (uint point = 0u; point < local.count; ++point) {
            narrowScratchContacts[lane][point] = {
                local.contacts[point].point,
                local.contacts[point].normal,
                local.contacts[point].penetration,
                local.contacts[point].impact_fraction};
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            for (uint item = 0u; item < min(32u, candidate_count - base); ++item)
                for (uint point = 0u; point < narrowScratchCounts[item]; ++point) {
                    const MetalContactGeometry geometry =
                        narrowScratchContacts[item][point];
                    MetalNpContactRecord contact = {};
                    contact.point = geometry.point;
                    contact.normal = geometry.normal;
                    contact.penetration = geometry.penetration;
                    contact.found = 1u;
                    contact.impact_fraction = geometry.impact_fraction;
                    pm_add_pair_manifold_contact(
                        result, contact, separation, body_hinge, collider_hinge);
                }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane != 0u) return;
    manifolds[active_index] = result;
    const bool persistent = body_mesh.solid_plane_count != 0u &&
        body_mesh.index_count <= 96u &&
        (parameters[collider].motion != 2u ||
         (collider_mesh.solid_plane_count != 0u &&
          collider_mesh.index_count <= 96u));
    pm_initialize_contact_solve(
        manifolds[active_index], states[body], states[collider],
        body_hinge, collider_hinge, persistent);
    if (step.rigid_sleeping != 0u && result.count != 0u &&
        pm_rigid_is_asleep(sleep_states, body) !=
            pm_rigid_is_asleep(sleep_states, collider)) {
        pm_rigid_wake(sleep_states, body);
        if (parameters[collider].inverse_mass > 0.0f)
            pm_rigid_wake(sleep_states, collider);
    }
}
