// SPDX-License-Identifier: MIT
#include <metal_stdlib>

using namespace metal;

// Keep this layout intentionally scalar. SIMD float3 has 16-byte alignment,
// while the public C++ contract is a packed 12-byte vector.
struct PMPackedVec3 {
    float x;
    float y;
    float z;
};

struct PMPackedVec2 {
    float x;
    float y;
};

static_assert(sizeof(PMPackedVec3) == 12);
static_assert(alignof(PMPackedVec3) == 4);

kernel void pm_noop(uint thread_index [[thread_position_in_grid]]) {
    (void)thread_index;
}

struct PMQuaternion {
    float x;
    float y;
    float z;
    float w;
};

struct PMRigidBodyState {
    PMPackedVec3 position;
    PMQuaternion orientation;
    PMPackedVec3 linear_velocity;
    PMPackedVec3 angular_velocity;
};

struct PMRigidParameters {
    uint motion;
    float inverse_mass;
    PMPackedVec3 inverse_inertia;
    float linear_damping;
    float angular_damping;
    float maximum_linear_speed;
    float maximum_angular_speed;
    uint has_kinematic_target;
    PMRigidBodyState kinematic_target;
    uint mesh_index;
    float friction;
    float restitution;
    float collision_margin;
};

struct PMTriangleMeshInfo {
    uint vertex_offset;
    uint vertex_count;
    uint index_offset;
    uint index_count;
    PMPackedVec3 minimum;
    PMPackedVec3 maximum;
    PMPackedVec3 bounding_center;
    float radius;
    uint bvh_node_offset;
    uint bvh_node_count;
    uint solid_plane_offset;
    uint solid_plane_count;
};

struct PMCollisionPlane {
    PMPackedVec3 normal;
    float offset;
};

struct PMContactRecord {
    PMPackedVec3 point;
    PMPackedVec3 normal;
    float penetration;
    uint found;
    float accumulated_normal_impulse;
    PMPackedVec3 accumulated_friction_impulse;
    float initial_normal_speed;
    float impact_fraction;
    uint persistent;
    uint warm_started;
};

struct PMContactManifold {
    PMContactRecord contacts[8];
    uint count;
    uint event_offset;
    uint color;
    PMPackedVec3 initial_relative_position;
    uint face_patch;
    uint body_fixed_member;
    uint collider_fixed_member;
    uint cached;
};

struct PMCachedContact {
    PMPackedVec3 local_point;
    PMPackedVec3 normal;
    float normal_impulse;
    PMPackedVec3 friction_impulse;
};

struct PMCachedContactPair {
    uint body_index;
    uint body_generation;
    uint collider_index;
    uint collider_generation;
    ulong epoch;
    float timestep;
    uint count;
    PMCachedContact contacts[8];
};

struct PMBvhNode {
    PMPackedVec3 minimum;
    PMPackedVec3 maximum;
    uint left;
    uint right;
    uint first_triangle;
    uint triangle_count;
};

struct PMMeshLeafInfo {
    uint offset;
    uint count;
};

struct PMWorldAabb {
    PMPackedVec3 minimum;
    PMPackedVec3 maximum;
};

struct PMHandle {
    uint index;
    uint generation;
};

struct PMRigidContactEvent {
    PMHandle body;
    PMHandle collider;
    PMPackedVec3 position;
    PMPackedVec3 normal;
    float penetration;
    float normal_impulse;
    PMPackedVec3 friction_impulse;
};

struct PMContactEvent {
    PMHandle fluid;
    uint stable_particle_id;
    PMHandle rigid_body;
    PMPackedVec3 position;
    PMPackedVec3 normal;
    float normal_impulse;
};

struct PMStepConstants {
    float timestep;
    PMPackedVec3 gravity;
    uint body_count;
    uint constraint_capacity;
    uint collect_rigid_contacts;
    uint rigid_event_capacity;
    uint substeps;
    uint ordinary_rigid_stack;
};

struct PMRigidConstraintResource {
    uint generation;
    uint alive;
    uint type;
    uint body_a;
    uint body_b;
    uint enabled;
    uint broken;
    PMPackedVec3 local_anchor_a;
    PMPackedVec3 local_anchor_b;
    PMQuaternion local_orientation_a;
    PMQuaternion local_orientation_b;
    float breaking_impulse_threshold;
    float applied_impulse;
    uint linear_limit_axes;
    PMPackedVec3 linear_limit_lower;
    PMPackedVec3 linear_limit_upper;
    uint angular_limit_axes;
    PMPackedVec3 angular_limit_lower;
    PMPackedVec3 angular_limit_upper;
    uint linear_spring_axes;
    PMPackedVec3 linear_spring_stiffness;
    PMPackedVec3 linear_spring_damping;
    uint angular_spring_axes;
    PMPackedVec3 angular_spring_stiffness;
    PMPackedVec3 angular_spring_damping;
    uint linear_motor_enabled;
    uint angular_motor_enabled;
    float linear_target_velocity;
    float linear_maximum_impulse;
    float angular_target_velocity;
    float angular_maximum_impulse;
    uint solver_iterations;
    uint disable_collisions;
};

struct PMRigidConstraintAxisGeometry {
    PMPackedVec3 axis;
    PMPackedVec3 inverse_angular_a;
    PMPackedVec3 inverse_angular_b;
    float angular_denominator;
    float linear_denominator;
};

struct PMRigidConstraintGeometry {
    uint valid;
    PMPackedVec3 arm_a;
    PMPackedVec3 arm_b;
    PMPackedVec3 anchor_error;
    PMPackedVec3 rotation_error;
    PMPackedVec3 hinge_alignment_error;
    PMPackedVec3 piston_alignment_error;
    PMRigidConstraintAxisGeometry axes[3];
};

struct PMRigidCompound {
    uint root;
    uint member_count;
    uint eligible;
    uint blocked;
    PMPackedVec3 center;
    float inverse_mass;
    PMPackedVec3 inverse_inertia[3];
    uint projection_root;
    uint projection_movable;
    PMPackedVec3 projection_translation;
};

static_assert(sizeof(PMQuaternion) == 16);
static_assert(sizeof(PMRigidBodyState) == 52);
static_assert(sizeof(PMRigidParameters) == 108);
static_assert(sizeof(PMTriangleMeshInfo) == 72);
static_assert(sizeof(PMCollisionPlane) == 16);
static_assert(sizeof(PMContactManifold) == 552);
static_assert(sizeof(PMCachedContact) == 40);
static_assert(sizeof(PMCachedContactPair) == 352);
static_assert(sizeof(PMBvhNode) == 40);
static_assert(sizeof(PMMeshLeafInfo) == 8);
static_assert(sizeof(PMWorldAabb) == 24);
static_assert(sizeof(PMHandle) == 8);
static_assert(sizeof(PMRigidContactEvent) == 60);
static_assert(sizeof(PMContactEvent) == 48);
static_assert(sizeof(PMStepConstants) == 40);
static_assert(sizeof(PMRigidConstraintResource) == 236);
static_assert(sizeof(PMRigidConstraintAxisGeometry) == 44);
static_assert(sizeof(PMRigidConstraintGeometry) == 208);
static_assert(sizeof(PMRigidCompound) == 88);

static float3 pm_load(PMPackedVec3 value) {
    return float3(value.x, value.y, value.z);
}

static PMPackedVec3 pm_store(float3 value) {
    return {value.x, value.y, value.z};
}

static float3 pm_rotate(PMQuaternion q, float3 point) {
    const float3 vector_part = float3(q.x, q.y, q.z);
    const float3 twice_cross = 2.0f * cross(vector_part, point);
    return point +
        (q.w * twice_cross + cross(vector_part, twice_cross));
}

static PMQuaternion pm_quaternion_multiply(PMQuaternion a, PMQuaternion b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

static PMQuaternion pm_quaternion_conjugate(PMQuaternion value) {
    return {-value.x, -value.y, -value.z, value.w};
}

static PMQuaternion pm_quaternion_normalize(PMQuaternion value) {
    const float inverse = rsqrt(
        max(value.x * value.x + value.y * value.y +
                value.z * value.z + value.w * value.w,
            1.0e-12f));
    return {value.x * inverse, value.y * inverse, value.z * inverse,
            value.w * inverse};
}

static void pm_apply_orientation_delta(device PMRigidBodyState &state,
                                       float3 delta) {
    const PMQuaternion rotation = pm_quaternion_normalize(
        {0.5f * delta.x, 0.5f * delta.y, 0.5f * delta.z, 1.0f});
    state.orientation = pm_quaternion_normalize(
        pm_quaternion_multiply(rotation, state.orientation));
}

static float3 pm_limit(float3 value, float maximum_length) {
    const float squared_length = dot(value, value);
    const float squared_maximum = maximum_length * maximum_length;
    return squared_length > squared_maximum
               ? value * (maximum_length * rsqrt(squared_length))
               : value;
}

static float3 pm_inverse_inertia_mul(
    device const PMRigidParameters &body,
    device const PMRigidBodyState &state, float3 value) {
    const float3 local = pm_rotate(
        pm_quaternion_conjugate(state.orientation), value);
    return pm_rotate(state.orientation,
                     local * pm_load(body.inverse_inertia));
}

static float3 pm_inverse_inertia_mul_orientation(
    device const PMRigidParameters &body, PMQuaternion orientation,
    float3 value) {
    const float3 local = pm_rotate(
        pm_quaternion_conjugate(orientation), value);
    return pm_rotate(orientation,
                     local * pm_load(body.inverse_inertia));
}

static float3 pm_quaternion_delta_velocity(
    PMQuaternion from, PMQuaternion to, float timestep) {
    PMQuaternion delta = pm_quaternion_multiply(
        to, pm_quaternion_conjugate(from));
    if (delta.w < 0.0f)
        delta = {-delta.x, -delta.y, -delta.z, -delta.w};
    delta = pm_quaternion_normalize(delta);
    const float vector_size = length(float3(delta.x, delta.y, delta.z));
    if (vector_size <= 1.0e-6f || timestep <= 0.0f) return 0.0f;
    const float angle =
        2.0f * atan2(vector_size, clamp(delta.w, -1.0f, 1.0f));
    return float3(delta.x, delta.y, delta.z) *
           (angle / (vector_size * timestep));
}

struct PMGuidedFrame {
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

static PMGuidedFrame pm_rigid_guided_frame(
    device const PMRigidParameters *parameters, uint body_count, uint body,
    device const PMRigidBodyState *states,
    device const PMRigidConstraintResource *constraints,
    uint constraint_capacity, thread PMQuaternion &axial_orientation) {
    PMGuidedFrame result{};
    axial_orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    if (body >= body_count || parameters[body].inverse_mass <= 0.0f)
        return result;
    uint incident_count = 0u;
    for (uint index = 0u; index < constraint_capacity; ++index) {
        device const PMRigidConstraintResource &constraint =
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
        const PMQuaternion local_orientation = is_a
            ? constraint.local_orientation_a
            : constraint.local_orientation_b;
        const float3 other_local_anchor = pm_load(
            is_a ? constraint.local_anchor_b : constraint.local_anchor_a);
        const PMQuaternion other_local_orientation = is_a
            ? constraint.local_orientation_b
            : constraint.local_orientation_a;
        device const PMRigidBodyState &other_state = states[other];
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

static float pm_guided_inverse_moment(
    device const PMRigidParameters &body,
    device const PMRigidBodyState &state,
    thread const PMGuidedFrame &frame) {
    if (!frame.axial_rotation || body.inverse_mass <= 1.0e-6f) return 0.0f;
    const float3 local_axis = pm_rotate(
        pm_quaternion_conjugate(state.orientation), frame.axis);
    const float3 inverse_inertia = pm_load(body.inverse_inertia);
    const float center_moment =
        local_axis.x * local_axis.x /
            max(inverse_inertia.x, 1.0e-6f) +
        local_axis.y * local_axis.y /
            max(inverse_inertia.y, 1.0e-6f) +
        local_axis.z * local_axis.z /
            max(inverse_inertia.z, 1.0e-6f);
    const float3 center_arm = pm_load(state.position) - frame.anchor;
    const float3 perpendicular = center_arm -
        frame.axis * dot(center_arm, frame.axis);
    const float pivot_moment = center_moment +
        dot(perpendicular, perpendicular) / body.inverse_mass;
    return pivot_moment > 1.0e-6f ? 1.0f / pivot_moment : 0.0f;
}

kernel void pm_rigid_integrate(
    device PMRigidBodyState *states [[buffer(0)]],
    device PMRigidParameters *parameters [[buffer(1)]],
    device PMPackedVec3 *forces [[buffer(2)]],
    device PMPackedVec3 *torques [[buffer(3)]],
    constant PMStepConstants &step [[buffer(4)]],
    device const PMRigidConstraintResource *constraints [[buffer(9)]],
    device PMRigidBodyState *previous_states [[buffer(14)]],
    device const uint &substep_index [[buffer(20)]],
    uint body_index [[thread_position_in_grid]]) {
    if (body_index >= step.body_count) {
        return;
    }

    device PMRigidParameters &body = parameters[body_index];
    device PMRigidBodyState &state = states[body_index];
    previous_states[body_index] = state;
    if (body.motion == 0u) {
        state.linear_velocity = {};
        state.angular_velocity = {};
    } else if (body.motion == 1u) {
        if (body.has_kinematic_target != 0u) {
            const uint remaining_substeps =
                max(1u, step.substeps - substep_index);
            const float fraction = 1.0f / float(remaining_substeps);
            const float3 previous_position = pm_load(state.position);
            const PMQuaternion previous_orientation = state.orientation;
            state.position = pm_store(
                previous_position +
                (pm_load(body.kinematic_target.position) -
                 previous_position) * fraction);
            PMQuaternion target = body.kinematic_target.orientation;
            const float orientation_dot =
                previous_orientation.x * target.x +
                previous_orientation.y * target.y +
                previous_orientation.z * target.z +
                previous_orientation.w * target.w;
            if (orientation_dot < 0.0f)
                target = {-target.x, -target.y, -target.z, -target.w};
            state.orientation = pm_quaternion_normalize({
                previous_orientation.x +
                    (target.x - previous_orientation.x) * fraction,
                previous_orientation.y +
                    (target.y - previous_orientation.y) * fraction,
                previous_orientation.z +
                    (target.z - previous_orientation.z) * fraction,
                previous_orientation.w +
                    (target.w - previous_orientation.w) * fraction});
            state.linear_velocity = pm_store(
                (pm_load(state.position) - previous_position) /
                step.timestep);
            state.angular_velocity = pm_store(
                pm_quaternion_delta_velocity(
                    previous_orientation, state.orientation,
                    step.timestep));
            if (remaining_substeps == 1u)
                body.has_kinematic_target = 0u;
        } else {
            state.linear_velocity = {};
            state.angular_velocity = {};
        }
    } else if (body.motion == 2u) {
        const float dt = step.timestep;
        float3 linear_velocity = pm_load(state.linear_velocity);
        linear_velocity +=
            (pm_load(step.gravity) + pm_load(forces[body_index]) *
                                         body.inverse_mass) *
            dt;
        linear_velocity /= 1.0f + body.linear_damping * dt;
        linear_velocity =
            pm_limit(linear_velocity, body.maximum_linear_speed);
        state.linear_velocity = pm_store(linear_velocity);

        float3 angular_velocity = pm_load(state.angular_velocity);
        angular_velocity += pm_inverse_inertia_mul(
                                body, state,
                                pm_load(torques[body_index])) *
                            dt;
        angular_velocity /= 1.0f + body.angular_damping * dt;
        angular_velocity =
            pm_limit(angular_velocity, body.maximum_angular_speed);
        state.angular_velocity = pm_store(angular_velocity);

        PMQuaternion axial_orientation{};
        const PMGuidedFrame guide = pm_rigid_guided_frame(
            parameters, step.body_count, body_index, states, constraints,
            step.constraint_capacity, axial_orientation);
        if (guide.active) {
            const float3 local_omega = pm_rotate(
                pm_quaternion_conjugate(state.orientation),
                angular_velocity);
            const float3 inverse_inertia = pm_load(body.inverse_inertia);
            const float3 angular_momentum = pm_rotate(
                state.orientation,
                float3(
                    local_omega.x / max(inverse_inertia.x, 1.0e-6f),
                    local_omega.y / max(inverse_inertia.y, 1.0e-6f),
                    local_omega.z / max(inverse_inertia.z, 1.0e-6f)));
            const float3 arm = pm_load(state.position) - guide.anchor;
            const float spin = dot(
                guide.axis,
                angular_momentum +
                    cross(arm, linear_velocity / body.inverse_mass)) *
                pm_guided_inverse_moment(body, state, guide);
            angular_velocity = guide.axis * spin;
            linear_velocity =
                guide.axis * dot(linear_velocity, guide.axis) +
                cross(angular_velocity, arm);
            state.angular_velocity = pm_store(angular_velocity);
            state.linear_velocity = pm_store(linear_velocity);
        }

        state.position =
            pm_store(pm_load(state.position) + linear_velocity * dt);

        const PMQuaternion angular{
            angular_velocity.x, angular_velocity.y, angular_velocity.z,
            0.0f};
        const PMQuaternion derivative =
            pm_quaternion_multiply(angular, state.orientation);
        state.orientation = pm_quaternion_normalize({
            state.orientation.x + 0.5f * derivative.x * dt,
            state.orientation.y + 0.5f * derivative.y * dt,
            state.orientation.z + 0.5f * derivative.z * dt,
            state.orientation.w + 0.5f * derivative.w * dt});
        if (guide.active) {
            if (guide.axial_rotation) {
                const PMQuaternion relative = pm_quaternion_multiply(
                    state.orientation,
                    pm_quaternion_conjugate(axial_orientation));
                const float twist = dot(
                    float3(relative.x, relative.y, relative.z),
                    guide.axis);
                const PMQuaternion rotation = pm_quaternion_normalize({
                    guide.axis.x * twist,
                    guide.axis.y * twist,
                    guide.axis.z * twist,
                    relative.w});
                state.orientation = pm_quaternion_normalize(
                    pm_quaternion_multiply(rotation, axial_orientation));
            } else {
                state.orientation = axial_orientation;
            }
            const float3 anchor = pm_load(state.position) +
                pm_rotate(state.orientation, guide.local_anchor);
            state.position = pm_store(
                guide.anchor +
                    guide.axis * dot(anchor - guide.anchor, guide.axis) -
                    pm_rotate(state.orientation, guide.local_anchor));
        }
    }
}

static void pm_load_rigid_contact_cache(
    device PMContactManifold &manifold,
    device const PMRigidBodyState &body_state,
    device const PMHandle &body_id,
    device const PMHandle &collider_id,
    device const PMCachedContactPair &saved,
    ulong epoch, float timestep) {
    if (saved.epoch + 1ul != epoch ||
        saved.body_index != body_id.index ||
        saved.body_generation != body_id.generation ||
        saved.collider_index != collider_id.index ||
        saved.collider_generation != collider_id.generation ||
        abs(saved.timestep - timestep) > 1.0e-7f)
        return;
    uint used = 0u;
    for (uint point = 0u; point < manifold.count; ++point) {
        device PMContactRecord &contact = manifold.contacts[point];
        if (contact.persistent == 0u) continue;
        const float3 local = pm_rotate(
            pm_quaternion_conjugate(body_state.orientation),
            pm_load(contact.point) - pm_load(body_state.position));
        float nearest = 0.02f * 0.02f;
        uint match = 8u;
        for (uint previous = 0u;
             previous < min(saved.count, 8u); ++previous) {
            if ((used & (1u << previous)) != 0u ||
                dot(pm_load(contact.normal),
                    pm_load(saved.contacts[previous].normal)) < 0.99f)
                continue;
            const float3 delta =
                local - pm_load(saved.contacts[previous].local_point);
            const float distance = dot(delta, delta);
            if (distance < nearest) {
                nearest = distance;
                match = previous;
            }
        }
        if (match == 8u) continue;
        used |= 1u << match;
        manifold.cached = 1u;
        device const PMCachedContact &previous = saved.contacts[match];
        contact.accumulated_normal_impulse = previous.normal_impulse;
        const float3 friction = pm_load(previous.friction_impulse);
        const float3 normal = pm_load(contact.normal);
        contact.accumulated_friction_impulse = pm_store(
            friction - normal * dot(friction, normal));
    }
}

static void pm_save_rigid_contact_cache(
    device const PMContactManifold &manifold,
    device const PMRigidBodyState &body_state,
    device const PMHandle &body_id,
    device const PMHandle &collider_id,
    device PMCachedContactPair &saved,
    ulong epoch, float timestep) {
    saved.body_index = body_id.index;
    saved.body_generation = body_id.generation;
    saved.collider_index = collider_id.index;
    saved.collider_generation = collider_id.generation;
    saved.epoch = epoch;
    saved.timestep = timestep;
    saved.count = manifold.count;
    for (uint point = 0u; point < manifold.count; ++point) {
        device const PMContactRecord &contact = manifold.contacts[point];
        device PMCachedContact &output = saved.contacts[point];
        output.local_point = pm_store(pm_rotate(
            pm_quaternion_conjugate(body_state.orientation),
            pm_load(contact.point) - pm_load(body_state.position)));
        output.normal = contact.normal;
        output.normal_impulse = contact.accumulated_normal_impulse;
        output.friction_impulse = contact.accumulated_friction_impulse;
    }
}

kernel void pm_rigid_advance_substep(
    device const PMRigidBodyState *states [[buffer(0)]],
    constant PMStepConstants &step [[buffer(4)]],
    device const PMContactManifold *manifolds [[buffer(8)]],
    device const PMHandle *ids [[buffer(10)]],
    device const uint *active_pairs [[buffer(18)]],
    device const uint &active_pair_count [[buffer(19)]],
    device uint &substep_index [[buffer(20)]],
    device PMCachedContactPair *cache [[buffer(27)]],
    device const ulong &epoch_base [[buffer(28)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    const ulong epoch = epoch_base + ulong(substep_index);
    for (uint active = 0u; active < active_pair_count; ++active) {
        const uint pair = active_pairs[active];
        const uint body = pair / step.body_count;
        const uint collider = pair % step.body_count;
        pm_save_rigid_contact_cache(
            manifolds[pair], states[body], ids[body], ids[collider],
            cache[pair], epoch, step.timestep);
    }
    ++substep_index;
}

kernel void pm_rigid_clear_accumulators(
    device PMPackedVec3 *forces [[buffer(2)]],
    device PMPackedVec3 *torques [[buffer(3)]],
    constant PMStepConstants &step [[buffer(4)]],
    uint body_index [[thread_position_in_grid]]) {
    if (body_index >= step.body_count) {
        return;
    }
    forces[body_index] = {0.0f, 0.0f, 0.0f};
    torques[body_index] = {0.0f, 0.0f, 0.0f};
}

struct PMSymmetricMatrix3 {
    float xx;
    float xy;
    float xz;
    float yy;
    float yz;
    float zz;
};

static uint pm_rigid_compound_root(
    device const PMRigidCompound *compounds, uint body) {
    while (compounds[body].root != body) body = compounds[body].root;
    return body;
}

static bool pm_compound_fixed_edge(
    device const PMRigidConstraintResource &constraint,
    device const PMRigidParameters *parameters, uint a, uint b) {
    return constraint.type == 0u && constraint.disable_collisions != 0u &&
           constraint.breaking_impulse_threshold <= 0.0f &&
           parameters[a].motion == 2u && parameters[b].motion == 2u;
}

static void pm_add_inertia_axis(
    thread PMSymmetricMatrix3 &matrix, float3 axis, float moment) {
    matrix.xx += moment * axis.x * axis.x;
    matrix.xy += moment * axis.x * axis.y;
    matrix.xz += moment * axis.x * axis.z;
    matrix.yy += moment * axis.y * axis.y;
    matrix.yz += moment * axis.y * axis.z;
    matrix.zz += moment * axis.z * axis.z;
}

static float3 pm_multiply_symmetric(
    thread const PMSymmetricMatrix3 &matrix, float3 value) {
    return {matrix.xx * value.x + matrix.xy * value.y +
                matrix.xz * value.z,
            matrix.xy * value.x + matrix.yy * value.y +
                matrix.yz * value.z,
            matrix.xz * value.x + matrix.yz * value.y +
                matrix.zz * value.z};
}

static bool pm_invert_symmetric(
    thread const PMSymmetricMatrix3 &matrix, thread float3 &row0,
    thread float3 &row1, thread float3 &row2) {
    const float c00 = matrix.yy * matrix.zz - matrix.yz * matrix.yz;
    const float c01 = matrix.xz * matrix.yz - matrix.xy * matrix.zz;
    const float c02 = matrix.xy * matrix.yz - matrix.xz * matrix.yy;
    const float c11 = matrix.xx * matrix.zz - matrix.xz * matrix.xz;
    const float c12 = matrix.xy * matrix.xz - matrix.xx * matrix.yz;
    const float c22 = matrix.xx * matrix.yy - matrix.xy * matrix.xy;
    const float determinant =
        matrix.xx * c00 + matrix.xy * c01 + matrix.xz * c02;
    if (!isfinite(determinant) || abs(determinant) <= 1.0e-6f)
        return false;
    const float scale = 1.0f / determinant;
    row0 = {c00 * scale, c01 * scale, c02 * scale};
    row1 = {c01 * scale, c11 * scale, c12 * scale};
    row2 = {c02 * scale, c12 * scale, c22 * scale};
    return true;
}

static float3 pm_compound_inverse_inertia(
    device const PMRigidCompound &compound, float3 value) {
    return {dot(pm_load(compound.inverse_inertia[0]), value),
            dot(pm_load(compound.inverse_inertia[1]), value),
            dot(pm_load(compound.inverse_inertia[2]), value)};
}

kernel void pm_build_rigid_compounds(
    device PMRigidBodyState *states [[buffer(0)]],
    device PMRigidParameters *parameters [[buffer(1)]],
    constant PMStepConstants &step [[buffer(4)]],
    device const PMRigidConstraintResource *constraints [[buffer(9)]],
    device PMRigidCompound *compounds [[buffer(25)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    for (uint body = 0u; body < step.body_count; ++body) {
        compounds[body] = {};
        compounds[body].root = body;
    }

    for (uint index = 0u; index < step.constraint_capacity; ++index) {
        device const PMRigidConstraintResource &constraint =
            constraints[index];
        if (constraint.alive == 0u || constraint.enabled == 0u ||
            constraint.broken != 0u ||
            constraint.body_a >= step.body_count ||
            constraint.body_b >= step.body_count ||
            !pm_compound_fixed_edge(
                constraint, parameters, constraint.body_a,
                constraint.body_b))
            continue;
        const uint root_a = pm_rigid_compound_root(
            compounds, constraint.body_a);
        const uint root_b = pm_rigid_compound_root(
            compounds, constraint.body_b);
        if (root_a != root_b)
            compounds[max(root_a, root_b)].root = min(root_a, root_b);
    }
    for (uint body = 0u; body < step.body_count; ++body)
        compounds[body].root = pm_rigid_compound_root(compounds, body);

    for (uint index = 0u; index < step.constraint_capacity; ++index) {
        device const PMRigidConstraintResource &constraint =
            constraints[index];
        if (constraint.alive == 0u || constraint.enabled == 0u ||
            constraint.broken != 0u ||
            constraint.body_a >= step.body_count ||
            constraint.body_b >= step.body_count)
            continue;
        const uint a = constraint.body_a;
        const uint b = constraint.body_b;
        if (pm_compound_fixed_edge(constraint, parameters, a, b)) continue;
        compounds[compounds[a].root].blocked = 1u;
        compounds[compounds[b].root].blocked = 1u;
    }
    for (uint body = 0u; body < step.body_count; ++body)
        ++compounds[compounds[body].root].member_count;

    for (uint root = 0u; root < step.body_count; ++root) {
        device PMRigidCompound &compound = compounds[root];
        if (compound.root != root || compound.member_count < 2u ||
            compound.blocked != 0u)
            continue;
        compound.eligible = 1u;
        for (uint pass = 1u; pass < compound.member_count; ++pass) {
            bool changed = false;
            for (uint index = 0u; index < step.constraint_capacity;
                 ++index) {
                device const PMRigidConstraintResource &constraint =
                    constraints[index];
                if (constraint.alive == 0u || constraint.enabled == 0u ||
                    constraint.broken != 0u ||
                    constraint.body_a >= step.body_count ||
                    constraint.body_b >= step.body_count)
                    continue;
                const uint a = constraint.body_a;
                const uint b = constraint.body_b;
                if (compounds[a].root != root ||
                    compounds[b].root != root ||
                    !pm_compound_fixed_edge(
                        constraint, parameters, a, b) ||
                    compounds[a].eligible == compounds[b].eligible)
                    continue;
                if (compounds[a].eligible != 0u) {
                    states[b].orientation = pm_quaternion_normalize(
                        pm_quaternion_multiply(
                            pm_quaternion_multiply(
                                states[a].orientation,
                                constraint.local_orientation_a),
                            pm_quaternion_conjugate(
                                constraint.local_orientation_b)));
                    states[b].position = pm_store(
                        pm_load(states[a].position) +
                        pm_rotate(states[a].orientation,
                                  pm_load(constraint.local_anchor_a)) -
                        pm_rotate(states[b].orientation,
                                  pm_load(constraint.local_anchor_b)));
                    compounds[b].eligible = 1u;
                } else {
                    states[a].orientation = pm_quaternion_normalize(
                        pm_quaternion_multiply(
                            pm_quaternion_multiply(
                                states[b].orientation,
                                constraint.local_orientation_b),
                            pm_quaternion_conjugate(
                                constraint.local_orientation_a)));
                    states[a].position = pm_store(
                        pm_load(states[b].position) +
                        pm_rotate(states[b].orientation,
                                  pm_load(constraint.local_anchor_b)) -
                        pm_rotate(states[a].orientation,
                                  pm_load(constraint.local_anchor_a)));
                    compounds[a].eligible = 1u;
                }
                changed = true;
            }
            if (!changed) break;
        }
        bool connected = true;
        for (uint member = 0u; member < step.body_count; ++member)
            if (compounds[member].root == root &&
                compounds[member].eligible == 0u)
                connected = false;
        if (!connected) {
            for (uint member = 0u; member < step.body_count; ++member)
                if (compounds[member].root == root)
                    compounds[member].eligible = 0u;
            continue;
        }

        float mass_sum = 0.0f;
        float3 weighted_center = 0.0f;
        float3 linear_momentum = 0.0f;
        for (uint member = 0u; member < step.body_count; ++member) {
            if (compounds[member].root != root) continue;
            const float inverse_mass = parameters[member].inverse_mass;
            if (inverse_mass <= 1.0e-6f) {
                compound.blocked = 1u;
                break;
            }
            const float mass = 1.0f / inverse_mass;
            mass_sum += mass;
            weighted_center += pm_load(states[member].position) * mass;
            linear_momentum +=
                pm_load(states[member].linear_velocity) * mass;
        }
        if (compound.blocked != 0u || mass_sum <= 1.0e-6f) {
            for (uint member = 0u; member < step.body_count; ++member)
                if (compounds[member].root == root)
                    compounds[member].eligible = 0u;
            continue;
        }
        compound.center = pm_store(weighted_center / mass_sum);

        PMSymmetricMatrix3 inertia{};
        float3 angular_momentum = 0.0f;
        for (uint member = 0u; member < step.body_count; ++member) {
            if (compounds[member].root != root) continue;
            device const PMRigidParameters &body = parameters[member];
            device const PMRigidBodyState &state = states[member];
            const float mass = 1.0f / body.inverse_mass;
            const float3 inverse_inertia = pm_load(body.inverse_inertia);
            const float3 local_moment = {
                1.0f / max(inverse_inertia.x, 1.0e-6f),
                1.0f / max(inverse_inertia.y, 1.0e-6f),
                1.0f / max(inverse_inertia.z, 1.0e-6f)};
            PMSymmetricMatrix3 member_inertia{};
            pm_add_inertia_axis(
                member_inertia,
                pm_rotate(state.orientation, float3(1.0f, 0.0f, 0.0f)),
                local_moment.x);
            pm_add_inertia_axis(
                member_inertia,
                pm_rotate(state.orientation, float3(0.0f, 1.0f, 0.0f)),
                local_moment.y);
            pm_add_inertia_axis(
                member_inertia,
                pm_rotate(state.orientation, float3(0.0f, 0.0f, 1.0f)),
                local_moment.z);
            inertia.xx += member_inertia.xx;
            inertia.xy += member_inertia.xy;
            inertia.xz += member_inertia.xz;
            inertia.yy += member_inertia.yy;
            inertia.yz += member_inertia.yz;
            inertia.zz += member_inertia.zz;
            const float3 arm =
                pm_load(state.position) - pm_load(compound.center);
            const float radius_squared = dot(arm, arm);
            inertia.xx += mass * (radius_squared - arm.x * arm.x);
            inertia.xy -= mass * arm.x * arm.y;
            inertia.xz -= mass * arm.x * arm.z;
            inertia.yy += mass * (radius_squared - arm.y * arm.y);
            inertia.yz -= mass * arm.y * arm.z;
            inertia.zz += mass * (radius_squared - arm.z * arm.z);
            angular_momentum +=
                pm_multiply_symmetric(
                    member_inertia, pm_load(state.angular_velocity)) +
                cross(arm, pm_load(state.linear_velocity) * mass);
        }
        float3 inverse0, inverse1, inverse2;
        if (!pm_invert_symmetric(
                inertia, inverse0, inverse1, inverse2)) {
            for (uint member = 0u; member < step.body_count; ++member)
                if (compounds[member].root == root)
                    compounds[member].eligible = 0u;
            continue;
        }
        compound.inverse_inertia[0] = pm_store(inverse0);
        compound.inverse_inertia[1] = pm_store(inverse1);
        compound.inverse_inertia[2] = pm_store(inverse2);
        compound.inverse_mass = 1.0f / mass_sum;
        compound.eligible = 1u;
        const float3 linear_velocity =
            linear_momentum * compound.inverse_mass;
        const float3 angular_velocity =
            pm_compound_inverse_inertia(compound, angular_momentum);
        for (uint member = 0u; member < step.body_count; ++member) {
            if (compounds[member].root != root) continue;
            compounds[member].eligible = 1u;
            states[member].angular_velocity = pm_store(angular_velocity);
            states[member].linear_velocity = pm_store(
                linear_velocity +
                cross(angular_velocity,
                      pm_load(states[member].position) -
                          pm_load(compound.center)));
        }
    }
}

static float pm_solve_linear_constraint_axis(
    device const PMRigidParameters &a, thread PMRigidBodyState &state_a,
    device const PMRigidParameters &b, thread PMRigidBodyState &state_b,
    float3 arm_a, float3 arm_b,
    device const PMRigidConstraintAxisGeometry &geometry, float error,
    float timestep, float stiffness, float damping, bool spring) {
    const float3 axis = pm_load(geometry.axis);
    const float3 velocity_a = pm_load(state_a.linear_velocity) +
        cross(pm_load(state_a.angular_velocity), arm_a);
    const float3 velocity_b = pm_load(state_b.linear_velocity) +
        cross(pm_load(state_b.angular_velocity), arm_b);
    const float relative_velocity = dot(velocity_b - velocity_a, axis);
    const float denominator = geometry.linear_denominator;
    if (denominator <= 1.0e-6f) return 0.0f;
    const float impulse = spring
        ? -(relative_velocity + stiffness * error * timestep) /
              (denominator + damping * timestep)
        : -(relative_velocity + 0.35f * error / timestep) / denominator;
    const float3 vector = axis * impulse;
    if (a.inverse_mass > 0.0f) {
        state_a.linear_velocity = pm_store(
            pm_load(state_a.linear_velocity) - vector * a.inverse_mass);
        state_a.angular_velocity = pm_store(
            pm_load(state_a.angular_velocity) -
            pm_inverse_inertia_mul_orientation(
                a, state_a.orientation, cross(arm_a, vector)));
    }
    if (b.inverse_mass > 0.0f) {
        state_b.linear_velocity = pm_store(
            pm_load(state_b.linear_velocity) + vector * b.inverse_mass);
        state_b.angular_velocity = pm_store(
            pm_load(state_b.angular_velocity) +
            pm_inverse_inertia_mul_orientation(
                b, state_b.orientation, cross(arm_b, vector)));
    }
    return abs(impulse);
}

static float pm_solve_angular_constraint_axis(
    device const PMRigidParameters &a, thread PMRigidBodyState &state_a,
    device const PMRigidParameters &b, thread PMRigidBodyState &state_b,
    device const PMRigidConstraintAxisGeometry &geometry, float error,
    float timestep, float stiffness, float damping, bool spring) {
    const float3 axis = pm_load(geometry.axis);
    const float relative_velocity = dot(
        pm_load(state_b.angular_velocity) -
            pm_load(state_a.angular_velocity),
        axis);
    const float3 inverse_a = pm_load(geometry.inverse_angular_a);
    const float3 inverse_b = pm_load(geometry.inverse_angular_b);
    const float denominator = geometry.angular_denominator;
    if (denominator <= 1.0e-6f) return 0.0f;
    const float impulse = spring
        ? -(relative_velocity + stiffness * error * timestep) /
              (denominator + damping * timestep)
        : -(relative_velocity + 0.30f * error / timestep) / denominator;
    if (a.inverse_mass > 0.0f)
        state_a.angular_velocity = pm_store(
            pm_load(state_a.angular_velocity) - inverse_a * impulse);
    if (b.inverse_mass > 0.0f)
        state_b.angular_velocity = pm_store(
            pm_load(state_b.angular_velocity) + inverse_b * impulse);
    return abs(impulse);
}

static float pm_solve_motor_axis(
    device const PMRigidParameters &a, thread PMRigidBodyState &state_a,
    device const PMRigidParameters &b, thread PMRigidBodyState &state_b,
    device const PMRigidConstraintAxisGeometry &geometry, float3 arm_a,
    float3 arm_b, float target_velocity, float maximum_impulse,
    bool angular) {
    const float3 axis = pm_load(geometry.axis);
    float denominator = 0.0f;
    float relative_velocity = 0.0f;
    if (angular) {
        relative_velocity = dot(
            pm_load(state_b.angular_velocity) -
                pm_load(state_a.angular_velocity),
            axis);
        denominator = geometry.angular_denominator;
    } else {
        const float3 velocity_a = pm_load(state_a.linear_velocity) +
            cross(pm_load(state_a.angular_velocity), arm_a);
        const float3 velocity_b = pm_load(state_b.linear_velocity) +
            cross(pm_load(state_b.angular_velocity), arm_b);
        relative_velocity = dot(velocity_b - velocity_a, axis);
        denominator = geometry.linear_denominator;
    }
    if (denominator <= 1.0e-6f || maximum_impulse <= 0.0f)
        return 0.0f;
    const float impulse = clamp(
        (target_velocity - relative_velocity) / denominator,
        -maximum_impulse, maximum_impulse);
    if (angular) {
        if (a.inverse_mass > 0.0f)
            state_a.angular_velocity = pm_store(
                pm_load(state_a.angular_velocity) -
                pm_load(geometry.inverse_angular_a) * impulse);
        if (b.inverse_mass > 0.0f)
            state_b.angular_velocity = pm_store(
                pm_load(state_b.angular_velocity) +
                pm_load(geometry.inverse_angular_b) * impulse);
    } else {
        const float3 vector = axis * impulse;
        if (a.inverse_mass > 0.0f) {
            state_a.linear_velocity = pm_store(
                pm_load(state_a.linear_velocity) -
                vector * a.inverse_mass);
            state_a.angular_velocity = pm_store(
                pm_load(state_a.angular_velocity) -
                pm_inverse_inertia_mul_orientation(
                    a, state_a.orientation, cross(arm_a, vector)));
        }
        if (b.inverse_mass > 0.0f) {
            state_b.linear_velocity = pm_store(
                pm_load(state_b.linear_velocity) +
                vector * b.inverse_mass);
            state_b.angular_velocity = pm_store(
                pm_load(state_b.angular_velocity) +
                pm_inverse_inertia_mul_orientation(
                    b, state_b.orientation, cross(arm_b, vector)));
        }
    }
    return abs(impulse);
}

static float3 pm_relative_rotation_vector(PMQuaternion frame_a,
                                          PMQuaternion frame_b) {
    PMQuaternion relative = pm_quaternion_normalize(
        pm_quaternion_multiply(pm_quaternion_conjugate(frame_a), frame_b));
    if (relative.w < 0.0f)
        relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    const float size = sqrt(
        relative.x * relative.x + relative.y * relative.y +
        relative.z * relative.z);
    if (size <= 1.0e-6f) return 0.0f;
    const float angle = 2.0f * atan2(
        size, clamp(relative.w, -1.0f, 1.0f));
    return float3(relative.x, relative.y, relative.z) * (angle / size);
}

static float3 pm_basis_axis(uint axis) {
    return axis == 0u ? float3(1.0f, 0.0f, 0.0f)
         : axis == 1u ? float3(0.0f, 1.0f, 0.0f)
                      : float3(0.0f, 0.0f, 1.0f);
}

static bool pm_axis_enabled(uint mask, uint axis) {
    return (mask & (1u << axis)) != 0u;
}

static float pm_limit_error(float value, float lower, float upper) {
    return value < lower ? value - lower
         : value > upper ? value - upper
                         : 0.0f;
}

static bool pm_body_is_fixed_member(
    uint body, device const PMRigidConstraintResource *constraints,
    uint constraint_capacity) {
    for (uint index = 0u; index < constraint_capacity; ++index) {
        device const PMRigidConstraintResource &constraint =
            constraints[index];
        if (constraint.alive != 0u && constraint.enabled != 0u &&
            constraint.broken == 0u && constraint.type == 0u &&
            (constraint.body_a == body || constraint.body_b == body))
            return true;
    }
    return false;
}

static float3 pm_normalized_or(float3 value, float3 fallback);

struct PMHingeContactFrame {
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

static bool pm_guided_static_contact(
    thread const PMHingeContactFrame &body,
    thread const PMHingeContactFrame &collider);

static float pm_fixed_hinge_inverse_moment(
    device const PMRigidParameters &body,
    device const PMRigidBodyState &state,
    thread const PMHingeContactFrame &hinge);

static PMHingeContactFrame pm_rigid_hinge_contact_frame(
    device const PMRigidParameters *parameters, uint body_count, uint body,
    device const PMRigidBodyState *states,
    device const PMRigidConstraintResource *constraints,
    uint constraint_capacity, thread float3 &reference) {
    PMHingeContactFrame result{};
    result.static_body = body < body_count && parameters[body].motion == 0u;
    if (body >= body_count || parameters[body].inverse_mass <= 0.0f) {
        reference = body < body_count
            ? pm_load(states[body].position) : float3(0.0f);
        return result;
    }
    uint incident_count = 0u;
    for (uint index = 0u; index < constraint_capacity; ++index) {
        device const PMRigidConstraintResource &constraint =
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
        const PMQuaternion local_orientation = is_a
            ? constraint.local_orientation_a
            : constraint.local_orientation_b;
        device const PMRigidBodyState &state = states[body];
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
            const PMQuaternion other_local_orientation = is_a
                ? constraint.local_orientation_b
                : constraint.local_orientation_a;
            device const PMRigidBodyState &other_state = states[other];
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

static uint pm_fixed_projection_root(
    device const PMRigidCompound *compounds, uint body) {
    while (compounds[body].projection_root != body)
        body = compounds[body].projection_root;
    return body;
}

static void pm_resolve_rigid_contact_pair(
    device PMRigidBodyState *states,
    device PMRigidParameters *parameters,
    constant PMStepConstants &step,
    device PMContactManifold &manifold,
    device PMRigidContactEvent *events, uint body, uint collider,
    bool correct_position, bool warm_start_only,
    device const PMRigidConstraintResource *constraints,
    device const PMRigidCompound *compounds);

kernel void pm_rigid_constraints_serial(
    device PMRigidBodyState *states [[buffer(0)]],
    device PMRigidParameters *parameters [[buffer(1)]],
    device PMPackedVec3 *forces [[buffer(2)]],
    device PMPackedVec3 *torques [[buffer(3)]],
    constant PMStepConstants &step [[buffer(4)]],
    device const PMPackedVec3 *vertices [[buffer(5)]],
    device const uint *indices [[buffer(6)]],
    device const PMTriangleMeshInfo *meshes [[buffer(7)]],
    device PMContactManifold *manifolds [[buffer(8)]],
    device PMRigidConstraintResource *constraints [[buffer(9)]],
    device PMRigidContactEvent *events [[buffer(11)]],
    device PMRigidConstraintGeometry *constraint_geometry [[buffer(22)]],
    device PMRigidCompound *compounds [[buffer(25)]],
    uint thread_index [[thread_position_in_grid]]) {
    (void)forces;
    (void)torques;
    (void)vertices;
    (void)indices;
    (void)meshes;
    if (thread_index != 0u) return;
    bool fixed_contacts = false;
    uint iterations = 0u;
    for (uint index = 0u; index < step.constraint_capacity; ++index) {
        device PMRigidConstraintResource &constraint = constraints[index];
        device PMRigidConstraintGeometry &geometry =
            constraint_geometry[index];
        geometry.valid = 0u;
        if (constraint.alive == 0u) continue;
        if (constraint.broken == 0u)
            constraint.applied_impulse = 0.0f;
        constraint.enabled =
            constraint.enabled != 0u && constraint.broken == 0u ? 1u : 0u;
        if (constraint.enabled == 0u ||
            constraint.body_a >= step.body_count ||
            constraint.body_b >= step.body_count)
            continue;
        const bool absorbed =
            constraint.type == 0u &&
            constraint.disable_collisions != 0u &&
            constraint.breaking_impulse_threshold <= 0.0f &&
            compounds[constraint.body_a].eligible != 0u &&
            compounds[constraint.body_b].eligible != 0u &&
            compounds[constraint.body_a].root ==
                compounds[constraint.body_b].root;
        if (absorbed) continue;
        fixed_contacts = fixed_contacts || constraint.type == 0u;
        device const PMRigidParameters &a =
            parameters[constraint.body_a];
        device const PMRigidParameters &b =
            parameters[constraint.body_b];
        device const PMRigidBodyState &state_a = states[constraint.body_a];
        device const PMRigidBodyState &state_b = states[constraint.body_b];
        const float3 arm_a = pm_rotate(
            state_a.orientation, pm_load(constraint.local_anchor_a));
        const float3 arm_b = pm_rotate(
            state_b.orientation, pm_load(constraint.local_anchor_b));
        geometry.valid = 1u;
        geometry.arm_a = pm_store(arm_a);
        geometry.arm_b = pm_store(arm_b);
        geometry.anchor_error = pm_store(
            (pm_load(state_b.position) + arm_b) -
            (pm_load(state_a.position) + arm_a));
        const PMQuaternion frame_a = pm_quaternion_normalize(
            pm_quaternion_multiply(
                state_a.orientation, constraint.local_orientation_a));
        const PMQuaternion frame_b = pm_quaternion_normalize(
            pm_quaternion_multiply(
                state_b.orientation, constraint.local_orientation_b));
        geometry.rotation_error = pm_store(
            pm_relative_rotation_vector(frame_a, frame_b));
        geometry.hinge_alignment_error = pm_store(cross(
            pm_rotate(frame_a, float3(0.0f, 0.0f, 1.0f)),
            pm_rotate(frame_b, float3(0.0f, 0.0f, 1.0f))));
        geometry.piston_alignment_error = pm_store(cross(
            pm_rotate(frame_a, float3(1.0f, 0.0f, 0.0f)),
            pm_rotate(frame_b, float3(1.0f, 0.0f, 0.0f))));
        for (uint axis_index = 0u; axis_index < 3u; ++axis_index) {
            device PMRigidConstraintAxisGeometry &row =
                geometry.axes[axis_index];
            const float3 axis = pm_rotate(
                frame_a, pm_basis_axis(axis_index));
            const float3 inverse_angular_a =
                pm_inverse_inertia_mul(a, state_a, axis);
            const float3 inverse_angular_b =
                pm_inverse_inertia_mul(b, state_b, axis);
            row.axis = pm_store(axis);
            row.inverse_angular_a = pm_store(inverse_angular_a);
            row.inverse_angular_b = pm_store(inverse_angular_b);
            row.angular_denominator = dot(
                inverse_angular_a + inverse_angular_b, axis);
            const float3 angular_a = cross(
                pm_inverse_inertia_mul(
                    a, state_a, cross(arm_a, axis)), arm_a);
            const float3 angular_b = cross(
                pm_inverse_inertia_mul(
                    b, state_b, cross(arm_b, axis)), arm_b);
            row.linear_denominator = a.inverse_mass + b.inverse_mass +
                dot(angular_a + angular_b, axis);
        }
        iterations = max(iterations, constraint.solver_iterations);
    }
    const uint contact_sweeps = fixed_contacts ? 8u : 1u;
    for (uint iteration = 0u;
         iteration < iterations * contact_sweeps; ++iteration) {
        // Match CUDA's fixed-joint solve cadence: support/contact impulses
        // and weld impulses converge together instead of pulling welded
        // members back through a contact after the contact pass has ended.
        for (uint pair = 0u;
             fixed_contacts && pair < step.body_count * step.body_count;
             ++pair) {
            device PMContactManifold &manifold = manifolds[pair];
            if (manifold.count == 0u) continue;
            const uint body = pair / step.body_count;
            const uint collider = pair % step.body_count;
            if (manifold.body_fixed_member == 0u &&
                manifold.collider_fixed_member == 0u)
                continue;
            if (compounds[body].eligible != 0u ||
                compounds[collider].eligible != 0u)
                continue;
            pm_resolve_rigid_contact_pair(
                states, parameters, step, manifold, events, body, collider,
                false, false, constraints, compounds);
        }
        for (uint index = 0u; index < step.constraint_capacity; ++index) {
            device PMRigidConstraintResource &constraint = constraints[index];
            device const PMRigidConstraintGeometry &geometry =
                constraint_geometry[index];
            if (constraint.alive == 0u || constraint.enabled == 0u ||
                iteration >= constraint.solver_iterations * contact_sweeps ||
                geometry.valid == 0u)
                continue;
            device const PMRigidParameters &a =
                parameters[constraint.body_a];
            device const PMRigidParameters &b =
                parameters[constraint.body_b];
            PMRigidBodyState state_a = states[constraint.body_a];
            PMRigidBodyState state_b = states[constraint.body_b];
            const float3 arm_a = pm_load(geometry.arm_a);
            const float3 arm_b = pm_load(geometry.arm_b);
            const float3 anchor_error = pm_load(geometry.anchor_error);
            const float3 rotation_error = pm_load(geometry.rotation_error);
            const float3 hinge_alignment_error =
                pm_load(geometry.hinge_alignment_error);
            const float3 piston_alignment_error =
                pm_load(geometry.piston_alignment_error);
            float applied = 0.0f;
            for (uint axis_index = 0u; axis_index < 3u; ++axis_index) {
                device const PMRigidConstraintAxisGeometry &row =
                    geometry.axes[axis_index];
                const float3 world_axis = pm_load(row.axis);
                const bool generic = constraint.type == 5u ||
                                     constraint.type == 6u;
                const bool motor = constraint.type == 7u;
                const bool linear_lock =
                    constraint.type == 0u || constraint.type == 1u ||
                    constraint.type == 2u ||
                    ((constraint.type == 3u || constraint.type == 4u) &&
                     axis_index != 0u) ||
                    (motor &&
                     (axis_index != 0u ||
                      constraint.linear_motor_enabled == 0u));
                const bool linear_spring = constraint.type == 6u &&
                    pm_axis_enabled(
                        constraint.linear_spring_axes, axis_index);
                const bool linear_limit = generic && pm_axis_enabled(
                    constraint.linear_limit_axes, axis_index);
                float linear_error = dot(anchor_error, world_axis);
                if (linear_limit && !linear_lock && !linear_spring)
                    linear_error = pm_limit_error(
                        linear_error,
                        pm_load(constraint.linear_limit_lower)[axis_index],
                        pm_load(constraint.linear_limit_upper)[axis_index]);
                if (linear_lock || linear_spring ||
                    (linear_limit && linear_error != 0.0f)) {
                    applied += pm_solve_linear_constraint_axis(
                        a, state_a, b, state_b, arm_a, arm_b, row,
                        linear_error, step.timestep,
                        pm_load(constraint.linear_spring_stiffness)[axis_index],
                        pm_load(constraint.linear_spring_damping)[axis_index],
                        linear_spring && !linear_lock);
                }

                const bool angular_lock =
                    constraint.type == 0u || constraint.type == 3u ||
                    (constraint.type == 2u && axis_index != 2u) ||
                    (constraint.type == 4u && axis_index != 0u) ||
                    (motor &&
                     (axis_index != 0u ||
                      constraint.angular_motor_enabled == 0u));
                const bool angular_spring = constraint.type == 6u &&
                    pm_axis_enabled(
                        constraint.angular_spring_axes, axis_index);
                const bool angular_limit = generic && pm_axis_enabled(
                    constraint.angular_limit_axes, axis_index);
                float angular_error =
                    constraint.type == 2u && axis_index != 2u
                        ? dot(hinge_alignment_error, world_axis)
                        : constraint.type == 4u && axis_index != 0u
                            ? dot(piston_alignment_error, world_axis)
                            : rotation_error[axis_index];
                if (angular_limit && !angular_lock && !angular_spring)
                    angular_error = pm_limit_error(
                        angular_error,
                        pm_load(constraint.angular_limit_lower)[axis_index],
                        pm_load(constraint.angular_limit_upper)[axis_index]);
                if (angular_lock || angular_spring ||
                    (angular_limit && angular_error != 0.0f)) {
                    applied += pm_solve_angular_constraint_axis(
                        a, state_a, b, state_b, row, angular_error,
                        step.timestep,
                        pm_load(constraint.angular_spring_stiffness)[axis_index],
                        pm_load(constraint.angular_spring_damping)[axis_index],
                        angular_spring && !angular_lock);
                }
            }
            if (constraint.type == 2u &&
                pm_axis_enabled(constraint.angular_limit_axes, 2u)) {
                const float error = pm_limit_error(
                    rotation_error.z, constraint.angular_limit_lower.z,
                    constraint.angular_limit_upper.z);
                if (error != 0.0f)
                    applied += pm_solve_angular_constraint_axis(
                        a, state_a, b, state_b, geometry.axes[2], error,
                        step.timestep, 0.0f, 0.0f, false);
            }
            if (constraint.type == 3u &&
                pm_axis_enabled(constraint.linear_limit_axes, 0u)) {
                const float3 slider_axis = pm_load(geometry.axes[0].axis);
                const float error = pm_limit_error(
                    dot(anchor_error, slider_axis),
                    constraint.linear_limit_lower.x,
                    constraint.linear_limit_upper.x);
                if (error != 0.0f)
                    applied += pm_solve_linear_constraint_axis(
                        a, state_a, b, state_b, arm_a, arm_b,
                        geometry.axes[0], error, step.timestep,
                        0.0f, 0.0f, false);
            }
            if (constraint.type == 4u) {
                const float3 piston_axis = pm_load(geometry.axes[0].axis);
                if (pm_axis_enabled(constraint.linear_limit_axes, 0u)) {
                    const float error = pm_limit_error(
                        dot(anchor_error, piston_axis),
                        constraint.linear_limit_lower.x,
                        constraint.linear_limit_upper.x);
                    if (error != 0.0f)
                        applied += pm_solve_linear_constraint_axis(
                            a, state_a, b, state_b, arm_a, arm_b,
                            geometry.axes[0], error, step.timestep,
                            0.0f, 0.0f, false);
                }
                if (pm_axis_enabled(constraint.angular_limit_axes, 0u)) {
                    const float error = pm_limit_error(
                        rotation_error.x,
                        constraint.angular_limit_lower.x,
                        constraint.angular_limit_upper.x);
                    if (error != 0.0f)
                        applied += pm_solve_angular_constraint_axis(
                            a, state_a, b, state_b, geometry.axes[0], error,
                            step.timestep, 0.0f, 0.0f, false);
                }
            }
            if (constraint.type == 7u) {
                const float inverse_iterations =
                    1.0f / float(
                        constraint.solver_iterations * contact_sweeps);
                if (constraint.linear_motor_enabled != 0u)
                    applied += pm_solve_motor_axis(
                        a, state_a, b, state_b, geometry.axes[0], arm_a, arm_b,
                        constraint.linear_target_velocity,
                        constraint.linear_maximum_impulse * inverse_iterations,
                        false);
                if (constraint.angular_motor_enabled != 0u)
                    applied += pm_solve_motor_axis(
                        a, state_a, b, state_b, geometry.axes[0], arm_a, arm_b,
                        constraint.angular_target_velocity,
                        constraint.angular_maximum_impulse * inverse_iterations,
                        true);
            }
            states[constraint.body_a] = state_a;
            states[constraint.body_b] = state_b;
            constraint.applied_impulse += applied;
            if (constraint.breaking_impulse_threshold > 0.0f &&
                constraint.applied_impulse >
                    constraint.breaking_impulse_threshold) {
                constraint.broken = 1u;
                constraint.enabled = 0u;
            }
        }
    }
    for (uint index = 0u; index < step.constraint_capacity; ++index) {
        device const PMRigidConstraintResource &constraint = constraints[index];
        if (constraint.alive == 0u || constraint.enabled == 0u ||
            constraint.type != 2u || constraint.body_a >= step.body_count ||
            constraint.body_b >= step.body_count)
            continue;
        device const PMRigidParameters &a = parameters[constraint.body_a];
        device const PMRigidParameters &b = parameters[constraint.body_b];
        const float inverse_mass_sum = a.inverse_mass + b.inverse_mass;
        if (inverse_mass_sum <= 1.0e-6f) continue;
        device PMRigidBodyState &state_a = states[constraint.body_a];
        device PMRigidBodyState &state_b = states[constraint.body_b];
        const float3 anchor_a = pm_load(state_a.position) + pm_rotate(
            state_a.orientation, pm_load(constraint.local_anchor_a));
        const float3 anchor_b = pm_load(state_b.position) + pm_rotate(
            state_b.orientation, pm_load(constraint.local_anchor_b));
        const float3 error = anchor_b - anchor_a;
        state_a.position = pm_store(
            pm_load(state_a.position) +
            error * (a.inverse_mass / inverse_mass_sum));
        state_b.position = pm_store(
            pm_load(state_b.position) -
            error * (b.inverse_mass / inverse_mass_sum));
    }

    // CUDA performs split positional recovery for free welded groups after
    // the velocity solve. Translate the whole component so contact recovery
    // cannot tear a fixed joint or leave its root embedded in a static mesh.
    bool welded = false;
    for (uint body = 0u; body < step.body_count; ++body) {
        compounds[body].projection_root = body;
        compounds[body].projection_movable = 1u;
        compounds[body].projection_translation = {};
    }
    for (uint index = 0u; index < step.constraint_capacity; ++index) {
        device const PMRigidConstraintResource &constraint =
            constraints[index];
        if (constraint.alive == 0u || constraint.enabled == 0u ||
            constraint.broken != 0u || constraint.type != 0u ||
            constraint.body_a >= step.body_count ||
            constraint.body_b >= step.body_count)
            continue;
        const uint root_a = pm_fixed_projection_root(
            compounds, constraint.body_a);
        const uint root_b = pm_fixed_projection_root(
            compounds, constraint.body_b);
        if (root_a != root_b)
            compounds[max(root_a, root_b)].projection_root =
                min(root_a, root_b);
        welded = true;
    }
    if (welded) {
        for (uint body = 0u; body < step.body_count; ++body) {
            compounds[body].projection_root =
                pm_fixed_projection_root(compounds, body);
            if (parameters[body].motion != 2u)
                compounds[compounds[body].projection_root]
                    .projection_movable = 0u;
        }
        for (uint index = 0u; index < step.constraint_capacity; ++index) {
            device const PMRigidConstraintResource &constraint =
                constraints[index];
            if (constraint.alive == 0u || constraint.enabled == 0u ||
                constraint.broken != 0u || constraint.type == 0u)
                continue;
            if (constraint.body_a < step.body_count)
                compounds[compounds[constraint.body_a].projection_root]
                    .projection_movable = 0u;
            if (constraint.body_b < step.body_count)
                compounds[compounds[constraint.body_b].projection_root]
                    .projection_movable = 0u;
        }
        for (uint pass = 0u; pass < 8u; ++pass) {
            for (uint pair = 0u;
                 pair < step.body_count * step.body_count; ++pair) {
                device const PMContactManifold &manifold = manifolds[pair];
                if (manifold.count == 0u) continue;
                const uint a = pair / step.body_count;
                const uint b = pair % step.body_count;
                const bool first = parameters[a].motion == 2u &&
                    parameters[b].motion != 2u &&
                    pm_body_is_fixed_member(
                        a, constraints, step.constraint_capacity);
                const bool second = parameters[b].motion == 2u &&
                    parameters[a].motion != 2u &&
                    pm_body_is_fixed_member(
                        b, constraints, step.constraint_capacity);
                if (!first && !second) continue;
                device PMRigidCompound &group = compounds[
                    compounds[first ? a : b].projection_root];
                if (group.projection_movable == 0u) continue;
                for (uint point = 0u; point < manifold.count; ++point) {
                    const float3 normal =
                        pm_load(manifold.contacts[point].normal) *
                        (first ? 1.0f : -1.0f);
                    const float depth =
                        manifold.contacts[point].penetration -
                        dot(normal,
                            pm_load(group.projection_translation));
                    if (depth > 0.0f)
                        group.projection_translation = pm_store(
                            pm_load(group.projection_translation) +
                            normal *
                                (depth + 1.0e-5f));
                }
            }
        }
        for (uint body = 0u; body < step.body_count; ++body) {
            device const PMRigidCompound &group = compounds[
                compounds[body].projection_root];
            states[body].position = pm_store(
                pm_load(states[body].position) +
                pm_load(group.projection_translation));
        }
    }
}

static float3 pm_world_point(device const PMRigidBodyState &state,
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
                                thread float &first_fraction,
                                thread float &second_fraction,
                                thread float3 &first_point,
                                thread float3 &second_point) {
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
    thread float &segment_fraction, thread float3 &segment_point,
    thread float3 &triangle_point, thread float3 &triangle_weights) {
    float best_squared = INFINITY;
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
            pm_closest_point_triangle(endpoints[endpoint], a, b, c);
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

struct PMContactCandidate {
    float3 point;
    float3 normal;
    float penetration;
    bool found;
};

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
    device const PMRigidBodyState &state, float3 local_minimum,
    float3 local_maximum, float margin, thread float3 &minimum,
    thread float3 &maximum) {
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
    device const PMRigidBodyState &previous_state,
    device const PMRigidBodyState &state, float3 local_minimum,
    float3 local_maximum, float margin, thread float3 &minimum,
    thread float3 &maximum) {
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
    device const PMRigidBodyState &previous_state,
    device const PMRigidBodyState &state,
    device const PMTriangleMeshInfo &mesh) {
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
    device const PMRigidBodyState &previous_body_state,
    device const PMRigidBodyState &body_state,
    device const PMTriangleMeshInfo &body_mesh,
    device const PMRigidBodyState &previous_collider_state,
    device const PMRigidBodyState &collider_state,
    device const PMTriangleMeshInfo &collider_mesh, float threshold) {
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

static bool pm_requires_swept_contact(
    device const PMRigidBodyState &previous_state,
    device const PMRigidBodyState &state,
    device const PMTriangleMeshInfo &mesh, float threshold) {
    return length(pm_load(state.position) -
                  pm_load(previous_state.position)) +
               pm_rotational_motion_bound(previous_state, state, mesh) >
           threshold;
}

static bool pm_bounding_spheres_may_contact(
    device const PMRigidBodyState &previous_body_state,
    device const PMRigidBodyState &body_state,
    device const PMTriangleMeshInfo &body_mesh,
    device const PMRigidBodyState &previous_collider_state,
    device const PMRigidBodyState &collider_state,
    device const PMTriangleMeshInfo &collider_mesh, float margin) {
    const float3 previous_relative =
        pm_load(previous_body_state.position) +
            pm_rotate(previous_body_state.orientation,
                      pm_load(body_mesh.bounding_center)) -
        pm_load(previous_collider_state.position) -
            pm_rotate(previous_collider_state.orientation,
                      pm_load(collider_mesh.bounding_center));
    const float3 current_relative =
        pm_load(body_state.position) +
            pm_rotate(body_state.orientation,
                      pm_load(body_mesh.bounding_center)) -
        pm_load(collider_state.position) -
            pm_rotate(collider_state.orientation,
                      pm_load(collider_mesh.bounding_center));
    const float3 movement = current_relative - previous_relative;
    const float squared_movement = dot(movement, movement);
    const float time = squared_movement > 1.0e-12f
        ? clamp(-dot(previous_relative, movement) / squared_movement,
                0.0f, 1.0f)
        : 0.0f;
    const float3 nearest = previous_relative + movement * time;
    const float radius = body_mesh.radius + collider_mesh.radius + margin +
        1.0e-5f +
        pm_rotational_motion_bound(
            previous_body_state, body_state, body_mesh) +
        pm_rotational_motion_bound(
            previous_collider_state, collider_state, collider_mesh);
    return dot(nearest, nearest) <= radius * radius;
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

static void pm_closest_triangle_pair(
    float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2,
    thread float3 &point_a, thread float3 &point_b,
    bool stable_face = false) {
    const float3 a[3] = {a0, a1, a2};
    const float3 b[3] = {b0, b1, b2};

    const auto point_in_triangle = [](float3 point, float3 first,
                                      float3 second, float3 third,
                                      float3 normal) {
        const float tolerance = -1.0e-5f * dot(normal, normal);
        return dot(cross(second - first, point - first), normal) >= tolerance &&
               dot(cross(third - second, point - second), normal) >= tolerance &&
               dot(cross(first - third, point - third), normal) >= tolerance;
    };
    const auto segment_hit = [&](float3 first, float3 second, float3 ta,
                                 float3 tb, float3 tc,
                                 thread float3 &intersection) {
        const float3 normal = cross(tb - ta, tc - ta);
        const float3 direction = second - first;
        const float denominator = dot(normal, direction);
        if (dot(normal, normal) <= 1.0e-12f ||
            abs(denominator) <= 1.0e-6f)
            return false;
        const float amount = dot(normal, ta - first) / denominator;
        if (amount < 0.0f || amount > 1.0f) return false;
        intersection = first + direction * amount;
        return point_in_triangle(intersection, ta, tb, tc, normal);
    };
    float3 intersection = 0.0f;
    float best_squared = INFINITY;
    for (uint edge = 0u; edge < 3u; ++edge) {
        if (segment_hit(a[edge], a[(edge + 1u) % 3u], b0, b1, b2,
                        intersection)) {
            point_a = intersection;
            point_b = intersection;
            return;
        }
    }
    for (uint edge = 0u; edge < 3u; ++edge) {
        if (segment_hit(b[edge], b[(edge + 1u) % 3u], a0, a1, a2,
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
            pm_closest_segments(
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
    thread float3 &point_a, thread float3 &point_b) {
    const float3 a[3] = {a0, a1, a2};
    const float3 b[3] = {b0, b1, b2};
    float best_squared = INFINITY;
    for (uint edge = 0u; edge < 3u; ++edge) {
        float fraction = 0.0f;
        float3 on_segment = 0.0f;
        float3 on_triangle = 0.0f;
        float3 weights = 0.0f;
        pm_closest_segment_triangle(
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
        pm_closest_segment_triangle(
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
    thread const float3 *a, thread const float3 *b, float3 separation) {
    const float3 normal = pm_normalized_or(separation, float3(0.0f));
    const float3 face_a = pm_normalized_or(
        cross(a[1] - a[0], a[2] - a[0]), float3(0.0f));
    const float3 face_b = pm_normalized_or(
        cross(b[1] - b[0], b[2] - b[0]), float3(0.0f));
    return abs(dot(normal, face_a)) > 0.9999f ||
           abs(dot(normal, face_b)) > 0.9999f;
}

static void pm_add_manifold_contact(
    thread PMContactManifold &manifold, PMContactRecord candidate,
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
    PMContactRecord contact,
    thread const PMHingeContactFrame &body_hinge,
    thread const PMHingeContactFrame &collider_hinge) {
    thread const PMHingeContactFrame &frame = body_hinge.axial
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
    thread PMContactManifold &manifold, PMContactRecord candidate,
    float separation,
    thread const PMHingeContactFrame &body_hinge,
    thread const PMHingeContactFrame &collider_hinge) {
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
        constexpr float simultaneous = 1.0e-4f;
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

constant float pm_rigid_surface_tolerance = 1.0e-5f;
constant float pm_rigid_maximum_rest_offset = 1.0e-3f;

static float pm_rigid_rest_offset(float margin) {
    return min(margin, pm_rigid_maximum_rest_offset);
}

static PMContactRecord pm_make_contact_record(
    float3 point, float3 normal, float penetration,
    float impact_fraction) {
    PMContactRecord contact{};
    contact.point = pm_store(point);
    contact.normal = pm_store(normal);
    contact.penetration = penetration;
    contact.found = 1u;
    contact.impact_fraction = impact_fraction;
    return contact;
}

static float pm_contact_normal_speed(
    device const PMRigidBodyState &body_state,
    device const PMRigidBodyState &collider_state, float3 point,
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
    device const PMRigidBodyState &body_state,
    device const PMRigidBodyState &collider_state, float3 point,
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
    device const PMRigidBodyState &body_state,
    device const PMTriangleMeshInfo &body_mesh,
    device const PMRigidBodyState &collider_state,
    device const PMTriangleMeshInfo &collider_mesh,
    device const PMPackedVec3 *vertices, device const uint *indices,
    device const PMCollisionPlane *solid_planes, float margin,
    thread PMContactManifold &output) {
    constexpr uint maximum_faces = 32u;
    if (body_mesh.solid_plane_count == 0u ||
        collider_mesh.solid_plane_count == 0u ||
        body_mesh.index_count > maximum_faces * 3u ||
        collider_mesh.index_count > maximum_faces * 3u)
        return false;

    float best_separation = -INFINITY;
    uint reference_face = 0u;
    bool reference_is_body = false;
    for (uint side = 0u; side < 2u; ++side) {
        device const PMTriangleMeshInfo *reference_mesh =
            side == 0u ? &collider_mesh : &body_mesh;
        device const PMRigidBodyState *reference_state =
            side == 0u ? &collider_state : &body_state;
        device const PMTriangleMeshInfo *incident_mesh =
            side == 0u ? &body_mesh : &collider_mesh;
        device const PMRigidBodyState *incident_state =
            side == 0u ? &body_state : &collider_state;
        for (uint face = 0u; face < reference_mesh->index_count / 3u;
             ++face) {
            const PMCollisionPlane plane = solid_planes[
                reference_mesh->solid_plane_offset + face];
            const float3 normal = pm_rotate(
                reference_state->orientation, pm_load(plane.normal));
            const float3 incident_normal = pm_rotate(
                pm_quaternion_conjugate(incident_state->orientation),
                normal);
            float support = INFINITY;
            for (uint vertex_index = 0u;
                 vertex_index < incident_mesh->vertex_count;
                 ++vertex_index)
                support = min(
                    support,
                    dot(incident_normal,
                        pm_load(vertices[incident_mesh->vertex_offset +
                                         vertex_index])));
            const float separation = support +
                dot(normal,
                    pm_load(incident_state->position) -
                        pm_load(reference_state->position)) -
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

    device const PMTriangleMeshInfo *reference_mesh =
        reference_is_body ? &body_mesh : &collider_mesh;
    device const PMRigidBodyState *reference_state =
        reference_is_body ? &body_state : &collider_state;
    device const PMTriangleMeshInfo *incident_mesh =
        reference_is_body ? &collider_mesh : &body_mesh;
    device const PMRigidBodyState *incident_state =
        reference_is_body ? &collider_state : &body_state;
    const PMCollisionPlane reference = solid_planes[
        reference_mesh->solid_plane_offset + reference_face];
    const float3 reference_normal = pm_load(reference.normal);
    const float3 outward = pm_rotate(
        reference_state->orientation, reference_normal);
    const float3 incident_axis = pm_rotate(
        pm_quaternion_conjugate(incident_state->orientation), outward);
    float alignment = 1.0f;
    uint incident_face = 0u;
    for (uint face = 0u; face < incident_mesh->index_count / 3u; ++face) {
        const PMCollisionPlane plane = solid_planes[
            incident_mesh->solid_plane_offset + face];
        const float value = dot(incident_axis, pm_load(plane.normal));
        if (value < alignment) {
            alignment = value;
            incident_face = face;
        }
    }
    if (alignment > -0.98f) return false;

    PMContactManifold manifold{};
    const float3 normal = reference_is_body ? -outward : outward;
    const PMCollisionPlane incident = solid_planes[
        incident_mesh->solid_plane_offset + incident_face];
    const float3 incident_normal = pm_load(incident.normal);
    for (uint triangle = 0u; triangle < incident_mesh->index_count / 3u;
         ++triangle) {
        const PMCollisionPlane face = solid_planes[
            incident_mesh->solid_plane_offset + triangle];
        if (dot(pm_load(face.normal), incident_normal) < 0.99999f ||
            abs(face.offset - incident.offset) > pm_rigid_surface_tolerance)
            continue;
        float3 polygon[maximum_faces + 4u];
        float3 clipped[maximum_faces + 4u];
        uint count = 3u;
        for (uint corner = 0u; corner < 3u; ++corner) {
            const uint local_index = indices[
                incident_mesh->index_offset + triangle * 3u + corner];
            const float3 world = pm_world_point(
                *incident_state,
                pm_load(vertices[incident_mesh->vertex_offset +
                                 local_index]));
            polygon[corner] = pm_rotate(
                pm_quaternion_conjugate(reference_state->orientation),
                world - pm_load(reference_state->position));
        }
        // Clip the incident triangle against the reference solid's side
        // faces. Only the supporting face is expanded for speculative
        // contacts.
        for (uint plane_index = 0u;
             plane_index < reference_mesh->index_count / 3u && count != 0u;
             ++plane_index) {
            const PMCollisionPlane plane = solid_planes[
                reference_mesh->solid_plane_offset + plane_index];
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
                *reference_state, local_point);
            if (distance <= margin) {
                const PMContactRecord contact = pm_make_contact_record(
                    point, normal, -distance, 1.0f);
                pm_add_manifold_contact(
                    manifold, contact, max(margin * 2.0f, 1.0e-4f));
            }
        }
    }
    // Parallel supporting faces with no overlap are separated, including
    // adjacent corners that only coincide within floating-point roundoff.
    // Falling back to intersecting triangles invents penetration there.
    if (manifold.count == 0u && alignment > -0.999999f) return false;
    manifold.face_patch = 1u;
    output = manifold;
    return true;
}

static void pm_reduce_collinear_face_contacts(
    thread PMContactManifold &manifold) {
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
    device const PMRigidBodyState &state,
    device const PMTriangleMeshInfo &mesh,
    device const PMPackedVec3 *vertices, device const uint *indices,
    uint triangle, thread float3 &a, thread float3 &b, thread float3 &c) {
    const uint base = mesh.index_offset + triangle * 3u;
    a = pm_world_point(
        state, pm_load(vertices[mesh.vertex_offset + indices[base]]));
    b = pm_world_point(
        state, pm_load(vertices[mesh.vertex_offset + indices[base + 1u]]));
    c = pm_world_point(
        state, pm_load(vertices[mesh.vertex_offset + indices[base + 2u]]));
}

static void pm_collide_triangle_ranges(
    device const PMRigidBodyState &previous_body_state,
    device const PMRigidBodyState &body_state,
    float3 body_reference,
    device const PMTriangleMeshInfo &body_mesh, uint body_first,
    uint body_count,
    device const PMRigidBodyState &previous_collider_state,
    device const PMRigidBodyState &collider_state,
    device const PMTriangleMeshInfo &collider_mesh, uint collider_first,
    uint collider_count, device const PMPackedVec3 *vertices,
    device const uint *indices, float margin, float timestep,
    bool robust_closest, bool fixed_cluster_contact,
    thread const PMHingeContactFrame &body_hinge,
    thread const PMHingeContactFrame &collider_hinge,
    thread PMContactManifold &manifold) {
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
                device const PMRigidBodyState &surface_state =
                    convex_a ? collider_state : body_state;
                device const PMRigidBodyState &previous_surface =
                    convex_a ? previous_collider_state : previous_body_state;
                device const PMRigidBodyState &previous_convex =
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
            PMContactRecord contact = pm_make_contact_record(
                point, normal, penetration, 1.0f);
            pm_add_pair_manifold_contact(
                manifold, contact, max(margin * 2.0f, 1.0e-4f),
                body_hinge, collider_hinge);
        }
    }
}

static void pm_collide_triangle_ranges_swept(
    device const PMRigidBodyState &previous_body_state,
    device const PMRigidBodyState &body_state,
    float3 previous_body_reference, float3 body_reference,
    device const PMTriangleMeshInfo &body_mesh, uint body_first,
    uint body_count,
    device const PMRigidBodyState &previous_collider_state,
    device const PMRigidBodyState &collider_state,
    device const PMTriangleMeshInfo &collider_mesh, uint collider_first,
    uint collider_count, device const PMPackedVec3 *vertices,
    device const uint *indices, float margin, float timestep,
    bool robust_closest,
    thread const PMHingeContactFrame &body_hinge,
    thread const PMHingeContactFrame &collider_hinge,
    thread PMContactManifold &manifold) {
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
                    thread const PMHingeContactFrame &guide = body_hinge.axial
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
                    const float3 normal =
                        guided && have_separating_normal &&
                                distance <= pm_rigid_surface_tolerance
                            ? separating_normal
                            : guided
                                ? pm_guided_triangle_normal(
                                      a[0], a[1], a[2], b[0], b[1], b[2],
                                      point_a, point_b,
                                      have_separating_normal
                                          ? separating_normal : fallback)
                                : distance <= pm_rigid_surface_tolerance
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
                    PMContactRecord contact = pm_make_contact_record(
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
                    float minimum_a = INFINITY;
                    float maximum_b = -INFINITY;
                    float minimum_speed_a = INFINITY;
                    float maximum_speed_b = -INFINITY;
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

static PMContactManifold pm_collide_meshes(
    device const PMRigidBodyState &previous_body_state,
    device const PMRigidBodyState &body_state,
    float3 previous_body_reference, float3 body_reference,
    device const PMTriangleMeshInfo &body_mesh,
    device const PMRigidBodyState &previous_collider_state,
    device const PMRigidBodyState &collider_state,
    device const PMTriangleMeshInfo &collider_mesh,
    device const PMPackedVec3 *vertices, device const uint *indices,
    device const PMBvhNode *bvh_nodes,
    device const PMMeshLeafInfo &body_leaf_info,
    device const PMMeshLeafInfo &collider_leaf_info,
    device const uint *bvh_leaves, float margin, float timestep,
    bool robust_closest, bool fixed_cluster_contact, bool swept_only,
    thread const PMHingeContactFrame &body_hinge,
    thread const PMHingeContactFrame &collider_hinge) {
    PMContactManifold manifold{};
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
                device const PMBvhNode &body_node =
                    bvh_nodes[bvh_leaves[
                        body_leaf_info.offset + body_leaf]];
                device const PMBvhNode &collider_node =
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
                PMContactManifold local{};
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
    bool overflow = body_mesh.bvh_node_count == 0u ||
                    collider_mesh.bvh_node_count == 0u;
    if (!overflow)
        stack[stack_size++] =
            uint2(body_mesh.bvh_node_offset, collider_mesh.bvh_node_offset);
    while (stack_size > 0u && !overflow) {
        const uint2 pair = stack[--stack_size];
        device const PMBvhNode &body_node = bvh_nodes[pair.x];
        device const PMBvhNode &collider_node = bvh_nodes[pair.y];
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

struct PMAppliedContactImpulse {
    float normal;
    float3 friction;
};

static float pm_fixed_hinge_inverse_moment(
    device const PMRigidParameters &body,
    device const PMRigidBodyState &state,
    thread const PMHingeContactFrame &hinge) {
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

static float3 pm_contact_point_velocity(
    device const PMRigidBodyState &state,
    thread const PMHingeContactFrame &hinge, float3 point) {
    if (!hinge.fixed && !hinge.axial)
        return pm_load(state.linear_velocity) +
            cross(pm_load(state.angular_velocity),
                  point - pm_load(state.position));
    const float3 angular = hinge.axis *
        (hinge.fixed || hinge.axial_rotation
             ? dot(pm_load(state.angular_velocity), hinge.axis)
             : 0.0f);
    return (hinge.axial
                ? hinge.axis * dot(pm_load(state.linear_velocity), hinge.axis)
                : float3(0.0f)) +
        cross(angular, point - hinge.anchor);
}

static float pm_contact_direction_inverse_mass(
    device const PMRigidParameters &body,
    device const PMRigidBodyState &state,
    thread const PMHingeContactFrame &hinge,
    float3 point, float3 direction,
    device const PMRigidCompound *compounds, uint index) {
    if (compounds[index].eligible != 0u) {
        device const PMRigidCompound &compound =
            compounds[compounds[index].root];
        const float3 arm = point - pm_load(compound.center);
        const float3 angular = cross(arm, direction);
        return compound.inverse_mass +
            dot(cross(pm_compound_inverse_inertia(compound, angular), arm),
                direction);
    }
    if (hinge.fixed || hinge.axial) {
        const float jacobian = dot(
            cross(hinge.axis, point - hinge.anchor), direction);
        const float axial = hinge.axial
            ? dot(hinge.axis, direction) : 0.0f;
        return body.inverse_mass * axial * axial +
            jacobian * jacobian *
            pm_fixed_hinge_inverse_moment(body, state, hinge);
    }
    const float3 arm = point - pm_load(state.position);
    const float3 angular = cross(arm, direction);
    return body.inverse_mass +
        dot(cross(pm_inverse_inertia_mul(body, state, angular), arm),
            direction);
}

static void pm_apply_contact_velocity_impulse(
    device PMRigidBodyState *states,
    device const PMRigidParameters *parameters, uint count, uint index,
    thread const PMHingeContactFrame &hinge,
    float3 point, float3 impulse,
    device const PMRigidCompound *compounds) {
    if (compounds[index].eligible != 0u) {
        const uint root = compounds[index].root;
        device const PMRigidCompound &compound = compounds[root];
        const float3 linear_delta = impulse * compound.inverse_mass;
        const float3 angular_delta = pm_compound_inverse_inertia(
            compound, cross(point - pm_load(compound.center), impulse));
        for (uint member = 0u; member < count; ++member) {
            if (compounds[member].eligible == 0u ||
                compounds[member].root != root)
                continue;
            states[member].linear_velocity = pm_store(
                pm_load(states[member].linear_velocity) + linear_delta +
                cross(angular_delta,
                      pm_load(states[member].position) -
                          pm_load(compound.center)));
            states[member].angular_velocity = pm_store(
                pm_load(states[member].angular_velocity) + angular_delta);
        }
        return;
    }
    device const PMRigidParameters &body = parameters[index];
    if (body.inverse_mass <= 0.0f) return;
    device PMRigidBodyState &state = states[index];
    if (hinge.fixed || hinge.axial) {
        const float angular_impulse = dot(
            hinge.axis, cross(point - hinge.anchor, impulse));
        const float3 angular_delta = hinge.axis *
            (angular_impulse *
             pm_fixed_hinge_inverse_moment(body, state, hinge));
        state.angular_velocity = pm_store(
            pm_load(state.angular_velocity) + angular_delta);
        state.linear_velocity = pm_store(
            pm_load(state.linear_velocity) +
            (hinge.axial
                 ? hinge.axis *
                       (body.inverse_mass * dot(impulse, hinge.axis))
                 : float3(0.0f)) +
            cross(angular_delta,
                  pm_load(state.position) - hinge.anchor));
        return;
    }
    state.linear_velocity = pm_store(
        pm_load(state.linear_velocity) + impulse * body.inverse_mass);
    state.angular_velocity = pm_store(
        pm_load(state.angular_velocity) +
        pm_inverse_inertia_mul(
            body, state,
            cross(point - pm_load(state.position), impulse)));
}

static PMAppliedContactImpulse pm_apply_contact_impulse(
    device PMRigidBodyState *states,
    device PMRigidParameters *parameters, uint count, uint body_index,
    uint collider_index,
    thread const PMContactCandidate &contact, float timestep,
    thread const PMHingeContactFrame &body_hinge,
    thread const PMHingeContactFrame &collider_hinge,
    device const PMRigidCompound *compounds) {
    device PMRigidBodyState &body_state = states[body_index];
    device PMRigidParameters &body = parameters[body_index];
    device PMRigidBodyState &collider_state = states[collider_index];
    device PMRigidParameters &collider = parameters[collider_index];
    PMAppliedContactImpulse applied{};
    const float3 body_velocity = pm_contact_point_velocity(
        body_state, body_hinge, contact.point);
    const float3 collider_velocity = pm_contact_point_velocity(
        collider_state, collider_hinge, contact.point);
    float3 relative_velocity = body_velocity - collider_velocity;
    const float normal_speed = dot(relative_velocity, contact.normal);
    const float separation = max(0.0f, -contact.penetration);
    float target_speed = separation > pm_rigid_surface_tolerance
        ? -separation / max(timestep, 1.0e-6f)
        : 0.0f;
    const bool fixed_cluster_contact =
        body_hinge.fixed_member || collider_hinge.fixed_member;
    if (fixed_cluster_contact && contact.penetration > 0.0f)
        target_speed = max(
            target_speed,
            0.2f * contact.penetration / max(timestep, 1.0e-6f));
    const float restitution = min(body.restitution, collider.restitution);
    if (separation <= pm_rigid_surface_tolerance && normal_speed < 0.0f)
        target_speed = max(target_speed, -restitution * normal_speed);
    if (normal_speed >= target_speed) return applied;

    const float denominator = pm_contact_direction_inverse_mass(
        body, body_state, body_hinge, contact.point, contact.normal, compounds,
        body_index) + pm_contact_direction_inverse_mass(
        collider, collider_state, collider_hinge, contact.point,
        contact.normal, compounds, collider_index);
    if (denominator <= 1.0e-6f) return applied;

    applied.normal = (target_speed - normal_speed) / denominator;
    const float3 normal_vector = contact.normal * applied.normal;
    pm_apply_contact_velocity_impulse(
        states, parameters, count, body_index, body_hinge, contact.point,
        normal_vector, compounds);
    pm_apply_contact_velocity_impulse(
        states, parameters, count, collider_index, collider_hinge,
        contact.point,
        -normal_vector, compounds);

    if (separation > pm_rigid_surface_tolerance) return applied;
    relative_velocity =
        pm_contact_point_velocity(body_state, body_hinge, contact.point) -
        pm_contact_point_velocity(
            collider_state, collider_hinge, contact.point);
    float3 tangent = relative_velocity -
                     contact.normal *
                         dot(relative_velocity, contact.normal);
    const float tangent_length = length(tangent);
    if (tangent_length <= 1.0e-6f) return applied;
    tangent /= tangent_length;
    const float tangent_denominator = pm_contact_direction_inverse_mass(
        body, body_state, body_hinge, contact.point, tangent, compounds,
        body_index) +
        pm_contact_direction_inverse_mass(
            collider, collider_state, collider_hinge, contact.point, tangent,
            compounds, collider_index);
    if (tangent_denominator <= 1.0e-6f) return applied;
    float tangent_impulse =
        -dot(relative_velocity, tangent) / tangent_denominator;
    const float friction_limit =
        sqrt(body.friction * collider.friction) * applied.normal;
    tangent_impulse =
        clamp(tangent_impulse, -friction_limit, friction_limit);
    applied.friction = tangent * tangent_impulse;
    pm_apply_contact_velocity_impulse(
        states, parameters, count, body_index, body_hinge, contact.point,
        applied.friction, compounds);
    pm_apply_contact_velocity_impulse(
        states, parameters, count, collider_index, collider_hinge,
        contact.point,
        -applied.friction, compounds);
    return applied;
}

static float3 pm_clamp_vector_length(float3 value, float maximum) {
    const float squared = dot(value, value);
    if (squared <= maximum * maximum || squared <= 1.0e-12f)
        return value;
    return value * (maximum / sqrt(squared));
}

static PMAppliedContactImpulse pm_apply_persistent_contact_impulse(
    device PMRigidBodyState *states,
    device PMRigidParameters *parameters, uint count, uint body_index,
    uint collider_index,
    thread const PMContactCandidate &contact, float initial_normal_speed,
    thread float &accumulated_normal_impulse,
    thread float3 &accumulated_friction_impulse,
    float timestep,
    thread const PMHingeContactFrame &body_hinge,
    thread const PMHingeContactFrame &collider_hinge,
    device const PMRigidCompound *compounds) {
    device PMRigidBodyState &body_state = states[body_index];
    device const PMRigidParameters &body = parameters[body_index];
    device PMRigidBodyState &collider_state = states[collider_index];
    device const PMRigidParameters &collider = parameters[collider_index];
    PMAppliedContactImpulse applied{};
    float3 relative_velocity = pm_contact_point_velocity(
        body_state, body_hinge, contact.point) -
        pm_contact_point_velocity(
            collider_state, collider_hinge, contact.point);
    const float normal_speed = dot(relative_velocity, contact.normal);
    const float separation = max(0.0f, -contact.penetration);
    float target_speed = separation > pm_rigid_surface_tolerance
        ? -separation / max(timestep, 1.0e-6f)
        : 0.0f;
    const bool fixed_cluster_contact =
        body_hinge.fixed_member || collider_hinge.fixed_member;
    if (fixed_cluster_contact && contact.penetration > 0.0f)
        target_speed = max(
            target_speed,
            0.2f * contact.penetration / max(timestep, 1.0e-6f));
    if (separation <= pm_rigid_surface_tolerance &&
        initial_normal_speed < 0.0f)
        target_speed = max(
            target_speed,
            -min(body.restitution, collider.restitution) *
                initial_normal_speed);

    const float denominator = pm_contact_direction_inverse_mass(
        body, body_state, body_hinge, contact.point, contact.normal,
        compounds, body_index) +
        pm_contact_direction_inverse_mass(
            collider, collider_state, collider_hinge, contact.point,
            contact.normal, compounds, collider_index);
    if (denominator <= 1.0e-6f) return applied;

    const float accumulated_normal = max(
        0.0f, accumulated_normal_impulse +
                  (target_speed - normal_speed) / denominator);
    applied.normal = accumulated_normal - accumulated_normal_impulse;
    accumulated_normal_impulse = accumulated_normal;
    const float3 normal_vector = contact.normal * applied.normal;
    pm_apply_contact_velocity_impulse(
        states, parameters, count, body_index, body_hinge, contact.point,
        normal_vector, compounds);
    pm_apply_contact_velocity_impulse(
        states, parameters, count, collider_index, collider_hinge,
        contact.point, -normal_vector, compounds);

    const bool friction_active = separation <=
        (pm_guided_static_contact(body_hinge, collider_hinge)
             ? pm_rigid_surface_tolerance
             : pm_rigid_rest_offset(
                   body.collision_margin + collider.collision_margin));
    relative_velocity = pm_contact_point_velocity(
        body_state, body_hinge, contact.point) -
        pm_contact_point_velocity(
            collider_state, collider_hinge, contact.point);
    float3 tangent = relative_velocity -
        contact.normal * dot(relative_velocity, contact.normal);
    const float tangent_length = length(tangent);
    float3 friction = friction_active
        ? accumulated_friction_impulse : float3(0.0f);
    if (friction_active && tangent_length > 1.0e-6f) {
        tangent /= tangent_length;
        const float tangent_denominator =
            pm_contact_direction_inverse_mass(
                body, body_state, body_hinge, contact.point, tangent,
                compounds, body_index) +
            pm_contact_direction_inverse_mass(
                collider, collider_state, collider_hinge, contact.point,
                tangent, compounds, collider_index);
        if (tangent_denominator > 1.0e-6f)
            friction -= tangent *
                (tangent_length / tangent_denominator);
    }
    friction = pm_clamp_vector_length(
        friction,
        sqrt(body.friction * collider.friction) * accumulated_normal);
    applied.friction = friction - accumulated_friction_impulse;
    accumulated_friction_impulse = friction;
    pm_apply_contact_velocity_impulse(
        states, parameters, count, body_index, body_hinge, contact.point,
        applied.friction, compounds);
    pm_apply_contact_velocity_impulse(
        states, parameters, count, collider_index, collider_hinge,
        contact.point, -applied.friction, compounds);
    return applied;
}

static float pm_contact_position_inverse_mass(
    device const PMRigidParameters &body,
    device const PMRigidBodyState &state,
    thread const PMHingeContactFrame &hinge,
    float3 point, float3 normal) {
    if (!hinge.fixed && !hinge.axial) return body.inverse_mass;
    const float jacobian = dot(
        cross(hinge.axis, point - hinge.anchor), normal);
    const float axial = hinge.axial ? dot(hinge.axis, normal) : 0.0f;
    return body.inverse_mass * axial * axial + jacobian * jacobian *
        pm_fixed_hinge_inverse_moment(body, state, hinge);
}

static void pm_apply_contact_position_delta(
    device const PMRigidParameters &body,
    device PMRigidBodyState &state,
    thread const PMHingeContactFrame &hinge,
    float3 point, float3 correction) {
    if (body.inverse_mass <= 0.0f) return;
    if (!hinge.fixed && !hinge.axial) {
        state.position = pm_store(
            pm_load(state.position) + correction * body.inverse_mass);
        return;
    }
    const float angular_correction = dot(
        hinge.axis, cross(point - hinge.anchor, correction)) *
        pm_fixed_hinge_inverse_moment(body, state, hinge);
    const float3 current_anchor = pm_load(state.position) +
        pm_rotate(state.orientation, hinge.local_anchor);
    const float3 anchor = hinge.axial
        ? current_anchor + hinge.axis *
              (body.inverse_mass * dot(correction, hinge.axis))
        : hinge.anchor;
    pm_apply_orientation_delta(state, hinge.axis * angular_correction);
    state.position = pm_store(
        anchor - pm_rotate(state.orientation, hinge.local_anchor));
}

static void pm_apply_contact_position_correction(
    device const PMRigidParameters &body,
    device PMRigidBodyState &body_state,
    thread const PMHingeContactFrame &body_hinge,
    device const PMRigidParameters &collider,
    device PMRigidBodyState &collider_state,
    thread const PMHingeContactFrame &collider_hinge,
    thread const PMContactCandidate &contact, float penetration) {
    const float denominator = pm_contact_position_inverse_mass(
        body, body_state, body_hinge, contact.point, contact.normal) +
        pm_contact_position_inverse_mass(
            collider, collider_state, collider_hinge,
            contact.point, contact.normal);
    if (denominator <= 1.0e-6f || penetration <= 0.0f) return;
    const float3 correction = contact.normal * (penetration / denominator);
    pm_apply_contact_position_delta(
        body, body_state, body_hinge, contact.point, correction);
    pm_apply_contact_position_delta(
        collider, collider_state, collider_hinge,
        contact.point, -correction);
}

kernel void pm_rigid_world_bounds(
    device const PMRigidBodyState *states [[buffer(0)]],
    device const PMRigidParameters *parameters [[buffer(1)]],
    constant PMStepConstants &step [[buffer(4)]],
    device const PMTriangleMeshInfo *meshes [[buffer(7)]],
    device const PMRigidBodyState *previous_states [[buffer(14)]],
    device PMWorldAabb *world_bounds [[buffer(16)]],
    uint body [[thread_position_in_grid]]) {
    if (body >= step.body_count) return;
    device const PMTriangleMeshInfo &mesh =
        meshes[parameters[body].mesh_index];
    float3 minimum = 0.0f;
    float3 maximum = 0.0f;
    if (pm_requires_swept_contact(
            previous_states[body], states[body], mesh,
            parameters[body].collision_margin)) {
        pm_transformed_motion_bounds(
            previous_states[body], states[body], pm_load(mesh.minimum),
            pm_load(mesh.maximum), 0.0f, minimum, maximum);
    } else {
        pm_transformed_bounds(
            states[body], pm_load(mesh.minimum), pm_load(mesh.maximum),
            0.0f, minimum, maximum);
    }
    world_bounds[body] = {pm_store(minimum), pm_store(maximum)};
}

kernel void pm_rigid_pair_filter(
    device const PMRigidBodyState *states [[buffer(0)]],
    device const PMRigidParameters *parameters [[buffer(1)]],
    constant PMStepConstants &step [[buffer(4)]],
    device const PMTriangleMeshInfo *meshes [[buffer(7)]],
    device PMContactManifold *manifolds [[buffer(8)]],
    device const PMRigidConstraintResource *constraints [[buffer(9)]],
    device const PMRigidBodyState *previous_states [[buffer(14)]],
    device const PMWorldAabb *world_bounds [[buffer(16)]],
    device uint *active_flags [[buffer(17)]],
    device const PMRigidCompound *compounds [[buffer(25)]],
    uint pair [[thread_position_in_grid]]) {
    const uint pair_count = step.body_count * step.body_count;
    if (pair >= pair_count) return;
    manifolds[pair] = {};
    const uint body = pair / step.body_count;
    const uint collider = pair % step.body_count;
    bool active = parameters[body].inverse_mass > 0.0f && body != collider;
    if (active && parameters[collider].inverse_mass > 0.0f &&
        collider < body)
        active = false;
    if (active && compounds[body].eligible != 0u &&
        compounds[collider].eligible != 0u &&
        compounds[body].root == compounds[collider].root)
        active = false;
    if (active) {
        for (uint constraint_index = 0u;
             constraint_index < step.constraint_capacity;
             ++constraint_index) {
            device const PMRigidConstraintResource &constraint =
                constraints[constraint_index];
            if (constraint.alive != 0u && constraint.enabled != 0u &&
                constraint.broken == 0u &&
                constraint.disable_collisions != 0u &&
                ((constraint.body_a == body &&
                  constraint.body_b == collider) ||
                 (constraint.body_a == collider &&
                  constraint.body_b == body))) {
                active = false;
                break;
            }
        }
    }
    const float margin = parameters[body].collision_margin +
                         parameters[collider].collision_margin;
    if (active) {
        const float3 body_minimum =
            pm_load(world_bounds[body].minimum) - margin;
        const float3 body_maximum =
            pm_load(world_bounds[body].maximum) + margin;
        active = pm_bounds_overlap(
            body_minimum, body_maximum,
            pm_load(world_bounds[collider].minimum),
            pm_load(world_bounds[collider].maximum));
    }
    if (active) {
        active = pm_bounding_spheres_may_contact(
            previous_states[body], states[body],
            meshes[parameters[body].mesh_index],
            previous_states[collider], states[collider],
            meshes[parameters[collider].mesh_index], margin);
    }
    active_flags[pair] = active ? 1u : 0u;
}

kernel void pm_rigid_pair_count_rows(
    constant PMStepConstants &step [[buffer(4)]],
    device const uint *active_flags [[buffer(17)]],
    device uint *row_offsets [[buffer(21)]],
    uint body [[thread_position_in_grid]]) {
    if (body >= step.body_count) return;
    uint count = 0u;
    const uint row_begin = body * step.body_count;
    for (uint collider = 0u; collider < step.body_count; ++collider) {
        count += active_flags[row_begin + collider] != 0u ? 1u : 0u;
    }
    row_offsets[body + 1u] = count;
    if (body == 0u) row_offsets[0] = 0u;
}

kernel void pm_rigid_pair_prefix_rows(
    constant PMStepConstants &step [[buffer(4)]],
    device uint &active_pair_count [[buffer(19)]],
    device uint *row_offsets [[buffer(21)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    uint cursor = 0u;
    for (uint body = 0u; body < step.body_count; ++body) {
        const uint count = row_offsets[body + 1u];
        row_offsets[body] = cursor;
        cursor += count;
    }
    row_offsets[step.body_count] = cursor;
    active_pair_count = cursor;
}

kernel void pm_rigid_pair_scatter_rows(
    constant PMStepConstants &step [[buffer(4)]],
    device const uint *active_flags [[buffer(17)]],
    device uint *active_pairs [[buffer(18)]],
    device const uint *row_offsets [[buffer(21)]],
    uint body [[thread_position_in_grid]]) {
    if (body >= step.body_count) return;
    uint cursor = row_offsets[body];
    const uint row_begin = body * step.body_count;
    for (uint collider = 0u; collider < step.body_count; ++collider) {
        const uint pair = row_begin + collider;
        if (active_flags[pair] != 0u) active_pairs[cursor++] = pair;
    }
}

static bool pm_guided_static_contact(
    thread const PMHingeContactFrame &body,
    thread const PMHingeContactFrame &collider) {
    return (body.axial && collider.static_body) ||
           (collider.axial && body.static_body);
}

static void pm_initialize_contact_solve(
    device PMContactManifold &manifold,
    device const PMRigidBodyState &body,
    device const PMRigidBodyState &collider,
    thread const PMHingeContactFrame &body_hinge,
    thread const PMHingeContactFrame &collider_hinge,
    bool persistent_pair) {
    manifold.initial_relative_position = pm_store(
        pm_load(body.position) - pm_load(collider.position));
    manifold.body_fixed_member = body_hinge.fixed_member ? 1u : 0u;
    manifold.collider_fixed_member =
        collider_hinge.fixed_member ? 1u : 0u;
    const bool guided = pm_guided_static_contact(
        body_hinge, collider_hinge);
    for (uint point = 0u; point < manifold.count; ++point) {
        device PMContactRecord &contact = manifold.contacts[point];
        contact.accumulated_normal_impulse = 0.0f;
        contact.accumulated_friction_impulse = {};
        contact.initial_normal_speed = pm_contact_normal_speed(
            body, collider, pm_load(contact.point), pm_load(contact.normal));
        contact.persistent = guided ||
            (persistent_pair && !body_hinge.present &&
             !collider_hinge.present && !body_hinge.fixed_member &&
             !collider_hinge.fixed_member);
        contact.warm_started = 0u;
    }
}

static bool pm_motor_constraint_member(
    uint body, device const PMRigidConstraintResource *constraints,
    uint constraint_capacity) {
    for (uint index = 0u; index < constraint_capacity; ++index) {
        device const PMRigidConstraintResource &constraint =
            constraints[index];
        if (constraint.alive != 0u && constraint.enabled != 0u &&
            constraint.broken == 0u && constraint.type == 7u &&
            (constraint.body_a == body || constraint.body_b == body))
            return true;
    }
    return false;
}

static void pm_stabilize_motor_collider_normals(
    device PMContactManifold &manifold,
    device const PMRigidBodyState &collider_state,
    device const PMTriangleMeshInfo &collider_mesh,
    device const PMCollisionPlane *solid_planes) {
    if (collider_mesh.solid_plane_count == 0u) return;
    for (uint point = 0u; point < manifold.count; ++point) {
        device PMContactRecord &contact = manifold.contacts[point];
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
    }
}

kernel void pm_rigid_contact_generate(
    device PMRigidBodyState *states [[buffer(0)]],
    device PMRigidParameters *parameters [[buffer(1)]],
    device PMPackedVec3 *forces [[buffer(2)]],
    device PMPackedVec3 *torques [[buffer(3)]],
    constant PMStepConstants &step [[buffer(4)]],
    device const PMPackedVec3 *vertices [[buffer(5)]],
    device const uint *indices [[buffer(6)]],
    device const PMTriangleMeshInfo *meshes [[buffer(7)]],
    device PMContactManifold *manifolds [[buffer(8)]],
    device const PMRigidConstraintResource *constraints [[buffer(9)]],
    device const PMBvhNode *bvh_nodes [[buffer(13)]],
    device const PMRigidBodyState *previous_states [[buffer(14)]],
    device const uint *active_pairs [[buffer(18)]],
    device const uint &active_pair_count [[buffer(19)]],
    device const PMMeshLeafInfo *mesh_leaf_infos [[buffer(23)]],
    device const uint *bvh_leaves [[buffer(24)]],
    device const PMCollisionPlane *solid_planes [[buffer(26)]],
    uint active_index [[thread_position_in_grid]]) {
    (void)forces;
    (void)torques;
    if (active_index >= active_pair_count) return;
    const uint pair = active_pairs[active_index];
    device PMContactManifold &output = manifolds[pair];
    const uint body = pair / step.body_count;
    const uint collider = pair % step.body_count;
    float3 previous_body_reference = 0.0f;
    float3 body_reference = 0.0f;
    float3 collider_reference = 0.0f;
    (void)pm_rigid_hinge_contact_frame(
        parameters, step.body_count, body, previous_states, constraints,
        step.constraint_capacity, previous_body_reference);
    const PMHingeContactFrame body_hinge = pm_rigid_hinge_contact_frame(
        parameters, step.body_count, body, states, constraints,
        step.constraint_capacity, body_reference);
    const PMHingeContactFrame collider_hinge = pm_rigid_hinge_contact_frame(
        parameters, step.body_count, collider, states, constraints,
        step.constraint_capacity, collider_reference);
    PMQuaternion body_axial_orientation{};
    PMQuaternion collider_axial_orientation{};
    const PMGuidedFrame body_guide = pm_rigid_guided_frame(
        parameters, step.body_count, body, states, constraints,
        step.constraint_capacity, body_axial_orientation);
    const PMGuidedFrame collider_guide = pm_rigid_guided_frame(
        parameters, step.body_count, collider, states, constraints,
        step.constraint_capacity, collider_axial_orientation);
    const bool guided_static_pair =
        (body_guide.active && parameters[collider].motion == 0u) ||
        (collider_guide.active && parameters[body].motion == 0u);
    const bool fixed_cluster_contact =
        body_hinge.fixed_member || collider_hinge.fixed_member;
    device const PMTriangleMeshInfo &body_mesh =
        meshes[parameters[body].mesh_index];
    device const PMTriangleMeshInfo &collider_mesh =
        meshes[parameters[collider].mesh_index];
    device const PMMeshLeafInfo &body_leaf_info =
        mesh_leaf_infos[parameters[body].mesh_index];
    device const PMMeshLeafInfo &collider_leaf_info =
        mesh_leaf_infos[parameters[collider].mesh_index];
    const float margin = parameters[body].collision_margin +
                         parameters[collider].collision_margin;
    const bool persistent_pair =
        body_mesh.solid_plane_count != 0u && body_mesh.index_count <= 96u &&
        (parameters[collider].motion != 2u ||
         (collider_mesh.solid_plane_count != 0u &&
          collider_mesh.index_count <= 96u));
    const bool face_pair = !guided_static_pair &&
        body_mesh.solid_plane_count != 0u &&
        collider_mesh.solid_plane_count != 0u &&
        body_mesh.index_count <= 96u && collider_mesh.index_count <= 96u &&
        !pm_requires_swept_pair_contact(
            previous_states[body], states[body], body_mesh,
            previous_states[collider], states[collider], collider_mesh,
            margin);
    PMContactManifold face_manifold{};
    if (face_pair && pm_convex_face_manifold(
            states[body], body_mesh, states[collider], collider_mesh,
            vertices, indices, solid_planes, margin, face_manifold)) {
        pm_reduce_collinear_face_contacts(face_manifold);
        output = face_manifold;
        pm_initialize_contact_solve(
            output, states[body], states[collider], body_hinge,
            collider_hinge, persistent_pair);
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
        device const PMContactRecord &contact =
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
        PMContactManifold fallback = pm_collide_meshes(
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
                const PMContactRecord contact =
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
}

static uint pm_contact_color_priority(uint pair) {
    return pair * 2654435761u + 1013904223u;
}

static float3 pm_local_contact_point_velocity(
    thread const PMRigidBodyState &state, float3 point) {
    return pm_load(state.linear_velocity) +
        cross(pm_load(state.angular_velocity),
              point - pm_load(state.position));
}

static float3 pm_local_inverse_inertia_mul(
    device const PMRigidParameters &body,
    thread const PMRigidBodyState &state, float3 value) {
    const PMQuaternion inverse =
        pm_quaternion_conjugate(state.orientation);
    const float3 local = pm_rotate(inverse, value);
    const float3 transformed =
        local * pm_load(body.inverse_inertia);
    return pm_rotate(state.orientation, transformed);
}

static float pm_local_contact_inverse_mass(
    device const PMRigidParameters &body,
    thread const PMRigidBodyState &state, float3 point,
    float3 direction) {
    const float3 arm = point - pm_load(state.position);
    const float3 angular = cross(arm, direction);
    return body.inverse_mass +
        dot(cross(pm_local_inverse_inertia_mul(body, state, angular), arm),
            direction);
}

static void pm_apply_local_contact_impulse(
    device const PMRigidParameters &body,
    thread PMRigidBodyState &state, float3 point, float3 impulse) {
    if (body.inverse_mass <= 0.0f) return;
    state.linear_velocity = pm_store(
        pm_load(state.linear_velocity) + impulse * body.inverse_mass);
    state.angular_velocity = pm_store(
        pm_load(state.angular_velocity) +
        pm_local_inverse_inertia_mul(
            body, state, cross(point - pm_load(state.position), impulse)));
}

// CUDA deliberately resolves an ordinary persistent patch with both body
// states in local storage.  Besides avoiding global-memory traffic, that
// prevents contact-record pointers from aliasing body-state pointers and
// changing the compiler's dependent reloads between rows.  Keep the same
// two-body path for large worlds here. Small analytic/articulated scenes,
// guided contacts, fixed compounds, and kinematic colliders retain the
// established device-backed path until their tighter trajectories are
// independently cross-validated.
static void pm_resolve_local_persistent_pair(
    thread PMRigidBodyState &body_state,
    device const PMRigidParameters &body,
    thread PMRigidBodyState &collider_state,
    device const PMRigidParameters &collider,
    device PMContactManifold &manifold,
    device PMRigidContactEvent *events,
    constant PMStepConstants &step,
    bool warm_start_only) {
    for (uint contact_index = 0u;
         contact_index < manifold.count; ++contact_index) {
        device PMContactRecord &record = manifold.contacts[contact_index];
        if (record.persistent == 0u || record.warm_started != 0u) continue;
        record.warm_started = 1u;
        const float3 impulse = pm_load(record.normal) *
                record.accumulated_normal_impulse +
            pm_load(record.accumulated_friction_impulse);
        const float3 point = pm_load(record.point);
        pm_apply_local_contact_impulse(body, body_state, point, impulse);
        pm_apply_local_contact_impulse(
            collider, collider_state, point, -impulse);
        const uint event_index = manifold.event_offset + contact_index;
        if (step.collect_rigid_contacts != 0u &&
            event_index < step.rigid_event_capacity) {
            events[event_index].normal_impulse +=
                record.accumulated_normal_impulse;
            events[event_index].friction_impulse = pm_store(
                pm_load(events[event_index].friction_impulse) +
                pm_load(record.accumulated_friction_impulse));
        }
    }
    if (warm_start_only) return;

    const float inverse_mass_sum =
        body.inverse_mass + collider.inverse_mass;
    if (inverse_mass_sum > 1.0e-6f) {
        const float contact_weight = 1.0f / float(manifold.count);
        for (uint contact_index = 0u;
             contact_index < manifold.count; ++contact_index) {
            device const PMContactRecord &record =
                manifold.contacts[contact_index];
            const float3 normal = pm_load(record.normal);
            const float penetration = record.penetration - dot(
                (pm_load(body_state.position) -
                 pm_load(collider_state.position)) -
                    pm_load(manifold.initial_relative_position),
                normal);
            if (penetration <= 0.0f) continue;
            const float3 correction = normal *
                ((penetration * contact_weight) / inverse_mass_sum);
            if (body.inverse_mass > 0.0f)
                body_state.position = pm_store(
                    pm_load(body_state.position) +
                    correction * body.inverse_mass);
            if (collider.inverse_mass > 0.0f)
                collider_state.position = pm_store(
                    pm_load(collider_state.position) -
                    correction * collider.inverse_mass);
        }
    }

    for (uint contact_index = 0u;
         contact_index < manifold.count; ++contact_index) {
        device PMContactRecord &record = manifold.contacts[contact_index];
        const float3 point = pm_load(record.point);
        const float3 normal = pm_load(record.normal);
        float3 relative_velocity =
            pm_local_contact_point_velocity(body_state, point) -
            pm_local_contact_point_velocity(collider_state, point);
        const float normal_speed = dot(relative_velocity, normal);
        const float separation = max(0.0f, -record.penetration);
        float target_speed = separation > pm_rigid_surface_tolerance
            ? -separation / max(step.timestep, 1.0e-6f)
            : 0.0f;
        if (separation <= pm_rigid_surface_tolerance &&
            record.initial_normal_speed < 0.0f)
            target_speed = max(
                target_speed,
                -min(body.restitution, collider.restitution) *
                    record.initial_normal_speed);

        const float denominator =
            pm_local_contact_inverse_mass(
                body, body_state, point, normal) +
            pm_local_contact_inverse_mass(
                collider, collider_state, point, normal);
        if (denominator <= 1.0e-6f) continue;

        const float accumulated_normal = max(
            0.0f, record.accumulated_normal_impulse +
                      (target_speed - normal_speed) / denominator);
        const float normal_impulse =
            accumulated_normal - record.accumulated_normal_impulse;
        record.accumulated_normal_impulse = accumulated_normal;
        const float3 normal_vector = normal * normal_impulse;
        pm_apply_local_contact_impulse(
            body, body_state, point, normal_vector);
        pm_apply_local_contact_impulse(
            collider, collider_state, point, -normal_vector);

        relative_velocity =
            pm_local_contact_point_velocity(body_state, point) -
            pm_local_contact_point_velocity(collider_state, point);
        float3 tangent = relative_velocity -
            normal * dot(relative_velocity, normal);
        const float tangent_length = length(tangent);
        float3 friction = separation <= pm_rigid_rest_offset(
                body.collision_margin + collider.collision_margin)
            ? pm_load(record.accumulated_friction_impulse)
            : float3(0.0f);
        if (separation <= pm_rigid_rest_offset(
                body.collision_margin + collider.collision_margin) &&
            tangent_length > 1.0e-6f) {
            tangent /= tangent_length;
            const float tangent_denominator =
                pm_local_contact_inverse_mass(
                    body, body_state, point, tangent) +
                pm_local_contact_inverse_mass(
                    collider, collider_state, point, tangent);
            if (tangent_denominator > 1.0e-6f)
                friction -= tangent *
                    (tangent_length / tangent_denominator);
        }
        friction = pm_clamp_vector_length(
            friction,
            sqrt(body.friction * collider.friction) * accumulated_normal);
        const float3 friction_impulse =
            friction - pm_load(record.accumulated_friction_impulse);
        record.accumulated_friction_impulse = pm_store(friction);
        pm_apply_local_contact_impulse(
            body, body_state, point, friction_impulse);
        pm_apply_local_contact_impulse(
            collider, collider_state, point, -friction_impulse);

        const uint event_index = manifold.event_offset + contact_index;
        if (step.collect_rigid_contacts != 0u &&
            event_index < step.rigid_event_capacity) {
            events[event_index].normal_impulse += normal_impulse;
            events[event_index].friction_impulse = pm_store(
                pm_load(events[event_index].friction_impulse) +
                friction_impulse);
        }
    }
}

static void pm_resolve_rigid_contact_pair(
    device PMRigidBodyState *states,
    device PMRigidParameters *parameters,
    constant PMStepConstants &step,
    device PMContactManifold &manifold,
    device PMRigidContactEvent *events, uint body, uint collider,
    bool correct_position, bool warm_start_only,
    device const PMRigidConstraintResource *constraints,
    device const PMRigidCompound *compounds) {
    if (manifold.count == 0u) return;
    float3 body_reference = 0.0f;
    float3 collider_reference = 0.0f;
    PMHingeContactFrame body_hinge = pm_rigid_hinge_contact_frame(
        parameters, step.body_count, body, states, constraints,
        step.constraint_capacity, body_reference);
    PMHingeContactFrame collider_hinge = pm_rigid_hinge_contact_frame(
        parameters, step.body_count, collider, states, constraints,
        step.constraint_capacity, collider_reference);
    body_hinge.fixed_member = manifold.body_fixed_member != 0u;
    collider_hinge.fixed_member =
        manifold.collider_fixed_member != 0u;
    const bool fixed_cluster_contact =
        body_hinge.fixed_member || collider_hinge.fixed_member;
    for (uint contact_index = 0u;
         contact_index < manifold.count; ++contact_index) {
        device PMContactRecord &record = manifold.contacts[contact_index];
        if (record.persistent == 0u || record.warm_started != 0u) continue;
        record.warm_started = 1u;
        const float3 impulse = pm_load(record.normal) *
                record.accumulated_normal_impulse +
            pm_load(record.accumulated_friction_impulse);
        pm_apply_contact_velocity_impulse(
            states, parameters, step.body_count, body, body_hinge,
            pm_load(record.point), impulse, compounds);
        pm_apply_contact_velocity_impulse(
            states, parameters, step.body_count, collider, collider_hinge,
            pm_load(record.point), -impulse, compounds);
        const uint event_index = manifold.event_offset + contact_index;
        if (step.collect_rigid_contacts != 0u &&
            event_index < step.rigid_event_capacity) {
            events[event_index].normal_impulse +=
                record.accumulated_normal_impulse;
            events[event_index].friction_impulse = pm_store(
                pm_load(events[event_index].friction_impulse) +
                pm_load(record.accumulated_friction_impulse));
        }
    }
    if (warm_start_only) return;

    const float inverse_mass_sum =
        parameters[body].inverse_mass + parameters[collider].inverse_mass;
    const bool guided = pm_guided_static_contact(
        body_hinge, collider_hinge);
    const bool local_persistent_pair =
        step.body_count >= 32u &&
        manifold.contacts[0].persistent != 0u && !guided &&
        !body_hinge.present && !collider_hinge.present &&
        !fixed_cluster_contact && compounds[body].eligible == 0u &&
        compounds[collider].eligible == 0u &&
        parameters[collider].motion != 1u;
    if (local_persistent_pair) {
        PMRigidBodyState local_body = states[body];
        PMRigidBodyState local_collider = states[collider];
        pm_resolve_local_persistent_pair(
            local_body, parameters[body], local_collider,
            parameters[collider], manifold, events, step,
            warm_start_only);
        if (parameters[body].inverse_mass > 0.0f)
            states[body] = local_body;
        if (parameters[collider].inverse_mass > 0.0f)
            states[collider] = local_collider;
        return;
    }
    const bool translational_projection =
        manifold.contacts[0].persistent != 0u && !guided;
    if (!guided && (correct_position || translational_projection) &&
        inverse_mass_sum > 1.0e-6f) {
        uint correction_count = translational_projection
            ? manifold.count : 0u;
        if (!translational_projection) {
            for (uint contact_index = 0u;
                 contact_index < manifold.count; ++contact_index) {
                device const PMContactRecord &record =
                    manifold.contacts[contact_index];
                if (record.penetration > 0.0f && !fixed_cluster_contact)
                    ++correction_count;
            }
        }
        const float contact_weight = correction_count == 0u
            ? 0.0f
            : 1.0f / float(correction_count);
        for (uint contact_index = 0u;
             contact_index < manifold.count; ++contact_index) {
            device const PMContactRecord &record =
                manifold.contacts[contact_index];
            if (fixed_cluster_contact) continue;
            const float penetration = record.penetration -
                (record.persistent == 0u
                     ? 0.0f
                     : dot(
                           (pm_load(states[body].position) -
                            pm_load(states[collider].position)) -
                               pm_load(manifold.initial_relative_position),
                           pm_load(record.normal)));
            if (penetration <= 0.0f) continue;
            const PMContactCandidate contact{
                pm_load(record.point), pm_load(record.normal),
                record.penetration, true};
            const float limited_penetration = min(
                penetration,
                body_hinge.fixed || collider_hinge.fixed
                    ? pm_rigid_maximum_rest_offset
                    : penetration);
            pm_apply_contact_position_correction(
                parameters[body], states[body], body_hinge,
                parameters[collider], states[collider], collider_hinge,
                contact,
                (limited_penetration +
                 (record.persistent != 0u
                      ? 0.0f : pm_rigid_surface_tolerance)) *
                    contact_weight);
        }
    }
    for (uint contact_index = 0u;
         contact_index < manifold.count; ++contact_index) {
        device PMContactRecord &record =
            manifold.contacts[contact_index];
        const PMContactCandidate contact{
            pm_load(record.point), pm_load(record.normal),
            record.penetration, true};
        PMAppliedContactImpulse applied{};
        if (record.persistent != 0u) {
            float accumulated_normal = record.accumulated_normal_impulse;
            float3 accumulated_friction =
                pm_load(record.accumulated_friction_impulse);
            applied = pm_apply_persistent_contact_impulse(
                states, parameters, step.body_count, body, collider, contact,
                record.initial_normal_speed, accumulated_normal,
                accumulated_friction, step.timestep, body_hinge,
                collider_hinge, compounds);
            record.accumulated_normal_impulse = accumulated_normal;
            record.accumulated_friction_impulse =
                pm_store(accumulated_friction);
        } else {
            applied = pm_apply_contact_impulse(
                states, parameters, step.body_count, body, collider, contact,
                step.timestep, body_hinge, collider_hinge, compounds);
        }
        const uint event_index = manifold.event_offset + contact_index;
        if (step.collect_rigid_contacts != 0u &&
            event_index < step.rigid_event_capacity) {
            events[event_index].normal_impulse += applied.normal;
            events[event_index].friction_impulse = pm_store(
                pm_load(events[event_index].friction_impulse) +
                applied.friction);
        }
    }
}

kernel void pm_rigid_contact_reduce(
    device PMRigidBodyState *states [[buffer(0)]],
    device PMRigidParameters *parameters [[buffer(1)]],
    device PMPackedVec3 *forces [[buffer(2)]],
    device PMPackedVec3 *torques [[buffer(3)]],
    constant PMStepConstants &step [[buffer(4)]],
    device const PMPackedVec3 *vertices [[buffer(5)]],
    device const uint *indices [[buffer(6)]],
    device const PMTriangleMeshInfo *meshes [[buffer(7)]],
    device PMContactManifold *manifolds [[buffer(8)]],
    device const PMRigidConstraintResource *constraints [[buffer(9)]],
    device const PMHandle *ids [[buffer(10)]],
    device PMRigidContactEvent *events [[buffer(11)]],
    device uint &event_count [[buffer(12)]],
    device atomic_uint *color_owners [[buffer(15)]],
    device const uint *active_pairs [[buffer(18)]],
    device const uint &active_pair_count [[buffer(19)]],
    device const uint &substep_index [[buffer(20)]],
    device const PMRigidCompound *compounds [[buffer(25)]],
    device const PMCachedContactPair *cache [[buffer(27)]],
    device const ulong &epoch_base [[buffer(28)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    (void)forces;
    (void)torques;
    (void)vertices;
    (void)indices;
    (void)meshes;
    constexpr uint color_round_count = 24u;
    constexpr uint uncolored = 0xffffffffu;
    const uint lane_count = threads_per_group.x;
    threadgroup atomic_uint used_colors;
    threadgroup atomic_uint overflow_count;
    threadgroup uint iterations;

    const ulong epoch = epoch_base + ulong(substep_index);
    for (uint active_index = thread_index;
         active_index < active_pair_count; active_index += lane_count) {
        const uint pair = active_pairs[active_index];
        const uint body = pair / step.body_count;
        const uint collider = pair % step.body_count;
        pm_load_rigid_contact_cache(
            manifolds[pair], states[body], ids[body], ids[collider],
            cache[pair], epoch, step.timestep);
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint active_index = thread_index;
         active_index < active_pair_count; active_index += lane_count)
        manifolds[active_pairs[active_index]].color = uncolored;
    threadgroup_barrier(mem_flags::mem_device);

    if (thread_index == 0u) {
        uint cursor = 0u;
        for (uint active_index = 0u;
             active_index < active_pair_count; ++active_index) {
            const uint pair = active_pairs[active_index];
            device PMContactManifold &manifold = manifolds[pair];
            if (manifold.count == 0u) continue;
            manifold.event_offset = cursor;
            const uint body = pair / step.body_count;
            const uint collider = pair % step.body_count;
            if (step.collect_rigid_contacts != 0u) {
                const uint remaining = cursor < step.rigid_event_capacity
                    ? step.rigid_event_capacity - cursor
                    : 0u;
                const uint retained = min(manifold.count, remaining);
                for (uint contact_index = 0u;
                     contact_index < retained; ++contact_index) {
                    device const PMContactRecord &record =
                        manifold.contacts[contact_index];
                    events[cursor + contact_index] = {
                        ids[body], ids[collider], record.point,
                        record.normal, max(0.0f, record.penetration),
                        0.0f, {}};
                }
            }
            cursor += manifold.count;
        }
        // Match CUDA: a contact-free later substep keeps the previous
        // substep's events, while a non-empty substep replaces them.
        if (step.collect_rigid_contacts != 0u && cursor > 0u)
            event_count = min(cursor, step.rigid_event_capacity);
        atomic_store_explicit(
            &used_colors, 0u, memory_order_relaxed);
        atomic_store_explicit(
            &overflow_count, 0u, memory_order_relaxed);
        iterations = 8u;
        for (uint active_index = 0u;
             active_index < active_pair_count; ++active_index) {
            const uint pair = active_pairs[active_index];
            if (manifolds[pair].face_patch == 0u) continue;
            iterations = 32u;
            if (step.ordinary_rigid_stack != 0u &&
                manifolds[pair].contacts[0].persistent != 0u &&
                manifolds[pair].cached == 0u)
                iterations = 64u;
        }
    }
    threadgroup_barrier(
        mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint color = 0u; color < color_round_count; ++color) {
        for (uint body = thread_index; body < step.body_count;
             body += lane_count)
            atomic_store_explicit(
                color_owners + body, uncolored, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_device);

        for (uint active_index = thread_index;
             active_index < active_pair_count;
             active_index += lane_count) {
            const uint pair = active_pairs[active_index];
            device const PMContactManifold &manifold = manifolds[pair];
            if (manifold.count == 0u || manifold.color != uncolored)
                continue;
            const uint body = pair / step.body_count;
            const uint collider = pair % step.body_count;
            const uint body_owner = compounds[body].eligible != 0u
                ? compounds[body].root : body;
            const uint collider_owner =
                compounds[collider].eligible != 0u
                    ? compounds[collider].root : collider;
            const uint priority = pm_contact_color_priority(pair);
            atomic_fetch_min_explicit(
                color_owners + body_owner, priority, memory_order_relaxed);
            if (parameters[collider].inverse_mass > 0.0f)
                atomic_fetch_min_explicit(
                    color_owners + collider_owner, priority,
                    memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_device);

        for (uint active_index = thread_index;
             active_index < active_pair_count;
             active_index += lane_count) {
            const uint pair = active_pairs[active_index];
            device PMContactManifold &manifold = manifolds[pair];
            if (manifold.count == 0u || manifold.color != uncolored)
                continue;
            const uint body = pair / step.body_count;
            const uint collider = pair % step.body_count;
            const uint body_owner = compounds[body].eligible != 0u
                ? compounds[body].root : body;
            const uint collider_owner =
                compounds[collider].eligible != 0u
                    ? compounds[collider].root : collider;
            const uint priority = pm_contact_color_priority(pair);
            const bool body_owned = atomic_load_explicit(
                color_owners + body_owner, memory_order_relaxed) == priority;
            const bool collider_owned =
                parameters[collider].inverse_mass <= 0.0f ||
                atomic_load_explicit(
                    color_owners + collider_owner,
                    memory_order_relaxed) == priority;
            if (body_owned && collider_owned) {
                manifold.color = color;
                atomic_fetch_max_explicit(
                    &used_colors, color + 1u, memory_order_relaxed);
            } else if (color + 1u == color_round_count) {
                atomic_fetch_add_explicit(
                    &overflow_count, 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(
            mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    const uint used_color_count = atomic_load_explicit(
        &used_colors, memory_order_relaxed);
    const bool has_overflow = atomic_load_explicit(
        &overflow_count, memory_order_relaxed) != 0u;
    for (uint pass = 0u; pass <= iterations; ++pass) {
        for (uint color = 0u; color < used_color_count; ++color) {
            for (uint active_index = thread_index;
                 active_index < active_pair_count;
                 active_index += lane_count) {
                const uint pair = active_pairs[active_index];
                device PMContactManifold &manifold = manifolds[pair];
                if (manifold.color != color) continue;
                const uint body = pair / step.body_count;
                const uint collider = pair % step.body_count;
                pm_resolve_rigid_contact_pair(
                    states, parameters, step, manifold, events,
                    body, collider, pass == 1u, pass == 0u,
                    constraints, compounds);
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        if (thread_index == 0u && has_overflow) {
            for (uint active_index = 0u;
                 active_index < active_pair_count; ++active_index) {
                const uint pair = active_pairs[active_index];
                device PMContactManifold &manifold = manifolds[pair];
                if (manifold.count == 0u || manifold.color != uncolored)
                    continue;
                const uint body = pair / step.body_count;
                const uint collider = pair % step.body_count;
                pm_resolve_rigid_contact_pair(
                    states, parameters, step, manifold, events,
                    body, collider, pass == 1u, pass == 0u,
                    constraints, compounds);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Deterministic reusable primitives. One cooperative threadgroup processes
// fixed-size chunks in input order. This preserves stable ordering without
// device-wide atomics or relying on cross-threadgroup scheduling.
kernel void pm_exclusive_scan_u32(
    device const uint *input [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    threadgroup uint prefix[256];
    threadgroup uint carried;
    const uint lane_count = threads_per_group.x;
    if (lane == 0u) carried = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint block = 0u; block < count; block += lane_count) {
        const uint index = block + lane;
        prefix[lane] = index < count ? input[index] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 1u; offset < lane_count; offset <<= 1u) {
            const uint addend = lane >= offset ? prefix[lane - offset] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix[lane] += addend;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (index < count)
            output[index] = carried + (lane == 0u ? 0u : prefix[lane - 1u]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) carried += prefix[lane_count - 1u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void pm_radix_sort_u64_pass(
    device const ulong *input_keys [[buffer(0)]],
    device const uint *input_values [[buffer(1)]],
    device ulong *output_keys [[buffer(2)]],
    device uint *output_values [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant uint &shift [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    threadgroup uint offsets[256];
    const uint lane_count = threads_per_group.x;
    for (uint bucket = lane; bucket < 256u; bucket += lane_count) {
        uint bucket_count = 0u;
        for (uint index = 0u; index < count; ++index)
            bucket_count +=
                uint(((input_keys[index] >> shift) & 0xfful) == bucket);
        offsets[bucket] = bucket_count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
        uint offset = 0u;
        for (uint bucket = 0u; bucket < 256u; ++bucket) {
            const uint bucket_count = offsets[bucket];
            offsets[bucket] = offset;
            offset += bucket_count;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint bucket = lane; bucket < 256u; bucket += lane_count) {
        uint destination = offsets[bucket];
        for (uint index = 0u; index < count; ++index) {
            const ulong key = input_keys[index];
            if (uint((key >> shift) & 0xfful) != bucket) continue;
            output_keys[destination] = key;
            output_values[destination++] = input_values[index];
        }
    }
}

kernel void pm_select_flagged_u32(
    device const uint *input [[buffer(0)]],
    device const uchar *flags [[buffer(1)]],
    device uint *output [[buffer(2)]],
    device uint *output_count [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    threadgroup uint prefix[256];
    threadgroup uint selected;
    const uint lane_count = threads_per_group.x;
    if (lane == 0u) selected = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint block = 0u; block < count; block += lane_count) {
        const uint index = block + lane;
        prefix[lane] = index < count && flags[index] != 0u ? 1u : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 1u; offset < lane_count; offset <<= 1u) {
            const uint addend = lane >= offset ? prefix[lane - offset] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix[lane] += addend;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (index < count && flags[index] != 0u)
            output[selected + (lane == 0u ? 0u : prefix[lane - 1u])] =
                input[index];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) selected += prefix[lane_count - 1u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) *output_count = selected;
}

kernel void pm_segmented_sum_f32(
    device const ulong *sorted_keys [[buffer(0)]],
    device const float *values [[buffer(1)]],
    device ulong *output_keys [[buffer(2)]],
    device float *output_sums [[buffer(3)]],
    device uint *output_count [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    const uint lane_count = threads_per_group.x;
    if (count == 0) {
        if (lane == 0u) *output_count = 0u;
        return;
    }

    for (uint index = lane; index < count; index += lane_count) {
        const ulong key = sorted_keys[index];
        if (index != 0u && sorted_keys[index - 1u] == key) continue;
        uint segment = 0u;
        for (uint prior = 0u; prior < index; ++prior)
            segment += uint(prior == 0u ||
                            sorted_keys[prior - 1u] != sorted_keys[prior]);
        float sum = values[index];
        for (uint next = index + 1u;
             next < count && sorted_keys[next] == key; ++next) {
            sum += values[next];
        }
        output_keys[segment] = key;
        output_sums[segment] = sum;
    }
    if (lane == 0u) {
        uint segment_count = 1u;
        for (uint index = 1u; index < count; ++index)
            segment_count += uint(sorted_keys[index - 1u] != sorted_keys[index]);
        *output_count = segment_count;
    }
}

kernel void pm_minmax_f32(
    device const float *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    threadgroup float minimums[256];
    threadgroup float maximums[256];
    const uint lane_count = threads_per_group.x;
    if (count == 0) {
        if (lane == 0u) *output = float2(INFINITY, -INFINITY);
        return;
    }

    float minimum = INFINITY;
    float maximum = -INFINITY;
    for (uint index = lane; index < count; index += lane_count) {
        minimum = metal::min(minimum, input[index]);
        maximum = metal::max(maximum, input[index]);
    }
    minimums[lane] = minimum;
    maximums[lane] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count >> 1u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            minimums[lane] = metal::min(minimums[lane],
                                        minimums[lane + stride]);
            maximums[lane] = metal::max(maximums[lane],
                                        maximums[lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) *output = float2(minimums[0], maximums[0]);
}

// Correctness-first particle-system kernels. Each system keeps stable public
// ordering; constraint and coupling passes intentionally use one lane until
// the stable sort/reduction versions replace them.
struct PMFluidConstants {
    float timestep;
    PMPackedVec3 gravity;
    uint count;
    float particle_radius;
    float support_radius;
    float repulsion;
    float viscosity;
    float normal_damping;
    float velocity_damping;
    float maximum_speed;
    float maximum_pair_acceleration;
    float rest_density;
    uint maximum_neighbors;
};

struct PMParticleMetadata {
    uint count;
    uint next_stable_id;
    uint revision;
    uint capacity;
    ulong emitted;
    ulong destroyed;
    ulong boiled;
};

struct PMSplitCounter {
    uint low;
    uint high;
};

struct PMFluidSmokeConstants {
    float timestep;
    PMPackedVec3 gravity;
    PMPackedVec3 heater_center;
    PMQuaternion heater_orientation;
    PMPackedVec2 heater_half_extents;
    float heater_temperature;
    float boiling_temperature;
    float heat_transfer_rate;
    float wind_drag;
    float steam_rise_speed;
    float water_radius;
    float smoke_radius;
    float smoke_rest_number_density;
    float smoke_maximum_speed;
    float smoke_lifetime;
    uint fluid_capacity;
    uint smoke_capacity;
    uint grid_resolution;
    uint grid_vertical_resolution;
    float grid_spacing;
    PMPackedVec3 grid_minimum;
};

struct PMSourceConstants {
    uint site_count;
    uint capacity;
    PMPackedVec3 initial_velocity;
    float initial_temperature;
    float clearance;
    uint enabled;
};

struct PMDestroyConstants {
    PMPackedVec3 center;
    PMQuaternion orientation;
    PMPackedVec2 half_extents;
    uint crossing;
    uint enabled;
};

struct PMSmokeMetadata {
    uint count;
    uint next_particle;
    float emission_remainder;
    uint revision;
    ulong emitted;
};

struct PMSmokeConstants {
    float timestep;
    PMPackedVec3 gravity;
    PMPackedVec3 emitter_center;
    PMPackedVec3 initial_velocity;
    PMPackedVec3 wind;
    PMPackedVec2 emitter_half_extents;
    uint command;
    float lifetime;
    float particle_radius;
    float buoyancy;
    float response;
    float maximum_speed;
    float rest_number_density;
    float pressure_stiffness;
    float viscosity;
    float vorticity_confinement;
    uint capacity;
    uint grid_resolution;
    uint grid_vertical_resolution;
    uint grid_pressure_iterations;
    PMPackedVec3 grid_minimum;
    float grid_spacing;
    float grid_kinematic_viscosity;
    float grid_les_coefficient;
    float grid_pressure_tolerance;
};

struct PMSmokeGridContribution {
    ulong density;
    ulong temperature;
};

struct PMDeformableConstants {
    float timestep;
    PMPackedVec3 gravity;
    uint count;
    uint bond_count;
    uint solver_iterations;
    float compliance;
    float velocity_damping;
    float maximum_speed;
    float radius;
    float break_strain;
    uint fracture_persistence;
    float impact_break_impulse;
    uint surface_count;
    uint triangle_index_count;
    uint preserve_volume;
    float target_volume;
    float volume_compliance;
    float shape_matching_stiffness;
    float maximum_projection_fraction;
    uint self_collision;
    float spring_damping;
    float constraint_velocity_response;
};

struct PMMetalBond {
    uint first;
    uint second;
    float rest_length;
    float compliance;
    uint active;
};

struct PMDeformableNeighbor {
    uint index;
    float rest_length;
    float compliance;
    uint bond;
};

struct PMMetalSurfaceBinding {
    uint nodes[4];
    float weights[4];
};

struct PMCouplingConstants {
    float timestep;
    uint count_a;
    uint count_b;
    uint mode;
    float contact_distance;
    float stiffness;
    float damping;
    float friction;
    float maximum_force;
    uint first_vertex;
    uint last_vertex;
    uint enabled;
};

struct PMFluidClothConstants {
    float timestep;
    uint surface_index_count;
    float contact_distance;
    float interaction_radius;
    float stiffness;
    float damping;
    float tangential_drag;
    float maximum_force;
    float particle_mass;
    float orientation;
    float maximum_particle_speed;
    uint surface_count;
    uint enabled;
    uint cloth_count;
};

struct PMFluidSoftConstants {
    float timestep;
    uint node_count;
    uint surface_count;
    uint surface_index_count;
    float contact_distance;
    float friction;
    float particle_mass;
    float maximum_particle_speed;
    float maximum_soft_speed;
    float orientation;
    uint enabled;
    float maximum_projection;
    float frame_inverse_timestep;
    uint rigid_count;
    float rigid_recovery_radius;
};

struct PMSoftClothConstants {
    float timestep;
    uint soft_count;
    uint soft_surface_count;
    uint cloth_count;
    uint cloth_surface_count;
    uint cloth_surface_index_count;
    float contact_distance;
    float friction;
    float maximum_soft_speed;
    uint solver_iterations;
    uint enabled;
};

struct PMSoftClothContact {
    uint vertices[3];
    float weights[3];
    float soft_inverse_mass_fraction;
    uint active;
    PMPackedVec3 position_impulse;
    PMPackedVec3 velocity_impulse;
};

struct PMSmokeSurfaceConstants {
    float timestep;
    uint target_count;
    uint surface_index_count;
    uint mode;
    float lifetime;
    float contact_distance;
    float wind_radius;
    float rest_number_density;
    float wind_drag;
    float maximum_wind_acceleration;
    float maximum_target_speed;
    float maximum_smoke_speed;
    uint enabled;
    uint grid_resolution;
    uint grid_vertical_resolution;
    float grid_spacing;
    PMPackedVec3 grid_minimum;
    float grid_kinematic_viscosity;
    float grid_les_coefficient;
    uint command;
    uint raster_triangle_base;
};

struct PMSmokeRigidRasterEntry {
    PMHandle body;
    uint triangle_base;
};

struct PMSmokeRopeConstants {
    float timestep;
    uint smoke_capacity;
    uint rope_count;
    float lifetime;
    float contact_distance;
    float wind_radius;
    float rest_number_density;
    float wind_drag;
    float maximum_wind_acceleration;
    float maximum_rope_speed;
    uint skip_first;
    uint skip_last;
    uint enabled;
    uint grid_resolution;
    uint grid_vertical_resolution;
    float grid_spacing;
    PMPackedVec3 grid_minimum;
};

struct PMRopeSoftConstants {
    float timestep;
    uint rope_count;
    uint soft_count;
    uint surface_count;
    uint surface_index_count;
    float contact_distance;
    float friction;
    float maximum_soft_acceleration;
    float maximum_rope_speed;
    float maximum_soft_speed;
    float node_radius;
    float anchor_support_radius_scale;
    float anchor_contact_support_radius_scale;
    float orientation;
    uint attach_first;
    uint attach_last;
    uint first_triangle;
    PMPackedVec3 first_weights;
    PMPackedVec3 first_offset;
    uint last_triangle;
    PMPackedVec3 last_weights;
    PMPackedVec3 last_offset;
    uint enabled;
    float frame_inverse_timestep;
};

struct PMParticleRigidConstants {
    uint particle_count;
    uint rigid_count;
    float radius;
    float friction;
    float restitution;
    float maximum_reaction_speed;
    float timestep;
    uint diagnostic_is_acceleration;
    PMHandle fluid;
    uint collect_contacts;
    float particle_inverse_mass;
    uint first_iteration;
    uint solid_contacts;
    uint share_position;
    uint first_spawned;
    uint recover_spawn;
    float spawn_clearance;
    PMPackedVec3 up;
    PMPackedVec3 gravity;
    float movable_mass;
};

struct PMParticleRigidContact {
    PMPackedVec3 normal;
    PMPackedVec3 point;
    float penetration;
    uint body;
};

struct PMSoftContactAccumulator {
    PMPackedVec3 momentum_delta;
    PMPackedVec3 friction_delta;
    PMPackedVec3 arm;
    float normal_delta;
};

struct PMSoftContactState {
    atomic_uint dynamic_contact_flag;
    PMPackedVec3 predicted_momentum;
};

struct PMClothRigidConstants {
    float timestep;
    uint vertex_count;
    uint triangle_index_count;
    uint surface_count;
    float thickness;
    float friction;
    uint rigid_count;
    uint fracture_enabled;
};

struct PMClothBodyCorrection {
    PMPackedVec3 offset;
    PMPackedVec3 impulse;
    PMPackedVec3 contact;
    float support_radius;
    float weight_sum;
    uint vertices[3];
    uint active;
};

struct PMPaintConstants {
    uint mode;
    PMHandle source;
    PMHandle target;
    uint mesh_index;
    uint width;
    uint height;
    float reach;
    float particle_radius;
    float cloth_thickness;
    uint rigid_count;
    uint cloth_count;
    uint cloth_index_count;
    uint enabled;
};

struct PMSmokeRigidConstants {
    PMHandle body;
    float particle_radius;
    float lifetime;
    float air_density;
    float drag_coefficient;
    float contact_distance;
    uint tracer_contact;
    uint enabled;
    uint rigid_count;
    uint grid_resolution;
    uint grid_vertical_resolution;
    float grid_spacing;
    PMPackedVec3 grid_minimum;
    float density_scale;
    float kinematic_viscosity;
    float les_coefficient;
    float timestep;
    float maximum_speed;
    float pressure_stiffness;
};

struct PMRopeAttachmentConstants {
    PMHandle first_body;
    PMPackedVec3 first_anchor;
    uint first_enabled;
    PMHandle last_body;
    PMPackedVec3 last_anchor;
    uint last_enabled;
    uint rigid_count;
    uint first_soft;
    uint last_soft;
    uint first_contact_skip;
    uint last_contact_skip;
    uint rigid_capacity;
};

struct PMRopeAnchorState {
    PMPackedVec3 position;
    PMPackedVec3 velocity;
    PMPackedVec3 impulse;
    float inverse_mass;
};

static_assert(sizeof(PMFluidConstants) == 60);
static_assert(sizeof(PMParticleMetadata) == 40);
static_assert(sizeof(PMSplitCounter) == 8);
static_assert(sizeof(PMFluidSmokeConstants) == 124);
static_assert(sizeof(PMSourceConstants) == 32);
static_assert(sizeof(PMDestroyConstants) == 44);
static_assert(sizeof(PMSmokeMetadata) == 24);
static_assert(sizeof(PMSmokeConstants) == 144);
static_assert(sizeof(PMSmokeGridContribution) == 16);
static_assert(sizeof(PMDeformableConstants) == 96);
static_assert(sizeof(PMMetalBond) == 20);
static_assert(sizeof(PMDeformableNeighbor) == 16);
static_assert(sizeof(PMMetalSurfaceBinding) == 32);
static_assert(sizeof(PMCouplingConstants) == 48);
static_assert(sizeof(PMFluidClothConstants) == 56);
static_assert(sizeof(PMFluidSoftConstants) == 60);
static_assert(sizeof(PMSoftClothConstants) == 44);
static_assert(sizeof(PMSoftClothContact) == 56);
static_assert(sizeof(PMSmokeSurfaceConstants) == 92);
static_assert(sizeof(PMSmokeRigidRasterEntry) == 12);
static_assert(sizeof(PMSmokeRopeConstants) == 76);
static_assert(sizeof(PMRopeSoftConstants) == 128);
static_assert(sizeof(PMParticleRigidConstants) == 100);
static_assert(sizeof(PMParticleRigidContact) == 32);
static_assert(sizeof(PMSoftContactAccumulator) == 40);
static_assert(sizeof(PMSoftContactState) == 16);
static_assert(sizeof(PMClothRigidConstants) == 32);
static_assert(sizeof(PMClothBodyCorrection) == 60);
static_assert(sizeof(PMSmokeRigidConstants) == 88);
static_assert(sizeof(PMRopeAttachmentConstants) == 72);
static_assert(sizeof(PMRopeAnchorState) == 40);
static_assert(sizeof(PMPaintConstants) == 60);

static ulong pm_fluid_cell_key(int x, int y, int z) {
    constexpr int bias = 1 << 20;
    x = clamp(x, -bias, bias - 1);
    y = clamp(y, -bias, bias - 1);
    z = clamp(z, -bias, bias - 1);
    return (ulong(x + bias) << 42u) |
           (ulong(y + bias) << 21u) |
           ulong(z + bias);
}

static uint pm_fluid_lower_bound(device const ulong *keys, uint size,
                                 ulong key) {
    uint lower = 0u;
    uint upper = size;
    while (lower < upper) {
        const uint middle = lower + (upper - lower) / 2u;
        if (keys[middle] < key)
            lower = middle + 1u;
        else
            upper = middle;
    }
    return lower;
}

constant constexpr uint pm_fluid_radix_block_size = 256u;

kernel void pm_fluid_cell_keys(
    device const PMPackedVec3 *positions [[buffer(0)]],
    constant PMFluidConstants &constants [[buffer(6)]],
    device const PMParticleMetadata &metadata [[buffer(7)]],
    device ulong *keys_a [[buffer(10)]],
    device uint *indices_a [[buffer(12)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= metadata.count) return;
    const float inverse_radius = 1.0f / constants.support_radius;
    constexpr float minimum_cell = float(-(1 << 20));
    constexpr float maximum_cell = float((1 << 20) - 1);
    const float3 scaled = floor(pm_load(positions[index]) * inverse_radius);
    const int3 cell = int3(clamp(
        scaled, float3(minimum_cell), float3(maximum_cell)));
    keys_a[index] = pm_fluid_cell_key(cell.x, cell.y, cell.z);
    indices_a[index] = index;
}

static void pm_fluid_radix_histogram(
    device const ulong *input_keys,
    device const PMParticleMetadata &metadata,
    device uint *histograms, uint shift, uint block) {
    const uint block_count =
        (metadata.capacity + pm_fluid_radix_block_size - 1u) /
        pm_fluid_radix_block_size;
    if (block >= block_count) return;
    uint local[256];
    for (uint bucket = 0u; bucket < 256u; ++bucket) local[bucket] = 0u;
    const uint begin = block * pm_fluid_radix_block_size;
    const uint end = min(begin + pm_fluid_radix_block_size, metadata.count);
    for (uint index = begin; index < end; ++index) {
        const uint bucket = uint((input_keys[index] >> shift) & 0xfful);
        ++local[bucket];
    }
    for (uint bucket = 0u; bucket < 256u; ++bucket)
        histograms[bucket * block_count + block] = local[bucket];
}

#define PM_FLUID_RADIX_HISTOGRAM_KERNEL(NAME, KEY_BUFFER, SHIFT)             \
kernel void NAME(                                                            \
    device const ulong *input_keys [[buffer(KEY_BUFFER)]],                   \
    device const PMParticleMetadata &metadata [[buffer(7)]],                 \
    device uint *histograms [[buffer(16)]],                                  \
    uint block [[thread_position_in_grid]]) {                                \
    pm_fluid_radix_histogram(input_keys, metadata, histograms, SHIFT, block); \
}

PM_FLUID_RADIX_HISTOGRAM_KERNEL(pm_fluid_radix_histogram_0, 10, 0u)
PM_FLUID_RADIX_HISTOGRAM_KERNEL(pm_fluid_radix_histogram_1, 11, 8u)
PM_FLUID_RADIX_HISTOGRAM_KERNEL(pm_fluid_radix_histogram_2, 10, 16u)
PM_FLUID_RADIX_HISTOGRAM_KERNEL(pm_fluid_radix_histogram_3, 11, 24u)
PM_FLUID_RADIX_HISTOGRAM_KERNEL(pm_fluid_radix_histogram_4, 10, 32u)
PM_FLUID_RADIX_HISTOGRAM_KERNEL(pm_fluid_radix_histogram_5, 11, 40u)
PM_FLUID_RADIX_HISTOGRAM_KERNEL(pm_fluid_radix_histogram_6, 10, 48u)
PM_FLUID_RADIX_HISTOGRAM_KERNEL(pm_fluid_radix_histogram_7, 11, 56u)

#undef PM_FLUID_RADIX_HISTOGRAM_KERNEL

kernel void pm_fluid_radix_prefix_blocks(
    device const PMParticleMetadata &metadata [[buffer(7)]],
    device uint *histograms [[buffer(16)]],
    device uint *bucket_offsets [[buffer(17)]],
    uint bucket [[thread_position_in_grid]]) {
    if (bucket >= 256u) return;
    const uint block_count =
        (metadata.capacity + pm_fluid_radix_block_size - 1u) /
        pm_fluid_radix_block_size;
    uint cursor = 0u;
    for (uint block = 0u; block < block_count; ++block) {
        const uint index = bucket * block_count + block;
        const uint count = histograms[index];
        histograms[index] = cursor;
        cursor += count;
    }
    bucket_offsets[bucket] = cursor;
}

kernel void pm_fluid_radix_prefix_buckets(
    device uint *bucket_offsets [[buffer(17)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    uint cursor = 0u;
    for (uint bucket = 0u; bucket < 256u; ++bucket) {
        const uint count = bucket_offsets[bucket];
        bucket_offsets[bucket] = cursor;
        cursor += count;
    }
}

static void pm_fluid_radix_scatter(
    device const ulong *input_keys, device const uint *input_indices,
    device ulong *output_keys, device uint *output_indices,
    device const PMParticleMetadata &metadata,
    device const uint *histograms, device const uint *bucket_offsets,
    uint shift, uint block) {
    const uint block_count =
        (metadata.capacity + pm_fluid_radix_block_size - 1u) /
        pm_fluid_radix_block_size;
    if (block >= block_count) return;
    uint local_offsets[256];
    for (uint bucket = 0u; bucket < 256u; ++bucket)
        local_offsets[bucket] = 0u;
    const uint begin = block * pm_fluid_radix_block_size;
    const uint end = min(begin + pm_fluid_radix_block_size, metadata.count);
    for (uint index = begin; index < end; ++index) {
        const ulong key = input_keys[index];
        const uint bucket = uint((key >> shift) & 0xfful);
        const uint destination = bucket_offsets[bucket] +
            histograms[bucket * block_count + block] +
            local_offsets[bucket]++;
        output_keys[destination] = key;
        output_indices[destination] = input_indices[index];
    }
}

#define PM_FLUID_RADIX_SCATTER_KERNEL(                                       \
    NAME, INPUT_KEY_BUFFER, INPUT_INDEX_BUFFER, OUTPUT_KEY_BUFFER,            \
    OUTPUT_INDEX_BUFFER, SHIFT)                                               \
kernel void NAME(                                                             \
    device const ulong *input_keys [[buffer(INPUT_KEY_BUFFER)]],              \
    device const uint *input_indices [[buffer(INPUT_INDEX_BUFFER)]],           \
    device ulong *output_keys [[buffer(OUTPUT_KEY_BUFFER)]],                   \
    device uint *output_indices [[buffer(OUTPUT_INDEX_BUFFER)]],               \
    device const PMParticleMetadata &metadata [[buffer(7)]],                  \
    device const uint *histograms [[buffer(16)]],                             \
    device const uint *bucket_offsets [[buffer(17)]],                         \
    uint block [[thread_position_in_grid]]) {                                 \
    pm_fluid_radix_scatter(                                                   \
        input_keys, input_indices, output_keys, output_indices, metadata,      \
        histograms, bucket_offsets, SHIFT, block);                            \
}

PM_FLUID_RADIX_SCATTER_KERNEL(pm_fluid_radix_scatter_0, 10, 12, 11, 13, 0u)
PM_FLUID_RADIX_SCATTER_KERNEL(pm_fluid_radix_scatter_1, 11, 13, 10, 12, 8u)
PM_FLUID_RADIX_SCATTER_KERNEL(pm_fluid_radix_scatter_2, 10, 12, 11, 13, 16u)
PM_FLUID_RADIX_SCATTER_KERNEL(pm_fluid_radix_scatter_3, 11, 13, 10, 12, 24u)
PM_FLUID_RADIX_SCATTER_KERNEL(pm_fluid_radix_scatter_4, 10, 12, 11, 13, 32u)
PM_FLUID_RADIX_SCATTER_KERNEL(pm_fluid_radix_scatter_5, 11, 13, 10, 12, 40u)
PM_FLUID_RADIX_SCATTER_KERNEL(pm_fluid_radix_scatter_6, 10, 12, 11, 13, 48u)
PM_FLUID_RADIX_SCATTER_KERNEL(pm_fluid_radix_scatter_7, 11, 13, 10, 12, 56u)

#undef PM_FLUID_RADIX_SCATTER_KERNEL

kernel void pm_fluid_forces(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const PMPackedVec3 *velocities [[buffer(1)]],
    device PMPackedVec3 *accelerations [[buffer(2)]],
    device const uint *stable_ids [[buffer(3)]],
    device float *foam [[buffer(4)]],
    device const float *temperatures [[buffer(5)]],
    constant PMFluidConstants &constants [[buffer(6)]],
    device const PMParticleMetadata &metadata [[buffer(7)]],
    device const PMPackedVec3 *previous [[buffer(8)]],
    device float *foam_sources [[buffer(9)]],
    device const ulong *cell_keys [[buffer(10)]],
    device const uint *sorted_indices [[buffer(12)]],
    device atomic_uint *neighbor_overflow [[buffer(14)]],
    device atomic_uint *maximum_neighbor_count [[buffer(15)]],
    uint index [[thread_position_in_grid]]) {
    (void)stable_ids;
    (void)temperatures;
    (void)previous;
    if (index >= metadata.count) return;
    const float3 position = pm_load(positions[index]);
    const float3 velocity = pm_load(velocities[index]);
    float3 acceleration = 0.0f;
    float3 outward = 0.0f;
    float weight = 0.0f;
    float relative_speed_squared = 0.0f;
    float neighboring_foam = 0.0f;
    uint neighbors = 0u;
    const float support = constants.support_radius;
    const float support_squared = support * support;
    constexpr int bias = 1 << 20;
    const float inverse_radius = 1.0f / support;
    const int3 center = int3(clamp(
        floor(position * inverse_radius), float3(float(-bias)),
        float3(float(bias - 1))));
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int3 cell = center + int3(dx, dy, dz);
                if (any(cell < int3(-bias)) ||
                    any(cell >= int3(bias)))
                    continue;
                const ulong key = pm_fluid_cell_key(
                    cell.x, cell.y, cell.z);
                for (uint item = pm_fluid_lower_bound(
                         cell_keys, metadata.count, key);
                     item < metadata.count && cell_keys[item] == key;
                     ++item) {
                    const uint other = sorted_indices[item];
                    if (other == index) continue;
                    const float3 delta =
                        position - pm_load(positions[other]);
                    const float distance_squared = dot(delta, delta);
                    if (distance_squared >= support_squared) continue;
                    ++neighbors;
                    const float distance =
                        sqrt(max(distance_squared, 1.0e-12f));
                    const float3 normal = distance_squared > 1.0e-12f
                        ? delta / distance
                        : (index < other
                               ? float3(-1.0f, 0.0f, 0.0f)
                               : float3(1.0f, 0.0f, 0.0f));
                    const float neighbor_weight =
                        1.0f - distance / support;
                    outward += normal * neighbor_weight;
                    weight += neighbor_weight;
                    const float3 relative_velocity =
                        pm_load(velocities[other]) - velocity;
                    relative_speed_squared +=
                        dot(relative_velocity, relative_velocity) *
                        neighbor_weight;
                    neighboring_foam = max(
                        neighboring_foam,
                        foam[other] * neighbor_weight);
                    const float normal_speed =
                        dot(relative_velocity, normal);
                    const float pair_acceleration =
                        constants.repulsion *
                            (1000.0f / constants.rest_density) *
                            neighbor_weight * neighbor_weight +
                        constants.normal_damping * normal_speed;
                    acceleration += normal * pair_acceleration;
                    acceleration += relative_velocity *
                        (constants.viscosity * neighbor_weight);
                }
            }
        }
    }
    atomic_fetch_max_explicit(maximum_neighbor_count, neighbors,
                              memory_order_relaxed);
    if (neighbors > constants.maximum_neighbors)
        atomic_fetch_add_explicit(neighbor_overflow, 1u,
                                  memory_order_relaxed);
    if (constants.maximum_pair_acceleration > 0.0f)
        acceleration = pm_limit(acceleration,
                                constants.maximum_pair_acceleration);
    accelerations[index] = pm_store(acceleration);
    const float3 gravity = pm_load(constants.gravity);
    const float3 up = dot(gravity, gravity) > 1.0e-12f
                          ? -normalize(gravity)
                          : float3(0.0f, 1.0f, 0.0f);
    const float exposure = length(outward) / max(weight, 1.0e-6f);
    const float upward = max(0.0f, dot(length(outward) > 1.0e-12f
                                          ? normalize(outward)
                                          : up,
                                      up));
    const float agitation = sqrt(relative_speed_squared /
                                  max(weight, 1.0e-6f));
    foam_sources[index] = max(
        clamp((exposure - 0.12f) * 2.0f, 0.0f, 1.0f) * upward *
            clamp((agitation - 0.15f) * 1.5f, 0.0f, 1.0f),
        neighboring_foam * upward * 0.9f);
}

kernel void pm_fluid_integrate(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *velocities [[buffer(1)]],
    device const PMPackedVec3 *accelerations [[buffer(2)]],
    device const uint *stable_ids [[buffer(3)]],
    device float *foam [[buffer(4)]],
    device const float *temperatures [[buffer(5)]],
    constant PMFluidConstants &constants [[buffer(6)]],
    device const PMParticleMetadata &metadata [[buffer(7)]],
    device PMPackedVec3 *previous [[buffer(8)]],
    device const float *foam_sources [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
    (void)stable_ids;
    (void)foam;
    (void)temperatures;
    if (index >= metadata.count) return;
    previous[index] = positions[index];
    float3 velocity = pm_load(velocities[index]) +
                      (pm_load(constants.gravity) +
                       pm_load(accelerations[index])) * constants.timestep;
    velocity *= exp(-constants.velocity_damping * constants.timestep);
    velocity = pm_limit(velocity, constants.maximum_speed);
    velocities[index] = pm_store(velocity);
    positions[index] = pm_store(pm_load(positions[index]) +
                                velocity * constants.timestep);
    foam[index] = max(max(0.0f, foam[index] - constants.timestep * 0.7f),
                      foam_sources[index]);
}

kernel void pm_fluid_source(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *velocities [[buffer(1)]],
    device uint *stable_ids [[buffer(2)]],
    device float *temperatures [[buffer(3)]],
    device PMParticleMetadata &metadata [[buffer(4)]],
    device const PMPackedVec3 *sites [[buffer(5)]],
    constant PMSourceConstants &constants [[buffer(6)]],
    device PMPackedVec3 *previous [[buffer(7)]],
    device float *foam [[buffer(8)]],
    device float *foam_sources [[buffer(9)]],
    device PMPackedVec3 *accelerations [[buffer(10)]],
    device PMContactEvent *contact_samples [[buffer(11)]],
    device uint *contact_flags [[buffer(12)]],
    device PMSplitCounter &capacity_misses [[buffer(13)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    if (constants.enabled == 0u) return;
    threadgroup uint blocked[64];
    threadgroup uint live;
    threadgroup uint next_stable_id;
    threadgroup uint emitted;
    const uint lane_count = threads_per_group.x;
    const float clearance_squared = constants.clearance * constants.clearance;
    const uint capacity = metadata.capacity;
    if (lane == 0u) {
        live = min(metadata.count, capacity);
        next_stable_id = metadata.next_stable_id;
        emitted = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint site = 0u; site < constants.site_count; ++site) {
        const float3 candidate = pm_load(sites[site]);
        uint occupied = 0u;
        for (uint particle = lane; particle < live; particle += lane_count) {
            const float3 delta = pm_load(positions[particle]) - candidate;
            if (dot(delta, delta) < clearance_squared) {
                occupied = 1u;
                break;
            }
        }
        blocked[lane] = occupied;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = lane_count >> 1u; stride != 0u; stride >>= 1u) {
            if (lane < stride) blocked[lane] |= blocked[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u && blocked[0] == 0u) {
            if (live >= capacity) {
                const uint next = capacity_misses.low + 1u;
                capacity_misses.low = next;
                if (next == 0u) ++capacity_misses.high;
            } else {
                const uint destination = live++;
                positions[destination] = sites[site];
                previous[destination] = sites[site];
                velocities[destination] = constants.initial_velocity;
                stable_ids[destination] = next_stable_id++;
                temperatures[destination] = constants.initial_temperature;
                foam[destination] = 0.0f;
                foam_sources[destination] = 0.0f;
                accelerations[destination] = {0.0f, 0.0f, 0.0f};
                contact_samples[destination] = {};
                contact_flags[destination] = 0u;
                ++emitted;
            }
        }
        threadgroup_barrier(mem_flags::mem_device |
                            mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        metadata.count = live;
        metadata.next_stable_id = next_stable_id;
        metadata.emitted += emitted;
        metadata.revision += emitted;
    }
}

kernel void pm_fluid_capture_spawn_baseline(
    device const PMParticleMetadata &metadata [[buffer(7)]],
    device uint &spawn_baseline [[buffer(18)]],
    uint index [[thread_position_in_grid]]) {
    if (index == 0u)
        spawn_baseline = min(metadata.count, metadata.capacity);
}

kernel void pm_fluid_destroy(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device PMPackedVec3 *accelerations [[buffer(3)]],
    device uint *stable_ids [[buffer(4)]],
    device float *foam [[buffer(5)]],
    device float *temperatures [[buffer(6)]],
    device PMParticleMetadata &metadata [[buffer(7)]],
    device float *foam_sources [[buffer(8)]],
    constant PMDestroyConstants &constants [[buffer(9)]],
    device PMContactEvent *contact_samples [[buffer(10)]],
    device uint *contact_flags [[buffer(11)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    if (constants.enabled == 0u) return;
    threadgroup uint prefix[64];
    threadgroup uint destination_base;
    threadgroup uint removed;
    const uint lane_count = threads_per_group.x;
    const uint source_count = metadata.count;
    const PMQuaternion inverse =
        pm_quaternion_conjugate(constants.orientation);
    if (lane == 0u) {
        destination_base = 0u;
        removed = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint block = 0u; block < source_count; block += lane_count) {
        const uint source = block + lane;
        const bool valid = source < source_count;
        PMPackedVec3 position{};
        PMPackedVec3 old_position{};
        PMPackedVec3 velocity{};
        PMPackedVec3 acceleration{};
        uint stable_id = 0u;
        float foam_value = 0.0f;
        float foam_source = 0.0f;
        float temperature = 0.0f;
        PMContactEvent contact_sample{};
        uint contact_flag = 0u;
        uint keep = 0u;
        if (valid) {
            position = positions[source];
            old_position = previous[source];
            velocity = velocities[source];
            acceleration = accelerations[source];
            stable_id = stable_ids[source];
            foam_value = foam[source];
            foam_source = foam_sources[source];
            temperature = temperatures[source];
            contact_sample = contact_samples[source];
            contact_flag = contact_flags[source];
            const float3 old_relative =
                pm_load(old_position) - pm_load(constants.center);
            const float3 new_relative =
                pm_load(position) - pm_load(constants.center);
            const float3 old_local = pm_rotate(inverse, old_relative);
            const float3 new_local = pm_rotate(inverse, new_relative);
            const bool along = old_local.y <= 0.0f && new_local.y > 0.0f;
            const bool against = old_local.y >= 0.0f && new_local.y < 0.0f;
            const bool direction = constants.crossing == 0u
                                       ? along
                                       : (constants.crossing == 1u
                                              ? against
                                              : (along || against));
            const bool inside =
                abs(new_local.x) <= constants.half_extents.x &&
                abs(new_local.z) <= constants.half_extents.y;
            keep = uint(!(direction && inside));
        }
        prefix[lane] = keep;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 1u; offset < lane_count; offset <<= 1u) {
            const uint addend = lane >= offset ? prefix[lane - offset] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix[lane] += addend;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (keep != 0u) {
            const uint destination = destination_base +
                (lane == 0u ? 0u : prefix[lane - 1u]);
            positions[destination] = position;
            previous[destination] = old_position;
            velocities[destination] = velocity;
            accelerations[destination] = acceleration;
            stable_ids[destination] = stable_id;
            foam[destination] = foam_value;
            foam_sources[destination] = foam_source;
            temperatures[destination] = temperature;
            contact_samples[destination] = contact_sample;
            contact_flags[destination] = contact_flag;
        }
        threadgroup_barrier(mem_flags::mem_device |
                            mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint kept = prefix[lane_count - 1u];
            destination_base += kept;
            removed += min(lane_count, source_count - block) - kept;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        metadata.count = destination_base;
        metadata.revision += removed;
        metadata.destroyed += removed;
    }
}

static uint pm_hash(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

static float pm_hash_unit(uint value) {
    return float(pm_hash(value) & 0x00ffffffu) / 16777216.0f;
}

static uint pm_smoke_face_count(uint axis, uint n, uint height) {
    if (axis == 0u) return (n + 1u) * height * n;
    if (axis == 1u) return n * (height + 1u) * n;
    return n * height * (n + 1u);
}

static uint pm_smoke_face_offset(uint axis, uint n, uint height) {
    if (axis == 0u) return 0u;
    const uint x_count = pm_smoke_face_count(0u, n, height);
    if (axis == 1u) return x_count;
    return x_count + pm_smoke_face_count(1u, n, height);
}

static uint pm_smoke_face_total(uint n, uint height) {
    return pm_smoke_face_count(0u, n, height) +
           pm_smoke_face_count(1u, n, height) +
           pm_smoke_face_count(2u, n, height);
}

static uint pm_smoke_face_index(uint axis, int x, int y, int z,
                                uint n, uint height) {
    uint local = 0u;
    if (axis == 0u)
        local = uint(x) + (n + 1u) * (uint(y) + height * uint(z));
    else if (axis == 1u)
        local = uint(x) + n * (uint(y) + (height + 1u) * uint(z));
    else
        local = uint(x) + n * (uint(y) + height * uint(z));
    return pm_smoke_face_offset(axis, n, height) + local;
}

static void pm_smoke_face_coordinates(uint axis, uint local, uint n,
                                      uint height, thread int &x,
                                      thread int &y, thread int &z) {
    if (axis == 0u) {
        x = int(local % (n + 1u));
        y = int(local / (n + 1u) % height);
        z = int(local / ((n + 1u) * height));
    } else if (axis == 1u) {
        x = int(local % n);
        y = int(local / n % (height + 1u));
        z = int(local / (n * (height + 1u)));
    } else {
        x = int(local % n);
        y = int(local / n % height);
        z = int(local / (n * height));
    }
}

static bool pm_smoke_flat_face_coordinates(
    uint face, uint n, uint height, thread uint &axis, thread uint &local,
    thread int &x, thread int &y, thread int &z) {
    const uint x_count = pm_smoke_face_count(0u, n, height);
    const uint y_count = pm_smoke_face_count(1u, n, height);
    if (face < x_count) {
        axis = 0u;
        local = face;
    } else if (face < x_count + y_count) {
        axis = 1u;
        local = face - x_count;
    } else if (face < pm_smoke_face_total(n, height)) {
        axis = 2u;
        local = face - x_count - y_count;
    } else {
        return false;
    }
    pm_smoke_face_coordinates(axis, local, n, height, x, y, z);
    return true;
}

static float3 pm_smoke_face_position(uint axis, int x, int y, int z,
                                     float3 minimum, float spacing) {
    float3 offset = float3(float(x) + 0.5f, float(y) + 0.5f,
                           float(z) + 0.5f);
    if (axis == 0u) offset.x = float(x);
    if (axis == 1u) offset.y = float(y);
    if (axis == 2u) offset.z = float(z);
    return minimum + offset * spacing;
}

static float pm_smoke_sample_cell_scalar(device const float *values,
                                         float3 point, float3 minimum,
                                         float spacing, uint n,
                                         uint height) {
    const float3 p = (point - minimum) / spacing - 0.5f;
    const int3 lower = int3(floor(p));
    const float3 fraction = p - float3(lower);
    float result = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int x = clamp(lower.x + dx, 0, int(n) - 1);
                const int y = clamp(lower.y + dy, 0, int(height) - 1);
                const int z = clamp(lower.z + dz, 0, int(n) - 1);
                const float weight =
                    (dx != 0 ? fraction.x : 1.0f - fraction.x) *
                    (dy != 0 ? fraction.y : 1.0f - fraction.y) *
                    (dz != 0 ? fraction.z : 1.0f - fraction.z);
                const uint cell = uint(x) + n * (uint(y) + height * uint(z));
                result += values[cell] * weight;
            }
    return result;
}

static float3 pm_smoke_sample_cell_vector(
    device const PMPackedVec3 *values, float3 point, float3 minimum,
    float spacing, uint n, uint height) {
    const float3 p = (point - minimum) / spacing - 0.5f;
    const int3 lower = int3(floor(p));
    const float3 fraction = p - float3(lower);
    float3 result = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int x = clamp(lower.x + dx, 0, int(n) - 1);
                const int y = clamp(lower.y + dy, 0, int(height) - 1);
                const int z = clamp(lower.z + dz, 0, int(n) - 1);
                const float weight =
                    (dx != 0 ? fraction.x : 1.0f - fraction.x) *
                    (dy != 0 ? fraction.y : 1.0f - fraction.y) *
                    (dz != 0 ? fraction.z : 1.0f - fraction.z);
                const uint cell = uint(x) + n * (uint(y) + height * uint(z));
                result += pm_load(values[cell]) * weight;
            }
    return result;
}

static float pm_smoke_sample_face_component(
    device const float *values, uint axis, float3 point, float3 minimum,
    float spacing, uint n, uint height) {
    float3 p = (point - minimum) / spacing;
    int sx = int(n), sy = int(height), sz = int(n);
    if (axis == 0u) { p.y -= 0.5f; p.z -= 0.5f; ++sx; }
    if (axis == 1u) { p.x -= 0.5f; p.z -= 0.5f; ++sy; }
    if (axis == 2u) { p.x -= 0.5f; p.y -= 0.5f; ++sz; }
    const int3 lower = int3(floor(p));
    const float3 fraction = p - float3(lower);
    float result = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int x = clamp(lower.x + dx, 0, sx - 1);
                const int y = clamp(lower.y + dy, 0, sy - 1);
                const int z = clamp(lower.z + dz, 0, sz - 1);
                const float weight =
                    (dx != 0 ? fraction.x : 1.0f - fraction.x) *
                    (dy != 0 ? fraction.y : 1.0f - fraction.y) *
                    (dz != 0 ? fraction.z : 1.0f - fraction.z);
                result += values[pm_smoke_face_index(
                    axis, x, y, z, n, height)] * weight;
            }
    return result;
}

static float3 pm_smoke_sample_face_velocity(
    device const float *values, float3 point, float3 minimum, float spacing,
    uint n, uint height) {
    return float3(
        pm_smoke_sample_face_component(values, 0u, point, minimum, spacing,
                                       n, height),
        pm_smoke_sample_face_component(values, 1u, point, minimum, spacing,
                                       n, height),
        pm_smoke_sample_face_component(values, 2u, point, minimum, spacing,
                                       n, height));
}

static float pm_smoke_sample_face_strain(
    device const float *values, float3 point, float3 minimum, float spacing,
    uint n, uint height) {
    const float3 velocity_x_minus = pm_smoke_sample_face_velocity(
        values, point - float3(spacing, 0.0f, 0.0f), minimum, spacing,
        n, height);
    const float3 velocity_x_plus = pm_smoke_sample_face_velocity(
        values, point + float3(spacing, 0.0f, 0.0f), minimum, spacing,
        n, height);
    const float3 velocity_y_minus = pm_smoke_sample_face_velocity(
        values, point - float3(0.0f, spacing, 0.0f), minimum, spacing,
        n, height);
    const float3 velocity_y_plus = pm_smoke_sample_face_velocity(
        values, point + float3(0.0f, spacing, 0.0f), minimum, spacing,
        n, height);
    const float3 velocity_z_minus = pm_smoke_sample_face_velocity(
        values, point - float3(0.0f, 0.0f, spacing), minimum, spacing,
        n, height);
    const float3 velocity_z_plus = pm_smoke_sample_face_velocity(
        values, point + float3(0.0f, 0.0f, spacing), minimum, spacing,
        n, height);
    const float inverse_two_spacing = 0.5f / spacing;
    const float3 dx =
        (velocity_x_plus - velocity_x_minus) * inverse_two_spacing;
    const float3 dy =
        (velocity_y_plus - velocity_y_minus) * inverse_two_spacing;
    const float3 dz =
        (velocity_z_plus - velocity_z_minus) * inverse_two_spacing;
    const float sxy = 0.5f * (dx.y + dy.x);
    const float sxz = 0.5f * (dx.z + dz.x);
    const float syz = 0.5f * (dy.z + dz.y);
    return sqrt(max(
        0.0f, 2.0f * (dx.x * dx.x + dy.y * dy.y + dz.z * dz.z +
                       2.0f * (sxy * sxy + sxz * sxz + syz * syz))));
}

static float pm_smoke_pressure_open(device const float *face_boundary,
                                    uint direction, int x, int y, int z,
                                    uint n, uint height) {
    if (direction == 0u)
        return face_boundary[pm_smoke_face_index(
            0u, x, y, z, n, height)];
    if (direction == 1u)
        return face_boundary[pm_smoke_face_index(
            0u, x + 1, y, z, n, height)];
    if (direction == 2u)
        return face_boundary[pm_smoke_face_index(
            1u, x, y, z, n, height)];
    if (direction == 3u)
        return face_boundary[pm_smoke_face_index(
            1u, x, y + 1, z, n, height)];
    if (direction == 4u)
        return face_boundary[pm_smoke_face_index(
            2u, x, y, z, n, height)];
    return face_boundary[pm_smoke_face_index(
        2u, x, y, z + 1, n, height)];
}

static void pm_smoke_pressure_clear(device float *values, uint count) {
    for (uint index = 0u; index < count; ++index) values[index] = 0.0f;
}

static void pm_smoke_pressure_restrict_open_face(
    device const float *fine, uint fine_n, uint fine_height,
    device float *coarse, uint coarse_n, uint coarse_height, uint face) {
    const uint x_faces = pm_smoke_face_count(0u, coarse_n, coarse_height);
    const uint y_faces = pm_smoke_face_count(1u, coarse_n, coarse_height);
    uint axis = 0u;
    uint local = face;
    if (local >= x_faces) {
        local -= x_faces;
        axis = 1u;
    }
    if (axis == 1u && local >= y_faces) {
        local -= y_faces;
        axis = 2u;
    }
    int x = 0, y = 0, z = 0;
    pm_smoke_face_coordinates(
        axis, local, coarse_n, coarse_height, x, y, z);
    float sum = 0.0f;
    for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a) {
            int xx = 2 * x;
            int yy = 2 * y;
            int zz = 2 * z;
            if (axis == 0u) { yy += a; zz += b; }
            if (axis == 1u) { xx += a; zz += b; }
            if (axis == 2u) { xx += a; yy += b; }
            const int sx = int(fine_n) + (axis == 0u ? 1 : 0);
            const int sy = int(fine_height) + (axis == 1u ? 1 : 0);
            const int sz = int(fine_n) + (axis == 2u ? 1 : 0);
            xx = min(sx - 1, xx);
            yy = min(sy - 1, yy);
            zz = min(sz - 1, zz);
            sum += fine[pm_smoke_face_index(
                axis, xx, yy, zz, fine_n, fine_height)];
        }
    coarse[face] = 0.25f * sum;
}

static void pm_smoke_pressure_restrict_open(
    device const float *fine, uint fine_n, uint fine_height,
    device float *coarse, uint coarse_n, uint coarse_height) {
    const uint count = pm_smoke_face_total(coarse_n, coarse_height);
    for (uint face = 0u; face < count; ++face)
        pm_smoke_pressure_restrict_open_face(
            fine, fine_n, fine_height, coarse, coarse_n, coarse_height, face);
}

static void pm_smoke_pressure_sweep_cell(
    uint n, uint height, float spacing, device const float *rhs,
    device const float *pressure, device float *next,
    device const float *face_open, uint cell) {
    const int x = int(cell % n);
    const int y = int(cell / n % height);
    const int z = int(cell / (n * height));
    const int direction_x[6] = {-1, 1, 0, 0, 0, 0};
    const int direction_y[6] = {0, 0, -1, 1, 0, 0};
    const int direction_z[6] = {0, 0, 0, 0, -1, 1};
    const float spacing_squared = spacing * spacing;
    float sum = 0.0f;
    float diagonal = 0.0f;
    for (uint direction = 0u; direction < 6u; ++direction) {
        const float coefficient = pm_smoke_pressure_open(
            face_open, direction, x, y, z, n, height);
        if (coefficient <= 1.0e-5f) continue;
        const int xx = x + direction_x[direction];
        const int yy = y + direction_y[direction];
        const int zz = z + direction_z[direction];
        if (xx >= 0 && yy >= 0 && zz >= 0 && xx < int(n) &&
            yy < int(height) && zz < int(n))
            sum += coefficient * pressure[
                uint(xx) + n * (uint(yy) + height * uint(zz))];
        diagonal += coefficient;
    }
    const float jacobi = diagonal > 1.0e-6f
        ? (sum - spacing_squared * rhs[cell]) / diagonal : 0.0f;
    constexpr float omega = 2.0f / 3.0f;
    next[cell] = pressure[cell] + omega * (jacobi - pressure[cell]);
}

static void pm_smoke_pressure_sweep(
    uint n, uint height, float spacing, device const float *rhs,
    device const float *pressure, device float *next,
    device const float *face_open) {
    const uint count = n * height * n;
    for (uint cell = 0u; cell < count; ++cell)
        pm_smoke_pressure_sweep_cell(
            n, height, spacing, rhs, pressure, next, face_open, cell);
}

static void pm_smoke_pressure_smooth_pairs(
    uint n, uint height, float spacing, device const float *rhs,
    device float *primary, device float *alternate,
    device const float *face_open, uint pairs) {
    for (uint pair = 0u; pair < pairs; ++pair) {
        pm_smoke_pressure_sweep(
            n, height, spacing, rhs, primary, alternate, face_open);
        pm_smoke_pressure_sweep(
            n, height, spacing, rhs, alternate, primary, face_open);
    }
}

static float pm_smoke_pressure_residual_cell(
    uint n, uint height, float spacing, device const float *rhs,
    device const float *pressure, device float *residual,
    device const float *face_open, uint cell) {
    const int direction_x[6] = {-1, 1, 0, 0, 0, 0};
    const int direction_y[6] = {0, 0, -1, 1, 0, 0};
    const int direction_z[6] = {0, 0, 0, 0, -1, 1};
    const float inverse_spacing_squared = 1.0f / (spacing * spacing);
    const int x = int(cell % n);
    const int y = int(cell / n % height);
    const int z = int(cell / (n * height));
    float sum = 0.0f;
    float diagonal = 0.0f;
    for (uint direction = 0u; direction < 6u; ++direction) {
        const float coefficient = pm_smoke_pressure_open(
            face_open, direction, x, y, z, n, height);
        if (coefficient <= 1.0e-5f) continue;
        const int xx = x + direction_x[direction];
        const int yy = y + direction_y[direction];
        const int zz = z + direction_z[direction];
        if (xx >= 0 && yy >= 0 && zz >= 0 && xx < int(n) &&
            yy < int(height) && zz < int(n))
            sum += coefficient * pressure[
                uint(xx) + n * (uint(yy) + height * uint(zz))];
        diagonal += coefficient;
    }
    const float value = rhs[cell] -
        (sum - diagonal * pressure[cell]) * inverse_spacing_squared;
    residual[cell] = value;
    return abs(value);
}

static float pm_smoke_pressure_residual(
    uint n, uint height, float spacing, device const float *rhs,
    device const float *pressure, device float *residual,
    device const float *face_open) {
    const uint count = n * height * n;
    float maximum = 0.0f;
    for (uint cell = 0u; cell < count; ++cell)
        maximum = max(maximum, pm_smoke_pressure_residual_cell(
            n, height, spacing, rhs, pressure, residual, face_open, cell));
    return maximum;
}

constant constexpr uint pm_smoke_grid_splat_width = 27u;
constant constexpr uint pm_smoke_grid_splat_block_size = 256u;

static uint pm_smoke_grid_splat_count(
    device const PMSmokeMetadata &metadata) {
    return metadata.count * pm_smoke_grid_splat_width;
}

static float pm_smoke_grid_splat_weight(float value, int offset) {
    if (offset < 0)
        return 0.5f * (0.5f - value) * (0.5f - value);
    if (offset == 0) return 0.75f - value * value;
    return 0.5f * (0.5f + value) * (0.5f + value);
}

kernel void pm_smoke_grid_splat_generate(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const float *ages [[buffer(1)]],
    device const float *thermal_lift [[buffer(2)]],
    device const PMSmokeMetadata &metadata [[buffer(3)]],
    constant PMSmokeConstants &constants [[buffer(4)]],
    device uint *keys_a [[buffer(5)]],
    device PMSmokeGridContribution *contributions_a [[buffer(7)]],
    uint contribution [[thread_position_in_grid]]) {
    const uint contribution_count = pm_smoke_grid_splat_count(metadata);
    if (contribution >= contribution_count) return;
    keys_a[contribution] = 0xffffffffu;
    contributions_a[contribution] = {0ul, 0ul};
    const uint particle = contribution / pm_smoke_grid_splat_width;
    if (particle >= metadata.count || ages[particle] >= constants.lifetime ||
        constants.grid_resolution == 0u || constants.grid_spacing <= 0.0f)
        return;

    const uint stencil = contribution % pm_smoke_grid_splat_width;
    const int dx = int(stencil % 3u) - 1;
    const int dy = int(stencil / 3u % 3u) - 1;
    const int dz = int(stencil / 9u) - 1;
    const float3 minimum = pm_load(constants.grid_minimum);
    const float3 g = (pm_load(positions[particle]) - minimum) /
                         constants.grid_spacing -
                     0.5f;
    const int3 center = int3(floor(g + 0.5f));
    const int3 coordinate = center + int3(dx, dy, dz);
    const int resolution = int(constants.grid_resolution);
    const int vertical = int(constants.grid_vertical_resolution);
    if (any(coordinate < int3(0)) || coordinate.x >= resolution ||
        coordinate.y >= vertical || coordinate.z >= resolution)
        return;

    const float3 delta = g - float3(center);
    const float weight =
        pm_smoke_grid_splat_weight(delta.x, dx) *
        pm_smoke_grid_splat_weight(delta.y, dy) *
        pm_smoke_grid_splat_weight(delta.z, dz) /
        max(1.0f, constants.rest_number_density);
    constexpr float scale = 16777216.0f;
    keys_a[contribution] = uint(coordinate.x) +
        constants.grid_resolution *
            (uint(coordinate.y) +
             constants.grid_vertical_resolution * uint(coordinate.z));
    contributions_a[contribution] = {
        ulong(weight * scale + 0.5f),
        ulong(weight * clamp(thermal_lift[particle], 0.0f,
                             constants.maximum_speed) *
                  scale +
              0.5f)};
}

static void pm_smoke_grid_splat_histogram(
    device const uint *input_keys,
    device const PMSmokeMetadata &metadata,
    constant PMSmokeConstants &constants,
    device uint *histograms, uint shift, uint block) {
    const uint contribution_count = pm_smoke_grid_splat_count(metadata);
    const uint block_count =
        (contribution_count + pm_smoke_grid_splat_block_size - 1u) /
        pm_smoke_grid_splat_block_size;
    if (block >= block_count) return;
    uint local[256];
    for (uint bucket = 0u; bucket < 256u; ++bucket) local[bucket] = 0u;
    const uint begin = block * pm_smoke_grid_splat_block_size;
    const uint end = min(begin + pm_smoke_grid_splat_block_size,
                         contribution_count);
    for (uint index = begin; index < end; ++index)
        ++local[(input_keys[index] >> shift) & 0xffu];
    for (uint bucket = 0u; bucket < 256u; ++bucket)
        histograms[bucket * block_count + block] = local[bucket];
}

#define PM_SMOKE_GRID_SPLAT_HISTOGRAM_KERNEL(NAME, KEY_BUFFER, SHIFT)        \
kernel void NAME(                                                            \
    device const uint *input_keys [[buffer(KEY_BUFFER)]],                    \
    device const PMSmokeMetadata &metadata [[buffer(3)]],                    \
    constant PMSmokeConstants &constants [[buffer(4)]],                      \
    device uint *histograms [[buffer(9)]],                                   \
    uint block [[thread_position_in_grid]]) {                                \
    pm_smoke_grid_splat_histogram(                                           \
        input_keys, metadata, constants, histograms, SHIFT, block);          \
}

PM_SMOKE_GRID_SPLAT_HISTOGRAM_KERNEL(
    pm_smoke_grid_splat_histogram_0, 5, 0u)
PM_SMOKE_GRID_SPLAT_HISTOGRAM_KERNEL(
    pm_smoke_grid_splat_histogram_1, 6, 8u)
PM_SMOKE_GRID_SPLAT_HISTOGRAM_KERNEL(
    pm_smoke_grid_splat_histogram_2, 5, 16u)
PM_SMOKE_GRID_SPLAT_HISTOGRAM_KERNEL(
    pm_smoke_grid_splat_histogram_3, 6, 24u)

#undef PM_SMOKE_GRID_SPLAT_HISTOGRAM_KERNEL

kernel void pm_smoke_grid_splat_prefix_blocks(
    device const PMSmokeMetadata &metadata [[buffer(3)]],
    constant PMSmokeConstants &constants [[buffer(4)]],
    device uint *histograms [[buffer(9)]],
    device uint *bucket_offsets [[buffer(10)]],
    uint bucket [[thread_position_in_grid]]) {
    if (bucket >= 256u) return;
    const uint contribution_count = pm_smoke_grid_splat_count(metadata);
    const uint block_count =
        (contribution_count + pm_smoke_grid_splat_block_size - 1u) /
        pm_smoke_grid_splat_block_size;
    uint cursor = 0u;
    for (uint block = 0u; block < block_count; ++block) {
        const uint index = bucket * block_count + block;
        const uint count = histograms[index];
        histograms[index] = cursor;
        cursor += count;
    }
    bucket_offsets[bucket] = cursor;
}

kernel void pm_smoke_grid_splat_prefix_buckets(
    device uint *bucket_offsets [[buffer(10)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    uint cursor = 0u;
    for (uint bucket = 0u; bucket < 256u; ++bucket) {
        const uint count = bucket_offsets[bucket];
        bucket_offsets[bucket] = cursor;
        cursor += count;
    }
}

static void pm_smoke_grid_splat_scatter(
    device const uint *input_keys,
    device const PMSmokeGridContribution *input_contributions,
    device uint *output_keys,
    device PMSmokeGridContribution *output_contributions,
    device const PMSmokeMetadata &metadata,
    constant PMSmokeConstants &constants,
    device const uint *histograms,
    device const uint *bucket_offsets,
    uint shift, uint block) {
    const uint contribution_count = pm_smoke_grid_splat_count(metadata);
    const uint block_count =
        (contribution_count + pm_smoke_grid_splat_block_size - 1u) /
        pm_smoke_grid_splat_block_size;
    if (block >= block_count) return;
    uint local_offsets[256];
    for (uint bucket = 0u; bucket < 256u; ++bucket)
        local_offsets[bucket] = 0u;
    const uint begin = block * pm_smoke_grid_splat_block_size;
    const uint end = min(begin + pm_smoke_grid_splat_block_size,
                         contribution_count);
    for (uint index = begin; index < end; ++index) {
        const uint key = input_keys[index];
        const uint bucket = (key >> shift) & 0xffu;
        const uint destination = bucket_offsets[bucket] +
            histograms[bucket * block_count + block] +
            local_offsets[bucket]++;
        output_keys[destination] = key;
        output_contributions[destination] = input_contributions[index];
    }
}

#define PM_SMOKE_GRID_SPLAT_SCATTER_KERNEL(                                  \
    NAME, INPUT_KEY, INPUT_VALUE, OUTPUT_KEY, OUTPUT_VALUE, SHIFT)           \
kernel void NAME(                                                             \
    device const uint *input_keys [[buffer(INPUT_KEY)]],                     \
    device const PMSmokeGridContribution *input_contributions                \
        [[buffer(INPUT_VALUE)]],                                              \
    device uint *output_keys [[buffer(OUTPUT_KEY)]],                         \
    device PMSmokeGridContribution *output_contributions                     \
        [[buffer(OUTPUT_VALUE)]],                                             \
    device const PMSmokeMetadata &metadata [[buffer(3)]],                    \
    constant PMSmokeConstants &constants [[buffer(4)]],                      \
    device const uint *histograms [[buffer(9)]],                             \
    device const uint *bucket_offsets [[buffer(10)]],                        \
    uint block [[thread_position_in_grid]]) {                                \
    pm_smoke_grid_splat_scatter(                                             \
        input_keys, input_contributions, output_keys, output_contributions,  \
        metadata, constants, histograms, bucket_offsets, SHIFT, block);      \
}

PM_SMOKE_GRID_SPLAT_SCATTER_KERNEL(
    pm_smoke_grid_splat_scatter_0, 5, 7, 6, 8, 0u)
PM_SMOKE_GRID_SPLAT_SCATTER_KERNEL(
    pm_smoke_grid_splat_scatter_1, 6, 8, 5, 7, 8u)
PM_SMOKE_GRID_SPLAT_SCATTER_KERNEL(
    pm_smoke_grid_splat_scatter_2, 5, 7, 6, 8, 16u)
PM_SMOKE_GRID_SPLAT_SCATTER_KERNEL(
    pm_smoke_grid_splat_scatter_3, 6, 8, 5, 7, 24u)

#undef PM_SMOKE_GRID_SPLAT_SCATTER_KERNEL

kernel void pm_smoke_grid_splat_resolve(
    device const PMSmokeMetadata &metadata [[buffer(3)]],
    constant PMSmokeConstants &constants [[buffer(4)]],
    device const uint *keys [[buffer(5)]],
    device const PMSmokeGridContribution *contributions [[buffer(7)]],
    device float *grid_density [[buffer(11)]],
    device float *grid_temperature [[buffer(12)]],
    uint cell [[thread_position_in_grid]]) {
    const uint cell_count = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    if (cell >= cell_count) return;
    const uint contribution_count = pm_smoke_grid_splat_count(metadata);
    uint lower = 0u;
    uint upper = contribution_count;
    while (lower < upper) {
        const uint middle = lower + (upper - lower) / 2u;
        if (keys[middle] < cell)
            lower = middle + 1u;
        else
            upper = middle;
    }
    ulong density = 0ul;
    ulong temperature = 0ul;
    while (lower < contribution_count && keys[lower] == cell) {
        density += contributions[lower].density;
        temperature += contributions[lower].temperature;
        ++lower;
    }
    constexpr float inverse_scale = 1.0f / 16777216.0f;
    grid_density[cell] = float(density) * inverse_scale;
    grid_temperature[cell] = float(temperature) * inverse_scale;
}

static void pm_smoke_pressure_restrict_residual_cell(
    device const float *fine, uint fine_n, uint fine_height,
    device float *coarse, uint coarse_n, uint coarse_height, uint cell) {
    const int x = int(cell % coarse_n);
    const int y = int(cell / coarse_n % coarse_height);
    const int z = int(cell / (coarse_n * coarse_height));
    float sum = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int xx = min(int(fine_n) - 1, 2 * x + dx);
                const int yy = min(int(fine_height) - 1, 2 * y + dy);
                const int zz = min(int(fine_n) - 1, 2 * z + dz);
                sum += fine[uint(xx) + fine_n *
                    (uint(yy) + fine_height * uint(zz))];
            }
    coarse[cell] = 0.125f * sum;
}

static void pm_smoke_pressure_restrict_residual(
    device const float *fine, uint fine_n, uint fine_height,
    device float *coarse, uint coarse_n, uint coarse_height) {
    const uint count = coarse_n * coarse_height * coarse_n;
    for (uint cell = 0u; cell < count; ++cell)
        pm_smoke_pressure_restrict_residual_cell(
            fine, fine_n, fine_height, coarse, coarse_n, coarse_height, cell);
}

static void pm_smoke_pressure_prolong_add_cell(
    device const float *coarse, uint coarse_n, uint coarse_height,
    device float *fine, uint fine_n, uint fine_height, uint cell) {
    const int x = int(cell % fine_n);
    const int y = int(cell / fine_n % fine_height);
    const int z = int(cell / (fine_n * fine_height));
    const float gx = 0.5f * float(x) - 0.25f;
    const float gy = 0.5f * float(y) - 0.25f;
    const float gz = 0.5f * float(z) - 0.25f;
    const int x0 = int(floor(gx));
    const int y0 = int(floor(gy));
    const int z0 = int(floor(gz));
    const float fx = gx - float(x0);
    const float fy = gy - float(y0);
    const float fz = gz - float(z0);
    float correction = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int cx = clamp(x0 + dx, 0, int(coarse_n) - 1);
                const int cy = clamp(y0 + dy, 0, int(coarse_height) - 1);
                const int cz = clamp(z0 + dz, 0, int(coarse_n) - 1);
                const float weight =
                    (dx != 0 ? fx : 1.0f - fx) *
                    (dy != 0 ? fy : 1.0f - fy) *
                    (dz != 0 ? fz : 1.0f - fz);
                correction += weight * coarse[
                    uint(cx) + coarse_n *
                        (uint(cy) + coarse_height * uint(cz))];
            }
    fine[cell] += 0.5f * correction;
}

static void pm_smoke_pressure_prolong_add(
    device const float *coarse, uint coarse_n, uint coarse_height,
    device float *fine, uint fine_n, uint fine_height) {
    const uint count = fine_n * fine_height * fine_n;
    for (uint cell = 0u; cell < count; ++cell)
        pm_smoke_pressure_prolong_add_cell(
            coarse, coarse_n, coarse_height, fine, fine_n, fine_height, cell);
}

struct PMSmokePressureLayout {
    uint n[4];
    uint height[4];
    uint cells[4];
    uint faces[4];
    uint primary_offset[4];
    uint alternate_offset[4];
    uint rhs_offset[4];
    uint residual_offset[4];
    uint open_offset[4];
};

static PMSmokePressureLayout pm_smoke_pressure_layout(
    constant PMSmokeConstants &constants) {
    PMSmokePressureLayout layout{};
    layout.n[0] = constants.grid_resolution;
    layout.height[0] = constants.grid_vertical_resolution;
    for (uint level = 1u; level < 4u; ++level) {
        layout.n[level] = max(2u, layout.n[level - 1u] / 2u);
        layout.height[level] = max(
            2u, layout.height[level - 1u] / 2u);
    }
    for (uint level = 0u; level < 4u; ++level) {
        layout.cells[level] =
            layout.n[level] * layout.height[level] * layout.n[level];
        layout.faces[level] = pm_smoke_face_total(
            layout.n[level], layout.height[level]);
    }
    uint cursor = 0u;
    layout.alternate_offset[0] = cursor;
    cursor += layout.cells[0];
    layout.residual_offset[0] = cursor;
    cursor += layout.cells[0];
    for (uint level = 1u; level < 3u; ++level) {
        layout.primary_offset[level] = cursor;
        cursor += layout.cells[level];
        layout.alternate_offset[level] = cursor;
        cursor += layout.cells[level];
        layout.rhs_offset[level] = cursor;
        cursor += layout.cells[level];
        layout.residual_offset[level] = cursor;
        cursor += layout.cells[level];
        layout.open_offset[level] = cursor;
        cursor += layout.faces[level];
    }
    layout.primary_offset[3] = cursor;
    cursor += layout.cells[3];
    layout.alternate_offset[3] = cursor;
    cursor += layout.cells[3];
    layout.rhs_offset[3] = cursor;
    cursor += layout.cells[3];
    layout.open_offset[3] = cursor;
    return layout;
}

static device float *pm_smoke_pressure_primary(
    device float *grid_pressure, device float *scratch,
    thread const PMSmokePressureLayout &layout, uint level) {
    return level == 0u ? grid_pressure : scratch + layout.primary_offset[level];
}

static device float *pm_smoke_pressure_alternate(
    device float *scratch, thread const PMSmokePressureLayout &layout,
    uint level) {
    return scratch + layout.alternate_offset[level];
}

static device float *pm_smoke_pressure_rhs(
    device float *grid_divergence, device float *scratch,
    thread const PMSmokePressureLayout &layout, uint level) {
    return level == 0u ? grid_divergence : scratch + layout.rhs_offset[level];
}

static device float *pm_smoke_pressure_residual_buffer(
    device float *scratch, thread const PMSmokePressureLayout &layout,
    uint level) {
    return scratch + layout.residual_offset[level];
}

static device float *pm_smoke_pressure_open_buffer(
    device float *grid_face_boundary, device float *scratch,
    thread const PMSmokePressureLayout &layout, uint level) {
    return level == 0u
        ? grid_face_boundary : scratch + layout.open_offset[level];
}

static bool pm_smoke_pressure_converged(device atomic_uint *state) {
    return atomic_load_explicit(state + 2u, memory_order_relaxed) != 0u;
}

static void pm_smoke_pressure_restrict_open_level(
    constant PMSmokeConstants &constants,
    device float *grid_face_boundary, device float *scratch,
    device atomic_uint *state, uint coarse_level, uint face) {
    if (pm_smoke_pressure_converged(state)) return;
    const PMSmokePressureLayout layout =
        pm_smoke_pressure_layout(constants);
    if (face >= layout.faces[coarse_level]) return;
    device float *fine = pm_smoke_pressure_open_buffer(
        grid_face_boundary, scratch, layout, coarse_level - 1u);
    device float *coarse = pm_smoke_pressure_open_buffer(
        grid_face_boundary, scratch, layout, coarse_level);
    pm_smoke_pressure_restrict_open_face(
        fine, layout.n[coarse_level - 1u],
        layout.height[coarse_level - 1u], coarse,
        layout.n[coarse_level], layout.height[coarse_level], face);
}

static void pm_smoke_pressure_smooth_level(
    constant PMSmokeConstants &constants, device float *grid_pressure,
    device float *grid_divergence, device float *grid_face_boundary,
    device float *scratch, device atomic_uint *state, uint level,
    bool backward, uint cell) {
    if (pm_smoke_pressure_converged(state)) return;
    const PMSmokePressureLayout layout =
        pm_smoke_pressure_layout(constants);
    if (cell >= layout.cells[level]) return;
    device float *primary = pm_smoke_pressure_primary(
        grid_pressure, scratch, layout, level);
    device float *alternate = pm_smoke_pressure_alternate(
        scratch, layout, level);
    device float *rhs = pm_smoke_pressure_rhs(
        grid_divergence, scratch, layout, level);
    device float *open = pm_smoke_pressure_open_buffer(
        grid_face_boundary, scratch, layout, level);
    pm_smoke_pressure_sweep_cell(
        layout.n[level], layout.height[level],
        constants.grid_spacing * float(1u << level), rhs,
        backward ? alternate : primary,
        backward ? primary : alternate, open, cell);
}

static void pm_smoke_pressure_residual_level(
    constant PMSmokeConstants &constants, device float *grid_pressure,
    device float *grid_divergence, device float *grid_face_boundary,
    device float *scratch, device atomic_uint *state, uint level,
    bool track_maximum, uint cell) {
    if (pm_smoke_pressure_converged(state)) return;
    const PMSmokePressureLayout layout =
        pm_smoke_pressure_layout(constants);
    if (cell >= layout.cells[level]) return;
    device float *primary = pm_smoke_pressure_primary(
        grid_pressure, scratch, layout, level);
    device float *rhs = pm_smoke_pressure_rhs(
        grid_divergence, scratch, layout, level);
    device float *residual = pm_smoke_pressure_residual_buffer(
        scratch, layout, level);
    device float *open = pm_smoke_pressure_open_buffer(
        grid_face_boundary, scratch, layout, level);
    const float magnitude = pm_smoke_pressure_residual_cell(
        layout.n[level], layout.height[level],
        constants.grid_spacing * float(1u << level), rhs, primary,
        residual, open, cell);
    if (track_maximum)
        atomic_fetch_max_explicit(
            state + 1u, as_type<uint>(magnitude), memory_order_relaxed);
}

static void pm_smoke_pressure_restrict_residual_level(
    constant PMSmokeConstants &constants, device float *grid_divergence,
    device float *scratch, device atomic_uint *state, uint fine_level,
    uint cell) {
    if (pm_smoke_pressure_converged(state)) return;
    const PMSmokePressureLayout layout =
        pm_smoke_pressure_layout(constants);
    const uint coarse_level = fine_level + 1u;
    if (cell >= layout.cells[coarse_level]) return;
    device float *fine = pm_smoke_pressure_residual_buffer(
        scratch, layout, fine_level);
    device float *coarse = pm_smoke_pressure_rhs(
        grid_divergence, scratch, layout, coarse_level);
    pm_smoke_pressure_restrict_residual_cell(
        fine, layout.n[fine_level], layout.height[fine_level], coarse,
        layout.n[coarse_level], layout.height[coarse_level], cell);
}

static void pm_smoke_pressure_clear_level(
    constant PMSmokeConstants &constants, device float *grid_pressure,
    device float *scratch, device atomic_uint *state, uint level, uint cell) {
    if (pm_smoke_pressure_converged(state)) return;
    const PMSmokePressureLayout layout =
        pm_smoke_pressure_layout(constants);
    if (cell >= layout.cells[level]) return;
    pm_smoke_pressure_primary(
        grid_pressure, scratch, layout, level)[cell] = 0.0f;
    pm_smoke_pressure_alternate(scratch, layout, level)[cell] = 0.0f;
}

static void pm_smoke_pressure_prolong_level(
    constant PMSmokeConstants &constants, device float *grid_pressure,
    device float *scratch, device atomic_uint *state, uint coarse_level,
    uint cell) {
    if (pm_smoke_pressure_converged(state)) return;
    const PMSmokePressureLayout layout =
        pm_smoke_pressure_layout(constants);
    const uint fine_level = coarse_level - 1u;
    if (cell >= layout.cells[fine_level]) return;
    device float *coarse = pm_smoke_pressure_primary(
        grid_pressure, scratch, layout, coarse_level);
    device float *fine = pm_smoke_pressure_primary(
        grid_pressure, scratch, layout, fine_level);
    pm_smoke_pressure_prolong_add_cell(
        coarse, layout.n[coarse_level], layout.height[coarse_level], fine,
        layout.n[fine_level], layout.height[fine_level], cell);
}

#define PM_SMOKE_PRESSURE_RESTRICT_OPEN_KERNEL(NAME, LEVEL)                  \
kernel void NAME(                                                            \
    constant PMSmokeConstants &constants [[buffer(0)]],                      \
    device float *grid_face_boundary [[buffer(3)]],                          \
    device float *scratch [[buffer(4)]],                                     \
    device atomic_uint *state [[buffer(6)]],                                 \
    uint face [[thread_position_in_grid]]) {                                 \
    pm_smoke_pressure_restrict_open_level(                                   \
        constants, grid_face_boundary, scratch, state, LEVEL, face);         \
}

PM_SMOKE_PRESSURE_RESTRICT_OPEN_KERNEL(pm_smoke_pressure_restrict_open_1, 1u)
PM_SMOKE_PRESSURE_RESTRICT_OPEN_KERNEL(pm_smoke_pressure_restrict_open_2, 2u)
PM_SMOKE_PRESSURE_RESTRICT_OPEN_KERNEL(pm_smoke_pressure_restrict_open_3, 3u)

#undef PM_SMOKE_PRESSURE_RESTRICT_OPEN_KERNEL

#define PM_SMOKE_PRESSURE_SMOOTH_KERNEL(NAME, LEVEL, BACKWARD)               \
kernel void NAME(                                                             \
    constant PMSmokeConstants &constants [[buffer(0)]],                       \
    device float *grid_pressure [[buffer(1)]],                                \
    device float *grid_divergence [[buffer(2)]],                              \
    device float *grid_face_boundary [[buffer(3)]],                           \
    device float *scratch [[buffer(4)]],                                      \
    device atomic_uint *state [[buffer(6)]],                                  \
    uint cell [[thread_position_in_grid]]) {                                  \
    pm_smoke_pressure_smooth_level(                                           \
        constants, grid_pressure, grid_divergence, grid_face_boundary,        \
        scratch, state, LEVEL, BACKWARD, cell);                               \
}

PM_SMOKE_PRESSURE_SMOOTH_KERNEL(pm_smoke_pressure_smooth_0_forward, 0u, false)
PM_SMOKE_PRESSURE_SMOOTH_KERNEL(pm_smoke_pressure_smooth_0_backward, 0u, true)
PM_SMOKE_PRESSURE_SMOOTH_KERNEL(pm_smoke_pressure_smooth_1_forward, 1u, false)
PM_SMOKE_PRESSURE_SMOOTH_KERNEL(pm_smoke_pressure_smooth_1_backward, 1u, true)
PM_SMOKE_PRESSURE_SMOOTH_KERNEL(pm_smoke_pressure_smooth_2_forward, 2u, false)
PM_SMOKE_PRESSURE_SMOOTH_KERNEL(pm_smoke_pressure_smooth_2_backward, 2u, true)
PM_SMOKE_PRESSURE_SMOOTH_KERNEL(pm_smoke_pressure_smooth_3_forward, 3u, false)
PM_SMOKE_PRESSURE_SMOOTH_KERNEL(pm_smoke_pressure_smooth_3_backward, 3u, true)

#undef PM_SMOKE_PRESSURE_SMOOTH_KERNEL

#define PM_SMOKE_PRESSURE_RESIDUAL_KERNEL(NAME, LEVEL, TRACK)                 \
kernel void NAME(                                                             \
    constant PMSmokeConstants &constants [[buffer(0)]],                       \
    device float *grid_pressure [[buffer(1)]],                                \
    device float *grid_divergence [[buffer(2)]],                              \
    device float *grid_face_boundary [[buffer(3)]],                           \
    device float *scratch [[buffer(4)]],                                      \
    device atomic_uint *state [[buffer(6)]],                                  \
    uint cell [[thread_position_in_grid]]) {                                  \
    pm_smoke_pressure_residual_level(                                         \
        constants, grid_pressure, grid_divergence, grid_face_boundary,        \
        scratch, state, LEVEL, TRACK, cell);                                  \
}

PM_SMOKE_PRESSURE_RESIDUAL_KERNEL(pm_smoke_pressure_residual_0, 0u, true)
PM_SMOKE_PRESSURE_RESIDUAL_KERNEL(pm_smoke_pressure_residual_1, 1u, false)
PM_SMOKE_PRESSURE_RESIDUAL_KERNEL(pm_smoke_pressure_residual_2, 2u, false)

#undef PM_SMOKE_PRESSURE_RESIDUAL_KERNEL

#define PM_SMOKE_PRESSURE_RESTRICT_RESIDUAL_KERNEL(NAME, LEVEL)               \
kernel void NAME(                                                             \
    constant PMSmokeConstants &constants [[buffer(0)]],                       \
    device float *grid_divergence [[buffer(2)]],                              \
    device float *scratch [[buffer(4)]],                                      \
    device atomic_uint *state [[buffer(6)]],                                  \
    uint cell [[thread_position_in_grid]]) {                                  \
    pm_smoke_pressure_restrict_residual_level(                                \
        constants, grid_divergence, scratch, state, LEVEL, cell);             \
}

PM_SMOKE_PRESSURE_RESTRICT_RESIDUAL_KERNEL(
    pm_smoke_pressure_restrict_residual_0, 0u)
PM_SMOKE_PRESSURE_RESTRICT_RESIDUAL_KERNEL(
    pm_smoke_pressure_restrict_residual_1, 1u)
PM_SMOKE_PRESSURE_RESTRICT_RESIDUAL_KERNEL(
    pm_smoke_pressure_restrict_residual_2, 2u)

#undef PM_SMOKE_PRESSURE_RESTRICT_RESIDUAL_KERNEL

#define PM_SMOKE_PRESSURE_CLEAR_KERNEL(NAME, LEVEL)                           \
kernel void NAME(                                                             \
    constant PMSmokeConstants &constants [[buffer(0)]],                       \
    device float *grid_pressure [[buffer(1)]],                                \
    device float *scratch [[buffer(4)]],                                      \
    device atomic_uint *state [[buffer(6)]],                                  \
    uint cell [[thread_position_in_grid]]) {                                  \
    pm_smoke_pressure_clear_level(                                            \
        constants, grid_pressure, scratch, state, LEVEL, cell);               \
}

PM_SMOKE_PRESSURE_CLEAR_KERNEL(pm_smoke_pressure_clear_1, 1u)
PM_SMOKE_PRESSURE_CLEAR_KERNEL(pm_smoke_pressure_clear_2, 2u)
PM_SMOKE_PRESSURE_CLEAR_KERNEL(pm_smoke_pressure_clear_3, 3u)

#undef PM_SMOKE_PRESSURE_CLEAR_KERNEL

#define PM_SMOKE_PRESSURE_PROLONG_KERNEL(NAME, LEVEL)                         \
kernel void NAME(                                                             \
    constant PMSmokeConstants &constants [[buffer(0)]],                       \
    device float *grid_pressure [[buffer(1)]],                                \
    device float *scratch [[buffer(4)]],                                      \
    device atomic_uint *state [[buffer(6)]],                                  \
    uint cell [[thread_position_in_grid]]) {                                  \
    pm_smoke_pressure_prolong_level(                                          \
        constants, grid_pressure, scratch, state, LEVEL, cell);               \
}

PM_SMOKE_PRESSURE_PROLONG_KERNEL(pm_smoke_pressure_prolong_1, 1u)
PM_SMOKE_PRESSURE_PROLONG_KERNEL(pm_smoke_pressure_prolong_2, 2u)
PM_SMOKE_PRESSURE_PROLONG_KERNEL(pm_smoke_pressure_prolong_3, 3u)

#undef PM_SMOKE_PRESSURE_PROLONG_KERNEL

kernel void pm_smoke_pressure_cycle_begin(
    device atomic_uint *state [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u || pm_smoke_pressure_converged(state)) return;
    atomic_store_explicit(state + 1u, 0u, memory_order_relaxed);
}

kernel void pm_smoke_pressure_cycle_end(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device float *relative_residual [[buffer(5)]],
    device atomic_uint *state [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    const float rhs = as_type<float>(
        atomic_load_explicit(state, memory_order_relaxed));
    const float residual = as_type<float>(
        atomic_load_explicit(state + 1u, memory_order_relaxed));
    const float denominator = max(rhs, 1.0e-5f);
    relative_residual[0] = residual / denominator;
    atomic_store_explicit(
        state + 2u,
        residual <= constants.grid_pressure_tolerance * denominator ? 1u : 0u,
        memory_order_relaxed);
}

static float pm_smoke_grid_face_sample(
    device const float *values, uint axis, int x, int y, int z,
    uint n, uint height) {
    const int sx = int(n) + (axis == 0u ? 1 : 0);
    const int sy = int(height) + (axis == 1u ? 1 : 0);
    const int sz = int(n) + (axis == 2u ? 1 : 0);
    return values[pm_smoke_face_index(
        axis, clamp(x, 0, sx - 1), clamp(y, 0, sy - 1),
        clamp(z, 0, sz - 1), n, height)];
}

static float pm_smoke_grid_vorticity_magnitude(
    device const PMPackedVec3 *vorticity, int x, int y, int z,
    uint n, uint height) {
    x = clamp(x, 0, int(n) - 1);
    y = clamp(y, 0, int(height) - 1);
    z = clamp(z, 0, int(n) - 1);
    return length(pm_load(vorticity[
        uint(x) + n * (uint(y) + height * uint(z))]));
}

struct PMSmokeCellDiagnostics {
    float3 velocity;
    float3 vorticity;
    float strain;
    float divergence;
};

static PMSmokeCellDiagnostics pm_smoke_grid_cell_diagnostics(
    device const float *faces, uint n, uint height, float spacing,
    uint cell) {
    const int x = int(cell % n);
    const int y = int(cell / n % height);
    const int z = int(cell / (n * height));
    const float u0 = pm_smoke_grid_face_sample(
        faces, 0u, x, y, z, n, height);
    const float u1 = pm_smoke_grid_face_sample(
        faces, 0u, x + 1, y, z, n, height);
    const float v0 = pm_smoke_grid_face_sample(
        faces, 1u, x, y, z, n, height);
    const float v1 = pm_smoke_grid_face_sample(
        faces, 1u, x, y + 1, z, n, height);
    const float w0 = pm_smoke_grid_face_sample(
        faces, 2u, x, y, z, n, height);
    const float w1 = pm_smoke_grid_face_sample(
        faces, 2u, x, y, z + 1, n, height);
    const float inverse = 1.0f / spacing;
    const float transverse = 0.25f * inverse;
    const float dux = (u1 - u0) * inverse;
    const float duy =
        (pm_smoke_grid_face_sample(faces, 0u, x, y + 1, z, n, height) +
         pm_smoke_grid_face_sample(faces, 0u, x + 1, y + 1, z, n, height) -
         pm_smoke_grid_face_sample(faces, 0u, x, y - 1, z, n, height) -
         pm_smoke_grid_face_sample(faces, 0u, x + 1, y - 1, z, n, height)) *
        transverse;
    const float duz =
        (pm_smoke_grid_face_sample(faces, 0u, x, y, z + 1, n, height) +
         pm_smoke_grid_face_sample(faces, 0u, x + 1, y, z + 1, n, height) -
         pm_smoke_grid_face_sample(faces, 0u, x, y, z - 1, n, height) -
         pm_smoke_grid_face_sample(faces, 0u, x + 1, y, z - 1, n, height)) *
        transverse;
    const float dvx =
        (pm_smoke_grid_face_sample(faces, 1u, x + 1, y, z, n, height) +
         pm_smoke_grid_face_sample(faces, 1u, x + 1, y + 1, z, n, height) -
         pm_smoke_grid_face_sample(faces, 1u, x - 1, y, z, n, height) -
         pm_smoke_grid_face_sample(faces, 1u, x - 1, y + 1, z, n, height)) *
        transverse;
    const float dvy = (v1 - v0) * inverse;
    const float dvz =
        (pm_smoke_grid_face_sample(faces, 1u, x, y, z + 1, n, height) +
         pm_smoke_grid_face_sample(faces, 1u, x, y + 1, z + 1, n, height) -
         pm_smoke_grid_face_sample(faces, 1u, x, y, z - 1, n, height) -
         pm_smoke_grid_face_sample(faces, 1u, x, y + 1, z - 1, n, height)) *
        transverse;
    const float dwx =
        (pm_smoke_grid_face_sample(faces, 2u, x + 1, y, z, n, height) +
         pm_smoke_grid_face_sample(faces, 2u, x + 1, y, z + 1, n, height) -
         pm_smoke_grid_face_sample(faces, 2u, x - 1, y, z, n, height) -
         pm_smoke_grid_face_sample(faces, 2u, x - 1, y, z + 1, n, height)) *
        transverse;
    const float dwy =
        (pm_smoke_grid_face_sample(faces, 2u, x, y + 1, z, n, height) +
         pm_smoke_grid_face_sample(faces, 2u, x, y + 1, z + 1, n, height) -
         pm_smoke_grid_face_sample(faces, 2u, x, y - 1, z, n, height) -
         pm_smoke_grid_face_sample(faces, 2u, x, y - 1, z + 1, n, height)) *
        transverse;
    const float dwz = (w1 - w0) * inverse;
    const float sxy = 0.5f * (duy + dvx);
    const float sxz = 0.5f * (duz + dwx);
    const float syz = 0.5f * (dvz + dwy);
    return {
        float3(0.5f * (u0 + u1), 0.5f * (v0 + v1), 0.5f * (w0 + w1)),
        float3(dwy - dvz, duz - dwx, dvx - duy),
        sqrt(max(0.0f, 2.0f *
            (dux * dux + dvy * dvy + dwz * dwz +
             2.0f * (sxy * sxy + sxz * sxz + syz * syz)))),
        dux + dvy + dwz};
}

kernel void pm_smoke_grid_pressure_begin(
    device float *relative_residual [[buffer(5)]],
    device atomic_uint *state [[buffer(6)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    relative_residual[0] = 0.0f;
    atomic_store_explicit(state, 0u, memory_order_relaxed);
    atomic_store_explicit(state + 1u, 0u, memory_order_relaxed);
    atomic_store_explicit(state + 2u, 0u, memory_order_relaxed);
}

kernel void pm_smoke_grid_mark_domain_boundaries(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device float *face_boundary [[buffer(3)]],
    uint face [[thread_position_in_grid]]) {
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    uint axis = 0u, local = 0u;
    int x = 0, y = 0, z = 0;
    if (!pm_smoke_flat_face_coordinates(
            face, n, height, axis, local, x, y, z))
        return;
    const int extent = axis == 0u ? int(n)
                       : axis == 1u ? int(height)
                                    : int(n);
    const int coordinate = axis == 0u ? x : axis == 1u ? y : z;
    if (coordinate != 0 && coordinate != extent) return;
    const float3 wind = pm_load(constants.wind);
    const float component = axis == 0u ? wind.x
                          : axis == 1u ? wind.y : wind.z;
    const float outward = coordinate == 0 ? -1.0f : 1.0f;
    if (component * outward >= 0.0f) return;
    const uint face_total = pm_smoke_face_total(n, height);
    face_boundary[face] = 0.0f;
    face_boundary[face_total + face] = component;
}

kernel void pm_smoke_grid_advect_forward(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device float *scratch [[buffer(4)]],
    device const float *face_velocity [[buffer(7)]],
    uint face [[thread_position_in_grid]]) {
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    uint axis = 0u, local = 0u;
    int x = 0, y = 0, z = 0;
    if (!pm_smoke_flat_face_coordinates(
            face, n, height, axis, local, x, y, z))
        return;
    const float3 minimum = pm_load(constants.grid_minimum);
    const float spacing = constants.grid_spacing;
    const float3 point = pm_smoke_face_position(
        axis, x, y, z, minimum, spacing);
    const float3 first = pm_smoke_sample_face_velocity(
        face_velocity, point, minimum, spacing, n, height);
    const float3 midpoint = point - first * (0.5f * constants.timestep);
    const float3 flow = pm_smoke_sample_face_velocity(
        face_velocity, midpoint, minimum, spacing, n, height);
    const float3 departure = point - flow * constants.timestep;
    scratch[face] = pm_smoke_sample_face_component(
        face_velocity, axis, departure, minimum, spacing, n, height);
}

kernel void pm_smoke_grid_advect_reverse(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device float *scratch [[buffer(4)]],
    uint face [[thread_position_in_grid]]) {
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    uint axis = 0u, local = 0u;
    int x = 0, y = 0, z = 0;
    if (!pm_smoke_flat_face_coordinates(
            face, n, height, axis, local, x, y, z))
        return;
    const uint face_total = pm_smoke_face_total(n, height);
    device const float *predicted = scratch;
    device float *reversed = scratch + face_total;
    const float3 minimum = pm_load(constants.grid_minimum);
    const float spacing = constants.grid_spacing;
    const float3 point = pm_smoke_face_position(
        axis, x, y, z, minimum, spacing);
    const float3 first = pm_smoke_sample_face_velocity(
        predicted, point, minimum, spacing, n, height);
    const float3 midpoint = point + first * (0.5f * constants.timestep);
    const float3 flow = pm_smoke_sample_face_velocity(
        predicted, midpoint, minimum, spacing, n, height);
    reversed[face] = pm_smoke_sample_face_component(
        predicted, axis, point + flow * constants.timestep,
        minimum, spacing, n, height);
}

kernel void pm_smoke_grid_correct_face(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device float *scratch [[buffer(4)]],
    device const float *face_velocity [[buffer(7)]],
    uint face [[thread_position_in_grid]]) {
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    uint axis = 0u, local = 0u;
    int x = 0, y = 0, z = 0;
    if (!pm_smoke_flat_face_coordinates(
            face, n, height, axis, local, x, y, z))
        return;
    const uint face_total = pm_smoke_face_total(n, height);
    device float *predicted = scratch;
    device const float *reversed = scratch + face_total;
    const float3 minimum = pm_load(constants.grid_minimum);
    const float spacing = constants.grid_spacing;
    const float3 point = pm_smoke_face_position(
        axis, x, y, z, minimum, spacing);
    const float3 first = pm_smoke_sample_face_velocity(
        face_velocity, point, minimum, spacing, n, height);
    const float3 midpoint = point - first * (0.5f * constants.timestep);
    const float3 flow = pm_smoke_sample_face_velocity(
        face_velocity, midpoint, minimum, spacing, n, height);
    float3 sample_position =
        (point - flow * constants.timestep - minimum) / spacing;
    int sx = int(n), sy = int(height), sz = int(n);
    if (axis == 0u) {
        sample_position.y -= 0.5f;
        sample_position.z -= 0.5f;
        ++sx;
    }
    if (axis == 1u) {
        sample_position.x -= 0.5f;
        sample_position.z -= 0.5f;
        ++sy;
    }
    if (axis == 2u) {
        sample_position.x -= 0.5f;
        sample_position.y -= 0.5f;
        ++sz;
    }
    const int3 lower = int3(floor(sample_position));
    float low = INFINITY;
    float high = -INFINITY;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int xx = clamp(lower.x + dx, 0, sx - 1);
                const int yy = clamp(lower.y + dy, 0, sy - 1);
                const int zz = clamp(lower.z + dz, 0, sz - 1);
                const float value = face_velocity[pm_smoke_face_index(
                    axis, xx, yy, zz, n, height)];
                low = min(low, value);
                high = max(high, value);
            }
    predicted[face] = clamp(
        predicted[face] + 0.5f * (face_velocity[face] - reversed[face]),
        low, high);
}

kernel void pm_smoke_grid_cell_diagnostics_pre(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device float *grid_divergence [[buffer(2)]],
    device const float *scratch [[buffer(4)]],
    device PMPackedVec3 *grid_vorticity [[buffer(9)]],
    uint cell [[thread_position_in_grid]]) {
    const uint count = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    if (cell >= count) return;
    const PMSmokeCellDiagnostics diagnostics =
        pm_smoke_grid_cell_diagnostics(
            scratch, constants.grid_resolution,
            constants.grid_vertical_resolution, constants.grid_spacing, cell);
    grid_vorticity[cell] = pm_store(diagnostics.vorticity);
    grid_divergence[cell] = diagnostics.strain;
}

kernel void pm_smoke_grid_subgrid_force(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device const PMPackedVec3 *grid_vorticity [[buffer(9)]],
    device PMPackedVec3 *grid_scratch [[buffer(12)]],
    uint cell [[thread_position_in_grid]]) {
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    const uint count = n * height * n;
    if (cell >= count) return;
    const int x = int(cell % n);
    const int y = int(cell / n % height);
    const int z = int(cell / (n * height));
    const float3 gradient = float3(
        pm_smoke_grid_vorticity_magnitude(
            grid_vorticity, x + 1, y, z, n, height) -
            pm_smoke_grid_vorticity_magnitude(
                grid_vorticity, x - 1, y, z, n, height),
        pm_smoke_grid_vorticity_magnitude(
            grid_vorticity, x, y + 1, z, n, height) -
            pm_smoke_grid_vorticity_magnitude(
                grid_vorticity, x, y - 1, z, n, height),
        pm_smoke_grid_vorticity_magnitude(
            grid_vorticity, x, y, z + 1, n, height) -
            pm_smoke_grid_vorticity_magnitude(
                grid_vorticity, x, y, z - 1, n, height)) *
        (0.5f / constants.grid_spacing);
    if (dot(gradient, gradient) < 1.0e-12f) {
        grid_scratch[cell] = {0.0f, 0.0f, 0.0f};
        return;
    }
    grid_scratch[cell] = pm_store(
        cross(pm_normalized_or(gradient, float3(1.0f, 0.0f, 0.0f)),
              pm_load(grid_vorticity[cell])) *
        (constants.vorticity_confinement * constants.grid_spacing));
}

kernel void pm_smoke_grid_apply_face_forces(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device const float *grid_divergence [[buffer(2)]],
    device float *face_boundary [[buffer(3)]],
    device float *scratch [[buffer(4)]],
    device const float *grid_density [[buffer(10)]],
    device const float *grid_temperature [[buffer(11)]],
    device const PMPackedVec3 *grid_scratch [[buffer(12)]],
    uint face [[thread_position_in_grid]]) {
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    uint axis = 0u, local = 0u;
    int x = 0, y = 0, z = 0;
    if (!pm_smoke_flat_face_coordinates(
            face, n, height, axis, local, x, y, z))
        return;
    const uint face_total = pm_smoke_face_total(n, height);
    device const float *source = scratch;
    device float *destination = scratch + face_total;
    device const float *open = face_boundary;
    device const float *wall = face_boundary + face_total;
    if (open[face] <= 1.0e-4f) {
        destination[face] = wall[face];
        return;
    }
    const float center = source[face];
    const float laplacian =
        (pm_smoke_grid_face_sample(source, axis, x - 1, y, z, n, height) +
         pm_smoke_grid_face_sample(source, axis, x + 1, y, z, n, height) +
         pm_smoke_grid_face_sample(source, axis, x, y - 1, z, n, height) +
         pm_smoke_grid_face_sample(source, axis, x, y + 1, z, n, height) +
         pm_smoke_grid_face_sample(source, axis, x, y, z - 1, n, height) +
         pm_smoke_grid_face_sample(source, axis, x, y, z + 1, n, height) -
         6.0f * center) /
        (constants.grid_spacing * constants.grid_spacing);
    const float3 minimum = pm_load(constants.grid_minimum);
    const float3 point = pm_smoke_face_position(
        axis, x, y, z, minimum, constants.grid_spacing);
    const float strain = pm_smoke_sample_cell_scalar(
        grid_divergence, point, minimum, constants.grid_spacing, n, height);
    const float viscosity = constants.grid_kinematic_viscosity +
        constants.grid_les_coefficient * constants.grid_les_coefficient *
            constants.grid_spacing * constants.grid_spacing * strain;
    const float3 curl_force = pm_smoke_sample_cell_vector(
        grid_scratch, point, minimum, constants.grid_spacing, n, height);
    const float density = pm_smoke_sample_cell_scalar(
        grid_density, point, minimum, constants.grid_spacing, n, height);
    const float thermal_loading = pm_smoke_sample_cell_scalar(
        grid_temperature, point, minimum, constants.grid_spacing, n, height);
    const float heat = density > 1.0e-5f
        ? clamp(thermal_loading / density, 0.0f, constants.maximum_speed)
        : 0.0f;
    const float3 up = pm_normalized_or(
        -pm_load(constants.gravity), float3(0.0f, 1.0f, 0.0f));
    const float3 buoyancy = up *
        (constants.buoyancy * density * constants.rest_number_density +
         2.0f * heat);
    const float body = axis == 0u ? curl_force.x + buoyancy.x
                     : axis == 1u ? curl_force.y + buoyancy.y
                                  : curl_force.z + buoyancy.z;
    destination[face] = clamp(
        center + constants.timestep * (viscosity * laplacian + body),
        -constants.maximum_speed, constants.maximum_speed);
}

kernel void pm_smoke_grid_apply_face_boundaries(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device const float *face_boundary [[buffer(3)]],
    device const float *scratch [[buffer(4)]],
    device float *face_velocity [[buffer(7)]],
    uint face [[thread_position_in_grid]]) {
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    uint axis = 0u, local = 0u;
    int x = 0, y = 0, z = 0;
    if (!pm_smoke_flat_face_coordinates(
            face, n, height, axis, local, x, y, z))
        return;
    const uint face_total = pm_smoke_face_total(n, height);
    device const float *forced = scratch + face_total;
    device const float *open = face_boundary;
    device const float *wall = face_boundary + face_total;
    const int extent = axis == 0u ? int(n)
                       : axis == 1u ? int(height)
                                    : int(n);
    const int coordinate = axis == 0u ? x : axis == 1u ? y : z;
    float value = forced[face];
    if (coordinate == 0 || coordinate == extent) {
        if (open[face] <= 1.0e-4f) {
            value = wall[face];
        } else {
            const int nx = axis == 0u
                ? (coordinate == 0 ? 1 : extent - 1) : x;
            const int ny = axis == 1u
                ? (coordinate == 0 ? 1 : extent - 1) : y;
            const int nz = axis == 2u
                ? (coordinate == 0 ? 1 : extent - 1) : z;
            value = forced[pm_smoke_face_index(
                axis, nx, ny, nz, n, height)];
        }
    }
    const float3 point = pm_smoke_face_position(
        axis, x, y, z, pm_load(constants.grid_minimum),
        constants.grid_spacing);
    if (abs(point.x - constants.emitter_center.x) <
            1.5f * constants.grid_spacing &&
        abs(point.y - constants.emitter_center.y) <=
            constants.emitter_half_extents.x +
                0.5f * constants.grid_spacing &&
        abs(point.z - constants.emitter_center.z) <=
            constants.emitter_half_extents.y +
                0.5f * constants.grid_spacing)
        value = axis == 0u ? constants.initial_velocity.x
              : axis == 1u ? constants.initial_velocity.y
                           : constants.initial_velocity.z;
    face_velocity[face] =
        open[face] * value + (1.0f - open[face]) * wall[face];
}

kernel void pm_smoke_grid_divergence_parallel(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device float *grid_divergence [[buffer(2)]],
    device atomic_uint *state [[buffer(6)]],
    device const float *face_velocity [[buffer(7)]],
    uint cell [[thread_position_in_grid]]) {
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    const uint count = n * height * n;
    if (cell >= count) return;
    const int x = int(cell % n);
    const int y = int(cell / n % height);
    const int z = int(cell / (n * height));
    const float flux =
        face_velocity[pm_smoke_face_index(0u, x + 1, y, z, n, height)] -
        face_velocity[pm_smoke_face_index(0u, x, y, z, n, height)] +
        face_velocity[pm_smoke_face_index(1u, x, y + 1, z, n, height)] -
        face_velocity[pm_smoke_face_index(1u, x, y, z, n, height)] +
        face_velocity[pm_smoke_face_index(2u, x, y, z + 1, n, height)] -
        face_velocity[pm_smoke_face_index(2u, x, y, z, n, height)];
    const float value = flux / constants.grid_spacing /
        max(constants.timestep, 1.0e-12f);
    grid_divergence[cell] = value;
    atomic_fetch_max_explicit(
        state, as_type<uint>(abs(value)), memory_order_relaxed);
}

kernel void pm_smoke_grid_project_face_parallel(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device const float *grid_pressure [[buffer(1)]],
    device const float *face_boundary [[buffer(3)]],
    device float *face_velocity [[buffer(7)]],
    uint face [[thread_position_in_grid]]) {
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    uint axis = 0u, local = 0u;
    int x = 0, y = 0, z = 0;
    if (!pm_smoke_flat_face_coordinates(
            face, n, height, axis, local, x, y, z))
        return;
    const uint face_total = pm_smoke_face_total(n, height);
    device const float *open = face_boundary;
    device const float *wall = face_boundary + face_total;
    const int extent = axis == 0u ? int(n)
                       : axis == 1u ? int(height)
                                    : int(n);
    const int coordinate = axis == 0u ? x : axis == 1u ? y : z;
    if (open[face] <= 1.0e-4f) {
        face_velocity[face] = wall[face];
        return;
    }
    float gradient = 0.0f;
    if (coordinate == 0 || coordinate == extent) {
        int cx = x, cy = y, cz = z;
        if (axis == 0u) cx = coordinate == 0 ? 0 : int(n) - 1;
        if (axis == 1u) cy = coordinate == 0 ? 0 : int(height) - 1;
        if (axis == 2u) cz = coordinate == 0 ? 0 : int(n) - 1;
        const float inside = grid_pressure[
            uint(cx) + n * (uint(cy) + height * uint(cz))];
        gradient = (coordinate == 0 ? inside : -inside) /
            constants.grid_spacing;
    } else {
        int lx = x, ly = y, lz = z;
        int rx = x, ry = y, rz = z;
        if (axis == 0u) { lx = x - 1; rx = x; }
        if (axis == 1u) { ly = y - 1; ry = y; }
        if (axis == 2u) { lz = z - 1; rz = z; }
        gradient =
            (grid_pressure[uint(rx) + n * (uint(ry) + height * uint(rz))] -
             grid_pressure[uint(lx) + n * (uint(ly) + height * uint(lz))]) /
            constants.grid_spacing;
    }
    face_velocity[face] -=
        constants.timestep * open[face] * gradient;
}

kernel void pm_smoke_grid_cell_diagnostics_post(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device float *grid_divergence [[buffer(2)]],
    device const float *face_velocity [[buffer(7)]],
    device PMPackedVec3 *grid_velocity [[buffer(8)]],
    device PMPackedVec3 *grid_vorticity [[buffer(9)]],
    uint cell [[thread_position_in_grid]]) {
    const uint count = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    if (cell >= count) return;
    const PMSmokeCellDiagnostics diagnostics =
        pm_smoke_grid_cell_diagnostics(
            face_velocity, constants.grid_resolution,
            constants.grid_vertical_resolution, constants.grid_spacing, cell);
    grid_velocity[cell] = pm_store(diagnostics.velocity);
    grid_vorticity[cell] = pm_store(diagnostics.vorticity);
    grid_divergence[cell] = diagnostics.divergence;
}

kernel void pm_smoke_grid_clear(
    constant PMSmokeConstants &constants [[buffer(7)]],
    device PMPackedVec3 *grid_velocity [[buffer(9)]],
    device float *grid_density [[buffer(11)]],
    device float *grid_temperature [[buffer(12)]],
    device uint *grid_solid [[buffer(13)]],
    device PMPackedVec3 *grid_vorticity [[buffer(14)]],
    device float *grid_divergence [[buffer(15)]],
    device float *grid_pressure_relative_residual [[buffer(18)]],
    device uint *grid_deformable_solid [[buffer(27)]],
    device float *grid_face_boundary [[buffer(30)]],
    uint index [[thread_position_in_grid]]) {
    if (constants.grid_resolution == 0u) return;
    const uint n = constants.grid_resolution;
    const uint height = constants.grid_vertical_resolution;
    const uint cells = n * height * n;
    const uint faces = pm_smoke_face_total(n, height);
    if (index < cells) {
        grid_velocity[index] = {0.0f, 0.0f, 0.0f};
        grid_density[index] = 0.0f;
        grid_temperature[index] = 0.0f;
        grid_solid[index] = 0u;
        grid_vorticity[index] = {0.0f, 0.0f, 0.0f};
        grid_divergence[index] = 0.0f;
        grid_deformable_solid[index] = 0u;
    }
    if (index < faces) {
        grid_face_boundary[index] = 1.0f;
        grid_face_boundary[faces + index] = 0.0f;
    }
    if (index == 0u) grid_pressure_relative_residual[0] = 0.0f;
}

static void pm_smoke_atomic_min_positive(device atomic_uint *address,
                                         float value) {
    atomic_fetch_min_explicit(address, as_type<uint>(value),
                              memory_order_relaxed);
}

kernel void pm_smoke_grid_raster_clear(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device uint *cell_nearest_triangle [[buffer(1)]],
    device uint *face_nearest_triangle [[buffer(3)]],
    device const uint *counts [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (constants.grid_resolution == 0u) return;
    const uint cells = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    const uint faces = pm_smoke_face_total(
        constants.grid_resolution, constants.grid_vertical_resolution);
    if (index < cells) cell_nearest_triangle[index] = 0xffffffffu;
    if (index < faces && counts[3] == 0u)
        face_nearest_triangle[index] = 0xffffffffu;
}

static bool pm_smoke_rigid_raster_triangle(
    uint flat_triangle,
    device const PMSmokeRigidRasterEntry *entries,
    uint entry_count,
    uint rigid_count,
    device const PMHandle *rigid_ids,
    device const PMRigidBodyState *rigid_states,
    device const PMRigidParameters *rigid_parameters,
    device const PMPackedVec3 *vertices,
    device const uint *indices,
    device const PMTriangleMeshInfo *meshes,
    thread uint &global_triangle,
    thread uint &body_index,
    thread float3 &a,
    thread float3 &b,
    thread float3 &c) {
    uint cursor = flat_triangle;
    for (uint entry_index = 0u; entry_index < entry_count; ++entry_index) {
        const PMSmokeRigidRasterEntry entry = entries[entry_index];
        uint dense = rigid_count;
        for (uint body = 0u; body < rigid_count; ++body) {
            if (rigid_ids[body].index == entry.body.index &&
                rigid_ids[body].generation == entry.body.generation) {
                dense = body;
                break;
            }
        }
        if (dense == rigid_count) continue;
        device const PMRigidParameters &parameters = rigid_parameters[dense];
        device const PMTriangleMeshInfo &mesh = meshes[parameters.mesh_index];
        const uint triangle_count = mesh.index_count / 3u;
        if (cursor >= triangle_count) {
            cursor -= triangle_count;
            continue;
        }
        const uint local = cursor * 3u;
        device const PMRigidBodyState &state = rigid_states[dense];
        a = pm_world_point(
            state, pm_load(vertices[mesh.vertex_offset +
                                    indices[mesh.index_offset + local]]));
        b = pm_world_point(
            state, pm_load(vertices[mesh.vertex_offset +
                                    indices[mesh.index_offset + local + 1u]]));
        c = pm_world_point(
            state, pm_load(vertices[mesh.vertex_offset +
                                    indices[mesh.index_offset + local + 2u]]));
        global_triangle = entry.triangle_base + cursor;
        body_index = dense;
        return true;
    }
    return false;
}

static void pm_smoke_raster_triangle_mark(
    float3 a, float3 b, float3 c, uint global_triangle,
    uint resolution, uint vertical, float spacing, float3 minimum,
    device atomic_uint *cell_nearest_triangle,
    device atomic_uint *face_boundary) {
    const float inverse_spacing = 1.0f / spacing;
    const float radius = 0.55f * spacing;
    const float radius_squared = radius * radius;
    const float3 lower = min(a, min(b, c)) - radius;
    const float3 upper = max(a, max(b, c)) + radius;
    const int x0 = max(0, int(floor(
        (lower.x - minimum.x) * inverse_spacing)));
    const int y0 = max(0, int(floor(
        (lower.y - minimum.y) * inverse_spacing)));
    const int z0 = max(0, int(floor(
        (lower.z - minimum.z) * inverse_spacing)));
    const int x1 = min(int(resolution) - 1, int(floor(
        (upper.x - minimum.x) * inverse_spacing)));
    const int y1 = min(int(vertical) - 1, int(floor(
        (upper.y - minimum.y) * inverse_spacing)));
    const int z1 = min(int(resolution) - 1, int(floor(
        (upper.z - minimum.z) * inverse_spacing)));
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const float3 point = minimum +
                    (float3(float(x), float(y), float(z)) + 0.5f) * spacing;
                const float3 closest = pm_closest_point_triangle(point, a, b, c);
                const float3 delta = point - closest;
                if (dot(delta, delta) > radius_squared) continue;
                const uint cell = uint(x) + resolution *
                    (uint(y) + vertical * uint(z));
                atomic_fetch_min_explicit(
                    cell_nearest_triangle + cell, global_triangle,
                    memory_order_relaxed);
            }
    for (uint axis = 0u; axis < 3u; ++axis) {
        float3 offset = 0.5f;
        if (axis == 0u) offset.x = 0.0f;
        if (axis == 1u) offset.y = 0.0f;
        if (axis == 2u) offset.z = 0.0f;
        const int sx = int(resolution) + (axis == 0u ? 1 : 0);
        const int sy = int(vertical) + (axis == 1u ? 1 : 0);
        const int sz = int(resolution) + (axis == 2u ? 1 : 0);
        const int fx0 = max(0, int(floor(
            (lower.x - minimum.x) * inverse_spacing - offset.x)));
        const int fy0 = max(0, int(floor(
            (lower.y - minimum.y) * inverse_spacing - offset.y)));
        const int fz0 = max(0, int(floor(
            (lower.z - minimum.z) * inverse_spacing - offset.z)));
        const int fx1 = min(sx - 1, int(floor(
            (upper.x - minimum.x) * inverse_spacing - offset.x)));
        const int fy1 = min(sy - 1, int(floor(
            (upper.y - minimum.y) * inverse_spacing - offset.y)));
        const int fz1 = min(sz - 1, int(floor(
            (upper.z - minimum.z) * inverse_spacing - offset.z)));
        for (int z = fz0; z <= fz1; ++z)
            for (int y = fy0; y <= fy1; ++y)
                for (int x = fx0; x <= fx1; ++x) {
                    const float3 point = minimum +
                        (float3(float(x), float(y), float(z)) + offset) *
                            spacing;
                    const float3 closest =
                        pm_closest_point_triangle(point, a, b, c);
                    const float distance_value = distance(point, closest);
                    if (distance_value > radius) continue;
                    const uint face = pm_smoke_face_index(
                        axis, x, y, z, resolution, vertical);
                    pm_smoke_atomic_min_positive(
                        face_boundary + face,
                        clamp(distance_value / radius, 0.0f, 1.0f));
                }
    }
}

static void pm_smoke_select_triangle_faces(
    float3 a, float3 b, float3 c, uint global_triangle,
    uint resolution, uint vertical, float spacing, float3 minimum,
    device const float *face_boundary,
    device atomic_uint *face_nearest_triangle) {
    const float inverse_spacing = 1.0f / spacing;
    const float radius = 0.55f * spacing;
    const float3 lower = min(a, min(b, c)) - radius;
    const float3 upper = max(a, max(b, c)) + radius;
    for (uint axis = 0u; axis < 3u; ++axis) {
        float3 offset = 0.5f;
        if (axis == 0u) offset.x = 0.0f;
        if (axis == 1u) offset.y = 0.0f;
        if (axis == 2u) offset.z = 0.0f;
        const int sx = int(resolution) + (axis == 0u ? 1 : 0);
        const int sy = int(vertical) + (axis == 1u ? 1 : 0);
        const int sz = int(resolution) + (axis == 2u ? 1 : 0);
        const int x0 = max(0, int(floor(
            (lower.x - minimum.x) * inverse_spacing - offset.x)));
        const int y0 = max(0, int(floor(
            (lower.y - minimum.y) * inverse_spacing - offset.y)));
        const int z0 = max(0, int(floor(
            (lower.z - minimum.z) * inverse_spacing - offset.z)));
        const int x1 = min(sx - 1, int(floor(
            (upper.x - minimum.x) * inverse_spacing - offset.x)));
        const int y1 = min(sy - 1, int(floor(
            (upper.y - minimum.y) * inverse_spacing - offset.y)));
        const int z1 = min(sz - 1, int(floor(
            (upper.z - minimum.z) * inverse_spacing - offset.z)));
        for (int z = z0; z <= z1; ++z)
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    const float3 point = minimum +
                        (float3(float(x), float(y), float(z)) + offset) *
                            spacing;
                    const float distance_value = distance(
                        point, pm_closest_point_triangle(point, a, b, c));
                    if (distance_value > radius) continue;
                    const uint face = pm_smoke_face_index(
                        axis, x, y, z, resolution, vertical);
                    const float aperture = clamp(
                        distance_value / radius, 0.0f, 1.0f);
                    if (aperture <= face_boundary[face] + 1.0e-7f)
                        atomic_fetch_min_explicit(
                            face_nearest_triangle + face, global_triangle,
                            memory_order_relaxed);
                }
    }
}

kernel void pm_smoke_grid_raster_rigid(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device atomic_uint *cell_nearest_triangle [[buffer(1)]],
    device atomic_uint *face_boundary [[buffer(2)]],
    device const PMSmokeRigidRasterEntry *entries [[buffer(5)]],
    device const uint *counts [[buffer(6)]],
    device const PMHandle *rigid_ids [[buffer(7)]],
    device const PMRigidBodyState *rigid_states [[buffer(8)]],
    device const PMRigidParameters *rigid_parameters [[buffer(9)]],
    device const PMPackedVec3 *vertices [[buffer(10)]],
    device const uint *indices [[buffer(11)]],
    device const PMTriangleMeshInfo *meshes [[buffer(12)]],
    uint triangle [[thread_position_in_grid]]) {
    if (triangle >= counts[2] || constants.grid_resolution == 0u) return;
    uint global_triangle = 0u, body_index = 0u;
    float3 a = 0.0f, b = 0.0f, c = 0.0f;
    if (!pm_smoke_rigid_raster_triangle(
            triangle, entries, counts[0], counts[1], rigid_ids, rigid_states,
            rigid_parameters, vertices, indices, meshes, global_triangle,
            body_index, a, b, c))
        return;
    pm_smoke_raster_triangle_mark(
        a, b, c, global_triangle, constants.grid_resolution,
        constants.grid_vertical_resolution, constants.grid_spacing,
        pm_load(constants.grid_minimum), cell_nearest_triangle, face_boundary);
}

kernel void pm_smoke_grid_select_rigid(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device const float *face_boundary [[buffer(2)]],
    device atomic_uint *face_nearest_triangle [[buffer(3)]],
    device const PMSmokeRigidRasterEntry *entries [[buffer(5)]],
    device const uint *counts [[buffer(6)]],
    device const PMHandle *rigid_ids [[buffer(7)]],
    device const PMRigidBodyState *rigid_states [[buffer(8)]],
    device const PMRigidParameters *rigid_parameters [[buffer(9)]],
    device const PMPackedVec3 *vertices [[buffer(10)]],
    device const uint *indices [[buffer(11)]],
    device const PMTriangleMeshInfo *meshes [[buffer(12)]],
    uint triangle [[thread_position_in_grid]]) {
    if (triangle >= counts[2] || constants.grid_resolution == 0u) return;
    uint global_triangle = 0u, body_index = 0u;
    float3 a = 0.0f, b = 0.0f, c = 0.0f;
    if (!pm_smoke_rigid_raster_triangle(
            triangle, entries, counts[0], counts[1], rigid_ids, rigid_states,
            rigid_parameters, vertices, indices, meshes, global_triangle,
            body_index, a, b, c))
        return;
    pm_smoke_select_triangle_faces(
        a, b, c, global_triangle, constants.grid_resolution,
        constants.grid_vertical_resolution, constants.grid_spacing,
        pm_load(constants.grid_minimum), face_boundary,
        face_nearest_triangle);
}

kernel void pm_smoke_grid_resolve_rigid(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device const uint *cell_nearest_triangle [[buffer(1)]],
    device float *face_boundary [[buffer(2)]],
    device const uint *face_nearest_triangle [[buffer(3)]],
    device PMPackedVec3 *face_normal [[buffer(4)]],
    device const PMSmokeRigidRasterEntry *entries [[buffer(5)]],
    device const uint *counts [[buffer(6)]],
    device const PMHandle *rigid_ids [[buffer(7)]],
    device const PMRigidBodyState *rigid_states [[buffer(8)]],
    device const PMRigidParameters *rigid_parameters [[buffer(9)]],
    device const PMPackedVec3 *vertices [[buffer(10)]],
    device const uint *indices [[buffer(11)]],
    device const PMTriangleMeshInfo *meshes [[buffer(12)]],
    device PMPackedVec3 *grid_scratch [[buffer(14)]],
    uint triangle [[thread_position_in_grid]]) {
    if (triangle >= counts[2] || constants.grid_resolution == 0u) return;
    uint global_triangle = 0u, body_index = 0u;
    float3 a = 0.0f, b = 0.0f, c = 0.0f;
    if (!pm_smoke_rigid_raster_triangle(
            triangle, entries, counts[0], counts[1], rigid_ids, rigid_states,
            rigid_parameters, vertices, indices, meshes, global_triangle,
            body_index, a, b, c))
        return;
    const uint resolution = constants.grid_resolution;
    const uint vertical = constants.grid_vertical_resolution;
    const float spacing = constants.grid_spacing;
    const float inverse_spacing = 1.0f / spacing;
    const float radius = 0.55f * spacing;
    const float radius_squared = radius * radius;
    const float3 minimum = pm_load(constants.grid_minimum);
    const float3 lower = min(a, min(b, c)) - radius;
    const float3 upper = max(a, max(b, c)) + radius;
    const int x0 = max(0, int(floor(
        (lower.x - minimum.x) * inverse_spacing)));
    const int y0 = max(0, int(floor(
        (lower.y - minimum.y) * inverse_spacing)));
    const int z0 = max(0, int(floor(
        (lower.z - minimum.z) * inverse_spacing)));
    const int x1 = min(int(resolution) - 1, int(floor(
        (upper.x - minimum.x) * inverse_spacing)));
    const int y1 = min(int(vertical) - 1, int(floor(
        (upper.y - minimum.y) * inverse_spacing)));
    const int z1 = min(int(resolution) - 1, int(floor(
        (upper.z - minimum.z) * inverse_spacing)));
    device const PMRigidBodyState &state = rigid_states[body_index];
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const uint cell = uint(x) + resolution *
                    (uint(y) + vertical * uint(z));
                if (cell_nearest_triangle[cell] != global_triangle) continue;
                const float3 point = minimum +
                    (float3(float(x), float(y), float(z)) + 0.5f) * spacing;
                const float3 closest = pm_closest_point_triangle(point, a, b, c);
                const float3 delta = point - closest;
                if (dot(delta, delta) > radius_squared) continue;
                grid_scratch[cell] = pm_store(
                    pm_load(state.linear_velocity) +
                    cross(pm_load(state.angular_velocity),
                          closest - pm_load(state.position)));
            }
    const uint face_total = pm_smoke_face_total(resolution, vertical);
    const float3 normal = pm_normalized_or(
        cross(b - a, c - a), float3(0.0f, 1.0f, 0.0f));
    for (uint axis = 0u; axis < 3u; ++axis) {
        float3 offset = 0.5f;
        if (axis == 0u) offset.x = 0.0f;
        if (axis == 1u) offset.y = 0.0f;
        if (axis == 2u) offset.z = 0.0f;
        const int sx = int(resolution) + (axis == 0u ? 1 : 0);
        const int sy = int(vertical) + (axis == 1u ? 1 : 0);
        const int sz = int(resolution) + (axis == 2u ? 1 : 0);
        const int fx0 = max(0, int(floor(
            (lower.x - minimum.x) * inverse_spacing - offset.x)));
        const int fy0 = max(0, int(floor(
            (lower.y - minimum.y) * inverse_spacing - offset.y)));
        const int fz0 = max(0, int(floor(
            (lower.z - minimum.z) * inverse_spacing - offset.z)));
        const int fx1 = min(sx - 1, int(floor(
            (upper.x - minimum.x) * inverse_spacing - offset.x)));
        const int fy1 = min(sy - 1, int(floor(
            (upper.y - minimum.y) * inverse_spacing - offset.y)));
        const int fz1 = min(sz - 1, int(floor(
            (upper.z - minimum.z) * inverse_spacing - offset.z)));
        for (int z = fz0; z <= fz1; ++z)
            for (int y = fy0; y <= fy1; ++y)
                for (int x = fx0; x <= fx1; ++x) {
                    const uint face = pm_smoke_face_index(
                        axis, x, y, z, resolution, vertical);
                    if (face_nearest_triangle[face] != global_triangle) continue;
                    const float3 point = minimum +
                        (float3(float(x), float(y), float(z)) + offset) * spacing;
                    const float3 closest = pm_closest_point_triangle(point, a, b, c);
                    const float3 wall_velocity =
                        pm_load(state.linear_velocity) +
                        cross(pm_load(state.angular_velocity),
                              closest - pm_load(state.position));
                    face_boundary[face_total + face] =
                        axis == 0u ? wall_velocity.x
                        : axis == 1u ? wall_velocity.y
                                     : wall_velocity.z;
                    face_normal[face] = pm_store(normal);
                }
    }
}

kernel void pm_smoke_grid_merge_obstacles(
    constant PMSmokeConstants &constants [[buffer(0)]],
    device const uint *cell_nearest_triangle [[buffer(1)]],
    device PMPackedVec3 *grid_velocity [[buffer(13)]],
    device const PMPackedVec3 *grid_scratch [[buffer(14)]],
    device uint *grid_solid [[buffer(15)]],
    uint cell [[thread_position_in_grid]]) {
    const uint cell_count = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    if (cell >= cell_count || cell_nearest_triangle[cell] == 0xffffffffu)
        return;
    grid_solid[cell] = 1u;
    grid_velocity[cell] = grid_scratch[cell];
}

kernel void pm_smoke_emit(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device float *ages [[buffer(3)]],
    device float *densities [[buffer(4)]],
    device float *pressures [[buffer(5)]],
    device PMPackedVec3 *vorticities [[buffer(6)]],
    device float *thermal_lift [[buffer(7)]],
    device PMSmokeMetadata &metadata [[buffer(8)]],
    constant PMSmokeConstants &constants [[buffer(9)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    const uint spawn_count = min(constants.command,
                                 constants.capacity);
    const uint first_slot = metadata.next_particle;
    const ulong first_serial = metadata.emitted;
    const uint thread_count = threads_per_group.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint spawn = thread_index; spawn < spawn_count;
         spawn += thread_count) {
        const uint slot = (first_slot + spawn) % constants.capacity;
        const uint serial = uint(first_serial + ulong(spawn));
        const float y =
            (pm_hash_unit(serial ^ 0x132aef41u) * 2.0f - 1.0f) *
            constants.emitter_half_extents.x;
        const float z =
            (pm_hash_unit(serial ^ 0xa385c9d3u) * 2.0f - 1.0f) *
            constants.emitter_half_extents.y;
        const PMPackedVec3 point = pm_store(
            pm_load(constants.emitter_center) + float3(0.0f, y, z));
        positions[slot] = point;
        previous[slot] = point;
        velocities[slot] = constants.initial_velocity;
        ages[slot] = 0.0f;
        densities[slot] = 0.0f;
        pressures[slot] = 0.0f;
        vorticities[slot] = {0.0f, 0.0f, 0.0f};
        thermal_lift[slot] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (thread_index == 0u) {
        metadata.count = min(constants.capacity,
                             metadata.count + spawn_count);
        metadata.revision += spawn_count;
        metadata.next_particle =
            (first_slot + spawn_count) % constants.capacity;
        metadata.emitted += ulong(spawn_count);
    }
}

kernel void pm_smoke_particle_density(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const float *ages [[buffer(2)]],
    device float *densities [[buffer(3)]],
    device float *pressures [[buffer(4)]],
    device const PMSmokeMetadata &metadata [[buffer(6)]],
    constant PMSmokeConstants &constants [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= metadata.count || constants.command != 2u ||
        constants.grid_resolution != 0u)
        return;
    if (ages[index] >= constants.lifetime) {
        densities[index] = 0.0f;
        pressures[index] = 0.0f;
        return;
    }
    const float support = 3.0f * constants.particle_radius;
    const float support_squared = support * support;
    const float3 position = pm_load(positions[index]);
    float density = 0.0f;
    for (uint other = 0u; other < metadata.count; ++other) {
        if (ages[other] >= constants.lifetime) continue;
        const float3 delta = position - pm_load(positions[other]);
        const float squared = dot(delta, delta);
        if (squared >= support_squared) continue;
        const float q = 1.0f - sqrt(max(squared, 0.0f)) / support;
        density += q * q * q;
    }
    densities[index] = density;
    const float crowding = constants.pressure_stiffness * max(
        density / constants.rest_number_density - 1.0f, 0.0f);
    pressures[index] = min(
        100.0f,
        crowding + pressures[index] * exp(-constants.timestep / 0.05f));
}

kernel void pm_smoke_particle_vorticity(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const PMPackedVec3 *velocities [[buffer(1)]],
    device const float *ages [[buffer(2)]],
    device const float *densities [[buffer(3)]],
    device PMPackedVec3 *vorticities [[buffer(5)]],
    device const PMSmokeMetadata &metadata [[buffer(6)]],
    constant PMSmokeConstants &constants [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= metadata.count || constants.command != 2u ||
        constants.grid_resolution != 0u)
        return;
    float3 curl = 0.0f;
    if (ages[index] < constants.lifetime &&
        constants.vorticity_confinement > 0.0f) {
        const float support = 3.0f * constants.particle_radius;
        const float support_squared = support * support;
        const float3 position = pm_load(positions[index]);
        const float3 velocity = pm_load(velocities[index]);
        for (uint other = 0u; other < metadata.count; ++other) {
            if (other == index || ages[other] >= constants.lifetime) continue;
            const float3 delta = position - pm_load(positions[other]);
            const float squared = dot(delta, delta);
            if (squared >= support_squared || squared < 1.0e-12f) continue;
            const float distance_value = sqrt(squared);
            const float q = 1.0f - distance_value / support;
            const float pair_density = max(
                1.0f, sqrt(densities[index] * densities[other]));
            const float3 gradient = delta *
                (-3.0f * q * q /
                 (support * distance_value * pair_density));
            curl += cross(pm_load(velocities[other]) - velocity, gradient);
        }
    }
    vorticities[index] = pm_store(curl);
}

kernel void pm_smoke_particle_forces(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const PMPackedVec3 *velocities [[buffer(1)]],
    device const float *ages [[buffer(2)]],
    device const float *densities [[buffer(3)]],
    device const float *pressures [[buffer(4)]],
    device const PMPackedVec3 *vorticities [[buffer(5)]],
    device const PMSmokeMetadata &metadata [[buffer(6)]],
    constant PMSmokeConstants &constants [[buffer(7)]],
    device PMPackedVec3 *particle_scratch [[buffer(16)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= metadata.count || constants.command != 2u ||
        constants.grid_resolution != 0u)
        return;
    float3 acceleration = 0.0f;
    float3 confinement_gradient = 0.0f;
    if (ages[index] < constants.lifetime) {
        const float support = 3.0f * constants.particle_radius;
        const float support_squared = support * support;
        const float3 position = pm_load(positions[index]);
        const float3 velocity = pm_load(velocities[index]);
        const float vorticity = length(pm_load(vorticities[index]));
        for (uint other = 0u; other < metadata.count; ++other) {
            if (other == index || ages[other] >= constants.lifetime) continue;
            const float3 delta = position - pm_load(positions[other]);
            const float squared = dot(delta, delta);
            if (squared >= support_squared || squared < 1.0e-12f) continue;
            const float distance_value = sqrt(squared);
            const float q = 1.0f - distance_value / support;
            const float pair_density = max(
                1.0f, sqrt(densities[index] * densities[other]));
            const float pressure =
                0.5f * (pressures[index] + pressures[other]);
            acceleration += delta *
                (3.0f * pressure * q * q /
                 (support * distance_value * pair_density));
            acceleration += (pm_load(velocities[other]) - velocity) *
                (constants.viscosity * q * q / pair_density);
            if (constants.vorticity_confinement > 0.0f) {
                const float3 gradient = delta *
                    (-3.0f * q * q /
                     (support * distance_value * pair_density));
                confinement_gradient += gradient *
                    (length(pm_load(vorticities[other])) - vorticity);
            }
        }
        if (dot(confinement_gradient, confinement_gradient) > 1.0e-10f)
            acceleration += cross(normalize(confinement_gradient),
                                  pm_load(vorticities[index])) *
                            (constants.vorticity_confinement * support);
    }
    particle_scratch[index] = pm_store(pm_limit(acceleration, 50.0f));
}

kernel void pm_smoke_particle_integrate(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *velocities [[buffer(1)]],
    device float *ages [[buffer(2)]],
    device float *densities [[buffer(3)]],
    device float *pressures [[buffer(4)]],
    device const PMSmokeMetadata &metadata [[buffer(6)]],
    constant PMSmokeConstants &constants [[buffer(7)]],
    device float *thermal_lift [[buffer(8)]],
    device PMPackedVec3 *particle_scratch [[buffer(16)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= metadata.count || constants.command != 2u ||
        constants.grid_resolution != 0u)
        return;
    if (ages[index] >= constants.lifetime) {
        densities[index] = 0.0f;
        pressures[index] = 0.0f;
        return;
    }
    const float3 gravity = pm_load(constants.gravity);
    const float3 up = dot(gravity, gravity) > 1.0e-12f
        ? -normalize(gravity)
        : float3(0.0f, 1.0f, 0.0f);
    const float3 old_position = pm_load(positions[index]);
    float3 velocity = pm_load(velocities[index]) +
                      pm_load(particle_scratch[index]) * constants.timestep;
    velocity += up * ((constants.buoyancy + thermal_lift[index]) *
                      constants.timestep);
    const float response =
        1.0f - exp(-constants.response * constants.timestep);
    velocity += (pm_load(constants.wind) - velocity) * response;
    velocity = pm_limit(velocity, constants.maximum_speed);
    velocities[index] = pm_store(velocity);
    particle_scratch[index] = pm_store(old_position);
    positions[index] = pm_store(old_position + velocity * constants.timestep);
    ages[index] += constants.timestep;
    thermal_lift[index] *= exp(-constants.timestep / 3.0f);
}

kernel void pm_smoke_grid_advect_particles(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *velocities [[buffer(1)]],
    device float *ages [[buffer(2)]],
    device float *densities [[buffer(3)]],
    device float *pressures [[buffer(4)]],
    device PMPackedVec3 *vorticities [[buffer(5)]],
    device const PMSmokeMetadata &metadata [[buffer(6)]],
    constant PMSmokeConstants &constants [[buffer(7)]],
    device float *thermal_lift [[buffer(8)]],
    device const float *grid_pressure [[buffer(10)]],
    device const float *grid_density [[buffer(11)]],
    device const PMPackedVec3 *grid_vorticity [[buffer(14)]],
    device PMPackedVec3 *particle_scratch [[buffer(16)]],
    device const float *grid_face_velocity [[buffer(28)]],
    uint particle [[thread_position_in_grid]]) {
    if (particle >= metadata.count || constants.command != 2u ||
        constants.grid_resolution == 0u ||
        constants.grid_spacing <= 0.0f ||
        ages[particle] >= constants.lifetime)
        return;
    const uint resolution = constants.grid_resolution;
    const uint vertical = constants.grid_vertical_resolution;
    const float spacing = constants.grid_spacing;
    const float3 minimum = pm_load(constants.grid_minimum);
    const float3 point = pm_load(positions[particle]);
    particle_scratch[particle] = pm_store(point);
    float3 velocity = pm_load(velocities[particle]);
    const float3 local = (point - minimum) / spacing;
    const bool inside = all(local >= 0.0f) &&
        local.x < float(resolution) && local.y < float(vertical) &&
        local.z < float(resolution);
    if (inside && ages[particle] > 0.0f) {
        const float3 first = pm_smoke_sample_face_velocity(
            grid_face_velocity, point, minimum, spacing, resolution,
            vertical);
        velocity = pm_smoke_sample_face_velocity(
            grid_face_velocity,
            point + first * (0.5f * constants.timestep), minimum, spacing,
            resolution, vertical);
        densities[particle] = pm_smoke_sample_cell_scalar(
            grid_density, point, minimum, spacing, resolution, vertical) *
            constants.rest_number_density;
        pressures[particle] = pm_smoke_sample_cell_scalar(
            grid_pressure, point, minimum, spacing, resolution, vertical);
        vorticities[particle] = pm_store(pm_smoke_sample_cell_vector(
            grid_vorticity, point, minimum, spacing, resolution, vertical));
    }
    velocity = pm_limit(velocity, constants.maximum_speed);
    velocities[particle] = pm_store(velocity);
    positions[particle] = pm_store(point + velocity * constants.timestep);
    ages[particle] += constants.timestep;
    thermal_lift[particle] *= exp(-constants.timestep / 3.0f);
}

kernel void pm_smoke_step_serial(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *velocities [[buffer(1)]],
    device float *ages [[buffer(2)]],
    device float *densities [[buffer(3)]],
    device float *pressures [[buffer(4)]],
    device PMPackedVec3 *vorticities [[buffer(5)]],
    device PMSmokeMetadata &metadata [[buffer(6)]],
    constant PMSmokeConstants &constants [[buffer(7)]],
    device float *thermal_lift [[buffer(8)]],
    device PMPackedVec3 *grid_velocity [[buffer(9)]],
    device float *grid_pressure [[buffer(10)]],
    device float *grid_density [[buffer(11)]],
    device float *grid_temperature [[buffer(12)]],
    device uint *grid_solid [[buffer(13)]],
    device PMPackedVec3 *grid_vorticity [[buffer(14)]],
    device float *grid_divergence [[buffer(15)]],
    device PMPackedVec3 *particle_scratch [[buffer(16)]],
    device PMPackedVec3 *grid_scratch [[buffer(17)]],
    device float *grid_pressure_relative_residual [[buffer(18)]],
    device atomic_uint *grid_pressure_state [[buffer(19)]],
    device const uint *grid_rigid_counts [[buffer(20)]],
    device const PMHandle *rigid_ids [[buffer(21)]],
    device const PMRigidBodyState *rigid_states [[buffer(22)]],
    device const PMRigidParameters *rigid_parameters [[buffer(23)]],
    device const PMPackedVec3 *rigid_vertices [[buffer(24)]],
    device const uint *rigid_indices [[buffer(25)]],
    device const PMTriangleMeshInfo *rigid_meshes [[buffer(26)]],
    device uint *grid_deformable_solid [[buffer(27)]],
    device float *grid_face_velocity [[buffer(28)]],
    device float *grid_face_advection [[buffer(29)]],
    device float *grid_face_boundary [[buffer(30)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    if (constants.command != 2u && constants.command != 4u)
        grid_pressure_relative_residual[0] = 0.0f;
    if (constants.grid_resolution == 0u || constants.grid_spacing <= 0.0f) {
    if (constants.command == 1u) return;
    const float support = 3.0f * constants.particle_radius;
    const float support_squared = support * support;
    for (uint index = 0; index < metadata.count; ++index) {
        if (ages[index] >= constants.lifetime) {
            densities[index] = 0.0f;
            pressures[index] = 0.0f;
            continue;
        }
        float density = 0.0f;
        for (uint other = 0; other < metadata.count; ++other) {
            if (ages[other] >= constants.lifetime) continue;
            const float squared = dot(pm_load(positions[index]) -
                                          pm_load(positions[other]),
                                      pm_load(positions[index]) -
                                          pm_load(positions[other]));
            if (squared >= support_squared) continue;
            const float q = 1.0f - sqrt(max(squared, 0.0f)) / support;
            density += q * q * q;
        }
        densities[index] = density;
        const float crowding = constants.pressure_stiffness * max(
            density / constants.rest_number_density - 1.0f, 0.0f);
        pressures[index] = min(
            100.0f,
            crowding + pressures[index] * exp(-constants.timestep / 0.05f));
    }
    for (uint index = 0; index < metadata.count; ++index) {
        float3 curl = 0.0f;
        if (ages[index] < constants.lifetime &&
            constants.vorticity_confinement > 0.0f) {
            for (uint other = 0; other < metadata.count; ++other) {
                if (other == index || ages[other] >= constants.lifetime)
                    continue;
                const float3 delta = pm_load(positions[index]) -
                                     pm_load(positions[other]);
                const float squared = dot(delta, delta);
                if (squared >= support_squared || squared < 1.0e-12f)
                    continue;
                const float distance_value = sqrt(squared);
                const float q = 1.0f - distance_value / support;
                const float pair_density = max(
                    1.0f, sqrt(densities[index] * densities[other]));
                const float3 gradient = delta *
                    (-3.0f * q * q /
                     (support * distance_value * pair_density));
                curl += cross(pm_load(velocities[other]) -
                                  pm_load(velocities[index]),
                              gradient);
            }
        }
        vorticities[index] = pm_store(curl);
    }
    for (uint index = 0; index < metadata.count; ++index) {
        float3 acceleration = 0.0f;
        float3 confinement_gradient = 0.0f;
        if (ages[index] < constants.lifetime) {
            for (uint other = 0; other < metadata.count; ++other) {
                if (other == index || ages[other] >= constants.lifetime)
                    continue;
                const float3 delta = pm_load(positions[index]) -
                                     pm_load(positions[other]);
                const float squared = dot(delta, delta);
                if (squared >= support_squared || squared < 1.0e-12f)
                    continue;
                const float distance_value = sqrt(squared);
                const float q = 1.0f - distance_value / support;
                const float pair_density = max(
                    1.0f, sqrt(densities[index] * densities[other]));
                const float pressure =
                    0.5f * (pressures[index] + pressures[other]);
                acceleration += delta *
                    (3.0f * pressure * q * q /
                     (support * distance_value * pair_density));
                acceleration +=
                    (pm_load(velocities[other]) -
                     pm_load(velocities[index])) *
                    (constants.viscosity * q * q / pair_density);
                if (constants.vorticity_confinement > 0.0f) {
                    const float3 gradient = delta *
                        (-3.0f * q * q /
                         (support * distance_value * pair_density));
                    confinement_gradient += gradient *
                        (length(pm_load(vorticities[other])) -
                         length(pm_load(vorticities[index])));
                }
            }
            if (dot(confinement_gradient, confinement_gradient) > 1.0e-10f)
                acceleration += cross(normalize(confinement_gradient),
                                      pm_load(vorticities[index])) *
                                (constants.vorticity_confinement * support);
        }
        particle_scratch[index] = pm_store(pm_limit(acceleration, 50.0f));
    }
    const float3 wind = pm_load(constants.wind);
    const float3 gravity = pm_load(constants.gravity);
    const float3 up = dot(gravity, gravity) > 1.0e-12f
        ? -normalize(gravity)
        : float3(0.0f, 1.0f, 0.0f);
    for (uint index = 0; index < metadata.count; ++index) {
        if (ages[index] >= constants.lifetime) {
            densities[index] = 0.0f;
            pressures[index] = 0.0f;
            continue;
        }
        const float3 old_position = pm_load(positions[index]);
        float3 velocity = pm_load(velocities[index]) +
                          pm_load(particle_scratch[index]) * constants.timestep;
        velocity += up *
            ((constants.buoyancy + thermal_lift[index]) *
             constants.timestep);
        const float response =
            1.0f - exp(-constants.response * constants.timestep);
        velocity += (wind - velocity) * response;
        velocity = pm_limit(velocity, constants.maximum_speed);
        velocities[index] = pm_store(velocity);
        particle_scratch[index] = pm_store(old_position);
        positions[index] = pm_store(old_position +
                                    velocity * constants.timestep);
        ages[index] += constants.timestep;
        thermal_lift[index] *= exp(-constants.timestep / 3.0f);
    }
    return;
    }
    {
    const uint resolution = constants.grid_resolution;
    const uint vertical = constants.grid_vertical_resolution;
    const uint face_total = pm_smoke_face_total(resolution, vertical);
    device float *predicted_faces = grid_face_advection;
    device float *reverse_faces = grid_face_advection + face_total;
    device float *face_open = grid_face_boundary;
    device float *face_wall = grid_face_boundary + face_total;
    const float3 minimum = pm_load(constants.grid_minimum);
    const float spacing = constants.grid_spacing;
    const float inverse_spacing = 1.0f / spacing;
    const float spacing_squared = spacing * spacing;

    if (constants.command != 2u && constants.command != 4u) {

    // Particle density and heat were generated, stable-sorted by cell, and
    // reduced with local 64-bit arithmetic before this ordered grid phase.

    // Triangle obstacles and their stable cell/face winners were rasterized
    // and resolved in parallel before this ordered grid phase.

    // Inflow-normal domain faces are pressure walls. Other far-field faces
    // retain p=0 outside, matching CUDA's open-boundary operator.
    const float3 wind = pm_load(constants.wind);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint count = pm_smoke_face_count(axis, resolution, vertical);
        const int extent = axis == 0u ? int(resolution)
                           : axis == 1u ? int(vertical)
                                        : int(resolution);
        const float component = axis == 0u ? wind.x
                              : axis == 1u ? wind.y
                                           : wind.z;
        for (uint local = 0u; local < count; ++local) {
            int x = 0, y = 0, z = 0;
            pm_smoke_face_coordinates(axis, local, resolution, vertical,
                                      x, y, z);
            const int coordinate = axis == 0u ? x : axis == 1u ? y : z;
            if (coordinate != 0 && coordinate != extent) continue;
            const float outward = coordinate == 0 ? -1.0f : 1.0f;
            if (component * outward < 0.0f) {
                const uint face = pm_smoke_face_offset(
                    axis, resolution, vertical) + local;
                face_open[face] = 0.0f;
                face_wall[face] = component;
            }
        }
    }

    // RK2 MacCormack self-advection on the staggered velocity components.
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint count = pm_smoke_face_count(axis, resolution, vertical);
        const uint offset = pm_smoke_face_offset(axis, resolution, vertical);
        for (uint local = 0u; local < count; ++local) {
            int x = 0, y = 0, z = 0;
            pm_smoke_face_coordinates(axis, local, resolution, vertical,
                                      x, y, z);
            const float3 point = pm_smoke_face_position(
                axis, x, y, z, minimum, spacing);
            const float3 first = pm_smoke_sample_face_velocity(
                grid_face_velocity, point, minimum, spacing, resolution,
                vertical);
            const float3 midpoint =
                point - first * (0.5f * constants.timestep);
            const float3 flow = pm_smoke_sample_face_velocity(
                grid_face_velocity, midpoint, minimum, spacing, resolution,
                vertical);
            const float3 departure = point - flow * constants.timestep;
            predicted_faces[offset + local] =
                pm_smoke_sample_face_component(
                    grid_face_velocity, axis, departure, minimum, spacing,
                    resolution, vertical);
        }
    }
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint count = pm_smoke_face_count(axis, resolution, vertical);
        const uint offset = pm_smoke_face_offset(axis, resolution, vertical);
        for (uint local = 0u; local < count; ++local) {
            int x = 0, y = 0, z = 0;
            pm_smoke_face_coordinates(axis, local, resolution, vertical,
                                      x, y, z);
            const float3 point = pm_smoke_face_position(
                axis, x, y, z, minimum, spacing);
            const float3 first = pm_smoke_sample_face_velocity(
                predicted_faces, point, minimum, spacing, resolution,
                vertical);
            const float3 midpoint =
                point + first * (0.5f * constants.timestep);
            const float3 flow = pm_smoke_sample_face_velocity(
                predicted_faces, midpoint, minimum, spacing, resolution,
                vertical);
            const float3 reverse_departure =
                point + flow * constants.timestep;
            reverse_faces[offset + local] =
                pm_smoke_sample_face_component(
                    predicted_faces, axis, reverse_departure, minimum,
                    spacing, resolution, vertical);

            const float3 original_first = pm_smoke_sample_face_velocity(
                grid_face_velocity, point, minimum, spacing, resolution,
                vertical);
            const float3 original_midpoint =
                point - original_first * (0.5f * constants.timestep);
            const float3 original_flow = pm_smoke_sample_face_velocity(
                grid_face_velocity, original_midpoint, minimum, spacing,
                resolution, vertical);
            const float3 departure =
                point - original_flow * constants.timestep;
            float3 sample_position = (departure - minimum) / spacing;
            int sx = int(resolution), sy = int(vertical), sz = int(resolution);
            if (axis == 0u) {
                sample_position.y -= 0.5f;
                sample_position.z -= 0.5f;
                ++sx;
            }
            if (axis == 1u) {
                sample_position.x -= 0.5f;
                sample_position.z -= 0.5f;
                ++sy;
            }
            if (axis == 2u) {
                sample_position.x -= 0.5f;
                sample_position.y -= 0.5f;
                ++sz;
            }
            const int3 lower = int3(floor(sample_position));
            float low = INFINITY;
            float high = -INFINITY;
            for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const int xx = clamp(lower.x + dx, 0, sx - 1);
                        const int yy = clamp(lower.y + dy, 0, sy - 1);
                        const int zz = clamp(lower.z + dz, 0, sz - 1);
                        const float sample = grid_face_velocity[
                            pm_smoke_face_index(axis, xx, yy, zz,
                                                resolution, vertical)];
                        low = min(low, sample);
                        high = max(high, sample);
                    }
            predicted_faces[offset + local] = clamp(
                predicted_faces[offset + local] +
                    0.5f * (grid_face_velocity[offset + local] -
                            reverse_faces[offset + local]),
                low, high);
        }
    }

    // Cell diagnostics from the advected face field. Strain temporarily uses
    // grid_divergence until the pressure RHS is assembled below.
    for (uint z = 0u; z < resolution; ++z)
        for (uint y = 0u; y < vertical; ++y)
            for (uint x = 0u; x < resolution; ++x) {
                const uint cell = x + resolution * (y + vertical * z);
                const float3 point = minimum +
                    (float3(float(x), float(y), float(z)) + 0.5f) * spacing;
                const float3 vxm = pm_smoke_sample_face_velocity(
                    predicted_faces, point - float3(spacing, 0.0f, 0.0f),
                    minimum, spacing, resolution, vertical);
                const float3 vxp = pm_smoke_sample_face_velocity(
                    predicted_faces, point + float3(spacing, 0.0f, 0.0f),
                    minimum, spacing, resolution, vertical);
                const float3 vym = pm_smoke_sample_face_velocity(
                    predicted_faces, point - float3(0.0f, spacing, 0.0f),
                    minimum, spacing, resolution, vertical);
                const float3 vyp = pm_smoke_sample_face_velocity(
                    predicted_faces, point + float3(0.0f, spacing, 0.0f),
                    minimum, spacing, resolution, vertical);
                const float3 vzm = pm_smoke_sample_face_velocity(
                    predicted_faces, point - float3(0.0f, 0.0f, spacing),
                    minimum, spacing, resolution, vertical);
                const float3 vzp = pm_smoke_sample_face_velocity(
                    predicted_faces, point + float3(0.0f, 0.0f, spacing),
                    minimum, spacing, resolution, vertical);
                const float3 dx = (vxp - vxm) * (0.5f * inverse_spacing);
                const float3 dy = (vyp - vym) * (0.5f * inverse_spacing);
                const float3 dz = (vzp - vzm) * (0.5f * inverse_spacing);
                grid_vorticity[cell] = pm_store(float3(
                    dy.z - dz.y, dz.x - dx.z, dx.y - dy.x));
                const float sxy = 0.5f * (dx.y + dy.x);
                const float sxz = 0.5f * (dx.z + dz.x);
                const float syz = 0.5f * (dy.z + dz.y);
                const float strain = sqrt(max(
                    0.0f, 2.0f * (dx.x * dx.x + dy.y * dy.y +
                                   dz.z * dz.z +
                                   2.0f * (sxy * sxy + sxz * sxz +
                                           syz * syz))));
                grid_divergence[cell] = strain;
            }

    const float3 gravity = pm_load(constants.gravity);
    const float3 up = dot(gravity, gravity) > 1.0e-12f
        ? -normalize(gravity)
        : float3(0.0f, 1.0f, 0.0f);
    for (uint z = 0u; z < resolution; ++z)
        for (uint y = 0u; y < vertical; ++y)
            for (uint x = 0u; x < resolution; ++x) {
                const uint cell = x + resolution * (y + vertical * z);
                const uint xm = (x == 0u ? x : x - 1u) +
                    resolution * (y + vertical * z);
                const uint xp = min(x + 1u, resolution - 1u) +
                    resolution * (y + vertical * z);
                const uint ym = x + resolution *
                    ((y == 0u ? y : y - 1u) + vertical * z);
                const uint yp = x + resolution *
                    (min(y + 1u, vertical - 1u) + vertical * z);
                const uint zm = x + resolution *
                    (y + vertical * (z == 0u ? z : z - 1u));
                const uint zp = x + resolution *
                    (y + vertical * min(z + 1u, resolution - 1u));
                const float3 magnitude_gradient = float3(
                    length(pm_load(grid_vorticity[xp])) -
                        length(pm_load(grid_vorticity[xm])),
                    length(pm_load(grid_vorticity[yp])) -
                        length(pm_load(grid_vorticity[ym])),
                    length(pm_load(grid_vorticity[zp])) -
                        length(pm_load(grid_vorticity[zm]))) *
                    (0.5f * inverse_spacing);
                float3 confinement = 0.0f;
                if (dot(magnitude_gradient, magnitude_gradient) > 1.0e-12f)
                    confinement = cross(normalize(magnitude_gradient),
                                        pm_load(grid_vorticity[cell])) *
                        (constants.vorticity_confinement * spacing);
                const float heat = grid_density[cell] > 1.0e-5f
                    ? clamp(grid_temperature[cell] / grid_density[cell],
                            0.0f, constants.maximum_speed)
                    : 0.0f;
                const float3 buoyancy = up *
                    (constants.buoyancy * grid_density[cell] *
                         constants.rest_number_density +
                     2.0f * heat);
                grid_scratch[cell] = pm_store(confinement + buoyancy);
            }

    // Apply viscosity, LES, confinement, buoyancy, domain conditions, and
    // cut-face wall blending. reverse_faces is now a force-output scratch.
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint count = pm_smoke_face_count(axis, resolution, vertical);
        const uint offset = pm_smoke_face_offset(axis, resolution, vertical);
        const int sx = int(resolution) + (axis == 0u ? 1 : 0);
        const int sy = int(vertical) + (axis == 1u ? 1 : 0);
        const int sz = int(resolution) + (axis == 2u ? 1 : 0);
        for (uint local = 0u; local < count; ++local) {
            int x = 0, y = 0, z = 0;
            pm_smoke_face_coordinates(axis, local, resolution, vertical,
                                      x, y, z);
            const float center = predicted_faces[offset + local];
            const int xm = max(0, x - 1), xp = min(sx - 1, x + 1);
            const int ym = max(0, y - 1), yp = min(sy - 1, y + 1);
            const int zm = max(0, z - 1), zp = min(sz - 1, z + 1);
            const float laplacian =
                (predicted_faces[pm_smoke_face_index(
                     axis, xm, y, z, resolution, vertical)] +
                 predicted_faces[pm_smoke_face_index(
                     axis, xp, y, z, resolution, vertical)] +
                 predicted_faces[pm_smoke_face_index(
                     axis, x, ym, z, resolution, vertical)] +
                 predicted_faces[pm_smoke_face_index(
                     axis, x, yp, z, resolution, vertical)] +
                 predicted_faces[pm_smoke_face_index(
                     axis, x, y, zm, resolution, vertical)] +
                 predicted_faces[pm_smoke_face_index(
                     axis, x, y, zp, resolution, vertical)] -
                 6.0f * center) /
                spacing_squared;
            const float3 point = pm_smoke_face_position(
                axis, x, y, z, minimum, spacing);
            const float strain = pm_smoke_sample_cell_scalar(
                grid_divergence, point, minimum, spacing, resolution,
                vertical);
            const float viscosity = constants.grid_kinematic_viscosity +
                constants.grid_les_coefficient *
                    constants.grid_les_coefficient * spacing_squared * strain;
            const float3 body = pm_smoke_sample_cell_vector(
                grid_scratch, point, minimum, spacing, resolution, vertical);
            const float component =
                axis == 0u ? body.x : axis == 1u ? body.y : body.z;
            reverse_faces[offset + local] = clamp(
                center + constants.timestep *
                             (viscosity * laplacian + component),
                -constants.maximum_speed, constants.maximum_speed);
        }
    }
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint count = pm_smoke_face_count(axis, resolution, vertical);
        const uint offset = pm_smoke_face_offset(axis, resolution, vertical);
        const int extent = axis == 0u ? int(resolution)
                           : axis == 1u ? int(vertical)
                                        : int(resolution);
        for (uint local = 0u; local < count; ++local) {
            int x = 0, y = 0, z = 0;
            pm_smoke_face_coordinates(axis, local, resolution, vertical,
                                      x, y, z);
            const uint face = offset + local;
            float value = reverse_faces[face];
            const int coordinate = axis == 0u ? x : axis == 1u ? y : z;
            if ((coordinate == 0 || coordinate == extent) &&
                face_open[face] > 1.0e-4f) {
                const int nx = axis == 0u
                    ? (coordinate == 0 ? 1 : extent - 1) : x;
                const int ny = axis == 1u
                    ? (coordinate == 0 ? 1 : extent - 1) : y;
                const int nz = axis == 2u
                    ? (coordinate == 0 ? 1 : extent - 1) : z;
                value = reverse_faces[pm_smoke_face_index(
                    axis, nx, ny, nz, resolution, vertical)];
            }
            const float3 point = pm_smoke_face_position(
                axis, x, y, z, minimum, spacing);
            if (abs(point.x - constants.emitter_center.x) < 1.5f * spacing &&
                abs(point.y - constants.emitter_center.y) <=
                    constants.emitter_half_extents.x + 0.5f * spacing &&
                abs(point.z - constants.emitter_center.z) <=
                    constants.emitter_half_extents.y + 0.5f * spacing)
                value = axis == 0u ? constants.initial_velocity.x
                      : axis == 1u ? constants.initial_velocity.y
                                   : constants.initial_velocity.z;
            grid_face_velocity[face] =
                face_open[face] * value +
                (1.0f - face_open[face]) * face_wall[face];
        }
    }

    float rhs_maximum = 0.0f;
    for (uint z = 0u; z < resolution; ++z)
        for (uint y = 0u; y < vertical; ++y)
            for (uint x = 0u; x < resolution; ++x) {
                const uint cell = x + resolution * (y + vertical * z);
                const float flux =
                    grid_face_velocity[pm_smoke_face_index(
                        0u, int(x) + 1, int(y), int(z), resolution,
                        vertical)] -
                    grid_face_velocity[pm_smoke_face_index(
                        0u, int(x), int(y), int(z), resolution, vertical)] +
                    grid_face_velocity[pm_smoke_face_index(
                        1u, int(x), int(y) + 1, int(z), resolution,
                        vertical)] -
                    grid_face_velocity[pm_smoke_face_index(
                        1u, int(x), int(y), int(z), resolution, vertical)] +
                    grid_face_velocity[pm_smoke_face_index(
                        2u, int(x), int(y), int(z) + 1, resolution,
                        vertical)] -
                    grid_face_velocity[pm_smoke_face_index(
                        2u, int(x), int(y), int(z), resolution, vertical)];
                grid_divergence[cell] =
                    flux * inverse_spacing /
                    max(constants.timestep, 1.0e-12f);
                rhs_maximum = max(rhs_maximum, abs(grid_divergence[cell]));
            }

    if (constants.command == 3u) {
        atomic_store_explicit(
            grid_pressure_state, as_type<uint>(rhs_maximum),
            memory_order_relaxed);
        atomic_store_explicit(
            grid_pressure_state + 1u, 0u, memory_order_relaxed);
        atomic_store_explicit(
            grid_pressure_state + 2u, 0u, memory_order_relaxed);
        return;
    }

    // CUDA-equivalent four-level aperture-weighted multigrid. The advection
    // ping-pong allocation is dead after face forces are resolved, so it is
    // reused as fixed-capacity pressure scratch rather than allocating while
    // stepping.
    const uint level0_count = resolution * vertical * resolution;
    const uint level1_n = max(2u, resolution / 2u);
    const uint level1_height = max(2u, vertical / 2u);
    const uint level2_n = max(2u, level1_n / 2u);
    const uint level2_height = max(2u, level1_height / 2u);
    const uint level3_n = max(2u, level2_n / 2u);
    const uint level3_height = max(2u, level2_height / 2u);
    const uint level1_count = level1_n * level1_height * level1_n;
    const uint level2_count = level2_n * level2_height * level2_n;
    const uint level3_count = level3_n * level3_height * level3_n;
    const uint level1_faces = pm_smoke_face_total(
        level1_n, level1_height);
    const uint level2_faces = pm_smoke_face_total(
        level2_n, level2_height);

    device float *pressure_cursor = grid_face_advection;
    device float *level0_alternate = pressure_cursor;
    pressure_cursor += level0_count;
    device float *level0_residual = pressure_cursor;
    pressure_cursor += level0_count;
    device float *level1_pressure = pressure_cursor;
    pressure_cursor += level1_count;
    device float *level1_alternate = pressure_cursor;
    pressure_cursor += level1_count;
    device float *level1_rhs = pressure_cursor;
    pressure_cursor += level1_count;
    device float *level1_residual = pressure_cursor;
    pressure_cursor += level1_count;
    device float *level1_open = pressure_cursor;
    pressure_cursor += level1_faces;
    device float *level2_pressure = pressure_cursor;
    pressure_cursor += level2_count;
    device float *level2_alternate = pressure_cursor;
    pressure_cursor += level2_count;
    device float *level2_rhs = pressure_cursor;
    pressure_cursor += level2_count;
    device float *level2_residual = pressure_cursor;
    pressure_cursor += level2_count;
    device float *level2_open = pressure_cursor;
    pressure_cursor += level2_faces;
    device float *level3_pressure = pressure_cursor;
    pressure_cursor += level3_count;
    device float *level3_alternate = pressure_cursor;
    pressure_cursor += level3_count;
    device float *level3_rhs = pressure_cursor;
    pressure_cursor += level3_count;
    device float *level3_open = pressure_cursor;

    pm_smoke_pressure_restrict_open(
        face_open, resolution, vertical, level1_open,
        level1_n, level1_height);
    pm_smoke_pressure_restrict_open(
        level1_open, level1_n, level1_height, level2_open,
        level2_n, level2_height);
    pm_smoke_pressure_restrict_open(
        level2_open, level2_n, level2_height, level3_open,
        level3_n, level3_height);

    const uint pressure_cycles = max(
        1u, (constants.grid_pressure_iterations + 4u) / 5u);
    for (uint cycle = 0u; cycle < pressure_cycles; ++cycle) {
        pm_smoke_pressure_smooth_pairs(
            resolution, vertical, spacing, grid_divergence,
            grid_pressure, level0_alternate, face_open, 1u);
        pm_smoke_pressure_residual(
            resolution, vertical, spacing, grid_divergence,
            grid_pressure, level0_residual, face_open);
        pm_smoke_pressure_restrict_residual(
            level0_residual, resolution, vertical, level1_rhs,
            level1_n, level1_height);
        pm_smoke_pressure_clear(level1_pressure, level1_count);
        pm_smoke_pressure_clear(level1_alternate, level1_count);

        pm_smoke_pressure_smooth_pairs(
            level1_n, level1_height, 2.0f * spacing, level1_rhs,
            level1_pressure, level1_alternate, level1_open, 1u);
        pm_smoke_pressure_residual(
            level1_n, level1_height, 2.0f * spacing, level1_rhs,
            level1_pressure, level1_residual, level1_open);
        pm_smoke_pressure_restrict_residual(
            level1_residual, level1_n, level1_height, level2_rhs,
            level2_n, level2_height);
        pm_smoke_pressure_clear(level2_pressure, level2_count);
        pm_smoke_pressure_clear(level2_alternate, level2_count);

        pm_smoke_pressure_smooth_pairs(
            level2_n, level2_height, 4.0f * spacing, level2_rhs,
            level2_pressure, level2_alternate, level2_open, 1u);
        pm_smoke_pressure_residual(
            level2_n, level2_height, 4.0f * spacing, level2_rhs,
            level2_pressure, level2_residual, level2_open);
        pm_smoke_pressure_restrict_residual(
            level2_residual, level2_n, level2_height, level3_rhs,
            level3_n, level3_height);
        pm_smoke_pressure_clear(level3_pressure, level3_count);
        pm_smoke_pressure_clear(level3_alternate, level3_count);

        pm_smoke_pressure_smooth_pairs(
            level3_n, level3_height, 8.0f * spacing, level3_rhs,
            level3_pressure, level3_alternate, level3_open, 6u);
        pm_smoke_pressure_prolong_add(
            level3_pressure, level3_n, level3_height,
            level2_pressure, level2_n, level2_height);
        pm_smoke_pressure_smooth_pairs(
            level2_n, level2_height, 4.0f * spacing, level2_rhs,
            level2_pressure, level2_alternate, level2_open, 1u);
        pm_smoke_pressure_prolong_add(
            level2_pressure, level2_n, level2_height,
            level1_pressure, level1_n, level1_height);
        pm_smoke_pressure_smooth_pairs(
            level1_n, level1_height, 2.0f * spacing, level1_rhs,
            level1_pressure, level1_alternate, level1_open, 1u);
        pm_smoke_pressure_prolong_add(
            level1_pressure, level1_n, level1_height,
            grid_pressure, resolution, vertical);
        pm_smoke_pressure_smooth_pairs(
            resolution, vertical, spacing, grid_divergence,
            grid_pressure, level0_alternate, face_open, 1u);

        const float maximum_residual = pm_smoke_pressure_residual(
            resolution, vertical, spacing, grid_divergence,
            grid_pressure, level0_residual, face_open);
        const float relative = rhs_maximum > 1.0e-12f
            ? maximum_residual / rhs_maximum : 0.0f;
        grid_pressure_relative_residual[0] = relative;
        if (relative <= constants.grid_pressure_tolerance) break;
    }

    }

    if (constants.command != 2u) {
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint count = pm_smoke_face_count(axis, resolution, vertical);
        const uint offset = pm_smoke_face_offset(axis, resolution, vertical);
        const int extent = axis == 0u ? int(resolution)
                           : axis == 1u ? int(vertical)
                                        : int(resolution);
        for (uint local = 0u; local < count; ++local) {
            int x = 0, y = 0, z = 0;
            pm_smoke_face_coordinates(axis, local, resolution, vertical,
                                      x, y, z);
            const uint face = offset + local;
            const int coordinate = axis == 0u ? x : axis == 1u ? y : z;
            if (face_open[face] <= 1.0e-4f) {
                grid_face_velocity[face] = face_wall[face];
                continue;
            }
            float gradient = 0.0f;
            if (coordinate == 0 || coordinate == extent) {
                int cx = x, cy = y, cz = z;
                if (axis == 0u) cx = coordinate == 0 ? 0 : int(resolution) - 1;
                if (axis == 1u) cy = coordinate == 0 ? 0 : int(vertical) - 1;
                if (axis == 2u) cz = coordinate == 0 ? 0 : int(resolution) - 1;
                const float inside = grid_pressure[
                    uint(cx) + resolution *
                        (uint(cy) + vertical * uint(cz))];
                gradient = (coordinate == 0 ? inside : -inside) *
                           inverse_spacing;
            } else {
                int lx = x, ly = y, lz = z;
                int rx = x, ry = y, rz = z;
                if (axis == 0u) { lx = x - 1; rx = x; }
                if (axis == 1u) { ly = y - 1; ry = y; }
                if (axis == 2u) { lz = z - 1; rz = z; }
                const float left = grid_pressure[
                    uint(lx) + resolution *
                        (uint(ly) + vertical * uint(lz))];
                const float right = grid_pressure[
                    uint(rx) + resolution *
                        (uint(ry) + vertical * uint(rz))];
                gradient = (right - left) * inverse_spacing;
            }
            grid_face_velocity[face] -=
                constants.timestep * face_open[face] * gradient;
        }
    }

    // Reconstruct public cell-centered fields from the projected staggered
    // faces, and expose the actual post-projection divergence.
    for (uint z = 0u; z < resolution; ++z)
        for (uint y = 0u; y < vertical; ++y)
            for (uint x = 0u; x < resolution; ++x) {
                const uint cell = x + resolution * (y + vertical * z);
                const float u0 = grid_face_velocity[pm_smoke_face_index(
                    0u, int(x), int(y), int(z), resolution, vertical)];
                const float u1 = grid_face_velocity[pm_smoke_face_index(
                    0u, int(x) + 1, int(y), int(z), resolution, vertical)];
                const float v0 = grid_face_velocity[pm_smoke_face_index(
                    1u, int(x), int(y), int(z), resolution, vertical)];
                const float v1 = grid_face_velocity[pm_smoke_face_index(
                    1u, int(x), int(y) + 1, int(z), resolution, vertical)];
                const float w0 = grid_face_velocity[pm_smoke_face_index(
                    2u, int(x), int(y), int(z), resolution, vertical)];
                const float w1 = grid_face_velocity[pm_smoke_face_index(
                    2u, int(x), int(y), int(z) + 1, resolution, vertical)];
                grid_velocity[cell] = pm_store(float3(
                    0.5f * (u0 + u1), 0.5f * (v0 + v1),
                    0.5f * (w0 + w1)));
                grid_divergence[cell] =
                    ((u1 - u0) + (v1 - v0) + (w1 - w0)) *
                    inverse_spacing;
                const float3 point = minimum +
                    (float3(float(x), float(y), float(z)) + 0.5f) * spacing;
                const float3 vxm = pm_smoke_sample_face_velocity(
                    grid_face_velocity,
                    point - float3(spacing, 0.0f, 0.0f), minimum, spacing,
                    resolution, vertical);
                const float3 vxp = pm_smoke_sample_face_velocity(
                    grid_face_velocity,
                    point + float3(spacing, 0.0f, 0.0f), minimum, spacing,
                    resolution, vertical);
                const float3 vym = pm_smoke_sample_face_velocity(
                    grid_face_velocity,
                    point - float3(0.0f, spacing, 0.0f), minimum, spacing,
                    resolution, vertical);
                const float3 vyp = pm_smoke_sample_face_velocity(
                    grid_face_velocity,
                    point + float3(0.0f, spacing, 0.0f), minimum, spacing,
                    resolution, vertical);
                const float3 vzm = pm_smoke_sample_face_velocity(
                    grid_face_velocity,
                    point - float3(0.0f, 0.0f, spacing), minimum, spacing,
                    resolution, vertical);
                const float3 vzp = pm_smoke_sample_face_velocity(
                    grid_face_velocity,
                    point + float3(0.0f, 0.0f, spacing), minimum, spacing,
                    resolution, vertical);
                const float3 dx = (vxp - vxm) * (0.5f * inverse_spacing);
                const float3 dy = (vyp - vym) * (0.5f * inverse_spacing);
                const float3 dz = (vzp - vzm) * (0.5f * inverse_spacing);
                grid_vorticity[cell] = pm_store(float3(
                    dy.z - dz.y, dz.x - dx.z, dx.y - dy.x));
            }
    }

    if (constants.command == 1u || constants.command == 4u) return;

    for (uint particle = 0u; particle < metadata.count; ++particle) {
        if (ages[particle] >= constants.lifetime) continue;
        const float3 point = pm_load(positions[particle]);
        particle_scratch[particle] = pm_store(point);
        float3 velocity = pm_load(velocities[particle]);
        const float3 local = (point - minimum) / spacing;
        const bool inside = all(local >= 0.0f) &&
            local.x < float(resolution) && local.y < float(vertical) &&
            local.z < float(resolution);
        if (inside && ages[particle] > 0.0f) {
            const float3 first = pm_smoke_sample_face_velocity(
                grid_face_velocity, point, minimum, spacing, resolution,
                vertical);
            velocity = pm_smoke_sample_face_velocity(
                grid_face_velocity,
                point + first * (0.5f * constants.timestep), minimum,
                spacing, resolution, vertical);
            densities[particle] = pm_smoke_sample_cell_scalar(
                grid_density, point, minimum, spacing, resolution, vertical) *
                constants.rest_number_density;
            pressures[particle] = pm_smoke_sample_cell_scalar(
                grid_pressure, point, minimum, spacing, resolution, vertical);
            vorticities[particle] = pm_store(pm_smoke_sample_cell_vector(
                grid_vorticity, point, minimum, spacing, resolution,
                vertical));
        }
        velocity = pm_limit(velocity, constants.maximum_speed);
        velocities[particle] = pm_store(velocity);
        positions[particle] = pm_store(
            point + velocity * constants.timestep);
        ages[particle] += constants.timestep;
        thermal_lift[particle] *= exp(-constants.timestep / 3.0f);
    }
    return;
    }
}

static void pm_integrate_deformable_node(
    device PMPackedVec3 *positions, device PMPackedVec3 *previous,
    device PMPackedVec3 *velocities, device const float *inverse_masses,
    constant PMDeformableConstants &constants, uint index) {
    if (index >= constants.count) return;
    previous[index] = positions[index];
    if (inverse_masses[index] <= 0.0f) {
        velocities[index] = {0.0f, 0.0f, 0.0f};
        return;
    }
    float3 velocity = pm_load(velocities[index]) *
        (1.0f /
         (1.0f + constants.velocity_damping * constants.timestep));
    velocity += pm_load(constants.gravity) * constants.timestep;
    velocities[index] = pm_store(velocity);
    positions[index] = pm_store(pm_load(positions[index]) +
                                velocity * constants.timestep);
}

static void pm_project_bond_node(
    device const PMPackedVec3 *positions,
    device const float *inverse_masses,
    device const PMMetalBond *bonds,
    device PMPackedVec3 *scratch,
    constant PMDeformableConstants &constants, uint node) {
    if (node >= constants.count) return;
    const float3 position = pm_load(positions[node]);
    const float self_inverse = inverse_masses[node];
    if (self_inverse <= 0.0f) {
        scratch[node] = positions[node];
        return;
    }
    float3 correction = 0.0f;
    float shortest_rest_length = INFINITY;
    uint degree = 0u;
    for (uint index = 0u; index < constants.bond_count; ++index) {
        const PMMetalBond bond = bonds[index];
        uint other = constants.count;
        if (bond.first == node) other = bond.second;
        else if (bond.second == node) other = bond.first;
        if (other >= constants.count) continue;
        ++degree;
        if (bond.active == 0u) continue;
        shortest_rest_length = min(shortest_rest_length, bond.rest_length);
        const float3 difference = position - pm_load(positions[other]);
        const float length_value = length(difference);
        if (length_value <= 1.0e-7f) continue;
        const float denominator = self_inverse + inverse_masses[other] +
            bond.compliance /
                max(constants.timestep * constants.timestep, 1.0e-12f);
        if (denominator <= 0.0f) continue;
        correction += difference *
            (-self_inverse * (length_value - bond.rest_length) /
             (denominator * length_value));
    }
    float3 proposal = correction / float(max(degree, 1u));
    if (constants.maximum_projection_fraction > 0.0f &&
        isfinite(shortest_rest_length))
        proposal = pm_limit(
            proposal,
            constants.maximum_projection_fraction * shortest_rest_length);
    scratch[node] = pm_store(position + proposal);
}

static void pm_project_cloth_neighbor_node(
    device const PMPackedVec3 *positions,
    device const float *inverse_masses,
    device const uchar *active_bonds,
    device const uint *offsets,
    device const PMDeformableNeighbor *neighbors,
    device PMPackedVec3 *scratch,
    constant PMDeformableConstants &constants, uint node) {
    if (node >= constants.count) return;
    const float3 position = pm_load(positions[node]);
    const float self_inverse = inverse_masses[node];
    if (self_inverse <= 0.0f) {
        scratch[node] = positions[node];
        return;
    }
    float3 correction = 0.0f;
    float shortest_rest_length = INFINITY;
    const uint first = offsets[node];
    const uint last = offsets[node + 1u];
    for (uint edge = first; edge < last; ++edge) {
        const PMDeformableNeighbor neighbor = neighbors[edge];
        if (neighbor.bond != 0xffffffffu &&
            active_bonds[neighbor.bond] == 0u)
            continue;
        shortest_rest_length = min(shortest_rest_length,
                                   neighbor.rest_length);
        const float3 difference =
            position - pm_load(positions[neighbor.index]);
        const float length_value = length(difference);
        if (length_value < 1.0e-7f) continue;
        const float denominator = self_inverse +
            inverse_masses[neighbor.index] +
            neighbor.compliance /
                (constants.timestep * constants.timestep);
        if (denominator <= 0.0f) continue;
        correction += difference *
            (-self_inverse * (length_value - neighbor.rest_length) /
             (denominator * length_value));
    }
    float3 proposal = correction / float(max(1u, last - first));
    if (constants.maximum_projection_fraction > 0.0f &&
        isfinite(shortest_rest_length))
        proposal = pm_limit(
            proposal,
            constants.maximum_projection_fraction * shortest_rest_length);
    scratch[node] = pm_store(position + proposal);
}

static void pm_update_cloth_damage(
    device const PMPackedVec3 *positions, device PMMetalBond *bonds,
    device uchar *active_bonds, device uchar *damage,
    device const PMPackedVec3 *rigid_impulses,
    constant PMDeformableConstants &constants, uint index) {
    if (index >= constants.bond_count) return;
    if (constants.break_strain <= 0.0f &&
        constants.impact_break_impulse <= 0.0f)
        return;
    const uint persistence = max(constants.fracture_persistence, 1u);
    device PMMetalBond &bond = bonds[index];
    if (bond.active == 0u) return;
    if (constants.impact_break_impulse > 0.0f &&
        length(pm_load(rigid_impulses[bond.first])) +
                length(pm_load(rigid_impulses[bond.second])) >
            constants.impact_break_impulse) {
        bond.active = 0u;
        active_bonds[index] = 0u;
        damage[index] = 0u;
        return;
    }
    if (constants.break_strain <= 0.0f) return;
    const float length_value = distance(pm_load(positions[bond.first]),
                                        pm_load(positions[bond.second]));
    if (isfinite(length_value) &&
        length_value >
            bond.rest_length * (1.0f + constants.break_strain)) {
        const uint next = min(persistence, uint(damage[index]) + 1u);
        damage[index] = uchar(next);
        if (next >= persistence) {
            bond.active = 0u;
            active_bonds[index] = 0u;
            damage[index] = 0u;
        }
    } else {
        damage[index] = 0u;
    }
}

static void pm_finalize_deformable_node(
    device const PMPackedVec3 *positions,
    device const PMPackedVec3 *previous,
    device PMPackedVec3 *velocities,
    device const float *inverse_masses,
    constant PMDeformableConstants &constants, uint index) {
    if (index >= constants.count) return;
    if (inverse_masses[index] <= 0.0f) {
        velocities[index] = {0.0f, 0.0f, 0.0f};
        return;
    }
    const float3 projected =
        (pm_load(positions[index]) - pm_load(previous[index])) /
        constants.timestep;
    const float response =
        clamp(constants.constraint_velocity_response, 0.0f, 1.0f);
    velocities[index] = pm_store(pm_limit(
        mix(pm_load(velocities[index]), projected, response),
        constants.maximum_speed));
}

kernel void pm_cloth_predict(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device const float *inverse_masses [[buffer(3)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    pm_integrate_deformable_node(
        positions, previous, velocities, inverse_masses, constants, index);
}

kernel void pm_cloth_damage(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device PMMetalBond *bonds [[buffer(4)]],
    device uchar *active_bonds [[buffer(5)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    device uchar *bond_damage [[buffer(9)]],
    device const PMPackedVec3 *rigid_impulses [[buffer(16)]],
    uint bond [[thread_position_in_grid]]) {
    pm_update_cloth_damage(positions, bonds, active_bonds, bond_damage,
                           rigid_impulses, constants, bond);
}

kernel void pm_cloth_project_bonds(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const float *inverse_masses [[buffer(3)]],
    device const uchar *active_bonds [[buffer(5)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    device PMPackedVec3 *scratch [[buffer(12)]],
    device const uint *offsets [[buffer(13)]],
    device const PMDeformableNeighbor *neighbors [[buffer(14)]],
    uint node [[thread_position_in_grid]]) {
    pm_project_cloth_neighbor_node(
        positions, inverse_masses, active_bonds, offsets, neighbors,
        scratch, constants, node);
}

kernel void pm_cloth_apply_bonds(
    device PMPackedVec3 *positions [[buffer(0)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    device const PMPackedVec3 *scratch [[buffer(12)]],
    uint node [[thread_position_in_grid]]) {
    if (node < constants.count) positions[node] = scratch[node];
}

kernel void pm_cloth_limit_strain_serial(
    device PMPackedVec3 *positions [[buffer(0)]],
    device const float *inverse_masses [[buffer(3)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    device PMPackedVec3 *scratch [[buffer(12)]],
    device const uint *offsets [[buffer(13)]],
    device const PMDeformableNeighbor *neighbors [[buffer(14)]],
    device const uchar *free_nodes [[buffer(15)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    for (uint pass = 0u; pass < 32u; ++pass) {
        bool changed = false;
        for (uint node = 0u; node < constants.count; ++node) {
            const float3 position = pm_load(positions[node]);
            float3 correction = 0.0f;
            uint degree = 0u;
            if (free_nodes[node] != 0u) {
                for (uint item = offsets[node]; item < offsets[node + 1u];
                     ++item) {
                    const PMDeformableNeighbor edge = neighbors[item];
                    if (edge.bond != 0xffffffffu) continue;
                    const float weight = inverse_masses[node] +
                                         inverse_masses[edge.index];
                    const float3 delta =
                        pm_load(positions[edge.index]) - position;
                    const float length_value = length(delta);
                    const float maximum = edge.rest_length * 1.10f;
                    if (weight <= 0.0f || length_value <= maximum ||
                        inverse_masses[node] == 0.0f)
                        continue;
                    correction += delta *
                        ((length_value - maximum) * inverse_masses[node] /
                         (length_value * weight));
                    changed = changed ||
                              length_value > maximum * 1.0001f;
                    ++degree;
                }
            }
            scratch[node] = pm_store(
                position + correction / float(max(1u, degree)));
        }
        for (uint node = 0u; node < constants.count; ++node)
            positions[node] = scratch[node];
        if (!changed) break;
    }
}

kernel void pm_cloth_global_constraints_serial(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device const float *inverse_masses [[buffer(3)]],
    device PMMetalBond *bonds [[buffer(4)]],
    device uchar *active_bonds [[buffer(5)]],
    device PMPackedVec3 *surface_positions [[buffer(6)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    device const uint *triangle_indices [[buffer(8)]],
    device uchar *bond_damage [[buffer(9)]],
    device const uint *surface_source_indices [[buffer(10)]],
    device const PMPackedVec3 *rigid_impulses [[buffer(16)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    if (constants.preserve_volume != 0u &&
        constants.triangle_index_count >= 3u) {
        float volume = 0.0f;
        for (uint triangle = 0u;
             triangle < constants.triangle_index_count; triangle += 3u) {
            const float3 a = pm_load(positions[triangle_indices[triangle]]);
            const float3 b = pm_load(positions[triangle_indices[triangle + 1u]]);
            const float3 c = pm_load(positions[triangle_indices[triangle + 2u]]);
            volume += dot(a, cross(b, c)) / 6.0f;
        }
        const float alpha = constants.volume_compliance /
            max(constants.timestep * constants.timestep, 1.0e-12f);
        float denominator = alpha;
        for (uint vertex_index = 0u; vertex_index < constants.count;
             ++vertex_index) {
            float3 gradient = 0.0f;
            for (uint triangle = 0u;
                 triangle < constants.triangle_index_count; triangle += 3u) {
                const uint a_index = triangle_indices[triangle];
                const uint b_index = triangle_indices[triangle + 1u];
                const uint c_index = triangle_indices[triangle + 2u];
                const float3 a = pm_load(positions[a_index]);
                const float3 b = pm_load(positions[b_index]);
                const float3 c = pm_load(positions[c_index]);
                if (vertex_index == a_index) gradient += cross(b, c) / 6.0f;
                if (vertex_index == b_index) gradient += cross(c, a) / 6.0f;
                if (vertex_index == c_index) gradient += cross(a, b) / 6.0f;
            }
            denominator += inverse_masses[vertex_index] * dot(gradient, gradient);
        }
        if (denominator > 1.0e-12f) {
            const float lambda =
                (volume - constants.target_volume) / denominator;
            for (uint vertex_index = 0u; vertex_index < constants.count;
                 ++vertex_index) {
                float3 gradient = 0.0f;
                for (uint triangle = 0u;
                     triangle < constants.triangle_index_count;
                     triangle += 3u) {
                    const uint a_index = triangle_indices[triangle];
                    const uint b_index = triangle_indices[triangle + 1u];
                    const uint c_index = triangle_indices[triangle + 2u];
                    const float3 a = pm_load(positions[a_index]);
                    const float3 b = pm_load(positions[b_index]);
                    const float3 c = pm_load(positions[c_index]);
                    if (vertex_index == a_index)
                        gradient += cross(b, c) / 6.0f;
                    if (vertex_index == b_index)
                        gradient += cross(c, a) / 6.0f;
                    if (vertex_index == c_index)
                        gradient += cross(a, b) / 6.0f;
                }
                positions[vertex_index] = pm_store(
                    pm_load(positions[vertex_index]) -
                    gradient * inverse_masses[vertex_index] * lambda);
            }
        }
    }
}

kernel void pm_cloth_project_volume(
    device PMPackedVec3 *positions [[buffer(0)]],
    device const float *inverse_masses [[buffer(3)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    device const uint *triangle_indices [[buffer(8)]],
    device PMPackedVec3 *gradients [[buffer(12)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    const uint lane_count = threads_per_group.x;
    const uint triangle_count = constants.triangle_index_count / 3u;
    threadgroup float scalar_terms[128];
    threadgroup float volume;
    threadgroup float denominator;
    threadgroup float lambda;
    if (lane == 0u) volume = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint wave = 0u; wave < triangle_count; wave += lane_count) {
        const uint triangle = wave + lane;
        float contribution = 0.0f;
        if (triangle < triangle_count) {
            const uint base = 3u * triangle;
            const float3 a =
                pm_load(positions[triangle_indices[base]]);
            const float3 b =
                pm_load(positions[triangle_indices[base + 1u]]);
            const float3 c =
                pm_load(positions[triangle_indices[base + 2u]]);
            contribution = dot(a, cross(b, c)) / 6.0f;
        }
        scalar_terms[lane] = contribution;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint count = min(lane_count, triangle_count - wave);
            for (uint item = 0u; item < count; ++item)
                volume += scalar_terms[item];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint node = lane; node < constants.count;
         node += lane_count) {
        float3 gradient = 0.0f;
        for (uint base = 0u; base < constants.triangle_index_count;
             base += 3u) {
            const uint a_index = triangle_indices[base];
            const uint b_index = triangle_indices[base + 1u];
            const uint c_index = triangle_indices[base + 2u];
            const float3 a = pm_load(positions[a_index]);
            const float3 b = pm_load(positions[b_index]);
            const float3 c = pm_load(positions[c_index]);
            if (node == a_index) gradient += cross(b, c) / 6.0f;
            if (node == b_index) gradient += cross(c, a) / 6.0f;
            if (node == c_index) gradient += cross(a, b) / 6.0f;
        }
        gradients[node] = pm_store(gradient);
    }
    threadgroup_barrier(
        mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (lane == 0u)
        denominator = constants.volume_compliance /
            max(constants.timestep * constants.timestep, 1.0e-12f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint wave = 0u; wave < constants.count; wave += lane_count) {
        const uint node = wave + lane;
        scalar_terms[lane] = node < constants.count
            ? inverse_masses[node] *
                  dot(pm_load(gradients[node]),
                      pm_load(gradients[node]))
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint count = min(lane_count, constants.count - wave);
            for (uint item = 0u; item < count; ++item)
                denominator += scalar_terms[item];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        lambda = denominator > 1.0e-12f
            ? (volume - constants.target_volume) / denominator
            : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint node = lane; node < constants.count;
         node += lane_count)
        positions[node] = pm_store(
            pm_load(positions[node]) -
            pm_load(gradients[node]) * inverse_masses[node] * lambda);
}

kernel void pm_cloth_finalize(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device const float *inverse_masses [[buffer(3)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    pm_finalize_deformable_node(
        positions, previous, velocities, inverse_masses, constants, index);
}

kernel void pm_cloth_surface_update(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *surface_positions [[buffer(6)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    device const uint *surface_source_indices [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= constants.surface_count) return;
    const uint source = surface_source_indices[index];
    if (source < constants.count) surface_positions[index] = positions[source];
}

kernel void pm_cloth_impact(
    device PMMetalBond *bonds [[buffer(4)]],
    device uchar *active_bonds [[buffer(5)]],
    constant PMDeformableConstants &constants [[buffer(7)]],
    device const PMPackedVec3 *rigid_impulses [[buffer(16)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= constants.bond_count ||
        constants.impact_break_impulse <= 0.0f)
        return;
    device PMMetalBond &bond = bonds[index];
    if (bond.active == 0u) return;
    const float impulse =
        length(pm_load(rigid_impulses[bond.first])) +
        length(pm_load(rigid_impulses[bond.second]));
    if (impulse <= constants.impact_break_impulse) return;
    bond.active = 0u;
    active_bonds[index] = 0u;
}

// CUDA resolves both directions of rigid-cloth contact: cloth vertices
// against rigid triangles, then each dynamic body's conservative bounding
// sphere against cloth triangle interiors.  Keep this second phase serial so
// every body samples the same cloth state before corrections are reduced.
kernel void pm_cloth_rigid_surface_serial(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *velocities [[buffer(1)]],
    device const float *inverse_masses [[buffer(2)]],
    device const uint *triangle_indices [[buffer(3)]],
    device PMRigidBodyState *rigid_states [[buffer(4)]],
    device const PMRigidParameters *rigid_parameters [[buffer(5)]],
    device const PMPackedVec3 *mesh_vertices [[buffer(6)]],
    device const PMTriangleMeshInfo *meshes [[buffer(7)]],
    device const PMRigidBodyState *previous_states [[buffer(8)]],
    constant PMClothRigidConstants &constants [[buffer(9)]],
    device PMClothBodyCorrection *corrections [[buffer(10)]],
    device PMPackedVec3 *surface_positions [[buffer(11)]],
    device const uint *surface_physical_indices [[buffer(12)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;

    for (uint body_index = 0u; body_index < constants.rigid_count;
         ++body_index) {
        corrections[body_index] = {
            {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f,
            {0xffffffffu, 0xffffffffu, 0xffffffffu}, 0u};
        device const PMRigidParameters &body =
            rigid_parameters[body_index];
        if (body.motion != 2u || constants.fracture_enabled != 0u)
            continue;

        device PMRigidBodyState &state = rigid_states[body_index];
        device const PMTriangleMeshInfo &mesh = meshes[body.mesh_index];
        const float3 local_center =
            (pm_load(mesh.minimum) + pm_load(mesh.maximum)) * 0.5f;
        float mesh_radius = 0.0f;
        for (uint node = 0u; node < mesh.vertex_count; ++node) {
            const float3 delta =
                pm_load(mesh_vertices[mesh.vertex_offset + node]) -
                local_center;
            mesh_radius = max(mesh_radius, length(delta));
        }
        const float3 center = pm_world_point(state, local_center);
        const float3 previous_center =
            pm_world_point(previous_states[body_index], local_center);
        const float radius = mesh_radius + body.collision_margin +
                             constants.thickness;

        float best_penetration = 0.0f;
        float3 best_normal = 0.0f;
        float3 best_contact = 0.0f;
        uint best_triangle = 0u;
        for (uint base = 0u; base + 2u < constants.triangle_index_count;
             base += 3u) {
            const uint ia = triangle_indices[base];
            const uint ib = triangle_indices[base + 1u];
            const uint ic = triangle_indices[base + 2u];
            if (ia >= constants.vertex_count ||
                ib >= constants.vertex_count ||
                ic >= constants.vertex_count)
                continue;
            const float3 a = pm_load(positions[ia]);
            const float3 b = pm_load(positions[ib]);
            const float3 c = pm_load(positions[ic]);
            const float3 face_value = cross(b - a, c - a);
            if (dot(face_value, face_value) < 1.0e-12f) continue;
            const float3 face = normalize(face_value);
            const float3 nearest =
                pm_closest_point_triangle(center, a, b, c);
            const float3 delta = center - nearest;
            const float distance_value = length(delta);
            float3 contact_normal = distance_value > 1.0e-6f
                ? delta / distance_value
                : face * (dot(previous_center - a, face) >= 0.0f
                              ? 1.0f : -1.0f);
            float penetration = radius - distance_value;
            const float before = dot(previous_center - a, face);
            const float after = dot(center - a, face);
            if (before * after < 0.0f) {
                const float fraction = before / (before - after);
                const float3 crossing =
                    previous_center + (center - previous_center) * fraction;
                const float3 crossing_nearest =
                    pm_closest_point_triangle(crossing, a, b, c);
                if (dot(crossing_nearest - crossing,
                        crossing_nearest - crossing) < radius * radius) {
                    contact_normal = face * (before > 0.0f ? 1.0f : -1.0f);
                    penetration = max(penetration, radius + abs(after));
                }
            } else if (penetration > 0.0f && before * after > 0.0f) {
                contact_normal = face * (before > 0.0f ? 1.0f : -1.0f);
            }
            if (penetration > best_penetration) {
                best_penetration = penetration;
                best_normal = contact_normal;
                best_contact = nearest;
                best_triangle = base;
            }
        }
        if (best_penetration <= 0.0f) continue;

        const uint a = triangle_indices[best_triangle];
        const uint b = triangle_indices[best_triangle + 1u];
        const uint c = triangle_indices[best_triangle + 2u];
        const float free_fraction =
            ((inverse_masses[a] > 0.0f ? 1.0f : 0.0f) +
             (inverse_masses[b] > 0.0f ? 1.0f : 0.0f) +
             (inverse_masses[c] > 0.0f ? 1.0f : 0.0f)) / 3.0f;
        constexpr float cloth_share = 0.5f;
        const float cloth_shift = cloth_share * best_penetration;
        const float3 arm = best_contact - pm_load(state.position);
        const float incoming =
            dot(pm_load(state.linear_velocity), best_normal);
        const float3 normal_axis = cross(arm, best_normal);
        const float normal_impulse = incoming < 0.0f &&
                                             body.inverse_mass > 0.0f
                                         ? -incoming / body.inverse_mass
                                         : 0.0f;
        const float support_radius = max(0.15f, mesh_radius * 0.8f);
        const float inverse_support_squared =
            1.0f / (support_radius * support_radius);
        float weight_sum = 0.0f;
        float maximum_weighted_inverse_mass = 0.0f;
        float weighted_inverse_mass_squared = 0.0f;
        float3 weighted_cloth_velocity = 0.0f;
        for (uint node = 0u; node < constants.vertex_count; ++node) {
            const float inverse_mass = inverse_masses[node];
            if (inverse_mass <= 0.0f) continue;
            const float3 delta =
                pm_load(positions[node]) - best_contact;
            const float weight = max(
                0.0f, 1.0f - dot(delta, delta) * inverse_support_squared);
            const float weighted = weight * weight;
            weight_sum += weighted;
            weighted_inverse_mass_squared +=
                inverse_mass * weighted * weighted;
            weighted_cloth_velocity +=
                pm_load(velocities[node]) * weighted;
            maximum_weighted_inverse_mass = max(
                maximum_weighted_inverse_mass, inverse_mass * weighted);
        }
        const float maximum_cloth_impulse =
            maximum_weighted_inverse_mass > 0.0f
                ? 0.1f * constants.thickness * weight_sum /
                      (constants.timestep * maximum_weighted_inverse_mass)
                : normal_impulse;
        const float cloth_impulse =
            min(normal_impulse, maximum_cloth_impulse);
        float3 tangent_impulse = 0.0f;
        if (weight_sum > 1.0e-8f && constants.friction > 0.0f) {
            const float3 cloth_velocity =
                weighted_cloth_velocity / weight_sum;
            const float3 body_velocity =
                pm_load(state.linear_velocity) +
                cross(pm_load(state.angular_velocity), arm);
            const float3 relative = body_velocity - cloth_velocity;
            const float separating_speed = dot(relative, best_normal);
            const float3 tangent =
                relative - best_normal * separating_speed;
            const float tangent_speed = length(tangent);
            const float release_weight =
                max(0.0f, 1.0f - max(incoming, 0.0f) / 0.5f);
            if (release_weight > 0.0f && separating_speed <= 0.0f &&
                tangent_speed > 1.0e-6f) {
                const float3 direction = tangent / tangent_speed;
                const float3 tangent_axis = cross(arm, direction);
                const float cloth_inverse_mass =
                    weighted_inverse_mass_squared /
                    (weight_sum * weight_sum);
                const float effective_inverse_mass = body.inverse_mass +
                    dot(cross(pm_inverse_inertia_mul(
                                  body, state, tangent_axis), arm),
                        direction) +
                    cloth_inverse_mass;
                const float correction_impulse =
                    best_penetration > 0.0f && body.inverse_mass > 0.0f
                        ? 0.2f * best_penetration /
                              (constants.timestep * body.inverse_mass)
                        : 0.0f;
                const float friction_limit = constants.friction *
                    release_weight * max(normal_impulse, correction_impulse);
                const float magnitude = effective_inverse_mass > 1.0e-6f
                    ? min(tangent_speed / effective_inverse_mass,
                          friction_limit)
                    : 0.0f;
                tangent_impulse = direction * -magnitude;
            }
        }
        const float3 cloth_tangent_impulse =
            pm_limit(tangent_impulse, maximum_cloth_impulse);
        corrections[body_index] = {
            pm_store(best_normal * -cloth_shift),
            pm_store(best_normal * -cloth_impulse - cloth_tangent_impulse),
            pm_store(best_contact), support_radius, weight_sum,
            {a, b, c}, 1u};

        state.position = pm_store(
            pm_load(state.position) + best_normal *
                (best_penetration - cloth_shift * free_fraction));
        if (normal_impulse > 0.0f) {
            state.linear_velocity = pm_store(pm_limit(
                pm_load(state.linear_velocity) +
                    best_normal * (normal_impulse * body.inverse_mass),
                body.maximum_linear_speed));
            state.angular_velocity = pm_store(pm_limit(
                pm_load(state.angular_velocity) +
                    pm_inverse_inertia_mul(
                        body, state, normal_axis * normal_impulse),
                body.maximum_angular_speed));
        }
        state.linear_velocity = pm_store(pm_limit(
            pm_load(state.linear_velocity) +
                tangent_impulse * body.inverse_mass,
            body.maximum_linear_speed));
        state.angular_velocity = pm_store(pm_limit(
            pm_load(state.angular_velocity) +
                pm_inverse_inertia_mul(
                    body, state, cross(arm, tangent_impulse)),
            body.maximum_angular_speed));
    }

    for (uint node = 0u; node < constants.vertex_count; ++node) {
        if (inverse_masses[node] <= 0.0f) continue;
        float3 offset = 0.0f;
        float3 velocity = pm_load(velocities[node]);
        for (uint body = 0u; body < constants.rigid_count; ++body) {
            const PMClothBodyCorrection correction = corrections[body];
            if (correction.active == 0u) continue;
            if (correction.vertices[0] == node ||
                correction.vertices[1] == node ||
                correction.vertices[2] == node)
                offset += pm_load(correction.offset);
            if (correction.weight_sum > 1.0e-8f) {
                const float3 delta =
                    pm_load(positions[node]) -
                    pm_load(correction.contact);
                const float support_squared =
                    correction.support_radius * correction.support_radius;
                const float weight = max(
                    0.0f, 1.0f - dot(delta, delta) / support_squared);
                velocity += pm_load(correction.impulse) *
                    (inverse_masses[node] * weight * weight /
                     correction.weight_sum);
            }
        }
        positions[node] = pm_store(pm_load(positions[node]) + offset);
        velocities[node] = pm_store(pm_limit(velocity, 20.0f));
    }
    for (uint surface = 0u; surface < constants.surface_count; ++surface) {
        const uint source = surface_physical_indices[surface];
        if (source < constants.vertex_count)
            surface_positions[surface] = positions[source];
    }
}

kernel void pm_soft_body_predict(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device const float *inverse_masses [[buffer(3)]],
    constant PMDeformableConstants &constants [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    pm_integrate_deformable_node(
        positions, previous, velocities, inverse_masses, constants, index);
}

kernel void pm_soft_body_project_bonds(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const float *inverse_masses [[buffer(3)]],
    device const PMMetalBond *bonds [[buffer(4)]],
    constant PMDeformableConstants &constants [[buffer(8)]],
    device PMPackedVec3 *scratch [[buffer(10)]],
    uint node [[thread_position_in_grid]]) {
    pm_project_bond_node(
        positions, inverse_masses, bonds, scratch, constants, node);
}

kernel void pm_soft_body_apply_bonds(
    device PMPackedVec3 *positions [[buffer(0)]],
    constant PMDeformableConstants &constants [[buffer(8)]],
    device const PMPackedVec3 *scratch [[buffer(10)]],
    uint node [[thread_position_in_grid]]) {
    if (node < constants.count) positions[node] = scratch[node];
}

kernel void pm_soft_body_global_constraints_serial(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device const float *inverse_masses [[buffer(3)]],
    device PMMetalBond *bonds [[buffer(4)]],
    device PMPackedVec3 *surface_positions [[buffer(5)]],
    device const PMPackedVec3 *surface_rest [[buffer(6)]],
    device const PMMetalSurfaceBinding *bindings [[buffer(7)]],
    constant PMDeformableConstants &constants [[buffer(8)]],
    device const PMPackedVec3 *rest_positions [[buffer(9)]],
    device PMPackedVec3 *corrections [[buffer(10)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    if (constants.shape_matching_stiffness > 0.0f) {
        float movable_mass = 0.0f;
        float3 current_center = 0.0f;
        float3 rest_center = 0.0f;
        for (uint node = 0u; node < constants.count; ++node) {
            if (inverse_masses[node] <= 0.0f) continue;
            const float mass = 1.0f / inverse_masses[node];
            movable_mass += mass;
            current_center += pm_load(positions[node]) * mass;
            rest_center += pm_load(rest_positions[node]) * mass;
        }
        if (movable_mass > 0.0f) {
            current_center /= movable_mass;
            rest_center /= movable_mass;
            float3 rest_covariance[3]{float3(0.0f), float3(0.0f),
                                      float3(0.0f)};
            float3 covariance[3]{float3(0.0f), float3(0.0f),
                                 float3(0.0f)};
            for (uint node = 0u; node < constants.count; ++node) {
                if (inverse_masses[node] <= 0.0f) continue;
                const float mass = 1.0f / inverse_masses[node];
                const float3 current =
                    pm_load(positions[node]) - current_center;
                const float3 rest =
                    pm_load(rest_positions[node]) - rest_center;
                for (uint column = 0u; column < 3u; ++column) {
                    rest_covariance[column] += rest * (mass * rest[column]);
                    covariance[column] += current * (mass * rest[column]);
                }
            }
            const float3 inverse_rows[3]{
                cross(rest_covariance[1], rest_covariance[2]),
                cross(rest_covariance[2], rest_covariance[0]),
                cross(rest_covariance[0], rest_covariance[1])};
            const float determinant =
                dot(rest_covariance[0], inverse_rows[0]);
            if (abs(determinant) > 1.0e-10f) {
                const float inverse_determinant = 1.0f / determinant;
                const float3 inverse_rest[3]{
                    float3(inverse_rows[0].x, inverse_rows[1].x,
                           inverse_rows[2].x) * inverse_determinant,
                    float3(inverse_rows[0].y, inverse_rows[1].y,
                           inverse_rows[2].y) * inverse_determinant,
                    float3(inverse_rows[0].z, inverse_rows[1].z,
                           inverse_rows[2].z) * inverse_determinant};
                float3 deformation[3];
                for (uint column = 0u; column < 3u; ++column)
                    deformation[column] =
                        covariance[0] * inverse_rest[column].x +
                        covariance[1] * inverse_rest[column].y +
                        covariance[2] * inverse_rest[column].z;
                PMQuaternion orientation{0.0f, 0.0f, 0.0f, 1.0f};
                for (uint iteration = 0u; iteration < 12u; ++iteration) {
                    const float3 axes[3]{
                        pm_rotate(orientation, float3(1.0f, 0.0f, 0.0f)),
                        pm_rotate(orientation, float3(0.0f, 1.0f, 0.0f)),
                        pm_rotate(orientation, float3(0.0f, 0.0f, 1.0f))};
                    float3 angular = cross(axes[0], deformation[0]) +
                                     cross(axes[1], deformation[1]) +
                                     cross(axes[2], deformation[2]);
                    const float denominator =
                        abs(dot(axes[0], deformation[0]) +
                            dot(axes[1], deformation[1]) +
                            dot(axes[2], deformation[2])) +
                        1.0e-9f;
                    angular /= denominator;
                    const float magnitude = length(angular);
                    if (magnitude < 1.0e-6f) break;
                    const float angle = min(magnitude, 0.5f);
                    const float half_angle = 0.5f * angle;
                    const float3 axis = angular / magnitude;
                    const PMQuaternion delta{axis.x * sin(half_angle),
                                             axis.y * sin(half_angle),
                                             axis.z * sin(half_angle),
                                             cos(half_angle)};
                    orientation = pm_quaternion_normalize(
                        pm_quaternion_multiply(delta, orientation));
                }
                float3 weighted_correction = 0.0f;
                for (uint node = 0u; node < constants.count; ++node) {
                    if (inverse_masses[node] <= 0.0f) {
                        corrections[node] = {0.0f, 0.0f, 0.0f};
                        continue;
                    }
                    float shortest = INFINITY;
                    for (uint bond = 0u; bond < constants.bond_count; ++bond)
                        if (bonds[bond].first == node ||
                            bonds[bond].second == node)
                            shortest = min(shortest, bonds[bond].rest_length);
                    const float maximum = isfinite(shortest)
                        ? shortest * constants.maximum_projection_fraction
                        : INFINITY;
                    const float3 target = current_center +
                        pm_rotate(orientation,
                                  pm_load(rest_positions[node]) - rest_center);
                    const float3 correction = pm_limit(
                        (target - pm_load(positions[node])) *
                            constants.shape_matching_stiffness,
                        maximum);
                    corrections[node] = pm_store(correction);
                    weighted_correction += correction /
                                           inverse_masses[node];
                }
                const float3 center_correction =
                    weighted_correction / movable_mass;
                for (uint node = 0u; node < constants.count; ++node)
                    if (inverse_masses[node] > 0.0f)
                        positions[node] = pm_store(
                            pm_load(positions[node]) +
                            pm_load(corrections[node]) - center_correction);
            }
        }
    }
}

kernel void pm_soft_body_shape_matching(
    device PMPackedVec3 *positions [[buffer(0)]],
    device const float *inverse_masses [[buffer(3)]],
    device const PMMetalBond *bonds [[buffer(4)]],
    constant PMDeformableConstants &constants [[buffer(8)]],
    device const PMPackedVec3 *rest_positions [[buffer(9)]],
    device PMPackedVec3 *corrections [[buffer(10)]],
    device PMQuaternion *stored_orientation [[buffer(11)]],
    device const PMSoftContactState *contact_state [[buffer(12)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    if (atomic_load_explicit(
            &contact_state->dynamic_contact_flag,
            memory_order_relaxed) != 0u)
        return;
    const uint lane_count = threads_per_group.x;
    threadgroup float scalar_terms[128];
    threadgroup float3 current_terms[128];
    threadgroup float3 rest_terms[128];
    threadgroup float movable_mass;
    threadgroup float3 current_center;
    threadgroup float3 rest_center;
    threadgroup float3 rest_covariance[3];
    threadgroup float3 covariance[3];
    threadgroup PMQuaternion orientation;
    threadgroup float3 center_correction;
    threadgroup float maximum_projection;
    threadgroup bool valid;

    if (lane == 0u) {
        movable_mass = 0.0f;
        current_center = 0.0f;
        rest_center = 0.0f;
        center_correction = 0.0f;
        maximum_projection = INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint wave = 0u; wave < constants.count; wave += lane_count) {
        const uint node = wave + lane;
        float mass = 0.0f;
        float3 current = 0.0f;
        float3 rest = 0.0f;
        if (node < constants.count && inverse_masses[node] > 0.0f) {
            mass = 1.0f / inverse_masses[node];
            current = pm_load(positions[node]) * mass;
            rest = pm_load(rest_positions[node]) * mass;
        }
        scalar_terms[lane] = mass;
        current_terms[lane] = current;
        rest_terms[lane] = rest;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint count = min(lane_count, constants.count - wave);
            for (uint item = 0u; item < count; ++item) {
                movable_mass += scalar_terms[item];
                current_center += current_terms[item];
                rest_center += rest_terms[item];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        valid = movable_mass > 0.0f;
        if (valid) {
            current_center /= movable_mass;
            rest_center /= movable_mass;
        }
        for (uint column = 0u; column < 3u; ++column) {
            rest_covariance[column] = 0.0f;
            covariance[column] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint column = 0u; column < 3u; ++column) {
        for (uint wave = 0u; wave < constants.count; wave += lane_count) {
            const uint node = wave + lane;
            float3 rest_term = 0.0f;
            float3 current_term = 0.0f;
            if (valid && node < constants.count &&
                inverse_masses[node] > 0.0f) {
                const float mass = 1.0f / inverse_masses[node];
                const float3 rest =
                    pm_load(rest_positions[node]) - rest_center;
                const float3 current =
                    pm_load(positions[node]) - current_center;
                rest_term = rest * (mass * rest[column]);
                current_term = current * (mass * rest[column]);
            }
            rest_terms[lane] = rest_term;
            current_terms[lane] = current_term;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                const uint count = min(lane_count, constants.count - wave);
                for (uint item = 0u; item < count; ++item) {
                    rest_covariance[column] += rest_terms[item];
                    covariance[column] += current_terms[item];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (lane == 0u && valid) {
        const float3 inverse_rows[3]{
            cross(rest_covariance[1], rest_covariance[2]),
            cross(rest_covariance[2], rest_covariance[0]),
            cross(rest_covariance[0], rest_covariance[1])};
        const float determinant =
            dot(rest_covariance[0], inverse_rows[0]);
        valid = abs(determinant) > 1.0e-10f;
        if (valid) {
            const float inverse_determinant = 1.0f / determinant;
            const float3 inverse_rest[3]{
                float3(inverse_rows[0].x, inverse_rows[1].x,
                       inverse_rows[2].x) * inverse_determinant,
                float3(inverse_rows[0].y, inverse_rows[1].y,
                       inverse_rows[2].y) * inverse_determinant,
                float3(inverse_rows[0].z, inverse_rows[1].z,
                       inverse_rows[2].z) * inverse_determinant};
            float3 deformation[3];
            for (uint column = 0u; column < 3u; ++column)
                deformation[column] =
                    covariance[0] * inverse_rest[column].x +
                    covariance[1] * inverse_rest[column].y +
                    covariance[2] * inverse_rest[column].z;
            orientation = pm_quaternion_normalize(*stored_orientation);
            if (orientation.x == 0.0f && orientation.y == 0.0f &&
                orientation.z == 0.0f && orientation.w == 0.0f)
                orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            for (uint iteration = 0u; iteration < 12u; ++iteration) {
                const float3 axes[3]{
                    pm_rotate(orientation, float3(1.0f, 0.0f, 0.0f)),
                    pm_rotate(orientation, float3(0.0f, 1.0f, 0.0f)),
                    pm_rotate(orientation, float3(0.0f, 0.0f, 1.0f))};
                float3 angular = cross(axes[0], deformation[0]) +
                                 cross(axes[1], deformation[1]) +
                                 cross(axes[2], deformation[2]);
                const float denominator =
                    abs(dot(axes[0], deformation[0]) +
                        dot(axes[1], deformation[1]) +
                        dot(axes[2], deformation[2])) +
                    1.0e-9f;
                angular /= denominator;
                const float magnitude = length(angular);
                if (magnitude < 1.0e-6f) break;
                const float angle = min(magnitude, 0.5f);
                const float half_angle = 0.5f * angle;
                const float3 axis = angular / magnitude;
                const PMQuaternion delta{
                    axis.x * sin(half_angle), axis.y * sin(half_angle),
                    axis.z * sin(half_angle), cos(half_angle)};
                orientation = pm_quaternion_normalize(
                    pm_quaternion_multiply(delta, orientation));
            }
            *stored_orientation = orientation;
            float shortest = INFINITY;
            for (uint bond = 0u; bond < constants.bond_count; ++bond)
                shortest = min(shortest, bonds[bond].rest_length);
            maximum_projection =
                shortest * constants.maximum_projection_fraction;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint wave = 0u; wave < constants.count; wave += lane_count) {
        const uint node = wave + lane;
        float3 weighted = 0.0f;
        if (node < constants.count) {
            float3 correction = 0.0f;
            if (valid && inverse_masses[node] > 0.0f) {
                const float3 target = current_center +
                    pm_rotate(orientation,
                              pm_load(rest_positions[node]) - rest_center);
                correction = pm_limit(
                    (target - pm_load(positions[node])) *
                        constants.shape_matching_stiffness,
                    maximum_projection);
                weighted = correction / inverse_masses[node];
            }
            corrections[node] = pm_store(correction);
        }
        current_terms[lane] = weighted;
        threadgroup_barrier(
            mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint count = min(lane_count, constants.count - wave);
            for (uint item = 0u; item < count; ++item)
                center_correction += current_terms[item];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u && valid) center_correction /= movable_mass;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid) return;
    for (uint node = lane; node < constants.count; node += lane_count)
        if (inverse_masses[node] > 0.0f)
            positions[node] = pm_store(
                pm_load(positions[node]) + pm_load(corrections[node]) -
                center_correction);
}

kernel void pm_soft_body_finalize(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device const float *inverse_masses [[buffer(3)]],
    constant PMDeformableConstants &constants [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    pm_finalize_deformable_node(
        positions, previous, velocities, inverse_masses, constants, index);
}

kernel void pm_soft_body_damping_prepare(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device const PMPackedVec3 *velocities [[buffer(2)]],
    device const float *inverse_masses [[buffer(3)]],
    device const PMMetalBond *bonds [[buffer(4)]],
    constant PMDeformableConstants &constants [[buffer(8)]],
    device PMPackedVec3 *corrections [[buffer(10)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= constants.count || constants.spring_damping <= 0.0f) return;
    if (inverse_masses[node] <= 0.0f) {
        corrections[node] = {0.0f, 0.0f, 0.0f};
        return;
    }
    float3 correction = 0.0f;
    uint active_count = 0u;
    for (uint bond = 0u; bond < constants.bond_count; ++bond) {
        uint other = constants.count;
        if (bonds[bond].first == node) other = bonds[bond].second;
        if (bonds[bond].second == node) other = bonds[bond].first;
        if (other >= constants.count || bonds[bond].active == 0u) continue;
        const float3 difference =
            pm_load(positions[other]) - pm_load(positions[node]);
        if (dot(difference, difference) <= 1.0e-12f) continue;
        const float3 axis = normalize(difference);
        correction += axis * dot(pm_load(velocities[other]) -
                                     pm_load(velocities[node]),
                                 axis);
        ++active_count;
    }
    if (active_count != 0u)
        correction *= 0.5f * constants.spring_damping /
                      float(active_count);
    corrections[node] = pm_store(pm_limit(
        pm_load(velocities[node]) + correction, constants.maximum_speed));
}

kernel void pm_soft_body_damping_apply(
    device PMPackedVec3 *velocities [[buffer(2)]],
    constant PMDeformableConstants &constants [[buffer(8)]],
    device const PMPackedVec3 *corrections [[buffer(10)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= constants.count || constants.spring_damping <= 0.0f) return;
    velocities[node] = corrections[node];
}

kernel void pm_soft_body_surface_update(
    device const PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *surface_positions [[buffer(5)]],
    device const PMPackedVec3 *surface_rest [[buffer(6)]],
    device const PMMetalSurfaceBinding *bindings [[buffer(7)]],
    constant PMDeformableConstants &constants [[buffer(8)]],
    device const PMPackedVec3 *rest_positions [[buffer(9)]],
    uint surface_index [[thread_position_in_grid]]) {
    if (surface_index >= constants.surface_count) return;
    float3 delta = 0.0f;
    for (uint index = 0u; index < 4u; ++index) {
        const uint node = bindings[surface_index].nodes[index];
        if (node < constants.count) {
            delta += (pm_load(positions[node]) -
                      pm_load(rest_positions[node])) *
                     bindings[surface_index].weights[index];
        }
    }
    surface_positions[surface_index] = pm_store(
        pm_load(surface_rest[surface_index]) + delta);
}

static float3 pm_rigid_point_velocity(
    device const PMRigidBodyState &state, float3 local_anchor) {
    const float3 arm = pm_rotate(state.orientation, local_anchor);
    return pm_load(state.linear_velocity) +
           cross(pm_load(state.angular_velocity), arm);
}

static float pm_rigid_direction_weight(
    device const PMRigidParameters &body,
    device const PMRigidBodyState &state, float3 local_anchor,
    float3 direction) {
    const float3 arm = pm_rotate(state.orientation, local_anchor);
    const float3 torque = cross(arm, direction);
    return body.inverse_mass +
           dot(torque, pm_inverse_inertia_mul(body, state, torque));
}

static float3 pm_rigid_impulse_movement(
    device const PMRigidParameters &body,
    device const PMRigidBodyState &state, float3 local_anchor,
    float3 impulse) {
    const float3 arm = pm_rotate(state.orientation, local_anchor);
    return impulse * body.inverse_mass +
           cross(pm_inverse_inertia_mul(body, state, cross(arm, impulse)),
                 arm);
}

static void pm_move_rigid_by_rope(
    device PMRigidParameters &body, device PMRigidBodyState &state,
    float3 local_anchor, float3 impulse, float timestep) {
    if (body.inverse_mass <= 0.0f) return;
    const float3 arm = pm_rotate(state.orientation, local_anchor);
    const float3 translation = impulse * body.inverse_mass;
    const float3 rotation = pm_inverse_inertia_mul(
        body, state, cross(arm, impulse));
    state.position = pm_store(pm_load(state.position) + translation);
    const PMQuaternion spin = pm_quaternion_multiply(
        {rotation.x, rotation.y, rotation.z, 0.0f}, state.orientation);
    state.orientation = pm_quaternion_normalize(
        {fma(0.5f, spin.x, state.orientation.x),
         fma(0.5f, spin.y, state.orientation.y),
         fma(0.5f, spin.z, state.orientation.z),
         fma(0.5f, spin.w, state.orientation.w)});
    state.linear_velocity = pm_store(
        pm_load(state.linear_velocity) + translation / timestep);
    state.angular_velocity = pm_store(
        pm_load(state.angular_velocity) + rotation / timestep);
}

static float pm_rope_node_inverse(
    uint node, uint count, bool first_attached, bool last_attached,
    bool first_soft, bool last_soft, device const float *inverse_masses,
    device const PMRopeAnchorState *soft_anchors) {
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
    device const PMPackedVec3 *contact_normals,
    device const PMPackedVec3 *contact_normals2) {
    const float3 first = pm_load(contact_normals[node]);
    const float3 second = pm_load(contact_normals2[node]);
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
    constant PMRopeAttachmentConstants &attachments,
    device const float *inverse_masses,
    device const PMRopeAnchorState *soft_anchors,
    device const PMRigidParameters *rigid_parameters,
    device const PMRigidBodyState *rigid_states,
    device const PMPackedVec3 *contact_normals,
    device const PMPackedVec3 *contact_normals2) {
    if (node == 0u && first_attached)
        return pm_rigid_direction_weight(
            rigid_parameters[first_body], rigid_states[first_body],
            pm_load(attachments.first_anchor), direction);
    if (node + 1u == count && last_attached)
        return pm_rigid_direction_weight(
            rigid_parameters[last_body], rigid_states[last_body],
            pm_load(attachments.last_anchor), direction);
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
    constant PMRopeAttachmentConstants &attachments,
    device const float *inverse_masses,
    device const PMRopeAnchorState *soft_anchors,
    device const PMRigidParameters *rigid_parameters,
    device const PMRigidBodyState *rigid_states) {
    if (node == 0u && first_attached)
        return pm_rigid_direction_weight(
            rigid_parameters[first_body], rigid_states[first_body],
            pm_load(attachments.first_anchor), direction);
    if (node + 1u == count && last_attached)
        return pm_rigid_direction_weight(
            rigid_parameters[last_body], rigid_states[last_body],
            pm_load(attachments.last_anchor), direction);
    return pm_rope_node_inverse(
        node, count, first_attached, last_attached, first_soft, last_soft,
        inverse_masses, soft_anchors);
}

struct PMRopeRigidHit {
    float depth;
    float fraction;
    float3 normal;
    float3 point;
    uint body;
    bool found;
};

static PMRopeRigidHit pm_rope_find_rigid_contact(
    device const PMPackedVec3 *positions,
    device const PMPackedVec3 *previous, uint count, uint node,
    bool segment, float rope_radius, uint first_body, uint last_body,
    constant PMRopeAttachmentConstants &attachments,
    device const PMRigidBodyState *rigid_states,
    device const PMRigidBodyState *old_states,
    device const PMRigidParameters *rigid_parameters,
    device const PMPackedVec3 *vertices, device const uint *indices,
    device const PMTriangleMeshInfo *meshes,
    device const PMCollisionPlane *solid_planes,
    device uint *solid_hints) {
    PMRopeRigidHit best{0.0f, 0.0f, 0.0f, 0.0f,
                        attachments.rigid_count, false};
    const uint next = segment ? node + 1u : node;
    for (uint body = 0u; body < attachments.rigid_count; ++body) {
        if ((body == first_body &&
             node < attachments.first_contact_skip) ||
            (body == last_body &&
             next + attachments.last_contact_skip >= count))
            continue;
        device const PMRigidBodyState &state = rigid_states[body];
        device const PMRigidBodyState &old_state = old_states[body];
        device const PMRigidParameters &parameters = rigid_parameters[body];
        device const PMTriangleMeshInfo &mesh =
            meshes[parameters.mesh_index];
        const float radius = rope_radius + parameters.collision_margin;
        const PMQuaternion inverse =
            pm_quaternion_conjugate(state.orientation);
        const PMQuaternion old_inverse =
            pm_quaternion_conjugate(old_state.orientation);
        const float3 first = pm_rotate(
            inverse, pm_load(positions[node]) - pm_load(state.position));
        const float3 second = pm_rotate(
            inverse, pm_load(positions[next]) - pm_load(state.position));
        const float3 origin = pm_rotate(
            old_inverse,
            pm_load(previous[node]) - pm_load(old_state.position));
        const float3 lower = min(first, segment ? second : origin) - radius;
        const float3 upper = max(first, segment ? second : origin) + radius;
        if (any(upper < pm_load(mesh.minimum)) ||
            any(lower > pm_load(mesh.maximum)))
            continue;
        const bool solid = mesh.solid_plane_count != 0u;
        if (solid) {
            device uint &hint = solid_hints[
                node * attachments.rigid_capacity + body];
            const bool stationary =
                pm_load(state.position).x == pm_load(old_state.position).x &&
                pm_load(state.position).y == pm_load(old_state.position).y &&
                pm_load(state.position).z == pm_load(old_state.position).z &&
                state.orientation.x == old_state.orientation.x &&
                state.orientation.y == old_state.orientation.y &&
                state.orientation.z == old_state.orientation.z &&
                state.orientation.w == old_state.orientation.w;
            bool outside = false;
            bool separated = false;
            float side = -INFINITY;
            float entry_time = -1.0f;
            float3 nearest_normal = 0.0f;
            float nearest_offset = 0.0f;
            float3 entry_normal = 0.0f;
            float entry_offset = 0.0f;
            if (hint < mesh.solid_plane_count) {
                device const PMCollisionPlane &plane =
                    solid_planes[mesh.solid_plane_offset + hint];
                const float3 plane_normal = pm_load(plane.normal);
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
                device const PMCollisionPlane &plane =
                    solid_planes[mesh.solid_plane_offset + local];
                const float3 plane_normal = pm_load(plane.normal);
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
                if (entry_time >= 0.0f) {
                    nearest_normal = entry_normal;
                    nearest_offset = entry_offset;
                    side = dot(entry_normal, first) - entry_offset;
                }
                if (radius - side > best.depth) {
                    best.depth = radius - side;
                    best.fraction = 0.0f;
                    best.normal = pm_rotate(
                        state.orientation, nearest_normal);
                    best.point = pm_world_point(
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
                const float3 a = pm_load(vertices[
                    mesh.vertex_offset + indices[mesh.index_offset + local]]);
                const float3 b = pm_load(vertices[
                    mesh.vertex_offset +
                    indices[mesh.index_offset + local + 1u]]);
                const float3 c = pm_load(vertices[
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
                        pm_closest_point_triangle(first, a, b, c);
                }
                const float3 delta = rope_point - triangle_point;
                const float squared = dot(delta, delta);
                const float3 face_value = cross(b - a, c - a);
                const float3 face = solid
                    ? pm_load(solid_planes[
                          mesh.solid_plane_offset + local / 3u].normal)
                    : pm_normalized_or(
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
                            pm_closest_point_triangle(crossing, a, b, c);
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
                best.normal = pm_rotate(state.orientation, normal);
                best.point = pm_world_point(state, triangle_point);
                best.body = body;
                best.found = true;
            }
        }
    }
    return best;
}

static void pm_rope_accumulate_rigid_movement(
    uint body, float3 impulse, float3 local_point,
    device const PMRigidParameters *rigid_parameters,
    device const PMRigidBodyState *rigid_states,
    device PMPackedVec3 *body_translation,
    device PMPackedVec3 *body_rotation) {
    const float3 arm = pm_rotate(rigid_states[body].orientation, local_point);
    body_translation[body] = pm_store(
        pm_load(body_translation[body]) +
        impulse * rigid_parameters[body].inverse_mass);
    body_rotation[body] = pm_store(
        pm_load(body_rotation[body]) +
        pm_inverse_inertia_mul(rigid_parameters[body], rigid_states[body],
                               cross(arm, impulse)));
}

static void pm_rope_accumulate_rigid_movement_at_arm(
    uint body, float3 impulse, float3 arm,
    device const PMRigidParameters *rigid_parameters,
    device const PMRigidBodyState *rigid_states,
    device PMPackedVec3 *body_translation,
    device PMPackedVec3 *body_rotation) {
    body_translation[body] = pm_store(
        pm_load(body_translation[body]) +
        impulse * rigid_parameters[body].inverse_mass);
    body_rotation[body] = pm_store(
        pm_load(body_rotation[body]) +
        pm_inverse_inertia_mul(rigid_parameters[body], rigid_states[body],
                               cross(arm, impulse)));
}

static void pm_rope_contact_move(
    uint node, uint count, float3 impulse, bool first_attached,
    bool last_attached, bool first_soft, bool last_soft,
    uint first_body, uint last_body,
    constant PMRopeAttachmentConstants &attachments,
    device PMPackedVec3 *positions, device const float *inverse_masses,
    device const PMRopeAnchorState *soft_anchors,
    device const PMRigidParameters *rigid_parameters,
    device const PMRigidBodyState *rigid_states,
    device PMPackedVec3 *body_translation,
    device PMPackedVec3 *body_rotation) {
    if (node == 0u && first_attached) {
        pm_rope_accumulate_rigid_movement(
            first_body, impulse, pm_load(attachments.first_anchor),
            rigid_parameters, rigid_states, body_translation, body_rotation);
        return;
    }
    if (node + 1u == count && last_attached) {
        pm_rope_accumulate_rigid_movement(
            last_body, impulse, pm_load(attachments.last_anchor),
            rigid_parameters, rigid_states, body_translation, body_rotation);
        return;
    }
    const float inverse = pm_rope_node_inverse(
        node, count, first_attached, last_attached, first_soft, last_soft,
        inverse_masses, soft_anchors);
    positions[node] = pm_store(pm_load(positions[node]) + impulse * inverse);
}

kernel void pm_rope_cloth_sample(
    device PMPackedVec3 *rope_positions [[buffer(0)]],
    device PMPackedVec3 *rope_velocities [[buffer(1)]],
    device const float *rope_inverse_masses [[buffer(2)]],
    device const PMPackedVec3 *cloth_positions [[buffer(3)]],
    device const PMPackedVec3 *cloth_velocities [[buffer(4)]],
    device const float *cloth_inverse_masses [[buffer(5)]],
    constant PMCouplingConstants &constants [[buffer(6)]],
    device PMRopeAnchorState *anchors [[buffer(12)]],
    uint thread_index [[thread_position_in_grid]]) {
    (void)rope_positions;
    (void)rope_velocities;
    (void)rope_inverse_masses;
    if (thread_index >= 2u || constants.enabled == 0u) return;
    const uint vertices[2] = {constants.first_vertex,
                              constants.last_vertex};
    const uint end = thread_index;
    const uint cloth_node = vertices[end];
    if (cloth_node == 0xffffffffu || cloth_node >= constants.count_b) return;
    anchors[end].position = cloth_positions[cloth_node];
    anchors[end].velocity = cloth_velocities[cloth_node];
    anchors[end].impulse = {0.0f, 0.0f, 0.0f};
    anchors[end].inverse_mass = min(
        cloth_inverse_masses[cloth_node],
        max(constants.stiffness, 0.0f));
}

constant uint pm_rope_soft_header_words = 32u;
constant uint pm_rope_soft_valid = 0u;
constant uint pm_rope_soft_node_count = 1u;
constant uint pm_rope_soft_surface_count = 2u;
constant uint pm_rope_soft_index_count = 3u;
constant uint pm_rope_soft_surface_offset = 4u;
constant uint pm_rope_soft_previous_offset = 5u;
constant uint pm_rope_soft_velocity_offset = 6u;
constant uint pm_rope_soft_inverse_offset = 7u;
constant uint pm_rope_soft_index_offset = 8u;
constant uint pm_rope_soft_binding_offset = 9u;
constant uint pm_rope_soft_impulse_offset = 10u;
constant uint pm_rope_soft_contact_count = 11u;
constant uint pm_rope_soft_maximum_penetration = 12u;
constant uint pm_rope_soft_attach_first = 13u;
constant uint pm_rope_soft_attach_last = 14u;
constant uint pm_rope_soft_distance = 15u;
constant uint pm_rope_soft_friction = 16u;
constant uint pm_rope_soft_orientation = 17u;
constant uint pm_rope_soft_prior_contact_count = 18u;
constant uint pm_rope_soft_first_anchor_weight = 19u;
constant uint pm_rope_soft_last_anchor_weight = 20u;

static float pm_rope_soft_load_float(device const uint *packed, uint word) {
    return as_type<float>(packed[word]);
}

static float3 pm_rope_soft_load_vec3(device const uint *packed, uint word) {
    return float3(as_type<float>(packed[word]),
                  as_type<float>(packed[word + 1u]),
                  as_type<float>(packed[word + 2u]));
}

static void pm_rope_soft_store_vec3(device uint *packed, uint word,
                                    float3 value) {
    packed[word] = as_type<uint>(value.x);
    packed[word + 1u] = as_type<uint>(value.y);
    packed[word + 2u] = as_type<uint>(value.z);
}

static void pm_rope_soft_add_vec3(device uint *packed, uint word,
                                  float3 value) {
    pm_rope_soft_store_vec3(
        packed, word, pm_rope_soft_load_vec3(packed, word) + value);
}

static bool pm_rope_soft_inside_packed(
    float3 point, device const uint *packed, uint surface_offset,
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

kernel void pm_rope_soft_pack(
    device PMPackedVec3 *rope_positions [[buffer(0)]],
    device PMPackedVec3 *rope_previous [[buffer(1)]],
    device PMPackedVec3 *rope_velocities [[buffer(2)]],
    device const float *rope_inverse_masses [[buffer(3)]],
    device PMPackedVec3 *rope_forces [[buffer(4)]],
    device PMPackedVec3 *soft_positions [[buffer(5)]],
    device const PMPackedVec3 *soft_velocities [[buffer(6)]],
    device const float *soft_inverse_masses [[buffer(7)]],
    device const PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(10)]],
    device PMPackedVec3 *soft_forces [[buffer(11)]],
    device const PMPackedVec3 *soft_rest_positions [[buffer(12)]],
    device const PMPackedVec3 *surface_rest_positions [[buffer(13)]],
    constant PMRopeSoftConstants &constants [[buffer(14)]],
    device uint *diagnostic_contact_count [[buffer(15)]],
    device float *diagnostic_maximum_penetration [[buffer(16)]],
    device PMRopeAnchorState *anchors [[buffer(17)]],
    device uint *packed [[buffer(18)]],
    device const PMPackedVec3 *previous_surface [[buffer(19)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    (void)rope_positions;
    (void)rope_previous;
    (void)rope_velocities;
    (void)rope_inverse_masses;
    (void)rope_forces;
    (void)soft_positions;
    (void)soft_forces;
    (void)soft_rest_positions;
    (void)surface_rest_positions;
    (void)diagnostic_maximum_penetration;
    (void)anchors;
    if (constants.enabled == 0u) {
        if (thread_index == 0u) packed[pm_rope_soft_valid] = 0u;
        return;
    }
    const uint thread_count = threads_per_group.x;
    const uint surface_offset = pm_rope_soft_header_words;
    const uint previous_offset =
        surface_offset + 3u * constants.surface_count;
    const uint velocity_offset =
        previous_offset + 3u * constants.surface_count;
    const uint inverse_offset = velocity_offset + 3u * constants.soft_count;
    const uint index_offset = inverse_offset + constants.soft_count;
    const uint binding_offset = index_offset + constants.surface_index_count;
    const uint impulse_offset =
        binding_offset + 8u * constants.surface_count;
    if (thread_index == 0u) {
        packed[pm_rope_soft_valid] = 1u;
        packed[pm_rope_soft_node_count] = constants.soft_count;
        packed[pm_rope_soft_surface_count] = constants.surface_count;
        packed[pm_rope_soft_index_count] = constants.surface_index_count;
        packed[pm_rope_soft_surface_offset] = surface_offset;
        packed[pm_rope_soft_previous_offset] = previous_offset;
        packed[pm_rope_soft_velocity_offset] = velocity_offset;
        packed[pm_rope_soft_inverse_offset] = inverse_offset;
        packed[pm_rope_soft_index_offset] = index_offset;
        packed[pm_rope_soft_binding_offset] = binding_offset;
        packed[pm_rope_soft_impulse_offset] = impulse_offset;
        packed[pm_rope_soft_contact_count] = 0u;
        packed[pm_rope_soft_maximum_penetration] = as_type<uint>(0.0f);
        packed[pm_rope_soft_attach_first] = constants.attach_first;
        packed[pm_rope_soft_attach_last] = constants.attach_last;
        packed[pm_rope_soft_distance] =
            as_type<uint>(constants.contact_distance);
        packed[pm_rope_soft_friction] = as_type<uint>(constants.friction);
        packed[pm_rope_soft_orientation] =
            as_type<uint>(constants.orientation);
        packed[pm_rope_soft_prior_contact_count] =
            diagnostic_contact_count[0];
    }
    for (uint surface = thread_index; surface < constants.surface_count;
         surface += thread_count) {
        pm_rope_soft_store_vec3(
            packed, surface_offset + 3u * surface,
            pm_load(surface_positions[surface]));
        pm_rope_soft_store_vec3(
            packed, previous_offset + 3u * surface,
            pm_load(previous_surface[surface]));
        const PMMetalSurfaceBinding binding = surface_bindings[surface];
        const uint base = binding_offset + 8u * surface;
        for (uint slot = 0u; slot < 4u; ++slot) {
            packed[base + slot] = binding.nodes[slot];
            packed[base + 4u + slot] =
                as_type<uint>(binding.weights[slot]);
        }
    }
    for (uint node = thread_index; node < constants.soft_count;
         node += thread_count) {
        pm_rope_soft_store_vec3(
            packed, velocity_offset + 3u * node,
            pm_load(soft_velocities[node]));
        packed[inverse_offset + node] =
            as_type<uint>(soft_inverse_masses[node]);
        pm_rope_soft_store_vec3(
            packed, impulse_offset + 3u * node, 0.0f);
    }
    for (uint index = thread_index; index < constants.surface_index_count;
         index += thread_count)
        packed[index_offset + index] = surface_indices[index];
}

kernel void pm_rope_soft_sample(
    device PMPackedVec3 *rope_positions [[buffer(0)]],
    device PMPackedVec3 *rope_previous [[buffer(1)]],
    device PMPackedVec3 *rope_velocities [[buffer(2)]],
    device const float *rope_inverse_masses [[buffer(3)]],
    device PMPackedVec3 *rope_forces [[buffer(4)]],
    device PMPackedVec3 *soft_positions [[buffer(5)]],
    device const PMPackedVec3 *soft_velocities [[buffer(6)]],
    device const float *soft_inverse_masses [[buffer(7)]],
    device const PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(10)]],
    device PMPackedVec3 *soft_forces [[buffer(11)]],
    device const PMPackedVec3 *soft_rest_positions [[buffer(12)]],
    device const PMPackedVec3 *surface_rest_positions [[buffer(13)]],
    constant PMRopeSoftConstants &constants [[buffer(14)]],
    device uint *diagnostic_contact_count [[buffer(15)]],
    device float *diagnostic_maximum_penetration [[buffer(16)]],
    device PMRopeAnchorState *anchors [[buffer(17)]],
    uint thread_index [[thread_position_in_grid]]) {
    (void)rope_positions;
    (void)rope_previous;
    (void)rope_velocities;
    (void)rope_inverse_masses;
    (void)rope_forces;
    (void)soft_positions;
    (void)soft_inverse_masses;
    (void)soft_forces;
    (void)soft_rest_positions;
    (void)surface_rest_positions;
    (void)diagnostic_contact_count;
    (void)diagnostic_maximum_penetration;
    if (thread_index >= 2u || constants.enabled == 0u) return;
    const uint end = thread_index;
    const bool attached = end == 0u ? constants.attach_first != 0u
                                    : constants.attach_last != 0u;
    if (!attached) return;
    const uint triangle = end == 0u ? constants.first_triangle
                                    : constants.last_triangle;
    if (triangle == 0xffffffffu ||
        triangle + 2u >= constants.surface_index_count)
        return;
    const float3 weights = end == 0u
                               ? pm_load(constants.first_weights)
                               : pm_load(constants.last_weights);
    float3 position = end == 0u ? pm_load(constants.first_offset)
                                 : pm_load(constants.last_offset);
    float3 velocity = 0.0f;
    for (uint corner = 0u; corner < 3u; ++corner) {
        const float weight = weights[corner];
        const uint surface = surface_indices[triangle + corner];
        position += pm_load(surface_positions[surface]) * weight;
        const PMMetalSurfaceBinding binding = surface_bindings[surface];
        for (uint slot = 0u; slot < 4u; ++slot)
            velocity += pm_load(soft_velocities[binding.nodes[slot]]) *
                        (weight * binding.weights[slot]);
    }
    anchors[end].position = pm_store(position);
    anchors[end].velocity = pm_store(velocity);
    anchors[end].impulse = {0.0f, 0.0f, 0.0f};
    anchors[end].inverse_mass = 0.0f;
}

// Match fluid_closest_triangle_barycentric in the CUDA backend. Returning
// the closest point and its weights from the same region tests avoids a
// second projection whose rounding can move reaction weight between soft
// nodes at triangle edges.
static float3 pm_closest_point_triangle_weights(
    float3 point, float3 a, float3 b, float3 c,
    thread float3 &weights) {
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
    device uint *packed, uint node, bool segment, uint rope_count,
    float timestep, bool first_attached, bool last_attached,
    bool first_soft, bool last_soft, uint first_body, uint last_body,
    constant PMRopeAttachmentConstants &attachments,
    device PMPackedVec3 *positions, device const PMPackedVec3 *previous,
    device const float *inverse_masses,
    device const PMRopeAnchorState *soft_anchors,
    device const PMRigidParameters *rigid_parameters,
    device const PMRigidBodyState *rigid_states,
    device PMPackedVec3 *body_translation,
    device PMPackedVec3 *body_rotation,
    device PMPackedVec3 *contact_normals,
    device PMPackedVec3 *contact_normals2,
    device PMPackedVec3 *rope_soft_forces) {
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
    const float3 first = pm_load(positions[node]);
    const float3 second = pm_load(positions[next]);
    const float3 old = pm_load(previous[node]);
    const float3 query_lower =
        min(first, segment ? second : old) - radius;
    const float3 query_upper =
        max(first, segment ? second : old) + radius;
    float best_depth = 0.0f;
    float best_fraction = 0.0f;
    float3 best_normal = 0.0f;
    float3 best_weights = 0.0f;
    float3 best_surface_point = 0.0f;
    float nearest_squared = INFINITY;
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
        (pm_load(positions[node]) - pm_load(previous[node])) *
            first_fraction +
        (pm_load(positions[next]) - pm_load(previous[next])) *
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
    rope_soft_forces[node] = pm_store(
        pm_load(rope_soft_forces[node]) +
        impulse * (first_fraction /
            max(timestep * timestep, 1.0e-12f)));
    if (segment) {
        pm_rope_contact_move(
            next, rope_count, impulse * second_fraction, first_attached,
            last_attached, first_soft, last_soft, first_body, last_body,
            attachments, positions, inverse_masses, soft_anchors,
            rigid_parameters, rigid_states, body_translation, body_rotation);
        rope_soft_forces[next] = pm_store(
            pm_load(rope_soft_forces[next]) +
            impulse * (second_fraction /
                max(timestep * timestep, 1.0e-12f)));
    }
    if (soft_weight < 1.0e-9f) {
        for (uint item = node; item <= next; ++item) {
            const float3 first_normal = pm_load(contact_normals[item]);
            if (dot(first_normal, first_normal) < 0.5f ||
                dot(first_normal, best_normal) > 0.95f)
                contact_normals[item] = pm_store(best_normal);
            else
                contact_normals2[item] = pm_store(best_normal);
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
        as_type<uint>(max(maximum, best_depth));
}

kernel void pm_rope_predict(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device const float *inverse_masses [[buffer(3)]],
    constant PMDeformableConstants &constants [[buffer(5)]],
    constant PMRopeAttachmentConstants &attachments [[buffer(6)]],
    device const PMHandle *rigid_ids [[buffer(7)]],
    device const PMRigidBodyState *rigid_states [[buffer(8)]],
    device const PMRopeAnchorState *soft_anchors [[buffer(17)]],
    device const PMRigidBodyState *old_rigid_states [[buffer(22)]],
    device PMPackedVec3 *contact_normals [[buffer(23)]],
    device PMPackedVec3 *contact_normals2 [[buffer(24)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= constants.count) return;
    uint first_body = attachments.rigid_count;
    uint last_body = attachments.rigid_count;
    for (uint body = 0u; body < attachments.rigid_count; ++body) {
        if (rigid_ids[body].index == attachments.first_body.index &&
            rigid_ids[body].generation == attachments.first_body.generation)
            first_body = body;
        if (rigid_ids[body].index == attachments.last_body.index &&
            rigid_ids[body].generation == attachments.last_body.generation)
            last_body = body;
    }
    const bool first_attached = attachments.first_enabled != 0u &&
                                first_body < attachments.rigid_count;
    const bool last_attached = attachments.last_enabled != 0u &&
                               last_body < attachments.rigid_count;
    const bool first_soft = attachments.first_soft != 0u;
    const bool last_soft = attachments.last_soft != 0u;
    const bool first = index == 0u;
    const bool last = index + 1u == constants.count;

    // Preserve attached endpoints from the pre-integration transform in the
    // swept path, then synchronize them to the current transform.
    if (first && first_attached)
        positions[index] = pm_store(pm_world_point(
            old_rigid_states[first_body], pm_load(attachments.first_anchor)));
    if (last && last_attached)
        positions[index] = pm_store(pm_world_point(
            old_rigid_states[last_body], pm_load(attachments.last_anchor)));
    if (first && first_soft) {
        positions[index] = soft_anchors[0].position;
        velocities[index] = soft_anchors[0].velocity;
    }
    if (last && last_soft) {
        positions[index] = soft_anchors[1].position;
        velocities[index] = soft_anchors[1].velocity;
    }
    previous[index] = positions[index];
    contact_normals[index] = {0.0f, 0.0f, 0.0f};
    contact_normals2[index] = {0.0f, 0.0f, 0.0f};
    const bool attached =
        (first && (first_attached || first_soft)) ||
        (last && (last_attached || last_soft));
    if (!attached && inverse_masses[index] > 0.0f) {
        float3 velocity = pm_load(velocities[index]) +
                          pm_load(constants.gravity) * constants.timestep;
        velocity *= exp(-constants.velocity_damping * constants.timestep);
        velocity = pm_limit(velocity, constants.maximum_speed);
        velocities[index] = pm_store(velocity);
        positions[index] = pm_store(
            pm_load(positions[index]) + velocity * constants.timestep);
    }
    if (first && first_attached) {
        positions[index] = pm_store(pm_world_point(
            rigid_states[first_body], pm_load(attachments.first_anchor)));
        velocities[index] = pm_store(pm_rigid_point_velocity(
            rigid_states[first_body], pm_load(attachments.first_anchor)));
    }
    if (last && last_attached) {
        positions[index] = pm_store(pm_world_point(
            rigid_states[last_body], pm_load(attachments.last_anchor)));
        velocities[index] = pm_store(pm_rigid_point_velocity(
            rigid_states[last_body], pm_load(attachments.last_anchor)));
    }
    if (first && first_soft) {
        positions[index] = soft_anchors[0].position;
        velocities[index] = soft_anchors[0].velocity;
    }
    if (last && last_soft) {
        positions[index] = soft_anchors[1].position;
        velocities[index] = soft_anchors[1].velocity;
    }
}

kernel void pm_rope_step_serial(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *previous [[buffer(1)]],
    device PMPackedVec3 *velocities [[buffer(2)]],
    device const float *inverse_masses [[buffer(3)]],
    device PMMetalBond *bonds [[buffer(4)]],
    constant PMDeformableConstants &constants [[buffer(5)]],
    constant PMRopeAttachmentConstants &attachments [[buffer(6)]],
    device const PMHandle *rigid_ids [[buffer(7)]],
    device PMRigidBodyState *rigid_states [[buffer(8)]],
    device PMPackedVec3 *directions [[buffer(9)]],
    device float *diagonal [[buffer(10)]],
    device float *upper [[buffer(11)]],
    device float *rhs [[buffer(12)]],
    device float *lambdas [[buffer(13)]],
    device PMPackedVec3 *scratch [[buffer(14)]],
    device PMPackedVec3 *constraint_forces [[buffer(15)]],
    device PMRigidParameters *rigid_parameters [[buffer(16)]],
    device PMRopeAnchorState *soft_anchors [[buffer(17)]],
    device PMPackedVec3 *contact_forces [[buffer(18)]],
    device const PMPackedVec3 *mesh_vertices [[buffer(19)]],
    device const uint *mesh_indices [[buffer(20)]],
    device const PMTriangleMeshInfo *meshes [[buffer(21)]],
    device const PMRigidBodyState *old_rigid_states [[buffer(22)]],
    device PMPackedVec3 *contact_normals [[buffer(23)]],
    device PMPackedVec3 *contact_normals2 [[buffer(24)]],
    device PMPackedVec3 *body_translation [[buffer(25)]],
    device PMPackedVec3 *body_rotation [[buffer(26)]],
    device uint *soft_target_a [[buffer(27)]],
    device uint *soft_target_b [[buffer(28)]],
    device PMPackedVec3 *rope_soft_forces [[buffer(29)]],
    device const PMCollisionPlane *solid_planes [[buffer(30)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u) return;
    device uint *solid_hints = reinterpret_cast<device uint *>(
        body_rotation + attachments.rigid_capacity);
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
            const float3 delta = pm_load(positions[edge + 1u]) -
                                 pm_load(positions[edge]);
            directions[edge] = pm_store(
                dot(delta, delta) > 1.0e-12f
                    ? normalize(delta)
                    : float3(1.0f, 0.0f, 0.0f));
        }
        // Contact planes are unilateral constraints.  Rebuild the reduced
        // tridiagonal system when the tentative stretch impulse points away
        // from one, matching CUDA's active-set release before moving anything.
        for (uint active_pass = 0u; active_pass < 4u; ++active_pass) {
            for (uint edge = 0u; edge < edges; ++edge) {
                const float3 direction = pm_load(directions[edge]);
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
                    distance(pm_load(positions[edge + 1u]),
                             pm_load(positions[edge])) -
                    bonds[edge].rest_length;
                rhs[edge] = -length_error - alpha * lambdas[edge];
                diagonal[edge] = first_weight + second_weight + alpha;
                if (edge + 1u < edges) {
                    upper[edge] = -dot(
                        pm_load(directions[edge]),
                        pm_rope_projected_mass(
                            edge + 1u,
                            pm_load(directions[edge + 1u]),
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
                        pm_load(directions[node - 1u]) * rhs[node - 1u];
                if (node < edges)
                    impulse -= pm_load(directions[node]) * rhs[node];
                scratch[node] = pm_store(impulse);
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
                        pm_load(contact_normals[node]);
                    if (dot(impulse, first_normal) > release_impulse) {
                        contact_normals[node] = {0.0f, 0.0f, 0.0f};
                        released = true;
                    }
                    const float3 second_normal =
                        pm_load(contact_normals2[node]);
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
            const float3 impulse = pm_load(scratch[node]);
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
                    pm_load(attachments.first_anchor), impulse);
            if (node + 1u == constants.count && last_attached)
                movement = pm_rigid_impulse_movement(
                    rigid_parameters[last_body], rigid_states[last_body],
                    pm_load(attachments.last_anchor), impulse);
            maximum_movement = max(maximum_movement, length(movement));
        }
        const float scale = min(
            1.0f, constants.radius / max(maximum_movement, 1.0e-10f));
        for (uint edge = 0u; edge < edges; ++edge)
            lambdas[edge] += scale * rhs[edge];
        for (uint node = 0u; node < constants.count; ++node) {
            const float3 impulse = pm_load(scratch[node]) * scale;
            constraint_forces[node] = pm_store(
                pm_load(constraint_forces[node]) +
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
                positions[node] = pm_store(pm_load(positions[node]) +
                    pm_rope_projected_mass(
                        node, impulse, node_inverse, contact_normals,
                        contact_normals2));
            if (soft_end >= 0) {
                soft_anchors[uint(soft_end)].impulse = pm_store(
                    pm_load(soft_anchors[uint(soft_end)].impulse) +
                    impulse / max(constants.timestep, 1.0e-12f));
                if (node_inverse > 0.0f)
                    soft_anchors[uint(soft_end)].position = positions[node];
            }
        }
        if (first_attached)
            pm_move_rigid_by_rope(
                rigid_parameters[first_body], rigid_states[first_body],
                pm_load(attachments.first_anchor), pm_load(scratch[0]) * scale,
                constants.timestep);
        if (last_attached)
            pm_move_rigid_by_rope(
                rigid_parameters[last_body], rigid_states[last_body],
                pm_load(attachments.last_anchor),
                pm_load(scratch[constants.count - 1u]) * scale,
                constants.timestep);
        if (first_attached)
            positions[0] = pm_store(pm_world_point(
                rigid_states[first_body], pm_load(attachments.first_anchor)));
        if (last_attached)
            positions[constants.count - 1u] = pm_store(pm_world_point(
                rigid_states[last_body], pm_load(attachments.last_anchor)));
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
                body_rotation[body] = {0.0f, 0.0f, 0.0f};
            }
            const uint begin = phase == 0u ? 0u : phase - 1u;
            const uint stride = phase == 0u ? 1u : 2u;
            for (uint node = begin;
                 node < (phase == 0u ? constants.count : edges);
                 node += stride) {
                const bool segment = phase != 0u;
                const PMRopeRigidHit hit = pm_rope_find_rigid_contact(
                    positions, previous, constants.count, node, segment,
                    constants.radius, first_body, last_body, attachments,
                    rigid_states, old_rigid_states, rigid_parameters,
                    mesh_vertices, mesh_indices, meshes, solid_planes,
                    solid_hints);
                if (hit.found) {
                const uint next = segment ? node + 1u : node;
                const float first_fraction = 1.0f - hit.fraction;
                const float second_fraction = hit.fraction;
                const float first_weight = pm_rope_contact_weight(
                    node, constants.count, hit.normal, first_attached,
                    last_attached, first_soft, last_soft, first_body,
                    last_body, attachments, inverse_masses, soft_anchors,
                    rigid_parameters, rigid_states);
                const float second_weight = segment
                    ? pm_rope_contact_weight(
                          next, constants.count, hit.normal,
                          first_attached, last_attached, first_soft,
                          last_soft, first_body, last_body, attachments,
                          inverse_masses, soft_anchors, rigid_parameters,
                          rigid_states)
                    : 0.0f;
                const float3 body_arm =
                    hit.point - pm_load(rigid_states[hit.body].position);
                const float3 body_torque = cross(body_arm, hit.normal);
                const float body_weight =
                    rigid_parameters[hit.body].inverse_mass +
                    dot(body_torque,
                        pm_inverse_inertia_mul(
                            rigid_parameters[hit.body],
                            rigid_states[hit.body], body_torque));
                const float denominator =
                    first_weight * first_fraction * first_fraction +
                    second_weight * second_fraction * second_fraction +
                    body_weight;
                if (denominator > 1.0e-12f) {
                float3 impulse = hit.normal * (hit.depth / denominator);
                const float3 rope_motion =
                    (pm_load(positions[node]) - pm_load(previous[node])) *
                        first_fraction +
                    (pm_load(positions[next]) - pm_load(previous[next])) *
                        second_fraction;
                const float3 body_velocity =
                    pm_load(rigid_states[hit.body].linear_velocity) +
                    cross(pm_load(rigid_states[hit.body].angular_velocity),
                          body_arm);
                const float3 relative_motion =
                    rope_motion - body_velocity * constants.timestep;
                const float3 tangent = relative_motion -
                    hit.normal * dot(relative_motion, hit.normal);
                const float tangent_length = length(tangent);
                if (tangent_length > 1.0e-8f) {
                    const float3 direction = tangent / tangent_length;
                    const float tangent_first = pm_rope_contact_weight(
                        node, constants.count, direction, first_attached,
                        last_attached, first_soft, last_soft, first_body,
                        last_body, attachments, inverse_masses,
                        soft_anchors, rigid_parameters, rigid_states);
                    const float tangent_second = segment
                        ? pm_rope_contact_weight(
                              next, constants.count, direction,
                              first_attached, last_attached, first_soft,
                              last_soft, first_body, last_body, attachments,
                              inverse_masses, soft_anchors,
                              rigid_parameters, rigid_states)
                        : 0.0f;
                    const float3 axis = cross(body_arm, direction);
                    const float tangent_denominator =
                        tangent_first * first_fraction * first_fraction +
                        tangent_second * second_fraction * second_fraction +
                        rigid_parameters[hit.body].inverse_mass +
                        dot(axis, pm_inverse_inertia_mul(
                                      rigid_parameters[hit.body],
                                      rigid_states[hit.body], axis));
                    impulse -= direction * min(
                        tangent_length / max(tangent_denominator, 1.0e-12f),
                        constants.spring_damping * hit.depth / denominator);
                }
                pm_rope_contact_move(
                    node, constants.count, impulse * first_fraction,
                    first_attached, last_attached, first_soft, last_soft,
                    first_body, last_body, attachments, positions,
                    inverse_masses, soft_anchors, rigid_parameters,
                    rigid_states, body_translation, body_rotation);
                contact_forces[node] = pm_store(
                    pm_load(contact_forces[node]) +
                    impulse * (first_fraction /
                        max(constants.timestep * constants.timestep,
                            1.0e-12f)));
                if (segment) {
                    pm_rope_contact_move(
                        next, constants.count, impulse * second_fraction,
                        first_attached, last_attached, first_soft, last_soft,
                        first_body, last_body, attachments, positions,
                        inverse_masses, soft_anchors, rigid_parameters,
                        rigid_states, body_translation, body_rotation);
                    contact_forces[next] = pm_store(
                        pm_load(contact_forces[next]) +
                        impulse * (second_fraction /
                            max(constants.timestep * constants.timestep,
                                1.0e-12f)));
                }
                if (rigid_parameters[hit.body].inverse_mass == 0.0f) {
                    for (uint item = node; item <= next; ++item) {
                        const float3 first_normal =
                            pm_load(contact_normals[item]);
                        if (dot(first_normal, first_normal) < 0.5f ||
                            dot(first_normal, hit.normal) > 0.95f)
                            contact_normals[item] = pm_store(hit.normal);
                        else
                            contact_normals2[item] = pm_store(hit.normal);
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
                const float3 translation = pm_load(body_translation[body]);
                const float3 rotation = pm_load(body_rotation[body]);
                rigid_states[body].position = pm_store(
                    pm_load(rigid_states[body].position) + translation);
                const PMQuaternion spin = pm_quaternion_multiply(
                    {rotation.x, rotation.y, rotation.z, 0.0f},
                    rigid_states[body].orientation);
                rigid_states[body].orientation = pm_quaternion_normalize(
                    {fma(0.5f, spin.x,
                         rigid_states[body].orientation.x),
                     fma(0.5f, spin.y,
                         rigid_states[body].orientation.y),
                     fma(0.5f, spin.z,
                         rigid_states[body].orientation.z),
                     fma(0.5f, spin.w,
                         rigid_states[body].orientation.w)});
                rigid_states[body].linear_velocity = pm_store(
                    pm_load(rigid_states[body].linear_velocity) +
                    translation / max(constants.timestep, 1.0e-12f));
                rigid_states[body].angular_velocity = pm_store(
                    pm_load(rigid_states[body].angular_velocity) +
                    rotation / max(constants.timestep, 1.0e-12f));
            }
            if (first_attached)
                positions[0] = pm_store(pm_world_point(
                    rigid_states[first_body],
                    pm_load(attachments.first_anchor)));
            if (last_attached)
                positions[constants.count - 1u] = pm_store(pm_world_point(
                    rigid_states[last_body],
                    pm_load(attachments.last_anchor)));
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
                    const float3 delta = pm_load(positions[first]) -
                                         pm_load(positions[second]);
                    const float distance_value = length(delta);
                    const float inverse_sum = first_inverse + second_inverse;
                    if (distance_value >= diameter ||
                        inverse_sum <= 0.0f)
                        continue;
                    correction += pm_normalized_or(
                        delta, float3(1.0f, 0.0f, 0.0f)) *
                        ((diameter - distance_value) * first_inverse /
                         inverse_sum);
                    ++hits;
                }
                scratch[first] = pm_store(
                    hits != 0u ? correction / float(hits) : 0.0f);
            }
            for (uint node = 0u; node < constants.count; ++node)
                positions[node] = pm_store(
                    pm_load(positions[node]) + pm_load(scratch[node]));
        }
        float strain = 0.0f;
        for (uint edge = 0u; edge < edges; ++edge)
            strain = max(
                strain,
                abs(distance(pm_load(positions[edge + 1u]),
                             pm_load(positions[edge])) /
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
            velocities[node] = pm_store(pm_rigid_point_velocity(
                rigid_states[first_body], pm_load(attachments.first_anchor)));
            continue;
        }
        if (node + 1u == constants.count && last_attached) {
            velocities[node] = pm_store(pm_rigid_point_velocity(
                rigid_states[last_body], pm_load(attachments.last_anchor)));
            continue;
        }
        int soft_end = -1;
        if (node == 0u && first_soft) soft_end = 0;
        if (node + 1u == constants.count && last_soft) soft_end = 1;
        if (soft_end >= 0) {
            if (soft_anchors[uint(soft_end)].inverse_mass > 0.0f) {
                velocities[node] = pm_store(pm_limit(
                    (pm_load(positions[node]) - pm_load(previous[node])) /
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
        velocities[node] = pm_store(pm_limit(
            (pm_load(positions[node]) - pm_load(previous[node])) /
                constants.timestep,
            constants.maximum_speed));
    }
    // Remove axial velocity using the same tridiagonal mass matrix so a taut
    // chain does not store residual stretch as kinetic energy.
    for (uint edge = 0u; edge < edges; ++edge) {
        const float3 delta = pm_load(positions[edge + 1u]) -
                             pm_load(positions[edge]);
        directions[edge] = pm_store(dot(delta, delta) > 1.0e-12f
                                        ? normalize(delta)
                                        : float3(1.0f, 0.0f, 0.0f));
        const float3 direction = pm_load(directions[edge]);
        const float first_weight = edge == 0u && first_attached
            ? pm_rigid_direction_weight(
                  rigid_parameters[first_body], rigid_states[first_body],
                  pm_load(attachments.first_anchor), direction)
            : (edge == 0u && first_soft
                   ? soft_anchors[0].inverse_mass
                   : inverse_masses[edge]);
        const float second_weight =
            (edge + 2u == constants.count && last_attached)
                ? pm_rigid_direction_weight(
                      rigid_parameters[last_body], rigid_states[last_body],
                      pm_load(attachments.last_anchor), direction)
                : (edge + 2u == constants.count && last_soft
                       ? soft_anchors[1].inverse_mass
                       : inverse_masses[edge + 1u]);
        diagonal[edge] = first_weight + second_weight + alpha;
        rhs[edge] = -dot(pm_load(directions[edge]),
                         pm_load(velocities[edge + 1u]) -
                             pm_load(velocities[edge]));
        if (edge + 1u < edges) {
            const float shared_weight = inverse_masses[edge + 1u];
            upper[edge] = -dot(pm_load(directions[edge]),
                               pm_load(directions[edge + 1u])) *
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
            impulse += pm_load(directions[node - 1u]) * rhs[node - 1u];
        if (node < edges)
            impulse -= pm_load(directions[node]) * rhs[node];
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
            velocities[node] = pm_store(pm_limit(
                pm_load(velocities[node]) + impulse * node_inverse,
                constants.maximum_speed));
        if (soft_end >= 0) {
            soft_anchors[uint(soft_end)].impulse = pm_store(
                pm_load(soft_anchors[uint(soft_end)].impulse) + impulse);
            if (node_inverse > 0.0f)
                soft_anchors[uint(soft_end)].velocity = velocities[node];
            else
                velocities[node] = soft_anchors[uint(soft_end)].velocity;
        }
        constraint_forces[node] = pm_store(
            pm_load(constraint_forces[node]) +
            impulse / max(constants.timestep, 1.0e-12f));
    }
    if (first_attached) {
        const float3 impulse =
            pm_load(directions[0]) * -rhs[0];
        rigid_states[first_body].linear_velocity = pm_store(
            pm_load(rigid_states[first_body].linear_velocity) +
            impulse * rigid_parameters[first_body].inverse_mass);
        const float3 arm = pm_rotate(rigid_states[first_body].orientation,
                                     pm_load(attachments.first_anchor));
        rigid_states[first_body].angular_velocity = pm_store(
            pm_load(rigid_states[first_body].angular_velocity) +
            pm_inverse_inertia_mul(rigid_parameters[first_body],
                                   rigid_states[first_body],
                                   cross(arm, impulse)));
    }
    if (last_attached) {
        const float3 impulse =
            pm_load(directions[edges - 1u]) * rhs[edges - 1u];
        rigid_states[last_body].linear_velocity = pm_store(
            pm_load(rigid_states[last_body].linear_velocity) +
            impulse * rigid_parameters[last_body].inverse_mass);
        const float3 arm = pm_rotate(rigid_states[last_body].orientation,
                                     pm_load(attachments.last_anchor));
        rigid_states[last_body].angular_velocity = pm_store(
            pm_load(rigid_states[last_body].angular_velocity) +
            pm_inverse_inertia_mul(rigid_parameters[last_body],
                                   rigid_states[last_body],
                                   cross(arm, impulse)));
    }
    if (first_attached) {
        positions[0] = pm_store(pm_world_point(
            rigid_states[first_body], pm_load(attachments.first_anchor)));
        velocities[0] = pm_store(pm_rigid_point_velocity(
            rigid_states[first_body], pm_load(attachments.first_anchor)));
    }
    if (last_attached) {
        positions[constants.count - 1u] = pm_store(pm_world_point(
            rigid_states[last_body], pm_load(attachments.last_anchor)));
        velocities[constants.count - 1u] = pm_store(pm_rigid_point_velocity(
            rigid_states[last_body], pm_load(attachments.last_anchor)));
    }
}

kernel void pm_smoke_surface_grid_raster(
    device const PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    constant PMSmokeSurfaceConstants &constants [[buffer(12)]],
    device atomic_uint *cell_nearest_triangle [[buffer(14)]],
    device atomic_uint *face_boundary [[buffer(15)]],
    uint triangle [[thread_position_in_grid]]) {
    if (triangle >= constants.surface_index_count / 3u ||
        constants.enabled == 0u ||
        constants.grid_resolution == 0u || constants.grid_spacing <= 0.0f)
        return;
    const uint base = triangle * 3u;
    const float3 a = pm_load(surface_positions[surface_indices[base]]);
    const float3 b = pm_load(surface_positions[surface_indices[base + 1u]]);
    const float3 c = pm_load(surface_positions[surface_indices[base + 2u]]);
    pm_smoke_raster_triangle_mark(
        a, b, c, constants.raster_triangle_base + triangle,
        constants.grid_resolution, constants.grid_vertical_resolution,
        constants.grid_spacing, pm_load(constants.grid_minimum),
        cell_nearest_triangle, face_boundary);
}

kernel void pm_smoke_surface_grid_select(
    device const PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    constant PMSmokeSurfaceConstants &constants [[buffer(12)]],
    device const float *face_boundary [[buffer(15)]],
    device atomic_uint *face_nearest_triangle [[buffer(20)]],
    uint triangle [[thread_position_in_grid]]) {
    if (triangle >= constants.surface_index_count / 3u ||
        constants.enabled == 0u || constants.grid_resolution == 0u ||
        constants.grid_spacing <= 0.0f)
        return;
    const uint base = triangle * 3u;
    pm_smoke_select_triangle_faces(
        pm_load(surface_positions[surface_indices[base]]),
        pm_load(surface_positions[surface_indices[base + 1u]]),
        pm_load(surface_positions[surface_indices[base + 2u]]),
        constants.raster_triangle_base + triangle,
        constants.grid_resolution, constants.grid_vertical_resolution,
        constants.grid_spacing, pm_load(constants.grid_minimum),
        face_boundary, face_nearest_triangle);
}

static float3 pm_smoke_surface_vertex_velocity(
    uint surface, uint mode,
    device const PMPackedVec3 *target_velocities,
    device const uint *surface_sources,
    device const PMMetalSurfaceBinding *surface_bindings) {
    if (mode == 0u)
        return pm_load(target_velocities[surface_sources[surface]]);
    const PMMetalSurfaceBinding binding = surface_bindings[surface];
    float3 velocity = 0.0f;
    for (uint slot = 0u; slot < 4u; ++slot)
        velocity += pm_load(target_velocities[binding.nodes[slot]]) *
                    binding.weights[slot];
    return velocity;
}

kernel void pm_smoke_surface_grid_resolve(
    device const PMPackedVec3 *target_velocities [[buffer(6)]],
    device const PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    device const uint *surface_sources [[buffer(10)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(11)]],
    constant PMSmokeSurfaceConstants &constants [[buffer(12)]],
    device PMPackedVec3 *grid_wall_velocity [[buffer(13)]],
    device const uint *cell_nearest_triangle [[buffer(14)]],
    device float *face_boundary [[buffer(15)]],
    device const uint *face_nearest_triangle [[buffer(20)]],
    device PMPackedVec3 *face_normal [[buffer(21)]],
    uint triangle [[thread_position_in_grid]]) {
    if (triangle >= constants.surface_index_count / 3u ||
        constants.enabled == 0u || constants.grid_resolution == 0u ||
        constants.grid_spacing <= 0.0f)
        return;
    const uint base = triangle * 3u;
    const uint ia = surface_indices[base];
    const uint ib = surface_indices[base + 1u];
    const uint ic = surface_indices[base + 2u];
    const float3 a = pm_load(surface_positions[ia]);
    const float3 b = pm_load(surface_positions[ib]);
    const float3 c = pm_load(surface_positions[ic]);
    const uint global_triangle = constants.raster_triangle_base + triangle;
    const uint resolution = constants.grid_resolution;
    const uint vertical = constants.grid_vertical_resolution;
    const float spacing = constants.grid_spacing;
    const float inverse_spacing = 1.0f / spacing;
    const float radius = 0.55f * spacing;
    const float radius_squared = radius * radius;
    const float3 minimum = pm_load(constants.grid_minimum);
    const float3 lower = min(a, min(b, c)) - radius;
    const float3 upper = max(a, max(b, c)) + radius;
    const int x0 = max(0, int(floor(
        (lower.x - minimum.x) * inverse_spacing)));
    const int y0 = max(0, int(floor(
        (lower.y - minimum.y) * inverse_spacing)));
    const int z0 = max(0, int(floor(
        (lower.z - minimum.z) * inverse_spacing)));
    const int x1 = min(int(resolution) - 1, int(floor(
        (upper.x - minimum.x) * inverse_spacing)));
    const int y1 = min(int(vertical) - 1, int(floor(
        (upper.y - minimum.y) * inverse_spacing)));
    const int z1 = min(int(resolution) - 1, int(floor(
        (upper.z - minimum.z) * inverse_spacing)));
    const uint vertices[3] = {ia, ib, ic};
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                const uint cell = uint(x) + resolution *
                    (uint(y) + vertical * uint(z));
                if (cell_nearest_triangle[cell] != global_triangle) continue;
                const float3 point = minimum +
                    (float3(float(x), float(y), float(z)) + 0.5f) * spacing;
                const float3 closest = pm_closest_point_triangle(point, a, b, c);
                const float3 delta = point - closest;
                if (dot(delta, delta) > radius_squared) continue;
                const float3 weights = pm_triangle_weights(closest, a, b, c);
                const float corner_weights[3] = {
                    weights.x, weights.y, weights.z};
                float3 velocity = 0.0f;
                for (uint corner = 0u; corner < 3u; ++corner)
                    velocity += pm_smoke_surface_vertex_velocity(
                        vertices[corner], constants.mode, target_velocities,
                        surface_sources, surface_bindings) *
                        corner_weights[corner];
                grid_wall_velocity[cell] = pm_store(velocity);
            }
    const uint face_total = pm_smoke_face_total(resolution, vertical);
    const float3 normal = pm_normalized_or(
        cross(b - a, c - a), float3(0.0f, 1.0f, 0.0f));
    for (uint axis = 0u; axis < 3u; ++axis) {
        float3 offset = 0.5f;
        if (axis == 0u) offset.x = 0.0f;
        if (axis == 1u) offset.y = 0.0f;
        if (axis == 2u) offset.z = 0.0f;
        const int sx = int(resolution) + (axis == 0u ? 1 : 0);
        const int sy = int(vertical) + (axis == 1u ? 1 : 0);
        const int sz = int(resolution) + (axis == 2u ? 1 : 0);
        const int fx0 = max(0, int(floor(
            (lower.x - minimum.x) * inverse_spacing - offset.x)));
        const int fy0 = max(0, int(floor(
            (lower.y - minimum.y) * inverse_spacing - offset.y)));
        const int fz0 = max(0, int(floor(
            (lower.z - minimum.z) * inverse_spacing - offset.z)));
        const int fx1 = min(sx - 1, int(floor(
            (upper.x - minimum.x) * inverse_spacing - offset.x)));
        const int fy1 = min(sy - 1, int(floor(
            (upper.y - minimum.y) * inverse_spacing - offset.y)));
        const int fz1 = min(sz - 1, int(floor(
            (upper.z - minimum.z) * inverse_spacing - offset.z)));
        for (int z = fz0; z <= fz1; ++z)
            for (int y = fy0; y <= fy1; ++y)
                for (int x = fx0; x <= fx1; ++x) {
                    const uint face = pm_smoke_face_index(
                        axis, x, y, z, resolution, vertical);
                    if (face_nearest_triangle[face] != global_triangle) continue;
                    const float3 point = minimum +
                        (float3(float(x), float(y), float(z)) + offset) * spacing;
                    const float3 closest = pm_closest_point_triangle(point, a, b, c);
                    const float3 weights = pm_triangle_weights(closest, a, b, c);
                    const float corner_weights[3] = {
                        weights.x, weights.y, weights.z};
                    float3 velocity = 0.0f;
                    for (uint corner = 0u; corner < 3u; ++corner)
                        velocity += pm_smoke_surface_vertex_velocity(
                            vertices[corner], constants.mode, target_velocities,
                            surface_sources, surface_bindings) *
                            corner_weights[corner];
                    face_boundary[face_total + face] =
                        axis == 0u ? velocity.x
                        : axis == 1u ? velocity.y
                                     : velocity.z;
                    face_normal[face] = pm_store(normal);
                }
    }
}

kernel void pm_smoke_surface_grid_force(
    device PMPackedVec3 *target_velocities [[buffer(6)]],
    device const float *target_inverse_masses [[buffer(7)]],
    device const PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    device const uint *surface_sources [[buffer(10)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(11)]],
    constant PMSmokeSurfaceConstants &constants [[buffer(12)]],
    device const float *grid_pressure [[buffer(16)]],
    device const float *grid_density [[buffer(17)]],
    device const float *grid_face_velocity [[buffer(18)]],
    device PMPackedVec3 *target_forces [[buffer(19)]],
    uint target [[thread_position_in_grid]]) {
    if (target >= constants.target_count || constants.enabled == 0u ||
        constants.grid_resolution == 0u || constants.grid_spacing <= 0.0f)
        return;
    float3 accumulated_force = 0.0f;

    const uint resolution = constants.grid_resolution;
    const uint vertical = constants.grid_vertical_resolution;
    const float spacing = constants.grid_spacing;
    const float3 minimum = pm_load(constants.grid_minimum);
    const float3 maximum = minimum + spacing *
        float3(float(resolution), float(vertical), float(resolution));
    const float surface_offset = 1.5f * spacing;
    for (uint base = 0u; base + 2u < constants.surface_index_count;
         base += 3u) {
        const uint ia = surface_indices[base];
        const uint ib = surface_indices[base + 1u];
        const uint ic = surface_indices[base + 2u];
        const float3 a = pm_load(surface_positions[ia]);
        const float3 b = pm_load(surface_positions[ib]);
        const float3 c = pm_load(surface_positions[ic]);
        const float3 twice_area = cross(b - a, c - a);
        const float area = 0.5f * length(twice_area);
        if (area <= 1.0e-9f) continue;
        const float3 normal = normalize(twice_area);
        const float3 center = (a + b + c) / 3.0f;
        const float3 plus = center + normal * surface_offset;
        const float3 minus = center - normal * surface_offset;
        if (any(plus < minimum) || any(plus >= maximum) ||
            any(minus < minimum) || any(minus >= maximum))
            continue;
        const float plus_density = constants.rest_number_density *
            pm_smoke_sample_cell_scalar(
                grid_density, plus, minimum, spacing, resolution, vertical);
        const float minus_density = constants.rest_number_density *
            pm_smoke_sample_cell_scalar(
                grid_density, minus, minimum, spacing, resolution, vertical);
        if (plus_density + minus_density < 1.0e-4f) continue;
        const float3 plus_air = pm_smoke_sample_face_velocity(
            grid_face_velocity, plus, minimum, spacing, resolution, vertical);
        const float3 minus_air = pm_smoke_sample_face_velocity(
            grid_face_velocity, minus, minimum, spacing, resolution, vertical);

        const uint corners[3] = {ia, ib, ic};
        float3 body_velocity = 0.0f;
        for (uint corner = 0u; corner < 3u; ++corner) {
            if (constants.mode == 0u) {
                const uint node = surface_sources[corners[corner]];
                if (node < constants.target_count)
                    body_velocity += pm_load(target_velocities[node]);
            } else {
                const PMMetalSurfaceBinding binding =
                    surface_bindings[corners[corner]];
                for (uint slot = 0u; slot < 4u; ++slot) {
                    const uint node = binding.nodes[slot];
                    if (node < constants.target_count)
                        body_velocity += pm_load(target_velocities[node]) *
                                         binding.weights[slot];
                }
            }
        }
        body_velocity /= 3.0f;
        const float3 plus_relative = plus_air - body_velocity;
        const float3 minus_relative = minus_air - body_velocity;
        const float3 plus_tangent =
            plus_relative - normal * dot(plus_relative, normal);
        const float3 minus_tangent =
            minus_relative - normal * dot(minus_relative, normal);
        const float pressure_difference =
            minus_density * pm_smoke_sample_cell_scalar(
                grid_pressure, minus, minimum, spacing, resolution,
                vertical) -
            plus_density * pm_smoke_sample_cell_scalar(
                grid_pressure, plus, minimum, spacing, resolution,
                vertical);
        const float strain = pm_smoke_sample_face_strain(
            grid_face_velocity, center, minimum, spacing, resolution,
            vertical);
        const float viscosity = constants.grid_kinematic_viscosity +
            constants.grid_les_coefficient * constants.grid_les_coefficient *
                spacing * spacing * strain;
        const float3 force =
            (normal * pressure_difference +
             (plus_tangent * plus_density +
              minus_tangent * minus_density) *
                 (constants.wind_drag * viscosity / surface_offset)) * area;
        for (uint corner = 0u; corner < 3u; ++corner) {
            if (constants.mode == 0u) {
                const uint node = surface_sources[corners[corner]];
                if (node == target) accumulated_force += force / 3.0f;
            } else {
                const PMMetalSurfaceBinding binding =
                    surface_bindings[corners[corner]];
                for (uint slot = 0u; slot < 4u; ++slot) {
                    const uint node = binding.nodes[slot];
                    const float weight = binding.weights[slot];
                    if (node == target && weight > 0.0f)
                        accumulated_force += force * (weight / 3.0f);
                }
            }
        }
    }

    target_forces[target] = pm_store(accumulated_force);
    const float inverse_mass = target_inverse_masses[target];
    if (inverse_mass <= 0.0f) return;
    const float3 delta = pm_limit(
        accumulated_force * (inverse_mass * constants.timestep),
        constants.maximum_wind_acceleration * constants.timestep);
    target_velocities[target] = pm_store(pm_limit(
        pm_load(target_velocities[target]) + delta,
        constants.maximum_target_speed));
}

kernel void pm_smoke_surface(
    device PMPackedVec3 *smoke_positions [[buffer(0)]],
    device const PMPackedVec3 *smoke_previous [[buffer(1)]],
    device PMPackedVec3 *smoke_velocities [[buffer(2)]],
    device const float *smoke_ages [[buffer(3)]],
    device const PMSmokeMetadata &smoke_metadata [[buffer(4)]],
    device const PMPackedVec3 *target_positions [[buffer(5)]],
    device PMPackedVec3 *target_velocities [[buffer(6)]],
    device const float *target_inverse_masses [[buffer(7)]],
    device const PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    device const uint *surface_sources [[buffer(10)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(11)]],
    constant PMSmokeSurfaceConstants &constants [[buffer(12)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (constants.enabled == 0u) return;
    const uint smoke_count = smoke_metadata.count;
    const float wind_squared = constants.wind_radius * constants.wind_radius;
    if (constants.command != 2u && constants.grid_resolution == 0u) {
        const uint target = thread_index;
        if (target >= constants.target_count ||
            target_inverse_masses[target] <= 0.0f)
            return;
        float3 average = 0.0f;
        float density = 0.0f;
        for (uint particle = 0u; particle < smoke_count; ++particle) {
            if (smoke_ages[particle] >= constants.lifetime) continue;
            const float3 delta = pm_load(smoke_positions[particle]) -
                                 pm_load(target_positions[target]);
            const float squared = dot(delta, delta);
            if (squared >= wind_squared) continue;
            const float q = sqrt(max(squared, 0.0f)) /
                            max(constants.wind_radius, 1.0e-6f);
            const float weight = (1.0f - q) * (1.0f - q) * (1.0f - q);
            average += pm_load(smoke_velocities[particle]) * weight;
            density += weight;
        }
        if (density <= 1.0e-6f) return;
        average /= density;
        const float occupancy = min(
            1.0f, density / max(constants.rest_number_density, 1.0e-6f));
        const float response =
            1.0f - exp(-constants.wind_drag * occupancy * constants.timestep);
        float3 change =
            (average - pm_load(target_velocities[target])) * response;
        change = pm_limit(
            change,
            constants.maximum_wind_acceleration * constants.timestep);
        target_velocities[target] = pm_store(pm_limit(
            pm_load(target_velocities[target]) + change,
            constants.maximum_target_speed));
    }

    if (constants.command == 1u) return;
    const uint particle = thread_index;
    if (particle >= smoke_count || smoke_ages[particle] >= constants.lifetime)
        return;
        const float3 point = pm_load(smoke_positions[particle]);
        const float3 before = pm_load(smoke_previous[particle]);
        float nearest_squared =
            constants.contact_distance * constants.contact_distance;
        float earliest = 2.0f;
        uint best = constants.surface_index_count;
        float3 nearest = 0.0f;
        float3 face_normal = 0.0f;
        float3 weights = 0.0f;
        for (uint base = 0u; base + 2u < constants.surface_index_count;
             base += 3u) {
            const float3 a = pm_load(surface_positions[surface_indices[base]]);
            const float3 b =
                pm_load(surface_positions[surface_indices[base + 1u]]);
            const float3 c =
                pm_load(surface_positions[surface_indices[base + 2u]]);
            const float3 face = cross(b - a, c - a);
            const float face_squared = dot(face, face);
            if (face_squared <= 1.0e-14f) continue;
            const float3 normal = face * rsqrt(face_squared);
            const float side_before = dot(before - a, normal);
            const float side_after = dot(point - a, normal);
            if (side_before * side_after < 0.0f) {
                const float fraction = side_before /
                                       (side_before - side_after);
                if (fraction < earliest) {
                    const float3 hit = before + (point - before) * fraction;
                    const float3 candidate =
                        pm_closest_point_triangle(hit, a, b, c);
                    if (dot(hit - candidate, hit - candidate) < 1.0e-8f) {
                        earliest = fraction;
                        best = base;
                        nearest = candidate;
                        face_normal = normal;
                        weights = pm_triangle_weights(candidate, a, b, c);
                    }
                }
            }
            if (earliest <= 1.0f) continue;
            const float3 candidate = pm_closest_point_triangle(point, a, b, c);
            const float squared = dot(point - candidate, point - candidate);
            if (squared >= nearest_squared) continue;
            nearest_squared = squared;
            best = base;
            nearest = candidate;
            face_normal = normal;
            weights = pm_triangle_weights(candidate, a, b, c);
        }
        if (best == constants.surface_index_count) return;
        float side = dot(before - nearest, face_normal);
        if (abs(side) < 1.0e-5f)
            side = dot(point - nearest, face_normal);
        const float3 normal = side >= 0.0f ? face_normal : -face_normal;
        smoke_positions[particle] =
            pm_store(nearest + normal * constants.contact_distance);
        const float corner_weights[3] = {weights.x, weights.y, weights.z};
        float3 surface_velocity = 0.0f;
        for (uint corner = 0u; corner < 3u; ++corner) {
            const uint surface = surface_indices[best + corner];
            float3 corner_velocity = 0.0f;
            if (constants.mode == 0u) {
                corner_velocity =
                    pm_load(target_velocities[surface_sources[surface]]);
            } else {
                const PMMetalSurfaceBinding binding =
                    surface_bindings[surface];
                for (uint slot = 0u; slot < 4u; ++slot)
                    corner_velocity +=
                        pm_load(target_velocities[binding.nodes[slot]]) *
                        binding.weights[slot];
            }
            surface_velocity += corner_velocity * corner_weights[corner];
        }
        float3 relative =
            pm_load(smoke_velocities[particle]) - surface_velocity;
        relative -= normal * min(dot(relative, normal), 0.0f);
        smoke_velocities[particle] = pm_store(pm_limit(
            surface_velocity + relative, constants.maximum_smoke_speed));
}

kernel void pm_smoke_rope_wind(
    device PMPackedVec3 *smoke_positions [[buffer(0)]],
    device const PMPackedVec3 *smoke_previous [[buffer(1)]],
    device PMPackedVec3 *smoke_velocities [[buffer(2)]],
    device const float *smoke_ages [[buffer(3)]],
    device const PMPackedVec3 *rope_positions [[buffer(4)]],
    device PMPackedVec3 *rope_velocities [[buffer(5)]],
    device const float *rope_inverse_masses [[buffer(6)]],
    constant PMSmokeRopeConstants &constants [[buffer(7)]],
    device const PMSmokeMetadata &smoke_metadata [[buffer(8)]],
    device const float *grid_density [[buffer(9)]],
    device const float *grid_face_velocity [[buffer(10)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= constants.rope_count || constants.enabled == 0u) return;
    (void)smoke_previous;
    const uint smoke_count = min(smoke_metadata.count,
                                 constants.smoke_capacity);
    const float wind_radius = max(constants.wind_radius, 1.0e-5f);
    const float wind_radius_squared = wind_radius * wind_radius;
    const float3 grid_minimum = pm_load(constants.grid_minimum);
    const float3 grid_maximum = grid_minimum + constants.grid_spacing *
        float3(float(constants.grid_resolution),
               float(constants.grid_vertical_resolution),
               float(constants.grid_resolution));
    if (rope_inverse_masses[node] <= 0.0f ||
        (node == 0u && constants.skip_first != 0u) ||
        (node + 1u == constants.rope_count && constants.skip_last != 0u))
        return;
        const float3 rope_point = pm_load(rope_positions[node]);
        float3 flow = 0.0f;
        float density = 0.0f;
        const bool inside_grid = constants.grid_resolution != 0u &&
            all(rope_point >= grid_minimum) &&
            all(rope_point < grid_maximum);
        if (inside_grid) {
            flow = pm_smoke_sample_face_velocity(
                grid_face_velocity, rope_point, grid_minimum,
                constants.grid_spacing, constants.grid_resolution,
                constants.grid_vertical_resolution);
            density = pm_smoke_sample_cell_scalar(
                grid_density, rope_point, grid_minimum,
                constants.grid_spacing, constants.grid_resolution,
                constants.grid_vertical_resolution) *
                constants.rest_number_density;
        } else {
            for (uint particle = 0u; particle < smoke_count;
                 ++particle) {
                if (smoke_ages[particle] >= constants.lifetime) continue;
                const float3 delta =
                    pm_load(smoke_positions[particle]) - rope_point;
                const float squared = dot(delta, delta);
                if (squared >= wind_radius_squared) continue;
                const float q =
                    1.0f - sqrt(max(squared, 0.0f)) / wind_radius;
                const float weight = q * q * q;
                flow += pm_load(smoke_velocities[particle]) * weight;
                density += weight;
            }
            if (density > 1.0e-6f) flow /= density;
        }
        if (density <= 1.0e-6f) return;
        const float occupancy = min(
            1.0f, inside_grid
                      ? density
                      : density /
                            max(constants.rest_number_density, 1.0e-6f));
        const float response =
            1.0f - exp(-constants.wind_drag * occupancy *
                       constants.timestep);
        const float3 change = pm_limit(
            (flow - pm_load(rope_velocities[node])) * response,
            constants.maximum_wind_acceleration * constants.timestep);
        rope_velocities[node] = pm_store(pm_limit(
            pm_load(rope_velocities[node]) + change,
            constants.maximum_rope_speed));
}

kernel void pm_smoke_rope(
    device PMPackedVec3 *smoke_positions [[buffer(0)]],
    device const PMPackedVec3 *smoke_previous [[buffer(1)]],
    device PMPackedVec3 *smoke_velocities [[buffer(2)]],
    device const float *smoke_ages [[buffer(3)]],
    device const PMPackedVec3 *rope_positions [[buffer(4)]],
    device PMPackedVec3 *rope_velocities [[buffer(5)]],
    device const float *rope_inverse_masses [[buffer(6)]],
    constant PMSmokeRopeConstants &constants [[buffer(7)]],
    device const PMSmokeMetadata &smoke_metadata [[buffer(8)]],
    uint particle [[thread_position_in_grid]]) {
    if (constants.enabled == 0u) return;
    (void)rope_inverse_masses;
    const uint smoke_count = min(smoke_metadata.count,
                                 constants.smoke_capacity);
    if (particle >= smoke_count || smoke_ages[particle] >= constants.lifetime)
        return;

    // Sweep every live tracer path against every current rope capsule.  The
    // earliest hit wins, with squared distance as the stable tie-breaker.
    const float clearance = max(constants.contact_distance, 1.0e-5f);
    const float clearance_squared = clearance * clearance;
        const float3 point = pm_load(smoke_positions[particle]);
        const float3 before = pm_load(smoke_previous[particle]);
        float earliest = 2.0f;
        float nearest_squared = clearance_squared;
        uint best_segment = constants.rope_count;
        float3 smoke_hit = 0.0f;
        float3 rope_hit = 0.0f;
        for (uint segment = 0u; segment + 1u < constants.rope_count;
             ++segment) {
            const float3 first = pm_load(rope_positions[segment]);
            const float3 second = pm_load(rope_positions[segment + 1u]);
            if (max(point.x, before.x) < min(first.x, second.x) - clearance ||
                min(point.x, before.x) > max(first.x, second.x) + clearance ||
                max(point.y, before.y) < min(first.y, second.y) - clearance ||
                min(point.y, before.y) > max(first.y, second.y) + clearance ||
                max(point.z, before.z) < min(first.z, second.z) - clearance ||
                min(point.z, before.z) > max(first.z, second.z) + clearance)
                continue;
            float path_fraction = 0.0f;
            float rope_fraction = 0.0f;
            float3 on_path = 0.0f;
            float3 on_rope = 0.0f;
            pm_closest_segments(before, point, first, second, path_fraction,
                                rope_fraction, on_path, on_rope);
            const float3 separation = on_path - on_rope;
            const float squared = dot(separation, separation);
            if (squared >= clearance_squared ||
                path_fraction > earliest + 1.0e-6f ||
                (abs(path_fraction - earliest) <= 1.0e-6f &&
                 squared >= nearest_squared))
                continue;
            earliest = path_fraction;
            nearest_squared = squared;
            best_segment = segment;
            smoke_hit = on_path;
            rope_hit = on_rope;
        }
        if (best_segment == constants.rope_count) return;
        const float3 first = pm_load(rope_positions[best_segment]);
        const float3 second = pm_load(rope_positions[best_segment + 1u]);
        const float3 axis = second - first;
        const float axis_squared = max(dot(axis, axis), 1.0e-12f);
        const float rope_fraction =
            clamp(dot(rope_hit - first, axis) / axis_squared, 0.0f, 1.0f);
        const float3 rope_velocity =
            pm_load(rope_velocities[best_segment]) * (1.0f - rope_fraction) +
            pm_load(rope_velocities[best_segment + 1u]) * rope_fraction;
        float3 normal = smoke_hit - rope_hit;
        if (dot(normal, normal) < 1.0e-10f) normal = before - rope_hit;
        if (dot(normal, normal) < 1.0e-10f) {
            const float3 relative =
                pm_load(smoke_velocities[particle]) - rope_velocity;
            normal = -relative + axis * (dot(relative, axis) / axis_squared);
        }
        const float normal_squared = dot(normal, normal);
        normal = normal_squared > 1.0e-12f
                     ? normal * rsqrt(normal_squared)
                     : float3(0.0f, 1.0f, 0.0f);
        smoke_positions[particle] = pm_store(rope_hit + normal * clearance);
        float3 relative =
            pm_load(smoke_velocities[particle]) - rope_velocity;
        relative -= normal * min(0.0f, dot(relative, normal));
        smoke_velocities[particle] = pm_store(rope_velocity + relative);
}

kernel void pm_rope_soft_anchor_weights(
    device const PMPackedVec3 *soft_positions [[buffer(5)]],
    constant PMRopeSoftConstants &constants [[buffer(14)]],
    device const PMRopeAnchorState *anchor_states [[buffer(17)]],
    device uint *packed [[buffer(18)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u || packed[pm_rope_soft_valid] == 0u ||
        constants.enabled == 0u)
        return;
    const uint contact_count = packed[pm_rope_soft_prior_contact_count] +
                               packed[pm_rope_soft_contact_count];
    const float broad = constants.anchor_support_radius_scale *
                        constants.node_radius;
    const float local = constants.anchor_contact_support_radius_scale *
                        constants.node_radius;
    const float support = local + (broad - local) /
        (1.0f + float(contact_count) * 0.5f);
    const float support_squared = support * support;
    for (uint end = 0u; end < 2u; ++end) {
        const bool attached = end == 0u
                                  ? constants.attach_first != 0u
                                  : constants.attach_last != 0u;
        const uint triangle = end == 0u ? constants.first_triangle
                                         : constants.last_triangle;
        float sums[256];
        for (uint lane = 0u; lane < 256u; ++lane) {
            float local_sum = 0.0f;
            if (attached && triangle != 0xffffffffu &&
                triangle + 2u < constants.surface_index_count &&
                support > 1.0e-6f) {
                const float3 anchor = pm_load(anchor_states[end].position);
                for (uint node = lane; node < constants.soft_count;
                     node += 256u) {
                    const float3 delta =
                        pm_load(soft_positions[node]) - anchor;
                    const float squared = dot(delta, delta);
                    if (squared >= support_squared) continue;
                    const float value =
                        1.0f - squared / support_squared;
                    local_sum += value * value;
                }
            }
            sums[lane] = local_sum;
        }
        for (uint stride = 128u; stride != 0u; stride /= 2u)
            for (uint lane = 0u; lane < stride; ++lane)
                sums[lane] += sums[lane + stride];
        packed[end == 0u ? pm_rope_soft_first_anchor_weight
                         : pm_rope_soft_last_anchor_weight] =
            as_type<uint>(sums[0]);
    }
}

kernel void pm_rope_soft_apply(
    device PMPackedVec3 *rope_positions [[buffer(0)]],
    device const PMPackedVec3 *rope_previous [[buffer(1)]],
    device PMPackedVec3 *rope_velocities [[buffer(2)]],
    device const float *rope_inverse_masses [[buffer(3)]],
    device PMPackedVec3 *rope_forces [[buffer(4)]],
    device PMPackedVec3 *soft_positions [[buffer(5)]],
    device PMPackedVec3 *soft_velocities [[buffer(6)]],
    device const float *soft_inverse_masses [[buffer(7)]],
    device PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(10)]],
    device PMPackedVec3 *soft_forces [[buffer(11)]],
    device const PMPackedVec3 *soft_rest_positions [[buffer(12)]],
    device const PMPackedVec3 *surface_rest_positions [[buffer(13)]],
    constant PMRopeSoftConstants &constants [[buffer(14)]],
    device uint *diagnostic_contact_count [[buffer(15)]],
    device float *diagnostic_maximum_penetration [[buffer(16)]],
    device const PMRopeAnchorState *anchor_states [[buffer(17)]],
    device const uint *packed [[buffer(18)]],
    device PMPackedVec3 *previous_surface [[buffer(19)]],
    uint node [[thread_position_in_grid]]) {
    (void)rope_positions;
    (void)rope_previous;
    (void)rope_velocities;
    (void)rope_inverse_masses;
    (void)rope_forces;
    (void)surface_positions;
    (void)soft_rest_positions;
    (void)surface_rest_positions;
    (void)previous_surface;
    if (packed[pm_rope_soft_valid] == 0u || constants.enabled == 0u ||
        node >= constants.soft_count)
        return;
    const uint impulse_offset = packed[pm_rope_soft_impulse_offset];
    const uint contact_count = packed[pm_rope_soft_contact_count];
    const uint frame_contact_count =
        packed[pm_rope_soft_prior_contact_count] + contact_count;
    if (node == 0u) {
        diagnostic_contact_count[0] += contact_count;
        diagnostic_maximum_penetration[0] = max(
            diagnostic_maximum_penetration[0],
            pm_rope_soft_load_float(
                packed, pm_rope_soft_maximum_penetration));
    }
    float3 impulse = pm_rope_soft_load_vec3(
        packed, impulse_offset + 3u * node);
    for (uint end = 0u; end < 2u; ++end) {
        const bool attached = end == 0u
                                  ? constants.attach_first != 0u
                                  : constants.attach_last != 0u;
        if (!attached) continue;
        const uint triangle = end == 0u ? constants.first_triangle
                                         : constants.last_triangle;
        if (triangle == 0xffffffffu ||
            triangle + 2u >= constants.surface_index_count)
            continue;
        const float3 anchor = pm_load(anchor_states[end].position);
        const float3 reaction = pm_load(anchor_states[end].impulse);
        const float broad = constants.anchor_support_radius_scale *
                            constants.node_radius;
        const float local =
            constants.anchor_contact_support_radius_scale *
            constants.node_radius;
        const float support = local + (broad - local) /
            (1.0f + float(frame_contact_count) * 0.5f);
        const float support_squared = support * support;
        const float weight_sum = pm_rope_soft_load_float(
            packed, end == 0u ? pm_rope_soft_first_anchor_weight
                              : pm_rope_soft_last_anchor_weight);
        if (weight_sum > 1.0e-8f) {
            const float3 delta = pm_load(soft_positions[node]) - anchor;
            const float squared = dot(delta, delta);
            if (squared < support_squared) {
                const float value = 1.0f - squared / support_squared;
                impulse += reaction * (value * value / weight_sum);
            }
            continue;
        }
        const float3 anchor_weights = end == 0u
            ? pm_load(constants.first_weights)
            : pm_load(constants.last_weights);
        const float corner_weights[3] = {
            anchor_weights.x, anchor_weights.y, anchor_weights.z};
        float fallback_weight = 0.0f;
        for (uint corner = 0u; corner < 3u; ++corner) {
            const PMMetalSurfaceBinding binding =
                surface_bindings[surface_indices[triangle + corner]];
            for (uint slot = 0u; slot < 4u; ++slot)
                if (binding.nodes[slot] == node)
                    fallback_weight +=
                        corner_weights[corner] * binding.weights[slot];
        }
        impulse += reaction * fallback_weight;
    }

    const float inverse = soft_inverse_masses[node];
    if (inverse <= 0.0f) return;
    float3 change = pm_limit(
        impulse * inverse,
        constants.maximum_soft_acceleration * constants.timestep);
    soft_velocities[node] = pm_store(pm_limit(
        pm_load(soft_velocities[node]) + change,
        constants.maximum_soft_speed));
    soft_positions[node] = pm_store(
        pm_load(soft_positions[node]) + change * constants.timestep);
    soft_forces[node] = pm_store(
        pm_load(soft_forces[node]) +
        change * (constants.frame_inverse_timestep / inverse));
}

kernel void pm_rope_soft_surface_update(
    device const PMPackedVec3 *soft_positions [[buffer(5)]],
    device PMPackedVec3 *surface_positions [[buffer(8)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(10)]],
    device const PMPackedVec3 *soft_rest_positions [[buffer(12)]],
    device const PMPackedVec3 *surface_rest_positions [[buffer(13)]],
    constant PMRopeSoftConstants &constants [[buffer(14)]],
    device const uint *packed [[buffer(18)]],
    device PMPackedVec3 *previous_surface [[buffer(19)]],
    uint surface [[thread_position_in_grid]]) {
    if (packed[pm_rope_soft_valid] == 0u || constants.enabled == 0u ||
        surface >= constants.surface_count)
        return;
    const PMMetalSurfaceBinding binding = surface_bindings[surface];
    float3 point = pm_load(surface_rest_positions[surface]);
    for (uint slot = 0u; slot < 4u; ++slot) {
        const uint node = binding.nodes[slot];
        point += (pm_load(soft_positions[node]) -
                  pm_load(soft_rest_positions[node])) *
                 binding.weights[slot];
    }
    surface_positions[surface] = pm_store(point);
    previous_surface[surface] = pm_store(point);
}

kernel void pm_rope_soft_update_endpoint(
    device PMPackedVec3 *rope_positions [[buffer(0)]],
    device PMPackedVec3 *rope_velocities [[buffer(2)]],
    device const PMPackedVec3 *soft_velocities [[buffer(6)]],
    device const PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(10)]],
    constant PMRopeSoftConstants &constants [[buffer(14)]],
    device PMRopeAnchorState *anchor_states [[buffer(17)]],
    device const uint *packed [[buffer(18)]],
    uint end [[thread_position_in_grid]]) {
    (void)rope_positions;
    (void)rope_velocities;
    if (end >= 2u || packed[pm_rope_soft_valid] == 0u ||
        constants.enabled == 0u)
        return;
    const bool attached = end == 0u
                              ? constants.attach_first != 0u
                              : constants.attach_last != 0u;
    if (!attached) return;
    const uint triangle = end == 0u ? constants.first_triangle
                                     : constants.last_triangle;
    if (triangle == 0xffffffffu ||
        triangle + 2u >= constants.surface_index_count)
        return;
    const float3 weights = end == 0u
                               ? pm_load(constants.first_weights)
                               : pm_load(constants.last_weights);
    const float3 offset = end == 0u
                              ? pm_load(constants.first_offset)
                              : pm_load(constants.last_offset);
    const float corner_weights[3] = {weights.x, weights.y, weights.z};
    float3 position = offset;
    float3 velocity = 0.0f;
    for (uint corner = 0u; corner < 3u; ++corner) {
        const uint surface = surface_indices[triangle + corner];
        position += pm_load(surface_positions[surface]) *
                    corner_weights[corner];
        const PMMetalSurfaceBinding binding = surface_bindings[surface];
        for (uint slot = 0u; slot < 4u; ++slot)
            velocity += pm_load(soft_velocities[binding.nodes[slot]]) *
                        (corner_weights[corner] * binding.weights[slot]);
    }
    anchor_states[end].position = pm_store(position);
    anchor_states[end].velocity = pm_store(velocity);
    anchor_states[end].impulse = {0.0f, 0.0f, 0.0f};
    anchor_states[end].inverse_mass = 0.0f;
}

kernel void pm_rope_soft_finalize(
    device uint *packed [[buffer(18)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index == 0u) packed[pm_rope_soft_valid] = 0u;
}

kernel void pm_rope_soft_serial(
    device PMPackedVec3 *rope_positions [[buffer(0)]],
    device const PMPackedVec3 *rope_previous [[buffer(1)]],
    device PMPackedVec3 *rope_velocities [[buffer(2)]],
    device const float *rope_inverse_masses [[buffer(3)]],
    device PMPackedVec3 *rope_forces [[buffer(4)]],
    device PMPackedVec3 *soft_positions [[buffer(5)]],
    device PMPackedVec3 *soft_velocities [[buffer(6)]],
    device const float *soft_inverse_masses [[buffer(7)]],
    device PMPackedVec3 *surface_positions [[buffer(8)]],
    device const uint *surface_indices [[buffer(9)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(10)]],
    device PMPackedVec3 *soft_forces [[buffer(11)]],
    device const PMPackedVec3 *soft_rest_positions [[buffer(12)]],
    device const PMPackedVec3 *surface_rest_positions [[buffer(13)]],
    constant PMRopeSoftConstants &constants [[buffer(14)]],
    device uint *diagnostic_contact_count [[buffer(15)]],
    device float *diagnostic_maximum_penetration [[buffer(16)]],
    device PMRopeAnchorState *anchor_states [[buffer(17)]],
    device uint *packed [[buffer(18)]],
    device PMPackedVec3 *previous_surface [[buffer(19)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u || constants.enabled == 0u) return;
    if (packed[pm_rope_soft_valid] != 0u) {
        const uint soft_count = packed[pm_rope_soft_node_count];
        const uint surface_count = packed[pm_rope_soft_surface_count];
        const uint impulse_offset = packed[pm_rope_soft_impulse_offset];
        const uint contact_count = packed[pm_rope_soft_contact_count];
        const uint frame_contact_count =
            packed[pm_rope_soft_prior_contact_count] + contact_count;
        diagnostic_contact_count[0] += contact_count;
        diagnostic_maximum_penetration[0] = max(
            diagnostic_maximum_penetration[0],
            pm_rope_soft_load_float(
                packed, pm_rope_soft_maximum_penetration));

        // Scatter endpoint reactions into the same deterministic node-force
        // buffer used by surface contacts before applying the capped load.
        for (uint end = 0u; end < 2u; ++end) {
            const bool attached = end == 0u
                                      ? constants.attach_first != 0u
                                      : constants.attach_last != 0u;
            if (!attached) continue;
            const uint triangle = end == 0u ? constants.first_triangle
                                             : constants.last_triangle;
            if (triangle == 0xffffffffu ||
                triangle + 2u >= constants.surface_index_count)
                continue;
            const float3 anchor = pm_load(anchor_states[end].position);
            const float3 reaction = pm_load(anchor_states[end].impulse);
            const float broad = constants.anchor_support_radius_scale *
                                constants.node_radius;
            const float local =
                constants.anchor_contact_support_radius_scale *
                constants.node_radius;
            const float support = local + (broad - local) /
                (1.0f + float(frame_contact_count) * 0.5f);
            const float support_squared = support * support;
            float weight_sum = 0.0f;
            if (support > 1.0e-6f) {
                for (uint node = 0u; node < soft_count; ++node) {
                    const float3 delta =
                        pm_load(soft_positions[node]) - anchor;
                    const float squared = dot(delta, delta);
                    if (squared >= support_squared) continue;
                    const float value =
                        1.0f - squared / support_squared;
                    weight_sum += value * value;
                }
            }
            if (weight_sum > 1.0e-8f) {
                for (uint node = 0u; node < soft_count; ++node) {
                    const float3 delta =
                        pm_load(soft_positions[node]) - anchor;
                    const float squared = dot(delta, delta);
                    if (squared >= support_squared) continue;
                    const float value =
                        1.0f - squared / support_squared;
                    pm_rope_soft_add_vec3(
                        packed, impulse_offset + 3u * node,
                        reaction * (value * value / weight_sum));
                }
            } else {
                const float3 anchor_weights = end == 0u
                    ? pm_load(constants.first_weights)
                    : pm_load(constants.last_weights);
                const float corner_weights[3] = {
                    anchor_weights.x, anchor_weights.y, anchor_weights.z};
                for (uint corner = 0u; corner < 3u; ++corner) {
                    const PMMetalSurfaceBinding binding = surface_bindings[
                        surface_indices[triangle + corner]];
                    for (uint slot = 0u; slot < 4u; ++slot) {
                        const float weight = corner_weights[corner] *
                                             binding.weights[slot];
                        if (weight <= 0.0f) continue;
                        pm_rope_soft_add_vec3(
                            packed,
                            impulse_offset + 3u * binding.nodes[slot],
                            reaction * weight);
                    }
                }
            }
        }

        for (uint node = 0u; node < soft_count; ++node) {
            const float inverse = soft_inverse_masses[node];
            if (inverse <= 0.0f) continue;
            float3 change = pm_rope_soft_load_vec3(
                                packed, impulse_offset + 3u * node) *
                            inverse;
            change = pm_limit(
                change,
                constants.maximum_soft_acceleration * constants.timestep);
            soft_velocities[node] = pm_store(pm_limit(
                pm_load(soft_velocities[node]) + change,
                constants.maximum_soft_speed));
            soft_positions[node] = pm_store(
                pm_load(soft_positions[node]) +
                change * constants.timestep);
            soft_forces[node] = pm_store(
                pm_load(soft_forces[node]) +
                change * (constants.frame_inverse_timestep / inverse));
        }
        for (uint surface = 0u; surface < surface_count; ++surface) {
            const PMMetalSurfaceBinding binding = surface_bindings[surface];
            float3 point = pm_load(surface_rest_positions[surface]);
            for (uint slot = 0u; slot < 4u; ++slot) {
                const uint node = binding.nodes[slot];
                point += (pm_load(soft_positions[node]) -
                          pm_load(soft_rest_positions[node])) *
                         binding.weights[slot];
            }
            surface_positions[surface] = pm_store(point);
            previous_surface[surface] = pm_store(point);
        }
        for (uint end = 0u; end < 2u; ++end) {
            const bool attached = end == 0u
                                      ? constants.attach_first != 0u
                                      : constants.attach_last != 0u;
            if (!attached) continue;
            const uint triangle = end == 0u ? constants.first_triangle
                                             : constants.last_triangle;
            if (triangle == 0xffffffffu ||
                triangle + 2u >= constants.surface_index_count)
                continue;
            const float3 weights = end == 0u
                                       ? pm_load(constants.first_weights)
                                       : pm_load(constants.last_weights);
            const float3 offset = end == 0u
                                      ? pm_load(constants.first_offset)
                                      : pm_load(constants.last_offset);
            const float corner_weights[3] = {
                weights.x, weights.y, weights.z};
            float3 position = offset;
            float3 velocity = 0.0f;
            for (uint corner = 0u; corner < 3u; ++corner) {
                const uint surface =
                    surface_indices[triangle + corner];
                position += pm_load(surface_positions[surface]) *
                            corner_weights[corner];
                const PMMetalSurfaceBinding binding =
                    surface_bindings[surface];
                for (uint slot = 0u; slot < 4u; ++slot)
                    velocity += pm_load(
                        soft_velocities[binding.nodes[slot]]) *
                        (corner_weights[corner] *
                         binding.weights[slot]);
            }
            const uint endpoint =
                end == 0u ? 0u : constants.rope_count - 1u;
            rope_positions[endpoint] = pm_store(position);
            rope_velocities[endpoint] = pm_store(velocity);
            anchor_states[end].position = pm_store(position);
            anchor_states[end].velocity = pm_store(velocity);
        }
        packed[pm_rope_soft_valid] = 0u;
        return;
    }
    return;
}

kernel void pm_fluid_cloth_forces(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device PMPackedVec3 *fluid_velocities [[buffer(1)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(2)]],
    device float *fluid_foam_sources [[buffer(3)]],
    device PMPackedVec3 *cloth_positions [[buffer(4)]],
    device PMPackedVec3 *cloth_velocities [[buffer(5)]],
    device const float *cloth_inverse_masses [[buffer(6)]],
    device PMPackedVec3 *surface_positions [[buffer(7)]],
    device const uint *surface_indices [[buffer(8)]],
    device const uint *surface_sources [[buffer(9)]],
    device PMPackedVec3 *cloth_forces [[buffer(10)]],
    constant PMFluidClothConstants &constants [[buffer(11)]],
    device PMPackedVec3 *fluid_accelerations [[buffer(12)]],
    device uint *contribution_nodes [[buffer(13)]],
    device PMPackedVec3 *contribution_forces [[buffer(14)]],
    uint particle [[thread_position_in_grid]]) {
    (void)cloth_positions;
    (void)cloth_inverse_masses;
    (void)cloth_forces;
    const uint contribution = particle * 3u;
    if (particle >= fluid_metadata.count || constants.enabled == 0u) return;
    for (uint corner = 0u; corner < 3u; ++corner) {
        contribution_nodes[contribution + corner] = 0xffffffffu;
        contribution_forces[contribution + corner] = pm_store(float3(0.0f));
    }
    const float3 point = pm_load(fluid_positions[particle]);
    float nearest_squared = INFINITY;
    float3 nearest = 0.0f;
    float3 face_normal = 0.0f;
    float3 weights = 0.0f;
    uint best = constants.surface_index_count;
    for (uint base = 0u; base + 2u < constants.surface_index_count;
         base += 3u) {
        const float3 a = pm_load(surface_positions[surface_indices[base]]);
        const float3 b =
            pm_load(surface_positions[surface_indices[base + 1u]]);
        const float3 c =
            pm_load(surface_positions[surface_indices[base + 2u]]);
        const float3 face = cross(b - a, c - a);
        const float face_squared = dot(face, face);
        if (face_squared <= 1.0e-14f) continue;
        const float3 candidate = pm_closest_point_triangle(point, a, b, c);
        const float squared = dot(point - candidate, point - candidate);
        if (squared >= nearest_squared) continue;
        nearest_squared = squared;
        nearest = candidate;
        face_normal = face * rsqrt(face_squared) * constants.orientation;
        weights = pm_triangle_weights(candidate, a, b, c);
        best = base;
    }
    if (best == constants.surface_index_count) return;
    const float signed_distance = dot(point - nearest, face_normal);
    const float violation = constants.contact_distance + signed_distance;
    const bool outside = signed_distance > 0.0f;
    if (violation <= 0.0f ||
        (!outside && nearest_squared >
                         constants.interaction_radius *
                             constants.interaction_radius))
        return;
    const uint surface_vertices[3] = {
        surface_indices[best], surface_indices[best + 1u],
        surface_indices[best + 2u]};
    const uint physical_vertices[3] = {
        surface_sources[surface_vertices[0]],
        surface_sources[surface_vertices[1]],
        surface_sources[surface_vertices[2]]};
    const float corner_weights[3] = {weights.x, weights.y, weights.z};
    float3 surface_velocity = 0.0f;
    for (uint corner = 0u; corner < 3u; ++corner)
        surface_velocity +=
            pm_load(cloth_velocities[physical_vertices[corner]]) *
            corner_weights[corner];
    const float3 velocity = pm_load(fluid_velocities[particle]);
    const float3 relative = velocity - surface_velocity;
    const float normal_speed = dot(relative, face_normal);
    const float magnitude = min(
        constants.maximum_force,
        max(0.0f, constants.stiffness * violation +
                      constants.damping * normal_speed));
    const float3 tangent = relative - face_normal * normal_speed;
    const float3 force = pm_limit(
        -face_normal * magnitude - tangent * constants.tangential_drag,
        constants.maximum_force);
    fluid_accelerations[particle] = pm_store(
        pm_load(fluid_accelerations[particle]) +
        force / max(constants.particle_mass, 1.0e-12f));
    fluid_foam_sources[particle] = max(
        fluid_foam_sources[particle],
        clamp(magnitude / max(constants.maximum_force, 1.0f),
              0.0f, 1.0f));
    for (uint corner = 0u; corner < 3u; ++corner) {
        contribution_nodes[contribution + corner] =
            physical_vertices[corner];
        contribution_forces[contribution + corner] =
            pm_store(-force * corner_weights[corner]);
    }
}

kernel void pm_fluid_cloth_apply(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device PMPackedVec3 *fluid_velocities [[buffer(1)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(2)]],
    device float *fluid_foam_sources [[buffer(3)]],
    device PMPackedVec3 *cloth_positions [[buffer(4)]],
    device PMPackedVec3 *cloth_velocities [[buffer(5)]],
    device const float *cloth_inverse_masses [[buffer(6)]],
    device PMPackedVec3 *surface_positions [[buffer(7)]],
    device const uint *surface_indices [[buffer(8)]],
    device const uint *surface_sources [[buffer(9)]],
    device PMPackedVec3 *cloth_forces [[buffer(10)]],
    constant PMFluidClothConstants &constants [[buffer(11)]],
    device PMPackedVec3 *fluid_accelerations [[buffer(12)]],
    device const uint *contribution_nodes [[buffer(13)]],
    device const PMPackedVec3 *contribution_forces [[buffer(14)]],
    uint node [[thread_position_in_grid]]) {
    (void)fluid_positions;
    (void)fluid_velocities;
    (void)fluid_foam_sources;
    (void)surface_positions;
    (void)surface_indices;
    (void)surface_sources;
    (void)fluid_accelerations;
    if (node >= constants.cloth_count || constants.enabled == 0u) return;
    float3 force = 0.0f;
    for (uint particle = 0u; particle < fluid_metadata.count; ++particle) {
        const uint contribution = particle * 3u;
        for (uint corner = 0u; corner < 3u; ++corner) {
            const uint item = contribution + corner;
            if (contribution_nodes[item] == node)
                force += pm_load(contribution_forces[item]);
        }
    }
    cloth_forces[node] = pm_store(pm_load(cloth_forces[node]) + force);
    const float inverse = cloth_inverse_masses[node];
    if (inverse <= 0.0f) return;
    const float3 change = pm_limit(
        force * (inverse * constants.timestep), 2.0f);
    cloth_velocities[node] = pm_store(pm_limit(
        pm_load(cloth_velocities[node]) + change, 20.0f));
    cloth_positions[node] = pm_store(
        pm_load(cloth_positions[node]) + change * constants.timestep);
}

kernel void pm_fluid_cloth_surface_update(
    device const PMPackedVec3 *cloth_positions [[buffer(4)]],
    device PMPackedVec3 *surface_positions [[buffer(7)]],
    device const uint *surface_sources [[buffer(9)]],
    constant PMFluidClothConstants &constants [[buffer(11)]],
    uint surface [[thread_position_in_grid]]) {
    if (surface >= constants.surface_count || constants.enabled == 0u) return;
    surface_positions[surface] = cloth_positions[surface_sources[surface]];
}

kernel void pm_fluid_cloth_project(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device PMPackedVec3 *fluid_velocities [[buffer(1)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(2)]],
    device float *fluid_foam_sources [[buffer(3)]],
    device PMPackedVec3 *cloth_positions [[buffer(4)]],
    device PMPackedVec3 *cloth_velocities [[buffer(5)]],
    device const float *cloth_inverse_masses [[buffer(6)]],
    device PMPackedVec3 *surface_positions [[buffer(7)]],
    device const uint *surface_indices [[buffer(8)]],
    device const uint *surface_sources [[buffer(9)]],
    device PMPackedVec3 *cloth_forces [[buffer(10)]],
    constant PMFluidClothConstants &constants [[buffer(11)]],
    device PMPackedVec3 *fluid_accelerations [[buffer(12)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index >= fluid_metadata.count || constants.enabled == 0u)
        return;
    (void)fluid_foam_sources;
    (void)cloth_positions;
    (void)cloth_inverse_masses;
    (void)cloth_forces;
    (void)fluid_accelerations;
    const uint particle = thread_index;
        const float3 point = pm_load(fluid_positions[particle]);
        float nearest_squared = INFINITY;
        float3 nearest = 0.0f;
        float3 normal = 0.0f;
        float3 weights = 0.0f;
        uint best = constants.surface_index_count;
        for (uint base = 0u; base + 2u < constants.surface_index_count;
             base += 3u) {
            const float3 a = pm_load(surface_positions[surface_indices[base]]);
            const float3 b =
                pm_load(surface_positions[surface_indices[base + 1u]]);
            const float3 c =
                pm_load(surface_positions[surface_indices[base + 2u]]);
            const float3 face = cross(b - a, c - a);
            const float face_squared = dot(face, face);
            if (face_squared <= 1.0e-14f) continue;
            const float3 candidate = pm_closest_point_triangle(point, a, b, c);
            const float squared = dot(point - candidate, point - candidate);
            if (squared >= nearest_squared) continue;
            nearest_squared = squared;
            nearest = candidate;
            normal = face * rsqrt(face_squared) * constants.orientation;
            weights = pm_triangle_weights(candidate, a, b, c);
            best = base;
        }
        if (best == constants.surface_index_count) return;
        const float violation =
            constants.contact_distance + dot(point - nearest, normal);
        if (violation <= 0.0f) return;
        fluid_positions[particle] = pm_store(point - normal * violation);
        const uint physical_a =
            surface_sources[surface_indices[best]];
        const uint physical_b =
            surface_sources[surface_indices[best + 1u]];
        const uint physical_c =
            surface_sources[surface_indices[best + 2u]];
        const float3 surface_velocity =
            pm_load(cloth_velocities[physical_a]) * weights.x +
            pm_load(cloth_velocities[physical_b]) * weights.y +
            pm_load(cloth_velocities[physical_c]) * weights.z;
        float3 relative =
            pm_load(fluid_velocities[particle]) - surface_velocity;
        const float outward_speed = dot(relative, normal);
        if (outward_speed > 0.0f)
            relative -= normal * outward_speed;
        fluid_velocities[particle] = pm_store(surface_velocity + relative);
}

struct PMFluidSoftNearest {
    float3 point;
    float3 normal;
    float3 weights;
    float penetration;
    uint triangle;
};

static bool pm_fluid_soft_inside(
    float3 point, device const PMPackedVec3 *surface,
    device const uint *indices, uint index_count) {
    const float3 direction = float3(1.0f, 0.371f, 0.173f);
    int winding = 0;
    bool ambiguous = false;
    for (uint base = 0u; base + 2u < index_count; base += 3u) {
        const float3 a = pm_load(surface[indices[base]]);
        const float3 edge1 =
            pm_load(surface[indices[base + 1u]]) - a;
        const float3 edge2 =
            pm_load(surface[indices[base + 2u]]) - a;
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
        const float3 a = pm_load(surface[indices[base]]) - point;
        const float3 b = pm_load(surface[indices[base + 1u]]) - point;
        const float3 c = pm_load(surface[indices[base + 2u]]) - point;
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

static PMFluidSoftNearest pm_fluid_soft_nearest(
    float3 point, float3 start,
    device const PMPackedVec3 *surface,
    device const PMPackedVec3 *previous_surface,
    device const uint *indices, uint index_count, float orientation,
    float distance, bool sweep) {
    PMFluidSoftNearest result = {
        float3(0.0f), float3(0.0f), float3(0.0f), 0.0f,
        0xffffffffu};
    float best = INFINITY;
    float earliest = 2.0f;
    for (uint base = 0u; base + 2u < index_count; base += 3u) {
        const uint ia = indices[base];
        const uint ib = indices[base + 1u];
        const uint ic = indices[base + 2u];
        const float3 a = pm_load(surface[ia]);
        const float3 b = pm_load(surface[ib]);
        const float3 c = pm_load(surface[ic]);
        const float3 face = cross(b - a, c - a);
        const float face_squared = dot(face, face);
        if (face_squared < 1.0e-16f) continue;
        const float3 normal =
            face * rsqrt(face_squared) * orientation;
        const float3 near = pm_closest_point_triangle(point, a, b, c);
        const float3 weights = pm_triangle_weights(near, a, b, c);
        const float squared = dot(point - near, point - near);
        if (earliest > 1.0f && squared < best) {
            best = squared;
            result = {near, normal, weights,
                      distance - dot(point - near, normal), base};
        }
        if (!sweep) continue;
        const float3 old_near =
            pm_load(previous_surface[ia]) * weights.x +
            pm_load(previous_surface[ib]) * weights.y +
            pm_load(previous_surface[ic]) * weights.z;
        const float3 transported_start = start + (near - old_near);
        const float first = dot(transported_start - a, normal);
        const float last = dot(point - a, normal);
        if (first < distance || last >= distance ||
            first - last < 1.0e-8f)
            continue;
        const float fraction = (first - distance) / (first - last);
        if (fraction >= earliest) continue;
        const float3 crossing =
            transported_start + (point - transported_start) * fraction -
            normal * distance;
        const float3 hit = pm_closest_point_triangle(crossing, a, b, c);
        if (dot(crossing - hit, crossing - hit) > 1.0e-8f) continue;
        earliest = fraction;
        result = {hit, normal, pm_triangle_weights(hit, a, b, c),
                  distance - last, base};
    }
    if (result.triangle == 0xffffffffu) return result;
    if (earliest > 1.0f) {
        const float radius = sqrt(best);
        const bool face_interior = result.weights.x > 1.0e-4f &&
            result.weights.y > 1.0e-4f &&
            result.weights.z > 1.0e-4f;
        const bool inside = face_interior
            ? dot(point - result.point, result.normal) < 0.0f
            : pm_fluid_soft_inside(point, surface, indices, index_count);
        if (!inside) {
            result.penetration = distance - radius;
            if (radius > 1.0e-7f)
                result.normal = (point - result.point) / radius;
        } else {
            result.penetration = distance + radius;
            if (radius > 1.0e-7f)
                result.normal = (result.point - point) / radius;
        }
    }
    if (result.penetration <= 0.0f) result.triangle = 0xffffffffu;
    return result;
}

kernel void pm_fluid_soft_contact(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device const PMPackedVec3 *fluid_previous [[buffer(1)]],
    device PMPackedVec3 *fluid_velocities [[buffer(2)]],
    device PMPackedVec3 *fluid_accelerations [[buffer(3)]],
    device float *fluid_foam [[buffer(4)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(5)]],
    device const PMPackedVec3 *node_positions [[buffer(6)]],
    device const PMPackedVec3 *node_velocities [[buffer(7)]],
    device const float *node_inverse_masses [[buffer(8)]],
    device const PMPackedVec3 *surface_positions [[buffer(9)]],
    device const uint *surface_indices [[buffer(10)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(11)]],
    device PMPackedVec3 *node_forces [[buffer(12)]],
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    device const PMPackedVec3 *node_rest_positions [[buffer(14)]],
    device const PMPackedVec3 *surface_rest_positions [[buffer(15)]],
    device atomic_uint *diagnostic_contact_count [[buffer(16)]],
    device atomic_uint *diagnostic_maximum_penetration [[buffer(17)]],
    device uint *contribution_nodes [[buffer(18)]],
    device PMPackedVec3 *contribution_positions [[buffer(19)]],
    device PMPackedVec3 *contribution_changes [[buffer(20)]],
    device PMPackedVec3 *contribution_forces [[buffer(21)]],
    device atomic_uint *node_contact_counts [[buffer(22)]],
    device const PMPackedVec3 *previous_surface [[buffer(23)]],
    uint particle [[thread_position_in_grid]]) {
    (void)node_forces;
    (void)node_rest_positions;
    (void)surface_rest_positions;
    if (particle >= fluid_metadata.count || constants.enabled == 0u) return;
    const uint contribution = particle * 12u;
    for (uint item = 0u; item < 12u; ++item) {
        contribution_nodes[contribution + item] = 0xffffffffu;
        contribution_positions[contribution + item] = pm_store(float3(0.0f));
        contribution_changes[contribution + item] = pm_store(float3(0.0f));
        contribution_forces[contribution + item] = pm_store(float3(0.0f));
    }

    const bool sweep = atomic_load_explicit(
        node_contact_counts + constants.node_count,
        memory_order_relaxed) != 0u;
    const PMFluidSoftNearest hit = pm_fluid_soft_nearest(
        pm_load(fluid_positions[particle]),
        pm_load(fluid_previous[particle]), surface_positions,
        previous_surface, surface_indices, constants.surface_index_count,
        constants.orientation, constants.contact_distance, sweep);
    if (hit.triangle == 0xffffffffu) return;
    const uint best = hit.triangle;
    const float3 normal = hit.normal;
    const float3 weights = hit.weights;
    const float penetration = hit.penetration;

    uint contact_nodes[12];
    float contact_weights[12];
    uint node_count = 0u;
    const float corner_weights[3] = {weights.x, weights.y, weights.z};
    for (uint corner = 0u; corner < 3u; ++corner) {
        const PMMetalSurfaceBinding binding =
            surface_bindings[surface_indices[best + corner]];
        for (uint slot = 0u; slot < 4u; ++slot) {
            const float weight =
                corner_weights[corner] * binding.weights[slot];
            if (weight <= 0.0f) continue;
            const uint node = binding.nodes[slot];
            uint item = 0u;
            while (item < node_count && contact_nodes[item] != node) ++item;
            if (item == node_count && node_count < 12u) {
                contact_nodes[node_count] = node;
                contact_weights[node_count] = 0.0f;
                ++node_count;
            }
            if (item < node_count) contact_weights[item] += weight;
        }
    }
    if (node_count == 0u) return;
    float3 surface_velocity = 0.0f;
    for (uint item = 0u; item < node_count; ++item) {
        const uint node = contact_nodes[item];
        const float weight = contact_weights[item];
        surface_velocity += pm_load(node_velocities[node]) * weight;
        atomic_fetch_add_explicit(
            node_contact_counts + node, 1u, memory_order_relaxed);
        contribution_nodes[contribution + item] = node;
        contribution_positions[contribution + item] = pm_store(float3(
            weight, item == 0u ? penetration : 0.0f, 0.0f));
    }
    contribution_changes[contribution] = pm_store(
        pm_load(fluid_velocities[particle]) - surface_velocity);
    contribution_forces[contribution] = pm_store(normal);
    atomic_fetch_add_explicit(
        diagnostic_contact_count, 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(
        diagnostic_maximum_penetration, as_type<uint>(penetration),
        memory_order_relaxed);
}

kernel void pm_fluid_soft_solve(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device PMPackedVec3 *fluid_velocities [[buffer(2)]],
    device PMPackedVec3 *fluid_accelerations [[buffer(3)]],
    device float *fluid_foam [[buffer(4)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(5)]],
    device const PMPackedVec3 *node_velocities [[buffer(7)]],
    device const float *node_inverse_masses [[buffer(8)]],
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    device const uint *contribution_nodes [[buffer(18)]],
    device PMPackedVec3 *contribution_positions [[buffer(19)]],
    device PMPackedVec3 *contribution_changes [[buffer(20)]],
    device PMPackedVec3 *contribution_forces [[buffer(21)]],
    device atomic_uint *node_contact_counts [[buffer(22)]],
    uint particle [[thread_position_in_grid]]) {
    if (particle >= fluid_metadata.count || constants.enabled == 0u) return;
    const uint contribution = particle * 12u;
    if (contribution_nodes[contribution] == 0xffffffffu) return;
    const float particle_inverse =
        1.0f / max(constants.particle_mass, 1.0e-12f);
    const float3 normal = pm_load(contribution_forces[contribution]);
    const float3 relative = pm_load(contribution_changes[contribution]);
    const float penetration =
        pm_load(contribution_positions[contribution]).y;
    float denominator = particle_inverse;
    for (uint item = 0u; item < 12u; ++item) {
        const uint index = contribution + item;
        const uint node = contribution_nodes[index];
        if (node == 0xffffffffu) break;
        const float weight = pm_load(contribution_positions[index]).x;
        const uint contact_count = max(
            atomic_load_explicit(
                node_contact_counts + node, memory_order_relaxed),
            1u);
        denominator += weight * node_inverse_masses[node] *
                       float(contact_count);
    }
    if (denominator <= 1.0e-12f) return;
    const float normal_speed = dot(relative, normal);
    const float normal_impulse = max(0.0f, -normal_speed) / denominator;
    const float3 tangent = relative - normal * normal_speed;
    float3 impulse =
        normal * normal_impulse -
        pm_limit(tangent / denominator,
                 constants.friction * normal_impulse);
    float scale = 1.0f;
    for (uint item = 0u; item < 12u; ++item) {
        const uint index = contribution + item;
        const uint node = contribution_nodes[index];
        if (node == 0xffffffffu) break;
        const float weight = pm_load(contribution_positions[index]).x;
        const float inverse = node_inverse_masses[node];
        const float count = float(max(
            atomic_load_explicit(
                node_contact_counts + node, memory_order_relaxed),
            1u));
        const float3 delta = -impulse * (weight * inverse * count);
        const float squared = dot(delta, delta);
        if (squared <= 1.0e-20f) continue;
        const float3 velocity = pm_load(node_velocities[node]);
        const float projection = dot(velocity, delta);
        const float excess = min(
            0.0f,
            dot(velocity, velocity) -
                constants.maximum_soft_speed *
                    constants.maximum_soft_speed);
        const float limit =
            (-projection + sqrt(max(
                 0.0f, projection * projection - squared * excess))) /
            squared;
        scale = min(scale, max(0.0f, limit));
    }
    impulse *= scale;
    const float lambda =
        min(penetration, constants.maximum_projection) / denominator;
    fluid_positions[particle] = pm_store(
        pm_load(fluid_positions[particle]) +
        normal * (particle_inverse * lambda));
    const float3 velocity_change = impulse * particle_inverse;
    fluid_velocities[particle] = pm_store(
        pm_load(fluid_velocities[particle]) + velocity_change);
    fluid_accelerations[particle] = pm_store(
        pm_load(fluid_accelerations[particle]) +
        velocity_change / max(constants.timestep, 1.0e-12f));
    fluid_foam[particle] = max(
        fluid_foam[particle],
        min(1.0f, normal_impulse * particle_inverse * 0.1f));
    for (uint item = 0u; item < 12u; ++item) {
        const uint index = contribution + item;
        const uint node = contribution_nodes[index];
        if (node == 0xffffffffu) break;
        const float weight = pm_load(contribution_positions[index]).x;
        const float inverse = node_inverse_masses[node];
        contribution_positions[index] = pm_store(
            -normal * (weight * inverse * lambda));
        contribution_changes[index] = pm_store(
            -impulse * (weight * inverse));
        contribution_forces[index] = pm_store(
            -impulse *
            (weight * constants.frame_inverse_timestep));
    }
}

kernel void pm_fluid_soft_apply(
    device PMPackedVec3 *node_positions [[buffer(6)]],
    device PMPackedVec3 *node_velocities [[buffer(7)]],
    device PMPackedVec3 *node_forces [[buffer(12)]],
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(5)]],
    device const uint *contribution_nodes [[buffer(18)]],
    device const PMPackedVec3 *contribution_positions [[buffer(19)]],
    device const PMPackedVec3 *contribution_changes [[buffer(20)]],
    device const PMPackedVec3 *contribution_forces [[buffer(21)]],
    device atomic_uint *node_contact_counts [[buffer(22)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= constants.node_count || constants.enabled == 0u) return;
    float3 position_change = 0.0f;
    float3 velocity_change = 0.0f;
    float3 force = 0.0f;
    for (uint particle = 0u; particle < fluid_metadata.count; ++particle) {
        const uint contribution = particle * 12u;
        for (uint item = 0u; item < 12u; ++item) {
            const uint index = contribution + item;
            if (contribution_nodes[index] != node) continue;
            position_change += pm_load(contribution_positions[index]);
            velocity_change += pm_load(contribution_changes[index]);
            force += pm_load(contribution_forces[index]);
        }
    }
    node_positions[node] = pm_store(
        pm_load(node_positions[node]) + position_change);
    node_velocities[node] = pm_store(pm_limit(
        pm_load(node_velocities[node]) + velocity_change,
        constants.maximum_soft_speed));
    node_forces[node] = pm_store(pm_load(node_forces[node]) + force);
    atomic_store_explicit(
        node_contact_counts + node, 0u, memory_order_relaxed);
    if (node == 0u)
        atomic_store_explicit(
            node_contact_counts + constants.node_count, 0u,
            memory_order_relaxed);
}

kernel void pm_fluid_soft_surface_update(
    device const PMPackedVec3 *node_positions [[buffer(6)]],
    device PMPackedVec3 *surface_positions [[buffer(9)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(11)]],
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    device const PMPackedVec3 *node_rest_positions [[buffer(14)]],
    device const PMPackedVec3 *surface_rest_positions [[buffer(15)]],
    uint surface [[thread_position_in_grid]]) {
    if (surface >= constants.surface_count || constants.enabled == 0u) return;
    const PMMetalSurfaceBinding binding = surface_bindings[surface];
    float3 point = pm_load(surface_rest_positions[surface]);
    for (uint slot = 0u; slot < 4u; ++slot) {
        const uint node = binding.nodes[slot];
        point += (pm_load(node_positions[node]) -
                  pm_load(node_rest_positions[node])) *
                 binding.weights[slot];
    }
    surface_positions[surface] = pm_store(point);
}

kernel void pm_fluid_soft_sweep_on(
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    device atomic_uint *node_contact_counts [[buffer(22)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index == 0u)
        atomic_store_explicit(
            node_contact_counts + constants.node_count, 1u,
            memory_order_relaxed);
}

kernel void pm_fluid_soft_recover(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device const PMPackedVec3 *fluid_previous [[buffer(1)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(5)]],
    device const PMPackedVec3 *surface_positions [[buffer(9)]],
    device const uint *surface_indices [[buffer(10)]],
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    device const PMPackedVec3 *previous_surface [[buffer(23)]],
    uint particle [[thread_position_in_grid]]) {
    if (particle >= fluid_metadata.count || constants.enabled == 0u) return;
    const PMFluidSoftNearest hit = pm_fluid_soft_nearest(
        pm_load(fluid_positions[particle]),
        pm_load(fluid_previous[particle]), surface_positions,
        previous_surface, surface_indices, constants.surface_index_count,
        constants.orientation, constants.contact_distance, true);
    if (hit.triangle != 0xffffffffu)
        fluid_positions[particle] = pm_store(
            pm_load(fluid_positions[particle]) +
            hit.normal * hit.penetration);
}

kernel void pm_fluid_soft_recover_current(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(5)]],
    device const PMPackedVec3 *surface_positions [[buffer(9)]],
    device const uint *surface_indices [[buffer(10)]],
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    uint particle [[thread_position_in_grid]]) {
    if (particle >= fluid_metadata.count || constants.enabled == 0u) return;
    const float3 point = pm_load(fluid_positions[particle]);
    const PMFluidSoftNearest hit = pm_fluid_soft_nearest(
        point, point, surface_positions, surface_positions,
        surface_indices, constants.surface_index_count,
        constants.orientation, constants.contact_distance, false);
    if (hit.triangle != 0xffffffffu)
        fluid_positions[particle] = pm_store(
            point + hit.normal * hit.penetration);
}

kernel void pm_fluid_soft_copy_surface(
    device const PMPackedVec3 *surface_positions [[buffer(9)]],
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    device PMPackedVec3 *previous_surface [[buffer(23)]],
    uint surface [[thread_position_in_grid]]) {
    if (surface >= constants.surface_count || constants.enabled == 0u) return;
    previous_surface[surface] = surface_positions[surface];
}

kernel void pm_fluid_soft_serial(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device const PMPackedVec3 *fluid_previous [[buffer(1)]],
    device PMPackedVec3 *fluid_velocities [[buffer(2)]],
    device PMPackedVec3 *fluid_accelerations [[buffer(3)]],
    device float *fluid_foam [[buffer(4)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(5)]],
    device PMPackedVec3 *node_positions [[buffer(6)]],
    device PMPackedVec3 *node_velocities [[buffer(7)]],
    device const float *node_inverse_masses [[buffer(8)]],
    device PMPackedVec3 *surface_positions [[buffer(9)]],
    device const uint *surface_indices [[buffer(10)]],
    device const PMMetalSurfaceBinding *surface_bindings [[buffer(11)]],
    device PMPackedVec3 *node_forces [[buffer(12)]],
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    device const PMPackedVec3 *node_rest_positions [[buffer(14)]],
    device const PMPackedVec3 *surface_rest_positions [[buffer(15)]],
    device uint *diagnostic_contact_count [[buffer(16)]],
    device float *diagnostic_maximum_penetration [[buffer(17)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u || constants.enabled == 0u) return;
    const float particle_inverse =
        1.0f / max(constants.particle_mass, 1.0e-12f);
    uint contact_count = 0u;
    float maximum_penetration = 0.0f;
    for (uint particle = 0u; particle < fluid_metadata.count; ++particle) {
        const float3 start = pm_load(fluid_previous[particle]);
        float3 point = pm_load(fluid_positions[particle]);
        float nearest_squared = INFINITY;
        float earliest = 2.0f;
        float3 nearest = 0.0f;
        float3 normal = 0.0f;
        float3 weights = 0.0f;
        float penetration = 0.0f;
        uint best = constants.surface_index_count;
        for (uint base = 0u; base + 2u < constants.surface_index_count;
             base += 3u) {
            const float3 a = pm_load(surface_positions[surface_indices[base]]);
            const float3 b =
                pm_load(surface_positions[surface_indices[base + 1u]]);
            const float3 c =
                pm_load(surface_positions[surface_indices[base + 2u]]);
            const float3 face = cross(b - a, c - a);
            const float face_squared = dot(face, face);
            if (face_squared <= 1.0e-14f) continue;
            const float3 outward =
                face * rsqrt(face_squared) * constants.orientation;
            const float first_side = dot(start - a, outward);
            const float last_side = dot(point - a, outward);
            if (first_side >= constants.contact_distance &&
                last_side < constants.contact_distance &&
                first_side - last_side > 1.0e-8f) {
                const float fraction =
                    (first_side - constants.contact_distance) /
                    (first_side - last_side);
                if (fraction < earliest) {
                    const float3 center = start + (point - start) * fraction;
                    const float3 on_face =
                        center - outward * constants.contact_distance;
                    const float3 candidate =
                        pm_closest_point_triangle(on_face, a, b, c);
                    if (dot(on_face - candidate, on_face - candidate) <
                        1.0e-8f) {
                        earliest = fraction;
                        best = base;
                        nearest = candidate;
                        normal = outward;
                        weights = pm_triangle_weights(candidate, a, b, c);
                        penetration =
                            constants.contact_distance - last_side;
                    }
                }
            }
            if (earliest <= 1.0f) continue;
            const float3 candidate = pm_closest_point_triangle(point, a, b, c);
            const float squared = dot(point - candidate, point - candidate);
            if (squared >= nearest_squared) continue;
            nearest_squared = squared;
            nearest = candidate;
            weights = pm_triangle_weights(candidate, a, b, c);
            const float separation = sqrt(max(squared, 0.0f));
            const float signed_distance = dot(point - candidate, outward);
            if (signed_distance >= 0.0f) {
                penetration = constants.contact_distance - separation;
                normal = separation > 1.0e-6f
                             ? (point - candidate) / separation
                             : outward;
            } else {
                penetration = constants.contact_distance + separation;
                normal = separation > 1.0e-6f
                             ? (candidate - point) / separation
                             : outward;
            }
            best = base;
        }
        if (best == constants.surface_index_count || penetration <= 0.0f)
            continue;
        ++contact_count;
        maximum_penetration = max(maximum_penetration, penetration);

        uint contact_nodes[12];
        float contact_weights[12];
        uint contact_count = 0u;
        const float corner_weights[3] = {weights.x, weights.y, weights.z};
        for (uint corner = 0u; corner < 3u; ++corner) {
            const PMMetalSurfaceBinding binding =
                surface_bindings[surface_indices[best + corner]];
            for (uint slot = 0u; slot < 4u; ++slot) {
                const float weight =
                    corner_weights[corner] * binding.weights[slot];
                if (weight <= 0.0f) continue;
                const uint node = binding.nodes[slot];
                uint item = 0u;
                while (item < contact_count && contact_nodes[item] != node)
                    ++item;
                if (item == contact_count && contact_count < 12u) {
                    contact_nodes[contact_count] = node;
                    contact_weights[contact_count] = 0.0f;
                    ++contact_count;
                }
                if (item < contact_count) contact_weights[item] += weight;
            }
        }
        if (contact_count == 0u) continue;
        float denominator = particle_inverse;
        float3 surface_velocity = 0.0f;
        for (uint item = 0u; item < contact_count; ++item) {
            const uint node = contact_nodes[item];
            const float weight = contact_weights[item];
            denominator +=
                weight * weight * node_inverse_masses[node];
            surface_velocity +=
                pm_load(node_velocities[node]) * weight;
        }
        if (denominator <= 1.0e-12f) continue;
        const float lambda =
            min(penetration, 2.0f * constants.contact_distance) /
            denominator;
        point += normal * (particle_inverse * lambda);
        float3 velocity = pm_load(fluid_velocities[particle]);
        const float3 relative = velocity - surface_velocity;
        const float normal_speed = dot(relative, normal);
        const float normal_impulse = max(0.0f, -normal_speed) / denominator;
        const float3 tangent = relative - normal * normal_speed;
        const float3 impulse =
            normal * normal_impulse -
            pm_limit(tangent / denominator,
                     constants.friction * normal_impulse);
        const float3 velocity_change = impulse * particle_inverse;
        velocity = pm_limit(velocity + velocity_change,
                            constants.maximum_particle_speed);
        fluid_positions[particle] = pm_store(point);
        fluid_velocities[particle] = pm_store(velocity);
        fluid_accelerations[particle] = pm_store(
            pm_load(fluid_accelerations[particle]) +
            velocity_change / max(constants.timestep, 1.0e-12f));
        fluid_foam[particle] = max(
            fluid_foam[particle],
            min(1.0f, normal_impulse * particle_inverse * 0.1f));
        for (uint item = 0u; item < contact_count; ++item) {
            const uint node = contact_nodes[item];
            const float weight = contact_weights[item];
            const float inverse = node_inverse_masses[node];
            if (inverse > 0.0f) {
                node_positions[node] = pm_store(
                    pm_load(node_positions[node]) -
                    normal * (weight * inverse * lambda));
                node_velocities[node] = pm_store(pm_limit(
                    pm_load(node_velocities[node]) -
                        impulse * (weight * inverse),
                    constants.maximum_soft_speed));
            }
            node_forces[node] = pm_store(
                pm_load(node_forces[node]) -
                impulse * (weight / max(constants.timestep, 1.0e-12f)));
        }
    }
    for (uint surface = 0u; surface < constants.surface_count; ++surface) {
        const PMMetalSurfaceBinding binding = surface_bindings[surface];
        float3 point = pm_load(surface_rest_positions[surface]);
        for (uint slot = 0u; slot < 4u; ++slot) {
            const uint node = binding.nodes[slot];
            point += (pm_load(node_positions[node]) -
                      pm_load(node_rest_positions[node])) *
                     binding.weights[slot];
        }
        surface_positions[surface] = pm_store(point);
    }
    diagnostic_contact_count[0] += contact_count;
    diagnostic_maximum_penetration[0] = max(
        diagnostic_maximum_penetration[0], maximum_penetration);
}

kernel void pm_soft_cloth_contact(
    device const PMPackedVec3 *soft_positions [[buffer(0)]],
    device const PMPackedVec3 *soft_previous [[buffer(1)]],
    device const PMPackedVec3 *soft_velocities [[buffer(2)]],
    device const float *soft_inverse_masses [[buffer(3)]],
    device const PMPackedVec3 *cloth_positions [[buffer(5)]],
    device const PMPackedVec3 *cloth_previous [[buffer(6)]],
    device const PMPackedVec3 *cloth_velocities [[buffer(7)]],
    device const float *cloth_inverse_masses [[buffer(8)]],
    device const PMPackedVec3 *cloth_surface_positions [[buffer(9)]],
    device const uint *cloth_surface_indices [[buffer(10)]],
    device const uint *cloth_surface_sources [[buffer(11)]],
    constant PMSoftClothConstants &constants [[buffer(13)]],
    device PMPackedVec3 *soft_surface_positions [[buffer(14)]],
    device const PMMetalSurfaceBinding *soft_surface_bindings [[buffer(15)]],
    device const PMPackedVec3 *soft_rest_positions [[buffer(16)]],
    device const PMPackedVec3 *soft_surface_rest_positions [[buffer(17)]],
    device PMSoftClothContact *contacts [[buffer(22)]],
    device atomic_uint *contact_counts [[buffer(23)]],
    uint soft_node [[thread_position_in_grid]]) {
    (void)cloth_positions;
    (void)soft_surface_positions;
    (void)soft_surface_bindings;
    (void)soft_rest_positions;
    (void)soft_surface_rest_positions;
    if (soft_node >= constants.soft_count || constants.enabled == 0u) return;
    contacts[soft_node].active = 0u;
    const float soft_inverse = soft_inverse_masses[soft_node];
    if (soft_inverse <= 0.0f) return;
    float3 point = pm_load(soft_positions[soft_node]);
    const float3 start = pm_load(soft_previous[soft_node]);
    float nearest_squared = INFINITY;
    float3 nearest = 0.0f;
    float3 best_normal = 0.0f;
    float3 best_weights = 0.0f;
    uint best = constants.cloth_surface_index_count;
    for (uint base = 0u;
         base + 2u < constants.cloth_surface_index_count;
         base += 3u) {
        const uint surface_a = cloth_surface_indices[base];
        const uint surface_b = cloth_surface_indices[base + 1u];
        const uint surface_c = cloth_surface_indices[base + 2u];
        const float3 a = pm_load(cloth_surface_positions[surface_a]);
        const float3 b = pm_load(cloth_surface_positions[surface_b]);
        const float3 c = pm_load(cloth_surface_positions[surface_c]);
        const float3 face = cross(b - a, c - a);
        const float face_squared = dot(face, face);
        if (face_squared <= 1.0e-14f) continue;
        const float3 face_normal = face * rsqrt(face_squared);
        const float3 candidate = pm_closest_point_triangle(point, a, b, c);
        const float3 delta = point - candidate;
        const float squared = dot(delta, delta);
        const float3 weights = pm_triangle_weights(candidate, a, b, c);
        const uint physical_a = cloth_surface_sources[surface_a];
        const uint physical_b = cloth_surface_sources[surface_b];
        const uint physical_c = cloth_surface_sources[surface_c];
        const float3 old_nearest =
            pm_load(cloth_previous[physical_a]) * weights.x +
            pm_load(cloth_previous[physical_b]) * weights.y +
            pm_load(cloth_previous[physical_c]) * weights.z;
        const float before = dot(start - old_nearest, face_normal);
        const float after = dot(delta, face_normal);
        const float travel =
            length(point - start) + length(candidate - old_nearest);
        const bool crossed = before * after < 0.0f &&
            squared <= (travel + constants.contact_distance) *
                           (travel + constants.contact_distance);
        if (!crossed &&
            squared >= constants.contact_distance * constants.contact_distance)
            continue;
        if (squared >= nearest_squared) continue;
        nearest_squared = squared;
        nearest = candidate;
        best_weights = weights;
        best_normal = crossed || squared <= 1.0e-14f
                          ? face_normal * (before >= 0.0f ? 1.0f : -1.0f)
                          : delta * rsqrt(squared);
        best = base;
    }
    if (best == constants.cloth_surface_index_count) return;
    const float penetration =
        constants.contact_distance - dot(point - nearest, best_normal);
    if (penetration <= 0.0f) return;
    const uint surface_vertices[3] = {
        cloth_surface_indices[best],
        cloth_surface_indices[best + 1u],
        cloth_surface_indices[best + 2u]};
    const uint physical_vertices[3] = {
        cloth_surface_sources[surface_vertices[0]],
        cloth_surface_sources[surface_vertices[1]],
        cloth_surface_sources[surface_vertices[2]]};
    const float corner_weights[3] = {
        best_weights.x, best_weights.y, best_weights.z};
    float denominator = soft_inverse;
    float3 surface_velocity = 0.0f;
    for (uint corner = 0u; corner < 3u; ++corner) {
        const uint cloth_node = physical_vertices[corner];
        const float weight = corner_weights[corner];
        denominator += weight * weight * cloth_inverse_masses[cloth_node];
        surface_velocity += pm_load(cloth_velocities[cloth_node]) * weight;
    }
    if (denominator <= 1.0e-12f) return;
    PMSoftClothContact contact{};
    contact.position_impulse = pm_store(
        best_normal *
        (min(penetration, 2.0f * constants.contact_distance) /
         denominator));
    contact.soft_inverse_mass_fraction = soft_inverse / denominator;
    const float3 relative =
        pm_load(soft_velocities[soft_node]) - surface_velocity;
    const float normal_speed = dot(relative, best_normal);
    const float normal_impulse = max(0.0f, -normal_speed) / denominator;
    const float3 tangent = relative - best_normal * normal_speed;
    const float3 velocity_impulse =
        best_normal * normal_impulse -
        pm_limit(tangent / denominator,
                 constants.friction * normal_impulse);
    contact.velocity_impulse = pm_store(velocity_impulse);
    for (uint corner = 0u; corner < 3u; ++corner) {
        const uint cloth_node = physical_vertices[corner];
        const float weight = corner_weights[corner];
        contact.vertices[corner] = cloth_node;
        contact.weights[corner] = weight;
        if (weight > 1.0e-6f &&
            cloth_inverse_masses[cloth_node] > 0.0f)
            atomic_fetch_add_explicit(
                contact_counts + cloth_node, 1u,
                memory_order_relaxed);
    }
    contact.active = 1u;
    contacts[soft_node] = contact;
}

kernel void pm_soft_cloth_clear_counts(
    constant PMSoftClothConstants &constants [[buffer(13)]],
    device atomic_uint *contact_counts [[buffer(23)]],
    uint cloth_node [[thread_position_in_grid]]) {
    if (cloth_node < constants.cloth_count)
        atomic_store_explicit(
            contact_counts + cloth_node, 0u, memory_order_relaxed);
}

static float pm_soft_cloth_relaxation(
    PMSoftClothContact contact,
    device const atomic_uint *contact_counts) {
    uint degree = 1u;
    for (uint corner = 0u; corner < 3u; ++corner)
        if (contact.weights[corner] > 1.0e-6f)
            degree = max(
                degree,
                atomic_load_explicit(
                    contact_counts + contact.vertices[corner],
                    memory_order_relaxed));
    const float soft_fraction = contact.soft_inverse_mass_fraction;
    return 1.0f /
        (soft_fraction + float(degree) * (1.0f - soft_fraction));
}

kernel void pm_soft_cloth_apply_soft(
    device PMPackedVec3 *soft_positions [[buffer(0)]],
    device PMPackedVec3 *soft_velocities [[buffer(2)]],
    device const float *soft_inverse_masses [[buffer(3)]],
    device PMPackedVec3 *soft_forces [[buffer(4)]],
    constant PMSoftClothConstants &constants [[buffer(13)]],
    device const PMSoftClothContact *contacts [[buffer(22)]],
    device const atomic_uint *contact_counts [[buffer(23)]],
    uint soft_node [[thread_position_in_grid]]) {
    if (soft_node >= constants.soft_count || constants.enabled == 0u ||
        contacts[soft_node].active == 0u)
        return;
    const PMSoftClothContact contact = contacts[soft_node];
    const float relaxation =
        pm_soft_cloth_relaxation(contact, contact_counts);
    const float inverse = soft_inverse_masses[soft_node];
    soft_positions[soft_node] = pm_store(
        pm_load(soft_positions[soft_node]) +
        pm_load(contact.position_impulse) * (relaxation * inverse));
    const float3 impulse =
        pm_load(contact.velocity_impulse) * relaxation;
    soft_velocities[soft_node] = pm_store(pm_limit(
        pm_load(soft_velocities[soft_node]) + impulse * inverse,
        constants.maximum_soft_speed));
    soft_forces[soft_node] = pm_store(
        pm_load(soft_forces[soft_node]) +
        impulse / max(constants.timestep, 1.0e-12f));
}

kernel void pm_soft_cloth_apply(
    device PMPackedVec3 *cloth_positions [[buffer(5)]],
    device PMPackedVec3 *cloth_velocities [[buffer(7)]],
    device const float *cloth_inverse_masses [[buffer(8)]],
    device PMPackedVec3 *cloth_forces [[buffer(12)]],
    constant PMSoftClothConstants &constants [[buffer(13)]],
    device const PMSoftClothContact *contacts [[buffer(22)]],
    device const atomic_uint *contact_counts [[buffer(23)]],
    uint cloth_node [[thread_position_in_grid]]) {
    if (cloth_node >= constants.cloth_count || constants.enabled == 0u)
        return;
    float3 position_change = 0.0f;
    float3 velocity_change = 0.0f;
    float3 force = 0.0f;
    for (uint soft_node = 0u; soft_node < constants.soft_count; ++soft_node) {
        const PMSoftClothContact contact = contacts[soft_node];
        if (contact.active == 0u) continue;
        const float relaxation =
            pm_soft_cloth_relaxation(contact, contact_counts);
        for (uint corner = 0u; corner < 3u; ++corner) {
            if (contact.vertices[corner] != cloth_node) continue;
            const float scale = -contact.weights[corner] * relaxation;
            position_change +=
                pm_load(contact.position_impulse) * scale;
            velocity_change +=
                pm_load(contact.velocity_impulse) * scale;
        }
    }
    const float inverse = cloth_inverse_masses[cloth_node];
    cloth_positions[cloth_node] = pm_store(
        pm_load(cloth_positions[cloth_node]) + position_change * inverse);
    cloth_velocities[cloth_node] = pm_store(pm_limit(
        pm_load(cloth_velocities[cloth_node]) + velocity_change * inverse,
        20.0f));
    force = velocity_change / max(constants.timestep, 1.0e-12f);
    cloth_forces[cloth_node] = pm_store(
        pm_load(cloth_forces[cloth_node]) + force);
}

kernel void pm_soft_cloth_surface_update(
    device const PMPackedVec3 *cloth_positions [[buffer(5)]],
    device PMPackedVec3 *cloth_surface_positions [[buffer(9)]],
    device const uint *cloth_surface_sources [[buffer(11)]],
    constant PMSoftClothConstants &constants [[buffer(13)]],
    uint surface [[thread_position_in_grid]]) {
    if (surface >= constants.cloth_surface_count || constants.enabled == 0u)
        return;
    cloth_surface_positions[surface] =
        cloth_positions[cloth_surface_sources[surface]];
}

kernel void pm_soft_cloth_soft_surface_update(
    device const PMPackedVec3 *soft_positions [[buffer(0)]],
    constant PMSoftClothConstants &constants [[buffer(13)]],
    device PMPackedVec3 *soft_surface_positions [[buffer(14)]],
    device const PMMetalSurfaceBinding *soft_surface_bindings [[buffer(15)]],
    device const PMPackedVec3 *soft_rest_positions [[buffer(16)]],
    device const PMPackedVec3 *soft_surface_rest_positions [[buffer(17)]],
    uint surface [[thread_position_in_grid]]) {
    if (surface >= constants.soft_surface_count || constants.enabled == 0u)
        return;
    const PMMetalSurfaceBinding binding = soft_surface_bindings[surface];
    float3 point = pm_load(soft_surface_rest_positions[surface]);
    for (uint slot = 0u; slot < 4u; ++slot) {
        const uint node = binding.nodes[slot];
        point += (pm_load(soft_positions[node]) -
                  pm_load(soft_rest_positions[node])) *
                 binding.weights[slot];
    }
    soft_surface_positions[surface] = pm_store(point);
}

kernel void pm_soft_cloth_serial(
    device PMPackedVec3 *soft_positions [[buffer(0)]],
    device const PMPackedVec3 *soft_previous [[buffer(1)]],
    device PMPackedVec3 *soft_velocities [[buffer(2)]],
    device const float *soft_inverse_masses [[buffer(3)]],
    device PMPackedVec3 *soft_forces [[buffer(4)]],
    device PMPackedVec3 *cloth_positions [[buffer(5)]],
    device const PMPackedVec3 *cloth_previous [[buffer(6)]],
    device PMPackedVec3 *cloth_velocities [[buffer(7)]],
    device const float *cloth_inverse_masses [[buffer(8)]],
    device PMPackedVec3 *cloth_surface_positions [[buffer(9)]],
    device const uint *cloth_surface_indices [[buffer(10)]],
    device const uint *cloth_surface_sources [[buffer(11)]],
    device PMPackedVec3 *cloth_forces [[buffer(12)]],
    constant PMSoftClothConstants &constants [[buffer(13)]],
    device PMPackedVec3 *soft_surface_positions [[buffer(14)]],
    device const PMMetalSurfaceBinding *soft_surface_bindings [[buffer(15)]],
    device const PMPackedVec3 *soft_rest_positions [[buffer(16)]],
    device const PMPackedVec3 *soft_surface_rest_positions [[buffer(17)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u || constants.enabled == 0u) return;
    const uint iterations = max(constants.solver_iterations, 1u);
    for (uint pass = 0u; pass < iterations; ++pass) {
        for (uint soft_node = 0u; soft_node < constants.soft_count;
             ++soft_node) {
            const float soft_inverse = soft_inverse_masses[soft_node];
            if (soft_inverse <= 0.0f) continue;
            float3 point = pm_load(soft_positions[soft_node]);
            const float3 start = pm_load(soft_previous[soft_node]);
            float nearest_squared = INFINITY;
            float3 nearest = 0.0f;
            float3 best_normal = 0.0f;
            float3 best_weights = 0.0f;
            uint best = constants.cloth_surface_index_count;
            for (uint base = 0u;
                 base + 2u < constants.cloth_surface_index_count;
                 base += 3u) {
                const uint surface_a = cloth_surface_indices[base];
                const uint surface_b = cloth_surface_indices[base + 1u];
                const uint surface_c = cloth_surface_indices[base + 2u];
                const float3 a = pm_load(cloth_surface_positions[surface_a]);
                const float3 b = pm_load(cloth_surface_positions[surface_b]);
                const float3 c = pm_load(cloth_surface_positions[surface_c]);
                const float3 face = cross(b - a, c - a);
                const float face_squared = dot(face, face);
                if (face_squared <= 1.0e-14f) continue;
                const float3 face_normal = face * rsqrt(face_squared);
                const float3 candidate =
                    pm_closest_point_triangle(point, a, b, c);
                const float3 delta = point - candidate;
                const float squared = dot(delta, delta);
                const float3 weights =
                    pm_triangle_weights(candidate, a, b, c);
                const uint physical_a = cloth_surface_sources[surface_a];
                const uint physical_b = cloth_surface_sources[surface_b];
                const uint physical_c = cloth_surface_sources[surface_c];
                const float3 old_nearest =
                    pm_load(cloth_previous[physical_a]) * weights.x +
                    pm_load(cloth_previous[physical_b]) * weights.y +
                    pm_load(cloth_previous[physical_c]) * weights.z;
                const float before = dot(start - old_nearest, face_normal);
                const float after = dot(delta, face_normal);
                const float travel = length(point - start) +
                                     length(candidate - old_nearest);
                const bool crossed = before * after < 0.0f &&
                    squared <= (travel + constants.contact_distance) *
                                   (travel + constants.contact_distance);
                if (!crossed &&
                    squared >= constants.contact_distance *
                                   constants.contact_distance)
                    continue;
                if (squared >= nearest_squared) continue;
                nearest_squared = squared;
                nearest = candidate;
                best_weights = weights;
                best_normal = crossed || squared <= 1.0e-14f
                                  ? face_normal *
                                        (before >= 0.0f ? 1.0f : -1.0f)
                                  : delta * rsqrt(squared);
                best = base;
            }
            if (best == constants.cloth_surface_index_count) continue;
            const float penetration =
                constants.contact_distance - dot(point - nearest, best_normal);
            if (penetration <= 0.0f) continue;
            const uint surface_vertices[3] = {
                cloth_surface_indices[best],
                cloth_surface_indices[best + 1u],
                cloth_surface_indices[best + 2u]};
            const uint physical_vertices[3] = {
                cloth_surface_sources[surface_vertices[0]],
                cloth_surface_sources[surface_vertices[1]],
                cloth_surface_sources[surface_vertices[2]]};
            const float corner_weights[3] = {
                best_weights.x, best_weights.y, best_weights.z};
            float denominator = soft_inverse;
            float3 surface_velocity = 0.0f;
            for (uint corner = 0u; corner < 3u; ++corner) {
                const uint cloth_node = physical_vertices[corner];
                const float weight = corner_weights[corner];
                denominator += weight * weight *
                               cloth_inverse_masses[cloth_node];
                surface_velocity +=
                    pm_load(cloth_velocities[cloth_node]) * weight;
            }
            if (denominator <= 1.0e-12f) continue;
            const float position_lambda =
                min(penetration, 2.0f * constants.contact_distance) /
                denominator;
            point += best_normal * (position_lambda * soft_inverse);
            soft_positions[soft_node] = pm_store(point);
            const float3 relative =
                pm_load(soft_velocities[soft_node]) - surface_velocity;
            const float normal_speed = dot(relative, best_normal);
            const float normal_impulse = max(0.0f, -normal_speed) /
                                         denominator;
            const float3 tangent =
                relative - best_normal * normal_speed;
            const float3 velocity_impulse =
                best_normal * normal_impulse -
                pm_limit(tangent / denominator,
                         constants.friction * normal_impulse);
            soft_velocities[soft_node] = pm_store(pm_limit(
                pm_load(soft_velocities[soft_node]) +
                    velocity_impulse * soft_inverse,
                constants.maximum_soft_speed));
            soft_forces[soft_node] = pm_store(
                pm_load(soft_forces[soft_node]) +
                velocity_impulse / max(constants.timestep, 1.0e-12f));
            for (uint corner = 0u; corner < 3u; ++corner) {
                const uint cloth_node = physical_vertices[corner];
                const float weight = corner_weights[corner];
                const float inverse = cloth_inverse_masses[cloth_node];
                if (inverse > 0.0f) {
                    cloth_positions[cloth_node] = pm_store(
                        pm_load(cloth_positions[cloth_node]) -
                        best_normal *
                            (position_lambda * weight * inverse));
                    cloth_velocities[cloth_node] = pm_store(pm_limit(
                        pm_load(cloth_velocities[cloth_node]) -
                            velocity_impulse * (weight * inverse),
                        20.0f));
                }
                cloth_forces[cloth_node] = pm_store(
                    pm_load(cloth_forces[cloth_node]) -
                    velocity_impulse *
                        (weight / max(constants.timestep, 1.0e-12f)));
            }
        }
        for (uint item = 0u;
             item < constants.cloth_surface_index_count; ++item) {
            const uint surface_vertex = cloth_surface_indices[item];
            cloth_surface_positions[surface_vertex] =
                cloth_positions[cloth_surface_sources[surface_vertex]];
        }
    }
    for (uint surface = 0u; surface < constants.soft_surface_count; ++surface) {
        const PMMetalSurfaceBinding binding = soft_surface_bindings[surface];
        float3 point = pm_load(soft_surface_rest_positions[surface]);
        for (uint slot = 0u; slot < 4u; ++slot) {
            const uint node = binding.nodes[slot];
            point += (pm_load(soft_positions[node]) -
                      pm_load(soft_rest_positions[node])) *
                     binding.weights[slot];
        }
        soft_surface_positions[surface] = pm_store(point);
    }
}

kernel void pm_fluid_smoke_update(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device PMPackedVec3 *fluid_velocities [[buffer(2)]],
    device float *fluid_temperatures [[buffer(7)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(8)]],
    device const PMPackedVec3 *smoke_positions [[buffer(9)]],
    device const PMPackedVec3 *smoke_velocities [[buffer(11)]],
    device const float *smoke_ages [[buffer(12)]],
    device const PMSmokeMetadata &smoke_metadata [[buffer(17)]],
    constant PMFluidSmokeConstants &constants [[buffer(18)]],
    device const float *grid_density [[buffer(21)]],
    device const float *grid_face_velocity [[buffer(22)]],
    uint source [[thread_position_in_grid]]) {
    const uint source_count = min(fluid_metadata.count,
                                  constants.fluid_capacity);
    if (source >= source_count) return;
    const uint smoke_count = min(smoke_metadata.count,
                                 constants.smoke_capacity);
    const float support = 3.0f * constants.smoke_radius;
    const float support_squared = support * support;
    const float3 position = pm_load(fluid_positions[source]);
    float3 velocity = pm_load(fluid_velocities[source]);
    float smoke_weight = 0.0f;
    float3 smoke_velocity = 0.0f;
    float occupancy = 0.0f;
    const float3 grid_point =
        (position - pm_load(constants.grid_minimum)) /
        max(constants.grid_spacing, 1.0e-6f);
    const bool inside_grid = constants.grid_resolution != 0u &&
        smoke_count != 0u && grid_point.x >= 0.0f &&
        grid_point.y >= 0.0f && grid_point.z >= 0.0f &&
        grid_point.x < float(constants.grid_resolution) &&
        grid_point.y < float(constants.grid_vertical_resolution) &&
        grid_point.z < float(constants.grid_resolution);
    if (inside_grid) {
        smoke_velocity = pm_smoke_sample_face_velocity(
            grid_face_velocity, position, pm_load(constants.grid_minimum),
            constants.grid_spacing, constants.grid_resolution,
            constants.grid_vertical_resolution);
        const float density = pm_smoke_sample_cell_scalar(
            grid_density, position, pm_load(constants.grid_minimum),
            constants.grid_spacing, constants.grid_resolution,
            constants.grid_vertical_resolution);
        occupancy = clamp(density * constants.smoke_rest_number_density,
                          0.0f, 1.0f);
    } else {
        for (uint smoke = 0u; smoke < smoke_count; ++smoke) {
            if (smoke_ages[smoke] >= constants.smoke_lifetime) continue;
            const float3 delta = position - pm_load(smoke_positions[smoke]);
            const float squared = dot(delta, delta);
            if (squared >= support_squared) continue;
            const float q = 1.0f - sqrt(max(squared, 0.0f)) / support;
            const float weight = q * q * q;
            smoke_weight += weight;
            smoke_velocity += pm_load(smoke_velocities[smoke]) * weight;
        }
        if (smoke_weight > 1.0e-6f) {
            smoke_velocity /= smoke_weight;
            occupancy = min(
                1.0f, smoke_weight / constants.smoke_rest_number_density);
        }
    }
    if (occupancy > 1.0e-6f) {
        const float response = 1.0f -
            exp(-constants.wind_drag * occupancy * constants.timestep);
        const float maximum = max(constants.smoke_maximum_speed,
                                  length(velocity));
        velocity = pm_limit(
            velocity + (smoke_velocity - velocity) * response, maximum);
    }
    fluid_velocities[source] = pm_store(velocity);
    const PMQuaternion inverse_heater =
        pm_quaternion_conjugate(constants.heater_orientation);
    const float3 local = pm_rotate(
        inverse_heater, position - pm_load(constants.heater_center));
    const bool heated =
        abs(local.x) <= constants.heater_half_extents.x &&
        abs(local.z) <= constants.heater_half_extents.y &&
        local.y >= -0.5f * constants.water_radius &&
        local.y <= 2.5f * constants.water_radius;
    float temperature = fluid_temperatures[source];
    if (heated && constants.heater_temperature > temperature) {
        temperature += (constants.heater_temperature - temperature) *
            (1.0f - exp(-constants.heat_transfer_rate * constants.timestep));
    }
    fluid_temperatures[source] = temperature;
}

kernel void pm_fluid_smoke_compact(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device PMPackedVec3 *fluid_previous [[buffer(1)]],
    device PMPackedVec3 *fluid_velocities [[buffer(2)]],
    device PMPackedVec3 *fluid_accelerations [[buffer(3)]],
    device uint *fluid_ids [[buffer(4)]],
    device float *fluid_foam [[buffer(5)]],
    device float *fluid_foam_sources [[buffer(6)]],
    device float *fluid_temperatures [[buffer(7)]],
    device PMParticleMetadata &fluid_metadata [[buffer(8)]],
    device PMPackedVec3 *smoke_positions [[buffer(9)]],
    device PMPackedVec3 *smoke_previous [[buffer(10)]],
    device PMPackedVec3 *smoke_velocities [[buffer(11)]],
    device float *smoke_ages [[buffer(12)]],
    device float *smoke_densities [[buffer(13)]],
    device float *smoke_pressures [[buffer(14)]],
    device PMPackedVec3 *smoke_vorticities [[buffer(15)]],
    device float *smoke_thermal_lift [[buffer(16)]],
    device PMSmokeMetadata &smoke_metadata [[buffer(17)]],
    constant PMFluidSmokeConstants &constants [[buffer(18)]],
    device PMContactEvent *contact_samples [[buffer(19)]],
    device uint *contact_flags [[buffer(20)]],
    device const float *grid_density [[buffer(21)]],
    device const float *grid_face_velocity [[buffer(22)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    threadgroup uint prefix[64];
    threadgroup uint destination_base;
    threadgroup uint converted_base;
    threadgroup uint smoke_count;
    const uint lane_count = threads_per_group.x;
    const uint source_count = min(fluid_metadata.count,
                                  constants.fluid_capacity);
    const uint first_smoke_particle = smoke_metadata.next_particle;
    (void)grid_density;
    (void)grid_face_velocity;
    const float gravity_squared = dot(pm_load(constants.gravity),
                                      pm_load(constants.gravity));
    const float3 up = gravity_squared > 1.0e-12f
                          ? -normalize(pm_load(constants.gravity))
                          : float3(0.0f, 1.0f, 0.0f);
    if (lane == 0u) {
        destination_base = 0u;
        converted_base = 0u;
        smoke_count = smoke_metadata.count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint block = 0u; block < source_count; block += lane_count) {
        const uint source = block + lane;
        const bool valid = source < source_count;
        PMPackedVec3 position{};
        PMPackedVec3 old_position{};
        PMPackedVec3 velocity{};
        PMPackedVec3 acceleration{};
        uint stable_id = 0u;
        float foam_value = 0.0f;
        float foam_source = 0.0f;
        float temperature = 0.0f;
        PMContactEvent contact_sample{};
        uint contact_flag = 0u;
        if (valid) {
            position = fluid_positions[source];
            old_position = fluid_previous[source];
            velocity = fluid_velocities[source];
            acceleration = fluid_accelerations[source];
            stable_id = fluid_ids[source];
            foam_value = fluid_foam[source];
            foam_source = fluid_foam_sources[source];
            temperature = fluid_temperatures[source];
            contact_sample = contact_samples[source];
            contact_flag = contact_flags[source];
        }

        const uint hot = uint(valid &&
                              temperature >= constants.boiling_temperature);
        prefix[lane] = hot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 1u; offset < lane_count; offset <<= 1u) {
            const uint addend = lane >= offset ? prefix[lane - offset] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix[lane] += addend;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const uint hot_count = prefix[lane_count - 1u];
        const uint ordinal = converted_base +
            (hot == 0u ? 0u : prefix[lane] - 1u);
        const uint accepted = uint(hot != 0u &&
                                   ordinal < constants.smoke_capacity);
        uint smoke_slot = 0u;
        if (accepted != 0u) {
            smoke_slot = (first_smoke_particle + ordinal) %
                         constants.smoke_capacity;
            smoke_positions[smoke_slot] = position;
            smoke_previous[smoke_slot] = position;
            smoke_velocities[smoke_slot] = pm_store(
                pm_load(velocity) + up * constants.steam_rise_speed);
            smoke_ages[smoke_slot] = 0.0f;
            smoke_densities[smoke_slot] = 0.0f;
            smoke_pressures[smoke_slot] = 0.0f;
            smoke_vorticities[smoke_slot] = {0.0f, 0.0f, 0.0f};
            smoke_thermal_lift[smoke_slot] = constants.steam_rise_speed;
        }
        threadgroup_barrier(mem_flags::mem_device);

        prefix[lane] = accepted != 0u ? smoke_slot + 1u : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = lane_count >> 1u; stride != 0u; stride >>= 1u) {
            if (lane < stride)
                prefix[lane] = max(prefix[lane], prefix[lane + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) smoke_count = max(smoke_count, prefix[0]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint keep = uint(valid && accepted == 0u);
        prefix[lane] = keep;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 1u; offset < lane_count; offset <<= 1u) {
            const uint addend = lane >= offset ? prefix[lane - offset] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix[lane] += addend;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (keep != 0u) {
            const uint destination = destination_base +
                (lane == 0u ? 0u : prefix[lane - 1u]);
            fluid_positions[destination] = position;
            fluid_previous[destination] = old_position;
            fluid_velocities[destination] = velocity;
            fluid_accelerations[destination] = acceleration;
            fluid_ids[destination] = stable_id;
            fluid_foam[destination] = foam_value;
            fluid_foam_sources[destination] = foam_source;
            fluid_temperatures[destination] = temperature;
            contact_samples[destination] = contact_sample;
            contact_flags[destination] = contact_flag;
        }
        threadgroup_barrier(mem_flags::mem_device |
                            mem_flags::mem_threadgroup);
        if (lane == 0u) {
            const uint accepted_count = min(
                hot_count, constants.smoke_capacity - converted_base);
            converted_base += accepted_count;
            destination_base += prefix[lane_count - 1u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        fluid_metadata.count = destination_base;
        fluid_metadata.boiled += converted_base;
        if (converted_base != 0u) ++fluid_metadata.revision;
        smoke_metadata.next_particle = first_smoke_particle + converted_base;
        smoke_metadata.count = min(constants.smoke_capacity, smoke_count);
        smoke_metadata.emitted += converted_base;
        smoke_metadata.revision += converted_base;
    }
}

kernel void pm_particle_coupling_serial(
    device PMPackedVec3 *positions_a [[buffer(0)]],
    device PMPackedVec3 *velocities_a [[buffer(1)]],
    device const float *inverse_masses_a [[buffer(2)]],
    device PMPackedVec3 *positions_b [[buffer(3)]],
    device PMPackedVec3 *velocities_b [[buffer(4)]],
    device const float *inverse_masses_b [[buffer(5)]],
    constant PMCouplingConstants &constants [[buffer(6)]],
    device PMPackedVec3 *diagnostics_a [[buffer(7)]],
    device PMPackedVec3 *diagnostics_b [[buffer(8)]],
    device const PMParticleMetadata &particle_metadata [[buffer(9)]],
    device uint *diagnostic_contact_count [[buffer(10)]],
    device float *diagnostic_maximum_penetration [[buffer(11)]],
    uint thread_index [[thread_position_in_grid]]) {
    if (thread_index != 0u || constants.enabled == 0u) return;
    if (constants.mode == 1u) {
        if (constants.first_vertex != 0xffffffffu && constants.count_a != 0u &&
            constants.first_vertex < constants.count_b) {
            const uint cloth_node = constants.first_vertex;
            float3 cloth_position = pm_load(positions_b[cloth_node]);
            float3 cloth_velocity = pm_load(velocities_b[cloth_node]);
            if (inverse_masses_b[cloth_node] > 0.0f) {
                const float3 movement = pm_limit(
                    pm_load(positions_a[0]) - cloth_position,
                    constants.maximum_force * constants.timestep *
                        constants.timestep);
                const float3 velocity_change =
                    movement / max(constants.timestep, 1.0e-12f);
                cloth_position += movement;
                cloth_velocity += velocity_change;
                positions_b[cloth_node] = pm_store(cloth_position);
                velocities_b[cloth_node] = pm_store(cloth_velocity);
                diagnostics_b[cloth_node] = pm_store(
                    pm_load(diagnostics_b[cloth_node]) +
                    velocity_change *
                        (constants.damping /
                         inverse_masses_b[cloth_node]));
            }
            positions_a[0] = pm_store(cloth_position);
            velocities_a[0] = pm_store(cloth_velocity);
        }
        if (constants.last_vertex != 0xffffffffu && constants.count_a != 0u &&
            constants.last_vertex < constants.count_b) {
            const uint endpoint = constants.count_a - 1u;
            const uint cloth_node = constants.last_vertex;
            float3 cloth_position = pm_load(positions_b[cloth_node]);
            float3 cloth_velocity = pm_load(velocities_b[cloth_node]);
            if (inverse_masses_b[cloth_node] > 0.0f) {
                const float3 movement = pm_limit(
                    pm_load(positions_a[endpoint]) - cloth_position,
                    constants.maximum_force * constants.timestep *
                        constants.timestep);
                const float3 velocity_change =
                    movement / max(constants.timestep, 1.0e-12f);
                cloth_position += movement;
                cloth_velocity += velocity_change;
                positions_b[cloth_node] = pm_store(cloth_position);
                velocities_b[cloth_node] = pm_store(cloth_velocity);
                diagnostics_b[cloth_node] = pm_store(
                    pm_load(diagnostics_b[cloth_node]) +
                    velocity_change *
                        (constants.damping /
                         inverse_masses_b[cloth_node]));
            }
            positions_a[endpoint] = pm_store(cloth_position);
            velocities_a[endpoint] = pm_store(cloth_velocity);
        }
        return;
    }
    if (constants.mode == 2u) {
        const float radius_squared =
            constants.contact_distance * constants.contact_distance;
        for (uint target = 0; target < constants.count_b; ++target) {
            if (inverse_masses_b[target] <= 0.0f) continue;
            float3 average = 0.0f;
            uint neighbors = 0u;
            for (uint source = 0; source < constants.count_a; ++source) {
                const float3 delta = pm_load(positions_a[source]) -
                                     pm_load(positions_b[target]);
                if (dot(delta, delta) <= radius_squared) {
                    average += pm_load(velocities_a[source]);
                    ++neighbors;
                }
            }
            if (neighbors == 0u) continue;
            average /= float(neighbors);
            float3 change = (average - pm_load(velocities_b[target])) *
                            (constants.stiffness * constants.timestep);
            change = pm_limit(change,
                              constants.maximum_force * constants.timestep);
            velocities_b[target] = pm_store(
                pm_load(velocities_b[target]) + change);
        }
        return;
    }
    if (constants.mode == 4u) {
        // Fluid particles collide with the nearest moving rope capsule.  The
        // particle is projected out and its momentum change is distributed to
        // the segment endpoints, matching the CUDA coupling's two-way split.
        const float distance_limit =
            max(constants.contact_distance, 1.0e-5f);
        uint contact_count = 0u;
        float maximum_penetration = 0.0f;
        for (uint particle = 0u; particle < particle_metadata.count;
             ++particle) {
            const float3 particle_position = pm_load(positions_a[particle]);
            float nearest_squared = distance_limit * distance_limit;
            uint nearest_segment = constants.count_b;
            float nearest_fraction = 0.0f;
            float3 nearest_point = 0.0f;
            for (uint segment = 0u; segment + 1u < constants.count_b;
                 ++segment) {
                const float3 first = pm_load(positions_b[segment]);
                const float3 second = pm_load(positions_b[segment + 1u]);
                const float3 axis = second - first;
                const float fraction = clamp(
                    dot(particle_position - first, axis) /
                        max(dot(axis, axis), 1.0e-12f),
                    0.0f, 1.0f);
                const float3 point = first + axis * fraction;
                const float squared = dot(particle_position - point,
                                          particle_position - point);
                if (squared >= nearest_squared) continue;
                nearest_squared = squared;
                nearest_segment = segment;
                nearest_fraction = fraction;
                nearest_point = point;
            }
            if (nearest_segment == constants.count_b) continue;
            const uint next = nearest_segment + 1u;
            const float distance_value = sqrt(max(nearest_squared, 0.0f));
            ++contact_count;
            maximum_penetration = max(
                maximum_penetration, distance_limit - distance_value);
            const float3 rope_velocity =
                pm_load(velocities_b[nearest_segment]) *
                    (1.0f - nearest_fraction) +
                pm_load(velocities_b[next]) * nearest_fraction;
            const float3 relative =
                pm_load(velocities_a[particle]) - rope_velocity;
            float3 normal;
            if (distance_value > 1.0e-6f) {
                normal = (particle_position - nearest_point) / distance_value;
            } else {
                const float relative_squared = dot(relative, relative);
                normal = relative_squared > 1.0e-12f
                             ? -relative * rsqrt(relative_squared)
                             : float3(0.0f, 1.0f, 0.0f);
            }
            positions_a[particle] = pm_store(
                particle_position +
                normal * (distance_limit - distance_value));
            const float approach = min(dot(relative, normal), 0.0f);
            const float3 tangent =
                relative - normal * dot(relative, normal);
            const float3 particle_change =
                -tangent * max(constants.friction, 0.0f) -
                normal * approach;
            velocities_a[particle] = pm_store(
                pm_load(velocities_a[particle]) + particle_change);
            diagnostics_a[particle] = pm_store(
                pm_load(diagnostics_a[particle]) +
                particle_change /
                    max(constants.timestep, 1.0e-12f));

            const float3 reaction_impulse =
                -particle_change * max(constants.stiffness, 0.0f);
            const float endpoint_weights[2] = {
                1.0f - nearest_fraction, nearest_fraction};
            const uint endpoint_indices[2] = {nearest_segment, next};
            for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
                const uint node = endpoint_indices[endpoint];
                const float node_inverse = inverse_masses_b[node];
                if (node_inverse <= 0.0f) continue;
                float3 node_change = reaction_impulse *
                                     endpoint_weights[endpoint] * node_inverse;
                node_change = pm_limit(
                    node_change,
                    constants.maximum_force * constants.timestep);
                velocities_b[node] = pm_store(
                    pm_load(velocities_b[node]) + node_change);
                diagnostics_b[node] = pm_store(
                    pm_load(diagnostics_b[node]) +
                    node_change * (constants.damping / node_inverse));
            }
        }
        diagnostic_contact_count[0] += contact_count;
        diagnostic_maximum_penetration[0] = max(
            diagnostic_maximum_penetration[0], maximum_penetration);
        return;
    }
    if (constants.mode == 5u) {
        // Smoke first transfers local wind velocity to free rope nodes, then
        // tracers are projected from the current capsule chain.  Smoke is a
        // one-way contact medium here; only the wind phase bends the rope.
        const float distance_limit =
            max(constants.contact_distance, 1.0e-5f);
        const float radius_squared = distance_limit * distance_limit;
        for (uint node = 0u; node < constants.count_b; ++node) {
            if (inverse_masses_b[node] <= 0.0f) continue;
            float3 average = 0.0f;
            uint neighbors = 0u;
            for (uint particle = 0u; particle < constants.count_a;
                 ++particle) {
                const float3 delta = pm_load(positions_a[particle]) -
                                     pm_load(positions_b[node]);
                if (dot(delta, delta) <= radius_squared) {
                    average += pm_load(velocities_a[particle]);
                    ++neighbors;
                }
            }
            if (neighbors != 0u) {
                average /= float(neighbors);
                float3 change =
                    (average - pm_load(velocities_b[node])) *
                    (constants.stiffness * constants.timestep);
                change = pm_limit(
                    change,
                    constants.maximum_force * constants.timestep);
                velocities_b[node] = pm_store(
                    pm_load(velocities_b[node]) + change);
            }
        }
        for (uint particle = 0u; particle < constants.count_a; ++particle) {
            float nearest_squared = radius_squared;
            uint nearest_segment = constants.count_b;
            float nearest_fraction = 0.0f;
            float3 nearest_point = 0.0f;
            const float3 particle_position = pm_load(positions_a[particle]);
            for (uint segment = 0u; segment + 1u < constants.count_b;
                 ++segment) {
                const float3 first = pm_load(positions_b[segment]);
                const float3 second = pm_load(positions_b[segment + 1u]);
                const float3 axis = second - first;
                const float fraction = clamp(
                    dot(particle_position - first, axis) /
                        max(dot(axis, axis), 1.0e-12f),
                    0.0f, 1.0f);
                const float3 point = first + axis * fraction;
                const float squared = dot(particle_position - point,
                                          particle_position - point);
                if (squared >= nearest_squared) continue;
                nearest_squared = squared;
                nearest_segment = segment;
                nearest_fraction = fraction;
                nearest_point = point;
            }
            if (nearest_segment == constants.count_b) continue;
            const uint next = nearest_segment + 1u;
            const float3 rope_velocity =
                pm_load(velocities_b[nearest_segment]) *
                    (1.0f - nearest_fraction) +
                pm_load(velocities_b[next]) * nearest_fraction;
            float3 relative =
                pm_load(velocities_a[particle]) - rope_velocity;
            const float separation = sqrt(max(nearest_squared, 0.0f));
            float3 normal;
            if (separation > 1.0e-6f) {
                normal = (particle_position - nearest_point) / separation;
            } else {
                const float relative_squared = dot(relative, relative);
                normal = relative_squared > 1.0e-12f
                             ? -relative * rsqrt(relative_squared)
                             : float3(0.0f, 1.0f, 0.0f);
            }
            positions_a[particle] =
                pm_store(nearest_point + normal * distance_limit);
            relative -= normal * min(dot(relative, normal), 0.0f);
            velocities_a[particle] = pm_store(rope_velocity + relative);
        }
        return;
    }
    if (constants.mode == 0u &&
        (constants.first_vertex != 0xffffffffu ||
         constants.last_vertex != 0xffffffffu)) {
        // Rope/soft-body endpoint anchors use the nearest current soft node as
        // the compact Metal baseline.  The full CUDA path binds to a skinned
        // triangle and scatters barycentric load; this still preserves the
        // bidirectional joint and acceleration bound without CPU readback.
        const uint endpoints[2] = {constants.first_vertex,
                                   constants.last_vertex};
        for (uint selected = 0u; selected < 2u; ++selected) {
            const uint endpoint = endpoints[selected];
            if (endpoint == 0xffffffffu || endpoint >= constants.count_a ||
                constants.count_b == 0u)
                continue;
            const float3 rope_position = pm_load(positions_a[endpoint]);
            float nearest_squared = INFINITY;
            uint nearest_node = 0u;
            for (uint node = 0u; node < constants.count_b; ++node) {
                const float3 delta =
                    pm_load(positions_b[node]) - rope_position;
                const float squared = dot(delta, delta);
                if (squared < nearest_squared) {
                    nearest_squared = squared;
                    nearest_node = node;
                }
            }
            const float rope_inverse = inverse_masses_a[endpoint];
            const float soft_inverse = inverse_masses_b[nearest_node];
            const float inverse_sum = rope_inverse + soft_inverse;
            if (inverse_sum <= 0.0f) continue;
            float3 soft_position = pm_load(positions_b[nearest_node]);
            const float3 separation = soft_position - rope_position;
            float3 soft_change =
                -separation * (soft_inverse / inverse_sum);
            soft_change = pm_limit(
                soft_change,
                constants.maximum_force * constants.timestep *
                    constants.timestep);
            soft_position += soft_change;
            // Keep the endpoint exactly on the soft anchor after applying the
            // bounded soft response; rope strain carries any remainder.
            float3 rope_velocity = pm_load(velocities_a[endpoint]);
            float3 soft_velocity = pm_load(velocities_b[nearest_node]);
            const float3 relative_velocity =
                rope_velocity - soft_velocity;
            float3 soft_velocity_change =
                relative_velocity * (soft_inverse / inverse_sum);
            soft_velocity_change = pm_limit(
                soft_velocity_change,
                constants.maximum_force * constants.timestep);
            soft_velocity += soft_velocity_change;
            positions_b[nearest_node] = pm_store(soft_position);
            velocities_b[nearest_node] = pm_store(soft_velocity);
            positions_a[endpoint] = pm_store(soft_position);
            velocities_a[endpoint] = pm_store(soft_velocity);
        }
    }
    const float distance_limit = max(constants.contact_distance, 1.0e-5f);
    for (uint first = 0; first < constants.count_a; ++first) {
        if (constants.mode == 0u &&
            (first == constants.first_vertex ||
             first == constants.last_vertex))
            continue;
        for (uint second = 0; second < constants.count_b; ++second) {
            float3 a = pm_load(positions_a[first]);
            float3 b = pm_load(positions_b[second]);
            const float3 delta = b - a;
            const float squared = dot(delta, delta);
            if (squared >= distance_limit * distance_limit ||
                squared <= 1.0e-12f) continue;
            const float distance_value = sqrt(squared);
            const float3 normal = delta / distance_value;
            if (constants.mode == 3u) {
                const float blend = min(constants.damping * constants.timestep,
                                        1.0f);
                velocities_a[first] = pm_store(mix(
                    pm_load(velocities_a[first]),
                    pm_load(velocities_b[second]), blend));
                continue;
            }
            const float inverse_a = inverse_masses_a[first];
            const float inverse_b = inverse_masses_b[second];
            const float inverse_sum = inverse_a + inverse_b;
            if (inverse_sum <= 0.0f) continue;
            const float penetration = distance_limit - distance_value;
            const float correction_scale = min(
                penetration * max(constants.stiffness, 1.0f) *
                    constants.timestep,
                penetration);
            const float3 correction = normal * correction_scale;
            a -= correction * (inverse_a / inverse_sum);
            b += correction * (inverse_b / inverse_sum);
            positions_a[first] = pm_store(a);
            positions_b[second] = pm_store(b);
            const float3 relative = pm_load(velocities_b[second]) -
                                    pm_load(velocities_a[first]);
            const float normal_speed = dot(relative, normal);
            float impulse = -normal_speed * max(constants.damping, 0.0f);
            if (constants.maximum_force > 0.0f)
                impulse = clamp(impulse,
                                -constants.maximum_force * constants.timestep,
                                constants.maximum_force * constants.timestep);
            velocities_a[first] = pm_store(
                pm_load(velocities_a[first]) -
                normal * impulse * (inverse_a / inverse_sum));
            velocities_b[second] = pm_store(
                pm_load(velocities_b[second]) +
                normal * impulse * (inverse_b / inverse_sum));
        }
    }
}

// Fluid particles own their contact solve, so this phase has no write races.
// Each hit emits two endpoint contributions in particle order; a second
// node-parallel pass gathers those contributions deterministically instead of
// relying on order-dependent floating-point atomics.
kernel void pm_fluid_rope_contacts(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device PMPackedVec3 *fluid_velocities [[buffer(1)]],
    device const float *fluid_inverse_masses [[buffer(2)]],
    device const PMPackedVec3 *rope_positions [[buffer(3)]],
    device const PMPackedVec3 *rope_velocities [[buffer(4)]],
    device const float *rope_inverse_masses [[buffer(5)]],
    constant PMCouplingConstants &constants [[buffer(6)]],
    device PMPackedVec3 *fluid_accelerations [[buffer(7)]],
    device PMPackedVec3 *rope_fluid_forces [[buffer(8)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(9)]],
    device atomic_uint *diagnostic_contact_count [[buffer(10)]],
    device atomic_uint *diagnostic_maximum_penetration [[buffer(11)]],
    device uint *contribution_nodes [[buffer(13)]],
    device PMPackedVec3 *contribution_changes [[buffer(14)]],
    device PMPackedVec3 *contribution_forces [[buffer(15)]],
    uint particle [[thread_position_in_grid]]) {
    (void)fluid_inverse_masses;
    (void)rope_fluid_forces;
    if (particle >= fluid_metadata.count || constants.enabled == 0u) return;
    const uint contribution = particle * 2u;
    contribution_nodes[contribution] = 0xffffffffu;
    contribution_nodes[contribution + 1u] = 0xffffffffu;
    contribution_changes[contribution] = pm_store(float3(0.0f));
    contribution_changes[contribution + 1u] = pm_store(float3(0.0f));
    contribution_forces[contribution] = pm_store(float3(0.0f));
    contribution_forces[contribution + 1u] = pm_store(float3(0.0f));
    if (constants.count_b < 2u) return;

    const float distance_limit = max(constants.contact_distance, 1.0e-5f);
    const float3 particle_position = pm_load(fluid_positions[particle]);
    float nearest_squared = distance_limit * distance_limit;
    uint nearest_segment = constants.count_b;
    float nearest_fraction = 0.0f;
    float3 nearest_point = 0.0f;
    for (uint segment = 0u; segment + 1u < constants.count_b; ++segment) {
        const float3 first = pm_load(rope_positions[segment]);
        const float3 second = pm_load(rope_positions[segment + 1u]);
        if (particle_position.x < min(first.x, second.x) - distance_limit ||
            particle_position.x > max(first.x, second.x) + distance_limit ||
            particle_position.y < min(first.y, second.y) - distance_limit ||
            particle_position.y > max(first.y, second.y) + distance_limit ||
            particle_position.z < min(first.z, second.z) - distance_limit ||
            particle_position.z > max(first.z, second.z) + distance_limit)
            continue;
        const float3 axis = second - first;
        const float fraction = clamp(
            dot(particle_position - first, axis) /
                max(dot(axis, axis), 1.0e-12f),
            0.0f, 1.0f);
        const float3 point = first + axis * fraction;
        const float squared = dot(particle_position - point,
                                  particle_position - point);
        if (squared >= nearest_squared) continue;
        nearest_squared = squared;
        nearest_segment = segment;
        nearest_fraction = fraction;
        nearest_point = point;
    }
    if (nearest_segment == constants.count_b) return;

    const uint next = nearest_segment + 1u;
    const float distance_value = sqrt(max(nearest_squared, 0.0f));
    const float penetration = distance_limit - distance_value;
    const float3 rope_velocity =
        pm_load(rope_velocities[nearest_segment]) *
            (1.0f - nearest_fraction) +
        pm_load(rope_velocities[next]) * nearest_fraction;
    const float3 relative =
        pm_load(fluid_velocities[particle]) - rope_velocity;
    float3 normal;
    if (distance_value > 1.0e-6f) {
        normal = (particle_position - nearest_point) / distance_value;
    } else {
        const float relative_squared = dot(relative, relative);
        normal = relative_squared > 1.0e-12f
                     ? -relative * rsqrt(relative_squared)
                     : float3(0.0f, 1.0f, 0.0f);
    }
    fluid_positions[particle] = pm_store(
        particle_position + normal * penetration);
    const float approach = min(dot(relative, normal), 0.0f);
    const float3 tangent = relative - normal * dot(relative, normal);
    const float3 fluid_change =
        -tangent * max(constants.friction, 0.0f) - normal * approach;
    fluid_velocities[particle] = pm_store(
        pm_load(fluid_velocities[particle]) + fluid_change);
    (void)fluid_accelerations;

    atomic_fetch_add_explicit(
        diagnostic_contact_count, 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(
        diagnostic_maximum_penetration, as_type<uint>(penetration),
        memory_order_relaxed);

    const float3 reaction_impulse =
        -fluid_change * max(constants.stiffness, 0.0f);
    const uint endpoint_nodes[2] = {nearest_segment, next};
    const float endpoint_weights[2] = {
        1.0f - nearest_fraction, nearest_fraction};
    for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
        const uint node = endpoint_nodes[endpoint];
        contribution_nodes[contribution + endpoint] = node;
        if (node == constants.first_vertex ||
            node == constants.last_vertex ||
            rope_inverse_masses[node] <= 0.0f)
            continue;
        contribution_changes[contribution + endpoint] = pm_store(
            reaction_impulse * endpoint_weights[endpoint]);
    }
}

kernel void pm_fluid_rope_apply(
    device PMPackedVec3 *fluid_positions [[buffer(0)]],
    device PMPackedVec3 *fluid_velocities [[buffer(1)]],
    device const float *fluid_inverse_masses [[buffer(2)]],
    device PMPackedVec3 *rope_positions [[buffer(3)]],
    device PMPackedVec3 *rope_velocities [[buffer(4)]],
    device const float *rope_inverse_masses [[buffer(5)]],
    constant PMCouplingConstants &constants [[buffer(6)]],
    device PMPackedVec3 *fluid_accelerations [[buffer(7)]],
    device PMPackedVec3 *rope_fluid_forces [[buffer(8)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(9)]],
    device uint *diagnostic_contact_count [[buffer(10)]],
    device float *diagnostic_maximum_penetration [[buffer(11)]],
    device const uint *contribution_nodes [[buffer(13)]],
    device const PMPackedVec3 *contribution_changes [[buffer(14)]],
    device const PMPackedVec3 *contribution_forces [[buffer(15)]],
    uint node [[thread_position_in_grid]]) {
    (void)fluid_positions;
    (void)fluid_velocities;
    (void)fluid_inverse_masses;
    (void)rope_positions;
    (void)rope_inverse_masses;
    (void)fluid_accelerations;
    (void)diagnostic_contact_count;
    (void)diagnostic_maximum_penetration;
    if (node >= constants.count_b || constants.enabled == 0u) return;
    if (node == constants.first_vertex || node == constants.last_vertex)
        return;
    float3 reaction_impulse = 0.0f;
    for (uint particle = 0u; particle < fluid_metadata.count; ++particle) {
        const uint contribution = particle * 2u;
        for (uint endpoint = 0u; endpoint < 2u; ++endpoint) {
            const uint item = contribution + endpoint;
            if (contribution_nodes[item] != node) continue;
            reaction_impulse += pm_load(contribution_changes[item]);
        }
    }
    const float inverse = rope_inverse_masses[node];
    if (inverse <= 0.0f) return;
    const float3 velocity_change = pm_limit(
        reaction_impulse * inverse,
        constants.maximum_force * constants.timestep);
    rope_velocities[node] = pm_store(
        pm_load(rope_velocities[node]) + velocity_change);
    rope_fluid_forces[node] = pm_store(
        pm_load(rope_fluid_forces[node]) +
        velocity_change * (constants.damping / inverse));
}

static bool pm_solid_interior_contact(
    float3 position, float contact_distance,
    device const PMTriangleMeshInfo &mesh,
    device const PMCollisionPlane *solid_planes,
    thread float3 &normal, thread float &penetration,
    thread float3 &contact) {
    if (mesh.solid_plane_count == 0u) return false;
    float nearest_side = -INFINITY;
    float3 nearest_normal = 0.0f;
    for (uint local = 0u; local < mesh.solid_plane_count; ++local) {
        device const PMCollisionPlane &plane =
            solid_planes[mesh.solid_plane_offset + local];
        const float side = dot(pm_load(plane.normal), position) -
                           plane.offset;
        if (side > 0.0f) return false;
        if (side > nearest_side) {
            nearest_side = side;
            nearest_normal = pm_load(plane.normal);
        }
    }
    normal = nearest_normal;
    penetration = contact_distance - nearest_side;
    contact = position - nearest_normal * nearest_side;
    return true;
}

kernel void pm_fluid_soft_rigid_recover(
    device PMPackedVec3 *node_positions [[buffer(6)]],
    device const float *node_inverse_masses [[buffer(8)]],
    constant PMFluidSoftConstants &constants [[buffer(13)]],
    device const PMPackedVec3 *node_previous [[buffer(24)]],
    device const PMRigidBodyState *rigid_states [[buffer(25)]],
    device const PMRigidParameters *rigid_parameters [[buffer(26)]],
    device const PMPackedVec3 *vertices [[buffer(27)]],
    device const uint *indices [[buffer(28)]],
    device const PMTriangleMeshInfo *meshes [[buffer(29)]],
    device const PMCollisionPlane *solid_planes [[buffer(30)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= constants.node_count || constants.enabled == 0u ||
        node_inverse_masses[node] <= 0.0f)
        return;
    const float3 start = pm_load(node_previous[node]);
    const float3 end = pm_load(node_positions[node]);
    float best_penetration = 0.0f;
    float3 best_normal = 0.0f;
    for (uint body = 0u; body < constants.rigid_count; ++body) {
        device const PMRigidBodyState &state = rigid_states[body];
        device const PMRigidParameters &parameters =
            rigid_parameters[body];
        device const PMTriangleMeshInfo &mesh =
            meshes[parameters.mesh_index];
        const PMQuaternion inverse =
            pm_quaternion_conjugate(state.orientation);
        const float3 local_start = pm_rotate(
            inverse, start - pm_load(state.position));
        const float3 local_end = pm_rotate(
            inverse, end - pm_load(state.position));
        const float contact_distance =
            constants.rigid_recovery_radius + parameters.collision_margin;
        const float3 lower = min(local_start, local_end) - contact_distance;
        const float3 upper = max(local_start, local_end) + contact_distance;
        if (any(upper < pm_load(mesh.minimum)) ||
            any(lower > pm_load(mesh.maximum)))
            continue;
        if (mesh.solid_plane_count != 0u) {
            float3 normal = 0.0f;
            float penetration = 0.0f;
            float3 contact = 0.0f;
            if (pm_solid_interior_contact(
                    local_end, contact_distance, mesh, solid_planes,
                    normal, penetration, contact)) {
                if (penetration > best_penetration) {
                    best_penetration = penetration;
                    best_normal =
                        pm_rotate(state.orientation, normal);
                }
                continue;
            }
        }
        for (uint local = 0u; local < mesh.index_count; local += 3u) {
            const uint ia = indices[mesh.index_offset + local];
            const uint ib = indices[mesh.index_offset + local + 1u];
            const uint ic = indices[mesh.index_offset + local + 2u];
            const float3 a = pm_load(vertices[mesh.vertex_offset + ia]);
            const float3 b = pm_load(vertices[mesh.vertex_offset + ib]);
            const float3 c = pm_load(vertices[mesh.vertex_offset + ic]);
            if (any(upper < min(a, min(b, c))) ||
                any(lower > max(a, max(b, c))))
                continue;
            const bool solid = mesh.solid_plane_count != 0u;
            const float3 face_value = cross(b - a, c - a);
            const float face_squared = dot(face_value, face_value);
            if (!solid && face_squared <= 1.0e-16f) continue;
            const float3 face = solid
                ? pm_load(solid_planes[
                      mesh.solid_plane_offset + local / 3u].normal)
                : face_value * rsqrt(face_squared);
            const float3 nearest =
                pm_closest_point_triangle(local_end, a, b, c);
            const float3 delta = local_end - nearest;
            const float distance = length(delta);
            float3 normal = distance > 1.0e-6f
                ? delta / distance
                : face * (dot(local_start - a, face) >= 0.0f
                              ? 1.0f
                              : -1.0f);
            float penetration = contact_distance - distance;
            const float before = dot(local_start - a, face);
            const float after = dot(local_end - a, face);
            if (before * after < 0.0f && (!solid || before > 0.0f)) {
                const float fraction = before / (before - after);
                const float3 crossing =
                    local_start + (local_end - local_start) * fraction;
                const float3 hit =
                    pm_closest_point_triangle(crossing, a, b, c);
                if (dot(hit - crossing, hit - crossing) <
                    constants.rigid_recovery_radius *
                        constants.rigid_recovery_radius) {
                    normal = face * (before > 0.0f ? 1.0f : -1.0f);
                    penetration = max(
                        penetration, contact_distance + abs(after));
                }
            }
            if (penetration <= best_penetration) continue;
            best_penetration = penetration;
            best_normal = pm_rotate(state.orientation, normal);
        }
    }
    if (best_penetration > 0.0f)
        node_positions[node] =
            pm_store(end + best_normal * best_penetration);
}

kernel void pm_soft_contact_accumulators_clear(
    device PMSoftContactAccumulator *contact_accumulators [[buffer(27)]],
    device PMSoftContactState *contact_state [[buffer(28)]],
    uint node [[thread_position_in_grid]]) {
    contact_accumulators[node] = {
        pm_store(float3(0.0f)), pm_store(float3(0.0f)),
        pm_store(float3(0.0f)), 0.0f};
    if (node == 0u)
        atomic_store_explicit(
            &contact_state->dynamic_contact_flag, 0u,
            memory_order_relaxed);
}

kernel void pm_soft_body_measure_momentum(
    device const PMPackedVec3 *velocities [[buffer(1)]],
    device const float *inverse_masses [[buffer(2)]],
    constant PMParticleRigidConstants &constants [[buffer(8)]],
    device PMSoftContactState *contact_state [[buffer(28)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    threadgroup float3 values[128];
    const uint lane_count = threads_per_group.x;
    float3 local = 0.0f;
    for (uint node = lane; node < constants.particle_count;
         node += lane_count) {
        const float inverse = inverse_masses[node];
        if (inverse > 0.0f)
            local += pm_load(velocities[node]) / inverse;
    }
    values[lane] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2u; stride != 0u; stride /= 2u) {
        if (lane < stride) values[lane] += values[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u)
        contact_state->predicted_momentum = pm_store(values[0]);
}

kernel void pm_soft_body_restore_momentum(
    device PMPackedVec3 *velocities [[buffer(1)]],
    device const float *inverse_masses [[buffer(2)]],
    constant PMParticleRigidConstants &constants [[buffer(8)]],
    device const PMSoftContactAccumulator *contact_accumulators
        [[buffer(27)]],
    device const PMSoftContactState *contact_state [[buffer(28)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    if (atomic_load_explicit(
            &contact_state->dynamic_contact_flag,
            memory_order_relaxed) == 0u)
        return;
    threadgroup float3 actual_values[128];
    threadgroup float3 contact_values[128];
    threadgroup float3 correction;
    const uint lane_count = threads_per_group.x;
    float3 actual = 0.0f;
    float3 contact = 0.0f;
    for (uint node = lane; node < constants.particle_count;
         node += lane_count) {
        const float inverse = inverse_masses[node];
        if (inverse <= 0.0f) continue;
        actual += pm_load(velocities[node]) / inverse;
        contact += pm_load(contact_accumulators[node].momentum_delta);
    }
    actual_values[lane] = actual;
    contact_values[lane] = contact;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count / 2u; stride != 0u; stride /= 2u) {
        if (lane < stride) {
            actual_values[lane] += actual_values[lane + stride];
            contact_values[lane] += contact_values[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        const float3 target =
            pm_load(contact_state->predicted_momentum) + contact_values[0];
        correction =
            (target - actual_values[0]) /
            max(constants.movable_mass, 1.0e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint node = lane; node < constants.particle_count;
         node += lane_count)
        if (inverse_masses[node] > 0.0f)
            velocities[node] = pm_store(pm_limit(
                pm_load(velocities[node]) + correction,
                constants.maximum_reaction_speed));
}

kernel void pm_particles_rigid(
    device PMPackedVec3 *particle_positions [[buffer(0)]],
    device PMPackedVec3 *particle_velocities [[buffer(1)]],
    device const float *particle_inverse_masses [[buffer(2)]],
    device PMRigidBodyState *rigid_states [[buffer(3)]],
    device PMRigidParameters *rigid_parameters [[buffer(4)]],
    device const PMPackedVec3 *vertices [[buffer(5)]],
    device const uint *indices [[buffer(6)]],
    device const PMTriangleMeshInfo *meshes [[buffer(7)]],
    constant PMParticleRigidConstants &constants [[buffer(8)]],
    device PMPackedVec3 *diagnostics [[buffer(9)]],
    device const uint *stable_ids [[buffer(10)]],
    device const PMHandle *rigid_ids [[buffer(11)]],
    device PMContactEvent *contact_samples [[buffer(12)]],
    device uint *contact_flags [[buffer(13)]],
    device const PMParticleMetadata &particle_metadata [[buffer(14)]],
    device const PMPackedVec3 *particle_previous [[buffer(15)]],
    device const PMRigidBodyState *old_rigid_states [[buffer(16)]],
    device float *foam [[buffer(17)]],
    device PMParticleRigidContact *contacts [[buffer(18)]],
    device atomic_uint *body_contact_counts [[buffer(19)]],
    device PMPackedVec3 *body_linear_impulses [[buffer(20)]],
    device PMPackedVec3 *body_angular_impulses [[buffer(21)]],
    device const PMCollisionPlane *solid_planes [[buffer(22)]],
    device PMPackedVec3 *body_position_corrections [[buffer(23)]],
    device PMPackedVec3 *particle_linear_impulses [[buffer(24)]],
    device PMPackedVec3 *particle_angular_impulses [[buffer(25)]],
    device PMPackedVec3 *particle_position_corrections [[buffer(26)]],
    device PMSoftContactAccumulator *contact_accumulators [[buffer(27)]],
    device PMSoftContactState *contact_state [[buffer(28)]],
    device const uint &spawn_baseline [[buffer(29)]],
    uint thread_index [[thread_position_in_threadgroup]],
    uint thread_count [[threads_per_threadgroup]]) {
    threadgroup float3 reaction_linear[128];
    threadgroup float3 reaction_angular[128];
    threadgroup float3 reaction_position[128];
    const bool fluid_particles = constants.fluid.generation != 0u;
    const bool cloth_particles = !fluid_particles &&
                                 constants.solid_contacts == 0u;
    const bool soft_particles = !fluid_particles &&
                                constants.solid_contacts != 0u;
    const uint particle_count = fluid_particles
                                    ? particle_metadata.count
                                    : constants.particle_count;
    for (uint body = thread_index; body < constants.rigid_count;
         body += thread_count) {
        atomic_store_explicit(
            body_contact_counts + body, 0u, memory_order_relaxed);
        body_linear_impulses[body] = {0.0f, 0.0f, 0.0f};
        body_angular_impulses[body] = {0.0f, 0.0f, 0.0f};
        body_position_corrections[body] = {0.0f, 0.0f, 0.0f};
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Detect from immutable pre-solve state first. This mirrors CUDA's
    // Jacobi moving-contact phase: all particles see the same rigid velocity,
    // and body reactions are reduced only after every contact is resolved.
    for (uint particle = thread_index; particle < particle_count;
         particle += thread_count) {
        const float particle_inverse = fluid_particles
                                           ? constants.particle_inverse_mass
                                           : particle_inverse_masses[particle];
        contacts[particle] = {
            {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0.0f, 0xffffffffu};
        if (particle_inverse <= 0.0f)
            continue;
        const float3 particle_position =
            pm_load(particle_positions[particle]);
        const float3 previous_position =
            pm_load(particle_previous[particle]);
        PMParticleRigidContact best = contacts[particle];
        for (uint body = 0; body < constants.rigid_count; ++body) {
            device const PMRigidBodyState &body_state = rigid_states[body];
            device const PMRigidParameters &body_parameters =
                rigid_parameters[body];
            if (body_parameters.motion == 0u && fluid_particles) continue;
            device const PMTriangleMeshInfo &mesh =
                meshes[body_parameters.mesh_index];
            const PMRigidBodyState old_state =
                constants.first_iteration != 0u
                    ? old_rigid_states[body]
                    : body_state;
            const PMQuaternion inverse =
                pm_quaternion_conjugate(body_state.orientation);
            const PMQuaternion old_inverse =
                pm_quaternion_conjugate(old_state.orientation);
            const float3 position = pm_rotate(
                inverse, particle_position - pm_load(body_state.position));
            const float3 origin = pm_rotate(
                old_inverse,
                previous_position - pm_load(old_state.position));
            const float contact_distance =
                constants.radius +
                (fluid_particles ? 0.0f
                                 : body_parameters.collision_margin);
            const float3 lower = min(origin, position) - contact_distance;
            const float3 upper = max(origin, position) + contact_distance;
            if (any(upper < pm_load(mesh.minimum)) ||
                any(lower > pm_load(mesh.maximum)))
                continue;
            if (constants.solid_contacts != 0u) {
                float3 normal = 0.0f;
                float penetration = 0.0f;
                float3 contact = 0.0f;
                if (pm_solid_interior_contact(
                        position, contact_distance, mesh, solid_planes,
                        normal, penetration, contact)) {
                    if (best.body == 0xffffffffu ||
                        penetration > best.penetration) {
                        best.normal = pm_store(pm_rotate(
                            body_state.orientation, normal));
                        best.point = pm_store(
                            pm_world_point(body_state, contact));
                        best.penetration = penetration;
                        best.body = body;
                    }
                    continue;
                }
            }
            for (uint local = 0; local < mesh.index_count; local += 3u) {
                const uint first = indices[mesh.index_offset + local];
                const uint second = indices[mesh.index_offset + local + 1u];
                const uint third = indices[mesh.index_offset + local + 2u];
                const float3 a =
                    pm_load(vertices[mesh.vertex_offset + first]);
                const float3 b =
                    pm_load(vertices[mesh.vertex_offset + second]);
                const float3 c =
                    pm_load(vertices[mesh.vertex_offset + third]);
                if (any(upper < min(a, min(b, c))) ||
                    any(lower > max(a, max(b, c))))
                    continue;
                const float3 face_value = cross(b - a, c - a);
                if (dot(face_value, face_value) <= 1.0e-14f) continue;
                const float3 face = pm_normalized_or(
                    face_value, float3(0.0f, 1.0f, 0.0f));
                const float3 candidate =
                    pm_closest_point_triangle(position, a, b, c);
                const float3 delta = position - candidate;
                const float distance_value = length(delta);
                float3 normal = distance_value >
                                        (fluid_particles || cloth_particles
                                             ? 1.0e-6f
                                             : 1.0e-7f)
                                    ? delta / distance_value
                                    : face * (dot(origin - a, face) >= 0.0f
                                                  ? 1.0f
                                                  : -1.0f);
                float penetration = contact_distance - distance_value;
                float3 contact = candidate;
                const float before = dot(origin - a, face);
                const float after = dot(position - a, face);
                if (before * after < 0.0f &&
                    (!soft_particles || mesh.solid_plane_count == 0u ||
                     before > 0.0f)) {
                    const float fraction = before / (before - after);
                    const float3 crossing =
                        origin + (position - origin) * fraction;
                    const float3 nearest =
                        pm_closest_point_triangle(crossing, a, b, c);
                    const float crossing_radius = cloth_particles
                                                      ? constants.radius
                                                      : contact_distance;
                    const float crossing_distance =
                        dot(crossing - nearest, crossing - nearest);
                    if (cloth_particles
                            ? crossing_distance <
                                  crossing_radius * crossing_radius
                            : crossing_distance <=
                                  crossing_radius * crossing_radius) {
                        normal = face * (cloth_particles
                                             ? (before > 0.0f ? 1.0f
                                                              : -1.0f)
                                             : (before >= 0.0f ? 1.0f
                                                               : -1.0f));
                        penetration = max(
                            penetration, contact_distance + abs(after));
                    }
                }
                if (penetration <= 0.0f ||
                    (best.body != 0xffffffffu &&
                     penetration <= best.penetration))
                    continue;
                best.normal = pm_store(
                    pm_rotate(body_state.orientation, normal));
                best.point = pm_store(pm_world_point(body_state, contact));
                best.penetration = penetration;
                best.body = body;
            }
        }
        contacts[particle] = best;
        if (best.body != 0xffffffffu)
            atomic_fetch_add_explicit(
                body_contact_counts + best.body, 1u,
                memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_device);

    for (uint particle = thread_index; particle < particle_count;
         particle += thread_count) {
        particle_linear_impulses[particle] = pm_store(float3(0.0f));
        particle_angular_impulses[particle] = pm_store(float3(0.0f));
        particle_position_corrections[particle] = pm_store(float3(0.0f));
        if (soft_particles)
            contact_accumulators[particle].arm = pm_store(float3(0.0f));
        const PMParticleRigidContact contact = contacts[particle];
        if (contact.body == 0xffffffffu) continue;
        const uint body = contact.body;
        device const PMRigidBodyState &body_state = rigid_states[body];
        device const PMRigidParameters &body_parameters =
            rigid_parameters[body];
        const float particle_inverse = fluid_particles
                                           ? constants.particle_inverse_mass
                                           : particle_inverse_masses[particle];
        const float particle_mass = 1.0f / max(particle_inverse, 1.0e-12f);
        const float3 normal = pm_load(contact.normal);
        const float3 point = pm_load(contact.point);
        const float position_denominator =
            particle_inverse + body_parameters.inverse_mass;
        const float particle_position_share =
            constants.share_position != 0u && position_denominator > 1.0e-12f
                ? particle_inverse / position_denominator
                : 1.0f;
        particle_positions[particle] = pm_store(
            pm_load(particle_positions[particle]) +
            normal * (contact.penetration * particle_position_share));
        if (constants.share_position != 0u &&
            body_parameters.motion == 2u && position_denominator > 1.0e-12f)
            particle_position_corrections[particle] = pm_store(
                -normal * (contact.penetration *
                           body_parameters.inverse_mass /
                           position_denominator));
        float3 particle_velocity = soft_particles
            ? (pm_load(particle_positions[particle]) -
               normal * (contact.penetration * particle_position_share) -
               pm_load(particle_previous[particle])) /
                  max(constants.timestep, 1.0e-12f)
            : pm_load(particle_velocities[particle]);
        const float3 initial_velocity = particle_velocity;
        const float3 arm = point - pm_load(body_state.position);
        if (soft_particles) {
            contact_accumulators[particle].normal_delta +=
                contact.penetration;
            contact_accumulators[particle].arm = pm_store(arm);
            if (body_parameters.motion == 2u)
                atomic_store_explicit(
                    &contact_state->dynamic_contact_flag, 1u,
                    memory_order_relaxed);
        }
        const float3 body_velocity =
            pm_load(body_state.linear_velocity) +
            cross(pm_load(body_state.angular_velocity), arm);
        const float3 relative = particle_velocity - body_velocity;
        const float incoming = dot(relative, normal);
        float normal_impulse = 0.0f;
        float3 impulse = 0.0f;
        if (cloth_particles && incoming < 0.0f) {
            const float3 normal_cross = cross(arm, normal);
            const float denominator = particle_inverse +
                body_parameters.inverse_mass +
                dot(cross(pm_inverse_inertia_mul(
                              body_parameters, body_state, normal_cross),
                          arm),
                    normal);
            if (denominator > 1.0e-12f) {
                normal_impulse =
                    -(1.0f + body_parameters.restitution) * incoming /
                    denominator;
                impulse = normal * normal_impulse;
                particle_velocity += impulse * particle_inverse;
                particle_linear_impulses[particle] = pm_store(-impulse);
                particle_angular_impulses[particle] = pm_store(
                    -cross(arm, impulse));
            }
        } else if (soft_particles && incoming < 0.0f) {
            const float3 normal_cross = cross(arm, normal);
            const float denominator = particle_inverse +
                body_parameters.inverse_mass +
                dot(cross(pm_inverse_inertia_mul(
                              body_parameters, body_state, normal_cross),
                          arm),
                    normal);
            if (denominator > 1.0e-12f) {
                normal_impulse =
                    -(1.0f + body_parameters.restitution) * incoming /
                    denominator;
                impulse = normal * normal_impulse;
                const float3 velocity_delta =
                    impulse * particle_inverse;
                particle_velocity += velocity_delta;
                contact_accumulators[particle].momentum_delta = pm_store(
                    pm_load(contact_accumulators[particle].momentum_delta) +
                    impulse);
                if (body_parameters.motion == 2u)
                    particle_positions[particle] = pm_store(
                        pm_load(particle_positions[particle]) +
                        velocity_delta * constants.timestep);
                particle_linear_impulses[particle] = pm_store(-impulse);
                particle_angular_impulses[particle] = pm_store(
                    -cross(arm, impulse));
            }
        } else if (fluid_particles) {
            const float recovery_speed = min(
                constants.maximum_reaction_speed,
                min(0.5f * constants.radius, 0.2f * contact.penetration) /
                    max(constants.timestep, 1.0e-12f));
            if (incoming < recovery_speed) {
            const float batch_size = 1.0f +
                float(max(atomic_load_explicit(
                              body_contact_counts + body,
                              memory_order_relaxed),
                          1u) -
                      1u) *
                    min(1.0f,
                        4.0f * particle_mass * body_parameters.inverse_mass);
            const float3 normal_cross = cross(arm, normal);
            const float body_normal_inverse =
                body_parameters.inverse_mass +
                dot(cross(pm_inverse_inertia_mul(
                              body_parameters, body_state, normal_cross),
                          arm),
                    normal);
            const float denominator =
                particle_inverse + batch_size * body_normal_inverse;
            if (denominator > 1.0e-12f) {
                normal_impulse =
                    (recovery_speed - incoming) / denominator;
                impulse = normal * normal_impulse;
                particle_velocity += impulse * particle_inverse;
                const float3 tangent_velocity =
                    relative - normal * incoming;
                const float tangent_speed = length(tangent_velocity);
                if (tangent_speed > 1.0e-8f) {
                    const float3 tangent = tangent_velocity / tangent_speed;
                    const float3 tangent_cross = cross(arm, tangent);
                    const float tangent_denominator =
                        particle_inverse + batch_size *
                            (body_parameters.inverse_mass +
                             dot(cross(pm_inverse_inertia_mul(
                                           body_parameters, body_state,
                                           tangent_cross),
                                       arm),
                                 tangent));
                    if (tangent_denominator > 1.0e-12f) {
                        const float friction = fluid_particles
                                                   ? body_parameters.friction
                                                   : constants.friction;
                        const float tangent_impulse = min(
                            tangent_speed / tangent_denominator,
                            max(friction, 0.0f) * normal_impulse);
                        const float3 friction_impulse =
                            -tangent * tangent_impulse;
                        impulse += friction_impulse;
                        particle_velocity +=
                            friction_impulse * particle_inverse;
                    }
                }
                particle_linear_impulses[particle] = pm_store(-impulse);
                particle_angular_impulses[particle] = pm_store(
                    -cross(arm, impulse));
                if (fluid_particles)
                    foam[particle] = max(
                        foam[particle], min(1.0f, -incoming * 0.35f));
            }
            }
        }
        particle_velocity = pm_limit(
            particle_velocity, constants.maximum_reaction_speed);
        particle_velocities[particle] = pm_store(particle_velocity);
        const float inverse_dt =
            1.0f / max(constants.timestep, 1.0e-12f);
        const float3 correction_velocity =
            normal * (contact.penetration * particle_position_share *
                      inverse_dt);
        const float3 acceleration =
            (particle_velocity - initial_velocity + correction_velocity) *
            inverse_dt;
        diagnostics[particle] = cloth_particles || soft_particles
            ? pm_store(impulse * inverse_dt)
            : pm_store(
                  pm_load(diagnostics[particle]) +
                  (constants.diagnostic_is_acceleration != 0u
                       ? acceleration
                       : acceleration / particle_inverse));
        if (constants.collect_contacts != 0u) {
            if (contact_flags[particle] != 2u ||
                normal_impulse > contact_samples[particle].normal_impulse) {
                contact_samples[particle] = {
                    constants.fluid, stable_ids[particle], rigid_ids[body],
                    contact.point, contact.normal, normal_impulse};
            }
            contact_flags[particle] = 2u;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    // Match CUDA's reduce_point_body_impulses exactly: strided lane-local
    // accumulation followed by the same fixed-width binary tree.  A serial
    // sum changes impact trajectories enough to alter deterministic fracture
    // and coupling decisions.
    for (uint body = 0u; body < constants.rigid_count; ++body) {
        float3 linear_impulse = 0.0f;
        float3 angular_impulse = 0.0f;
        float3 position_correction = 0.0f;
        for (uint particle = thread_index; particle < particle_count;
             particle += thread_count) {
            if (contacts[particle].body != body) continue;
            linear_impulse += pm_load(particle_linear_impulses[particle]);
            angular_impulse += pm_load(particle_angular_impulses[particle]);
            position_correction +=
                pm_load(particle_position_corrections[particle]);
        }
        reaction_linear[thread_index] = linear_impulse;
        reaction_angular[thread_index] = angular_impulse;
        reaction_position[thread_index] = position_correction;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = thread_count / 2u; stride != 0u; stride /= 2u) {
            if (thread_index < stride) {
                reaction_linear[thread_index] +=
                    reaction_linear[thread_index + stride];
                reaction_angular[thread_index] +=
                    reaction_angular[thread_index + stride];
                reaction_position[thread_index] +=
                    reaction_position[thread_index + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index == 0u) {
            body_linear_impulses[body] = pm_store(reaction_linear[0]);
            body_angular_impulses[body] = pm_store(reaction_angular[0]);
            body_position_corrections[body] = pm_store(reaction_position[0]);
            device PMRigidParameters &body_parameters = rigid_parameters[body];
            if (body_parameters.motion == 2u) {
                device PMRigidBodyState &body_state = rigid_states[body];
                body_state.position = pm_store(
                    pm_load(body_state.position) + reaction_position[0]);
                body_state.linear_velocity = pm_store(pm_limit(
                    pm_load(body_state.linear_velocity) +
                        reaction_linear[0] * body_parameters.inverse_mass,
                    body_parameters.maximum_linear_speed));
                body_state.angular_velocity = pm_store(pm_limit(
                    pm_load(body_state.angular_velocity) +
                        pm_inverse_inertia_mul(
                            body_parameters, body_state,
                            reaction_angular[0]),
                    body_parameters.maximum_angular_speed));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (cloth_particles || soft_particles) return;

    // CUDA resolves its one swept moving-body contact first, then visits
    // every static body in stable dense order. Keep that second phase
    // separate: a moving wall must not suppress a floor (or another static
    // boundary) hit produced by the corrected particle position.
    for (uint particle = thread_index; particle < particle_count;
         particle += thread_count) {
        const float particle_inverse = fluid_particles
                                           ? constants.particle_inverse_mass
                                           : particle_inverse_masses[particle];
        if (particle_inverse <= 0.0f) continue;
        for (uint body = 0u; body < constants.rigid_count; ++body) {
            device const PMRigidBodyState &body_state = rigid_states[body];
            device const PMRigidParameters &body_parameters =
                rigid_parameters[body];
            if (body_parameters.motion != 0u) continue;
            device const PMTriangleMeshInfo &mesh =
                meshes[body_parameters.mesh_index];
            const PMQuaternion inverse =
                pm_quaternion_conjugate(body_state.orientation);
            const float contact_distance =
                constants.radius +
                (fluid_particles ? 0.0f
                                 : body_parameters.collision_margin);
            const bool newly_spawned = fluid_particles &&
                constants.recover_spawn != 0u &&
                particle >= spawn_baseline;
            const float query_distance = newly_spawned
                ? max(contact_distance, constants.spawn_clearance)
                : contact_distance;
            const float3 world_position =
                pm_load(particle_positions[particle]);
            const float3 position = pm_rotate(
                inverse, world_position - pm_load(body_state.position));
            const float3 origin = pm_rotate(
                inverse,
                pm_load(particle_previous[particle]) -
                    pm_load(body_state.position));
            const float3 lower = min(origin, position) - query_distance;
            const float3 upper = max(origin, position) + query_distance;
            if (any(upper < pm_load(mesh.minimum)) ||
                any(lower > pm_load(mesh.maximum)))
                continue;
            float best_penetration = 0.0f;
            float3 best_normal = 0.0f;
            float3 best_contact = 0.0f;
            const bool interior = constants.solid_contacts != 0u &&
                pm_solid_interior_contact(
                    position, contact_distance, mesh, solid_planes,
                    best_normal, best_penetration, best_contact);
            for (uint local = 0u;
                 !interior && local < mesh.index_count; local += 3u) {
                const uint first = indices[mesh.index_offset + local];
                const uint second = indices[mesh.index_offset + local + 1u];
                const uint third = indices[mesh.index_offset + local + 2u];
                const float3 a =
                    pm_load(vertices[mesh.vertex_offset + first]);
                const float3 b =
                    pm_load(vertices[mesh.vertex_offset + second]);
                const float3 c =
                    pm_load(vertices[mesh.vertex_offset + third]);
                if (any(upper < min(a, min(b, c))) ||
                    any(lower > max(a, max(b, c))))
                    continue;
                const float3 face_value = cross(b - a, c - a);
                if (dot(face_value, face_value) <= 1.0e-14f) continue;
                const float3 face = normalize(face_value);
                const float3 candidate =
                    pm_closest_point_triangle(position, a, b, c);
                const float3 delta = position - candidate;
                const float distance_value = length(delta);
                float3 normal = distance_value >
                                        (fluid_particles ? 1.0e-6f
                                                         : 1.0e-7f)
                                    ? delta / distance_value
                                    : face * (dot(origin - a, face) >= 0.0f
                                                  ? 1.0f
                                                  : -1.0f);
                float penetration = contact_distance - distance_value;
                float3 contact = candidate;
                const float before = dot(origin - a, face);
                const float after = dot(position - a, face);
                if (before * after < 0.0f &&
                    (!soft_particles || mesh.solid_plane_count == 0u ||
                     before > 0.0f)) {
                    const float fraction = before / (before - after);
                    const float3 crossing =
                        origin + (position - origin) * fraction;
                    const float3 nearest =
                        pm_closest_point_triangle(crossing, a, b, c);
                    if (dot(crossing - nearest, crossing - nearest) <=
                        contact_distance * contact_distance) {
                        normal = face * (before >= 0.0f ? 1.0f : -1.0f);
                        penetration = max(
                            penetration, contact_distance + abs(after));
                        contact = nearest;
                    }
                }
                if (newly_spawned) {
                    const float3 local_up = pm_rotate(inverse,
                                                      pm_load(constants.up));
                    if (abs(dot(face, local_up)) > 0.7f) {
                        const float3 floor_normal = dot(face, local_up) > 0.0f
                            ? face : -face;
                        const float side =
                            dot(position - a, floor_normal);
                        if (side < constants.radius &&
                            side > -constants.spawn_clearance &&
                            distance_value < constants.spawn_clearance) {
                            normal = floor_normal;
                            penetration = max(
                                penetration, constants.radius - side);
                        }
                    }
                }
                if (penetration <= best_penetration) continue;
                best_penetration = penetration;
                best_normal = normal;
                best_contact = contact;
            }
            if (best_penetration <= 0.0f) continue;
            const float3 world_normal =
                pm_rotate(body_state.orientation, best_normal);
            const float3 world_contact =
                pm_world_point(body_state, best_contact);
            contacts[particle] = {
                pm_store(world_normal), pm_store(world_contact),
                best_penetration, body};
            atomic_fetch_add_explicit(
                body_contact_counts + body, 1u, memory_order_relaxed);
            particle_positions[particle] = pm_store(
                world_position + world_normal * best_penetration);
            float3 particle_velocity =
                pm_load(particle_velocities[particle]);
            const float3 initial_velocity = particle_velocity;
            const float incoming = dot(particle_velocity, world_normal);
            float normal_impulse = 0.0f;
            if (incoming < 0.0f) {
                const float restitution = fluid_particles
                                              ? body_parameters.restitution
                                              : constants.restitution;
                normal_impulse =
                    -incoming * (1.0f + restitution) / particle_inverse;
                particle_velocity -=
                    world_normal * incoming * (1.0f + restitution);
                const float3 tangent_velocity =
                    particle_velocity -
                    world_normal * dot(particle_velocity, world_normal);
                const float tangent_speed = length(tangent_velocity);
                if (tangent_speed > 1.0e-8f) {
                    const float friction = fluid_particles
                                               ? body_parameters.friction
                                               : constants.friction;
                    const float friction_speed = min(
                        tangent_speed,
                        max(friction, 0.0f) * normal_impulse *
                            particle_inverse);
                    particle_velocity -= tangent_velocity *
                        (friction_speed / tangent_speed);
                }
                if (fluid_particles)
                    foam[particle] = max(
                        foam[particle], min(1.0f, -incoming * 0.35f));
            }
            particle_velocity = pm_limit(
                particle_velocity, constants.maximum_reaction_speed);
            particle_velocities[particle] = pm_store(particle_velocity);
            const float inverse_dt =
                1.0f / max(constants.timestep, 1.0e-12f);
            const float3 correction_velocity =
                world_normal * (best_penetration * inverse_dt);
            const float3 acceleration =
                (particle_velocity - initial_velocity +
                 correction_velocity) *
                inverse_dt;
            diagnostics[particle] = pm_store(
                pm_load(diagnostics[particle]) +
                (constants.diagnostic_is_acceleration != 0u
                     ? acceleration
                     : acceleration / particle_inverse));
            if (constants.collect_contacts != 0u &&
                (contact_flags[particle] == 0u ||
                 (contact_flags[particle] == 1u &&
                  normal_impulse >
                      contact_samples[particle].normal_impulse))) {
                contact_samples[particle] = {
                    constants.fluid, stable_ids[particle], rigid_ids[body],
                    pm_store(world_contact), pm_store(world_normal),
                    normal_impulse};
                contact_flags[particle] = 1u;
            }
        }
    }
}

kernel void pm_soft_contact_friction(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *velocities [[buffer(1)]],
    device const float *inverse_masses [[buffer(2)]],
    device const PMRigidParameters *rigid_parameters [[buffer(4)]],
    constant PMParticleRigidConstants &constants [[buffer(8)]],
    device PMPackedVec3 *diagnostics [[buffer(9)]],
    device const PMParticleRigidContact *contacts [[buffer(18)]],
    device const atomic_uint *body_contact_counts [[buffer(19)]],
    device PMPackedVec3 *particle_linear_impulses [[buffer(24)]],
    device PMPackedVec3 *particle_angular_impulses [[buffer(25)]],
    device PMSoftContactAccumulator *contact_accumulators [[buffer(27)]],
    uint particle [[thread_position_in_grid]]) {
    if (particle >= constants.particle_count) return;
    particle_linear_impulses[particle] = pm_store(float3(0.0f));
    particle_angular_impulses[particle] = pm_store(float3(0.0f));
    if (
        constants.fluid.generation != 0u ||
        constants.solid_contacts == 0u ||
        inverse_masses[particle] <= 0.0f ||
        constants.friction <= 0.0f)
        return;
    const PMParticleRigidContact contact = contacts[particle];
    if (contact.body == 0xffffffffu ||
        contact.body >= constants.rigid_count)
        return;
    uint supported = 0u;
    for (uint body = 0u; body < constants.rigid_count; ++body)
        supported += atomic_load_explicit(
            body_contact_counts + body, memory_order_relaxed);
    if (supported == 0u) return;
    const float3 normal = normalize(pm_load(contact.normal));
    const float3 original = pm_load(velocities[particle]);
    const float3 tangent = original - normal * dot(original, normal);
    const float inverse = inverse_masses[particle];
    const float3 force = pm_load(diagnostics[particle]);
    const float impulse_acceleration = dot(force, force) > 1.0e-10f
        ? length(force) * inverse
        : 0.0f;
    const float supported_acceleration =
        abs(dot(pm_load(constants.gravity), normal)) *
        constants.movable_mass * inverse / float(supported);
    const float constraint_delta =
        contact_accumulators[particle].normal_delta /
        max(constants.timestep, 1.0e-12f);
    device const PMRigidParameters &body = rigid_parameters[contact.body];
    const float friction = body.motion == 2u
        ? sqrt(max(constants.friction * body.friction, 0.0f))
        : constants.friction;
    const float maximum_delta = friction * max(
        max(impulse_acceleration, supported_acceleration) *
            constants.timestep,
        constraint_delta);
    float3 accumulated =
        pm_load(contact_accumulators[particle].friction_delta);
    accumulated -= normal * dot(accumulated, normal);
    const float3 next_accumulated = pm_limit(
        accumulated - tangent, maximum_delta);
    const float3 change = next_accumulated - accumulated;
    const float3 velocity = pm_limit(
        original + change, constants.maximum_reaction_speed);
    velocities[particle] = pm_store(velocity);
    positions[particle] = pm_store(
        pm_load(positions[particle]) + (velocity - original) *
            constants.timestep);
    contact_accumulators[particle].friction_delta =
        pm_store(next_accumulated);
    const float3 node_impulse = change / inverse;
    contact_accumulators[particle].momentum_delta = pm_store(
        pm_load(contact_accumulators[particle].momentum_delta) +
        node_impulse);
    diagnostics[particle] = pm_store(
        pm_load(diagnostics[particle]) +
        node_impulse / max(constants.timestep, 1.0e-12f));
    const float3 reaction = -node_impulse;
    particle_linear_impulses[particle] = pm_store(reaction);
    particle_angular_impulses[particle] = pm_store(
        cross(pm_load(contact_accumulators[particle].arm), reaction));
}

kernel void pm_soft_contact_finish(
    device PMRigidBodyState *rigid_states [[buffer(3)]],
    device const PMRigidParameters *rigid_parameters [[buffer(4)]],
    constant PMParticleRigidConstants &constants [[buffer(8)]],
    device const PMParticleRigidContact *contacts [[buffer(18)]],
    device const PMPackedVec3 *particle_linear_impulses [[buffer(24)]],
    device const PMPackedVec3 *particle_angular_impulses [[buffer(25)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    const uint lane_count = threads_per_group.x;
    for (uint body = lane; body < constants.rigid_count;
         body += lane_count) {
        device const PMRigidParameters &parameters = rigid_parameters[body];
        if (parameters.motion != 2u) continue;
        float3 linear = 0.0f;
        float3 angular = 0.0f;
        for (uint particle = 0u; particle < constants.particle_count;
             ++particle) {
            if (contacts[particle].body != body) continue;
            linear += pm_load(particle_linear_impulses[particle]);
            angular += pm_load(particle_angular_impulses[particle]);
        }
        device PMRigidBodyState &state = rigid_states[body];
        state.linear_velocity = pm_store(pm_limit(
            pm_load(state.linear_velocity) +
                linear * parameters.inverse_mass,
            parameters.maximum_linear_speed));
        state.angular_velocity = pm_store(pm_limit(
            pm_load(state.angular_velocity) +
                pm_inverse_inertia_mul(parameters, state, angular),
            parameters.maximum_angular_speed));
    }
}

kernel void pm_smoke_rigid(
    device PMPackedVec3 *positions [[buffer(0)]],
    device PMPackedVec3 *velocities [[buffer(1)]],
    device const float *ages [[buffer(2)]],
    device const PMSmokeMetadata &metadata [[buffer(3)]],
    device const PMHandle *rigid_ids [[buffer(4)]],
    device PMRigidBodyState *rigid_states [[buffer(5)]],
    device PMRigidParameters *rigid_parameters [[buffer(6)]],
    device const PMPackedVec3 *vertices [[buffer(7)]],
    device const uint *indices [[buffer(8)]],
    device const PMTriangleMeshInfo *meshes [[buffer(9)]],
    constant PMSmokeRigidConstants &constants [[buffer(10)]],
    device const float *grid_pressure [[buffer(11)]],
    device const float *grid_density [[buffer(12)]],
    device const float *grid_face_velocity [[buffer(13)]],
    device const PMPackedVec3 *previous_positions [[buffer(14)]],
    device const PMRigidBodyState *old_rigid_states [[buffer(15)]],
    device float *particle_pressures [[buffer(16)]],
    device PMPackedVec3 *particle_linear_impulses [[buffer(17)]],
    device PMPackedVec3 *particle_angular_impulses [[buffer(18)]],
    uint thread_index [[thread_position_in_threadgroup]],
    uint thread_count [[threads_per_threadgroup]]) {
    if (constants.enabled == 0u) return;
    uint body_index = constants.rigid_count;
    for (uint index = 0; index < constants.rigid_count; ++index) {
        if (rigid_ids[index].index == constants.body.index &&
            rigid_ids[index].generation == constants.body.generation) {
            body_index = index;
            break;
        }
    }
    if (body_index == constants.rigid_count) return;
    device PMRigidBodyState &body_state = rigid_states[body_index];
    device PMRigidParameters &body_parameters = rigid_parameters[body_index];
    device const PMTriangleMeshInfo &mesh =
        meshes[body_parameters.mesh_index];

    if (thread_index == 0u && constants.grid_resolution != 0u &&
        body_parameters.motion == 2u) {
        const uint resolution = constants.grid_resolution;
        const uint vertical = constants.grid_vertical_resolution;
        const float spacing = constants.grid_spacing;
        const float3 minimum = pm_load(constants.grid_minimum);
        const float3 maximum = minimum + spacing *
            float3(float(resolution), float(vertical), float(resolution));
        float3 linear_force = 0.0f;
        float3 angular_force = 0.0f;
        for (uint local = 0u; local < mesh.index_count; local += 3u) {
            const uint first = indices[mesh.index_offset + local];
            const uint second = indices[mesh.index_offset + local + 1u];
            const uint third = indices[mesh.index_offset + local + 2u];
            const float3 a = pm_world_point(
                body_state, pm_load(vertices[mesh.vertex_offset + first]));
            const float3 b = pm_world_point(
                body_state, pm_load(vertices[mesh.vertex_offset + second]));
            const float3 c = pm_world_point(
                body_state, pm_load(vertices[mesh.vertex_offset + third]));
            const float3 twice_area = cross(b - a, c - a);
            const float area = 0.5f * length(twice_area);
            if (area <= 1.0e-9f) continue;
            const float3 normal = normalize(twice_area);
            const float3 center = (a + b + c) / 3.0f;
            const float3 plus = center + normal * (1.5f * spacing);
            const float3 minus = center - normal * (1.5f * spacing);
            if (any(plus < minimum) || any(plus >= maximum) ||
                any(minus < minimum) || any(minus >= maximum))
                continue;
            const float plus_density = constants.density_scale *
                pm_smoke_sample_cell_scalar(
                    grid_density, plus, minimum, spacing, resolution,
                    vertical);
            const float minus_density = constants.density_scale *
                pm_smoke_sample_cell_scalar(
                    grid_density, minus, minimum, spacing, resolution,
                    vertical);
            if (plus_density + minus_density < 1.0e-4f) continue;
            const float3 plus_air = pm_smoke_sample_face_velocity(
                grid_face_velocity, plus, minimum, spacing, resolution,
                vertical);
            const float3 minus_air = pm_smoke_sample_face_velocity(
                grid_face_velocity, minus, minimum, spacing, resolution,
                vertical);
            const float3 arm = center - pm_load(body_state.position);
            const float3 wall = pm_load(body_state.linear_velocity) +
                cross(pm_load(body_state.angular_velocity), arm);
            const float3 plus_relative = plus_air - wall;
            const float3 minus_relative = minus_air - wall;
            const float3 plus_tangent =
                plus_relative - normal * dot(plus_relative, normal);
            const float3 minus_tangent =
                minus_relative - normal * dot(minus_relative, normal);
            const float pressure_difference =
                minus_density * pm_smoke_sample_cell_scalar(
                    grid_pressure, minus, minimum, spacing, resolution,
                    vertical) -
                plus_density * pm_smoke_sample_cell_scalar(
                    grid_pressure, plus, minimum, spacing, resolution,
                    vertical);
            const float strain = pm_smoke_sample_face_strain(
                grid_face_velocity, center, minimum, spacing, resolution,
                vertical);
            const float viscosity = constants.kinematic_viscosity +
                constants.les_coefficient * constants.les_coefficient *
                    spacing * spacing * strain;
            const float3 force =
                (normal * pressure_difference +
                 (plus_tangent * plus_density +
                  minus_tangent * minus_density) *
                     (constants.drag_coefficient * viscosity /
                      (1.5f * spacing))) *
                (constants.air_density * area);
            linear_force += force;
            angular_force += cross(arm, force);
        }
        body_state.linear_velocity = pm_store(pm_limit(
            pm_load(body_state.linear_velocity) +
                linear_force *
                    (constants.timestep * body_parameters.inverse_mass),
            body_parameters.maximum_linear_speed));
        body_state.angular_velocity = pm_store(pm_limit(
            pm_load(body_state.angular_velocity) + pm_inverse_inertia_mul(
                body_parameters, body_state,
                angular_force * constants.timestep),
            body_parameters.maximum_angular_speed));
    }

    threadgroup_barrier(mem_flags::mem_device);

    if (constants.tracer_contact == 0u) return;
    const PMRigidBodyState contact_state = body_state;
    const PMRigidBodyState old_contact_state = old_rigid_states[body_index];
    const float3 initial_body_linear =
        pm_load(contact_state.linear_velocity);
    const float3 initial_body_angular =
        pm_load(contact_state.angular_velocity);
    const float clearance = constants.contact_distance;
    const float boundary = 3.0f * clearance;
    const float particle_mass = constants.grid_resolution == 0u
        ? constants.air_density * powr(2.0f * constants.particle_radius, 3.0f)
        : 0.0f;
    const PMQuaternion inverse =
        pm_quaternion_conjugate(contact_state.orientation);
    const PMQuaternion old_inverse =
        pm_quaternion_conjugate(old_contact_state.orientation);
    for (uint particle = thread_index; particle < metadata.count;
         particle += thread_count) {
        particle_linear_impulses[particle] = pm_store(float3(0.0f));
        particle_angular_impulses[particle] = pm_store(float3(0.0f));
        if (ages[particle] >= constants.lifetime) continue;
        const float3 world_position = pm_load(positions[particle]);
        const float3 before = pm_rotate(
            old_inverse,
            pm_load(previous_positions[particle]) -
                pm_load(old_contact_state.position));
        const float3 point = pm_rotate(
            inverse, world_position - pm_load(contact_state.position));
        const float3 lower = min(before, point) - boundary;
        const float3 upper = max(before, point) + boundary;
        if (any(upper < pm_load(mesh.minimum)) ||
            any(lower > pm_load(mesh.maximum)))
            continue;
        float nearest_squared = boundary * boundary;
        float earliest = 2.0f;
        bool found = false;
        bool swept = false;
        float3 nearest = 0.0f;
        float3 face_normal = float3(1.0f, 0.0f, 0.0f);
        const float3 path = point - before;
        for (uint local = 0; local < mesh.index_count; local += 3u) {
            const uint first = indices[mesh.index_offset + local];
            const uint second = indices[mesh.index_offset + local + 1u];
            const uint third = indices[mesh.index_offset + local + 2u];
            const float3 a =
                pm_load(vertices[mesh.vertex_offset + first]);
            const float3 b =
                pm_load(vertices[mesh.vertex_offset + second]);
            const float3 c =
                pm_load(vertices[mesh.vertex_offset + third]);
            if (any(upper < min(a, min(b, c))) ||
                any(lower > max(a, max(b, c))))
                continue;
            const float3 raw_normal = cross(b - a, c - a);
            if (dot(raw_normal, raw_normal) < 1.0e-12f) continue;
            const float side_before = dot(before - a, raw_normal);
            const float side_after = dot(point - a, raw_normal);
            if (side_before * side_after < 0.0f) {
                const float fraction =
                    side_before / (side_before - side_after);
                if (fraction < earliest) {
                    const float3 hit = before + path * fraction;
                    const float3 on_face =
                        pm_closest_point_triangle(hit, a, b, c);
                    if (dot(hit - on_face, hit - on_face) < 1.0e-8f) {
                        earliest = fraction;
                        nearest = on_face;
                        face_normal = normalize(raw_normal);
                        swept = true;
                        found = true;
                    }
                }
            }
            if (swept) continue;
            const float3 candidate =
                pm_closest_point_triangle(point, a, b, c);
            const float squared = dot(point - candidate, point - candidate);
            if (squared >= nearest_squared) continue;
            nearest_squared = squared;
            nearest = candidate;
            face_normal = normalize(raw_normal);
            found = true;
        }
        if (!found) continue;
        float side = dot(before - nearest, face_normal);
        if (abs(side) < 1.0e-5f)
            side = dot(point - nearest, face_normal);
        const float3 local_normal = side >= 0.0f
                                        ? face_normal
                                        : -face_normal;
        const float3 normal =
            pm_rotate(contact_state.orientation, local_normal);
        const float3 arm = pm_rotate(contact_state.orientation, nearest);
        const float3 surface_point =
            pm_load(contact_state.position) + arm;
        const bool touching = swept || nearest_squared < clearance * clearance;
        const float3 trace_position =
            surface_point + normal *
                (constants.grid_resolution != 0u
                     ? 1.5f * constants.grid_spacing
                     : clearance);
        if (touching) positions[particle] = pm_store(trace_position);
        const float3 surface_velocity =
            initial_body_linear + cross(initial_body_angular, arm);
        const float3 old_velocity = pm_load(velocities[particle]);
        float3 tracer_velocity = old_velocity;
        if (constants.grid_resolution != 0u && touching) {
            const float3 minimum = pm_load(constants.grid_minimum);
            const float3 local =
                (trace_position - minimum) / constants.grid_spacing;
            if (all(local >= 0.0f) &&
                local.x < float(constants.grid_resolution) &&
                local.y < float(constants.grid_vertical_resolution) &&
                local.z < float(constants.grid_resolution))
                tracer_velocity = pm_smoke_sample_face_velocity(
                    grid_face_velocity, trace_position, minimum,
                    constants.grid_spacing, constants.grid_resolution,
                    constants.grid_vertical_resolution);
        }
        float3 relative = tracer_velocity - surface_velocity;
        const float normal_inflow = max(0.0f, -dot(relative, normal));
        if (touching)
            relative -= normal * min(0.0f, dot(relative, normal));
        if (constants.grid_resolution != 0u && touching) {
            const float source_speed = max(
                length(tracer_velocity - surface_velocity),
                length(old_velocity - surface_velocity));
            const float tangent_speed = length(relative);
            if (tangent_speed > 1.0e-6f && tangent_speed < source_speed)
                relative *= source_speed / tangent_speed;
        }
        const float contact_distance = touching
                                           ? clearance
                                           : sqrt(nearest_squared);
        const float q = max(
            0.0f, 1.0f - contact_distance / max(boundary, 1.0e-12f));
        if (constants.grid_resolution == 0u)
            relative *= 1.0f -
                (1.0f - exp(-constants.drag_coefficient * q * q *
                            constants.timestep));
        if (touching && constants.grid_resolution == 0u)
            particle_pressures[particle] = max(
                particle_pressures[particle],
                10.0f * constants.pressure_stiffness *
                    normal_inflow * normal_inflow);
        const float3 new_velocity = pm_limit(
            surface_velocity + relative, constants.maximum_speed);
        velocities[particle] = pm_store(new_velocity);
        const float3 reaction =
            (old_velocity - new_velocity) * particle_mass;
        particle_linear_impulses[particle] = pm_store(reaction);
        particle_angular_impulses[particle] = pm_store(
            cross(arm, reaction));
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (thread_index == 0u && body_parameters.motion == 2u &&
        particle_mass > 0.0f) {
        float3 accumulated_linear_impulse = 0.0f;
        float3 accumulated_angular_impulse = 0.0f;
        for (uint particle = 0u; particle < metadata.count; ++particle) {
            accumulated_linear_impulse +=
                pm_load(particle_linear_impulses[particle]);
            accumulated_angular_impulse +=
                pm_load(particle_angular_impulses[particle]);
        }
        body_state.linear_velocity = pm_store(pm_limit(
            pm_load(body_state.linear_velocity) +
                accumulated_linear_impulse * body_parameters.inverse_mass,
            body_parameters.maximum_linear_speed));
        body_state.angular_velocity = pm_store(pm_limit(
            pm_load(body_state.angular_velocity) +
                pm_inverse_inertia_mul(body_parameters, body_state,
                                       accumulated_angular_impulse),
            body_parameters.maximum_angular_speed));
    }
}

kernel void pm_paint(
    device const PMPackedVec3 *fluid_positions [[buffer(0)]],
    device const PMParticleMetadata &fluid_metadata [[buffer(1)]],
    device const PMPackedVec2 *uvs [[buffer(2)]],
    device atomic_uint *pixels [[buffer(3)]],
    device const PMHandle *rigid_ids [[buffer(4)]],
    device const PMRigidBodyState *rigid_states [[buffer(5)]],
    device const PMRigidParameters *rigid_parameters [[buffer(6)]],
    device const PMPackedVec3 *vertices [[buffer(7)]],
    device const uint *indices [[buffer(8)]],
    device const PMTriangleMeshInfo *meshes [[buffer(9)]],
    device const PMPackedVec3 *cloth_positions [[buffer(10)]],
    device const uint *cloth_indices [[buffer(11)]],
    device const uint *cloth_sources [[buffer(12)]],
    constant PMPaintConstants &constants [[buffer(13)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    if (constants.enabled == 0u ||
        constants.width == 0u || constants.height == 0u)
        return;
    const uint lane_count = threads_per_group.x;

    uint body_index = constants.rigid_count;
    const PMHandle body_handle = constants.mode == 0u
                                     ? constants.target
                                     : constants.source;
    for (uint index = 0u; index < constants.rigid_count; ++index) {
        if (rigid_ids[index].index == body_handle.index &&
            rigid_ids[index].generation == body_handle.generation) {
            body_index = index;
            break;
        }
    }
    if (body_index == constants.rigid_count) return;

    device const PMRigidBodyState &state = rigid_states[body_index];
    if (constants.mode == 0u) {
        device const PMTriangleMeshInfo &mesh = meshes[constants.mesh_index];
        const float query_radius = constants.particle_radius + constants.reach;
        const float query_squared = query_radius * query_radius;
        const PMQuaternion inverse_orientation =
            pm_quaternion_conjugate(state.orientation);
        for (uint particle = lane; particle < fluid_metadata.count;
             particle += lane_count) {
            const float3 local_particle = pm_rotate(
                inverse_orientation,
                pm_load(fluid_positions[particle]) - pm_load(state.position));
            float best_squared = query_squared;
            float2 best_uv = 0.0f;
            uint best_side = 0u;
            for (uint local = 0u; local < mesh.index_count; local += 3u) {
                const uint ia = indices[mesh.index_offset + local];
                const uint ib = indices[mesh.index_offset + local + 1u];
                const uint ic = indices[mesh.index_offset + local + 2u];
                const float3 a = pm_load(vertices[mesh.vertex_offset + ia]);
                const float3 b = pm_load(vertices[mesh.vertex_offset + ib]);
                const float3 c = pm_load(vertices[mesh.vertex_offset + ic]);
                const float3 nearest =
                    pm_closest_point_triangle(local_particle, a, b, c);
                const float3 delta = local_particle - nearest;
                const float squared = dot(delta, delta);
                if (squared >= best_squared) continue;
                const float3 weights = pm_triangle_weights(nearest, a, b, c);
                const float2 ua = float2(uvs[ia].x, uvs[ia].y);
                const float2 ub = float2(uvs[ib].x, uvs[ib].y);
                const float2 uc = float2(uvs[ic].x, uvs[ic].y);
                best_uv = weights.x * ua + weights.y * ub + weights.z * uc;
                best_side = dot(cross(b - a, c - a), delta) >= 0.0f
                                ? 1u
                                : 2u;
                best_squared = squared;
            }
            if (best_side == 0u) continue;
            int x = int(floor(best_uv.x * float(constants.width))) %
                    int(constants.width);
            if (x < 0) x += int(constants.width);
            const int y = clamp(
                int(floor(best_uv.y * float(constants.height))), 0,
                int(constants.height) - 1);
            atomic_fetch_or_explicit(
                &pixels[uint(y) * constants.width + uint(x)], best_side,
                memory_order_relaxed);
        }
        return;
    }

    device const PMRigidParameters &body = rigid_parameters[body_index];
    device const PMTriangleMeshInfo &mesh = meshes[body.mesh_index];
    const float3 center = pm_load(state.position);
    const float contact_radius =
        mesh.radius + body.collision_margin + constants.cloth_thickness;
    threadgroup float best_penetrations[64];
    threadgroup float3 best_contacts[64];
    threadgroup uint best_triangles[64];
    float best_penetration = 0.0f;
    float3 contact = 0.0f;
    uint best_triangle = 0xffffffffu;
    const uint triangle_count = constants.cloth_index_count / 3u;
    for (uint triangle = lane; triangle < triangle_count;
         triangle += lane_count) {
        const uint local = 3u * triangle;
        const uint ia = cloth_indices[local];
        const uint ib = cloth_indices[local + 1u];
        const uint ic = cloth_indices[local + 2u];
        if (ia >= constants.cloth_count || ib >= constants.cloth_count ||
            ic >= constants.cloth_count)
            continue;
        const float3 a = pm_load(cloth_positions[ia]);
        const float3 b = pm_load(cloth_positions[ib]);
        const float3 c = pm_load(cloth_positions[ic]);
        if (dot(cross(b - a, c - a), cross(b - a, c - a)) <= 1.0e-12f)
            continue;
        const float3 nearest = pm_closest_point_triangle(center, a, b, c);
        const float penetration = contact_radius - distance(center, nearest);
        if (penetration > best_penetration) {
            best_penetration = penetration;
            contact = nearest;
            best_triangle = triangle;
        }
    }
    best_penetrations[lane] = best_penetration;
    best_contacts[lane] = contact;
    best_triangles[lane] = best_triangle;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lane_count >> 1u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            const float candidate = best_penetrations[lane + stride];
            const uint candidate_triangle = best_triangles[lane + stride];
            if (candidate > best_penetrations[lane] ||
                (candidate == best_penetrations[lane] &&
                 candidate_triangle < best_triangles[lane])) {
                best_penetrations[lane] = candidate;
                best_contacts[lane] = best_contacts[lane + stride];
                best_triangles[lane] = candidate_triangle;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    best_penetration = best_penetrations[0];
    contact = best_contacts[0];
    if (best_penetration <= 0.0f) return;

    const float brush_squared = constants.reach * constants.reach;
    for (uint triangle = lane; triangle < triangle_count;
         triangle += lane_count) {
        const uint local = 3u * triangle;
        const uint ia = cloth_indices[local];
        const uint ib = cloth_indices[local + 1u];
        const uint ic = cloth_indices[local + 2u];
        if (ia >= constants.cloth_count || ib >= constants.cloth_count ||
            ic >= constants.cloth_count)
            continue;
        const float3 a = pm_load(cloth_positions[ia]);
        const float3 b = pm_load(cloth_positions[ib]);
        const float3 c = pm_load(cloth_positions[ic]);
        const float3 nearest = pm_closest_point_triangle(contact, a, b, c);
        if (dot(contact - nearest, contact - nearest) > brush_squared)
            continue;
        const float2 ua = float2(uvs[cloth_sources[ia]].x,
                                 uvs[cloth_sources[ia]].y);
        const float2 ub = float2(uvs[cloth_sources[ib]].x,
                                 uvs[cloth_sources[ib]].y);
        const float2 uc = float2(uvs[cloth_sources[ic]].x,
                                 uvs[cloth_sources[ic]].y);
        const float2 first_edge = ub - ua;
        const float2 second_edge = uc - ua;
        const float determinant = first_edge.x * second_edge.y -
                                  first_edge.y * second_edge.x;
        if (abs(determinant) < 1.0e-10f) continue;
        const float minimum_u = min(ua.x, min(ub.x, uc.x));
        const float maximum_u = max(ua.x, max(ub.x, uc.x));
        const float minimum_v = min(ua.y, min(ub.y, uc.y));
        const float maximum_v = max(ua.y, max(ub.y, uc.y));
        const int x0 = max(0, int(floor(minimum_u * constants.width)));
        const int x1 = min(int(constants.width) - 1,
                           int(floor(maximum_u * constants.width)));
        const int y0 = max(0, int(floor(minimum_v * constants.height)));
        const int y1 = min(int(constants.height) - 1,
                           int(floor(maximum_v * constants.height)));
        if (x0 > x1 || y0 > y1) continue;
        const uint side = dot(cross(b - a, c - a), center - contact) >= 0.0f
                              ? 1u
                              : 2u;
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const float2 query =
                    (float2(float(x) + 0.5f, float(y) + 0.5f) /
                         float2(float(constants.width),
                                float(constants.height))) -
                    ua;
                const float v = (query.x * second_edge.y -
                                 query.y * second_edge.x) /
                                determinant;
                const float w = (first_edge.x * query.y -
                                 first_edge.y * query.x) /
                                determinant;
                const float u = 1.0f - v - w;
                if (u < -1.0e-4f || v < -1.0e-4f || w < -1.0e-4f)
                    continue;
                const float3 point = u * a + v * b + w * c;
                if (dot(point - contact, point - contact) <= brush_squared)
                    atomic_fetch_or_explicit(
                        &pixels[uint(y) * constants.width + uint(x)], side,
                        memory_order_relaxed);
            }
        }
    }
}
