// SPDX-License-Identifier: MIT
// Slang-backed Metal compatibility adapter for AVBD rigid constraints.

#include "constraint_math.slang"

// These names are deliberately private to this generated AIR object. Metal's
// linker does not coalesce Slang-generated aggregate initializers across
// independently compiled modules.
struct MetalAvbdPackedVec3
{
    float x;
    float y;
    float z;
};

struct MetalAvbdQuaternion
{
    float x;
    float y;
    float z;
    float w;
};

struct MetalAvbdRigidState
{
    MetalAvbdPackedVec3 position;
    MetalAvbdQuaternion orientation;
    MetalAvbdPackedVec3 linear_velocity;
    MetalAvbdPackedVec3 angular_velocity;
};

struct MetalAvbdRigidParameters
{
    uint motion;
    float inverse_mass;
    MetalAvbdPackedVec3 inverse_inertia;
    float linear_damping;
    float angular_damping;
    float maximum_linear_speed;
    float maximum_angular_speed;
    uint has_kinematic_target;
    MetalAvbdRigidState kinematic_target;
    uint mesh_index;
    float friction;
    float restitution;
    float collision_margin;
};

struct MetalAvbdStepConstants
{
    float timestep;
    MetalAvbdPackedVec3 gravity;
    uint body_count;
    uint constraint_capacity;
    uint collect_rigid_contacts;
    uint rigid_event_capacity;
    uint substeps;
    uint cooperative_contact_generation;
    uint rigid_sleeping;
    uint rigid_contact_pass_limit;
};

struct MetalAvbdContactDual
{
    float lambda;
    float penalty;
};

struct MetalAvbdContactRecord
{
    MetalAvbdPackedVec3 point;
    MetalAvbdPackedVec3 normal;
    float penetration;
    uint found;
    float accumulated_normal_impulse;
    MetalAvbdPackedVec3 accumulated_friction_impulse;
    float initial_normal_speed;
    float impact_fraction;
    uint persistent;
    uint warm_started;
    MetalAvbdContactDual avbd_dual[3];
    MetalAvbdPackedVec3 avbd_anchor_a;
    MetalAvbdPackedVec3 avbd_anchor_b;
    MetalAvbdPackedVec3 avbd_error;
    uint avbd_cached;
};

struct MetalAvbdContactManifold
{
    MetalAvbdContactRecord contacts[8];
    uint count;
    uint event_offset;
    uint color;
    MetalAvbdPackedVec3 initial_relative_position;
    uint face_patch;
    uint body_fixed_member;
    uint collider_fixed_member;
    uint cached;
    uint avbd_next_a;
    uint avbd_next_b;
};

struct MetalAvbdHandle
{
    uint index;
    uint generation;
};

struct MetalAvbdCollisionGroup
{
    uint root;
    uint eligible;
};

struct MetalAvbdConstraintResource
{
    uint generation;
    uint alive;
    uint type;
    uint body_a;
    uint body_b;
    uint enabled;
    uint broken;
    MetalAvbdPackedVec3 local_anchor_a;
    MetalAvbdPackedVec3 local_anchor_b;
    MetalAvbdQuaternion local_orientation_a;
    MetalAvbdQuaternion local_orientation_b;
    float breaking_impulse_threshold;
    float applied_impulse;
    uint linear_limit_axes;
    MetalAvbdPackedVec3 linear_limit_lower;
    MetalAvbdPackedVec3 linear_limit_upper;
    uint angular_limit_axes;
    MetalAvbdPackedVec3 angular_limit_lower;
    MetalAvbdPackedVec3 angular_limit_upper;
    uint linear_spring_axes;
    MetalAvbdPackedVec3 linear_spring_stiffness;
    MetalAvbdPackedVec3 linear_spring_damping;
    uint angular_spring_axes;
    MetalAvbdPackedVec3 angular_spring_stiffness;
    MetalAvbdPackedVec3 angular_spring_damping;
    uint linear_motor_enabled;
    uint angular_motor_enabled;
    float linear_target_velocity;
    float linear_maximum_impulse;
    float angular_target_velocity;
    float angular_maximum_impulse;
    uint solver_iterations;
    uint disable_collisions;
    AvbdRow avbd_rows[18];
    uint avbd_count;
    uint avbd_next_a;
    uint avbd_next_b;
};

struct MetalAvbdBody
{
    MetalAvbdRigidState target;
    MetalAvbdQuaternion current_orientation;
    AvbdVector6 displacement;
    AvbdBlock inertia;
    uint contact_head;
    uint joint_head;
    uint color;
    uint parent;
    uint impact;
    MetalAvbdPackedVec3 previous_velocity;
    uint history_valid;
};

struct MetalAvbdSchedule
{
    uint budget;
    uint colors;
    uint maximum_passes;
    uint candidates;
    uint contacts;
    uint islands;
    uint reserved[2];
};

struct MetalRigidContactEvent
{
    MetalAvbdHandle body;
    MetalAvbdHandle collider;
    MetalAvbdPackedVec3 position;
    MetalAvbdPackedVec3 normal;
    float penetration;
    float normal_impulse;
    MetalAvbdPackedVec3 friction_impulse;
};

static const uint pm_avbd_cached_bodies = 512u;
static const uint pm_avbd_cached_contacts = 512u;
static const uint pm_avbd_cached_joints = 64u;

struct PMAvbdGraphBody
{
    uint parent;
    uint color;
    uint contact_head;
    uint joint_head;
    uint flags;
};

struct PMAvbdGraphContact
{
    uint a;
    uint b;
    uint next_a;
    uint next_b;
    uint flags;
};

struct PMAvbdGraphJoint
{
    uint a;
    uint b;
    uint next_a;
    uint next_b;
    uint count;
    uint iterations;
};

groupshared PMAvbdGraphBody body_cache[pm_avbd_cached_bodies];
groupshared PMAvbdGraphContact contact_cache[pm_avbd_cached_contacts];
groupshared PMAvbdGraphJoint joint_cache[pm_avbd_cached_joints];

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalAvbdRigidState> rigidStates : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<MetalAvbdRigidParameters> rigidParameters : register(t1);
[[vk::binding(4, 0)]] ConstantBuffer<MetalAvbdStepConstants> rigidStep : register(b4);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalAvbdContactManifold> contactManifolds : register(u8);
[[vk::binding(9, 0)]] RWStructuredBuffer<MetalAvbdConstraintResource> rigidConstraints : register(u9);
[[vk::binding(10, 0)]] StructuredBuffer<MetalAvbdHandle> rigidIds : register(t10);
[[vk::binding(11, 0)]] RWStructuredBuffer<MetalRigidContactEvent> rigidEvents : register(u11);
[[vk::binding(12, 0)]] RWStructuredBuffer<uint> eventCountBuffer : register(u12);
[[vk::binding(14, 0)]] StructuredBuffer<MetalAvbdRigidState> previousStates : register(t14);
[[vk::binding(18, 0)]] StructuredBuffer<uint> activePairs : register(t18);
[[vk::binding(19, 0)]] StructuredBuffer<uint> pairCountBuffer : register(t19);
[[vk::binding(22, 0)]] RWStructuredBuffer<MetalAvbdBody> avbdBodies : register(u22);
[[vk::binding(25, 0)]] RWStructuredBuffer<MetalAvbdCollisionGroup> collisionGroups : register(u25);
[[vk::binding(29, 0)]] RWStructuredBuffer<MetalAvbdSchedule> scheduleBuffer : register(u29);

static float3 pm_avbd_load(MetalAvbdPackedVec3 value)
{
    return float3(value.x, value.y, value.z);
}

static MetalAvbdPackedVec3 pm_avbd_store(float3 value)
{
    return {value.x, value.y, value.z};
}

static float3 pm_avbd_rotate(MetalAvbdQuaternion q, float3 point)
{
    float3 vectorPart = float3(q.x, q.y, q.z);
    float3 twiceCross = 2.0f * cross(vectorPart, point);
    return point + q.w * twiceCross + cross(vectorPart, twiceCross);
}

static MetalAvbdQuaternion pm_avbd_quaternion_multiply(
    MetalAvbdQuaternion a, MetalAvbdQuaternion b)
{
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

static MetalAvbdQuaternion pm_avbd_quaternion_conjugate(MetalAvbdQuaternion value)
{
    return {-value.x, -value.y, -value.z, value.w};
}

static MetalAvbdQuaternion pm_avbd_quaternion_normalize(MetalAvbdQuaternion value)
{
    float inverseLength = rsqrt(max(
        value.x * value.x + value.y * value.y +
        value.z * value.z + value.w * value.w, 1.0e-12f));
    return {value.x * inverseLength, value.y * inverseLength,
        value.z * inverseLength, value.w * inverseLength};
}

static float3 pm_avbd_limit(float3 value, float maximumLength)
{
    float squaredLength = dot(value, value);
    float squaredMaximum = maximumLength * maximumLength;
    return squaredLength > squaredMaximum
        ? value * (maximumLength * rsqrt(squaredLength)) : value;
}

static float3 pm_avbd_quaternion_delta_velocity(
    MetalAvbdQuaternion from, MetalAvbdQuaternion to, float timestep)
{
    MetalAvbdQuaternion delta = pm_avbd_quaternion_multiply(
        to, pm_avbd_quaternion_conjugate(from));
    if (delta.w < 0.0f)
        delta = {-delta.x, -delta.y, -delta.z, -delta.w};
    delta = pm_avbd_quaternion_normalize(delta);
    float vectorSize = length(float3(delta.x, delta.y, delta.z));
    if (timestep <= 0.0f) return float3(0.0f);
    if (vectorSize <= 1.0e-6f)
        return float3(delta.x, delta.y, delta.z) * (2.0f / timestep);
    float angle = 2.0f * atan2(vectorSize, clamp(delta.w, -1.0f, 1.0f));
    return float3(delta.x, delta.y, delta.z) *
        (angle / (vectorSize * timestep));
}

static float3 pm_avbd_inverse_inertia_mul_orientation(
    MetalAvbdRigidParameters body, MetalAvbdQuaternion orientation, float3 value)
{
    float3 local = pm_avbd_rotate(pm_avbd_quaternion_conjugate(orientation), value);
    return pm_avbd_rotate(orientation, local * pm_avbd_load(body.inverse_inertia));
}

static float3 pm_avbd_relative_rotation_vector(
    MetalAvbdQuaternion frameA, MetalAvbdQuaternion frameB)
{
    MetalAvbdQuaternion relative = pm_avbd_quaternion_normalize(
        pm_avbd_quaternion_multiply(pm_avbd_quaternion_conjugate(frameA), frameB));
    if (relative.w < 0.0f)
        relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    float size = length(float3(relative.x, relative.y, relative.z));
    if (size <= 1.0e-6f)
        return 2.0f * float3(relative.x, relative.y, relative.z);
    float angle = 2.0f * atan2(size, clamp(relative.w, -1.0f, 1.0f));
    return float3(relative.x, relative.y, relative.z) * (angle / size);
}

static uint pm_avbd_collision_group_root(uint body)
{
    while (collisionGroups[body].root != body)
        body = collisionGroups[body].root;
    return body;
}

static PMAvbdGraphBody pm_avbd_graph_body_device(uint body)
{
    MetalAvbdBody value = avbdBodies[body];
    return {value.parent, value.color, value.contact_head, value.joint_head,
        (rigidParameters[body].motion == 2u ? 1u : 0u) |
        (value.impact != 0u ? 2u : 0u)};
}

static PMAvbdGraphBody pm_avbd_graph_body(uint body)
{
    return body < pm_avbd_cached_bodies
        ? body_cache[body] : pm_avbd_graph_body_device(body);
}

static void pm_avbd_graph_store_body_device(
    uint body, PMAvbdGraphBody value)
{
    MetalAvbdBody target = avbdBodies[body];
    target.parent = value.parent;
    target.color = value.color;
    target.contact_head = value.contact_head;
    target.joint_head = value.joint_head;
    target.impact = (value.flags & 2u) != 0u ? 1u : 0u;
    avbdBodies[body] = target;
}

static void pm_avbd_graph_store_body(uint body, PMAvbdGraphBody value)
{
    if (body < pm_avbd_cached_bodies) body_cache[body] = value;
    else pm_avbd_graph_store_body_device(body, value);
}

static PMAvbdGraphContact pm_avbd_graph_contact_device(uint index)
{
    MetalAvbdContactManifold manifold = contactManifolds[index];
    return {activePairs[index] / rigidStep.body_count,
        activePairs[index] % rigidStep.body_count,
        manifold.avbd_next_a, manifold.avbd_next_b,
        manifold.count != 0u ? 1u : 0u};
}

static PMAvbdGraphContact pm_avbd_graph_contact(uint index)
{
    return index < pm_avbd_cached_contacts
        ? contact_cache[index] : pm_avbd_graph_contact_device(index);
}

static void pm_avbd_graph_store_contact_device(
    uint index, PMAvbdGraphContact value)
{
    MetalAvbdContactManifold manifold = contactManifolds[index];
    manifold.avbd_next_a = value.next_a;
    manifold.avbd_next_b = value.next_b;
    contactManifolds[index] = manifold;
}

static void pm_avbd_graph_store_contact(
    uint index, PMAvbdGraphContact value)
{
    if (index < pm_avbd_cached_contacts) contact_cache[index] = value;
    else pm_avbd_graph_store_contact_device(index, value);
}

static PMAvbdGraphJoint pm_avbd_graph_joint_device(uint index)
{
    MetalAvbdConstraintResource joint = rigidConstraints[index];
    return {joint.body_a, joint.body_b, joint.avbd_next_a,
        joint.avbd_next_b, joint.avbd_count, joint.solver_iterations};
}

static PMAvbdGraphJoint pm_avbd_graph_joint(uint index)
{
    return index < pm_avbd_cached_joints
        ? joint_cache[index] : pm_avbd_graph_joint_device(index);
}

static void pm_avbd_graph_store_joint_device(
    uint index, PMAvbdGraphJoint value)
{
    MetalAvbdConstraintResource joint = rigidConstraints[index];
    joint.avbd_next_a = value.next_a;
    joint.avbd_next_b = value.next_b;
    rigidConstraints[index] = joint;
}

static void pm_avbd_graph_store_joint(
    uint index, PMAvbdGraphJoint value)
{
    if (index < pm_avbd_cached_joints) joint_cache[index] = value;
    else pm_avbd_graph_store_joint_device(index, value);
}

static uint pm_avbd_graph_root(uint body)
{
    for (;;)
    {
        uint parent = pm_avbd_graph_body(body).parent;
        if (parent == body) return body;
        body = parent;
    }
}
static void pm_avbd_graph_union(uint a, uint b)
{
    if ((pm_avbd_graph_body(a).flags & 1u) == 0u ||
        (pm_avbd_graph_body(b).flags & 1u) == 0u) return;
    a = pm_avbd_graph_root(a);
    b = pm_avbd_graph_root(b);
    if (a != b)
    {
        uint highIndex = max(a, b);
        PMAvbdGraphBody high = pm_avbd_graph_body(highIndex);
        high.parent = min(a, b);
        pm_avbd_graph_store_body(highIndex, high);
    }
}

static AvbdVector6 pm_avbd_jacobian(float3 linear, float3 angular) {
    return {{linear.x, linear.y, linear.z, angular.x, angular.y, angular.z}};
}
static float3 pm_avbd_translation(AvbdVector6 value) {
    return float3(value.values[0], value.values[1], value.values[2]);
}
static float3 pm_avbd_rotation(AvbdVector6 value) {
    return float3(value.values[3], value.values[4], value.values[5]);
}
static float3 pm_avbd_axis(uint axis) {
    return axis == 0u ? float3(1, 0, 0)
        : axis == 1u ? float3(0, 1, 0) : float3(0, 0, 1);
}
static MetalAvbdRigidState pm_avbd_pose(MetalAvbdBody body) {
    MetalAvbdRigidState state = body.target;
    state.position = pm_avbd_store(pm_avbd_load(state.position) + pm_avbd_translation(body.displacement));
    state.orientation = body.current_orientation;
    return state;
}
static void pm_avbd_apply_pose_update(inout MetalAvbdBody body, AvbdVector6 update) {
    for (uint i = 0u; i < 3u; ++i) body.displacement.values[i] += update.values[i];
    const float3 rotation = pm_avbd_rotation(update);
    const MetalAvbdQuaternion dq = pm_avbd_quaternion_multiply({rotation.x, rotation.y, rotation.z, 0}, body.current_orientation);
    body.current_orientation = pm_avbd_quaternion_normalize({
        body.current_orientation.x + 0.5f * dq.x, body.current_orientation.y + 0.5f * dq.y,
        body.current_orientation.z + 0.5f * dq.z, body.current_orientation.w + 0.5f * dq.w});
    MetalAvbdQuaternion relative = pm_avbd_quaternion_multiply(body.current_orientation, pm_avbd_quaternion_conjugate(body.target.orientation));
    if (relative.w < 0.0f) relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    body.displacement.values[3] = 2.0f * relative.x;
    body.displacement.values[4] = 2.0f * relative.y;
    body.displacement.values[5] = 2.0f * relative.z;
}
static float3 pm_avbd_log_axis(MetalAvbdQuaternion frame, float3 logarithm, uint axis) {
    const float squared = dot(logarithm, logarithm);
    float coefficient = 1.0f / 12.0f + squared / 720.0f;
    if (squared > 1.0e-6f) {
        const float angle = sqrt(squared), half_angle = 0.5f * angle;
        coefficient = (1.0f - half_angle * cos(half_angle) / sin(half_angle)) / squared;
    }
    const float3 unit = pm_avbd_axis(axis);
    return pm_avbd_rotate(frame, unit + 0.5f * cross(logarithm, unit)
        + coefficient * cross(logarithm, cross(logarithm, unit)));
}
static void pm_avbd_basis(float3 normal, out float3 t0, out float3 t1) {
    t0 = normalize(cross(normal, abs(normal.x) < 0.57735f
        ? float3(1, 0, 0) : float3(0, 1, 0)));
    t1 = cross(normal, t0);
}
static float pm_avbd_pair_stiffness(MetalAvbdRigidParameters a, MetalAvbdRigidParameters b, float dt) {
    return 1.0f / (max(a.inverse_mass + b.inverse_mass, 1.0e-6f) * dt * dt);
}
static bool pm_avbd_axially_guided(
    uint body_count, uint constraint_capacity, uint body)
{
    if (body >= body_count || rigidParameters[body].inverse_mass <= 0.0f)
        return false;
    uint incident_count = 0u;
    bool axial_static_guide = false;
    for (uint index = 0u; index < constraint_capacity; ++index)
    {
        MetalAvbdConstraintResource joint = rigidConstraints[index];
        if (joint.alive == 0u || joint.enabled == 0u || joint.broken != 0u)
            continue;
        bool is_a = joint.body_a == body;
        bool is_b = joint.body_b == body;
        if (!is_a && !is_b) continue;
        ++incident_count;
        uint other = is_a ? joint.body_b : joint.body_a;
        bool axial = joint.type == 3u || joint.type == 4u;
        axial_static_guide = axial_static_guide ||
            (axial && other < body_count &&
             rigidParameters[other].motion == 0u &&
             joint.breaking_impulse_threshold <= 0.0f);
    }
    return incident_count == 1u && axial_static_guide;
}

static float pm_avbd_contact_skin(
    uint body_count, uint constraint_capacity, uint a, uint b) {
    bool guided =
        (pm_avbd_axially_guided(body_count, constraint_capacity, a) &&
         rigidParameters[b].motion == 0u) ||
        (pm_avbd_axially_guided(body_count, constraint_capacity, b) &&
         rigidParameters[a].motion == 0u);
    // Match the sharper guided CCD stand-off, not an ordinary 1 mm skin.
    const float offset = guided
        ? min(rigidParameters[a].collision_margin + rigidParameters[b].collision_margin, 1.0e-4f)
        : min(rigidParameters[a].collision_margin +
              rigidParameters[b].collision_margin, 1.0e-3f);
    return offset + 2.0e-5f;
}
static bool pm_avbd_prepare_contact_impact(
    MetalAvbdContactManifold manifold, float skin, float dt) {
    bool impact = false;
    for (uint point = 0u; point < manifold.count; ++point) {
        MetalAvbdContactRecord c = manifold.contacts[point];
        const bool new_impact = (c.avbd_cached == 0u || c.avbd_dual[0].lambda <= 1.0e-6f
            || (c.impact_fraction > 0.0f && c.impact_fraction < 1.0f))
            && c.initial_normal_speed < -0.1f && c.penetration >= -1.0e-5f;
        const float previous_penetration = c.penetration + c.initial_normal_speed * dt;
        impact |= new_impact || previous_penetration > skin + 1.0e-4f;
    }
    return impact;
}
static void pm_avbd_contact_rows(MetalAvbdContactRecord contact, uint a, uint b,
    out AvbdVector6 ja[3], out AvbdVector6 jb[3], out float error[3]) {
    float3 t0 = float3(0.0f);
    float3 t1 = float3(0.0f);
    const float3 normal = pm_avbd_load(contact.normal);
    pm_avbd_basis(normal, t0, t1);
    const float3 axes[3] = {normal, t0, t1};
    const float3 arm_a = pm_avbd_load(contact.point) - pm_avbd_load(avbdBodies[a].target.position);
    const float3 arm_b = pm_avbd_load(contact.point) - pm_avbd_load(avbdBodies[b].target.position);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float3 ra = axis == 0u ? arm_a
            : pm_avbd_rotate(avbdBodies[a].target.orientation, pm_avbd_load(contact.avbd_anchor_a));
        const float3 rb = axis == 0u ? arm_b
            : pm_avbd_rotate(avbdBodies[b].target.orientation, pm_avbd_load(contact.avbd_anchor_b));
        ja[axis] = pm_avbd_jacobian(-axes[axis], -cross(ra, axes[axis]));
        jb[axis] = pm_avbd_jacobian(axes[axis], cross(rb, axes[axis]));
        error[axis] = pm_avbd_load(contact.avbd_error)[axis]
            + avbdDot6(ja[axis], avbdBodies[a].displacement)
            + avbdDot6(jb[axis], avbdBodies[b].displacement);
    }
}
static AvbdContactForce pm_avbd_contact_trial(MetalAvbdContactRecord contact,
    float error[3]) {
    return {
        contact.avbd_dual[0].lambda + contact.avbd_dual[0].penalty * error[0],
        contact.avbd_dual[1].lambda + contact.avbd_dual[1].penalty * error[1],
        contact.avbd_dual[2].lambda + contact.avbd_dual[2].penalty * error[2]};
}
static float pm_avbd_contact_normal_row(MetalAvbdContactRecord contact, uint a, uint b,
    out AvbdVector6 ja, out AvbdVector6 jb) {
    const float3 normal = pm_avbd_load(contact.normal);
    const float3 arm_a = pm_avbd_load(contact.point) - pm_avbd_load(avbdBodies[a].target.position);
    const float3 arm_b = pm_avbd_load(contact.point) - pm_avbd_load(avbdBodies[b].target.position);
    ja = pm_avbd_jacobian(-normal, -cross(arm_a, normal));
    jb = pm_avbd_jacobian(normal, cross(arm_b, normal));
    return contact.avbd_error.x + avbdDot6(ja, avbdBodies[a].displacement)
        + avbdDot6(jb, avbdBodies[b].displacement);
}
static AvbdContactForce pm_avbd_contact_force(MetalAvbdContactRecord contact,
    float error[3], float friction) {
    return avbdProjectContact(pm_avbd_contact_trial(contact, error), friction);
}

static const uint pm_avbd_initialize_joint = 0u;
static const uint pm_avbd_accumulate_joint = 1u;
static const uint pm_avbd_advance_joint = 2u;
static const uint pm_avbd_impact_joint = 3u;
// Authored limits retain the established 20% recovery per substep.
static const float pm_avbd_limit_stabilization = 0.8f;

static void pm_avbd_visit_joint_row(inout MetalAvbdConstraintResource joint,
    inout AvbdBlock block, inout uint index, inout float impulse, bool first,
    AvbdVector6 ja, AvbdVector6 jb, float current, float previousValue,
    float penalty, float dt, bool angular, float stiffness = kAvbdMaximumPenalty,
    float damping = 0, float lo = -kAvbdMaximumPenalty, float hi = kAvbdMaximumPenalty,
    bool motor = false, float motor_speed = 0, float3 curvature_a = float3(0), float3 curvature_b = float3(0),
    float stabilization = kAvbdAlpha, uint visit = pm_avbd_initialize_joint) {
    const uint row_index = index++;
    AvbdRow row = joint.avbd_rows[row_index];
    const uint a = joint.body_a, b = joint.body_b;
    const float linearized = avbdDot6(ja, avbdBodies[a].displacement) + avbdDot6(jb, avbdBodies[b].displacement);
    row.a = ja; row.b = jb; row.stiffness = stiffness; row.damping = damping;
    if (visit == pm_avbd_initialize_joint) {
        if (!(row.dual.penalty > kAvbdMinimumPenalty)) row.dual.penalty = penalty;
        row.dual = avbdWarmStart(row.dual, stiffness);
        if (stiffness < kAvbdMaximumPenalty) row.dual.lambda = 0;
        row.beta = penalty * (angular ? 10.0f : 100.0f);
        row.referenceForce = 0;
    }
    // The shared core adds J*displacement back, yielding exactly C(current)
    // with a fresh current-pose tangent Jacobian on every body visit.
    row.error = current - (stiffness < kAvbdMaximumPenalty ? 0.0f : stabilization * previousValue) - linearized;
    row.velocity = motor ? motor_speed : (current - previousValue - linearized) / dt;
    row.lower = lo; row.upper = hi;
    if (visit == pm_avbd_accumulate_joint) {
        avbdAccumulate(block, row, first, avbdBodies[a].displacement, avbdBodies[b].displacement, dt);
        if (!angular) {
            const float3 axis = pm_avbd_translation(jb), arm = first ? curvature_a : curvature_b;
            const float force = avbdRowForce(row, avbdBodies[a].displacement, avbdBodies[b].displacement, dt);
            for (uint column = 0u; column < 3u; ++column) {
                const float3 value = 0.5f * (axis * arm[column] + arm * axis[column])
                    - pm_avbd_axis(column) * dot(axis, arm);
                block.h[3u + column][3u + column] += abs(force) * length(value);
            }
        }
    } else if (visit == pm_avbd_advance_joint) {
        impulse += abs(avbdRowForce(row, avbdBodies[a].displacement, avbdBodies[b].displacement, dt)) * dt;
        joint.avbd_rows[row_index] = avbdAdvance(row, avbdBodies[a].displacement, avbdBodies[b].displacement);
    } else if (visit == pm_avbd_impact_joint) {
        if (stiffness < kAvbdMaximumPenalty) {
            row.referenceForce = avbdRowForce(row, avbdBodies[a].displacement, avbdBodies[b].displacement, dt);
        } else {
            // Velocity limits use the actual gap, not the relaxed pose error.
            const float gap = lo == 0.0f ? min(0.0f, current) : hi == 0.0f ? max(0.0f, current) : 0.0f;
            const AvbdVector6 va = pm_avbd_jacobian(pm_avbd_load(avbdBodies[a].target.linear_velocity) * dt,
                pm_avbd_load(avbdBodies[a].target.angular_velocity) * dt);
            const AvbdVector6 vb = pm_avbd_jacobian(pm_avbd_load(avbdBodies[b].target.linear_velocity) * dt,
                pm_avbd_load(avbdBodies[b].target.angular_velocity) * dt);
            row.referenceForce = row.dual.lambda;
            row.error = avbdDot6(ja, va) + avbdDot6(jb, vb) + gap - (motor ? motor_speed * dt : 0.0f);
        }
        joint.avbd_rows[row_index] = row;
    } else {
        joint.avbd_rows[row_index] = row;
    }
}

static float pm_avbd_visit_joint(inout MetalAvbdConstraintResource j,
    float dt, inout AvbdBlock block, bool first = true,
    uint visit = pm_avbd_initialize_joint) {
    uint index = 0u; float impulse = 0;
    const uint a = j.body_a, b = j.body_b;
    const MetalAvbdRigidState sa = pm_avbd_pose(avbdBodies[a]), sb = pm_avbd_pose(avbdBodies[b]);
    const float3 ra = pm_avbd_rotate(sa.orientation, pm_avbd_load(j.local_anchor_a));
    const float3 rb = pm_avbd_rotate(sb.orientation, pm_avbd_load(j.local_anchor_b));
    const float3 error = pm_avbd_load(sb.position) + rb - pm_avbd_load(sa.position) - ra;
    const float3 old_error = pm_avbd_load(previousStates[b].position)
        + pm_avbd_rotate(previousStates[b].orientation, pm_avbd_load(j.local_anchor_b))
        - pm_avbd_load(previousStates[a].position) - pm_avbd_rotate(previousStates[a].orientation, pm_avbd_load(j.local_anchor_a));
    const MetalAvbdQuaternion fa = pm_avbd_quaternion_normalize(pm_avbd_quaternion_multiply(sa.orientation, j.local_orientation_a));
    const MetalAvbdQuaternion fb = pm_avbd_quaternion_normalize(pm_avbd_quaternion_multiply(sb.orientation, j.local_orientation_b));
    const MetalAvbdQuaternion old_fa = pm_avbd_quaternion_normalize(pm_avbd_quaternion_multiply(previousStates[a].orientation, j.local_orientation_a));
    const MetalAvbdQuaternion old_fb = pm_avbd_quaternion_normalize(pm_avbd_quaternion_multiply(previousStates[b].orientation, j.local_orientation_b));
    const float3 twist = pm_avbd_relative_rotation_vector(fa, fb), old_twist = pm_avbd_relative_rotation_vector(old_fa, old_fb);
    const bool swing = j.type == 2u || j.type == 4u || (j.type == 7u && j.angular_motor_enabled != 0u);
    const float3 free_axis = j.type == 2u ? float3(0, 0, 1) : float3(1, 0, 0);
    const float3 swing_axis = pm_avbd_rotate(pm_avbd_quaternion_conjugate(fa), pm_avbd_rotate(fb, free_axis));
    const float3 old_swing_axis = pm_avbd_rotate(pm_avbd_quaternion_conjugate(old_fa), pm_avbd_rotate(old_fb, free_axis));
    const float3 rotation = swing ? cross(free_axis, swing_axis) : twist;
    const float3 old_rotation = swing ? cross(free_axis, old_swing_axis) : old_twist;
    const float linear_penalty = pm_avbd_pair_stiffness(rigidParameters[a], rigidParameters[b], dt);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float3 n = pm_avbd_rotate(fa, pm_avbd_axis(axis)), old_n = pm_avbd_rotate(old_fa, pm_avbd_axis(axis));
        const bool generic_joint = j.type == 5u || j.type == 6u, motor = j.type == 7u;
        for (uint angular = 0u; angular < 2u; ++angular) {
            const bool lock = angular != 0u
                ? j.type == 0u || j.type == 3u || (j.type == 2u && axis != 2u)
                    || (j.type == 4u && axis != 0u)
                    || (motor && (axis != 0u || j.angular_motor_enabled == 0u))
                : j.type == 0u || j.type == 1u || j.type == 2u
                    || ((j.type == 3u || j.type == 4u) && axis != 0u)
                    || (motor && (axis != 0u || j.linear_motor_enabled == 0u));
            const float c = angular != 0u ? rotation[axis] : dot(error, n);
            const float old_c = angular != 0u ? old_rotation[axis] : dot(old_error, old_n);
            const float3 angular_axis = swing
                ? pm_avbd_rotate(fa, pm_avbd_axis(axis) * dot(free_axis, swing_axis) - free_axis * swing_axis[axis])
                : pm_avbd_log_axis(fa, twist, axis);
            const AvbdVector6 ja = angular != 0u ? pm_avbd_jacobian(float3(0), -angular_axis)
                : pm_avbd_jacobian(-n, -cross(ra, n) + cross(n, error));
            const AvbdVector6 jb = angular != 0u ? pm_avbd_jacobian(float3(0), angular_axis) : pm_avbd_jacobian(n, cross(rb, n));
            const float penalty = angular != 0u
                ? 1.0f / (max(dot(n, pm_avbd_inverse_inertia_mul_orientation(rigidParameters[a], sa.orientation, n)
                    + pm_avbd_inverse_inertia_mul_orientation(rigidParameters[b], sb.orientation, n)), 1.0e-6f) * dt * dt) : linear_penalty;
            // Parallel-axis floor conditions off-axis hard locks without
            // changing physical inertia, motors, or finite spring forces.
            // Factor two bounds the two coupled anchor contributions.
            const float3 lever_a = cross(n, ra), lever_b = cross(n, rb);
            const float lock_penalty = angular != 0u ? max(penalty,
                2.0f * linear_penalty * max(dot(lever_a, lever_a), dot(lever_b, lever_b))) : penalty;
            if (lock) {
                if (visit == pm_avbd_initialize_joint && angular != 0u)
                    j.avbd_rows[index].dual.penalty = max(j.avbd_rows[index].dual.penalty, lock_penalty / kAvbdGamma);
                pm_avbd_visit_joint_row(j, block, index, impulse, first,
                    ja, jb, c, old_c, lock_penalty, dt, angular != 0u, kAvbdMaximumPenalty, 0,
                    -kAvbdMaximumPenalty, kAvbdMaximumPenalty, false, 0,
                    error + ra, rb, kAvbdAlpha, visit);
            }
            const bool free_limit = angular != 0u
                ? (j.type == 2u && axis == 2u) || (j.type == 4u && axis == 0u)
                : (j.type == 3u || j.type == 4u) && axis == 0u;
            const uint limit_axes = angular != 0u ? j.angular_limit_axes : j.linear_limit_axes;
            if ((generic_joint || free_limit) && (limit_axes & (1u << axis)) != 0u) {
                const float limit_c = angular != 0u ? twist[axis] : c;
                const float old_limit_c = angular != 0u ? old_twist[axis] : old_c;
                const float3 limit_axis = pm_avbd_log_axis(fa, twist, axis);
                AvbdVector6 limit_ja = ja, limit_jb = jb;
                if (angular != 0u) {
                    limit_ja = pm_avbd_jacobian(float3(0), -limit_axis);
                    limit_jb = pm_avbd_jacobian(float3(0), limit_axis);
                }
                const float lower = pm_avbd_load(angular != 0u ? j.angular_limit_lower : j.linear_limit_lower)[axis];
                const float upper = pm_avbd_load(angular != 0u ? j.angular_limit_upper : j.linear_limit_upper)[axis];
                if (lower == upper) {
                    const float equal_penalty = angular != 0u && generic_joint ? lock_penalty : penalty;
                    if (visit == pm_avbd_initialize_joint && angular != 0u && generic_joint)
                        j.avbd_rows[index].dual.penalty = max(j.avbd_rows[index].dual.penalty, equal_penalty / kAvbdGamma);
                    pm_avbd_visit_joint_row(j, block, index, impulse, first,
                        limit_ja, limit_jb, limit_c - lower, old_limit_c - lower, equal_penalty, dt, angular != 0u,
                        kAvbdMaximumPenalty, 0, -kAvbdMaximumPenalty, kAvbdMaximumPenalty, false, 0, error + ra, rb,
                        kAvbdAlpha, visit);
                }
                else if (lower < upper) {
                    pm_avbd_visit_joint_row(j, block, index, impulse, first,
                        limit_ja, limit_jb, limit_c - upper, max(old_limit_c - upper, 0.0f), penalty, dt, angular != 0u,
                        kAvbdMaximumPenalty, 0, 0, kAvbdMaximumPenalty, false, 0, error + ra, rb,
                        pm_avbd_limit_stabilization, visit);
                    pm_avbd_visit_joint_row(j, block, index, impulse, first,
                        limit_ja, limit_jb, limit_c - lower, min(old_limit_c - lower, 0.0f), penalty, dt, angular != 0u,
                        kAvbdMaximumPenalty, 0, -kAvbdMaximumPenalty, 0, false, 0, error + ra, rb,
                        pm_avbd_limit_stabilization, visit);
                }
            }
            const uint spring_axes = angular != 0u ? j.angular_spring_axes : j.linear_spring_axes;
            if (j.type == 6u && (spring_axes & (1u << axis)) != 0u)
                pm_avbd_visit_joint_row(j, block, index, impulse, first,
                    ja, jb, c, old_c, penalty, dt, angular != 0u,
                    pm_avbd_load(angular != 0u ? j.angular_spring_stiffness : j.linear_spring_stiffness)[axis],
                    pm_avbd_load(angular != 0u ? j.angular_spring_damping : j.linear_spring_damping)[axis],
                    -kAvbdMaximumPenalty, kAvbdMaximumPenalty, false, 0,
                    error + ra, rb, kAvbdAlpha, visit);
            if (motor && axis == 0u && (angular != 0u ? j.angular_motor_enabled : j.linear_motor_enabled) != 0u) {
                const float speed = angular != 0u ? j.angular_target_velocity : j.linear_target_velocity;
                const float bound = (angular != 0u ? j.angular_maximum_impulse : j.linear_maximum_impulse) / dt;
                const MetalAvbdQuaternion relative = pm_avbd_quaternion_multiply(pm_avbd_quaternion_conjugate(fa), fb);
                const MetalAvbdQuaternion old_relative = pm_avbd_quaternion_multiply(pm_avbd_quaternion_conjugate(old_fa), old_fb);
                const float3 motor_rotation = pm_avbd_relative_rotation_vector({0, 0, 0, 1},
                    pm_avbd_quaternion_multiply(relative, pm_avbd_quaternion_conjugate(old_relative)));
                const float3 motor_axis = pm_avbd_log_axis(fa, motor_rotation, axis);
                AvbdVector6 motor_ja = ja, motor_jb = jb;
                if (angular != 0u) {
                    motor_ja = pm_avbd_jacobian(float3(0), -motor_axis);
                    motor_jb = pm_avbd_jacobian(float3(0), motor_axis);
                }
                const float travel = angular != 0u ? motor_rotation[axis] : c - old_c;
                pm_avbd_visit_joint_row(j, block, index, impulse, first,
                    motor_ja, motor_jb, travel - speed * dt, 0, penalty, dt, angular != 0u,
                    kAvbdMaximumPenalty, 0, -bound, bound, true, speed,
                    error + ra, rb, kAvbdAlpha, visit);
            }
        }
    }
    if (visit == pm_avbd_initialize_joint) j.avbd_count = index;
    return impulse;
}

// Collision groups are topology only. No compound momentum averaging or
// guided pose projection is performed outside the AVBD constraint graph.
[shader("compute")]
[numthreads(64, 1, 1)]
void pm_avbd_collision_groups(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint lane = dispatchThreadId.x;
    if (lane != 0u) return;
    for (uint body = 0u; body < rigidStep.body_count; ++body) {
        collisionGroups[body] = {}; collisionGroups[body].root = body;
    }
    for (uint index = 0u; index < rigidStep.constraint_capacity; ++index) {
        MetalAvbdConstraintResource j = rigidConstraints[index];
        if (j.alive == 0u || j.enabled == 0u || j.broken != 0u || j.type != 0u
            || j.disable_collisions == 0u || j.breaking_impulse_threshold > 0.0f
            || j.body_a >= rigidStep.body_count || j.body_b >= rigidStep.body_count) continue;
        const uint a = pm_avbd_collision_group_root(j.body_a);
        const uint b = pm_avbd_collision_group_root(j.body_b);
        collisionGroups[max(a, b)].root = min(a, b);
    }
    for (uint body = 0u; body < rigidStep.body_count; ++body) {
        collisionGroups[body].root = pm_avbd_collision_group_root(body);
        collisionGroups[body].eligible = 1u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pm_avbd_prepare(uint3 groupThreadId : SV_GroupThreadID) {
    uint lane = groupThreadId.x;
    uint3 group_size = uint3(256u, 1u, 1u);
    const float dt = rigidStep.timestep;
    for (uint i = lane; i < rigidStep.body_count; i += group_size.x) {
        const float3 previous_velocity = pm_avbd_load(avbdBodies[i].previous_velocity);
        const bool history_valid = avbdBodies[i].history_valid != 0u;
        avbdBodies[i] = {};
        avbdBodies[i].contact_head = avbdBodies[i].joint_head = avbdBodies[i].color = 0xffffffffu;
        avbdBodies[i].parent = i;
        avbdBodies[i].target = rigidStates[i];
        avbdBodies[i].current_orientation = rigidStates[i].orientation;
        if (i < pm_avbd_cached_bodies)
            body_cache[i] = {i, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                rigidParameters[i].motion == 2u ? 1u : 0u};
        if (rigidParameters[i].motion != 2u) continue;
        const float3 gravity = pm_avbd_load(rigidStep.gravity);
        const float gravity_squared = dot(gravity, gravity);
        const float3 acceleration = (pm_avbd_load(previousStates[i].linear_velocity) - previous_velocity) / dt;
        float weight = history_valid && gravity_squared > 0.0f
            ? clamp(dot(acceleration, gravity) / gravity_squared, 0.0f, 1.0f) : 0.0f;
        if (!isfinite(weight)) weight = 0.0f;
        const float3 correction = gravity * (-(1.0f - weight) * dt * dt);
        avbdBodies[i].displacement.values[0] = correction.x;
        avbdBodies[i].displacement.values[1] = correction.y;
        avbdBodies[i].displacement.values[2] = correction.z;
        avbdBodies[i].previous_velocity = previousStates[i].linear_velocity;
        avbdBodies[i].history_valid = 1u;
        const float mass = 1.0f / max(rigidParameters[i].inverse_mass, 1.0e-12f);
        for (uint axis = 0u; axis < 3u; ++axis) {
            avbdBodies[i].inertia.h[axis][axis] = mass / (dt * dt);
            const float3 local_axis = pm_avbd_rotate(pm_avbd_quaternion_conjugate(rigidStates[i].orientation), pm_avbd_axis(axis));
            const float3 column = pm_avbd_rotate(rigidStates[i].orientation,
                local_axis / max(pm_avbd_load(rigidParameters[i].inverse_inertia), float3(1.0e-12f)));
            for (uint row = axis; row < 3u; ++row)
                avbdBodies[i].inertia.h[3u + row][3u + axis] = column[row] / (dt * dt);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint active = lane; active < pairCountBuffer[0]; active += group_size.x) {
        MetalAvbdContactManifold m = contactManifolds[active];
        const uint a = activePairs[active] / rigidStep.body_count, b = activePairs[active] % rigidStep.body_count;
        const float penalty = pm_avbd_pair_stiffness(rigidParameters[a], rigidParameters[b], dt);
        const float skin = pm_avbd_contact_skin(
            rigidStep.body_count, rigidStep.constraint_capacity, a, b);
        for (uint point = 0u; point < m.count; ++point) {
            MetalAvbdContactRecord c = m.contacts[point];
            const float3 normal = pm_avbd_load(c.normal), position = pm_avbd_load(c.point);
            if (c.avbd_cached == 0u) {
                c.avbd_anchor_a = pm_avbd_store(pm_avbd_rotate(pm_avbd_quaternion_conjugate(previousStates[a].orientation), position - pm_avbd_load(previousStates[a].position)));
                c.avbd_anchor_b = pm_avbd_store(pm_avbd_rotate(pm_avbd_quaternion_conjugate(previousStates[b].orientation), position - pm_avbd_load(previousStates[b].position)));
            }
            const float3 motion_a = pm_avbd_load(rigidStates[a].position) - pm_avbd_load(previousStates[a].position);
            const float3 motion_b = pm_avbd_load(rigidStates[b].position) - pm_avbd_load(previousStates[b].position);
            const float3 wa = pm_avbd_quaternion_delta_velocity(previousStates[a].orientation, rigidStates[a].orientation, dt) * dt;
            const float3 wb = pm_avbd_quaternion_delta_velocity(previousStates[b].orientation, rigidStates[b].orientation, dt) * dt;
            const float3 ra = position - pm_avbd_load(rigidStates[a].position), rb = position - pm_avbd_load(rigidStates[b].position);
            const float closing = dot(normal, motion_b + cross(wb, rb) - motion_a - cross(wa, ra));
            const float old_penetration = c.penetration - closing;
            float3 t0 = float3(0.0f);
            float3 t1 = float3(0.0f);
            pm_avbd_basis(normal, t0, t1);
            const float3 anchor_delta = pm_avbd_load(rigidStates[b].position)
                + pm_avbd_rotate(rigidStates[b].orientation, pm_avbd_load(c.avbd_anchor_b))
                - pm_avbd_load(rigidStates[a].position) - pm_avbd_rotate(rigidStates[a].orientation, pm_avbd_load(c.avbd_anchor_a));
            const float3 old_anchor_delta = pm_avbd_load(previousStates[b].position)
                + pm_avbd_rotate(previousStates[b].orientation, pm_avbd_load(c.avbd_anchor_b))
                - pm_avbd_load(previousStates[a].position) - pm_avbd_rotate(previousStates[a].orientation, pm_avbd_load(c.avbd_anchor_a));
            c.initial_normal_speed = -closing / dt;
            c.avbd_error = {c.penetration - clamp(old_penetration, 0.0f, skin),
                dot(anchor_delta - kAvbdAlpha * old_anchor_delta, t0),
                dot(anchor_delta - kAvbdAlpha * old_anchor_delta, t1)};
            for (uint axis = 0u; axis < 3u; ++axis) {
                if (c.avbd_cached == 0u) c.avbd_dual[axis] = {0, penalty};
                AvbdDual dual = {
                    c.avbd_dual[axis].lambda, c.avbd_dual[axis].penalty};
                dual = avbdWarmStart(dual, kAvbdMaximumPenalty);
                c.avbd_dual[axis] = {dual.lambda, dual.penalty};
            }
            m.contacts[point] = c;
        }
        if (active < pm_avbd_cached_contacts)
            contact_cache[active] = {a, b, m.avbd_next_a, m.avbd_next_b,
                (m.count != 0u ? 1u : 0u) |
                (pm_avbd_prepare_contact_impact(m, skin, dt) ? 2u : 0u)};
        contactManifolds[active] = m;
    }
    for (uint index = lane; index < rigidStep.constraint_capacity; index += group_size.x) {
        MetalAvbdConstraintResource j = rigidConstraints[index];
        j.avbd_count = 0u;
        if (j.alive == 0u || j.enabled == 0u || j.broken != 0u)
        {
            rigidConstraints[index] = j;
            continue;
        }
        j.applied_impulse = 0;
        if (j.body_a >= rigidStep.body_count || j.body_b >= rigidStep.body_count)
        {
            rigidConstraints[index] = j;
            continue;
        }
        AvbdBlock scratch = avbdZeroBlock();
        pm_avbd_visit_joint(
            j, dt, scratch, true, pm_avbd_initialize_joint);
        rigidConstraints[index] = j;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint index = lane; index < min(rigidStep.constraint_capacity, uint(pm_avbd_cached_joints)); index += group_size.x)
        joint_cache[index] = pm_avbd_graph_joint_device(index);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
    scheduleBuffer[0] = {};
    scheduleBuffer[0].budget = rigidStep.rigid_contact_pass_limit != 0u ? rigidStep.rigid_contact_pass_limit : kAvbdDefaultIterations;
    scheduleBuffer[0].candidates = pairCountBuffer[0];
    uint event_cursor = 0u;
    for (uint active = 0u; active < pairCountBuffer[0]; ++active) {
        PMAvbdGraphContact contact = pm_avbd_graph_contact(active);
        if ((contact.flags & 1u) == 0u) continue;
        ++scheduleBuffer[0].contacts;
        const uint a = contact.a, b = contact.b;
        pm_avbd_graph_union(a, b);
        PMAvbdGraphBody body_a = pm_avbd_graph_body(a);
        contact.next_a = body_a.contact_head; body_a.contact_head = active;
        pm_avbd_graph_store_body(a, body_a);
        PMAvbdGraphBody body_b = pm_avbd_graph_body(b);
        contact.next_b = body_b.contact_head; body_b.contact_head = active;
        pm_avbd_graph_store_body(b, body_b);
        pm_avbd_graph_store_contact(active, contact);
        MetalAvbdContactManifold m = contactManifolds[active];
        m.event_offset = event_cursor;
        if (rigidStep.collect_rigid_contacts != 0u) {
            for (uint point = 0u; point < m.count; ++point) {
                if (event_cursor + point >= rigidStep.rigid_event_capacity) break;
                const MetalAvbdContactRecord c = m.contacts[point];
                rigidEvents[event_cursor + point] = {rigidIds[a], rigidIds[b], c.point, c.normal, max(c.penetration, 0.0f), 0, {}};
            }
            event_cursor += m.count;
        }
        contactManifolds[active] = m;
    }
    if (event_cursor != 0u) eventCountBuffer[0] = min(event_cursor, rigidStep.rigid_event_capacity);
    for (uint index = 0u; index < rigidStep.constraint_capacity; ++index) {
        PMAvbdGraphJoint joint = pm_avbd_graph_joint(index);
        if (joint.count == 0u) continue;
        if (rigidStep.rigid_contact_pass_limit == 0u)
            scheduleBuffer[0].budget = max(scheduleBuffer[0].budget, joint.iterations);
        pm_avbd_graph_union(joint.a, joint.b);
        PMAvbdGraphBody body_a = pm_avbd_graph_body(joint.a);
        joint.next_a = body_a.joint_head; body_a.joint_head = index;
        pm_avbd_graph_store_body(joint.a, body_a);
        PMAvbdGraphBody body_b = pm_avbd_graph_body(joint.b);
        joint.next_b = body_b.joint_head; body_b.joint_head = index;
        pm_avbd_graph_store_body(joint.b, body_b);
        pm_avbd_graph_store_joint(index, joint);
    }
    for (uint body = 0u; body < rigidStep.body_count; ++body) {
        PMAvbdGraphBody value = pm_avbd_graph_body(body);
        value.parent = pm_avbd_graph_root(body);
        pm_avbd_graph_store_body(body, value);
    }
    for (uint active = 0u; active < pairCountBuffer[0]; ++active) {
        const PMAvbdGraphContact contact = pm_avbd_graph_contact(active);
        const uint a = contact.a, b = contact.b;
        const bool impact = active < pm_avbd_cached_contacts
            ? (contact.flags & 2u) != 0u
            : pm_avbd_prepare_contact_impact(contactManifolds[active],
                pm_avbd_contact_skin(
                    rigidStep.body_count, rigidStep.constraint_capacity, a, b),
                dt);
        if (impact) {
            const PMAvbdGraphBody body_a = pm_avbd_graph_body(a);
            if ((body_a.flags & 1u) != 0u) {
                PMAvbdGraphBody root = pm_avbd_graph_body(body_a.parent);
                root.flags |= 2u;
                pm_avbd_graph_store_body(body_a.parent, root);
            }
            const PMAvbdGraphBody body_b = pm_avbd_graph_body(b);
            if ((body_b.flags & 1u) != 0u) {
                PMAvbdGraphBody root = pm_avbd_graph_body(body_b.parent);
                root.flags |= 2u;
                pm_avbd_graph_store_body(body_b.parent, root);
            }
            scheduleBuffer[0].reserved[0] = 1u;
        }
    }
    for (uint body = 0u; body < rigidStep.body_count; ++body) {
        PMAvbdGraphBody value = pm_avbd_graph_body(body);
        if ((value.flags & 1u) == 0u) continue;
        value.flags = (value.flags & 1u) | (pm_avbd_graph_body(value.parent).flags & 2u);
        pm_avbd_graph_store_body(body, value);
        if (value.parent == body && (value.contact_head != 0xffffffffu || value.joint_head != 0xffffffffu))
            ++scheduleBuffer[0].islands;
    }
    for (uint body = 0u; body < rigidStep.body_count; ++body) {
        PMAvbdGraphBody value = pm_avbd_graph_body(body);
        if ((value.flags & 1u) == 0u) continue;
        uint color = 0u;
        for (;; ++color) {
            bool conflict = false;
            for (uint active = value.contact_head; active != 0xffffffffu;) {
                const PMAvbdGraphContact contact = pm_avbd_graph_contact(active);
                const bool first = body == contact.a;
                conflict = conflict || pm_avbd_graph_body(first ? contact.b : contact.a).color == color;
                active = first ? contact.next_a : contact.next_b;
            }
            for (uint index = value.joint_head; index != 0xffffffffu;) {
                const PMAvbdGraphJoint joint = pm_avbd_graph_joint(index);
                const bool first = body == joint.a;
                conflict = conflict || pm_avbd_graph_body(first ? joint.b : joint.a).color == color;
                index = first ? joint.next_a : joint.next_b;
            }
            if (!conflict) break;
        }
        value.color = color;
        pm_avbd_graph_store_body(body, value);
        scheduleBuffer[0].colors = max(scheduleBuffer[0].colors, color + 1u);
    }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint body = lane; body < min(rigidStep.body_count, uint(pm_avbd_cached_bodies)); body += group_size.x)
        pm_avbd_graph_store_body_device(body, body_cache[body]);
    for (uint active = lane; active < min(pairCountBuffer[0], uint(pm_avbd_cached_contacts)); active += group_size.x)
        pm_avbd_graph_store_contact_device(active, contact_cache[active]);
    for (uint index = lane; index < min(rigidStep.constraint_capacity, uint(pm_avbd_cached_joints)); index += group_size.x)
        pm_avbd_graph_store_joint_device(index, joint_cache[index]);
}

static void pm_avbd_minimize_free_translation(
    uint count, bool velocity_phase, uint lane, uint stride) {
    // Common translation is an exact null mode of internal constraints.
    // Minimize the same inertial energy along it, preserving COM despite
    // finite Gauss-Seidel sweeps. Static/kinematic endpoints remain anchored.
    for (uint root = lane; root < count; root += stride) {
        if (rigidParameters[root].motion != 2u || avbdBodies[root].parent != root) continue;
        for (uint axis = 0u; axis < 3u; ++axis) avbdBodies[root].inertia.g[axis] = 0;
        if ((velocity_phase && avbdBodies[root].impact == 0u)
            || (avbdBodies[root].contact_head == 0xffffffffu && avbdBodies[root].joint_head == 0xffffffffu)) continue;
        float3 weighted = float3(0);
        float total_mass = 0;
        bool anchored = false;
        for (uint body = 0u; body < count && !anchored; ++body) {
            if (rigidParameters[body].motion != 2u || avbdBodies[body].parent != root) continue;
            const float mass = 1.0f / max(rigidParameters[body].inverse_mass, 1.0e-12f);
            weighted += pm_avbd_translation(avbdBodies[body].displacement) * mass;
            total_mass += mass;
            for (uint active = avbdBodies[body].contact_head; active != 0xffffffffu;) {
                MetalAvbdContactManifold m = contactManifolds[active];
                const uint a = activePairs[active] / count, b = activePairs[active] % count;
                const bool first = body == a;
                anchored = anchored || rigidParameters[first ? b : a].motion != 2u;
                active = first ? m.avbd_next_a : m.avbd_next_b;
            }
            for (uint index = avbdBodies[body].joint_head; index != 0xffffffffu;) {
                MetalAvbdConstraintResource j = rigidConstraints[index];
                const bool first = body == j.body_a;
                // Preserve reactions from a joint broken during this phase.
                anchored = anchored || rigidParameters[first ? j.body_b : j.body_a].motion != 2u;
                index = first ? j.avbd_next_a : j.avbd_next_b;
            }
        }
        if (!anchored && total_mass > 0.0f) {
            weighted /= total_mass;
            for (uint axis = 0u; axis < 3u; ++axis) avbdBodies[root].inertia.g[axis] = weighted[axis];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint body = lane; body < count; body += stride) {
        if (rigidParameters[body].motion != 2u || (velocity_phase && avbdBodies[body].impact == 0u)) continue;
        AvbdBlock mean = avbdBodies[avbdBodies[body].parent].inertia;
        for (uint axis = 0u; axis < 3u; ++axis) avbdBodies[body].displacement.values[axis] -= mean.g[axis];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint body = lane; body < count; body += stride)
        for (uint axis = 0u; axis < 3u; ++axis) avbdBodies[body].inertia.g[axis] = 0;
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pm_avbd_solve(uint3 groupThreadId : SV_GroupThreadID) {
    uint lane = groupThreadId.x;
    uint3 group_size = uint3(256u, 1u, 1u);
    const float dt = rigidStep.timestep;
    // Both pose and impact-velocity correction use the same AVBD body block,
    // graph colors, cone bounds and dual update. No legacy impulse solver.
    for (uint phase = 0u; phase < 2u; ++phase) {
    if (phase != 0u) {
        if (scheduleBuffer[0].reserved[0] == 0u) break;
        for (uint active = lane; active < pairCountBuffer[0]; active += group_size.x) {
            MetalAvbdContactManifold m = contactManifolds[active];
            const uint a = activePairs[active] / rigidStep.body_count, b = activePairs[active] % rigidStep.body_count;
            if (avbdBodies[a].impact == 0u && avbdBodies[b].impact == 0u) continue;
            for (uint point = 0u; point < m.count; ++point) {
                MetalAvbdContactRecord c = m.contacts[point];
                AvbdVector6 ja = avbdZeroVector6();
                AvbdVector6 jb = avbdZeroVector6();
                pm_avbd_contact_normal_row(c, a, b, ja, jb);
                const float gap = max(0.0f, -(c.penetration + avbdDot6(ja, avbdBodies[a].displacement)
                    + avbdDot6(jb, avbdBodies[b].displacement)));
                c.avbd_error.x = gap;
                m.contacts[point] = c;
            }
            contactManifolds[active] = m;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint body = lane; body < rigidStep.body_count; body += group_size.x) {
            if (avbdBodies[body].impact != 0u || rigidParameters[body].motion != 2u) {
                avbdBodies[body].target = rigidStates[body]; avbdBodies[body].displacement = {};
                avbdBodies[body].current_orientation = rigidStates[body].orientation;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint index = lane; index < rigidStep.constraint_capacity; index += group_size.x) {
            MetalAvbdConstraintResource j = rigidConstraints[index];
            if (j.avbd_count == 0u || j.enabled == 0u || j.broken != 0u) continue;
            if (avbdBodies[j.body_a].impact == 0u && avbdBodies[j.body_b].impact == 0u) continue;
            AvbdBlock scratch = avbdZeroBlock();
            pm_avbd_visit_joint(j, dt, scratch, true, pm_avbd_impact_joint);
            rigidConstraints[index] = j;
        }
        for (uint active = lane; active < pairCountBuffer[0]; active += group_size.x) {
            MetalAvbdContactManifold m = contactManifolds[active];
            const uint a = activePairs[active] / rigidStep.body_count, b = activePairs[active] % rigidStep.body_count;
            if (avbdBodies[a].impact == 0u && avbdBodies[b].impact == 0u) continue;
            const AvbdVector6 va = pm_avbd_jacobian(pm_avbd_load(rigidStates[a].linear_velocity) * dt, pm_avbd_load(rigidStates[a].angular_velocity) * dt);
            const AvbdVector6 vb = pm_avbd_jacobian(pm_avbd_load(rigidStates[b].linear_velocity) * dt, pm_avbd_load(rigidStates[b].angular_velocity) * dt);
            for (uint point = 0u; point < m.count; ++point) {
                MetalAvbdContactRecord c = m.contacts[point];
                const float gap = c.avbd_error.x;
                AvbdVector6 ja[3];
                AvbdVector6 jb[3];
                float error[3];
                pm_avbd_contact_rows(c, a, b, ja, jb, error);
                const float restitution = c.avbd_cached == 0u && c.initial_normal_speed < -1.0f
                    ? min(rigidParameters[a].restitution, rigidParameters[b].restitution) : 0.0f;
                c.avbd_error = {
                    avbdDot6(ja[0], va) + avbdDot6(jb[0], vb) - gap - restitution * c.initial_normal_speed * dt,
                    avbdDot6(ja[1], va) + avbdDot6(jb[1], vb),
                    avbdDot6(ja[2], va) + avbdDot6(jb[2], vb)};
                m.contacts[point] = c;
            }
            contactManifolds[active] = m;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint iteration = 0u; iteration < scheduleBuffer[0].budget; ++iteration) {
        for (uint color = 0u; color < scheduleBuffer[0].colors; ++color) {
            for (uint body = lane; body < rigidStep.body_count; body += group_size.x) {
                if (avbdBodies[body].color != color) continue;
                if (phase != 0u && avbdBodies[body].impact == 0u) continue;
                AvbdBlock block = avbdBodies[body].inertia;
                const AvbdVector6 displacement = avbdBodies[body].displacement;
                for (uint i = 0u; i < 6u; ++i)
                    for (uint j = 0u; j < 6u; ++j)
                        block.g[i] += block.h[max(i, j)][min(i, j)] * displacement.values[j];
                for (uint active = avbdBodies[body].contact_head; active != 0xffffffffu;) {
                    MetalAvbdContactManifold m = contactManifolds[active];
                    const uint a = activePairs[active] / rigidStep.body_count, b = activePairs[active] % rigidStep.body_count;
                    const bool first = body == a;
                    const float friction = sqrt(rigidParameters[a].friction * rigidParameters[b].friction);
                    for (uint point = 0u; point < m.count; ++point) {
                        const MetalAvbdContactRecord c = m.contacts[point];
                        AvbdVector6 ja[3];
                        AvbdVector6 jb[3];
                        float error[3];
                        pm_avbd_contact_rows(c, a, b, ja, jb, error);
                        const AvbdContactForce trial = pm_avbd_contact_trial(c, error);
                        const AvbdContactForce force = avbdProjectContact(trial, friction);
                        const AvbdContactForce scales = avbdContactStiffnessScales(trial, friction);
                        const float stiffness_scales[3] = {scales.normal, scales.tangent0, scales.tangent1};
                        float3 t0 = float3(0.0f);
                        float3 t1 = float3(0.0f);
                        pm_avbd_basis(pm_avbd_load(c.normal), t0, t1);
                        const float f[3] = {
                            force.normal - (phase != 0u ? c.accumulated_normal_impulse / dt : 0.0f),
                            force.tangent0 - (phase != 0u ? dot(pm_avbd_load(c.accumulated_friction_impulse), t0) / dt : 0.0f),
                            force.tangent1 - (phase != 0u ? dot(pm_avbd_load(c.accumulated_friction_impulse), t1) / dt : 0.0f)};
                        for (uint axis = 0u; axis < 3u; ++axis)
                            avbdAddRow(block, first ? ja[axis] : jb[axis],
                                c.avbd_dual[axis].penalty * stiffness_scales[axis], f[axis]);
                    }
                    active = first ? m.avbd_next_a : m.avbd_next_b;
                }
                for (uint index = avbdBodies[body].joint_head; index != 0xffffffffu;) {
                    MetalAvbdConstraintResource j = rigidConstraints[index];
                    const bool first = body == j.body_a;
                    if (j.enabled != 0u && j.broken == 0u) {
                        if (phase == 0u) {
                            pm_avbd_visit_joint(j, dt, block, first,
                                pm_avbd_accumulate_joint);
                            rigidConstraints[index] = j;
                        } else {
                            for (uint row = 0u; row < j.avbd_count; ++row)
                                if (j.avbd_rows[row].stiffness >= kAvbdMaximumPenalty)
                                    avbdAccumulate(block, j.avbd_rows[row], first,
                                        avbdBodies[j.body_a].displacement, avbdBodies[j.body_b].displacement, dt);
                        }
                    }
                    index = first ? j.avbd_next_a : j.avbd_next_b;
                }
                AvbdVector6 update = avbdZeroVector6();
                if (avbdSolve(block, update)) {
                    {
                        // Re-solve crossed corners on their predicted active
                        // side. Include affine force and Hessian together:
                        // Hessian-only damping creates artificial pose gaps.
                        // Separating rows remain completely inactive.
                        bool crossing = false;
                        for (uint active = avbdBodies[body].contact_head; active != 0xffffffffu;) {
                            MetalAvbdContactManifold m = contactManifolds[active];
                            const uint a = activePairs[active] / rigidStep.body_count, b = activePairs[active] % rigidStep.body_count;
                            const bool first = body == a;
                            for (uint point = 0u; point < m.count; ++point) {
                                const MetalAvbdContactRecord c = m.contacts[point];
                                AvbdVector6 ja = avbdZeroVector6();
                                AvbdVector6 jb = avbdZeroVector6();
                                const float error =
                                    pm_avbd_contact_normal_row(c, a, b, ja, jb);
                                const float trial = c.avbd_dual[0].lambda + c.avbd_dual[0].penalty * error;
                                const AvbdVector6 normal_row = first ? ja : jb;
                                // Ignore roundoff-sized boundary crossings.
                                if (trial <= 0.0f && trial + c.avbd_dual[0].penalty * avbdDot6(normal_row, update)
                                    > c.avbd_dual[0].penalty * 1.0e-6f) {
                                    avbdAddRow(block, normal_row, c.avbd_dual[0].penalty, trial);
                                    crossing = true;
                                }
                            }
                            active = first ? m.avbd_next_a : m.avbd_next_b;
                        }
                        if (crossing) {
                            AvbdVector6 safeguarded = avbdZeroVector6();
                            if (avbdSolve(block, safeguarded)) update = safeguarded;
                        }
                    }
                    if (phase == 0u) pm_avbd_apply_pose_update(avbdBodies[body], update);
                    else for (uint i = 0u; i < 6u; ++i) avbdBodies[body].displacement.values[i] += update.values[i];
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        for (uint active = lane; active < pairCountBuffer[0]; active += group_size.x) {
            MetalAvbdContactManifold m = contactManifolds[active];
            const uint a = activePairs[active] / rigidStep.body_count, b = activePairs[active] % rigidStep.body_count;
            if (phase != 0u && avbdBodies[a].impact == 0u && avbdBodies[b].impact == 0u) continue;
            const float friction = sqrt(rigidParameters[a].friction * rigidParameters[b].friction);
            const float beta = pm_avbd_pair_stiffness(rigidParameters[a], rigidParameters[b], dt) * 100.0f;
            for (uint point = 0u; point < m.count; ++point) {
                MetalAvbdContactRecord c = m.contacts[point];
                AvbdVector6 ja[3];
                AvbdVector6 jb[3];
                float error[3];
                pm_avbd_contact_rows(c, a, b, ja, jb, error);
                const AvbdContactForce force = pm_avbd_contact_force(c, error, friction);
                const float tangential = length(float2(c.avbd_dual[1].lambda + c.avbd_dual[1].penalty * error[1],
                    c.avbd_dual[2].lambda + c.avbd_dual[2].penalty * error[2]));
                const bool stick = tangential <= friction * force.normal;
                c.avbd_dual[0].lambda = force.normal;
                c.avbd_dual[1].lambda = force.tangent0;
                c.avbd_dual[2].lambda = force.tangent1;
                for (uint axis = 0u; axis < 3u; ++axis)
                    if (axis == 0u ? force.normal > 0.0f : stick)
                        c.avbd_dual[axis].penalty = min(kAvbdMaximumPenalty,
                            c.avbd_dual[axis].penalty + beta * abs(error[axis]));
                if (iteration + 1u == scheduleBuffer[0].budget) {
                    float3 t0 = float3(0.0f);
                    float3 t1 = float3(0.0f);
                    pm_avbd_basis(pm_avbd_load(c.normal), t0, t1);
                    c.accumulated_normal_impulse = force.normal * dt;
                    c.accumulated_friction_impulse = pm_avbd_store((t0 * force.tangent0 + t1 * force.tangent1) * dt);
                    const uint event = m.event_offset + point;
                    if (rigidStep.collect_rigid_contacts != 0u && event < rigidStep.rigid_event_capacity) {
                        rigidEvents[event].normal_impulse = c.accumulated_normal_impulse;
                        rigidEvents[event].friction_impulse = c.accumulated_friction_impulse;
                    }
                    if (!stick) {
                        MetalAvbdRigidState pose_a = avbdBodies[a].target, pose_b = avbdBodies[b].target;
                        if (phase == 0u) {
                            pose_a = pm_avbd_pose(avbdBodies[a]); pose_b = pm_avbd_pose(avbdBodies[b]);
                        }
                        c.avbd_anchor_a = pm_avbd_store(pm_avbd_rotate(pm_avbd_quaternion_conjugate(pose_a.orientation), pm_avbd_load(c.point) - pm_avbd_load(pose_a.position)));
                        c.avbd_anchor_b = pm_avbd_store(pm_avbd_rotate(pm_avbd_quaternion_conjugate(pose_b.orientation), pm_avbd_load(c.point) - pm_avbd_load(pose_b.position)));
                    }
                }
                m.contacts[point] = c;
            }
            contactManifolds[active] = m;
        }
        for (uint index = lane; index < rigidStep.constraint_capacity; index += group_size.x) {
            MetalAvbdConstraintResource j = rigidConstraints[index];
            if (j.avbd_count == 0u || j.enabled == 0u || j.broken != 0u) continue;
            if (phase != 0u && avbdBodies[j.body_a].impact == 0u && avbdBodies[j.body_b].impact == 0u) continue;
            float impulse = 0;
            if (phase == 0u) {
                AvbdBlock scratch = avbdZeroBlock();
                impulse = pm_avbd_visit_joint(
                    j, dt, scratch, true, pm_avbd_advance_joint);
            } else {
                for (uint row = 0u; row < j.avbd_count; ++row) {
                    if (j.avbd_rows[row].stiffness < kAvbdMaximumPenalty) {
                        impulse += abs(j.avbd_rows[row].referenceForce) * dt;
                        continue;
                    }
                    impulse += abs(avbdRowForce(j.avbd_rows[row], avbdBodies[j.body_a].displacement,
                        avbdBodies[j.body_b].displacement, dt)) * dt;
                    j.avbd_rows[row] = avbdAdvance(j.avbd_rows[row], avbdBodies[j.body_a].displacement, avbdBodies[j.body_b].displacement);
                }
            }
            j.applied_impulse = impulse;
            if (j.breaking_impulse_threshold > 0.0f && impulse > j.breaking_impulse_threshold) {
                j.broken = 1u; j.enabled = 0u;
            }
            rigidConstraints[index] = j;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    pm_avbd_minimize_free_translation(
        rigidStep.body_count, phase != 0u, lane, group_size.x);
    for (uint body = lane; body < rigidStep.body_count; body += group_size.x) {
        if (rigidParameters[body].motion != 2u) continue;
        if (phase != 0u && avbdBodies[body].impact == 0u) continue;
        MetalAvbdRigidState state = avbdBodies[body].target;
        const AvbdVector6 delta = avbdBodies[body].displacement;
        if (phase != 0u) {
            state.linear_velocity = pm_avbd_store(pm_avbd_limit(pm_avbd_load(state.linear_velocity) + pm_avbd_translation(delta) / dt,
                rigidParameters[body].maximum_linear_speed));
            state.angular_velocity = pm_avbd_store(pm_avbd_limit(pm_avbd_load(state.angular_velocity) + pm_avbd_rotation(delta) / dt,
                rigidParameters[body].maximum_angular_speed));
            rigidStates[body] = state;
            continue;
        }
        state.position = pm_avbd_store(pm_avbd_load(state.position) + pm_avbd_translation(delta));
        state.orientation = avbdBodies[body].current_orientation;
        state.linear_velocity = pm_avbd_store(pm_avbd_load(avbdBodies[body].target.linear_velocity) + pm_avbd_translation(delta) / dt);
        state.angular_velocity = pm_avbd_store(pm_avbd_load(avbdBodies[body].target.angular_velocity)
            + pm_avbd_quaternion_delta_velocity(avbdBodies[body].target.orientation, state.orientation, dt));
        if (avbdBodies[body].impact == 0u) {
            state.linear_velocity = pm_avbd_store(pm_avbd_limit(pm_avbd_load(state.linear_velocity), rigidParameters[body].maximum_linear_speed));
            state.angular_velocity = pm_avbd_store(pm_avbd_limit(pm_avbd_load(state.angular_velocity), rigidParameters[body].maximum_angular_speed));
        }
        rigidStates[body] = state;
    }
    if (lane == 0u) scheduleBuffer[0].maximum_passes += scheduleBuffer[0].budget;
    GroupMemoryBarrierWithGroupSync();
    }
}
