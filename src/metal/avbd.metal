// SPDX-License-Identifier: MIT
// Metal geometry/storage adapter for the shared AVBD numerical contract.
// Included after kernels.metal's packed ABI types and collision helpers.
namespace avbd = parallel_mater::avbd;

// Cache only graph metadata, not solver state. This preserves the serial
// body/pair traversal and color assignment while avoiding dependent reads
// through large device-memory records. Prefix overflow uses the same graph.
// 22,016 bytes fits the Metal 32 KiB threadgroup-memory budget; CUDA uses a
// larger prefix with identical metadata and ordering.
enum { pm_avbd_cached_bodies = 512, pm_avbd_cached_contacts = 512,
       pm_avbd_cached_joints = 64 };
struct PMAvbdGraphBody {
    uint parent, color, contact_head, joint_head, flags;
};
struct PMAvbdGraphContact {
    uint a, b, next_a, next_b, flags;
};
struct PMAvbdGraphJoint {
    uint a, b, next_a, next_b, count, iterations;
};
static_assert(sizeof(PMAvbdGraphBody) * pm_avbd_cached_bodies +
              sizeof(PMAvbdGraphContact) * pm_avbd_cached_contacts +
              sizeof(PMAvbdGraphJoint) * pm_avbd_cached_joints == 22016);
struct PMAvbdGraphCache {
    device PMAvbdBody *bodies;
    device const PMRigidParameters *parameters;
    device PMContactManifold *manifolds;
    device const uint *pairs;
    device PMRigidConstraintResource *joints;
    threadgroup PMAvbdGraphBody *body_cache;
    threadgroup PMAvbdGraphContact *contact_cache;
    threadgroup PMAvbdGraphJoint *joint_cache;
    uint body_count;
};
static PMAvbdGraphBody pm_avbd_graph_body_device(
    thread const PMAvbdGraphCache &graph, uint body) {
    device const PMAvbdBody &b = graph.bodies[body];
    return {b.parent, b.color, b.contact_head, b.joint_head,
        (graph.parameters[body].motion == 2u ? 1u : 0u) | (b.impact != 0u ? 2u : 0u)};
}
static PMAvbdGraphBody pm_avbd_graph_body(
    thread const PMAvbdGraphCache &graph, uint body) {
    if (body < pm_avbd_cached_bodies) return graph.body_cache[body];
    return pm_avbd_graph_body_device(graph, body);
}
static void pm_avbd_graph_store_body_device(
    thread const PMAvbdGraphCache &graph, uint body, PMAvbdGraphBody value) {
    device PMAvbdBody &b = graph.bodies[body];
    b.parent = value.parent; b.color = value.color;
    b.contact_head = value.contact_head; b.joint_head = value.joint_head;
    b.impact = (value.flags & 2u) != 0u ? 1u : 0u;
}
static void pm_avbd_graph_store_body(
    thread const PMAvbdGraphCache &graph, uint body, PMAvbdGraphBody value) {
    if (body < pm_avbd_cached_bodies) graph.body_cache[body] = value;
    else pm_avbd_graph_store_body_device(graph, body, value);
}
static PMAvbdGraphContact pm_avbd_graph_contact_device(
    thread const PMAvbdGraphCache &graph, uint index) {
    device const PMContactManifold &m = graph.manifolds[index];
    return {graph.pairs[index] / graph.body_count, graph.pairs[index] % graph.body_count,
        m.avbd_next_a, m.avbd_next_b, m.count != 0u ? 1u : 0u};
}
static PMAvbdGraphContact pm_avbd_graph_contact(
    thread const PMAvbdGraphCache &graph, uint index) {
    if (index < pm_avbd_cached_contacts) return graph.contact_cache[index];
    return pm_avbd_graph_contact_device(graph, index);
}
static void pm_avbd_graph_store_contact_device(
    thread const PMAvbdGraphCache &graph, uint index, PMAvbdGraphContact value) {
    graph.manifolds[index].avbd_next_a = value.next_a;
    graph.manifolds[index].avbd_next_b = value.next_b;
}
static void pm_avbd_graph_store_contact(
    thread const PMAvbdGraphCache &graph, uint index, PMAvbdGraphContact value) {
    if (index < pm_avbd_cached_contacts) graph.contact_cache[index] = value;
    else pm_avbd_graph_store_contact_device(graph, index, value);
}
static PMAvbdGraphJoint pm_avbd_graph_joint_device(
    thread const PMAvbdGraphCache &graph, uint index) {
    device const PMRigidConstraintResource &j = graph.joints[index];
    return {j.body_a, j.body_b, j.avbd_next_a, j.avbd_next_b, j.avbd_count, j.solver_iterations};
}
static PMAvbdGraphJoint pm_avbd_graph_joint(
    thread const PMAvbdGraphCache &graph, uint index) {
    if (index < pm_avbd_cached_joints) return graph.joint_cache[index];
    return pm_avbd_graph_joint_device(graph, index);
}
static void pm_avbd_graph_store_joint_device(
    thread const PMAvbdGraphCache &graph, uint index, PMAvbdGraphJoint value) {
    graph.joints[index].avbd_next_a = value.next_a;
    graph.joints[index].avbd_next_b = value.next_b;
}
static void pm_avbd_graph_store_joint(
    thread const PMAvbdGraphCache &graph, uint index, PMAvbdGraphJoint value) {
    if (index < pm_avbd_cached_joints) graph.joint_cache[index] = value;
    else pm_avbd_graph_store_joint_device(graph, index, value);
}
static uint pm_avbd_graph_root(thread const PMAvbdGraphCache &graph, uint body) {
    for (;;) {
        const uint parent = pm_avbd_graph_body(graph, body).parent;
        if (parent == body) return body;
        body = parent;
    }
}
static void pm_avbd_graph_union(thread const PMAvbdGraphCache &graph, uint a, uint b) {
    if ((pm_avbd_graph_body(graph, a).flags & 1u) == 0u ||
        (pm_avbd_graph_body(graph, b).flags & 1u) == 0u) return;
    a = pm_avbd_graph_root(graph, a); b = pm_avbd_graph_root(graph, b);
    if (a != b) {
        PMAvbdGraphBody high = pm_avbd_graph_body(graph, max(a, b));
        high.parent = min(a, b);
        pm_avbd_graph_store_body(graph, max(a, b), high);
    }
}

static avbd::Vector6 pm_avbd_jacobian(float3 linear, float3 angular) {
    return {{linear.x, linear.y, linear.z, angular.x, angular.y, angular.z}};
}
static float3 pm_avbd_translation(avbd::Vector6 value) {
    return float3(value.v[0], value.v[1], value.v[2]);
}
static float3 pm_avbd_rotation(avbd::Vector6 value) {
    return float3(value.v[3], value.v[4], value.v[5]);
}
static float3 pm_avbd_axis(uint axis) {
    return axis == 0u ? float3(1, 0, 0)
        : axis == 1u ? float3(0, 1, 0) : float3(0, 0, 1);
}
static PMRigidBodyState pm_avbd_pose(device const PMAvbdBody &body) {
    PMRigidBodyState state = body.target;
    state.position = pm_store(pm_load(state.position) + pm_avbd_translation(body.displacement));
    state.orientation = body.current_orientation;
    return state;
}
static void pm_avbd_apply_pose_update(device PMAvbdBody &body, avbd::Vector6 update) {
    for (uint i = 0u; i < 3u; ++i) body.displacement.v[i] += update.v[i];
    const float3 rotation = pm_avbd_rotation(update);
    const PMQuaternion dq = pm_quaternion_multiply({rotation.x, rotation.y, rotation.z, 0}, body.current_orientation);
    body.current_orientation = pm_quaternion_normalize({
        body.current_orientation.x + 0.5f * dq.x, body.current_orientation.y + 0.5f * dq.y,
        body.current_orientation.z + 0.5f * dq.z, body.current_orientation.w + 0.5f * dq.w});
    PMQuaternion relative = pm_quaternion_multiply(body.current_orientation, pm_quaternion_conjugate(body.target.orientation));
    if (relative.w < 0.0f) relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    body.displacement.v[3] = 2.0f * relative.x;
    body.displacement.v[4] = 2.0f * relative.y;
    body.displacement.v[5] = 2.0f * relative.z;
}
static float3 pm_avbd_log_axis(PMQuaternion frame, float3 logarithm, uint axis) {
    const float squared = dot(logarithm, logarithm);
    float coefficient = 1.0f / 12.0f + squared / 720.0f;
    if (squared > 1.0e-6f) {
        const float angle = sqrt(squared), half_angle = 0.5f * angle;
        coefficient = (1.0f - half_angle * cos(half_angle) / sin(half_angle)) / squared;
    }
    const float3 unit = pm_avbd_axis(axis);
    return pm_rotate(frame, unit + 0.5f * cross(logarithm, unit)
        + coefficient * cross(logarithm, cross(logarithm, unit)));
}
static void pm_avbd_basis(float3 normal, thread float3 &t0, thread float3 &t1) {
    t0 = normalize(cross(normal, abs(normal.x) < 0.57735f
        ? float3(1, 0, 0) : float3(0, 1, 0)));
    t1 = cross(normal, t0);
}
static float pm_avbd_pair_stiffness(PMRigidParameters a, PMRigidParameters b, float dt) {
    return 1.0f / (max(a.inverse_mass + b.inverse_mass, 1.0e-6f) * dt * dt);
}
static float pm_avbd_contact_skin(device const PMRigidParameters *parameters,
    device const PMRigidBodyState *states, device const PMRigidConstraintResource *joints,
    uint body_count, uint constraint_capacity, uint a, uint b) {
    bool guided = false;
    if (constraint_capacity != 0u && (parameters[a].motion == 0u || parameters[b].motion == 0u)) {
        float3 reference_a = float3(0), reference_b = float3(0);
        const PMHingeContactFrame guide_a = pm_rigid_hinge_contact_frame(
            parameters, body_count, a, states, joints, constraint_capacity, reference_a);
        const PMHingeContactFrame guide_b = pm_rigid_hinge_contact_frame(
            parameters, body_count, b, states, joints, constraint_capacity, reference_b);
        guided = pm_guided_static_contact(guide_a, guide_b);
    }
    // Match the sharper guided CCD stand-off, not an ordinary 1 mm skin.
    const float offset = guided
        ? min(parameters[a].collision_margin + parameters[b].collision_margin, 1.0e-4f)
        : pm_rigid_rest_offset(parameters[a].collision_margin + parameters[b].collision_margin);
    return offset + 2.0f * pm_rigid_surface_tolerance;
}
static bool pm_avbd_prepare_contact_impact(
    device const PMContactManifold &manifold, float skin, float dt) {
    bool impact = false;
    for (uint point = 0u; point < manifold.count; ++point) {
        device const PMContactRecord &c = manifold.contacts[point];
        const bool new_impact = (c.avbd_cached == 0u || c.avbd_dual[0].lambda <= 1.0e-6f
            || (c.impact_fraction > 0.0f && c.impact_fraction < 1.0f))
            && c.initial_normal_speed < -0.1f && c.penetration >= -pm_rigid_surface_tolerance;
        const float previous_penetration = c.penetration + c.initial_normal_speed * dt;
        impact |= new_impact || previous_penetration > skin + 1.0e-4f;
    }
    return impact;
}
static void pm_avbd_contact_rows(PMContactRecord contact, uint a, uint b,
    device const PMAvbdBody *bodies, thread avbd::Vector6 *ja,
    thread avbd::Vector6 *jb, thread float *error) {
    float3 t0{}, t1{};
    const float3 normal = pm_load(contact.normal);
    pm_avbd_basis(normal, t0, t1);
    const float3 axes[3] = {normal, t0, t1};
    const float3 arm_a = pm_load(contact.point) - pm_load(bodies[a].target.position);
    const float3 arm_b = pm_load(contact.point) - pm_load(bodies[b].target.position);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float3 ra = axis == 0u ? arm_a
            : pm_rotate(bodies[a].target.orientation, pm_load(contact.avbd_anchor_a));
        const float3 rb = axis == 0u ? arm_b
            : pm_rotate(bodies[b].target.orientation, pm_load(contact.avbd_anchor_b));
        ja[axis] = pm_avbd_jacobian(-axes[axis], -cross(ra, axes[axis]));
        jb[axis] = pm_avbd_jacobian(axes[axis], cross(rb, axes[axis]));
        error[axis] = pm_load(contact.avbd_error)[axis]
            + avbd::dot6(ja[axis], bodies[a].displacement)
            + avbd::dot6(jb[axis], bodies[b].displacement);
    }
}
static avbd::ContactForce pm_avbd_contact_trial(PMContactRecord contact,
    thread const float *error) {
    return {
        contact.avbd_dual[0].lambda + contact.avbd_dual[0].penalty * error[0],
        contact.avbd_dual[1].lambda + contact.avbd_dual[1].penalty * error[1],
        contact.avbd_dual[2].lambda + contact.avbd_dual[2].penalty * error[2]};
}
static float pm_avbd_contact_normal_row(PMContactRecord contact, uint a, uint b,
    device const PMAvbdBody *bodies, thread avbd::Vector6 &ja, thread avbd::Vector6 &jb) {
    const float3 normal = pm_load(contact.normal);
    const float3 arm_a = pm_load(contact.point) - pm_load(bodies[a].target.position);
    const float3 arm_b = pm_load(contact.point) - pm_load(bodies[b].target.position);
    ja = pm_avbd_jacobian(-normal, -cross(arm_a, normal));
    jb = pm_avbd_jacobian(normal, cross(arm_b, normal));
    return contact.avbd_error.x + avbd::dot6(ja, bodies[a].displacement)
        + avbd::dot6(jb, bodies[b].displacement);
}
static avbd::ContactForce pm_avbd_contact_force(PMContactRecord contact,
    thread const float *error, float friction) {
    return avbd::project_contact(pm_avbd_contact_trial(contact, error), friction);
}

enum PMAvbdJointVisit : uint {
    pm_avbd_initialize_joint, pm_avbd_accumulate_joint,
    pm_avbd_advance_joint, pm_avbd_impact_joint
};
// Authored limits retain the established 20% recovery per substep.
constant float pm_avbd_limit_stabilization = 0.8f;

template<PMAvbdJointVisit Visit>
static void pm_avbd_visit_joint_row(device PMRigidConstraintResource &joint,
    device const PMAvbdBody *bodies, thread avbd::Block &block,
    thread uint &index, thread float &impulse, bool first,
    avbd::Vector6 ja, avbd::Vector6 jb, float current, float previous,
    float penalty, float dt, bool angular, float stiffness = avbd::maximum_penalty,
    float damping = 0, float lo = -avbd::maximum_penalty, float hi = avbd::maximum_penalty,
    bool motor = false, float motor_speed = 0, float3 curvature_a = float3(0), float3 curvature_b = float3(0),
    float stabilization = avbd::alpha) {
    const uint row_index = index++;
    avbd::Row row = joint.avbd_rows[row_index];
    const uint a = joint.body_a, b = joint.body_b;
    const float linearized = avbd::dot6(ja, bodies[a].displacement) + avbd::dot6(jb, bodies[b].displacement);
    row.a = ja; row.b = jb; row.stiffness = stiffness; row.damping = damping;
    if (Visit == pm_avbd_initialize_joint) {
        if (!(row.dual.penalty > avbd::minimum_penalty)) row.dual.penalty = penalty;
        row.dual = avbd::warm_start(row.dual, stiffness);
        if (stiffness < avbd::maximum_penalty) row.dual.lambda = 0;
        row.beta = penalty * (angular ? 10.0f : 100.0f);
        row.reference_force = 0;
    }
    // The shared core adds J*displacement back, yielding exactly C(current)
    // with a fresh current-pose tangent Jacobian on every body visit.
    row.error = current - (stiffness < avbd::maximum_penalty ? 0.0f : stabilization * previous) - linearized;
    row.velocity = motor ? motor_speed : (current - previous - linearized) / dt;
    row.lower = lo; row.upper = hi;
    if (Visit == pm_avbd_accumulate_joint) {
        avbd::accumulate(block, row, first, bodies[a].displacement, bodies[b].displacement, dt);
        if (!angular) {
            const float3 axis = pm_avbd_translation(jb), arm = first ? curvature_a : curvature_b;
            const float force = avbd::row_force(row, bodies[a].displacement, bodies[b].displacement, dt);
            for (uint column = 0u; column < 3u; ++column) {
                const float3 value = 0.5f * (axis * arm[column] + arm * axis[column])
                    - pm_avbd_axis(column) * dot(axis, arm);
                block.h[3u + column][3u + column] += abs(force) * length(value);
            }
        }
    } else if (Visit == pm_avbd_advance_joint) {
        impulse += abs(avbd::row_force(row, bodies[a].displacement, bodies[b].displacement, dt)) * dt;
        joint.avbd_rows[row_index] = avbd::advance(row, bodies[a].displacement, bodies[b].displacement);
    } else if (Visit == pm_avbd_impact_joint) {
        if (stiffness < avbd::maximum_penalty) {
            row.reference_force = avbd::row_force(row, bodies[a].displacement, bodies[b].displacement, dt);
        } else {
            // Velocity limits use the actual gap, not the relaxed pose error.
            const float gap = lo == 0.0f ? min(0.0f, current) : hi == 0.0f ? max(0.0f, current) : 0.0f;
            const avbd::Vector6 va = pm_avbd_jacobian(pm_load(bodies[a].target.linear_velocity) * dt,
                pm_load(bodies[a].target.angular_velocity) * dt);
            const avbd::Vector6 vb = pm_avbd_jacobian(pm_load(bodies[b].target.linear_velocity) * dt,
                pm_load(bodies[b].target.angular_velocity) * dt);
            row.reference_force = row.dual.lambda;
            row.error = avbd::dot6(ja, va) + avbd::dot6(jb, vb) + gap - (motor ? motor_speed * dt : 0.0f);
        }
        joint.avbd_rows[row_index] = row;
    } else {
        joint.avbd_rows[row_index] = row;
    }
}

template<PMAvbdJointVisit Visit>
static float pm_avbd_visit_joint(device PMRigidConstraintResource &j,
    device const PMRigidParameters *parameters, device const PMRigidBodyState *previous,
    device const PMAvbdBody *bodies, float dt, thread avbd::Block &block, bool first = true) {
    uint index = 0u; float impulse = 0;
    const uint a = j.body_a, b = j.body_b;
    const PMRigidBodyState sa = pm_avbd_pose(bodies[a]), sb = pm_avbd_pose(bodies[b]);
    const float3 ra = pm_rotate(sa.orientation, pm_load(j.local_anchor_a));
    const float3 rb = pm_rotate(sb.orientation, pm_load(j.local_anchor_b));
    const float3 error = pm_load(sb.position) + rb - pm_load(sa.position) - ra;
    const float3 old_error = pm_load(previous[b].position)
        + pm_rotate(previous[b].orientation, pm_load(j.local_anchor_b))
        - pm_load(previous[a].position) - pm_rotate(previous[a].orientation, pm_load(j.local_anchor_a));
    const PMQuaternion fa = pm_quaternion_normalize(pm_quaternion_multiply(sa.orientation, j.local_orientation_a));
    const PMQuaternion fb = pm_quaternion_normalize(pm_quaternion_multiply(sb.orientation, j.local_orientation_b));
    const PMQuaternion old_fa = pm_quaternion_normalize(pm_quaternion_multiply(previous[a].orientation, j.local_orientation_a));
    const PMQuaternion old_fb = pm_quaternion_normalize(pm_quaternion_multiply(previous[b].orientation, j.local_orientation_b));
    const float3 twist = pm_relative_rotation_vector(fa, fb), old_twist = pm_relative_rotation_vector(old_fa, old_fb);
    const bool swing = j.type == 2u || j.type == 4u || (j.type == 7u && j.angular_motor_enabled != 0u);
    const float3 free_axis = j.type == 2u ? float3(0, 0, 1) : float3(1, 0, 0);
    const float3 swing_axis = pm_rotate(pm_quaternion_conjugate(fa), pm_rotate(fb, free_axis));
    const float3 old_swing_axis = pm_rotate(pm_quaternion_conjugate(old_fa), pm_rotate(old_fb, free_axis));
    const float3 rotation = swing ? cross(free_axis, swing_axis) : twist;
    const float3 old_rotation = swing ? cross(free_axis, old_swing_axis) : old_twist;
    const float linear_penalty = pm_avbd_pair_stiffness(parameters[a], parameters[b], dt);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const float3 n = pm_rotate(fa, pm_avbd_axis(axis)), old_n = pm_rotate(old_fa, pm_avbd_axis(axis));
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
                ? pm_rotate(fa, pm_avbd_axis(axis) * dot(free_axis, swing_axis) - free_axis * swing_axis[axis])
                : pm_avbd_log_axis(fa, twist, axis);
            const avbd::Vector6 ja = angular != 0u ? pm_avbd_jacobian(float3(0), -angular_axis)
                : pm_avbd_jacobian(-n, -cross(ra, n) + cross(n, error));
            const avbd::Vector6 jb = angular != 0u ? pm_avbd_jacobian(float3(0), angular_axis) : pm_avbd_jacobian(n, cross(rb, n));
            const float penalty = angular != 0u
                ? 1.0f / (max(dot(n, pm_inverse_inertia_mul_orientation(parameters[a], sa.orientation, n)
                    + pm_inverse_inertia_mul_orientation(parameters[b], sb.orientation, n)), 1.0e-6f) * dt * dt) : linear_penalty;
            // Parallel-axis floor conditions off-axis hard locks without
            // changing physical inertia, motors, or finite spring forces.
            // Factor two bounds the two coupled anchor contributions.
            const float3 lever_a = cross(n, ra), lever_b = cross(n, rb);
            const float lock_penalty = angular != 0u ? max(penalty,
                2.0f * linear_penalty * max(dot(lever_a, lever_a), dot(lever_b, lever_b))) : penalty;
            if (lock) {
                if (Visit == pm_avbd_initialize_joint && angular != 0u)
                    j.avbd_rows[index].dual.penalty = max(j.avbd_rows[index].dual.penalty, lock_penalty / avbd::gamma);
                pm_avbd_visit_joint_row<Visit>(j, bodies, block, index, impulse, first,
                    ja, jb, c, old_c, lock_penalty, dt, angular != 0u, avbd::maximum_penalty, 0,
                    -avbd::maximum_penalty, avbd::maximum_penalty, false, 0, error + ra, rb);
            }
            const bool free_limit = angular != 0u
                ? (j.type == 2u && axis == 2u) || (j.type == 4u && axis == 0u)
                : (j.type == 3u || j.type == 4u) && axis == 0u;
            const uint limit_axes = angular != 0u ? j.angular_limit_axes : j.linear_limit_axes;
            if ((generic_joint || free_limit) && (limit_axes & (1u << axis)) != 0u) {
                const float limit_c = angular != 0u ? twist[axis] : c;
                const float old_limit_c = angular != 0u ? old_twist[axis] : old_c;
                const float3 limit_axis = pm_avbd_log_axis(fa, twist, axis);
                avbd::Vector6 limit_ja = ja, limit_jb = jb;
                if (angular != 0u) {
                    limit_ja = pm_avbd_jacobian(float3(0), -limit_axis);
                    limit_jb = pm_avbd_jacobian(float3(0), limit_axis);
                }
                const float lower = pm_load(angular != 0u ? j.angular_limit_lower : j.linear_limit_lower)[axis];
                const float upper = pm_load(angular != 0u ? j.angular_limit_upper : j.linear_limit_upper)[axis];
                if (lower == upper) {
                    const float equal_penalty = angular != 0u && generic_joint ? lock_penalty : penalty;
                    if (Visit == pm_avbd_initialize_joint && angular != 0u && generic_joint)
                        j.avbd_rows[index].dual.penalty = max(j.avbd_rows[index].dual.penalty, equal_penalty / avbd::gamma);
                    pm_avbd_visit_joint_row<Visit>(j, bodies, block, index, impulse, first,
                        limit_ja, limit_jb, limit_c - lower, old_limit_c - lower, equal_penalty, dt, angular != 0u,
                        avbd::maximum_penalty, 0, -avbd::maximum_penalty, avbd::maximum_penalty, false, 0, error + ra, rb,
                        avbd::alpha);
                }
                else if (lower < upper) {
                    pm_avbd_visit_joint_row<Visit>(j, bodies, block, index, impulse, first,
                        limit_ja, limit_jb, limit_c - upper, max(old_limit_c - upper, 0.0f), penalty, dt, angular != 0u,
                        avbd::maximum_penalty, 0, 0, avbd::maximum_penalty, false, 0, error + ra, rb,
                        pm_avbd_limit_stabilization);
                    pm_avbd_visit_joint_row<Visit>(j, bodies, block, index, impulse, first,
                        limit_ja, limit_jb, limit_c - lower, min(old_limit_c - lower, 0.0f), penalty, dt, angular != 0u,
                        avbd::maximum_penalty, 0, -avbd::maximum_penalty, 0, false, 0, error + ra, rb,
                        pm_avbd_limit_stabilization);
                }
            }
            const uint spring_axes = angular != 0u ? j.angular_spring_axes : j.linear_spring_axes;
            if (j.type == 6u && (spring_axes & (1u << axis)) != 0u)
                pm_avbd_visit_joint_row<Visit>(j, bodies, block, index, impulse, first,
                    ja, jb, c, old_c, penalty, dt, angular != 0u,
                    pm_load(angular != 0u ? j.angular_spring_stiffness : j.linear_spring_stiffness)[axis],
                    pm_load(angular != 0u ? j.angular_spring_damping : j.linear_spring_damping)[axis],
                    -avbd::maximum_penalty, avbd::maximum_penalty, false, 0, error + ra, rb);
            if (motor && axis == 0u && (angular != 0u ? j.angular_motor_enabled : j.linear_motor_enabled) != 0u) {
                const float speed = angular != 0u ? j.angular_target_velocity : j.linear_target_velocity;
                const float bound = (angular != 0u ? j.angular_maximum_impulse : j.linear_maximum_impulse) / dt;
                const PMQuaternion relative = pm_quaternion_multiply(pm_quaternion_conjugate(fa), fb);
                const PMQuaternion old_relative = pm_quaternion_multiply(pm_quaternion_conjugate(old_fa), old_fb);
                const float3 motor_rotation = pm_relative_rotation_vector({0, 0, 0, 1},
                    pm_quaternion_multiply(relative, pm_quaternion_conjugate(old_relative)));
                const float3 motor_axis = pm_avbd_log_axis(fa, motor_rotation, axis);
                avbd::Vector6 motor_ja = ja, motor_jb = jb;
                if (angular != 0u) {
                    motor_ja = pm_avbd_jacobian(float3(0), -motor_axis);
                    motor_jb = pm_avbd_jacobian(float3(0), motor_axis);
                }
                const float travel = angular != 0u ? motor_rotation[axis] : c - old_c;
                pm_avbd_visit_joint_row<Visit>(j, bodies, block, index, impulse, first,
                    motor_ja, motor_jb, travel - speed * dt, 0, penalty, dt, angular != 0u,
                    avbd::maximum_penalty, 0, -bound, bound, true, speed, error + ra, rb);
            }
        }
    }
    if (Visit == pm_avbd_initialize_joint) j.avbd_count = index;
    return impulse;
}

// Collision groups are topology only. No compound momentum averaging or
// guided pose projection is performed outside the AVBD constraint graph.
kernel void pm_avbd_collision_groups(
    constant PMStepConstants &step [[buffer(4)]],
    device const PMRigidConstraintResource *joints [[buffer(9)]],
    device PMRigidCollisionGroup *groups [[buffer(25)]],
    uint lane [[thread_position_in_grid]]) {
    if (lane != 0u) return;
    for (uint body = 0u; body < step.body_count; ++body) {
        groups[body] = {}; groups[body].root = body;
    }
    for (uint index = 0u; index < step.constraint_capacity; ++index) {
        device const PMRigidConstraintResource &j = joints[index];
        if (j.alive == 0u || j.enabled == 0u || j.broken != 0u || j.type != 0u
            || j.disable_collisions == 0u || j.breaking_impulse_threshold > 0.0f
            || j.body_a >= step.body_count || j.body_b >= step.body_count) continue;
        const uint a = pm_collision_group_root(groups, j.body_a);
        const uint b = pm_collision_group_root(groups, j.body_b);
        groups[max(a, b)].root = min(a, b);
    }
    for (uint body = 0u; body < step.body_count; ++body) {
        groups[body].root = pm_collision_group_root(groups, body);
        groups[body].eligible = 1u;
    }
}

kernel void pm_avbd_prepare(
    device const PMRigidBodyState *predicted [[buffer(0)]],
    device const PMRigidParameters *parameters [[buffer(1)]],
    constant PMStepConstants &step [[buffer(4)]],
    device PMContactManifold *manifolds [[buffer(8)]],
    device PMRigidConstraintResource *joints [[buffer(9)]],
    device const PMHandle *ids [[buffer(10)]],
    device PMRigidContactEvent *events [[buffer(11)]],
    device uint &event_count [[buffer(12)]],
    device const PMRigidBodyState *previous [[buffer(14)]],
    device const uint *pairs [[buffer(18)]],
    device const uint &pair_count [[buffer(19)]],
    device PMAvbdBody *bodies [[buffer(22)]],
    device PMAvbdSchedule &schedule [[buffer(29)]],
    uint lane [[thread_index_in_threadgroup]], uint3 group_size [[threads_per_threadgroup]]) {
    threadgroup PMAvbdGraphBody body_cache[pm_avbd_cached_bodies];
    threadgroup PMAvbdGraphContact contact_cache[pm_avbd_cached_contacts];
    threadgroup PMAvbdGraphJoint joint_cache[pm_avbd_cached_joints];
    const PMAvbdGraphCache graph{bodies, parameters, manifolds, pairs, joints,
        body_cache, contact_cache, joint_cache, step.body_count};
    const float dt = step.timestep;
    for (uint i = lane; i < step.body_count; i += group_size.x) {
        const float3 previous_velocity = pm_load(bodies[i].previous_velocity);
        const bool history_valid = bodies[i].history_valid != 0u;
        bodies[i] = {};
        bodies[i].contact_head = bodies[i].joint_head = bodies[i].color = 0xffffffffu;
        bodies[i].parent = i;
        bodies[i].target = predicted[i];
        bodies[i].current_orientation = predicted[i].orientation;
        if (i < pm_avbd_cached_bodies)
            body_cache[i] = {i, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                parameters[i].motion == 2u ? 1u : 0u};
        if (parameters[i].motion != 2u) continue;
        const float3 gravity = pm_load(step.gravity);
        const float gravity_squared = dot(gravity, gravity);
        const float3 acceleration = (pm_load(previous[i].linear_velocity) - previous_velocity) / dt;
        float weight = history_valid && gravity_squared > 0.0f
            ? clamp(dot(acceleration, gravity) / gravity_squared, 0.0f, 1.0f) : 0.0f;
        if (!isfinite(weight)) weight = 0.0f;
        const float3 correction = gravity * (-(1.0f - weight) * dt * dt);
        bodies[i].displacement.v[0] = correction.x;
        bodies[i].displacement.v[1] = correction.y;
        bodies[i].displacement.v[2] = correction.z;
        bodies[i].previous_velocity = previous[i].linear_velocity;
        bodies[i].history_valid = 1u;
        const float mass = 1.0f / max(parameters[i].inverse_mass, 1.0e-12f);
        for (uint axis = 0u; axis < 3u; ++axis) {
            bodies[i].inertia.h[axis][axis] = mass / (dt * dt);
            const float3 local_axis = pm_rotate(pm_quaternion_conjugate(predicted[i].orientation), pm_avbd_axis(axis));
            const float3 column = pm_rotate(predicted[i].orientation,
                local_axis / max(pm_load(parameters[i].inverse_inertia), float3(1.0e-12f)));
            for (uint row = axis; row < 3u; ++row)
                bodies[i].inertia.h[3u + row][3u + axis] = column[row] / (dt * dt);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint active = lane; active < pair_count; active += group_size.x) {
        device PMContactManifold &m = manifolds[active];
        const uint a = pairs[active] / step.body_count, b = pairs[active] % step.body_count;
        const float penalty = pm_avbd_pair_stiffness(parameters[a], parameters[b], dt);
        const float skin = pm_avbd_contact_skin(parameters, predicted, joints,
            step.body_count, step.constraint_capacity, a, b);
        for (uint point = 0u; point < m.count; ++point) {
            device PMContactRecord &c = m.contacts[point];
            const float3 normal = pm_load(c.normal), position = pm_load(c.point);
            if (c.avbd_cached == 0u) {
                c.avbd_anchor_a = pm_store(pm_rotate(pm_quaternion_conjugate(previous[a].orientation), position - pm_load(previous[a].position)));
                c.avbd_anchor_b = pm_store(pm_rotate(pm_quaternion_conjugate(previous[b].orientation), position - pm_load(previous[b].position)));
            }
            const float3 motion_a = pm_load(predicted[a].position) - pm_load(previous[a].position);
            const float3 motion_b = pm_load(predicted[b].position) - pm_load(previous[b].position);
            const float3 wa = pm_quaternion_delta_velocity(previous[a].orientation, predicted[a].orientation, dt) * dt;
            const float3 wb = pm_quaternion_delta_velocity(previous[b].orientation, predicted[b].orientation, dt) * dt;
            const float3 ra = position - pm_load(predicted[a].position), rb = position - pm_load(predicted[b].position);
            const float closing = dot(normal, motion_b + cross(wb, rb) - motion_a - cross(wa, ra));
            const float old_penetration = c.penetration - closing;
            float3 t0{}, t1{}; pm_avbd_basis(normal, t0, t1);
            const float3 anchor_delta = pm_load(predicted[b].position)
                + pm_rotate(predicted[b].orientation, pm_load(c.avbd_anchor_b))
                - pm_load(predicted[a].position) - pm_rotate(predicted[a].orientation, pm_load(c.avbd_anchor_a));
            const float3 old_anchor_delta = pm_load(previous[b].position)
                + pm_rotate(previous[b].orientation, pm_load(c.avbd_anchor_b))
                - pm_load(previous[a].position) - pm_rotate(previous[a].orientation, pm_load(c.avbd_anchor_a));
            c.initial_normal_speed = -closing / dt;
            c.avbd_error = {c.penetration - clamp(old_penetration, 0.0f, skin),
                dot(anchor_delta - avbd::alpha * old_anchor_delta, t0),
                dot(anchor_delta - avbd::alpha * old_anchor_delta, t1)};
            for (uint axis = 0u; axis < 3u; ++axis) {
                if (c.avbd_cached == 0u) c.avbd_dual[axis] = {0, penalty};
                c.avbd_dual[axis] = avbd::warm_start(c.avbd_dual[axis]);
            }
        }
        if (active < pm_avbd_cached_contacts)
            contact_cache[active] = {a, b, m.avbd_next_a, m.avbd_next_b,
                (m.count != 0u ? 1u : 0u) |
                (pm_avbd_prepare_contact_impact(m, skin, dt) ? 2u : 0u)};
    }
    for (uint index = lane; index < step.constraint_capacity; index += group_size.x) {
        device PMRigidConstraintResource &j = joints[index];
        j.avbd_count = 0u;
        if (j.alive == 0u || j.enabled == 0u || j.broken != 0u) continue;
        j.applied_impulse = 0;
        if (j.body_a >= step.body_count || j.body_b >= step.body_count) continue;
        avbd::Block scratch{};
        pm_avbd_visit_joint<pm_avbd_initialize_joint>(j, parameters, previous, bodies, dt, scratch);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint index = lane; index < min(step.constraint_capacity, uint(pm_avbd_cached_joints)); index += group_size.x)
        joint_cache[index] = pm_avbd_graph_joint_device(graph, index);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (lane == 0u) {
    schedule = {};
    schedule.budget = step.rigid_contact_pass_limit != 0u ? step.rigid_contact_pass_limit : avbd::default_iterations;
    schedule.candidates = pair_count;
    uint event_cursor = 0u;
    for (uint active = 0u; active < pair_count; ++active) {
        PMAvbdGraphContact contact = pm_avbd_graph_contact(graph, active);
        if ((contact.flags & 1u) == 0u) continue;
        ++schedule.contacts;
        const uint a = contact.a, b = contact.b;
        pm_avbd_graph_union(graph, a, b);
        PMAvbdGraphBody body_a = pm_avbd_graph_body(graph, a);
        contact.next_a = body_a.contact_head; body_a.contact_head = active;
        pm_avbd_graph_store_body(graph, a, body_a);
        PMAvbdGraphBody body_b = pm_avbd_graph_body(graph, b);
        contact.next_b = body_b.contact_head; body_b.contact_head = active;
        pm_avbd_graph_store_body(graph, b, body_b);
        pm_avbd_graph_store_contact(graph, active, contact);
        device PMContactManifold &m = manifolds[active];
        m.event_offset = event_cursor;
        if (step.collect_rigid_contacts != 0u) {
            for (uint point = 0u; point < m.count; ++point) {
                if (event_cursor + point >= step.rigid_event_capacity) break;
                const PMContactRecord c = m.contacts[point];
                events[event_cursor + point] = {ids[a], ids[b], c.point, c.normal, max(c.penetration, 0.0f), 0, {}};
            }
            event_cursor += m.count;
        }
    }
    if (event_cursor != 0u) event_count = min(event_cursor, step.rigid_event_capacity);
    for (uint index = 0u; index < step.constraint_capacity; ++index) {
        PMAvbdGraphJoint joint = pm_avbd_graph_joint(graph, index);
        if (joint.count == 0u) continue;
        if (step.rigid_contact_pass_limit == 0u)
            schedule.budget = max(schedule.budget, joint.iterations);
        pm_avbd_graph_union(graph, joint.a, joint.b);
        PMAvbdGraphBody body_a = pm_avbd_graph_body(graph, joint.a);
        joint.next_a = body_a.joint_head; body_a.joint_head = index;
        pm_avbd_graph_store_body(graph, joint.a, body_a);
        PMAvbdGraphBody body_b = pm_avbd_graph_body(graph, joint.b);
        joint.next_b = body_b.joint_head; body_b.joint_head = index;
        pm_avbd_graph_store_body(graph, joint.b, body_b);
        pm_avbd_graph_store_joint(graph, index, joint);
    }
    for (uint body = 0u; body < step.body_count; ++body) {
        PMAvbdGraphBody value = pm_avbd_graph_body(graph, body);
        value.parent = pm_avbd_graph_root(graph, body);
        pm_avbd_graph_store_body(graph, body, value);
    }
    for (uint active = 0u; active < pair_count; ++active) {
        const PMAvbdGraphContact contact = pm_avbd_graph_contact(graph, active);
        const uint a = contact.a, b = contact.b;
        const bool impact = active < pm_avbd_cached_contacts
            ? (contact.flags & 2u) != 0u
            : pm_avbd_prepare_contact_impact(manifolds[active],
                pm_avbd_contact_skin(parameters, predicted, joints,
                    step.body_count, step.constraint_capacity, a, b), dt);
        if (impact) {
            const PMAvbdGraphBody body_a = pm_avbd_graph_body(graph, a);
            if ((body_a.flags & 1u) != 0u) {
                PMAvbdGraphBody root = pm_avbd_graph_body(graph, body_a.parent);
                root.flags |= 2u;
                pm_avbd_graph_store_body(graph, body_a.parent, root);
            }
            const PMAvbdGraphBody body_b = pm_avbd_graph_body(graph, b);
            if ((body_b.flags & 1u) != 0u) {
                PMAvbdGraphBody root = pm_avbd_graph_body(graph, body_b.parent);
                root.flags |= 2u;
                pm_avbd_graph_store_body(graph, body_b.parent, root);
            }
            schedule.reserved[0] = 1u;
        }
    }
    for (uint body = 0u; body < step.body_count; ++body) {
        PMAvbdGraphBody value = pm_avbd_graph_body(graph, body);
        if ((value.flags & 1u) == 0u) continue;
        value.flags = (value.flags & 1u) | (pm_avbd_graph_body(graph, value.parent).flags & 2u);
        pm_avbd_graph_store_body(graph, body, value);
        if (value.parent == body && (value.contact_head != 0xffffffffu || value.joint_head != 0xffffffffu))
            ++schedule.islands;
    }
    for (uint body = 0u; body < step.body_count; ++body) {
        PMAvbdGraphBody value = pm_avbd_graph_body(graph, body);
        if ((value.flags & 1u) == 0u) continue;
        uint color = 0u;
        for (;; ++color) {
            bool conflict = false;
            for (uint active = value.contact_head; active != 0xffffffffu;) {
                const PMAvbdGraphContact contact = pm_avbd_graph_contact(graph, active);
                const bool first = body == contact.a;
                conflict = conflict || pm_avbd_graph_body(graph, first ? contact.b : contact.a).color == color;
                active = first ? contact.next_a : contact.next_b;
            }
            for (uint index = value.joint_head; index != 0xffffffffu;) {
                const PMAvbdGraphJoint joint = pm_avbd_graph_joint(graph, index);
                const bool first = body == joint.a;
                conflict = conflict || pm_avbd_graph_body(graph, first ? joint.b : joint.a).color == color;
                index = first ? joint.next_a : joint.next_b;
            }
            if (!conflict) break;
        }
        value.color = color;
        pm_avbd_graph_store_body(graph, body, value);
        schedule.colors = max(schedule.colors, color + 1u);
    }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint body = lane; body < min(step.body_count, uint(pm_avbd_cached_bodies)); body += group_size.x)
        pm_avbd_graph_store_body_device(graph, body, body_cache[body]);
    for (uint active = lane; active < min(pair_count, uint(pm_avbd_cached_contacts)); active += group_size.x)
        pm_avbd_graph_store_contact_device(graph, active, contact_cache[active]);
    for (uint index = lane; index < min(step.constraint_capacity, uint(pm_avbd_cached_joints)); index += group_size.x)
        pm_avbd_graph_store_joint_device(graph, index, joint_cache[index]);
}

static void pm_avbd_minimize_free_translation(device const PMRigidParameters *parameters,
    uint count, device const PMContactManifold *manifolds, device const uint *pairs,
    device const PMRigidConstraintResource *joints, device PMAvbdBody *bodies,
    bool velocity_phase, uint lane, uint stride) {
    // Common translation is an exact null mode of internal constraints.
    // Minimize the same inertial energy along it, preserving COM despite
    // finite Gauss-Seidel sweeps. Static/kinematic endpoints remain anchored.
    for (uint root = lane; root < count; root += stride) {
        if (parameters[root].motion != 2u || bodies[root].parent != root) continue;
        for (uint axis = 0u; axis < 3u; ++axis) bodies[root].inertia.g[axis] = 0;
        if ((velocity_phase && bodies[root].impact == 0u)
            || (bodies[root].contact_head == 0xffffffffu && bodies[root].joint_head == 0xffffffffu)) continue;
        float3 weighted = float3(0);
        float total_mass = 0;
        bool anchored = false;
        for (uint body = 0u; body < count && !anchored; ++body) {
            if (parameters[body].motion != 2u || bodies[body].parent != root) continue;
            const float mass = 1.0f / max(parameters[body].inverse_mass, 1.0e-12f);
            weighted += pm_avbd_translation(bodies[body].displacement) * mass;
            total_mass += mass;
            for (uint active = bodies[body].contact_head; active != 0xffffffffu;) {
                device const PMContactManifold &m = manifolds[active];
                const uint a = pairs[active] / count, b = pairs[active] % count;
                const bool first = body == a;
                anchored = anchored || parameters[first ? b : a].motion != 2u;
                active = first ? m.avbd_next_a : m.avbd_next_b;
            }
            for (uint index = bodies[body].joint_head; index != 0xffffffffu;) {
                device const PMRigidConstraintResource &j = joints[index];
                const bool first = body == j.body_a;
                // Preserve reactions from a joint broken during this phase.
                anchored = anchored || parameters[first ? j.body_b : j.body_a].motion != 2u;
                index = first ? j.avbd_next_a : j.avbd_next_b;
            }
        }
        if (!anchored && total_mass > 0.0f) {
            weighted /= total_mass;
            for (uint axis = 0u; axis < 3u; ++axis) bodies[root].inertia.g[axis] = weighted[axis];
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint body = lane; body < count; body += stride) {
        if (parameters[body].motion != 2u || (velocity_phase && bodies[body].impact == 0u)) continue;
        device const avbd::Block &mean = bodies[bodies[body].parent].inertia;
        for (uint axis = 0u; axis < 3u; ++axis) bodies[body].displacement.v[axis] -= mean.g[axis];
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint body = lane; body < count; body += stride)
        for (uint axis = 0u; axis < 3u; ++axis) bodies[body].inertia.g[axis] = 0;
    threadgroup_barrier(mem_flags::mem_device);
}

kernel void pm_avbd_solve(
    device PMRigidBodyState *states [[buffer(0)]],
    device const PMRigidParameters *parameters [[buffer(1)]],
    constant PMStepConstants &step [[buffer(4)]],
    device PMContactManifold *manifolds [[buffer(8)]],
    device PMRigidConstraintResource *joints [[buffer(9)]],
    device PMRigidContactEvent *events [[buffer(11)]],
    device const PMRigidBodyState *previous [[buffer(14)]],
    device const uint *pairs [[buffer(18)]], device const uint &pair_count [[buffer(19)]],
    device PMAvbdBody *bodies [[buffer(22)]], device PMAvbdSchedule &schedule [[buffer(29)]],
    uint lane [[thread_index_in_threadgroup]], uint3 group_size [[threads_per_threadgroup]]) {
    const float dt = step.timestep;
    // Both pose and impact-velocity correction use the same AVBD body block,
    // graph colors, cone bounds and dual update. No legacy impulse solver.
    for (uint phase = 0u; phase < 2u; ++phase) {
    if (phase != 0u) {
        if (schedule.reserved[0] == 0u) break;
        for (uint active = lane; active < pair_count; active += group_size.x) {
            device PMContactManifold &m = manifolds[active];
            const uint a = pairs[active] / step.body_count, b = pairs[active] % step.body_count;
            if (bodies[a].impact == 0u && bodies[b].impact == 0u) continue;
            for (uint point = 0u; point < m.count; ++point) {
                device PMContactRecord &c = m.contacts[point];
                avbd::Vector6 ja{}, jb{};
                pm_avbd_contact_normal_row(c, a, b, bodies, ja, jb);
                const float gap = max(0.0f, -(c.penetration + avbd::dot6(ja, bodies[a].displacement)
                    + avbd::dot6(jb, bodies[b].displacement)));
                c.avbd_error.x = gap;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint body = lane; body < step.body_count; body += group_size.x) {
            if (bodies[body].impact != 0u || parameters[body].motion != 2u) {
                bodies[body].target = states[body]; bodies[body].displacement = {};
                bodies[body].current_orientation = states[body].orientation;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint index = lane; index < step.constraint_capacity; index += group_size.x) {
            device PMRigidConstraintResource &j = joints[index];
            if (j.avbd_count == 0u || j.enabled == 0u || j.broken != 0u) continue;
            if (bodies[j.body_a].impact == 0u && bodies[j.body_b].impact == 0u) continue;
            avbd::Block scratch{};
            pm_avbd_visit_joint<pm_avbd_impact_joint>(j, parameters, previous, bodies, dt, scratch);
        }
        for (uint active = lane; active < pair_count; active += group_size.x) {
            device PMContactManifold &m = manifolds[active];
            const uint a = pairs[active] / step.body_count, b = pairs[active] % step.body_count;
            if (bodies[a].impact == 0u && bodies[b].impact == 0u) continue;
            const avbd::Vector6 va = pm_avbd_jacobian(pm_load(states[a].linear_velocity) * dt, pm_load(states[a].angular_velocity) * dt);
            const avbd::Vector6 vb = pm_avbd_jacobian(pm_load(states[b].linear_velocity) * dt, pm_load(states[b].angular_velocity) * dt);
            for (uint point = 0u; point < m.count; ++point) {
                device PMContactRecord &c = m.contacts[point];
                const float gap = c.avbd_error.x;
                avbd::Vector6 ja[3]{}, jb[3]{}; float error[3]{};
                pm_avbd_contact_rows(c, a, b, bodies, ja, jb, error);
                const float restitution = c.avbd_cached == 0u && c.initial_normal_speed < -1.0f
                    ? min(parameters[a].restitution, parameters[b].restitution) : 0.0f;
                c.avbd_error = {
                    avbd::dot6(ja[0], va) + avbd::dot6(jb[0], vb) - gap - restitution * c.initial_normal_speed * dt,
                    avbd::dot6(ja[1], va) + avbd::dot6(jb[1], vb),
                    avbd::dot6(ja[2], va) + avbd::dot6(jb[2], vb)};
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    for (uint iteration = 0u; iteration < schedule.budget; ++iteration) {
        for (uint color = 0u; color < schedule.colors; ++color) {
            for (uint body = lane; body < step.body_count; body += group_size.x) {
                if (bodies[body].color != color) continue;
                if (phase != 0u && bodies[body].impact == 0u) continue;
                avbd::Block block = bodies[body].inertia;
                const avbd::Vector6 displacement = bodies[body].displacement;
                for (uint i = 0u; i < 6u; ++i)
                    for (uint j = 0u; j < 6u; ++j)
                        block.g[i] += block.h[max(i, j)][min(i, j)] * displacement.v[j];
                for (uint active = bodies[body].contact_head; active != 0xffffffffu;) {
                    device const PMContactManifold &m = manifolds[active];
                    const uint a = pairs[active] / step.body_count, b = pairs[active] % step.body_count;
                    const bool first = body == a;
                    const float friction = sqrt(parameters[a].friction * parameters[b].friction);
                    for (uint point = 0u; point < m.count; ++point) {
                        const PMContactRecord c = m.contacts[point];
                        avbd::Vector6 ja[3]{}, jb[3]{}; float error[3]{};
                        pm_avbd_contact_rows(c, a, b, bodies, ja, jb, error);
                        const avbd::ContactForce trial = pm_avbd_contact_trial(c, error);
                        const avbd::ContactForce force = avbd::project_contact(trial, friction);
                        const avbd::ContactForce scales = avbd::contact_stiffness_scales(trial, friction);
                        const float stiffness_scales[3] = {scales.normal, scales.tangent0, scales.tangent1};
                        float3 t0{}, t1{}; pm_avbd_basis(pm_load(c.normal), t0, t1);
                        const float f[3] = {
                            force.normal - (phase != 0u ? c.accumulated_normal_impulse / dt : 0.0f),
                            force.tangent0 - (phase != 0u ? dot(pm_load(c.accumulated_friction_impulse), t0) / dt : 0.0f),
                            force.tangent1 - (phase != 0u ? dot(pm_load(c.accumulated_friction_impulse), t1) / dt : 0.0f)};
                        for (uint axis = 0u; axis < 3u; ++axis)
                            avbd::add_row(block, first ? ja[axis] : jb[axis],
                                c.avbd_dual[axis].penalty * stiffness_scales[axis], f[axis]);
                    }
                    active = first ? m.avbd_next_a : m.avbd_next_b;
                }
                for (uint index = bodies[body].joint_head; index != 0xffffffffu;) {
                    device PMRigidConstraintResource &j = joints[index];
                    const bool first = body == j.body_a;
                    if (j.enabled != 0u && j.broken == 0u) {
                        if (phase == 0u) {
                            pm_avbd_visit_joint<pm_avbd_accumulate_joint>(j, parameters, previous, bodies, dt, block, first);
                        } else {
                            for (uint row = 0u; row < j.avbd_count; ++row)
                                if (j.avbd_rows[row].stiffness >= avbd::maximum_penalty)
                                    avbd::accumulate(block, j.avbd_rows[row], first,
                                        bodies[j.body_a].displacement, bodies[j.body_b].displacement, dt);
                        }
                    }
                    index = first ? j.avbd_next_a : j.avbd_next_b;
                }
                avbd::Vector6 update{};
                if (avbd::solve(block, update)) {
                    {
                        // Re-solve crossed corners on their predicted active
                        // side. Include affine force and Hessian together:
                        // Hessian-only damping creates artificial pose gaps.
                        // Separating rows remain completely inactive.
                        bool crossing = false;
                        for (uint active = bodies[body].contact_head; active != 0xffffffffu;) {
                            device const PMContactManifold &m = manifolds[active];
                            const uint a = pairs[active] / step.body_count, b = pairs[active] % step.body_count;
                            const bool first = body == a;
                            for (uint point = 0u; point < m.count; ++point) {
                                const PMContactRecord c = m.contacts[point];
                                avbd::Vector6 ja{}, jb{};
                                const float error = pm_avbd_contact_normal_row(c, a, b, bodies, ja, jb);
                                const float trial = c.avbd_dual[0].lambda + c.avbd_dual[0].penalty * error;
                                const avbd::Vector6 normal_row = first ? ja : jb;
                                // Ignore roundoff-sized boundary crossings.
                                if (trial <= 0.0f && trial + c.avbd_dual[0].penalty * avbd::dot6(normal_row, update)
                                    > c.avbd_dual[0].penalty * 1.0e-6f) {
                                    avbd::add_row(block, normal_row, c.avbd_dual[0].penalty, trial);
                                    crossing = true;
                                }
                            }
                            active = first ? m.avbd_next_a : m.avbd_next_b;
                        }
                        if (crossing) {
                            avbd::Vector6 safeguarded{};
                            if (avbd::solve(block, safeguarded)) update = safeguarded;
                        }
                    }
                    if (phase == 0u) pm_avbd_apply_pose_update(bodies[body], update);
                    else for (uint i = 0u; i < 6u; ++i) bodies[body].displacement.v[i] += update.v[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        for (uint active = lane; active < pair_count; active += group_size.x) {
            device PMContactManifold &m = manifolds[active];
            const uint a = pairs[active] / step.body_count, b = pairs[active] % step.body_count;
            if (phase != 0u && bodies[a].impact == 0u && bodies[b].impact == 0u) continue;
            const float friction = sqrt(parameters[a].friction * parameters[b].friction);
            const float beta = pm_avbd_pair_stiffness(parameters[a], parameters[b], dt) * 100.0f;
            for (uint point = 0u; point < m.count; ++point) {
                device PMContactRecord &c = m.contacts[point];
                avbd::Vector6 ja[3]{}, jb[3]{}; float error[3]{};
                pm_avbd_contact_rows(c, a, b, bodies, ja, jb, error);
                const avbd::ContactForce force = pm_avbd_contact_force(c, error, friction);
                const float tangential = length(float2(c.avbd_dual[1].lambda + c.avbd_dual[1].penalty * error[1],
                    c.avbd_dual[2].lambda + c.avbd_dual[2].penalty * error[2]));
                const bool stick = tangential <= friction * force.normal;
                c.avbd_dual[0].lambda = force.normal;
                c.avbd_dual[1].lambda = force.tangent0;
                c.avbd_dual[2].lambda = force.tangent1;
                for (uint axis = 0u; axis < 3u; ++axis)
                    if (axis == 0u ? force.normal > 0.0f : stick)
                        c.avbd_dual[axis].penalty = min(avbd::maximum_penalty,
                            c.avbd_dual[axis].penalty + beta * abs(error[axis]));
                if (iteration + 1u == schedule.budget) {
                    float3 t0{}, t1{}; pm_avbd_basis(pm_load(c.normal), t0, t1);
                    c.accumulated_normal_impulse = force.normal * dt;
                    c.accumulated_friction_impulse = pm_store((t0 * force.tangent0 + t1 * force.tangent1) * dt);
                    const uint event = m.event_offset + point;
                    if (step.collect_rigid_contacts != 0u && event < step.rigid_event_capacity) {
                        events[event].normal_impulse = c.accumulated_normal_impulse;
                        events[event].friction_impulse = c.accumulated_friction_impulse;
                    }
                    if (!stick) {
                        PMRigidBodyState pose_a = bodies[a].target, pose_b = bodies[b].target;
                        if (phase == 0u) {
                            pose_a = pm_avbd_pose(bodies[a]); pose_b = pm_avbd_pose(bodies[b]);
                        }
                        c.avbd_anchor_a = pm_store(pm_rotate(pm_quaternion_conjugate(pose_a.orientation), pm_load(c.point) - pm_load(pose_a.position)));
                        c.avbd_anchor_b = pm_store(pm_rotate(pm_quaternion_conjugate(pose_b.orientation), pm_load(c.point) - pm_load(pose_b.position)));
                    }
                }
            }
        }
        for (uint index = lane; index < step.constraint_capacity; index += group_size.x) {
            device PMRigidConstraintResource &j = joints[index];
            if (j.avbd_count == 0u || j.enabled == 0u || j.broken != 0u) continue;
            if (phase != 0u && bodies[j.body_a].impact == 0u && bodies[j.body_b].impact == 0u) continue;
            float impulse = 0;
            if (phase == 0u) {
                avbd::Block scratch{};
                impulse = pm_avbd_visit_joint<pm_avbd_advance_joint>(j, parameters, previous, bodies, dt, scratch);
            } else {
                for (uint row = 0u; row < j.avbd_count; ++row) {
                    if (j.avbd_rows[row].stiffness < avbd::maximum_penalty) {
                        impulse += abs(j.avbd_rows[row].reference_force) * dt;
                        continue;
                    }
                    impulse += abs(avbd::row_force(j.avbd_rows[row], bodies[j.body_a].displacement,
                        bodies[j.body_b].displacement, dt)) * dt;
                    j.avbd_rows[row] = avbd::advance(j.avbd_rows[row], bodies[j.body_a].displacement, bodies[j.body_b].displacement);
                }
            }
            j.applied_impulse = impulse;
            if (j.breaking_impulse_threshold > 0.0f && impulse > j.breaking_impulse_threshold) {
                j.broken = 1u; j.enabled = 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    pm_avbd_minimize_free_translation(parameters, step.body_count, manifolds, pairs,
        joints, bodies, phase != 0u, lane, group_size.x);
    for (uint body = lane; body < step.body_count; body += group_size.x) {
        if (parameters[body].motion != 2u) continue;
        if (phase != 0u && bodies[body].impact == 0u) continue;
        PMRigidBodyState state = bodies[body].target;
        const avbd::Vector6 delta = bodies[body].displacement;
        if (phase != 0u) {
            state.linear_velocity = pm_store(pm_limit(pm_load(state.linear_velocity) + pm_avbd_translation(delta) / dt,
                parameters[body].maximum_linear_speed));
            state.angular_velocity = pm_store(pm_limit(pm_load(state.angular_velocity) + pm_avbd_rotation(delta) / dt,
                parameters[body].maximum_angular_speed));
            states[body] = state;
            continue;
        }
        state.position = pm_store(pm_load(state.position) + pm_avbd_translation(delta));
        state.orientation = bodies[body].current_orientation;
        state.linear_velocity = pm_store(pm_load(bodies[body].target.linear_velocity) + pm_avbd_translation(delta) / dt);
        state.angular_velocity = pm_store(pm_load(bodies[body].target.angular_velocity)
            + pm_quaternion_delta_velocity(bodies[body].target.orientation, state.orientation, dt));
        if (bodies[body].impact == 0u) {
            state.linear_velocity = pm_store(pm_limit(pm_load(state.linear_velocity), parameters[body].maximum_linear_speed));
            state.angular_velocity = pm_store(pm_limit(pm_load(state.angular_velocity), parameters[body].maximum_angular_speed));
        }
        states[body] = state;
    }
    if (lane == 0u) schedule.maximum_passes += schedule.budget;
    threadgroup_barrier(mem_flags::mem_device);
    }
}
