// SPDX-License-Identifier: MIT
// CUDA geometry/scheduling adapter for the shared AVBD numerical core.
// Included after geometry_constraints.cuh, inside parallel_mater's internals.

struct AvbdBody {
    RigidBodyState target{};
    Quaternion current_orientation{};
    avbd::Vector6 displacement{};
    avbd::Block inertia{};
    unsigned contact_head{UINT32_MAX}, joint_head{UINT32_MAX}, color{UINT32_MAX};
    unsigned parent{}, impact{};
    Vec3 previous_velocity{};
    unsigned history_valid{};
};

__device__ unsigned avbd_root(const AvbdBody *bodies, unsigned body) {
    while (body != bodies[body].parent) body = bodies[body].parent;
    return body;
}
__device__ void avbd_union(AvbdBody *bodies, unsigned a, unsigned b, const BodyParameters *parameters) {
    if (parameters[a].motion != MotionType::dynamic || parameters[b].motion != MotionType::dynamic) return;
    a = avbd_root(bodies, a); b = avbd_root(bodies, b);
    if (a != b) bodies[max(a, b)].parent = min(a, b);
}

__device__ avbd::Vector6 avbd_jacobian(Vec3 linear, Vec3 angular) {
    return {{linear.x, linear.y, linear.z, angular.x, angular.y, angular.z}};
}
__device__ Vec3 avbd_translation(avbd::Vector6 value) {
    return {value.v[0], value.v[1], value.v[2]};
}
__device__ Vec3 avbd_rotation(avbd::Vector6 value) {
    return {value.v[3], value.v[4], value.v[5]};
}
__device__ RigidBodyState avbd_pose(const AvbdBody &body) {
    auto state = body.target;
    state.position = add(state.position, avbd_translation(body.displacement));
    state.orientation = body.current_orientation;
    return state;
}
__device__ void avbd_apply_pose_update(AvbdBody &body, avbd::Vector6 update) {
    for (unsigned i = 0; i < 3; ++i) body.displacement.v[i] += update.v[i];
    const Vec3 rotation = avbd_rotation(update);
    const Quaternion dq = quaternion_multiply({rotation.x, rotation.y, rotation.z, 0}, body.current_orientation);
    body.current_orientation = normalized_quaternion({
        body.current_orientation.x + 0.5F * dq.x, body.current_orientation.y + 0.5F * dq.y,
        body.current_orientation.z + 0.5F * dq.z, body.current_orientation.w + 0.5F * dq.w});
    Quaternion relative = quaternion_multiply(body.current_orientation, conjugate(body.target.orientation));
    if (relative.w < 0) relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    // Equation 20: rotation differences are quaternion differences, not the
    // sum of successive tangent-space Newton increments.
    body.displacement.v[3] = 2 * relative.x;
    body.displacement.v[4] = 2 * relative.y;
    body.displacement.v[5] = 2 * relative.z;
}

// Row of the inverse SO(3) left Jacobian, mapped to world coordinates.
// Log(exp(dw) R) = Log(R) + J_left^-1(Log(R)) dw + O(dw^2).
__device__ Vec3 avbd_log_axis(Quaternion frame, Vec3 logarithm, unsigned axis) {
    const float squared = length_squared(logarithm);
    float coefficient = 1.0F / 12.0F + squared / 720.0F;
    if (squared > 1e-6F) {
        const float angle = sqrtf(squared), half = 0.5F * angle;
        coefficient = (1 - half * cosf(half) / sinf(half)) / squared;
    }
    const Vec3 unit = basis_axis(axis);
    const Vec3 local = add(add(unit, multiply(cross(logarithm, unit), 0.5F)),
        multiply(cross(logarithm, cross(logarithm, unit)), coefficient));
    return rotate(frame, local);
}
__device__ void avbd_basis(Vec3 normal, Vec3 &t0, Vec3 &t1) {
    t0 = normalized_or(cross(normal, fabsf(normal.x) < 0.57735F
        ? Vec3{1, 0, 0} : Vec3{0, 1, 0}), {0, 0, 1});
    t1 = cross(normal, t0);
}

__device__ float avbd_pair_stiffness(const BodyParameters &a,
                                    const BodyParameters &b, float dt) {
    // Scale the penalty with effective mass and substep duration. Using the
    // demo's unit penalty unchanged at 240 Hz makes tiny contacts too soft.
    return 1.0F / (fmaxf(a.inverse_mass + b.inverse_mass, 1.0e-6F) * dt * dt);
}

__device__ float avbd_contact_skin(const Contact &contact,
    const BodyParameters &a, const BodyParameters &b) {
    // Guided CCD uses its sharper tooth/edge stand-off, not the ordinary
    // manifold's 1 mm rest offset. Preserve that same geometry contract.
    const float offset = guided_static_contact(contact)
        ? fminf(a.collision_margin + b.collision_margin, k_rigid_guided_rest_offset)
        : rigid_rest_offset(a.collision_margin + b.collision_margin);
    return offset + 2 * k_rigid_surface_tolerance;
}

__device__ void avbd_contact_rows(const Contact &contact,
    unsigned a, unsigned b, const AvbdBody *bodies,
    avbd::Vector6 (&ja)[3], avbd::Vector6 (&jb)[3], float (&error)[3]) {
    Vec3 t0{}, t1{};
    avbd_basis(contact.normal, t0, t1);
    const Vec3 axes[3]{contact.normal, t0, t1};
    const Vec3 arm_a = subtract(contact.point, bodies[a].target.position);
    const Vec3 arm_b = subtract(contact.point, bodies[b].target.position);
    for (unsigned axis = 0; axis < 3; ++axis) {
        const Vec3 ra = axis == 0 ? arm_a : rotate(bodies[a].target.orientation, contact.avbd_anchor_a);
        const Vec3 rb = axis == 0 ? arm_b : rotate(bodies[b].target.orientation, contact.avbd_anchor_b);
        ja[axis] = avbd_jacobian(multiply(axes[axis], -1), multiply(cross(ra, axes[axis]), -1));
        jb[axis] = avbd_jacobian(axes[axis], cross(rb, axes[axis]));
        error[axis] = component(contact.avbd_error, axis)
            + avbd::dot6(ja[axis], bodies[a].displacement)
            + avbd::dot6(jb[axis], bodies[b].displacement);
    }
}

// Active-set checks need only the normal. Do not construct tangent frames or
// rotate friction anchors on this second traversal of every contact patch.
__device__ float avbd_contact_normal_row(const Contact &contact,
    unsigned a, unsigned b, const AvbdBody *bodies,
    avbd::Vector6 &ja, avbd::Vector6 &jb) {
    const Vec3 arm_a = subtract(contact.point, bodies[a].target.position);
    const Vec3 arm_b = subtract(contact.point, bodies[b].target.position);
    ja = avbd_jacobian(multiply(contact.normal, -1), multiply(cross(arm_a, contact.normal), -1));
    jb = avbd_jacobian(contact.normal, cross(arm_b, contact.normal));
    return contact.avbd_error.x + avbd::dot6(ja, bodies[a].displacement)
        + avbd::dot6(jb, bodies[b].displacement);
}

__device__ avbd::ContactForce avbd_contact_trial(const Contact &contact, const float (&error)[3]) {
    return {
        contact.avbd_dual[0].lambda + contact.avbd_dual[0].penalty * error[0],
        contact.avbd_dual[1].lambda + contact.avbd_dual[1].penalty * error[1],
        contact.avbd_dual[2].lambda + contact.avbd_dual[2].penalty * error[2]};
}
__device__ avbd::ContactForce avbd_contact_force(const Contact &contact,
                                                const float (&error)[3], float friction) {
    return avbd::project_contact(avbd_contact_trial(contact, error), friction);
}

enum AvbdJointVisit : unsigned { avbd_initialize_joint, avbd_accumulate_joint,
    avbd_advance_joint, avbd_impact_joint };
// Authored joint limits retain the established 20% recovery per substep.
// Only the prior violation is stabilized; the permitted interior gap is not.
constexpr float avbd_limit_stabilization = 0.8F;

template<AvbdJointVisit Visit>
__device__ void avbd_visit_joint_row(RigidConstraintResource &joint,
    const AvbdBody *bodies, avbd::Block &block, unsigned &index, float &impulse, bool first,
    avbd::Vector6 ja, avbd::Vector6 jb, float current, float previous,
    float penalty, float dt, bool angular, float stiffness = avbd::maximum_penalty,
    float damping = 0, float lo = -avbd::maximum_penalty, float hi = avbd::maximum_penalty,
    bool motor = false, float motor_speed = 0, Vec3 curvature_a = {}, Vec3 curvature_b = {},
    float stabilization = avbd::alpha) {
    const unsigned row_index = index++;
    auto row = joint.avbd_rows[row_index];
    const unsigned a = joint.geometry.dense_a, b = joint.geometry.dense_b;
    const float linearized = avbd::dot6(ja, bodies[a].displacement)
        + avbd::dot6(jb, bodies[b].displacement);
    row.a = ja; row.b = jb;
    row.stiffness = stiffness; row.damping = damping;
    if constexpr (Visit == avbd_initialize_joint) {
        if (!(row.dual.penalty > avbd::minimum_penalty)) row.dual.penalty = penalty;
        row.dual = avbd::warm_start(row.dual, stiffness);
        if (stiffness < avbd::maximum_penalty) row.dual.lambda = 0;
        row.beta = penalty * (angular ? 10.0F : 100.0F);
        row.reference_force = 0;
    }
    // An affine local representation of the current nonlinear constraint.
    // The shared core adds J*displacement, giving exactly C(current), while
    // its gradient and Hessian use the freshly evaluated tangent Jacobian.
    row.error = current - (stiffness < avbd::maximum_penalty ? 0.0F : stabilization * previous) - linearized;
    row.velocity = motor ? motor_speed : (current - previous - linearized) / dt;
    row.lower = lo; row.upper = hi;
    if constexpr (Visit == avbd_accumulate_joint) {
        avbd::accumulate(block, row, first, bodies[a].displacement, bodies[b].displacement, dt);
        if (!angular) {
            // PSD diagonal approximation of force-weighted rotational
            // curvature (paper Sec. 3.5). For A the local frame itself rotates;
            // for B only its anchor rotates. Never use an indefinite Hessian.
            const Vec3 axis = avbd_translation(jb), arm = first ? curvature_a : curvature_b;
            const float force = avbd::row_force(row, bodies[a].displacement, bodies[b].displacement, dt);
            for (unsigned column = 0; column < 3; ++column) {
                const Vec3 value = subtract(multiply(add(multiply(axis, component(arm, column)),
                    multiply(arm, component(axis, column))), 0.5F), multiply(basis_axis(column), dot(axis, arm)));
                block.h[3 + column][3 + column] += fabsf(force) * sqrtf(length_squared(value));
            }
        }
    } else if constexpr (Visit == avbd_advance_joint) {
        impulse += fabsf(avbd::row_force(row, bodies[a].displacement, bodies[b].displacement, dt)) * dt;
        joint.avbd_rows[row_index] = avbd::advance(row, bodies[a].displacement, bodies[b].displacement);
    } else if constexpr (Visit == avbd_impact_joint) {
        if (stiffness < avbd::maximum_penalty) {
            row.reference_force = avbd::row_force(row, bodies[a].displacement, bodies[b].displacement, dt);
        } else {
            // Velocity limits use the actual geometric gap, not the relaxed
            // pose-phase error: an already violated bound permits no further
            // outward travel even while its old violation is being recovered.
            const float gap = lo == 0 ? fminf(0, current) : hi == 0 ? fmaxf(0, current) : 0;
            const auto va = avbd_jacobian(multiply(bodies[a].target.linear_velocity, dt),
                multiply(bodies[a].target.angular_velocity, dt));
            const auto vb = avbd_jacobian(multiply(bodies[b].target.linear_velocity, dt),
                multiply(bodies[b].target.angular_velocity, dt));
            row.reference_force = row.dual.lambda;
            row.error = avbd::dot6(ja, va) + avbd::dot6(jb, vb) + gap - (motor ? motor_speed * dt : 0);
        }
        joint.avbd_rows[row_index] = row;
    } else {
        joint.avbd_rows[row_index] = row;
    }
}

template<AvbdJointVisit Visit>
__device__ float avbd_visit_joint(RigidConstraintResource &joint,
    const BodyParameters *parameters, const RigidBodyState *previous,
    const AvbdBody *bodies, float dt, avbd::Block &block, bool first = true) {
    unsigned index = 0;
    float impulse = 0;
    const auto &o = joint.options;
    const unsigned a = joint.geometry.dense_a, b = joint.geometry.dense_b;
    const auto sa = avbd_pose(bodies[a]), sb = avbd_pose(bodies[b]);
    const Vec3 ra = rotate(sa.orientation, o.local_anchor_a);
    const Vec3 rb = rotate(sb.orientation, o.local_anchor_b);
    const Vec3 error = subtract(add(sb.position, rb), add(sa.position, ra));
    const Vec3 old_error = subtract(add(previous[b].position, rotate(previous[b].orientation, o.local_anchor_b)),
        add(previous[a].position, rotate(previous[a].orientation, o.local_anchor_a)));
    const Quaternion fa = normalized_quaternion(quaternion_multiply(sa.orientation, o.local_orientation_a));
    const Quaternion fb = normalized_quaternion(quaternion_multiply(sb.orientation, o.local_orientation_b));
    const Quaternion old_fa = normalized_quaternion(quaternion_multiply(previous[a].orientation, o.local_orientation_a));
    const Quaternion old_fb = normalized_quaternion(quaternion_multiply(previous[b].orientation, o.local_orientation_b));
    const Vec3 twist = relative_rotation_vector(fa, fb);
    const Vec3 old_twist = relative_rotation_vector(old_fa, old_fb);
    const bool swing = o.type == RigidConstraintType::hinge || o.type == RigidConstraintType::piston
        || (o.type == RigidConstraintType::motor && o.motor.angular_enabled);
    const Vec3 free_axis = o.type == RigidConstraintType::hinge ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
    const Vec3 swing_axis = inverse_rotate(fa, rotate(fb, free_axis));
    const Vec3 old_swing_axis = inverse_rotate(old_fa, rotate(old_fb, free_axis));
    const Vec3 rotation = swing ? cross(free_axis, swing_axis) : twist;
    const Vec3 old_rotation = swing ? cross(free_axis, old_swing_axis) : old_twist;
    const float linear_penalty = avbd_pair_stiffness(parameters[a], parameters[b], dt);
    for (unsigned axis = 0; axis < 3; ++axis) {
        const Vec3 n = rotate(fa, basis_axis(axis));
        const Vec3 old_n = rotate(old_fa, basis_axis(axis));
        const bool generic = o.type == RigidConstraintType::generic || o.type == RigidConstraintType::generic_spring;
        const bool motor = o.type == RigidConstraintType::motor;
        for (unsigned angular = 0; angular < 2; ++angular) {
            const bool lock = angular
                ? o.type == RigidConstraintType::fixed || o.type == RigidConstraintType::slider
                    || (o.type == RigidConstraintType::hinge && axis != 2)
                    || (o.type == RigidConstraintType::piston && axis != 0)
                    || (motor && (axis != 0 || !o.motor.angular_enabled))
                : o.type == RigidConstraintType::fixed || o.type == RigidConstraintType::point
                    || o.type == RigidConstraintType::hinge
                    || ((o.type == RigidConstraintType::slider || o.type == RigidConstraintType::piston) && axis != 0)
                    || (motor && (axis != 0 || !o.motor.linear_enabled));
            const auto &limits = angular ? o.angular_limits : o.linear_limits;
            const auto &springs = angular ? o.angular_springs : o.linear_springs;
            const float c = angular ? component(rotation, axis) : dot(error, n);
            const float old_c = angular ? component(old_rotation, axis) : dot(old_error, old_n);
            const Vec3 angular_axis = swing
                ? rotate(fa, subtract(multiply(basis_axis(axis), dot(free_axis, swing_axis)),
                    multiply(free_axis, component(swing_axis, axis))))
                : avbd_log_axis(fa, twist, axis);
            const avbd::Vector6 ja = angular ? avbd_jacobian({}, multiply(angular_axis, -1))
                : avbd_jacobian(multiply(n, -1), add(multiply(cross(ra, n), -1), cross(n, error)));
            const avbd::Vector6 jb = angular ? avbd_jacobian({}, angular_axis) : avbd_jacobian(n, cross(rb, n));
            const float penalty = angular
                ? 1.0F / (fmaxf(dot(n, add(inverse_inertia_world(parameters[a], sa, n),
                    inverse_inertia_world(parameters[b], sb, n))), 1e-6F) * dt * dt)
                : linear_penalty;
            // Off-axis anchor rows couple linear penalty into rotation through
            // their lever arms. COM inertia alone underconditions a hard lock;
            // a factor-two margin bounds two coupled anchor contributions.
            // This parallel-axis floor changes convergence, not physical mass.
            const float lock_penalty = angular ? fmaxf(penalty,
                2 * linear_penalty * fmaxf(length_squared(cross(n, ra)), length_squared(cross(n, rb)))) : penalty;
            if (lock) {
                if constexpr (Visit == avbd_initialize_joint) {
                    if (angular) joint.avbd_rows[index].dual.penalty = fmaxf(
                        joint.avbd_rows[index].dual.penalty, lock_penalty / avbd::gamma);
                }
                avbd_visit_joint_row<Visit>(joint, bodies, block, index, impulse, first,
                    ja, jb, c, old_c, lock_penalty, dt, angular, avbd::maximum_penalty, 0,
                    -avbd::maximum_penalty, avbd::maximum_penalty, false, 0, add(error, ra), rb);
            }
            const bool free_limit = angular
                ? (o.type == RigidConstraintType::hinge && axis == 2) || (o.type == RigidConstraintType::piston && axis == 0)
                : (o.type == RigidConstraintType::slider || o.type == RigidConstraintType::piston) && axis == 0;
            if ((generic || free_limit) && axis_enabled(limits.axes, axis)) {
                const float limit_c = angular ? component(twist, axis) : c;
                const float old_limit_c = angular ? component(old_twist, axis) : old_c;
                const Vec3 limit_axis = avbd_log_axis(fa, twist, axis);
                const auto limit_ja = angular ? avbd_jacobian({}, multiply(limit_axis, -1)) : ja;
                const auto limit_jb = angular ? avbd_jacobian({}, limit_axis) : jb;
                const float lower = component(limits.lower, axis), upper = component(limits.upper, axis);
                if (lower == upper) {
                    const float equal_penalty = angular && generic ? lock_penalty : penalty;
                    if constexpr (Visit == avbd_initialize_joint) {
                        if (angular && generic) joint.avbd_rows[index].dual.penalty = fmaxf(
                            joint.avbd_rows[index].dual.penalty, equal_penalty / avbd::gamma);
                    }
                    avbd_visit_joint_row<Visit>(joint, bodies, block, index, impulse, first,
                        limit_ja, limit_jb, limit_c - lower, old_limit_c - lower, equal_penalty, dt, angular,
                        avbd::maximum_penalty, 0, -avbd::maximum_penalty, avbd::maximum_penalty, false, 0, add(error, ra), rb,
                        avbd::alpha);
                }
                else if (lower < upper) {
                    // Both one-sided bounds remain present even before impact.
                    avbd_visit_joint_row<Visit>(joint, bodies, block, index, impulse, first,
                        limit_ja, limit_jb, limit_c - upper, fmaxf(old_limit_c - upper, 0), penalty, dt, angular,
                        avbd::maximum_penalty, 0, 0, avbd::maximum_penalty, false, 0, add(error, ra), rb,
                        avbd_limit_stabilization);
                    avbd_visit_joint_row<Visit>(joint, bodies, block, index, impulse, first,
                        limit_ja, limit_jb, limit_c - lower, fminf(old_limit_c - lower, 0), penalty, dt, angular,
                        avbd::maximum_penalty, 0, -avbd::maximum_penalty, 0, false, 0, add(error, ra), rb,
                        avbd_limit_stabilization);
                }
            }
            if (o.type == RigidConstraintType::generic_spring && axis_enabled(springs.axes, axis))
                avbd_visit_joint_row<Visit>(joint, bodies, block, index, impulse, first,
                    ja, jb, c, old_c, penalty, dt, angular,
                    component(springs.stiffness, axis), component(springs.damping, axis),
                    -avbd::maximum_penalty, avbd::maximum_penalty, false, 0, add(error, ra), rb);
            if (motor && axis == 0 && (angular ? o.motor.angular_enabled : o.motor.linear_enabled)) {
                const float speed = angular ? o.motor.angular_target_velocity : o.motor.linear_target_velocity;
                const float bound = (angular ? o.motor.angular_maximum_impulse : o.motor.linear_maximum_impulse) / dt;
                // Velocity motor: displacement relative to the old pose, not
                // a positional lock to the joint's original frame.
                const Quaternion relative = quaternion_multiply(conjugate(fa), fb);
                const Quaternion old_relative = quaternion_multiply(conjugate(old_fa), old_fb);
                const Vec3 motor_rotation = relative_rotation_vector({},
                    quaternion_multiply(relative, conjugate(old_relative)));
                const Vec3 motor_axis = avbd_log_axis(fa, motor_rotation, axis);
                const auto motor_ja = angular ? avbd_jacobian({}, multiply(motor_axis, -1)) : ja;
                const auto motor_jb = angular ? avbd_jacobian({}, motor_axis) : jb;
                const float travel = angular ? component(motor_rotation, axis) : c - old_c;
                avbd_visit_joint_row<Visit>(joint, bodies, block, index, impulse, first,
                    motor_ja, motor_jb, travel - speed * dt, 0, penalty, dt, angular,
                    avbd::maximum_penalty, 0, -bound, bound, true, speed, add(error, ra), rb);
            }
        }
    }
    if constexpr (Visit == avbd_initialize_joint) joint.avbd_count = index;
    return impulse;
}

// Preparing the graph is deliberately serial: its stable union/color order is
// part of finite-iteration reproducibility. Cache only its compact metadata,
// not body matrices or contact rows. Larger graphs use the same accessors and
// traversal with global storage beyond each bounded shared-memory prefix.
constexpr unsigned avbd_prepare_body_cache_size = 1024;
constexpr unsigned avbd_prepare_contact_cache_size = 1024;
constexpr unsigned avbd_prepare_joint_cache_size = 128;
struct AvbdPrepareBodyMetadata {
    unsigned parent, color, contact_head, joint_head, flags;
};
struct AvbdPrepareContactMetadata {
    unsigned a, b, next_a, next_b, flags;
};
struct AvbdPrepareJointMetadata {
    unsigned a, b, next_a, next_b, count, iterations;
};
static_assert(sizeof(AvbdPrepareBodyMetadata) * avbd_prepare_body_cache_size +
              sizeof(AvbdPrepareContactMetadata) * avbd_prepare_contact_cache_size +
              sizeof(AvbdPrepareJointMetadata) * avbd_prepare_joint_cache_size <= 48 * 1024);

__device__ bool avbd_prepare_contact_impact(const ContactManifold &manifold,
    const BodyParameters &a, const BodyParameters &b, float dt) {
    bool impact = false;
    for (unsigned point = 0; point < manifold.count; ++point) {
        const auto &c = manifold.contacts[point];
        const float skin = avbd_contact_skin(c, a, b);
        const bool new_impact = (!c.avbd_cached || c.avbd_dual[0].lambda <= 1e-6F
            || (c.impact_fraction > 0 && c.impact_fraction < 1))
            && c.initial_normal_speed < -0.1F && c.penetration >= -k_rigid_surface_tolerance;
        // Repair an initial overlap without turning it into an ejection.
        const float previous_penetration = c.penetration + c.initial_normal_speed * dt;
        impact |= new_impact || previous_penetration > skin + 1e-4F;
    }
    return impact;
}

struct AvbdPrepareGraph {
    AvbdPrepareBodyMetadata *body_cache;
    AvbdPrepareContactMetadata *contact_cache;
    AvbdPrepareJointMetadata *joint_cache;
    AvbdBody *bodies;
    ContactManifold *manifolds;
    RigidConstraintResource *joints;
    const BodyParameters *parameters;
    const unsigned *pairs;
    unsigned body_count;

    __device__ unsigned &parent(unsigned body) {
        return body < avbd_prepare_body_cache_size ? body_cache[body].parent : bodies[body].parent;
    }
    __device__ unsigned &color(unsigned body) {
        return body < avbd_prepare_body_cache_size ? body_cache[body].color : bodies[body].color;
    }
    __device__ unsigned &contact_head(unsigned body) {
        return body < avbd_prepare_body_cache_size ? body_cache[body].contact_head : bodies[body].contact_head;
    }
    __device__ unsigned &joint_head(unsigned body) {
        return body < avbd_prepare_body_cache_size ? body_cache[body].joint_head : bodies[body].joint_head;
    }
    __device__ bool dynamic(unsigned body) const {
        return body < avbd_prepare_body_cache_size ? (body_cache[body].flags & 1U) != 0
            : parameters[body].motion == MotionType::dynamic;
    }
    __device__ unsigned impact(unsigned body) const {
        return body < avbd_prepare_body_cache_size ? (body_cache[body].flags >> 1U) : bodies[body].impact;
    }
    __device__ void set_impact(unsigned body, unsigned value) {
        if (body < avbd_prepare_body_cache_size)
            body_cache[body].flags = (body_cache[body].flags & 1U) | (value << 1U);
        else bodies[body].impact = value;
    }
    __device__ unsigned root(unsigned body) {
        while (body != parent(body)) body = parent(body);
        return body;
    }
    __device__ void unite(unsigned a, unsigned b) {
        if (!dynamic(a) || !dynamic(b)) return;
        a = root(a); b = root(b);
        if (a != b) parent(max(a, b)) = min(a, b);
    }
    __device__ bool contact_nonempty(unsigned active) const {
        return active < avbd_prepare_contact_cache_size ? (contact_cache[active].flags & 1U) != 0
            : manifolds[active].count != 0;
    }
    __device__ unsigned contact_a(unsigned active) const {
        return active < avbd_prepare_contact_cache_size ? contact_cache[active].a : pairs[active] / body_count;
    }
    __device__ unsigned contact_b(unsigned active) const {
        return active < avbd_prepare_contact_cache_size ? contact_cache[active].b : pairs[active] % body_count;
    }
    __device__ unsigned &contact_next(unsigned active, bool first) {
        if (active < avbd_prepare_contact_cache_size)
            return first ? contact_cache[active].next_a : contact_cache[active].next_b;
        return first ? manifolds[active].avbd_next_a : manifolds[active].avbd_next_b;
    }
    __device__ bool contact_impact(unsigned active, unsigned a, unsigned b, float dt) const {
        return active < avbd_prepare_contact_cache_size ? (contact_cache[active].flags & 2U) != 0
            : avbd_prepare_contact_impact(manifolds[active], parameters[a], parameters[b], dt);
    }
    __device__ unsigned joint_count(unsigned index) const {
        return index < avbd_prepare_joint_cache_size ? joint_cache[index].count : joints[index].avbd_count;
    }
    __device__ unsigned joint_iterations(unsigned index) const {
        return index < avbd_prepare_joint_cache_size ? joint_cache[index].iterations : joints[index].options.solver_iterations;
    }
    __device__ unsigned joint_a(unsigned index) const {
        return index < avbd_prepare_joint_cache_size ? joint_cache[index].a : joints[index].geometry.dense_a;
    }
    __device__ unsigned joint_b(unsigned index) const {
        return index < avbd_prepare_joint_cache_size ? joint_cache[index].b : joints[index].geometry.dense_b;
    }
    __device__ unsigned &joint_next(unsigned index, bool first) {
        if (index < avbd_prepare_joint_cache_size)
            return first ? joint_cache[index].next_a : joint_cache[index].next_b;
        return first ? joints[index].avbd_next_a : joints[index].avbd_next_b;
    }
};

__global__ void prepare_avbd_kernel(const BodyParameters *parameters,
    const RigidBodyState *previous, const RigidBodyState *predicted, unsigned count,
    ContactManifold *manifolds, const unsigned *pairs, const unsigned *pair_count,
    RigidConstraintResource *joints, unsigned joint_capacity, const RigidBodyId *ids,
    AvbdBody *bodies, ContactSchedule *schedule, float dt, Vec3 gravity,
    unsigned requested_iterations, unsigned grid_blocks) {
    __shared__ AvbdPrepareBodyMetadata body_cache[avbd_prepare_body_cache_size];
    __shared__ AvbdPrepareContactMetadata contact_cache[avbd_prepare_contact_cache_size];
    __shared__ AvbdPrepareJointMetadata joint_cache[avbd_prepare_joint_cache_size];
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        const Vec3 previous_velocity = bodies[i].previous_velocity;
        const bool history_valid = bodies[i].history_valid != 0;
        bodies[i] = {};
        bodies[i].parent = i;
        bodies[i].target = predicted[i];
        bodies[i].current_orientation = predicted[i].orientation;
        if (i < avbd_prepare_body_cache_size)
            body_cache[i] = {i, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                parameters[i].motion == MotionType::dynamic ? 1U : 0U};
        if (parameters[i].motion != MotionType::dynamic) continue;
        // VBD adaptive primal warm-start: a supported body starts without
        // the new gravity drift, while a freely accelerating body starts at
        // the inertial target. This changes only the initial guess, not the
        // objective or the free-motion result after the block solve.
        const float gravity_squared = length_squared(gravity);
        const Vec3 acceleration = multiply(subtract(previous[i].linear_velocity, previous_velocity), 1 / dt);
        float weight = history_valid && gravity_squared > 0
            ? clamp_scalar(dot(acceleration, gravity) / gravity_squared, 0, 1) : 0;
        if (!isfinite(weight)) weight = 0;
        const Vec3 correction = multiply(gravity, -(1 - weight) * dt * dt);
        bodies[i].displacement.v[0] = correction.x;
        bodies[i].displacement.v[1] = correction.y;
        bodies[i].displacement.v[2] = correction.z;
        bodies[i].previous_velocity = previous[i].linear_velocity;
        bodies[i].history_valid = 1;
        auto &inertia = bodies[i].inertia;
        const float mass = 1.0F / fmaxf(parameters[i].inverse_mass, 1e-12F);
        for (unsigned axis = 0; axis < 3; ++axis) {
            inertia.h[axis][axis] = mass / (dt * dt);
            const Vec3 local_axis = inverse_rotate(predicted[i].orientation, basis_axis(axis));
            const Vec3 column = rotate(predicted[i].orientation, {
                local_axis.x / fmaxf(parameters[i].inverse_inertia_local.x, 1e-12F),
                local_axis.y / fmaxf(parameters[i].inverse_inertia_local.y, 1e-12F),
                local_axis.z / fmaxf(parameters[i].inverse_inertia_local.z, 1e-12F)});
            for (unsigned row = axis; row < 3; ++row)
                inertia.h[3 + row][3 + axis] = component(column, row) / (dt * dt);
        }
    }
    __syncthreads();
    for (unsigned active = threadIdx.x; active < *pair_count; active += blockDim.x) {
        auto &m = manifolds[active];
        const unsigned a = pairs[active] / count, b = pairs[active] % count;
        const float penalty = avbd_pair_stiffness(parameters[a], parameters[b], dt);
        for (unsigned point = 0; point < m.count; ++point) {
            auto &c = m.contacts[point];
            if (!c.avbd_cached) {
                c.avbd_anchor_a = inverse_rotate(previous[a].orientation, subtract(c.point, previous[a].position));
                c.avbd_anchor_b = inverse_rotate(previous[b].orientation, subtract(c.point, previous[b].position));
            }
            const Vec3 motion_a = subtract(predicted[a].position, previous[a].position);
            const Vec3 motion_b = subtract(predicted[b].position, previous[b].position);
            const Vec3 wa = multiply(quaternion_delta_velocity(previous[a].orientation, predicted[a].orientation, dt), dt);
            const Vec3 wb = multiply(quaternion_delta_velocity(previous[b].orientation, predicted[b].orientation, dt), dt);
            const Vec3 ra = subtract(c.point, predicted[a].position), rb = subtract(c.point, predicted[b].position);
            const float closing = dot(c.normal, subtract(add(motion_b, cross(wb, rb)), add(motion_a, cross(wa, ra))));
            const float old_penetration = c.penetration - closing;
            Vec3 t0{}, t1{}; avbd_basis(c.normal, t0, t1);
            const Vec3 anchor_delta = subtract(
                add(predicted[b].position, rotate(predicted[b].orientation, c.avbd_anchor_b)),
                add(predicted[a].position, rotate(predicted[a].orientation, c.avbd_anchor_a)));
            const Vec3 old_anchor_delta = subtract(
                add(previous[b].position, rotate(previous[b].orientation, c.avbd_anchor_b)),
                add(previous[a].position, rotate(previous[a].orientation, c.avbd_anchor_a)));
            c.initial_normal_speed = -closing / dt;
            // Treat the existing rest skin as a permitted slop band. An
            // authored touching face must not gain separation velocity just
            // because broadphase retained a 1 mm collision skin.
            const float skin = avbd_contact_skin(c, parameters[a], parameters[b]);
            c.avbd_error = {c.penetration - clamp_scalar(old_penetration, 0, skin),
                dot(subtract(anchor_delta, multiply(old_anchor_delta, avbd::alpha)), t0),
                dot(subtract(anchor_delta, multiply(old_anchor_delta, avbd::alpha)), t1)};
            for (unsigned axis = 0; axis < 3; ++axis) {
                if (!c.avbd_cached) c.avbd_dual[axis] = {0, penalty};
                c.avbd_dual[axis] = avbd::warm_start(c.avbd_dual[axis]);
            }
        }
        if (active < avbd_prepare_contact_cache_size)
            contact_cache[active] = {a, b, m.avbd_next_a, m.avbd_next_b,
                (m.count != 0 ? 1U : 0U) |
                (avbd_prepare_contact_impact(m, parameters[a], parameters[b], dt) ? 2U : 0U)};
    }
    for (unsigned i = threadIdx.x; i < joint_capacity; i += blockDim.x) {
        auto &joint = joints[i];
        joint.avbd_count = 0;
        if (!joint.alive) continue;
        joint.state.enabled = joint.options.enabled && !joint.state.broken;
        if (!joint.state.enabled) continue;
        joint.state.applied_impulse = 0;
        joint.geometry.dense_a = find_rigid_body_dense(joint.options.body_a, ids, count);
        joint.geometry.dense_b = find_rigid_body_dense(joint.options.body_b, ids, count);
        if (joint.geometry.dense_a == UINT32_MAX || joint.geometry.dense_b == UINT32_MAX) continue;
        avbd::Block scratch{};
        avbd_visit_joint<avbd_initialize_joint>(joint, parameters, previous, bodies, dt, scratch);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < min(joint_capacity, avbd_prepare_joint_cache_size); i += blockDim.x) {
        const auto &j = joints[i];
        joint_cache[i] = {j.geometry.dense_a, j.geometry.dense_b,
            j.avbd_next_a, j.avbd_next_b, j.avbd_count, j.options.solver_iterations};
    }
    __syncthreads();
    if (threadIdx.x == 0) {
    AvbdPrepareGraph graph{body_cache, contact_cache, joint_cache, bodies,
        manifolds, joints, parameters, pairs, count};
    *schedule = {};
    schedule->budget = requested_iterations != 0 ? requested_iterations : avbd::default_iterations;
    schedule->candidates = *pair_count;
    for (unsigned active = 0; active < *pair_count; ++active) {
        if (!graph.contact_nonempty(active)) continue;
        ++schedule->contacts;
        const unsigned a = graph.contact_a(active), b = graph.contact_b(active);
        graph.unite(a, b);
        graph.contact_next(active, true) = graph.contact_head(a); graph.contact_head(a) = active;
        graph.contact_next(active, false) = graph.contact_head(b); graph.contact_head(b) = active;
    }
    for (unsigned i = 0; i < joint_capacity; ++i) {
        if (!graph.joint_count(i)) continue;
        if (!requested_iterations) schedule->budget = max(schedule->budget, graph.joint_iterations(i));
        const unsigned a = graph.joint_a(i), b = graph.joint_b(i);
        graph.unite(a, b);
        graph.joint_next(i, true) = graph.joint_head(a); graph.joint_head(a) = i;
        graph.joint_next(i, false) = graph.joint_head(b); graph.joint_head(b) = i;
    }
    for (unsigned body = 0; body < count; ++body) graph.parent(body) = graph.root(body);
    for (unsigned active = 0; active < *pair_count; ++active) {
        const unsigned a = graph.contact_a(active), b = graph.contact_b(active);
        if (graph.contact_impact(active, a, b, dt)) {
            if (graph.dynamic(a)) graph.set_impact(graph.parent(a), 1);
            if (graph.dynamic(b)) graph.set_impact(graph.parent(b), 1);
            schedule->counts[0] = 1;
        }
    }
    for (unsigned body = 0; body < count; ++body) {
        if (!graph.dynamic(body)) continue;
        graph.set_impact(body, graph.impact(graph.parent(body)));
        if (graph.parent(body) == body && (graph.contact_head(body) != UINT32_MAX || graph.joint_head(body) != UINT32_MAX))
            ++schedule->island_count;
    }
    // Greedy vertex coloring; adjacency lists avoid all-pairs scans during
    // solving. Static neighbors do not require different colors.
    for (unsigned body = 0; body < count; ++body) {
        if (!graph.dynamic(body)) continue;
        unsigned color = 0;
        for (;; ++color) {
            bool conflict = false;
            for (unsigned active = graph.contact_head(body); active != UINT32_MAX;) {
                const unsigned a = graph.contact_a(active), b = graph.contact_b(active);
                const bool first = body == a;
                conflict |= graph.color(first ? b : a) == color;
                active = graph.contact_next(active, first);
            }
            for (unsigned index = graph.joint_head(body); index != UINT32_MAX;) {
                const bool first = body == graph.joint_a(index);
                conflict |= graph.color(first ? graph.joint_b(index) : graph.joint_a(index)) == color;
                index = graph.joint_next(index, first);
            }
            if (!conflict) break;
        }
        graph.color(body) = color;
        schedule->remaining = max(schedule->remaining, color + 1);
    }
    schedule->blocks = grid_blocks;
    // No hidden contact or joint pass outside the reported iteration count.
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < min(count, avbd_prepare_body_cache_size); i += blockDim.x) {
        const auto &cached = body_cache[i];
        bodies[i].parent = cached.parent;
        bodies[i].color = cached.color;
        bodies[i].contact_head = cached.contact_head;
        bodies[i].joint_head = cached.joint_head;
        bodies[i].impact = cached.flags >> 1U;
    }
    for (unsigned i = threadIdx.x; i < min(*pair_count, avbd_prepare_contact_cache_size); i += blockDim.x) {
        manifolds[i].avbd_next_a = contact_cache[i].next_a;
        manifolds[i].avbd_next_b = contact_cache[i].next_b;
    }
    for (unsigned i = threadIdx.x; i < min(joint_capacity, avbd_prepare_joint_cache_size); i += blockDim.x) {
        joints[i].avbd_next_a = joint_cache[i].next_a;
        joints[i].avbd_next_b = joint_cache[i].next_b;
    }
}

__device__ void avbd_barrier() {
    if (gridDim.x == 1) __syncthreads();
    else cooperative_groups::this_grid().sync();
}

__device__ void avbd_minimize_free_translation(const BodyParameters *parameters,
    unsigned count, const ContactManifold *manifolds, const unsigned *pairs,
    const RigidConstraintResource *joints, AvbdBody *bodies, bool velocity_phase) {
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    // A common translation is an exact null mode of every internal contact
    // and joint. Minimize the SAME inertial energy along that mode to remove
    // finite-sweep Gauss-Seidel COM drift. Never change an island that can
    // exchange momentum with a static/kinematic endpoint.
    // One thread reduces each root in body-ID order: no nondeterministic
    // floating atomics, extra buffers, or contact/velocity solver are needed.
    for (unsigned root = lane; root < count; root += stride) {
        if (parameters[root].motion != MotionType::dynamic || bodies[root].parent != root) continue;
        for (unsigned axis = 0; axis < 3; ++axis) bodies[root].inertia.g[axis] = 0;
        if ((velocity_phase && !bodies[root].impact)
            || (bodies[root].contact_head == UINT32_MAX && bodies[root].joint_head == UINT32_MAX)) continue;
        Vec3 weighted{};
        float total_mass = 0;
        bool anchored = false;
        for (unsigned body = 0; body < count && !anchored; ++body) {
            if (parameters[body].motion != MotionType::dynamic || bodies[body].parent != root) continue;
            const float mass = 1 / fmaxf(parameters[body].inverse_mass, 1e-12F);
            weighted = add(weighted, multiply(avbd_translation(bodies[body].displacement), mass));
            total_mass += mass;
            for (unsigned active = bodies[body].contact_head; active != UINT32_MAX;) {
                const auto &m = manifolds[active];
                const unsigned a = pairs[active] / count, b = pairs[active] % count;
                const bool first = body == a;
                anchored |= parameters[first ? b : a].motion != MotionType::dynamic;
                active = first ? m.avbd_next_a : m.avbd_next_b;
            }
            for (unsigned index = bodies[body].joint_head; index != UINT32_MAX;) {
                const auto &j = joints[index];
                const bool first = body == j.geometry.dense_a;
                // A joint broken during this phase may already have applied
                // an external reaction; retain that reaction as well.
                anchored |= parameters[first ? j.geometry.dense_b : j.geometry.dense_a].motion != MotionType::dynamic;
                index = first ? j.avbd_next_a : j.avbd_next_b;
            }
        }
        if (!anchored && total_mass > 0) {
            weighted = multiply(weighted, 1 / total_mass);
            bodies[root].inertia.g[0] = weighted.x;
            bodies[root].inertia.g[1] = weighted.y;
            bodies[root].inertia.g[2] = weighted.z;
        }
    }
    avbd_barrier();
    for (unsigned body = lane; body < count; body += stride) {
        if (parameters[body].motion != MotionType::dynamic || (velocity_phase && !bodies[body].impact)) continue;
        const auto &mean = bodies[bodies[body].parent].inertia;
        for (unsigned axis = 0; axis < 3; ++axis) bodies[body].displacement.v[axis] -= mean.g[axis];
    }
    avbd_barrier();
    // Restore the zero inertial-gradient template before the impact phase.
    for (unsigned body = lane; body < count; body += stride)
        for (unsigned axis = 0; axis < 3; ++axis) bodies[body].inertia.g[axis] = 0;
    avbd_barrier();
}

__global__ void solve_avbd_kernel(const BodyParameters *parameters,
    const RigidBodyState *previous, RigidBodyState *states, unsigned count,
    ContactManifold *manifolds, const unsigned *pairs, const unsigned *pair_count,
    RigidConstraintResource *joints, unsigned joint_capacity,
    AvbdBody *bodies, ContactSchedule *schedule, float dt) {
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = blockDim.x * gridDim.x;
    // Impact velocity stabilization uses the SAME body blocks and dual loop.
    // Phase 0 solves poses; phase 1 (only on impact) solves u=h*v increments,
    // retaining total force bounds so friction/motor budgets aren't doubled.
    for (unsigned phase = 0; phase < 2; ++phase) {
    if (phase != 0) {
        if (!schedule->counts[0]) break;
        for (unsigned active = lane; active < *pair_count; active += stride) {
            auto &m = manifolds[active];
            const unsigned a = pairs[active] / count, b = pairs[active] % count;
            if (!bodies[a].impact && !bodies[b].impact) continue;
            for (unsigned point = 0; point < m.count; ++point) {
                auto &c = m.contacts[point];
                avbd::Vector6 ja{}, jb{};
                avbd_contact_normal_row(c, a, b, bodies, ja, jb);
                const float gap = fmaxf(0, -(c.penetration + avbd::dot6(ja, bodies[a].displacement)
                    + avbd::dot6(jb, bodies[b].displacement)));
                c.avbd_error.x = gap;
            }
        }
        avbd_barrier();
        for (unsigned body = lane; body < count; body += stride) {
            if (bodies[body].impact || parameters[body].motion != MotionType::dynamic) {
                bodies[body].target = states[body]; bodies[body].displacement = {};
                bodies[body].current_orientation = states[body].orientation;
            }
        }
        avbd_barrier();
        for (unsigned index = lane; index < joint_capacity; index += stride) {
            auto &j = joints[index]; if (!j.avbd_count || !j.state.enabled) continue;
            const unsigned a = j.geometry.dense_a, b = j.geometry.dense_b;
            if (!bodies[a].impact && !bodies[b].impact) continue;
            avbd::Block scratch{};
            avbd_visit_joint<avbd_impact_joint>(j, parameters, previous, bodies, dt, scratch);
        }
        for (unsigned active = lane; active < *pair_count; active += stride) {
            auto &m = manifolds[active];
            const unsigned a = pairs[active] / count, b = pairs[active] % count;
            if (!bodies[a].impact && !bodies[b].impact) continue;
            const auto va = avbd_jacobian(multiply(states[a].linear_velocity, dt), multiply(states[a].angular_velocity, dt));
            const auto vb = avbd_jacobian(multiply(states[b].linear_velocity, dt), multiply(states[b].angular_velocity, dt));
            for (unsigned point = 0; point < m.count; ++point) {
                auto &c = m.contacts[point];
                const float gap = c.avbd_error.x;
                avbd::Vector6 ja[3]{}, jb[3]{}; float error[3]{};
                avbd_contact_rows(c, a, b, bodies, ja, jb, error);
                const float restitution = !c.avbd_cached && c.initial_normal_speed < -1.0F
                    ? fminf(parameters[a].restitution, parameters[b].restitution) : 0;
                c.avbd_error = {
                    avbd::dot6(ja[0], va) + avbd::dot6(jb[0], vb) - gap - restitution * c.initial_normal_speed * dt,
                    avbd::dot6(ja[1], va) + avbd::dot6(jb[1], vb),
                    avbd::dot6(ja[2], va) + avbd::dot6(jb[2], vb)};
            }
        }
        avbd_barrier();
    }
    for (unsigned iteration = 0; iteration < schedule->budget; ++iteration) {
        for (unsigned color = 0; color < schedule->remaining; ++color) {
            for (unsigned body = lane; body < count; body += stride) {
                if (bodies[body].color != color) continue;
                if (phase && !bodies[body].impact) continue;
                auto block = bodies[body].inertia;
                const auto displacement = bodies[body].displacement;
                for (unsigned i = 0; i < 6; ++i)
                    for (unsigned j = 0; j < 6; ++j)
                        block.g[i] += block.h[i > j ? i : j][i > j ? j : i] * displacement.v[j];
                for (unsigned active = bodies[body].contact_head; active != UINT32_MAX;) {
                    const auto &m = manifolds[active];
                    const unsigned a = pairs[active] / count, b = pairs[active] % count;
                    const bool first = body == a;
                    const float friction = sqrtf(parameters[a].friction * parameters[b].friction);
                    for (unsigned point = 0; point < m.count; ++point) {
                        const auto &c = m.contacts[point];
                        avbd::Vector6 ja[3]{}, jb[3]{}; float error[3]{};
                        avbd_contact_rows(c, a, b, bodies, ja, jb, error);
                        const auto trial = avbd_contact_trial(c, error);
                        const auto force = avbd::project_contact(trial, friction);
                        const auto scales = avbd::contact_stiffness_scales(trial, friction);
                        const float stiffness[3]{scales.normal, scales.tangent0, scales.tangent1};
                        Vec3 t0{}, t1{}; avbd_basis(c.normal, t0, t1);
                        const float f[3]{force.normal - (phase ? c.accumulated_normal_impulse / dt : 0),
                            force.tangent0 - (phase ? dot(c.accumulated_friction_impulse, t0) / dt : 0),
                            force.tangent1 - (phase ? dot(c.accumulated_friction_impulse, t1) / dt : 0)};
                        for (unsigned axis = 0; axis < 3; ++axis)
                            avbd::add_row(block, first ? ja[axis] : jb[axis], c.avbd_dual[axis].penalty * stiffness[axis], f[axis]);
                    }
                    active = first ? m.avbd_next_a : m.avbd_next_b;
                }
                for (unsigned index = bodies[body].joint_head; index != UINT32_MAX;) {
                    auto &j = joints[index];
                    const unsigned a = j.geometry.dense_a, b = j.geometry.dense_b;
                    const bool first = body == a;
                    if (j.state.enabled) {
                        if (!phase) {
                            avbd_visit_joint<avbd_accumulate_joint>(j, parameters, previous, bodies, dt, block, first);
                        } else {
                            for (unsigned row = 0; row < j.avbd_count; ++row)
                                if (j.avbd_rows[row].stiffness >= avbd::maximum_penalty)
                                    avbd::accumulate(block, j.avbd_rows[row], first,
                                        bodies[a].displacement, bodies[b].displacement, dt);
                        }
                    }
                    index = first ? j.avbd_next_a : j.avbd_next_b;
                }
                avbd::Vector6 update{};
                if (avbd::solve(block, update)) {
                    {
                        // A full Newton step can activate an inactive corner
                        // and alternate contact sets indefinitely. Safeguard
                        // rows crossed by this proposed step, then solve the
                        // same body block once more on their active side.
                        // Include the affine force as well as its Hessian:
                        // Hessian-only damping creates artificial pose gaps.
                        // Separating rows remain completely inactive.
                        bool crossing = false;
                        for (unsigned active = bodies[body].contact_head; active != UINT32_MAX;) {
                            const auto &m = manifolds[active];
                            const unsigned a = pairs[active] / count, b = pairs[active] % count;
                            const bool first = body == a;
                            for (unsigned point = 0; point < m.count; ++point) {
                                const auto &c = m.contacts[point];
                                avbd::Vector6 ja{}, jb{};
                                const float error = avbd_contact_normal_row(c, a, b, bodies, ja, jb);
                                const float trial = c.avbd_dual[0].lambda + c.avbd_dual[0].penalty * error;
                                const auto normal_row = first ? ja : jb;
                                // Ignore roundoff-sized boundary crossings.
                                if (trial <= 0 && trial + c.avbd_dual[0].penalty * avbd::dot6(normal_row, update)
                                    > c.avbd_dual[0].penalty * 1.0e-6F) {
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
                    if (!phase) avbd_apply_pose_update(bodies[body], update);
                    else for (unsigned i = 0; i < 6; ++i) bodies[body].displacement.v[i] += update.v[i];
                }
            }
            avbd_barrier();
        }
        for (unsigned active = lane; active < *pair_count; active += stride) {
            auto &m = manifolds[active];
            const unsigned a = pairs[active] / count, b = pairs[active] % count;
            if (phase && !bodies[a].impact && !bodies[b].impact) continue;
            const float friction = sqrtf(parameters[a].friction * parameters[b].friction);
            const float beta = avbd_pair_stiffness(parameters[a], parameters[b], dt) * 100.0F;
            for (unsigned point = 0; point < m.count; ++point) {
                auto &c = m.contacts[point];
                avbd::Vector6 ja[3]{}, jb[3]{}; float error[3]{};
                avbd_contact_rows(c, a, b, bodies, ja, jb, error);
                const auto force = avbd_contact_force(c, error, friction);
                const float tangential = hypotf(c.avbd_dual[1].lambda + c.avbd_dual[1].penalty * error[1],
                    c.avbd_dual[2].lambda + c.avbd_dual[2].penalty * error[2]);
                const bool stick = tangential <= friction * force.normal;
                c.avbd_dual[0].lambda = force.normal;
                c.avbd_dual[1].lambda = force.tangent0;
                c.avbd_dual[2].lambda = force.tangent1;
                for (unsigned axis = 0; axis < 3; ++axis)
                    if (axis == 0 ? force.normal > 0 : stick)
                        c.avbd_dual[axis].penalty = fminf(avbd::maximum_penalty,
                            c.avbd_dual[axis].penalty + beta * fabsf(error[axis]));
                if (iteration + 1 == schedule->budget) {
                    Vec3 t0{}, t1{}; avbd_basis(c.normal, t0, t1);
                    c.accumulated_normal_impulse = c.reported_normal_impulse = force.normal * dt;
                    c.accumulated_friction_impulse = c.reported_friction_impulse = multiply(add(
                        multiply(t0, force.tangent0), multiply(t1, force.tangent1)), dt);
                    // Sliding contacts start a fresh static-friction anchor.
                    if (!stick) {
                        const auto pose_a = phase ? bodies[a].target : avbd_pose(bodies[a]);
                        const auto pose_b = phase ? bodies[b].target : avbd_pose(bodies[b]);
                        c.avbd_anchor_a = inverse_rotate(pose_a.orientation, subtract(c.point, pose_a.position));
                        c.avbd_anchor_b = inverse_rotate(pose_b.orientation, subtract(c.point, pose_b.position));
                    }
                }
            }
        }
        for (unsigned index = lane; index < joint_capacity; index += stride) {
            auto &j = joints[index]; if (!j.avbd_count || !j.state.enabled) continue;
            const unsigned a = j.geometry.dense_a, b = j.geometry.dense_b;
            if (phase && !bodies[a].impact && !bodies[b].impact) continue;
            float impulse = 0;
            if (!phase) {
                avbd::Block scratch{};
                impulse = avbd_visit_joint<avbd_advance_joint>(j, parameters, previous, bodies, dt, scratch);
            } else {
                for (unsigned row = 0; row < j.avbd_count; ++row) {
                    impulse += fabsf(j.avbd_rows[row].stiffness < avbd::maximum_penalty
                        ? j.avbd_rows[row].reference_force
                        : avbd::row_force(j.avbd_rows[row], bodies[a].displacement, bodies[b].displacement, dt)) * dt;
                    if (j.avbd_rows[row].stiffness >= avbd::maximum_penalty)
                        j.avbd_rows[row] = avbd::advance(j.avbd_rows[row], bodies[a].displacement, bodies[b].displacement);
                }
            }
            j.state.applied_impulse = impulse;
            if (j.options.breaking_impulse_threshold > 0 && impulse > j.options.breaking_impulse_threshold) {
                j.state.broken = true; j.state.enabled = false;
            }
        }
        avbd_barrier();
    }
    avbd_minimize_free_translation(parameters, count, manifolds, pairs, joints, bodies, phase != 0);
    for (unsigned body = lane; body < count; body += stride) {
        if (parameters[body].motion != MotionType::dynamic) continue;
        if (phase && !bodies[body].impact) continue;
        auto state = bodies[body].target;
        const auto delta = bodies[body].displacement;
        if (phase) {
            state.linear_velocity = add(state.linear_velocity, multiply(avbd_translation(delta), 1 / dt));
            state.angular_velocity = add(state.angular_velocity, multiply(avbd_rotation(delta), 1 / dt));
            states[body] = state;
            continue;
        }
        state.position = add(state.position, avbd_translation(delta));
        state.orientation = bodies[body].current_orientation;
        state.linear_velocity = add(state.linear_velocity, multiply(avbd_translation(delta), 1 / dt));
        state.angular_velocity = add(state.angular_velocity,
            quaternion_delta_velocity(bodies[body].target.orientation, state.orientation, dt));
        states[body] = state;
    }
    if (lane == 0) schedule->maximum_passes += schedule->budget;
    avbd_barrier();
    }
}
