// SPDX-License-Identifier: MIT
#include "dispatch_shape.slangh"
// Complete AVBD rigid contact and joint schedule. The numerical block/dual
// core is shared with the cross-target conformance runner.

#ifndef PM_GEOMETRY_CONSTRAINTS_ADAPTER
#define PM_GEOMETRY_CONSTRAINTS_ADAPTER 0
#endif
#ifndef PM_TARGET_D3D12
#define PM_TARGET_D3D12 0
#endif

#include "constraint_math.slang"

#if PM_GEOMETRY_CONSTRAINTS_ADAPTER
#include "geometry.slang"

#define PM_AVBD_BUFFER(T) PMBuffer<T>
#define PM_AVBD_MANIFOLD_BUFFER PMBuffer<ContactManifold>
#define PM_AVBD_MANIFOLD_COUNT(buffer, index) buffer[index].count
#define PM_AVBD_MANIFOLD_NEXT_A(buffer, index) buffer[index].avbd_next_a
#define PM_AVBD_MANIFOLD_NEXT_B(buffer, index) buffer[index].avbd_next_b
#define PM_AVBD_MANIFOLD_CONTACT(buffer, index, point) \
    buffer[index].contacts[point]

static const float kRigidSurfaceTolerance = 1.0e-5f;
static const float kRigidRestOffset = 1.0e-3f;
static const float kRigidGuidedRestOffset = 1.0e-4f;
static const float kAvbdLimitStabilization = 0.8f;
static const uint kRigidRealtimeProfileVersion = 1u;
static const uint kRigidRealtimeSweeps = 8u;
static const float kRigidRealtimeSolverDamping = 0.05f;

static const uint kAvbdInitializeJoint = 0u;
static const uint kAvbdAccumulateJoint = 1u;
static const uint kAvbdAdvanceJoint = 2u;
static const uint kAvbdImpactJoint = 3u;

struct AvbdBody
{
    RigidBodyState target;
    Quaternion orientation_delta;
    AvbdVector6 displacement;
    AvbdBlock inertia;
    uint contact_head;
    uint joint_head;
    uint color;
    uint parent;
    uint impact;
    Vec3 previous_velocity;
    uint history_valid;
};

float rigid_avbd_component(Vec3 value, uint axis)
{
    return axis == 0u ? value.x : (axis == 1u ? value.y : value.z);
}

Vec3 rigid_avbd_basis_axis(uint axis)
{
    return axis == 0u ? Vec3(1.0f, 0.0f, 0.0f) :
        (axis == 1u ? Vec3(0.0f, 1.0f, 0.0f) :
            Vec3(0.0f, 0.0f, 1.0f));
}

Vec3 rigid_avbd_contact_delta_velocity(Quaternion from, Quaternion to,
    float timestep)
{
    return quaternion_delta_velocity(from, to, timestep);
}

bool rigid_avbd_axis_enabled(uint mask, uint axis)
{
    return (mask & (1u << axis)) != 0u;
}

Vec3 rigid_avbd_relative_rotation_vector(Quaternion frameA,
    Quaternion frameB)
{
    return quaternion_rotation_vector(relative_quaternion(frameA, frameB, true));
}

uint rigid_avbd_find_dense(RigidBodyId id, PM_AVBD_BUFFER(RigidBodyId) ids,
    uint count)
{
    for (uint index = 0u; index < count; ++index)
        if (ids[index].index == id.index &&
            ids[index].generation == id.generation) return index;
    return kInvalidDense;
}

bool rigid_avbd_guided_contact(Contact contact)
{
    return (contact.body_hinge.axial != 0u &&
                contact.collider_hinge.static_body != 0u) ||
           (contact.collider_hinge.axial != 0u &&
                contact.body_hinge.static_body != 0u);
}

float rigid_avbd_rest_offset(float margin)
{
    return min(margin, kRigidRestOffset);
}

AvbdVector6 rigid_avbd_jacobian(Vec3 linear, Vec3 angular)
{
    AvbdVector6 result = avbdZeroVector6();
    result.values[0] = linear.x;
    result.values[1] = linear.y;
    result.values[2] = linear.z;
    result.values[3] = angular.x;
    result.values[4] = angular.y;
    result.values[5] = angular.z;
    return result;
}

Vec3 rigid_avbd_translation(AvbdVector6 value)
{
    return {value.values[0], value.values[1], value.values[2]};
}

Vec3 rigid_avbd_rotation(AvbdVector6 value)
{
    return {value.values[3], value.values[4], value.values[5]};
}

RigidBodyState rigid_avbd_pose(AvbdBody body)
{
    RigidBodyState state = body.target;
    state.position = add(state.position,
        rigid_avbd_translation(body.displacement));
    Quaternion delta = body.orientation_delta;
    if (delta.x != 0.0f || delta.y != 0.0f || delta.z != 0.0f)
        state.orientation = normalized_quaternion(
            quaternion_multiply(delta, state.orientation));
    return state;
}

void rigid_avbd_apply_pose_update(inout AvbdBody body, AvbdVector6 update)
{
    for (uint axis = 0u; axis < 3u; ++axis)
        body.displacement.values[axis] += update.values[axis];
    Vec3 rotation = rigid_avbd_rotation(update);
    if (rotation.x == 0.0f && rotation.y == 0.0f && rotation.z == 0.0f)
        return;
    // Accumulate the small correction around identity. Updating a world-space
    // orientation and subtracting the target after every iteration loses the
    // small correction before it can contribute to angular velocity.
    Quaternion relative = body.orientation_delta;
    Quaternion derivative = quaternion_multiply(
        {rotation.x, rotation.y, rotation.z, 0.0f},
        relative);
    relative = normalized_quaternion({
        relative.x + 0.5f * derivative.x,
        relative.y + 0.5f * derivative.y,
        relative.z + 0.5f * derivative.z,
        relative.w + 0.5f * derivative.w});
    if (relative.w < 0.0f)
        relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    body.orientation_delta = relative;
    body.displacement.values[3] = 2.0f * relative.x;
    body.displacement.values[4] = 2.0f * relative.y;
    body.displacement.values[5] = 2.0f * relative.z;
}

Vec3 rigid_avbd_log_axis(Quaternion frame, Vec3 logarithm, uint axis)
{
    float squared = length_squared(logarithm);
    // 1 - (angle/2) * cot(angle/2) loses its significant bits near zero.
    // The series remains accurate throughout the small-angle region.
    float coefficient = 1.0f / 12.0f + squared * (1.0f / 720.0f + squared *
        (1.0f / 30240.0f + squared * (1.0f / 1209600.0f +
        squared / 47900160.0f)));
    // Through one radian the omitted term is below float roundoff. Evaluating
    // 1-half*cot(half) directly loses significant bits well beyond 0.1 rad.
    if (squared > 1.0f)
    {
        float angle = sqrt(squared);
        float half = 0.5f * angle;
        coefficient = (1.0f - half * cos(half) / sin(half)) / squared;
    }
    Vec3 unit = rigid_avbd_basis_axis(axis);
    Vec3 local = add(add(unit,
        multiply(cross(logarithm, unit), 0.5f)),
        multiply(cross(logarithm, cross(logarithm, unit)), coefficient));
    return rotate(frame, local);
}

void rigid_avbd_basis(Vec3 normal, out Vec3 tangent0, out Vec3 tangent1)
{
    tangent0 = normalized_or(cross(normal, abs(normal.x) < 0.57735f
        ? Vec3(1.0f, 0.0f, 0.0f) : Vec3(0.0f, 1.0f, 0.0f)),
        {0.0f, 0.0f, 1.0f});
    tangent1 = cross(normal, tangent0);
}

float rigid_avbd_pair_stiffness(BodyParameters a, BodyParameters b, float dt)
{
    return 1.0f / (max(a.inverse_mass + b.inverse_mass, 1.0e-6f) *
        dt * dt);
}

float rigid_avbd_contact_skin(Contact contact, BodyParameters a,
    BodyParameters b)
{
    float offset = rigid_avbd_guided_contact(contact)
        ? min(a.collision_margin + b.collision_margin,
            kRigidGuidedRestOffset)
        : rigid_avbd_rest_offset(a.collision_margin + b.collision_margin);
    return offset + 2.0f * kRigidSurfaceTolerance;
}

void rigid_avbd_contact_rows(Contact contact, uint a, uint b,
    PM_AVBD_BUFFER(AvbdBody) bodies, out AvbdVector6 jacobianA[3],
    out AvbdVector6 jacobianB[3], out float error[3])
{
    Vec3 tangent0;
    Vec3 tangent1;
    rigid_avbd_basis(contact.normal, tangent0, tangent1);
    Vec3 axes[3] = {contact.normal, tangent0, tangent1};
    Vec3 armA = subtract(contact.point, bodies[a].target.position);
    Vec3 armB = subtract(contact.point, bodies[b].target.position);
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        Vec3 ra = axis == 0u ? armA : rotate(
            bodies[a].target.orientation, contact.avbd_anchor_a);
        Vec3 rb = axis == 0u ? armB : rotate(
            bodies[b].target.orientation, contact.avbd_anchor_b);
        jacobianA[axis] = rigid_avbd_jacobian(multiply(axes[axis], -1.0f),
            multiply(cross(ra, axes[axis]), -1.0f));
        jacobianB[axis] = rigid_avbd_jacobian(
            axes[axis], cross(rb, axes[axis]));
        error[axis] = rigid_avbd_component(contact.avbd_error, axis) +
            avbdDot6(jacobianA[axis], bodies[a].displacement) +
            avbdDot6(jacobianB[axis], bodies[b].displacement);
    }
}

float rigid_avbd_contact_normal_row(Contact contact, uint a, uint b,
    PM_AVBD_BUFFER(AvbdBody) bodies, out AvbdVector6 jacobianA,
    out AvbdVector6 jacobianB)
{
    Vec3 armA = subtract(contact.point, bodies[a].target.position);
    Vec3 armB = subtract(contact.point, bodies[b].target.position);
    jacobianA = rigid_avbd_jacobian(multiply(contact.normal, -1.0f),
        multiply(cross(armA, contact.normal), -1.0f));
    jacobianB = rigid_avbd_jacobian(
        contact.normal, cross(armB, contact.normal));
    return contact.avbd_error.x +
        avbdDot6(jacobianA, bodies[a].displacement) +
        avbdDot6(jacobianB, bodies[b].displacement);
}

AvbdContactForce rigid_avbd_contact_trial(Contact contact, float error[3])
{
    AvbdContactForce result;
    result.normal = contact.avbd_dual[0].lambda +
        contact.avbd_dual[0].penalty * error[0];
    result.tangent0 = contact.avbd_dual[1].lambda +
        contact.avbd_dual[1].penalty * error[1];
    result.tangent1 = contact.avbd_dual[2].lambda +
        contact.avbd_dual[2].penalty * error[2];
    return result;
}

AvbdContactForce rigid_avbd_contact_force(Contact contact, float error[3],
    float friction)
{
    return avbdProjectContact(
        rigid_avbd_contact_trial(contact, error), friction);
}

AvbdRow rigid_avbd_row(RigidAvbdRow source)
{
    AvbdRow result;
    for (uint axis = 0u; axis < 6u; ++axis)
    {
        result.a.values[axis] = source.a.values[axis];
        result.b.values[axis] = source.b.values[axis];
    }
    result.dual = {source.dual.lambda, source.dual.penalty};
    result.error = source.error;
    result.velocity = source.velocity;
    result.lower = source.lower;
    result.upper = source.upper;
    result.stiffness = source.stiffness;
    result.damping = source.damping;
    result.beta = source.beta;
    result.referenceForce = source.reference_force;
    return result;
}

RigidAvbdRow rigid_avbd_stored_row(AvbdRow source)
{
    RigidAvbdRow result;
    for (uint axis = 0u; axis < 6u; ++axis)
    {
        result.a.values[axis] = source.a.values[axis];
        result.b.values[axis] = source.b.values[axis];
    }
    result.dual = {source.dual.lambda, source.dual.penalty};
    result.error = source.error;
    result.velocity = source.velocity;
    result.lower = source.lower;
    result.upper = source.upper;
    result.stiffness = source.stiffness;
    result.damping = source.damping;
    result.beta = source.beta;
    result.reference_force = source.referenceForce;
    return result;
}

void rigid_avbd_visit_joint_row(inout RigidConstraintResource joint,
    PM_AVBD_BUFFER(AvbdBody) bodies, inout AvbdBlock block, inout uint index,
    inout float impulse, bool first, AvbdVector6 jacobianA,
    AvbdVector6 jacobianB, float current, float previous, float penalty,
    float dt, bool angular, float stiffness, float damping, float lower,
    float upper, bool motor, float motorSpeed, Vec3 curvatureA,
    Vec3 curvatureB, float stabilization, uint visit)
{
    uint rowIndex = index++;
    AvbdRow row = rigid_avbd_row(joint.avbd_rows[rowIndex]);
    uint a = joint.geometry.dense_a;
    uint b = joint.geometry.dense_b;
    float linearized = avbdDot6(jacobianA, bodies[a].displacement) +
        avbdDot6(jacobianB, bodies[b].displacement);
    row.a = jacobianA;
    row.b = jacobianB;
    row.stiffness = stiffness;
    row.damping = damping;
    if (visit == kAvbdInitializeJoint)
    {
        if (!(row.dual.penalty > kAvbdMinimumPenalty))
            row.dual.penalty = penalty;
        row.dual = avbdWarmStart(row.dual, stiffness);
        if (stiffness < kAvbdMaximumPenalty) row.dual.lambda = 0.0f;
        row.beta = penalty * (angular ? 10.0f : 100.0f);
        row.referenceForce = 0.0f;
    }
    row.error = current - (stiffness < kAvbdMaximumPenalty
        ? 0.0f : stabilization * previous) - linearized;
    row.velocity = motor ? motorSpeed :
        (current - previous - linearized) / dt;
    row.lower = lower;
    row.upper = upper;
    if (visit == kAvbdAccumulateJoint)
    {
        avbdAccumulate(block, row, first, bodies[a].displacement,
            bodies[b].displacement, dt);
        if (!angular)
        {
            Vec3 axis = rigid_avbd_translation(jacobianB);
            Vec3 arm = first ? curvatureA : curvatureB;
            float force = avbdRowForce(row, bodies[a].displacement,
                bodies[b].displacement, dt);
            for (uint column = 0u; column < 3u; ++column)
            {
                Vec3 value = subtract(multiply(add(
                    multiply(axis, rigid_avbd_component(arm, column)),
                    multiply(arm, rigid_avbd_component(axis, column))),
                    0.5f), multiply(rigid_avbd_basis_axis(column),
                        dot(axis, arm)));
                block.h[3u + column][3u + column] +=
                    abs(force) * sqrt(length_squared(value));
            }
        }
    }
    else if (visit == kAvbdAdvanceJoint)
    {
        impulse += abs(avbdRowForce(row, bodies[a].displacement,
            bodies[b].displacement, dt)) * dt;
        joint.avbd_rows[rowIndex] = rigid_avbd_stored_row(
            avbdAdvance(row, bodies[a].displacement, bodies[b].displacement));
    }
    else if (visit == kAvbdImpactJoint)
    {
        if (stiffness < kAvbdMaximumPenalty)
            row.referenceForce = avbdRowForce(row,
                bodies[a].displacement, bodies[b].displacement, dt);
        else
        {
            float gap = lower == 0.0f ? min(0.0f, current) :
                (upper == 0.0f ? max(0.0f, current) : 0.0f);
            AvbdVector6 velocityA = rigid_avbd_jacobian(
                multiply(bodies[a].target.linear_velocity, dt),
                multiply(bodies[a].target.angular_velocity, dt));
            AvbdVector6 velocityB = rigid_avbd_jacobian(
                multiply(bodies[b].target.linear_velocity, dt),
                multiply(bodies[b].target.angular_velocity, dt));
            row.referenceForce = row.dual.lambda;
            row.error = avbdDot6(jacobianA, velocityA) +
                avbdDot6(jacobianB, velocityB) + gap -
                (motor ? motorSpeed * dt : 0.0f);
        }
        joint.avbd_rows[rowIndex] = rigid_avbd_stored_row(row);
    }
    else
        joint.avbd_rows[rowIndex] = rigid_avbd_stored_row(row);
}

float rigid_avbd_visit_joint(inout RigidConstraintResource joint,
    PM_AVBD_BUFFER(BodyParameters) parameters,
    PM_AVBD_BUFFER(RigidBodyState) previous,
    PM_AVBD_BUFFER(AvbdBody) bodies,
    float dt, inout AvbdBlock block, bool first, uint visit)
{
    uint index = 0u;
    float impulse = 0.0f;
    RigidConstraintOptions options = joint.options;
    uint a = joint.geometry.dense_a;
    uint b = joint.geometry.dense_b;
    RigidBodyState stateA = rigid_avbd_pose(bodies[a]);
    RigidBodyState stateB = rigid_avbd_pose(bodies[b]);
    Vec3 armA = rotate(stateA.orientation, options.local_anchor_a);
    Vec3 armB = rotate(stateB.orientation, options.local_anchor_b);
    Vec3 error = subtract(add(stateB.position, armB),
        add(stateA.position, armA));
    Vec3 oldError = subtract(add(previous[b].position,
        rotate(previous[b].orientation, options.local_anchor_b)),
        add(previous[a].position,
            rotate(previous[a].orientation, options.local_anchor_a)));
    Quaternion frameA = normalized_quaternion(quaternion_multiply(
        stateA.orientation, options.local_orientation_a));
    Quaternion frameB = normalized_quaternion(quaternion_multiply(
        stateB.orientation, options.local_orientation_b));
    Quaternion oldFrameA = normalized_quaternion(quaternion_multiply(
        previous[a].orientation, options.local_orientation_a));
    Quaternion oldFrameB = normalized_quaternion(quaternion_multiply(
        previous[b].orientation, options.local_orientation_b));
    Vec3 twist = rigid_avbd_relative_rotation_vector(frameA, frameB);
    Vec3 oldTwist = rigid_avbd_relative_rotation_vector(
        oldFrameA, oldFrameB);
    bool swing = options.type == RigidConstraintType.hinge ||
        options.type == RigidConstraintType.piston ||
        (options.type == RigidConstraintType.motor &&
            options.motor.angular_enabled != 0u);
    Vec3 freeAxis = options.type == RigidConstraintType.hinge
        ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 swingAxis = inverse_rotate(frameA, rotate(frameB, freeAxis));
    Vec3 oldSwingAxis = inverse_rotate(
        oldFrameA, rotate(oldFrameB, freeAxis));
    Vec3 rotation = swing ? cross(freeAxis, swingAxis) : twist;
    Vec3 oldRotation = swing ? cross(freeAxis, oldSwingAxis) : oldTwist;
    float linearPenalty = rigid_avbd_pair_stiffness(
        parameters[a], parameters[b], dt);
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        Vec3 direction = rotate(frameA, rigid_avbd_basis_axis(axis));
        Vec3 oldDirection = rotate(
            oldFrameA, rigid_avbd_basis_axis(axis));
        bool generic = options.type == RigidConstraintType.generic ||
            options.type == RigidConstraintType.generic_spring;
        bool motorType = options.type == RigidConstraintType.motor;
        for (uint angularValue = 0u; angularValue < 2u; ++angularValue)
        {
            bool angular = angularValue != 0u;
            bool lock = angular
                ? options.type == RigidConstraintType.fixed ||
                    options.type == RigidConstraintType.slider ||
                    (options.type == RigidConstraintType.hinge && axis != 2u) ||
                    (options.type == RigidConstraintType.piston && axis != 0u) ||
                    (motorType && (axis != 0u ||
                        options.motor.angular_enabled == 0u))
                : options.type == RigidConstraintType.fixed ||
                    options.type == RigidConstraintType.point ||
                    options.type == RigidConstraintType.hinge ||
                    ((options.type == RigidConstraintType.slider ||
                        options.type == RigidConstraintType.piston) &&
                        axis != 0u) ||
                    (motorType && (axis != 0u ||
                        options.motor.linear_enabled == 0u));
            RigidConstraintLimitOptions limits = angular
                ? options.angular_limits : options.linear_limits;
            RigidConstraintSpringOptions springs = angular
                ? options.angular_springs : options.linear_springs;
            float current = angular ? rigid_avbd_component(rotation, axis)
                : dot(error, direction);
            float oldCurrent = angular
                ? rigid_avbd_component(oldRotation, axis)
                : dot(oldError, oldDirection);
            Vec3 angularAxis = swing ? rotate(frameA, subtract(
                multiply(rigid_avbd_basis_axis(axis),
                    dot(freeAxis, swingAxis)),
                multiply(freeAxis,
                    rigid_avbd_component(swingAxis, axis))))
                : rigid_avbd_log_axis(frameA, twist, axis);
            AvbdVector6 jacobianA = angular
                ? rigid_avbd_jacobian({}, multiply(angularAxis, -1.0f))
                : rigid_avbd_jacobian(multiply(direction, -1.0f),
                    add(multiply(cross(armA, direction), -1.0f),
                        cross(direction, error)));
            AvbdVector6 jacobianB = angular
                ? rigid_avbd_jacobian({}, angularAxis)
                : rigid_avbd_jacobian(
                    direction, cross(armB, direction));
            float penalty = angular ? 1.0f / (max(dot(direction, add(
                inverse_inertia_world(parameters[a], stateA, direction),
                inverse_inertia_world(parameters[b], stateB, direction))),
                1.0e-6f) * dt * dt) : linearPenalty;
            float lockPenalty = angular ? max(penalty,
                2.0f * linearPenalty * max(
                    length_squared(cross(direction, armA)),
                    length_squared(cross(direction, armB)))) : penalty;
            if (lock)
            {
                if (visit == kAvbdInitializeJoint && angular)
                    joint.avbd_rows[index].dual.penalty = max(
                        joint.avbd_rows[index].dual.penalty,
                        lockPenalty / kAvbdGamma);
                rigid_avbd_visit_joint_row(joint, bodies, block, index,
                    impulse, first, jacobianA, jacobianB, current,
                    oldCurrent, lockPenalty, dt, angular,
                    kAvbdMaximumPenalty, 0.0f, -kAvbdMaximumPenalty,
                    kAvbdMaximumPenalty, false, 0.0f, add(error, armA),
                    armB, kAvbdAlpha, visit);
            }
            bool freeLimit = angular
                ? (options.type == RigidConstraintType.hinge && axis == 2u) ||
                    (options.type == RigidConstraintType.piston && axis == 0u)
                : (options.type == RigidConstraintType.slider ||
                    options.type == RigidConstraintType.piston) && axis == 0u;
            if ((generic || freeLimit) &&
                rigid_avbd_axis_enabled(limits.axes, axis))
            {
                float limitCurrent = angular
                    ? rigid_avbd_component(twist, axis) : current;
                float oldLimitCurrent = angular
                    ? rigid_avbd_component(oldTwist, axis) : oldCurrent;
                Vec3 limitAxis = rigid_avbd_log_axis(frameA, twist, axis);
                AvbdVector6 limitA = angular
                    ? rigid_avbd_jacobian({}, multiply(limitAxis, -1.0f))
                    : jacobianA;
                AvbdVector6 limitB = angular
                    ? rigid_avbd_jacobian({}, limitAxis) : jacobianB;
                float lower = rigid_avbd_component(limits.lower, axis);
                float upper = rigid_avbd_component(limits.upper, axis);
                if (lower == upper)
                {
                    float equalPenalty = angular && generic
                        ? lockPenalty : penalty;
                    if (visit == kAvbdInitializeJoint && angular && generic)
                        joint.avbd_rows[index].dual.penalty = max(
                            joint.avbd_rows[index].dual.penalty,
                            equalPenalty / kAvbdGamma);
                    rigid_avbd_visit_joint_row(joint, bodies, block, index,
                        impulse, first, limitA, limitB, limitCurrent - lower,
                        oldLimitCurrent - lower, equalPenalty, dt, angular,
                        kAvbdMaximumPenalty, 0.0f, -kAvbdMaximumPenalty,
                        kAvbdMaximumPenalty, false, 0.0f, add(error, armA),
                        armB, kAvbdAlpha, visit);
                }
                else if (lower < upper)
                {
                    rigid_avbd_visit_joint_row(joint, bodies, block, index,
                        impulse, first, limitA, limitB, limitCurrent - upper,
                        max(oldLimitCurrent - upper, 0.0f), penalty, dt,
                        angular, kAvbdMaximumPenalty, 0.0f, 0.0f,
                        kAvbdMaximumPenalty, false, 0.0f, add(error, armA),
                        armB, kAvbdLimitStabilization, visit);
                    rigid_avbd_visit_joint_row(joint, bodies, block, index,
                        impulse, first, limitA, limitB, limitCurrent - lower,
                        min(oldLimitCurrent - lower, 0.0f), penalty, dt,
                        angular, kAvbdMaximumPenalty, 0.0f,
                        -kAvbdMaximumPenalty, 0.0f, false, 0.0f,
                        add(error, armA), armB, kAvbdLimitStabilization,
                        visit);
                }
            }
            if (options.type == RigidConstraintType.generic_spring &&
                rigid_avbd_axis_enabled(springs.axes, axis))
                rigid_avbd_visit_joint_row(joint, bodies, block, index,
                    impulse, first, jacobianA, jacobianB, current, oldCurrent,
                    penalty, dt, angular,
                    rigid_avbd_component(springs.stiffness, axis),
                    rigid_avbd_component(springs.damping, axis),
                    -kAvbdMaximumPenalty, kAvbdMaximumPenalty, false, 0.0f,
                    add(error, armA), armB, kAvbdAlpha, visit);
            if (motorType && axis == 0u && (angular
                ? options.motor.angular_enabled != 0u
                : options.motor.linear_enabled != 0u))
            {
                float speed = angular ? options.motor.angular_target_velocity
                    : options.motor.linear_target_velocity;
                float bound = (angular
                    ? options.motor.angular_maximum_impulse
                    : options.motor.linear_maximum_impulse) / dt;
                Quaternion relative = relative_quaternion(frameA, frameB, true);
                Quaternion oldRelative = relative_quaternion(oldFrameA, oldFrameB, true);
                Vec3 motorRotation = rigid_avbd_relative_rotation_vector(
                    {0.0f, 0.0f, 0.0f, 1.0f},
                    relative_quaternion(oldRelative, relative, false));
                Vec3 motorAxis = rigid_avbd_log_axis(
                    frameA, motorRotation, axis);
                AvbdVector6 motorA = angular
                    ? rigid_avbd_jacobian({}, multiply(motorAxis, -1.0f))
                    : jacobianA;
                AvbdVector6 motorB = angular
                    ? rigid_avbd_jacobian({}, motorAxis) : jacobianB;
                float travel = angular
                    ? rigid_avbd_component(motorRotation, axis)
                    : current - oldCurrent;
                rigid_avbd_visit_joint_row(joint, bodies, block, index,
                    impulse, first, motorA, motorB, travel - speed * dt,
                    0.0f, penalty, dt, angular, kAvbdMaximumPenalty, 0.0f,
                    -bound, bound, true, speed, add(error, armA), armB,
                    kAvbdAlpha, visit);
            }
        }
    }
    if (visit == kAvbdInitializeJoint) joint.avbd_count = index;
    return impulse;
}

#ifndef PM_AVBD_PREPARE_SMALL_CACHE
#if PM_TARGET_D3D12 || defined(PM_TARGET_VULKAN)
#define PM_AVBD_PREPARE_SMALL_CACHE 1
#else
#define PM_AVBD_PREPARE_SMALL_CACHE 0
#endif
#endif
#ifndef PM_AVBD_PREPARE_CACHE_SIZE
#if PM_AVBD_PREPARE_SMALL_CACHE
#define PM_AVBD_PREPARE_CACHE_SIZE 512
#else
#define PM_AVBD_PREPARE_CACHE_SIZE 1024
#endif
#endif
// The tail already uses device memory; only the fast cache is smaller.
// Adapters budget scratch for native validation instrumentation as well.
static const uint kAvbdPrepareBodyCacheSize = PM_AVBD_PREPARE_CACHE_SIZE;
static const uint kAvbdPrepareContactCacheSize = PM_AVBD_PREPARE_CACHE_SIZE;
static const uint kAvbdPrepareJointCacheSize = 128u;

struct AvbdPrepareBodyMetadata
{
    uint parent;
    uint color;
    uint contact_head;
    uint joint_head;
    uint flags;
};

struct AvbdPrepareContactMetadata
{
    uint a;
    uint b;
    uint next_a;
    uint next_b;
    uint flags;
};

struct AvbdPrepareJointMetadata
{
    uint a;
    uint b;
    uint next_a;
    uint next_b;
    uint count;
    uint iterations;
};

groupshared AvbdPrepareBodyMetadata
    rigidAvbdBodyCache[kAvbdPrepareBodyCacheSize];
groupshared AvbdPrepareContactMetadata
    rigidAvbdContactCache[kAvbdPrepareContactCacheSize];
groupshared AvbdPrepareJointMetadata
    rigidAvbdJointCache[kAvbdPrepareJointCacheSize];

bool rigid_avbd_prepare_contact_impact(uint active,
    PM_AVBD_MANIFOLD_BUFFER manifolds, BodyParameters a, BodyParameters b,
    float dt)
{
    bool impact = false;
    for (uint point = 0u;
         point < PM_AVBD_MANIFOLD_COUNT(manifolds, active); ++point)
    {
        Contact contact = PM_AVBD_MANIFOLD_CONTACT(
            manifolds, active, point);
        float skin = rigid_avbd_contact_skin(contact, a, b);
        bool newImpact = (contact.avbd_cached == 0u ||
            contact.avbd_dual[0].lambda <= 1.0e-6f ||
            (contact.impact_fraction > 0.0f &&
                contact.impact_fraction < 1.0f)) &&
            contact.initial_normal_speed < -0.1f &&
            contact.penetration >= -kRigidSurfaceTolerance;
        float previousPenetration = contact.penetration +
            contact.initial_normal_speed * dt;
        impact = impact || newImpact ||
            previousPenetration > skin + 1.0e-4f;
    }
    return impact;
}

uint rigid_avbd_graph_parent(uint body, PM_AVBD_BUFFER(AvbdBody) bodies)
{
    return body < kAvbdPrepareBodyCacheSize
        ? rigidAvbdBodyCache[body].parent : bodies[body].parent;
}

void rigid_avbd_graph_set_parent(uint body, uint value,
    PM_AVBD_BUFFER(AvbdBody) bodies)
{
    if (body < kAvbdPrepareBodyCacheSize)
        rigidAvbdBodyCache[body].parent = value;
    else bodies[body].parent = value;
}

uint rigid_avbd_graph_color(uint body, PM_AVBD_BUFFER(AvbdBody) bodies)
{
    return body < kAvbdPrepareBodyCacheSize
        ? rigidAvbdBodyCache[body].color : bodies[body].color;
}

void rigid_avbd_graph_set_color(uint body, uint value,
    PM_AVBD_BUFFER(AvbdBody) bodies)
{
    if (body < kAvbdPrepareBodyCacheSize)
        rigidAvbdBodyCache[body].color = value;
    else bodies[body].color = value;
}

uint rigid_avbd_graph_contact_head(uint body,
    PM_AVBD_BUFFER(AvbdBody) bodies)
{
    return body < kAvbdPrepareBodyCacheSize
        ? rigidAvbdBodyCache[body].contact_head : bodies[body].contact_head;
}

void rigid_avbd_graph_set_contact_head(
    uint body, uint value, PM_AVBD_BUFFER(AvbdBody) bodies)
{
    if (body < kAvbdPrepareBodyCacheSize)
        rigidAvbdBodyCache[body].contact_head = value;
    else bodies[body].contact_head = value;
}

uint rigid_avbd_graph_joint_head(uint body, PM_AVBD_BUFFER(AvbdBody) bodies)
{
    return body < kAvbdPrepareBodyCacheSize
        ? rigidAvbdBodyCache[body].joint_head : bodies[body].joint_head;
}

void rigid_avbd_graph_set_joint_head(uint body, uint value,
    PM_AVBD_BUFFER(AvbdBody) bodies)
{
    if (body < kAvbdPrepareBodyCacheSize)
        rigidAvbdBodyCache[body].joint_head = value;
    else bodies[body].joint_head = value;
}

bool rigid_avbd_graph_dynamic(
    uint body, PM_AVBD_BUFFER(AvbdBody) bodies,
    PM_AVBD_BUFFER(BodyParameters) parameters)
{
    return body < kAvbdPrepareBodyCacheSize
        ? (rigidAvbdBodyCache[body].flags & 1u) != 0u
        : parameters[body].motion == MotionType.dynamic;
}

uint rigid_avbd_graph_impact(uint body, PM_AVBD_BUFFER(AvbdBody) bodies)
{
    return body < kAvbdPrepareBodyCacheSize
        ? rigidAvbdBodyCache[body].flags >> 1u : bodies[body].impact;
}

void rigid_avbd_graph_set_impact(uint body, uint value,
    PM_AVBD_BUFFER(AvbdBody) bodies)
{
    if (body < kAvbdPrepareBodyCacheSize)
        rigidAvbdBodyCache[body].flags =
            (rigidAvbdBodyCache[body].flags & 1u) | (value << 1u);
    else bodies[body].impact = value;
}

uint rigid_avbd_graph_root(uint body, PM_AVBD_BUFFER(AvbdBody) bodies)
{
    while (body != rigid_avbd_graph_parent(body, bodies))
        body = rigid_avbd_graph_parent(body, bodies);
    return body;
}

void rigid_avbd_graph_unite(uint a, uint b,
    PM_AVBD_BUFFER(AvbdBody) bodies,
    PM_AVBD_BUFFER(BodyParameters) parameters)
{
    if (!rigid_avbd_graph_dynamic(a, bodies, parameters) ||
        !rigid_avbd_graph_dynamic(b, bodies, parameters)) return;
    a = rigid_avbd_graph_root(a, bodies);
    b = rigid_avbd_graph_root(b, bodies);
    if (a != b)
        rigid_avbd_graph_set_parent(max(a, b), min(a, b), bodies);
}

bool rigid_avbd_graph_contact_nonempty(
    uint active, PM_AVBD_MANIFOLD_BUFFER manifolds)
{
    return active < kAvbdPrepareContactCacheSize
        ? (rigidAvbdContactCache[active].flags & 1u) != 0u
        : PM_AVBD_MANIFOLD_COUNT(manifolds, active) != 0u;
}

uint rigid_avbd_graph_contact_a(uint active, PM_AVBD_BUFFER(uint) pairs,
    uint bodyCount)
{
    return active < kAvbdPrepareContactCacheSize
        ? rigidAvbdContactCache[active].a : pairs[active] / bodyCount;
}

uint rigid_avbd_graph_contact_b(uint active, PM_AVBD_BUFFER(uint) pairs,
    uint bodyCount)
{
    return active < kAvbdPrepareContactCacheSize
        ? rigidAvbdContactCache[active].b : pairs[active] % bodyCount;
}

uint rigid_avbd_graph_contact_next(
    uint active, bool first, PM_AVBD_MANIFOLD_BUFFER manifolds)
{
    if (active < kAvbdPrepareContactCacheSize)
        return first ? rigidAvbdContactCache[active].next_a
            : rigidAvbdContactCache[active].next_b;
    return first ? PM_AVBD_MANIFOLD_NEXT_A(manifolds, active)
        : PM_AVBD_MANIFOLD_NEXT_B(manifolds, active);
}

void rigid_avbd_graph_set_contact_next(uint active, bool first, uint value,
    PM_AVBD_MANIFOLD_BUFFER manifolds)
{
    if (active < kAvbdPrepareContactCacheSize)
    {
        if (first) rigidAvbdContactCache[active].next_a = value;
        else rigidAvbdContactCache[active].next_b = value;
    }
    else if (first) PM_AVBD_MANIFOLD_NEXT_A(manifolds, active) = value;
    else PM_AVBD_MANIFOLD_NEXT_B(manifolds, active) = value;
}

bool rigid_avbd_graph_contact_impact(uint active, uint a, uint b, float dt,
    PM_AVBD_MANIFOLD_BUFFER manifolds,
    PM_AVBD_BUFFER(BodyParameters) parameters)
{
    return active < kAvbdPrepareContactCacheSize
        ? (rigidAvbdContactCache[active].flags & 2u) != 0u
        : rigid_avbd_prepare_contact_impact(
            active, manifolds, parameters[a], parameters[b], dt);
}

uint rigid_avbd_graph_joint_count(
    uint index, PM_AVBD_BUFFER(RigidConstraintResource) joints)
{
    return index < kAvbdPrepareJointCacheSize
        ? rigidAvbdJointCache[index].count : joints[index].avbd_count;
}

uint rigid_avbd_graph_joint_iterations(
    uint index, PM_AVBD_BUFFER(RigidConstraintResource) joints)
{
    return index < kAvbdPrepareJointCacheSize
        ? rigidAvbdJointCache[index].iterations
        : joints[index].options.solver_iterations;
}

uint rigid_avbd_graph_joint_a(uint index,
    PM_AVBD_BUFFER(RigidConstraintResource) joints)
{
    return index < kAvbdPrepareJointCacheSize
        ? rigidAvbdJointCache[index].a : joints[index].geometry.dense_a;
}

uint rigid_avbd_graph_joint_b(uint index,
    PM_AVBD_BUFFER(RigidConstraintResource) joints)
{
    return index < kAvbdPrepareJointCacheSize
        ? rigidAvbdJointCache[index].b : joints[index].geometry.dense_b;
}

uint rigid_avbd_graph_joint_next(
    uint index, bool first,
    PM_AVBD_BUFFER(RigidConstraintResource) joints)
{
    if (index < kAvbdPrepareJointCacheSize)
        return first ? rigidAvbdJointCache[index].next_a
            : rigidAvbdJointCache[index].next_b;
    return first ? joints[index].avbd_next_a : joints[index].avbd_next_b;
}

void rigid_avbd_graph_set_joint_next(uint index, bool first, uint value,
    PM_AVBD_BUFFER(RigidConstraintResource) joints)
{
    if (index < kAvbdPrepareJointCacheSize)
    {
        if (first) rigidAvbdJointCache[index].next_a = value;
        else rigidAvbdJointCache[index].next_b = value;
    }
    else if (first) joints[index].avbd_next_a = value;
    else joints[index].avbd_next_b = value;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void prepare_avbd_kernel(
    uniform PM_AVBD_BUFFER(BodyParameters) parameters,
    uniform PM_AVBD_BUFFER(RigidBodyState) previous,
    uniform PM_AVBD_BUFFER(RigidBodyState) states, uniform uint count,
    uniform PM_AVBD_MANIFOLD_BUFFER manifolds,
    uniform PM_AVBD_BUFFER(uint) pairs,
    uniform PM_AVBD_BUFFER(uint) pair_count,
    uniform PM_AVBD_BUFFER(RigidConstraintResource) joints,
    uniform uint joint_capacity,
    uniform PM_AVBD_BUFFER(RigidBodyId) ids,
    uniform PM_AVBD_BUFFER(AvbdBody) bodies,
    uniform PM_AVBD_BUFFER(ContactSchedule) schedule,
    uniform float dt, uniform Vec3 gravity,
    uniform uint base_iterations, uniform uint requested_iterations,
    uniform uint profile_version,
    uniform uint grid_blocks PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    bool realtime = profile_version == kRigidRealtimeProfileVersion;
    uint lane = pmGroupThreadIndex().x;
    for (uint bodyIndex = lane; bodyIndex < count;
         bodyIndex += pmGroupSize().x)
    {
        Vec3 oldVelocity = bodies[bodyIndex].previous_velocity;
        bool historyValid = bodies[bodyIndex].history_valid != 0u;
        AvbdBody body = {};
        body.parent = bodyIndex;
        body.color = kInvalidDense;
        body.contact_head = kInvalidDense;
        body.joint_head = kInvalidDense;
        body.target = states[bodyIndex];
        body.orientation_delta = {0.0f, 0.0f, 0.0f, 1.0f};
        bodies[bodyIndex] = body;
        if (bodyIndex < kAvbdPrepareBodyCacheSize)
        {
            AvbdPrepareBodyMetadata metadata;
            metadata.parent = bodyIndex;
            metadata.color = kInvalidDense;
            metadata.contact_head = kInvalidDense;
            metadata.joint_head = kInvalidDense;
            metadata.flags = parameters[bodyIndex].motion == MotionType.dynamic
                ? 1u : 0u;
            rigidAvbdBodyCache[bodyIndex] = metadata;
        }
        if (parameters[bodyIndex].motion != MotionType.dynamic) continue;
        float gravitySquared = length_squared(gravity);
        Vec3 acceleration = multiply(subtract(
            previous[bodyIndex].linear_velocity, oldVelocity), 1.0f / dt);
        float weight = historyValid && gravitySquared > 0.0f
            ? clamp_scalar(dot(acceleration, gravity) / gravitySquared,
                0.0f, 1.0f) : 0.0f;
        if (!isfinite(weight)) weight = 0.0f;
        Vec3 correction = multiply(gravity,
            -(1.0f - weight) * dt * dt);
        body = bodies[bodyIndex];
        body.displacement.values[0] = correction.x;
        body.displacement.values[1] = correction.y;
        body.displacement.values[2] = correction.z;
        body.previous_velocity = previous[bodyIndex].linear_velocity;
        body.history_valid = 1u;
        float mass = 1.0f /
            max(parameters[bodyIndex].inverse_mass, 1.0e-12f);
        for (uint axis = 0u; axis < 3u; ++axis)
        {
            body.inertia.h[axis][axis] = mass / (dt * dt);
            Vec3 localAxis = inverse_rotate(states[bodyIndex].orientation,
                rigid_avbd_basis_axis(axis));
            Vec3 column = rotate(states[bodyIndex].orientation, {
                localAxis.x / max(
                    parameters[bodyIndex].inverse_inertia_local.x, 1.0e-12f),
                localAxis.y / max(
                    parameters[bodyIndex].inverse_inertia_local.y, 1.0e-12f),
                localAxis.z / max(
                    parameters[bodyIndex].inverse_inertia_local.z, 1.0e-12f)});
            for (uint row = axis; row < 3u; ++row)
                body.inertia.h[3u + row][3u + axis] =
                    rigid_avbd_component(column, row) / (dt * dt);
        }
        bodies[bodyIndex] = body;
    }
    pmGroupBarrier();
    for (uint active = lane; active < pair_count[0];
         active += pmGroupSize().x)
    {
        uint a = pairs[active] / count;
        uint b = pairs[active] % count;
        float penalty = rigid_avbd_pair_stiffness(
            parameters[a], parameters[b], dt);
        ContactManifold manifold = manifolds[active];
        for (uint point = 0u; point < manifold.count; ++point)
        {
            Contact contact = manifold.contacts[point];
            if (contact.avbd_cached == 0u)
            {
                contact.avbd_anchor_a = inverse_rotate(
                    previous[a].orientation,
                    subtract(contact.point, previous[a].position));
                contact.avbd_anchor_b = inverse_rotate(
                    previous[b].orientation,
                    subtract(contact.point, previous[b].position));
            }
            Vec3 motionA = subtract(
                states[a].position, previous[a].position);
            Vec3 motionB = subtract(
                states[b].position, previous[b].position);
            Vec3 angularA = multiply(rigid_avbd_contact_delta_velocity(
                previous[a].orientation, states[a].orientation, dt), dt);
            Vec3 angularB = multiply(rigid_avbd_contact_delta_velocity(
                previous[b].orientation, states[b].orientation, dt), dt);
            Vec3 armA = subtract(contact.point, states[a].position);
            Vec3 armB = subtract(contact.point, states[b].position);
            float closing = dot(contact.normal, subtract(
                add(motionB, cross(angularB, armB)),
                add(motionA, cross(angularA, armA))));
            float oldPenetration = contact.penetration - closing;
            Vec3 tangent0;
            Vec3 tangent1;
            rigid_avbd_basis(contact.normal, tangent0, tangent1);
            Vec3 anchorMotion = add(subtract(motionB, motionA), subtract(
                rotation_displacement(previous[b].orientation,
                    states[b].orientation, contact.avbd_anchor_b),
                rotation_displacement(previous[a].orientation,
                    states[a].orientation, contact.avbd_anchor_a)));
            // New anchors represent the same world point by construction.
            // Do not turn inverse-rotation roundoff into initial friction.
            Vec3 oldAnchorDelta = contact.avbd_cached == 0u ? Vec3() : add(
                subtract(previous[b].position, previous[a].position),
                subtract(rotate(previous[b].orientation, contact.avbd_anchor_b),
                    rotate(previous[a].orientation, contact.avbd_anchor_a)));
            // Algebraically currentDelta - alpha*oldDelta, without subtracting
            // nearly coincident world-space anchor positions. Penalty stiffness
            // otherwise amplifies their roundoff into artificial friction.
            Vec3 tangentError = add(anchorMotion,
                multiply(oldAnchorDelta, 1.0f - kAvbdAlpha));
            contact.initial_normal_speed = -closing / dt;
            float skin = rigid_avbd_contact_skin(
                contact, parameters[a], parameters[b]);
            float impactDisplacement = realtime &&
                    contact.avbd_cached == 0u &&
                    contact.initial_normal_speed < -0.1f
                ? -(1.0f + min(parameters[a].restitution,
                               parameters[b].restitution)) *
                    contact.initial_normal_speed * dt
                : 0.0f;
            contact.avbd_error = {
                contact.penetration -
                    clamp_scalar(oldPenetration, 0.0f, skin) +
                    impactDisplacement,
                dot(tangentError, tangent0),
                dot(tangentError, tangent1)};
            for (uint axis = 0u; axis < 3u; ++axis)
            {
                AvbdDual dual = {contact.avbd_dual[axis].lambda,
                    contact.avbd_dual[axis].penalty};
                if (contact.avbd_cached == 0u) dual = {0.0f, penalty};
                dual = avbdWarmStart(dual, kAvbdMaximumPenalty);
                contact.avbd_dual[axis] = {dual.lambda, dual.penalty};
            }
            manifold.contacts[point] = contact;
        }
        manifolds[active] = manifold;
        if (active < kAvbdPrepareContactCacheSize)
        {
            AvbdPrepareContactMetadata metadata;
            metadata.a = a;
            metadata.b = b;
            metadata.next_a = PM_AVBD_MANIFOLD_NEXT_A(manifolds, active);
            metadata.next_b = PM_AVBD_MANIFOLD_NEXT_B(manifolds, active);
            metadata.flags = (PM_AVBD_MANIFOLD_COUNT(
                    manifolds, active) != 0u ? 1u : 0u) |
                (rigid_avbd_prepare_contact_impact(active, manifolds,
                    parameters[a], parameters[b], dt) ? 2u : 0u);
            rigidAvbdContactCache[active] = metadata;
        }
    }
    for (uint jointIndex = lane; jointIndex < joint_capacity;
         jointIndex += pmGroupSize().x)
    {
        joints[jointIndex].avbd_count = 0u;
        if (joints[jointIndex].alive != 0u)
        {
            joints[jointIndex].state.enabled =
                joints[jointIndex].options.enabled != 0u &&
                joints[jointIndex].state.broken == 0u;
            if (joints[jointIndex].state.enabled != 0u)
            {
                joints[jointIndex].state.applied_impulse = 0.0f;
                joints[jointIndex].geometry.dense_a = rigid_avbd_find_dense(
                    joints[jointIndex].options.body_a, ids, count);
                joints[jointIndex].geometry.dense_b = rigid_avbd_find_dense(
                    joints[jointIndex].options.body_b, ids, count);
                if (joints[jointIndex].geometry.dense_a != kInvalidDense &&
                    joints[jointIndex].geometry.dense_b != kInvalidDense)
                {
                    AvbdBlock scratch = avbdZeroBlock();
                    rigid_avbd_visit_joint(joints[jointIndex], parameters,
                        previous, bodies, dt, scratch, true,
                        kAvbdInitializeJoint);
                }
            }
        }
    }
    pmGroupBarrier();
    for (uint jointIndex = lane;
         jointIndex < min(joint_capacity, kAvbdPrepareJointCacheSize);
         jointIndex += pmGroupSize().x)
    {
        AvbdPrepareJointMetadata metadata;
        metadata.a = joints[jointIndex].geometry.dense_a;
        metadata.b = joints[jointIndex].geometry.dense_b;
        metadata.next_a = joints[jointIndex].avbd_next_a;
        metadata.next_b = joints[jointIndex].avbd_next_b;
        metadata.count = joints[jointIndex].avbd_count;
        metadata.iterations = joints[jointIndex].options.solver_iterations;
        rigidAvbdJointCache[jointIndex] = metadata;
    }
    pmGroupBarrier();
    if (lane == 0u)
    {
        ContactSchedule value = {};
        value.budget = realtime ? kRigidRealtimeSweeps :
            (requested_iterations != 0u
                ? requested_iterations : base_iterations);
        value.offsets[0] = profile_version;
        value.offsets[1] = base_iterations;
        // Early exit requires a per-island constraint-residual test. A small
        // pose update alone is not sufficient for motors, springs, limits, or
        // coupled contacts, so the production schedule remains exact.
        value.counts[2] = 0u;
        value.candidates = pair_count[0];
        schedule[0] = value;
        for (uint active = 0u; active < pair_count[0]; ++active)
        {
            if (!rigid_avbd_graph_contact_nonempty(active, manifolds))
                continue;
            schedule[0].contacts++;
            uint a = rigid_avbd_graph_contact_a(active, pairs, count);
            uint b = rigid_avbd_graph_contact_b(active, pairs, count);
            rigid_avbd_graph_unite(a, b, bodies, parameters);
            rigid_avbd_graph_set_contact_next(active, true,
                rigid_avbd_graph_contact_head(a, bodies), manifolds);
            rigid_avbd_graph_set_contact_head(a, active, bodies);
            rigid_avbd_graph_set_contact_next(active, false,
                rigid_avbd_graph_contact_head(b, bodies), manifolds);
            rigid_avbd_graph_set_contact_head(b, active, bodies);
        }
        for (uint jointIndex = 0u; jointIndex < joint_capacity; ++jointIndex)
        {
            if (rigid_avbd_graph_joint_count(jointIndex, joints) == 0u)
                continue;
            uint a = rigid_avbd_graph_joint_a(jointIndex, joints);
            uint b = rigid_avbd_graph_joint_b(jointIndex, joints);
            if (!realtime && requested_iterations == 0u)
            {
                schedule[0].budget = max(schedule[0].budget,
                    rigid_avbd_graph_joint_iterations(jointIndex, joints));
            }
            rigid_avbd_graph_unite(a, b, bodies, parameters);
            rigid_avbd_graph_set_joint_next(jointIndex, true,
                rigid_avbd_graph_joint_head(a, bodies), joints);
            rigid_avbd_graph_set_joint_head(a, jointIndex, bodies);
            rigid_avbd_graph_set_joint_next(jointIndex, false,
                rigid_avbd_graph_joint_head(b, bodies), joints);
            rigid_avbd_graph_set_joint_head(b, jointIndex, bodies);
        }
        for (uint bodyIndex = 0u; bodyIndex < count; ++bodyIndex)
            rigid_avbd_graph_set_parent(bodyIndex,
                rigid_avbd_graph_root(bodyIndex, bodies), bodies);
        // Preserve one established scene-wide authored budget. Candidate and
        // color order are physics-visible in a finite AVBD solve; there is no
        // second per-island iteration path.
        for (uint active = 0u; active < pair_count[0]; ++active)
        {
            uint a = rigid_avbd_graph_contact_a(active, pairs, count);
            uint b = rigid_avbd_graph_contact_b(active, pairs, count);
            if (rigid_avbd_graph_contact_impact(active, a, b, dt,
                manifolds, parameters))
            {
                if (rigid_avbd_graph_dynamic(a, bodies, parameters))
                    rigid_avbd_graph_set_impact(
                        rigid_avbd_graph_parent(a, bodies), 1u, bodies);
                if (rigid_avbd_graph_dynamic(b, bodies, parameters))
                    rigid_avbd_graph_set_impact(
                        rigid_avbd_graph_parent(b, bodies), 1u, bodies);
                ContactSchedule value = schedule[0];
                value.counts[0] = 1u;
                schedule[0] = value;
            }
        }
        for (uint bodyIndex = 0u; bodyIndex < count; ++bodyIndex)
        {
            if (!rigid_avbd_graph_dynamic(bodyIndex, bodies, parameters))
                continue;
            rigid_avbd_graph_set_impact(bodyIndex,
                rigid_avbd_graph_impact(
                    rigid_avbd_graph_parent(bodyIndex, bodies), bodies),
                bodies);
            if (rigid_avbd_graph_parent(bodyIndex, bodies) == bodyIndex &&
                (rigid_avbd_graph_contact_head(bodyIndex, bodies) !=
                    kInvalidDense ||
                 rigid_avbd_graph_joint_head(bodyIndex, bodies) !=
                    kInvalidDense))
                schedule[0].island_count++;
        }
        for (uint bodyIndex = 0u; bodyIndex < count; ++bodyIndex)
        {
            if (!rigid_avbd_graph_dynamic(bodyIndex, bodies, parameters))
                continue;
            uint color = 0u;
            for (;; ++color)
            {
                bool conflict = false;
                for (uint active = rigid_avbd_graph_contact_head(
                         bodyIndex, bodies);
                     active != kInvalidDense;)
                {
                    uint a = rigid_avbd_graph_contact_a(active, pairs, count);
                    uint b = rigid_avbd_graph_contact_b(active, pairs, count);
                    bool first = bodyIndex == a;
                    conflict = conflict || rigid_avbd_graph_color(
                        first ? b : a, bodies) == color;
                    active = rigid_avbd_graph_contact_next(
                        active, first, manifolds);
                }
                for (uint jointIndex = rigid_avbd_graph_joint_head(
                         bodyIndex, bodies);
                     jointIndex != kInvalidDense;)
                {
                    bool first = bodyIndex ==
                        rigid_avbd_graph_joint_a(jointIndex, joints);
                    conflict = conflict || rigid_avbd_graph_color(first
                        ? rigid_avbd_graph_joint_b(jointIndex, joints)
                        : rigid_avbd_graph_joint_a(jointIndex, joints),
                        bodies) == color;
                    jointIndex = rigid_avbd_graph_joint_next(
                        jointIndex, first, joints);
                }
                if (!conflict) break;
            }
            rigid_avbd_graph_set_color(bodyIndex, color, bodies);
            schedule[0].remaining = max(
                schedule[0].remaining, color + 1u);
        }
        schedule[0].blocks = grid_blocks;
    }
    pmGroupBarrier();
    for (uint bodyIndex = lane;
         bodyIndex < min(count, kAvbdPrepareBodyCacheSize);
         bodyIndex += pmGroupSize().x)
    {
        AvbdPrepareBodyMetadata metadata = rigidAvbdBodyCache[bodyIndex];
        bodies[bodyIndex].parent = metadata.parent;
        bodies[bodyIndex].color = metadata.color;
        bodies[bodyIndex].contact_head = metadata.contact_head;
        bodies[bodyIndex].joint_head = metadata.joint_head;
        bodies[bodyIndex].impact = metadata.flags >> 1u;
    }
    for (uint active = lane;
         active < min(pair_count[0], kAvbdPrepareContactCacheSize);
         active += pmGroupSize().x)
    {
        PM_AVBD_MANIFOLD_NEXT_A(manifolds, active) =
            rigidAvbdContactCache[active].next_a;
        PM_AVBD_MANIFOLD_NEXT_B(manifolds, active) =
            rigidAvbdContactCache[active].next_b;
    }
    for (uint jointIndex = lane;
         jointIndex < min(joint_capacity, kAvbdPrepareJointCacheSize);
         jointIndex += pmGroupSize().x)
    {
        joints[jointIndex].avbd_next_a =
            rigidAvbdJointCache[jointIndex].next_a;
        joints[jointIndex].avbd_next_b =
            rigidAvbdJointCache[jointIndex].next_b;
    }
}

#include "geometry_constraints_solve.slang"

#endif
