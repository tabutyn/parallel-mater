// SPDX-License-Identifier: MIT
// Slang mirror of src/avbd_cuda.cuh. Uses the shared Slang AVBD block solver
// and the same rigid/contact ABI as geometry_constraints.slang.

#include "avbd.slang"
#include "geometry_constraints.slang"

struct AvbdBody
{
    RigidBodyState target;
    Quaternion current_orientation;
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
    state.orientation = body.current_orientation;
    return state;
}

void rigid_avbd_apply_update(inout AvbdBody body, AvbdVector6 update)
{
    for (uint axis = 0u; axis < 3u; ++axis)
        body.displacement.values[axis] += update.values[axis];
    Vec3 rotation = rigid_avbd_rotation(update);
    Quaternion delta = quaternion_multiply(
        {rotation.x, rotation.y, rotation.z, 0.0f},
        body.current_orientation);
    body.current_orientation = normalized_quaternion({
        body.current_orientation.x + 0.5f * delta.x,
        body.current_orientation.y + 0.5f * delta.y,
        body.current_orientation.z + 0.5f * delta.z,
        body.current_orientation.w + 0.5f * delta.w});
    Quaternion relative = quaternion_multiply(
        body.current_orientation, conjugate(body.target.orientation));
    if (relative.w < 0.0f)
        relative = {-relative.x, -relative.y, -relative.z, -relative.w};
    body.displacement.values[3] = 2.0f * relative.x;
    body.displacement.values[4] = 2.0f * relative.y;
    body.displacement.values[5] = 2.0f * relative.z;
}

float rigid_avbd_pair_stiffness(
    BodyParameters a, BodyParameters b, float dt)
{
    return 1.0f / (max(a.inverse_mass + b.inverse_mass, 1.0e-6f) *
        dt * dt);
}

void rigid_avbd_add_contact_rows(
    inout AvbdBlock block, uint body, uint a, uint b, Contact contact,
    BodyParameters* parameters, AvbdBody* bodies, float dt)
{
    bool first = body == a;
    RigidBodyState pose = rigid_avbd_pose(bodies[body]);
    Vec3 direction = first ? multiply(contact.normal, -1.0f) : contact.normal;
    Vec3 arm = subtract(contact.point, pose.position);
    AvbdVector6 jacobian = rigid_avbd_jacobian(
        direction, cross(arm, direction));
    float stiffness = rigid_avbd_pair_stiffness(
        parameters[a], parameters[b], dt);
    float force = stiffness * max(0.0f, contact.penetration);
    avbdAddRow(block, jacobian, stiffness, force);
}

void rigid_avbd_add_joint_rows(
    inout AvbdBlock block, uint body, RigidConstraintResource joint,
    BodyParameters* parameters, AvbdBody* bodies, float dt)
{
    uint a = joint.geometry.dense_a, b = joint.geometry.dense_b;
    if (body != a && body != b) return;
    bool first = body == a;
    RigidBodyState pa = rigid_avbd_pose(bodies[a]);
    RigidBodyState pb = rigid_avbd_pose(bodies[b]);
    Vec3 anchorA = transform_point(pa, joint.options.local_anchor_a);
    Vec3 anchorB = transform_point(pb, joint.options.local_anchor_b);
    Vec3 error = subtract(anchorB, anchorA);
    float stiffness = 1.0f / (max(
        parameters[a].inverse_mass + parameters[b].inverse_mass, 1.0e-6f) *
        dt * dt);
    Vec3 axes[3] = {{1.0f, 0.0f, 0.0f},
                    {0.0f, 1.0f, 0.0f},
                    {0.0f, 0.0f, 1.0f}};
    Vec3 arm = first ? subtract(anchorA, pa.position) :
        subtract(anchorB, pb.position);
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        Vec3 direction = first ? multiply(axes[axis], -1.0f) : axes[axis];
        AvbdVector6 jacobian = rigid_avbd_jacobian(
            direction, cross(arm, direction));
        avbdAddRow(block, jacobian, stiffness,
            stiffness * (axis == 0u ? error.x :
                (axis == 1u ? error.y : error.z)));
    }
    if (joint.options.type == RigidConstraintType.fixed)
    {
        Quaternion relative = quaternion_multiply(
            pb.orientation, conjugate(pa.orientation));
        if (relative.w < 0.0f)
            relative = {-relative.x, -relative.y, -relative.z, -relative.w};
        Vec3 rotationError = {2.0f * relative.x,
            2.0f * relative.y, 2.0f * relative.z};
        for (uint axis = 0u; axis < 3u; ++axis)
        {
            Vec3 angular = first ? multiply(axes[axis], -1.0f) : axes[axis];
            AvbdVector6 jacobian = rigid_avbd_jacobian({}, angular);
            float errorValue = axis == 0u ? rotationError.x :
                (axis == 1u ? rotationError.y : rotationError.z);
            avbdAddRow(block, jacobian, stiffness,
                stiffness * errorValue);
        }
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void prepare_avbd_kernel(
    uniform BodyParameters* parameters, uniform RigidBodyState* previous,
    uniform RigidBodyState* predicted, uniform uint count,
    uniform ContactManifold* manifolds, uniform uint* pairs,
    uniform uint* pair_count, uniform RigidConstraintResource* joints,
    uniform uint joint_capacity, uniform RigidBodyId* ids,
    uniform AvbdBody* bodies, uniform ContactSchedule* schedule,
    uniform float dt, uniform Vec3 gravity,
    uniform uint requested_iterations, uniform uint grid_blocks)
{
    uint lane = pmThreadIndex();
    if (lane < count)
    {
        Vec3 previousVelocity = bodies[lane].previous_velocity;
        uint historyValid = bodies[lane].history_valid;
        AvbdBody body = {};
        body.parent = lane;
        body.contact_head = kInvalidDense;
        body.joint_head = kInvalidDense;
        body.color = lane;
        body.target = predicted[lane];
        body.current_orientation = predicted[lane].orientation;
        body.previous_velocity = previous[lane].linear_velocity;
        body.history_valid = 1u;
        body.inertia = avbdZeroBlock();
        if (parameters[lane].motion == MotionType.dynamic)
        {
            float mass = 1.0f / max(parameters[lane].inverse_mass, 1.0e-12f);
            for (uint axis = 0u; axis < 3u; ++axis)
                body.inertia.h[axis][axis] = mass / (dt * dt);
            Vec3 inverseInertia = parameters[lane].inverse_inertia_local;
            body.inertia.h[3][3] = 1.0f /
                (max(inverseInertia.x, 1.0e-12f) * dt * dt);
            body.inertia.h[4][4] = 1.0f /
                (max(inverseInertia.y, 1.0e-12f) * dt * dt);
            body.inertia.h[5][5] = 1.0f /
                (max(inverseInertia.z, 1.0e-12f) * dt * dt);
            float gravitySquared = length_squared(gravity);
            Vec3 acceleration = multiply(subtract(
                previous[lane].linear_velocity, previousVelocity), 1.0f / dt);
            float weight = historyValid != 0u && gravitySquared > 0.0f
                ? clamp_scalar(dot(acceleration, gravity) /
                    gravitySquared, 0.0f, 1.0f) : 0.0f;
            Vec3 correction = multiply(gravity, -(1.0f - weight) * dt * dt);
            body.displacement.values[0] = correction.x;
            body.displacement.values[1] = correction.y;
            body.displacement.values[2] = correction.z;
        }
        bodies[lane] = body;
    }
    if (lane < *pair_count)
    {
        ContactManifold manifold = manifolds[lane];
        for (uint point = 0u; point < manifold.count; ++point)
        {
            Contact contact = manifold.contacts[point];
            float penalty = rigid_avbd_pair_stiffness(
                parameters[pairs[lane] / count],
                parameters[pairs[lane] % count], dt);
            for (uint axis = 0u; axis < 3u; ++axis)
            {
                contact.avbd_dual[axis].lambda *= 0.99f * 0.999f;
                contact.avbd_dual[axis].penalty = max(
                    1.0f, contact.avbd_dual[axis].penalty * 0.999f);
                if (!contact.avbd_cached)
                    contact.avbd_dual[axis].penalty = penalty;
            }
            manifold.contacts[point] = contact;
        }
        manifolds[lane] = manifold;
    }
    if (lane < joint_capacity && joints[lane].alive)
    {
        RigidConstraintResource joint = joints[lane];
        joint.geometry.dense_a = find_rigid_body_dense(
            joint.options.body_a, ids, count);
        joint.geometry.dense_b = find_rigid_body_dense(
            joint.options.body_b, ids, count);
        joint.avbd_count = 0u;
        joints[lane] = joint;
    }
    if (lane == 0u)
    {
        ContactSchedule value = {};
        value.budget = max(1u, requested_iterations);
        value.remaining = count == 0u ? 0u : count;
        value.blocks = grid_blocks;
        value.candidates = *pair_count;
        schedule[0] = value;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void solve_avbd_kernel(
    uniform BodyParameters* parameters, uniform RigidBodyState* previous,
    uniform RigidBodyState* states, uniform uint count,
    uniform ContactManifold* manifolds, uniform uint* pairs,
    uniform uint* pair_count, uniform RigidConstraintResource* joints,
    uniform uint joint_capacity, uniform AvbdBody* bodies,
    uniform ContactSchedule* schedule, uniform float dt)
{
    uint iterations = max(1u, schedule[0].budget);
    for (uint iteration = 0u; iteration < iterations; ++iteration)
    {
        for (uint bodyIndex = 0u; bodyIndex < count; ++bodyIndex)
        {
            if (parameters[bodyIndex].motion != MotionType.dynamic) continue;
            AvbdBlock block = bodies[bodyIndex].inertia;
            AvbdVector6 displacement = bodies[bodyIndex].displacement;
            for (uint row = 0u; row < 6u; ++row)
                for (uint column = 0u; column < 6u; ++column)
                    block.g[row] += block.h[max(row, column)][min(row, column)] *
                        displacement.values[column];
            for (uint active = 0u; active < *pair_count; ++active)
            {
                uint a = pairs[active] / count, b = pairs[active] % count;
                if (bodyIndex != a && bodyIndex != b) continue;
                ContactManifold manifold = manifolds[active];
                for (uint point = 0u; point < manifold.count; ++point)
                    rigid_avbd_add_contact_rows(block, bodyIndex, a, b,
                        manifold.contacts[point], parameters, bodies, dt);
            }
            for (uint jointIndex = 0u; jointIndex < joint_capacity; ++jointIndex)
            {
                RigidConstraintResource joint = joints[jointIndex];
                if (!joint.alive || !joint.options.enabled ||
                    !joint.state.enabled || joint.state.broken ||
                    joint.geometry.dense_a == kInvalidDense ||
                    joint.geometry.dense_b == kInvalidDense) continue;
                rigid_avbd_add_joint_rows(
                    block, bodyIndex, joint, parameters, bodies, dt);
            }
            AvbdVector6 update = avbdZeroVector6();
            if (avbdSolve(block, update))
            {
                AvbdBody body = bodies[bodyIndex];
                rigid_avbd_apply_update(body, update);
                bodies[bodyIndex] = body;
            }
        }
    }
    for (uint bodyIndex = 0u; bodyIndex < count; ++bodyIndex)
    {
        if (parameters[bodyIndex].motion != MotionType.dynamic) continue;
        RigidBodyState state = rigid_avbd_pose(bodies[bodyIndex]);
        state.linear_velocity = clamp_length(multiply(subtract(
            state.position, previous[bodyIndex].position), 1.0f / dt),
            parameters[bodyIndex].maximum_linear_speed);
        state.angular_velocity = clamp_length(quaternion_delta_velocity(
            previous[bodyIndex].orientation, state.orientation, dt),
            parameters[bodyIndex].maximum_angular_speed);
        states[bodyIndex] = state;
    }
    schedule[0].maximum_passes += iterations;
}
