// SPDX-License-Identifier: MIT
#include "dispatch_shape.slangh"
// Shared AVBD operations. Cooperative CUDA and multi-dispatch adapters use
// these exact equations; only the synchronization schedule differs.

#define PM_AVBD_STAGE(name, value) static const uint kAvbdStage_##name = value;
#include "avbd_stages.inc"
#undef PM_AVBD_STAGE

void rigid_avbd_barrier()
{
    if (pmGridDim().x == 1u) pmGroupBarrier();
    else pmGridBarrier();
}

// The iterative hinge rows keep motion tangent to the anchor, while the finite
// quaternion update follows the chord of that motion. Remove the
// resulting positional residue once per substep so off-centre hinges do not
// accumulate drift through full rotations or contact impulses. This is the
// final projection of the AVBD solve, not a separate constraint solver.
void rigid_avbd_project_hinge_anchors(
    PM_AVBD_BUFFER(BodyParameters) parameters,
    PM_AVBD_BUFFER(RigidConstraintResource) joints, uint jointCapacity,
    PM_AVBD_BUFFER(RigidBodyState) states)
{
    if (pmThreadIndex() == 0u)
    {
        for (uint jointIndex = 0u; jointIndex < jointCapacity; ++jointIndex)
        {
            if (joints[jointIndex].avbd_count == 0u ||
                joints[jointIndex].state.enabled == 0u ||
                joints[jointIndex].options.type != RigidConstraintType.hinge)
                continue;
            uint a = joints[jointIndex].geometry.dense_a;
            uint b = joints[jointIndex].geometry.dense_b;
            float inverseMassA = parameters[a].motion == MotionType.dynamic
                ? parameters[a].inverse_mass : 0.0f;
            float inverseMassB = parameters[b].motion == MotionType.dynamic
                ? parameters[b].inverse_mass : 0.0f;
            float inverseMassSum = inverseMassA + inverseMassB;
            if (inverseMassSum <= kEpsilon) continue;

            RigidBodyState stateA = states[a];
            RigidBodyState stateB = states[b];
            Vec3 anchorA = add(stateA.position, rotate(stateA.orientation,
                joints[jointIndex].options.local_anchor_a));
            Vec3 anchorB = add(stateB.position, rotate(stateB.orientation,
                joints[jointIndex].options.local_anchor_b));
            Vec3 error = subtract(anchorB, anchorA);
            Vec3 correctionA = multiply(error, inverseMassA / inverseMassSum);
            Vec3 correctionB = multiply(error, inverseMassB / inverseMassSum);
            states[a].position = add(states[a].position, correctionA);
            states[b].position = subtract(states[b].position, correctionB);
        }
    }
}

void rigid_avbd_solve_stage(
    PM_AVBD_BUFFER(BodyParameters) parameters,
    PM_AVBD_BUFFER(RigidBodyState) previous,
    PM_AVBD_BUFFER(RigidBodyState) states, uint count,
    PM_AVBD_MANIFOLD_BUFFER manifolds,
    PM_AVBD_BUFFER(uint) pairs,
    PM_AVBD_BUFFER(uint) pair_count,
    PM_AVBD_BUFFER(RigidConstraintResource) joints,
    uint joint_capacity,
    PM_AVBD_BUFFER(AvbdBody) bodies,
    PM_AVBD_BUFFER(ContactSchedule) schedule, float dt,
    uint operation, uint phase, uint iteration, uint color)
{
    // Uniform guards permit a host-known upper-bound schedule without reading
    // GPU-generated color/impact counts back to the host.
    if (phase > 1u || (phase != 0u && schedule[0].counts[0] == 0u)) return;
    if ((operation == kAvbdStage_solve_color ||
         operation == kAvbdStage_advance_duals) &&
        iteration >= schedule[0].budget) return;
    if (operation == kAvbdStage_solve_color && color >= schedule[0].remaining)
        return;
    uint lane = pmThreadIndex();
    uint stride = pmGroupSize().x * pmGridDim().x;
    bool velocityPhase = phase != 0u;
    if (operation == kAvbdStage_impact_gaps)
    {
        for (uint active = lane; active < pair_count[0]; active += stride)
        {
            uint a = pairs[active] / count;
            uint b = pairs[active] % count;
            if (bodies[a].impact == 0u && bodies[b].impact == 0u)
                continue;
            ContactManifold manifold = manifolds[active];
            for (uint point = 0u; point < manifold.count; ++point)
            {
                Contact contact = manifold.contacts[point];
                AvbdVector6 jacobianA;
                AvbdVector6 jacobianB;
                rigid_avbd_contact_normal_row(contact, a, b, bodies,
                    jacobianA, jacobianB);
                float gap = max(0.0f, -(contact.penetration +
                    avbdDot6(jacobianA, bodies[a].displacement) +
                    avbdDot6(jacobianB, bodies[b].displacement)));
                contact.avbd_error.x = gap;
                manifold.contacts[point] = contact;
            }
            manifolds[active] = manifold;
        }
    }
    if (operation == kAvbdStage_impact_bodies)
    {
        for (uint bodyIndex = lane; bodyIndex < count; bodyIndex += stride)
        {
            if (bodies[bodyIndex].impact != 0u ||
                parameters[bodyIndex].motion != MotionType.dynamic)
            {
                bodies[bodyIndex].target = states[bodyIndex];
                bodies[bodyIndex].displacement = avbdZeroVector6();
                bodies[bodyIndex].orientation_delta =
                    Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
            }
        }
    }
    if (operation == kAvbdStage_impact_constraints)
    {
        for (uint jointIndex = lane; jointIndex < joint_capacity;
             jointIndex += stride)
        {
            if (joints[jointIndex].avbd_count == 0u ||
                joints[jointIndex].state.enabled == 0u) continue;
            uint a = joints[jointIndex].geometry.dense_a;
            uint b = joints[jointIndex].geometry.dense_b;
            if (bodies[a].impact == 0u && bodies[b].impact == 0u)
                continue;
            AvbdBlock scratch = avbdZeroBlock();
            rigid_avbd_visit_joint(joints[jointIndex], parameters,
                previous, bodies, dt, scratch, true, kAvbdImpactJoint);
        }
        for (uint active = lane; active < pair_count[0]; active += stride)
        {
            uint a = pairs[active] / count;
            uint b = pairs[active] % count;
            if (bodies[a].impact == 0u && bodies[b].impact == 0u)
                continue;
            AvbdVector6 velocityA = rigid_avbd_jacobian(
                multiply(states[a].linear_velocity, dt),
                multiply(states[a].angular_velocity, dt));
            AvbdVector6 velocityB = rigid_avbd_jacobian(
                multiply(states[b].linear_velocity, dt),
                multiply(states[b].angular_velocity, dt));
            ContactManifold manifold = manifolds[active];
            for (uint point = 0u; point < manifold.count; ++point)
            {
                Contact contact = manifold.contacts[point];
                float gap = contact.avbd_error.x;
                AvbdVector6 jacobianA[3];
                AvbdVector6 jacobianB[3];
                float error[3];
                rigid_avbd_contact_rows(contact, a, b, bodies,
                    jacobianA, jacobianB, error);
                float restitution = contact.avbd_cached == 0u &&
                    contact.initial_normal_speed < -1.0f
                    ? min(parameters[a].restitution,
                        parameters[b].restitution) : 0.0f;
                contact.avbd_error = {
                    avbdDot6(jacobianA[0], velocityA) +
                        avbdDot6(jacobianB[0], velocityB) - gap -
                        restitution * contact.initial_normal_speed * dt,
                    avbdDot6(jacobianA[1], velocityA) +
                        avbdDot6(jacobianB[1], velocityB),
                    avbdDot6(jacobianA[2], velocityA) +
                        avbdDot6(jacobianB[2], velocityB)};
                manifold.contacts[point] = contact;
            }
            manifolds[active] = manifold;
        }
    }
    if (operation == kAvbdStage_solve_color)
    {
        for (uint bodyIndex = lane; bodyIndex < count;
            bodyIndex += stride)
        {
            if (bodies[bodyIndex].color != color) continue;
            if (phase != 0u && bodies[bodyIndex].impact == 0u)
                continue;
            AvbdBlock block = bodies[bodyIndex].inertia;
            AvbdVector6 displacement = bodies[bodyIndex].displacement;
            for (uint row = 0u; row < 6u; ++row)
                for (uint column = 0u; column < 6u; ++column)
                    block.g[row] += block.h[
                        max(row, column)][min(row, column)] *
                        displacement.values[column];
            for (uint active = bodies[bodyIndex].contact_head;
                 active != kInvalidDense;)
            {
                uint a = pairs[active] / count;
                uint b = pairs[active] % count;
                bool first = bodyIndex == a;
                float friction = sqrt(
                    parameters[a].friction * parameters[b].friction);
                for (uint point = 0u;
                     point < PM_AVBD_MANIFOLD_COUNT(
                        manifolds, active); ++point)
                {
                    Contact contact = PM_AVBD_MANIFOLD_CONTACT(
                        manifolds, active, point);
                    AvbdVector6 jacobianA[3];
                    AvbdVector6 jacobianB[3];
                    float error[3];
                    rigid_avbd_contact_rows(contact, a, b, bodies,
                        jacobianA, jacobianB, error);
                    AvbdContactForce trial =
                        rigid_avbd_contact_trial(contact, error);
                    AvbdContactForce force =
                        avbdProjectContact(trial, friction);
                    AvbdContactForce scales =
                        avbdContactStiffnessScales(trial, friction);
                    float stiffness[3] = {scales.normal,
                        scales.tangent0, scales.tangent1};
                    Vec3 tangent0;
                    Vec3 tangent1;
                    rigid_avbd_basis(
                        contact.normal, tangent0, tangent1);
                    float forces[3] = {
                        force.normal - (phase != 0u
                            ? contact.accumulated_normal_impulse / dt
                            : 0.0f),
                        force.tangent0 - (phase != 0u
                            ? dot(contact.accumulated_friction_impulse,
                                tangent0) / dt : 0.0f),
                        force.tangent1 - (phase != 0u
                            ? dot(contact.accumulated_friction_impulse,
                                tangent1) / dt : 0.0f)};
                    for (uint axis = 0u; axis < 3u; ++axis)
                        avbdAddRow(block, first ? jacobianA[axis]
                            : jacobianB[axis],
                            contact.avbd_dual[axis].penalty *
                                stiffness[axis], forces[axis]);
                }
                active = first
                    ? PM_AVBD_MANIFOLD_NEXT_A(manifolds, active)
                    : PM_AVBD_MANIFOLD_NEXT_B(manifolds, active);
            }
            for (uint jointIndex = bodies[bodyIndex].joint_head;
                 jointIndex != kInvalidDense;)
            {
                uint a = joints[jointIndex].geometry.dense_a;
                uint b = joints[jointIndex].geometry.dense_b;
                bool first = bodyIndex == a;
                if (joints[jointIndex].state.enabled != 0u)
                {
                    if (phase == 0u)
                        rigid_avbd_visit_joint(joints[jointIndex],
                            parameters, previous, bodies, dt, block,
                            first, kAvbdAccumulateJoint);
                    else
                        for (uint row = 0u;
                             row < joints[jointIndex].avbd_count;
                             ++row)
                        {
                            AvbdRow stored = rigid_avbd_row(
                                joints[jointIndex].avbd_rows[row]);
                            if (stored.stiffness >=
                                kAvbdMaximumPenalty)
                                avbdAccumulate(block, stored, first,
                                    bodies[a].displacement,
                                    bodies[b].displacement, dt);
                        }
                }
                jointIndex = first ? joints[jointIndex].avbd_next_a
                    : joints[jointIndex].avbd_next_b;
            }
            AvbdVector6 update = avbdZeroVector6();
            if (avbdSolve(block, update))
            {
                bool crossing = false;
                for (uint active = bodies[bodyIndex].contact_head;
                     active != kInvalidDense;)
                {
                    uint a = pairs[active] / count;
                    uint b = pairs[active] % count;
                    bool first = bodyIndex == a;
                    for (uint point = 0u;
                         point < PM_AVBD_MANIFOLD_COUNT(
                            manifolds, active); ++point)
                    {
                        Contact contact = PM_AVBD_MANIFOLD_CONTACT(
                            manifolds, active, point);
                        AvbdVector6 jacobianA;
                        AvbdVector6 jacobianB;
                        float error = rigid_avbd_contact_normal_row(
                            contact, a, b, bodies,
                            jacobianA, jacobianB);
                        float trial = contact.avbd_dual[0].lambda +
                            contact.avbd_dual[0].penalty * error;
                        AvbdVector6 normalRow = first
                            ? jacobianA : jacobianB;
                        if (trial <= 0.0f && trial +
                            contact.avbd_dual[0].penalty *
                                avbdDot6(normalRow, update) >
                            contact.avbd_dual[0].penalty * 1.0e-6f)
                        {
                            avbdAddRow(block, normalRow,
                                contact.avbd_dual[0].penalty, trial);
                            crossing = true;
                        }
                    }
                    active = first
                        ? PM_AVBD_MANIFOLD_NEXT_A(manifolds, active)
                        : PM_AVBD_MANIFOLD_NEXT_B(manifolds, active);
                }
                if (crossing)
                {
                    AvbdVector6 safeguarded = avbdZeroVector6();
                    if (avbdSolve(block, safeguarded))
                        update = safeguarded;
                }
                if (phase == 0u)
                {
                    rigid_avbd_apply_pose_update(
                        bodies[bodyIndex], update);
                }
                else
                {
                    AvbdBody body = bodies[bodyIndex];
                    for (uint axis = 0u; axis < 6u; ++axis)
                        body.displacement.values[axis] +=
                            update.values[axis];
                    bodies[bodyIndex] = body;
                }
            }
        }
    }
    if (operation == kAvbdStage_advance_duals)
    {
        for (uint active = lane; active < pair_count[0]; active += stride)
        {
            uint a = pairs[active] / count;
            uint b = pairs[active] % count;
            if (phase != 0u && bodies[a].impact == 0u &&
                bodies[b].impact == 0u) continue;
            float friction = sqrt(
                parameters[a].friction * parameters[b].friction);
            float beta = rigid_avbd_pair_stiffness(
                parameters[a], parameters[b], dt) * 100.0f;
            ContactManifold manifold = manifolds[active];
            for (uint point = 0u; point < manifold.count; ++point)
            {
                Contact contact = manifold.contacts[point];
                AvbdVector6 jacobianA[3];
                AvbdVector6 jacobianB[3];
                float error[3];
                rigid_avbd_contact_rows(contact, a, b, bodies,
                    jacobianA, jacobianB, error);
                AvbdContactForce force = rigid_avbd_contact_force(
                    contact, error, friction);
                float tangentX = contact.avbd_dual[1].lambda +
                    contact.avbd_dual[1].penalty * error[1];
                float tangentY = contact.avbd_dual[2].lambda +
                    contact.avbd_dual[2].penalty * error[2];
                float frictionBound = friction * force.normal;
                bool stick = tangentX * tangentX + tangentY * tangentY <=
                    frictionBound * frictionBound;
                contact.avbd_dual[0].lambda = force.normal;
                contact.avbd_dual[1].lambda = force.tangent0;
                contact.avbd_dual[2].lambda = force.tangent1;
                for (uint axis = 0u; axis < 3u; ++axis)
                    if (axis == 0u ? force.normal > 0.0f : stick)
                        contact.avbd_dual[axis].penalty = min(
                            kAvbdMaximumPenalty,
                            contact.avbd_dual[axis].penalty +
                                beta * abs(error[axis]));
                if (iteration + 1u == schedule[0].budget)
                {
                    Vec3 tangent0;
                    Vec3 tangent1;
                    rigid_avbd_basis(
                        contact.normal, tangent0, tangent1);
                    contact.accumulated_normal_impulse =
                        force.normal * dt;
                    contact.reported_normal_impulse =
                        contact.accumulated_normal_impulse;
                    contact.accumulated_friction_impulse = multiply(add(
                        multiply(tangent0, force.tangent0),
                        multiply(tangent1, force.tangent1)), dt);
                    contact.reported_friction_impulse =
                        contact.accumulated_friction_impulse;
                    if (!stick)
                    {
                        RigidBodyState poseA = phase != 0u
                            ? bodies[a].target : rigid_avbd_pose(bodies[a]);
                        RigidBodyState poseB = phase != 0u
                            ? bodies[b].target : rigid_avbd_pose(bodies[b]);
                        contact.avbd_anchor_a = inverse_rotate(
                            poseA.orientation,
                            subtract(contact.point, poseA.position));
                        contact.avbd_anchor_b = inverse_rotate(
                            poseB.orientation,
                            subtract(contact.point, poseB.position));
                    }
                }
                manifold.contacts[point] = contact;
            }
            manifolds[active] = manifold;
        }
        for (uint jointIndex = lane; jointIndex < joint_capacity;
             jointIndex += stride)
        {
            if (joints[jointIndex].avbd_count == 0u ||
                joints[jointIndex].state.enabled == 0u) continue;
            uint a = joints[jointIndex].geometry.dense_a;
            uint b = joints[jointIndex].geometry.dense_b;
            if (phase != 0u && bodies[a].impact == 0u &&
                bodies[b].impact == 0u) continue;
            float impulse = 0.0f;
            if (phase == 0u)
            {
                AvbdBlock scratch = avbdZeroBlock();
                impulse = rigid_avbd_visit_joint(joints[jointIndex],
                    parameters, previous, bodies, dt, scratch, true,
                    kAvbdAdvanceJoint);
            }
            else
            {
                for (uint rowIndex = 0u;
                     rowIndex < joints[jointIndex].avbd_count; ++rowIndex)
                {
                    AvbdRow row = rigid_avbd_row(
                        joints[jointIndex].avbd_rows[rowIndex]);
                    impulse += abs(row.stiffness < kAvbdMaximumPenalty
                        ? row.referenceForce
                        : avbdRowForce(row, bodies[a].displacement,
                            bodies[b].displacement, dt)) * dt;
                    if (row.stiffness >= kAvbdMaximumPenalty)
                    {
                        row = avbdAdvance(row, bodies[a].displacement,
                            bodies[b].displacement);
                        RigidConstraintResource joint =
                            joints[jointIndex];
                        joint.avbd_rows[rowIndex] =
                            rigid_avbd_stored_row(row);
                        joints[jointIndex] = joint;
                    }
                }
            }
            joints[jointIndex].state.applied_impulse = impulse;
            if (joints[jointIndex].options.breaking_impulse_threshold >
                    0.0f &&
                impulse >
                    joints[jointIndex].options.breaking_impulse_threshold)
            {
                joints[jointIndex].state.broken = true;
                joints[jointIndex].state.enabled = false;
            }
        }
    }
    if (operation == kAvbdStage_free_mean)
    {
        for (uint root = lane; root < count; root += stride)
        {
            if (parameters[root].motion != MotionType.dynamic ||
                bodies[root].parent != root) continue;
            AvbdBody rootBody = bodies[root];
            for (uint axis = 0u; axis < 3u; ++axis)
                rootBody.inertia.g[axis] = 0.0f;
            bodies[root] = rootBody;
            if ((velocityPhase && bodies[root].impact == 0u) ||
                (bodies[root].contact_head == kInvalidDense &&
                 bodies[root].joint_head == kInvalidDense)) continue;
            Vec3 weighted = {};
            float totalMass = 0.0f;
            bool anchored = false;
            for (uint bodyIndex = 0u; bodyIndex < count && !anchored; ++bodyIndex)
            {
                if (parameters[bodyIndex].motion != MotionType.dynamic ||
                    bodies[bodyIndex].parent != root) continue;
                float mass = 1.0f /
                    max(parameters[bodyIndex].inverse_mass, 1.0e-12f);
                weighted = add(weighted, multiply(rigid_avbd_translation(
                    bodies[bodyIndex].displacement), mass));
                totalMass += mass;
                for (uint active = bodies[bodyIndex].contact_head;
                     active != kInvalidDense;)
                {
                    uint a = pairs[active] / count;
                    uint b = pairs[active] % count;
                    bool first = bodyIndex == a;
                    anchored = anchored || parameters[first ? b : a].motion !=
                        MotionType.dynamic;
                    active = first ? PM_AVBD_MANIFOLD_NEXT_A(manifolds, active)
                        : PM_AVBD_MANIFOLD_NEXT_B(manifolds, active);
                }
                for (uint jointIndex = bodies[bodyIndex].joint_head;
                     jointIndex != kInvalidDense;)
                {
                    bool first = bodyIndex == joints[jointIndex].geometry.dense_a;
                    anchored = anchored || parameters[first
                        ? joints[jointIndex].geometry.dense_b
                        : joints[jointIndex].geometry.dense_a].motion
                        != MotionType.dynamic;
                    jointIndex = first ? joints[jointIndex].avbd_next_a
                        : joints[jointIndex].avbd_next_b;
                }
            }
            if (!anchored && totalMass > 0.0f)
            {
                weighted = multiply(weighted, 1.0f / totalMass);
                rootBody = bodies[root];
                rootBody.inertia.g[0] = weighted.x;
                rootBody.inertia.g[1] = weighted.y;
                rootBody.inertia.g[2] = weighted.z;
                bodies[root] = rootBody;
            }
        }
    }
    if (operation == kAvbdStage_free_subtract)
    {
        for (uint bodyIndex = lane; bodyIndex < count; bodyIndex += stride)
        {
            if (parameters[bodyIndex].motion != MotionType.dynamic ||
                (velocityPhase && bodies[bodyIndex].impact == 0u)) continue;
            AvbdBlock mean = bodies[bodies[bodyIndex].parent].inertia;
            AvbdBody body = bodies[bodyIndex];
            for (uint axis = 0u; axis < 3u; ++axis)
                body.displacement.values[axis] -= mean.g[axis];
            bodies[bodyIndex] = body;
        }
    }
    if (operation == kAvbdStage_free_clear)
    {
        for (uint bodyIndex = lane; bodyIndex < count; bodyIndex += stride)
        {
            AvbdBody body = bodies[bodyIndex];
            for (uint axis = 0u; axis < 3u; ++axis)
                body.inertia.g[axis] = 0.0f;
            bodies[bodyIndex] = body;
        }
    }
    if (operation == kAvbdStage_apply_states)
    {
        for (uint bodyIndex = lane; bodyIndex < count; bodyIndex += stride)
        {
            if (parameters[bodyIndex].motion != MotionType.dynamic) continue;
            if (phase != 0u && bodies[bodyIndex].impact == 0u) continue;
            RigidBodyState state = bodies[bodyIndex].target;
            AvbdVector6 delta = bodies[bodyIndex].displacement;
            if (phase != 0u)
            {
                state.linear_velocity = add(state.linear_velocity,
                    multiply(rigid_avbd_translation(delta), 1.0f / dt));
                state.angular_velocity = add(state.angular_velocity,
                    multiply(rigid_avbd_rotation(delta), 1.0f / dt));
                states[bodyIndex] = state;
                continue;
            }
            state.position = add(state.position,
                rigid_avbd_translation(delta));
            state.orientation = rigid_avbd_pose(bodies[bodyIndex]).orientation;
            state.linear_velocity = add(state.linear_velocity,
                multiply(rigid_avbd_translation(delta), 1.0f / dt));
            state.angular_velocity = add(state.angular_velocity,
                multiply(quaternion_rotation_vector(
                    bodies[bodyIndex].orientation_delta), 1.0f / dt));
            states[bodyIndex] = state;
        }
    }
    if (operation == kAvbdStage_finish_phase)
    {
        if (lane == 0u)
        {
            ContactSchedule value = schedule[0];
            value.counts[phase == 0u ? 6u : 7u] = value.budget;
            value.maximum_passes += value.budget;
            schedule[0] = value;
        }
    }
    if (operation == kAvbdStage_project_hinges && phase == 0u)
        rigid_avbd_project_hinge_anchors(
            parameters, joints, joint_capacity, states);
}

[shader("compute")]
[numthreads(PM_THREADS(32), 1, 1)]
void solve_avbd_stage_kernel(
    uniform PM_AVBD_BUFFER(BodyParameters) parameters,
    uniform PM_AVBD_BUFFER(RigidBodyState) previous,
    uniform PM_AVBD_BUFFER(RigidBodyState) states, uniform uint count,
    uniform PM_AVBD_MANIFOLD_BUFFER manifolds,
    uniform PM_AVBD_BUFFER(uint) pairs,
    uniform PM_AVBD_BUFFER(uint) pair_count,
    uniform PM_AVBD_BUFFER(RigidConstraintResource) joints,
    uniform uint joint_capacity,
    uniform PM_AVBD_BUFFER(AvbdBody) bodies,
    uniform PM_AVBD_BUFFER(ContactSchedule) schedule, uniform float dt,
    uniform uint operation, uniform uint phase,
    uniform uint iteration, uniform uint color PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    rigid_avbd_solve_stage(parameters, previous, states, count, manifolds, pairs,
        pair_count, joints, joint_capacity, bodies, schedule, dt,
        operation, phase, iteration, color);
}

[shader("compute")]
[numthreads(PM_THREADS(32), 1, 1)]
void solve_avbd_kernel(
    uniform PM_AVBD_BUFFER(BodyParameters) parameters,
    uniform PM_AVBD_BUFFER(RigidBodyState) previous,
    uniform PM_AVBD_BUFFER(RigidBodyState) states, uniform uint count,
    uniform PM_AVBD_MANIFOLD_BUFFER manifolds,
    uniform PM_AVBD_BUFFER(uint) pairs,
    uniform PM_AVBD_BUFFER(uint) pair_count,
    uniform PM_AVBD_BUFFER(RigidConstraintResource) joints,
    uniform uint joint_capacity,
    uniform PM_AVBD_BUFFER(AvbdBody) bodies,
    uniform PM_AVBD_BUFFER(ContactSchedule) schedule, uniform float dt
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    for (uint phase = 0u; phase < 2u; ++phase)
    {
        if (phase != 0u && schedule[0].counts[0] == 0u) break;
        if (phase != 0u)
        {
            for (uint operation = kAvbdStage_impact_gaps;
                 operation <= kAvbdStage_impact_constraints; ++operation)
            {
                rigid_avbd_solve_stage(parameters, previous, states, count,
                    manifolds, pairs, pair_count, joints, joint_capacity,
                    bodies, schedule, dt,
                    operation, phase, 0u, 0u);
                rigid_avbd_barrier();
            }
        }
        for (uint iteration = 0u; iteration < schedule[0].budget; ++iteration)
        {
            for (uint color = 0u; color < schedule[0].remaining; ++color)
            {
                rigid_avbd_solve_stage(parameters, previous, states, count,
                    manifolds, pairs, pair_count, joints, joint_capacity,
                    bodies, schedule, dt,
                    kAvbdStage_solve_color, phase, iteration, color);
                rigid_avbd_barrier();
            }
            rigid_avbd_solve_stage(parameters, previous, states, count,
                manifolds, pairs, pair_count, joints, joint_capacity,
                bodies, schedule, dt,
                kAvbdStage_advance_duals, phase, iteration, 0u);
            rigid_avbd_barrier();
        }
        for (uint operation = kAvbdStage_free_mean;
             operation <= kAvbdStage_finish_phase; ++operation)
        {
            rigid_avbd_solve_stage(parameters, previous, states, count,
                manifolds, pairs, pair_count, joints, joint_capacity,
                bodies, schedule, dt,
                operation, phase, 0u, 0u);
            rigid_avbd_barrier();
        }
    }
}
