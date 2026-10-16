// SPDX-License-Identifier: MIT
#include "dispatch_shape.slangh"
// Rigid triangle geometry, contact generation, and contact-cache kernels.
// Backend-neutral constraint solving lives in geometry_constraints.slang.

import types;
import math;
import device_intrinsics;
import graphics_resources;

#ifndef PM_TARGET_D3D12
#define PM_TARGET_D3D12 0
#endif

// DXIL does not support i8 structured-buffer memory operations.
#if PM_TARGET_D3D12
#define PM_GEOMETRY_UINT8 uint
#define PM_D3D12_DISPATCH_PARAMETER \
    , uint3 pmD3D12DispatchThreadId : SV_DispatchThreadID
#define PM_D3D12_DISPATCH_PROLOGUE() \
    pmSetD3D12DispatchThreadId(pmD3D12DispatchThreadId)
#else
#define PM_GEOMETRY_UINT8 uint8_t
#define PM_D3D12_DISPATCH_PARAMETER
#define PM_D3D12_DISPATCH_PROLOGUE()
#endif

struct RigidAvbdVector6 { float values[6]; };
struct RigidAvbdDual { float lambda; float penalty; };
struct RigidAvbdRow
{
    RigidAvbdVector6 a;
    RigidAvbdVector6 b;
    RigidAvbdDual dual;
    float error;
    float velocity;
    float lower;
    float upper;
    float stiffness;
    float damping;
    float beta;
    float reference_force;
};

struct BodyAccumulator
{
    Vec3 force;
    Vec3 torque;
    Vec3 impulse;
    Vec3 angular_impulse;
};

struct KinematicTarget
{
    RigidBodyState state;
    uint active;
};

enum RigidConstraintType : uint
{
    fixed,
    point,
    hinge,
    slider,
    piston,
    generic,
    generic_spring,
    motor,
};

struct RigidConstraintLimitOptions
{
    uint axes;
    Vec3 lower;
    Vec3 upper;
};

struct RigidConstraintSpringOptions
{
    uint axes;
    Vec3 stiffness;
    Vec3 damping;
};

struct RigidConstraintMotorOptions
{
    uint linear_enabled;
    uint angular_enabled;
    float linear_target_velocity;
    float linear_maximum_impulse;
    float angular_target_velocity;
    float angular_maximum_impulse;
};

struct RigidConstraintOptions
{
    RigidConstraintType type;
    RigidBodyId body_a;
    RigidBodyId body_b;
    Vec3 local_anchor_a;
    Vec3 local_anchor_b;
    Quaternion local_orientation_a;
    Quaternion local_orientation_b;
    RigidConstraintLimitOptions linear_limits;
    RigidConstraintLimitOptions angular_limits;
    RigidConstraintSpringOptions linear_springs;
    RigidConstraintSpringOptions angular_springs;
    RigidConstraintMotorOptions motor;
    uint enabled;
    uint disable_collisions;
    float breaking_impulse_threshold;
    uint solver_iterations;
};

struct RigidConstraintState
{
    uint enabled;
    uint broken;
    float applied_impulse;
};

struct RigidConstraintGeometry
{
    uint dense_a;
    uint dense_b;
};

struct RigidConstraintResource
{
    RigidConstraintOptions options;
    RigidConstraintState state;
    uint generation;
    uint alive;
    RigidConstraintGeometry geometry;
    RigidAvbdRow avbd_rows[18];
    uint avbd_count;
    uint avbd_next_a;
    uint avbd_next_b;
};

struct FixedContactProjection { uint root; };

struct HingeContactFrame
{
    Vec3 anchor;
    Vec3 axis;
    Vec3 local_anchor;
    float inverse_mass;
    float inverse_moment;
    uint present;
    uint fixed;
    uint axial;
    uint axial_rotation;
    uint fixed_member;
    uint point_member;
    uint static_body;
};

struct Contact
{
    Vec3 normal;
    Vec3 point;
    float penetration;
    HingeContactFrame body_hinge;
    HingeContactFrame collider_hinge;
    float accumulated_normal_impulse;
    Vec3 accumulated_friction_impulse;
    float initial_normal_speed;
    uint warm_started;
    uint persistent;
    float impact_fraction;
    float reported_normal_impulse;
    Vec3 reported_friction_impulse;
    RigidAvbdDual avbd_dual[3];
    Vec3 avbd_error;
    Vec3 avbd_anchor_a;
    Vec3 avbd_anchor_b;
    uint avbd_cached;
};

struct ContactManifold
{
    Contact contacts[8];
    uint count;
    Vec3 initial_relative_position;
    uint face_patch;
    uint response_slot;
    uint avbd_next_a;
    uint avbd_next_b;
};

struct CachedContact
{
    Vec3 local_point;
    Vec3 normal;
    float normal_impulse;
    Vec3 friction_impulse;
    RigidAvbdDual avbd_dual[3];
    Vec3 avbd_anchor_a;
    Vec3 avbd_anchor_b;
};

struct CachedContactPair
{
    uint pair;
    RigidBodyId body;
    RigidBodyId collider;
    uint64_t epoch;
    float timestep;
    uint count;
    CachedContact contacts[8];
};

struct ContactSchedule
{
    uint counts[29];
    uint offsets[29];
    uint budget;
    uint remaining;
    uint island_count;
    uint early_exit_count;
    uint maximum_passes;
    uint blocks;
    uint candidates;
    uint contacts;
};

struct LeafPair
{
    uint body_first;
    uint body_count;
    uint collider_first;
    uint collider_count;
};

struct RigidContactEvent
{
    RigidBodyId body;
    RigidBodyId collider;
    Vec3 position;
    Vec3 normal;
    float penetration;
    float normal_impulse;
    Vec3 friction_impulse;
};

struct PhysicsDebugRigidSample
{
    RigidBodyId id;
    RigidBodyState state;
    Vec3 applied_force;
    Vec3 applied_torque;
};

uint find_rigid_body_dense(RigidBodyId id, PMBuffer<RigidBodyId> ids, uint count)
{
    for (uint index = 0u; index < count; ++index)
        if (ids[index].index == id.index &&
            ids[index].generation == id.generation) return index;
    return kInvalidDense;
}

bool same_id(RigidBodyId a, RigidBodyId b)
{
    return a.index == b.index && a.generation == b.generation;
}

uint rigid_find_dense(RigidBodyId id, PMBuffer<RigidBodyId> ids, uint count)
{
    for (uint index = 0u; index < count; ++index)
        if (same_id(ids[index], id)) return index;
    return kInvalidDense;
}

bool rigid_guided_pair(HingeContactFrame a, HingeContactFrame b)
{
    return (a.axial != 0u && b.static_body != 0u) ||
        (b.axial != 0u && a.static_body != 0u);
}

bool rigid_guided_contact(Contact contact)
{
    return rigid_guided_pair(contact.body_hinge, contact.collider_hinge);
}

float rigid_fixed_hinge_inverse_moment(BodyParameters parameters,
    RigidBodyState state, HingeContactFrame frame)
{
    if ((frame.fixed == 0u && frame.axial_rotation == 0u) ||
        parameters.inverse_mass <= kEpsilon) return 0.0f;
    Vec3 localAxis = inverse_rotate(state.orientation, frame.axis);
    float centerMoment = localAxis.x * localAxis.x /
            max(parameters.inverse_inertia_local.x, kEpsilon) +
        localAxis.y * localAxis.y /
            max(parameters.inverse_inertia_local.y, kEpsilon) +
        localAxis.z * localAxis.z /
            max(parameters.inverse_inertia_local.z, kEpsilon);
    Vec3 centerArm = subtract(state.position, frame.anchor);
    Vec3 perpendicular = subtract(centerArm,
        multiply(frame.axis, dot(centerArm, frame.axis)));
    float pivotMoment = centerMoment + length_squared(perpendicular) /
        parameters.inverse_mass;
    return pivotMoment > kEpsilon ? 1.0f / pivotMoment : 0.0f;
}

HingeContactFrame rigid_hinge_contact_frame(
    PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyId> ids, uint bodyCount,
    uint dense, PMBuffer<RigidBodyState> states, __constref RigidBodyState state,
    PMBuffer<RigidConstraintResource> constraints, uint constraintCapacity,
    out Vec3 reference)
{
    HingeContactFrame result = {};
    result.static_body = parameters[dense].motion == MotionType.static_body;
    if (parameters[dense].inverse_mass <= 0.0f)
    {
        reference = state.position;
        return result;
    }
    RigidBodyId body = ids[dense];
    uint incidentCount = 0u;
    for (uint index = 0u; index < constraintCapacity; ++index)
    {
        if (constraints[index].alive == 0u ||
            constraints[index].options.enabled == 0u ||
            constraints[index].state.broken != 0u) continue;
        bool isA = same_id(constraints[index].options.body_a, body);
        bool isB = same_id(constraints[index].options.body_b, body);
        if (!isA && !isB) continue;
        ++incidentCount;
        if (constraints[index].options.type == RigidConstraintType.fixed)
            result.fixed_member = true;
        if (constraints[index].options.type == RigidConstraintType.point)
            result.point_member = true;
        RigidConstraintType type = constraints[index].options.type;
        bool axial = type == RigidConstraintType.piston ||
                     type == RigidConstraintType.slider;
        RigidBodyId other = isA ? constraints[index].options.body_b
                                : constraints[index].options.body_a;
        uint otherDense = rigid_find_dense(other, ids, bodyCount);
        bool staticAnchor = otherDense != kInvalidDense &&
            parameters[otherDense].motion == MotionType.static_body;
        if (result.present != 0u || (type != RigidConstraintType.hinge &&
            !(axial && staticAnchor &&
              constraints[index].options.breaking_impulse_threshold <= 0.0f)))
            continue;
        Vec3 localAxis = axial ? Vec3(1.0f, 0.0f, 0.0f)
                               : Vec3(0.0f, 0.0f, 1.0f);
        result.local_anchor = isA
            ? constraints[index].options.local_anchor_a
            : constraints[index].options.local_anchor_b;
        Quaternion localOrientation = isA
            ? constraints[index].options.local_orientation_a
            : constraints[index].options.local_orientation_b;
        result.anchor = add(state.position,
            rotate(state.orientation, result.local_anchor));
        result.axis = normalized_or(rotate(quaternion_multiply(
            state.orientation, localOrientation), localAxis), localAxis);
        result.present = true;
        result.fixed = staticAnchor && !axial;
        result.axial = staticAnchor && axial;
        result.axial_rotation = result.axial != 0u &&
            type == RigidConstraintType.piston;
        if (staticAnchor)
        {
            Vec3 otherLocalAnchor = isA
                ? constraints[index].options.local_anchor_b
                : constraints[index].options.local_anchor_a;
            Quaternion otherLocalOrientation = isA
                ? constraints[index].options.local_orientation_b
                : constraints[index].options.local_orientation_a;
            RigidBodyState otherState = states[otherDense];
            result.anchor = add(otherState.position,
                rotate(otherState.orientation, otherLocalAnchor));
            result.axis = normalized_or(rotate(quaternion_multiply(
                otherState.orientation, otherLocalOrientation), localAxis),
                result.axis);
        }
    }
    if (result.axial != 0u && incidentCount != 1u)
    {
        result.axial = false;
        result.axial_rotation = false;
        result.present = false;
    }
    if (result.axial != 0u)
    {
        result.inverse_mass = parameters[dense].inverse_mass;
        result.inverse_moment = rigid_fixed_hinge_inverse_moment(
            parameters[dense], state, result);
    }
    reference = result.present != 0u && result.axial == 0u
        ? result.anchor : state.position;
    return result;
}

bool constrained_collision_disabled(
    RigidBodyId a, RigidBodyId b, PMBuffer<RigidConstraintResource> constraints,
    uint capacity)
{
    for (uint index = 0u; index < capacity; ++index)
    {
        RigidConstraintResource constraint = constraints[index];
        if (constraint.alive == 0u || constraint.options.enabled == 0u ||
            constraint.state.broken != 0u ||
            constraint.options.disable_collisions == 0u)
            continue;
        if ((same_id(constraint.options.body_a, a) &&
             same_id(constraint.options.body_b, b)) ||
            (same_id(constraint.options.body_a, b) &&
             same_id(constraint.options.body_b, a))) return true;
    }
    return false;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_integrate_rigid_bodies(
    uniform PMBuffer<BodyParameters> parameters, uniform PMBuffer<BodyAccumulator> accumulators,
    uniform PMBuffer<KinematicTarget> targets, uniform PMBuffer<RigidBodyState> input,
    uniform PMBuffer<RigidBodyState> output, uniform uint count, uniform Vec3 gravity,
    uniform float timestep, uniform uint remaining_substeps,
    uniform bool apply_impulses PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index >= count) return;
    BodyParameters body = parameters[index];
    RigidBodyState previous = input[index], next = previous;
    if (body.motion == MotionType.static_body)
    {
        next.linear_velocity = {};
        next.angular_velocity = {};
        output[index] = next;
        return;
    }
    if (body.motion == MotionType.kinematic)
    {
        if (targets[index].active != 0u)
        {
            float fraction = 1.0f / float(remaining_substeps);
            next.position = add(previous.position, multiply(subtract(
                targets[index].state.position, previous.position), fraction));
            Quaternion target = targets[index].state.orientation;
            float orientationDot = previous.orientation.x * target.x +
                previous.orientation.y * target.y +
                previous.orientation.z * target.z +
                previous.orientation.w * target.w;
            if (orientationDot < 0.0f)
                target = {-target.x, -target.y, -target.z, -target.w};
            next.orientation = normalized_quaternion({
                previous.orientation.x +
                    (target.x - previous.orientation.x) * fraction,
                previous.orientation.y +
                    (target.y - previous.orientation.y) * fraction,
                previous.orientation.z +
                    (target.z - previous.orientation.z) * fraction,
                previous.orientation.w +
                    (target.w - previous.orientation.w) * fraction});
            next.linear_velocity = multiply(
                subtract(next.position, previous.position), 1.0f / timestep);
            next.angular_velocity = quaternion_delta_velocity(
                previous.orientation, next.orientation, timestep);
        }
        else
        {
            next.linear_velocity = {};
            next.angular_velocity = {};
        }
        output[index] = next;
        return;
    }
    BodyAccumulator accumulator = accumulators[index];
    next.linear_velocity = add(next.linear_velocity, multiply(add(gravity,
        multiply(accumulator.force, body.inverse_mass)), timestep));
    next.angular_velocity = add(next.angular_velocity,
        multiply(inverse_inertia_world(body, next, accumulator.torque),
            timestep));
    if (apply_impulses)
    {
        next.linear_velocity = add(next.linear_velocity,
            multiply(accumulator.impulse, body.inverse_mass));
        next.angular_velocity = add(next.angular_velocity,
            inverse_inertia_world(body, next, accumulator.angular_impulse));
    }
    next.linear_velocity = multiply(next.linear_velocity,
        1.0f / (1.0f + body.linear_damping * timestep));
    next.angular_velocity = multiply(next.angular_velocity,
        1.0f / (1.0f + body.angular_damping * timestep));
    next.linear_velocity = clamp_length(
        next.linear_velocity, body.maximum_linear_speed);
    next.angular_velocity = clamp_length(
        next.angular_velocity, body.maximum_angular_speed);
    next.position = add(next.position, multiply(next.linear_velocity, timestep));
    Quaternion angular = {next.angular_velocity.x, next.angular_velocity.y,
                          next.angular_velocity.z, 0.0f};
    Quaternion derivative = quaternion_multiply(angular, next.orientation);
    next.orientation = normalized_quaternion({
        next.orientation.x + 0.5f * derivative.x * timestep,
        next.orientation.y + 0.5f * derivative.y * timestep,
        next.orientation.z + 0.5f * derivative.z * timestep,
        next.orientation.w + 0.5f * derivative.w * timestep});
    output[index] = next;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_compute_rigid_world_bounds(
    uniform PMBuffer<BodyParameters> parameters,
    uniform PMBuffer<RigidBodyState> previous_states, uniform PMBuffer<RigidBodyState> states,
    uniform uint count, uniform PMBuffer<TriangleMeshResource> meshes,
    uniform PMBuffer<WorldAabb> world_bounds PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index >= count) return;
    TriangleMeshResource mesh = meshes[parameters[index].mesh.index];
    bool swept = vector_length(subtract(states[index].position,
        previous_states[index].position)) + rotational_motion_bound(
            previous_states[index], states[index], mesh) >
        parameters[index].collision_margin;
    BoundsTransform previousTransform = {};
    if (swept) previousTransform = bounds_transform(previous_states[index]);
    Vec3 minimum = {}, maximum = {};
    transformed_motion_bounds(mesh.minimum, mesh.maximum, previousTransform,
        bounds_transform(states[index]), swept, 0.0f, minimum, maximum);
    world_bounds[index] = {minimum, maximum};
}

bool rigid_bounding_spheres_may_contact(
    __constref RigidBodyState previousBody,
    __constref RigidBodyState body,
    __constref TriangleMeshResource bodyMesh,
    __constref RigidBodyState previousCollider,
    __constref RigidBodyState collider,
    __constref TriangleMeshResource colliderMesh,
    float margin)
{
    Vec3 previousRelative = subtract(
        add(previousBody.position,
            rotate(previousBody.orientation, bodyMesh.bounding_center)),
        add(previousCollider.position,
            rotate(previousCollider.orientation,
                colliderMesh.bounding_center)));
    Vec3 currentRelative = subtract(
        add(body.position,
            rotate(body.orientation, bodyMesh.bounding_center)),
        add(collider.position,
            rotate(collider.orientation, colliderMesh.bounding_center)));
    Vec3 movement = subtract(currentRelative, previousRelative);
    float squaredMovement = length_squared(movement);
    float time = squaredMovement > kEpsilon * kEpsilon
        ? clamp_scalar(-dot(previousRelative, movement) / squaredMovement,
            0.0f, 1.0f)
        : 0.0f;
    Vec3 nearest = add(previousRelative, multiply(movement, time));
    float radius = bodyMesh.bounding_radius + colliderMesh.bounding_radius +
        margin + 1.0e-5f +
        rotational_motion_bound(previousBody, body, bodyMesh) +
        rotational_motion_bound(previousCollider, collider, colliderMesh);
    return length_squared(nearest) <= radius * radius;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_broad_phase_rigid_pairs(
    uniform PMBuffer<BodyParameters> parameters, uniform PMBuffer<WorldAabb> world_bounds,
    uniform PMBuffer<RigidBodyState> previous_states, uniform PMBuffer<RigidBodyState> states,
    uniform PMBuffer<TriangleMeshResource> meshes, uniform PMBuffer<RigidBodyId> ids,
    uniform PMBuffer<RigidConstraintResource> constraints,
    uniform uint constraint_capacity,
    uniform PMBuffer<FixedContactProjection> fixed_groups, uniform uint count,
    uniform PMBuffer<PMResourceUint8> active_flags
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint pair = pmThreadIndex();
    if (pair >= count * count) return;
    uint bodyIndex = pair / count, colliderIndex = pair % count;
    bool active = parameters[bodyIndex].motion == MotionType.dynamic &&
        bodyIndex != colliderIndex;
    if (active && parameters[colliderIndex].motion == MotionType.dynamic &&
        colliderIndex < bodyIndex) active = false;
    if (active && pmBufferPresent(fixed_groups) &&
        fixed_groups[bodyIndex].root == fixed_groups[colliderIndex].root)
        active = false;
    if (active && constrained_collision_disabled(ids[bodyIndex],
        ids[colliderIndex], constraints, constraint_capacity)) active = false;
    if (active)
    {
        float margin = parameters[bodyIndex].collision_margin +
            parameters[colliderIndex].collision_margin;
        WorldAabb a = world_bounds[bodyIndex];
        WorldAabb b = world_bounds[colliderIndex];
        Vec3 expansion = {margin, margin, margin};
        active = bounds_overlap(subtract(a.minimum, expansion),
            add(a.maximum, expansion), b.minimum, b.maximum);
        if (active)
            active = rigid_bounding_spheres_may_contact(
                previous_states[bodyIndex], states[bodyIndex],
                meshes[parameters[bodyIndex].mesh.index],
                previous_states[colliderIndex], states[colliderIndex],
                meshes[parameters[colliderIndex].mesh.index], margin);
    }
    active_flags[pair] = active ? PMResourceUint8(1u) : PMResourceUint8(0u);
}

void rigid_closest_segments(Vec3 p0, Vec3 p1, Vec3 q0, Vec3 q1,
    out Vec3 onP, out Vec3 onQ)
{
    Vec3 d0 = subtract(p1, p0);
    Vec3 d1 = subtract(q1, q0);
    Vec3 r = subtract(p0, q0);
    float a = dot(d0, d0);
    float e = dot(d1, d1);
    float s = 0.0f;
    float t = 0.0f;
    if (a <= 1.0e-12f && e <= 1.0e-12f)
    {
        onP = p0;
        onQ = q0;
        return;
    }
    if (a <= 1.0e-12f)
    {
        t = clamp_scalar(dot(d1, r) / e, 0.0f, 1.0f);
    }
    else
    {
        float c = dot(d0, r);
        if (e <= 1.0e-12f)
        {
            s = clamp_scalar(-c / a, 0.0f, 1.0f);
        }
        else
        {
            float b = dot(d0, d1);
            float f = dot(d1, r);
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
    onP = add(p0, multiply(d0, s));
    onQ = add(q0, multiply(d1, t));
}

public Vec3 rigid_closest_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c,
    bool stableFace)
{
    Vec3 ab = subtract(b, a);
    Vec3 ac = subtract(c, a);
    Vec3 ap = subtract(p, a);
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    Vec3 bp = subtract(p, b);
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
        return add(a, multiply(ab, d1 / (d1 - d3)));
    Vec3 cp = subtract(p, c);
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
        return add(a, multiply(ac, d2 / (d2 - d6)));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f)
        return add(b, multiply(subtract(c, b),
            (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    if (stableFace)
    {
        Vec3 normal = cross(ab, ac);
        return subtract(p, multiply(normal,
            dot(ap, normal) / length_squared(normal)));
    }
    float inverse = 1.0f / (va + vb + vc);
    return add(a, add(multiply(ab, vb * inverse),
                      multiply(ac, vc * inverse)));
}

bool rigid_point_in_triangle(Vec3 point, Vec3 a, Vec3 b, Vec3 c,
    Vec3 normal)
{
    float tolerance = -1.0e-5f * length_squared(normal);
    return dot(cross(subtract(b, a), subtract(point, a)), normal) >= tolerance &&
           dot(cross(subtract(c, b), subtract(point, b)), normal) >= tolerance &&
           dot(cross(subtract(a, c), subtract(point, c)), normal) >= tolerance;
}

public bool rigid_segment_hits_triangle(Vec3 first, Vec3 second,
    Vec3 a, Vec3 b, Vec3 c, out Vec3 intersection)
{
    intersection = {};
    Vec3 normal = cross(subtract(b, a), subtract(c, a));
    Vec3 direction = subtract(second, first);
    float denominator = dot(normal, direction);
    if (length_squared(normal) <= kEpsilon * kEpsilon ||
        abs(denominator) <= kEpsilon) return false;
    float amount = dot(normal, subtract(a, first)) / denominator;
    if (amount < 0.0f || amount > 1.0f) return false;
    intersection = add(first, multiply(direction, amount));
    return rigid_point_in_triangle(intersection, a, b, c, normal);
}

public void rigid_consider_pair(Vec3 a, Vec3 b, inout float bestSquared,
    inout Vec3 bestA, inout Vec3 bestB)
{
    float squared = length_squared(subtract(a, b));
    if (squared < bestSquared)
    {
        bestSquared = squared;
        bestA = a;
        bestB = b;
    }
}

public void rigid_closest_triangle_pair(Vec3 a0, Vec3 a1, Vec3 a2,
    Vec3 b0, Vec3 b1, Vec3 b2, out Vec3 bestA, out Vec3 bestB,
    bool stableFace)
{
    Vec3 av[3] = {a0, a1, a2};
    Vec3 bv[3] = {b0, b1, b2};
    Vec3 intersection = {};
    for (uint edge = 0u; edge < 3u; ++edge)
    {
        if (rigid_segment_hits_triangle(av[edge], av[(edge + 1u) % 3u],
            b0, b1, b2, intersection))
        {
            bestA = intersection; bestB = intersection; return;
        }
    }
    for (uint edge = 0u; edge < 3u; ++edge)
    {
        if (rigid_segment_hits_triangle(bv[edge], bv[(edge + 1u) % 3u],
            a0, a1, a2, intersection))
        {
            bestA = intersection; bestB = intersection; return;
        }
    }
    float bestSquared = 3.402823466e+38f;
    for (uint i = 0u; i < 3u; ++i)
        rigid_consider_pair(av[i], rigid_closest_triangle(av[i], b0, b1, b2,
                            stableFace),
                            bestSquared, bestA, bestB);
    for (uint i = 0u; i < 3u; ++i)
        rigid_consider_pair(rigid_closest_triangle(bv[i], a0, a1, a2,
                            stableFace), bv[i],
                            bestSquared, bestA, bestB);
    for (uint i = 0u; i < 3u; ++i)
        for (uint j = 0u; j < 3u; ++j)
        {
            Vec3 onA;
            Vec3 onB;
            rigid_closest_segments(av[i], av[(i + 1u) % 3u],
                                   bv[j], bv[(j + 1u) % 3u], onA, onB);
            rigid_consider_pair(onA, onB, bestSquared, bestA, bestB);
        }
}

public bool rigid_triangle_bounds_overlap(Vec3 a0, Vec3 a1, Vec3 a2,
    Vec3 b0, Vec3 b1, Vec3 b2, float margin)
{
    Vec3 expansion = {margin, margin, margin};
    Vec3 minimumA = subtract(component_min(a0, component_min(a1, a2)), expansion);
    Vec3 maximumA = add(component_max(a0, component_max(a1, a2)), expansion);
    Vec3 minimumB = component_min(b0, component_min(b1, b2));
    Vec3 maximumB = component_max(b0, component_max(b1, b2));
    return bounds_overlap(minimumA, maximumA, minimumB, maximumB);
}

Vec3 rigid_guide_direction(Contact contact)
{
    HingeContactFrame frame = contact.body_hinge.axial != 0u
        ? contact.body_hinge : contact.collider_hinge;
    return {dot(contact.normal, frame.axis) * sqrt(frame.inverse_mass),
        frame.axial_rotation != 0u ? dot(frame.axis,
            cross(subtract(contact.point, frame.anchor), contact.normal)) *
            sqrt(frame.inverse_moment) : 0.0f,
        0.0f};
}

void rigid_add_contact(inout ContactManifold manifold, Contact contact,
    float spacing)
{
    bool guided = rigid_guided_contact(contact);
    Vec3 candidateDirection = guided ? rigid_guide_direction(contact) : Vec3();
    float candidateLength = guided ? vector_length(candidateDirection) : 1.0f;
    if (guided && candidateLength * candidateLength <= kEpsilon) return;
    if (guided && contact.impact_fraction > 0.0f)
    {
        float firstImpact = 1.0f;
        for (uint row = 0u; row < manifold.count; ++row)
            if (manifold.contacts[row].impact_fraction > 0.0f)
                firstImpact = min(firstImpact,
                    manifold.contacts[row].impact_fraction);
        if (contact.impact_fraction > firstImpact + 1.0e-4f) return;
        if (contact.impact_fraction < firstImpact - 1.0e-4f)
        {
            uint kept = 0u;
            for (uint row = 0u; row < manifold.count; ++row)
                if (manifold.contacts[row].impact_fraction == 0.0f)
                    manifold.contacts[kept++] = manifold.contacts[row];
            manifold.count = kept;
        }
    }
    float spacingSquared = spacing * spacing;
    for (uint i = 0u; i < manifold.count; ++i)
    {
        if (guided)
        {
            Vec3 direction = rigid_guide_direction(manifold.contacts[i]);
            float size = vector_length(direction);
            if (dot(candidateDirection, direction) >
                0.9999f * candidateLength * size)
            {
                if (contact.penetration / candidateLength >
                    manifold.contacts[i].penetration / size)
                    manifold.contacts[i] = contact;
                return;
            }
            continue;
        }
        if (length_squared(subtract(manifold.contacts[i].point,
                                    contact.point)) < spacingSquared)
        {
            if (contact.penetration > manifold.contacts[i].penetration)
                manifold.contacts[i] = contact;
            return;
        }
    }
    if (manifold.count < 8u)
    {
        manifold.contacts[manifold.count++] = contact;
        return;
    }
    uint shallowest = 0u;
    for (uint i = 1u; i < manifold.count; ++i)
        if (manifold.contacts[i].penetration <
            manifold.contacts[shallowest].penetration) shallowest = i;
    if (contact.penetration > manifold.contacts[shallowest].penetration)
        manifold.contacts[shallowest] = contact;
}

float rigid_contact_normal_speed(RigidBodyState body,
    RigidBodyState collider, Vec3 point, Vec3 normal)
{
    Vec3 bodyVelocity = add(body.linear_velocity,
        cross(body.angular_velocity, subtract(point, body.position)));
    Vec3 colliderVelocity = add(collider.linear_velocity,
        cross(collider.angular_velocity, subtract(point, collider.position)));
    return dot(subtract(bodyVelocity, colliderVelocity), normal);
}

bool rigid_contact_reaches_rest_offset(RigidBodyState body,
    RigidBodyState collider, Vec3 point, Vec3 normal, float distance,
    float restOffset, float timestep)
{
    if (distance <= restOffset + 1.0e-5f) return true;
    float speed = rigid_contact_normal_speed(body, collider, point, normal);
    if (speed > 1.0e-5f) return false;
    return -speed * timestep + restOffset + 1.0e-5f >= distance;
}

void rigid_collide_triangle_ranges(__constref TriangleMeshResource aMesh,
    __constref RigidBodyState aPreviousState,
    __constref RigidBodyState aState,
    Vec3 bodyReference, __constref HingeContactFrame bodyHinge,
    uint aFirst, uint aCount,
    __constref TriangleMeshResource bMesh,
    __constref RigidBodyState bPreviousState,
    __constref RigidBodyState bState,
    __constref HingeContactFrame colliderHinge, uint bFirst, uint bCount,
    float margin, float timestep,
    inout ContactManifold manifold)
{
    if (rigid_guided_pair(bodyHinge, colliderHinge)) return;
    float restOffset = min(margin, 1.0e-3f);
    for (uint ai = 0u; ai < aCount; ++ai)
    {
        uint at = aFirst + ai;
        if (3u * at + 2u >= aMesh.index_count) break;
        Vec3 a0 = transform_point(aState, aMesh.vertices[aMesh.indices[3u * at]]);
        Vec3 a1 = transform_point(aState, aMesh.vertices[aMesh.indices[3u * at + 1u]]);
        Vec3 a2 = transform_point(aState, aMesh.vertices[aMesh.indices[3u * at + 2u]]);
        for (uint bi = 0u; bi < bCount; ++bi)
        {
            uint bt = bFirst + bi;
            if (3u * bt + 2u >= bMesh.index_count) break;
            Vec3 b0 = transform_point(bState, bMesh.vertices[bMesh.indices[3u * bt]]);
            Vec3 b1 = transform_point(bState, bMesh.vertices[bMesh.indices[3u * bt + 1u]]);
            Vec3 b2 = transform_point(bState, bMesh.vertices[bMesh.indices[3u * bt + 2u]]);
            if (!rigid_triangle_bounds_overlap(a0, a1, a2, b0, b1, b2, margin))
                continue;
            Vec3 pointA;
            Vec3 pointB;
            rigid_closest_triangle_pair(a0, a1, a2, b0, b1, b2,
                                        pointA, pointB, false);
            Vec3 delta = subtract(pointA, pointB);
            float squared = length_squared(delta);
            if (squared > margin * margin) continue;
            Vec3 colliderNormal = normalized_or(
                cross(subtract(b1, b0), subtract(b2, b0)),
                {0.0f, 1.0f, 0.0f});
            Vec3 fallback = dot(colliderNormal,
                subtract(bodyReference, pointB)) >= 0.0f
                ? colliderNormal : multiply(colliderNormal, -1.0f);
            float distance = sqrt(max(squared, 0.0f));
            Vec3 point = multiply(add(pointA, pointB), 0.5f);
            Vec3 planeProjection = subtract(pointA, multiply(colliderNormal,
                dot(subtract(pointA, b0), colliderNormal)));
            bool onTriangleFace = length_squared(subtract(planeProjection,
                rigid_closest_triangle(planeProjection, b0, b1, b2, false))) <=
                1.0e-10f;
            bool smallConvex = pmBufferPresent(aMesh.solid_planes) &&
                               aMesh.index_count <= 96u;
            Vec3 normal = (distance <= 1.0e-5f ||
                (smallConvex && onTriangleFace))
                ? fallback : normalized_or(delta, fallback);
            // A concave body's origin can lie in its cavity, on the wrong
            // side of the contacting face. Orient that face from the convex
            // body's previous position so contact is independent of body
            // allocation order and cannot recover further into the cavity.
            bool convexSurfaceFace = false;
            float faceSeparation = 0.0f;
            bool convexA = pmBufferPresent(aMesh.solid_planes);
            bool convexB = pmBufferPresent(bMesh.solid_planes);
            if (convexA != convexB && distance <= 1.0e-5f)
            {
                Vec3 face0 = convexA ? b0 : a0;
                Vec3 face1 = convexA ? b1 : a1;
                Vec3 face2 = convexA ? b2 : a2;
                RigidBodyState surfaceState = convexA ? bState : aState;
                RigidBodyState previousSurface = convexA
                    ? bPreviousState : aPreviousState;
                RigidBodyState previousConvex = convexA
                    ? aPreviousState : bPreviousState;
                Vec3 outward = normalized_or(cross(subtract(face1, face0),
                    subtract(face2, face0)), {});
                Vec3 localNormal = inverse_rotate(
                    surfaceState.orientation, outward);
                Vec3 oldNormal = rotate(
                    previousSurface.orientation, localNormal);
                Vec3 oldPoint = transform_point(previousSurface,
                    inverse_rotate(surfaceState.orientation,
                        subtract(face0, surfaceState.position)));
                if (dot(oldNormal,
                        subtract(previousConvex.position, oldPoint)) < 0.0f)
                    outward = multiply(outward, -1.0f);
                Vec3 incidentPoint = convexA ? pointA : pointB;
                Vec3 projection = subtract(incidentPoint, multiply(outward,
                    dot(subtract(incidentPoint, face0), outward)));
                convexSurfaceFace = length_squared(outward) > 0.5f &&
                    length_squared(subtract(projection,
                        rigid_closest_triangle(projection, face0, face1,
                            face2, false))) <= 1.0e-10f;
                if (convexSurfaceFace)
                {
                    normal = multiply(outward, convexA ? 1.0f : -1.0f);
                    Vec3 incident0 = convexA ? a0 : b0;
                    Vec3 incident1 = convexA ? a1 : b1;
                    Vec3 incident2 = convexA ? a2 : b2;
                    faceSeparation = min(
                        dot(subtract(incident0, face0), outward),
                        min(dot(subtract(incident1, face0), outward),
                            dot(subtract(incident2, face0), outward)));
                }
            }
            if (!rigid_contact_reaches_rest_offset(aState, bState, point,
                normal, distance, restOffset, timestep)) continue;
            Contact contact = {};
            contact.normal = normal;
            contact.point = point;
            contact.body_hinge = bodyHinge;
            contact.collider_hinge = colliderHinge;
            contact.penetration = restOffset - distance + 1.0e-5f;
            if (convexSurfaceFace)
            {
                contact.penetration = min(margin,
                    restOffset - faceSeparation) + 1.0e-5f;
            }
            else if (distance <= 1.0e-5f)
            {
                if (bodyHinge.fixed_member != 0u ||
                    colliderHinge.fixed_member != 0u ||
                    smallConvex)
                {
                    float intersectionDepth = max(0.0f, -min(
                        dot(subtract(a0, pointB), normal), min(
                        dot(subtract(a1, pointB), normal),
                        dot(subtract(a2, pointB), normal))));
                    contact.penetration = min(margin,
                        intersectionDepth + restOffset) + 1.0e-5f;
                }
                else contact.penetration = margin + 1.0e-5f;
            }
            contact.impact_fraction = 1.0f;
            contact.initial_normal_speed = rigid_contact_normal_speed(
                aState, bState, point, normal);
            rigid_add_contact(manifold, contact, max(2.0f * margin, 1.0e-4f));
        }
    }
}

bool rigid_requires_swept_pair_contact(__constref RigidBodyState previousA,
    __constref RigidBodyState currentA,
    __constref TriangleMeshResource meshA,
    __constref RigidBodyState previousB,
    __constref RigidBodyState currentB,
    __constref TriangleMeshResource meshB, float threshold)
{
    Vec3 relativeTranslation = subtract(
        subtract(currentA.position, previousA.position),
        subtract(currentB.position, previousB.position));
    return vector_length(relativeTranslation) +
        rotational_motion_bound(previousA, currentA, meshA) +
        rotational_motion_bound(previousB, currentB, meshB) > threshold;
}

bool rigid_convex_face_manifold(RigidBodyState bodyState,
    TriangleMeshResource bodyMesh, HingeContactFrame bodyHinge,
    RigidBodyState colliderState, TriangleMeshResource colliderMesh,
    HingeContactFrame colliderHinge, float margin,
    out ContactManifold output)
{
    output = {};
    if (!pmBufferPresent(bodyMesh.solid_planes) ||
        !pmBufferPresent(colliderMesh.solid_planes) ||
        bodyMesh.index_count > 96u || colliderMesh.index_count > 96u)
        return false;
    float bestSeparation = -3.402823466e+38f;
    uint referenceFace = 0u;
    bool referenceIsBody = false;
    for (uint side = 0u; side < 2u; ++side)
    {
        TriangleMeshResource referenceMesh = side == 0u
            ? colliderMesh : bodyMesh;
        RigidBodyState referenceState = side == 0u
            ? colliderState : bodyState;
        TriangleMeshResource incidentMesh = side == 0u
            ? bodyMesh : colliderMesh;
        RigidBodyState incidentState = side == 0u
            ? bodyState : colliderState;
        for (uint face = 0u; face < referenceMesh.index_count / 3u; ++face)
        {
            CollisionPlane plane = referenceMesh.solid_planes[face];
            Vec3 normal = rotate(referenceState.orientation, plane.normal);
            Vec3 incidentNormal = inverse_rotate(incidentState.orientation,
                                                  normal);
            float support = 3.402823466e+38f;
            for (uint vertex = 0u; vertex < incidentMesh.vertex_count; ++vertex)
                support = min(support,
                    dot(incidentNormal, incidentMesh.vertices[vertex]));
            float separation = support + dot(normal, subtract(
                incidentState.position, referenceState.position)) - plane.offset;
            if (separation > bestSeparation)
            {
                bestSeparation = separation;
                referenceFace = face;
                referenceIsBody = side != 0u;
            }
        }
    }
    if (bestSeparation > margin) return true;

    TriangleMeshResource referenceMesh = referenceIsBody
        ? bodyMesh : colliderMesh;
    RigidBodyState referenceState = referenceIsBody
        ? bodyState : colliderState;
    TriangleMeshResource incidentMesh = referenceIsBody
        ? colliderMesh : bodyMesh;
    RigidBodyState incidentState = referenceIsBody
        ? colliderState : bodyState;
    CollisionPlane reference = referenceMesh.solid_planes[referenceFace];
    Vec3 outward = rotate(referenceState.orientation, reference.normal);
    Vec3 incidentAxis = inverse_rotate(incidentState.orientation, outward);
    float alignment = 1.0f;
    uint incidentFace = 0u;
    for (uint face = 0u; face < incidentMesh.index_count / 3u; ++face)
    {
        float value = dot(incidentAxis,
            incidentMesh.solid_planes[face].normal);
        if (value < alignment)
        {
            alignment = value;
            incidentFace = face;
        }
    }
    if (alignment > -0.98f) return false;
    Vec3 normal = multiply(outward, referenceIsBody ? -1.0f : 1.0f);
    CollisionPlane incident = incidentMesh.solid_planes[incidentFace];
    ContactManifold manifold = {};
    for (uint triangle = 0u; triangle < incidentMesh.index_count / 3u;
         ++triangle)
    {
        CollisionPlane face = incidentMesh.solid_planes[triangle];
        if (dot(face.normal, incident.normal) < 0.99999f ||
            abs(face.offset - incident.offset) > 1.0e-5f) continue;
        Vec3 polygon[36];
        Vec3 clipped[36];
        uint count = 3u;
        for (uint corner = 0u; corner < 3u; ++corner)
            polygon[corner] = inverse_rotate(referenceState.orientation,
                subtract(transform_point(incidentState,
                    incidentMesh.vertices[incidentMesh.indices[
                        triangle * 3u + corner]]), referenceState.position));
        for (uint planeIndex = 0u;
             planeIndex < referenceMesh.index_count / 3u && count != 0u;
             ++planeIndex)
        {
            CollisionPlane plane = referenceMesh.solid_planes[planeIndex];
            float offset = plane.offset +
                (dot(plane.normal, reference.normal) > 0.99999f
                    ? margin : 0.0f);
            uint clippedCount = 0u;
            Vec3 previous = polygon[count - 1u];
            float previousDistance = dot(plane.normal, previous) - offset;
            for (uint vertex = 0u; vertex < count; ++vertex)
            {
                Vec3 current = polygon[vertex];
                float distance = dot(plane.normal, current) - offset;
                if ((distance <= 0.0f) != (previousDistance <= 0.0f))
                    clipped[clippedCount++] = add(previous,
                        multiply(subtract(current, previous),
                            previousDistance / (previousDistance - distance)));
                if (distance <= 0.0f) clipped[clippedCount++] = current;
                previous = current;
                previousDistance = distance;
            }
            count = clippedCount;
            for (uint vertex = 0u; vertex < count; ++vertex)
                polygon[vertex] = clipped[vertex];
        }
        for (uint vertex = 0u; vertex < count; ++vertex)
        {
            float distance = dot(reference.normal, polygon[vertex]) -
                             reference.offset;
            if (distance > margin) continue;
            Contact contact = {};
            contact.normal = normal;
            contact.point = transform_point(referenceState,
                subtract(polygon[vertex],
                    multiply(reference.normal, distance * 0.5f)));
            contact.penetration = -distance;
            contact.body_hinge = bodyHinge;
            contact.collider_hinge = colliderHinge;
            contact.impact_fraction = 1.0f;
            rigid_add_contact(manifold, contact,
                max(margin * 2.0f, 1.0e-4f));
        }
    }
    if (manifold.count == 0u && alignment > -0.999999f) return false;
    manifold.face_patch = true;
    output = manifold;
    return true;
}

Vec3 rigid_guided_triangle_normal(Vec3 a0, Vec3 a1, Vec3 a2,
    Vec3 b0, Vec3 b1, Vec3 b2, Vec3 pointA, Vec3 pointB, Vec3 fallback)
{
    Vec3 delta = subtract(pointA, pointB);
    Vec3 normal = normalized_or(delta, fallback);
    Vec3 faces[2] = {
        normalized_or(cross(subtract(a1, a0), subtract(a2, a0)), normal),
        normalized_or(cross(subtract(b1, b0), subtract(b2, b0)), normal)};
    Vec3 vertices[6] = {a0, a1, a2, b0, b1, b2};
    Vec3 points[2] = {pointA, pointB};
    float faceAlignment = -1.0f;
    Vec3 interiorNormal = {};
    for (uint side = 0u; side < 2u; ++side)
    {
        uint base = side * 3u;
        Vec3 raw = cross(subtract(vertices[base + 1u], vertices[base]),
                         subtract(vertices[base + 2u], vertices[base]));
        float tolerance = 1.0e-5f * length_squared(raw);
        bool interior = tolerance > 0.0f;
        for (uint edge = 0u; edge < 3u; ++edge)
            interior = interior && dot(cross(
                subtract(vertices[base + (edge + 1u) % 3u],
                         vertices[base + edge]),
                subtract(points[side], vertices[base + edge])), raw) > tolerance;
        float alignment = abs(dot(faces[side], normal));
        if (interior && alignment > faceAlignment)
        {
            faceAlignment = alignment;
            float projection = dot(delta, faces[side]);
            float direction = abs(projection) > kEpsilon
                ? projection : dot(fallback, faces[side]);
            interiorNormal = direction >= 0.0f ? faces[side]
                : multiply(faces[side], -1.0f);
        }
    }
    if (faceAlignment >= 0.0f) return interiorNormal;
    float bestError = 4.0f * kEpsilon * kEpsilon;
    if (length_squared(delta) > kEpsilon * kEpsilon)
    {
        for (uint side = 0u; side < 2u; ++side)
        {
            float projection = dot(delta, faces[side]);
            float error = length_squared(subtract(delta,
                multiply(faces[side], projection)));
            if (error < bestError)
            {
                bestError = error;
                normal = projection >= 0.0f ? faces[side]
                    : multiply(faces[side], -1.0f);
            }
        }
    }
    return normal;
}

bool rigid_triangle_pair_face_contact(Vec3 a[3], Vec3 b[3], Vec3 separation)
{
    Vec3 normal = normalized_or(separation, {});
    Vec3 faceA = normalized_or(cross(subtract(a[1], a[0]),
        subtract(a[2], a[0])), {});
    Vec3 faceB = normalized_or(cross(subtract(b[1], b[0]),
        subtract(b[2], b[0])), {});
    return abs(dot(normal, faceA)) > 0.9999f ||
           abs(dot(normal, faceB)) > 0.9999f;
}

void rigid_collide_triangle_ranges_swept(
    __constref RigidBodyState previousAState,
    __constref RigidBodyState aState,
    Vec3 previousBodyReference, Vec3 bodyReference,
    __constref HingeContactFrame bodyHinge,
    __constref TriangleMeshResource aMesh, uint aFirst, uint aCount,
    __constref RigidBodyState previousBState,
    __constref RigidBodyState bState,
    __constref TriangleMeshResource bMesh, uint bFirst, uint bCount,
    __constref HingeContactFrame colliderHinge,
    float margin, float timestep,
    inout ContactManifold manifold)
{
    bool guided = rigid_guided_pair(bodyHinge, colliderHinge);
    float restOffset = guided ? min(margin, 1.0e-4f)
                              : min(margin, 1.0e-3f);
    for (uint ai = 0u; ai < aCount; ++ai)
    {
        uint at = aFirst + ai;
        if (3u * at + 2u >= aMesh.index_count) break;
        Vec3 previousA[3];
        Vec3 currentA[3];
        for (uint vertex = 0u; vertex < 3u; ++vertex)
        {
            Vec3 local = aMesh.vertices[aMesh.indices[3u * at + vertex]];
            previousA[vertex] = transform_point(previousAState, local);
            currentA[vertex] = transform_point(aState, local);
        }
        Vec3 minimumA = component_min(previousA[0], currentA[0]);
        Vec3 maximumA = component_max(previousA[0], currentA[0]);
        for (uint vertex = 1u; vertex < 3u; ++vertex)
        {
            minimumA = component_min(minimumA,
                component_min(previousA[vertex], currentA[vertex]));
            maximumA = component_max(maximumA,
                component_max(previousA[vertex], currentA[vertex]));
        }
        for (uint bi = 0u; bi < bCount; ++bi)
        {
            uint bt = bFirst + bi;
            if (3u * bt + 2u >= bMesh.index_count) break;
            Vec3 previousB[3];
            Vec3 currentB[3];
            for (uint vertex = 0u; vertex < 3u; ++vertex)
            {
                Vec3 local = bMesh.vertices[bMesh.indices[3u * bt + vertex]];
                previousB[vertex] = transform_point(previousBState, local);
                currentB[vertex] = transform_point(bState, local);
            }
            Vec3 minimumB = component_min(previousB[0], currentB[0]);
            Vec3 maximumB = component_max(previousB[0], currentB[0]);
            for (uint vertex = 1u; vertex < 3u; ++vertex)
            {
                minimumB = component_min(minimumB,
                    component_min(previousB[vertex], currentB[vertex]));
                maximumB = component_max(maximumB,
                    component_max(previousB[vertex], currentB[vertex]));
            }
            Vec3 expansion = {margin, margin, margin};
            if (!bounds_overlap(subtract(minimumA, expansion),
                add(maximumA, expansion), minimumB, maximumB)) continue;

            Vec3 deltaA[3];
            Vec3 deltaB[3];
            float absoluteSpeed = 0.0f;
            for (uint vertex = 0u; vertex < 3u; ++vertex)
            {
                deltaA[vertex] = subtract(currentA[vertex], previousA[vertex]);
                deltaB[vertex] = subtract(currentB[vertex], previousB[vertex]);
                absoluteSpeed = max(absoluteSpeed, vector_length(deltaA[vertex]));
            }
            float colliderSpeed = 0.0f;
            for (uint vertex = 0u; vertex < 3u; ++vertex)
                colliderSpeed = max(colliderSpeed, vector_length(deltaB[vertex]));
            absoluteSpeed += colliderSpeed;
            Vec3 commonMotion = deltaB[0];
            float relativeA = 0.0f;
            float relativeB = 0.0f;
            for (uint vertex = 0u; vertex < 3u; ++vertex)
            {
                relativeA = max(relativeA, vector_length(
                    subtract(deltaA[vertex], commonMotion)));
                relativeB = max(relativeB, vector_length(
                    subtract(deltaB[vertex], commonMotion)));
            }
            float speedBound = min(absoluteSpeed, relativeA + relativeB);
            if (speedBound <= kEpsilon) continue;

            float time = 0.0f;
            bool startsNearContact = false;
            Vec3 separatingNormal = {};
            bool haveSeparatingNormal = false;
            for (uint iteration = 0u; iteration < 32u; ++iteration)
            {
                Vec3 a[3];
                Vec3 b[3];
                for (uint vertex = 0u; vertex < 3u; ++vertex)
                {
                    a[vertex] = add(previousA[vertex],
                        multiply(deltaA[vertex], time));
                    b[vertex] = add(previousB[vertex],
                        multiply(deltaB[vertex], time));
                }
                Vec3 pointA;
                Vec3 pointB;
                rigid_closest_triangle_pair(a[0], a[1], a[2],
                    b[0], b[1], b[2], pointA, pointB, guided);
                Vec3 delta = subtract(pointA, pointB);
                float distance = sqrt(max(0.0f, length_squared(delta)));
                float contactOffset = restOffset;
                if (guided && !rigid_triangle_pair_face_contact(a, b, delta))
                {
                    HingeContactFrame guide = bodyHinge.axial != 0u
                        ? bodyHinge : colliderHinge;
                    Vec3 direction = normalized_or(delta, {});
                    float axial = dot(direction, guide.axis);
                    float angular = dot(guide.axis,
                        cross(subtract(pointA, guide.anchor), direction));
                    if (abs(axial) > 1.0e-3f && abs(angular) > 1.0e-3f)
                        contactOffset = 0.0f;
                }
                if (iteration == 0u)
                    startsNearContact = distance <= contactOffset + 5.0e-5f;
                if (distance <= contactOffset + 1.0e-5f)
                {
                    if (iteration == 0u && !guided) break;
                    Vec3 colliderNormal = normalized_or(
                        cross(subtract(b[1], b[0]), subtract(b[2], b[0])),
                        {0.0f, 1.0f, 0.0f});
                    Vec3 center = add(previousAState.position,
                        multiply(subtract(aState.position,
                            previousAState.position), time));
                    Vec3 fallback = dot(colliderNormal,
                        subtract(center, pointB)) >= 0.0f
                        ? colliderNormal : multiply(colliderNormal, -1.0f);
                    if (bodyHinge.present != 0u)
                    {
                        Vec3 reference = add(previousBodyReference,
                            multiply(subtract(bodyReference,
                                previousBodyReference), time));
                        fallback = dot(colliderNormal,
                            subtract(reference, pointB)) >= 0.0f
                            ? colliderNormal : multiply(colliderNormal, -1.0f);
                    }
                    Vec3 outwardA = pmBufferPresent(aMesh.shell_normals)
                        ? rotate(aState.orientation, aMesh.shell_normals[at])
                        : Vec3();
                    Vec3 outwardB = pmBufferPresent(bMesh.shell_normals)
                        ? rotate(bState.orientation, bMesh.shell_normals[bt])
                        : Vec3();
                    if (guided && length_squared(outwardB) > 0.5f)
                        fallback = outwardB;
                    else if (guided && length_squared(outwardA) > 0.5f)
                        fallback = multiply(outwardA, -1.0f);
                    Vec3 planeProjection = subtract(pointA,
                        multiply(colliderNormal,
                            dot(subtract(pointA, b[0]), colliderNormal)));
                    bool onFace = length_squared(subtract(planeProjection,
                        rigid_closest_triangle(planeProjection, b[0], b[1],
                            b[2], false))) <= 1.0e-10f;
                    bool smallConvex = pmBufferPresent(aMesh.solid_planes) &&
                        aMesh.index_count <= 96u;
                    Vec3 normal = guided
                        ? rigid_guided_triangle_normal(a[0], a[1], a[2],
                            b[0], b[1], b[2], pointA, pointB,
                            haveSeparatingNormal ? separatingNormal : fallback)
                        : (distance <= 1.0e-5f || (smallConvex && onFace))
                            ? fallback : normalized_or(delta, fallback);
                    if (guided && (dot(normal, outwardA) > 1.0e-3f ||
                                   dot(normal, outwardB) < -1.0e-3f)) break;
                    Vec3 point = multiply(add(pointA, pointB), 0.5f);
                    float normalSpeed = rigid_contact_normal_speed(
                        aState, bState, point, normal);
                    if (normalSpeed > 1.0e-5f) break;
                    float remaining = -normalSpeed * timestep *
                        (1.0f - time) - distance;
                    Contact contact = {};
                    contact.normal = normal;
                    contact.point = point;
                    contact.penetration = (guided
                        ? max(0.0f, remaining + contactOffset)
                        : max(0.0f, remaining) + restOffset) + 1.0e-5f;
                    contact.body_hinge = bodyHinge;
                    contact.collider_hinge = colliderHinge;
                    contact.impact_fraction = startsNearContact ? 0.0f : time;
                    contact.initial_normal_speed = normalSpeed;
                    rigid_add_contact(manifold, contact,
                        max(margin * 2.0f, 1.0e-4f));
                    break;
                }
                separatingNormal = rigid_guided_triangle_normal(
                    a[0], a[1], a[2], b[0], b[1], b[2], pointA, pointB,
                    normalized_or(delta, {0.0f, 1.0f, 0.0f}));
                haveSeparatingNormal = true;
                float advancement = (distance - contactOffset) /
                    (speedBound + kEpsilon) * 0.9f;
                if (guided)
                {
                    Vec3 plane = rigid_guided_triangle_normal(
                        a[0], a[1], a[2], b[0], b[1], b[2], pointA,
                        pointB, separatingNormal);
                    float minimumAPlane = 3.402823466e+38f;
                    float maximumBPlane = -3.402823466e+38f;
                    float minimumSpeedA = 3.402823466e+38f;
                    float maximumSpeedB = -3.402823466e+38f;
                    for (uint vertex = 0u; vertex < 3u; ++vertex)
                    {
                        minimumAPlane = min(minimumAPlane,
                            dot(subtract(a[vertex], pointB), plane));
                        maximumBPlane = max(maximumBPlane,
                            dot(subtract(b[vertex], pointB), plane));
                        minimumSpeedA = min(minimumSpeedA,
                            dot(deltaA[vertex], plane));
                        maximumSpeedB = max(maximumSpeedB,
                            dot(deltaB[vertex], plane));
                    }
                    float gap = minimumAPlane - maximumBPlane - contactOffset;
                    float closingSpeed = maximumSpeedB - minimumSpeedA;
                    if (gap > 1.0e-5f && closingSpeed <= 0.0f) break;
                    if (gap > 0.0f && closingSpeed > kEpsilon)
                        advancement = max(advancement,
                            0.9f * gap / closingSpeed);
                }
                time += max(advancement, 1.0e-5f);
                if (time > 1.0f) break;
            }
        }
    }
}

ContactManifold rigid_collide_meshes(
    BodyParameters body, RigidBodyState previousBodyState,
    RigidBodyState bodyState, Vec3 previousBodyReference,
    Vec3 bodyReference, HingeContactFrame bodyHinge,
    TriangleMeshResource bodyMesh, BodyParameters collider,
    RigidBodyState previousColliderState, RigidBodyState colliderState,
    TriangleMeshResource colliderMesh, HingeContactFrame colliderHinge,
    float timestep)
{
    ContactManifold manifold = {};
    float margin = body.collision_margin + collider.collision_margin;
    bool swept = rigid_guided_pair(bodyHinge, colliderHinge) ||
        rigid_requires_swept_pair_contact(previousBodyState, bodyState,
            bodyMesh, previousColliderState, colliderState,
            colliderMesh, margin);
    BoundsTransform previousBodyTransform = swept
        ? bounds_transform(previousBodyState) : BoundsTransform();
    BoundsTransform previousColliderTransform = swept
        ? bounds_transform(previousColliderState) : BoundsTransform();
    BoundsTransform bodyTransform = bounds_transform(bodyState);
    BoundsTransform colliderTransform = bounds_transform(colliderState);
    Vec3 bodyMinimum;
    Vec3 bodyMaximum;
    Vec3 colliderMinimum;
    Vec3 colliderMaximum;
    transformed_motion_bounds(bodyMesh.minimum, bodyMesh.maximum,
        previousBodyTransform, bodyTransform, swept, margin,
        bodyMinimum, bodyMaximum);
    transformed_motion_bounds(colliderMesh.minimum, colliderMesh.maximum,
        previousColliderTransform, colliderTransform, swept, 0.0f,
        colliderMinimum, colliderMaximum);
    if (!bounds_overlap(bodyMinimum, bodyMaximum,
        colliderMinimum, colliderMaximum)) return manifold;

    uint stackBody[256] = {};
    uint stackCollider[256] = {};
    uint stackSize = 1u;
    bool overflow = bodyMesh.bvh_node_count == 0u ||
                    colliderMesh.bvh_node_count == 0u;
    while (stackSize > 0u && !overflow)
    {
        --stackSize;
        uint bodyNodeIndex = stackBody[stackSize];
        uint colliderNodeIndex = stackCollider[stackSize];
        BvhNode bodyNode = bodyMesh.bvh_nodes[bodyNodeIndex];
        BvhNode colliderNode = colliderMesh.bvh_nodes[colliderNodeIndex];
        transformed_motion_bounds(bodyNode.minimum, bodyNode.maximum,
            previousBodyTransform, bodyTransform, swept, margin,
            bodyMinimum, bodyMaximum);
        transformed_motion_bounds(colliderNode.minimum,
            colliderNode.maximum, previousColliderTransform,
            colliderTransform, swept, 0.0f,
            colliderMinimum, colliderMaximum);
        if (!bounds_overlap(bodyMinimum, bodyMaximum,
            colliderMinimum, colliderMaximum)) continue;
        bool bodyLeaf = bodyNode.triangle_count != 0u;
        bool colliderLeaf = colliderNode.triangle_count != 0u;
        if (bodyLeaf && colliderLeaf)
        {
            rigid_collide_triangle_ranges(bodyMesh, previousBodyState,
                bodyState, bodyReference, bodyHinge,
                bodyNode.first_triangle, bodyNode.triangle_count,
                colliderMesh, previousColliderState, colliderState,
                colliderHinge, colliderNode.first_triangle,
                colliderNode.triangle_count, margin, timestep, manifold);
            if (swept)
                rigid_collide_triangle_ranges_swept(previousBodyState,
                    bodyState, previousBodyReference, bodyReference,
                    bodyHinge, bodyMesh, bodyNode.first_triangle,
                    bodyNode.triangle_count, previousColliderState,
                    colliderState, colliderMesh,
                    colliderNode.first_triangle,
                    colliderNode.triangle_count, colliderHinge,
                    margin, timestep, manifold);
            continue;
        }
        uint required = bodyLeaf || colliderLeaf ? 2u : 4u;
        if (stackSize + required > 256u)
        {
            overflow = true;
            break;
        }
        if (bodyLeaf)
        {
            stackBody[stackSize] = bodyNodeIndex;
            stackCollider[stackSize++] = colliderNode.right;
            stackBody[stackSize] = bodyNodeIndex;
            stackCollider[stackSize++] = colliderNode.left;
        }
        else if (colliderLeaf)
        {
            stackBody[stackSize] = bodyNode.right;
            stackCollider[stackSize++] = colliderNodeIndex;
            stackBody[stackSize] = bodyNode.left;
            stackCollider[stackSize++] = colliderNodeIndex;
        }
        else
        {
            stackBody[stackSize] = bodyNode.right;
            stackCollider[stackSize++] = colliderNode.right;
            stackBody[stackSize] = bodyNode.right;
            stackCollider[stackSize++] = colliderNode.left;
            stackBody[stackSize] = bodyNode.left;
            stackCollider[stackSize++] = colliderNode.right;
            stackBody[stackSize] = bodyNode.left;
            stackCollider[stackSize++] = colliderNode.left;
        }
    }
    if (overflow)
    {
        manifold = {};
        rigid_collide_triangle_ranges(bodyMesh, previousBodyState,
            bodyState, bodyReference, bodyHinge, 0u,
            bodyMesh.index_count / 3u, colliderMesh,
            previousColliderState, colliderState, colliderHinge, 0u,
            colliderMesh.index_count / 3u, margin, timestep, manifold);
        if (swept)
            rigid_collide_triangle_ranges_swept(previousBodyState,
                bodyState, previousBodyReference, bodyReference,
                bodyHinge, bodyMesh, 0u, bodyMesh.index_count / 3u,
                previousColliderState, colliderState, colliderMesh, 0u,
                colliderMesh.index_count / 3u, colliderHinge,
                margin, timestep, manifold);
    }
    return manifold;
}

// Keep the native candidate traversal width. Candidate order determines
// which eight contacts survive manifold reduction, so changing this width is
// a physics change rather than a launch-only tuning choice.
#if !PM_TARGET_D3D12 || defined(PM_DXIL_ENTRY_pm_generate_rigid_leaf_pairs)
groupshared uint rigidLeafOffsets[128];
groupshared uint rigidLeafCandidateCount;
// Match the native broad/narrow-phase handoff exactly. Reusing one transformed
// bound per leaf is both the dense-mesh fast path and part of deterministic
// candidate selection near overlap boundaries.
groupshared WorldAabb rigidLeafCachedBounds[1024];

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void pm_generate_rigid_leaf_pairs(
    uniform PMBuffer<BodyParameters> parameters,
    uniform PMBuffer<RigidBodyState> previous_states, uniform PMBuffer<RigidBodyState> states,
    uniform uint count, uniform PMBuffer<TriangleMeshResource> meshes,
    uniform uint mesh_capacity, uniform PMBuffer<uint> active_pairs,
    uniform PMBuffer<uint> active_pair_count, uniform PMBuffer<LeafPair> leaf_pairs,
    uniform PMBuffer<uint> leaf_pair_counts,
    uniform uint leaf_pair_cache_slot_capacity,
    uniform uint leaf_pairs_per_slot, uniform uint shared_leaf_capacity
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint lane = pmGroupThreadIndex().x;
    for (uint activeIndex = pmBlockIndex().x;
         activeIndex < *active_pair_count; activeIndex += pmGridDim().x)
    {
        uint pair = active_pairs[activeIndex];
        if (lane == 0u) leaf_pair_counts[pair] = 0u;
        pmGroupBarrier();
        if (activeIndex >= leaf_pair_cache_slot_capacity ||
            leaf_pairs_per_slot == 0u)
        {
            if (lane == 0u) leaf_pair_counts[pair] = 0xffffffffu;
            pmGroupBarrier();
            continue;
        }
        uint a = pair / count;
        uint b = pair % count;
        TriangleMeshId aId = parameters[a].mesh;
        TriangleMeshId bId = parameters[b].mesh;
        if (aId.index >= mesh_capacity || bId.index >= mesh_capacity)
        {
            pmGroupBarrier();
            continue;
        }
        TriangleMeshResource aMesh = meshes[aId.index];
        TriangleMeshResource bMesh = meshes[bId.index];
        if (aMesh.alive == 0u || aMesh.generation != aId.generation ||
            bMesh.alive == 0u || bMesh.generation != bId.generation)
        {
            pmGroupBarrier();
            continue;
        }
        float margin = parameters[a].collision_margin +
            parameters[b].collision_margin;
        BoundsTransform aNow = bounds_transform(states[a]);
        BoundsTransform bNow = bounds_transform(states[b]);
        bool swept = rigid_requires_swept_pair_contact(
            previous_states[a], states[a], aMesh,
            previous_states[b], states[b], bMesh, margin);
        BoundsTransform aPrevious = {};
        BoundsTransform bPrevious = {};
        if (swept)
        {
            aPrevious = bounds_transform(previous_states[a]);
            bPrevious = bounds_transform(previous_states[b]);
        }
        uint totalLeaves = aMesh.bvh_leaf_count + bMesh.bvh_leaf_count;
        bool cacheBounds = totalLeaves <= shared_leaf_capacity &&
            totalLeaves <= 1024u;
        if (cacheBounds)
        {
            for (uint leaf = lane; leaf < totalLeaves;
                 leaf += pmGroupSize().x)
            {
                bool body = leaf < aMesh.bvh_leaf_count;
                TriangleMeshResource mesh = body ? aMesh : bMesh;
                uint localLeaf = body ? leaf : leaf - aMesh.bvh_leaf_count;
                BvhNode node = mesh.bvh_nodes[mesh.bvh_leaves[localLeaf]];
                transformed_motion_bounds(node.minimum, node.maximum,
                    body ? aPrevious : bPrevious,
                    body ? aNow : bNow, swept, body ? margin : 0.0f,
                    rigidLeafCachedBounds[leaf].minimum,
                    rigidLeafCachedBounds[leaf].maximum);
            }
        }
        pmGroupBarrier();
        uint colliderLeaves = bMesh.bvh_leaf_count;
        uint bodyLeaf = colliderLeaves != 0u ? lane / colliderLeaves : 0u;
        uint colliderLeaf = colliderLeaves != 0u ? lane % colliderLeaves : 0u;
        uint bodyStride = colliderLeaves != 0u
            ? pmGroupSize().x / colliderLeaves : 0u;
        uint colliderStride = colliderLeaves != 0u
            ? pmGroupSize().x % colliderLeaves : 0u;
        uint localCount = 0u;
        while (bodyLeaf < aMesh.bvh_leaf_count && colliderLeaves != 0u)
        {
            BvhNode bodyNode =
                aMesh.bvh_nodes[aMesh.bvh_leaves[bodyLeaf]];
            BvhNode colliderNode =
                bMesh.bvh_nodes[bMesh.bvh_leaves[colliderLeaf]];
            Vec3 bodyMinimum;
            Vec3 bodyMaximum;
            Vec3 colliderMinimum;
            Vec3 colliderMaximum;
            if (cacheBounds)
            {
                bodyMinimum = rigidLeafCachedBounds[bodyLeaf].minimum;
                bodyMaximum = rigidLeafCachedBounds[bodyLeaf].maximum;
                colliderMinimum = rigidLeafCachedBounds[
                    aMesh.bvh_leaf_count + colliderLeaf].minimum;
                colliderMaximum = rigidLeafCachedBounds[
                    aMesh.bvh_leaf_count + colliderLeaf].maximum;
            }
            else
            {
                transformed_motion_bounds(bodyNode.minimum, bodyNode.maximum,
                    aPrevious, aNow, swept, margin,
                    bodyMinimum, bodyMaximum);
                transformed_motion_bounds(colliderNode.minimum,
                    colliderNode.maximum, bPrevious, bNow, swept, 0.0f,
                    colliderMinimum, colliderMaximum);
            }
            if (bounds_overlap(bodyMinimum, bodyMaximum,
                colliderMinimum, colliderMaximum)) ++localCount;
            bodyLeaf += bodyStride;
            colliderLeaf += colliderStride;
            if (colliderLeaf >= colliderLeaves)
            {
                colliderLeaf -= colliderLeaves;
                ++bodyLeaf;
            }
        }
        rigidLeafOffsets[lane] = localCount;
        pmGroupBarrier();
        if (lane == 0u)
        {
            uint prefix = 0u;
            for (uint worker = 0u; worker < pmGroupSize().x; ++worker)
            {
                uint workerCount = rigidLeafOffsets[worker];
                rigidLeafOffsets[worker] = prefix;
                prefix += workerCount;
            }
            rigidLeafCandidateCount = prefix;
            leaf_pair_counts[pair] = prefix > leaf_pairs_per_slot
                ? 0xffffffffu : prefix;
        }
        pmGroupBarrier();
        if (rigidLeafCandidateCount <= leaf_pairs_per_slot)
        {
            bodyLeaf = colliderLeaves != 0u ? lane / colliderLeaves : 0u;
            colliderLeaf = colliderLeaves != 0u
                ? lane % colliderLeaves : 0u;
            uint outputIndex = rigidLeafOffsets[lane];
            while (bodyLeaf < aMesh.bvh_leaf_count && colliderLeaves != 0u)
            {
                BvhNode bodyNode =
                    aMesh.bvh_nodes[aMesh.bvh_leaves[bodyLeaf]];
                BvhNode colliderNode =
                    bMesh.bvh_nodes[bMesh.bvh_leaves[colliderLeaf]];
                Vec3 bodyMinimum;
                Vec3 bodyMaximum;
                Vec3 colliderMinimum;
                Vec3 colliderMaximum;
                if (cacheBounds)
                {
                    bodyMinimum = rigidLeafCachedBounds[bodyLeaf].minimum;
                    bodyMaximum = rigidLeafCachedBounds[bodyLeaf].maximum;
                    colliderMinimum = rigidLeafCachedBounds[
                        aMesh.bvh_leaf_count + colliderLeaf].minimum;
                    colliderMaximum = rigidLeafCachedBounds[
                        aMesh.bvh_leaf_count + colliderLeaf].maximum;
                }
                else
                {
                    transformed_motion_bounds(bodyNode.minimum,
                        bodyNode.maximum, aPrevious, aNow, swept, margin,
                        bodyMinimum, bodyMaximum);
                    transformed_motion_bounds(colliderNode.minimum,
                        colliderNode.maximum, bPrevious, bNow, swept, 0.0f,
                        colliderMinimum, colliderMaximum);
                }
                if (bounds_overlap(bodyMinimum, bodyMaximum,
                    colliderMinimum, colliderMaximum))
                    leaf_pairs[activeIndex * leaf_pairs_per_slot +
                        outputIndex++] = {bodyNode.first_triangle,
                            bodyNode.triangle_count,
                            colliderNode.first_triangle,
                            colliderNode.triangle_count};
                bodyLeaf += bodyStride;
                colliderLeaf += colliderStride;
                if (colliderLeaf >= colliderLeaves)
                {
                    colliderLeaf -= colliderLeaves;
                    ++bodyLeaf;
                }
            }
        }
        pmGroupBarrier();
    }
}
#endif

#if !PM_TARGET_D3D12 || defined(PM_DXIL_ENTRY_pm_evaluate_rigid_leaf_pairs)
#if !PM_TARGET_D3D12
#if defined(PM_TARGET_VULKAN)
// Eight workers keep this contact reduction within a 32 KiB mobile workgroup
// budget. The adapter specializes the launch to the same number of workers.
groupshared ContactManifold rigidLeafWorkerManifolds[8];
#else
groupshared ContactManifold rigidLeafWorkerManifolds[16];
#endif
groupshared ContactManifold rigidLeafReducedManifold;
#endif
groupshared HingeContactFrame rigidLeafPreviousBodyHinge;
groupshared HingeContactFrame rigidLeafBodyHinge;
groupshared HingeContactFrame rigidLeafColliderHinge;
groupshared Vec3 rigidLeafPreviousBodyReference;
groupshared Vec3 rigidLeafBodyReference;
groupshared float rigidLeafMargin;
groupshared uint rigidLeafSwept;

[shader("compute")]
[numthreads(PM_THREADS(16), 1, 1)]
void pm_evaluate_rigid_leaf_pairs(
    uniform PMBuffer<BodyParameters> parameters,
    uniform PMBuffer<RigidBodyState> previous_states, uniform PMBuffer<RigidBodyState> states,
    uniform uint count, uniform PMBuffer<TriangleMeshResource> meshes,
    uniform PMBuffer<RigidBodyId> ids, uniform PMBuffer<RigidConstraintResource> constraints,
    uniform uint constraint_capacity, uniform PMBuffer<uint> active_pairs,
    uniform PMBuffer<uint> active_pair_count, uniform PMBuffer<LeafPair> leaf_pairs,
    uniform PMBuffer<uint> leaf_pair_counts, uniform uint leaf_pairs_per_slot,
    uniform PMBuffer<ContactManifold> leaf_manifolds, uniform uint blocks_per_pair,
    uniform float timestep, uniform PMBuffer<ContactManifold> manifolds
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint lane = pmGroupThreadIndex().x;
    uint pairBlock = pmBlockIndex().x % blocks_per_pair;
    for (uint activeIndex = pmBlockIndex().x / blocks_per_pair;
         activeIndex < *active_pair_count;
         activeIndex += pmGridDim().x / blocks_per_pair)
    {
        uint pair = active_pairs[activeIndex];
        uint a = pair / count, b = pair % count;
        uint candidateCount = leaf_pair_counts[pair];
        if (candidateCount == 0xfffffffeu) continue;
        if (candidateCount == 0u || candidateCount == 0xffffffffu)
        {
            if (lane == 0u && pairBlock == 0u)
            {
                manifolds[activeIndex].count = 0u;
                manifolds[activeIndex].face_patch = false;
            }
            pmGroupBarrier();
            continue;
        }

        if (lane == 0u)
        {
            Vec3 colliderReference;
            rigidLeafPreviousBodyHinge = rigid_hinge_contact_frame(
                parameters, ids, count, a, previous_states,
                previous_states[a], constraints, constraint_capacity,
                rigidLeafPreviousBodyReference);
            rigidLeafBodyHinge = rigid_hinge_contact_frame(
                parameters, ids, count, a, states, states[a], constraints,
                constraint_capacity, rigidLeafBodyReference);
            rigidLeafColliderHinge = rigid_hinge_contact_frame(
                parameters, ids, count, b, states, states[b], constraints,
                constraint_capacity, colliderReference);
            rigidLeafMargin = parameters[a].collision_margin +
                parameters[b].collision_margin;
            rigidLeafSwept = (rigid_guided_pair(
                    rigidLeafBodyHinge, rigidLeafColliderHinge) ||
                rigid_requires_swept_pair_contact(
                    previous_states[a], states[a],
                    meshes[parameters[a].mesh.index], previous_states[b],
                    states[b], meshes[parameters[b].mesh.index],
                    rigidLeafMargin)) ? 1u : 0u;
#if !PM_TARGET_D3D12
            rigidLeafReducedManifold.count = 0u;
            rigidLeafReducedManifold.face_patch = false;
#endif
        }
        pmGroupBarrier();
#if PM_TARGET_D3D12
        // Large worlds avoid the multi-gigabyte per-leaf manifold cache. CUDA
        // reduces those candidates in group shared memory; DXIL evaluates the
        // same candidate order on lane zero because a full ContactManifold
        // array exceeds the D3D12 thread-group shared-memory budget.
        if (!pmBufferPresent(leaf_manifolds))
        {
            if (lane == 0u)
            {
                ContactManifold reduced = {};
                float spacing = max(rigidLeafMargin * 2.0f, 1.0e-4f);
                for (uint candidate = 0u; candidate < candidateCount; ++candidate)
                {
                    ContactManifold local = {};
                    LeafPair range = leaf_pairs[
                        activeIndex * leaf_pairs_per_slot + candidate];
                    rigid_collide_triangle_ranges(
                        meshes[parameters[a].mesh.index],
                        previous_states[a], states[a],
                        rigidLeafBodyReference, rigidLeafBodyHinge,
                        range.body_first, range.body_count,
                        meshes[parameters[b].mesh.index],
                        previous_states[b], states[b],
                        rigidLeafColliderHinge, range.collider_first,
                        range.collider_count, rigidLeafMargin, timestep,
                        local);
                    if (rigidLeafSwept != 0u)
                        rigid_collide_triangle_ranges_swept(
                            previous_states[a], states[a],
                            rigidLeafPreviousBodyReference,
                            rigidLeafBodyReference, rigidLeafBodyHinge,
                            meshes[parameters[a].mesh.index],
                            range.body_first, range.body_count,
                            previous_states[b], states[b],
                            meshes[parameters[b].mesh.index],
                            range.collider_first, range.collider_count,
                            rigidLeafColliderHinge, rigidLeafMargin,
                            timestep, local);
                    for (uint point = 0u; point < local.count; ++point)
                        rigid_add_contact(reduced, local.contacts[point], spacing);
                }
                manifolds[activeIndex] = reduced;
            }
            pmGroupBarrier();
            continue;
        }
#endif
        for (uint wave = pairBlock * pmGroupSize().x;
             wave < candidateCount;
             wave += pmGroupSize().x * blocks_per_pair)
        {
            uint candidate = wave + lane;
            ContactManifold local = {};
            if (candidate < candidateCount)
            {
                LeafPair range = leaf_pairs[
                    activeIndex * leaf_pairs_per_slot + candidate];
                rigid_collide_triangle_ranges(
                    meshes[parameters[a].mesh.index],
                    previous_states[a], states[a],
                    rigidLeafBodyReference, rigidLeafBodyHinge,
                    range.body_first, range.body_count,
                    meshes[parameters[b].mesh.index],
                    previous_states[b], states[b],
                    rigidLeafColliderHinge, range.collider_first,
                    range.collider_count, rigidLeafMargin, timestep,
                    local);
                if (rigidLeafSwept != 0u)
                    rigid_collide_triangle_ranges_swept(
                        previous_states[a], states[a],
                        rigidLeafPreviousBodyReference,
                        rigidLeafBodyReference, rigidLeafBodyHinge,
                        meshes[parameters[a].mesh.index],
                        range.body_first, range.body_count,
                        previous_states[b], states[b],
                        meshes[parameters[b].mesh.index],
                        range.collider_first, range.collider_count,
                        rigidLeafColliderHinge, rigidLeafMargin,
                        timestep, local);
#if PM_TARGET_D3D12
                leaf_manifolds[
                    activeIndex * leaf_pairs_per_slot + candidate] = local;
#else
                rigidLeafWorkerManifolds[lane] = local;
                if (pmBufferPresent(leaf_manifolds))
                    leaf_manifolds[
                        activeIndex * leaf_pairs_per_slot + candidate] = local;
#endif
            }
            pmGroupBarrier();
#if !PM_TARGET_D3D12
            if (!pmBufferPresent(leaf_manifolds) && lane == 0u)
            {
                float spacing = max(rigidLeafMargin * 2.0f, 1.0e-4f);
                uint waveCount = min(pmGroupSize().x,
                    candidateCount - wave);
                for (uint worker = 0u; worker < waveCount; ++worker)
                {
                    ContactManifold partial = rigidLeafWorkerManifolds[worker];
                    for (uint point = 0u; point < partial.count; ++point)
                        rigid_add_contact(rigidLeafReducedManifold,
                            partial.contacts[point], spacing);
                }
            }
#endif
            pmGroupBarrier();
        }
#if !PM_TARGET_D3D12
        if (!pmBufferPresent(leaf_manifolds) && lane == 0u)
            manifolds[activeIndex] = rigidLeafReducedManifold;
#endif
        pmGroupBarrier();
    }
}
#endif

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_reduce_rigid_leaf_manifolds(
    uniform PMBuffer<BodyParameters> parameters, uniform uint count,
    uniform PMBuffer<uint> active_pairs, uniform PMBuffer<uint> active_pair_count,
    uniform PMBuffer<uint> leaf_pair_counts, uniform uint leaf_pairs_per_slot,
    uniform PMBuffer<ContactManifold> leaf_manifolds,
    uniform PMBuffer<ContactManifold> manifolds PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint activeIndex = pmThreadIndex();
    if (activeIndex >= *active_pair_count) return;
    uint pair = active_pairs[activeIndex];
    uint candidates = leaf_pair_counts[pair];
    if (candidates == 0xffffffffu || candidates == 0xfffffffeu) return;
    float spacing = max((parameters[pair / count].collision_margin +
        parameters[pair % count].collision_margin) * 2.0f, 1.0e-4f);
    ContactManifold result = {};
    for (uint candidate = 0u; candidate < candidates; ++candidate)
    {
        ContactManifold local = leaf_manifolds[
            activeIndex * leaf_pairs_per_slot + candidate];
        for (uint contact = 0u; contact < local.count; ++contact)
            rigid_add_contact(result, local.contacts[contact], spacing);
    }
    manifolds[activeIndex] = result;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_finalize_rigid_contact_manifolds(
    uniform PMBuffer<BodyParameters> parameters,
    uniform PMBuffer<RigidBodyState> previous_states, uniform PMBuffer<RigidBodyState> states,
    uniform uint count, uniform PMBuffer<TriangleMeshResource> meshes,
    uniform PMBuffer<RigidBodyId> ids, uniform PMBuffer<RigidConstraintResource> constraints,
    uniform uint constraint_capacity, uniform PMBuffer<uint> active_pairs,
    uniform PMBuffer<uint> active_pair_count, uniform PMBuffer<uint> leaf_pair_counts,
    uniform float timestep, uniform PMBuffer<ContactManifold> manifolds,
    uniform bool prepare_faces PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint activeIndex = pmThreadIndex();
    if (activeIndex >= *active_pair_count) return;
    uint pair = active_pairs[activeIndex];
    uint a = pair / count;
    uint b = pair % count;
    TriangleMeshResource bodyMesh = meshes[parameters[a].mesh.index];
    TriangleMeshResource colliderMesh = meshes[parameters[b].mesh.index];
    float margin = parameters[a].collision_margin +
                   parameters[b].collision_margin;
    if (prepare_faces)
    {
        bool facePair = pmBufferPresent(bodyMesh.solid_planes) &&
            pmBufferPresent(colliderMesh.solid_planes) &&
            bodyMesh.index_count <= 96u && colliderMesh.index_count <= 96u &&
            !rigid_requires_swept_pair_contact(previous_states[a], states[a],
                bodyMesh, previous_states[b], states[b], colliderMesh, margin);
        if (!facePair) return;
        Vec3 bodyReference;
        Vec3 colliderReference;
        HingeContactFrame bodyHinge = rigid_hinge_contact_frame(parameters,
            ids, count, a, states, states[a], constraints,
            constraint_capacity, bodyReference);
        HingeContactFrame colliderHinge = rigid_hinge_contact_frame(parameters,
            ids, count, b, states, states[b], constraints,
            constraint_capacity, colliderReference);
        if (rigid_guided_pair(bodyHinge, colliderHinge)) return;
        ContactManifold faceManifold;
        if (rigid_convex_face_manifold(states[a], bodyMesh, bodyHinge,
            states[b], colliderMesh, colliderHinge, margin, faceManifold))
        {
            manifolds[activeIndex] = faceManifold;
            leaf_pair_counts[pair] = 0xfffffffeu;
        }
        return;
    }
    bool persistent = pmBufferPresent(bodyMesh.solid_planes) &&
        bodyMesh.index_count <= 96u &&
        (parameters[b].motion != MotionType.dynamic ||
         (pmBufferPresent(colliderMesh.solid_planes) &&
          colliderMesh.index_count <= 96u));
    ContactManifold manifold = manifolds[activeIndex];
    if (leaf_pair_counts[pair] == 0xffffffffu)
    {
        Vec3 previousBodyReference;
        Vec3 bodyReference;
        Vec3 colliderReference;
        HingeContactFrame previousBodyHinge = rigid_hinge_contact_frame(
            parameters, ids, count, a, previous_states,
            previous_states[a], constraints, constraint_capacity,
            previousBodyReference);
        HingeContactFrame bodyHinge = rigid_hinge_contact_frame(
            parameters, ids, count, a, states, states[a], constraints,
            constraint_capacity, bodyReference);
        HingeContactFrame colliderHinge = rigid_hinge_contact_frame(
            parameters, ids, count, b, states, states[b], constraints,
            constraint_capacity, colliderReference);
        manifold = rigid_collide_meshes(parameters[a], previous_states[a],
            states[a], previousBodyReference, bodyReference, bodyHinge,
            bodyMesh, parameters[b], previous_states[b], states[b],
            colliderMesh, colliderHinge, timestep);
    }
    manifold.initial_relative_position = subtract(
        states[a].position, states[b].position);
    for (uint point = 0u; point < manifold.count; ++point)
    {
        manifold.contacts[point].persistent =
            rigid_guided_contact(manifold.contacts[point]) ||
            (persistent &&
             manifold.contacts[point].body_hinge.present == 0u &&
             manifold.contacts[point].collider_hinge.present == 0u &&
             manifold.contacts[point].body_hinge.fixed_member == 0u &&
             manifold.contacts[point].collider_hinge.fixed_member == 0u);
        manifold.contacts[point].initial_normal_speed =
            rigid_contact_normal_speed(states[a], states[b],
                manifold.contacts[point].point,
                manifold.contacts[point].normal);
        manifold.contacts[point].reported_normal_impulse =
            manifold.contacts[point].accumulated_normal_impulse;
        manifold.contacts[point].reported_friction_impulse =
            manifold.contacts[point].accumulated_friction_impulse;
    }
    manifolds[activeIndex] = manifold;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_clear_rigid_inputs(
    uniform PMBuffer<BodyAccumulator> accumulators, uniform PMBuffer<KinematicTarget> targets,
    uniform uint count PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index >= count) return;
    accumulators[index] = {};
    targets[index].active = false;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_capture_rigid_inputs(
    uniform PMBuffer<BodyAccumulator> accumulators, uniform PMBuffer<Vec3> forces,
    uniform PMBuffer<Vec3> torques, uniform uint count
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index >= count) return;
    forces[index] = accumulators[index].force;
    torques[index] = accumulators[index].torque;
}

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void pm_prepare_parallel_contact_events(
    uniform uint count, uniform PMBuffer<ContactManifold> manifolds,
    uniform PMBuffer<uint> active_pairs, uniform PMBuffer<uint> active_pair_count,
    uniform PMBuffer<RigidBodyId> ids, uniform PMBuffer<uint> event_offsets,
    uniform PMBuffer<RigidContactEvent> events, uniform uint event_capacity,
    uniform PMBuffer<uint> event_count, uniform bool collect_events,
    uniform bool reset_events PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (reset_events) event_count[0] = 0u;
    uint cursor = 0u;
    uint responseSlot = 0u;
    for (uint active = 0u; active < *active_pair_count; ++active)
    {
        uint manifoldCount = manifolds[active].count;
        manifolds[active].response_slot = manifoldCount != 0u
            ? responseSlot++ : kInvalidDense;
        if (!collect_events) continue;
        event_offsets[active] = cursor;
        uint pair = active_pairs[active];
        uint a = pair / count, b = pair % count;
        uint remaining = cursor < event_capacity
            ? event_capacity - cursor : 0u;
        uint retained = min(manifoldCount, remaining);
        for (uint point = 0u; point < retained; ++point)
        {
            events[cursor + point] = {ids[a], ids[b],
                manifolds[active].contacts[point].point,
                manifolds[active].contacts[point].normal,
                max(0.0f, manifolds[active].contacts[point].penetration),
                0.0f, {}};
        }
        cursor += manifoldCount;
    }
    if (cursor > 0u) event_count[0] = min(cursor, event_capacity);
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_load_rigid_contact_cache(
    uniform PMBuffer<ContactManifold> manifolds, uniform PMBuffer<uint> active_pairs,
    uniform PMBuffer<uint> active_count, uniform PMBuffer<RigidBodyState> states,
    uniform PMBuffer<RigidBodyId> ids, uniform uint body_count,
    uniform PMBuffer<CachedContactPair> cache, uniform PMBuffer<uint> slots,
    uniform uint capacity, uniform uint64_t epoch, uniform float timestep
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint active = pmThreadIndex();
    if (active >= *active_count) return;
    uint pair = active_pairs[active], slot = slots[pair];
    uint body = pair / body_count, collider = pair % body_count;
    if (slot >= capacity) return;
    CachedContactPair saved = cache[slot];
    if (saved.pair != pair || saved.epoch + 1ull != epoch ||
        !same_id(saved.body, ids[body]) ||
        !same_id(saved.collider, ids[collider]) ||
        abs(saved.timestep - timestep) > 1.0e-7f) return;
    ContactManifold manifold = manifolds[active];
    uint used = 0u;
    for (uint point = 0u; point < manifold.count; ++point)
    {
        Contact contact = manifold.contacts[point];
        Vec3 local = inverse_rotate(states[body].orientation,
            subtract(contact.point, states[body].position));
        float nearest = 0.02f * 0.02f;
        uint match = 8u;
        for (uint previous = 0u; previous < saved.count; ++previous)
        {
            if ((used & (1u << previous)) != 0u ||
                dot(contact.normal, saved.contacts[previous].normal) < 0.99f)
                continue;
            float distance = length_squared(subtract(
                local, saved.contacts[previous].local_point));
            if (distance < nearest)
            {
                nearest = distance;
                match = previous;
            }
        }
        if (match == 8u) continue;
        used |= 1u << match;
        CachedContact previous = saved.contacts[match];
        for (uint axis = 0u; axis < 3u; ++axis)
            contact.avbd_dual[axis] = previous.avbd_dual[axis];
        Vec3 tangent0 = normalized_or(cross(contact.normal,
            abs(contact.normal.x) < 0.57735f
                ? Vec3(1.0f, 0.0f, 0.0f)
                : Vec3(0.0f, 1.0f, 0.0f)),
            Vec3(0.0f, 0.0f, 1.0f));
        Vec3 tangent1 = cross(contact.normal, tangent0);
        contact.avbd_dual[1].lambda =
            dot(previous.friction_impulse, tangent0) / timestep;
        contact.avbd_dual[2].lambda =
            dot(previous.friction_impulse, tangent1) / timestep;
        contact.avbd_anchor_a = previous.avbd_anchor_a;
        contact.avbd_anchor_b = previous.avbd_anchor_b;
        contact.avbd_cached = true;
        contact.accumulated_normal_impulse = previous.normal_impulse;
        contact.accumulated_friction_impulse = subtract(
            previous.friction_impulse, multiply(contact.normal,
                dot(previous.friction_impulse, contact.normal)));
        manifold.contacts[point] = contact;
    }
    manifolds[active] = manifold;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_save_rigid_contact_cache(
    uniform PMBuffer<ContactManifold> manifolds, uniform PMBuffer<uint> active_pairs,
    uniform PMBuffer<uint> active_count, uniform PMBuffer<RigidBodyState> states,
    uniform PMBuffer<RigidBodyId> ids, uniform uint body_count,
    uniform PMBuffer<CachedContactPair> cache, uniform PMBuffer<uint> slots,
    uniform uint capacity, uniform uint64_t epoch, uniform float timestep,
    uniform PMBuffer<uint> event_offsets, uniform PMBuffer<RigidContactEvent> events,
    uniform uint event_capacity PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint active = pmThreadIndex();
    if (active >= *active_count) return;
    uint pair = active_pairs[active];
    uint body = pair / body_count, collider = pair % body_count;
    ContactManifold manifold = manifolds[active];
    if (event_capacity > 0u)
    {
        uint offset = event_offsets[active];
        for (uint point = 0u; point < manifold.count &&
             offset < event_capacity && point < event_capacity - offset;
             ++point)
        {
            Contact contact = manifold.contacts[point];
            events[offset + point].normal_impulse = contact.persistent != 0u
                ? contact.accumulated_normal_impulse
                : contact.reported_normal_impulse;
            events[offset + point].friction_impulse = contact.persistent != 0u
                ? contact.accumulated_friction_impulse
                : contact.reported_friction_impulse;
        }
    }
    if (manifold.response_slot >= capacity) return;
    CachedContactPair saved = {};
    saved.pair = pair;
    saved.body = ids[body];
    saved.collider = ids[collider];
    saved.epoch = epoch;
    saved.timestep = timestep;
    saved.count = manifold.count;
    for (uint point = 0u; point < saved.count; ++point)
    {
        Contact contact = manifold.contacts[point];
        saved.contacts[point].local_point = inverse_rotate(
            states[body].orientation,
            subtract(contact.point, states[body].position));
        saved.contacts[point].normal = contact.normal;
        saved.contacts[point].normal_impulse =
            contact.accumulated_normal_impulse;
        saved.contacts[point].friction_impulse =
            contact.accumulated_friction_impulse;
        for (uint axis = 0u; axis < 3u; ++axis)
            saved.contacts[point].avbd_dual[axis] = contact.avbd_dual[axis];
        saved.contacts[point].avbd_anchor_a = contact.avbd_anchor_a;
        saved.contacts[point].avbd_anchor_b = contact.avbd_anchor_b;
    }
    cache[manifold.response_slot] = saved;
    slots[pair] = manifold.response_slot;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_gather_rigid_debug_samples(
    uniform PMBuffer<RigidBodyId> ids, uniform PMBuffer<RigidBodyState> states,
    uniform PMBuffer<Vec3> forces, uniform PMBuffer<Vec3> torques, uniform uint count,
    uniform PMBuffer<PhysicsDebugRigidSample> samples
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index < count)
        samples[index] = {ids[index], states[index], forces[index], torques[index]};
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void pm_clamp_rigid_speeds(
    uniform PMBuffer<BodyParameters> parameters, uniform PMBuffer<RigidBodyState> states,
    uniform uint count PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index >= count || parameters[index].motion != MotionType.dynamic)
        return;
    states[index].linear_velocity = clamp_length(
        states[index].linear_velocity, parameters[index].maximum_linear_speed);
    states[index].angular_velocity = clamp_length(
        states[index].angular_velocity, parameters[index].maximum_angular_speed);
}

uint fixed_projection_root(PMBuffer<FixedContactProjection> groups, uint body)
{
    while (groups[body].root != body) body = groups[body].root;
    return body;
}

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void pm_build_fixed_collision_groups(
    uniform PMBuffer<RigidConstraintResource> constraints, uniform uint capacity,
    uniform PMBuffer<RigidBodyId> ids, uniform uint count,
    uniform PMBuffer<FixedContactProjection> groups
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    for (uint body = 0u; body < count; ++body) groups[body].root = body;
    for (uint index = 0u; index < capacity; ++index)
    {
        RigidConstraintResource constraint = constraints[index];
        if (constraint.alive == 0u || constraint.options.enabled == 0u ||
            constraint.state.broken != 0u ||
            constraint.options.disable_collisions == 0u ||
            constraint.options.type != RigidConstraintType.fixed) continue;
        uint a = find_rigid_body_dense(
            constraint.options.body_a, ids, count);
        uint b = find_rigid_body_dense(
            constraint.options.body_b, ids, count);
        if (a == kInvalidDense || b == kInvalidDense) continue;
        a = fixed_projection_root(groups, a);
        b = fixed_projection_root(groups, b);
        if (a != b) groups[max(a, b)].root = min(a, b);
    }
    for (uint body = 0u; body < count; ++body)
        groups[body].root = fixed_projection_root(groups, body);
}
