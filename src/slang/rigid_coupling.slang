#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Batched rigid/deformable and rigid/smoke exchange for realtime-single-step.

import types;
import math;
import device_intrinsics;
import graphics_resources;
import geometry;

static const uint kRigidCouplingProfileVersion = 1u;
static const float kRigidCouplingTimestep = 1.0f / 60.0f;
static const uint kRigidCouplingSweeps = 4u;
static const uint kRigidCouplingCloth = 1u;
static const uint kRigidCouplingSoftBody = 2u;
static const uint kRigidCouplingRope = 3u;
static const uint kRigidCouplingSmokeParticle = 4u;
static const uint kRigidCouplingSmokeGrid = 5u;
static const uint kRigidCouplingContact = 1u;
static const uint kRigidCouplingAttachment = 2u;
static const uint kRigidCouplingDrag = 3u;
static const uint kRigidCouplingInvalidProfile = 1u << 17u;
static const uint kRigidCouplingInvalidTimestep = 1u << 18u;
static const uint kRigidCouplingInvalidMaterial = 1u << 19u;
static const uint kRigidCouplingNonfiniteState = 1u << 20u;
static const uint kRigidCouplingCorrectionExceeded = 1u << 21u;

bool rigid_coupling_finite_vec3(Vec3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

bool rigid_coupling_profile_valid(uint profile, float dt, uint sweep)
{
    return profile == kRigidCouplingProfileVersion && isfinite(dt) &&
        abs(dt - kRigidCouplingTimestep) <= 1.0e-7f &&
        sweep < kRigidCouplingSweeps;
}

bool rigid_coupling_mode_valid(uint family, uint mode)
{
    if (family == kRigidCouplingCloth ||
        family == kRigidCouplingSoftBody)
        return mode == kRigidCouplingContact;
    if (family == kRigidCouplingRope)
        return mode == kRigidCouplingContact ||
            mode == kRigidCouplingAttachment;
    if (family == kRigidCouplingSmokeParticle ||
        family == kRigidCouplingSmokeGrid)
        return mode == kRigidCouplingContact || mode == kRigidCouplingDrag;
    return false;
}

void rigid_coupling_record_profile_invalid(
    uint profile, float dt, uint sweep, PMBuffer<uint> status)
{
    if (profile != kRigidCouplingProfileVersion ||
        sweep >= kRigidCouplingSweeps)
        pmAtomicOr(status, kRigidCouplingInvalidProfile);
    if (!isfinite(dt) || abs(dt - kRigidCouplingTimestep) > 1.0e-7f)
        pmAtomicOr(status, kRigidCouplingInvalidTimestep);
}

// Proposal phase. Constraint records are generated once from the F8-10 epoch
// and reused for all four sweeps. Generation mismatch means a retired edge and
// produces a zero proposal; it never retargets a replacement resource.
[shader("compute")]
[numthreads(128, 1, 1)]
void rigid_coupling_project(
    uniform PMBuffer<Vec3> endpoint_positions,
    uniform PMBuffer<Vec3> endpoint_previous_positions,
    uniform PMBuffer<Vec3> endpoint_velocities,
    uniform PMBuffer<float> endpoint_inverse_masses,
    uniform PMBuffer<uint> endpoint_generations, uniform uint endpoint_count,
    uniform PMBuffer<BodyParameters> body_parameters,
    uniform PMBuffer<RigidBodyState> body_previous_states,
    uniform PMBuffer<RigidBodyState> body_states,
    uniform PMBuffer<uint> body_generations, uniform uint body_count,
    uniform PMBuffer<uint> constraint_endpoints,
    uniform PMBuffer<uint> constraint_bodies,
    uniform PMBuffer<uint> constraint_endpoint_generations,
    uniform PMBuffer<uint> constraint_body_generations,
    uniform PMBuffer<uint> constraint_families,
    uniform PMBuffer<uint> constraint_modes,
    uniform PMBuffer<Vec3> constraint_normals,
    uniform PMBuffer<Vec3> constraint_body_local_points,
    uniform PMBuffer<float> constraint_contact_distances,
    uniform PMBuffer<float> constraint_frictions,
    uniform PMBuffer<float> constraint_responses,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform float characteristic_length, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version, uniform uint sweep,
    uniform PMBuffer<Vec3> endpoint_position_deltas,
    uniform PMBuffer<Vec3> body_position_deltas,
    uniform PMBuffer<Vec3> endpoint_impulses,
    uniform PMBuffer<Vec3> body_angular_impulses,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint constraint = pmThreadIndex();
    if (!rigid_coupling_profile_valid(profile_version, dt, sweep))
    {
        if (constraint == 0u)
            rigid_coupling_record_profile_invalid(
                profile_version, dt, sweep, status);
        return;
    }
    if (constraint >= constraint_count) return;
    endpoint_position_deltas[constraint] = {};
    body_position_deltas[constraint] = {};
    endpoint_impulses[constraint] = {};
    body_angular_impulses[constraint] = {};
    if (constraint_active[constraint] == 0u) return;

    uint endpoint = constraint_endpoints[constraint];
    uint body = constraint_bodies[constraint];
    if (endpoint >= endpoint_count || body >= body_count)
    {
        pmAtomicOr(status, kRigidCouplingInvalidMaterial);
        return;
    }
    // Generation records are constraint-aligned so endpoints from one
    // resource can be dispatched as a batch without a per-node generation
    // allocation.
    if (endpoint_generations[constraint] !=
            constraint_endpoint_generations[constraint] ||
        body_generations[body] != constraint_body_generations[constraint])
        return;

    uint family = constraint_families[constraint];
    uint mode = constraint_modes[constraint];
    float distance = constraint_contact_distances[constraint];
    float friction = constraint_frictions[constraint];
    float response = constraint_responses[constraint];
    BodyParameters parameters = body_parameters[body];
    RigidBodyState bodyState = body_states[body];
    RigidBodyState previousBodyState = body_previous_states[body];
    Vec3 endpointPosition = endpoint_positions[endpoint];
    Vec3 previousEndpointPosition = endpoint_previous_positions[endpoint];
    Vec3 endpointVelocity = endpoint_velocities[endpoint];
    float endpointInverseMass = endpoint_inverse_masses[endpoint];
    float bodyInverseMass = parameters.motion == MotionType.dynamic
        ? parameters.inverse_mass : 0.0f;
    // A rope attachment reacts through the complete chain, not one node.
    // The endpoint buffer stores per-node inverse mass, so convert it to the
    // chain's translational inverse mass for the attachment proposal.
    if (family == kRigidCouplingRope &&
        mode == kRigidCouplingAttachment && endpoint_count != 0u)
        endpointInverseMass /= float(endpoint_count);
    Vec3 localPoint = constraint_body_local_points[constraint];
    Vec3 bodyPoint = transform_point(bodyState, localPoint);
    Vec3 previousBodyPoint = transform_point(previousBodyState, localPoint);
    Vec3 arm = subtract(bodyPoint, bodyState.position);
    Vec3 bodyPointVelocity = add(bodyState.linear_velocity,
        cross(bodyState.angular_velocity, arm));
    Vec3 relativeVelocity = subtract(endpointVelocity, bodyPointVelocity);
    float massRatio = endpointInverseMass > 0.0f && bodyInverseMass > 0.0f
        ? max(endpointInverseMass, bodyInverseMass) /
            min(endpointInverseMass, bodyInverseMass) : 1.0f;
    bool endpointMassValid = endpointInverseMass == 0.0f ||
        (endpointInverseMass >= 0.0001f && endpointInverseMass <= 1000.0f);
    bool bodyMassValid = bodyInverseMass == 0.0f ||
        (bodyInverseMass >= 0.0001f && bodyInverseMass <= 1000.0f);
    bool baseValid = rigid_coupling_mode_valid(family, mode) &&
        isfinite(characteristic_length) && characteristic_length >= 0.005f &&
        characteristic_length <= 2.0f && isfinite(maximum_speed) &&
        maximum_speed >= 0.0f && maximum_speed <= 20.0f &&
        maximum_speed * dt <= 0.5f * characteristic_length &&
        isfinite(endpointInverseMass) && endpointInverseMass >= 0.0f &&
        isfinite(bodyInverseMass) && bodyInverseMass >= 0.0f &&
        endpointMassValid && bodyMassValid &&
        endpointInverseMass + bodyInverseMass > 0.0f &&
        massRatio <= 100.0f && rigid_coupling_finite_vec3(localPoint) &&
        rigid_coupling_finite_vec3(endpointPosition) &&
        rigid_coupling_finite_vec3(previousEndpointPosition) &&
        rigid_coupling_finite_vec3(endpointVelocity) &&
        rigid_coupling_finite_vec3(bodyState.position) &&
        rigid_coupling_finite_vec3(bodyState.linear_velocity) &&
        rigid_coupling_finite_vec3(bodyState.angular_velocity) &&
        rigid_coupling_finite_vec3(bodyPoint) &&
        rigid_coupling_finite_vec3(previousBodyPoint) &&
        rigid_coupling_finite_vec3(relativeVelocity) &&
        abs(endpointPosition.x) <= 10000.0f &&
        abs(endpointPosition.y) <= 10000.0f &&
        abs(endpointPosition.z) <= 10000.0f;
    if (!baseValid)
    {
        pmAtomicOr(status, kRigidCouplingInvalidMaterial);
        return;
    }

    Vec3 endpointCorrection = {};
    Vec3 endpointImpulse = {};
    if (mode == kRigidCouplingContact)
    {
        Vec3 normal = constraint_normals[constraint];
        float normalLength = vector_length(normal);
        bool valid = rigid_coupling_finite_vec3(normal) &&
            normalLength > 1.0e-6f && isfinite(distance) &&
            distance > 0.0f && distance <= characteristic_length &&
            isfinite(friction) && friction >= 0.0f && friction <= 2.0f &&
            response == 0.0f;
        if (!valid)
        {
            pmAtomicOr(status, kRigidCouplingInvalidMaterial);
            return;
        }
        normal = multiply(normal, 1.0f / normalLength);
        float separation = dot(subtract(endpointPosition, bodyPoint), normal);
        float previousSeparation = dot(subtract(
            previousEndpointPosition, previousBodyPoint), normal);
        if (!isfinite(separation) || !isfinite(previousSeparation))
        {
            pmAtomicOr(status, kRigidCouplingNonfiniteState);
            return;
        }
        // Current penetration resolves both overlap and a thin swept crossing;
        // previous separation is retained in the testable equation path.
        float penetration = max(0.0f, distance - separation);
        if (previousSeparation >= distance && separation < distance)
            penetration = max(penetration, distance - separation);
        float linearInverseMass = endpointInverseMass + bodyInverseMass;
        float correction = min(penetration,
            0.25f * characteristic_length);
        endpointCorrection = multiply(normal,
            correction * endpointInverseMass / linearInverseMass);
        body_position_deltas[constraint] = multiply(normal,
            -correction * bodyInverseMass / linearInverseMass);

        Vec3 angularAxis = cross(arm, normal);
        float effectiveInverseMass = linearInverseMass + dot(cross(
            inverse_inertia_world(parameters, bodyState, angularAxis), arm),
            normal);
        float closingSpeed = dot(relativeVelocity, normal);
        float normalMagnitude = closingSpeed < 0.0f &&
            effectiveInverseMass > kEpsilon
            ? -closingSpeed / effectiveInverseMass : 0.0f;
        endpointImpulse = multiply(normal, normalMagnitude);
        Vec3 tangent = subtract(relativeVelocity,
            multiply(normal, closingSpeed));
        float tangentSpeed = vector_length(tangent);
        if (tangentSpeed > 1.0e-6f && friction > 0.0f &&
            linearInverseMass > kEpsilon)
        {
            Vec3 tangentImpulse = multiply(tangent,
                -min(tangentSpeed / linearInverseMass,
                    friction * normalMagnitude) / tangentSpeed);
            endpointImpulse = add(endpointImpulse, tangentImpulse);
        }
    }
    else
    {
        bool valid = distance == 0.0f && friction == 0.0f &&
            isfinite(response) && response >= 0.05f && response <= 0.995f;
        if (!valid)
        {
            pmAtomicOr(status, kRigidCouplingInvalidMaterial);
            return;
        }
        float linearInverseMass = endpointInverseMass + bodyInverseMass;
        if (mode == kRigidCouplingAttachment)
        {
            Vec3 error = subtract(bodyPoint, endpointPosition);
            Vec3 correction = clamp_length(multiply(error, response),
                0.25f * characteristic_length);
            if (family == kRigidCouplingRope)
            {
                // A point-only endpoint delta creates a local segment kink
                // after the rope's domain sweeps are complete. The attachment
                // sync projects the static correction across the chain; only
                // the dynamic rigid side receives a positional proposal here.
                endpointCorrection = {};
                body_position_deltas[constraint] = bodyInverseMass > 0.0f
                    ? multiply(correction, -1.0f) : Vec3(0.0f);
            }
            else
            {
                endpointCorrection = multiply(correction,
                    endpointInverseMass / linearInverseMass);
                body_position_deltas[constraint] = multiply(correction,
                    -bodyInverseMass / linearInverseMass);
            }
        }
        endpointImpulse = multiply(relativeVelocity,
            -response / linearInverseMass);
    }
    if (!rigid_coupling_finite_vec3(endpointCorrection) ||
        !rigid_coupling_finite_vec3(endpointImpulse))
    {
        pmAtomicOr(status, kRigidCouplingNonfiniteState);
        endpoint_position_deltas[constraint] = {};
        body_position_deltas[constraint] = {};
        return;
    }
    endpoint_position_deltas[constraint] = endpointCorrection;
    endpoint_impulses[constraint] = endpointImpulse;
    body_angular_impulses[constraint] = cross(
        arm, multiply(endpointImpulse, -1.0f));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rigid_rope_sync_attachments(
    uniform PMBuffer<Vec3> rope_positions,
    uniform PMBuffer<Vec3> rope_previous_positions,
    uniform PMBuffer<Vec3> rope_velocities, uniform uint rope_count,
    uniform PMBuffer<float> rope_rest_lengths,
    uniform PMBuffer<Vec3> rope_anchor_targets,
    uniform PMBuffer<BodyParameters> body_parameters,
    uniform PMBuffer<RigidBodyState> body_states,
    uniform PMBuffer<uint> body_generations, uniform uint body_count,
    uniform PMBuffer<uint> constraint_endpoints,
    uniform PMBuffer<uint> constraint_bodies,
    uniform PMBuffer<uint> constraint_body_generations,
    uniform PMBuffer<Vec3> constraint_body_local_points,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform float dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    // One thread owns the chain-wide projection. A point-only rope snap after
    // the final domain sweep puts the complete attachment error into one
    // segment. Static anchors instead translate the solved chain without
    // changing segment lengths; dynamic bodies follow the solved endpoint.
    // The preceding reciprocal proposal still owns momentum exchange.
    if (pmThreadIndex() != 0u || rope_count == 0u || dt <= 0.0f) return;
    bool firstActive = false;
    bool lastActive = false;
    bool firstStatic = false;
    bool lastStatic = false;
    Vec3 firstCorrection = {};
    Vec3 lastCorrection = {};
    Vec3 firstVelocity = {};
    Vec3 lastVelocity = {};
    for (uint constraint = 0u; constraint < constraint_count; ++constraint)
    {
        if (constraint_active[constraint] == 0u) continue;
        uint endpoint = constraint_endpoints[constraint];
        uint body = constraint_bodies[constraint];
        if (endpoint >= rope_count || body >= body_count ||
            constraint_body_generations[constraint] != body_generations[body])
            continue;
        RigidBodyState state = body_states[body];
        Vec3 position = transform_point(
            state, constraint_body_local_points[constraint]);
        Vec3 arm = subtract(position, state.position);
        Vec3 velocity = add(state.linear_velocity,
            cross(state.angular_velocity, arm));
        Vec3 correction = subtract(position, rope_positions[endpoint]);
        if (endpoint == 0u)
        {
            firstActive = true;
            firstStatic = body_parameters[body].motion != MotionType.dynamic;
            firstCorrection = correction;
            firstVelocity = velocity;
            rope_anchor_targets[0u] = position;
        }
        else if (endpoint + 1u == rope_count)
        {
            lastActive = true;
            lastStatic = body_parameters[body].motion != MotionType.dynamic;
            lastCorrection = correction;
            lastVelocity = velocity;
            rope_anchor_targets[1u] = position;
        }
    }
    if (!firstActive && !lastActive) return;
    Vec3 chainFirstCorrection = {};
    Vec3 chainLastCorrection = {};
    if (firstStatic && lastStatic)
    {
        chainFirstCorrection = firstCorrection;
        chainLastCorrection = lastCorrection;
    }
    else if (firstStatic)
    {
        chainFirstCorrection = firstCorrection;
        chainLastCorrection = firstCorrection;
    }
    else if (lastStatic)
    {
        chainFirstCorrection = lastCorrection;
        chainLastCorrection = lastCorrection;
    }
    float denominator = rope_count > 1u ? float(rope_count - 1u) : 1.0f;
    for (uint node = 0u; node < rope_count; ++node)
    {
        float weight = float(node) / denominator;
        Vec3 correction = add(
            multiply(chainFirstCorrection, 1.0f - weight),
            multiply(chainLastCorrection, weight));
        Vec3 position = add(rope_positions[node], correction);
        rope_positions[node] = position;
        rope_previous_positions[node] = add(
            rope_previous_positions[node], correction);
    }
    if (firstStatic && !lastStatic)
    {
        for (uint node = 1u; node < rope_count; ++node)
        {
            Vec3 previous = rope_positions[node];
            Vec3 direction = normalized_or(subtract(previous,
                rope_positions[node - 1u]), Vec3(1.0f, 0.0f, 0.0f));
            Vec3 position = add(rope_positions[node - 1u],
                multiply(direction, rope_rest_lengths[node - 1u]));
            Vec3 correction = subtract(position, previous);
            rope_positions[node] = position;
            rope_previous_positions[node] = add(
                rope_previous_positions[node], correction);
        }
    }
    else if (lastStatic && !firstStatic)
    {
        for (uint node = rope_count - 1u; node > 0u; --node)
        {
            Vec3 previous = rope_positions[node - 1u];
            Vec3 direction = normalized_or(subtract(previous,
                rope_positions[node]), Vec3(-1.0f, 0.0f, 0.0f));
            Vec3 position = add(rope_positions[node],
                multiply(direction, rope_rest_lengths[node - 1u]));
            Vec3 correction = subtract(position, previous);
            rope_positions[node - 1u] = position;
            rope_previous_positions[node - 1u] = add(
                rope_previous_positions[node - 1u], correction);
        }
    }
    if (firstStatic)
    {
        rope_previous_positions[0u] = subtract(
            rope_positions[0u], multiply(firstVelocity, dt));
        rope_velocities[0u] = firstVelocity;
    }
    if (lastStatic)
    {
        uint endpoint = rope_count - 1u;
        rope_previous_positions[endpoint] = subtract(
            rope_positions[endpoint], multiply(lastVelocity, dt));
        rope_velocities[endpoint] = lastVelocity;
    }
    for (uint constraint = 0u; constraint < constraint_count; ++constraint)
    {
        if (constraint_active[constraint] == 0u) continue;
        uint endpoint = constraint_endpoints[constraint];
        uint body = constraint_bodies[constraint];
        if (endpoint >= rope_count || body >= body_count ||
            constraint_body_generations[constraint] != body_generations[body] ||
            body_parameters[body].motion != MotionType.dynamic)
            continue;
        RigidBodyState state = body_states[body];
        Vec3 bodyPoint = transform_point(
            state, constraint_body_local_points[constraint]);
        state.position = add(state.position,
            subtract(rope_positions[endpoint], bodyPoint));
        body_states[body] = state;
        Vec3 position = rope_positions[endpoint];
        Vec3 arm = subtract(position, state.position);
        Vec3 velocity = add(state.linear_velocity,
            cross(state.angular_velocity, arm));
        rope_previous_positions[endpoint] = subtract(
            position, multiply(velocity, dt));
        rope_velocities[endpoint] = velocity;
        rope_anchor_targets[endpoint == 0u ? 0u : 1u] = position;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rigid_rope_apply_attachment_reactions(
    uniform PMBuffer<Vec3> previous_rope_velocities,
    uniform PMBuffer<Vec3> current_rope_velocities,
    uniform PMBuffer<float> rope_inverse_masses, uniform uint rope_count,
    uniform PMBuffer<float> rope_rest_lengths,
    uniform float rope_maximum_speed,
    uniform uint attached_end_mask, uniform uint attached_end_count,
    uniform Vec3 gravity, uniform float dt,
    uniform PMBuffer<Vec3> rope_positions,
    uniform PMBuffer<Vec3> rope_velocities,
    uniform PMBuffer<Vec3> rope_constraint_forces,
    uniform PMBuffer<Vec3> rope_anchor_targets,
    uniform PMBuffer<BodyParameters> body_parameters,
    uniform PMBuffer<BodyAccumulator> body_accumulators,
    uniform PMBuffer<RigidBodyState> body_previous_states,
    uniform PMBuffer<RigidBodyState> body_states,
    uniform PMBuffer<uint> body_generations, uniform uint body_count,
    uniform PMBuffer<uint> constraint_endpoints,
    uniform PMBuffer<uint> constraint_bodies,
    uniform PMBuffer<uint> constraint_body_generations,
    uniform PMBuffer<Vec3> constraint_body_local_points,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint constraint = pmThreadIndex();
    if (constraint >= constraint_count ||
        constraint_active[constraint] == 0u || attached_end_count == 0u ||
        dt <= 0.0f) return;
    uint endpoint = constraint_endpoints[constraint];
    uint body = constraint_bodies[constraint];
    if (endpoint >= rope_count || body >= body_count ||
        constraint_body_generations[constraint] != body_generations[body])
        return;
    Vec3 currentImpulse = {};
    Vec3 previousMomentum = {};
    float freeMass = 0.0f;
    for (uint node = 0u; node < rope_count; ++node)
    {
        bool attached = (node == 0u && (attached_end_mask & 1u) != 0u) ||
            (node + 1u == rope_count && (attached_end_mask & 2u) != 0u);
        float inverseMass = rope_inverse_masses[node];
        if (attached || inverseMass <= 0.0f) continue;
        float mass = 1.0f / inverseMass;
        freeMass += mass;
        previousMomentum = add(previousMomentum,
            multiply(previous_rope_velocities[node], mass));
        currentImpulse = add(currentImpulse, subtract(
            multiply(subtract(current_rope_velocities[node],
                previous_rope_velocities[node]), mass),
            multiply(gravity, mass * dt)));
    }
    if (freeMass <= 0.0f) return;
    BodyParameters parameters = body_parameters[body];
    BodyAccumulator accumulator = body_accumulators[body];
    RigidBodyState previousState = body_previous_states[body];
    RigidBodyState state = body_states[body];
    Vec3 position = transform_point(
        state, constraint_body_local_points[constraint]);
    Vec3 arm = subtract(position, state.position);
    Vec3 bodyImpulse = {};
    if (parameters.motion == MotionType.dynamic)
    {
        Vec3 baseLinear = add(previousState.linear_velocity,
            multiply(add(gravity,
                multiply(accumulator.force, parameters.inverse_mass)), dt));
        baseLinear = add(baseLinear,
            multiply(accumulator.impulse, parameters.inverse_mass));
        baseLinear = multiply(baseLinear,
            1.0f / (1.0f + parameters.linear_damping * dt));
        baseLinear = clamp_length(
            baseLinear, parameters.maximum_linear_speed);
        Vec3 baseAngular = add(previousState.angular_velocity,
            multiply(inverse_inertia_world(parameters, previousState,
                accumulator.torque), dt));
        baseAngular = add(baseAngular,
            inverse_inertia_world(parameters, previousState,
                accumulator.angular_impulse));
        baseAngular = multiply(baseAngular,
            1.0f / (1.0f + parameters.angular_damping * dt));
        baseAngular = clamp_length(
            baseAngular, parameters.maximum_angular_speed);
        Vec3 baseAnchorVelocity = add(baseLinear,
            cross(baseAngular, arm));
        Vec3 inertialRopeVelocity = add(
            multiply(previousMomentum, 1.0f / freeMass),
            multiply(gravity, dt));
        Vec3 relative = subtract(
            inertialRopeVelocity, baseAnchorVelocity);
        float relativeSpeed = vector_length(relative);
        if (relativeSpeed > 1.0e-8f)
        {
            Vec3 direction = multiply(relative, 1.0f / relativeSpeed);
            Vec3 angularVelocity = inverse_inertia_world(
                parameters, state, cross(arm, direction));
            float effectiveInverseMass = parameters.inverse_mass +
                1.0f / freeMass +
                dot(cross(angularVelocity, arm), direction);
            if (effectiveInverseMass > 1.0e-8f)
                bodyImpulse = multiply(direction,
                    relativeSpeed / effectiveInverseMass);
        }
        bodyImpulse = multiply(bodyImpulse,
            1.0f / float(attached_end_count));
        Vec3 ropeAdjustment = subtract(
            multiply(bodyImpulse, -1.0f), currentImpulse);
        Vec3 ropeVelocityDelta = multiply(
            ropeAdjustment, 1.0f / freeMass);
        for (uint node = 0u; node < rope_count; ++node)
        {
            bool attached =
                (node == 0u && (attached_end_mask & 1u) != 0u) ||
                (node + 1u == rope_count &&
                    (attached_end_mask & 2u) != 0u);
            if (attached || rope_inverse_masses[node] <= 0.0f) continue;
            current_rope_velocities[node] = clamp_length(add(
                current_rope_velocities[node], ropeVelocityDelta),
                rope_maximum_speed);
            rope_positions[node] = add(rope_positions[node],
                multiply(ropeVelocityDelta, dt));
        }
        state.linear_velocity = clamp_length(add(baseLinear,
            multiply(bodyImpulse, parameters.inverse_mass)),
            parameters.maximum_linear_speed);
        state.angular_velocity = clamp_length(add(baseAngular,
            inverse_inertia_world(parameters, state,
                cross(arm, bodyImpulse))),
            parameters.maximum_angular_speed);
        body_states[body] = state;
    }
    Vec3 velocity = add(state.linear_velocity,
        cross(state.angular_velocity, arm));
    rope_positions[endpoint] = position;
    rope_velocities[endpoint] = velocity;
    rope_constraint_forces[endpoint] = multiply(bodyImpulse, 1.0f / dt);
    rope_anchor_targets[endpoint == 0u ? 0u : 1u] = position;
    // Final attachment projection preserves authored segment lengths without
    // feeding a point-only correction back into the next tick.
    if (attached_end_mask == 1u)
    {
        for (uint node = 1u; node < rope_count; ++node)
        {
            Vec3 direction = normalized_or(subtract(
                rope_positions[node], rope_positions[node - 1u]),
                Vec3(0.0f, -1.0f, 0.0f));
            rope_positions[node] = add(rope_positions[node - 1u],
                multiply(direction, rope_rest_lengths[node - 1u]));
        }
    }
    else if (attached_end_mask == 2u)
    {
        for (uint node = rope_count - 1u; node > 0u; --node)
        {
            Vec3 direction = normalized_or(subtract(
                rope_positions[node - 1u], rope_positions[node]),
                Vec3(0.0f, 1.0f, 0.0f));
            rope_positions[node - 1u] = add(rope_positions[node],
                multiply(direction, rope_rest_lengths[node - 1u]));
        }
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rigid_coupling_apply_endpoints(
    uniform PMBuffer<Vec3> endpoint_positions,
    uniform PMBuffer<Vec3> endpoint_velocities,
    uniform PMBuffer<float> endpoint_inverse_masses,
    uniform PMBuffer<float> endpoint_correction_used,
    uniform PMBuffer<uint> endpoint_generations, uniform uint endpoint_count,
    uniform PMBuffer<uint> constraint_endpoints,
    uniform PMBuffer<uint> constraint_endpoint_generations,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform PMBuffer<Vec3> endpoint_position_deltas,
    uniform PMBuffer<Vec3> endpoint_impulses,
    uniform float characteristic_length, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version, uniform uint sweep,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint endpoint = pmThreadIndex();
    if (!rigid_coupling_profile_valid(profile_version, dt, sweep))
    {
        if (endpoint == 0u)
            rigid_coupling_record_profile_invalid(
                profile_version, dt, sweep, status);
        return;
    }
    if (endpoint >= endpoint_count) return;
    float inverseMass = endpoint_inverse_masses[endpoint];
    if (inverseMass == 0.0f) return;
    Vec3 correction = {}, impulse = {};
    for (uint constraint = 0u; constraint < constraint_count; ++constraint)
    {
        if (constraint_active[constraint] == 0u ||
            constraint_endpoints[constraint] != endpoint ||
            constraint_endpoint_generations[constraint] !=
                endpoint_generations[constraint])
            continue;
        correction = add(correction, endpoint_position_deltas[constraint]);
        impulse = add(impulse, endpoint_impulses[constraint]);
    }
    float allowance = max(0.0f,
        0.25f * characteristic_length - endpoint_correction_used[endpoint]);
    Vec3 applied = clamp_length(correction, allowance);
    if (vector_length(correction) > allowance + 1.0e-6f)
        pmAtomicOr(status, kRigidCouplingCorrectionExceeded);
    endpoint_correction_used[endpoint] += vector_length(applied);
    Vec3 position = add(endpoint_positions[endpoint], applied);
    Vec3 velocity = clamp_length(add(endpoint_velocities[endpoint],
        multiply(impulse, inverseMass)), maximum_speed);
    if (!rigid_coupling_finite_vec3(position) ||
        !rigid_coupling_finite_vec3(velocity))
    {
        pmAtomicOr(status, kRigidCouplingNonfiniteState);
        return;
    }
    endpoint_positions[endpoint] = position;
    endpoint_velocities[endpoint] = velocity;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rigid_coupling_apply_bodies(
    uniform PMBuffer<BodyParameters> body_parameters,
    uniform PMBuffer<RigidBodyState> body_states,
    uniform PMBuffer<float> body_correction_used,
    uniform PMBuffer<uint> body_generations, uniform uint body_count,
    uniform PMBuffer<uint> constraint_bodies,
    uniform PMBuffer<uint> constraint_body_generations,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform PMBuffer<Vec3> body_position_deltas,
    uniform PMBuffer<Vec3> endpoint_impulses,
    uniform PMBuffer<Vec3> body_angular_impulses,
    uniform float characteristic_length, uniform float dt,
    uniform uint profile_version, uniform uint sweep,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint body = pmThreadIndex();
    if (!rigid_coupling_profile_valid(profile_version, dt, sweep))
    {
        if (body == 0u)
            rigid_coupling_record_profile_invalid(
                profile_version, dt, sweep, status);
        return;
    }
    if (body >= body_count ||
        body_parameters[body].motion != MotionType.dynamic) return;
    Vec3 correction = {}, linearImpulse = {}, angularImpulse = {};
    for (uint constraint = 0u; constraint < constraint_count; ++constraint)
    {
        if (constraint_active[constraint] == 0u ||
            constraint_bodies[constraint] != body ||
            constraint_body_generations[constraint] != body_generations[body])
            continue;
        correction = add(correction, body_position_deltas[constraint]);
        linearImpulse = subtract(linearImpulse,
            endpoint_impulses[constraint]);
        angularImpulse = add(angularImpulse,
            body_angular_impulses[constraint]);
    }
    float allowance = max(0.0f,
        0.25f * characteristic_length - body_correction_used[body]);
    Vec3 applied = clamp_length(correction, allowance);
    if (vector_length(correction) > allowance + 1.0e-6f)
        pmAtomicOr(status, kRigidCouplingCorrectionExceeded);
    body_correction_used[body] += vector_length(applied);
    BodyParameters parameters = body_parameters[body];
    RigidBodyState state = body_states[body];
    state.position = add(state.position, applied);
    state.linear_velocity = clamp_length(add(state.linear_velocity,
        multiply(linearImpulse, parameters.inverse_mass)),
        parameters.maximum_linear_speed);
    state.angular_velocity = clamp_length(add(state.angular_velocity,
        inverse_inertia_world(parameters, state, angularImpulse)),
        parameters.maximum_angular_speed);
    if (!rigid_coupling_finite_vec3(state.position) ||
        !rigid_coupling_finite_vec3(state.linear_velocity) ||
        !rigid_coupling_finite_vec3(state.angular_velocity))
    {
        pmAtomicOr(status, kRigidCouplingNonfiniteState);
        return;
    }
    body_states[body] = state;
}
