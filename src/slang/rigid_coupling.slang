#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Batched rigid/deformable and rigid/smoke exchange for realtime-single-step.

import types;
import math;
import device_intrinsics;
import graphics_resources;

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
    if (endpoint_generations[endpoint] !=
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
            endpointCorrection = multiply(correction,
                endpointInverseMass / linearInverseMass);
            body_position_deltas[constraint] = multiply(correction,
                -bodyInverseMass / linearInverseMass);
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
                endpoint_generations[endpoint])
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
