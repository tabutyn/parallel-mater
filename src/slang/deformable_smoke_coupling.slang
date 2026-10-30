#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Batched deformable/deformable and directed deformable/smoke exchange for
// realtime-single-step.

import types;
import math;
import device_intrinsics;
import graphics_resources;

static const uint kDeformableSmokeProfileVersion = 1u;
static const float kDeformableSmokeTimestep = 1.0f / 60.0f;
static const uint kDeformableSmokeSweeps = 4u;
static const uint kClothSoftBody = 1u;
static const uint kClothRope = 2u;
static const uint kClothSmokeParticle = 3u;
static const uint kClothSmokeGrid = 4u;
static const uint kSoftBodyRope = 5u;
static const uint kSoftBodySmokeParticle = 6u;
static const uint kSoftBodySmokeGrid = 7u;
static const uint kRopeSmokeParticle = 8u;
static const uint kRopeSmokeGrid = 9u;
static const uint kContact = 1u;
static const uint kAttachment = 2u;
static const uint kDrag = 3u;
static const uint kInvalidProfile = 1u << 28u;
static const uint kInvalidTimestep = 1u << 29u;
static const uint kInvalidMaterial = 1u << 30u;
static const uint kCorrectionExceeded = 1u << 31u;

bool deformable_smoke_finite_vec3(Vec3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

bool deformable_smoke_profile_valid(uint profile, float dt, uint sweep)
{
    return profile == kDeformableSmokeProfileVersion && isfinite(dt) &&
        abs(dt - kDeformableSmokeTimestep) <= 1.0e-7f &&
        sweep < kDeformableSmokeSweeps;
}

bool deformable_smoke_pair_has_smoke(uint pair)
{
    return pair == kClothSmokeParticle || pair == kClothSmokeGrid ||
        pair == kSoftBodySmokeParticle || pair == kSoftBodySmokeGrid ||
        pair == kRopeSmokeParticle || pair == kRopeSmokeGrid;
}

bool deformable_smoke_mode_valid(uint pair, uint mode)
{
    if (pair == kClothSoftBody) return mode == kContact;
    if (pair == kClothRope) return mode == kAttachment;
    if (pair == kSoftBodyRope)
        return mode == kContact || mode == kAttachment;
    if (deformable_smoke_pair_has_smoke(pair))
        return mode == kContact || mode == kDrag;
    return false;
}

void deformable_smoke_record_profile_invalid(
    uint profile, float dt, uint sweep, PMBuffer<uint> status)
{
    if (profile != kDeformableSmokeProfileVersion ||
        sweep >= kDeformableSmokeSweeps)
        pmAtomicOr(status, kInvalidProfile);
    if (!isfinite(dt) || abs(dt - kDeformableSmokeTimestep) > 1.0e-7f)
        pmAtomicOr(status, kInvalidTimestep);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void deformable_smoke_coupling_project(
    uniform PMBuffer<Vec3> first_positions,
    uniform PMBuffer<Vec3> first_previous_positions,
    uniform PMBuffer<Vec3> first_velocities,
    uniform PMBuffer<float> first_inverse_masses,
    uniform PMBuffer<uint> first_generations, uniform uint first_count,
    uniform PMBuffer<Vec3> second_positions,
    uniform PMBuffer<Vec3> second_previous_positions,
    uniform PMBuffer<Vec3> second_velocities,
    uniform PMBuffer<float> second_inverse_masses,
    uniform PMBuffer<uint> second_generations, uniform uint second_count,
    uniform PMBuffer<uint> constraint_first_endpoints,
    uniform PMBuffer<uint> constraint_second_endpoints,
    uniform PMBuffer<uint> constraint_first_generations,
    uniform PMBuffer<uint> constraint_second_generations,
    uniform PMBuffer<uint> constraint_pairs,
    uniform PMBuffer<uint> constraint_modes,
    uniform PMBuffer<Vec3> constraint_normals,
    uniform PMBuffer<float> constraint_contact_distances,
    uniform PMBuffer<float> constraint_frictions,
    uniform PMBuffer<float> constraint_responses,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform float characteristic_length, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version, uniform uint sweep,
    uniform PMBuffer<Vec3> first_position_deltas,
    uniform PMBuffer<Vec3> second_position_deltas,
    uniform PMBuffer<Vec3> first_impulses,
    uniform PMBuffer<Vec3> second_impulses,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint constraint = pmThreadIndex();
    if (!deformable_smoke_profile_valid(profile_version, dt, sweep))
    {
        if (constraint == 0u)
            deformable_smoke_record_profile_invalid(
                profile_version, dt, sweep, status);
        return;
    }
    if (constraint >= constraint_count) return;
    first_position_deltas[constraint] = {};
    second_position_deltas[constraint] = {};
    first_impulses[constraint] = {};
    second_impulses[constraint] = {};
    if (constraint_active[constraint] == 0u) return;

    uint first = constraint_first_endpoints[constraint];
    uint second = constraint_second_endpoints[constraint];
    if (first >= first_count || second >= second_count)
    {
        pmAtomicOr(status, kInvalidMaterial);
        return;
    }
    // Resource generations are constraint-aligned. This keeps known endpoint
    // attachments batchable without allocating per-vertex generation arrays.
    if (first_generations[constraint] !=
            constraint_first_generations[constraint] ||
        second_generations[constraint] !=
            constraint_second_generations[constraint])
        return;

    uint pair = constraint_pairs[constraint];
    uint mode = constraint_modes[constraint];
    float distance = constraint_contact_distances[constraint];
    float friction = constraint_frictions[constraint];
    float response = constraint_responses[constraint];
    Vec3 firstPosition = first_positions[first];
    Vec3 secondPosition = second_positions[second];
    Vec3 previousFirst = first_previous_positions[first];
    Vec3 previousSecond = second_previous_positions[second];
    Vec3 firstVelocity = first_velocities[first];
    Vec3 secondVelocity = second_velocities[second];
    float firstInverseMass = first_inverse_masses[first];
    float secondInverseMass = second_inverse_masses[second];
    bool smokePair = deformable_smoke_pair_has_smoke(pair);
    bool firstMassValid = firstInverseMass == 0.0f ||
        (firstInverseMass >= 0.0001f && firstInverseMass <= 1000.0f);
    bool secondMassValid = secondInverseMass == 0.0f ||
        (secondInverseMass >= 0.0001f && secondInverseMass <= 1000.0f);
    float ratio = firstInverseMass > 0.0f && secondInverseMass > 0.0f
        ? max(firstInverseMass, secondInverseMass) /
            min(firstInverseMass, secondInverseMass) : 1.0f;
    bool receiverValid = smokePair
        ? (mode == kContact ? secondInverseMass > 0.0f
                            : firstInverseMass > 0.0f)
        : firstInverseMass + secondInverseMass > 0.0f;
    bool baseValid = deformable_smoke_mode_valid(pair, mode) &&
        isfinite(characteristic_length) && characteristic_length >= 0.005f &&
        characteristic_length <= 2.0f && isfinite(maximum_speed) &&
        maximum_speed >= 0.0f && maximum_speed <= 20.0f &&
        maximum_speed * dt <= 0.5f * characteristic_length &&
        isfinite(firstInverseMass) && firstInverseMass >= 0.0f &&
        isfinite(secondInverseMass) && secondInverseMass >= 0.0f &&
        firstMassValid && secondMassValid && receiverValid &&
        (smokePair || ratio <= 100.0f) &&
        deformable_smoke_finite_vec3(firstPosition) &&
        deformable_smoke_finite_vec3(secondPosition) &&
        deformable_smoke_finite_vec3(previousFirst) &&
        deformable_smoke_finite_vec3(previousSecond) &&
        deformable_smoke_finite_vec3(firstVelocity) &&
        deformable_smoke_finite_vec3(secondVelocity) &&
        abs(firstPosition.x) <= 10000.0f &&
        abs(firstPosition.y) <= 10000.0f &&
        abs(firstPosition.z) <= 10000.0f &&
        abs(secondPosition.x) <= 10000.0f &&
        abs(secondPosition.y) <= 10000.0f &&
        abs(secondPosition.z) <= 10000.0f;
    if (!baseValid)
    {
        pmAtomicOr(status, kInvalidMaterial);
        return;
    }

    Vec3 relativeVelocity = subtract(firstVelocity, secondVelocity);
    Vec3 firstCorrection = {};
    Vec3 secondCorrection = {};
    Vec3 firstImpulse = {};
    Vec3 secondImpulse = {};
    if (mode == kContact)
    {
        Vec3 normal = constraint_normals[constraint];
        float normalLength = vector_length(normal);
        bool valid = deformable_smoke_finite_vec3(normal) &&
            normalLength > 1.0e-6f && isfinite(distance) &&
            distance > 0.0f && distance <= characteristic_length &&
            isfinite(friction) && friction >= 0.0f && friction <= 2.0f &&
            response == 0.0f;
        if (!valid)
        {
            pmAtomicOr(status, kInvalidMaterial);
            return;
        }
        normal = multiply(normal, 1.0f / normalLength);
        float separation = dot(subtract(firstPosition, secondPosition), normal);
        float previousSeparation = dot(
            subtract(previousFirst, previousSecond), normal);
        if (!isfinite(separation) || !isfinite(previousSeparation))
        {
            pmAtomicOr(status, kInvalidMaterial);
            return;
        }
        float penetration = max(0.0f, distance - separation);
        if (previousSeparation >= distance && separation < distance)
            penetration = max(penetration, distance - separation);
        float correction = min(penetration,
            0.25f * characteristic_length);
        float closingSpeed = dot(relativeVelocity, normal);
        Vec3 tangent = subtract(relativeVelocity,
            multiply(normal, closingSpeed));
        float tangentSpeed = vector_length(tangent);
        if (smokePair)
        {
            secondCorrection = multiply(normal, -correction);
            float normalMagnitude = closingSpeed < 0.0f
                ? -closingSpeed / secondInverseMass : 0.0f;
            secondImpulse = multiply(normal, -normalMagnitude);
            if (tangentSpeed > 1.0e-6f && friction > 0.0f)
            {
                float tangentMagnitude = min(
                    tangentSpeed / secondInverseMass,
                    friction * normalMagnitude);
                secondImpulse = add(secondImpulse,
                    multiply(tangent, tangentMagnitude / tangentSpeed));
            }
        }
        else
        {
            float inverseMass = firstInverseMass + secondInverseMass;
            firstCorrection = multiply(normal,
                correction * firstInverseMass / inverseMass);
            secondCorrection = multiply(normal,
                -correction * secondInverseMass / inverseMass);
            float normalMagnitude = closingSpeed < 0.0f
                ? -closingSpeed / inverseMass : 0.0f;
            firstImpulse = multiply(normal, normalMagnitude);
            if (tangentSpeed > 1.0e-6f && friction > 0.0f)
            {
                float tangentMagnitude = min(tangentSpeed / inverseMass,
                    friction * normalMagnitude);
                firstImpulse = add(firstImpulse, multiply(tangent,
                    -tangentMagnitude / tangentSpeed));
            }
            secondImpulse = multiply(firstImpulse, -1.0f);
        }
    }
    else
    {
        bool valid = distance == 0.0f && friction == 0.0f &&
            isfinite(response) && response >= 0.05f && response <= 0.995f;
        if (!valid)
        {
            pmAtomicOr(status, kInvalidMaterial);
            return;
        }
        if (mode == kAttachment)
        {
            float inverseMass = firstInverseMass + secondInverseMass;
            Vec3 correction = clamp_length(multiply(
                subtract(secondPosition, firstPosition), response),
                0.25f * characteristic_length);
            firstCorrection = multiply(correction,
                firstInverseMass / inverseMass);
            secondCorrection = multiply(correction,
                -secondInverseMass / inverseMass);
            firstImpulse = multiply(relativeVelocity,
                -response / inverseMass);
            secondImpulse = multiply(firstImpulse, -1.0f);
        }
        else
        {
            // API 0.2 smoke drag is smoke -> deformable. The smoke sample is a
            // prescribed field/tracer value and receives no opposite proposal.
            firstImpulse = multiply(relativeVelocity,
                -response / firstInverseMass);
        }
    }
    if (!deformable_smoke_finite_vec3(firstCorrection) ||
        !deformable_smoke_finite_vec3(secondCorrection) ||
        !deformable_smoke_finite_vec3(firstImpulse) ||
        !deformable_smoke_finite_vec3(secondImpulse))
    {
        pmAtomicOr(status, kInvalidMaterial);
        return;
    }
    first_position_deltas[constraint] = firstCorrection;
    second_position_deltas[constraint] = secondCorrection;
    first_impulses[constraint] = firstImpulse;
    second_impulses[constraint] = secondImpulse;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void deformable_smoke_coupling_apply_first(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<uint> generations, uniform uint endpoint_count,
    uniform PMBuffer<uint> constraint_endpoints,
    uniform PMBuffer<uint> constraint_generations,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<Vec3> impulses,
    uniform float characteristic_length, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version, uniform uint sweep,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint endpoint = pmThreadIndex();
    if (!deformable_smoke_profile_valid(profile_version, dt, sweep))
    {
        if (endpoint == 0u)
            deformable_smoke_record_profile_invalid(
                profile_version, dt, sweep, status);
        return;
    }
    if (endpoint >= endpoint_count || inverse_masses[endpoint] == 0.0f)
        return;
    Vec3 correction = {}, impulse = {};
    for (uint constraint = 0u; constraint < constraint_count; ++constraint)
    {
        if (constraint_active[constraint] == 0u ||
            constraint_endpoints[constraint] != endpoint ||
            constraint_generations[constraint] != generations[constraint])
            continue;
        correction = add(correction, position_deltas[constraint]);
        impulse = add(impulse, impulses[constraint]);
    }
    float allowance = max(0.0f,
        0.25f * characteristic_length - correction_used[endpoint]);
    Vec3 applied = clamp_length(correction, allowance);
    if (vector_length(correction) > allowance + 1.0e-6f)
    {
        pmAtomicOr(status, kCorrectionExceeded);
        return;
    }
    correction_used[endpoint] += vector_length(applied);
    Vec3 velocityDelta = multiply(impulse, inverse_masses[endpoint]);
    Vec3 position = add(add(positions[endpoint], applied),
        multiply(velocityDelta, dt));
    Vec3 previousPosition = add(previous_positions[endpoint], applied);
    Vec3 velocity = clamp_length(add(
        velocities[endpoint], velocityDelta), maximum_speed);
    if (!deformable_smoke_finite_vec3(position) ||
        !deformable_smoke_finite_vec3(previousPosition) ||
        !deformable_smoke_finite_vec3(velocity))
    {
        pmAtomicOr(status, kInvalidMaterial);
        return;
    }
    positions[endpoint] = position;
    previous_positions[endpoint] = previousPosition;
    velocities[endpoint] = velocity;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void deformable_smoke_coupling_apply_second(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<uint> generations, uniform uint endpoint_count,
    uniform PMBuffer<uint> constraint_endpoints,
    uniform PMBuffer<uint> constraint_generations,
    uniform PMByteBuffer constraint_active, uniform uint constraint_count,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<Vec3> impulses,
    uniform float characteristic_length, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version, uniform uint sweep,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint endpoint = pmThreadIndex();
    if (!deformable_smoke_profile_valid(profile_version, dt, sweep))
    {
        if (endpoint == 0u)
            deformable_smoke_record_profile_invalid(
                profile_version, dt, sweep, status);
        return;
    }
    if (endpoint >= endpoint_count || inverse_masses[endpoint] == 0.0f)
        return;
    Vec3 correction = {}, impulse = {};
    for (uint constraint = 0u; constraint < constraint_count; ++constraint)
    {
        if (constraint_active[constraint] == 0u ||
            constraint_endpoints[constraint] != endpoint ||
            constraint_generations[constraint] != generations[constraint])
            continue;
        correction = add(correction, position_deltas[constraint]);
        impulse = add(impulse, impulses[constraint]);
    }
    float allowance = max(0.0f,
        0.25f * characteristic_length - correction_used[endpoint]);
    Vec3 applied = clamp_length(correction, allowance);
    if (vector_length(correction) > allowance + 1.0e-6f)
    {
        pmAtomicOr(status, kCorrectionExceeded);
        return;
    }
    correction_used[endpoint] += vector_length(applied);
    Vec3 velocityDelta = multiply(impulse, inverse_masses[endpoint]);
    Vec3 position = add(add(positions[endpoint], applied),
        multiply(velocityDelta, dt));
    Vec3 previousPosition = add(previous_positions[endpoint], applied);
    Vec3 velocity = clamp_length(add(
        velocities[endpoint], velocityDelta), maximum_speed);
    if (!deformable_smoke_finite_vec3(position) ||
        !deformable_smoke_finite_vec3(previousPosition) ||
        !deformable_smoke_finite_vec3(velocity))
    {
        pmAtomicOr(status, kInvalidMaterial);
        return;
    }
    positions[endpoint] = position;
    previous_positions[endpoint] = previousPosition;
    velocities[endpoint] = velocity;
}
