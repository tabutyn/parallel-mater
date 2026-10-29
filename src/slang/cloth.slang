#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Shared cloth simulation kernels.

import types;
import math;
import device_intrinsics;
import graphics_resources;

static const uint kClothVolumeThreads = 128u;
groupshared float clothScalarTerms[kClothVolumeThreads];
groupshared float clothSignedVolume;
groupshared float clothDeltaLambda;
groupshared uint clothApplyCorrection;
groupshared uint clothChanged;

// realtime-single-step v1 advances once at 1/60 s and performs exactly eight
// algebraic sweeps. Legacy cloth entry points below remain profile 0.
static const uint kClothRealtimeProfileVersion = 1u;
static const float kClothRealtimeTimestep = 1.0f / 60.0f;
static const float kClothRealtimeSolverDamping = 0.05f;
static const float kClothRealtimeMaximumCorrectionFraction = 0.25f;
static const float kClothRealtimeStructuralCorrectionFraction = 0.20f;
static const uint kClothRealtimeInvalidProfile = 1u << 17u;
static const uint kClothRealtimeInvalidTimestep = 1u << 18u;
static const uint kClothRealtimeInvalidMaterial = 1u << 19u;
static const uint kClothRealtimeNonfiniteState = 1u << 20u;
static const uint kClothRealtimeUnresolvedContact = 1u << 21u;

groupshared float clothRealtimeScalarTerms[kClothVolumeThreads];
groupshared float clothRealtimeSignedVolume;
groupshared float clothRealtimeDeltaLambda;
groupshared uint clothRealtimeApplyCorrection;

bool cloth_realtime_finite_vec3(Vec3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

bool cloth_realtime_profile_valid(uint profile_version, float dt)
{
    return profile_version == kClothRealtimeProfileVersion && isfinite(dt) &&
        abs(dt - kClothRealtimeTimestep) <= 1.0e-7f;
}

void cloth_realtime_record_profile_invalid(
    uint profile_version, float dt, PMBuffer<uint> status)
{
    if (profile_version != kClothRealtimeProfileVersion)
        pmAtomicOr(status, kClothRealtimeInvalidProfile);
    if (!isfinite(dt) || abs(dt - kClothRealtimeTimestep) > 1.0e-7f)
        pmAtomicOr(status, kClothRealtimeInvalidTimestep);
}

bool cloth_realtime_vertex_material_valid(
    float inverse_mass, float characteristic_length, float maximum_speed,
    float velocity_damping, Vec3 gravity, float dt)
{
    float mass = inverse_mass > 0.0f ? 1.0f / inverse_mass : 0.001f;
    return isfinite(inverse_mass) && inverse_mass >= 0.0f &&
        isfinite(characteristic_length) &&
        characteristic_length >= 0.005f && characteristic_length <= 2.0f &&
        mass >= 0.001f && mass <= 10000.0f &&
        isfinite(maximum_speed) && maximum_speed >= 0.0f &&
        maximum_speed <= 20.0f && maximum_speed * dt <=
            0.5f * characteristic_length && isfinite(velocity_damping) &&
        velocity_damping >= 0.0f && cloth_realtime_finite_vec3(gravity) &&
        vector_length(gravity) * dt * dt <=
            0.25f * characteristic_length;
}

bool cloth_realtime_compliance_valid(
    float compliance, float effective_inverse_mass, float dt)
{
    if (!isfinite(compliance) || compliance < 0.0f ||
        !isfinite(effective_inverse_mass) || effective_inverse_mass <= 0.0f)
        return false;
    if (compliance == 0.0f) return true;
    float ratio = compliance /
        (dt * dt * effective_inverse_mass);
    return ratio >= 0.005025125628140708f && ratio <= 19.0f;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_realtime_predict(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform uint count, uniform uint capacity, uniform Vec3 gravity,
    uniform float velocity_damping, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> time_advances,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (!cloth_realtime_profile_valid(profile_version, dt))
    {
        if (vertex == 0u)
            cloth_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (vertex == 0u) ++time_advances[0];
    if (vertex >= capacity || vertex >= count) return;
    float inverseMass = inverse_masses[vertex];
    float characteristicLength = characteristic_lengths[vertex];
    if (!cloth_realtime_vertex_material_valid(inverseMass,
        characteristicLength, maximum_speed, velocity_damping, gravity, dt))
    {
        pmAtomicOr(status, kClothRealtimeInvalidMaterial);
        return;
    }
    Vec3 position = positions[vertex];
    previous_positions[vertex] = position;
    correction_used[vertex] = 0.0f;
    if (inverseMass == 0.0f)
    {
        velocities[vertex] = Vec3(0.0f);
        return;
    }
    Vec3 velocity = add(velocities[vertex], multiply(gravity, dt));
    velocity = multiply(velocity, exp(-velocity_damping * dt));
    velocity = clamp_length(velocity, min(maximum_speed,
        0.5f * characteristicLength / dt));
    position = add(position, multiply(velocity, dt));
    if (!cloth_realtime_finite_vec3(position) ||
        !cloth_realtime_finite_vec3(velocity) ||
        abs(position.x) > 10000.0f || abs(position.y) > 10000.0f ||
        abs(position.z) > 10000.0f)
    {
        pmAtomicOr(status, kClothRealtimeNonfiniteState);
        return;
    }
    positions[vertex] = position;
    velocities[vertex] = velocity;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_realtime_project_links(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform PMBuffer<uint> offsets,
    uniform PMBuffer<DeformableNeighbor> neighbors,
    uniform PMByteBuffer bond_active, uniform uint count,
    uniform uint capacity, uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (!cloth_realtime_profile_valid(profile_version, dt))
    {
        if (vertex == 0u)
            cloth_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (vertex >= capacity || vertex >= count) return;
    position_deltas[vertex] = Vec3(0.0f);
    float selfMass = inverse_masses[vertex];
    if (selfMass == 0.0f) return;
    Vec3 position = positions[vertex];
    Vec3 correction = Vec3(0.0f);
    uint activeCount = 0u;
    for (uint edge = offsets[vertex]; edge < offsets[vertex + 1u]; ++edge)
    {
        DeformableNeighbor neighbor = neighbors[edge];
        if (neighbor.index >= count)
        {
            pmAtomicOr(status, kClothRealtimeInvalidMaterial);
            continue;
        }
        if (pmBufferPresent(bond_active) && neighbor.bond != kInvalidDense &&
            bond_active[neighbor.bond] == 0u) continue;
        float otherMass = inverse_masses[neighbor.index];
        float effectiveInverseMass = selfMass + otherMass;
        if (!isfinite(neighbor.rest_length) || neighbor.rest_length < 0.005f ||
            neighbor.rest_length > 2.0f ||
            !cloth_realtime_compliance_valid(neighbor.compliance,
                effectiveInverseMass, dt))
        {
            pmAtomicOr(status, kClothRealtimeInvalidMaterial);
            continue;
        }
        Vec3 difference = subtract(position, positions[neighbor.index]);
        float edgeLength = vector_length(difference);
        if (!isfinite(edgeLength) || edgeLength < 1.0e-7f)
        {
            if (!isfinite(edgeLength))
                pmAtomicOr(status, kClothRealtimeNonfiniteState);
            continue;
        }
        float denominator = effectiveInverseMass +
            neighbor.compliance / (dt * dt);
        float amount = -selfMass * (edgeLength - neighbor.rest_length) /
            (denominator * edgeLength);
        correction = add(correction, multiply(difference, amount));
        ++activeCount;
    }
    if (activeCount != 0u)
        correction = multiply(correction, 1.0f / float(activeCount));
    // Reserve the final 0.05 characteristic length for non-penetration.
    float maximumCorrection = kClothRealtimeStructuralCorrectionFraction *
        characteristic_lengths[vertex];
    float remaining = max(0.0f,
        maximumCorrection - correction_used[vertex]);
    correction = clamp_length(correction, remaining);
    if (!cloth_realtime_finite_vec3(correction))
    {
        pmAtomicOr(status, kClothRealtimeNonfiniteState);
        correction = Vec3(0.0f);
    }
    position_deltas[vertex] = correction;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_realtime_apply_position_delta(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform uint count, uniform uint capacity, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (!cloth_realtime_profile_valid(profile_version, dt))
    {
        if (vertex == 0u)
            cloth_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (vertex >= capacity || vertex >= count ||
        inverse_masses[vertex] == 0.0f) return;
    Vec3 delta = position_deltas[vertex];
    float used = correction_used[vertex] + vector_length(delta);
    float maximumCorrection = kClothRealtimeMaximumCorrectionFraction *
        characteristic_lengths[vertex];
    Vec3 position = add(positions[vertex], delta);
    if (!cloth_realtime_finite_vec3(position) || !isfinite(used) ||
        used > maximumCorrection + 1.0e-6f)
    {
        pmAtomicOr(status, kClothRealtimeNonfiniteState);
        return;
    }
    positions[vertex] = position;
    correction_used[vertex] = min(used, maximumCorrection);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_realtime_project_volume(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<uint> indices, uniform uint vertex_count,
    uniform uint triangle_count, uniform PMBuffer<uint> corner_offsets,
    uniform PMBuffer<uint> corner_indices, uniform PMBuffer<Vec3> gradients,
    uniform float target_volume, uniform float orientation,
    uniform float compliance, uniform bool enabled, uniform float dt,
    uniform uint profile_version, uniform PMBuffer<float> lambda,
    uniform bool reset_lambda,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u) return;
    uint thread = pmGroupThreadIndex().x;
    if (!cloth_realtime_profile_valid(profile_version, dt))
    {
        if (thread == 0u)
            cloth_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (!enabled) return;
    if (vertex_count == 0u || triangle_count == 0u ||
        !isfinite(target_volume) || target_volume <= 0.0f ||
        !isfinite(orientation) || abs(orientation) != 1.0f ||
        !isfinite(compliance) || compliance < 0.0f)
    {
        if (thread == 0u)
            pmAtomicOr(status, kClothRealtimeInvalidMaterial);
        return;
    }
    if (thread == 0u)
    {
        if (reset_lambda) lambda[0] = 0.0f;
        clothRealtimeSignedVolume = 0.0f;
    }
    pmGroupBarrier();
    for (uint wave = 0u; wave < triangle_count; wave += pmGroupSize().x)
    {
        uint triangle = wave + thread;
        if (triangle < triangle_count)
        {
            uint aIndex = indices[3u * triangle];
            uint bIndex = indices[3u * triangle + 1u];
            uint cIndex = indices[3u * triangle + 2u];
            Vec3 a = positions[aIndex];
            Vec3 b = positions[bIndex];
            Vec3 c = positions[cIndex];
            clothRealtimeScalarTerms[thread] =
                dot(a, cross(b, c)) / 6.0f;
        }
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint itemCount = min(pmGroupSize().x, triangle_count - wave);
            for (uint item = 0u; item < itemCount; ++item)
                clothRealtimeSignedVolume += clothRealtimeScalarTerms[item];
        }
        pmGroupBarrier();
    }
    for (uint vertex = thread; vertex < vertex_count;
         vertex += pmGroupSize().x)
    {
        Vec3 gradient = Vec3(0.0f);
        for (uint item = corner_offsets[vertex];
             item < corner_offsets[vertex + 1u]; ++item)
        {
            uint corner = corner_indices[item];
            uint base = corner - corner % 3u;
            Vec3 a = positions[indices[base]];
            Vec3 b = positions[indices[base + 1u]];
            Vec3 c = positions[indices[base + 2u]];
            Vec3 contribution = corner % 3u == 0u ? cross(b, c) :
                corner % 3u == 1u ? cross(c, a) : cross(a, b);
            gradient = add(gradient,
                multiply(contribution, orientation / 6.0f));
        }
        gradients[vertex] = gradient;
    }
    pmGroupBarrier();
    // Reduce deterministically on thread zero. The bounded F8 workload keeps
    // this one-group serial reduction cheaper than atomic accumulation.
    if (thread == 0u)
    {
        float inverseMassSum = 0.0f;
        for (uint vertex = 0u; vertex < vertex_count; ++vertex)
            inverseMassSum += inverse_masses[vertex] *
                length_squared(gradients[vertex]);
        clothRealtimeApplyCorrection = inverseMassSum > 1.0e-12f ? 1u : 0u;
        if (clothRealtimeApplyCorrection != 0u)
        {
            float alpha = compliance / (dt * dt);
            float ratio = alpha / inverseMassSum;
            if (compliance > 0.0f &&
                (ratio < 0.005025125628140708f || ratio > 19.0f))
            {
                pmAtomicOr(status, kClothRealtimeInvalidMaterial);
                clothRealtimeApplyCorrection = 0u;
            }
            float constraint = orientation * clothRealtimeSignedVolume -
                target_volume;
            if (clothRealtimeApplyCorrection != 0u)
            {
                clothRealtimeDeltaLambda =
                    (-constraint - alpha * lambda[0]) /
                    (inverseMassSum + alpha);
                lambda[0] += clothRealtimeDeltaLambda;
            }
        }
    }
    pmGroupBarrier();
    if (clothRealtimeApplyCorrection == 0u) return;
    for (uint vertex = thread; vertex < vertex_count;
         vertex += pmGroupSize().x)
    {
        if (inverse_masses[vertex] == 0.0f) continue;
        Vec3 correction = multiply(gradients[vertex],
            inverse_masses[vertex] * clothRealtimeDeltaLambda);
        float maximumCorrection =
            kClothRealtimeStructuralCorrectionFraction *
            characteristic_lengths[vertex];
        float remaining = max(0.0f,
            maximumCorrection - correction_used[vertex]);
        correction = clamp_length(correction, remaining);
        Vec3 position = add(positions[vertex], correction);
        if (!cloth_realtime_finite_vec3(position) ||
            !cloth_realtime_finite_vec3(correction))
        {
            pmAtomicOr(status, kClothRealtimeNonfiniteState);
            continue;
        }
        positions[vertex] = position;
        correction_used[vertex] += vector_length(correction);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_realtime_project_contacts(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<Vec3> contact_normals,
    uniform PMBuffer<float> contact_offsets,
    uniform PMByteBuffer contact_active,
    uniform uint count, uniform uint capacity, uniform float thickness,
    uniform float friction, uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (!cloth_realtime_profile_valid(profile_version, dt))
    {
        if (vertex == 0u)
            cloth_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (vertex >= capacity || vertex >= count ||
        inverse_masses[vertex] == 0.0f || contact_active[vertex] == 0u)
        return;
    Vec3 normal = contact_normals[vertex];
    float normalLength = vector_length(normal);
    if (!cloth_realtime_finite_vec3(normal) || normalLength < 1.0e-6f ||
        !isfinite(contact_offsets[vertex]) || !isfinite(thickness) ||
        thickness < 0.0f || !isfinite(friction) ||
        friction < 0.0f || friction > 2.0f)
    {
        pmAtomicOr(status, kClothRealtimeInvalidMaterial);
        return;
    }
    normal = multiply(normal, 1.0f / normalLength);
    Vec3 position = positions[vertex];
    float target = contact_offsets[vertex] + thickness;
    float signedDistance = dot(position, normal);
    float penetration = target - signedDistance;
    if (penetration <= 0.0f) return;
    Vec3 motion = subtract(position, previous_positions[vertex]);
    Vec3 tangent = subtract(motion, multiply(normal, dot(motion, normal)));
    float tangentLength = vector_length(tangent);
    Vec3 frictionCorrection = tangentLength > 1.0e-7f
        ? multiply(tangent,
            -min(tangentLength, friction * penetration) / tangentLength)
        : Vec3(0.0f);
    Vec3 correction = add(multiply(normal, penetration), frictionCorrection);
    float maximumCorrection = kClothRealtimeMaximumCorrectionFraction *
        characteristic_lengths[vertex];
    float remaining = max(0.0f,
        maximumCorrection - correction_used[vertex]);
    correction = clamp_length(correction, remaining);
    position = add(position, correction);
    if (!cloth_realtime_finite_vec3(position) ||
        !cloth_realtime_finite_vec3(correction))
    {
        pmAtomicOr(status, kClothRealtimeNonfiniteState);
        return;
    }
    if (dot(position, normal) < target - 1.0e-6f)
        pmAtomicOr(status, kClothRealtimeUnresolvedContact);
    positions[vertex] = position;
    correction_used[vertex] += vector_length(correction);
}

void cloth_evaluate_bond_break(
    uint edgeIndex, PMBuffer<Vec3> positions, PMBuffer<ClothBond> bonds,
    PMByteBuffer active, PMByteBuffer damage, uint count,
    float break_strain, uint persistence,
    PMBuffer<FluidBodyImpulse> contact_impulses, float impact_threshold)
{
    if (edgeIndex >= count || active[edgeIndex] == 0u) return;
    ClothBond bond = bonds[edgeIndex];
    if (impact_threshold > 0.0f && pmBufferPresent(contact_impulses) &&
        vector_length(contact_impulses[bond.first].linear) +
        vector_length(contact_impulses[bond.second].linear) > impact_threshold)
    {
        active[edgeIndex] = 0u;
        damage[edgeIndex] = 0u;
        return;
    }
    if (break_strain <= 0.0f) return;
    float currentLength = vector_length(subtract(
        positions[bond.first], positions[bond.second]));
    if (!isfinite(currentLength) ||
        currentLength <= bond.rest_length * (1.0f + break_strain))
    {
        damage[edgeIndex] = 0u;
        return;
    }
    uint next = min(persistence, uint(damage[edgeIndex]) + 1u);
    damage[edgeIndex] = PMResourceUint8(next);
    if (next >= persistence)
    {
        active[edgeIndex] = 0u;
        damage[edgeIndex] = 0u;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_realtime_commit(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform PMBuffer<Vec3> contact_normals,
    uniform PMBuffer<float> contact_offsets,
    uniform PMByteBuffer contact_active,
    uniform PMBuffer<ClothBond> bonds, uniform PMByteBuffer bond_active,
    uniform PMByteBuffer bond_damage, uniform uint bond_count,
    uniform float break_strain, uniform uint fracture_persistence,
    uniform PMBuffer<FluidBodyImpulse> contact_impulses,
    uniform float impact_break_threshold,
    uniform uint count, uniform uint capacity, uniform float thickness,
    uniform float friction, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (!cloth_realtime_profile_valid(profile_version, dt))
    {
        if (vertex == 0u)
            cloth_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    cloth_evaluate_bond_break(vertex, positions, bonds, bond_active,
        bond_damage, bond_count, break_strain, fracture_persistence,
        contact_impulses, impact_break_threshold);
    if (vertex >= capacity || vertex >= count) return;
    if (inverse_masses[vertex] == 0.0f)
    {
        velocities[vertex] = Vec3(0.0f);
        return;
    }
    Vec3 velocity = multiply(subtract(positions[vertex],
        previous_positions[vertex]),
        (1.0f - kClothRealtimeSolverDamping) / dt);
    if (contact_active[vertex] != 0u)
    {
        Vec3 normal = normalized_or(contact_normals[vertex], Vec3(0.0f));
        float target = contact_offsets[vertex] + thickness;
        if (vector_length(normal) > 0.0f &&
            dot(positions[vertex], normal) <=
                target + 1.0e-5f * characteristic_lengths[vertex])
        {
            float normalSpeed = dot(velocity, normal);
            Vec3 normalVelocity = multiply(normal, max(normalSpeed, 0.0f));
            Vec3 tangentVelocity = subtract(velocity,
                multiply(normal, normalSpeed));
            velocity = add(normalVelocity, multiply(tangentVelocity,
                1.0f - min(friction, 1.0f)));
        }
    }
    velocity = clamp_length(velocity, maximum_speed);
    if (!cloth_realtime_finite_vec3(positions[vertex]) ||
        !cloth_realtime_finite_vec3(velocity))
    {
        pmAtomicOr(status, kClothRealtimeNonfiniteState);
        positions[vertex] = previous_positions[vertex];
        velocities[vertex] = Vec3(0.0f);
        return;
    }
    velocities[vertex] = velocity;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_project_volume(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<uint> indices, uniform uint vertex_count,
    uniform uint triangle_count, uniform PMBuffer<uint> corner_offsets,
    uniform PMBuffer<uint> corner_indices, uniform PMBuffer<Vec3> gradients,
    uniform float target_volume, uniform float orientation,
    uniform float compliance, uniform float dt, uniform PMBuffer<float> lambda,
    uniform bool reset_lambda PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u) return;
    uint thread = pmGroupThreadIndex().x;
    if (thread == 0u)
    {
        if (reset_lambda) lambda[0] = 0.0f;
        clothSignedVolume = 0.0f;
    }
    pmGroupBarrier();
    for (uint wave = 0u; wave < triangle_count; wave += pmGroupSize().x)
    {
        uint triangle = wave + thread;
        if (triangle < triangle_count)
        {
            uint aIndex = indices[3u * triangle];
            uint bIndex = indices[3u * triangle + 1u];
            uint cIndex = indices[3u * triangle + 2u];
            Vec3 a = positions[aIndex];
            Vec3 b = positions[bIndex];
            Vec3 c = positions[cIndex];
            clothScalarTerms[thread] = dot(a, cross(b, c)) / 6.0f;
        }
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint count = min(pmGroupSize().x, triangle_count - wave);
            for (uint item = 0u; item < count; ++item)
                clothSignedVolume += clothScalarTerms[item];
        }
        pmGroupBarrier();
    }
    for (uint vertex = thread; vertex < vertex_count;
         vertex += pmGroupSize().x)
    {
        Vec3 gradient = {};
        for (uint item = corner_offsets[vertex];
             item < corner_offsets[vertex + 1u]; ++item)
        {
            uint corner = corner_indices[item];
            uint base = corner - corner % 3u;
            Vec3 a = positions[indices[base]];
            Vec3 b = positions[indices[base + 1u]];
            Vec3 c = positions[indices[base + 2u]];
            Vec3 contribution = corner % 3u == 0u ? cross(b, c) :
                corner % 3u == 1u ? cross(c, a) : cross(a, b);
            gradient = add(gradient,
                multiply(contribution, orientation / 6.0f));
        }
        gradients[vertex] = gradient;
    }
    pmGroupBarrier();
    if (thread == 0u) clothApplyCorrection = 0u;
    pmGroupBarrier();
    float inverseMassSum = 0.0f;
    for (uint wave = 0u; wave < vertex_count; wave += pmGroupSize().x)
    {
        uint vertex = wave + thread;
        if (vertex < vertex_count)
            clothScalarTerms[thread] = inverse_masses[vertex] *
                length_squared(gradients[vertex]);
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint count = min(pmGroupSize().x, vertex_count - wave);
            for (uint item = 0u; item < count; ++item)
                inverseMassSum += clothScalarTerms[item];
        }
        pmGroupBarrier();
    }
    if (thread == 0u)
    {
        clothApplyCorrection = inverseMassSum > 1.0e-12f ? 1u : 0u;
        if (clothApplyCorrection != 0u)
        {
            float alpha = compliance / (dt * dt);
            float constraint =
                orientation * clothSignedVolume - target_volume;
            clothDeltaLambda =
                (-constraint - alpha * (*lambda)) / (inverseMassSum + alpha);
            lambda[0] += clothDeltaLambda;
        }
    }
    pmGroupBarrier();
    if (clothApplyCorrection == 0u) return;
    for (uint vertex = thread; vertex < vertex_count;
         vertex += pmGroupSize().x)
    {
        if (inverse_masses[vertex] == 0.0f) continue;
        positions[vertex] = add(positions[vertex], multiply(
            gradients[vertex], inverse_masses[vertex] * clothDeltaLambda));
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_break_bonds(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<ClothBond> bonds,
    uniform PMByteBuffer active, uniform PMByteBuffer damage, uniform uint count,
    uniform float break_strain, uniform uint persistence,
    uniform PMBuffer<FluidBodyImpulse> contact_impulses,
    uniform float impact_threshold PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint edgeIndex = pmThreadIndex();
    cloth_evaluate_bond_break(edgeIndex, positions, bonds, active, damage,
        count, break_strain, persistence, contact_impulses, impact_threshold);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_limit_strain(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> scratch,
    uniform PMBuffer<float> inverse_masses, uniform PMBuffer<uint> offsets,
    uniform PMBuffer<DeformableNeighbor> neighbors, uniform PMByteBuffer free_nodes,
    uniform uint count PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint thread = pmGroupThreadIndex().x;
    for (uint pass = 0u; pass < 32u; ++pass)
    {
        if (thread == 0u) clothChanged = 0u;
        pmGroupBarrier();
        bool changed = false;
        for (uint node = thread; node < count; node += pmGroupSize().x)
        {
            Vec3 correction = {};
            uint degree = 0u;
            for (uint item = offsets[node];
                 free_nodes[node] != 0u && item < offsets[node + 1u]; ++item)
            {
                DeformableNeighbor edge = neighbors[item];
                if (edge.bond != kInvalidDense) continue;
                float weight = inverse_masses[node] +
                    inverse_masses[edge.index];
                Vec3 delta = subtract(positions[edge.index], positions[node]);
                float currentLength = vector_length(delta);
                float maximum = edge.rest_length * 1.10f;
                if (weight <= 0.0f || currentLength <= maximum ||
                    inverse_masses[node] == 0.0f) continue;
                correction = add(correction, multiply(delta,
                    (currentLength - maximum) * inverse_masses[node] /
                    (currentLength * weight)));
                changed = changed || currentLength > maximum * 1.0001f;
                ++degree;
            }
            scratch[node] = add(positions[node],
                multiply(correction, 1.0f / float(max(1u, degree))));
        }
        if (changed)
        {
            uint ignored;
            InterlockedOr(clothChanged, 1u, ignored);
        }
        pmGroupBarrier();
        for (uint node = thread; node < count; node += pmGroupSize().x)
            positions[node] = scratch[node];
        pmGroupBarrier();
        if (clothChanged == 0u) break;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_update_surface(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<uint> source_indices,
    uniform PMBuffer<Vec3> surface_positions, uniform uint triangle_count PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint triangle = pmThreadIndex();
    if (triangle >= triangle_count) return;
    uint base = 3u * triangle;
    for (uint corner = 0u; corner < 3u; ++corner)
        surface_positions[base + corner] =
            positions[source_indices[base + corner]];
}
