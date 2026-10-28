#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Shared soft-body and deformable simulation kernels.

import types;
import math;
import device_intrinsics;
import graphics_resources;

static const uint kSoftReductionThreads = 128u;
static const uint kSoftRealtimeBatchCapacity = 10u;
#ifndef PM_SOFT_NEIGHBOR_GROUP_SIZE
#define PM_SOFT_NEIGHBOR_GROUP_SIZE 16
#endif
// The adapter selects 16 masked lanes or a full 32-lane SIMD group. This
// changes launch geometry, not the ordered neighbor accumulation below.
static const uint kSoftNeighborGroupSize = PM_SOFT_NEIGHBOR_GROUP_SIZE;

// Native adapters may group independent resources into one command. These
// records contain no alternate equations: the batch entry points below call
// the ordinary shared kernels with one record per dispatch-grid row.
struct SoftBodyRealtimePredictBatchItem
{
    PMBuffer<Vec3> positions;
    PMBuffer<Vec3> velocities;
    PMBuffer<Vec3> previous_positions;
    PMBuffer<float> correction_used;
    PMBuffer<float> inverse_masses;
    PMBuffer<float> characteristic_lengths;
    uint count;
    uint capacity;
    float velocity_damping;
    float maximum_speed;
    PMBuffer<uint> time_advances;
};
struct SoftBodyRealtimePredictBatch
{
    SoftBodyRealtimePredictBatchItem items[kSoftRealtimeBatchCapacity];
};

struct SoftBodyRealtimeLinksBatchItem
{
    PMBuffer<Vec3> positions;
    PMBuffer<Vec3> position_deltas;
    PMBuffer<float> correction_used;
    PMBuffer<float> inverse_masses;
    PMBuffer<float> characteristic_lengths;
    PMBuffer<uint> offsets;
    PMBuffer<SoftBodyNeighbor> neighbors;
    uint count;
    uint capacity;
    float compliance;
};
struct SoftBodyRealtimeLinksBatch
{
    SoftBodyRealtimeLinksBatchItem items[kSoftRealtimeBatchCapacity];
};

struct SoftBodyRealtimeApplyBatchItem
{
    PMBuffer<Vec3> positions;
    PMBuffer<Vec3> position_deltas;
    PMBuffer<float> correction_used;
    PMBuffer<float> inverse_masses;
    PMBuffer<float> characteristic_lengths;
    uint count;
    uint capacity;
};
struct SoftBodyRealtimeApplyBatch
{
    SoftBodyRealtimeApplyBatchItem items[kSoftRealtimeBatchCapacity];
};

struct SoftBodyRealtimeShapeBatchItem
{
    PMBuffer<Vec3> positions;
    PMBuffer<Vec3> corrections;
    PMBuffer<Vec3> rest_positions;
    PMBuffer<float> inverse_masses;
    PMBuffer<float> characteristic_lengths;
    PMBuffer<float> correction_used;
    uint count;
    uint capacity;
    float movable_mass;
    Vec3 rest_center;
    ShapeMatrix inverse_rest;
    PMBuffer<Quaternion> stored_orientation;
    float stiffness;
};
struct SoftBodyRealtimeShapeBatch
{
    SoftBodyRealtimeShapeBatchItem items[kSoftRealtimeBatchCapacity];
};

struct SoftBodyRealtimeContactsBatchItem
{
    PMBuffer<Vec3> positions;
    PMBuffer<Vec3> previous_positions;
    PMBuffer<float> inverse_masses;
    PMBuffer<float> characteristic_lengths;
    PMBuffer<float> correction_used;
    PMBuffer<Vec3> contact_normals;
    PMBuffer<float> contact_offsets;
    PMByteBuffer contact_active;
    uint count;
    uint capacity;
    float node_radius;
    float friction;
};
struct SoftBodyRealtimeContactsBatch
{
    SoftBodyRealtimeContactsBatchItem items[kSoftRealtimeBatchCapacity];
};

struct SoftBodyRealtimeCommitBatchItem
{
    PMBuffer<Vec3> positions;
    PMBuffer<Vec3> previous_positions;
    PMBuffer<Vec3> velocities;
    PMBuffer<float> inverse_masses;
    PMBuffer<float> characteristic_lengths;
    PMBuffer<Vec3> contact_normals;
    PMBuffer<float> contact_offsets;
    PMByteBuffer contact_active;
    uint count;
    uint capacity;
    float node_radius;
    float friction;
    float maximum_speed;
};
struct SoftBodyRealtimeCommitBatch
{
    SoftBodyRealtimeCommitBatchItem items[kSoftRealtimeBatchCapacity];
};

struct SoftBodySurfaceBatchItem
{
    PMBuffer<Vec3> positions;
    PMBuffer<Vec3> rest_positions;
    PMBuffer<Vec3> surface_rest_positions;
    PMBuffer<SoftBodySurfaceBinding> bindings;
    PMBuffer<Vec3> surface_positions;
    uint surface_vertex_count;
};
struct SoftBodySurfaceBatch
{
    SoftBodySurfaceBatchItem items[kSoftRealtimeBatchCapacity];
};

struct SoftBodyContactFrictionBatchItem
{
    PMBuffer<Vec3> positions;
    PMBuffer<Vec3> velocities;
    PMBuffer<float> inverse_masses;
    PMBuffer<Vec3> contact_forces;
    PMBuffer<Vec3> contact_normals;
    PMBuffer<Vec3> contact_arms;
    PMBuffer<FluidBodyImpulse> body_impulses;
    PMBuffer<Vec3> contact_momentum_delta;
    PMBuffer<Vec3> accumulated_friction_delta;
    PMBuffer<float> accumulated_normal_delta;
    PMBuffer<uint> contact_count;
    uint count;
    float movable_mass;
    float contact_friction;
    float maximum_speed;
};
struct SoftBodyContactFrictionBatch
{
    SoftBodyContactFrictionBatchItem items[kSoftRealtimeBatchCapacity];
};
groupshared Vec3 softValuesA[kSoftReductionThreads];
groupshared Vec3 softValuesB[kSoftReductionThreads];
groupshared float softInverseMasses[kSoftReductionThreads];
groupshared Vec3 softCenter;
groupshared Vec3 softCenterCorrection;
groupshared Quaternion softOrientation;
groupshared uint softRealtimeShapeValid;

// realtime-single-step v1 advances once at 1/60 s and performs exactly eight
// algebraic sweeps. Legacy soft-body entry points below remain profile 0.
static const uint kSoftRealtimeProfileVersion = 1u;
static const float kSoftRealtimeTimestep = 1.0f / 60.0f;
static const float kSoftRealtimeSolverDamping = 0.05f;
static const float kSoftRealtimeMaximumCorrectionFraction = 0.25f;
static const float kSoftRealtimeStructuralCorrectionFraction = 0.20f;
static const uint kSoftRealtimeInvalidProfile = 1u << 17u;
static const uint kSoftRealtimeInvalidTimestep = 1u << 18u;
static const uint kSoftRealtimeInvalidMaterial = 1u << 19u;
static const uint kSoftRealtimeNonfiniteState = 1u << 20u;
static const uint kSoftRealtimeUnresolvedContact = 1u << 21u;
static const uint kSoftRealtimeInverted = 1u << 22u;

bool soft_body_realtime_finite_vec3(Vec3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

bool soft_body_realtime_profile_valid(uint profile_version, float dt)
{
    return profile_version == kSoftRealtimeProfileVersion && isfinite(dt) &&
        abs(dt - kSoftRealtimeTimestep) <= 1.0e-7f;
}

void soft_body_realtime_record_profile_invalid(
    uint profile_version, float dt, PMBuffer<uint> status)
{
    if (profile_version != kSoftRealtimeProfileVersion)
        pmAtomicOr(status, kSoftRealtimeInvalidProfile);
    if (!isfinite(dt) || abs(dt - kSoftRealtimeTimestep) > 1.0e-7f)
        pmAtomicOr(status, kSoftRealtimeInvalidTimestep);
}

bool soft_body_realtime_node_material_valid(
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
        velocity_damping >= 0.0f &&
        soft_body_realtime_finite_vec3(gravity) &&
        vector_length(gravity) * dt * dt <=
            0.25f * characteristic_length;
}

bool soft_body_realtime_compliance_valid(
    float compliance, float effective_inverse_mass, float dt)
{
    if (!isfinite(compliance) || compliance < 0.0f ||
        !isfinite(effective_inverse_mass) || effective_inverse_mass <= 0.0f)
        return false;
    if (compliance == 0.0f) return true;
    float ratio = compliance / (dt * dt * effective_inverse_mass);
    return ratio >= 0.005025125628140708f && ratio <= 19.0f;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_predict(
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
    uint node = pmThreadIndex();
    if (!soft_body_realtime_profile_valid(profile_version, dt))
    {
        if (node == 0u)
            soft_body_realtime_record_profile_invalid(
                profile_version, dt, status);
        return;
    }
    if (node == 0u) ++time_advances[0];
    if (node >= capacity || node >= count) return;
    float inverseMass = inverse_masses[node];
    float characteristicLength = characteristic_lengths[node];
    if (!soft_body_realtime_node_material_valid(inverseMass,
        characteristicLength, maximum_speed, velocity_damping, gravity, dt))
    {
        pmAtomicOr(status, kSoftRealtimeInvalidMaterial);
        return;
    }
    Vec3 position = positions[node];
    previous_positions[node] = position;
    correction_used[node] = 0.0f;
    if (inverseMass == 0.0f)
    {
        velocities[node] = Vec3(0.0f);
        return;
    }
    Vec3 velocity = add(velocities[node], multiply(gravity, dt));
    velocity = multiply(velocity, exp(-velocity_damping * dt));
    velocity = clamp_length(velocity, min(maximum_speed,
        0.5f * characteristicLength / dt));
    position = add(position, multiply(velocity, dt));
    if (!soft_body_realtime_finite_vec3(position) ||
        !soft_body_realtime_finite_vec3(velocity) ||
        abs(position.x) > 10000.0f || abs(position.y) > 10000.0f ||
        abs(position.z) > 10000.0f)
    {
        pmAtomicOr(status, kSoftRealtimeNonfiniteState);
        return;
    }
    positions[node] = position;
    velocities[node] = velocity;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_project_links(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform PMBuffer<uint> offsets,
    uniform PMBuffer<SoftBodyNeighbor> neighbors,
    uniform uint count, uniform uint capacity, uniform float compliance,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (!soft_body_realtime_profile_valid(profile_version, dt))
    {
        if (node == 0u)
            soft_body_realtime_record_profile_invalid(
                profile_version, dt, status);
        return;
    }
    if (node >= capacity || node >= count) return;
    position_deltas[node] = Vec3(0.0f);
    float selfMass = inverse_masses[node];
    if (selfMass == 0.0f) return;
    Vec3 position = positions[node];
    Vec3 correction = Vec3(0.0f);
    uint activeCount = 0u;
    for (uint edge = offsets[node]; edge < offsets[node + 1u]; ++edge)
    {
        SoftBodyNeighbor neighbor = neighbors[edge];
        if (neighbor.index >= count)
        {
            pmAtomicOr(status, kSoftRealtimeInvalidMaterial);
            continue;
        }
        float otherMass = inverse_masses[neighbor.index];
        float effectiveInverseMass = selfMass + otherMass;
        float massRatio = otherMass > 0.0f
            ? max(selfMass, otherMass) / min(selfMass, otherMass) : 1.0f;
        if (!isfinite(neighbor.rest_length) || neighbor.rest_length < 0.005f ||
            neighbor.rest_length > 2.0f || massRatio > 100.0f ||
            !soft_body_realtime_compliance_valid(
                compliance, effectiveInverseMass, dt))
        {
            pmAtomicOr(status, kSoftRealtimeInvalidMaterial);
            continue;
        }
        Vec3 difference = subtract(position, positions[neighbor.index]);
        float edgeLength = vector_length(difference);
        if (!isfinite(edgeLength) || edgeLength < 1.0e-7f)
        {
            if (!isfinite(edgeLength))
                pmAtomicOr(status, kSoftRealtimeNonfiniteState);
            continue;
        }
        float denominator = effectiveInverseMass + compliance / (dt * dt);
        float amount = -selfMass * (edgeLength - neighbor.rest_length) /
            (denominator * edgeLength);
        correction = add(correction, multiply(difference, amount));
        ++activeCount;
    }
    if (activeCount != 0u)
        correction = multiply(correction, 1.0f / float(activeCount));
    float remaining = max(0.0f,
        kSoftRealtimeStructuralCorrectionFraction *
            characteristic_lengths[node] - correction_used[node]);
    correction = clamp_length(correction, remaining);
    if (!soft_body_realtime_finite_vec3(correction))
    {
        pmAtomicOr(status, kSoftRealtimeNonfiniteState);
        correction = Vec3(0.0f);
    }
    position_deltas[node] = correction;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_apply_position_delta(
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
    uint node = pmThreadIndex();
    if (!soft_body_realtime_profile_valid(profile_version, dt))
    {
        if (node == 0u)
            soft_body_realtime_record_profile_invalid(
                profile_version, dt, status);
        return;
    }
    if (node >= capacity || node >= count ||
        inverse_masses[node] == 0.0f) return;
    Vec3 delta = position_deltas[node];
    float used = correction_used[node] + vector_length(delta);
    float maximumCorrection = kSoftRealtimeMaximumCorrectionFraction *
        characteristic_lengths[node];
    Vec3 position = add(positions[node], delta);
    if (!soft_body_realtime_finite_vec3(position) || !isfinite(used) ||
        used > maximumCorrection + 1.0e-6f)
    {
        pmAtomicOr(status, kSoftRealtimeNonfiniteState);
        return;
    }
    positions[node] = position;
    correction_used[node] = min(used, maximumCorrection);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void deformable_predict(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> previous,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> inverse_masses,
    uniform uint count, uniform Vec3 gravity, uniform float dt,
    uniform float damping PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (vertex >= count) return;
    Vec3 old = positions[vertex];
    previous[vertex] = old;
    if (inverse_masses[vertex] == 0.0f)
    {
        velocities[vertex] = {};
        return;
    }
    Vec3 velocity = multiply(
        velocities[vertex], 1.0f / (1.0f + damping * dt));
    velocity = add(velocity, multiply(gravity, dt));
    positions[vertex] = add(old, multiply(velocity, dt));
    velocities[vertex] = velocity;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_measure_momentum(
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> inverse_masses,
    uniform uint count, uniform PMBuffer<Vec3> output PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint thread = pmGroupThreadIndex().x;
    Vec3 local = {};
    for (uint node = thread; node < count; node += pmGroupSize().x)
    {
        float inverseMass = inverse_masses[node];
        if (inverseMass > 0.0f)
            local = add(local, multiply(velocities[node], 1.0f / inverseMass));
    }
    softValuesA[thread] = local;
    pmGroupBarrier();
    for (uint stride = pmGroupSize().x / 2u; stride != 0u; stride /= 2u)
    {
        if (thread < stride)
            softValuesA[thread] = add(
                softValuesA[thread], softValuesA[thread + stride]);
        pmGroupBarrier();
    }
    if (thread == 0u) output[0] = softValuesA[0];
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_restore_momentum(
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<Vec3> contact_momentum_delta, uniform uint count,
    uniform float movable_mass, uniform PMBuffer<Vec3> predicted_momentum,
    uniform PMBuffer<uint> dynamic_contact_flag, uniform float maximum_speed PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (*dynamic_contact_flag == 0u) return;
    uint thread = pmGroupThreadIndex().x;
    Vec3 actual = {};
    Vec3 contact = {};
    for (uint node = thread; node < count; node += pmGroupSize().x)
    {
        float inverseMass = inverse_masses[node];
        if (inverseMass <= 0.0f) continue;
        actual = add(actual, multiply(velocities[node], 1.0f / inverseMass));
        contact = add(contact, contact_momentum_delta[node]);
    }
    softValuesA[thread] = actual;
    softValuesB[thread] = contact;
    pmGroupBarrier();
    for (uint stride = pmGroupSize().x / 2u; stride != 0u; stride /= 2u)
    {
        if (thread < stride)
        {
            softValuesA[thread] = add(
                softValuesA[thread], softValuesA[thread + stride]);
            softValuesB[thread] = add(
                softValuesB[thread], softValuesB[thread + stride]);
        }
        pmGroupBarrier();
    }
    if (thread == 0u)
    {
        Vec3 target = add(*predicted_momentum, softValuesB[0]);
        softCenterCorrection = multiply(
            subtract(target, softValuesA[0]), 1.0f / movable_mass);
    }
    pmGroupBarrier();
    for (uint node = thread; node < count; node += pmGroupSize().x)
        if (inverse_masses[node] > 0.0f)
            velocities[node] = clamp_length(
                add(velocities[node], softCenterCorrection), maximum_speed);
}

ShapeMatrix shape_matrix_multiply(
    ShapeMatrix first, ShapeMatrix second)
{
    ShapeMatrix result = {};
    for (uint column = 0u; column < 3u; ++column)
    {
        Vec3 weights = second.columns[column];
        result.columns[column] = add(
            multiply(first.columns[0], weights.x),
            add(multiply(first.columns[1], weights.y),
                multiply(first.columns[2], weights.z)));
    }
    return result;
}

float shape_matrix_determinant(ShapeMatrix value)
{
    return dot(value.columns[0],
        cross(value.columns[1], value.columns[2]));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_project_shape(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> corrections,
    uniform PMBuffer<Vec3> rest_positions,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform PMBuffer<float> correction_used,
    uniform uint count, uniform uint capacity, uniform float movable_mass,
    uniform Vec3 rest_center, uniform ShapeMatrix inverse_rest,
    uniform PMBuffer<Quaternion> stored_orientation, uniform float stiffness,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u) return;
    uint thread = pmGroupThreadIndex().x;
    if (!soft_body_realtime_profile_valid(profile_version, dt))
    {
        if (thread == 0u)
            soft_body_realtime_record_profile_invalid(
                profile_version, dt, status);
        return;
    }
    if (count > capacity || count == 0u || !isfinite(movable_mass) ||
        movable_mass <= 0.0f || !isfinite(stiffness) ||
        (stiffness != 0.0f && (stiffness < 0.05f || stiffness > 0.995f)) ||
        !soft_body_realtime_finite_vec3(rest_center))
    {
        if (thread == 0u) pmAtomicOr(status, kSoftRealtimeInvalidMaterial);
        return;
    }
    if (stiffness == 0.0f) return;

    Vec3 currentCenter = Vec3(0.0f);
    for (uint base = 0u; base < count; base += pmGroupSize().x)
    {
        uint node = base + thread;
        if (node < count)
        {
            softValuesA[thread] = positions[node];
            softInverseMasses[thread] = inverse_masses[node];
        }
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint valid = min(pmGroupSize().x, count - base);
            for (uint item = 0u; item < valid; ++item)
            {
                float inverseMass = softInverseMasses[item];
                if (inverseMass > 0.0f)
                    currentCenter = add(currentCenter,
                        multiply(softValuesA[item], 1.0f / inverseMass));
            }
        }
        pmGroupBarrier();
    }
    if (thread == 0u)
        softCenter = multiply(currentCenter, 1.0f / movable_mass);
    pmGroupBarrier();

    ShapeMatrix covariance = {};
    for (uint base = 0u; base < count; base += pmGroupSize().x)
    {
        uint node = base + thread;
        if (node < count)
        {
            softValuesA[thread] = positions[node];
            softValuesB[thread] = rest_positions[node];
            softInverseMasses[thread] = inverse_masses[node];
        }
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint valid = min(pmGroupSize().x, count - base);
            for (uint item = 0u; item < valid; ++item)
            {
                float inverseMass = softInverseMasses[item];
                if (inverseMass <= 0.0f) continue;
                float mass = 1.0f / inverseMass;
                Vec3 current = subtract(softValuesA[item], softCenter);
                Vec3 rest = subtract(softValuesB[item], rest_center);
                covariance.columns[0] = add(covariance.columns[0],
                    multiply(current, mass * rest.x));
                covariance.columns[1] = add(covariance.columns[1],
                    multiply(current, mass * rest.y));
                covariance.columns[2] = add(covariance.columns[2],
                    multiply(current, mass * rest.z));
            }
        }
        pmGroupBarrier();
    }
    if (thread == 0u)
    {
        ShapeMatrix deformation = shape_matrix_multiply(
            covariance, inverse_rest);
        float determinant = shape_matrix_determinant(deformation);
        softRealtimeShapeValid = isfinite(determinant) &&
            determinant > 1.0e-4f ? 1u : 0u;
        if (softRealtimeShapeValid == 0u)
        {
            pmAtomicOr(status, isfinite(determinant)
                ? kSoftRealtimeInverted : kSoftRealtimeNonfiniteState);
        }
        else
        {
            Quaternion orientation = normalized_quaternion(
                stored_orientation[0]);
            if (orientation.x == 0.0f && orientation.y == 0.0f &&
                orientation.z == 0.0f && orientation.w == 0.0f)
                orientation.w = 1.0f;
            for (uint iteration = 0u; iteration < 12u; ++iteration)
            {
                Vec3 axes[3];
                axes[0] = rotate(orientation, {1.0f, 0.0f, 0.0f});
                axes[1] = rotate(orientation, {0.0f, 1.0f, 0.0f});
                axes[2] = rotate(orientation, {0.0f, 0.0f, 1.0f});
                Vec3 angular = add(cross(axes[0], deformation.columns[0]),
                    add(cross(axes[1], deformation.columns[1]),
                        cross(axes[2], deformation.columns[2])));
                float denominator = abs(
                    dot(axes[0], deformation.columns[0]) +
                    dot(axes[1], deformation.columns[1]) +
                    dot(axes[2], deformation.columns[2])) + 1.0e-9f;
                angular = multiply(angular, 1.0f / denominator);
                float magnitude = vector_length(angular);
                if (magnitude < 1.0e-6f) break;
                float angle = min(magnitude, 0.5f);
                float half = 0.5f * angle;
                Vec3 axis = multiply(angular, 1.0f / magnitude);
                Quaternion delta = {axis.x * sin(half), axis.y * sin(half),
                    axis.z * sin(half), cos(half)};
                orientation = normalized_quaternion(
                    quaternion_multiply(delta, orientation));
            }
            stored_orientation[0] = orientation;
            softOrientation = orientation;
        }
    }
    pmGroupBarrier();
    if (softRealtimeShapeValid == 0u) return;

    for (uint node = thread; node < count; node += pmGroupSize().x)
    {
        if (inverse_masses[node] <= 0.0f)
        {
            corrections[node] = Vec3(0.0f);
            continue;
        }
        Vec3 target = add(softCenter, rotate(softOrientation,
            subtract(rest_positions[node], rest_center)));
        float remaining = max(0.0f,
            kSoftRealtimeStructuralCorrectionFraction *
                characteristic_lengths[node] - correction_used[node]);
        corrections[node] = clamp_length(multiply(
            subtract(target, positions[node]), stiffness), remaining);
    }
    pmGroupBarrier();
    Vec3 weightedCorrection = Vec3(0.0f);
    for (uint base = 0u; base < count; base += pmGroupSize().x)
    {
        uint node = base + thread;
        if (node < count)
        {
            softValuesA[thread] = corrections[node];
            softInverseMasses[thread] = inverse_masses[node];
        }
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint valid = min(pmGroupSize().x, count - base);
            for (uint item = 0u; item < valid; ++item)
            {
                float inverseMass = softInverseMasses[item];
                if (inverseMass > 0.0f)
                    weightedCorrection = add(weightedCorrection,
                        multiply(softValuesA[item], 1.0f / inverseMass));
            }
        }
        pmGroupBarrier();
    }
    if (thread == 0u)
        softCenterCorrection = multiply(
            weightedCorrection, 1.0f / movable_mass);
    pmGroupBarrier();
    for (uint node = thread; node < count; node += pmGroupSize().x)
    {
        if (inverse_masses[node] <= 0.0f) continue;
        float maximumCorrection = kSoftRealtimeStructuralCorrectionFraction *
            characteristic_lengths[node];
        float remaining = max(0.0f,
            maximumCorrection - correction_used[node]);
        Vec3 correction = clamp_length(subtract(
            corrections[node], softCenterCorrection), remaining);
        Vec3 position = add(positions[node], correction);
        if (!soft_body_realtime_finite_vec3(position) ||
            !soft_body_realtime_finite_vec3(correction))
        {
            pmAtomicOr(status, kSoftRealtimeNonfiniteState);
            continue;
        }
        positions[node] = position;
        correction_used[node] = min(maximumCorrection,
            correction_used[node] + vector_length(correction));
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_project_contacts(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<Vec3> contact_normals,
    uniform PMBuffer<float> contact_offsets,
    uniform PMByteBuffer contact_active,
    uniform uint count, uniform uint capacity, uniform float node_radius,
    uniform float friction, uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (!soft_body_realtime_profile_valid(profile_version, dt))
    {
        if (node == 0u)
            soft_body_realtime_record_profile_invalid(
                profile_version, dt, status);
        return;
    }
    if (node >= capacity || node >= count ||
        inverse_masses[node] == 0.0f || contact_active[node] == 0u)
        return;
    Vec3 normal = contact_normals[node];
    float normalLength = vector_length(normal);
    if (!soft_body_realtime_finite_vec3(normal) ||
        normalLength < 1.0e-6f || !isfinite(contact_offsets[node]) ||
        !isfinite(node_radius) || node_radius < 0.0f ||
        node_radius > 0.5f * characteristic_lengths[node] ||
        !isfinite(friction) || friction < 0.0f || friction > 2.0f)
    {
        pmAtomicOr(status, kSoftRealtimeInvalidMaterial);
        return;
    }
    normal = multiply(normal, 1.0f / normalLength);
    Vec3 position = positions[node];
    float target = contact_offsets[node] + node_radius;
    float penetration = target - dot(position, normal);
    if (penetration <= 0.0f) return;
    Vec3 motion = subtract(position, previous_positions[node]);
    Vec3 tangent = subtract(motion, multiply(normal, dot(motion, normal)));
    float tangentLength = vector_length(tangent);
    Vec3 frictionCorrection = tangentLength > 1.0e-7f
        ? multiply(tangent,
            -min(tangentLength, friction * penetration) / tangentLength)
        : Vec3(0.0f);
    Vec3 correction = add(multiply(normal, penetration), frictionCorrection);
    float maximumCorrection = kSoftRealtimeMaximumCorrectionFraction *
        characteristic_lengths[node];
    float remaining = max(0.0f,
        maximumCorrection - correction_used[node]);
    correction = clamp_length(correction, remaining);
    position = add(position, correction);
    if (!soft_body_realtime_finite_vec3(position) ||
        !soft_body_realtime_finite_vec3(correction))
    {
        pmAtomicOr(status, kSoftRealtimeNonfiniteState);
        return;
    }
    if (dot(position, normal) < target - 1.0e-6f)
        pmAtomicOr(status, kSoftRealtimeUnresolvedContact);
    positions[node] = position;
    correction_used[node] += vector_length(correction);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_commit(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> characteristic_lengths,
    uniform PMBuffer<Vec3> contact_normals,
    uniform PMBuffer<float> contact_offsets,
    uniform PMByteBuffer contact_active,
    uniform uint count, uniform uint capacity, uniform float node_radius,
    uniform float friction, uniform float maximum_speed,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (!soft_body_realtime_profile_valid(profile_version, dt))
    {
        if (node == 0u)
            soft_body_realtime_record_profile_invalid(
                profile_version, dt, status);
        return;
    }
    if (node >= capacity || node >= count) return;
    if (inverse_masses[node] == 0.0f)
    {
        velocities[node] = Vec3(0.0f);
        return;
    }
    Vec3 velocity = multiply(subtract(
        positions[node], previous_positions[node]),
        (1.0f - kSoftRealtimeSolverDamping) / dt);
    if (contact_active[node] != 0u)
    {
        Vec3 normal = normalized_or(contact_normals[node], Vec3(0.0f));
        float target = contact_offsets[node] + node_radius;
        if (vector_length(normal) > 0.0f &&
            dot(positions[node], normal) <=
                target + 1.0e-5f * characteristic_lengths[node])
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
    if (!soft_body_realtime_finite_vec3(positions[node]) ||
        !soft_body_realtime_finite_vec3(velocity))
    {
        pmAtomicOr(status, kSoftRealtimeNonfiniteState);
        positions[node] = previous_positions[node];
        velocities[node] = Vec3(0.0f);
        return;
    }
    velocities[node] = velocity;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_predict_batch(
    uniform SoftBodyRealtimePredictBatch batch, uniform uint batch_count,
    uniform Vec3 gravity, uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint resource = pmBlockIndex().y;
    if (resource >= batch_count) return;
    SoftBodyRealtimePredictBatchItem item = batch.items[resource];
    soft_body_realtime_predict(item.positions, item.velocities,
        item.previous_positions, item.correction_used, item.inverse_masses,
        item.characteristic_lengths, item.count, item.capacity, gravity,
        item.velocity_damping, item.maximum_speed, dt, profile_version,
        item.time_advances, status PM_D3D12_DISPATCH_ARGUMENT);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_project_links_batch(
    uniform SoftBodyRealtimeLinksBatch batch, uniform uint batch_count,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint resource = pmBlockIndex().y;
    if (resource >= batch_count) return;
    SoftBodyRealtimeLinksBatchItem item = batch.items[resource];
    soft_body_realtime_project_links(item.positions, item.position_deltas,
        item.correction_used, item.inverse_masses,
        item.characteristic_lengths, item.offsets, item.neighbors, item.count,
        item.capacity, item.compliance, dt, profile_version,
        status PM_D3D12_DISPATCH_ARGUMENT);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_apply_position_delta_batch(
    uniform SoftBodyRealtimeApplyBatch batch, uniform uint batch_count,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint resource = pmBlockIndex().y;
    if (resource >= batch_count) return;
    SoftBodyRealtimeApplyBatchItem item = batch.items[resource];
    soft_body_realtime_apply_position_delta(item.positions,
        item.position_deltas, item.correction_used, item.inverse_masses,
        item.characteristic_lengths, item.count, item.capacity, dt,
        profile_version, status PM_D3D12_DISPATCH_ARGUMENT);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_project_shape_batch(
    uniform SoftBodyRealtimeShapeBatch batch, uniform uint batch_count,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint resource = pmBlockIndex().y;
    if (resource >= batch_count) return;
    SoftBodyRealtimeShapeBatchItem item = batch.items[resource];
    soft_body_realtime_project_shape(item.positions, item.corrections,
        item.rest_positions, item.inverse_masses,
        item.characteristic_lengths, item.correction_used, item.count,
        item.capacity, item.movable_mass, item.rest_center,
        item.inverse_rest, item.stored_orientation, item.stiffness, dt,
        profile_version, status PM_D3D12_DISPATCH_ARGUMENT);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_project_contacts_batch(
    uniform SoftBodyRealtimeContactsBatch batch, uniform uint batch_count,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint resource = pmBlockIndex().y;
    if (resource >= batch_count) return;
    SoftBodyRealtimeContactsBatchItem item = batch.items[resource];
    soft_body_realtime_project_contacts(item.positions,
        item.previous_positions, item.inverse_masses,
        item.characteristic_lengths, item.correction_used,
        item.contact_normals, item.contact_offsets, item.contact_active,
        item.count, item.capacity, item.node_radius, item.friction, dt,
        profile_version, status PM_D3D12_DISPATCH_ARGUMENT);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_realtime_commit_batch(
    uniform SoftBodyRealtimeCommitBatch batch, uniform uint batch_count,
    uniform float dt, uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint resource = pmBlockIndex().y;
    if (resource >= batch_count) return;
    SoftBodyRealtimeCommitBatchItem item = batch.items[resource];
    soft_body_realtime_commit(item.positions, item.previous_positions,
        item.velocities, item.inverse_masses, item.characteristic_lengths,
        item.contact_normals, item.contact_offsets, item.contact_active,
        item.count, item.capacity, item.node_radius, item.friction,
        item.maximum_speed, dt, profile_version,
        status PM_D3D12_DISPATCH_ARGUMENT);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_project_rest_shape(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> corrections,
    uniform PMBuffer<Vec3> rest_positions, uniform PMBuffer<float> inverse_masses,
    uniform uint count, uniform float movable_mass,
    uniform Vec3 rest_center, uniform ShapeMatrix inverse_rest,
    uniform PMBuffer<Quaternion> stored_orientation, uniform float stiffness,
    uniform float maximum_projection,
    uniform PMBuffer<uint> dynamic_contact_flag PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u || stiffness <= 0.0f ||
        *dynamic_contact_flag != 0u) return;
    uint thread = pmGroupThreadIndex().x;
    Vec3 currentCenter = {};
    for (uint base = 0u; base < count; base += pmGroupSize().x)
    {
        uint node = base + thread;
        if (node < count)
        {
            softValuesA[thread] = positions[node];
            softInverseMasses[thread] = inverse_masses[node];
        }
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint valid = min(pmGroupSize().x, count - base);
            for (uint item = 0u; item < valid; ++item)
            {
                float inverseMass = softInverseMasses[item];
                if (inverseMass > 0.0f)
                    currentCenter = add(currentCenter,
                        multiply(softValuesA[item], 1.0f / inverseMass));
            }
        }
        pmGroupBarrier();
    }
    if (thread == 0u)
    {
        currentCenter = multiply(currentCenter, 1.0f / movable_mass);
        softCenter = currentCenter;
    }
    pmGroupBarrier();

    ShapeMatrix covariance = {};
    for (uint base = 0u; base < count; base += pmGroupSize().x)
    {
        uint node = base + thread;
        if (node < count)
        {
            softValuesA[thread] = positions[node];
            softValuesB[thread] = rest_positions[node];
            softInverseMasses[thread] = inverse_masses[node];
        }
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint valid = min(pmGroupSize().x, count - base);
            for (uint item = 0u; item < valid; ++item)
            {
                float inverseMass = softInverseMasses[item];
                if (inverseMass <= 0.0f) continue;
                float mass = 1.0f / inverseMass;
                Vec3 current = subtract(softValuesA[item], currentCenter);
                Vec3 rest = subtract(softValuesB[item], rest_center);
                covariance.columns[0] = add(covariance.columns[0],
                    multiply(current, mass * rest.x));
                covariance.columns[1] = add(covariance.columns[1],
                    multiply(current, mass * rest.y));
                covariance.columns[2] = add(covariance.columns[2],
                    multiply(current, mass * rest.z));
            }
        }
        pmGroupBarrier();
    }
    if (thread == 0u)
    {
        ShapeMatrix deformation = shape_matrix_multiply(covariance, inverse_rest);
        Quaternion orientation = normalized_quaternion(*stored_orientation);
        if (orientation.x == 0.0f && orientation.y == 0.0f &&
            orientation.z == 0.0f && orientation.w == 0.0f)
            orientation.w = 1.0f;
        for (uint iteration = 0u; iteration < 12u; ++iteration)
        {
            Vec3 axes[3];
            axes[0] = rotate(orientation, {1.0f, 0.0f, 0.0f});
            axes[1] = rotate(orientation, {0.0f, 1.0f, 0.0f});
            axes[2] = rotate(orientation, {0.0f, 0.0f, 1.0f});
            Vec3 angular = add(cross(axes[0], deformation.columns[0]),
                add(cross(axes[1], deformation.columns[1]),
                    cross(axes[2], deformation.columns[2])));
            float denominator = abs(
                dot(axes[0], deformation.columns[0]) +
                dot(axes[1], deformation.columns[1]) +
                dot(axes[2], deformation.columns[2])) + 1.0e-9f;
            angular = multiply(angular, 1.0f / denominator);
            float magnitude = vector_length(angular);
            if (magnitude < 1.0e-6f) break;
            float angle = min(magnitude, 0.5f);
            float half = 0.5f * angle;
            Vec3 axis = multiply(angular, 1.0f / magnitude);
            Quaternion delta = {
                axis.x * sin(half), axis.y * sin(half),
                axis.z * sin(half), cos(half)};
            orientation = normalized_quaternion(
                quaternion_multiply(delta, orientation));
        }
        stored_orientation[0] = orientation;
        softOrientation = orientation;
    }
    pmGroupBarrier();
    for (uint node = thread; node < count; node += pmGroupSize().x)
    {
        float inverseMass = inverse_masses[node];
        if (inverseMass <= 0.0f) continue;
        Vec3 target = add(softCenter, rotate(softOrientation,
            subtract(rest_positions[node], rest_center)));
        corrections[node] = clamp_length(multiply(
            subtract(target, positions[node]), stiffness), maximum_projection);
    }
    pmGroupBarrier();
    Vec3 weightedCorrection = {};
    for (uint base = 0u; base < count; base += pmGroupSize().x)
    {
        uint node = base + thread;
        if (node < count)
        {
            softValuesA[thread] = corrections[node];
            softInverseMasses[thread] = inverse_masses[node];
        }
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint valid = min(pmGroupSize().x, count - base);
            for (uint item = 0u; item < valid; ++item)
            {
                float inverseMass = softInverseMasses[item];
                if (inverseMass > 0.0f)
                    weightedCorrection = add(weightedCorrection,
                        multiply(softValuesA[item], 1.0f / inverseMass));
            }
        }
        pmGroupBarrier();
    }
    if (thread == 0u)
        softCenterCorrection = multiply(
            weightedCorrection, 1.0f / movable_mass);
    pmGroupBarrier();
    for (uint node = thread; node < count; node += pmGroupSize().x)
        if (inverse_masses[node] > 0.0f)
            positions[node] = add(positions[node],
                subtract(corrections[node], softCenterCorrection));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void deformable_project_links(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> scratch,
    uniform PMBuffer<float> inverse_masses, uniform PMBuffer<uint> offsets,
    uniform PMBuffer<DeformableNeighbor> neighbors, uniform PMByteBuffer bond_active,
    uniform uint count, uniform float dt,
    uniform float maximum_projection_fraction PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (vertex >= count) return;
    float selfMass = inverse_masses[vertex];
    Vec3 position = positions[vertex];
    if (selfMass == 0.0f)
    {
        scratch[vertex] = position;
        return;
    }
    Vec3 correction = {};
    float shortestRestLength = 3.402823466e38f;
    uint first = offsets[vertex];
    uint last = offsets[vertex + 1u];
    for (uint edge = first; edge < last; ++edge)
    {
        DeformableNeighbor neighbor = neighbors[edge];
        if (pmBufferPresent(bond_active) && neighbor.bond != kInvalidDense &&
            bond_active[neighbor.bond] == 0u) continue;
        shortestRestLength = min(shortestRestLength, neighbor.rest_length);
        Vec3 difference = subtract(position, positions[neighbor.index]);
        float edgeLength = vector_length(difference);
        if (edgeLength < 1.0e-7f) continue;
        float denominator = selfMass + inverse_masses[neighbor.index] +
            neighbor.compliance / (dt * dt);
        float amount = -selfMass * (edgeLength - neighbor.rest_length) /
            (denominator * edgeLength);
        correction = add(correction, multiply(difference, amount));
    }
    float divisor = float(max(1u, last - first));
    Vec3 proposal = multiply(correction, 1.0f / divisor);
    if (maximum_projection_fraction > 0.0f &&
        shortestRestLength < 3.402823466e38f)
        proposal = clamp_length(proposal,
            maximum_projection_fraction * shortestRestLength);
    scratch[vertex] = add(position, proposal);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void deformable_project_links_warp(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> scratch,
    uniform PMBuffer<float> inverse_masses, uniform PMBuffer<uint> offsets,
    uniform PMBuffer<SoftBodyNeighbor> neighbors,
    uniform PMBuffer<float> minimum_rest_lengths, uniform uint count,
    uniform float compliance, uniform float dt,
    uniform float maximum_projection_fraction PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    const uint groupSize = kSoftNeighborGroupSize;
    uint lane = pmGroupThreadIndex().x & (groupSize - 1u);
    uint group = pmBlockIndex().x * (pmGroupSize().x / groupSize) +
        pmGroupThreadIndex().x / groupSize;
    if (group >= count) return;
    uint groupMask = (0xffffffffu >> (32u - groupSize)) <<
        (pmGroupThreadIndex().x & (32u - groupSize));
    float selfMass = inverse_masses[group];
    Vec3 position = positions[group];
    if (selfMass == 0.0f)
    {
        if (lane == 0u) scratch[group] = position;
        return;
    }
    uint first = offsets[group];
    uint last = offsets[group + 1u];
    Vec3 correction = {};
    for (uint base = first; base < last; base += groupSize)
    {
        uint edge = base + lane;
        Vec3 edgeCorrection = {};
        if (edge < last)
        {
            SoftBodyNeighbor neighbor = neighbors[edge];
            Vec3 difference = subtract(position, positions[neighbor.index]);
            float edgeLength = vector_length(difference);
            if (edgeLength >= 1.0e-7f)
            {
                float denominator = selfMass + inverse_masses[neighbor.index] +
                    compliance / (dt * dt);
                float amount = -selfMass *
                    (edgeLength - neighbor.rest_length) /
                    (denominator * edgeLength);
                edgeCorrection = multiply(difference, amount);
            }
        }
        softValuesA[pmGroupThreadIndex().x] = edgeCorrection;
        pmSyncWarp(groupMask);
        if (lane == 0u)
        {
            uint valid = min(groupSize, last - base);
            uint batch = pmGroupThreadIndex().x;
            for (uint item = 0u; item < valid; ++item)
                correction = add(correction, softValuesA[batch + item]);
        }
        pmSyncWarp(groupMask);
    }
    if (lane != 0u) return;
    Vec3 proposal = multiply(correction,
        1.0f / float(max(1u, last - first)));
    if (maximum_projection_fraction > 0.0f)
        proposal = clamp_length(proposal,
            maximum_projection_fraction * minimum_rest_lengths[group]);
    scratch[group] = add(position, proposal);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_finalize_velocities(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> previous,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<float> inverse_masses,
    uniform uint count, uniform float inverse_dt,
    uniform float projection_response, uniform float maximum_speed PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (node >= count) return;
    if (inverse_masses[node] == 0.0f)
    {
        velocities[node] = {};
        return;
    }
    Vec3 projectedVelocity = multiply(
        subtract(positions[node], previous[node]), inverse_dt);
    Vec3 velocity = add(velocities[node], multiply(
        subtract(projectedVelocity, velocities[node]), projection_response));
    velocities[node] = clamp_length(velocity, maximum_speed);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_apply_contact_friction(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> inverse_masses, uniform PMBuffer<Vec3> contact_forces,
    uniform PMBuffer<Vec3> contact_normals, uniform PMBuffer<Vec3> contact_arms,
    uniform PMBuffer<FluidBodyImpulse> body_impulses,
    uniform PMBuffer<BodyParameters> body_parameters,
    uniform PMBuffer<Vec3> contact_momentum_delta,
    uniform PMBuffer<Vec3> accumulated_friction_delta,
    uniform PMBuffer<float> accumulated_normal_delta,
    uniform PMBuffer<uint> contact_count, uniform uint count,
    uniform float movable_mass, uniform Vec3 gravity, uniform float dt,
    uniform float contact_friction, uniform float maximum_speed PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (node >= count || inverse_masses[node] == 0.0f ||
        contact_friction <= 0.0f) return;
    uint supportedNodes = *contact_count;
    if (supportedNodes == 0u) return;
    float normalSquared = length_squared(contact_normals[node]);
    if (normalSquared <= 1.0e-10f) return;
    uint bodyIndex = body_impulses[node].body;
    if (bodyIndex == kInvalidDense) return;
    Vec3 normal = multiply(contact_normals[node], rsqrt(normalSquared));
    Vec3 originalVelocity = velocities[node];
    Vec3 velocity = originalVelocity;
    Vec3 force = contact_forces[node];
    float forceSquared = length_squared(force);
    Vec3 tangent = subtract(velocity,
        multiply(normal, dot(velocity, normal)));
    float impulseAcceleration = forceSquared > 1.0e-10f
        ? sqrt(forceSquared) * inverse_masses[node] : 0.0f;
    float supportedAcceleration = abs(dot(gravity, normal)) *
        movable_mass * inverse_masses[node] / float(supportedNodes);
    float normalAcceleration = max(
        impulseAcceleration, supportedAcceleration);
    float constraintDelta = pmBufferPresent(accumulated_normal_delta)
        ? accumulated_normal_delta[node] / dt : 0.0f;
    BodyParameters body = body_parameters[bodyIndex];
    float friction = body.motion == MotionType.dynamic
        ? sqrt(contact_friction * body.friction) : contact_friction;
    float maximumDelta = friction *
        max(normalAcceleration * dt, constraintDelta);
    Vec3 accumulated = accumulated_friction_delta[node];
    accumulated = subtract(accumulated,
        multiply(normal, dot(accumulated, normal)));
    Vec3 nextAccumulated = clamp_length(
        subtract(accumulated, tangent), maximumDelta);
    Vec3 appliedDelta = subtract(nextAccumulated, accumulated);
    velocity = clamp_length(add(velocity, appliedDelta), maximum_speed);
    positions[node] = add(positions[node], multiply(
        subtract(velocity, originalVelocity), dt));
    velocities[node] = velocity;
    accumulated_friction_delta[node] = nextAccumulated;
    float mass = 1.0f / inverse_masses[node];
    Vec3 nodeImpulse = multiply(appliedDelta, mass);
    contact_momentum_delta[node] = add(
        contact_momentum_delta[node], nodeImpulse);
    contact_forces[node] = add(
        contact_forces[node], multiply(nodeImpulse, 1.0f / dt));
    Vec3 reaction = multiply(nodeImpulse, -1.0f);
    body_impulses[node].linear = add(body_impulses[node].linear, reaction);
    body_impulses[node].angular = add(body_impulses[node].angular,
        cross(contact_arms[node], reaction));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_apply_contact_friction_batch(
    uniform SoftBodyContactFrictionBatch batch, uniform uint batch_count,
    uniform PMBuffer<BodyParameters> body_parameters, uniform Vec3 gravity,
    uniform float dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint resource = pmBlockIndex().y;
    if (resource >= batch_count) return;
    SoftBodyContactFrictionBatchItem item = batch.items[resource];
    soft_body_apply_contact_friction(item.positions, item.velocities,
        item.inverse_masses, item.contact_forces, item.contact_normals,
        item.contact_arms, item.body_impulses, body_parameters,
        item.contact_momentum_delta, item.accumulated_friction_delta,
        item.accumulated_normal_delta, item.contact_count, item.count,
        item.movable_mass, gravity, dt, item.contact_friction,
        item.maximum_speed PM_D3D12_DISPATCH_ARGUMENT);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_damp_springs(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> output, uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<uint> offsets, uniform PMBuffer<DeformableNeighbor> neighbors,
    uniform PMByteBuffer bond_active, uniform uint count,
    uniform float damping, uniform float maximum_speed PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (node >= count) return;
    if (inverse_masses[node] == 0.0f)
    {
        output[node] = {};
        return;
    }
    Vec3 correction = {};
    uint activeCount = 0u;
    for (uint item = offsets[node]; item < offsets[node + 1u]; ++item)
    {
        DeformableNeighbor neighbor = neighbors[item];
        if (bond_active[neighbor.bond] == 0u) continue;
        Vec3 axis = normalized_or(
            subtract(positions[neighbor.index], positions[node]), {});
        correction = add(correction, multiply(axis, dot(
            subtract(velocities[neighbor.index], velocities[node]), axis)));
        ++activeCount;
    }
    if (activeCount != 0u)
        correction = multiply(correction, 0.5f * damping / float(activeCount));
    output[node] = clamp_length(
        add(velocities[node], correction), maximum_speed);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_damp_springs_warp(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> output, uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<uint> offsets, uniform PMBuffer<SoftBodyNeighbor> neighbors,
    uniform uint count, uniform float damping, uniform float maximum_speed PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    const uint groupSize = kSoftNeighborGroupSize;
    uint lane = pmGroupThreadIndex().x & (groupSize - 1u);
    uint group = pmBlockIndex().x * (pmGroupSize().x / groupSize) +
        pmGroupThreadIndex().x / groupSize;
    if (group >= count) return;
    uint groupMask = (0xffffffffu >> (32u - groupSize)) <<
        (pmGroupThreadIndex().x & (32u - groupSize));
    if (inverse_masses[group] == 0.0f)
    {
        if (lane == 0u) output[group] = {};
        return;
    }
    uint first = offsets[group];
    uint last = offsets[group + 1u];
    Vec3 position = positions[group];
    Vec3 velocity = velocities[group];
    Vec3 correction = {};
    for (uint base = first; base < last; base += groupSize)
    {
        uint edge = base + lane;
        Vec3 edgeCorrection = {};
        if (edge < last)
        {
            SoftBodyNeighbor neighbor = neighbors[edge];
            Vec3 axis = normalized_or(
                subtract(positions[neighbor.index], position), {});
            edgeCorrection = multiply(axis, dot(
                subtract(velocities[neighbor.index], velocity), axis));
        }
        softValuesA[pmGroupThreadIndex().x] = edgeCorrection;
        pmSyncWarp(groupMask);
        if (lane == 0u)
        {
            uint valid = min(groupSize, last - base);
            uint batch = pmGroupThreadIndex().x;
            for (uint item = 0u; item < valid; ++item)
                correction = add(correction, softValuesA[batch + item]);
        }
        pmSyncWarp(groupMask);
    }
    if (lane != 0u) return;
    uint activeCount = last - first;
    if (activeCount != 0u)
        correction = multiply(correction, 0.5f * damping / float(activeCount));
    output[group] = clamp_length(add(velocity, correction), maximum_speed);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_update_surface(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> rest_positions,
    uniform PMBuffer<Vec3> surface_rest_positions,
    uniform PMBuffer<SoftBodySurfaceBinding> bindings,
    uniform PMBuffer<Vec3> surface_positions, uniform uint surface_vertex_count PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint vertex = pmThreadIndex();
    if (vertex >= surface_vertex_count) return;
    SoftBodySurfaceBinding binding = bindings[vertex];
    Vec3 delta = {};
    for (uint slot = 0u; slot < 4u; ++slot)
    {
        if (binding.weights[slot] == 0.0f) continue;
        delta = add(delta, multiply(subtract(
            positions[binding.nodes[slot]], rest_positions[binding.nodes[slot]]),
            binding.weights[slot]));
    }
    surface_positions[vertex] = add(surface_rest_positions[vertex], delta);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_update_surface_batch(
    uniform SoftBodySurfaceBatch batch, uniform uint batch_count
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint resource = pmBlockIndex().y;
    if (resource >= batch_count) return;
    SoftBodySurfaceBatchItem item = batch.items[resource];
    soft_body_update_surface(item.positions, item.rest_positions,
        item.surface_rest_positions, item.bindings, item.surface_positions,
        item.surface_vertex_count PM_D3D12_DISPATCH_ARGUMENT);
}
