// SPDX-License-Identifier: MIT
// Shared soft-body and deformable simulation kernels.

#ifdef PM_TARGET_METAL
import metal_common;
#else
import types;
import math;
import device_intrinsics;
#endif

#ifndef PM_TARGET_METAL
static const uint kSoftReductionThreads = 128u;
groupshared Vec3 softValuesA[kSoftReductionThreads];
groupshared Vec3 softValuesB[kSoftReductionThreads];
groupshared float softInverseMasses[kSoftReductionThreads];
groupshared Vec3 softCenter;
groupshared Vec3 softCenterCorrection;
groupshared Quaternion softOrientation;

[shader("compute")]
[numthreads(256, 1, 1)]
void deformable_predict(
    uniform Vec3* positions, uniform Vec3* previous,
    uniform Vec3* velocities, uniform float* inverse_masses,
    uniform uint count, uniform Vec3 gravity, uniform float dt,
    uniform float damping)
{
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
    uniform Vec3* velocities, uniform float* inverse_masses,
    uniform uint count, uniform Vec3* output)
{
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
    if (thread == 0u) *output = softValuesA[0];
}

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_restore_momentum(
    uniform Vec3* velocities, uniform float* inverse_masses,
    uniform Vec3* contact_momentum_delta, uniform uint count,
    uniform float movable_mass, uniform Vec3* predicted_momentum,
    uniform uint* dynamic_contact_flag, uniform float maximum_speed)
{
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

[shader("compute")]
[numthreads(128, 1, 1)]
void soft_body_project_rest_shape(
    uniform Vec3* positions, uniform Vec3* corrections,
    uniform Vec3* rest_positions, uniform float* inverse_masses,
    uniform uint count, uniform float movable_mass,
    uniform Vec3 rest_center, uniform ShapeMatrix inverse_rest,
    uniform Quaternion* stored_orientation, uniform float stiffness,
    uniform float maximum_projection,
    uniform uint* dynamic_contact_flag)
{
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
        *stored_orientation = orientation;
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
[numthreads(256, 1, 1)]
void deformable_project_links(
    uniform Vec3* positions, uniform Vec3* scratch,
    uniform float* inverse_masses, uniform uint* offsets,
    uniform DeformableNeighbor* neighbors, uniform uint8_t* bond_active,
    uniform uint count, uniform float dt,
    uniform float maximum_projection_fraction)
{
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
        if (bond_active != nullptr && neighbor.bond != kInvalidDense &&
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
    uniform Vec3* positions, uniform Vec3* scratch,
    uniform float* inverse_masses, uniform uint* offsets,
    uniform SoftBodyNeighbor* neighbors,
    uniform float* minimum_rest_lengths, uniform uint count,
    uniform float compliance, uniform float dt,
    uniform float maximum_projection_fraction)
{
    const uint groupSize = 16u;
    uint lane = pmGroupThreadIndex().x & (groupSize - 1u);
    uint group = pmBlockIndex().x * (pmGroupSize().x / groupSize) +
        pmGroupThreadIndex().x / groupSize;
    if (group >= count) return;
    uint groupMask = 0xffffu << (pmGroupThreadIndex().x & 16u);
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
[numthreads(256, 1, 1)]
void soft_body_finalize_velocities(
    uniform Vec3* positions, uniform Vec3* previous,
    uniform Vec3* velocities, uniform float* inverse_masses,
    uniform uint count, uniform float inverse_dt,
    uniform float projection_response, uniform float maximum_speed)
{
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
[numthreads(256, 1, 1)]
void soft_body_apply_contact_friction(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* inverse_masses, uniform Vec3* contact_forces,
    uniform Vec3* contact_normals, uniform Vec3* contact_arms,
    uniform FluidBodyImpulse* body_impulses,
    uniform BodyParameters* body_parameters,
    uniform Vec3* contact_momentum_delta,
    uniform Vec3* accumulated_friction_delta,
    uniform float* accumulated_normal_delta,
    uniform uint* contact_count, uniform uint count,
    uniform float movable_mass, uniform Vec3 gravity, uniform float dt,
    uniform float contact_friction, uniform float maximum_speed)
{
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
    float constraintDelta = accumulated_normal_delta != nullptr
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
[numthreads(256, 1, 1)]
void soft_body_damp_springs(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* output, uniform float* inverse_masses,
    uniform uint* offsets, uniform DeformableNeighbor* neighbors,
    uniform uint8_t* bond_active, uniform uint count,
    uniform float damping, uniform float maximum_speed)
{
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
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* output, uniform float* inverse_masses,
    uniform uint* offsets, uniform SoftBodyNeighbor* neighbors,
    uniform uint count, uniform float damping, uniform float maximum_speed)
{
    const uint groupSize = 16u;
    uint lane = pmGroupThreadIndex().x & (groupSize - 1u);
    uint group = pmBlockIndex().x * (pmGroupSize().x / groupSize) +
        pmGroupThreadIndex().x / groupSize;
    if (group >= count) return;
    uint groupMask = 0xffffu << (pmGroupThreadIndex().x & 16u);
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
[numthreads(256, 1, 1)]
void soft_body_update_surface(
    uniform Vec3* positions, uniform Vec3* rest_positions,
    uniform Vec3* surface_rest_positions,
    uniform SoftBodySurfaceBinding* bindings,
    uniform Vec3* surface_positions, uniform uint surface_vertex_count)
{
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
#endif

#ifdef PM_TARGET_METAL
groupshared float metalSoftScalarTerms[128];
groupshared float3 metalSoftCurrentTerms[128];
groupshared float3 metalSoftRestTerms[128];
groupshared float metalSoftMovableMass;
groupshared float3 metalSoftCurrentCenter;
groupshared float3 metalSoftRestCenter;
groupshared float3 metalSoftRestCovariance[3];
groupshared float3 metalSoftCovariance[3];
groupshared MetalQuaternion metalSoftOrientation;
groupshared float3 metalSoftCenterCorrection;
groupshared float metalSoftMaximumProjection;
groupshared uint metalSoftValid;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftPrevious : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftVelocities : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<float>
    metalSoftInverseMasses : register(t3);
[[vk::binding(4, 0)]] StructuredBuffer<MetalBond>
    metalSoftBonds : register(t4);
[[vk::binding(5, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftSurfacePositions : register(u5);
[[vk::binding(6, 0)]] StructuredBuffer<MetalPackedVec3>
    metalSoftSurfaceRest : register(t6);
[[vk::binding(7, 0)]] StructuredBuffer<MetalSurfaceBinding>
    metalSoftSurfaceBindings : register(t7);
[[vk::binding(8, 0)]] ConstantBuffer<MetalDeformableConstants>
    metalSoftConstants : register(b8);
[[vk::binding(9, 0)]] StructuredBuffer<MetalPackedVec3>
    metalSoftRestPositions : register(t9);
[[vk::binding(10, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSoftCorrections : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<MetalQuaternion>
    metalSoftStoredOrientation : register(u11);
[[vk::binding(12, 0)]] StructuredBuffer<MetalSoftContactState>
    metalSoftContactState : register(t12);

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_body_predict(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    metalIntegrateDeformableNode(
        metalSoftPositions, metalSoftPrevious, metalSoftVelocities,
        metalSoftInverseMasses, metalSoftConstants, dispatchThreadId.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_body_project_bonds(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalDeformableConstants constants = metalSoftConstants;
    if (node >= constants.count) return;
    float3 position = metalLoad(metalSoftPositions[node]);
    float selfInverse = metalSoftInverseMasses[node];
    if (selfInverse <= 0.0f)
    {
        metalSoftCorrections[node] = metalSoftPositions[node];
        return;
    }
    float3 correction = float3(0.0f);
    float shortestRestLength = 3.402823466e+38f;
    uint degree = 0u;
    for (uint index = 0u; index < constants.bond_count; ++index)
    {
        MetalBond bond = metalSoftBonds[index];
        uint other = constants.count;
        if (bond.first == node) other = bond.second;
        else if (bond.second == node) other = bond.first;
        if (other >= constants.count) continue;
        ++degree;
        if (bond.active == 0u) continue;
        shortestRestLength = min(shortestRestLength, bond.rest_length);
        float3 difference =
            position - metalLoad(metalSoftPositions[other]);
        float currentLength = length(difference);
        if (currentLength <= 1.0e-7f) continue;
        float denominator = selfInverse + metalSoftInverseMasses[other] +
            bond.compliance /
                max(constants.timestep * constants.timestep, 1.0e-12f);
        if (denominator <= 0.0f) continue;
        correction += difference *
            (-selfInverse * (currentLength - bond.rest_length) /
             (denominator * currentLength));
    }
    float3 proposal = correction / float(max(degree, 1u));
    if (constants.maximum_projection_fraction > 0.0f &&
        shortestRestLength < 3.402823466e+38f)
        proposal = metalLimit(proposal,
            constants.maximum_projection_fraction * shortestRestLength);
    metalSoftCorrections[node] = metalStore(position + proposal);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_body_apply_bonds(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    if (node < metalSoftConstants.count)
        metalSoftPositions[node] = metalSoftCorrections[node];
}

[shader("compute")]
[numthreads(128, 1, 1)]
public void pm_soft_body_shape_matching(uint groupIndex : SV_GroupIndex)
{
    if (metalSoftContactState[0].dynamic_contact_flag != 0u) return;
    uint lane = groupIndex;
    MetalDeformableConstants constants = metalSoftConstants;
    if (lane == 0u)
    {
        metalSoftMovableMass = 0.0f;
        metalSoftCurrentCenter = float3(0.0f);
        metalSoftRestCenter = float3(0.0f);
        metalSoftCenterCorrection = float3(0.0f);
        metalSoftMaximumProjection = 3.402823466e+38f;
    }
    pmGroupBarrier();
    for (uint wave = 0u; wave < constants.count; wave += 128u)
    {
        uint node = wave + lane;
        float mass = 0.0f;
        float3 current = float3(0.0f);
        float3 rest = float3(0.0f);
        if (node < constants.count && metalSoftInverseMasses[node] > 0.0f)
        {
            mass = 1.0f / metalSoftInverseMasses[node];
            current = metalLoad(metalSoftPositions[node]) * mass;
            rest = metalLoad(metalSoftRestPositions[node]) * mass;
        }
        metalSoftScalarTerms[lane] = mass;
        metalSoftCurrentTerms[lane] = current;
        metalSoftRestTerms[lane] = rest;
        pmGroupBarrier();
        if (lane == 0u)
        {
            uint count = min(128u, constants.count - wave);
            for (uint item = 0u; item < count; ++item)
            {
                metalSoftMovableMass += metalSoftScalarTerms[item];
                metalSoftCurrentCenter += metalSoftCurrentTerms[item];
                metalSoftRestCenter += metalSoftRestTerms[item];
            }
        }
        pmGroupBarrier();
    }
    if (lane == 0u)
    {
        metalSoftValid = metalSoftMovableMass > 0.0f ? 1u : 0u;
        if (metalSoftValid != 0u)
        {
            metalSoftCurrentCenter /= metalSoftMovableMass;
            metalSoftRestCenter /= metalSoftMovableMass;
        }
        for (uint column = 0u; column < 3u; ++column)
        {
            metalSoftRestCovariance[column] = float3(0.0f);
            metalSoftCovariance[column] = float3(0.0f);
        }
    }
    pmGroupBarrier();

    for (uint column = 0u; column < 3u; ++column)
    {
        for (uint wave = 0u; wave < constants.count; wave += 128u)
        {
            uint node = wave + lane;
            float3 restTerm = float3(0.0f);
            float3 currentTerm = float3(0.0f);
            if (metalSoftValid != 0u && node < constants.count &&
                metalSoftInverseMasses[node] > 0.0f)
            {
                float mass = 1.0f / metalSoftInverseMasses[node];
                float3 rest = metalLoad(metalSoftRestPositions[node]) -
                    metalSoftRestCenter;
                float3 current = metalLoad(metalSoftPositions[node]) -
                    metalSoftCurrentCenter;
                restTerm = rest * (mass * rest[column]);
                currentTerm = current * (mass * rest[column]);
            }
            metalSoftRestTerms[lane] = restTerm;
            metalSoftCurrentTerms[lane] = currentTerm;
            pmGroupBarrier();
            if (lane == 0u)
            {
                uint count = min(128u, constants.count - wave);
                for (uint item = 0u; item < count; ++item)
                {
                    metalSoftRestCovariance[column] +=
                        metalSoftRestTerms[item];
                    metalSoftCovariance[column] +=
                        metalSoftCurrentTerms[item];
                }
            }
            pmGroupBarrier();
        }
    }
    if (lane == 0u && metalSoftValid != 0u)
    {
        float3 inverseRow0 = cross(
            metalSoftRestCovariance[1], metalSoftRestCovariance[2]);
        float3 inverseRow1 = cross(
            metalSoftRestCovariance[2], metalSoftRestCovariance[0]);
        float3 inverseRow2 = cross(
            metalSoftRestCovariance[0], metalSoftRestCovariance[1]);
        float determinant = dot(metalSoftRestCovariance[0], inverseRow0);
        metalSoftValid = abs(determinant) > 1.0e-10f ? 1u : 0u;
        if (metalSoftValid != 0u)
        {
            float inverseDeterminant = 1.0f / determinant;
            float3 inverseRest0 = float3(
                inverseRow0.x, inverseRow1.x, inverseRow2.x) *
                inverseDeterminant;
            float3 inverseRest1 = float3(
                inverseRow0.y, inverseRow1.y, inverseRow2.y) *
                inverseDeterminant;
            float3 inverseRest2 = float3(
                inverseRow0.z, inverseRow1.z, inverseRow2.z) *
                inverseDeterminant;
            float3 deformation0 =
                metalSoftCovariance[0] * inverseRest0.x +
                metalSoftCovariance[1] * inverseRest0.y +
                metalSoftCovariance[2] * inverseRest0.z;
            float3 deformation1 =
                metalSoftCovariance[0] * inverseRest1.x +
                metalSoftCovariance[1] * inverseRest1.y +
                metalSoftCovariance[2] * inverseRest1.z;
            float3 deformation2 =
                metalSoftCovariance[0] * inverseRest2.x +
                metalSoftCovariance[1] * inverseRest2.y +
                metalSoftCovariance[2] * inverseRest2.z;
            MetalQuaternion orientation = metalQuaternionNormalize(
                metalSoftStoredOrientation[0]);
            if (orientation.x == 0.0f && orientation.y == 0.0f &&
                orientation.z == 0.0f && orientation.w == 0.0f)
            {
                orientation.x = 0.0f;
                orientation.y = 0.0f;
                orientation.z = 0.0f;
                orientation.w = 1.0f;
            }
            for (uint iteration = 0u; iteration < 12u; ++iteration)
            {
                float3 axis0 = metalRotate(
                    orientation, float3(1.0f, 0.0f, 0.0f));
                float3 axis1 = metalRotate(
                    orientation, float3(0.0f, 1.0f, 0.0f));
                float3 axis2 = metalRotate(
                    orientation, float3(0.0f, 0.0f, 1.0f));
                float3 angular = cross(axis0, deformation0) +
                    cross(axis1, deformation1) +
                    cross(axis2, deformation2);
                float denominator = abs(dot(axis0, deformation0) +
                    dot(axis1, deformation1) +
                    dot(axis2, deformation2)) + 1.0e-9f;
                angular /= denominator;
                float magnitude = length(angular);
                if (magnitude < 1.0e-6f) break;
                float angle = min(magnitude, 0.5f);
                float halfAngle = 0.5f * angle;
                float3 axis = angular / magnitude;
                MetalQuaternion delta;
                delta.x = axis.x * sin(halfAngle);
                delta.y = axis.y * sin(halfAngle);
                delta.z = axis.z * sin(halfAngle);
                delta.w = cos(halfAngle);
                orientation = metalQuaternionNormalize(
                    metalQuaternionMultiply(delta, orientation));
            }
            metalSoftStoredOrientation[0] = orientation;
            metalSoftOrientation = orientation;
            float shortest = 3.402823466e+38f;
            for (uint bond = 0u; bond < constants.bond_count; ++bond)
                shortest = min(shortest, metalSoftBonds[bond].rest_length);
            metalSoftMaximumProjection =
                shortest * constants.maximum_projection_fraction;
        }
    }
    pmGroupBarrier();

    for (uint wave = 0u; wave < constants.count; wave += 128u)
    {
        uint node = wave + lane;
        float3 weighted = float3(0.0f);
        if (node < constants.count)
        {
            float3 correction = float3(0.0f);
            if (metalSoftValid != 0u &&
                metalSoftInverseMasses[node] > 0.0f)
            {
                float3 target = metalSoftCurrentCenter + metalRotate(
                    metalSoftOrientation,
                    metalLoad(metalSoftRestPositions[node]) -
                        metalSoftRestCenter);
                correction = metalLimit(
                    (target - metalLoad(metalSoftPositions[node])) *
                        constants.shape_matching_stiffness,
                    metalSoftMaximumProjection);
                weighted = correction / metalSoftInverseMasses[node];
            }
            metalSoftCorrections[node] = metalStore(correction);
        }
        metalSoftCurrentTerms[lane] = weighted;
        AllMemoryBarrierWithGroupSync();
        if (lane == 0u)
        {
            uint count = min(128u, constants.count - wave);
            for (uint item = 0u; item < count; ++item)
                metalSoftCenterCorrection += metalSoftCurrentTerms[item];
        }
        pmGroupBarrier();
    }
    if (lane == 0u && metalSoftValid != 0u)
        metalSoftCenterCorrection /= metalSoftMovableMass;
    pmGroupBarrier();
    if (metalSoftValid == 0u) return;
    for (uint node = lane; node < constants.count; node += 128u)
        if (metalSoftInverseMasses[node] > 0.0f)
            metalSoftPositions[node] = metalStore(
                metalLoad(metalSoftPositions[node]) +
                metalLoad(metalSoftCorrections[node]) -
                metalSoftCenterCorrection);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_body_finalize(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    metalFinalizeDeformableNode(
        metalSoftPositions, metalSoftPrevious, metalSoftVelocities,
        metalSoftInverseMasses, metalSoftConstants, dispatchThreadId.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_body_damping_prepare(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalDeformableConstants constants = metalSoftConstants;
    if (node >= constants.count || constants.spring_damping <= 0.0f) return;
    if (metalSoftInverseMasses[node] <= 0.0f)
    {
        metalSoftCorrections[node] = metalStore(float3(0.0f));
        return;
    }
    float3 correction = float3(0.0f);
    uint activeCount = 0u;
    for (uint bondIndex = 0u; bondIndex < constants.bond_count; ++bondIndex)
    {
        MetalBond bond = metalSoftBonds[bondIndex];
        uint other = constants.count;
        if (bond.first == node) other = bond.second;
        if (bond.second == node) other = bond.first;
        if (other >= constants.count || bond.active == 0u) continue;
        float3 difference = metalLoad(metalSoftPositions[other]) -
            metalLoad(metalSoftPositions[node]);
        if (dot(difference, difference) <= 1.0e-12f) continue;
        float3 axis = normalize(difference);
        correction += axis * dot(
            metalLoad(metalSoftVelocities[other]) -
                metalLoad(metalSoftVelocities[node]), axis);
        ++activeCount;
    }
    if (activeCount != 0u)
        correction *= 0.5f * constants.spring_damping / float(activeCount);
    metalSoftCorrections[node] = metalStore(metalLimit(
        metalLoad(metalSoftVelocities[node]) + correction,
        constants.maximum_speed));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_body_damping_apply(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalDeformableConstants constants = metalSoftConstants;
    if (node >= constants.count || constants.spring_damping <= 0.0f) return;
    metalSoftVelocities[node] = metalSoftCorrections[node];
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_soft_body_surface_update(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint surfaceIndex = dispatchThreadId.x;
    MetalDeformableConstants constants = metalSoftConstants;
    if (surfaceIndex >= constants.surface_count) return;
    float3 delta = float3(0.0f);
    MetalSurfaceBinding binding = metalSoftSurfaceBindings[surfaceIndex];
    for (uint index = 0u; index < 4u; ++index)
    {
        uint node = binding.nodes[index];
        if (node < constants.count)
            delta += (metalLoad(metalSoftPositions[node]) -
                metalLoad(metalSoftRestPositions[node])) *
                binding.weights[index];
    }
    metalSoftSurfacePositions[surfaceIndex] = metalStore(
        metalLoad(metalSoftSurfaceRest[surfaceIndex]) + delta);
}
#endif
