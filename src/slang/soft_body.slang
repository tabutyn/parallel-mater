// SPDX-License-Identifier: MIT
// Slang mirror of src/soft_body.cuh.

import cuda_common;

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
    uint thread = cudaThreadIdx().x;
    Vec3 local = {};
    for (uint node = thread; node < count; node += cudaBlockDim().x)
    {
        float inverseMass = inverse_masses[node];
        if (inverseMass > 0.0f)
            local = add(local, multiply(velocities[node], 1.0f / inverseMass));
    }
    softValuesA[thread] = local;
    pmGroupBarrier();
    for (uint stride = cudaBlockDim().x / 2u; stride != 0u; stride /= 2u)
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
    uint thread = cudaThreadIdx().x;
    Vec3 actual = {};
    Vec3 contact = {};
    for (uint node = thread; node < count; node += cudaBlockDim().x)
    {
        float inverseMass = inverse_masses[node];
        if (inverseMass <= 0.0f) continue;
        actual = add(actual, multiply(velocities[node], 1.0f / inverseMass));
        contact = add(contact, contact_momentum_delta[node]);
    }
    softValuesA[thread] = actual;
    softValuesB[thread] = contact;
    pmGroupBarrier();
    for (uint stride = cudaBlockDim().x / 2u; stride != 0u; stride /= 2u)
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
    for (uint node = thread; node < count; node += cudaBlockDim().x)
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
    if (cudaBlockIdx().x != 0u || stiffness <= 0.0f ||
        *dynamic_contact_flag != 0u) return;
    uint thread = cudaThreadIdx().x;
    Vec3 currentCenter = {};
    for (uint base = 0u; base < count; base += cudaBlockDim().x)
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
            uint valid = min(cudaBlockDim().x, count - base);
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
    for (uint base = 0u; base < count; base += cudaBlockDim().x)
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
            uint valid = min(cudaBlockDim().x, count - base);
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
    for (uint node = thread; node < count; node += cudaBlockDim().x)
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
    for (uint base = 0u; base < count; base += cudaBlockDim().x)
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
            uint valid = min(cudaBlockDim().x, count - base);
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
    for (uint node = thread; node < count; node += cudaBlockDim().x)
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
    uint lane = cudaThreadIdx().x & (groupSize - 1u);
    uint group = cudaBlockIdx().x * (cudaBlockDim().x / groupSize) +
        cudaThreadIdx().x / groupSize;
    if (group >= count) return;
    uint groupMask = 0xffffu << (cudaThreadIdx().x & 16u);
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
        softValuesA[cudaThreadIdx().x] = edgeCorrection;
        pmSyncWarp(groupMask);
        if (lane == 0u)
        {
            uint valid = min(groupSize, last - base);
            uint batch = cudaThreadIdx().x;
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
    uint lane = cudaThreadIdx().x & (groupSize - 1u);
    uint group = cudaBlockIdx().x * (cudaBlockDim().x / groupSize) +
        cudaThreadIdx().x / groupSize;
    if (group >= count) return;
    uint groupMask = 0xffffu << (cudaThreadIdx().x & 16u);
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
        softValuesA[cudaThreadIdx().x] = edgeCorrection;
        pmSyncWarp(groupMask);
        if (lane == 0u)
        {
            uint valid = min(groupSize, last - base);
            uint batch = cudaThreadIdx().x;
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
