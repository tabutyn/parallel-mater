#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Shared rope simulation, collision, coupling, and constraint kernels.

import types;
import math;
import device_intrinsics;
import geometry;
import geometry_fluid;
import fluid_soft_body;
import graphics_resources;

// realtime-single-step v1 advances once at 1/60 s and performs exactly eight
// algebraic sweeps. Legacy rope entry points below remain profile 0.
static const uint kRopeRealtimeProfileVersion = 1u;
static const float kRopeRealtimeTimestep = 1.0f / 60.0f;
static const float kRopeRealtimeSolverDamping = 0.05f;
static const float kRopeRealtimeMaximumCorrectionFraction = 0.25f;
static const float kRopeRealtimeStructuralCorrectionFraction = 0.20f;
static const uint kRopeRealtimeInvalidProfile = 1u << 17u;
static const uint kRopeRealtimeInvalidTimestep = 1u << 18u;
static const uint kRopeRealtimeInvalidMaterial = 1u << 19u;
static const uint kRopeRealtimeNonfiniteState = 1u << 20u;
static const uint kRopeRealtimeUnresolvedContact = 1u << 21u;

groupshared float ropeDiagonal[1024];
groupshared float ropeUpper[1024];
groupshared float ropeRhs[1024];
groupshared Vec3 ropeRealtimeDirections[1024];
groupshared Vec3 ropeRealtimeContactPositions[1024];

bool rope_realtime_finite_vec3(Vec3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

bool rope_realtime_profile_valid(uint profile_version, float dt)
{
    return profile_version == kRopeRealtimeProfileVersion && isfinite(dt) &&
        abs(dt - kRopeRealtimeTimestep) <= 1.0e-7f;
}

void rope_realtime_record_profile_invalid(
    uint profile_version, float dt, PMBuffer<uint> status)
{
    if (profile_version != kRopeRealtimeProfileVersion)
        pmAtomicOr(status, kRopeRealtimeInvalidProfile);
    if (!isfinite(dt) || abs(dt - kRopeRealtimeTimestep) > 1.0e-7f)
        pmAtomicOr(status, kRopeRealtimeInvalidTimestep);
}

bool rope_realtime_anchor_active(
    uint node, uint count, PMByteBuffer anchor_active)
{
    return (node == 0u && anchor_active[0] != 0u) ||
        (node + 1u == count && anchor_active[1] != 0u);
}

uint rope_realtime_anchor_index(uint node, uint count)
{
    return node + 1u == count ? 1u : 0u;
}

float rope_realtime_inverse_mass(
    uint node, uint count, PMBuffer<float> inverse_masses,
    PMByteBuffer anchor_active)
{
    return rope_realtime_anchor_active(node, count, anchor_active)
        ? 0.0f : inverse_masses[node];
}

bool rope_realtime_material_valid(
    float inverse_mass, float node_spacing, float radius,
    float maximum_speed, float velocity_damping, float friction,
    Vec3 gravity, float dt)
{
    float mass = inverse_mass > 0.0f ? 1.0f / inverse_mass : 0.001f;
    return isfinite(inverse_mass) && inverse_mass >= 0.0f &&
        isfinite(node_spacing) && node_spacing >= 0.005f &&
        node_spacing <= 2.0f && isfinite(radius) &&
        radius >= 0.5f * node_spacing && radius <= node_spacing &&
        mass >= 0.001f && mass <= 10000.0f &&
        isfinite(maximum_speed) && maximum_speed >= 0.0f &&
        maximum_speed <= 20.0f && maximum_speed * dt <=
            0.5f * node_spacing && isfinite(velocity_damping) &&
        velocity_damping >= 0.0f && isfinite(friction) &&
        friction >= 0.0f && friction <= 2.0f &&
        rope_realtime_finite_vec3(gravity) &&
        vector_length(gravity) * dt * dt <= 0.25f * node_spacing;
}

bool rope_realtime_compliance_valid(
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
void rope_realtime_predict(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<Vec3> anchor_targets,
    uniform PMByteBuffer anchor_active, uniform uint count,
    uniform uint capacity, uniform float node_spacing, uniform float radius,
    uniform Vec3 gravity, uniform float velocity_damping,
    uniform float friction, uniform float maximum_speed, uniform float dt,
    uniform uint profile_version, uniform PMBuffer<uint> time_advances,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (!rope_realtime_profile_valid(profile_version, dt))
    {
        if (node == 0u)
            rope_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (node == 0u)
    {
        ++time_advances[0];
        if (count < 2u || count > 1024u || count > capacity)
            pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
    }
    if (node >= capacity || node >= count) return;
    float inverseMass = inverse_masses[node];
    if (!rope_realtime_material_valid(inverseMass, node_spacing, radius,
        maximum_speed, velocity_damping, friction, gravity, dt))
    {
        pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
        return;
    }
    Vec3 position = positions[node];
    previous_positions[node] = position;
    correction_used[node] = 0.0f;
    if (rope_realtime_anchor_active(node, count, anchor_active))
    {
        Vec3 target = anchor_targets[rope_realtime_anchor_index(node, count)];
        Vec3 motion = subtract(target, position);
        if (!rope_realtime_finite_vec3(target) ||
            vector_length(motion) > 0.5f * node_spacing + 1.0e-6f)
        {
            pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
            return;
        }
        Vec3 velocity = multiply(motion, 1.0f / dt);
        if (vector_length(velocity) > maximum_speed + 1.0e-5f)
        {
            pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
            return;
        }
        positions[node] = target;
        velocities[node] = velocity;
        return;
    }
    if (inverseMass == 0.0f)
    {
        velocities[node] = Vec3(0.0f);
        return;
    }
    Vec3 velocity = add(velocities[node], multiply(gravity, dt));
    velocity = multiply(velocity, exp(-velocity_damping * dt));
    velocity = clamp_length(velocity, min(maximum_speed,
        0.5f * node_spacing / dt));
    position = add(position, multiply(velocity, dt));
    if (!rope_realtime_finite_vec3(position) ||
        !rope_realtime_finite_vec3(velocity) ||
        abs(position.x) > 10000.0f || abs(position.y) > 10000.0f ||
        abs(position.z) > 10000.0f)
    {
        pmAtomicOr(status, kRopeRealtimeNonfiniteState);
        return;
    }
    positions[node] = position;
    velocities[node] = velocity;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rope_realtime_project_constraints(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> rest_lengths,
    uniform PMBuffer<float> rest_bend_lengths,
    uniform PMByteBuffer anchor_active, uniform uint count,
    uniform uint capacity, uniform float node_spacing,
    uniform float structural_compliance, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint lane = pmGroupThreadIndex().x;
    if (pmBlockIndex().x != 0u) return;
    if (!rope_realtime_profile_valid(profile_version, dt))
    {
        if (lane == 0u)
            rope_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (count < 2u || count > 1024u || count > capacity)
    {
        if (lane == 0u) pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
        return;
    }

    uint edges = count - 1u;
    float alpha = structural_compliance / (dt * dt);
    for (uint edge = lane; edge < edges; edge += pmGroupSize().x)
    {
        float firstMass = rope_realtime_inverse_mass(
            edge, count, inverse_masses, anchor_active);
        float secondMass = rope_realtime_inverse_mass(
            edge + 1u, count, inverse_masses, anchor_active);
        float effectiveInverseMass = firstMass + secondMass;
        float restLength = rest_lengths[edge];
        Vec3 difference = subtract(positions[edge + 1u], positions[edge]);
        float currentLength = vector_length(difference);
        bool valid = isfinite(restLength) && restLength >= 0.005f &&
            restLength <= node_spacing * 1.25f &&
            rope_realtime_compliance_valid(structural_compliance,
                effectiveInverseMass, dt) && isfinite(currentLength) &&
            currentLength >= 1.0e-7f;
        if (!valid)
        {
            pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
            ropeRealtimeDirections[edge] = Vec3(1.0f, 0.0f, 0.0f);
            ropeDiagonal[edge] = 1.0f;
            ropeRhs[edge] = 0.0f;
            continue;
        }
        ropeRealtimeDirections[edge] = multiply(
            difference, 1.0f / currentLength);
        ropeDiagonal[edge] = effectiveInverseMass + alpha;
        ropeRhs[edge] = -(currentLength - restLength);
    }
    pmGroupBarrier();
    for (uint edge = lane; edge + 1u < edges; edge += pmGroupSize().x)
    {
        float sharedMass = rope_realtime_inverse_mass(
            edge + 1u, count, inverse_masses, anchor_active);
        ropeUpper[edge] = -sharedMass * dot(
            ropeRealtimeDirections[edge], ropeRealtimeDirections[edge + 1u]);
    }
    pmGroupBarrier();
    if (lane == 0u)
    {
        for (uint edge = 1u; edge < edges; ++edge)
        {
            float factor = ropeUpper[edge - 1u] /
                max(ropeDiagonal[edge - 1u], 1.0e-10f);
            ropeDiagonal[edge] -= factor * ropeUpper[edge - 1u];
            ropeRhs[edge] -= factor * ropeRhs[edge - 1u];
        }
        ropeRhs[edges - 1u] /= max(
            ropeDiagonal[edges - 1u], 1.0e-10f);
        for (int edge = int(edges) - 2; edge >= 0; --edge)
            ropeRhs[edge] = (ropeRhs[edge] -
                ropeUpper[edge] * ropeRhs[edge + 1]) /
                max(ropeDiagonal[edge], 1.0e-10f);
    }
    pmGroupBarrier();

    for (uint node = lane; node < count; node += pmGroupSize().x)
    {
        float selfMass = rope_realtime_inverse_mass(
            node, count, inverse_masses, anchor_active);
        Vec3 correction = Vec3(0.0f);
        if (selfMass != 0.0f)
        {
            if (node != 0u)
                correction = add(correction, multiply(
                    ropeRealtimeDirections[node - 1u],
                    selfMass * ropeRhs[node - 1u]));
            if (node < edges)
                correction = subtract(correction, multiply(
                    ropeRealtimeDirections[node], selfMass * ropeRhs[node]));

            // A two-edge chord supplies bounded bending without changing the
            // public v1 rope material. Half-relaxed Jacobi keeps the two
            // parity chains stable while the direct stretch solve carries
            // tension across the complete rope.
            Vec3 bendCorrection = Vec3(0.0f);
            Vec3 position = positions[node];
            for (uint side = 0u; side < 2u; ++side)
            {
                int otherSigned = int(node) + (side == 0u ? -2 : 2);
                if (otherSigned < 0 || otherSigned >= int(count)) continue;
                uint other = uint(otherSigned);
                uint constraint = min(node, other);
                float restLength = rest_bend_lengths[constraint];
                float otherMass = rope_realtime_inverse_mass(
                    other, count, inverse_masses, anchor_active);
                float effectiveInverseMass = selfMass + otherMass;
                if (!isfinite(restLength) || restLength < 0.005f ||
                    restLength > 2.5f * node_spacing ||
                    !rope_realtime_compliance_valid(structural_compliance,
                        effectiveInverseMass, dt))
                {
                    pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
                    continue;
                }
                Vec3 difference = subtract(position, positions[other]);
                float currentLength = vector_length(difference);
                if (!isfinite(currentLength) || currentLength < 1.0e-7f)
                {
                    pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
                    continue;
                }
                float denominator = effectiveInverseMass + alpha;
                float amount = -selfMass *
                    (currentLength - restLength) /
                    (denominator * currentLength);
                bendCorrection = add(bendCorrection,
                    multiply(difference, amount));
            }
            correction = add(correction,
                multiply(bendCorrection, 0.5f));
        }
        float maximumCorrection = kRopeRealtimeStructuralCorrectionFraction *
            node_spacing;
        float remaining = max(0.0f,
            maximumCorrection - correction_used[node]);
        correction = clamp_length(correction, remaining);
        if (!rope_realtime_finite_vec3(correction))
        {
            pmAtomicOr(status, kRopeRealtimeNonfiniteState);
            correction = Vec3(0.0f);
        }
        position_deltas[node] = correction;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rope_realtime_apply_position_delta(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<float> correction_used,
    uniform PMBuffer<float> inverse_masses,
    uniform PMByteBuffer anchor_active, uniform uint count,
    uniform uint capacity, uniform float node_spacing, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (!rope_realtime_profile_valid(profile_version, dt))
    {
        if (node == 0u)
            rope_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (node >= capacity || node >= count ||
        rope_realtime_inverse_mass(node, count, inverse_masses,
            anchor_active) == 0.0f) return;
    Vec3 delta = position_deltas[node];
    float used = correction_used[node] + vector_length(delta);
    float maximumCorrection = kRopeRealtimeMaximumCorrectionFraction *
        node_spacing;
    Vec3 position = add(positions[node], delta);
    if (!rope_realtime_finite_vec3(position) || !isfinite(used) ||
        used > maximumCorrection + 1.0e-6f)
    {
        pmAtomicOr(status, kRopeRealtimeNonfiniteState);
        return;
    }
    positions[node] = position;
    correction_used[node] = min(used, maximumCorrection);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rope_realtime_project_contacts(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<float> correction_used,
    uniform PMByteBuffer anchor_active,
    uniform PMBuffer<Vec3> contact_normals,
    uniform PMBuffer<float> contact_offsets,
    uniform PMByteBuffer contact_active, uniform uint count,
    uniform uint capacity, uniform float node_spacing, uniform float radius,
    uniform uint self_collision, uniform float friction, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint lane = pmGroupThreadIndex().x;
    if (pmBlockIndex().x != 0u) return;
    if (!rope_realtime_profile_valid(profile_version, dt))
    {
        if (lane == 0u)
            rope_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (count > 1024u || count > capacity || !isfinite(radius) ||
        radius < 0.5f * node_spacing || radius > node_spacing ||
        !isfinite(friction) || friction < 0.0f || friction > 2.0f)
    {
        if (lane == 0u) pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
        return;
    }
    for (uint node = lane; node < count; node += pmGroupSize().x)
        ropeRealtimeContactPositions[node] = positions[node];
    pmGroupBarrier();

    for (uint node = lane; node < count; node += pmGroupSize().x)
    {
        float inverseMass = rope_realtime_inverse_mass(
            node, count, inverse_masses, anchor_active);
        if (inverseMass == 0.0f) continue;
        Vec3 position = ropeRealtimeContactPositions[node];
        Vec3 correction = Vec3(0.0f);
        if (self_collision != 0u)
        {
            float targetSeparation = 2.0f * radius;
            for (uint other = 0u; other < count; ++other)
            {
                uint distanceInTopology = node > other
                    ? node - other : other - node;
                if (distanceInTopology <= 2u) continue;
                float otherInverseMass = rope_realtime_inverse_mass(
                    other, count, inverse_masses, anchor_active);
                float totalInverseMass = inverseMass + otherInverseMass;
                if (totalInverseMass <= 0.0f) continue;
                Vec3 separation = subtract(
                    position, ropeRealtimeContactPositions[other]);
                float distance = vector_length(separation);
                if (!isfinite(distance) || distance >= targetSeparation)
                    continue;
                Vec3 direction = distance > 1.0e-7f
                    ? multiply(separation, 1.0f / distance)
                    : (node < other ? Vec3(-1.0f, 0.0f, 0.0f)
                                    : Vec3(1.0f, 0.0f, 0.0f));
                correction = add(correction, multiply(direction,
                    (targetSeparation - distance) * inverseMass /
                        totalInverseMass));
            }
        }

        Vec3 normal = contact_normals[node];
        float target = 0.0f;
        bool rigidContact = contact_active[node] != 0u;
        if (rigidContact)
        {
            float normalLength = vector_length(normal);
            if (!rope_realtime_finite_vec3(normal) ||
                normalLength < 1.0e-6f ||
                !isfinite(contact_offsets[node]))
            {
                pmAtomicOr(status, kRopeRealtimeInvalidMaterial);
                continue;
            }
            normal = multiply(normal, 1.0f / normalLength);
            target = contact_offsets[node] + radius;
            float penetration = target - dot(position, normal);
            if (penetration > 0.0f)
            {
                Vec3 motion = subtract(position, previous_positions[node]);
                Vec3 tangent = subtract(
                    motion, multiply(normal, dot(motion, normal)));
                float tangentLength = vector_length(tangent);
                Vec3 frictionCorrection = tangentLength > 1.0e-7f
                    ? multiply(tangent,
                        -min(tangentLength, friction * penetration) /
                            tangentLength)
                    : Vec3(0.0f);
                correction = add(correction,
                    add(multiply(normal, penetration), frictionCorrection));
            }
        }

        float maximumCorrection = kRopeRealtimeMaximumCorrectionFraction *
            node_spacing;
        float remaining = max(0.0f,
            maximumCorrection - correction_used[node]);
        correction = clamp_length(correction, remaining);
        position = add(position, correction);
        if (!rope_realtime_finite_vec3(position) ||
            !rope_realtime_finite_vec3(correction))
        {
            pmAtomicOr(status, kRopeRealtimeNonfiniteState);
            continue;
        }
        if (rigidContact && dot(position, normal) < target - 1.0e-6f)
            pmAtomicOr(status, kRopeRealtimeUnresolvedContact);
        positions[node] = position;
        correction_used[node] += vector_length(correction);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rope_realtime_commit(
    uniform PMBuffer<Vec3> positions,
    uniform PMBuffer<Vec3> previous_positions,
    uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> inverse_masses,
    uniform PMByteBuffer anchor_active,
    uniform PMBuffer<Vec3> contact_normals,
    uniform PMBuffer<float> contact_offsets,
    uniform PMByteBuffer contact_active, uniform uint count,
    uniform uint capacity, uniform float node_spacing, uniform float radius,
    uniform float friction, uniform float maximum_speed, uniform float dt,
    uniform uint profile_version,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (!rope_realtime_profile_valid(profile_version, dt))
    {
        if (node == 0u)
            rope_realtime_record_profile_invalid(profile_version, dt, status);
        return;
    }
    if (node >= capacity || node >= count) return;
    if (rope_realtime_anchor_active(node, count, anchor_active))
    {
        Vec3 velocity = multiply(subtract(
            positions[node], previous_positions[node]), 1.0f / dt);
        if (!rope_realtime_finite_vec3(velocity) ||
            vector_length(velocity) > maximum_speed + 1.0e-5f)
        {
            pmAtomicOr(status, kRopeRealtimeNonfiniteState);
            velocity = Vec3(0.0f);
        }
        velocities[node] = velocity;
        return;
    }
    if (inverse_masses[node] == 0.0f)
    {
        velocities[node] = Vec3(0.0f);
        return;
    }
    Vec3 velocity = multiply(subtract(
        positions[node], previous_positions[node]),
        (1.0f - kRopeRealtimeSolverDamping) / dt);
    if (contact_active[node] != 0u)
    {
        Vec3 normal = normalized_or(contact_normals[node], Vec3(0.0f));
        float target = contact_offsets[node] + radius;
        if (vector_length(normal) > 0.0f &&
            dot(positions[node], normal) <=
                target + 1.0e-5f * node_spacing)
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
    if (!rope_realtime_finite_vec3(positions[node]) ||
        !rope_realtime_finite_vec3(velocity))
    {
        pmAtomicOr(status, kRopeRealtimeNonfiniteState);
        positions[node] = previous_positions[node];
        velocities[node] = Vec3(0.0f);
        return;
    }
    velocities[node] = velocity;
}

groupshared uint ropeMaximumBits;
groupshared uint ropeReleased;
groupshared uint ropeConverged;
groupshared float ropeStrain[128];

public int rope_anchor_body(RopeData rope, uint node, int first, int last)
{
    return node == 0u ? first : (node + 1u == rope.count ? last : -1);
}

public bool rope_soft_anchor(RopeData rope, uint node)
{
    return (node == 0u && rope.soft_first) ||
           (node + 1u == rope.count && rope.soft_last);
}

public uint rope_soft_end(RopeData rope, uint node)
{
    return node == 0u ? 0u : 1u;
}

public Vec3 rope_anchor_local(RopeData rope, uint node)
{
    return node == 0u
        ? rope.options.first.local_anchor : rope.options.last.local_anchor;
}

float rope_node_weight(RopeData rope, uint node, int first, int last)
{
    if (rope_anchor_body(rope, node, first, last) >= 0) return 0.0f;
    if (rope_soft_anchor(rope, node))
        return rope.soft_anchor_inverse_masses[rope_soft_end(rope, node)];
    return float(rope.count) / rope.options.mass;
}

Vec3 rope_mass(RopeData rope, uint node, Vec3 value, int first, int last)
{
    Vec3 normal = length_squared(rope.normals[node]) > 0.5f
        ? rope.normals[node] : rope.normals2[node];
    if (length_squared(normal) > 0.5f)
    {
        Vec3 edge = cross(normal, rope.normals2[node]);
        if (length_squared(edge) > 1.0e-4f)
        {
            Vec3 tangent = normalized_or(edge, {});
            value = multiply(tangent, dot(tangent, value));
        }
        else value = subtract(value, multiply(normal, dot(normal, value)));
    }
    return multiply(value, rope_node_weight(rope, node, first, last));
}

float rope_direction_weight(RopeData rope, uint node, Vec3 direction,
    int first, int last, PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states)
{
    int body = rope_anchor_body(rope, node, first, last);
    if (body < 0)
        return dot(direction, rope_mass(rope, node, direction, first, last));
    Vec3 arm = rotate(states[body].orientation,
        rope_anchor_local(rope, node));
    Vec3 torque = cross(arm, direction);
    return parameters[body].inverse_mass + dot(torque,
        inverse_inertia_world(parameters[body], states[body], torque));
}

void rope_move_body<let canonicalizeCoupledState : bool>(
    inout RigidBodyState state, Vec3 translation, Vec3 rotation, float dt,
    bool stabilizeState)
{
    state.position = add(state.position, translation);
    Quaternion spin = quaternion_multiply(
        {rotation.x, rotation.y, rotation.z, 0.0f}, state.orientation);
    state.orientation = normalized_quaternion({
        state.orientation.x + 0.5f * spin.x,
        state.orientation.y + 0.5f * spin.y,
        state.orientation.z + 0.5f * spin.z,
        state.orientation.w + 0.5f * spin.w});
    if (canonicalizeCoupledState && stabilizeState)
    {
        // Rope-soft feedback amplifies target-specific sub-ULP normalization
        // differences. This shared binary grid is below contact tolerances.
        const float orientationScale = 1048576.0f;
        state.orientation = {
            round(state.orientation.x * orientationScale) / orientationScale,
            round(state.orientation.y * orientationScale) / orientationScale,
            round(state.orientation.z * orientationScale) / orientationScale,
            round(state.orientation.w * orientationScale) / orientationScale};
    }
    state.linear_velocity = add(state.linear_velocity,
        multiply(translation, 1.0f / dt));
    state.angular_velocity = add(state.angular_velocity,
        multiply(rotation, 1.0f / dt));
    if (canonicalizeCoupledState && stabilizeState)
    {
        const float angularVelocityScale = 4194304.0f;
        state.angular_velocity = {
            round(state.angular_velocity.x * angularVelocityScale) /
                angularVelocityScale,
            round(state.angular_velocity.y * angularVelocityScale) /
                angularVelocityScale,
            round(state.angular_velocity.z * angularVelocityScale) /
                angularVelocityScale};
    }
}

void rope_sync_anchors(
    RopeData rope, int first, int last, PMBuffer<RigidBodyState> states)
{
    if (first >= 0)
        rope.positions[0] = transform_point(
            states[first], rope.options.first.local_anchor);
    if (last >= 0)
        rope.positions[rope.count - 1u] = transform_point(
            states[last], rope.options.last.local_anchor);
    if (rope.soft_first) rope.positions[0] = rope.soft_anchor_positions[0];
    if (rope.soft_last)
        rope.positions[rope.count - 1u] = rope.soft_anchor_positions[1];
}

void rope_solve_tridiagonal(uint edges)
{
    for (uint edge = 1u; edge < edges; ++edge)
    {
        float factor = ropeUpper[edge - 1u] /
            max(ropeDiagonal[edge - 1u], 1.0e-10f);
        ropeDiagonal[edge] -= factor * ropeUpper[edge - 1u];
        ropeRhs[edge] -= factor * ropeRhs[edge - 1u];
    }
    ropeRhs[edges - 1u] /= max(ropeDiagonal[edges - 1u], 1.0e-10f);
    for (int edge = int(edges) - 2; edge >= 0; --edge)
        ropeRhs[edge] = (ropeRhs[edge] -
            ropeUpper[edge] * ropeRhs[edge + 1]) /
            max(ropeDiagonal[edge], 1.0e-10f);
}

void rope_project_stretch<let canonicalizeCoupledState : bool>(
    RopeData rope, float dt, int first, int last,
    PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states,
    bool stabilizeState)
{
    uint lane = pmGroupThreadIndex().x;
    uint edges = rope.count - 1u;
    if (lane == 0u) ropeMaximumBits = 0u;
    float alpha = max(rope.options.stretch_compliance / (dt * dt),
        1.0e-5f * float(rope.count) / rope.options.mass);
    for (uint edge = lane; edge < edges; edge += pmGroupSize().x)
        rope.directions[edge] = normalized_or(subtract(
            rope.positions[edge + 1u], rope.positions[edge]),
            {1.0f, 0.0f, 0.0f});
    pmGroupBarrier();
    for (uint activePass = 0u; activePass < 4u; ++activePass)
    {
        if (lane == 0u) ropeReleased = 0u;
        for (uint edge = lane; edge < edges; edge += pmGroupSize().x)
        {
            ropeRhs[edge] = -(vector_length(subtract(
                rope.positions[edge + 1u], rope.positions[edge])) -
                rope.rest[edge]) - alpha * rope.lambda[edge];
            ropeDiagonal[edge] = rope_direction_weight(rope, edge,
                rope.directions[edge], first, last, parameters, states) +
                rope_direction_weight(rope, edge + 1u,
                    rope.directions[edge], first, last, parameters, states) +
                alpha;
            if (edge != 0u)
                ropeUpper[edge - 1u] = -dot(rope.directions[edge - 1u],
                    rope_mass(rope, edge, rope.directions[edge],
                        first, last));
        }
        pmGroupBarrier();
        if (lane == 0u) rope_solve_tridiagonal(edges);
        pmGroupBarrier();
        for (uint node = lane; node < rope.count; node += pmGroupSize().x)
        {
            Vec3 impulse = {};
            if (node != 0u) impulse = add(impulse,
                multiply(rope.directions[node - 1u], ropeRhs[node - 1u]));
            if (node < edges) impulse = subtract(impulse,
                multiply(rope.directions[node], ropeRhs[node]));
            rope.scratch[node] = impulse;
            int body = rope_anchor_body(rope, node, first, last);
            if (body < 0 && activePass < 3u)
            {
                float releaseImpulse = 1.0e-6f / max(
                    rope_node_weight(rope, node, first, last), 1.0e-8f);
                if (dot(impulse, rope.normals[node]) > releaseImpulse)
                {
                    rope.normals[node] = {};
                    uint ignored;
                    InterlockedExchange(ropeReleased, 1u, ignored);
                }
                if (dot(impulse, rope.normals2[node]) > releaseImpulse)
                {
                    rope.normals2[node] = {};
                    uint ignored;
                    InterlockedExchange(ropeReleased, 1u, ignored);
                }
            }
        }
        pmGroupBarrier();
        bool unchanged = ropeReleased == 0u;
        pmGroupBarrier();
        if (unchanged) break;
    }
    for (uint node = lane; node < rope.count; node += pmGroupSize().x)
    {
        Vec3 impulse = rope.scratch[node];
        int body = rope_anchor_body(rope, node, first, last);
        Vec3 movement = rope_mass(rope, node, impulse, first, last);
        if (body >= 0)
        {
            Vec3 arm = rotate(states[body].orientation,
                rope_anchor_local(rope, node));
            movement = add(multiply(impulse, parameters[body].inverse_mass),
                cross(inverse_inertia_world(parameters[body], states[body],
                    cross(arm, impulse)), arm));
        }
        uint ignored;
        InterlockedMax(ropeMaximumBits,
            asuint(vector_length(movement)), ignored);
    }
    pmGroupBarrier();
    float scale = min(1.0f,
        rope.options.radius / max(asfloat(ropeMaximumBits), 1.0e-10f));
    for (uint edge = lane; edge < edges; edge += pmGroupSize().x)
        rope.lambda[edge] += scale * ropeRhs[edge];
    for (uint node = lane; node < rope.count; node += pmGroupSize().x)
    {
        Vec3 impulse = multiply(rope.scratch[node], scale);
        rope.constraint_forces[node] = add(rope.constraint_forces[node],
            multiply(impulse, 1.0f / (dt * dt)));
        if (rope_anchor_body(rope, node, first, last) < 0)
            rope.positions[node] = add(rope.positions[node],
                rope_mass(rope, node, impulse, first, last));
    }
    pmGroupBarrier();
    if (lane == 0u)
    {
        uint endpoints[2] = {0u, rope.count - 1u};
        for (uint endpoint = 0u; endpoint < 2u; ++endpoint)
        {
            uint node = endpoints[endpoint];
            int body = rope_anchor_body(rope, node, first, last);
            Vec3 impulse = multiply(rope.scratch[node], scale);
            if (rope_soft_anchor(rope, node))
            {
                uint end = rope_soft_end(rope, node);
                rope.soft_anchor_impulses[end] = add(
                    rope.soft_anchor_impulses[end],
                    multiply(impulse, 1.0f / dt));
                if (rope.soft_anchor_inverse_masses[end] > 0.0f)
                    rope.soft_anchor_positions[end] = rope.positions[node];
                continue;
            }
            if (body < 0) continue;
            Vec3 arm = rotate(states[body].orientation,
                rope_anchor_local(rope, node));
            RigidBodyState state = states[body];
            rope_move_body<canonicalizeCoupledState>(state,
                multiply(impulse, parameters[body].inverse_mass),
                inverse_inertia_world(parameters[body], state,
                    cross(arm, impulse)), dt, stabilizeState);
            states[body] = state;
        }
        rope_sync_anchors(rope, first, last, states);
    }
    pmGroupBarrier();
}

public void closest_segments(
    Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2, out Vec3 c1, out Vec3 c2)
{
    Vec3 d1 = subtract(q1, p1);
    Vec3 d2 = subtract(q2, p2);
    Vec3 r = subtract(p1, p2);
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    float s = 0.0f;
    float t = 0.0f;
    float squaredEpsilon = kEpsilon * kEpsilon;
    if (a <= squaredEpsilon && e <= squaredEpsilon)
    {
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= squaredEpsilon)
        t = clamp_scalar(f / e, 0.0f, 1.0f);
    else
    {
        float c = dot(d1, r);
        if (e <= squaredEpsilon)
            s = clamp_scalar(-c / a, 0.0f, 1.0f);
        else
        {
            float b = dot(d1, d2);
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
    c1 = add(p1, multiply(d1, s));
    c2 = add(p2, multiply(d2, t));
}

struct RopeHit
{
    float depth;
    Vec3 normal;
    Vec3 point;
    int body;
    float fraction;
};

void rope_closest_segment_triangle(Vec3 a, Vec3 b, Vec3 x, Vec3 y,
    Vec3 z, out Vec3 p, out Vec3 q)
{
    if (rigid_segment_hits_triangle(a, b, x, y, z, p))
    {
        q = p;
        return;
    }
    float best = 3.402823466e+38f;
    p = a;
    q = rigid_closest_triangle(a, x, y, z);
    Vec3 firstCandidate = q;
    rigid_consider_pair(a, firstCandidate, best, p, q);
    rigid_consider_pair(b, rigid_closest_triangle(b, x, y, z), best,
        p, q);
    Vec3 corners[3] = {x, y, z};
    for (uint edge = 0u; edge < 3u; ++edge)
    {
        Vec3 u;
        Vec3 v;
        closest_segments(a, b, corners[edge], corners[(edge + 1u) % 3u],
            u, v);
        rigid_consider_pair(u, v, best, p, q);
    }
}

RopeHit rope_find_contact(RopeData rope, uint index, bool segment,
    int first, int last, PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states,
    PMBuffer<RigidBodyState> old_states, PMBuffer<TriangleMeshResource> meshes, uint body_count)
{
    RopeHit best = {};
    best.body = -1;
    uint other = segment ? index + 1u : index;
    for (uint bodyIndex = 0u; bodyIndex < body_count; ++bodyIndex)
    {
        if ((int(bodyIndex) == first &&
                index < rope.first_attachment_contact_skip) ||
            (int(bodyIndex) == last &&
                other + rope.last_attachment_contact_skip >= rope.count))
            continue;
        RigidBodyState state = states[bodyIndex];
        RigidBodyState oldState = pmBufferPresent(old_states)
            ? old_states[bodyIndex] : state;
        TriangleMeshResource mesh = meshes[parameters[bodyIndex].mesh.index];
        float radius = rope.options.radius +
            parameters[bodyIndex].collision_margin;
        Vec3 a = inverse_rotate(state.orientation,
            subtract(rope.positions[index], state.position));
        Vec3 b = inverse_rotate(state.orientation,
            subtract(rope.positions[other], state.position));
        Vec3 origin = inverse_rotate(oldState.orientation,
            subtract(rope.previous[index], oldState.position));
        Vec3 margin = {radius, radius, radius};
        Vec3 low = subtract(component_min(a, segment ? b : origin), margin);
        Vec3 high = add(component_max(a, segment ? b : origin), margin);
        if (!bounds_overlap(low, high, mesh.minimum, mesh.maximum)) continue;

        if (pmBufferPresent(mesh.solid_planes))
        {
            uint hintIndex = index * rope.body_capacity + bodyIndex;
            uint hint = rope.solid_hint[hintIndex];
            uint triangleCount = mesh.index_count / 3u;
            bool stationary = state.position.x == oldState.position.x &&
                state.position.y == oldState.position.y &&
                state.position.z == oldState.position.z &&
                state.orientation.x == oldState.orientation.x &&
                state.orientation.y == oldState.orientation.y &&
                state.orientation.z == oldState.orientation.z &&
                state.orientation.w == oldState.orientation.w;
            bool outside = false;
            bool separated = false;
            if (hint < triangleCount)
            {
                CollisionPlane plane = mesh.solid_planes[hint];
                float side = dot(plane.normal, a) - plane.offset;
                if (side > 0.0f)
                {
                    outside = true;
                    separated = stationary && side > radius &&
                        (segment
                            ? dot(plane.normal, b) - plane.offset > radius
                            : dot(plane.normal, origin) - plane.offset > radius);
                }
            }
            float side = -3.402823466e+38f;
            float entryTime = -1.0f;
            CollisionPlane nearest = {};
            CollisionPlane entry = {};
            for (uint triangle = 0u;
                 !outside && triangle < triangleCount; ++triangle)
            {
                CollisionPlane plane = mesh.solid_planes[triangle];
                float planeSide = dot(plane.normal, a) - plane.offset;
                if (planeSide > side)
                {
                    side = planeSide;
                    nearest = plane;
                }
                if (!segment)
                {
                    float before = dot(plane.normal, origin) - plane.offset;
                    if (before > 0.0f && planeSide <= 0.0f)
                    {
                        float time = before / (before - planeSide);
                        if (time > entryTime)
                        {
                            entryTime = time;
                            entry = plane;
                        }
                    }
                }
                if (planeSide > 0.0f)
                {
                    hint = triangle;
                    outside = true;
                    separated = stationary && planeSide > radius &&
                        (segment
                            ? dot(plane.normal, b) - plane.offset > radius
                            : dot(plane.normal, origin) - plane.offset > radius);
                }
            }
            rope.solid_hint[hintIndex] = hint;
            if (separated) continue;
            if (!outside && !segment)
            {
                rope.solid_hint[hintIndex] = kInvalidDense;
                if (entryTime >= 0.0f)
                {
                    nearest = entry;
                    side = dot(entry.normal, a) - entry.offset;
                }
                if (radius - side > best.depth)
                {
                    best.depth = radius - side;
                    best.normal = rotate(state.orientation, nearest.normal);
                    best.point = transform_point(state,
                        subtract(a, multiply(nearest.normal, side)));
                    best.body = int(bodyIndex);
                    best.fraction = 0.0f;
                }
                continue;
            }
        }

        uint stack[64] = {};
        int pending = mesh.bvh_node_count != 0u ? 1 : 0;
        while (pending != 0)
        {
            BvhNode node = mesh.bvh_nodes[stack[--pending]];
            if (!bounds_overlap(low, high, node.minimum, node.maximum))
                continue;
            if (node.triangle_count == 0u)
            {
                if (pending + 2 <= 64)
                {
                    stack[pending++] = node.left;
                    stack[pending++] = node.right;
                }
                continue;
            }
            for (uint triangle = node.first_triangle;
                 triangle < node.first_triangle + node.triangle_count;
                 ++triangle)
            {
                Vec3 x = mesh.vertices[mesh.indices[3u * triangle]];
                Vec3 y = mesh.vertices[mesh.indices[3u * triangle + 1u]];
                Vec3 z = mesh.vertices[mesh.indices[3u * triangle + 2u]];
                if (!bounds_overlap(low, high,
                    component_min(x, component_min(y, z)),
                    component_max(x, component_max(y, z)))) continue;
                Vec3 p = a;
                Vec3 q = {};
                if (segment)
                    rope_closest_segment_triangle(a, b, x, y, z, p, q);
                else q = rigid_closest_triangle(a, x, y, z);
                Vec3 delta = subtract(p, q);
                float distance = vector_length(delta);
                Vec3 face = normalized_or(cross(subtract(y, x),
                    subtract(z, x)), {0.0f, 1.0f, 0.0f});
                if (pmBufferPresent(mesh.solid_planes))
                    face = mesh.solid_planes[triangle].normal;
                float before = dot(subtract(origin, x), face);
                float after = dot(subtract(a, x), face);
                Vec3 normal = distance > 1.0e-7f
                    ? multiply(delta, 1.0f / distance)
                    : multiply(face, before >= 0.0f ? 1.0f : -1.0f);
                float depth = radius - distance;
                if (segment && pmBufferPresent(mesh.solid_planes) &&
                    distance < radius && dot(delta, face) <= 1.0e-7f)
                {
                    normal = face;
                    depth = radius + distance;
                }
                if (!segment && before * after < 0.0f)
                {
                    Vec3 crossing = add(origin, multiply(subtract(a, origin),
                        before / (before - after)));
                    if (length_squared(subtract(
                        rigid_closest_triangle(crossing, x, y, z),
                        crossing))
                        <= radius * radius)
                    {
                        normal = multiply(face, before >= 0.0f ? 1.0f : -1.0f);
                        depth = max(depth, radius + abs(after));
                    }
                }
                if (depth > best.depth)
                {
                    Vec3 edge = subtract(b, a);
                    float fraction = segment ? clamp_scalar(
                        dot(subtract(p, a), edge) /
                            max(length_squared(edge), 1.0e-12f),
                        0.0f, 1.0f) : 0.0f;
                    best.depth = depth;
                    best.normal = rotate(state.orientation, normal);
                    best.point = transform_point(state, q);
                    best.body = int(bodyIndex);
                    best.fraction = fraction;
                }
            }
        }
    }
    return best;
}

void rope_accumulate_body(RopeData rope, uint body, Vec3 impulse, Vec3 arm,
    PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states)
{
    Vec3 translation = multiply(impulse, parameters[body].inverse_mass);
    Vec3 rotation = inverse_inertia_world(parameters[body], states[body],
        cross(arm, impulse));
    atomic_add(rope.body_translation + body, translation);
    atomic_add(rope.body_rotation + body, rotation);
}

float rope_contact_weight(RopeData rope, uint node, Vec3 direction,
    int first, int last, PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states)
{
    return rope_anchor_body(rope, node, first, last) < 0
        ? rope_node_weight(rope, node, first, last)
        : rope_direction_weight(rope, node, direction, first, last,
            parameters, states);
}

void rope_contact_move(RopeData rope, uint node, Vec3 impulse,
    int first, int last, PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states)
{
    int body = rope_anchor_body(rope, node, first, last);
    if (body < 0)
        rope.positions[node] = add(rope.positions[node], multiply(
            impulse, rope_node_weight(rope, node, first, last)));
    else
        rope_accumulate_body(rope, uint(body), impulse,
            rotate(states[body].orientation, rope_anchor_local(rope, node)),
            parameters, states);
}

uint rope_contact_support(RopeData rope, uint node, Vec3 impulse)
{
    return (length_squared(rope.normals[node]) > 0.5f &&
                dot(impulse, rope.normals[node]) <= 0.0f ? 1u : 0u) |
           (length_squared(rope.normals2[node]) > 0.5f &&
                dot(impulse, rope.normals2[node]) <= 0.0f ? 2u : 0u);
}

Vec3 rope_contact_project(RopeData rope, uint node, Vec3 value, uint support)
{
    if (support == 0u) return value;
    Vec3 normal = (support & 1u) != 0u ? rope.normals[node]
        : ((support & 2u) != 0u ? rope.normals2[node] : Vec3(0.0f));
    Vec3 edge = support == 3u ? cross(normal, rope.normals2[node])
        : Vec3(0.0f);
    if (length_squared(edge) > 1.0e-4f)
    {
        Vec3 tangent = normalized_or(edge, {});
        return multiply(tangent, dot(tangent, value));
    }
    return subtract(value, multiply(normal, dot(normal, value)));
}

float rope_position_contact_weight(RopeData rope, uint node, Vec3 direction,
    uint support, int first, int last, PMBuffer<BodyParameters> parameters,
    PMBuffer<RigidBodyState> states)
{
    if (support == 0u || rope_anchor_body(rope, node, first, last) >= 0)
        return rope_contact_weight(rope, node, direction, first, last,
            parameters, states);
    return max(0.0f, dot(direction,
        rope_contact_project(rope, node, direction, support))) *
        rope_node_weight(rope, node, first, last);
}

Vec3 rope_position_contact_response(RopeData rope, uint node, Vec3 direction,
    uint support, int first, int last, PMBuffer<BodyParameters> parameters,
    PMBuffer<RigidBodyState> states)
{
    int body = rope_anchor_body(rope, node, first, last);
    if (body < 0)
        return multiply(rope_contact_project(rope, node, direction, support),
            rope_node_weight(rope, node, first, last));
    Vec3 arm = rotate(states[body].orientation, rope_anchor_local(rope, node));
    return add(multiply(direction, parameters[body].inverse_mass),
        cross(inverse_inertia_world(parameters[body], states[body],
            cross(arm, direction)), arm));
}

Vec3 rope_contact_pair_response(RopeData rope, uint firstNode, uint lastNode,
    Vec3 direction, uint firstSupport, uint lastSupport, float firstWeight,
    float lastWeight, Vec3 arm, uint collider, int first, int last,
    PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states)
{
    Vec3 response = multiply(rope_position_contact_response(rope, firstNode,
        direction, firstSupport, first, last, parameters, states),
        firstWeight * firstWeight);
    if (lastWeight > 0.0f)
        response = add(response, multiply(rope_position_contact_response(rope,
            lastNode, direction, lastSupport, first, last, parameters, states),
            lastWeight * lastWeight));
    response = add(response,
        multiply(direction, parameters[collider].inverse_mass));
    return add(response, cross(inverse_inertia_world(parameters[collider],
        states[collider], cross(arm, direction)), arm));
}

void rope_position_contact_move(RopeData rope, uint node, Vec3 impulse,
    uint support, int first, int last, PMBuffer<BodyParameters> parameters,
    PMBuffer<RigidBodyState> states)
{
    if (rope_anchor_body(rope, node, first, last) >= 0)
        rope_contact_move(rope, node, impulse, first, last, parameters, states);
    else
        rope.positions[node] = add(rope.positions[node], multiply(
            rope_contact_project(rope, node, impulse, support),
            rope_node_weight(rope, node, first, last)));
    if ((support & 1u) == 0u) rope.normals[node] = {};
    if ((support & 2u) == 0u) rope.normals2[node] = {};
}

struct RopeFriction
{
    float normal;
    float tangent;
};

RopeFriction rope_coupled_contact_friction(float normalImpulse,
    float normalMass, float crossMass, float tangentMass,
    float tangentResidual, float friction)
{
    RopeFriction result = {normalImpulse, 0.0f};
    if (!(normalImpulse > 0.0f) || !(normalMass > 1.0e-12f) ||
        !(tangentResidual > 0.0f) || !(friction > 0.0f)) return result;
    float coupling = crossMass / normalMass;
    float schur = tangentMass - crossMass * coupling;
    if (!(schur > 1.0e-12f)) return result;
    float magnitude = tangentResidual / schur;
    float coneDenominator = 1.0f - friction * coupling;
    if (coneDenominator > 0.0f)
        magnitude = min(magnitude,
            friction * normalImpulse / coneDenominator);
    result.normal += coupling * magnitude;
    result.tangent = -magnitude;
    return result;
}

void rope_contact(RopeData rope, uint index, bool segment, float dt,
    int first, int last, PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states,
    PMBuffer<RigidBodyState> old_states, PMBuffer<TriangleMeshResource> meshes, uint body_count)
{
    RopeHit hit = rope_find_contact(rope, index, segment, first, last,
        parameters, states, old_states, meshes, body_count);
    if (hit.body < 0) return;
    uint other = segment ? index + 1u : index;
    float firstWeight = 1.0f - hit.fraction;
    float lastWeight = hit.fraction;
    BodyParameters body = parameters[hit.body];
    RigidBodyState state = states[hit.body];
    Vec3 arm = subtract(hit.point, state.position);
    Vec3 torque = cross(arm, hit.normal);
    float bodyWeight = body.inverse_mass + dot(torque,
        inverse_inertia_world(body, state, torque));
    Vec3 movement = subtract(add(
        multiply(subtract(rope.positions[index], rope.previous[index]),
            firstWeight),
        multiply(subtract(rope.positions[other], rope.previous[other]),
            lastWeight)),
        multiply(add(state.linear_velocity,
            cross(state.angular_velocity, arm)), dt));
    uint firstSupport = rope_contact_support(rope, index, hit.normal);
    uint lastSupport = segment
        ? rope_contact_support(rope, other, hit.normal) : 0u;
    Vec3 impulse = {};
    for (uint activeSet = 0u; activeSet < 5u; ++activeSet)
    {
        float firstInverseMass = rope_position_contact_weight(rope, index,
            hit.normal, firstSupport, first, last, parameters, states);
        float lastInverseMass = segment ? rope_position_contact_weight(rope,
            other, hit.normal, lastSupport, first, last, parameters, states)
            : 0.0f;
        float sum = firstInverseMass * firstWeight * firstWeight +
            lastInverseMass * lastWeight * lastWeight + bodyWeight;
        if (sum <= 1.0e-12f) return;
        float normalImpulse = hit.depth / sum;
        impulse = multiply(hit.normal, normalImpulse);
        if (rope.options.friction > 0.0f)
        {
            Vec3 normalResponse = rope_contact_pair_response(rope, index,
                other, hit.normal, firstSupport, lastSupport, firstWeight,
                lastWeight, arm, uint(hit.body), first, last,
                parameters, states);
            Vec3 correctedMotion = add(movement,
                multiply(normalResponse, normalImpulse));
            Vec3 tangent = subtract(correctedMotion,
                multiply(hit.normal, dot(correctedMotion, hit.normal)));
            float tangentLength = vector_length(tangent);
            if (tangentLength > 1.0e-8f)
            {
                Vec3 direction = multiply(tangent, 1.0f / tangentLength);
                Vec3 tangentResponse = rope_contact_pair_response(rope,
                    index, other, direction, firstSupport, lastSupport,
                    firstWeight, lastWeight, arm, uint(hit.body), first,
                    last, parameters, states);
                RopeFriction coupled = rope_coupled_contact_friction(
                    normalImpulse, sum, dot(hit.normal, tangentResponse),
                    dot(direction, tangentResponse), tangentLength,
                    rope.options.friction);
                impulse = add(multiply(hit.normal, coupled.normal),
                    multiply(direction, coupled.tangent));
            }
        }
        uint nextFirst = rope_contact_support(rope, index, impulse);
        uint nextLast = segment
            ? rope_contact_support(rope, other, impulse) : 0u;
        if ((nextFirst == firstSupport && nextLast == lastSupport) ||
            activeSet == 4u) break;
        if (activeSet == 3u)
        {
            firstSupport = rope_contact_support(rope, index, {});
            lastSupport = segment
                ? rope_contact_support(rope, other, {}) : 0u;
        }
        else
        {
            firstSupport = nextFirst;
            lastSupport = nextLast;
        }
    }
    Vec3 firstForce = rope_anchor_body(rope, index, first, last) < 0
        ? rope_contact_project(rope, index, impulse, firstSupport) : impulse;
    Vec3 lastForce = rope_anchor_body(rope, other, first, last) < 0
        ? rope_contact_project(rope, other, impulse, lastSupport) : impulse;
    rope_position_contact_move(rope, index, multiply(impulse, firstWeight),
        firstSupport, first, last, parameters, states);
    rope.contact_forces[index] = add(rope.contact_forces[index],
        multiply(firstForce, firstWeight / (dt * dt)));
    if (segment)
    {
        rope_position_contact_move(rope, other,
            multiply(impulse, lastWeight), lastSupport,
            first, last, parameters, states);
        rope.contact_forces[other] = add(rope.contact_forces[other],
            multiply(lastForce, lastWeight / (dt * dt)));
    }
    if (body.inverse_mass == 0.0f)
    {
        for (uint node = index; node <= other; ++node)
        {
            if (length_squared(rope.normals[node]) < 0.5f ||
                dot(rope.normals[node], hit.normal) > 0.95f)
                rope.normals[node] = hit.normal;
            else rope.normals2[node] = hit.normal;
        }
    }
    rope_accumulate_body(rope, uint(hit.body), multiply(impulse, -1.0f),
        arm, parameters, states);
}

struct RopeSoftHit
{
    float depth;
    float fraction;
    uint triangle;
    Vec3 normal;
    Vec3 weights;
    Vec3 point;
};

RopeSoftHit rope_soft_find_contact(RopeData rope, RopeSoftTarget target,
    uint index, bool segment)
{
    RopeSoftHit best = {};
    best.triangle = kInvalidDense;
    if (!pmBufferPresent(target.surface) ||
        (target.first_anchor && index == 0u) ||
        (target.last_anchor &&
            index + (segment ? 2u : 1u) == rope.count)) return best;
    uint other = segment ? index + 1u : index;
    Vec3 a = rope.positions[index];
    Vec3 b = rope.positions[other];
    Vec3 oldPosition = rope.previous[index];
    float radius = target.distance;
    if (!segment)
    {
        FluidSoftNearest hit = fluid_soft_nearest(a, oldPosition,
            target.surface, target.previous_surface, target.indices,
            target.triangle_count * 3u, target.orientation, radius,
            target.bounds, true, target.tree, target.order);
        if (hit.triangle != kInvalidDense)
        {
            best.depth = hit.penetration;
            best.fraction = 0.0f;
            best.triangle = hit.triangle / 3u;
            best.normal = hit.normal;
            best.weights = hit.weights;
            best.point = hit.point;
        }
        return best;
    }
    Vec3 margin = {radius, radius, radius};
    Vec3 low = subtract(component_min(a, segment ? b : oldPosition), margin);
    Vec3 high = add(component_max(a, segment ? b : oldPosition), margin);
    if (!bounds_overlap(low, high, target.bounds[0], target.bounds[1]))
        return best;
    uint stack[64] = {};
    uint pending = 1u;
    while (pending != 0u)
    {
        BvhNode node = target.tree[stack[--pending]];
        if (!bounds_overlap(low, high, node.minimum, node.maximum)) continue;
        if (node.triangle_count == 0u)
        {
            if (pending + 2u <= 64u)
            {
                stack[pending++] = node.left;
                stack[pending++] = node.right;
            }
            continue;
        }
        for (uint leaf = 0u; leaf < node.triangle_count; ++leaf)
        {
            uint triangle = target.order[node.first_triangle + leaf];
            uint base = 3u * triangle;
            Vec3 x = target.surface[target.indices[base]];
            Vec3 y = target.surface[target.indices[base + 1u]];
            Vec3 z = target.surface[target.indices[base + 2u]];
            if (!bounds_overlap(low, high,
                component_min(x, component_min(y, z)),
                component_max(x, component_max(y, z)))) continue;
            Vec3 p = a;
            Vec3 q = {};
            if (segment)
                rope_closest_segment_triangle(a, b, x, y, z, p, q);
            else q = rigid_closest_triangle(a, x, y, z);
            Vec3 delta = subtract(p, q);
            float distance = vector_length(delta);
            if (distance >= radius && segment) continue;
            Vec3 face = multiply(normalized_or(cross(subtract(y, x),
                subtract(z, x)), {}), target.orientation);
            if (length_squared(face) < 0.5f) continue;
            float side = dot(delta, face);
            Vec3 normal = distance > 1.0e-7f
                ? multiply(delta, 1.0f / distance) : face;
            float depth = radius - distance;
            if (side < -1.0e-7f)
            {
                normal = face;
                depth = radius + distance;
            }
            if (!segment)
            {
                float before = dot(subtract(oldPosition, x), face);
                float after = dot(subtract(a, x), face);
                if (before > 0.0f && after < 0.0f)
                {
                    Vec3 crossing = add(oldPosition,
                        multiply(subtract(a, oldPosition),
                            before / max(before - after, 1.0e-12f)));
                    if (length_squared(subtract(crossing,
                        rigid_closest_triangle(crossing, x, y, z)))
                        <= radius * radius)
                    {
                        normal = face;
                        depth = max(depth, radius - after);
                    }
                }
            }
            if (depth <= best.depth) continue;
            Vec3 edge = subtract(b, a);
            float fraction = segment ? clamp_scalar(
                dot(subtract(p, a), edge) /
                    max(length_squared(edge), 1.0e-12f),
                0.0f, 1.0f) : 0.0f;
            Vec3 weights = {};
            q = fluid_closest_triangle_barycentric(q, x, y, z, weights);
            best.depth = depth;
            best.fraction = fraction;
            best.triangle = triangle;
            best.normal = normal;
            best.weights = weights;
            best.point = q;
        }
    }
    return best;
}

void rope_soft_contact(RopeData rope, RopeSoftTarget target, uint index,
    bool segment, float dt, int first, int last,
    PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states)
{
    RopeSoftHit hit = rope_soft_find_contact(rope, target, index, segment);
    if (hit.triangle == kInvalidDense) return;
    uint other = segment ? index + 1u : index;
    float firstWeight = 1.0f - hit.fraction;
    float lastWeight = hit.fraction;
    float firstInverseMass = rope_contact_weight(rope, index, hit.normal,
        first, last, parameters, states);
    float lastInverseMass = segment ? rope_contact_weight(rope, other,
        hit.normal, first, last, parameters, states) : 0.0f;
    uint base = 3u * hit.triangle;
    float weights[3] = {hit.weights.x, hit.weights.y, hit.weights.z};
    float softWeight = 0.0f;
    Vec3 softVelocity = {};
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        SoftBodySurfaceBinding binding =
            target.bindings[target.indices[base + corner]];
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            uint node = binding.nodes[slot];
            float weight = weights[corner] * binding.weights[slot];
            softWeight += weight * weight * target.inverse_masses[node];
            softVelocity = add(softVelocity,
                multiply(target.velocities[node], weight));
        }
    }
    float sum = firstInverseMass * firstWeight * firstWeight +
        lastInverseMass * lastWeight * lastWeight;
    if (sum <= 1.0e-12f) return;
    float correction = hit.depth / sum;
    Vec3 impulse = multiply(hit.normal, correction);
    Vec3 motion = subtract(add(
        multiply(subtract(rope.positions[index], rope.previous[index]),
            firstWeight),
        multiply(subtract(rope.positions[other], rope.previous[other]),
            lastWeight)), multiply(softVelocity, dt));
    Vec3 tangent = subtract(motion,
        multiply(hit.normal, dot(motion, hit.normal)));
    float speed = vector_length(tangent);
    if (speed > 1.0e-8f)
    {
        Vec3 direction = multiply(tangent, 1.0f / speed);
        float tangentWeight = rope_contact_weight(rope, index, direction,
            first, last, parameters, states) * firstWeight * firstWeight +
            (segment ? rope_contact_weight(rope, other, direction,
                first, last, parameters, states) * lastWeight * lastWeight
                : 0.0f);
        impulse = subtract(impulse, multiply(direction,
            min(speed / max(tangentWeight, 1.0e-12f),
                target.friction * correction)));
    }
    rope_contact_move(rope, index, multiply(impulse, firstWeight),
        first, last, parameters, states);
    rope.soft_body_contact_forces[index] = add(
        rope.soft_body_contact_forces[index],
        multiply(impulse, firstWeight / (dt * dt)));
    if (segment)
    {
        rope_contact_move(rope, other, multiply(impulse, lastWeight),
            first, last, parameters, states);
        rope.soft_body_contact_forces[other] = add(
            rope.soft_body_contact_forces[other],
            multiply(impulse, lastWeight / (dt * dt)));
    }
    if (softWeight < 1.0e-9f)
    {
        for (uint node = index; node <= other; ++node)
        {
            if (length_squared(rope.normals[node]) < 0.5f ||
                dot(rope.normals[node], hit.normal) > 0.95f)
                rope.normals[node] = hit.normal;
            else rope.normals2[node] = hit.normal;
        }
    }
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        SoftBodySurfaceBinding binding =
            target.bindings[target.indices[base + corner]];
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            float weight = weights[corner] * binding.weights[slot];
            if (weight > 0.0f)
                atomic_add(target.node_impulses + binding.nodes[slot],
                    multiply(impulse, -weight / dt));
        }
    }
    pmAtomicAdd(target.contact_count, 1u);
    pmAtomicMaxPositiveFloat(target.maximum_penetration, hit.depth);
}

void rope_project_velocities(RopeData rope, float dt, int first, int last,
    PMBuffer<BodyParameters> parameters, PMBuffer<RigidBodyState> states)
{
    uint lane = pmGroupThreadIndex().x;
    uint edges = rope.count - 1u;
    float alpha = max(rope.options.stretch_compliance / (dt * dt),
        1.0e-5f * float(rope.count) / rope.options.mass);
    for (uint edge = lane; edge < edges; edge += pmGroupSize().x)
    {
        rope.directions[edge] = normalized_or(subtract(
            rope.positions[edge + 1u], rope.positions[edge]),
            {1.0f, 0.0f, 0.0f});
        ropeRhs[edge] = -dot(rope.directions[edge], subtract(
            rope.velocities[edge + 1u], rope.velocities[edge]));
    }
    pmGroupBarrier();
    for (uint edge = lane; edge < edges; edge += pmGroupSize().x)
    {
        ropeDiagonal[edge] = rope_contact_weight(rope, edge,
            rope.directions[edge], first, last, parameters, states) +
            rope_contact_weight(rope, edge + 1u, rope.directions[edge],
                first, last, parameters, states) + alpha;
        if (edge != 0u)
            ropeUpper[edge - 1u] = -dot(rope.directions[edge - 1u],
                rope.directions[edge]) *
                rope_node_weight(rope, edge, first, last);
    }
    pmGroupBarrier();
    if (lane == 0u) rope_solve_tridiagonal(edges);
    pmGroupBarrier();
    for (uint node = lane; node < rope.count; node += pmGroupSize().x)
    {
        Vec3 impulse = {};
        if (node != 0u)
            impulse = add(impulse,
                multiply(rope.directions[node - 1u], ropeRhs[node - 1u]));
        if (node < edges)
            impulse = subtract(impulse,
                multiply(rope.directions[node], ropeRhs[node]));
        rope.scratch[node] = impulse;
        rope.constraint_forces[node] = add(rope.constraint_forces[node],
            multiply(impulse, 1.0f / dt));
        if (rope_anchor_body(rope, node, first, last) < 0)
            rope.velocities[node] = add(rope.velocities[node], multiply(
                impulse, rope_node_weight(rope, node, first, last)));
    }
    pmGroupBarrier();
    if (lane == 0u)
    {
        for (uint endpoint = 0u; endpoint < 2u; ++endpoint)
        {
            uint node = endpoint == 0u ? 0u : rope.count - 1u;
            if (rope_soft_anchor(rope, node))
            {
                uint end = rope_soft_end(rope, node);
                rope.soft_anchor_impulses[end] = add(
                    rope.soft_anchor_impulses[end], rope.scratch[node]);
                if (rope.soft_anchor_inverse_masses[end] > 0.0f)
                    rope.soft_anchor_velocities[end] = rope.velocities[node];
                else rope.velocities[node] = rope.soft_anchor_velocities[end];
                continue;
            }
            int body = rope_anchor_body(rope, node, first, last);
            if (body < 0) continue;
            Vec3 arm = rotate(states[body].orientation,
                rope_anchor_local(rope, node));
            states[body].linear_velocity = add(
                states[body].linear_velocity,
                multiply(rope.scratch[node], parameters[body].inverse_mass));
            states[body].angular_velocity = add(
                states[body].angular_velocity,
                inverse_inertia_world(parameters[body], states[body],
                    cross(arm, rope.scratch[node])));
            rope.velocities[node] = add(states[body].linear_velocity,
                cross(states[body].angular_velocity, arm));
        }
    }
    pmGroupBarrier();
}

void rope_advance_impl<let canonicalizeCoupledState : bool>(
    RopeData rope, float dt, Vec3 gravity,
    int first, int last, PMBuffer<BodyParameters> parameters,
    PMBuffer<RigidBodyState> states, PMBuffer<RigidBodyState> old_states,
    PMBuffer<TriangleMeshResource> meshes, uint body_count,
    bool first_substep, RopeSoftTarget soft_a, RopeSoftTarget soft_b)
{
    uint lane = pmGroupThreadIndex().x;
    if (lane == 0u) ropeConverged = 0u;
    if (lane == 0u)
        rope_sync_anchors(rope, first, last,
            pmBufferPresent(old_states) ? old_states : states);
    pmGroupBarrier();
    for (uint node = lane; node < rope.count; node += pmGroupSize().x)
    {
        rope.previous[node] = rope.positions[node];
        rope.normals[node] = {};
        rope.normals2[node] = {};
        if (first_substep)
        {
            rope.constraint_forces[node] = {};
            rope.contact_forces[node] = {};
            rope.fluid_contact_forces[node] = {};
            rope.soft_body_contact_forces[node] = {};
        }
        if (rope_anchor_body(rope, node, first, last) < 0 &&
            !rope_soft_anchor(rope, node))
        {
            rope.velocities[node] = clamp_length(multiply(add(
                rope.velocities[node], multiply(gravity, dt)),
                exp(-rope.options.velocity_damping * dt)),
                rope.options.maximum_speed);
            rope.positions[node] = add(rope.positions[node],
                multiply(rope.velocities[node], dt));
        }
        if (node + 1u < rope.count) rope.lambda[node] = 0.0f;
    }
    pmGroupBarrier();
    if (lane == 0u) rope_sync_anchors(rope, first, last, states);
    pmGroupBarrier();
    uint maximumIterations = min(32u, 4u * rope.options.solver_iterations);
    for (uint iteration = 0u; iteration < maximumIterations; ++iteration)
    {
        rope_project_stretch<canonicalizeCoupledState>(
            rope, dt, first, last, parameters, states, first_substep);
        pmGroupBarrier();
        for (uint phase = 0u; phase < 3u; ++phase)
        {
            for (uint body = lane; body < body_count;
                 body += pmGroupSize().x)
            {
                rope.body_translation[body] = {};
                rope.body_rotation[body] = {};
            }
            pmGroupBarrier();
            if (phase == 0u)
            {
                for (uint node = lane; node < rope.count;
                     node += pmGroupSize().x)
                {
                    rope_contact(rope, node, false, dt, first, last,
                        parameters, states, old_states, meshes, body_count);
                    rope_soft_contact(rope, soft_a, node, false, dt,
                        first, last, parameters, states);
                    rope_soft_contact(rope, soft_b, node, false, dt,
                        first, last, parameters, states);
                }
            }
            else
            {
                for (uint node = 2u * lane + phase - 1u;
                     node + 1u < rope.count;
                     node += 2u * pmGroupSize().x)
                {
                    rope_contact(rope, node, true, dt, first, last,
                        parameters, states, old_states, meshes, body_count);
                    rope_soft_contact(rope, soft_a, node, true, dt,
                        first, last, parameters, states);
                    rope_soft_contact(rope, soft_b, node, true, dt,
                        first, last, parameters, states);
                }
            }
            pmGroupBarrier();
            if (lane == 0u)
            {
                for (uint body = 0u; body < body_count; ++body)
                    rope_move_body<canonicalizeCoupledState>(states[body],
                        rope.body_translation[body],
                        rope.body_rotation[body], dt, first_substep);
                rope_sync_anchors(rope, first, last, states);
            }
            pmGroupBarrier();
        }
        if (rope.options.self_collision)
        {
            for (uint node = lane; node < rope.count;
                 node += pmGroupSize().x)
            {
                Vec3 correction = {};
                uint hits = 0u;
                float wi = rope_node_weight(rope, node, first, last);
                for (uint other = 0u; other < rope.count && wi > 0.0f;
                     ++other)
                {
                    if (abs(int(node) - int(other)) <= 2) continue;
                    Vec3 delta = subtract(
                        rope.positions[node], rope.positions[other]);
                    float length = vector_length(delta);
                    float wj = rope_node_weight(rope, other, first, last);
                    if (length >= 2.0f * rope.options.radius ||
                        wi + wj <= 0.0f) continue;
                    correction = add(correction, multiply(normalized_or(
                        delta, {1.0f, 0.0f, 0.0f}),
                        (2.0f * rope.options.radius - length) * wi /
                            (wi + wj)));
                    ++hits;
                }
                rope.scratch[node] = {};
                if (hits != 0u)
                    rope.scratch[node] = multiply(
                        correction, 1.0f / float(hits));
            }
            pmGroupBarrier();
            for (uint node = lane; node < rope.count;
                 node += pmGroupSize().x)
                rope.positions[node] = add(
                    rope.positions[node], rope.scratch[node]);
            pmGroupBarrier();
        }
        float strain = 0.0f;
        for (uint edge = lane; edge + 1u < rope.count;
             edge += pmGroupSize().x)
            strain = max(strain, abs(vector_length(subtract(
                rope.positions[edge + 1u], rope.positions[edge])) /
                rope.rest[edge] - 1.0f));
        ropeStrain[lane] = strain;
        pmGroupBarrier();
        for (uint stride = pmGroupSize().x / 2u; stride != 0u;
             stride /= 2u)
        {
            if (lane < stride)
                ropeStrain[lane] = max(
                    ropeStrain[lane], ropeStrain[lane + stride]);
            pmGroupBarrier();
        }
        if (lane == 0u)
            ropeConverged = ((iteration >= 1u && ropeStrain[0] < 1.0e-3f) ||
                (iteration + 1u >= rope.options.solver_iterations &&
                    ropeStrain[0] < 0.005f)) ? 1u : 0u;
        pmGroupBarrier();
        if (ropeConverged != 0u) break;
    }
    if (ropeConverged == 0u)
    {
        for (uint node = lane; node < rope.count; node += pmGroupSize().x)
        {
            rope.normals[node] = {};
            rope.normals2[node] = {};
        }
        pmGroupBarrier();
        for (uint pass = 0u; pass < 8u; ++pass)
            rope_project_stretch<canonicalizeCoupledState>(
                rope, dt, first, last, parameters, states, first_substep);
    }
    for (uint node = lane; node < rope.count; node += pmGroupSize().x)
    {
        int body = rope_anchor_body(rope, node, first, last);
        if (rope_soft_anchor(rope, node))
        {
            uint end = rope_soft_end(rope, node);
            rope.velocities[node] = rope.soft_anchor_inverse_masses[end] > 0.0f
                ? clamp_length(multiply(subtract(
                    rope.positions[node], rope.previous[node]), 1.0f / dt),
                    rope.options.maximum_speed)
                : rope.soft_anchor_velocities[end];
        }
        else if (body < 0)
            rope.velocities[node] = clamp_length(multiply(subtract(
                rope.positions[node], rope.previous[node]), 1.0f / dt),
                rope.options.maximum_speed);
        else
        {
            Vec3 arm = rotate(states[body].orientation,
                rope_anchor_local(rope, node));
            rope.velocities[node] = add(states[body].linear_velocity,
                cross(states[body].angular_velocity, arm));
        }
    }
    pmGroupBarrier();
    rope_project_velocities(rope, dt, first, last, parameters, states);
}

#ifndef PM_TARGET_VULKAN
#define PM_TARGET_VULKAN 0
#endif

[shader("compute")]
[numthreads(128, 1, 1)]
void rope_advance(
#if PM_TARGET_VULKAN
    uniform PMBuffer<RopeData> rope_buffer,
#else
    uniform RopeData rope,
#endif
    uniform float dt, uniform Vec3 gravity,
    uniform int first, uniform int last, uniform PMBuffer<BodyParameters> parameters,
    uniform PMBuffer<RigidBodyState> states, uniform PMBuffer<RigidBodyState> old_states,
    uniform PMBuffer<TriangleMeshResource> meshes, uniform uint body_count,
    uniform bool first_substep,
#if PM_TARGET_VULKAN
    uniform PMBuffer<RopeSoftTarget> soft_a_buffer,
    uniform PMBuffer<RopeSoftTarget> soft_b_buffer)
#else
    uniform RopeSoftTarget soft_a,
    uniform RopeSoftTarget soft_b PM_D3D12_DISPATCH_PARAMETER)
#endif
{
#if PM_TARGET_VULKAN
    RopeData rope = rope_buffer[0];
    RopeSoftTarget soft_a = soft_a_buffer[0];
    RopeSoftTarget soft_b = soft_b_buffer[0];
#else
    PM_D3D12_DISPATCH_PROLOGUE();
#endif
    rope_advance_impl<false>(rope, dt, gravity, first, last, parameters,
        states, old_states, meshes, body_count, first_substep, soft_a, soft_b);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void rope_advance_coupled(
#if PM_TARGET_VULKAN
    uniform PMBuffer<RopeData> rope_buffer,
#else
    uniform RopeData rope,
#endif
    uniform float dt, uniform Vec3 gravity,
    uniform int first, uniform int last, uniform PMBuffer<BodyParameters> parameters,
    uniform PMBuffer<RigidBodyState> states, uniform PMBuffer<RigidBodyState> old_states,
    uniform PMBuffer<TriangleMeshResource> meshes, uniform uint body_count,
    uniform bool first_substep,
#if PM_TARGET_VULKAN
    uniform PMBuffer<RopeSoftTarget> soft_a_buffer,
    uniform PMBuffer<RopeSoftTarget> soft_b_buffer)
#else
    uniform RopeSoftTarget soft_a,
    uniform RopeSoftTarget soft_b PM_D3D12_DISPATCH_PARAMETER)
#endif
{
#if PM_TARGET_VULKAN
    RopeData rope = rope_buffer[0];
    RopeSoftTarget soft_a = soft_a_buffer[0];
    RopeSoftTarget soft_b = soft_b_buffer[0];
#else
    PM_D3D12_DISPATCH_PROLOGUE();
#endif
    rope_advance_impl<true>(rope, dt, gravity, first, last, parameters,
        states, old_states, meshes, body_count, first_substep, soft_a, soft_b);
}
