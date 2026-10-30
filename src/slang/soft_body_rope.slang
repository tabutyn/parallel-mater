#include "dispatch_entry.slang"
#include "dispatch_shape.slangh"
#include "coupling_spatial.slangh"
// SPDX-License-Identifier: MIT
// Shared soft-body and rope endpoint coupling kernels. Contact traversal is
// shared with rope.slang.

import types;
import math;
import device_intrinsics;
import graphics_resources;
import geometry_fluid;

groupshared float ropeSoftSums[256];

bool rope_soft_inside_shared(
    Vec3 point, PMBuffer<Vec3> surface, PMBuffer<uint> indices,
    uint index_count)
{
    float angle = 0.0f;
    for (uint base = 0u; base < index_count; base += 3u)
    {
        Vec3 a = subtract(surface[indices[base]], point);
        Vec3 b = subtract(surface[indices[base + 1u]], point);
        Vec3 c = subtract(surface[indices[base + 2u]], point);
        float la = vector_length(a), lb = vector_length(b), lc = vector_length(c);
        angle += 2.0f * atan2(dot(a, cross(b, c)),
            la * lb * lc + dot(a, b) * lc + dot(b, c) * la +
            dot(c, a) * lb);
    }
    return abs(angle) > 6.2831853f;
}

struct RopeSoftNearest
{
    uint triangle;
    Vec3 point;
    Vec3 normal;
    Vec3 weights;
    float penetration;
};

RopeSoftNearest rope_soft_nearest_shared(
    Vec3 point, Vec3 start, PMBuffer<Vec3> surface, PMBuffer<Vec3> previous,
    PMBuffer<uint> indices, uint index_count, float orientation,
    float distance, bool sweep, PMBuffer<uint> triangle_generations,
    uint soft_owner_slot, uint soft_owner_generation,
    uint soft_topology_revision,
    PMBuffer<CouplingSpatialLeafMetadata> spatial_metadata,
    PMByteBuffer spatial_admission, uint spatial_leaf_capacity,
    PMBuffer<uint> candidate_counts, PMBuffer<uint> candidate_sources,
    uint candidates_per_query, uint query, PMBuffer<uint> spatial_status)
{
    RopeSoftNearest result = {};
    result.triangle = kInvalidDense;
    float best = 3.402823466e38f, earliest = 2.0f;
    uint candidateCount = candidate_counts[query];
    if (candidateCount > candidates_per_query)
    {
        pmAtomicOr(spatial_status, kCouplingSpatialQueryOverflow);
        return result;
    }
    uint candidateBase = query * candidates_per_query;
    for (uint candidateIndex = 0u;
         candidateIndex < candidateCount; ++candidateIndex)
    {
        uint source = candidate_sources[candidateBase + candidateIndex];
        if (source >= spatial_leaf_capacity || spatial_admission[source] == 0u)
        {
            pmAtomicOr(spatial_status, kCouplingSpatialAdmissionInvalid);
            return result;
        }
        CouplingSpatialLeafMetadata leaf = spatial_metadata[source];
        if (leaf.identity.family != kCouplingSpatialSoftFamily ||
            leaf.identity.owner_slot != soft_owner_slot ||
            leaf.identity.owner_generation != soft_owner_generation ||
            leaf.identity.topology_revision != soft_topology_revision ||
            leaf.identity.primitive_kind != kCouplingSpatialTrianglePrimitive)
            continue;
        uint triangle = leaf.payload_ref;
        if (triangle >= index_count / 3u ||
            leaf.identity.primitive_slot != triangle ||
            leaf.identity.primitive_generation !=
                triangle_generations[triangle])
        {
            pmAtomicOr(spatial_status, kCouplingSpatialStaleEpoch);
            return result;
        }
        uint base = triangle * 3u;
        Vec3 a = surface[indices[base]];
        Vec3 b = surface[indices[base + 1u]];
        Vec3 c = surface[indices[base + 2u]];
        Vec3 crossNormal = cross(subtract(b, a), subtract(c, a));
        if (length_squared(crossNormal) < 1.0e-16f) continue;
        Vec3 normal = multiply(normalized_or(crossNormal, {}), orientation);
        Vec3 weights = {};
        Vec3 nearest = fluid_closest_triangle_barycentric(
            point, a, b, c, weights);
        float squared = length_squared(subtract(point, nearest));
        if (earliest > 1.0f && squared < best)
        {
            best = squared;
            result = {base, nearest, normal, weights,
                distance - dot(subtract(point, nearest), normal)};
        }
        if (!sweep) continue;
        Vec3 oldNear = add(multiply(previous[indices[base]], weights.x),
            add(multiply(previous[indices[base + 1u]], weights.y),
                multiply(previous[indices[base + 2u]], weights.z)));
        Vec3 transportedStart = add(start, subtract(nearest, oldNear));
        float first = dot(subtract(transportedStart, a), normal);
        float last = dot(subtract(point, a), normal);
        if (first < distance || last >= distance ||
            first - last < 1.0e-8f) continue;
        float time = (first - distance) / (first - last);
        if (time >= earliest) continue;
        Vec3 crossing = subtract(add(transportedStart,
            multiply(subtract(point, transportedStart), time)),
            multiply(normal, distance));
        Vec3 hitWeights = {};
        Vec3 hit = fluid_closest_triangle_barycentric(
            crossing, a, b, c, hitWeights);
        if (length_squared(subtract(crossing, hit)) > 1.0e-8f) continue;
        earliest = time;
        result = {base, hit, normal, hitWeights, distance - last};
    }
    if (result.triangle == kInvalidDense) return result;
    if (earliest > 1.0f)
    {
        float radius = sqrt(best);
        bool faceInterior = result.weights.x > 1.0e-4f &&
            result.weights.y > 1.0e-4f && result.weights.z > 1.0e-4f;
        bool inside = faceInterior
            ? dot(subtract(point, result.point), result.normal) < 0.0f
            : rope_soft_inside_shared(point, surface, indices, index_count);
        if (!inside)
        {
            result.penetration = distance - radius;
            if (radius > 1.0e-7f)
                result.normal = multiply(
                    subtract(point, result.point), 1.0f / radius);
        }
        else
        {
            result.penetration = distance + radius;
            if (radius > 1.0e-7f)
                result.normal = multiply(
                    subtract(result.point, point), 1.0f / radius);
        }
    }
    if (result.penetration <= 0.0f) result.triangle = kInvalidDense;
    return result;
}

[shader("compute")]
[numthreads(PM_THREADS(128), 1, 1)]
void rope_soft_realtime_contact(
    uniform PMBuffer<Vec3> rope_positions,
    uniform PMBuffer<Vec3> rope_previous_positions,
    uniform PMBuffer<Vec3> rope_velocities,
    uniform PMBuffer<float> rope_inverse_masses, uniform uint rope_count,
    uniform float rope_maximum_speed, uniform bool attach_first,
    uniform bool attach_last, uniform PMBuffer<Vec3> rope_forces,
    uniform PMBuffer<Vec3> surface,
    uniform PMBuffer<Vec3> previous_surface,
    uniform PMBuffer<uint> indices,
    uniform uint index_count, uniform PMBuffer<SoftBodySurfaceBinding> bindings,
    uniform PMBuffer<Vec3> soft_velocities,
    uniform PMBuffer<float> soft_inverse_masses,
    uniform float orientation, uniform float distance, uniform float friction,
    uniform PMBuffer<uint> triangle_generations,
    uniform uint soft_owner_slot, uniform uint soft_owner_generation,
    uniform uint soft_topology_revision,
    uniform PMBuffer<CouplingSpatialLeafMetadata> spatial_metadata,
    uniform PMByteBuffer spatial_admission,
    uniform uint spatial_leaf_capacity,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_sources,
    uniform uint candidates_per_query, uniform PMBuffer<uint> spatial_status,
    uniform PMBuffer<Vec3> node_impulses,
    uniform PMBuffer<uint> contact_count,
    uniform PMBuffer<float> maximum_penetration,
    uniform float dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint ropeNode = pmThreadIndex();
    if (ropeNode >= rope_count ||
        (ropeNode == 0u && attach_first) ||
        (ropeNode + 1u == rope_count && attach_last)) return;
    float ropeInverseMass = rope_inverse_masses[ropeNode];
    if (ropeInverseMass <= 0.0f) return;
    float ropeMass = 0.0f;
    for (uint node = 0u; node < rope_count; ++node)
    {
        bool attached = (node == 0u && attach_first) ||
            (node + 1u == rope_count && attach_last);
        if (!attached && rope_inverse_masses[node] > 0.0f)
            ropeMass += 1.0f / rope_inverse_masses[node];
    }
    if (ropeMass <= 0.0f) return;
    float coherentRopeInverseMass = 1.0f / ropeMass;
    RopeSoftNearest hit = rope_soft_nearest_shared(
        rope_positions[ropeNode], rope_previous_positions[ropeNode],
        surface, previous_surface, indices, index_count, orientation,
        distance, true, triangle_generations, soft_owner_slot,
        soft_owner_generation, soft_topology_revision, spatial_metadata,
        spatial_admission, spatial_leaf_capacity, candidate_counts,
        candidate_sources, candidates_per_query, ropeNode, spatial_status);
    if (hit.triangle == kInvalidDense) return;
    RopeSoftNearest previousHit = rope_soft_nearest_shared(
        rope_previous_positions[ropeNode],
        rope_previous_positions[ropeNode], previous_surface,
        previous_surface, indices, index_count, orientation, distance,
        false, triangle_generations, soft_owner_slot,
        soft_owner_generation, soft_topology_revision, spatial_metadata,
        spatial_admission, spatial_leaf_capacity, candidate_counts,
        candidate_sources, candidates_per_query, ropeNode, spatial_status);
    Vec3 responseNormal = previousHit.triangle != kInvalidDense
        ? previousHit.normal : hit.normal;

    FluidSoftContact contact = {};
    float barycentric[3] = {hit.weights.x, hit.weights.y, hit.weights.z};
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        SoftBodySurfaceBinding binding =
            bindings[indices[hit.triangle + corner]];
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            float weight = barycentric[corner] * binding.weights[slot];
            if (weight <= 0.0f) continue;
            uint node = binding.nodes[slot];
            uint item = 0u;
            while (item < contact.count && contact.nodes[item] != node)
                ++item;
            if (item == contact.count)
                contact.nodes[contact.count++] = node;
            contact.weights[item] += weight;
        }
    }
    float inverseMass = coherentRopeInverseMass;
    Vec3 softVelocity = {};
    for (uint item = 0u; item < contact.count; ++item)
    {
        uint node = contact.nodes[item];
        float weight = contact.weights[item];
        inverseMass += weight * weight * soft_inverse_masses[node];
        softVelocity = add(softVelocity,
            multiply(soft_velocities[node], weight));
    }
    if (inverseMass <= 1.0e-12f) return;

    Vec3 relativeVelocity = subtract(
        rope_velocities[ropeNode], softVelocity);
    float normalSpeed = dot(relativeVelocity, responseNormal);
    float normalImpulse = max(0.0f, -normalSpeed) / inverseMass;
    Vec3 impulse = multiply(responseNormal, normalImpulse);
    Vec3 tangent = subtract(relativeVelocity,
        multiply(responseNormal, normalSpeed));
    float tangentSpeed = vector_length(tangent);
    if (tangentSpeed > 1.0e-7f && friction > 0.0f)
        impulse = add(impulse, multiply(tangent,
            -min(tangentSpeed / inverseMass,
                friction * normalImpulse) / tangentSpeed));

    Vec3 correction = multiply(hit.normal, hit.penetration);
    Vec3 velocityDelta = multiply(impulse, coherentRopeInverseMass);
    // Move the coherent rope by the geometric projection. A point-only move
    // leaves stored stretch energy that pulls the node straight back through
    // the surface as soon as the graph edge is removed.
    for (uint node = 0u; node < rope_count; ++node)
    {
        bool attached = (node == 0u && attach_first) ||
            (node + 1u == rope_count && attach_last);
        atomic_add(rope_positions + node, correction);
        atomic_add(rope_previous_positions + node, correction);
        if (!attached && rope_inverse_masses[node] > 0.0f)
        {
            atomic_add(rope_positions + node,
                multiply(velocityDelta, dt));
            atomic_add(rope_velocities + node, velocityDelta);
        }
    }
    rope_forces[ropeNode] = add(
        rope_forces[ropeNode], multiply(impulse, 1.0f / dt));
    for (uint item = 0u; item < contact.count; ++item)
        atomic_add(node_impulses + contact.nodes[item], multiply(
            impulse, -contact.weights[item]));
    pmAtomicAdd(contact_count, 1u);
    pmAtomicMaxPositiveFloat(maximum_penetration, hit.penetration);
}

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void rope_soft_sample_anchor(
    uniform PMBuffer<Vec3> soft_anchor_positions,
    uniform PMBuffer<Vec3> soft_anchor_velocities,
    uniform PMBuffer<Vec3> soft_anchor_impulses,
    uniform PMBuffer<float> soft_anchor_inverse_masses, uniform uint end,
    uniform PMBuffer<Vec3> surface,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<uint> indices,
    uniform PMBuffer<SoftBodySurfaceBinding> bindings, uniform uint triangle,
    uniform Vec3 weights, uniform Vec3 offset PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmGroupThreadIndex().x != 0u || pmBlockIndex().x != 0u) return;
    uint base = 3u * triangle;
    float w[3];
    w[0] = weights.x;
    w[1] = weights.y;
    w[2] = weights.z;
    Vec3 position = offset;
    Vec3 velocity = {};
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        uint vertex = indices[base + corner];
        position = add(position, multiply(surface[vertex], w[corner]));
        SoftBodySurfaceBinding binding = bindings[vertex];
        for (uint slot = 0u; slot < 4u; ++slot)
            velocity = add(velocity, multiply(
                velocities[binding.nodes[slot]],
                w[corner] * binding.weights[slot]));
    }
    soft_anchor_positions[end] = position;
    soft_anchor_velocities[end] = velocity;
    soft_anchor_impulses[end] = {};
    soft_anchor_inverse_masses[end] = 0.0f;
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void rope_soft_scatter_anchor(
    uniform PMBuffer<Vec3> soft_anchor_positions,
    uniform PMBuffer<Vec3> soft_anchor_impulses, uniform uint end,
    uniform PMBuffer<Vec3> node_positions, uniform uint node_count,
    uniform float support_radius, uniform float contact_support_radius,
    uniform PMBuffer<uint> contact_count, uniform PMBuffer<uint> indices,
    uniform PMBuffer<SoftBodySurfaceBinding> bindings, uniform uint triangle,
    uniform Vec3 weights, uniform PMBuffer<Vec3> node_impulses PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u) return;
    uint thread = pmGroupThreadIndex().x;
    Vec3 anchor = soft_anchor_positions[end];
    support_radius = contact_support_radius +
        (support_radius - contact_support_radius) /
        (1.0f + float(*contact_count) / 2.0f);
    float radiusSquared = support_radius * support_radius;
    float localSum = 0.0f;
    for (uint node = thread; node < node_count; node += pmGroupSize().x)
    {
        float squared = length_squared(subtract(node_positions[node], anchor));
        if (squared < radiusSquared)
        {
            float t = 1.0f - squared / radiusSquared;
            localSum += t * t;
        }
    }
    ropeSoftSums[thread] = localSum;
    pmGroupBarrier();
    for (uint stride = pmGroupSize().x / 2u; stride != 0u; stride /= 2u)
    {
        if (thread < stride)
            ropeSoftSums[thread] += ropeSoftSums[thread + stride];
        pmGroupBarrier();
    }
    if (ropeSoftSums[0] > 1.0e-8f)
    {
        Vec3 impulse = multiply(
            soft_anchor_impulses[end], 1.0f / ropeSoftSums[0]);
        for (uint node = thread; node < node_count; node += pmGroupSize().x)
        {
            float squared = length_squared(
                subtract(node_positions[node], anchor));
            if (squared >= radiusSquared) continue;
            float t = 1.0f - squared / radiusSquared;
            node_impulses[node] = add(
                node_impulses[node], multiply(impulse, t * t));
        }
        return;
    }
    if (thread != 0u) return;
    float w[3];
    w[0] = weights.x;
    w[1] = weights.y;
    w[2] = weights.z;
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        SoftBodySurfaceBinding binding =
            bindings[indices[3u * triangle + corner]];
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            float weight = w[corner] * binding.weights[slot];
            if (weight > 0.0f)
                node_impulses[binding.nodes[slot]] = add(
                    node_impulses[binding.nodes[slot]], multiply(
                        soft_anchor_impulses[end], weight));
        }
    }
}

[shader("compute")]
[numthreads(PM_THREADS(256), 1, 1)]
void rope_soft_apply(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> forces, uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<Vec3> impulses, uniform uint count,
    uniform float maximum_acceleration, uniform float maximum_speed,
    uniform float force_scale, uniform float dt,
    uniform float frame_inverse_dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (node >= count || inverse_masses[node] == 0.0f) return;
    Vec3 impulse = clamp_length(
        multiply(impulses[node], inverse_masses[node]),
        maximum_acceleration * dt);
    velocities[node] = clamp_length(
        add(velocities[node], impulse), maximum_speed);
    positions[node] = add(positions[node], multiply(impulse, dt));
    forces[node] = add(forces[node], multiply(
        impulse, force_scale * frame_inverse_dt / inverse_masses[node]));
}

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void rope_soft_sync_attachment(
    uniform PMBuffer<Vec3> rope_positions,
    uniform PMBuffer<Vec3> rope_previous_positions,
    uniform PMBuffer<Vec3> rope_velocities, uniform uint rope_count,
    uniform PMBuffer<Vec3> rope_anchor_targets,
    uniform PMBuffer<Vec3> anchor_positions,
    uniform PMBuffer<Vec3> anchor_velocities, uniform uint end,
    uniform float dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u || pmGroupThreadIndex().x != 0u ||
        rope_count == 0u || dt <= 0.0f) return;
    uint node = end == 0u ? 0u : rope_count - 1u;
    Vec3 position = anchor_positions[end];
    Vec3 velocity = anchor_velocities[end];
    rope_positions[node] = position;
    rope_previous_positions[node] = subtract(
        position, multiply(velocity, dt));
    rope_velocities[node] = velocity;
    rope_anchor_targets[end] = position;
}

[shader("compute")]
[numthreads(PM_THREADS(1), 1, 1)]
void rope_soft_accumulate_attachment_reaction(
    uniform PMBuffer<Vec3> previous_rope_velocities,
    uniform PMBuffer<Vec3> current_rope_velocities,
    uniform PMBuffer<float> rope_inverse_masses, uniform uint rope_count,
    uniform uint attached_end_mask, uniform uint attached_end_count,
    uniform Vec3 gravity, uniform float dt,
    uniform PMBuffer<Vec3> soft_forces,
    uniform PMBuffer<uint> indices,
    uniform PMBuffer<SoftBodySurfaceBinding> bindings,
    uniform uint triangle, uniform Vec3 weights PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u || pmGroupThreadIndex().x != 0u ||
        attached_end_count == 0u || dt <= 0.0f) return;
    Vec3 externalImpulse = {};
    for (uint node = 0u; node < rope_count; ++node)
    {
        bool attached = (node == 0u && (attached_end_mask & 1u) != 0u) ||
            (node + 1u == rope_count && (attached_end_mask & 2u) != 0u);
        float inverseMass = rope_inverse_masses[node];
        if (attached || inverseMass <= 0.0f) continue;
        float mass = 1.0f / inverseMass;
        externalImpulse = add(externalImpulse, subtract(
            multiply(subtract(current_rope_velocities[node],
                previous_rope_velocities[node]), mass),
            multiply(gravity, mass * dt)));
    }
    Vec3 reaction = multiply(externalImpulse,
        -1.0f / (dt * float(attached_end_count)));
    float w[3] = {weights.x, weights.y, weights.z};
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        SoftBodySurfaceBinding binding =
            bindings[indices[3u * triangle + corner]];
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            float weight = w[corner] * binding.weights[slot];
            if (weight > 0.0f)
                soft_forces[binding.nodes[slot]] = add(
                    soft_forces[binding.nodes[slot]],
                    multiply(reaction, weight));
        }
    }
}
