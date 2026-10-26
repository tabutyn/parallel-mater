#include "dispatch_entry.slang"
#include "coupling_spatial.slangh"
// SPDX-License-Identifier: MIT
// Shared fluid/soft-body coupling kernels.

import types;
import math;
import device_intrinsics;
import geometry_fluid;
import graphics_resources;

public struct FluidSoftNearest
{
    public uint triangle;
    public Vec3 point;
    public Vec3 normal;
    public Vec3 weights;
    public float penetration;
};

float fluid_soft_bounds_distance(Vec3 point, BvhNode node)
{
    Vec3 delta = component_max(component_max(
        subtract(node.minimum, point), subtract(point, node.maximum)), {});
    return length_squared(delta);
}

bool fluid_soft_inside(
    Vec3 point, PMBuffer<Vec3> surface, PMBuffer<uint> indices, uint index_count,
    PMBuffer<BvhNode> tree, PMBuffer<uint> order)
{
    Vec3 direction = {1.0f, 0.371f, 0.173f};
    uint stack[64] = {};
    uint size = 1u;
    int winding = 0;
    bool ambiguous = false;
    while (size != 0u)
    {
        BvhNode node = tree[stack[--size]];
        Vec3 low = subtract(node.minimum, point);
        Vec3 high = subtract(node.maximum, point);
        float begin = max(0.0f, max(low.x,
            max(low.y / direction.y, low.z / direction.z)));
        float end = min(high.x,
            min(high.y / direction.y, high.z / direction.z));
        if (end < begin) continue;
        if (node.triangle_count == 0u)
        {
            stack[size++] = node.left;
            stack[size++] = node.right;
            continue;
        }
        for (uint item = 0u; item < node.triangle_count; ++item)
        {
            uint base = order[node.first_triangle + item] * 3u;
            Vec3 a = surface[indices[base]];
            Vec3 edge1 = subtract(surface[indices[base + 1u]], a);
            Vec3 edge2 = subtract(surface[indices[base + 2u]], a);
            Vec3 h = cross(direction, edge2), s = subtract(point, a);
            float determinant = dot(edge1, h);
            if (abs(determinant) < 1.0e-12f) continue;
            float u = dot(s, h) / determinant;
            Vec3 q = cross(s, edge1);
            float v = dot(direction, q) / determinant;
            float t = dot(edge2, q) / determinant;
            if (t < 0.0f || u < -1.0e-5f || v < -1.0e-5f ||
                u + v > 1.00001f) continue;
            if (u < 1.0e-5f || v < 1.0e-5f || u + v > 0.99999f ||
                t < 1.0e-7f) ambiguous = true;
            winding += determinant < 0.0f ? 1 : -1;
        }
    }
    if (!ambiguous) return winding != 0;
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

public FluidSoftNearest fluid_soft_nearest(
    Vec3 point, Vec3 start, PMBuffer<Vec3> surface, PMBuffer<Vec3> previous,
    PMBuffer<uint> indices, uint index_count, float orientation, float distance,
    PMBuffer<Vec3> bounds, bool sweep, PMBuffer<BvhNode> tree, PMBuffer<uint> order)
{
    FluidSoftNearest result = {};
    result.triangle = kInvalidDense;
    Vec3 margin = {distance, distance, distance};
    if (!bounds_overlap(component_min(point, start), component_max(point, start),
        subtract(bounds[0], margin), add(bounds[1], margin))) return result;
    float best = 3.402823466e+38f, earliest = 2.0f;
    uint stack[64] = {};
    uint size = 1u;
    while (size != 0u)
    {
        BvhNode node = tree[stack[--size]];
        if (fluid_soft_bounds_distance(point, node) > best &&
            (!sweep || !fluid_segment_bounds(start, point, node, distance)))
            continue;
        if (node.triangle_count == 0u)
        {
            bool leftNearer = fluid_soft_bounds_distance(point, tree[node.left]) <
                fluid_soft_bounds_distance(point, tree[node.right]);
            stack[size++] = leftNearer ? node.right : node.left;
            stack[size++] = leftNearer ? node.left : node.right;
            continue;
        }
        for (uint item = 0u; item < node.triangle_count; ++item)
        {
            uint base = order[node.first_triangle + item] * 3u;
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
    }
    if (result.triangle == kInvalidDense) return result;
    if (earliest > 1.0f)
    {
        float radius = sqrt(best);
        bool faceInterior = result.weights.x > 1.0e-4f &&
            result.weights.y > 1.0e-4f && result.weights.z > 1.0e-4f;
        bool inside = faceInterior
            ? dot(subtract(point, result.point), result.normal) < 0.0f
            : fluid_soft_inside(point, surface, indices, index_count, tree, order);
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

bool fluid_soft_inside_shared(
    Vec3 point, PMBuffer<Vec3> surface, PMBuffer<uint> indices,
    uint index_count)
{
    // Only used after a shared candidate produced a near boundary hit.  The
    // solid-angle fallback classifies containment without rebuilding a
    // private per-coupling BVH in every solve sweep.
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

FluidSoftNearest fluid_soft_nearest_shared(
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
    FluidSoftNearest result = {};
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
            : fluid_soft_inside_shared(point, surface, indices, index_count);
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
[numthreads(128, 1, 1)]
void fluid_soft_refit(
    uniform PMBuffer<Vec3> surface, uniform PMBuffer<Vec3> previous, uniform PMBuffer<uint> indices,
    uniform PMBuffer<uint> order, uniform PMBuffer<BvhNode> tree, uniform PMBuffer<uint> parents,
    uniform PMBuffer<uint> ready, uniform uint tree_count, uniform PMBuffer<Vec3> bounds PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index >= tree_count || tree[index].triangle_count == 0u) return;
    Vec3 low = {3.402823466e+38f, 3.402823466e+38f, 3.402823466e+38f};
    Vec3 high = {-3.402823466e+38f, -3.402823466e+38f, -3.402823466e+38f};
    for (uint item = 0u; item < tree[index].triangle_count; ++item)
    {
        uint base = order[tree[index].first_triangle + item] * 3u;
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            uint vertex = indices[base + corner];
            low = component_min(low,
                component_min(surface[vertex], previous[vertex]));
            high = component_max(high,
                component_max(surface[vertex], previous[vertex]));
        }
    }
    tree[index].minimum = low;
    tree[index].maximum = high;
    while (parents[index] != kInvalidDense)
    {
        index = parents[index];
        if (pmAtomicAdd(ready + index, 1u) == 0u) return;
        uint left = tree[index].left, right = tree[index].right;
        tree[index].minimum = component_min(
            tree[left].minimum, tree[right].minimum);
        tree[index].maximum = component_max(
            tree[left].maximum, tree[right].maximum);
    }
    bounds[0] = tree[0].minimum;
    bounds[1] = tree[0].maximum;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fluid_soft_detect(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> previous,
    uniform PMBuffer<Vec3> velocities, uniform PMBuffer<uint> count, uniform PMBuffer<Vec3> surface,
    uniform PMBuffer<Vec3> old_surface, uniform PMBuffer<uint> indices,
    uniform uint index_count, uniform PMBuffer<SoftBodySurfaceBinding> bindings,
    uniform PMBuffer<Vec3> soft_velocities, uniform float orientation,
    uniform float distance, uniform bool sweep,
    uniform PMBuffer<uint> triangle_generations,
    uniform uint soft_owner_slot, uniform uint soft_owner_generation,
    uniform uint soft_topology_revision,
    uniform PMBuffer<CouplingSpatialLeafMetadata> spatial_metadata,
    uniform PMByteBuffer spatial_admission, uniform uint spatial_leaf_capacity,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_sources,
    uniform uint candidates_per_query, uniform PMBuffer<uint> spatial_status,
    uniform PMBuffer<FluidSoftContact> contacts, uniform PMBuffer<uint> counts,
    uniform PMBuffer<uint> contact_count, uniform PMBuffer<float> maximum_penetration PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= *count) return;
    FluidSoftContact contact = {};
    FluidSoftNearest hit = fluid_soft_nearest_shared(positions[particle],
        previous[particle], surface, old_surface, indices, index_count,
        orientation, distance, sweep, triangle_generations, soft_owner_slot,
        soft_owner_generation, soft_topology_revision, spatial_metadata,
        spatial_admission, spatial_leaf_capacity, candidate_counts,
        candidate_sources, candidates_per_query, particle, spatial_status);
    if (hit.triangle != kInvalidDense)
    {
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
        Vec3 surfaceVelocity = {};
        for (uint index = 0u; index < contact.count; ++index)
        {
            pmAtomicAdd(counts + contact.nodes[index], 1u);
            surfaceVelocity = add(surfaceVelocity, multiply(
                soft_velocities[contact.nodes[index]], contact.weights[index]));
        }
        contact.normal = hit.normal;
        contact.relative_velocity = subtract(
            velocities[particle], surfaceVelocity);
        contact.penetration = hit.penetration;
        pmAtomicAdd(contact_count, 1u);
        pmAtomicMaxPositiveFloat(maximum_penetration, hit.penetration);
    }
    contacts[particle] = contact;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fluid_soft_solve(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<Vec3> accelerations, uniform PMBuffer<float> foam, uniform PMBuffer<uint> count,
    uniform PMBuffer<FluidSoftContact> contacts, uniform PMBuffer<uint> counts,
    uniform PMBuffer<float> inverse_masses, uniform PMBuffer<Vec3> soft_velocities,
    uniform float maximum_soft_speed, uniform float inverse_particle_mass,
    uniform float friction, uniform float maximum_projection,
    uniform float dt, uniform PMBuffer<Vec3> node_positions,
    uniform PMBuffer<Vec3> node_impulses PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= *count) return;
    FluidSoftContact contact = contacts[particle];
    if (contact.count == 0u) return;
    float denominator = inverse_particle_mass;
    for (uint index = 0u; index < contact.count; ++index)
    {
        uint node = contact.nodes[index];
        denominator += contact.weights[index] * inverse_masses[node] *
            float(counts[node]);
    }
    float normalSpeed = dot(contact.relative_velocity, contact.normal);
    float normalImpulse = max(0.0f, -normalSpeed) / denominator;
    Vec3 tangent = subtract(contact.relative_velocity,
        multiply(contact.normal, normalSpeed));
    Vec3 impulse = subtract(multiply(contact.normal, normalImpulse),
        clamp_length(multiply(tangent, 1.0f / denominator),
            friction * normalImpulse));
    float scale = 1.0f;
    for (uint index = 0u; index < contact.count; ++index)
    {
        uint node = contact.nodes[index];
        Vec3 velocity = soft_velocities[node];
        Vec3 delta = multiply(impulse, -contact.weights[index] *
            inverse_masses[node] * float(counts[node]));
        float a = length_squared(delta);
        if (a <= 1.0e-20f) continue;
        float b = dot(velocity, delta);
        float c = min(0.0f, length_squared(velocity) -
            maximum_soft_speed * maximum_soft_speed);
        float limit = (-b + sqrt(max(0.0f, b * b - a * c))) / a;
        scale = min(scale, max(0.0f, limit));
    }
    impulse = multiply(impulse, scale);
    float lambda = min(contact.penetration, maximum_projection) / denominator;
    positions[particle] = add(positions[particle],
        multiply(contact.normal, inverse_particle_mass * lambda));
    Vec3 velocityDelta = multiply(impulse, inverse_particle_mass);
    velocities[particle] = add(velocities[particle], velocityDelta);
    accelerations[particle] = add(accelerations[particle],
        multiply(velocityDelta, 1.0f / dt));
    foam[particle] = max(foam[particle],
        min(1.0f, normalImpulse * inverse_particle_mass * 0.1f));
    for (uint index = 0u; index < contact.count; ++index)
    {
        uint node = contact.nodes[index];
        atomic_add(node_positions + node, multiply(contact.normal,
            -contact.weights[index] * inverse_masses[node] * lambda));
        atomic_add(node_impulses + node,
            multiply(impulse, -contact.weights[index]));
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fluid_soft_apply(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> velocities,
    uniform PMBuffer<float> inverse_masses, uniform PMBuffer<Vec3> position_deltas,
    uniform PMBuffer<Vec3> impulses, uniform PMBuffer<Vec3> forces, uniform uint count,
    uniform float inverse_frame_dt PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint node = pmThreadIndex();
    if (node >= count) return;
    positions[node] = add(positions[node], position_deltas[node]);
    velocities[node] = add(velocities[node],
        multiply(impulses[node], inverse_masses[node]));
    forces[node] = add(forces[node],
        multiply(impulses[node], inverse_frame_dt));
}

[shader("compute")]
[numthreads(128, 1, 1)]
void fluid_soft_recover(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> previous, uniform PMBuffer<uint> count,
    uniform PMBuffer<Vec3> surface, uniform PMBuffer<Vec3> old_surface,
    uniform PMBuffer<uint> indices, uniform uint index_count, uniform float orientation,
    uniform float distance, uniform bool sweep,
    uniform PMBuffer<uint> triangle_generations,
    uniform uint soft_owner_slot, uniform uint soft_owner_generation,
    uniform uint soft_topology_revision,
    uniform PMBuffer<CouplingSpatialLeafMetadata> spatial_metadata,
    uniform PMByteBuffer spatial_admission, uniform uint spatial_leaf_capacity,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidate_sources,
    uniform uint candidates_per_query,
    uniform PMBuffer<uint> spatial_status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint particle = pmThreadIndex();
    if (particle >= *count) return;
    FluidSoftNearest hit = fluid_soft_nearest_shared(positions[particle],
        previous[particle], surface, old_surface, indices, index_count,
        orientation, distance, sweep, triangle_generations, soft_owner_slot,
        soft_owner_generation, soft_topology_revision, spatial_metadata,
        spatial_admission, spatial_leaf_capacity, candidate_counts,
        candidate_sources, candidates_per_query, particle, spatial_status);
    if (hit.triangle != kInvalidDense)
        positions[particle] = add(positions[particle],
            multiply(hit.normal, hit.penetration));
}
