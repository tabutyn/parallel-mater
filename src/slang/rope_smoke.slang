// SPDX-License-Identifier: MIT
// Shared rope/smoke coupling for the CUDA mirror and Metal backend.

#ifdef PM_TARGET_METAL
import smoke_common;
import metal_common;

struct MetalSmokeRopeConstants
{
    float timestep;
    uint smoke_capacity;
    uint rope_count;
    float lifetime;
    float contact_distance;
    float wind_radius;
    float rest_number_density;
    float wind_drag;
    float maximum_wind_acceleration;
    float maximum_rope_speed;
    uint skip_first;
    uint skip_last;
    uint enabled;
    uint grid_resolution;
    uint grid_vertical_resolution;
    float grid_spacing;
    MetalPackedVec3 grid_minimum;
};

struct MetalSegmentPair
{
    float first_fraction;
    float second_fraction;
    float3 first_point;
    float3 second_point;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSmokeRopeSmokePositions : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<MetalPackedVec3>
    metalSmokeRopeSmokePrevious : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSmokeRopeSmokeVelocities : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<float>
    metalSmokeRopeSmokeAges : register(t3);
[[vk::binding(4, 0)]] StructuredBuffer<MetalPackedVec3>
    metalSmokeRopeRopePositions : register(t4);
[[vk::binding(5, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalSmokeRopeRopeVelocities : register(u5);
[[vk::binding(6, 0)]] StructuredBuffer<float>
    metalSmokeRopeRopeInverseMasses : register(t6);
[[vk::binding(7, 0)]] ConstantBuffer<MetalSmokeRopeConstants>
    metalSmokeRopeConstants : register(b7);
[[vk::binding(8, 0)]] StructuredBuffer<MetalSmokeMetadata>
    metalSmokeRopeMetadata : register(t8);
[[vk::binding(9, 0)]] RWStructuredBuffer<float>
    metalSmokeRopeGridDensity : register(u9);
[[vk::binding(10, 0)]] RWStructuredBuffer<float>
    metalSmokeRopeGridFaceVelocity : register(u10);

[ForceInline]
static MetalSegmentPair metalSmokeRopeClosestSegments(
    float3 firstA, float3 firstB, float3 secondA, float3 secondB)
{
    MetalSegmentPair result;
    float3 firstAxis = firstB - firstA;
    float3 secondAxis = secondB - secondA;
    float3 offset = firstA - secondA;
    float firstSize = dot(firstAxis, firstAxis);
    float secondSize = dot(secondAxis, secondAxis);
    float secondOffset = dot(secondAxis, offset);
    if (firstSize <= 1.0e-12f && secondSize <= 1.0e-12f)
    {
        result.first_fraction = 0.0f;
        result.second_fraction = 0.0f;
    }
    else if (firstSize <= 1.0e-12f)
    {
        result.first_fraction = 0.0f;
        result.second_fraction = clamp(
            secondOffset / secondSize, 0.0f, 1.0f);
    }
    else
    {
        float firstOffset = dot(firstAxis, offset);
        if (secondSize <= 1.0e-12f)
        {
            result.second_fraction = 0.0f;
            result.first_fraction = clamp(
                -firstOffset / firstSize, 0.0f, 1.0f);
        }
        else
        {
            float axes = dot(firstAxis, secondAxis);
            float denominator = firstSize * secondSize - axes * axes;
            result.first_fraction = abs(denominator) >
                    1.0e-6f * firstSize * secondSize
                ? clamp((axes * secondOffset - firstOffset * secondSize) /
                    denominator, 0.0f, 1.0f)
                : 0.0f;
            result.second_fraction =
                (axes * result.first_fraction + secondOffset) / secondSize;
            if (result.second_fraction < 0.0f)
            {
                result.second_fraction = 0.0f;
                result.first_fraction = clamp(
                    -firstOffset / firstSize, 0.0f, 1.0f);
            }
            else if (result.second_fraction > 1.0f)
            {
                result.second_fraction = 1.0f;
                result.first_fraction = clamp(
                    (axes - firstOffset) / firstSize, 0.0f, 1.0f);
            }
        }
    }
    result.first_point = firstA + firstAxis * result.first_fraction;
    result.second_point = secondA + secondAxis * result.second_fraction;
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_rope_wind(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalSmokeRopeConstants constants = metalSmokeRopeConstants;
    if (node >= constants.rope_count || constants.enabled == 0u) return;
    uint smokeCount = min(
        metalSmokeRopeMetadata[0].count, constants.smoke_capacity);
    float windRadius = max(constants.wind_radius, 1.0e-5f);
    float windRadiusSquared = windRadius * windRadius;
    float3 gridMinimum = metalLoad(constants.grid_minimum);
    float3 gridMaximum = gridMinimum + constants.grid_spacing * float3(
        float(constants.grid_resolution),
        float(constants.grid_vertical_resolution),
        float(constants.grid_resolution));
    if (metalSmokeRopeRopeInverseMasses[node] <= 0.0f ||
        (node == 0u && constants.skip_first != 0u) ||
        (node + 1u == constants.rope_count && constants.skip_last != 0u))
        return;
    float3 ropePoint = metalLoad(metalSmokeRopeRopePositions[node]);
    float3 flow = float3(0.0f);
    float density = 0.0f;
    bool insideGrid = constants.grid_resolution != 0u &&
        all(ropePoint >= gridMinimum) && all(ropePoint < gridMaximum);
    if (insideGrid)
    {
        flow = metalSmokeSampleFaceVelocity(
            metalSmokeRopeGridFaceVelocity, 0u, ropePoint, gridMinimum,
            constants.grid_spacing, constants.grid_resolution,
            constants.grid_vertical_resolution);
        density = metalSmokeSampleCellScalar(
            metalSmokeRopeGridDensity, ropePoint, gridMinimum,
            constants.grid_spacing, constants.grid_resolution,
            constants.grid_vertical_resolution) *
            constants.rest_number_density;
    }
    else
    {
        for (uint particle = 0u; particle < smokeCount; ++particle)
        {
            if (metalSmokeRopeSmokeAges[particle] >= constants.lifetime)
                continue;
            float3 delta =
                metalLoad(metalSmokeRopeSmokePositions[particle]) - ropePoint;
            float squared = dot(delta, delta);
            if (squared >= windRadiusSquared) continue;
            float q = 1.0f - sqrt(max(squared, 0.0f)) / windRadius;
            float weight = q * q * q;
            flow += metalLoad(
                metalSmokeRopeSmokeVelocities[particle]) * weight;
            density += weight;
        }
        if (density > 1.0e-6f) flow /= density;
    }
    if (density <= 1.0e-6f) return;
    float occupancy = min(1.0f, insideGrid ? density :
        density / max(constants.rest_number_density, 1.0e-6f));
    float response = 1.0f - exp(
        -constants.wind_drag * occupancy * constants.timestep);
    float3 change = metalLimit(
        (flow - metalLoad(metalSmokeRopeRopeVelocities[node])) * response,
        constants.maximum_wind_acceleration * constants.timestep);
    metalSmokeRopeRopeVelocities[node] = metalStore(metalLimit(
        metalLoad(metalSmokeRopeRopeVelocities[node]) + change,
        constants.maximum_rope_speed));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_rope(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalSmokeRopeConstants constants = metalSmokeRopeConstants;
    if (constants.enabled == 0u) return;
    uint smokeCount = min(
        metalSmokeRopeMetadata[0].count, constants.smoke_capacity);
    if (particle >= smokeCount ||
        metalSmokeRopeSmokeAges[particle] >= constants.lifetime)
        return;
    float clearance = max(constants.contact_distance, 1.0e-5f);
    float clearanceSquared = clearance * clearance;
    float3 point = metalLoad(metalSmokeRopeSmokePositions[particle]);
    float3 before = metalLoad(metalSmokeRopeSmokePrevious[particle]);
    float earliest = 2.0f;
    float nearestSquared = clearanceSquared;
    uint bestSegment = constants.rope_count;
    float3 smokeHit = float3(0.0f);
    float3 ropeHit = float3(0.0f);
    for (uint segment = 0u; segment + 1u < constants.rope_count; ++segment)
    {
        float3 first = metalLoad(metalSmokeRopeRopePositions[segment]);
        float3 second = metalLoad(metalSmokeRopeRopePositions[segment + 1u]);
        if (max(point.x, before.x) < min(first.x, second.x) - clearance ||
            min(point.x, before.x) > max(first.x, second.x) + clearance ||
            max(point.y, before.y) < min(first.y, second.y) - clearance ||
            min(point.y, before.y) > max(first.y, second.y) + clearance ||
            max(point.z, before.z) < min(first.z, second.z) - clearance ||
            min(point.z, before.z) > max(first.z, second.z) + clearance)
            continue;
        MetalSegmentPair pair = metalSmokeRopeClosestSegments(
            before, point, first, second);
        float3 separation = pair.first_point - pair.second_point;
        float squared = dot(separation, separation);
        if (squared >= clearanceSquared ||
            pair.first_fraction > earliest + 1.0e-6f ||
            (abs(pair.first_fraction - earliest) <= 1.0e-6f &&
                squared >= nearestSquared))
            continue;
        earliest = pair.first_fraction;
        nearestSquared = squared;
        bestSegment = segment;
        smokeHit = pair.first_point;
        ropeHit = pair.second_point;
    }
    if (bestSegment == constants.rope_count) return;
    float3 first = metalLoad(metalSmokeRopeRopePositions[bestSegment]);
    float3 second = metalLoad(metalSmokeRopeRopePositions[bestSegment + 1u]);
    float3 axis = second - first;
    float axisSquared = max(dot(axis, axis), 1.0e-12f);
    float ropeFraction = clamp(
        dot(ropeHit - first, axis) / axisSquared, 0.0f, 1.0f);
    float3 ropeVelocity =
        metalLoad(metalSmokeRopeRopeVelocities[bestSegment]) *
            (1.0f - ropeFraction) +
        metalLoad(metalSmokeRopeRopeVelocities[bestSegment + 1u]) *
            ropeFraction;
    float3 normal = smokeHit - ropeHit;
    if (dot(normal, normal) < 1.0e-10f) normal = before - ropeHit;
    if (dot(normal, normal) < 1.0e-10f)
    {
        float3 relative =
            metalLoad(metalSmokeRopeSmokeVelocities[particle]) - ropeVelocity;
        normal = -relative + axis * (dot(relative, axis) / axisSquared);
    }
    float normalSquared = dot(normal, normal);
    normal = normalSquared > 1.0e-12f
        ? normal * rsqrt(normalSquared) : float3(0.0f, 1.0f, 0.0f);
    metalSmokeRopeSmokePositions[particle] =
        metalStore(ropeHit + normal * clearance);
    float3 relative =
        metalLoad(metalSmokeRopeSmokeVelocities[particle]) - ropeVelocity;
    relative -= normal * min(0.0f, dot(relative, normal));
    metalSmokeRopeSmokeVelocities[particle] =
        metalStore(ropeVelocity + relative);
}

#else
import types;
import math;
import device_intrinsics;
import rope;
import geometry_smoke;

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_rope_wind(
    uniform Vec3* rope_positions, uniform Vec3* rope_velocities,
    uniform uint rope_count, uniform float rope_maximum_speed,
    uniform bool first_attached, uniform bool last_attached,
    uniform bool soft_first, uniform bool soft_last,
    uniform SmokeOptions smoke,
    uniform Vec3* smoke_positions, uniform Vec3* smoke_velocities,
    uniform uint64_t* keys, uniform uint* indices,
    uniform SmokeGridField grid, uniform float drag,
    uniform float maximum_acceleration, uniform float dt)
{
    uint node = pmThreadIndex();
    if (node >= rope_count ||
        (node == 0u && (first_attached || soft_first)) ||
        (node + 1u == rope_count && (last_attached || soft_last))) return;
    rope_velocities[node] = clamp_length(add(rope_velocities[node],
        smoke_wind_delta(rope_positions[node], rope_velocities[node],
            smoke_positions, smoke_velocities, keys, indices, smoke, grid,
            drag, maximum_acceleration, dt)), rope_maximum_speed);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_rope_contact(
    uniform Vec3* positions, uniform Vec3* previous_positions,
    uniform Vec3* velocities, uniform float* ages, uniform uint count,
    uniform float lifetime, uniform Vec3* rope_positions,
    uniform Vec3* rope_velocities, uniform uint rope_count,
    uniform Vec3* minimum,
    uniform Vec3* maximum, uniform float clearance)
{
    uint particle = pmThreadIndex();
    if (particle >= count || ages[particle] >= lifetime) return;
    Vec3 point = positions[particle];
    Vec3 before = previous_positions[particle];
    Vec3 low = *minimum;
    Vec3 high = *maximum;
    if (max(point.x, before.x) < low.x - clearance ||
        min(point.x, before.x) > high.x + clearance ||
        max(point.y, before.y) < low.y - clearance ||
        min(point.y, before.y) > high.y + clearance ||
        max(point.z, before.z) < low.z - clearance ||
        min(point.z, before.z) > high.z + clearance) return;
    float earliest = 2.0f;
    float nearestSquared = clearance * clearance;
    uint segment = rope_count;
    Vec3 smokeHit = {};
    Vec3 ropeHit = {};
    Vec3 path = subtract(point, before);
    float pathSquared = length_squared(path);
    for (uint index = 0u; index + 1u < rope_count; ++index)
    {
        Vec3 a = rope_positions[index];
        Vec3 b = rope_positions[index + 1u];
        if (max(point.x, before.x) < min(a.x, b.x) - clearance ||
            min(point.x, before.x) > max(a.x, b.x) + clearance ||
            max(point.y, before.y) < min(a.y, b.y) - clearance ||
            min(point.y, before.y) > max(a.y, b.y) + clearance ||
            max(point.z, before.z) < min(a.z, b.z) - clearance ||
            min(point.z, before.z) > max(a.z, b.z) + clearance) continue;
        Vec3 onPath = {};
        Vec3 onRope = {};
        closest_segments(before, point, a, b, onPath, onRope);
        float distanceSquared = length_squared(subtract(onPath, onRope));
        if (distanceSquared >= clearance * clearance) continue;
        float fraction = pathSquared > 1.0e-12f
            ? clamp_scalar(dot(subtract(onPath, before), path) /
                pathSquared, 0.0f, 1.0f)
            : 0.0f;
        if (fraction > earliest + 1.0e-6f ||
            (abs(fraction - earliest) <= 1.0e-6f &&
             distanceSquared >= nearestSquared)) continue;
        earliest = fraction;
        nearestSquared = distanceSquared;
        segment = index;
        smokeHit = onPath;
        ropeHit = onRope;
    }
    if (segment == rope_count) return;
    Vec3 a = rope_positions[segment];
    Vec3 b = rope_positions[segment + 1u];
    Vec3 axis = subtract(b, a);
    float fraction = clamp_scalar(dot(subtract(ropeHit, a), axis) /
        max(length_squared(axis), 1.0e-12f), 0.0f, 1.0f);
    Vec3 ropeVelocity = add(
        multiply(rope_velocities[segment], 1.0f - fraction),
        multiply(rope_velocities[segment + 1u], fraction));
    Vec3 normal = subtract(smokeHit, ropeHit);
    if (length_squared(normal) < 1.0e-10f)
        normal = subtract(before, ropeHit);
    if (length_squared(normal) < 1.0e-10f)
    {
        Vec3 relativeVelocity = subtract(velocities[particle], ropeVelocity);
        normal = subtract(multiply(relativeVelocity, -1.0f),
            multiply(axis, -dot(relativeVelocity, axis) /
                max(length_squared(axis), 1.0e-12f)));
    }
    normal = normalized_or(normal, {0.0f, 1.0f, 0.0f});
    positions[particle] = add(ropeHit, multiply(normal, clearance));
    Vec3 relative = subtract(velocities[particle], ropeVelocity);
    relative = subtract(relative,
        multiply(normal, min(0.0f, dot(relative, normal))));
    velocities[particle] = add(ropeVelocity, relative);
}
#endif
