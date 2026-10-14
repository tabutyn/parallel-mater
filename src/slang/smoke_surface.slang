// SPDX-License-Identifier: MIT
// Production smoke/surface coupling for Metal.

import smoke_common;
import metal_common;

struct MetalSmokeSurfaceConstants
{
    float timestep;
    uint target_count;
    uint surface_index_count;
    uint mode;
    float lifetime;
    float contact_distance;
    float wind_radius;
    float rest_number_density;
    float wind_drag;
    float maximum_wind_acceleration;
    float maximum_target_speed;
    float maximum_smoke_speed;
    uint enabled;
    uint grid_resolution;
    uint grid_vertical_resolution;
    float grid_spacing;
    MetalPackedVec3 grid_minimum;
    float grid_kinematic_viscosity;
    float grid_les_coefficient;
    uint command;
    uint raster_triangle_base;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeSurfaceSmokePositions : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<MetalPackedVec3>
    smokeSurfaceSmokePrevious : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeSurfaceSmokeVelocities : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<float>
    smokeSurfaceSmokeAges : register(t3);
[[vk::binding(4, 0)]] StructuredBuffer<MetalSmokeMetadata>
    smokeSurfaceSmokeMetadata : register(t4);
[[vk::binding(5, 0)]] StructuredBuffer<MetalPackedVec3>
    smokeSurfaceTargetPositions : register(t5);
[[vk::binding(6, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeSurfaceTargetVelocities : register(u6);
[[vk::binding(7, 0)]] StructuredBuffer<float>
    smokeSurfaceTargetInverseMasses : register(t7);
[[vk::binding(8, 0)]] StructuredBuffer<MetalPackedVec3>
    smokeSurfacePositions : register(t8);
[[vk::binding(9, 0)]] StructuredBuffer<uint>
    smokeSurfaceIndices : register(t9);
[[vk::binding(10, 0)]] StructuredBuffer<uint>
    smokeSurfaceSources : register(t10);
[[vk::binding(11, 0)]] StructuredBuffer<MetalSurfaceBinding>
    smokeSurfaceBindings : register(t11);
[[vk::binding(12, 0)]] ConstantBuffer<MetalSmokeSurfaceConstants>
    smokeSurfaceConstants : register(b12);
[[vk::binding(13, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeSurfaceGridWallVelocity : register(u13);
[[vk::binding(14, 0)]] RWStructuredBuffer<uint>
    smokeSurfaceCellNearestTriangle : register(u14);
// Positive float values are atomically minimized through their uint bits.
[[vk::binding(15, 0)]] RWStructuredBuffer<uint>
    smokeSurfaceFaceBoundaryBits : register(u15);
[[vk::binding(16, 0)]] RWStructuredBuffer<float>
    smokeSurfaceGridPressure : register(u16);
[[vk::binding(17, 0)]] RWStructuredBuffer<float>
    smokeSurfaceGridDensity : register(u17);
[[vk::binding(18, 0)]] RWStructuredBuffer<float>
    smokeSurfaceGridFaceVelocity : register(u18);
[[vk::binding(19, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeSurfaceTargetForces : register(u19);
[[vk::binding(20, 0)]] RWStructuredBuffer<uint>
    smokeSurfaceFaceNearestTriangle : register(u20);
[[vk::binding(21, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeSurfaceFaceNormal : register(u21);

[ForceInline]
static uint smokeSurfaceFaceTotal(uint n, uint height)
{
    return metalSmokeFaceCount(0u, n, height) +
        metalSmokeFaceCount(1u, n, height) +
        metalSmokeFaceCount(2u, n, height);
}

[ForceInline]
static float3 smokeSurfaceNormalizedOr(float3 value, float3 fallback)
{
    float squared = dot(value, value);
    return squared > 1.0e-20f ? value * rsqrt(squared) : fallback;
}

[ForceInline]
static float3 smokeSurfaceVertexVelocity(uint surface)
{
    MetalSmokeSurfaceConstants constants = smokeSurfaceConstants;
    if (constants.mode == 0u)
        return metalLoad(smokeSurfaceTargetVelocities[
            smokeSurfaceSources[surface]]);
    MetalSurfaceBinding binding = smokeSurfaceBindings[surface];
    float3 velocity = float3(0.0f);
    for (uint slot = 0u; slot < 4u; ++slot)
        velocity += metalLoad(smokeSurfaceTargetVelocities[
            binding.nodes[slot]]) * binding.weights[slot];
    return velocity;
}

[ForceInline]
static void smokeSurfaceRasterTriangleMark(
    float3 a, float3 b, float3 c, uint globalTriangle)
{
    MetalSmokeSurfaceConstants constants = smokeSurfaceConstants;
    uint resolution = constants.grid_resolution;
    uint vertical = constants.grid_vertical_resolution;
    float spacing = constants.grid_spacing;
    float3 minimum = metalLoad(constants.grid_minimum);
    float inverseSpacing = 1.0f / spacing;
    float radius = 0.55f * spacing;
    float radiusSquared = radius * radius;
    float3 lower = min(a, min(b, c)) - radius;
    float3 upper = max(a, max(b, c)) + radius;
    int x0 = max(0, int(floor((lower.x - minimum.x) * inverseSpacing)));
    int y0 = max(0, int(floor((lower.y - minimum.y) * inverseSpacing)));
    int z0 = max(0, int(floor((lower.z - minimum.z) * inverseSpacing)));
    int x1 = min(int(resolution) - 1,
        int(floor((upper.x - minimum.x) * inverseSpacing)));
    int y1 = min(int(vertical) - 1,
        int(floor((upper.y - minimum.y) * inverseSpacing)));
    int z1 = min(int(resolution) - 1,
        int(floor((upper.z - minimum.z) * inverseSpacing)));
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
            {
                float3 point = minimum +
                    (float3(float(x), float(y), float(z)) + 0.5f) * spacing;
                float3 closest = metalClosestPointTriangle(point, a, b, c);
                float3 delta = point - closest;
                if (dot(delta, delta) > radiusSquared) continue;
                uint cell = uint(x) + resolution *
                    (uint(y) + vertical * uint(z));
                InterlockedMin(
                    smokeSurfaceCellNearestTriangle[cell], globalTriangle);
            }
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        float3 offset = 0.5f;
        if (axis == 0u) offset.x = 0.0f;
        if (axis == 1u) offset.y = 0.0f;
        if (axis == 2u) offset.z = 0.0f;
        int sx = int(resolution) + (axis == 0u ? 1 : 0);
        int sy = int(vertical) + (axis == 1u ? 1 : 0);
        int sz = int(resolution) + (axis == 2u ? 1 : 0);
        int fx0 = max(0, int(floor(
            (lower.x - minimum.x) * inverseSpacing - offset.x)));
        int fy0 = max(0, int(floor(
            (lower.y - minimum.y) * inverseSpacing - offset.y)));
        int fz0 = max(0, int(floor(
            (lower.z - minimum.z) * inverseSpacing - offset.z)));
        int fx1 = min(sx - 1, int(floor(
            (upper.x - minimum.x) * inverseSpacing - offset.x)));
        int fy1 = min(sy - 1, int(floor(
            (upper.y - minimum.y) * inverseSpacing - offset.y)));
        int fz1 = min(sz - 1, int(floor(
            (upper.z - minimum.z) * inverseSpacing - offset.z)));
        for (int z = fz0; z <= fz1; ++z)
            for (int y = fy0; y <= fy1; ++y)
                for (int x = fx0; x <= fx1; ++x)
                {
                    float3 point = minimum +
                        (float3(float(x), float(y), float(z)) + offset) *
                            spacing;
                    float distanceValue = length(
                        point - metalClosestPointTriangle(point, a, b, c));
                    if (distanceValue > radius) continue;
                    uint face = metalSmokeFaceIndex(
                        axis, x, y, z, resolution, vertical);
                    InterlockedMin(smokeSurfaceFaceBoundaryBits[face],
                        asuint(clamp(distanceValue / radius, 0.0f, 1.0f)));
                }
    }
}

[ForceInline]
static void smokeSurfaceSelectTriangleFaces(
    float3 a, float3 b, float3 c, uint globalTriangle)
{
    MetalSmokeSurfaceConstants constants = smokeSurfaceConstants;
    uint resolution = constants.grid_resolution;
    uint vertical = constants.grid_vertical_resolution;
    float spacing = constants.grid_spacing;
    float3 minimum = metalLoad(constants.grid_minimum);
    float inverseSpacing = 1.0f / spacing;
    float radius = 0.55f * spacing;
    float3 lower = min(a, min(b, c)) - radius;
    float3 upper = max(a, max(b, c)) + radius;
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        float3 offset = 0.5f;
        if (axis == 0u) offset.x = 0.0f;
        if (axis == 1u) offset.y = 0.0f;
        if (axis == 2u) offset.z = 0.0f;
        int sx = int(resolution) + (axis == 0u ? 1 : 0);
        int sy = int(vertical) + (axis == 1u ? 1 : 0);
        int sz = int(resolution) + (axis == 2u ? 1 : 0);
        int x0 = max(0, int(floor(
            (lower.x - minimum.x) * inverseSpacing - offset.x)));
        int y0 = max(0, int(floor(
            (lower.y - minimum.y) * inverseSpacing - offset.y)));
        int z0 = max(0, int(floor(
            (lower.z - minimum.z) * inverseSpacing - offset.z)));
        int x1 = min(sx - 1, int(floor(
            (upper.x - minimum.x) * inverseSpacing - offset.x)));
        int y1 = min(sy - 1, int(floor(
            (upper.y - minimum.y) * inverseSpacing - offset.y)));
        int z1 = min(sz - 1, int(floor(
            (upper.z - minimum.z) * inverseSpacing - offset.z)));
        for (int z = z0; z <= z1; ++z)
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                {
                    float3 point = minimum +
                        (float3(float(x), float(y), float(z)) + offset) *
                            spacing;
                    float distanceValue = length(
                        point - metalClosestPointTriangle(point, a, b, c));
                    if (distanceValue > radius) continue;
                    uint face = metalSmokeFaceIndex(
                        axis, x, y, z, resolution, vertical);
                    float aperture =
                        clamp(distanceValue / radius, 0.0f, 1.0f);
                    if (aperture <=
                        asfloat(smokeSurfaceFaceBoundaryBits[face]) + 1.0e-7f)
                        InterlockedMin(
                            smokeSurfaceFaceNearestTriangle[face],
                            globalTriangle);
                }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_surface_grid_raster(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint triangle = dispatchThreadId.x;
    MetalSmokeSurfaceConstants constants = smokeSurfaceConstants;
    if (triangle >= constants.surface_index_count / 3u ||
        constants.enabled == 0u || constants.grid_resolution == 0u ||
        constants.grid_spacing <= 0.0f)
        return;
    uint base = triangle * 3u;
    smokeSurfaceRasterTriangleMark(
        metalLoad(smokeSurfacePositions[smokeSurfaceIndices[base]]),
        metalLoad(smokeSurfacePositions[smokeSurfaceIndices[base + 1u]]),
        metalLoad(smokeSurfacePositions[smokeSurfaceIndices[base + 2u]]),
        constants.raster_triangle_base + triangle);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_surface_grid_select(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint triangle = dispatchThreadId.x;
    MetalSmokeSurfaceConstants constants = smokeSurfaceConstants;
    if (triangle >= constants.surface_index_count / 3u ||
        constants.enabled == 0u || constants.grid_resolution == 0u ||
        constants.grid_spacing <= 0.0f)
        return;
    uint base = triangle * 3u;
    smokeSurfaceSelectTriangleFaces(
        metalLoad(smokeSurfacePositions[smokeSurfaceIndices[base]]),
        metalLoad(smokeSurfacePositions[smokeSurfaceIndices[base + 1u]]),
        metalLoad(smokeSurfacePositions[smokeSurfaceIndices[base + 2u]]),
        constants.raster_triangle_base + triangle);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_surface_grid_resolve(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint triangle = dispatchThreadId.x;
    MetalSmokeSurfaceConstants constants = smokeSurfaceConstants;
    if (triangle >= constants.surface_index_count / 3u ||
        constants.enabled == 0u || constants.grid_resolution == 0u ||
        constants.grid_spacing <= 0.0f)
        return;
    uint base = triangle * 3u;
    uint ia = smokeSurfaceIndices[base];
    uint ib = smokeSurfaceIndices[base + 1u];
    uint ic = smokeSurfaceIndices[base + 2u];
    float3 a = metalLoad(smokeSurfacePositions[ia]);
    float3 b = metalLoad(smokeSurfacePositions[ib]);
    float3 c = metalLoad(smokeSurfacePositions[ic]);
    uint globalTriangle = constants.raster_triangle_base + triangle;
    uint resolution = constants.grid_resolution;
    uint vertical = constants.grid_vertical_resolution;
    float spacing = constants.grid_spacing;
    float inverseSpacing = 1.0f / spacing;
    float radius = 0.55f * spacing;
    float radiusSquared = radius * radius;
    float3 minimum = metalLoad(constants.grid_minimum);
    float3 lower = min(a, min(b, c)) - radius;
    float3 upper = max(a, max(b, c)) + radius;
    int x0 = max(0, int(floor((lower.x - minimum.x) * inverseSpacing)));
    int y0 = max(0, int(floor((lower.y - minimum.y) * inverseSpacing)));
    int z0 = max(0, int(floor((lower.z - minimum.z) * inverseSpacing)));
    int x1 = min(int(resolution) - 1,
        int(floor((upper.x - minimum.x) * inverseSpacing)));
    int y1 = min(int(vertical) - 1,
        int(floor((upper.y - minimum.y) * inverseSpacing)));
    int z1 = min(int(resolution) - 1,
        int(floor((upper.z - minimum.z) * inverseSpacing)));
    uint vertices[3];
    vertices[0] = ia;
    vertices[1] = ib;
    vertices[2] = ic;
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
            {
                uint cell = uint(x) + resolution *
                    (uint(y) + vertical * uint(z));
                if (smokeSurfaceCellNearestTriangle[cell] != globalTriangle)
                    continue;
                float3 point = minimum +
                    (float3(float(x), float(y), float(z)) + 0.5f) * spacing;
                float3 closest = metalClosestPointTriangle(point, a, b, c);
                float3 delta = point - closest;
                if (dot(delta, delta) > radiusSquared) continue;
                float3 weights = metalTriangleWeights(closest, a, b, c);
                float cornerWeights[3];
                cornerWeights[0] = weights.x;
                cornerWeights[1] = weights.y;
                cornerWeights[2] = weights.z;
                float3 velocity = float3(0.0f);
                for (uint corner = 0u; corner < 3u; ++corner)
                    velocity += smokeSurfaceVertexVelocity(vertices[corner]) *
                        cornerWeights[corner];
                smokeSurfaceGridWallVelocity[cell] = metalStore(velocity);
            }
    uint faceTotal = smokeSurfaceFaceTotal(resolution, vertical);
    float3 normal = smokeSurfaceNormalizedOr(
        cross(b - a, c - a), float3(0.0f, 1.0f, 0.0f));
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        float3 offset = 0.5f;
        if (axis == 0u) offset.x = 0.0f;
        if (axis == 1u) offset.y = 0.0f;
        if (axis == 2u) offset.z = 0.0f;
        int sx = int(resolution) + (axis == 0u ? 1 : 0);
        int sy = int(vertical) + (axis == 1u ? 1 : 0);
        int sz = int(resolution) + (axis == 2u ? 1 : 0);
        int fx0 = max(0, int(floor(
            (lower.x - minimum.x) * inverseSpacing - offset.x)));
        int fy0 = max(0, int(floor(
            (lower.y - minimum.y) * inverseSpacing - offset.y)));
        int fz0 = max(0, int(floor(
            (lower.z - minimum.z) * inverseSpacing - offset.z)));
        int fx1 = min(sx - 1, int(floor(
            (upper.x - minimum.x) * inverseSpacing - offset.x)));
        int fy1 = min(sy - 1, int(floor(
            (upper.y - minimum.y) * inverseSpacing - offset.y)));
        int fz1 = min(sz - 1, int(floor(
            (upper.z - minimum.z) * inverseSpacing - offset.z)));
        for (int z = fz0; z <= fz1; ++z)
            for (int y = fy0; y <= fy1; ++y)
                for (int x = fx0; x <= fx1; ++x)
                {
                    uint face = metalSmokeFaceIndex(
                        axis, x, y, z, resolution, vertical);
                    if (smokeSurfaceFaceNearestTriangle[face] !=
                        globalTriangle)
                        continue;
                    float3 point = minimum +
                        (float3(float(x), float(y), float(z)) + offset) *
                            spacing;
                    float3 closest =
                        metalClosestPointTriangle(point, a, b, c);
                    float3 weights = metalTriangleWeights(closest, a, b, c);
                    float cornerWeights[3];
                    cornerWeights[0] = weights.x;
                    cornerWeights[1] = weights.y;
                    cornerWeights[2] = weights.z;
                    float3 velocity = float3(0.0f);
                    for (uint corner = 0u; corner < 3u; ++corner)
                        velocity += smokeSurfaceVertexVelocity(
                            vertices[corner]) * cornerWeights[corner];
                    float component = axis == 0u ? velocity.x
                        : axis == 1u ? velocity.y : velocity.z;
                    smokeSurfaceFaceBoundaryBits[faceTotal + face] =
                        asuint(component);
                    smokeSurfaceFaceNormal[face] = metalStore(normal);
                }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_surface_grid_force(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint target = dispatchThreadId.x;
    MetalSmokeSurfaceConstants constants = smokeSurfaceConstants;
    if (target >= constants.target_count || constants.enabled == 0u ||
        constants.grid_resolution == 0u || constants.grid_spacing <= 0.0f)
        return;
    float3 accumulatedForce = float3(0.0f);
    uint resolution = constants.grid_resolution;
    uint vertical = constants.grid_vertical_resolution;
    float spacing = constants.grid_spacing;
    float3 minimum = metalLoad(constants.grid_minimum);
    float3 maximum = minimum + spacing *
        float3(float(resolution), float(vertical), float(resolution));
    float surfaceOffset = 1.5f * spacing;
    for (uint base = 0u; base + 2u < constants.surface_index_count;
         base += 3u)
    {
        uint ia = smokeSurfaceIndices[base];
        uint ib = smokeSurfaceIndices[base + 1u];
        uint ic = smokeSurfaceIndices[base + 2u];
        float3 a = metalLoad(smokeSurfacePositions[ia]);
        float3 b = metalLoad(smokeSurfacePositions[ib]);
        float3 c = metalLoad(smokeSurfacePositions[ic]);
        float3 twiceArea = cross(b - a, c - a);
        float area = 0.5f * length(twiceArea);
        if (area <= 1.0e-9f) continue;
        float3 normal = normalize(twiceArea);
        float3 center = (a + b + c) / 3.0f;
        float3 plus = center + normal * surfaceOffset;
        float3 minus = center - normal * surfaceOffset;
        if (any(plus < minimum) || any(plus >= maximum) ||
            any(minus < minimum) || any(minus >= maximum))
            continue;
        float plusDensity = constants.rest_number_density *
            metalSmokeSampleCellScalar(smokeSurfaceGridDensity,
                plus, minimum, spacing, resolution, vertical);
        float minusDensity = constants.rest_number_density *
            metalSmokeSampleCellScalar(smokeSurfaceGridDensity,
                minus, minimum, spacing, resolution, vertical);
        if (plusDensity + minusDensity < 1.0e-4f) continue;
        float3 plusAir = metalSmokeSampleFaceVelocity(
            smokeSurfaceGridFaceVelocity, 0u, plus, minimum, spacing,
            resolution, vertical);
        float3 minusAir = metalSmokeSampleFaceVelocity(
            smokeSurfaceGridFaceVelocity, 0u, minus, minimum, spacing,
            resolution, vertical);

        uint corners[3];
        corners[0] = ia;
        corners[1] = ib;
        corners[2] = ic;
        float3 bodyVelocity = float3(0.0f);
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            if (constants.mode == 0u)
            {
                uint node = smokeSurfaceSources[corners[corner]];
                if (node < constants.target_count)
                    bodyVelocity += metalLoad(
                        smokeSurfaceTargetVelocities[node]);
            }
            else
            {
                MetalSurfaceBinding binding =
                    smokeSurfaceBindings[corners[corner]];
                for (uint slot = 0u; slot < 4u; ++slot)
                {
                    uint node = binding.nodes[slot];
                    if (node < constants.target_count)
                        bodyVelocity += metalLoad(
                            smokeSurfaceTargetVelocities[node]) *
                            binding.weights[slot];
                }
            }
        }
        bodyVelocity /= 3.0f;
        float3 plusRelative = plusAir - bodyVelocity;
        float3 minusRelative = minusAir - bodyVelocity;
        float3 plusTangent =
            plusRelative - normal * dot(plusRelative, normal);
        float3 minusTangent =
            minusRelative - normal * dot(minusRelative, normal);
        float pressureDifference =
            minusDensity * metalSmokeSampleCellScalar(
                smokeSurfaceGridPressure, minus, minimum, spacing,
                resolution, vertical) -
            plusDensity * metalSmokeSampleCellScalar(
                smokeSurfaceGridPressure, plus, minimum, spacing,
                resolution, vertical);
        float strain = metalSmokeSampleFaceStrain(
            smokeSurfaceGridFaceVelocity, 0u, center, minimum, spacing,
            resolution, vertical);
        float viscosity = constants.grid_kinematic_viscosity +
            constants.grid_les_coefficient * constants.grid_les_coefficient *
                spacing * spacing * strain;
        float3 force = (normal * pressureDifference +
            (plusTangent * plusDensity + minusTangent * minusDensity) *
                (constants.wind_drag * viscosity / surfaceOffset)) * area;
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            if (constants.mode == 0u)
            {
                uint node = smokeSurfaceSources[corners[corner]];
                if (node == target) accumulatedForce += force / 3.0f;
            }
            else
            {
                MetalSurfaceBinding binding =
                    smokeSurfaceBindings[corners[corner]];
                for (uint slot = 0u; slot < 4u; ++slot)
                {
                    uint node = binding.nodes[slot];
                    float weight = binding.weights[slot];
                    if (node == target && weight > 0.0f)
                        accumulatedForce += force * (weight / 3.0f);
                }
            }
        }
    }
    smokeSurfaceTargetForces[target] = metalStore(accumulatedForce);
    float inverseMass = smokeSurfaceTargetInverseMasses[target];
    if (inverseMass <= 0.0f) return;
    float3 delta = metalLimit(
        accumulatedForce * (inverseMass * constants.timestep),
        constants.maximum_wind_acceleration * constants.timestep);
    smokeSurfaceTargetVelocities[target] = metalStore(metalLimit(
        metalLoad(smokeSurfaceTargetVelocities[target]) + delta,
        constants.maximum_target_speed));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_surface(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint threadIndex = dispatchThreadId.x;
    MetalSmokeSurfaceConstants constants = smokeSurfaceConstants;
    if (constants.enabled == 0u) return;
    uint smokeCount = smokeSurfaceSmokeMetadata[0].count;
    float windSquared = constants.wind_radius * constants.wind_radius;
    if (constants.command != 2u && constants.grid_resolution == 0u)
    {
        uint target = threadIndex;
        if (target >= constants.target_count ||
            smokeSurfaceTargetInverseMasses[target] <= 0.0f)
            return;
        float3 average = float3(0.0f);
        float density = 0.0f;
        for (uint particle = 0u; particle < smokeCount; ++particle)
        {
            if (smokeSurfaceSmokeAges[particle] >= constants.lifetime)
                continue;
            float3 delta = metalLoad(smokeSurfaceSmokePositions[particle]) -
                metalLoad(smokeSurfaceTargetPositions[target]);
            float squared = dot(delta, delta);
            if (squared >= windSquared) continue;
            float q = sqrt(max(squared, 0.0f)) /
                max(constants.wind_radius, 1.0e-6f);
            float weight = (1.0f - q) * (1.0f - q) * (1.0f - q);
            average += metalLoad(smokeSurfaceSmokeVelocities[particle]) *
                weight;
            density += weight;
        }
        if (density <= 1.0e-6f) return;
        average /= density;
        float occupancy = min(1.0f,
            density / max(constants.rest_number_density, 1.0e-6f));
        float response = 1.0f - exp(
            -constants.wind_drag * occupancy * constants.timestep);
        float3 change =
            (average - metalLoad(smokeSurfaceTargetVelocities[target])) *
            response;
        change = metalLimit(change,
            constants.maximum_wind_acceleration * constants.timestep);
        smokeSurfaceTargetVelocities[target] = metalStore(metalLimit(
            metalLoad(smokeSurfaceTargetVelocities[target]) + change,
            constants.maximum_target_speed));
    }

    if (constants.command == 1u) return;
    uint particle = threadIndex;
    if (particle >= smokeCount ||
        smokeSurfaceSmokeAges[particle] >= constants.lifetime)
        return;
    float3 point = metalLoad(smokeSurfaceSmokePositions[particle]);
    float3 before = metalLoad(smokeSurfaceSmokePrevious[particle]);
    float nearestSquared =
        constants.contact_distance * constants.contact_distance;
    float earliest = 2.0f;
    uint best = constants.surface_index_count;
    float3 nearest = float3(0.0f);
    float3 faceNormal = float3(0.0f);
    float3 weights = float3(0.0f);
    for (uint base = 0u; base + 2u < constants.surface_index_count;
         base += 3u)
    {
        float3 a = metalLoad(
            smokeSurfacePositions[smokeSurfaceIndices[base]]);
        float3 b = metalLoad(
            smokeSurfacePositions[smokeSurfaceIndices[base + 1u]]);
        float3 c = metalLoad(
            smokeSurfacePositions[smokeSurfaceIndices[base + 2u]]);
        float3 face = cross(b - a, c - a);
        float faceSquared = dot(face, face);
        if (faceSquared <= 1.0e-14f) continue;
        float3 normal = face * rsqrt(faceSquared);
        float sideBefore = dot(before - a, normal);
        float sideAfter = dot(point - a, normal);
        if (sideBefore * sideAfter < 0.0f)
        {
            float fraction = sideBefore / (sideBefore - sideAfter);
            if (fraction < earliest)
            {
                float3 hit = before + (point - before) * fraction;
                float3 candidate =
                    metalClosestPointTriangle(hit, a, b, c);
                if (dot(hit - candidate, hit - candidate) < 1.0e-8f)
                {
                    earliest = fraction;
                    best = base;
                    nearest = candidate;
                    faceNormal = normal;
                    weights = metalTriangleWeights(candidate, a, b, c);
                }
            }
        }
        if (earliest <= 1.0f) continue;
        float3 candidate = metalClosestPointTriangle(point, a, b, c);
        float squared = dot(point - candidate, point - candidate);
        if (squared >= nearestSquared) continue;
        nearestSquared = squared;
        best = base;
        nearest = candidate;
        faceNormal = normal;
        weights = metalTriangleWeights(candidate, a, b, c);
    }
    if (best == constants.surface_index_count) return;
    float side = dot(before - nearest, faceNormal);
    if (abs(side) < 1.0e-5f) side = dot(point - nearest, faceNormal);
    float3 normal = side >= 0.0f ? faceNormal : -faceNormal;
    smokeSurfaceSmokePositions[particle] = metalStore(
        nearest + normal * constants.contact_distance);
    float cornerWeights[3];
    cornerWeights[0] = weights.x;
    cornerWeights[1] = weights.y;
    cornerWeights[2] = weights.z;
    float3 surfaceVelocity = float3(0.0f);
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        uint surface = smokeSurfaceIndices[best + corner];
        float3 cornerVelocity = float3(0.0f);
        if (constants.mode == 0u)
            cornerVelocity = metalLoad(smokeSurfaceTargetVelocities[
                smokeSurfaceSources[surface]]);
        else
        {
            MetalSurfaceBinding binding = smokeSurfaceBindings[surface];
            for (uint slot = 0u; slot < 4u; ++slot)
                cornerVelocity += metalLoad(smokeSurfaceTargetVelocities[
                    binding.nodes[slot]]) * binding.weights[slot];
        }
        surfaceVelocity += cornerVelocity * cornerWeights[corner];
    }
    float3 relative =
        metalLoad(smokeSurfaceSmokeVelocities[particle]) - surfaceVelocity;
    relative -= normal * min(dot(relative, normal), 0.0f);
    smokeSurfaceSmokeVelocities[particle] = metalStore(metalLimit(
        surfaceVelocity + relative, constants.maximum_smoke_speed));
}
