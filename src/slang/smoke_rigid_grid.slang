// SPDX-License-Identifier: MIT
// Rigid-body obstacle rasterization for the Metal smoke grid.

import smoke_common;
import metal_common;

struct MetalSmokeRigidRasterEntry
{
    MetalHandle body;
    uint triangle_base;
};

[[vk::binding(0, 0)]] ConstantBuffer<MetalSmokeConstants>
    smokeRigidConstants : register(b0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>
    smokeRigidCellNearestTriangle : register(u1);
// Positive float values are atomically minimized through their uint bits.
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>
    smokeRigidFaceBoundaryBits : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>
    smokeRigidFaceNearestTriangle : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeRigidFaceNormal : register(u4);
[[vk::binding(5, 0)]] StructuredBuffer<MetalSmokeRigidRasterEntry>
    smokeRigidEntries : register(t5);
[[vk::binding(6, 0)]] StructuredBuffer<uint>
    smokeRigidCounts : register(t6);
[[vk::binding(7, 0)]] StructuredBuffer<MetalHandle>
    smokeRigidIds : register(t7);
[[vk::binding(8, 0)]] StructuredBuffer<MetalRigidState>
    smokeRigidStates : register(t8);
[[vk::binding(9, 0)]] StructuredBuffer<MetalRigidParameters>
    smokeRigidParameters : register(t9);
[[vk::binding(10, 0)]] StructuredBuffer<MetalPackedVec3>
    smokeRigidVertices : register(t10);
[[vk::binding(11, 0)]] StructuredBuffer<uint>
    smokeRigidIndices : register(t11);
[[vk::binding(12, 0)]] StructuredBuffer<MetalTriangleMeshInfo>
    smokeRigidMeshes : register(t12);
[[vk::binding(14, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeRigidGridScratch : register(u14);

[ForceInline]
static uint smokeRigidFaceTotal(uint n, uint height)
{
    return metalSmokeFaceCount(0u, n, height) +
        metalSmokeFaceCount(1u, n, height) +
        metalSmokeFaceCount(2u, n, height);
}

[ForceInline]
static float3 smokeRigidNormalizedOr(float3 value, float3 fallback)
{
    float squared = dot(value, value);
    return squared > 1.0e-20f ? value * rsqrt(squared) : fallback;
}

[ForceInline]
static bool smokeRigidRasterTriangle(
    uint flatTriangle,
    out uint globalTriangle,
    out uint bodyIndex,
    out float3 a,
    out float3 b,
    out float3 c)
{
    uint cursor = flatTriangle;
    uint entryCount = smokeRigidCounts[0];
    uint rigidCount = smokeRigidCounts[1];
    for (uint entryIndex = 0u; entryIndex < entryCount; ++entryIndex)
    {
        MetalSmokeRigidRasterEntry entry = smokeRigidEntries[entryIndex];
        uint dense = rigidCount;
        for (uint body = 0u; body < rigidCount; ++body)
        {
            MetalHandle id = smokeRigidIds[body];
            if (id.index == entry.body.index &&
                id.generation == entry.body.generation)
            {
                dense = body;
                break;
            }
        }
        if (dense == rigidCount) continue;
        MetalRigidParameters parameters = smokeRigidParameters[dense];
        MetalTriangleMeshInfo mesh =
            smokeRigidMeshes[parameters.mesh_index];
        uint triangleCount = mesh.index_count / 3u;
        if (cursor >= triangleCount)
        {
            cursor -= triangleCount;
            continue;
        }
        uint local = cursor * 3u;
        MetalRigidState state = smokeRigidStates[dense];
        a = metalWorldPoint(state, metalLoad(smokeRigidVertices[
            mesh.vertex_offset +
            smokeRigidIndices[mesh.index_offset + local]]));
        b = metalWorldPoint(state, metalLoad(smokeRigidVertices[
            mesh.vertex_offset +
            smokeRigidIndices[mesh.index_offset + local + 1u]]));
        c = metalWorldPoint(state, metalLoad(smokeRigidVertices[
            mesh.vertex_offset +
            smokeRigidIndices[mesh.index_offset + local + 2u]]));
        globalTriangle = entry.triangle_base + cursor;
        bodyIndex = dense;
        return true;
    }
    globalTriangle = 0u;
    bodyIndex = 0u;
    a = float3(0.0f);
    b = float3(0.0f);
    c = float3(0.0f);
    return false;
}

static void smokeRigidRasterTriangleMark(
    float3 a, float3 b, float3 c, uint globalTriangle)
{
    MetalSmokeConstants constants = smokeRigidConstants;
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
                    smokeRigidCellNearestTriangle[cell], globalTriangle);
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
                    InterlockedMin(smokeRigidFaceBoundaryBits[face],
                        asuint(clamp(distanceValue / radius, 0.0f, 1.0f)));
                }
    }
}

static void smokeRigidSelectTriangleFaces(
    float3 a, float3 b, float3 c, uint globalTriangle)
{
    MetalSmokeConstants constants = smokeRigidConstants;
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
                        asfloat(smokeRigidFaceBoundaryBits[face]) + 1.0e-7f)
                        InterlockedMin(
                            smokeRigidFaceNearestTriangle[face],
                            globalTriangle);
                }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_grid_raster_rigid(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint triangle = dispatchThreadId.x;
    MetalSmokeConstants constants = smokeRigidConstants;
    if (triangle >= smokeRigidCounts[2] || constants.grid_resolution == 0u)
        return;
    uint globalTriangle;
    uint bodyIndex;
    float3 a;
    float3 b;
    float3 c;
    if (!smokeRigidRasterTriangle(
        triangle, globalTriangle, bodyIndex, a, b, c))
        return;
    smokeRigidRasterTriangleMark(a, b, c, globalTriangle);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_grid_select_rigid(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint triangle = dispatchThreadId.x;
    MetalSmokeConstants constants = smokeRigidConstants;
    if (triangle >= smokeRigidCounts[2] || constants.grid_resolution == 0u)
        return;
    uint globalTriangle;
    uint bodyIndex;
    float3 a;
    float3 b;
    float3 c;
    if (!smokeRigidRasterTriangle(
        triangle, globalTriangle, bodyIndex, a, b, c))
        return;
    smokeRigidSelectTriangleFaces(a, b, c, globalTriangle);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_grid_resolve_rigid(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint triangle = dispatchThreadId.x;
    MetalSmokeConstants constants = smokeRigidConstants;
    if (triangle >= smokeRigidCounts[2] || constants.grid_resolution == 0u)
        return;
    uint globalTriangle;
    uint bodyIndex;
    float3 a;
    float3 b;
    float3 c;
    if (!smokeRigidRasterTriangle(
        triangle, globalTriangle, bodyIndex, a, b, c))
        return;

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
    MetalRigidState state = smokeRigidStates[bodyIndex];
    float3 linearVelocity = metalLoad(state.linear_velocity);
    float3 angularVelocity = metalLoad(state.angular_velocity);
    float3 position = metalLoad(state.position);
    for (int z = z0; z <= z1; ++z)
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
            {
                uint cell = uint(x) + resolution *
                    (uint(y) + vertical * uint(z));
                if (smokeRigidCellNearestTriangle[cell] != globalTriangle)
                    continue;
                float3 point = minimum +
                    (float3(float(x), float(y), float(z)) + 0.5f) * spacing;
                float3 closest = metalClosestPointTriangle(point, a, b, c);
                float3 delta = point - closest;
                if (dot(delta, delta) > radiusSquared) continue;
                smokeRigidGridScratch[cell] = metalStore(
                    linearVelocity + cross(angularVelocity,
                        closest - position));
            }
    uint faceTotal = smokeRigidFaceTotal(resolution, vertical);
    float3 normal = smokeRigidNormalizedOr(
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
                    uint face = metalSmokeFaceIndex(
                        axis, x, y, z, resolution, vertical);
                    if (smokeRigidFaceNearestTriangle[face] !=
                        globalTriangle)
                        continue;
                    float3 point = minimum +
                        (float3(float(x), float(y), float(z)) + offset) *
                            spacing;
                    float3 closest =
                        metalClosestPointTriangle(point, a, b, c);
                    float3 wallVelocity = linearVelocity +
                        cross(angularVelocity, closest - position);
                    float component = axis == 0u ? wallVelocity.x
                        : axis == 1u ? wallVelocity.y : wallVelocity.z;
                    smokeRigidFaceBoundaryBits[faceTotal + face] =
                        asuint(component);
                    smokeRigidFaceNormal[face] = metalStore(normal);
                }
    }
}
