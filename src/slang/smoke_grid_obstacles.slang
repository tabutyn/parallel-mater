// SPDX-License-Identifier: MIT
// Smoke obstacle-grid bookkeeping shared by rigid and deformable rasterizers.

import smoke_common;
import metal_common;

[[vk::binding(0, 0)]] ConstantBuffer<MetalSmokeConstants>
    obstacleConstants : register(b0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>
    obstacleCellNearestTriangle : register(u1);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>
    obstacleFaceNearestTriangle : register(u3);
[[vk::binding(6, 0)]] StructuredBuffer<uint>
    obstacleCounts : register(t6);
[[vk::binding(13, 0)]] RWStructuredBuffer<MetalPackedVec3>
    obstacleGridVelocity : register(u13);
[[vk::binding(14, 0)]] StructuredBuffer<MetalPackedVec3>
    obstacleGridScratch : register(t14);
[[vk::binding(15, 0)]] RWStructuredBuffer<uint>
    obstacleGridSolid : register(u15);

[ForceInline]
static uint obstacleFaceTotal(uint n, uint height)
{
    return metalSmokeFaceCount(0u, n, height) +
        metalSmokeFaceCount(1u, n, height) +
        metalSmokeFaceCount(2u, n, height);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_grid_raster_clear(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalSmokeConstants constants = obstacleConstants;
    if (constants.grid_resolution == 0u) return;
    uint cells = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    uint faces = obstacleFaceTotal(
        constants.grid_resolution, constants.grid_vertical_resolution);
    if (index < cells)
        obstacleCellNearestTriangle[index] = 0xffffffffu;
    if (index < faces && obstacleCounts[3] == 0u)
        obstacleFaceNearestTriangle[index] = 0xffffffffu;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_smoke_grid_merge_obstacles(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint cell = dispatchThreadId.x;
    MetalSmokeConstants constants = obstacleConstants;
    uint cellCount = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    if (cell >= cellCount ||
        obstacleCellNearestTriangle[cell] == 0xffffffffu)
        return;
    obstacleGridSolid[cell] = 1u;
    obstacleGridVelocity[cell] = obstacleGridScratch[cell];
}
