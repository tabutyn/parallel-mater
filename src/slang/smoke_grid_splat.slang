// SPDX-License-Identifier: MIT
// Production smoke particle-to-grid splat and radix kernels for Metal.

import smoke_common;
import metal_common;

struct MetalSmokeGridContribution
{
    uint64_t density;
    uint64_t temperature;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    splatPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float>
    splatAges : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float>
    splatThermalLift : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<MetalSmokeMetadata>
    splatMetadata : register(t3);
[[vk::binding(4, 0)]] ConstantBuffer<MetalSmokeConstants>
    splatConstants : register(b4);
[[vk::binding(5, 0)]] RWStructuredBuffer<uint>
    splatKeysA : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<uint>
    splatKeysB : register(u6);
[[vk::binding(7, 0)]] RWStructuredBuffer<MetalSmokeGridContribution>
    splatContributionsA : register(u7);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalSmokeGridContribution>
    splatContributionsB : register(u8);
[[vk::binding(9, 0)]] RWStructuredBuffer<uint>
    splatHistograms : register(u9);
[[vk::binding(10, 0)]] RWStructuredBuffer<uint>
    splatBucketOffsets : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<float>
    splatGridDensity : register(u11);
[[vk::binding(12, 0)]] RWStructuredBuffer<float>
    splatGridTemperature : register(u12);

static const uint smokeGridSplatWidth = 27u;
static const uint smokeGridSplatBlockSize = 256u;

[ForceInline]
static uint smokeGridSplatCount()
{
    return splatMetadata[0].count * smokeGridSplatWidth;
}

[ForceInline]
static float smokeGridSplatWeight(float value, int offset)
{
    if (offset < 0)
        return 0.5f * (0.5f - value) * (0.5f - value);
    if (offset == 0) return 0.75f - value * value;
    return 0.5f * (0.5f + value) * (0.5f + value);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_splat_generate(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint contribution = dispatchThreadId.x;
    uint contributionCount = smokeGridSplatCount();
    if (contribution >= contributionCount) return;
    splatKeysA[contribution] = 0xffffffffu;
    MetalSmokeGridContribution empty;
    empty.density = uint64_t(0u);
    empty.temperature = uint64_t(0u);
    splatContributionsA[contribution] = empty;
    uint particle = contribution / smokeGridSplatWidth;
    MetalSmokeConstants constants = splatConstants;
    if (particle >= splatMetadata[0].count ||
        splatAges[particle] >= constants.lifetime ||
        constants.grid_resolution == 0u || constants.grid_spacing <= 0.0f)
        return;

    uint stencil = contribution % smokeGridSplatWidth;
    int dx = int(stencil % 3u) - 1;
    int dy = int(stencil / 3u % 3u) - 1;
    int dz = int(stencil / 9u) - 1;
    float3 minimum = metalLoad(constants.grid_minimum);
    float3 g = (metalLoad(splatPositions[particle]) - minimum) /
        constants.grid_spacing - 0.5f;
    int3 center = int3(floor(g + 0.5f));
    int3 coordinate = center + int3(dx, dy, dz);
    int resolution = int(constants.grid_resolution);
    int vertical = int(constants.grid_vertical_resolution);
    if (coordinate.x < 0 || coordinate.y < 0 || coordinate.z < 0 ||
        coordinate.x >= resolution || coordinate.y >= vertical ||
        coordinate.z >= resolution)
        return;

    float3 delta = g - float3(center);
    float weight =
        smokeGridSplatWeight(delta.x, dx) *
        smokeGridSplatWeight(delta.y, dy) *
        smokeGridSplatWeight(delta.z, dz) /
        max(1.0f, constants.rest_number_density);
    static const float scale = 16777216.0f;
    splatKeysA[contribution] = uint(coordinate.x) +
        constants.grid_resolution *
            (uint(coordinate.y) +
             constants.grid_vertical_resolution * uint(coordinate.z));
    MetalSmokeGridContribution value;
    value.density = uint64_t(weight * scale + 0.5f);
    value.temperature = uint64_t(
        weight * clamp(splatThermalLift[particle],
            0.0f, constants.maximum_speed) * scale + 0.5f);
    splatContributionsA[contribution] = value;
}

[ForceInline]
static void smokeGridSplatHistogram(
    RWStructuredBuffer<uint> inputKeys, uint shift, uint block)
{
    uint contributionCount = smokeGridSplatCount();
    uint blockCount =
        (contributionCount + smokeGridSplatBlockSize - 1u) /
        smokeGridSplatBlockSize;
    if (block >= blockCount) return;
    uint local[256];
    for (uint bucket = 0u; bucket < 256u; ++bucket) local[bucket] = 0u;
    uint begin = block * smokeGridSplatBlockSize;
    uint end = min(begin + smokeGridSplatBlockSize, contributionCount);
    for (uint index = begin; index < end; ++index)
        ++local[(inputKeys[index] >> shift) & 0xffu];
    for (uint bucket = 0u; bucket < 256u; ++bucket)
        splatHistograms[bucket * blockCount + block] = local[bucket];
}

#define PM_SMOKE_SPLAT_HISTOGRAM_ENTRY(NAME, KEYS, SHIFT) \
[shader("compute")] \
[numthreads(64, 1, 1)] \
void NAME(uint3 dispatchThreadId : SV_DispatchThreadID) \
{ smokeGridSplatHistogram(KEYS, SHIFT, dispatchThreadId.x); }

PM_SMOKE_SPLAT_HISTOGRAM_ENTRY(
    pm_smoke_grid_splat_histogram_0, splatKeysA, 0u)
PM_SMOKE_SPLAT_HISTOGRAM_ENTRY(
    pm_smoke_grid_splat_histogram_1, splatKeysB, 8u)
PM_SMOKE_SPLAT_HISTOGRAM_ENTRY(
    pm_smoke_grid_splat_histogram_2, splatKeysA, 16u)
PM_SMOKE_SPLAT_HISTOGRAM_ENTRY(
    pm_smoke_grid_splat_histogram_3, splatKeysB, 24u)

#undef PM_SMOKE_SPLAT_HISTOGRAM_ENTRY

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_splat_prefix_blocks(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint bucket = dispatchThreadId.x;
    if (bucket >= 256u) return;
    uint contributionCount = smokeGridSplatCount();
    uint blockCount =
        (contributionCount + smokeGridSplatBlockSize - 1u) /
        smokeGridSplatBlockSize;
    uint cursor = 0u;
    for (uint block = 0u; block < blockCount; ++block)
    {
        uint index = bucket * blockCount + block;
        uint count = splatHistograms[index];
        splatHistograms[index] = cursor;
        cursor += count;
    }
    splatBucketOffsets[bucket] = cursor;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void pm_smoke_grid_splat_prefix_buckets(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u) return;
    uint cursor = 0u;
    for (uint bucket = 0u; bucket < 256u; ++bucket)
    {
        uint count = splatBucketOffsets[bucket];
        splatBucketOffsets[bucket] = cursor;
        cursor += count;
    }
}

[ForceInline]
static void smokeGridSplatScatter(
    RWStructuredBuffer<uint> inputKeys,
    RWStructuredBuffer<MetalSmokeGridContribution> inputContributions,
    RWStructuredBuffer<uint> outputKeys,
    RWStructuredBuffer<MetalSmokeGridContribution> outputContributions,
    uint shift, uint block)
{
    uint contributionCount = smokeGridSplatCount();
    uint blockCount =
        (contributionCount + smokeGridSplatBlockSize - 1u) /
        smokeGridSplatBlockSize;
    if (block >= blockCount) return;
    uint localOffsets[256];
    for (uint bucket = 0u; bucket < 256u; ++bucket)
        localOffsets[bucket] = 0u;
    uint begin = block * smokeGridSplatBlockSize;
    uint end = min(begin + smokeGridSplatBlockSize, contributionCount);
    for (uint index = begin; index < end; ++index)
    {
        uint key = inputKeys[index];
        uint bucket = (key >> shift) & 0xffu;
        uint destination = splatBucketOffsets[bucket] +
            splatHistograms[bucket * blockCount + block] +
            localOffsets[bucket]++;
        outputKeys[destination] = key;
        outputContributions[destination] = inputContributions[index];
    }
}

#define PM_SMOKE_SPLAT_SCATTER_ENTRY( \
    NAME, INPUT_KEYS, INPUT_VALUES, OUTPUT_KEYS, OUTPUT_VALUES, SHIFT) \
[shader("compute")] \
[numthreads(64, 1, 1)] \
void NAME(uint3 dispatchThreadId : SV_DispatchThreadID) \
{ smokeGridSplatScatter(INPUT_KEYS, INPUT_VALUES, OUTPUT_KEYS, OUTPUT_VALUES, \
      SHIFT, dispatchThreadId.x); }

PM_SMOKE_SPLAT_SCATTER_ENTRY(pm_smoke_grid_splat_scatter_0,
    splatKeysA, splatContributionsA, splatKeysB, splatContributionsB, 0u)
PM_SMOKE_SPLAT_SCATTER_ENTRY(pm_smoke_grid_splat_scatter_1,
    splatKeysB, splatContributionsB, splatKeysA, splatContributionsA, 8u)
PM_SMOKE_SPLAT_SCATTER_ENTRY(pm_smoke_grid_splat_scatter_2,
    splatKeysA, splatContributionsA, splatKeysB, splatContributionsB, 16u)
PM_SMOKE_SPLAT_SCATTER_ENTRY(pm_smoke_grid_splat_scatter_3,
    splatKeysB, splatContributionsB, splatKeysA, splatContributionsA, 24u)

#undef PM_SMOKE_SPLAT_SCATTER_ENTRY

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_splat_resolve(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint cell = dispatchThreadId.x;
    MetalSmokeConstants constants = splatConstants;
    uint cellCount = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    if (cell >= cellCount) return;
    uint contributionCount = smokeGridSplatCount();
    uint lower = 0u;
    uint upper = contributionCount;
    while (lower < upper)
    {
        uint middle = lower + (upper - lower) / 2u;
        if (splatKeysA[middle] < cell) lower = middle + 1u;
        else upper = middle;
    }
    uint64_t density = uint64_t(0u);
    uint64_t temperature = uint64_t(0u);
    while (lower < contributionCount && splatKeysA[lower] == cell)
    {
        density += splatContributionsA[lower].density;
        temperature += splatContributionsA[lower].temperature;
        ++lower;
    }
    static const float inverseScale = 1.0f / 16777216.0f;
    splatGridDensity[cell] = float(density) * inverseScale;
    splatGridTemperature[cell] = float(temperature) * inverseScale;
}
