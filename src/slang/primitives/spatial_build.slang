// SPDX-License-Identifier: MIT
// Low-dispatch, backend-neutral linear-BVH build primitives. The caller owns
// storage and synchronization; every active count stays in device memory.

import device_intrinsics;
import graphics_resources;

static const uint kPmSpatialTile = 256u;
static const uint kPmSpatialSmallCapacity = 1024u;
static const uint kPmSpatialRadix = 256u;
static const uint kPmSpatialInvalid = 0xffffffffu;
static const uint kPmSpatialInvalidBounds = 1u;
static const uint kPmSpatialDuplicateKey = 2u;
static const uint kPmSpatialCountOverflow = 4u;
static const uint kPmSpatialPolicyMismatch = 8u;

public struct PmSpatialAabb
{
    float4 minimum;
    float4 maximum;
};

groupshared float pmSpatialMinX[256];
groupshared float pmSpatialMinY[256];
groupshared float pmSpatialMinZ[256];
groupshared float pmSpatialMaxX[256];
groupshared float pmSpatialMaxY[256];
groupshared float pmSpatialMaxZ[256];
groupshared uint pmSpatialValid[256];
groupshared uint pmSpatialRadixCounts[256];
groupshared uint pmSpatialRadixDigits[256];
groupshared uint64_t pmSpatialSmallKeys[1024];
groupshared uint pmSpatialSmallValues[1024];

uint pmSpatialLiveCount(PMBuffer<uint> activeCount, uint capacity,
    PMBuffer<uint> status)
{
    uint count = activeCount[0];
    if (count > capacity)
    {
        pmAtomicOr(status, kPmSpatialCountOverflow);
        count = capacity;
    }
    return count;
}

bool pmSpatialValidBounds(PmSpatialAabb bounds)
{
    return all(isfinite(bounds.minimum.xyz)) &&
        all(isfinite(bounds.maximum.xyz)) &&
        all(bounds.minimum.xyz <= bounds.maximum.xyz);
}

PmSpatialAabb pmSpatialEmptyBounds()
{
    PmSpatialAabb bounds;
    bounds.minimum = float4(1.0f / 0.0f);
    bounds.maximum = float4(-1.0f / 0.0f);
    return bounds;
}

PmSpatialAabb pmSpatialUnion(PmSpatialAabb left, PmSpatialAabb right)
{
    PmSpatialAabb result;
    result.minimum = float4(min(left.minimum.xyz, right.minimum.xyz), 0.0f);
    result.maximum = float4(max(left.maximum.xyz, right.maximum.xyz), 0.0f);
    return result;
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_reduce_bounds_blocks(
    uniform PMBuffer<uint> active_indices,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<PmSpatialAabb> input_bounds,
    uniform PMBuffer<PmSpatialAabb> block_bounds,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint lane = pmGroupThreadIndex().x;
    uint index = pmThreadIndex();
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    bool present = index < count;
    PmSpatialAabb bounds = pmSpatialEmptyBounds();
    if (present)
    {
        bounds = input_bounds[active_indices[index]];
        if (!pmSpatialValidBounds(bounds))
        {
            pmAtomicOr(status, kPmSpatialInvalidBounds);
            bounds = pmSpatialEmptyBounds();
            present = false;
        }
    }
    pmSpatialMinX[lane] = bounds.minimum.x;
    pmSpatialMinY[lane] = bounds.minimum.y;
    pmSpatialMinZ[lane] = bounds.minimum.z;
    pmSpatialMaxX[lane] = bounds.maximum.x;
    pmSpatialMaxY[lane] = bounds.maximum.y;
    pmSpatialMaxZ[lane] = bounds.maximum.z;
    pmSpatialValid[lane] = present ? 1u : 0u;
    pmGroupBarrier();

    for (uint stride = 128u; stride != 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            pmSpatialMinX[lane] = min(pmSpatialMinX[lane], pmSpatialMinX[lane + stride]);
            pmSpatialMinY[lane] = min(pmSpatialMinY[lane], pmSpatialMinY[lane + stride]);
            pmSpatialMinZ[lane] = min(pmSpatialMinZ[lane], pmSpatialMinZ[lane + stride]);
            pmSpatialMaxX[lane] = max(pmSpatialMaxX[lane], pmSpatialMaxX[lane + stride]);
            pmSpatialMaxY[lane] = max(pmSpatialMaxY[lane], pmSpatialMaxY[lane + stride]);
            pmSpatialMaxZ[lane] = max(pmSpatialMaxZ[lane], pmSpatialMaxZ[lane + stride]);
            pmSpatialValid[lane] += pmSpatialValid[lane + stride];
        }
        pmGroupBarrier();
    }
    if (lane == 0u)
    {
        PmSpatialAabb output;
        output.minimum = float4(pmSpatialMinX[0], pmSpatialMinY[0],
            pmSpatialMinZ[0], float(pmSpatialValid[0]));
        output.maximum = float4(pmSpatialMaxX[0], pmSpatialMaxY[0],
            pmSpatialMaxZ[0], 0.0f);
        block_bounds[pmBlockIndex().x] = output;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_reduce_bounds_final(
    uniform PMBuffer<PmSpatialAabb> block_bounds, uniform uint block_count,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<PmSpatialAabb> scene_bounds,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint lane = pmGroupThreadIndex().x;
    PmSpatialAabb bounds = pmSpatialEmptyBounds();
    uint present = 0u;
    for (uint block = lane; block < block_count; block += 256u)
    {
        PmSpatialAabb candidate = block_bounds[block];
        if (candidate.minimum.w > 0.0f)
        {
            bounds = present == 0u ? candidate : pmSpatialUnion(bounds, candidate);
            present += uint(candidate.minimum.w);
        }
    }
    pmSpatialMinX[lane] = bounds.minimum.x;
    pmSpatialMinY[lane] = bounds.minimum.y;
    pmSpatialMinZ[lane] = bounds.minimum.z;
    pmSpatialMaxX[lane] = bounds.maximum.x;
    pmSpatialMaxY[lane] = bounds.maximum.y;
    pmSpatialMaxZ[lane] = bounds.maximum.z;
    pmSpatialValid[lane] = present;
    pmGroupBarrier();
    for (uint stride = 128u; stride != 0u; stride >>= 1u)
    {
        if (lane < stride)
        {
            pmSpatialMinX[lane] = min(pmSpatialMinX[lane], pmSpatialMinX[lane + stride]);
            pmSpatialMinY[lane] = min(pmSpatialMinY[lane], pmSpatialMinY[lane + stride]);
            pmSpatialMinZ[lane] = min(pmSpatialMinZ[lane], pmSpatialMinZ[lane + stride]);
            pmSpatialMaxX[lane] = max(pmSpatialMaxX[lane], pmSpatialMaxX[lane + stride]);
            pmSpatialMaxY[lane] = max(pmSpatialMaxY[lane], pmSpatialMaxY[lane + stride]);
            pmSpatialMaxZ[lane] = max(pmSpatialMaxZ[lane], pmSpatialMaxZ[lane + stride]);
            pmSpatialValid[lane] += pmSpatialValid[lane + stride];
        }
        pmGroupBarrier();
    }
    if (lane == 0u)
    {
        uint count = pmSpatialLiveCount(active_count, capacity, status);
        PmSpatialAabb output;
        if (count == 0u)
        {
            output.minimum = float4(0.0f);
            output.maximum = float4(0.0f);
        }
        else
        {
            output.minimum = float4(pmSpatialMinX[0], pmSpatialMinY[0],
                pmSpatialMinZ[0], 0.0f);
            output.maximum = float4(pmSpatialMaxX[0], pmSpatialMaxY[0],
                pmSpatialMaxZ[0], 0.0f);
            if (pmSpatialValid[0] != count)
                pmAtomicOr(status, kPmSpatialInvalidBounds);
        }
        scene_bounds[0] = output;
    }
}

uint pmSpatialExpand10(uint value)
{
    value &= 0x000003ffu;
    value = (value | (value << 16u)) & 0x030000ffu;
    value = (value | (value << 8u)) & 0x0300f00fu;
    value = (value | (value << 4u)) & 0x030c30c3u;
    value = (value | (value << 2u)) & 0x09249249u;
    return value;
}

uint pmSpatialMorton(float3 point, PmSpatialAabb scene)
{
    float3 extent = scene.maximum.xyz - scene.minimum.xyz;
    float3 normalized = float3(
        extent.x > 0.0f ? (point.x - scene.minimum.x) / extent.x : 0.0f,
        extent.y > 0.0f ? (point.y - scene.minimum.y) / extent.y : 0.0f,
        extent.z > 0.0f ? (point.z - scene.minimum.z) / extent.z : 0.0f);
    uint3 quantized = uint3(clamp(normalized, 0.0f, 1.0f) * 1023.0f);
    return pmSpatialExpand10(quantized.x) |
        (pmSpatialExpand10(quantized.y) << 1u) |
        (pmSpatialExpand10(quantized.z) << 2u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_emit_keys(
    uniform PMBuffer<uint> active_indices,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<uint> identities,
    uniform PMBuffer<PmSpatialAabb> input_bounds,
    uniform PMBuffer<PmSpatialAabb> scene_bounds,
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> values,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    if (index >= count) return;
    uint source = active_indices[index];
    PmSpatialAabb bounds = input_bounds[source];
    float3 center = 0.5f * (bounds.minimum.xyz + bounds.maximum.xyz);
    uint morton = pmSpatialMorton(center, scene_bounds[0]);
    keys[index] = (uint64_t(morton) << 32u) | uint64_t(identities[source]);
    values[index] = source;
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_sort_small_u64_u32(
    uniform PMBuffer<uint64_t> keys, uniform PMBuffer<uint> values,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint lane = pmGroupThreadIndex().x;
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    if (count > kPmSpatialSmallCapacity)
    {
        if (lane == 0u) pmAtomicOr(status, kPmSpatialPolicyMismatch);
        return;
    }
    for (uint slot = 0u; slot < 4u; ++slot)
    {
        uint index = lane + slot * 256u;
        pmSpatialSmallKeys[index] = index < count ? keys[index] : ~0ull;
        pmSpatialSmallValues[index] = index < count ? values[index] : kPmSpatialInvalid;
    }
    pmGroupBarrier();
    for (uint width = 2u; width <= kPmSpatialSmallCapacity; width <<= 1u)
    {
        for (uint stride = width >> 1u; stride != 0u; stride >>= 1u)
        {
            for (uint slot = 0u; slot < 4u; ++slot)
            {
                uint index = lane + slot * 256u;
                uint partner = index ^ stride;
                if (partner > index)
                {
                    bool ascending = (index & width) == 0u;
                    uint64_t left = pmSpatialSmallKeys[index];
                    uint64_t right = pmSpatialSmallKeys[partner];
                    if (ascending ? left > right : left < right)
                    {
                        pmSpatialSmallKeys[index] = right;
                        pmSpatialSmallKeys[partner] = left;
                        uint leftValue = pmSpatialSmallValues[index];
                        pmSpatialSmallValues[index] = pmSpatialSmallValues[partner];
                        pmSpatialSmallValues[partner] = leftValue;
                    }
                }
            }
            pmGroupBarrier();
        }
    }
    for (uint slot = 0u; slot < 4u; ++slot)
    {
        uint index = lane + slot * 256u;
        if (index < count)
        {
            keys[index] = pmSpatialSmallKeys[index];
            values[index] = pmSpatialSmallValues[index];
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_radix_histogram_u64(
    uniform PMBuffer<uint64_t> keys,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform uint shift, uniform uint block_count,
    uniform PMBuffer<uint> histograms,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint block = pmBlockIndex().x;
    uint lane = pmGroupThreadIndex().x;
    if (block >= block_count) return;
    pmSpatialRadixCounts[lane] = 0u;
    pmGroupBarrier();
    uint index = block * kPmSpatialTile + lane;
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    if (index < count)
    {
        uint digit = uint((keys[index] >> shift) & 255ull);
        uint ignored;
        InterlockedAdd(pmSpatialRadixCounts[digit], 1u, ignored);
    }
    pmGroupBarrier();
    histograms[lane * block_count + block] = pmSpatialRadixCounts[lane];
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_radix_prefix_u32(
    uniform PMBuffer<uint> histograms, uniform uint block_count,
    uniform PMBuffer<uint> offsets PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint digit = pmGroupThreadIndex().x;
    uint running = 0u;
    for (uint block = 0u; block < block_count; ++block)
    {
        uint address = digit * block_count + block;
        offsets[address] = running;
        running += histograms[address];
    }
    pmSpatialRadixCounts[digit] = running;
    pmGroupBarrier();
    if (digit == 0u)
    {
        uint base = 0u;
        for (uint index = 0u; index < kPmSpatialRadix; ++index)
        {
            uint total = pmSpatialRadixCounts[index];
            pmSpatialRadixCounts[index] = base;
            base += total;
        }
    }
    pmGroupBarrier();
    uint base = pmSpatialRadixCounts[digit];
    for (uint block = 0u; block < block_count; ++block)
        offsets[digit * block_count + block] += base;
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_radix_scatter_u64_u32(
    uniform PMBuffer<uint64_t> input_keys,
    uniform PMBuffer<uint> input_values,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform uint shift, uniform uint block_count,
    uniform PMBuffer<uint> offsets,
    uniform PMBuffer<uint64_t> output_keys,
    uniform PMBuffer<uint> output_values,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint block = pmBlockIndex().x;
    uint lane = pmGroupThreadIndex().x;
    if (block >= block_count) return;
    uint index = block * kPmSpatialTile + lane;
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    uint64_t key = index < count ? input_keys[index] : 0ull;
    uint digit = index < count ? uint((key >> shift) & 255ull) : 256u;
    pmSpatialRadixDigits[lane] = digit;
    pmGroupBarrier();
    if (index >= count) return;
    uint rank = 0u;
    for (uint earlier = 0u; earlier < lane; ++earlier)
        rank += pmSpatialRadixDigits[earlier] == digit ? 1u : 0u;
    uint destination = offsets[digit * block_count + block] + rank;
    output_keys[destination] = key;
    output_values[destination] = input_values[index];
}

int pmSpatialLeadingZeros(uint value)
{
    if (value == 0u) return 32;
    int result = 0;
    if ((value & 0xffff0000u) == 0u) { result += 16; value <<= 16u; }
    if ((value & 0xff000000u) == 0u) { result += 8; value <<= 8u; }
    if ((value & 0xf0000000u) == 0u) { result += 4; value <<= 4u; }
    if ((value & 0xc0000000u) == 0u) { result += 2; value <<= 2u; }
    if ((value & 0x80000000u) == 0u) result += 1;
    return result;
}

int pmSpatialDelta(PMBuffer<uint64_t> keys, uint count, int left, int right)
{
    if (left < 0 || right < 0 || left >= int(count) || right >= int(count))
        return -1;
    uint64_t difference = keys[uint(left)] ^ keys[uint(right)];
    uint high = uint(difference >> 32u);
    if (high != 0u) return pmSpatialLeadingZeros(high);
    uint low = uint(difference);
    return 32 + pmSpatialLeadingZeros(low);
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_build_topology(
    uniform PMBuffer<uint64_t> sorted_keys,
    uniform PMBuffer<uint> sorted_values,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<PmSpatialAabb> input_bounds,
    uniform PMBuffer<uint> left_children,
    uniform PMBuffer<uint> right_children,
    uniform PMBuffer<uint> parents,
    uniform PMBuffer<uint> leaf_sources,
    uniform PMBuffer<PmSpatialAabb> node_bounds,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    if (index >= count) return;
    if (index == 0u) parents[0] = kPmSpatialInvalid;
    uint leaf = count - 1u + index;
    uint source = sorted_values[index];
    leaf_sources[index] = source;
    node_bounds[leaf] = input_bounds[source];
    if (index > 0u && sorted_keys[index] == sorted_keys[index - 1u])
        pmAtomicOr(status, kPmSpatialDuplicateKey);
    if (count == 1u || index >= count - 1u) return;

    int node = int(index);
    int direction = pmSpatialDelta(sorted_keys, count, node, node + 1) >=
        pmSpatialDelta(sorted_keys, count, node, node - 1) ? 1 : -1;
    int minimumDelta = pmSpatialDelta(sorted_keys, count, node, node - direction);
    int maximumLength = 2;
    while (pmSpatialDelta(sorted_keys, count,
        node, node + maximumLength * direction) > minimumDelta)
        maximumLength <<= 1;
    int length = 0;
    for (int step = maximumLength >> 1; step >= 1; step >>= 1)
        if (pmSpatialDelta(sorted_keys, count,
            node, node + (length + step) * direction) > minimumDelta)
            length += step;
    int other = node + length * direction;
    int nodeDelta = pmSpatialDelta(sorted_keys, count, node, other);
    int splitOffset = 0;
    int step = length;
    do
    {
        step = (step + 1) >> 1;
        if (splitOffset + step < length &&
            pmSpatialDelta(sorted_keys, count, node,
                node + (splitOffset + step) * direction) > nodeDelta)
            splitOffset += step;
    }
    while (step > 1);
    int split = node + splitOffset * direction + min(direction, 0);
    int first = min(node, other);
    int last = max(node, other);
    uint left = split == first ? count - 1u + uint(split) : uint(split);
    uint right = split + 1 == last ? count + uint(split) : uint(split + 1);
    left_children[index] = left;
    right_children[index] = right;
    parents[left] = index;
    parents[right] = index;
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_union_bounds(
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<uint> left_children,
    uniform PMBuffer<uint> right_children,
    uniform PMBuffer<uint> parents,
    uniform PMBuffer<PmSpatialAabb> node_bounds,
    uniform PMBuffer<uint> arrivals,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    if (index >= count || count <= 1u) return;
    uint node = count - 1u + index;
    for (;;)
    {
        uint parent = parents[node];
        pmDeviceBarrier();
        if (pmAtomicAdd(arrivals + parent, 1u) == 0u) return;
        PmSpatialAabb left = node_bounds[left_children[parent]];
        PmSpatialAabb right = node_bounds[right_children[parent]];
        node_bounds[parent] = pmSpatialUnion(left, right);
        if (parent == 0u) return;
        node = parent;
    }
}
