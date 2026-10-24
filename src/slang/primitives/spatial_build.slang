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
static const uint kPmSpatialInvalidIdentity = 1u << 8u;
static const uint kPmSpatialStaleEpoch = 1u << 9u;
static const uint kPmSpatialQueryOverflow = 1u << 10u;
static const uint kPmSpatialEnvelopeEscape = 1u << 11u;
static const uint kPmSpatialAdmissionInvalid = 1u << 12u;
static const uint kPmSpatialDuplicateUpdate = 1u << 13u;
static const uint kPmSpatialDuplicatePublication = 1u << 14u;
static const uint kPmSpatialLeafRetained = 1u << 0u;
static const uint kPmSpatialLeafProvisional = 1u << 1u;
static const uint kPmSpatialLeafTransferSource = 1u << 2u;
static const uint kPmSpatialLeafTransferDestination = 1u << 3u;
static const uint kPmSpatialLeafInactive = 1u << 4u;
static const uint kPmSpatialQueryStackCapacity = 64u;

public struct PmSpatialAabb
{
    float4 minimum;
    float4 maximum;
};

public struct PmSpatialIdentity
{
    uint family;
    uint owner_slot;
    uint owner_generation;
    uint topology_revision;
    uint primitive_kind;
    uint primitive_slot;
    uint primitive_generation;
};

public struct PmSpatialLeafMetadata
{
    PmSpatialIdentity identity;
    uint sort_key;
    uint payload_ref;
    uint query_mask;
    uint admission_mask;
    uint flags;
    uint transfer_group;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

public struct PmSpatialEpochHeader
{
    uint64_t tick;
    uint64_t graph_revision;
    uint64_t resource_revision;
    uint profile_version;
    uint slot_generation;
    uint leaf_count;
    uint node_count;
    uint status;
    uint build_count;
    uint refit_count;
    uint publication_count;
    uint reserved_query_count;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    uint reserved3;
    uint reserved4;
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

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_refit_leaves(
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<uint> leaf_sources,
    uniform PMBuffer<PmSpatialAabb> input_bounds,
    uniform PMBuffer<PmSpatialAabb> node_bounds,
    uniform PMBuffer<uint> arrivals,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint rank = pmThreadIndex();
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    if (rank >= count) return;
    if (rank < count - 1u) arrivals[rank] = 0u;
    uint source = leaf_sources[rank];
    if (source >= capacity || !pmSpatialValidBounds(input_bounds[source]))
    {
        pmAtomicOr(status, kPmSpatialInvalidBounds);
        return;
    }
    node_bounds[count - 1u + rank] = input_bounds[source];
}

bool pmSpatialIdentityValid(PmSpatialIdentity identity)
{
    return identity.family != 0u && identity.owner_generation != 0u &&
        identity.topology_revision != 0u &&
        identity.primitive_generation != 0u;
}

bool pmSpatialIdentityEqual(PmSpatialIdentity left, PmSpatialIdentity right)
{
    return left.family == right.family &&
        left.owner_slot == right.owner_slot &&
        left.owner_generation == right.owner_generation &&
        left.topology_revision == right.topology_revision &&
        left.primitive_kind == right.primitive_kind &&
        left.primitive_slot == right.primitive_slot &&
        left.primitive_generation == right.primitive_generation;
}

bool pmSpatialIdentityLess(PmSpatialIdentity left, PmSpatialIdentity right)
{
    if (left.family != right.family) return left.family < right.family;
    if (left.owner_slot != right.owner_slot)
        return left.owner_slot < right.owner_slot;
    if (left.owner_generation != right.owner_generation)
        return left.owner_generation < right.owner_generation;
    if (left.topology_revision != right.topology_revision)
        return left.topology_revision < right.topology_revision;
    if (left.primitive_kind != right.primitive_kind)
        return left.primitive_kind < right.primitive_kind;
    if (left.primitive_slot != right.primitive_slot)
        return left.primitive_slot < right.primitive_slot;
    return left.primitive_generation < right.primitive_generation;
}

bool pmSpatialContains(PmSpatialAabb outer, PmSpatialAabb inner)
{
    return all(inner.minimum.xyz >= outer.minimum.xyz) &&
        all(inner.maximum.xyz <= outer.maximum.xyz);
}

bool pmSpatialOverlaps(PmSpatialAabb left, PmSpatialAabb right)
{
    return all(left.maximum.xyz >= right.minimum.xyz) &&
        all(right.maximum.xyz >= left.minimum.xyz);
}

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_spatial_initialize_epoch(
    uniform PMBuffer<PmSpatialEpochHeader> header,
    uniform uint64_t tick, uniform uint64_t graph_revision,
    uniform uint64_t resource_revision, uniform uint profile_version,
    uniform uint slot_generation, uniform uint leaf_count,
    uniform uint update_kind,
    uniform PMBuffer<uint> query_counter,
    uniform PMBuffer<uint> query_status,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    status[0] = 0u;
    query_status[0] = 0u;
    PmSpatialEpochHeader value;
    value.tick = tick;
    value.graph_revision = graph_revision;
    value.resource_revision = resource_revision;
    value.profile_version = profile_version;
    value.slot_generation = slot_generation;
    value.leaf_count = leaf_count;
    value.node_count = leaf_count == 0u ? 0u : 2u * leaf_count - 1u;
    value.status = 0u;
    value.build_count = update_kind == 1u ? 1u : 0u;
    value.refit_count = update_kind == 2u ? 1u : 0u;
    value.publication_count = 0u;
    value.reserved_query_count = 0u;
    value.reserved0 = 0u;
    value.reserved1 = 0u;
    value.reserved2 = 0u;
    value.reserved3 = 0u;
    value.reserved4 = 0u;
    query_counter[0] = 0u;
    if ((leaf_count == 0u) != (update_kind == 0u) || update_kind > 2u)
    {
        value.status |= kPmSpatialDuplicateUpdate;
        status[0] |= kPmSpatialDuplicateUpdate;
    }
    header[0] = value;
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_stage_leaves(
    uniform PMBuffer<PmSpatialLeafMetadata> metadata,
    uniform PMBuffer<PmSpatialAabb> bounds,
    uniform uint leaf_count, uniform uint capacity,
    uniform PMByteBuffer build_flags,
    uniform PMByteBuffer admission,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index >= capacity) return;
    if (index >= leaf_count)
    {
        build_flags[index] = PMResourceUint8(0u);
        admission[index] = PMResourceUint8(0u);
        return;
    }
    PmSpatialLeafMetadata leaf = metadata[index];
    if ((leaf.flags & kPmSpatialLeafInactive) != 0u)
    {
        build_flags[index] = PMResourceUint8(0u);
        admission[index] = PMResourceUint8(0u);
        return;
    }
    uint role = leaf.flags &
        (kPmSpatialLeafRetained | kPmSpatialLeafProvisional);
    bool identity_valid = pmSpatialIdentityValid(leaf.identity) &&
        leaf.sort_key != 0u;
    bool bounds_valid = pmSpatialValidBounds(bounds[index]);
    bool transfer_source =
        (leaf.flags & kPmSpatialLeafTransferSource) != 0u;
    bool transfer_destination =
        (leaf.flags & kPmSpatialLeafTransferDestination) != 0u;
    bool role_valid =
        (role == kPmSpatialLeafRetained ||
         role == kPmSpatialLeafProvisional) &&
        !(transfer_source && transfer_destination) &&
        (!transfer_source || role == kPmSpatialLeafRetained) &&
        (!transfer_destination || role == kPmSpatialLeafProvisional) &&
        (!(transfer_source || transfer_destination) ||
         leaf.transfer_group != 0u);
    bool valid = identity_valid && bounds_valid && role_valid;
    build_flags[index] = PMResourceUint8(valid ? 1u : 0u);
    admission[index] = PMResourceUint8(valid &&
        (leaf.flags & kPmSpatialLeafRetained) != 0u ? 1u : 0u);
    if (!identity_valid) pmAtomicOr(status, kPmSpatialInvalidIdentity);
    if (!bounds_valid) pmAtomicOr(status, kPmSpatialInvalidBounds);
    if (!role_valid) pmAtomicOr(status, kPmSpatialAdmissionInvalid);
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_resolve_admission(
    uniform PMBuffer<PmSpatialLeafMetadata> metadata,
    uniform PMBuffer<PmSpatialAabb> leaf_bounds,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<uint> active_indices,
    uniform PMBuffer<uint> left_children,
    uniform PMBuffer<uint> right_children,
    uniform PMBuffer<uint> leaf_sources,
    uniform PMBuffer<PmSpatialAabb> node_bounds,
    uniform PMByteBuffer admission,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint rank = pmThreadIndex();
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    if (rank >= count) return;
    uint source = active_indices[rank];
    if (source >= capacity)
    {
        pmAtomicOr(status, kPmSpatialAdmissionInvalid);
        return;
    }
    PmSpatialLeafMetadata leaf = metadata[source];
    if ((leaf.flags & kPmSpatialLeafProvisional) == 0u) return;
    uint pending[kPmSpatialQueryStackCapacity] = {};
    uint pending_count = count == 0u ? 0u : 1u;
    bool conflict = false;
    bool overflow = false;
    while (pending_count != 0u && !conflict && !overflow)
    {
        uint node = pending[--pending_count];
        if (node >= 2u * count - 1u)
        {
            overflow = true;
            pmAtomicOr(status, kPmSpatialAdmissionInvalid);
            continue;
        }
        if (!pmSpatialOverlaps(node_bounds[node], leaf_bounds[source])) continue;
        if (node >= count - 1u)
        {
            uint other_source = leaf_sources[node - (count - 1u)];
            if (other_source >= capacity)
            {
                overflow = true;
                pmAtomicOr(status, kPmSpatialAdmissionInvalid);
                continue;
            }
            if (other_source == source) continue;
            PmSpatialLeafMetadata other = metadata[other_source];
            if ((leaf.admission_mask & other.admission_mask) == 0u) continue;
            bool own_source = leaf.transfer_group != 0u &&
                leaf.transfer_group == other.transfer_group &&
                (leaf.flags & kPmSpatialLeafTransferDestination) != 0u &&
                (other.flags & kPmSpatialLeafTransferSource) != 0u;
            if (own_source) continue;
            bool retained = (other.flags & kPmSpatialLeafRetained) != 0u;
            bool lower_proposal =
                (other.flags & kPmSpatialLeafProvisional) != 0u &&
                pmSpatialIdentityLess(other.identity, leaf.identity);
            conflict = retained || lower_proposal;
            continue;
        }
        if (pending_count + 2u > kPmSpatialQueryStackCapacity)
        {
            overflow = true;
            continue;
        }
        pending[pending_count++] = right_children[node];
        pending[pending_count++] = left_children[node];
    }
    admission[source] = PMResourceUint8(!conflict && !overflow ? 1u : 0u);
    if (overflow) pmAtomicOr(status, kPmSpatialQueryOverflow);
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_finalize_transfers(
    uniform PMBuffer<PmSpatialLeafMetadata> metadata,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<uint> active_indices,
    uniform PMByteBuffer admission,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint rank = pmThreadIndex();
    uint count = pmSpatialLiveCount(active_count, capacity, status);
    if (rank >= count) return;
    uint source = active_indices[rank];
    if (source >= capacity)
    {
        pmAtomicOr(status, kPmSpatialAdmissionInvalid);
        return;
    }
    if (
        (metadata[source].flags & kPmSpatialLeafTransferSource) == 0u) return;
    uint group = metadata[source].transfer_group;
    uint destination = kPmSpatialInvalid;
    uint source_count = 0u;
    for (uint index = 0u; index < count; ++index)
    {
        uint other_source = active_indices[index];
        if (other_source >= capacity)
        {
            pmAtomicOr(status, kPmSpatialAdmissionInvalid);
            return;
        }
        if (metadata[other_source].transfer_group != group ||
            (metadata[other_source].flags & kPmSpatialLeafInactive) != 0u)
            continue;
        if ((metadata[other_source].flags &
             kPmSpatialLeafTransferSource) != 0u) ++source_count;
        if ((metadata[other_source].flags &
             kPmSpatialLeafTransferDestination) == 0u) continue;
        if (destination != kPmSpatialInvalid)
        {
            pmAtomicOr(status, kPmSpatialAdmissionInvalid);
            return;
        }
        destination = other_source;
    }
    if (group == 0u || source_count != 1u ||
        destination == kPmSpatialInvalid)
    {
        pmAtomicOr(status, kPmSpatialAdmissionInvalid);
        return;
    }
    admission[source] = PMResourceUint8(
        admission[destination] == 0u ? 1u : 0u);
}

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_spatial_publish_epoch(
    uniform PMBuffer<PmSpatialEpochHeader> header,
    uniform PMBuffer<uint> active_count, uniform uint capacity,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    PmSpatialEpochHeader value = header[0];
    if (value.publication_count != 0u)
    {
        pmAtomicOr(status, kPmSpatialDuplicatePublication);
        value.status |= kPmSpatialDuplicatePublication;
    }
    else
    {
        value.publication_count = 1u;
    }
    value.leaf_count = pmSpatialLiveCount(active_count, capacity, status);
    value.node_count = value.leaf_count == 0u ? 0u : 2u * value.leaf_count - 1u;
    value.status |= status[0];
    header[0] = value;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_spatial_query_aabbs(
    uniform PMBuffer<PmSpatialEpochHeader> header,
    uniform PMBuffer<uint> query_counter,
    uniform PMBuffer<PmSpatialAabb> queries,
    uniform PMBuffer<uint> query_masks,
    uniform PMBuffer<uint> excluded_sort_keys,
    uniform uint query_count, uniform uint candidates_per_query,
    uniform uint capacity,
    uniform PMBuffer<uint> left_children,
    uniform PMBuffer<uint> right_children,
    uniform PMBuffer<uint> leaf_sources,
    uniform PMBuffer<PmSpatialAabb> node_bounds,
    uniform PMBuffer<PmSpatialLeafMetadata> metadata,
    uniform PMByteBuffer admission,
    uniform PMBuffer<uint> candidate_counts,
    uniform PMBuffer<uint> candidates,
    uniform PMBuffer<uint> visited_nodes,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint query_index = pmThreadIndex();
    if (query_index >= query_count) return;
    candidate_counts[query_index] = 0u;
    visited_nodes[query_index] = 0u;
    PmSpatialEpochHeader epoch = header[0];
    if (epoch.publication_count != 1u || epoch.status != 0u)
    {
        if (query_index == 0u)
            pmAtomicOr(status, epoch.publication_count != 1u
                ? kPmSpatialStaleEpoch : epoch.status);
        return;
    }
    if (query_index == 0u)
    {
        uint previous = pmAtomicAdd(query_counter, query_count);
        if (query_count > 0xffffffffu - previous)
            pmAtomicOr(status, kPmSpatialCountOverflow);
    }
    uint count = epoch.leaf_count;
    if (count > capacity)
    {
        if (query_index == 0u) pmAtomicOr(status, kPmSpatialCountOverflow);
        return;
    }
    PmSpatialAabb query = queries[query_index];
    if (!pmSpatialValidBounds(query))
    {
        pmAtomicOr(status, kPmSpatialInvalidBounds);
        return;
    }
    if (count == 0u) return;
    uint pending[kPmSpatialQueryStackCapacity] = {};
    uint pending_count = 1u;
    uint written = 0u;
    bool overflow = false;
    while (pending_count != 0u && !overflow)
    {
        uint node = pending[--pending_count];
        visited_nodes[query_index] += 1u;
        if (node >= 2u * count - 1u)
        {
            overflow = true;
            pmAtomicOr(status, kPmSpatialAdmissionInvalid);
            continue;
        }
        if (!pmSpatialOverlaps(node_bounds[node], query)) continue;
        if (node >= count - 1u)
        {
            uint source = leaf_sources[node - (count - 1u)];
            if (source >= capacity)
            {
                overflow = true;
                pmAtomicOr(status, kPmSpatialAdmissionInvalid);
                continue;
            }
            PmSpatialLeafMetadata leaf = metadata[source];
            if (admission[source] == 0u ||
                (leaf.query_mask & query_masks[query_index]) == 0u ||
                leaf.sort_key == excluded_sort_keys[query_index]) continue;
            if (written == candidates_per_query)
            {
                overflow = true;
                continue;
            }
            uint base = query_index * candidates_per_query;
            uint insert = written;
            while (insert != 0u && pmSpatialIdentityLess(
                leaf.identity, metadata[candidates[base + insert - 1u]].identity))
            {
                candidates[base + insert] = candidates[base + insert - 1u];
                --insert;
            }
            candidates[base + insert] = source;
            ++written;
            continue;
        }
        if (pending_count + 2u > kPmSpatialQueryStackCapacity)
        {
            overflow = true;
            continue;
        }
        pending[pending_count++] = right_children[node];
        pending[pending_count++] = left_children[node];
    }
    if (overflow)
    {
        candidate_counts[query_index] = 0u;
        pmAtomicOr(status, kPmSpatialQueryOverflow);
    }
    else
    {
        candidate_counts[query_index] = written;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_spatial_validate_envelopes(
    uniform PMBuffer<PmSpatialLeafMetadata> epoch_metadata,
    uniform PMBuffer<PmSpatialAabb> epoch_bounds,
    uniform uint epoch_count,
    uniform PMBuffer<PmSpatialLeafMetadata> current_metadata,
    uniform PMBuffer<PmSpatialAabb> current_bounds,
    uniform uint current_count,
    uniform PMBuffer<uint> status PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint index = pmThreadIndex();
    if (index == 0u && current_count != epoch_count)
        pmAtomicOr(status, kPmSpatialStaleEpoch);
    if (index >= min(epoch_count, current_count)) return;
    if (!pmSpatialIdentityEqual(epoch_metadata[index].identity,
                                current_metadata[index].identity) ||
        epoch_metadata[index].sort_key != current_metadata[index].sort_key)
        pmAtomicOr(status, kPmSpatialStaleEpoch);
    if (!pmSpatialValidBounds(current_bounds[index]) ||
        !pmSpatialContains(epoch_bounds[index], current_bounds[index]))
        pmAtomicOr(status, kPmSpatialEnvelopeEscape);
}
