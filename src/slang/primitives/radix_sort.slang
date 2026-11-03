// SPDX-License-Identifier: MIT
// Stable four-bit LSD radix sort for 64-bit keys and 32-bit values.

import device_intrinsics;
import graphics_resources;

static const uint kPmRadix = 16u;
groupshared uint pmRadixCounts[16];
groupshared uint pmRadixDigits[256];

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_radix_histogram_u64(
    uniform PMBuffer<uint64_t> keys, uniform uint count, uniform uint tile_size,
    uniform uint shift, uniform uint block_count, uniform PMBuffer<uint> histograms
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint block = pmBlockIndex().x;
    uint thread = pmGroupThreadIndex().x;
    if (block >= block_count) return;
    if (thread < kPmRadix) pmRadixCounts[thread] = 0u;
    pmGroupBarrier();
    uint index = block * tile_size + thread;
    if (index < count)
    {
        uint digit = uint((keys[index] >> shift) & 15ull);
        uint ignored;
        InterlockedAdd(pmRadixCounts[digit], 1u, ignored);
    }
    pmGroupBarrier();
    if (thread < kPmRadix)
        histograms[thread * block_count + block] = pmRadixCounts[thread];
}

[shader("compute")]
[numthreads(16, 1, 1)]
public void pm_radix_prefix_u32(
    uniform PMBuffer<uint> histograms, uniform uint block_count,
    uniform PMBuffer<uint> offsets PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint digit = pmGroupThreadIndex().x;
    if (pmBlockIndex().x != 0u || digit >= kPmRadix) return;
    uint total = 0u;
    for (uint block = 0u; block < block_count; ++block)
        total += histograms[digit * block_count + block];
    pmRadixCounts[digit] = total;
    pmGroupBarrier();
    uint running = 0u;
    for (uint earlier = 0u; earlier < digit; ++earlier)
        running += pmRadixCounts[earlier];
    for (uint block = 0u; block < block_count; ++block)
    {
        uint address = digit * block_count + block;
        offsets[address] = running;
        running += histograms[address];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_radix_scatter_u64_u32(
    uniform PMBuffer<uint64_t> input_keys, uniform PMBuffer<uint> input_values,
    uniform uint count, uniform uint tile_size, uniform uint shift,
    uniform uint block_count, uniform PMBuffer<uint> offsets,
    uniform PMBuffer<uint64_t> output_keys, uniform PMBuffer<uint> output_values
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint block = pmBlockIndex().x;
    uint thread = pmGroupThreadIndex().x;
    if (block >= block_count) return;
    uint index = block * tile_size + thread;
    uint64_t key = index < count ? input_keys[index] : 0ull;
    uint digit = index < count ? uint((key >> shift) & 15ull) : kPmRadix;
    pmRadixDigits[thread] = digit;
    pmGroupBarrier();
    if (index >= count) return;
    uint rank = 0u;
    for (uint lane = 0u; lane < thread; ++lane)
        rank += pmRadixDigits[lane] == digit ? 1u : 0u;
    uint destination = offsets[digit * block_count + block] + rank;
    output_keys[destination] = key;
    output_values[destination] = input_values[index];
}
