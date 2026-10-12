// SPDX-License-Identifier: MIT
// Stable four-bit LSD radix sort for 64-bit keys and 32-bit values.

import device_intrinsics;

static const uint kPmRadix = 16u;
groupshared uint pmRadixCounts[16];
groupshared uint pmRadixDigits[256];

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_radix_histogram_u64(
    uniform uint64_t* keys, uniform uint count, uniform uint tile_size,
    uniform uint shift, uniform uint block_count, uniform uint* histograms)
{
    uint block = cudaBlockIdx().x;
    uint thread = cudaThreadIdx().x;
    if (block >= block_count) return;
    if (thread < kPmRadix) pmRadixCounts[thread] = 0u;
    pmGroupBarrier();
    uint index = block * tile_size + thread;
    if (index < count)
    {
        uint digit = uint((keys[index] >> shift) & 15ull);
        pmAtomicAdd(&pmRadixCounts[digit], 1u);
    }
    pmGroupBarrier();
    if (thread < kPmRadix)
        histograms[thread * block_count + block] = pmRadixCounts[thread];
}

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_radix_prefix_u32(
    uniform uint* histograms, uniform uint block_count,
    uniform uint* offsets)
{
    if (pmThreadIndex() != 0u) return;
    uint base = 0u;
    for (uint digit = 0u; digit < kPmRadix; ++digit)
    {
        uint running = base;
        for (uint block = 0u; block < block_count; ++block)
        {
            uint address = digit * block_count + block;
            offsets[address] = running;
            running += histograms[address];
        }
        base = running;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_radix_scatter_u64_u32(
    uniform uint64_t* input_keys, uniform uint* input_values,
    uniform uint count, uniform uint tile_size, uniform uint shift,
    uniform uint block_count, uniform uint* offsets,
    uniform uint64_t* output_keys, uniform uint* output_values)
{
    uint block = cudaBlockIdx().x;
    uint thread = cudaThreadIdx().x;
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
