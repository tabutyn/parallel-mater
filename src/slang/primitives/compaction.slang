// SPDX-License-Identifier: MIT
// Stable flagged-index compaction. One 256-thread workgroup owns one tile.

import device_intrinsics;
import graphics_resources;

groupshared uint pmCompactionFlags[256];
groupshared uint pmCompactionCount;

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_compact_count_blocks(
    uniform PMByteBuffer flags, uniform uint count, uniform uint tile_size,
    uniform PMBuffer<uint> block_counts PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint block = pmBlockIndex().x;
    uint thread = pmGroupThreadIndex().x;
    uint index = block * tile_size + thread;
    if (thread == 0u) pmCompactionCount = 0u;
    pmGroupBarrier();
    if (index < count && flags[index] != 0u)
    {
        uint ignored;
        InterlockedAdd(pmCompactionCount, 1u, ignored);
    }
    pmGroupBarrier();
    if (thread == 0u) block_counts[block] = pmCompactionCount;
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_compact_scatter_indices(
    uniform PMByteBuffer flags, uniform uint count, uniform uint tile_size,
    uniform PMBuffer<uint> block_offsets, uniform PMBuffer<uint> output
    PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint block = pmBlockIndex().x;
    uint thread = pmGroupThreadIndex().x;
    uint index = block * tile_size + thread;
    uint selected = index < count && flags[index] != 0u ? 1u : 0u;
    pmCompactionFlags[thread] = selected;
    pmGroupBarrier();
    if (selected == 0u) return;
    uint rank = 0u;
    for (uint lane = 0u; lane < thread; ++lane)
        rank += pmCompactionFlags[lane];
    output[block_offsets[block] + rank] = index;
}
