// SPDX-License-Identifier: MIT
// Stable flagged-index compaction. One 256-thread workgroup owns one tile.

import device_intrinsics;

groupshared uint pmCompactionFlags[256];
groupshared uint pmCompactionCount;

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_compact_count_blocks(
    uniform uint8_t* flags, uniform uint count, uniform uint tile_size,
    uniform uint* block_counts)
{
    uint block = pmBlockIndex().x;
    uint thread = pmGroupThreadIndex().x;
    uint index = block * tile_size + thread;
    if (thread == 0u) pmCompactionCount = 0u;
    pmGroupBarrier();
    if (index < count && flags[index] != 0u)
        pmAtomicAdd(&pmCompactionCount, 1u);
    pmGroupBarrier();
    if (thread == 0u) block_counts[block] = pmCompactionCount;
}

[shader("compute")]
[numthreads(256, 1, 1)]
public void pm_compact_scatter_indices(
    uniform uint8_t* flags, uniform uint count, uniform uint tile_size,
    uniform uint* block_offsets, uniform uint* output)
{
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
