// SPDX-License-Identifier: MIT
#include "rigid.hlsl"
RWStructuredBuffer<uint> compact_blocks : register(u20);
groupshared uint compact_scan[64];

// Stable block-local prefix sums: IDs stay in ascending pair order.
[numthreads(64,1,1)]
void rigid_compact_count(uint3 group : SV_GroupID,uint lane : SV_GroupIndex) {
    uint pair_count=step_body_count*step_body_count;
    uint pair=group.x*64u+lane;
    uint live=0u;
    if(pair<pair_count) live=manifolds[pair].count!=0u ? 1u : 0u;
    compact_scan[lane]=live;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for(uint offset=1u;offset<64u;offset*=2u) {
        uint source_lane=lane>=offset ? lane-offset : 0u;
        uint addend=lane>=offset ? compact_scan[source_lane] : 0u;
        GroupMemoryBarrierWithGroupSync();
        compact_scan[lane]+=addend;
        GroupMemoryBarrierWithGroupSync();
    }
    if(live!=0u)
        active_pairs[pair_count+group.x*64u+compact_scan[lane]-1u]=pair;
    if(lane==63u) compact_blocks[group.x]=compact_scan[63];
}

[numthreads(1,1,1)]
void rigid_compact_prefix(uint3 unused : SV_DispatchThreadID) {
    uint block_count=(step_body_count*step_body_count+63u)/64u;
    uint cursor=0u;
    [loop] for(uint block=0u;block<block_count;++block) {
        uint count=compact_blocks[block];
        compact_blocks[block]=cursor;
        cursor+=count;
    }
    compact_blocks[block_count]=cursor;
    counters[6]=cursor;
}

[numthreads(64,1,1)]
void rigid_compact_scatter(uint3 group : SV_GroupID,uint lane : SV_GroupIndex) {
    uint pair_count=step_body_count*step_body_count;
    uint block_count=(pair_count+63u)/64u;
    if(group.x>=block_count) return;
    uint begin=compact_blocks[group.x],end=compact_blocks[group.x+1u];
    if(begin+lane<end)
        active_pairs[begin+lane]=active_pairs[pair_count+group.x*64u+lane];
}
