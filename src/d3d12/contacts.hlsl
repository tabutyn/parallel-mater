// SPDX-License-Identifier: MIT
#include "rigid.hlsl"

void solve_contact_color(uint lane,uint lane_stride) {
    if(step_solver_pass_begin>counters[5]) return;
    uint color=step_solver_color;
    uint begin,end;
    if(color==24u) {
        if(lane!=0u || counters[3]==0u) return;
        begin=counters[31];end=counters[32];lane_stride=1u;
    } else {
        if(color>=counters[2]) return;
        begin=counters[7u+color];end=counters[8u+color];
    }
    uint pair_count=step_body_count*step_body_count;
    [loop] for(uint index=begin+lane;index<end;index+=lane_stride)
        solve_compact_pair(active_pairs[pair_count+index],
                           step_solver_pass_begin,counters[4]!=0u);
}

// Separate from constraint/finalization code so its register footprint cannot
// reduce contact-kernel occupancy or inflate empty-color dispatches.
[numthreads(64,1,1)]
void rigid_contacts(uint3 id : SV_DispatchThreadID) {
    solve_contact_color(id.x,step_solver_pass_count);
}
[numthreads(1,1,1)]
void rigid_contacts_warp(uint lane : SV_GroupIndex) { solve_contact_color(lane,1u); }
