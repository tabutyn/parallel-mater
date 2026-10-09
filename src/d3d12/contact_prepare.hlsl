// SPDX-License-Identifier: MIT
#include "rigid.hlsl"

// Reads shared body state but writes only this pair's manifold/response rows.
[numthreads(64,1,1)]
void rigid_contact_prepare(uint3 id : SV_DispatchThreadID) {
    uint stride=((step_body_count+63u)/64u)*64u;
    [loop] for(uint candidate=id.x;candidate<counters[6];candidate+=stride) {
        uint pair=active_pairs[candidate];
        ContactManifold manifold=manifolds[pair];
        prepare_persistent_pair(pair,manifold);
        manifolds[pair]=manifold;
    }
}

[numthreads(64,1,1)]
void rigid_contact_events(uint3 id : SV_DispatchThreadID) {
    uint stride=((step_body_count+63u)/64u)*64u;
    [loop] for(uint candidate=id.x;candidate<counters[6];candidate+=stride) {
        uint pair=active_pairs[candidate];
        ContactManifold manifold=manifolds[pair];
        uint body=pair/step_body_count,collider=pair-body*step_body_count;
        [loop] for(uint contact_index=0u;contact_index<manifold.count;++contact_index) {
            uint index=manifold.event_offset+contact_index;
            if(index>=step_event_capacity) continue;
            ContactRecord record=manifold.contacts[contact_index];
            RigidContactEvent event;
            event.body=body_ids[body];event.collider=body_ids[collider];
            event.position=record.position;event.normal=record.normal;
            event.penetration=max(0.0f,record.penetration);
            event.normal_impulse=0.0f;
            event.friction_impulse=store3(float3(0,0,0));
            contact_events[index]=event;
        }
    }
}
