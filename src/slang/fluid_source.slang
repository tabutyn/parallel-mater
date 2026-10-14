// SPDX-License-Identifier: MIT
// Production deterministic fluid source kernel for Metal.

import metal_common;

struct MetalSourceConstants
{
    uint site_count;
    uint capacity;
    MetalPackedVec3 initial_velocity;
    float initial_temperature;
    float clearance;
    uint enabled;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    sourcePositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    sourceVelocities : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>
    sourceStableIds : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<float>
    sourceTemperatures : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalParticleMetadata>
    sourceMetadata : register(u4);
[[vk::binding(5, 0)]] StructuredBuffer<MetalPackedVec3>
    sourceSites : register(t5);
[[vk::binding(6, 0)]] ConstantBuffer<MetalSourceConstants>
    sourceConstants : register(b6);
[[vk::binding(7, 0)]] RWStructuredBuffer<MetalPackedVec3>
    sourcePrevious : register(u7);
[[vk::binding(8, 0)]] RWStructuredBuffer<float>
    sourceFoam : register(u8);
[[vk::binding(9, 0)]] RWStructuredBuffer<float>
    sourceFoamSources : register(u9);
[[vk::binding(10, 0)]] RWStructuredBuffer<MetalPackedVec3>
    sourceAccelerations : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<MetalContactEvent>
    sourceContactSamples : register(u11);
[[vk::binding(12, 0)]] RWStructuredBuffer<uint>
    sourceContactFlags : register(u12);
[[vk::binding(13, 0)]] RWStructuredBuffer<MetalSplitCounter>
    sourceCapacityMisses : register(u13);

groupshared uint sourceBlocked[64];
groupshared uint sourceLive;
groupshared uint sourceNextStableId;
groupshared uint sourceEmitted;

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_source(uint groupIndex : SV_GroupIndex)
{
    MetalSourceConstants constants = sourceConstants;
    if (constants.enabled == 0u) return;
    uint lane = groupIndex;
    float clearanceSquared = constants.clearance * constants.clearance;
    uint capacity = sourceMetadata[0].capacity;
    if (lane == 0u)
    {
        sourceLive = min(sourceMetadata[0].count, capacity);
        sourceNextStableId = sourceMetadata[0].next_stable_id;
        sourceEmitted = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint site = 0u; site < constants.site_count; ++site)
    {
        float3 candidate = metalLoad(sourceSites[site]);
        uint occupied = 0u;
        for (uint particle = lane; particle < sourceLive; particle += 64u)
        {
            float3 delta = metalLoad(sourcePositions[particle]) - candidate;
            if (dot(delta, delta) < clearanceSquared)
            {
                occupied = 1u;
                break;
            }
        }
        sourceBlocked[lane] = occupied;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 32u; stride != 0u; stride >>= 1u)
        {
            if (lane < stride)
                sourceBlocked[lane] |= sourceBlocked[lane + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0u && sourceBlocked[0] == 0u)
        {
            if (sourceLive >= capacity)
            {
                MetalSplitCounter misses = sourceCapacityMisses[0];
                uint next = misses.low + 1u;
                misses.low = next;
                if (next == 0u) ++misses.high;
                sourceCapacityMisses[0] = misses;
            }
            else
            {
                uint destination = sourceLive++;
                MetalPackedVec3 point = sourceSites[site];
                sourcePositions[destination] = point;
                sourcePrevious[destination] = point;
                sourceVelocities[destination] = constants.initial_velocity;
                sourceStableIds[destination] = sourceNextStableId++;
                sourceTemperatures[destination] = constants.initial_temperature;
                sourceFoam[destination] = 0.0f;
                sourceFoamSources[destination] = 0.0f;
                sourceAccelerations[destination] = metalStore(float3(0.0f));
                MetalContactEvent contact = metalZeroContactEvent();
                sourceContactSamples[destination] = contact;
                sourceContactFlags[destination] = 0u;
                ++sourceEmitted;
            }
        }
        AllMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
    {
        MetalParticleMetadata metadata = sourceMetadata[0];
        metadata.count = sourceLive;
        metadata.next_stable_id = sourceNextStableId;
        metadata.emitted += uint64_t(sourceEmitted);
        metadata.revision += sourceEmitted;
        sourceMetadata[0] = metadata;
    }
}
