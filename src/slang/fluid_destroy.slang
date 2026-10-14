// SPDX-License-Identifier: MIT
// Production deterministic fluid destroy-volume kernel for Metal.

import metal_common;

struct MetalDestroyConstants
{
    MetalPackedVec3 center;
    MetalQuaternion orientation;
    MetalPackedVec2 half_extents;
    uint crossing;
    uint enabled;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    destroyPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    destroyPrevious : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    destroyVelocities : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<MetalPackedVec3>
    destroyAccelerations : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>
    destroyStableIds : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<float>
    destroyFoam : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<float>
    destroyTemperatures : register(u6);
[[vk::binding(7, 0)]] RWStructuredBuffer<MetalParticleMetadata>
    destroyMetadata : register(u7);
[[vk::binding(8, 0)]] RWStructuredBuffer<float>
    destroyFoamSources : register(u8);
[[vk::binding(9, 0)]] ConstantBuffer<MetalDestroyConstants>
    destroyConstants : register(b9);
[[vk::binding(10, 0)]] RWStructuredBuffer<MetalContactEvent>
    destroyContactSamples : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<uint>
    destroyContactFlags : register(u11);

groupshared uint destroyPrefix[64];
groupshared uint destroyDestinationBase;
groupshared uint destroyRemoved;

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_destroy(uint groupIndex : SV_GroupIndex)
{
    MetalDestroyConstants constants = destroyConstants;
    if (constants.enabled == 0u) return;
    uint lane = groupIndex;
    uint sourceCount = destroyMetadata[0].count;
    MetalQuaternion inverse = constants.orientation;
    inverse.x = -inverse.x;
    inverse.y = -inverse.y;
    inverse.z = -inverse.z;
    if (lane == 0u)
    {
        destroyDestinationBase = 0u;
        destroyRemoved = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint block = 0u; block < sourceCount; block += 64u)
    {
        uint source = block + lane;
        bool valid = source < sourceCount;
        MetalPackedVec3 position = metalStore(float3(0.0f));
        MetalPackedVec3 oldPosition = metalStore(float3(0.0f));
        MetalPackedVec3 velocity = metalStore(float3(0.0f));
        MetalPackedVec3 acceleration = metalStore(float3(0.0f));
        uint stableId = 0u;
        float foamValue = 0.0f;
        float foamSource = 0.0f;
        float temperature = 0.0f;
        MetalContactEvent contactSample = metalZeroContactEvent();
        uint contactFlag = 0u;
        uint keep = 0u;
        if (valid)
        {
            position = destroyPositions[source];
            oldPosition = destroyPrevious[source];
            velocity = destroyVelocities[source];
            acceleration = destroyAccelerations[source];
            stableId = destroyStableIds[source];
            foamValue = destroyFoam[source];
            foamSource = destroyFoamSources[source];
            temperature = destroyTemperatures[source];
            contactSample = destroyContactSamples[source];
            contactFlag = destroyContactFlags[source];
            float3 oldLocal = metalRotate(
                inverse, metalLoad(oldPosition) - metalLoad(constants.center));
            float3 newLocal = metalRotate(
                inverse, metalLoad(position) - metalLoad(constants.center));
            bool along = oldLocal.y <= 0.0f && newLocal.y > 0.0f;
            bool against = oldLocal.y >= 0.0f && newLocal.y < 0.0f;
            bool direction = constants.crossing == 0u
                ? along : (constants.crossing == 1u
                    ? against : (along || against));
            bool inside = abs(newLocal.x) <= constants.half_extents.x &&
                abs(newLocal.z) <= constants.half_extents.y;
            keep = uint(!(direction && inside));
        }
        destroyPrefix[lane] = keep;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 1u; offset < 64u; offset <<= 1u)
        {
            uint addend = lane >= offset
                ? destroyPrefix[lane - offset] : 0u;
            GroupMemoryBarrierWithGroupSync();
            destroyPrefix[lane] += addend;
            GroupMemoryBarrierWithGroupSync();
        }
        if (keep != 0u)
        {
            uint destination = destroyDestinationBase +
                (lane == 0u ? 0u : destroyPrefix[lane - 1u]);
            destroyPositions[destination] = position;
            destroyPrevious[destination] = oldPosition;
            destroyVelocities[destination] = velocity;
            destroyAccelerations[destination] = acceleration;
            destroyStableIds[destination] = stableId;
            destroyFoam[destination] = foamValue;
            destroyFoamSources[destination] = foamSource;
            destroyTemperatures[destination] = temperature;
            destroyContactSamples[destination] = contactSample;
            destroyContactFlags[destination] = contactFlag;
        }
        AllMemoryBarrierWithGroupSync();
        if (lane == 0u)
        {
            uint kept = destroyPrefix[63];
            destroyDestinationBase += kept;
            destroyRemoved += min(64u, sourceCount - block) - kept;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
    {
        MetalParticleMetadata metadata = destroyMetadata[0];
        metadata.count = destroyDestinationBase;
        metadata.revision += destroyRemoved;
        metadata.destroyed += uint64_t(destroyRemoved);
        destroyMetadata[0] = metadata;
    }
}
