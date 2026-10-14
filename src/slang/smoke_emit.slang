// SPDX-License-Identifier: MIT
// Production smoke emission kernel for Metal.

import smoke_common;
import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    emitPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    emitPrevious : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    emitVelocities : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<float>
    emitAges : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<float>
    emitDensities : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<float>
    emitPressures : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<MetalPackedVec3>
    emitVorticities : register(u6);
[[vk::binding(7, 0)]] RWStructuredBuffer<float>
    emitThermalLift : register(u7);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalSmokeMetadata>
    emitMetadata : register(u8);
[[vk::binding(9, 0)]] ConstantBuffer<MetalSmokeConstants>
    emitConstants : register(b9);

[ForceInline]
static uint smokeHash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

[ForceInline]
static float smokeHashUnit(uint value)
{
    return float(smokeHash(value) & 0x00ffffffu) / 16777216.0f;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void pm_smoke_emit(uint groupIndex : SV_GroupIndex)
{
    MetalSmokeConstants constants = emitConstants;
    uint spawnCount = min(constants.command, constants.capacity);
    uint firstSlot = emitMetadata[0].next_particle;
    uint64_t firstSerial = emitMetadata[0].emitted;
    GroupMemoryBarrierWithGroupSync();
    for (uint spawn = groupIndex; spawn < spawnCount; spawn += 128u)
    {
        uint slot = (firstSlot + spawn) % constants.capacity;
        uint serial = uint(firstSerial + uint64_t(spawn));
        float y = (smokeHashUnit(serial ^ 0x132aef41u) * 2.0f - 1.0f) *
            constants.emitter_half_extents.x;
        float z = (smokeHashUnit(serial ^ 0xa385c9d3u) * 2.0f - 1.0f) *
            constants.emitter_half_extents.y;
        MetalPackedVec3 point = metalStore(
            metalLoad(constants.emitter_center) + float3(0.0f, y, z));
        emitPositions[slot] = point;
        emitPrevious[slot] = point;
        emitVelocities[slot] = constants.initial_velocity;
        emitAges[slot] = 0.0f;
        emitDensities[slot] = 0.0f;
        emitPressures[slot] = 0.0f;
        emitVorticities[slot] = metalStore(float3(0.0f));
        emitThermalLift[slot] = 0.0f;
    }
    AllMemoryBarrierWithGroupSync();
    if (groupIndex == 0u)
    {
        MetalSmokeMetadata metadata = emitMetadata[0];
        metadata.count = min(constants.capacity, metadata.count + spawnCount);
        metadata.revision += spawnCount;
        metadata.next_particle =
            (firstSlot + spawnCount) % constants.capacity;
        metadata.emitted += uint64_t(spawnCount);
        emitMetadata[0] = metadata;
    }
}
