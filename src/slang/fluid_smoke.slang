// SPDX-License-Identifier: MIT
// Shared fluid-smoke kernels for the CUDA mirror and production Metal backend.

#ifdef PM_TARGET_METAL
import smoke_common;
import metal_common;

struct MetalFluidSmokeConstants
{
    float timestep;
    MetalPackedVec3 gravity;
    MetalPackedVec3 heater_center;
    MetalQuaternion heater_orientation;
    MetalPackedVec2 heater_half_extents;
    float heater_temperature;
    float boiling_temperature;
    float heat_transfer_rate;
    float wind_drag;
    float steam_rise_speed;
    float water_radius;
    float smoke_radius;
    float smoke_rest_number_density;
    float smoke_maximum_speed;
    float smoke_lifetime;
    uint fluid_capacity;
    uint smoke_capacity;
    uint grid_resolution;
    uint grid_vertical_resolution;
    float grid_spacing;
    MetalPackedVec3 grid_minimum;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSmokeFluidPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSmokeFluidPrevious : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSmokeFluidVelocities : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSmokeFluidAccelerations : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>
    metalFluidSmokeFluidIds : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<float>
    metalFluidSmokeFluidFoam : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<float>
    metalFluidSmokeFluidFoamSources : register(u6);
[[vk::binding(7, 0)]] RWStructuredBuffer<float>
    metalFluidSmokeFluidTemperatures : register(u7);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalParticleMetadata>
    metalFluidSmokeFluidMetadata : register(u8);
[[vk::binding(9, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSmokeSmokePositions : register(u9);
[[vk::binding(10, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSmokeSmokePrevious : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSmokeSmokeVelocities : register(u11);
[[vk::binding(12, 0)]] RWStructuredBuffer<float>
    metalFluidSmokeSmokeAges : register(u12);
[[vk::binding(13, 0)]] RWStructuredBuffer<float>
    metalFluidSmokeSmokeDensities : register(u13);
[[vk::binding(14, 0)]] RWStructuredBuffer<float>
    metalFluidSmokeSmokePressures : register(u14);
[[vk::binding(15, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSmokeSmokeVorticities : register(u15);
[[vk::binding(16, 0)]] RWStructuredBuffer<float>
    metalFluidSmokeSmokeThermalLift : register(u16);
[[vk::binding(17, 0)]] RWStructuredBuffer<MetalSmokeMetadata>
    metalFluidSmokeSmokeMetadata : register(u17);
[[vk::binding(18, 0)]] ConstantBuffer<MetalFluidSmokeConstants>
    metalFluidSmokeConstants : register(b18);
[[vk::binding(19, 0)]] RWStructuredBuffer<MetalContactEvent>
    metalFluidSmokeContactSamples : register(u19);
[[vk::binding(20, 0)]] RWStructuredBuffer<uint>
    metalFluidSmokeContactFlags : register(u20);
[[vk::binding(21, 0)]] RWStructuredBuffer<float>
    metalFluidSmokeGridDensity : register(u21);
[[vk::binding(22, 0)]] RWStructuredBuffer<float>
    metalFluidSmokeGridFaceVelocity : register(u22);

groupshared uint metalFluidSmokePrefix[64];
groupshared uint metalFluidSmokeDestinationBase;
groupshared uint metalFluidSmokeConvertedBase;
groupshared uint metalFluidSmokeSmokeCount;

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_smoke_update(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint source = dispatchThreadId.x;
    MetalFluidSmokeConstants constants = metalFluidSmokeConstants;
    uint sourceCount = min(
        metalFluidSmokeFluidMetadata[0].count, constants.fluid_capacity);
    if (source >= sourceCount) return;
    uint smokeCount = min(
        metalFluidSmokeSmokeMetadata[0].count, constants.smoke_capacity);
    float support = 3.0f * constants.smoke_radius;
    float supportSquared = support * support;
    float3 position = metalLoad(metalFluidSmokeFluidPositions[source]);
    float3 velocity = metalLoad(metalFluidSmokeFluidVelocities[source]);
    float smokeWeight = 0.0f;
    float3 smokeVelocity = float3(0.0f);
    float occupancy = 0.0f;
    float3 gridPoint = (position - metalLoad(constants.grid_minimum)) /
        max(constants.grid_spacing, 1.0e-6f);
    bool insideGrid = constants.grid_resolution != 0u && smokeCount != 0u &&
        all(gridPoint >= float3(0.0f)) &&
        gridPoint.x < float(constants.grid_resolution) &&
        gridPoint.y < float(constants.grid_vertical_resolution) &&
        gridPoint.z < float(constants.grid_resolution);
    if (insideGrid)
    {
        smokeVelocity = metalSmokeSampleFaceVelocity(
            metalFluidSmokeGridFaceVelocity, 0u, position,
            metalLoad(constants.grid_minimum), constants.grid_spacing,
            constants.grid_resolution, constants.grid_vertical_resolution);
        float density = metalSmokeSampleCellScalar(
            metalFluidSmokeGridDensity, position,
            metalLoad(constants.grid_minimum), constants.grid_spacing,
            constants.grid_resolution, constants.grid_vertical_resolution);
        occupancy = clamp(
            density * constants.smoke_rest_number_density, 0.0f, 1.0f);
    }
    else
    {
        for (uint smoke = 0u; smoke < smokeCount; ++smoke)
        {
            if (metalFluidSmokeSmokeAges[smoke] >= constants.smoke_lifetime)
                continue;
            float3 delta = position -
                metalLoad(metalFluidSmokeSmokePositions[smoke]);
            float squared = dot(delta, delta);
            if (squared >= supportSquared) continue;
            float q = 1.0f - sqrt(max(squared, 0.0f)) / support;
            float weight = q * q * q;
            smokeWeight += weight;
            smokeVelocity +=
                metalLoad(metalFluidSmokeSmokeVelocities[smoke]) * weight;
        }
        if (smokeWeight > 1.0e-6f)
        {
            smokeVelocity /= smokeWeight;
            occupancy = min(
                1.0f, smokeWeight / constants.smoke_rest_number_density);
        }
    }
    if (occupancy > 1.0e-6f)
    {
        float response = 1.0f -
            exp(-constants.wind_drag * occupancy * constants.timestep);
        float maximum = max(constants.smoke_maximum_speed, length(velocity));
        velocity = metalLimit(
            velocity + (smokeVelocity - velocity) * response, maximum);
    }
    metalFluidSmokeFluidVelocities[source] = metalStore(velocity);

    MetalQuaternion inverseHeater = constants.heater_orientation;
    inverseHeater.x = -inverseHeater.x;
    inverseHeater.y = -inverseHeater.y;
    inverseHeater.z = -inverseHeater.z;
    float3 local = metalRotate(
        inverseHeater, position - metalLoad(constants.heater_center));
    bool heated = abs(local.x) <= constants.heater_half_extents.x &&
        abs(local.z) <= constants.heater_half_extents.y &&
        local.y >= -0.5f * constants.water_radius &&
        local.y <= 2.5f * constants.water_radius;
    float temperature = metalFluidSmokeFluidTemperatures[source];
    if (heated && constants.heater_temperature > temperature)
    {
        temperature += (constants.heater_temperature - temperature) *
            (1.0f - exp(-constants.heat_transfer_rate * constants.timestep));
    }
    metalFluidSmokeFluidTemperatures[source] = temperature;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_smoke_compact(uint groupIndex : SV_GroupIndex)
{
    uint lane = groupIndex;
    MetalFluidSmokeConstants constants = metalFluidSmokeConstants;
    uint sourceCount = min(
        metalFluidSmokeFluidMetadata[0].count, constants.fluid_capacity);
    uint firstSmokeParticle = metalFluidSmokeSmokeMetadata[0].next_particle;
    float3 gravity = metalLoad(constants.gravity);
    float gravitySquared = dot(gravity, gravity);
    float3 up = gravitySquared > 1.0e-12f
        ? -normalize(gravity) : float3(0.0f, 1.0f, 0.0f);
    if (lane == 0u)
    {
        metalFluidSmokeDestinationBase = 0u;
        metalFluidSmokeConvertedBase = 0u;
        metalFluidSmokeSmokeCount = metalFluidSmokeSmokeMetadata[0].count;
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
        if (valid)
        {
            position = metalFluidSmokeFluidPositions[source];
            oldPosition = metalFluidSmokeFluidPrevious[source];
            velocity = metalFluidSmokeFluidVelocities[source];
            acceleration = metalFluidSmokeFluidAccelerations[source];
            stableId = metalFluidSmokeFluidIds[source];
            foamValue = metalFluidSmokeFluidFoam[source];
            foamSource = metalFluidSmokeFluidFoamSources[source];
            temperature = metalFluidSmokeFluidTemperatures[source];
            contactSample = metalFluidSmokeContactSamples[source];
            contactFlag = metalFluidSmokeContactFlags[source];
        }

        uint hot = uint(valid && temperature >= constants.boiling_temperature);
        metalFluidSmokePrefix[lane] = hot;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 1u; offset < 64u; offset <<= 1u)
        {
            uint addend = lane >= offset
                ? metalFluidSmokePrefix[lane - offset] : 0u;
            GroupMemoryBarrierWithGroupSync();
            metalFluidSmokePrefix[lane] += addend;
            GroupMemoryBarrierWithGroupSync();
        }
        uint hotCount = metalFluidSmokePrefix[63];
        uint ordinal = metalFluidSmokeConvertedBase +
            (hot == 0u ? 0u : metalFluidSmokePrefix[lane] - 1u);
        uint accepted = uint(
            hot != 0u && ordinal < constants.smoke_capacity);
        uint smokeSlot = 0u;
        if (accepted != 0u)
        {
            smokeSlot = (firstSmokeParticle + ordinal) %
                constants.smoke_capacity;
            metalFluidSmokeSmokePositions[smokeSlot] = position;
            metalFluidSmokeSmokePrevious[smokeSlot] = position;
            metalFluidSmokeSmokeVelocities[smokeSlot] = metalStore(
                metalLoad(velocity) + up * constants.steam_rise_speed);
            metalFluidSmokeSmokeAges[smokeSlot] = 0.0f;
            metalFluidSmokeSmokeDensities[smokeSlot] = 0.0f;
            metalFluidSmokeSmokePressures[smokeSlot] = 0.0f;
            metalFluidSmokeSmokeVorticities[smokeSlot] =
                metalStore(float3(0.0f));
            metalFluidSmokeSmokeThermalLift[smokeSlot] =
                constants.steam_rise_speed;
        }
        AllMemoryBarrierWithGroupSync();

        metalFluidSmokePrefix[lane] =
            accepted != 0u ? smokeSlot + 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 32u; stride != 0u; stride >>= 1u)
        {
            if (lane < stride)
                metalFluidSmokePrefix[lane] = max(
                    metalFluidSmokePrefix[lane],
                    metalFluidSmokePrefix[lane + stride]);
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0u)
            metalFluidSmokeSmokeCount = max(
                metalFluidSmokeSmokeCount, metalFluidSmokePrefix[0]);
        GroupMemoryBarrierWithGroupSync();

        uint keep = uint(valid && accepted == 0u);
        metalFluidSmokePrefix[lane] = keep;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 1u; offset < 64u; offset <<= 1u)
        {
            uint addend = lane >= offset
                ? metalFluidSmokePrefix[lane - offset] : 0u;
            GroupMemoryBarrierWithGroupSync();
            metalFluidSmokePrefix[lane] += addend;
            GroupMemoryBarrierWithGroupSync();
        }
        if (keep != 0u)
        {
            uint destination = metalFluidSmokeDestinationBase +
                (lane == 0u ? 0u : metalFluidSmokePrefix[lane - 1u]);
            metalFluidSmokeFluidPositions[destination] = position;
            metalFluidSmokeFluidPrevious[destination] = oldPosition;
            metalFluidSmokeFluidVelocities[destination] = velocity;
            metalFluidSmokeFluidAccelerations[destination] = acceleration;
            metalFluidSmokeFluidIds[destination] = stableId;
            metalFluidSmokeFluidFoam[destination] = foamValue;
            metalFluidSmokeFluidFoamSources[destination] = foamSource;
            metalFluidSmokeFluidTemperatures[destination] = temperature;
            metalFluidSmokeContactSamples[destination] = contactSample;
            metalFluidSmokeContactFlags[destination] = contactFlag;
        }
        AllMemoryBarrierWithGroupSync();
        if (lane == 0u)
        {
            uint acceptedCount = min(
                hotCount, constants.smoke_capacity -
                    metalFluidSmokeConvertedBase);
            metalFluidSmokeConvertedBase += acceptedCount;
            metalFluidSmokeDestinationBase += metalFluidSmokePrefix[63];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u)
    {
        MetalParticleMetadata fluidMetadata = metalFluidSmokeFluidMetadata[0];
        fluidMetadata.count = metalFluidSmokeDestinationBase;
        fluidMetadata.boiled += uint64_t(metalFluidSmokeConvertedBase);
        if (metalFluidSmokeConvertedBase != 0u) ++fluidMetadata.revision;
        metalFluidSmokeFluidMetadata[0] = fluidMetadata;

        MetalSmokeMetadata smokeMetadata = metalFluidSmokeSmokeMetadata[0];
        smokeMetadata.next_particle =
            firstSmokeParticle + metalFluidSmokeConvertedBase;
        smokeMetadata.count = min(
            constants.smoke_capacity, metalFluidSmokeSmokeCount);
        smokeMetadata.emitted += uint64_t(metalFluidSmokeConvertedBase);
        smokeMetadata.revision += metalFluidSmokeConvertedBase;
        metalFluidSmokeSmokeMetadata[0] = smokeMetadata;
    }
}

#else
import types;
import math;
import device_intrinsics;
import geometry_smoke;
import smoke_grid;

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_smoke_drag(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform uint* water_count, uniform FluidSmokeCouplingOptions coupling,
    uniform SmokeOptions smoke, uniform Vec3* smoke_positions,
    uniform Vec3* smoke_velocities, uniform uint64_t* smoke_keys,
    uniform uint* smoke_indices, uniform float dt)
{
    uint index = pmThreadIndex();
    if (index >= *water_count) return;
    SmokeFlowSample local = smoke_sample_flow(positions[index],
        smoke_positions, smoke_velocities, smoke_keys, smoke_indices, smoke);
    if (local.number_density <= 1.0e-6f) return;
    float occupancy = min(
        1.0f, local.number_density / smoke.rest_number_density);
    float response = 1.0f - exp(-coupling.wind_drag * occupancy * dt);
    Vec3 velocity = velocities[index];
    velocities[index] = clamp_length(add(velocity, multiply(
        subtract(local.velocity, velocity), response)),
        max(smoke.maximum_speed, sqrt(length_squared(velocity))));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_smoke_grid_drag(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform uint* water_count, uniform FluidSmokeCouplingOptions coupling,
    uniform SmokeOptions smoke, uniform SmokeGridField grid, uniform float dt)
{
    uint index = pmThreadIndex();
    if (index >= *water_count ||
        !smoke_grid_contains(positions[index], grid)) return;
    Vec3 air = {};
    float density = 0.0f;
    smoke_grid_sample(positions[index], grid, air, density);
    if (density <= 1.0e-6f) return;
    float response = 1.0f - exp(-coupling.wind_drag *
        clamp_scalar(density * smoke.rest_number_density, 0.0f, 1.0f) * dt);
    Vec3 velocity = velocities[index];
    velocities[index] = clamp_length(add(velocity, multiply(
        subtract(air, velocity), response)),
        max(smoke.maximum_speed, sqrt(length_squared(velocity))));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_smoke_exchange(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* temperatures, uniform uint* water_count,
    uniform float water_radius, uniform FluidSmokeCouplingOptions coupling,
    uniform SmokeOptions smoke, uniform Vec3* smoke_positions,
    uniform Vec3* smoke_velocities, uniform float* smoke_ages,
    uniform float* smoke_thermal_lift, uniform float* smoke_densities,
    uniform float* smoke_pressures, uniform Vec3* smoke_vorticities,
    uniform uint first_smoke_slot, uniform uint* converted,
    uniform uint8_t* keep, uniform float dt, uniform Vec3 gravity)
{
    uint index = pmThreadIndex();
    if (index >= *water_count) return;
    Vec3 position = positions[index];
    Vec3 velocity = velocities[index];
    Vec3 local = inverse_rotate(coupling.heater.orientation,
        subtract(position, coupling.heater.center));
    bool heated = abs(local.x) <= coupling.heater.half_extents.x &&
        abs(local.z) <= coupling.heater.half_extents.y &&
        local.y >= -water_radius * 0.5f &&
        local.y <= water_radius * 2.5f;
    float temperature = temperatures[index];
    if (heated && coupling.heater_temperature > temperature)
        temperature += (coupling.heater_temperature - temperature) *
            (1.0f - exp(-coupling.heat_transfer_rate * dt));
    temperatures[index] = temperature;
    keep[index] = 1u;
    if (temperature < coupling.boiling_temperature) return;
    uint ordinal = pmAtomicAdd(converted, 1u);
    if (ordinal >= smoke.capacity) return;
    uint slot = (first_smoke_slot + ordinal) % smoke.capacity;
    smoke_positions[slot] = position;
    smoke_velocities[slot] = add(velocity, multiply(
        normalized_or(multiply(gravity, -1.0f), {0.0f, 1.0f, 0.0f}),
        coupling.steam_rise_speed));
    smoke_ages[slot] = 0.0f;
    smoke_thermal_lift[slot] = coupling.steam_rise_speed;
    smoke_densities[slot] = 0.0f;
    smoke_pressures[slot] = 0.0f;
    smoke_vorticities[slot] = {};
    keep[index] = 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_clear_inactive_keep(
    uniform uint8_t* keep, uniform uint* count, uniform uint capacity)
{
    uint index = pmThreadIndex();
    if (index >= capacity) return;
    if (index >= *count) keep[index] = 0u;
}
#endif
