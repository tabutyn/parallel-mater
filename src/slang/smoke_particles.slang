// SPDX-License-Identifier: MIT
// Production particle-mode smoke kernels shared with the Metal backend.

import smoke_common;
import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokePositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeVelocities : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float>
    smokeAges : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<float>
    smokeDensities : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<float>
    smokePressures : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeVorticities : register(u5);
[[vk::binding(6, 0)]] StructuredBuffer<MetalSmokeMetadata>
    smokeMetadata : register(t6);
[[vk::binding(7, 0)]] ConstantBuffer<MetalSmokeConstants>
    smokeConstants : register(b7);
[[vk::binding(8, 0)]] RWStructuredBuffer<float>
    smokeThermalLift : register(u8);
[[vk::binding(9, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeGridVelocity : register(u9);
[[vk::binding(10, 0)]] RWStructuredBuffer<float>
    smokeGridPressure : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<float>
    smokeGridDensity : register(u11);
[[vk::binding(12, 0)]] RWStructuredBuffer<float>
    smokeGridTemperature : register(u12);
[[vk::binding(13, 0)]] RWStructuredBuffer<uint>
    smokeGridSolid : register(u13);
[[vk::binding(14, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeGridVorticity : register(u14);
[[vk::binding(15, 0)]] RWStructuredBuffer<float>
    smokeGridDivergence : register(u15);
[[vk::binding(16, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeParticleScratch : register(u16);
[[vk::binding(18, 0)]] RWStructuredBuffer<float>
    smokeGridPressureResidual : register(u18);
[[vk::binding(27, 0)]] RWStructuredBuffer<uint>
    smokeGridDeformableSolid : register(u27);
[[vk::binding(28, 0)]] RWStructuredBuffer<float>
    smokeGridFaceVelocity : register(u28);
[[vk::binding(30, 0)]] RWStructuredBuffer<float>
    smokeGridFaceBoundary : register(u30);

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_clear(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalSmokeConstants constants = smokeConstants;
    if (constants.grid_resolution == 0u) return;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    uint cells = n * height * n;
    uint faces = metalSmokeFaceCount(0u, n, height) +
        metalSmokeFaceCount(1u, n, height) +
        metalSmokeFaceCount(2u, n, height);
    if (index < cells)
    {
        smokeGridVelocity[index] = metalStore(float3(0.0f));
        smokeGridDensity[index] = 0.0f;
        smokeGridTemperature[index] = 0.0f;
        smokeGridSolid[index] = 0u;
        smokeGridVorticity[index] = metalStore(float3(0.0f));
        smokeGridDivergence[index] = 0.0f;
        smokeGridDeformableSolid[index] = 0u;
    }
    if (index < faces)
    {
        smokeGridFaceBoundary[index] = 1.0f;
        smokeGridFaceBoundary[faces + index] = 0.0f;
    }
    if (index == 0u) smokeGridPressureResidual[0] = 0.0f;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_particle_density(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalSmokeConstants constants = smokeConstants;
    uint count = smokeMetadata[0].count;
    if (index >= count || constants.command != 2u ||
        constants.grid_resolution != 0u) return;
    if (smokeAges[index] >= constants.lifetime)
    {
        smokeDensities[index] = 0.0f;
        smokePressures[index] = 0.0f;
        return;
    }
    float support = 3.0f * constants.particle_radius;
    float supportSquared = support * support;
    float3 position = metalLoad(smokePositions[index]);
    float density = 0.0f;
    for (uint other = 0u; other < count; ++other)
    {
        if (smokeAges[other] >= constants.lifetime) continue;
        float3 delta = position - metalLoad(smokePositions[other]);
        float squared = dot(delta, delta);
        if (squared >= supportSquared) continue;
        float q = 1.0f - sqrt(max(squared, 0.0f)) / support;
        density += q * q * q;
    }
    smokeDensities[index] = density;
    float crowding = constants.pressure_stiffness * max(
        density / constants.rest_number_density - 1.0f, 0.0f);
    smokePressures[index] = min(100.0f,
        crowding + smokePressures[index] *
            exp(-constants.timestep / 0.05f));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_particle_vorticity(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalSmokeConstants constants = smokeConstants;
    uint count = smokeMetadata[0].count;
    if (index >= count || constants.command != 2u ||
        constants.grid_resolution != 0u) return;
    float3 curl = float3(0.0f);
    if (smokeAges[index] < constants.lifetime &&
        constants.vorticity_confinement > 0.0f)
    {
        float support = 3.0f * constants.particle_radius;
        float supportSquared = support * support;
        float3 position = metalLoad(smokePositions[index]);
        float3 velocity = metalLoad(smokeVelocities[index]);
        for (uint other = 0u; other < count; ++other)
        {
            if (other == index || smokeAges[other] >= constants.lifetime)
                continue;
            float3 delta = position - metalLoad(smokePositions[other]);
            float squared = dot(delta, delta);
            if (squared >= supportSquared || squared < 1.0e-12f) continue;
            float distanceValue = sqrt(squared);
            float q = 1.0f - distanceValue / support;
            float pairDensity = max(1.0f,
                sqrt(smokeDensities[index] * smokeDensities[other]));
            float3 gradient = delta * (-3.0f * q * q /
                (support * distanceValue * pairDensity));
            curl += cross(
                metalLoad(smokeVelocities[other]) - velocity, gradient);
        }
    }
    smokeVorticities[index] = metalStore(curl);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_particle_forces(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalSmokeConstants constants = smokeConstants;
    uint count = smokeMetadata[0].count;
    if (index >= count || constants.command != 2u ||
        constants.grid_resolution != 0u) return;
    float3 acceleration = float3(0.0f);
    float3 confinementGradient = float3(0.0f);
    if (smokeAges[index] < constants.lifetime)
    {
        float support = 3.0f * constants.particle_radius;
        float supportSquared = support * support;
        float3 position = metalLoad(smokePositions[index]);
        float3 velocity = metalLoad(smokeVelocities[index]);
        float vorticity = length(metalLoad(smokeVorticities[index]));
        for (uint other = 0u; other < count; ++other)
        {
            if (other == index || smokeAges[other] >= constants.lifetime)
                continue;
            float3 delta = position - metalLoad(smokePositions[other]);
            float squared = dot(delta, delta);
            if (squared >= supportSquared || squared < 1.0e-12f) continue;
            float distanceValue = sqrt(squared);
            float q = 1.0f - distanceValue / support;
            float pairDensity = max(1.0f,
                sqrt(smokeDensities[index] * smokeDensities[other]));
            float pressure = 0.5f *
                (smokePressures[index] + smokePressures[other]);
            acceleration += delta * (3.0f * pressure * q * q /
                (support * distanceValue * pairDensity));
            acceleration +=
                (metalLoad(smokeVelocities[other]) - velocity) *
                (constants.viscosity * q * q / pairDensity);
            if (constants.vorticity_confinement > 0.0f)
            {
                float3 gradient = delta * (-3.0f * q * q /
                    (support * distanceValue * pairDensity));
                confinementGradient += gradient *
                    (length(metalLoad(smokeVorticities[other])) - vorticity);
            }
        }
        if (dot(confinementGradient, confinementGradient) > 1.0e-10f)
            acceleration += cross(normalize(confinementGradient),
                metalLoad(smokeVorticities[index])) *
                (constants.vorticity_confinement * support);
    }
    smokeParticleScratch[index] = metalStore(
        metalLimit(acceleration, 50.0f));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_particle_integrate(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalSmokeConstants constants = smokeConstants;
    uint count = smokeMetadata[0].count;
    if (index >= count || constants.command != 2u ||
        constants.grid_resolution != 0u) return;
    if (smokeAges[index] >= constants.lifetime)
    {
        smokeDensities[index] = 0.0f;
        smokePressures[index] = 0.0f;
        return;
    }
    float3 gravity = metalLoad(constants.gravity);
    float3 up = dot(gravity, gravity) > 1.0e-12f
        ? -normalize(gravity) : float3(0.0f, 1.0f, 0.0f);
    float3 oldPosition = metalLoad(smokePositions[index]);
    float3 velocity = metalLoad(smokeVelocities[index]) +
        metalLoad(smokeParticleScratch[index]) * constants.timestep;
    velocity += up *
        ((constants.buoyancy + smokeThermalLift[index]) * constants.timestep);
    float response = 1.0f - exp(-constants.response * constants.timestep);
    velocity += (metalLoad(constants.wind) - velocity) * response;
    velocity = metalLimit(velocity, constants.maximum_speed);
    smokeVelocities[index] = metalStore(velocity);
    smokeParticleScratch[index] = metalStore(oldPosition);
    smokePositions[index] = metalStore(
        oldPosition + velocity * constants.timestep);
    smokeAges[index] += constants.timestep;
    smokeThermalLift[index] *= exp(-constants.timestep / 3.0f);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_advect_particles(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalSmokeConstants constants = smokeConstants;
    if (particle >= smokeMetadata[0].count || constants.command != 2u ||
        constants.grid_resolution == 0u || constants.grid_spacing <= 0.0f ||
        smokeAges[particle] >= constants.lifetime)
        return;
    uint resolution = constants.grid_resolution;
    uint vertical = constants.grid_vertical_resolution;
    float spacing = constants.grid_spacing;
    float3 minimum = metalLoad(constants.grid_minimum);
    float3 point = metalLoad(smokePositions[particle]);
    smokeParticleScratch[particle] = metalStore(point);
    float3 velocity = metalLoad(smokeVelocities[particle]);
    float3 local = (point - minimum) / spacing;
    bool inside = local.x >= 0.0f && local.y >= 0.0f && local.z >= 0.0f &&
        local.x < float(resolution) && local.y < float(vertical) &&
        local.z < float(resolution);
    if (inside && smokeAges[particle] > 0.0f)
    {
        float3 first = metalSmokeSampleFaceVelocity(
            smokeGridFaceVelocity, 0u, point, minimum, spacing,
            resolution, vertical);
        velocity = metalSmokeSampleFaceVelocity(
            smokeGridFaceVelocity, 0u,
            point + first * (0.5f * constants.timestep),
            minimum, spacing, resolution, vertical);
        smokeDensities[particle] = metalSmokeSampleCellScalar(
            smokeGridDensity, point, minimum, spacing, resolution, vertical) *
            constants.rest_number_density;
        smokePressures[particle] = metalSmokeSampleCellScalar(
            smokeGridPressure, point, minimum, spacing, resolution, vertical);
        smokeVorticities[particle] = metalStore(metalSmokeSampleCellVector(
            smokeGridVorticity, point, minimum,
            spacing, resolution, vertical));
    }
    velocity = metalLimit(velocity, constants.maximum_speed);
    smokeVelocities[particle] = metalStore(velocity);
    smokePositions[particle] = metalStore(
        point + velocity * constants.timestep);
    smokeAges[particle] += constants.timestep;
    smokeThermalLift[particle] *= exp(-constants.timestep / 3.0f);
}
