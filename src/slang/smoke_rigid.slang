// SPDX-License-Identifier: MIT
// Metal compatibility adapter for smoke/rigid coupling.

import smoke_common;
import metal_common;

struct MetalSmokeRigidConstants
{
    MetalHandle body;
    float particle_radius;
    float lifetime;
    float air_density;
    float drag_coefficient;
    float contact_distance;
    uint tracer_contact;
    uint enabled;
    uint rigid_count;
    uint grid_resolution;
    uint grid_vertical_resolution;
    float grid_spacing;
    MetalPackedVec3 grid_minimum;
    float density_scale;
    float kinematic_viscosity;
    float les_coefficient;
    float timestep;
    float maximum_speed;
    float pressure_stiffness;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeRigidPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeRigidVelocities : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<float>
    smokeRigidAges : register(t2);
[[vk::binding(3, 0)]] StructuredBuffer<MetalSmokeMetadata>
    smokeRigidMetadata : register(t3);
[[vk::binding(4, 0)]] StructuredBuffer<MetalHandle>
    smokeRigidIds : register(t4);
[[vk::binding(5, 0)]] RWStructuredBuffer<MetalRigidState>
    smokeRigidStates : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<MetalRigidParameters>
    smokeRigidParameters : register(u6);
[[vk::binding(7, 0)]] StructuredBuffer<MetalPackedVec3>
    smokeRigidVertices : register(t7);
[[vk::binding(8, 0)]] StructuredBuffer<uint>
    smokeRigidIndices : register(t8);
[[vk::binding(9, 0)]] StructuredBuffer<MetalTriangleMeshInfo>
    smokeRigidMeshes : register(t9);
[[vk::binding(10, 0)]] ConstantBuffer<MetalSmokeRigidConstants>
    smokeRigidConstants : register(b10);
[[vk::binding(11, 0)]] RWStructuredBuffer<float>
    smokeRigidGridPressure : register(u11);
[[vk::binding(12, 0)]] RWStructuredBuffer<float>
    smokeRigidGridDensity : register(u12);
[[vk::binding(13, 0)]] RWStructuredBuffer<float>
    smokeRigidGridFaceVelocity : register(u13);
[[vk::binding(14, 0)]] StructuredBuffer<MetalPackedVec3>
    smokeRigidPreviousPositions : register(t14);
[[vk::binding(15, 0)]] StructuredBuffer<MetalRigidState>
    smokeRigidOldStates : register(t15);
[[vk::binding(16, 0)]] RWStructuredBuffer<float>
    smokeRigidParticlePressures : register(u16);
[[vk::binding(17, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeRigidParticleLinearImpulses : register(u17);
[[vk::binding(18, 0)]] RWStructuredBuffer<MetalPackedVec3>
    smokeRigidParticleAngularImpulses : register(u18);

[shader("compute")]
[numthreads(128, 1, 1)]
public void pm_smoke_rigid(uint groupIndex : SV_GroupIndex)
{
    MetalSmokeRigidConstants constants = smokeRigidConstants;
    if (constants.enabled == 0u) return;
    uint bodyIndex = constants.rigid_count;
    for (uint index = 0u; index < constants.rigid_count; ++index)
    {
        MetalHandle id = smokeRigidIds[index];
        if (id.index == constants.body.index &&
            id.generation == constants.body.generation)
        {
            bodyIndex = index;
            break;
        }
    }
    if (bodyIndex == constants.rigid_count) return;
    MetalRigidParameters bodyParameters = smokeRigidParameters[bodyIndex];
    MetalTriangleMeshInfo mesh =
        smokeRigidMeshes[bodyParameters.mesh_index];

    if (groupIndex == 0u && constants.grid_resolution != 0u &&
        bodyParameters.motion == 2u)
    {
        MetalRigidState bodyState = smokeRigidStates[bodyIndex];
        uint resolution = constants.grid_resolution;
        uint vertical = constants.grid_vertical_resolution;
        float spacing = constants.grid_spacing;
        float3 minimum = metalLoad(constants.grid_minimum);
        float3 maximum = minimum + spacing *
            float3(float(resolution), float(vertical), float(resolution));
        float3 linearForce = float3(0.0f);
        float3 angularForce = float3(0.0f);
        for (uint local = 0u; local < mesh.index_count; local += 3u)
        {
            uint first = smokeRigidIndices[mesh.index_offset + local];
            uint second = smokeRigidIndices[mesh.index_offset + local + 1u];
            uint third = smokeRigidIndices[mesh.index_offset + local + 2u];
            float3 a = metalWorldPoint(bodyState, metalLoad(
                smokeRigidVertices[mesh.vertex_offset + first]));
            float3 b = metalWorldPoint(bodyState, metalLoad(
                smokeRigidVertices[mesh.vertex_offset + second]));
            float3 c = metalWorldPoint(bodyState, metalLoad(
                smokeRigidVertices[mesh.vertex_offset + third]));
            float3 twiceArea = cross(b - a, c - a);
            float area = 0.5f * length(twiceArea);
            if (area <= 1.0e-9f) continue;
            float3 normal = normalize(twiceArea);
            float3 center = (a + b + c) / 3.0f;
            float3 plus = center + normal * (1.5f * spacing);
            float3 minus = center - normal * (1.5f * spacing);
            if (any(plus < minimum) || any(plus >= maximum) ||
                any(minus < minimum) || any(minus >= maximum))
                continue;
            float plusDensity = constants.density_scale *
                metalSmokeSampleCellScalar(smokeRigidGridDensity,
                    plus, minimum, spacing, resolution, vertical);
            float minusDensity = constants.density_scale *
                metalSmokeSampleCellScalar(smokeRigidGridDensity,
                    minus, minimum, spacing, resolution, vertical);
            if (plusDensity + minusDensity < 1.0e-4f) continue;
            float3 plusAir = metalSmokeSampleFaceVelocity(
                smokeRigidGridFaceVelocity, 0u, plus, minimum, spacing,
                resolution, vertical);
            float3 minusAir = metalSmokeSampleFaceVelocity(
                smokeRigidGridFaceVelocity, 0u, minus, minimum, spacing,
                resolution, vertical);
            float3 arm = center - metalLoad(bodyState.position);
            float3 wall = metalLoad(bodyState.linear_velocity) +
                cross(metalLoad(bodyState.angular_velocity), arm);
            float3 plusRelative = plusAir - wall;
            float3 minusRelative = minusAir - wall;
            float3 plusTangent =
                plusRelative - normal * dot(plusRelative, normal);
            float3 minusTangent =
                minusRelative - normal * dot(minusRelative, normal);
            float pressureDifference =
                minusDensity * metalSmokeSampleCellScalar(
                    smokeRigidGridPressure, minus, minimum, spacing,
                    resolution, vertical) -
                plusDensity * metalSmokeSampleCellScalar(
                    smokeRigidGridPressure, plus, minimum, spacing,
                    resolution, vertical);
            float strain = metalSmokeSampleFaceStrain(
                smokeRigidGridFaceVelocity, 0u, center, minimum, spacing,
                resolution, vertical);
            float viscosity = constants.kinematic_viscosity +
                constants.les_coefficient * constants.les_coefficient *
                    spacing * spacing * strain;
            float3 force = (normal * pressureDifference +
                (plusTangent * plusDensity + minusTangent * minusDensity) *
                    (constants.drag_coefficient * viscosity /
                        (1.5f * spacing))) *
                (constants.air_density * area);
            linearForce += force;
            angularForce += cross(arm, force);
        }
        bodyState.linear_velocity = metalStore(metalLimit(
            metalLoad(bodyState.linear_velocity) + linearForce *
                (constants.timestep * bodyParameters.inverse_mass),
            bodyParameters.maximum_linear_speed));
        bodyState.angular_velocity = metalStore(metalLimit(
            metalLoad(bodyState.angular_velocity) + metalInverseInertiaMul(
                bodyParameters, bodyState, angularForce * constants.timestep),
            bodyParameters.maximum_angular_speed));
        smokeRigidStates[bodyIndex] = bodyState;
    }

    DeviceMemoryBarrierWithGroupSync();

    if (constants.tracer_contact == 0u) return;
    MetalRigidState contactState = smokeRigidStates[bodyIndex];
    MetalRigidState oldContactState = smokeRigidOldStates[bodyIndex];
    float3 initialBodyLinear = metalLoad(contactState.linear_velocity);
    float3 initialBodyAngular = metalLoad(contactState.angular_velocity);
    float clearance = constants.contact_distance;
    float boundary = 3.0f * clearance;
    float particleMass = constants.grid_resolution == 0u
        ? constants.air_density *
            pow(2.0f * constants.particle_radius, 3.0f)
        : 0.0f;
    MetalQuaternion inverse =
        metalQuaternionConjugate(contactState.orientation);
    MetalQuaternion oldInverse =
        metalQuaternionConjugate(oldContactState.orientation);
    MetalSmokeMetadata metadata = smokeRigidMetadata[0];
    for (uint particle = groupIndex; particle < metadata.count;
         particle += 128u)
    {
        smokeRigidParticleLinearImpulses[particle] =
            metalStore(float3(0.0f));
        smokeRigidParticleAngularImpulses[particle] =
            metalStore(float3(0.0f));
        if (smokeRigidAges[particle] >= constants.lifetime) continue;
        float3 worldPosition = metalLoad(smokeRigidPositions[particle]);
        float3 before = metalRotate(oldInverse,
            metalLoad(smokeRigidPreviousPositions[particle]) -
                metalLoad(oldContactState.position));
        float3 point = metalRotate(inverse,
            worldPosition - metalLoad(contactState.position));
        float3 lower = min(before, point) - boundary;
        float3 upper = max(before, point) + boundary;
        if (any(upper < metalLoad(mesh.minimum)) ||
            any(lower > metalLoad(mesh.maximum)))
            continue;
        float nearestSquared = boundary * boundary;
        float earliest = 2.0f;
        bool found = false;
        bool swept = false;
        float3 nearest = float3(0.0f);
        float3 faceNormal = float3(1.0f, 0.0f, 0.0f);
        float3 path = point - before;
        for (uint local = 0u; local < mesh.index_count; local += 3u)
        {
            uint first = smokeRigidIndices[mesh.index_offset + local];
            uint second = smokeRigidIndices[mesh.index_offset + local + 1u];
            uint third = smokeRigidIndices[mesh.index_offset + local + 2u];
            float3 a = metalLoad(
                smokeRigidVertices[mesh.vertex_offset + first]);
            float3 b = metalLoad(
                smokeRigidVertices[mesh.vertex_offset + second]);
            float3 c = metalLoad(
                smokeRigidVertices[mesh.vertex_offset + third]);
            if (any(upper < min(a, min(b, c))) ||
                any(lower > max(a, max(b, c))))
                continue;
            float3 rawNormal = cross(b - a, c - a);
            if (dot(rawNormal, rawNormal) < 1.0e-12f) continue;
            float sideBefore = dot(before - a, rawNormal);
            float sideAfter = dot(point - a, rawNormal);
            if (sideBefore * sideAfter < 0.0f)
            {
                float fraction = sideBefore / (sideBefore - sideAfter);
                if (fraction < earliest)
                {
                    float3 hit = before + path * fraction;
                    float3 onFace =
                        metalClosestPointTriangle(hit, a, b, c);
                    if (dot(hit - onFace, hit - onFace) < 1.0e-8f)
                    {
                        earliest = fraction;
                        nearest = onFace;
                        faceNormal = normalize(rawNormal);
                        swept = true;
                        found = true;
                    }
                }
            }
            if (swept) continue;
            float3 candidate =
                metalClosestPointTriangle(point, a, b, c);
            float squared = dot(point - candidate, point - candidate);
            if (squared >= nearestSquared) continue;
            nearestSquared = squared;
            nearest = candidate;
            faceNormal = normalize(rawNormal);
            found = true;
        }
        if (!found) continue;
        float side = dot(before - nearest, faceNormal);
        if (abs(side) < 1.0e-5f) side = dot(point - nearest, faceNormal);
        float3 localNormal = side >= 0.0f ? faceNormal : -faceNormal;
        float3 normal = metalRotate(contactState.orientation, localNormal);
        float3 arm = metalRotate(contactState.orientation, nearest);
        float3 surfacePoint = metalLoad(contactState.position) + arm;
        bool touching = swept || nearestSquared < clearance * clearance;
        float3 tracePosition = surfacePoint + normal *
            (constants.grid_resolution != 0u
                ? 1.5f * constants.grid_spacing : clearance);
        if (touching)
            smokeRigidPositions[particle] = metalStore(tracePosition);
        float3 surfaceVelocity =
            initialBodyLinear + cross(initialBodyAngular, arm);
        float3 oldVelocity = metalLoad(smokeRigidVelocities[particle]);
        float3 tracerVelocity = oldVelocity;
        if (constants.grid_resolution != 0u && touching)
        {
            float3 minimum = metalLoad(constants.grid_minimum);
            float3 local =
                (tracePosition - minimum) / constants.grid_spacing;
            if (all(local >= 0.0f) &&
                local.x < float(constants.grid_resolution) &&
                local.y < float(constants.grid_vertical_resolution) &&
                local.z < float(constants.grid_resolution))
                tracerVelocity = metalSmokeSampleFaceVelocity(
                    smokeRigidGridFaceVelocity, 0u, tracePosition, minimum,
                    constants.grid_spacing, constants.grid_resolution,
                    constants.grid_vertical_resolution);
        }
        float3 relative = tracerVelocity - surfaceVelocity;
        float normalInflow = max(0.0f, -dot(relative, normal));
        if (touching)
            relative -= normal * min(0.0f, dot(relative, normal));
        if (constants.grid_resolution != 0u && touching)
        {
            float sourceSpeed = max(
                length(tracerVelocity - surfaceVelocity),
                length(oldVelocity - surfaceVelocity));
            float tangentSpeed = length(relative);
            if (tangentSpeed > 1.0e-6f && tangentSpeed < sourceSpeed)
                relative *= sourceSpeed / tangentSpeed;
        }
        float contactDistance =
            touching ? clearance : sqrt(nearestSquared);
        float q = max(0.0f,
            1.0f - contactDistance / max(boundary, 1.0e-12f));
        if (constants.grid_resolution == 0u)
            relative *= 1.0f - (1.0f - exp(
                -constants.drag_coefficient * q * q * constants.timestep));
        if (touching && constants.grid_resolution == 0u)
            smokeRigidParticlePressures[particle] = max(
                smokeRigidParticlePressures[particle],
                10.0f * constants.pressure_stiffness *
                    normalInflow * normalInflow);
        float3 newVelocity = metalLimit(
            surfaceVelocity + relative, constants.maximum_speed);
        smokeRigidVelocities[particle] = metalStore(newVelocity);
        float3 reaction = (oldVelocity - newVelocity) * particleMass;
        smokeRigidParticleLinearImpulses[particle] = metalStore(reaction);
        smokeRigidParticleAngularImpulses[particle] =
            metalStore(cross(arm, reaction));
    }

    DeviceMemoryBarrierWithGroupSync();
    if (groupIndex == 0u && bodyParameters.motion == 2u &&
        particleMass > 0.0f)
    {
        float3 accumulatedLinearImpulse = float3(0.0f);
        float3 accumulatedAngularImpulse = float3(0.0f);
        for (uint particle = 0u; particle < metadata.count; ++particle)
        {
            accumulatedLinearImpulse +=
                metalLoad(smokeRigidParticleLinearImpulses[particle]);
            accumulatedAngularImpulse +=
                metalLoad(smokeRigidParticleAngularImpulses[particle]);
        }
        MetalRigidState bodyState = smokeRigidStates[bodyIndex];
        bodyState.linear_velocity = metalStore(metalLimit(
            metalLoad(bodyState.linear_velocity) +
                accumulatedLinearImpulse * bodyParameters.inverse_mass,
            bodyParameters.maximum_linear_speed));
        bodyState.angular_velocity = metalStore(metalLimit(
            metalLoad(bodyState.angular_velocity) + metalInverseInertiaMul(
                bodyParameters, bodyState, accumulatedAngularImpulse),
            bodyParameters.maximum_angular_speed));
        smokeRigidStates[bodyIndex] = bodyState;
    }
}
