// SPDX-License-Identifier: MIT
// Metal compatibility adapter for fluid/cloth/soft particle-rigid contact.

import metal_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    particleRigidPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    particleRigidVelocities : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<float>
    particleRigidInverseMasses : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<MetalRigidState>
    particleRigidStates : register(u3);
[[vk::binding(4, 0)]] StructuredBuffer<MetalRigidParameters>
    particleRigidParameters : register(t4);
[[vk::binding(5, 0)]] StructuredBuffer<MetalPackedVec3>
    particleRigidVertices : register(t5);
[[vk::binding(6, 0)]] StructuredBuffer<uint>
    particleRigidIndices : register(t6);
[[vk::binding(7, 0)]] StructuredBuffer<MetalTriangleMeshInfo>
    particleRigidMeshes : register(t7);
[[vk::binding(8, 0)]] ConstantBuffer<MetalParticleRigidConstants>
    particleRigidConstants : register(b8);
[[vk::binding(9, 0)]] RWStructuredBuffer<MetalPackedVec3>
    particleRigidDiagnostics : register(u9);
[[vk::binding(10, 0)]] StructuredBuffer<uint>
    particleRigidStableIds : register(t10);
[[vk::binding(11, 0)]] StructuredBuffer<MetalHandle>
    particleRigidIds : register(t11);
[[vk::binding(12, 0)]] RWStructuredBuffer<MetalContactEvent>
    particleRigidContactSamples : register(u12);
[[vk::binding(13, 0)]] RWStructuredBuffer<uint>
    particleRigidContactFlags : register(u13);
[[vk::binding(14, 0)]] StructuredBuffer<MetalParticleMetadata>
    particleRigidMetadata : register(t14);
[[vk::binding(15, 0)]] StructuredBuffer<MetalPackedVec3>
    particleRigidPrevious : register(t15);
[[vk::binding(16, 0)]] StructuredBuffer<MetalRigidState>
    particleRigidOldStates : register(t16);
[[vk::binding(17, 0)]] RWStructuredBuffer<float>
    particleRigidFoam : register(u17);
[[vk::binding(18, 0)]] RWStructuredBuffer<MetalParticleRigidContact>
    particleRigidContacts : register(u18);
[[vk::binding(19, 0)]] RWStructuredBuffer<uint>
    particleRigidBodyContactCounts : register(u19);
[[vk::binding(20, 0)]] RWStructuredBuffer<MetalPackedVec3>
    particleRigidBodyLinearImpulses : register(u20);
[[vk::binding(21, 0)]] RWStructuredBuffer<MetalPackedVec3>
    particleRigidBodyAngularImpulses : register(u21);
[[vk::binding(22, 0)]] StructuredBuffer<MetalCollisionPlane>
    particleRigidSolidPlanes : register(t22);
[[vk::binding(23, 0)]] RWStructuredBuffer<MetalPackedVec3>
    particleRigidBodyPositionCorrections : register(u23);
[[vk::binding(24, 0)]] RWStructuredBuffer<MetalPackedVec3>
    particleRigidParticleLinearImpulses : register(u24);
[[vk::binding(25, 0)]] RWStructuredBuffer<MetalPackedVec3>
    particleRigidParticleAngularImpulses : register(u25);
[[vk::binding(26, 0)]] RWStructuredBuffer<MetalPackedVec3>
    particleRigidParticlePositionCorrections : register(u26);
[[vk::binding(27, 0)]] RWStructuredBuffer<MetalSoftContactAccumulator>
    particleRigidAccumulators : register(u27);
// First word of MetalSoftContactState; this kernel only sets the flag.
[[vk::binding(28, 0)]] RWStructuredBuffer<uint>
    particleRigidContactStateWords : register(u28);
[[vk::binding(29, 0)]] StructuredBuffer<uint>
    particleRigidSpawnBaseline : register(t29);

groupshared float3 particleRigidReactionLinear[128];
groupshared float3 particleRigidReactionAngular[128];
groupshared float3 particleRigidReactionPosition[128];

[ForceInline]
static float3 particleRigidNormalizedOr(float3 value, float3 fallback)
{
    float squared = dot(value, value);
    return squared > 1.0e-12f ? value * rsqrt(squared) : fallback;
}

static bool particleRigidSolidInteriorContact(
    float3 position,
    float contactDistance,
    MetalTriangleMeshInfo mesh,
    out float3 normal,
    out float penetration,
    out float3 contact)
{
    if (mesh.solid_plane_count == 0u)
    {
        normal = float3(0.0f);
        penetration = 0.0f;
        contact = float3(0.0f);
        return false;
    }
    float nearestSide = -3.402823466e38f;
    float3 nearestNormal = float3(0.0f);
    for (uint local = 0u; local < mesh.solid_plane_count; ++local)
    {
        MetalCollisionPlane plane =
            particleRigidSolidPlanes[mesh.solid_plane_offset + local];
        float side = dot(metalLoad(plane.normal), position) - plane.offset;
        if (side > 0.0f)
        {
            normal = float3(0.0f);
            penetration = 0.0f;
            contact = float3(0.0f);
            return false;
        }
        if (side > nearestSide)
        {
            nearestSide = side;
            nearestNormal = metalLoad(plane.normal);
        }
    }
    normal = nearestNormal;
    penetration = contactDistance - nearestSide;
    contact = position - nearestNormal * nearestSide;
    return true;
}

static MetalParticleRigidContact particleRigidDetect(
    uint particle,
    bool fluidParticles,
    bool clothParticles,
    bool softParticles)
{
    MetalParticleRigidConstants constants = particleRigidConstants;
    MetalParticleRigidContact best;
    best.normal = metalStore(float3(0.0f));
    best.point = metalStore(float3(0.0f));
    best.penetration = 0.0f;
    best.body = 0xffffffffu;
    float particleInverse = fluidParticles
        ? constants.particle_inverse_mass
        : particleRigidInverseMasses[particle];
    if (particleInverse <= 0.0f) return best;
    float3 particlePosition = metalLoad(particleRigidPositions[particle]);
    float3 previousPosition = metalLoad(particleRigidPrevious[particle]);
    for (uint body = 0u; body < constants.rigid_count; ++body)
    {
        MetalRigidState bodyState = particleRigidStates[body];
        MetalRigidParameters bodyParameters =
            particleRigidParameters[body];
        if (bodyParameters.motion == 0u && fluidParticles) continue;
        MetalTriangleMeshInfo mesh =
            particleRigidMeshes[bodyParameters.mesh_index];
        MetalRigidState oldState = constants.first_iteration != 0u
            ? particleRigidOldStates[body] : bodyState;
        MetalQuaternion inverse =
            metalQuaternionConjugate(bodyState.orientation);
        MetalQuaternion oldInverse =
            metalQuaternionConjugate(oldState.orientation);
        float3 position = metalRotate(inverse,
            particlePosition - metalLoad(bodyState.position));
        float3 origin = metalRotate(oldInverse,
            previousPosition - metalLoad(oldState.position));
        float contactDistance = constants.radius +
            (fluidParticles ? 0.0f : bodyParameters.collision_margin);
        float3 lower = min(origin, position) - contactDistance;
        float3 upper = max(origin, position) + contactDistance;
        if (any(upper < metalLoad(mesh.minimum)) ||
            any(lower > metalLoad(mesh.maximum)))
            continue;
        if (constants.solid_contacts != 0u)
        {
            float3 normal;
            float penetration;
            float3 contact;
            if (particleRigidSolidInteriorContact(position, contactDistance,
                mesh, normal, penetration, contact))
            {
                if (best.body == 0xffffffffu ||
                    penetration > best.penetration)
                {
                    best.normal = metalStore(
                        metalRotate(bodyState.orientation, normal));
                    best.point = metalStore(
                        metalWorldPoint(bodyState, contact));
                    best.penetration = penetration;
                    best.body = body;
                }
                continue;
            }
        }
        for (uint local = 0u; local < mesh.index_count; local += 3u)
        {
            uint first = particleRigidIndices[mesh.index_offset + local];
            uint second =
                particleRigidIndices[mesh.index_offset + local + 1u];
            uint third =
                particleRigidIndices[mesh.index_offset + local + 2u];
            float3 a = metalLoad(
                particleRigidVertices[mesh.vertex_offset + first]);
            float3 b = metalLoad(
                particleRigidVertices[mesh.vertex_offset + second]);
            float3 c = metalLoad(
                particleRigidVertices[mesh.vertex_offset + third]);
            if (any(upper < min(a, min(b, c))) ||
                any(lower > max(a, max(b, c))))
                continue;
            float3 faceValue = cross(b - a, c - a);
            if (dot(faceValue, faceValue) <= 1.0e-14f) continue;
            float3 face = particleRigidNormalizedOr(
                faceValue, float3(0.0f, 1.0f, 0.0f));
            float3 candidate =
                metalClosestPointTriangle(position, a, b, c);
            float3 delta = position - candidate;
            float distanceValue = length(delta);
            float threshold = fluidParticles || clothParticles
                ? 1.0e-6f : 1.0e-7f;
            float3 normal = distanceValue > threshold
                ? delta / distanceValue
                : face * (dot(origin - a, face) >= 0.0f ? 1.0f : -1.0f);
            float penetration = contactDistance - distanceValue;
            float3 contact = candidate;
            float before = dot(origin - a, face);
            float after = dot(position - a, face);
            if (before * after < 0.0f &&
                (!softParticles || mesh.solid_plane_count == 0u ||
                 before > 0.0f))
            {
                float fraction = before / (before - after);
                float3 crossing = origin + (position - origin) * fraction;
                float3 nearest =
                    metalClosestPointTriangle(crossing, a, b, c);
                float crossingRadius =
                    clothParticles ? constants.radius : contactDistance;
                float crossingSquared =
                    dot(crossing - nearest, crossing - nearest);
                bool crossed = clothParticles
                    ? crossingSquared < crossingRadius * crossingRadius
                    : crossingSquared <= crossingRadius * crossingRadius;
                if (crossed)
                {
                    normal = face * (clothParticles
                        ? (before > 0.0f ? 1.0f : -1.0f)
                        : (before >= 0.0f ? 1.0f : -1.0f));
                    penetration = max(
                        penetration, contactDistance + abs(after));
                }
            }
            if (penetration <= 0.0f ||
                (best.body != 0xffffffffu &&
                 penetration <= best.penetration))
                continue;
            best.normal = metalStore(
                metalRotate(bodyState.orientation, normal));
            best.point = metalStore(metalWorldPoint(bodyState, contact));
            best.penetration = penetration;
            best.body = body;
        }
    }
    return best;
}

static void particleRigidResolveMoving(
    uint particle,
    bool fluidParticles,
    bool clothParticles,
    bool softParticles)
{
    MetalParticleRigidConstants constants = particleRigidConstants;
    particleRigidParticleLinearImpulses[particle] =
        metalStore(float3(0.0f));
    particleRigidParticleAngularImpulses[particle] =
        metalStore(float3(0.0f));
    particleRigidParticlePositionCorrections[particle] =
        metalStore(float3(0.0f));
    MetalSoftContactAccumulator accumulator =
        particleRigidAccumulators[particle];
    if (softParticles)
        accumulator.arm = metalStore(float3(0.0f));
    MetalParticleRigidContact contact = particleRigidContacts[particle];
    if (contact.body == 0xffffffffu)
    {
        if (softParticles) particleRigidAccumulators[particle] = accumulator;
        return;
    }
    uint body = contact.body;
    MetalRigidState bodyState = particleRigidStates[body];
    MetalRigidParameters bodyParameters = particleRigidParameters[body];
    float particleInverse = fluidParticles
        ? constants.particle_inverse_mass
        : particleRigidInverseMasses[particle];
    float particleMass = 1.0f / max(particleInverse, 1.0e-12f);
    float3 normal = metalLoad(contact.normal);
    float3 point = metalLoad(contact.point);
    float positionDenominator =
        particleInverse + bodyParameters.inverse_mass;
    float particlePositionShare =
        constants.share_position != 0u && positionDenominator > 1.0e-12f
        ? particleInverse / positionDenominator : 1.0f;
    particleRigidPositions[particle] = metalStore(
        metalLoad(particleRigidPositions[particle]) +
        normal * (contact.penetration * particlePositionShare));
    if (constants.share_position != 0u && bodyParameters.motion == 2u &&
        positionDenominator > 1.0e-12f)
        particleRigidParticlePositionCorrections[particle] = metalStore(
            -normal * (contact.penetration * bodyParameters.inverse_mass /
                positionDenominator));
    float3 particleVelocity = softParticles
        ? (metalLoad(particleRigidPositions[particle]) -
           normal * (contact.penetration * particlePositionShare) -
           metalLoad(particleRigidPrevious[particle])) /
            max(constants.timestep, 1.0e-12f)
        : metalLoad(particleRigidVelocities[particle]);
    float3 initialVelocity = particleVelocity;
    float3 arm = point - metalLoad(bodyState.position);
    if (softParticles)
    {
        accumulator.normal_delta += contact.penetration;
        accumulator.arm = metalStore(arm);
        particleRigidAccumulators[particle] = accumulator;
        if (bodyParameters.motion == 2u)
        {
            uint ignored;
            InterlockedExchange(
                particleRigidContactStateWords[0], 1u, ignored);
        }
    }
    float3 bodyVelocity = metalLoad(bodyState.linear_velocity) +
        cross(metalLoad(bodyState.angular_velocity), arm);
    float3 relative = particleVelocity - bodyVelocity;
    float incoming = dot(relative, normal);
    float normalImpulse = 0.0f;
    float3 impulse = float3(0.0f);
    if (clothParticles && incoming < 0.0f)
    {
        float3 normalCross = cross(arm, normal);
        float denominator = particleInverse + bodyParameters.inverse_mass +
            dot(cross(metalInverseInertiaMul(
                bodyParameters, bodyState, normalCross), arm), normal);
        if (denominator > 1.0e-12f)
        {
            normalImpulse = -(1.0f + bodyParameters.restitution) *
                incoming / denominator;
            impulse = normal * normalImpulse;
            particleVelocity += impulse * particleInverse;
            particleRigidParticleLinearImpulses[particle] =
                metalStore(-impulse);
            particleRigidParticleAngularImpulses[particle] =
                metalStore(-cross(arm, impulse));
        }
    }
    else if (softParticles && incoming < 0.0f)
    {
        float3 normalCross = cross(arm, normal);
        float denominator = particleInverse + bodyParameters.inverse_mass +
            dot(cross(metalInverseInertiaMul(
                bodyParameters, bodyState, normalCross), arm), normal);
        if (denominator > 1.0e-12f)
        {
            normalImpulse = -(1.0f + bodyParameters.restitution) *
                incoming / denominator;
            impulse = normal * normalImpulse;
            float3 velocityDelta = impulse * particleInverse;
            particleVelocity += velocityDelta;
            accumulator = particleRigidAccumulators[particle];
            accumulator.momentum_delta = metalStore(
                metalLoad(accumulator.momentum_delta) + impulse);
            particleRigidAccumulators[particle] = accumulator;
            if (bodyParameters.motion == 2u)
                particleRigidPositions[particle] = metalStore(
                    metalLoad(particleRigidPositions[particle]) +
                    velocityDelta * constants.timestep);
            particleRigidParticleLinearImpulses[particle] =
                metalStore(-impulse);
            particleRigidParticleAngularImpulses[particle] =
                metalStore(-cross(arm, impulse));
        }
    }
    else if (fluidParticles)
    {
        float recoverySpeed = min(constants.maximum_reaction_speed,
            min(0.5f * constants.radius, 0.2f * contact.penetration) /
                max(constants.timestep, 1.0e-12f));
        if (incoming < recoverySpeed)
        {
            float batchSize = 1.0f +
                float(max(particleRigidBodyContactCounts[body], 1u) - 1u) *
                min(1.0f,
                    4.0f * particleMass * bodyParameters.inverse_mass);
            float3 normalCross = cross(arm, normal);
            float bodyNormalInverse = bodyParameters.inverse_mass +
                dot(cross(metalInverseInertiaMul(
                    bodyParameters, bodyState, normalCross), arm), normal);
            float denominator =
                particleInverse + batchSize * bodyNormalInverse;
            if (denominator > 1.0e-12f)
            {
                normalImpulse = (recoverySpeed - incoming) / denominator;
                impulse = normal * normalImpulse;
                particleVelocity += impulse * particleInverse;
                float3 tangentVelocity = relative - normal * incoming;
                float tangentSpeed = length(tangentVelocity);
                if (tangentSpeed > 1.0e-8f)
                {
                    float3 tangent = tangentVelocity / tangentSpeed;
                    float3 tangentCross = cross(arm, tangent);
                    float tangentDenominator = particleInverse + batchSize *
                        (bodyParameters.inverse_mass +
                         dot(cross(metalInverseInertiaMul(bodyParameters,
                             bodyState, tangentCross), arm), tangent));
                    if (tangentDenominator > 1.0e-12f)
                    {
                        float friction = bodyParameters.friction;
                        float tangentImpulse = min(
                            tangentSpeed / tangentDenominator,
                            max(friction, 0.0f) * normalImpulse);
                        float3 frictionImpulse = -tangent * tangentImpulse;
                        impulse += frictionImpulse;
                        particleVelocity +=
                            frictionImpulse * particleInverse;
                    }
                }
                particleRigidParticleLinearImpulses[particle] =
                    metalStore(-impulse);
                particleRigidParticleAngularImpulses[particle] =
                    metalStore(-cross(arm, impulse));
                particleRigidFoam[particle] = max(
                    particleRigidFoam[particle], min(1.0f, -incoming * 0.35f));
            }
        }
    }
    particleVelocity =
        metalLimit(particleVelocity, constants.maximum_reaction_speed);
    particleRigidVelocities[particle] = metalStore(particleVelocity);
    float inverseDt = 1.0f / max(constants.timestep, 1.0e-12f);
    float3 correctionVelocity = normal *
        (contact.penetration * particlePositionShare * inverseDt);
    float3 acceleration =
        (particleVelocity - initialVelocity + correctionVelocity) * inverseDt;
    particleRigidDiagnostics[particle] = clothParticles || softParticles
        ? metalStore(impulse * inverseDt)
        : metalStore(metalLoad(particleRigidDiagnostics[particle]) +
            (constants.diagnostic_is_acceleration != 0u
                ? acceleration : acceleration / particleInverse));
    if (constants.collect_contacts != 0u)
    {
        if (particleRigidContactFlags[particle] != 2u ||
            normalImpulse >
                particleRigidContactSamples[particle].normal_impulse)
        {
            MetalContactEvent sample;
            sample.fluid = constants.fluid;
            sample.stable_particle_id = particleRigidStableIds[particle];
            sample.rigid_body = particleRigidIds[body];
            sample.position = contact.point;
            sample.normal = contact.normal;
            sample.normal_impulse = normalImpulse;
            particleRigidContactSamples[particle] = sample;
        }
        particleRigidContactFlags[particle] = 2u;
    }
}

static void particleRigidResolveStatic(uint particle, bool fluidParticles)
{
    MetalParticleRigidConstants constants = particleRigidConstants;
    float particleInverse = fluidParticles
        ? constants.particle_inverse_mass
        : particleRigidInverseMasses[particle];
    if (particleInverse <= 0.0f) return;
    for (uint body = 0u; body < constants.rigid_count; ++body)
    {
        MetalRigidState bodyState = particleRigidStates[body];
        MetalRigidParameters bodyParameters = particleRigidParameters[body];
        if (bodyParameters.motion != 0u) continue;
        MetalTriangleMeshInfo mesh =
            particleRigidMeshes[bodyParameters.mesh_index];
        MetalQuaternion inverse =
            metalQuaternionConjugate(bodyState.orientation);
        float contactDistance = constants.radius +
            (fluidParticles ? 0.0f : bodyParameters.collision_margin);
        bool newlySpawned = fluidParticles &&
            constants.recover_spawn != 0u &&
            particle >= particleRigidSpawnBaseline[0];
        float queryDistance = newlySpawned
            ? max(contactDistance, constants.spawn_clearance)
            : contactDistance;
        float3 worldPosition = metalLoad(particleRigidPositions[particle]);
        float3 position = metalRotate(
            inverse, worldPosition - metalLoad(bodyState.position));
        float3 origin = metalRotate(inverse,
            metalLoad(particleRigidPrevious[particle]) -
                metalLoad(bodyState.position));
        float3 lower = min(origin, position) - queryDistance;
        float3 upper = max(origin, position) + queryDistance;
        if (any(upper < metalLoad(mesh.minimum)) ||
            any(lower > metalLoad(mesh.maximum)))
            continue;
        float bestPenetration = 0.0f;
        float3 bestNormal = float3(0.0f);
        float3 bestContact = float3(0.0f);
        bool interior = constants.solid_contacts != 0u &&
            particleRigidSolidInteriorContact(position, contactDistance,
                mesh, bestNormal, bestPenetration, bestContact);
        for (uint local = 0u;
            !interior && local < mesh.index_count; local += 3u)
        {
            uint first = particleRigidIndices[mesh.index_offset + local];
            uint second =
                particleRigidIndices[mesh.index_offset + local + 1u];
            uint third =
                particleRigidIndices[mesh.index_offset + local + 2u];
            float3 a = metalLoad(
                particleRigidVertices[mesh.vertex_offset + first]);
            float3 b = metalLoad(
                particleRigidVertices[mesh.vertex_offset + second]);
            float3 c = metalLoad(
                particleRigidVertices[mesh.vertex_offset + third]);
            if (any(upper < min(a, min(b, c))) ||
                any(lower > max(a, max(b, c))))
                continue;
            float3 faceValue = cross(b - a, c - a);
            if (dot(faceValue, faceValue) <= 1.0e-14f) continue;
            float3 face = normalize(faceValue);
            float3 candidate =
                metalClosestPointTriangle(position, a, b, c);
            float3 delta = position - candidate;
            float distanceValue = length(delta);
            float3 normal = distanceValue >
                    (fluidParticles ? 1.0e-6f : 1.0e-7f)
                ? delta / distanceValue
                : face * (dot(origin - a, face) >= 0.0f ? 1.0f : -1.0f);
            float penetration = contactDistance - distanceValue;
            float3 contact = candidate;
            float before = dot(origin - a, face);
            float after = dot(position - a, face);
            if (before * after < 0.0f)
            {
                float fraction = before / (before - after);
                float3 crossing = origin + (position - origin) * fraction;
                float3 nearest =
                    metalClosestPointTriangle(crossing, a, b, c);
                if (dot(crossing - nearest, crossing - nearest) <=
                    contactDistance * contactDistance)
                {
                    normal = face * (before >= 0.0f ? 1.0f : -1.0f);
                    penetration = max(
                        penetration, contactDistance + abs(after));
                    contact = nearest;
                }
            }
            if (newlySpawned)
            {
                float3 localUp =
                    metalRotate(inverse, metalLoad(constants.up));
                if (abs(dot(face, localUp)) > 0.7f)
                {
                    float3 floorNormal =
                        dot(face, localUp) > 0.0f ? face : -face;
                    float side = dot(position - a, floorNormal);
                    if (side < constants.radius &&
                        side > -constants.spawn_clearance &&
                        distanceValue < constants.spawn_clearance)
                    {
                        normal = floorNormal;
                        penetration = max(
                            penetration, constants.radius - side);
                    }
                }
            }
            if (penetration <= bestPenetration) continue;
            bestPenetration = penetration;
            bestNormal = normal;
            bestContact = contact;
        }
        if (bestPenetration <= 0.0f) continue;
        float3 worldNormal =
            metalRotate(bodyState.orientation, bestNormal);
        float3 worldContact = metalWorldPoint(bodyState, bestContact);
        MetalParticleRigidContact contact;
        contact.normal = metalStore(worldNormal);
        contact.point = metalStore(worldContact);
        contact.penetration = bestPenetration;
        contact.body = body;
        particleRigidContacts[particle] = contact;
        uint oldCount;
        InterlockedAdd(particleRigidBodyContactCounts[body], 1u, oldCount);
        particleRigidPositions[particle] =
            metalStore(worldPosition + worldNormal * bestPenetration);
        float3 particleVelocity =
            metalLoad(particleRigidVelocities[particle]);
        float3 initialVelocity = particleVelocity;
        float incoming = dot(particleVelocity, worldNormal);
        float normalImpulse = 0.0f;
        if (incoming < 0.0f)
        {
            float restitution = fluidParticles
                ? bodyParameters.restitution : constants.restitution;
            normalImpulse =
                -incoming * (1.0f + restitution) / particleInverse;
            particleVelocity -=
                worldNormal * incoming * (1.0f + restitution);
            float3 tangentVelocity = particleVelocity -
                worldNormal * dot(particleVelocity, worldNormal);
            float tangentSpeed = length(tangentVelocity);
            if (tangentSpeed > 1.0e-8f)
            {
                float friction = fluidParticles
                    ? bodyParameters.friction : constants.friction;
                float frictionSpeed = min(tangentSpeed,
                    max(friction, 0.0f) * normalImpulse * particleInverse);
                particleVelocity -=
                    tangentVelocity * (frictionSpeed / tangentSpeed);
            }
            if (fluidParticles)
                particleRigidFoam[particle] = max(
                    particleRigidFoam[particle],
                    min(1.0f, -incoming * 0.35f));
        }
        particleVelocity =
            metalLimit(particleVelocity, constants.maximum_reaction_speed);
        particleRigidVelocities[particle] = metalStore(particleVelocity);
        float inverseDt = 1.0f / max(constants.timestep, 1.0e-12f);
        float3 correctionVelocity =
            worldNormal * (bestPenetration * inverseDt);
        float3 acceleration =
            (particleVelocity - initialVelocity + correctionVelocity) *
                inverseDt;
        particleRigidDiagnostics[particle] = metalStore(
            metalLoad(particleRigidDiagnostics[particle]) +
            (constants.diagnostic_is_acceleration != 0u
                ? acceleration : acceleration / particleInverse));
        if (constants.collect_contacts != 0u &&
            (particleRigidContactFlags[particle] == 0u ||
             (particleRigidContactFlags[particle] == 1u &&
              normalImpulse >
                particleRigidContactSamples[particle].normal_impulse)))
        {
            MetalContactEvent sample;
            sample.fluid = constants.fluid;
            sample.stable_particle_id = particleRigidStableIds[particle];
            sample.rigid_body = particleRigidIds[body];
            sample.position = metalStore(worldContact);
            sample.normal = metalStore(worldNormal);
            sample.normal_impulse = normalImpulse;
            particleRigidContactSamples[particle] = sample;
            particleRigidContactFlags[particle] = 1u;
        }
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
public void pm_particles_rigid(uint groupIndex : SV_GroupIndex)
{
    uint lane = groupIndex;
    MetalParticleRigidConstants constants = particleRigidConstants;
    bool fluidParticles = constants.fluid.generation != 0u;
    bool clothParticles =
        !fluidParticles && constants.solid_contacts == 0u;
    bool softParticles =
        !fluidParticles && constants.solid_contacts != 0u;
    uint particleCount = fluidParticles
        ? particleRigidMetadata[0].count : constants.particle_count;
    for (uint body = lane; body < constants.rigid_count; body += 128u)
    {
        particleRigidBodyContactCounts[body] = 0u;
        particleRigidBodyLinearImpulses[body] = metalStore(float3(0.0f));
        particleRigidBodyAngularImpulses[body] = metalStore(float3(0.0f));
        particleRigidBodyPositionCorrections[body] =
            metalStore(float3(0.0f));
    }
    DeviceMemoryBarrierWithGroupSync();

    for (uint particle = lane; particle < particleCount; particle += 128u)
    {
        MetalParticleRigidContact contact = particleRigidDetect(
            particle, fluidParticles, clothParticles, softParticles);
        particleRigidContacts[particle] = contact;
        if (contact.body != 0xffffffffu)
        {
            uint oldCount;
            InterlockedAdd(particleRigidBodyContactCounts[contact.body],
                1u, oldCount);
        }
    }
    DeviceMemoryBarrierWithGroupSync();

    for (uint particle = lane; particle < particleCount; particle += 128u)
        particleRigidResolveMoving(
            particle, fluidParticles, clothParticles, softParticles);
    DeviceMemoryBarrierWithGroupSync();

    for (uint body = 0u; body < constants.rigid_count; ++body)
    {
        float3 linearImpulse = float3(0.0f);
        float3 angularImpulse = float3(0.0f);
        float3 positionCorrection = float3(0.0f);
        for (uint particle = lane; particle < particleCount;
             particle += 128u)
        {
            if (particleRigidContacts[particle].body != body) continue;
            linearImpulse += metalLoad(
                particleRigidParticleLinearImpulses[particle]);
            angularImpulse += metalLoad(
                particleRigidParticleAngularImpulses[particle]);
            positionCorrection += metalLoad(
                particleRigidParticlePositionCorrections[particle]);
        }
        particleRigidReactionLinear[lane] = linearImpulse;
        particleRigidReactionAngular[lane] = angularImpulse;
        particleRigidReactionPosition[lane] = positionCorrection;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 64u; stride != 0u; stride >>= 1u)
        {
            if (lane < stride)
            {
                particleRigidReactionLinear[lane] +=
                    particleRigidReactionLinear[lane + stride];
                particleRigidReactionAngular[lane] +=
                    particleRigidReactionAngular[lane + stride];
                particleRigidReactionPosition[lane] +=
                    particleRigidReactionPosition[lane + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0u)
        {
            particleRigidBodyLinearImpulses[body] =
                metalStore(particleRigidReactionLinear[0]);
            particleRigidBodyAngularImpulses[body] =
                metalStore(particleRigidReactionAngular[0]);
            particleRigidBodyPositionCorrections[body] =
                metalStore(particleRigidReactionPosition[0]);
            MetalRigidParameters parameters = particleRigidParameters[body];
            if (parameters.motion == 2u)
            {
                MetalRigidState state = particleRigidStates[body];
                state.position = metalStore(metalLoad(state.position) +
                    particleRigidReactionPosition[0]);
                state.linear_velocity = metalStore(metalLimit(
                    metalLoad(state.linear_velocity) +
                        particleRigidReactionLinear[0] *
                            parameters.inverse_mass,
                    parameters.maximum_linear_speed));
                state.angular_velocity = metalStore(metalLimit(
                    metalLoad(state.angular_velocity) +
                        metalInverseInertiaMul(parameters, state,
                            particleRigidReactionAngular[0]),
                    parameters.maximum_angular_speed));
                particleRigidStates[body] = state;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    DeviceMemoryBarrierWithGroupSync();

    if (clothParticles || softParticles) return;
    for (uint particle = lane; particle < particleCount; particle += 128u)
        particleRigidResolveStatic(particle, fluidParticles);
}
