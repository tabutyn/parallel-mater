// SPDX-License-Identifier: MIT
// Shared fluid/soft-body coupling for the CUDA mirror and Metal backend.

#ifdef PM_TARGET_METAL
import metal_common;

struct MetalFluidSoftConstants
{
    float timestep;
    uint node_count;
    uint surface_count;
    uint surface_index_count;
    float contact_distance;
    float friction;
    float particle_mass;
    float maximum_particle_speed;
    float maximum_soft_speed;
    float orientation;
    uint enabled;
    float maximum_projection;
    float frame_inverse_timestep;
    uint rigid_count;
    float rigid_recovery_radius;
};

struct MetalFluidSoftNearest
{
    float3 point;
    float3 normal;
    float3 weights;
    float penetration;
    uint triangle;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftFluidPositions : register(u0);
[[vk::binding(1, 0)]] StructuredBuffer<MetalPackedVec3>
    metalFluidSoftFluidPrevious : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftFluidVelocities : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftFluidAccelerations : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<float>
    metalFluidSoftFluidFoam : register(u4);
[[vk::binding(5, 0)]] StructuredBuffer<MetalParticleMetadata>
    metalFluidSoftFluidMetadata : register(t5);
[[vk::binding(6, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftNodePositions : register(u6);
[[vk::binding(7, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftNodeVelocities : register(u7);
[[vk::binding(8, 0)]] StructuredBuffer<float>
    metalFluidSoftNodeInverseMasses : register(t8);
[[vk::binding(9, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftSurfacePositions : register(u9);
[[vk::binding(10, 0)]] StructuredBuffer<uint>
    metalFluidSoftSurfaceIndices : register(t10);
[[vk::binding(11, 0)]] StructuredBuffer<MetalSurfaceBinding>
    metalFluidSoftSurfaceBindings : register(t11);
[[vk::binding(12, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftNodeForces : register(u12);
[[vk::binding(13, 0)]] ConstantBuffer<MetalFluidSoftConstants>
    metalFluidSoftConstants : register(b13);
[[vk::binding(14, 0)]] StructuredBuffer<MetalPackedVec3>
    metalFluidSoftNodeRestPositions : register(t14);
[[vk::binding(15, 0)]] StructuredBuffer<MetalPackedVec3>
    metalFluidSoftSurfaceRestPositions : register(t15);
[[vk::binding(16, 0)]] RWStructuredBuffer<uint>
    metalFluidSoftDiagnosticContactCount : register(u16);
[[vk::binding(17, 0)]] RWStructuredBuffer<uint>
    metalFluidSoftDiagnosticMaximumPenetration : register(u17);
[[vk::binding(18, 0)]] RWStructuredBuffer<uint>
    metalFluidSoftContributionNodes : register(u18);
[[vk::binding(19, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftContributionPositions : register(u19);
[[vk::binding(20, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftContributionChanges : register(u20);
[[vk::binding(21, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftContributionForces : register(u21);
[[vk::binding(22, 0)]] RWStructuredBuffer<uint>
    metalFluidSoftNodeContactCounts : register(u22);
[[vk::binding(23, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalFluidSoftPreviousSurface : register(u23);
[[vk::binding(24, 0)]] StructuredBuffer<MetalPackedVec3>
    metalFluidSoftNodePrevious : register(t24);
[[vk::binding(25, 0)]] StructuredBuffer<MetalRigidState>
    metalFluidSoftRigidStates : register(t25);
[[vk::binding(26, 0)]] StructuredBuffer<MetalRigidParameters>
    metalFluidSoftRigidParameters : register(t26);
[[vk::binding(27, 0)]] StructuredBuffer<MetalPackedVec3>
    metalFluidSoftMeshVertices : register(t27);
[[vk::binding(28, 0)]] StructuredBuffer<uint>
    metalFluidSoftMeshIndices : register(t28);
[[vk::binding(29, 0)]] StructuredBuffer<MetalTriangleMeshInfo>
    metalFluidSoftMeshes : register(t29);
[[vk::binding(30, 0)]] StructuredBuffer<MetalCollisionPlane>
    metalFluidSoftSolidPlanes : register(t30);

[ForceInline]
static bool metalFluidSoftInside(float3 point)
{
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    float3 direction = float3(1.0f, 0.371f, 0.173f);
    int winding = 0;
    bool ambiguous = false;
    for (uint base = 0u; base + 2u < constants.surface_index_count;
         base += 3u)
    {
        float3 a = metalLoad(metalFluidSoftSurfacePositions[
            metalFluidSoftSurfaceIndices[base]]);
        float3 edge1 = metalLoad(metalFluidSoftSurfacePositions[
            metalFluidSoftSurfaceIndices[base + 1u]]) - a;
        float3 edge2 = metalLoad(metalFluidSoftSurfacePositions[
            metalFluidSoftSurfaceIndices[base + 2u]]) - a;
        float3 h = cross(direction, edge2);
        float3 s = point - a;
        float determinant = dot(edge1, h);
        if (abs(determinant) < 1.0e-12f) continue;
        float u = dot(s, h) / determinant;
        float3 q = cross(s, edge1);
        float v = dot(direction, q) / determinant;
        float t = dot(edge2, q) / determinant;
        if (t < 0.0f || u < -1.0e-5f || v < -1.0e-5f ||
            u + v > 1.00001f)
            continue;
        if (u < 1.0e-5f || v < 1.0e-5f ||
            u + v > 0.99999f || t < 1.0e-7f)
            ambiguous = true;
        winding += determinant < 0.0f ? 1 : -1;
    }
    if (!ambiguous) return winding != 0;
    float angle = 0.0f;
    for (uint base = 0u; base + 2u < constants.surface_index_count;
         base += 3u)
    {
        float3 a = metalLoad(metalFluidSoftSurfacePositions[
            metalFluidSoftSurfaceIndices[base]]) - point;
        float3 b = metalLoad(metalFluidSoftSurfacePositions[
            metalFluidSoftSurfaceIndices[base + 1u]]) - point;
        float3 c = metalLoad(metalFluidSoftSurfacePositions[
            metalFluidSoftSurfaceIndices[base + 2u]]) - point;
        float la = length(a);
        float lb = length(b);
        float lc = length(c);
        angle += 2.0f * atan2(dot(a, cross(b, c)),
            la * lb * lc + dot(a, b) * lc + dot(b, c) * la +
                dot(c, a) * lb);
    }
    return abs(angle) > 6.2831853f;
}

[ForceInline]
static MetalFluidSoftNearest metalFluidSoftNearest(
    float3 point, float3 start, bool sweep)
{
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    MetalFluidSoftNearest result;
    result.point = float3(0.0f);
    result.normal = float3(0.0f);
    result.weights = float3(0.0f);
    result.penetration = 0.0f;
    result.triangle = 0xffffffffu;
    float best = 3.402823466e38f;
    float earliest = 2.0f;
    for (uint base = 0u; base + 2u < constants.surface_index_count;
         base += 3u)
    {
        uint ia = metalFluidSoftSurfaceIndices[base];
        uint ib = metalFluidSoftSurfaceIndices[base + 1u];
        uint ic = metalFluidSoftSurfaceIndices[base + 2u];
        float3 a = metalLoad(metalFluidSoftSurfacePositions[ia]);
        float3 b = metalLoad(metalFluidSoftSurfacePositions[ib]);
        float3 c = metalLoad(metalFluidSoftSurfacePositions[ic]);
        float3 face = cross(b - a, c - a);
        float faceSquared = dot(face, face);
        if (faceSquared < 1.0e-16f) continue;
        float3 normal = face * rsqrt(faceSquared) * constants.orientation;
        float3 nearPoint = metalClosestPointTriangle(point, a, b, c);
        float3 weights = metalTriangleWeights(nearPoint, a, b, c);
        float squared = dot(point - nearPoint, point - nearPoint);
        if (earliest > 1.0f && squared < best)
        {
            best = squared;
            result.point = nearPoint;
            result.normal = normal;
            result.weights = weights;
            result.penetration = constants.contact_distance -
                dot(point - nearPoint, normal);
            result.triangle = base;
        }
        if (!sweep) continue;
        float3 oldNear =
            metalLoad(metalFluidSoftPreviousSurface[ia]) * weights.x +
            metalLoad(metalFluidSoftPreviousSurface[ib]) * weights.y +
            metalLoad(metalFluidSoftPreviousSurface[ic]) * weights.z;
        float3 transportedStart = start + (nearPoint - oldNear);
        float first = dot(transportedStart - a, normal);
        float last = dot(point - a, normal);
        if (first < constants.contact_distance ||
            last >= constants.contact_distance || first - last < 1.0e-8f)
            continue;
        float fraction = (first - constants.contact_distance) /
            (first - last);
        if (fraction >= earliest) continue;
        float3 crossing = transportedStart +
            (point - transportedStart) * fraction -
            normal * constants.contact_distance;
        float3 hit = metalClosestPointTriangle(crossing, a, b, c);
        if (dot(crossing - hit, crossing - hit) > 1.0e-8f) continue;
        earliest = fraction;
        result.point = hit;
        result.normal = normal;
        result.weights = metalTriangleWeights(hit, a, b, c);
        result.penetration = constants.contact_distance - last;
        result.triangle = base;
    }
    if (result.triangle == 0xffffffffu) return result;
    if (earliest > 1.0f)
    {
        float radius = sqrt(best);
        bool faceInterior = result.weights.x > 1.0e-4f &&
            result.weights.y > 1.0e-4f &&
            result.weights.z > 1.0e-4f;
        bool inside = faceInterior
            ? dot(point - result.point, result.normal) < 0.0f
            : metalFluidSoftInside(point);
        if (!inside)
        {
            result.penetration = constants.contact_distance - radius;
            if (radius > 1.0e-7f)
                result.normal = (point - result.point) / radius;
        }
        else
        {
            result.penetration = constants.contact_distance + radius;
            if (radius > 1.0e-7f)
                result.normal = (result.point - point) / radius;
        }
    }
    if (result.penetration <= 0.0f) result.triangle = 0xffffffffu;
    return result;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_soft_contact(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    if (particle >= metalFluidSoftFluidMetadata[0].count ||
        constants.enabled == 0u)
        return;
    uint contribution = particle * 12u;
    for (uint item = 0u; item < 12u; ++item)
    {
        metalFluidSoftContributionNodes[contribution + item] = 0xffffffffu;
        metalFluidSoftContributionPositions[contribution + item] =
            metalStore(float3(0.0f));
        metalFluidSoftContributionChanges[contribution + item] =
            metalStore(float3(0.0f));
        metalFluidSoftContributionForces[contribution + item] =
            metalStore(float3(0.0f));
    }
    bool sweep = metalFluidSoftNodeContactCounts[constants.node_count] != 0u;
    MetalFluidSoftNearest hit = metalFluidSoftNearest(
        metalLoad(metalFluidSoftFluidPositions[particle]),
        metalLoad(metalFluidSoftFluidPrevious[particle]), sweep);
    if (hit.triangle == 0xffffffffu) return;

    uint contactNodes[12];
    float contactWeights[12];
    for (uint item = 0u; item < 12u; ++item)
    {
        contactNodes[item] = 0xffffffffu;
        contactWeights[item] = 0.0f;
    }
    uint nodeCount = 0u;
    float cornerWeights[3];
    cornerWeights[0] = hit.weights.x;
    cornerWeights[1] = hit.weights.y;
    cornerWeights[2] = hit.weights.z;
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        MetalSurfaceBinding binding = metalFluidSoftSurfaceBindings[
            metalFluidSoftSurfaceIndices[hit.triangle + corner]];
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            float weight = cornerWeights[corner] * binding.weights[slot];
            if (weight <= 0.0f) continue;
            uint node = binding.nodes[slot];
            uint item = 0u;
            while (item < nodeCount && contactNodes[item] != node) ++item;
            if (item == nodeCount && nodeCount < 12u)
            {
                contactNodes[nodeCount] = node;
                contactWeights[nodeCount] = 0.0f;
                ++nodeCount;
            }
            if (item < nodeCount) contactWeights[item] += weight;
        }
    }
    if (nodeCount == 0u) return;
    float3 surfaceVelocity = float3(0.0f);
    for (uint item = 0u; item < nodeCount; ++item)
    {
        uint node = contactNodes[item];
        float weight = contactWeights[item];
        surfaceVelocity += metalLoad(metalFluidSoftNodeVelocities[node]) *
            weight;
        InterlockedAdd(metalFluidSoftNodeContactCounts[node], 1u);
        metalFluidSoftContributionNodes[contribution + item] = node;
        metalFluidSoftContributionPositions[contribution + item] =
            metalStore(float3(weight, item == 0u ? hit.penetration : 0.0f,
                0.0f));
    }
    metalFluidSoftContributionChanges[contribution] = metalStore(
        metalLoad(metalFluidSoftFluidVelocities[particle]) - surfaceVelocity);
    metalFluidSoftContributionForces[contribution] = metalStore(hit.normal);
    InterlockedAdd(metalFluidSoftDiagnosticContactCount[0], 1u);
    InterlockedMax(metalFluidSoftDiagnosticMaximumPenetration[0],
        asuint(hit.penetration));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_soft_solve(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    if (particle >= metalFluidSoftFluidMetadata[0].count ||
        constants.enabled == 0u)
        return;
    uint contribution = particle * 12u;
    if (metalFluidSoftContributionNodes[contribution] == 0xffffffffu) return;
    float particleInverse = 1.0f / max(constants.particle_mass, 1.0e-12f);
    float3 normal =
        metalLoad(metalFluidSoftContributionForces[contribution]);
    float3 relative =
        metalLoad(metalFluidSoftContributionChanges[contribution]);
    float penetration =
        metalLoad(metalFluidSoftContributionPositions[contribution]).y;
    float denominator = particleInverse;
    for (uint item = 0u; item < 12u; ++item)
    {
        uint index = contribution + item;
        uint node = metalFluidSoftContributionNodes[index];
        if (node == 0xffffffffu) break;
        float weight = metalLoad(
            metalFluidSoftContributionPositions[index]).x;
        uint contactCount = max(metalFluidSoftNodeContactCounts[node], 1u);
        denominator += weight * metalFluidSoftNodeInverseMasses[node] *
            float(contactCount);
    }
    if (denominator <= 1.0e-12f) return;
    float normalSpeed = dot(relative, normal);
    float normalImpulse = max(0.0f, -normalSpeed) / denominator;
    float3 tangent = relative - normal * normalSpeed;
    float3 impulse = normal * normalImpulse - metalLimit(
        tangent / denominator, constants.friction * normalImpulse);
    float scale = 1.0f;
    for (uint item = 0u; item < 12u; ++item)
    {
        uint index = contribution + item;
        uint node = metalFluidSoftContributionNodes[index];
        if (node == 0xffffffffu) break;
        float weight = metalLoad(
            metalFluidSoftContributionPositions[index]).x;
        float inverse = metalFluidSoftNodeInverseMasses[node];
        float count = float(max(metalFluidSoftNodeContactCounts[node], 1u));
        float3 delta = -impulse * (weight * inverse * count);
        float squared = dot(delta, delta);
        if (squared <= 1.0e-20f) continue;
        float3 velocity = metalLoad(metalFluidSoftNodeVelocities[node]);
        float projection = dot(velocity, delta);
        float excess = min(0.0f, dot(velocity, velocity) -
            constants.maximum_soft_speed * constants.maximum_soft_speed);
        float limit = (-projection + sqrt(max(0.0f,
            projection * projection - squared * excess))) / squared;
        scale = min(scale, max(0.0f, limit));
    }
    impulse *= scale;
    float lambda = min(penetration, constants.maximum_projection) /
        denominator;
    metalFluidSoftFluidPositions[particle] = metalStore(
        metalLoad(metalFluidSoftFluidPositions[particle]) +
        normal * (particleInverse * lambda));
    float3 velocityChange = impulse * particleInverse;
    metalFluidSoftFluidVelocities[particle] = metalStore(
        metalLoad(metalFluidSoftFluidVelocities[particle]) + velocityChange);
    metalFluidSoftFluidAccelerations[particle] = metalStore(
        metalLoad(metalFluidSoftFluidAccelerations[particle]) +
        velocityChange / max(constants.timestep, 1.0e-12f));
    metalFluidSoftFluidFoam[particle] = max(
        metalFluidSoftFluidFoam[particle],
        min(1.0f, normalImpulse * particleInverse * 0.1f));
    for (uint item = 0u; item < 12u; ++item)
    {
        uint index = contribution + item;
        uint node = metalFluidSoftContributionNodes[index];
        if (node == 0xffffffffu) break;
        float weight = metalLoad(
            metalFluidSoftContributionPositions[index]).x;
        float inverse = metalFluidSoftNodeInverseMasses[node];
        metalFluidSoftContributionPositions[index] = metalStore(
            -normal * (weight * inverse * lambda));
        metalFluidSoftContributionChanges[index] = metalStore(
            -impulse * (weight * inverse));
        metalFluidSoftContributionForces[index] = metalStore(
            -impulse * (weight * constants.frame_inverse_timestep));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_soft_apply(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    if (node >= constants.node_count || constants.enabled == 0u) return;
    float3 positionChange = float3(0.0f);
    float3 velocityChange = float3(0.0f);
    float3 force = float3(0.0f);
    for (uint particle = 0u;
         particle < metalFluidSoftFluidMetadata[0].count; ++particle)
    {
        uint contribution = particle * 12u;
        for (uint item = 0u; item < 12u; ++item)
        {
            uint index = contribution + item;
            if (metalFluidSoftContributionNodes[index] != node) continue;
            positionChange += metalLoad(
                metalFluidSoftContributionPositions[index]);
            velocityChange += metalLoad(
                metalFluidSoftContributionChanges[index]);
            force += metalLoad(metalFluidSoftContributionForces[index]);
        }
    }
    metalFluidSoftNodePositions[node] = metalStore(
        metalLoad(metalFluidSoftNodePositions[node]) + positionChange);
    metalFluidSoftNodeVelocities[node] = metalStore(metalLimit(
        metalLoad(metalFluidSoftNodeVelocities[node]) + velocityChange,
        constants.maximum_soft_speed));
    metalFluidSoftNodeForces[node] = metalStore(
        metalLoad(metalFluidSoftNodeForces[node]) + force);
    metalFluidSoftNodeContactCounts[node] = 0u;
    if (node == 0u)
        metalFluidSoftNodeContactCounts[constants.node_count] = 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_soft_surface_update(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint surface = dispatchThreadId.x;
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    if (surface >= constants.surface_count || constants.enabled == 0u) return;
    MetalSurfaceBinding binding = metalFluidSoftSurfaceBindings[surface];
    float3 point =
        metalLoad(metalFluidSoftSurfaceRestPositions[surface]);
    for (uint slot = 0u; slot < 4u; ++slot)
    {
        uint node = binding.nodes[slot];
        point += (metalLoad(metalFluidSoftNodePositions[node]) -
            metalLoad(metalFluidSoftNodeRestPositions[node])) *
            binding.weights[slot];
    }
    metalFluidSoftSurfacePositions[surface] = metalStore(point);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_soft_sweep_on(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x == 0u)
        metalFluidSoftNodeContactCounts[
            metalFluidSoftConstants.node_count] = 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_soft_recover(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    if (particle >= metalFluidSoftFluidMetadata[0].count ||
        constants.enabled == 0u)
        return;
    MetalFluidSoftNearest hit = metalFluidSoftNearest(
        metalLoad(metalFluidSoftFluidPositions[particle]),
        metalLoad(metalFluidSoftFluidPrevious[particle]), true);
    if (hit.triangle != 0xffffffffu)
        metalFluidSoftFluidPositions[particle] = metalStore(
            metalLoad(metalFluidSoftFluidPositions[particle]) +
            hit.normal * hit.penetration);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_soft_recover_current(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint particle = dispatchThreadId.x;
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    if (particle >= metalFluidSoftFluidMetadata[0].count ||
        constants.enabled == 0u)
        return;
    float3 point = metalLoad(metalFluidSoftFluidPositions[particle]);
    MetalFluidSoftNearest hit = metalFluidSoftNearest(point, point, false);
    if (hit.triangle != 0xffffffffu)
        metalFluidSoftFluidPositions[particle] = metalStore(
            point + hit.normal * hit.penetration);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_soft_copy_surface(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint surface = dispatchThreadId.x;
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    if (surface >= constants.surface_count || constants.enabled == 0u) return;
    metalFluidSoftPreviousSurface[surface] =
        metalFluidSoftSurfacePositions[surface];
}

[ForceInline]
static bool metalFluidSoftSolidInteriorContact(
    float3 position,
    float contactDistance,
    MetalTriangleMeshInfo mesh,
    out float3 normal,
    out float penetration)
{
    if (mesh.solid_plane_count == 0u)
    {
        normal = float3(0.0f);
        penetration = 0.0f;
        return false;
    }
    float nearestSide = -3.402823466e38f;
    float3 nearestNormal = float3(0.0f);
    for (uint local = 0u; local < mesh.solid_plane_count; ++local)
    {
        MetalCollisionPlane plane =
            metalFluidSoftSolidPlanes[mesh.solid_plane_offset + local];
        float side = dot(metalLoad(plane.normal), position) - plane.offset;
        if (side > 0.0f)
        {
            normal = float3(0.0f);
            penetration = 0.0f;
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
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_fluid_soft_rigid_recover(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalFluidSoftConstants constants = metalFluidSoftConstants;
    if (node >= constants.node_count || constants.enabled == 0u ||
        metalFluidSoftNodeInverseMasses[node] <= 0.0f)
        return;
    float3 start = metalLoad(metalFluidSoftNodePrevious[node]);
    float3 end = metalLoad(metalFluidSoftNodePositions[node]);
    float bestPenetration = 0.0f;
    float3 bestNormal = float3(0.0f);
    for (uint body = 0u; body < constants.rigid_count; ++body)
    {
        MetalRigidState state = metalFluidSoftRigidStates[body];
        MetalRigidParameters parameters =
            metalFluidSoftRigidParameters[body];
        MetalTriangleMeshInfo mesh =
            metalFluidSoftMeshes[parameters.mesh_index];
        MetalQuaternion inverse =
            metalQuaternionConjugate(state.orientation);
        float3 localStart = metalRotate(
            inverse, start - metalLoad(state.position));
        float3 localEnd = metalRotate(
            inverse, end - metalLoad(state.position));
        float contactDistance =
            constants.rigid_recovery_radius + parameters.collision_margin;
        float3 lower = min(localStart, localEnd) - contactDistance;
        float3 upper = max(localStart, localEnd) + contactDistance;
        if (any(upper < metalLoad(mesh.minimum)) ||
            any(lower > metalLoad(mesh.maximum)))
            continue;
        if (mesh.solid_plane_count != 0u)
        {
            float3 normal;
            float penetration;
            if (metalFluidSoftSolidInteriorContact(
                localEnd, contactDistance, mesh, normal, penetration))
            {
                if (penetration > bestPenetration)
                {
                    bestPenetration = penetration;
                    bestNormal = metalRotate(state.orientation, normal);
                }
                continue;
            }
        }
        for (uint local = 0u; local < mesh.index_count; local += 3u)
        {
            uint ia = metalFluidSoftMeshIndices[mesh.index_offset + local];
            uint ib =
                metalFluidSoftMeshIndices[mesh.index_offset + local + 1u];
            uint ic =
                metalFluidSoftMeshIndices[mesh.index_offset + local + 2u];
            float3 a = metalLoad(
                metalFluidSoftMeshVertices[mesh.vertex_offset + ia]);
            float3 b = metalLoad(
                metalFluidSoftMeshVertices[mesh.vertex_offset + ib]);
            float3 c = metalLoad(
                metalFluidSoftMeshVertices[mesh.vertex_offset + ic]);
            if (any(upper < min(a, min(b, c))) ||
                any(lower > max(a, max(b, c))))
                continue;
            bool solid = mesh.solid_plane_count != 0u;
            float3 faceValue = cross(b - a, c - a);
            float faceSquared = dot(faceValue, faceValue);
            if (!solid && faceSquared <= 1.0e-16f) continue;
            float3 face = solid
                ? metalLoad(metalFluidSoftSolidPlanes[
                    mesh.solid_plane_offset + local / 3u].normal)
                : faceValue * rsqrt(faceSquared);
            float3 nearest =
                metalClosestPointTriangle(localEnd, a, b, c);
            float3 delta = localEnd - nearest;
            float distanceValue = length(delta);
            float3 normal = distanceValue > 1.0e-6f
                ? delta / distanceValue
                : face * (dot(localStart - a, face) >= 0.0f
                    ? 1.0f : -1.0f);
            float penetration = contactDistance - distanceValue;
            float before = dot(localStart - a, face);
            float after = dot(localEnd - a, face);
            if (before * after < 0.0f && (!solid || before > 0.0f))
            {
                float fraction = before / (before - after);
                float3 crossing =
                    localStart + (localEnd - localStart) * fraction;
                float3 hit = metalClosestPointTriangle(crossing, a, b, c);
                if (dot(hit - crossing, hit - crossing) <
                    constants.rigid_recovery_radius *
                        constants.rigid_recovery_radius)
                {
                    normal = face * (before > 0.0f ? 1.0f : -1.0f);
                    penetration = max(
                        penetration, contactDistance + abs(after));
                }
            }
            if (penetration <= bestPenetration) continue;
            bestPenetration = penetration;
            bestNormal = metalRotate(state.orientation, normal);
        }
    }
    if (bestPenetration > 0.0f)
        metalFluidSoftNodePositions[node] =
            metalStore(end + bestNormal * bestPenetration);
}

#else
import types;
import math;
import device_intrinsics;
import geometry_fluid;

public struct FluidSoftNearest
{
    public uint triangle;
    public Vec3 point;
    public Vec3 normal;
    public Vec3 weights;
    public float penetration;
};

float fluid_soft_bounds_distance(Vec3 point, BvhNode node)
{
    Vec3 delta = component_max(component_max(
        subtract(node.minimum, point), subtract(point, node.maximum)), {});
    return length_squared(delta);
}

bool fluid_soft_inside(
    Vec3 point, Vec3* surface, uint* indices, uint index_count,
    BvhNode* tree, uint* order)
{
    Vec3 direction = {1.0f, 0.371f, 0.173f};
    uint stack[64] = {};
    uint size = 1u;
    int winding = 0;
    bool ambiguous = false;
    while (size != 0u)
    {
        BvhNode node = tree[stack[--size]];
        Vec3 low = subtract(node.minimum, point);
        Vec3 high = subtract(node.maximum, point);
        float begin = max(0.0f, max(low.x,
            max(low.y / direction.y, low.z / direction.z)));
        float end = min(high.x,
            min(high.y / direction.y, high.z / direction.z));
        if (end < begin) continue;
        if (node.triangle_count == 0u)
        {
            stack[size++] = node.left;
            stack[size++] = node.right;
            continue;
        }
        for (uint item = 0u; item < node.triangle_count; ++item)
        {
            uint base = order[node.first_triangle + item] * 3u;
            Vec3 a = surface[indices[base]];
            Vec3 edge1 = subtract(surface[indices[base + 1u]], a);
            Vec3 edge2 = subtract(surface[indices[base + 2u]], a);
            Vec3 h = cross(direction, edge2), s = subtract(point, a);
            float determinant = dot(edge1, h);
            if (abs(determinant) < 1.0e-12f) continue;
            float u = dot(s, h) / determinant;
            Vec3 q = cross(s, edge1);
            float v = dot(direction, q) / determinant;
            float t = dot(edge2, q) / determinant;
            if (t < 0.0f || u < -1.0e-5f || v < -1.0e-5f ||
                u + v > 1.00001f) continue;
            if (u < 1.0e-5f || v < 1.0e-5f || u + v > 0.99999f ||
                t < 1.0e-7f) ambiguous = true;
            winding += determinant < 0.0f ? 1 : -1;
        }
    }
    if (!ambiguous) return winding != 0;
    float angle = 0.0f;
    for (uint base = 0u; base < index_count; base += 3u)
    {
        Vec3 a = subtract(surface[indices[base]], point);
        Vec3 b = subtract(surface[indices[base + 1u]], point);
        Vec3 c = subtract(surface[indices[base + 2u]], point);
        float la = vector_length(a), lb = vector_length(b), lc = vector_length(c);
        angle += 2.0f * atan2(dot(a, cross(b, c)),
            la * lb * lc + dot(a, b) * lc + dot(b, c) * la +
            dot(c, a) * lb);
    }
    return abs(angle) > 6.2831853f;
}

public FluidSoftNearest fluid_soft_nearest(
    Vec3 point, Vec3 start, Vec3* surface, Vec3* previous,
    uint* indices, uint index_count, float orientation, float distance,
    Vec3* bounds, bool sweep, BvhNode* tree, uint* order)
{
    FluidSoftNearest result = {};
    result.triangle = kInvalidDense;
    Vec3 margin = {distance, distance, distance};
    if (!bounds_overlap(component_min(point, start), component_max(point, start),
        subtract(bounds[0], margin), add(bounds[1], margin))) return result;
    float best = 3.402823466e+38f, earliest = 2.0f;
    uint stack[64] = {};
    uint size = 1u;
    while (size != 0u)
    {
        BvhNode node = tree[stack[--size]];
        if (fluid_soft_bounds_distance(point, node) > best &&
            (!sweep || !fluid_segment_bounds(start, point, node, distance)))
            continue;
        if (node.triangle_count == 0u)
        {
            bool leftNearer = fluid_soft_bounds_distance(point, tree[node.left]) <
                fluid_soft_bounds_distance(point, tree[node.right]);
            stack[size++] = leftNearer ? node.right : node.left;
            stack[size++] = leftNearer ? node.left : node.right;
            continue;
        }
        for (uint item = 0u; item < node.triangle_count; ++item)
        {
            uint base = order[node.first_triangle + item] * 3u;
            Vec3 a = surface[indices[base]];
            Vec3 b = surface[indices[base + 1u]];
            Vec3 c = surface[indices[base + 2u]];
            Vec3 crossNormal = cross(subtract(b, a), subtract(c, a));
            if (length_squared(crossNormal) < 1.0e-16f) continue;
            Vec3 normal = multiply(normalized_or(crossNormal, {}), orientation);
            Vec3 weights = {};
            Vec3 nearest = fluid_closest_triangle_barycentric(
                point, a, b, c, weights);
            float squared = length_squared(subtract(point, nearest));
            if (earliest > 1.0f && squared < best)
            {
                best = squared;
                result = {base, nearest, normal, weights,
                    distance - dot(subtract(point, nearest), normal)};
            }
            if (!sweep) continue;
            Vec3 oldNear = add(multiply(previous[indices[base]], weights.x),
                add(multiply(previous[indices[base + 1u]], weights.y),
                    multiply(previous[indices[base + 2u]], weights.z)));
            Vec3 transportedStart = add(start, subtract(nearest, oldNear));
            float first = dot(subtract(transportedStart, a), normal);
            float last = dot(subtract(point, a), normal);
            if (first < distance || last >= distance ||
                first - last < 1.0e-8f) continue;
            float time = (first - distance) / (first - last);
            if (time >= earliest) continue;
            Vec3 crossing = subtract(add(transportedStart,
                multiply(subtract(point, transportedStart), time)),
                multiply(normal, distance));
            Vec3 hitWeights = {};
            Vec3 hit = fluid_closest_triangle_barycentric(
                crossing, a, b, c, hitWeights);
            if (length_squared(subtract(crossing, hit)) > 1.0e-8f) continue;
            earliest = time;
            result = {base, hit, normal, hitWeights, distance - last};
        }
    }
    if (result.triangle == kInvalidDense) return result;
    if (earliest > 1.0f)
    {
        float radius = sqrt(best);
        bool faceInterior = result.weights.x > 1.0e-4f &&
            result.weights.y > 1.0e-4f && result.weights.z > 1.0e-4f;
        bool inside = faceInterior
            ? dot(subtract(point, result.point), result.normal) < 0.0f
            : fluid_soft_inside(point, surface, indices, index_count, tree, order);
        if (!inside)
        {
            result.penetration = distance - radius;
            if (radius > 1.0e-7f)
                result.normal = multiply(
                    subtract(point, result.point), 1.0f / radius);
        }
        else
        {
            result.penetration = distance + radius;
            if (radius > 1.0e-7f)
                result.normal = multiply(
                    subtract(result.point, point), 1.0f / radius);
        }
    }
    if (result.penetration <= 0.0f) result.triangle = kInvalidDense;
    return result;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_soft_refit(
    uniform Vec3* surface, uniform Vec3* previous, uniform uint* indices,
    uniform uint* order, uniform BvhNode* tree, uniform uint* parents,
    uniform uint* ready, uniform uint tree_count, uniform Vec3* bounds)
{
    uint index = pmThreadIndex();
    if (index >= tree_count || tree[index].triangle_count == 0u) return;
    Vec3 low = {3.402823466e+38f, 3.402823466e+38f, 3.402823466e+38f};
    Vec3 high = {-3.402823466e+38f, -3.402823466e+38f, -3.402823466e+38f};
    for (uint item = 0u; item < tree[index].triangle_count; ++item)
    {
        uint base = order[tree[index].first_triangle + item] * 3u;
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            uint vertex = indices[base + corner];
            low = component_min(low,
                component_min(surface[vertex], previous[vertex]));
            high = component_max(high,
                component_max(surface[vertex], previous[vertex]));
        }
    }
    tree[index].minimum = low;
    tree[index].maximum = high;
    while (parents[index] != kInvalidDense)
    {
        index = parents[index];
        if (pmAtomicAdd(&ready[index], 1u) == 0u) return;
        uint left = tree[index].left, right = tree[index].right;
        tree[index].minimum = component_min(
            tree[left].minimum, tree[right].minimum);
        tree[index].maximum = component_max(
            tree[left].maximum, tree[right].maximum);
    }
    bounds[0] = tree[0].minimum;
    bounds[1] = tree[0].maximum;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_soft_detect(
    uniform Vec3* positions, uniform Vec3* previous,
    uniform Vec3* velocities, uniform uint* count, uniform Vec3* surface,
    uniform Vec3* old_surface, uniform uint* indices,
    uniform uint index_count, uniform SoftBodySurfaceBinding* bindings,
    uniform Vec3* soft_velocities, uniform float orientation,
    uniform float distance, uniform Vec3* bounds, uniform bool sweep,
    uniform BvhNode* tree, uniform uint* order,
    uniform FluidSoftContact* contacts, uniform uint* counts,
    uniform uint* contact_count, uniform float* maximum_penetration)
{
    uint particle = pmThreadIndex();
    if (particle >= *count) return;
    FluidSoftContact contact = {};
    FluidSoftNearest hit = fluid_soft_nearest(positions[particle],
        previous[particle], surface, old_surface, indices, index_count,
        orientation, distance, bounds, sweep, tree, order);
    if (hit.triangle != kInvalidDense)
    {
        float barycentric[3] = {hit.weights.x, hit.weights.y, hit.weights.z};
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            SoftBodySurfaceBinding binding =
                bindings[indices[hit.triangle + corner]];
            for (uint slot = 0u; slot < 4u; ++slot)
            {
                float weight = barycentric[corner] * binding.weights[slot];
                if (weight <= 0.0f) continue;
                uint node = binding.nodes[slot];
                uint item = 0u;
                while (item < contact.count && contact.nodes[item] != node)
                    ++item;
                if (item == contact.count)
                    contact.nodes[contact.count++] = node;
                contact.weights[item] += weight;
            }
        }
        Vec3 surfaceVelocity = {};
        for (uint index = 0u; index < contact.count; ++index)
        {
            pmAtomicAdd(&counts[contact.nodes[index]], 1u);
            surfaceVelocity = add(surfaceVelocity, multiply(
                soft_velocities[contact.nodes[index]], contact.weights[index]));
        }
        contact.normal = hit.normal;
        contact.relative_velocity = subtract(
            velocities[particle], surfaceVelocity);
        contact.penetration = hit.penetration;
        pmAtomicAdd(contact_count, 1u);
        pmAtomicMax(reinterpret<uint*>(maximum_penetration),
            asuint(hit.penetration));
    }
    contacts[particle] = contact;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_soft_solve(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* accelerations, uniform float* foam, uniform uint* count,
    uniform FluidSoftContact* contacts, uniform uint* counts,
    uniform float* inverse_masses, uniform Vec3* soft_velocities,
    uniform float maximum_soft_speed, uniform float inverse_particle_mass,
    uniform float friction, uniform float maximum_projection,
    uniform float dt, uniform Vec3* node_positions,
    uniform Vec3* node_impulses)
{
    uint particle = pmThreadIndex();
    if (particle >= *count) return;
    FluidSoftContact contact = contacts[particle];
    if (contact.count == 0u) return;
    float denominator = inverse_particle_mass;
    for (uint index = 0u; index < contact.count; ++index)
    {
        uint node = contact.nodes[index];
        denominator += contact.weights[index] * inverse_masses[node] *
            float(counts[node]);
    }
    float normalSpeed = dot(contact.relative_velocity, contact.normal);
    float normalImpulse = max(0.0f, -normalSpeed) / denominator;
    Vec3 tangent = subtract(contact.relative_velocity,
        multiply(contact.normal, normalSpeed));
    Vec3 impulse = subtract(multiply(contact.normal, normalImpulse),
        clamp_length(multiply(tangent, 1.0f / denominator),
            friction * normalImpulse));
    float scale = 1.0f;
    for (uint index = 0u; index < contact.count; ++index)
    {
        uint node = contact.nodes[index];
        Vec3 velocity = soft_velocities[node];
        Vec3 delta = multiply(impulse, -contact.weights[index] *
            inverse_masses[node] * float(counts[node]));
        float a = length_squared(delta);
        if (a <= 1.0e-20f) continue;
        float b = dot(velocity, delta);
        float c = min(0.0f, length_squared(velocity) -
            maximum_soft_speed * maximum_soft_speed);
        float limit = (-b + sqrt(max(0.0f, b * b - a * c))) / a;
        scale = min(scale, max(0.0f, limit));
    }
    impulse = multiply(impulse, scale);
    float lambda = min(contact.penetration, maximum_projection) / denominator;
    positions[particle] = add(positions[particle],
        multiply(contact.normal, inverse_particle_mass * lambda));
    Vec3 velocityDelta = multiply(impulse, inverse_particle_mass);
    velocities[particle] = add(velocities[particle], velocityDelta);
    accelerations[particle] = add(accelerations[particle],
        multiply(velocityDelta, 1.0f / dt));
    foam[particle] = max(foam[particle],
        min(1.0f, normalImpulse * inverse_particle_mass * 0.1f));
    for (uint index = 0u; index < contact.count; ++index)
    {
        uint node = contact.nodes[index];
        atomic_add(&node_positions[node], multiply(contact.normal,
            -contact.weights[index] * inverse_masses[node] * lambda));
        atomic_add(&node_impulses[node],
            multiply(impulse, -contact.weights[index]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_soft_apply(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* inverse_masses, uniform Vec3* position_deltas,
    uniform Vec3* impulses, uniform Vec3* forces, uniform uint count,
    uniform float inverse_frame_dt)
{
    uint node = pmThreadIndex();
    if (node >= count) return;
    positions[node] = add(positions[node], position_deltas[node]);
    velocities[node] = add(velocities[node],
        multiply(impulses[node], inverse_masses[node]));
    forces[node] = add(forces[node],
        multiply(impulses[node], inverse_frame_dt));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fluid_soft_recover(
    uniform Vec3* positions, uniform Vec3* previous, uniform uint* count,
    uniform Vec3* surface, uniform Vec3* old_surface,
    uniform uint* indices, uniform uint index_count, uniform float orientation,
    uniform float distance, uniform Vec3* bounds, uniform bool sweep,
    uniform BvhNode* tree, uniform uint* order)
{
    uint particle = pmThreadIndex();
    if (particle >= *count) return;
    FluidSoftNearest hit = fluid_soft_nearest(positions[particle],
        previous[particle], surface, old_surface, indices, index_count,
        orientation, distance, bounds, sweep, tree, order);
    if (hit.triangle != kInvalidDense)
        positions[particle] = add(positions[particle],
            multiply(hit.normal, hit.penetration));
}

#endif
