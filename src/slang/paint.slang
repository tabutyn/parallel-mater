// SPDX-License-Identifier: MIT
// Metal compatibility paint transfer. The command runtime shares the same
// contact-driven paint equations through geometry_fluid/geometry_cloth.

import metal_common;

struct MetalPaintConstants
{
    uint mode;
    MetalHandle source;
    MetalHandle target;
    uint mesh_index;
    uint width;
    uint height;
    float reach;
    float particle_radius;
    float cloth_thickness;
    uint rigid_count;
    uint cloth_count;
    uint cloth_index_count;
    uint enabled;
};

[[vk::binding(0, 0)]] StructuredBuffer<MetalPackedVec3>
    paintFluidPositions : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<MetalParticleMetadata>
    paintFluidMetadata : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<MetalPackedVec2>
    paintUvs : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>
    paintPixels : register(u3);
[[vk::binding(4, 0)]] StructuredBuffer<MetalHandle>
    paintRigidIds : register(t4);
[[vk::binding(5, 0)]] StructuredBuffer<MetalRigidState>
    paintRigidStates : register(t5);
[[vk::binding(6, 0)]] StructuredBuffer<MetalRigidParameters>
    paintRigidParameters : register(t6);
[[vk::binding(7, 0)]] StructuredBuffer<MetalPackedVec3>
    paintMeshVertices : register(t7);
[[vk::binding(8, 0)]] StructuredBuffer<uint>
    paintMeshIndices : register(t8);
[[vk::binding(9, 0)]] StructuredBuffer<MetalTriangleMeshInfo>
    paintMeshes : register(t9);
[[vk::binding(10, 0)]] StructuredBuffer<MetalPackedVec3>
    paintClothPositions : register(t10);
[[vk::binding(11, 0)]] StructuredBuffer<uint>
    paintClothIndices : register(t11);
[[vk::binding(12, 0)]] StructuredBuffer<uint>
    paintClothSources : register(t12);
[[vk::binding(13, 0)]] ConstantBuffer<MetalPaintConstants>
    paintConstants : register(b13);

[ForceInline]
static void paintStamp(float2 uv, uint side)
{
    int x = int(floor(uv.x * float(paintConstants.width))) %
        int(paintConstants.width);
    if (x < 0) x += int(paintConstants.width);
    int y = clamp(int(floor(uv.y * float(paintConstants.height))), 0,
        int(paintConstants.height) - 1);
    uint ignored;
    InterlockedOr(paintPixels[uint(y) * paintConstants.width + uint(x)],
        side, ignored);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_paint(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint3 groupThreadId : SV_GroupThreadID)
{
    MetalPaintConstants constants = paintConstants;
    if (constants.enabled == 0u || constants.width == 0u ||
        constants.height == 0u) return;

    uint bodyIndex = constants.rigid_count;
    MetalHandle bodyHandle = constants.mode == 0u
        ? constants.target : constants.source;
    for (uint index = 0u; index < constants.rigid_count; ++index)
    {
        MetalHandle candidate = paintRigidIds[index];
        if (candidate.index == bodyHandle.index &&
            candidate.generation == bodyHandle.generation)
        {
            bodyIndex = index;
            break;
        }
    }
    if (bodyIndex == constants.rigid_count) return;

    MetalRigidState state = paintRigidStates[bodyIndex];
    if (constants.mode == 0u)
    {
        MetalTriangleMeshInfo mesh = paintMeshes[constants.mesh_index];
        float queryRadius = constants.particle_radius + constants.reach;
        float querySquared = queryRadius * queryRadius;
        MetalQuaternion inverseOrientation =
            metalQuaternionConjugate(state.orientation);
        MetalParticleMetadata metadata = paintFluidMetadata[0];
        for (uint particle = dispatchThreadId.x;
             particle < metadata.count; particle += 64u)
        {
            float3 localParticle = metalRotate(inverseOrientation,
                metalLoad(paintFluidPositions[particle]) -
                    metalLoad(state.position));
            float bestSquared = querySquared;
            float2 bestUv = float2(0.0f);
            uint bestSide = 0u;
            for (uint local = 0u; local < mesh.index_count; local += 3u)
            {
                uint ia = paintMeshIndices[mesh.index_offset + local];
                uint ib = paintMeshIndices[mesh.index_offset + local + 1u];
                uint ic = paintMeshIndices[mesh.index_offset + local + 2u];
                float3 a = metalLoad(
                    paintMeshVertices[mesh.vertex_offset + ia]);
                float3 b = metalLoad(
                    paintMeshVertices[mesh.vertex_offset + ib]);
                float3 c = metalLoad(
                    paintMeshVertices[mesh.vertex_offset + ic]);
                float3 nearest =
                    metalClosestPointTriangle(localParticle, a, b, c);
                float3 delta = localParticle - nearest;
                float squared = dot(delta, delta);
                if (squared >= bestSquared) continue;
                float3 weights = metalTriangleWeights(nearest, a, b, c);
                float2 ua = float2(paintUvs[ia].x, paintUvs[ia].y);
                float2 ub = float2(paintUvs[ib].x, paintUvs[ib].y);
                float2 uc = float2(paintUvs[ic].x, paintUvs[ic].y);
                bestUv = weights.x * ua + weights.y * ub + weights.z * uc;
                bestSide = dot(cross(b - a, c - a), delta) >= 0.0f
                    ? 1u : 2u;
                bestSquared = squared;
            }
            if (bestSide != 0u) paintStamp(bestUv, bestSide);
        }
        return;
    }

    // One deterministic lane owns cloth stamping. Paint is not on the solver
    // critical path, and this avoids backend-specific reduction code.
    if (groupThreadId.x != 0u) return;
    MetalRigidParameters body = paintRigidParameters[bodyIndex];
    MetalTriangleMeshInfo mesh = paintMeshes[body.mesh_index];
    float3 center = metalLoad(state.position);
    float contactRadius =
        mesh.radius + body.collision_margin + constants.cloth_thickness;
    float bestPenetration = 0.0f;
    float3 contact = float3(0.0f);
    uint triangleCount = constants.cloth_index_count / 3u;
    for (uint triangle = 0u; triangle < triangleCount; ++triangle)
    {
        uint local = 3u * triangle;
        uint ia = paintClothIndices[local];
        uint ib = paintClothIndices[local + 1u];
        uint ic = paintClothIndices[local + 2u];
        if (ia >= constants.cloth_count || ib >= constants.cloth_count ||
            ic >= constants.cloth_count) continue;
        float3 a = metalLoad(paintClothPositions[ia]);
        float3 b = metalLoad(paintClothPositions[ib]);
        float3 c = metalLoad(paintClothPositions[ic]);
        float3 normal = cross(b - a, c - a);
        if (dot(normal, normal) <= 1.0e-12f) continue;
        float3 nearest = metalClosestPointTriangle(center, a, b, c);
        float penetration = contactRadius - distance(center, nearest);
        if (penetration > bestPenetration)
        {
            bestPenetration = penetration;
            contact = nearest;
        }
    }
    if (bestPenetration <= 0.0f) return;

    float brushSquared = constants.reach * constants.reach;
    for (uint triangle = 0u; triangle < triangleCount; ++triangle)
    {
        uint local = 3u * triangle;
        uint ia = paintClothIndices[local];
        uint ib = paintClothIndices[local + 1u];
        uint ic = paintClothIndices[local + 2u];
        if (ia >= constants.cloth_count || ib >= constants.cloth_count ||
            ic >= constants.cloth_count) continue;
        float3 a = metalLoad(paintClothPositions[ia]);
        float3 b = metalLoad(paintClothPositions[ib]);
        float3 c = metalLoad(paintClothPositions[ic]);
        float3 nearest = metalClosestPointTriangle(contact, a, b, c);
        if (dot(contact - nearest, contact - nearest) > brushSquared) continue;
        float2 ua = float2(paintUvs[paintClothSources[ia]].x,
                           paintUvs[paintClothSources[ia]].y);
        float2 ub = float2(paintUvs[paintClothSources[ib]].x,
                           paintUvs[paintClothSources[ib]].y);
        float2 uc = float2(paintUvs[paintClothSources[ic]].x,
                           paintUvs[paintClothSources[ic]].y);
        float2 firstEdge = ub - ua;
        float2 secondEdge = uc - ua;
        float determinant = firstEdge.x * secondEdge.y -
                            firstEdge.y * secondEdge.x;
        if (abs(determinant) < 1.0e-10f) continue;
        int x0 = max(0, int(floor(min(ua.x, min(ub.x, uc.x)) *
                                  constants.width)));
        int x1 = min(int(constants.width) - 1,
            int(floor(max(ua.x, max(ub.x, uc.x)) * constants.width)));
        int y0 = max(0, int(floor(min(ua.y, min(ub.y, uc.y)) *
                                  constants.height)));
        int y1 = min(int(constants.height) - 1,
            int(floor(max(ua.y, max(ub.y, uc.y)) * constants.height)));
        uint side = dot(cross(b - a, c - a), center - contact) >= 0.0f
            ? 1u : 2u;
        for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
        {
            float2 query = (float2(float(x) + 0.5f, float(y) + 0.5f) /
                float2(float(constants.width), float(constants.height))) - ua;
            float v = (query.x * secondEdge.y -
                       query.y * secondEdge.x) / determinant;
            float w = (firstEdge.x * query.y -
                       firstEdge.y * query.x) / determinant;
            float u = 1.0f - v - w;
            if (u < -1.0e-4f || v < -1.0e-4f || w < -1.0e-4f) continue;
            float3 point = u * a + v * b + w * c;
            if (dot(point - contact, point - contact) > brushSquared) continue;
            uint ignored;
            InterlockedOr(paintPixels[uint(y) * constants.width + uint(x)],
                side, ignored);
        }
    }
}
