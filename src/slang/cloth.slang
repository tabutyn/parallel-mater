// SPDX-License-Identifier: MIT
// Shared cloth simulation kernels.

#ifdef PM_TARGET_METAL
import metal_common;
#else
import types;
import math;
import device_intrinsics;
#endif

#ifndef PM_TARGET_METAL
static const uint kClothVolumeThreads = 128u;
groupshared float clothScalarTerms[kClothVolumeThreads];
groupshared float clothSignedVolume;
groupshared float clothDeltaLambda;
groupshared uint clothApplyCorrection;
groupshared uint clothChanged;

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_project_volume(
    uniform Vec3* positions, uniform float* inverse_masses,
    uniform uint* indices, uniform uint vertex_count,
    uniform uint triangle_count, uniform uint* corner_offsets,
    uniform uint* corner_indices, uniform Vec3* gradients,
    uniform float target_volume, uniform float orientation,
    uniform float compliance, uniform float dt, uniform float* lambda,
    uniform bool reset_lambda)
{
    if (pmBlockIndex().x != 0u) return;
    uint thread = pmGroupThreadIndex().x;
    if (thread == 0u)
    {
        if (reset_lambda) *lambda = 0.0f;
        clothSignedVolume = 0.0f;
    }
    pmGroupBarrier();
    for (uint wave = 0u; wave < triangle_count; wave += pmGroupSize().x)
    {
        uint triangle = wave + thread;
        if (triangle < triangle_count)
        {
            uint aIndex = indices[3u * triangle];
            uint bIndex = indices[3u * triangle + 1u];
            uint cIndex = indices[3u * triangle + 2u];
            Vec3 a = positions[aIndex];
            Vec3 b = positions[bIndex];
            Vec3 c = positions[cIndex];
            clothScalarTerms[thread] = dot(a, cross(b, c)) / 6.0f;
        }
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint count = min(pmGroupSize().x, triangle_count - wave);
            for (uint item = 0u; item < count; ++item)
                clothSignedVolume += clothScalarTerms[item];
        }
        pmGroupBarrier();
    }
    for (uint vertex = thread; vertex < vertex_count;
         vertex += pmGroupSize().x)
    {
        Vec3 gradient = {};
        for (uint item = corner_offsets[vertex];
             item < corner_offsets[vertex + 1u]; ++item)
        {
            uint corner = corner_indices[item];
            uint base = corner - corner % 3u;
            Vec3 a = positions[indices[base]];
            Vec3 b = positions[indices[base + 1u]];
            Vec3 c = positions[indices[base + 2u]];
            Vec3 contribution = corner % 3u == 0u ? cross(b, c) :
                corner % 3u == 1u ? cross(c, a) : cross(a, b);
            gradient = add(gradient,
                multiply(contribution, orientation / 6.0f));
        }
        gradients[vertex] = gradient;
    }
    pmGroupBarrier();
    if (thread == 0u) clothApplyCorrection = 0u;
    pmGroupBarrier();
    float inverseMassSum = 0.0f;
    for (uint wave = 0u; wave < vertex_count; wave += pmGroupSize().x)
    {
        uint vertex = wave + thread;
        if (vertex < vertex_count)
            clothScalarTerms[thread] = inverse_masses[vertex] *
                length_squared(gradients[vertex]);
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint count = min(pmGroupSize().x, vertex_count - wave);
            for (uint item = 0u; item < count; ++item)
                inverseMassSum += clothScalarTerms[item];
        }
        pmGroupBarrier();
    }
    if (thread == 0u)
    {
        clothApplyCorrection = inverseMassSum > 1.0e-12f ? 1u : 0u;
        if (clothApplyCorrection != 0u)
        {
            float alpha = compliance / (dt * dt);
            float constraint =
                orientation * clothSignedVolume - target_volume;
            clothDeltaLambda =
                (-constraint - alpha * (*lambda)) / (inverseMassSum + alpha);
            *lambda += clothDeltaLambda;
        }
    }
    pmGroupBarrier();
    if (clothApplyCorrection == 0u) return;
    for (uint vertex = thread; vertex < vertex_count;
         vertex += pmGroupSize().x)
    {
        if (inverse_masses[vertex] == 0.0f) continue;
        positions[vertex] = add(positions[vertex], multiply(
            gradients[vertex], inverse_masses[vertex] * clothDeltaLambda));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_break_bonds(
    uniform Vec3* positions, uniform ClothBond* bonds,
    uniform uint8_t* active, uniform uint8_t* damage, uniform uint count,
    uniform float break_strain, uniform uint persistence,
    uniform FluidBodyImpulse* contact_impulses,
    uniform float impact_threshold)
{
    uint edgeIndex = pmThreadIndex();
    if (edgeIndex >= count || active[edgeIndex] == 0u) return;
    ClothBond bond = bonds[edgeIndex];
    if (impact_threshold > 0.0f && contact_impulses != nullptr &&
        vector_length(contact_impulses[bond.first].linear) +
        vector_length(contact_impulses[bond.second].linear) > impact_threshold)
    {
        active[edgeIndex] = 0u;
        damage[edgeIndex] = 0u;
        return;
    }
    if (break_strain <= 0.0f) return;
    float currentLength = vector_length(subtract(
        positions[bond.first], positions[bond.second]));
    if (!isfinite(currentLength) ||
        currentLength <= bond.rest_length * (1.0f + break_strain))
    {
        damage[edgeIndex] = 0u;
        return;
    }
    uint next = min(persistence, uint(damage[edgeIndex]) + 1u);
    damage[edgeIndex] = uint8_t(next);
    if (next >= persistence)
    {
        active[edgeIndex] = 0u;
        damage[edgeIndex] = 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_limit_strain(
    uniform Vec3* positions, uniform Vec3* scratch,
    uniform float* inverse_masses, uniform uint* offsets,
    uniform DeformableNeighbor* neighbors, uniform uint8_t* free_nodes,
    uniform uint count)
{
    uint thread = pmGroupThreadIndex().x;
    for (uint pass = 0u; pass < 32u; ++pass)
    {
        if (thread == 0u) clothChanged = 0u;
        pmGroupBarrier();
        bool changed = false;
        for (uint node = thread; node < count; node += pmGroupSize().x)
        {
            Vec3 correction = {};
            uint degree = 0u;
            for (uint item = offsets[node];
                 free_nodes[node] != 0u && item < offsets[node + 1u]; ++item)
            {
                DeformableNeighbor edge = neighbors[item];
                if (edge.bond != kInvalidDense) continue;
                float weight = inverse_masses[node] +
                    inverse_masses[edge.index];
                Vec3 delta = subtract(positions[edge.index], positions[node]);
                float currentLength = vector_length(delta);
                float maximum = edge.rest_length * 1.10f;
                if (weight <= 0.0f || currentLength <= maximum ||
                    inverse_masses[node] == 0.0f) continue;
                correction = add(correction, multiply(delta,
                    (currentLength - maximum) * inverse_masses[node] /
                    (currentLength * weight)));
                changed = changed || currentLength > maximum * 1.0001f;
                ++degree;
            }
            scratch[node] = add(positions[node],
                multiply(correction, 1.0f / float(max(1u, degree))));
        }
        if (changed)
        {
            uint ignored;
            InterlockedOr(clothChanged, 1u, ignored);
        }
        pmGroupBarrier();
        for (uint node = thread; node < count; node += pmGroupSize().x)
            positions[node] = scratch[node];
        pmGroupBarrier();
        if (clothChanged == 0u) break;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cloth_update_surface(
    uniform Vec3* positions, uniform uint* source_indices,
    uniform Vec3* surface_positions, uniform uint triangle_count)
{
    uint triangle = pmThreadIndex();
    if (triangle >= triangle_count) return;
    uint base = 3u * triangle;
    for (uint corner = 0u; corner < 3u; ++corner)
        surface_positions[base + corner] =
            positions[source_indices[base + corner]];
}
#endif

#ifdef PM_TARGET_METAL
groupshared float clothScalarTerms[128];
groupshared float clothSignedVolume;
groupshared float clothDeltaLambda;

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalClothPositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalClothPrevious : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalClothVelocities : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<float>
    metalClothInverseMasses : register(t3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalBond>
    metalClothBonds : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<uint8_t>
    metalClothActiveBonds : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalClothSurfacePositions : register(u6);
[[vk::binding(7, 0)]] ConstantBuffer<MetalDeformableConstants>
    metalClothConstants : register(b7);
[[vk::binding(8, 0)]] StructuredBuffer<uint>
    metalClothTriangleIndices : register(t8);
[[vk::binding(9, 0)]] RWStructuredBuffer<uint8_t>
    metalClothBondDamage : register(u9);
[[vk::binding(10, 0)]] StructuredBuffer<uint>
    metalClothSurfaceSources : register(t10);
[[vk::binding(12, 0)]] RWStructuredBuffer<MetalPackedVec3>
    metalClothScratch : register(u12);
[[vk::binding(13, 0)]] StructuredBuffer<uint>
    metalClothNeighborOffsets : register(t13);
[[vk::binding(14, 0)]] StructuredBuffer<MetalDeformableNeighbor>
    metalClothNeighbors : register(t14);
[[vk::binding(15, 0)]] StructuredBuffer<uint8_t>
    metalClothFreeNodes : register(t15);
[[vk::binding(16, 0)]] StructuredBuffer<MetalPackedVec3>
    metalClothRigidImpulses : register(t16);

groupshared float metalClothDenominator;

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_cloth_predict(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    metalIntegrateDeformableNode(
        metalClothPositions, metalClothPrevious, metalClothVelocities,
        metalClothInverseMasses, metalClothConstants, dispatchThreadId.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_cloth_project_bonds(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalDeformableConstants constants = metalClothConstants;
    if (node >= constants.count) return;
    float3 position = metalLoad(metalClothPositions[node]);
    float selfInverse = metalClothInverseMasses[node];
    if (selfInverse <= 0.0f)
    {
        metalClothScratch[node] = metalClothPositions[node];
        return;
    }
    float3 correction = float3(0.0f);
    float shortestRestLength = 3.402823466e+38f;
    uint first = metalClothNeighborOffsets[node];
    uint last = metalClothNeighborOffsets[node + 1u];
    for (uint edgeIndex = first; edgeIndex < last; ++edgeIndex)
    {
        MetalDeformableNeighbor edge = metalClothNeighbors[edgeIndex];
        if (edge.bond != 0xffffffffu &&
            metalClothActiveBonds[edge.bond] == 0u) continue;
        shortestRestLength = min(shortestRestLength, edge.rest_length);
        float3 difference = position -
            metalLoad(metalClothPositions[edge.index]);
        float currentLength = length(difference);
        if (currentLength < 1.0e-7f) continue;
        float denominator = selfInverse +
            metalClothInverseMasses[edge.index] + edge.compliance /
            (constants.timestep * constants.timestep);
        if (denominator <= 0.0f) continue;
        correction += difference *
            (-selfInverse * (currentLength - edge.rest_length) /
             (denominator * currentLength));
    }
    float3 proposal = correction / float(max(1u, last - first));
    if (constants.maximum_projection_fraction > 0.0f &&
        shortestRestLength < 3.402823466e+38f)
        proposal = metalLimit(proposal,
            constants.maximum_projection_fraction * shortestRestLength);
    metalClothScratch[node] = metalStore(position + proposal);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_cloth_apply_bonds(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    if (node < metalClothConstants.count)
        metalClothPositions[node] = metalClothScratch[node];
}

[shader("compute")]
[numthreads(1, 1, 1)]
public void pm_cloth_limit_strain_serial(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u) return;
    MetalDeformableConstants constants = metalClothConstants;
    for (uint pass = 0u; pass < 32u; ++pass)
    {
        bool changed = false;
        for (uint node = 0u; node < constants.count; ++node)
        {
            float3 position = metalLoad(metalClothPositions[node]);
            float3 correction = float3(0.0f);
            uint degree = 0u;
            if (metalClothFreeNodes[node] != 0u)
            {
                for (uint item = metalClothNeighborOffsets[node];
                     item < metalClothNeighborOffsets[node + 1u]; ++item)
                {
                    MetalDeformableNeighbor edge = metalClothNeighbors[item];
                    if (edge.bond != 0xffffffffu) continue;
                    float weight = metalClothInverseMasses[node] +
                        metalClothInverseMasses[edge.index];
                    float3 delta =
                        metalLoad(metalClothPositions[edge.index]) - position;
                    float currentLength = length(delta);
                    float maximum = edge.rest_length * 1.10f;
                    if (weight <= 0.0f || currentLength <= maximum ||
                        metalClothInverseMasses[node] == 0.0f) continue;
                    correction += delta * ((currentLength - maximum) *
                        metalClothInverseMasses[node] /
                        (currentLength * weight));
                    changed = changed ||
                        currentLength > maximum * 1.0001f;
                    ++degree;
                }
            }
            metalClothScratch[node] = metalStore(
                position + correction / float(max(1u, degree)));
        }
        for (uint node = 0u; node < constants.count; ++node)
            metalClothPositions[node] = metalClothScratch[node];
        if (!changed) break;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
public void pm_cloth_project_volume(uint groupIndex : SV_GroupIndex)
{
    uint lane = groupIndex;
    MetalDeformableConstants constants = metalClothConstants;
    uint triangleCount = constants.triangle_index_count / 3u;
    if (lane == 0u) clothSignedVolume = 0.0f;
    pmGroupBarrier();
    for (uint wave = 0u; wave < triangleCount; wave += 128u)
    {
        uint triangle = wave + lane;
        float contribution = 0.0f;
        if (triangle < triangleCount)
        {
            uint base = 3u * triangle;
            float3 a = metalLoad(metalClothPositions[
                metalClothTriangleIndices[base]]);
            float3 b = metalLoad(metalClothPositions[
                metalClothTriangleIndices[base + 1u]]);
            float3 c = metalLoad(metalClothPositions[
                metalClothTriangleIndices[base + 2u]]);
            contribution = dot(a, cross(b, c)) / 6.0f;
        }
        clothScalarTerms[lane] = contribution;
        pmGroupBarrier();
        if (lane == 0u)
        {
            uint count = min(128u, triangleCount - wave);
            for (uint item = 0u; item < count; ++item)
                clothSignedVolume += clothScalarTerms[item];
        }
        pmGroupBarrier();
    }
    for (uint node = lane; node < constants.count; node += 128u)
    {
        float3 gradient = float3(0.0f);
        for (uint base = 0u; base < constants.triangle_index_count;
             base += 3u)
        {
            uint aIndex = metalClothTriangleIndices[base];
            uint bIndex = metalClothTriangleIndices[base + 1u];
            uint cIndex = metalClothTriangleIndices[base + 2u];
            float3 a = metalLoad(metalClothPositions[aIndex]);
            float3 b = metalLoad(metalClothPositions[bIndex]);
            float3 c = metalLoad(metalClothPositions[cIndex]);
            if (node == aIndex) gradient += cross(b, c) / 6.0f;
            if (node == bIndex) gradient += cross(c, a) / 6.0f;
            if (node == cIndex) gradient += cross(a, b) / 6.0f;
        }
        metalClothScratch[node] = metalStore(gradient);
    }
    pmGroupBarrier();
    if (lane == 0u)
        metalClothDenominator = constants.volume_compliance /
            max(constants.timestep * constants.timestep, 1.0e-12f);
    pmGroupBarrier();
    for (uint wave = 0u; wave < constants.count; wave += 128u)
    {
        uint node = wave + lane;
        clothScalarTerms[lane] = node < constants.count
            ? metalClothInverseMasses[node] *
                dot(metalLoad(metalClothScratch[node]),
                    metalLoad(metalClothScratch[node]))
            : 0.0f;
        pmGroupBarrier();
        if (lane == 0u)
        {
            uint count = min(128u, constants.count - wave);
            for (uint item = 0u; item < count; ++item)
                metalClothDenominator += clothScalarTerms[item];
        }
        pmGroupBarrier();
    }
    if (lane == 0u)
        clothDeltaLambda = metalClothDenominator > 1.0e-12f
            ? (clothSignedVolume - constants.target_volume) /
                metalClothDenominator
            : 0.0f;
    pmGroupBarrier();
    for (uint node = lane; node < constants.count; node += 128u)
        metalClothPositions[node] = metalStore(
            metalLoad(metalClothPositions[node]) -
            metalLoad(metalClothScratch[node]) *
                metalClothInverseMasses[node] * clothDeltaLambda);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_cloth_damage(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalDeformableConstants constants = metalClothConstants;
    if (index >= constants.bond_count ||
        (constants.break_strain <= 0.0f &&
         constants.impact_break_impulse <= 0.0f)) return;
    MetalBond bond = metalClothBonds[index];
    if (bond.active == 0u) return;
    if (constants.impact_break_impulse > 0.0f &&
        length(metalLoad(metalClothRigidImpulses[bond.first])) +
        length(metalLoad(metalClothRigidImpulses[bond.second])) >
            constants.impact_break_impulse)
    {
        bond.active = 0u;
        metalClothBonds[index] = bond;
        metalClothActiveBonds[index] = 0u;
        metalClothBondDamage[index] = 0u;
        return;
    }
    if (constants.break_strain <= 0.0f) return;
    float currentLength = length(
        metalLoad(metalClothPositions[bond.first]) -
        metalLoad(metalClothPositions[bond.second]));
    if (isfinite(currentLength) &&
        currentLength > bond.rest_length * (1.0f + constants.break_strain))
    {
        uint persistence = max(constants.fracture_persistence, 1u);
        uint next = min(
            persistence, uint(metalClothBondDamage[index]) + 1u);
        metalClothBondDamage[index] = uint8_t(next);
        if (next >= persistence)
        {
            bond.active = 0u;
            metalClothBonds[index] = bond;
            metalClothActiveBonds[index] = 0u;
            metalClothBondDamage[index] = 0u;
        }
    }
    else
    {
        metalClothBondDamage[index] = 0u;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_cloth_finalize(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    metalFinalizeDeformableNode(
        metalClothPositions, metalClothPrevious, metalClothVelocities,
        metalClothInverseMasses, metalClothConstants, dispatchThreadId.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_cloth_surface_update(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalDeformableConstants constants = metalClothConstants;
    if (index >= constants.surface_count) return;
    uint source = metalClothSurfaceSources[index];
    if (source < constants.count)
        metalClothSurfacePositions[index] = metalClothPositions[source];
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_cloth_impact(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint index = dispatchThreadId.x;
    MetalDeformableConstants constants = metalClothConstants;
    if (index >= constants.bond_count ||
        constants.impact_break_impulse <= 0.0f) return;
    MetalBond bond = metalClothBonds[index];
    if (bond.active == 0u) return;
    float impulse =
        length(metalLoad(metalClothRigidImpulses[bond.first])) +
        length(metalLoad(metalClothRigidImpulses[bond.second]));
    if (impulse <= constants.impact_break_impulse) return;
    bond.active = 0u;
    metalClothBonds[index] = bond;
    metalClothActiveBonds[index] = 0u;
}
#endif
