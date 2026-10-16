#include "dispatch_entry.slang"
// SPDX-License-Identifier: MIT
// Shared cloth simulation kernels.

import types;
import math;
import device_intrinsics;
import graphics_resources;

static const uint kClothVolumeThreads = 128u;
groupshared float clothScalarTerms[kClothVolumeThreads];
groupshared float clothSignedVolume;
groupshared float clothDeltaLambda;
groupshared uint clothApplyCorrection;
groupshared uint clothChanged;

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_project_volume(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<float> inverse_masses,
    uniform PMBuffer<uint> indices, uniform uint vertex_count,
    uniform uint triangle_count, uniform PMBuffer<uint> corner_offsets,
    uniform PMBuffer<uint> corner_indices, uniform PMBuffer<Vec3> gradients,
    uniform float target_volume, uniform float orientation,
    uniform float compliance, uniform float dt, uniform PMBuffer<float> lambda,
    uniform bool reset_lambda PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    if (pmBlockIndex().x != 0u) return;
    uint thread = pmGroupThreadIndex().x;
    if (thread == 0u)
    {
        if (reset_lambda) lambda[0] = 0.0f;
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
            lambda[0] += clothDeltaLambda;
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
[numthreads(128, 1, 1)]
void cloth_break_bonds(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<ClothBond> bonds,
    uniform PMByteBuffer active, uniform PMByteBuffer damage, uniform uint count,
    uniform float break_strain, uniform uint persistence,
    uniform PMBuffer<FluidBodyImpulse> contact_impulses,
    uniform float impact_threshold PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint edgeIndex = pmThreadIndex();
    if (edgeIndex >= count || active[edgeIndex] == 0u) return;
    ClothBond bond = bonds[edgeIndex];
    if (impact_threshold > 0.0f && pmBufferPresent(contact_impulses) &&
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
    damage[edgeIndex] = PMResourceUint8(next);
    if (next >= persistence)
    {
        active[edgeIndex] = 0u;
        damage[edgeIndex] = 0u;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void cloth_limit_strain(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<Vec3> scratch,
    uniform PMBuffer<float> inverse_masses, uniform PMBuffer<uint> offsets,
    uniform PMBuffer<DeformableNeighbor> neighbors, uniform PMByteBuffer free_nodes,
    uniform uint count PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
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
[numthreads(128, 1, 1)]
void cloth_update_surface(
    uniform PMBuffer<Vec3> positions, uniform PMBuffer<uint> source_indices,
    uniform PMBuffer<Vec3> surface_positions, uniform uint triangle_count PM_D3D12_DISPATCH_PARAMETER)
{
    PM_D3D12_DISPATCH_PROLOGUE();
    uint triangle = pmThreadIndex();
    if (triangle >= triangle_count) return;
    uint base = 3u * triangle;
    for (uint corner = 0u; corner < 3u; ++corner)
        surface_positions[base + corner] =
            positions[source_indices[base + corner]];
}
