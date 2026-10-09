// SPDX-License-Identifier: MIT
// Slang mirror of src/cloth.cuh.

import cuda_common;

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
    if (cudaBlockIdx().x != 0u) return;
    uint thread = cudaThreadIdx().x;
    if (thread == 0u)
    {
        if (reset_lambda) *lambda = 0.0f;
        clothSignedVolume = 0.0f;
    }
    pmGroupBarrier();
    for (uint wave = 0u; wave < triangle_count; wave += cudaBlockDim().x)
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
            uint count = min(cudaBlockDim().x, triangle_count - wave);
            for (uint item = 0u; item < count; ++item)
                clothSignedVolume += clothScalarTerms[item];
        }
        pmGroupBarrier();
    }
    for (uint vertex = thread; vertex < vertex_count;
         vertex += cudaBlockDim().x)
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
    for (uint wave = 0u; wave < vertex_count; wave += cudaBlockDim().x)
    {
        uint vertex = wave + thread;
        if (vertex < vertex_count)
            clothScalarTerms[thread] = inverse_masses[vertex] *
                length_squared(gradients[vertex]);
        pmGroupBarrier();
        if (thread == 0u)
        {
            uint count = min(cudaBlockDim().x, vertex_count - wave);
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
         vertex += cudaBlockDim().x)
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
    uint thread = cudaThreadIdx().x;
    for (uint pass = 0u; pass < 32u; ++pass)
    {
        if (thread == 0u) clothChanged = 0u;
        pmGroupBarrier();
        bool changed = false;
        for (uint node = thread; node < count; node += cudaBlockDim().x)
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
        for (uint node = thread; node < count; node += cudaBlockDim().x)
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
