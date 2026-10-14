// SPDX-License-Identifier: MIT
// Production rope/soft-body state exchange for Metal.

import metal_common;

struct MetalRopeSoftConstants
{
    float timestep;
    uint rope_count;
    uint soft_count;
    uint surface_count;
    uint surface_index_count;
    float contact_distance;
    float friction;
    float maximum_soft_acceleration;
    float maximum_rope_speed;
    float maximum_soft_speed;
    float node_radius;
    float anchor_support_radius_scale;
    float anchor_contact_support_radius_scale;
    float orientation;
    uint attach_first;
    uint attach_last;
    uint first_triangle;
    MetalPackedVec3 first_weights;
    MetalPackedVec3 first_offset;
    uint last_triangle;
    MetalPackedVec3 last_weights;
    MetalPackedVec3 last_offset;
    uint enabled;
    float frame_inverse_timestep;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropeSoftRopePositions : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropeSoftRopePrevious : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropeSoftRopeVelocities : register(u2);
[[vk::binding(3, 0)]] StructuredBuffer<float>
    ropeSoftRopeInverseMasses : register(t3);
[[vk::binding(4, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropeSoftRopeForces : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropeSoftSoftPositions : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropeSoftSoftVelocities : register(u6);
[[vk::binding(7, 0)]] StructuredBuffer<float>
    ropeSoftSoftInverseMasses : register(t7);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropeSoftSurfacePositions : register(u8);
[[vk::binding(9, 0)]] StructuredBuffer<uint>
    ropeSoftSurfaceIndices : register(t9);
[[vk::binding(10, 0)]] StructuredBuffer<MetalSurfaceBinding>
    ropeSoftSurfaceBindings : register(t10);
[[vk::binding(11, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropeSoftSoftForces : register(u11);
[[vk::binding(12, 0)]] StructuredBuffer<MetalPackedVec3>
    ropeSoftSoftRestPositions : register(t12);
[[vk::binding(13, 0)]] StructuredBuffer<MetalPackedVec3>
    ropeSoftSurfaceRestPositions : register(t13);
[[vk::binding(14, 0)]] ConstantBuffer<MetalRopeSoftConstants>
    ropeSoftConstants : register(b14);
[[vk::binding(15, 0)]] RWStructuredBuffer<uint>
    ropeSoftDiagnosticContactCount : register(u15);
[[vk::binding(16, 0)]] RWStructuredBuffer<float>
    ropeSoftDiagnosticMaximumPenetration : register(u16);
[[vk::binding(17, 0)]] RWStructuredBuffer<MetalRopeAnchorState>
    ropeSoftAnchors : register(u17);
[[vk::binding(18, 0)]] RWStructuredBuffer<uint>
    ropeSoftPacked : register(u18);
[[vk::binding(19, 0)]] RWStructuredBuffer<MetalPackedVec3>
    ropeSoftPreviousSurface : register(u19);

static const uint ropeSoftHeaderWords = 32u;
static const uint ropeSoftValid = 0u;
static const uint ropeSoftNodeCount = 1u;
static const uint ropeSoftSurfaceCount = 2u;
static const uint ropeSoftIndexCount = 3u;
static const uint ropeSoftSurfaceOffset = 4u;
static const uint ropeSoftPreviousOffset = 5u;
static const uint ropeSoftVelocityOffset = 6u;
static const uint ropeSoftInverseOffset = 7u;
static const uint ropeSoftIndexOffset = 8u;
static const uint ropeSoftBindingOffset = 9u;
static const uint ropeSoftImpulseOffset = 10u;
static const uint ropeSoftContactCount = 11u;
static const uint ropeSoftMaximumPenetration = 12u;
static const uint ropeSoftAttachFirst = 13u;
static const uint ropeSoftAttachLast = 14u;
static const uint ropeSoftDistance = 15u;
static const uint ropeSoftFriction = 16u;
static const uint ropeSoftOrientation = 17u;
static const uint ropeSoftPriorContactCount = 18u;
static const uint ropeSoftFirstAnchorWeight = 19u;
static const uint ropeSoftLastAnchorWeight = 20u;

[ForceInline]
static float ropeSoftLoadFloat(uint word)
{
    return asfloat(ropeSoftPacked[word]);
}

[ForceInline]
static float3 ropeSoftLoadVec3(uint word)
{
    return float3(asfloat(ropeSoftPacked[word]),
        asfloat(ropeSoftPacked[word + 1u]),
        asfloat(ropeSoftPacked[word + 2u]));
}

[ForceInline]
static void ropeSoftStoreVec3(uint word, float3 value)
{
    ropeSoftPacked[word] = asuint(value.x);
    ropeSoftPacked[word + 1u] = asuint(value.y);
    ropeSoftPacked[word + 2u] = asuint(value.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rope_soft_pack(uint groupIndex : SV_GroupIndex)
{
    MetalRopeSoftConstants constants = ropeSoftConstants;
    if (constants.enabled == 0u)
    {
        if (groupIndex == 0u) ropeSoftPacked[ropeSoftValid] = 0u;
        return;
    }
    uint surfaceOffset = ropeSoftHeaderWords;
    uint previousOffset = surfaceOffset + 3u * constants.surface_count;
    uint velocityOffset = previousOffset + 3u * constants.surface_count;
    uint inverseOffset = velocityOffset + 3u * constants.soft_count;
    uint indexOffset = inverseOffset + constants.soft_count;
    uint bindingOffset = indexOffset + constants.surface_index_count;
    uint impulseOffset = bindingOffset + 8u * constants.surface_count;
    if (groupIndex == 0u)
    {
        ropeSoftPacked[ropeSoftValid] = 1u;
        ropeSoftPacked[ropeSoftNodeCount] = constants.soft_count;
        ropeSoftPacked[ropeSoftSurfaceCount] = constants.surface_count;
        ropeSoftPacked[ropeSoftIndexCount] = constants.surface_index_count;
        ropeSoftPacked[ropeSoftSurfaceOffset] = surfaceOffset;
        ropeSoftPacked[ropeSoftPreviousOffset] = previousOffset;
        ropeSoftPacked[ropeSoftVelocityOffset] = velocityOffset;
        ropeSoftPacked[ropeSoftInverseOffset] = inverseOffset;
        ropeSoftPacked[ropeSoftIndexOffset] = indexOffset;
        ropeSoftPacked[ropeSoftBindingOffset] = bindingOffset;
        ropeSoftPacked[ropeSoftImpulseOffset] = impulseOffset;
        ropeSoftPacked[ropeSoftContactCount] = 0u;
        ropeSoftPacked[ropeSoftMaximumPenetration] = asuint(0.0f);
        ropeSoftPacked[ropeSoftAttachFirst] = constants.attach_first;
        ropeSoftPacked[ropeSoftAttachLast] = constants.attach_last;
        ropeSoftPacked[ropeSoftDistance] = asuint(constants.contact_distance);
        ropeSoftPacked[ropeSoftFriction] = asuint(constants.friction);
        ropeSoftPacked[ropeSoftOrientation] = asuint(constants.orientation);
        ropeSoftPacked[ropeSoftPriorContactCount] =
            ropeSoftDiagnosticContactCount[0];
    }
    for (uint surface = groupIndex; surface < constants.surface_count;
         surface += 64u)
    {
        ropeSoftStoreVec3(surfaceOffset + 3u * surface,
            metalLoad(ropeSoftSurfacePositions[surface]));
        ropeSoftStoreVec3(previousOffset + 3u * surface,
            metalLoad(ropeSoftPreviousSurface[surface]));
        MetalSurfaceBinding binding = ropeSoftSurfaceBindings[surface];
        uint base = bindingOffset + 8u * surface;
        for (uint slot = 0u; slot < 4u; ++slot)
        {
            ropeSoftPacked[base + slot] = binding.nodes[slot];
            ropeSoftPacked[base + 4u + slot] =
                asuint(binding.weights[slot]);
        }
    }
    for (uint node = groupIndex; node < constants.soft_count; node += 64u)
    {
        ropeSoftStoreVec3(velocityOffset + 3u * node,
            metalLoad(ropeSoftSoftVelocities[node]));
        ropeSoftPacked[inverseOffset + node] =
            asuint(ropeSoftSoftInverseMasses[node]);
        ropeSoftStoreVec3(impulseOffset + 3u * node, float3(0.0f));
    }
    for (uint index = groupIndex; index < constants.surface_index_count;
         index += 64u)
        ropeSoftPacked[indexOffset + index] = ropeSoftSurfaceIndices[index];
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rope_soft_sample(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint end = dispatchThreadId.x;
    MetalRopeSoftConstants constants = ropeSoftConstants;
    if (end >= 2u || constants.enabled == 0u) return;
    bool attached = end == 0u
        ? constants.attach_first != 0u : constants.attach_last != 0u;
    if (!attached) return;
    uint triangle = end == 0u
        ? constants.first_triangle : constants.last_triangle;
    if (triangle == 0xffffffffu ||
        triangle + 2u >= constants.surface_index_count)
        return;
    float3 weights = end == 0u
        ? metalLoad(constants.first_weights)
        : metalLoad(constants.last_weights);
    float3 position = end == 0u
        ? metalLoad(constants.first_offset)
        : metalLoad(constants.last_offset);
    float3 velocity = float3(0.0f);
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        float weight = weights[corner];
        uint surface = ropeSoftSurfaceIndices[triangle + corner];
        position += metalLoad(ropeSoftSurfacePositions[surface]) * weight;
        MetalSurfaceBinding binding = ropeSoftSurfaceBindings[surface];
        for (uint slot = 0u; slot < 4u; ++slot)
            velocity += metalLoad(
                ropeSoftSoftVelocities[binding.nodes[slot]]) *
                (weight * binding.weights[slot]);
    }
    MetalRopeAnchorState anchor;
    anchor.position = metalStore(position);
    anchor.velocity = metalStore(velocity);
    anchor.impulse = metalStore(float3(0.0f));
    anchor.inverse_mass = 0.0f;
    ropeSoftAnchors[end] = anchor;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rope_soft_anchor_weights(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u || ropeSoftPacked[ropeSoftValid] == 0u)
        return;
    MetalRopeSoftConstants constants = ropeSoftConstants;
    if (constants.enabled == 0u) return;
    uint contactCount = ropeSoftPacked[ropeSoftPriorContactCount] +
        ropeSoftPacked[ropeSoftContactCount];
    float broad = constants.anchor_support_radius_scale * constants.node_radius;
    float local = constants.anchor_contact_support_radius_scale *
        constants.node_radius;
    float support = local + (broad - local) /
        (1.0f + float(contactCount) * 0.5f);
    float supportSquared = support * support;
    for (uint end = 0u; end < 2u; ++end)
    {
        bool attached = end == 0u
            ? constants.attach_first != 0u : constants.attach_last != 0u;
        uint triangle = end == 0u
            ? constants.first_triangle : constants.last_triangle;
        float sums[256];
        for (uint lane = 0u; lane < 256u; ++lane)
        {
            float localSum = 0.0f;
            if (attached && triangle != 0xffffffffu &&
                triangle + 2u < constants.surface_index_count &&
                support > 1.0e-6f)
            {
                float3 anchor = metalLoad(ropeSoftAnchors[end].position);
                for (uint node = lane; node < constants.soft_count;
                     node += 256u)
                {
                    float3 delta =
                        metalLoad(ropeSoftSoftPositions[node]) - anchor;
                    float squared = dot(delta, delta);
                    if (squared >= supportSquared) continue;
                    float value = 1.0f - squared / supportSquared;
                    localSum += value * value;
                }
            }
            sums[lane] = localSum;
        }
        for (uint stride = 128u; stride != 0u; stride /= 2u)
            for (uint lane = 0u; lane < stride; ++lane)
                sums[lane] += sums[lane + stride];
        ropeSoftPacked[end == 0u
            ? ropeSoftFirstAnchorWeight : ropeSoftLastAnchorWeight] =
            asuint(sums[0]);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rope_soft_apply(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint node = dispatchThreadId.x;
    MetalRopeSoftConstants constants = ropeSoftConstants;
    if (ropeSoftPacked[ropeSoftValid] == 0u || constants.enabled == 0u ||
        node >= constants.soft_count)
        return;
    uint impulseOffset = ropeSoftPacked[ropeSoftImpulseOffset];
    uint contactCount = ropeSoftPacked[ropeSoftContactCount];
    uint frameContactCount =
        ropeSoftPacked[ropeSoftPriorContactCount] + contactCount;
    if (node == 0u)
    {
        ropeSoftDiagnosticContactCount[0] += contactCount;
        ropeSoftDiagnosticMaximumPenetration[0] = max(
            ropeSoftDiagnosticMaximumPenetration[0],
            ropeSoftLoadFloat(ropeSoftMaximumPenetration));
    }
    float3 impulse = ropeSoftLoadVec3(impulseOffset + 3u * node);
    for (uint end = 0u; end < 2u; ++end)
    {
        bool attached = end == 0u
            ? constants.attach_first != 0u : constants.attach_last != 0u;
        if (!attached) continue;
        uint triangle = end == 0u
            ? constants.first_triangle : constants.last_triangle;
        if (triangle == 0xffffffffu ||
            triangle + 2u >= constants.surface_index_count)
            continue;
        float3 anchor = metalLoad(ropeSoftAnchors[end].position);
        float3 reaction = metalLoad(ropeSoftAnchors[end].impulse);
        float broad =
            constants.anchor_support_radius_scale * constants.node_radius;
        float local = constants.anchor_contact_support_radius_scale *
            constants.node_radius;
        float support = local + (broad - local) /
            (1.0f + float(frameContactCount) * 0.5f);
        float supportSquared = support * support;
        float weightSum = ropeSoftLoadFloat(end == 0u
            ? ropeSoftFirstAnchorWeight : ropeSoftLastAnchorWeight);
        if (weightSum > 1.0e-8f)
        {
            float3 delta = metalLoad(ropeSoftSoftPositions[node]) - anchor;
            float squared = dot(delta, delta);
            if (squared < supportSquared)
            {
                float value = 1.0f - squared / supportSquared;
                impulse += reaction * (value * value / weightSum);
            }
            continue;
        }
        float3 anchorWeights = end == 0u
            ? metalLoad(constants.first_weights)
            : metalLoad(constants.last_weights);
        float cornerWeights[3];
        cornerWeights[0] = anchorWeights.x;
        cornerWeights[1] = anchorWeights.y;
        cornerWeights[2] = anchorWeights.z;
        float fallbackWeight = 0.0f;
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            MetalSurfaceBinding binding = ropeSoftSurfaceBindings[
                ropeSoftSurfaceIndices[triangle + corner]];
            for (uint slot = 0u; slot < 4u; ++slot)
                if (binding.nodes[slot] == node)
                    fallbackWeight +=
                        cornerWeights[corner] * binding.weights[slot];
        }
        impulse += reaction * fallbackWeight;
    }
    float inverse = ropeSoftSoftInverseMasses[node];
    if (inverse <= 0.0f) return;
    float3 change = metalLimit(impulse * inverse,
        constants.maximum_soft_acceleration * constants.timestep);
    ropeSoftSoftVelocities[node] = metalStore(metalLimit(
        metalLoad(ropeSoftSoftVelocities[node]) + change,
        constants.maximum_soft_speed));
    ropeSoftSoftPositions[node] = metalStore(
        metalLoad(ropeSoftSoftPositions[node]) + change * constants.timestep);
    ropeSoftSoftForces[node] = metalStore(
        metalLoad(ropeSoftSoftForces[node]) +
        change * (constants.frame_inverse_timestep / inverse));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rope_soft_surface_update(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint surface = dispatchThreadId.x;
    MetalRopeSoftConstants constants = ropeSoftConstants;
    if (ropeSoftPacked[ropeSoftValid] == 0u || constants.enabled == 0u ||
        surface >= constants.surface_count)
        return;
    MetalSurfaceBinding binding = ropeSoftSurfaceBindings[surface];
    float3 point = metalLoad(ropeSoftSurfaceRestPositions[surface]);
    for (uint slot = 0u; slot < 4u; ++slot)
    {
        uint node = binding.nodes[slot];
        point += (metalLoad(ropeSoftSoftPositions[node]) -
            metalLoad(ropeSoftSoftRestPositions[node])) *
            binding.weights[slot];
    }
    ropeSoftSurfacePositions[surface] = metalStore(point);
    ropeSoftPreviousSurface[surface] = metalStore(point);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rope_soft_update_endpoint(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint end = dispatchThreadId.x;
    MetalRopeSoftConstants constants = ropeSoftConstants;
    if (end >= 2u || ropeSoftPacked[ropeSoftValid] == 0u ||
        constants.enabled == 0u)
        return;
    bool attached = end == 0u
        ? constants.attach_first != 0u : constants.attach_last != 0u;
    if (!attached) return;
    uint triangle = end == 0u
        ? constants.first_triangle : constants.last_triangle;
    if (triangle == 0xffffffffu ||
        triangle + 2u >= constants.surface_index_count)
        return;
    float3 weights = end == 0u
        ? metalLoad(constants.first_weights)
        : metalLoad(constants.last_weights);
    float3 offset = end == 0u
        ? metalLoad(constants.first_offset)
        : metalLoad(constants.last_offset);
    float cornerWeights[3];
    cornerWeights[0] = weights.x;
    cornerWeights[1] = weights.y;
    cornerWeights[2] = weights.z;
    float3 position = offset;
    float3 velocity = float3(0.0f);
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        uint surface = ropeSoftSurfaceIndices[triangle + corner];
        position += metalLoad(ropeSoftSurfacePositions[surface]) *
            cornerWeights[corner];
        MetalSurfaceBinding binding = ropeSoftSurfaceBindings[surface];
        for (uint slot = 0u; slot < 4u; ++slot)
            velocity += metalLoad(
                ropeSoftSoftVelocities[binding.nodes[slot]]) *
                (cornerWeights[corner] * binding.weights[slot]);
    }
    MetalRopeAnchorState anchor;
    anchor.position = metalStore(position);
    anchor.velocity = metalStore(velocity);
    anchor.impulse = metalStore(float3(0.0f));
    anchor.inverse_mass = 0.0f;
    ropeSoftAnchors[end] = anchor;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void pm_rope_soft_finalize(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x == 0u) ropeSoftPacked[ropeSoftValid] = 0u;
}
