// SPDX-License-Identifier: MIT
// Presentation-only Zhu-Bridson field shared by native drawing adapters.
public struct SurfaceGrid
{
    public float3 minimum;
    public float3 cell_size;
    public uint3 dimensions;
    public float support_radius;
    public float particle_radius;
};

public float4 surfaceContribution(float3 relative, float support)
{
    float squared = dot(relative, relative);
    float radiusSquared = support * support;
    if (squared >= radiusSquared) return 0.0;
    float a = 1.0 - squared / radiusSquared;
    float weight = a * a * a;
    return float4(relative * weight, weight);
}

public float surfaceValue(float4 sum, float support, float radius)
{
    if (sum.w <= 1.0e-10) return support;
    float3 offset = sum.xyz / sum.w;
    float upper = support / 3.0;
    float lower = min(upper, radius * 0.8);
    float blend = clamp(offset.y / radius, 0.0, 1.0);
    float smooth = length(offset) - (lower + (upper - lower) * blend);
    // Preserve the existing surface's lower-lobe and terrain clearance rules.
    return max(smooth, -offset.y - radius * 0.9) + 0.002;
}

public uint64_t surfaceCellKey(int3 cell)
{
    const int bias = 1 << 20;
    cell = clamp(cell, int3(-bias), int3(bias - 1));
    return (uint64_t(cell.x + bias) << 42) |
           (uint64_t(cell.y + bias) << 21) | uint64_t(cell.z + bias);
}
