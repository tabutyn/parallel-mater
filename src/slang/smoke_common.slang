// SPDX-License-Identifier: MIT

import metal_common;

struct MetalSmokeMetadata
{
    uint count;
    uint next_particle;
    float emission_remainder;
    uint revision;
    uint64_t emitted;
};

struct MetalSmokeConstants
{
    float timestep;
    MetalPackedVec3 gravity;
    MetalPackedVec3 emitter_center;
    MetalPackedVec3 initial_velocity;
    MetalPackedVec3 wind;
    MetalPackedVec2 emitter_half_extents;
    uint command;
    float lifetime;
    float particle_radius;
    float buoyancy;
    float response;
    float maximum_speed;
    float rest_number_density;
    float pressure_stiffness;
    float viscosity;
    float vorticity_confinement;
    uint capacity;
    uint grid_resolution;
    uint grid_vertical_resolution;
    uint grid_pressure_iterations;
    MetalPackedVec3 grid_minimum;
    float grid_spacing;
    float grid_kinematic_viscosity;
    float grid_les_coefficient;
    float grid_pressure_tolerance;
};

[ForceInline]
static uint metalSmokeFaceCount(uint axis, uint n, uint height)
{
    if (axis == 0u) return (n + 1u) * height * n;
    if (axis == 1u) return n * (height + 1u) * n;
    return n * height * (n + 1u);
}

[ForceInline]
static uint metalSmokeFaceOffset(uint axis, uint n, uint height)
{
    if (axis == 0u) return 0u;
    uint xCount = metalSmokeFaceCount(0u, n, height);
    return axis == 1u ? xCount :
        xCount + metalSmokeFaceCount(1u, n, height);
}

[ForceInline]
static uint metalSmokeFaceIndex(
    uint axis, int x, int y, int z, uint n, uint height)
{
    uint local = axis == 0u
        ? uint(x) + (n + 1u) * (uint(y) + height * uint(z))
        : axis == 1u
            ? uint(x) + n * (uint(y) + (height + 1u) * uint(z))
            : uint(x) + n * (uint(y) + height * uint(z));
    return metalSmokeFaceOffset(axis, n, height) + local;
}

[ForceInline]
static float metalSmokeSampleFaceComponent(
    RWStructuredBuffer<float> values, uint baseOffset, uint axis,
    float3 point, float3 minimum, float spacing, uint n, uint height)
{
    float3 p = (point - minimum) / spacing;
    int sx = int(n);
    int sy = int(height);
    int sz = int(n);
    if (axis == 0u) { p.y -= 0.5f; p.z -= 0.5f; ++sx; }
    if (axis == 1u) { p.x -= 0.5f; p.z -= 0.5f; ++sy; }
    if (axis == 2u) { p.x -= 0.5f; p.y -= 0.5f; ++sz; }
    int3 lower = int3(floor(p));
    float3 fraction = p - float3(lower);
    float result = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int x = clamp(lower.x + dx, 0, sx - 1);
                int y = clamp(lower.y + dy, 0, sy - 1);
                int z = clamp(lower.z + dz, 0, sz - 1);
                float weight =
                    (dx != 0 ? fraction.x : 1.0f - fraction.x) *
                    (dy != 0 ? fraction.y : 1.0f - fraction.y) *
                    (dz != 0 ? fraction.z : 1.0f - fraction.z);
                result += values[baseOffset + metalSmokeFaceIndex(
                    axis, x, y, z, n, height)] * weight;
            }
    return result;
}

[ForceInline]
static float3 metalSmokeSampleFaceVelocity(
    RWStructuredBuffer<float> values, uint baseOffset, float3 point,
    float3 minimum, float spacing, uint n, uint height)
{
    return float3(
        metalSmokeSampleFaceComponent(
            values, baseOffset, 0u, point, minimum, spacing, n, height),
        metalSmokeSampleFaceComponent(
            values, baseOffset, 1u, point, minimum, spacing, n, height),
        metalSmokeSampleFaceComponent(
            values, baseOffset, 2u, point, minimum, spacing, n, height));
}

[ForceInline]
static float metalSmokeSampleFaceStrain(
    RWStructuredBuffer<float> values, uint baseOffset, float3 point,
    float3 minimum, float spacing, uint n, uint height)
{
    float3 velocityXMinus = metalSmokeSampleFaceVelocity(
        values, baseOffset, point - float3(spacing, 0.0f, 0.0f),
        minimum, spacing, n, height);
    float3 velocityXPlus = metalSmokeSampleFaceVelocity(
        values, baseOffset, point + float3(spacing, 0.0f, 0.0f),
        minimum, spacing, n, height);
    float3 velocityYMinus = metalSmokeSampleFaceVelocity(
        values, baseOffset, point - float3(0.0f, spacing, 0.0f),
        minimum, spacing, n, height);
    float3 velocityYPlus = metalSmokeSampleFaceVelocity(
        values, baseOffset, point + float3(0.0f, spacing, 0.0f),
        minimum, spacing, n, height);
    float3 velocityZMinus = metalSmokeSampleFaceVelocity(
        values, baseOffset, point - float3(0.0f, 0.0f, spacing),
        minimum, spacing, n, height);
    float3 velocityZPlus = metalSmokeSampleFaceVelocity(
        values, baseOffset, point + float3(0.0f, 0.0f, spacing),
        minimum, spacing, n, height);
    float inverseTwoSpacing = 0.5f / spacing;
    float3 dx = (velocityXPlus - velocityXMinus) * inverseTwoSpacing;
    float3 dy = (velocityYPlus - velocityYMinus) * inverseTwoSpacing;
    float3 dz = (velocityZPlus - velocityZMinus) * inverseTwoSpacing;
    float sxy = 0.5f * (dx.y + dy.x);
    float sxz = 0.5f * (dx.z + dz.x);
    float syz = 0.5f * (dy.z + dz.y);
    return sqrt(max(0.0f, 2.0f *
        (dx.x * dx.x + dy.y * dy.y + dz.z * dz.z +
         2.0f * (sxy * sxy + sxz * sxz + syz * syz))));
}

[ForceInline]
static float metalSmokeSampleCellScalar(
    RWStructuredBuffer<float> values, float3 point, float3 minimum,
    float spacing, uint n, uint height)
{
    float3 p = (point - minimum) / spacing - 0.5f;
    int3 lower = int3(floor(p));
    float3 fraction = p - float3(lower);
    float result = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int x = clamp(lower.x + dx, 0, int(n) - 1);
                int y = clamp(lower.y + dy, 0, int(height) - 1);
                int z = clamp(lower.z + dz, 0, int(n) - 1);
                float weight =
                    (dx != 0 ? fraction.x : 1.0f - fraction.x) *
                    (dy != 0 ? fraction.y : 1.0f - fraction.y) *
                    (dz != 0 ? fraction.z : 1.0f - fraction.z);
                uint cell = uint(x) + n * (uint(y) + height * uint(z));
                result += values[cell] * weight;
            }
    return result;
}

[ForceInline]
static float3 metalSmokeSampleCellVector(
    RWStructuredBuffer<MetalPackedVec3> values, float3 point, float3 minimum,
    float spacing, uint n, uint height)
{
    float3 p = (point - minimum) / spacing - 0.5f;
    int3 lower = int3(floor(p));
    float3 fraction = p - float3(lower);
    float3 result = float3(0.0f);
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int x = clamp(lower.x + dx, 0, int(n) - 1);
                int y = clamp(lower.y + dy, 0, int(height) - 1);
                int z = clamp(lower.z + dz, 0, int(n) - 1);
                float weight =
                    (dx != 0 ? fraction.x : 1.0f - fraction.x) *
                    (dy != 0 ? fraction.y : 1.0f - fraction.y) *
                    (dz != 0 ? fraction.z : 1.0f - fraction.z);
                uint cell = uint(x) + n * (uint(y) + height * uint(z));
                result += metalLoad(values[cell]) * weight;
            }
    return result;
}
