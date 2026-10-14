// SPDX-License-Identifier: MIT
// Production smoke pressure-grid kernels for Metal.

import smoke_common;
import metal_common;

[[vk::binding(0, 0)]] ConstantBuffer<MetalSmokeConstants>
    pressureConstants : register(b0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float>
    pressureValues : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<float>
    pressureDivergence : register(u2);
[[vk::binding(5, 0)]] RWStructuredBuffer<float>
    pressureRelativeResidual : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<uint>
    pressureState : register(u6);
[[vk::binding(7, 0)]] RWStructuredBuffer<float>
    pressureFaceVelocity : register(u7);
[[vk::binding(8, 0)]] RWStructuredBuffer<MetalPackedVec3>
    pressureGridVelocity : register(u8);
[[vk::binding(9, 0)]] RWStructuredBuffer<MetalPackedVec3>
    pressureGridVorticity : register(u9);
[[vk::binding(10, 0)]] RWStructuredBuffer<float>
    pressureGridDensity : register(u10);
[[vk::binding(11, 0)]] RWStructuredBuffer<float>
    pressureGridTemperature : register(u11);
[[vk::binding(12, 0)]] RWStructuredBuffer<MetalPackedVec3>
    pressureGridScratch : register(u12);
[[vk::binding(3, 0)]] RWStructuredBuffer<float>
    pressureFaceBoundary : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<float>
    pressureScratch : register(u4);

[ForceInline]
static uint pressureFaceCount(uint axis, uint n, uint height)
{
    if (axis == 0u) return (n + 1u) * height * n;
    if (axis == 1u) return n * (height + 1u) * n;
    return n * height * (n + 1u);
}

[ForceInline]
static uint pressureFaceTotal(uint n, uint height)
{
    return pressureFaceCount(0u, n, height) +
        pressureFaceCount(1u, n, height) +
        pressureFaceCount(2u, n, height);
}

[ForceInline]
static uint pressureFaceOffset(uint axis, uint n, uint height)
{
    if (axis == 0u) return 0u;
    uint xCount = pressureFaceCount(0u, n, height);
    return axis == 1u ? xCount :
        xCount + pressureFaceCount(1u, n, height);
}

[ForceInline]
static uint pressureFaceIndex(
    uint axis, int x, int y, int z, uint n, uint height)
{
    uint local = axis == 0u
        ? uint(x) + (n + 1u) * (uint(y) + height * uint(z))
        : axis == 1u
            ? uint(x) + n * (uint(y) + (height + 1u) * uint(z))
            : uint(x) + n * (uint(y) + height * uint(z));
    return pressureFaceOffset(axis, n, height) + local;
}

[ForceInline]
static float3 pressureFacePosition(
    uint axis, int x, int y, int z, float3 minimum, float spacing)
{
    float3 offset = float3(
        float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
    if (axis == 0u) offset.x = float(x);
    if (axis == 1u) offset.y = float(y);
    if (axis == 2u) offset.z = float(z);
    return minimum + offset * spacing;
}

[ForceInline]
static float pressureVorticityMagnitude(
    int x, int y, int z, uint n, uint height)
{
    x = clamp(x, 0, int(n) - 1);
    y = clamp(y, 0, int(height) - 1);
    z = clamp(z, 0, int(n) - 1);
    return length(metalLoad(pressureGridVorticity[
        uint(x) + n * (uint(y) + height * uint(z))]));
}

struct PressureCellDiagnostics
{
    float3 velocity;
    float3 vorticity;
    float strain;
    float divergence;
};

[ForceInline]
static float pressureFaceSample(
    RWStructuredBuffer<float> values, uint axis,
    int x, int y, int z, uint n, uint height)
{
    int sx = int(n) + (axis == 0u ? 1 : 0);
    int sy = int(height) + (axis == 1u ? 1 : 0);
    int sz = int(n) + (axis == 2u ? 1 : 0);
    return values[pressureFaceIndex(axis,
        clamp(x, 0, sx - 1), clamp(y, 0, sy - 1),
        clamp(z, 0, sz - 1), n, height)];
}

[ForceInline]
static PressureCellDiagnostics pressureCellDiagnostics(
    RWStructuredBuffer<float> faces,
    uint n, uint height, float spacing, uint cell)
{
    int x = int(cell % n);
    int y = int(cell / n % height);
    int z = int(cell / (n * height));
    float u0 = pressureFaceSample(faces, 0u, x, y, z, n, height);
    float u1 = pressureFaceSample(faces, 0u, x + 1, y, z, n, height);
    float v0 = pressureFaceSample(faces, 1u, x, y, z, n, height);
    float v1 = pressureFaceSample(faces, 1u, x, y + 1, z, n, height);
    float w0 = pressureFaceSample(faces, 2u, x, y, z, n, height);
    float w1 = pressureFaceSample(faces, 2u, x, y, z + 1, n, height);
    float inverse = 1.0f / spacing;
    float transverse = 0.25f * inverse;
    float dux = (u1 - u0) * inverse;
    float duy = (pressureFaceSample(faces, 0u, x, y + 1, z, n, height) +
        pressureFaceSample(faces, 0u, x + 1, y + 1, z, n, height) -
        pressureFaceSample(faces, 0u, x, y - 1, z, n, height) -
        pressureFaceSample(faces, 0u, x + 1, y - 1, z, n, height)) *
        transverse;
    float duz = (pressureFaceSample(faces, 0u, x, y, z + 1, n, height) +
        pressureFaceSample(faces, 0u, x + 1, y, z + 1, n, height) -
        pressureFaceSample(faces, 0u, x, y, z - 1, n, height) -
        pressureFaceSample(faces, 0u, x + 1, y, z - 1, n, height)) *
        transverse;
    float dvx = (pressureFaceSample(faces, 1u, x + 1, y, z, n, height) +
        pressureFaceSample(faces, 1u, x + 1, y + 1, z, n, height) -
        pressureFaceSample(faces, 1u, x - 1, y, z, n, height) -
        pressureFaceSample(faces, 1u, x - 1, y + 1, z, n, height)) *
        transverse;
    float dvy = (v1 - v0) * inverse;
    float dvz = (pressureFaceSample(faces, 1u, x, y, z + 1, n, height) +
        pressureFaceSample(faces, 1u, x, y + 1, z + 1, n, height) -
        pressureFaceSample(faces, 1u, x, y, z - 1, n, height) -
        pressureFaceSample(faces, 1u, x, y + 1, z - 1, n, height)) *
        transverse;
    float dwx = (pressureFaceSample(faces, 2u, x + 1, y, z, n, height) +
        pressureFaceSample(faces, 2u, x + 1, y, z + 1, n, height) -
        pressureFaceSample(faces, 2u, x - 1, y, z, n, height) -
        pressureFaceSample(faces, 2u, x - 1, y, z + 1, n, height)) *
        transverse;
    float dwy = (pressureFaceSample(faces, 2u, x, y + 1, z, n, height) +
        pressureFaceSample(faces, 2u, x, y + 1, z + 1, n, height) -
        pressureFaceSample(faces, 2u, x, y - 1, z, n, height) -
        pressureFaceSample(faces, 2u, x, y - 1, z + 1, n, height)) *
        transverse;
    float dwz = (w1 - w0) * inverse;
    float sxy = 0.5f * (duy + dvx);
    float sxz = 0.5f * (duz + dwx);
    float syz = 0.5f * (dvz + dwy);
    PressureCellDiagnostics result;
    result.velocity = float3(
        0.5f * (u0 + u1), 0.5f * (v0 + v1), 0.5f * (w0 + w1));
    result.vorticity = float3(dwy - dvz, duz - dwx, dvx - duy);
    result.strain = sqrt(max(0.0f, 2.0f *
        (dux * dux + dvy * dvy + dwz * dwz +
         2.0f * (sxy * sxy + sxz * sxz + syz * syz))));
    result.divergence = dux + dvy + dwz;
    return result;
}

[ForceInline]
static void pressureFaceCoordinates(
    uint axis, uint local, uint n, uint height,
    out int x, out int y, out int z)
{
    if (axis == 0u)
    {
        x = int(local % (n + 1u));
        y = int(local / (n + 1u) % height);
        z = int(local / ((n + 1u) * height));
    }
    else if (axis == 1u)
    {
        x = int(local % n);
        y = int(local / n % (height + 1u));
        z = int(local / (n * (height + 1u)));
    }
    else
    {
        x = int(local % n);
        y = int(local / n % height);
        z = int(local / (n * height));
    }
}

[ForceInline]
static bool pressureFlatFaceCoordinates(
    uint face, uint n, uint height,
    out uint axis, out uint local, out int x, out int y, out int z)
{
    axis = 0u;
    local = 0u;
    x = 0;
    y = 0;
    z = 0;
    uint xCount = pressureFaceCount(0u, n, height);
    uint yCount = pressureFaceCount(1u, n, height);
    if (face < xCount)
    {
        axis = 0u;
        local = face;
    }
    else if (face < xCount + yCount)
    {
        axis = 1u;
        local = face - xCount;
    }
    else if (face < pressureFaceTotal(n, height))
    {
        axis = 2u;
        local = face - xCount - yCount;
    }
    else return false;
    pressureFaceCoordinates(axis, local, n, height, x, y, z);
    return true;
}

[ForceInline]
static float3 pressureNormalizedOr(float3 value, float3 fallback)
{
    float squared = dot(value, value);
    return squared > 1.0e-12f ? value * rsqrt(squared) : fallback;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_advect_forward(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint face = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    uint axis = 0u;
    uint local = 0u;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!pressureFlatFaceCoordinates(
        face, n, height, axis, local, x, y, z)) return;
    float3 minimum = metalLoad(constants.grid_minimum);
    float spacing = constants.grid_spacing;
    float3 point = pressureFacePosition(axis, x, y, z, minimum, spacing);
    float3 first = metalSmokeSampleFaceVelocity(
        pressureFaceVelocity, 0u, point, minimum, spacing, n, height);
    float3 midpoint = point - first * (0.5f * constants.timestep);
    float3 flow = metalSmokeSampleFaceVelocity(
        pressureFaceVelocity, 0u, midpoint, minimum, spacing, n, height);
    float3 departure = point - flow * constants.timestep;
    pressureScratch[face] = metalSmokeSampleFaceComponent(
        pressureFaceVelocity, 0u, axis, departure,
        minimum, spacing, n, height);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_advect_reverse(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint face = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    uint axis = 0u;
    uint local = 0u;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!pressureFlatFaceCoordinates(
        face, n, height, axis, local, x, y, z)) return;
    uint faceTotal = pressureFaceTotal(n, height);
    float3 minimum = metalLoad(constants.grid_minimum);
    float spacing = constants.grid_spacing;
    float3 point = pressureFacePosition(axis, x, y, z, minimum, spacing);
    float3 first = metalSmokeSampleFaceVelocity(
        pressureScratch, 0u, point, minimum, spacing, n, height);
    float3 midpoint = point + first * (0.5f * constants.timestep);
    float3 flow = metalSmokeSampleFaceVelocity(
        pressureScratch, 0u, midpoint, minimum, spacing, n, height);
    pressureScratch[faceTotal + face] = metalSmokeSampleFaceComponent(
        pressureScratch, 0u, axis, point + flow * constants.timestep,
        minimum, spacing, n, height);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_correct_face(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint face = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    uint axis = 0u;
    uint local = 0u;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!pressureFlatFaceCoordinates(
        face, n, height, axis, local, x, y, z)) return;
    uint faceTotal = pressureFaceTotal(n, height);
    float3 minimum = metalLoad(constants.grid_minimum);
    float spacing = constants.grid_spacing;
    float3 point = pressureFacePosition(axis, x, y, z, minimum, spacing);
    float3 first = metalSmokeSampleFaceVelocity(
        pressureFaceVelocity, 0u, point, minimum, spacing, n, height);
    float3 midpoint = point - first * (0.5f * constants.timestep);
    float3 flow = metalSmokeSampleFaceVelocity(
        pressureFaceVelocity, 0u, midpoint, minimum, spacing, n, height);
    float3 samplePosition =
        (point - flow * constants.timestep - minimum) / spacing;
    int sx = int(n);
    int sy = int(height);
    int sz = int(n);
    if (axis == 0u)
    {
        samplePosition.y -= 0.5f;
        samplePosition.z -= 0.5f;
        ++sx;
    }
    if (axis == 1u)
    {
        samplePosition.x -= 0.5f;
        samplePosition.z -= 0.5f;
        ++sy;
    }
    if (axis == 2u)
    {
        samplePosition.x -= 0.5f;
        samplePosition.y -= 0.5f;
        ++sz;
    }
    int3 lower = int3(floor(samplePosition));
    float low = 3.402823466e+38f;
    float high = -3.402823466e+38f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int xx = clamp(lower.x + dx, 0, sx - 1);
                int yy = clamp(lower.y + dy, 0, sy - 1);
                int zz = clamp(lower.z + dz, 0, sz - 1);
                float value = pressureFaceVelocity[
                    pressureFaceIndex(axis, xx, yy, zz, n, height)];
                low = min(low, value);
                high = max(high, value);
            }
    pressureScratch[face] = clamp(
        pressureScratch[face] + 0.5f *
            (pressureFaceVelocity[face] - pressureScratch[faceTotal + face]),
        low, high);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_apply_face_forces(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint face = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    uint axis = 0u;
    uint local = 0u;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!pressureFlatFaceCoordinates(
        face, n, height, axis, local, x, y, z)) return;
    uint faceTotal = pressureFaceTotal(n, height);
    if (pressureFaceBoundary[face] <= 1.0e-4f)
    {
        pressureScratch[faceTotal + face] =
            pressureFaceBoundary[faceTotal + face];
        return;
    }
    float center = pressureScratch[face];
    float laplacian =
        (pressureFaceSample(
            pressureScratch, axis, x - 1, y, z, n, height) +
         pressureFaceSample(
            pressureScratch, axis, x + 1, y, z, n, height) +
         pressureFaceSample(
            pressureScratch, axis, x, y - 1, z, n, height) +
         pressureFaceSample(
            pressureScratch, axis, x, y + 1, z, n, height) +
         pressureFaceSample(
            pressureScratch, axis, x, y, z - 1, n, height) +
         pressureFaceSample(
            pressureScratch, axis, x, y, z + 1, n, height) -
         6.0f * center) /
        (constants.grid_spacing * constants.grid_spacing);
    float3 minimum = metalLoad(constants.grid_minimum);
    float3 point = pressureFacePosition(
        axis, x, y, z, minimum, constants.grid_spacing);
    float strain = metalSmokeSampleCellScalar(
        pressureDivergence, point, minimum,
        constants.grid_spacing, n, height);
    float viscosity = constants.grid_kinematic_viscosity +
        constants.grid_les_coefficient * constants.grid_les_coefficient *
            constants.grid_spacing * constants.grid_spacing * strain;
    float3 curlForce = metalSmokeSampleCellVector(
        pressureGridScratch, point, minimum,
        constants.grid_spacing, n, height);
    float density = metalSmokeSampleCellScalar(
        pressureGridDensity, point, minimum,
        constants.grid_spacing, n, height);
    float thermalLoading = metalSmokeSampleCellScalar(
        pressureGridTemperature, point, minimum,
        constants.grid_spacing, n, height);
    float heat = density > 1.0e-5f
        ? clamp(thermalLoading / density, 0.0f, constants.maximum_speed)
        : 0.0f;
    float3 up = pressureNormalizedOr(
        -metalLoad(constants.gravity), float3(0.0f, 1.0f, 0.0f));
    float3 buoyancy = up *
        (constants.buoyancy * density * constants.rest_number_density +
         2.0f * heat);
    float body = axis == 0u ? curlForce.x + buoyancy.x :
        axis == 1u ? curlForce.y + buoyancy.y : curlForce.z + buoyancy.z;
    pressureScratch[faceTotal + face] = clamp(
        center + constants.timestep * (viscosity * laplacian + body),
        -constants.maximum_speed, constants.maximum_speed);
}

struct PressureLayout
{
    uint n[4];
    uint height[4];
    uint cells[4];
    uint faces[4];
    uint primaryOffset[4];
    uint alternateOffset[4];
    uint rhsOffset[4];
    uint residualOffset[4];
    uint openOffset[4];
};

[ForceInline]
static PressureLayout pressureLayout(MetalSmokeConstants constants)
{
    PressureLayout layout = {};
    layout.n[0] = constants.grid_resolution;
    layout.height[0] = constants.grid_vertical_resolution;
    for (uint level = 1u; level < 4u; ++level)
    {
        layout.n[level] = max(2u, layout.n[level - 1u] / 2u);
        layout.height[level] = max(2u, layout.height[level - 1u] / 2u);
    }
    for (uint level = 0u; level < 4u; ++level)
    {
        layout.cells[level] =
            layout.n[level] * layout.height[level] * layout.n[level];
        layout.faces[level] =
            pressureFaceTotal(layout.n[level], layout.height[level]);
    }
    uint cursor = 0u;
    layout.alternateOffset[0] = cursor;
    cursor += layout.cells[0];
    layout.residualOffset[0] = cursor;
    cursor += layout.cells[0];
    for (uint level = 1u; level < 3u; ++level)
    {
        layout.primaryOffset[level] = cursor;
        cursor += layout.cells[level];
        layout.alternateOffset[level] = cursor;
        cursor += layout.cells[level];
        layout.rhsOffset[level] = cursor;
        cursor += layout.cells[level];
        layout.residualOffset[level] = cursor;
        cursor += layout.cells[level];
        layout.openOffset[level] = cursor;
        cursor += layout.faces[level];
    }
    layout.primaryOffset[3] = cursor;
    cursor += layout.cells[3];
    layout.alternateOffset[3] = cursor;
    cursor += layout.cells[3];
    layout.rhsOffset[3] = cursor;
    cursor += layout.cells[3];
    layout.openOffset[3] = cursor;
    return layout;
}

[ForceInline]
static float pressurePrimaryLoad(
    PressureLayout layout, uint level, uint index)
{
    return level == 0u ? pressureValues[index] :
        pressureScratch[layout.primaryOffset[level] + index];
}

[ForceInline]
static void pressurePrimaryStore(
    PressureLayout layout, uint level, uint index, float value)
{
    if (level == 0u) pressureValues[index] = value;
    else pressureScratch[layout.primaryOffset[level] + index] = value;
}

[ForceInline]
static float pressureAlternateLoad(
    PressureLayout layout, uint level, uint index)
{
    return pressureScratch[layout.alternateOffset[level] + index];
}

[ForceInline]
static void pressureAlternateStore(
    PressureLayout layout, uint level, uint index, float value)
{
    pressureScratch[layout.alternateOffset[level] + index] = value;
}

[ForceInline]
static float pressureRhsLoad(
    PressureLayout layout, uint level, uint index)
{
    return level == 0u ? pressureDivergence[index] :
        pressureScratch[layout.rhsOffset[level] + index];
}

[ForceInline]
static void pressureRhsStore(
    PressureLayout layout, uint level, uint index, float value)
{
    if (level == 0u) pressureDivergence[index] = value;
    else pressureScratch[layout.rhsOffset[level] + index] = value;
}

[ForceInline]
static float pressureResidualLoad(
    PressureLayout layout, uint level, uint index)
{
    return pressureScratch[layout.residualOffset[level] + index];
}

[ForceInline]
static void pressureResidualStore(
    PressureLayout layout, uint level, uint index, float value)
{
    pressureScratch[layout.residualOffset[level] + index] = value;
}

[ForceInline]
static float pressureOpenLoad(
    PressureLayout layout, uint level, uint index)
{
    return level == 0u ? pressureFaceBoundary[index] :
        pressureScratch[layout.openOffset[level] + index];
}

[ForceInline]
static void pressureOpenStore(
    PressureLayout layout, uint level, uint index, float value)
{
    if (level == 0u) pressureFaceBoundary[index] = value;
    else pressureScratch[layout.openOffset[level] + index] = value;
}

[ForceInline]
static float pressureOpenCoefficient(
    PressureLayout layout, uint level, uint direction,
    int x, int y, int z)
{
    uint n = layout.n[level];
    uint height = layout.height[level];
    if (direction == 0u)
        return pressureOpenLoad(layout, level,
            pressureFaceIndex(0u, x, y, z, n, height));
    if (direction == 1u)
        return pressureOpenLoad(layout, level,
            pressureFaceIndex(0u, x + 1, y, z, n, height));
    if (direction == 2u)
        return pressureOpenLoad(layout, level,
            pressureFaceIndex(1u, x, y, z, n, height));
    if (direction == 3u)
        return pressureOpenLoad(layout, level,
            pressureFaceIndex(1u, x, y + 1, z, n, height));
    if (direction == 4u)
        return pressureOpenLoad(layout, level,
            pressureFaceIndex(2u, x, y, z, n, height));
    return pressureOpenLoad(layout, level,
        pressureFaceIndex(2u, x, y, z + 1, n, height));
}

[ForceInline]
static void pressureNeighbor(
    uint direction, int x, int y, int z,
    out int xx, out int yy, out int zz)
{
    xx = x;
    yy = y;
    zz = z;
    if (direction == 0u) --xx;
    else if (direction == 1u) ++xx;
    else if (direction == 2u) --yy;
    else if (direction == 3u) ++yy;
    else if (direction == 4u) --zz;
    else ++zz;
}

[ForceInline]
static bool pressureConverged()
{
    return pressureState[2] != 0u;
}

[ForceInline]
static void pressureRestrictOpenLevel(uint coarseLevel, uint face)
{
    if (pressureConverged()) return;
    PressureLayout layout = pressureLayout(pressureConstants);
    if (face >= layout.faces[coarseLevel]) return;
    uint fineLevel = coarseLevel - 1u;
    uint xFaces = pressureFaceCount(
        0u, layout.n[coarseLevel], layout.height[coarseLevel]);
    uint yFaces = pressureFaceCount(
        1u, layout.n[coarseLevel], layout.height[coarseLevel]);
    uint axis = 0u;
    uint local = face;
    if (local >= xFaces)
    {
        local -= xFaces;
        axis = 1u;
    }
    if (axis == 1u && local >= yFaces)
    {
        local -= yFaces;
        axis = 2u;
    }
    int x = 0;
    int y = 0;
    int z = 0;
    pressureFaceCoordinates(axis, local,
        layout.n[coarseLevel], layout.height[coarseLevel], x, y, z);
    float sum = 0.0f;
    for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a)
        {
            int xx = 2 * x;
            int yy = 2 * y;
            int zz = 2 * z;
            if (axis == 0u) { yy += a; zz += b; }
            if (axis == 1u) { xx += a; zz += b; }
            if (axis == 2u) { xx += a; yy += b; }
            int sx = int(layout.n[fineLevel]) + (axis == 0u ? 1 : 0);
            int sy = int(layout.height[fineLevel]) + (axis == 1u ? 1 : 0);
            int sz = int(layout.n[fineLevel]) + (axis == 2u ? 1 : 0);
            xx = min(sx - 1, xx);
            yy = min(sy - 1, yy);
            zz = min(sz - 1, zz);
            sum += pressureOpenLoad(layout, fineLevel, pressureFaceIndex(
                axis, xx, yy, zz,
                layout.n[fineLevel], layout.height[fineLevel]));
        }
    pressureOpenStore(layout, coarseLevel, face, 0.25f * sum);
}

[ForceInline]
static void pressureSmoothLevel(uint level, bool backward, uint cell)
{
    if (pressureConverged()) return;
    MetalSmokeConstants constants = pressureConstants;
    PressureLayout layout = pressureLayout(constants);
    if (cell >= layout.cells[level]) return;
    uint n = layout.n[level];
    uint height = layout.height[level];
    int x = int(cell % n);
    int y = int(cell / n % height);
    int z = int(cell / (n * height));
    float sum = 0.0f;
    float diagonal = 0.0f;
    for (uint direction = 0u; direction < 6u; ++direction)
    {
        float coefficient = pressureOpenCoefficient(
            layout, level, direction, x, y, z);
        if (coefficient <= 1.0e-5f) continue;
        int xx;
        int yy;
        int zz;
        pressureNeighbor(direction, x, y, z, xx, yy, zz);
        if (xx >= 0 && yy >= 0 && zz >= 0 &&
            xx < int(n) && yy < int(height) && zz < int(n))
        {
            uint neighbor = uint(xx) + n * (uint(yy) + height * uint(zz));
            float value = backward
                ? pressureAlternateLoad(layout, level, neighbor)
                : pressurePrimaryLoad(layout, level, neighbor);
            sum += coefficient * value;
        }
        diagonal += coefficient;
    }
    float current = backward
        ? pressureAlternateLoad(layout, level, cell)
        : pressurePrimaryLoad(layout, level, cell);
    float spacing = constants.grid_spacing * float(1u << level);
    float jacobi = diagonal > 1.0e-6f
        ? (sum - spacing * spacing * pressureRhsLoad(layout, level, cell)) /
            diagonal
        : 0.0f;
    float next = current + (2.0f / 3.0f) * (jacobi - current);
    if (backward) pressurePrimaryStore(layout, level, cell, next);
    else pressureAlternateStore(layout, level, cell, next);
}

[ForceInline]
static void pressureResidualLevel(uint level, bool trackMaximum, uint cell)
{
    if (pressureConverged()) return;
    MetalSmokeConstants constants = pressureConstants;
    PressureLayout layout = pressureLayout(constants);
    if (cell >= layout.cells[level]) return;
    uint n = layout.n[level];
    uint height = layout.height[level];
    int x = int(cell % n);
    int y = int(cell / n % height);
    int z = int(cell / (n * height));
    float sum = 0.0f;
    float diagonal = 0.0f;
    for (uint direction = 0u; direction < 6u; ++direction)
    {
        float coefficient = pressureOpenCoefficient(
            layout, level, direction, x, y, z);
        if (coefficient <= 1.0e-5f) continue;
        int xx;
        int yy;
        int zz;
        pressureNeighbor(direction, x, y, z, xx, yy, zz);
        if (xx >= 0 && yy >= 0 && zz >= 0 &&
            xx < int(n) && yy < int(height) && zz < int(n))
        {
            uint neighbor = uint(xx) + n * (uint(yy) + height * uint(zz));
            sum += coefficient * pressurePrimaryLoad(layout, level, neighbor);
        }
        diagonal += coefficient;
    }
    float spacing = constants.grid_spacing * float(1u << level);
    float value = pressureRhsLoad(layout, level, cell) -
        (sum - diagonal * pressurePrimaryLoad(layout, level, cell)) /
            (spacing * spacing);
    pressureResidualStore(layout, level, cell, value);
    if (trackMaximum)
        InterlockedMax(pressureState[1], asuint(abs(value)));
}

[ForceInline]
static void pressureRestrictResidualLevel(uint fineLevel, uint cell)
{
    if (pressureConverged()) return;
    PressureLayout layout = pressureLayout(pressureConstants);
    uint coarseLevel = fineLevel + 1u;
    if (cell >= layout.cells[coarseLevel]) return;
    uint coarseN = layout.n[coarseLevel];
    uint coarseHeight = layout.height[coarseLevel];
    int x = int(cell % coarseN);
    int y = int(cell / coarseN % coarseHeight);
    int z = int(cell / (coarseN * coarseHeight));
    float sum = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int xx = min(int(layout.n[fineLevel]) - 1, 2 * x + dx);
                int yy = min(
                    int(layout.height[fineLevel]) - 1, 2 * y + dy);
                int zz = min(int(layout.n[fineLevel]) - 1, 2 * z + dz);
                uint fineCell = uint(xx) + layout.n[fineLevel] *
                    (uint(yy) + layout.height[fineLevel] * uint(zz));
                sum += pressureResidualLoad(layout, fineLevel, fineCell);
            }
    pressureRhsStore(layout, coarseLevel, cell, 0.125f * sum);
}

[ForceInline]
static void pressureClearLevel(uint level, uint cell)
{
    if (pressureConverged()) return;
    PressureLayout layout = pressureLayout(pressureConstants);
    if (cell >= layout.cells[level]) return;
    pressurePrimaryStore(layout, level, cell, 0.0f);
    pressureAlternateStore(layout, level, cell, 0.0f);
}

[ForceInline]
static void pressureProlongLevel(uint coarseLevel, uint cell)
{
    if (pressureConverged()) return;
    PressureLayout layout = pressureLayout(pressureConstants);
    uint fineLevel = coarseLevel - 1u;
    if (cell >= layout.cells[fineLevel]) return;
    uint fineN = layout.n[fineLevel];
    uint fineHeight = layout.height[fineLevel];
    int x = int(cell % fineN);
    int y = int(cell / fineN % fineHeight);
    int z = int(cell / (fineN * fineHeight));
    float gx = 0.5f * float(x) - 0.25f;
    float gy = 0.5f * float(y) - 0.25f;
    float gz = 0.5f * float(z) - 0.25f;
    int x0 = int(floor(gx));
    int y0 = int(floor(gy));
    int z0 = int(floor(gz));
    float fx = gx - float(x0);
    float fy = gy - float(y0);
    float fz = gz - float(z0);
    float correction = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int cx = clamp(
                    x0 + dx, 0, int(layout.n[coarseLevel]) - 1);
                int cy = clamp(
                    y0 + dy, 0, int(layout.height[coarseLevel]) - 1);
                int cz = clamp(
                    z0 + dz, 0, int(layout.n[coarseLevel]) - 1);
                float weight =
                    (dx != 0 ? fx : 1.0f - fx) *
                    (dy != 0 ? fy : 1.0f - fy) *
                    (dz != 0 ? fz : 1.0f - fz);
                uint coarseCell = uint(cx) + layout.n[coarseLevel] *
                    (uint(cy) + layout.height[coarseLevel] * uint(cz));
                correction += weight *
                    pressurePrimaryLoad(layout, coarseLevel, coarseCell);
            }
    pressurePrimaryStore(layout, fineLevel, cell,
        pressurePrimaryLoad(layout, fineLevel, cell) + 0.5f * correction);
}

#define PM_PRESSURE_RESTRICT_OPEN_ENTRY(NAME, LEVEL) \
[shader("compute")] \
[numthreads(64, 1, 1)] \
void NAME(uint3 dispatchThreadId : SV_DispatchThreadID) \
{ pressureRestrictOpenLevel(LEVEL, dispatchThreadId.x); }

PM_PRESSURE_RESTRICT_OPEN_ENTRY(pm_smoke_pressure_restrict_open_1, 1u)
PM_PRESSURE_RESTRICT_OPEN_ENTRY(pm_smoke_pressure_restrict_open_2, 2u)
PM_PRESSURE_RESTRICT_OPEN_ENTRY(pm_smoke_pressure_restrict_open_3, 3u)

#undef PM_PRESSURE_RESTRICT_OPEN_ENTRY

#define PM_PRESSURE_SMOOTH_ENTRY(NAME, LEVEL, BACKWARD) \
[shader("compute")] \
[numthreads(64, 1, 1)] \
void NAME(uint3 dispatchThreadId : SV_DispatchThreadID) \
{ pressureSmoothLevel(LEVEL, BACKWARD, dispatchThreadId.x); }

PM_PRESSURE_SMOOTH_ENTRY(pm_smoke_pressure_smooth_0_forward, 0u, false)
PM_PRESSURE_SMOOTH_ENTRY(pm_smoke_pressure_smooth_0_backward, 0u, true)
PM_PRESSURE_SMOOTH_ENTRY(pm_smoke_pressure_smooth_1_forward, 1u, false)
PM_PRESSURE_SMOOTH_ENTRY(pm_smoke_pressure_smooth_1_backward, 1u, true)
PM_PRESSURE_SMOOTH_ENTRY(pm_smoke_pressure_smooth_2_forward, 2u, false)
PM_PRESSURE_SMOOTH_ENTRY(pm_smoke_pressure_smooth_2_backward, 2u, true)
PM_PRESSURE_SMOOTH_ENTRY(pm_smoke_pressure_smooth_3_forward, 3u, false)
PM_PRESSURE_SMOOTH_ENTRY(pm_smoke_pressure_smooth_3_backward, 3u, true)

#undef PM_PRESSURE_SMOOTH_ENTRY

#define PM_PRESSURE_RESIDUAL_ENTRY(NAME, LEVEL, TRACK) \
[shader("compute")] \
[numthreads(64, 1, 1)] \
void NAME(uint3 dispatchThreadId : SV_DispatchThreadID) \
{ pressureResidualLevel(LEVEL, TRACK, dispatchThreadId.x); }

PM_PRESSURE_RESIDUAL_ENTRY(pm_smoke_pressure_residual_0, 0u, true)
PM_PRESSURE_RESIDUAL_ENTRY(pm_smoke_pressure_residual_1, 1u, false)
PM_PRESSURE_RESIDUAL_ENTRY(pm_smoke_pressure_residual_2, 2u, false)

#undef PM_PRESSURE_RESIDUAL_ENTRY

#define PM_PRESSURE_RESTRICT_RESIDUAL_ENTRY(NAME, LEVEL) \
[shader("compute")] \
[numthreads(64, 1, 1)] \
void NAME(uint3 dispatchThreadId : SV_DispatchThreadID) \
{ pressureRestrictResidualLevel(LEVEL, dispatchThreadId.x); }

PM_PRESSURE_RESTRICT_RESIDUAL_ENTRY(
    pm_smoke_pressure_restrict_residual_0, 0u)
PM_PRESSURE_RESTRICT_RESIDUAL_ENTRY(
    pm_smoke_pressure_restrict_residual_1, 1u)
PM_PRESSURE_RESTRICT_RESIDUAL_ENTRY(
    pm_smoke_pressure_restrict_residual_2, 2u)

#undef PM_PRESSURE_RESTRICT_RESIDUAL_ENTRY

#define PM_PRESSURE_CLEAR_ENTRY(NAME, LEVEL) \
[shader("compute")] \
[numthreads(64, 1, 1)] \
void NAME(uint3 dispatchThreadId : SV_DispatchThreadID) \
{ pressureClearLevel(LEVEL, dispatchThreadId.x); }

PM_PRESSURE_CLEAR_ENTRY(pm_smoke_pressure_clear_1, 1u)
PM_PRESSURE_CLEAR_ENTRY(pm_smoke_pressure_clear_2, 2u)
PM_PRESSURE_CLEAR_ENTRY(pm_smoke_pressure_clear_3, 3u)

#undef PM_PRESSURE_CLEAR_ENTRY

#define PM_PRESSURE_PROLONG_ENTRY(NAME, LEVEL) \
[shader("compute")] \
[numthreads(64, 1, 1)] \
void NAME(uint3 dispatchThreadId : SV_DispatchThreadID) \
{ pressureProlongLevel(LEVEL, dispatchThreadId.x); }

PM_PRESSURE_PROLONG_ENTRY(pm_smoke_pressure_prolong_1, 1u)
PM_PRESSURE_PROLONG_ENTRY(pm_smoke_pressure_prolong_2, 2u)
PM_PRESSURE_PROLONG_ENTRY(pm_smoke_pressure_prolong_3, 3u)

#undef PM_PRESSURE_PROLONG_ENTRY

[shader("compute")]
[numthreads(1, 1, 1)]
void pm_smoke_pressure_cycle_begin(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u || pressureState[2] != 0u) return;
    pressureState[1] = 0u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void pm_smoke_pressure_cycle_end(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u) return;
    float rhs = asfloat(pressureState[0]);
    float residual = asfloat(pressureState[1]);
    float denominator = max(rhs, 1.0e-5f);
    pressureRelativeResidual[0] = residual / denominator;
    pressureState[2] = residual <=
        pressureConstants.grid_pressure_tolerance * denominator ? 1u : 0u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void pm_smoke_grid_pressure_begin(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0u) return;
    pressureRelativeResidual[0] = 0.0f;
    pressureState[0] = 0u;
    pressureState[1] = 0u;
    pressureState[2] = 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_mark_domain_boundaries(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint face = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    uint axis = 0u;
    uint local = 0u;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!pressureFlatFaceCoordinates(
        face, n, height, axis, local, x, y, z)) return;
    int extent = axis == 0u ? int(n) :
        axis == 1u ? int(height) : int(n);
    int coordinate = axis == 0u ? x : axis == 1u ? y : z;
    if (coordinate != 0 && coordinate != extent) return;
    float3 wind = metalLoad(constants.wind);
    float component = axis == 0u ? wind.x : axis == 1u ? wind.y : wind.z;
    float outward = coordinate == 0 ? -1.0f : 1.0f;
    if (component * outward >= 0.0f) return;
    uint faceTotal = pressureFaceTotal(n, height);
    pressureFaceBoundary[face] = 0.0f;
    pressureFaceBoundary[faceTotal + face] = component;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_divergence_parallel(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint cell = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    uint count = n * height * n;
    if (cell >= count) return;
    int x = int(cell % n);
    int y = int(cell / n % height);
    int z = int(cell / (n * height));
    float flux =
        pressureFaceVelocity[pressureFaceIndex(0u, x + 1, y, z, n, height)] -
        pressureFaceVelocity[pressureFaceIndex(0u, x, y, z, n, height)] +
        pressureFaceVelocity[pressureFaceIndex(1u, x, y + 1, z, n, height)] -
        pressureFaceVelocity[pressureFaceIndex(1u, x, y, z, n, height)] +
        pressureFaceVelocity[pressureFaceIndex(2u, x, y, z + 1, n, height)] -
        pressureFaceVelocity[pressureFaceIndex(2u, x, y, z, n, height)];
    float value = flux / constants.grid_spacing /
        max(constants.timestep, 1.0e-12f);
    pressureDivergence[cell] = value;
    InterlockedMax(pressureState[0], asuint(abs(value)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_project_face_parallel(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint face = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    uint axis = 0u;
    uint local = 0u;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!pressureFlatFaceCoordinates(
        face, n, height, axis, local, x, y, z)) return;
    uint faceTotal = pressureFaceTotal(n, height);
    int extent = axis == 0u ? int(n) :
        axis == 1u ? int(height) : int(n);
    int coordinate = axis == 0u ? x : axis == 1u ? y : z;
    if (pressureFaceBoundary[face] <= 1.0e-4f)
    {
        pressureFaceVelocity[face] = pressureFaceBoundary[faceTotal + face];
        return;
    }
    float gradient = 0.0f;
    if (coordinate == 0 || coordinate == extent)
    {
        int cx = x;
        int cy = y;
        int cz = z;
        if (axis == 0u) cx = coordinate == 0 ? 0 : int(n) - 1;
        if (axis == 1u) cy = coordinate == 0 ? 0 : int(height) - 1;
        if (axis == 2u) cz = coordinate == 0 ? 0 : int(n) - 1;
        float inside = pressureValues[
            uint(cx) + n * (uint(cy) + height * uint(cz))];
        gradient = (coordinate == 0 ? inside : -inside) /
            constants.grid_spacing;
    }
    else
    {
        int lx = x;
        int ly = y;
        int lz = z;
        int rx = x;
        int ry = y;
        int rz = z;
        if (axis == 0u) { lx = x - 1; rx = x; }
        if (axis == 1u) { ly = y - 1; ry = y; }
        if (axis == 2u) { lz = z - 1; rz = z; }
        gradient = (pressureValues[
            uint(rx) + n * (uint(ry) + height * uint(rz))] -
            pressureValues[
            uint(lx) + n * (uint(ly) + height * uint(lz))]) /
            constants.grid_spacing;
    }
    pressureFaceVelocity[face] -= constants.timestep *
        pressureFaceBoundary[face] * gradient;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_apply_face_boundaries(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint face = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    uint axis = 0u;
    uint local = 0u;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!pressureFlatFaceCoordinates(
        face, n, height, axis, local, x, y, z)) return;
    uint faceTotal = pressureFaceTotal(n, height);
    int extent = axis == 0u ? int(n) :
        axis == 1u ? int(height) : int(n);
    int coordinate = axis == 0u ? x : axis == 1u ? y : z;
    float value = pressureScratch[faceTotal + face];
    if (coordinate == 0 || coordinate == extent)
    {
        if (pressureFaceBoundary[face] <= 1.0e-4f)
            value = pressureFaceBoundary[faceTotal + face];
        else
        {
            int nx = axis == 0u ?
                (coordinate == 0 ? 1 : extent - 1) : x;
            int ny = axis == 1u ?
                (coordinate == 0 ? 1 : extent - 1) : y;
            int nz = axis == 2u ?
                (coordinate == 0 ? 1 : extent - 1) : z;
            value = pressureScratch[faceTotal +
                pressureFaceIndex(axis, nx, ny, nz, n, height)];
        }
    }
    float3 point = pressureFacePosition(axis, x, y, z,
        metalLoad(constants.grid_minimum), constants.grid_spacing);
    if (abs(point.x - constants.emitter_center.x) <
            1.5f * constants.grid_spacing &&
        abs(point.y - constants.emitter_center.y) <=
            constants.emitter_half_extents.x +
                0.5f * constants.grid_spacing &&
        abs(point.z - constants.emitter_center.z) <=
            constants.emitter_half_extents.y +
                0.5f * constants.grid_spacing)
        value = axis == 0u ? constants.initial_velocity.x :
            axis == 1u ? constants.initial_velocity.y :
                constants.initial_velocity.z;
    float open = pressureFaceBoundary[face];
    float wall = pressureFaceBoundary[faceTotal + face];
    pressureFaceVelocity[face] = open * value + (1.0f - open) * wall;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_subgrid_force(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint cell = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint n = constants.grid_resolution;
    uint height = constants.grid_vertical_resolution;
    if (cell >= n * height * n) return;
    int x = int(cell % n);
    int y = int(cell / n % height);
    int z = int(cell / (n * height));
    float3 gradient = float3(
        pressureVorticityMagnitude(x + 1, y, z, n, height) -
            pressureVorticityMagnitude(x - 1, y, z, n, height),
        pressureVorticityMagnitude(x, y + 1, z, n, height) -
            pressureVorticityMagnitude(x, y - 1, z, n, height),
        pressureVorticityMagnitude(x, y, z + 1, n, height) -
            pressureVorticityMagnitude(x, y, z - 1, n, height)) *
        (0.5f / constants.grid_spacing);
    if (dot(gradient, gradient) < 1.0e-12f)
    {
        pressureGridScratch[cell] = metalStore(float3(0.0f));
        return;
    }
    pressureGridScratch[cell] = metalStore(
        cross(normalize(gradient), metalLoad(pressureGridVorticity[cell])) *
        (constants.vorticity_confinement * constants.grid_spacing));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_cell_diagnostics_pre(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint cell = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint count = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    if (cell >= count) return;
    PressureCellDiagnostics diagnostics = pressureCellDiagnostics(
        pressureScratch, constants.grid_resolution,
        constants.grid_vertical_resolution, constants.grid_spacing, cell);
    pressureGridVorticity[cell] = metalStore(diagnostics.vorticity);
    pressureDivergence[cell] = diagnostics.strain;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pm_smoke_grid_cell_diagnostics_post(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint cell = dispatchThreadId.x;
    MetalSmokeConstants constants = pressureConstants;
    uint count = constants.grid_resolution *
        constants.grid_vertical_resolution * constants.grid_resolution;
    if (cell >= count) return;
    PressureCellDiagnostics diagnostics = pressureCellDiagnostics(
        pressureFaceVelocity, constants.grid_resolution,
        constants.grid_vertical_resolution, constants.grid_spacing, cell);
    pressureGridVelocity[cell] = metalStore(diagnostics.velocity);
    pressureGridVorticity[cell] = metalStore(diagnostics.vorticity);
    pressureDivergence[cell] = diagnostics.divergence;
}
