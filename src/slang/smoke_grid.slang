// SPDX-License-Identifier: MIT
// Slang mirror of src/smoke_grid.cuh. Grid entry points follow these shared
// sampling helpers.

import cuda_common;

public uint smoke_grid_index(int x, int y, int z, int n, int height)
{
    return uint(x + n * (y + height * z));
}

public uint smoke_grid_face_index(
    int axis, int x, int y, int z, int n, int height)
{
    if (axis == 0) return uint(x + (n + 1) * (y + height * z));
    if (axis == 1) return uint(x + n * (y + (height + 1) * z));
    return uint(x + n * (y + height * z));
}

public bool smoke_grid_contains(Vec3 point, SmokeGridField grid)
{
    Vec3 p = multiply(subtract(point, grid.minimum), 1.0f / grid.spacing);
    return p.x >= 0.0f && p.y >= 0.0f && p.z >= 0.0f &&
        p.x < float(grid.n) && p.y < float(grid.height) && p.z < float(grid.n);
}

public float smoke_grid_sample_scalar(
    float* values, Vec3 point, SmokeGridField grid)
{
    if (values == nullptr) return 0.0f;
    Vec3 p = subtract(multiply(subtract(point, grid.minimum),
        1.0f / grid.spacing), {0.5f, 0.5f, 0.5f});
    int x0 = int(floor(p.x));
    int y0 = int(floor(p.y));
    int z0 = int(floor(p.z));
    float fx = p.x - float(x0);
    float fy = p.y - float(y0);
    float fz = p.z - float(z0);
    float result = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int x = max(0, min(int(grid.n) - 1, x0 + dx));
                int y = max(0, min(int(grid.height) - 1, y0 + dy));
                int z = max(0, min(int(grid.n) - 1, z0 + dz));
                float weight = (dx != 0 ? fx : 1.0f - fx) *
                    (dy != 0 ? fy : 1.0f - fy) *
                    (dz != 0 ? fz : 1.0f - fz);
                result += values[smoke_grid_index(
                    x, y, z, int(grid.n), int(grid.height))] * weight;
            }
    return result;
}

public Vec3 smoke_grid_sample_vector(
    Vec3* values, Vec3 point, SmokeGridField grid)
{
    if (values == nullptr) return {};
    Vec3 p = subtract(multiply(subtract(point, grid.minimum),
        1.0f / grid.spacing), {0.5f, 0.5f, 0.5f});
    int x0 = int(floor(p.x));
    int y0 = int(floor(p.y));
    int z0 = int(floor(p.z));
    float fx = p.x - float(x0);
    float fy = p.y - float(y0);
    float fz = p.z - float(z0);
    Vec3 result = {};
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int x = max(0, min(int(grid.n) - 1, x0 + dx));
                int y = max(0, min(int(grid.height) - 1, y0 + dy));
                int z = max(0, min(int(grid.n) - 1, z0 + dz));
                float weight = (dx != 0 ? fx : 1.0f - fx) *
                    (dy != 0 ? fy : 1.0f - fy) *
                    (dz != 0 ? fz : 1.0f - fz);
                result = add(result, multiply(values[smoke_grid_index(
                    x, y, z, int(grid.n), int(grid.height))], weight));
            }
    return result;
}

public float smoke_grid_sample_face_component(
    int axis, float* values, Vec3 point, SmokeGridField grid)
{
    if (values == nullptr) return 0.0f;
    Vec3 p = multiply(subtract(point, grid.minimum), 1.0f / grid.spacing);
    int sx = int(grid.n);
    int sy = int(grid.height);
    int sz = int(grid.n);
    if (axis == 0) { p.y -= 0.5f; p.z -= 0.5f; ++sx; }
    if (axis == 1) { p.x -= 0.5f; p.z -= 0.5f; ++sy; }
    if (axis == 2) { p.x -= 0.5f; p.y -= 0.5f; ++sz; }
    int x0 = int(floor(p.x));
    int y0 = int(floor(p.y));
    int z0 = int(floor(p.z));
    float fx = p.x - float(x0);
    float fy = p.y - float(y0);
    float fz = p.z - float(z0);
    float result = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int x = max(0, min(sx - 1, x0 + dx));
                int y = max(0, min(sy - 1, y0 + dy));
                int z = max(0, min(sz - 1, z0 + dz));
                float weight = (dx != 0 ? fx : 1.0f - fx) *
                    (dy != 0 ? fy : 1.0f - fy) *
                    (dz != 0 ? fz : 1.0f - fz);
                result += values[smoke_grid_face_index(
                    axis, x, y, z, int(grid.n), int(grid.height))] * weight;
            }
    return result;
}

public Vec3 smoke_grid_sample_velocity(Vec3 point, SmokeGridField grid)
{
    if (grid.face[0] != nullptr && grid.face[1] != nullptr &&
        grid.face[2] != nullptr)
        return {
            smoke_grid_sample_face_component(0, grid.face[0], point, grid),
            smoke_grid_sample_face_component(1, grid.face[1], point, grid),
            smoke_grid_sample_face_component(2, grid.face[2], point, grid)};
    return smoke_grid_sample_vector(grid.velocity, point, grid);
}

public void smoke_grid_sample(
    Vec3 point, SmokeGridField grid, out Vec3 velocity, out float density)
{
    velocity = smoke_grid_sample_velocity(point, grid);
    density = smoke_grid_sample_scalar(grid.density, point, grid);
}

public float smoke_grid_sample_pressure(Vec3 point, SmokeGridField grid)
{
    return smoke_grid_sample_scalar(grid.pressure, point, grid);
}

public Vec3 smoke_soft_surface_velocity(
    SoftBodySurfaceBinding binding, Vec3* velocities)
{
    Vec3 result = {};
    for (int i = 0; i < 4; ++i)
        result = add(result, multiply(
            velocities[binding.nodes[i]], binding.weights[i]));
    return result;
}
