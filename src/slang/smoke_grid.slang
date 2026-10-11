// SPDX-License-Identifier: MIT
// Slang mirror of src/smoke_grid.cuh. Grid entry points follow these shared
// sampling helpers.

import cuda_common;
import geometry_fluid;

groupshared Vec3 smoke_grid_rigid_linear[128];
groupshared Vec3 smoke_grid_rigid_angular[128];

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

public uint smoke_grid_face_count(int axis, int n, int height)
{
    if (axis == 0) return uint((n + 1) * height * n);
    if (axis == 1) return uint(n * (height + 1) * n);
    return uint(n * height * (n + 1));
}

public void smoke_grid_face_coordinates(
    int axis, uint face, int n, int height, out int x, out int y, out int z)
{
    if (axis == 0)
    {
        x = int(face % uint(n + 1));
        y = int(face / uint(n + 1) % uint(height));
        z = int(face / uint((n + 1) * height));
    }
    else if (axis == 1)
    {
        x = int(face % uint(n));
        y = int(face / uint(n) % uint(height + 1));
        z = int(face / uint(n * (height + 1)));
    }
    else
    {
        x = int(face % uint(n));
        y = int(face / uint(n) % uint(height));
        z = int(face / uint(n * height));
    }
}

public Vec3 smoke_grid_face_position(
    int axis, int x, int y, int z, SmokeGridField grid)
{
    Vec3 offset = {float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f};
    if (axis == 0) offset.x = float(x);
    if (axis == 1) offset.y = float(y);
    if (axis == 2) offset.z = float(z);
    return add(grid.minimum, multiply(offset, grid.spacing));
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

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_initialize_cells(
    uniform Vec3* velocity, uniform float* density,
    uniform float* temperature, uniform float* pressure,
    uniform float* divergence, uniform Vec3* vorticity,
    uniform float* strain, uniform Vec3* subgrid_force,
    uniform uint* solid, uniform uint count, uniform Vec3 wind)
{
    uint cell = pmThreadIndex();
    if (cell >= count) return;
    velocity[cell] = wind;
    density[cell] = 0.0f;
    temperature[cell] = 0.0f;
    pressure[cell] = 0.0f;
    divergence[cell] = 0.0f;
    strain[cell] = 0.0f;
    vorticity[cell] = {};
    subgrid_force[cell] = {};
    solid[cell] = 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_initialize_faces(
    uniform float* values, uniform float* reverse, uniform float* open,
    uniform float* wall, uniform Vec3* normal,
    uniform uint* nearest_triangle, uniform uint count, uniform float wind)
{
    uint face = pmThreadIndex();
    if (face >= count) return;
    values[face] = wind;
    reverse[face] = wind;
    open[face] = 1.0f;
    wall[face] = 0.0f;
    normal[face] = {};
    nearest_triangle[face] = 0x7fffffffu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_clear_obstacles(
    uniform uint* solid, uniform uint cell_count, uniform float* open_x,
    uniform float* open_y, uniform float* open_z, uniform float* wall_x,
    uniform float* wall_y, uniform float* wall_z, uniform uint x_count,
    uniform uint y_count, uniform uint z_count)
{
    uint index = pmThreadIndex();
    if (index < cell_count) solid[index] = 0u;
    if (index < x_count) { open_x[index] = 1.0f; wall_x[index] = 0.0f; }
    if (index < y_count) { open_y[index] = 1.0f; wall_y[index] = 0.0f; }
    if (index < z_count) { open_z[index] = 1.0f; wall_z[index] = 0.0f; }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_resolve_particle_fields(
    uniform uint64_t* density_accumulator,
    uniform uint64_t* temperature_accumulator, uniform float* density,
    uniform float* temperature, uniform uint count)
{
    uint cell = pmThreadIndex();
    if (cell >= count) return;
    static const float inverseScale = 1.0f / 16777216.0f;
    density[cell] = float(density_accumulator[cell]) * inverseScale;
    temperature[cell] = float(temperature_accumulator[cell]) * inverseScale;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_splat_particles(
    uniform Vec3* positions, uniform float* ages,
    uniform float* thermal_lift, uniform uint count,
    uniform SmokeOptions options, uniform SmokeGridField grid,
    uniform uint64_t* density_accumulator,
    uniform uint64_t* temperature_accumulator)
{
    uint particle = pmThreadIndex();
    if (particle >= count || ages[particle] >= options.lifetime) return;
    Vec3 g = subtract(multiply(subtract(positions[particle], grid.minimum),
        1.0f / grid.spacing), {0.5f, 0.5f, 0.5f});
    int center[3] = {int(floor(g.x + 0.5f)), int(floor(g.y + 0.5f)),
                     int(floor(g.z + 0.5f))};
    float d[3] = {g.x - float(center[0]), g.y - float(center[1]),
                  g.z - float(center[2])};
    float weight[3][3] = {};
    for (int axis = 0; axis < 3; ++axis)
    {
        weight[axis][0] = 0.5f * (0.5f - d[axis]) * (0.5f - d[axis]);
        weight[axis][1] = 0.75f - d[axis] * d[axis];
        weight[axis][2] = 0.5f * (0.5f + d[axis]) * (0.5f + d[axis]);
    }
    float normalization = 1.0f / max(1.0f, options.rest_number_density);
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
            {
                int x = center[0] + dx;
                int y = center[1] + dy;
                int z = center[2] + dz;
                if (x < 0 || y < 0 || z < 0 || x >= int(grid.n) ||
                    y >= int(grid.height) || z >= int(grid.n)) continue;
                float w = weight[0][dx + 1] * weight[1][dy + 1] *
                    weight[2][dz + 1] * normalization;
                uint cell = smoke_grid_index(
                    x, y, z, int(grid.n), int(grid.height));
                static const float scale = 16777216.0f;
                pmAtomicAdd(&density_accumulator[cell],
                    uint64_t(w * scale + 0.5f));
                pmAtomicAdd(&temperature_accumulator[cell], uint64_t(
                    w * clamp_scalar(thermal_lift[particle], 0.0f,
                        options.maximum_speed) * scale + 0.5f));
            }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_advect_face(
    uniform int axis, uniform SmokeGridField grid, uniform float* source,
    uniform float* destination, uniform float dt)
{
    uint face = pmThreadIndex();
    uint count = smoke_grid_face_count(axis, int(grid.n), int(grid.height));
    if (face >= count) return;
    int x = 0, y = 0, z = 0;
    smoke_grid_face_coordinates(
        axis, face, int(grid.n), int(grid.height), x, y, z);
    Vec3 point = smoke_grid_face_position(axis, x, y, z, grid);
    Vec3 v0 = smoke_grid_sample_velocity(point, grid);
    Vec3 midpoint = subtract(point, multiply(v0, 0.5f * dt));
    Vec3 velocity = smoke_grid_sample_velocity(midpoint, grid);
    Vec3 departure = subtract(point, multiply(velocity, dt));
    destination[face] = smoke_grid_sample_face_component(
        axis, source, departure, grid);
}

void smoke_grid_face_minmax(
    int axis, float* values, Vec3 point, SmokeGridField grid,
    out float low, out float high)
{
    Vec3 p = multiply(subtract(point, grid.minimum), 1.0f / grid.spacing);
    int sx = int(grid.n), sy = int(grid.height), sz = int(grid.n);
    if (axis == 0) { p.y -= 0.5f; p.z -= 0.5f; ++sx; }
    if (axis == 1) { p.x -= 0.5f; p.z -= 0.5f; ++sy; }
    if (axis == 2) { p.x -= 0.5f; p.y -= 0.5f; ++sz; }
    int x0 = int(floor(p.x)), y0 = int(floor(p.y)), z0 = int(floor(p.z));
    low = 3.402823466e+38f;
    high = -3.402823466e+38f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int x = max(0, min(sx - 1, x0 + dx));
                int y = max(0, min(sy - 1, y0 + dy));
                int z = max(0, min(sz - 1, z0 + dz));
                float value = values[smoke_grid_face_index(
                    axis, x, y, z, int(grid.n), int(grid.height))];
                low = min(low, value);
                high = max(high, value);
            }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_correct_face(
    uniform int axis, uniform SmokeGridField original,
    uniform float* source, uniform float* predicted,
    uniform float* reversed, uniform float* corrected, uniform float dt)
{
    uint face = pmThreadIndex();
    uint count = smoke_grid_face_count(
        axis, int(original.n), int(original.height));
    if (face >= count) return;
    int x = 0, y = 0, z = 0;
    smoke_grid_face_coordinates(
        axis, face, int(original.n), int(original.height), x, y, z);
    Vec3 point = smoke_grid_face_position(axis, x, y, z, original);
    Vec3 midpoint = subtract(point, multiply(
        smoke_grid_sample_velocity(point, original), 0.5f * dt));
    Vec3 departure = subtract(point, multiply(
        smoke_grid_sample_velocity(midpoint, original), dt));
    float low = 0.0f, high = 0.0f;
    smoke_grid_face_minmax(axis, source, departure, original, low, high);
    corrected[face] = clamp_scalar(predicted[face] +
        0.5f * (source[face] - reversed[face]), low, high);
}

float smoke_grid_read_face(
    SmokeGridField grid, int axis, int x, int y, int z)
{
    int n = int(grid.n), h = int(grid.height);
    if (axis == 0)
    {
        x = max(0, min(n, x));
        y = max(0, min(h - 1, y));
        z = max(0, min(n - 1, z));
    }
    else if (axis == 1)
    {
        x = max(0, min(n - 1, x));
        y = max(0, min(h, y));
        z = max(0, min(n - 1, z));
    }
    else
    {
        x = max(0, min(n - 1, x));
        y = max(0, min(h - 1, y));
        z = max(0, min(n, z));
    }
    return grid.face[axis][smoke_grid_face_index(axis, x, y, z, n, h)];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_cell_diagnostics(
    uniform SmokeGridField grid, uniform Vec3* cell_velocity,
    uniform Vec3* vorticity, uniform float* strain,
    uniform float* divergence)
{
    uint cell = pmThreadIndex();
    if (cell >= grid.n * grid.height * grid.n) return;
    int n = int(grid.n), h = int(grid.height);
    int x = int(cell % grid.n);
    int y = int(cell / grid.n % grid.height);
    int z = int(cell / (grid.n * grid.height));
    float u0 = smoke_grid_read_face(grid, 0, x, y, z);
    float u1 = smoke_grid_read_face(grid, 0, x + 1, y, z);
    float v0 = smoke_grid_read_face(grid, 1, x, y, z);
    float v1 = smoke_grid_read_face(grid, 1, x, y + 1, z);
    float w0 = smoke_grid_read_face(grid, 2, x, y, z);
    float w1 = smoke_grid_read_face(grid, 2, x, y, z + 1);
    float inverse = 1.0f / grid.spacing;
    float transverse = 0.25f * inverse;
    float dux = (u1 - u0) * inverse;
    float duy = (smoke_grid_read_face(grid, 0, x, y + 1, z) +
        smoke_grid_read_face(grid, 0, x + 1, y + 1, z) -
        smoke_grid_read_face(grid, 0, x, y - 1, z) -
        smoke_grid_read_face(grid, 0, x + 1, y - 1, z)) * transverse;
    float duz = (smoke_grid_read_face(grid, 0, x, y, z + 1) +
        smoke_grid_read_face(grid, 0, x + 1, y, z + 1) -
        smoke_grid_read_face(grid, 0, x, y, z - 1) -
        smoke_grid_read_face(grid, 0, x + 1, y, z - 1)) * transverse;
    float dvx = (smoke_grid_read_face(grid, 1, x + 1, y, z) +
        smoke_grid_read_face(grid, 1, x + 1, y + 1, z) -
        smoke_grid_read_face(grid, 1, x - 1, y, z) -
        smoke_grid_read_face(grid, 1, x - 1, y + 1, z)) * transverse;
    float dvy = (v1 - v0) * inverse;
    float dvz = (smoke_grid_read_face(grid, 1, x, y, z + 1) +
        smoke_grid_read_face(grid, 1, x, y + 1, z + 1) -
        smoke_grid_read_face(grid, 1, x, y, z - 1) -
        smoke_grid_read_face(grid, 1, x, y + 1, z - 1)) * transverse;
    float dwx = (smoke_grid_read_face(grid, 2, x + 1, y, z) +
        smoke_grid_read_face(grid, 2, x + 1, y, z + 1) -
        smoke_grid_read_face(grid, 2, x - 1, y, z) -
        smoke_grid_read_face(grid, 2, x - 1, y, z + 1)) * transverse;
    float dwy = (smoke_grid_read_face(grid, 2, x, y + 1, z) +
        smoke_grid_read_face(grid, 2, x, y + 1, z + 1) -
        smoke_grid_read_face(grid, 2, x, y - 1, z) -
        smoke_grid_read_face(grid, 2, x, y - 1, z + 1)) * transverse;
    float dwz = (w1 - w0) * inverse;
    if (cell_velocity != nullptr)
        cell_velocity[cell] = {
            0.5f * (u0 + u1), 0.5f * (v0 + v1), 0.5f * (w0 + w1)};
    vorticity[cell] = {dwy - dvz, duz - dwx, dvx - duy};
    float sxy = 0.5f * (duy + dvx), sxz = 0.5f * (duz + dwx);
    float syz = 0.5f * (dvz + dwy);
    strain[cell] = sqrt(2.0f * (dux * dux + dvy * dvy + dwz * dwz +
        2.0f * (sxy * sxy + sxz * sxz + syz * syz)));
    if (divergence != nullptr) divergence[cell] = dux + dvy + dwz;
}

float smoke_grid_vorticity_magnitude(
    Vec3* vorticity, int x, int y, int z, int n, int h)
{
    x = max(0, min(n - 1, x));
    y = max(0, min(h - 1, y));
    z = max(0, min(n - 1, z));
    return vector_length(vorticity[smoke_grid_index(x, y, z, n, h)]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_subgrid_force(
    uniform SmokeGridField grid, uniform Vec3* vorticity,
    uniform Vec3* force, uniform float strength)
{
    uint cell = pmThreadIndex();
    if (cell >= grid.n * grid.height * grid.n) return;
    int n = int(grid.n), h = int(grid.height);
    int x = int(cell % grid.n);
    int y = int(cell / grid.n % grid.height);
    int z = int(cell / (grid.n * grid.height));
    Vec3 gradient = {
        smoke_grid_vorticity_magnitude(vorticity, x + 1, y, z, n, h) -
            smoke_grid_vorticity_magnitude(vorticity, x - 1, y, z, n, h),
        smoke_grid_vorticity_magnitude(vorticity, x, y + 1, z, n, h) -
            smoke_grid_vorticity_magnitude(vorticity, x, y - 1, z, n, h),
        smoke_grid_vorticity_magnitude(vorticity, x, y, z + 1, n, h) -
            smoke_grid_vorticity_magnitude(vorticity, x, y, z - 1, n, h)};
    gradient = multiply(gradient, 0.5f / grid.spacing);
    if (length_squared(gradient) < 1.0e-12f)
    {
        force[cell] = {};
        return;
    }
    force[cell] = multiply(cross(
        normalized_or(gradient, {1.0f, 0.0f, 0.0f}), vorticity[cell]),
        strength * grid.spacing);
}

float smoke_grid_source_face(
    int axis, float* source, int x, int y, int z, SmokeGridField grid)
{
    int sx = int(grid.n) + (axis == 0 ? 1 : 0);
    int sy = int(grid.height) + (axis == 1 ? 1 : 0);
    int sz = int(grid.n) + (axis == 2 ? 1 : 0);
    x = max(0, min(sx - 1, x));
    y = max(0, min(sy - 1, y));
    z = max(0, min(sz - 1, z));
    return source[smoke_grid_face_index(
        axis, x, y, z, int(grid.n), int(grid.height))];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_apply_face_forces(
    uniform int axis, uniform SmokeGridField grid, uniform float* source,
    uniform float* destination, uniform float* open, uniform float* wall,
    uniform float* strain, uniform Vec3* subgrid,
    uniform float* temperature, uniform SmokeOptions options,
    uniform float dt, uniform Vec3 gravity)
{
    uint face = pmThreadIndex();
    uint count = smoke_grid_face_count(axis, int(grid.n), int(grid.height));
    if (face >= count) return;
    if (open[face] <= 1.0e-4f)
    {
        destination[face] = wall[face];
        return;
    }
    int x = 0, y = 0, z = 0;
    smoke_grid_face_coordinates(
        axis, face, int(grid.n), int(grid.height), x, y, z);
    Vec3 point = smoke_grid_face_position(axis, x, y, z, grid);
    float center = source[face];
    float laplacian = (smoke_grid_source_face(axis, source, x - 1, y, z, grid) +
        smoke_grid_source_face(axis, source, x + 1, y, z, grid) +
        smoke_grid_source_face(axis, source, x, y - 1, z, grid) +
        smoke_grid_source_face(axis, source, x, y + 1, z, grid) +
        smoke_grid_source_face(axis, source, x, y, z - 1, grid) +
        smoke_grid_source_face(axis, source, x, y, z + 1, grid) -
        6.0f * center) / (grid.spacing * grid.spacing);
    float localStrain = smoke_grid_sample_scalar(strain, point, grid);
    float viscosity = options.grid_kinematic_viscosity +
        options.grid_les_coefficient * options.grid_les_coefficient *
        grid.spacing * grid.spacing * localStrain;
    Vec3 curlForce = smoke_grid_sample_vector(subgrid, point, grid);
    float density = smoke_grid_sample_scalar(grid.density, point, grid);
    float thermalLoading = smoke_grid_sample_scalar(temperature, point, grid);
    float heat = density > 1.0e-5f ? clamp_scalar(
        thermalLoading / density, 0.0f, options.maximum_speed) : 0.0f;
    Vec3 up = normalized_or(multiply(gravity, -1.0f), {0.0f, 1.0f, 0.0f});
    Vec3 buoyancy = multiply(up, options.buoyancy * density *
        options.rest_number_density + 2.0f * heat);
    float body = axis == 0 ? curlForce.x + buoyancy.x :
        (axis == 1 ? curlForce.y + buoyancy.y : curlForce.z + buoyancy.z);
    float value = center + dt * (viscosity * laplacian + body);
    destination[face] = clamp_scalar(
        value, -options.maximum_speed, options.maximum_speed);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_apply_face_boundaries(
    uniform int axis, uniform SmokeGridField grid, uniform float* values,
    uniform float* open, uniform float* wall, uniform SmokeOptions options)
{
    uint face = pmThreadIndex();
    uint count = smoke_grid_face_count(axis, int(grid.n), int(grid.height));
    if (face >= count) return;
    int x = 0, y = 0, z = 0;
    smoke_grid_face_coordinates(
        axis, face, int(grid.n), int(grid.height), x, y, z);
    int extent = axis == 0 ? int(grid.n) :
        (axis == 1 ? int(grid.height) : int(grid.n));
    int coordinate = axis == 0 ? x : (axis == 1 ? y : z);
    if (coordinate == 0 || coordinate == extent)
    {
        if (open[face] <= 1.0e-4f) values[face] = wall[face];
        else
        {
            int nx = axis == 0 ? (coordinate == 0 ? 1 : extent - 1) : x;
            int ny = axis == 1 ? (coordinate == 0 ? 1 : extent - 1) : y;
            int nz = axis == 2 ? (coordinate == 0 ? 1 : extent - 1) : z;
            values[face] = values[smoke_grid_face_index(
                axis, nx, ny, nz, int(grid.n), int(grid.height))];
        }
    }
    Vec3 point = smoke_grid_face_position(axis, x, y, z, grid);
    if (abs(point.x - options.emitter_center.x) < 1.5f * grid.spacing &&
        abs(point.y - options.emitter_center.y) <=
            options.emitter_half_extents.x + 0.5f * grid.spacing &&
        abs(point.z - options.emitter_center.z) <=
            options.emitter_half_extents.y + 0.5f * grid.spacing)
        values[face] = axis == 0 ? options.initial_velocity.x :
            (axis == 1 ? options.initial_velocity.y : options.initial_velocity.z);
    values[face] = open[face] * values[face] +
        (1.0f - open[face]) * wall[face];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_divergence(
    uniform SmokeGridField grid, uniform float* divergence,
    uniform float* rhs_max, uniform float dt)
{
    uint cell = pmThreadIndex();
    if (cell >= grid.n * grid.height * grid.n) return;
    int n = int(grid.n), h = int(grid.height);
    int x = int(cell % grid.n), y = int(cell / grid.n % grid.height);
    int z = int(cell / (grid.n * grid.height));
    float value = (grid.face[0][smoke_grid_face_index(0, x + 1, y, z, n, h)] -
        grid.face[0][smoke_grid_face_index(0, x, y, z, n, h)] +
        grid.face[1][smoke_grid_face_index(1, x, y + 1, z, n, h)] -
        grid.face[1][smoke_grid_face_index(1, x, y, z, n, h)] +
        grid.face[2][smoke_grid_face_index(2, x, y, z + 1, n, h)] -
        grid.face[2][smoke_grid_face_index(2, x, y, z, n, h)]) / grid.spacing;
    divergence[cell] = value / dt;
    if (rhs_max != nullptr)
        pmAtomicMax(reinterpret<uint*>(rhs_max), asuint(abs(value / dt)));
}

float smoke_pressure_open(
    float* open_x, float* open_y, float* open_z, int direction,
    int x, int y, int z, int n, int h)
{
    if (direction == 0) return open_x[smoke_grid_face_index(0, x, y, z, n, h)];
    if (direction == 1) return open_x[smoke_grid_face_index(0, x + 1, y, z, n, h)];
    if (direction == 2) return open_y[smoke_grid_face_index(1, x, y, z, n, h)];
    if (direction == 3) return open_y[smoke_grid_face_index(1, x, y + 1, z, n, h)];
    if (direction == 4) return open_z[smoke_grid_face_index(2, x, y, z, n, h)];
    return open_z[smoke_grid_face_index(2, x, y, z + 1, n, h)];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_pressure_smooth(
    uniform int n, uniform int h, uniform float spacing, uniform float* rhs,
    uniform float* pressure, uniform float* next, uniform float* open_x,
    uniform float* open_y, uniform float* open_z, uniform uint* converged)
{
    uint cell = pmThreadIndex();
    if (cell >= uint(n * h * n) ||
        (converged != nullptr && *converged != 0u)) return;
    int x = int(cell % uint(n));
    int y = int(cell / uint(n) % uint(h));
    int z = int(cell / uint(n * h));
    int dx[6] = {-1, 1, 0, 0, 0, 0};
    int dy[6] = {0, 0, -1, 1, 0, 0};
    int dz[6] = {0, 0, 0, 0, -1, 1};
    float sum = 0.0f, diagonal = 0.0f;
    for (int direction = 0; direction < 6; ++direction)
    {
        float coefficient = smoke_pressure_open(
            open_x, open_y, open_z, direction, x, y, z, n, h);
        if (coefficient <= 1.0e-5f) continue;
        int xx = x + dx[direction], yy = y + dy[direction];
        int zz = z + dz[direction];
        if (xx >= 0 && yy >= 0 && zz >= 0 && xx < n && yy < h && zz < n)
            sum += coefficient * pressure[smoke_grid_index(xx, yy, zz, n, h)];
        diagonal += coefficient;
    }
    float jacobi = diagonal > 1.0e-6f
        ? (sum - spacing * spacing * rhs[cell]) / diagonal : 0.0f;
    static const float omega = 2.0f / 3.0f;
    next[cell] = pressure[cell] + omega * (jacobi - pressure[cell]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_pressure_residual(
    uniform int n, uniform int h, uniform float spacing, uniform float* rhs,
    uniform float* pressure, uniform float* residual, uniform float* open_x,
    uniform float* open_y, uniform float* open_z, uniform float* maximum,
    uniform uint* converged)
{
    uint cell = pmThreadIndex();
    if (cell >= uint(n * h * n) ||
        (converged != nullptr && *converged != 0u)) return;
    int x = int(cell % uint(n));
    int y = int(cell / uint(n) % uint(h));
    int z = int(cell / uint(n * h));
    int dx[6] = {-1, 1, 0, 0, 0, 0};
    int dy[6] = {0, 0, -1, 1, 0, 0};
    int dz[6] = {0, 0, 0, 0, -1, 1};
    float sum = 0.0f, diagonal = 0.0f;
    for (int direction = 0; direction < 6; ++direction)
    {
        float coefficient = smoke_pressure_open(
            open_x, open_y, open_z, direction, x, y, z, n, h);
        if (coefficient <= 1.0e-5f) continue;
        int xx = x + dx[direction], yy = y + dy[direction];
        int zz = z + dz[direction];
        if (xx >= 0 && yy >= 0 && zz >= 0 && xx < n && yy < h && zz < n)
            sum += coefficient * pressure[smoke_grid_index(xx, yy, zz, n, h)];
        diagonal += coefficient;
    }
    float laplacian = (sum - diagonal * pressure[cell]) /
        (spacing * spacing);
    residual[cell] = rhs[cell] - laplacian;
    if (maximum != nullptr)
        pmAtomicMax(reinterpret<uint*>(maximum), asuint(abs(residual[cell])));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_restrict_residual(
    uniform float* fine, uniform int fine_n, uniform int fine_h,
    uniform float* coarse, uniform int coarse_n, uniform int coarse_h)
{
    uint cell = pmThreadIndex();
    if (cell >= uint(coarse_n * coarse_h * coarse_n)) return;
    int x = int(cell % uint(coarse_n));
    int y = int(cell / uint(coarse_n) % uint(coarse_h));
    int z = int(cell / uint(coarse_n * coarse_h));
    float sum = 0.0f;
    int samples = 0;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int xx = min(fine_n - 1, 2 * x + dx);
                int yy = min(fine_h - 1, 2 * y + dy);
                int zz = min(fine_n - 1, 2 * z + dz);
                sum += fine[smoke_grid_index(xx, yy, zz, fine_n, fine_h)];
                ++samples;
            }
    coarse[cell] = sum / float(samples);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_clear_scalar(uniform float* values, uniform uint count)
{
    uint index = pmThreadIndex();
    if (index < count) values[index] = 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_restrict_open(
    uniform int axis, uniform float* fine, uniform int fine_n,
    uniform int fine_h, uniform float* coarse, uniform int coarse_n,
    uniform int coarse_h)
{
    uint face = pmThreadIndex();
    uint count = smoke_grid_face_count(axis, coarse_n, coarse_h);
    if (face >= count) return;
    int x = 0, y = 0, z = 0;
    smoke_grid_face_coordinates(axis, face, coarse_n, coarse_h, x, y, z);
    float sum = 0.0f;
    int samples = 0;
    for (int b = 0; b < 2; ++b)
        for (int a = 0; a < 2; ++a)
        {
            int xx = 2 * x, yy = 2 * y, zz = 2 * z;
            if (axis == 0) { yy += a; zz += b; }
            if (axis == 1) { xx += a; zz += b; }
            if (axis == 2) { xx += a; yy += b; }
            int sx = fine_n + (axis == 0 ? 1 : 0);
            int sy = fine_h + (axis == 1 ? 1 : 0);
            int sz = fine_n + (axis == 2 ? 1 : 0);
            xx = min(sx - 1, xx); yy = min(sy - 1, yy); zz = min(sz - 1, zz);
            sum += fine[smoke_grid_face_index(axis, xx, yy, zz, fine_n, fine_h)];
            ++samples;
        }
    coarse[face] = sum / float(samples);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_mark_domain_boundaries(
    uniform int axis, uniform SmokeGridField grid, uniform float* open,
    uniform float* wall, uniform SmokeOptions options)
{
    uint face = pmThreadIndex();
    uint count = smoke_grid_face_count(axis, int(grid.n), int(grid.height));
    if (face >= count) return;
    int x = 0, y = 0, z = 0;
    smoke_grid_face_coordinates(
        axis, face, int(grid.n), int(grid.height), x, y, z);
    int extent = axis == 0 ? int(grid.n) :
        (axis == 1 ? int(grid.height) : int(grid.n));
    int coordinate = axis == 0 ? x : (axis == 1 ? y : z);
    if (coordinate != 0 && coordinate != extent) return;
    float wind = axis == 0 ? options.wind.x :
        (axis == 1 ? options.wind.y : options.wind.z);
    float outward = coordinate == 0 ? -1.0f : 1.0f;
    if (wind * outward < 0.0f)
    {
        open[face] = 0.0f;
        wall[face] = wind;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_prolong_add(
    uniform float* coarse, uniform int coarse_n, uniform int coarse_h,
    uniform float* fine, uniform int fine_n, uniform int fine_h)
{
    uint cell = pmThreadIndex();
    if (cell >= uint(fine_n * fine_h * fine_n)) return;
    int x = int(cell % uint(fine_n));
    int y = int(cell / uint(fine_n) % uint(fine_h));
    int z = int(cell / uint(fine_n * fine_h));
    float gx = 0.5f * float(x) - 0.25f;
    float gy = 0.5f * float(y) - 0.25f;
    float gz = 0.5f * float(z) - 0.25f;
    int x0 = int(floor(gx)), y0 = int(floor(gy)), z0 = int(floor(gz));
    float fx = gx - float(x0), fy = gy - float(y0), fz = gz - float(z0);
    float correction = 0.0f;
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx)
            {
                int cx = max(0, min(coarse_n - 1, x0 + dx));
                int cy = max(0, min(coarse_h - 1, y0 + dy));
                int cz = max(0, min(coarse_n - 1, z0 + dz));
                float weight = (dx != 0 ? fx : 1.0f - fx) *
                    (dy != 0 ? fy : 1.0f - fy) *
                    (dz != 0 ? fz : 1.0f - fz);
                correction += weight * coarse[smoke_grid_index(
                    cx, cy, cz, coarse_n, coarse_h)];
            }
    fine[cell] += 0.5f * correction;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_compare_residual(
    uniform float* rhs_max, uniform float* residual_max,
    uniform float tolerance, uniform uint* converged)
{
    if (pmThreadIndex() != 0u) return;
    *converged = *residual_max <= tolerance * max(*rhs_max, 1.0e-5f) ? 1u : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_reset_residual(
    uniform float* residual_max, uniform uint* converged)
{
    if (pmThreadIndex() == 0u && *converged == 0u) *residual_max = 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_project_face(
    uniform int axis, uniform SmokeGridField grid, uniform float* pressure,
    uniform float* velocity, uniform float* open, uniform float* wall,
    uniform float dt)
{
    uint face = pmThreadIndex();
    uint count = smoke_grid_face_count(axis, int(grid.n), int(grid.height));
    if (face >= count) return;
    int x = 0, y = 0, z = 0;
    smoke_grid_face_coordinates(
        axis, face, int(grid.n), int(grid.height), x, y, z);
    int extent = axis == 0 ? int(grid.n) :
        (axis == 1 ? int(grid.height) : int(grid.n));
    int coordinate = axis == 0 ? x : (axis == 1 ? y : z);
    if (coordinate == 0 || coordinate == extent)
    {
        if (open[face] <= 1.0e-4f)
        {
            velocity[face] = wall[face];
            return;
        }
        int cx = x, cy = y, cz = z;
        if (axis == 0) cx = coordinate == 0 ? 0 : int(grid.n) - 1;
        if (axis == 1) cy = coordinate == 0 ? 0 : int(grid.height) - 1;
        if (axis == 2) cz = coordinate == 0 ? 0 : int(grid.n) - 1;
        float inside = pressure[smoke_grid_index(
            cx, cy, cz, int(grid.n), int(grid.height))];
        float gradient = (coordinate == 0 ? inside : -inside) / grid.spacing;
        velocity[face] -= dt * open[face] * gradient;
        return;
    }
    int lx = x, ly = y, lz = z, rx = x, ry = y, rz = z;
    if (axis == 0) { lx = x - 1; rx = x; }
    if (axis == 1) { ly = y - 1; ry = y; }
    if (axis == 2) { lz = z - 1; rz = z; }
    float gradient = (pressure[smoke_grid_index(
        rx, ry, rz, int(grid.n), int(grid.height))] -
        pressure[smoke_grid_index(
            lx, ly, lz, int(grid.n), int(grid.height))]) / grid.spacing;
    velocity[face] = open[face] <= 1.0e-4f ? wall[face] :
        velocity[face] - dt * open[face] * gradient;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_trace(
    uniform Vec3* positions, uniform Vec3* previous_positions,
    uniform Vec3* velocities, uniform float* ages,
    uniform float* thermal_lift, uniform float* number_densities,
    uniform float* pressures, uniform Vec3* particle_vorticities,
    uniform uint count, uniform SmokeOptions options,
    uniform SmokeGridField grid, uniform float dt, uniform Vec3 gravity)
{
    uint particle = pmThreadIndex();
    if (particle >= count || ages[particle] >= options.lifetime) return;
    Vec3 point = positions[particle];
    previous_positions[particle] = point;
    Vec3 velocity = velocities[particle];
    if (smoke_grid_contains(point, grid) && ages[particle] > 0.0f)
    {
        Vec3 v0 = smoke_grid_sample_velocity(point, grid);
        Vec3 midpoint = add(point, multiply(v0, 0.5f * dt));
        velocity = smoke_grid_sample_velocity(midpoint, grid);
        number_densities[particle] = smoke_grid_sample_scalar(
            grid.density, point, grid) * options.rest_number_density;
        pressures[particle] = smoke_grid_sample_pressure(point, grid);
        particle_vorticities[particle] = smoke_grid_sample_vector(
            grid.vorticity, point, grid);
    }
    velocity = clamp_length(velocity, options.maximum_speed);
    positions[particle] = add(point, multiply(velocity, dt));
    velocities[particle] = velocity;
    ages[particle] += dt;
    thermal_lift[particle] *= exp(-dt / 3.0f);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_soft_body_force(
    uniform Vec3* surface, uniform uint* indices, uniform uint index_count,
    uniform SoftBodySurfaceBinding* bindings,
    uniform Vec3* node_velocity_snapshot, uniform Vec3* node_velocities,
    uniform float* inverse_masses, uniform SmokeGridField grid,
    uniform float drag_coefficient, uniform float density_scale,
    uniform float kinematic_viscosity, uniform float les_coefficient,
    uniform float maximum_acceleration, uniform float dt)
{
    uint triangle = pmThreadIndex();
    if (triangle >= index_count / 3u) return;
    uint base = triangle * 3u;
    uint ia = indices[base], ib = indices[base + 1u], ic = indices[base + 2u];
    Vec3 a = surface[ia], b = surface[ib], c = surface[ic];
    Vec3 twiceArea = cross(subtract(b, a), subtract(c, a));
    float area = 0.5f * vector_length(twiceArea);
    if (area <= 1.0e-9f) return;
    Vec3 normal = normalized_or(twiceArea, {0.0f, 1.0f, 0.0f});
    Vec3 center = multiply(add(add(a, b), c), 1.0f / 3.0f);
    float offset = 1.5f * grid.spacing;
    Vec3 plus = add(center, multiply(normal, offset));
    Vec3 minus = subtract(center, multiply(normal, offset));
    if (!smoke_grid_contains(plus, grid) ||
        !smoke_grid_contains(minus, grid)) return;
    Vec3 plusAir = {}, minusAir = {};
    float plusDensity = 0.0f, minusDensity = 0.0f;
    smoke_grid_sample(plus, grid, plusAir, plusDensity);
    smoke_grid_sample(minus, grid, minusAir, minusDensity);
    plusDensity *= density_scale;
    minusDensity *= density_scale;
    if (plusDensity + minusDensity < 1.0e-4f) return;
    Vec3 body = multiply(add(add(
        smoke_soft_surface_velocity(bindings[ia], node_velocity_snapshot),
        smoke_soft_surface_velocity(bindings[ib], node_velocity_snapshot)),
        smoke_soft_surface_velocity(bindings[ic], node_velocity_snapshot)),
        1.0f / 3.0f);
    Vec3 plusRelative = subtract(plusAir, body);
    Vec3 minusRelative = subtract(minusAir, body);
    Vec3 plusTangent = subtract(plusRelative,
        multiply(normal, dot(plusRelative, normal)));
    Vec3 minusTangent = subtract(minusRelative,
        multiply(normal, dot(minusRelative, normal)));
    float pressureDifference = minusDensity *
        smoke_grid_sample_pressure(minus, grid) - plusDensity *
        smoke_grid_sample_pressure(plus, grid);
    float localStrain = smoke_grid_sample_scalar(grid.strain, center, grid);
    float viscosity = kinematic_viscosity + les_coefficient * les_coefficient *
        grid.spacing * grid.spacing * localStrain;
    Vec3 force = multiply(add(multiply(normal, pressureDifference),
        multiply(add(multiply(plusTangent, plusDensity),
            multiply(minusTangent, minusDensity)),
            drag_coefficient * viscosity / offset)), area);
    uint corners[3] = {ia, ib, ic};
    for (int corner = 0; corner < 3; ++corner)
    {
        SoftBodySurfaceBinding binding = bindings[corners[corner]];
        for (int j = 0; j < 4; ++j)
        {
            uint node = binding.nodes[j];
            float weight = binding.weights[j];
            if (weight <= 0.0f || inverse_masses[node] == 0.0f) continue;
            Vec3 deltaVelocity = clamp_length(multiply(force,
                dt * inverse_masses[node] * weight / 3.0f),
                maximum_acceleration * dt * weight / 3.0f);
            pmAtomicAdd(&node_velocities[node].x, deltaVelocity.x);
            pmAtomicAdd(&node_velocities[node].y, deltaVelocity.y);
            pmAtomicAdd(&node_velocities[node].z, deltaVelocity.z);
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_cloth_force(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* inverse_masses, uniform uint* indices,
    uniform uint index_count, uniform SmokeGridField grid,
    uniform float drag_coefficient, uniform float density_scale,
    uniform float kinematic_viscosity, uniform float les_coefficient,
    uniform Vec3* forces)
{
    uint triangle = pmThreadIndex();
    if (triangle >= index_count / 3u) return;
    uint base = triangle * 3u;
    uint ia = indices[base], ib = indices[base + 1u], ic = indices[base + 2u];
    if (inverse_masses[ia] == 0.0f && inverse_masses[ib] == 0.0f &&
        inverse_masses[ic] == 0.0f) return;
    Vec3 a = positions[ia], b = positions[ib], c = positions[ic];
    Vec3 twiceArea = cross(subtract(b, a), subtract(c, a));
    float area = 0.5f * vector_length(twiceArea);
    if (area <= 1.0e-9f) return;
    Vec3 normal = normalized_or(twiceArea, {0.0f, 1.0f, 0.0f});
    Vec3 center = multiply(add(add(a, b), c), 1.0f / 3.0f);
    float offset = 1.5f * grid.spacing;
    Vec3 plus = add(center, multiply(normal, offset));
    Vec3 minus = subtract(center, multiply(normal, offset));
    if (!smoke_grid_contains(plus, grid) ||
        !smoke_grid_contains(minus, grid)) return;
    Vec3 plusAir = {}, minusAir = {};
    float plusDensity = 0.0f, minusDensity = 0.0f;
    smoke_grid_sample(plus, grid, plusAir, plusDensity);
    smoke_grid_sample(minus, grid, minusAir, minusDensity);
    plusDensity *= density_scale;
    minusDensity *= density_scale;
    if (plusDensity + minusDensity < 1.0e-4f) return;
    Vec3 clothVelocity = multiply(add(add(
        velocities[ia], velocities[ib]), velocities[ic]), 1.0f / 3.0f);
    Vec3 plusRelative = subtract(plusAir, clothVelocity);
    Vec3 minusRelative = subtract(minusAir, clothVelocity);
    Vec3 plusTangent = subtract(plusRelative,
        multiply(normal, dot(plusRelative, normal)));
    Vec3 minusTangent = subtract(minusRelative,
        multiply(normal, dot(minusRelative, normal)));
    float pressureDifference = minusDensity *
        smoke_grid_sample_pressure(minus, grid) - plusDensity *
        smoke_grid_sample_pressure(plus, grid);
    float localStrain = smoke_grid_sample_scalar(grid.strain, center, grid);
    float viscosity = kinematic_viscosity + les_coefficient * les_coefficient *
        grid.spacing * grid.spacing * localStrain;
    Vec3 cornerForce = multiply(add(multiply(normal, pressureDifference),
        multiply(add(multiply(plusTangent, plusDensity),
            multiply(minusTangent, minusDensity)),
            drag_coefficient * viscosity / offset)), area / 3.0f);
    uint corners[3] = {ia, ib, ic};
    for (int corner = 0; corner < 3; ++corner)
    {
        uint vertex = corners[corner];
        pmAtomicAdd(&forces[vertex].x, cornerForce.x);
        pmAtomicAdd(&forces[vertex].y, cornerForce.y);
        pmAtomicAdd(&forces[vertex].z, cornerForce.z);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_apply_cloth_force(
    uniform Vec3* velocities, uniform float* inverse_masses,
    uniform Vec3* forces, uniform uint count,
    uniform float maximum_acceleration, uniform float dt)
{
    uint vertex = pmThreadIndex();
    if (vertex >= count || inverse_masses[vertex] == 0.0f) return;
    Vec3 delta = clamp_length(multiply(forces[vertex],
        inverse_masses[vertex] * dt), maximum_acceleration * dt);
    velocities[vertex] = add(velocities[vertex], delta);
}

void smoke_grid_raster_triangle(
    Vec3 a, Vec3 b, Vec3 c, Vec3 va, Vec3 vb, Vec3 vc,
    SmokeGridField grid, uint* solid,
    float* open_x, float* open_y, float* open_z,
    float* wall_x, float* wall_y, float* wall_z,
    Vec3* normal_x, Vec3* normal_y, Vec3* normal_z,
    uint* triangle_x, uint* triangle_y, uint* triangle_z,
    uint triangle_id, bool resolve_metadata)
{
    float radius = 0.55f * grid.spacing;
    float inverse = 1.0f / grid.spacing;
    int n = int(grid.n), h = int(grid.height);
    Vec3 triangleNormal = normalized_or(
        cross(subtract(b, a), subtract(c, a)), {0.0f, 1.0f, 0.0f});
    if (!resolve_metadata)
    {
        int cx0 = max(0, int(floor((min(a.x, min(b.x, c.x)) - radius -
            grid.minimum.x) * inverse)));
        int cy0 = max(0, int(floor((min(a.y, min(b.y, c.y)) - radius -
            grid.minimum.y) * inverse)));
        int cz0 = max(0, int(floor((min(a.z, min(b.z, c.z)) - radius -
            grid.minimum.z) * inverse)));
        int cx1 = min(n - 1, int(floor((max(a.x, max(b.x, c.x)) + radius -
            grid.minimum.x) * inverse)));
        int cy1 = min(h - 1, int(floor((max(a.y, max(b.y, c.y)) + radius -
            grid.minimum.y) * inverse)));
        int cz1 = min(n - 1, int(floor((max(a.z, max(b.z, c.z)) + radius -
            grid.minimum.z) * inverse)));
        for (int z = cz0; z <= cz1; ++z)
            for (int y = cy0; y <= cy1; ++y)
                for (int x = cx0; x <= cx1; ++x)
                {
                    Vec3 point = add(grid.minimum, multiply({
                        float(x) + 0.5f, float(y) + 0.5f,
                        float(z) + 0.5f}, grid.spacing));
                    Vec3 weights = {};
                    Vec3 nearest = fluid_closest_triangle_barycentric(
                        point, a, b, c, weights);
                    if (length_squared(subtract(point, nearest)) <=
                        radius * radius)
                        pmAtomicExchange(&solid[smoke_grid_index(
                            x, y, z, n, h)], 1u);
                }
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        Vec3 offset = {0.5f, 0.5f, 0.5f};
        if (axis == 0) offset.x = 0.0f;
        if (axis == 1) offset.y = 0.0f;
        if (axis == 2) offset.z = 0.0f;
        int sx = n + (axis == 0 ? 1 : 0);
        int sy = h + (axis == 1 ? 1 : 0);
        int sz = n + (axis == 2 ? 1 : 0);
        int x0 = max(0, int(floor((min(a.x, min(b.x, c.x)) - radius -
            grid.minimum.x) * inverse - offset.x)));
        int y0 = max(0, int(floor((min(a.y, min(b.y, c.y)) - radius -
            grid.minimum.y) * inverse - offset.y)));
        int z0 = max(0, int(floor((min(a.z, min(b.z, c.z)) - radius -
            grid.minimum.z) * inverse - offset.z)));
        int x1 = min(sx - 1, int(floor((max(a.x, max(b.x, c.x)) + radius -
            grid.minimum.x) * inverse - offset.x)));
        int y1 = min(sy - 1, int(floor((max(a.y, max(b.y, c.y)) + radius -
            grid.minimum.y) * inverse - offset.y)));
        int z1 = min(sz - 1, int(floor((max(a.z, max(b.z, c.z)) + radius -
            grid.minimum.z) * inverse - offset.z)));
        for (int z = z0; z <= z1; ++z)
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x)
                {
                    Vec3 point = add(grid.minimum, multiply({
                        float(x) + offset.x, float(y) + offset.y,
                        float(z) + offset.z}, grid.spacing));
                    Vec3 weights = {};
                    Vec3 nearest = fluid_closest_triangle_barycentric(
                        point, a, b, c, weights);
                    float distance = vector_length(subtract(point, nearest));
                    if (distance > radius) continue;
                    float aperture = clamp_scalar(distance / radius, 0.0f, 1.0f);
                    uint face = smoke_grid_face_index(axis, x, y, z, n, h);
                    float* open = axis == 0 ? open_x :
                        (axis == 1 ? open_y : open_z);
                    float* wall = axis == 0 ? wall_x :
                        (axis == 1 ? wall_y : wall_z);
                    Vec3* faceNormal = axis == 0 ? normal_x :
                        (axis == 1 ? normal_y : normal_z);
                    uint* nearestTriangle = axis == 0 ? triangle_x :
                        (axis == 1 ? triangle_y : triangle_z);
                    if (!resolve_metadata)
                        pmAtomicMin(reinterpret<uint*>(&open[face]),
                            asuint(aperture));
                    else if (aperture <= open[face] + 1.0e-7f)
                    {
                        Vec3 velocity = add(multiply(va, weights.x),
                            add(multiply(vb, weights.y),
                                multiply(vc, weights.z)));
                        wall[face] = axis == 0 ? velocity.x :
                            (axis == 1 ? velocity.y : velocity.z);
                        faceNormal[face] = triangleNormal;
                        nearestTriangle[face] = triangle_id;
                    }
                }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_raster_rigid(
    uniform TriangleMeshResource mesh, uniform RigidBodyState* states,
    uniform uint body, uniform SmokeGridField grid, uniform uint* solid,
    uniform float* open_x, uniform float* open_y, uniform float* open_z,
    uniform float* wall_x, uniform float* wall_y, uniform float* wall_z,
    uniform Vec3* normal_x, uniform Vec3* normal_y, uniform Vec3* normal_z,
    uniform uint* triangle_x, uniform uint* triangle_y,
    uniform uint* triangle_z, uniform bool resolve_metadata)
{
    uint triangle = pmThreadIndex();
    if (triangle >= mesh.index_count / 3u) return;
    uint base = triangle * 3u;
    RigidBodyState state = states[body];
    Vec3 a = transform_point(state, mesh.vertices[mesh.indices[base]]);
    Vec3 b = transform_point(state, mesh.vertices[mesh.indices[base + 1u]]);
    Vec3 c = transform_point(state, mesh.vertices[mesh.indices[base + 2u]]);
    Vec3 va = add(state.linear_velocity,
        cross(state.angular_velocity, subtract(a, state.position)));
    Vec3 vb = add(state.linear_velocity,
        cross(state.angular_velocity, subtract(b, state.position)));
    Vec3 vc = add(state.linear_velocity,
        cross(state.angular_velocity, subtract(c, state.position)));
    smoke_grid_raster_triangle(a, b, c, va, vb, vc, grid, solid,
        open_x, open_y, open_z, wall_x, wall_y, wall_z,
        normal_x, normal_y, normal_z, triangle_x, triangle_y, triangle_z,
        body * 0x9e3779b9u + triangle, resolve_metadata);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_raster_deformable(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform uint* indices, uniform uint index_count,
    uniform SmokeGridField grid, uniform uint* solid,
    uniform float* open_x, uniform float* open_y, uniform float* open_z,
    uniform float* wall_x, uniform float* wall_y, uniform float* wall_z,
    uniform Vec3* normal_x, uniform Vec3* normal_y, uniform Vec3* normal_z,
    uniform uint* triangle_x, uniform uint* triangle_y,
    uniform uint* triangle_z, uniform bool resolve_metadata)
{
    uint triangle = pmThreadIndex();
    if (triangle >= index_count / 3u) return;
    uint base = triangle * 3u;
    uint ia = indices[base], ib = indices[base + 1u], ic = indices[base + 2u];
    Vec3 zero = {};
    smoke_grid_raster_triangle(
        positions[ia], positions[ib], positions[ic],
        velocities != nullptr ? velocities[ia] : zero,
        velocities != nullptr ? velocities[ib] : zero,
        velocities != nullptr ? velocities[ic] : zero,
        grid, solid, open_x, open_y, open_z, wall_x, wall_y, wall_z,
        normal_x, normal_y, normal_z, triangle_x, triangle_y, triangle_z,
        triangle, resolve_metadata);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_grid_raster_soft(
    uniform Vec3* positions, uniform SoftBodySurfaceBinding* bindings,
    uniform Vec3* node_velocities, uniform uint* indices,
    uniform uint index_count, uniform SmokeGridField grid,
    uniform uint* solid, uniform float* open_x, uniform float* open_y,
    uniform float* open_z, uniform float* wall_x, uniform float* wall_y,
    uniform float* wall_z, uniform Vec3* normal_x, uniform Vec3* normal_y,
    uniform Vec3* normal_z, uniform uint* triangle_x,
    uniform uint* triangle_y, uniform uint* triangle_z,
    uniform bool resolve_metadata)
{
    uint triangle = pmThreadIndex();
    if (triangle >= index_count / 3u) return;
    uint base = triangle * 3u;
    uint ia = indices[base], ib = indices[base + 1u], ic = indices[base + 2u];
    smoke_grid_raster_triangle(
        positions[ia], positions[ib], positions[ic],
        smoke_soft_surface_velocity(bindings[ia], node_velocities),
        smoke_soft_surface_velocity(bindings[ib], node_velocities),
        smoke_soft_surface_velocity(bindings[ic], node_velocities),
        grid, solid, open_x, open_y, open_z, wall_x, wall_y, wall_z,
        normal_x, normal_y, normal_z, triangle_x, triangle_y, triangle_z,
        triangle, resolve_metadata);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void smoke_grid_rigid_force(
    uniform TriangleMeshResource mesh, uniform uint body,
    uniform BodyParameters* parameters, uniform RigidBodyState* states,
    uniform SmokeGridField grid, uniform float air_density,
    uniform float drag_coefficient, uniform float density_scale,
    uniform float kinematic_viscosity, uniform float les_coefficient,
    uniform float dt)
{
    uint lane = cudaThreadIdx().x;
    Vec3 localLinear = {};
    Vec3 localAngular = {};
    RigidBodyState state = states[body];
    for (uint triangle = lane; triangle < mesh.index_count / 3u;
         triangle += cudaBlockDim().x)
    {
        uint base = triangle * 3u;
        Vec3 a = transform_point(state, mesh.vertices[mesh.indices[base]]);
        Vec3 b = transform_point(state, mesh.vertices[mesh.indices[base + 1u]]);
        Vec3 c = transform_point(state, mesh.vertices[mesh.indices[base + 2u]]);
        Vec3 twiceArea = cross(subtract(b, a), subtract(c, a));
        float area = 0.5f * vector_length(twiceArea);
        if (area <= 1.0e-9f) continue;
        Vec3 normal = normalized_or(twiceArea, {1.0f, 0.0f, 0.0f});
        Vec3 center = multiply(add(add(a, b), c), 1.0f / 3.0f);
        float offset = 1.5f * grid.spacing;
        Vec3 plus = add(center, multiply(normal, offset));
        Vec3 minus = subtract(center, multiply(normal, offset));
        if (!smoke_grid_contains(plus, grid) ||
            !smoke_grid_contains(minus, grid)) continue;
        Vec3 plusAir = {}, minusAir = {};
        float plusDensity = 0.0f, minusDensity = 0.0f;
        smoke_grid_sample(plus, grid, plusAir, plusDensity);
        smoke_grid_sample(minus, grid, minusAir, minusDensity);
        plusDensity *= density_scale;
        minusDensity *= density_scale;
        if (plusDensity + minusDensity < 1.0e-4f) continue;
        Vec3 arm = subtract(center, state.position);
        Vec3 wall = add(state.linear_velocity,
            cross(state.angular_velocity, arm));
        Vec3 plusRelative = subtract(plusAir, wall);
        Vec3 minusRelative = subtract(minusAir, wall);
        Vec3 plusTangent = subtract(plusRelative,
            multiply(normal, dot(plusRelative, normal)));
        Vec3 minusTangent = subtract(minusRelative,
            multiply(normal, dot(minusRelative, normal)));
        float pressureDifference = minusDensity *
            smoke_grid_sample_pressure(minus, grid) - plusDensity *
            smoke_grid_sample_pressure(plus, grid);
        float localStrain = smoke_grid_sample_scalar(grid.strain, center, grid);
        float viscosity = kinematic_viscosity + les_coefficient *
            les_coefficient * grid.spacing * grid.spacing * localStrain;
        Vec3 force = multiply(add(multiply(normal, pressureDifference),
            multiply(add(multiply(plusTangent, plusDensity),
                multiply(minusTangent, minusDensity)),
                drag_coefficient * viscosity / offset)), air_density * area);
        localLinear = add(localLinear, force);
        localAngular = add(localAngular, cross(arm, force));
    }
    smoke_grid_rigid_linear[lane] = localLinear;
    smoke_grid_rigid_angular[lane] = localAngular;
    pmGroupBarrier();
    for (uint stride = cudaBlockDim().x / 2u; stride != 0u; stride /= 2u)
    {
        if (lane < stride)
        {
            smoke_grid_rigid_linear[lane] = add(
                smoke_grid_rigid_linear[lane],
                smoke_grid_rigid_linear[lane + stride]);
            smoke_grid_rigid_angular[lane] = add(
                smoke_grid_rigid_angular[lane],
                smoke_grid_rigid_angular[lane + stride]);
        }
        pmGroupBarrier();
    }
    if (lane != 0u || parameters[body].motion != MotionType.dynamic) return;
    RigidBodyState next = state;
    BodyParameters settings = parameters[body];
    next.linear_velocity = clamp_length(add(next.linear_velocity, multiply(
        smoke_grid_rigid_linear[0], dt * settings.inverse_mass)),
        settings.maximum_linear_speed);
    next.angular_velocity = clamp_length(add(next.angular_velocity,
        inverse_inertia_world(settings, next,
            multiply(smoke_grid_rigid_angular[0], dt))),
        settings.maximum_angular_speed);
    states[body] = next;
}
