// SPDX-License-Identifier: MIT
// Slang mirror of src/soft_body_smoke.cuh.

import cuda_common;
import geometry_fluid;
import geometry_smoke;
import smoke_grid;

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_soft_body_wind(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* inverse_masses, uniform uint node_count,
    uniform SmokeOptions smoke, uniform Vec3* smoke_positions,
    uniform Vec3* smoke_velocities, uniform uint64_t* keys,
    uniform uint* indices, uniform SmokeGridField grid, uniform float drag,
    uniform float maximum_acceleration, uniform float dt,
    uniform float maximum_speed)
{
    uint node = pmThreadIndex();
    if (node >= node_count || inverse_masses[node] == 0.0f) return;
    velocities[node] = clamp_length(add(velocities[node], smoke_wind_delta(
        positions[node], velocities[node], smoke_positions, smoke_velocities,
        keys, indices, smoke, grid, drag, maximum_acceleration, dt)),
        maximum_speed);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_soft_body_contact(
    uniform Vec3* positions, uniform Vec3* previous_positions,
    uniform Vec3* velocities, uniform float* ages, uniform uint count,
    uniform float lifetime, uniform Vec3* surface,
    uniform SoftBodySurfaceBinding* bindings, uniform uint* indices,
    uniform uint index_count, uniform Vec3* node_velocities,
    uniform Vec3* minimum, uniform Vec3* maximum, uniform float clearance,
    uniform SmokeGridField grid)
{
    uint particle = pmThreadIndex();
    if (particle >= count || ages[particle] >= lifetime) return;
    Vec3 point = positions[particle];
    Vec3 before = previous_positions[particle];
    Vec3 low = *minimum;
    Vec3 high = *maximum;
    if (max(point.x, before.x) < low.x - clearance ||
        min(point.x, before.x) > high.x + clearance ||
        max(point.y, before.y) < low.y - clearance ||
        min(point.y, before.y) > high.y + clearance ||
        max(point.z, before.z) < low.z - clearance ||
        min(point.z, before.z) > high.z + clearance) return;
    float nearestSquared = clearance * clearance;
    float earliest = 2.0f;
    uint best = index_count;
    Vec3 nearest = {};
    Vec3 faceNormal = {};
    Vec3 weights = {};
    Vec3 path = subtract(point, before);
    for (uint base = 0u; base < index_count; base += 3u)
    {
        Vec3 a = surface[indices[base]];
        Vec3 b = surface[indices[base + 1u]];
        Vec3 c = surface[indices[base + 2u]];
        Vec3 rawNormal = cross(subtract(b, a), subtract(c, a));
        if (length_squared(rawNormal) < 1.0e-12f) continue;
        float sideBefore = dot(subtract(before, a), rawNormal);
        float sideAfter = dot(subtract(point, a), rawNormal);
        if (sideBefore * sideAfter < 0.0f)
        {
            float fraction = sideBefore / (sideBefore - sideAfter);
            if (fraction < earliest)
            {
                Vec3 hit = add(before, multiply(path, fraction));
                Vec3 hitWeights = {};
                Vec3 onFace = fluid_closest_triangle_barycentric(
                    hit, a, b, c, hitWeights);
                if (length_squared(subtract(hit, onFace)) < 1.0e-8f)
                {
                    earliest = fraction;
                    best = base;
                    nearest = onFace;
                    faceNormal = normalized_or(rawNormal, {1.0f, 0.0f, 0.0f});
                    weights = hitWeights;
                }
            }
        }
        if (earliest <= 1.0f) continue;
        Vec3 barycentric = {};
        Vec3 candidate = fluid_closest_triangle_barycentric(
            point, a, b, c, barycentric);
        float distanceSquared = length_squared(subtract(point, candidate));
        if (distanceSquared < nearestSquared)
        {
            nearestSquared = distanceSquared;
            best = base;
            nearest = candidate;
            faceNormal = normalized_or(rawNormal, {1.0f, 0.0f, 0.0f});
            weights = barycentric;
        }
    }
    if (best == index_count) return;
    float side = dot(subtract(before, nearest), faceNormal);
    if (abs(side) < 1.0e-5f)
        side = dot(subtract(point, nearest), faceNormal);
    Vec3 normal = side >= 0.0f ? faceNormal : multiply(faceNormal, -1.0f);
    Vec3 tracePosition = add(nearest, multiply(normal,
        grid.n != 0u ? 1.5f * grid.spacing : clearance));
    positions[particle] = tracePosition;
    Vec3 va = smoke_soft_surface_velocity(
        bindings[indices[best]], node_velocities);
    Vec3 vb = smoke_soft_surface_velocity(
        bindings[indices[best + 1u]], node_velocities);
    Vec3 vc = smoke_soft_surface_velocity(
        bindings[indices[best + 2u]], node_velocities);
    Vec3 bodyVelocity = add(multiply(va, weights.x),
        add(multiply(vb, weights.y), multiply(vc, weights.z)));
    Vec3 relative = subtract(
        grid.n != 0u && smoke_grid_contains(tracePosition, grid)
            ? smoke_grid_sample_velocity(tracePosition, grid)
            : velocities[particle],
        bodyVelocity);
    float sourceSpeed = max(vector_length(relative), vector_length(
        subtract(velocities[particle], bodyVelocity)));
    relative = subtract(relative,
        multiply(normal, min(0.0f, dot(relative, normal))));
    float tangentSpeed = vector_length(relative);
    if (grid.n != 0u && tangentSpeed > 1.0e-6f &&
        tangentSpeed < sourceSpeed)
        relative = multiply(relative, sourceSpeed / tangentSpeed);
    velocities[particle] = add(bodyVelocity, relative);
}
