// SPDX-License-Identifier: MIT
// Slang mirror of src/cloth_smoke.cuh.

import cuda_common;
import geometry_fluid;
import geometry_smoke;
import smoke_grid;

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_cloth_wind(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform float* inverse_masses, uniform uint vertex_count,
    uniform SmokeOptions smoke, uniform Vec3* smoke_positions,
    uniform Vec3* smoke_velocities, uniform uint64_t* keys,
    uniform uint* indices, uniform SmokeGridField grid, uniform float drag,
    uniform float maximum_acceleration, uniform float dt)
{
    uint vertex = pmThreadIndex();
    if (vertex >= vertex_count || inverse_masses[vertex] == 0.0f) return;
    velocities[vertex] = add(velocities[vertex], smoke_wind_delta(
        positions[vertex], velocities[vertex], smoke_positions,
        smoke_velocities, keys, indices, smoke, grid, drag,
        maximum_acceleration, dt));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void smoke_cloth_contact(
    uniform Vec3* positions, uniform Vec3* previous_positions,
    uniform Vec3* velocities, uniform float* ages, uniform float* pressures,
    uniform uint count, uniform float lifetime,
    uniform Vec3* cloth_positions, uniform Vec3* cloth_velocities,
    uniform uint* indices, uniform uint index_count, uniform Vec3* minimum,
    uniform Vec3* maximum, uniform float clearance,
    uniform float pressure_stiffness, uniform float maximum_speed,
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
    uint best = index_count;
    Vec3 nearest = {};
    Vec3 faceNormal = {};
    Vec3 weights = {};
    float earliest = 2.0f;
    uint sweptBest = index_count;
    Vec3 sweptNearest = {};
    Vec3 sweptNormal = {};
    Vec3 sweptWeights = {};
    for (uint base = 0u; base < index_count; base += 3u)
    {
        Vec3 a = cloth_positions[indices[base]];
        Vec3 b = cloth_positions[indices[base + 1u]];
        Vec3 c = cloth_positions[indices[base + 2u]];
        Vec3 normal = cross(subtract(b, a), subtract(c, a));
        if (length_squared(normal) < 1.0e-12f) continue;
        float sideBefore = dot(subtract(before, a), normal);
        float sideAfter = dot(subtract(point, a), normal);
        if (sideBefore * sideAfter < 0.0f)
        {
            float fraction = sideBefore / (sideBefore - sideAfter);
            if (fraction < earliest)
            {
                Vec3 hit = add(before,
                    multiply(subtract(point, before), fraction));
                Vec3 hitWeights = {};
                Vec3 hitNearest = fluid_closest_triangle_barycentric(
                    hit, a, b, c, hitWeights);
                if (length_squared(subtract(hit, hitNearest)) < 1.0e-8f)
                {
                    earliest = fraction;
                    sweptBest = base;
                    sweptNearest = hitNearest;
                    sweptNormal = normalized_or(normal, {1.0f, 0.0f, 0.0f});
                    sweptWeights = hitWeights;
                }
            }
        }
        Vec3 barycentric = {};
        Vec3 candidate = fluid_closest_triangle_barycentric(
            point, a, b, c, barycentric);
        float distanceSquared = length_squared(subtract(point, candidate));
        if (distanceSquared < nearestSquared)
        {
            nearestSquared = distanceSquared;
            best = base;
            nearest = candidate;
            faceNormal = normalized_or(normal, {1.0f, 0.0f, 0.0f});
            weights = barycentric;
        }
    }
    if (sweptBest != index_count)
    {
        best = sweptBest;
        nearest = sweptNearest;
        faceNormal = sweptNormal;
        weights = sweptWeights;
    }
    if (best == index_count) return;
    float side = dot(subtract(before, nearest), faceNormal);
    if (abs(side) < 1.0e-5f)
        side = dot(subtract(point, nearest), faceNormal);
    Vec3 normal = side >= 0.0f ? faceNormal : multiply(faceNormal, -1.0f);
    Vec3 tracePosition = add(nearest, multiply(normal,
        grid.n != 0u ? 1.5f * grid.spacing : clearance));
    positions[particle] = tracePosition;
    Vec3 clothVelocity = add(
        multiply(cloth_velocities[indices[best]], weights.x),
        add(multiply(cloth_velocities[indices[best + 1u]], weights.y),
            multiply(cloth_velocities[indices[best + 2u]], weights.z)));
    Vec3 incoming = subtract(
        grid.n != 0u && smoke_grid_contains(tracePosition, grid)
            ? smoke_grid_sample_velocity(tracePosition, grid)
            : velocities[particle],
        clothVelocity);
    Vec3 relative = incoming;
    float normalInflow = max(0.0f, -dot(relative, normal));
    relative = subtract(relative,
        multiply(normal, min(0.0f, dot(relative, normal))));
    if (grid.n != 0u)
    {
        float sourceSpeed = max(vector_length(incoming), vector_length(
            subtract(velocities[particle], clothVelocity)));
        float tangentSpeed = vector_length(relative);
        if (tangentSpeed > 1.0e-6f && tangentSpeed < sourceSpeed)
            relative = multiply(relative, sourceSpeed / tangentSpeed);
    }
    if (grid.n == 0u)
    {
        pressures[particle] = max(pressures[particle],
            5.0f * pressure_stiffness * normalInflow * normalInflow);
        Vec3 center = multiply(add(low, high), 0.5f);
        Vec3 offset = subtract(nearest, center);
        Vec3 tangent = subtract(offset,
            multiply(normal, dot(offset, normal)));
        if (pressures[particle] > 0.1f &&
            length_squared(tangent) > 1.0e-10f)
        {
            Vec3 outward = normalized_or(tangent, {0.0f, 1.0f, 0.0f});
            float current = dot(relative, outward);
            float target = min(maximum_speed, sqrt(
                length_squared(incoming) + 0.5f * pressures[particle]));
            relative = add(relative, multiply(outward,
                max(0.0f, target - current)));
        }
    }
    velocities[particle] = add(clothVelocity,
        clamp_length(relative, maximum_speed));
}
