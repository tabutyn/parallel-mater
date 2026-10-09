// SPDX-License-Identifier: MIT
// Slang mirror of src/soft_body_cloth.cuh.

import cuda_common;
import geometry_fluid;

[shader("compute")]
[numthreads(256, 1, 1)]
void soft_cloth_detect(
    uniform Vec3* positions, uniform Vec3* previous,
    uniform Vec3* velocities, uniform float* inverse_masses,
    uniform uint count, uniform Vec3* cloth_positions,
    uniform Vec3* cloth_previous, uniform Vec3* cloth_velocities,
    uniform float* cloth_inverse_masses, uniform uint* indices,
    uniform Vec3* surface, uniform uint triangle_count,
    uniform float distance, uniform float friction,
    uniform SoftClothContact* contacts, uniform uint* contact_counts)
{
    uint node = pmThreadIndex();
    if (node >= count) return;
    contacts[node] = {};
    if (inverse_masses[node] == 0.0f) return;
    Vec3 point = positions[node];
    Vec3 start = previous[node];
    float bestSquared = 3.402823466e38f;
    Vec3 bestNormal = {};
    Vec3 bestWeights = {};
    Vec3 bestPoint = {};
    uint best = kInvalidDense;
    for (uint triangle = 0u; triangle < triangle_count; ++triangle)
    {
        uint base = triangle * 3u;
        Vec3 a = surface != nullptr ? surface[base] :
            cloth_positions[indices[base]];
        Vec3 b = surface != nullptr ? surface[base + 1u] :
            cloth_positions[indices[base + 1u]];
        Vec3 c = surface != nullptr ? surface[base + 2u] :
            cloth_positions[indices[base + 2u]];
        Vec3 oldA = cloth_previous[indices[base]];
        Vec3 oldB = cloth_previous[indices[base + 1u]];
        Vec3 oldC = cloth_previous[indices[base + 2u]];
        Vec3 expansion = {distance, distance, distance};
        if (!bounds_overlap(component_min(point, start), component_max(point, start),
            subtract(component_min(component_min(a, component_min(b, c)),
                component_min(oldA, component_min(oldB, oldC))), expansion),
            add(component_max(component_max(a, component_max(b, c)),
                component_max(oldA, component_max(oldB, oldC))), expansion)))
            continue;
        Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) < 1.0e-14f) continue;
        Vec3 normal = normalized_or(face, {});
        Vec3 weights = {};
        Vec3 nearest = fluid_closest_triangle_barycentric(
            point, a, b, c, weights);
        Vec3 delta = subtract(point, nearest);
        float squared = length_squared(delta);
        if (squared >= bestSquared) continue;
        Vec3 oldNearest = add(multiply(
            cloth_previous[indices[base]], weights.x), add(multiply(
            cloth_previous[indices[base + 1u]], weights.y), multiply(
            cloth_previous[indices[base + 2u]], weights.z)));
        float before = dot(subtract(start, oldNearest), normal);
        float after = dot(delta, normal);
        float travel = vector_length(subtract(point, start)) +
            vector_length(subtract(nearest, oldNearest));
        bool crossed = before * after < 0.0f &&
            squared <= (travel + distance) * (travel + distance);
        if (!crossed && squared >= distance * distance) continue;
        best = triangle;
        bestSquared = squared;
        bestPoint = nearest;
        bestWeights = weights;
        bestNormal = crossed || squared < 1.0e-14f
            ? multiply(normal, before >= 0.0f ? 1.0f : -1.0f)
            : multiply(delta, rsqrt(squared));
    }
    if (best == kInvalidDense) return;
    float penetration = distance - dot(
        subtract(point, bestPoint), bestNormal);
    if (penetration <= 0.0f) return;
    SoftClothContact contact = {};
    contact.weights[0] = bestWeights.x;
    contact.weights[1] = bestWeights.y;
    contact.weights[2] = bestWeights.z;
    float denominator = inverse_masses[node];
    Vec3 clothVelocity = {};
    for (uint corner = 0u; corner < 3u; ++corner)
    {
        uint vertex = indices[3u * best + corner];
        float weight = contact.weights[corner];
        contact.vertices[corner] = vertex;
        denominator += weight * weight * cloth_inverse_masses[vertex];
        clothVelocity = add(clothVelocity,
            multiply(cloth_velocities[vertex], weight));
        if (weight > 1.0e-6f && cloth_inverse_masses[vertex] > 0.0f)
            pmAtomicAdd(&contact_counts[vertex], 1u);
    }
    contact.position_impulse = multiply(bestNormal,
        min(penetration, 2.0f * distance) / denominator);
    contact.soft_inverse_mass_fraction = inverse_masses[node] / denominator;
    Vec3 relative = subtract(velocities[node], clothVelocity);
    float normalSpeed = dot(relative, bestNormal);
    float normalImpulse = max(0.0f, -normalSpeed) / denominator;
    Vec3 tangent = subtract(relative, multiply(bestNormal, normalSpeed));
    contact.velocity_impulse = subtract(multiply(bestNormal, normalImpulse),
        clamp_length(multiply(tangent, 1.0f / denominator),
            friction * normalImpulse));
    contact.active = true;
    contacts[node] = contact;
}

float soft_cloth_relaxation(SoftClothContact contact, uint* counts)
{
    uint degree = 1u;
    for (uint corner = 0u; corner < 3u; ++corner)
        if (contact.weights[corner] > 1.0e-6f)
            degree = max(degree, counts[contact.vertices[corner]]);
    float softFraction = contact.soft_inverse_mass_fraction;
    return 1.0f /
        (softFraction + float(degree) * (1.0f - softFraction));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void soft_cloth_apply_soft(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* forces, uniform float* inverse_masses,
    uniform uint count, uniform SoftClothContact* contacts,
    uniform uint* counts, uniform float dt, uniform float maximum_speed)
{
    uint node = pmThreadIndex();
    if (node >= count || !contacts[node].active) return;
    SoftClothContact contact = contacts[node];
    float relaxation = soft_cloth_relaxation(contact, counts);
    positions[node] = add(positions[node], multiply(
        contact.position_impulse, relaxation * inverse_masses[node]));
    Vec3 impulse = multiply(contact.velocity_impulse, relaxation);
    velocities[node] = clamp_length(add(velocities[node],
        multiply(impulse, inverse_masses[node])), maximum_speed);
    forces[node] = add(forces[node], multiply(impulse, 1.0f / dt));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void soft_cloth_apply_cloth(
    uniform Vec3* positions, uniform Vec3* velocities,
    uniform Vec3* forces, uniform float* inverse_masses,
    uniform uint count, uniform SoftClothContact* contacts,
    uniform uint contact_count, uniform uint* counts, uniform float dt)
{
    uint vertex = pmThreadIndex();
    if (vertex >= count) return;
    Vec3 positionImpulse = {};
    Vec3 velocityImpulse = {};
    for (uint node = 0u; node < contact_count; ++node)
    {
        SoftClothContact contact = contacts[node];
        if (!contact.active) continue;
        for (uint corner = 0u; corner < 3u; ++corner)
        {
            if (contact.vertices[corner] != vertex) continue;
            float scale = -contact.weights[corner] *
                soft_cloth_relaxation(contact, counts);
            positionImpulse = add(positionImpulse,
                multiply(contact.position_impulse, scale));
            velocityImpulse = add(velocityImpulse,
                multiply(contact.velocity_impulse, scale));
        }
    }
    positions[vertex] = add(positions[vertex],
        multiply(positionImpulse, inverse_masses[vertex]));
    velocities[vertex] = clamp_length(add(velocities[vertex],
        multiply(velocityImpulse, inverse_masses[vertex])), 20.0f);
    forces[vertex] = add(forces[vertex],
        multiply(velocityImpulse, 1.0f / dt));
}
