// SPDX-License-Identifier: MIT
// Bidirectional soft-body and cloth coupling kernels.

__global__ void soft_cloth_detect(
    const Vec3 *positions, const Vec3 *previous, const Vec3 *velocities,
    const float *inverse_masses, std::uint32_t count,
    const Vec3 *cloth_positions, const Vec3 *cloth_previous,
    const Vec3 *cloth_velocities, const float *cloth_inverse_masses,
    const std::uint32_t *indices, const Vec3 *surface,
    std::uint32_t triangle_count, float distance, float friction,
    SoftClothContact *contacts, std::uint32_t *contact_counts) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    contacts[node] = {};
    if (inverse_masses[node] == 0.0F) return;
    const Vec3 point = positions[node], start = previous[node];
    float best_squared = FLT_MAX;
    Vec3 best_normal{}, best_weights{}, best_point{};
    std::uint32_t best = k_invalid_dense;
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t base = triangle * 3U;
        const Vec3 a = surface ? surface[base] : cloth_positions[indices[base]];
        const Vec3 b = surface ? surface[base + 1U] : cloth_positions[indices[base + 1U]];
        const Vec3 c = surface ? surface[base + 2U] : cloth_positions[indices[base + 2U]];
        const Vec3 old_a = cloth_previous[indices[base]];
        const Vec3 old_b = cloth_previous[indices[base + 1U]];
        const Vec3 old_c = cloth_previous[indices[base + 2U]];
        const Vec3 expansion{distance, distance, distance};
        if (!bounds_overlap(component_min(point, start), component_max(point, start),
            subtract(component_min(component_min(a, component_min(b, c)),
                component_min(old_a, component_min(old_b, old_c))), expansion),
            add(component_max(component_max(a, component_max(b, c)),
                component_max(old_a, component_max(old_b, old_c))), expansion))) continue;
        const Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) < 1.0e-14F) continue;
        const Vec3 normal = normalized_or(face, {});
        Vec3 weights{};
        const Vec3 nearest = fluid_closest_triangle_barycentric(point, a, b, c, weights);
        const Vec3 delta = subtract(point, nearest);
        const float squared = length_squared(delta);
        if (squared >= best_squared) continue;
        const Vec3 old_nearest = add(multiply(cloth_previous[indices[base]], weights.x),
            add(multiply(cloth_previous[indices[base + 1U]], weights.y),
                multiply(cloth_previous[indices[base + 2U]], weights.z)));
        const float before = dot(subtract(start, old_nearest), normal);
        const float after = dot(delta, normal);
        const float travel = vector_length(subtract(point, start)) +
                             vector_length(subtract(nearest, old_nearest));
        const bool crossed = before * after < 0.0F &&
            squared <= (travel + distance) * (travel + distance);
        if (!crossed && squared >= distance * distance) continue;
        best = triangle;
        best_squared = squared;
        best_point = nearest;
        best_weights = weights;
        best_normal = crossed || squared < 1.0e-14F
            ? multiply(normal, before >= 0.0F ? 1.0F : -1.0F)
            : multiply(delta, rsqrtf(squared));
    }
    if (best == k_invalid_dense) return;
    const float penetration = distance - dot(subtract(point, best_point), best_normal);
    if (penetration <= 0.0F) return;
    SoftClothContact contact{};
    contact.weights[0] = best_weights.x;
    contact.weights[1] = best_weights.y;
    contact.weights[2] = best_weights.z;
    float denominator = inverse_masses[node];
    Vec3 cloth_velocity{};
    for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
        const auto vertex = indices[3U * best + corner];
        const float weight = contact.weights[corner];
        contact.vertices[corner] = vertex;
        denominator += weight * weight * cloth_inverse_masses[vertex];
        cloth_velocity = add(cloth_velocity, multiply(cloth_velocities[vertex], weight));
        if (weight > 1.0e-6F && cloth_inverse_masses[vertex] > 0.0F)
            atomicAdd(contact_counts + vertex, 1U);
    }
    contact.position_impulse = multiply(best_normal,
        fminf(penetration, 2.0F * distance) / denominator);
    contact.soft_inverse_mass_fraction = inverse_masses[node] / denominator;
    const Vec3 relative = subtract(velocities[node], cloth_velocity);
    const float normal_speed = dot(relative, best_normal);
    const float normal_impulse = fmaxf(0.0F, -normal_speed) / denominator;
    const Vec3 tangent = subtract(relative, multiply(best_normal, normal_speed));
    contact.velocity_impulse = subtract(multiply(best_normal, normal_impulse),
        clamp_length(multiply(tangent, 1.0F / denominator), friction * normal_impulse));
    contact.active = true;
    contacts[node] = contact;
}

__device__ float soft_cloth_relaxation(
    const SoftClothContact &contact, const std::uint32_t *counts) {
    std::uint32_t degree = 1U;
    for (std::uint32_t corner = 0U; corner < 3U; ++corner)
        if (contact.weights[corner] > 1.0e-6F)
            degree = max(degree, counts[contact.vertices[corner]]);
    // Only cloth vertices are shared by several contacts. Multiplying the
    // soft node's independent inverse mass by that degree over-damps recovery
    // and allows a fast sheet to pass through it during impact.
    const float soft_fraction = contact.soft_inverse_mass_fraction;
    return 1.0F / (soft_fraction + static_cast<float>(degree) * (1.0F - soft_fraction));
}

__global__ void soft_cloth_apply_soft(
    Vec3 *positions, Vec3 *velocities, Vec3 *forces,
    const float *inverse_masses, std::uint32_t count,
    const SoftClothContact *contacts, const std::uint32_t *counts,
    float dt, float maximum_speed) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count || !contacts[node].active) return;
    const SoftClothContact contact = contacts[node];
    const float relaxation = soft_cloth_relaxation(contact, counts);
    positions[node] = add(positions[node],
        multiply(contact.position_impulse, relaxation * inverse_masses[node]));
    const Vec3 impulse = multiply(contact.velocity_impulse, relaxation);
    velocities[node] = clamp_length(add(velocities[node],
        multiply(impulse, inverse_masses[node])), maximum_speed);
    forces[node] = add(forces[node], multiply(impulse, 1.0F / dt));
}

__global__ void soft_cloth_apply_cloth(
    Vec3 *positions, Vec3 *velocities, Vec3 *forces,
    const float *inverse_masses, std::uint32_t count,
    const SoftClothContact *contacts, std::uint32_t contact_count,
    const std::uint32_t *counts, float dt) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    Vec3 position_impulse{}, velocity_impulse{};
    for (std::uint32_t node = 0U; node < contact_count; ++node) {
        const SoftClothContact contact = contacts[node];
        if (!contact.active) continue;
        for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
            if (contact.vertices[corner] != vertex) continue;
            const float scale = -contact.weights[corner] *
                                soft_cloth_relaxation(contact, counts);
            position_impulse = add(position_impulse, multiply(contact.position_impulse, scale));
            velocity_impulse = add(velocity_impulse, multiply(contact.velocity_impulse, scale));
        }
    }
    positions[vertex] = add(positions[vertex], multiply(position_impulse, inverse_masses[vertex]));
    velocities[vertex] = clamp_length(add(velocities[vertex],
        multiply(velocity_impulse, inverse_masses[vertex])), 20.0F);
    forces[vertex] = add(forces[vertex], multiply(velocity_impulse, 1.0F / dt));
}

// Test the actual skin triangles against verified solid triangle meshes.
// Node contacts can leave a face cutting through a collider between its nodes.
// Select the least-displacing supporting plane, then constrain every corner
// to its outside half-space. The whole face is outside once all corners are.
