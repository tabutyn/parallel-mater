// SPDX-License-Identifier: MIT
// Bidirectional fluid and cloth coupling kernels.

__global__ void fluid_cloth_containment_forces(
    const Vec3 *particle_positions, const Vec3 *particle_velocities,
    const std::uint32_t *particle_count, Vec3 *particle_accelerations,
    float *foam_source, float particle_mass,
    const Vec3 *cloth_positions, const Vec3 *cloth_velocities,
    const std::uint32_t *cloth_indices, std::uint32_t triangle_count,
    float cloth_orientation, FluidClothCouplingOptions options,
    Vec3 *cloth_forces) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *particle_count) return;
    const Vec3 point = particle_positions[particle];
    float best_squared = FLT_MAX;
    Vec3 best_point{}, best_normal{}, best_weights{};
    std::uint32_t best_triangle = 0U;
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t a_index = cloth_indices[3U * triangle];
        const std::uint32_t b_index = cloth_indices[3U * triangle + 1U];
        const std::uint32_t c_index = cloth_indices[3U * triangle + 2U];
        const Vec3 a = cloth_positions[a_index];
        const Vec3 b = cloth_positions[b_index];
        const Vec3 c = cloth_positions[c_index];
        Vec3 weights{};
        const Vec3 nearest = fluid_closest_triangle_barycentric(
            point, a, b, c, weights);
        const float squared = length_squared(subtract(point, nearest));
        if (squared >= best_squared) continue;
        const Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) <= 1.0e-14F) continue;
        best_squared = squared;
        best_point = nearest;
        best_normal = multiply(normalized_or(face, {0.0F, 1.0F, 0.0F}),
                               cloth_orientation);
        best_weights = weights;
        best_triangle = triangle;
    }
    if (best_squared == FLT_MAX) return;
    const float signed_distance = dot(subtract(point, best_point), best_normal);
    const float violation = options.contact_distance + signed_distance;
    const bool outside = signed_distance > 0.0F;
    if (violation <= 0.0F ||
        (!outside && best_squared >
            options.interaction_radius * options.interaction_radius)) return;
    const std::uint32_t a = cloth_indices[3U * best_triangle];
    const std::uint32_t b = cloth_indices[3U * best_triangle + 1U];
    const std::uint32_t c = cloth_indices[3U * best_triangle + 2U];
    const Vec3 surface_velocity = add(
        multiply(cloth_velocities[a], best_weights.x),
        add(multiply(cloth_velocities[b], best_weights.y),
            multiply(cloth_velocities[c], best_weights.z)));
    const Vec3 relative_velocity = subtract(
        particle_velocities[particle], surface_velocity);
    const float normal_speed = dot(relative_velocity, best_normal);
    const float magnitude = fminf(options.maximum_force,
        fmaxf(0.0F, options.stiffness * violation +
                     options.damping * normal_speed));
    const Vec3 tangent = subtract(relative_velocity,
                                  multiply(best_normal, normal_speed));
    const Vec3 force = clamp_length(add(
        multiply(best_normal, -magnitude),
        multiply(tangent, -options.tangential_drag)), options.maximum_force);
    particle_accelerations[particle] = add(
        particle_accelerations[particle], multiply(force, 1.0F / particle_mass));
    foam_source[particle] = fmaxf(foam_source[particle],
        clamp_scalar(magnitude / fmaxf(options.maximum_force, 1.0F),
                     0.0F, 1.0F));
    const Vec3 reaction = multiply(force, -1.0F);
    atomic_add(cloth_forces + a, multiply(reaction, best_weights.x));
    atomic_add(cloth_forces + b, multiply(reaction, best_weights.y));
    atomic_add(cloth_forces + c, multiply(reaction, best_weights.z));
}

__global__ void cloth_apply_fluid_forces(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    const Vec3 *forces, std::uint32_t count, float dt) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count || inverse_masses[vertex] == 0.0F) return;
    const Vec3 velocity_change = clamp_length(
        multiply(forces[vertex], inverse_masses[vertex] * dt), 2.0F);
    velocities[vertex] = clamp_length(
        add(velocities[vertex], velocity_change), 20.0F);
    positions[vertex] = add(positions[vertex], multiply(velocity_change, dt));
}

__global__ void fluid_project_inside_cloth(
    Vec3 *particle_positions, Vec3 *particle_velocities,
    const std::uint32_t *particle_count, float contact_distance,
    const Vec3 *cloth_positions, const Vec3 *cloth_velocities,
    const std::uint32_t *cloth_indices, std::uint32_t triangle_count,
    float cloth_orientation) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *particle_count) return;
    const Vec3 point = particle_positions[particle];
    float best_squared = FLT_MAX;
    Vec3 best_point{}, best_normal{}, best_weights{};
    std::uint32_t best_triangle = 0U;
    for (std::uint32_t triangle = 0U; triangle < triangle_count; ++triangle) {
        const std::uint32_t a_index = cloth_indices[3U * triangle];
        const std::uint32_t b_index = cloth_indices[3U * triangle + 1U];
        const std::uint32_t c_index = cloth_indices[3U * triangle + 2U];
        const Vec3 a = cloth_positions[a_index];
        const Vec3 b = cloth_positions[b_index];
        const Vec3 c = cloth_positions[c_index];
        Vec3 weights{};
        const Vec3 nearest = fluid_closest_triangle_barycentric(
            point, a, b, c, weights);
        const float squared = length_squared(subtract(point, nearest));
        if (squared >= best_squared) continue;
        const Vec3 face = cross(subtract(b, a), subtract(c, a));
        if (length_squared(face) <= 1.0e-14F) continue;
        best_squared = squared;
        best_point = nearest;
        best_normal = multiply(normalized_or(face, {0.0F, 1.0F, 0.0F}),
                               cloth_orientation);
        best_weights = weights;
        best_triangle = triangle;
    }
    if (best_squared == FLT_MAX) return;
    const float signed_distance = dot(subtract(point, best_point), best_normal);
    const float violation = contact_distance + signed_distance;
    if (violation <= 0.0F) return;
    const std::uint32_t a = cloth_indices[3U * best_triangle];
    const std::uint32_t b = cloth_indices[3U * best_triangle + 1U];
    const std::uint32_t c = cloth_indices[3U * best_triangle + 2U];
    particle_positions[particle] = subtract(
        particle_positions[particle], multiply(best_normal, violation));
    const Vec3 surface_velocity = add(
        multiply(cloth_velocities[a], best_weights.x),
        add(multiply(cloth_velocities[b], best_weights.y),
            multiply(cloth_velocities[c], best_weights.z)));
    Vec3 relative = subtract(particle_velocities[particle], surface_velocity);
    const float outward_speed = dot(relative, best_normal);
    if (outward_speed > 0.0F)
        relative = subtract(relative, multiply(best_normal, outward_speed));
    particle_velocities[particle] = add(surface_velocity, relative);
}
