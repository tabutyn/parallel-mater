// SPDX-License-Identifier: MIT
// Included inside world.cu's private namespace, after shared contact helpers.

__global__ void fluid_soft_bounds(const Vec3 *surface, const Vec3 *previous,
                                  std::uint32_t count, Vec3 *bounds) {
    Vec3 lo{FLT_MAX, FLT_MAX, FLT_MAX}, hi{-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (std::uint32_t i = 0; i < count; ++i) {
        lo = component_min(lo, component_min(surface[i], previous[i]));
        hi = component_max(hi, component_max(surface[i], previous[i]));
    }
    bounds[0] = lo;
    bounds[1] = hi;
}

struct FluidSoftNearest {
    std::uint32_t triangle{k_invalid_dense};
    Vec3 point{}, normal{}, weights{};
    float penetration{};
};

__device__ FluidSoftNearest fluid_soft_nearest(
    Vec3 point, Vec3 start, const Vec3 *surface, const Vec3 *previous,
    const std::uint32_t *indices, std::uint32_t index_count,
    float orientation, float distance, const Vec3 *bounds, bool sweep,
    bool robust_inside = false) {
    FluidSoftNearest result{};
    const Vec3 margin{distance, distance, distance};
    if (!bounds_overlap(component_min(point, start), component_max(point, start),
        subtract(bounds[0], margin), add(bounds[1], margin))) return result;
    float best = FLT_MAX, earliest = 2.0F, solid_angle = 0.0F;
    for (std::uint32_t base = 0; base < index_count; base += 3) {
        const Vec3 a = surface[indices[base]], b = surface[indices[base + 1]],
                   c = surface[indices[base + 2]];
        const Vec3 cross_normal = cross(subtract(b, a), subtract(c, a));
        if (length_squared(cross_normal) < 1.0e-16F) continue;
        const Vec3 normal = multiply(normalized_or(cross_normal, {}), orientation);
        if (robust_inside) {
            const Vec3 pa = subtract(a, point), pb = subtract(b, point), pc = subtract(c, point);
            const float la = vector_length(pa), lb = vector_length(pb), lc = vector_length(pc);
            solid_angle += 2.0F * atan2f(dot(pa, cross(pb, pc)),
                la*lb*lc + dot(pa,pb)*lc + dot(pb,pc)*la + dot(pc,pa)*lb);
        }
        Vec3 weights{};
        const Vec3 near = fluid_closest_triangle_barycentric(point, a, b, c, weights);
        const float squared = length_squared(subtract(point, near));
        if (earliest > 1.0F && squared < best) {
            best = squared;
            result = {base, near, normal, weights,
                      distance - dot(subtract(point, near), normal)};
        }
        if (!sweep) continue;
        // Follow the same material point on the previous skin. This also
        // catches a moving soft face sweeping over an initially still particle.
        const Vec3 old_near = add(multiply(previous[indices[base]], weights.x),
            add(multiply(previous[indices[base + 1]], weights.y),
                multiply(previous[indices[base + 2]], weights.z)));
        const Vec3 transported_start = add(start, subtract(near, old_near));
        const float first = dot(subtract(transported_start, a), normal);
        const float last = dot(subtract(point, a), normal);
        if (first < distance || last >= distance || first - last < 1.0e-8F) continue;
        const float t = (first - distance) / (first - last);
        if (t >= earliest) continue;
        const Vec3 crossing = subtract(add(transported_start,
            multiply(subtract(point, transported_start), t)), multiply(normal, distance));
        Vec3 hit_weights{};
        const Vec3 hit = fluid_closest_triangle_barycentric(crossing, a, b, c, hit_weights);
        if (length_squared(subtract(crossing, hit)) > 1.0e-8F) continue;
        earliest = t;
        result = {base, hit, normal, hit_weights, distance - last};
    }
    if (result.triangle == k_invalid_dense) return result;
    // Closest edge/vertex normals are radial outside the solid, preventing an
    // incident triangle's plane from extending indefinitely beyond its edge.
    if (earliest > 1.0F) {
        const float signed_distance = dot(subtract(point, result.point), result.normal);
        const float radius = sqrtf(best);
        const bool inside = robust_inside ? fabsf(solid_angle) > 6.2831853F : signed_distance < 0.0F;
        if (!inside) {
            result.penetration = distance - radius;
            if (radius > 1.0e-7F)
                result.normal = multiply(subtract(point, result.point), 1.0F / radius);
        } else {
            result.penetration = distance + radius;
            if (radius > 1.0e-7F)
                result.normal = multiply(subtract(result.point, point), 1.0F / radius);
        }
    }
    if (result.penetration <= 0.0F) result.triangle = k_invalid_dense;
    return result;
}

__global__ void fluid_soft_detect(
    const Vec3 *positions, const Vec3 *previous, const Vec3 *velocities,
    const std::uint32_t *count, const Vec3 *surface, const Vec3 *old_surface,
    const std::uint32_t *indices, std::uint32_t index_count,
    const SoftBodySurfaceBinding *bindings, const Vec3 *soft_velocities,
    float orientation, float distance, const Vec3 *bounds, bool sweep,
    FluidSoftContact *contacts, std::uint32_t *counts,
    std::uint32_t *contact_count, float *maximum_penetration) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    FluidSoftContact contact{};
    const auto hit = fluid_soft_nearest(positions[particle], previous[particle],
        surface, old_surface, indices, index_count, orientation, distance, bounds, sweep, true);
    if (hit.triangle != k_invalid_dense) {
        const float bary[3]{hit.weights.x, hit.weights.y, hit.weights.z};
        for (std::uint32_t corner = 0; corner < 3; ++corner) {
            const auto binding = bindings[indices[hit.triangle + corner]];
            for (std::uint32_t slot = 0; slot < 4; ++slot) {
                const float weight = bary[corner] * binding.weights[slot];
                if (weight <= 0.0F) continue;
                const auto node = binding.nodes[slot];
                std::uint32_t item = 0;
                while (item < contact.count && contact.nodes[item] != node) ++item;
                if (item == contact.count) contact.nodes[contact.count++] = node;
                contact.weights[item] += weight;
            }
        }
        Vec3 surface_velocity{};
        for (std::uint32_t i = 0; i < contact.count; ++i) {
            atomicAdd(counts + contact.nodes[i], 1U);
            surface_velocity = add(surface_velocity,
                multiply(soft_velocities[contact.nodes[i]], contact.weights[i]));
        }
        contact.normal = hit.normal;
        contact.relative_velocity = subtract(velocities[particle], surface_velocity);
        contact.penetration = hit.penetration;
        atomicAdd(contact_count, 1U);
        atomicMax(reinterpret_cast<unsigned int *>(maximum_penetration),
                  __float_as_uint(hit.penetration));
    }
    contacts[particle] = contact;
}

__global__ void fluid_soft_solve(
    Vec3 *positions, Vec3 *velocities, Vec3 *accelerations, float *foam,
    const std::uint32_t *count, const FluidSoftContact *contacts,
    const std::uint32_t *counts, const float *inverse_masses,
    const Vec3 *soft_velocities, float maximum_soft_speed,
    float inverse_particle_mass, float friction, float maximum_projection,
    float dt, Vec3 *node_positions, Vec3 *node_impulses) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    const auto contact = contacts[particle];
    if (contact.count == 0) return;
    float denominator = inverse_particle_mass;
    for (std::uint32_t i = 0; i < contact.count; ++i) {
        const auto node = contact.nodes[i];
        // Same contact-degree relaxation on BOTH endpoints. Summing N
        // independent full-strength reactions used to launch light lattices.
        denominator += contact.weights[i] *
            inverse_masses[node] * counts[node];
    }
    const float normal_speed = dot(contact.relative_velocity, contact.normal);
    const float normal_impulse = fmaxf(0.0F, -normal_speed) / denominator;
    const Vec3 tangent = subtract(contact.relative_velocity,
                                 multiply(contact.normal, normal_speed));
    Vec3 impulse = subtract(multiply(contact.normal, normal_impulse),
        clamp_length(multiply(tangent, 1.0F / denominator), friction * normal_impulse));
    float scale = 1.0F;
    for (std::uint32_t i = 0; i < contact.count; ++i) {
        const auto node = contact.nodes[i];
        const Vec3 v = soft_velocities[node];
        const Vec3 delta = multiply(impulse,
            -contact.weights[i] * inverse_masses[node] * counts[node]);
        const float a = length_squared(delta);
        if (a <= 1.0e-20F) continue;
        const float b = dot(v, delta);
        const float c = fminf(0.0F, length_squared(v) - maximum_soft_speed * maximum_soft_speed);
        const float limit = (-b + sqrtf(fmaxf(0.0F, b*b - a*c))) / a;
        scale = fminf(scale, fmaxf(0.0F, limit));
    }
    // Each node receives an average of count speed-safe proposals. Limit the
    // shared impulse, not the final node velocity: its water reaction remains
    // exactly opposite even when the configured speed ceiling is reached.
    impulse = multiply(impulse, scale);
    const float lambda = fminf(contact.penetration, maximum_projection) / denominator;
    positions[particle] = add(positions[particle],
        multiply(contact.normal, inverse_particle_mass * lambda));
    const Vec3 velocity_delta = multiply(impulse, inverse_particle_mass);
    velocities[particle] = add(velocities[particle], velocity_delta);
    accelerations[particle] = add(accelerations[particle], multiply(velocity_delta, 1.0F / dt));
    foam[particle] = fmaxf(foam[particle], fminf(1.0F, normal_impulse * inverse_particle_mass * 0.1F));
    for (std::uint32_t i = 0; i < contact.count; ++i) {
        const auto node = contact.nodes[i];
        atomic_add(node_positions + node, multiply(contact.normal,
            -contact.weights[i] * inverse_masses[node] * lambda));
        // Include pinned reactions in diagnostics: their support absorbs it.
        atomic_add(node_impulses + node, multiply(impulse, -contact.weights[i]));
    }
}

__global__ void fluid_soft_apply(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    const Vec3 *position_deltas, const Vec3 *impulses, Vec3 *forces,
    std::uint32_t count, float inverse_frame_dt) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    positions[node] = add(positions[node], position_deltas[node]);
    velocities[node] = add(velocities[node], multiply(impulses[node], inverse_masses[node]));
    forces[node] = add(forces[node], multiply(impulses[node], inverse_frame_dt));
}

// Geometric recovery is split from kinetic impulse. It adds no bounce/energy
// and never moves the soft graph through another collider to rescue a particle.
__global__ void fluid_soft_recover(
    Vec3 *positions, const Vec3 *previous, const std::uint32_t *count,
    const Vec3 *surface, const Vec3 *old_surface, const std::uint32_t *indices,
    std::uint32_t index_count, float orientation, float distance,
    const Vec3 *bounds, bool sweep) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    const auto hit = fluid_soft_nearest(positions[particle], previous[particle],
        surface, old_surface, indices, index_count, orientation, distance, bounds, sweep, true);
    if (hit.triangle != k_invalid_dense)
        positions[particle] = add(positions[particle], multiply(hit.normal, hit.penetration));
}
