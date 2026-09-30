// SPDX-License-Identifier: MIT
// Included inside world.cu's private namespace, after shared contact helpers.

// Fixed topology, refitted swept bounds. The second child to finish owns its
// parent's reduction; release/acquire fences make child bounds visible.
__global__ void fluid_soft_refit(const Vec3 *surface, const Vec3 *previous,
    const std::uint32_t *indices, const std::uint32_t *order,
    BvhNode *tree, const std::uint32_t *parents, std::uint32_t *ready,
    std::uint32_t tree_count, Vec3 *bounds) {
    std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tree_count || tree[index].triangle_count == 0) return;
    Vec3 lo{FLT_MAX, FLT_MAX, FLT_MAX}, hi{-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (std::uint32_t item = 0; item < tree[index].triangle_count; ++item) {
        const auto base = order[tree[index].first_triangle + item] * 3U;
        for (unsigned corner = 0; corner < 3; ++corner) {
            const auto vertex = indices[base + corner];
            lo = component_min(lo, component_min(surface[vertex], previous[vertex]));
            hi = component_max(hi, component_max(surface[vertex], previous[vertex]));
        }
    }
    tree[index].minimum = lo;
    tree[index].maximum = hi;
    while (parents[index] != k_invalid_dense) {
        index = parents[index];
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device> completed(ready[index]);
        if (completed.fetch_add(1U, cuda::memory_order_acq_rel) == 0) return;
        const auto left = tree[index].left, right = tree[index].right;
        tree[index].minimum = component_min(tree[left].minimum, tree[right].minimum);
        tree[index].maximum = component_max(tree[left].maximum, tree[right].maximum);
    }
    bounds[0] = tree[0].minimum;
    bounds[1] = tree[0].maximum;
}

__device__ float fluid_soft_bounds_distance(Vec3 p, const BvhNode &node) {
    const Vec3 delta = component_max(component_max(subtract(node.minimum, p),
        subtract(p, node.maximum)), {});
    return length_squared(delta);
}

// Signed ray crossings give the same winding classification as solid angle,
// but skip almost all triangles through the hierarchy. A ray on an edge falls
// back to solid angle rather than double-counting or missing a shared edge.
__device__ bool fluid_soft_inside(Vec3 point, const Vec3 *surface,
    const std::uint32_t *indices, std::uint32_t index_count,
    const BvhNode *tree, const std::uint32_t *order) {
    const Vec3 direction{1.0F, 0.371F, 0.173F};
    std::uint32_t stack[64]; unsigned size = 1; stack[0] = 0;
    int winding = 0; bool ambiguous = false;
    while (size) {
        const auto &node = tree[stack[--size]];
        const Vec3 lo = subtract(node.minimum, point), hi = subtract(node.maximum, point);
        const float begin = fmaxf(0, fmaxf(lo.x, fmaxf(lo.y / direction.y, lo.z / direction.z)));
        const float end = fminf(hi.x, fminf(hi.y / direction.y, hi.z / direction.z));
        if (end < begin) continue;
        if (!node.triangle_count) { stack[size++] = node.left; stack[size++] = node.right; continue; }
        for (unsigned item = 0; item < node.triangle_count; ++item) {
            const auto base = order[node.first_triangle + item] * 3U;
            const Vec3 a = surface[indices[base]], e1 = subtract(surface[indices[base+1]], a),
                       e2 = subtract(surface[indices[base+2]], a);
            const Vec3 h = cross(direction, e2), s = subtract(point, a);
            const float determinant = dot(e1, h);
            if (fabsf(determinant) < 1.0e-12F) continue;
            const float u = dot(s, h) / determinant;
            const Vec3 q = cross(s, e1);
            const float v = dot(direction, q) / determinant, t = dot(e2, q) / determinant;
            if (t < 0 || u < -1.0e-5F || v < -1.0e-5F || u+v > 1.00001F) continue;
            if (u < 1.0e-5F || v < 1.0e-5F || u+v > 0.99999F || t < 1.0e-7F) ambiguous = true;
            winding += determinant < 0 ? 1 : -1;
        }
    }
    if (!ambiguous) return winding != 0;
    float angle = 0;
    for (std::uint32_t base = 0; base < index_count; base += 3) {
        const Vec3 a = subtract(surface[indices[base]], point),
                   b = subtract(surface[indices[base+1]], point),
                   c = subtract(surface[indices[base+2]], point);
        const float la = vector_length(a), lb = vector_length(b), lc = vector_length(c);
        angle += 2 * atan2f(dot(a, cross(b,c)), la*lb*lc + dot(a,b)*lc + dot(b,c)*la + dot(c,a)*lb);
    }
    return fabsf(angle) > 6.2831853F;
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
    const BvhNode *tree, const std::uint32_t *order) {
    FluidSoftNearest result{};
    const Vec3 margin{distance, distance, distance};
    if (!bounds_overlap(component_min(point, start), component_max(point, start),
        subtract(bounds[0], margin), add(bounds[1], margin))) return result;
    float best = FLT_MAX, earliest = 2.0F;
    std::uint32_t stack[64]; unsigned size = 1; stack[0] = 0;
    while (size) {
        const auto &node = tree[stack[--size]];
        if (fluid_soft_bounds_distance(point, node) > best &&
            (!sweep || !fluid_segment_bounds(start, point, node, distance))) continue;
        if (!node.triangle_count) {
            const bool left_nearer = fluid_soft_bounds_distance(point, tree[node.left]) <
                fluid_soft_bounds_distance(point, tree[node.right]);
            stack[size++] = left_nearer ? node.right : node.left;
            stack[size++] = left_nearer ? node.left : node.right;
            continue;
        }
        for (unsigned item = 0; item < node.triangle_count; ++item) {
            const auto base = order[node.first_triangle + item] * 3U;
            const Vec3 a = surface[indices[base]], b = surface[indices[base + 1]],
                       c = surface[indices[base + 2]];
            const Vec3 cross_normal = cross(subtract(b, a), subtract(c, a));
            if (length_squared(cross_normal) < 1.0e-16F) continue;
            const Vec3 normal = multiply(normalized_or(cross_normal, {}), orientation);
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
    }
    if (result.triangle == k_invalid_dense) return result;
    // Closest edge/vertex normals are radial outside the solid, preventing an
    // incident triangle's plane from extending indefinitely beyond its edge.
    if (earliest > 1.0F) {
        const float radius = sqrtf(best);
        // For a unique closest point in a triangle interior, the oriented
        // face normal is the exact local signed-distance direction of a
        // closed skin. Only edge/vertex cases need a whole-skin ray query.
        const bool face_interior=result.weights.x>1.0e-4F &&
            result.weights.y>1.0e-4F && result.weights.z>1.0e-4F;
        const bool inside=face_interior ?
            dot(subtract(point,result.point),result.normal)<0.0F :
            fluid_soft_inside(point,surface,indices,index_count,tree,order);
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
    const BvhNode *tree, const std::uint32_t *order,
    FluidSoftContact *contacts, std::uint32_t *counts,
    std::uint32_t *contact_count, float *maximum_penetration) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    FluidSoftContact contact{};
    const auto hit = fluid_soft_nearest(positions[particle], previous[particle],
        surface, old_surface, indices, index_count, orientation, distance, bounds, sweep, tree, order);
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
    const Vec3 *bounds, bool sweep, const BvhNode *tree, const std::uint32_t *order) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    const auto hit = fluid_soft_nearest(positions[particle], previous[particle],
        surface, old_surface, indices, index_count, orientation, distance, bounds, sweep, tree, order);
    if (hit.triangle != k_invalid_dense)
        positions[particle] = add(positions[particle], multiply(hit.normal, hit.penetration));
}
