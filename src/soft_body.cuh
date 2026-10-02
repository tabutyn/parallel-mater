// SPDX-License-Identifier: MIT
// Soft-body prediction, shape matching, damping, and surface kernels.

__global__ void deformable_predict(Vec3 *positions, Vec3 *previous,
                              Vec3 *velocities, const float *inverse_masses,
                              std::uint32_t count, Vec3 gravity, float dt,
                              float damping) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    const Vec3 old = positions[vertex];
    previous[vertex] = old;
    if (inverse_masses[vertex] == 0.0F) {
        velocities[vertex] = {};
        return;
    }
    Vec3 velocity = multiply(
        velocities[vertex], 1.0F / (1.0F + damping * dt));
    velocity = add(velocity, multiply(gravity, dt));
    positions[vertex] = add(old, multiply(velocity, dt));
    velocities[vertex] = velocity;
}

__global__ void soft_body_measure_momentum(
    const Vec3 *velocities, const float *inverse_masses,
    std::uint32_t count, Vec3 *output) {
    __shared__ Vec3 values[128];
    Vec3 local{};
    for (std::uint32_t node = threadIdx.x; node < count;
         node += blockDim.x) {
        const float inverse_mass = inverse_masses[node];
        if (inverse_mass > 0.0F)
            local = add(local, multiply(velocities[node], 1.0F / inverse_mass));
    }
    values[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0U;
         stride /= 2U) {
        if (threadIdx.x < stride)
            values[threadIdx.x] = add(
                values[threadIdx.x], values[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0U) *output = values[0];
}

__global__ void soft_body_restore_momentum(
    Vec3 *velocities, const float *inverse_masses,
    const Vec3 *contact_momentum_delta, std::uint32_t count,
    float movable_mass, const Vec3 *predicted_momentum,
    const std::uint32_t *dynamic_contact_flag, float maximum_speed) {
    if (*dynamic_contact_flag == 0U) return;
    __shared__ Vec3 actual_values[128];
    __shared__ Vec3 contact_values[128];
    __shared__ Vec3 correction;
    Vec3 actual{}, contact{};
    for (std::uint32_t node = threadIdx.x; node < count;
         node += blockDim.x) {
        const float inverse_mass = inverse_masses[node];
        if (inverse_mass <= 0.0F) continue;
        actual = add(actual,
            multiply(velocities[node], 1.0F / inverse_mass));
        contact = add(contact, contact_momentum_delta[node]);
    }
    actual_values[threadIdx.x] = actual;
    contact_values[threadIdx.x] = contact;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride != 0U;
         stride /= 2U) {
        if (threadIdx.x < stride) {
            actual_values[threadIdx.x] = add(
                actual_values[threadIdx.x],
                actual_values[threadIdx.x + stride]);
            contact_values[threadIdx.x] = add(
                contact_values[threadIdx.x],
                contact_values[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        const Vec3 target = add(*predicted_momentum, contact_values[0]);
        correction = multiply(
            subtract(target, actual_values[0]), 1.0F / movable_mass);
    }
    __syncthreads();
    for (std::uint32_t node = threadIdx.x; node < count;
         node += blockDim.x) {
        if (inverse_masses[node] > 0.0F)
            velocities[node] = clamp_length(
                add(velocities[node], correction), maximum_speed);
    }
}

__device__ ShapeMatrix shape_matrix_multiply(
    const ShapeMatrix &first, const ShapeMatrix &second) {
    ShapeMatrix result{};
    for (std::uint32_t column = 0U; column < 3U; ++column) {
        const Vec3 weights = second.columns[column];
        result.columns[column] = add(
            multiply(first.columns[0], weights.x),
            add(multiply(first.columns[1], weights.y),
                multiply(first.columns[2], weights.z)));
    }
    return result;
}

// Keep the center, transform, and momentum sums in their original order while
// projecting independent nodes in parallel. The current center of mass and
// best-fit rotation keep this goal free to translate and roll.
__global__ void soft_body_project_rest_shape(
    Vec3 *positions, Vec3 *corrections, const Vec3 *rest_positions,
    const float *inverse_masses, std::uint32_t count, float movable_mass,
    Vec3 rest_center, ShapeMatrix inverse_rest,
    Quaternion *stored_orientation, float stiffness,
    float maximum_projection, const std::uint32_t *dynamic_contact_flag) {
    if (blockIdx.x != 0U || stiffness <= 0.0F ||
        *dynamic_contact_flag != 0U) return;
    __shared__ Vec3 shared_center, shared_center_correction;
    __shared__ Quaternion shared_orientation;
    __shared__ Vec3 batch_positions[128], batch_rest_positions[128];
    __shared__ float batch_inverse_masses[128];
    Vec3 current_center{};
    for (std::uint32_t base = 0U; base < count; base += blockDim.x) {
        const std::uint32_t node = base + threadIdx.x;
        if (node < count) {
            batch_positions[threadIdx.x] = positions[node];
            batch_inverse_masses[threadIdx.x] = inverse_masses[node];
        }
        __syncthreads();
        if (threadIdx.x == 0U) {
            const std::uint32_t valid = min(blockDim.x, count - base);
            for (std::uint32_t item = 0U; item < valid; ++item) {
                const float inverse_mass = batch_inverse_masses[item];
                if (inverse_mass <= 0.0F) continue;
                current_center = add(current_center,
                    multiply(batch_positions[item], 1.0F / inverse_mass));
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        current_center = multiply(current_center, 1.0F / movable_mass);
        shared_center = current_center;
    }
    __syncthreads();

    ShapeMatrix covariance{};
    for (std::uint32_t base = 0U; base < count; base += blockDim.x) {
        const std::uint32_t node = base + threadIdx.x;
        if (node < count) {
            batch_positions[threadIdx.x] = positions[node];
            batch_rest_positions[threadIdx.x] = rest_positions[node];
            batch_inverse_masses[threadIdx.x] = inverse_masses[node];
        }
        __syncthreads();
        if (threadIdx.x == 0U) {
            const std::uint32_t valid = min(blockDim.x, count - base);
            for (std::uint32_t item = 0U; item < valid; ++item) {
                const float inverse_mass = batch_inverse_masses[item];
                if (inverse_mass <= 0.0F) continue;
                const float mass = 1.0F / inverse_mass;
                const Vec3 current = subtract(
                    batch_positions[item], current_center);
                const Vec3 rest = subtract(
                    batch_rest_positions[item], rest_center);
                covariance.columns[0] = add(covariance.columns[0],
                    multiply(current, mass * rest.x));
                covariance.columns[1] = add(covariance.columns[1],
                    multiply(current, mass * rest.y));
                covariance.columns[2] = add(covariance.columns[2],
                    multiply(current, mass * rest.z));
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        const ShapeMatrix deformation = shape_matrix_multiply(
            covariance, inverse_rest);
        Quaternion orientation = normalized_quaternion(*stored_orientation);
        if (orientation.x == 0.0F && orientation.y == 0.0F &&
            orientation.z == 0.0F && orientation.w == 0.0F)
            orientation.w = 1.0F;
        for (std::uint32_t iteration = 0U; iteration < 12U; ++iteration) {
            const Vec3 axes[3]{
                rotate(orientation, {1.0F, 0.0F, 0.0F}),
                rotate(orientation, {0.0F, 1.0F, 0.0F}),
                rotate(orientation, {0.0F, 0.0F, 1.0F})};
            Vec3 angular = add(cross(axes[0], deformation.columns[0]),
                add(cross(axes[1], deformation.columns[1]),
                    cross(axes[2], deformation.columns[2])));
            const float denominator = fabsf(
                dot(axes[0], deformation.columns[0]) +
                dot(axes[1], deformation.columns[1]) +
                dot(axes[2], deformation.columns[2])) + 1.0e-9F;
            angular = multiply(angular, 1.0F / denominator);
            const float magnitude = vector_length(angular);
            if (magnitude < 1.0e-6F) break;
            const float angle = fminf(magnitude, 0.5F);
            const float half = 0.5F * angle;
            const Vec3 axis = multiply(angular, 1.0F / magnitude);
            const Quaternion delta{axis.x * sinf(half), axis.y * sinf(half),
                                   axis.z * sinf(half), cosf(half)};
            orientation = normalized_quaternion(
                quaternion_multiply(delta, orientation));
        }
        *stored_orientation = orientation;
        shared_orientation = orientation;
    }
    __syncthreads();
    for (std::uint32_t node = threadIdx.x; node < count;
         node += blockDim.x) {
        const float inverse_mass = inverse_masses[node];
        if (inverse_mass <= 0.0F) continue;
        const Vec3 target = add(shared_center, rotate(shared_orientation,
            subtract(rest_positions[node], rest_center)));
        corrections[node] = clamp_length(multiply(
            subtract(target, positions[node]), stiffness),
            maximum_projection);
    }
    __syncthreads();
    Vec3 weighted_correction{};
    for (std::uint32_t base = 0U; base < count; base += blockDim.x) {
        const std::uint32_t node = base + threadIdx.x;
        if (node < count) {
            batch_positions[threadIdx.x] = corrections[node];
            batch_inverse_masses[threadIdx.x] = inverse_masses[node];
        }
        __syncthreads();
        if (threadIdx.x == 0U) {
            const std::uint32_t valid = min(blockDim.x, count - base);
            for (std::uint32_t item = 0U; item < valid; ++item) {
                const float inverse_mass = batch_inverse_masses[item];
                if (inverse_mass <= 0.0F) continue;
                weighted_correction = add(weighted_correction,
                    multiply(batch_positions[item], 1.0F / inverse_mass));
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        shared_center_correction = multiply(
            weighted_correction, 1.0F / movable_mass);
    }
    __syncthreads();
    for (std::uint32_t node = threadIdx.x; node < count;
         node += blockDim.x) {
        if (inverse_masses[node] <= 0.0F) continue;
        positions[node] = add(positions[node],
            subtract(corrections[node], shared_center_correction));
    }
}

__global__ void deformable_project_links(
    const Vec3 *positions, Vec3 *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const DeformableNeighbor *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float dt,
    float maximum_projection_fraction) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= count) return;
    const float self_mass = inverse_masses[vertex];
    const Vec3 position = positions[vertex];
    if (self_mass == 0.0F) {
        scratch[vertex] = position;
        return;
    }
    Vec3 correction{};
    float shortest_rest_length = FLT_MAX;
    const std::uint32_t first = offsets[vertex];
    const std::uint32_t last = offsets[vertex + 1U];
    for (std::uint32_t edge = first; edge < last; ++edge) {
        const DeformableNeighbor neighbor = neighbors[edge];
        if (bond_active != nullptr && neighbor.bond != k_invalid_dense &&
            bond_active[neighbor.bond] == 0U) continue;
        shortest_rest_length = fminf(shortest_rest_length,
                                     neighbor.rest_length);
        const Vec3 difference = subtract(position, positions[neighbor.index]);
        const float length = vector_length(difference);
        if (length < 1.0e-7F) continue;
        const float other_mass = inverse_masses[neighbor.index];
        const float denominator = self_mass + other_mass +
            neighbor.compliance / (dt * dt);
        const float amount = -self_mass * (length - neighbor.rest_length) /
            (denominator * length);
        correction = add(correction, multiply(difference, amount));
    }
    // Keep the rest-graph degree so a broken bond cannot make its surviving
    // neighbors abruptly stiffer and start a fracture cascade.
    const float divisor = static_cast<float>(max(1U, last - first));
    Vec3 proposal = multiply(correction, 1.0F / divisor);
    if (maximum_projection_fraction > 0.0F &&
        shortest_rest_length < FLT_MAX) {
        proposal = clamp_length(
            proposal, maximum_projection_fraction * shortest_rest_length);
    }
    scratch[vertex] = add(position, proposal);
}

// Dense soft-body lattices have hundreds of neighbors per node but often only
// a few hundred nodes. Give each node a warp subgroup so independent bond reads
// fill the GPU while lane leaders fold proposals concurrently. CSR keeps
// each group's neighbor descriptors contiguous and in their original order.
__global__ void deformable_project_links_warp(
    const Vec3 *positions, Vec3 *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const SoftBodyNeighbor *neighbors,
    const float *minimum_rest_lengths, std::uint32_t count,
    float compliance, float dt,
    float maximum_projection_fraction) {
    constexpr std::uint32_t group_size = 16U;
    const std::uint32_t lane = threadIdx.x & (group_size - 1U);
    const std::uint32_t group =
        blockIdx.x * (blockDim.x / group_size) + threadIdx.x / group_size;
    if (group >= count) return;
    const unsigned group_mask = 0xffffU << (threadIdx.x & 16U);

    const float self_mass = inverse_masses[group];
    const Vec3 position = positions[group];
    if (self_mass == 0.0F) {
        if (lane == 0U) scratch[group] = position;
        return;
    }

    __shared__ Vec3 batch_corrections[128];
    const std::uint32_t first = offsets[group];
    const std::uint32_t last = offsets[group + 1U];
    Vec3 correction{};
    for (std::uint32_t base = first; base < last; base += group_size) {
        const std::uint32_t edge = base + lane;
        Vec3 edge_correction{};
        if (edge < last) {
            const SoftBodyNeighbor neighbor = neighbors[edge];
            const Vec3 difference = subtract(
                position, positions[neighbor.index]);
            const float length = vector_length(difference);
            if (length >= 1.0e-7F) {
                const float denominator = self_mass +
                    inverse_masses[neighbor.index] + compliance / (dt * dt);
                const float amount = -self_mass *
                    (length - neighbor.rest_length) / (denominator * length);
                edge_correction = multiply(difference, amount);
            }
        }
        batch_corrections[threadIdx.x] = edge_correction;
        __syncwarp(group_mask);
        if (lane == 0U) {
            const std::uint32_t valid = min(group_size, last - base);
            const std::uint32_t batch = threadIdx.x;
            for (std::uint32_t item = 0U; item < valid; ++item)
                correction = add(correction, batch_corrections[batch + item]);
        }
        __syncwarp(group_mask);
    }
    if (lane != 0U) return;
    const float divisor = static_cast<float>(max(1U, last - first));
    Vec3 proposal = multiply(correction, 1.0F / divisor);
    if (maximum_projection_fraction > 0.0F)
        proposal = clamp_length(
            proposal, maximum_projection_fraction * minimum_rest_lengths[group]);
    scratch[group] = add(position, proposal);
}

// Give each vertex its incident corners in face order. Triangles and vertices
// then evaluate independently, while scalar volume/mass sums retain their
// original order (and do not require unordered floating-point atomics).

__global__ void soft_body_finalize_velocities(
    const Vec3 *positions, const Vec3 *previous, Vec3 *velocities,
    const float *inverse_masses, std::uint32_t count, float inverse_dt,
    float projection_response, float maximum_speed) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    if (inverse_masses[node] == 0.0F) {
        velocities[node] = {};
        return;
    }
    const Vec3 projected_velocity = multiply(
        subtract(positions[node], previous[node]), inverse_dt);
    Vec3 velocity = add(velocities[node], multiply(
        subtract(projected_velocity, velocities[node]), projection_response));
    velocities[node] = clamp_length(velocity, maximum_speed);
}

__global__ void soft_body_apply_contact_friction(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    Vec3 *contact_forces, const Vec3 *contact_normals,
    const Vec3 *contact_arms, FluidBodyImpulse *body_impulses,
    const BodyParameters *body_parameters,
    Vec3 *contact_momentum_delta,
    Vec3 *accumulated_friction_delta,
    const float *accumulated_normal_delta,
    const std::uint32_t *contact_count, std::uint32_t count,
    float movable_mass, Vec3 gravity, float dt, float contact_friction,
    float maximum_speed) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count || inverse_masses[node] == 0.0F ||
        contact_friction <= 0.0F) return;
    const std::uint32_t supported_nodes = *contact_count;
    if (supported_nodes == 0U) return;
    const float normal_squared = length_squared(contact_normals[node]);
    if (normal_squared <= 1.0e-10F) return;
    const std::uint32_t body_index = body_impulses[node].body;
    if (body_index == k_invalid_dense) return;
    const Vec3 normal = multiply(
        contact_normals[node], rsqrtf(normal_squared));
    const Vec3 original_velocity = velocities[node];
    Vec3 velocity = original_velocity;
    const Vec3 force = contact_forces[node];
    const float force_squared = length_squared(force);
    const Vec3 normal_velocity = multiply(normal, dot(velocity, normal));
    const Vec3 tangent = subtract(velocity, normal_velocity);
    const float impulse_acceleration = force_squared > 1.0e-10F
        ? sqrtf(force_squared) * inverse_masses[node] : 0.0F;
    // Contacts are generated before the graph solve, so a surface node's
    // direct impulse contains only its own predicted weight. Distribute the
    // whole movable mass across the active support nodes; otherwise a dense
    // lattice slides because its interior load never reaches friction.
    const float supported_acceleration = fabsf(dot(gravity, normal)) *
        movable_mass * inverse_masses[node] /
        static_cast<float>(supported_nodes);
    const float normal_acceleration =
        fmaxf(impulse_acceleration, supported_acceleration);
    const float constraint_delta = accumulated_normal_delta != nullptr
        ? accumulated_normal_delta[node] / dt : 0.0F;
    const BodyParameters body = body_parameters[body_index];
    const float friction = body.motion == MotionType::dynamic
        ? sqrtf(contact_friction * body.friction)
        : contact_friction;
    const float maximum_delta = friction *
        fmaxf(normal_acceleration * dt, constraint_delta);
    Vec3 accumulated = accumulated_friction_delta[node];
    accumulated = subtract(accumulated,
        multiply(normal, dot(accumulated, normal)));
    Vec3 next_accumulated = subtract(accumulated, tangent);
    next_accumulated = clamp_length(next_accumulated, maximum_delta);
    const Vec3 applied_delta = subtract(next_accumulated, accumulated);
    velocity = add(velocity, applied_delta);
    velocity = clamp_length(velocity, maximum_speed);
    positions[node] = add(positions[node], multiply(
        subtract(velocity, original_velocity), dt));
    velocities[node] = velocity;
    accumulated_friction_delta[node] = next_accumulated;
    const float mass = 1.0F / inverse_masses[node];
    const Vec3 node_impulse = multiply(applied_delta, mass);
    contact_momentum_delta[node] = add(
        contact_momentum_delta[node], node_impulse);
    contact_forces[node] = add(
        contact_forces[node], multiply(node_impulse, 1.0F / dt));
    const Vec3 reaction = multiply(node_impulse, -1.0F);
    body_impulses[node].linear = add(
        body_impulses[node].linear, reaction);
    body_impulses[node].angular = add(
        body_impulses[node].angular,
        cross(contact_arms[node], reaction));
}

__global__ void soft_body_damp_springs(
    const Vec3 *positions, const Vec3 *velocities, Vec3 *output,
    const float *inverse_masses, const std::uint32_t *offsets,
    const DeformableNeighbor *neighbors, const std::uint8_t *bond_active,
    std::uint32_t count, float damping, float maximum_speed) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= count) return;
    if (inverse_masses[node] == 0.0F) {
        output[node] = {};
        return;
    }
    Vec3 correction{};
    std::uint32_t active_count = 0U;
    for (std::uint32_t item = offsets[node]; item < offsets[node + 1U]; ++item) {
        const DeformableNeighbor neighbor = neighbors[item];
        if (bond_active[neighbor.bond] == 0U) continue;
        const Vec3 axis = normalized_or(
            subtract(positions[neighbor.index], positions[node]), {});
        correction = add(correction, multiply(axis,
            dot(subtract(velocities[neighbor.index], velocities[node]), axis)));
        ++active_count;
    }
    if (active_count != 0U)
        correction = multiply(correction,
            0.5F * damping / static_cast<float>(active_count));
    output[node] = clamp_length(add(velocities[node], correction), maximum_speed);
}

__global__ void soft_body_damp_springs_warp(
    const Vec3 *positions, const Vec3 *velocities, Vec3 *output,
    const float *inverse_masses, const std::uint32_t *offsets,
    const SoftBodyNeighbor *neighbors, std::uint32_t count,
    float damping, float maximum_speed) {
    constexpr std::uint32_t group_size = 16U;
    const std::uint32_t lane = threadIdx.x & (group_size - 1U);
    const std::uint32_t group =
        blockIdx.x * (blockDim.x / group_size) + threadIdx.x / group_size;
    if (group >= count) return;
    const unsigned group_mask = 0xffffU << (threadIdx.x & 16U);
    if (inverse_masses[group] == 0.0F) {
        if (lane == 0U) output[group] = {};
        return;
    }

    __shared__ Vec3 batch_corrections[128];
    const std::uint32_t first = offsets[group];
    const std::uint32_t last = offsets[group + 1U];
    const Vec3 position = positions[group];
    const Vec3 velocity = velocities[group];
    Vec3 correction{};
    for (std::uint32_t base = first; base < last; base += group_size) {
        const std::uint32_t edge = base + lane;
        Vec3 edge_correction{};
        if (edge < last) {
            const SoftBodyNeighbor neighbor = neighbors[edge];
            const Vec3 axis = normalized_or(
                subtract(positions[neighbor.index], position), {});
            edge_correction = multiply(axis, dot(
                subtract(velocities[neighbor.index], velocity), axis));
        }
        batch_corrections[threadIdx.x] = edge_correction;
        __syncwarp(group_mask);
        if (lane == 0U) {
            const std::uint32_t valid = min(group_size, last - base);
            const std::uint32_t batch = threadIdx.x;
            for (std::uint32_t item = 0U; item < valid; ++item)
                correction = add(correction, batch_corrections[batch + item]);
        }
        __syncwarp(group_mask);
    }
    if (lane != 0U) return;
    const std::uint32_t active_count = last - first;
    if (active_count != 0U)
        correction = multiply(correction,
            0.5F * damping / static_cast<float>(active_count));
    output[group] = clamp_length(add(velocity, correction), maximum_speed);
}

__global__ void soft_body_update_surface(
    const Vec3 *positions, const Vec3 *rest_positions,
    const Vec3 *surface_rest_positions,
    const SoftBodySurfaceBinding *bindings, Vec3 *surface_positions,
    std::uint32_t surface_vertex_count) {
    const std::uint32_t vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= surface_vertex_count) return;
    const SoftBodySurfaceBinding binding = bindings[vertex];
    Vec3 delta{};
    for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
        if (binding.weights[slot] == 0.0F) continue;
        delta = add(delta, multiply(subtract(positions[binding.nodes[slot]],
                                             rest_positions[binding.nodes[slot]]),
                                    binding.weights[slot]));
    }
    surface_positions[vertex] = add(surface_rest_positions[vertex], delta);
}

// Detection reads stable position buffers. Integer contact counts provide a
// shared relaxation for both sides; the later gathers use no float atomics.
