// SPDX-License-Identifier: MIT
// Cloth volume, fracture, strain, and surface kernels.

constexpr std::uint32_t k_cloth_volume_threads = 128U;
__global__ void cloth_project_volume(
    Vec3 *positions, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t vertex_count,
    std::uint32_t triangle_count, const std::uint32_t *corner_offsets,
    const std::uint32_t *corner_indices, Vec3 *gradients, float target_volume,
    float orientation, float compliance, float dt, float *lambda,
    bool reset_lambda) {
    if (blockIdx.x != 0U) return;
    const std::uint32_t thread = threadIdx.x;
    __shared__ float scalar_terms[k_cloth_volume_threads];
    __shared__ float signed_volume, shared_delta_lambda;
    __shared__ bool apply_correction;
    if (thread == 0U) {
        if (reset_lambda) *lambda = 0.0F;
        signed_volume = 0.0F;
    }
    __syncthreads();
    for (std::uint32_t wave = 0U; wave < triangle_count;
         wave += blockDim.x) {
        const std::uint32_t triangle = wave + thread;
        if (triangle < triangle_count) {
            const std::uint32_t a_index = indices[3U * triangle];
            const std::uint32_t b_index = indices[3U * triangle + 1U];
            const std::uint32_t c_index = indices[3U * triangle + 2U];
            const Vec3 a = positions[a_index];
            const Vec3 b = positions[b_index];
            const Vec3 c = positions[c_index];
            scalar_terms[thread] = dot(a, cross(b, c)) / 6.0F;
        }
        __syncthreads();
        if (thread == 0U) {
            const std::uint32_t count = min(blockDim.x, triangle_count - wave);
            for (std::uint32_t item = 0U; item < count; ++item)
                signed_volume += scalar_terms[item];
        }
        __syncthreads();
    }
    for (std::uint32_t vertex = thread; vertex < vertex_count;
         vertex += blockDim.x) {
        Vec3 gradient{};
        for (std::uint32_t item = corner_offsets[vertex];
             item < corner_offsets[vertex + 1U]; ++item) {
            const std::uint32_t corner = corner_indices[item];
            const std::uint32_t base = corner - corner % 3U;
            const Vec3 a = positions[indices[base]];
            const Vec3 b = positions[indices[base + 1U]];
            const Vec3 c = positions[indices[base + 2U]];
            const Vec3 contribution = corner % 3U == 0U ? cross(b, c) :
                corner % 3U == 1U ? cross(c, a) : cross(a, b);
            gradient = add(gradient,
                multiply(contribution, orientation / 6.0F));
        }
        gradients[vertex] = gradient;
    }
    __syncthreads();
    if (thread == 0U) {
        apply_correction = false;
    }
    __syncthreads();
    float inverse_mass_sum = 0.0F;
    for (std::uint32_t wave = 0U; wave < vertex_count;
         wave += blockDim.x) {
        const std::uint32_t vertex = wave + thread;
        if (vertex < vertex_count)
            scalar_terms[thread] = inverse_masses[vertex] *
                length_squared(gradients[vertex]);
        __syncthreads();
        if (thread == 0U) {
            const std::uint32_t count = min(blockDim.x, vertex_count - wave);
            for (std::uint32_t item = 0U; item < count; ++item)
                inverse_mass_sum += scalar_terms[item];
        }
        __syncthreads();
    }
    if (thread == 0U) {
        apply_correction = inverse_mass_sum > 1.0e-12F;
        if (apply_correction) {
            const float alpha = compliance / (dt * dt);
            const float constraint = orientation * signed_volume - target_volume;
            shared_delta_lambda =
                (-constraint - alpha * *lambda) / (inverse_mass_sum + alpha);
            *lambda += shared_delta_lambda;
        }
    }
    __syncthreads();
    if (!apply_correction) return;
    for (std::uint32_t vertex = thread; vertex < vertex_count;
         vertex += blockDim.x) {
        if (inverse_masses[vertex] == 0.0F) continue;
        positions[vertex] = add(positions[vertex], multiply(
            gradients[vertex], inverse_masses[vertex] * shared_delta_lambda));
    }
}

__global__ void cloth_break_bonds(const Vec3 *positions,
    const ClothBond *bonds, std::uint8_t *active, std::uint8_t *damage,
    std::uint32_t count, float break_strain, std::uint32_t persistence,
    const FluidBodyImpulse *contact_impulses, float impact_threshold) {
    const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= count || active[edge] == 0U) return;
    const ClothBond bond = bonds[edge];
    if (impact_threshold > 0.0F && contact_impulses != nullptr &&
        vector_length(contact_impulses[bond.first].linear) +
        vector_length(contact_impulses[bond.second].linear) >
            impact_threshold) {
        active[edge] = 0U;
        damage[edge] = 0U;
        return;
    }
    if (break_strain <= 0.0F) return;
    const float length = vector_length(subtract(
        positions[bond.first], positions[bond.second]));
    if (!isfinite(length) || length <=
        bond.rest_length * (1.0F + break_strain)) {
        damage[edge] = 0U;
        return;
    }
    const std::uint32_t next = min(persistence,
        static_cast<std::uint32_t>(damage[edge]) + 1U);
    damage[edge] = static_cast<std::uint8_t>(next);
    if (next >= persistence) {
        active[edge] = 0U;
        damage[edge] = 0U;
    }
}

// Fracture releases seams, not the material within a face. Bound isolated
// triangles in physical space during contact, rather than fitting a render
// triangle to remote nodes. Do not erase attached material's tearing strain.
// One cooperative block performs
// deterministic Jacobi iterations without shared-vertex writes or host waits.
__global__ void cloth_limit_strain(Vec3 *positions, Vec3 *scratch,
    const float *inverse_masses, const std::uint32_t *offsets,
    const DeformableNeighbor *neighbors, const std::uint8_t *free_nodes,
    std::uint32_t count) {
    for (std::uint32_t pass = 0; pass < 32U; ++pass) {
        bool changed = false;
        for (std::uint32_t node = threadIdx.x; node < count; node += blockDim.x) {
            Vec3 correction{};
            std::uint32_t degree = 0U;
            for (auto item = offsets[node]; free_nodes[node] && item < offsets[node + 1U]; ++item) {
                const auto edge = neighbors[item];
                if (edge.bond != k_invalid_dense) continue;
                const float weight = inverse_masses[node] + inverse_masses[edge.index];
                const Vec3 delta = subtract(positions[edge.index], positions[node]);
                const float length = vector_length(delta);
                const float maximum = edge.rest_length * 1.10F;
                if (weight <= 0.0F || length <= maximum || inverse_masses[node] == 0.0F) continue;
                correction = add(correction, multiply(delta,
                    (length - maximum) * inverse_masses[node] / (length * weight)));
                changed |= length > maximum * 1.0001F;
                ++degree;
            }
            scratch[node] = add(positions[node], multiply(correction, 1.0F / max(1U,degree)));
        }
        __syncthreads();
        for (std::uint32_t node = threadIdx.x; node < count; node += blockDim.x)
            positions[node] = scratch[node];
        if (!__syncthreads_or(changed)) break;
    }
}

__global__ void cloth_update_surface(const Vec3 *positions,
    const std::uint32_t *source_indices,
    Vec3 *surface_positions, std::uint32_t triangle_count) {
    const std::uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= triangle_count) return;
    const std::uint32_t base = 3U * triangle;
    for (std::uint32_t corner = 0U; corner < 3U; ++corner)
        surface_positions[base + corner] = positions[source_indices[base + corner]];
}
