// SPDX-License-Identifier: MIT
// Fluid simulation, source, contact-event, and lifecycle kernels.

__host__ __device__ std::uint64_t fluid_cell_key(int x, int y, int z) noexcept {
    x = max(-k_fluid_cell_bias, min(k_fluid_cell_bias - 1, x));
    y = max(-k_fluid_cell_bias, min(k_fluid_cell_bias - 1, y));
    z = max(-k_fluid_cell_bias, min(k_fluid_cell_bias - 1, z));
    return (static_cast<std::uint64_t>(x + k_fluid_cell_bias) << 42U) |
           (static_cast<std::uint64_t>(y + k_fluid_cell_bias) << 21U) |
           static_cast<std::uint64_t>(z + k_fluid_cell_bias);
}

__device__ std::uint32_t fluid_lower_bound(const std::uint64_t *keys,
                                           std::uint32_t size,
                                           std::uint64_t key) noexcept {
    std::uint32_t lo = 0U, hi = size;
    while (lo < hi) {
        const std::uint32_t middle = lo + (hi - lo) / 2U;
        if (keys[middle] < key) lo = middle + 1U;
        else hi = middle;
    }
    return lo;
}

__global__ void fluid_emit_cells(const Vec3 *positions, const std::uint32_t *count,
                                 std::uint32_t capacity, float inverse_radius,
                                 std::uint64_t *keys,
                                 std::uint32_t *indices) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= capacity) return;
    indices[index] = index;
    if (index >= *count) {
        keys[index] = k_fluid_empty_cell;
        return;
    }
    const Vec3 position = positions[index];
    keys[index] = fluid_cell_key(
        __float2int_rd(position.x * inverse_radius),
        __float2int_rd(position.y * inverse_radius),
        __float2int_rd(position.z * inverse_radius));
}

__global__ void fluid_compute_forces(
    const Vec3 *positions, const Vec3 *velocities, const std::uint32_t *count,
    const std::uint64_t *keys, const std::uint32_t *indices,
    const float *foam, FluidOptions options, Vec3 up,
    Vec3 *forces, float *foam_source, std::uint32_t *overflow,
    std::uint32_t *maximum_neighbor_count) {
    const std::uint32_t particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= *count) return;
    const Vec3 p = positions[particle], v = velocities[particle];
    const float inverse_radius = 1.0F / options.support_radius;
    const int cx = __float2int_rd(p.x * inverse_radius);
    const int cy = __float2int_rd(p.y * inverse_radius);
    const int cz = __float2int_rd(p.z * inverse_radius);
    Vec3 acceleration{};
    Vec3 outward{};
    float weight = 0.0F;
    float relative_speed_squared = 0.0F;
    float neighboring_foam = 0.0F;
    std::uint32_t neighbors = 0U;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (cx + dx < -k_fluid_cell_bias ||
                    cx + dx >= k_fluid_cell_bias ||
                    cy + dy < -k_fluid_cell_bias ||
                    cy + dy >= k_fluid_cell_bias ||
                    cz + dz < -k_fluid_cell_bias ||
                    cz + dz >= k_fluid_cell_bias) continue;
                const std::uint64_t key = fluid_cell_key(cx + dx, cy + dy, cz + dz);
                for (std::uint32_t item = fluid_lower_bound(keys, options.capacity, key);
                     item < options.capacity && keys[item] == key; ++item) {
                    const std::uint32_t other = indices[item];
                    if (other == particle) continue;
                    const Vec3 delta = subtract(p, positions[other]);
                    const float squared = length_squared(delta);
                    if (squared >= options.support_radius * options.support_radius)
                        continue;
                    ++neighbors;
                    const float distance = sqrtf(fmaxf(squared, 1.0e-12F));
                    // Stable fallback separates coincident particles without NaNs.
                    const Vec3 direction = squared > 1.0e-12F
                        ? multiply(delta, 1.0F / distance)
                        : (particle < other ? Vec3{-1.0F, 0.0F, 0.0F}
                                            : Vec3{1.0F, 0.0F, 0.0F});
                    const float q = 1.0F - distance * inverse_radius;
                    outward = add(outward, multiply(direction, q));
                    weight += q;
                    const Vec3 relative_velocity =
                        subtract(velocities[other], v);
                    relative_speed_squared +=
                        length_squared(relative_velocity) * q;
                    neighboring_foam = fmaxf(neighboring_foam,
                                               foam[other] * q);
                    const float radial_speed = dot(relative_velocity, direction);
                    acceleration = add(acceleration,
                        add(multiply(direction, options.repulsion *
                            (1'000.0F / options.rest_density) * q * q +
                            options.normal_damping * radial_speed),
                            multiply(relative_velocity,
                                     options.viscosity * q)));
                }
            }
        }
    }
    if (options.maximum_pair_acceleration > 0.0F)
        acceleration = clamp_length(acceleration,
                                    options.maximum_pair_acceleration);
    forces[particle] = acceleration;
    const float exposure = vector_length(outward) / fmaxf(weight, 1.0e-6F);
    const float upward = fmaxf(0.0F, dot(normalized_or(outward, up), up));
    const float agitation = sqrtf(relative_speed_squared /
                                  fmaxf(weight, 1.0e-6F));
    foam_source[particle] = fmaxf(
        clamp_scalar((exposure - 0.12F) * 2.0F, 0.0F, 1.0F) * upward *
            clamp_scalar((agitation - 0.15F) * 1.5F, 0.0F, 1.0F),
        neighboring_foam * upward * 0.9F);
    atomicMax(maximum_neighbor_count, neighbors);
    if (neighbors > options.maximum_neighbors) atomicAdd(overflow, 1U);
}

__global__ void fluid_integrate(Vec3 *positions, Vec3 *velocities,
                                Vec3 *previous, float *foam,
                                const Vec3 *forces, const float *foam_source,
                                const std::uint32_t *count,
                                FluidOptions options, Vec3 gravity, float dt) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= *count) return;
    previous[index] = positions[index];
    Vec3 velocity = add(velocities[index],
                        multiply(add(gravity, forces[index]), dt));
    velocity = clamp_length(
        multiply(velocity, expf(-options.velocity_damping * dt)),
        options.maximum_speed);
    const Vec3 position = add(positions[index], multiply(velocity, dt));
    if (isfinite(position.x) && isfinite(position.y) && isfinite(position.z) &&
        isfinite(velocity.x) && isfinite(velocity.y) && isfinite(velocity.z)) {
        positions[index] = position;
        velocities[index] = velocity;
    }
    foam[index] = fmaxf(fmaxf(0.0F, foam[index] - dt * 0.7F),
                        foam_source[index]);
}

__global__ void fluid_reserve_contact_events(
    const std::uint32_t *selected_count, std::uint32_t *offset,
    std::uint32_t *world_count, std::uint32_t *overflow,
    std::uint32_t capacity) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    const std::uint32_t used = *world_count;
    const std::uint32_t available = capacity - used;
    const std::uint32_t retained = min(*selected_count, available);
    *offset = used;
    *world_count = used + retained;
    *overflow += *selected_count - retained;
}

__global__ void fluid_gather_contact_events(
    const std::uint32_t *selected, const std::uint32_t *selected_count,
    const std::uint32_t *offset, const FluidContactSample *samples,
    const std::uint32_t *stable_ids, const RigidBodyId *body_ids,
    FluidId fluid, ContactEvent *events, std::uint32_t capacity) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= *selected_count || *offset + item >= capacity) return;
    const std::uint32_t particle = selected[item];
    const FluidContactSample sample = samples[particle];
    events[*offset + item] = {fluid, stable_ids[particle],
        body_ids[sample.body], sample.position, sample.normal,
        sample.normal_impulse};
}

__global__ void fluid_source_vacancies(
    const Vec3 *points, std::uint32_t amount, float spacing,
    const Vec3 *positions, const std::uint64_t *keys,
    const std::uint32_t *indices, std::uint32_t capacity,
    float inverse_cell_size, std::uint8_t *vacant) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= amount) return;
    const Vec3 p = points[item];
    const int cx = max(-k_fluid_cell_bias, min(k_fluid_cell_bias-1, __float2int_rd(p.x*inverse_cell_size)));
    const int cy = max(-k_fluid_cell_bias, min(k_fluid_cell_bias-1, __float2int_rd(p.y*inverse_cell_size)));
    const int cz = max(-k_fluid_cell_bias, min(k_fluid_cell_bias-1, __float2int_rd(p.z*inverse_cell_size)));
    vacant[item] = 0;
    for(int z=-1;z<=1;++z) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
        const auto key = fluid_cell_key(cx+x, cy+y, cz+z);
        for(auto i=fluid_lower_bound(keys,capacity,key);i<capacity && keys[i]==key;++i)
            if(length_squared(subtract(p,positions[indices[i]])) < spacing*spacing)
                return;
    }
    vacant[item] = 1;
}

// A bounded deterministic commit pass avoids duplicate emission at seams and
// intersecting sources. Vacancies above are tested in parallel against the
// spatial index; only particles appended since that index need a direct check.
__global__ void fluid_source_emit(
    const Vec3 *points, const std::uint8_t *vacant, std::uint32_t amount,
    float spacing, Vec3 velocity, Vec3 *positions, Vec3 *velocities,
    std::uint32_t *ids, float *foam, float *temperatures,
    float initial_temperature, std::uint32_t *count,
    std::uint32_t capacity, std::uint32_t first_spawned,
    std::uint32_t first_id, std::uint32_t *misses) {
    if(threadIdx.x || blockIdx.x) return;
    *misses=0;
    const auto previous_count=*count;
    for(std::uint32_t item=0;item<amount;++item) {
        if(!vacant[item]) continue;
        const Vec3 p=points[item];
        bool occupied=false;
        // This source's sites are already separated by the host sampler.
        // Only earlier sources need a cross-source conflict check.
        for(std::uint32_t i=first_spawned;i<previous_count;++i)
            if(length_squared(subtract(p,positions[i])) < spacing*spacing) {occupied=true;break;}
        if(occupied) continue;
        if(*count==capacity) {++*misses;continue;}
        const auto index=(*count)++;
        positions[index]=p;
        velocities[index]=velocity;
        ids[index]=first_id+index-first_spawned;
        foam[index]=0;
        temperatures[index]=initial_temperature;
    }
}

__global__ void fluid_destroy_flags(const Vec3 *positions,
                                    const Vec3 *previous,
                                    const std::uint32_t *count,
                                    std::uint32_t capacity,
                                    ParticleDestroyPlaneOptions plane,
                                    bool combine,
                                    std::uint8_t *keep) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= capacity) return;
    if (item >= *count) { keep[item] = 0U; return; }
    const Vec3 before = inverse_rotate(plane.plane.orientation,
        subtract(previous[item], plane.plane.center));
    const Vec3 after = inverse_rotate(plane.plane.orientation,
        subtract(positions[item], plane.plane.center));
    const bool along = before.y < 0.0F && after.y >= 0.0F;
    const bool against = before.y > 0.0F && after.y <= 0.0F;
    const bool direction = plane.crossing == CrossingDirection::either
        ? along || against
        : plane.crossing == CrossingDirection::along_normal ? along : against;
    const float fraction = before.y / (before.y - after.y + 1.0e-20F);
    const Vec3 crossing = add(before, multiply(subtract(after, before), fraction));
    const bool in_bounds = fabsf(crossing.x) <= plane.plane.half_extents.x &&
                           fabsf(crossing.z) <= plane.plane.half_extents.y;
    const std::uint8_t survives = static_cast<std::uint8_t>(!direction || !in_bounds);
    keep[item] = combine ? keep[item] & survives : survives;
}

__global__ void fluid_gather(const Vec3 *positions, const Vec3 *velocities,
                             const std::uint32_t *ids, const float *foam,
                             const float *temperatures,
                             const FluidContactSample *contact_samples,
                             const std::uint8_t *contact_flags,
                             bool copy_contacts,
                             const std::uint32_t *selected,
                             const std::uint32_t *count,
                             std::uint32_t capacity, Vec3 *next_positions,
                             Vec3 *next_velocities, std::uint32_t *next_ids,
                             float *next_foam, float *next_temperatures,
                             FluidContactSample *next_contact_samples,
                             std::uint8_t *next_contact_flags) {
    const std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= capacity || item >= *count) return;
    const std::uint32_t source = selected[item];
    next_positions[item] = positions[source];
    next_velocities[item] = velocities[source];
    next_ids[item] = ids[source];
    next_foam[item] = foam[source];
    next_temperatures[item] = temperatures[source];
    if (copy_contacts) {
        next_contact_samples[item] = contact_samples[source];
        next_contact_flags[item] = contact_flags[source];
    }
}

__global__ void fluid_copy_initial(const FluidParticle *input,
                                   std::uint32_t count, Vec3 *positions,
                                   Vec3 *velocities, std::uint32_t *ids,
                                   float *foam, float *temperatures,
                                   std::uint32_t *invalid) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    if (!isfinite(input[index].position.x) ||
        !isfinite(input[index].position.y) ||
        !isfinite(input[index].position.z) ||
        !isfinite(input[index].velocity.x) ||
        !isfinite(input[index].velocity.y) ||
        !isfinite(input[index].velocity.z) ||
        !isfinite(input[index].temperature)) {
        atomicExch(invalid, 1U);
        return;
    }
    positions[index] = input[index].position;
    velocities[index] = input[index].velocity;
    ids[index] = index;
    foam[index] = 0.0F;
    temperatures[index] = input[index].temperature;
}
