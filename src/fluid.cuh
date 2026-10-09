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

// Core fluid cell indexing, force evaluation, and integration are generated
// from src/slang/fluid.slang. Lifecycle kernels remain native below.

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
