// SPDX-License-Identifier: MIT
// Dilute smoke tracers: a potential-flow sphere deflection plus a translating
// alternating vortex street. All integration and emission remain in the API.
struct SmokeStorage {
    SmokeOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    Vec3 *positions{};
    Vec3 *velocities{};
    float *ages{};
    float *thermal_lift{};
    std::uint32_t count{};
    std::uint32_t next_slot{};
    std::uint64_t emitted{};
    float emission_fraction{};
    float time{};
    ~SmokeStorage() {
        release_managed(positions);
        release_managed(velocities);
        release_managed(ages);
        release_managed(thermal_lift);
    }
};

__host__ __device__ std::uint32_t smoke_hash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

__device__ float smoke_random(std::uint32_t seed) {
    return float(smoke_hash(seed) & 0x00ffffffU) / 16777216.0F;
}

__device__ Vec3 smoke_velocity_field(Vec3 point, Vec3 center,
    const SmokeOptions &options, float time) {
    const Vec3 relative = subtract(point, center);
    const float radius = options.obstacle_radius;
    const float distance2 = fmaxf(length_squared(relative), radius * radius * 1.001F);
    const float distance = sqrtf(distance2);
    const float ratio = radius * radius * radius / (distance2 * distance);
    // Analytic incompressible potential flow has zero normal velocity at the
    // sphere, and accelerates the stream around its sides.
    Vec3 velocity = subtract(
        multiply(options.wind, 1.0F + 0.5F * ratio),
        multiply(relative, 1.5F * ratio * dot(options.wind, relative) / distance2));
    if (relative.x > 0.15F * radius) {
        const float spacing = 1.25F * radius;
        const float phase = fmodf(fmaxf(time * options.wind.x, 0.0F), spacing);
        const float width = radius;
        for (unsigned vortex = 0U; vortex < 5U; ++vortex) {
            const float sign = vortex & 1U ? -1.0F : 1.0F;
            const float dx = relative.x - (0.75F * radius +
                float(vortex) * spacing + phase);
            const float dy = relative.y - sign * 0.42F * radius;
            const float dz = relative.z - sign * 0.20F * radius;
            const float envelope = expf(-(dx * dx + dy * dy +
                0.4F * dz * dz) / (width * width));
            const float strength = sign * options.wake_strength * envelope / width;
            velocity = add(velocity, {strength * -dy, strength * dx,
                0.38F * strength * -dx});
        }
    }
    velocity.y += options.buoyancy;
    return velocity;
}

__global__ void smoke_advect(Vec3 *positions, Vec3 *velocities, float *ages,
    float *thermal_lift,
    std::uint32_t count, SmokeOptions options,
    const RigidBodyState *states, const RigidBodyState *previous_states,
    bool moving_body, std::uint32_t obstacle, float time, float dt) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || ages[index] >= options.lifetime) return;
    Vec3 point = positions[index];
    Vec3 velocity = velocities[index];
    const Vec3 center = states[obstacle].position;
    const Vec3 previous_center = moving_body
        ? previous_states[obstacle].position : center;
    const Vec3 center_path = subtract(center, previous_center);
    const Vec3 obstacle_velocity = multiply(center_path, 1.0F / dt);
    Vec3 desired = smoke_velocity_field(point, center, options, time);
    desired.y += thermal_lift[index];
    const float response = 1.0F - expf(-options.response * dt);
    velocity = clamp_length(add(velocity,
        multiply(subtract(desired, velocity), response)), options.maximum_speed);
    const Vec3 previous = point;
    point = add(point, multiply(velocity, dt));
    const float clearance = options.obstacle_radius + 0.15F * options.particle_radius;
    const Vec3 path = subtract(point, previous);
    // Solve the swept collision in the obstacle's translating frame.
    const Vec3 start = subtract(previous, previous_center);
    const Vec3 relative_path = subtract(path, center_path);
    const float path2 = length_squared(relative_path);
    const float projection = dot(start, relative_path);
    const float discriminant = projection * projection - path2 *
        (length_squared(start) - clearance * clearance);
    const bool swept_hit = length_squared(start) >= clearance * clearance &&
        path2 > 1.0e-12F && projection < 0.0F && discriminant >= 0.0F &&
        -projection - sqrtf(discriminant) <= path2;
    if (length_squared(subtract(point, center)) < clearance * clearance || swept_hit) {
        float collision_fraction = 1.0F;
        if (swept_hit) {
            collision_fraction = fmaxf(0.0F,
                (-projection - sqrtf(discriminant)) / path2);
            point = add(previous, multiply(path, collision_fraction));
        }
        const Vec3 contact_center = add(previous_center,
            multiply(center_path, collision_fraction));
        const Vec3 normal = normalized_or(subtract(point, contact_center),
            {-1.0F, 0.0F, 0.0F});
        point = add(contact_center, multiply(normal, clearance));
        Vec3 relative_velocity = subtract(velocity, obstacle_velocity);
        relative_velocity = subtract(relative_velocity,
            multiply(normal, fminf(0.0F, dot(relative_velocity, normal))));
        velocity = add(relative_velocity, obstacle_velocity);
        if (swept_hit)
            point = add(point, multiply(velocity,
                dt * (1.0F - collision_fraction)));
        const Vec3 final_separation = subtract(point, center);
        if (length_squared(final_separation) < clearance * clearance)
            point = add(center, multiply(normalized_or(final_separation, normal),
                clearance));
    }
    positions[index] = point;
    velocities[index] = velocity;
    ages[index] += dt;
    thermal_lift[index] *= expf(-dt / 3.0F);
}

__global__ void smoke_emit(Vec3 *positions, Vec3 *velocities, float *ages,
    float *thermal_lift,
    SmokeOptions options, std::uint32_t first_slot,
    std::uint32_t count, std::uint64_t first_serial) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const std::uint32_t slot = (first_slot + index) % options.capacity;
    const std::uint32_t serial = std::uint32_t(first_serial + index);
    const float y = 2.0F * smoke_random(serial ^ 0x132aef41U) - 1.0F;
    const float z = 2.0F * smoke_random(serial ^ 0xa385c9d3U) - 1.0F;
    positions[slot] = add(options.emitter_center, {0.0F,
        y * options.emitter_half_extents.x,
        z * options.emitter_half_extents.y});
    velocities[slot] = options.initial_velocity;
    ages[slot] = 0.0F;
    thermal_lift[slot] = 0.0F;
}
