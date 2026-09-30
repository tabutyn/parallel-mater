// SPDX-License-Identifier: MIT
// Water temperature, one-way carrier-gas drag, and liquid-to-smoke transfer.
// The gas is represented by a prescribed flow with tracer particles; this
// coupling samples that same field without an O(water * smoke) particle scan.
struct FluidSmokeCouplingSlot {
    FluidSmokeCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    std::uint32_t *converted{};
    ~FluidSmokeCouplingSlot() { release_managed(converted); }
    FluidSmokeCouplingSlot() = default;
    FluidSmokeCouplingSlot(const FluidSmokeCouplingSlot &) = delete;
    FluidSmokeCouplingSlot &operator=(const FluidSmokeCouplingSlot &) = delete;
};

__global__ void fluid_smoke_exchange(
    const Vec3 *positions, Vec3 *velocities, float *temperatures,
    const std::uint32_t *water_count, float water_radius,
    FluidSmokeCouplingOptions coupling, SmokeOptions smoke,
    Vec3 obstacle_center, float smoke_time, std::uint32_t smoke_count,
    Vec3 *smoke_positions, Vec3 *smoke_velocities, float *smoke_ages,
    float *smoke_thermal_lift, std::uint32_t first_smoke_slot,
    std::uint32_t *converted, std::uint8_t *keep, float dt,
    Vec3 gravity) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *water_count) return;
    const Vec3 position = positions[i];
    Vec3 velocity = velocities[i];
    if (smoke_count != 0U && coupling.wind_drag > 0.0F) {
        const Vec3 relative = subtract(position, smoke.emitter_center);
        const float travel = fmaxf(smoke.lifetime * fmaxf(smoke.wind.x, 0.1F),
                                   smoke.obstacle_radius * 2.0F);
        const float lateral = fmaxf(smoke.emitter_half_extents.x,
                                   smoke.emitter_half_extents.y) +
                              2.5F * smoke.obstacle_radius;
        if (relative.x >= -smoke.particle_radius && relative.x <= travel &&
            fabsf(relative.y) <= lateral && fabsf(relative.z) <= lateral) {
            const Vec3 gas_velocity = smoke_velocity_field(
                position, obstacle_center, smoke, smoke_time, gravity);
            const float response = 1.0F - expf(-coupling.wind_drag * dt);
            velocity = clamp_length(add(velocity, multiply(
                subtract(gas_velocity, velocity), response)),
                fmaxf(smoke.maximum_speed, sqrtf(length_squared(velocity))));
            velocities[i] = velocity;
        }
    }
    const Vec3 local = inverse_rotate(coupling.heater.orientation,
                                     subtract(position, coupling.heater.center));
    const bool heated = fabsf(local.x) <= coupling.heater.half_extents.x &&
        fabsf(local.z) <= coupling.heater.half_extents.y &&
        local.y >= -water_radius * 0.5F && local.y <= water_radius * 2.5F;
    float temperature = temperatures[i];
    if (heated && coupling.heater_temperature > temperature) {
        temperature += (coupling.heater_temperature - temperature) *
            (1.0F - expf(-coupling.heat_transfer_rate * dt));
    }
    temperatures[i] = temperature;
    keep[i] = 1U;
    if (temperature < coupling.boiling_temperature) return;
    const std::uint32_t ordinal = atomicAdd(converted, 1U);
    if (ordinal >= smoke.capacity) return; // retain excess water for next frame
    const std::uint32_t slot = (first_smoke_slot + ordinal) % smoke.capacity;
    smoke_positions[slot] = position;
    smoke_velocities[slot] = add(velocity, multiply(
        normalized_or(multiply(gravity, -1.0F), {0.0F, 1.0F, 0.0F}),
        coupling.steam_rise_speed));
    smoke_ages[slot] = 0.0F;
    smoke_thermal_lift[slot] = coupling.steam_rise_speed;
    keep[i] = 0U;
}

__global__ void fluid_clear_inactive_keep(std::uint8_t *keep,
                                          const std::uint32_t *count,
                                          std::uint32_t capacity) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    if (i >= *count) keep[i] = 0U;
}
