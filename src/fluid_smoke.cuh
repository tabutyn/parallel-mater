// SPDX-License-Identifier: MIT
// Water temperature, local smoke-particle drag, and phase transfer.
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

__global__ void fluid_smoke_drag(
    const Vec3 *positions, Vec3 *velocities,
    const std::uint32_t *water_count, FluidSmokeCouplingOptions coupling,
    SmokeOptions smoke, const Vec3 *smoke_positions,
    const Vec3 *smoke_velocities,
    const std::uint64_t *smoke_keys, const std::uint32_t *smoke_indices,
    float dt) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *water_count) return;
    const SmokeFlowSample local = smoke_sample_flow(positions[i],
        smoke_positions, smoke_velocities, smoke_keys, smoke_indices, smoke);
    if (local.number_density <= 1.0e-6F) return;
    const float occupancy = fminf(1.0F,
        local.number_density / smoke.rest_number_density);
    const float response = 1.0F - expf(-coupling.wind_drag * occupancy * dt);
    const Vec3 velocity = velocities[i];
    velocities[i] = clamp_length(add(velocity, multiply(
        subtract(local.velocity, velocity), response)),
        fmaxf(smoke.maximum_speed, sqrtf(length_squared(velocity))));
}

__global__ void fluid_smoke_grid_drag(const Vec3 *positions,Vec3 *velocities,
    const std::uint32_t *water_count,FluidSmokeCouplingOptions coupling,
    SmokeOptions smoke,SmokeGridField grid,float dt) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=*water_count||!smoke_grid_contains(positions[i],grid))return;
    Vec3 air{};float density{};
    smoke_grid_sample(positions[i],grid,air,density);
    if(density<=1.0e-6F)return;
    const float response=1.0F-expf(-coupling.wind_drag*
        clamp_scalar(density*smoke.rest_number_density,0.0F,1.0F)*dt);
    const Vec3 velocity=velocities[i];
    velocities[i]=clamp_length(add(velocity,multiply(
        subtract(air,velocity),response)),
        fmaxf(smoke.maximum_speed,sqrtf(length_squared(velocity))));
}

__global__ void fluid_smoke_exchange(
    const Vec3 *positions, Vec3 *velocities, float *temperatures,
    const std::uint32_t *water_count, float water_radius,
    FluidSmokeCouplingOptions coupling, SmokeOptions smoke,
    Vec3 *smoke_positions, Vec3 *smoke_velocities, float *smoke_ages,
    float *smoke_thermal_lift, float *smoke_densities,
    float *smoke_pressures, Vec3 *smoke_vorticities,
    std::uint32_t first_smoke_slot,
    std::uint32_t *converted, std::uint8_t *keep, float dt,
    Vec3 gravity) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *water_count) return;
    const Vec3 position = positions[i];
    const Vec3 velocity = velocities[i];
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
    smoke_densities[slot] = 0.0F;
    smoke_pressures[slot] = 0.0F;
    smoke_vorticities[slot] = {};
    keep[i] = 0U;
}

__global__ void fluid_clear_inactive_keep(std::uint8_t *keep,
                                          const std::uint32_t *count,
                                          std::uint32_t capacity) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    if (i >= *count) keep[i] = 0U;
}
