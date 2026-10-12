// SPDX-License-Identifier: MIT
#include "fluid_smoke_dispatch.hpp"

#include "parallel_mater_fluid_smoke_runtime.cu"

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

Vec3_0 vector(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}

SmokeOptions_0 smoke_options(const FluidSmokeDispatchOptions &value) noexcept {
    SmokeOptions_0 result{};
    result.capacity_0 = value.smoke.capacity;
    result.particle_radius_0 = value.smoke.particle_radius;
    result.lifetime_0 = value.smoke.lifetime;
    result.rest_number_density_0 = value.smoke.rest_number_density;
    result.pressure_stiffness_0 = value.smoke.pressure_stiffness;
    result.maximum_speed_0 = value.smoke.maximum_speed;
    return result;
}

FluidSmokeCouplingOptions_0 coupling_options(
    const FluidSmokeDispatchOptions &value) noexcept {
    FluidSmokeCouplingOptions_0 result{};
    result.heater_0.center_0 = vector(value.heater_center);
    result.heater_0.orientation_0 = {
        value.heater_orientation[0], value.heater_orientation[1],
        value.heater_orientation[2], value.heater_orientation[3]};
    result.heater_0.half_extents_0 = {
        value.heater_half_extents[0], value.heater_half_extents[1]};
    result.heater_temperature_0 = value.heater_temperature;
    result.boiling_temperature_0 = value.boiling_temperature;
    result.heat_transfer_rate_0 = value.heat_transfer_rate;
    result.wind_drag_0 = value.wind_drag;
    result.steam_rise_speed_0 = value.steam_rise_speed;
    return result;
}

SmokeGridField_0 grid_field(const SmokeGridDispatchField &value) noexcept {
    SmokeGridField_0 result{};
    result.n_0 = value.resolution;
    result.height_0 = value.height;
    result.minimum_0 = vector(value.minimum);
    result.spacing_0 = value.spacing;
    result.velocity_3 = const_cast<Vec3_0 *>(as<Vec3_0>(value.velocity));
    result.density_0 = const_cast<float *>(value.density);
    result.pressure_0 = const_cast<float *>(value.pressure);
    for (int axis = 0; axis < 3; ++axis)
        result.face_0.m_data[axis] =
            const_cast<float *>(value.face_velocity[axis]);
    result.vorticity_0 = const_cast<Vec3_0 *>(as<Vec3_0>(value.vorticity));
    result.strain_0 = const_cast<float *>(value.strain);
    return result;
}

} // namespace

void drag_fluid_with_smoke_particles(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    void *velocities, const std::uint32_t *particle_count,
    const FluidSmokeDispatchOptions &options, const void *smoke_positions,
    const void *smoke_velocities, const std::uint64_t *smoke_keys,
    const std::uint32_t *smoke_indices, float timestep) noexcept {
    fluid_smoke_drag<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)), as<Vec3_0>(velocities),
        const_cast<std::uint32_t *>(particle_count), coupling_options(options),
        smoke_options(options),
        const_cast<Vec3_0 *>(as<Vec3_0>(smoke_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(smoke_velocities)),
        reinterpret_cast<ulonglong *>(
            const_cast<std::uint64_t *>(smoke_keys)),
        const_cast<std::uint32_t *>(smoke_indices), timestep);
}

void drag_fluid_with_smoke_grid(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    void *velocities, const std::uint32_t *particle_count,
    const FluidSmokeDispatchOptions &options,
    const SmokeGridDispatchField &smoke_grid, float timestep) noexcept {
    fluid_smoke_grid_drag<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)), as<Vec3_0>(velocities),
        const_cast<std::uint32_t *>(particle_count), coupling_options(options),
        smoke_options(options), grid_field(smoke_grid), timestep);
}

void exchange_fluid_with_smoke(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    void *velocities, float *temperatures,
    const std::uint32_t *particle_count, float particle_radius,
    const FluidSmokeDispatchOptions &options, void *smoke_positions,
    void *smoke_velocities, float *smoke_ages, float *smoke_thermal_lift,
    float *smoke_densities, float *smoke_pressures,
    void *smoke_vorticities, std::uint32_t first_smoke_slot,
    std::uint32_t *converted, std::uint8_t *keep, float timestep,
    const float gravity[3]) noexcept {
    fluid_smoke_exchange<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)), as<Vec3_0>(velocities),
        temperatures, const_cast<std::uint32_t *>(particle_count),
        particle_radius, coupling_options(options), smoke_options(options),
        as<Vec3_0>(smoke_positions), as<Vec3_0>(smoke_velocities), smoke_ages,
        smoke_thermal_lift, smoke_densities, smoke_pressures,
        as<Vec3_0>(smoke_vorticities), first_smoke_slot, converted, keep,
        timestep, vector(gravity));
}

void clear_inactive_fluid_keep(
    dim3 grid, dim3 block, cudaStream_t stream, std::uint8_t *keep,
    const std::uint32_t *particle_count, std::uint32_t capacity) noexcept {
    fluid_clear_inactive_keep<<<grid, block, 0U, stream>>>(
        keep, const_cast<std::uint32_t *>(particle_count), capacity);
}

} // namespace parallel_mater::simulation
