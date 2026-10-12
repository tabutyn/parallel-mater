// SPDX-License-Identifier: MIT
#include "geometry_smoke_dispatch.hpp"

#include "parallel_mater_geometry_smoke_runtime.cu"

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

SmokeOptions_0 smoke_options(const SmokeDomainDispatchOptions &value) noexcept {
    SmokeOptions_0 result{};
    result.capacity_0 = value.capacity;
    result.emitter_center_0 = vector(value.emitter_center);
    result.emitter_half_extents_0 = {
        value.emitter_half_extents[0], value.emitter_half_extents[1]};
    result.initial_velocity_0 = vector(value.initial_velocity);
    result.wind_0 = vector(value.wind);
    result.particles_per_second_0 = value.particles_per_second;
    result.lifetime_0 = value.lifetime;
    result.particle_radius_0 = value.particle_radius;
    result.buoyancy_0 = value.buoyancy;
    result.response_0 = value.response;
    result.rest_number_density_0 = value.rest_number_density;
    result.pressure_stiffness_0 = value.pressure_stiffness;
    result.viscosity_0 = value.viscosity;
    result.vorticity_confinement_0 = value.vorticity_confinement;
    result.maximum_speed_0 = value.maximum_speed;
    result.grid_resolution_0 = value.grid_resolution;
    result.grid_vertical_resolution_0 = value.grid_vertical_resolution;
    result.grid_pressure_iterations_0 = value.grid_pressure_iterations;
    result.grid_kinematic_viscosity_0 = value.grid_kinematic_viscosity;
    result.grid_les_coefficient_0 = value.grid_les_coefficient;
    result.grid_pressure_tolerance_0 = value.grid_pressure_tolerance;
    result.grid_minimum_0 = vector(value.grid_minimum);
    result.grid_edge_length_0 = value.grid_edge_length;
    return result;
}

SmokeGridField_0 grid_field(const SmokeGridDispatchField &value) noexcept {
    SmokeGridField_0 result{};
    result.n_0 = value.resolution;
    result.height_0 = value.height;
    result.minimum_3 = vector(value.minimum);
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

void index_smoke_particles(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, void *keys,
    std::uint32_t *indices) noexcept {
    smoke_emit_cells<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<float *>(ages), count, smoke_options(options),
        as<unsigned long long>(keys), indices);
}

void calculate_smoke_density_pressure(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *keys,
    const std::uint32_t *indices, float *densities, float *pressures,
    float timestep) noexcept {
    smoke_density_pressure<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<float *>(ages), count, smoke_options(options),
        const_cast<unsigned long long *>(as<unsigned long long>(keys)),
        const_cast<std::uint32_t *>(indices), densities, pressures, timestep);
}

void calculate_smoke_vorticity(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *velocities, const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *keys,
    const std::uint32_t *indices, const float *densities,
    void *vorticities, float *magnitudes) noexcept {
    smoke_compute_vorticity<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<float *>(ages), count, smoke_options(options),
        const_cast<unsigned long long *>(as<unsigned long long>(keys)),
        const_cast<std::uint32_t *>(indices), const_cast<float *>(densities),
        as<Vec3_0>(vorticities), magnitudes);
}

void calculate_smoke_pair_forces(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *velocities, const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *keys,
    const std::uint32_t *indices, const float *densities,
    const float *pressures, const void *vorticities,
    const float *vorticity_magnitudes, void *accelerations) noexcept {
    smoke_pair_forces<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<float *>(ages), count, smoke_options(options),
        const_cast<unsigned long long *>(as<unsigned long long>(keys)),
        const_cast<std::uint32_t *>(indices), const_cast<float *>(densities),
        const_cast<float *>(pressures),
        const_cast<Vec3_0 *>(as<Vec3_0>(vorticities)),
        const_cast<float *>(vorticity_magnitudes), as<Vec3_0>(accelerations));
}

void advect_smoke_particles(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *accelerations,
    float timestep, const float gravity[3]) noexcept {
    smoke_advect<<<grid, block, 0U, stream>>>(
        as<Vec3_0>(positions), as<Vec3_0>(previous_positions),
        as<Vec3_0>(velocities), ages, thermal_lift, count,
        smoke_options(options),
        const_cast<Vec3_0 *>(as<Vec3_0>(accelerations)), timestep,
        vector(gravity));
}

void emit_smoke_particles(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, float *densities, float *pressures,
    void *vorticities, const SmokeDomainDispatchOptions &options,
    std::uint32_t first_slot, std::uint32_t count,
    std::uint64_t first_serial) noexcept {
    smoke_emit<<<grid, block, 0U, stream>>>(
        as<Vec3_0>(positions), as<Vec3_0>(previous_positions),
        as<Vec3_0>(velocities), ages, thermal_lift, densities, pressures,
        as<Vec3_0>(vorticities), smoke_options(options), first_slot, count,
        static_cast<unsigned long long>(first_serial));
}

void calculate_smoke_deformable_bounds(
    dim3 grid, dim3 block, cudaStream_t stream, const void *surface,
    std::uint32_t count, void *minimum, void *maximum) noexcept {
    smoke_deformable_bounds<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)), count,
        as<Vec3_0>(minimum), as<Vec3_0>(maximum));
}

void collide_smoke_with_rigid_body(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    float *pressures, std::uint32_t count, float lifetime,
    const void *mesh, void *body_states, const void *previous_body_states,
    std::uint32_t body, float clearance, float particle_mass,
    float wall_drag, float pressure_stiffness, float timestep,
    float maximum_speed, const SmokeGridDispatchField &smoke_grid,
    void *impulses) noexcept {
    smoke_rigid_contact<<<grid, block, 0U, stream>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        as<Vec3_0>(velocities), const_cast<float *>(ages), pressures, count,
        lifetime, *as<TriangleMeshResource_0>(mesh),
        as<RigidBodyState_0>(body_states),
        const_cast<RigidBodyState_0 *>(
            as<RigidBodyState_0>(previous_body_states)),
        body, clearance, particle_mass, wall_drag, pressure_stiffness,
        timestep, maximum_speed, grid_field(smoke_grid),
        as<FluidBodyImpulse_0>(impulses));
}

void apply_smoke_rigid_body_impulses(
    dim3 grid, dim3 block, cudaStream_t stream, const void *impulses,
    std::uint32_t count, std::uint32_t body, const void *body_parameters,
    void *body_states) noexcept {
    smoke_apply_rigid_impulses<<<grid, block, 0U, stream>>>(
        const_cast<FluidBodyImpulse_0 *>(as<FluidBodyImpulse_0>(impulses)),
        count, body,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states));
}

} // namespace parallel_mater::simulation
