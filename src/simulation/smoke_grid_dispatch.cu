// SPDX-License-Identifier: MIT
#include "smoke_grid_dispatch.hpp"

#include "parallel_mater_smoke_grid_runtime.cu"

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
    result.wind_2 = vector(value.wind);
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
    result.n_2 = value.resolution;
    result.height_2 = value.height;
    result.minimum_1 = vector(value.minimum);
    result.spacing_0 = value.spacing;
    result.velocity_1 = const_cast<Vec3_0 *>(as<Vec3_0>(value.velocity));
    result.density_1 = const_cast<float *>(value.density);
    result.pressure_1 = const_cast<float *>(value.pressure);
    for (int axis = 0; axis < 3; ++axis)
        result.face_1.m_data[axis] =
            const_cast<float *>(value.face_velocity[axis]);
    result.vorticity_1 = const_cast<Vec3_0 *>(as<Vec3_0>(value.vorticity));
    result.strain_1 = const_cast<float *>(value.strain);
    return result;
}
} // namespace

void begin_smoke_grid_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, float timestep,
    std::uint32_t profile_version, float *pressure, float *divergence,
    std::uint32_t *time_advances, std::uint32_t *status) noexcept {
    smoke_realtime_grid_begin<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        grid_field(field), smoke_options(options), timestep, profile_version,
        pressure, divergence, time_advances, status);
}

void apply_smoke_grid_realtime_source(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *density_rate,
    const float *temperature_rate, float *density, float *temperature,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    smoke_realtime_grid_source<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        grid_field(field), smoke_options(options),
        const_cast<float *>(density_rate),
        const_cast<float *>(temperature_rate), density, temperature,
        timestep, profile_version, status);
}

void advect_smoke_grid_realtime_scalar(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *source,
    float *destination, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    smoke_realtime_grid_advect_scalar<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        grid_field(field), smoke_options(options), const_cast<float *>(source),
        destination, timestep, profile_version, status);
}

void advect_smoke_grid_realtime_face(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *source,
    float *destination, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    smoke_realtime_grid_advect_face<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(field), smoke_options(options),
        const_cast<float *>(source), destination, timestep, profile_version,
        status);
}

void apply_smoke_grid_realtime_face_forces(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *source,
    float *destination, const float *open, const float *wall,
    const float *strain, const void *subgrid, const float *temperature,
    float timestep, const float gravity[3], std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    smoke_realtime_grid_face_forces<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(field), smoke_options(options),
        const_cast<float *>(source), destination, const_cast<float *>(open),
        const_cast<float *>(wall), const_cast<float *>(strain),
        const_cast<Vec3_0 *>(as<Vec3_0>(subgrid)),
        const_cast<float *>(temperature), timestep, vector(gravity),
        profile_version, status);
}

void apply_smoke_grid_realtime_boundaries(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, float *values,
    const float *open, const float *wall, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    smoke_realtime_grid_boundaries<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(field), smoke_options(options), values,
        const_cast<float *>(open), const_cast<float *>(wall), timestep,
        profile_version, status);
}

void calculate_smoke_grid_realtime_divergence(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, float *divergence,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    smoke_realtime_grid_divergence<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        grid_field(field), smoke_options(options), divergence, timestep,
        profile_version, status);
}

void solve_smoke_grid_realtime_pressure_cycle(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *rhs,
    const float *pressure, float *next, const float *open_x,
    const float *open_y, const float *open_z, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    smoke_realtime_grid_pressure_cycle<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        grid_field(field), smoke_options(options), const_cast<float *>(rhs),
        const_cast<float *>(pressure), next, const_cast<float *>(open_x),
        const_cast<float *>(open_y), const_cast<float *>(open_z), timestep,
        profile_version, status);
}

void project_smoke_grid_realtime_face(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *pressure,
    float *velocity, const float *open, const float *wall, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    smoke_realtime_grid_project_face<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(field), smoke_options(options),
        const_cast<float *>(pressure), velocity, const_cast<float *>(open),
        const_cast<float *>(wall), timestep, profile_version, status);
}

void trace_smoke_grid_realtime_particles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, float *number_densities, float *pressures,
    void *vorticities, std::uint32_t count, std::uint32_t first_emit_slot,
    std::uint32_t emit_count, std::uint64_t first_emit_serial,
    const SmokeDomainDispatchOptions &options,
    const SmokeGridDispatchField &field, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    smoke_realtime_grid_trace<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(previous_positions),
        as<Vec3_0>(velocities), ages, thermal_lift, number_densities,
        pressures, as<Vec3_0>(vorticities), count, first_emit_slot,
        emit_count, first_emit_serial, smoke_options(options),
        grid_field(field), timestep, profile_version, status);
}

void initialize_smoke_grid_cells(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *velocity,
    float *density, float *temperature, float *pressure, float *divergence,
    void *vorticity, float *strain, void *subgrid_force,
    std::uint32_t *solid, std::uint32_t count, const float wind[3]) noexcept {
    smoke_grid_initialize_cells<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(velocity), density, temperature, pressure, divergence,
        as<Vec3_0>(vorticity), strain, as<Vec3_0>(subgrid_force), solid,
        count, vector(wind));
}

void initialize_smoke_grid_faces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, float *values,
    float *reverse, float *open, float *wall, void *normal,
    std::uint32_t *nearest_triangle, std::uint32_t count, float wind) noexcept {
    smoke_grid_initialize_faces<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        values, reverse, open, wall, as<Vec3_0>(normal), nearest_triangle,
        count, wind);
}

void clear_smoke_grid_obstacles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, std::uint32_t *solid,
    std::uint32_t cell_count, float *open_x, float *open_y, float *open_z,
    float *wall_x, float *wall_y, float *wall_z, std::uint32_t x_count,
    std::uint32_t y_count, std::uint32_t z_count) noexcept {
    smoke_grid_clear_obstacles<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        solid, cell_count, open_x, open_y, open_z, wall_x, wall_y, wall_z,
        x_count, y_count, z_count);
}

void raster_smoke_grid_rigid_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *mesh,
    const void *body_states, std::uint32_t body,
    const SmokeGridDispatchField &field,
    const SmokeGridBoundaryDispatch &boundary,
    bool resolve_metadata) noexcept {
    smoke_grid_raster_rigid<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        *as<TriangleMeshResource_0>(mesh),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        body, grid_field(field), boundary.solid,
        boundary.open[0], boundary.open[1], boundary.open[2],
        boundary.wall_velocity[0], boundary.wall_velocity[1],
        boundary.wall_velocity[2], as<Vec3_0>(boundary.normal[0]),
        as<Vec3_0>(boundary.normal[1]), as<Vec3_0>(boundary.normal[2]),
        boundary.nearest_triangle[0], boundary.nearest_triangle[1],
        boundary.nearest_triangle[2], resolve_metadata);
}

void raster_smoke_grid_deformable(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, const std::uint32_t *indices,
    std::uint32_t index_count, const SmokeGridDispatchField &field,
    const SmokeGridBoundaryDispatch &boundary,
    bool resolve_metadata) noexcept {
    smoke_grid_raster_deformable<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<std::uint32_t *>(indices), index_count, grid_field(field),
        boundary.solid, boundary.open[0], boundary.open[1], boundary.open[2],
        boundary.wall_velocity[0], boundary.wall_velocity[1],
        boundary.wall_velocity[2], as<Vec3_0>(boundary.normal[0]),
        as<Vec3_0>(boundary.normal[1]), as<Vec3_0>(boundary.normal[2]),
        boundary.nearest_triangle[0], boundary.nearest_triangle[1],
        boundary.nearest_triangle[2], resolve_metadata);
}

void raster_smoke_grid_soft_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *bindings, const void *node_velocities,
    const std::uint32_t *indices, std::uint32_t index_count,
    const SmokeGridDispatchField &field,
    const SmokeGridBoundaryDispatch &boundary,
    bool resolve_metadata) noexcept {
    smoke_grid_raster_soft<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        const_cast<Vec3_0 *>(as<Vec3_0>(node_velocities)),
        const_cast<std::uint32_t *>(indices), index_count, grid_field(field),
        boundary.solid, boundary.open[0], boundary.open[1], boundary.open[2],
        boundary.wall_velocity[0], boundary.wall_velocity[1],
        boundary.wall_velocity[2], as<Vec3_0>(boundary.normal[0]),
        as<Vec3_0>(boundary.normal[1]), as<Vec3_0>(boundary.normal[2]),
        boundary.nearest_triangle[0], boundary.nearest_triangle[1],
        boundary.nearest_triangle[2], resolve_metadata);
}

void resolve_smoke_grid_particle_fields(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const std::uint64_t *density_accumulator,
    const std::uint64_t *temperature_accumulator, float *density,
    float *temperature, std::uint32_t count) noexcept {
    smoke_grid_resolve_particle_fields<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<unsigned long long *>(
            reinterpret_cast<const unsigned long long *>(density_accumulator)),
        const_cast<unsigned long long *>(
            reinterpret_cast<const unsigned long long *>(temperature_accumulator)),
        density, temperature, count);
}

void splat_smoke_grid_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const float *ages, const float *thermal_lift, std::uint32_t count,
    const SmokeDomainDispatchOptions &options,
    const SmokeGridDispatchField &field, std::uint64_t *density_accumulator,
    std::uint64_t *temperature_accumulator) noexcept {
    smoke_grid_splat_particles<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<float *>(ages), const_cast<float *>(thermal_lift), count,
        smoke_options(options), grid_field(field),
        reinterpret_cast<unsigned long long *>(density_accumulator),
        reinterpret_cast<unsigned long long *>(temperature_accumulator));
}

void advect_smoke_grid_face(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, const float *source,
    float *destination, float timestep) noexcept {
    smoke_grid_advect_face<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(field), const_cast<float *>(source), destination,
        timestep);
}

void correct_smoke_grid_face(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &original, const float *source,
    const float *predicted, const float *reversed, float *corrected,
    float timestep) noexcept {
    smoke_grid_correct_face<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(original), const_cast<float *>(source),
        const_cast<float *>(predicted), const_cast<float *>(reversed),
        corrected, timestep);
}

void calculate_smoke_grid_cell_diagnostics(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const SmokeGridDispatchField &field, void *cell_velocity,
    void *vorticity, float *strain, float *divergence) noexcept {
    smoke_grid_cell_diagnostics<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        grid_field(field), as<Vec3_0>(cell_velocity), as<Vec3_0>(vorticity),
        strain, divergence);
}

void calculate_smoke_grid_subgrid_force(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const SmokeGridDispatchField &field, const void *vorticity,
    void *force, float strength) noexcept {
    smoke_grid_subgrid_force<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        grid_field(field), const_cast<Vec3_0 *>(as<Vec3_0>(vorticity)),
        as<Vec3_0>(force), strength);
}

void apply_smoke_grid_face_forces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, const float *source,
    float *destination, const float *open, const float *wall,
    const float *strain, const void *subgrid, const float *temperature,
    const SmokeDomainDispatchOptions &options, float timestep,
    const float gravity[3]) noexcept {
    smoke_grid_apply_face_forces<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(field), const_cast<float *>(source), destination,
        const_cast<float *>(open), const_cast<float *>(wall),
        const_cast<float *>(strain),
        const_cast<Vec3_0 *>(as<Vec3_0>(subgrid)),
        const_cast<float *>(temperature), smoke_options(options), timestep,
        vector(gravity));
}

void apply_smoke_grid_face_boundaries(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, float *values, const float *open,
    const float *wall, const SmokeDomainDispatchOptions &options) noexcept {
    smoke_grid_apply_face_boundaries<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(field), values, const_cast<float *>(open),
        const_cast<float *>(wall), smoke_options(options));
}

void calculate_smoke_grid_divergence(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const SmokeGridDispatchField &field, float *divergence, float *rhs_max,
    float timestep) noexcept {
    smoke_grid_divergence<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        grid_field(field), divergence, rhs_max, timestep);
}

void smooth_smoke_grid_pressure(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int resolution, int height,
    float spacing, const float *rhs, const float *pressure, float *next,
    const float *open_x, const float *open_y, const float *open_z,
    const std::uint32_t *converged) noexcept {
    smoke_grid_pressure_smooth<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        resolution, height, spacing, const_cast<float *>(rhs),
        const_cast<float *>(pressure), next, const_cast<float *>(open_x),
        const_cast<float *>(open_y), const_cast<float *>(open_z),
        const_cast<std::uint32_t *>(converged));
}

void calculate_smoke_grid_pressure_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int resolution, int height,
    float spacing, const float *rhs, const float *pressure, float *residual,
    const float *open_x, const float *open_y, const float *open_z,
    float *maximum, const std::uint32_t *converged) noexcept {
    smoke_grid_pressure_residual<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        resolution, height, spacing, const_cast<float *>(rhs),
        const_cast<float *>(pressure), residual, const_cast<float *>(open_x),
        const_cast<float *>(open_y), const_cast<float *>(open_z), maximum,
        const_cast<std::uint32_t *>(converged));
}

void restrict_smoke_grid_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const float *fine,
    int fine_resolution, int fine_height, float *coarse,
    int coarse_resolution, int coarse_height) noexcept {
    smoke_grid_restrict_residual<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<float *>(fine), fine_resolution, fine_height, coarse,
        coarse_resolution, coarse_height);
}

void clear_smoke_grid_scalar(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, float *values,
    std::uint32_t count) noexcept {
    smoke_grid_clear_scalar<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(values, count);
}

void restrict_smoke_grid_open_faces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const float *fine, int fine_resolution, int fine_height, float *coarse,
    int coarse_resolution, int coarse_height) noexcept {
    smoke_grid_restrict_open<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, const_cast<float *>(fine), fine_resolution, fine_height, coarse,
        coarse_resolution, coarse_height);
}

void mark_smoke_grid_domain_boundaries(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, float *open, float *wall,
    const SmokeDomainDispatchOptions &options) noexcept {
    smoke_grid_mark_domain_boundaries<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(field), open, wall, smoke_options(options));
}

void prolong_smoke_grid_pressure(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const float *coarse,
    int coarse_resolution, int coarse_height, float *fine,
    int fine_resolution, int fine_height) noexcept {
    smoke_grid_prolong_add<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<float *>(coarse), coarse_resolution, coarse_height, fine,
        fine_resolution, fine_height);
}

void compare_smoke_grid_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const float *rhs_max,
    const float *residual_max, float tolerance,
    std::uint32_t *converged) noexcept {
    smoke_grid_compare_residual<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<float *>(rhs_max), const_cast<float *>(residual_max),
        tolerance, converged);
}

void reset_smoke_grid_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, float *residual_max,
    const std::uint32_t *converged) noexcept {
    smoke_grid_reset_residual<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        residual_max, const_cast<std::uint32_t *>(converged));
}

void project_smoke_grid_face(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, const float *pressure,
    float *velocity, const float *open, const float *wall,
    float timestep) noexcept {
    smoke_grid_project_face<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        axis, grid_field(field), const_cast<float *>(pressure), velocity,
        const_cast<float *>(open), const_cast<float *>(wall), timestep);
}

void trace_smoke_grid_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, float *number_densities, float *pressures,
    void *vorticities, std::uint32_t count,
    const SmokeDomainDispatchOptions &options,
    const SmokeGridDispatchField &field, float timestep,
    const float gravity[3]) noexcept {
    smoke_grid_trace<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(previous_positions),
        as<Vec3_0>(velocities), ages, thermal_lift, number_densities,
        pressures, as<Vec3_0>(vorticities), count, smoke_options(options),
        grid_field(field), timestep, vector(gravity));
}

void apply_smoke_grid_force_to_soft_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    const std::uint32_t *indices, std::uint32_t index_count,
    const void *bindings, const void *node_velocity_snapshot,
    void *node_velocities, const float *inverse_masses,
    const SmokeGridDispatchField &field, float drag_coefficient,
    float density_scale, float kinematic_viscosity, float les_coefficient,
    float maximum_acceleration, float timestep) noexcept {
    smoke_grid_soft_body_force<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)),
        const_cast<std::uint32_t *>(indices), index_count,
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        const_cast<Vec3_0 *>(as<Vec3_0>(node_velocity_snapshot)),
        as<Vec3_0>(node_velocities), const_cast<float *>(inverse_masses),
        grid_field(field), drag_coefficient, density_scale,
        kinematic_viscosity, les_coefficient, maximum_acceleration, timestep);
}

void accumulate_smoke_grid_force_on_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t index_count,
    const SmokeGridDispatchField &field, float drag_coefficient,
    float density_scale, float kinematic_viscosity, float les_coefficient,
    void *forces) noexcept {
    smoke_grid_cloth_force<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<float *>(inverse_masses),
        const_cast<std::uint32_t *>(indices), index_count, grid_field(field),
        drag_coefficient, density_scale, kinematic_viscosity,
        les_coefficient, as<Vec3_0>(forces));
}

void apply_smoke_grid_force_to_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *velocities,
    const float *inverse_masses, const void *forces, std::uint32_t count,
    float maximum_acceleration, float timestep) noexcept {
    smoke_grid_apply_cloth_force<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(velocities), const_cast<float *>(inverse_masses),
        const_cast<Vec3_0 *>(as<Vec3_0>(forces)), count,
        maximum_acceleration, timestep);
}

void apply_smoke_grid_force_to_rigid_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *meshes,
    std::uint32_t mesh_index, std::uint32_t body,
    const void *body_parameters, void *body_states,
    const SmokeGridDispatchField &field, float air_density,
    float drag_coefficient, float density_scale, float kinematic_viscosity,
    float les_coefficient, float timestep) noexcept {
    smoke_grid_rigid_force<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        mesh_index, body,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states), grid_field(field), air_density,
        drag_coefficient, density_scale, kinematic_viscosity,
        les_coefficient, timestep);
}

} // namespace parallel_mater::simulation
