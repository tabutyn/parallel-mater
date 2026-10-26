// SPDX-License-Identifier: MIT
#include "smoke_grid_dispatch.hpp"

#include "metal_dispatch_options.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
using metal_bindings::vector;
using metal_bindings::smoke_options;
} // namespace

void begin_smoke_grid_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, float timestep,
    std::uint32_t profile_version, float *pressure, float *divergence,
    std::uint32_t *time_advances, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "smoke_realtime_grid_begin", grid,
        metal::grid_record(field), smoke_options(options), timestep,
        profile_version, pressure, divergence, time_advances, status);
}

void apply_smoke_grid_realtime_source(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *density_rate,
    const float *temperature_rate, float *density, float *temperature,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "smoke_realtime_grid_source", grid,
        metal::grid_record(field), smoke_options(options), density_rate,
        temperature_rate, density, temperature, timestep, profile_version,
        status);
}

void advect_smoke_grid_realtime_scalar(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *source,
    float *destination, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "smoke_realtime_grid_advect_scalar", grid,
        metal::grid_record(field), smoke_options(options), source,
        destination, timestep, profile_version, status);
}

void advect_smoke_grid_realtime_face(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *source,
    float *destination, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "smoke_realtime_grid_advect_face", grid,
        axis, metal::grid_record(field), smoke_options(options), source,
        destination, timestep, profile_version, status);
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
    (void)metal::launch(stream, "smoke_realtime_grid_face_forces", grid,
        axis, metal::grid_record(field), smoke_options(options), source,
        destination, open, wall, strain, subgrid, temperature, timestep,
        vector(gravity), profile_version, status);
}

void apply_smoke_grid_realtime_boundaries(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, float *values,
    const float *open, const float *wall, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "smoke_realtime_grid_boundaries", grid,
        axis, metal::grid_record(field), smoke_options(options), values,
        open, wall, timestep, profile_version, status);
}

void calculate_smoke_grid_realtime_divergence(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, float *divergence,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "smoke_realtime_grid_divergence", grid,
        metal::grid_record(field), smoke_options(options), divergence,
        timestep, profile_version, status);
}

void solve_smoke_grid_realtime_pressure_cycle(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *rhs,
    const float *pressure, float *next, const float *open_x,
    const float *open_y, const float *open_z, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "smoke_realtime_grid_pressure_cycle", grid,
        metal::grid_record(field), smoke_options(options), rhs, pressure,
        next, open_x, open_y, open_z, timestep, profile_version, status);
}

void project_smoke_grid_realtime_face(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *pressure,
    float *velocity, const float *open, const float *wall, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "smoke_realtime_grid_project_face", grid,
        axis, metal::grid_record(field), smoke_options(options), pressure,
        velocity, open, wall, timestep, profile_version, status);
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
    (void)metal::launch(stream, "smoke_realtime_grid_trace", grid,
        positions, previous_positions, velocities, ages, thermal_lift,
        number_densities, pressures, vorticities, count, first_emit_slot,
        emit_count, first_emit_serial,
        smoke_options(options), metal::grid_record(field), timestep,
        profile_version, status);
}

void initialize_smoke_grid_cells(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *velocity,
    float *density, float *temperature, float *pressure, float *divergence,
    void *vorticity, float *strain, void *subgrid_force,
    std::uint32_t *solid, std::uint32_t count, const float wind[3]) noexcept {
    (void)metal::launch(stream, "smoke_grid_initialize_cells", grid,
        velocity, density, temperature, pressure, divergence,
        vorticity, strain, subgrid_force, solid,
        count, vector(wind));
}

void initialize_smoke_grid_faces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, float *values,
    float *reverse, float *open, float *wall, void *normal,
    std::uint32_t *nearest_triangle, std::uint32_t count, float wind) noexcept {
    (void)metal::launch(stream, "smoke_grid_initialize_faces", grid,
        values, reverse, open, wall, normal, nearest_triangle,
        count, wind);
}

void clear_smoke_grid_obstacles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, std::uint32_t *solid,
    std::uint32_t cell_count, float *open_x, float *open_y, float *open_z,
    float *wall_x, float *wall_y, float *wall_z, std::uint32_t x_count,
    std::uint32_t y_count, std::uint32_t z_count) noexcept {
    (void)metal::launch(stream, "smoke_grid_clear_obstacles", grid,
        solid, cell_count, open_x, open_y, open_z, wall_x, wall_y, wall_z,
        x_count, y_count, z_count);
}

void raster_smoke_grid_rigid_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *mesh,
    const void *body_states, std::uint32_t body,
    const SmokeGridDispatchField &field,
    const SmokeGridBoundaryDispatch &boundary,
    bool resolve_metadata) noexcept {
    (void)metal::launch(stream, "smoke_grid_raster_rigid", grid,
        metal::mesh_record(*static_cast<const TriangleMeshResource *>(mesh)),
        body_states,
        body, metal::grid_record(field), boundary.solid,
        boundary.open[0], boundary.open[1], boundary.open[2],
        boundary.wall_velocity[0], boundary.wall_velocity[1],
        boundary.wall_velocity[2], boundary.normal[0],
        boundary.normal[1], boundary.normal[2],
        boundary.nearest_triangle[0], boundary.nearest_triangle[1],
        boundary.nearest_triangle[2], resolve_metadata);
}

void raster_smoke_grid_deformable(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, const std::uint32_t *indices,
    std::uint32_t index_count, const SmokeGridDispatchField &field,
    const SmokeGridBoundaryDispatch &boundary,
    bool resolve_metadata) noexcept {
    (void)metal::launch(stream, "smoke_grid_raster_deformable", grid,
        positions,
        velocities,
        indices, index_count, metal::grid_record(field),
        boundary.solid, boundary.open[0], boundary.open[1], boundary.open[2],
        boundary.wall_velocity[0], boundary.wall_velocity[1],
        boundary.wall_velocity[2], boundary.normal[0],
        boundary.normal[1], boundary.normal[2],
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
    (void)metal::launch(stream, "smoke_grid_raster_soft", grid,
        positions,
        bindings,
        node_velocities,
        indices, index_count, metal::grid_record(field),
        boundary.solid, boundary.open[0], boundary.open[1], boundary.open[2],
        boundary.wall_velocity[0], boundary.wall_velocity[1],
        boundary.wall_velocity[2], boundary.normal[0],
        boundary.normal[1], boundary.normal[2],
        boundary.nearest_triangle[0], boundary.nearest_triangle[1],
        boundary.nearest_triangle[2], resolve_metadata);
}

void resolve_smoke_grid_particle_fields(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const std::uint64_t *density_accumulator,
    const std::uint64_t *temperature_accumulator, float *density,
    float *temperature, std::uint32_t count) noexcept {
    (void)metal::launch(stream, "smoke_grid_resolve_particle_fields", grid,
        const_cast<std::uint64_t *>(density_accumulator),
        const_cast<std::uint64_t *>(temperature_accumulator),
        density, temperature, count);
}

void splat_smoke_grid_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const float *ages, const float *thermal_lift, std::uint32_t count,
    const SmokeDomainDispatchOptions &options,
    const SmokeGridDispatchField &field, std::uint64_t *density_accumulator,
    std::uint64_t *temperature_accumulator) noexcept {
    (void)metal::launch(stream, "smoke_grid_splat_particles", grid,
        positions,
        ages, thermal_lift, count,
        smoke_options(options), metal::grid_record(field),
        density_accumulator,
        temperature_accumulator);
}

void advect_smoke_grid_face(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, const float *source,
    float *destination, float timestep) noexcept {
    (void)metal::launch(stream, "smoke_grid_advect_face", grid,
        axis, metal::grid_record(field), source, destination,
        timestep);
}

void correct_smoke_grid_face(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &original, const float *source,
    const float *predicted, const float *reversed, float *corrected,
    float timestep) noexcept {
    (void)metal::launch(stream, "smoke_grid_correct_face", grid,
        axis, metal::grid_record(original), source,
        predicted, reversed,
        corrected, timestep);
}

void calculate_smoke_grid_cell_diagnostics(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const SmokeGridDispatchField &field, void *cell_velocity,
    void *vorticity, float *strain, float *divergence) noexcept {
    (void)metal::launch(stream, "smoke_grid_cell_diagnostics", grid,
        metal::grid_record(field), cell_velocity, vorticity,
        strain, divergence);
}

void calculate_smoke_grid_subgrid_force(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const SmokeGridDispatchField &field, const void *vorticity,
    void *force, float strength) noexcept {
    (void)metal::launch(stream, "smoke_grid_subgrid_force", grid,
        metal::grid_record(field), vorticity,
        force, strength);
}

void apply_smoke_grid_face_forces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, const float *source,
    float *destination, const float *open, const float *wall,
    const float *strain, const void *subgrid, const float *temperature,
    const SmokeDomainDispatchOptions &options, float timestep,
    const float gravity[3]) noexcept {
    (void)metal::launch(stream, "smoke_grid_apply_face_forces", grid,
        axis, metal::grid_record(field), source, destination,
        open, wall,
        strain,
        subgrid,
        temperature, smoke_options(options), timestep,
        vector(gravity));
}

void apply_smoke_grid_face_boundaries(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, float *values, const float *open,
    const float *wall, const SmokeDomainDispatchOptions &options) noexcept {
    (void)metal::launch(stream, "smoke_grid_apply_face_boundaries", grid,
        axis, metal::grid_record(field), values, open,
        wall, smoke_options(options));
}

void calculate_smoke_grid_divergence(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const SmokeGridDispatchField &field, float *divergence, float *rhs_max,
    float timestep) noexcept {
    (void)metal::launch(stream, "smoke_grid_divergence", grid,
        metal::grid_record(field), divergence, rhs_max, timestep);
}

void smooth_smoke_grid_pressure(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int resolution, int height,
    float spacing, const float *rhs, const float *pressure, float *next,
    const float *open_x, const float *open_y, const float *open_z,
    const std::uint32_t *converged) noexcept {
    (void)metal::launch(stream, "smoke_grid_pressure_smooth", grid,
        resolution, height, spacing, rhs,
        pressure, next, open_x,
        open_y, open_z,
        converged);
}

void calculate_smoke_grid_pressure_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int resolution, int height,
    float spacing, const float *rhs, const float *pressure, float *residual,
    const float *open_x, const float *open_y, const float *open_z,
    float *maximum, const std::uint32_t *converged) noexcept {
    (void)metal::launch(stream, "smoke_grid_pressure_residual", grid,
        resolution, height, spacing, rhs,
        pressure, residual, open_x,
        open_y, open_z, maximum,
        converged);
}

void restrict_smoke_grid_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const float *fine,
    int fine_resolution, int fine_height, float *coarse,
    int coarse_resolution, int coarse_height) noexcept {
    (void)metal::launch(stream, "smoke_grid_restrict_residual", grid,
        fine, fine_resolution, fine_height, coarse,
        coarse_resolution, coarse_height);
}

void clear_smoke_grid_scalar(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, float *values,
    std::uint32_t count) noexcept {
    (void)metal::launch(stream, "smoke_grid_clear_scalar", grid,values, count);
}

void restrict_smoke_grid_open_faces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const float *fine, int fine_resolution, int fine_height, float *coarse,
    int coarse_resolution, int coarse_height) noexcept {
    (void)metal::launch(stream, "smoke_grid_restrict_open", grid,
        axis, fine, fine_resolution, fine_height, coarse,
        coarse_resolution, coarse_height);
}

void mark_smoke_grid_domain_boundaries(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, float *open, float *wall,
    const SmokeDomainDispatchOptions &options) noexcept {
    (void)metal::launch(stream, "smoke_grid_mark_domain_boundaries", grid,
        axis, metal::grid_record(field), open, wall, smoke_options(options));
}

void prolong_smoke_grid_pressure(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const float *coarse,
    int coarse_resolution, int coarse_height, float *fine,
    int fine_resolution, int fine_height) noexcept {
    (void)metal::launch(stream, "smoke_grid_prolong_add", grid,
        coarse, coarse_resolution, coarse_height, fine,
        fine_resolution, fine_height);
}

void compare_smoke_grid_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const float *rhs_max,
    const float *residual_max, float tolerance,
    std::uint32_t *converged) noexcept {
    (void)metal::launch(stream, "smoke_grid_compare_residual", grid,
        rhs_max, residual_max,
        tolerance, converged);
}

void reset_smoke_grid_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, float *residual_max,
    const std::uint32_t *converged) noexcept {
    (void)metal::launch(stream, "smoke_grid_reset_residual", grid,
        residual_max, converged);
}

void project_smoke_grid_face(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, const float *pressure,
    float *velocity, const float *open, const float *wall,
    float timestep) noexcept {
    (void)metal::launch(stream, "smoke_grid_project_face", grid,
        axis, metal::grid_record(field), pressure, velocity,
        open, wall, timestep);
}

void trace_smoke_grid_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, float *number_densities, float *pressures,
    void *vorticities, std::uint32_t count,
    const SmokeDomainDispatchOptions &options,
    const SmokeGridDispatchField &field, float timestep,
    const float gravity[3]) noexcept {
    (void)metal::launch(stream, "smoke_grid_trace", grid,
        positions, previous_positions,
        velocities, ages, thermal_lift, number_densities,
        pressures, vorticities, count, smoke_options(options),
        metal::grid_record(field), timestep, vector(gravity));
}

void apply_smoke_grid_force_to_soft_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    const std::uint32_t *indices, std::uint32_t index_count,
    const void *bindings, const void *node_velocity_snapshot,
    void *node_velocities, const float *inverse_masses,
    const SmokeGridDispatchField &field, float drag_coefficient,
    float density_scale, float kinematic_viscosity, float les_coefficient,
    float maximum_acceleration, float timestep) noexcept {
    (void)metal::launch(stream, "smoke_grid_soft_body_force", grid,
        surface,
        indices, index_count,
        bindings,
        node_velocity_snapshot,
        node_velocities, inverse_masses,
        metal::grid_record(field), drag_coefficient, density_scale,
        kinematic_viscosity, les_coefficient, maximum_acceleration, timestep);
}

void accumulate_smoke_grid_force_on_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t index_count,
    const SmokeGridDispatchField &field, float drag_coefficient,
    float density_scale, float kinematic_viscosity, float les_coefficient,
    void *forces) noexcept {
    (void)metal::launch(stream, "smoke_grid_cloth_force", grid,
        positions,
        velocities,
        inverse_masses,
        indices, index_count, metal::grid_record(field),
        drag_coefficient, density_scale, kinematic_viscosity,
        les_coefficient, forces);
}

void apply_smoke_grid_force_to_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *velocities,
    const float *inverse_masses, const void *forces, std::uint32_t count,
    float maximum_acceleration, float timestep) noexcept {
    (void)metal::launch(stream, "smoke_grid_apply_cloth_force", grid,
        velocities, inverse_masses,
        forces, count,
        maximum_acceleration, timestep);
}

void apply_smoke_grid_force_to_rigid_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *meshes,
    std::uint32_t mesh_index, std::uint32_t body_index,
    const void *body_parameters, void *body_states,
    const SmokeGridDispatchField &field, float air_density,
    float drag_coefficient, float density_scale, float kinematic_viscosity,
    float les_coefficient, float timestep) noexcept {
    (void)metal::launch(stream, "smoke_grid_rigid_force", grid,
        meshes, mesh_index, body_index,
        body_parameters,
        body_states, metal::grid_record(field), air_density,
        drag_coefficient, density_scale, kinematic_viscosity,
        les_coefficient, timestep);
}

} // namespace parallel_mater::simulation
