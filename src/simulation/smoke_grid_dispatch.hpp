// SPDX-License-Identifier: MIT
#pragma once

#include "smoke_dispatch.hpp"

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

struct SmokeGridBoundaryDispatch {
    std::uint32_t *solid{};
    float *open[3]{};
    float *wall_velocity[3]{};
    void *normal[3]{};
    std::uint32_t *nearest_triangle[3]{};
};

void begin_smoke_grid_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, float timestep,
    std::uint32_t profile_version, float *pressure, float *divergence,
    std::uint32_t *time_advances, std::uint32_t *status) noexcept;

void apply_smoke_grid_realtime_source(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *density_rate,
    const float *temperature_rate, float *density, float *temperature,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void advect_smoke_grid_realtime_scalar(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *source,
    float *destination, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void advect_smoke_grid_realtime_face(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *source,
    float *destination, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void apply_smoke_grid_realtime_face_forces(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *source,
    float *destination, const float *open, const float *wall,
    const float *strain, const void *subgrid, const float *temperature,
    float timestep, const float gravity[3], std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void apply_smoke_grid_realtime_boundaries(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, float *values,
    const float *open, const float *wall, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void calculate_smoke_grid_realtime_divergence(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, float *divergence,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept;

void solve_smoke_grid_realtime_pressure_cycle(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *rhs,
    const float *pressure, float *next, const float *open_x,
    const float *open_y, const float *open_z, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void project_smoke_grid_realtime_face(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field,
    const SmokeDomainDispatchOptions &options, const float *pressure,
    float *velocity, const float *open, const float *wall, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void trace_smoke_grid_realtime_particles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, float *number_densities, float *pressures,
    void *vorticities, std::uint32_t count, std::uint32_t first_emit_slot,
    std::uint32_t emit_count, std::uint64_t first_emit_serial,
    const SmokeDomainDispatchOptions &options,
    const SmokeGridDispatchField &field, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept;

void initialize_smoke_grid_cells(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *velocity,
    float *density, float *temperature, float *pressure, float *divergence,
    void *vorticity, float *strain, void *subgrid_force,
    std::uint32_t *solid, std::uint32_t count, const float wind[3]) noexcept;

void initialize_smoke_grid_faces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, float *values,
    float *reverse, float *open, float *wall, void *normal,
    std::uint32_t *nearest_triangle, std::uint32_t count, float wind) noexcept;

void clear_smoke_grid_obstacles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, std::uint32_t *solid,
    std::uint32_t cell_count, float *open_x, float *open_y, float *open_z,
    float *wall_x, float *wall_y, float *wall_z, std::uint32_t x_count,
    std::uint32_t y_count, std::uint32_t z_count) noexcept;

void raster_smoke_grid_rigid_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *mesh,
    const void *body_states, std::uint32_t body,
    const SmokeGridDispatchField &field,
    const SmokeGridBoundaryDispatch &boundary,
    bool resolve_metadata) noexcept;

void raster_smoke_grid_deformable(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, const std::uint32_t *indices,
    std::uint32_t index_count, const SmokeGridDispatchField &field,
    const SmokeGridBoundaryDispatch &boundary,
    bool resolve_metadata) noexcept;

void raster_smoke_grid_soft_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *bindings, const void *node_velocities,
    const std::uint32_t *indices, std::uint32_t index_count,
    const SmokeGridDispatchField &field,
    const SmokeGridBoundaryDispatch &boundary,
    bool resolve_metadata) noexcept;

void resolve_smoke_grid_particle_fields(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const std::uint64_t *density_accumulator,
    const std::uint64_t *temperature_accumulator, float *density,
    float *temperature, std::uint32_t count) noexcept;

void splat_smoke_grid_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const float *ages, const float *thermal_lift, std::uint32_t count,
    const SmokeDomainDispatchOptions &options,
    const SmokeGridDispatchField &field, std::uint64_t *density_accumulator,
    std::uint64_t *temperature_accumulator) noexcept;

void advect_smoke_grid_face(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, const float *source,
    float *destination, float timestep) noexcept;

void correct_smoke_grid_face(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &original, const float *source,
    const float *predicted, const float *reversed, float *corrected,
    float timestep) noexcept;

void calculate_smoke_grid_cell_diagnostics(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const SmokeGridDispatchField &field, void *cell_velocity,
    void *vorticity, float *strain, float *divergence) noexcept;

void calculate_smoke_grid_subgrid_force(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const SmokeGridDispatchField &field, const void *vorticity,
    void *force, float strength) noexcept;

void apply_smoke_grid_face_forces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, const float *source,
    float *destination, const float *open, const float *wall,
    const float *strain, const void *subgrid, const float *temperature,
    const SmokeDomainDispatchOptions &options, float timestep,
    const float gravity[3]) noexcept;

void apply_smoke_grid_face_boundaries(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, float *values, const float *open,
    const float *wall, const SmokeDomainDispatchOptions &options) noexcept;

void calculate_smoke_grid_divergence(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const SmokeGridDispatchField &field, float *divergence, float *rhs_max,
    float timestep) noexcept;

void smooth_smoke_grid_pressure(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int resolution, int height,
    float spacing, const float *rhs, const float *pressure, float *next,
    const float *open_x, const float *open_y, const float *open_z,
    const std::uint32_t *converged) noexcept;

void calculate_smoke_grid_pressure_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int resolution, int height,
    float spacing, const float *rhs, const float *pressure, float *residual,
    const float *open_x, const float *open_y, const float *open_z,
    float *maximum, const std::uint32_t *converged) noexcept;

void restrict_smoke_grid_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const float *fine,
    int fine_resolution, int fine_height, float *coarse,
    int coarse_resolution, int coarse_height) noexcept;

void clear_smoke_grid_scalar(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, float *values,
    std::uint32_t count) noexcept;

void restrict_smoke_grid_open_faces(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const float *fine, int fine_resolution, int fine_height, float *coarse,
    int coarse_resolution, int coarse_height) noexcept;

void mark_smoke_grid_domain_boundaries(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, float *open, float *wall,
    const SmokeDomainDispatchOptions &options) noexcept;

void prolong_smoke_grid_pressure(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const float *coarse,
    int coarse_resolution, int coarse_height, float *fine,
    int fine_resolution, int fine_height) noexcept;

void compare_smoke_grid_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const float *rhs_max,
    const float *residual_max, float tolerance,
    std::uint32_t *converged) noexcept;

void reset_smoke_grid_residual(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, float *residual_max,
    const std::uint32_t *converged) noexcept;

void project_smoke_grid_face(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, int axis,
    const SmokeGridDispatchField &field, const float *pressure,
    float *velocity, const float *open, const float *wall,
    float timestep) noexcept;

void trace_smoke_grid_particles(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, float *number_densities, float *pressures,
    void *vorticities, std::uint32_t count,
    const SmokeDomainDispatchOptions &options,
    const SmokeGridDispatchField &field, float timestep,
    const float gravity[3]) noexcept;

void apply_smoke_grid_force_to_soft_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    const std::uint32_t *indices, std::uint32_t index_count,
    const void *bindings, const void *node_velocity_snapshot,
    void *node_velocities, const float *inverse_masses,
    const SmokeGridDispatchField &field, float drag_coefficient,
    float density_scale, float kinematic_viscosity, float les_coefficient,
    float maximum_acceleration, float timestep) noexcept;

void accumulate_smoke_grid_force_on_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t index_count,
    const SmokeGridDispatchField &field, float drag_coefficient,
    float density_scale, float kinematic_viscosity, float les_coefficient,
    void *forces) noexcept;

void apply_smoke_grid_force_to_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *velocities,
    const float *inverse_masses, const void *forces, std::uint32_t count,
    float maximum_acceleration, float timestep) noexcept;

void apply_smoke_grid_force_to_rigid_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *mesh,
    std::uint32_t body, const void *body_parameters, void *body_states,
    const SmokeGridDispatchField &field, float air_density,
    float drag_coefficient, float density_scale, float kinematic_viscosity,
    float les_coefficient, float timestep) noexcept;

} // namespace parallel_mater::simulation
