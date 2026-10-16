// SPDX-License-Identifier: MIT
#include "geometry_smoke_dispatch.hpp"

#include "metal_dispatch_options.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
using metal_bindings::vector;
using metal_bindings::smoke_options;
} // namespace

void index_smoke_particles(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, void *keys,
    std::uint32_t *indices) noexcept {
    (void)metal::launch(stream, "smoke_emit_cells", grid,
        positions,
        ages, count, smoke_options(options),
        keys, indices);
}

void calculate_smoke_density_pressure(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *keys,
    const std::uint32_t *indices, float *densities, float *pressures,
    float timestep) noexcept {
    (void)metal::launch(stream, "smoke_density_pressure", grid,
        positions,
        ages, count, smoke_options(options),
        keys,
        indices, densities, pressures, timestep);
}

void calculate_smoke_vorticity(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *velocities, const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *keys,
    const std::uint32_t *indices, const float *densities,
    void *vorticities, float *magnitudes) noexcept {
    (void)metal::launch(stream, "smoke_compute_vorticity", grid,
        positions,
        velocities,
        ages, count, smoke_options(options),
        keys,
        indices, densities,
        vorticities, magnitudes);
}

void calculate_smoke_pair_forces(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *velocities, const float *ages, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *keys,
    const std::uint32_t *indices, const float *densities,
    const float *pressures, const void *vorticities,
    const float *vorticity_magnitudes, void *accelerations) noexcept {
    (void)metal::launch(stream, "smoke_pair_forces", grid,
        positions,
        velocities,
        ages, count, smoke_options(options),
        keys,
        indices, densities,
        pressures,
        vorticities,
        vorticity_magnitudes, accelerations);
}

void advect_smoke_particles(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, std::uint32_t count,
    const SmokeDomainDispatchOptions &options, const void *accelerations,
    float timestep, const float gravity[3]) noexcept {
    (void)metal::launch(stream, "smoke_advect", grid,
        positions, previous_positions,
        velocities, ages, thermal_lift, count,
        smoke_options(options),
        accelerations, timestep,
        vector(gravity));
}

void emit_smoke_particles(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *previous_positions, void *velocities, float *ages,
    float *thermal_lift, float *densities, float *pressures,
    void *vorticities, const SmokeDomainDispatchOptions &options,
    std::uint32_t first_slot, std::uint32_t count,
    std::uint64_t first_serial) noexcept {
    (void)metal::launch(stream, "smoke_emit", grid,
        positions, previous_positions,
        velocities, ages, thermal_lift, densities, pressures,
        vorticities, smoke_options(options), first_slot, count,
        static_cast<unsigned long long>(first_serial));
}

void calculate_smoke_deformable_bounds(
    dim3 grid, dim3 block, cudaStream_t stream, const void *surface,
    std::uint32_t count, void *minimum, void *maximum) noexcept {
    (void)metal::launch(stream, "smoke_deformable_bounds", grid,
        surface, count,
        minimum, maximum);
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
    (void)metal::launch(stream, "smoke_rigid_contact", grid,
        positions,
        previous_positions,
        velocities, ages, pressures, count,
        lifetime, metal::mesh_record(*static_cast<const TriangleMeshResource *>(mesh)),
        body_states,
        previous_body_states,
        body, clearance, particle_mass, wall_drag, pressure_stiffness,
        timestep, maximum_speed, metal::grid_record(smoke_grid),
        impulses);
}

void apply_smoke_rigid_body_impulses(
    dim3 grid, dim3 block, cudaStream_t stream, const void *impulses,
    std::uint32_t count, std::uint32_t body, const void *body_parameters,
    void *body_states) noexcept {
    (void)metal::launch(stream, "smoke_apply_rigid_impulses", grid,
        impulses,
        count, body,
        body_parameters,
        body_states);
}

} // namespace parallel_mater::simulation
