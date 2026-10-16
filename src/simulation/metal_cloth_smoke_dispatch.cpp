// SPDX-License-Identifier: MIT
#include "smoke_coupling_dispatch.hpp"

#include "metal_dispatch_options.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
using metal_bindings::smoke_options;

} // namespace

void apply_smoke_wind_to_cloth(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    void *velocities, const float *inverse_masses, std::uint32_t vertex_count,
    const SmokeDispatchOptions &smoke, const void *smoke_positions,
    const void *smoke_velocities, const std::uint64_t *keys,
    const std::uint32_t *indices, const SmokeGridDispatchField &smoke_grid,
    float drag, float maximum_acceleration, float timestep) noexcept {
    (void)metal::launch(stream, "smoke_cloth_wind", grid,
        positions, velocities,
        inverse_masses, vertex_count, smoke_options(smoke),
        smoke_positions,
        smoke_velocities,
        keys,
        indices, metal::grid_record(smoke_grid), drag,
        maximum_acceleration, timestep);
}

void collide_smoke_with_cloth(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    float *pressures, std::uint32_t count,
    const SmokeDispatchOptions &smoke, const void *cloth_positions,
    const void *cloth_velocities, const std::uint32_t *cloth_indices,
    std::uint32_t index_count, const void *minimum, const void *maximum,
    float clearance, const SmokeGridDispatchField &smoke_grid) noexcept {
    (void)metal::launch(stream, "smoke_cloth_contact", grid,
        positions,
        previous_positions,
        velocities, ages, pressures, count,
        smoke.lifetime,
        cloth_positions,
        cloth_velocities,
        cloth_indices, index_count,
        minimum,
        maximum, clearance,
        smoke.pressure_stiffness, smoke.maximum_speed, metal::grid_record(smoke_grid));
}

} // namespace parallel_mater::simulation
