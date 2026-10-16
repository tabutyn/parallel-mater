// SPDX-License-Identifier: MIT
#include "smoke_coupling_dispatch.hpp"

#include "metal_dispatch_options.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
using metal_bindings::smoke_options;

} // namespace

void apply_smoke_wind_to_soft_body(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    void *velocities, const float *inverse_masses, std::uint32_t node_count,
    const SmokeDispatchOptions &smoke, const void *smoke_positions,
    const void *smoke_velocities, const std::uint64_t *keys,
    const std::uint32_t *indices, const SmokeGridDispatchField &smoke_grid,
    float drag, float maximum_acceleration, float timestep,
    float maximum_speed) noexcept {
    (void)metal::launch(stream, "smoke_soft_body_wind", grid,
        positions, velocities,
        inverse_masses, node_count, smoke_options(smoke),
        smoke_positions,
        smoke_velocities,
        keys,
        indices, metal::grid_record(smoke_grid), drag,
        maximum_acceleration, timestep, maximum_speed);
}

void collide_smoke_with_soft_body(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    std::uint32_t count, const SmokeDispatchOptions &smoke,
    const void *surface_positions, const void *surface_bindings,
    const std::uint32_t *surface_indices, std::uint32_t index_count,
    const void *node_velocities, const void *minimum, const void *maximum,
    float clearance, const SmokeGridDispatchField &smoke_grid) noexcept {
    (void)metal::launch(stream, "smoke_soft_body_contact", grid,
        positions,
        previous_positions,
        velocities, ages, count,
        smoke.lifetime,
        surface_positions,
        surface_bindings,
        surface_indices, index_count,
        node_velocities,
        minimum,
        maximum, clearance,
        metal::grid_record(smoke_grid));
}

} // namespace parallel_mater::simulation
