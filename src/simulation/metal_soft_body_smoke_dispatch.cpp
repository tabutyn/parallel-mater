// SPDX-License-Identifier: MIT
#include "smoke_coupling_dispatch.hpp"

#include "metal_dispatch_options.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
using metal_bindings::smoke_options;

} // namespace

void apply_smoke_wind_to_soft_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *velocities, const float *inverse_masses, std::uint32_t node_count,
    const SmokeDispatchOptions &smoke, const void *smoke_positions,
    const void *smoke_velocities, std::uint32_t smoke_owner_slot,
    std::uint32_t smoke_owner_generation,
    std::uint32_t smoke_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    const SmokeGridDispatchField &smoke_grid,
    float drag, float maximum_acceleration, float timestep,
    float maximum_speed) noexcept {
    (void)metal::launch(stream, "smoke_soft_body_wind", grid,
        positions, velocities,
        inverse_masses, node_count, smoke_options(smoke),
        smoke_positions,
        smoke_velocities,
        smoke_owner_slot, smoke_owner_generation, smoke_topology_revision,
        spatial_metadata, spatial_admission, spatial_leaf_capacity,
        candidate_counts, candidate_sources, candidates_per_query,
        spatial_status, metal::grid_record(smoke_grid), drag,
        maximum_acceleration, timestep, maximum_speed);
}

void collide_smoke_with_soft_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous_positions, void *velocities, const float *ages,
    std::uint32_t count, const SmokeDispatchOptions &smoke,
    const void *surface_positions, const void *surface_bindings,
    const std::uint32_t *surface_indices, std::uint32_t index_count,
    const void *node_velocities, const void *minimum, const void *maximum,
    float clearance, const std::uint32_t *triangle_generations,
    std::uint32_t soft_owner_slot, std::uint32_t soft_owner_generation,
    std::uint32_t soft_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    const SmokeGridDispatchField &smoke_grid) noexcept {
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
        triangle_generations, soft_owner_slot, soft_owner_generation,
        soft_topology_revision, spatial_metadata, spatial_admission,
        spatial_leaf_capacity, candidate_counts, candidate_sources,
        candidates_per_query, spatial_status,
        metal::grid_record(smoke_grid));
}

} // namespace parallel_mater::simulation
