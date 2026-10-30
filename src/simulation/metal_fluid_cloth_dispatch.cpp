// SPDX-License-Identifier: MIT
#include "fluid_cloth_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"
#include <array>

namespace parallel_mater::simulation {
namespace {
FluidClothCouplingOptions converted(
    const FluidClothDispatchOptions &value) noexcept {
    FluidClothCouplingOptions result{};
    result.contact_distance = value.contact_distance;
    result.interaction_radius = value.interaction_radius;
    result.stiffness = value.stiffness;
    result.damping = value.damping;
    result.tangential_drag = value.tangential_drag;
    result.maximum_force = value.maximum_force;
    result.enabled = true;
    return result;
}

} // namespace

void apply_fluid_cloth_forces(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *particle_positions, const void *particle_velocities,
    const std::uint32_t *particle_count, void *particle_velocity_deltas,
    void *particle_accelerations,
    float *foam_source, float particle_mass, const void *cloth_positions,
    const void *cloth_velocities, const std::uint32_t *cloth_indices,
    std::uint32_t triangle_count,
    const std::uint32_t *triangle_generations,
    std::uint32_t cloth_owner_slot, std::uint32_t cloth_owner_generation,
    std::uint32_t cloth_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    float cloth_orientation,
    const FluidClothDispatchOptions &options, float timestep,
    float force_average_scale, void *cloth_step_forces,
    void *cloth_average_forces) noexcept {
    (void)metal::launch(stream, "fluid_cloth_containment_forces", grid,
        particle_positions,
        particle_velocities,
        particle_count,
        particle_velocity_deltas, particle_accelerations,
        foam_source, particle_mass,
        cloth_positions,
        cloth_velocities,
        cloth_indices, triangle_count, triangle_generations,
        cloth_owner_slot, cloth_owner_generation, cloth_topology_revision,
        spatial_metadata, spatial_admission, spatial_leaf_capacity,
        candidate_counts, candidate_sources, candidates_per_query,
        spatial_status,
        cloth_orientation, converted(options), timestep, force_average_scale,
        cloth_step_forces, cloth_average_forces);
}

void apply_fluid_forces_to_cloth(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *positions, void *velocities, const float *inverse_masses,
    const void *forces, std::uint32_t count, float impulse_timestep,
    float position_timestep) noexcept {
    (void)metal::launch(stream, "cloth_apply_fluid_forces", grid,
        positions, velocities,
        inverse_masses,
        forces, count, impulse_timestep, position_timestep);
}

void project_fluid_inside_cloth(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *particle_positions, void *particle_velocities,
    const std::uint32_t *particle_count, float contact_distance,
    const void *cloth_positions, const void *cloth_velocities,
    const std::uint32_t *cloth_indices, std::uint32_t triangle_count,
    const std::uint32_t *triangle_generations,
    std::uint32_t cloth_owner_slot, std::uint32_t cloth_owner_generation,
    std::uint32_t cloth_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    float cloth_orientation) noexcept {
    (void)metal::launch(stream, "fluid_project_inside_cloth", grid,
        particle_positions, particle_velocities,
        particle_count, contact_distance,
        cloth_positions,
        cloth_velocities,
        cloth_indices, triangle_count, triangle_generations,
        cloth_owner_slot, cloth_owner_generation, cloth_topology_revision,
        spatial_metadata, spatial_admission, spatial_leaf_capacity,
        candidate_counts, candidate_sources, candidates_per_query,
        spatial_status,
        cloth_orientation);
}

} // namespace parallel_mater::simulation
