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

void apply_fluid_cloth_forces(dim3 grid, dim3 block, cudaStream_t stream,
    const void *particle_positions, const void *particle_velocities,
    const std::uint32_t *particle_count, void *particle_accelerations,
    float *foam_source, float particle_mass, const void *cloth_positions,
    const void *cloth_velocities, const std::uint32_t *cloth_indices,
    std::uint32_t triangle_count, float cloth_orientation,
    const FluidClothDispatchOptions &options, void *cloth_forces) noexcept {
    (void)metal::launch(stream, "fluid_cloth_containment_forces", grid,
        particle_positions,
        particle_velocities,
        particle_count,
        particle_accelerations, foam_source, particle_mass,
        cloth_positions,
        cloth_velocities,
        cloth_indices, triangle_count,
        cloth_orientation, converted(options), cloth_forces);
}

void apply_fluid_forces_to_cloth(dim3 grid, dim3 block, cudaStream_t stream,
    void *positions, void *velocities, const float *inverse_masses,
    const void *forces, std::uint32_t count, float timestep) noexcept {
    (void)metal::launch(stream, "cloth_apply_fluid_forces", grid,
        positions, velocities,
        inverse_masses,
        forces, count, timestep);
}

void project_fluid_inside_cloth(dim3 grid, dim3 block, cudaStream_t stream,
    void *particle_positions, void *particle_velocities,
    const std::uint32_t *particle_count, float contact_distance,
    const void *cloth_positions, const void *cloth_velocities,
    const std::uint32_t *cloth_indices, std::uint32_t triangle_count,
    float cloth_orientation) noexcept {
    (void)metal::launch(stream, "fluid_project_inside_cloth", grid,
        particle_positions, particle_velocities,
        particle_count, contact_distance,
        cloth_positions,
        cloth_velocities,
        cloth_indices, triangle_count,
        cloth_orientation);
}

} // namespace parallel_mater::simulation
