// SPDX-License-Identifier: MIT
#include "fluid_cloth_dispatch.hpp"

#include "parallel_mater_fluid_cloth_runtime.cu"

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
FluidClothCouplingOptions_0 converted(
    const FluidClothDispatchOptions &value) noexcept {
    FluidClothCouplingOptions_0 result{};
    result.contact_distance_0 = value.contact_distance;
    result.interaction_radius_0 = value.interaction_radius;
    result.stiffness_0 = value.stiffness;
    result.damping_0 = value.damping;
    result.tangential_drag_0 = value.tangential_drag;
    result.maximum_force_0 = value.maximum_force;
    result.enabled_0 = true;
    return result;
}

} // namespace

void apply_fluid_cloth_forces(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *particle_positions, const void *particle_velocities,
    const std::uint32_t *particle_count, void *particle_accelerations,
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
    const FluidClothDispatchOptions &options, void *cloth_forces) noexcept {
    fluid_cloth_containment_forces<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(particle_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(particle_velocities)),
        const_cast<std::uint32_t *>(particle_count),
        as<Vec3_0>(particle_accelerations), foam_source, particle_mass,
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_velocities)),
        const_cast<std::uint32_t *>(cloth_indices), triangle_count,
        const_cast<std::uint32_t *>(triangle_generations), cloth_owner_slot,
        cloth_owner_generation, cloth_topology_revision,
        const_cast<CouplingSpatialLeafMetadata_0 *>(
            as<CouplingSpatialLeafMetadata_0>(spatial_metadata)),
        const_cast<std::uint8_t *>(spatial_admission), spatial_leaf_capacity,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        spatial_status,
        cloth_orientation, converted(options), as<Vec3_0>(cloth_forces));
}

void apply_fluid_forces_to_cloth(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *positions, void *velocities, const float *inverse_masses,
    const void *forces, std::uint32_t count, float timestep) noexcept {
    cloth_apply_fluid_forces<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        const_cast<float *>(inverse_masses),
        const_cast<Vec3_0 *>(as<Vec3_0>(forces)), count, timestep);
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
    fluid_project_inside_cloth<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(particle_positions), as<Vec3_0>(particle_velocities),
        const_cast<std::uint32_t *>(particle_count), contact_distance,
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_velocities)),
        const_cast<std::uint32_t *>(cloth_indices), triangle_count,
        const_cast<std::uint32_t *>(triangle_generations), cloth_owner_slot,
        cloth_owner_generation, cloth_topology_revision,
        const_cast<CouplingSpatialLeafMetadata_0 *>(
            as<CouplingSpatialLeafMetadata_0>(spatial_metadata)),
        const_cast<std::uint8_t *>(spatial_admission), spatial_leaf_capacity,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        spatial_status,
        cloth_orientation);
}

} // namespace parallel_mater::simulation
