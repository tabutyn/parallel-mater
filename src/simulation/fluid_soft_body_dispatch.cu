// SPDX-License-Identifier: MIT
#include "fluid_soft_body_dispatch.hpp"

#include "parallel_mater_fluid_soft_body_runtime.cu"

namespace parallel_mater::simulation {
namespace {
template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
} // namespace

void refit_fluid_soft_surface(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    const void *previous_surface, const std::uint32_t *surface_indices,
    const std::uint32_t *triangle_order, void *tree,
    const std::uint32_t *parents, std::uint32_t *ready,
    std::uint32_t tree_count, void *bounds) noexcept {
    fluid_soft_refit<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_surface)),
        const_cast<std::uint32_t *>(surface_indices),
        const_cast<std::uint32_t *>(triangle_order), as<BvhNode_0>(tree),
        const_cast<std::uint32_t *>(parents), ready, tree_count,
        as<Vec3_0>(bounds));
}

void detect_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *previous, const void *velocities,
    const std::uint32_t *particle_count, const void *surface,
    const void *previous_surface, const std::uint32_t *surface_indices,
    std::uint32_t surface_index_count, const void *surface_bindings,
    const void *soft_velocities, float orientation, float distance,
    bool sweep, const std::uint32_t *triangle_generations,
    std::uint32_t soft_owner_slot, std::uint32_t soft_owner_generation,
    std::uint32_t soft_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    void *contacts,
    std::uint32_t *node_contact_counts, std::uint32_t *contact_count,
    float *maximum_penetration) noexcept {
    fluid_soft_detect<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<std::uint32_t *>(particle_count),
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_surface)),
        const_cast<std::uint32_t *>(surface_indices), surface_index_count,
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(surface_bindings)),
        const_cast<Vec3_0 *>(as<Vec3_0>(soft_velocities)), orientation,
        distance, sweep, const_cast<std::uint32_t *>(triangle_generations),
        soft_owner_slot, soft_owner_generation, soft_topology_revision,
        const_cast<CouplingSpatialLeafMetadata_0 *>(
            as<CouplingSpatialLeafMetadata_0>(spatial_metadata)),
        const_cast<std::uint8_t *>(spatial_admission), spatial_leaf_capacity,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        spatial_status,
        as<FluidSoftContact_0>(contacts), node_contact_counts, contact_count,
        maximum_penetration);
}

void solve_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, void *accelerations, float *foam,
    const std::uint32_t *particle_count, const void *contacts,
    const std::uint32_t *node_contact_counts, const float *inverse_masses,
    const void *soft_velocities, float maximum_soft_speed,
    float inverse_particle_mass, float friction, float maximum_projection,
    float timestep, void *node_position_deltas, void *node_impulses) noexcept {
    fluid_soft_solve<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        as<Vec3_0>(accelerations), foam,
        const_cast<std::uint32_t *>(particle_count),
        const_cast<FluidSoftContact_0 *>(as<FluidSoftContact_0>(contacts)),
        const_cast<std::uint32_t *>(node_contact_counts),
        const_cast<float *>(inverse_masses),
        const_cast<Vec3_0 *>(as<Vec3_0>(soft_velocities)), maximum_soft_speed,
        inverse_particle_mass, friction, maximum_projection, timestep,
        as<Vec3_0>(node_position_deltas), as<Vec3_0>(node_impulses));
}

void apply_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const float *inverse_masses,
    const void *position_deltas, const void *impulses, void *forces,
    std::uint32_t node_count, float inverse_frame_timestep) noexcept {
    fluid_soft_apply<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        const_cast<float *>(inverse_masses),
        const_cast<Vec3_0 *>(as<Vec3_0>(position_deltas)),
        const_cast<Vec3_0 *>(as<Vec3_0>(impulses)), as<Vec3_0>(forces),
        node_count, inverse_frame_timestep);
}

void recover_fluid_soft_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *previous, const std::uint32_t *particle_count,
    const void *surface, const void *previous_surface,
    const std::uint32_t *surface_indices, std::uint32_t surface_index_count,
    float orientation, float distance, bool sweep,
    const std::uint32_t *triangle_generations,
    std::uint32_t soft_owner_slot, std::uint32_t soft_owner_generation,
    std::uint32_t soft_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query,
    std::uint32_t *spatial_status) noexcept {
    fluid_soft_recover<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), const_cast<Vec3_0 *>(as<Vec3_0>(previous)),
        const_cast<std::uint32_t *>(particle_count),
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_surface)),
        const_cast<std::uint32_t *>(surface_indices), surface_index_count,
        orientation, distance, sweep,
        const_cast<std::uint32_t *>(triangle_generations), soft_owner_slot,
        soft_owner_generation, soft_topology_revision,
        const_cast<CouplingSpatialLeafMetadata_0 *>(
            as<CouplingSpatialLeafMetadata_0>(spatial_metadata)),
        const_cast<std::uint8_t *>(spatial_admission), spatial_leaf_capacity,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        spatial_status);
}

} // namespace parallel_mater::simulation
