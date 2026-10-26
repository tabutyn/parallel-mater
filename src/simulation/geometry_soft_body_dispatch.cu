// SPDX-License-Identifier: MIT
#include "geometry_soft_body_dispatch.hpp"

#include "parallel_mater_geometry_soft_body_runtime.cu"

namespace parallel_mater::simulation {
namespace {
template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
Vec3_0 vec3(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}
} // namespace

void collide_deformable_with_rigid_geometry(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const void *previous, const float *inverse_masses,
    std::uint32_t count, float thickness, float timestep,
    const void *body_parameters, const void *previous_body_states,
    const void *body_states, const void *meshes, std::uint32_t body_count,
    std::uint32_t first_excluded_body, std::uint32_t first_excluded_count,
    std::uint32_t last_excluded_body, std::uint32_t last_excluded_count,
    void *impulses, void *contact_forces, void *contact_normals,
    void *contact_arms, void *contact_momentum_delta,
    float *accumulated_normal_delta,
    std::uint32_t *deformable_contact_count,
    std::uint32_t *dynamic_contact_flag, std::uint32_t *contact_flags,
    bool couple_dynamic, bool position_only, bool static_only,
    bool reconstruct_free_velocities, float maximum_speed,
    void *body_position_corrections) noexcept {
    deformable_collide<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)),
        const_cast<float *>(inverse_masses), count, thickness, timestep,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(
            as<RigidBodyState_0>(previous_body_states)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        body_count, first_excluded_body, first_excluded_count,
        last_excluded_body, last_excluded_count,
        as<FluidBodyImpulse_0>(impulses),
        as<Vec3_0>(contact_forces), as<Vec3_0>(contact_normals),
        as<Vec3_0>(contact_arms), as<Vec3_0>(contact_momentum_delta),
        accumulated_normal_delta, deformable_contact_count,
        dynamic_contact_flag, contact_flags, couple_dynamic, position_only,
        static_only, reconstruct_free_velocities, maximum_speed,
        as<Vec3_0>(body_position_corrections));
}

void calculate_soft_body_surface_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *surface,
    const std::uint32_t *indices, std::uint32_t triangle_count,
    const void *body_parameters, const void *body_states,
    const void *meshes, std::uint32_t body_count, void *corrections) noexcept {
    soft_body_surface_contacts<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)),
        const_cast<std::uint32_t *>(indices), triangle_count,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        body_count, as<Vec3_0>(corrections));
}

void apply_soft_body_surface_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    const void *corrections, const std::uint32_t *offsets,
    const void *influences, std::uint32_t node_count) noexcept {
    soft_body_apply_surface_contacts<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(corrections)),
        const_cast<std::uint32_t *>(offsets),
        const_cast<SoftSurfaceInfluence_0 *>(
            as<SoftSurfaceInfluence_0>(influences)),
        node_count);
}

void emit_deformable_spatial_primitives(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *previous_positions,
    const void *predicted_positions, std::uint32_t vertex_count,
    const std::uint32_t *indices,
    const std::uint32_t *primitive_generations,
    std::uint32_t primitive_count, std::uint32_t vertices_per_primitive,
    std::uint32_t family, std::uint32_t owner_slot,
    std::uint32_t owner_generation, std::uint32_t topology_revision,
    std::uint32_t primitive_kind, std::uint32_t leaf_offset,
    std::uint32_t leaf_capacity, std::uint32_t query_offset,
    std::uint32_t query_capacity, std::uint32_t sort_key_base,
    std::uint32_t query_mask, std::uint32_t admission_mask, float thickness,
    const float path_deviation[3], const float correction_budget[3],
    float epsilon, void *metadata, void *bounds, std::uint8_t *build_flags,
    void *query_bounds, std::uint32_t *query_masks,
    std::uint32_t *excluded_sort_keys, std::uint32_t *query_leaf_sources,
    std::uint32_t *status) noexcept {
    pm_emit_deformable_spatial_primitives<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(predicted_positions)), vertex_count,
        const_cast<std::uint32_t *>(indices),
        const_cast<std::uint32_t *>(primitive_generations), primitive_count,
        vertices_per_primitive, family, owner_slot, owner_generation,
        topology_revision, primitive_kind, leaf_offset, leaf_capacity,
        query_offset, query_capacity, sort_key_base, query_mask,
        admission_mask, thickness, vec3(path_deviation),
        vec3(correction_budget), epsilon,
        as<DeformableSpatialLeafMetadata_0>(metadata),
        as<DeformableSpatialAabb_0>(bounds), build_flags,
        as<DeformableSpatialAabb_0>(query_bounds), query_masks,
        excluded_sort_keys, query_leaf_sources, status);
}

void filter_deformable_spatial_candidates(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    const std::uint32_t *query_leaf_sources, std::uint32_t query_count,
    std::uint32_t query_family, std::uint32_t query_owner_slot,
    std::uint32_t query_owner_generation,
    std::uint32_t query_topology_revision,
    std::uint32_t query_primitive_kind,
    const std::uint32_t *query_primitive_generations,
    std::uint32_t query_primitive_count,
    const std::uint32_t *query_indices,
    std::uint32_t query_vertices_per_primitive,
    std::uint32_t target_family, std::uint32_t target_owner_slot,
    std::uint32_t target_owner_generation,
    std::uint32_t target_topology_revision,
    std::uint32_t target_primitive_kind,
    const std::uint32_t *target_primitive_generations,
    std::uint32_t target_primitive_count,
    const std::uint32_t *target_indices,
    std::uint32_t target_vertices_per_primitive,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *filtered_counts,
    std::uint32_t *filtered_sources, std::uint32_t *status) noexcept {
    pm_filter_deformable_spatial_candidates<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<DeformableSpatialLeafMetadata_0 *>(
            as<DeformableSpatialLeafMetadata_0>(metadata)),
        const_cast<std::uint8_t *>(admission), leaf_capacity,
        const_cast<std::uint32_t *>(query_leaf_sources), query_count,
        query_family, query_owner_slot, query_owner_generation,
        query_topology_revision, query_primitive_kind,
        const_cast<std::uint32_t *>(query_primitive_generations),
        query_primitive_count, const_cast<std::uint32_t *>(query_indices),
        query_vertices_per_primitive, target_family, target_owner_slot,
        target_owner_generation, target_topology_revision,
        target_primitive_kind,
        const_cast<std::uint32_t *>(target_primitive_generations),
        target_primitive_count, const_cast<std::uint32_t *>(target_indices),
        target_vertices_per_primitive,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        filtered_counts, filtered_sources, status);
}

void emit_smoke_spatial_queries(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *cell_centers,
    std::uint32_t query_count, std::uint32_t query_offset,
    std::uint32_t query_capacity, std::uint32_t query_mask,
    const float half_extent[3], const float path_deviation[3],
    const float correction_budget[3], float epsilon, void *query_bounds,
    std::uint32_t *query_masks, std::uint32_t *excluded_sort_keys,
    std::uint32_t *status) noexcept {
    pm_emit_smoke_spatial_queries<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(cell_centers)), query_count,
        query_offset, query_capacity, query_mask, vec3(half_extent),
        vec3(path_deviation), vec3(correction_budget), epsilon,
        as<DeformableSpatialAabb_0>(query_bounds), query_masks,
        excluded_sort_keys, status);
}

void filter_smoke_spatial_obstacles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    std::uint32_t query_count, std::uint32_t accepted_family_mask,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *filtered_counts,
    std::uint32_t *filtered_sources, std::uint32_t *status) noexcept {
    pm_filter_smoke_spatial_obstacles<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<DeformableSpatialLeafMetadata_0 *>(
            as<DeformableSpatialLeafMetadata_0>(metadata)),
        const_cast<std::uint8_t *>(admission), leaf_capacity, query_count,
        accepted_family_mask, const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        filtered_counts, filtered_sources, status);
}

} // namespace parallel_mater::simulation
