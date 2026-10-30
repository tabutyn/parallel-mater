// SPDX-License-Identifier: MIT
#include "rope_coupling_dispatch.hpp"

#include "parallel_mater_soft_body_rope_runtime.cu"

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
Vec3_0 vector(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}

} // namespace

void apply_realtime_rope_soft_body_contacts(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *rope_positions,
    void *rope_previous_positions, void *rope_velocities,
    const float *rope_inverse_masses, std::uint32_t rope_count,
    float rope_maximum_speed, bool attach_first, bool attach_last,
    void *rope_forces, const void *surface, const void *previous_surface,
    const std::uint32_t *indices, std::uint32_t index_count,
    const void *bindings,
    const void *soft_velocities, const float *soft_inverse_masses,
    float orientation, float distance, float friction,
    const std::uint32_t *triangle_generations,
    std::uint32_t soft_owner_slot, std::uint32_t soft_owner_generation,
    std::uint32_t soft_topology_revision, const void *spatial_metadata,
    const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    void *node_impulses,
    std::uint32_t *contact_count, float *maximum_penetration,
    float timestep) noexcept {
    rope_soft_realtime_contact<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(rope_positions), as<Vec3_0>(rope_previous_positions),
        as<Vec3_0>(rope_velocities),
        const_cast<float *>(rope_inverse_masses), rope_count,
        rope_maximum_speed, attach_first, attach_last,
        as<Vec3_0>(rope_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_surface)),
        const_cast<std::uint32_t *>(indices), index_count,
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        const_cast<Vec3_0 *>(as<Vec3_0>(soft_velocities)),
        const_cast<float *>(soft_inverse_masses), orientation, distance,
        friction, const_cast<std::uint32_t *>(triangle_generations),
        soft_owner_slot, soft_owner_generation, soft_topology_revision,
        const_cast<CouplingSpatialLeafMetadata_0 *>(
            as<CouplingSpatialLeafMetadata_0>(spatial_metadata)),
        const_cast<std::uint8_t *>(spatial_admission), spatial_leaf_capacity,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources),
        candidates_per_query, spatial_status, as<Vec3_0>(node_impulses),
        contact_count, maximum_penetration, timestep);
}

void sample_rope_soft_body_anchor(device::ExecutionQueue stream,
    void *anchor_positions, void *anchor_velocities, void *anchor_impulses,
    float *anchor_inverse_masses, std::uint32_t end, const void *surface,
    const void *velocities, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    const float offset[3]) noexcept {
    rope_soft_sample_anchor<<<1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(anchor_positions), as<Vec3_0>(anchor_velocities),
        as<Vec3_0>(anchor_impulses), anchor_inverse_masses, end,
        const_cast<Vec3_0 *>(as<Vec3_0>(surface)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<std::uint32_t *>(indices),
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        triangle, vector(weights), vector(offset));
}

void scatter_rope_soft_body_anchor(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *anchor_positions,
    const void *anchor_impulses, std::uint32_t end,
    const void *node_positions, std::uint32_t node_count,
    float support_radius, float contact_support_radius,
    const std::uint32_t *contact_count, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    void *node_impulses) noexcept {
    rope_soft_scatter_anchor<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(anchor_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(anchor_impulses)), end,
        const_cast<Vec3_0 *>(as<Vec3_0>(node_positions)), node_count,
        support_radius, contact_support_radius,
        const_cast<std::uint32_t *>(contact_count),
        const_cast<std::uint32_t *>(indices),
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        triangle, vector(weights), as<Vec3_0>(node_impulses));
}

void apply_rope_soft_body_impulses(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, const void *impulses, std::uint32_t count,
    float maximum_acceleration, float maximum_speed, float force_scale,
    float timestep, float frame_inverse_timestep) noexcept {
    rope_soft_apply<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities), as<Vec3_0>(forces),
        const_cast<float *>(inverse_masses),
        const_cast<Vec3_0 *>(as<Vec3_0>(impulses)), count,
        maximum_acceleration, maximum_speed, force_scale, timestep,
        frame_inverse_timestep);
}

void sync_rope_soft_body_attachment(device::ExecutionQueue stream,
    void *rope_positions, void *rope_previous_positions,
    void *rope_velocities, std::uint32_t rope_count,
    void *rope_anchor_targets,
    const void *anchor_positions, const void *anchor_velocities,
    std::uint32_t end, float timestep) noexcept {
    rope_soft_sync_attachment<<<1U, 1U, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(rope_positions), as<Vec3_0>(rope_previous_positions),
        as<Vec3_0>(rope_velocities), rope_count,
        as<Vec3_0>(rope_anchor_targets),
        const_cast<Vec3_0 *>(as<Vec3_0>(anchor_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(anchor_velocities)), end, timestep);
}

void accumulate_rope_soft_body_attachment_reaction(
    device::ExecutionQueue stream,
    const void *previous_rope_velocities,
    const void *current_rope_velocities,
    const float *rope_inverse_masses, std::uint32_t rope_count,
    std::uint32_t attached_end_mask, std::uint32_t attached_end_count,
    const float gravity[3], float timestep, void *soft_forces,
    const std::uint32_t *indices, const void *bindings,
    std::uint32_t triangle, const float weights[3]) noexcept {
    rope_soft_accumulate_attachment_reaction<<<1U, 1U, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_rope_velocities)),
        const_cast<Vec3_0 *>(as<Vec3_0>(current_rope_velocities)),
        const_cast<float *>(rope_inverse_masses), rope_count,
        attached_end_mask, attached_end_count, vector(gravity), timestep,
        as<Vec3_0>(soft_forces), const_cast<std::uint32_t *>(indices),
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        triangle, vector(weights));
}

} // namespace parallel_mater::simulation
