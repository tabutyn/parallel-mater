// SPDX-License-Identifier: MIT
#include "geometry_cloth_dispatch.hpp"

#include "parallel_mater_geometry_cloth_runtime.cu"

namespace parallel_mater::simulation {
namespace {
template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
} // namespace

void constrain_rigid_bodies_with_cloth(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *cloth_positions,
    const void *cloth_velocities, const std::uint32_t *cloth_indices,
    const std::uint32_t *vertex_sources, const void *surface_positions,
    const float *inverse_masses, std::uint32_t vertex_count,
    std::uint32_t triangle_count, float thickness, float timestep,
    float contact_friction, const void *body_parameters,
    const void *previous_body_states, void *body_states, const void *meshes,
    const void *body_ids, std::uint32_t body_count,
    std::uint32_t cloth_index, std::uint32_t cloth_generation,
    const void *paint_fields, std::uint32_t paint_field_capacity,
    const void *paint_rules, std::uint32_t paint_rule_capacity,
    void *corrections) noexcept {
    cloth_constrain_bodies<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(cloth_velocities)),
        const_cast<std::uint32_t *>(cloth_indices),
        const_cast<std::uint32_t *>(vertex_sources),
        const_cast<Vec3_0 *>(as<Vec3_0>(surface_positions)),
        const_cast<float *>(inverse_masses), vertex_count, triangle_count,
        thickness, timestep, contact_friction,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(
            as<RigidBodyState_0>(previous_body_states)),
        as<RigidBodyState_0>(body_states),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        cloth_index, cloth_generation,
        const_cast<PaintFieldResource_0 *>(
            as<PaintFieldResource_0>(paint_fields)),
        paint_field_capacity,
        const_cast<PaintRuleResource_0 *>(
            as<PaintRuleResource_0>(paint_rules)),
        paint_rule_capacity, as<ClothBodyCorrection_0>(corrections));
}

void apply_cloth_body_corrections(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const float *inverse_masses, std::uint32_t vertex_count,
    const void *corrections, std::uint32_t body_count) noexcept {
    cloth_apply_body_corrections<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        const_cast<float *>(inverse_masses), vertex_count,
        const_cast<ClothBodyCorrection_0 *>(
            as<ClothBodyCorrection_0>(corrections)),
        body_count);
}

} // namespace parallel_mater::simulation
