// SPDX-License-Identifier: MIT
#include "geometry_fluid_dispatch.hpp"

#include "parallel_mater_geometry_fluid_runtime.cu"

namespace parallel_mater::simulation {
namespace {
template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
} // namespace

void collide_fluid_with_static_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const void *previous, float *foam,
    const std::uint32_t *count, float radius, float spawn_clearance,
    std::uint32_t first_spawned, bool recover_spawn, const void *up,
    const void *body_parameters, const void *body_states, const void *meshes,
    const void *body_ids,
    std::uint32_t body_index, float particle_mass, bool collect_contacts,
    void *contact_samples, std::uint8_t *contact_flags,
    std::uint32_t fluid_index, std::uint32_t fluid_generation,
    bool apply_paint, const void *paint_fields,
    std::uint32_t paint_field_capacity, const void *paint_rules,
    std::uint32_t paint_rule_capacity) noexcept {
    const auto fluid_id = FluidId_0{fluid_index, fluid_generation};
    fluid_static_contacts<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)), foam,
        const_cast<std::uint32_t *>(count), radius, spawn_clearance,
        first_spawned, recover_spawn, *as<Vec3_0>(up),
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)),
        body_index, particle_mass, collect_contacts,
        as<FluidContactSample_0>(contact_samples), contact_flags,
        fluid_id, apply_paint,
        const_cast<PaintFieldResource_0 *>(
            as<PaintFieldResource_0>(paint_fields)),
        paint_field_capacity,
        const_cast<PaintRuleResource_0 *>(
            as<PaintRuleResource_0>(paint_rules)),
        paint_rule_capacity);
}

void calculate_fluid_body_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    const void *previous_states, const void *current_states,
    const void *meshes, std::uint32_t body_count, void *bounds) noexcept {
    fluid_body_bounds_kernel<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(previous_states)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(current_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        body_count, as<WorldAabb_0>(bounds));
}

void index_fluid_body_cells(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    const void *bounds, std::uint32_t body_count, std::uint32_t words,
    float padding, void *masks, void *global_masks) noexcept {
    fluid_index_body_cells<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<WorldAabb_0 *>(as<WorldAabb_0>(bounds)), body_count, words,
        padding, as<unsigned long long>(masks),
        as<unsigned long long>(global_masks));
}

void detect_fluid_moving_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *previous, const std::uint32_t *count, float radius,
    const void *body_parameters, const void *previous_states,
    const void *current_states, const void *meshes, const void *bounds,
    std::uint32_t body_count, const void *masks,
    const void *global_masks, std::uint32_t words,
    bool first_iteration, void *contacts, std::uint32_t *contact_counts) noexcept {
    fluid_detect_moving_contacts<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)),
        const_cast<std::uint32_t *>(count), radius,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(previous_states)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(current_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        const_cast<WorldAabb_0 *>(as<WorldAabb_0>(bounds)), body_count,
        const_cast<unsigned long long *>(as<unsigned long long>(masks)),
        const_cast<unsigned long long *>(as<unsigned long long>(global_masks)),
        words, first_iteration,
        as<FluidMovingContact_0>(contacts), contact_counts);
}

void resolve_fluid_moving_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, float *foam, const std::uint32_t *count, float radius,
    float particle_mass, float timestep, float maximum_speed,
    const void *body_parameters, const void *body_states, const void *meshes,
    const void *contacts, const std::uint32_t *contact_counts, void *impulses,
    bool collect_contacts, void *contact_samples,
    std::uint8_t *particle_contact_flags, std::uint32_t fluid_index,
    std::uint32_t fluid_generation, const void *body_ids, bool apply_paint,
    const void *paint_fields, std::uint32_t paint_field_capacity,
    const void *paint_rules, std::uint32_t paint_rule_capacity) noexcept {
    const auto fluid_id = FluidId_0{fluid_index, fluid_generation};
    fluid_resolve_moving_contacts<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities), foam,
        const_cast<std::uint32_t *>(count), radius, particle_mass, timestep,
        maximum_speed,
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        const_cast<FluidMovingContact_0 *>(
            as<FluidMovingContact_0>(contacts)),
        const_cast<std::uint32_t *>(contact_counts),
        as<FluidBodyImpulse_0>(impulses), collect_contacts,
        as<FluidContactSample_0>(contact_samples), particle_contact_flags,
        fluid_id,
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), apply_paint,
        const_cast<PaintFieldResource_0 *>(
            as<PaintFieldResource_0>(paint_fields)),
        paint_field_capacity,
        const_cast<PaintRuleResource_0 *>(
            as<PaintRuleResource_0>(paint_rules)),
        paint_rule_capacity);
}

void reduce_fluid_body_impulses(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *impulses,
    const std::uint32_t *particle_count, const void *body_parameters,
    void *body_states, const std::uint32_t *contact_flags,
    std::uint32_t body_count, const void *position_corrections) noexcept {
    reduce_point_body_impulses<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<FluidBodyImpulse_0 *>(as<FluidBodyImpulse_0>(impulses)),
        const_cast<std::uint32_t *>(particle_count),
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states),
        const_cast<std::uint32_t *>(contact_flags), body_count,
        const_cast<Vec3_0 *>(as<Vec3_0>(position_corrections)));
}

} // namespace parallel_mater::simulation
