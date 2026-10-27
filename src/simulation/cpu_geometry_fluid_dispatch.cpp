// SPDX-License-Identifier: MIT
#include "geometry_fluid_dispatch.hpp"

#include "../device/cpu_dispatch.hpp"
#include "parallel_mater_geometry_fluid_runtime.cpp"

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
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, void *positions,
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
    EntryPointParams_0 args{
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
        paint_rule_capacity};
    device::dispatch_cpu(grid, block, fluid_static_contacts_Thread, &args);
}

void calculate_fluid_body_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, const void *body_parameters,
    const void *previous_states, const void *current_states,
    const void *meshes, std::uint32_t body_count, void *bounds) noexcept {
    EntryPointParams_1 args{
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(previous_states)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(current_states)),
        const_cast<TriangleMeshResource_0 *>(
            as<TriangleMeshResource_0>(meshes)),
        body_count, as<WorldAabb_0>(bounds)};
    device::dispatch_cpu(grid, block, fluid_body_bounds_kernel_Thread, &args);
}

void index_fluid_body_cells(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, const void *body_parameters,
    const void *bounds, std::uint32_t body_count, std::uint32_t words,
    float padding, void *masks, void *global_masks) noexcept {
    EntryPointParams_2 args{
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        const_cast<WorldAabb_0 *>(as<WorldAabb_0>(bounds)), body_count, words,
        padding, as<std::uint64_t>(masks),
        as<std::uint64_t>(global_masks)};
    device::dispatch_cpu(grid, block, fluid_index_body_cells_Thread, &args);
}

void detect_fluid_moving_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, const void *positions,
    const void *previous, const std::uint32_t *count, float radius,
    const void *body_parameters, const void *previous_states,
    const void *current_states, const void *meshes, const void *body_ids,
    std::uint32_t body_count, std::uint32_t body_topology_revision,
    const void *spatial_metadata, const std::uint8_t *spatial_admission,
    std::uint32_t spatial_leaf_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *spatial_status,
    bool first_iteration, void *contacts, std::uint32_t *contact_counts) noexcept {
    EntryPointParams_3 args{};
    args.positions_1 = const_cast<Vec3_0 *>(as<Vec3_0>(positions));
    args.previous_2 = const_cast<Vec3_0 *>(as<Vec3_0>(previous));
    args.count_2 = const_cast<std::uint32_t *>(count);
    args.radius_1 = radius;
    args.parameters_3 =
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters));
    args.previous_states_0 =
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(previous_states));
    args.states_1 =
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(current_states));
    args.meshes_2 = const_cast<TriangleMeshResource_0 *>(
        as<TriangleMeshResource_0>(meshes));
    args.body_ids_1 = const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids));
    args.body_count_1 = body_count;
    args.body_topology_revision_0 = body_topology_revision;
    args.spatial_metadata_0 = const_cast<CouplingSpatialLeafMetadata_0 *>(
        as<CouplingSpatialLeafMetadata_0>(spatial_metadata));
    args.spatial_admission_0 = const_cast<std::uint8_t *>(spatial_admission);
    args.spatial_leaf_capacity_0 = spatial_leaf_capacity;
    args.candidate_counts_0 = const_cast<std::uint32_t *>(candidate_counts);
    args.candidate_sources_0 = const_cast<std::uint32_t *>(candidate_sources);
    args.candidates_per_query_0 = candidates_per_query;
    args.spatial_status_0 = spatial_status;
    args.first_iteration_0 = first_iteration;
    args.contacts_0 = as<FluidMovingContact_0>(contacts);
    args.contact_counts_0 = contact_counts;
    device::dispatch_cpu(grid, block, fluid_detect_moving_contacts_Thread, &args);
}

void resolve_fluid_moving_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, void *positions,
    const void *previous_positions, void *velocities, void *velocity_deltas,
    float *foam, const std::uint32_t *count, float radius,
    float particle_mass, float timestep, float maximum_speed,
    const void *body_parameters, const void *body_states, const void *meshes,
    const void *contacts, const std::uint32_t *contact_counts, void *impulses,
    bool collect_contacts, void *contact_samples,
    std::uint8_t *particle_contact_flags, std::uint32_t fluid_index,
    std::uint32_t fluid_generation, const void *body_ids, bool apply_paint,
    const void *paint_fields, std::uint32_t paint_field_capacity,
    const void *paint_rules, std::uint32_t paint_rule_capacity,
    float frame_inverse_timestep, float commit_velocity_scale) noexcept {
    const auto fluid_id = FluidId_0{fluid_index, fluid_generation};
    EntryPointParams_4 args{
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        as<Vec3_0>(velocities), as<Vec3_0>(velocity_deltas), foam,
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
        paint_rule_capacity, frame_inverse_timestep, commit_velocity_scale};
    device::dispatch_cpu(grid, block, fluid_resolve_moving_contacts_Thread, &args);
}

void reduce_fluid_body_impulses(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, const void *impulses,
    const std::uint32_t *particle_count, const void *body_parameters,
    void *body_states, const std::uint32_t *contact_flags,
    std::uint32_t body_count, const void *position_corrections) noexcept {
    EntryPointParams_5 args{
        const_cast<FluidBodyImpulse_0 *>(as<FluidBodyImpulse_0>(impulses)),
        const_cast<std::uint32_t *>(particle_count),
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states),
        const_cast<std::uint32_t *>(contact_flags), body_count,
        const_cast<Vec3_0 *>(as<Vec3_0>(position_corrections))};
    device::dispatch_cpu_serial_groups(grid, reduce_point_body_impulses_Thread, &args);
}

} // namespace parallel_mater::simulation
