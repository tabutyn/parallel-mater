// SPDX-License-Identifier: MIT
#include "geometry_dispatch.hpp"
#include "geometry_storage.hpp"
#include "rigid_spatial.hpp"
#include "../device/cpu_dispatch.hpp"

#include "parallel_mater_geometry_runtime.cpp"

#include <type_traits>

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

static_assert(std::is_trivially_copyable_v<BodyParameters>);
static_assert(sizeof(BodyParameters) == sizeof(BodyParameters_0));
static_assert(alignof(BodyParameters) == alignof(BodyParameters_0));
static_assert(sizeof(BodyAccumulator) == sizeof(BodyAccumulator_0));
static_assert(sizeof(KinematicTarget) == sizeof(KinematicTarget_0));
static_assert(sizeof(RigidConstraintResource) == sizeof(RigidConstraintResource_0));
static_assert(sizeof(FixedContactProjection) == sizeof(FixedContactProjection_0));
static_assert(sizeof(ContactManifold) == sizeof(ContactManifold_0));
static_assert(sizeof(CachedContactPair) == sizeof(CachedContactPair_0));
static_assert(sizeof(LeafPair) == sizeof(LeafPair_0));
static_assert(sizeof(BvhNode) == sizeof(BvhNode_0));
static_assert(sizeof(WorldAabb) == sizeof(WorldAabb_0));
static_assert(sizeof(CollisionPlane) == sizeof(CollisionPlane_0));
static_assert(sizeof(TriangleMeshResource) == sizeof(TriangleMeshResource_0));
static_assert(sizeof(SpatialAabb) == sizeof(RigidSpatialAabb_0));
static_assert(sizeof(SpatialIdentity) == sizeof(RigidSpatialIdentity_0));
static_assert(sizeof(SpatialLeafMetadata) ==
              sizeof(RigidSpatialLeafMetadata_0));

} // namespace

PmResult integrate_rigid_bodies(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, void *parameters,
    void *accumulators, void *targets, void *input, void *output,
    std::uint32_t count, const float gravity[3], float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) noexcept {
    EntryPointParams_0 args{as<BodyParameters_0>(parameters),
        as<BodyAccumulator_0>(accumulators), as<KinematicTarget_0>(targets),
        as<RigidBodyState_0>(input), as<RigidBodyState_0>(output), count,
        Vec3_0{gravity[0], gravity[1], gravity[2]}, timestep,
        remaining_substeps, apply_impulses};
    device::dispatch_cpu(grid, block, pm_integrate_rigid_bodies_Thread, &args);
    return PM_SUCCESS;
}

PmResult compute_rigid_world_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, void *parameters,
    void *previous_states, void *states, std::uint32_t count,
    void *meshes, void *world_bounds) noexcept {
    EntryPointParams_1 args{as<BodyParameters_0>(parameters),
        as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
        count, as<TriangleMeshResource_0>(meshes),
        as<WorldAabb_0>(world_bounds)};
    device::dispatch_cpu(grid, block, pm_compute_rigid_world_bounds_Thread, &args);
    return PM_SUCCESS;
}

PmResult broad_phase_rigid_pairs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, void *parameters,
    void *world_bounds, void *previous_states, void *states, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t count,
    std::uint8_t *active_flags) noexcept {
    EntryPointParams_2 args{as<BodyParameters_0>(parameters),
        as<WorldAabb_0>(world_bounds), as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(states), as<TriangleMeshResource_0>(meshes),
        as<RigidBodyId_0>(ids), as<RigidConstraintResource_0>(constraints),
        constraint_capacity, as<FixedContactProjection_0>(fixed_groups),
        count, active_flags};
    device::dispatch_cpu(grid, block, pm_broad_phase_rigid_pairs_Thread, &args);
    return PM_SUCCESS;
}

PmResult emit_rigid_spatial_proxies(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue, void *parameters, void *previous_states,
    void *predicted_states, void *meshes, std::uint32_t mesh_capacity,
    void *ids, std::uint32_t body_count, std::uint32_t leaf_offset,
    std::uint32_t leaf_capacity, std::uint32_t sort_key_base,
    std::uint32_t topology_revision, std::uint32_t query_mask,
    const float path_deviation[3], const float correction_budget[3],
    float epsilon, void *metadata, void *bounds,
    std::uint8_t *build_flags, void *query_bounds,
    std::uint32_t *query_masks, std::uint32_t *excluded_sort_keys,
    std::uint32_t *status) noexcept {
    if (path_deviation == nullptr || correction_budget == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    EntryPointParams_16 args{
        as<BodyParameters_0>(parameters),
        as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(predicted_states),
        as<TriangleMeshResource_0>(meshes), mesh_capacity,
        as<RigidBodyId_0>(ids), body_count, leaf_offset, leaf_capacity,
        sort_key_base, topology_revision, query_mask,
        Vec3_0{path_deviation[0], path_deviation[1], path_deviation[2]},
        Vec3_0{correction_budget[0], correction_budget[1],
               correction_budget[2]},
        epsilon, as<RigidSpatialLeafMetadata_0>(metadata),
        as<RigidSpatialAabb_0>(bounds), build_flags,
        as<RigidSpatialAabb_0>(query_bounds), query_masks,
        excluded_sort_keys, status};
    device::dispatch_cpu(
        grid, block, pm_emit_rigid_spatial_proxies_Thread, &args);
    return PM_SUCCESS;
}

PmResult filter_rigid_spatial_candidates(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue, void *metadata,
    std::uint32_t leaf_capacity, std::uint32_t rigid_leaf_offset,
    std::uint32_t *candidate_counts, std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_body,
    std::uint32_t candidate_slot_capacity, void *parameters, void *ids,
    void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t body_count,
    std::uint32_t *pair_codes, std::uint8_t *pair_flags,
    std::uint32_t *status) noexcept {
    EntryPointParams_17 args{
        as<RigidSpatialLeafMetadata_0>(metadata), leaf_capacity,
        rigid_leaf_offset, candidate_counts, candidate_sources,
        candidates_per_body, candidate_slot_capacity,
        as<BodyParameters_0>(parameters), as<RigidBodyId_0>(ids),
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        as<FixedContactProjection_0>(fixed_groups), body_count, pair_codes,
        pair_flags, status};
    device::dispatch_cpu(
        grid, block, pm_filter_rigid_spatial_candidates_Thread, &args);
    return PM_SUCCESS;
}

PmResult gather_rigid_spatial_pairs(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue, std::uint32_t *compacted_slots,
    std::uint32_t *active_pair_count,
    std::uint32_t *candidate_pair_codes,
    std::uint32_t candidate_slot_capacity, std::uint32_t *active_pairs,
    std::uint32_t *status) noexcept {
    EntryPointParams_18 args{compacted_slots, active_pair_count,
        candidate_pair_codes, candidate_slot_capacity, active_pairs, status};
    device::dispatch_cpu(
        grid, block, pm_gather_rigid_spatial_pairs_Thread, &args);
    return PM_SUCCESS;
}

PmResult generate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize, std::size_t, device::ExecutionQueue, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    std::uint32_t mesh_capacity, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, void *leaf_pairs,
    std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity,
    std::uint32_t leaf_pairs_per_slot,
    std::uint32_t shared_leaf_capacity) noexcept {
    EntryPointParams_3 args{as<BodyParameters_0>(parameters),
        as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
        count, as<TriangleMeshResource_0>(meshes), mesh_capacity,
        active_pairs, active_pair_count, as<LeafPair_0>(leaf_pairs),
        leaf_pair_counts, leaf_pair_cache_slot_capacity, leaf_pairs_per_slot,
        shared_leaf_capacity};
    device::dispatch_cpu_serial_groups(
        grid, pm_generate_rigid_leaf_pairs_Thread, &args);
    return PM_SUCCESS;
}

PmResult evaluate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize, std::size_t, device::ExecutionQueue, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *leaf_pairs, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    std::uint32_t blocks_per_pair, float timestep,
    bool canonicalize_swept_witness, void *manifolds) noexcept {
    EntryPointParams_4 args{as<BodyParameters_0>(parameters),
        as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
        count, as<TriangleMeshResource_0>(meshes), as<RigidBodyId_0>(ids),
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        active_pairs, active_pair_count, as<LeafPair_0>(leaf_pairs),
        leaf_pair_counts, leaf_pairs_per_slot,
        as<ContactManifold_0>(leaf_manifolds), blocks_per_pair, timestep,
        canonicalize_swept_witness, as<ContactManifold_0>(manifolds)};
    device::dispatch_cpu_serial_groups(
        grid, pm_evaluate_rigid_leaf_pairs_Thread, &args);
    return PM_SUCCESS;
}

PmResult reduce_rigid_leaf_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, void *parameters,
    std::uint32_t count, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    bool canonicalize_swept_witness, void *manifolds) noexcept {
    EntryPointParams_5 args{as<BodyParameters_0>(parameters), count,
        active_pairs, active_pair_count, leaf_pair_counts, leaf_pairs_per_slot,
        as<ContactManifold_0>(leaf_manifolds), canonicalize_swept_witness,
        as<ContactManifold_0>(manifolds)};
    device::dispatch_cpu(
        grid, block, pm_reduce_rigid_leaf_manifolds_Thread, &args);
    return PM_SUCCESS;
}

PmResult finalize_rigid_contact_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep, void *manifolds,
    bool prepare_faces) noexcept {
    if (prepare_faces) {
        EntryPointParams_15 args{as<BodyParameters_0>(parameters),
            as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
            count, as<TriangleMeshResource_0>(meshes), as<RigidBodyId_0>(ids),
            as<RigidConstraintResource_0>(constraints), constraint_capacity,
            active_pairs, active_pair_count, leaf_pair_counts, timestep,
            as<ContactManifold_0>(manifolds), prepare_faces};
        device::dispatch_cpu(
            grid, block, pm_prepare_rigid_face_manifolds_Thread, &args);
        return PM_SUCCESS;
    }
    EntryPointParams_6 args{as<BodyParameters_0>(parameters),
        as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
        count, as<TriangleMeshResource_0>(meshes), as<RigidBodyId_0>(ids),
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        active_pairs, active_pair_count, leaf_pair_counts, timestep,
        as<ContactManifold_0>(manifolds), prepare_faces};
    device::dispatch_cpu(
        grid, block, pm_finalize_rigid_contact_manifolds_Thread, &args);
    return PM_SUCCESS;
}

PmResult clear_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, void *accumulators,
    void *targets, std::uint32_t count) noexcept {
    EntryPointParams_7 args{as<BodyAccumulator_0>(accumulators),
        as<KinematicTarget_0>(targets), count};
    device::dispatch_cpu(grid, block, pm_clear_rigid_inputs_Thread, &args);
    return PM_SUCCESS;
}

PmResult capture_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, const void *accumulators,
    void *forces, void *torques, std::uint32_t count) noexcept {
    EntryPointParams_8 args{const_cast<BodyAccumulator_0 *>(
        as<BodyAccumulator_0>(accumulators)), as<Vec3_0>(forces),
        as<Vec3_0>(torques), count};
    device::dispatch_cpu(grid, block, pm_capture_rigid_inputs_Thread, &args);
    return PM_SUCCESS;
}

PmResult prepare_parallel_contact_events(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, std::uint32_t body_count,
    void *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_ids,
    std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) noexcept {
    EntryPointParams_9 args{body_count, as<ContactManifold_0>(manifolds),
        const_cast<std::uint32_t *>(active_pairs),
        const_cast<std::uint32_t *>(active_pair_count),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)),
        event_offsets, as<RigidContactEvent_0>(events), event_capacity,
        event_count, collect_events, reset_events};
    device::dispatch_cpu(
        grid, block, pm_prepare_parallel_contact_events_Thread, &args);
    return PM_SUCCESS;
}

PmResult load_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, const void *cache,
    const std::uint32_t *slots, std::uint32_t capacity,
    std::uint64_t epoch, float timestep) noexcept {
    EntryPointParams_10 args{as<ContactManifold_0>(manifolds),
        const_cast<std::uint32_t *>(active_pairs),
        const_cast<std::uint32_t *>(active_pair_count),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        const_cast<CachedContactPair_0 *>(as<CachedContactPair_0>(cache)),
        const_cast<std::uint32_t *>(slots), capacity, epoch, timestep};
    device::dispatch_cpu(grid, block, pm_load_rigid_contact_cache_Thread, &args);
    return PM_SUCCESS;
}

PmResult save_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, const void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, void *cache,
    std::uint32_t *slots, std::uint32_t capacity, std::uint64_t epoch,
    float timestep, const std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity) noexcept {
    EntryPointParams_11 args{
        const_cast<ContactManifold_0 *>(as<ContactManifold_0>(manifolds)),
        const_cast<std::uint32_t *>(active_pairs),
        const_cast<std::uint32_t *>(active_pair_count),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        as<CachedContactPair_0>(cache), slots, capacity, epoch, timestep,
        const_cast<std::uint32_t *>(event_offsets),
        as<RigidContactEvent_0>(events), event_capacity};
    device::dispatch_cpu(grid, block, pm_save_rigid_contact_cache_Thread, &args);
    return PM_SUCCESS;
}

PmResult gather_rigid_debug_samples(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, const void *body_ids,
    const void *body_states, const void *forces, const void *torques,
    std::uint32_t count, void *samples) noexcept {
    EntryPointParams_12 args{
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<Vec3_0 *>(as<Vec3_0>(forces)),
        const_cast<Vec3_0 *>(as<Vec3_0>(torques)), count,
        as<PhysicsDebugRigidSample_0>(samples)};
    device::dispatch_cpu(grid, block, pm_gather_rigid_debug_samples_Thread, &args);
    return PM_SUCCESS;
}

PmResult clamp_rigid_speeds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, const void *body_parameters,
    void *body_states, std::uint32_t count) noexcept {
    EntryPointParams_13 args{
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states), count};
    device::dispatch_cpu(grid, block, pm_clamp_rigid_speeds_Thread, &args);
    return PM_SUCCESS;
}

PmResult build_fixed_collision_groups(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue, const void *constraints,
    std::uint32_t constraint_capacity, const void *body_ids,
    std::uint32_t body_count, void *groups) noexcept {
    EntryPointParams_14 args{const_cast<RigidConstraintResource_0 *>(
        as<RigidConstraintResource_0>(constraints)), constraint_capacity,
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        as<FixedContactProjection_0>(groups)};
    device::dispatch_cpu(
        grid, block, pm_build_fixed_collision_groups_Thread, &args);
    return PM_SUCCESS;
}

} // namespace parallel_mater::simulation
