// SPDX-License-Identifier: MIT
#include "geometry_dispatch.hpp"
#include "geometry_storage.hpp"

#include "parallel_mater_geometry_runtime.cu"

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
static_assert(sizeof(RigidConstraintResource) ==
              sizeof(RigidConstraintResource_0));
static_assert(sizeof(FixedContactProjection) ==
              sizeof(FixedContactProjection_0));
static_assert(sizeof(ContactManifold) == sizeof(ContactManifold_0));
static_assert(sizeof(CachedContactPair) == sizeof(CachedContactPair_0));
static_assert(sizeof(LeafPair) == sizeof(LeafPair_0));
static_assert(sizeof(BvhNode) == sizeof(BvhNode_0));
static_assert(sizeof(WorldAabb) == sizeof(WorldAabb_0));
static_assert(sizeof(CollisionPlane) == sizeof(CollisionPlane_0));
static_assert(sizeof(TriangleMeshResource) == sizeof(TriangleMeshResource_0));

} // namespace

PmResult integrate_rigid_bodies(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *accumulators, void *targets, void *input, void *output,
    std::uint32_t count, const float gravity[3], float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) noexcept {
    const Vec3_0 gravity_value{gravity[0], gravity[1], gravity[2]};
    pm_integrate_rigid_bodies<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<BodyParameters_0>(parameters), as<BodyAccumulator_0>(accumulators),
        as<KinematicTarget_0>(targets), as<RigidBodyState_0>(input),
        as<RigidBodyState_0>(output), count, gravity_value, timestep,
        remaining_substeps, apply_impulses);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult compute_rigid_world_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count,
    void *meshes, void *world_bounds) noexcept {
    pm_compute_rigid_world_bounds<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<BodyParameters_0>(parameters), as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(states), count, as<TriangleMeshResource_0>(meshes),
        as<WorldAabb_0>(world_bounds));
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult broad_phase_rigid_pairs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *world_bounds, void *previous_states, void *states, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t count, std::uint8_t *active_flags) noexcept {
    pm_broad_phase_rigid_pairs<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<BodyParameters_0>(parameters), as<WorldAabb_0>(world_bounds),
        as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
        as<TriangleMeshResource_0>(meshes), as<RigidBodyId_0>(ids),
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        as<FixedContactProjection_0>(fixed_groups), count, active_flags);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult emit_rigid_spatial_proxies(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *parameters, void *previous_states,
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
    const Vec3_0 deviation{path_deviation[0], path_deviation[1],
                           path_deviation[2]};
    const Vec3_0 correction{correction_budget[0], correction_budget[1],
                            correction_budget[2]};
    pm_emit_rigid_spatial_proxies<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<BodyParameters_0>(parameters),
        as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(predicted_states),
        as<TriangleMeshResource_0>(meshes), mesh_capacity,
        as<RigidBodyId_0>(ids), body_count, leaf_offset, leaf_capacity,
        sort_key_base, topology_revision, query_mask, deviation, correction,
        epsilon, as<RigidSpatialLeafMetadata_0>(metadata),
        as<RigidSpatialAabb_0>(bounds), build_flags,
        as<RigidSpatialAabb_0>(query_bounds), query_masks,
        excluded_sort_keys, status);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult filter_rigid_spatial_candidates(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *metadata,
    std::uint32_t leaf_capacity, std::uint32_t rigid_leaf_offset,
    std::uint32_t *candidate_counts, std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_body,
    std::uint32_t candidate_slot_capacity, void *parameters, void *ids,
    void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t body_count,
    std::uint32_t *pair_codes, std::uint8_t *pair_flags,
    std::uint32_t *status) noexcept {
    pm_filter_rigid_spatial_candidates<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<RigidSpatialLeafMetadata_0>(metadata), leaf_capacity,
        rigid_leaf_offset, candidate_counts, candidate_sources,
        candidates_per_body, candidate_slot_capacity,
        as<BodyParameters_0>(parameters), as<RigidBodyId_0>(ids),
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        as<FixedContactProjection_0>(fixed_groups), body_count, pair_codes,
        pair_flags, status);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult gather_rigid_spatial_pairs(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, std::uint32_t *compacted_slots,
    std::uint32_t *active_pair_count,
    std::uint32_t *candidate_pair_codes,
    std::uint32_t candidate_slot_capacity, std::uint32_t *active_pairs,
    std::uint32_t *status) noexcept {
    pm_gather_rigid_spatial_pairs<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        compacted_slots, active_pair_count, candidate_pair_codes,
        candidate_slot_capacity, active_pairs, status);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult generate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize block, std::size_t shared_bytes, device::ExecutionQueue stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, std::uint32_t mesh_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *leaf_pairs, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity,
    std::uint32_t leaf_pairs_per_slot,
    std::uint32_t shared_leaf_capacity) noexcept {
    pm_generate_rigid_leaf_pairs<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), shared_bytes, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<BodyParameters_0>(parameters), as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(states), count, as<TriangleMeshResource_0>(meshes),
        mesh_capacity, active_pairs, active_pair_count, as<LeafPair_0>(leaf_pairs),
        leaf_pair_counts, leaf_pair_cache_slot_capacity, leaf_pairs_per_slot,
        shared_leaf_capacity);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult evaluate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize block, std::size_t shared_bytes, device::ExecutionQueue stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, void *ids, void *constraints,
    std::uint32_t constraint_capacity, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, void *leaf_pairs,
    std::uint32_t *leaf_pair_counts, std::uint32_t leaf_pairs_per_slot,
    void *leaf_manifolds, std::uint32_t blocks_per_pair, float timestep,
    bool canonicalize_swept_witness, void *manifolds) noexcept {
    pm_evaluate_rigid_leaf_pairs<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), shared_bytes, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<BodyParameters_0>(parameters), as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(states), count, as<TriangleMeshResource_0>(meshes),
        as<RigidBodyId_0>(ids), as<RigidConstraintResource_0>(constraints),
        constraint_capacity, active_pairs, active_pair_count,
        as<LeafPair_0>(leaf_pairs), leaf_pair_counts, leaf_pairs_per_slot,
        as<ContactManifold_0>(leaf_manifolds), blocks_per_pair, timestep,
        canonicalize_swept_witness, as<ContactManifold_0>(manifolds));
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult reduce_rigid_leaf_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    std::uint32_t count, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    bool canonicalize_swept_witness, void *manifolds) noexcept {
    pm_reduce_rigid_leaf_manifolds<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<BodyParameters_0>(parameters), count, active_pairs,
        active_pair_count, leaf_pair_counts, leaf_pairs_per_slot,
        as<ContactManifold_0>(leaf_manifolds), canonicalize_swept_witness,
        as<ContactManifold_0>(manifolds));
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult finalize_rigid_contact_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep, void *manifolds,
    bool prepare_faces) noexcept {
    if (prepare_faces) {
        pm_prepare_rigid_face_manifolds<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
            as<BodyParameters_0>(parameters), as<RigidBodyState_0>(previous_states),
            as<RigidBodyState_0>(states), count, as<TriangleMeshResource_0>(meshes),
            as<RigidBodyId_0>(ids), as<RigidConstraintResource_0>(constraints),
            constraint_capacity, active_pairs, active_pair_count, leaf_pair_counts,
            timestep, as<ContactManifold_0>(manifolds), prepare_faces);
    } else {
        pm_finalize_rigid_contact_manifolds<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
            as<BodyParameters_0>(parameters), as<RigidBodyState_0>(previous_states),
            as<RigidBodyState_0>(states), count, as<TriangleMeshResource_0>(meshes),
            as<RigidBodyId_0>(ids), as<RigidConstraintResource_0>(constraints),
            constraint_capacity, active_pairs, active_pair_count, leaf_pair_counts,
            timestep, as<ContactManifold_0>(manifolds), prepare_faces);
    }
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult clear_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *accumulators,
    void *targets, std::uint32_t count) noexcept {
    pm_clear_rigid_inputs<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<BodyAccumulator_0>(accumulators), as<KinematicTarget_0>(targets),
        count);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult capture_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *accumulators,
    void *forces, void *torques, std::uint32_t count) noexcept {
    pm_capture_rigid_inputs<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<BodyAccumulator_0 *>(
            as<BodyAccumulator_0>(accumulators)),
        as<Vec3_0>(forces), as<Vec3_0>(torques), count);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult prepare_parallel_contact_events(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, std::uint32_t body_count,
    void *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_ids,
    std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) noexcept {
    pm_prepare_parallel_contact_events<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        body_count, as<ContactManifold_0>(manifolds),
        const_cast<std::uint32_t *>(active_pairs),
        const_cast<std::uint32_t *>(active_pair_count),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)),
        event_offsets, as<RigidContactEvent_0>(events), event_capacity,
        event_count, collect_events, reset_events);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult load_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, const void *cache,
    const std::uint32_t *slots, std::uint32_t capacity,
    std::uint64_t epoch, float timestep) noexcept {
    pm_load_rigid_contact_cache<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<ContactManifold_0>(manifolds),
        const_cast<std::uint32_t *>(active_pairs),
        const_cast<std::uint32_t *>(active_pair_count),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        const_cast<CachedContactPair_0 *>(as<CachedContactPair_0>(cache)),
        const_cast<std::uint32_t *>(slots), capacity,
        static_cast<unsigned long long>(epoch), timestep);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult save_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, void *cache,
    std::uint32_t *slots, std::uint32_t capacity, std::uint64_t epoch,
    float timestep, const std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity) noexcept {
    pm_save_rigid_contact_cache<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<ContactManifold_0 *>(as<ContactManifold_0>(manifolds)),
        const_cast<std::uint32_t *>(active_pairs),
        const_cast<std::uint32_t *>(active_pair_count),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        as<CachedContactPair_0>(cache), slots, capacity,
        static_cast<unsigned long long>(epoch), timestep,
        const_cast<std::uint32_t *>(event_offsets),
        as<RigidContactEvent_0>(events), event_capacity);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult gather_rigid_debug_samples(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_ids,
    const void *body_states, const void *forces, const void *torques,
    std::uint32_t count, void *samples) noexcept {
    pm_gather_rigid_debug_samples<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<Vec3_0 *>(as<Vec3_0>(forces)),
        const_cast<Vec3_0 *>(as<Vec3_0>(torques)), count,
        as<PhysicsDebugRigidSample_0>(samples));
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult clamp_rigid_speeds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    void *body_states, std::uint32_t count) noexcept {
    pm_clamp_rigid_speeds<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states), count);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult build_fixed_collision_groups(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *constraints,
    std::uint32_t constraint_capacity, const void *body_ids,
    std::uint32_t body_count, void *groups) noexcept {
    pm_build_fixed_collision_groups<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<RigidConstraintResource_0 *>(
            as<RigidConstraintResource_0>(constraints)),
        constraint_capacity,
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        as<FixedContactProjection_0>(groups));
    return device::from_driver_error(cudaPeekAtLastError());
}

} // namespace parallel_mater::simulation
