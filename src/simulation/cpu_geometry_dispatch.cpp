// SPDX-License-Identifier: MIT
#include "geometry_dispatch.hpp"
#include "geometry_storage.hpp"
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

} // namespace

cudaError_t integrate_rigid_bodies(
    dim3 grid, dim3 block, cudaStream_t, void *parameters,
    void *accumulators, void *targets, void *input, void *output,
    std::uint32_t count, const float gravity[3], float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) noexcept {
    EntryPointParams_0 args{as<BodyParameters_0>(parameters),
        as<BodyAccumulator_0>(accumulators), as<KinematicTarget_0>(targets),
        as<RigidBodyState_0>(input), as<RigidBodyState_0>(output), count,
        Vec3_0{gravity[0], gravity[1], gravity[2]}, timestep,
        remaining_substeps, apply_impulses};
    device::dispatch_cpu(grid, block, pm_integrate_rigid_bodies_Thread, &args);
    return cudaSuccess;
}

cudaError_t compute_rigid_world_bounds(
    dim3 grid, dim3 block, cudaStream_t, void *parameters,
    void *previous_states, void *states, std::uint32_t count,
    void *meshes, void *world_bounds) noexcept {
    EntryPointParams_1 args{as<BodyParameters_0>(parameters),
        as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
        count, as<TriangleMeshResource_0>(meshes),
        as<WorldAabb_0>(world_bounds)};
    device::dispatch_cpu(grid, block, pm_compute_rigid_world_bounds_Thread, &args);
    return cudaSuccess;
}

cudaError_t broad_phase_rigid_pairs(
    dim3 grid, dim3 block, cudaStream_t, void *parameters,
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
    return cudaSuccess;
}

cudaError_t generate_rigid_leaf_pairs(
    dim3 grid, dim3, std::size_t, cudaStream_t, void *parameters,
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
    return cudaSuccess;
}

cudaError_t evaluate_rigid_leaf_pairs(
    dim3 grid, dim3, std::size_t, cudaStream_t, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *leaf_pairs, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    std::uint32_t blocks_per_pair, float timestep, void *manifolds) noexcept {
    EntryPointParams_4 args{as<BodyParameters_0>(parameters),
        as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
        count, as<TriangleMeshResource_0>(meshes), as<RigidBodyId_0>(ids),
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        active_pairs, active_pair_count, as<LeafPair_0>(leaf_pairs),
        leaf_pair_counts, leaf_pairs_per_slot,
        as<ContactManifold_0>(leaf_manifolds), blocks_per_pair, timestep,
        as<ContactManifold_0>(manifolds)};
    device::dispatch_cpu_serial_groups(
        grid, pm_evaluate_rigid_leaf_pairs_Thread, &args);
    return cudaSuccess;
}

cudaError_t reduce_rigid_leaf_manifolds(
    dim3 grid, dim3 block, cudaStream_t, void *parameters,
    std::uint32_t count, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    void *manifolds) noexcept {
    EntryPointParams_5 args{as<BodyParameters_0>(parameters), count,
        active_pairs, active_pair_count, leaf_pair_counts, leaf_pairs_per_slot,
        as<ContactManifold_0>(leaf_manifolds), as<ContactManifold_0>(manifolds)};
    device::dispatch_cpu(
        grid, block, pm_reduce_rigid_leaf_manifolds_Thread, &args);
    return cudaSuccess;
}

cudaError_t finalize_rigid_contact_manifolds(
    dim3 grid, dim3 block, cudaStream_t, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep, void *manifolds,
    bool prepare_faces) noexcept {
    EntryPointParams_6 args{as<BodyParameters_0>(parameters),
        as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
        count, as<TriangleMeshResource_0>(meshes), as<RigidBodyId_0>(ids),
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        active_pairs, active_pair_count, leaf_pair_counts, timestep,
        as<ContactManifold_0>(manifolds), prepare_faces};
    device::dispatch_cpu(
        grid, block, pm_finalize_rigid_contact_manifolds_Thread, &args);
    return cudaSuccess;
}

cudaError_t clear_rigid_inputs(
    dim3 grid, dim3 block, cudaStream_t, void *accumulators,
    void *targets, std::uint32_t count) noexcept {
    EntryPointParams_7 args{as<BodyAccumulator_0>(accumulators),
        as<KinematicTarget_0>(targets), count};
    device::dispatch_cpu(grid, block, pm_clear_rigid_inputs_Thread, &args);
    return cudaSuccess;
}

cudaError_t capture_rigid_inputs(
    dim3 grid, dim3 block, cudaStream_t, const void *accumulators,
    void *forces, void *torques, std::uint32_t count) noexcept {
    EntryPointParams_8 args{const_cast<BodyAccumulator_0 *>(
        as<BodyAccumulator_0>(accumulators)), as<Vec3_0>(forces),
        as<Vec3_0>(torques), count};
    device::dispatch_cpu(grid, block, pm_capture_rigid_inputs_Thread, &args);
    return cudaSuccess;
}

cudaError_t prepare_parallel_contact_events(
    dim3 grid, dim3 block, cudaStream_t, std::uint32_t body_count,
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
    return cudaSuccess;
}

cudaError_t load_rigid_contact_cache(
    dim3 grid, dim3 block, cudaStream_t, void *manifolds,
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
    return cudaSuccess;
}

cudaError_t save_rigid_contact_cache(
    dim3 grid, dim3 block, cudaStream_t, const void *manifolds,
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
    return cudaSuccess;
}

cudaError_t gather_rigid_debug_samples(
    dim3 grid, dim3 block, cudaStream_t, const void *body_ids,
    const void *body_states, const void *forces, const void *torques,
    std::uint32_t count, void *samples) noexcept {
    EntryPointParams_12 args{
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<Vec3_0 *>(as<Vec3_0>(forces)),
        const_cast<Vec3_0 *>(as<Vec3_0>(torques)), count,
        as<PhysicsDebugRigidSample_0>(samples)};
    device::dispatch_cpu(grid, block, pm_gather_rigid_debug_samples_Thread, &args);
    return cudaSuccess;
}

cudaError_t clamp_rigid_speeds(
    dim3 grid, dim3 block, cudaStream_t, const void *body_parameters,
    void *body_states, std::uint32_t count) noexcept {
    EntryPointParams_13 args{
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states), count};
    device::dispatch_cpu(grid, block, pm_clamp_rigid_speeds_Thread, &args);
    return cudaSuccess;
}

cudaError_t build_fixed_collision_groups(
    dim3 grid, dim3 block, cudaStream_t, const void *constraints,
    std::uint32_t constraint_capacity, const void *body_ids,
    std::uint32_t body_count, void *groups) noexcept {
    EntryPointParams_14 args{const_cast<RigidConstraintResource_0 *>(
        as<RigidConstraintResource_0>(constraints)), constraint_capacity,
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        as<FixedContactProjection_0>(groups)};
    device::dispatch_cpu(
        grid, block, pm_build_fixed_collision_groups_Thread, &args);
    return cudaSuccess;
}

} // namespace parallel_mater::simulation
