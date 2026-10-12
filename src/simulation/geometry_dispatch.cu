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

cudaError_t integrate_rigid_bodies(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *accumulators, void *targets, void *input, void *output,
    std::uint32_t count, const float gravity[3], float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) noexcept {
    const Vec3_0 gravity_value{gravity[0], gravity[1], gravity[2]};
    pm_integrate_rigid_bodies<<<grid, block, 0U, stream>>>(
        as<BodyParameters_0>(parameters), as<BodyAccumulator_0>(accumulators),
        as<KinematicTarget_0>(targets), as<RigidBodyState_0>(input),
        as<RigidBodyState_0>(output), count, gravity_value, timestep,
        remaining_substeps, apply_impulses);
    return cudaPeekAtLastError();
}

cudaError_t compute_rigid_world_bounds(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count,
    void *meshes, void *world_bounds) noexcept {
    pm_compute_rigid_world_bounds<<<grid, block, 0U, stream>>>(
        as<BodyParameters_0>(parameters), as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(states), count, as<TriangleMeshResource_0>(meshes),
        as<WorldAabb_0>(world_bounds));
    return cudaPeekAtLastError();
}

cudaError_t broad_phase_rigid_pairs(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *world_bounds, void *previous_states, void *states, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t count, std::uint8_t *active_flags) noexcept {
    pm_broad_phase_rigid_pairs<<<grid, block, 0U, stream>>>(
        as<BodyParameters_0>(parameters), as<WorldAabb_0>(world_bounds),
        as<RigidBodyState_0>(previous_states), as<RigidBodyState_0>(states),
        as<TriangleMeshResource_0>(meshes), as<RigidBodyId_0>(ids),
        as<RigidConstraintResource_0>(constraints), constraint_capacity,
        as<FixedContactProjection_0>(fixed_groups), count, active_flags);
    return cudaPeekAtLastError();
}

cudaError_t generate_rigid_leaf_pairs(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, std::uint32_t mesh_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *leaf_pairs, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity,
    std::uint32_t leaf_pairs_per_slot,
    std::uint32_t shared_leaf_capacity) noexcept {
    pm_generate_rigid_leaf_pairs<<<grid, block, shared_bytes, stream>>>(
        as<BodyParameters_0>(parameters), as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(states), count, as<TriangleMeshResource_0>(meshes),
        mesh_capacity, active_pairs, active_pair_count, as<LeafPair_0>(leaf_pairs),
        leaf_pair_counts, leaf_pair_cache_slot_capacity, leaf_pairs_per_slot,
        shared_leaf_capacity);
    return cudaPeekAtLastError();
}

cudaError_t evaluate_rigid_leaf_pairs(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, void *ids, void *constraints,
    std::uint32_t constraint_capacity, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, void *leaf_pairs,
    std::uint32_t *leaf_pair_counts, std::uint32_t leaf_pairs_per_slot,
    void *leaf_manifolds, std::uint32_t blocks_per_pair, float timestep,
    void *manifolds) noexcept {
    pm_evaluate_rigid_leaf_pairs<<<grid, block, shared_bytes, stream>>>(
        as<BodyParameters_0>(parameters), as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(states), count, as<TriangleMeshResource_0>(meshes),
        as<RigidBodyId_0>(ids), as<RigidConstraintResource_0>(constraints),
        constraint_capacity, active_pairs, active_pair_count,
        as<LeafPair_0>(leaf_pairs), leaf_pair_counts, leaf_pairs_per_slot,
        as<ContactManifold_0>(leaf_manifolds), blocks_per_pair, timestep,
        as<ContactManifold_0>(manifolds));
    return cudaPeekAtLastError();
}

cudaError_t reduce_rigid_leaf_manifolds(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    std::uint32_t count, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    void *manifolds) noexcept {
    pm_reduce_rigid_leaf_manifolds<<<grid, block, 0U, stream>>>(
        as<BodyParameters_0>(parameters), count, active_pairs,
        active_pair_count, leaf_pair_counts, leaf_pairs_per_slot,
        as<ContactManifold_0>(leaf_manifolds), as<ContactManifold_0>(manifolds));
    return cudaPeekAtLastError();
}

cudaError_t finalize_rigid_contact_manifolds(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep, void *manifolds,
    bool prepare_faces) noexcept {
    pm_finalize_rigid_contact_manifolds<<<grid, block, 0U, stream>>>(
        as<BodyParameters_0>(parameters), as<RigidBodyState_0>(previous_states),
        as<RigidBodyState_0>(states), count, as<TriangleMeshResource_0>(meshes),
        as<RigidBodyId_0>(ids), as<RigidConstraintResource_0>(constraints),
        constraint_capacity, active_pairs, active_pair_count, leaf_pair_counts,
        timestep, as<ContactManifold_0>(manifolds), prepare_faces);
    return cudaPeekAtLastError();
}

cudaError_t clear_rigid_inputs(
    dim3 grid, dim3 block, cudaStream_t stream, void *accumulators,
    void *targets, std::uint32_t count) noexcept {
    pm_clear_rigid_inputs<<<grid, block, 0U, stream>>>(
        as<BodyAccumulator_0>(accumulators), as<KinematicTarget_0>(targets),
        count);
    return cudaPeekAtLastError();
}

cudaError_t capture_rigid_inputs(
    dim3 grid, dim3 block, cudaStream_t stream, const void *accumulators,
    void *forces, void *torques, std::uint32_t count) noexcept {
    pm_capture_rigid_inputs<<<grid, block, 0U, stream>>>(
        const_cast<BodyAccumulator_0 *>(
            as<BodyAccumulator_0>(accumulators)),
        as<Vec3_0>(forces), as<Vec3_0>(torques), count);
    return cudaPeekAtLastError();
}

cudaError_t prepare_parallel_contact_events(
    dim3 grid, dim3 block, cudaStream_t stream, std::uint32_t body_count,
    void *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_ids,
    std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) noexcept {
    pm_prepare_parallel_contact_events<<<grid, block, 0U, stream>>>(
        body_count, as<ContactManifold_0>(manifolds),
        const_cast<std::uint32_t *>(active_pairs),
        const_cast<std::uint32_t *>(active_pair_count),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)),
        event_offsets, as<RigidContactEvent_0>(events), event_capacity,
        event_count, collect_events, reset_events);
    return cudaPeekAtLastError();
}

cudaError_t load_rigid_contact_cache(
    dim3 grid, dim3 block, cudaStream_t stream, void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, const void *cache,
    const std::uint32_t *slots, std::uint32_t capacity,
    std::uint64_t epoch, float timestep) noexcept {
    pm_load_rigid_contact_cache<<<grid, block, 0U, stream>>>(
        as<ContactManifold_0>(manifolds),
        const_cast<std::uint32_t *>(active_pairs),
        const_cast<std::uint32_t *>(active_pair_count),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        const_cast<CachedContactPair_0 *>(as<CachedContactPair_0>(cache)),
        const_cast<std::uint32_t *>(slots), capacity,
        static_cast<unsigned long long>(epoch), timestep);
    return cudaPeekAtLastError();
}

cudaError_t save_rigid_contact_cache(
    dim3 grid, dim3 block, cudaStream_t stream, const void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, void *cache,
    std::uint32_t *slots, std::uint32_t capacity, std::uint64_t epoch,
    float timestep, const std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity) noexcept {
    pm_save_rigid_contact_cache<<<grid, block, 0U, stream>>>(
        const_cast<ContactManifold_0 *>(as<ContactManifold_0>(manifolds)),
        const_cast<std::uint32_t *>(active_pairs),
        const_cast<std::uint32_t *>(active_pair_count),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        as<CachedContactPair_0>(cache), slots, capacity,
        static_cast<unsigned long long>(epoch), timestep,
        const_cast<std::uint32_t *>(event_offsets),
        as<RigidContactEvent_0>(events), event_capacity);
    return cudaPeekAtLastError();
}

cudaError_t gather_rigid_debug_samples(
    dim3 grid, dim3 block, cudaStream_t stream, const void *body_ids,
    const void *body_states, const void *forces, const void *torques,
    std::uint32_t count, void *samples) noexcept {
    pm_gather_rigid_debug_samples<<<grid, block, 0U, stream>>>(
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)),
        const_cast<RigidBodyState_0 *>(as<RigidBodyState_0>(body_states)),
        const_cast<Vec3_0 *>(as<Vec3_0>(forces)),
        const_cast<Vec3_0 *>(as<Vec3_0>(torques)), count,
        as<PhysicsDebugRigidSample_0>(samples));
    return cudaPeekAtLastError();
}

cudaError_t clamp_rigid_speeds(
    dim3 grid, dim3 block, cudaStream_t stream, const void *body_parameters,
    void *body_states, std::uint32_t count) noexcept {
    pm_clamp_rigid_speeds<<<grid, block, 0U, stream>>>(
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<RigidBodyState_0>(body_states), count);
    return cudaPeekAtLastError();
}

cudaError_t build_fixed_collision_groups(
    dim3 grid, dim3 block, cudaStream_t stream, const void *constraints,
    std::uint32_t constraint_capacity, const void *body_ids,
    std::uint32_t body_count, void *groups) noexcept {
    pm_build_fixed_collision_groups<<<grid, block, 0U, stream>>>(
        const_cast<RigidConstraintResource_0 *>(
            as<RigidConstraintResource_0>(constraints)),
        constraint_capacity,
        const_cast<RigidBodyId_0 *>(as<RigidBodyId_0>(body_ids)), body_count,
        as<FixedContactProjection_0>(groups));
    return cudaPeekAtLastError();
}

} // namespace parallel_mater::simulation
