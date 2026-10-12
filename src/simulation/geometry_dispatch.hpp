// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace parallel_mater::simulation {

cudaError_t integrate_rigid_bodies(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *accumulators, void *targets, void *input, void *output,
    std::uint32_t count, const float gravity[3], float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) noexcept;

cudaError_t compute_rigid_world_bounds(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count,
    void *meshes, void *world_bounds) noexcept;

cudaError_t broad_phase_rigid_pairs(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *world_bounds, void *previous_states, void *states, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t count, std::uint8_t *active_flags) noexcept;

cudaError_t generate_rigid_leaf_pairs(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, std::uint32_t mesh_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *leaf_pairs, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity,
    std::uint32_t leaf_pairs_per_slot,
    std::uint32_t shared_leaf_capacity) noexcept;

cudaError_t evaluate_rigid_leaf_pairs(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, void *ids, void *constraints,
    std::uint32_t constraint_capacity, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, void *leaf_pairs,
    std::uint32_t *leaf_pair_counts, std::uint32_t leaf_pairs_per_slot,
    void *leaf_manifolds, std::uint32_t blocks_per_pair, float timestep,
    void *manifolds) noexcept;

cudaError_t reduce_rigid_leaf_manifolds(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    std::uint32_t count, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    void *manifolds) noexcept;

cudaError_t finalize_rigid_contact_manifolds(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep, void *manifolds,
    bool prepare_faces) noexcept;

cudaError_t clear_rigid_inputs(
    dim3 grid, dim3 block, cudaStream_t stream, void *accumulators,
    void *targets, std::uint32_t count) noexcept;

cudaError_t capture_rigid_inputs(
    dim3 grid, dim3 block, cudaStream_t stream, const void *accumulators,
    void *forces, void *torques, std::uint32_t count) noexcept;

cudaError_t prepare_parallel_contact_events(
    dim3 grid, dim3 block, cudaStream_t stream, std::uint32_t body_count,
    void *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_ids,
    std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) noexcept;

cudaError_t load_rigid_contact_cache(
    dim3 grid, dim3 block, cudaStream_t stream, void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, const void *cache,
    const std::uint32_t *slots, std::uint32_t capacity,
    std::uint64_t epoch, float timestep) noexcept;

cudaError_t save_rigid_contact_cache(
    dim3 grid, dim3 block, cudaStream_t stream, const void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, void *cache,
    std::uint32_t *slots, std::uint32_t capacity, std::uint64_t epoch,
    float timestep, const std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity) noexcept;

cudaError_t gather_rigid_debug_samples(
    dim3 grid, dim3 block, cudaStream_t stream, const void *body_ids,
    const void *body_states, const void *forces, const void *torques,
    std::uint32_t count, void *samples) noexcept;

cudaError_t clamp_rigid_speeds(
    dim3 grid, dim3 block, cudaStream_t stream, const void *body_parameters,
    void *body_states, std::uint32_t count) noexcept;

cudaError_t build_fixed_collision_groups(
    dim3 grid, dim3 block, cudaStream_t stream, const void *constraints,
    std::uint32_t constraint_capacity, const void *body_ids,
    std::uint32_t body_count, void *groups) noexcept;

} // namespace parallel_mater::simulation
