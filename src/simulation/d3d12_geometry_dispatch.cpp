// SPDX-License-Identifier: MIT
#include "geometry_dispatch.hpp"
#include "geometry_constraints_dispatch.hpp"
#include "geometry_storage.hpp"
#include "primitives.hpp"

#include "driver/d3d12/schedule_context.hpp"

#include <array>
#include <cstring>
#include <type_traits>

namespace parallel_mater::simulation {
namespace {

template <std::size_t Size> struct Constants {
    std::array<std::byte, Size> bytes{};

    template <typename T>
    void set(std::size_t offset, const T &value) noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        if (offset + sizeof(T) <= bytes.size())
            std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }

    void boolean(std::size_t offset, bool value) noexcept {
        set(offset, static_cast<std::uint32_t>(value));
    }

    void buffer(std::size_t offset, d3d12::ScheduleContext &execution,
                const void *pointer) noexcept {
        set(offset, execution.address(pointer));
    }
};

template <std::size_t Size>
cudaError_t launch(cudaStream_t stream, std::string_view module,
                   std::string_view entry, dim3 grid, dim3 block,
                   const Constants<Size> &constants) noexcept {
    auto *execution = d3d12::context(stream);
    return execution == nullptr
        ? cudaErrorInvalidResourceHandle
        : execution->dispatch(module, entry, grid, block, constants.bytes);
}

} // namespace

cudaError_t integrate_rigid_bodies(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *accumulators, void *targets, void *input, void *output,
    std::uint32_t count, const float gravity[3], float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr || gravity == nullptr)
        return cudaErrorInvalidValue;
    Constants<104> values;
    values.buffer(0U, *execution, parameters);
    values.buffer(16U, *execution, accumulators);
    values.buffer(32U, *execution, targets);
    values.buffer(48U, *execution, input);
    values.buffer(64U, *execution, output);
    values.set(76U, count);
    values.set(80U, gravity[0]);
    values.set(84U, gravity[1]);
    values.set(88U, gravity[2]);
    values.set(92U, timestep);
    values.set(96U, remaining_substeps);
    values.boolean(100U, apply_impulses);
    return launch(stream, "geometry", "pm_integrate_rigid_bodies",
                  grid, block, values);
}

cudaError_t compute_rigid_world_bounds(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count,
    void *meshes, void *world_bounds) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<80> values;
    values.buffer(0U, *execution, parameters);
    values.buffer(16U, *execution, previous_states);
    values.buffer(32U, *execution, states);
    values.set(44U, count);
    values.set(48U, execution->translated_meshes(
                        meshes, parameters, count));
    values.buffer(64U, *execution, world_bounds);
    return launch(stream, "geometry", "pm_compute_rigid_world_bounds",
                  grid, block, values);
}

cudaError_t broad_phase_rigid_pairs(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *world_bounds, void *previous_states, void *states, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t count,
    std::uint8_t *active_flags) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<144> values;
    values.buffer(0U, *execution, parameters);
    values.buffer(16U, *execution, world_bounds);
    values.buffer(32U, *execution, previous_states);
    values.buffer(48U, *execution, states);
    values.set(64U, execution->translated_meshes(
                        meshes, parameters, count));
    values.buffer(80U, *execution, ids);
    values.buffer(96U, *execution, constraints);
    values.set(108U, constraint_capacity);
    values.buffer(112U, *execution, fixed_groups);
    values.set(124U, count);
    values.set(128U, execution->widened_flags(
                         active_flags, count * count));
    return launch(stream, "geometry", "pm_broad_phase_rigid_pairs",
                  grid, block, values);
}

cudaError_t generate_rigid_leaf_pairs(
    dim3 grid, dim3 block, std::size_t, cudaStream_t stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, std::uint32_t mesh_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *leaf_pairs, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity,
    std::uint32_t leaf_pairs_per_slot,
    std::uint32_t shared_leaf_capacity) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<144> values;
    values.buffer(0U, *execution, parameters);
    values.buffer(16U, *execution, previous_states);
    values.buffer(32U, *execution, states);
    values.set(44U, count);
    values.set(48U, execution->translated_meshes(
                        meshes, parameters, count, mesh_capacity));
    values.set(60U, mesh_capacity);
    values.buffer(64U, *execution, active_pairs);
    values.buffer(80U, *execution, active_pair_count);
    values.buffer(96U, *execution, leaf_pairs);
    values.buffer(112U, *execution, leaf_pair_counts);
    values.set(124U, leaf_pair_cache_slot_capacity);
    values.set(128U, leaf_pairs_per_slot);
    values.set(132U, shared_leaf_capacity);
    return launch(stream, "geometry", "pm_generate_rigid_leaf_pairs",
                  grid, block, values);
}

cudaError_t evaluate_rigid_leaf_pairs(
    dim3 grid, dim3 block, std::size_t, cudaStream_t stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, void *ids, void *constraints,
    std::uint32_t constraint_capacity, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, void *leaf_pairs,
    std::uint32_t *leaf_pair_counts, std::uint32_t leaf_pairs_per_slot,
    void *leaf_manifolds, std::uint32_t blocks_per_pair, float timestep,
    void *manifolds) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<208> values;
    values.buffer(0U, *execution, parameters);
    values.buffer(16U, *execution, previous_states);
    values.buffer(32U, *execution, states);
    values.set(44U, count);
    values.set(48U, execution->translated_meshes(
                        meshes, parameters, count));
    values.buffer(64U, *execution, ids);
    values.buffer(80U, *execution, constraints);
    values.set(92U, constraint_capacity);
    values.buffer(96U, *execution, active_pairs);
    values.buffer(112U, *execution, active_pair_count);
    values.buffer(128U, *execution, leaf_pairs);
    values.buffer(144U, *execution, leaf_pair_counts);
    values.set(156U, leaf_pairs_per_slot);
    values.buffer(160U, *execution, leaf_manifolds);
    values.set(172U, blocks_per_pair);
    values.set(176U, timestep);
    values.buffer(192U, *execution, manifolds);
    return launch(stream, "geometry", "pm_evaluate_rigid_leaf_pairs",
                  grid, block, values);
}

cudaError_t reduce_rigid_leaf_manifolds(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    std::uint32_t count, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    void *manifolds) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<96> values;
    values.buffer(0U, *execution, parameters);
    values.set(12U, count);
    values.buffer(16U, *execution, active_pairs);
    values.buffer(32U, *execution, active_pair_count);
    values.buffer(48U, *execution, leaf_pair_counts);
    values.set(60U, leaf_pairs_per_slot);
    values.buffer(64U, *execution, leaf_manifolds);
    values.buffer(80U, *execution, manifolds);
    return launch(stream, "geometry", "pm_reduce_rigid_leaf_manifolds",
                  grid, block, values);
}

cudaError_t finalize_rigid_contact_manifolds(
    dim3 grid, dim3 block, cudaStream_t stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep, void *manifolds,
    bool prepare_faces) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<160> values;
    values.buffer(0U, *execution, parameters);
    values.buffer(16U, *execution, previous_states);
    values.buffer(32U, *execution, states);
    values.set(44U, count);
    values.set(48U, execution->translated_meshes(
                        meshes, parameters, count));
    values.buffer(64U, *execution, ids);
    values.buffer(80U, *execution, constraints);
    values.set(92U, constraint_capacity);
    values.buffer(96U, *execution, active_pairs);
    values.buffer(112U, *execution, active_pair_count);
    values.buffer(128U, *execution, leaf_pair_counts);
    values.set(140U, timestep);
    values.buffer(144U, *execution, manifolds);
    values.boolean(156U, prepare_faces);
    return launch(stream, "geometry", "pm_finalize_rigid_contact_manifolds",
                  grid, block, values);
}

cudaError_t clear_rigid_inputs(
    dim3 grid, dim3 block, cudaStream_t stream, void *accumulators,
    void *targets, std::uint32_t count) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<32> values;
    values.buffer(0U, *execution, accumulators);
    values.buffer(16U, *execution, targets);
    values.set(28U, count);
    return launch(stream, "geometry", "pm_clear_rigid_inputs",
                  grid, block, values);
}

cudaError_t capture_rigid_inputs(
    dim3 grid, dim3 block, cudaStream_t stream, const void *accumulators,
    void *forces, void *torques, std::uint32_t count) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<48> values;
    values.buffer(0U, *execution, accumulators);
    values.buffer(16U, *execution, forces);
    values.buffer(32U, *execution, torques);
    values.set(44U, count);
    return launch(stream, "geometry", "pm_capture_rigid_inputs",
                  grid, block, values);
}

cudaError_t prepare_parallel_contact_events(
    dim3 grid, dim3 block, cudaStream_t stream, std::uint32_t body_count,
    void *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_ids,
    std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<144> values;
    values.set(0U, body_count);
    values.buffer(16U, *execution, manifolds);
    values.buffer(32U, *execution, active_pairs);
    values.buffer(48U, *execution, active_pair_count);
    values.buffer(64U, *execution, body_ids);
    values.buffer(80U, *execution, event_offsets);
    values.buffer(96U, *execution, events);
    values.set(108U, event_capacity);
    values.buffer(112U, *execution, event_count);
    values.boolean(124U, collect_events);
    values.boolean(128U, reset_events);
    return launch(stream, "geometry", "pm_prepare_parallel_contact_events",
                  grid, block, values);
}

cudaError_t load_rigid_contact_cache(
    dim3 grid, dim3 block, cudaStream_t stream, void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, const void *cache,
    const std::uint32_t *slots, std::uint32_t capacity,
    std::uint64_t epoch, float timestep) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<128> values;
    values.buffer(0U, *execution, manifolds);
    values.buffer(16U, *execution, active_pairs);
    values.buffer(32U, *execution, active_pair_count);
    values.buffer(48U, *execution, body_states);
    values.buffer(64U, *execution, body_ids);
    values.set(76U, body_count);
    values.buffer(80U, *execution, cache);
    values.buffer(96U, *execution, slots);
    values.set(108U, capacity);
    values.set(112U, epoch);
    values.set(120U, timestep);
    return launch(stream, "geometry", "pm_load_rigid_contact_cache",
                  grid, block, values);
}

cudaError_t save_rigid_contact_cache(
    dim3 grid, dim3 block, cudaStream_t stream, const void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, void *cache,
    std::uint32_t *slots, std::uint32_t capacity, std::uint64_t epoch,
    float timestep, const std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<160> values;
    values.buffer(0U, *execution, manifolds);
    values.buffer(16U, *execution, active_pairs);
    values.buffer(32U, *execution, active_pair_count);
    values.buffer(48U, *execution, body_states);
    values.buffer(64U, *execution, body_ids);
    values.set(76U, body_count);
    values.buffer(80U, *execution, cache);
    values.buffer(96U, *execution, slots);
    values.set(108U, capacity);
    values.set(112U, epoch);
    values.set(120U, timestep);
    values.buffer(128U, *execution, event_offsets);
    values.buffer(144U, *execution, events);
    values.set(156U, event_capacity);
    return launch(stream, "geometry", "pm_save_rigid_contact_cache",
                  grid, block, values);
}

cudaError_t gather_rigid_debug_samples(
    dim3 grid, dim3 block, cudaStream_t stream, const void *body_ids,
    const void *body_states, const void *forces, const void *torques,
    std::uint32_t count, void *samples) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<80> values;
    values.buffer(0U, *execution, body_ids);
    values.buffer(16U, *execution, body_states);
    values.buffer(32U, *execution, forces);
    values.buffer(48U, *execution, torques);
    values.set(60U, count);
    values.buffer(64U, *execution, samples);
    return launch(stream, "geometry", "pm_gather_rigid_debug_samples",
                  grid, block, values);
}

cudaError_t clamp_rigid_speeds(
    dim3 grid, dim3 block, cudaStream_t stream, const void *body_parameters,
    void *body_states, std::uint32_t count) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<32> values;
    values.buffer(0U, *execution, body_parameters);
    values.buffer(16U, *execution, body_states);
    values.set(28U, count);
    return launch(stream, "geometry", "pm_clamp_rigid_speeds",
                  grid, block, values);
}

cudaError_t build_fixed_collision_groups(
    dim3 grid, dim3 block, cudaStream_t stream, const void *constraints,
    std::uint32_t constraint_capacity, const void *body_ids,
    std::uint32_t body_count, void *groups) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<48> values;
    values.buffer(0U, *execution, constraints);
    values.set(12U, constraint_capacity);
    values.buffer(16U, *execution, body_ids);
    values.set(28U, body_count);
    values.buffer(32U, *execution, groups);
    return launch(stream, "geometry", "pm_build_fixed_collision_groups",
                  grid, block, values);
}

cudaError_t geometry_constraints_occupancy(
    int *resident_blocks, std::uint32_t) noexcept {
    if (resident_blocks == nullptr) return cudaErrorInvalidValue;
    *resident_blocks = 1;
    return cudaSuccess;
}

cudaError_t prepare_geometry_constraints(
    cudaStream_t stream, const void *parameters,
    const RigidBodyState *previous, RigidBodyState *predicted,
    std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    RigidBodyId *ids, GeometryConstraintBody *bodies, void *schedule,
    float timestep, Vec3 gravity, std::uint32_t base_iterations,
    std::uint32_t requested_iterations,
    std::uint32_t grid_blocks) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<192> values;
    values.buffer(0U, *execution, parameters);
    values.buffer(16U, *execution, previous);
    values.buffer(32U, *execution, predicted);
    values.set(44U, body_count);
    values.buffer(48U, *execution, manifolds);
    values.buffer(64U, *execution, active_pairs);
    values.buffer(80U, *execution, active_pair_count);
    values.buffer(96U, *execution, constraints);
    values.set(108U, constraint_capacity);
    values.buffer(112U, *execution, ids);
    values.buffer(128U, *execution, bodies);
    values.buffer(144U, *execution, schedule);
    values.set(156U, timestep);
    values.set(160U, gravity);
    values.set(172U, base_iterations);
    values.set(176U, requested_iterations);
    values.set(180U, grid_blocks);
    return launch(stream, "geometry_constraints", "prepare_avbd_kernel",
                  dim3(grid_blocks), dim3(128U), values);
}

cudaError_t solve_geometry_constraints(
    cudaStream_t stream, const void *parameters,
    const RigidBodyState *previous, RigidBodyState *states,
    std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t grid_blocks) noexcept {
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    Constants<144> values;
    values.buffer(0U, *execution, parameters);
    values.buffer(16U, *execution, previous);
    values.buffer(32U, *execution, states);
    values.set(44U, body_count);
    values.buffer(48U, *execution, manifolds);
    values.buffer(64U, *execution, active_pairs);
    values.buffer(80U, *execution, active_pair_count);
    values.buffer(96U, *execution, constraints);
    values.set(108U, constraint_capacity);
    values.buffer(112U, *execution, bodies);
    values.buffer(128U, *execution, schedule);
    values.set(140U, timestep);
    return launch(stream, "geometry_constraints", "solve_avbd_kernel",
                  dim3(grid_blocks), dim3(32U), values);
}

cudaError_t compact_flagged_indices(
    cudaStream_t stream, const std::uint8_t *flags, std::uint32_t count,
    std::uint32_t *output, std::uint32_t *output_count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U)
        return cudaMemsetAsync(output_count, 0, sizeof(*output_count), stream);
    const std::uint32_t blocks = primitive_block_count(count);
    if (flags == nullptr || output == nullptr || output_count == nullptr ||
        scratch == nullptr || scratch_bytes < compaction_scratch_bytes(count))
        return cudaErrorInvalidValue;
    auto *execution = d3d12::context(stream);
    if (execution == nullptr) return cudaErrorInvalidResourceHandle;
    auto *counts = static_cast<std::uint32_t *>(scratch);
    auto *offsets = counts + blocks;
    const auto wide_flags = execution->widened_flags(flags, count);
    Constants<48> count_values;
    count_values.set(0U, wide_flags);
    count_values.set(12U, count);
    count_values.set(16U, primitive_tile_size);
    count_values.buffer(32U, *execution, counts);
    cudaError_t result = launch(
        stream, "primitives", "pm_compact_count_blocks",
        dim3(blocks), dim3(primitive_tile_size), count_values);
    if (result != cudaSuccess) return result;
    Constants<48> scan_values;
    scan_values.buffer(0U, *execution, counts);
    scan_values.set(12U, blocks);
    scan_values.buffer(16U, *execution, offsets);
    scan_values.buffer(32U, *execution, output_count);
    result = launch(stream, "primitives", "pm_exclusive_scan_u32",
                    dim3(1U), dim3(1U), scan_values);
    if (result != cudaSuccess) return result;
    Constants<64> scatter_values;
    scatter_values.set(0U, wide_flags);
    scatter_values.set(12U, count);
    scatter_values.set(16U, primitive_tile_size);
    scatter_values.buffer(32U, *execution, offsets);
    scatter_values.buffer(48U, *execution, output);
    return launch(stream, "primitives", "pm_compact_scatter_indices",
                  dim3(blocks), dim3(primitive_tile_size), scatter_values);
}

cudaError_t radix_sort_pairs_u64_u32(
    cudaStream_t, std::uint64_t *, std::uint64_t *, std::uint32_t *,
    std::uint32_t *, std::uint32_t, void *, std::size_t) noexcept {
    return cudaErrorNotSupported;
}

} // namespace parallel_mater::simulation
