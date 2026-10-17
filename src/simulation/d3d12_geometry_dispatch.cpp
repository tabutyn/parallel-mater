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
PmResult launch(device::ExecutionQueue stream, std::string_view module,
                   std::string_view entry, device::DispatchSize grid, device::DispatchSize block,
                   const Constants<Size> &constants) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    return execution == nullptr
        ? PM_ERROR_INVALID_ARGUMENT
        : execution->dispatch(module, entry, grid, block, constants.bytes);
}

} // namespace

PmResult integrate_rigid_bodies(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *accumulators, void *targets, void *input, void *output,
    std::uint32_t count, const float gravity[3], float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr || gravity == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
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

PmResult compute_rigid_world_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count,
    void *meshes, void *world_bounds) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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

PmResult broad_phase_rigid_pairs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *world_bounds, void *previous_states, void *states, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t count,
    std::uint8_t *active_flags) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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
    values.buffer(128U, *execution, active_flags);
    return launch(stream, "geometry", "pm_broad_phase_rigid_pairs",
                  grid, block, values);
}

PmResult generate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize block, std::size_t, device::ExecutionQueue stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, std::uint32_t mesh_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *leaf_pairs, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity,
    std::uint32_t leaf_pairs_per_slot,
    std::uint32_t shared_leaf_capacity) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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

PmResult evaluate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize block, std::size_t, device::ExecutionQueue stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, void *ids, void *constraints,
    std::uint32_t constraint_capacity, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, void *leaf_pairs,
    std::uint32_t *leaf_pair_counts, std::uint32_t leaf_pairs_per_slot,
    void *leaf_manifolds, std::uint32_t blocks_per_pair, float timestep,
    void *manifolds) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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

PmResult reduce_rigid_leaf_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    std::uint32_t count, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    void *manifolds) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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

PmResult finalize_rigid_contact_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep, void *manifolds,
    bool prepare_faces) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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
    const char *entry = prepare_faces ? "pm_prepare_rigid_face_manifolds"
                                      : "pm_finalize_rigid_contact_manifolds";
    return launch(stream, "geometry", entry,
                  grid, block, values);
}

PmResult clear_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *accumulators,
    void *targets, std::uint32_t count) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    Constants<32> values;
    values.buffer(0U, *execution, accumulators);
    values.buffer(16U, *execution, targets);
    values.set(28U, count);
    return launch(stream, "geometry", "pm_clear_rigid_inputs",
                  grid, block, values);
}

PmResult capture_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *accumulators,
    void *forces, void *torques, std::uint32_t count) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    Constants<48> values;
    values.buffer(0U, *execution, accumulators);
    values.buffer(16U, *execution, forces);
    values.buffer(32U, *execution, torques);
    values.set(44U, count);
    return launch(stream, "geometry", "pm_capture_rigid_inputs",
                  grid, block, values);
}

PmResult prepare_parallel_contact_events(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, std::uint32_t body_count,
    void *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_ids,
    std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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

PmResult load_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, const void *cache,
    const std::uint32_t *slots, std::uint32_t capacity,
    std::uint64_t epoch, float timestep) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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

PmResult save_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, void *cache,
    std::uint32_t *slots, std::uint32_t capacity, std::uint64_t epoch,
    float timestep, const std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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

PmResult gather_rigid_debug_samples(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_ids,
    const void *body_states, const void *forces, const void *torques,
    std::uint32_t count, void *samples) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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

PmResult clamp_rigid_speeds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    void *body_states, std::uint32_t count) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    Constants<32> values;
    values.buffer(0U, *execution, body_parameters);
    values.buffer(16U, *execution, body_states);
    values.set(28U, count);
    return launch(stream, "geometry", "pm_clamp_rigid_speeds",
                  grid, block, values);
}

PmResult build_fixed_collision_groups(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *constraints,
    std::uint32_t constraint_capacity, const void *body_ids,
    std::uint32_t body_count, void *groups) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    Constants<48> values;
    values.buffer(0U, *execution, constraints);
    values.set(12U, constraint_capacity);
    values.buffer(16U, *execution, body_ids);
    values.set(28U, body_count);
    values.buffer(32U, *execution, groups);
    return launch(stream, "geometry", "pm_build_fixed_collision_groups",
                  grid, block, values);
}

PmResult geometry_constraints_occupancy(
    int *resident_blocks, std::uint32_t) noexcept {
    if (resident_blocks == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *resident_blocks = 1;
    return PM_SUCCESS;
}

PmResult prepare_geometry_constraints(
    device::ExecutionQueue stream, const void *parameters,
    const RigidBodyState *previous, RigidBodyState *predicted,
    std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    RigidBodyId *ids, GeometryConstraintBody *bodies, void *schedule,
    float timestep, Vec3 gravity, std::uint32_t base_iterations,
    std::uint32_t requested_iterations,
    std::uint32_t grid_blocks) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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
                  device::DispatchSize(grid_blocks), device::DispatchSize(128U), values);
}

PmResult solve_geometry_constraints(
    device::ExecutionQueue stream, const void *parameters,
    const RigidBodyState *previous, RigidBodyState *states,
    std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t grid_blocks) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
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
                  device::DispatchSize(grid_blocks), device::DispatchSize(32U), values);
}

PmResult compact_flagged_indices(
    device::ExecutionQueue stream, const std::uint8_t *flags, std::uint32_t count,
    std::uint32_t *output, std::uint32_t *output_count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U)
        return stream.services->clear(output_count, 0, sizeof(*output_count));
    const std::uint32_t blocks = primitive_block_count(count);
    if (flags == nullptr || output == nullptr || output_count == nullptr ||
        scratch == nullptr || scratch_bytes < compaction_scratch_bytes(count))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    auto *counts = static_cast<std::uint32_t *>(scratch);
    auto *offsets = counts + blocks;
    const auto packed_flags = execution->address(flags);
    Constants<48> count_values;
    count_values.set(0U, packed_flags);
    count_values.set(12U, count);
    count_values.set(16U, primitive_tile_size);
    count_values.buffer(32U, *execution, counts);
    PmResult result = launch(
        stream, "primitives", "pm_compact_count_blocks",
        device::DispatchSize(blocks), device::DispatchSize(primitive_tile_size), count_values);
    if (result != PM_SUCCESS) return result;
    Constants<48> scan_values;
    scan_values.buffer(0U, *execution, counts);
    scan_values.set(12U, blocks);
    scan_values.buffer(16U, *execution, offsets);
    scan_values.buffer(32U, *execution, output_count);
    result = launch(stream, "primitives", "pm_exclusive_scan_u32",
                    device::DispatchSize(1U), device::DispatchSize(1U), scan_values);
    if (result != PM_SUCCESS) return result;
    Constants<64> scatter_values;
    scatter_values.set(0U, packed_flags);
    scatter_values.set(12U, count);
    scatter_values.set(16U, primitive_tile_size);
    scatter_values.buffer(32U, *execution, offsets);
    scatter_values.buffer(48U, *execution, output);
    return launch(stream, "primitives", "pm_compact_scatter_indices",
                  device::DispatchSize(blocks), device::DispatchSize(primitive_tile_size), scatter_values);
}

PmResult radix_sort_pairs_u64_u32(
    device::ExecutionQueue stream, std::uint64_t *input_keys,
    std::uint64_t *output_keys, std::uint32_t *input_values,
    std::uint32_t *output_values, std::uint32_t count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U) return PM_SUCCESS;
    if (input_keys == nullptr || output_keys == nullptr ||
        input_values == nullptr || output_values == nullptr ||
        scratch == nullptr || scratch_bytes < radix_sort_scratch_bytes(count))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    const auto blocks = primitive_block_count(count);
    auto *histograms = static_cast<std::uint32_t *>(scratch);
    auto *offsets = histograms + 16U * blocks;
    auto *source_keys = input_keys;
    auto *target_keys = output_keys;
    auto *source_values = input_values;
    auto *target_values = output_values;
    for (std::uint32_t shift = 0U; shift < 64U; shift += 4U) {
        Constants<44> histogram;
        histogram.buffer(0U, *execution, source_keys);
        histogram.set(12U, count);
        histogram.set(16U, primitive_tile_size);
        histogram.set(20U, shift);
        histogram.set(24U, blocks);
        histogram.buffer(32U, *execution, histograms);
        auto result = launch(stream, "primitives", "pm_radix_histogram_u64",
                             device::DispatchSize(blocks), device::DispatchSize(primitive_tile_size), histogram);
        if (result != PM_SUCCESS) return result;
        Constants<28> prefix;
        prefix.buffer(0U, *execution, histograms);
        prefix.set(12U, blocks);
        prefix.buffer(16U, *execution, offsets);
        result = launch(stream, "primitives", "pm_radix_prefix_u32",
                        device::DispatchSize(1U), device::DispatchSize(1U), prefix);
        if (result != PM_SUCCESS) return result;
        Constants<92> scatter;
        scatter.buffer(0U, *execution, source_keys);
        scatter.buffer(16U, *execution, source_values);
        scatter.set(28U, count);
        scatter.set(32U, primitive_tile_size);
        scatter.set(36U, shift);
        scatter.set(40U, blocks);
        scatter.buffer(48U, *execution, offsets);
        scatter.buffer(64U, *execution, target_keys);
        scatter.buffer(80U, *execution, target_values);
        result = launch(stream, "primitives", "pm_radix_scatter_u64_u32",
                        device::DispatchSize(blocks), device::DispatchSize(primitive_tile_size), scatter);
        if (result != PM_SUCCESS) return result;
        std::swap(source_keys, target_keys);
        std::swap(source_values, target_values);
    }
    auto result = execution->copy_device(output_keys, input_keys,
        static_cast<std::size_t>(count) * sizeof(*input_keys));
    if (result == PM_SUCCESS)
        result = execution->copy_device(output_values, input_values,
            static_cast<std::size_t>(count) * sizeof(*input_values));
    return result;
}

} // namespace parallel_mater::simulation
