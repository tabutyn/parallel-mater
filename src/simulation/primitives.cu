// SPDX-License-Identifier: MIT
#include "primitives.hpp"

#include "parallel_mater_primitives_runtime.cu"

#include <utility>

namespace parallel_mater::simulation {

namespace {

template <typename T>
T *spatial_range(std::byte *base, SpatialBufferRange range) noexcept {
    return reinterpret_cast<T *>(base + range.offset);
}

} // namespace

PmResult compact_flagged_indices(
    device::ExecutionQueue stream, const std::uint8_t *flags, std::uint32_t count,
    std::uint32_t *output, std::uint32_t *output_count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U)
        return device::from_driver_error(cudaMemsetAsync(output_count, 0, sizeof(*output_count), static_cast<cudaStream_t>(stream.native_handle)));
    const std::uint32_t blocks = primitive_block_count(count);
    if (flags == nullptr || output == nullptr || output_count == nullptr ||
        scratch == nullptr || scratch_bytes < compaction_scratch_bytes(count))
        return device::from_driver_error(cudaErrorInvalidValue);
    auto *counts = static_cast<std::uint32_t *>(scratch);
    auto *offsets = counts + blocks;
    pm_compact_count_blocks<<<blocks, primitive_tile_size, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<std::uint8_t *>(flags), count, primitive_tile_size, counts);
    pm_exclusive_scan_u32<<<1U, primitive_tile_size, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        counts, blocks, offsets, output_count);
    pm_compact_scatter_indices<<<blocks, primitive_tile_size, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<std::uint8_t *>(flags), count, primitive_tile_size,
        offsets, output);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult radix_sort_pairs_u64_u32(
    device::ExecutionQueue stream, std::uint64_t *input_keys,
    std::uint64_t *output_keys, std::uint32_t *input_values,
    std::uint32_t *output_values, std::uint32_t count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U) return device::from_driver_error(cudaSuccess);
    const std::uint32_t blocks = primitive_block_count(count);
    if (input_keys == nullptr || output_keys == nullptr ||
        input_values == nullptr || output_values == nullptr ||
        scratch == nullptr || scratch_bytes < radix_sort_scratch_bytes(count))
        return device::from_driver_error(cudaErrorInvalidValue);
    auto *histograms = static_cast<std::uint32_t *>(scratch);
    auto *offsets = histograms + 16U * blocks;
    auto *source_keys = reinterpret_cast<unsigned long long *>(input_keys);
    auto *target_keys = reinterpret_cast<unsigned long long *>(output_keys);
    auto *source_values = input_values;
    auto *target_values = output_values;
    for (std::uint32_t shift = 0U; shift < 64U; shift += 4U) {
        pm_radix_histogram_u64<<<blocks, primitive_tile_size, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
            source_keys, count, primitive_tile_size, shift, blocks, histograms);
        pm_radix_prefix_u32<<<1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
            histograms, blocks, offsets);
        pm_radix_scatter_u64_u32<<<blocks, primitive_tile_size, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
            source_keys, source_values, count, primitive_tile_size, shift,
            blocks, offsets, target_keys, target_values);
        auto *swap_keys = source_keys;
        source_keys = target_keys;
        target_keys = swap_keys;
        auto *swap_values = source_values;
        source_values = target_values;
        target_values = swap_values;
    }
    // Sixteen passes leave the final result in the original buffers. The
    // scheduler expects the sorted result in the alternate buffers, matching
    // its previous double-buffer contract.
    cudaError_t error = cudaMemcpyAsync(output_keys, input_keys,
        static_cast<std::size_t>(count) * sizeof(*output_keys),
        cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream.native_handle));
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(output_values, input_values,
            static_cast<std::size_t>(count) * sizeof(*output_values),
            cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream.native_handle));
    return device::from_driver_error(error == cudaSuccess ? cudaPeekAtLastError() : error);
}

PmResult initialize_spatial_epoch_device(
    device::ExecutionQueue stream, const SpatialBuildPlan &plan,
    const SpatialDeviceBuffers &buffers,
    const SpatialDeviceEpoch &epoch) noexcept {
    if (buffers.header == nullptr || buffers.query_counter == nullptr ||
        buffers.query_status == nullptr || buffers.build_scratch == nullptr ||
        epoch.leaf_count > plan.capacity)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *status = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.status);
    pm_spatial_initialize_epoch<<<
        1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(buffers.header),
        epoch.tick, epoch.graph_revision, epoch.resource_revision,
        epoch.profile_version, epoch.generation, epoch.leaf_count,
        epoch.update_kind, buffers.query_counter,
        buffers.query_status, status);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult build_spatial_index_device(
    device::ExecutionQueue stream, const SpatialBuildPlan &plan,
    const SpatialDeviceBuffers &buffers, const std::uint32_t *sort_keys,
    std::uint32_t expected_active_count) noexcept {
    if (expected_active_count > plan.capacity ||
        (plan.capacity != 0U && (buffers.metadata == nullptr ||
            buffers.bounds == nullptr || buffers.build_flags == nullptr ||
            buffers.build_output == nullptr ||
            buffers.build_scratch == nullptr ||
            (expected_active_count != 0U && sort_keys == nullptr))))
        return PM_ERROR_INVALID_ARGUMENT;
    if (plan.capacity == 0U) return PM_SUCCESS;

    auto *counts = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.compaction_counts);
    auto *active = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.active_indices);
    auto *active_count = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.active_count);
    auto *keys0 = spatial_range<std::uint64_t>(
        buffers.build_scratch, plan.keys[0]);
    auto *keys1 = spatial_range<std::uint64_t>(
        buffers.build_scratch, plan.keys[1]);
    auto *values0 = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.values[0]);
    auto *values1 = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.values[1]);
    auto *histograms = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.radix_histograms);
    auto *radix_offsets = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.radix_offsets);
    auto *block_bounds = spatial_range<SpatialAabb>(
        buffers.build_scratch, plan.block_bounds);
    auto *scene_bounds = spatial_range<SpatialAabb>(
        buffers.build_scratch, plan.scene_bounds);
    auto *arrivals = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.arrivals);
    auto *status = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.status);
    auto *left = spatial_range<std::uint32_t>(
        buffers.build_output, plan.left_children);
    auto *right = spatial_range<std::uint32_t>(
        buffers.build_output, plan.right_children);
    auto *parents = spatial_range<std::uint32_t>(
        buffers.build_output, plan.parents);
    auto *leaf_sources = spatial_range<std::uint32_t>(
        buffers.build_output, plan.leaf_sources);
    auto *node_bounds = spatial_range<SpatialAabb>(
        buffers.build_output, plan.node_bounds);
    const auto capacity_blocks = primitive_block_count(plan.capacity);
    const auto active_blocks = primitive_block_count(expected_active_count);
    constexpr auto tile = primitive_tile_size;

    pm_spatial_stage_leaves<<<
        capacity_blocks, tile, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        reinterpret_cast<PmSpatialLeafMetadata_0 *>(buffers.metadata),
        reinterpret_cast<PmSpatialAabb_0 *>(buffers.bounds),
        expected_active_count, plan.capacity, buffers.build_flags,
        buffers.admission, status);
    auto result = compact_flagged_indices(
        stream, buffers.build_flags, plan.capacity, active, active_count,
        counts, plan.compaction_offsets.offset + plan.compaction_offsets.size -
            plan.compaction_counts.offset);
    if (result != PM_SUCCESS || expected_active_count == 0U) return result;
    if (plan.arrivals.size != 0U) {
        result = stream.services->clear(arrivals, 0, plan.arrivals.size);
        if (result != PM_SUCCESS) return result;
    }
    pm_spatial_reduce_bounds_blocks<<<
        active_blocks, tile, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        active, active_count, plan.capacity,
        reinterpret_cast<PmSpatialAabb_0 *>(buffers.bounds),
        reinterpret_cast<PmSpatialAabb_0 *>(block_bounds), status);
    pm_spatial_reduce_bounds_final<<<
        1U, tile, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        reinterpret_cast<PmSpatialAabb_0 *>(block_bounds), active_blocks,
        active_count, plan.capacity,
        reinterpret_cast<PmSpatialAabb_0 *>(scene_bounds), status);
    pm_spatial_emit_keys<<<
        active_blocks, tile, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        active, active_count, plan.capacity,
        const_cast<std::uint32_t *>(sort_keys),
        reinterpret_cast<PmSpatialAabb_0 *>(buffers.bounds),
        reinterpret_cast<PmSpatialAabb_0 *>(scene_bounds),
        reinterpret_cast<unsigned long long *>(keys0), values0, status);
    if (expected_active_count > 1U &&
        expected_active_count <= spatial_small_sort_capacity) {
        pm_spatial_sort_small_u64_u32<<<
            1U, tile, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
            reinterpret_cast<unsigned long long *>(keys0), values0,
            active_count, plan.capacity, status);
    } else if (expected_active_count > spatial_small_sort_capacity) {
        auto *source_keys = keys0;
        auto *target_keys = keys1;
        auto *source_values = values0;
        auto *target_values = values1;
        for (std::uint32_t shift = 0U; shift < 64U; shift += 8U) {
            pm_spatial_radix_histogram_u64<<<
                active_blocks, tile, 0U,
                static_cast<cudaStream_t>(stream.native_handle)>>>(
                reinterpret_cast<unsigned long long *>(source_keys),
                active_count, plan.capacity, shift, active_blocks,
                histograms, status);
            pm_spatial_radix_prefix_u32<<<
                1U, tile, 0U,
                static_cast<cudaStream_t>(stream.native_handle)>>>(
                histograms, active_blocks, radix_offsets);
            pm_spatial_radix_scatter_u64_u32<<<
                active_blocks, tile, 0U,
                static_cast<cudaStream_t>(stream.native_handle)>>>(
                reinterpret_cast<unsigned long long *>(source_keys),
                source_values, active_count, plan.capacity, shift,
                active_blocks, radix_offsets,
                reinterpret_cast<unsigned long long *>(target_keys),
                target_values, status);
            std::swap(source_keys, target_keys);
            std::swap(source_values, target_values);
        }
    }
    pm_spatial_build_topology<<<
        active_blocks, tile, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        reinterpret_cast<unsigned long long *>(keys0), values0,
        active_count, plan.capacity,
        reinterpret_cast<PmSpatialAabb_0 *>(buffers.bounds), left, right,
        parents, leaf_sources,
        reinterpret_cast<PmSpatialAabb_0 *>(node_bounds), status);
    if (expected_active_count > 1U)
        pm_spatial_union_bounds<<<
            active_blocks, tile, 0U,
            static_cast<cudaStream_t>(stream.native_handle)>>>(
            active_count, plan.capacity, left, right, parents,
            reinterpret_cast<PmSpatialAabb_0 *>(node_bounds), arrivals,
            status);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult publish_spatial_epoch_device(
    device::ExecutionQueue stream, const SpatialBuildPlan &plan,
    const SpatialDeviceBuffers &buffers) noexcept {
    if (buffers.header == nullptr || buffers.build_scratch == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *active_count = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.active_count);
    auto *status = spatial_range<std::uint32_t>(
        buffers.build_scratch, plan.status);
    pm_spatial_publish_epoch<<<
        1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(buffers.header),
        active_count, plan.capacity, status);
    return device::from_driver_error(cudaPeekAtLastError());
}

PmResult query_spatial_epoch_device(
    device::ExecutionQueue stream, const SpatialBuildPlan &plan,
    const SpatialDeviceBuffers &buffers, const SpatialAabb *queries,
    const std::uint32_t *query_masks,
    const std::uint32_t *excluded_sort_keys, std::uint32_t query_count,
    std::uint32_t candidates_per_query) noexcept {
    if (query_count == 0U) return PM_SUCCESS;
    if (buffers.header == nullptr || buffers.metadata == nullptr ||
        buffers.admission == nullptr || buffers.query_counts == nullptr ||
        buffers.query_candidates == nullptr ||
        buffers.query_visited_nodes == nullptr ||
        buffers.query_counter == nullptr || buffers.query_status == nullptr ||
        buffers.build_output == nullptr || queries == nullptr ||
        query_masks == nullptr || excluded_sort_keys == nullptr ||
        candidates_per_query == 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *left = spatial_range<std::uint32_t>(
        buffers.build_output, plan.left_children);
    auto *right = spatial_range<std::uint32_t>(
        buffers.build_output, plan.right_children);
    auto *leaf_sources = spatial_range<std::uint32_t>(
        buffers.build_output, plan.leaf_sources);
    auto *node_bounds = spatial_range<SpatialAabb>(
        buffers.build_output, plan.node_bounds);
    pm_spatial_query_aabbs<<<
        (query_count + 63U) / 64U, 64U, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(buffers.header),
        buffers.query_counter,
        reinterpret_cast<PmSpatialAabb_0 *>(
            const_cast<SpatialAabb *>(queries)),
        const_cast<std::uint32_t *>(query_masks),
        const_cast<std::uint32_t *>(excluded_sort_keys), query_count,
        candidates_per_query, plan.capacity, left, right, leaf_sources,
        reinterpret_cast<PmSpatialAabb_0 *>(node_bounds),
        reinterpret_cast<PmSpatialLeafMetadata_0 *>(buffers.metadata),
        buffers.admission, buffers.query_counts, buffers.query_candidates,
        buffers.query_visited_nodes, buffers.query_status);
    return device::from_driver_error(cudaPeekAtLastError());
}

} // namespace parallel_mater::simulation
