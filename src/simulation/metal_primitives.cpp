// SPDX-License-Identifier: MIT
#include "primitives.hpp"

#include "driver/metal/schedule_context.hpp"

namespace parallel_mater::simulation {

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
    auto *counts = static_cast<std::uint32_t *>(scratch);
    auto *offsets = counts + blocks;
    if (const auto result = metal::launch(stream, "pm_compact_count_blocks", blocks,
        const_cast<std::uint8_t *>(flags), count, primitive_tile_size, counts); result != PM_SUCCESS) return result;
    if (const auto result = metal::launch(stream, "pm_exclusive_scan_u32", 1U,
        counts, blocks, offsets, output_count); result != PM_SUCCESS) return result;
    if (const auto result = metal::launch(stream, "pm_compact_scatter_indices", blocks,
        const_cast<std::uint8_t *>(flags), count, primitive_tile_size,
        offsets, output); result != PM_SUCCESS) return result;
    return stream.services->last_error(false);
}

PmResult radix_sort_pairs_u64_u32(
    device::ExecutionQueue stream, std::uint64_t *input_keys,
    std::uint64_t *output_keys, std::uint32_t *input_values,
    std::uint32_t *output_values, std::uint32_t count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U) return PM_SUCCESS;
    const std::uint32_t blocks = primitive_block_count(count);
    if (input_keys == nullptr || output_keys == nullptr ||
        input_values == nullptr || output_values == nullptr ||
        scratch == nullptr || scratch_bytes < radix_sort_scratch_bytes(count))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *histograms = static_cast<std::uint32_t *>(scratch);
    auto *offsets = histograms + 16U * blocks;
    auto *source_keys = reinterpret_cast<unsigned long long *>(input_keys);
    auto *target_keys = reinterpret_cast<unsigned long long *>(output_keys);
    auto *source_values = input_values;
    auto *target_values = output_values;
    for (std::uint32_t shift = 0U; shift < 64U; shift += 4U) {
        if (const auto result = metal::launch(stream, "pm_radix_histogram_u64", blocks,
            source_keys, count, primitive_tile_size, shift, blocks, histograms); result != PM_SUCCESS) return result;
        if (const auto result = metal::launch(stream, "pm_radix_prefix_u32", 1U,
            histograms, blocks, offsets); result != PM_SUCCESS) return result;
        if (const auto result = metal::launch(stream, "pm_radix_scatter_u64_u32", blocks,
            source_keys, source_values, count, primitive_tile_size, shift,
            blocks, offsets, target_keys, target_values); result != PM_SUCCESS) return result;
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
    PmResult error = stream.services->copy(output_keys, input_keys,
        static_cast<std::size_t>(count) * sizeof(*output_keys),
        device::CopyDirection::device_to_device);
    if (error == PM_SUCCESS)
        error = stream.services->copy(output_values, input_values,
            static_cast<std::size_t>(count) * sizeof(*output_values),
            device::CopyDirection::device_to_device);
    return error == PM_SUCCESS ? stream.services->last_error(false) : error;
}

} // namespace parallel_mater::simulation
