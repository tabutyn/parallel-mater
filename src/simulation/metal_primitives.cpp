// SPDX-License-Identifier: MIT
#include "primitives.hpp"

#include "driver/metal/schedule_context.hpp"

namespace parallel_mater::simulation {

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
    auto *counts = static_cast<std::uint32_t *>(scratch);
    auto *offsets = counts + blocks;
    if (const auto result = metal::launch(stream, "pm_compact_count_blocks", blocks,
        const_cast<std::uint8_t *>(flags), count, primitive_tile_size, counts); result != cudaSuccess) return result;
    if (const auto result = metal::launch(stream, "pm_exclusive_scan_u32", 1U,
        counts, blocks, offsets, output_count); result != cudaSuccess) return result;
    if (const auto result = metal::launch(stream, "pm_compact_scatter_indices", blocks,
        const_cast<std::uint8_t *>(flags), count, primitive_tile_size,
        offsets, output); result != cudaSuccess) return result;
    return cudaPeekAtLastError();
}

cudaError_t radix_sort_pairs_u64_u32(
    cudaStream_t stream, std::uint64_t *input_keys,
    std::uint64_t *output_keys, std::uint32_t *input_values,
    std::uint32_t *output_values, std::uint32_t count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U) return cudaSuccess;
    const std::uint32_t blocks = primitive_block_count(count);
    if (input_keys == nullptr || output_keys == nullptr ||
        input_values == nullptr || output_values == nullptr ||
        scratch == nullptr || scratch_bytes < radix_sort_scratch_bytes(count))
        return cudaErrorInvalidValue;
    auto *histograms = static_cast<std::uint32_t *>(scratch);
    auto *offsets = histograms + 16U * blocks;
    auto *source_keys = reinterpret_cast<unsigned long long *>(input_keys);
    auto *target_keys = reinterpret_cast<unsigned long long *>(output_keys);
    auto *source_values = input_values;
    auto *target_values = output_values;
    for (std::uint32_t shift = 0U; shift < 64U; shift += 4U) {
        if (const auto result = metal::launch(stream, "pm_radix_histogram_u64", blocks,
            source_keys, count, primitive_tile_size, shift, blocks, histograms); result != cudaSuccess) return result;
        if (const auto result = metal::launch(stream, "pm_radix_prefix_u32", 1U,
            histograms, blocks, offsets); result != cudaSuccess) return result;
        if (const auto result = metal::launch(stream, "pm_radix_scatter_u64_u32", blocks,
            source_keys, source_values, count, primitive_tile_size, shift,
            blocks, offsets, target_keys, target_values); result != cudaSuccess) return result;
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
        cudaMemcpyDeviceToDevice, stream);
    if (error == cudaSuccess)
        error = cudaMemcpyAsync(output_values, input_values,
            static_cast<std::size_t>(count) * sizeof(*output_values),
            cudaMemcpyDeviceToDevice, stream);
    return error == cudaSuccess ? cudaPeekAtLastError() : error;
}

} // namespace parallel_mater::simulation
