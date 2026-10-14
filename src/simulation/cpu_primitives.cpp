// SPDX-License-Identifier: MIT
#include "primitives.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace parallel_mater::simulation {

cudaError_t compact_flagged_indices(
    cudaStream_t, const std::uint8_t *flags, std::uint32_t count,
    std::uint32_t *output, std::uint32_t *output_count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (output_count == nullptr || (count != 0U &&
        (flags == nullptr || output == nullptr || scratch == nullptr)) ||
        scratch_bytes < compaction_scratch_bytes(count))
        return cudaErrorInvalidValue;
    std::uint32_t written = 0U;
    for (std::uint32_t index = 0U; index < count; ++index)
        if (flags[index] != 0U) output[written++] = index;
    *output_count = written;
    return cudaSuccess;
}

cudaError_t radix_sort_pairs_u64_u32(
    cudaStream_t, std::uint64_t *input_keys, std::uint64_t *output_keys,
    std::uint32_t *input_values, std::uint32_t *output_values,
    std::uint32_t count, void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U) return cudaSuccess;
    if (input_keys == nullptr || output_keys == nullptr ||
        input_values == nullptr || output_values == nullptr ||
        scratch == nullptr || scratch_bytes < radix_sort_scratch_bytes(count))
        return cudaErrorInvalidValue;
    try {
        std::vector<std::uint32_t> order(count);
        std::iota(order.begin(), order.end(), 0U);
        std::stable_sort(order.begin(), order.end(),
            [&](std::uint32_t left, std::uint32_t right) {
                return input_keys[left] < input_keys[right];
            });
        for (std::uint32_t index = 0U; index < count; ++index) {
            output_keys[index] = input_keys[order[index]];
            output_values[index] = input_values[order[index]];
        }
        return cudaSuccess;
    } catch (...) {
        return cudaErrorMemoryAllocation;
    }
}

} // namespace parallel_mater::simulation
