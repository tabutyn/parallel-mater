// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstddef>
#include <cstdint>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t primitive_tile_size = 256U;

[[nodiscard]] constexpr std::uint32_t primitive_block_count(
    std::uint32_t count) noexcept {
    return count / primitive_tile_size + (count % primitive_tile_size != 0U);
}

[[nodiscard]] constexpr std::size_t compaction_scratch_bytes(
    std::uint32_t capacity) noexcept {
    return 2U * primitive_block_count(capacity) * sizeof(std::uint32_t);
}

[[nodiscard]] constexpr std::size_t radix_sort_scratch_bytes(
    std::uint32_t capacity) noexcept {
    return 2U * 16U * primitive_block_count(capacity) * sizeof(std::uint32_t);
}

PmResult compact_flagged_indices(
    device::ExecutionQueue stream, const std::uint8_t *flags, std::uint32_t count,
    std::uint32_t *output, std::uint32_t *output_count,
    void *scratch, std::size_t scratch_bytes) noexcept;

PmResult radix_sort_pairs_u64_u32(
    device::ExecutionQueue stream, std::uint64_t *input_keys,
    std::uint64_t *output_keys, std::uint32_t *input_values,
    std::uint32_t *output_values, std::uint32_t count,
    void *scratch, std::size_t scratch_bytes) noexcept;

} // namespace parallel_mater::simulation
