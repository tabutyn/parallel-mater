// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"
#include "spatial_build.hpp"

#include <cstddef>
#include <cstdint>

namespace parallel_mater::simulation {

struct SpatialEpochHeader;
struct SpatialLeafMetadata;

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

// Native shared-spatial dispatch view.  The storage owner retains allocation
// and epoch lifecycle; these helpers only encode the fixed device work.
struct SpatialDeviceBuffers {
    SpatialEpochHeader *header{};
    SpatialLeafMetadata *metadata{};
    SpatialAabb *bounds{};
    std::uint8_t *build_flags{};
    std::uint8_t *admission{};
    std::uint32_t *query_counts{};
    std::uint32_t *query_candidates{};
    std::uint32_t *query_visited_nodes{};
    std::uint32_t *query_counter{};
    std::uint32_t *query_status{};
    std::byte *build_output{};
    std::byte *build_scratch{};
};

struct SpatialDeviceEpoch {
    std::uint64_t tick{};
    std::uint64_t graph_revision{};
    std::uint64_t resource_revision{};
    std::uint32_t profile_version{};
    std::uint32_t generation{};
    std::uint32_t leaf_count{};
    std::uint32_t update_kind{};
};

PmResult initialize_spatial_epoch_device(
    device::ExecutionQueue stream, const SpatialBuildPlan &plan,
    const SpatialDeviceBuffers &buffers,
    const SpatialDeviceEpoch &epoch) noexcept;

PmResult build_spatial_index_device(
    device::ExecutionQueue stream, const SpatialBuildPlan &plan,
    const SpatialDeviceBuffers &buffers, const std::uint32_t *sort_keys,
    std::uint32_t expected_active_count) noexcept;

PmResult publish_spatial_epoch_device(
    device::ExecutionQueue stream, const SpatialBuildPlan &plan,
    const SpatialDeviceBuffers &buffers) noexcept;

PmResult query_spatial_epoch_device(
    device::ExecutionQueue stream, const SpatialBuildPlan &plan,
    const SpatialDeviceBuffers &buffers, const SpatialAabb *queries,
    const std::uint32_t *query_masks,
    const std::uint32_t *excluded_sort_keys, std::uint32_t query_count,
    std::uint32_t candidates_per_query) noexcept;

} // namespace parallel_mater::simulation
