// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t spatial_build_tile_size = 256U;
inline constexpr std::uint32_t spatial_small_sort_capacity = 1024U;
inline constexpr std::uint32_t spatial_radix_size = 256U;
inline constexpr std::uint32_t spatial_radix_passes = 8U;
// Topology range searches use signed 32-bit offsets and may double once past
// the live range, so keep both that sentinel and 2*N-1 node indices positive.
inline constexpr std::uint32_t spatial_max_capacity = 0x3fffffffU;
inline constexpr std::uint32_t spatial_status_invalid_bounds = 1U;
inline constexpr std::uint32_t spatial_status_duplicate_key = 2U;
inline constexpr std::uint32_t spatial_status_count_overflow = 4U;
inline constexpr std::uint32_t spatial_status_policy_mismatch = 8U;

enum class SpatialSortPolicy : std::uint8_t {
    none,
    workgroup_bitonic,
    radix8,
};

struct alignas(16) SpatialAabb {
    float minimum[4]{};
    float maximum[4]{};
};

static_assert(sizeof(SpatialAabb) == 32U);
static_assert(alignof(SpatialAabb) == 16U);

struct SpatialBufferRange {
    std::size_t offset{};
    std::size_t size{};
};

struct SpatialBuildPlan {
    std::uint32_t capacity{};
    std::uint32_t block_capacity{};
    std::uint32_t node_capacity{};
    std::size_t scratch_bytes{};
    std::size_t output_bytes{};

    SpatialBufferRange compaction_counts{};
    SpatialBufferRange compaction_offsets{};
    SpatialBufferRange active_indices{};
    SpatialBufferRange active_count{};
    SpatialBufferRange keys[2]{};
    SpatialBufferRange values[2]{};
    SpatialBufferRange radix_histograms{};
    SpatialBufferRange radix_offsets{};
    SpatialBufferRange block_bounds{};
    SpatialBufferRange scene_bounds{};
    SpatialBufferRange arrivals{};
    SpatialBufferRange status{};

    SpatialBufferRange left_children{};
    SpatialBufferRange right_children{};
    SpatialBufferRange parents{};
    SpatialBufferRange leaf_sources{};
    SpatialBufferRange node_bounds{};
};

struct SpatialBuildDispatchBudget {
    SpatialSortPolicy sort_policy{SpatialSortPolicy::none};
    std::uint32_t active_count{};
    std::uint32_t capacity{};
    std::uint32_t active_blocks{};
    std::uint32_t capacity_blocks{};
    std::uint32_t sort_dispatches{};
    std::uint32_t total_dispatches{};
    std::uint64_t total_workgroups{};
};

namespace spatial_build_detail {

[[nodiscard]] constexpr bool checked_add(
    std::size_t left, std::size_t right, std::size_t &result) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    result = left + right;
    return true;
}

[[nodiscard]] constexpr bool checked_multiply(
    std::size_t left, std::size_t right, std::size_t &result) noexcept {
    if (left != 0U &&
        right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

[[nodiscard]] constexpr bool append_range(
    std::size_t &cursor, std::size_t elements, std::size_t stride,
    SpatialBufferRange &range) noexcept {
    constexpr std::size_t alignment = 16U;
    const std::size_t padding = (alignment - cursor % alignment) % alignment;
    std::size_t offset{};
    std::size_t size{};
    std::size_t end{};
    if (!checked_add(cursor, padding, offset) ||
        !checked_multiply(elements, stride, size) ||
        !checked_add(offset, size, end))
        return false;
    range = {offset, size};
    cursor = end;
    return true;
}

} // namespace spatial_build_detail

[[nodiscard]] inline PmResult make_spatial_build_plan(
    std::uint32_t capacity, SpatialBuildPlan &plan) noexcept {
    plan = {};
    if (capacity > spatial_max_capacity) return PM_ERROR_CAPACITY_EXCEEDED;
    plan.capacity = capacity;
    if (capacity == 0U) return PM_SUCCESS;
    plan.block_capacity = capacity / spatial_build_tile_size +
        (capacity % spatial_build_tile_size != 0U);
    plan.node_capacity = capacity * 2U - 1U;
    const std::size_t internal_capacity = capacity - 1U;
    const std::size_t block_capacity = plan.block_capacity;
    std::size_t radix_slots{};
    if (!spatial_build_detail::checked_multiply(
            spatial_radix_size, block_capacity, radix_slots))
        return PM_ERROR_CAPACITY_EXCEEDED;

    std::size_t scratch{};
    using spatial_build_detail::append_range;
    if (!append_range(scratch, block_capacity, sizeof(std::uint32_t),
                      plan.compaction_counts) ||
        !append_range(scratch, block_capacity, sizeof(std::uint32_t),
                      plan.compaction_offsets) ||
        !append_range(scratch, capacity, sizeof(std::uint32_t),
                      plan.active_indices) ||
        !append_range(scratch, 1U, sizeof(std::uint32_t), plan.active_count) ||
        !append_range(scratch, capacity, sizeof(std::uint64_t), plan.keys[0]) ||
        !append_range(scratch, capacity, sizeof(std::uint64_t), plan.keys[1]) ||
        !append_range(scratch, capacity, sizeof(std::uint32_t), plan.values[0]) ||
        !append_range(scratch, capacity, sizeof(std::uint32_t), plan.values[1]) ||
        !append_range(scratch, radix_slots, sizeof(std::uint32_t),
                      plan.radix_histograms) ||
        !append_range(scratch, radix_slots, sizeof(std::uint32_t),
                      plan.radix_offsets) ||
        !append_range(scratch, block_capacity, sizeof(SpatialAabb),
                      plan.block_bounds) ||
        !append_range(scratch, 1U, sizeof(SpatialAabb), plan.scene_bounds) ||
        !append_range(scratch, internal_capacity, sizeof(std::uint32_t),
                      plan.arrivals) ||
        !append_range(scratch, 1U, sizeof(std::uint32_t), plan.status))
        return PM_ERROR_CAPACITY_EXCEEDED;
    plan.scratch_bytes = scratch;

    std::size_t output{};
    if (!append_range(output, internal_capacity, sizeof(std::uint32_t),
                      plan.left_children) ||
        !append_range(output, internal_capacity, sizeof(std::uint32_t),
                      plan.right_children) ||
        !append_range(output, plan.node_capacity, sizeof(std::uint32_t),
                      plan.parents) ||
        !append_range(output, capacity, sizeof(std::uint32_t),
                      plan.leaf_sources) ||
        !append_range(output, plan.node_capacity, sizeof(SpatialAabb),
                      plan.node_bounds))
        return PM_ERROR_CAPACITY_EXCEEDED;
    plan.output_bytes = output;
    return PM_SUCCESS;
}

[[nodiscard]] constexpr SpatialBuildDispatchBudget spatial_build_dispatch_budget(
    std::uint32_t capacity, std::uint32_t active_count) noexcept {
    SpatialBuildDispatchBudget result{};
    result.capacity = capacity;
    result.active_count = std::min(capacity, active_count);
    result.capacity_blocks = capacity / spatial_build_tile_size +
        (capacity % spatial_build_tile_size != 0U);
    result.active_blocks = result.active_count / spatial_build_tile_size +
        (result.active_count % spatial_build_tile_size != 0U);
    if (result.active_count == 0U) return result;

    if (result.active_count == 1U) {
        result.sort_policy = SpatialSortPolicy::none;
    } else if (result.active_count <= spatial_small_sort_capacity) {
        result.sort_policy = SpatialSortPolicy::workgroup_bitonic;
        result.sort_dispatches = 1U;
    } else {
        result.sort_policy = SpatialSortPolicy::radix8;
        result.sort_dispatches = 3U * spatial_radix_passes;
    }

    // Stable flag compaction, two-stage bounds reduction, key emission, sort,
    // topology, optional bottom-up union, admission resolution, and transfer
    // finalization.
    result.total_dispatches = 3U + 2U + 1U + result.sort_dispatches + 1U +
        (result.active_count > 1U ? 1U : 0U) + 2U;
    result.total_workgroups =
        2ULL * result.capacity_blocks + 1ULL +
        result.active_blocks + 1ULL + result.active_blocks +
        result.active_blocks +
        (result.active_count > 1U ? result.active_blocks : 0U) +
        2ULL * result.active_blocks;
    if (result.sort_policy == SpatialSortPolicy::workgroup_bitonic)
        result.total_workgroups += 1U;
    else if (result.sort_policy == SpatialSortPolicy::radix8)
        result.total_workgroups += spatial_radix_passes *
            (2ULL * result.active_blocks + 1ULL);
    return result;
}

} // namespace parallel_mater::simulation
