// SPDX-License-Identifier: MIT
#pragma once

#include "spatial_service.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t rigid_spatial_family = 1U;
inline constexpr std::uint32_t rigid_spatial_body_primitive = 1U;
inline constexpr std::uint32_t rigid_spatial_query_mask = 1U << 0U;
inline constexpr std::uint32_t rigid_spatial_solve_sweeps = 8U;

struct RigidSpatialPlan {
    std::uint32_t body_capacity{};
    std::uint32_t candidates_per_body{};
    std::uint32_t candidate_slot_capacity{};
    std::uint32_t compaction_blocks{};

    SpatialBufferRange candidate_pair_flags{};
    SpatialBufferRange candidate_pair_codes{};
    SpatialBufferRange compaction_counts{};
    SpatialBufferRange compaction_offsets{};
    SpatialBufferRange compacted_slots{};
    SpatialBufferRange status{};
    std::size_t scratch_bytes{};

    SpatialBufferRange active_pairs{};
    SpatialBufferRange active_pair_count{};
    std::size_t output_bytes{};
};

struct RigidSpatialWorkBudget {
    std::uint32_t body_count{};
    std::uint32_t candidates_per_body{};
    std::uint32_t candidate_slot_capacity{};
    std::uint32_t consumer_dispatches{};
    std::uint64_t consumer_workgroups{};
    std::uint64_t legacy_candidate_flag_bytes{};
    std::uint64_t bounded_candidate_bytes{};
    std::uint64_t bounded_pair_scratch_bytes{};
    std::uint64_t bounded_pair_output_bytes{};
};

[[nodiscard]] inline PmResult make_rigid_spatial_plan(
    std::uint32_t body_capacity, std::uint32_t candidates_per_body,
    RigidSpatialPlan &plan) noexcept {
    plan = {};
    if ((body_capacity == 0U) != (candidates_per_body == 0U))
        return PM_ERROR_INVALID_ARGUMENT;
    // Pair codes retain the established body * count + collider ABI.
    if (body_capacity > 65536U) return PM_ERROR_CAPACITY_EXCEEDED;

    std::size_t slots{};
    if (!spatial_build_detail::checked_multiply(
            body_capacity, candidates_per_body, slots) ||
        slots > std::numeric_limits<std::uint32_t>::max())
        return PM_ERROR_CAPACITY_EXCEEDED;
    plan.body_capacity = body_capacity;
    plan.candidates_per_body = candidates_per_body;
    plan.candidate_slot_capacity = static_cast<std::uint32_t>(slots);
    plan.compaction_blocks = plan.candidate_slot_capacity / 256U +
        (plan.candidate_slot_capacity % 256U != 0U);

    using spatial_build_detail::append_range;
    std::size_t scratch{};
    if (!append_range(scratch, slots, sizeof(std::uint8_t),
                      plan.candidate_pair_flags) ||
        !append_range(scratch, slots, sizeof(std::uint32_t),
                      plan.candidate_pair_codes) ||
        !append_range(scratch, plan.compaction_blocks, sizeof(std::uint32_t),
                      plan.compaction_counts) ||
        !append_range(scratch, plan.compaction_blocks, sizeof(std::uint32_t),
                      plan.compaction_offsets) ||
        !append_range(scratch, slots, sizeof(std::uint32_t),
                      plan.compacted_slots) ||
        !append_range(scratch, 1U, sizeof(std::uint32_t), plan.status))
        return PM_ERROR_CAPACITY_EXCEEDED;
    plan.scratch_bytes = scratch;

    std::size_t output{};
    if (!append_range(output, slots, sizeof(std::uint32_t),
                      plan.active_pairs) ||
        !append_range(output, 1U, sizeof(std::uint32_t),
                      plan.active_pair_count))
        return PM_ERROR_CAPACITY_EXCEEDED;
    plan.output_bytes = output;
    return PM_SUCCESS;
}

[[nodiscard]] constexpr RigidSpatialWorkBudget rigid_spatial_work_budget(
    std::uint32_t body_count, std::uint32_t candidates_per_body) noexcept {
    RigidSpatialWorkBudget result{};
    result.body_count = body_count;
    result.candidates_per_body = candidates_per_body;
    result.candidate_slot_capacity =
        body_count != 0U && candidates_per_body >
                std::numeric_limits<std::uint32_t>::max() / body_count
        ? std::numeric_limits<std::uint32_t>::max()
        : body_count * candidates_per_body;
    const std::uint64_t slots = result.candidate_slot_capacity;
    const std::uint64_t compaction_blocks = (slots + 255U) / 256U;
    const std::uint64_t filter_blocks = (slots + 127U) / 128U;
    // Proxy emission, one shared-index query, pair filtering, stable flag
    // compaction (count/scan/scatter), and pair-code gather.
    result.consumer_dispatches = body_count == 0U ? 0U : 7U;
    result.consumer_workgroups = body_count == 0U ? 0U :
        (body_count + 127U) / 128U +
        (body_count + 63U) / 64U +
        2U * filter_blocks + 2U * compaction_blocks + 1U;
    result.legacy_candidate_flag_bytes =
        std::uint64_t{body_count} * body_count;
    result.bounded_candidate_bytes = slots * sizeof(std::uint32_t);
    auto append_aligned = [](std::uint64_t &cursor,
                             std::uint64_t bytes) constexpr {
        constexpr std::uint64_t alignment = 16U;
        cursor += (alignment - cursor % alignment) % alignment;
        cursor += bytes;
    };
    append_aligned(result.bounded_pair_scratch_bytes,
                   slots * sizeof(std::uint8_t));
    append_aligned(result.bounded_pair_scratch_bytes,
                   slots * sizeof(std::uint32_t));
    append_aligned(result.bounded_pair_scratch_bytes,
                   compaction_blocks * sizeof(std::uint32_t));
    append_aligned(result.bounded_pair_scratch_bytes,
                   compaction_blocks * sizeof(std::uint32_t));
    append_aligned(result.bounded_pair_scratch_bytes,
                   slots * sizeof(std::uint32_t));
    append_aligned(result.bounded_pair_scratch_bytes, sizeof(std::uint32_t));
    append_aligned(result.bounded_pair_output_bytes,
                   slots * sizeof(std::uint32_t));
    append_aligned(result.bounded_pair_output_bytes, sizeof(std::uint32_t));
    return result;
}

[[nodiscard]] inline PmResult rigid_spatial_proxy_envelope(
    const float start[3], const float predicted[3],
    const float mesh_center[3], float mesh_radius, float collision_margin,
    const float path_deviation[3], const float correction[3], float epsilon,
    SpatialAabb &output) noexcept {
    if (mesh_center == nullptr || !std::isfinite(mesh_radius) ||
        !std::isfinite(collision_margin) || !std::isfinite(epsilon) ||
        mesh_radius < 0.0F || collision_margin < 0.0F || epsilon < 0.0F)
        return PM_ERROR_INVALID_ARGUMENT;
    double center_squared{};
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        if (!std::isfinite(mesh_center[axis]))
            return PM_ERROR_INVALID_ARGUMENT;
        center_squared += static_cast<double>(mesh_center[axis]) *
            mesh_center[axis];
    }
    const double reach = std::sqrt(center_squared) + mesh_radius +
        collision_margin + epsilon;
    if (!std::isfinite(reach) ||
        reach > std::numeric_limits<float>::max())
        return PM_ERROR_CAPACITY_EXCEEDED;
    return spatial_point_envelope(start, predicted, path_deviation, correction,
                                  static_cast<float>(reach), output);
}

// A rigid tick materializes candidates once, then every contact/joint sweep
// must retain the exact epoch token and its one selected build/refit.
class RigidSpatialSolveEpoch {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        std::uint32_t sweep_budget = rigid_spatial_solve_sweeps) noexcept {
        if (active_ || token.tick == 0U || sweep_budget == 0U ||
            counters.publications != 1U ||
            counters.builds + counters.refits != 1U ||
            counters.overflows != 0U || counters.queries == 0U)
            return PM_ERROR_INVALID_STATE;
        token_ = token;
        builds_ = counters.builds;
        refits_ = counters.refits;
        publications_ = counters.publications;
        sweep_budget_ = sweep_budget;
        completed_sweeps_ = 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_sweep(
        const SpatialEpochToken &token,
        const SpatialEpochCounters &counters) noexcept {
        if (!active_ || token != token_ ||
            counters.builds != builds_ || counters.refits != refits_ ||
            counters.publications != publications_ ||
            counters.builds + counters.refits != 1U ||
            counters.overflows != 0U || completed_sweeps_ >= sweep_budget_)
            return PM_ERROR_INVALID_STATE;
        ++completed_sweeps_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult finish() noexcept {
        if (!active_ || completed_sweeps_ != sweep_budget_)
            return PM_ERROR_INVALID_STATE;
        active_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t completed_sweeps() const noexcept {
        return completed_sweeps_;
    }

private:
    SpatialEpochToken token_{};
    std::uint32_t builds_{};
    std::uint32_t refits_{};
    std::uint32_t publications_{};
    std::uint32_t sweep_budget_{};
    std::uint32_t completed_sweeps_{};
    bool active_{};
};

} // namespace parallel_mater::simulation
