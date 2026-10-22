// SPDX-License-Identifier: MIT
#pragma once

#include "spatial_service.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t deformable_spatial_cloth_family = 3U;
inline constexpr std::uint32_t deformable_spatial_rope_family = 4U;
inline constexpr std::uint32_t deformable_spatial_soft_family = 5U;
inline constexpr std::uint32_t deformable_spatial_static_family = 6U;
inline constexpr std::uint32_t deformable_spatial_segment_primitive = 1U;
inline constexpr std::uint32_t deformable_spatial_triangle_primitive = 2U;
inline constexpr std::uint32_t deformable_spatial_contact_mask = 1U << 2U;
inline constexpr std::uint32_t smoke_spatial_obstacle_mask = 1U << 3U;
inline constexpr std::uint32_t deformable_spatial_collision_passes = 4U;

struct DeformableSpatialPlan {
    std::uint32_t query_capacity{};
    std::uint32_t candidates_per_query{};
    std::uint32_t candidate_slot_capacity{};
    SpatialBufferRange query_leaf_sources{};
    SpatialBufferRange filtered_counts{};
    SpatialBufferRange filtered_sources{};
    SpatialBufferRange status{};
    std::size_t scratch_bytes{};
};

struct DeformableSpatialWorkBudget {
    std::uint32_t live_primitives{};
    std::uint32_t primitive_capacity{};
    std::uint32_t smoke_obstacle_queries{};
    std::uint32_t deformable_domains{};
    std::uint32_t candidates_per_query{};
    std::uint32_t boundary_dispatches{};
    std::uint32_t spatial_updates{};
    std::uint32_t index_dispatches_in_solve{};
    std::uint32_t per_edge_rebuild_dispatches{};
    std::uint64_t live_query_workgroups{};
    std::uint64_t capacity_query_workgroups{};
    std::uint64_t bounded_candidate_bytes{};
    std::uint64_t avoided_per_consumer_bounds_bytes{};
};

struct DeformableSpatialPointMotion {
    std::array<float, 3> previous{};
    std::array<float, 3> predicted{};
    std::array<float, 3> path_deviation{};
    std::array<float, 3> correction_budget{};
};

[[nodiscard]] inline bool deformable_spatial_family_valid(
    std::uint32_t family) noexcept {
    return family >= deformable_spatial_cloth_family &&
        family <= deformable_spatial_static_family;
}

[[nodiscard]] inline PmResult make_deformable_spatial_plan(
    std::uint32_t query_capacity, std::uint32_t candidates_per_query,
    DeformableSpatialPlan &plan) noexcept {
    plan = {};
    if ((query_capacity == 0U) != (candidates_per_query == 0U))
        return PM_ERROR_INVALID_ARGUMENT;
    std::size_t slots{};
    if (!spatial_build_detail::checked_multiply(
            query_capacity, candidates_per_query, slots) ||
        slots > std::numeric_limits<std::uint32_t>::max())
        return PM_ERROR_CAPACITY_EXCEEDED;
    plan.query_capacity = query_capacity;
    plan.candidates_per_query = candidates_per_query;
    plan.candidate_slot_capacity = static_cast<std::uint32_t>(slots);
    using spatial_build_detail::append_range;
    std::size_t scratch{};
    if (!append_range(scratch, query_capacity, sizeof(std::uint32_t),
                      plan.query_leaf_sources) ||
        !append_range(scratch, query_capacity, sizeof(std::uint32_t),
                      plan.filtered_counts) ||
        !append_range(scratch, slots, sizeof(std::uint32_t),
                      plan.filtered_sources) ||
        !append_range(scratch, 1U, sizeof(std::uint32_t), plan.status))
        return PM_ERROR_CAPACITY_EXCEEDED;
    plan.scratch_bytes = scratch;
    return PM_SUCCESS;
}

[[nodiscard]] constexpr DeformableSpatialWorkBudget
deformable_spatial_work_budget(
    std::uint32_t live_primitives, std::uint32_t primitive_capacity,
    std::uint32_t smoke_obstacle_queries, std::uint32_t deformable_domains,
    std::uint32_t candidates_per_query) noexcept {
    DeformableSpatialWorkBudget result{};
    result.live_primitives = live_primitives;
    result.primitive_capacity = primitive_capacity;
    result.smoke_obstacle_queries = smoke_obstacle_queries;
    result.deformable_domains = deformable_domains;
    result.candidates_per_query = candidates_per_query;
    // One emitter per deformable domain, one smoke query emitter, and one
    // owner-aware filter for each consumer class. The service build/query is
    // shared and accounted once at the tick boundary.
    result.boundary_dispatches = deformable_domains +
        (smoke_obstacle_queries != 0U ? 2U : 0U) + 2U;
    result.spatial_updates = live_primitives == 0U &&
            smoke_obstacle_queries == 0U ? 0U : 1U;
    result.index_dispatches_in_solve = 0U;
    result.per_edge_rebuild_dispatches = 0U;
    result.live_query_workgroups =
        (std::uint64_t{live_primitives} + smoke_obstacle_queries + 63U) / 64U;
    result.capacity_query_workgroups =
        (std::uint64_t{primitive_capacity} + smoke_obstacle_queries + 63U) /
        64U;
    const std::uint64_t queries =
        std::uint64_t{primitive_capacity} + smoke_obstacle_queries;
    result.bounded_candidate_bytes = queries * candidates_per_query *
        sizeof(std::uint32_t);
    // Legacy refit tables repeated bounds for every deformable/smoke
    // consumer. Exclude BVH/sort scratch, so this remains a lower bound.
    result.avoided_per_consumer_bounds_bytes =
        std::uint64_t{primitive_capacity} * sizeof(SpatialAabb) *
        (deformable_domains + (smoke_obstacle_queries != 0U ? 1U : 0U));
    return result;
}

[[nodiscard]] inline PmResult deformable_spatial_primitive_envelope(
    std::span<const DeformableSpatialPointMotion> vertices, float thickness,
    float epsilon, SpatialAabb &output) noexcept {
    if ((vertices.size() != 2U && vertices.size() != 3U) ||
        !std::isfinite(thickness) || thickness < 0.0F ||
        !std::isfinite(epsilon) || epsilon <= 0.0F)
        return PM_ERROR_INVALID_ARGUMENT;
    const double radius = static_cast<double>(thickness) + epsilon;
    if (!std::isfinite(radius) ||
        radius > std::numeric_limits<float>::max())
        return PM_ERROR_CAPACITY_EXCEEDED;
    for (std::size_t vertex = 0U; vertex < vertices.size(); ++vertex) {
        SpatialAabb envelope{};
        const auto &point = vertices[vertex];
        const auto result = spatial_point_envelope(
            point.previous.data(), point.predicted.data(),
            point.path_deviation.data(), point.correction_budget.data(),
            static_cast<float>(radius), envelope);
        if (result != PM_SUCCESS) return result;
        output = vertex == 0U ? envelope : spatial_union(output, envelope);
    }
    return PM_SUCCESS;
}

// Deformable contact iterations and smoke obstacle consumers pin one published
// token. No pass may rebuild/refit, publish, or append another broad query.
class DeformableSpatialSolveEpoch {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        std::uint32_t pass_budget =
            deformable_spatial_collision_passes) noexcept {
        if (active_ || token.tick == 0U || pass_budget == 0U ||
            counters.publications != 1U ||
            counters.builds + counters.refits != 1U ||
            counters.overflows != 0U || counters.queries == 0U)
            return PM_ERROR_INVALID_STATE;
        token_ = token;
        counters_ = counters;
        pass_budget_ = pass_budget;
        completed_passes_ = 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_pass(
        const SpatialEpochToken &token,
        const SpatialEpochCounters &counters) noexcept {
        if (!active_ || token != token_ || counters.builds != counters_.builds ||
            counters.refits != counters_.refits ||
            counters.publications != counters_.publications ||
            counters.queries != counters_.queries || counters.overflows != 0U ||
            completed_passes_ >= pass_budget_)
            return PM_ERROR_INVALID_STATE;
        ++completed_passes_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult finish() noexcept {
        if (!active_ || completed_passes_ != pass_budget_)
            return PM_ERROR_INVALID_STATE;
        active_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t completed_passes() const noexcept {
        return completed_passes_;
    }

private:
    SpatialEpochToken token_{};
    SpatialEpochCounters counters_{};
    std::uint32_t pass_budget_{};
    std::uint32_t completed_passes_{};
    bool active_{};
};

} // namespace parallel_mater::simulation
