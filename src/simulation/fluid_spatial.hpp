// SPDX-License-Identifier: MIT
#pragma once

#include "spatial_service.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t fluid_spatial_family = 2U;
inline constexpr std::uint32_t fluid_spatial_particle_primitive = 1U;
inline constexpr std::uint32_t fluid_spatial_query_mask = 1U << 1U;
inline constexpr std::uint32_t fluid_spatial_density_sweeps = 6U;

// The shared service owns the hierarchy and unfiltered query output. This plan
// owns only a reusable owner-filtered view; it is never multiplied by a fluid
// pair and contains no radix-sort keys or workspace.
struct FluidSpatialPlan {
    std::uint32_t query_capacity{};
    std::uint32_t candidates_per_query{};
    std::uint32_t candidate_slot_capacity{};
    SpatialBufferRange query_leaf_sources{};
    SpatialBufferRange filtered_counts{};
    SpatialBufferRange filtered_sources{};
    SpatialBufferRange status{};
    std::size_t scratch_bytes{};
};

struct FluidSpatialWorkBudget {
    std::uint32_t live_particles{};
    std::uint32_t particle_capacity{};
    std::uint32_t source_proposals{};
    std::uint32_t populations{};
    std::uint32_t directed_pair_queries{};
    std::uint32_t candidates_per_query{};
    std::uint32_t boundary_dispatches{};
    std::uint32_t neighbor_index_dispatches_in_solve{};
    std::uint32_t neighbor_sort_dispatches{};
    std::uint64_t live_query_workgroups{};
    std::uint64_t capacity_query_workgroups{};
    std::uint64_t filtered_candidate_bytes{};
    std::uint64_t avoided_pair_key_bytes{};
};

[[nodiscard]] inline PmResult make_fluid_spatial_plan(
    std::uint32_t query_capacity, std::uint32_t candidates_per_query,
    FluidSpatialPlan &plan) noexcept {
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

[[nodiscard]] constexpr FluidSpatialWorkBudget fluid_spatial_work_budget(
    std::uint32_t live_particles, std::uint32_t particle_capacity,
    std::uint32_t source_proposals, std::uint32_t populations,
    std::uint32_t directed_pair_queries,
    std::uint32_t candidates_per_query) noexcept {
    FluidSpatialWorkBudget result{};
    result.live_particles = live_particles;
    result.particle_capacity = particle_capacity;
    result.source_proposals = source_proposals;
    result.populations = populations;
    result.directed_pair_queries = directed_pair_queries;
    result.candidates_per_query = candidates_per_query;
    // Population/proposal emission, one service query, one owner filter for
    // every self/directed-pair view, and one admission application.
    result.boundary_dispatches = populations +
        (source_proposals != 0U ? 2U : 0U) + 1U + populations +
        directed_pair_queries;
    // Algebraic sweeps reuse the owner-filtered lists. They may repeat exact
    // distance evaluation, but never emit keys, sort, build, refit or publish.
    result.neighbor_index_dispatches_in_solve = 0U;
    result.neighbor_sort_dispatches = 0U;
    result.live_query_workgroups =
        (std::uint64_t{live_particles} + source_proposals + 63U) / 64U;
    result.capacity_query_workgroups =
        (std::uint64_t{particle_capacity} + source_proposals + 63U) / 64U;
    const std::uint64_t query_capacity =
        std::uint64_t{particle_capacity} + source_proposals;
    result.filtered_candidate_bytes = query_capacity * candidates_per_query *
        sizeof(std::uint32_t);
    // A pair-private legacy index carries two key buffers and two identity
    // buffers. This deliberately excludes backend radix scratch, so the
    // avoided-byte figure is a conservative lower bound.
    result.avoided_pair_key_bytes =
        std::uint64_t{directed_pair_queries / 2U} *
        std::uint64_t{particle_capacity} *
        (2U * sizeof(std::uint64_t) + 2U * sizeof(std::uint32_t));
    return result;
}

[[nodiscard]] inline PmResult fluid_spatial_envelope(
    const float start[3], const float predicted[3],
    const float path_deviation[3], const float correction[3], float radius,
    float epsilon, SpatialAabb &output) noexcept {
    if (!std::isfinite(epsilon) || epsilon <= 0.0F ||
        !std::isfinite(radius) || radius < 0.0F)
        return PM_ERROR_INVALID_ARGUMENT;
    const double expanded = static_cast<double>(radius) + epsilon;
    if (!std::isfinite(expanded) ||
        expanded > std::numeric_limits<float>::max())
        return PM_ERROR_CAPACITY_EXCEEDED;
    return spatial_point_envelope(start, predicted, path_deviation,
                                  correction, static_cast<float>(expanded),
                                  output);
}

// A fluid tick admits/removes particles and publishes once. Every algebraic
// sweep must retain that token and exact build/refit/publication counters.
class FluidSpatialSolveEpoch {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        std::uint32_t sweep_budget = fluid_spatial_density_sweeps) noexcept {
        if (active_ || token.tick == 0U || sweep_budget == 0U ||
            counters.publications != 1U ||
            counters.builds + counters.refits != 1U ||
            counters.overflows != 0U || counters.queries == 0U)
            return PM_ERROR_INVALID_STATE;
        token_ = token;
        builds_ = counters.builds;
        refits_ = counters.refits;
        publications_ = counters.publications;
        query_count_ = counters.queries;
        sweep_budget_ = sweep_budget;
        completed_sweeps_ = 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_sweep(
        const SpatialEpochToken &token,
        const SpatialEpochCounters &counters) noexcept {
        if (!active_ || token != token_ || counters.builds != builds_ ||
            counters.refits != refits_ ||
            counters.publications != publications_ ||
            counters.queries != query_count_ ||
            counters.builds + counters.refits != 1U ||
            counters.overflows != 0U ||
            completed_sweeps_ >= sweep_budget_)
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
    std::uint32_t query_count_{};
    std::uint32_t sweep_budget_{};
    std::uint32_t completed_sweeps_{};
    bool active_{};
};

} // namespace parallel_mater::simulation
