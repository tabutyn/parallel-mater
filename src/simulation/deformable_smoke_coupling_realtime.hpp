// SPDX-License-Identifier: MIT
#pragma once

#include "deformable_spatial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t deformable_smoke_coupling_profile_version = 1U;
inline constexpr std::uint32_t deformable_smoke_coupling_sweeps = 4U;
inline constexpr std::uint32_t
    deformable_smoke_coupling_dispatches_per_sweep = 3U;
inline constexpr std::uint32_t deformable_smoke_coupling_maximum_edges = 16U;
inline constexpr float deformable_smoke_coupling_timestep = 1.0F / 60.0F;
inline constexpr std::uint32_t deformable_smoke_coupling_status_invalid_profile =
    1U << 28U;
inline constexpr std::uint32_t deformable_smoke_coupling_status_invalid_timestep =
    1U << 29U;
inline constexpr std::uint32_t deformable_smoke_coupling_status_invalid_material =
    1U << 30U;
inline constexpr std::uint32_t
    deformable_smoke_coupling_status_correction_exceeded = 1U << 31U;

// Pair order is canonical. For smoke pairs the deformable endpoint is first and
// the smoke endpoint is second. Contact moves smoke only; drag moves the
// deformable only, matching the API 0.2 directional contract.
enum class DeformableSmokeCouplingPair : std::uint32_t {
    cloth_soft_body = 1U,
    cloth_rope = 2U,
    cloth_smoke_particle = 3U,
    cloth_smoke_grid = 4U,
    soft_body_rope = 5U,
    soft_body_smoke_particle = 6U,
    soft_body_smoke_grid = 7U,
    rope_smoke_particle = 8U,
    rope_smoke_grid = 9U,
};

enum class DeformableSmokeCouplingMode : std::uint32_t {
    contact = 1U,
    attachment = 2U,
    drag = 3U,
};

[[nodiscard]] constexpr bool deformable_smoke_pair_has_smoke(
    DeformableSmokeCouplingPair pair) noexcept {
    return pair == DeformableSmokeCouplingPair::cloth_smoke_particle ||
        pair == DeformableSmokeCouplingPair::cloth_smoke_grid ||
        pair == DeformableSmokeCouplingPair::soft_body_smoke_particle ||
        pair == DeformableSmokeCouplingPair::soft_body_smoke_grid ||
        pair == DeformableSmokeCouplingPair::rope_smoke_particle ||
        pair == DeformableSmokeCouplingPair::rope_smoke_grid;
}

[[nodiscard]] constexpr bool deformable_smoke_coupling_mode_supported(
    DeformableSmokeCouplingPair pair,
    DeformableSmokeCouplingMode mode) noexcept {
    switch (pair) {
    case DeformableSmokeCouplingPair::cloth_soft_body:
        return mode == DeformableSmokeCouplingMode::contact;
    case DeformableSmokeCouplingPair::cloth_rope:
        return mode == DeformableSmokeCouplingMode::attachment;
    case DeformableSmokeCouplingPair::soft_body_rope:
        return mode == DeformableSmokeCouplingMode::contact ||
            mode == DeformableSmokeCouplingMode::attachment;
    case DeformableSmokeCouplingPair::cloth_smoke_particle:
    case DeformableSmokeCouplingPair::cloth_smoke_grid:
    case DeformableSmokeCouplingPair::soft_body_smoke_particle:
    case DeformableSmokeCouplingPair::soft_body_smoke_grid:
    case DeformableSmokeCouplingPair::rope_smoke_particle:
    case DeformableSmokeCouplingPair::rope_smoke_grid:
        return mode == DeformableSmokeCouplingMode::contact ||
            mode == DeformableSmokeCouplingMode::drag;
    }
    return false;
}

[[nodiscard]] constexpr bool deformable_smoke_coupling_is_two_way(
    DeformableSmokeCouplingPair pair) noexcept {
    return !deformable_smoke_pair_has_smoke(pair);
}

[[nodiscard]] inline bool deformable_smoke_coupling_step_request_valid(
    float timestep, std::uint32_t substeps) noexcept {
    const float lower = std::nextafter(
        deformable_smoke_coupling_timestep, 0.0F);
    const float upper = std::nextafter(deformable_smoke_coupling_timestep,
        std::numeric_limits<float>::infinity());
    return substeps == 1U && std::isfinite(timestep) &&
        timestep >= lower && timestep <= upper;
}

struct DeformableSmokeCouplingRealtimeInput {
    DeformableSmokeCouplingPair pair{};
    DeformableSmokeCouplingMode mode{};
    float world_coordinate{};
    float characteristic_length{};
    float first_mass{};
    float second_mass{};
    bool first_dynamic{true};
    bool second_dynamic{true};
    float linear_speed{};
    float acceleration_displacement{};
    float solver_correction{};
    float contact_distance{};
    float friction{};
    float response{};
};

[[nodiscard]] inline PmResult validate_deformable_smoke_coupling_input(
    const DeformableSmokeCouplingRealtimeInput &input) noexcept {
    const float values[]{input.world_coordinate, input.characteristic_length,
        input.first_mass, input.second_mass, input.linear_speed,
        input.acceleration_displacement, input.solver_correction,
        input.contact_distance, input.friction, input.response};
    for (const float value : values)
        if (!std::isfinite(value)) return PM_ERROR_INVALID_ARGUMENT;
    if (!deformable_smoke_coupling_mode_supported(input.pair, input.mode))
        return PM_ERROR_INVALID_ARGUMENT;
    const auto valid_mass = [](float mass, bool dynamic) noexcept {
        return dynamic ? mass >= 0.001F && mass <= 10'000.0F : mass == 0.0F;
    };
    if (!valid_mass(input.first_mass, input.first_dynamic) ||
        !valid_mass(input.second_mass, input.second_dynamic))
        return PM_ERROR_INVALID_ARGUMENT;
    const bool smoke_pair = deformable_smoke_pair_has_smoke(input.pair);
    if ((!smoke_pair && !input.first_dynamic && !input.second_dynamic) ||
        (smoke_pair && input.mode == DeformableSmokeCouplingMode::contact &&
            !input.second_dynamic) ||
        (smoke_pair && input.mode == DeformableSmokeCouplingMode::drag &&
            !input.first_dynamic))
        return PM_ERROR_INVALID_ARGUMENT;
    if (!smoke_pair && input.first_dynamic && input.second_dynamic) {
        const float ratio = std::max(input.first_mass, input.second_mass) /
            std::min(input.first_mass, input.second_mass);
        if (ratio > 100.0F) return PM_ERROR_INVALID_ARGUMENT;
    }
    if (std::abs(input.world_coordinate) > 10'000.0F ||
        input.characteristic_length < 0.005F ||
        input.characteristic_length > 2.0F || input.linear_speed < 0.0F ||
        input.linear_speed > 20.0F ||
        input.linear_speed * deformable_smoke_coupling_timestep >
            0.5F * input.characteristic_length ||
        input.acceleration_displacement < 0.0F ||
        input.acceleration_displacement >
            0.25F * input.characteristic_length ||
        input.solver_correction < 0.0F ||
        input.solver_correction > 0.25F * input.characteristic_length ||
        input.friction < 0.0F || input.friction > 2.0F)
        return PM_ERROR_INVALID_ARGUMENT;
    if (input.mode == DeformableSmokeCouplingMode::contact) {
        if (input.contact_distance <= 0.0F ||
            input.contact_distance > input.characteristic_length ||
            input.response != 0.0F)
            return PM_ERROR_INVALID_ARGUMENT;
    } else if (input.contact_distance != 0.0F || input.friction != 0.0F ||
               input.response < 0.05F || input.response > 0.995F) {
        return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}

struct DeformableSmokeCouplingRegistration {
    std::uint64_t edge_id{};
    DeformableSmokeCouplingPair pair{};
    DeformableSmokeCouplingMode mode{};
    std::uint32_t first_generation{};
    std::uint32_t second_generation{};
    std::uint32_t constraint_count{};

    auto operator<=>(const DeformableSmokeCouplingRegistration &) const = default;
};

struct DeformableSmokeCouplingOperatorBudget {
    DeformableSmokeCouplingRegistration registration{};
    std::uint32_t algebraic_sweeps{};
    std::uint64_t constraint_visits{};
};

struct DeformableSmokeCouplingWorkBudget {
    std::uint32_t edge_count{};
    std::uint32_t first_endpoint_count{};
    std::uint32_t second_endpoint_count{};
    std::uint32_t constraint_count{};
    std::uint32_t coupling_sweeps_per_edge{};
    std::uint64_t edge_sweeps{};
    std::uint64_t constraint_visits{};
    std::uint32_t time_advances{};
    std::uint32_t spatial_updates_during_solve{};
    std::uint32_t batched_dispatches{};
    std::uint64_t batched_workgroups{};
    std::uint32_t intra_tick_host_readbacks{};
    std::size_t solver_scratch_bytes{};
    std::array<DeformableSmokeCouplingOperatorBudget,
               deformable_smoke_coupling_maximum_edges> operators{};
};

[[nodiscard]] inline PmResult make_deformable_smoke_coupling_work_budget(
    std::span<const DeformableSmokeCouplingRegistration> registrations,
    std::uint32_t first_endpoint_count, std::uint32_t second_endpoint_count,
    DeformableSmokeCouplingWorkBudget &result) noexcept {
    result = {};
    if (registrations.size() > deformable_smoke_coupling_maximum_edges)
        return PM_ERROR_CAPACITY_EXCEEDED;
    result.edge_count = static_cast<std::uint32_t>(registrations.size());
    result.first_endpoint_count = first_endpoint_count;
    result.second_endpoint_count = second_endpoint_count;
    result.coupling_sweeps_per_edge = deformable_smoke_coupling_sweeps;
    for (std::size_t index = 0U; index < registrations.size(); ++index) {
        const auto &registration = registrations[index];
        if (registration.edge_id == 0U ||
            registration.first_generation == 0U ||
            registration.second_generation == 0U ||
            registration.constraint_count == 0U ||
            !deformable_smoke_coupling_mode_supported(
                registration.pair, registration.mode) ||
            result.constraint_count >
                std::numeric_limits<std::uint32_t>::max() -
                    registration.constraint_count)
            return PM_ERROR_INVALID_ARGUMENT;
        result.constraint_count += registration.constraint_count;
        result.operators[index] = {registration,
            deformable_smoke_coupling_sweeps,
            std::uint64_t{registration.constraint_count} *
                deformable_smoke_coupling_sweeps};
    }
    result.edge_sweeps = std::uint64_t{result.edge_count} *
        deformable_smoke_coupling_sweeps;
    result.constraint_visits = std::uint64_t{result.constraint_count} *
        deformable_smoke_coupling_sweeps;
    result.batched_dispatches = result.constraint_count == 0U ? 0U :
        deformable_smoke_coupling_dispatches_per_sweep *
            deformable_smoke_coupling_sweeps;
    const std::uint64_t constraint_groups =
        (std::uint64_t{result.constraint_count} + 127U) / 128U;
    const std::uint64_t first_groups =
        (std::uint64_t{first_endpoint_count} + 127U) / 128U;
    const std::uint64_t second_groups =
        (std::uint64_t{second_endpoint_count} + 127U) / 128U;
    result.batched_workgroups = result.constraint_count == 0U ? 0U :
        deformable_smoke_coupling_sweeps *
            (constraint_groups + first_groups + second_groups);
    // Four Vec3 proposal streams plus correction-consumption for both endpoint
    // arrays. Candidate and hierarchy storage belong to the shared F8-10 epoch.
    result.solver_scratch_bytes =
        static_cast<std::size_t>(result.constraint_count) * 4U *
            3U * sizeof(float) +
        (static_cast<std::size_t>(first_endpoint_count) +
            second_endpoint_count) * sizeof(float);
    return PM_SUCCESS;
}

class DeformableSmokeCouplingTickContract {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        const DeformableSmokeCouplingWorkBudget &budget,
        std::span<const DeformableSmokeCouplingRegistration> registrations,
        std::uint64_t topology_revision) noexcept {
        if (active_ || token.tick == 0U || topology_revision == 0U ||
            registrations.size() != budget.edge_count ||
            registrations.size() > registrations_.size() ||
            budget.coupling_sweeps_per_edge !=
                deformable_smoke_coupling_sweeps ||
            budget.edge_sweeps != registrations.size() *
                deformable_smoke_coupling_sweeps ||
            budget.time_advances != 0U ||
            budget.spatial_updates_during_solve != 0U ||
            budget.intra_tick_host_readbacks != 0U ||
            counters.publications != 1U ||
            counters.builds + counters.refits > 1U ||
            counters.overflows != 0U)
            return PM_ERROR_INVALID_STATE;
        for (std::size_t index = 0U; index < registrations.size(); ++index) {
            const auto &registration = registrations[index];
            if (registration != budget.operators[index].registration ||
                registration.edge_id == 0U ||
                registration.first_generation == 0U ||
                registration.second_generation == 0U ||
                registration.constraint_count == 0U ||
                !deformable_smoke_coupling_mode_supported(
                    registration.pair, registration.mode) ||
                (index != 0U && !canonical_less(
                    registrations[index - 1U], registration)))
                return PM_ERROR_INVALID_STATE;
            registrations_[index] = registration;
        }
        token_ = token;
        counters_ = counters;
        topology_revision_ = topology_revision;
        edge_count_ = static_cast<std::uint32_t>(registrations.size());
        next_edge_ = 0U;
        completed_sweeps_ = edge_count_ == 0U
            ? deformable_smoke_coupling_sweeps : 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_edge(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        std::uint64_t topology_revision,
        const DeformableSmokeCouplingRegistration &registration) noexcept {
        if (!active_ || edge_count_ == 0U || token != token_ ||
            counters.builds != counters_.builds ||
            counters.refits != counters_.refits ||
            counters.publications != counters_.publications ||
            counters.overflows != 0U ||
            topology_revision != topology_revision_ ||
            completed_sweeps_ >= deformable_smoke_coupling_sweeps ||
            registration != registrations_[next_edge_])
            return PM_ERROR_INVALID_STATE;
        ++next_edge_;
        if (next_edge_ == edge_count_) {
            next_edge_ = 0U;
            ++completed_sweeps_;
        }
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult finish() noexcept {
        if (!active_ || next_edge_ != 0U ||
            completed_sweeps_ != deformable_smoke_coupling_sweeps)
            return PM_ERROR_INVALID_STATE;
        active_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t completed_sweeps() const noexcept {
        return completed_sweeps_;
    }

private:
    [[nodiscard]] static constexpr bool canonical_less(
        const DeformableSmokeCouplingRegistration &left,
        const DeformableSmokeCouplingRegistration &right) noexcept {
        if (left.pair != right.pair) return left.pair < right.pair;
        if (left.mode != right.mode) return left.mode < right.mode;
        return left.edge_id < right.edge_id;
    }

    SpatialEpochToken token_{};
    SpatialEpochCounters counters_{};
    std::array<DeformableSmokeCouplingRegistration,
               deformable_smoke_coupling_maximum_edges> registrations_{};
    std::uint64_t topology_revision_{};
    std::uint32_t edge_count_{};
    std::uint32_t next_edge_{};
    std::uint32_t completed_sweeps_{};
    bool active_{};
};

} // namespace parallel_mater::simulation
