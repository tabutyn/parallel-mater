// SPDX-License-Identifier: MIT
#pragma once

#include "deformable_spatial.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t rope_realtime_profile_version = 1U;
inline constexpr std::uint32_t rope_realtime_constraint_sweeps = 8U;
inline constexpr std::uint32_t rope_realtime_contact_passes = 4U;
inline constexpr float rope_realtime_timestep = 1.0F / 60.0F;
inline constexpr float rope_realtime_solver_damping = 0.05F;
inline constexpr float rope_realtime_maximum_correction_fraction = 0.25F;
inline constexpr std::uint32_t rope_realtime_status_invalid_profile =
    1U << 17U;
inline constexpr std::uint32_t rope_realtime_status_invalid_timestep =
    1U << 18U;
inline constexpr std::uint32_t rope_realtime_status_invalid_material =
    1U << 19U;
inline constexpr std::uint32_t rope_realtime_status_nonfinite_state =
    1U << 20U;
inline constexpr std::uint32_t rope_realtime_status_unresolved_contact =
    1U << 21U;

static_assert(rope_realtime_contact_passes ==
              deformable_spatial_collision_passes);

[[nodiscard]] inline bool rope_realtime_step_request_valid(
    float timestep, std::uint32_t substeps) noexcept {
    const float lower = std::nextafter(rope_realtime_timestep, 0.0F);
    const float upper = std::nextafter(
        rope_realtime_timestep, std::numeric_limits<float>::infinity());
    return std::isfinite(timestep) && timestep >= lower &&
        timestep <= upper && substeps == 1U;
}

[[nodiscard]] inline float rope_realtime_compliance_response(
    float compliance, float effective_inverse_mass,
    float timestep = rope_realtime_timestep) noexcept {
    if (!std::isfinite(compliance) ||
        !std::isfinite(effective_inverse_mass) || !std::isfinite(timestep) ||
        compliance < 0.0F || effective_inverse_mass <= 0.0F ||
        timestep <= 0.0F)
        return std::numeric_limits<float>::quiet_NaN();
    if (compliance == 0.0F) return 1.0F;
    const double ratio = static_cast<double>(compliance) /
        (static_cast<double>(timestep) * timestep *
         effective_inverse_mass);
    return static_cast<float>(1.0 / (1.0 + ratio));
}

// Version 1 uses the existing rope stretch-compliance scalar for both adjacent
// segment length and two-segment chord constraints. This adds bounded bending
// without changing legacy RopeOptions or inventing an unversioned material.
struct RopeRealtimeInput {
    float world_coordinate{};
    float node_spacing{};
    float radius{};
    float total_mass{};
    std::uint32_t node_count{};
    float interacting_mass_ratio{1.0F};
    float linear_speed{};
    float acceleration_displacement{};
    float solver_correction{};
    float authored_strain{};
    float structural_compliance{};
    float effective_inverse_mass{};
    float velocity_damping{};
    float contact_friction{};
    float solver_damping{rope_realtime_solver_damping};
};

[[nodiscard]] inline PmResult validate_rope_realtime_input(
    const RopeRealtimeInput &input) noexcept {
    const float values[]{input.world_coordinate, input.node_spacing,
        input.radius, input.total_mass, input.interacting_mass_ratio,
        input.linear_speed, input.acceleration_displacement,
        input.solver_correction, input.authored_strain,
        input.structural_compliance, input.effective_inverse_mass,
        input.velocity_damping, input.contact_friction,
        input.solver_damping};
    for (const float value : values)
        if (!std::isfinite(value)) return PM_ERROR_INVALID_ARGUMENT;
    const float node_mass = input.node_count == 0U ? 0.0F :
        input.total_mass / static_cast<float>(input.node_count);
    if (std::abs(input.world_coordinate) > 10'000.0F ||
        input.node_count < 2U || input.node_count > 1'024U ||
        input.node_spacing < 0.005F || input.node_spacing > 2.0F ||
        input.radius < 0.5F * input.node_spacing ||
        input.radius > input.node_spacing || node_mass < 0.001F ||
        node_mass > 10'000.0F || input.interacting_mass_ratio < 1.0F ||
        input.interacting_mass_ratio > 100.0F || input.linear_speed < 0.0F ||
        input.linear_speed > 20.0F ||
        input.linear_speed * rope_realtime_timestep >
            0.5F * input.node_spacing ||
        input.acceleration_displacement < 0.0F ||
        input.acceleration_displacement > 0.25F * input.node_spacing ||
        input.solver_correction < 0.0F ||
        input.solver_correction >
            rope_realtime_maximum_correction_fraction * input.node_spacing ||
        input.authored_strain < 0.0F || input.authored_strain > 0.25F ||
        input.structural_compliance < 0.0F ||
        input.effective_inverse_mass <= 0.0F ||
        input.velocity_damping < 0.0F || input.contact_friction < 0.0F ||
        input.contact_friction > 2.0F || input.solver_damping < 0.0F ||
        input.solver_damping > 0.12F)
        return PM_ERROR_INVALID_ARGUMENT;
    if (input.structural_compliance != 0.0F) {
        constexpr float tolerance =
            2.0F * std::numeric_limits<float>::epsilon();
        const float response = rope_realtime_compliance_response(
            input.structural_compliance, input.effective_inverse_mass);
        if (!std::isfinite(response) || response < 0.05F - tolerance ||
            response > 0.995F + tolerance)
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}

struct RopeRealtimeWorkBudget {
    std::uint32_t live_nodes{};
    std::uint32_t node_capacity{};
    std::uint32_t segment_count{};
    std::uint32_t bend_constraint_count{};
    std::uint32_t contact_constraint_count{};
    std::uint32_t constraint_sweeps{};
    std::uint32_t contact_passes{};
    std::uint32_t time_advances{};
    std::uint32_t spatial_updates_during_solve{};
    std::uint32_t domain_dispatches{};
    std::uint64_t domain_workgroups{};
    std::uint64_t stretch_constraint_visits{};
    std::uint64_t bend_constraint_visits{};
    std::uint64_t contact_visits{};
    std::size_t solver_scratch_bytes{};
};

[[nodiscard]] constexpr RopeRealtimeWorkBudget rope_realtime_work_budget(
    std::uint32_t live_nodes, std::uint32_t node_capacity,
    std::uint32_t contact_constraint_count) noexcept {
    RopeRealtimeWorkBudget result{};
    result.live_nodes = live_nodes;
    result.node_capacity = node_capacity;
    result.segment_count = live_nodes > 0U ? live_nodes - 1U : 0U;
    result.bend_constraint_count = live_nodes > 1U ? live_nodes - 2U : 0U;
    result.contact_constraint_count = contact_constraint_count;
    result.constraint_sweeps = rope_realtime_constraint_sweeps;
    result.contact_passes = rope_realtime_contact_passes;
    result.time_advances = 1U;
    result.spatial_updates_during_solve = 0U;
    // Predict + eight (constraint proposal, apply) + four contact passes +
    // final self-contact reconciliation + velocity commit.
    result.domain_dispatches = 3U + 2U * rope_realtime_constraint_sweeps +
        rope_realtime_contact_passes;
    const std::uint64_t node_groups =
        (std::uint64_t{live_nodes} + 127U) / 128U;
    // The eight chain-wide tridiagonal solves each use one cooperative group.
    // Predictor, delta apply, contact and commit remain node-parallel.
    result.domain_workgroups =
        1U + rope_realtime_constraint_sweeps + node_groups *
            (2U + rope_realtime_constraint_sweeps +
             rope_realtime_contact_passes);
    result.stretch_constraint_visits =
        std::uint64_t{result.segment_count} *
        rope_realtime_constraint_sweeps;
    result.bend_constraint_visits =
        2U * std::uint64_t{result.bend_constraint_count} *
        rope_realtime_constraint_sweeps;
    result.contact_visits = std::uint64_t{contact_constraint_count} *
        rope_realtime_contact_passes;
    // Previous position, proposal and scalar correction consumption, the
    // one-advance counter/status, and one fixed 1024-row cooperative-group
    // tridiagonal workspace (three scalars plus one direction per row).
    result.solver_scratch_bytes = static_cast<std::size_t>(node_capacity) *
            (2U * 3U * sizeof(float) + sizeof(float)) +
        2U * sizeof(std::uint32_t) +
        1024U * (6U * sizeof(float));
    return result;
}

// Pins topology and the one F8-10 publication across eight algebraic sweeps.
// Contact runs on sweeps 2/4/6/8, consuming exactly four collision passes.
class RopeRealtimeTickContract {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        const RopeRealtimeWorkBudget &budget,
        std::uint64_t topology_revision) noexcept {
        if (active_ || topology_revision == 0U ||
            budget.constraint_sweeps != rope_realtime_constraint_sweeps ||
            budget.contact_passes != rope_realtime_contact_passes ||
            budget.time_advances != 1U ||
            budget.spatial_updates_during_solve != 0U ||
            budget.domain_dispatches != 23U)
            return PM_ERROR_INVALID_STATE;
        const auto result = spatial_.begin(
            token, counters, rope_realtime_contact_passes);
        if (result != PM_SUCCESS) return result;
        topology_revision_ = topology_revision;
        advances_ = 0U;
        sweeps_ = 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_advance(
        float timestep, std::uint32_t substeps) noexcept {
        if (!active_ || advances_ != 0U ||
            !rope_realtime_step_request_valid(timestep, substeps))
            return PM_ERROR_INVALID_STATE;
        ++advances_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_sweep(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        std::uint64_t topology_revision, bool contact_pass) noexcept {
        const bool expected_contact = (sweeps_ & 1U) != 0U;
        if (!active_ || advances_ != 1U ||
            sweeps_ >= rope_realtime_constraint_sweeps ||
            topology_revision != topology_revision_ ||
            contact_pass != expected_contact)
            return PM_ERROR_INVALID_STATE;
        if (contact_pass) {
            const auto result = spatial_.record_pass(token, counters);
            if (result != PM_SUCCESS) return result;
        }
        ++sweeps_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult finish() noexcept {
        if (!active_ || advances_ != 1U ||
            sweeps_ != rope_realtime_constraint_sweeps)
            return PM_ERROR_INVALID_STATE;
        const auto result = spatial_.finish();
        if (result != PM_SUCCESS) return result;
        active_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t advances() const noexcept {
        return advances_;
    }
    [[nodiscard]] constexpr std::uint32_t sweeps() const noexcept {
        return sweeps_;
    }
    [[nodiscard]] constexpr std::uint32_t contact_passes() const noexcept {
        return spatial_.completed_passes();
    }

private:
    DeformableSpatialSolveEpoch spatial_{};
    std::uint64_t topology_revision_{};
    std::uint32_t advances_{};
    std::uint32_t sweeps_{};
    bool active_{};
};

} // namespace parallel_mater::simulation
