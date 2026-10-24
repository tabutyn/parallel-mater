// SPDX-License-Identifier: MIT
#pragma once

#include "deformable_spatial.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t soft_body_realtime_profile_version = 1U;
inline constexpr std::uint32_t soft_body_realtime_constraint_sweeps = 8U;
inline constexpr std::uint32_t soft_body_realtime_contact_passes = 4U;
inline constexpr float soft_body_realtime_timestep = 1.0F / 60.0F;
inline constexpr float soft_body_realtime_solver_damping = 0.05F;
inline constexpr float soft_body_realtime_maximum_correction_fraction = 0.25F;
inline constexpr std::uint32_t soft_body_realtime_status_invalid_profile =
    1U << 17U;
inline constexpr std::uint32_t soft_body_realtime_status_invalid_timestep =
    1U << 18U;
inline constexpr std::uint32_t soft_body_realtime_status_invalid_material =
    1U << 19U;
inline constexpr std::uint32_t soft_body_realtime_status_nonfinite_state =
    1U << 20U;
inline constexpr std::uint32_t soft_body_realtime_status_unresolved_contact =
    1U << 21U;
inline constexpr std::uint32_t soft_body_realtime_status_inverted =
    1U << 22U;

static_assert(soft_body_realtime_contact_passes ==
              deformable_spatial_collision_passes);

[[nodiscard]] inline bool soft_body_realtime_step_request_valid(
    float timestep, std::uint32_t substeps) noexcept {
    const float lower = std::nextafter(soft_body_realtime_timestep, 0.0F);
    const float upper = std::nextafter(
        soft_body_realtime_timestep, std::numeric_limits<float>::infinity());
    return std::isfinite(timestep) && timestep >= lower &&
        timestep <= upper && substeps == 1U;
}

[[nodiscard]] inline float soft_body_realtime_compliance_response(
    float compliance, float effective_inverse_mass,
    float timestep = soft_body_realtime_timestep) noexcept {
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

// Version 1 accepts only the envelope qualified by F8-14. Shape matching is
// the volume-retention model exposed by the existing soft-body public API;
// there is no hidden tetrahedral-volume material parameter.
struct SoftBodyRealtimeInput {
    float world_coordinate{};
    float characteristic_length{};
    float mass{};
    float interacting_mass_ratio{1.0F};
    float linear_speed{};
    float acceleration_displacement{};
    float solver_correction{};
    float authored_strain{};
    float stretch_compliance{};
    float effective_inverse_mass{};
    float node_radius{};
    float velocity_damping{};
    float spring_damping{};
    float contact_friction{};
    float shape_matching_stiffness{};
    float rest_shape_determinant{1.0F};
    float solver_damping{soft_body_realtime_solver_damping};
};

[[nodiscard]] inline PmResult validate_soft_body_realtime_input(
    const SoftBodyRealtimeInput &input) noexcept {
    const float values[]{input.world_coordinate,
        input.characteristic_length, input.mass,
        input.interacting_mass_ratio, input.linear_speed,
        input.acceleration_displacement, input.solver_correction,
        input.authored_strain, input.stretch_compliance,
        input.effective_inverse_mass, input.node_radius,
        input.velocity_damping, input.spring_damping,
        input.contact_friction, input.shape_matching_stiffness,
        input.rest_shape_determinant, input.solver_damping};
    for (const float value : values)
        if (!std::isfinite(value)) return PM_ERROR_INVALID_ARGUMENT;
    if (std::abs(input.world_coordinate) > 10'000.0F ||
        input.characteristic_length < 0.005F ||
        input.characteristic_length > 2.0F || input.mass < 0.001F ||
        input.mass > 10'000.0F || input.interacting_mass_ratio < 1.0F ||
        input.interacting_mass_ratio > 100.0F || input.linear_speed < 0.0F ||
        input.linear_speed > 20.0F ||
        input.linear_speed * soft_body_realtime_timestep >
            0.5F * input.characteristic_length ||
        input.acceleration_displacement < 0.0F ||
        input.acceleration_displacement >
            0.25F * input.characteristic_length ||
        input.solver_correction < 0.0F ||
        input.solver_correction >
            soft_body_realtime_maximum_correction_fraction *
                input.characteristic_length ||
        input.authored_strain < 0.0F || input.authored_strain > 0.25F ||
        input.stretch_compliance < 0.0F ||
        input.effective_inverse_mass <= 0.0F || input.node_radius < 0.0F ||
        input.node_radius > 0.5F * input.characteristic_length ||
        input.velocity_damping < 0.0F || input.spring_damping != 0.0F ||
        input.contact_friction < 0.0F || input.contact_friction > 2.0F ||
        (input.shape_matching_stiffness != 0.0F &&
         (input.shape_matching_stiffness < 0.05F ||
          input.shape_matching_stiffness > 0.995F)) ||
        input.rest_shape_determinant <= 1.0e-8F ||
        input.solver_damping < 0.0F || input.solver_damping > 0.12F)
        return PM_ERROR_INVALID_ARGUMENT;
    if (input.stretch_compliance != 0.0F) {
        constexpr float tolerance =
            2.0F * std::numeric_limits<float>::epsilon();
        const float response = soft_body_realtime_compliance_response(
            input.stretch_compliance, input.effective_inverse_mass);
        if (!std::isfinite(response) || response < 0.05F - tolerance ||
            response > 0.995F + tolerance)
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}

struct SoftBodyRealtimeWorkBudget {
    std::uint32_t live_nodes{};
    std::uint32_t node_capacity{};
    std::uint32_t directed_bond_count{};
    std::uint32_t contact_constraint_count{};
    std::uint32_t constraint_sweeps{};
    std::uint32_t contact_passes{};
    std::uint32_t time_advances{};
    std::uint32_t spatial_updates_during_solve{};
    std::uint32_t domain_dispatches{};
    std::uint64_t domain_workgroups{};
    std::uint64_t bond_visits{};
    std::uint64_t contact_visits{};
    std::size_t solver_scratch_bytes{};
};

[[nodiscard]] constexpr SoftBodyRealtimeWorkBudget
soft_body_realtime_work_budget(std::uint32_t live_nodes,
    std::uint32_t node_capacity, std::uint32_t directed_bond_count,
    std::uint32_t contact_constraint_count) noexcept {
    SoftBodyRealtimeWorkBudget result{};
    result.live_nodes = live_nodes;
    result.node_capacity = node_capacity;
    result.directed_bond_count = directed_bond_count;
    result.contact_constraint_count = contact_constraint_count;
    result.constraint_sweeps = soft_body_realtime_constraint_sweeps;
    result.contact_passes = soft_body_realtime_contact_passes;
    result.time_advances = 1U;
    // predict + eight (links, apply, rest-shape) + four contacts + commit.
    result.domain_dispatches = 1U + 3U * soft_body_realtime_constraint_sweeps +
        soft_body_realtime_contact_passes + 1U;
    const std::uint64_t node_groups =
        (std::uint64_t{live_nodes} + 127U) / 128U;
    result.domain_workgroups = node_groups *
            (1U + 2U * soft_body_realtime_constraint_sweeps +
             soft_body_realtime_contact_passes + 1U) +
        soft_body_realtime_constraint_sweeps;
    result.bond_visits = std::uint64_t{directed_bond_count} *
        soft_body_realtime_constraint_sweeps;
    result.contact_visits = std::uint64_t{contact_constraint_count} *
        soft_body_realtime_contact_passes;
    // Previous position, Jacobi delta, scalar correction consumption,
    // orientation, advance count and status.
    result.solver_scratch_bytes = static_cast<std::size_t>(node_capacity) *
            (2U * 3U * sizeof(float) + sizeof(float)) +
        4U * sizeof(float) + 2U * sizeof(std::uint32_t);
    return result;
}

// Pins topology and one F8-10 publication across eight algebraic sweeps.
// Contact runs on sweeps 2/4/6/8, consuming exactly four collision passes.
class SoftBodyRealtimeTickContract {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        const SoftBodyRealtimeWorkBudget &budget,
        std::uint64_t topology_revision) noexcept {
        if (active_ || topology_revision == 0U ||
            budget.constraint_sweeps != soft_body_realtime_constraint_sweeps ||
            budget.contact_passes != soft_body_realtime_contact_passes ||
            budget.time_advances != 1U ||
            budget.spatial_updates_during_solve != 0U ||
            budget.domain_dispatches != 30U)
            return PM_ERROR_INVALID_STATE;
        const auto result = spatial_.begin(
            token, counters, soft_body_realtime_contact_passes);
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
            !soft_body_realtime_step_request_valid(timestep, substeps))
            return PM_ERROR_INVALID_STATE;
        ++advances_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_sweep(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        std::uint64_t topology_revision, bool contact_pass) noexcept {
        const bool expected_contact = (sweeps_ & 1U) != 0U;
        if (!active_ || advances_ != 1U ||
            sweeps_ >= soft_body_realtime_constraint_sweeps ||
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
            sweeps_ != soft_body_realtime_constraint_sweeps)
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
