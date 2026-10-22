// SPDX-License-Identifier: MIT
#pragma once

#include "geometry_storage.hpp"
#include "rigid_spatial.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t rigid_realtime_profile_version = 1U;
inline constexpr std::uint32_t rigid_realtime_sweeps = 8U;
inline constexpr float rigid_realtime_timestep = 1.0F / 60.0F;
inline constexpr float rigid_realtime_solver_damping = 0.05F;

[[nodiscard]] inline bool rigid_realtime_step_request_valid(
    float timestep, std::uint32_t substeps) noexcept {
    const float lower = std::nextafter(rigid_realtime_timestep, 0.0F);
    const float upper = std::nextafter(rigid_realtime_timestep,
        std::numeric_limits<float>::infinity());
    return std::isfinite(timestep) && timestep >= lower &&
        timestep <= upper && substeps == 1U;
}

struct RigidRealtimeInput {
    float world_coordinate{};
    float characteristic_length{};
    float mass{};
    float interacting_mass_ratio{1.0F};
    float linear_speed{};
    float angular_speed{};
    float acceleration_displacement{};
    float solver_correction{};
    float friction{};
    float restitution{};
    float compliance{};
    float effective_inverse_mass{};
    float solver_damping{rigid_realtime_solver_damping};
};

[[nodiscard]] inline float rigid_realtime_compliance_response(
    float compliance, float effective_inverse_mass,
    float timestep = rigid_realtime_timestep) noexcept {
    if (!std::isfinite(compliance) || !std::isfinite(effective_inverse_mass) ||
        !std::isfinite(timestep) || compliance < 0.0F ||
        effective_inverse_mass <= 0.0F || timestep <= 0.0F)
        return std::numeric_limits<float>::quiet_NaN();
    if (compliance == 0.0F) return 1.0F;
    const double ratio = static_cast<double>(compliance) /
        (static_cast<double>(timestep) * timestep *
         effective_inverse_mass);
    return static_cast<float>(1.0 / (1.0 + ratio));
}

[[nodiscard]] inline PmResult validate_rigid_realtime_input(
    const RigidRealtimeInput &input) noexcept {
    const float values[]{input.world_coordinate, input.characteristic_length,
        input.mass, input.interacting_mass_ratio, input.linear_speed,
        input.angular_speed, input.acceleration_displacement,
        input.solver_correction, input.friction, input.restitution,
        input.compliance, input.effective_inverse_mass,
        input.solver_damping};
    for (const float value : values)
        if (!std::isfinite(value)) return PM_ERROR_INVALID_ARGUMENT;
    constexpr float pi = 3.14159265358979323846F;
    if (std::abs(input.world_coordinate) > 10'000.0F ||
        input.characteristic_length < 0.005F ||
        input.characteristic_length > 2.0F || input.mass < 0.001F ||
        input.mass > 10'000.0F || input.interacting_mass_ratio < 1.0F ||
        input.interacting_mass_ratio > 100.0F || input.linear_speed < 0.0F ||
        input.linear_speed > 20.0F ||
        input.linear_speed * rigid_realtime_timestep >
            0.5F * input.characteristic_length ||
        input.angular_speed < 0.0F || input.angular_speed > 10.0F * pi ||
        input.angular_speed * rigid_realtime_timestep > pi / 6.0F ||
        input.acceleration_displacement < 0.0F ||
        input.acceleration_displacement >
            0.25F * input.characteristic_length ||
        input.solver_correction < 0.0F ||
        input.solver_correction > 0.25F * input.characteristic_length ||
        input.friction < 0.0F || input.friction > 2.0F ||
        input.restitution < 0.0F || input.restitution > 0.8F ||
        input.compliance < 0.0F || input.effective_inverse_mass <= 0.0F ||
        input.solver_damping < 0.0F || input.solver_damping > 0.12F)
        return PM_ERROR_INVALID_ARGUMENT;
    if (input.compliance != 0.0F) {
        const float response = rigid_realtime_compliance_response(
            input.compliance, input.effective_inverse_mass);
        constexpr float response_tolerance = 2.0F *
            std::numeric_limits<float>::epsilon();
        if (!std::isfinite(response) ||
            response < 0.05F - response_tolerance ||
            response > 0.995F + response_tolerance)
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}

struct RigidRealtimeWorkBudget {
    std::uint32_t body_count{};
    std::uint32_t contact_count{};
    std::uint32_t joint_count{};
    std::uint32_t color_count{};
    std::uint32_t algebraic_sweeps{};
    std::uint32_t time_advances{};
    std::uint32_t spatial_updates_during_solve{};
    std::uint32_t cooperative_dispatches{};
    std::uint32_t staged_dispatches{};
    std::uint64_t staged_workgroups{};
    std::size_t persistent_bytes{};
};

[[nodiscard]] constexpr RigidRealtimeWorkBudget rigid_realtime_work_budget(
    std::uint32_t body_count, std::uint32_t contact_count,
    std::uint32_t joint_count, std::uint32_t color_count,
    std::uint32_t grid_blocks) noexcept {
    RigidRealtimeWorkBudget result{};
    result.body_count = body_count;
    result.contact_count = contact_count;
    result.joint_count = joint_count;
    result.color_count = color_count;
    result.algebraic_sweeps = rigid_realtime_sweeps;
    result.time_advances = 1U;
    result.cooperative_dispatches = body_count == 0U ? 0U : 2U;
    // prepare + eight (one dispatch per color + dual advance) + six final
    // pose stages. Realtime v1 has no second impact phase.
    result.staged_dispatches = body_count == 0U ? 0U :
        1U + rigid_realtime_sweeps * (color_count + 1U) + 6U;
    result.staged_workgroups = static_cast<std::uint64_t>(
        result.staged_dispatches) * grid_blocks;
    result.persistent_bytes =
        static_cast<std::size_t>(body_count) * sizeof(GeometryConstraintBody) +
        static_cast<std::size_t>(contact_count) * sizeof(ContactManifold) +
        static_cast<std::size_t>(joint_count) * sizeof(RigidConstraintResource) +
        sizeof(ContactSchedule);
    return result;
}

// Pins the one spatial publication while all eight rigid sweeps run. The
// domain solve may not advance time or rebuild/refit the shared hierarchy.
class RigidRealtimeTickContract {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        const RigidRealtimeWorkBudget &budget) noexcept {
        if (active_ || token.tick == 0U ||
            budget.algebraic_sweeps != rigid_realtime_sweeps ||
            budget.time_advances != 1U ||
            budget.spatial_updates_during_solve != 0U ||
            counters.publications != 1U ||
            counters.builds + counters.refits > 1U ||
            counters.overflows != 0U)
            return PM_ERROR_INVALID_STATE;
        token_ = token;
        counters_ = counters;
        sweeps_ = 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_sweep(
        const SpatialEpochToken &token,
        const SpatialEpochCounters &counters) noexcept {
        if (!active_ || token != token_ || counters.builds != counters_.builds ||
            counters.refits != counters_.refits ||
            counters.publications != counters_.publications ||
            counters.overflows != 0U || sweeps_ >= rigid_realtime_sweeps)
            return PM_ERROR_INVALID_STATE;
        ++sweeps_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult finish() noexcept {
        if (!active_ || sweeps_ != rigid_realtime_sweeps)
            return PM_ERROR_INVALID_STATE;
        active_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t sweeps() const noexcept {
        return sweeps_;
    }

private:
    SpatialEpochToken token_{};
    SpatialEpochCounters counters_{};
    std::uint32_t sweeps_{};
    bool active_{};
};

} // namespace parallel_mater::simulation
