// SPDX-License-Identifier: MIT
#pragma once

#include "fluid_spatial.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t fluid_realtime_profile_version = 1U;
inline constexpr std::uint32_t fluid_realtime_density_sweeps = 6U;
inline constexpr float fluid_realtime_timestep = 1.0F / 60.0F;
inline constexpr float fluid_realtime_solver_damping = 0.05F;
inline constexpr float fluid_realtime_maximum_correction_fraction = 0.25F;
inline constexpr std::uint32_t fluid_realtime_status_invalid_profile =
    1U << 13U;
inline constexpr std::uint32_t fluid_realtime_status_invalid_timestep =
    1U << 14U;
inline constexpr std::uint32_t fluid_realtime_status_invalid_material =
    1U << 15U;
inline constexpr std::uint32_t fluid_realtime_status_nonfinite_state =
    1U << 16U;

[[nodiscard]] inline bool fluid_realtime_step_request_valid(
    float timestep, std::uint32_t substeps) noexcept {
    const float lower = std::nextafter(fluid_realtime_timestep, 0.0F);
    const float upper = std::nextafter(
        fluid_realtime_timestep, std::numeric_limits<float>::infinity());
    return std::isfinite(timestep) && timestep >= lower &&
        timestep <= upper && substeps == 1U;
}

// Host-side validation for the profile-specific interpretation of FluidOptions.
// repulsion is an inverse-second density response rate; viscosity,
// normal_damping and velocity_damping remain inverse-second rates.
struct FluidRealtimeInput {
    float world_coordinate{};
    float support_radius{};
    float particle_radius{};
    float rest_density{};
    float rest_particle_volume{};
    float maximum_speed{};
    float acceleration_displacement{};
    float solver_correction{};
    float repulsion{};
    float viscosity{};
    float normal_damping{};
    float velocity_damping{};
    float solver_damping{fluid_realtime_solver_damping};
};

[[nodiscard]] inline float fluid_realtime_density_response(
    float repulsion, float timestep = fluid_realtime_timestep) noexcept {
    if (!std::isfinite(repulsion) || !std::isfinite(timestep) ||
        repulsion < 0.0F || timestep <= 0.0F)
        return std::numeric_limits<float>::quiet_NaN();
    return repulsion == 0.0F ? 0.0F :
        1.0F - std::exp(-repulsion * timestep);
}

[[nodiscard]] inline PmResult validate_fluid_realtime_input(
    const FluidRealtimeInput &input) noexcept {
    const float values[]{input.world_coordinate, input.support_radius,
        input.particle_radius, input.rest_density,
        input.rest_particle_volume, input.maximum_speed,
        input.acceleration_displacement, input.solver_correction,
        input.repulsion, input.viscosity, input.normal_damping,
        input.velocity_damping, input.solver_damping};
    for (const float value : values)
        if (!std::isfinite(value)) return PM_ERROR_INVALID_ARGUMENT;
    if (std::abs(input.world_coordinate) > 10'000.0F ||
        input.support_radius < 0.005F || input.support_radius > 2.0F ||
        input.particle_radius <= 0.0F ||
        input.particle_radius > 0.5F * input.support_radius ||
        input.rest_density <= 0.0F || input.rest_particle_volume <= 0.0F ||
        input.maximum_speed < 0.0F || input.maximum_speed > 20.0F ||
        input.maximum_speed * fluid_realtime_timestep >
            0.5F * input.support_radius ||
        input.acceleration_displacement < 0.0F ||
        input.acceleration_displacement >
            0.25F * input.support_radius ||
        input.solver_correction < 0.0F ||
        input.solver_correction >
            fluid_realtime_maximum_correction_fraction *
                input.support_radius ||
        input.repulsion < 0.0F || input.viscosity < 0.0F ||
        input.normal_damping < 0.0F || input.velocity_damping < 0.0F ||
        input.solver_damping < 0.0F || input.solver_damping > 0.12F)
        return PM_ERROR_INVALID_ARGUMENT;
    const float response = fluid_realtime_density_response(input.repulsion);
    constexpr float tolerance = 2.0F * std::numeric_limits<float>::epsilon();
    if (input.repulsion != 0.0F &&
        (!std::isfinite(response) || response < 0.05F - tolerance ||
         response > 0.995F + tolerance))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}

struct FluidRealtimeWorkBudget {
    std::uint32_t live_particles{};
    std::uint32_t particle_capacity{};
    std::uint32_t candidates_per_query{};
    std::uint32_t density_sweeps{};
    std::uint32_t time_advances{};
    std::uint32_t spatial_updates_during_solve{};
    std::uint32_t domain_dispatches{};
    std::uint64_t domain_workgroups{};
    std::uint64_t candidate_visits{};
    std::size_t solver_scratch_bytes{};
};

[[nodiscard]] constexpr FluidRealtimeWorkBudget fluid_realtime_work_budget(
    std::uint32_t live_particles, std::uint32_t particle_capacity,
    std::uint32_t candidates_per_query) noexcept {
    FluidRealtimeWorkBudget result{};
    result.live_particles = live_particles;
    result.particle_capacity = particle_capacity;
    result.candidates_per_query = candidates_per_query;
    result.density_sweeps = fluid_realtime_density_sweeps;
    result.time_advances = 1U;
    // predict + six (density/lambda, correction, apply) + viscosity + commit.
    result.domain_dispatches = 3U * fluid_realtime_density_sweeps + 3U;
    const std::uint64_t groups =
        (std::uint64_t{live_particles} + 127U) / 128U;
    result.domain_workgroups = groups * result.domain_dispatches;
    // Density/lambda and correction visit candidates in each sweep; viscosity
    // visits them once after the final positions are available.
    result.candidate_visits = std::uint64_t{live_particles} *
        candidates_per_query * (2U * fluid_realtime_density_sweeps + 1U);
    // density, lambda, position delta, accumulated correction, velocity delta.
    result.solver_scratch_bytes =
        static_cast<std::size_t>(particle_capacity) *
            (2U * sizeof(float) + 3U * 3U * sizeof(float)) +
        2U * sizeof(std::uint32_t);
    return result;
}

// Pins the single published fluid epoch while exactly one predictor and six
// algebraic sweeps execute. A sweep cannot advance time or mutate the index.
class FluidRealtimeTickContract {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        const FluidRealtimeWorkBudget &budget) noexcept {
        if (active_ || budget.density_sweeps !=
                fluid_realtime_density_sweeps ||
            budget.time_advances != 1U ||
            budget.spatial_updates_during_solve != 0U ||
            budget.domain_dispatches !=
                3U * fluid_realtime_density_sweeps + 3U)
            return PM_ERROR_INVALID_STATE;
        const auto result = spatial_.begin(
            token, counters, fluid_realtime_density_sweeps);
        if (result != PM_SUCCESS) return result;
        advances_ = 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_advance(
        float timestep, std::uint32_t substeps) noexcept {
        if (!active_ || advances_ != 0U ||
            !fluid_realtime_step_request_valid(timestep, substeps))
            return PM_ERROR_INVALID_STATE;
        ++advances_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_sweep(
        const SpatialEpochToken &token,
        const SpatialEpochCounters &counters) noexcept {
        if (!active_ || advances_ != 1U)
            return PM_ERROR_INVALID_STATE;
        return spatial_.record_sweep(token, counters);
    }

    [[nodiscard]] PmResult finish() noexcept {
        if (!active_ || advances_ != 1U) return PM_ERROR_INVALID_STATE;
        const auto result = spatial_.finish();
        if (result != PM_SUCCESS) return result;
        active_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t advances() const noexcept {
        return advances_;
    }

    [[nodiscard]] constexpr std::uint32_t sweeps() const noexcept {
        return spatial_.completed_sweeps();
    }

private:
    FluidSpatialSolveEpoch spatial_{};
    std::uint32_t advances_{};
    bool active_{};
};

} // namespace parallel_mater::simulation
