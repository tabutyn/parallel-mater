// SPDX-License-Identifier: MIT
#pragma once

#include "deformable_spatial.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t smoke_realtime_profile_version = 1U;
inline constexpr std::uint32_t smoke_realtime_particle_sweeps = 6U;
inline constexpr std::uint32_t smoke_realtime_grid_pressure_cycles = 12U;
inline constexpr float smoke_realtime_timestep = 1.0F / 60.0F;
inline constexpr float smoke_realtime_solver_damping = 0.05F;
inline constexpr float smoke_realtime_maximum_correction_fraction = 0.25F;
inline constexpr std::uint32_t smoke_realtime_status_invalid_profile =
    1U << 17U;
inline constexpr std::uint32_t smoke_realtime_status_invalid_timestep =
    1U << 18U;
inline constexpr std::uint32_t smoke_realtime_status_invalid_material =
    1U << 19U;
inline constexpr std::uint32_t smoke_realtime_status_nonfinite_state =
    1U << 20U;

enum class SmokeRealtimeRepresentation : std::uint32_t {
    particles = 0U,
    grid = 1U,
};

[[nodiscard]] inline bool smoke_realtime_step_request_valid(
    float timestep, std::uint32_t substeps) noexcept {
    const float lower = std::nextafter(smoke_realtime_timestep, 0.0F);
    const float upper = std::nextafter(
        smoke_realtime_timestep, std::numeric_limits<float>::infinity());
    return std::isfinite(timestep) && timestep >= lower &&
        timestep <= upper && substeps == 1U;
}

// Profile-specific interpretation of SmokeOptions. Rates are inverse seconds.
// Particle characteristic length is the support radius (3 * radius); grid
// characteristic length is one cell. Grid viscosity is kinematic, in m^2/s.
struct SmokeRealtimeInput {
    SmokeRealtimeRepresentation representation{};
    float world_coordinate{};
    float characteristic_length{};
    float maximum_speed{};
    float acceleration_displacement{};
    float solver_correction{};
    float response{};
    float pressure_stiffness{};
    float viscosity{};
    float solver_damping{smoke_realtime_solver_damping};
    std::uint32_t grid_resolution{};
    std::uint32_t grid_vertical_resolution{};
};

[[nodiscard]] inline float smoke_realtime_rate_response(
    float rate, float timestep = smoke_realtime_timestep) noexcept {
    if (!std::isfinite(rate) || !std::isfinite(timestep) || rate < 0.0F ||
        timestep <= 0.0F)
        return std::numeric_limits<float>::quiet_NaN();
    return rate == 0.0F ? 0.0F : 1.0F - std::exp(-rate * timestep);
}

[[nodiscard]] inline float smoke_realtime_pressure_response(
    float pressure_stiffness) noexcept {
    if (!std::isfinite(pressure_stiffness) || pressure_stiffness < 0.0F)
        return std::numeric_limits<float>::quiet_NaN();
    return pressure_stiffness == 0.0F ? 0.0F :
        1.0F - std::exp(-pressure_stiffness);
}

[[nodiscard]] inline PmResult validate_smoke_realtime_input(
    const SmokeRealtimeInput &input) noexcept {
    const float values[]{input.world_coordinate, input.characteristic_length,
        input.maximum_speed, input.acceleration_displacement,
        input.solver_correction, input.response, input.pressure_stiffness,
        input.viscosity, input.solver_damping};
    for (const float value : values)
        if (!std::isfinite(value)) return PM_ERROR_INVALID_ARGUMENT;
    if (std::abs(input.world_coordinate) > 10'000.0F ||
        input.characteristic_length < 0.005F ||
        input.characteristic_length > 2.0F || input.maximum_speed < 0.0F ||
        input.maximum_speed > 20.0F ||
        input.acceleration_displacement < 0.0F ||
        input.acceleration_displacement >
            0.25F * input.characteristic_length ||
        input.solver_correction < 0.0F ||
        input.solver_correction >
            smoke_realtime_maximum_correction_fraction *
                input.characteristic_length ||
        input.response < 0.0F || input.pressure_stiffness < 0.0F ||
        input.viscosity < 0.0F || input.solver_damping < 0.0F ||
        input.solver_damping > 0.12F)
        return PM_ERROR_INVALID_ARGUMENT;
    if (input.representation == SmokeRealtimeRepresentation::particles) {
        if (input.grid_resolution != 0U ||
            input.maximum_speed * smoke_realtime_timestep >
                0.5F * input.characteristic_length)
            return PM_ERROR_INVALID_ARGUMENT;
    } else if (input.representation == SmokeRealtimeRepresentation::grid) {
        if (input.grid_resolution < 4U ||
            input.grid_vertical_resolution < 4U ||
            input.maximum_speed * smoke_realtime_timestep >
                input.characteristic_length)
            return PM_ERROR_INVALID_ARGUMENT;
    } else {
        return PM_ERROR_INVALID_ARGUMENT;
    }
    constexpr float tolerance = 2.0F * std::numeric_limits<float>::epsilon();
    const float pressure_response =
        smoke_realtime_pressure_response(input.pressure_stiffness);
    if (input.pressure_stiffness != 0.0F &&
        (!std::isfinite(pressure_response) ||
         pressure_response < 0.05F - tolerance ||
         pressure_response > 0.995F + tolerance))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}

struct SmokeRealtimeWorkBudget {
    SmokeRealtimeRepresentation representation{};
    std::uint32_t live_particles{};
    std::uint32_t particle_capacity{};
    std::uint32_t candidates_per_particle{};
    std::uint32_t grid_cells{};
    std::uint32_t grid_faces{};
    std::uint32_t algebraic_iterations{};
    std::uint32_t spatial_passes{};
    std::uint32_t time_advances{};
    std::uint32_t spatial_updates_during_solve{};
    std::uint32_t domain_dispatches{};
    std::uint64_t domain_workgroups{};
    std::uint64_t candidate_visits{};
    std::uint64_t cell_visits{};
    std::size_t solver_scratch_bytes{};
};

[[nodiscard]] constexpr SmokeRealtimeWorkBudget
smoke_realtime_particle_work_budget(std::uint32_t live_particles,
    std::uint32_t particle_capacity,
    std::uint32_t candidates_per_particle) noexcept {
    SmokeRealtimeWorkBudget result{};
    result.representation = SmokeRealtimeRepresentation::particles;
    result.live_particles = live_particles;
    result.particle_capacity = particle_capacity;
    result.candidates_per_particle = candidates_per_particle;
    result.algebraic_iterations = smoke_realtime_particle_sweeps;
    result.spatial_passes = smoke_realtime_particle_sweeps;
    result.time_advances = 1U;
    // source + predict + six (density, correction, apply) + commit.
    result.domain_dispatches = 3U * smoke_realtime_particle_sweeps + 3U;
    const std::uint64_t groups =
        (std::uint64_t{live_particles} + 127U) / 128U;
    result.domain_workgroups = groups * result.domain_dispatches;
    result.candidate_visits = std::uint64_t{live_particles} *
        candidates_per_particle * 2U * smoke_realtime_particle_sweeps;
    // previous position, density, pressure, Jacobi delta, accumulated delta,
    // one advance counter and one status word.
    result.solver_scratch_bytes = static_cast<std::size_t>(particle_capacity) *
            (3U * 3U * sizeof(float) + 2U * sizeof(float)) +
        2U * sizeof(std::uint32_t);
    return result;
}

[[nodiscard]] constexpr SmokeRealtimeWorkBudget
smoke_realtime_grid_work_budget(std::uint32_t live_particles,
    std::uint32_t particle_capacity, std::uint32_t n,
    std::uint32_t height) noexcept {
    SmokeRealtimeWorkBudget result{};
    result.representation = SmokeRealtimeRepresentation::grid;
    result.live_particles = live_particles;
    result.particle_capacity = particle_capacity;
    result.grid_cells = n * height * n;
    result.grid_faces = (n + 1U) * height * n +
        n * (height + 1U) * n + n * height * (n + 1U);
    result.algebraic_iterations = smoke_realtime_grid_pressure_cycles;
    result.spatial_passes = 1U;
    result.time_advances = 1U;
    // begin, bounded source injection, scalar advection (density/temperature),
    // face advection/force/boundary, divergence, pressure clear, twelve fixed
    // Jacobi cycles, projection and tracer commit.
    result.domain_dispatches = 30U;
    const std::uint64_t particle_groups =
        (std::uint64_t{live_particles} + 127U) / 128U;
    const std::uint64_t cell_groups =
        (std::uint64_t{result.grid_cells} + 127U) / 128U;
    const std::uint64_t face_groups =
        (std::uint64_t{result.grid_faces} + 127U) / 128U;
    result.domain_workgroups = particle_groups +
        (1U + 1U + 2U + 1U + smoke_realtime_grid_pressure_cycles) *
            cell_groups +
        4U * face_groups;
    result.cell_visits = std::uint64_t{result.grid_cells} *
        (5U + smoke_realtime_grid_pressure_cycles) +
        4U * result.grid_faces + live_particles;
    // Pressure ping/pong, divergence, two scalar advection destinations and
    // one complete staggered-face advection destination.
    // Source-rate fields are immutable inputs, not solver scratch.
    // per-particle previous position, advance counter and status.
    result.solver_scratch_bytes = static_cast<std::size_t>(result.grid_cells) *
            5U * sizeof(float) +
        static_cast<std::size_t>(result.grid_faces) * sizeof(float) +
        static_cast<std::size_t>(particle_capacity) * 3U * sizeof(float) +
        2U * sizeof(std::uint32_t);
    return result;
}

// Pins the one F8-10 publication while the representation's exact fixed
// algebraic schedule runs. No pass may rebuild, refit or republish the index.
class SmokeRealtimeTickContract {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        const SmokeRealtimeWorkBudget &budget) noexcept {
        const bool particle = budget.representation ==
            SmokeRealtimeRepresentation::particles;
        const bool grid = budget.representation ==
            SmokeRealtimeRepresentation::grid;
        const std::uint32_t expected_iterations = particle
            ? smoke_realtime_particle_sweeps
            : smoke_realtime_grid_pressure_cycles;
        const std::uint32_t expected_dispatches = particle ? 21U : 30U;
        const std::uint32_t expected_passes = particle ? 6U : 1U;
        if (active_ || (!particle && !grid) ||
            budget.algebraic_iterations != expected_iterations ||
            budget.spatial_passes != expected_passes ||
            budget.time_advances != 1U ||
            budget.spatial_updates_during_solve != 0U ||
            budget.domain_dispatches != expected_dispatches)
            return PM_ERROR_INVALID_STATE;
        const auto result = spatial_.begin(token, counters, expected_passes);
        if (result != PM_SUCCESS) return result;
        representation_ = budget.representation;
        advances_ = 0U;
        iterations_ = 0U;
        spatial_passes_ = 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_advance(
        float timestep, std::uint32_t substeps) noexcept {
        if (!active_ || advances_ != 0U ||
            !smoke_realtime_step_request_valid(timestep, substeps))
            return PM_ERROR_INVALID_STATE;
        ++advances_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_spatial_pass(
        const SpatialEpochToken &token,
        const SpatialEpochCounters &counters) noexcept {
        if (!active_ || advances_ != 1U)
            return PM_ERROR_INVALID_STATE;
        const auto result = spatial_.record_pass(token, counters);
        if (result == PM_SUCCESS) ++spatial_passes_;
        return result;
    }

    [[nodiscard]] PmResult record_iteration() noexcept {
        if (!active_ || advances_ != 1U) return PM_ERROR_INVALID_STATE;
        const std::uint32_t expected = representation_ ==
                SmokeRealtimeRepresentation::particles
            ? smoke_realtime_particle_sweeps
            : smoke_realtime_grid_pressure_cycles;
        if (iterations_ >= expected) return PM_ERROR_INVALID_STATE;
        ++iterations_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult finish() noexcept {
        const std::uint32_t expected_iterations = representation_ ==
                SmokeRealtimeRepresentation::particles
            ? smoke_realtime_particle_sweeps
            : smoke_realtime_grid_pressure_cycles;
        const std::uint32_t expected_passes = representation_ ==
                SmokeRealtimeRepresentation::particles ? 6U : 1U;
        if (!active_ || advances_ != 1U ||
            iterations_ != expected_iterations ||
            spatial_passes_ != expected_passes)
            return PM_ERROR_INVALID_STATE;
        const auto result = spatial_.finish();
        if (result != PM_SUCCESS) return result;
        active_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t advances() const noexcept {
        return advances_;
    }
    [[nodiscard]] constexpr std::uint32_t iterations() const noexcept {
        return iterations_;
    }

private:
    DeformableSpatialSolveEpoch spatial_{};
    SmokeRealtimeRepresentation representation_{};
    std::uint32_t advances_{};
    std::uint32_t iterations_{};
    std::uint32_t spatial_passes_{};
    bool active_{};
};

} // namespace parallel_mater::simulation
