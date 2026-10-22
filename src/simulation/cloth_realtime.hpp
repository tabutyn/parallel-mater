// SPDX-License-Identifier: MIT
#pragma once

#include "deformable_spatial.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t cloth_realtime_profile_version = 1U;
inline constexpr std::uint32_t cloth_realtime_constraint_sweeps = 8U;
inline constexpr std::uint32_t cloth_realtime_contact_passes = 4U;
inline constexpr float cloth_realtime_timestep = 1.0F / 60.0F;
inline constexpr float cloth_realtime_solver_damping = 0.05F;
inline constexpr float cloth_realtime_maximum_correction_fraction = 0.25F;
inline constexpr std::uint32_t cloth_realtime_status_invalid_profile =
    1U << 17U;
inline constexpr std::uint32_t cloth_realtime_status_invalid_timestep =
    1U << 18U;
inline constexpr std::uint32_t cloth_realtime_status_invalid_material =
    1U << 19U;
inline constexpr std::uint32_t cloth_realtime_status_nonfinite_state =
    1U << 20U;
inline constexpr std::uint32_t cloth_realtime_status_unresolved_contact =
    1U << 21U;

static_assert(cloth_realtime_contact_passes ==
              deformable_spatial_collision_passes);

[[nodiscard]] inline bool cloth_realtime_step_request_valid(
    float timestep, std::uint32_t substeps) noexcept {
    const float lower = std::nextafter(cloth_realtime_timestep, 0.0F);
    const float upper = std::nextafter(
        cloth_realtime_timestep, std::numeric_limits<float>::infinity());
    return std::isfinite(timestep) && timestep >= lower &&
        timestep <= upper && substeps == 1U;
}

[[nodiscard]] inline float cloth_realtime_compliance_response(
    float compliance, float effective_inverse_mass,
    float timestep = cloth_realtime_timestep) noexcept {
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

// Host validation for realtime-single-step's interpretation of cloth inputs.
// Compliance zero means a rigid constraint. Positive stretch/bend compliance
// is XPBD compliance and must map into the declared dimensionless response.
struct ClothRealtimeInput {
    float world_coordinate{};
    float characteristic_length{};
    float mass{};
    float interacting_mass_ratio{1.0F};
    float linear_speed{};
    float acceleration_displacement{};
    float solver_correction{};
    float authored_strain{};
    float stretch_compliance{};
    float bending_compliance{};
    float volume_compliance{};
    float effective_inverse_mass{};
    float thickness{};
    float velocity_damping{};
    float contact_friction{};
    float solver_damping{cloth_realtime_solver_damping};
};

[[nodiscard]] inline PmResult validate_cloth_realtime_input(
    const ClothRealtimeInput &input) noexcept {
    const float values[]{input.world_coordinate,
        input.characteristic_length, input.mass,
        input.interacting_mass_ratio, input.linear_speed,
        input.acceleration_displacement, input.solver_correction,
        input.authored_strain, input.stretch_compliance,
        input.bending_compliance, input.volume_compliance,
        input.effective_inverse_mass, input.thickness,
        input.velocity_damping, input.contact_friction,
        input.solver_damping};
    for (const float value : values)
        if (!std::isfinite(value)) return PM_ERROR_INVALID_ARGUMENT;
    if (std::abs(input.world_coordinate) > 10'000.0F ||
        input.characteristic_length < 0.005F ||
        input.characteristic_length > 2.0F || input.mass < 0.001F ||
        input.mass > 10'000.0F || input.interacting_mass_ratio < 1.0F ||
        input.interacting_mass_ratio > 100.0F || input.linear_speed < 0.0F ||
        input.linear_speed > 20.0F ||
        input.linear_speed * cloth_realtime_timestep >
            0.5F * input.characteristic_length ||
        input.acceleration_displacement < 0.0F ||
        input.acceleration_displacement >
            0.25F * input.characteristic_length ||
        input.solver_correction < 0.0F ||
        input.solver_correction >
            cloth_realtime_maximum_correction_fraction *
                input.characteristic_length ||
        input.authored_strain < 0.0F || input.authored_strain > 0.25F ||
        input.stretch_compliance < 0.0F ||
        input.bending_compliance < 0.0F ||
        input.volume_compliance < 0.0F ||
        input.effective_inverse_mass <= 0.0F || input.thickness < 0.0F ||
        input.thickness > 0.5F * input.characteristic_length ||
        input.velocity_damping < 0.0F || input.contact_friction < 0.0F ||
        input.contact_friction > 2.0F || input.solver_damping < 0.0F ||
        input.solver_damping > 0.12F)
        return PM_ERROR_INVALID_ARGUMENT;
    constexpr float tolerance =
        2.0F * std::numeric_limits<float>::epsilon();
    for (const float compliance :
         {input.stretch_compliance, input.bending_compliance}) {
        if (compliance == 0.0F) continue;
        const float response = cloth_realtime_compliance_response(
            compliance, input.effective_inverse_mass);
        if (!std::isfinite(response) || response < 0.05F - tolerance ||
            response > 0.995F + tolerance)
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}

struct ClothRealtimeWorkBudget {
    std::uint32_t live_vertices{};
    std::uint32_t vertex_capacity{};
    std::uint32_t triangle_count{};
    std::uint32_t directed_link_count{};
    std::uint32_t contact_constraint_count{};
    std::uint32_t constraint_sweeps{};
    std::uint32_t contact_passes{};
    std::uint32_t time_advances{};
    std::uint32_t spatial_updates_during_solve{};
    std::uint32_t domain_dispatches{};
    std::uint64_t domain_workgroups{};
    std::uint64_t link_visits{};
    std::uint64_t contact_visits{};
    std::size_t solver_scratch_bytes{};
};

[[nodiscard]] constexpr ClothRealtimeWorkBudget cloth_realtime_work_budget(
    std::uint32_t live_vertices, std::uint32_t vertex_capacity,
    std::uint32_t triangle_count, std::uint32_t directed_link_count,
    std::uint32_t contact_constraint_count) noexcept {
    ClothRealtimeWorkBudget result{};
    result.live_vertices = live_vertices;
    result.vertex_capacity = vertex_capacity;
    result.triangle_count = triangle_count;
    result.directed_link_count = directed_link_count;
    result.contact_constraint_count = contact_constraint_count;
    result.constraint_sweeps = cloth_realtime_constraint_sweeps;
    result.contact_passes = cloth_realtime_contact_passes;
    result.time_advances = 1U;
    // predict + eight (links, apply, volume) + four contact projections +
    // commit + surface update.
    result.domain_dispatches = 1U + 3U * cloth_realtime_constraint_sweeps +
        cloth_realtime_contact_passes + 2U;
    const std::uint64_t vertex_groups =
        (std::uint64_t{live_vertices} + 127U) / 128U;
    const std::uint64_t triangle_groups =
        (std::uint64_t{triangle_count} + 127U) / 128U;
    result.domain_workgroups =
        vertex_groups * (1U + 2U * cloth_realtime_constraint_sweeps +
            cloth_realtime_contact_passes + 1U) +
        cloth_realtime_constraint_sweeps + triangle_groups;
    result.link_visits = std::uint64_t{directed_link_count} *
        cloth_realtime_constraint_sweeps;
    result.contact_visits = std::uint64_t{contact_constraint_count} *
        cloth_realtime_contact_passes;
    // Previous position, Jacobi delta, volume gradient, scalar correction
    // consumption, volume lambda, advance count and status.
    result.solver_scratch_bytes = static_cast<std::size_t>(vertex_capacity) *
            (3U * 3U * sizeof(float) + sizeof(float)) +
        sizeof(float) + 2U * sizeof(std::uint32_t);
    return result;
}

// Pins topology and the one F8-10 publication across eight algebraic sweeps.
// Contact runs on sweeps 2/4/6/8, consuming exactly four collision passes.
class ClothRealtimeTickContract {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        const ClothRealtimeWorkBudget &budget,
        std::uint64_t topology_revision) noexcept {
        if (active_ || topology_revision == 0U ||
            budget.constraint_sweeps != cloth_realtime_constraint_sweeps ||
            budget.contact_passes != cloth_realtime_contact_passes ||
            budget.time_advances != 1U ||
            budget.spatial_updates_during_solve != 0U ||
            budget.domain_dispatches != 31U)
            return PM_ERROR_INVALID_STATE;
        const auto result = spatial_.begin(
            token, counters, cloth_realtime_contact_passes);
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
            !cloth_realtime_step_request_valid(timestep, substeps))
            return PM_ERROR_INVALID_STATE;
        ++advances_;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_sweep(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        std::uint64_t topology_revision, bool contact_pass) noexcept {
        const bool expected_contact = (sweeps_ & 1U) != 0U;
        if (!active_ || advances_ != 1U ||
            sweeps_ >= cloth_realtime_constraint_sweeps ||
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
            sweeps_ != cloth_realtime_constraint_sweeps)
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
