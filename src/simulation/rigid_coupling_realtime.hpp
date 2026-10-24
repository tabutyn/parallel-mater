// SPDX-License-Identifier: MIT
#pragma once

#include "rigid_spatial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t rigid_coupling_realtime_profile_version = 1U;
inline constexpr std::uint32_t rigid_coupling_realtime_sweeps = 4U;
inline constexpr std::uint32_t rigid_coupling_realtime_dispatches_per_sweep =
    3U;
inline constexpr std::uint32_t rigid_coupling_realtime_maximum_edges = 16U;
inline constexpr float rigid_coupling_realtime_timestep = 1.0F / 60.0F;
inline constexpr std::uint32_t rigid_coupling_realtime_status_invalid_profile =
    1U << 17U;
inline constexpr std::uint32_t rigid_coupling_realtime_status_invalid_timestep =
    1U << 18U;
inline constexpr std::uint32_t rigid_coupling_realtime_status_invalid_material =
    1U << 19U;
inline constexpr std::uint32_t rigid_coupling_realtime_status_nonfinite_state =
    1U << 20U;
inline constexpr std::uint32_t rigid_coupling_realtime_status_correction_exceeded =
    1U << 21U;

enum class RigidCouplingFamily : std::uint32_t {
    cloth = 1U,
    soft_body = 2U,
    rope = 3U,
    smoke_particle = 4U,
    smoke_grid = 5U,
};

enum class RigidCouplingMode : std::uint32_t {
    contact = 1U,
    attachment = 2U,
    drag = 3U,
};

[[nodiscard]] constexpr bool rigid_coupling_mode_supported(
    RigidCouplingFamily family, RigidCouplingMode mode) noexcept {
    switch (family) {
    case RigidCouplingFamily::cloth:
    case RigidCouplingFamily::soft_body:
        return mode == RigidCouplingMode::contact;
    case RigidCouplingFamily::rope:
        return mode == RigidCouplingMode::contact ||
            mode == RigidCouplingMode::attachment;
    case RigidCouplingFamily::smoke_particle:
    case RigidCouplingFamily::smoke_grid:
        return mode == RigidCouplingMode::contact ||
            mode == RigidCouplingMode::drag;
    }
    return false;
}

[[nodiscard]] inline bool rigid_coupling_realtime_step_request_valid(
    float timestep, std::uint32_t substeps) noexcept {
    const float lower = std::nextafter(
        rigid_coupling_realtime_timestep, 0.0F);
    const float upper = std::nextafter(rigid_coupling_realtime_timestep,
        std::numeric_limits<float>::infinity());
    return std::isfinite(timestep) && timestep >= lower &&
        timestep <= upper && substeps == 1U;
}

struct RigidCouplingRealtimeInput {
    RigidCouplingFamily family{};
    RigidCouplingMode mode{};
    float world_coordinate{};
    float characteristic_length{};
    float endpoint_mass{};
    float rigid_mass{};
    bool endpoint_dynamic{true};
    bool rigid_dynamic{true};
    float linear_speed{};
    float angular_speed{};
    float acceleration_displacement{};
    float solver_correction{};
    float contact_distance{};
    float friction{};
    float response{};
};

[[nodiscard]] inline PmResult validate_rigid_coupling_realtime_input(
    const RigidCouplingRealtimeInput &input) noexcept {
    const float values[]{input.world_coordinate, input.characteristic_length,
        input.endpoint_mass, input.rigid_mass, input.linear_speed,
        input.angular_speed, input.acceleration_displacement,
        input.solver_correction, input.contact_distance, input.friction,
        input.response};
    for (const float value : values)
        if (!std::isfinite(value)) return PM_ERROR_INVALID_ARGUMENT;
    if (!rigid_coupling_mode_supported(input.family, input.mode) ||
        (!input.endpoint_dynamic && !input.rigid_dynamic))
        return PM_ERROR_INVALID_ARGUMENT;
    const auto valid_mass = [](float mass, bool dynamic) noexcept {
        return dynamic ? mass >= 0.001F && mass <= 10'000.0F : mass == 0.0F;
    };
    if (!valid_mass(input.endpoint_mass, input.endpoint_dynamic) ||
        !valid_mass(input.rigid_mass, input.rigid_dynamic))
        return PM_ERROR_INVALID_ARGUMENT;
    if (input.endpoint_dynamic && input.rigid_dynamic) {
        const float ratio = std::max(input.endpoint_mass, input.rigid_mass) /
            std::min(input.endpoint_mass, input.rigid_mass);
        if (ratio > 100.0F) return PM_ERROR_INVALID_ARGUMENT;
    }
    constexpr float pi = 3.14159265358979323846F;
    if (std::abs(input.world_coordinate) > 10'000.0F ||
        input.characteristic_length < 0.005F ||
        input.characteristic_length > 2.0F || input.linear_speed < 0.0F ||
        input.linear_speed > 20.0F ||
        input.linear_speed * rigid_coupling_realtime_timestep >
            0.5F * input.characteristic_length ||
        input.angular_speed < 0.0F || input.angular_speed > 10.0F * pi ||
        input.angular_speed * rigid_coupling_realtime_timestep > pi / 6.0F ||
        input.acceleration_displacement < 0.0F ||
        input.acceleration_displacement >
            0.25F * input.characteristic_length ||
        input.solver_correction < 0.0F ||
        input.solver_correction > 0.25F * input.characteristic_length ||
        input.friction < 0.0F || input.friction > 2.0F)
        return PM_ERROR_INVALID_ARGUMENT;
    if (input.mode == RigidCouplingMode::contact) {
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

struct RigidCouplingRegistration {
    std::uint64_t edge_id{};
    RigidCouplingFamily family{};
    RigidCouplingMode mode{};
    std::uint32_t endpoint_generation{};
    std::uint32_t rigid_generation{};
    std::uint32_t constraint_count{};

    auto operator<=>(const RigidCouplingRegistration &) const = default;
};

struct RigidCouplingOperatorBudget {
    RigidCouplingRegistration registration{};
    std::uint32_t algebraic_sweeps{};
    std::uint64_t constraint_visits{};
};

struct RigidCouplingRealtimeWorkBudget {
    std::uint32_t edge_count{};
    std::uint32_t endpoint_count{};
    std::uint32_t rigid_body_count{};
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
    std::array<RigidCouplingOperatorBudget,
               rigid_coupling_realtime_maximum_edges> operators{};
};

[[nodiscard]] inline PmResult make_rigid_coupling_realtime_work_budget(
    std::span<const RigidCouplingRegistration> registrations,
    std::uint32_t endpoint_count, std::uint32_t rigid_body_count,
    RigidCouplingRealtimeWorkBudget &result) noexcept {
    result = {};
    if (registrations.size() > rigid_coupling_realtime_maximum_edges)
        return PM_ERROR_CAPACITY_EXCEEDED;
    result.edge_count = static_cast<std::uint32_t>(registrations.size());
    result.endpoint_count = endpoint_count;
    result.rigid_body_count = rigid_body_count;
    result.coupling_sweeps_per_edge = rigid_coupling_realtime_sweeps;
    for (std::size_t index = 0U; index < registrations.size(); ++index) {
        const auto &registration = registrations[index];
        if (registration.edge_id == 0U ||
            registration.endpoint_generation == 0U ||
            registration.rigid_generation == 0U ||
            registration.constraint_count == 0U ||
            !rigid_coupling_mode_supported(
                registration.family, registration.mode) ||
            result.constraint_count >
                std::numeric_limits<std::uint32_t>::max() -
                    registration.constraint_count)
            return PM_ERROR_INVALID_ARGUMENT;
        result.constraint_count += registration.constraint_count;
        result.operators[index] = {registration,
            rigid_coupling_realtime_sweeps,
            std::uint64_t{registration.constraint_count} *
                rigid_coupling_realtime_sweeps};
    }
    result.edge_sweeps = std::uint64_t{result.edge_count} *
        rigid_coupling_realtime_sweeps;
    result.constraint_visits = std::uint64_t{result.constraint_count} *
        rigid_coupling_realtime_sweeps;
    result.batched_dispatches = result.constraint_count == 0U ? 0U :
        rigid_coupling_realtime_dispatches_per_sweep *
            rigid_coupling_realtime_sweeps;
    const std::uint64_t constraint_groups =
        (std::uint64_t{result.constraint_count} + 127U) / 128U;
    const std::uint64_t endpoint_groups =
        (std::uint64_t{endpoint_count} + 127U) / 128U;
    const std::uint64_t body_groups =
        (std::uint64_t{rigid_body_count} + 127U) / 128U;
    result.batched_workgroups = result.constraint_count == 0U ? 0U :
        rigid_coupling_realtime_sweeps *
            (constraint_groups + endpoint_groups + body_groups);
    // Four Vec3 proposal streams plus one correction-consumption scalar per
    // endpoint and rigid body. No candidate or BVH storage is allocated here.
    result.solver_scratch_bytes =
        static_cast<std::size_t>(result.constraint_count) * 4U *
            3U * sizeof(float) +
        (static_cast<std::size_t>(endpoint_count) + rigid_body_count) *
            sizeof(float);
    return PM_SUCCESS;
}

// One registration pass contains four inner sweeps. Registrations must be in
// canonical family/mode/identity order so three-plus-family results do not
// depend on caller insertion order. Every sweep retains one F8-10 epoch.
class RigidCouplingRealtimeTickContract {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        const RigidCouplingRealtimeWorkBudget &budget,
        std::span<const RigidCouplingRegistration> registrations,
        std::uint64_t topology_revision) noexcept {
        if (active_ || token.tick == 0U || topology_revision == 0U ||
            registrations.size() != budget.edge_count ||
            registrations.size() > registrations_.size() ||
            budget.coupling_sweeps_per_edge !=
                rigid_coupling_realtime_sweeps ||
            budget.edge_sweeps != registrations.size() *
                rigid_coupling_realtime_sweeps ||
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
                registration.endpoint_generation == 0U ||
                registration.rigid_generation == 0U ||
                registration.constraint_count == 0U ||
                !rigid_coupling_mode_supported(
                    registration.family, registration.mode) ||
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
            ? rigid_coupling_realtime_sweeps : 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_edge(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        std::uint64_t topology_revision,
        const RigidCouplingRegistration &registration) noexcept {
        if (!active_ || edge_count_ == 0U || token != token_ ||
            counters.builds != counters_.builds ||
            counters.refits != counters_.refits ||
            counters.publications != counters_.publications ||
            counters.overflows != 0U ||
            topology_revision != topology_revision_ ||
            completed_sweeps_ >= rigid_coupling_realtime_sweeps ||
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
            completed_sweeps_ != rigid_coupling_realtime_sweeps)
            return PM_ERROR_INVALID_STATE;
        active_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t completed_sweeps() const noexcept {
        return completed_sweeps_;
    }

private:
    [[nodiscard]] static constexpr bool canonical_less(
        const RigidCouplingRegistration &left,
        const RigidCouplingRegistration &right) noexcept {
        if (left.family != right.family) return left.family < right.family;
        if (left.mode != right.mode) return left.mode < right.mode;
        return left.edge_id < right.edge_id;
    }

    SpatialEpochToken token_{};
    SpatialEpochCounters counters_{};
    std::array<RigidCouplingRegistration,
               rigid_coupling_realtime_maximum_edges> registrations_{};
    std::uint64_t topology_revision_{};
    std::uint32_t edge_count_{};
    std::uint32_t next_edge_{};
    std::uint32_t completed_sweeps_{};
    bool active_{};
};

} // namespace parallel_mater::simulation
