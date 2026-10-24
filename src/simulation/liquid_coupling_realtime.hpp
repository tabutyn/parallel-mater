// SPDX-License-Identifier: MIT
#pragma once

#include "fluid_spatial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t liquid_coupling_realtime_profile_version = 1U;
inline constexpr std::uint32_t liquid_coupling_realtime_sweeps = 4U;
inline constexpr std::uint32_t liquid_coupling_realtime_dispatches_per_sweep =
    3U;
inline constexpr std::uint32_t liquid_coupling_realtime_maximum_edges = 16U;
inline constexpr std::uint32_t liquid_coupling_realtime_maximum_mutations = 16U;
inline constexpr float liquid_coupling_realtime_timestep = 1.0F / 60.0F;
inline constexpr std::uint32_t liquid_coupling_realtime_status_invalid_profile =
    1U << 22U;
inline constexpr std::uint32_t liquid_coupling_realtime_status_invalid_timestep =
    1U << 23U;
inline constexpr std::uint32_t liquid_coupling_realtime_status_invalid_material =
    1U << 24U;
inline constexpr std::uint32_t liquid_coupling_realtime_status_nonfinite_state =
    1U << 25U;
inline constexpr std::uint32_t liquid_coupling_realtime_status_correction_exceeded =
    1U << 26U;
inline constexpr std::uint32_t liquid_coupling_realtime_status_capacity_exceeded =
    1U << 27U;

enum class LiquidCouplingFamily : std::uint32_t {
    liquid = 1U,
    rigid = 2U,
    cloth = 3U,
    soft_body = 4U,
    rope = 5U,
    smoke_particle = 6U,
    smoke_grid = 7U,
};

enum class LiquidCouplingMode : std::uint32_t {
    contact = 1U,
    containment = 2U,
    pressure = 3U,
    viscosity = 4U,
    drag = 5U,
    thermal = 6U,
    phase = 7U,
};

[[nodiscard]] inline bool liquid_coupling_realtime_step_request_valid(
    float dt, std::uint32_t step_count) noexcept {
    const float lower = std::nextafter(
        liquid_coupling_realtime_timestep, 0.0F);
    const float upper = std::nextafter(liquid_coupling_realtime_timestep,
        std::numeric_limits<float>::infinity());
    return step_count == 1U && std::isfinite(dt) && dt >= lower && dt <= upper;
}

[[nodiscard]] constexpr bool liquid_coupling_mode_supported(
    LiquidCouplingFamily family, LiquidCouplingMode mode) noexcept {
    switch (family) {
    case LiquidCouplingFamily::liquid:
        return mode == LiquidCouplingMode::contact ||
            mode == LiquidCouplingMode::pressure ||
            mode == LiquidCouplingMode::viscosity;
    case LiquidCouplingFamily::rigid:
    case LiquidCouplingFamily::cloth:
        return mode == LiquidCouplingMode::contact ||
            mode == LiquidCouplingMode::containment;
    case LiquidCouplingFamily::soft_body:
    case LiquidCouplingFamily::rope:
        return mode == LiquidCouplingMode::contact;
    case LiquidCouplingFamily::smoke_particle:
    case LiquidCouplingFamily::smoke_grid:
        return mode == LiquidCouplingMode::drag ||
            mode == LiquidCouplingMode::thermal ||
            mode == LiquidCouplingMode::phase;
    }
    return false;
}

[[nodiscard]] constexpr bool liquid_coupling_is_two_way_mechanical(
    LiquidCouplingFamily family, LiquidCouplingMode mode) noexcept {
    return family != LiquidCouplingFamily::smoke_particle &&
        family != LiquidCouplingFamily::smoke_grid &&
        mode != LiquidCouplingMode::thermal &&
        mode != LiquidCouplingMode::phase;
}

struct LiquidCouplingRealtimeInput {
    LiquidCouplingFamily family{};
    LiquidCouplingMode mode{};
    float world_coordinate{};
    float characteristic_length{};
    float liquid_mass{};
    float partner_mass{};
    float linear_speed{};
    float angular_speed{};
    float acceleration_displacement{};
    float solver_correction{};
    float contact_distance{};
    float friction{};
    float response{};
    float liquid_temperature{};
    float heater_temperature{};
    float transfer_mass{};
    std::uint64_t liquid_material_signature{};
    std::uint64_t partner_material_signature{};
};

[[nodiscard]] inline PmResult validate_liquid_coupling_realtime_input(
    const LiquidCouplingRealtimeInput &input) noexcept {
    const float values[]{input.world_coordinate, input.characteristic_length,
        input.liquid_mass, input.partner_mass, input.linear_speed,
        input.angular_speed, input.acceleration_displacement,
        input.solver_correction, input.contact_distance, input.friction,
        input.response, input.liquid_temperature, input.heater_temperature,
        input.transfer_mass};
    for (const float value : values)
        if (!std::isfinite(value)) return PM_ERROR_INVALID_ARGUMENT;
    if (!liquid_coupling_mode_supported(input.family, input.mode) ||
        input.liquid_mass < 0.001F || input.liquid_mass > 10'000.0F)
        return PM_ERROR_INVALID_ARGUMENT;
    const bool two_way = liquid_coupling_is_two_way_mechanical(
        input.family, input.mode);
    if (two_way) {
        if (input.partner_mass < 0.001F || input.partner_mass > 10'000.0F)
            return PM_ERROR_INVALID_ARGUMENT;
        const float ratio = std::max(input.liquid_mass, input.partner_mass) /
            std::min(input.liquid_mass, input.partner_mass);
        if (ratio > 100.0F) return PM_ERROR_INVALID_ARGUMENT;
    } else if (input.partner_mass != 0.0F) {
        return PM_ERROR_INVALID_ARGUMENT;
    }
    constexpr float pi = 3.14159265358979323846F;
    if (std::abs(input.world_coordinate) > 10'000.0F ||
        input.characteristic_length < 0.005F ||
        input.characteristic_length > 2.0F || input.linear_speed < 0.0F ||
        input.linear_speed > 20.0F ||
        input.linear_speed * liquid_coupling_realtime_timestep >
            0.5F * input.characteristic_length ||
        input.angular_speed < 0.0F || input.angular_speed > 10.0F * pi ||
        input.angular_speed * liquid_coupling_realtime_timestep > pi / 6.0F ||
        input.acceleration_displacement < 0.0F ||
        input.acceleration_displacement >
            0.25F * input.characteristic_length ||
        input.solver_correction < 0.0F ||
        input.solver_correction > 0.25F * input.characteristic_length ||
        input.friction < 0.0F || input.friction > 2.0F ||
        input.liquid_temperature < -273.15F ||
        input.heater_temperature < -273.15F)
        return PM_ERROR_INVALID_ARGUMENT;
    if (input.family == LiquidCouplingFamily::liquid &&
        (input.liquid_material_signature == 0U ||
         input.liquid_material_signature != input.partner_material_signature))
        return PM_ERROR_INVALID_ARGUMENT;

    switch (input.mode) {
    case LiquidCouplingMode::contact:
    case LiquidCouplingMode::containment:
        if (input.contact_distance <= 0.0F ||
            input.contact_distance > input.characteristic_length ||
            input.response != 0.0F || input.transfer_mass != 0.0F)
            return PM_ERROR_INVALID_ARGUMENT;
        break;
    case LiquidCouplingMode::pressure:
    case LiquidCouplingMode::viscosity:
    case LiquidCouplingMode::drag:
        if (input.contact_distance <= 0.0F ||
            input.contact_distance > input.characteristic_length ||
            input.friction != 0.0F ||
            input.response < 0.05F || input.response > 0.995F ||
            input.transfer_mass != 0.0F)
            return PM_ERROR_INVALID_ARGUMENT;
        break;
    case LiquidCouplingMode::thermal:
        if (input.contact_distance != 0.0F || input.friction != 0.0F ||
            input.response < 0.05F || input.response > 0.995F ||
            input.heater_temperature < input.liquid_temperature ||
            input.transfer_mass != 0.0F)
            return PM_ERROR_INVALID_ARGUMENT;
        break;
    case LiquidCouplingMode::phase:
        if (input.contact_distance != 0.0F || input.friction != 0.0F ||
            input.response != 0.0F || input.transfer_mass <= 0.0F ||
            input.transfer_mass > input.liquid_mass)
            return PM_ERROR_INVALID_ARGUMENT;
        break;
    }
    return PM_SUCCESS;
}

struct LiquidCouplingRegistration {
    std::uint64_t edge_id{};
    LiquidCouplingFamily family{};
    LiquidCouplingMode mode{};
    std::uint32_t liquid_generation{};
    std::uint32_t partner_generation{};
    std::uint32_t constraint_count{};

    auto operator<=>(const LiquidCouplingRegistration &) const = default;
};

struct LiquidCouplingOperatorBudget {
    LiquidCouplingRegistration registration{};
    std::uint32_t algebraic_sweeps{};
    std::uint64_t constraint_visits{};
};

struct LiquidCouplingRealtimeWorkBudget {
    std::uint32_t edge_count{};
    std::uint32_t liquid_particle_count{};
    std::uint32_t partner_endpoint_count{};
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
    std::array<LiquidCouplingOperatorBudget,
               liquid_coupling_realtime_maximum_edges> operators{};
};

[[nodiscard]] inline PmResult make_liquid_coupling_realtime_work_budget(
    std::span<const LiquidCouplingRegistration> registrations,
    std::uint32_t liquid_particle_count,
    std::uint32_t partner_endpoint_count,
    LiquidCouplingRealtimeWorkBudget &result) noexcept {
    result = {};
    if (registrations.size() > liquid_coupling_realtime_maximum_edges)
        return PM_ERROR_CAPACITY_EXCEEDED;
    result.edge_count = static_cast<std::uint32_t>(registrations.size());
    result.liquid_particle_count = liquid_particle_count;
    result.partner_endpoint_count = partner_endpoint_count;
    result.coupling_sweeps_per_edge = liquid_coupling_realtime_sweeps;
    for (std::size_t index = 0U; index < registrations.size(); ++index) {
        const auto &registration = registrations[index];
        if (registration.edge_id == 0U ||
            registration.liquid_generation == 0U ||
            registration.partner_generation == 0U ||
            registration.constraint_count == 0U ||
            !liquid_coupling_mode_supported(
                registration.family, registration.mode) ||
            result.constraint_count >
                std::numeric_limits<std::uint32_t>::max() -
                    registration.constraint_count)
            return PM_ERROR_INVALID_ARGUMENT;
        result.constraint_count += registration.constraint_count;
        result.operators[index] = {registration,
            liquid_coupling_realtime_sweeps,
            std::uint64_t{registration.constraint_count} *
                liquid_coupling_realtime_sweeps};
    }
    result.edge_sweeps = std::uint64_t{result.edge_count} *
        liquid_coupling_realtime_sweeps;
    result.constraint_visits = std::uint64_t{result.constraint_count} *
        liquid_coupling_realtime_sweeps;
    result.batched_dispatches = result.constraint_count == 0U ? 0U :
        liquid_coupling_realtime_dispatches_per_sweep *
            liquid_coupling_realtime_sweeps;
    const std::uint64_t constraint_groups =
        (std::uint64_t{result.constraint_count} + 127U) / 128U;
    const std::uint64_t liquid_groups =
        (std::uint64_t{liquid_particle_count} + 127U) / 128U;
    const std::uint64_t partner_groups =
        (std::uint64_t{partner_endpoint_count} + 127U) / 128U;
    result.batched_workgroups = result.constraint_count == 0U ? 0U :
        liquid_coupling_realtime_sweeps *
            (constraint_groups + liquid_groups + partner_groups);
    // Four Vec3 and two scalar proposal streams per constraint, plus one
    // correction-consumption scalar per endpoint. No pair sort/BVH lives here.
    result.solver_scratch_bytes =
        static_cast<std::size_t>(result.constraint_count) *
            (4U * 3U + 2U) * sizeof(float) +
        (static_cast<std::size_t>(liquid_particle_count) +
            partner_endpoint_count) * sizeof(float);
    return PM_SUCCESS;
}

class LiquidCouplingRealtimeTickContract {
public:
    [[nodiscard]] PmResult begin(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        const LiquidCouplingRealtimeWorkBudget &budget,
        std::span<const LiquidCouplingRegistration> registrations,
        std::uint64_t topology_revision) noexcept {
        if (active_ || token.tick == 0U || topology_revision == 0U ||
            registrations.size() != budget.edge_count ||
            registrations.size() > registrations_.size() ||
            budget.coupling_sweeps_per_edge !=
                liquid_coupling_realtime_sweeps ||
            budget.edge_sweeps != registrations.size() *
                liquid_coupling_realtime_sweeps ||
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
                registration.liquid_generation == 0U ||
                registration.partner_generation == 0U ||
                registration.constraint_count == 0U ||
                !liquid_coupling_mode_supported(
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
            ? liquid_coupling_realtime_sweeps : 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_edge(
        const SpatialEpochToken &token, const SpatialEpochCounters &counters,
        std::uint64_t topology_revision,
        const LiquidCouplingRegistration &registration) noexcept {
        if (!active_ || edge_count_ == 0U || token != token_ ||
            counters.builds != counters_.builds ||
            counters.refits != counters_.refits ||
            counters.publications != counters_.publications ||
            counters.overflows != 0U ||
            topology_revision != topology_revision_ ||
            completed_sweeps_ >= liquid_coupling_realtime_sweeps ||
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
            completed_sweeps_ != liquid_coupling_realtime_sweeps)
            return PM_ERROR_INVALID_STATE;
        active_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t completed_sweeps() const noexcept {
        return completed_sweeps_;
    }

private:
    [[nodiscard]] static constexpr bool canonical_less(
        const LiquidCouplingRegistration &left,
        const LiquidCouplingRegistration &right) noexcept {
        if (left.family != right.family) return left.family < right.family;
        if (left.mode != right.mode) return left.mode < right.mode;
        return left.edge_id < right.edge_id;
    }

    SpatialEpochToken token_{};
    SpatialEpochCounters counters_{};
    std::array<LiquidCouplingRegistration,
               liquid_coupling_realtime_maximum_edges> registrations_{};
    std::uint64_t topology_revision_{};
    std::uint32_t edge_count_{};
    std::uint32_t next_edge_{};
    std::uint32_t completed_sweeps_{};
    bool active_{};
};

enum class LiquidBoundaryMutationKind : std::uint32_t {
    source = 1U,
    outflow = 2U,
    phase = 3U,
};

struct LiquidBoundaryMutation {
    LiquidBoundaryMutationKind kind{};
    std::uint64_t stable_id{};
};

struct LiquidBoundaryCounts {
    std::uint32_t source{};
    std::uint32_t destination{};
};

// Solve-discovered phase work and authored source/outflow changes become
// visible at exactly the next boundary. A failed admission leaves all counts
// and pending stable identities intact for caller-controlled recovery.
class LiquidCouplingBoundaryContract {
public:
    [[nodiscard]] PmResult begin(
        std::uint32_t source_generation,
        std::uint32_t destination_generation,
        LiquidBoundaryCounts counts) noexcept {
        if (active_ || ready_ || source_generation == 0U ||
            destination_generation == 0U)
            return PM_ERROR_INVALID_STATE;
        source_generation_ = source_generation;
        destination_generation_ = destination_generation;
        counts_ = counts;
        mutation_count_ = 0U;
        active_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult stage(LiquidBoundaryMutation mutation) noexcept {
        if (!active_ || mutation.stable_id == 0U ||
            (mutation.kind != LiquidBoundaryMutationKind::source &&
             mutation.kind != LiquidBoundaryMutationKind::outflow &&
             mutation.kind != LiquidBoundaryMutationKind::phase))
            return PM_ERROR_INVALID_STATE;
        if (mutation_count_ >= mutations_.size())
            return PM_ERROR_CAPACITY_EXCEEDED;
        for (std::uint32_t index = 0U; index < mutation_count_; ++index)
            if (mutations_[index].stable_id == mutation.stable_id)
                return PM_ERROR_INVALID_ARGUMENT;
        mutations_[mutation_count_++] = mutation;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult seal_tick() noexcept {
        if (!active_) return PM_ERROR_INVALID_STATE;
        active_ = false;
        ready_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult apply_next_boundary(
        std::uint32_t source_generation,
        std::uint32_t destination_generation,
        std::uint32_t source_capacity,
        std::uint32_t destination_capacity) noexcept {
        if (!ready_ || source_generation != source_generation_ ||
            destination_generation != destination_generation_)
            return PM_ERROR_INVALID_STATE;
        std::uint32_t sources{}, removals{}, transfers{};
        for (std::uint32_t index = 0U; index < mutation_count_; ++index) {
            if (mutations_[index].kind == LiquidBoundaryMutationKind::source)
                ++sources;
            else {
                ++removals;
                if (mutations_[index].kind ==
                    LiquidBoundaryMutationKind::phase)
                    ++transfers;
            }
        }
        if (removals > counts_.source ||
            std::uint64_t{counts_.source - removals} + sources >
                source_capacity ||
            std::uint64_t{counts_.destination} + transfers >
                destination_capacity)
            return PM_ERROR_CAPACITY_EXCEEDED;
        counts_.source = counts_.source - removals + sources;
        counts_.destination += transfers;
        mutation_count_ = 0U;
        ready_ = false;
        ++applied_boundaries_;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr LiquidBoundaryCounts counts() const noexcept {
        return counts_;
    }
    [[nodiscard]] constexpr std::uint32_t pending() const noexcept {
        return mutation_count_;
    }
    [[nodiscard]] constexpr std::uint32_t applied_boundaries() const noexcept {
        return applied_boundaries_;
    }

private:
    std::array<LiquidBoundaryMutation,
               liquid_coupling_realtime_maximum_mutations> mutations_{};
    LiquidBoundaryCounts counts_{};
    std::uint32_t source_generation_{};
    std::uint32_t destination_generation_{};
    std::uint32_t mutation_count_{};
    std::uint32_t applied_boundaries_{};
    bool active_{};
    bool ready_{};
};

} // namespace parallel_mater::simulation
