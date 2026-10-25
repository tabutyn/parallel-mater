// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace parallel_mater::simulation {

// Resource families are finer grained than schedule domains because the two
// smoke representations have different storage and binding obligations.
enum class RealtimeResourceFamily : std::uint8_t {
    rigid,
    liquid,
    cloth,
    soft_body,
    rope,
    particle_smoke,
    grid_smoke,
    count,
};

inline constexpr std::size_t realtime_resource_family_count =
    static_cast<std::size_t>(RealtimeResourceFamily::count);

[[nodiscard]] constexpr std::size_t realtime_index(
    RealtimeResourceFamily value) noexcept {
    return static_cast<std::size_t>(value);
}

enum class RealtimeResourcePhase : std::uint32_t {
    boundary = 1U << 0U,
    predict = 1U << 1U,
    spatial = 1U << 2U,
    domain = 1U << 3U,
    coupling = 1U << 4U,
    lifecycle = 1U << 5U,
    output = 1U << 6U,
    commit = 1U << 7U,
};

[[nodiscard]] constexpr std::uint32_t realtime_phase_bit(
    RealtimeResourcePhase phase) noexcept {
    return static_cast<std::uint32_t>(phase);
}

inline constexpr std::uint32_t realtime_resource_phase_mask =
    (1U << 8U) - 1U;
inline constexpr std::uint32_t realtime_domain_execution_phase_mask =
    realtime_phase_bit(RealtimeResourcePhase::predict) |
    realtime_phase_bit(RealtimeResourcePhase::spatial) |
    realtime_phase_bit(RealtimeResourcePhase::domain);

// Every visit operates on completed public state plus a private staged state.
// Spatial publication is common and singular; only commit exposes staged
// state. These flags are compile-time obligations for F8-40--49 adapters.
struct RealtimePhaseBinding {
    RealtimeResourcePhase phase{};
    bool reads_public{};
    bool reads_staged{};
    bool writes_staged{};
    bool reads_spatial{};
    bool writes_spatial{};
    bool publishes_public{};
    bool permits_host_readback{};
    bool permits_host_wait{};
};

inline constexpr std::array<RealtimePhaseBinding, 8U>
    realtime_phase_bindings{{
        {RealtimeResourcePhase::boundary, true, false, true, false, false,
         false, false, false},
        {RealtimeResourcePhase::predict, true, true, true, false, false,
         false, false, false},
        {RealtimeResourcePhase::spatial, false, true, false, true, true,
         false, false, false},
        {RealtimeResourcePhase::domain, false, true, true, true, false,
         false, false, false},
        {RealtimeResourcePhase::coupling, false, true, true, true, false,
         false, false, false},
        {RealtimeResourcePhase::lifecycle, false, true, true, true, false,
         false, false, false},
        {RealtimeResourcePhase::output, false, true, true, true, false,
         false, false, false},
        {RealtimeResourcePhase::commit, false, true, false, false, false,
         true, false, false},
    }};

struct RealtimeResourceCapacity {
    std::uint32_t slots{};
    std::uint64_t elements{};

    [[nodiscard]] constexpr bool operator==(
        const RealtimeResourceCapacity &) const noexcept = default;
};

struct RealtimeResourcePlan {
    std::array<RealtimeResourceCapacity, realtime_resource_family_count>
        families{};
    // Particle and grid smoke share the public smoke slot table.
    std::uint32_t shared_smoke_slots{};
};

struct RealtimeResourceRegistration {
    std::uint32_t live_slots{};
    std::uint64_t reserved_elements{};
    std::uint64_t identity_hash{};
    std::uint32_t operator_phases{};

    [[nodiscard]] constexpr bool operator==(
        const RealtimeResourceRegistration &) const noexcept = default;
};

struct RealtimeResourceToken {
    std::uint64_t tick{};
    std::uint64_t graph_revision{};
    std::uint64_t resource_revision{};
    std::uint64_t generation{};
    std::uint32_t input_generation{};
    std::uint32_t render_generation{};

    [[nodiscard]] constexpr bool operator==(
        const RealtimeResourceToken &) const noexcept = default;
};

struct RealtimeResourceSnapshot {
    RealtimeResourceToken token{};
    std::array<RealtimeResourceRegistration,
               realtime_resource_family_count> families{};
    std::uint32_t spatial_publications{};

    [[nodiscard]] constexpr bool operator==(
        const RealtimeResourceSnapshot &) const noexcept = default;
};

// Host-side ownership for production resource descriptors. Domain buffers
// remain in their existing device allocations; this fixed-size registry owns
// which generations a tick may read and when staged state becomes public.
class ProductionRealtimeResources {
public:
    [[nodiscard]] PmResult configure(
        const RealtimeResourcePlan &plan) noexcept {
        if (configured_) return PM_ERROR_INVALID_STATE;
        for (const auto &capacity : plan.families) {
            if ((capacity.slots == 0U) != (capacity.elements == 0U))
                return PM_ERROR_INVALID_ARGUMENT;
            if (capacity.elements < capacity.slots)
                return PM_ERROR_INVALID_ARGUMENT;
        }
        const auto particle = plan.families[realtime_index(
            RealtimeResourceFamily::particle_smoke)].slots;
        const auto grid = plan.families[realtime_index(
            RealtimeResourceFamily::grid_smoke)].slots;
        if (plan.shared_smoke_slots > particle ||
            plan.shared_smoke_slots > grid)
            return PM_ERROR_INVALID_ARGUMENT;
        plan_ = plan;
        configured_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] bool configured() const noexcept { return configured_; }

    [[nodiscard]] PmResult stage(
        std::uint64_t tick, std::uint64_t graph_revision,
        std::uint64_t resource_revision, std::uint32_t input_generation,
        std::uint32_t render_generation,
        const std::array<RealtimeResourceRegistration,
                         realtime_resource_family_count> &families) noexcept {
        if (!configured_ || pending_.has_value() || tick == 0U ||
            graph_revision == 0U || resource_revision == 0U ||
            input_generation == 0U || render_generation == 0U ||
            (current_ && (tick <= current_->token.tick ||
                          graph_revision <= current_->token.graph_revision ||
                          resource_revision <=
                              current_->token.resource_revision)))
            return PM_ERROR_INVALID_STATE;
        if (resource_revision <= invalidated_resource_revision_)
            return PM_ERROR_INVALID_STATE;

        std::uint64_t smoke_slots{};
        for (std::size_t index = 0U; index < families.size(); ++index) {
            const auto &registration = families[index];
            const auto &capacity = plan_.families[index];
            if (registration.live_slots > capacity.slots ||
                registration.reserved_elements > capacity.elements)
                return PM_ERROR_CAPACITY_EXCEEDED;
            if ((registration.live_slots == 0U) !=
                    (registration.reserved_elements == 0U) ||
                (registration.live_slots != 0U &&
                 registration.identity_hash == 0U) ||
                registration.reserved_elements < registration.live_slots ||
                (registration.operator_phases &
                 ~realtime_resource_phase_mask) != 0U)
                return PM_ERROR_INVALID_ARGUMENT;
            if (index == realtime_index(
                    RealtimeResourceFamily::particle_smoke) ||
                index == realtime_index(
                    RealtimeResourceFamily::grid_smoke))
                smoke_slots += registration.live_slots;
        }
        if (smoke_slots > plan_.shared_smoke_slots)
            return PM_ERROR_CAPACITY_EXCEEDED;
        if (next_generation_ == 0U) return PM_ERROR_CAPACITY_EXCEEDED;

        pending_ = RealtimeResourceSnapshot{
            {tick, graph_revision, resource_revision, next_generation_++,
             input_generation, render_generation}, families, 0U};
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult execution_status() const noexcept {
        if (!pending_) return PM_ERROR_INVALID_STATE;
        for (const auto &family : pending_->families)
            if (family.live_slots != 0U &&
                (family.operator_phases &
                 realtime_domain_execution_phase_mask) !=
                    realtime_domain_execution_phase_mask)
                return PM_ERROR_FEATURE_NOT_PRESENT;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult publish_spatial() noexcept {
        if (!pending_ || pending_->spatial_publications != 0U)
            return PM_ERROR_INVALID_STATE;
        pending_->spatial_publications = 1U;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult prepare_commit() const noexcept {
        if (!pending_ || pending_->spatial_publications != 1U)
            return PM_ERROR_INVALID_STATE;
        return execution_status();
    }

    [[nodiscard]] PmResult commit() noexcept {
        const auto result = prepare_commit();
        if (result != PM_SUCCESS) return result;
        current_ = *pending_;
        pending_.reset();
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult abort() noexcept {
        if (!pending_) return PM_ERROR_INVALID_STATE;
        pending_.reset();
        return PM_SUCCESS;
    }

    void invalidate(std::uint64_t resource_revision) noexcept {
        pending_.reset();
        current_.reset();
        if (resource_revision > invalidated_resource_revision_)
            invalidated_resource_revision_ = resource_revision;
        if (next_generation_ != 0U) ++next_generation_;
    }

    [[nodiscard]] const RealtimeResourcePlan &plan() const noexcept {
        return plan_;
    }
    [[nodiscard]] const std::optional<RealtimeResourceSnapshot> &pending()
        const noexcept { return pending_; }
    [[nodiscard]] const std::optional<RealtimeResourceSnapshot> &current()
        const noexcept { return current_; }

private:
    RealtimeResourcePlan plan_{};
    std::optional<RealtimeResourceSnapshot> pending_{};
    std::optional<RealtimeResourceSnapshot> current_{};
    std::uint64_t next_generation_{1U};
    std::uint64_t invalidated_resource_revision_{};
    bool configured_{};
};

} // namespace parallel_mater::simulation
