// SPDX-License-Identifier: MIT
#pragma once

#include "realtime_schedule.hpp"

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t realtime_telemetry_schema_version = 1U;
inline constexpr std::uint32_t realtime_telemetry_budget_version = 1U;

// Stable report phases.  Native adapters may emit several kernels for one
// schedule visit and may attribute candidate generation separately from the
// index build that made it possible.
enum class RealtimeTelemetryPhase : std::uint8_t {
    boundary,
    predict,
    index,
    candidates,
    domain,
    coupling,
    commit,
    count,
};

enum class RealtimeGpuTimingStatus : std::uint8_t {
    unsupported,
    pending,
    valid,
};

inline constexpr std::size_t realtime_telemetry_phase_count =
    static_cast<std::size_t>(RealtimeTelemetryPhase::count);

[[nodiscard]] constexpr std::size_t realtime_index(
    RealtimeTelemetryPhase value) noexcept {
    return static_cast<std::size_t>(value);
}

struct RealtimeWorkCounters {
    std::uint64_t dispatches{};
    std::uint64_t workgroups{};
    std::uint64_t bytes_moved{};
    std::uint32_t index_builds{};
    std::uint32_t index_refits{};
    std::uint32_t index_publications{};
    std::uint32_t numerical_iterations{};
    std::uint32_t barriers{};
    std::uint32_t submissions{};
    std::uint32_t readbacks{};
    std::uint32_t host_waits{};

    auto operator<=>(const RealtimeWorkCounters &) const = default;
};

struct RealtimePhaseTiming {
    RealtimeGpuTimingStatus gpu_status{
        RealtimeGpuTimingStatus::unsupported};
    std::uint64_t cpu_encode_ns{};
    std::uint64_t cpu_submit_ns{};
    std::uint64_t cpu_wait_ns{};
    std::uint64_t gpu_ns{};
};

struct RealtimePhaseTelemetry {
    RealtimeWorkCounters work{};
    RealtimePhaseTiming timing{};
};

// Query-result traffic is instrumentation cost, not simulation work.  It is
// resolved only after normal completion and must never add a forced wait.
struct RealtimeInstrumentationCost {
    std::uint32_t query_resets{};
    std::uint32_t timestamp_queries{};
    std::uint32_t asynchronous_result_reads{};
    std::uint32_t forced_waits{};
    std::uint64_t result_bytes{};
    std::uint64_t cpu_overhead_ns{};
};

struct RealtimeTickTelemetry {
    std::uint32_t schema_version{realtime_telemetry_schema_version};
    std::uint32_t profile_version{rigid_realtime_profile_version};
    std::uint32_t budget_version{realtime_telemetry_budget_version};
    std::uint64_t tick{};
    std::array<RealtimePhaseTelemetry, realtime_telemetry_phase_count> phases{};
    RealtimeWorkCounters total{};
    RealtimeInstrumentationCost instrumentation{};
};

struct RealtimeTelemetryLimits {
    std::uint32_t profile_version{rigid_realtime_profile_version};
    std::uint32_t budget_version{realtime_telemetry_budget_version};
    std::array<RealtimeWorkCounters, realtime_telemetry_phase_count> phase_maximum{};
    RealtimeWorkCounters total_maximum{};
    std::uint32_t timestamp_query_capacity{};
    std::uint64_t instrumentation_result_bytes_maximum{};
};

struct RealtimeTelemetryOperation {
    RealtimeSchedulePhase schedule_phase{};
    std::uint32_t first{};
    std::uint32_t second{};
};

struct RealtimeOperationMeasurement {
    std::array<RealtimeWorkCounters, realtime_telemetry_phase_count> phases{};
};

static_assert(std::is_trivially_copyable_v<RealtimeWorkCounters>);
static_assert(std::is_trivially_copyable_v<RealtimePhaseTiming>);
static_assert(std::is_trivially_copyable_v<RealtimeTickTelemetry>);
static_assert(std::is_trivially_copyable_v<RealtimeTelemetryLimits>);
static_assert(std::is_trivially_copyable_v<RealtimeOperationMeasurement>);

namespace realtime_telemetry_detail {

template <typename T>
[[nodiscard]] constexpr bool checked_add(T &destination, T value) noexcept {
    if (value > std::numeric_limits<T>::max() - destination) return false;
    destination += value;
    return true;
}

[[nodiscard]] constexpr bool add(
    RealtimeWorkCounters &destination,
    const RealtimeWorkCounters &value) noexcept {
    return checked_add(destination.dispatches, value.dispatches) &&
        checked_add(destination.workgroups, value.workgroups) &&
        checked_add(destination.bytes_moved, value.bytes_moved) &&
        checked_add(destination.index_builds, value.index_builds) &&
        checked_add(destination.index_refits, value.index_refits) &&
        checked_add(destination.index_publications,
                    value.index_publications) &&
        checked_add(destination.numerical_iterations,
                    value.numerical_iterations) &&
        checked_add(destination.barriers, value.barriers) &&
        checked_add(destination.submissions, value.submissions) &&
        checked_add(destination.readbacks, value.readbacks) &&
        checked_add(destination.host_waits, value.host_waits);
}

[[nodiscard]] constexpr bool within(
    const RealtimeWorkCounters &value,
    const RealtimeWorkCounters &maximum) noexcept {
    return value.dispatches <= maximum.dispatches &&
        value.workgroups <= maximum.workgroups &&
        value.bytes_moved <= maximum.bytes_moved &&
        value.index_builds <= maximum.index_builds &&
        value.index_refits <= maximum.index_refits &&
        value.index_publications <= maximum.index_publications &&
        value.numerical_iterations <= maximum.numerical_iterations &&
        value.barriers <= maximum.barriers &&
        value.submissions <= maximum.submissions &&
        value.readbacks <= maximum.readbacks &&
        value.host_waits <= maximum.host_waits;
}

[[nodiscard]] constexpr std::uint32_t expected_iterations(
    const RealtimeScheduleLedger &ledger) noexcept {
    std::uint32_t result{};
    for (const auto value : ledger.domain_iterations) result += value;
    for (const auto value : ledger.coupling_sweeps) result += value;
    return result;
}

[[nodiscard]] constexpr std::uint32_t expected_iterations(
    const RealtimeScheduleScene &scene) noexcept {
    const auto budget = realtime_schedule_budget(scene);
    std::uint32_t result{};
    for (const auto value : budget.domain_iterations) result += value;
    for (const auto value : budget.coupling_sweeps) result += value;
    return result;
}

[[nodiscard]] constexpr bool expected_spatial_progress(
    const RealtimeScheduleScene &scene,
    const RealtimeWorkCounters &work, bool final) noexcept {
    const std::uint32_t builds = scene.spatial_update ==
            RealtimeSpatialUpdate::build ? 1U : 0U;
    const std::uint32_t refits = scene.spatial_update ==
            RealtimeSpatialUpdate::refit ? 1U : 0U;
    if (work.index_builds > builds || work.index_refits > refits ||
        work.index_publications > 1U)
        return false;
    return !final || (work.index_builds == builds &&
        work.index_refits == refits && work.index_publications == 1U);
}

} // namespace realtime_telemetry_detail

[[nodiscard]] constexpr bool realtime_telemetry_progress_valid(
    const RealtimeScheduleScene &scene,
    const RealtimeScheduleLedger &ledger,
    const RealtimeTelemetryLimits &limits,
    const RealtimeTickTelemetry &telemetry,
    bool final = false) noexcept {
    RealtimeWorkCounters summed{};
    for (const auto &phase : telemetry.phases)
        if (!realtime_telemetry_detail::add(summed, phase.work)) return false;
    if (summed != telemetry.total ||
        telemetry.schema_version != realtime_telemetry_schema_version ||
        telemetry.profile_version != limits.profile_version ||
        telemetry.budget_version != limits.budget_version ||
        telemetry.tick != scene.tick ||
        telemetry.total.readbacks != 0U ||
        telemetry.total.host_waits != 0U ||
        !realtime_telemetry_detail::expected_spatial_progress(
            scene, telemetry.total, final) ||
        telemetry.total.numerical_iterations > (final
            ? realtime_telemetry_detail::expected_iterations(ledger)
            : realtime_telemetry_detail::expected_iterations(scene)) ||
        !realtime_telemetry_detail::within(
            telemetry.total, limits.total_maximum))
        return false;
    for (std::size_t index = 0U; index < realtime_telemetry_phase_count;
         ++index)
        if (!realtime_telemetry_detail::within(
                telemetry.phases[index].work,
                limits.phase_maximum[index]))
            return false;
    if (final && telemetry.total.numerical_iterations !=
            realtime_telemetry_detail::expected_iterations(ledger))
        return false;
    return true;
}

[[nodiscard]] constexpr bool realtime_telemetry_timing_ready(
    const RealtimeTickTelemetry &telemetry) noexcept {
    for (const auto &phase : telemetry.phases) {
        if (phase.timing.gpu_status == RealtimeGpuTimingStatus::pending)
            return false;
        if (phase.timing.gpu_status ==
                RealtimeGpuTimingStatus::unsupported &&
            phase.timing.gpu_ns != 0U)
            return false;
    }
    return true;
}

[[nodiscard]] constexpr bool realtime_telemetry_report_valid(
    const RealtimeScheduleScene &scene,
    const RealtimeScheduleLedger &ledger,
    const RealtimeTelemetryLimits &limits,
    const RealtimeTickTelemetry &telemetry) noexcept {
    if (!realtime_telemetry_progress_valid(
            scene, ledger, limits, telemetry, true) ||
        !realtime_telemetry_timing_ready(telemetry) ||
        telemetry.total.submissions == 0U ||
        telemetry.instrumentation.forced_waits != 0U ||
        telemetry.instrumentation.timestamp_queries >
            limits.timestamp_query_capacity ||
        telemetry.instrumentation.result_bytes >
            limits.instrumentation_result_bytes_maximum)
        return false;
    return true;
}

template <typename Backend>
class MeasuredRealtimeBackend {
public:
    MeasuredRealtimeBackend(
        Backend &backend, const RealtimeScheduleScene &scene,
        RealtimeScheduleLedger &ledger,
        const RealtimeTelemetryLimits &limits,
        RealtimeTickTelemetry &telemetry) noexcept
        : backend_(backend), scene_(scene), ledger_(ledger), limits_(limits),
          telemetry_(telemetry) {}

    [[nodiscard]] PmResult begin_tick(
        const RealtimeScheduleScene &scene) noexcept {
        auto result = backend_.begin_tick(scene);
        if (result != PM_SUCCESS) return result;
        result = capture({RealtimeSchedulePhase::boundary, 0U, 0U});
        if (result != PM_SUCCESS) (void)backend_.abort_tick(scene);
        return result;
    }
    [[nodiscard]] PmResult boundary(
        const RealtimeScheduleScene &scene) noexcept {
        const auto result = backend_.boundary(scene);
        return result == PM_SUCCESS
            ? capture({RealtimeSchedulePhase::boundary, 1U, 0U}) : result;
    }
    [[nodiscard]] PmResult predict(
        RealtimeDomain domain, const RealtimeScheduleScene &scene) noexcept {
        const auto result = backend_.predict(domain, scene);
        return result == PM_SUCCESS ? capture({
            RealtimeSchedulePhase::predict,
            static_cast<std::uint32_t>(domain), 0U}) : result;
    }
    [[nodiscard]] PmResult update_spatial(
        RealtimeSpatialUpdate update,
        const RealtimeScheduleScene &scene) noexcept {
        const auto result = backend_.update_spatial(update, scene);
        return result == PM_SUCCESS ? capture({
            RealtimeSchedulePhase::spatial_update,
            static_cast<std::uint32_t>(update), 0U}) : result;
    }
    [[nodiscard]] PmResult publish_spatial(
        const RealtimeScheduleScene &scene) noexcept {
        const auto result = backend_.publish_spatial(scene);
        return result == PM_SUCCESS ? capture({
            RealtimeSchedulePhase::spatial_publish, 0U, 0U}) : result;
    }
    [[nodiscard]] PmResult solve_domain(
        RealtimeDomain domain, std::uint32_t iteration,
        const RealtimeScheduleScene &scene) noexcept {
        const auto result = backend_.solve_domain(domain, iteration, scene);
        return result == PM_SUCCESS ? capture({
            RealtimeSchedulePhase::domain_solve,
            static_cast<std::uint32_t>(domain), iteration}) : result;
    }
    [[nodiscard]] PmResult solve_couplings(
        RealtimeCouplingFamily family, std::uint32_t sweep,
        std::uint32_t edges, const RealtimeScheduleScene &scene) noexcept {
        const auto result = backend_.solve_couplings(
            family, sweep, edges, scene);
        return result == PM_SUCCESS ? capture({
            RealtimeSchedulePhase::coupling_solve,
            static_cast<std::uint32_t>(family), sweep}) : result;
    }
    [[nodiscard]] PmResult commit_tick(
        const RealtimeScheduleScene &scene) noexcept {
        if (!realtime_telemetry_progress_valid(
                scene_, ledger_, limits_, telemetry_, true))
            return PM_ERROR_CAPACITY_EXCEEDED;
        const auto result = backend_.commit_tick(scene);
        return result == PM_SUCCESS ? capture({
            RealtimeSchedulePhase::commit, 0U, 0U}) : result;
    }
    [[nodiscard]] PmResult abort_tick(
        const RealtimeScheduleScene &scene) noexcept {
        return backend_.abort_tick(scene);
    }

private:
    [[nodiscard]] PmResult capture(
        const RealtimeTelemetryOperation &operation) noexcept {
        RealtimeOperationMeasurement measurement{};
        const auto result = backend_.take_realtime_measurement(
            operation, measurement);
        if (result != PM_SUCCESS) return result;
        for (std::size_t index = 0U;
             index < realtime_telemetry_phase_count; ++index) {
            if (!realtime_telemetry_detail::add(
                    telemetry_.phases[index].work,
                    measurement.phases[index]) ||
                !realtime_telemetry_detail::add(
                    telemetry_.total, measurement.phases[index]))
                return PM_ERROR_CAPACITY_EXCEEDED;
        }
        return realtime_telemetry_progress_valid(
            scene_, ledger_, limits_, telemetry_)
            ? PM_SUCCESS : PM_ERROR_CAPACITY_EXCEEDED;
    }

    Backend &backend_;
    const RealtimeScheduleScene &scene_;
    RealtimeScheduleLedger &ledger_;
    const RealtimeTelemetryLimits &limits_;
    RealtimeTickTelemetry &telemetry_;
};

// Measured assembly is opt-in.  The original assemble_realtime_tick template
// neither requires telemetry methods nor performs a telemetry branch, query,
// readback, allocation, or wait.
template <typename Backend>
[[nodiscard]] PmResult assemble_realtime_tick_measured(
    Backend &backend, const RealtimeScheduleScene &scene,
    RealtimeScheduleLedger &ledger, const RealtimeTelemetryLimits &limits,
    RealtimeTickTelemetry &telemetry) noexcept {
    telemetry = {};
    telemetry.profile_version = limits.profile_version;
    telemetry.budget_version = limits.budget_version;
    telemetry.tick = scene.tick;
    auto result = backend.begin_realtime_telemetry(scene);
    if (result != PM_SUCCESS) return result;
    MeasuredRealtimeBackend<Backend> measured{
        backend, scene, ledger, limits, telemetry};
    result = assemble_realtime_tick(measured, scene, ledger);
    if (result != PM_SUCCESS) {
        (void)backend.abort_realtime_telemetry(scene);
        return result;
    }
    result = backend.end_realtime_telemetry(scene, telemetry);
    if (result != PM_SUCCESS) return result;
    return telemetry.total.submissions != 0U &&
        realtime_telemetry_progress_valid(
            scene, ledger, limits, telemetry, true)
        ? PM_SUCCESS : PM_ERROR_INVALID_STATE;
}

// Fixed-size query ownership for native adapters.  Full storage returns
// PM_NOT_READY; it never waits for or evicts an in-flight tick.
template <std::size_t SlotCount, std::uint32_t QueriesPerSlot>
class RealtimeTelemetryQueryRing {
    static_assert(SlotCount != 0U);
    static_assert(QueriesPerSlot != 0U);

public:
    struct Handle {
        std::uint32_t slot{};
        std::uint32_t generation{};
    };

    [[nodiscard]] PmResult begin(
        std::uint64_t tick, bool timing_supported, Handle &handle) noexcept {
        if (tick == 0U) return PM_ERROR_INVALID_ARGUMENT;
        for (std::size_t index = 0U; index < slots_.size(); ++index) {
            auto &slot = slots_[index];
            if (slot.state != State::empty) continue;
            ++slot.generation;
            if (slot.generation == 0U) ++slot.generation;
            slot.tick = tick;
            slot.query_count = 0U;
            slot.timing_supported = timing_supported;
            slot.state = State::recording;
            handle = {static_cast<std::uint32_t>(index), slot.generation};
            return PM_SUCCESS;
        }
        return PM_NOT_READY;
    }

    [[nodiscard]] PmResult record_queries(
        Handle handle, std::uint32_t count) noexcept {
        auto *slot = find(handle, State::recording);
        if (slot == nullptr || count > QueriesPerSlot - slot->query_count)
            return PM_ERROR_CAPACITY_EXCEEDED;
        if (slot->timing_supported) slot->query_count += count;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult submit(Handle handle) noexcept {
        auto *slot = find(handle, State::recording);
        if (slot == nullptr) return PM_ERROR_INVALID_STATE;
        slot->state = State::pending;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult resolve(
        Handle handle, bool ready,
        const std::array<std::uint64_t, realtime_telemetry_phase_count> &gpu_ns,
        RealtimeTickTelemetry &telemetry) noexcept {
        auto *slot = find(handle, State::pending);
        if (slot == nullptr) return PM_ERROR_INVALID_STATE;
        if (!ready) return PM_NOT_READY;
        if (telemetry.tick != slot->tick) return PM_ERROR_INVALID_ARGUMENT;
        for (std::size_t index = 0U;
             index < realtime_telemetry_phase_count; ++index) {
            telemetry.phases[index].timing.gpu_status =
                slot->timing_supported ? RealtimeGpuTimingStatus::valid
                                       : RealtimeGpuTimingStatus::unsupported;
            telemetry.phases[index].timing.gpu_ns =
                slot->timing_supported ? gpu_ns[index] : 0U;
        }
        if (slot->timing_supported) {
            telemetry.instrumentation.query_resets += 1U;
            telemetry.instrumentation.timestamp_queries += slot->query_count;
            telemetry.instrumentation.asynchronous_result_reads += 1U;
            telemetry.instrumentation.result_bytes +=
                static_cast<std::uint64_t>(slot->query_count) *
                sizeof(std::uint64_t);
        }
        *slot = Slot{.generation = slot->generation};
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult abort(Handle handle) noexcept {
        if (handle.slot >= slots_.size()) return PM_ERROR_INVALID_ARGUMENT;
        auto &slot = slots_[handle.slot];
        if (slot.state == State::empty || slot.generation != handle.generation)
            return PM_ERROR_INVALID_STATE;
        slot = Slot{.generation = slot.generation};
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::size_t pending() const noexcept {
        std::size_t result{};
        for (const auto &slot : slots_)
            result += slot.state == State::pending ? 1U : 0U;
        return result;
    }

private:
    enum class State : std::uint8_t { empty, recording, pending };
    struct Slot {
        std::uint64_t tick{};
        std::uint32_t query_count{};
        std::uint32_t generation{};
        State state{};
        bool timing_supported{};
    };

    [[nodiscard]] Slot *find(Handle handle, State state) noexcept {
        if (handle.slot >= slots_.size()) return nullptr;
        auto &slot = slots_[handle.slot];
        return slot.generation == handle.generation && slot.state == state
            ? &slot : nullptr;
    }

    std::array<Slot, SlotCount> slots_{};
};

} // namespace parallel_mater::simulation
