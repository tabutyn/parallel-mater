// SPDX-License-Identifier: MIT
#include "simulation/realtime_telemetry.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>

using namespace parallel_mater::simulation;

#define CHECK(value) do { if (!(value)) {                                  \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value);              \
    std::abort();                                                           \
} } while (false)

namespace {

constexpr RealtimeWorkCounters generous_maximum() {
    return {
        .dispatches = 10'000U,
        .workgroups = 1'000'000U,
        .bytes_moved = 1'000'000'000U,
        .index_builds = 1U,
        .index_refits = 1U,
        .index_publications = 1U,
        .numerical_iterations = 1'000U,
        .barriers = 10'000U,
        .submissions = 8U,
        .readbacks = 0U,
        .host_waits = 0U,
    };
}

RealtimeTelemetryLimits limits() {
    RealtimeTelemetryLimits result{};
    result.total_maximum = generous_maximum();
    result.phase_maximum.fill(generous_maximum());
    result.timestamp_query_capacity = 32U;
    result.instrumentation_result_bytes_maximum = 256U;
    return result;
}

RealtimeScheduleScene scene() {
    RealtimeScheduleScene result{};
    result.domains = {true, true, true, true, true, true};
    result.coupling_edges = {2U, 3U, 4U};
    result.spatial_update = RealtimeSpatialUpdate::build;
    result.tick = 41U;
    result.graph_revision = 2U;
    result.resource_revision = 3U;
    result.input_snapshot = 4U;
    result.render_snapshot = 5U;
    return result;
}

struct MeasuredBackend {
    using QueryRing = RealtimeTelemetryQueryRing<3U, 32U>;

    RealtimeOperationMeasurement pending{};
    QueryRing queries{};
    QueryRing::Handle query_handle{};
    std::uint32_t telemetry_begins{};
    std::uint32_t telemetry_ends{};
    std::uint32_t telemetry_aborts{};
    std::uint32_t measurements{};
    std::uint32_t schedule_aborts{};
    bool timing_supported{true};
    bool extra_iteration{};
    bool extra_rebuild{};
    bool extra_dispatch{};

    RealtimeWorkCounters &phase(RealtimeTelemetryPhase value) noexcept {
        return pending.phases[realtime_index(value)];
    }
    void dispatch(RealtimeTelemetryPhase value, std::uint64_t groups = 1U) {
        auto &work = phase(value);
        ++work.dispatches;
        work.workgroups += groups;
        ++work.barriers;
    }

    PmResult begin_realtime_telemetry(
        const RealtimeScheduleScene &input) noexcept {
        ++telemetry_begins;
        pending = {};
        return queries.begin(input.tick, timing_supported, query_handle);
    }
    PmResult take_realtime_measurement(
        const RealtimeTelemetryOperation &,
        RealtimeOperationMeasurement &measurement) noexcept {
        ++measurements;
        measurement = pending;
        pending = {};
        return PM_SUCCESS;
    }
    PmResult end_realtime_telemetry(
        const RealtimeScheduleScene &,
        RealtimeTickTelemetry &telemetry) noexcept {
        ++telemetry_ends;
        for (auto &entry : telemetry.phases) {
            if (entry.work.dispatches != 0U) {
                entry.timing.cpu_encode_ns = 100U;
                entry.timing.cpu_submit_ns = 10U;
                entry.timing.cpu_wait_ns = 20U;
            }
            entry.timing.gpu_status = timing_supported
                ? RealtimeGpuTimingStatus::pending
                : RealtimeGpuTimingStatus::unsupported;
        }
        const auto recorded = queries.record_queries(
            query_handle, 2U * realtime_telemetry_phase_count);
        return recorded == PM_SUCCESS ? queries.submit(query_handle) : recorded;
    }
    PmResult abort_realtime_telemetry(
        const RealtimeScheduleScene &) noexcept {
        ++telemetry_aborts;
        return queries.abort(query_handle);
    }
    PmResult resolve(bool ready, RealtimeTickTelemetry &telemetry) noexcept {
        std::array<std::uint64_t, realtime_telemetry_phase_count> gpu_ns{};
        for (std::size_t index = 0U; index < gpu_ns.size(); ++index)
            gpu_ns[index] = 1'000U + index;
        return queries.resolve(query_handle, ready, gpu_ns, telemetry);
    }

    PmResult begin_tick(const RealtimeScheduleScene &) noexcept {
        dispatch(RealtimeTelemetryPhase::boundary);
        return PM_SUCCESS;
    }
    PmResult boundary(const RealtimeScheduleScene &) noexcept {
        dispatch(RealtimeTelemetryPhase::boundary);
        phase(RealtimeTelemetryPhase::boundary).bytes_moved += 64U;
        return PM_SUCCESS;
    }
    PmResult predict(RealtimeDomain,
                     const RealtimeScheduleScene &) noexcept {
        dispatch(RealtimeTelemetryPhase::predict, 2U);
        return PM_SUCCESS;
    }
    PmResult update_spatial(RealtimeSpatialUpdate update,
                            const RealtimeScheduleScene &) noexcept {
        dispatch(RealtimeTelemetryPhase::index, 8U);
        auto &index = phase(RealtimeTelemetryPhase::index);
        index.index_builds += update == RealtimeSpatialUpdate::build
            ? (extra_rebuild ? 2U : 1U) : 0U;
        index.index_refits += update == RealtimeSpatialUpdate::refit
            ? (extra_rebuild ? 2U : 1U) : 0U;
        dispatch(RealtimeTelemetryPhase::candidates, 12U);
        return PM_SUCCESS;
    }
    PmResult publish_spatial(const RealtimeScheduleScene &) noexcept {
        ++phase(RealtimeTelemetryPhase::index).index_publications;
        return PM_SUCCESS;
    }
    PmResult solve_domain(RealtimeDomain, std::uint32_t,
                          const RealtimeScheduleScene &) noexcept {
        dispatch(RealtimeTelemetryPhase::domain, 4U);
        phase(RealtimeTelemetryPhase::domain).numerical_iterations +=
            extra_iteration ? 2U : 1U;
        if (extra_dispatch)
            ++phase(RealtimeTelemetryPhase::domain).dispatches;
        return PM_SUCCESS;
    }
    PmResult solve_couplings(RealtimeCouplingFamily, std::uint32_t,
                             std::uint32_t,
                             const RealtimeScheduleScene &) noexcept {
        dispatch(RealtimeTelemetryPhase::coupling, 3U);
        ++phase(RealtimeTelemetryPhase::coupling).numerical_iterations;
        return PM_SUCCESS;
    }
    PmResult commit_tick(const RealtimeScheduleScene &) noexcept {
        dispatch(RealtimeTelemetryPhase::commit);
        ++phase(RealtimeTelemetryPhase::commit).submissions;
        return PM_SUCCESS;
    }
    PmResult abort_tick(const RealtimeScheduleScene &) noexcept {
        ++schedule_aborts;
        return PM_SUCCESS;
    }
};

void measured_tick_is_bounded_and_async() {
    auto input = scene();
    auto budget = limits();
    MeasuredBackend backend{};
    RealtimeScheduleLedger ledger{};
    RealtimeTickTelemetry telemetry{};
    CHECK(assemble_realtime_tick_measured(
        backend, input, ledger, budget, telemetry) == PM_SUCCESS);
    CHECK(realtime_schedule_complete(input, ledger));
    CHECK(backend.telemetry_begins == 1U);
    CHECK(backend.telemetry_ends == 1U);
    CHECK(backend.telemetry_aborts == 0U);
    CHECK(backend.queries.pending() == 1U);
    CHECK(telemetry.total.index_builds == 1U);
    CHECK(telemetry.total.index_refits == 0U);
    CHECK(telemetry.total.index_publications == 1U);
    CHECK(telemetry.total.numerical_iterations == 56U);
    CHECK(telemetry.total.submissions == 1U);
    CHECK(telemetry.total.readbacks == 0U);
    CHECK(telemetry.total.host_waits == 0U);
    CHECK(!realtime_telemetry_timing_ready(telemetry));
    CHECK(backend.resolve(false, telemetry) == PM_NOT_READY);
    CHECK(telemetry.instrumentation.forced_waits == 0U);
    CHECK(backend.resolve(true, telemetry) == PM_SUCCESS);
    CHECK(backend.queries.pending() == 0U);
    CHECK(telemetry.instrumentation.timestamp_queries == 14U);
    CHECK(telemetry.instrumentation.asynchronous_result_reads == 1U);
    CHECK(telemetry.instrumentation.result_bytes == 112U);
    CHECK(realtime_telemetry_report_valid(
        input, ledger, budget, telemetry));
}

void unsupported_timing_is_explicit() {
    auto input = scene();
    auto budget = limits();
    MeasuredBackend backend{};
    backend.timing_supported = false;
    RealtimeScheduleLedger ledger{};
    RealtimeTickTelemetry telemetry{};
    CHECK(assemble_realtime_tick_measured(
        backend, input, ledger, budget, telemetry) == PM_SUCCESS);
    CHECK(backend.resolve(true, telemetry) == PM_SUCCESS);
    for (const auto &entry : telemetry.phases) {
        CHECK(entry.timing.gpu_status ==
              RealtimeGpuTimingStatus::unsupported);
        CHECK(entry.timing.gpu_ns == 0U);
    }
    CHECK(telemetry.instrumentation.timestamp_queries == 0U);
    CHECK(telemetry.instrumentation.result_bytes == 0U);
    CHECK(realtime_telemetry_report_valid(
        input, ledger, budget, telemetry));
}

void disabled_telemetry_has_no_adapter_cost() {
    auto input = scene();
    MeasuredBackend backend{};
    RealtimeScheduleLedger ledger{};
    CHECK(assemble_realtime_tick(backend, input, ledger) == PM_SUCCESS);
    CHECK(backend.telemetry_begins == 0U);
    CHECK(backend.telemetry_ends == 0U);
    CHECK(backend.telemetry_aborts == 0U);
    CHECK(backend.measurements == 0U);
    CHECK(backend.queries.pending() == 0U);
}

void regressions_fail_before_commit() {
    const auto input = scene();
    const auto budget = limits();
    for (const auto kind : {1U, 2U, 3U}) {
        MeasuredBackend backend{};
        backend.extra_iteration = kind == 1U;
        backend.extra_rebuild = kind == 2U;
        backend.extra_dispatch = kind == 3U;
        auto strict = budget;
        if (kind == 3U) {
            strict.total_maximum.dispatches = 4U;
            strict.phase_maximum[realtime_index(
                RealtimeTelemetryPhase::domain)].dispatches = 1U;
        }
        RealtimeScheduleLedger ledger{};
        RealtimeTickTelemetry telemetry{};
        CHECK(assemble_realtime_tick_measured(
            backend, input, ledger, strict, telemetry) ==
            PM_ERROR_CAPACITY_EXCEEDED);
        CHECK(ledger.commits == 0U);
        CHECK(backend.schedule_aborts == 1U);
        CHECK(backend.telemetry_aborts == 1U);
    }
}

void query_storage_never_waits_or_evicts() {
    using Ring = RealtimeTelemetryQueryRing<2U, 4U>;
    Ring ring{};
    Ring::Handle first{}, second{}, blocked{};
    CHECK(ring.begin(1U, true, first) == PM_SUCCESS);
    CHECK(ring.record_queries(first, 5U) == PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(ring.record_queries(first, 4U) == PM_SUCCESS);
    CHECK(ring.submit(first) == PM_SUCCESS);
    CHECK(ring.begin(2U, true, second) == PM_SUCCESS);
    CHECK(ring.submit(second) == PM_SUCCESS);
    CHECK(ring.begin(3U, true, blocked) == PM_NOT_READY);
    RealtimeTickTelemetry telemetry{};
    telemetry.tick = 1U;
    std::array<std::uint64_t, realtime_telemetry_phase_count> durations{};
    CHECK(ring.resolve(first, false, durations, telemetry) == PM_NOT_READY);
    CHECK(ring.pending() == 2U);
    CHECK(ring.resolve(first, true, durations, telemetry) == PM_SUCCESS);
    CHECK(ring.pending() == 1U);
    CHECK(ring.begin(3U, true, blocked) == PM_SUCCESS);
}

} // namespace

int main() {
    measured_tick_is_bounded_and_async();
    unsupported_timing_is_explicit();
    disabled_telemetry_has_no_adapter_cost();
    regressions_fail_before_commit();
    query_storage_never_waits_or_evicts();
    std::puts("PASS: realtime telemetry measures bounded work without stalls");
}
