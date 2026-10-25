// SPDX-License-Identifier: MIT
#include "simulation/realtime_schedule.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace parallel_mater::simulation;

#define CHECK(value) do { if (!(value)) {                                  \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value);              \
    std::abort();                                                           \
} } while (false)

namespace {

struct Event {
    RealtimeSchedulePhase phase{};
    std::uint32_t first{};
    std::uint32_t second{};
};

struct RecordingBackend {
    std::array<Event, 256U> events{};
    std::size_t event_count{};
    std::size_t calls{};
    std::size_t fail_call{static_cast<std::size_t>(-1)};
    std::uint32_t aborts{};
    bool began{};

    PmResult record(RealtimeSchedulePhase phase, std::uint32_t first = 0U,
                    std::uint32_t second = 0U) noexcept {
        const auto call = calls++;
        if (call == fail_call) return PM_ERROR_INVALID_STATE;
        CHECK(event_count < events.size());
        events[event_count++] = {phase, first, second};
        return PM_SUCCESS;
    }

    PmResult begin_tick(const RealtimeScheduleScene &) noexcept {
        began = true;
        ++calls;
        return calls - 1U == fail_call ? PM_ERROR_INVALID_STATE : PM_SUCCESS;
    }
    PmResult boundary(const RealtimeScheduleScene &) noexcept {
        return record(RealtimeSchedulePhase::boundary);
    }
    PmResult predict(RealtimeDomain domain,
                     const RealtimeScheduleScene &) noexcept {
        return record(RealtimeSchedulePhase::predict,
                      static_cast<std::uint32_t>(domain));
    }
    PmResult update_spatial(RealtimeSpatialUpdate update,
                            const RealtimeScheduleScene &) noexcept {
        return record(RealtimeSchedulePhase::spatial_update,
                      static_cast<std::uint32_t>(update));
    }
    PmResult publish_spatial(const RealtimeScheduleScene &) noexcept {
        return record(RealtimeSchedulePhase::spatial_publish);
    }
    PmResult solve_domain(RealtimeDomain domain, std::uint32_t iteration,
                          const RealtimeScheduleScene &) noexcept {
        return record(RealtimeSchedulePhase::domain_solve,
                      static_cast<std::uint32_t>(domain), iteration);
    }
    PmResult solve_couplings(RealtimeCouplingFamily family,
                             std::uint32_t sweep, std::uint32_t edges,
                             const RealtimeScheduleScene &) noexcept {
        CHECK(edges != 0U);
        return record(RealtimeSchedulePhase::coupling_solve,
                      static_cast<std::uint32_t>(family), sweep);
    }
    PmResult commit_tick(const RealtimeScheduleScene &) noexcept {
        return record(RealtimeSchedulePhase::commit);
    }
    PmResult abort_tick(const RealtimeScheduleScene &) noexcept {
        ++aborts;
        return PM_SUCCESS;
    }
};

RealtimeScheduleScene scene() {
    RealtimeScheduleScene result{};
    result.tick = 1U;
    result.graph_revision = 2U;
    result.resource_revision = 3U;
    result.input_snapshot = 4U;
    result.render_snapshot = 5U;
    return result;
}

void check_order(const RecordingBackend &backend) {
    CHECK(backend.event_count != 0U);
    for (std::size_t index = 1U; index < backend.event_count; ++index)
        CHECK(backend.events[index - 1U].phase <= backend.events[index].phase);
}

void empty_tick() {
    auto input = scene();
    input.spatial_update = RealtimeSpatialUpdate::none;
    RecordingBackend backend{};
    RealtimeScheduleLedger ledger{};
    CHECK(assemble_realtime_tick(backend, input, ledger) == PM_SUCCESS);
    CHECK(realtime_schedule_complete(input, ledger));
    CHECK(ledger.boundary_passes == 1U);
    CHECK(ledger.time_advances == 1U);
    CHECK(ledger.spatial_updates == 0U);
    CHECK(ledger.spatial_publications == 1U);
    CHECK(ledger.commits == 1U);
    CHECK(backend.event_count == 3U);
    check_order(backend);
}

void every_single_domain() {
    for (std::size_t domain = 0U; domain < realtime_domain_count; ++domain) {
        auto input = scene();
        input.domains[domain] = true;
        input.spatial_update = RealtimeSpatialUpdate::refit;
        if (domain == realtime_index(RealtimeDomain::smoke))
            input.smoke_representation = SmokeRealtimeRepresentation::grid;
        RecordingBackend backend{};
        RealtimeScheduleLedger ledger{};
        CHECK(assemble_realtime_tick(backend, input, ledger) == PM_SUCCESS);
        CHECK(realtime_schedule_complete(input, ledger));
        CHECK(ledger.predictions[domain] == 1U);
        CHECK(ledger.spatial_updates == 1U);
        const auto budget = realtime_schedule_budget(input);
        CHECK(ledger.domain_iterations[domain] ==
              budget.domain_iterations[domain]);
        for (std::size_t other = 0U; other < realtime_domain_count; ++other)
            if (other != domain) CHECK(ledger.domain_iterations[other] == 0U);
        check_order(backend);
    }
}

void mixed_tick() {
    auto input = scene();
    input.tick = 99U;
    input.domains = {true, true, true, true, true, true};
    input.coupling_edges = {4U, 6U, 9U};
    RecordingBackend backend{};
    RealtimeScheduleLedger ledger{};
    CHECK(assemble_realtime_tick(backend, input, ledger) == PM_SUCCESS);
    CHECK(realtime_schedule_complete(input, ledger));
    constexpr std::array<std::uint32_t, realtime_domain_count>
        expected_domain_iterations{8U, 6U, 8U, 8U, 8U, 6U};
    constexpr std::array<std::uint32_t, realtime_coupling_family_count>
        expected_coupling_sweeps{4U, 4U, 4U};
    CHECK(ledger.domain_iterations == expected_domain_iterations);
    CHECK(ledger.coupling_sweeps == expected_coupling_sweeps);
    CHECK(ledger.time_advances == 1U);
    CHECK(ledger.spatial_updates == 1U);
    CHECK(ledger.spatial_publications == 1U);
    CHECK(ledger.intra_tick_host_waits == 0U);
    CHECK(ledger.intra_tick_host_readbacks == 0U);
    check_order(backend);
}

void failure_is_atomic() {
    auto input = scene();
    input.domains[realtime_index(RealtimeDomain::fluid)] = true;
    RecordingBackend begin_failure{};
    begin_failure.fail_call = 0U;
    RealtimeScheduleLedger ledger{};
    CHECK(assemble_realtime_tick(begin_failure, input, ledger) ==
          PM_ERROR_INVALID_STATE);
    CHECK(ledger.failed);
    CHECK(begin_failure.aborts == 0U);

    RecordingBackend backend{};
    // begin, boundary, predict, update, publication, then first solve fails.
    backend.fail_call = 5U;
    CHECK(assemble_realtime_tick(backend, input, ledger) ==
          PM_ERROR_INVALID_STATE);
    CHECK(ledger.failed);
    CHECK(ledger.commits == 0U);
    CHECK(backend.aborts == 1U);
}

void snapshot_ownership() {
    RealtimeSnapshotOwnership ownership{};
    CHECK(ownership.begin(1U, 1U) == PM_SUCCESS);
    CHECK(ownership.pending());
    CHECK(ownership.commit() == PM_SUCCESS);
    CHECK(ownership.completed_input() == 1U);
    CHECK(ownership.previous_render() == 0U);
    CHECK(ownership.current_render() == 1U);
    CHECK(ownership.begin(2U, 2U) == PM_SUCCESS);
    CHECK(ownership.abort() == PM_SUCCESS);
    CHECK(ownership.completed_input() == 1U);
    CHECK(ownership.current_render() == 1U);
    CHECK(ownership.begin(2U, 2U) == PM_SUCCESS);
    CHECK(ownership.commit() == PM_SUCCESS);
    CHECK(ownership.completed_input() == 2U);
    CHECK(ownership.previous_render() == 1U);
    CHECK(ownership.current_render() == 2U);
    CHECK(ownership.begin(2U, 3U) == PM_ERROR_INVALID_STATE);
}

void invalid_scene_rejected_before_recording() {
    auto input = scene();
    input.tick = 0U;
    RecordingBackend backend{};
    RealtimeScheduleLedger ledger{};
    CHECK(assemble_realtime_tick(backend, input, ledger) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(!backend.began);
    CHECK(backend.calls == 0U);
    input = scene();
    input.coupling_edges[0] = rigid_coupling_realtime_maximum_edges + 1U;
    CHECK(assemble_realtime_tick(backend, input, ledger) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = scene();
    input.domains[0] = true;
    input.spatial_update = RealtimeSpatialUpdate::none;
    CHECK(assemble_realtime_tick(backend, input, ledger) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = scene();
    input.substeps = 2U;
    CHECK(assemble_realtime_tick(backend, input, ledger) ==
          PM_ERROR_INVALID_ARGUMENT);
}

void sustained_allocation_free_window() {
    auto input = scene();
    input.domains = {true, true, true, true, true, true};
    input.coupling_edges = {4U, 4U, 4U};
    RecordingBackend backend{};
    RealtimeScheduleLedger ledger{};
    const auto tick = [&] {
        backend.event_count = 0U;
        backend.calls = 0U;
        backend.aborts = 0U;
        CHECK(assemble_realtime_tick(backend, input, ledger) == PM_SUCCESS);
    };
    for (std::uint32_t warmup = 0U; warmup < 600U; ++warmup) tick();
    const auto start = std::chrono::steady_clock::now();
    for (std::uint32_t measured = 0U; measured < 7'200U; ++measured) tick();
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const double simulated_seconds = 7'200.0 / 60.0;
    const double simulated_to_wall = simulated_seconds / seconds;
    CHECK(simulated_to_wall >= 0.99);
    std::printf("F8_SCHEDULE_ORCHESTRATION ticks=7200 dt=%.9g "
                "wall_seconds=%.9g simulated_to_wall=%.9g\n",
                input.timestep, seconds, simulated_to_wall);
}

} // namespace

int main() {
    empty_tick();
    every_single_domain();
    mixed_tick();
    failure_is_atomic();
    snapshot_ownership();
    invalid_scene_rejected_before_recording();
    sustained_allocation_free_window();
    std::puts("PASS: realtime schedule assembles one atomic bounded tick");
}
