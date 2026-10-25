// SPDX-License-Identifier: MIT
#pragma once

#include "cloth_realtime.hpp"
#include "deformable_smoke_coupling_realtime.hpp"
#include "fluid_realtime.hpp"
#include "liquid_coupling_realtime.hpp"
#include "rigid_coupling_realtime.hpp"
#include "rigid_realtime.hpp"
#include "rope_realtime.hpp"
#include "smoke_realtime.hpp"
#include "soft_body_realtime.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

namespace parallel_mater::simulation {

// Stable phase vocabulary shared by native adapters and schedule tests.  A
// backend may encode several kernels for one visit, but it may not advance the
// clock, publish another spatial epoch, wait for the host, or subdivide time.
enum class RealtimeSchedulePhase : std::uint8_t {
    boundary,
    predict,
    spatial_update,
    spatial_publish,
    domain_solve,
    coupling_solve,
    commit,
};

enum class RealtimeDomain : std::uint8_t {
    rigid,
    fluid,
    cloth,
    soft_body,
    rope,
    smoke,
    count,
};

enum class RealtimeCouplingFamily : std::uint8_t {
    rigid,
    liquid,
    deformable_smoke,
    count,
};

enum class RealtimeSpatialUpdate : std::uint8_t {
    none,
    build,
    refit,
};

inline constexpr std::size_t realtime_domain_count =
    static_cast<std::size_t>(RealtimeDomain::count);
inline constexpr std::size_t realtime_coupling_family_count =
    static_cast<std::size_t>(RealtimeCouplingFamily::count);

[[nodiscard]] constexpr std::size_t realtime_index(
    RealtimeDomain value) noexcept {
    return static_cast<std::size_t>(value);
}

[[nodiscard]] constexpr std::size_t realtime_index(
    RealtimeCouplingFamily value) noexcept {
    return static_cast<std::size_t>(value);
}

struct RealtimeScheduleScene {
    // Domain presence is host graph metadata. Element counts remain in the
    // backend's GPU buffers and never cross the schedule boundary.
    std::array<bool, realtime_domain_count> domains{};
    std::array<std::uint32_t, realtime_coupling_family_count> coupling_edges{};
    RealtimeSpatialUpdate spatial_update{RealtimeSpatialUpdate::build};
    SmokeRealtimeRepresentation smoke_representation{
        SmokeRealtimeRepresentation::particles};
    float timestep{rigid_realtime_timestep};
    std::uint32_t substeps{1U};
    std::uint64_t tick{};
    std::uint64_t graph_revision{};
    std::uint64_t resource_revision{};
    std::uint32_t input_snapshot{};
    std::uint32_t render_snapshot{};
};

struct RealtimeScheduleBudget {
    std::array<std::uint32_t, realtime_domain_count> domain_iterations{};
    std::array<std::uint32_t, realtime_coupling_family_count>
        coupling_sweeps{};
    std::uint32_t time_advances{1U};
    std::uint32_t spatial_updates{};
    std::uint32_t spatial_publications{1U};
    std::uint32_t intra_tick_host_waits{};
    std::uint32_t intra_tick_host_readbacks{};
};

[[nodiscard]] constexpr RealtimeScheduleBudget realtime_schedule_budget(
    const RealtimeScheduleScene &scene) noexcept {
    RealtimeScheduleBudget result{};
    result.domain_iterations[realtime_index(RealtimeDomain::rigid)] =
        !scene.domains[realtime_index(RealtimeDomain::rigid)]
            ? 0U : rigid_realtime_sweeps;
    result.domain_iterations[realtime_index(RealtimeDomain::fluid)] =
        !scene.domains[realtime_index(RealtimeDomain::fluid)]
            ? 0U : fluid_realtime_density_sweeps;
    result.domain_iterations[realtime_index(RealtimeDomain::cloth)] =
        !scene.domains[realtime_index(RealtimeDomain::cloth)]
            ? 0U : cloth_realtime_constraint_sweeps;
    result.domain_iterations[realtime_index(RealtimeDomain::soft_body)] =
        !scene.domains[realtime_index(RealtimeDomain::soft_body)]
            ? 0U : soft_body_realtime_constraint_sweeps;
    result.domain_iterations[realtime_index(RealtimeDomain::rope)] =
        !scene.domains[realtime_index(RealtimeDomain::rope)]
            ? 0U : rope_realtime_constraint_sweeps;
    result.domain_iterations[realtime_index(RealtimeDomain::smoke)] =
        !scene.domains[realtime_index(RealtimeDomain::smoke)]
            ? 0U
            : scene.smoke_representation ==
                    SmokeRealtimeRepresentation::particles
                ? smoke_realtime_particle_sweeps
                : smoke_realtime_grid_pressure_cycles;
    result.coupling_sweeps[realtime_index(RealtimeCouplingFamily::rigid)] =
        scene.coupling_edges[
            realtime_index(RealtimeCouplingFamily::rigid)] == 0U
            ? 0U : rigid_coupling_realtime_sweeps;
    result.coupling_sweeps[realtime_index(RealtimeCouplingFamily::liquid)] =
        scene.coupling_edges[
            realtime_index(RealtimeCouplingFamily::liquid)] == 0U
            ? 0U : liquid_coupling_realtime_sweeps;
    result.coupling_sweeps[
        realtime_index(RealtimeCouplingFamily::deformable_smoke)] =
        scene.coupling_edges[
            realtime_index(RealtimeCouplingFamily::deformable_smoke)] == 0U
            ? 0U : deformable_smoke_coupling_sweeps;
    result.spatial_updates = scene.spatial_update == RealtimeSpatialUpdate::none
        ? 0U : 1U;
    return result;
}

struct RealtimeScheduleLedger {
    std::array<std::uint32_t, realtime_domain_count> predictions{};
    std::array<std::uint32_t, realtime_domain_count> domain_iterations{};
    std::array<std::uint32_t, realtime_coupling_family_count>
        coupling_sweeps{};
    std::uint32_t boundary_passes{};
    std::uint32_t time_advances{};
    std::uint32_t spatial_updates{};
    std::uint32_t spatial_publications{};
    std::uint32_t commits{};
    std::uint32_t intra_tick_host_waits{};
    std::uint32_t intra_tick_host_readbacks{};
    bool failed{};
};

static_assert(std::is_trivially_copyable_v<RealtimeScheduleScene>);
static_assert(std::is_trivially_copyable_v<RealtimeScheduleBudget>);
static_assert(std::is_trivially_copyable_v<RealtimeScheduleLedger>);

[[nodiscard]] inline bool realtime_scene_valid(
    const RealtimeScheduleScene &scene) noexcept {
    if (scene.tick == 0U || scene.graph_revision == 0U ||
        scene.resource_revision == 0U || scene.input_snapshot == 0U ||
        scene.render_snapshot == 0U ||
        !rigid_realtime_step_request_valid(scene.timestep, scene.substeps) ||
        (scene.smoke_representation != SmokeRealtimeRepresentation::particles &&
         scene.smoke_representation != SmokeRealtimeRepresentation::grid))
        return false;
    bool has_domain = false;
    for (const bool present : scene.domains) has_domain |= present;
    if (has_domain && scene.spatial_update == RealtimeSpatialUpdate::none)
        return false;
    for (const auto edges : scene.coupling_edges)
        if (edges > rigid_coupling_realtime_maximum_edges) return false;
    return true;
}

[[nodiscard]] constexpr bool realtime_schedule_complete(
    const RealtimeScheduleScene &scene,
    const RealtimeScheduleLedger &ledger) noexcept {
    const auto budget = realtime_schedule_budget(scene);
    if (ledger.failed || ledger.boundary_passes != 1U ||
        ledger.time_advances != budget.time_advances ||
        ledger.spatial_updates != budget.spatial_updates ||
        ledger.spatial_publications != budget.spatial_publications ||
        ledger.commits != 1U ||
        ledger.intra_tick_host_waits != 0U ||
        ledger.intra_tick_host_readbacks != 0U ||
        ledger.domain_iterations != budget.domain_iterations ||
        ledger.coupling_sweeps != budget.coupling_sweeps)
        return false;
    for (std::size_t index = 0U; index < realtime_domain_count; ++index) {
        const auto expected = scene.domains[index] ? 1U : 0U;
        if (ledger.predictions[index] != expected) return false;
    }
    return true;
}

// Owns completed input/render generations without allocation.  Input is pinned
// for one pending tick.  Render state becomes visible only with atomic commit;
// a failed tick leaves both completed slots and pending input unchanged.
class RealtimeSnapshotOwnership {
public:
    [[nodiscard]] PmResult begin(std::uint32_t input_generation,
                                 std::uint32_t render_generation) noexcept {
        if (pending_ || input_generation == 0U || render_generation == 0U ||
            input_generation <= completed_input_ ||
            render_generation <= current_render_)
            return PM_ERROR_INVALID_STATE;
        pending_input_ = input_generation;
        pending_render_ = render_generation;
        pending_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult commit() noexcept {
        if (!pending_) return PM_ERROR_INVALID_STATE;
        completed_input_ = pending_input_;
        previous_render_ = current_render_;
        current_render_ = pending_render_;
        pending_input_ = 0U;
        pending_render_ = 0U;
        pending_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult abort() noexcept {
        if (!pending_) return PM_ERROR_INVALID_STATE;
        pending_input_ = 0U;
        pending_render_ = 0U;
        pending_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] constexpr std::uint32_t completed_input() const noexcept {
        return completed_input_;
    }
    [[nodiscard]] constexpr std::uint32_t previous_render() const noexcept {
        return previous_render_;
    }
    [[nodiscard]] constexpr std::uint32_t current_render() const noexcept {
        return current_render_;
    }
    [[nodiscard]] constexpr bool pending() const noexcept { return pending_; }

private:
    std::uint32_t completed_input_{};
    std::uint32_t previous_render_{};
    std::uint32_t current_render_{};
    std::uint32_t pending_input_{};
    std::uint32_t pending_render_{};
    bool pending_{};
};

// Backend contract:
//   begin_tick(scene), boundary(scene), predict(domain, scene),
//   update_spatial(kind, scene), publish_spatial(scene),
//   solve_domain(domain, iteration, scene),
//   solve_couplings(family, sweep, edge_count, scene), commit_tick(scene),
//   abort_tick(scene).
// Every operation returns PmResult.  Counts stay GPU-resident behind those
// calls; scheduler never asks backend to read a live count or wait mid-tick.
template <typename Backend>
[[nodiscard]] PmResult assemble_realtime_tick(
    Backend &backend, const RealtimeScheduleScene &scene,
    RealtimeScheduleLedger &ledger) noexcept {
    ledger = {};
    if (!realtime_scene_valid(scene)) return PM_ERROR_INVALID_ARGUMENT;

    bool began = false;
    const auto fail = [&](PmResult result) noexcept {
        ledger.failed = true;
        if (began) (void)backend.abort_tick(scene);
        return result;
    };
    auto result = backend.begin_tick(scene);
    if (result != PM_SUCCESS) return fail(result);
    began = true;

    result = backend.boundary(scene);
    if (result != PM_SUCCESS) return fail(result);
    ++ledger.boundary_passes;
    ++ledger.time_advances;

    for (std::size_t index = 0U; index < realtime_domain_count; ++index) {
        if (!scene.domains[index]) continue;
        const auto domain = static_cast<RealtimeDomain>(index);
        result = backend.predict(domain, scene);
        if (result != PM_SUCCESS) return fail(result);
        ++ledger.predictions[index];
    }

    if (scene.spatial_update != RealtimeSpatialUpdate::none) {
        result = backend.update_spatial(scene.spatial_update, scene);
        if (result != PM_SUCCESS) return fail(result);
        ++ledger.spatial_updates;
    }
    result = backend.publish_spatial(scene);
    if (result != PM_SUCCESS) return fail(result);
    ++ledger.spatial_publications;

    const auto budget = realtime_schedule_budget(scene);
    for (std::size_t index = 0U; index < realtime_domain_count; ++index) {
        const auto domain = static_cast<RealtimeDomain>(index);
        for (std::uint32_t iteration = 0U;
             iteration < budget.domain_iterations[index]; ++iteration) {
            result = backend.solve_domain(domain, iteration, scene);
            if (result != PM_SUCCESS) return fail(result);
            ++ledger.domain_iterations[index];
        }
    }
    for (std::size_t index = 0U;
         index < realtime_coupling_family_count; ++index) {
        const auto family = static_cast<RealtimeCouplingFamily>(index);
        for (std::uint32_t sweep = 0U;
             sweep < budget.coupling_sweeps[index]; ++sweep) {
            result = backend.solve_couplings(
                family, sweep, scene.coupling_edges[index], scene);
            if (result != PM_SUCCESS) return fail(result);
            ++ledger.coupling_sweeps[index];
        }
    }

    result = backend.commit_tick(scene);
    if (result != PM_SUCCESS) return fail(result);
    ++ledger.commits;
    return realtime_schedule_complete(scene, ledger)
        ? PM_SUCCESS : fail(PM_ERROR_INVALID_STATE);
}

} // namespace parallel_mater::simulation
