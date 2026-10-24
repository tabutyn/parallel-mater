// SPDX-License-Identifier: MIT
#include "simulation/liquid_coupling_realtime.hpp"

#include <array>
#include <cstdio>
#include <limits>

namespace {
using namespace parallel_mater::simulation;

int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "%s:%d check failed: %s\n", __FILE__, __LINE__, \
                 #condition); \
    ++failures; \
} } while (false)

LiquidCouplingRealtimeInput valid_contact() {
    LiquidCouplingRealtimeInput result{};
    result.family = LiquidCouplingFamily::cloth;
    result.mode = LiquidCouplingMode::contact;
    result.world_coordinate = 10'000.0F;
    result.characteristic_length = 0.1F;
    result.liquid_mass = 1.0F;
    result.partner_mass = 100.0F;
    result.linear_speed = 2.9F;
    result.angular_speed = 10.0F;
    result.acceleration_displacement = 0.025F;
    result.solver_correction = 0.025F;
    result.contact_distance = 0.01F;
    result.friction = 2.0F;
    return result;
}

void test_profile_modes_and_ranges() {
    CHECK(liquid_coupling_realtime_sweeps == 4U);
    CHECK(liquid_coupling_realtime_step_request_valid(
        liquid_coupling_realtime_timestep, 1U));
    CHECK(!liquid_coupling_realtime_step_request_valid(
        liquid_coupling_realtime_timestep, 2U));
    CHECK(!liquid_coupling_realtime_step_request_valid(std::nextafter(
        std::nextafter(liquid_coupling_realtime_timestep,
            std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity()), 1U));

    auto input = valid_contact();
    CHECK(validate_liquid_coupling_realtime_input(input) == PM_SUCCESS);
    input.partner_mass = 100.001F;
    CHECK(validate_liquid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_contact();
    input.linear_speed = 3.0001F;
    CHECK(validate_liquid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_contact();
    input.family = LiquidCouplingFamily::liquid;
    input.mode = LiquidCouplingMode::pressure;
    input.contact_distance = 0.05F;
    input.friction = 0.0F;
    input.response = 0.5F;
    input.liquid_material_signature = 17U;
    input.partner_material_signature = 17U;
    CHECK(validate_liquid_coupling_realtime_input(input) == PM_SUCCESS);
    input.partner_material_signature = 18U;
    CHECK(validate_liquid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);

    input = valid_contact();
    input.family = LiquidCouplingFamily::smoke_particle;
    input.mode = LiquidCouplingMode::drag;
    input.partner_mass = 0.0F;
    input.contact_distance = 0.05F;
    input.friction = 0.0F;
    input.response = 0.25F;
    CHECK(validate_liquid_coupling_realtime_input(input) == PM_SUCCESS);
    input.family = LiquidCouplingFamily::smoke_grid;
    CHECK(validate_liquid_coupling_realtime_input(input) == PM_SUCCESS);
    input.mode = LiquidCouplingMode::contact;
    CHECK(validate_liquid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);

    input = valid_contact();
    input.family = LiquidCouplingFamily::rigid;
    input.mode = LiquidCouplingMode::viscosity;
    input.contact_distance = 0.05F;
    input.friction = 0.0F;
    input.response = 0.5F;
    CHECK(validate_liquid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
}

constexpr std::array<LiquidCouplingRegistration, 16> registrations{{
    {1U, LiquidCouplingFamily::liquid, LiquidCouplingMode::contact, 1U, 1U, 1U},
    {2U, LiquidCouplingFamily::liquid, LiquidCouplingMode::contact, 1U, 1U, 1U},
    {3U, LiquidCouplingFamily::liquid, LiquidCouplingMode::pressure, 1U, 1U, 1U},
    {4U, LiquidCouplingFamily::liquid, LiquidCouplingMode::viscosity, 1U, 1U, 1U},
    {5U, LiquidCouplingFamily::rigid, LiquidCouplingMode::contact, 1U, 1U, 1U},
    {6U, LiquidCouplingFamily::rigid, LiquidCouplingMode::containment, 1U, 1U, 1U},
    {7U, LiquidCouplingFamily::cloth, LiquidCouplingMode::contact, 1U, 1U, 1U},
    {8U, LiquidCouplingFamily::cloth, LiquidCouplingMode::containment, 1U, 1U, 1U},
    {9U, LiquidCouplingFamily::soft_body, LiquidCouplingMode::contact, 1U, 1U, 1U},
    {10U, LiquidCouplingFamily::rope, LiquidCouplingMode::contact, 1U, 1U, 1U},
    {11U, LiquidCouplingFamily::smoke_particle, LiquidCouplingMode::drag, 1U, 1U, 1U},
    {12U, LiquidCouplingFamily::smoke_particle, LiquidCouplingMode::thermal, 1U, 1U, 1U},
    {13U, LiquidCouplingFamily::smoke_particle, LiquidCouplingMode::phase, 1U, 1U, 1U},
    {14U, LiquidCouplingFamily::smoke_grid, LiquidCouplingMode::drag, 1U, 1U, 1U},
    {15U, LiquidCouplingFamily::smoke_grid, LiquidCouplingMode::thermal, 1U, 1U, 1U},
    {16U, LiquidCouplingFamily::smoke_grid, LiquidCouplingMode::phase, 1U, 1U, 1U},
}};

void test_budget_epoch_order_and_rebinding() {
    LiquidCouplingRealtimeWorkBudget budget{};
    CHECK(make_liquid_coupling_realtime_work_budget(
        registrations, 4'096U, 4'128U, budget) == PM_SUCCESS);
    CHECK(budget.edge_count == 16U);
    CHECK(budget.constraint_count == 16U);
    CHECK(budget.coupling_sweeps_per_edge == 4U);
    CHECK(budget.edge_sweeps == 64U);
    CHECK(budget.constraint_visits == 64U);
    CHECK(budget.time_advances == 0U);
    CHECK(budget.spatial_updates_during_solve == 0U);
    CHECK(budget.batched_dispatches == 12U);
    CHECK(budget.batched_workgroups == 264U);
    CHECK(budget.intra_tick_host_readbacks == 0U);
    CHECK(budget.solver_scratch_bytes == 33'792U);
    std::printf(
        "F8_LIQUID_COUPLING_BUDGET edges=%u constraints=%u "
        "sweeps_per_edge=%u edge_sweeps=%llu constraint_visits=%llu "
        "time_advances=%u solve_index_updates=%u dispatches=%u "
        "workgroups=%llu host_readbacks=%u scratch_bytes=%zu\n",
        budget.edge_count, budget.constraint_count,
        budget.coupling_sweeps_per_edge,
        static_cast<unsigned long long>(budget.edge_sweeps),
        static_cast<unsigned long long>(budget.constraint_visits),
        budget.time_advances, budget.spatial_updates_during_solve,
        budget.batched_dispatches,
        static_cast<unsigned long long>(budget.batched_workgroups),
        budget.intra_tick_host_readbacks, budget.solver_scratch_bytes);

    const SpatialEpochToken token{73U, 9U, 14U, 1U, 1U, 4U};
    SpatialEpochCounters counters{};
    counters.refits = 1U;
    counters.publications = 1U;
    counters.queries = 8'224U;
    LiquidCouplingRealtimeTickContract tick{};
    CHECK(tick.begin(token, counters, budget, registrations, 23U) ==
          PM_SUCCESS);
    for (std::uint32_t sweep = 0U;
         sweep < liquid_coupling_realtime_sweeps; ++sweep)
        for (const auto &registration : registrations)
            CHECK(tick.record_edge(
                token, counters, 23U, registration) == PM_SUCCESS);
    CHECK(tick.completed_sweeps() == 4U);
    CHECK(tick.finish() == PM_SUCCESS);

    auto reordered = registrations;
    std::swap(reordered[0], reordered[1]);
    LiquidCouplingRealtimeWorkBudget reordered_budget{};
    CHECK(make_liquid_coupling_realtime_work_budget(
        reordered, 4'096U, 4'128U, reordered_budget) == PM_SUCCESS);
    LiquidCouplingRealtimeTickContract wrong_order{};
    CHECK(wrong_order.begin(token, counters, reordered_budget,
              reordered, 23U) == PM_ERROR_INVALID_STATE);

    LiquidCouplingRealtimeTickContract lifecycle{};
    CHECK(lifecycle.begin(token, counters, budget, registrations, 23U) ==
          PM_SUCCESS);
    auto rebound = registrations[0];
    ++rebound.liquid_generation;
    CHECK(lifecycle.record_edge(token, counters, 23U, rebound) ==
          PM_ERROR_INVALID_STATE);
    CHECK(lifecycle.record_edge(token, counters, 24U, registrations[0]) ==
          PM_ERROR_INVALID_STATE);
    CHECK(lifecycle.record_edge(token, counters, 23U, registrations[0]) ==
          PM_SUCCESS);

    LiquidCouplingRealtimeWorkBudget empty_budget{};
    const std::span<const LiquidCouplingRegistration> empty{};
    CHECK(make_liquid_coupling_realtime_work_budget(
        empty, 0U, 0U, empty_budget) == PM_SUCCESS);
    LiquidCouplingRealtimeTickContract empty_tick{};
    CHECK(empty_tick.begin(token, counters, empty_budget, empty, 23U) ==
          PM_SUCCESS);
    CHECK(empty_tick.finish() == PM_SUCCESS);
}

void test_boundary_staging_and_atomic_recovery() {
    LiquidCouplingBoundaryContract boundary{};
    CHECK(boundary.begin(7U, 11U, {8U, 4U}) == PM_SUCCESS);
    CHECK(boundary.stage({LiquidBoundaryMutationKind::source, 101U}) ==
          PM_SUCCESS);
    CHECK(boundary.stage({LiquidBoundaryMutationKind::outflow, 102U}) ==
          PM_SUCCESS);
    CHECK(boundary.stage({LiquidBoundaryMutationKind::phase, 103U}) ==
          PM_SUCCESS);
    CHECK(boundary.stage({LiquidBoundaryMutationKind::phase, 103U}) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(boundary.counts().source == 8U);
    CHECK(boundary.counts().destination == 4U);
    CHECK(boundary.seal_tick() == PM_SUCCESS);
    CHECK(boundary.apply_next_boundary(7U, 11U, 8U, 4U) ==
          PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(boundary.pending() == 3U);
    CHECK(boundary.counts().source == 8U);
    CHECK(boundary.counts().destination == 4U);
    CHECK(boundary.apply_next_boundary(7U, 11U, 8U, 5U) == PM_SUCCESS);
    CHECK(boundary.pending() == 0U);
    CHECK(boundary.applied_boundaries() == 1U);
    CHECK(boundary.counts().source == 7U);
    CHECK(boundary.counts().destination == 5U);
    std::puts("F8_LIQUID_BOUNDARY source=1 outflow=1 phase=1 "
              "visibility=next-boundary overflow=atomic recovery=retained");
}

} // namespace

int main() {
    test_profile_modes_and_ranges();
    test_budget_epoch_order_and_rebinding();
    test_boundary_staging_and_atomic_recovery();
    if (failures == 0)
        std::puts("F8-18 liquid coupling realtime contract tests passed");
    return failures == 0 ? 0 : 1;
}
