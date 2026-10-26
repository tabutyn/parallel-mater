// SPDX-License-Identifier: MIT
#include "simulation/rigid_coupling_realtime.hpp"

#include <array>
#include <cmath>
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

RigidCouplingRealtimeInput valid_contact() {
    RigidCouplingRealtimeInput result{};
    result.family = RigidCouplingFamily::cloth;
    result.mode = RigidCouplingMode::contact;
    result.world_coordinate = 10'000.0F;
    result.characteristic_length = 0.1F;
    result.endpoint_mass = 1.0F;
    result.rigid_mass = 100.0F;
    result.linear_speed = 2.9F;
    result.angular_speed = 10.0F;
    result.acceleration_displacement = 0.025F;
    result.solver_correction = 0.025F;
    result.contact_distance = 0.01F;
    result.friction = 2.0F;
    return result;
}

void test_profile_and_modes() {
    CHECK(rigid_coupling_realtime_sweeps == 4U);
    CHECK(rigid_coupling_realtime_step_request_valid(
        rigid_coupling_realtime_timestep, 1U));
    CHECK(!rigid_coupling_realtime_step_request_valid(
        rigid_coupling_realtime_timestep, 2U));
    CHECK(!rigid_coupling_realtime_step_request_valid(std::nextafter(
        std::nextafter(rigid_coupling_realtime_timestep,
            std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity()), 1U));

    auto input = valid_contact();
    CHECK(validate_rigid_coupling_realtime_input(input) == PM_SUCCESS);
    input.rigid_mass = 100.001F;
    CHECK(validate_rigid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_contact();
    input.linear_speed = 3.0001F;
    CHECK(validate_rigid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_contact();
    input.solver_correction = 0.02501F;
    CHECK(validate_rigid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_contact();
    input.family = RigidCouplingFamily::cloth;
    input.mode = RigidCouplingMode::drag;
    input.contact_distance = 0.0F;
    input.friction = 0.0F;
    input.response = 0.5F;
    CHECK(validate_rigid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input.family = RigidCouplingFamily::rope;
    input.mode = RigidCouplingMode::attachment;
    CHECK(validate_rigid_coupling_realtime_input(input) == PM_SUCCESS);
    input.family = RigidCouplingFamily::smoke_particle;
    input.mode = RigidCouplingMode::drag;
    CHECK(validate_rigid_coupling_realtime_input(input) == PM_SUCCESS);
    input.family = RigidCouplingFamily::smoke_grid;
    CHECK(validate_rigid_coupling_realtime_input(input) == PM_SUCCESS);
    input.mode = RigidCouplingMode::attachment;
    CHECK(validate_rigid_coupling_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
}

constexpr std::array<RigidCouplingRegistration, 16> registrations{{
    {1U, RigidCouplingFamily::cloth, RigidCouplingMode::contact, 1U, 1U, 1U},
    {2U, RigidCouplingFamily::cloth, RigidCouplingMode::contact, 1U, 1U, 1U},
    {3U, RigidCouplingFamily::soft_body, RigidCouplingMode::contact, 1U, 1U, 1U},
    {4U, RigidCouplingFamily::soft_body, RigidCouplingMode::contact, 1U, 1U, 1U},
    {5U, RigidCouplingFamily::rope, RigidCouplingMode::contact, 1U, 1U, 1U},
    {6U, RigidCouplingFamily::rope, RigidCouplingMode::contact, 1U, 1U, 1U},
    {7U, RigidCouplingFamily::rope, RigidCouplingMode::attachment, 1U, 1U, 1U},
    {8U, RigidCouplingFamily::rope, RigidCouplingMode::attachment, 1U, 1U, 1U},
    {9U, RigidCouplingFamily::smoke_particle, RigidCouplingMode::contact, 1U, 1U, 1U},
    {10U, RigidCouplingFamily::smoke_particle, RigidCouplingMode::contact, 1U, 1U, 1U},
    {11U, RigidCouplingFamily::smoke_particle, RigidCouplingMode::drag, 1U, 1U, 1U},
    {12U, RigidCouplingFamily::smoke_particle, RigidCouplingMode::drag, 1U, 1U, 1U},
    {13U, RigidCouplingFamily::smoke_grid, RigidCouplingMode::contact, 1U, 1U, 1U},
    {14U, RigidCouplingFamily::smoke_grid, RigidCouplingMode::contact, 1U, 1U, 1U},
    {15U, RigidCouplingFamily::smoke_grid, RigidCouplingMode::drag, 1U, 1U, 1U},
    {16U, RigidCouplingFamily::smoke_grid, RigidCouplingMode::drag, 1U, 1U, 1U},
}};

void test_budget_epoch_order_and_rebinding() {
    RigidCouplingRealtimeWorkBudget budget{};
    CHECK(make_rigid_coupling_realtime_work_budget(
        registrations, 6'062U, 32U, budget) == PM_SUCCESS);
    CHECK(budget.edge_count == 16U);
    CHECK(budget.constraint_count == 16U);
    CHECK(budget.coupling_sweeps_per_edge == 4U);
    CHECK(budget.edge_sweeps == 64U);
    CHECK(budget.constraint_visits == 64U);
    CHECK(budget.time_advances == 0U);
    CHECK(budget.spatial_updates_during_solve == 0U);
    CHECK(budget.batched_dispatches == 12U);
    CHECK(budget.batched_workgroups == 200U);
    CHECK(budget.intra_tick_host_readbacks == 0U);
    CHECK(budget.solver_scratch_bytes == 25'144U);
    for (std::size_t index = 0U; index < registrations.size(); ++index) {
        CHECK(budget.operators[index].algebraic_sweeps == 4U);
        CHECK(budget.operators[index].constraint_visits == 4U);
    }

    auto multi_endpoint = registrations;
    multi_endpoint[6].constraint_count = 2U;
    multi_endpoint[7].constraint_count = 2U;
    RigidCouplingRealtimeWorkBudget multi_endpoint_budget{};
    CHECK(make_rigid_coupling_realtime_work_budget(
        multi_endpoint, 6'062U, 32U, multi_endpoint_budget) == PM_SUCCESS);
    CHECK(multi_endpoint_budget.constraint_count == 18U);
    CHECK(multi_endpoint_budget.batched_dispatches == 12U);
    CHECK(multi_endpoint_budget.batched_dispatches <
        3U * multi_endpoint_budget.constraint_count *
            rigid_coupling_realtime_sweeps);
    std::printf(
        "F8_RIGID_COUPLING_BUDGET edges=%u constraints=%u "
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

    const SpatialEpochToken token{71U, 8U, 13U, 1U, 1U, 4U};
    SpatialEpochCounters counters{};
    counters.refits = 1U;
    counters.publications = 1U;
    counters.queries = 4'096U;
    RigidCouplingRealtimeTickContract tick{};
    CHECK(tick.begin(token, counters, budget, registrations, 19U) ==
          PM_SUCCESS);
    for (std::uint32_t sweep = 0U;
         sweep < rigid_coupling_realtime_sweeps; ++sweep)
        for (const auto &registration : registrations)
            CHECK(tick.record_edge(
                token, counters, 19U, registration) == PM_SUCCESS);
    CHECK(tick.completed_sweeps() == 4U);
    CHECK(tick.finish() == PM_SUCCESS);

    auto reordered = registrations;
    std::swap(reordered[0], reordered[1]);
    RigidCouplingRealtimeWorkBudget reordered_budget{};
    CHECK(make_rigid_coupling_realtime_work_budget(
        reordered, 6'062U, 32U, reordered_budget) == PM_SUCCESS);
    RigidCouplingRealtimeTickContract wrong_order{};
    CHECK(wrong_order.begin(token, counters, reordered_budget,
              reordered, 19U) == PM_ERROR_INVALID_STATE);

    RigidCouplingRealtimeTickContract lifecycle{};
    CHECK(lifecycle.begin(token, counters, budget, registrations, 19U) ==
          PM_SUCCESS);
    auto rebound = registrations[0];
    ++rebound.endpoint_generation;
    CHECK(lifecycle.record_edge(token, counters, 19U, rebound) ==
          PM_ERROR_INVALID_STATE);
    CHECK(lifecycle.record_edge(token, counters, 20U, registrations[0]) ==
          PM_ERROR_INVALID_STATE);
    CHECK(lifecycle.record_edge(token, counters, 19U, registrations[0]) ==
          PM_SUCCESS);
    auto changed = counters;
    ++changed.builds;
    CHECK(lifecycle.record_edge(token, changed, 19U, registrations[1]) ==
          PM_ERROR_INVALID_STATE);

    RigidCouplingRealtimeWorkBudget empty_budget{};
    const std::span<const RigidCouplingRegistration> empty{};
    CHECK(make_rigid_coupling_realtime_work_budget(
        empty, 0U, 0U, empty_budget) == PM_SUCCESS);
    RigidCouplingRealtimeTickContract empty_tick{};
    CHECK(empty_tick.begin(token, counters, empty_budget, empty, 19U) ==
          PM_SUCCESS);
    CHECK(empty_tick.finish() == PM_SUCCESS);
}

} // namespace

int main() {
    test_profile_and_modes();
    test_budget_epoch_order_and_rebinding();
    if (failures == 0)
        std::puts("F8-17 rigid coupling realtime contract tests passed");
    return failures == 0 ? 0 : 1;
}
