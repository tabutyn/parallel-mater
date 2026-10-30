// SPDX-License-Identifier: MIT
#include "simulation/rope_realtime.hpp"

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

RopeRealtimeInput valid_input() {
    RopeRealtimeInput result{};
    result.world_coordinate = 10'000.0F;
    result.node_spacing = 0.1F;
    result.radius = 0.05F;
    result.node_count = 72U;
    result.total_mass = 7.2F;
    result.interacting_mass_ratio = 100.0F;
    result.linear_speed = 2.9F;
    result.acceleration_displacement = 0.024F;
    result.solver_correction = 0.025F;
    result.authored_strain = 0.25F;
    result.effective_inverse_mass = 20.0F;
    result.structural_compliance = rope_realtime_timestep *
        rope_realtime_timestep * result.effective_inverse_mass;
    result.velocity_damping = 5.0F;
    result.contact_friction = 2.0F;
    return result;
}

void test_input_contract() {
    CHECK(rope_realtime_constraint_sweeps == 8U);
    CHECK(rope_realtime_contact_passes == 4U);
    CHECK(rope_realtime_step_request_valid(
        rope_realtime_timestep, 1U));
    CHECK(rope_realtime_step_request_valid(std::nextafter(
        rope_realtime_timestep, 0.0F), 1U));
    CHECK(rope_realtime_step_request_valid(std::nextafter(
        rope_realtime_timestep,
        std::numeric_limits<float>::infinity()), 1U));
    CHECK(!rope_realtime_step_request_valid(
        rope_realtime_timestep, 2U));
    CHECK(!rope_realtime_step_request_valid(std::nextafter(std::nextafter(
        rope_realtime_timestep,
        std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity()), 1U));

    auto input = valid_input();
    CHECK(validate_rope_realtime_input(input) == PM_SUCCESS);
    CHECK(std::abs(rope_realtime_compliance_response(
                       input.structural_compliance,
                       input.effective_inverse_mass) - 0.5F) < 1.0e-6F);

    const float denominator = rope_realtime_timestep *
        rope_realtime_timestep * input.effective_inverse_mass;
    input.structural_compliance = 19.0F * denominator;
    CHECK(std::abs(rope_realtime_compliance_response(
                       input.structural_compliance,
                       input.effective_inverse_mass) - 0.05F) < 1.0e-6F);
    CHECK(validate_rope_realtime_input(input) == PM_SUCCESS);
    input.structural_compliance = (19.0F + 1.0e-3F) * denominator;
    CHECK(validate_rope_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);

    input = valid_input();
    input.node_count = 1U;
    CHECK(validate_rope_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.radius = 0.049F;
    CHECK(validate_rope_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.linear_speed = 3.0001F;
    CHECK(validate_rope_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.solver_correction = 0.05001F;
    CHECK(validate_rope_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.authored_strain = 0.2501F;
    CHECK(validate_rope_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.solver_damping = 0.1201F;
    CHECK(validate_rope_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.structural_compliance = 0.0F;
    CHECK(validate_rope_realtime_input(input) == PM_SUCCESS);
}

void test_budget_and_epoch() {
    const auto budget = rope_realtime_work_budget(72U, 72U, 72U);
    CHECK(budget.segment_count == 71U);
    CHECK(budget.bend_constraint_count == 70U);
    CHECK(budget.constraint_sweeps == 8U);
    CHECK(budget.contact_passes == 4U);
    CHECK(budget.time_advances == 1U);
    CHECK(budget.spatial_updates_during_solve == 0U);
    CHECK(budget.domain_dispatches == 23U);
    CHECK(budget.domain_workgroups == 23U);
    CHECK(budget.stretch_constraint_visits == 568U);
    CHECK(budget.bend_constraint_visits == 1'120U);
    CHECK(budget.contact_visits == 288U);
    CHECK(budget.solver_scratch_bytes == 26'600U);
    std::printf(
        "F8_ROPE_60HZ_BUDGET nodes=%u capacity=%u segments=%u bends=%u "
        "contacts=%u sweeps=%u contact_passes=%u time_advances=%u "
        "solve_index_updates=%u dispatches=%u workgroups=%llu "
        "stretch_visits=%llu bend_visits=%llu contact_visits=%llu "
        "scratch_bytes=%zu\n",
        budget.live_nodes, budget.node_capacity, budget.segment_count,
        budget.bend_constraint_count, budget.contact_constraint_count,
        budget.constraint_sweeps, budget.contact_passes,
        budget.time_advances, budget.spatial_updates_during_solve,
        budget.domain_dispatches,
        static_cast<unsigned long long>(budget.domain_workgroups),
        static_cast<unsigned long long>(budget.stretch_constraint_visits),
        static_cast<unsigned long long>(budget.bend_constraint_visits),
        static_cast<unsigned long long>(budget.contact_visits),
        budget.solver_scratch_bytes);

    const SpatialEpochToken token{61U, 4U, 11U, 1U, 0U, 4U};
    SpatialEpochCounters counters{};
    counters.builds = 1U;
    counters.publications = 1U;
    counters.queries = 512U;
    RopeRealtimeTickContract tick{};
    CHECK(tick.begin(token, counters, budget, 23U) == PM_SUCCESS);
    CHECK(tick.record_advance(rope_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(tick.record_advance(rope_realtime_timestep, 1U) ==
          PM_ERROR_INVALID_STATE);
    for (std::uint32_t sweep = 0U;
         sweep < rope_realtime_constraint_sweeps; ++sweep)
        CHECK(tick.record_sweep(token, counters, 23U,
                  (sweep & 1U) != 0U) == PM_SUCCESS);
    CHECK(tick.advances() == 1U);
    CHECK(tick.sweeps() == 8U);
    CHECK(tick.contact_passes() == 4U);
    CHECK(tick.finish() == PM_SUCCESS);

    RopeRealtimeTickContract changed_topology{};
    CHECK(changed_topology.begin(token, counters, budget, 23U) == PM_SUCCESS);
    CHECK(changed_topology.record_advance(
              rope_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(changed_topology.record_sweep(
              token, counters, 24U, false) == PM_ERROR_INVALID_STATE);

    RopeRealtimeTickContract wrong_cadence{};
    CHECK(wrong_cadence.begin(token, counters, budget, 23U) == PM_SUCCESS);
    CHECK(wrong_cadence.record_advance(
              rope_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(wrong_cadence.record_sweep(
              token, counters, 23U, true) == PM_ERROR_INVALID_STATE);

    auto stale_counters = counters;
    ++stale_counters.refits;
    RopeRealtimeTickContract stale{};
    CHECK(stale.begin(token, counters, budget, 23U) == PM_SUCCESS);
    CHECK(stale.record_advance(rope_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(stale.record_sweep(token, counters, 23U, false) == PM_SUCCESS);
    CHECK(stale.record_sweep(token, stale_counters, 23U, true) ==
          PM_ERROR_INVALID_STATE);
}

} // namespace

int main() {
    test_input_contract();
    test_budget_and_epoch();
    if (failures == 0)
        std::puts("F8-15 rope realtime contract tests passed");
    return failures == 0 ? 0 : 1;
}
