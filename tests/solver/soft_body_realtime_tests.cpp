// SPDX-License-Identifier: MIT
#include "simulation/soft_body_realtime.hpp"

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

SoftBodyRealtimeInput valid_input() {
    SoftBodyRealtimeInput result{};
    result.world_coordinate = 10'000.0F;
    result.characteristic_length = 0.1F;
    result.mass = 0.02F;
    result.interacting_mass_ratio = 100.0F;
    result.linear_speed = 2.9F;
    result.acceleration_displacement = 0.024F;
    result.solver_correction = 0.025F;
    result.authored_strain = 0.25F;
    result.effective_inverse_mass = 100.0F;
    result.stretch_compliance = soft_body_realtime_timestep *
        soft_body_realtime_timestep * result.effective_inverse_mass;
    result.node_radius = 0.025F;
    result.velocity_damping = 5.0F;
    result.spring_damping = 0.0F;
    result.contact_friction = 2.0F;
    result.shape_matching_stiffness = 0.995F;
    result.rest_shape_determinant = 1.0e-6F;
    return result;
}

void test_input_contract() {
    CHECK(soft_body_realtime_constraint_sweeps == 8U);
    CHECK(soft_body_realtime_contact_passes == 4U);
    CHECK(soft_body_realtime_step_request_valid(
        soft_body_realtime_timestep, 1U));
    CHECK(!soft_body_realtime_step_request_valid(
        soft_body_realtime_timestep, 2U));
    CHECK(!soft_body_realtime_step_request_valid(std::nextafter(std::nextafter(
        soft_body_realtime_timestep,
        std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity()), 1U));

    auto input = valid_input();
    CHECK(validate_soft_body_realtime_input(input) == PM_SUCCESS);
    CHECK(std::abs(soft_body_realtime_compliance_response(
                       input.stretch_compliance,
                       input.effective_inverse_mass) - 0.5F) < 1.0e-6F);
    input.interacting_mass_ratio = 100.01F;
    CHECK(validate_soft_body_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.rest_shape_determinant = -1.0F;
    CHECK(validate_soft_body_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.shape_matching_stiffness = 1.001F;
    CHECK(validate_soft_body_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.shape_matching_stiffness = 0.049F;
    CHECK(validate_soft_body_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.spring_damping = 0.001F;
    CHECK(validate_soft_body_realtime_input(input) ==
          PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.stretch_compliance = 0.0F;
    CHECK(validate_soft_body_realtime_input(input) == PM_SUCCESS);
}

void test_budget_and_epoch() {
    const auto budget = soft_body_realtime_work_budget(
        912U, 912U, 130'033U, 912U);
    CHECK(budget.constraint_sweeps == 8U);
    CHECK(budget.contact_passes == 4U);
    CHECK(budget.time_advances == 1U);
    CHECK(budget.spatial_updates_during_solve == 0U);
    CHECK(budget.domain_dispatches == 30U);
    CHECK(budget.domain_workgroups == 184U);
    CHECK(budget.bond_visits == 1'040'264U);
    CHECK(budget.contact_visits == 3'648U);
    CHECK(budget.solver_scratch_bytes == 25'560U);
    std::printf(
        "F8_SOFT_BODY_60HZ_BUDGET nodes=%u capacity=%u directed_bonds=%u "
        "contacts=%u sweeps=%u contact_passes=%u time_advances=%u "
        "solve_index_updates=%u dispatches=%u workgroups=%llu "
        "bond_visits=%llu contact_visits=%llu scratch_bytes=%zu\n",
        budget.live_nodes, budget.node_capacity, budget.directed_bond_count,
        budget.contact_constraint_count, budget.constraint_sweeps,
        budget.contact_passes, budget.time_advances,
        budget.spatial_updates_during_solve, budget.domain_dispatches,
        static_cast<unsigned long long>(budget.domain_workgroups),
        static_cast<unsigned long long>(budget.bond_visits),
        static_cast<unsigned long long>(budget.contact_visits),
        budget.solver_scratch_bytes);

    const SpatialEpochToken token{52U, 3U, 9U, 1U, 0U, 4U};
    SpatialEpochCounters counters{};
    counters.builds = 1U;
    counters.publications = 1U;
    counters.queries = 1'024U;
    SoftBodyRealtimeTickContract tick{};
    CHECK(tick.begin(token, counters, budget, 17U) == PM_SUCCESS);
    CHECK(tick.record_advance(soft_body_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(tick.record_advance(soft_body_realtime_timestep, 1U) ==
          PM_ERROR_INVALID_STATE);
    for (std::uint32_t sweep = 0U;
         sweep < soft_body_realtime_constraint_sweeps; ++sweep)
        CHECK(tick.record_sweep(token, counters, 17U,
                  (sweep & 1U) != 0U) == PM_SUCCESS);
    CHECK(tick.advances() == 1U);
    CHECK(tick.sweeps() == 8U);
    CHECK(tick.contact_passes() == 4U);
    CHECK(tick.finish() == PM_SUCCESS);

    SoftBodyRealtimeTickContract changed_topology{};
    CHECK(changed_topology.begin(token, counters, budget, 17U) == PM_SUCCESS);
    CHECK(changed_topology.record_advance(
              soft_body_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(changed_topology.record_sweep(
              token, counters, 18U, false) == PM_ERROR_INVALID_STATE);

    auto stale_counters = counters;
    ++stale_counters.refits;
    SoftBodyRealtimeTickContract stale{};
    CHECK(stale.begin(token, counters, budget, 17U) == PM_SUCCESS);
    CHECK(stale.record_advance(soft_body_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(stale.record_sweep(token, counters, 17U, false) == PM_SUCCESS);
    CHECK(stale.record_sweep(token, stale_counters, 17U, true) ==
          PM_ERROR_INVALID_STATE);
}

} // namespace

int main() {
    test_input_contract();
    test_budget_and_epoch();
    if (failures == 0)
        std::puts("F8-14 soft-body realtime contract tests passed");
    return failures == 0 ? 0 : 1;
}
