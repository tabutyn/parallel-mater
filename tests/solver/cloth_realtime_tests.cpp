// SPDX-License-Identifier: MIT
#include "simulation/cloth_realtime.hpp"

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

ClothRealtimeInput valid_input() {
    ClothRealtimeInput result{};
    result.world_coordinate = 10'000.0F;
    result.characteristic_length = 0.1F;
    result.mass = 0.02F;
    result.interacting_mass_ratio = 100.0F;
    result.linear_speed = 2.9F;
    result.acceleration_displacement = 0.024F;
    result.solver_correction = 0.025F;
    result.authored_strain = 0.25F;
    result.effective_inverse_mass = 100.0F;
    result.stretch_compliance = cloth_realtime_timestep *
        cloth_realtime_timestep * result.effective_inverse_mass;
    result.bending_compliance = result.stretch_compliance;
    result.volume_compliance = 1.0e-7F;
    result.thickness = 0.025F;
    result.velocity_damping = 5.0F;
    result.contact_friction = 2.0F;
    return result;
}

void test_input_contract() {
    CHECK(cloth_realtime_constraint_sweeps == 8U);
    CHECK(cloth_realtime_contact_passes == 4U);
    CHECK(cloth_realtime_step_request_valid(
        cloth_realtime_timestep, 1U));
    CHECK(cloth_realtime_step_request_valid(std::nextafter(
        cloth_realtime_timestep, 0.0F), 1U));
    CHECK(cloth_realtime_step_request_valid(std::nextafter(
        cloth_realtime_timestep,
        std::numeric_limits<float>::infinity()), 1U));
    CHECK(!cloth_realtime_step_request_valid(
        cloth_realtime_timestep, 2U));
    CHECK(!cloth_realtime_step_request_valid(std::nextafter(std::nextafter(
        cloth_realtime_timestep,
        std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity()), 1U));

    auto input = valid_input();
    CHECK(validate_cloth_realtime_input(input) == PM_SUCCESS);
    CHECK(std::abs(cloth_realtime_compliance_response(
                       input.stretch_compliance,
                       input.effective_inverse_mass) - 0.5F) < 1.0e-6F);

    const float denominator = cloth_realtime_timestep *
        cloth_realtime_timestep * input.effective_inverse_mass;
    input.stretch_compliance = 19.0F * denominator;
    CHECK(std::abs(cloth_realtime_compliance_response(
                       input.stretch_compliance,
                       input.effective_inverse_mass) - 0.05F) < 1.0e-6F);
    CHECK(validate_cloth_realtime_input(input) == PM_SUCCESS);
    input.stretch_compliance = (19.0F + 1.0e-3F) * denominator;
    CHECK(validate_cloth_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);

    input = valid_input();
    input.linear_speed = 3.0001F;
    CHECK(validate_cloth_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.solver_correction = 0.02501F;
    CHECK(validate_cloth_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.authored_strain = 0.2501F;
    CHECK(validate_cloth_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.solver_damping = 0.1201F;
    CHECK(validate_cloth_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.stretch_compliance = 0.0F;
    input.bending_compliance = 0.0F;
    CHECK(validate_cloth_realtime_input(input) == PM_SUCCESS);
}

void test_budget_and_epoch() {
    const auto budget = cloth_realtime_work_budget(
        578U, 578U, 1'024U, 6'144U, 578U);
    CHECK(budget.constraint_sweeps == 8U);
    CHECK(budget.contact_passes == 4U);
    CHECK(budget.time_advances == 1U);
    CHECK(budget.spatial_updates_during_solve == 0U);
    CHECK(budget.domain_dispatches == 31U);
    CHECK(budget.domain_workgroups == 126U);
    CHECK(budget.link_visits == 49'152U);
    CHECK(budget.contact_visits == 2'312U);
    CHECK(budget.solver_scratch_bytes == 23'132U);
    std::printf(
        "F8_CLOTH_60HZ_BUDGET vertices=%u capacity=%u triangles=%u "
        "directed_links=%u contacts=%u sweeps=%u contact_passes=%u "
        "time_advances=%u solve_index_updates=%u dispatches=%u "
        "workgroups=%llu link_visits=%llu contact_visits=%llu "
        "scratch_bytes=%zu\n",
        budget.live_vertices, budget.vertex_capacity, budget.triangle_count,
        budget.directed_link_count, budget.contact_constraint_count,
        budget.constraint_sweeps, budget.contact_passes,
        budget.time_advances, budget.spatial_updates_during_solve,
        budget.domain_dispatches,
        static_cast<unsigned long long>(budget.domain_workgroups),
        static_cast<unsigned long long>(budget.link_visits),
        static_cast<unsigned long long>(budget.contact_visits),
        budget.solver_scratch_bytes);

    const SpatialEpochToken token{52U, 3U, 9U, 1U, 0U, 4U};
    SpatialEpochCounters counters{};
    counters.builds = 1U;
    counters.publications = 1U;
    counters.queries = 1'024U;
    ClothRealtimeTickContract tick{};
    CHECK(tick.begin(token, counters, budget, 17U) == PM_SUCCESS);
    CHECK(tick.record_advance(cloth_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(tick.record_advance(cloth_realtime_timestep, 1U) ==
          PM_ERROR_INVALID_STATE);
    for (std::uint32_t sweep = 0U;
         sweep < cloth_realtime_constraint_sweeps; ++sweep)
        CHECK(tick.record_sweep(token, counters, 17U,
                  (sweep & 1U) != 0U) == PM_SUCCESS);
    CHECK(tick.advances() == 1U);
    CHECK(tick.sweeps() == 8U);
    CHECK(tick.contact_passes() == 4U);
    CHECK(tick.finish() == PM_SUCCESS);

    ClothRealtimeTickContract changed_topology{};
    CHECK(changed_topology.begin(token, counters, budget, 17U) == PM_SUCCESS);
    CHECK(changed_topology.record_advance(
              cloth_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(changed_topology.record_sweep(
              token, counters, 18U, false) == PM_ERROR_INVALID_STATE);

    ClothRealtimeTickContract wrong_cadence{};
    CHECK(wrong_cadence.begin(token, counters, budget, 17U) == PM_SUCCESS);
    CHECK(wrong_cadence.record_advance(
              cloth_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(wrong_cadence.record_sweep(
              token, counters, 17U, true) == PM_ERROR_INVALID_STATE);

    auto stale_counters = counters;
    ++stale_counters.refits;
    ClothRealtimeTickContract stale{};
    CHECK(stale.begin(token, counters, budget, 17U) == PM_SUCCESS);
    CHECK(stale.record_advance(cloth_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(stale.record_sweep(token, counters, 17U, false) == PM_SUCCESS);
    CHECK(stale.record_sweep(token, stale_counters, 17U, true) ==
          PM_ERROR_INVALID_STATE);
}

} // namespace

int main() {
    test_input_contract();
    test_budget_and_epoch();
    if (failures == 0)
        std::puts("F8-13 cloth realtime contract tests passed");
    return failures == 0 ? 0 : 1;
}
