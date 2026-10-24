// SPDX-License-Identifier: MIT
#include "simulation/fluid_realtime.hpp"

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

FluidRealtimeInput valid_input() {
    FluidRealtimeInput result{};
    result.world_coordinate = 10'000.0F;
    result.support_radius = 0.09F;
    result.particle_radius = 0.0225F;
    result.rest_density = 1'000.0F;
    result.rest_particle_volume = 0.045F * 0.045F * 0.045F;
    result.maximum_speed = 2.7F;
    result.acceleration_displacement = 0.022F;
    result.solver_correction = 0.0225F;
    result.repulsion = 50.0F;
    result.viscosity = 0.02F;
    result.normal_damping = 0.0F;
    result.velocity_damping = 0.2F;
    return result;
}

void test_input_contract() {
    CHECK(fluid_realtime_density_sweeps == fluid_spatial_density_sweeps);
    CHECK(fluid_realtime_step_request_valid(
        fluid_realtime_timestep, 1U));
    CHECK(fluid_realtime_step_request_valid(std::nextafter(
        fluid_realtime_timestep, 0.0F), 1U));
    CHECK(fluid_realtime_step_request_valid(std::nextafter(
        fluid_realtime_timestep,
        std::numeric_limits<float>::infinity()), 1U));
    CHECK(!fluid_realtime_step_request_valid(
        fluid_realtime_timestep, 2U));
    CHECK(!fluid_realtime_step_request_valid(std::nextafter(std::nextafter(
        fluid_realtime_timestep,
        std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity()), 1U));

    auto input = valid_input();
    CHECK(validate_fluid_realtime_input(input) == PM_SUCCESS);
    CHECK(std::abs(fluid_realtime_density_response(input.repulsion) -
                   0.56540179F) < 1.0e-6F);
    input.maximum_speed = 2.7001F;
    CHECK(validate_fluid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.solver_correction = 0.02251F;
    CHECK(validate_fluid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.solver_damping = 0.1201F;
    CHECK(validate_fluid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.repulsion = 1.0F;
    CHECK(validate_fluid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input.repulsion = 0.0F;
    CHECK(validate_fluid_realtime_input(input) == PM_SUCCESS);
    input = valid_input();
    input.rest_particle_volume = 0.0F;
    CHECK(validate_fluid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
}

void test_budget_and_epoch() {
    const auto budget = fluid_realtime_work_budget(30'000U, 30'000U, 128U);
    CHECK(budget.density_sweeps == 6U);
    CHECK(budget.time_advances == 1U);
    CHECK(budget.spatial_updates_during_solve == 0U);
    CHECK(budget.domain_dispatches == 21U);
    CHECK(budget.domain_workgroups == 4'935U);
    CHECK(budget.candidate_visits == 49'920'000U);
    CHECK(budget.solver_scratch_bytes == 1'320'008U);
    std::printf(
        "F8_FLUID_60HZ_BUDGET particles=%u capacity=%u candidates=%u "
        "sweeps=%u time_advances=%u solve_index_updates=%u "
        "domain_dispatches=%u domain_workgroups=%llu "
        "candidate_visits=%llu solver_scratch_bytes=%zu\n",
        budget.live_particles, budget.particle_capacity,
        budget.candidates_per_query, budget.density_sweeps,
        budget.time_advances, budget.spatial_updates_during_solve,
        budget.domain_dispatches,
        static_cast<unsigned long long>(budget.domain_workgroups),
        static_cast<unsigned long long>(budget.candidate_visits),
        budget.solver_scratch_bytes);

    const SpatialEpochToken token{42U, 2U, 7U, 1U, 0U, 3U};
    const SpatialEpochCounters counters{1U, 0U, 1U, 30'000U, 0U,
                                        3'840'000U};
    FluidRealtimeTickContract tick{};
    CHECK(tick.begin(token, counters, budget) == PM_SUCCESS);
    CHECK(tick.record_sweep(token, counters) == PM_ERROR_INVALID_STATE);
    CHECK(tick.record_advance(fluid_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(tick.record_advance(fluid_realtime_timestep, 1U) ==
          PM_ERROR_INVALID_STATE);
    for (std::uint32_t sweep = 0U;
         sweep < fluid_realtime_density_sweeps; ++sweep)
        CHECK(tick.record_sweep(token, counters) == PM_SUCCESS);
    CHECK(tick.sweeps() == fluid_realtime_density_sweeps);
    CHECK(tick.advances() == 1U);
    CHECK(tick.record_sweep(token, counters) == PM_ERROR_INVALID_STATE);
    CHECK(tick.finish() == PM_SUCCESS);

    FluidRealtimeTickContract changed{};
    CHECK(changed.begin(token, counters, budget) == PM_SUCCESS);
    CHECK(changed.record_advance(fluid_realtime_timestep, 1U) == PM_SUCCESS);
    auto rebuilt = counters;
    ++rebuilt.builds;
    CHECK(changed.record_sweep(token, rebuilt) == PM_ERROR_INVALID_STATE);

    auto bad_budget = budget;
    bad_budget.domain_dispatches = 22U;
    FluidRealtimeTickContract bad{};
    CHECK(bad.begin(token, counters, bad_budget) == PM_ERROR_INVALID_STATE);
}

} // namespace

int main() {
    test_input_contract();
    test_budget_and_epoch();
    if (failures == 0)
        std::puts("F8-12 fluid realtime contract tests passed");
    return failures == 0 ? 0 : 1;
}
