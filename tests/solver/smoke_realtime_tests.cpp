// SPDX-License-Identifier: MIT
#include "simulation/smoke_realtime.hpp"

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

SmokeRealtimeInput particle_input() {
    SmokeRealtimeInput result{};
    result.representation = SmokeRealtimeRepresentation::particles;
    result.world_coordinate = 10'000.0F;
    result.characteristic_length = 0.255F;
    result.maximum_speed = 4.0F;
    result.acceleration_displacement = 0.04F;
    result.solver_correction = 0.06F;
    result.response = 0.5F;
    result.pressure_stiffness = 2.0F;
    result.viscosity = 0.02F;
    return result;
}

SmokeRealtimeInput grid_input() {
    auto result = particle_input();
    result.representation = SmokeRealtimeRepresentation::grid;
    result.characteristic_length = 0.075F;
    result.maximum_speed = 4.0F;
    result.acceleration_displacement = 0.018F;
    result.solver_correction = 0.018F;
    result.grid_resolution = 64U;
    result.grid_vertical_resolution = 32U;
    return result;
}

void test_input_contract() {
    CHECK(smoke_realtime_step_request_valid(
        smoke_realtime_timestep, 1U));
    CHECK(smoke_realtime_step_request_valid(std::nextafter(
        smoke_realtime_timestep, 0.0F), 1U));
    CHECK(smoke_realtime_step_request_valid(std::nextafter(
        smoke_realtime_timestep,
        std::numeric_limits<float>::infinity()), 1U));
    CHECK(!smoke_realtime_step_request_valid(
        smoke_realtime_timestep, 2U));
    CHECK(!smoke_realtime_step_request_valid(std::nextafter(std::nextafter(
        smoke_realtime_timestep,
        std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity()), 1U));

    auto particle = particle_input();
    CHECK(validate_smoke_realtime_input(particle) == PM_SUCCESS);
    CHECK(std::abs(smoke_realtime_pressure_response(2.0F) -
                   0.86466472F) < 1.0e-6F);
    particle.maximum_speed = 7.66F;
    CHECK(validate_smoke_realtime_input(particle) ==
          PM_ERROR_INVALID_ARGUMENT);
    particle = particle_input();
    particle.solver_correction = 0.06376F;
    CHECK(validate_smoke_realtime_input(particle) ==
          PM_ERROR_INVALID_ARGUMENT);
    particle = particle_input();
    particle.solver_damping = 0.1201F;
    CHECK(validate_smoke_realtime_input(particle) ==
          PM_ERROR_INVALID_ARGUMENT);
    particle = particle_input();
    particle.pressure_stiffness = 0.01F;
    CHECK(validate_smoke_realtime_input(particle) ==
          PM_ERROR_INVALID_ARGUMENT);

    auto grid = grid_input();
    CHECK(validate_smoke_realtime_input(grid) == PM_SUCCESS);
    grid.maximum_speed = 4.51F;
    CHECK(validate_smoke_realtime_input(grid) ==
          PM_ERROR_INVALID_ARGUMENT);
    grid = grid_input();
    grid.grid_resolution = 0U;
    CHECK(validate_smoke_realtime_input(grid) ==
          PM_ERROR_INVALID_ARGUMENT);
}

void run_contract(const SmokeRealtimeWorkBudget &budget) {
    const SpatialEpochToken token{42U, 2U, 7U, 1U, 0U, 3U};
    const SpatialEpochCounters counters{1U, 0U, 1U, 4'500U, 0U, 288'000U};
    SmokeRealtimeTickContract tick{};
    CHECK(tick.begin(token, counters, budget) == PM_SUCCESS);
    CHECK(tick.record_iteration() == PM_ERROR_INVALID_STATE);
    CHECK(tick.record_advance(smoke_realtime_timestep, 1U) == PM_SUCCESS);
    CHECK(tick.record_advance(smoke_realtime_timestep, 1U) ==
          PM_ERROR_INVALID_STATE);
    for (std::uint32_t pass = 0U; pass < budget.spatial_passes; ++pass)
        CHECK(tick.record_spatial_pass(token, counters) == PM_SUCCESS);
    for (std::uint32_t iteration = 0U;
         iteration < budget.algebraic_iterations; ++iteration)
        CHECK(tick.record_iteration() == PM_SUCCESS);
    CHECK(tick.advances() == 1U);
    CHECK(tick.iterations() == budget.algebraic_iterations);
    CHECK(tick.record_iteration() == PM_ERROR_INVALID_STATE);
    CHECK(tick.finish() == PM_SUCCESS);

    SmokeRealtimeTickContract changed{};
    CHECK(changed.begin(token, counters, budget) == PM_SUCCESS);
    CHECK(changed.record_advance(smoke_realtime_timestep, 1U) == PM_SUCCESS);
    auto rebuilt = counters;
    ++rebuilt.refits;
    CHECK(changed.record_spatial_pass(token, rebuilt) ==
          PM_ERROR_INVALID_STATE);
}

void test_budget_and_epoch() {
    const auto particle =
        smoke_realtime_particle_work_budget(4'500U, 4'500U, 64U);
    CHECK(particle.algebraic_iterations == 6U);
    CHECK(particle.spatial_passes == 6U);
    CHECK(particle.time_advances == 1U);
    CHECK(particle.spatial_updates_during_solve == 0U);
    CHECK(particle.domain_dispatches == 21U);
    CHECK(particle.domain_workgroups == 756U);
    CHECK(particle.candidate_visits == 3'456'000U);
    CHECK(particle.solver_scratch_bytes == 198'008U);
    run_contract(particle);

    const auto grid = smoke_realtime_grid_work_budget(
        4'500U, 4'500U, 64U, 32U);
    CHECK(grid.algebraic_iterations == 12U);
    CHECK(grid.spatial_passes == 1U);
    CHECK(grid.time_advances == 1U);
    CHECK(grid.spatial_updates_during_solve == 0U);
    CHECK(grid.grid_cells == 131'072U);
    CHECK(grid.grid_faces == 401'408U);
    CHECK(grid.domain_dispatches == 30U);
    CHECK(smoke_realtime_grid_block_size == 256U);
    CHECK(grid.domain_workgroups == 15'012U);
    CHECK(grid.cell_visits == 3'838'356U);
    CHECK(grid.solver_scratch_bytes == 4'281'080U);
    run_contract(grid);

    std::printf(
        "F8_SMOKE_60HZ_BUDGET particles_dispatches=%u "
        "particle_candidate_visits=%llu grid_dispatches=%u "
        "grid_workgroups=%llu grid_cell_visits=%llu\n",
        particle.domain_dispatches,
        static_cast<unsigned long long>(particle.candidate_visits),
        grid.domain_dispatches,
        static_cast<unsigned long long>(grid.domain_workgroups),
        static_cast<unsigned long long>(grid.cell_visits));
}

} // namespace

int main() {
    test_input_contract();
    test_budget_and_epoch();
    if (failures == 0)
        std::puts("F8-16 smoke realtime contract tests passed");
    return failures == 0 ? 0 : 1;
}
