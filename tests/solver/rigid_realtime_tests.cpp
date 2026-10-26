// SPDX-License-Identifier: MIT
#include "simulation/rigid_realtime.hpp"

#include <cmath>
#include <cstdint>
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

RigidRealtimeInput valid_input() {
    RigidRealtimeInput result{};
    result.world_coordinate = 10'000.0F;
    result.characteristic_length = 0.1F;
    result.mass = 1.0F;
    result.interacting_mass_ratio = 99.0F;
    result.linear_speed = 2.9F;
    result.angular_speed = 10.0F;
    result.acceleration_displacement = 0.024F;
    result.solver_correction = 0.024F;
    result.friction = 2.0F;
    result.restitution = 0.8F;
    result.effective_inverse_mass = 1.0F;
    const float ratio = 1.0F;
    result.compliance = ratio * rigid_realtime_timestep *
        rigid_realtime_timestep * result.effective_inverse_mass;
    return result;
}

void test_input_contract() {
    CHECK(rigid_realtime_step_request_valid(rigid_realtime_timestep, 1U));
    CHECK(rigid_realtime_step_request_valid(
        std::nextafter(rigid_realtime_timestep, 0.0F), 1U));
    CHECK(rigid_realtime_step_request_valid(std::nextafter(
        rigid_realtime_timestep, std::numeric_limits<float>::infinity()), 1U));
    CHECK(!rigid_realtime_step_request_valid(rigid_realtime_timestep, 2U));
    CHECK(!rigid_realtime_step_request_valid(std::nextafter(std::nextafter(
        rigid_realtime_timestep, std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity()), 1U));
    CHECK(!rigid_realtime_step_request_valid(
        std::numeric_limits<float>::quiet_NaN(), 1U));

    auto input = valid_input();
    CHECK(validate_rigid_realtime_input(input) == PM_SUCCESS);
    CHECK(std::abs(rigid_realtime_compliance_response(
                       input.compliance, input.effective_inverse_mass) - 0.5F) <
          1.0e-6F);

    const float denominator = rigid_realtime_timestep *
        rigid_realtime_timestep * input.effective_inverse_mass;
    input.compliance = 19.0F * denominator;
    CHECK(std::abs(rigid_realtime_compliance_response(
                       input.compliance, input.effective_inverse_mass) - 0.05F) <
          1.0e-6F);
    CHECK(validate_rigid_realtime_input(input) == PM_SUCCESS);
    input.compliance = (19.0F + 1.0e-3F) * denominator;
    CHECK(validate_rigid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);

    input = valid_input();
    input.linear_speed = 3.0001F;
    CHECK(validate_rigid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.solver_correction = 0.02501F;
    CHECK(validate_rigid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.interacting_mass_ratio = 100.01F;
    CHECK(validate_rigid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
    input = valid_input();
    input.solver_damping = 0.1201F;
    CHECK(validate_rigid_realtime_input(input) == PM_ERROR_INVALID_ARGUMENT);
}

void test_budget_and_epoch() {
    const auto budget = rigid_realtime_work_budget(
        386U, 65'536U, 32U, 8U, 13U);
    CHECK(budget.algebraic_sweeps == 8U);
    CHECK(budget.time_advances == 1U);
    CHECK(budget.spatial_updates_during_solve == 0U);
    CHECK(budget.cooperative_dispatches == 2U);
    CHECK(budget.staged_dispatches == 79U);
    CHECK(budget.staged_workgroups == 79ULL * 13ULL);
    CHECK(budget.persistent_bytes >
          65'536ULL * sizeof(ContactManifold));
    std::printf(
        "F8_RIGID_60HZ_BUDGET bodies=%u contacts=%u joints=%u colors=%u "
        "sweeps=%u time_advances=%u solve_index_updates=%u "
        "cooperative_dispatches=%u staged_dispatches=%u "
        "staged_workgroups=%llu persistent_bytes=%zu\n",
        budget.body_count, budget.contact_count, budget.joint_count,
        budget.color_count, budget.algebraic_sweeps, budget.time_advances,
        budget.spatial_updates_during_solve, budget.cooperative_dispatches,
        budget.staged_dispatches,
        static_cast<unsigned long long>(budget.staged_workgroups),
        budget.persistent_bytes);

    SpatialEpochToken token{22U, 1U, 1U, 1U, 0U, 9U};
    SpatialEpochCounters counters{};
    counters.builds = 1U;
    counters.publications = 1U;
    counters.queries = 386U;
    RigidRealtimeTickContract tick{};
    CHECK(tick.begin(token, counters, budget) == PM_SUCCESS);
    for (std::uint32_t sweep = 0U; sweep < rigid_realtime_sweeps; ++sweep)
        CHECK(tick.record_sweep(token, counters) == PM_SUCCESS);
    CHECK(tick.finish() == PM_SUCCESS);

    auto changed = counters;
    RigidRealtimeTickContract stale{};
    CHECK(stale.begin(token, counters, budget) == PM_SUCCESS);
    ++changed.refits;
    CHECK(stale.record_sweep(token, changed) == PM_ERROR_INVALID_STATE);
    auto bad_budget = budget;
    bad_budget.time_advances = 2U;
    RigidRealtimeTickContract bad{};
    CHECK(bad.begin(token, counters, bad_budget) == PM_ERROR_INVALID_STATE);
}

void test_metal_dispatch_ceilings() {
    struct Fixture {
        const char *name;
        std::uint32_t bodies;
        std::uint32_t constraints;
        std::uint32_t grid_blocks;
        std::uint32_t staged_solve_baseline;
    };
    // Independent F8-54 acceptance values. These are deliberately explicit,
    // not generated from the production Metal adapter or telemetry limits.
    constexpr Fixture fixtures[]{
        {"small-active", 8U, 7U, 1U, 78U},
        {"sparse-capacity", 64U, 1U, 2U, 526U},
        {"medium-active", 128U, 32U, 4U, 1'038U},
        {"desktop-386", 386U, 32U, 13U, 3'102U},
    };
    constexpr std::uint32_t solve_dispatch_ceiling = 1U;
    constexpr std::uint32_t total_avbd_dispatch_ceiling = 2U;
    constexpr std::uint32_t solve_workgroup_ceiling = 1U;
    for (const auto &fixture : fixtures) {
        const auto budget = rigid_realtime_work_budget(
            fixture.bodies, 65'536U, fixture.constraints, fixture.bodies,
            fixture.grid_blocks);
        CHECK(budget.cooperative_dispatches <= total_avbd_dispatch_ceiling);
        CHECK(budget.cooperative_dispatches - 1U <= solve_dispatch_ceiling);
        CHECK(budget.cooperative_dispatches - 1U <=
              solve_workgroup_ceiling);
        CHECK(budget.staged_dispatches ==
              fixture.staged_solve_baseline + 1U);
        CHECK(fixture.staged_solve_baseline >=
              4U * solve_dispatch_ceiling);
        std::printf(
            "F8_METAL_AVBD_CEILING fixture=%s bodies=%u constraints=%u "
            "old_solve_dispatches=%u solve_dispatch_ceiling=%u "
            "avbd_dispatch_ceiling=%u solve_workgroup_ceiling=%u\n",
            fixture.name, fixture.bodies, fixture.constraints,
            fixture.staged_solve_baseline, solve_dispatch_ceiling,
            total_avbd_dispatch_ceiling, solve_workgroup_ceiling);
    }
}

} // namespace

int main() {
    test_input_contract();
    test_budget_and_epoch();
    test_metal_dispatch_ceilings();
    if (failures == 0)
        std::puts("F8-11 rigid realtime contract tests passed");
    return failures == 0 ? 0 : 1;
}
