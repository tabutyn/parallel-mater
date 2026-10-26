// SPDX-License-Identifier: MIT
#include "simulation/deformable_smoke_coupling_realtime.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

namespace {
using namespace parallel_mater::simulation;

int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "%s:%d check failed: %s\n", __FILE__, __LINE__, \
        #condition); ++failures; } } while (false)

constexpr std::array<DeformableSmokeCouplingRegistration, 16> registrations{{
    {1U, DeformableSmokeCouplingPair::cloth_soft_body,
        DeformableSmokeCouplingMode::contact, 1U, 11U, 1U},
    {2U, DeformableSmokeCouplingPair::cloth_rope,
        DeformableSmokeCouplingMode::attachment, 2U, 12U, 1U},
    {3U, DeformableSmokeCouplingPair::cloth_smoke_particle,
        DeformableSmokeCouplingMode::contact, 3U, 13U, 1U},
    {4U, DeformableSmokeCouplingPair::cloth_smoke_particle,
        DeformableSmokeCouplingMode::drag, 3U, 13U, 1U},
    {5U, DeformableSmokeCouplingPair::cloth_smoke_grid,
        DeformableSmokeCouplingMode::contact, 4U, 14U, 1U},
    {6U, DeformableSmokeCouplingPair::cloth_smoke_grid,
        DeformableSmokeCouplingMode::drag, 4U, 14U, 1U},
    {7U, DeformableSmokeCouplingPair::soft_body_rope,
        DeformableSmokeCouplingMode::contact, 5U, 15U, 1U},
    {8U, DeformableSmokeCouplingPair::soft_body_rope,
        DeformableSmokeCouplingMode::attachment, 5U, 15U, 1U},
    {9U, DeformableSmokeCouplingPair::soft_body_smoke_particle,
        DeformableSmokeCouplingMode::contact, 6U, 16U, 1U},
    {10U, DeformableSmokeCouplingPair::soft_body_smoke_particle,
        DeformableSmokeCouplingMode::drag, 6U, 16U, 1U},
    {11U, DeformableSmokeCouplingPair::soft_body_smoke_grid,
        DeformableSmokeCouplingMode::contact, 7U, 17U, 1U},
    {12U, DeformableSmokeCouplingPair::soft_body_smoke_grid,
        DeformableSmokeCouplingMode::drag, 7U, 17U, 1U},
    {13U, DeformableSmokeCouplingPair::rope_smoke_particle,
        DeformableSmokeCouplingMode::contact, 8U, 18U, 1U},
    {14U, DeformableSmokeCouplingPair::rope_smoke_particle,
        DeformableSmokeCouplingMode::drag, 8U, 18U, 1U},
    {15U, DeformableSmokeCouplingPair::rope_smoke_grid,
        DeformableSmokeCouplingMode::contact, 9U, 19U, 1U},
    {16U, DeformableSmokeCouplingPair::rope_smoke_grid,
        DeformableSmokeCouplingMode::drag, 9U, 19U, 1U},
}};

DeformableSmokeCouplingRealtimeInput valid_input(
    DeformableSmokeCouplingPair pair,
    DeformableSmokeCouplingMode mode) {
    DeformableSmokeCouplingRealtimeInput input{};
    input.pair = pair;
    input.mode = mode;
    input.characteristic_length = 1.0F;
    input.first_mass = 1.0F;
    input.second_mass = 100.0F;
    input.linear_speed = 1.0F;
    if (mode == DeformableSmokeCouplingMode::contact) {
        input.contact_distance = 0.02F;
        input.friction = 0.5F;
    } else {
        input.response = 0.5F;
    }
    return input;
}

void test_registered_matrix_and_validation() {
    for (const auto &registration : registrations) {
        CHECK(deformable_smoke_coupling_mode_supported(
            registration.pair, registration.mode));
        CHECK(validate_deformable_smoke_coupling_input(valid_input(
            registration.pair, registration.mode)) == PM_SUCCESS);
    }
    CHECK(!deformable_smoke_coupling_mode_supported(
        DeformableSmokeCouplingPair::cloth_soft_body,
        DeformableSmokeCouplingMode::drag));
    CHECK(!deformable_smoke_coupling_mode_supported(
        DeformableSmokeCouplingPair::cloth_rope,
        DeformableSmokeCouplingMode::contact));
    CHECK(deformable_smoke_coupling_is_two_way(
        DeformableSmokeCouplingPair::soft_body_rope));
    CHECK(!deformable_smoke_coupling_is_two_way(
        DeformableSmokeCouplingPair::rope_smoke_grid));

    auto invalid = valid_input(
        DeformableSmokeCouplingPair::cloth_smoke_particle,
        DeformableSmokeCouplingMode::contact);
    invalid.second_dynamic = false;
    invalid.second_mass = 0.0F;
    CHECK(validate_deformable_smoke_coupling_input(invalid) ==
        PM_ERROR_INVALID_ARGUMENT);
    invalid = valid_input(DeformableSmokeCouplingPair::cloth_smoke_grid,
        DeformableSmokeCouplingMode::drag);
    invalid.first_dynamic = false;
    invalid.first_mass = 0.0F;
    CHECK(validate_deformable_smoke_coupling_input(invalid) ==
        PM_ERROR_INVALID_ARGUMENT);
    invalid = valid_input(DeformableSmokeCouplingPair::soft_body_rope,
        DeformableSmokeCouplingMode::contact);
    invalid.second_mass = 101.0F;
    CHECK(validate_deformable_smoke_coupling_input(invalid) ==
        PM_ERROR_INVALID_ARGUMENT);
    invalid = valid_input(DeformableSmokeCouplingPair::cloth_soft_body,
        DeformableSmokeCouplingMode::contact);
    invalid.world_coordinate = std::numeric_limits<float>::infinity();
    CHECK(validate_deformable_smoke_coupling_input(invalid) ==
        PM_ERROR_INVALID_ARGUMENT);

    CHECK(deformable_smoke_coupling_step_request_valid(1.0F / 60.0F, 1U));
    CHECK(!deformable_smoke_coupling_step_request_valid(1.0F / 120.0F, 2U));
}

void test_bounded_work() {
    DeformableSmokeCouplingWorkBudget budget{};
    CHECK(make_deformable_smoke_coupling_work_budget(
        registrations, 6144U, 6144U, budget) == PM_SUCCESS);
    CHECK(budget.edge_count == 16U);
    CHECK(budget.constraint_count == 16U);
    CHECK(budget.coupling_sweeps_per_edge == 4U);
    CHECK(budget.edge_sweeps == 64U);
    CHECK(budget.constraint_visits == 64U);
    CHECK(budget.time_advances == 0U);
    CHECK(budget.spatial_updates_during_solve == 0U);
    CHECK(budget.batched_dispatches == 12U);
    CHECK(budget.batched_workgroups == 388U);
    CHECK(budget.intra_tick_host_readbacks == 0U);
    CHECK(budget.solver_scratch_bytes == 49'920U);
    for (std::size_t index = 0U; index < registrations.size(); ++index) {
        CHECK(budget.operators[index].registration == registrations[index]);
        CHECK(budget.operators[index].algebraic_sweeps == 4U);
        CHECK(budget.operators[index].constraint_visits == 4U);
    }

    std::array<DeformableSmokeCouplingRegistration, 65> too_many{};
    too_many.fill(registrations.front());
    CHECK(make_deformable_smoke_coupling_work_budget(
        too_many, 1U, 1U, budget) == PM_ERROR_CAPACITY_EXCEEDED);
}

void test_shared_epoch_and_lifecycle() {
    DeformableSmokeCouplingWorkBudget budget{};
    CHECK(make_deformable_smoke_coupling_work_budget(
        registrations, 64U, 64U, budget) == PM_SUCCESS);
    const SpatialEpochToken token{42U, 7U, 9U, 1U, 0U, 3U};
    const SpatialEpochCounters counters{1U, 0U, 1U, 16U, 0U, 16U};

    DeformableSmokeCouplingTickContract tick{};
    CHECK(tick.begin(token, counters, budget, registrations, 5U) == PM_SUCCESS);
    CHECK(tick.record_edge(token, counters, 5U, registrations[1]) ==
        PM_ERROR_INVALID_STATE);
    for (std::uint32_t sweep = 0U;
         sweep < deformable_smoke_coupling_sweeps; ++sweep) {
        for (const auto &registration : registrations)
            CHECK(tick.record_edge(token, counters, 5U, registration) ==
                PM_SUCCESS);
    }
    CHECK(tick.completed_sweeps() == 4U);
    CHECK(tick.finish() == PM_SUCCESS);

    DeformableSmokeCouplingTickContract changed_topology{};
    CHECK(changed_topology.begin(token, counters, budget,
        registrations, 5U) == PM_SUCCESS);
    CHECK(changed_topology.record_edge(token, counters, 6U,
        registrations[0]) == PM_ERROR_INVALID_STATE);

    auto stale_token = token;
    ++stale_token.generation;
    DeformableSmokeCouplingTickContract stale{};
    CHECK(stale.begin(token, counters, budget, registrations, 5U) == PM_SUCCESS);
    CHECK(stale.record_edge(stale_token, counters, 5U, registrations[0]) ==
        PM_ERROR_INVALID_STATE);

    DeformableSmokeCouplingWorkBudget empty_budget{};
    const std::span<const DeformableSmokeCouplingRegistration> empty{};
    CHECK(make_deformable_smoke_coupling_work_budget(
        empty, 0U, 0U, empty_budget) == PM_SUCCESS);
    DeformableSmokeCouplingTickContract empty_tick{};
    CHECK(empty_tick.begin(token, counters, empty_budget, empty, 5U) ==
        PM_SUCCESS);
    CHECK(empty_tick.finish() == PM_SUCCESS);
}

} // namespace

int main() {
    test_registered_matrix_and_validation();
    test_bounded_work();
    test_shared_epoch_and_lifecycle();
    if (failures != 0)
        std::fprintf(stderr, "%d deformable/smoke coupling checks failed\n",
            failures);
    return failures == 0 ? 0 : 1;
}
