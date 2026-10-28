// SPDX-License-Identifier: MIT
#include "simulation/fluid_spatial.hpp"
#include "simulation/deformable_spatial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <set>
#include <span>
#include <vector>

namespace {
using namespace parallel_mater::simulation;

int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "%s:%d check failed: %s\n", __FILE__, __LINE__, \
                 #condition); \
    ++failures; \
} } while (false)

SpatialIdentity particle_identity(std::uint32_t population,
                                  std::uint32_t generation,
                                  std::uint32_t stable_id,
                                  std::uint32_t material_revision = 1U) {
    return {fluid_spatial_family, population, generation, material_revision,
            fluid_spatial_particle_primitive, stable_id, 1U};
}

SpatialAabb box(const std::array<float, 3> &point, float radius) {
    SpatialAabb result{};
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        result.minimum[axis] = point[axis] - radius;
        result.maximum[axis] = point[axis] + radius;
    }
    return result;
}

void test_plan_and_budget() {
    FluidSpatialPlan plan{};
    CHECK(make_fluid_spatial_plan(4096U, 128U, plan) == PM_SUCCESS);
    CHECK(plan.query_capacity == 4096U);
    CHECK(plan.candidate_slot_capacity == 4096U * 128U);
    CHECK(plan.query_leaf_sources.size == 4096U * sizeof(std::uint32_t));
    CHECK(plan.filtered_counts.size == 4096U * sizeof(std::uint32_t));
    CHECK(plan.filtered_sources.size ==
          4096ULL * 128ULL * sizeof(std::uint32_t));
    CHECK(plan.scratch_bytes >= plan.filtered_sources.size +
          2U * plan.query_leaf_sources.size + sizeof(std::uint32_t));
    CHECK(make_fluid_spatial_plan(0U, 0U, plan) == PM_SUCCESS);
    CHECK(make_fluid_spatial_plan(1U, 0U, plan) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(make_fluid_spatial_plan(
              std::numeric_limits<std::uint32_t>::max(), 2U, plan) ==
          PM_ERROR_CAPACITY_EXCEEDED);

    const auto sparse = fluid_spatial_work_budget(
        128U, 30000U, 64U, 2U, 2U, 128U);
    CHECK(sparse.boundary_dispatches == 9U);
    CHECK(sparse.neighbor_index_dispatches_in_solve == 0U);
    CHECK(sparse.neighbor_sort_dispatches == 0U);
    CHECK(sparse.live_query_workgroups == 3U);
    CHECK(sparse.capacity_query_workgroups == 470U);
    CHECK(sparse.filtered_candidate_bytes ==
          30064ULL * 128ULL * sizeof(std::uint32_t));
    CHECK(sparse.avoided_pair_key_bytes ==
          30000ULL * (2ULL * sizeof(std::uint64_t) +
                      2ULL * sizeof(std::uint32_t)));
    std::printf(
        "F8_FLUID_BUDGET sparse live=%u capacity=%u proposals=%u "
        "populations=%u directed_pairs=%u candidates=%u "
        "boundary_dispatches=%u solve_index_dispatches=%u "
        "sort_dispatches=%u live_query_workgroups=%llu "
        "capacity_query_workgroups=%llu filtered_candidate_bytes=%llu "
        "avoided_pair_key_bytes=%llu\n",
        sparse.live_particles, sparse.particle_capacity,
        sparse.source_proposals, sparse.populations,
        sparse.directed_pair_queries, sparse.candidates_per_query,
        sparse.boundary_dispatches,
        sparse.neighbor_index_dispatches_in_solve,
        sparse.neighbor_sort_dispatches,
        static_cast<unsigned long long>(sparse.live_query_workgroups),
        static_cast<unsigned long long>(sparse.capacity_query_workgroups),
        static_cast<unsigned long long>(sparse.filtered_candidate_bytes),
        static_cast<unsigned long long>(sparse.avoided_pair_key_bytes));

    const auto full = fluid_spatial_work_budget(
        30000U, 30000U, 0U, 1U, 0U, 128U);
    CHECK(full.boundary_dispatches == 3U);
    CHECK(full.live_query_workgroups == full.capacity_query_workgroups);
    std::printf(
        "F8_FLUID_BUDGET full live=%u capacity=%u proposals=%u "
        "populations=%u directed_pairs=%u candidates=%u "
        "boundary_dispatches=%u solve_index_dispatches=%u "
        "sort_dispatches=%u live_query_workgroups=%llu "
        "capacity_query_workgroups=%llu filtered_candidate_bytes=%llu\n",
        full.live_particles, full.particle_capacity, full.source_proposals,
        full.populations, full.directed_pair_queries,
        full.candidates_per_query, full.boundary_dispatches,
        full.neighbor_index_dispatches_in_solve,
        full.neighbor_sort_dispatches,
        static_cast<unsigned long long>(full.live_query_workgroups),
        static_cast<unsigned long long>(full.capacity_query_workgroups),
        static_cast<unsigned long long>(full.filtered_candidate_bytes));
}

struct MovingParticle {
    std::array<float, 3> start{};
    std::array<float, 3> predicted{};
    std::array<float, 3> deviation{};
    std::array<float, 3> correction{};
    float support{};
    std::uint32_t population{};
};

std::array<float, 3> sample_position(const MovingParticle &particle,
                                     float fraction,
                                     const std::array<float, 3> &offset) {
    std::array<float, 3> result{};
    for (std::uint32_t axis = 0U; axis < 3U; ++axis)
        result[axis] = particle.start[axis] + fraction *
            (particle.predicted[axis] - particle.start[axis]) + offset[axis];
    return result;
}

float squared_distance(const std::array<float, 3> &left,
                       const std::array<float, 3> &right) {
    float result{};
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        const float delta = left[axis] - right[axis];
        result += delta * delta;
    }
    return result;
}

void test_full_tick_heterogeneous_completeness() {
    std::mt19937 random(0xf809U);
    std::uniform_real_distribution<float> position(-2.0F, 2.0F);
    std::uniform_real_distribution<float> movement(-0.8F, 0.8F);
    std::uniform_real_distribution<float> budget(0.0F, 0.12F);
    std::uniform_real_distribution<float> support(0.08F, 0.65F);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    std::vector<MovingParticle> particles(48U);
    std::vector<SpatialAabb> leaf_bounds(particles.size());
    for (std::size_t index = 0U; index < particles.size(); ++index) {
        auto &particle = particles[index];
        particle.population = index % 3U;
        particle.support = support(random);
        for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
            particle.start[axis] = position(random);
            particle.predicted[axis] =
                particle.start[axis] + movement(random);
            particle.deviation[axis] = budget(random);
            particle.correction[axis] = budget(random);
        }
        CHECK(fluid_spatial_envelope(
                  particle.start.data(), particle.predicted.data(),
                  particle.deviation.data(), particle.correction.data(),
                  0.0F, 1.0e-5F, leaf_bounds[index]) == PM_SUCCESS);
    }

    std::uint64_t relevant{};
    for (std::uint32_t sample = 0U; sample < 64U; ++sample) {
        std::vector<std::array<float, 3>> current(particles.size());
        for (std::size_t index = 0U; index < particles.size(); ++index) {
            std::array<float, 3> offset{};
            for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
                const float reach = particles[index].deviation[axis] +
                    particles[index].correction[axis];
                offset[axis] = (2.0F * unit(random) - 1.0F) * reach;
            }
            current[index] = sample_position(
                particles[index], unit(random), offset);
            CHECK(spatial_contains(leaf_bounds[index],
                                   box(current[index], 0.0F)));
        }
        for (std::size_t query = 0U; query < particles.size(); ++query) {
            for (std::size_t target = 0U; target < particles.size(); ++target) {
                if (query == target) continue;
                const float radius = particles[query].population ==
                        particles[target].population
                    ? particles[query].support
                    : std::max(particles[query].support,
                               particles[target].support);
                if (squared_distance(current[query], current[target]) >=
                    radius * radius) continue;
                SpatialAabb query_bounds{};
                CHECK(fluid_spatial_envelope(
                          particles[query].start.data(),
                          particles[query].predicted.data(),
                          particles[query].deviation.data(),
                          particles[query].correction.data(), radius,
                          1.0e-5F, query_bounds) == PM_SUCCESS);
                CHECK(spatial_overlaps(query_bounds, leaf_bounds[target]));
                ++relevant;
            }
        }
    }
    CHECK(relevant > 100U);

    SpatialAabb invalid{};
    const float point[3]{};
    const float negative[3]{-1.0F, 0.0F, 0.0F};
    CHECK(fluid_spatial_envelope(point, point, negative, point, 1.0F,
                                 1.0e-5F, invalid) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(fluid_spatial_envelope(point, point, point, point, 1.0F, 0.0F,
                                 invalid) == PM_ERROR_INVALID_ARGUMENT);
}

void test_population_identity_and_boundary_admission() {
    std::vector<SpatialLeafMetadata> metadata;
    std::vector<SpatialAabb> bounds;
    auto add = [&](std::uint32_t population, std::uint32_t generation,
                   std::uint32_t stable_id, std::array<float, 3> point,
                   std::uint32_t flags, std::uint32_t sort_key) {
        SpatialLeafMetadata leaf{};
        leaf.identity = particle_identity(
            population, generation, stable_id, 11U + population);
        leaf.sort_key = sort_key;
        leaf.payload_ref = stable_id;
        leaf.query_mask = fluid_spatial_query_mask;
        leaf.admission_mask = fluid_spatial_query_mask;
        leaf.flags = flags;
        metadata.push_back(leaf);
        bounds.push_back(box(point, 0.1F));
    };

    // Outflow is compacted exactly once before staging. Dense slots change;
    // stable IDs and population generations do not.
    std::vector<std::uint32_t> stable_ids{7U, 8U, 9U, 10U};
    const std::array<std::uint8_t, 4> keep{1U, 0U, 1U, 0U};
    std::vector<std::uint32_t> compacted;
    for (std::size_t index = 0U; index < stable_ids.size(); ++index)
        if (keep[index] != 0U) compacted.push_back(stable_ids[index]);
    stable_ids = std::move(compacted);
    CHECK((stable_ids == std::vector<std::uint32_t>{7U, 9U}));
    add(3U, 4U, stable_ids[0], {0.0F, 0.0F, 0.0F},
        spatial_leaf_retained, 1U);
    add(3U, 4U, stable_ids[1], {2.0F, 0.0F, 0.0F},
        spatial_leaf_retained, 2U);
    // Equal local/stable IDs in another live population remain distinct.
    add(5U, 9U, stable_ids[0], {4.0F, 0.0F, 0.0F},
        spatial_leaf_retained, 3U);
    CHECK(metadata[0].identity != metadata[2].identity);

    // Lower-identity proposal conflicts with retained state. The next one
    // conflicts with that lower proposal even though it is itself rejected;
    // the final proposal is admitted. This is the documented deterministic
    // underfill rule and consumes no second source-specific index.
    add(3U, 4U, 100U, {0.05F, 0.0F, 0.0F},
        spatial_leaf_provisional, 4U);
    add(3U, 4U, 101U, {0.24F, 0.0F, 0.0F},
        spatial_leaf_provisional, 5U);
    add(3U, 4U, 102U, {3.0F, 0.0F, 0.0F},
        spatial_leaf_provisional, 6U);
    std::vector<std::uint8_t> admission(metadata.size());
    CHECK(spatial_resolve_admission(metadata, bounds, admission) ==
          PM_SUCCESS);
    CHECK(admission[0] == 1U && admission[1] == 1U && admission[2] == 1U);
    CHECK(admission[3] == 0U);
    CHECK(admission[4] == 0U);
    CHECK(admission[5] == 1U);

    std::set<SpatialIdentity> identities;
    for (std::size_t index = 0U; index < metadata.size(); ++index)
        if (admission[index] != 0U)
            CHECK(identities.insert(metadata[index].identity).second);
    CHECK(identities.contains(particle_identity(3U, 4U, 102U, 14U)));

    // Zero particles and inactive reserved capacity are valid: neither enters
    // the hierarchy nor needs a source/solver sort.
    std::vector<SpatialLeafMetadata> empty_metadata(64U);
    std::vector<SpatialAabb> empty_bounds(64U);
    std::vector<std::uint8_t> empty_admission(64U, 1U);
    for (auto &leaf : empty_metadata) leaf.flags = spatial_leaf_inactive;
    CHECK(spatial_resolve_admission(
              empty_metadata, empty_bounds, empty_admission) == PM_SUCCESS);
    CHECK(std::ranges::none_of(empty_admission,
        [](std::uint8_t value) { return value != 0U; }));
}

void test_coupling_visibility_does_not_reject_source() {
    std::array<SpatialLeafMetadata, 2> metadata{};
    metadata[0].identity = particle_identity(8U, 1U, 1U);
    metadata[0].sort_key = 1U;
    metadata[0].query_mask = coupling_spatial_query_mask;
    metadata[0].admission_mask = coupling_spatial_query_mask;
    metadata[0].flags = spatial_leaf_retained;

    metadata[1].identity = particle_identity(9U, 1U, 2U);
    metadata[1].sort_key = 2U;
    metadata[1].query_mask = fluid_spatial_query_mask |
        coupling_spatial_query_mask;
    metadata[1].admission_mask = fluid_source_admission_mask;
    metadata[1].flags = spatial_leaf_provisional;

    const std::array<SpatialAabb, 2> bounds{
        box({0.0F, 0.0F, 0.0F}, 10.0F),
        box({0.0F, 0.0F, 0.0F}, 0.1F)};
    std::array<std::uint8_t, 2> admission{};
    CHECK(spatial_resolve_admission(metadata, bounds, admission) ==
          PM_SUCCESS);
    CHECK(admission[0] == 1U);
    CHECK(admission[1] == 1U);
    CHECK((metadata[1].query_mask & coupling_spatial_query_mask) != 0U);
}

void test_one_update_reused_by_sweeps() {
    const SpatialEpochToken token{42U, 7U, 9U, 1U, 0U, 3U};
    SpatialEpochCounters counters{1U, 0U, 1U, 96U, 0U, 3840U};
    FluidSpatialSolveEpoch epoch;
    CHECK(epoch.begin(token, counters) == PM_SUCCESS);
    for (std::uint32_t sweep = 0U;
         sweep < fluid_spatial_density_sweeps; ++sweep)
        CHECK(epoch.record_sweep(token, counters) == PM_SUCCESS);
    CHECK(epoch.completed_sweeps() == fluid_spatial_density_sweeps);
    CHECK(epoch.record_sweep(token, counters) == PM_ERROR_INVALID_STATE);
    CHECK(epoch.finish() == PM_SUCCESS);

    FluidSpatialSolveEpoch rebuilt;
    CHECK(rebuilt.begin(token, counters) == PM_SUCCESS);
    auto changed = counters;
    ++changed.builds;
    CHECK(rebuilt.record_sweep(token, changed) == PM_ERROR_INVALID_STATE);

    FluidSpatialSolveEpoch requeried;
    CHECK(requeried.begin(token, counters) == PM_SUCCESS);
    changed = counters;
    ++changed.queries;
    CHECK(requeried.record_sweep(token, changed) == PM_ERROR_INVALID_STATE);

    FluidSpatialSolveEpoch empty;
    counters = {0U, 0U, 1U, 0U, 0U, 0U};
    CHECK(empty.begin(token, counters) == PM_ERROR_INVALID_STATE);
}

} // namespace

int main() {
    test_plan_and_budget();
    test_full_tick_heterogeneous_completeness();
    test_population_identity_and_boundary_admission();
    test_coupling_visibility_does_not_reject_source();
    test_one_update_reused_by_sweeps();
    if (failures == 0)
        std::printf("F8-09 host fluid spatial contract passed\n");
    return failures == 0 ? 0 : 1;
}
