// SPDX-License-Identifier: MIT
#include "simulation/rigid_spatial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <span>
#include <utility>
#include <vector>

namespace {
using namespace parallel_mater::simulation;

int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "%s:%d check failed: %s\n", __FILE__, __LINE__, \
                 #condition); \
    ++failures; \
} } while (false)

SpatialAabb box(float x0, float y0, float z0,
                float x1, float y1, float z1) {
    SpatialAabb result{};
    result.minimum[0] = x0;
    result.minimum[1] = y0;
    result.minimum[2] = z0;
    result.maximum[0] = x1;
    result.maximum[1] = y1;
    result.maximum[2] = z1;
    return result;
}

SpatialIdentity identity(std::uint32_t slot, std::uint32_t generation = 1U,
                         std::uint32_t mesh = 1U,
                         std::uint32_t mesh_generation = 1U) {
    return {rigid_spatial_family, slot, generation, 1U,
            rigid_spatial_body_primitive, mesh, mesh_generation};
}

void test_plan_and_budgets() {
    RigidSpatialPlan plan{};
    CHECK(make_rigid_spatial_plan(386U, 64U, plan) == PM_SUCCESS);
    CHECK(plan.candidate_slot_capacity == 24704U);
    CHECK(plan.compaction_blocks == 97U);
    CHECK(plan.candidate_pair_flags.size == 24704U);
    CHECK(plan.candidate_pair_codes.size == 24704U * sizeof(std::uint32_t));
    CHECK(plan.active_pairs.size == 24704U * sizeof(std::uint32_t));
    CHECK(plan.scratch_bytes > plan.candidate_pair_codes.size);
    CHECK(plan.output_bytes >= plan.active_pairs.size + sizeof(std::uint32_t));

    const auto sparse = rigid_spatial_work_budget(386U, 64U);
    CHECK(sparse.consumer_dispatches == 7U);
    CHECK(sparse.legacy_candidate_flag_bytes == 386ULL * 386ULL);
    CHECK(sparse.candidate_slot_capacity == 386U * 64U);
    CHECK(sparse.candidate_slot_capacity <
          sparse.legacy_candidate_flag_bytes);
    CHECK(sparse.bounded_candidate_bytes ==
          386ULL * 64ULL * sizeof(std::uint32_t));
    CHECK(sparse.bounded_pair_scratch_bytes == plan.scratch_bytes);
    CHECK(sparse.bounded_pair_output_bytes == plan.output_bytes);
    std::printf(
        "F8_RIGID_BUDGET sparse bodies=%u candidates_per_body=%u "
        "candidate_slots=%u consumer_dispatches=%u consumer_workgroups=%llu "
        "legacy_flag_bytes=%llu candidate_bytes=%llu pair_scratch_bytes=%zu "
        "pair_output_bytes=%zu\n",
        sparse.body_count, sparse.candidates_per_body,
        sparse.candidate_slot_capacity, sparse.consumer_dispatches,
        static_cast<unsigned long long>(sparse.consumer_workgroups),
        static_cast<unsigned long long>(sparse.legacy_candidate_flag_bytes),
        static_cast<unsigned long long>(sparse.bounded_candidate_bytes),
        plan.scratch_bytes, plan.output_bytes);

    const auto dense = rigid_spatial_work_budget(386U, 385U);
    CHECK(dense.candidate_slot_capacity == 386U * 385U);
    CHECK(dense.consumer_dispatches == sparse.consumer_dispatches);
    CHECK(dense.bounded_pair_output_bytes >=
          386ULL * 385ULL * sizeof(std::uint32_t) + sizeof(std::uint32_t));
    CHECK(make_rigid_spatial_plan(386U, 385U, plan) == PM_SUCCESS);
    CHECK(dense.bounded_pair_scratch_bytes == plan.scratch_bytes);
    CHECK(dense.bounded_pair_output_bytes == plan.output_bytes);
    std::printf(
        "F8_RIGID_BUDGET dense bodies=%u candidates_per_body=%u "
        "candidate_slots=%u consumer_dispatches=%u consumer_workgroups=%llu "
        "legacy_flag_bytes=%llu candidate_bytes=%llu pair_scratch_bytes=%zu "
        "pair_output_bytes=%zu\n",
        dense.body_count, dense.candidates_per_body,
        dense.candidate_slot_capacity, dense.consumer_dispatches,
        static_cast<unsigned long long>(dense.consumer_workgroups),
        static_cast<unsigned long long>(dense.legacy_candidate_flag_bytes),
        static_cast<unsigned long long>(dense.bounded_candidate_bytes),
        plan.scratch_bytes, plan.output_bytes);

    CHECK(make_rigid_spatial_plan(0U, 0U, plan) == PM_SUCCESS);
    CHECK(make_rigid_spatial_plan(1U, 0U, plan) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(make_rigid_spatial_plan(std::numeric_limits<std::uint32_t>::max(),
                                  2U, plan) ==
          PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(make_rigid_spatial_plan(65537U, 1U, plan) ==
          PM_ERROR_CAPACITY_EXCEEDED);
}

void test_proxy_envelopes() {
    const float zero[3]{};
    const float deviation[3]{0.05F, 0.1F, 0.15F};
    const float correction[3]{0.25F, 0.25F, 0.25F};
    const float fast_start[3]{-10.0F, 0.0F, 0.0F};
    const float fast_end[3]{10.0F, 0.0F, 0.0F};
    SpatialAabb fast{};
    CHECK(rigid_spatial_proxy_envelope(
              fast_start, fast_end, zero, 0.2F, 0.01F,
              deviation, correction, 1.0e-5F, fast) == PM_SUCCESS);
    CHECK(spatial_overlaps(fast, box(-0.1F, -0.1F, -0.1F,
                                     0.1F, 0.1F, 0.1F)));

    // The cached sphere center is offset from the rotation origin. Its
    // center length is part of the proxy reach, so a full turn stays inside.
    const float offset_center[3]{2.0F, 0.0F, 0.0F};
    SpatialAabb rotating{};
    CHECK(rigid_spatial_proxy_envelope(
              zero, zero, offset_center, 0.1F, 0.0F, zero, zero,
              1.0e-5F, rotating) == PM_SUCCESS);
    for (std::uint32_t degree = 0U; degree <= 360U; ++degree) {
        const float angle = degree * 3.14159265358979323846F / 180.0F;
        const float x = 2.0F * std::cos(angle);
        const float y = 2.0F * std::sin(angle);
        CHECK(spatial_contains(rotating, box(x, y, 0.0F, x, y, 0.0F)));
    }

    const float left_position[3]{0.0F, 0.0F, 0.0F};
    const float right_position[3]{0.6F, 0.0F, 0.0F};
    SpatialAabb left{}, right{};
    CHECK(rigid_spatial_proxy_envelope(
              left_position, left_position, zero, 0.2F, 0.11F, zero, zero,
              1.0e-5F, left) == PM_SUCCESS);
    CHECK(rigid_spatial_proxy_envelope(
              right_position, right_position, zero, 0.2F, 0.11F, zero, zero,
              1.0e-5F, right) == PM_SUCCESS);
    CHECK(spatial_overlaps(left, right));

    const float bad_center[3]{std::numeric_limits<float>::infinity(), 0, 0};
    CHECK(rigid_spatial_proxy_envelope(
              zero, zero, bad_center, 1.0F, 0.0F, zero, zero, 0.0F,
              left) == PM_ERROR_INVALID_ARGUMENT);
}

struct Body {
    SpatialIdentity id{};
    SpatialAabb bounds{};
    std::uint32_t fixed_root{};
    bool dynamic{};
};

std::set<std::uint32_t> brute_force_pairs(
    std::span<const Body> bodies,
    const std::set<std::pair<std::uint32_t, std::uint32_t>> &disabled) {
    std::set<std::uint32_t> result;
    const auto count = static_cast<std::uint32_t>(bodies.size());
    for (std::uint32_t a = 0U; a < count; ++a) {
        if (!bodies[a].dynamic) continue;
        for (std::uint32_t b = 0U; b < count; ++b) {
            if (a == b || !spatial_overlaps(bodies[a].bounds,
                                            bodies[b].bounds)) continue;
            if (bodies[b].dynamic && !(bodies[a].id < bodies[b].id)) continue;
            if (bodies[a].fixed_root != std::numeric_limits<std::uint32_t>::max() &&
                bodies[a].fixed_root == bodies[b].fixed_root) continue;
            const auto owners = std::minmax(bodies[a].id.owner_slot,
                                            bodies[b].id.owner_slot);
            if (disabled.contains({owners.first, owners.second})) continue;
            result.insert(a * count + b);
        }
    }
    return result;
}

void test_shared_queries_match_oracle() {
    const float zero[3]{};
    std::vector<Body> bodies(10U);
    for (std::uint32_t index = 0U; index < bodies.size(); ++index) {
        bodies[index].id = identity(100U + index, 7U + index,
                                    20U + index, 30U + index);
        bodies[index].fixed_root =
            std::numeric_limits<std::uint32_t>::max();
    }

    const float fast_start[3]{-5.0F, 0.0F, 0.0F};
    const float fast_end[3]{5.0F, 0.0F, 0.0F};
    CHECK(rigid_spatial_proxy_envelope(fast_start, fast_end, zero, 0.2F,
        0.0F, zero, zero, 1.0e-5F, bodies[0].bounds) == PM_SUCCESS);
    bodies[0].dynamic = true;
    bodies[1].bounds = box(-0.2F, -0.2F, -0.2F, 0.2F, 0.2F, 0.2F);

    const float rotation_center[3]{2.0F, 0.0F, 0.0F};
    CHECK(rigid_spatial_proxy_envelope(zero, zero, rotation_center, 0.1F,
        0.0F, zero, zero, 1.0e-5F, bodies[2].bounds) == PM_SUCCESS);
    bodies[2].dynamic = true;
    bodies[3].bounds = box(-0.1F, 1.9F, -0.1F, 0.1F, 2.1F, 0.1F);

    const float margin_a[3]{10.0F, 0.0F, 0.0F};
    const float margin_b[3]{10.6F, 0.0F, 0.0F};
    CHECK(rigid_spatial_proxy_envelope(margin_a, margin_a, zero, 0.2F,
        0.11F, zero, zero, 1.0e-5F, bodies[4].bounds) == PM_SUCCESS);
    CHECK(rigid_spatial_proxy_envelope(margin_b, margin_b, zero, 0.2F,
        0.11F, zero, zero, 1.0e-5F, bodies[5].bounds) == PM_SUCCESS);
    bodies[4].dynamic = true;

    bodies[6].bounds = box(20, 0, 0, 21, 1, 1);
    bodies[7].bounds = box(20, 0, 0, 21, 1, 1);
    bodies[6].dynamic = true;
    bodies[8].bounds = box(30, 0, 0, 31, 1, 1);
    bodies[9].bounds = box(30, 0, 0, 31, 1, 1);
    bodies[8].dynamic = true;
    bodies[8].fixed_root = bodies[9].fixed_root = 55U;
    const std::set<std::pair<std::uint32_t, std::uint32_t>> disabled{
        {bodies[6].id.owner_slot, bodies[7].id.owner_slot}};

    const std::uint32_t count = static_cast<std::uint32_t>(bodies.size());
    std::vector<SpatialLeafMetadata> metadata(count);
    std::vector<std::uint8_t> admitted(count, 1U);
    for (std::uint32_t index = 0U; index < count; ++index) {
        metadata[index].identity = bodies[index].id;
        metadata[index].sort_key = index + 1U;
        metadata[index].payload_ref = index;
        metadata[index].query_mask = rigid_spatial_query_mask;
        metadata[index].admission_mask = rigid_spatial_query_mask;
        metadata[index].flags = spatial_leaf_retained;
    }

    std::vector<std::uint32_t> left(count - 1U), right(count - 1U),
        sources(count);
    std::vector<SpatialAabb> nodes(2U * count - 1U);
    std::uint32_t next_internal{};
    auto build = [&](auto &&self, std::uint32_t begin,
                     std::uint32_t end) -> std::uint32_t {
        if (end - begin == 1U) {
            sources[begin] = begin;
            const auto node = count - 1U + begin;
            nodes[node] = bodies[begin].bounds;
            return node;
        }
        const auto node = next_internal++;
        const auto middle = begin + (end - begin) / 2U;
        left[node] = self(self, begin, middle);
        right[node] = self(self, middle, end);
        nodes[node] = spatial_union(nodes[left[node]], nodes[right[node]]);
        return node;
    };
    CHECK(build(build, 0U, count) == 0U);
    SpatialTreeView tree{left, right, sources, nodes, metadata, admitted, count};
    std::array<std::uint32_t, 64> traversal{};
    std::vector<std::uint32_t> candidates(count);
    std::set<std::uint32_t> actual;
    for (std::uint32_t query = 0U; query < count; ++query) {
        if (!bodies[query].dynamic) continue;
        std::uint32_t written{}, visited{};
        CHECK(spatial_query_aabb(tree, bodies[query].bounds,
            rigid_spatial_query_mask, metadata[query].sort_key, traversal,
            candidates, written, visited) == PM_SUCCESS);
        for (std::uint32_t index = 0U; index < written; ++index) {
            const auto target = metadata[candidates[index]].payload_ref;
            if (bodies[target].dynamic &&
                !(bodies[query].id < bodies[target].id)) continue;
            if (bodies[query].fixed_root !=
                    std::numeric_limits<std::uint32_t>::max() &&
                bodies[query].fixed_root == bodies[target].fixed_root) continue;
            const auto owners = std::minmax(bodies[query].id.owner_slot,
                                            bodies[target].id.owner_slot);
            if (disabled.contains({owners.first, owners.second})) continue;
            actual.insert(query * count + target);
        }
    }
    const auto expected = brute_force_pairs(bodies, disabled);
    CHECK(actual == expected);
    CHECK(actual.contains(0U * count + 1U));
    CHECK(actual.contains(2U * count + 3U));
    CHECK(actual.contains(4U * count + 5U));
    CHECK(!actual.contains(6U * count + 7U));
    CHECK(!actual.contains(8U * count + 9U));

    // Reusing an owner slot with a new generation is a distinct epoch leaf.
    auto replacement = metadata;
    replacement[1].identity.owner_generation += 1U;
    CHECK(!spatial_can_refit(metadata, replacement));
    CHECK(replacement[1].identity != metadata[1].identity);
}

void test_one_epoch_for_all_sweeps() {
    SpatialEpochToken token{9U, 3U, 4U, 1U, 0U, 7U};
    SpatialEpochCounters counters{};
    counters.builds = 1U;
    counters.publications = 1U;
    counters.queries = 10U;
    counters.candidates = 25U;
    RigidSpatialSolveEpoch use{};
    CHECK(use.begin(token, counters) == PM_SUCCESS);
    for (std::uint32_t sweep = 0U;
         sweep < rigid_spatial_solve_sweeps; ++sweep)
        CHECK(use.record_sweep(token, counters) == PM_SUCCESS);
    CHECK(use.finish() == PM_SUCCESS);

    CHECK(use.begin(token, counters) == PM_SUCCESS);
    auto changed = counters;
    ++changed.refits;
    CHECK(use.record_sweep(token, changed) == PM_ERROR_INVALID_STATE);
    auto other = token;
    ++other.generation;
    CHECK(use.record_sweep(other, counters) == PM_ERROR_INVALID_STATE);
    CHECK(use.finish() == PM_ERROR_INVALID_STATE);
}

} // namespace

int main() {
    test_plan_and_budgets();
    test_proxy_envelopes();
    test_shared_queries_match_oracle();
    test_one_epoch_for_all_sweeps();
    if (failures == 0)
        std::printf("F8-08 rigid shared-spatial contract tests passed\n");
    return failures == 0 ? 0 : 1;
}
