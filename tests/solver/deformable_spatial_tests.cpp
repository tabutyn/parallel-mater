// SPDX-License-Identifier: MIT
#include "simulation/deformable_spatial.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
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

void test_plan_and_budget() {
    DeformableSpatialPlan plan{};
    CHECK(make_deformable_spatial_plan(10240U, 64U, plan) == PM_SUCCESS);
    CHECK(plan.candidate_slot_capacity == 10240U * 64U);
    CHECK(plan.query_leaf_sources.size == 10240U * sizeof(std::uint32_t));
    CHECK(plan.filtered_counts.size == 10240U * sizeof(std::uint32_t));
    CHECK(plan.filtered_sources.size ==
          10240ULL * 64ULL * sizeof(std::uint32_t));
    CHECK(make_deformable_spatial_plan(0U, 0U, plan) == PM_SUCCESS);
    CHECK(make_deformable_spatial_plan(1U, 0U, plan) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(make_deformable_spatial_plan(
              std::numeric_limits<std::uint32_t>::max(), 2U, plan) ==
          PM_ERROR_CAPACITY_EXCEEDED);

    const auto sparse = deformable_spatial_work_budget(
        512U, 8192U, 2048U, 3U, 64U);
    CHECK(sparse.boundary_dispatches == 7U);
    CHECK(sparse.spatial_updates == 1U);
    CHECK(sparse.index_dispatches_in_solve == 0U);
    CHECK(sparse.per_edge_rebuild_dispatches == 0U);
    CHECK(sparse.live_query_workgroups == 40U);
    CHECK(sparse.capacity_query_workgroups == 160U);
    CHECK(sparse.bounded_candidate_bytes == 2621440ULL);
    CHECK(sparse.avoided_per_consumer_bounds_bytes == 1048576ULL);
    std::printf(
        "F8_DEFORMABLE_BUDGET live=%u capacity=%u smoke_queries=%u "
        "domains=%u candidates=%u boundary_dispatches=%u updates=%u "
        "solve_index_dispatches=%u per_edge_rebuilds=%u "
        "live_query_workgroups=%llu capacity_query_workgroups=%llu "
        "candidate_bytes=%llu avoided_bounds_bytes=%llu\n",
        sparse.live_primitives, sparse.primitive_capacity,
        sparse.smoke_obstacle_queries, sparse.deformable_domains,
        sparse.candidates_per_query, sparse.boundary_dispatches,
        sparse.spatial_updates, sparse.index_dispatches_in_solve,
        sparse.per_edge_rebuild_dispatches,
        static_cast<unsigned long long>(sparse.live_query_workgroups),
        static_cast<unsigned long long>(sparse.capacity_query_workgroups),
        static_cast<unsigned long long>(sparse.bounded_candidate_bytes),
        static_cast<unsigned long long>(
            sparse.avoided_per_consumer_bounds_bytes));

    const auto full = deformable_spatial_work_budget(
        8192U, 8192U, 0U, 3U, 64U);
    CHECK(full.boundary_dispatches == 5U);
    CHECK(full.live_query_workgroups == 128U);
    CHECK(full.live_query_workgroups == full.capacity_query_workgroups);
    CHECK(full.bounded_candidate_bytes == 2097152ULL);
    std::printf(
        "F8_DEFORMABLE_BUDGET full live=%u capacity=%u smoke_queries=%u "
        "domains=%u candidates=%u boundary_dispatches=%u "
        "live_query_workgroups=%llu capacity_query_workgroups=%llu "
        "candidate_bytes=%llu avoided_bounds_bytes=%llu\n",
        full.live_primitives, full.primitive_capacity,
        full.smoke_obstacle_queries, full.deformable_domains,
        full.candidates_per_query, full.boundary_dispatches,
        static_cast<unsigned long long>(full.live_query_workgroups),
        static_cast<unsigned long long>(full.capacity_query_workgroups),
        static_cast<unsigned long long>(full.bounded_candidate_bytes),
        static_cast<unsigned long long>(
            full.avoided_per_consumer_bounds_bytes));
}

SpatialAabb exact_bounds(
    std::span<const std::array<float, 3>> points, float thickness) {
    SpatialAabb result{};
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        result.minimum[axis] = points[0][axis] - thickness;
        result.maximum[axis] = points[0][axis] + thickness;
        for (std::size_t point = 1U; point < points.size(); ++point) {
            result.minimum[axis] = std::min(
                result.minimum[axis], points[point][axis] - thickness);
            result.maximum[axis] = std::max(
                result.maximum[axis], points[point][axis] + thickness);
        }
    }
    return result;
}

void test_motion_envelope_completeness() {
    std::mt19937 random(0xf810U);
    std::uniform_real_distribution<float> position(-3.0F, 3.0F);
    std::uniform_real_distribution<float> movement(-0.9F, 0.9F);
    std::uniform_real_distribution<float> budget(0.0F, 0.18F);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    std::uint64_t contained{};
    for (std::uint32_t primitive = 0U; primitive < 64U; ++primitive) {
        const std::size_t count = primitive % 2U == 0U ? 2U : 3U;
        std::vector<DeformableSpatialPointMotion> motion(count);
        for (auto &vertex : motion) {
            for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
                vertex.previous[axis] = position(random);
                vertex.predicted[axis] =
                    vertex.previous[axis] + movement(random);
                vertex.path_deviation[axis] = budget(random);
                vertex.correction_budget[axis] = budget(random);
            }
        }
        SpatialAabb envelope{};
        CHECK(deformable_spatial_primitive_envelope(
                  motion, 0.015F, 1.0e-5F, envelope) == PM_SUCCESS);
        for (std::uint32_t sample = 0U; sample < 64U; ++sample) {
            std::vector<std::array<float, 3>> current(count);
            const float fraction = unit(random);
            for (std::size_t vertex = 0U; vertex < count; ++vertex) {
                for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
                    const float reach =
                        motion[vertex].path_deviation[axis] +
                        motion[vertex].correction_budget[axis];
                    const float offset = (2.0F * unit(random) - 1.0F) * reach;
                    current[vertex][axis] = motion[vertex].previous[axis] +
                        fraction * (motion[vertex].predicted[axis] -
                                    motion[vertex].previous[axis]) + offset;
                }
            }
            CHECK(spatial_contains(envelope, exact_bounds(current, 0.015F)));
            ++contained;
        }
    }
    CHECK(contained == 4096U);

    // A thin segment crossing a triangle between endpoints is retained by the
    // full-tick primitive envelopes even though neither endpoint overlaps.
    std::array<DeformableSpatialPointMotion, 2> segment{};
    segment[0].previous = {-1.0F, 0.0F, 0.0F};
    segment[0].predicted = {1.0F, 0.0F, 0.0F};
    segment[1].previous = {-1.0F, 0.02F, 0.0F};
    segment[1].predicted = {1.0F, 0.02F, 0.0F};
    std::array<DeformableSpatialPointMotion, 3> triangle{};
    triangle[0].previous = triangle[0].predicted = {0.0F, -0.5F, -0.5F};
    triangle[1].previous = triangle[1].predicted = {0.0F, 0.5F, -0.5F};
    triangle[2].previous = triangle[2].predicted = {0.0F, 0.0F, 0.5F};
    SpatialAabb segment_bounds{}, triangle_bounds{};
    CHECK(deformable_spatial_primitive_envelope(
              segment, 0.01F, 1.0e-5F, segment_bounds) == PM_SUCCESS);
    CHECK(deformable_spatial_primitive_envelope(
              triangle, 0.01F, 1.0e-5F, triangle_bounds) == PM_SUCCESS);
    CHECK(spatial_overlaps(segment_bounds, triangle_bounds));

    SpatialAabb invalid{};
    CHECK(deformable_spatial_primitive_envelope(
              std::span<const DeformableSpatialPointMotion>{}, 0.0F,
              1.0e-5F, invalid) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(deformable_spatial_primitive_envelope(
              segment, -1.0F, 1.0e-5F, invalid) ==
          PM_ERROR_INVALID_ARGUMENT);
}

bool share_vertex(std::span<const std::uint32_t> left,
                  std::span<const std::uint32_t> right) {
    for (const auto a : left)
        for (const auto b : right)
            if (a == b) return true;
    return false;
}

void test_identities_exclusions_and_static_geometry() {
    SpatialLeafMetadata cloth{};
    cloth.identity = {deformable_spatial_cloth_family, 4U, 7U, 11U,
        deformable_spatial_triangle_primitive, 2U, 9U};
    cloth.sort_key = 3U;
    cloth.query_mask = deformable_spatial_contact_mask |
        smoke_spatial_obstacle_mask;
    cloth.admission_mask = cloth.query_mask;
    cloth.flags = spatial_leaf_retained;
    CHECK(spatial_identity_valid(cloth.identity));

    auto regenerated = cloth.identity;
    ++regenerated.owner_generation;
    CHECK(regenerated != cloth.identity);
    auto retopologized = cloth.identity;
    ++retopologized.topology_revision;
    CHECK(retopologized != cloth.identity);
    auto replaced = cloth.identity;
    ++replaced.primitive_generation;
    CHECK(replaced != cloth.identity);

    const std::array<std::uint32_t, 3> triangle_a{0U, 1U, 2U};
    const std::array<std::uint32_t, 3> triangle_b{2U, 3U, 4U};
    const std::array<std::uint32_t, 3> triangle_c{5U, 6U, 7U};
    const std::array<std::uint32_t, 2> rope_a{10U, 11U};
    const std::array<std::uint32_t, 2> rope_b{11U, 12U};
    CHECK(share_vertex(triangle_a, triangle_b));
    CHECK(!share_vertex(triangle_a, triangle_c));
    CHECK(share_vertex(rope_a, rope_b));

    std::array<DeformableSpatialPointMotion, 3> stationary{};
    stationary[0].previous = stationary[0].predicted = {-1.0F, 0.0F, 0.0F};
    stationary[1].previous = stationary[1].predicted = {1.0F, 0.0F, 0.0F};
    stationary[2].previous = stationary[2].predicted = {0.0F, 1.0F, 0.0F};
    SpatialAabb static_bounds{};
    CHECK(deformable_spatial_primitive_envelope(
              stationary, 0.0F, 1.0e-5F, static_bounds) == PM_SUCCESS);
    CHECK(static_bounds.minimum[0] < -1.0F);
    CHECK(static_bounds.maximum[1] > 1.0F);
    CHECK(deformable_spatial_family_valid(
        deformable_spatial_static_family));
}

void test_one_epoch_for_all_consumers() {
    const SpatialEpochToken token{84U, 12U, 20U, 2U, 0U, 5U};
    SpatialEpochCounters counters{0U, 1U, 1U, 144U, 0U, 9216U};
    DeformableSpatialSolveEpoch epoch;
    CHECK(epoch.begin(token, counters) == PM_SUCCESS);
    for (std::uint32_t pass = 0U;
         pass < deformable_spatial_collision_passes; ++pass)
        CHECK(epoch.record_pass(token, counters) == PM_SUCCESS);
    CHECK(epoch.finish() == PM_SUCCESS);

    DeformableSpatialSolveEpoch rebuilt;
    CHECK(rebuilt.begin(token, counters) == PM_SUCCESS);
    auto changed = counters;
    ++changed.refits;
    CHECK(rebuilt.record_pass(token, changed) == PM_ERROR_INVALID_STATE);

    DeformableSpatialSolveEpoch requeried;
    CHECK(requeried.begin(token, counters) == PM_SUCCESS);
    changed = counters;
    ++changed.queries;
    CHECK(requeried.record_pass(token, changed) == PM_ERROR_INVALID_STATE);
}

} // namespace

int main() {
    test_plan_and_budget();
    test_motion_envelope_completeness();
    test_identities_exclusions_and_static_geometry();
    test_one_epoch_for_all_consumers();
    if (failures == 0)
        std::printf("F8-10 host deformable/smoke spatial contract passed\n");
    return failures == 0 ? 0 : 1;
}
