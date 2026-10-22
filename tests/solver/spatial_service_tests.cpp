// SPDX-License-Identifier: MIT
#include "simulation/spatial_service.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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

SpatialIdentity identity(std::uint32_t slot, std::uint32_t owner = 0U,
                         std::uint32_t owner_generation = 1U,
                         std::uint32_t primitive_generation = 1U) {
    return {1U, owner, owner_generation, 1U, 1U, slot,
            primitive_generation};
}

SpatialLeafMetadata leaf(std::uint32_t slot, std::uint32_t sort_key,
                         std::uint32_t flags = spatial_leaf_retained,
                         std::uint32_t admission_mask = 1U,
                         std::uint32_t transfer_group = 0U) {
    SpatialLeafMetadata result{};
    result.identity = identity(slot);
    result.sort_key = sort_key;
    result.payload_ref = slot;
    result.query_mask = 1U << (slot % 3U);
    result.admission_mask = admission_mask;
    result.flags = flags;
    result.transfer_group = transfer_group;
    return result;
}

void test_plan() {
    SpatialServicePlan plan{};
    CHECK(make_spatial_service_plan(30065U, 64U, 128U, 3U, plan) ==
          PM_SUCCESS);
    CHECK(plan.leaf_capacity == 30065U);
    CHECK(plan.snapshot_count == 3U);
    CHECK(plan.build.node_capacity == 60129U);
    CHECK(plan.persistent_bytes > 3U * plan.build.output_bytes);
    CHECK(plan.scratch_bytes == plan.build.scratch_bytes);
    for (std::uint32_t index = 0U; index < plan.snapshot_count; ++index) {
        const auto &slot = plan.snapshots[index];
        CHECK(slot.header.offset % 16U == 0U);
        CHECK(slot.metadata.offset % 16U == 0U);
        CHECK(slot.bounds.offset % 16U == 0U);
        CHECK(slot.build_output.offset + slot.build_output.size <=
              plan.persistent_bytes);
        CHECK(slot.query_candidates.size ==
              64U * 128U * sizeof(std::uint32_t));
        CHECK(slot.query_visited_nodes.size ==
              64U * sizeof(std::uint32_t));
        CHECK(slot.query_counter.size == sizeof(std::uint32_t));
        CHECK(slot.query_status.offset + slot.query_status.size <=
              plan.persistent_bytes);
        if (index != 0U)
            CHECK(plan.snapshots[index - 1U].query_status.offset +
                  plan.snapshots[index - 1U].query_status.size <=
                  slot.header.offset);
    }
    CHECK(make_spatial_service_plan(4U, 0U, 1U, 3U, plan) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(make_spatial_service_plan(4U, 1U, 1U, 1U, plan) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(make_spatial_service_plan(spatial_max_capacity + 1U, 1U, 1U, 2U,
                                    plan) == PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(make_spatial_service_plan(
              4U, std::numeric_limits<std::uint32_t>::max(), 2U, 2U, plan) ==
          PM_ERROR_CAPACITY_EXCEEDED);
}

void test_envelopes() {
    const float start[3]{-1.0F, 2.0F, 0.5F};
    const float predicted[3]{3.0F, -2.0F, 1.5F};
    const float deviation[3]{0.1F, 0.2F, 0.3F};
    const float correction[3]{0.4F, 0.5F, 0.6F};
    SpatialAabb envelope{};
    CHECK(spatial_point_envelope(start, predicted, deviation, correction,
                                 0.25F, envelope) == PM_SUCCESS);
    CHECK(envelope.minimum[0] < -1.75F);
    CHECK(envelope.maximum[1] > 2.95F);

    std::mt19937 random(0xF807U);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    for (std::uint32_t sample = 0U; sample < 1000U; ++sample) {
        const float t = unit(random);
        float point[3]{};
        for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
            const float path = start[axis] + t * (predicted[axis] - start[axis]);
            const float shift = (2.0F * unit(random) - 1.0F) *
                (deviation[axis] + correction[axis] + 0.25F);
            point[axis] = path + shift;
            CHECK(point[axis] >= envelope.minimum[axis]);
            CHECK(point[axis] <= envelope.maximum[axis]);
        }
    }

    // A full rigid turn cannot use endpoint-only vertex bounds. The enclosing
    // radius around the rotation origin covers every intermediate orientation.
    const float origin[3]{0.0F, 0.0F, 0.0F};
    const float zero[3]{};
    SpatialAabb rotating{};
    CHECK(spatial_point_envelope(origin, origin, zero, zero, 2.5F,
                                 rotating) == PM_SUCCESS);
    for (std::uint32_t degree = 0U; degree <= 360U; ++degree) {
        const float angle = float(degree) * 3.14159265358979323846F / 180.0F;
        const auto point = box(2.5F * std::cos(angle),
                               2.5F * std::sin(angle), 0.0F,
                               2.5F * std::cos(angle),
                               2.5F * std::sin(angle), 0.0F);
        CHECK(spatial_contains(rotating, point));
    }

    const std::array sources{
        box(0, 0, 0, 1, 1, 1), box(2, 2, 2, 3, 3, 3)};
    const std::array weights{2.0F, -1.0F};
    SpatialAabb weighted{};
    CHECK(spatial_weighted_envelope(sources, weights, weighted) == PM_SUCCESS);
    CHECK(weighted.minimum[0] < -3.0F);
    CHECK(weighted.maximum[0] > 0.0F);
    const std::array bad_weights{2.0F,
        std::numeric_limits<float>::infinity()};
    CHECK(spatial_weighted_envelope(sources, bad_weights, weighted) ==
          PM_ERROR_INVALID_ARGUMENT);
}

void test_admission() {
    std::vector<SpatialLeafMetadata> metadata{
        leaf(0U, 10U),
        leaf(1U, 20U, spatial_leaf_provisional),
        leaf(2U, 30U, spatial_leaf_provisional),
        leaf(3U, 40U, spatial_leaf_provisional)};
    std::vector<SpatialAabb> bounds{
        box(0, 0, 0, 1, 1, 1),
        box(0.5F, 0, 0, 1.5F, 1, 1),
        box(3, 0, 0, 4, 1, 1),
        box(3.5F, 0, 0, 4.5F, 1, 1)};
    std::vector<std::uint8_t> admitted(metadata.size());
    CHECK(spatial_resolve_admission(metadata, bounds, admitted) == PM_SUCCESS);
    CHECK((admitted == std::vector<std::uint8_t>{1U, 0U, 1U, 0U}));

    // Removed/tombstoned slots do not participate in identity validation or
    // admission, and do not disturb stable source indices during compaction.
    metadata.insert(metadata.begin() + 2U, SpatialLeafMetadata{});
    metadata[2].flags = spatial_leaf_inactive;
    bounds.insert(bounds.begin() + 2U, SpatialAabb{});
    admitted.resize(metadata.size());
    CHECK(spatial_resolve_admission(metadata, bounds, admitted) == PM_SUCCESS);
    CHECK(admitted[2] == 0U);
    metadata.erase(metadata.begin() + 2U);
    bounds.erase(bounds.begin() + 2U);
    admitted.resize(metadata.size());

    // Input order does not change the lower-key priority decision.
    std::reverse(metadata.begin() + 1, metadata.end());
    std::reverse(bounds.begin() + 1, bounds.end());
    CHECK(spatial_resolve_admission(metadata, bounds, admitted) == PM_SUCCESS);
    for (std::size_t index = 0U; index < metadata.size(); ++index) {
        const auto key = metadata[index].sort_key;
        CHECK(admitted[index] == (key == 10U || key == 30U));
    }

    metadata = {leaf(9U, 10U, spatial_leaf_provisional),
                leaf(1U, 20U, spatial_leaf_provisional)};
    bounds = {box(0, 0, 0, 1, 1, 1), box(0, 0, 0, 1, 1, 1)};
    admitted.resize(2U);
    CHECK(spatial_resolve_admission(metadata, bounds, admitted) == PM_SUCCESS);
    CHECK((admitted == std::vector<std::uint8_t>{0U, 1U}));

    std::vector<SpatialLeafMetadata> transfer{
        leaf(0U, 10U, spatial_leaf_retained | spatial_leaf_transfer_source,
             1U, 7U),
        leaf(1U, 20U, spatial_leaf_provisional |
             spatial_leaf_transfer_destination, 1U, 7U)};
    std::vector<SpatialAabb> transfer_bounds{
        box(0, 0, 0, 1, 1, 1), box(0, 0, 0, 1, 1, 1)};
    admitted.resize(2U);
    CHECK(spatial_resolve_admission(transfer, transfer_bounds, admitted) ==
          PM_SUCCESS);
    CHECK((admitted == std::vector<std::uint8_t>{0U, 1U}));
    transfer.push_back(leaf(2U, 30U,
        spatial_leaf_provisional | spatial_leaf_transfer_destination, 1U, 7U));
    transfer_bounds.push_back(box(2, 0, 0, 3, 1, 1));
    admitted.resize(3U);
    CHECK(spatial_resolve_admission(transfer, transfer_bounds, admitted) ==
          PM_ERROR_INVALID_ARGUMENT);

    transfer = {
        leaf(0U, 10U, spatial_leaf_retained | spatial_leaf_transfer_source,
             1U, 9U),
        leaf(1U, 20U, spatial_leaf_retained | spatial_leaf_transfer_source,
             1U, 9U),
        leaf(2U, 30U, spatial_leaf_provisional |
             spatial_leaf_transfer_destination, 1U, 9U)};
    transfer_bounds = {box(0, 0, 0, 1, 1, 1), box(2, 0, 0, 3, 1, 1),
                       box(4, 0, 0, 5, 1, 1)};
    admitted.resize(3U);
    CHECK(spatial_resolve_admission(transfer, transfer_bounds, admitted) ==
          PM_ERROR_INVALID_ARGUMENT);

    metadata = {leaf(0U, 10U), leaf(1U, 10U)};
    bounds = {box(0, 0, 0, 1, 1, 1), box(2, 0, 0, 3, 1, 1)};
    admitted.resize(2U);
    CHECK(spatial_resolve_admission(metadata, bounds, admitted) ==
          PM_ERROR_INVALID_ARGUMENT);
    metadata = {leaf(0U, 10U, spatial_leaf_retained |
        spatial_leaf_provisional)};
    bounds = {box(0, 0, 0, 1, 1, 1)};
    admitted.resize(1U);
    CHECK(spatial_resolve_admission(metadata, bounds, admitted) ==
          PM_ERROR_INVALID_ARGUMENT);

    std::array previous{leaf(0U, 10U), leaf(1U, 20U)};
    auto staged = previous;
    CHECK(spatial_can_refit(previous, staged));
    staged[1].identity.primitive_generation += 1U;
    CHECK(!spatial_can_refit(previous, staged));
    staged = previous;
    staged[1].flags |= spatial_leaf_inactive;
    CHECK(!spatial_can_refit(previous, staged));
}

void test_queries() {
    // Root 0 owns internal node 1 and leaf C (4); node 1 owns leaves
    // A (2) and B (3).
    std::vector<SpatialLeafMetadata> metadata{
        leaf(0U, 30U), leaf(1U, 10U), leaf(2U, 20U)};
    metadata[0].identity = identity(2U);
    metadata[1].identity = identity(0U);
    metadata[2].identity = identity(1U);
    metadata[0].query_mask = metadata[1].query_mask =
        metadata[2].query_mask = 1U;
    const std::vector<SpatialAabb> nodes{
        box(0, 0, 0, 5, 1, 1), box(0, 0, 0, 3, 1, 1),
        box(0, 0, 0, 1, 1, 1), box(2, 0, 0, 3, 1, 1),
        box(4, 0, 0, 5, 1, 1)};
    const std::array<std::uint32_t, 2> left{1U, 2U};
    const std::array<std::uint32_t, 2> right{4U, 3U};
    // Leaf ranks 0,1,2 map to metadata sources 1,2,0.
    const std::array<std::uint32_t, 3> sources{1U, 2U, 0U};
    std::vector<std::uint8_t> admitted{1U, 1U, 1U};
    SpatialTreeView tree{left, right, sources, nodes, metadata, admitted, 3U};
    std::array<std::uint32_t, 8> traversal{};
    std::array<std::uint32_t, 3> output{};
    std::uint32_t written{}, visited{};
    CHECK(spatial_query_aabb(tree, box(1, 0, 0, 4, 1, 1), 1U,
                             spatial_invalid_index, traversal, output,
                             written, visited) == PM_SUCCESS);
    CHECK(written == 3U && visited == 5U);
    CHECK(metadata[output[0]].identity == identity(0U));
    CHECK(metadata[output[1]].identity == identity(1U));
    CHECK(metadata[output[2]].identity == identity(2U));

    CHECK(spatial_query_aabb(tree, box(1, 0, 0, 4, 1, 1), 1U, 20U,
                             traversal, output, written, visited) == PM_SUCCESS);
    CHECK(written == 2U);
    admitted[2] = 0U;
    CHECK(spatial_query_aabb(tree, box(1, 0, 0, 4, 1, 1), 1U,
                             spatial_invalid_index, traversal, output,
                             written, visited) == PM_SUCCESS);
    CHECK(written == 2U);

    std::array<std::uint32_t, 1> too_small{};
    CHECK(spatial_query_aabb(tree, box(0, 0, 0, 5, 1, 1), 1U,
                             spatial_invalid_index, traversal, too_small,
                             written, visited) == PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(written == 0U);
    std::array<std::uint32_t, 1> short_stack{};
    CHECK(spatial_query_aabb(tree, box(0, 0, 0, 5, 1, 1), 1U,
                             spatial_invalid_index, short_stack, output,
                             written, visited) == PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(written == 0U);
}

void test_candidate_completeness() {
    constexpr std::uint32_t count = 37U;
    std::mt19937 random(0xB8F807U);
    std::uniform_real_distribution<float> coordinate(-8.0F, 8.0F);
    std::uniform_real_distribution<float> extent(0.0F, 0.8F);
    std::vector<SpatialLeafMetadata> metadata;
    std::vector<SpatialAabb> leaf_bounds(count);
    metadata.reserve(count);
    for (std::uint32_t index = 0U; index < count; ++index) {
        metadata.push_back(leaf(index, index + 1U));
        metadata.back().query_mask = 1U;
        float start[3]{coordinate(random), coordinate(random),
                       coordinate(random)};
        float predicted[3]{coordinate(random), coordinate(random),
                           coordinate(random)};
        float deviation[3]{extent(random), extent(random), extent(random)};
        float correction[3]{extent(random), extent(random), extent(random)};
        CHECK(spatial_point_envelope(start, predicted, deviation, correction,
                                     extent(random), leaf_bounds[index]) ==
              PM_SUCCESS);
    }
    // A full-turn rigid proxy uses its rotation-origin radius, so every
    // orientation participates in the same brute-force overlap oracle.
    const float origin[3]{};
    const float zero[3]{};
    CHECK(spatial_point_envelope(origin, origin, zero, zero, 5.0F,
                                 leaf_bounds[0]) == PM_SUCCESS);

    std::vector<std::uint32_t> left(count - 1U), right(count - 1U);
    std::vector<std::uint32_t> sources(count);
    std::vector<SpatialAabb> nodes(2U * count - 1U);
    std::uint32_t next_internal{};
    auto build = [&](auto &&self, std::uint32_t begin,
                     std::uint32_t end) -> std::uint32_t {
        if (end - begin == 1U) {
            sources[begin] = begin;
            const auto node = count - 1U + begin;
            nodes[node] = leaf_bounds[begin];
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
    CHECK(next_internal == count - 1U);

    std::vector<std::uint8_t> admitted(count, 1U);
    SpatialTreeView tree{left, right, sources, nodes, metadata, admitted, count};
    std::array<std::uint32_t, 2U * count> traversal{};
    std::array<std::uint32_t, count> output{};
    for (std::uint32_t query_index = 0U; query_index < count; ++query_index) {
        std::vector<std::uint32_t> expected;
        for (std::uint32_t source = 0U; source < count; ++source)
            if (spatial_overlaps(leaf_bounds[query_index], leaf_bounds[source]))
                expected.push_back(source);
        std::uint32_t written{}, visited{};
        CHECK(spatial_query_aabb(tree, leaf_bounds[query_index], 1U,
                                 spatial_invalid_index, traversal, output,
                                 written, visited) == PM_SUCCESS);
        CHECK(written == expected.size());
        CHECK(std::equal(expected.begin(), expected.end(), output.begin()));
        CHECK(visited <= nodes.size());
    }
}

void test_lifecycle() {
    SpatialEpochLifecycle invalid_service(8U, 1U);
    CHECK(invalid_service.stage_boundary(1U, 1U, 1U, 1U, 1U) ==
          PM_ERROR_INVALID_STATE);
    SpatialEpochLifecycle service(8U, 3U);
    SpatialEpochToken first{};
    CHECK(service.stage_boundary(0U, 1U, 10U, 20U, 3U) ==
          PM_ERROR_INVALID_STATE);
    CHECK(service.stage_boundary(1U, 1U, 10U, 20U, 3U) == PM_SUCCESS);
    CHECK(service.record_update(SpatialUpdateKind::build) == PM_SUCCESS);
    CHECK(service.record_update(SpatialUpdateKind::build) ==
          PM_ERROR_INVALID_STATE);
    CHECK(service.publish(first) == PM_SUCCESS);
    CHECK(service.acquire(first) == PM_SUCCESS);
    CHECK(service.validate_epoch(first, 10U, 20U) == PM_SUCCESS);
    CHECK(service.validate_epoch(first, 11U, 20U) == PM_ERROR_INVALID_STATE);
    CHECK(service.record_query(first, 7U, false) == PM_SUCCESS);

    std::array<SpatialAabb, 3> envelopes{
        box(0, 0, 0, 1, 1, 1), box(2, 0, 0, 3, 1, 1),
        box(4, 0, 0, 5, 1, 1)};
    auto current = envelopes;
    CHECK(service.validate_envelopes(first, envelopes, current) == PM_SUCCESS);
    CHECK(service.commit(first) == PM_SUCCESS);
    const auto *first_counts = service.counters(first);
    CHECK(first_counts != nullptr && first_counts->builds == 1U &&
          first_counts->refits == 0U && first_counts->publications == 1U &&
          first_counts->queries == 1U && first_counts->candidates == 7U);

    SpatialEpochToken second{};
    CHECK(service.stage_boundary(2U, 1U, 10U, 20U, 3U, true) == PM_SUCCESS);
    CHECK(service.record_update(SpatialUpdateKind::refit) == PM_SUCCESS);
    CHECK(service.publish(second) == PM_SUCCESS);
    CHECK(service.commit(second) == PM_SUCCESS);
    // A bounded explicit query against committed state reports its own
    // overflow without retroactively invalidating an already committed tick.
    CHECK(service.record_query(second, 0U, true) ==
          PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(service.validate_epoch(second, 10U, 20U) == PM_SUCCESS);
    // Acquired work from the previous epoch remains pinned after retirement.
    CHECK(service.validate_epoch(first, 10U, 20U) == PM_SUCCESS);
    CHECK(service.release(first) == PM_SUCCESS);
    CHECK(service.validate_epoch(first, 10U, 20U) == PM_ERROR_INVALID_STATE);

    // Reused slots get a new generation, so stale tokens cannot alias them.
    SpatialEpochToken third{};
    CHECK(service.stage_boundary(3U, 1U, 12U, 22U, 1U) == PM_SUCCESS);
    CHECK(service.record_update(SpatialUpdateKind::refit) ==
          PM_ERROR_INVALID_STATE);
    CHECK(service.record_update(SpatialUpdateKind::build) == PM_SUCCESS);
    CHECK(service.publish(third) == PM_SUCCESS);
    CHECK(third.generation != first.generation || third.slot != first.slot);
    CHECK(service.commit(third) == PM_SUCCESS);

    // Overflow and envelope escape invalidate the staged tick atomically.
    SpatialEpochToken fourth{};
    CHECK(service.stage_boundary(4U, 1U, 13U, 23U, 1U) == PM_SUCCESS);
    CHECK(service.record_update(SpatialUpdateKind::build) == PM_SUCCESS);
    CHECK(service.publish(fourth) == PM_SUCCESS);
    CHECK(service.record_query(fourth, 0U, true) ==
          PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(service.commit(fourth) == PM_ERROR_INVALID_STATE);
    service.abort();
    CHECK(service.validate_epoch(third, 12U, 22U) == PM_SUCCESS);

    CHECK(service.stage_boundary(4U, 1U, 13U, 23U, 1U) == PM_SUCCESS);
    CHECK(service.record_update(SpatialUpdateKind::build) == PM_SUCCESS);
    CHECK(service.publish(fourth) == PM_SUCCESS);
    std::array one_envelope{box(0, 0, 0, 1, 1, 1)};
    std::array escaped{box(0, 0, 0, 2, 1, 1)};
    CHECK(service.validate_envelopes(fourth, one_envelope, escaped) ==
          PM_ERROR_INVALID_STATE);
    CHECK(service.commit(fourth) == PM_ERROR_INVALID_STATE);
    service.abort();

    // Empty epochs publish without a build/refit, then reset retires history.
    CHECK(service.stage_boundary(4U, 1U, 14U, 24U, 0U) == PM_SUCCESS);
    CHECK(service.record_update(SpatialUpdateKind::empty) == PM_SUCCESS);
    CHECK(service.publish(fourth) == PM_SUCCESS);
    CHECK(service.commit(fourth) == PM_SUCCESS);
    const auto *empty_counts = service.counters(fourth);
    CHECK(empty_counts != nullptr && empty_counts->builds == 0U &&
          empty_counts->refits == 0U && empty_counts->publications == 1U);
    service.reset();
    CHECK(service.validate_epoch(fourth, 14U, 24U) == PM_ERROR_INVALID_STATE);

    // Primitive slot reuse is a distinct full identity despite equal dense slot.
    CHECK(identity(7U, 2U, 3U, 1U) != identity(7U, 2U, 3U, 2U));
    CHECK(identity(7U, 2U, 3U, 1U) != identity(7U, 2U, 4U, 1U));
}
} // namespace

int main() {
    test_plan();
    test_envelopes();
    test_admission();
    test_queries();
    test_candidate_completeness();
    test_lifecycle();
    if (failures == 0)
        std::printf("F8-07 spatial service contract tests passed\n");
    return failures == 0 ? 0 : 1;
}
