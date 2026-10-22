// SPDX-License-Identifier: MIT
#include "simulation/spatial_build.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {
using parallel_mater::simulation::SpatialAabb;
using parallel_mater::simulation::SpatialBuildPlan;
using parallel_mater::simulation::SpatialSortPolicy;
using parallel_mater::simulation::make_spatial_build_plan;
using parallel_mater::simulation::spatial_build_dispatch_budget;
using parallel_mater::simulation::spatial_max_capacity;

int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "%s:%d check failed: %s\n", __FILE__, __LINE__, #condition); \
    ++failures; } } while (false)

SpatialAabb bounds(float x0, float y0, float z0,
                   float x1, float y1, float z1) {
    SpatialAabb result{};
    result.minimum[0] = x0; result.minimum[1] = y0; result.minimum[2] = z0;
    result.maximum[0] = x1; result.maximum[1] = y1; result.maximum[2] = z1;
    return result;
}

SpatialAabb unite(const SpatialAabb &left, const SpatialAabb &right) {
    SpatialAabb result{};
    for (unsigned axis = 0; axis < 3; ++axis) {
        result.minimum[axis] = std::min(left.minimum[axis], right.minimum[axis]);
        result.maximum[axis] = std::max(left.maximum[axis], right.maximum[axis]);
    }
    return result;
}

bool overlaps(const SpatialAabb &left, const SpatialAabb &right) {
    for (unsigned axis = 0; axis < 3; ++axis)
        if (left.maximum[axis] < right.minimum[axis] ||
            right.maximum[axis] < left.minimum[axis]) return false;
    return true;
}

std::uint32_t expand10(std::uint32_t value) {
    value &= 0x000003ffU;
    value = (value | value << 16U) & 0x030000ffU;
    value = (value | value << 8U) & 0x0300f00fU;
    value = (value | value << 4U) & 0x030c30c3U;
    value = (value | value << 2U) & 0x09249249U;
    return value;
}

std::uint32_t morton(const SpatialAabb &value, const SpatialAabb &scene) {
    std::uint32_t coordinates[3]{};
    for (unsigned axis = 0; axis < 3; ++axis) {
        const float center = 0.5F * (value.minimum[axis] + value.maximum[axis]);
        const float extent = scene.maximum[axis] - scene.minimum[axis];
        const float normalized = extent > 0.0F
            ? std::clamp((center - scene.minimum[axis]) / extent, 0.0F, 1.0F)
            : 0.0F;
        coordinates[axis] = static_cast<std::uint32_t>(normalized * 1023.0F);
    }
    return expand10(coordinates[0]) | expand10(coordinates[1]) << 1U |
        expand10(coordinates[2]) << 2U;
}

int delta(const std::vector<std::uint64_t> &keys, int left, int right) {
    if (left < 0 || right < 0 || left >= static_cast<int>(keys.size()) ||
        right >= static_cast<int>(keys.size())) return -1;
    return std::countl_zero(keys[static_cast<std::size_t>(left)] ^
                            keys[static_cast<std::size_t>(right)]);
}

struct OracleTree {
    std::vector<std::uint32_t> left;
    std::vector<std::uint32_t> right;
    std::vector<std::uint32_t> parent;
    std::vector<std::uint32_t> sources;
    std::vector<SpatialAabb> node_bounds;
    std::vector<std::uint64_t> keys;
};

SpatialAabb fill_bounds(OracleTree &tree, std::uint32_t node) {
    if (node >= tree.left.size()) return tree.node_bounds[node];
    tree.node_bounds[node] = unite(fill_bounds(tree, tree.left[node]),
                                   fill_bounds(tree, tree.right[node]));
    return tree.node_bounds[node];
}

OracleTree build_oracle(const std::vector<SpatialAabb> &input,
                        const std::vector<std::uint32_t> &identities,
                        const std::vector<std::uint8_t> &active) {
    std::vector<std::uint32_t> sources;
    for (std::uint32_t index = 0; index < active.size(); ++index)
        if (active[index] != 0U) sources.push_back(index);
    OracleTree tree{};
    if (sources.empty()) return tree;
    SpatialAabb scene = input[sources.front()];
    for (std::uint32_t source : sources) scene = unite(scene, input[source]);
    std::vector<std::pair<std::uint64_t, std::uint32_t>> keyed;
    for (std::uint32_t source : sources) {
        const auto key = (std::uint64_t{morton(input[source], scene)} << 32U) |
            identities[source];
        keyed.emplace_back(key, source);
    }
    std::sort(keyed.begin(), keyed.end());
    const std::uint32_t count = static_cast<std::uint32_t>(keyed.size());
    tree.keys.resize(count);
    tree.sources.resize(count);
    tree.node_bounds.resize(2U * count - 1U);
    tree.parent.assign(2U * count - 1U, std::numeric_limits<std::uint32_t>::max());
    for (std::uint32_t index = 0; index < count; ++index) {
        tree.keys[index] = keyed[index].first;
        tree.sources[index] = keyed[index].second;
        tree.node_bounds[count - 1U + index] = input[keyed[index].second];
    }
    if (count == 1U) return tree;
    tree.left.resize(count - 1U);
    tree.right.resize(count - 1U);
    for (std::uint32_t index = 0; index + 1U < count; ++index) {
        const int node = static_cast<int>(index);
        const int direction = delta(tree.keys, node, node + 1) >=
            delta(tree.keys, node, node - 1) ? 1 : -1;
        const int minimum_delta = delta(tree.keys, node, node - direction);
        int maximum_length = 2;
        while (delta(tree.keys, node, node + maximum_length * direction) >
               minimum_delta) maximum_length <<= 1;
        int length = 0;
        for (int step = maximum_length >> 1; step >= 1; step >>= 1)
            if (delta(tree.keys, node, node + (length + step) * direction) >
                minimum_delta) length += step;
        const int other = node + length * direction;
        const int node_delta = delta(tree.keys, node, other);
        int split_offset = 0;
        int step = length;
        do {
            step = (step + 1) >> 1;
            if (split_offset + step < length &&
                delta(tree.keys, node,
                      node + (split_offset + step) * direction) > node_delta)
                split_offset += step;
        } while (step > 1);
        const int split = node + split_offset * direction +
            std::min(direction, 0);
        const int first = std::min(node, other);
        const int last = std::max(node, other);
        const std::uint32_t left = split == first
            ? count - 1U + static_cast<std::uint32_t>(split)
            : static_cast<std::uint32_t>(split);
        const std::uint32_t right = split + 1 == last
            ? count + static_cast<std::uint32_t>(split)
            : static_cast<std::uint32_t>(split + 1);
        tree.left[index] = left;
        tree.right[index] = right;
        tree.parent[left] = index;
        tree.parent[right] = index;
    }
    fill_bounds(tree, 0U);
    return tree;
}

void collect_overlaps(const OracleTree &tree, std::uint32_t node,
                      const SpatialAabb &query,
                      std::vector<std::uint32_t> &result) {
    if (!overlaps(tree.node_bounds[node], query)) return;
    if (node >= tree.left.size()) {
        result.push_back(tree.sources[node - tree.left.size()]);
        return;
    }
    collect_overlaps(tree, tree.left[node], query, result);
    collect_overlaps(tree, tree.right[node], query, result);
}

void test_plan() {
    SpatialBuildPlan empty{};
    CHECK(make_spatial_build_plan(0U, empty) == PM_SUCCESS);
    CHECK(empty.scratch_bytes == 0U && empty.output_bytes == 0U);
    CHECK(spatial_build_dispatch_budget(65536U, 0U).total_dispatches == 0U);

    SpatialBuildPlan plan{};
    CHECK(make_spatial_build_plan(30065U, plan) == PM_SUCCESS);
    CHECK(plan.block_capacity == 118U);
    CHECK(plan.node_capacity == 60129U);
    CHECK(plan.scratch_bytes > 0U && plan.output_bytes > plan.scratch_bytes);
    const std::array scratch_ranges{
        plan.compaction_counts, plan.compaction_offsets, plan.active_indices,
        plan.active_count, plan.keys[0], plan.keys[1],
        plan.values[0], plan.values[1],
        plan.radix_histograms, plan.radix_offsets, plan.block_bounds,
        plan.scene_bounds, plan.arrivals, plan.status};
    for (std::size_t index = 0; index < scratch_ranges.size(); ++index) {
        CHECK(scratch_ranges[index].offset % 16U == 0U);
        CHECK(scratch_ranges[index].offset + scratch_ranges[index].size <=
              plan.scratch_bytes);
        for (std::size_t other = index + 1; other < scratch_ranges.size(); ++other) {
            if (scratch_ranges[index].size == 0U || scratch_ranges[other].size == 0U)
                continue;
            CHECK(scratch_ranges[index].offset + scratch_ranges[index].size <=
                  scratch_ranges[other].offset ||
                  scratch_ranges[other].offset + scratch_ranges[other].size <=
                  scratch_ranges[index].offset);
        }
    }
    const std::array output_ranges{
        plan.left_children, plan.right_children, plan.parents,
        plan.leaf_sources, plan.node_bounds};
    for (std::size_t index = 0; index < output_ranges.size(); ++index) {
        CHECK(output_ranges[index].offset % 16U == 0U);
        CHECK(output_ranges[index].offset + output_ranges[index].size <=
              plan.output_bytes);
        for (std::size_t other = index + 1; other < output_ranges.size(); ++other)
            CHECK(output_ranges[index].offset + output_ranges[index].size <=
                  output_ranges[other].offset ||
                  output_ranges[other].offset + output_ranges[other].size <=
                  output_ranges[index].offset);
    }
    const auto dense = spatial_build_dispatch_budget(30065U, 30065U);
    CHECK(dense.sort_policy == SpatialSortPolicy::radix8);
    CHECK(dense.sort_dispatches == 24U);
    CHECK(dense.total_dispatches == 32U);
    const auto small = spatial_build_dispatch_budget(1024U, 1024U);
    CHECK(small.sort_policy == SpatialSortPolicy::workgroup_bitonic);
    CHECK(small.sort_dispatches == 1U);
    const auto boundary = spatial_build_dispatch_budget(1025U, 1025U);
    CHECK(boundary.sort_policy == SpatialSortPolicy::radix8);
    const auto sparse = spatial_build_dispatch_budget(1U << 20U, 128U);
    CHECK(sparse.active_blocks == 1U && sparse.capacity_blocks == 4096U);
    CHECK(sparse.sort_policy == SpatialSortPolicy::workgroup_bitonic);
    const auto singleton = spatial_build_dispatch_budget(1U, 2U);
    CHECK(singleton.active_count == 1U && singleton.total_dispatches == 7U);
    CHECK(make_spatial_build_plan(spatial_max_capacity + 1U, plan)
          == PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(make_spatial_build_plan(std::numeric_limits<std::uint32_t>::max(), plan)
          == PM_ERROR_CAPACITY_EXCEEDED);
}

void test_order_topology_and_candidates() {
    std::mt19937 random(0xF806U);
    std::uniform_real_distribution<float> coordinate(-10.0F, 10.0F);
    constexpr std::uint32_t capacity = 257U;
    std::vector<SpatialAabb> input(capacity);
    std::vector<std::uint32_t> identities(capacity);
    std::vector<std::uint8_t> active(capacity);
    for (std::uint32_t index = 0; index < capacity; ++index) {
        const float x = coordinate(random);
        const float y = coordinate(random);
        const float z = coordinate(random);
        input[index] = bounds(x, y, z, x + 0.2F, y + 0.2F, z + 0.2F);
        identities[index] = capacity - index;
        active[index] = index % 3U != 0U;
    }
    // Coincident leaves exercise duplicate Morton codes while distinct identities
    // keep the full composite ordering deterministic.
    input[7] = input[8] = input[11] = bounds(1, 2, 3, 1, 2, 3);
    active[7] = active[8] = active[11] = 1U;
    OracleTree tree = build_oracle(input, identities, active);
    CHECK(!tree.keys.empty());
    CHECK(std::is_sorted(tree.keys.begin(), tree.keys.end()));
    CHECK(tree.left.size() + 1U == tree.sources.size());
    CHECK(tree.parent[0] == std::numeric_limits<std::uint32_t>::max());
    for (std::uint32_t source : tree.sources) CHECK(active[source] != 0U);
    for (std::uint32_t node = 0; node < tree.left.size(); ++node) {
        CHECK(tree.parent[tree.left[node]] == node);
        CHECK(tree.parent[tree.right[node]] == node);
    }

    for (unsigned sample = 0; sample < 128; ++sample) {
        const float x = coordinate(random);
        const float y = coordinate(random);
        const float z = coordinate(random);
        const SpatialAabb query = bounds(x, y, z, x + 2, y + 2, z + 2);
        std::vector<std::uint32_t> actual;
        collect_overlaps(tree, 0U, query, actual);
        std::sort(actual.begin(), actual.end());
        std::vector<std::uint32_t> expected;
        for (std::uint32_t index = 0; index < capacity; ++index)
            if (active[index] != 0U && overlaps(input[index], query))
                expected.push_back(index);
        CHECK(actual == expected);
    }

    // Inclusive overlap retains touching candidates.
    std::vector<SpatialAabb> touching{
        bounds(0, 0, 0, 1, 1, 1), bounds(1, 0, 0, 2, 1, 1)};
    const std::vector<std::uint32_t> ids{9U, 3U};
    const std::vector<std::uint8_t> flags{1U, 1U};
    tree = build_oracle(touching, ids, flags);
    std::vector<std::uint32_t> hits;
    collect_overlaps(tree, 0U, bounds(1, 0, 0, 1, 1, 1), hits);
    std::sort(hits.begin(), hits.end());
    CHECK((hits == std::vector<std::uint32_t>{0U, 1U}));

    const std::vector<std::uint8_t> none{0U, 0U};
    CHECK(build_oracle(touching, ids, none).keys.empty());

    // A zero-extent scene is valid. Morton values coincide, while the complete
    // identity suffix preserves deterministic ordering.
    const std::vector<SpatialAabb> coincident(3U,
        bounds(2, 2, 2, 2, 2, 2));
    const std::vector<std::uint32_t> coincident_ids{9U, 3U, 7U};
    const std::vector<std::uint8_t> coincident_flags{1U, 1U, 1U};
    tree = build_oracle(coincident, coincident_ids, coincident_flags);
    CHECK(tree.keys.size() == 3U);
    CHECK((tree.keys[0] >> 32U) == (tree.keys[1] >> 32U));
    CHECK((tree.keys[1] >> 32U) == (tree.keys[2] >> 32U));
    CHECK(std::uint32_t(tree.keys[0]) == 3U);
    CHECK(std::uint32_t(tree.keys[1]) == 7U);
    CHECK(std::uint32_t(tree.keys[2]) == 9U);

    // Match the desktop-full F8 leaf tier and the physical CUDA fixture. This
    // catches range/split errors that only appear after a 15-bit rank boundary.
    constexpr std::uint32_t large_count = 30065U;
    input.resize(large_count);
    identities.resize(large_count);
    active.assign(large_count, 1U);
    for (std::uint32_t index = 0; index < large_count; ++index) {
        const float x = float(index % 17U) * 0.25F;
        const float y = float((index / 17U) % 19U) * 0.2F;
        const float z = float((index / (17U * 19U)) % 23U) * 0.15F;
        input[index] = bounds(x, y, z, x + 0.1F, y + 0.1F, z + 0.1F);
        identities[index] = 0x9e3779b9U * (index + 1U);
    }
    tree = build_oracle(input, identities, active);
    CHECK(tree.left.size() + 1U == large_count);
    for (std::uint32_t node = 0; node < tree.left.size(); ++node) {
        CHECK(tree.parent[tree.left[node]] == node);
        CHECK(tree.parent[tree.right[node]] == node);
    }
}
} // namespace

int main() {
    test_plan();
    test_order_topology_and_candidates();
    if (failures == 0)
        std::printf("F8-06 spatial plan/oracle tests passed\n");
    return failures == 0 ? 0 : 1;
}
