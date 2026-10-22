// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

#include "simulation/spatial_build.hpp"
#include "simulation/spatial_service.hpp"
#include "parallel_mater_primitives_runtime.cu"

namespace {
using parallel_mater::simulation::SpatialSortPolicy;
using parallel_mater::simulation::SpatialEpochHeader;
using parallel_mater::simulation::SpatialLeafMetadata;
using parallel_mater::simulation::make_spatial_build_plan;
using parallel_mater::simulation::spatial_build_dispatch_budget;
using parallel_mater::simulation::spatial_epoch_status_duplicate_publication;
using parallel_mater::simulation::spatial_epoch_status_envelope_escape;
using parallel_mater::simulation::spatial_epoch_status_query_overflow;
using parallel_mater::simulation::spatial_epoch_status_stale_epoch;
using parallel_mater::simulation::spatial_leaf_provisional;
using parallel_mater::simulation::spatial_leaf_retained;
using parallel_mater::simulation::spatial_leaf_inactive;
using parallel_mater::simulation::spatial_leaf_transfer_destination;
using parallel_mater::simulation::spatial_leaf_transfer_source;
using parallel_mater::simulation::spatial_status_count_overflow;
using parallel_mater::simulation::spatial_status_duplicate_key;
using parallel_mater::simulation::spatial_status_invalid_bounds;
using parallel_mater::simulation::spatial_status_policy_mismatch;

constexpr unsigned tile = 256U;
static_assert(sizeof(SpatialLeafMetadata) == sizeof(PmSpatialLeafMetadata_0));
static_assert(sizeof(SpatialEpochHeader) == sizeof(PmSpatialEpochHeader_0));
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); } } while (false)

template <typename T>
T *managed(std::size_t count) {
    T *result{};
    if (cudaMallocManaged(&result, std::max<std::size_t>(count, 1U) * sizeof(T))
        != cudaSuccess) return nullptr;
    return result;
}

bool close(float left, float right) {
    return std::fabs(left - right) <= 2.0e-5F *
        std::max({1.0F, std::fabs(left), std::fabs(right)});
}

int host_delta(const unsigned long long *keys, unsigned count,
               int left, int right) {
    if (left < 0 || right < 0 || left >= static_cast<int>(count) ||
        right >= static_cast<int>(count)) return -1;
    return std::countl_zero(keys[left] ^ keys[right]);
}

struct Fixture {
    unsigned capacity{};
    unsigned live{};
    unsigned capacity_blocks{};
    unsigned active_blocks{};
    unsigned nodes{};
    unsigned internal{};
    std::uint8_t *flags{};
    unsigned *identities{};
    PmSpatialAabb_0 *input_bounds{};
    unsigned *counts{};
    unsigned *offsets{};
    unsigned *active_indices{};
    unsigned *active_count{};
    PmSpatialAabb_0 *block_bounds{};
    PmSpatialAabb_0 *scene_bounds{};
    unsigned long long *keys[2]{};
    unsigned *values[2]{};
    unsigned *histograms{};
    unsigned *radix_offsets{};
    unsigned *left{};
    unsigned *right{};
    unsigned *parents{};
    unsigned *leaf_sources{};
    PmSpatialAabb_0 *node_bounds{};
    unsigned *arrivals{};
    unsigned *status{};
    std::vector<unsigned> expected_sources;

    Fixture(unsigned capacity_value, unsigned live_value)
        : capacity(capacity_value), live(live_value),
          capacity_blocks((capacity + tile - 1U) / tile),
          active_blocks((live + tile - 1U) / tile),
          nodes(live == 0U ? 0U : 2U * live - 1U),
          internal(live == 0U ? 0U : live - 1U) {
        flags = managed<std::uint8_t>(capacity);
        identities = managed<unsigned>(capacity);
        input_bounds = managed<PmSpatialAabb_0>(capacity);
        counts = managed<unsigned>(capacity_blocks);
        offsets = managed<unsigned>(capacity_blocks);
        active_indices = managed<unsigned>(capacity);
        active_count = managed<unsigned>(1);
        block_bounds = managed<PmSpatialAabb_0>(std::max(active_blocks, 1U));
        scene_bounds = managed<PmSpatialAabb_0>(1);
        keys[0] = managed<unsigned long long>(capacity);
        keys[1] = managed<unsigned long long>(capacity);
        values[0] = managed<unsigned>(capacity);
        values[1] = managed<unsigned>(capacity);
        const std::size_t radix_slots = 256ULL * std::max(active_blocks, 1U);
        histograms = managed<unsigned>(radix_slots);
        radix_offsets = managed<unsigned>(radix_slots);
        left = managed<unsigned>(internal);
        right = managed<unsigned>(internal);
        parents = managed<unsigned>(std::max(nodes, 1U));
        leaf_sources = managed<unsigned>(capacity);
        node_bounds = managed<PmSpatialAabb_0>(std::max(nodes, 1U));
        arrivals = managed<unsigned>(internal);
        status = managed<unsigned>(1);
        initialize();
    }

    ~Fixture() {
        for (void *pointer : std::array<void *, 22>{
                 flags, identities, input_bounds, counts, offsets,
                 active_indices, active_count, block_bounds, scene_bounds,
                 keys[0], keys[1], values[0], values[1], histograms,
                 radix_offsets, left, right, parents, leaf_sources,
                 node_bounds, arrivals, status})
            cudaFree(pointer);
    }

    void initialize() {
        std::fill(flags, flags + capacity, std::uint8_t{});
        expected_sources.clear();
        for (unsigned selected = 0; selected < live; ++selected) {
            const unsigned source = static_cast<unsigned>(
                (std::uint64_t{selected} * capacity) / live);
            flags[source] = 1U;
            expected_sources.push_back(source);
        }
        for (unsigned index = 0; index < capacity; ++index) {
            identities[index] = 0x9e3779b9U * (index + 1U);
            const float x = float(index % 17U) * 0.25F;
            const float y = float((index / 17U) % 19U) * 0.2F;
            const float z = float((index / (17U * 19U)) % 23U) * 0.15F;
            input_bounds[index].minimum_0 = make_float4(x, y, z, 0.0F);
            input_bounds[index].maximum_0 =
                make_float4(x + 0.1F, y + 0.1F, z + 0.1F, 0.0F);
        }
        *active_count = 0U;
        *status = 0U;
        cudaDeviceSynchronize();
    }

    void launch_compaction() {
        pm_compact_count_blocks<<<capacity_blocks, tile>>>(
            flags, capacity, tile, counts);
        pm_exclusive_scan_u32<<<1U, tile>>>(
            counts, capacity_blocks, offsets, active_count);
        pm_compact_scatter_indices<<<capacity_blocks, tile>>>(
            flags, capacity, tile, offsets, active_indices);
    }

    void launch_sort() {
        if (live <= 1U) return;
        if (live <= 1024U) {
            pm_spatial_sort_small_u64_u32<<<1U, tile>>>(
                keys[0], values[0], active_count, capacity, status);
            return;
        }
        auto *source_keys = keys[0];
        auto *target_keys = keys[1];
        auto *source_values = values[0];
        auto *target_values = values[1];
        for (unsigned shift = 0U; shift < 64U; shift += 8U) {
            pm_spatial_radix_histogram_u64<<<active_blocks, tile>>>(
                source_keys, active_count, capacity, shift, active_blocks,
                histograms, status);
            pm_spatial_radix_prefix_u32<<<1U, tile>>>(
                histograms, active_blocks, radix_offsets);
            pm_spatial_radix_scatter_u64_u32<<<active_blocks, tile>>>(
                source_keys, source_values, active_count, capacity, shift,
                active_blocks, radix_offsets, target_keys, target_values, status);
            std::swap(source_keys, target_keys);
            std::swap(source_values, target_values);
        }
        CHECK(source_keys == keys[0]);
        CHECK(source_values == values[0]);
    }

    void launch_build(bool include_union = true) {
        cudaMemsetAsync(status, 0, sizeof(*status));
        if (internal != 0U)
            cudaMemsetAsync(arrivals, 0, internal * sizeof(*arrivals));
        launch_compaction();
        if (live == 0U) return;
        pm_spatial_reduce_bounds_blocks<<<active_blocks, tile>>>(
            active_indices, active_count, capacity, input_bounds,
            block_bounds, status);
        pm_spatial_reduce_bounds_final<<<1U, tile>>>(
            block_bounds, active_blocks, active_count, capacity,
            scene_bounds, status);
        pm_spatial_emit_keys<<<active_blocks, tile>>>(
            active_indices, active_count, capacity, identities, input_bounds,
            scene_bounds, keys[0], values[0], status);
        launch_sort();
        pm_spatial_build_topology<<<active_blocks, tile>>>(
            keys[0], values[0], active_count, capacity, input_bounds,
            left, right, parents, leaf_sources, node_bounds, status);
        if (include_union && live > 1U)
            pm_spatial_union_bounds<<<active_blocks, tile>>>(
                active_count, capacity, left, right, parents, node_bounds,
                arrivals, status);
    }

    void launch_union() {
        if (live > 1U)
            pm_spatial_union_bounds<<<active_blocks, tile>>>(
                active_count, capacity, left, right, parents, node_bounds,
                arrivals, status);
    }

    bool allocated() const {
        return flags && identities && input_bounds && counts && offsets &&
            active_indices && active_count && block_bounds && scene_bounds &&
            keys[0] && keys[1] && values[0] && values[1] && histograms &&
            radix_offsets && left && right && parents && leaf_sources &&
            node_bounds && arrivals && status;
    }
};

void validate_fixture(Fixture &fixture) {
    std::printf("F8_CASE capacity=%u live=%u\n", fixture.capacity, fixture.live);
    std::fflush(stdout);
    CHECK(fixture.allocated());
    fixture.launch_build(false);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.active_count == fixture.live);
    CHECK(*fixture.status == 0U);
    for (unsigned index = 0; index < fixture.live; ++index) {
        CHECK(fixture.active_indices[index] == fixture.expected_sources[index]);
        if (index != 0U) CHECK(fixture.keys[0][index - 1U] < fixture.keys[0][index]);
        const unsigned source = fixture.values[0][index];
        CHECK(source < fixture.capacity);
        if (source >= fixture.capacity) continue;
        CHECK(std::uint32_t(fixture.keys[0][index]) == fixture.identities[source]);
        CHECK(fixture.leaf_sources[index] == source);
    }
    if (fixture.live == 0U) return;
    CHECK(fixture.parents[0] == std::numeric_limits<unsigned>::max());
    const int topology_failures_before = failures;
    unsigned topology_reports = 0U;
    for (unsigned node = 0; node < fixture.internal; ++node) {
        const int index = static_cast<int>(node);
        const int direction = host_delta(fixture.keys[0], fixture.live,
            index, index + 1) >= host_delta(fixture.keys[0], fixture.live,
            index, index - 1) ? 1 : -1;
        const int minimum_delta = host_delta(fixture.keys[0], fixture.live,
            index, index - direction);
        int maximum_length = 2;
        while (host_delta(fixture.keys[0], fixture.live, index,
                          index + maximum_length * direction) > minimum_delta)
            maximum_length <<= 1;
        int length = 0;
        for (int step = maximum_length >> 1; step >= 1; step >>= 1)
            if (host_delta(fixture.keys[0], fixture.live, index,
                           index + (length + step) * direction) > minimum_delta)
                length += step;
        const int other = index + length * direction;
        const int node_delta = host_delta(
            fixture.keys[0], fixture.live, index, other);
        int split_offset = 0;
        int step = length;
        do {
            step = (step + 1) >> 1;
            if (split_offset + step < length &&
                host_delta(fixture.keys[0], fixture.live, index,
                           index + (split_offset + step) * direction) > node_delta)
                split_offset += step;
        } while (step > 1);
        const int split = index + split_offset * direction +
            std::min(direction, 0);
        const int first = std::min(index, other);
        const int last = std::max(index, other);
        const unsigned expected_left = split == first
            ? fixture.live - 1U + static_cast<unsigned>(split)
            : static_cast<unsigned>(split);
        const unsigned expected_right = split + 1 == last
            ? fixture.live + static_cast<unsigned>(split)
            : static_cast<unsigned>(split + 1);
        if ((fixture.left[node] != expected_left ||
             fixture.right[node] != expected_right) && topology_reports++ < 8U)
            std::fprintf(stderr,
                "topology node=%u actual=(%u,%u) expected=(%u,%u) "
                "keys=(%016llx,%016llx)\n", node, fixture.left[node],
                fixture.right[node], expected_left, expected_right,
                fixture.keys[0][node], fixture.keys[0][node + 1U]);
        CHECK(fixture.left[node] < fixture.nodes);
        CHECK(fixture.right[node] < fixture.nodes);
        if (fixture.left[node] >= fixture.nodes ||
            fixture.right[node] >= fixture.nodes) continue;
        CHECK(fixture.left[node] == expected_left);
        CHECK(fixture.right[node] == expected_right);
        CHECK(fixture.parents[fixture.left[node]] == node);
        CHECK(fixture.parents[fixture.right[node]] == node);
    }
    if (failures != topology_failures_before) return;
    fixture.launch_union();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    for (unsigned node = 0; node < fixture.internal; ++node) {
        CHECK(fixture.arrivals[node] == 2U);
        const auto &node_box = fixture.node_bounds[node];
        const auto &left_box = fixture.node_bounds[fixture.left[node]];
        const auto &right_box = fixture.node_bounds[fixture.right[node]];
        for (unsigned axis = 0; axis < 3; ++axis) {
            const float expected_min = std::min(
                (&left_box.minimum_0.x)[axis], (&right_box.minimum_0.x)[axis]);
            const float expected_max = std::max(
                (&left_box.maximum_0.x)[axis], (&right_box.maximum_0.x)[axis]);
            CHECK(close((&node_box.minimum_0.x)[axis], expected_min));
            CHECK(close((&node_box.maximum_0.x)[axis], expected_max));
        }
    }
    PmSpatialAabb_0 expected = fixture.input_bounds[fixture.expected_sources[0]];
    for (unsigned source : fixture.expected_sources)
        for (unsigned axis = 0; axis < 3; ++axis) {
            (&expected.minimum_0.x)[axis] = std::min(
                (&expected.minimum_0.x)[axis],
                (&fixture.input_bounds[source].minimum_0.x)[axis]);
            (&expected.maximum_0.x)[axis] = std::max(
                (&expected.maximum_0.x)[axis],
                (&fixture.input_bounds[source].maximum_0.x)[axis]);
        }
    for (unsigned axis = 0; axis < 3; ++axis) {
        CHECK(close((&fixture.node_bounds[0].minimum_0.x)[axis],
                    (&expected.minimum_0.x)[axis]));
        CHECK(close((&fixture.node_bounds[0].maximum_0.x)[axis],
                    (&expected.maximum_0.x)[axis]));
    }
}

void validate_errors() {
    Fixture fixture(64U, 8U);
    const unsigned first = fixture.expected_sources[0];
    const unsigned second = fixture.expected_sources[1];
    fixture.identities[second] = fixture.identities[first];
    fixture.input_bounds[second] = fixture.input_bounds[first];
    fixture.launch_build();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_status_duplicate_key) != 0U);

    fixture.initialize();
    fixture.input_bounds[first].minimum_0.x =
        std::numeric_limits<float>::quiet_NaN();
    fixture.launch_build();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_status_invalid_bounds) != 0U);

    fixture.initialize();
    std::iota(fixture.active_indices,
              fixture.active_indices + fixture.capacity, 0U);
    fixture.scene_bounds[0].minimum_0 = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    fixture.scene_bounds[0].maximum_0 = make_float4(5.0F, 5.0F, 5.0F, 0.0F);
    *fixture.active_count = fixture.capacity + 1U;
    *fixture.status = 0U;
    pm_spatial_emit_keys<<<1U, tile>>>(
        fixture.active_indices, fixture.active_count, fixture.capacity,
        fixture.identities, fixture.input_bounds, fixture.scene_bounds,
        fixture.keys[0], fixture.values[0], fixture.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_status_count_overflow) != 0U);

    Fixture wrong_policy(1025U, 0U);
    *wrong_policy.active_count = 1025U;
    *wrong_policy.status = 0U;
    pm_spatial_sort_small_u64_u32<<<1U, tile>>>(
        wrong_policy.keys[0], wrong_policy.values[0],
        wrong_policy.active_count, wrong_policy.capacity,
        wrong_policy.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*wrong_policy.status & spatial_status_policy_mismatch) != 0U);
}

void validate_service_kernels() {
    Fixture fixture(7U, 4U);
    auto *metadata = managed<SpatialLeafMetadata>(fixture.capacity);
    auto *admission = managed<std::uint8_t>(fixture.capacity);
    auto *header = managed<SpatialEpochHeader>(1U);
    auto *query_counter = managed<unsigned>(1U);
    auto *query_status = managed<unsigned>(1U);
    auto *queries = managed<PmSpatialAabb_0>(1U);
    auto *query_masks = managed<unsigned>(1U);
    auto *excluded = managed<unsigned>(1U);
    auto *candidate_counts = managed<unsigned>(1U);
    auto *candidates = managed<unsigned>(fixture.capacity);
    auto *visited = managed<unsigned>(1U);
    CHECK(fixture.allocated() && metadata && admission && header &&
          query_counter && query_status && queries &&
          query_masks && excluded && candidate_counts && candidates && visited);
    if (!fixture.allocated() || !metadata || !admission || !header ||
        !query_counter || !query_status || !queries || !query_masks ||
        !excluded || !candidate_counts || !candidates || !visited) return;

    const std::array<std::array<float, 2>, 4> x{{
        {{0.0F, 1.0F}}, {{0.5F, 1.5F}},
        {{3.0F, 4.0F}}, {{3.5F, 4.5F}}}};
    unsigned active_ordinal = 0U;
    for (unsigned index = 0U; index < fixture.capacity; ++index) {
        metadata[index] = {};
        if (active_ordinal == fixture.expected_sources.size() ||
            fixture.expected_sources[active_ordinal] != index) {
            metadata[index].flags = spatial_leaf_inactive;
            continue;
        }
        metadata[index].identity = {1U, 0U, 1U, 1U, 1U, index, 1U};
        metadata[index].sort_key = 10U * (active_ordinal + 1U);
        metadata[index].payload_ref = index;
        metadata[index].query_mask = 1U;
        metadata[index].admission_mask = 1U;
        metadata[index].flags = active_ordinal == 0U
            ? spatial_leaf_retained | spatial_leaf_transfer_source
            : spatial_leaf_provisional;
        if (active_ordinal == 1U)
            metadata[index].flags |= spatial_leaf_transfer_destination;
        if (active_ordinal <= 1U) metadata[index].transfer_group = 7U;
        fixture.identities[index] = metadata[index].sort_key;
        fixture.input_bounds[index].minimum_0 =
            make_float4(x[active_ordinal][0], 0.0F, 0.0F, 0.0F);
        fixture.input_bounds[index].maximum_0 =
            make_float4(x[active_ordinal][1], 1.0F, 1.0F, 0.0F);
        ++active_ordinal;
    }
    *fixture.status = 0U;
    pm_spatial_initialize_epoch<<<1U, 1U>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(header),
        1ULL, 7ULL, 9ULL, 1U, 1U, fixture.capacity, 1U,
        query_counter, query_status, fixture.status);
    pm_spatial_stage_leaves<<<1U, tile>>>(
        reinterpret_cast<PmSpatialLeafMetadata_0 *>(metadata),
        fixture.input_bounds, fixture.capacity, fixture.capacity,
        fixture.flags, admission, fixture.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    for (unsigned index = 0U; index < fixture.capacity; ++index)
        CHECK(fixture.flags[index] ==
              (std::find(fixture.expected_sources.begin(),
                         fixture.expected_sources.end(), index) !=
                       fixture.expected_sources.end() ? 1U : 0U));

    fixture.launch_build();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    pm_spatial_resolve_admission<<<1U, tile>>>(
        reinterpret_cast<PmSpatialLeafMetadata_0 *>(metadata),
        fixture.input_bounds, fixture.active_count, fixture.capacity,
        fixture.active_indices, fixture.left, fixture.right,
        fixture.leaf_sources,
        fixture.node_bounds, admission, fixture.status);
    pm_spatial_finalize_transfers<<<1U, tile>>>(
        reinterpret_cast<PmSpatialLeafMetadata_0 *>(metadata),
        fixture.active_count, fixture.capacity, fixture.active_indices,
        admission, fixture.status);
    pm_spatial_publish_epoch<<<1U, 1U>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(header),
        fixture.active_count, fixture.capacity, fixture.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK((std::array<std::uint8_t, 7>{
               admission[0], admission[1], admission[2], admission[3],
               admission[4], admission[5], admission[6]} ==
           std::array<std::uint8_t, 7>{0U, 1U, 0U, 1U, 0U, 0U, 0U}));
    CHECK(header->tick == 1U && header->graph_revision == 7U &&
          header->resource_revision == 9U && header->leaf_count == 4U &&
          header->node_count == 7U && header->build_count == 1U &&
          header->refit_count == 0U && header->publication_count == 1U);

    queries[0].minimum_0 = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    queries[0].maximum_0 = make_float4(5.0F, 1.0F, 1.0F, 0.0F);
    query_masks[0] = 1U;
    excluded[0] = std::numeric_limits<unsigned>::max();
    pm_spatial_query_aabbs<<<1U, 64U>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(header), query_counter,
        queries,
        query_masks, excluded, 1U, fixture.capacity,
        fixture.capacity, fixture.left, fixture.right,
        fixture.leaf_sources, fixture.node_bounds,
        reinterpret_cast<PmSpatialLeafMetadata_0 *>(metadata), admission,
        candidate_counts, candidates, visited, query_status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*query_status == 0U);
    CHECK(*candidate_counts == 2U);
    CHECK(metadata[candidates[0]].identity.primitive_slot == 1U);
    CHECK(metadata[candidates[1]].identity.primitive_slot == 3U);
    CHECK(*visited == 7U);

    *query_status = 0U;
    pm_spatial_query_aabbs<<<1U, 64U>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(header), query_counter,
        queries,
        query_masks, excluded, 1U, 1U,
        fixture.capacity, fixture.left, fixture.right,
        fixture.leaf_sources, fixture.node_bounds,
        reinterpret_cast<PmSpatialLeafMetadata_0 *>(metadata), admission,
        candidate_counts, candidates, visited, query_status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*query_status & spatial_epoch_status_query_overflow) != 0U);
    CHECK(*candidate_counts == 0U);
    CHECK(*query_counter == 2U);

    *query_status = 0U;
    auto *current_metadata = managed<SpatialLeafMetadata>(fixture.capacity);
    auto *current_bounds = managed<PmSpatialAabb_0>(fixture.capacity);
    CHECK(current_metadata && current_bounds);
    if (current_metadata && current_bounds) {
        std::copy(metadata, metadata + fixture.capacity, current_metadata);
        std::copy(fixture.input_bounds,
                  fixture.input_bounds + fixture.capacity, current_bounds);
        pm_spatial_validate_envelopes<<<1U, tile>>>(
            reinterpret_cast<PmSpatialLeafMetadata_0 *>(metadata),
            fixture.input_bounds, fixture.capacity,
            reinterpret_cast<PmSpatialLeafMetadata_0 *>(current_metadata),
            current_bounds, fixture.capacity, query_status);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        CHECK(*query_status == 0U);
        current_metadata[3].identity.primitive_generation = 2U;
        current_bounds[3].maximum_0.x = 5.0F;
        pm_spatial_validate_envelopes<<<1U, tile>>>(
            reinterpret_cast<PmSpatialLeafMetadata_0 *>(metadata),
            fixture.input_bounds, fixture.capacity,
            reinterpret_cast<PmSpatialLeafMetadata_0 *>(current_metadata),
            current_bounds, fixture.capacity, query_status);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        CHECK((*query_status & spatial_epoch_status_stale_epoch) != 0U);
        CHECK((*query_status & spatial_epoch_status_envelope_escape) != 0U);
    }

    *fixture.status = 0U;
    pm_spatial_publish_epoch<<<1U, 1U>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(header),
        fixture.active_count, fixture.capacity, fixture.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_epoch_status_duplicate_publication) != 0U);

    // Unchanged topology/identity can select a two-dispatch leaf refit + union
    // instead of another sort/build. Counts remain GPU resident in the header.
    *fixture.status = 0U;
    fixture.input_bounds[5].maximum_0.x = 6.0F;
    pm_spatial_initialize_epoch<<<1U, 1U>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(header),
        2ULL, 7ULL, 10ULL, 1U, 2U, fixture.live, 2U,
        query_counter, query_status, fixture.status);
    pm_spatial_refit_leaves<<<1U, tile>>>(
        fixture.active_count, fixture.capacity, fixture.leaf_sources,
        fixture.input_bounds, fixture.node_bounds, fixture.arrivals,
        fixture.status);
    pm_spatial_union_bounds<<<1U, tile>>>(
        fixture.active_count, fixture.capacity, fixture.left, fixture.right,
        fixture.parents, fixture.node_bounds, fixture.arrivals, fixture.status);
    pm_spatial_publish_epoch<<<1U, 1U>>>(
        reinterpret_cast<PmSpatialEpochHeader_0 *>(header),
        fixture.active_count, fixture.capacity, fixture.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(close(fixture.node_bounds[0].maximum_0.x, 6.0F));
    CHECK(header->build_count == 0U && header->refit_count == 1U &&
          header->publication_count == 1U);
    CHECK(*query_counter == 0U);

    cudaFree(current_bounds);
    cudaFree(current_metadata);
    cudaFree(visited);
    cudaFree(candidates);
    cudaFree(candidate_counts);
    cudaFree(excluded);
    cudaFree(query_masks);
    cudaFree(queries);
    cudaFree(query_status);
    cudaFree(query_counter);
    cudaFree(header);
    cudaFree(admission);
    cudaFree(metadata);
}

float percentile(std::vector<float> values, float fraction) {
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * float(values.size())) - 1.0F);
    return values[std::min(index, values.size() - 1U)];
}

template <typename Setup, typename Launch>
std::pair<float, float> measure(Setup setup, Launch launch,
                                unsigned warmup = 5U,
                                unsigned samples = 40U) {
    cudaEvent_t begin{}, end{};
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (unsigned index = 0; index < warmup; ++index) {
        setup();
        launch();
    }
    cudaDeviceSynchronize();
    std::vector<float> times;
    times.reserve(samples);
    for (unsigned index = 0; index < samples; ++index) {
        setup();
        cudaEventRecord(begin);
        launch();
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float milliseconds{};
        cudaEventElapsedTime(&milliseconds, begin, end);
        times.push_back(milliseconds);
    }
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    return {percentile(times, 0.50F), percentile(times, 0.95F)};
}

void launch_legacy_sort(Fixture &fixture) {
    auto *source_keys = fixture.keys[0];
    auto *target_keys = fixture.keys[1];
    auto *source_values = fixture.values[0];
    auto *target_values = fixture.values[1];
    for (unsigned shift = 0; shift < 64U; shift += 4U) {
        pm_radix_histogram_u64<<<fixture.active_blocks, tile>>>(
            source_keys, fixture.live, tile, shift, fixture.active_blocks,
            fixture.histograms);
        pm_radix_prefix_u32<<<1U, 1U>>>(
            fixture.histograms, fixture.active_blocks, fixture.radix_offsets);
        pm_radix_scatter_u64_u32<<<fixture.active_blocks, tile>>>(
            source_keys, source_values, fixture.live, tile, shift,
            fixture.active_blocks, fixture.radix_offsets,
            target_keys, target_values);
        std::swap(source_keys, target_keys);
        std::swap(source_values, target_values);
    }
}

void benchmark(unsigned capacity, unsigned live) {
    Fixture fixture(capacity, live);
    CHECK(fixture.allocated());
    fixture.launch_build();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    auto *seed_keys = managed<unsigned long long>(live);
    auto *seed_values = managed<unsigned>(live);
    std::copy(fixture.keys[0], fixture.keys[0] + live, seed_keys);
    std::copy(fixture.values[0], fixture.values[0] + live, seed_values);
    std::reverse(seed_keys, seed_keys + live);
    std::reverse(seed_values, seed_values + live);
#if CUDART_VERSION >= 13000
    const cudaMemLocation device_location{cudaMemLocationTypeDevice, 0};
    cudaMemPrefetchAsync(seed_keys, live * sizeof(*seed_keys),
                         device_location, 0U);
    cudaMemPrefetchAsync(seed_values, live * sizeof(*seed_values),
                         device_location, 0U);
#else
    cudaMemPrefetchAsync(seed_keys, live * sizeof(*seed_keys), 0);
    cudaMemPrefetchAsync(seed_values, live * sizeof(*seed_values), 0);
#endif
    cudaDeviceSynchronize();

    const auto reset_sort = [&] {
        cudaMemcpyAsync(fixture.keys[0], seed_keys,
            live * sizeof(*seed_keys), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(fixture.values[0], seed_values,
            live * sizeof(*seed_values), cudaMemcpyDeviceToDevice);
    };
    const auto modern = measure(reset_sort, [&] { fixture.launch_sort(); });
    const auto legacy = measure(reset_sort, [&] { launch_legacy_sort(fixture); });
    const auto full = measure([] {}, [&] { fixture.launch_build(); }, 3U, 24U);
    const auto dispatch = spatial_build_dispatch_budget(capacity, live);
    parallel_mater::simulation::SpatialBuildPlan plan{};
    CHECK(make_spatial_build_plan(capacity, plan) == PM_SUCCESS);
    const char *policy = dispatch.sort_policy == SpatialSortPolicy::workgroup_bitonic
        ? "workgroup-bitonic" : "radix8";
    std::printf(
        "F8_BENCH capacity=%u live=%u policy=%s sort_dispatches=%u "
        "legacy_sort_dispatches=48 build_dispatches=%u workgroups=%llu "
        "scratch_bytes=%zu output_bytes=%zu sort_p50_ms=%.6f "
        "sort_p95_ms=%.6f legacy_p50_ms=%.6f legacy_p95_ms=%.6f "
        "build_p50_ms=%.6f build_p95_ms=%.6f\n",
        capacity, live, policy, dispatch.sort_dispatches,
        dispatch.total_dispatches,
        static_cast<unsigned long long>(dispatch.total_workgroups),
        plan.scratch_bytes, plan.output_bytes,
        modern.first, modern.second, legacy.first, legacy.second,
        full.first, full.second);
    cudaFree(seed_values);
    cudaFree(seed_keys);
}
} // namespace

int main(int argc, char **argv) {
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return 77;
    cudaDeviceProp properties{};
    if (cudaSetDevice(0) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return 1;
    int driver{};
    int runtime{};
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("F8_DEVICE name=%s driver=%d runtime=%d cc=%d.%d\n",
        properties.name, driver, runtime, properties.major, properties.minor);

    for (const auto [capacity, live] :
         std::array<std::pair<unsigned, unsigned>, 8>{
             {{1U, 0U}, {1U, 1U}, {65536U, 128U}, {386U, 386U},
              {1024U, 1024U}, {4500U, 4500U}, {30065U, 30065U},
              {1U << 20U, 128U}}}) {
        Fixture fixture(capacity, live);
        const int before = failures;
        validate_fixture(fixture);
        std::printf("F8_CASE_RESULT capacity=%u live=%u failures=%d\n",
            capacity, live, failures - before);
    }
    validate_errors();
    validate_service_kernels();
    if (failures == 0)
        std::printf("F8-07 spatial service CUDA tests passed\n");
    if (argc > 1 && std::string_view(argv[1]) == "--benchmark") {
        for (const auto [capacity, live] :
             std::array<std::pair<unsigned, unsigned>, 5>{
                 {{65536U, 128U}, {386U, 386U}, {1024U, 1024U},
                  {4500U, 4500U}, {30065U, 30065U}}})
            benchmark(capacity, live);
    }
    if (failures == 0)
        std::printf("F8-06 shared Slang CUDA primitive tests passed\n");
    return failures == 0 ? 0 : 1;
}
