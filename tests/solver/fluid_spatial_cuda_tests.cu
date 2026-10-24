// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "simulation/fluid_spatial.hpp"
#include "simulation/types.hpp"
#include "parallel_mater_fluid_runtime.cu"

namespace {
using namespace parallel_mater;
using namespace parallel_mater::simulation;

constexpr unsigned threads = 128U;
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

static_assert(sizeof(Vec3) == sizeof(Vec3_0));
static_assert(sizeof(SpatialAabb) == sizeof(FluidSpatialAabb_0));
static_assert(sizeof(SpatialIdentity) == sizeof(FluidSpatialIdentity_0));
static_assert(sizeof(SpatialLeafMetadata) ==
              sizeof(FluidSpatialLeafMetadata_0));

template <typename T>
T *managed(std::size_t count) {
    T *result{};
    if (cudaMallocManaged(&result,
            std::max<std::size_t>(count, 1U) * sizeof(T)) != cudaSuccess)
        return nullptr;
    return result;
}

bool overlap(const SpatialAabb &left, const SpatialAabb &right) {
    return spatial_overlaps(left, right);
}

struct Fixture {
    static constexpr unsigned first_capacity = 6U;
    static constexpr unsigned second_capacity = 5U;
    static constexpr unsigned proposal_count = 4U;
    static constexpr unsigned leaf_capacity =
        first_capacity + second_capacity + proposal_count;
    static constexpr unsigned query_capacity = leaf_capacity;
    static constexpr unsigned candidates_per_query = leaf_capacity;
    static constexpr unsigned candidate_slots =
        query_capacity * candidates_per_query;
    static constexpr unsigned first_owner = 11U;
    static constexpr unsigned second_owner = 29U;
    static constexpr unsigned first_generation = 3U;
    static constexpr unsigned second_generation = 7U;
    static constexpr unsigned second_leaf_offset = first_capacity;
    static constexpr unsigned proposal_leaf_offset =
        first_capacity + second_capacity;
    static constexpr unsigned proposal_query_offset = proposal_leaf_offset;

    Vec3 *first_previous{managed<Vec3>(first_capacity)};
    Vec3 *first_positions{managed<Vec3>(first_capacity)};
    Vec3 *first_velocities{managed<Vec3>(first_capacity)};
    unsigned *first_ids{managed<unsigned>(first_capacity)};
    float *first_foam{managed<float>(first_capacity)};
    float *first_temperatures{managed<float>(first_capacity)};
    unsigned *first_count{managed<unsigned>(1U)};

    Vec3 *second_previous{managed<Vec3>(second_capacity)};
    Vec3 *second_positions{managed<Vec3>(second_capacity)};
    unsigned *second_ids{managed<unsigned>(second_capacity)};
    unsigned *second_count{managed<unsigned>(1U)};

    Vec3 *proposal_points{managed<Vec3>(proposal_count)};
    SpatialLeafMetadata *metadata{managed<SpatialLeafMetadata>(leaf_capacity)};
    SpatialAabb *bounds{managed<SpatialAabb>(leaf_capacity)};
    std::uint8_t *build_flags{managed<std::uint8_t>(leaf_capacity)};
    std::uint8_t *admission{managed<std::uint8_t>(leaf_capacity)};
    SpatialAabb *query_bounds{managed<SpatialAabb>(query_capacity)};
    unsigned *query_masks{managed<unsigned>(query_capacity)};
    unsigned *excluded{managed<unsigned>(query_capacity)};
    unsigned *query_sources{managed<unsigned>(query_capacity)};
    unsigned *candidate_counts{managed<unsigned>(query_capacity)};
    unsigned *candidate_sources{managed<unsigned>(candidate_slots)};
    unsigned *filtered_counts{managed<unsigned>(query_capacity)};
    unsigned *filtered_sources{managed<unsigned>(candidate_slots)};
    unsigned *neighbor_counts{managed<unsigned>(query_capacity)};
    unsigned *overflow{managed<unsigned>(1U)};
    unsigned *maximum{managed<unsigned>(1U)};
    unsigned *accepted{managed<unsigned>(1U)};
    unsigned *rejected{managed<unsigned>(1U)};
    unsigned *capacity_misses{managed<unsigned>(1U)};
    unsigned *status{managed<unsigned>(1U)};

    Fixture() {
        *first_count = 4U;
        *second_count = 3U;
        for (unsigned index = 0U; index < first_capacity; ++index) {
            first_previous[index] = {float(index) * 0.30F, 0.0F, 0.0F};
            first_positions[index] = first_previous[index];
            first_velocities[index] = {};
            first_ids[index] = 100U + index;
            first_foam[index] = 0.0F;
            first_temperatures[index] = 280.0F;
        }
        // The first particle sweeps across the second population. Its query
        // envelope must retain later cross-population neighbors.
        first_previous[0] = {-1.0F, 0.0F, 0.0F};
        first_positions[0] = {1.0F, 0.0F, 0.0F};
        for (unsigned index = 0U; index < second_capacity; ++index) {
            second_previous[index] = {0.70F + float(index) * 0.42F,
                                      0.0F, 0.0F};
            second_positions[index] = second_previous[index];
            second_ids[index] = 200U + index;
        }
        proposal_points[0] = first_positions[1];
        proposal_points[1] = {2.00F, 0.0F, 0.0F};
        proposal_points[2] = {3.00F, 0.0F, 0.0F};
        proposal_points[3] = {4.00F, 0.0F, 0.0F};
        std::fill(admission, admission + leaf_capacity, std::uint8_t{0U});
        *overflow = *maximum = *status = 0U;
    }

    ~Fixture() {
        for (void *pointer : std::array<void *, 29>{
                 first_previous, first_positions, first_velocities, first_ids,
                 first_foam, first_temperatures, first_count,
                 second_previous, second_positions, second_ids, second_count,
                 proposal_points, metadata, bounds, build_flags, admission,
                 query_bounds, query_masks, excluded, query_sources,
                 candidate_counts, candidate_sources, filtered_counts,
                 filtered_sources, neighbor_counts, overflow, maximum,
                 accepted, rejected})
            cudaFree(pointer);
        cudaFree(capacity_misses);
        cudaFree(status);
    }

    bool allocated() const {
        return first_previous && first_positions && first_velocities &&
            first_ids && first_foam && first_temperatures && first_count &&
            second_previous && second_positions && second_ids &&
            second_count && proposal_points && metadata && bounds &&
            build_flags && admission && query_bounds && query_masks &&
            excluded && query_sources && candidate_counts &&
            candidate_sources && filtered_counts && filtered_sources &&
            neighbor_counts && overflow && maximum && accepted && rejected &&
            capacity_misses && status;
    }

    void emit() {
        const Vec3_0 zero{};
        fluid_emit_spatial_particles<<<1U, threads>>>(
            reinterpret_cast<Vec3_0 *>(first_previous),
            reinterpret_cast<Vec3_0 *>(first_positions), first_ids,
            first_count, first_capacity, first_owner, first_generation, 13U,
            0U, leaf_capacity, 0U, query_capacity, 0U,
            fluid_spatial_query_mask, fluid_spatial_query_mask, 0.10F, 0.55F,
            zero, zero, 1.0e-5F,
            reinterpret_cast<FluidSpatialLeafMetadata_0 *>(metadata),
            reinterpret_cast<FluidSpatialAabb_0 *>(bounds), build_flags,
            reinterpret_cast<FluidSpatialAabb_0 *>(query_bounds), query_masks,
            excluded, query_sources, status);
        fluid_emit_spatial_particles<<<1U, threads>>>(
            reinterpret_cast<Vec3_0 *>(second_previous),
            reinterpret_cast<Vec3_0 *>(second_positions), second_ids,
            second_count, second_capacity, second_owner, second_generation,
            17U, second_leaf_offset, leaf_capacity, second_leaf_offset,
            query_capacity, second_leaf_offset, fluid_spatial_query_mask,
            fluid_spatial_query_mask, 0.10F, 0.55F, zero, zero, 1.0e-5F,
            reinterpret_cast<FluidSpatialLeafMetadata_0 *>(metadata),
            reinterpret_cast<FluidSpatialAabb_0 *>(bounds), build_flags,
            reinterpret_cast<FluidSpatialAabb_0 *>(query_bounds), query_masks,
            excluded, query_sources, status);
        fluid_emit_spatial_source_proposals<<<1U, threads>>>(
            reinterpret_cast<Vec3_0 *>(proposal_points), proposal_count, 0.20F,
            first_count, first_capacity, 500U, first_owner, first_generation,
            13U, proposal_leaf_offset, leaf_capacity, proposal_query_offset,
            query_capacity, proposal_leaf_offset, fluid_spatial_query_mask,
            fluid_spatial_query_mask, 0.55F, zero, zero, 1.0e-5F,
            reinterpret_cast<FluidSpatialLeafMetadata_0 *>(metadata),
            reinterpret_cast<FluidSpatialAabb_0 *>(bounds), build_flags,
            reinterpret_cast<FluidSpatialAabb_0 *>(query_bounds), query_masks,
            excluded, query_sources, status);
    }

    void resolve_and_apply_sources() {
        for (unsigned source = 0U; source < leaf_capacity; ++source)
            admission[source] = build_flags[source] != 0U ? 1U : 0U;
        admission[proposal_leaf_offset] = 0U;
        admission[proposal_leaf_offset + 1U] = 1U;
        const Vec3_0 velocity{0.25F, 0.0F, 0.0F};
        fluid_apply_spatial_source_admission<<<1U, 1U>>>(
            reinterpret_cast<Vec3_0 *>(proposal_points), proposal_count,
            velocity, 310.0F, proposal_leaf_offset, leaf_capacity,
            proposal_query_offset, query_capacity,
            reinterpret_cast<FluidSpatialLeafMetadata_0 *>(metadata),
            admission, query_masks,
            reinterpret_cast<Vec3_0 *>(first_positions),
            reinterpret_cast<Vec3_0 *>(first_velocities), first_ids,
            first_foam, first_temperatures, first_count, first_capacity,
            accepted, rejected, capacity_misses, status);
    }

    void make_candidates() {
        std::fill(candidate_counts, candidate_counts + query_capacity, 0U);
        std::fill(candidate_sources, candidate_sources + candidate_slots, 0U);
        for (unsigned query = 0U; query < query_capacity; ++query) {
            if (query_masks[query] == 0U) continue;
            std::vector<unsigned> sources;
            for (unsigned source = 0U; source < leaf_capacity; ++source) {
                if (admission[source] == 0U ||
                    metadata[source].sort_key == excluded[query] ||
                    (metadata[source].query_mask & query_masks[query]) == 0U ||
                    !overlap(query_bounds[query], bounds[source])) continue;
                sources.push_back(source);
            }
            std::sort(sources.begin(), sources.end(), [&](unsigned a,
                                                          unsigned b) {
                return metadata[a].identity < metadata[b].identity;
            });
            candidate_counts[query] = static_cast<unsigned>(sources.size());
            std::copy(sources.begin(), sources.end(),
                      candidate_sources + query * candidates_per_query);
        }
    }

    void filter(unsigned query_offset, unsigned query_count,
                unsigned query_owner, unsigned query_generation,
                unsigned *query_ids, unsigned *query_live,
                unsigned query_storage_capacity, unsigned target_owner,
                unsigned target_generation, unsigned *target_ids,
                unsigned *target_live, unsigned target_storage_capacity) {
        fluid_filter_spatial_candidates<<<1U, threads>>>(
            reinterpret_cast<FluidSpatialLeafMetadata_0 *>(metadata),
            admission, leaf_capacity, query_sources + query_offset,
            query_count, query_owner, query_generation, query_ids, query_live,
            query_storage_capacity, target_owner, target_generation,
            target_ids, target_live, target_storage_capacity,
            candidate_counts + query_offset,
            candidate_sources + query_offset * candidates_per_query,
            candidates_per_query, filtered_counts + query_offset,
            filtered_sources + query_offset * candidates_per_query, status);
    }

    void count(unsigned query_offset, unsigned query_count,
               Vec3 *query_positions_value, unsigned *query_live,
               unsigned query_storage_capacity, Vec3 *target_positions_value,
               unsigned *target_live, unsigned target_storage_capacity,
               float support, unsigned maximum_neighbors) {
        fluid_count_spatial_neighbors<<<1U, threads>>>(
            reinterpret_cast<FluidSpatialLeafMetadata_0 *>(metadata),
            admission, leaf_capacity, query_sources + query_offset,
            query_count, reinterpret_cast<Vec3_0 *>(query_positions_value),
            query_live, query_storage_capacity,
            reinterpret_cast<Vec3_0 *>(target_positions_value), target_live,
            target_storage_capacity, filtered_counts + query_offset,
            filtered_sources + query_offset * candidates_per_query,
            candidates_per_query, support, maximum_neighbors,
            neighbor_counts + query_offset, overflow, maximum, status);
    }
};

unsigned brute_count(const Vec3 &query, const Vec3 *targets,
                     unsigned target_count, float support,
                     bool self, unsigned self_index) {
    unsigned result{};
    for (unsigned index = 0U; index < target_count; ++index) {
        if (self && index == self_index) continue;
        const float dx = query.x - targets[index].x;
        const float dy = query.y - targets[index].y;
        const float dz = query.z - targets[index].z;
        if (dx * dx + dy * dy + dz * dz < support * support) ++result;
    }
    return result;
}

void run_fixture() {
    Fixture fixture;
    CHECK(fixture.allocated());
    fixture.emit();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    for (unsigned index = 0U; index < *fixture.first_count; ++index) {
        const auto &identity = fixture.metadata[index].identity;
        CHECK(identity.family == fluid_spatial_family);
        CHECK(identity.owner_slot == Fixture::first_owner);
        CHECK(identity.owner_generation == Fixture::first_generation);
        CHECK(identity.topology_revision == 13U);
        CHECK(identity.primitive_slot == fixture.first_ids[index]);
        CHECK(fixture.query_sources[index] == index);
    }
    CHECK(fixture.build_flags[4] == 0U);
    CHECK(fixture.build_flags[5] == 0U);
    CHECK(fixture.build_flags[Fixture::proposal_leaf_offset] == 1U);
    CHECK(fixture.build_flags[Fixture::proposal_leaf_offset + 1U] == 1U);
    CHECK(fixture.build_flags[Fixture::proposal_leaf_offset + 2U] == 0U);

    fixture.resolve_and_apply_sources();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(*fixture.first_count == 5U);
    CHECK(*fixture.accepted == 1U);
    CHECK(*fixture.rejected == 1U);
    CHECK(*fixture.capacity_misses == 2U);
    CHECK(fixture.first_ids[4] == 501U);
    CHECK(fixture.metadata[Fixture::proposal_leaf_offset + 1U].payload_ref ==
          4U);
    CHECK(fixture.query_masks[Fixture::proposal_query_offset] == 0U);

    fixture.make_candidates();
    // Self-population queries use the first support; cross-population queries
    // use max(first, second). Both consume the same candidate arrays.
    fixture.filter(0U, Fixture::first_capacity, Fixture::first_owner,
        Fixture::first_generation, fixture.first_ids, fixture.first_count,
        Fixture::first_capacity, Fixture::first_owner,
        Fixture::first_generation, fixture.first_ids, fixture.first_count,
        Fixture::first_capacity);
    fixture.count(0U, Fixture::first_capacity, fixture.first_positions,
        fixture.first_count, Fixture::first_capacity, fixture.first_positions,
        fixture.first_count, Fixture::first_capacity, 0.35F, 1U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    for (unsigned query = 0U; query < 4U; ++query)
        CHECK(fixture.neighbor_counts[query] == brute_count(
            fixture.first_positions[query], fixture.first_positions,
            *fixture.first_count, 0.35F, true, query));
    CHECK(*fixture.overflow > 0U);

    *fixture.overflow = *fixture.maximum = 0U;
    fixture.filter(0U, Fixture::first_capacity, Fixture::first_owner,
        Fixture::first_generation, fixture.first_ids, fixture.first_count,
        Fixture::first_capacity, Fixture::second_owner,
        Fixture::second_generation, fixture.second_ids, fixture.second_count,
        Fixture::second_capacity);
    fixture.count(0U, Fixture::first_capacity, fixture.first_positions,
        fixture.first_count, Fixture::first_capacity,
        fixture.second_positions, fixture.second_count,
        Fixture::second_capacity, 0.55F, 2U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    for (unsigned query = 0U; query < 4U; ++query)
        CHECK(fixture.neighbor_counts[query] == brute_count(
            fixture.first_positions[query], fixture.second_positions,
            *fixture.second_count, 0.55F, false, 0U));

    // Accepted proposal uses the same published hierarchy and stable identity.
    fixture.filter(Fixture::proposal_query_offset, Fixture::proposal_count,
        Fixture::first_owner, Fixture::first_generation, fixture.first_ids,
        fixture.first_count, Fixture::first_capacity, Fixture::second_owner,
        Fixture::second_generation, fixture.second_ids, fixture.second_count,
        Fixture::second_capacity);
    fixture.count(Fixture::proposal_query_offset, Fixture::proposal_count,
        fixture.first_positions, fixture.first_count, Fixture::first_capacity,
        fixture.second_positions, fixture.second_count,
        Fixture::second_capacity, 0.55F, 2U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(fixture.neighbor_counts[Fixture::proposal_query_offset + 1U] ==
          brute_count(fixture.first_positions[4], fixture.second_positions,
                      *fixture.second_count, 0.55F, false, 0U));

    // Shared query overflow is atomic; no partial neighbor list survives.
    fixture.candidate_counts[0] = Fixture::candidates_per_query + 1U;
    *fixture.status = 0U;
    fixture.filter(0U, 1U, Fixture::first_owner,
        Fixture::first_generation, fixture.first_ids, fixture.first_count,
        Fixture::first_capacity, Fixture::second_owner,
        Fixture::second_generation, fixture.second_ids, fixture.second_count,
        Fixture::second_capacity);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_epoch_status_query_overflow) != 0U);
    CHECK(fixture.filtered_counts[0] == 0U);

    // Generation/stable-ID drift cannot alias the pinned epoch.
    fixture.make_candidates();
    ++fixture.second_ids[0];
    *fixture.status = 0U;
    fixture.filter(0U, Fixture::first_capacity, Fixture::first_owner,
        Fixture::first_generation, fixture.first_ids, fixture.first_count,
        Fixture::first_capacity, Fixture::second_owner,
        Fixture::second_generation, fixture.second_ids, fixture.second_count,
        Fixture::second_capacity);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_epoch_status_stale_epoch) != 0U);
}

} // namespace

int main() {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    run_fixture();
    if (failures == 0)
        std::printf("F8-09 physical fluid spatial kernels passed\n");
    return failures == 0 ? 0 : 1;
}
