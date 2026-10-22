// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "simulation/deformable_spatial.hpp"
#include "parallel_mater_geometry_soft_body_runtime.cu"

namespace {
using namespace parallel_mater::simulation;

constexpr unsigned threads = 128U;
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

static_assert(sizeof(SpatialAabb) == sizeof(DeformableSpatialAabb_0));
static_assert(sizeof(SpatialLeafMetadata) ==
              sizeof(DeformableSpatialLeafMetadata_0));

template <typename T>
T *managed(std::size_t count) {
    T *result{};
    if (cudaMallocManaged(&result,
            std::max<std::size_t>(count, 1U) * sizeof(T)) != cudaSuccess)
        return nullptr;
    return result;
}

struct Fixture {
    static constexpr unsigned leaf_count = 6U;
    static constexpr unsigned query_count = 8U;
    static constexpr unsigned candidates_per_query = leaf_count;

    Vec3_0 *cloth_previous{managed<Vec3_0>(8U)};
    Vec3_0 *cloth_predicted{managed<Vec3_0>(8U)};
    unsigned *cloth_indices{managed<unsigned>(9U)};
    unsigned *cloth_generations{managed<unsigned>(3U)};
    Vec3_0 *rope_previous{managed<Vec3_0>(3U)};
    Vec3_0 *rope_predicted{managed<Vec3_0>(3U)};
    unsigned *rope_indices{managed<unsigned>(4U)};
    unsigned *rope_generations{managed<unsigned>(2U)};
    Vec3_0 *static_positions{managed<Vec3_0>(3U)};
    unsigned *static_indices{managed<unsigned>(3U)};
    unsigned *static_generations{managed<unsigned>(1U)};
    SpatialLeafMetadata *metadata{managed<SpatialLeafMetadata>(leaf_count)};
    SpatialAabb *bounds{managed<SpatialAabb>(leaf_count)};
    std::uint8_t *build_flags{managed<std::uint8_t>(leaf_count)};
    std::uint8_t *admission{managed<std::uint8_t>(leaf_count)};
    SpatialAabb *query_bounds{managed<SpatialAabb>(query_count)};
    unsigned *query_masks{managed<unsigned>(query_count)};
    unsigned *excluded{managed<unsigned>(query_count)};
    unsigned *query_leaf_sources{managed<unsigned>(query_count)};
    unsigned *candidate_counts{managed<unsigned>(query_count)};
    unsigned *candidate_sources{
        managed<unsigned>(query_count * candidates_per_query)};
    unsigned *filtered_counts{managed<unsigned>(query_count)};
    unsigned *filtered_sources{
        managed<unsigned>(query_count * candidates_per_query)};
    Vec3_0 *smoke_centers{managed<Vec3_0>(2U)};
    unsigned *status{managed<unsigned>(1U)};

    Fixture() {
        const std::array<Vec3_0, 8> cloth{{
            {-1.0F, 0.0F, 0.0F}, {0.0F, 0.8F, 0.0F},
            {0.0F, 0.0F, 0.8F}, {0.8F, 0.0F, 0.0F},
            {0.8F, 0.8F, 0.0F}, {-0.2F, 0.1F, 0.1F},
            {0.3F, 0.7F, 0.1F}, {0.3F, 0.1F, 0.7F}}};
        std::copy(cloth.begin(), cloth.end(), cloth_previous);
        std::copy(cloth.begin(), cloth.end(), cloth_predicted);
        cloth_predicted[0] = {1.0F, 0.0F, 0.0F};
        const std::array<unsigned, 9> ci{0U, 1U, 2U, 2U, 3U, 4U,
                                        5U, 6U, 7U};
        std::copy(ci.begin(), ci.end(), cloth_indices);
        std::fill(cloth_generations, cloth_generations + 3U, 1U);

        rope_previous[0] = rope_predicted[0] = {-0.5F, 0.2F, 0.2F};
        rope_previous[1] = rope_predicted[1] = {0.0F, 0.2F, 0.2F};
        rope_previous[2] = rope_predicted[2] = {0.5F, 0.2F, 0.2F};
        const std::array<unsigned, 4> ri{0U, 1U, 1U, 2U};
        std::copy(ri.begin(), ri.end(), rope_indices);
        std::fill(rope_generations, rope_generations + 2U, 2U);

        static_positions[0] = {-0.25F, -0.25F, 0.0F};
        static_positions[1] = {0.25F, -0.25F, 0.0F};
        static_positions[2] = {0.0F, 0.25F, 0.0F};
        static_indices[0] = 0U;
        static_indices[1] = 1U;
        static_indices[2] = 2U;
        static_generations[0] = 4U;

        smoke_centers[0] = {0.0F, 0.0F, 0.0F};
        smoke_centers[1] = {4.0F, 4.0F, 4.0F};
        std::fill(admission, admission + leaf_count, std::uint8_t{1U});
        std::fill(candidate_counts, candidate_counts + query_count,
                  leaf_count);
        for (unsigned query = 0U; query < query_count; ++query)
            for (unsigned source = 0U; source < leaf_count; ++source)
                candidate_sources[query * candidates_per_query + source] =
                    source;
        *status = 0U;
    }

    ~Fixture() {
        for (void *pointer : std::array<void *, 24>{
                 cloth_previous, cloth_predicted, cloth_indices,
                 cloth_generations, rope_previous, rope_predicted,
                 rope_indices, rope_generations, static_positions,
                 static_indices, static_generations, metadata, bounds,
                 build_flags, admission, query_bounds, query_masks, excluded,
                 query_leaf_sources, candidate_counts, candidate_sources,
                 filtered_counts, filtered_sources, smoke_centers})
            cudaFree(pointer);
        cudaFree(status);
    }

    bool allocated() const {
        return cloth_previous && cloth_predicted && cloth_indices &&
            cloth_generations && rope_previous && rope_predicted &&
            rope_indices && rope_generations && static_positions &&
            static_indices && static_generations && metadata && bounds &&
            build_flags && admission && query_bounds && query_masks &&
            excluded && query_leaf_sources && candidate_counts &&
            candidate_sources && filtered_counts && filtered_sources &&
            smoke_centers && status;
    }

    void emit_all() {
        const Vec3_0 deviation{0.02F, 0.03F, 0.04F};
        const Vec3_0 correction{0.01F, 0.01F, 0.01F};
        pm_emit_deformable_spatial_primitives<<<1U, threads>>>(
            cloth_previous, cloth_predicted, 8U, cloth_indices,
            cloth_generations, 3U, 3U, deformable_spatial_cloth_family,
            10U, 2U, 7U, deformable_spatial_triangle_primitive, 0U,
            leaf_count, 0U, query_count, 0U,
            deformable_spatial_contact_mask | smoke_spatial_obstacle_mask,
            deformable_spatial_contact_mask | smoke_spatial_obstacle_mask,
            0.01F, deviation, correction, 1.0e-5F,
            reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(metadata),
            reinterpret_cast<DeformableSpatialAabb_0 *>(bounds), build_flags,
            reinterpret_cast<DeformableSpatialAabb_0 *>(query_bounds),
            query_masks, excluded, query_leaf_sources, status);
        pm_emit_deformable_spatial_primitives<<<1U, threads>>>(
            rope_previous, rope_predicted, 3U, rope_indices,
            rope_generations, 2U, 2U, deformable_spatial_rope_family,
            20U, 3U, 8U, deformable_spatial_segment_primitive, 3U,
            leaf_count, 3U, query_count, 100U,
            deformable_spatial_contact_mask | smoke_spatial_obstacle_mask,
            deformable_spatial_contact_mask | smoke_spatial_obstacle_mask,
            0.02F, Vec3_0{}, Vec3_0{}, 1.0e-5F,
            reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(metadata),
            reinterpret_cast<DeformableSpatialAabb_0 *>(bounds), build_flags,
            reinterpret_cast<DeformableSpatialAabb_0 *>(query_bounds),
            query_masks, excluded, query_leaf_sources, status);
        pm_emit_deformable_spatial_primitives<<<1U, threads>>>(
            static_positions, static_positions, 3U, static_indices,
            static_generations, 1U, 3U, deformable_spatial_static_family,
            30U, 4U, 9U, deformable_spatial_triangle_primitive, 5U,
            leaf_count, 5U, query_count, 200U, 0U,
            smoke_spatial_obstacle_mask, 0.0F, Vec3_0{}, Vec3_0{}, 1.0e-5F,
            reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(metadata),
            reinterpret_cast<DeformableSpatialAabb_0 *>(bounds), build_flags,
            reinterpret_cast<DeformableSpatialAabb_0 *>(query_bounds),
            query_masks, excluded, query_leaf_sources, status);
    }

    void filter_cloth(unsigned target_family, unsigned target_owner,
                      unsigned target_generation, unsigned target_topology,
                      unsigned target_kind, unsigned *target_generations,
                      unsigned target_count, unsigned *target_indices,
                      unsigned target_vertices) {
        pm_filter_deformable_spatial_candidates<<<1U, threads>>>(
            reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(metadata),
            admission, leaf_count, query_leaf_sources, 3U,
            deformable_spatial_cloth_family, 10U, 2U, 7U,
            deformable_spatial_triangle_primitive, cloth_generations, 3U,
            cloth_indices, 3U, target_family, target_owner,
            target_generation, target_topology, target_kind,
            target_generations, target_count, target_indices, target_vertices,
            candidate_counts, candidate_sources, candidates_per_query,
            filtered_counts, filtered_sources, status);
    }
};

void run_fixture() {
    Fixture fixture;
    CHECK(fixture.allocated());
    fixture.emit_all();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    for (unsigned source = 0U; source < Fixture::leaf_count; ++source) {
        CHECK(fixture.build_flags[source] == 1U);
        CHECK(spatial_identity_valid(fixture.metadata[source].identity));
        CHECK(fixture.metadata[source].payload_ref ==
              (source < 3U ? source : source < 5U ? source - 3U : 0U));
        CHECK(fixture.bounds[source].minimum[0] <=
              fixture.bounds[source].maximum[0]);
    }
    CHECK(fixture.bounds[0].minimum[0] < -1.03F);
    CHECK(fixture.bounds[0].maximum[0] > 1.03F);
    CHECK(fixture.metadata[0].identity.family ==
          deformable_spatial_cloth_family);
    CHECK(fixture.metadata[3].identity.family ==
          deformable_spatial_rope_family);
    CHECK(fixture.metadata[5].identity.family ==
          deformable_spatial_static_family);

    fixture.filter_cloth(deformable_spatial_cloth_family, 10U, 2U, 7U,
        deformable_spatial_triangle_primitive, fixture.cloth_generations, 3U,
        fixture.cloth_indices, 3U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(fixture.filtered_counts[0] == 1U);
    CHECK(fixture.filtered_counts[1] == 1U);
    CHECK(fixture.filtered_counts[2] == 2U);
    CHECK(fixture.filtered_sources[0] == 2U);

    fixture.filter_cloth(deformable_spatial_rope_family, 20U, 3U, 8U,
        deformable_spatial_segment_primitive, fixture.rope_generations, 2U,
        fixture.rope_indices, 2U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(fixture.filtered_counts[0] == 2U);
    CHECK(fixture.filtered_sources[0] == 3U);
    CHECK(fixture.filtered_sources[1] == 4U);

    ++fixture.cloth_generations[2];
    fixture.filter_cloth(deformable_spatial_cloth_family, 10U, 2U, 7U,
        deformable_spatial_triangle_primitive, fixture.cloth_generations, 3U,
        fixture.cloth_indices, 3U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_epoch_status_stale_epoch) != 0U);
    fixture.cloth_generations[2] = 1U;
    *fixture.status = 0U;

    const Vec3_0 halfExtent{0.25F, 0.5F, 0.75F};
    pm_emit_smoke_spatial_queries<<<1U, threads>>>(
        fixture.smoke_centers, 2U, 6U, Fixture::query_count,
        smoke_spatial_obstacle_mask, halfExtent, Vec3_0{}, Vec3_0{},
        1.0e-5F,
        reinterpret_cast<DeformableSpatialAabb_0 *>(fixture.query_bounds),
        fixture.query_masks, fixture.excluded, fixture.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(fixture.query_masks[6] == smoke_spatial_obstacle_mask);
    CHECK(fixture.excluded[6] == 0U);
    CHECK(fixture.query_bounds[6].minimum[1] < -0.5F);

    const unsigned families =
        (1U << deformable_spatial_cloth_family) |
        (1U << deformable_spatial_rope_family) |
        (1U << deformable_spatial_static_family);
    pm_filter_smoke_spatial_obstacles<<<1U, threads>>>(
        reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(fixture.metadata),
        fixture.admission, Fixture::leaf_count, 2U, families,
        fixture.candidate_counts + 6U,
        fixture.candidate_sources + 6U * Fixture::candidates_per_query,
        Fixture::candidates_per_query, fixture.filtered_counts + 6U,
        fixture.filtered_sources + 6U * Fixture::candidates_per_query,
        fixture.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    CHECK(fixture.filtered_counts[6] == Fixture::leaf_count);
    CHECK(fixture.filtered_counts[7] == Fixture::leaf_count);

    fixture.metadata[0].identity.primitive_generation = 0U;
    pm_filter_smoke_spatial_obstacles<<<1U, threads>>>(
        reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(fixture.metadata),
        fixture.admission, Fixture::leaf_count, 1U, families,
        fixture.candidate_counts, fixture.candidate_sources,
        Fixture::candidates_per_query, fixture.filtered_counts,
        fixture.filtered_sources, fixture.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_epoch_status_stale_epoch) != 0U);
    fixture.metadata[0].identity.primitive_generation = 1U;
    *fixture.status = 0U;

    fixture.candidate_counts[0] = Fixture::candidates_per_query + 1U;
    fixture.filter_cloth(deformable_spatial_cloth_family, 10U, 2U, 7U,
        deformable_spatial_triangle_primitive, fixture.cloth_generations, 3U,
        fixture.cloth_indices, 3U);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_epoch_status_query_overflow) != 0U);
    CHECK(fixture.filtered_counts[0] == 0U);
}

} // namespace

int main() {
    int count{};
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    run_fixture();
    if (failures == 0)
        std::printf("F8-10 physical CUDA deformable/smoke spatial passed on %s\n",
                    properties.name);
    return failures == 0 ? 0 : 1;
}
