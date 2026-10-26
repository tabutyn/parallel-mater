// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "simulation/deformable_spatial.hpp"
#include "parallel_mater_geometry_soft_body_runtime.cu"
namespace soft_cloth_generated {
#include "parallel_mater_soft_body_cloth_runtime.cu"
}

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
static_assert(sizeof(SpatialLeafMetadata) ==
              sizeof(soft_cloth_generated::SoftClothSpatialLeafMetadata_0));

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

void run_point_query_fixture() {
    auto *previous = managed<Vec3_0>(2U);
    auto *predicted = managed<Vec3_0>(2U);
    auto *metadata = managed<SpatialLeafMetadata>(2U);
    auto *bounds = managed<SpatialAabb>(2U);
    auto *build_flags = managed<std::uint8_t>(2U);
    auto *query_bounds = managed<SpatialAabb>(2U);
    auto *query_masks = managed<unsigned>(2U);
    auto *excluded = managed<unsigned>(2U);
    auto *query_leaf_sources = managed<unsigned>(2U);
    auto *status = managed<unsigned>(1U);
    CHECK(previous && predicted && metadata && bounds && build_flags &&
          query_bounds && query_masks && excluded && query_leaf_sources &&
          status);
    previous[0] = {-1.0F, 0.0F, 0.0F};
    predicted[0] = {1.0F, 0.0F, 0.0F};
    previous[1] = predicted[1] = {0.0F, 2.0F, 0.0F};
    *status = 0U;
    const Vec3_0 deviation{0.05F, 0.0F, 0.0F};
    const Vec3_0 correction{0.1F, 0.1F, 0.1F};
    pm_emit_deformable_spatial_primitives<<<1U, threads>>>(
        previous, predicted, 2U, nullptr, nullptr, 2U, 1U,
        deformable_spatial_soft_family, 12U, 5U, 19U,
        deformable_spatial_point_primitive, 0U, 2U, 0U, 2U, 0U,
        deformable_spatial_contact_mask,
        deformable_spatial_query_identity_mask, 0.03F, deviation,
        correction, 1.0e-5F,
        reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(metadata),
        reinterpret_cast<DeformableSpatialAabb_0 *>(bounds), build_flags,
        reinterpret_cast<DeformableSpatialAabb_0 *>(query_bounds),
        query_masks, excluded, query_leaf_sources, status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*status == 0U);
    for (unsigned node = 0U; node < 2U; ++node) {
        CHECK(build_flags[node] == 1U);
        CHECK(metadata[node].identity.family ==
              deformable_spatial_soft_family);
        CHECK(metadata[node].identity.owner_slot == 12U);
        CHECK(metadata[node].identity.owner_generation == 5U);
        CHECK(metadata[node].identity.topology_revision == 19U);
        CHECK(metadata[node].identity.primitive_kind ==
              deformable_spatial_point_primitive);
        CHECK(metadata[node].identity.primitive_slot == node);
        CHECK(metadata[node].identity.primitive_generation == 5U);
        CHECK(metadata[node].payload_ref == node);
        CHECK(metadata[node].admission_mask ==
              deformable_spatial_query_identity_mask);
        CHECK(query_leaf_sources[node] == node);
        CHECK(query_masks[node] == deformable_spatial_contact_mask);
    }
    CHECK(bounds[0].minimum[0] < -1.18F);
    CHECK(bounds[0].maximum[0] > 1.18F);
    query_masks[0] = query_masks[1] = 0U;
    excluded[0] = excluded[1] = 99U;
    pm_emit_deformable_spatial_primitives<<<1U, threads>>>(
        previous, predicted, 2U, nullptr, nullptr, 2U, 1U,
        deformable_spatial_soft_family, 12U, 5U, 19U,
        deformable_spatial_point_primitive, 0U, 2U, 0U, 2U, 0U,
        coupling_spatial_query_mask, 0U, 0.03F, deviation, correction,
        1.0e-5F,
        reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(metadata),
        reinterpret_cast<DeformableSpatialAabb_0 *>(bounds), build_flags,
        reinterpret_cast<DeformableSpatialAabb_0 *>(query_bounds),
        query_masks, excluded, query_leaf_sources, status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*status == 0U);
    CHECK(query_masks[0] == coupling_spatial_query_mask);
    CHECK(query_masks[1] == coupling_spatial_query_mask);
    CHECK(excluded[0] == 0U);
    CHECK(excluded[1] == 0U);
    CHECK(query_bounds[0].minimum[0] < -1.18F);
    CHECK(query_bounds[0].maximum[0] > 1.18F);
    for (unsigned node = 0U; node < 2U; ++node) {
        CHECK(build_flags[node] == 1U);
        CHECK(metadata[node].admission_mask ==
              deformable_spatial_query_identity_mask);
        CHECK(query_leaf_sources[node] == node);
    }
    for (void *pointer : std::array<void *, 10>{
             previous, predicted, metadata, bounds, build_flags,
             query_bounds, query_masks, excluded, query_leaf_sources,
             status})
        cudaFree(pointer);
}

void run_soft_cloth_candidate_fixture() {
    auto *cloth_positions = managed<Vec3_0>(3U);
    auto *cloth_previous = managed<Vec3_0>(3U);
    auto *cloth_velocities = managed<Vec3_0>(3U);
    auto *cloth_inverse_masses = managed<float>(3U);
    auto *indices = managed<unsigned>(3U);
    auto *triangle_generations = managed<unsigned>(1U);
    auto *soft_positions = managed<Vec3_0>(1U);
    auto *soft_previous = managed<Vec3_0>(1U);
    auto *soft_velocities = managed<Vec3_0>(1U);
    auto *soft_inverse_masses = managed<float>(1U);
    auto *metadata = managed<SpatialLeafMetadata>(2U);
    auto *bounds = managed<SpatialAabb>(2U);
    auto *build_flags = managed<std::uint8_t>(2U);
    auto *query_bounds = managed<SpatialAabb>(2U);
    auto *query_masks = managed<unsigned>(2U);
    auto *excluded = managed<unsigned>(2U);
    auto *query_leaf_sources = managed<unsigned>(2U);
    auto *admission = managed<std::uint8_t>(2U);
    auto *candidate_counts = managed<unsigned>(1U);
    auto *candidate_sources = managed<unsigned>(2U);
    auto *contacts = managed<soft_cloth_generated::SoftClothContact_0>(1U);
    auto *contact_counts = managed<unsigned>(3U);
    auto *status = managed<unsigned>(1U);
    CHECK(cloth_positions && cloth_previous && cloth_velocities &&
          cloth_inverse_masses && indices && triangle_generations &&
          soft_positions && soft_previous && soft_velocities &&
          soft_inverse_masses && metadata && bounds && build_flags &&
          query_bounds && query_masks && excluded && query_leaf_sources &&
          admission && candidate_counts && candidate_sources && contacts &&
          contact_counts && status);
    cloth_positions[0] = cloth_previous[0] = {-0.5F, -0.5F, 0.0F};
    cloth_positions[1] = cloth_previous[1] = {0.5F, -0.5F, 0.0F};
    cloth_positions[2] = cloth_previous[2] = {0.0F, 0.5F, 0.0F};
    std::fill(cloth_velocities, cloth_velocities + 3U, Vec3_0{});
    std::fill(cloth_inverse_masses, cloth_inverse_masses + 3U, 1.0F);
    indices[0] = 0U;
    indices[1] = 1U;
    indices[2] = 2U;
    *triangle_generations = 3U;
    *soft_positions = *soft_previous = {0.0F, 0.0F, 0.02F};
    *soft_velocities = {};
    *soft_inverse_masses = 1.0F;
    *status = 0U;
    std::fill(contact_counts, contact_counts + 3U, 0U);
    const Vec3_0 zero{};
    pm_emit_deformable_spatial_primitives<<<1U, threads>>>(
        cloth_previous, cloth_positions, 3U, indices,
        triangle_generations, 1U, 3U,
        deformable_spatial_cloth_family, 4U, 8U, 11U,
        deformable_spatial_triangle_primitive, 0U, 2U, 0U, 2U, 0U,
        deformable_spatial_contact_mask, deformable_spatial_contact_mask,
        0.01F, zero, zero, 1.0e-5F,
        reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(metadata),
        reinterpret_cast<DeformableSpatialAabb_0 *>(bounds), build_flags,
        reinterpret_cast<DeformableSpatialAabb_0 *>(query_bounds),
        query_masks, excluded, query_leaf_sources, status);
    pm_emit_deformable_spatial_primitives<<<1U, threads>>>(
        soft_previous, soft_positions, 1U, nullptr, nullptr, 1U, 1U,
        deformable_spatial_soft_family, 6U, 9U, 13U,
        deformable_spatial_point_primitive, 1U, 2U, 1U, 2U, 1U,
        deformable_spatial_contact_mask,
        deformable_spatial_query_identity_mask, 0.04F, zero, zero,
        1.0e-5F,
        reinterpret_cast<DeformableSpatialLeafMetadata_0 *>(metadata),
        reinterpret_cast<DeformableSpatialAabb_0 *>(bounds), build_flags,
        reinterpret_cast<DeformableSpatialAabb_0 *>(query_bounds),
        query_masks, excluded, query_leaf_sources, status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    admission[0] = admission[1] = std::uint8_t{1U};
    *candidate_counts = 1U;
    candidate_sources[0] = 0U;
    candidate_sources[1] = 1U;
    soft_cloth_generated::soft_cloth_detect<<<1U, 256U>>>(
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(soft_positions),
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(soft_previous),
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(soft_velocities),
        soft_inverse_masses, 1U,
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(cloth_positions),
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(cloth_previous),
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(cloth_velocities),
        cloth_inverse_masses, indices, nullptr, 1U,
        triangle_generations, 4U, 8U, 11U,
        reinterpret_cast<soft_cloth_generated::SoftClothSpatialLeafMetadata_0 *>(metadata),
        admission, 2U, query_leaf_sources + 1U, candidate_counts,
        candidate_sources, 2U, 6U, 9U, 13U, status, 0.05F, 0.2F,
        contacts, contact_counts);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*status == 0U);
    CHECK(contacts[0].active_0);
    CHECK(contact_counts[0] + contact_counts[1] + contact_counts[2] != 0U);

    ++triangle_generations[0];
    soft_cloth_generated::soft_cloth_detect<<<1U, 256U>>>(
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(soft_positions),
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(soft_previous),
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(soft_velocities),
        soft_inverse_masses, 1U,
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(cloth_positions),
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(cloth_previous),
        reinterpret_cast<soft_cloth_generated::Vec3_0 *>(cloth_velocities),
        cloth_inverse_masses, indices, nullptr, 1U,
        triangle_generations, 4U, 8U, 11U,
        reinterpret_cast<soft_cloth_generated::SoftClothSpatialLeafMetadata_0 *>(metadata),
        admission, 2U, query_leaf_sources + 1U, candidate_counts,
        candidate_sources, 2U, 6U, 9U, 13U, status, 0.05F, 0.2F,
        contacts, contact_counts);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*status & spatial_epoch_status_stale_epoch) != 0U);

    for (void *pointer : std::array<void *, 21>{
             cloth_positions, cloth_previous, cloth_velocities,
             cloth_inverse_masses, indices, triangle_generations,
             soft_positions, soft_previous, soft_velocities,
             soft_inverse_masses, metadata, bounds, build_flags,
             query_bounds, query_masks, excluded, query_leaf_sources,
             admission, candidate_counts, candidate_sources, contacts})
        cudaFree(pointer);
    cudaFree(contact_counts);
    cudaFree(status);
}

} // namespace

int main() {
    int count{};
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, 0) == cudaSuccess);
    run_fixture();
    run_point_query_fixture();
    run_soft_cloth_candidate_fixture();
    if (failures == 0)
        std::printf("F8-10 physical CUDA deformable/smoke spatial passed on %s\n",
                    properties.name);
    return failures == 0 ? 0 : 1;
}
