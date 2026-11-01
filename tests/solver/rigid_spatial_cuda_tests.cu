// SPDX-License-Identifier: MIT
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include "simulation/geometry_storage.hpp"
#include "simulation/rigid_spatial.hpp"
#include "parallel_mater_geometry_runtime.cu"

namespace {
using namespace parallel_mater;
using namespace parallel_mater::simulation;

constexpr unsigned threads = 128U;
int failures{};
#define CHECK(condition) do { if (!(condition)) { \
    if (++failures <= 80) std::fprintf(stderr, "%s:%d check failed: %s\n", \
        __FILE__, __LINE__, #condition); \
} } while (false)

static_assert(sizeof(BodyParameters) == sizeof(BodyParameters_0));
static_assert(sizeof(RigidBodyState) == sizeof(RigidBodyState_0));
static_assert(sizeof(TriangleMeshResource) == sizeof(TriangleMeshResource_0));
static_assert(sizeof(RigidBodyId) == sizeof(RigidBodyId_0));
static_assert(sizeof(RigidConstraintResource) ==
              sizeof(RigidConstraintResource_0));
static_assert(sizeof(FixedContactProjection) ==
              sizeof(FixedContactProjection_0));
static_assert(sizeof(SpatialAabb) == sizeof(RigidSpatialAabb_0));
static_assert(sizeof(SpatialLeafMetadata) ==
              sizeof(RigidSpatialLeafMetadata_0));

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

bool id_less(const SpatialIdentity &left, const SpatialIdentity &right) {
    return left < right;
}

struct Fixture {
    static constexpr unsigned count = 10U;
    static constexpr unsigned slots = count * count;

    BodyParameters *parameters{managed<BodyParameters>(count)};
    RigidBodyState *previous{managed<RigidBodyState>(count)};
    RigidBodyState *predicted{managed<RigidBodyState>(count)};
    TriangleMeshResource *meshes{managed<TriangleMeshResource>(count)};
    RigidBodyId *ids{managed<RigidBodyId>(count)};
    RigidConstraintResource *constraints{managed<RigidConstraintResource>(1U)};
    FixedContactProjection *fixed_groups{managed<FixedContactProjection>(count)};
    SpatialLeafMetadata *metadata{managed<SpatialLeafMetadata>(count)};
    SpatialAabb *bounds{managed<SpatialAabb>(count)};
    std::uint8_t *build_flags{managed<std::uint8_t>(count)};
    SpatialAabb *queries{managed<SpatialAabb>(count)};
    unsigned *query_masks{managed<unsigned>(count)};
    unsigned *excluded{managed<unsigned>(count)};
    unsigned *candidate_counts{managed<unsigned>(count)};
    unsigned *candidate_sources{managed<unsigned>(slots)};
    unsigned *pair_codes{managed<unsigned>(slots)};
    std::uint8_t *pair_flags{managed<std::uint8_t>(slots)};
    unsigned *compacted_slots{managed<unsigned>(slots)};
    unsigned *active_count{managed<unsigned>(1U)};
    unsigned *active_pairs{managed<unsigned>(slots)};
    unsigned *status{managed<unsigned>(1U)};

    Fixture() {
        for (unsigned index = 0U; index < count; ++index) {
            parameters[index] = {};
            parameters[index].motion = MotionType::static_body;
            parameters[index].mesh = {index, 1U};
            parameters[index].collision_margin = 0.0F;
            previous[index] = {};
            previous[index].orientation.w = 1.0F;
            predicted[index] = previous[index];
            meshes[index] = {};
            meshes[index].generation = 1U;
            meshes[index].alive = 1U;
            meshes[index].bounding_radius = 0.2F;
            ids[index] = {100U + index, 7U + index};
            fixed_groups[index].root = index;
        }
        parameters[0].motion = MotionType::dynamic;
        previous[0].position = {-5.0F, 0.0F, 0.0F};
        predicted[0].position = {5.0F, 0.0F, 0.0F};

        parameters[2].motion = MotionType::dynamic;
        meshes[2].bounding_center = {2.0F, 0.0F, 0.0F};
        meshes[2].bounding_radius = 0.1F;
        parameters[3].motion = MotionType::kinematic;
        previous[3].position = predicted[3].position = {0.0F, 2.0F, 0.0F};
        meshes[3].bounding_radius = 0.1F;

        parameters[4].motion = MotionType::dynamic;
        parameters[4].collision_margin = 0.11F;
        parameters[5].collision_margin = 0.11F;
        previous[4].position = predicted[4].position = {10.0F, 0.0F, 0.0F};
        previous[5].position = predicted[5].position = {10.6F, 0.0F, 0.0F};

        parameters[6].motion = MotionType::dynamic;
        previous[6].position = predicted[6].position = {20.0F, 0.0F, 0.0F};
        previous[7].position = predicted[7].position = {20.0F, 0.0F, 0.0F};
        constraints[0] = RigidConstraintResource{};
        constraints[0].alive = 1U;
        constraints[0].options.enabled = 1U;
        constraints[0].options.disable_collisions = 1U;
        constraints[0].options.body_a = ids[6];
        constraints[0].options.body_b = ids[7];

        parameters[8].motion = MotionType::dynamic;
        previous[8].position = predicted[8].position = {30.0F, 0.0F, 0.0F};
        previous[9].position = predicted[9].position = {30.0F, 0.0F, 0.0F};
        fixed_groups[8].root = fixed_groups[9].root = 55U;
        *status = 0U;
        *active_count = 0U;
    }

    ~Fixture() {
        for (void *pointer : std::array<void *, 21>{
                 parameters, previous, predicted, meshes, ids, constraints,
                 fixed_groups, metadata, bounds, build_flags, queries,
                 query_masks, excluded, candidate_counts, candidate_sources,
                 pair_codes, pair_flags, compacted_slots, active_count,
                 active_pairs, status})
            cudaFree(pointer);
    }

    bool allocated() const {
        return parameters && previous && predicted && meshes && ids &&
            constraints && fixed_groups && metadata && bounds && build_flags &&
            queries && query_masks && excluded && candidate_counts &&
            candidate_sources && pair_codes && pair_flags && compacted_slots &&
            active_count && active_pairs && status;
    }

    void emit(float epsilon = 1.0e-5F) {
        pm_emit_rigid_spatial_proxies<<<1U, threads>>>(
            reinterpret_cast<BodyParameters_0 *>(parameters),
            reinterpret_cast<RigidBodyState_0 *>(previous),
            reinterpret_cast<RigidBodyState_0 *>(predicted),
            reinterpret_cast<TriangleMeshResource_0 *>(meshes), count,
            reinterpret_cast<RigidBodyId_0 *>(ids), count, 0U, count, 0U,
            1U, rigid_spatial_query_mask, Vec3_0{}, Vec3_0{}, epsilon,
            reinterpret_cast<RigidSpatialLeafMetadata_0 *>(metadata),
            reinterpret_cast<RigidSpatialAabb_0 *>(bounds), build_flags,
            reinterpret_cast<RigidSpatialAabb_0 *>(queries), query_masks,
            excluded, status);
    }

    void make_candidates() {
        std::fill(candidate_counts, candidate_counts + count, 0U);
        std::fill(candidate_sources, candidate_sources + slots, 0U);
        for (unsigned query = 0U; query < count; ++query) {
            for (unsigned target = 0U; target < count; ++target) {
                if (query == target || !overlap(bounds[query], bounds[target]))
                    continue;
                candidate_sources[query * count +
                    candidate_counts[query]++] = target;
            }
        }
    }

    void filter() {
        pm_filter_rigid_spatial_candidates<<<1U, threads>>>(
            reinterpret_cast<RigidSpatialLeafMetadata_0 *>(metadata), count,
            0U, candidate_counts, candidate_sources, count, slots,
            reinterpret_cast<BodyParameters_0 *>(parameters),
            reinterpret_cast<RigidBodyId_0 *>(ids),
            reinterpret_cast<RigidConstraintResource_0 *>(constraints), 1U,
            reinterpret_cast<FixedContactProjection_0 *>(fixed_groups), count,
            pair_codes, pair_flags, status);
    }
};

std::set<unsigned> expected_pairs(const Fixture &fixture) {
    std::set<unsigned> result;
    for (unsigned body = 0U; body < Fixture::count; ++body) {
        if (fixture.parameters[body].motion != MotionType::dynamic) continue;
        for (unsigned collider = 0U; collider < Fixture::count; ++collider) {
            if (body == collider ||
                !overlap(fixture.bounds[body], fixture.bounds[collider]))
                continue;
            if (fixture.parameters[collider].motion == MotionType::dynamic &&
                !id_less(fixture.metadata[body].identity,
                         fixture.metadata[collider].identity)) continue;
            if (fixture.fixed_groups[body].root ==
                fixture.fixed_groups[collider].root) continue;
            const bool disabled =
                (fixture.ids[body] == fixture.ids[6] &&
                 fixture.ids[collider] == fixture.ids[7]) ||
                (fixture.ids[body] == fixture.ids[7] &&
                 fixture.ids[collider] == fixture.ids[6]);
            if (disabled) continue;
            result.insert(body * Fixture::count + collider);
        }
    }
    return result;
}

void run_fixture() {
    Fixture fixture;
    CHECK(fixture.allocated());
    fixture.emit(0.0F);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_epoch_status_invalid_identity) != 0U);
    *fixture.status = 0U;
    fixture.emit();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    for (unsigned index = 0U; index < Fixture::count; ++index) {
        CHECK(fixture.build_flags[index] == 1U);
        CHECK(fixture.metadata[index].identity.family == rigid_spatial_family);
        CHECK(fixture.metadata[index].identity.owner_slot ==
              fixture.ids[index].index);
        CHECK(fixture.metadata[index].identity.owner_generation ==
              fixture.ids[index].generation);
        CHECK(fixture.metadata[index].identity.primitive_generation == 1U);
        CHECK(fixture.metadata[index].payload_ref == index);
        CHECK(fixture.metadata[index].query_mask == rigid_spatial_query_mask);
        CHECK(fixture.metadata[index].admission_mask ==
              rigid_spatial_query_mask);
        const auto expected_query_mask =
            fixture.parameters[index].motion == MotionType::dynamic
                ? rigid_spatial_query_mask : 0U;
        CHECK(fixture.query_masks[index] == expected_query_mask);
        CHECK(fixture.excluded[index] == index + 1U);
    }

    fixture.make_candidates();
    fixture.filter();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    unsigned compacted{};
    for (unsigned slot = 0U; slot < Fixture::slots; ++slot)
        if (fixture.pair_flags[slot] != 0U)
            fixture.compacted_slots[compacted++] = slot;
    *fixture.active_count = compacted;
    pm_gather_rigid_spatial_pairs<<<1U, threads>>>(
        fixture.compacted_slots, fixture.active_count, fixture.pair_codes,
        Fixture::slots, fixture.active_pairs, fixture.status);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(*fixture.status == 0U);
    std::set<unsigned> actual(fixture.active_pairs,
                              fixture.active_pairs + compacted);
    const auto expected = expected_pairs(fixture);
    CHECK(actual == expected);
    // Static and kinematic bodies do not issue body-indexed queries, but both
    // remain admitted leaves found by eligible dynamic queries.
    CHECK(fixture.parameters[1].motion == MotionType::static_body);
    CHECK(fixture.query_masks[1] == 0U);
    CHECK(actual.contains(0U * Fixture::count + 1U));
    CHECK(fixture.parameters[3].motion == MotionType::kinematic);
    CHECK(fixture.query_masks[3] == 0U);
    CHECK(actual.contains(2U * Fixture::count + 3U));
    CHECK(actual.contains(4U * Fixture::count + 5U));
    CHECK(!actual.contains(6U * Fixture::count + 7U));
    CHECK(!actual.contains(8U * Fixture::count + 9U));

    // A corrupted/overflowed shared query count fails atomically instead of
    // permitting a partial candidate list to reach narrow phase.
    fixture.candidate_counts[0] = Fixture::count + 1U;
    *fixture.status = 0U;
    fixture.filter();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_epoch_status_query_overflow) != 0U);
    fixture.make_candidates();

    // An in-place owner generation change cannot alias the pinned epoch.
    ++fixture.ids[1].generation;
    *fixture.status = 0U;
    fixture.filter();
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    CHECK((*fixture.status & spatial_epoch_status_stale_epoch) != 0U);
}

} // namespace

int main() {
    int devices{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    run_fixture();
    if (failures == 0)
        std::printf("F8-08 physical rigid spatial kernels passed\n");
    return failures == 0 ? 0 : 1;
}
