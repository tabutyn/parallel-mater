// SPDX-License-Identifier: MIT
#include "simulation/production_spatial_runtime.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace {
using namespace parallel_mater::simulation;

int failures{};
#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                     __FILE__, __LINE__, #condition); \
        ++failures; \
    } \
} while (false)

SpatialAabb bounds(float low, float high) {
    SpatialAabb result{};
    result.minimum[0] = low;
    result.minimum[1] = -0.25F;
    result.minimum[2] = -0.25F;
    result.maximum[0] = high;
    result.maximum[1] = 0.25F;
    result.maximum[2] = 0.25F;
    return result;
}

SpatialLeafMetadata rigid_leaf(
    std::uint32_t owner, std::uint32_t generation,
    std::uint32_t sort_key,
    std::uint32_t flags = spatial_leaf_retained,
    std::uint32_t transfer_group = 0U) {
    SpatialLeafMetadata result{};
    result.identity = {1U, owner, generation, 1U, 1U, owner, generation};
    result.sort_key = sort_key;
    result.payload_ref = owner;
    result.query_mask = 1U;
    result.admission_mask = 1U;
    result.flags = flags;
    result.transfer_group = transfer_group;
    return result;
}

void write_leaf(ProductionSpatialSnapshot &snapshot, std::uint32_t slot,
                SpatialLeafMetadata metadata, SpatialAabb envelope) {
    snapshot.metadata[slot] = metadata;
    snapshot.bounds[slot] = envelope;
    snapshot.build_flags[slot] =
        (metadata.flags & spatial_leaf_inactive) == 0U ? 1U : 0U;
}

void write_inactive(ProductionSpatialSnapshot &snapshot,
                    std::uint32_t slot) {
    snapshot.metadata[slot] = {};
    snapshot.metadata[slot].flags = spatial_leaf_inactive;
    snapshot.bounds[slot] = bounds(0.0F, 0.0F);
    snapshot.build_flags[slot] = 0U;
}

void validate_production_path() {
    SpatialServicePlan plan{};
    CHECK(make_spatial_service_plan(8U, 4U, 2U, 3U, plan) == PM_SUCCESS);
    std::vector<SpatialStorageBlock> persistent(
        spatial_storage_block_count(plan.persistent_bytes));
    std::vector<SpatialStorageBlock> scratch(
        spatial_storage_block_count(plan.scratch_bytes));
    ProductionSpatialRuntime runtime;
    CHECK(runtime.configure(plan, persistent.data(), scratch.data(), true) ==
          PM_SUCCESS);

    // Empty worlds still publish one generation-tagged epoch without an
    // invented build or refit.
    CHECK(runtime.begin_tick(1U, 1U, 1U, 1U, 0U) == PM_SUCCESS);
    CHECK(runtime.update(SpatialUpdateKind::empty) == PM_SUCCESS);
    SpatialEpochToken empty{};
    CHECK(runtime.publish(empty) == PM_SUCCESS);
    CHECK(runtime.commit(empty) == PM_SUCCESS);
    const auto *empty_counters = runtime.counters(empty);
    CHECK(empty_counters != nullptr);
    CHECK(empty_counters->builds == 0U);
    CHECK(empty_counters->refits == 0U);
    CHECK(empty_counters->publications == 1U);

    // Sparse rigid leaves include an inactive capacity slot. A crossing query
    // consumes the production tree and records counters on this exact epoch.
    CHECK(runtime.begin_tick(2U, 1U, 2U, 2U, 4U) == PM_SUCCESS);
    auto sparse = runtime.staging_snapshot();
    CHECK(bool(sparse));
    write_leaf(sparse, 0U, rigid_leaf(0U, 1U, 30U), bounds(-2.0F, 1.0F));
    write_leaf(sparse, 1U, rigid_leaf(1U, 1U, 10U), bounds(3.0F, 5.0F));
    write_inactive(sparse, 2U);
    write_leaf(sparse, 3U, rigid_leaf(3U, 1U, 20U), bounds(8.0F, 9.0F));
    CHECK(runtime.update(SpatialUpdateKind::build) == PM_SUCCESS);
    SpatialEpochToken sparse_token{};
    CHECK(runtime.publish(sparse_token) == PM_SUCCESS);
    CHECK(runtime.query(sparse_token, 0U, bounds(0.5F, 3.5F), 1U) ==
          PM_SUCCESS);
    CHECK(sparse.query_counts[0] == 2U);
    CHECK(sparse.metadata[sparse.query_candidates[0]].identity.owner_slot ==
          0U);
    CHECK(sparse.metadata[sparse.query_candidates[1]].identity.owner_slot ==
          1U);
    CHECK(runtime.validate(sparse_token,
        {sparse.metadata, 4U}, {sparse.bounds, 4U}, 2U, 2U) == PM_SUCCESS);
    CHECK(runtime.commit(sparse_token) == PM_SUCCESS);
    const auto *sparse_counters = runtime.counters(sparse_token);
    CHECK(sparse_counters != nullptr);
    CHECK(sparse_counters->builds == 1U);
    CHECK(sparse_counters->refits == 0U);
    CHECK(sparse_counters->publications == 1U);
    CHECK(sparse_counters->queries == 1U);
    CHECK(sparse_counters->candidates == 2U);

    // Exact identity/topology reuse selects one refit and keeps the same
    // canonical sources while conservative bounds move.
    CHECK(runtime.begin_tick(3U, 1U, 2U, 3U, 4U, true) == PM_SUCCESS);
    auto refit = runtime.staging_snapshot();
    write_leaf(refit, 0U, rigid_leaf(0U, 1U, 30U), bounds(-3.0F, 2.0F));
    write_leaf(refit, 1U, rigid_leaf(1U, 1U, 10U), bounds(2.5F, 5.5F));
    write_inactive(refit, 2U);
    write_leaf(refit, 3U, rigid_leaf(3U, 1U, 20U), bounds(7.0F, 10.0F));
    CHECK(runtime.update(SpatialUpdateKind::refit) == PM_SUCCESS);
    SpatialEpochToken refit_token{};
    CHECK(runtime.publish(refit_token) == PM_SUCCESS);
    CHECK(runtime.validate(refit_token,
        {refit.metadata, 4U}, {refit.bounds, 4U}, 2U, 3U) == PM_SUCCESS);
    CHECK(runtime.commit(refit_token) == PM_SUCCESS);
    const auto *refit_counters = runtime.counters(refit_token);
    CHECK(refit_counters != nullptr);
    CHECK(refit_counters->builds == 0U);
    CHECK(refit_counters->refits == 1U);
    CHECK(refit_counters->publications == 1U);

    // A replacement cannot masquerade as a refit. Abort leaves tick three as
    // the committed epoch, then the same tick can perform one full build.
    CHECK(runtime.begin_tick(4U, 1U, 3U, 4U, 4U, true) == PM_SUCCESS);
    auto replacement = runtime.staging_snapshot();
    write_leaf(replacement, 0U, rigid_leaf(0U, 2U, 30U), bounds(-1.0F, 1.0F));
    write_leaf(replacement, 1U, rigid_leaf(1U, 1U, 10U), bounds(0.0F, 1.0F));
    write_inactive(replacement, 2U);
    write_leaf(replacement, 3U, rigid_leaf(3U, 1U, 20U), bounds(4.0F, 5.0F));
    CHECK(runtime.update(SpatialUpdateKind::refit) == PM_ERROR_INVALID_STATE);
    runtime.abort();
    CHECK(runtime.current_token().has_value());
    CHECK(runtime.current_token()->tick == 3U);

    CHECK(runtime.begin_tick(4U, 1U, 3U, 4U, 4U) == PM_SUCCESS);
    auto transfer = runtime.staging_snapshot();
    write_leaf(transfer, 0U, rigid_leaf(0U, 2U, 30U,
        spatial_leaf_retained | spatial_leaf_transfer_source, 7U),
        bounds(0.0F, 1.0F));
    write_leaf(transfer, 1U, rigid_leaf(1U, 1U, 10U,
        spatial_leaf_provisional | spatial_leaf_transfer_destination, 7U),
        bounds(0.0F, 1.0F));
    write_inactive(transfer, 2U);
    write_leaf(transfer, 3U, rigid_leaf(3U, 1U, 20U),
        bounds(4.0F, 5.0F));
    CHECK(runtime.update(SpatialUpdateKind::build) == PM_SUCCESS);
    CHECK(transfer.admission[0] == 0U);
    CHECK(transfer.admission[1] == 1U);
    SpatialEpochToken transfer_token{};
    CHECK(runtime.publish(transfer_token) == PM_SUCCESS);
    CHECK(runtime.validate(transfer_token,
        {transfer.metadata, 4U}, {transfer.bounds, 4U}, 3U, 4U) == PM_SUCCESS);
    CHECK(runtime.commit(transfer_token) == PM_SUCCESS);

    // Capacity and invalid identity fail before publication and never replace
    // the prior committed token.
    CHECK(runtime.begin_tick(5U, 1U, 4U, 5U, 9U) ==
          PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(runtime.begin_tick(5U, 1U, 4U, 5U, 1U) == PM_SUCCESS);
    auto invalid = runtime.staging_snapshot();
    write_leaf(invalid, 0U, rigid_leaf(0U, 0U, 1U), bounds(0.0F, 1.0F));
    CHECK(runtime.update(SpatialUpdateKind::build) == PM_ERROR_INVALID_ARGUMENT);
    runtime.abort();
    CHECK(runtime.current_token()->tick == 4U);

    // Identity drift after publication latches failure just like an envelope
    // escape; it cannot be ignored by a caller that still attempts commit.
    CHECK(runtime.begin_tick(5U, 1U, 4U, 5U, 1U) == PM_SUCCESS);
    auto stale = runtime.staging_snapshot();
    write_leaf(stale, 0U, rigid_leaf(0U, 3U, 1U), bounds(0.0F, 1.0F));
    CHECK(runtime.update(SpatialUpdateKind::build) == PM_SUCCESS);
    SpatialEpochToken stale_token{};
    CHECK(runtime.publish(stale_token) == PM_SUCCESS);
    CHECK(bool(runtime.published_snapshot(stale_token)));
    std::array stale_metadata{stale.metadata[0]};
    stale_metadata[0].identity.primitive_generation += 1U;
    std::array stale_bounds{stale.bounds[0]};
    CHECK(runtime.validate(stale_token, stale_metadata, stale_bounds,
                           4U, 5U) == PM_ERROR_INVALID_STATE);
    CHECK(runtime.commit(stale_token) == PM_ERROR_INVALID_STATE);
    runtime.abort();
    CHECK(runtime.current_token()->tick == 4U);

    // Query overflow is all-or-none and invalidates only the unpublished tick.
    CHECK(runtime.begin_tick(5U, 1U, 4U, 5U, 3U) == PM_SUCCESS);
    auto overflow = runtime.staging_snapshot();
    for (std::uint32_t index = 0U; index < 3U; ++index)
        write_leaf(overflow, index, rigid_leaf(index, 1U, index + 1U),
                   bounds(0.0F, 1.0F));
    CHECK(runtime.update(SpatialUpdateKind::build) == PM_SUCCESS);
    SpatialEpochToken overflow_token{};
    CHECK(runtime.publish(overflow_token) == PM_SUCCESS);
    CHECK(runtime.query(overflow_token, 0U, bounds(0.0F, 1.0F), 1U) ==
          PM_ERROR_CAPACITY_EXCEEDED);
    CHECK(overflow.query_counts[0] == 0U);
    CHECK(runtime.commit(overflow_token) == PM_ERROR_INVALID_STATE);
    runtime.abort();
    CHECK(runtime.current_token()->tick == 4U);

    // Envelope escape also aborts atomically. Retrying the same tick succeeds.
    CHECK(runtime.begin_tick(5U, 1U, 4U, 5U, 1U) == PM_SUCCESS);
    auto escaped = runtime.staging_snapshot();
    write_leaf(escaped, 0U, rigid_leaf(0U, 3U, 1U), bounds(0.0F, 1.0F));
    CHECK(runtime.update(SpatialUpdateKind::build) == PM_SUCCESS);
    SpatialEpochToken escaped_token{};
    CHECK(runtime.publish(escaped_token) == PM_SUCCESS);
    std::array escaped_metadata{escaped.metadata[0]};
    std::array escaped_bounds{bounds(-1.0F, 2.0F)};
    CHECK(runtime.validate(escaped_token, escaped_metadata, escaped_bounds,
                           4U, 5U) == PM_ERROR_INVALID_STATE);
    CHECK(runtime.commit(escaped_token) == PM_ERROR_INVALID_STATE);
    runtime.abort();
    CHECK(runtime.current_token()->tick == 4U);

    CHECK(runtime.begin_tick(5U, 1U, 4U, 5U, 1U) == PM_SUCCESS);
    auto recovered = runtime.staging_snapshot();
    write_leaf(recovered, 0U, rigid_leaf(0U, 3U, 1U), bounds(-1.0F, 2.0F));
    CHECK(runtime.update(SpatialUpdateKind::build) == PM_SUCCESS);
    SpatialEpochToken recovered_token{};
    CHECK(runtime.publish(recovered_token) == PM_SUCCESS);
    CHECK(runtime.validate(recovered_token,
        {recovered.metadata, 1U}, {recovered.bounds, 1U}, 4U, 5U) ==
        PM_SUCCESS);
    CHECK(runtime.commit(recovered_token) == PM_SUCCESS);
    CHECK(runtime.current_token()->tick == 5U);
}

} // namespace

int main() {
    validate_production_path();
    if (failures == 0)
        std::printf("F8-33 production spatial runtime tests passed\n");
    return failures == 0 ? 0 : 1;
}
