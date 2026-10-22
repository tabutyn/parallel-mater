// SPDX-License-Identifier: MIT
#pragma once

#include "spatial_build.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t spatial_service_max_snapshots = 4U;
inline constexpr std::uint32_t spatial_invalid_index = 0xffffffffU;

inline constexpr std::uint32_t spatial_epoch_status_invalid_identity = 1U << 8U;
inline constexpr std::uint32_t spatial_epoch_status_stale_epoch = 1U << 9U;
inline constexpr std::uint32_t spatial_epoch_status_query_overflow = 1U << 10U;
inline constexpr std::uint32_t spatial_epoch_status_envelope_escape = 1U << 11U;
inline constexpr std::uint32_t spatial_epoch_status_admission_invalid = 1U << 12U;
inline constexpr std::uint32_t spatial_epoch_status_duplicate_update = 1U << 13U;
inline constexpr std::uint32_t spatial_epoch_status_duplicate_publication = 1U << 14U;

inline constexpr std::uint32_t spatial_leaf_retained = 1U << 0U;
inline constexpr std::uint32_t spatial_leaf_provisional = 1U << 1U;
inline constexpr std::uint32_t spatial_leaf_transfer_source = 1U << 2U;
inline constexpr std::uint32_t spatial_leaf_transfer_destination = 1U << 3U;
inline constexpr std::uint32_t spatial_leaf_inactive = 1U << 4U;

struct SpatialIdentity {
    std::uint32_t family{};
    std::uint32_t owner_slot{};
    std::uint32_t owner_generation{};
    std::uint32_t topology_revision{};
    std::uint32_t primitive_kind{};
    std::uint32_t primitive_slot{};
    std::uint32_t primitive_generation{};

    auto operator<=>(const SpatialIdentity &) const = default;
};

[[nodiscard]] constexpr bool spatial_identity_valid(
    const SpatialIdentity &identity) noexcept {
    return identity.family != 0U && identity.owner_generation != 0U &&
        identity.topology_revision != 0U &&
        identity.primitive_generation != 0U;
}

// The full identity is pinned with each epoch. sort_key is a unique compact
// rank key for the epoch; dense storage and Morton rank never become identity.
struct alignas(16) SpatialLeafMetadata {
    SpatialIdentity identity{};
    std::uint32_t sort_key{};
    std::uint32_t payload_ref{};
    std::uint32_t query_mask{};
    std::uint32_t admission_mask{};
    std::uint32_t flags{};
    std::uint32_t transfer_group{};
    std::uint32_t reserved[3]{};
};

static_assert(sizeof(SpatialLeafMetadata) == 64U);
static_assert(alignof(SpatialLeafMetadata) == 16U);

struct alignas(16) SpatialEpochHeader {
    std::uint64_t tick{};
    std::uint64_t graph_revision{};
    std::uint64_t resource_revision{};
    std::uint32_t profile_version{};
    std::uint32_t slot_generation{};
    std::uint32_t leaf_count{};
    std::uint32_t node_count{};
    std::uint32_t status{};
    std::uint32_t build_count{};
    std::uint32_t refit_count{};
    std::uint32_t publication_count{};
    std::uint32_t reserved_query_count{};
    std::uint32_t reserved[5]{};
};

static_assert(sizeof(SpatialEpochHeader) == 80U);
static_assert(alignof(SpatialEpochHeader) == 16U);

struct SpatialSnapshotPlan {
    SpatialBufferRange header{};
    SpatialBufferRange metadata{};
    SpatialBufferRange bounds{};
    SpatialBufferRange build_flags{};
    SpatialBufferRange admission{};
    SpatialBufferRange build_output{};
    SpatialBufferRange query_counts{};
    SpatialBufferRange query_candidates{};
    SpatialBufferRange query_visited_nodes{};
    SpatialBufferRange query_counter{};
    SpatialBufferRange query_status{};
};

struct SpatialServicePlan {
    std::uint32_t leaf_capacity{};
    std::uint32_t query_capacity{};
    std::uint32_t candidates_per_query{};
    std::uint32_t snapshot_count{};
    SpatialBuildPlan build{};
    std::size_t persistent_bytes{};
    std::size_t scratch_bytes{};
    std::array<SpatialSnapshotPlan, spatial_service_max_snapshots> snapshots{};
    SpatialBufferRange build_scratch{};
};

[[nodiscard]] inline PmResult make_spatial_service_plan(
    std::uint32_t leaf_capacity, std::uint32_t query_capacity,
    std::uint32_t candidates_per_query, std::uint32_t snapshot_count,
    SpatialServicePlan &plan) noexcept {
    plan = {};
    if (snapshot_count < 2U ||
        snapshot_count > spatial_service_max_snapshots)
        return PM_ERROR_INVALID_ARGUMENT;
    if ((query_capacity == 0U) != (candidates_per_query == 0U))
        return PM_ERROR_INVALID_ARGUMENT;
    const auto result = make_spatial_build_plan(leaf_capacity, plan.build);
    if (result != PM_SUCCESS) return result;
    plan.leaf_capacity = leaf_capacity;
    plan.query_capacity = query_capacity;
    plan.candidates_per_query = candidates_per_query;
    plan.snapshot_count = snapshot_count;

    std::size_t candidates{};
    if (!spatial_build_detail::checked_multiply(
            query_capacity, candidates_per_query, candidates) ||
        candidates > std::numeric_limits<std::uint32_t>::max())
        return PM_ERROR_CAPACITY_EXCEEDED;

    using spatial_build_detail::append_range;
    std::size_t persistent{};
    for (std::uint32_t index = 0U; index < snapshot_count; ++index) {
        auto &snapshot = plan.snapshots[index];
        if (!append_range(persistent, 1U, sizeof(SpatialEpochHeader),
                          snapshot.header) ||
            !append_range(persistent, leaf_capacity,
                          sizeof(SpatialLeafMetadata), snapshot.metadata) ||
            !append_range(persistent, leaf_capacity, sizeof(SpatialAabb),
                          snapshot.bounds) ||
            !append_range(persistent, leaf_capacity, sizeof(std::uint8_t),
                          snapshot.build_flags) ||
            !append_range(persistent, leaf_capacity, sizeof(std::uint8_t),
                          snapshot.admission) ||
            !append_range(persistent, plan.build.output_bytes, 1U,
                          snapshot.build_output) ||
            !append_range(persistent, query_capacity, sizeof(std::uint32_t),
                          snapshot.query_counts) ||
            !append_range(persistent, candidates, sizeof(std::uint32_t),
                          snapshot.query_candidates) ||
            !append_range(persistent, query_capacity, sizeof(std::uint32_t),
                          snapshot.query_visited_nodes) ||
            !append_range(persistent, 1U, sizeof(std::uint32_t),
                          snapshot.query_counter) ||
            !append_range(persistent, 1U, sizeof(std::uint32_t),
                          snapshot.query_status))
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    plan.persistent_bytes = persistent;

    std::size_t scratch{};
    if (!append_range(scratch, plan.build.scratch_bytes, 1U,
                      plan.build_scratch))
        return PM_ERROR_CAPACITY_EXCEEDED;
    plan.scratch_bytes = scratch;
    return PM_SUCCESS;
}

[[nodiscard]] inline bool spatial_finite(const SpatialAabb &bounds) noexcept {
    for (std::uint32_t axis = 0U; axis < 3U; ++axis)
        if (!std::isfinite(bounds.minimum[axis]) ||
            !std::isfinite(bounds.maximum[axis]) ||
            bounds.minimum[axis] > bounds.maximum[axis]) return false;
    return true;
}

[[nodiscard]] inline bool spatial_overlaps(
    const SpatialAabb &left, const SpatialAabb &right) noexcept {
    for (std::uint32_t axis = 0U; axis < 3U; ++axis)
        if (left.maximum[axis] < right.minimum[axis] ||
            right.maximum[axis] < left.minimum[axis]) return false;
    return true;
}

[[nodiscard]] inline bool spatial_contains(
    const SpatialAabb &outer, const SpatialAabb &inner) noexcept {
    for (std::uint32_t axis = 0U; axis < 3U; ++axis)
        if (inner.minimum[axis] < outer.minimum[axis] ||
            inner.maximum[axis] > outer.maximum[axis]) return false;
    return true;
}

[[nodiscard]] inline SpatialAabb spatial_union(
    const SpatialAabb &left, const SpatialAabb &right) noexcept {
    SpatialAabb output{};
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        output.minimum[axis] = std::min(left.minimum[axis], right.minimum[axis]);
        output.maximum[axis] = std::max(left.maximum[axis], right.maximum[axis]);
    }
    return output;
}

[[nodiscard]] inline PmResult spatial_point_envelope(
    const float start[3], const float predicted[3], const float path_deviation[3],
    const float correction[3], float radius, SpatialAabb &output) noexcept {
    if (start == nullptr || predicted == nullptr || path_deviation == nullptr ||
        correction == nullptr || !std::isfinite(radius) || radius < 0.0F)
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        if (!std::isfinite(start[axis]) || !std::isfinite(predicted[axis]) ||
            !std::isfinite(path_deviation[axis]) ||
            !std::isfinite(correction[axis]) || path_deviation[axis] < 0.0F ||
            correction[axis] < 0.0F) return PM_ERROR_INVALID_ARGUMENT;
        const float expansion = path_deviation[axis] + correction[axis] + radius;
        if (!std::isfinite(expansion)) return PM_ERROR_CAPACITY_EXCEEDED;
        output.minimum[axis] = std::nextafter(
            std::min(start[axis], predicted[axis]) - expansion,
            -std::numeric_limits<float>::infinity());
        output.maximum[axis] = std::nextafter(
            std::max(start[axis], predicted[axis]) + expansion,
            std::numeric_limits<float>::infinity());
    }
    return spatial_finite(output) ? PM_SUCCESS : PM_ERROR_CAPACITY_EXCEEDED;
}

[[nodiscard]] inline PmResult spatial_weighted_envelope(
    std::span<const SpatialAabb> sources, std::span<const float> weights,
    SpatialAabb &output) noexcept {
    if (sources.empty() || sources.size() != weights.size())
        return PM_ERROR_INVALID_ARGUMENT;
    output = {};
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        double minimum{};
        double maximum{};
        for (std::size_t index = 0U; index < sources.size(); ++index) {
            if (!spatial_finite(sources[index]) ||
                !std::isfinite(weights[index])) return PM_ERROR_INVALID_ARGUMENT;
            const double low = static_cast<double>(weights[index]) *
                sources[index].minimum[axis];
            const double high = static_cast<double>(weights[index]) *
                sources[index].maximum[axis];
            minimum += std::min(low, high);
            maximum += std::max(low, high);
        }
        if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
            minimum < -std::numeric_limits<float>::max() ||
            maximum > std::numeric_limits<float>::max())
            return PM_ERROR_CAPACITY_EXCEEDED;
        output.minimum[axis] = std::nextafter(static_cast<float>(minimum),
            -std::numeric_limits<float>::infinity());
        output.maximum[axis] = std::nextafter(static_cast<float>(maximum),
            std::numeric_limits<float>::infinity());
    }
    return PM_SUCCESS;
}

[[nodiscard]] inline PmResult spatial_resolve_admission(
    std::span<const SpatialLeafMetadata> metadata,
    std::span<const SpatialAabb> bounds,
    std::span<std::uint8_t> admitted) noexcept {
    if (metadata.size() != bounds.size() || admitted.size() != bounds.size())
        return PM_ERROR_INVALID_ARGUMENT;
    std::fill(admitted.begin(), admitted.end(), std::uint8_t{});
    for (std::size_t index = 0U; index < metadata.size(); ++index) {
        const auto &leaf = metadata[index];
        if ((leaf.flags & spatial_leaf_inactive) != 0U) continue;
        const auto role = leaf.flags &
            (spatial_leaf_retained | spatial_leaf_provisional);
        if (!spatial_identity_valid(leaf.identity) ||
            !spatial_finite(bounds[index]) || leaf.sort_key == 0U ||
            (role != spatial_leaf_retained &&
             role != spatial_leaf_provisional))
            return PM_ERROR_INVALID_ARGUMENT;
        for (std::size_t previous = 0U; previous < index; ++previous)
            if ((metadata[previous].flags & spatial_leaf_inactive) == 0U &&
                metadata[previous].sort_key == leaf.sort_key)
                return PM_ERROR_INVALID_ARGUMENT;
        const bool transfer_source =
            (leaf.flags & spatial_leaf_transfer_source) != 0U;
        const bool transfer_destination =
            (leaf.flags & spatial_leaf_transfer_destination) != 0U;
        if ((transfer_source && transfer_destination) ||
            (transfer_source && role != spatial_leaf_retained) ||
            (transfer_destination && role != spatial_leaf_provisional) ||
            ((transfer_source || transfer_destination) &&
             leaf.transfer_group == 0U))
            return PM_ERROR_INVALID_ARGUMENT;
        if ((leaf.flags & spatial_leaf_retained) != 0U)
            admitted[index] = 1U;
    }

    // Every proposal compares with all lower-identity proposals, even rejected
    // ones. This deterministic priority rule intentionally permits underfill.
    for (std::size_t index = 0U; index < metadata.size(); ++index) {
        const auto &leaf = metadata[index];
        if ((leaf.flags & spatial_leaf_inactive) != 0U) continue;
        if ((leaf.flags & spatial_leaf_provisional) == 0U) continue;
        bool conflict = false;
        for (std::size_t other = 0U; other < metadata.size(); ++other) {
            if (index == other ||
                (metadata[other].flags & spatial_leaf_inactive) != 0U ||
                (leaf.admission_mask & metadata[other].admission_mask) == 0U ||
                !spatial_overlaps(bounds[index], bounds[other])) continue;
            const bool own_source = leaf.transfer_group != 0U &&
                leaf.transfer_group == metadata[other].transfer_group &&
                (leaf.flags & spatial_leaf_transfer_destination) != 0U &&
                (metadata[other].flags & spatial_leaf_transfer_source) != 0U;
            if (own_source) continue;
            const bool retained =
                (metadata[other].flags & spatial_leaf_retained) != 0U;
            const bool lower_proposal =
                (metadata[other].flags & spatial_leaf_provisional) != 0U &&
                metadata[other].identity < leaf.identity;
            if (retained || lower_proposal) {
                conflict = true;
                break;
            }
        }
        admitted[index] = conflict ? 0U : 1U;
    }

    for (std::size_t index = 0U; index < metadata.size(); ++index) {
        const auto &source = metadata[index];
        if ((source.flags & spatial_leaf_inactive) != 0U) continue;
        if ((source.flags & spatial_leaf_transfer_source) == 0U) continue;
        std::size_t destination = metadata.size();
        std::size_t source_count{};
        for (std::size_t other = 0U; other < metadata.size(); ++other) {
            if (metadata[other].transfer_group != source.transfer_group ||
                (metadata[other].flags & spatial_leaf_inactive) != 0U) continue;
            if ((metadata[other].flags & spatial_leaf_transfer_source) != 0U)
                ++source_count;
            if ((metadata[other].flags &
                 spatial_leaf_transfer_destination) != 0U) {
                if (destination != metadata.size())
                    return PM_ERROR_INVALID_ARGUMENT;
                destination = other;
            }
        }
        if (source_count != 1U || destination == metadata.size())
            return PM_ERROR_INVALID_ARGUMENT;
        admitted[index] = admitted[destination] == 0U ? 1U : 0U;
    }
    return PM_SUCCESS;
}

[[nodiscard]] inline bool spatial_can_refit(
    std::span<const SpatialLeafMetadata> previous,
    std::span<const SpatialLeafMetadata> staged) noexcept {
    if (previous.size() != staged.size()) return false;
    for (std::size_t index = 0U; index < previous.size(); ++index) {
        const bool previous_active =
            (previous[index].flags & spatial_leaf_inactive) == 0U;
        const bool staged_active =
            (staged[index].flags & spatial_leaf_inactive) == 0U;
        if (previous_active != staged_active) return false;
        if (previous_active &&
            (previous[index].identity != staged[index].identity ||
             previous[index].sort_key != staged[index].sort_key))
            return false;
    }
    return true;
}

struct SpatialTreeView {
    std::span<const std::uint32_t> left_children{};
    std::span<const std::uint32_t> right_children{};
    std::span<const std::uint32_t> leaf_sources{};
    std::span<const SpatialAabb> node_bounds{};
    std::span<const SpatialLeafMetadata> metadata{};
    std::span<const std::uint8_t> admitted{};
    std::uint32_t leaf_count{};
};

[[nodiscard]] inline PmResult spatial_query_aabb(
    const SpatialTreeView &tree, const SpatialAabb &query,
    std::uint32_t query_mask, std::uint32_t excluded_sort_key,
    std::span<std::uint32_t> traversal,
    std::span<std::uint32_t> output, std::uint32_t &written,
    std::uint32_t &visited) noexcept {
    written = 0U;
    visited = 0U;
    if (!spatial_finite(query)) return PM_ERROR_INVALID_ARGUMENT;
    if (tree.leaf_count == 0U) return PM_SUCCESS;
    const std::size_t node_count = 2ULL * tree.leaf_count - 1ULL;
    const std::size_t internal_count = tree.leaf_count - 1U;
    if (tree.left_children.size() < internal_count ||
        tree.right_children.size() < internal_count ||
        tree.leaf_sources.size() < tree.leaf_count ||
        tree.node_bounds.size() < node_count ||
        tree.metadata.size() != tree.admitted.size() || traversal.empty())
        return PM_ERROR_INVALID_ARGUMENT;
    std::size_t pending = 1U;
    traversal[0] = 0U;
    while (pending != 0U) {
        const auto node = traversal[--pending];
        ++visited;
        if (node >= node_count) {
            written = 0U;
            return PM_ERROR_INVALID_STATE;
        }
        if (!spatial_overlaps(tree.node_bounds[node], query)) continue;
        if (node >= internal_count) {
            const auto rank = node - internal_count;
            const auto source = tree.leaf_sources[rank];
            if (source >= tree.metadata.size()) {
                written = 0U;
                return PM_ERROR_INVALID_STATE;
            }
            const auto &leaf = tree.metadata[source];
            if (tree.admitted[source] == 0U ||
                (leaf.query_mask & query_mask) == 0U ||
                leaf.sort_key == excluded_sort_key) continue;
            if (written == output.size()) {
                written = 0U;
                return PM_ERROR_CAPACITY_EXCEEDED;
            }
            output[written++] = source;
            continue;
        }
        if (pending + 2U > traversal.size()) {
            written = 0U;
            return PM_ERROR_CAPACITY_EXCEEDED;
        }
        traversal[pending++] = tree.right_children[node];
        traversal[pending++] = tree.left_children[node];
    }
    std::sort(output.begin(), output.begin() + written,
        [&](std::uint32_t left, std::uint32_t right) {
            return tree.metadata[left].identity < tree.metadata[right].identity;
        });
    return PM_SUCCESS;
}

enum class SpatialUpdateKind : std::uint8_t { empty, build, refit };
enum class SpatialEpochPhase : std::uint8_t {
    free, staged, updated, published, current, retired
};

struct SpatialEpochToken {
    std::uint64_t tick{};
    std::uint64_t graph_revision{};
    std::uint64_t resource_revision{};
    std::uint32_t profile_version{};
    std::uint32_t slot{};
    std::uint32_t generation{};

    auto operator<=>(const SpatialEpochToken &) const = default;
};

struct SpatialEpochCounters {
    std::uint32_t builds{};
    std::uint32_t refits{};
    std::uint32_t publications{};
    std::uint32_t queries{};
    std::uint32_t overflows{};
    std::uint64_t candidates{};
};

class SpatialEpochLifecycle {
public:
    explicit constexpr SpatialEpochLifecycle(
        std::uint32_t leaf_capacity,
        std::uint32_t snapshot_count = spatial_service_max_snapshots) noexcept
        : leaf_capacity_(leaf_capacity),
          snapshot_count_(snapshot_count) {}

    [[nodiscard]] PmResult stage_boundary(
        std::uint64_t tick, std::uint32_t profile_version,
        std::uint64_t graph_revision, std::uint64_t resource_revision,
        std::uint32_t staged_leaf_count,
        bool refit_eligible = false) noexcept {
        if (snapshot_count_ < 2U ||
            snapshot_count_ > spatial_service_max_snapshots ||
            staging_ != spatial_invalid_index || tick == 0U ||
            profile_version == 0U ||
            graph_revision == 0U || resource_revision == 0U ||
            staged_leaf_count > leaf_capacity_ ||
            (committed_tick_ != 0U && tick != committed_tick_ + 1U))
            return PM_ERROR_INVALID_STATE;
        for (std::uint32_t index = 0U; index < snapshot_count_; ++index) {
            auto &slot = slots_[index];
            if (slot.phase != SpatialEpochPhase::free &&
                !(slot.phase == SpatialEpochPhase::retired &&
                  slot.references == 0U)) continue;
            if (generation_ledger_[index] ==
                std::numeric_limits<std::uint32_t>::max())
                return PM_ERROR_INVALID_STATE;
            slot = {};
            slot.generation = ++generation_ledger_[index];
            if (slot.generation == 0U) return PM_ERROR_INVALID_STATE;
            slot.phase = SpatialEpochPhase::staged;
            slot.leaf_count = staged_leaf_count;
            slot.refit_eligible = refit_eligible;
            slot.token = {tick, graph_revision, resource_revision,
                          profile_version, index, slot.generation};
            staging_ = index;
            return PM_SUCCESS;
        }
        return PM_ERROR_CAPACITY_EXCEEDED;
    }

    [[nodiscard]] PmResult record_update(SpatialUpdateKind kind) noexcept {
        auto *slot = staging_slot();
        if (slot == nullptr || (slot->phase != SpatialEpochPhase::staged &&
                               slot->phase != SpatialEpochPhase::updated))
            return PM_ERROR_INVALID_STATE;
        if (slot->updated) return PM_ERROR_INVALID_STATE;
        if ((slot->leaf_count == 0U) != (kind == SpatialUpdateKind::empty))
            return PM_ERROR_INVALID_ARGUMENT;
        if (kind == SpatialUpdateKind::refit && !slot->refit_eligible)
            return PM_ERROR_INVALID_STATE;
        slot->updated = true;
        slot->phase = SpatialEpochPhase::updated;
        if (kind == SpatialUpdateKind::build) ++slot->counters.builds;
        if (kind == SpatialUpdateKind::refit) ++slot->counters.refits;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult publish(SpatialEpochToken &token) noexcept {
        auto *slot = staging_slot();
        if (slot == nullptr || slot->phase != SpatialEpochPhase::updated ||
            !slot->updated || slot->failed ||
            slot->counters.builds + slot->counters.refits > 1U ||
            slot->counters.publications != 0U)
            return PM_ERROR_INVALID_STATE;
        ++slot->counters.publications;
        slot->phase = SpatialEpochPhase::published;
        token = slot->token;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult acquire(const SpatialEpochToken &token) noexcept {
        auto *slot = find(token);
        if (slot == nullptr || slot->failed ||
            (slot->phase != SpatialEpochPhase::published &&
                               slot->phase != SpatialEpochPhase::current))
            return PM_ERROR_INVALID_STATE;
        if (slot->references == std::numeric_limits<std::uint32_t>::max())
            return PM_ERROR_CAPACITY_EXCEEDED;
        ++slot->references;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult release(const SpatialEpochToken &token) noexcept {
        auto *slot = find(token);
        if (slot == nullptr || slot->references == 0U)
            return PM_ERROR_INVALID_STATE;
        --slot->references;
        if (slot->references == 0U && slot->phase == SpatialEpochPhase::retired)
            slot->phase = SpatialEpochPhase::free;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_query(
        const SpatialEpochToken &token, std::uint32_t candidates,
        bool overflow) noexcept {
        auto *slot = find(token);
        if (slot == nullptr || slot->failed ||
            (slot->phase != SpatialEpochPhase::published &&
                               slot->phase != SpatialEpochPhase::current &&
                               !(slot->phase == SpatialEpochPhase::retired &&
                                 slot->references != 0U)))
            return PM_ERROR_INVALID_STATE;
        if (slot->counters.queries ==
                std::numeric_limits<std::uint32_t>::max() ||
            candidates > std::numeric_limits<std::uint64_t>::max() -
                slot->counters.candidates ||
            (overflow && slot->counters.overflows ==
                std::numeric_limits<std::uint32_t>::max())) {
            if (slot->phase == SpatialEpochPhase::published)
                slot->failed = true;
            return PM_ERROR_CAPACITY_EXCEEDED;
        }
        ++slot->counters.queries;
        slot->counters.candidates += candidates;
        if (overflow) {
            ++slot->counters.overflows;
            if (slot->phase == SpatialEpochPhase::published)
                slot->failed = true;
            return PM_ERROR_CAPACITY_EXCEEDED;
        }
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult validate_epoch(
        const SpatialEpochToken &token, std::uint64_t graph_revision,
        std::uint64_t resource_revision) const noexcept {
        const auto *slot = find(token);
        if (slot == nullptr || slot->failed ||
            (slot->phase != SpatialEpochPhase::published &&
             slot->phase != SpatialEpochPhase::current &&
             !(slot->phase == SpatialEpochPhase::retired &&
               slot->references != 0U)) ||
            token.graph_revision != graph_revision ||
            token.resource_revision != resource_revision)
            return PM_ERROR_INVALID_STATE;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult validate_envelopes(
        const SpatialEpochToken &token,
        std::span<const SpatialAabb> envelopes,
        std::span<const SpatialAabb> current) noexcept {
        auto *slot = find(token);
        if (slot == nullptr) return PM_ERROR_INVALID_STATE;
        if (slot->failed || slot->phase != SpatialEpochPhase::published ||
            staging_ != token.slot) return PM_ERROR_INVALID_STATE;
        if (current.size() != slot->leaf_count ||
            envelopes.size() != slot->leaf_count)
            return PM_ERROR_INVALID_ARGUMENT;
        for (std::size_t index = 0U; index < current.size(); ++index)
            if (!spatial_finite(current[index]) ||
                !spatial_contains(envelopes[index], current[index])) {
                slot->failed = true;
                return PM_ERROR_INVALID_STATE;
            }
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult commit(const SpatialEpochToken &token) noexcept {
        auto *slot = find(token);
        if (slot == nullptr || staging_ != token.slot || slot->failed ||
            slot->phase != SpatialEpochPhase::published ||
            slot->counters.publications != 1U ||
            slot->counters.builds + slot->counters.refits > 1U)
            return PM_ERROR_INVALID_STATE;
        if (current_ != spatial_invalid_index) {
            auto &previous = slots_[current_];
            previous.phase = previous.references == 0U
                ? SpatialEpochPhase::free : SpatialEpochPhase::retired;
        }
        slot->phase = SpatialEpochPhase::current;
        current_ = staging_;
        staging_ = spatial_invalid_index;
        committed_tick_ = token.tick;
        return PM_SUCCESS;
    }

    void abort() noexcept {
        if (staging_ == spatial_invalid_index) return;
        auto &slot = slots_[staging_];
        slot.phase = slot.references == 0U
            ? SpatialEpochPhase::free : SpatialEpochPhase::retired;
        staging_ = spatial_invalid_index;
    }

    void reset() noexcept {
        abort();
        if (current_ != spatial_invalid_index) {
            auto &slot = slots_[current_];
            slot.phase = slot.references == 0U
                ? SpatialEpochPhase::free : SpatialEpochPhase::retired;
            current_ = spatial_invalid_index;
        }
        committed_tick_ = 0U;
    }

    [[nodiscard]] const SpatialEpochCounters *counters(
        const SpatialEpochToken &token) const noexcept {
        const auto *slot = find(token);
        return slot == nullptr || slot->phase == SpatialEpochPhase::free
            ? nullptr : &slot->counters;
    }

private:
    struct Slot {
        SpatialEpochToken token{};
        SpatialEpochCounters counters{};
        SpatialEpochPhase phase{SpatialEpochPhase::free};
        std::uint32_t generation{};
        std::uint32_t leaf_count{};
        std::uint32_t references{};
        bool updated{};
        bool failed{};
        bool refit_eligible{};
    };

    [[nodiscard]] Slot *staging_slot() noexcept {
        return staging_ == spatial_invalid_index ? nullptr : &slots_[staging_];
    }
    [[nodiscard]] Slot *find(const SpatialEpochToken &token) noexcept {
        if (token.slot >= snapshot_count_) return nullptr;
        auto &slot = slots_[token.slot];
        return slot.generation == token.generation && slot.token == token
            ? &slot : nullptr;
    }
    [[nodiscard]] const Slot *find(const SpatialEpochToken &token) const noexcept {
        if (token.slot >= snapshot_count_) return nullptr;
        const auto &slot = slots_[token.slot];
        return slot.generation == token.generation && slot.token == token
            ? &slot : nullptr;
    }

    std::uint32_t leaf_capacity_{};
    std::uint32_t snapshot_count_{};
    std::uint32_t staging_{spatial_invalid_index};
    std::uint32_t current_{spatial_invalid_index};
    std::uint64_t committed_tick_{};
    std::array<std::uint32_t, spatial_service_max_snapshots>
        generation_ledger_{};
    std::array<Slot, spatial_service_max_snapshots> slots_{};
};

} // namespace parallel_mater::simulation
