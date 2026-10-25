// SPDX-License-Identifier: MIT
#pragma once

#include "spatial_service.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace parallel_mater::simulation {

struct alignas(16) SpatialStorageBlock {
    std::byte bytes[16];
};

[[nodiscard]] constexpr std::size_t spatial_storage_block_count(
    std::size_t bytes) noexcept {
    return bytes / sizeof(SpatialStorageBlock) +
        (bytes % sizeof(SpatialStorageBlock) != 0U);
}

struct ProductionSpatialSnapshot {
    SpatialEpochToken token{};
    SpatialEpochHeader *header{};
    SpatialLeafMetadata *metadata{};
    SpatialAabb *bounds{};
    std::uint8_t *build_flags{};
    std::uint8_t *admission{};
    std::uint32_t *query_counts{};
    std::uint32_t *query_candidates{};
    std::uint32_t *query_visited_nodes{};
    std::uint32_t *query_counter{};
    std::uint32_t *query_status{};
    std::byte *build_output{};
    std::byte *build_scratch{};
    std::uint32_t leaf_capacity{};
    std::uint32_t query_capacity{};
    std::uint32_t candidates_per_query{};

    [[nodiscard]] explicit operator bool() const noexcept {
        return header != nullptr;
    }
};

// Allocation-free production owner for the immutable spatial snapshots used by
// the realtime schedule. The CPU adapter executes this path directly over its
// native device allocation. GPU adapters retain the same resource/lifecycle
// seam and bind their generated dispatches in the native-schedule task.
class ProductionSpatialRuntime {
public:
    [[nodiscard]] PmResult configure(
        const SpatialServicePlan &plan, void *persistent,
        void *scratch, bool host_accessible) noexcept {
        if (lifecycle_.has_value() || plan.snapshot_count < 2U ||
            plan.snapshot_count > spatial_service_max_snapshots ||
            (plan.persistent_bytes != 0U && persistent == nullptr) ||
            (plan.scratch_bytes != 0U && scratch == nullptr) ||
            reinterpret_cast<std::uintptr_t>(persistent) % 16U != 0U ||
            reinterpret_cast<std::uintptr_t>(scratch) % 16U != 0U)
            return PM_ERROR_INVALID_ARGUMENT;
        plan_ = plan;
        persistent_ = static_cast<std::byte *>(persistent);
        scratch_ = static_cast<std::byte *>(scratch);
        host_accessible_ = host_accessible;
        lifecycle_.emplace(plan.leaf_capacity, plan.snapshot_count);
        return PM_SUCCESS;
    }

    [[nodiscard]] bool configured() const noexcept {
        return lifecycle_.has_value();
    }

    [[nodiscard]] PmResult begin_tick(
        std::uint64_t tick, std::uint32_t profile_version,
        std::uint64_t graph_revision, std::uint64_t resource_revision,
        std::uint32_t staged_leaf_slots,
        bool refit_eligible = false) noexcept {
        if (!host_accessible_) return PM_ERROR_FEATURE_NOT_PRESENT;
        if (!lifecycle_ || staging_.has_value())
            return PM_ERROR_INVALID_STATE;
        if (staged_leaf_slots > plan_.leaf_capacity)
            return PM_ERROR_CAPACITY_EXCEEDED;
        const auto result = lifecycle_->stage_boundary(
            tick, profile_version, graph_revision, resource_revision,
            staged_leaf_slots, refit_eligible);
        if (result != PM_SUCCESS) return result;
        const auto *token = lifecycle_->staging_token();
        if (token == nullptr) {
            lifecycle_->abort();
            return PM_ERROR_INVALID_STATE;
        }
        staging_ = *token;
        staged_leaf_slots_ = staged_leaf_slots;
        slot_leaf_slots_[token->slot] = staged_leaf_slots;
        active_leaf_count_ = 0U;
        update_kind_.reset();
        pinned_ = false;

        auto snapshot = snapshot_for(*token);
        if (!snapshot) {
            lifecycle_->abort();
            staging_.reset();
            return PM_ERROR_INVALID_STATE;
        }
        *snapshot.header = {tick, graph_revision, resource_revision,
            profile_version, token->generation};
        *snapshot.query_counter = 0U;
        *snapshot.query_status = 0U;
        if (staged_leaf_slots != 0U) {
            std::fill_n(snapshot.metadata, staged_leaf_slots,
                        SpatialLeafMetadata{});
            std::fill_n(snapshot.bounds, staged_leaf_slots, SpatialAabb{});
            std::fill_n(snapshot.build_flags, staged_leaf_slots,
                        std::uint8_t{});
            std::fill_n(snapshot.admission, staged_leaf_slots,
                        std::uint8_t{});
            for (std::uint32_t index = 0U;
                 index < staged_leaf_slots; ++index)
                snapshot.metadata[index].flags = spatial_leaf_inactive;
        }
        if (plan_.query_capacity != 0U) {
            std::fill_n(snapshot.query_counts, plan_.query_capacity, 0U);
            std::fill_n(snapshot.query_visited_nodes,
                        plan_.query_capacity, 0U);
        }
        return PM_SUCCESS;
    }

    // Device adapters own initialization and build dispatches.  This records
    // the identical host lifecycle without touching device-only storage.
    [[nodiscard]] PmResult begin_device_tick(
        std::uint64_t tick, std::uint32_t profile_version,
        std::uint64_t graph_revision, std::uint64_t resource_revision,
        std::uint32_t staged_leaf_slots,
        bool refit_eligible = false) noexcept {
        if (host_accessible_) return PM_ERROR_INVALID_STATE;
        if (!lifecycle_ || staging_.has_value())
            return PM_ERROR_INVALID_STATE;
        if (staged_leaf_slots > plan_.leaf_capacity)
            return PM_ERROR_CAPACITY_EXCEEDED;
        const auto result = lifecycle_->stage_boundary(
            tick, profile_version, graph_revision, resource_revision,
            staged_leaf_slots, refit_eligible);
        if (result != PM_SUCCESS) return result;
        const auto *token = lifecycle_->staging_token();
        if (token == nullptr) {
            lifecycle_->abort();
            return PM_ERROR_INVALID_STATE;
        }
        staging_ = *token;
        staged_leaf_slots_ = staged_leaf_slots;
        slot_leaf_slots_[token->slot] = staged_leaf_slots;
        active_leaf_count_ = staged_leaf_slots;
        update_kind_.reset();
        pinned_ = false;
        return PM_SUCCESS;
    }

    [[nodiscard]] ProductionSpatialSnapshot staging_snapshot() noexcept {
        return staging_.has_value()
            ? snapshot_for(*staging_) : ProductionSpatialSnapshot{};
    }

    [[nodiscard]] ProductionSpatialSnapshot published_snapshot(
        const SpatialEpochToken &token) noexcept {
        if (!lifecycle_ || lifecycle_->validate_epoch(
                token, token.graph_revision,
                token.resource_revision) != PM_SUCCESS)
            return {};
        return snapshot_for(token);
    }

    [[nodiscard]] PmResult update(SpatialUpdateKind kind) noexcept {
        if (!host_accessible_) return PM_ERROR_FEATURE_NOT_PRESENT;
        if (!lifecycle_ || !staging_ || update_kind_.has_value())
            return PM_ERROR_INVALID_STATE;
        auto snapshot = snapshot_for(*staging_);
        if (!snapshot) return PM_ERROR_INVALID_STATE;
        if ((staged_leaf_slots_ == 0U) !=
            (kind == SpatialUpdateKind::empty))
            return fail_update(snapshot, spatial_status_policy_mismatch,
                               PM_ERROR_INVALID_ARGUMENT);

        PmResult result = PM_SUCCESS;
        if (kind == SpatialUpdateKind::empty) {
            active_leaf_count_ = 0U;
        } else {
            result = prepare_staged_snapshot(snapshot);
            if (result == PM_SUCCESS) {
                result = kind == SpatialUpdateKind::refit
                    ? refit_snapshot(snapshot) : build_snapshot(snapshot);
            }
        }
        if (result != PM_SUCCESS) return result;
        result = lifecycle_->record_update(kind);
        if (result != PM_SUCCESS) return result;
        update_kind_ = kind;
        snapshot.header->leaf_count = active_leaf_count_;
        snapshot.header->node_count = active_leaf_count_ == 0U
            ? 0U : 2U * active_leaf_count_ - 1U;
        snapshot.header->build_count =
            kind == SpatialUpdateKind::build ? 1U : 0U;
        snapshot.header->refit_count =
            kind == SpatialUpdateKind::refit ? 1U : 0U;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_device_update(
        SpatialUpdateKind kind) noexcept {
        if (host_accessible_ || !lifecycle_ || !staging_ ||
            update_kind_.has_value() ||
            (staged_leaf_slots_ == 0U) != (kind == SpatialUpdateKind::empty))
            return PM_ERROR_INVALID_STATE;
        const auto result = lifecycle_->record_update(kind);
        if (result != PM_SUCCESS) return result;
        update_kind_ = kind;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult publish(SpatialEpochToken &token) noexcept {
        if (!host_accessible_) return PM_ERROR_FEATURE_NOT_PRESENT;
        if (!lifecycle_ || !staging_ || !update_kind_.has_value() || pinned_)
            return PM_ERROR_INVALID_STATE;
        auto snapshot = snapshot_for(*staging_);
        if (!snapshot || snapshot.header->status != 0U)
            return PM_ERROR_INVALID_STATE;
        auto result = lifecycle_->publish(token);
        if (result != PM_SUCCESS) return result;
        if (token != *staging_) return PM_ERROR_INVALID_STATE;
        result = lifecycle_->acquire(token);
        if (result != PM_SUCCESS) {
            lifecycle_->abort();
            staging_.reset();
            return result;
        }
        pinned_ = true;
        snapshot.header->publication_count = 1U;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult publish_device(
        SpatialEpochToken &token) noexcept {
        if (host_accessible_ || !lifecycle_ || !staging_ ||
            !update_kind_.has_value() || pinned_)
            return PM_ERROR_INVALID_STATE;
        auto result = lifecycle_->publish(token);
        if (result != PM_SUCCESS) return result;
        if (token != *staging_) return PM_ERROR_INVALID_STATE;
        result = lifecycle_->acquire(token);
        if (result != PM_SUCCESS) {
            lifecycle_->abort();
            staging_.reset();
            return result;
        }
        pinned_ = true;
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult record_device_queries(
        const SpatialEpochToken &token, std::uint32_t count) noexcept {
        if (host_accessible_ || !lifecycle_ || !staging_ ||
            token != *staging_ || !pinned_)
            return PM_ERROR_INVALID_STATE;
        for (std::uint32_t index = 0U; index < count; ++index) {
            const auto result = lifecycle_->record_query(token, 0U, false);
            if (result != PM_SUCCESS) return result;
        }
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult query(
        const SpatialEpochToken &token, std::uint32_t query_index,
        const SpatialAabb &bounds, std::uint32_t query_mask,
        std::uint32_t excluded_sort_key = spatial_invalid_index) noexcept {
        if (!host_accessible_) return PM_ERROR_FEATURE_NOT_PRESENT;
        if (!lifecycle_ || query_index >= plan_.query_capacity)
            return PM_ERROR_INVALID_ARGUMENT;
        auto snapshot = snapshot_for(token);
        if (!snapshot ||
            lifecycle_->validate_epoch(token, token.graph_revision,
                                       token.resource_revision) != PM_SUCCESS)
            return PM_ERROR_INVALID_STATE;
        const auto tree = tree_for(snapshot);
        std::array<std::uint32_t, 64U> traversal{};
        std::uint32_t written{};
        std::uint32_t visited{};
        const std::size_t candidate_offset =
            std::size_t(query_index) * plan_.candidates_per_query;
        const auto result = spatial_query_aabb(
            tree, bounds, query_mask, excluded_sort_key, traversal,
            std::span<std::uint32_t>(
                snapshot.query_candidates + candidate_offset,
                plan_.candidates_per_query),
            written, visited);
        snapshot.query_counts[query_index] =
            result == PM_SUCCESS ? written : 0U;
        snapshot.query_visited_nodes[query_index] = visited;
        ++*snapshot.query_counter;
        const bool overflow = result == PM_ERROR_CAPACITY_EXCEEDED;
        if (overflow)
            *snapshot.query_status |= spatial_epoch_status_query_overflow;
        const auto recorded = lifecycle_->record_query(
            token, result == PM_SUCCESS ? written : 0U, overflow);
        if (result != PM_SUCCESS && !overflow) {
            *snapshot.query_status |= spatial_status_invalid_bounds;
            (void)lifecycle_->mark_failed(token);
        }
        return result != PM_SUCCESS ? result : recorded;
    }

    [[nodiscard]] PmResult validate(
        const SpatialEpochToken &token,
        std::span<const SpatialLeafMetadata> current_metadata,
        std::span<const SpatialAabb> current_bounds,
        std::uint64_t graph_revision,
        std::uint64_t resource_revision) noexcept {
        if (!host_accessible_) return PM_ERROR_FEATURE_NOT_PRESENT;
        if (!lifecycle_ || !staging_ || token != *staging_ ||
            current_metadata.size() != staged_leaf_slots_ ||
            current_bounds.size() != staged_leaf_slots_)
            return PM_ERROR_INVALID_ARGUMENT;
        auto snapshot = snapshot_for(token);
        if (!snapshot) return PM_ERROR_INVALID_STATE;
        if (lifecycle_->validate_epoch(
                token, graph_revision, resource_revision) != PM_SUCCESS) {
            *snapshot.query_status |= spatial_epoch_status_stale_epoch;
            (void)lifecycle_->mark_failed(token);
            return PM_ERROR_INVALID_STATE;
        }
        for (std::size_t index = 0U; index < current_metadata.size(); ++index) {
            const bool staged_active =
                (snapshot.metadata[index].flags & spatial_leaf_inactive) == 0U;
            const bool current_active =
                (current_metadata[index].flags & spatial_leaf_inactive) == 0U;
            if (staged_active != current_active ||
                (staged_active && snapshot.metadata[index].identity !=
                                      current_metadata[index].identity)) {
                *snapshot.query_status |= spatial_epoch_status_stale_epoch;
                (void)lifecycle_->mark_failed(token);
                return PM_ERROR_INVALID_STATE;
            }
        }
        const auto result = lifecycle_->validate_envelopes(
            token,
            {snapshot.bounds, staged_leaf_slots_}, current_bounds);
        if (result != PM_SUCCESS)
            *snapshot.query_status |= spatial_epoch_status_envelope_escape;
        return result;
    }

    [[nodiscard]] PmResult commit(const SpatialEpochToken &token) noexcept {
        if (!lifecycle_ || !staging_ || token != *staging_ || !pinned_)
            return PM_ERROR_INVALID_STATE;
        auto result = lifecycle_->commit(token);
        if (result != PM_SUCCESS) return result;
        result = lifecycle_->release(token);
        if (result != PM_SUCCESS) return result;
        current_ = token;
        staging_.reset();
        update_kind_.reset();
        pinned_ = false;
        return PM_SUCCESS;
    }

    void abort() noexcept {
        if (!lifecycle_ || !staging_) return;
        if (pinned_) (void)lifecycle_->release(*staging_);
        lifecycle_->abort();
        staging_.reset();
        update_kind_.reset();
        pinned_ = false;
    }

    void reset() noexcept {
        if (!lifecycle_) return;
        abort();
        lifecycle_->reset();
        current_.reset();
    }

    [[nodiscard]] PmResult acquire(
        const SpatialEpochToken &token) noexcept {
        return lifecycle_ ? lifecycle_->acquire(token)
                          : PM_ERROR_INVALID_STATE;
    }

    [[nodiscard]] PmResult release(
        const SpatialEpochToken &token) noexcept {
        return lifecycle_ ? lifecycle_->release(token)
                          : PM_ERROR_INVALID_STATE;
    }

    [[nodiscard]] const SpatialEpochCounters *counters(
        const SpatialEpochToken &token) const noexcept {
        return lifecycle_ ? lifecycle_->counters(token) : nullptr;
    }

    [[nodiscard]] const std::optional<SpatialEpochToken> &current_token()
        const noexcept {
        return current_;
    }

private:
    struct TreeBuffers {
        std::uint32_t *left{};
        std::uint32_t *right{};
        std::uint32_t *parents{};
        std::uint32_t *leaf_sources{};
        SpatialAabb *node_bounds{};
    };

    template <typename T>
    [[nodiscard]] static T *range(
        std::byte *base, SpatialBufferRange value) noexcept {
        return reinterpret_cast<T *>(base + value.offset);
    }

    [[nodiscard]] ProductionSpatialSnapshot snapshot_for(
        const SpatialEpochToken &token) noexcept {
        if (token.slot >= plan_.snapshot_count || persistent_ == nullptr)
            return {};
        const auto &layout = plan_.snapshots[token.slot];
        return {token,
            range<SpatialEpochHeader>(persistent_, layout.header),
            range<SpatialLeafMetadata>(persistent_, layout.metadata),
            range<SpatialAabb>(persistent_, layout.bounds),
            range<std::uint8_t>(persistent_, layout.build_flags),
            range<std::uint8_t>(persistent_, layout.admission),
            range<std::uint32_t>(persistent_, layout.query_counts),
            range<std::uint32_t>(persistent_, layout.query_candidates),
            range<std::uint32_t>(persistent_, layout.query_visited_nodes),
            range<std::uint32_t>(persistent_, layout.query_counter),
            range<std::uint32_t>(persistent_, layout.query_status),
            persistent_ + layout.build_output.offset,
            scratch_ + plan_.build_scratch.offset,
            plan_.leaf_capacity, plan_.query_capacity,
            plan_.candidates_per_query};
    }

    [[nodiscard]] TreeBuffers buffers_for(
        const ProductionSpatialSnapshot &snapshot) noexcept {
        const auto &layout = plan_.snapshots[snapshot.token.slot];
        auto *base = persistent_ + layout.build_output.offset;
        return {range<std::uint32_t>(base, plan_.build.left_children),
            range<std::uint32_t>(base, plan_.build.right_children),
            range<std::uint32_t>(base, plan_.build.parents),
            range<std::uint32_t>(base, plan_.build.leaf_sources),
            range<SpatialAabb>(base, plan_.build.node_bounds)};
    }

    [[nodiscard]] SpatialTreeView tree_for(
        ProductionSpatialSnapshot &snapshot) noexcept {
        const auto tree = buffers_for(snapshot);
        const auto leaf_count = snapshot.header->leaf_count;
        const auto internal = leaf_count == 0U ? 0U : leaf_count - 1U;
        const auto nodes = leaf_count == 0U ? 0U : 2U * leaf_count - 1U;
        return {{tree.left, internal}, {tree.right, internal},
            {tree.leaf_sources, leaf_count},
            {tree.node_bounds, nodes},
            {snapshot.metadata, slot_leaf_slots_[snapshot.token.slot]},
            {snapshot.admission, slot_leaf_slots_[snapshot.token.slot]},
            leaf_count};
    }

    [[nodiscard]] PmResult fail_update(
        ProductionSpatialSnapshot &snapshot, std::uint32_t status,
        PmResult result) noexcept {
        snapshot.header->status |= status;
        return result;
    }

    [[nodiscard]] PmResult prepare_staged_snapshot(
        ProductionSpatialSnapshot &snapshot) noexcept {
        if (scratch_ == nullptr) return PM_ERROR_INVALID_STATE;
        auto *active = range<std::uint32_t>(
            scratch_, plan_.build.active_indices);
        active_leaf_count_ = 0U;
        for (std::uint32_t index = 0U;
             index < staged_leaf_slots_; ++index) {
            const bool inactive =
                (snapshot.metadata[index].flags & spatial_leaf_inactive) != 0U;
            const bool flagged = snapshot.build_flags[index] != 0U;
            if (inactive == flagged)
                return fail_update(snapshot,
                    spatial_epoch_status_invalid_identity,
                    PM_ERROR_INVALID_ARGUMENT);
            if (inactive) continue;
            if (!spatial_identity_valid(snapshot.metadata[index].identity) ||
                snapshot.metadata[index].sort_key == 0U ||
                !spatial_finite(snapshot.bounds[index]))
                return fail_update(snapshot,
                    spatial_epoch_status_invalid_identity,
                    PM_ERROR_INVALID_ARGUMENT);
            active[active_leaf_count_++] = index;
        }
        auto *active_count = range<std::uint32_t>(
            scratch_, plan_.build.active_count);
        *active_count = active_leaf_count_;
        const auto admission = spatial_resolve_admission(
            {snapshot.metadata, staged_leaf_slots_},
            {snapshot.bounds, staged_leaf_slots_},
            {snapshot.admission, staged_leaf_slots_});
        if (admission != PM_SUCCESS)
            return fail_update(snapshot,
                spatial_epoch_status_admission_invalid, admission);
        return PM_SUCCESS;
    }

    [[nodiscard]] PmResult build_snapshot(
        ProductionSpatialSnapshot &snapshot) noexcept {
        if (active_leaf_count_ == 0U) return PM_SUCCESS;
        auto *active = range<std::uint32_t>(
            scratch_, plan_.build.active_indices);
        std::sort(active, active + active_leaf_count_,
            [&](std::uint32_t left, std::uint32_t right) {
                const auto &left_leaf = snapshot.metadata[left];
                const auto &right_leaf = snapshot.metadata[right];
                return left_leaf.sort_key != right_leaf.sort_key
                    ? left_leaf.sort_key < right_leaf.sort_key
                    : left_leaf.identity < right_leaf.identity;
            });
        for (std::uint32_t index = 1U; index < active_leaf_count_; ++index)
            if (snapshot.metadata[active[index - 1U]].sort_key ==
                snapshot.metadata[active[index]].sort_key)
                return fail_update(snapshot, spatial_status_duplicate_key,
                                   PM_ERROR_INVALID_ARGUMENT);

        const auto tree = buffers_for(snapshot);
        const auto internal_count = active_leaf_count_ - 1U;
        std::uint32_t next_internal{};
        const auto build_node = [&](auto &&self, std::uint32_t begin,
                                    std::uint32_t end) -> std::uint32_t {
            if (end - begin == 1U) {
                const auto node = internal_count + begin;
                const auto source = active[begin];
                tree.leaf_sources[begin] = source;
                tree.node_bounds[node] = snapshot.bounds[source];
                return node;
            }
            const auto node = next_internal++;
            const auto middle = begin + (end - begin) / 2U;
            const auto left = self(self, begin, middle);
            const auto right = self(self, middle, end);
            tree.left[node] = left;
            tree.right[node] = right;
            tree.parents[left] = node;
            tree.parents[right] = node;
            tree.node_bounds[node] = spatial_union(
                tree.node_bounds[left], tree.node_bounds[right]);
            return node;
        };
        const auto root = build_node(
            build_node, 0U, active_leaf_count_);
        tree.parents[root] = spatial_invalid_index;
        return root == 0U && next_internal == internal_count
            ? PM_SUCCESS : fail_update(snapshot,
                spatial_status_policy_mismatch, PM_ERROR_INVALID_STATE);
    }

    [[nodiscard]] PmResult refit_snapshot(
        ProductionSpatialSnapshot &snapshot) noexcept {
        if (!current_) return PM_ERROR_INVALID_STATE;
        auto previous = snapshot_for(*current_);
        const auto previous_leaf_slots = slot_leaf_slots_[current_->slot];
        if (!previous || staged_leaf_slots_ != previous_leaf_slots ||
            !spatial_can_refit(
                {previous.metadata, previous_leaf_slots},
                {snapshot.metadata, staged_leaf_slots_}))
            return fail_update(snapshot,
                spatial_epoch_status_stale_epoch,
                PM_ERROR_INVALID_STATE);
        if (active_leaf_count_ == 0U) return PM_SUCCESS;
        const auto source_tree = buffers_for(previous);
        const auto target_tree = buffers_for(snapshot);
        const auto internal_count = active_leaf_count_ - 1U;
        const auto node_count = 2U * active_leaf_count_ - 1U;
        std::copy_n(source_tree.left, internal_count, target_tree.left);
        std::copy_n(source_tree.right, internal_count, target_tree.right);
        std::copy_n(source_tree.parents, node_count, target_tree.parents);
        std::copy_n(source_tree.leaf_sources, active_leaf_count_,
                    target_tree.leaf_sources);
        const auto refit_node = [&](auto &&self,
                                    std::uint32_t node) -> SpatialAabb {
            if (node >= internal_count) {
                const auto rank = node - internal_count;
                const auto source = target_tree.leaf_sources[rank];
                return target_tree.node_bounds[node] = snapshot.bounds[source];
            }
            const auto left = self(self, target_tree.left[node]);
            const auto right = self(self, target_tree.right[node]);
            return target_tree.node_bounds[node] = spatial_union(left, right);
        };
        (void)refit_node(refit_node, 0U);
        return PM_SUCCESS;
    }

    SpatialServicePlan plan_{};
    std::byte *persistent_{};
    std::byte *scratch_{};
    bool host_accessible_{};
    std::optional<SpatialEpochLifecycle> lifecycle_{};
    std::optional<SpatialEpochToken> staging_{};
    std::optional<SpatialEpochToken> current_{};
    std::optional<SpatialUpdateKind> update_kind_{};
    std::uint32_t staged_leaf_slots_{};
    std::array<std::uint32_t, spatial_service_max_snapshots>
        slot_leaf_slots_{};
    std::uint32_t active_leaf_count_{};
    bool pinned_{};
};

} // namespace parallel_mater::simulation
