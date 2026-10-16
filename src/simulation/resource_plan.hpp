// SPDX-License-Identifier: MIT
#pragma once
#include "memory_arena.hpp"
#include "geometry_storage.hpp"
#include "primitives.hpp"
#include "status.hpp"
#include "types.hpp"
#include <algorithm>
#include <limits>

namespace parallel_mater::simulation {
inline constexpr std::uint32_t k_fluid_body_buckets = 4096U;
struct StorageRequirements { std::size_t persistent{}, scratch{}; };
struct StoragePlanner {
    static constexpr bool materialize = false;
    StorageRequirements requirements{};
    template<class T> Status operator()(T *&, std::size_t count, MemoryClass kind) noexcept {
        if (!count) return {};
        auto &size = kind == MemoryClass::persistent ? requirements.persistent : requirements.scratch;
        constexpr auto maximum = std::numeric_limits<std::size_t>::max();
        if (count > (maximum - (alignof(T) - 1)) / sizeof(T))
            return {StatusCode::out_of_memory, 0, "resource plan size overflow"};
        const auto bytes = count * sizeof(T) + alignof(T) - 1;
        if (bytes > maximum - size)
            return {StatusCode::out_of_memory, 0, "resource plan total overflow"};
        size += bytes; // Worst-case padding, independent of allocation order.
        return {};
    }
};

// One traversal drives both requirements and allocation. State contains typed
// pointers, so element sizes and alignments cannot drift from runtime storage.
template<class State, class Visitor>
Status visit_global_storage(State &state, SimulationOptions options, Visitor &visit) noexcept {
    const std::size_t rigid = options.rigid_body_capacity;
    if (!rigid || rigid > std::numeric_limits<std::size_t>::max() / rigid)
        return {StatusCode::invalid_argument, 0, "invalid rigid storage capacity"};
    const auto pairs = rigid * rigid;
    if (pairs > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return {StatusCode::invalid_argument, 0, "rigid body capacity exceeds broad-phase range"};
    const auto manifolds = rigid * (rigid - 1) / 2;
    const auto slots = std::min(manifolds, std::max<std::size_t>(
        k_minimum_leaf_pair_cache_slots, rigid * k_leaf_pair_cache_slots_per_body));
    const auto leaves = rigid <= k_small_rigid_leaf_body_capacity
        ? k_small_rigid_leaf_pair_capacity : k_max_leaf_pairs_per_body_pair;
    if (slots > std::numeric_limits<std::size_t>::max() / leaves ||
        slots > std::numeric_limits<std::uint32_t>::max())
        return {StatusCode::invalid_argument, 0, "leaf-pair storage capacity overflow"};
    state.rigid_leaf_pair_slot_capacity = static_cast<std::uint32_t>(slots);
    state.rigid_leaf_pair_capacity = slots * leaves;
    state.rigid_leaf_pairs_per_slot = leaves;
    state.rigid_contact_capacity = options.contact_capacity;
    state.rigid_broad_phase_workspace_size = compaction_scratch_bytes(static_cast<std::uint32_t>(pairs));
#define PM_STORAGE(member, count, kind) do { \
    const auto result = visit(state.member, count, MemoryClass::kind); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(paint_fields, options.paint_field_capacity, persistent);
    if constexpr (Visitor::materialize)
        for (std::uint32_t i = 0; i < options.paint_field_capacity; ++i) state.paint_fields[i] = {};
    PM_STORAGE(paint_rules, options.paint_rule_capacity, persistent);
    PM_STORAGE(fluid_neighbor_overflow, 1, persistent);
    PM_STORAGE(fluid_maximum_neighbor_count, 1, persistent);
    PM_STORAGE(parameters, rigid, persistent);
    PM_STORAGE(accumulators, rigid, persistent);
    PM_STORAGE(targets, rigid, persistent);
    PM_STORAGE(rigid_constraints, options.rigid_constraint_capacity, persistent);
    PM_STORAGE(ids, rigid, persistent);
    PM_STORAGE(states[0], rigid, persistent);
    PM_STORAGE(states[1], rigid, persistent);
    PM_STORAGE(render_previous_states, rigid, persistent);
    if (options.physics_debug.frame_capacity) {
        PM_STORAGE(debug_applied_forces, rigid, persistent);
        PM_STORAGE(debug_applied_torques, rigid, persistent);
        PM_STORAGE(debug_rigid_samples, rigid, persistent);
    }
    PM_STORAGE(fluid_previous_states, rigid, persistent);
    PM_STORAGE(rigid_manifolds, manifolds, persistent);
    PM_STORAGE(avbd_bodies, rigid, persistent);
    PM_STORAGE(rigid_contact_cache, slots, persistent);
    PM_STORAGE(rigid_contact_cache_slots, pairs, persistent);
    PM_STORAGE(rigid_contact_schedule, 1, persistent);
    PM_STORAGE(rigid_contact_event_offsets, manifolds, persistent);
    PM_STORAGE(rigid_world_bounds, rigid, persistent);
    PM_STORAGE(hit_box_rigid_flags, rigid, persistent);
    PM_STORAGE(fluid_body_bounds, rigid, persistent);
    if (options.rigid_constraint_capacity) PM_STORAGE(fixed_contact_projection, rigid, persistent);
    const auto fluid_body_words = (rigid + 63) / 64;
    PM_STORAGE(fluid_body_masks, k_fluid_body_buckets * fluid_body_words, persistent);
    PM_STORAGE(fluid_global_body_masks, fluid_body_words, persistent);
    PM_STORAGE(fluid_body_contact_flags, rigid, persistent);
    PM_STORAGE(fluid_contact_events, options.contact_capacity, persistent);
    PM_STORAGE(fluid_contact_count, 1, persistent);
    PM_STORAGE(fluid_contact_overflow, 1, persistent);
    PM_STORAGE(rigid_active_pair_flags, pairs, persistent);
    PM_STORAGE(rigid_active_pairs, pairs, persistent);
    PM_STORAGE(rigid_active_pair_count, 1, persistent);
    PM_STORAGE(rigid_broad_phase_workspace, state.rigid_broad_phase_workspace_size, scratch);
    PM_STORAGE(rigid_leaf_pairs, state.rigid_leaf_pair_capacity, persistent);
    if (rigid <= k_small_rigid_leaf_body_capacity)
        PM_STORAGE(rigid_leaf_manifolds, state.rigid_leaf_pair_capacity, persistent);
    PM_STORAGE(rigid_leaf_pair_counts, pairs, persistent);
    PM_STORAGE(rigid_contact_events, options.contact_capacity, persistent);
    PM_STORAGE(rigid_contact_count, 1, persistent);
    PM_STORAGE(meshes, options.triangle_mesh_capacity, persistent);
    if constexpr (Visitor::materialize)
        for (std::uint32_t i = 0; i < options.triangle_mesh_capacity; ++i) state.meshes[i] = {};
#undef PM_STORAGE
    return {};
}

template<class State, class Visitor>
Status visit_fluid_storage(State &state, std::uint32_t capacity, Visitor &visit) noexcept {
    if (!capacity || capacity > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        return {StatusCode::invalid_argument, 0, "fluid capacity exceeds primitive range"};
#define PM_STORAGE(member, count, kind) do { \
    const auto result = visit(state.member, count, MemoryClass::kind); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(count, 1, persistent);
    PM_STORAGE(positions, capacity, persistent);
    PM_STORAGE(velocities, capacity, persistent);
    PM_STORAGE(previous, capacity, persistent);
    PM_STORAGE(ids, capacity, persistent);
    PM_STORAGE(foam, capacity, persistent);
    PM_STORAGE(temperatures, capacity, persistent);
    PM_STORAGE(foam_source, capacity, persistent);
    PM_STORAGE(next_positions, capacity, persistent);
    PM_STORAGE(next_velocities, capacity, persistent);
    PM_STORAGE(next_ids, capacity, persistent);
    PM_STORAGE(next_foam, capacity, persistent);
    PM_STORAGE(next_temperatures, capacity, persistent);
    PM_STORAGE(keep, capacity, persistent);
    PM_STORAGE(selected, capacity, persistent);
    PM_STORAGE(keys[0], capacity, persistent);
    PM_STORAGE(keys[1], capacity, persistent);
    PM_STORAGE(indices[0], capacity, persistent);
    PM_STORAGE(indices[1], capacity, persistent);
    PM_STORAGE(forces, capacity, persistent);
    PM_STORAGE(body_impulses, capacity, persistent);
    PM_STORAGE(moving_contacts, capacity, persistent);
    PM_STORAGE(contact_samples, capacity, persistent);
    PM_STORAGE(contact_flags, capacity, persistent);
    PM_STORAGE(hit_box_flags, capacity, persistent);
    PM_STORAGE(next_contact_samples, capacity, persistent);
    PM_STORAGE(next_contact_flags, capacity, persistent);
    PM_STORAGE(contact_count, 1U, persistent);
    PM_STORAGE(contact_offset, 1U, persistent);
    state.sort_workspace_size = radix_sort_scratch_bytes(capacity);
    state.select_workspace_size = compaction_scratch_bytes(capacity);
    PM_STORAGE(sort_workspace, state.sort_workspace_size, scratch);
    PM_STORAGE(select_workspace, state.select_workspace_size, scratch);
#undef PM_STORAGE
    return {};
}
} // namespace parallel_mater::simulation
