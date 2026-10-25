// SPDX-License-Identifier: MIT
#pragma once
#include "memory_arena.hpp"
#include "geometry_storage.hpp"
#include "primitives.hpp"
#include "production_spatial_runtime.hpp"
#include "rigid_spatial.hpp"
#include "status.hpp"
#include "types.hpp"
#include <algorithm>
#include <limits>
#include <iterator>

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

struct GeometryStoragePlan {
    Vec3 *vertices{};
    std::uint32_t *indices{};
    BvhNode *bvh_nodes{};
    std::uint32_t *bvh_leaves{};
    CollisionPlane *solid_planes{};
    Vec3 *shell_normals{};
    std::size_t vertex_capacity{};
    std::size_t index_capacity{};
    std::size_t bvh_node_capacity{};
    std::size_t bvh_leaf_capacity{};
    std::size_t solid_plane_capacity{};
    std::size_t shell_normal_capacity{};
};

// Geometry queries and allocation traverse this same typed plan. BVH nodes,
// leaves and convex data use topology-derived upper bounds; actual counts are
// retained separately in TriangleMeshResource after host preparation.
template<class State, class Visitor>
Status visit_geometry_storage(State &state, std::size_t vertices,
                              std::size_t indices, Visitor &visit) noexcept {
    if (vertices < 3U || indices < 3U || indices % 3U != 0U)
        return {StatusCode::invalid_argument, 0,
                "invalid triangle geometry topology"};
    const std::size_t triangles = indices / 3U;
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (triangles > maximum / 2U)
        return {StatusCode::out_of_memory, 0,
                "triangle geometry capacity overflow"};
    state.vertex_capacity = vertices;
    state.index_capacity = indices;
    state.bvh_node_capacity = triangles * 2U - 1U;
    state.bvh_leaf_capacity = triangles;
    state.solid_plane_capacity = triangles * 2U;
    state.shell_normal_capacity = triangles;
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(vertices, state.vertex_capacity);
    PM_STORAGE(indices, state.index_capacity);
    PM_STORAGE(bvh_nodes, state.bvh_node_capacity);
    PM_STORAGE(bvh_leaves, state.bvh_leaf_capacity);
    PM_STORAGE(solid_planes, state.solid_plane_capacity);
    PM_STORAGE(shell_normals, state.shell_normal_capacity);
#undef PM_STORAGE
    return {};
}

inline Status geometry_storage_requirements(
    std::size_t vertices, std::size_t indices,
    StorageRequirements &output) noexcept {
    output = {};
    GeometryStoragePlan state;
    StoragePlanner planner;
    const auto status = visit_geometry_storage(state, vertices, indices, planner);
    if (status) output = planner.requirements;
    return status;
}

struct PaintStoragePlan {
    Vec2 *uvs{};
    std::uint32_t *pixels{};
    std::size_t vertex_capacity{};
    std::size_t pixel_capacity{};
};

// Paint queries and materialization share this typed traversal. Width and
// height stay separate at the boundary so their product is checked before it
// becomes an allocation count.
template<class State, class Visitor>
Status visit_paint_storage(State &state, std::size_t vertices,
                           std::size_t width, std::size_t height,
                           Visitor &visit) noexcept {
    if (vertices == 0U || width == 0U || height == 0U)
        return {StatusCode::invalid_argument, 0,
                "paint storage capacities must be positive"};
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (width > maximum / height)
        return {StatusCode::out_of_memory, 0,
                "paint pixel capacity overflow"};
    state.vertex_capacity = vertices;
    state.pixel_capacity = width * height;
#define PM_STORAGE(member, count, kind) do { \
    const auto result = visit(state.member, count, MemoryClass::kind); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(uvs, state.vertex_capacity, persistent);
    PM_STORAGE(pixels, state.pixel_capacity, persistent);
#undef PM_STORAGE
    return {};
}

inline Status paint_storage_requirements(
    std::size_t vertices, std::size_t width, std::size_t height,
    StorageRequirements &output) noexcept {
    output = {};
    PaintStoragePlan state;
    StoragePlanner planner;
    const auto status =
        visit_paint_storage(state, vertices, width, height, planner);
    if (status) output = planner.requirements;
    return status;
}

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
    state.realtime_spatial_plan = {};
    state.realtime_rigid_spatial_plan = {};
    state.realtime_fluid_spatial_plan = {};
    state.realtime_deformable_spatial_plan = {};
    if (options.realtime_spatial_enabled) {
        const auto planned = make_spatial_service_plan(
            options.realtime_spatial_leaf_capacity,
            options.realtime_spatial_query_capacity,
            options.realtime_spatial_candidates_per_query,
            options.realtime_spatial_snapshot_count,
            state.realtime_spatial_plan);
        if (planned != PM_SUCCESS)
            return {StatusCode::invalid_argument,
                    static_cast<std::int32_t>(planned),
                    "invalid realtime spatial storage plan"};
        PM_STORAGE(realtime_spatial_persistent,
            spatial_storage_block_count(
                state.realtime_spatial_plan.persistent_bytes), persistent);
        PM_STORAGE(realtime_spatial_scratch,
            spatial_storage_block_count(
                state.realtime_spatial_plan.scratch_bytes), scratch);
        const auto rigid_planned = make_rigid_spatial_plan(
            options.rigid_body_capacity,
            options.realtime_spatial_candidates_per_query,
            state.realtime_rigid_spatial_plan);
        if (rigid_planned != PM_SUCCESS)
            return {StatusCode::invalid_argument,
                    static_cast<std::int32_t>(rigid_planned),
                    "invalid realtime rigid spatial storage plan"};
        PM_STORAGE(realtime_rigid_spatial_scratch,
            spatial_storage_block_count(
                state.realtime_rigid_spatial_plan.scratch_bytes), scratch);
        PM_STORAGE(realtime_rigid_spatial_output,
            spatial_storage_block_count(
                state.realtime_rigid_spatial_plan.output_bytes), persistent);
        const auto fluid_planned = make_fluid_spatial_plan(
            options.realtime_spatial_query_capacity,
            options.realtime_spatial_candidates_per_query,
            state.realtime_fluid_spatial_plan);
        if (fluid_planned != PM_SUCCESS)
            return {StatusCode::invalid_argument,
                    static_cast<std::int32_t>(fluid_planned),
                    "invalid realtime fluid spatial storage plan"};
        PM_STORAGE(realtime_fluid_spatial_scratch,
            spatial_storage_block_count(
                state.realtime_fluid_spatial_plan.scratch_bytes), scratch);
        const auto deformable_planned = make_deformable_spatial_plan(
            options.realtime_spatial_query_capacity,
            options.realtime_spatial_candidates_per_query,
            state.realtime_deformable_spatial_plan);
        if (deformable_planned != PM_SUCCESS)
            return {StatusCode::invalid_argument,
                    static_cast<std::int32_t>(deformable_planned),
                    "invalid realtime deformable spatial storage plan"};
        PM_STORAGE(realtime_deformable_spatial_scratch,
            spatial_storage_block_count(
                state.realtime_deformable_spatial_plan.scratch_bytes),
            scratch);
        PM_STORAGE(realtime_query_bounds,
            options.realtime_spatial_query_capacity, scratch);
        PM_STORAGE(realtime_query_masks,
            options.realtime_spatial_query_capacity, scratch);
        PM_STORAGE(realtime_excluded_sort_keys,
            options.realtime_spatial_query_capacity, scratch);
    }
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
    PM_STORAGE(realtime_accumulated_corrections, capacity, scratch);
    PM_STORAGE(realtime_position_deltas, capacity, scratch);
    PM_STORAGE(realtime_velocity_deltas, capacity, scratch);
    PM_STORAGE(realtime_time_advances, 1U, scratch);
    state.sort_workspace_size = radix_sort_scratch_bytes(capacity);
    state.select_workspace_size = compaction_scratch_bytes(capacity);
    PM_STORAGE(sort_workspace, state.sort_workspace_size, scratch);
    PM_STORAGE(select_workspace, state.select_workspace_size, scratch);
#undef PM_STORAGE
    return {};
}

// Private storage owned by one fluid-fluid graph edge. The combined index uses
// a population bit in each value, so equal local particle indices remain
// distinct while both populations share one spatial ordering.
struct FluidPairNeighborIndexPlan {
    std::uint64_t *keys[2]{};
    std::uint32_t *identities[2]{};
    std::uint32_t *total_neighbors[2]{};
    std::uint32_t *cross_neighbors[2]{};
    float *density_samples[2]{};
    // Immiscible reconstruction is pair-private and consumed by later pair
    // operators in the same step. Labels use -1 for the first endpoint and
    // +1 for the second; zero therefore remains an inactive-slot sentinel.
    float *phase_labels[2]{};
    Vec3 *interface_normals[2]{};
    float *interface_curvatures[2]{};
    std::uint32_t *interface_flags[2]{};
    std::uint32_t *neighbor_overflow[2]{};
    std::uint32_t *maximum_neighbor_count[2]{};
    std::uint32_t *interface_sample_count[2]{};
    std::uint32_t *under_resolved_interface_count[2]{};
    std::uint32_t *degenerate_interface_count[2]{};
    std::uint32_t *curvature_clamp_count[2]{};
    std::uint8_t *sort_workspace{};
    std::uint32_t total_capacity{};
    std::size_t sort_workspace_size{};
};

template<class State, class Visitor>
Status visit_fluid_pair_neighbor_index_storage(
    State &state, std::uint32_t first_capacity,
    std::uint32_t second_capacity, Visitor &visit) noexcept {
    constexpr auto primitive_maximum =
        static_cast<std::uint32_t>(std::numeric_limits<int>::max());
    if (!first_capacity || !second_capacity ||
        first_capacity > primitive_maximum ||
        second_capacity > primitive_maximum)
        return {StatusCode::invalid_argument, 0,
                "fluid pair capacity exceeds primitive range"};
    const auto combined = std::uint64_t(first_capacity) + second_capacity;
    if (combined > std::numeric_limits<std::uint32_t>::max())
        return {StatusCode::invalid_argument, 0,
                "fluid pair combined capacity exceeds index range"};
    state.total_capacity = static_cast<std::uint32_t>(combined);
#define PM_STORAGE(member, count, kind) do { \
    const auto result = visit(state.member, count, MemoryClass::kind); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(keys[0], state.total_capacity, persistent);
    PM_STORAGE(keys[1], state.total_capacity, persistent);
    PM_STORAGE(identities[0], state.total_capacity, persistent);
    PM_STORAGE(identities[1], state.total_capacity, persistent);
    PM_STORAGE(total_neighbors[0], first_capacity, persistent);
    PM_STORAGE(total_neighbors[1], second_capacity, persistent);
    PM_STORAGE(cross_neighbors[0], first_capacity, persistent);
    PM_STORAGE(cross_neighbors[1], second_capacity, persistent);
    // Density is an intermediate consumed by pair operators in the same step.
    PM_STORAGE(density_samples[0], first_capacity, scratch);
    PM_STORAGE(density_samples[1], second_capacity, scratch);
    PM_STORAGE(phase_labels[0], first_capacity, scratch);
    PM_STORAGE(phase_labels[1], second_capacity, scratch);
    PM_STORAGE(interface_normals[0], first_capacity, scratch);
    PM_STORAGE(interface_normals[1], second_capacity, scratch);
    PM_STORAGE(interface_curvatures[0], first_capacity, scratch);
    PM_STORAGE(interface_curvatures[1], second_capacity, scratch);
    PM_STORAGE(interface_flags[0], first_capacity, scratch);
    PM_STORAGE(interface_flags[1], second_capacity, scratch);
    PM_STORAGE(neighbor_overflow[0], 1U, persistent);
    PM_STORAGE(neighbor_overflow[1], 1U, persistent);
    PM_STORAGE(maximum_neighbor_count[0], 1U, persistent);
    PM_STORAGE(maximum_neighbor_count[1], 1U, persistent);
    PM_STORAGE(interface_sample_count[0], 1U, persistent);
    PM_STORAGE(interface_sample_count[1], 1U, persistent);
    PM_STORAGE(under_resolved_interface_count[0], 1U, persistent);
    PM_STORAGE(under_resolved_interface_count[1], 1U, persistent);
    PM_STORAGE(degenerate_interface_count[0], 1U, persistent);
    PM_STORAGE(degenerate_interface_count[1], 1U, persistent);
    PM_STORAGE(curvature_clamp_count[0], 1U, persistent);
    PM_STORAGE(curvature_clamp_count[1], 1U, persistent);
    state.sort_workspace_size = radix_sort_scratch_bytes(state.total_capacity);
    PM_STORAGE(sort_workspace, state.sort_workspace_size, scratch);
#undef PM_STORAGE
    return {};
}

inline Status fluid_pair_neighbor_index_storage_requirements(
    std::uint32_t first_capacity, std::uint32_t second_capacity,
    StorageRequirements &output) noexcept {
    output = {};
    FluidPairNeighborIndexPlan state;
    StoragePlanner planner;
    const auto status = visit_fluid_pair_neighbor_index_storage(
        state, first_capacity, second_capacity, planner);
    if (status) output = planner.requirements;
    return status;
}

// Capacity is the allocated node count, including possible fracture vertices.
template<class State, class Visitor>
Status visit_cloth_storage(State &state, std::size_t rigid, bool fracture,
    bool volume, bool rope, bool fluid, bool smoke, Visitor &visit) noexcept {
#define PM_STORAGE_KIND(member, count, kind) do { \
    const auto result = visit(state.member, count, MemoryClass::kind); \
    if (!result) return result; \
} while (false)
#define PM_STORAGE(member, count) PM_STORAGE_KIND(member, count, persistent)
#define PM_SCRATCH(member, count) PM_STORAGE_KIND(member, count, scratch)
    const auto vertices = state.vertex_capacity;
    const auto indices = state.index_count;
    const auto bonds = state.bond_count;
    PM_STORAGE(positions, vertices);
    PM_STORAGE(scratch, vertices);
    PM_STORAGE(previous, vertices);
    PM_STORAGE(velocities, vertices);
    PM_STORAGE(inverse_masses, vertices);
    PM_STORAGE(indices, indices);
    if (fracture) {
        PM_STORAGE(source_indices, indices);
        PM_STORAGE(vertex_sources, vertices);
        PM_STORAGE(free_triangle_nodes, vertices);
        PM_STORAGE(surface_positions, indices);
        PM_STORAGE(surface_triangle_indices, indices);
        PM_STORAGE(triangle_bonds, indices);
    }
    PM_STORAGE(bonds, bonds);
    PM_STORAGE(bond_active, bonds);
    PM_STORAGE(bond_damage, bonds);
    PM_STORAGE(offsets, std::size_t(vertices) + 1);
    PM_STORAGE(neighbors, state.neighbor_capacity);
    PM_STORAGE(body_impulses, vertices);
    PM_STORAGE(rigid_contact_forces, vertices);
    PM_STORAGE(soft_body_forces, vertices);
    if (rope) PM_STORAGE(rope_forces, vertices);
    PM_STORAGE(body_corrections, rigid);
    if (volume) {
        PM_STORAGE(volume_gradients, state.vertex_count);
        PM_STORAGE(volume_corner_offsets, std::size_t(state.vertex_count) + 1);
        PM_STORAGE(volume_corner_indices, indices);
        PM_STORAGE(volume_lambda, 1);
    }
    if (fluid) PM_STORAGE(fluid_forces, vertices);
    if (smoke) PM_STORAGE(smoke_forces, vertices);
    PM_STORAGE(count, 1);
    PM_STORAGE(realtime_positions, vertices);
    PM_STORAGE(realtime_velocities, vertices);
    PM_STORAGE(realtime_surface_positions, indices);
    PM_STORAGE(realtime_characteristic_lengths, vertices);
    PM_STORAGE(realtime_triangle_generations, indices / 3U);
    PM_SCRATCH(realtime_correction_used, vertices);
    PM_SCRATCH(realtime_contact_normals, vertices);
    PM_SCRATCH(realtime_contact_offsets, vertices);
    PM_SCRATCH(realtime_contact_active, vertices);
    PM_SCRATCH(realtime_time_advances, 1U);
#undef PM_SCRATCH
#undef PM_STORAGE
#undef PM_STORAGE_KIND
    return {};
}

template<class State, class Visitor>
Status visit_rope_storage(State &state, std::size_t nodes, std::size_t rigid,
                          Visitor &visit) noexcept {
    if (nodes && rigid > std::numeric_limits<std::size_t>::max() / nodes)
        return {StatusCode::out_of_memory, 0, "rope contact storage overflow"};
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(positions, nodes);
    PM_STORAGE(previous, nodes);
    PM_STORAGE(velocities, nodes);
    PM_STORAGE(constraint_forces, nodes);
    PM_STORAGE(contact_forces, nodes);
    PM_STORAGE(fluid_contact_forces, nodes);
    PM_STORAGE(soft_body_contact_forces, nodes);
    PM_STORAGE(directions, nodes);
    PM_STORAGE(scratch, nodes);
    PM_STORAGE(normals, nodes);
    PM_STORAGE(normals2, nodes);
    PM_STORAGE(rest, nodes);
    PM_STORAGE(lambda, nodes);
    PM_STORAGE(soft_anchor_positions, 2);
    PM_STORAGE(soft_anchor_velocities, 2);
    PM_STORAGE(soft_anchor_impulses, 2);
    PM_STORAGE(soft_anchor_inverse_masses, 2);
    PM_STORAGE(body_translation, rigid);
    PM_STORAGE(body_rotation, rigid);
    PM_STORAGE(solid_hint, nodes * rigid);
#undef PM_STORAGE
    return {};
}

template<class State, class Visitor>
Status visit_soft_body_storage(State &state, std::size_t neighbors,
    std::size_t warp_neighbors, std::size_t surface_influences, Visitor &visit) noexcept {
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(positions, state.node_count);
    PM_STORAGE(rest_positions, state.node_count);
    PM_STORAGE(scratch, state.node_count);
    PM_STORAGE(previous, state.node_count);
    PM_STORAGE(velocities, state.node_count);
    PM_STORAGE(velocity_scratch, state.node_count);
    PM_STORAGE(inverse_masses, state.node_count);
    PM_STORAGE(bonds, state.bond_count);
    PM_STORAGE(bond_active, state.bond_count);
    PM_STORAGE(offsets, std::size_t(state.node_count) + 1);
    PM_STORAGE(neighbors, neighbors);
    if (warp_neighbors) PM_STORAGE(warp_neighbors, warp_neighbors);
    if (warp_neighbors)
        PM_STORAGE(minimum_rest_lengths,state.node_count);
    PM_STORAGE(surface_rest_positions, state.surface_vertex_count);
    PM_STORAGE(surface_positions, state.surface_vertex_count);
    PM_STORAGE(surface_indices, state.surface_index_count);
    PM_STORAGE(surface_bindings, state.surface_vertex_count);
    PM_STORAGE(surface_corner_corrections, state.surface_index_count);
    PM_STORAGE(surface_node_offsets, std::size_t(state.node_count) + 1);
    PM_STORAGE(surface_node_influences, surface_influences);
    PM_STORAGE(body_impulses, state.node_count);
    PM_STORAGE(body_position_corrections, state.node_count);
    PM_STORAGE(cloth_forces, state.node_count);
    PM_STORAGE(rope_forces, state.node_count);
    PM_STORAGE(fluid_forces, state.node_count);
    PM_STORAGE(rigid_contact_forces, state.node_count);
    PM_STORAGE(contact_normals, state.node_count);
    PM_STORAGE(contact_arms, state.node_count);
    PM_STORAGE(contact_momentum_delta, state.node_count);
    PM_STORAGE(contact_friction_delta, state.node_count);
    PM_STORAGE(contact_normal_delta, state.node_count);
    PM_STORAGE(predicted_momentum, 1U);
    PM_STORAGE(shape_orientation, 1U);
    PM_STORAGE(dynamic_contact_flag, 1U);
    PM_STORAGE(contact_count, 1U);
    PM_STORAGE(count, 1U);
#undef PM_STORAGE
    return {};
}

template<class State, class Visitor>
Status visit_smoke_grid_storage(State &state, Visitor &visit) noexcept {
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(velocity, state.cell_count);
    PM_STORAGE(density, state.cell_count);
    PM_STORAGE(temperature, state.cell_count);
    PM_STORAGE(density_accumulator, state.cell_count);
    PM_STORAGE(temperature_accumulator, state.cell_count);
    PM_STORAGE(pressure[0], state.cell_count);
    PM_STORAGE(pressure[1], state.cell_count);
    PM_STORAGE(divergence, state.cell_count);
    PM_STORAGE(residual, state.cell_count);
    PM_STORAGE(vorticity, state.cell_count);
    PM_STORAGE(strain, state.cell_count);
    PM_STORAGE(subgrid_force, state.cell_count);
    PM_STORAGE(solid, state.cell_count);
    PM_STORAGE(rhs_max, 1);
    PM_STORAGE(residual_max, 1);
    PM_STORAGE(pressure_converged, 1);
    for (unsigned axis = 0; axis < 3; ++axis) {
        PM_STORAGE(face_velocity[axis][0], state.face_count[axis]);
        PM_STORAGE(face_velocity[axis][1], state.face_count[axis]);
        PM_STORAGE(face_reverse[axis], state.face_count[axis]);
        PM_STORAGE(face_open[axis], state.face_count[axis]);
        PM_STORAGE(face_wall_velocity[axis], state.face_count[axis]);
        PM_STORAGE(face_normal[axis], state.face_count[axis]);
        PM_STORAGE(face_nearest_triangle[axis], state.face_count[axis]);
    }
    for (std::size_t level = 0; level < std::size(state.coarse); ++level) {
        PM_STORAGE(coarse[level].pressure[0], state.coarse[level].cell_count);
        PM_STORAGE(coarse[level].pressure[1], state.coarse[level].cell_count);
        PM_STORAGE(coarse[level].rhs, state.coarse[level].cell_count);
        PM_STORAGE(coarse[level].residual, state.coarse[level].cell_count);
        for (unsigned axis = 0; axis < 3; ++axis)
            PM_STORAGE(coarse[level].open[axis], state.coarse[level].face_count[axis]);
    }
#undef PM_STORAGE
    return {};
}

template<class State, class Visitor>
Status visit_smoke_particles_storage(State &state, std::uint32_t capacity, Visitor &visit) noexcept {
#define PM_STORAGE(member, count, kind) do { \
    const auto result = visit(state.member, count, MemoryClass::kind); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(positions, capacity, persistent);
    PM_STORAGE(previous_positions, capacity, persistent);
    PM_STORAGE(velocities, capacity, persistent);
    PM_STORAGE(ages, capacity, persistent);
    PM_STORAGE(thermal_lift, capacity, persistent);
    PM_STORAGE(number_densities, capacity, persistent);
    PM_STORAGE(pressures, capacity, persistent);
    PM_STORAGE(accelerations, capacity, persistent);
    PM_STORAGE(vorticities, capacity, persistent);
    PM_STORAGE(vorticity_magnitudes, capacity, persistent);
    PM_STORAGE(keys[0], capacity, persistent);
    PM_STORAGE(keys[1], capacity, persistent);
    PM_STORAGE(indices[0], capacity, persistent);
    PM_STORAGE(indices[1], capacity, persistent);
    PM_STORAGE(rigid_impulses, capacity, persistent);
    state.sort_workspace_size = radix_sort_scratch_bytes(capacity);
    PM_STORAGE(sort_workspace, state.sort_workspace_size, scratch);
#undef PM_STORAGE
    return {};
}

struct CouplingStorageLimits {
    std::size_t fluid_particles{}, cloth_vertices{}, rope_nodes{};
    std::size_t soft_nodes{}, soft_vertices{}, soft_indices{};
};
template<class State, class Visitor>
Status visit_fluid_soft_storage(State &state, std::size_t particles, std::size_t nodes, std::size_t vertices, std::size_t tree_nodes, std::size_t triangles, Visitor &visit) noexcept {
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(contacts, particles);
    PM_STORAGE(counts, nodes);
    PM_STORAGE(position_deltas, nodes);
    PM_STORAGE(impulses, nodes);
    PM_STORAGE(previous_surface, vertices);
    PM_STORAGE(bounds, 2);
    PM_STORAGE(contact_count, 1);
    PM_STORAGE(maximum_penetration, 1);
    PM_STORAGE(tree, tree_nodes);
    PM_STORAGE(parents, tree_nodes);
    PM_STORAGE(ready, tree_nodes);
    PM_STORAGE(triangle_order, triangles);
#undef PM_STORAGE
    return {};
}
template<class State, class Visitor>
Status visit_rope_soft_tree_storage(State &state, std::size_t tree_nodes, std::size_t triangles, Visitor &visit) noexcept {
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(tree, tree_nodes);
    PM_STORAGE(order, triangles);
    PM_STORAGE(parents, tree_nodes);
    PM_STORAGE(ready, tree_nodes);
#undef PM_STORAGE
    return {};
}
template<class State, class Visitor>
Status visit_rope_soft_data_storage(State &state, std::size_t nodes, std::size_t vertices, Visitor &visit) noexcept {
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(node_impulses, nodes);
    PM_STORAGE(previous_surface, vertices);
    PM_STORAGE(bounds, 2);
    PM_STORAGE(contact_count, 1);
    PM_STORAGE(maximum_penetration, 1);
#undef PM_STORAGE
    return {};
}
template<class State, class Visitor>
Status visit_fluid_rope_storage(State &state, std::size_t particles, Visitor &visit) noexcept {
    constexpr auto maximum = std::numeric_limits<std::size_t>::max();
    if (particles > maximum / 2U)
        return {StatusCode::out_of_memory, 0,
                "fluid rope contribution capacity overflow"};
    const auto contributions = particles * 2U;
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(contribution_nodes, contributions);
    PM_STORAGE(contribution_impulses, contributions);
    PM_STORAGE(contact_count, 1);
    PM_STORAGE(maximum_penetration, 1);
#undef PM_STORAGE
    return {};
}
template<class State, class Visitor>
Status visit_soft_cloth_storage(State &state, std::size_t nodes, std::size_t vertices, Visitor &visit) noexcept {
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(contacts, nodes);
    PM_STORAGE(cloth_contact_counts, vertices);
#undef PM_STORAGE
    return {};
}
template<class State, class Visitor>
Status visit_emitter_storage(State &state, std::size_t sites, Visitor &visit) noexcept {
#define PM_STORAGE(member, count) do { \
    const auto result = visit(state.member, count, MemoryClass::persistent); \
    if (!result) return result; \
} while (false)
    PM_STORAGE(points, sites);
    PM_STORAGE(vacant, sites);
    PM_STORAGE(capacity_misses, 1);
#undef PM_STORAGE
    return {};
}
} // namespace parallel_mater::simulation
