// SPDX-License-Identifier: MIT
#include "geometry_dispatch.hpp"
#include "geometry_constraints_dispatch.hpp"
#include "geometry_storage.hpp"
#include "primitives.hpp"

#include "driver/d3d12/schedule_context.hpp"
#include "parallel_mater_slang_dxil.hpp"

#include <array>
#include <cassert>
#include <cstring>
#include <type_traits>

namespace parallel_mater::simulation {
namespace {

template <std::size_t Size> struct Constants {
    std::array<std::byte, Size> bytes{};

    template <typename T>
    void set(d3d12::generated::SlangUniformLayout uniform,
             const T &value) noexcept {
        static_assert(std::is_trivially_copyable_v<T>);
        assert(uniform.size == sizeof(T));
        if (uniform.size == sizeof(T) &&
            uniform.offset + sizeof(T) <= bytes.size())
            std::memcpy(bytes.data() + uniform.offset, &value, sizeof(T));
    }

    void boolean(d3d12::generated::SlangUniformLayout uniform,
                 bool value) noexcept {
        set(uniform, static_cast<std::uint32_t>(value));
    }

    void buffer(d3d12::generated::SlangUniformLayout uniform,
                d3d12::ScheduleContext &execution, const void *pointer) noexcept {
        set(uniform, execution.address(pointer));
    }
};

template <typename Left, typename Right>
consteval bool same_layout() {
    if constexpr (Left::constant_size != Right::constant_size ||
                  Left::uniforms.size() != Right::uniforms.size()) return false;
    for (std::size_t index = 0U; index < Left::uniforms.size(); ++index) {
        const auto &left = Left::uniforms[index];
        const auto &right = Right::uniforms[index];
        if (std::string_view(left.name) != right.name ||
            left.offset != right.offset || left.size != right.size) return false;
    }
    return true;
}

template <std::size_t Size>
PmResult launch(device::ExecutionQueue stream, std::string_view module,
                   std::string_view entry, device::DispatchSize grid, device::DispatchSize block,
                   const Constants<Size> &constants) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    return execution == nullptr
        ? PM_ERROR_INVALID_ARGUMENT
        : execution->dispatch(module, entry, grid, block, constants.bytes);
}

} // namespace

PmResult integrate_rigid_bodies(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *accumulators, void *targets, void *input, void *output,
    std::uint32_t count, const float gravity[3], float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr || gravity == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_integrate_rigid_bodies;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_accumulators, *execution, accumulators);
    values.buffer(Layout::arg_targets, *execution, targets);
    values.buffer(Layout::arg_input, *execution, input);
    values.buffer(Layout::arg_output, *execution, output);
    values.set(Layout::arg_count, count);
    values.set(Layout::arg_gravity, Vec3{gravity[0], gravity[1], gravity[2]});
    values.set(Layout::arg_timestep, timestep);
    values.set(Layout::arg_remaining_substeps, remaining_substeps);
    values.boolean(Layout::arg_apply_impulses, apply_impulses);
    return launch(stream, "geometry", "pm_integrate_rigid_bodies",
                  grid, block, values);
}

PmResult compute_rigid_world_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count,
    void *meshes, void *world_bounds) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_compute_rigid_world_bounds;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_previous_states, *execution, previous_states);
    values.buffer(Layout::arg_states, *execution, states);
    values.set(Layout::arg_count, count);
    values.set(Layout::arg_meshes, execution->translated_meshes(
                        meshes, parameters, count));
    values.buffer(Layout::arg_world_bounds, *execution, world_bounds);
    return launch(stream, "geometry", "pm_compute_rigid_world_bounds",
                  grid, block, values);
}

PmResult broad_phase_rigid_pairs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *world_bounds, void *previous_states, void *states, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t count,
    std::uint8_t *active_flags) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_broad_phase_rigid_pairs;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_world_bounds, *execution, world_bounds);
    values.buffer(Layout::arg_previous_states, *execution, previous_states);
    values.buffer(Layout::arg_states, *execution, states);
    values.set(Layout::arg_meshes, execution->translated_meshes(
                        meshes, parameters, count));
    values.buffer(Layout::arg_ids, *execution, ids);
    values.buffer(Layout::arg_constraints, *execution, constraints);
    values.set(Layout::arg_constraint_capacity, constraint_capacity);
    values.buffer(Layout::arg_fixed_groups, *execution, fixed_groups);
    values.set(Layout::arg_count, count);
    values.buffer(Layout::arg_active_flags, *execution, active_flags);
    return launch(stream, "geometry", "pm_broad_phase_rigid_pairs",
                  grid, block, values);
}

PmResult emit_rigid_spatial_proxies(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *parameters, void *previous_states,
    void *predicted_states, void *meshes, std::uint32_t mesh_capacity,
    void *ids, std::uint32_t body_count, std::uint32_t leaf_offset,
    std::uint32_t leaf_capacity, std::uint32_t sort_key_base,
    std::uint32_t topology_revision, std::uint32_t query_mask,
    const float path_deviation[3], const float correction_budget[3],
    float epsilon, void *metadata, void *bounds,
    std::uint8_t *build_flags, void *query_bounds,
    std::uint32_t *query_masks, std::uint32_t *excluded_sort_keys,
    std::uint32_t *status) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr || path_deviation == nullptr ||
        correction_budget == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_emit_rigid_spatial_proxies;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_previous_states, *execution, previous_states);
    values.buffer(Layout::arg_predicted_states, *execution, predicted_states);
    values.set(Layout::arg_meshes, execution->translated_meshes(
        meshes, parameters, body_count, mesh_capacity));
    values.set(Layout::arg_mesh_capacity, mesh_capacity);
    values.buffer(Layout::arg_ids, *execution, ids);
    values.set(Layout::arg_body_count, body_count);
    values.set(Layout::arg_leaf_offset, leaf_offset);
    values.set(Layout::arg_leaf_capacity, leaf_capacity);
    values.set(Layout::arg_sort_key_base, sort_key_base);
    values.set(Layout::arg_topology_revision, topology_revision);
    values.set(Layout::arg_query_mask, query_mask);
    values.set(Layout::arg_path_deviation,
        Vec3{path_deviation[0], path_deviation[1], path_deviation[2]});
    values.set(Layout::arg_correction_budget,
        Vec3{correction_budget[0], correction_budget[1],
             correction_budget[2]});
    values.set(Layout::arg_epsilon, epsilon);
    values.buffer(Layout::arg_metadata, *execution, metadata);
    values.buffer(Layout::arg_bounds, *execution, bounds);
    values.buffer(Layout::arg_build_flags, *execution, build_flags);
    values.buffer(Layout::arg_query_bounds, *execution, query_bounds);
    values.buffer(Layout::arg_query_masks, *execution, query_masks);
    values.buffer(Layout::arg_excluded_sort_keys, *execution,
                  excluded_sort_keys);
    values.buffer(Layout::arg_status, *execution, status);
    return launch(stream, "geometry", "pm_emit_rigid_spatial_proxies",
                  grid, block, values);
}

PmResult filter_rigid_spatial_candidates(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *metadata,
    std::uint32_t leaf_capacity, std::uint32_t rigid_leaf_offset,
    std::uint32_t *candidate_counts, std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_body,
    std::uint32_t candidate_slot_capacity, void *parameters, void *ids,
    void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t body_count,
    std::uint32_t *pair_codes, std::uint8_t *pair_flags,
    std::uint32_t *status) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_filter_rigid_spatial_candidates;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_metadata, *execution, metadata);
    values.set(Layout::arg_leaf_capacity, leaf_capacity);
    values.set(Layout::arg_rigid_leaf_offset, rigid_leaf_offset);
    values.buffer(Layout::arg_candidate_counts, *execution, candidate_counts);
    values.buffer(Layout::arg_candidate_sources, *execution,
                  candidate_sources);
    values.set(Layout::arg_candidates_per_body, candidates_per_body);
    values.set(Layout::arg_candidate_slot_capacity, candidate_slot_capacity);
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_ids, *execution, ids);
    values.buffer(Layout::arg_constraints, *execution, constraints);
    values.set(Layout::arg_constraint_capacity, constraint_capacity);
    values.buffer(Layout::arg_fixed_groups, *execution, fixed_groups);
    values.set(Layout::arg_body_count, body_count);
    values.buffer(Layout::arg_pair_codes, *execution, pair_codes);
    values.buffer(Layout::arg_pair_flags, *execution, pair_flags);
    values.buffer(Layout::arg_status, *execution, status);
    return launch(stream, "geometry", "pm_filter_rigid_spatial_candidates",
                  grid, block, values);
}

PmResult gather_rigid_spatial_pairs(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, std::uint32_t *compacted_slots,
    std::uint32_t *active_pair_count,
    std::uint32_t *candidate_pair_codes,
    std::uint32_t candidate_slot_capacity, std::uint32_t *active_pairs,
    std::uint32_t *status) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_gather_rigid_spatial_pairs;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_compacted_slots, *execution, compacted_slots);
    values.buffer(Layout::arg_active_pair_count, *execution,
                  active_pair_count);
    values.buffer(Layout::arg_candidate_pair_codes, *execution,
                  candidate_pair_codes);
    values.set(Layout::arg_candidate_slot_capacity, candidate_slot_capacity);
    values.buffer(Layout::arg_active_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_status, *execution, status);
    return launch(stream, "geometry", "pm_gather_rigid_spatial_pairs",
                  grid, block, values);
}

PmResult generate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize block, std::size_t, device::ExecutionQueue stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, std::uint32_t mesh_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *leaf_pairs, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity,
    std::uint32_t leaf_pairs_per_slot,
    std::uint32_t shared_leaf_capacity) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_generate_rigid_leaf_pairs;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_previous_states, *execution, previous_states);
    values.buffer(Layout::arg_states, *execution, states);
    values.set(Layout::arg_count, count);
    values.set(Layout::arg_meshes, execution->translated_meshes(
                        meshes, parameters, count, mesh_capacity));
    values.set(Layout::arg_mesh_capacity, mesh_capacity);
    values.buffer(Layout::arg_active_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_active_pair_count, *execution, active_pair_count);
    values.buffer(Layout::arg_leaf_pairs, *execution, leaf_pairs);
    values.buffer(Layout::arg_leaf_pair_counts, *execution, leaf_pair_counts);
    values.set(Layout::arg_leaf_pair_cache_slot_capacity,
               leaf_pair_cache_slot_capacity);
    values.set(Layout::arg_leaf_pairs_per_slot, leaf_pairs_per_slot);
    values.set(Layout::arg_shared_leaf_capacity, shared_leaf_capacity);
    return launch(stream, "geometry", "pm_generate_rigid_leaf_pairs",
                  grid, block, values);
}

PmResult evaluate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize block, std::size_t, device::ExecutionQueue stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, void *ids, void *constraints,
    std::uint32_t constraint_capacity, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, void *leaf_pairs,
    std::uint32_t *leaf_pair_counts, std::uint32_t leaf_pairs_per_slot,
    void *leaf_manifolds, std::uint32_t blocks_per_pair, float timestep,
    bool canonicalize_swept_witness, void *manifolds) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_evaluate_rigid_leaf_pairs;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_previous_states, *execution, previous_states);
    values.buffer(Layout::arg_states, *execution, states);
    values.set(Layout::arg_count, count);
    values.set(Layout::arg_meshes, execution->translated_meshes(
                        meshes, parameters, count));
    values.buffer(Layout::arg_ids, *execution, ids);
    values.buffer(Layout::arg_constraints, *execution, constraints);
    values.set(Layout::arg_constraint_capacity, constraint_capacity);
    values.buffer(Layout::arg_active_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_active_pair_count, *execution, active_pair_count);
    values.buffer(Layout::arg_leaf_pairs, *execution, leaf_pairs);
    values.buffer(Layout::arg_leaf_pair_counts, *execution, leaf_pair_counts);
    values.set(Layout::arg_leaf_pairs_per_slot, leaf_pairs_per_slot);
    values.buffer(Layout::arg_leaf_manifolds, *execution, leaf_manifolds);
    values.set(Layout::arg_blocks_per_pair, blocks_per_pair);
    values.set(Layout::arg_timestep, timestep);
    values.set(Layout::arg_canonicalize_swept_witness,
               canonicalize_swept_witness);
    values.buffer(Layout::arg_manifolds, *execution, manifolds);
    return launch(stream, "geometry", "pm_evaluate_rigid_leaf_pairs",
                  grid, block, values);
}

PmResult reduce_rigid_leaf_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    std::uint32_t count, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    bool canonicalize_swept_witness, void *manifolds) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_reduce_rigid_leaf_manifolds;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.set(Layout::arg_count, count);
    values.buffer(Layout::arg_active_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_active_pair_count, *execution, active_pair_count);
    values.buffer(Layout::arg_leaf_pair_counts, *execution, leaf_pair_counts);
    values.set(Layout::arg_leaf_pairs_per_slot, leaf_pairs_per_slot);
    values.buffer(Layout::arg_leaf_manifolds, *execution, leaf_manifolds);
    values.set(Layout::arg_canonicalize_swept_witness,
               canonicalize_swept_witness);
    values.buffer(Layout::arg_manifolds, *execution, manifolds);
    return launch(stream, "geometry", "pm_reduce_rigid_leaf_manifolds",
                  grid, block, values);
}

PmResult finalize_rigid_contact_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep, void *manifolds,
    bool prepare_faces) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_finalize_rigid_contact_manifolds;
    using FaceLayout = d3d12::generated::layout::geometry::Kernel_pm_prepare_rigid_face_manifolds;
    static_assert(same_layout<Layout, FaceLayout>());
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_previous_states, *execution, previous_states);
    values.buffer(Layout::arg_states, *execution, states);
    values.set(Layout::arg_count, count);
    values.set(Layout::arg_meshes, execution->translated_meshes(
                        meshes, parameters, count));
    values.buffer(Layout::arg_ids, *execution, ids);
    values.buffer(Layout::arg_constraints, *execution, constraints);
    values.set(Layout::arg_constraint_capacity, constraint_capacity);
    values.buffer(Layout::arg_active_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_active_pair_count, *execution, active_pair_count);
    values.buffer(Layout::arg_leaf_pair_counts, *execution, leaf_pair_counts);
    values.set(Layout::arg_timestep, timestep);
    values.buffer(Layout::arg_manifolds, *execution, manifolds);
    values.boolean(Layout::arg_prepare_faces, prepare_faces);
    const char *entry = prepare_faces ? "pm_prepare_rigid_face_manifolds"
                                      : "pm_finalize_rigid_contact_manifolds";
    return launch(stream, "geometry", entry,
                  grid, block, values);
}

PmResult clear_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *accumulators,
    void *targets, std::uint32_t count) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_clear_rigid_inputs;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_accumulators, *execution, accumulators);
    values.buffer(Layout::arg_targets, *execution, targets);
    values.set(Layout::arg_count, count);
    return launch(stream, "geometry", "pm_clear_rigid_inputs",
                  grid, block, values);
}

PmResult capture_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *accumulators,
    void *forces, void *torques, std::uint32_t count) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_capture_rigid_inputs;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_accumulators, *execution, accumulators);
    values.buffer(Layout::arg_forces, *execution, forces);
    values.buffer(Layout::arg_torques, *execution, torques);
    values.set(Layout::arg_count, count);
    return launch(stream, "geometry", "pm_capture_rigid_inputs",
                  grid, block, values);
}

PmResult prepare_parallel_contact_events(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, std::uint32_t body_count,
    void *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_ids,
    std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_prepare_parallel_contact_events;
    Constants<Layout::constant_size> values;
    values.set(Layout::arg_count, body_count);
    values.buffer(Layout::arg_manifolds, *execution, manifolds);
    values.buffer(Layout::arg_active_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_active_pair_count, *execution, active_pair_count);
    values.buffer(Layout::arg_ids, *execution, body_ids);
    values.buffer(Layout::arg_event_offsets, *execution, event_offsets);
    values.buffer(Layout::arg_events, *execution, events);
    values.set(Layout::arg_event_capacity, event_capacity);
    values.buffer(Layout::arg_event_count, *execution, event_count);
    values.boolean(Layout::arg_collect_events, collect_events);
    values.boolean(Layout::arg_reset_events, reset_events);
    return launch(stream, "geometry", "pm_prepare_parallel_contact_events",
                  grid, block, values);
}

PmResult load_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, const void *cache,
    const std::uint32_t *slots, std::uint32_t capacity,
    std::uint64_t epoch, float timestep) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_load_rigid_contact_cache;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_manifolds, *execution, manifolds);
    values.buffer(Layout::arg_active_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_active_count, *execution, active_pair_count);
    values.buffer(Layout::arg_states, *execution, body_states);
    values.buffer(Layout::arg_ids, *execution, body_ids);
    values.set(Layout::arg_body_count, body_count);
    values.buffer(Layout::arg_cache, *execution, cache);
    values.buffer(Layout::arg_slots, *execution, slots);
    values.set(Layout::arg_capacity, capacity);
    values.set(Layout::arg_epoch, epoch);
    values.set(Layout::arg_timestep, timestep);
    return launch(stream, "geometry", "pm_load_rigid_contact_cache",
                  grid, block, values);
}

PmResult save_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, void *cache,
    std::uint32_t *slots, std::uint32_t capacity, std::uint64_t epoch,
    float timestep, const std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_save_rigid_contact_cache;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_manifolds, *execution, manifolds);
    values.buffer(Layout::arg_active_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_active_count, *execution, active_pair_count);
    values.buffer(Layout::arg_states, *execution, body_states);
    values.buffer(Layout::arg_ids, *execution, body_ids);
    values.set(Layout::arg_body_count, body_count);
    values.buffer(Layout::arg_cache, *execution, cache);
    values.buffer(Layout::arg_slots, *execution, slots);
    values.set(Layout::arg_capacity, capacity);
    values.set(Layout::arg_epoch, epoch);
    values.set(Layout::arg_timestep, timestep);
    values.buffer(Layout::arg_event_offsets, *execution, event_offsets);
    values.buffer(Layout::arg_events, *execution, events);
    values.set(Layout::arg_event_capacity, event_capacity);
    return launch(stream, "geometry", "pm_save_rigid_contact_cache",
                  grid, block, values);
}

PmResult gather_rigid_debug_samples(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_ids,
    const void *body_states, const void *forces, const void *torques,
    std::uint32_t count, void *samples) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_gather_rigid_debug_samples;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_ids, *execution, body_ids);
    values.buffer(Layout::arg_states, *execution, body_states);
    values.buffer(Layout::arg_forces, *execution, forces);
    values.buffer(Layout::arg_torques, *execution, torques);
    values.set(Layout::arg_count, count);
    values.buffer(Layout::arg_samples, *execution, samples);
    return launch(stream, "geometry", "pm_gather_rigid_debug_samples",
                  grid, block, values);
}

PmResult clamp_rigid_speeds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    void *body_states, std::uint32_t count) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_clamp_rigid_speeds;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, body_parameters);
    values.buffer(Layout::arg_states, *execution, body_states);
    values.set(Layout::arg_count, count);
    return launch(stream, "geometry", "pm_clamp_rigid_speeds",
                  grid, block, values);
}

PmResult build_fixed_collision_groups(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *constraints,
    std::uint32_t constraint_capacity, const void *body_ids,
    std::uint32_t body_count, void *groups) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry::Kernel_pm_build_fixed_collision_groups;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_constraints, *execution, constraints);
    values.set(Layout::arg_capacity, constraint_capacity);
    values.buffer(Layout::arg_ids, *execution, body_ids);
    values.set(Layout::arg_count, body_count);
    values.buffer(Layout::arg_groups, *execution, groups);
    return launch(stream, "geometry", "pm_build_fixed_collision_groups",
                  grid, block, values);
}

PmResult geometry_constraints_occupancy(
    int *resident_blocks, std::uint32_t) noexcept {
    if (resident_blocks == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *resident_blocks = 1;
    return PM_SUCCESS;
}

PmResult prepare_geometry_constraints(
    device::ExecutionQueue stream, const void *parameters,
    const RigidBodyState *previous, RigidBodyState *predicted,
    std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    RigidBodyId *ids, GeometryConstraintBody *bodies, void *schedule,
    float timestep, Vec3 gravity, std::uint32_t base_iterations,
    std::uint32_t requested_iterations,
    std::uint32_t grid_blocks) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry_constraints::Kernel_prepare_avbd_kernel;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_previous, *execution, previous);
    values.buffer(Layout::arg_states, *execution, predicted);
    values.set(Layout::arg_count, body_count);
    values.buffer(Layout::arg_manifolds, *execution, manifolds);
    values.buffer(Layout::arg_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_pair_count, *execution, active_pair_count);
    values.buffer(Layout::arg_joints, *execution, constraints);
    values.set(Layout::arg_joint_capacity, constraint_capacity);
    values.buffer(Layout::arg_ids, *execution, ids);
    values.buffer(Layout::arg_bodies, *execution, bodies);
    values.buffer(Layout::arg_schedule, *execution, schedule);
    values.set(Layout::arg_dt, timestep);
    values.set(Layout::arg_gravity, gravity);
    values.set(Layout::arg_base_iterations, base_iterations);
    values.set(Layout::arg_requested_iterations, requested_iterations);
    values.set(Layout::arg_grid_blocks, grid_blocks);
    return launch(stream, "geometry_constraints", "prepare_avbd_kernel",
                  device::DispatchSize(grid_blocks), device::DispatchSize(128U), values);
}

PmResult solve_geometry_constraints(
    device::ExecutionQueue stream, const void *parameters,
    const RigidBodyState *previous, RigidBodyState *states,
    std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t grid_blocks) noexcept {
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    using Layout = d3d12::generated::layout::geometry_constraints::Kernel_solve_avbd_kernel;
    Constants<Layout::constant_size> values;
    values.buffer(Layout::arg_parameters, *execution, parameters);
    values.buffer(Layout::arg_previous, *execution, previous);
    values.buffer(Layout::arg_states, *execution, states);
    values.set(Layout::arg_count, body_count);
    values.buffer(Layout::arg_manifolds, *execution, manifolds);
    values.buffer(Layout::arg_pairs, *execution, active_pairs);
    values.buffer(Layout::arg_pair_count, *execution, active_pair_count);
    values.buffer(Layout::arg_joints, *execution, constraints);
    values.set(Layout::arg_joint_capacity, constraint_capacity);
    values.buffer(Layout::arg_bodies, *execution, bodies);
    values.buffer(Layout::arg_schedule, *execution, schedule);
    values.set(Layout::arg_dt, timestep);
    return launch(stream, "geometry_constraints", "solve_avbd_kernel",
                  device::DispatchSize(grid_blocks), device::DispatchSize(32U), values);
}

PmResult compact_flagged_indices(
    device::ExecutionQueue stream, const std::uint8_t *flags, std::uint32_t count,
    std::uint32_t *output, std::uint32_t *output_count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U)
        return stream.services->clear(output_count, 0, sizeof(*output_count));
    const std::uint32_t blocks = primitive_block_count(count);
    if (flags == nullptr || output == nullptr || output_count == nullptr ||
        scratch == nullptr || scratch_bytes < compaction_scratch_bytes(count))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    auto *counts = static_cast<std::uint32_t *>(scratch);
    auto *offsets = counts + blocks;
    const auto packed_flags = execution->address(flags);
    using CountLayout = d3d12::generated::layout::primitives::Kernel_pm_compact_count_blocks;
    Constants<CountLayout::constant_size> count_values;
    count_values.set(CountLayout::arg_flags, packed_flags);
    count_values.set(CountLayout::arg_count, count);
    count_values.set(CountLayout::arg_tile_size, primitive_tile_size);
    count_values.buffer(CountLayout::arg_block_counts, *execution, counts);
    PmResult result = launch(
        stream, "primitives", "pm_compact_count_blocks",
        device::DispatchSize(blocks), device::DispatchSize(primitive_tile_size), count_values);
    if (result != PM_SUCCESS) return result;
    using ScanLayout = d3d12::generated::layout::primitives::Kernel_pm_exclusive_scan_u32;
    Constants<ScanLayout::constant_size> scan_values;
    scan_values.buffer(ScanLayout::arg_input, *execution, counts);
    scan_values.set(ScanLayout::arg_count, blocks);
    scan_values.buffer(ScanLayout::arg_output, *execution, offsets);
    scan_values.buffer(ScanLayout::arg_total, *execution, output_count);
    result = launch(stream, "primitives", "pm_exclusive_scan_u32",
                    device::DispatchSize(1U), device::DispatchSize(primitive_tile_size), scan_values);
    if (result != PM_SUCCESS) return result;
    using ScatterLayout = d3d12::generated::layout::primitives::Kernel_pm_compact_scatter_indices;
    Constants<ScatterLayout::constant_size> scatter_values;
    scatter_values.set(ScatterLayout::arg_flags, packed_flags);
    scatter_values.set(ScatterLayout::arg_count, count);
    scatter_values.set(ScatterLayout::arg_tile_size, primitive_tile_size);
    scatter_values.buffer(ScatterLayout::arg_block_offsets, *execution, offsets);
    scatter_values.buffer(ScatterLayout::arg_output, *execution, output);
    return launch(stream, "primitives", "pm_compact_scatter_indices",
                  device::DispatchSize(blocks), device::DispatchSize(primitive_tile_size), scatter_values);
}

PmResult radix_sort_pairs_u64_u32(
    device::ExecutionQueue stream, std::uint64_t *input_keys,
    std::uint64_t *output_keys, std::uint32_t *input_values,
    std::uint32_t *output_values, std::uint32_t count,
    void *scratch, std::size_t scratch_bytes) noexcept {
    if (count == 0U) return PM_SUCCESS;
    if (input_keys == nullptr || output_keys == nullptr ||
        input_values == nullptr || output_values == nullptr ||
        scratch == nullptr || scratch_bytes < radix_sort_scratch_bytes(count))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *execution = d3d12::context(stream.native_handle);
    if (execution == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    const auto blocks = primitive_block_count(count);
    auto *histograms = static_cast<std::uint32_t *>(scratch);
    auto *offsets = histograms + 16U * blocks;
    auto *source_keys = input_keys;
    auto *target_keys = output_keys;
    auto *source_values = input_values;
    auto *target_values = output_values;
    for (std::uint32_t shift = 0U; shift < 64U; shift += 4U) {
        using HistogramLayout = d3d12::generated::layout::primitives::Kernel_pm_radix_histogram_u64;
        Constants<HistogramLayout::constant_size> histogram;
        histogram.buffer(HistogramLayout::arg_keys, *execution, source_keys);
        histogram.set(HistogramLayout::arg_count, count);
        histogram.set(HistogramLayout::arg_tile_size, primitive_tile_size);
        histogram.set(HistogramLayout::arg_shift, shift);
        histogram.set(HistogramLayout::arg_block_count, blocks);
        histogram.buffer(HistogramLayout::arg_histograms, *execution, histograms);
        auto result = launch(stream, "primitives", "pm_radix_histogram_u64",
                             device::DispatchSize(blocks), device::DispatchSize(primitive_tile_size), histogram);
        if (result != PM_SUCCESS) return result;
        using PrefixLayout = d3d12::generated::layout::primitives::Kernel_pm_radix_prefix_u32;
        Constants<PrefixLayout::constant_size> prefix;
        prefix.buffer(PrefixLayout::arg_histograms, *execution, histograms);
        prefix.set(PrefixLayout::arg_block_count, blocks);
        prefix.buffer(PrefixLayout::arg_offsets, *execution, offsets);
        result = launch(stream, "primitives", "pm_radix_prefix_u32",
                        device::DispatchSize(1U), device::DispatchSize(1U), prefix);
        if (result != PM_SUCCESS) return result;
        using RadixScatterLayout = d3d12::generated::layout::primitives::Kernel_pm_radix_scatter_u64_u32;
        Constants<RadixScatterLayout::constant_size> scatter;
        scatter.buffer(RadixScatterLayout::arg_input_keys, *execution, source_keys);
        scatter.buffer(RadixScatterLayout::arg_input_values, *execution, source_values);
        scatter.set(RadixScatterLayout::arg_count, count);
        scatter.set(RadixScatterLayout::arg_tile_size, primitive_tile_size);
        scatter.set(RadixScatterLayout::arg_shift, shift);
        scatter.set(RadixScatterLayout::arg_block_count, blocks);
        scatter.buffer(RadixScatterLayout::arg_offsets, *execution, offsets);
        scatter.buffer(RadixScatterLayout::arg_output_keys, *execution, target_keys);
        scatter.buffer(RadixScatterLayout::arg_output_values, *execution, target_values);
        result = launch(stream, "primitives", "pm_radix_scatter_u64_u32",
                        device::DispatchSize(blocks), device::DispatchSize(primitive_tile_size), scatter);
        if (result != PM_SUCCESS) return result;
        std::swap(source_keys, target_keys);
        std::swap(source_values, target_values);
    }
    auto result = execution->copy_device(output_keys, input_keys,
        static_cast<std::size_t>(count) * sizeof(*input_keys));
    if (result == PM_SUCCESS)
        result = execution->copy_device(output_values, input_values,
            static_cast<std::size_t>(count) * sizeof(*input_values));
    return result;
}

} // namespace parallel_mater::simulation
