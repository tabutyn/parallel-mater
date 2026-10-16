// SPDX-License-Identifier: MIT
// Native argument binding only. All equations stay in shared Slang.
#include "geometry_dispatch.hpp"
#include "geometry_storage.hpp"
#include "driver/metal/schedule_context.hpp"
namespace parallel_mater::simulation {
PmResult integrate_rigid_bodies(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *accumulators, void *targets, void *input, void *output,
    std::uint32_t count, const float gravity[3], float timestep,
    std::uint32_t remaining_substeps, bool apply_impulses) noexcept {
    const Vec3 gravity_value{gravity[0], gravity[1], gravity[2]};
    return metal::launch(stream, "pm_integrate_rigid_bodies", grid,
        parameters, accumulators,
        targets, input,
        output, count, gravity_value, timestep,
        remaining_substeps, apply_impulses);
}

PmResult compute_rigid_world_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count,
    void *meshes, void *world_bounds) noexcept {
    if (auto *execution = metal::context(stream)) {
        const auto result = execution->checked_result([&] { execution->meshes(meshes, parameters, count); });
        if (result != PM_SUCCESS) return result;
    }
    return metal::launch(stream, "pm_compute_rigid_world_bounds", grid,
        parameters, previous_states,
        states, count, meshes,
        world_bounds);
}

PmResult broad_phase_rigid_pairs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *world_bounds, void *previous_states, void *states, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    void *fixed_groups, std::uint32_t count, std::uint8_t *active_flags) noexcept {
    return metal::launch(stream, "pm_broad_phase_rigid_pairs", grid,
        parameters, world_bounds,
        previous_states, states,
        meshes, ids,
        constraints, constraint_capacity,
        fixed_groups, count, active_flags);
}

PmResult generate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize block, std::size_t shared_bytes, device::ExecutionQueue stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, std::uint32_t mesh_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *leaf_pairs, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pair_cache_slot_capacity,
    std::uint32_t leaf_pairs_per_slot,
    std::uint32_t shared_leaf_capacity) noexcept {
    return metal::launch(stream, "pm_generate_rigid_leaf_pairs", grid,
        parameters, previous_states,
        states, count, meshes,
        mesh_capacity, active_pairs, active_pair_count, leaf_pairs,
        leaf_pair_counts, leaf_pair_cache_slot_capacity, leaf_pairs_per_slot,
        shared_leaf_capacity);
}

PmResult evaluate_rigid_leaf_pairs(
    device::DispatchSize grid, device::DispatchSize block, std::size_t shared_bytes, device::ExecutionQueue stream,
    void *parameters, void *previous_states, void *states,
    std::uint32_t count, void *meshes, void *ids, void *constraints,
    std::uint32_t constraint_capacity, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, void *leaf_pairs,
    std::uint32_t *leaf_pair_counts, std::uint32_t leaf_pairs_per_slot,
    void *leaf_manifolds, std::uint32_t blocks_per_pair, float timestep,
    void *manifolds) noexcept {
    return metal::launch(stream, "pm_evaluate_rigid_leaf_pairs", grid,
        parameters, previous_states,
        states, count, meshes,
        ids, constraints,
        constraint_capacity, active_pairs, active_pair_count,
        leaf_pairs, leaf_pair_counts, leaf_pairs_per_slot,
        leaf_manifolds, blocks_per_pair, timestep,
        manifolds);
}

PmResult reduce_rigid_leaf_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    std::uint32_t count, std::uint32_t *active_pairs,
    std::uint32_t *active_pair_count, std::uint32_t *leaf_pair_counts,
    std::uint32_t leaf_pairs_per_slot, void *leaf_manifolds,
    void *manifolds) noexcept {
    return metal::launch(stream, "pm_reduce_rigid_leaf_manifolds", grid,
        parameters, count, active_pairs,
        active_pair_count, leaf_pair_counts, leaf_pairs_per_slot,
        leaf_manifolds, manifolds);
}

PmResult finalize_rigid_contact_manifolds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *parameters,
    void *previous_states, void *states, std::uint32_t count, void *meshes,
    void *ids, void *constraints, std::uint32_t constraint_capacity,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    std::uint32_t *leaf_pair_counts, float timestep, void *manifolds,
    bool prepare_faces) noexcept {
    return metal::launch(stream, "pm_finalize_rigid_contact_manifolds", grid,
        parameters, previous_states,
        states, count, meshes,
        ids, constraints,
        constraint_capacity, active_pairs, active_pair_count, leaf_pair_counts,
        timestep, manifolds, prepare_faces);
}

PmResult clear_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *accumulators,
    void *targets, std::uint32_t count) noexcept {
    return metal::launch(stream, "pm_clear_rigid_inputs", grid,
        accumulators, targets,
        count);
}

PmResult capture_rigid_inputs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *accumulators,
    void *forces, void *torques, std::uint32_t count) noexcept {
    return metal::launch(stream, "pm_capture_rigid_inputs", grid,
        accumulators,
        forces, torques, count);
}

PmResult prepare_parallel_contact_events(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, std::uint32_t body_count,
    void *manifolds, const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_ids,
    std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity, std::uint32_t *event_count,
    bool collect_events, bool reset_events) noexcept {
    return metal::launch(stream, "pm_prepare_parallel_contact_events", grid,
        body_count, manifolds,
        active_pairs,
        active_pair_count,
        body_ids,
        event_offsets, events, event_capacity,
        event_count, collect_events, reset_events);
}

PmResult load_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, const void *cache,
    const std::uint32_t *slots, std::uint32_t capacity,
    std::uint64_t epoch, float timestep) noexcept {
    return metal::launch(stream, "pm_load_rigid_contact_cache", grid,
        manifolds,
        active_pairs,
        active_pair_count,
        body_states,
        body_ids, body_count,
        cache,
        slots, capacity,
        static_cast<unsigned long long>(epoch), timestep);
}

PmResult save_rigid_contact_cache(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *manifolds,
    const std::uint32_t *active_pairs,
    const std::uint32_t *active_pair_count, const void *body_states,
    const void *body_ids, std::uint32_t body_count, void *cache,
    std::uint32_t *slots, std::uint32_t capacity, std::uint64_t epoch,
    float timestep, const std::uint32_t *event_offsets, void *events,
    std::uint32_t event_capacity) noexcept {
    return metal::launch(stream, "pm_save_rigid_contact_cache", grid,
        manifolds,
        active_pairs,
        active_pair_count,
        body_states,
        body_ids, body_count,
        cache, slots, capacity,
        static_cast<unsigned long long>(epoch), timestep,
        event_offsets,
        events, event_capacity);
}

PmResult gather_rigid_debug_samples(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_ids,
    const void *body_states, const void *forces, const void *torques,
    std::uint32_t count, void *samples) noexcept {
    return metal::launch(stream, "pm_gather_rigid_debug_samples", grid,
        body_ids,
        body_states,
        forces,
        torques, count,
        samples);
}

PmResult clamp_rigid_speeds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    void *body_states, std::uint32_t count) noexcept {
    return metal::launch(stream, "pm_clamp_rigid_speeds", grid,
        body_parameters,
        body_states, count);
}

PmResult build_fixed_collision_groups(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *constraints,
    std::uint32_t constraint_capacity, const void *body_ids,
    std::uint32_t body_count, void *groups) noexcept {
    return metal::launch(stream, "pm_build_fixed_collision_groups", grid,
        constraints,
        constraint_capacity,
        body_ids, body_count,
        groups);
}

} // namespace parallel_mater::simulation
