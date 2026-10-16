// SPDX-License-Identifier: MIT
#include "geometry_fluid_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"

namespace parallel_mater::simulation {

void collide_fluid_with_static_body(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const void *previous, float *foam,
    const std::uint32_t *count, float radius, float spawn_clearance,
    std::uint32_t first_spawned, bool recover_spawn, const void *up,
    const void *body_parameters, const void *body_states, const void *meshes,
    const void *body_ids,
    std::uint32_t body_index, float particle_mass, bool collect_contacts,
    void *contact_samples, std::uint8_t *contact_flags,
    std::uint32_t fluid_index, std::uint32_t fluid_generation,
    bool apply_paint, const void *paint_fields,
    std::uint32_t paint_field_capacity, const void *paint_rules,
    std::uint32_t paint_rule_capacity) noexcept {
    if (auto *execution = metal::context(stream)) {
        if (execution->checked_result([&] { execution->paint_fields(paint_fields, paint_field_capacity, meshes); }) != PM_SUCCESS) return;
    }
    const auto fluid_id = FluidId{fluid_index, fluid_generation};
    (void)metal::launch(stream, "fluid_static_contacts", grid,
        positions, velocities,
        previous, foam,
        count, radius, spawn_clearance,
        first_spawned, recover_spawn, *static_cast<const Vec3 *>(up),
        body_parameters,
        body_states,
        meshes,
        body_ids,
        body_index, particle_mass, collect_contacts,
        contact_samples, contact_flags,
        fluid_id, apply_paint, paint_fields, paint_field_capacity,
        paint_rules, paint_rule_capacity);
}

void calculate_fluid_body_bounds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    const void *previous_states, const void *current_states,
    const void *meshes, std::uint32_t body_count, void *bounds) noexcept {
    if (auto *execution = metal::context(stream)) {
        const auto result = execution->checked_result([&] { execution->meshes(meshes, body_parameters, body_count); });
        if (result != PM_SUCCESS) return;
    }
    (void)metal::launch(stream, "fluid_body_bounds_kernel", grid,
        body_parameters,
        previous_states,
        current_states,
        meshes,
        body_count, bounds);
}

void index_fluid_body_cells(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *body_parameters,
    const void *bounds, std::uint32_t body_count, std::uint32_t words,
    float padding, void *masks, void *global_masks) noexcept {
    (void)metal::launch(stream, "fluid_index_body_cells", grid,
        body_parameters,
        bounds, body_count, words,
        padding, masks,
        global_masks);
}

void detect_fluid_moving_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *previous, const std::uint32_t *count, float radius,
    const void *body_parameters, const void *previous_states,
    const void *current_states, const void *meshes, const void *bounds,
    std::uint32_t body_count, const void *masks,
    const void *global_masks, std::uint32_t words,
    bool first_iteration, void *contacts, std::uint32_t *contact_counts) noexcept {
    (void)metal::launch(stream, "fluid_detect_moving_contacts", grid,
        positions,
        previous,
        count, radius,
        body_parameters,
        previous_states,
        current_states,
        meshes,
        bounds, body_count,
        masks,
        global_masks,
        words, first_iteration,
        contacts, contact_counts);
}

void resolve_fluid_moving_contacts(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, float *foam, const std::uint32_t *count, float radius,
    float particle_mass, float timestep, float maximum_speed,
    const void *body_parameters, const void *body_states, const void *meshes,
    const void *contacts, const std::uint32_t *contact_counts, void *impulses,
    bool collect_contacts, void *contact_samples,
    std::uint8_t *particle_contact_flags, std::uint32_t fluid_index,
    std::uint32_t fluid_generation, const void *body_ids, bool apply_paint,
    const void *paint_fields, std::uint32_t paint_field_capacity,
    const void *paint_rules, std::uint32_t paint_rule_capacity) noexcept {
    if (auto *execution = metal::context(stream)) {
        if (execution->checked_result([&] { execution->paint_fields(paint_fields, paint_field_capacity, meshes); }) != PM_SUCCESS) return;
    }
    const auto fluid_id = FluidId{fluid_index, fluid_generation};
    (void)metal::launch(stream, "fluid_resolve_moving_contacts", grid,
        positions, velocities, foam,
        count, radius, particle_mass, timestep,
        maximum_speed,
        body_parameters,
        body_states,
        meshes,
        contacts,
        contact_counts,
        impulses, collect_contacts,
        contact_samples, particle_contact_flags,
        fluid_id, body_ids, apply_paint, paint_fields, paint_field_capacity,
        paint_rules, paint_rule_capacity);
}

void reduce_fluid_body_impulses(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *impulses,
    const std::uint32_t *particle_count, const void *body_parameters,
    void *body_states, const std::uint32_t *contact_flags,
    std::uint32_t body_count, const void *position_corrections) noexcept {
    (void)metal::launch(stream, "reduce_point_body_impulses", grid,
        impulses,
        particle_count,
        body_parameters,
        body_states,
        contact_flags, body_count,
        position_corrections);
}

} // namespace parallel_mater::simulation
