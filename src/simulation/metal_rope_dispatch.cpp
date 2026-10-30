// SPDX-License-Identifier: MIT
#include "rope_dispatch.hpp"
#include "driver/metal/resource_records.hpp"

namespace parallel_mater::simulation {
namespace {
Vec3 vec3(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}
} // namespace

void predict_rope_realtime(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, float *correction_used,
    const float *inverse_masses, const void *anchor_targets,
    const std::uint8_t *anchor_active, std::uint32_t count,
    std::uint32_t capacity, float node_spacing, float radius,
    const float gravity[3], float velocity_damping, float friction,
    float maximum_speed, float timestep, std::uint32_t profile_version,
    std::uint32_t *time_advances, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "rope_realtime_predict", grid,
        positions, velocities, previous_positions, correction_used,
        inverse_masses, anchor_targets, anchor_active, count, capacity,
        node_spacing, radius, vec3(gravity), velocity_damping, friction,
        maximum_speed, timestep, profile_version, time_advances, status);
}

void project_rope_realtime_constraints(
    device::ExecutionQueue stream, const void *positions,
    void *position_deltas, void *constraint_forces, float *correction_used,
    const float *inverse_masses, const float *rest_lengths,
    const float *rest_bend_lengths, const std::uint8_t *anchor_active,
    std::uint32_t count, std::uint32_t capacity, float node_spacing,
    float structural_compliance, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "rope_realtime_project_constraints", 1U,
        positions, position_deltas, constraint_forces, correction_used,
        inverse_masses, rest_lengths, rest_bend_lengths, anchor_active,
        count, capacity,
        node_spacing, structural_compliance, timestep, profile_version,
        status);
}

void apply_rope_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, float *correction_used,
    const float *inverse_masses, const std::uint8_t *anchor_active,
    std::uint32_t count, std::uint32_t capacity, float node_spacing,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "rope_realtime_apply_position_delta", grid,
        positions, position_deltas, correction_used, inverse_masses,
        anchor_active, count, capacity, node_spacing, timestep,
        profile_version, status);
}

void project_rope_realtime_contacts(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions,
    const void *previous_positions, const float *inverse_masses,
    float *correction_used, const std::uint8_t *anchor_active,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_spacing, float radius,
    bool self_collision, float friction, float timestep,
    std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "rope_realtime_project_contacts", grid,
        positions, previous_positions, inverse_masses, correction_used,
        anchor_active, contact_normals, contact_offsets, contact_active,
        count, capacity, node_spacing, radius,
        std::uint32_t{self_collision}, friction, timestep, profile_version,
        status);
}

void finalize_rope_realtime_self_contacts(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions, void *velocities,
    const void *previous_positions, const void *previous_velocities,
    const float *inverse_masses, const std::uint8_t *anchor_active,
    const std::uint8_t *contact_active,
    std::uint32_t count, std::uint32_t capacity, float node_spacing,
    float radius, bool self_collision, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "rope_realtime_finalize_self_contacts",
        grid, positions, velocities, previous_positions,
        previous_velocities, inverse_masses, anchor_active, contact_active,
        count, capacity, node_spacing, radius,
        std::uint32_t{self_collision},
        maximum_speed, timestep, profile_version, status);
}

void commit_rope_realtime(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const float *inverse_masses, const std::uint8_t *anchor_active,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_spacing, float radius,
    float friction, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "rope_realtime_commit", grid,
        positions, previous_positions, velocities, inverse_masses,
        anchor_active, contact_normals, contact_offsets, contact_active,
        count, capacity, node_spacing, radius, friction, maximum_speed,
        timestep, profile_version, status);
}

void advance_rope(device::ExecutionQueue stream, const RopeDispatchData &rope,
    float timestep, Vec3 gravity, int first, int last,
    const void *body_parameters, void *body_states,
    const void *previous_body_states, const void *meshes,
    std::uint32_t body_count, bool first_substep,
    const RopeSoftTargetDispatch &soft_a,
    const RopeSoftTargetDispatch &soft_b) noexcept {
    if (auto *execution = metal::context(stream)) {
        if (execution->checked_result([&] { execution->meshes(meshes, body_parameters, body_count); }) != PM_SUCCESS) return;
    }
    const bool coupled = soft_a.surface != nullptr ||
        soft_b.surface != nullptr || soft_a.first_anchor || soft_a.last_anchor ||
        soft_b.first_anchor || soft_b.last_anchor;
    (void)metal::launch(stream,
        coupled ? "rope_advance_coupled" : "rope_advance", 1U,
        metal::rope_record(rope),
        timestep, gravity, first, last, body_parameters, body_states,
        previous_body_states, meshes, body_count, first_substep,
        metal::target_record(soft_a), metal::target_record(soft_b));
}
} // namespace parallel_mater::simulation
