// SPDX-License-Identifier: MIT
#include "cloth_dispatch.hpp"

#include "types.hpp"
#include "driver/metal/schedule_context.hpp"

namespace parallel_mater::simulation {
void predict_cloth_realtime(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, const float gravity[3],
    float velocity_damping, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *time_advances,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "cloth_realtime_predict", grid,
        positions, velocities, previous_positions, correction_used,
        inverse_masses, characteristic_lengths, count, capacity,
        Vec3{gravity[0], gravity[1], gravity[2]}, velocity_damping,
        maximum_speed, timestep, profile_version, time_advances, status);
}

void project_cloth_realtime_links(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *positions,
    void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count,
    std::uint32_t capacity, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "cloth_realtime_project_links", grid,
        positions, position_deltas, correction_used, inverse_masses,
        characteristic_lengths, offsets, neighbors, bond_active, count,
        capacity, timestep, profile_version, status);
}

void apply_cloth_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "cloth_realtime_apply_position_delta", grid,
        positions, position_deltas, correction_used, inverse_masses,
        characteristic_lengths, count, capacity, timestep, profile_version,
        status);
}

void project_cloth_realtime_volume(
    device::ExecutionQueue stream, void *positions,
    const float *inverse_masses, const float *characteristic_lengths,
    float *correction_used, const std::uint32_t *indices,
    std::uint32_t vertex_count, std::uint32_t triangle_count,
    const std::uint32_t *corner_offsets,
    const std::uint32_t *corner_indices, void *gradients,
    float target_volume, float orientation, float compliance, bool enabled,
    float timestep, std::uint32_t profile_version, float *lambda,
    bool reset_lambda, std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "cloth_realtime_project_volume", 1U,
        positions, inverse_masses, characteristic_lengths, correction_used,
        indices, vertex_count, triangle_count, corner_offsets,
        corner_indices, gradients, target_volume, orientation, compliance,
        enabled, timestep, profile_version, lambda, reset_lambda, status);
}

void project_cloth_realtime_contacts(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, void *positions,
    const void *previous_positions, const float *inverse_masses,
    const float *characteristic_lengths, float *correction_used,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float thickness, float friction,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "cloth_realtime_project_contacts", grid,
        positions, previous_positions, inverse_masses,
        characteristic_lengths, correction_used, contact_normals,
        contact_offsets, contact_active, count, capacity, thickness,
        friction, timestep, profile_version, status);
}

void commit_cloth_realtime(
    device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const float *inverse_masses, const float *characteristic_lengths,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, const void *bonds,
    std::uint8_t *bond_active, std::uint8_t *bond_damage,
    std::uint32_t bond_count, float break_strain,
    std::uint32_t fracture_persistence, const void *contact_impulses,
    float impact_break_threshold, std::uint32_t count,
    std::uint32_t capacity, float thickness, float friction,
    float maximum_speed, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    (void)metal::launch(stream, "cloth_realtime_commit", grid,
        positions, previous_positions, velocities, inverse_masses,
        characteristic_lengths, contact_normals, contact_offsets,
        contact_active, bonds, bond_active, bond_damage, bond_count,
        break_strain, fracture_persistence, contact_impulses,
        impact_break_threshold, count, capacity, thickness, friction,
        maximum_speed, timestep, profile_version, status);
}

void project_cloth_volume(
    device::ExecutionQueue stream, void *positions, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t vertex_count,
    std::uint32_t triangle_count, const std::uint32_t *corner_offsets,
    const std::uint32_t *corner_indices, void *gradients, float target_volume,
    float orientation, float compliance, float timestep, float *lambda,
    bool reset_lambda) noexcept {
    (void)metal::launch(stream, "cloth_project_volume", 1U,
        positions, inverse_masses,
        indices, vertex_count, triangle_count,
        corner_offsets,
        corner_indices, gradients,
        target_volume, orientation, compliance, timestep, lambda, reset_lambda);
}

void break_cloth_bonds(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *bonds, std::uint8_t *active, std::uint8_t *damage,
    std::uint32_t count, float break_strain, std::uint32_t persistence,
    const void *contact_impulses, float impact_threshold) noexcept {
    (void)metal::launch(stream, "cloth_break_bonds", grid,
        positions,
        bonds, active, damage,
        count, break_strain, persistence,
        contact_impulses, impact_threshold);
}

void limit_cloth_strain(
    device::ExecutionQueue stream, void *positions, void *scratch,
    const float *inverse_masses, const std::uint32_t *offsets,
    const void *neighbors, const std::uint8_t *free_nodes,
    std::uint32_t count) noexcept {
    (void)metal::launch(stream, "cloth_limit_strain", 1U,
        positions, scratch,
        inverse_masses,
        offsets,
        neighbors,
        free_nodes, count);
}

void update_cloth_surface(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const std::uint32_t *source_indices, void *surface_positions,
    std::uint32_t triangle_count) noexcept {
    (void)metal::launch(stream, "cloth_update_surface", grid,
        positions,
        source_indices,
        surface_positions, triangle_count);
}

} // namespace parallel_mater::simulation
