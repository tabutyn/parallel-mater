// SPDX-License-Identifier: MIT
// Binding only: liquid equations and source/outflow operations live in Slang.
#include "fluid_dispatch.hpp"
#include "types.hpp"
#include "driver/metal/schedule_context.hpp"
#include <cstring>

namespace parallel_mater::simulation {
namespace {
Vec3 vector(const float value[3]) { return {value[0], value[1], value[2]}; }
FluidOptions options(const FluidDispatchConstants &value) {
    FluidOptions result{};
    // Active keys form the sorted prefix; inactive capacity entries are sentinels.
    result.capacity = value.count;
    result.particle_radius = value.particle_radius;
    result.support_radius = value.support_radius;
    result.rest_density = value.rest_density;
    result.maximum_neighbors = value.maximum_neighbors;
    result.repulsion = value.repulsion;
    result.viscosity = value.viscosity;
    result.normal_damping = value.normal_damping;
    result.velocity_damping = value.velocity_damping;
    result.maximum_speed = value.maximum_speed;
    result.maximum_pair_acceleration = value.maximum_pair_acceleration;
    result.rest_particle_volume = value.rest_particle_volume;
    return result;
}
}

void fluid_cell_keys(device::DispatchSize grid, device::DispatchSize, device::ExecutionQueue stream,
    void *positions, std::uint32_t *count, std::uint32_t capacity,
    float inverse_radius, std::uint64_t *keys, std::uint32_t *indices) noexcept {
    (void)metal::launch(stream, "fluid_emit_cells", grid,
        positions, count, capacity, inverse_radius, keys, indices);
}
void fluid_pair_cell_keys(device::DispatchSize grid, device::DispatchSize,
    device::ExecutionQueue stream, const void *first_positions,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    const void *second_positions, const std::uint32_t *second_count,
    std::uint32_t second_capacity, float inverse_radius,
    std::uint64_t *keys, std::uint32_t *identities) noexcept {
    (void)metal::launch(stream, "fluid_emit_pair_cells", grid,
        first_positions, first_count, first_capacity, second_positions,
        second_count, second_capacity, inverse_radius, keys, identities);
}
void fluid_pair_neighbor_counts(device::DispatchSize grid,
    device::DispatchSize, device::ExecutionQueue stream,
    const void *first_positions, const std::uint32_t *first_count,
    std::uint32_t first_capacity, std::uint32_t first_maximum_neighbors,
    const void *second_positions, const std::uint32_t *second_count,
    std::uint32_t second_capacity, std::uint32_t second_maximum_neighbors,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants,
    std::uint32_t *first_total_neighbors,
    std::uint32_t *first_cross_neighbors,
    float *first_density_samples,
    std::uint32_t *second_total_neighbors,
    std::uint32_t *second_cross_neighbors,
    float *second_density_samples,
    std::uint32_t *first_neighbor_overflow,
    std::uint32_t *second_neighbor_overflow,
    std::uint32_t *first_maximum_neighbor_count,
    std::uint32_t *second_maximum_neighbor_count) noexcept {
    (void)metal::launch(stream, "fluid_count_pair_neighbors", grid,
        first_positions, first_count, first_capacity, first_maximum_neighbors,
        second_positions, second_count, second_capacity,
        second_maximum_neighbors, keys, identities, total_capacity,
        options(first_constants), options(second_constants),
        first_total_neighbors, first_cross_neighbors, first_density_samples,
        second_total_neighbors, second_cross_neighbors, second_density_samples,
        first_neighbor_overflow, second_neighbor_overflow,
        first_maximum_neighbor_count, second_maximum_neighbor_count);
}
void fluid_pair_pressure_contact(device::DispatchSize grid,
    device::DispatchSize, device::ExecutionQueue stream,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    void *first_forces, const void *second_positions,
    const void *second_velocities, const std::uint32_t *second_count,
    std::uint32_t second_capacity, void *second_forces,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &values) noexcept {
    (void)metal::launch(stream, "fluid_apply_pair_pressure_contact", grid,
        first_positions, first_velocities, first_count, first_capacity,
        first_forces, second_positions, second_velocities, second_count,
        second_capacity, second_forces, keys, identities, total_capacity,
        options(values));
}
void fluid_pair_viscosity(device::DispatchSize grid,
    device::DispatchSize, device::ExecutionQueue stream,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    void *first_forces, const void *second_positions,
    const void *second_velocities, const std::uint32_t *second_count,
    std::uint32_t second_capacity, void *second_forces,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &values) noexcept {
    (void)metal::launch(stream, "fluid_apply_pair_viscosity", grid,
        first_positions, first_velocities, first_count, first_capacity,
        first_forces, second_positions, second_velocities, second_count,
        second_capacity, second_forces, keys, identities, total_capacity,
        options(values));
}
void fluid_forces(device::DispatchSize grid, device::DispatchSize, device::ExecutionQueue stream,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &values,
    const float up[3], float *foam_sources, std::uint64_t *keys,
    std::uint32_t *indices, std::uint32_t *overflow,
    std::uint32_t *maximum_neighbors) noexcept {
    (void)metal::launch(stream, "fluid_compute_forces", grid,
        positions, velocities, count, keys, indices, foam, options(values),
        vector(up), accelerations, foam_sources, overflow, maximum_neighbors);
}
void integrate_fluid(device::DispatchSize grid, device::DispatchSize, device::ExecutionQueue stream,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &values,
    void *previous, float *foam_sources) noexcept {
    (void)metal::launch(stream, "fluid_integrate", grid,
        positions, velocities, previous, foam, accelerations, foam_sources,
        count, options(values), vector(values.gravity), values.timestep);
}
void reserve_fluid_contact_events(device::ExecutionQueue stream,
    std::uint32_t *selected_count, std::uint32_t *offset,
    std::uint32_t *world_count, std::uint32_t *overflow,
    std::uint32_t capacity) noexcept {
    (void)metal::launch(stream, "fluid_reserve_contact_events", 1U,
        selected_count, offset, world_count, overflow, capacity);
}
void gather_fluid_contact_events(device::DispatchSize grid, device::DispatchSize, device::ExecutionQueue stream,
    std::uint32_t *selected, std::uint32_t *selected_count,
    std::uint32_t *offset, void *samples, std::uint32_t *stable_ids,
    void *body_ids, const void *fluid_id, void *events,
    std::uint32_t capacity) noexcept {
    FluidId id{};
    std::memcpy(&id, fluid_id, sizeof(id));
    (void)metal::launch(stream, "fluid_gather_contact_events", grid,
        selected, selected_count, offset, samples, stable_ids, body_ids, id,
        events, capacity);
}
void find_fluid_source_vacancies(device::DispatchSize grid, device::DispatchSize, device::ExecutionQueue stream,
    const void *points, std::uint32_t amount, float spacing,
    const void *positions, const std::uint64_t *keys,
    const std::uint32_t *indices, std::uint32_t capacity,
    float inverse_cell_size, std::uint8_t *vacant) noexcept {
    (void)metal::launch(stream, "fluid_source_vacancies", grid,
        points, amount, spacing, positions, keys, indices, capacity,
        inverse_cell_size, vacant);
}
void emit_fluid_source(device::ExecutionQueue stream, const void *points,
    const std::uint8_t *vacant, std::uint32_t amount, float spacing,
    const float velocity[3], void *positions, void *velocities,
    std::uint32_t *ids, float *foam, float *temperatures,
    float initial_temperature, std::uint32_t *count, std::uint32_t capacity,
    std::uint32_t first_spawned, std::uint32_t first_id,
    std::uint32_t *misses) noexcept {
    (void)metal::launch(stream, "fluid_source_emit", 1U,
        points, vacant, amount, spacing, vector(velocity), positions,
        velocities, ids, foam, temperatures, initial_temperature, count,
        capacity, first_spawned, first_id, misses);
}
void mark_fluid_destroy_flags(device::DispatchSize grid, device::DispatchSize, device::ExecutionQueue stream,
    const void *positions, const void *previous, const std::uint32_t *count,
    std::uint32_t capacity, const void *plane_options, bool combine,
    std::uint8_t *keep) noexcept {
    ParticleDestroyPlaneOptions plane{};
    std::memcpy(&plane, plane_options, sizeof(plane));
    (void)metal::launch(stream, "fluid_destroy_flags", grid,
        positions, previous, count, capacity, plane, combine, keep);
}
void gather_fluid(device::DispatchSize grid, device::DispatchSize, device::ExecutionQueue stream,
    const void *positions, const void *velocities, const std::uint32_t *ids,
    const float *foam, const float *temperatures, const void *contact_samples,
    const std::uint8_t *contact_flags, bool copy_contacts,
    const std::uint32_t *selected, const std::uint32_t *count,
    std::uint32_t capacity, void *next_positions, void *next_velocities,
    std::uint32_t *next_ids, float *next_foam, float *next_temperatures,
    void *next_contact_samples, std::uint8_t *next_contact_flags) noexcept {
    (void)metal::launch(stream, "fluid_gather", grid,
        positions, velocities, ids, foam, temperatures, contact_samples,
        contact_flags, copy_contacts, selected, count, capacity, next_positions,
        next_velocities, next_ids, next_foam, next_temperatures,
        next_contact_samples, next_contact_flags);
}
void copy_initial_fluid(device::DispatchSize grid, device::DispatchSize, device::ExecutionQueue stream,
    const void *input, std::uint32_t count, void *positions, void *velocities,
    std::uint32_t *ids, float *foam, float *temperatures,
    std::uint32_t *invalid) noexcept {
    (void)metal::launch(stream, "fluid_copy_initial", grid,
        input, count, positions, velocities, ids, foam, temperatures, invalid);
}
} // namespace parallel_mater::simulation
