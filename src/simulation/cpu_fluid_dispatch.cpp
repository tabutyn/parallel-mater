// SPDX-License-Identifier: MIT
#include "fluid_dispatch.hpp"

#include "../device/cpu_dispatch.hpp"
#include "parallel_mater_fluid_runtime.cpp"

#include <cstring>

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
Vec3_0 vec3(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}
FluidOptions_0 constants(const FluidDispatchConstants &value) noexcept {
    return {value.count, value.particle_radius, value.rest_density,
        value.support_radius, 0U, value.maximum_neighbors, value.repulsion,
        value.viscosity, value.velocity_damping, value.maximum_speed,
        value.normal_damping, value.rest_particle_volume,
        value.maximum_pair_acceleration};
}

} // namespace

void fluid_cell_keys(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue,
    void *positions, std::uint32_t *count, std::uint32_t capacity,
    float inverse_radius, std::uint64_t *keys,
    std::uint32_t *sorted_indices) noexcept {
    EntryPointParams_0 args{
        as<Vec3_0>(positions), count, capacity, inverse_radius,
        reinterpret_cast<std::uint64_t *>(keys), sorted_indices};
    device::dispatch_cpu(grid, block, fluid_emit_cells_Thread, &args);
}

void fluid_pair_cell_keys(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue, const void *first_positions,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    const void *second_positions, const std::uint32_t *second_count,
    std::uint32_t second_capacity, float inverse_radius,
    std::uint64_t *keys, std::uint32_t *identities) noexcept {
    EntryPointParams_10 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        inverse_radius, reinterpret_cast<std::uint64_t *>(keys), identities};
    device::dispatch_cpu(grid, block, fluid_emit_pair_cells_Thread, &args);
}

void fluid_pair_neighbor_counts(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue,
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
    EntryPointParams_11 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        first_maximum_neighbors,
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        second_maximum_neighbors,
        reinterpret_cast<std::uint64_t *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants),
        first_total_neighbors, first_cross_neighbors, first_density_samples,
        second_total_neighbors, second_cross_neighbors, second_density_samples,
        first_neighbor_overflow, second_neighbor_overflow,
        first_maximum_neighbor_count, second_maximum_neighbor_count};
    device::dispatch_cpu(grid, block, fluid_count_pair_neighbors_Thread, &args);
}

void fluid_pair_pressure_contact(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    void *first_forces, const void *second_positions,
    const void *second_velocities, const std::uint32_t *second_count,
    std::uint32_t second_capacity, void *second_forces,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &values) noexcept {
    EntryPointParams_12 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        as<Vec3_0>(first_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        as<Vec3_0>(second_forces),
        reinterpret_cast<std::uint64_t *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(values)};
    device::dispatch_cpu(
        grid, block, fluid_apply_pair_pressure_contact_Thread, &args);
}

void fluid_pair_viscosity(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    void *first_forces, const void *second_positions,
    const void *second_velocities, const std::uint32_t *second_count,
    std::uint32_t second_capacity, void *second_forces,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &values) noexcept {
    EntryPointParams_13 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        as<Vec3_0>(first_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        as<Vec3_0>(second_forces),
        reinterpret_cast<std::uint64_t *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(values)};
    device::dispatch_cpu(
        grid, block, fluid_apply_pair_viscosity_Thread, &args);
}

void fluid_miscible_pair_pressure_contact(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    void *first_forces, const float *first_density_samples,
    const void *second_positions, const void *second_velocities,
    const std::uint32_t *second_count, std::uint32_t second_capacity,
    void *second_forces, const float *second_density_samples,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants) noexcept {
    EntryPointParams_14 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        as<Vec3_0>(first_forces), const_cast<float *>(first_density_samples),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        as<Vec3_0>(second_forces), const_cast<float *>(second_density_samples),
        reinterpret_cast<std::uint64_t *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants)};
    device::dispatch_cpu(grid, block,
        fluid_apply_miscible_pair_pressure_contact_Thread, &args);
}

void fluid_miscible_pair_viscosity(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    void *first_forces, const void *second_positions,
    const void *second_velocities, const std::uint32_t *second_count,
    std::uint32_t second_capacity, void *second_forces,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants) noexcept {
    EntryPointParams_15 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        as<Vec3_0>(first_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        as<Vec3_0>(second_forces),
        reinterpret_cast<std::uint64_t *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants)};
    device::dispatch_cpu(
        grid, block, fluid_apply_miscible_pair_viscosity_Thread, &args);
}

void fluid_forces(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &values,
    const float up[3], float *foam_sources, std::uint64_t *keys,
    std::uint32_t *sorted_indices, std::uint32_t *neighbor_overflow,
    std::uint32_t *maximum_neighbor_count) noexcept {
    EntryPointParams_1 args{
        as<Vec3_0>(positions), as<Vec3_0>(velocities), count,
        reinterpret_cast<std::uint64_t *>(keys), sorted_indices, foam,
        constants(values), vec3(up), as<Vec3_0>(accelerations), foam_sources,
        neighbor_overflow, maximum_neighbor_count};
    device::dispatch_cpu(grid, block, fluid_compute_forces_Thread, &args);
}

void integrate_fluid(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &values,
    void *previous, float *foam_sources) noexcept {
    EntryPointParams_2 args{
        as<Vec3_0>(positions), as<Vec3_0>(velocities), as<Vec3_0>(previous),
        foam, as<Vec3_0>(accelerations), foam_sources, count, constants(values),
        vec3(values.gravity), values.timestep};
    device::dispatch_cpu(grid, block, fluid_integrate_Thread, &args);
}

void reserve_fluid_contact_events(device::ExecutionQueue,
    std::uint32_t *selected_count, std::uint32_t *offset,
    std::uint32_t *world_count, std::uint32_t *overflow,
    std::uint32_t capacity) noexcept {
    EntryPointParams_3 args{
        selected_count, offset, world_count, overflow, capacity};
    device::dispatch_cpu(1U, 1U, fluid_reserve_contact_events_Thread, &args);
}

void gather_fluid_contact_events(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue,
    std::uint32_t *selected, std::uint32_t *selected_count,
    std::uint32_t *offset, void *samples, std::uint32_t *stable_ids,
    void *body_ids, const void *fluid_id, void *events,
    std::uint32_t capacity) noexcept {
    FluidId_0 id{};
    std::memcpy(&id, fluid_id, sizeof id);
    EntryPointParams_4 args{
        selected, selected_count, offset, as<FluidContactSample_0>(samples),
        stable_ids, as<RigidBodyId_0>(body_ids), id,
        as<ContactEvent_0>(events), capacity};
    device::dispatch_cpu(grid, block, fluid_gather_contact_events_Thread, &args);
}

void find_fluid_source_vacancies(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue,
    const void *points, std::uint32_t amount, float spacing,
    const void *positions, const std::uint64_t *keys,
    const std::uint32_t *indices, std::uint32_t capacity,
    float inverse_cell_size, std::uint8_t *vacant) noexcept {
    EntryPointParams_5 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(points)), amount, spacing,
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        reinterpret_cast<std::uint64_t *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(indices), capacity, inverse_cell_size,
        vacant};
    device::dispatch_cpu(grid, block, fluid_source_vacancies_Thread, &args);
}

void emit_fluid_source(device::ExecutionQueue, const void *points,
    const std::uint8_t *vacant, std::uint32_t amount, float spacing,
    const float velocity[3], void *positions, void *velocities,
    std::uint32_t *ids, float *foam, float *temperatures,
    float initial_temperature, std::uint32_t *count, std::uint32_t capacity,
    std::uint32_t first_spawned, std::uint32_t first_id,
    std::uint32_t *misses) noexcept {
    EntryPointParams_6 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(points)),
        const_cast<std::uint8_t *>(vacant), amount, spacing, vec3(velocity),
        as<Vec3_0>(positions), as<Vec3_0>(velocities), ids, foam, temperatures,
        initial_temperature, count, capacity, first_spawned, first_id, misses};
    device::dispatch_cpu(1U, 1U, fluid_source_emit_Thread, &args);
}

void mark_fluid_destroy_flags(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue,
    const void *positions, const void *previous, const std::uint32_t *count,
    std::uint32_t capacity, const void *plane_options, bool combine,
    std::uint8_t *keep) noexcept {
    ParticleDestroyPlaneOptions_0 plane{};
    std::memcpy(&plane, plane_options, sizeof plane);
    EntryPointParams_7 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)),
        const_cast<std::uint32_t *>(count), capacity, plane, combine, keep};
    device::dispatch_cpu(grid, block, fluid_destroy_flags_Thread, &args);
}

void gather_fluid(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue,
    const void *positions, const void *velocities, const std::uint32_t *ids,
    const float *foam, const float *temperatures, const void *contact_samples,
    const std::uint8_t *contact_flags, bool copy_contacts,
    const std::uint32_t *selected, const std::uint32_t *count,
    std::uint32_t capacity, void *next_positions, void *next_velocities,
    std::uint32_t *next_ids, float *next_foam, float *next_temperatures,
    void *next_contact_samples, std::uint8_t *next_contact_flags) noexcept {
    EntryPointParams_8 args{
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<std::uint32_t *>(ids), const_cast<float *>(foam),
        const_cast<float *>(temperatures),
        const_cast<FluidContactSample_0 *>(
            as<FluidContactSample_0>(contact_samples)),
        const_cast<std::uint8_t *>(contact_flags), copy_contacts,
        const_cast<std::uint32_t *>(selected),
        const_cast<std::uint32_t *>(count), capacity,
        as<Vec3_0>(next_positions), as<Vec3_0>(next_velocities), next_ids,
        next_foam, next_temperatures,
        as<FluidContactSample_0>(next_contact_samples), next_contact_flags};
    device::dispatch_cpu(grid, block, fluid_gather_Thread, &args);
}

void copy_initial_fluid(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue,
    const void *input, std::uint32_t count, void *positions, void *velocities,
    std::uint32_t *ids, float *foam, float *temperatures,
    std::uint32_t *invalid) noexcept {
    EntryPointParams_9 args{
        const_cast<FluidParticle_0 *>(as<FluidParticle_0>(input)), count,
        as<Vec3_0>(positions), as<Vec3_0>(velocities), ids, foam,
        temperatures, invalid};
    device::dispatch_cpu(grid, block, fluid_copy_initial_Thread, &args);
}

} // namespace parallel_mater::simulation
