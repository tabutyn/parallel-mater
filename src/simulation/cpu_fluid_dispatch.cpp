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
PMPackedVec3_0 packed_vec3(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}
Vec3_0 vec3(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}
PMFluidConstants_0 constants(const FluidDispatchConstants &value) noexcept {
    return {value.timestep, packed_vec3(value.gravity), value.count,
        value.particle_radius, value.support_radius, value.repulsion,
        value.viscosity, value.normal_damping, value.velocity_damping,
        value.maximum_speed, value.maximum_pair_acceleration,
        value.rest_density, value.maximum_neighbors};
}

} // namespace

void fluid_cell_keys(dim3 grid, dim3 block, cudaStream_t,
    void *positions, std::uint32_t *count, std::uint32_t capacity,
    float inverse_radius, std::uint64_t *keys,
    std::uint32_t *sorted_indices) noexcept {
    EntryPointParams_0 args{
        as<PMPackedVec3_0>(positions), count, capacity, inverse_radius,
        reinterpret_cast<std::uint64_t *>(keys), sorted_indices};
    device::dispatch_cpu(dim3((grid.x * block.x + 63U) / 64U),
                         dim3(64U), pm_fluid_cell_keys_Thread, &args);
}

void fluid_forces(dim3 grid, dim3 block, cudaStream_t,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &values,
    const float up[3], float *foam_sources, std::uint64_t *keys,
    std::uint32_t *sorted_indices, std::uint32_t *neighbor_overflow,
    std::uint32_t *maximum_neighbor_count) noexcept {
    EntryPointParams_1 args{
        as<PMPackedVec3_0>(positions), as<PMPackedVec3_0>(velocities),
        as<PMPackedVec3_0>(accelerations), foam, count, constants(values),
        packed_vec3(up), foam_sources,
        reinterpret_cast<std::uint64_t *>(keys),
        sorted_indices, neighbor_overflow, maximum_neighbor_count};
    device::dispatch_cpu(dim3((grid.x * block.x + 63U) / 64U),
                         dim3(64U), pm_fluid_forces_Thread, &args);
}

void integrate_fluid(dim3 grid, dim3 block, cudaStream_t,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &values,
    void *previous, float *foam_sources) noexcept {
    EntryPointParams_2 args{
        as<PMPackedVec3_0>(positions), as<PMPackedVec3_0>(velocities),
        as<PMPackedVec3_0>(accelerations), foam, count, constants(values),
        as<PMPackedVec3_0>(previous), foam_sources};
    device::dispatch_cpu(dim3((grid.x * block.x + 63U) / 64U),
                         dim3(64U), pm_fluid_integrate_Thread, &args);
}

void reserve_fluid_contact_events(cudaStream_t,
    std::uint32_t *selected_count, std::uint32_t *offset,
    std::uint32_t *world_count, std::uint32_t *overflow,
    std::uint32_t capacity) noexcept {
    EntryPointParams_3 args{
        selected_count, offset, world_count, overflow, capacity};
    device::dispatch_cpu(1U, 1U, fluid_reserve_contact_events_Thread, &args);
}

void gather_fluid_contact_events(dim3 grid, dim3 block, cudaStream_t,
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

void find_fluid_source_vacancies(dim3 grid, dim3 block, cudaStream_t,
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

void emit_fluid_source(cudaStream_t, const void *points,
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

void mark_fluid_destroy_flags(dim3 grid, dim3 block, cudaStream_t,
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

void gather_fluid(dim3 grid, dim3 block, cudaStream_t,
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

void copy_initial_fluid(dim3 grid, dim3 block, cudaStream_t,
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
