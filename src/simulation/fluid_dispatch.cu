// SPDX-License-Identifier: MIT
#include "fluid_dispatch.hpp"

#include "parallel_mater_fluid_runtime.cu"

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
    FluidOptions_0 result{};
    result.capacity_1 = value.count;
    result.particle_radius_0 = value.particle_radius;
    result.support_radius_0 = value.support_radius;
    result.repulsion_0 = value.repulsion;
    result.viscosity_0 = value.viscosity;
    result.normal_damping_0 = value.normal_damping;
    result.velocity_damping_0 = value.velocity_damping;
    result.maximum_speed_0 = value.maximum_speed;
    result.maximum_pair_acceleration_0 = value.maximum_pair_acceleration;
    result.rest_density_0 = value.rest_density;
    result.rest_particle_volume_0 = value.rest_particle_volume;
    result.maximum_neighbors_0 = value.maximum_neighbors;
    return result;
}

} // namespace

void fluid_cell_keys(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *positions, std::uint32_t *count, std::uint32_t capacity,
    float inverse_radius, std::uint64_t *keys,
    std::uint32_t *sorted_indices) noexcept {
    fluid_emit_cells<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), count, capacity, inverse_radius,
        reinterpret_cast<unsigned long long *>(keys), sorted_indices);
}

void fluid_pair_cell_keys(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *first_positions,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    const void *second_positions, const std::uint32_t *second_count,
    std::uint32_t second_capacity, float inverse_radius,
    std::uint64_t *keys, std::uint32_t *identities) noexcept {
    fluid_emit_pair_cells<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        inverse_radius, reinterpret_cast<unsigned long long *>(keys), identities);
}

void fluid_pair_neighbor_counts(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue stream,
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
    fluid_count_pair_neighbors<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        first_maximum_neighbors,
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        second_maximum_neighbors,
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants),
        first_total_neighbors, first_cross_neighbors, first_density_samples,
        second_total_neighbors, second_cross_neighbors, second_density_samples,
        first_neighbor_overflow, second_neighbor_overflow,
        first_maximum_neighbor_count, second_maximum_neighbor_count);
}

void fluid_pair_pressure_contact(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue stream,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    std::uint32_t first_maximum_neighbors, void *first_forces,
    const void *second_positions,
    const void *second_velocities, const std::uint32_t *second_count,
    std::uint32_t second_capacity, std::uint32_t second_maximum_neighbors,
    void *second_forces,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &values, std::uint32_t *neighbor_overflow,
    std::uint32_t *maximum_neighbor_count, bool count_neighbors) noexcept {
    fluid_apply_pair_pressure_contact<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        first_maximum_neighbors,
        as<Vec3_0>(first_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        second_maximum_neighbors,
        as<Vec3_0>(second_forces),
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(values), neighbor_overflow, maximum_neighbor_count,
        static_cast<std::uint32_t>(count_neighbors));
}

void fluid_pair_viscosity(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue stream,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    void *first_forces, const void *second_positions,
    const void *second_velocities, const std::uint32_t *second_count,
    std::uint32_t second_capacity, void *second_forces,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &values) noexcept {
    fluid_apply_pair_viscosity<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        as<Vec3_0>(first_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        as<Vec3_0>(second_forces),
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(values));
}

void fluid_miscible_pair_pressure_contact(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue stream,
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
    fluid_apply_miscible_pair_pressure_contact<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        as<Vec3_0>(first_forces), const_cast<float *>(first_density_samples),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        as<Vec3_0>(second_forces), const_cast<float *>(second_density_samples),
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants));
}

void fluid_miscible_pair_viscosity(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue stream,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    void *first_forces, const void *second_positions,
    const void *second_velocities, const std::uint32_t *second_count,
    std::uint32_t second_capacity, void *second_forces,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants) noexcept {
    fluid_apply_miscible_pair_viscosity<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        as<Vec3_0>(first_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        as<Vec3_0>(second_forces),
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants));
}

void fluid_immiscible_interface_normals(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *first_positions,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    const std::uint32_t *first_total_neighbors,
    const std::uint32_t *first_cross_neighbors, float *first_phase_labels,
    void *first_interface_normals, std::uint32_t *first_interface_flags,
    const void *second_positions, const std::uint32_t *second_count,
    std::uint32_t second_capacity,
    const std::uint32_t *second_total_neighbors,
    const std::uint32_t *second_cross_neighbors, float *second_phase_labels,
    void *second_interface_normals, std::uint32_t *second_interface_flags,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants,
    std::uint32_t *first_interface_sample_count,
    std::uint32_t *second_interface_sample_count,
    std::uint32_t *first_under_resolved_interface_count,
    std::uint32_t *second_under_resolved_interface_count,
    std::uint32_t *first_degenerate_interface_count,
    std::uint32_t *second_degenerate_interface_count) noexcept {
    fluid_reconstruct_immiscible_interface_normals<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        const_cast<std::uint32_t *>(first_total_neighbors),
        const_cast<std::uint32_t *>(first_cross_neighbors), first_phase_labels,
        as<Vec3_0>(first_interface_normals), first_interface_flags,
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        const_cast<std::uint32_t *>(second_total_neighbors),
        const_cast<std::uint32_t *>(second_cross_neighbors),
        second_phase_labels, as<Vec3_0>(second_interface_normals),
        second_interface_flags,
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants),
        first_interface_sample_count, second_interface_sample_count,
        first_under_resolved_interface_count,
        second_under_resolved_interface_count,
        first_degenerate_interface_count,
        second_degenerate_interface_count);
}

void fluid_immiscible_interface_curvature(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *first_positions,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    const void *first_interface_normals,
    float *first_interface_curvatures,
    std::uint32_t *first_interface_flags,
    const void *second_positions, const std::uint32_t *second_count,
    std::uint32_t second_capacity, const void *second_interface_normals,
    float *second_interface_curvatures,
    std::uint32_t *second_interface_flags,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants,
    std::uint32_t *first_under_resolved_interface_count,
    std::uint32_t *second_under_resolved_interface_count,
    std::uint32_t *first_degenerate_interface_count,
    std::uint32_t *second_degenerate_interface_count,
    std::uint32_t *first_curvature_clamp_count,
    std::uint32_t *second_curvature_clamp_count) noexcept {
    fluid_reconstruct_immiscible_interface_curvature<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        const_cast<Vec3_0 *>(as<Vec3_0>(first_interface_normals)),
        first_interface_curvatures, first_interface_flags,
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        const_cast<Vec3_0 *>(as<Vec3_0>(second_interface_normals)),
        second_interface_curvatures, second_interface_flags,
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants),
        first_under_resolved_interface_count,
        second_under_resolved_interface_count,
        first_degenerate_interface_count,
        second_degenerate_interface_count, first_curvature_clamp_count,
        second_curvature_clamp_count);
}

void fluid_immiscible_pair_pressure_capillarity(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *first_positions,
    const void *first_velocities, const std::uint32_t *first_count,
    std::uint32_t first_capacity, void *first_forces,
    const float *first_density_samples, const void *first_interface_normals,
    const float *first_interface_curvatures,
    const std::uint32_t *first_interface_flags,
    const void *second_positions, const void *second_velocities,
    const std::uint32_t *second_count, std::uint32_t second_capacity,
    void *second_forces, const float *second_density_samples,
    const void *second_interface_normals,
    const float *second_interface_curvatures,
    const std::uint32_t *second_interface_flags,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants,
    float surface_tension) noexcept {
    fluid_apply_immiscible_pair_pressure_capillarity<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        as<Vec3_0>(first_forces),
        const_cast<float *>(first_density_samples),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_interface_normals)),
        const_cast<float *>(first_interface_curvatures),
        const_cast<std::uint32_t *>(first_interface_flags),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        as<Vec3_0>(second_forces),
        const_cast<float *>(second_density_samples),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_interface_normals)),
        const_cast<float *>(second_interface_curvatures),
        const_cast<std::uint32_t *>(second_interface_flags),
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants),
        surface_tension);
}

void fluid_immiscible_pair_viscosity(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *first_positions,
    const void *first_velocities, const std::uint32_t *first_count,
    std::uint32_t first_capacity, void *first_forces,
    const void *first_interface_normals,
    const std::uint32_t *first_interface_flags,
    const void *second_positions, const void *second_velocities,
    const std::uint32_t *second_count, std::uint32_t second_capacity,
    void *second_forces, const void *second_interface_normals,
    const std::uint32_t *second_interface_flags,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &first_constants,
    const FluidDispatchConstants &second_constants) noexcept {
    fluid_apply_immiscible_pair_viscosity<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(first_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_velocities)),
        const_cast<std::uint32_t *>(first_count), first_capacity,
        as<Vec3_0>(first_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(first_interface_normals)),
        const_cast<std::uint32_t *>(first_interface_flags),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_velocities)),
        const_cast<std::uint32_t *>(second_count), second_capacity,
        as<Vec3_0>(second_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(second_interface_normals)),
        const_cast<std::uint32_t *>(second_interface_flags),
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(identities), total_capacity,
        constants(first_constants), constants(second_constants));
}

void fluid_forces(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &values,
    const float up[3], float *foam_sources, std::uint64_t *keys,
    std::uint32_t *sorted_indices, std::uint32_t *neighbor_overflow,
    std::uint32_t *maximum_neighbor_count) noexcept {
    fluid_compute_forces<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities), count,
        reinterpret_cast<unsigned long long *>(keys),
        sorted_indices, foam, constants(values), vec3(up),
        as<Vec3_0>(accelerations), foam_sources, neighbor_overflow,
        maximum_neighbor_count);
}

void integrate_fluid(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &values,
    void *previous, float *foam_sources) noexcept {
    fluid_integrate<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        as<Vec3_0>(previous), foam, as<Vec3_0>(accelerations),
        foam_sources, count, constants(values), vec3(values.gravity),
        values.timestep);
}

void reserve_fluid_contact_events(device::ExecutionQueue stream,
    std::uint32_t *selected_count, std::uint32_t *offset,
    std::uint32_t *world_count, std::uint32_t *overflow,
    std::uint32_t capacity) noexcept {
    fluid_reserve_contact_events<<<1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        selected_count, offset, world_count, overflow, capacity);
}

void gather_fluid_contact_events(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    std::uint32_t *selected, std::uint32_t *selected_count,
    std::uint32_t *offset, void *samples, std::uint32_t *stable_ids,
    void *body_ids, const void *fluid_id, void *events,
    std::uint32_t capacity) noexcept {
    FluidId_0 id{};
    std::memcpy(&id, fluid_id, sizeof id);
    fluid_gather_contact_events<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        selected, selected_count, offset, as<FluidContactSample_0>(samples),
        stable_ids, as<RigidBodyId_0>(body_ids), id,
        as<ContactEvent_0>(events), capacity);
}

void find_fluid_source_vacancies(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *points, std::uint32_t amount, float spacing,
    const void *positions, const std::uint64_t *keys,
    const std::uint32_t *indices, std::uint32_t capacity,
    float inverse_cell_size, std::uint8_t *vacant) noexcept {
    fluid_source_vacancies<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(points)), amount, spacing,
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        reinterpret_cast<unsigned long long *>(
            const_cast<std::uint64_t *>(keys)),
        const_cast<std::uint32_t *>(indices), capacity, inverse_cell_size,
        vacant);
}

void emit_fluid_source(device::ExecutionQueue stream, const void *points,
    const std::uint8_t *vacant, std::uint32_t amount, float spacing,
    const float velocity[3], void *positions, void *velocities,
    std::uint32_t *ids, float *foam, float *temperatures,
    float initial_temperature, std::uint32_t *count, std::uint32_t capacity,
    std::uint32_t first_spawned, std::uint32_t first_id,
    std::uint32_t *misses) noexcept {
    fluid_source_emit<<<1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(points)),
        const_cast<std::uint8_t *>(vacant), amount, spacing, vec3(velocity),
        as<Vec3_0>(positions), as<Vec3_0>(velocities), ids, foam, temperatures,
        initial_temperature, count, capacity, first_spawned, first_id, misses);
}

void mark_fluid_destroy_flags(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *positions, const void *previous, const std::uint32_t *count,
    std::uint32_t capacity, const void *plane_options, bool combine,
    std::uint8_t *keep) noexcept {
    ParticleDestroyPlaneOptions_0 plane{};
    std::memcpy(&plane, plane_options, sizeof plane);
    fluid_destroy_flags<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)),
        const_cast<std::uint32_t *>(count), capacity, plane, combine, keep);
}

void gather_fluid(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *positions, const void *velocities, const std::uint32_t *ids,
    const float *foam, const float *temperatures, const void *contact_samples,
    const std::uint8_t *contact_flags, bool copy_contacts,
    const std::uint32_t *selected, const std::uint32_t *count,
    std::uint32_t capacity, void *next_positions, void *next_velocities,
    std::uint32_t *next_ids, float *next_foam, float *next_temperatures,
    void *next_contact_samples, std::uint8_t *next_contact_flags) noexcept {
    fluid_gather<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
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
        as<FluidContactSample_0>(next_contact_samples), next_contact_flags);
}

void copy_initial_fluid(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *input, std::uint32_t count, void *positions, void *velocities,
    std::uint32_t *ids, float *foam, float *temperatures,
    std::uint32_t *invalid) noexcept {
    fluid_copy_initial<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<FluidParticle_0 *>(as<FluidParticle_0>(input)), count,
        as<Vec3_0>(positions), as<Vec3_0>(velocities), ids, foam,
        temperatures, invalid);
}

void emit_fluid_spatial_particles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *previous_positions,
    const void *predicted_positions, const std::uint32_t *stable_ids,
    const std::uint32_t *count, std::uint32_t particle_capacity,
    std::uint32_t owner_slot, std::uint32_t owner_generation,
    std::uint32_t material_revision, std::uint32_t leaf_offset,
    std::uint32_t leaf_capacity, std::uint32_t query_offset,
    std::uint32_t query_capacity, std::uint32_t sort_key_base,
    std::uint32_t query_mask, std::uint32_t admission_mask,
    float admission_radius, float query_radius,
    const float path_deviation[3], const float correction_budget[3],
    float epsilon, void *metadata, void *bounds, std::uint8_t *build_flags,
    void *query_bounds, std::uint32_t *query_masks,
    std::uint32_t *excluded_sort_keys, std::uint32_t *query_leaf_sources,
    std::uint32_t *status) noexcept {
    fluid_emit_spatial_particles<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(predicted_positions)),
        const_cast<std::uint32_t *>(stable_ids),
        const_cast<std::uint32_t *>(count), particle_capacity, owner_slot,
        owner_generation, material_revision, leaf_offset, leaf_capacity,
        query_offset, query_capacity, sort_key_base, query_mask,
        admission_mask, admission_radius, query_radius,
        vec3(path_deviation), vec3(correction_budget), epsilon,
        as<FluidSpatialLeafMetadata_0>(metadata),
        as<FluidSpatialAabb_0>(bounds), build_flags,
        as<FluidSpatialAabb_0>(query_bounds), query_masks,
        excluded_sort_keys, query_leaf_sources, status);
}

void emit_fluid_spatial_source_proposals(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *points, std::uint32_t amount,
    float spacing, const std::uint32_t *count,
    std::uint32_t particle_capacity, std::uint32_t first_stable_id,
    std::uint32_t owner_slot, std::uint32_t owner_generation,
    std::uint32_t material_revision, std::uint32_t leaf_offset,
    std::uint32_t leaf_capacity, std::uint32_t query_offset,
    std::uint32_t query_capacity, std::uint32_t sort_key_base,
    std::uint32_t query_mask, std::uint32_t admission_mask,
    float query_radius, const float path_deviation[3],
    const float correction_budget[3], float epsilon, void *metadata,
    void *bounds, std::uint8_t *build_flags, void *query_bounds,
    std::uint32_t *query_masks, std::uint32_t *excluded_sort_keys,
    std::uint32_t *query_leaf_sources, std::uint32_t *status) noexcept {
    fluid_emit_spatial_source_proposals<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(points)), amount, spacing,
        const_cast<std::uint32_t *>(count), particle_capacity,
        first_stable_id, owner_slot, owner_generation, material_revision,
        leaf_offset, leaf_capacity, query_offset, query_capacity,
        sort_key_base, query_mask, admission_mask, query_radius,
        vec3(path_deviation), vec3(correction_budget), epsilon,
        as<FluidSpatialLeafMetadata_0>(metadata),
        as<FluidSpatialAabb_0>(bounds), build_flags,
        as<FluidSpatialAabb_0>(query_bounds), query_masks,
        excluded_sort_keys, query_leaf_sources, status);
}

void apply_fluid_spatial_source_admission(
    device::ExecutionQueue stream, const void *points, std::uint32_t amount,
    const float velocity[3], float initial_temperature,
    std::uint32_t proposal_leaf_offset, std::uint32_t leaf_capacity,
    std::uint32_t proposal_query_offset, std::uint32_t query_capacity,
    void *metadata, std::uint8_t *admission, std::uint32_t *query_masks,
    void *positions, void *velocities, std::uint32_t *stable_ids,
    float *foam, float *temperatures, std::uint32_t *count,
    std::uint32_t particle_capacity, std::uint32_t *accepted,
    std::uint32_t *rejected, std::uint32_t *capacity_misses,
    std::uint32_t *status) noexcept {
    fluid_apply_spatial_source_admission<<<
        1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(points)), amount, vec3(velocity),
        initial_temperature, proposal_leaf_offset, leaf_capacity,
        proposal_query_offset, query_capacity,
        as<FluidSpatialLeafMetadata_0>(metadata), admission, query_masks,
        as<Vec3_0>(positions), as<Vec3_0>(velocities), stable_ids, foam,
        temperatures, count, particle_capacity, accepted, rejected,
        capacity_misses, status);
}

void filter_fluid_spatial_candidates(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    const std::uint32_t *query_leaf_sources, std::uint32_t query_count,
    std::uint32_t query_owner_slot, std::uint32_t query_owner_generation,
    const std::uint32_t *query_stable_ids,
    const std::uint32_t *query_live_count, std::uint32_t query_capacity,
    std::uint32_t target_owner_slot, std::uint32_t target_owner_generation,
    const std::uint32_t *target_stable_ids,
    const std::uint32_t *target_live_count, std::uint32_t target_capacity,
    const std::uint32_t *candidate_counts,
    const std::uint32_t *candidate_sources,
    std::uint32_t candidates_per_query, std::uint32_t *filtered_counts,
    std::uint32_t *filtered_sources, std::uint32_t *status) noexcept {
    fluid_filter_spatial_candidates<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<FluidSpatialLeafMetadata_0 *>(
            as<FluidSpatialLeafMetadata_0>(metadata)),
        const_cast<std::uint8_t *>(admission), leaf_capacity,
        const_cast<std::uint32_t *>(query_leaf_sources), query_count,
        query_owner_slot, query_owner_generation,
        const_cast<std::uint32_t *>(query_stable_ids),
        const_cast<std::uint32_t *>(query_live_count), query_capacity,
        target_owner_slot, target_owner_generation,
        const_cast<std::uint32_t *>(target_stable_ids),
        const_cast<std::uint32_t *>(target_live_count), target_capacity,
        const_cast<std::uint32_t *>(candidate_counts),
        const_cast<std::uint32_t *>(candidate_sources), candidates_per_query,
        filtered_counts, filtered_sources, status);
}

void count_fluid_spatial_neighbors(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    const std::uint32_t *query_leaf_sources, std::uint32_t query_count,
    const void *query_positions, const std::uint32_t *query_live_count,
    std::uint32_t query_capacity, const void *target_positions,
    const std::uint32_t *target_live_count, std::uint32_t target_capacity,
    const std::uint32_t *filtered_counts,
    const std::uint32_t *filtered_sources,
    std::uint32_t candidates_per_query, float support_radius,
    std::uint32_t maximum_neighbors, std::uint32_t *neighbor_counts,
    std::uint32_t *neighbor_overflow,
    std::uint32_t *maximum_neighbor_count, std::uint32_t *status) noexcept {
    fluid_count_spatial_neighbors<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<FluidSpatialLeafMetadata_0 *>(
            as<FluidSpatialLeafMetadata_0>(metadata)),
        const_cast<std::uint8_t *>(admission), leaf_capacity,
        const_cast<std::uint32_t *>(query_leaf_sources), query_count,
        const_cast<Vec3_0 *>(as<Vec3_0>(query_positions)),
        const_cast<std::uint32_t *>(query_live_count), query_capacity,
        const_cast<Vec3_0 *>(as<Vec3_0>(target_positions)),
        const_cast<std::uint32_t *>(target_live_count), target_capacity,
        const_cast<std::uint32_t *>(filtered_counts),
        const_cast<std::uint32_t *>(filtered_sources), candidates_per_query,
        support_radius, maximum_neighbors, neighbor_counts,
        neighbor_overflow, maximum_neighbor_count, status);
}

void predict_fluid_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, const std::uint32_t *count,
    std::uint32_t capacity, const FluidDispatchConstants &values,
    const float gravity[3], float timestep, std::uint32_t profile_version,
    void *accumulated_corrections, std::uint32_t *time_advances,
    std::uint32_t *status) noexcept {
    fluid_realtime_predict<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        as<Vec3_0>(previous_positions),
        const_cast<std::uint32_t *>(count), capacity, constants(values),
        vec3(gravity), timestep, profile_version,
        as<Vec3_0>(accumulated_corrections), time_advances, status);
}

void solve_fluid_realtime_density(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    const std::uint32_t *query_leaf_sources, std::uint32_t query_count,
    const void *positions, const std::uint32_t *count,
    std::uint32_t capacity, const std::uint32_t *filtered_counts,
    const std::uint32_t *filtered_sources,
    std::uint32_t candidates_per_query,
    const FluidDispatchConstants &values, float timestep,
    std::uint32_t profile_version, float *densities, float *lambdas,
    std::uint32_t record_neighbors, std::uint32_t *neighbor_overflow,
    std::uint32_t *maximum_neighbor_count,
    std::uint32_t *status) noexcept {
    fluid_realtime_density_lambda<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<FluidSpatialLeafMetadata_0 *>(
            as<FluidSpatialLeafMetadata_0>(metadata)),
        const_cast<std::uint8_t *>(admission), leaf_capacity,
        const_cast<std::uint32_t *>(query_leaf_sources), query_count,
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<std::uint32_t *>(count), capacity,
        const_cast<std::uint32_t *>(filtered_counts),
        const_cast<std::uint32_t *>(filtered_sources), candidates_per_query,
        constants(values), timestep, profile_version, densities, lambdas,
        record_neighbors, neighbor_overflow, maximum_neighbor_count,
        status);
}

void solve_fluid_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    const std::uint32_t *query_leaf_sources, std::uint32_t query_count,
    const void *positions, const std::uint32_t *count,
    std::uint32_t capacity, const std::uint32_t *filtered_counts,
    const std::uint32_t *filtered_sources,
    std::uint32_t candidates_per_query,
    const FluidDispatchConstants &values, float timestep,
    std::uint32_t profile_version, const float *lambdas,
    const void *accumulated_corrections, void *position_deltas,
    std::uint32_t *status) noexcept {
    fluid_realtime_position_delta<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<FluidSpatialLeafMetadata_0 *>(
            as<FluidSpatialLeafMetadata_0>(metadata)),
        const_cast<std::uint8_t *>(admission), leaf_capacity,
        const_cast<std::uint32_t *>(query_leaf_sources), query_count,
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<std::uint32_t *>(count), capacity,
        const_cast<std::uint32_t *>(filtered_counts),
        const_cast<std::uint32_t *>(filtered_sources), candidates_per_query,
        constants(values), timestep, profile_version,
        const_cast<float *>(lambdas),
        const_cast<Vec3_0 *>(as<Vec3_0>(accumulated_corrections)),
        as<Vec3_0>(position_deltas), status);
}

void apply_fluid_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, void *accumulated_corrections,
    const std::uint32_t *count, std::uint32_t capacity, float timestep,
    std::uint32_t profile_version, const FluidDispatchConstants &values,
    std::uint32_t *status) noexcept {
    fluid_realtime_apply_position_delta<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(position_deltas)),
        as<Vec3_0>(accumulated_corrections),
        const_cast<std::uint32_t *>(count), capacity, timestep,
        profile_version, constants(values), status);
}

void solve_fluid_realtime_viscosity(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *metadata,
    const std::uint8_t *admission, std::uint32_t leaf_capacity,
    const std::uint32_t *query_leaf_sources, std::uint32_t query_count,
    const void *positions, const void *velocities,
    const std::uint32_t *count, std::uint32_t capacity,
    const std::uint32_t *filtered_counts,
    const std::uint32_t *filtered_sources,
    std::uint32_t candidates_per_query,
    const FluidDispatchConstants &values, float timestep,
    std::uint32_t profile_version, const float *densities,
    void *velocity_deltas, std::uint32_t *status) noexcept {
    fluid_realtime_viscosity_delta<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<FluidSpatialLeafMetadata_0 *>(
            as<FluidSpatialLeafMetadata_0>(metadata)),
        const_cast<std::uint8_t *>(admission), leaf_capacity,
        const_cast<std::uint32_t *>(query_leaf_sources), query_count,
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<std::uint32_t *>(count), capacity,
        const_cast<std::uint32_t *>(filtered_counts),
        const_cast<std::uint32_t *>(filtered_sources), candidates_per_query,
        constants(values), timestep, profile_version,
        const_cast<float *>(densities), as<Vec3_0>(velocity_deltas), status);
}

void commit_fluid_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const void *velocity_deltas, const std::uint32_t *count,
    std::uint32_t capacity, const FluidDispatchConstants &values,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    fluid_realtime_commit<<<
        dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        as<Vec3_0>(velocities),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocity_deltas)),
        const_cast<std::uint32_t *>(count), capacity, constants(values),
        timestep, profile_version, status);
}

} // namespace parallel_mater::simulation
