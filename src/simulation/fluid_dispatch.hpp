// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

struct FluidDispatchConstants {
    float timestep{};
    float gravity[3]{};
    std::uint32_t count{};
    float particle_radius{};
    float support_radius{};
    float repulsion{};
    float viscosity{};
    float normal_damping{};
    float velocity_damping{};
    float maximum_speed{};
    float maximum_pair_acceleration{};
    float rest_density{};
    std::uint32_t maximum_neighbors{};
};

void fluid_cell_keys(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *positions, std::uint32_t *count, std::uint32_t capacity,
    float inverse_radius, std::uint64_t *keys,
    std::uint32_t *sorted_indices) noexcept;

void fluid_pair_cell_keys(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *first_positions,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    const void *second_positions, const std::uint32_t *second_count,
    std::uint32_t second_capacity, float inverse_radius,
    std::uint64_t *keys, std::uint32_t *identities) noexcept;

void fluid_pair_neighbor_counts(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue stream,
    const void *first_positions, const std::uint32_t *first_count,
    std::uint32_t first_capacity, std::uint32_t first_maximum_neighbors,
    const void *second_positions, const std::uint32_t *second_count,
    std::uint32_t second_capacity, std::uint32_t second_maximum_neighbors,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity, float support_radius,
    std::uint32_t *first_total_neighbors,
    std::uint32_t *first_cross_neighbors,
    std::uint32_t *second_total_neighbors,
    std::uint32_t *second_cross_neighbors,
    std::uint32_t *first_neighbor_overflow,
    std::uint32_t *second_neighbor_overflow,
    std::uint32_t *first_maximum_neighbor_count,
    std::uint32_t *second_maximum_neighbor_count) noexcept;

void fluid_pair_pressure_contact(device::DispatchSize grid,
    device::DispatchSize block, device::ExecutionQueue stream,
    const void *first_positions, const void *first_velocities,
    const std::uint32_t *first_count, std::uint32_t first_capacity,
    void *first_forces, const void *second_positions,
    const void *second_velocities, const std::uint32_t *second_count,
    std::uint32_t second_capacity, void *second_forces,
    const std::uint64_t *keys, const std::uint32_t *identities,
    std::uint32_t total_capacity,
    const FluidDispatchConstants &constants) noexcept;

void fluid_forces(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &constants,
    const float up[3], float *foam_sources, std::uint64_t *keys,
    std::uint32_t *sorted_indices, std::uint32_t *neighbor_overflow,
    std::uint32_t *maximum_neighbor_count) noexcept;

void integrate_fluid(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *positions, void *velocities, void *accelerations, float *foam,
    std::uint32_t *count, const FluidDispatchConstants &constants,
    void *previous, float *foam_sources) noexcept;

void reserve_fluid_contact_events(device::ExecutionQueue stream,
    std::uint32_t *selected_count, std::uint32_t *offset,
    std::uint32_t *world_count, std::uint32_t *overflow,
    std::uint32_t capacity) noexcept;

void gather_fluid_contact_events(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    std::uint32_t *selected, std::uint32_t *selected_count,
    std::uint32_t *offset, void *samples, std::uint32_t *stable_ids,
    void *body_ids, const void *fluid_id, void *events,
    std::uint32_t capacity) noexcept;

void find_fluid_source_vacancies(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *points, std::uint32_t amount, float spacing,
    const void *positions, const std::uint64_t *keys,
    const std::uint32_t *indices, std::uint32_t capacity,
    float inverse_cell_size, std::uint8_t *vacant) noexcept;

void emit_fluid_source(device::ExecutionQueue stream, const void *points,
    const std::uint8_t *vacant, std::uint32_t amount, float spacing,
    const float velocity[3], void *positions, void *velocities,
    std::uint32_t *ids, float *foam, float *temperatures,
    float initial_temperature, std::uint32_t *count, std::uint32_t capacity,
    std::uint32_t first_spawned, std::uint32_t first_id,
    std::uint32_t *misses) noexcept;

void mark_fluid_destroy_flags(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *positions, const void *previous, const std::uint32_t *count,
    std::uint32_t capacity, const void *plane_options, bool combine,
    std::uint8_t *keep) noexcept;

void gather_fluid(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *positions, const void *velocities, const std::uint32_t *ids,
    const float *foam, const float *temperatures, const void *contact_samples,
    const std::uint8_t *contact_flags, bool copy_contacts,
    const std::uint32_t *selected, const std::uint32_t *count,
    std::uint32_t capacity, void *next_positions, void *next_velocities,
    std::uint32_t *next_ids, float *next_foam, float *next_temperatures,
    void *next_contact_samples, std::uint8_t *next_contact_flags) noexcept;

void copy_initial_fluid(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    const void *input, std::uint32_t count, void *positions, void *velocities,
    std::uint32_t *ids, float *foam, float *temperatures,
    std::uint32_t *invalid) noexcept;

} // namespace parallel_mater::simulation
