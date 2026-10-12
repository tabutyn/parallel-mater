// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace parallel_mater::simulation {

void constrain_rigid_bodies_with_cloth(
    dim3 grid, dim3 block, cudaStream_t stream, const void *cloth_positions,
    const void *cloth_velocities, const std::uint32_t *cloth_indices,
    const std::uint32_t *vertex_sources, const void *surface_positions,
    const float *inverse_masses, std::uint32_t vertex_count,
    std::uint32_t triangle_count, float thickness, float timestep,
    float contact_friction, const void *body_parameters,
    const void *previous_body_states, void *body_states, const void *meshes,
    const void *body_ids, std::uint32_t body_count,
    std::uint32_t cloth_index, std::uint32_t cloth_generation,
    const void *paint_fields, std::uint32_t paint_field_capacity,
    const void *paint_rules, std::uint32_t paint_rule_capacity,
    void *corrections) noexcept;

void apply_cloth_body_corrections(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *velocities, const float *inverse_masses, std::uint32_t vertex_count,
    const void *corrections, std::uint32_t body_count) noexcept;

} // namespace parallel_mater::simulation
