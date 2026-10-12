// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace parallel_mater::simulation {

void collide_deformable_with_rigid_geometry(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *velocities, const void *previous, const float *inverse_masses,
    std::uint32_t count, float thickness, float timestep,
    const void *body_parameters, const void *previous_body_states,
    const void *body_states, const void *meshes, std::uint32_t body_count,
    void *impulses, void *contact_forces, void *contact_normals,
    void *contact_arms, void *contact_momentum_delta,
    float *accumulated_normal_delta,
    std::uint32_t *deformable_contact_count,
    std::uint32_t *dynamic_contact_flag, std::uint32_t *contact_flags,
    bool couple_dynamic, bool position_only, bool static_only,
    bool reconstruct_free_velocities, float maximum_speed,
    void *body_position_corrections = nullptr) noexcept;

void calculate_soft_body_surface_contacts(
    dim3 grid, dim3 block, cudaStream_t stream, const void *surface,
    const std::uint32_t *indices, std::uint32_t triangle_count,
    const void *body_parameters, const void *body_states,
    const void *meshes, std::uint32_t body_count, void *corrections) noexcept;

void apply_soft_body_surface_contacts(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    const void *corrections, const std::uint32_t *offsets,
    const void *influences, std::uint32_t node_count) noexcept;

} // namespace parallel_mater::simulation
