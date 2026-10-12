// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace parallel_mater::simulation {

void predict_deformable(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, const float gravity[3], float timestep,
    float damping) noexcept;

void measure_soft_body_momentum(
    dim3 block, cudaStream_t stream, const void *velocities,
    const float *inverse_masses, std::uint32_t count, void *output) noexcept;

void restore_soft_body_momentum(
    dim3 block, cudaStream_t stream, void *velocities,
    const float *inverse_masses, const void *contact_momentum_delta,
    std::uint32_t count, float movable_mass, const void *predicted_momentum,
    const std::uint32_t *dynamic_contact_flag, float maximum_speed) noexcept;

void project_soft_body_rest_shape(
    cudaStream_t stream, void *positions, void *corrections,
    const void *rest_positions, const float *inverse_masses,
    std::uint32_t count, float movable_mass, const float rest_center[3],
    const void *inverse_rest, void *stored_orientation, float stiffness,
    float maximum_projection,
    const std::uint32_t *dynamic_contact_flag) noexcept;

void project_deformable_links(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float timestep,
    float maximum_projection_fraction) noexcept;

void project_deformable_links_warp(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const float *minimum_rest_lengths, std::uint32_t count,
    float compliance, float timestep,
    float maximum_projection_fraction) noexcept;

void finalize_soft_body_velocities(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, float inverse_timestep, float projection_response,
    float maximum_speed) noexcept;

void apply_soft_body_contact_friction(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *velocities, const float *inverse_masses, void *contact_forces,
    const void *contact_normals, const void *contact_arms,
    void *body_impulses, const void *body_parameters,
    void *contact_momentum_delta, void *accumulated_friction_delta,
    const float *accumulated_normal_delta,
    const std::uint32_t *contact_count, std::uint32_t count,
    float movable_mass, const float gravity[3], float timestep,
    float contact_friction, float maximum_speed) noexcept;

void damp_soft_body_springs(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float damping,
    float maximum_speed) noexcept;

void damp_soft_body_springs_warp(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    std::uint32_t count, float damping, float maximum_speed) noexcept;

void update_soft_body_surface(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *rest_positions, const void *surface_rest_positions,
    const void *bindings, void *surface_positions,
    std::uint32_t surface_vertex_count) noexcept;

} // namespace parallel_mater::simulation
