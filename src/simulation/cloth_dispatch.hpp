// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace parallel_mater::simulation {

void project_cloth_volume(
    cudaStream_t stream, void *positions, const float *inverse_masses,
    const std::uint32_t *indices, std::uint32_t vertex_count,
    std::uint32_t triangle_count, const std::uint32_t *corner_offsets,
    const std::uint32_t *corner_indices, void *gradients, float target_volume,
    float orientation, float compliance, float timestep, float *lambda,
    bool reset_lambda) noexcept;

void break_cloth_bonds(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *bonds, std::uint8_t *active, std::uint8_t *damage,
    std::uint32_t count, float break_strain, std::uint32_t persistence,
    const void *contact_impulses, float impact_threshold) noexcept;

void limit_cloth_strain(
    cudaStream_t stream, void *positions, void *scratch,
    const float *inverse_masses, const std::uint32_t *offsets,
    const void *neighbors, const std::uint8_t *free_nodes,
    std::uint32_t count) noexcept;

void update_cloth_surface(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const std::uint32_t *source_indices, void *surface_positions,
    std::uint32_t triangle_count) noexcept;

} // namespace parallel_mater::simulation
