// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace parallel_mater::simulation {

void detect_soft_body_cloth_contacts(dim3 grid, dim3 block,
    cudaStream_t stream, const void *positions, const void *previous,
    const void *velocities, const float *inverse_masses,
    std::uint32_t count, const void *cloth_positions,
    const void *cloth_previous, const void *cloth_velocities,
    const float *cloth_inverse_masses, const std::uint32_t *indices,
    const void *surface, std::uint32_t triangle_count, float distance,
    float friction, void *contacts, std::uint32_t *contact_counts) noexcept;

void apply_soft_body_cloth_contacts_to_soft_body(dim3 grid, dim3 block,
    cudaStream_t stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    const std::uint32_t *contact_counts, float timestep,
    float maximum_speed) noexcept;

void apply_soft_body_cloth_contacts_to_cloth(dim3 grid, dim3 block,
    cudaStream_t stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, std::uint32_t count, const void *contacts,
    std::uint32_t contact_count, const std::uint32_t *contact_counts,
    float timestep) noexcept;

} // namespace parallel_mater::simulation
