// SPDX-License-Identifier: MIT
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace parallel_mater::simulation {

void detect_fluid_rope_contacts(dim3 grid, dim3 block, cudaStream_t stream,
    void *particle_positions, void *particle_velocities,
    const std::uint32_t *particle_count, float particle_mass,
    const void *rope_positions, const void *rope_velocities,
    std::uint32_t rope_count, float distance, float friction,
    void *node_impulses, std::uint32_t *contact_count,
    float *maximum_penetration, bool transfer_impulse) noexcept;

void apply_fluid_rope_impulses(dim3 grid, dim3 block, cudaStream_t stream,
    void *rope_velocities, void *fluid_contact_forces,
    std::uint32_t rope_count, float rope_mass, float rope_maximum_speed,
    bool first_attached, bool last_attached, const void *node_impulses,
    float maximum_acceleration, float timestep,
    float frame_inverse_timestep) noexcept;

} // namespace parallel_mater::simulation
