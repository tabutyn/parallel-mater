// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

void sample_rope_cloth_anchor(device::ExecutionQueue stream, void *anchor_positions,
    void *anchor_velocities, void *anchor_impulses,
    float *anchor_inverse_masses, std::uint32_t end, const void *positions,
    const void *velocities, const float *inverse_masses,
    std::uint32_t vertex, float effective_mass) noexcept;

void apply_rope_cloth_anchor(device::ExecutionQueue stream, void *rope_positions,
    void *rope_velocities, std::uint32_t rope_count, void *anchor_positions,
    void *anchor_velocities, std::uint32_t end, void *positions,
    void *velocities, void *forces, const float *inverse_masses,
    std::uint32_t vertex, float maximum_acceleration, float timestep,
    float frame_inverse_timestep) noexcept;

void sample_rope_soft_body_anchor(device::ExecutionQueue stream,
    void *anchor_positions, void *anchor_velocities, void *anchor_impulses,
    float *anchor_inverse_masses, std::uint32_t end, const void *surface,
    const void *velocities, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    const float offset[3]) noexcept;

void scatter_rope_soft_body_anchor(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *anchor_positions,
    const void *anchor_impulses, std::uint32_t end,
    const void *node_positions, std::uint32_t node_count,
    float support_radius, float contact_support_radius,
    const std::uint32_t *contact_count, const std::uint32_t *indices,
    const void *bindings, std::uint32_t triangle, const float weights[3],
    void *node_impulses) noexcept;

void apply_rope_soft_body_impulses(device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities, void *forces,
    const float *inverse_masses, const void *impulses, std::uint32_t count,
    float maximum_acceleration, float maximum_speed, float timestep,
    float frame_inverse_timestep) noexcept;

} // namespace parallel_mater::simulation
