// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

struct LiquidCouplingParticleBuffers {
    void *positions{};
    const void *previous_positions{};
    void *velocities{};
    const float *inverse_masses{};
    float *temperatures{};
    float *correction_used{};
    const std::uint32_t *generations{};
    std::uint32_t count{};
};

struct LiquidCouplingPartnerBuffers {
    void *positions{};
    const void *previous_positions{};
    void *velocities{};
    const float *inverse_masses{};
    void *angular_velocities{};
    const void *inverse_inertias{};
    const float *temperatures{};
    float *correction_used{};
    const std::uint32_t *generations{};
    std::uint32_t count{};
};

struct LiquidCouplingConstraintBuffers {
    const std::uint32_t *liquids{};
    const std::uint32_t *partners{};
    const std::uint32_t *liquid_generations{};
    const std::uint32_t *partner_generations{};
    const std::uint32_t *families{};
    const std::uint32_t *modes{};
    const void *normals{};
    const void *partner_arms{};
    const float *contact_distances{};
    const float *frictions{};
    const float *responses{};
    const float *transfer_masses{};
    const std::uint8_t *material_equal{};
    const std::uint8_t *phase_capacity{};
    const std::uint8_t *active{};
    std::uint32_t count{};
};

struct LiquidCouplingScratchBuffers {
    void *liquid_position_deltas{};
    void *partner_position_deltas{};
    void *liquid_impulses{};
    void *partner_angular_impulses{};
    float *liquid_temperature_deltas{};
    float *phase_transfer_masses{};
};

void project_liquid_couplings(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const LiquidCouplingParticleBuffers &liquids,
    const LiquidCouplingPartnerBuffers &partners,
    const LiquidCouplingConstraintBuffers &constraints,
    const LiquidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept;

void apply_liquid_coupling_particles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const LiquidCouplingParticleBuffers &liquids,
    const LiquidCouplingConstraintBuffers &constraints,
    const LiquidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept;

void apply_liquid_coupling_partners(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const LiquidCouplingPartnerBuffers &partners,
    const LiquidCouplingConstraintBuffers &constraints,
    const LiquidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed,
    float maximum_angular_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept;

} // namespace parallel_mater::simulation
