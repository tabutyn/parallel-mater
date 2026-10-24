// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

struct RigidCouplingEndpointBuffers {
    void *positions{};
    const void *previous_positions{};
    void *velocities{};
    const float *inverse_masses{};
    float *correction_used{};
    const std::uint32_t *generations{};
    std::uint32_t count{};
};

struct RigidCouplingBodyBuffers {
    const void *parameters{};
    const void *previous_states{};
    void *states{};
    float *correction_used{};
    const std::uint32_t *generations{};
    std::uint32_t count{};
};

struct RigidCouplingConstraintBuffers {
    const std::uint32_t *endpoints{};
    const std::uint32_t *bodies{};
    const std::uint32_t *endpoint_generations{};
    const std::uint32_t *body_generations{};
    const std::uint32_t *families{};
    const std::uint32_t *modes{};
    const void *normals{};
    const void *body_local_points{};
    const float *contact_distances{};
    const float *frictions{};
    const float *responses{};
    const std::uint8_t *active{};
    std::uint32_t count{};
};

struct RigidCouplingScratchBuffers {
    void *endpoint_position_deltas{};
    void *body_position_deltas{};
    void *endpoint_impulses{};
    void *body_angular_impulses{};
};

void project_rigid_couplings(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const RigidCouplingEndpointBuffers &endpoints,
    const RigidCouplingBodyBuffers &bodies,
    const RigidCouplingConstraintBuffers &constraints,
    const RigidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept;

void apply_rigid_coupling_endpoints(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const RigidCouplingEndpointBuffers &endpoints,
    const RigidCouplingConstraintBuffers &constraints,
    const RigidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept;

void apply_rigid_coupling_bodies(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const RigidCouplingBodyBuffers &bodies,
    const RigidCouplingConstraintBuffers &constraints,
    const RigidCouplingScratchBuffers &scratch,
    float characteristic_length, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept;

} // namespace parallel_mater::simulation
