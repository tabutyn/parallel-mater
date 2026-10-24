// SPDX-License-Identifier: MIT
#pragma once

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

struct DeformableSmokeCouplingEndpointBuffers {
    void *positions{};
    const void *previous_positions{};
    void *velocities{};
    const float *inverse_masses{};
    float *correction_used{};
    const std::uint32_t *generations{};
    std::uint32_t count{};
};

struct DeformableSmokeCouplingConstraintBuffers {
    const std::uint32_t *first_endpoints{};
    const std::uint32_t *second_endpoints{};
    const std::uint32_t *first_generations{};
    const std::uint32_t *second_generations{};
    const std::uint32_t *pairs{};
    const std::uint32_t *modes{};
    const void *normals{};
    const float *contact_distances{};
    const float *frictions{};
    const float *responses{};
    const std::uint8_t *active{};
    std::uint32_t count{};
};

struct DeformableSmokeCouplingScratchBuffers {
    void *first_position_deltas{};
    void *second_position_deltas{};
    void *first_impulses{};
    void *second_impulses{};
};

void project_deformable_smoke_couplings(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const DeformableSmokeCouplingEndpointBuffers &first,
    const DeformableSmokeCouplingEndpointBuffers &second,
    const DeformableSmokeCouplingConstraintBuffers &constraints,
    const DeformableSmokeCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept;

void apply_deformable_smoke_coupling_first(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const DeformableSmokeCouplingEndpointBuffers &first,
    const DeformableSmokeCouplingConstraintBuffers &constraints,
    const DeformableSmokeCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept;

void apply_deformable_smoke_coupling_second(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const DeformableSmokeCouplingEndpointBuffers &second,
    const DeformableSmokeCouplingConstraintBuffers &constraints,
    const DeformableSmokeCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept;

} // namespace parallel_mater::simulation
