// SPDX-License-Identifier: MIT
#include "deformable_smoke_coupling_dispatch.hpp"

#include "parallel_mater_deformable_smoke_coupling_runtime.cu"

namespace parallel_mater::simulation {
namespace {
template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}
template <typename T> T *mutable_pointer(const T *value) noexcept {
    return const_cast<T *>(value);
}
} // namespace

void project_deformable_smoke_couplings(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const DeformableSmokeCouplingEndpointBuffers &first,
    const DeformableSmokeCouplingEndpointBuffers &second,
    const DeformableSmokeCouplingConstraintBuffers &constraints,
    const DeformableSmokeCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    deformable_smoke_coupling_project<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(first.positions),
        mutable_pointer(as<Vec3_0>(first.previous_positions)),
        as<Vec3_0>(first.velocities), mutable_pointer(first.inverse_masses),
        mutable_pointer(first.generations), first.count,
        as<Vec3_0>(second.positions),
        mutable_pointer(as<Vec3_0>(second.previous_positions)),
        as<Vec3_0>(second.velocities), mutable_pointer(second.inverse_masses),
        mutable_pointer(second.generations), second.count,
        mutable_pointer(constraints.first_endpoints),
        mutable_pointer(constraints.second_endpoints),
        mutable_pointer(constraints.first_generations),
        mutable_pointer(constraints.second_generations),
        mutable_pointer(constraints.pairs), mutable_pointer(constraints.modes),
        mutable_pointer(as<Vec3_0>(constraints.normals)),
        mutable_pointer(constraints.contact_distances),
        mutable_pointer(constraints.frictions),
        mutable_pointer(constraints.responses),
        mutable_pointer(constraints.active), constraints.count,
        characteristic_length, maximum_speed, timestep, profile_version,
        sweep, as<Vec3_0>(scratch.first_position_deltas),
        as<Vec3_0>(scratch.second_position_deltas),
        as<Vec3_0>(scratch.first_impulses),
        as<Vec3_0>(scratch.second_impulses), status);
}

void apply_deformable_smoke_coupling_first(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const DeformableSmokeCouplingEndpointBuffers &first,
    const DeformableSmokeCouplingConstraintBuffers &constraints,
    const DeformableSmokeCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    deformable_smoke_coupling_apply_first<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(first.positions),
        mutable_pointer(as<Vec3_0>(first.previous_positions)),
        as<Vec3_0>(first.velocities),
        mutable_pointer(first.inverse_masses), first.correction_used,
        mutable_pointer(first.generations), first.count,
        mutable_pointer(constraints.first_endpoints),
        mutable_pointer(constraints.first_generations),
        mutable_pointer(constraints.active), constraints.count,
        as<Vec3_0>(scratch.first_position_deltas),
        as<Vec3_0>(scratch.first_impulses), characteristic_length,
        maximum_speed, timestep, profile_version, sweep, status);
}

void apply_deformable_smoke_coupling_second(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const DeformableSmokeCouplingEndpointBuffers &second,
    const DeformableSmokeCouplingConstraintBuffers &constraints,
    const DeformableSmokeCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    deformable_smoke_coupling_apply_second<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(second.positions),
        mutable_pointer(as<Vec3_0>(second.previous_positions)),
        as<Vec3_0>(second.velocities),
        mutable_pointer(second.inverse_masses), second.correction_used,
        mutable_pointer(second.generations), second.count,
        mutable_pointer(constraints.second_endpoints),
        mutable_pointer(constraints.second_generations),
        mutable_pointer(constraints.active), constraints.count,
        as<Vec3_0>(scratch.second_position_deltas),
        as<Vec3_0>(scratch.second_impulses), characteristic_length,
        maximum_speed, timestep, profile_version, sweep, status);
}

} // namespace parallel_mater::simulation
