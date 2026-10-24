// SPDX-License-Identifier: MIT
#include "liquid_coupling_dispatch.hpp"

#include "parallel_mater_liquid_coupling_runtime.cu"

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

void project_liquid_couplings(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const LiquidCouplingParticleBuffers &liquids,
    const LiquidCouplingPartnerBuffers &partners,
    const LiquidCouplingConstraintBuffers &constraints,
    const LiquidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    liquid_coupling_project<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(liquids.positions),
        mutable_pointer(as<Vec3_0>(liquids.previous_positions)),
        as<Vec3_0>(liquids.velocities), mutable_pointer(liquids.inverse_masses),
        liquids.temperatures, mutable_pointer(liquids.generations), liquids.count,
        as<Vec3_0>(partners.positions),
        mutable_pointer(as<Vec3_0>(partners.previous_positions)),
        as<Vec3_0>(partners.velocities),
        mutable_pointer(partners.inverse_masses),
        as<Vec3_0>(partners.angular_velocities),
        mutable_pointer(as<Vec3_0>(partners.inverse_inertias)),
        mutable_pointer(partners.temperatures),
        mutable_pointer(partners.generations), partners.count,
        mutable_pointer(constraints.liquids),
        mutable_pointer(constraints.partners),
        mutable_pointer(constraints.liquid_generations),
        mutable_pointer(constraints.partner_generations),
        mutable_pointer(constraints.families), mutable_pointer(constraints.modes),
        mutable_pointer(as<Vec3_0>(constraints.normals)),
        mutable_pointer(as<Vec3_0>(constraints.partner_arms)),
        mutable_pointer(constraints.contact_distances),
        mutable_pointer(constraints.frictions),
        mutable_pointer(constraints.responses),
        mutable_pointer(constraints.transfer_masses),
        mutable_pointer(constraints.material_equal),
        mutable_pointer(constraints.phase_capacity),
        mutable_pointer(constraints.active), constraints.count,
        characteristic_length, maximum_speed, timestep, profile_version,
        sweep, as<Vec3_0>(scratch.liquid_position_deltas),
        as<Vec3_0>(scratch.partner_position_deltas),
        as<Vec3_0>(scratch.liquid_impulses),
        as<Vec3_0>(scratch.partner_angular_impulses),
        scratch.liquid_temperature_deltas, scratch.phase_transfer_masses,
        status);
}

void apply_liquid_coupling_particles(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const LiquidCouplingParticleBuffers &liquids,
    const LiquidCouplingConstraintBuffers &constraints,
    const LiquidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    liquid_coupling_apply_particles<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(liquids.positions), as<Vec3_0>(liquids.velocities),
        mutable_pointer(liquids.inverse_masses), liquids.temperatures,
        liquids.correction_used, mutable_pointer(liquids.generations),
        liquids.count, mutable_pointer(constraints.liquids),
        mutable_pointer(constraints.liquid_generations),
        mutable_pointer(constraints.active), constraints.count,
        as<Vec3_0>(scratch.liquid_position_deltas),
        as<Vec3_0>(scratch.liquid_impulses),
        scratch.liquid_temperature_deltas, characteristic_length,
        maximum_speed, timestep, profile_version, sweep, status);
}

void apply_liquid_coupling_partners(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream,
    const LiquidCouplingPartnerBuffers &partners,
    const LiquidCouplingConstraintBuffers &constraints,
    const LiquidCouplingScratchBuffers &scratch,
    float characteristic_length, float maximum_speed,
    float maximum_angular_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t sweep,
    std::uint32_t *status) noexcept {
    liquid_coupling_apply_partners<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(partners.positions), as<Vec3_0>(partners.velocities),
        mutable_pointer(partners.inverse_masses),
        as<Vec3_0>(partners.angular_velocities),
        mutable_pointer(as<Vec3_0>(partners.inverse_inertias)),
        partners.correction_used, mutable_pointer(partners.generations),
        partners.count, mutable_pointer(constraints.partners),
        mutable_pointer(constraints.partner_generations),
        mutable_pointer(constraints.active), constraints.count,
        as<Vec3_0>(scratch.partner_position_deltas),
        as<Vec3_0>(scratch.liquid_impulses),
        as<Vec3_0>(scratch.partner_angular_impulses), characteristic_length,
        maximum_speed, maximum_angular_speed, timestep, profile_version,
        sweep, status);
}

} // namespace parallel_mater::simulation
