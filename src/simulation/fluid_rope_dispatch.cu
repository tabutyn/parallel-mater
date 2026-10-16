// SPDX-License-Identifier: MIT
#include "fluid_rope_dispatch.hpp"

#include "parallel_mater_fluid_rope_runtime.cu"

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

} // namespace

void detect_fluid_rope_contacts(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *particle_positions, void *particle_velocities,
    const std::uint32_t *particle_count, float particle_mass,
    const void *rope_positions, const void *rope_velocities,
    std::uint32_t rope_count, float distance, float friction,
    void *node_impulses, std::uint32_t *contact_count,
    float *maximum_penetration, bool transfer_impulse) noexcept {
    fluid_rope_contacts<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(particle_positions), as<Vec3_0>(particle_velocities),
        const_cast<std::uint32_t *>(particle_count), particle_mass,
        const_cast<Vec3_0 *>(as<Vec3_0>(rope_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(rope_velocities)), rope_count,
        distance, friction, as<Vec3_0>(node_impulses), contact_count,
        maximum_penetration, transfer_impulse);
}

void apply_fluid_rope_impulses(device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream,
    void *rope_velocities, void *fluid_contact_forces,
    std::uint32_t rope_count, float rope_mass, float rope_maximum_speed,
    bool first_attached, bool last_attached, const void *node_impulses,
    float maximum_acceleration, float timestep,
    float frame_inverse_timestep) noexcept {
    fluid_rope_apply<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(rope_velocities), as<Vec3_0>(fluid_contact_forces),
        rope_count, rope_mass, rope_maximum_speed, first_attached,
        last_attached, const_cast<Vec3_0 *>(as<Vec3_0>(node_impulses)),
        maximum_acceleration, timestep, frame_inverse_timestep);
}

} // namespace parallel_mater::simulation
