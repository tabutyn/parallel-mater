// SPDX-License-Identifier: MIT
#include "rope_coupling_dispatch.hpp"

#include "parallel_mater_cloth_rope_runtime.cu"

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}
template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

} // namespace

void sample_rope_cloth_anchor(device::ExecutionQueue stream, void *anchor_positions,
    void *anchor_velocities, void *anchor_impulses,
    float *anchor_inverse_masses, std::uint32_t end, const void *positions,
    const void *velocities, const float *inverse_masses,
    std::uint32_t vertex, float effective_mass) noexcept {
    rope_cloth_sample_anchor<<<1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(anchor_positions), as<Vec3_0>(anchor_velocities),
        as<Vec3_0>(anchor_impulses), anchor_inverse_masses, end,
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<float *>(inverse_masses), vertex, effective_mass);
}

void apply_rope_cloth_anchor(device::ExecutionQueue stream, void *rope_positions,
    void *rope_velocities, std::uint32_t rope_count, void *anchor_positions,
    void *anchor_velocities, std::uint32_t end, void *positions,
    void *velocities, void *forces, const float *inverse_masses,
    std::uint32_t vertex, float maximum_acceleration, float timestep,
    float frame_inverse_timestep) noexcept {
    rope_cloth_apply_anchor<<<1U, 1U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(rope_positions), as<Vec3_0>(rope_velocities), rope_count,
        as<Vec3_0>(anchor_positions), as<Vec3_0>(anchor_velocities), end,
        as<Vec3_0>(positions), as<Vec3_0>(velocities), as<Vec3_0>(forces),
        const_cast<float *>(inverse_masses), vertex, maximum_acceleration,
        timestep, frame_inverse_timestep);
}

} // namespace parallel_mater::simulation
