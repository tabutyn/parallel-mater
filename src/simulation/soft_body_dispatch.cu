// SPDX-License-Identifier: MIT
#include "soft_body_dispatch.hpp"

#include "parallel_mater_soft_body_runtime.cu"

#include <cstring>

namespace parallel_mater::simulation {
namespace {

template <typename T> T *as(void *value) noexcept {
    return reinterpret_cast<T *>(value);
}

template <typename T> const T *as(const void *value) noexcept {
    return reinterpret_cast<const T *>(value);
}

Vec3_0 vec3(const float value[3]) noexcept {
    return {value[0], value[1], value[2]};
}

} // namespace

void predict_soft_body_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions, void *velocities,
    void *previous_positions, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, const float gravity[3],
    float velocity_damping, float maximum_speed, float timestep,
    std::uint32_t profile_version, std::uint32_t *time_advances,
    std::uint32_t *status) noexcept {
    soft_body_realtime_predict<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        as<Vec3_0>(previous_positions), correction_used,
        const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths), count, capacity,
        vec3(gravity), velocity_damping, maximum_speed, timestep,
        profile_version, time_advances, status);
}

void project_soft_body_realtime_links(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    const std::uint32_t *offsets, const void *neighbors,
    std::uint32_t count, std::uint32_t capacity, float compliance,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    soft_body_realtime_project_links<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        as<Vec3_0>(position_deltas), correction_used,
        const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths),
        const_cast<unsigned int *>(offsets),
        const_cast<SoftBodyNeighbor_0 *>(
            as<SoftBodyNeighbor_0>(neighbors)),
        count, capacity, compliance, timestep, profile_version, status);
}

void apply_soft_body_realtime_position_delta(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *position_deltas, float *correction_used,
    const float *inverse_masses, const float *characteristic_lengths,
    std::uint32_t count, std::uint32_t capacity, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    soft_body_realtime_apply_position_delta<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(position_deltas)), correction_used,
        const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths), count, capacity,
        timestep, profile_version, status);
}

void project_soft_body_realtime_shape(
    device::ExecutionQueue stream, void *positions, void *corrections,
    const void *rest_positions, const float *inverse_masses,
    const float *characteristic_lengths, float *correction_used,
    std::uint32_t count, std::uint32_t capacity, float movable_mass,
    const float rest_center[3], const void *inverse_rest,
    void *stored_orientation, float stiffness, float timestep,
    std::uint32_t profile_version, std::uint32_t *status) noexcept {
    ShapeMatrix_0 matrix{};
    std::memcpy(&matrix, inverse_rest, sizeof matrix);
    soft_body_realtime_project_shape<<<1U, 128U, 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(corrections),
        const_cast<Vec3_0 *>(as<Vec3_0>(rest_positions)),
        const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths), correction_used,
        count, capacity, movable_mass, vec3(rest_center), matrix,
        as<Quaternion_0>(stored_orientation), stiffness, timestep,
        profile_version, status);
}

void project_soft_body_realtime_contacts(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, void *positions,
    const void *previous_positions, const float *inverse_masses,
    const float *characteristic_lengths, float *correction_used,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_radius, float friction,
    float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    soft_body_realtime_project_contacts<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths), correction_used,
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_normals)),
        const_cast<float *>(contact_offsets),
        const_cast<unsigned char *>(contact_active), count, capacity,
        node_radius, friction, timestep, profile_version, status);
}

void commit_soft_body_realtime(
    device::DispatchSize grid, device::DispatchSize block,
    device::ExecutionQueue stream, const void *positions,
    const void *previous_positions, void *velocities,
    const float *inverse_masses, const float *characteristic_lengths,
    const void *contact_normals, const float *contact_offsets,
    const std::uint8_t *contact_active, std::uint32_t count,
    std::uint32_t capacity, float node_radius, float friction,
    float maximum_speed, float timestep, std::uint32_t profile_version,
    std::uint32_t *status) noexcept {
    soft_body_realtime_commit<<<dim3(grid.x, grid.y, grid.z),
        dim3(block.x, block.y, block.z), 0U,
        static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous_positions)),
        as<Vec3_0>(velocities), const_cast<float *>(inverse_masses),
        const_cast<float *>(characteristic_lengths),
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_normals)),
        const_cast<float *>(contact_offsets),
        const_cast<unsigned char *>(contact_active), count, capacity,
        node_radius, friction, maximum_speed, timestep, profile_version,
        status);
}

void predict_deformable(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, const float gravity[3], float timestep,
    float damping) noexcept {
    deformable_predict<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(previous), as<Vec3_0>(velocities),
        const_cast<float *>(inverse_masses), count, vec3(gravity), timestep,
        damping);
}

void measure_soft_body_momentum(
    device::DispatchSize block, device::ExecutionQueue stream, const void *velocities,
    const float *inverse_masses, std::uint32_t count, void *output) noexcept {
    soft_body_measure_momentum<<<1U, dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<float *>(inverse_masses), count, as<Vec3_0>(output));
}

void restore_soft_body_momentum(
    device::DispatchSize block, device::ExecutionQueue stream, void *velocities,
    const float *inverse_masses, const void *contact_momentum_delta,
    std::uint32_t count, float movable_mass, const void *predicted_momentum,
    const std::uint32_t *dynamic_contact_flag, float maximum_speed) noexcept {
    soft_body_restore_momentum<<<1U, dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(velocities), const_cast<float *>(inverse_masses),
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_momentum_delta)), count,
        movable_mass,
        const_cast<Vec3_0 *>(as<Vec3_0>(predicted_momentum)),
        const_cast<unsigned int *>(dynamic_contact_flag), maximum_speed);
}

void project_soft_body_rest_shape(
    device::ExecutionQueue stream, void *positions, void *corrections,
    const void *rest_positions, const float *inverse_masses,
    std::uint32_t count, float movable_mass, const float rest_center[3],
    const void *inverse_rest, void *stored_orientation, float stiffness,
    float maximum_projection,
    const std::uint32_t *dynamic_contact_flag) noexcept {
    ShapeMatrix_0 matrix{};
    std::memcpy(&matrix, inverse_rest, sizeof matrix);
    soft_body_project_rest_shape<<<1U, 128U, 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(corrections),
        const_cast<Vec3_0 *>(as<Vec3_0>(rest_positions)),
        const_cast<float *>(inverse_masses), count, movable_mass,
        vec3(rest_center), matrix, as<Quaternion_0>(stored_orientation),
        stiffness, maximum_projection,
        const_cast<unsigned int *>(dynamic_contact_flag));
}

void project_deformable_links(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float timestep,
    float maximum_projection_fraction) noexcept {
    deformable_project_links<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)), as<Vec3_0>(scratch),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(offsets),
        const_cast<DeformableNeighbor_0 *>(as<DeformableNeighbor_0>(neighbors)),
        const_cast<unsigned char *>(bond_active), count, timestep,
        maximum_projection_fraction);
}

void project_deformable_links_warp(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const float *minimum_rest_lengths, std::uint32_t count,
    float compliance, float timestep,
    float maximum_projection_fraction) noexcept {
    deformable_project_links_warp<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)), as<Vec3_0>(scratch),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(offsets),
        const_cast<SoftBodyNeighbor_0 *>(as<SoftBodyNeighbor_0>(neighbors)),
        const_cast<float *>(minimum_rest_lengths), count, compliance, timestep,
        maximum_projection_fraction);
}

void finalize_soft_body_velocities(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, float inverse_timestep, float projection_response,
    float maximum_speed) noexcept {
    soft_body_finalize_velocities<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)), as<Vec3_0>(velocities),
        const_cast<float *>(inverse_masses), count, inverse_timestep,
        projection_response, maximum_speed);
}

void apply_soft_body_contact_friction(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, void *positions,
    void *velocities, const float *inverse_masses, void *contact_forces,
    const void *contact_normals, const void *contact_arms,
    void *body_impulses, const void *body_parameters,
    void *contact_momentum_delta, void *accumulated_friction_delta,
    const float *accumulated_normal_delta,
    const std::uint32_t *contact_count, std::uint32_t count,
    float movable_mass, const float gravity[3], float timestep,
    float contact_friction, float maximum_speed) noexcept {
    soft_body_apply_contact_friction<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        as<Vec3_0>(positions), as<Vec3_0>(velocities),
        const_cast<float *>(inverse_masses), as<Vec3_0>(contact_forces),
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_normals)),
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_arms)),
        as<FluidBodyImpulse_0>(body_impulses),
        const_cast<BodyParameters_0 *>(as<BodyParameters_0>(body_parameters)),
        as<Vec3_0>(contact_momentum_delta),
        as<Vec3_0>(accumulated_friction_delta),
        const_cast<float *>(accumulated_normal_delta),
        const_cast<unsigned int *>(contact_count), count, movable_mass,
        vec3(gravity), timestep, contact_friction, maximum_speed);
}

void damp_soft_body_springs(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float damping,
    float maximum_speed) noexcept {
    soft_body_damp_springs<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)), as<Vec3_0>(output),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(offsets),
        const_cast<DeformableNeighbor_0 *>(as<DeformableNeighbor_0>(neighbors)),
        const_cast<unsigned char *>(bond_active), count, damping,
        maximum_speed);
}

void damp_soft_body_springs_warp(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    std::uint32_t count, float damping, float maximum_speed) noexcept {
    soft_body_damp_springs_warp<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)), as<Vec3_0>(output),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(offsets),
        const_cast<SoftBodyNeighbor_0 *>(as<SoftBodyNeighbor_0>(neighbors)),
        count, damping, maximum_speed);
}

void update_soft_body_surface(
    device::DispatchSize grid, device::DispatchSize block, device::ExecutionQueue stream, const void *positions,
    const void *rest_positions, const void *surface_rest_positions,
    const void *bindings, void *surface_positions,
    std::uint32_t surface_vertex_count) noexcept {
    soft_body_update_surface<<<dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), 0U, static_cast<cudaStream_t>(stream.native_handle)>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(rest_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(surface_rest_positions)),
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        as<Vec3_0>(surface_positions), surface_vertex_count);
}

} // namespace parallel_mater::simulation
