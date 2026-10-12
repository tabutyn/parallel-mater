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

void predict_deformable(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, const float gravity[3], float timestep,
    float damping) noexcept {
    deformable_predict<<<grid, block, 0U, stream>>>(
        as<Vec3_0>(positions), as<Vec3_0>(previous), as<Vec3_0>(velocities),
        const_cast<float *>(inverse_masses), count, vec3(gravity), timestep,
        damping);
}

void measure_soft_body_momentum(
    dim3 block, cudaStream_t stream, const void *velocities,
    const float *inverse_masses, std::uint32_t count, void *output) noexcept {
    soft_body_measure_momentum<<<1U, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)),
        const_cast<float *>(inverse_masses), count, as<Vec3_0>(output));
}

void restore_soft_body_momentum(
    dim3 block, cudaStream_t stream, void *velocities,
    const float *inverse_masses, const void *contact_momentum_delta,
    std::uint32_t count, float movable_mass, const void *predicted_momentum,
    const std::uint32_t *dynamic_contact_flag, float maximum_speed) noexcept {
    soft_body_restore_momentum<<<1U, block, 0U, stream>>>(
        as<Vec3_0>(velocities), const_cast<float *>(inverse_masses),
        const_cast<Vec3_0 *>(as<Vec3_0>(contact_momentum_delta)), count,
        movable_mass,
        const_cast<Vec3_0 *>(as<Vec3_0>(predicted_momentum)),
        const_cast<unsigned int *>(dynamic_contact_flag), maximum_speed);
}

void project_soft_body_rest_shape(
    cudaStream_t stream, void *positions, void *corrections,
    const void *rest_positions, const float *inverse_masses,
    std::uint32_t count, float movable_mass, const float rest_center[3],
    const void *inverse_rest, void *stored_orientation, float stiffness,
    float maximum_projection,
    const std::uint32_t *dynamic_contact_flag) noexcept {
    ShapeMatrix_0 matrix{};
    std::memcpy(&matrix, inverse_rest, sizeof matrix);
    soft_body_project_rest_shape<<<1U, 128U, 0U, stream>>>(
        as<Vec3_0>(positions), as<Vec3_0>(corrections),
        const_cast<Vec3_0 *>(as<Vec3_0>(rest_positions)),
        const_cast<float *>(inverse_masses), count, movable_mass,
        vec3(rest_center), matrix, as<Quaternion_0>(stored_orientation),
        stiffness, maximum_projection,
        const_cast<unsigned int *>(dynamic_contact_flag));
}

void project_deformable_links(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float timestep,
    float maximum_projection_fraction) noexcept {
    deformable_project_links<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)), as<Vec3_0>(scratch),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(offsets),
        const_cast<DeformableNeighbor_0 *>(as<DeformableNeighbor_0>(neighbors)),
        const_cast<unsigned char *>(bond_active), count, timestep,
        maximum_projection_fraction);
}

void project_deformable_links_warp(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    void *scratch, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const float *minimum_rest_lengths, std::uint32_t count,
    float compliance, float timestep,
    float maximum_projection_fraction) noexcept {
    deformable_project_links_warp<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)), as<Vec3_0>(scratch),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(offsets),
        const_cast<SoftBodyNeighbor_0 *>(as<SoftBodyNeighbor_0>(neighbors)),
        const_cast<float *>(minimum_rest_lengths), count, compliance, timestep,
        maximum_projection_fraction);
}

void finalize_soft_body_velocities(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *previous, void *velocities, const float *inverse_masses,
    std::uint32_t count, float inverse_timestep, float projection_response,
    float maximum_speed) noexcept {
    soft_body_finalize_velocities<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(previous)), as<Vec3_0>(velocities),
        const_cast<float *>(inverse_masses), count, inverse_timestep,
        projection_response, maximum_speed);
}

void apply_soft_body_contact_friction(
    dim3 grid, dim3 block, cudaStream_t stream, void *positions,
    void *velocities, const float *inverse_masses, void *contact_forces,
    const void *contact_normals, const void *contact_arms,
    void *body_impulses, const void *body_parameters,
    void *contact_momentum_delta, void *accumulated_friction_delta,
    const float *accumulated_normal_delta,
    const std::uint32_t *contact_count, std::uint32_t count,
    float movable_mass, const float gravity[3], float timestep,
    float contact_friction, float maximum_speed) noexcept {
    soft_body_apply_contact_friction<<<grid, block, 0U, stream>>>(
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
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    const std::uint8_t *bond_active, std::uint32_t count, float damping,
    float maximum_speed) noexcept {
    soft_body_damp_springs<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)), as<Vec3_0>(output),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(offsets),
        const_cast<DeformableNeighbor_0 *>(as<DeformableNeighbor_0>(neighbors)),
        const_cast<unsigned char *>(bond_active), count, damping,
        maximum_speed);
}

void damp_soft_body_springs_warp(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *velocities, void *output, const float *inverse_masses,
    const std::uint32_t *offsets, const void *neighbors,
    std::uint32_t count, float damping, float maximum_speed) noexcept {
    soft_body_damp_springs_warp<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(velocities)), as<Vec3_0>(output),
        const_cast<float *>(inverse_masses),
        const_cast<unsigned int *>(offsets),
        const_cast<SoftBodyNeighbor_0 *>(as<SoftBodyNeighbor_0>(neighbors)),
        count, damping, maximum_speed);
}

void update_soft_body_surface(
    dim3 grid, dim3 block, cudaStream_t stream, const void *positions,
    const void *rest_positions, const void *surface_rest_positions,
    const void *bindings, void *surface_positions,
    std::uint32_t surface_vertex_count) noexcept {
    soft_body_update_surface<<<grid, block, 0U, stream>>>(
        const_cast<Vec3_0 *>(as<Vec3_0>(positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(rest_positions)),
        const_cast<Vec3_0 *>(as<Vec3_0>(surface_rest_positions)),
        const_cast<SoftBodySurfaceBinding_0 *>(
            as<SoftBodySurfaceBinding_0>(bindings)),
        as<Vec3_0>(surface_positions), surface_vertex_count);
}

} // namespace parallel_mater::simulation
