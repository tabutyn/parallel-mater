// SPDX-License-Identifier: MIT
#include "exports.hpp"

#include "parallel_mater_queries_runtime.cu"

namespace parallel_mater::simulation {

namespace {

[[nodiscard]] cudaStream_t native_stream(
    device::ExecutionQueue queue) noexcept {
    return static_cast<cudaStream_t>(queue.native_handle);
}

[[nodiscard]] PmResult launch_result() noexcept {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) return PM_SUCCESS;
    if (error == cudaErrorMemoryAllocation) return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    if (error == cudaErrorNotSupported) return PM_ERROR_FEATURE_NOT_PRESENT;
    if (error == cudaErrorLaunchFailure || error == cudaErrorIllegalAddress)
        return PM_ERROR_DEVICE_LOST;
    return PM_ERROR_INVALID_STATE;
}

template <typename... Pointers>
[[nodiscard]] bool all_present(Pointers... pointers) noexcept {
    return ((pointers != nullptr) && ...);
}

} // namespace

PmResult query_rigid_hit_box(
    device::ExecutionQueue queue, const HitBox &box, const void *body_parameters,
    const void *body_states, const void *meshes, std::uint32_t body_count,
    std::uint8_t *matches) noexcept {
    if (body_count != 0U &&
        !all_present(body_parameters, body_states, meshes, matches))
        return PM_ERROR_INVALID_ARGUMENT;
    if (body_count == 0U) return PM_SUCCESS;
    const cudaStream_t stream = native_stream(queue);
    HitBox_0 converted{};
    converted.center_0 = {box.center.x, box.center.y, box.center.z};
    converted.orientation_3 = {box.orientation.x, box.orientation.y,
                               box.orientation.z, box.orientation.w};
    converted.half_extents_0 = {
        box.half_extents.x, box.half_extents.y, box.half_extents.z};
    constexpr std::uint32_t block_size = 256U;
    query_rigid_hit_box_kernel<<<
        (body_count + block_size - 1U) / block_size,
        block_size, 0U, stream>>>(
        converted,
        reinterpret_cast<BodyParameters_0 *>(
            const_cast<void *>(body_parameters)),
        reinterpret_cast<RigidBodyState_0 *>(const_cast<void *>(body_states)),
        reinterpret_cast<TriangleMeshResource_0 *>(const_cast<void *>(meshes)),
        body_count, matches);
    return launch_result();
}

PmResult query_particle_hit_box(
    device::ExecutionQueue queue, const HitBox &box, const Vec3 *positions,
    std::uint32_t particle_count, std::uint8_t *matches) noexcept {
    if (particle_count != 0U && !all_present(positions, matches))
        return PM_ERROR_INVALID_ARGUMENT;
    if (particle_count == 0U) return PM_SUCCESS;
    const cudaStream_t stream = native_stream(queue);
    HitBox_0 converted{};
    converted.center_0 = {box.center.x, box.center.y, box.center.z};
    converted.orientation_3 = {box.orientation.x, box.orientation.y,
                               box.orientation.z, box.orientation.w};
    converted.half_extents_0 = {
        box.half_extents.x, box.half_extents.y, box.half_extents.z};
    constexpr std::uint32_t block_size = 256U;
    query_particle_hit_box_kernel<<<
        (particle_count + block_size - 1U) / block_size,
        block_size, 0U, stream>>>(
        converted, reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(positions)),
        particle_count, matches);
    return launch_result();
}

PmResult export_fluid_particles(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *accelerations, const float *temperatures, const float *foam,
    const std::uint32_t *stable_ids, std::uint32_t count,
    std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmFluidParticleOutput *destination) noexcept {
    static_assert(sizeof(Vec3) == sizeof(Vec3_0));
    static_assert(sizeof(PmDomainOutputHeader) ==
                  sizeof(PortableDomainOutputHeader_0));
    static_assert(sizeof(PmFluidParticleOutput) ==
                  sizeof(PortableParticleState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, accelerations, temperatures, foam,
                     stable_ids, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    const cudaStream_t stream = native_stream(queue);
    constexpr std::uint32_t block_size = 256U;
    export_fluid_particles<<<((count == 0U ? 1U : count) + block_size - 1U) / block_size,
                             block_size, 0U, stream>>>(
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(positions)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(velocities)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(accelerations)),
        const_cast<float *>(temperatures), const_cast<float *>(foam),
        const_cast<std::uint32_t *>(stable_ids), count, capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableParticleState_0 *>(destination));
    return launch_result();
}

PmResult export_cloth_vertices(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *rigid_forces, const Vec3 *liquid_forces,
    const Vec3 *soft_body_forces, const Vec3 *rope_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmClothVertexState *destination) noexcept {
    static_assert(sizeof(PmDomainOutputHeader) ==
                  sizeof(PortableDomainOutputHeader_0));
    static_assert(sizeof(PmClothVertexState) ==
                  sizeof(PortableClothVertexState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, rigid_forces, liquid_forces,
                     soft_body_forces, rope_forces, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    const cudaStream_t stream = native_stream(queue);
    constexpr std::uint32_t block_size = 256U;
    export_cloth_vertices<<<((count == 0U ? 1U : count) + block_size - 1U) / block_size,
                            block_size, 0U, stream>>>(
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(positions)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(velocities)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(rigid_forces)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(liquid_forces)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(soft_body_forces)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(rope_forces)), count,
        capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableClothVertexState_0 *>(destination));
    return launch_result();
}

PmResult export_surface_positions(
    device::ExecutionQueue queue, const Vec3 *positions, std::uint32_t count,
    std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmFloat3 *destination) noexcept {
    static_assert(sizeof(Vec3) == sizeof(PmFloat3));
    static_assert(sizeof(PmDomainOutputHeader) ==
                  sizeof(PortableDomainOutputHeader_0));
    if (header == nullptr ||
        (count != 0U && !all_present(positions, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    const cudaStream_t stream = native_stream(queue);
    constexpr std::uint32_t block_size = 256U;
    export_surface_positions<<<
        ((count == 0U ? 1U : count) + block_size - 1U) / block_size,
        block_size, 0U, stream>>>(
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(positions)), count,
        capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<Vec3_0 *>(destination));
    return launch_result();
}

PmResult export_rope_nodes(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *constraint_forces, const Vec3 *contact_forces,
    const Vec3 *liquid_forces, const Vec3 *soft_body_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmRopeNodeState *destination) noexcept {
    static_assert(sizeof(PmDomainOutputHeader) ==
                  sizeof(PortableDomainOutputHeader_0));
    static_assert(sizeof(PmRopeNodeState) == sizeof(PortableRopeNodeState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, constraint_forces, contact_forces,
                     liquid_forces, soft_body_forces, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    const cudaStream_t stream = native_stream(queue);
    constexpr std::uint32_t block_size = 256U;
    export_rope_nodes<<<((count == 0U ? 1U : count) + block_size - 1U) / block_size,
                        block_size, 0U, stream>>>(
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(positions)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(velocities)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(constraint_forces)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(contact_forces)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(liquid_forces)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(soft_body_forces)), count,
        capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableRopeNodeState_0 *>(destination));
    return launch_result();
}

PmResult export_soft_body_nodes(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *rigid_forces, const Vec3 *cloth_forces,
    const Vec3 *liquid_forces, const Vec3 *rope_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmSoftBodyNodeState *destination) noexcept {
    static_assert(sizeof(PmDomainOutputHeader) ==
                  sizeof(PortableDomainOutputHeader_0));
    static_assert(sizeof(PmSoftBodyNodeState) ==
                  sizeof(PortableSoftBodyNodeState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, rigid_forces, cloth_forces,
                     liquid_forces, rope_forces, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    const cudaStream_t stream = native_stream(queue);
    constexpr std::uint32_t block_size = 256U;
    export_soft_body_nodes<<<((count == 0U ? 1U : count) + block_size - 1U) / block_size,
                             block_size, 0U, stream>>>(
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(positions)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(velocities)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(rigid_forces)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(cloth_forces)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(liquid_forces)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(rope_forces)), count,
        capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableSoftBodyNodeState_0 *>(destination));
    return launch_result();
}

PmResult export_smoke_particles(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const float *ages, const float *number_densities, const float *pressures,
    const Vec3 *vorticities, std::uint32_t count, std::uint32_t capacity,
    std::uint64_t revision, PmDomainOutputHeader *header,
    PmSmokeParticleState *destination) noexcept {
    static_assert(sizeof(PmDomainOutputHeader) ==
                  sizeof(PortableDomainOutputHeader_0));
    static_assert(sizeof(PmSmokeParticleState) ==
                  sizeof(PortableSmokeParticleState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, ages, number_densities, pressures,
                     vorticities, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    const cudaStream_t stream = native_stream(queue);
    constexpr std::uint32_t block_size = 256U;
    export_smoke_particles<<<((count == 0U ? 1U : count) + block_size - 1U) / block_size,
                             block_size, 0U, stream>>>(
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(positions)),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(velocities)),
        const_cast<float *>(ages), const_cast<float *>(number_densities),
        const_cast<float *>(pressures),
        reinterpret_cast<Vec3_0 *>(const_cast<Vec3 *>(vorticities)), count,
        capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableSmokeParticleState_0 *>(destination));
    return launch_result();
}

} // namespace parallel_mater::simulation
