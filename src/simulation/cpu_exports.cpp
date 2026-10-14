// SPDX-License-Identifier: MIT
#include "exports.hpp"
#include "geometry_storage.hpp"
#include "../device/cpu_dispatch.hpp"

#include "parallel_mater_queries_runtime.cpp"

namespace parallel_mater::simulation {
namespace {

template <typename... Pointers>
[[nodiscard]] bool all_present(Pointers... pointers) noexcept {
    return ((pointers != nullptr) && ...);
}

[[nodiscard]] dim3 blocks_for(std::uint32_t count) noexcept {
    constexpr std::uint32_t block_size = 256U;
    return dim3{((count == 0U ? 1U : count) + block_size - 1U) / block_size};
}

template <typename T> T *mutable_pointer(const T *value) noexcept {
    return const_cast<T *>(value);
}

} // namespace

cudaError_t query_rigid_hit_box(
    cudaStream_t, const HitBox &box, const void *body_parameters,
    const void *body_states, const void *meshes, std::uint32_t body_count,
    std::uint8_t *matches) noexcept {
    if (body_count != 0U &&
        !all_present(body_parameters, body_states, meshes, matches))
        return cudaErrorInvalidValue;
    if (body_count == 0U) return cudaSuccess;
    HitBox_0 converted{{box.center.x, box.center.y, box.center.z},
        {box.orientation.x, box.orientation.y, box.orientation.z,
         box.orientation.w},
        {box.half_extents.x, box.half_extents.y, box.half_extents.z}};
    EntryPointParams_6 args{converted,
        reinterpret_cast<BodyParameters_0 *>(const_cast<void *>(body_parameters)),
        reinterpret_cast<RigidBodyState_0 *>(const_cast<void *>(body_states)),
        reinterpret_cast<TriangleMeshResource_0 *>(const_cast<void *>(meshes)),
        body_count, matches};
    device::dispatch_cpu(blocks_for(body_count), dim3{256U},
                         query_rigid_hit_box_kernel_Thread, &args);
    return cudaSuccess;
}

cudaError_t query_particle_hit_box(
    cudaStream_t, const HitBox &box, const Vec3 *positions,
    std::uint32_t particle_count, std::uint8_t *matches) noexcept {
    if (particle_count != 0U && !all_present(positions, matches))
        return cudaErrorInvalidValue;
    if (particle_count == 0U) return cudaSuccess;
    HitBox_0 converted{{box.center.x, box.center.y, box.center.z},
        {box.orientation.x, box.orientation.y, box.orientation.z,
         box.orientation.w},
        {box.half_extents.x, box.half_extents.y, box.half_extents.z}};
    EntryPointParams_7 args{converted,
        reinterpret_cast<Vec3_0 *>(mutable_pointer(positions)),
        particle_count, matches};
    device::dispatch_cpu(blocks_for(particle_count), dim3{256U},
                         query_particle_hit_box_kernel_Thread, &args);
    return cudaSuccess;
}

PmResult export_fluid_particles(
    cudaStream_t, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *accelerations, const float *temperatures, const float *foam,
    const std::uint32_t *stable_ids, std::uint32_t count,
    std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmFluidParticleOutput *destination) noexcept {
    static_assert(sizeof(Vec3) == sizeof(Vec3_0));
    static_assert(sizeof(PmDomainOutputHeader) == sizeof(PortableDomainOutputHeader_0));
    static_assert(sizeof(PmFluidParticleOutput) == sizeof(PortableParticleState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, accelerations, temperatures, foam,
                     stable_ids, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    EntryPointParams_0 args{reinterpret_cast<Vec3_0 *>(mutable_pointer(positions)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(velocities)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(accelerations)),
        mutable_pointer(temperatures), mutable_pointer(foam),
        mutable_pointer(stable_ids), count, capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableParticleState_0 *>(destination)};
    device::dispatch_cpu(blocks_for(count), dim3{256U},
                         export_fluid_particles_Thread, &args);
    return PM_SUCCESS;
}

PmResult export_cloth_vertices(
    cudaStream_t, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *rigid_forces, const Vec3 *liquid_forces,
    const Vec3 *soft_body_forces, const Vec3 *rope_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmClothVertexState *destination) noexcept {
    static_assert(sizeof(PmDomainOutputHeader) == sizeof(PortableDomainOutputHeader_0));
    static_assert(sizeof(PmClothVertexState) == sizeof(PortableClothVertexState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, rigid_forces, liquid_forces,
                     soft_body_forces, rope_forces, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    EntryPointParams_1 args{reinterpret_cast<Vec3_0 *>(mutable_pointer(positions)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(velocities)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(rigid_forces)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(liquid_forces)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(soft_body_forces)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(rope_forces)), count,
        capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableClothVertexState_0 *>(destination)};
    device::dispatch_cpu(blocks_for(count), dim3{256U},
                         export_cloth_vertices_Thread, &args);
    return PM_SUCCESS;
}

PmResult export_surface_positions(
    cudaStream_t, const Vec3 *positions, std::uint32_t count,
    std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmFloat3 *destination) noexcept {
    static_assert(sizeof(Vec3) == sizeof(PmFloat3));
    if (header == nullptr ||
        (count != 0U && !all_present(positions, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    EntryPointParams_2 args{reinterpret_cast<Vec3_0 *>(mutable_pointer(positions)),
        count, capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<Vec3_0 *>(destination)};
    device::dispatch_cpu(blocks_for(count), dim3{256U},
                         export_surface_positions_Thread, &args);
    return PM_SUCCESS;
}

PmResult export_rope_nodes(
    cudaStream_t, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *constraint_forces, const Vec3 *contact_forces,
    const Vec3 *liquid_forces, const Vec3 *soft_body_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmRopeNodeState *destination) noexcept {
    static_assert(sizeof(PmRopeNodeState) == sizeof(PortableRopeNodeState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, constraint_forces, contact_forces,
                     liquid_forces, soft_body_forces, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    EntryPointParams_3 args{reinterpret_cast<Vec3_0 *>(mutable_pointer(positions)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(velocities)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(constraint_forces)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(contact_forces)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(liquid_forces)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(soft_body_forces)), count,
        capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableRopeNodeState_0 *>(destination)};
    device::dispatch_cpu(blocks_for(count), dim3{256U},
                         export_rope_nodes_Thread, &args);
    return PM_SUCCESS;
}

PmResult export_soft_body_nodes(
    cudaStream_t, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *rigid_forces, const Vec3 *cloth_forces,
    const Vec3 *liquid_forces, const Vec3 *rope_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmSoftBodyNodeState *destination) noexcept {
    static_assert(sizeof(PmSoftBodyNodeState) == sizeof(PortableSoftBodyNodeState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, rigid_forces, cloth_forces,
                     liquid_forces, rope_forces, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    EntryPointParams_4 args{reinterpret_cast<Vec3_0 *>(mutable_pointer(positions)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(velocities)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(rigid_forces)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(cloth_forces)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(liquid_forces)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(rope_forces)), count,
        capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableSoftBodyNodeState_0 *>(destination)};
    device::dispatch_cpu(blocks_for(count), dim3{256U},
                         export_soft_body_nodes_Thread, &args);
    return PM_SUCCESS;
}

PmResult export_smoke_particles(
    cudaStream_t, const Vec3 *positions, const Vec3 *velocities,
    const float *ages, const float *number_densities, const float *pressures,
    const Vec3 *vorticities, std::uint32_t count, std::uint32_t capacity,
    std::uint64_t revision, PmDomainOutputHeader *header,
    PmSmokeParticleState *destination) noexcept {
    static_assert(sizeof(PmSmokeParticleState) == sizeof(PortableSmokeParticleState_0));
    if (header == nullptr || (count != 0U &&
        !all_present(positions, velocities, ages, number_densities, pressures,
                     vorticities, destination)))
        return PM_ERROR_INVALID_ARGUMENT;
    EntryPointParams_5 args{reinterpret_cast<Vec3_0 *>(mutable_pointer(positions)),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(velocities)),
        mutable_pointer(ages), mutable_pointer(number_densities),
        mutable_pointer(pressures),
        reinterpret_cast<Vec3_0 *>(mutable_pointer(vorticities)), count,
        capacity, revision,
        reinterpret_cast<PortableDomainOutputHeader_0 *>(header),
        reinterpret_cast<PortableSmokeParticleState_0 *>(destination)};
    device::dispatch_cpu(blocks_for(count), dim3{256U},
                         export_smoke_particles_Thread, &args);
    return PM_SUCCESS;
}

} // namespace parallel_mater::simulation
