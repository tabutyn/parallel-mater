// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include "../device/device.hpp"
#include "types.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

[[nodiscard]] PmResult query_rigid_hit_box(
    device::ExecutionQueue queue, const HitBox &box, const void *body_parameters,
    const void *body_states, const void *meshes, std::uint32_t body_count,
    std::uint8_t *matches) noexcept;

[[nodiscard]] PmResult query_particle_hit_box(
    device::ExecutionQueue queue, const HitBox &box, const Vec3 *positions,
    std::uint32_t particle_count, std::uint8_t *matches) noexcept;

[[nodiscard]] PmResult export_fluid_particles(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *accelerations, const float *temperatures, const float *foam,
    const std::uint32_t *stable_ids, std::uint32_t count,
    std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmFluidParticleOutput *destination) noexcept;

[[nodiscard]] PmResult export_cloth_vertices(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *rigid_forces, const Vec3 *liquid_forces,
    const Vec3 *soft_body_forces, const Vec3 *rope_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmClothVertexState *destination) noexcept;

[[nodiscard]] PmResult export_surface_positions(
    device::ExecutionQueue queue, const Vec3 *positions, std::uint32_t count,
    std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmFloat3 *destination) noexcept;

[[nodiscard]] PmResult export_rope_nodes(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *constraint_forces, const Vec3 *contact_forces,
    const Vec3 *liquid_forces, const Vec3 *soft_body_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmRopeNodeState *destination) noexcept;

[[nodiscard]] PmResult export_soft_body_nodes(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const Vec3 *rigid_forces, const Vec3 *cloth_forces,
    const Vec3 *liquid_forces, const Vec3 *rope_forces,
    std::uint32_t count, std::uint32_t capacity, std::uint64_t revision,
    PmDomainOutputHeader *header, PmSoftBodyNodeState *destination) noexcept;

[[nodiscard]] PmResult export_smoke_particles(
    device::ExecutionQueue queue, const Vec3 *positions, const Vec3 *velocities,
    const float *ages, const float *number_densities, const float *pressures,
    const Vec3 *vorticities, std::uint32_t count, std::uint32_t capacity,
    std::uint64_t revision, PmDomainOutputHeader *header,
    PmSmokeParticleState *destination) noexcept;

} // namespace parallel_mater::simulation
