// SPDX-License-Identifier: MIT
#pragma once

#include "geometry_storage.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace parallel_mater::simulation {

cudaError_t geometry_constraints_occupancy(
    int *resident_blocks, std::uint32_t block_size) noexcept;

cudaError_t prepare_geometry_constraints(cudaStream_t stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *predicted, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    RigidBodyId *ids, GeometryConstraintBody *bodies, void *schedule,
    float timestep, Vec3 gravity, std::uint32_t base_iterations,
    std::uint32_t requested_iterations,
    std::uint32_t grid_blocks) noexcept;

cudaError_t solve_geometry_constraints(cudaStream_t stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *states, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t grid_blocks) noexcept;

} // namespace parallel_mater::simulation
