// SPDX-License-Identifier: MIT
#pragma once

#include <simulation/runtime.hpp>
#include "constraint_layout.hpp"

#include "device/device.hpp"

#include <cstdint>

namespace parallel_mater::simulation {

// Host storage mirrored by AvbdBody in shaders/geometry_constraints.slang.
// It is scheduling state only; the constraint equations live in Slang.
struct GeometryConstraintBody {
    RigidBodyState target{};
    Quaternion current_orientation{};
    avbd::Vector6 displacement{};
    avbd::Block inertia{};
    std::uint32_t contact_head{UINT32_MAX};
    std::uint32_t joint_head{UINT32_MAX};
    std::uint32_t color{UINT32_MAX};
    std::uint32_t parent{};
    std::uint32_t impact{};
    Vec3 previous_velocity{};
    std::uint32_t history_valid{};
};

PmResult geometry_constraints_occupancy(
    int *resident_blocks, std::uint32_t block_size) noexcept;

PmResult prepare_geometry_constraints(device::ExecutionQueue stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *predicted, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    RigidBodyId *ids, GeometryConstraintBody *bodies, void *schedule,
    float timestep, Vec3 gravity, std::uint32_t base_iterations,
    std::uint32_t requested_iterations,
    std::uint32_t grid_blocks) noexcept;

PmResult solve_geometry_constraints(device::ExecutionQueue stream,
    const void *parameters, const RigidBodyState *previous,
    RigidBodyState *states, std::uint32_t body_count, void *manifolds,
    std::uint32_t *active_pairs, std::uint32_t *active_pair_count,
    void *constraints, std::uint32_t constraint_capacity,
    GeometryConstraintBody *bodies, void *schedule, float timestep,
    std::uint32_t grid_blocks) noexcept;

} // namespace parallel_mater::simulation
