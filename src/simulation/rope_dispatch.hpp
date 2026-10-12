// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>

namespace parallel_mater::simulation {

// Host descriptor mirrored by RopeData in shaders/types.slang. It contains
// resource bindings only; all rope physics remains in rope.slang.
struct RopeDispatchData {
    RopeOptions options{};
    std::uint32_t count{};
    Vec3 *positions{}, *previous{}, *velocities{}, *constraint_forces{};
    Vec3 *contact_forces{}, *fluid_contact_forces{};
    Vec3 *soft_body_contact_forces{};
    Vec3 *soft_anchor_positions{}, *soft_anchor_velocities{};
    Vec3 *soft_anchor_impulses{};
    float *soft_anchor_inverse_masses{};
    bool soft_first{}, soft_last{};
    Vec3 *directions{}, *scratch{}, *body_translation{}, *body_rotation{};
    Vec3 *normals{}, *normals2{};
    float *rest{}, *lambda{};
    std::uint32_t *solid_hint{};
    std::uint32_t body_capacity{};
    std::uint32_t first_attachment_contact_skip{2U};
    std::uint32_t last_attachment_contact_skip{2U};
};

struct RopeSoftTargetDispatch {
    const Vec3 *surface{}, *previous_surface{}, *velocities{};
    const Vec3 *bounds{};
    const void *tree{};
    const std::uint32_t *order{}, *indices{};
    const void *bindings{};
    const float *inverse_masses{};
    Vec3 *node_impulses{};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    std::uint32_t triangle_count{}, surface_vertex_count{};
    float distance{}, friction{};
    float orientation{1.0F};
    bool first_anchor{}, last_anchor{};
};

void advance_rope(cudaStream_t stream, const RopeDispatchData &rope,
    float timestep, Vec3 gravity, int first, int last,
    const void *body_parameters, void *body_states,
    const void *previous_body_states, const void *meshes,
    std::uint32_t body_count, bool first_substep,
    const RopeSoftTargetDispatch &soft_a,
    const RopeSoftTargetDispatch &soft_b) noexcept;

} // namespace parallel_mater::simulation
