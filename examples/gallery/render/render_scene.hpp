// SPDX-License-Identifier: MIT
#pragma once

#include "../scene_builder.hpp"

#include <parallel_mater/types.hpp>

#include <cstdint>
#include <vector>

namespace parallel_mater::gallery {

struct RenderFluidParticle {
    Vec3 position{};
    Vec3 velocity{};
    Vec3 acceleration{};
    float temperature_celsius{};
    float foam{};
    std::uint32_t stable_id{};
};

struct RenderSmokeParticle {
    Vec3 position{};
    Vec3 velocity{};
    float age{};
    float number_density{};
    float pressure{};
    Vec3 vorticity{};
};

// Immutable, backend-neutral frame input. Renderer adapters may upload this
// data through their native drawing API, but cannot borrow simulation internals.
struct RenderSceneFrame {
    std::vector<RigidBodyState> rigid_states;
    std::vector<RigidBodyState> previous_rigid_states;
    std::vector<RenderFluidParticle> fluid_particles;
    std::vector<std::vector<Vec3>> cloth_vertices;
    std::vector<std::vector<Vec3>> cloth_surface_vertices;
    std::vector<std::vector<Vec3>> rope_nodes;
    std::vector<std::vector<Vec3>> soft_body_nodes;
    std::vector<std::vector<Vec3>> soft_body_surface_vertices;
    std::vector<RenderSmokeParticle> smoke_particles;
    std::vector<std::vector<std::uint32_t>> paint_fields;
    float fluid_particle_radius{};
    float fluid_support_radius{};
    float smoke_particle_radius{};
    float smoke_lifetime{};
    std::uint64_t revision{};
};

[[nodiscard]] bool make_render_scene_frame(
    const SceneDefinition &scene, const CommandSceneFrame &source,
    RenderSceneFrame &output, std::string &error);

} // namespace parallel_mater::gallery
