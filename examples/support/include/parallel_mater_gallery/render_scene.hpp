// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/scene.hpp>
#include <parallel_mater_scene/command_scene.hpp>

#include <cstdint>
#include <string>
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
    std::vector<RenderFluidParticle> previous_fluid_particles;
    std::vector<std::vector<Vec3>> cloth_vertices;
    std::vector<std::vector<Vec3>> previous_cloth_vertices;
    std::vector<std::vector<Vec3>> cloth_surface_vertices;
    std::vector<std::vector<Vec3>> previous_cloth_surface_vertices;
    std::vector<std::vector<Vec3>> rope_nodes;
    std::vector<std::vector<Vec3>> previous_rope_nodes;
    std::vector<std::vector<Vec3>> soft_body_nodes;
    std::vector<std::vector<Vec3>> previous_soft_body_nodes;
    std::vector<std::vector<Vec3>> soft_body_surface_vertices;
    std::vector<std::vector<Vec3>> previous_soft_body_surface_vertices;
    std::vector<RenderSmokeParticle> smoke_particles;
    std::vector<RenderSmokeParticle> previous_smoke_particles;
    std::vector<std::vector<std::uint32_t>> paint_fields;
    float fluid_particle_radius{};
    float fluid_support_radius{};
    float smoke_particle_radius{};
    float smoke_lifetime{};
    std::uint64_t revision{};
    std::uint64_t previous_revision{};
    bool previous_snapshot_available{};
};

[[nodiscard]] bool make_render_scene_frame(
    const SceneDefinition &scene, const CommandSceneFrame &source,
    RenderSceneFrame &output, std::string &error);

// Presentation owns committed snapshots, independent of solver internals.
// A reset seeds both snapshots with the same state. Advancing carries only the
// prior committed state; render helpers reject mismatched topology or identity.
void reset_render_scene_snapshots(RenderSceneFrame &frame);
void advance_render_scene_snapshots(const RenderSceneFrame &previous,
                                    RenderSceneFrame &current);

[[nodiscard]] RigidBodyState presentation_rigid_state(
    const RenderSceneFrame &frame, std::size_t index, float alpha) noexcept;
[[nodiscard]] Vec3 presentation_fluid_position(
    const RenderSceneFrame &frame, std::size_t index, float alpha) noexcept;
[[nodiscard]] Vec3 presentation_smoke_position(
    const RenderSceneFrame &frame, std::size_t index, float alpha) noexcept;
[[nodiscard]] Vec3 presentation_domain_position(
    const std::vector<std::vector<Vec3>> &previous,
    const std::vector<std::vector<Vec3>> &current,
    std::size_t domain, std::size_t index, float alpha) noexcept;

} // namespace parallel_mater::gallery
