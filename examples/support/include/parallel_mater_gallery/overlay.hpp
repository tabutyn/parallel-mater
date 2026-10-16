// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/frame_timing.hpp>
#include <parallel_mater_gallery/renderer.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace parallel_mater::gallery {

struct SceneDefinition;
struct CommandSceneFrame;

void draw_frame_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const FrameTimingCollector &timings);

[[nodiscard]] bool draw_rigid_contact_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const std::vector<PmContactEvent> &contacts,
    Camera camera, std::string &error);

struct ClothDebugOptions {
    bool normals{};
    bool rigid_contact_forces{};
    bool fluid_contact_forces{};
    bool wireframe{};
    bool bonds{};
};

[[nodiscard]] bool draw_cloth_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const SceneDefinition &scene,
    const CommandSceneFrame &frame, std::uint32_t cloth, Camera camera,
    ClothDebugOptions options, std::string &error);

[[nodiscard]] bool draw_soft_body_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const SceneDefinition &scene,
    const CommandSceneFrame &frame, std::uint32_t body, Camera camera,
    ClothDebugOptions options, std::string &error);

struct PhysicsDebugVisualizationOptions {
    bool contact_normals{};
    bool rigid_forces{};
    bool fluid_forces{};
    bool velocities{};
};

[[nodiscard]] bool draw_rope_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const CommandSceneFrame &frame,
    std::uint32_t rope, Camera camera, std::string &error);

void draw_physics_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const CommandSceneFrame &frame, Camera camera,
    PhysicsDebugVisualizationOptions options);

[[nodiscard]] bool draw_smoke_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const CommandSceneFrame &frame, Camera camera,
    SmokeDebugMode mode, std::string &error);

// Fluid loads the Blender-authored Flow scene. Gallery navigation belongs to
// examples, not the simulation runtime.
void draw_context_overlay(std::vector<std::uint32_t> &rgba,
                          std::uint32_t width, std::uint32_t height,
                          GalleryContext selection);

void draw_backend_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, std::span<const std::string> names,
    std::span<const std::string> details, std::size_t selection);

void draw_count_overlay(std::vector<std::uint32_t> &rgba,
                        std::uint32_t width, std::uint32_t height,
                        GalleryContext context, const std::string &value,
                        bool invalid);

void draw_brick_settings_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const std::array<std::string, 3> &values,
    std::size_t selected, bool invalid);

void draw_status_overlay(std::vector<std::uint32_t> &rgba,
                         std::uint32_t width, std::uint32_t height,
                         std::string_view message, bool failure);

} // namespace parallel_mater::gallery
