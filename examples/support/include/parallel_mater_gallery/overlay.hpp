// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/gallery_context.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

void draw_timing_overlay(std::vector<std::uint32_t> &rgba,
                         std::uint32_t width, std::uint32_t height,
                         const WorldStepTimings &timings);

void draw_smoke_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &physics,
                               const RendererTimings &renderer,
                               const WorldStatistics &statistics,
                               std::uint32_t capacity);

void draw_cloth_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &timings);
void draw_soft_body_timing_overlay(std::vector<std::uint32_t> &rgba,
                                   std::uint32_t width, std::uint32_t height,
                                   const WorldStepTimings &timings);

void draw_fluid_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &physics,
                               const RendererTimings &renderer,
                               const WorldStatistics &statistics,
                               std::uint32_t capacity);

[[nodiscard]] bool draw_rigid_contact_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, RigidContactDeviceView contacts, Camera camera,
    std::string &error);

struct ClothDebugOptions {
    bool normals{};
    bool rigid_contact_forces{};
    bool fluid_contact_forces{};
    bool wireframe{};
    bool bonds{};
};

// Shared cloth diagnostics. Data comes directly from ClothDeviceView so any
// client renderer can reproduce the gallery's Z/X/C/V inspection modes.
// Surface normals/wireframe follow fracture topology; bonds show only live
// physical constraints, and force vectors remain anchored to physical nodes.
[[nodiscard]] bool draw_cloth_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, ClothDeviceView cloth, Camera camera,
    ClothDebugOptions options, std::string &error);

[[nodiscard]] bool draw_soft_body_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, SoftBodyDeviceView body, Camera camera,
    ClothDebugOptions options, std::string &error);

struct PhysicsDebugVisualizationOptions {
    bool contact_normals{};
    bool rigid_forces{};
    bool fluid_forces{};
    bool velocities{};
};

[[nodiscard]] bool draw_rope_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width, std::uint32_t height,
    RopeDeviceView rope, Camera camera, std::string &error);

// Draws host-side state and force vectors captured by the opt-in World debug
// API. Rendering remains a client concern and does not enter the physics API.
void draw_physics_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, PhysicsDebugFrameView frame, Camera camera,
    PhysicsDebugVisualizationOptions options);

// Visualizes cell-centered fields exposed by SmokeDeviceView. The renderer is
// deliberately outside World: the API supplies solver state while clients
// choose slices, normalization, and color maps.
[[nodiscard]] bool draw_smoke_grid_debug_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, SmokeDeviceView smoke, Camera camera,
    SmokeDebugMode mode, std::string &error);

// Fluid loads the Blender-authored Flow scene. Gallery navigation belongs to
// examples, not World.
void draw_context_overlay(std::vector<std::uint32_t> &rgba,
                          std::uint32_t width, std::uint32_t height,
                          GalleryContext selection);

void draw_count_overlay(std::vector<std::uint32_t> &rgba,
                        std::uint32_t width, std::uint32_t height,
                        GalleryContext context, const std::string &value,
                        bool invalid);

} // namespace parallel_mater::gallery
