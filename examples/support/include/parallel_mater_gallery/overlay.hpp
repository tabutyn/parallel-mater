// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/renderer.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

struct SceneDefinition;
struct CommandSceneFrame;

struct GalleryKernelTiming {
    float total_milliseconds{};
    std::uint32_t launch_count{};
};

// Presentation-only timing data. The application fills this from public
// timestamp queries; it is independent of private simulation implementation.
struct GalleryStepTimings {
    std::uint64_t frame_index{};
    bool available{};
    GalleryKernelTiming rigid_integration{};
    GalleryKernelTiming rigid_world_bounds{};
    GalleryKernelTiming rigid_pair_filter{};
    GalleryKernelTiming rigid_pair_compaction{};
    GalleryKernelTiming rigid_leaf_pair_generation{};
    GalleryKernelTiming rigid_contact_evaluation{};
    GalleryKernelTiming rigid_contact_generation{};
    GalleryKernelTiming rigid_contact_solve{};
    GalleryKernelTiming rigid_input_clear{};
    GalleryKernelTiming fluid_spawn{};
    GalleryKernelTiming fluid_neighbor_sort{};
    GalleryKernelTiming fluid_neighbor_forces{};
    GalleryKernelTiming fluid_integration{};
    GalleryKernelTiming fluid_static_contacts{};
    GalleryKernelTiming fluid_body_index{};
    GalleryKernelTiming fluid_moving_contacts{};
    GalleryKernelTiming fluid_cloth_contacts{};
    GalleryKernelTiming fluid_contact_events{};
    GalleryKernelTiming fluid_outflow_compaction{};
    GalleryKernelTiming fluid_smoke_exchange{};
    float total_gpu_milliseconds{};
    GalleryKernelTiming cloth_prediction{};
    GalleryKernelTiming cloth_constraints{};
    GalleryKernelTiming cloth_contacts{};
    GalleryKernelTiming soft_body_prediction{};
    GalleryKernelTiming soft_body_constraints{};
    GalleryKernelTiming soft_body_contacts{};
    GalleryKernelTiming soft_body_cloth_contacts{};
    GalleryKernelTiming fluid_soft_body_contacts{};
    GalleryKernelTiming fluid_rope_contacts{};
    GalleryKernelTiming rope_solve{};
    GalleryKernelTiming rope_soft_body_contacts{};
    GalleryKernelTiming smoke_grid{};
    GalleryKernelTiming smoke_advection{};
    GalleryKernelTiming smoke_emission{};
};

struct GalleryStatistics {
    std::uint64_t frame_index{};
    std::uint32_t particle_count{};
    std::uint32_t smoke_particle_count{};
    std::uint64_t emitted_smoke_particle_count{};
    std::uint64_t boiled_particle_count{};
    std::uint32_t rigid_body_count{};
    std::uint32_t rigid_contact_maximum_passes{};
    std::uint32_t contact_count{};
    std::uint32_t contact_overflow_count{};
    std::uint64_t emitted_particle_count{};
    std::uint64_t destroyed_particle_count{};
    std::uint64_t spawn_capacity_miss_count{};
    std::uint32_t soft_body_count{};
    std::uint32_t rope_count{};
    std::uint32_t fluid_soft_body_contact_count{};
    float maximum_fluid_soft_body_penetration{};
    std::uint32_t fluid_rope_contact_count{};
    float maximum_fluid_rope_penetration{};
};

void draw_timing_overlay(std::vector<std::uint32_t> &rgba,
                         std::uint32_t width, std::uint32_t height,
                         const GalleryStepTimings &timings,
                         const GalleryStatistics &statistics);

void draw_smoke_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const GalleryStepTimings &physics,
                               const RendererTimings &renderer,
                               const GalleryStatistics &statistics,
                               std::uint32_t capacity);

void draw_cloth_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const GalleryStepTimings &timings);
void draw_soft_body_timing_overlay(std::vector<std::uint32_t> &rgba,
                                   std::uint32_t width, std::uint32_t height,
                                   const GalleryStepTimings &timings);

void draw_fluid_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const GalleryStepTimings &physics,
                               const RendererTimings &renderer,
                               const GalleryStatistics &statistics,
                               std::uint32_t capacity);

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

void draw_count_overlay(std::vector<std::uint32_t> &rgba,
                        std::uint32_t width, std::uint32_t height,
                        GalleryContext context, const std::string &value,
                        bool invalid);

void draw_brick_settings_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, const std::array<std::string, 3> &values,
    std::size_t selected, bool invalid);

} // namespace parallel_mater::gallery
