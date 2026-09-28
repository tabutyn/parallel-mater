// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/camera_controller.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace parallel_mater::gallery {

enum class GalleryContext : std::uint8_t {
    rigid_body,
    dump,
    fluid,
    fluid_rigid,
    peg_paint,
    cloth,
    cloth_tear,
    cloth_paint,
    water_cloth,
    soft_body,
    soft_body_rigid,
    soft_body_cloth,
};

enum class GallerySceneSource : std::uint8_t {
    default_scene,
    procedural_dump,
    fluid,
    fluid_rigid,
    peg_paint,
    cloth,
    cloth_tear,
    cloth_paint,
    water_cloth,
    soft_body,
    soft_body_rigid,
    soft_body_cloth,
};

enum class GalleryControlPolicy : std::uint8_t {
    rigid_gravity,
    dump_rotation,
    none,
    peg_gravity,
    cloth_gravity,
};

enum class GalleryCountKind : std::uint8_t {
    none,
    dump_spheres,
    fluid_particles,
};

struct GalleryColor {
    std::uint8_t red{};
    std::uint8_t green{};
    std::uint8_t blue{};
    std::uint8_t alpha{255U};
};

struct GalleryEntry {
    GalleryContext context{};
    GallerySceneSource source{};
    GalleryControlPolicy controls{};
    GalleryCountKind count_kind{};
    std::string_view command_line_option{};
    std::string_view name{};
    std::string_view help{};
    GalleryColor background{};
    GalleryColor icon{};
    CameraPreset camera{};
    bool has_fluid{};
    bool has_cloth{};
    std::uint32_t minimum_count{};
    std::uint32_t maximum_count{};
    bool has_soft_body{};
};

inline constexpr std::array gallery_entries{
    GalleryEntry{GalleryContext::rigid_body, GallerySceneSource::default_scene,
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none, {},
        "RIGID BODY", "AVAILABLE", {48, 55, 63, 235}, {170, 176, 184},
        {}, false, false, 0U, 0U},
    GalleryEntry{GalleryContext::dump, GallerySceneSource::procedural_dump,
        GalleryControlPolicy::dump_rotation, GalleryCountKind::dump_spheres, {},
        "DUMP", "AVAILABLE  P EDITS SPHERES", {62, 38, 22, 235},
        {245, 130, 45}, {.target = {0.5F, 2.2F, 0.0F}}, false, false,
        10U, 1'000U},
    GalleryEntry{GalleryContext::fluid, GallerySceneSource::fluid,
        GalleryControlPolicy::none, GalleryCountKind::fluid_particles, "--fluid",
        "FLUID", "P CAP  V PARTICLES  R RESET", {12, 42, 65, 235},
        {35, 150, 255}, {.target = {-2.0F, -1.0F, -2.0F},
                         .distance_scale = 1.6F}, true, false, 100U, 100'000U},
    GalleryEntry{GalleryContext::fluid_rigid, GallerySceneSource::fluid_rigid,
        GalleryControlPolicy::none, GalleryCountKind::fluid_particles,
        "--fluid-rigid", "FLUID RIGID", "64 FREE SPHERES  P CAP",
        {25, 52, 64, 235}, {35, 190, 230},
        {.target = {-2.0F, -1.0F, -2.0F}, .distance_scale = 1.6F},
        true, false, 100U, 100'000U},
    GalleryEntry{GalleryContext::peg_paint, GallerySceneSource::peg_paint,
        GalleryControlPolicy::peg_gravity, GalleryCountKind::fluid_particles,
        "--peg-paint", "PEG PAINT", "ARROWS GRAVITY  P CAP",
        {44, 28, 61, 235}, {42, 145, 255},
        {.target = {0.0F, -0.34F, 0.0F}, .distance_scale = 0.34F,
         .pitch = 0.72F}, true, false, 100U, 100'000U},
    GalleryEntry{GalleryContext::cloth, GallerySceneSource::cloth,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none, "--cloth",
        "CLOTH", "ARROWS GRAVITY  R RESET", {40, 42, 58, 235},
        {236, 188, 96}, {.target = {0.0F, 1.1F, 1.2F},
                         .distance_scale = 0.55F, .pitch = 0.42F},
        false, true, 0U, 0U},
    GalleryEntry{GalleryContext::cloth_tear, GallerySceneSource::cloth_tear,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--cloth-tear", "CLOTH TEAR", "ARROWS GRAVITY  R RESET",
        {53, 37, 48, 235}, {255, 126, 111},
        {.target = {0.0F, 1.1F, 1.2F}, .distance_scale = 0.55F,
         .pitch = 0.42F}, false, true, 0U, 0U},
    GalleryEntry{GalleryContext::cloth_paint, GallerySceneSource::cloth_paint,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--cloth-paint", "CLOTH PAINT", "ARROWS GRAVITY  R RESET",
        {28, 49, 55, 235}, {65, 177, 240},
        {.target = {0.0F, 1.1F, 1.2F}, .distance_scale = 0.55F,
         .pitch = 0.42F}, false, true, 0U, 0U},
    GalleryEntry{GalleryContext::water_cloth, GallerySceneSource::water_cloth,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::fluid_particles,
        "--water-cloth", "WATER CLOTH",
        "PRESSURE SKIN  ARROWS GRAVITY  P CAP", {20, 50, 68, 235},
        {45, 190, 235}, {.target = {0.0F, 1.1F, 1.2F},
                         .distance_scale = 0.55F, .pitch = 0.42F},
        true, true, 100U, 100'000U},
    GalleryEntry{GalleryContext::soft_body, GallerySceneSource::soft_body,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--soft-body", "SOFT BODY", "ARROWS GRAVITY  R RESET",
        {49, 35, 67, 235}, {166, 102, 255},
        {.target = {0.0F, 0.7F, 0.0F}, .distance_scale = 0.72F,
         .pitch = 0.35F}, false, false, 0U, 0U, true},
    GalleryEntry{GalleryContext::soft_body_rigid,
        GallerySceneSource::soft_body_rigid,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--soft-body-rigid", "SOFT BODY RIGID",
        "TWO-WAY IMPACTS  ARROWS GRAVITY  R RESET",
        {46, 38, 64, 235}, {190, 118, 255},
        {.target = {0.0F, 0.7F, 0.0F}, .distance_scale = 0.72F,
         .pitch = 0.35F}, false, false, 0U, 0U, true},
    GalleryEntry{GalleryContext::soft_body_cloth,
        GallerySceneSource::soft_body_cloth,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--soft-body-cloth", "SOFT BODY CLOTH",
        "INTACT BRIDGE  TEARABLE CURTAIN  ARROWS GRAVITY",
        {42, 40, 62, 235}, {174, 128, 245},
        {.target = {0.0F, 0.6F, -1.5F}, .distance_scale = 0.78F,
         .pitch = 0.70F}, false, true, 0U, 0U, true},
};

[[nodiscard]] constexpr const GalleryEntry &gallery_entry(
    GalleryContext context) noexcept {
    for (const GalleryEntry &entry : gallery_entries)
        if (entry.context == context) return entry;
    return gallery_entries.front();
}

[[nodiscard]] constexpr std::size_t gallery_context_index(
    GalleryContext context) noexcept {
    for (std::size_t index = 0U; index < gallery_entries.size(); ++index)
        if (gallery_entries[index].context == context) return index;
    return 0U;
}

[[nodiscard]] constexpr bool is_cloth_context(GalleryContext context) noexcept {
    return gallery_entry(context).has_cloth;
}

[[nodiscard]] constexpr bool is_fluid_context(GalleryContext context) noexcept {
    return gallery_entry(context).has_fluid;
}

[[nodiscard]] constexpr bool is_soft_body_context(
    GalleryContext context) noexcept {
    return gallery_entry(context).has_soft_body;
}

} // namespace parallel_mater::gallery
