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
    constraint_fixed,
    constraint_point,
    constraint_hinge,
    constraint_slider,
    constraint_piston,
    constraint_generic,
    constraint_generic_spring,
    constraint_motor,
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
    soft_body_fluid,
    rope,
    rope_fluid,
    rope_soft_body,
    rope_cloth,
    smoke,
    smoke_water,
    smoke_soft_body,
    smoke_cloth,
    smoke_rope,
};

enum class GallerySceneSource : std::uint8_t {
    default_scene,
    constraint_fixed,
    constraint_point,
    constraint_hinge,
    constraint_slider,
    constraint_piston,
    constraint_generic,
    constraint_generic_spring,
    constraint_motor,
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
    soft_body_fluid,
    rope,
    rope_fluid,
    rope_soft_body,
    rope_cloth,
    smoke,
    smoke_water,
    smoke_soft_body,
    smoke_cloth,
    smoke_rope,
};

enum class GalleryControlPolicy : std::uint8_t {
    rigid_gravity,
    constraint_toggle_gravity,
    tank_motor,
    dump_rotation,
    none,
    peg_gravity,
    cloth_gravity,
};

[[nodiscard]] constexpr bool uses_rigid_gravity(
    GalleryControlPolicy policy) noexcept {
    return policy == GalleryControlPolicy::rigid_gravity ||
           policy == GalleryControlPolicy::constraint_toggle_gravity;
}

[[nodiscard]] constexpr bool toggles_constraint(
    GalleryControlPolicy policy) noexcept {
    return policy == GalleryControlPolicy::constraint_toggle_gravity;
}

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
    bool has_rope{};
};

inline constexpr std::array gallery_entries{
    GalleryEntry{GalleryContext::rigid_body, GallerySceneSource::default_scene,
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none, {},
        "RIGID BODY", "AVAILABLE", {48, 55, 63, 235}, {170, 176, 184},
        {}, false, false, 0U, 0U},
    GalleryEntry{GalleryContext::constraint_fixed,
        GallerySceneSource::constraint_fixed,
        GalleryControlPolicy::constraint_toggle_gravity, GalleryCountKind::none,
        "--constraint-fixed", "CONSTRAINT: FIXED",
        "ARROWS GRAVITY  SPACE GLUE / RELEASE", {51, 34, 38, 235},
        {244, 98, 70},
        {.target = {0.0F, 0.5F, 0.0F}, .distance_scale = 0.72F},
        false, false, 0U, 0U},
    GalleryEntry{GalleryContext::constraint_point,
        GallerySceneSource::constraint_point,
        GalleryControlPolicy::constraint_toggle_gravity, GalleryCountKind::none,
        "--constraint-point", "CONSTRAINT: POINT",
        "ARROWS GRAVITY  SPACE RELEASE / ATTACH", {48, 40, 31, 235},
        {245, 151, 52},
        {.target = {0.0F, 0.8F, 0.0F}, .distance_scale = 0.72F},
        false, false, 0U, 0U},
    GalleryEntry{GalleryContext::constraint_hinge,
        GallerySceneSource::constraint_hinge,
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none,
        "--constraint-hinge", "CONSTRAINT: HINGE",
        "ARROWS GRAVITY  SPHERE DRIVE  2:1 GEARS", {31, 45, 56, 235},
        {48, 165, 224},
        {.target = {0.0F, 2.4F, 0.0F}, .distance_scale = 0.72F},
        false, false, 0U, 0U},
    GalleryEntry{GalleryContext::constraint_slider,
        GallerySceneSource::constraint_slider,
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none,
        "--constraint-slider", "CONSTRAINT: SLIDER",
        "ARROWS GRAVITY  -1M TO +1M", {29, 49, 52, 235}, {38, 190, 196},
        {.target = {0.0F, 0.6F, 0.0F}, .distance_scale = 0.72F},
        false, false, 0U, 0U},
    GalleryEntry{GalleryContext::constraint_piston,
        GallerySceneSource::constraint_piston,
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none,
        "--constraint-piston", "CONSTRAINT: PISTON",
        "ARROWS GRAVITY  SLIDE + ROTATE", {29, 44, 57, 235}, {53, 153, 229},
        {.target = {0.4F, 1.4F, 0.0F}, .distance_scale = 0.70F},
        false, false, 0U, 0U},
    GalleryEntry{GalleryContext::constraint_generic,
        GallerySceneSource::constraint_generic,
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none,
        "--constraint-generic", "CONSTRAINT: GENERIC",
        "ARROWS GRAVITY  LINEAR + ANGULAR LIMITS", {42, 38, 57, 235},
        {151, 113, 235},
        {.target = {0.0F, 0.7F, 0.0F}, .distance_scale = 0.70F},
        false, false, 0U, 0U},
    GalleryEntry{GalleryContext::constraint_generic_spring,
        GallerySceneSource::constraint_generic_spring,
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none,
        "--constraint-generic-spring", "CONSTRAINT: GENERIC SPRING",
        "ARROWS GRAVITY  LIMITED SPRINGS", {45, 36, 54, 235}, {210, 116, 220},
        {.target = {0.0F, 0.7F, 0.0F}, .distance_scale = 0.70F},
        false, false, 0U, 0U},
    GalleryEntry{GalleryContext::constraint_motor,
        GallerySceneSource::constraint_motor,
        GalleryControlPolicy::tank_motor, GalleryCountKind::none,
        "--constraint-motor", "CONSTRAINT: MOTOR",
        "ARROWS TANK DRIVE", {28, 42, 52, 235}, {50, 154, 228},
        {.target = {0.0F, 0.6F, 0.0F}, .distance_scale = 0.92F},
        false, false, 0U, 0U},
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
    GalleryEntry{GalleryContext::soft_body_fluid, GallerySceneSource::soft_body_fluid,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::fluid_particles,
        "--soft-body-fluid", "SOFT BODY FLUID",
        "GOAL PINS  INFLOW / OUTFLOW  ARROWS GRAVITY  P CAP",
        {25, 42, 63, 235}, {80, 160, 240},
        {.target = {0.0F, 0.8F, 0.0F}, .distance_scale = 0.42F,
         .pitch = 0.55F}, true, false, 100U, 100'000U, true},
    GalleryEntry{GalleryContext::rope, GallerySceneSource::rope,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--rope", "ROPE", "HOOK ANCHORS  ARROWS GRAVITY  V SEGMENTS",
        {45, 34, 26, 235}, {240, 130, 50},
        {.target = {-0.5F, 0.3F, 0.0F}, .distance_scale = 0.48F,
         .pitch = 0.60F}, false, false, 0U, 0U, false, true},
    GalleryEntry{GalleryContext::rope_fluid, GallerySceneSource::rope_fluid,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::fluid_particles,
        "--rope-fluid", "ROPE FLUID", "SPLASH  ARROWS GRAVITY  P CAP",
        {30, 47, 57, 235}, {65, 180, 235},
        {.target = {-0.5F, 0.35F, 0.0F}, .distance_scale = 0.55F,
         .pitch = 0.62F}, true, false, 100U, 100'000U, false, true},
    GalleryEntry{GalleryContext::rope_soft_body, GallerySceneSource::rope_soft_body,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--rope-soft-body", "ROPE SOFT BODY", "SOFT POST  ARROWS GRAVITY  V SEGMENTS",
        {45, 35, 54, 235}, {230, 150, 76},
        {.target = {-0.5F, 0.3F, 0.0F}, .distance_scale = 0.48F,
         .pitch = 0.60F}, false, false, 0U, 0U, true, true},
    GalleryEntry{GalleryContext::rope_cloth, GallerySceneSource::rope_cloth,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--rope-cloth", "ROPE CLOTH", "SUSPENDED SHEET  ARROWS GRAVITY  V SEGMENTS",
        {45, 42, 52, 235}, {220, 158, 90},
        {.target = {0.0F, 0.25F, 0.0F}, .distance_scale = 0.54F,
         .pitch = 0.62F}, false, true, 0U, 0U, false, true},
    GalleryEntry{GalleryContext::smoke, GallerySceneSource::smoke,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--smoke", "SMOKE", "Z X C V B N GRID MAPS  R RESET",
        {42, 49, 59, 235}, {205, 215, 225},
        {.target = {0.25F, 1.5F, 0.0F}, .distance_scale = 0.72F,
         .pitch = 0.22F}, false, false, 0U, 0U},
    GalleryEntry{GalleryContext::smoke_water, GallerySceneSource::smoke_water,
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::fluid_particles,
        "--smoke-water", "SMOKE WATER", "Z X C V B N GRID MAPS  P CAP",
        {42, 49, 59, 235}, {115, 198, 225},
        {.target = {0.0F, 0.9F, 0.0F}, .distance_scale = 0.80F,
         .pitch = 1.05F}, true, false, 100U, 100'000U},
    GalleryEntry{GalleryContext::smoke_soft_body,
        GallerySceneSource::smoke_soft_body, GalleryControlPolicy::cloth_gravity,
        GalleryCountKind::none, "--smoke-softbody", "SMOKE SOFT BODY",
        "Z X C V B N GRID MAPS  ARROWS GRAVITY",
        {42, 49, 59, 235}, {170, 225, 195},
        {.target = {1.0F, 0.8F, 0.0F}, .distance_scale = 0.80F,
         .pitch = 0.35F}, false, false, 0U, 0U, true},
    GalleryEntry{GalleryContext::smoke_cloth,
        GallerySceneSource::smoke_cloth, GalleryControlPolicy::cloth_gravity,
        GalleryCountKind::none, "--smoke-cloth", "SMOKE CLOTH",
        "Z X C V B N GRID MAPS  ARROWS GRAVITY",
        {42, 49, 59, 235}, {184, 206, 230},
        {.target = {0.2F, 0.9F, 0.0F}, .distance_scale = 0.82F,
         .pitch = 0.35F}, false, true, 0U, 0U},
    GalleryEntry{GalleryContext::smoke_rope,
        GallerySceneSource::smoke_rope, GalleryControlPolicy::cloth_gravity,
        GalleryCountKind::none, "--smoke-rope", "SMOKE ROPE",
        "Z X C V B N GRID MAPS  ARROWS GRAVITY",
        {42, 49, 59, 235}, {230, 170, 95},
        {.target = {0.2F, 1.0F, 0.0F}, .distance_scale = 0.82F,
         .pitch = 0.35F}, false, false, 0U, 0U, false, true},
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

[[nodiscard]] constexpr bool is_smoke_context(
    GalleryContext context) noexcept {
    return context == GalleryContext::smoke ||
           context == GalleryContext::smoke_water ||
           context == GalleryContext::smoke_soft_body ||
           context == GalleryContext::smoke_cloth ||
           context == GalleryContext::smoke_rope;
}

} // namespace parallel_mater::gallery
