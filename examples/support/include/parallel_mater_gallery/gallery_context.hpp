// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/camera_controller.hpp>
#include <parallel_mater_gallery/device_profiles.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace parallel_mater::gallery {

enum class GalleryContext : std::uint8_t {
    rigid_body,
    constraint_fixed,
    constraint_point,
    constraint_hinge,
    constraint_piston,
    constraint_generic,
    constraint_motor_spring,
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

enum class GalleryControlPolicy : std::uint8_t {
    rigid_gravity,
    collector_gravity,
    motor_drive,
    authored_force,
    none,
    peg_gravity,
    cloth_gravity,
};

// Platform shells translate keys, touch controls, gamepads, or engine input
// into these semantic actions. No platform input code belongs in the catalog.
enum class GalleryPrimaryAction : std::uint8_t {
    none,
    release_restore_constraints,
    toggle_dump_bed,
};

struct AxleSteeringAngles {
    float front{};
    float rear{};
};

[[nodiscard]] constexpr AxleSteeringAngles axle_steering_angles(
    float right_input, float maximum_angle) noexcept {
    return {right_input * maximum_angle, -right_input * maximum_angle};
}

[[nodiscard]] constexpr bool uses_rigid_gravity(
    GalleryControlPolicy policy) noexcept {
    return policy == GalleryControlPolicy::rigid_gravity ||
           policy == GalleryControlPolicy::collector_gravity;
}

[[nodiscard]] inline CameraPreset brick_camera_preset(
    BrickSceneConfig config) noexcept {
    const std::uint32_t per_wall =
        (config.brick_count + config.wall_planes - 1U) / config.wall_planes;
    const std::uint32_t columns = std::max(1U, static_cast<std::uint32_t>(
        std::ceil(std::sqrt(static_cast<double>(per_wall) * 2.0 / 3.0))));
    std::uint32_t rows = 0U;
    std::uint32_t remaining = per_wall;
    while (remaining != 0U) {
        const std::uint32_t capacity = rows % 2U == 0U || columns == 1U
            ? columns : columns - 1U;
        remaining -= std::min(remaining, capacity);
        ++rows;
    }
    const float wall_width = columns * 0.8F * config.brick_scale;
    const float wall_height = rows * 0.4F * config.brick_scale;
    const float depth_span = (config.wall_planes - 1U) *
        (0.4F * config.brick_scale + 2.0F);
    return {.target = {0.0F, std::max(1.0F, wall_height * 0.42F),
                       5.5F - depth_span * 0.5F},
            .distance_scale = std::max({1.7F, wall_width / 5.0F,
                                        wall_height / 3.2F,
                                        (depth_span + 7.0F) / 6.0F})};
}

enum class GalleryCountKind : std::uint8_t {
    none,
    brick_scene,
    dump_spheres,
    fluid_particles,
};

struct GalleryColor {
    std::uint8_t red{};
    std::uint8_t green{};
    std::uint8_t blue{};
    std::uint8_t alpha{255U};
};

enum class GalleryFeature : std::uint32_t {
    rigid_bodies = 1U << 0U,
    rigid_constraints = 1U << 1U,
    particle_liquids = 1U << 2U,
    liquid_rigid_coupling = 1U << 3U,
    cloth = 1U << 4U,
    rope = 1U << 5U,
    soft_bodies = 1U << 6U,
    smoke = 1U << 7U,
    domain_couplings = 1U << 8U,
    paint = 1U << 9U,
};

using GalleryFeatureSet = std::uint32_t;

[[nodiscard]] constexpr GalleryFeatureSet gallery_feature_bit(
    GalleryFeature feature) noexcept {
    return static_cast<GalleryFeatureSet>(feature);
}

template <typename... Feature>
[[nodiscard]] constexpr GalleryFeatureSet gallery_feature_set(
    Feature... features) noexcept {
    return (GalleryFeatureSet{} | ... | gallery_feature_bit(features));
}

// Each bit names one exact public driver interaction capability. The catalog
// translates these stable descriptor bits to PmInteractionCapabilityKey.
enum class GalleryInteraction : std::uint64_t {
    rigid_rigid_contact = 1ULL << 0U,
    rigid_rigid_attachment = 1ULL << 1U,
    fluid_fluid_contact = 1ULL << 2U,
    fluid_fluid_pressure = 1ULL << 3U,
    fluid_fluid_viscosity = 1ULL << 4U,
    fluid_rigid_contact = 1ULL << 5U,
    fluid_rigid_containment = 1ULL << 6U,
    cloth_rigid_contact = 1ULL << 7U,
    soft_body_rigid_contact = 1ULL << 8U,
    rope_rope_contact = 1ULL << 9U,
    rope_rigid_contact = 1ULL << 10U,
    rope_rigid_attachment = 1ULL << 11U,
    smoke_smoke_pressure = 1ULL << 12U,
    smoke_smoke_viscosity = 1ULL << 13U,
    smoke_rigid_contact = 1ULL << 14U,
    smoke_rigid_drag = 1ULL << 15U,
    fluid_cloth_contact = 1ULL << 16U,
    fluid_cloth_containment = 1ULL << 17U,
    fluid_soft_body_contact = 1ULL << 18U,
    fluid_rope_contact = 1ULL << 19U,
    cloth_soft_body_contact = 1ULL << 20U,
    cloth_rope_attachment = 1ULL << 21U,
    soft_body_rope_contact = 1ULL << 22U,
    soft_body_rope_attachment = 1ULL << 23U,
    fluid_smoke_drag = 1ULL << 24U,
    fluid_smoke_thermal = 1ULL << 25U,
    fluid_smoke_phase = 1ULL << 26U,
    cloth_smoke_contact = 1ULL << 27U,
    cloth_smoke_drag = 1ULL << 28U,
    soft_body_smoke_contact = 1ULL << 29U,
    soft_body_smoke_drag = 1ULL << 30U,
    rope_smoke_contact = 1ULL << 31U,
    rope_smoke_drag = 1ULL << 32U,
};

using GalleryInteractionSet = std::uint64_t;

[[nodiscard]] constexpr GalleryInteractionSet gallery_interaction_bit(
    GalleryInteraction interaction) noexcept {
    return static_cast<GalleryInteractionSet>(interaction);
}

template <typename... Interaction>
[[nodiscard]] constexpr GalleryInteractionSet gallery_interaction_set(
    Interaction... interactions) noexcept {
    return (GalleryInteractionSet{} | ... |
            gallery_interaction_bit(interactions));
}

namespace gallery_scene_requirements {
inline constexpr GalleryFeatureSet rigid_features =
    gallery_feature_set(GalleryFeature::rigid_bodies);
inline constexpr GalleryFeatureSet constraint_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::rigid_constraints);
inline constexpr GalleryFeatureSet fluid_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::particle_liquids);
inline constexpr GalleryFeatureSet fluid_rigid_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::particle_liquids,
    GalleryFeature::liquid_rigid_coupling);
inline constexpr GalleryFeatureSet cloth_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::cloth);
inline constexpr GalleryFeatureSet soft_body_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::soft_bodies);
inline constexpr GalleryFeatureSet rope_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::rope);
inline constexpr GalleryFeatureSet smoke_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::smoke);
inline constexpr GalleryFeatureSet fluid_cloth_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::particle_liquids,
    GalleryFeature::cloth, GalleryFeature::domain_couplings);
inline constexpr GalleryFeatureSet fluid_soft_body_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::particle_liquids,
    GalleryFeature::soft_bodies, GalleryFeature::domain_couplings);
inline constexpr GalleryFeatureSet fluid_rope_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::particle_liquids,
    GalleryFeature::rope, GalleryFeature::domain_couplings);
inline constexpr GalleryFeatureSet cloth_soft_body_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::cloth,
    GalleryFeature::soft_bodies, GalleryFeature::domain_couplings);
inline constexpr GalleryFeatureSet cloth_rope_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::cloth,
    GalleryFeature::rope, GalleryFeature::domain_couplings);
inline constexpr GalleryFeatureSet soft_body_rope_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::soft_bodies,
    GalleryFeature::rope, GalleryFeature::domain_couplings);
inline constexpr GalleryFeatureSet fluid_smoke_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::particle_liquids,
    GalleryFeature::smoke, GalleryFeature::domain_couplings);
inline constexpr GalleryFeatureSet cloth_smoke_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::cloth,
    GalleryFeature::smoke, GalleryFeature::domain_couplings);
inline constexpr GalleryFeatureSet soft_body_smoke_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::soft_bodies,
    GalleryFeature::smoke, GalleryFeature::domain_couplings);
inline constexpr GalleryFeatureSet rope_smoke_features = gallery_feature_set(
    GalleryFeature::rigid_bodies, GalleryFeature::rope,
    GalleryFeature::smoke, GalleryFeature::domain_couplings);

inline constexpr GalleryInteractionSet rigid_interactions =
    gallery_interaction_set(GalleryInteraction::rigid_rigid_contact);
inline constexpr GalleryInteractionSet constraint_interactions =
    rigid_interactions | gallery_interaction_set(
        GalleryInteraction::rigid_rigid_attachment);
inline constexpr GalleryInteractionSet fluid_interactions =
    rigid_interactions | gallery_interaction_set(
        GalleryInteraction::fluid_fluid_contact,
        GalleryInteraction::fluid_fluid_pressure,
        GalleryInteraction::fluid_fluid_viscosity,
        GalleryInteraction::fluid_rigid_contact,
        GalleryInteraction::fluid_rigid_containment);
inline constexpr GalleryInteractionSet cloth_interactions =
    rigid_interactions | gallery_interaction_set(
        GalleryInteraction::cloth_rigid_contact);
inline constexpr GalleryInteractionSet soft_body_interactions =
    rigid_interactions | gallery_interaction_set(
        GalleryInteraction::soft_body_rigid_contact);
inline constexpr GalleryInteractionSet rope_interactions =
    rigid_interactions | gallery_interaction_set(
        GalleryInteraction::rope_rope_contact,
        GalleryInteraction::rope_rigid_contact,
        GalleryInteraction::rope_rigid_attachment);
inline constexpr GalleryInteractionSet smoke_interactions =
    rigid_interactions | gallery_interaction_set(
        GalleryInteraction::smoke_smoke_pressure,
        GalleryInteraction::smoke_smoke_viscosity,
        GalleryInteraction::smoke_rigid_contact,
        GalleryInteraction::smoke_rigid_drag);
inline constexpr GalleryInteractionSet fluid_cloth_interactions =
    fluid_interactions | cloth_interactions | gallery_interaction_set(
        GalleryInteraction::fluid_cloth_contact,
        GalleryInteraction::fluid_cloth_containment);
inline constexpr GalleryInteractionSet fluid_soft_body_interactions =
    fluid_interactions | soft_body_interactions | gallery_interaction_set(
        GalleryInteraction::fluid_soft_body_contact);
inline constexpr GalleryInteractionSet fluid_rope_interactions =
    fluid_interactions | rope_interactions | gallery_interaction_set(
        GalleryInteraction::fluid_rope_contact);
inline constexpr GalleryInteractionSet cloth_soft_body_interactions =
    cloth_interactions | soft_body_interactions | gallery_interaction_set(
        GalleryInteraction::cloth_soft_body_contact);
inline constexpr GalleryInteractionSet cloth_rope_interactions =
    cloth_interactions | rope_interactions | gallery_interaction_set(
        GalleryInteraction::cloth_rope_attachment);
inline constexpr GalleryInteractionSet soft_body_rope_interactions =
    soft_body_interactions | rope_interactions | gallery_interaction_set(
        GalleryInteraction::soft_body_rope_contact,
        GalleryInteraction::soft_body_rope_attachment);
inline constexpr GalleryInteractionSet fluid_smoke_interactions =
    fluid_interactions | smoke_interactions | gallery_interaction_set(
        GalleryInteraction::fluid_smoke_drag,
        GalleryInteraction::fluid_smoke_thermal,
        GalleryInteraction::fluid_smoke_phase);
inline constexpr GalleryInteractionSet cloth_smoke_interactions =
    cloth_interactions | smoke_interactions | gallery_interaction_set(
        GalleryInteraction::cloth_smoke_contact,
        GalleryInteraction::cloth_smoke_drag);
inline constexpr GalleryInteractionSet soft_body_smoke_interactions =
    soft_body_interactions | smoke_interactions | gallery_interaction_set(
        GalleryInteraction::soft_body_smoke_contact,
        GalleryInteraction::soft_body_smoke_drag);
inline constexpr GalleryInteractionSet rope_smoke_interactions =
    rope_interactions | smoke_interactions | gallery_interaction_set(
        GalleryInteraction::rope_smoke_contact,
        GalleryInteraction::rope_smoke_drag);
} // namespace gallery_scene_requirements

struct GalleryEntry {
    GalleryContext context{};
    std::string_view id{};
    std::string_view asset_file{};
    GalleryControlPolicy controls{};
    GalleryCountKind count_kind{};
    std::string_view command_line_option{};
    std::string_view name{};
    std::string_view help{};
    GalleryColor background{};
    GalleryColor icon{};
    CameraPreset camera{};
    GalleryFeatureSet required_features{};
    GalleryInteractionSet required_interactions{};
    std::uint32_t minimum_count{};
    std::uint32_t maximum_count{};
    bool moves_kinematic{};
    GalleryPrimaryAction primary_action{};
};

namespace scene_requirements = gallery_scene_requirements;

inline constexpr std::array gallery_entries{
    GalleryEntry{GalleryContext::rigid_body, "rigid-body", "RigidBody.glb",
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::brick_scene, {},
        "RIGID BODY", "ARROWS ROLL BALL  P WALL SETTINGS  R RESET", {48, 55, 63, 235},
        {170, 176, 184},
        {.target = {0.0F, 1.0F, 5.5F}, .distance_scale = 1.7F},
        scene_requirements::rigid_features,
        scene_requirements::rigid_interactions,
        brick_minimum_count, brick_maximum_count, true},
    GalleryEntry{GalleryContext::constraint_fixed,
        "constraint-fixed", "ConstraintFixed.glb",
        GalleryControlPolicy::collector_gravity, GalleryCountKind::none,
        "--constraint-fixed", "CONSTRAINT: FIXED",
        "ARROWS ROLL  TOUCH BIG TO STICK  R RESET", {51, 34, 38, 235},
        {244, 98, 70},
        {.target = {0.0F, 0.5F, 0.0F}, .distance_scale = 0.72F},
        scene_requirements::constraint_features,
        scene_requirements::constraint_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::constraint_point,
        "constraint-point", "Celestial.glb",
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none,
        "--constraint-point", "CONSTRAINT: POINT",
        "ARROWS GRAVITY  SPACE RELEASE / RESTORE  R RESET", {25, 32, 54, 235},
        {235, 190, 101},
        {.target = {0.0F, 3.1F, 0.0F}, .distance_scale = 1.65F,
         .pitch = 0.40F}, scene_requirements::constraint_features,
        scene_requirements::constraint_interactions,
        0U, 0U, false,
        GalleryPrimaryAction::release_restore_constraints},
    GalleryEntry{GalleryContext::constraint_hinge,
        "constraint-hinge", "ConstraintHinge.glb",
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none,
        "--constraint-hinge", "CONSTRAINT: HINGE + SLIDER",
        "ARROWS GRAVITY  SLIDER ROD  THREE GEARS", {31, 45, 56, 235},
        {48, 165, 224},
        {.target = {0.0F, 2.4F, 0.0F}, .distance_scale = 0.72F},
        scene_requirements::constraint_features,
        scene_requirements::constraint_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::constraint_piston,
        "constraint-piston", "ConstraintPiston.glb",
        GalleryControlPolicy::rigid_gravity, GalleryCountKind::none,
        "--constraint-piston", "CONSTRAINT: PISTON",
        "ARROWS GRAVITY  SLIDE + ROTATE", {29, 44, 57, 235}, {53, 153, 229},
        {.target = {0.4F, 1.4F, 0.0F}, .distance_scale = 0.70F},
        scene_requirements::constraint_features,
        scene_requirements::constraint_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::constraint_generic,
        "constraint-generic", "ConstraintGeneric.glb",
        GalleryControlPolicy::authored_force, GalleryCountKind::none,
        "--constraint-generic", "CONSTRAINT: GENERIC",
        "ARROWS FORCE  LINEAR + ANGULAR LIMITS", {42, 38, 57, 235},
        {151, 113, 235},
        {.target = {0.0F, 0.7F, 0.0F}, .distance_scale = 0.70F},
        scene_requirements::constraint_features,
        scene_requirements::constraint_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::constraint_motor_spring,
        "constraint-motor-spring", "ConstraintMotorSpring.glb",
        GalleryControlPolicy::motor_drive, GalleryCountKind::none,
        "--constraint-motor-spring", "CONSTRAINT: MOTOR + SPRING",
        "UP/DOWN DRIVE  LEFT/RIGHT 4-WHEEL STEER", {31, 44, 48, 235}, {242, 164, 61},
        {.target = {0.0F, 1.0F, 0.0F}, .distance_scale = 0.92F},
        scene_requirements::constraint_features,
        scene_requirements::constraint_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::dump, "dump-truck", "DumpTruck.glb",
        GalleryControlPolicy::motor_drive, GalleryCountKind::dump_spheres, "--dump",
        "DUMP", "ARROWS DRIVE/STEER  SPACE TIP  P LOAD", {62, 38, 22, 235},
        {245, 130, 45}, {.target = {0.0F, 1.8F, 0.0F}, .distance_scale = 1.3F,
                       .pitch = 0.85F}, scene_requirements::constraint_features,
        scene_requirements::constraint_interactions, 10U, 1'000U, false,
        GalleryPrimaryAction::toggle_dump_bed},
    GalleryEntry{GalleryContext::fluid, "fluid", "Fluid.glb",
        GalleryControlPolicy::none, GalleryCountKind::fluid_particles, "--fluid",
        "FLUID", "P CAP  V PARTICLES  R RESET", {12, 42, 65, 235},
        {35, 150, 255}, {.target = {-2.0F, -1.0F, -2.0F},
                         .distance_scale = 1.6F},
        scene_requirements::fluid_features,
        scene_requirements::fluid_interactions,
        100U, 100'000U},
    GalleryEntry{GalleryContext::fluid_rigid, "fluid-rigid", "FluidRigid.glb",
        GalleryControlPolicy::none, GalleryCountKind::fluid_particles,
        "--fluid-rigid", "FLUID RIGID", "FREE SPHERES  P CAP",
        {25, 52, 64, 235}, {35, 190, 230},
        {.target = {-2.0F, -1.0F, -2.0F}, .distance_scale = 1.6F},
        scene_requirements::fluid_rigid_features,
        scene_requirements::fluid_interactions,
        100U, 100'000U},
    GalleryEntry{GalleryContext::peg_paint, "peg-paint", "Pegs.glb",
        GalleryControlPolicy::peg_gravity, GalleryCountKind::fluid_particles,
        "--peg-paint", "PEG PAINT", "ARROWS GRAVITY  P CAP",
        {44, 28, 61, 235}, {42, 145, 255},
        {.target = {0.0F, -0.34F, 0.0F}, .distance_scale = 0.34F,
         .pitch = 0.72F},
        scene_requirements::fluid_features |
            gallery_feature_bit(GalleryFeature::paint),
        scene_requirements::fluid_interactions,
        100U, 100'000U},
    GalleryEntry{GalleryContext::cloth, "cloth", "Cloth.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none, "--cloth",
        "CLOTH", "ARROWS GRAVITY  R RESET", {40, 42, 58, 235},
        {236, 188, 96}, {.target = {0.0F, 1.1F, 1.2F},
                         .distance_scale = 0.55F, .pitch = 0.42F},
        scene_requirements::cloth_features,
        scene_requirements::cloth_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::cloth_tear, "cloth-tear", "ClothTear.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--cloth-tear", "CLOTH TEAR", "ARROWS GRAVITY  R RESET",
        {53, 37, 48, 235}, {255, 126, 111},
        {.target = {0.0F, 1.1F, 1.2F}, .distance_scale = 0.55F,
         .pitch = 0.42F}, scene_requirements::cloth_features,
        scene_requirements::cloth_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::cloth_paint, "cloth-paint", "ClothPaint.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--cloth-paint", "CLOTH PAINT", "ARROWS GRAVITY  R RESET",
        {28, 49, 55, 235}, {65, 177, 240},
        {.target = {0.0F, 1.1F, 1.2F}, .distance_scale = 0.55F,
         .pitch = 0.42F},
        scene_requirements::cloth_features |
            gallery_feature_bit(GalleryFeature::paint),
        scene_requirements::cloth_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::water_cloth, "cloth-water", "ClothWater.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::fluid_particles,
        "--water-cloth", "WATER CLOTH",
        "PRESSURE SKIN  ARROWS GRAVITY  P CAP", {20, 50, 68, 235},
        {45, 190, 235}, {.target = {0.0F, 1.1F, 1.2F},
                         .distance_scale = 0.55F, .pitch = 0.42F},
        scene_requirements::fluid_cloth_features,
        scene_requirements::fluid_cloth_interactions,
        100U, 100'000U},
    GalleryEntry{GalleryContext::soft_body, "soft-body", "Softbody.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--soft-body", "SOFT BODY", "ARROWS GRAVITY  R RESET",
        {49, 35, 67, 235}, {166, 102, 255},
        {.target = {0.0F, 0.7F, 0.0F}, .distance_scale = 0.72F,
         .pitch = 0.35F}, scene_requirements::soft_body_features,
        scene_requirements::soft_body_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::soft_body_rigid,
        "soft-body-rigid", "SoftbodyRigidBody.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--soft-body-rigid", "SOFT BODY RIGID",
        "TWO-WAY IMPACTS  ARROWS GRAVITY  R RESET",
        {46, 38, 64, 235}, {190, 118, 255},
        {.target = {0.0F, 0.7F, 0.0F}, .distance_scale = 0.72F,
         .pitch = 0.35F}, scene_requirements::soft_body_features,
        scene_requirements::soft_body_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::soft_body_cloth,
        "soft-body-cloth", "SoftbodyCloth.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--soft-body-cloth", "SOFT BODY CLOTH",
        "INTACT BRIDGE  TEARABLE CURTAIN  ARROWS GRAVITY",
        {42, 40, 62, 235}, {174, 128, 245},
        {.target = {0.0F, 0.6F, -1.5F}, .distance_scale = 0.78F,
        .pitch = 0.70F}, scene_requirements::cloth_soft_body_features,
        scene_requirements::cloth_soft_body_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::soft_body_fluid, "soft-body-fluid", "SoftbodyFluid.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::fluid_particles,
        "--soft-body-fluid", "SOFT BODY FLUID",
        "GOAL PINS  INFLOW / OUTFLOW  ARROWS GRAVITY  P CAP",
        {25, 42, 63, 235}, {80, 160, 240},
        {.target = {0.0F, 0.8F, 0.0F}, .distance_scale = 0.42F,
         .pitch = 0.55F}, scene_requirements::fluid_soft_body_features,
        scene_requirements::fluid_soft_body_interactions,
        100U, 100'000U},
    GalleryEntry{GalleryContext::rope, "rope", "Rope.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--rope", "ROPE", "HOOK ANCHORS  ARROWS GRAVITY  V SEGMENTS",
        {45, 34, 26, 235}, {240, 130, 50},
        {.target = {-0.5F, 0.3F, 0.0F}, .distance_scale = 0.48F,
         .pitch = 0.60F}, scene_requirements::rope_features,
        scene_requirements::rope_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::rope_fluid, "rope-fluid", "RopeFluid.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::fluid_particles,
        "--rope-fluid", "ROPE FLUID", "SPLASH  ARROWS GRAVITY  P CAP",
        {30, 47, 57, 235}, {65, 180, 235},
        {.target = {-0.5F, 0.35F, 0.0F}, .distance_scale = 0.55F,
         .pitch = 0.62F}, scene_requirements::fluid_rope_features,
        scene_requirements::fluid_rope_interactions,
        100U, 100'000U},
    GalleryEntry{GalleryContext::rope_soft_body, "rope-soft-body", "RopeSoftbody.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--rope-soft-body", "ROPE SOFT BODY", "SOFT POST  ARROWS GRAVITY  V SEGMENTS",
        {45, 35, 54, 235}, {230, 150, 76},
        {.target = {-0.5F, 0.3F, 0.0F}, .distance_scale = 0.48F,
         .pitch = 0.60F}, scene_requirements::soft_body_rope_features,
        scene_requirements::soft_body_rope_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::rope_cloth, "rope-cloth", "RopeCloth.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--rope-cloth", "ROPE CLOTH", "SUSPENDED SHEET  ARROWS GRAVITY  V SEGMENTS",
        {45, 42, 52, 235}, {220, 158, 90},
        {.target = {0.0F, 0.25F, 0.0F}, .distance_scale = 0.54F,
         .pitch = 0.62F}, scene_requirements::cloth_rope_features,
        scene_requirements::cloth_rope_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::smoke, "smoke", "Smoke.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::none,
        "--smoke", "SMOKE", "Z X C V B N GRID MAPS  R RESET",
        {42, 49, 59, 235}, {205, 215, 225},
        {.target = {0.25F, 1.5F, 0.0F}, .distance_scale = 0.72F,
         .pitch = 0.22F}, scene_requirements::smoke_features,
        scene_requirements::smoke_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::smoke_water, "smoke-water", "SmokeWater.glb",
        GalleryControlPolicy::cloth_gravity, GalleryCountKind::fluid_particles,
        "--smoke-water", "SMOKE WATER", "Z X C V B N GRID MAPS  P CAP",
        {42, 49, 59, 235}, {115, 198, 225},
        {.target = {0.0F, 0.9F, 0.0F}, .distance_scale = 0.80F,
         .pitch = 1.05F}, scene_requirements::fluid_smoke_features,
        scene_requirements::fluid_smoke_interactions,
        100U, 100'000U},
    GalleryEntry{GalleryContext::smoke_soft_body,
        "smoke-soft-body", "SmokeSoftbody.glb", GalleryControlPolicy::cloth_gravity,
        GalleryCountKind::none, "--smoke-softbody", "SMOKE SOFT BODY",
        "Z X C V B N GRID MAPS  ARROWS GRAVITY",
        {42, 49, 59, 235}, {170, 225, 195},
        {.target = {1.0F, 0.8F, 0.0F}, .distance_scale = 0.80F,
         .pitch = 0.35F}, scene_requirements::soft_body_smoke_features,
        scene_requirements::soft_body_smoke_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::smoke_cloth,
        "smoke-cloth", "SmokeCloth.glb", GalleryControlPolicy::cloth_gravity,
        GalleryCountKind::none, "--smoke-cloth", "SMOKE CLOTH",
        "Z X C V B N GRID MAPS  ARROWS GRAVITY",
        {42, 49, 59, 235}, {184, 206, 230},
        {.target = {0.2F, 0.9F, 0.0F}, .distance_scale = 0.82F,
         .pitch = 0.35F}, scene_requirements::cloth_smoke_features,
        scene_requirements::cloth_smoke_interactions,
        0U, 0U},
    GalleryEntry{GalleryContext::smoke_rope,
        "smoke-rope", "SmokeRope.glb", GalleryControlPolicy::cloth_gravity,
        GalleryCountKind::none, "--smoke-rope", "SMOKE ROPE",
        "Z X C V B N GRID MAPS  ARROWS GRAVITY",
        {42, 49, 59, 235}, {230, 170, 95},
        {.target = {0.2F, 1.0F, 0.0F}, .distance_scale = 0.82F,
         .pitch = 0.35F}, scene_requirements::rope_smoke_features,
        scene_requirements::rope_smoke_interactions,
        0U, 0U},
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

[[nodiscard]] constexpr const GalleryEntry *gallery_entry_for_option(
    std::string_view option) noexcept {
    for (const GalleryEntry &entry : gallery_entries)
        if (!entry.command_line_option.empty() &&
            entry.command_line_option == option) return &entry;
    return nullptr;
}

[[nodiscard]] constexpr bool gallery_requires_feature(
    const GalleryEntry &entry, GalleryFeature feature) noexcept {
    return (entry.required_features & gallery_feature_bit(feature)) != 0U;
}

[[nodiscard]] constexpr bool is_cloth_context(GalleryContext context) noexcept {
    return gallery_requires_feature(
        gallery_entry(context), GalleryFeature::cloth);
}

[[nodiscard]] constexpr bool is_fluid_context(GalleryContext context) noexcept {
    return gallery_requires_feature(
        gallery_entry(context), GalleryFeature::particle_liquids);
}

[[nodiscard]] constexpr bool is_soft_body_context(
    GalleryContext context) noexcept {
    return gallery_requires_feature(
        gallery_entry(context), GalleryFeature::soft_bodies);
}

[[nodiscard]] constexpr bool is_rope_context(GalleryContext context) noexcept {
    return gallery_requires_feature(
        gallery_entry(context), GalleryFeature::rope);
}

[[nodiscard]] constexpr bool is_smoke_context(
    GalleryContext context) noexcept {
    return gallery_requires_feature(
        gallery_entry(context), GalleryFeature::smoke);
}

} // namespace parallel_mater::gallery
