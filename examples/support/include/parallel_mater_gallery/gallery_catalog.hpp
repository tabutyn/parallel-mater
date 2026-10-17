// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_gallery/gallery_context.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace parallel_mater::gallery {

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

struct GalleryCatalogItem {
    GalleryEntry scene{};
    std::string_view id{};
    std::string_view asset_file{};
    GalleryFeatureSet required_features{};
    bool available{};
    std::string required_feature_names{};
    std::string unavailable_reason{};
};

using GalleryCatalog = std::array<GalleryCatalogItem, gallery_entries.size()>;

[[nodiscard]] constexpr std::string_view gallery_scene_id(
    GalleryContext context) noexcept {
    switch (context) {
    case GalleryContext::rigid_body: return "rigid-body";
    case GalleryContext::constraint_fixed: return "constraint-fixed";
    case GalleryContext::constraint_point: return "constraint-point";
    case GalleryContext::constraint_hinge: return "constraint-hinge";
    case GalleryContext::constraint_piston: return "constraint-piston";
    case GalleryContext::constraint_generic: return "constraint-generic";
    case GalleryContext::constraint_motor_spring: return "constraint-motor-spring";
    case GalleryContext::dump: return "dump-truck";
    case GalleryContext::fluid: return "fluid";
    case GalleryContext::fluid_rigid: return "fluid-rigid";
    case GalleryContext::peg_paint: return "peg-paint";
    case GalleryContext::cloth: return "cloth";
    case GalleryContext::cloth_tear: return "cloth-tear";
    case GalleryContext::cloth_paint: return "cloth-paint";
    case GalleryContext::water_cloth: return "cloth-water";
    case GalleryContext::soft_body: return "soft-body";
    case GalleryContext::soft_body_rigid: return "soft-body-rigid";
    case GalleryContext::soft_body_cloth: return "soft-body-cloth";
    case GalleryContext::soft_body_fluid: return "soft-body-fluid";
    case GalleryContext::rope: return "rope";
    case GalleryContext::rope_fluid: return "rope-fluid";
    case GalleryContext::rope_soft_body: return "rope-soft-body";
    case GalleryContext::rope_cloth: return "rope-cloth";
    case GalleryContext::smoke: return "smoke";
    case GalleryContext::smoke_water: return "smoke-water";
    case GalleryContext::smoke_soft_body: return "smoke-soft-body";
    case GalleryContext::smoke_cloth: return "smoke-cloth";
    case GalleryContext::smoke_rope: return "smoke-rope";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view gallery_scene_asset(
    GallerySceneSource source) noexcept {
    switch (source) {
    case GallerySceneSource::default_scene: return "RigidBody.glb";
    case GallerySceneSource::constraint_fixed: return "ConstraintFixed.glb";
    case GallerySceneSource::constraint_point: return "Celestial.glb";
    case GallerySceneSource::constraint_hinge: return "ConstraintHinge.glb";
    case GallerySceneSource::constraint_piston: return "ConstraintPiston.glb";
    case GallerySceneSource::constraint_generic: return "ConstraintGeneric.glb";
    case GallerySceneSource::constraint_motor_spring: return "ConstraintMotorSpring.glb";
    case GallerySceneSource::dump_truck: return "DumpTruck.glb";
    case GallerySceneSource::fluid: return "Fluid.glb";
    case GallerySceneSource::fluid_rigid: return "FluidRigid.glb";
    case GallerySceneSource::peg_paint: return "Pegs.glb";
    case GallerySceneSource::cloth: return "Cloth.glb";
    case GallerySceneSource::cloth_tear: return "ClothTear.glb";
    case GallerySceneSource::cloth_paint: return "ClothPaint.glb";
    case GallerySceneSource::water_cloth: return "ClothWater.glb";
    case GallerySceneSource::soft_body: return "Softbody.glb";
    case GallerySceneSource::soft_body_rigid: return "SoftbodyRigidBody.glb";
    case GallerySceneSource::soft_body_cloth: return "SoftbodyCloth.glb";
    case GallerySceneSource::soft_body_fluid: return "SoftbodyFluid.glb";
    case GallerySceneSource::rope: return "Rope.glb";
    case GallerySceneSource::rope_fluid: return "RopeFluid.glb";
    case GallerySceneSource::rope_soft_body: return "RopeSoftbody.glb";
    case GallerySceneSource::rope_cloth: return "RopeCloth.glb";
    case GallerySceneSource::smoke: return "Smoke.glb";
    case GallerySceneSource::smoke_water: return "SmokeWater.glb";
    case GallerySceneSource::smoke_soft_body: return "SmokeSoftbody.glb";
    case GallerySceneSource::smoke_cloth: return "SmokeCloth.glb";
    case GallerySceneSource::smoke_rope: return "SmokeRope.glb";
    }
    return {};
}

[[nodiscard]] constexpr GalleryFeatureSet gallery_scene_requirements(
    GalleryContext context) noexcept {
    const GalleryEntry &entry = gallery_entry(context);
    GalleryFeatureSet result = gallery_feature_bit(GalleryFeature::rigid_bodies);
    if (context >= GalleryContext::constraint_fixed &&
        context <= GalleryContext::dump)
        result |= gallery_feature_bit(GalleryFeature::rigid_constraints);
    if (entry.has_fluid)
        result |= gallery_feature_bit(GalleryFeature::particle_liquids);
    if (context == GalleryContext::fluid_rigid)
        result |= gallery_feature_bit(GalleryFeature::liquid_rigid_coupling);
    if (entry.has_cloth) result |= gallery_feature_bit(GalleryFeature::cloth);
    if (entry.has_rope) result |= gallery_feature_bit(GalleryFeature::rope);
    if (entry.has_soft_body)
        result |= gallery_feature_bit(GalleryFeature::soft_bodies);
    if (is_smoke_context(context))
        result |= gallery_feature_bit(GalleryFeature::smoke);
    if (context == GalleryContext::peg_paint ||
        context == GalleryContext::cloth_paint)
        result |= gallery_feature_bit(GalleryFeature::paint);
    const unsigned domains = unsigned(entry.has_fluid) +
        unsigned(entry.has_cloth) + unsigned(entry.has_rope) +
        unsigned(entry.has_soft_body) + unsigned(is_smoke_context(context));
    if (domains >= 2U)
        result |= gallery_feature_bit(GalleryFeature::domain_couplings);
    return result;
}

[[nodiscard]] constexpr bool gallery_feature_supported(
    GalleryFeature feature, const PmPhysicalDeviceFeatures &features) noexcept {
    switch (feature) {
    case GalleryFeature::rigid_bodies: return features.rigidBodies == PM_TRUE;
    case GalleryFeature::rigid_constraints: return features.rigidConstraints == PM_TRUE;
    case GalleryFeature::particle_liquids: return features.particleLiquids == PM_TRUE;
    case GalleryFeature::liquid_rigid_coupling: return features.liquidRigidCoupling == PM_TRUE;
    case GalleryFeature::cloth: return features.cloth == PM_TRUE;
    case GalleryFeature::rope: return features.rope == PM_TRUE;
    case GalleryFeature::soft_bodies: return features.softBodies == PM_TRUE;
    case GalleryFeature::smoke: return features.smoke == PM_TRUE;
    case GalleryFeature::domain_couplings: return features.domainCouplings == PM_TRUE;
    case GalleryFeature::paint: return features.paint == PM_TRUE;
    }
    return false;
}

[[nodiscard]] constexpr std::string_view gallery_feature_name(
    GalleryFeature feature) noexcept {
    switch (feature) {
    case GalleryFeature::rigid_bodies: return "rigidBodies";
    case GalleryFeature::rigid_constraints: return "rigidConstraints";
    case GalleryFeature::particle_liquids: return "particleLiquids";
    case GalleryFeature::liquid_rigid_coupling: return "liquidRigidCoupling";
    case GalleryFeature::cloth: return "cloth";
    case GalleryFeature::rope: return "rope";
    case GalleryFeature::soft_bodies: return "softBodies";
    case GalleryFeature::smoke: return "smoke";
    case GalleryFeature::domain_couplings: return "domainCouplings";
    case GalleryFeature::paint: return "paint";
    }
    return "unknown";
}

inline constexpr std::array gallery_features{
    GalleryFeature::rigid_bodies,
    GalleryFeature::rigid_constraints,
    GalleryFeature::particle_liquids,
    GalleryFeature::liquid_rigid_coupling,
    GalleryFeature::cloth,
    GalleryFeature::rope,
    GalleryFeature::soft_bodies,
    GalleryFeature::smoke,
    GalleryFeature::domain_couplings,
    GalleryFeature::paint,
};

[[nodiscard]] inline GalleryCatalog make_gallery_catalog(
    const PmPhysicalDeviceFeatures *features = nullptr) {
    GalleryCatalog result{};
    for (std::size_t index = 0U; index < gallery_entries.size(); ++index) {
        auto &item = result[index];
        item.scene = gallery_entries[index];
        item.id = gallery_scene_id(item.scene.context);
        item.asset_file = gallery_scene_asset(item.scene.source);
        item.required_features = gallery_scene_requirements(item.scene.context);
        item.available = true;
        for (GalleryFeature feature : gallery_features) {
            const GalleryFeatureSet bit = gallery_feature_bit(feature);
            if ((item.required_features & bit) == 0U) continue;
            const std::string_view name = gallery_feature_name(feature);
            if (!item.required_feature_names.empty())
                item.required_feature_names += ", ";
            item.required_feature_names += name;
            if (features != nullptr &&
                !gallery_feature_supported(feature, *features)) {
                item.available = false;
                if (!item.unavailable_reason.empty())
                    item.unavailable_reason += ", ";
                item.unavailable_reason += name;
            }
        }
        if (!item.available)
            item.unavailable_reason =
                "Missing backend features: " + item.unavailable_reason;
    }
    return result;
}

[[nodiscard]] constexpr bool gallery_backend_supports(
    GalleryContext context, const PmPhysicalDeviceFeatures &features) noexcept {
    const GalleryFeatureSet required = gallery_scene_requirements(context);
    for (GalleryFeature feature : gallery_features)
        if ((required & gallery_feature_bit(feature)) != 0U &&
            !gallery_feature_supported(feature, features)) return false;
    return true;
}

} // namespace parallel_mater::gallery
