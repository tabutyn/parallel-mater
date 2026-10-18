// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_gallery/gallery_context.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace parallel_mater::gallery {

struct GalleryCatalogItem {
    GalleryEntry scene{};
    std::string_view id{};
    std::string_view asset_file{};
    GalleryFeatureSet required_features{};
    bool available{};
    std::string required_feature_names{};
    std::string required_interaction_names{};
    std::string unavailable_reason{};
};

using GalleryCatalog = std::array<GalleryCatalogItem, gallery_entries.size()>;

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

[[nodiscard]] constexpr bool gallery_interaction_key_equal(
    const PmInteractionCapabilityKey &left,
    const PmInteractionCapabilityKey &right) noexcept {
    return left.firstFamily == right.firstFamily &&
        left.secondFamily == right.secondFamily && left.scope == right.scope &&
        left.mode == right.mode && left.direction == right.direction &&
        left.materialProfile == right.materialProfile;
}

[[nodiscard]] constexpr std::string_view gallery_interaction_family_name(
    PmInteractionFamily family) noexcept {
    switch (family) {
    case PM_INTERACTION_FAMILY_FLUID: return "fluid";
    case PM_INTERACTION_FAMILY_CLOTH: return "cloth";
    case PM_INTERACTION_FAMILY_SOFT_BODY: return "soft-body";
    case PM_INTERACTION_FAMILY_ROPE: return "rope";
    case PM_INTERACTION_FAMILY_SMOKE: return "smoke";
    case PM_INTERACTION_FAMILY_RIGID_BODY: return "rigid-body";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view gallery_interaction_mode_name(
    PmInteractionMode mode) noexcept {
    switch (mode) {
    case PM_INTERACTION_MODE_CONTACT: return "contact";
    case PM_INTERACTION_MODE_ATTACHMENT: return "attachment";
    case PM_INTERACTION_MODE_DRAG: return "drag";
    case PM_INTERACTION_MODE_CONTAINMENT: return "containment";
    case PM_INTERACTION_MODE_PRESSURE: return "pressure";
    case PM_INTERACTION_MODE_VISCOSITY: return "viscosity";
    case PM_INTERACTION_MODE_THERMAL: return "thermal";
    case PM_INTERACTION_MODE_PHASE: return "phase";
    case PM_INTERACTION_MODE_INTERFACE: return "interface";
    }
    return "unknown";
}

[[nodiscard]] inline std::string gallery_interaction_name(
    const PmInteractionCapabilityKey &key) {
    std::string result{gallery_interaction_family_name(key.firstFamily)};
    if (key.firstFamily == key.secondFamily)
        result += " self ";
    else {
        result += '-';
        result += gallery_interaction_family_name(key.secondFamily);
        result += ' ';
    }
    result += gallery_interaction_mode_name(key.mode);
    return result;
}

struct GalleryInteractionRequirement {
    GalleryInteraction interaction{};
    PmInteractionCapabilityKey key{};
};

[[nodiscard]] constexpr GalleryInteractionRequirement gallery_interaction(
    GalleryInteraction interaction, PmInteractionFamily first,
    PmInteractionFamily second, PmInteractionScope scope,
    PmInteractionMode mode,
    PmInteractionDirection direction = PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
    PmInteractionMaterialProfile profile =
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE) noexcept {
    return {interaction, {first, second, scope, mode, direction, profile}};
}

inline constexpr std::array gallery_interaction_requirements{
    gallery_interaction(GalleryInteraction::rigid_rigid_contact,
        PM_INTERACTION_FAMILY_RIGID_BODY, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_SAME_DOMAIN, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::rigid_rigid_attachment,
        PM_INTERACTION_FAMILY_RIGID_BODY, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_SAME_DOMAIN, PM_INTERACTION_MODE_ATTACHMENT),
    gallery_interaction(GalleryInteraction::fluid_fluid_contact,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_SCOPE_SELF, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::fluid_fluid_pressure,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_SCOPE_SELF, PM_INTERACTION_MODE_PRESSURE),
    gallery_interaction(GalleryInteraction::fluid_fluid_viscosity,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_SCOPE_SELF, PM_INTERACTION_MODE_VISCOSITY),
    gallery_interaction(GalleryInteraction::fluid_rigid_contact,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::fluid_rigid_containment,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTAINMENT),
    gallery_interaction(GalleryInteraction::cloth_rigid_contact,
        PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::soft_body_rigid_contact,
        PM_INTERACTION_FAMILY_SOFT_BODY, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::rope_rope_contact,
        PM_INTERACTION_FAMILY_ROPE, PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_SCOPE_SELF, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::rope_rigid_contact,
        PM_INTERACTION_FAMILY_ROPE, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::rope_rigid_attachment,
        PM_INTERACTION_FAMILY_ROPE, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_ATTACHMENT),
    gallery_interaction(GalleryInteraction::smoke_smoke_pressure,
        PM_INTERACTION_FAMILY_SMOKE, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_SELF, PM_INTERACTION_MODE_PRESSURE,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::smoke_smoke_viscosity,
        PM_INTERACTION_FAMILY_SMOKE, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_SELF, PM_INTERACTION_MODE_VISCOSITY,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::smoke_rigid_contact,
        PM_INTERACTION_FAMILY_SMOKE, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::smoke_rigid_drag,
        PM_INTERACTION_FAMILY_SMOKE, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_DRAG,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::fluid_cloth_contact,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_CLOTH,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::fluid_cloth_containment,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_CLOTH,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTAINMENT),
    gallery_interaction(GalleryInteraction::fluid_soft_body_contact,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_SOFT_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::fluid_rope_contact,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::cloth_soft_body_contact,
        PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_SOFT_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::cloth_rope_attachment,
        PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_ATTACHMENT),
    gallery_interaction(GalleryInteraction::soft_body_rope_contact,
        PM_INTERACTION_FAMILY_SOFT_BODY, PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT),
    gallery_interaction(GalleryInteraction::soft_body_rope_attachment,
        PM_INTERACTION_FAMILY_SOFT_BODY, PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_ATTACHMENT),
    gallery_interaction(GalleryInteraction::fluid_smoke_drag,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_DRAG,
        PM_INTERACTION_DIRECTION_SECOND_TO_FIRST,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::fluid_smoke_thermal,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_THERMAL,
        PM_INTERACTION_DIRECTION_EXTERNAL_TO_FIRST,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::fluid_smoke_phase,
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_PHASE,
        PM_INTERACTION_DIRECTION_FIRST_TO_SECOND_TRANSFER,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::cloth_smoke_contact,
        PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_FIRST_TO_SECOND,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::cloth_smoke_drag,
        PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_DRAG,
        PM_INTERACTION_DIRECTION_SECOND_TO_FIRST,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::soft_body_smoke_contact,
        PM_INTERACTION_FAMILY_SOFT_BODY, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_FIRST_TO_SECOND,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::soft_body_smoke_drag,
        PM_INTERACTION_FAMILY_SOFT_BODY, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_DRAG,
        PM_INTERACTION_DIRECTION_SECOND_TO_FIRST,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::rope_smoke_contact,
        PM_INTERACTION_FAMILY_ROPE, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_FIRST_TO_SECOND,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
    gallery_interaction(GalleryInteraction::rope_smoke_drag,
        PM_INTERACTION_FAMILY_ROPE, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_DRAG,
        PM_INTERACTION_DIRECTION_SECOND_TO_FIRST,
        PM_INTERACTION_MATERIAL_SMOKE_GRID),
};

template <typename Visitor>
constexpr void visit_gallery_interaction_requirements(
    const GalleryEntry &entry, Visitor &&visit) {
    for (const auto &requirement : gallery_interaction_requirements)
        if ((entry.required_interactions &
             gallery_interaction_bit(requirement.interaction)) != 0U)
            visit(requirement.key);
}

template <typename Visitor>
constexpr void visit_gallery_interaction_requirements(
    GalleryContext context, Visitor &&visit) {
    visit_gallery_interaction_requirements(
        gallery_entry(context), static_cast<Visitor &&>(visit));
}

[[nodiscard]] inline bool gallery_interaction_supported(
    const PmInteractionCapabilityKey &required,
    std::span<const PmInteractionCapability> capabilities) noexcept {
    for (const auto &capability : capabilities)
        if (gallery_interaction_key_equal(required, capability.key))
            return true;
    return false;
}

[[nodiscard]] inline GalleryCatalog make_gallery_catalog(
    const PmPhysicalDeviceFeatures *features = nullptr,
    std::span<const PmInteractionCapability> interactions = {}) {
    GalleryCatalog result{};
    for (std::size_t index = 0U; index < gallery_entries.size(); ++index) {
        auto &item = result[index];
        item.scene = gallery_entries[index];
        item.id = item.scene.id;
        item.asset_file = item.scene.asset_file;
        item.required_features = item.scene.required_features;
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
        std::string missing_interactions;
        visit_gallery_interaction_requirements(item.scene,
            [&](const PmInteractionCapabilityKey &required) {
                const std::string name = gallery_interaction_name(required);
                if (!item.required_interaction_names.empty())
                    item.required_interaction_names += ", ";
                item.required_interaction_names += name;
                if (features != nullptr &&
                    !gallery_interaction_supported(required, interactions)) {
                    item.available = false;
                    if (!missing_interactions.empty())
                        missing_interactions += ", ";
                    missing_interactions += name;
                }
            });
        if (!item.available && !item.unavailable_reason.empty())
            item.unavailable_reason =
                "Missing backend features: " + item.unavailable_reason;
        if (!missing_interactions.empty()) {
            if (!item.unavailable_reason.empty())
                item.unavailable_reason += "; ";
            item.unavailable_reason +=
                "Missing backend interactions: " + missing_interactions;
        }
    }
    return result;
}

[[nodiscard]] constexpr bool gallery_backend_supports(
    GalleryContext context, const PmPhysicalDeviceFeatures &features) noexcept {
    const GalleryFeatureSet required = gallery_entry(context).required_features;
    for (GalleryFeature feature : gallery_features)
        if ((required & gallery_feature_bit(feature)) != 0U &&
            !gallery_feature_supported(feature, features)) return false;
    return true;
}

} // namespace parallel_mater::gallery
