// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_catalog.hpp>

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace parallel_mater::gallery;

constexpr bool registry_matches_path_order() {
    for (std::size_t index = 0; index < gallery_entries.size(); ++index) {
        const auto &entry = gallery_entries[index];
        if (static_cast<std::size_t>(entry.context) != index ||
            static_cast<std::size_t>(entry.source) != index ||
            gallery_context_index(entry.context) != index)
            return false;
    }
    return true;
}

static_assert(registry_matches_path_order());
static_assert(gallery_entries.size() ==
              static_cast<std::size_t>(GallerySceneSource::smoke_rope) + 1U);

static_assert(gallery_entries.size() == 28U);
static_assert(gallery_entry(GalleryContext::constraint_motor_spring).controls ==
              GalleryControlPolicy::motor_drive);
static_assert(gallery_entry(GalleryContext::dump).source ==
              GallerySceneSource::dump_truck);
static_assert(gallery_entry(GalleryContext::dump).controls ==
              GalleryControlPolicy::motor_drive);
static_assert(gallery_entry(GalleryContext::dump).primary_action ==
              GalleryPrimaryAction::toggle_dump_bed);
static_assert(gallery_entry(GalleryContext::constraint_point).primary_action ==
              GalleryPrimaryAction::release_restore_constraints);
static_assert(gallery_entry(GalleryContext::constraint_generic).controls ==
              GalleryControlPolicy::authored_force);
static_assert(gallery_entry(GalleryContext::rigid_body).moves_kinematic);

std::vector<PmInteractionCapability> interactions_for(
    std::initializer_list<GalleryContext> contexts) {
    std::vector<PmInteractionCapability> result;
    for (const GalleryContext context : contexts) {
        visit_gallery_interaction_requirements(
            context, [&](const PmInteractionCapabilityKey &key) {
                for (const auto &existing : result)
                    if (gallery_interaction_key_equal(existing.key, key))
                        return;
                result.push_back({key, PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE});
            });
    }
    return result;
}

std::vector<PmInteractionCapability> all_interactions() {
    std::vector<PmInteractionCapability> result;
    for (const auto &entry : gallery_entries) {
        visit_gallery_interaction_requirements(
            entry.context, [&](const PmInteractionCapabilityKey &key) {
                for (const auto &existing : result)
                    if (gallery_interaction_key_equal(existing.key, key))
                        return;
                result.push_back({key, PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE});
            });
    }
    return result;
}

int main() {
    const GalleryCatalog catalog = make_gallery_catalog();
    if (catalog.size() != gallery_entries.size())
        throw std::runtime_error("Catalog size changed");
    for (std::size_t index = 0; index < catalog.size(); ++index) {
        const auto &item = catalog[index];
        if (item.scene.context != gallery_entries[index].context ||
            item.id.empty() || item.scene.name.empty() ||
            item.scene.help.empty() || item.asset_file.empty() ||
            item.required_features == 0U ||
            item.required_feature_names.empty() ||
            item.required_interaction_names.empty())
            throw std::runtime_error("Catalog metadata is incomplete");
        for (std::size_t previous = 0; previous < index; ++previous)
            if (catalog[previous].id == item.id)
                throw std::runtime_error("Catalog scene ID is not unique");
    }

    PmPhysicalDeviceFeatures rigid{};
    rigid.rigidBodies = PM_TRUE;
    rigid.rigidConstraints = PM_TRUE;
    const auto rigid_interactions = interactions_for({
        GalleryContext::rigid_body, GalleryContext::constraint_point});
    const GalleryCatalog limited =
        make_gallery_catalog(&rigid, rigid_interactions);
    if (!limited[gallery_context_index(GalleryContext::rigid_body)].available ||
        !limited[gallery_context_index(GalleryContext::constraint_point)].available)
        throw std::runtime_error("Supported rigid scenes are unavailable");
    const auto &fluid = limited[gallery_context_index(GalleryContext::fluid)];
    if (fluid.available ||
        fluid.unavailable_reason.find("particleLiquids") == std::string::npos)
        throw std::runtime_error("Fluid availability reason is imprecise");

    PmPhysicalDeviceFeatures full{};
    full.rigidBodies = full.rigidConstraints = full.particleLiquids =
        full.liquidRigidCoupling = full.cloth = full.rope = full.softBodies =
        full.smoke = full.domainCouplings = full.paint = PM_TRUE;
    const auto interactions = all_interactions();
    const GalleryCatalog complete = make_gallery_catalog(&full, interactions);
    for (const auto &item : complete)
        if (!item.available || !item.unavailable_reason.empty())
            throw std::runtime_error("Fully capable backend lost a scene");

    auto partial_rope = interactions;
    std::erase_if(partial_rope, [](const PmInteractionCapability &candidate) {
        const auto &key = candidate.key;
        return (key.firstFamily == PM_INTERACTION_FAMILY_CLOTH &&
                   key.secondFamily == PM_INTERACTION_FAMILY_ROPE) ||
            (key.firstFamily == PM_INTERACTION_FAMILY_SOFT_BODY &&
             key.secondFamily == PM_INTERACTION_FAMILY_ROPE);
    });
    const GalleryCatalog exact = make_gallery_catalog(&full, partial_rope);
    const auto &rope = exact[gallery_context_index(GalleryContext::rope)];
    const auto &rope_fluid =
        exact[gallery_context_index(GalleryContext::rope_fluid)];
    const auto &rope_cloth =
        exact[gallery_context_index(GalleryContext::rope_cloth)];
    const auto &rope_soft =
        exact[gallery_context_index(GalleryContext::rope_soft_body)];
    if (!rope.available || !rope_fluid.available || rope_cloth.available ||
        rope_soft.available ||
        rope_cloth.unavailable_reason.find("cloth-rope attachment") ==
            std::string::npos ||
        rope_soft.unavailable_reason.find("soft-body-rope contact") ==
            std::string::npos)
        throw std::runtime_error(
            "Catalog did not preserve exact rope interaction availability");
    return 0;
}
