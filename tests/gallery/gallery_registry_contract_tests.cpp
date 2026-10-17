// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_catalog.hpp>

#include <stdexcept>
#include <string_view>

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
            item.required_feature_names.empty())
            throw std::runtime_error("Catalog metadata is incomplete");
        for (std::size_t previous = 0; previous < index; ++previous)
            if (catalog[previous].id == item.id)
                throw std::runtime_error("Catalog scene ID is not unique");
    }

    PmPhysicalDeviceFeatures rigid{};
    rigid.rigidBodies = PM_TRUE;
    rigid.rigidConstraints = PM_TRUE;
    const GalleryCatalog limited = make_gallery_catalog(&rigid);
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
    const GalleryCatalog complete = make_gallery_catalog(&full);
    for (const auto &item : complete)
        if (!item.available || !item.unavailable_reason.empty())
            throw std::runtime_error("Fully capable backend lost a scene");
    return 0;
}
