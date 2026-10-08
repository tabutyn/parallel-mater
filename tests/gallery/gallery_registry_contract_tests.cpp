// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/scene.hpp>

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

// Shared diagnostics must remain available through both public API headers.
static_assert(parallel_mater::WorldStatistics{}.rigid_contact_island_count == 0U);
static_assert(parallel_mater::WorldStatistics{}.rigid_contact_early_exit_count == 0U);
static_assert(parallel_mater::WorldStatistics{}.sleeping_rigid_body_count == 0U);

#if defined(PARALLEL_MATER_GALLERY_METAL)
static_assert(gallery_entries.size() == 29U);
static_assert(gallery_entry(GalleryContext::constraint_motor).controls ==
              GalleryControlPolicy::tank_motor);
static_assert(gallery_entry(GalleryContext::constraint_generic_spring).source ==
              GallerySceneSource::constraint_generic_spring);
static_assert(gallery_entry(GalleryContext::dump).source ==
              GallerySceneSource::procedural_dump);
static_assert(gallery_entry(GalleryContext::dump).controls ==
              GalleryControlPolicy::dump_rotation);
#else
static_assert(gallery_entries.size() == 28U);
static_assert(gallery_entry(GalleryContext::constraint_motor_spring).controls ==
              GalleryControlPolicy::motor_drive);
static_assert(gallery_entry(GalleryContext::dump).source ==
              GallerySceneSource::dump_truck);
static_assert(gallery_entry(GalleryContext::dump).controls ==
              GalleryControlPolicy::motor_drive);
#endif

int main() { return 0; }
