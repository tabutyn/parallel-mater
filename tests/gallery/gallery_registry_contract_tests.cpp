// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_context.hpp>

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

int main() { return 0; }
