// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_input.hpp>

#include <cstdlib>
#include <iostream>

namespace gallery = parallel_mater::gallery;

namespace {
void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
} // namespace

int main() {
    static_assert(gallery::gallery_entries.size() == 28U);

    for (std::size_t current = 0; current < gallery::gallery_entries.size();
         ++current) {
        gallery::GalleryInputState input;
        input.current = current;
        input.selected = current;
        input.arrow(gallery::GalleryArrowKey::right, true);

        if (gallery::gallery_entries[current].count_kind !=
            gallery::GalleryCountKind::none) {
            input.key(gallery::GalleryKey::parameters);
            require(input.parameters_visible,
                    "P must open the current scene parameter editor");
        }

        input.key(gallery::GalleryKey::catalog);
        require(input.catalog_visible,
                "Tab must open the catalog from every scene and modal state");
        require(!input.parameters_visible,
                "Opening the catalog must dismiss parameter editing");
        require(input.selected == current,
                "Opening the catalog must select the current scene");
        require(input.right_input() == 0.0F,
                "Opening a modal must clear held movement");

        while (input.selected + 1U < gallery::gallery_entries.size())
            input.key(gallery::GalleryKey::next);
        require(input.selected == gallery::gallery_entries.size() - 1U,
                "The last canonical catalog entry must be reachable");
        input.key(gallery::GalleryKey::enter);
        require(input.requested_scene == gallery::gallery_entries.size() - 1U,
                "Every canonical catalog entry must be selectable");
    }

    std::cout << "Gallery input contract passed\n";
}
