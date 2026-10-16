// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/gallery_context.hpp>

#include <optional>

namespace parallel_mater::gallery {

enum class GalleryKey : std::uint8_t {
    fps,
    catalog,
    escape,
    next,
    previous,
    enter,
    parameters,
    reset,
    action,
};

enum class GalleryArrowKey : std::uint8_t { left, right, up, down };

// Shared application input state. Platform event adapters translate native
// events to GalleryKey/GalleryArrowKey and do not own modal behavior.
struct GalleryInputState {
    bool show_fps{};
    bool catalog_visible{};
    bool parameters_visible{};
    bool parameters_discarded{};
    bool close_requested{};
    bool scene_action{};
    bool arrow_left{};
    bool arrow_right{};
    bool arrow_up{};
    bool arrow_down{};
    std::size_t current{};
    std::size_t selected{};
    std::optional<std::size_t> requested_scene{};
    CameraController camera{};

    void arrow(GalleryArrowKey key, bool pressed) noexcept {
        if (catalog_visible || parameters_visible) pressed = false;
        switch (key) {
        case GalleryArrowKey::left: arrow_left = pressed; break;
        case GalleryArrowKey::right: arrow_right = pressed; break;
        case GalleryArrowKey::up: arrow_up = pressed; break;
        case GalleryArrowKey::down: arrow_down = pressed; break;
        }
    }

    void clear_arrows() noexcept {
        arrow_left = arrow_right = arrow_up = arrow_down = false;
    }

    [[nodiscard]] float right_input() const noexcept {
        return static_cast<float>(arrow_right) - static_cast<float>(arrow_left);
    }

    [[nodiscard]] float up_input() const noexcept {
        return static_cast<float>(arrow_up) - static_cast<float>(arrow_down);
    }

    void key(GalleryKey key) {
        constexpr auto count = gallery_entries.size();
        parameters_discarded = false;
        switch (key) {
        case GalleryKey::fps:
            show_fps = !show_fps;
            break;
        case GalleryKey::catalog:
            if (parameters_visible) {
                parameters_visible = false;
                parameters_discarded = true;
                catalog_visible = true;
            } else {
                catalog_visible = !catalog_visible;
            }
            selected = current;
            camera.end_drag();
            clear_arrows();
            break;
        case GalleryKey::escape:
            if (parameters_visible) {
                parameters_visible = false;
                parameters_discarded = true;
                clear_arrows();
            } else if (catalog_visible) {
                catalog_visible = false;
                clear_arrows();
            } else {
                close_requested = true;
            }
            break;
        case GalleryKey::next:
            if (catalog_visible) selected = (selected + 1U) % count;
            break;
        case GalleryKey::previous:
            if (catalog_visible) selected = (selected + count - 1U) % count;
            break;
        case GalleryKey::enter:
            if (catalog_visible) {
                requested_scene = selected;
                catalog_visible = false;
                clear_arrows();
            }
            break;
        case GalleryKey::parameters:
            if (!catalog_visible &&
                gallery_entries[current].count_kind != GalleryCountKind::none) {
                parameters_visible = true;
                camera.end_drag();
                clear_arrows();
            }
            break;
        case GalleryKey::reset:
            if (!catalog_visible && !parameters_visible)
                requested_scene = current;
            break;
        case GalleryKey::action:
            if (!catalog_visible && !parameters_visible) scene_action = true;
            break;
        }
    }
};

} // namespace parallel_mater::gallery
