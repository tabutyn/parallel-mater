// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater_gallery/gallery_context.hpp>
#include <optional>

namespace parallel_mater::d3d12::viewer {
inline constexpr std::size_t rigid_scene_count =
    ::parallel_mater::gallery::gallery_context_index(
        ::parallel_mater::gallery::GalleryContext::fluid);

enum class Key { fps, catalog, escape, next, previous, enter, reset, pause, action };
enum class ArrowKey { left, right, up, down };

struct InputState {
    bool show_fps{};
    bool catalog_visible{};
    bool paused{};
    bool close_requested{};
    bool scene_action{};
    bool arrow_left{};
    bool arrow_right{};
    bool arrow_up{};
    bool arrow_down{};
    std::size_t current{};
    std::size_t selected{};
    std::optional<std::size_t> requested_scene{};
    ::parallel_mater::gallery::CameraController camera{};
    double fps{};
    double frame_ms{};
    double gpu_ms{};

    void arrow(ArrowKey key, bool pressed) noexcept {
        switch (key) {
        case ArrowKey::left: arrow_left = pressed; break;
        case ArrowKey::right: arrow_right = pressed; break;
        case ArrowKey::up: arrow_up = pressed; break;
        case ArrowKey::down: arrow_down = pressed; break;
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

    void key(Key key) {
        constexpr auto count = ::parallel_mater::gallery::gallery_entries.size();
        switch (key) {
        case Key::fps: show_fps = !show_fps; break;
        case Key::catalog:
            catalog_visible = !catalog_visible;
            selected = current;
            camera.end_drag();
            clear_arrows();
            break;
        case Key::escape:
            if (catalog_visible) catalog_visible = false;
            else close_requested = true;
            break;
        case Key::next:
            if (catalog_visible) selected = (selected + 1) % count;
            break;
        case Key::previous:
            if (catalog_visible) selected = (selected + count - 1) % count;
            break;
        case Key::enter:
            if (catalog_visible && selected < rigid_scene_count) {
                requested_scene = selected;
                catalog_visible = false;
            }
            break;
        case Key::reset:
            if (!catalog_visible) requested_scene = current;
            break;
        case Key::pause:
            if (!catalog_visible) paused = !paused;
            break;
        case Key::action:
            if (!catalog_visible) scene_action = true;
            break;
        }
    }
};

[[nodiscard]] inline ::parallel_mater::Vec3 control_gravity(
    ::parallel_mater::gallery::GalleryControlPolicy policy,
    bool authored_force_active, ::parallel_mater::gallery::Camera camera,
    float right_input, float up_input, float gravity_scale) noexcept {
    constexpr float gravity = 9.81F;
    const float magnitude = gravity * gravity_scale;
    if (authored_force_active ||
        !::parallel_mater::gallery::uses_rigid_gravity(policy))
        return {0.0F, -magnitude, 0.0F};
    return ::parallel_mater::gallery::screen_space_gravity(
        camera, right_input, up_input, magnitude, 30.0F);
}
} // namespace parallel_mater::d3d12::viewer
