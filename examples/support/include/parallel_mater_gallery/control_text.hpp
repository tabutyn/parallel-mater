// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/gallery_context.hpp>

#include <string_view>

namespace parallel_mater::gallery {

// User-facing semantic control text is common metadata. Platform adapters may
// choose their layout, but they must not reinterpret a scene's actions.
struct GalleryDirectionalControlText {
    std::string_view label;
    std::string_view left;
    std::string_view forward;
    std::string_view backward;
    std::string_view right;
};

[[nodiscard]] constexpr GalleryDirectionalControlText
gallery_directional_control_text(GalleryControlPolicy policy) noexcept {
    switch (policy) {
    case GalleryControlPolicy::authored_force:
        return {"Push objects", "Push left", "Push forward", "Push back",
                "Push right"};
    case GalleryControlPolicy::motor_drive:
        return {"Drive and steer", "Steer left", "Drive forward",
                "Drive reverse", "Steer right"};
    case GalleryControlPolicy::rigid_gravity:
    case GalleryControlPolicy::collector_gravity:
    case GalleryControlPolicy::peg_gravity:
    case GalleryControlPolicy::cloth_gravity:
        return {"Tilt gravity", "Tilt gravity left", "Tilt gravity forward",
                "Tilt gravity back", "Tilt gravity right"};
    case GalleryControlPolicy::none:
        return {"Camera only", {}, {}, {}, {}};
    }
    return {"Camera only", {}, {}, {}, {}};
}

[[nodiscard]] constexpr std::string_view gallery_primary_action_label(
    GalleryPrimaryAction primary, bool constraints_released,
    float dump_bed_angle) noexcept {
    switch (primary) {
    case GalleryPrimaryAction::release_restore_constraints:
        return constraints_released ? "Restore" : "Release";
    case GalleryPrimaryAction::toggle_dump_bed:
        return dump_bed_angle > 0.05F ? "Lower bed" : "Raise bed";
    case GalleryPrimaryAction::none:
        return {};
    }
    return {};
}

template <typename ControlState>
[[nodiscard]] constexpr std::string_view gallery_primary_action_label(
    const ControlState &controls) noexcept {
    return gallery_primary_action_label(controls.primary,
        controls.constraints_released, controls.dump_bed_angle);
}

} // namespace parallel_mater::gallery
