// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater_gallery/gallery_input.hpp>

namespace parallel_mater::d3d12::viewer {
using Key = ::parallel_mater::gallery::GalleryKey;
using ArrowKey = ::parallel_mater::gallery::GalleryArrowKey;
using InputState = ::parallel_mater::gallery::GalleryInputState;

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
