// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/renderer.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

void draw_timing_overlay(std::vector<std::uint32_t> &rgba,
                         std::uint32_t width, std::uint32_t height,
                         const WorldStepTimings &timings);

[[nodiscard]] bool draw_rigid_contact_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, RigidContactDeviceView contacts, Camera camera,
    std::string &error);

// Fluid is intentionally shown as unavailable until its Blender-authored
// acceptance scene exists. Gallery navigation belongs to examples, not World.
void draw_context_overlay(std::vector<std::uint32_t> &rgba,
                          std::uint32_t width, std::uint32_t height);

} // namespace parallel_mater::gallery
