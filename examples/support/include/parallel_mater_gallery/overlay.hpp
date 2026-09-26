// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/renderer.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace parallel_mater::gallery {

enum class GalleryContext : std::uint8_t {
    rigid_body,
    dump,
    fluid,
    fluid_rigid,
    peg_paint,
    cloth,
    cloth_tear,
    cloth_paint,
};

[[nodiscard]] constexpr bool is_cloth_context(GalleryContext context) noexcept {
    return context == GalleryContext::cloth ||
           context == GalleryContext::cloth_tear ||
           context == GalleryContext::cloth_paint;
}

[[nodiscard]] constexpr bool is_fluid_context(GalleryContext context) noexcept {
    return context == GalleryContext::fluid ||
           context == GalleryContext::fluid_rigid ||
           context == GalleryContext::peg_paint;
}

void draw_timing_overlay(std::vector<std::uint32_t> &rgba,
                         std::uint32_t width, std::uint32_t height,
                         const WorldStepTimings &timings);

void draw_cloth_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &timings);

void draw_fluid_timing_overlay(std::vector<std::uint32_t> &rgba,
                               std::uint32_t width, std::uint32_t height,
                               const WorldStepTimings &physics,
                               const RendererTimings &renderer,
                               const WorldStatistics &statistics,
                               std::uint32_t capacity);

[[nodiscard]] bool draw_rigid_contact_overlay(
    std::vector<std::uint32_t> &rgba, std::uint32_t width,
    std::uint32_t height, RigidContactDeviceView contacts, Camera camera,
    std::string &error);

// Fluid loads the Blender-authored Flow scene. Gallery navigation belongs to
// examples, not World.
void draw_context_overlay(std::vector<std::uint32_t> &rgba,
                          std::uint32_t width, std::uint32_t height,
                          GalleryContext selection);

void draw_count_overlay(std::vector<std::uint32_t> &rgba,
                        std::uint32_t width, std::uint32_t height,
                        GalleryContext context, const std::string &value,
                        bool invalid);

} // namespace parallel_mater::gallery
