// SPDX-License-Identifier: MIT
#include "foam_visuals.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    constexpr std::uint32_t width = 256U, height = 256U;
    constexpr std::size_t count = 256U;
    const Camera camera{{0.0F, 0.0F, 3.0F}, {0.0F, 0.0F, 0.0F}};
    std::vector<Vec3> positions(count);
    std::vector<float> signal(count, 1.0F);
    std::vector<std::uint32_t> ids(count);
    for (std::size_t i = 0; i < count; ++i) {
        positions[i] = {0.04F * (static_cast<float>(i % 16U) - 7.5F),
                        0.04F * (static_cast<float>(i / 16U) - 7.5F), 0.0F};
        ids[i] = static_cast<std::uint32_t>(i);
    }
    FoamVisuals peg, stream;
    peg.advance(positions, signal, ids, 0.03F);
    stream.advance(positions, signal, ids, 0.045F);
    if (peg.patch_count() < 10U || peg.patch_count() != stream.patch_count()) {
        std::cerr << "foam patch emission must be shared across scenes\n";
        return 1;
    }
    const std::vector<float> clear_depth(width * height,
                                         std::numeric_limits<float>::infinity());
    std::vector<std::uint32_t> peg_pixels(width * height, 0xff000000U);
    std::vector<std::uint32_t> stream_pixels = peg_pixels;
    peg.paint(0.12F, camera, width, height, clear_depth, clear_depth, peg_pixels);
    stream.paint(0.18F, camera, width, height, clear_depth, clear_depth,
                 stream_pixels);
    const auto painted = [](const std::vector<std::uint32_t> &pixels) {
        return std::count_if(pixels.begin(), pixels.end(),
                             [](std::uint32_t value) {
                                 return (value & 0x00ffffffU) != 0U;
                             });
    };
    const auto peg_coverage = painted(peg_pixels);
    const auto stream_coverage = painted(stream_pixels);
    if (peg_coverage == 0 || stream_coverage <= peg_coverage * 1.15) {
        std::cerr << "larger stream particles need visible foam patches\n";
        return 1;
    }
    std::vector<std::uint32_t> hidden(width * height, 0xff000000U);
    const std::vector<float> rigid_depth(width * height, 0.0F);
    stream.paint(0.18F, camera, width, height, clear_depth, rigid_depth, hidden);
    if (painted(hidden) != 0) {
        std::cerr << "rigid depth must occlude shared foam patches\n";
        return 1;
    }
    return 0;
}
