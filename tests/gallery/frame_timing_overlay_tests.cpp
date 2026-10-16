// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/overlay.hpp>

#include <cstdlib>
#include <iostream>

using namespace parallel_mater::gallery;

namespace {
void require(bool condition, const char *message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}
FrameTimingCollector::TimePoint at(int ms) {
    return FrameTimingCollector::TimePoint{} + std::chrono::milliseconds(ms);
}
} // namespace

int main() {
    for (const std::uint32_t width : {640U, 1280U, 1920U}) {
        const auto height = width * 9U / 16U;
        const std::vector<std::uint32_t> background(width * height, 0xff808080U);
        FrameTimingCollector timings;
        auto startup = background;
        draw_frame_timing_overlay(startup, width, height, timings);
        require(startup != background, "Startup must draw the no-sample panel");
        timings.begin_frame(at(0));
        timings.begin_phase(FramePhase::physics, at(0));
        timings.end_phase(at(4));
        timings.begin_phase(FramePhase::render, at(4));
        timings.end_phase(at(8));
        timings.begin_phase(FramePhase::present, at(8));
        timings.end_phase(at(16));
        require(timings.finish_frame(at(16)), "Sample must complete");
        auto completed = background;
        draw_frame_timing_overlay(completed, width, height, timings);
        require(completed != startup, "Completed frame must replace the placeholder");

        timings.begin_frame(at(16));
        timings.begin_phase(FramePhase::physics, at(16));
        auto pending = background;
        draw_frame_timing_overlay(pending, width, height, timings);
        require(pending == completed, "Panel must show previous completed sample, not partial work");
        timings.cancel_frame();
        timings.clear_history();
        auto reset = background;
        draw_frame_timing_overlay(reset, width, height, timings);
        require(reset == startup, "Scene reset/calibration must remove stale metrics");
    }
    std::cout << "Gallery timing overlay tests passed\n";
}
