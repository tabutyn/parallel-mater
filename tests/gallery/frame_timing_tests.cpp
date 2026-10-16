// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/frame_timing.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <locale>
#include <sstream>
#include <streambuf>

using namespace parallel_mater::gallery;

namespace {
void require(bool condition, const char *message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}
void near(double actual, double expected, const char *message) {
    require(std::abs(actual - expected) < 1.0e-8, message);
}
FrameTimingCollector::TimePoint at(int ms) {
    return FrameTimingCollector::TimePoint{} + std::chrono::milliseconds(ms);
}
struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
};
struct WriteFailure : std::streambuf {
    std::streamsize xsputn(const char *, std::streamsize) override { return 0; }
    int_type overflow(int_type) override { return traits_type::eof(); }
};
struct FlushFailure : std::stringbuf {
    int sync() override { return -1; }
};
} // namespace

int main() {
    FrameTimingCollector timings;
    require(!timings.latest(), "Startup must not invent a frame");
    near(timings.fps(), 0.0, "Startup FPS must be finite");
    require(!timings.finish_frame(at(0)), "An inactive frame cannot finish");
    timings.begin_frame(at(0));
    timings.begin_phase(FramePhase::physics, at(1));
    near(timings.end_phase(at(3)), 2.0, "Physics budget must get this step's duration");
    timings.begin_phase(FramePhase::physics, at(3));
    near(timings.end_phase(at(6)), 3.0, "Second step budget must not use cumulative time");
    timings.begin_phase(FramePhase::render, at(7));
    timings.end_phase(at(11));
    timings.begin_phase(FramePhase::present, at(12));
    timings.end_phase(at(20));
    require(timings.finish_frame(at(20)), "Complete frame must publish");
    const auto first = *timings.latest();
    require(first.frame == 1 && first.physics_steps == 2, "Count presented frames and physics steps separately");
    near(first.total_ms, 20.0, "Total must include overhead and VSync");
    near(first.physics_ms, 5.0, "Catch-up physics must sum all steps");
    near(first.render_ms, 4.0, "Render duration mismatch");
    near(*first.present_ms, 8.0, "Present wait missing");
    near(first.other_ms, 3.0, "Uninstrumented work must remain accounted for");
    near(first.total_ms, first.physics_ms + first.render_ms + *first.present_ms + first.other_ms,
         "Phases must partition the total");
    near(timings.fps(), 50.0, "FPS must use measured wall time");

    timings.begin_frame(at(20));
    timings.begin_phase(FramePhase::render, at(20));
    timings.end_phase(at(21));
    timings.begin_phase(FramePhase::present, at(21));
    timings.end_phase(at(21));
    require(timings.finish_frame(at(22)), "Zero-physics frame must publish");
    require(timings.latest()->physics_steps == 0, "Paused frame must not reuse old physics");
    near(timings.latest()->physics_ms, 0.0, "Paused physics must be zero");
    require(timings.latest()->present_ms == 0.0, "Zero wait is still a measured present");
    near(timings.fps(), 2000.0 / 22.0, "Do not average instantaneous FPS");
    const auto stats = timings.statistics();
    require(stats.count == 2, "Shared statistics must count completed frames");
    near(stats.mean_ms, 11, "Shared statistics mean mismatch");
    near(stats.median_ms, 11, "Even-size median must average middle samples");
    near(stats.p95_ms, 20, "Shared p95 must include stalls");
    near(stats.physics_ms, 2.5, "Statistics must include zero-step frames");
    near(stats.render_ms, 2.5, "Statistics render mean mismatch");
    near(stats.present_ms, 4, "Statistics present mean mismatch");

    timings.begin_frame(at(22));
    timings.begin_phase(FramePhase::render, at(23));
    require(!timings.finish_frame(at(30)), "An unfinished/failed phase cannot publish");
    timings.cancel_frame();
    require(!timings.finish_frame(at(30)), "Cancelled frame cannot publish");
    require(timings.latest()->frame == 2, "Failure must preserve last completed frame");
    timings.begin_frame(at(30));
    timings.begin_phase(FramePhase::physics, at(30));
    timings.begin_frame(at(31)); // Discard another interrupted frame.
    timings.clear_history(); // Scene change inside an active frame.
    require(!timings.latest(), "Scene reset must clear stale display");
    near(timings.fps(), 0.0, "Scene reset must clear FPS history");
    require(timings.statistics().count == 0, "Reset must clear shared statistics");
    timings.begin_phase(FramePhase::physics, at(31));
    timings.end_phase(at(331));
    timings.begin_phase(FramePhase::render, at(331));
    timings.end_phase(at(531));
    require(timings.finish_frame(at(531)), "Headless frame must publish without present");
    require(timings.latest()->frame == 3, "Reset/failure must not reuse CSV frame IDs");
    near(timings.latest()->total_ms, 500.0, "Slow frames must not clamp at simulation delta limit");
    require(!timings.latest()->present_ms, "Headless presentation must be unavailable, not zero");
    near(timings.fps(), 2.0, "Slow frame FPS mismatch");
    timings.begin_frame(at(531));
    timings.finish_frame(at(1031));
    timings.begin_frame(at(1031));
    timings.finish_frame(at(3031));
    near(timings.fps(), 0.5, "Rolling window must retire old frames");
    timings.clear_history();
    timings.begin_frame(at(3031));
    timings.finish_frame(at(3031));
    near(timings.fps(), 0.0, "Zero-duration frame must not divide by zero");
    timings.clear_history();
    for (int i = 0; i < 61; ++i) {
        timings.begin_frame(at(i * 100));
        timings.finish_frame(at(i * 100 + 10));
    }
    require(timings.statistics().count == 60, "Diagnostic history must be bounded");
    near(timings.statistics().cadence_fps, 60000.0 / 5910.0, "Cadence must include inter-frame idle time");
    near(timings.fps(), 100, "Work throughput must exclude inter-frame idle time");

    std::string error;
    std::ostringstream csv;
    const auto previous_locale = std::locale::global(std::locale(std::locale::classic(), new CommaDecimal));
    csv.imbue(std::locale());
    require(FrameTimingCsv::write_header(csv, error), "CSV header must write");
    require(FrameTimingCsv::write_sample(csv, first, "Test, \"scene\"", "interactive", error), "CSV row must write");
    auto headless = first;
    headless.frame = 2;
    headless.present_ms.reset();
    require(FrameTimingCsv::write_sample(csv, headless, "Point", "headless", error), "Headless CSV must write");
    require(FrameTimingCsv::flush(csv, error), "CSV must flush");
    std::locale::global(previous_locale);
    require(csv.str() ==
        "frame,scene,mode,physics_steps,total_ms,physics_ms,render_ms,present_ms,other_ms\n"
        "1,\"Test, \"\"scene\"\"\",\"interactive\",2,20.000000,5.000000,4.000000,8.000000,3.000000\n"
        "2,\"Point\",\"headless\",2,20.000000,5.000000,4.000000,,3.000000\n",
        "CSV schema, escaping, row count, locale, or unavailable present changed");

    WriteFailure write_failure;
    std::ostream bad_write(&write_failure);
    require(!FrameTimingCsv::write_header(bad_write, error) && error.find("header") != std::string::npos,
            "Header write errors must be reported");
    bad_write.clear();
    require(!FrameTimingCsv::write_sample(bad_write, first, "Point", "headless", error) &&
            error.find("write failed") != std::string::npos, "Row write errors must be reported");
    FlushFailure flush_failure;
    std::ostream bad_flush(&flush_failure);
    require(!FrameTimingCsv::flush(bad_flush, error) && error.find("flush failed") != std::string::npos,
            "Buffered flush errors must be reported");
    FrameTimingCsv disabled;
    require(disabled.write(first, "Point", "headless", error) && disabled.close(error),
            "Disabled CSV must be a no-op");
    FrameTimingCsv invalid;
    require(!invalid.open(std::filesystem::temp_directory_path(), error) &&
            error.find("open failed") != std::string::npos, "Invalid file path must fail clearly");
    const auto path = std::filesystem::temp_directory_path() /
        ("pm-frame-timing-" + std::to_string(FrameTimingCollector::Clock::now().time_since_epoch().count()) + ".csv");
    FrameTimingCsv file;
    require(file.open(path, error) && file.write(first, "Point", "interactive", error) && file.close(error),
            "File CSV must open, write and close");
    std::ifstream saved(path);
    const std::string contents{std::istreambuf_iterator<char>(saved), {}};
    require(contents.find("20.000000") != std::string::npos, "Close must flush buffered rows");
    saved.close();
    std::filesystem::remove(path);
    std::cout << "Gallery frame timing tests passed\n";
}
