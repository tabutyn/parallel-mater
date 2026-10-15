// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

namespace parallel_mater::gallery {

enum class FramePhase { physics, render, present };

struct FrameTimingSample {
    std::uint64_t frame{};
    std::uint32_t physics_steps{};
    double total_ms{};
    double physics_ms{};
    double render_ms{};
    std::optional<double> present_ms{};
    double other_ms{};
};

// Host wall time, not GPU kernel time. Call sites supply phase boundaries only.
// Explicit timestamps allow deterministic tests without sleeps or native clocks.
class FrameTimingCollector {
  public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    void begin_frame(TimePoint now = Clock::now());
    void begin_phase(FramePhase phase, TimePoint now = Clock::now());
    // Returns this phase's milliseconds (also used by the physics catch-up budget).
    double end_phase(TimePoint now = Clock::now());
    // Incomplete/failed frames must not be published. Starting another frame or
    // calling cancel_frame discards them without changing the last good sample.
    bool finish_frame(TimePoint now = Clock::now());
    void cancel_frame();
    void clear_history();
    [[nodiscard]] const std::optional<FrameTimingSample> &latest() const { return latest_; }
    [[nodiscard]] double fps() const;

  private:
    bool active_{};
    std::optional<FramePhase> phase_{};
    TimePoint frame_started_{};
    TimePoint phase_started_{};
    FrameTimingSample pending_{};
    std::optional<FrameTimingSample> latest_{};
    std::uint64_t completed_{};
    std::deque<double> history_{};
    double history_ms_{};
};

// A disabled writer is a no-op. The owner explicitly closes it to report flush
// errors even when the Gallery exits early. No per-frame flush or native APIs.
class FrameTimingCsv {
  public:
    bool open(const std::filesystem::path &path, std::string &error);
    bool write(const FrameTimingSample &sample, std::string_view scene,
               std::string_view mode, std::string &error);
    bool close(std::string &error);

    // Shared serialization also accepts a stream for deterministic I/O tests.
    static bool write_header(std::ostream &stream, std::string &error);
    static bool write_sample(std::ostream &stream, const FrameTimingSample &sample,
                             std::string_view scene, std::string_view mode,
                             std::string &error);
    static bool flush(std::ostream &stream, std::string &error);

  private:
    std::ofstream stream_{};
    std::filesystem::path path_{};
};

} // namespace parallel_mater::gallery
