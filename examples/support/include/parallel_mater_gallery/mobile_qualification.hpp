// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/mobile_session.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace parallel_mater::gallery {

inline constexpr std::string_view mobile_gallery_qualification_schema =
    "parallel-mater-mobile-gallery-qualification/v2";

struct MobileQualificationCheck {
    std::string id;
    bool applicable{true};
    bool passed{};
    std::string error;
};

struct MobileQualificationSceneResult {
    std::string id;
    std::string scene;
    std::string requirements;
    bool available{};
    bool skipped{};
    bool passed{};
    unsigned steps{};
    double mean_ms{};
    double p50_ms{};
    double p95_ms{};
    double p99_ms{};
    double physics_mean_ms{};
    double physics_p50_ms{};
    double physics_p95_ms{};
    double physics_p99_ms{};
    double render_mean_ms{};
    double render_p50_ms{};
    double render_p95_ms{};
    double render_p99_ms{};
    double present_mean_ms{};
    double present_p50_ms{};
    double present_p95_ms{};
    double present_p99_ms{};
    double simulation_to_wall_clock_rate{};
    std::size_t timing_window_frames{};
    PmRealtimeStepStatistics realtime_statistics{};
    std::string thermal_state_before;
    std::string thermal_state_after;
    std::uint64_t memory_footprint_bytes{};
    std::string reason;
    std::string error;
};

struct MobileQualificationFailure {
    std::string stage;
    std::string message;
};

struct MobileGalleryQualificationReport {
    std::string revision;
    std::string platform;
    std::string backend;
    std::string execution_contract;
    std::string device;
    std::string renderer;
    std::string readback;
    std::string nonce;
    unsigned frames_per_scene{};
    bool completed{};
    bool passed{};
    std::vector<MobileQualificationCheck> lifecycle;
    std::vector<MobileQualificationCheck> controls;
    std::vector<MobileQualificationSceneResult> scenes;
    std::vector<MobileQualificationFailure> failures;

    [[nodiscard]] std::string json() const;
};

// Native shells supply only their current render target, surface resize hook,
// process-memory observation and atomic report sink. The common coordinator
// owns the scripted event order, semantic controls, scene sweep and schema.
struct MobileGalleryQualificationConfig {
    unsigned frames_per_scene{3U};
    unsigned width{};
    unsigned height{};
    GpuRenderTarget render_target{};
    std::string nonce;
    std::function<bool(unsigned, unsigned, std::string &)> configure_surface;
    std::function<std::uint64_t()> memory_footprint;
    // Platform shells may expose a public thermal-pressure state. A missing
    // callback is reported explicitly as unavailable, never guessed from time.
    std::function<std::string()> thermal_state;
    std::function<void(std::string_view)> checkpoint;
};

[[nodiscard]] MobileGalleryQualificationReport qualify_mobile_gallery(
    MobileGallerySession &session,
    MobileGalleryQualificationConfig config);

} // namespace parallel_mater::gallery
