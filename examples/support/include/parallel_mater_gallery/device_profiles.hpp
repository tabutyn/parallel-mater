// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace parallel_mater::gallery {

inline constexpr std::uint32_t brick_profile_schema_version = 1U;
inline constexpr std::uint32_t brick_scene_version = 1U;
inline constexpr std::uint32_t brick_minimum_count = 1U;
inline constexpr std::uint32_t brick_maximum_count = 4'096U;
inline constexpr float brick_minimum_scale = 0.5F;
inline constexpr float brick_maximum_scale = 2.0F;
inline constexpr std::uint32_t brick_maximum_planes = 16U;
inline constexpr std::uint32_t brick_render_width = 1'920U;
inline constexpr std::uint32_t brick_render_height = 1'080U;

struct BrickSceneConfig {
    std::uint32_t brick_count{384U};
    float brick_scale{1.0F};
    std::uint32_t wall_planes{2U};

    friend bool operator==(const BrickSceneConfig &, const BrickSceneConfig &) = default;
};

struct HardwareIdentity {
    std::string machine_model{};
    std::string cpu_model{};
    std::string gpu_model{};
    std::string gpu_variant{};
    std::uint64_t memory_bytes{};
    std::string backend{};
    std::string operating_system{};
    std::string driver{};
    std::string power_mode{"default"};
};

struct CalibrationMetrics {
    double duration_seconds{};
    double quiet_p95_milliseconds{};
    double quiet_maximum_milliseconds{};
    double collision_p95_milliseconds{};
    double collision_maximum_milliseconds{};
    double simulation_progress_ratio{};
    bool stable{};
    bool no_dropped_steps{};
};

struct VerifiedBrickProfile {
    HardwareIdentity hardware{};
    BrickSceneConfig scene{};
    std::uint32_t width{brick_render_width};
    std::uint32_t height{brick_render_height};
    std::uint32_t scene_version{brick_scene_version};
    std::string solver_version{};
    std::string build_revision{};
    CalibrationMetrics metrics{};
    std::string verified_at{};
    std::string verifier{};
};

struct HardwareRecord {
    std::string vendor{};
    std::string model{};
    std::string variant{};
    std::uint32_t compute_units{};
    double memory_bandwidth_gigabytes_per_second{};
    std::uint64_t memory_bytes{};
    double power_watts{};
    std::string source_url{};
};

struct DeviceProfileCatalog {
    std::uint32_t schema_version{brick_profile_schema_version};
    std::vector<HardwareRecord> hardware{};
    std::vector<VerifiedBrickProfile> verified_profiles{};
};

struct CalibrationSample {
    double milliseconds{};
    bool collision{};
    bool simulation_advanced{};
};

[[nodiscard]] bool validate_brick_config(const BrickSceneConfig &config,
                                         std::string &error) noexcept;
[[nodiscard]] std::uint64_t estimated_metal_contact_bytes(
    std::uint32_t rigid_body_count) noexcept;
[[nodiscard]] const std::array<BrickSceneConfig, 12> &brick_calibration_presets() noexcept;
[[nodiscard]] CalibrationMetrics summarize_calibration(
    const std::vector<CalibrationSample> &samples, double duration_seconds,
    bool stable, bool dropped_steps) noexcept;
[[nodiscard]] bool calibration_passes(const CalibrationMetrics &metrics) noexcept;

[[nodiscard]] bool load_device_profiles(const std::filesystem::path &path,
                                        DeviceProfileCatalog &output,
                                        std::string &error);
[[nodiscard]] bool save_verified_profile(const std::filesystem::path &path,
                                         const VerifiedBrickProfile &profile,
                                         std::string &error);
[[nodiscard]] const VerifiedBrickProfile *find_matching_profile(
    const DeviceProfileCatalog &catalog, const HardwareIdentity &hardware,
    std::uint32_t width = brick_render_width,
    std::uint32_t height = brick_render_height,
    std::string_view solver_version = {}) noexcept;
[[nodiscard]] std::filesystem::path default_local_profiles_path();

} // namespace parallel_mater::gallery
