// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/frame_timing.hpp>
#include <parallel_mater_gallery/gallery_runtime.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace parallel_mater::gallery {

enum class GalleryRendererKind : std::uint8_t {
    software_image,
    native_gpu,
};

enum class GalleryReadbackMode : std::uint8_t {
    host_frame,
    diagnostic_only,
};

// View consumed by every Gallery shell. It contains facts and semantic actions
// only; each shell chooses its own words and input-device instructions. Text
// views remain valid until the session selects another scene or is destroyed.
struct GalleryPresentationState {
    std::string_view platform;
    std::string_view backend;
    std::string_view device;
    GalleryRendererKind renderer{GalleryRendererKind::software_image};
    GalleryReadbackMode readback{GalleryReadbackMode::host_frame};

    bool ready{};
    bool active{};
    bool paused{};
    unsigned selection{};
    std::uint64_t steps{};

    std::string_view scene_id;
    std::string_view scene_name;
    std::string_view scene_summary;
    std::string_view required_features;
    std::string_view required_interactions;
    GalleryControllerState controls{};
    GalleryCountKind count_kind{GalleryCountKind::none};

    std::string_view workload_profile;
    std::string_view execution_contract;
    std::uint32_t fluid_particle_limit{};
    std::uint32_t coupled_body_limit{};
    unsigned width{};
    unsigned height{};
    unsigned target_updates_per_second{};
    std::size_t particle_count{};
    std::size_t moving_body_count{};

    double host_work_frames_per_second{};
    double simulation_to_wall_clock_rate{};
    float interpolation_alpha{};
    bool interpolation_history_valid{};
    std::uint32_t presentation_latency_ticks{};
    std::optional<FrameTimingSample> latest_timing;
    std::string_view build_revision;
};

[[nodiscard]] constexpr std::string_view gallery_backend_label(
    std::string_view backend) noexcept {
    if (backend == "vulkan") return "Vulkan";
    if (backend == "cuda") return "CUDA";
    if (backend == "d3d12") return "Direct3D 12";
    if (backend == "metal") return "Metal";
    if (backend == "slang-cpu") return "Slang CPU";
    if (backend == "parallel-mater-loader") return "Parallel Mater loader";
    return backend.empty() ? std::string_view{"Unknown"} : backend;
}

} // namespace parallel_mater::gallery
