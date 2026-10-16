// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_gallery/device_profiles.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace parallel_mater::gallery {

struct GalleryBackendOption {
    PmNativeBackend backend{};
    std::string device_name{};
    PmPhysicalDeviceFeatures features{};
};

[[nodiscard]] constexpr const char *gallery_backend_name(
    PmNativeBackend backend) noexcept {
    switch (backend) {
    case PM_NATIVE_BACKEND_CUDA: return "cuda";
    case PM_NATIVE_BACKEND_D3D12: return "d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "vulkan";
    case PM_NATIVE_BACKEND_METAL: return "metal";
    }
    return "unknown";
}

[[nodiscard]] inline std::optional<PmNativeBackend> gallery_backend_from_name(
    std::string_view name) noexcept {
    if (name == "cuda") return PM_NATIVE_BACKEND_CUDA;
    if (name == "d3d12" || name == "dx12" || name == "directx12")
        return PM_NATIVE_BACKEND_D3D12;
    if (name == "vulkan") return PM_NATIVE_BACKEND_VULKAN;
    if (name == "metal") return PM_NATIVE_BACKEND_METAL;
    return std::nullopt;
}

// Device-adapter facts needed by the shared gallery application. Platform
// targets provide this narrow boundary; input, scenes, and simulation remain
// platform-neutral.
[[nodiscard]] std::vector<GalleryBackendOption> platform_gallery_backends();
[[nodiscard]] HardwareIdentity platform_hardware_identity(
    std::optional<PmNativeBackend> backend = std::nullopt);
[[nodiscard]] std::uint64_t platform_available_device_memory(
    std::optional<PmNativeBackend> backend = std::nullopt) noexcept;

} // namespace parallel_mater::gallery
