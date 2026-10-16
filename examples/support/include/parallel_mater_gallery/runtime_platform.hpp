// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_gallery/device_profiles.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>

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

// Keep unavailable domains visible in the catalog, without letting selection
// silently move a scene to a different physics backend.
[[nodiscard]] constexpr bool gallery_backend_supports(
    GalleryContext context, const PmPhysicalDeviceFeatures &features) noexcept {
    const auto &entry = gallery_entry(context);
    const bool smoke = is_smoke_context(context);
    if (features.rigidBodies != PM_TRUE ||
        (entry.has_fluid && features.particleLiquids != PM_TRUE) ||
        (entry.has_cloth && features.cloth != PM_TRUE) ||
        (entry.has_rope && features.rope != PM_TRUE) ||
        (entry.has_soft_body && features.softBodies != PM_TRUE) ||
        (smoke && features.smoke != PM_TRUE)) return false;
    if (context >= GalleryContext::constraint_fixed &&
        context <= GalleryContext::dump &&
        features.rigidConstraints != PM_TRUE) return false;
    if (context == GalleryContext::fluid_rigid &&
        features.liquidRigidCoupling != PM_TRUE) return false;
    if ((context == GalleryContext::peg_paint || context == GalleryContext::cloth_paint) &&
        features.paint != PM_TRUE) return false;
    const unsigned domains = unsigned(entry.has_fluid) + unsigned(entry.has_cloth) +
        unsigned(entry.has_rope) + unsigned(entry.has_soft_body) + unsigned(smoke);
    return domains < 2U || features.domainCouplings == PM_TRUE;
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
