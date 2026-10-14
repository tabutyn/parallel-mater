// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/device_profiles.hpp>

#include <cstdint>

namespace parallel_mater::gallery {

// Device-adapter facts needed by the shared gallery application. Platform
// targets provide this narrow boundary; input, scenes, and simulation remain
// platform-neutral.
[[nodiscard]] HardwareIdentity platform_hardware_identity();
[[nodiscard]] std::uint64_t platform_available_device_memory() noexcept;

} // namespace parallel_mater::gallery
