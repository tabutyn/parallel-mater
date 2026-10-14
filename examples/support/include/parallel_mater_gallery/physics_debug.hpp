// SPDX-License-Identifier: MIT
#pragma once

#include <filesystem>
#include <string>

namespace parallel_mater::gallery {

struct CommandSceneFrame;

// The frame is already host-visible public-API output; persistence never
// borrows private solver memory.
[[nodiscard]] bool save_physics_debug_capture(
    const CommandSceneFrame &frame, std::filesystem::path &output,
    std::string &error);

[[nodiscard]] bool write_physics_debug_capture(
    const CommandSceneFrame &frame, const std::filesystem::path &output,
    std::string &error);

} // namespace parallel_mater::gallery
