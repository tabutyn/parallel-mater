// SPDX-License-Identifier: MIT
#pragma once

#include <simulation/runtime.hpp>

#include <filesystem>
#include <string>

namespace parallel_mater::gallery {

struct CommandSceneFrame;

// Writes the API's chronological rolling capture to a self-describing text
// log. The gallery owns persistence; World only owns physics instrumentation.
[[nodiscard]] bool save_physics_debug_capture(
    World &world, std::filesystem::path &output, std::string &error);

[[nodiscard]] bool write_physics_debug_capture(
    World &world, const std::filesystem::path &output, std::string &error);

// Public-command-API capture path used by the unified gallery. The frame is
// already host-visible output, so persistence does not borrow solver memory.
[[nodiscard]] bool save_physics_debug_capture(
    const CommandSceneFrame &frame, std::filesystem::path &output,
    std::string &error);

[[nodiscard]] bool write_physics_debug_capture(
    const CommandSceneFrame &frame, const std::filesystem::path &output,
    std::string &error);

} // namespace parallel_mater::gallery
