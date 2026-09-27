// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.hpp>

#include <filesystem>
#include <string>

namespace parallel_mater::gallery {

// Writes the API's chronological rolling capture to a self-describing text
// log. The gallery owns persistence; World only owns physics instrumentation.
[[nodiscard]] bool save_physics_debug_capture(
    World &world, std::filesystem::path &output, std::string &error);

[[nodiscard]] bool write_physics_debug_capture(
    World &world, const std::filesystem::path &output, std::string &error);

} // namespace parallel_mater::gallery
