// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/physics_debug.hpp>
#include <scene_builder.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() try {
    parallel_mater::gallery::CommandSceneFrame frame;
    frame.revision = 42U;
    frame.rigid_states.resize(1U);
    frame.constraint_states.push_back({PM_TRUE, PM_FALSE, 3.5F});

    const auto stamp = std::chrono::steady_clock::now()
                           .time_since_epoch().count();
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() /
        ("parallel-mater-physics-debug-test-" + std::to_string(stamp) +
         ".log");
    std::string error;
    check(parallel_mater::gallery::write_physics_debug_capture(
              frame, output, error),
          error.empty() ? "write physics capture" : error.c_str());

    std::ifstream stream(output);
    const std::string contents{std::istreambuf_iterator<char>(stream),
                               std::istreambuf_iterator<char>()};
    std::error_code remove_error;
    std::filesystem::remove(output, remove_error);
    check(contents.find("parallel_mater_physics_capture 3") !=
              std::string::npos,
          "capture must report the current format");
    check(contents.find("frame 42 rigid 1 constraints 1") !=
              std::string::npos,
          "capture header must count constraint states");
    check(contents.find(
              "constraint 0 enabled 1 broken 0 applied_impulse 3.5") !=
              std::string::npos,
          "capture must include constraint state");
    return 0;
} catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
}
