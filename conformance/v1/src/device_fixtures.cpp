// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>
#include <parallel_mater_scene/command_scene.hpp>
#include <parallel_mater_scene/scene.hpp>

#include "../../../tests/gallery/rigid_mesh_checks.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int parallel_mater_fluid_pair_quality_fixture(int argc, char **argv);

namespace parallel_mater::conformance {
namespace {
using namespace parallel_mater::gallery;

std::filesystem::path bundled_asset(const std::filesystem::path &root,
                                    const char *name) {
    auto path = root / "examples/assets" / name;
    if (!std::filesystem::is_regular_file(path)) path = root / "assets" / name;
    return path;
}

Json driver_json(PmPhysicalDevice physical, const char *name = "slang-metal") {
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(physical, &properties);
    Json result = Json::object();
    result["name"] = name;
    result["device"] = properties.deviceName;
    return result;
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read fixture output " + path.string());
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

} // namespace

std::string conformance_rigid_brick_quality(const std::filesystem::path &root) {
    SceneDefinition authored{};
    std::string error;
    if (!load_glb_scene(bundled_asset(root, "RigidBody.glb"), authored, error))
        throw std::runtime_error("load rigid quality scene: " + error);
    SceneDefinition scene{};
    if (!make_brick_scene(authored, {8U, 1.0F, 1U}, scene, error))
        throw std::runtime_error("generate rigid quality scene: " + error);
    std::vector<rigid_mesh_checks::MeshSamples> geometry;
    geometry.reserve(scene.rigid_bodies.size());
    for (std::size_t body = 0U; body < scene.rigid_bodies.size(); ++body)
        geometry.emplace_back(scene, body);

    CommandScene simulation{};
    CommandSceneInstance instance{};
    CommandSceneFrame frame{};
    if (!CommandScene::create(scene, 2U, simulation, instance, error))
        throw std::runtime_error("create rigid quality scene: " + error);
    float maximum{};
    std::uint32_t maximum_frame{};
    for (std::uint32_t frame_index = 0U; frame_index < 300U; ++frame_index) {
        if (!simulation.step(1.0F / 60.0F, 4U, {0.0F, -9.81F, 0.0F},
                             frame, error, true))
            throw std::runtime_error("step rigid quality scene: " + error);
        const auto state = [&](std::size_t body) {
            const auto &value = frame.rigid_states.at(body);
            return RigidBodyState{
                {value.position[0], value.position[1], value.position[2]},
                {value.orientation[0], value.orientation[1],
                 value.orientation[2], value.orientation[3]}, {}, {}};
        };
        for (std::size_t first = 0U; first < geometry.size(); ++first)
            for (std::size_t second = first + 1U; second < geometry.size(); ++second) {
                if (!scene.rigid_bodies[first].name.starts_with("Brick_") ||
                    !scene.rigid_bodies[second].name.starts_with("Brick_")) continue;
                const float depth = std::max(
                    geometry[first].penetration(state(first), geometry[second], state(second)),
                    geometry[second].penetration(state(second), geometry[first], state(first)));
                if (depth > maximum) { maximum = depth; maximum_frame = frame_index; }
            }
    }
    Json fixture = Json::object();
    fixture["id"] = "eight-brick-stack";
    fixture["scene"] = "RigidBody.glb";
    fixture["body_count"] = static_cast<std::uint64_t>(8U);
    fixture["frames"] = static_cast<std::uint64_t>(300U);
    fixture["substeps"] = static_cast<std::uint64_t>(4U);
    fixture["timestep_seconds"] = 1.0 / 60.0;
    fixture["oracle"] = "independent-closed-mesh-surface-sampling";
    fixture["maximum_penetration_m"] = static_cast<double>(maximum);
    fixture["maximum_penetration_frame"] =
        static_cast<std::uint64_t>(maximum_frame);
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = "rigid-brick-penetration-v1";
    result["profile_sha256"] = sha256_file(
        root / "conformance/v1/quality/rigid-brick-penetration-v1.json");
    result["execution_status"] = "completed";
    result["driver"] = driver_json(simulation.physical_device());
    result["fixture"] = std::move(fixture);
    return result.serialize();
}

std::string conformance_fluid_pair_quality(
    const std::filesystem::path &temporary_output) {
    std::filesystem::create_directories(temporary_output.parent_path());
    std::string output = temporary_output.string();
    char program[] = "ipad-quality";
    char output_option[] = "--output";
    char driver_option[] = "--driver-name";
    char driver_name[] = "slang-metal";
    char *arguments[]{program, output_option, output.data(), driver_option,
                      driver_name};
    const int status = parallel_mater_fluid_pair_quality_fixture(5, arguments);
    if (status == 0) return read_file(temporary_output);
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = "core-conservation-stability-v1";
    result["profile_sha256"] =
        "a59a81c71b5ddf9ae1f0a47cd8e61e2074c24d929b15318852863b2c92cd9326";
    result["execution_status"] = status == 77 ? "unsupported" : "failed";
    result["reason"] = status == 77
        ? "The selected driver does not advertise separate-system equal-fluid coupling."
        : "The public-API equal-fluid quality fixture failed.";
    result["fixtures"] = Json::array();
    return result.serialize();
}

} // namespace parallel_mater::conformance
