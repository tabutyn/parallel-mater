// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/rigid_mesh_checks.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>
#include <parallel_mater_scene/command_scene.hpp>
#include <parallel_mater_scene/scene.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace parallel_mater::conformance {
namespace {
using namespace parallel_mater::gallery;

constexpr std::uint32_t brick_frames = 300U;
constexpr std::uint32_t attachment_frames = 2400U;
constexpr std::uint32_t substeps = 4U;
constexpr float timestep = 1.0F / 60.0F;

[[nodiscard]] std::filesystem::path bundled_asset(
    const std::filesystem::path &root, const char *name) {
    auto path = root / "examples/assets" / name;
    if (!std::filesystem::is_regular_file(path)) path = root / "assets" / name;
    return path;
}

[[nodiscard]] const char *driver_name(const CommandScene &simulation) {
    PmNativeDeviceInfo native{};
    const PmResult result = pmGetNativeDevice(simulation.device(), &native);
    if (result == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (result != PM_SUCCESS)
        throw std::runtime_error("query rigid quality backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unrecognized rigid quality backend");
}

[[nodiscard]] Json driver_json(const CommandScene &simulation) {
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(simulation.physical_device(), &properties);
    Json result = Json::object();
    result["name"] = driver_name(simulation);
    result["device"] = properties.deviceName;
    return result;
}

[[nodiscard]] Vec3 add(Vec3 a, Vec3 b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] Vec3 subtract(Vec3 a, Vec3 b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] Vec3 scale(Vec3 value, float amount) noexcept {
    return {value.x * amount, value.y * amount, value.z * amount};
}

[[nodiscard]] float dot(Vec3 a, Vec3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] Vec3 cross(Vec3 a, Vec3 b) noexcept {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

[[nodiscard]] float length(Vec3 value) noexcept {
    return std::sqrt(dot(value, value));
}

[[nodiscard]] Vec3 normalize_or(Vec3 value, Vec3 fallback) noexcept {
    const float magnitude = length(value);
    return magnitude > 1.0e-8F ? scale(value, 1.0F / magnitude) : fallback;
}

[[nodiscard]] Quaternion quaternion(const PmRigidState &state) noexcept {
    return {state.orientation[0], state.orientation[1],
            state.orientation[2], state.orientation[3]};
}

[[nodiscard]] Vec3 position(const PmRigidState &state) noexcept {
    return {state.position[0], state.position[1], state.position[2]};
}

[[nodiscard]] Quaternion multiply(Quaternion a, Quaternion b) noexcept {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

[[nodiscard]] Vec3 rotate(Quaternion orientation, Vec3 value) noexcept {
    const Quaternion result = multiply(
        multiply(orientation, {value.x, value.y, value.z, 0.0F}),
        {-orientation.x, -orientation.y, -orientation.z, orientation.w});
    return {result.x, result.y, result.z};
}

[[nodiscard]] Vec3 world_anchor(const PmRigidState &state,
                                Vec3 local_anchor) noexcept {
    return add(position(state), rotate(quaternion(state), local_anchor));
}

[[nodiscard]] Vec3 world_axis(const PmRigidState &state,
                              Quaternion local_orientation,
                              Vec3 local_axis) noexcept {
    return normalize_or(
        rotate(multiply(quaternion(state), local_orientation), local_axis),
        local_axis);
}

[[nodiscard]] std::size_t body_index(const SceneDefinition &scene,
                                     std::string_view name) {
    const auto found = std::find_if(
        scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
        [&](const RigidBodyDefinition &body) { return body.source_name == name; });
    if (found == scene.rigid_bodies.end())
        throw std::runtime_error("rigid attachment scene body is missing");
    return static_cast<std::size_t>(found - scene.rigid_bodies.begin());
}

[[nodiscard]] Json quality_result(const std::filesystem::path &root,
                                  const char *profile_id,
                                  const CommandScene &simulation,
                                  Json fixture) {
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality" / (std::string(profile_id) + ".json"));
    result["execution_status"] = "completed";
    result["driver"] = driver_json(simulation);
    result["fixture"] = std::move(fixture);
    return result;
}
} // namespace

std::string conformance_rigid_brick_quality(const std::filesystem::path &root) {
    const auto scene_path = bundled_asset(root, "RigidBody.glb");
    SceneDefinition authored{};
    std::string error;
    if (!load_glb_scene(scene_path, authored, error))
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
    for (std::uint32_t frame_index = 0U; frame_index < brick_frames;
         ++frame_index) {
        if (!simulation.step(timestep, substeps, {0.0F, -9.81F, 0.0F},
                             frame, error, true))
            throw std::runtime_error("step rigid quality scene: " + error);
        const auto state = [&](std::size_t body) {
            const auto &value = frame.rigid_states.at(body);
            return RigidBodyState{
                {value.position[0], value.position[1], value.position[2]},
                {value.orientation[0], value.orientation[1],
                 value.orientation[2], value.orientation[3]}, {}, {}};
        };
        for (std::size_t first = 0U; first < geometry.size(); ++first) {
            for (std::size_t second = first + 1U; second < geometry.size();
                 ++second) {
                if (!scene.rigid_bodies[first].name.starts_with("Brick_") ||
                    !scene.rigid_bodies[second].name.starts_with("Brick_"))
                    continue;
                const float depth = std::max(
                    geometry[first].penetration(
                        state(first), geometry[second], state(second)),
                    geometry[second].penetration(
                        state(second), geometry[first], state(first)));
                if (depth > maximum) {
                    maximum = depth;
                    maximum_frame = frame_index;
                }
            }
        }
    }
    Json fixture = Json::object();
    fixture["id"] = "eight-brick-stack";
    fixture["scene"] = "RigidBody.glb";
    fixture["scene_sha256"] = sha256_file(scene_path);
    fixture["body_count"] = static_cast<std::uint64_t>(8U);
    fixture["frames"] = static_cast<std::uint64_t>(brick_frames);
    fixture["substeps"] = static_cast<std::uint64_t>(substeps);
    fixture["timestep_seconds"] = 1.0 / 60.0;
    fixture["oracle"] = "independent-closed-mesh-surface-sampling";
    fixture["maximum_penetration_m"] = static_cast<double>(maximum);
    fixture["maximum_penetration_frame"] =
        static_cast<std::uint64_t>(maximum_frame);
    return quality_result(root, "rigid-brick-penetration-v1", simulation,
                          std::move(fixture)).serialize();
}

std::string conformance_rigid_attachment_quality(
    const std::filesystem::path &root) {
    const auto scene_path = bundled_asset(root, "ConstraintHinge.glb");
    SceneDefinition scene{};
    std::string error;
    if (!load_glb_scene(scene_path, scene, error))
        throw std::runtime_error("load rigid attachment scene: " + error);
    if (scene.rigid_bodies.size() != 6U ||
        scene.rigid_constraints.size() != 5U)
        throw std::runtime_error("rigid attachment scene topology changed");
    const std::size_t rod_index = body_index(scene, "Ground.002");
    const RigidBodyState initial_rod =
        scene.rigid_bodies[rod_index].options.initial_state;

    CommandScene simulation{};
    CommandSceneInstance instance{};
    CommandSceneFrame frame{};
    if (!CommandScene::create(scene, 2U, simulation, instance, error))
        throw std::runtime_error("create rigid attachment scene: " + error);
    float maximum_hinge_anchor_error{};
    float maximum_hinge_axis_error{};
    float maximum_slider_sideways_drift{};
    float maximum_slider_axis_error{};
    float maximum_slider_travel{};
    bool constraints_enabled = true;
    bool constraints_unbroken = true;
    for (std::uint32_t frame_index = 0U; frame_index < attachment_frames;
         ++frame_index) {
        if (!simulation.step(timestep, substeps, {0.0F, -9.81F, 0.0F},
                             frame, error, false))
            throw std::runtime_error("step rigid attachment scene: " + error);
        if (frame.rigid_states.size() != scene.rigid_bodies.size() ||
            frame.constraint_states.size() < scene.rigid_constraints.size())
            throw std::runtime_error("rigid attachment output topology changed");
        for (std::size_t index = 0U; index < scene.rigid_constraints.size();
             ++index) {
            const auto &constraint = scene.rigid_constraints[index];
            const PmRigidState &a = frame.rigid_states[constraint.body_a];
            const PmRigidState &b = frame.rigid_states[constraint.body_b];
            const Vec3 anchor_error = subtract(
                world_anchor(b, constraint.options.local_anchor_b),
                world_anchor(a, constraint.options.local_anchor_a));
            if (constraint.options.type == RigidConstraintType::hinge) {
                maximum_hinge_anchor_error = std::max(
                    maximum_hinge_anchor_error, length(anchor_error));
                const Vec3 axis_a = world_axis(
                    a, constraint.options.local_orientation_a,
                    {0.0F, 0.0F, 1.0F});
                const Vec3 axis_b = world_axis(
                    b, constraint.options.local_orientation_b,
                    {0.0F, 0.0F, 1.0F});
                maximum_hinge_axis_error = std::max(
                    maximum_hinge_axis_error, length(cross(axis_a, axis_b)));
            } else if (constraint.options.type == RigidConstraintType::slider) {
                const Vec3 axis_a = world_axis(
                    a, constraint.options.local_orientation_a,
                    {1.0F, 0.0F, 0.0F});
                const Vec3 axis_b = world_axis(
                    b, constraint.options.local_orientation_b,
                    {1.0F, 0.0F, 0.0F});
                const Vec3 sideways = subtract(
                    anchor_error, scale(axis_a, dot(anchor_error, axis_a)));
                maximum_slider_sideways_drift = std::max(
                    maximum_slider_sideways_drift, length(sideways));
                maximum_slider_axis_error = std::max(
                    maximum_slider_axis_error, length(cross(axis_a, axis_b)));
            }
            constraints_enabled = constraints_enabled &&
                frame.constraint_states[index].enabled == PM_TRUE;
            constraints_unbroken = constraints_unbroken &&
                frame.constraint_states[index].broken == PM_FALSE;
        }
        maximum_slider_travel = std::max(
            maximum_slider_travel,
            length(subtract(position(frame.rigid_states[rod_index]),
                            initial_rod.position)));
    }

    Json fixture = Json::object();
    fixture["id"] = "hinge-slider-settling";
    fixture["scene"] = "ConstraintHinge.glb";
    fixture["scene_sha256"] = sha256_file(scene_path);
    fixture["body_count"] = static_cast<std::uint64_t>(6U);
    fixture["constraint_count"] = static_cast<std::uint64_t>(5U);
    fixture["frames"] = static_cast<std::uint64_t>(attachment_frames);
    fixture["substeps"] = static_cast<std::uint64_t>(substeps);
    fixture["timestep_seconds"] = 1.0 / 60.0;
    fixture["oracle"] = "authored-frame-public-state-invariants";
    fixture["maximum_hinge_anchor_error_m"] =
        static_cast<double>(maximum_hinge_anchor_error);
    fixture["maximum_hinge_axis_error"] =
        static_cast<double>(maximum_hinge_axis_error);
    fixture["maximum_slider_sideways_drift_m"] =
        static_cast<double>(maximum_slider_sideways_drift);
    fixture["maximum_slider_axis_error"] =
        static_cast<double>(maximum_slider_axis_error);
    fixture["maximum_slider_travel_m"] =
        static_cast<double>(maximum_slider_travel);
    fixture["constraints_enabled"] = constraints_enabled;
    fixture["constraints_unbroken"] = constraints_unbroken;
    return quality_result(root, "rigid-attachment-v1", simulation,
                          std::move(fixture)).serialize();
}

} // namespace parallel_mater::conformance
