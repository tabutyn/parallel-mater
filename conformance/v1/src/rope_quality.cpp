// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>
#include <parallel_mater_scene/command_scene.hpp>
#include <parallel_mater_scene/scene.hpp>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace parallel_mater::conformance {
namespace {
using namespace parallel_mater::gallery;

constexpr float timestep = 0.01F;
constexpr std::uint32_t substeps = 1U;
constexpr std::uint32_t topology_exclusion = 2U;

[[nodiscard]] const char *driver_name(const CommandScene &simulation) {
    PmNativeDeviceInfo native{};
    const PmResult result = pmGetNativeDevice(simulation.device(), &native);
    if (result == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (result != PM_SUCCESS)
        throw std::runtime_error("query rope quality backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unrecognized rope quality backend");
}

[[nodiscard]] Json vector_json(PmFloat3 value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    return result;
}

[[nodiscard]] Json vector_json(Vec3 value) {
    return vector_json(PmFloat3{value.x, value.y, value.z});
}

[[nodiscard]] Json node_json(const PmRopeNodeState &node,
                             std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(node.position);
    result["velocity_m_s"] = vector_json(node.velocity);
    return result;
}

[[nodiscard]] Json nodes_json(const std::vector<PmRopeNodeState> &nodes) {
    Json result = Json::array();
    for (std::uint32_t index = 0U; index < nodes.size(); ++index)
        result.push_back(node_json(nodes[index], index));
    return result;
}

[[nodiscard]] Json centerline_json(const std::vector<Vec3> &centerline) {
    Json result = Json::array();
    for (Vec3 point : centerline) result.push_back(vector_json(point));
    return result;
}

[[nodiscard]] Json initial_nodes_json(const std::vector<Vec3> &centerline) {
    Json result = Json::array();
    for (std::uint32_t index = 0U; index < centerline.size(); ++index) {
        Json node = Json::object();
        node["index"] = static_cast<std::uint64_t>(index);
        node["position_m"] = vector_json(centerline[index]);
        node["velocity_m_s"] = vector_json(Vec3{});
        result.push_back(std::move(node));
    }
    return result;
}

[[nodiscard]] Json driver_json(const CommandScene &simulation) {
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(simulation.physical_device(), &properties);
    Json result = Json::object();
    result["name"] = driver_name(simulation);
    result["device"] = properties.deviceName;
    result["vendor_id"] = static_cast<std::uint64_t>(properties.vendorId);
    result["device_id"] = static_cast<std::uint64_t>(properties.deviceId);
    result["driver_version"] = std::to_string(properties.driverVersion);
    return result;
}

[[nodiscard]] RopeDefinition fixture_rope() {
    RopeDefinition rope{};
    rope.name = "folded-four-node-sampled-rope";
    rope.centerline = {
        {0.0F, 0.0F, 0.0F},
        {0.1F, 0.0F, 0.0F},
        {0.06F, 0.091651514F, 0.0F},
        {0.02F, 0.0F, 0.0F},
    };
    rope.options.node_spacing = 0.101F;
    rope.options.radius = 0.06F;
    rope.options.mass = 0.4F;
    rope.options.stretch_compliance = 0.0F;
    rope.options.velocity_damping = 0.0F;
    rope.options.maximum_substep_timestep = timestep;
    rope.options.friction = 0.0F;
    rope.options.maximum_speed = 100.0F;
    rope.options.solver_iterations = 24U;
    rope.options.self_collision = true;
    return rope;
}

void require_sampled_self_capability(const CommandScene &simulation) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_ROPE, PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_SCOPE_SELF, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            simulation.physical_device(), &key, &capability) != PM_SUCCESS)
        throw std::runtime_error(
            "advertised sampled rope self-contact capability is missing");
    if (capability.maxGraphEdgesPerSimulation !=
        PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(
            "intrinsic rope self-contact unexpectedly consumes graph edges");
}
} // namespace

std::string conformance_rope_self_contact_quality(
    const std::filesystem::path &root) {
    SceneDefinition scene{};
    scene.ropes.push_back(fixture_rope());

    CommandScene simulation{};
    CommandSceneInstance instance{};
    std::string error;
    if (!CommandScene::create(scene, 0U, simulation, instance, error))
        throw std::runtime_error("create rope quality fixture: " + error);
    require_sampled_self_capability(simulation);

    CommandSceneFrame final{};
    if (!simulation.step(timestep, substeps, {0.0F, 0.0F, 0.0F}, final,
                         error, false, true, true))
        throw std::runtime_error("step rope quality fixture: " + error);
    if (final.rope_nodes.size() != 1U ||
        final.rope_nodes.front().size() != scene.ropes.front().centerline.size())
        throw std::runtime_error("rope quality output topology changed");

    const RopeDefinition &rope = scene.ropes.front();
    Json fixture = Json::object();
    fixture["id"] = rope.name;
    fixture["point_count"] =
        static_cast<std::uint64_t>(rope.centerline.size());
    fixture["sampled_node_count"] =
        static_cast<std::uint64_t>(final.rope_nodes.front().size());
    fixture["timestep_seconds"] = static_cast<double>(timestep);
    fixture["substeps"] = static_cast<std::uint64_t>(substeps);
    fixture["gravity_m_s2"] = vector_json(Vec3{});
    fixture["node_spacing_m"] =
        static_cast<double>(rope.options.node_spacing);
    fixture["radius_m"] = static_cast<double>(rope.options.radius);
    fixture["mass_kg"] = static_cast<double>(rope.options.mass);
    fixture["stretch_compliance_m_per_n"] =
        static_cast<double>(rope.options.stretch_compliance);
    fixture["velocity_damping_per_s"] =
        static_cast<double>(rope.options.velocity_damping);
    fixture["maximum_substep_timestep_seconds"] =
        static_cast<double>(rope.options.maximum_substep_timestep);
    fixture["friction"] = static_cast<double>(rope.options.friction);
    fixture["maximum_speed_m_s"] =
        static_cast<double>(rope.options.maximum_speed);
    fixture["solver_iterations"] =
        static_cast<std::uint64_t>(rope.options.solver_iterations);
    fixture["self_collision"] = rope.options.self_collision;
    fixture["topology_exclusion_index_distance"] =
        static_cast<std::uint64_t>(topology_exclusion);
    Json excluded_pair = Json::array();
    excluded_pair.push_back(static_cast<std::uint64_t>(1U));
    excluded_pair.push_back(static_cast<std::uint64_t>(3U));
    fixture["excluded_probe_pair"] = std::move(excluded_pair);
    fixture["initial_centerline_m"] = centerline_json(rope.centerline);
    fixture["initial_nodes"] = initial_nodes_json(rope.centerline);
    fixture["final_nodes"] = nodes_json(final.rope_nodes.front());
    fixture["oracle"] =
        "public-input-output-sampled-node-pair-distance-and-topology";

    constexpr const char *profile_id = "rope-self-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality" /
        (std::string(profile_id) + ".json"));
    result["execution_status"] = "completed";
    result["driver"] = driver_json(simulation);
    result["fixture"] = std::move(fixture);
    return result.serialize();
}

} // namespace parallel_mater::conformance
