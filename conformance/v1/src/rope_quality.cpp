// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>
#include <parallel_mater_scene/command_scene.hpp>
#include <parallel_mater_scene/scene.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace parallel_mater::conformance {
namespace {
using namespace parallel_mater::gallery;

constexpr float timestep = 1.0F / 60.0F;
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
    rope.options.maximum_speed = 2.99F;
    rope.options.solver_iterations = 24U;
    rope.options.self_collision = true;
    return rope;
}

struct RopeAuditFixture {
    RopeDefinition rope{};
    std::uint32_t step_count{};
    std::vector<std::uint32_t> checkpoint_steps{};
    std::vector<std::array<std::uint32_t, 2U>> excluded_probe_pairs{};
    std::vector<std::uint32_t> fixed_endpoint_indices{};
};

[[nodiscard]] RopeDefinition audit_rope(
    std::string name, std::vector<Vec3> centerline, float node_spacing,
    float radius, float mass, float step, float damping,
    float maximum_speed, std::uint32_t iterations) {
    RopeDefinition rope{};
    rope.name = std::move(name);
    rope.centerline = std::move(centerline);
    rope.options.node_spacing = node_spacing;
    rope.options.radius = radius;
    rope.options.mass = mass;
    rope.options.stretch_compliance = 0.0F;
    rope.options.velocity_damping = damping;
    rope.options.maximum_substep_timestep = step;
    rope.options.friction = 0.0F;
    rope.options.maximum_speed = maximum_speed;
    rope.options.solver_iterations = iterations;
    rope.options.self_collision = true;
    return rope;
}

[[nodiscard]] RopeAuditFixture free_audit_fixture() {
    RopeAuditFixture fixture{};
    fixture.rope = audit_rope(
        "free-folded-four-node-long-run",
        {{0.0F, 0.0F, 0.0F},
         {0.1F, 0.0F, 0.0F},
         {0.06F, 0.091651514F, 0.0F},
         {0.02F, 0.0F, 0.0F}},
        0.101F, 0.06F, 0.4F, timestep, 0.5F, 2.99F, 24U);
    fixture.step_count = 240U;
    fixture.checkpoint_steps = {1U, 30U, 120U, 240U};
    fixture.excluded_probe_pairs = {{{1U, 3U}}};
    return fixture;
}

[[nodiscard]] RopeAuditFixture fixed_audit_fixture() {
    RopeAuditFixture fixture{};
    fixture.rope = audit_rope(
        "fixed-endpoint-folded-six-node",
        {{-0.1F, 0.0F, 0.0F},
         {0.0F, 0.0F, 0.0F},
         {0.1F, 0.0F, 0.0F},
         {0.06F, 0.091651514F, 0.0F},
         {0.02F, 0.0F, 0.0F},
         {0.12F, 0.0F, 0.0F}},
        0.101F, 0.06F, 0.6F, timestep, 0.5F, 2.99F, 24U);
    fixture.step_count = 60U;
    fixture.checkpoint_steps = {1U, 30U, 60U};
    fixture.excluded_probe_pairs = {{{2U, 4U}}};
    fixture.fixed_endpoint_indices = {0U, 5U};
    return fixture;
}

[[nodiscard]] RopeAuditFixture capacity_audit_fixture() {
    RopeAuditFixture fixture{};
    std::vector<Vec3> centerline;
    centerline.reserve(256U);
    for (std::uint32_t index = 0U; index < 256U; ++index)
        centerline.push_back({0.02F * static_cast<float>(index), 0.0F, 0.0F});
    fixture.rope = audit_rope(
        "exact-capacity-256-node-rope", std::move(centerline),
        0.0200001F, 0.011F, 2.56F, timestep, 0.2F, 0.59F, 8U);
    fixture.step_count = 4U;
    fixture.checkpoint_steps = {1U, 4U};
    return fixture;
}

[[nodiscard]] RopeAuditFixture free_fixed_tick_audit_fixture() {
    RopeAuditFixture fixture{};
    fixture.rope = audit_rope(
        "free-folded-four-node-fixed-tick",
        {{0.0F, 0.0F, 0.0F},
         {0.02F, 0.02F, 0.0F},
         {0.10F, 0.08F, 0.0F},
         {0.11F, 0.0F, 0.0F}},
        0.101F, 0.06F, 0.4F, 1.0F / 60.0F, 0.5F, 3.0F, 8U);
    fixture.step_count = 144U;
    fixture.checkpoint_steps = {1U, 18U, 72U, 144U};
    fixture.excluded_probe_pairs = {{{1U, 3U}}};
    return fixture;
}

[[nodiscard]] RopeAuditFixture fixed_endpoint_fixed_tick_audit_fixture() {
    RopeAuditFixture fixture{};
    fixture.rope = audit_rope(
        "fixed-endpoint-folded-six-node-fixed-tick",
        {{-0.1F, 0.0F, 0.0F},
         {0.0F, 0.0F, 0.0F},
         {0.02F, 0.02F, 0.0F},
         {0.10F, 0.08F, 0.0F},
         {0.11F, 0.0F, 0.0F},
         {0.21F, 0.0F, 0.0F}},
        0.101F, 0.06F, 0.6F, 1.0F / 60.0F, 0.5F, 3.0F, 8U);
    fixture.step_count = 36U;
    fixture.checkpoint_steps = {1U, 18U, 36U};
    fixture.excluded_probe_pairs = {{{2U, 4U}}};
    fixture.fixed_endpoint_indices = {0U, 5U};
    return fixture;
}

[[nodiscard]] RopeAuditFixture capacity_fixed_tick_audit_fixture() {
    RopeAuditFixture fixture{};
    std::vector<Vec3> centerline;
    centerline.reserve(256U);
    for (std::uint32_t index = 0U; index < 256U; ++index)
        centerline.push_back({0.02F * static_cast<float>(index), 0.0F, 0.0F});
    fixture.rope = audit_rope(
        "exact-capacity-256-node-fixed-tick", std::move(centerline),
        0.0200001F, 0.011F, 2.56F, 1.0F / 60.0F, 0.2F, 0.6F, 8U);
    fixture.step_count = 2U;
    fixture.checkpoint_steps = {1U, 2U};
    return fixture;
}

[[nodiscard]] TriangleMesh anchor_mesh() {
    TriangleMesh mesh{};
    mesh.name = "fixed-anchor-collision";
    constexpr float half = 0.001F;
    const std::array<Vec3, 8U> points{{
        {-half, -half, -half}, {half, -half, -half},
        {half, half, -half}, {-half, half, -half},
        {-half, -half, half}, {half, -half, half},
        {half, half, half}, {-half, half, half}}};
    for (Vec3 point : points) mesh.vertices.push_back({.position = point});
    mesh.indices = {0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,
                    1,5,6,1,6,2,2,6,7,2,7,3,3,7,4,3,4,0};
    return mesh;
}

[[nodiscard]] SceneDefinition audit_scene(const RopeAuditFixture &fixture) {
    SceneDefinition scene{};
    RopeDefinition rope = fixture.rope;
    if (!fixture.fixed_endpoint_indices.empty()) {
        scene.meshes.push_back(anchor_mesh());
        RigidBodyDefinition first{};
        first.name = "fixed-anchor-first";
        first.source_name = first.name;
        first.mesh_indices = {0U};
        first.options.motion = MotionType::static_body;
        first.options.initial_state.position = {-10.0F, 0.0F, 0.0F};
        RigidBodyDefinition last = first;
        last.name = "fixed-anchor-last";
        last.source_name = last.name;
        last.options.initial_state.position = {10.0F, 0.0F, 0.0F};
        scene.rigid_bodies = {first, last};
        rope.first_body = 0;
        rope.last_body = 1;
        rope.options.first.enabled = true;
        rope.options.last.enabled = true;
        rope.options.first.local_anchor = {
            rope.centerline.front().x - first.options.initial_state.position.x,
            rope.centerline.front().y - first.options.initial_state.position.y,
            rope.centerline.front().z - first.options.initial_state.position.z};
        rope.options.last.local_anchor = {
            rope.centerline.back().x - last.options.initial_state.position.x,
            rope.centerline.back().y - last.options.initial_state.position.y,
            rope.centerline.back().z - last.options.initial_state.position.z};
    }
    scene.ropes.push_back(std::move(rope));
    return scene;
}

[[nodiscard]] Json indices_json(const std::vector<std::uint32_t> &indices) {
    Json result = Json::array();
    for (std::uint32_t index : indices)
        result.push_back(static_cast<std::uint64_t>(index));
    return result;
}

[[nodiscard]] Json pairs_json(
    const std::vector<std::array<std::uint32_t, 2U>> &pairs) {
    Json result = Json::array();
    for (const auto pair : pairs) {
        Json encoded = Json::array();
        encoded.push_back(static_cast<std::uint64_t>(pair[0]));
        encoded.push_back(static_cast<std::uint64_t>(pair[1]));
        result.push_back(std::move(encoded));
    }
    return result;
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

[[nodiscard]] Json run_audit_fixture(const RopeAuditFixture &definition,
                                     Json &driver, bool &driver_written) {
    SceneDefinition scene = audit_scene(definition);
    CommandScene simulation{};
    CommandSceneInstance instance{};
    std::string error;
    if (!CommandScene::create(scene, 0U, simulation, instance, error))
        throw std::runtime_error("create rope audit fixture " +
                                 definition.rope.name + ": " + error);
    require_sampled_self_capability(simulation);
    if (!driver_written) {
        driver = driver_json(simulation);
        driver_written = true;
    }

    Json checkpoints = Json::array();
    CommandSceneFrame frame{};
    for (std::uint32_t step = 1U; step <= definition.step_count; ++step) {
        if (!simulation.step(definition.rope.options.maximum_substep_timestep,
                             substeps, {0.0F, 0.0F, 0.0F}, frame,
                             error, false, true, true))
            throw std::runtime_error("step rope audit fixture " +
                                     definition.rope.name + ": " + error);
        if (frame.rope_nodes.size() != 1U ||
            frame.rope_nodes.front().size() !=
                definition.rope.centerline.size())
            throw std::runtime_error("rope audit output topology changed for " +
                                     definition.rope.name);
        if (std::find(definition.checkpoint_steps.begin(),
                      definition.checkpoint_steps.end(), step) !=
            definition.checkpoint_steps.end()) {
            Json checkpoint = Json::object();
            checkpoint["step"] = static_cast<std::uint64_t>(step);
            checkpoint["nodes"] = nodes_json(frame.rope_nodes.front());
            checkpoints.push_back(std::move(checkpoint));
        }
    }

    if (definition.checkpoint_steps.empty() ||
        definition.checkpoint_steps.back() != definition.step_count)
        throw std::runtime_error("rope audit final checkpoint is missing for " +
                                 definition.rope.name);

    const RopeDefinition &rope = definition.rope;
    Json fixture = Json::object();
    fixture["id"] = rope.name;
    fixture["point_count"] =
        static_cast<std::uint64_t>(rope.centerline.size());
    fixture["sampled_node_count"] =
        static_cast<std::uint64_t>(frame.rope_nodes.front().size());
    fixture["declared_node_capacity"] =
        static_cast<std::uint64_t>(rope.centerline.size());
    fixture["timestep_seconds"] =
        static_cast<double>(rope.options.maximum_substep_timestep);
    fixture["step_count"] = static_cast<std::uint64_t>(definition.step_count);
    Json checkpoint_steps = Json::array();
    for (std::uint32_t step : definition.checkpoint_steps)
        checkpoint_steps.push_back(static_cast<std::uint64_t>(step));
    fixture["checkpoint_steps"] = std::move(checkpoint_steps);
    fixture["contact_checkpoint_step"] = static_cast<std::uint64_t>(1U);
    fixture["substeps"] = static_cast<std::uint64_t>(substeps);
    fixture["gravity_m_s2"] = vector_json(Vec3{});
    fixture["node_spacing_m"] = static_cast<double>(rope.options.node_spacing);
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
    fixture["excluded_probe_pairs"] =
        pairs_json(definition.excluded_probe_pairs);
    fixture["fixed_endpoint_indices"] =
        indices_json(definition.fixed_endpoint_indices);
    fixture["initial_centerline_m"] = centerline_json(rope.centerline);
    fixture["initial_nodes"] = initial_nodes_json(rope.centerline);
    fixture["checkpoints"] = std::move(checkpoints);
    fixture["oracle"] = "public-input-output-sampled-node-audit";
    return fixture;
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

std::string conformance_rope_self_contact_audit_quality(
    const std::filesystem::path &root) {
    constexpr const char *profile_id = "rope-self-contact-v2";
    Json driver = Json::object();
    bool driver_written = false;
    Json fixtures = Json::array();
    fixtures.push_back(run_audit_fixture(
        free_audit_fixture(), driver, driver_written));
    fixtures.push_back(run_audit_fixture(
        fixed_audit_fixture(), driver, driver_written));
    fixtures.push_back(run_audit_fixture(
        capacity_audit_fixture(), driver, driver_written));

    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality" /
        (std::string(profile_id) + ".json"));
    result["execution_status"] = "completed";
    result["driver"] = std::move(driver);
    result["claim"] = "sampled-nodes-only";
    result["fixtures"] = std::move(fixtures);
    return result.serialize();
}

std::string conformance_rope_self_contact_fixed_tick_audit_quality(
    const std::filesystem::path &root) {
    constexpr const char *profile_id = "rope-self-contact-v3";
    Json driver = Json::object();
    bool driver_written = false;
    Json fixtures = Json::array();
    fixtures.push_back(run_audit_fixture(
        free_fixed_tick_audit_fixture(), driver, driver_written));
    fixtures.push_back(run_audit_fixture(
        fixed_endpoint_fixed_tick_audit_fixture(), driver, driver_written));
    fixtures.push_back(run_audit_fixture(
        capacity_fixed_tick_audit_fixture(), driver, driver_written));

    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality" /
        (std::string(profile_id) + ".json"));
    result["execution_status"] = "completed";
    result["driver"] = std::move(driver);
    result["claim"] = "sampled-nodes-only";
    result["schedule"] = "realtime-single-step-v1";
    result["fixtures"] = std::move(fixtures);
    return result.serialize();
}

} // namespace parallel_mater::conformance
