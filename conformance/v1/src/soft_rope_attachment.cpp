// SPDX-License-Identifier: MIT
// Public-SDK soft-body/rope attachment quality fixture.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace parallel_mater::conformance {
namespace {
using parallel_mater::test::Buffer;
using parallel_mater::test::Context;

constexpr float timestep = 1.0F / 120.0F;
constexpr std::uint32_t step_count = 120U;
constexpr std::uint32_t substeps = 2U;
constexpr float soft_node_mass = 0.25F;
constexpr float rope_mass = 0.5F;
constexpr float rope_spacing = 0.05F;
constexpr std::uint32_t rope_capacity = 8U;
constexpr std::array<PmFloat3, 4U> soft_nodes{{
    {-0.1F, 0.0F, -0.1F},
    {0.1F, 0.0F, -0.1F},
    {0.0F, 0.0F, 0.1F},
    {0.0F, 0.2F, 0.0F},
}};
constexpr std::array<float, 4U> soft_inverse_masses{{
    1.0F / soft_node_mass, 1.0F / soft_node_mass,
    1.0F / soft_node_mass, 0.0F}};
constexpr std::array<PmSoftBodyBond, 6U> soft_bonds{{
    {0U, 1U, 0.2F},
    {0U, 2U, 0.223606798F},
    {0U, 3U, 0.244948974F},
    {1U, 2U, 0.223606798F},
    {1U, 3U, 0.244948974F},
    {2U, 3U, 0.223606798F},
}};
constexpr std::array<std::uint32_t, 12U> surface_indices{{
    0U, 1U, 2U,
    0U, 3U, 1U,
    0U, 2U, 3U,
    1U, 3U, 2U,
}};
constexpr std::array<PmFloat3, 2U> rope_centerline{{
    {0.0F, 0.0F, -0.0333333333F},
    {0.0F, -0.2F, -0.0333333333F},
}};

[[nodiscard]] std::array<PmSoftBodySurfaceBinding, 4U>
surface_bindings() {
    std::array<PmSoftBodySurfaceBinding, 4U> result{};
    for (std::uint32_t index = 0U; index < result.size(); ++index) {
        result[index].nodes[0] = index;
        result[index].weights[0] = 1.0F;
    }
    return result;
}

struct Snapshot {
    std::array<PmSoftBodyNodeState, 4U> soft{};
    std::array<PmFloat3, 4U> surface{};
    std::vector<PmRopeNodeState> rope{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange soft_output{};
    PmBufferRange surface_output{};
    PmBufferRange rope_output{};
};

[[nodiscard]] const char *native_driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult status = pmGetNativeDevice(device, &native);
    if (status == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (status != PM_SUCCESS)
        throw std::runtime_error("query soft-rope attachment backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown soft-rope attachment backend");
}

[[nodiscard]] Json driver_json(const Context &context) {
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    Json result = Json::object();
    result["name"] = native_driver_name(context.device);
    result["device"] = properties.deviceName;
    result["vendor_id"] = static_cast<std::uint64_t>(properties.vendorId);
    result["device_id"] = static_cast<std::uint64_t>(properties.deviceId);
    result["driver_version"] = std::to_string(properties.driverVersion);
    return result;
}

[[nodiscard]] Json unavailable_driver() {
    Json result = Json::object();
    result["name"] = "unavailable";
    result["device"] = "unavailable";
    return result;
}

[[nodiscard]] PmPhysicalDeviceFeatures required_features() {
    PmPhysicalDeviceFeatures needed{};
    needed.softBodies = PM_TRUE;
    needed.rope = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

void require_attachment_capability(const Context &context) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_SOFT_BODY,
        PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
        PM_INTERACTION_MODE_ATTACHMENT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error("unsupported: soft-rope attachment capability");
}

[[nodiscard]] PmSoftBodyMaterialState soft_material() {
    PmSoftBodyMaterialState material{};
    material.nodeMass = soft_node_mass;
    material.nodeRadius = 0.025F;
    material.stretchCompliance = 1.0e-7F;
    material.velocityDamping = 0.0F;
    material.springDamping = 0.0F;
    material.contactFriction = 0.0F;
    material.shapeMatchingStiffness = 0.0F;
    material.maximumProjectionFraction = 0.2F;
    material.constraintVelocityResponse = 0.0F;
    material.maximumSpeed = 20.0F;
    material.solverIterations = 8U;
    return material;
}

[[nodiscard]] PmRopeMaterialState rope_material() {
    PmRopeMaterialState material{};
    material.nodeSpacing = rope_spacing;
    material.radius = 0.025F;
    material.mass = rope_mass;
    material.stretchCompliance = 1.0e-7F;
    material.velocityDamping = 0.0F;
    material.maximumSubstepTimestep = timestep / substeps;
    material.friction = 0.0F;
    material.maximumSpeed = 20.0F;
    material.solverIterations = 8U;
    material.selfCollision = PM_FALSE;
    return material;
}

[[nodiscard]] PmCouplingState attachment_edge() {
    PmCouplingState edge{};
    edge.type = PM_COUPLING_ROPE_SOFT_BODY;
    edge.first = 0U;
    edge.second = 0U;
    // The public rope-soft-body representation requires its CONTACT and
    // ATTACHMENT modes together. The oracle below separately proves that the
    // contact channel is inactive and cannot substitute for attachment.
    edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_ATTACHMENT_BIT;
    edge.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    edge.parameters.mechanical.contactDistance = 0.01F;
    edge.parameters.mechanical.friction = 0.0F;
    edge.parameters.attachment.supportRadiusScale = 0.0F;
    edge.parameters.attachment.contactSupportRadiusScale = 0.0F;
    edge.parameters.attachment.maximumAcceleration = 500.0F;
    edge.parameters.attachment.attachFirst = PM_TRUE;
    return edge;
}

[[nodiscard]] FixtureObjects create_fixture(Context &context) {
    constexpr PmSoftBodyCapacity soft_capacity{
        static_cast<std::uint32_t>(soft_nodes.size()),
        static_cast<std::uint32_t>(soft_bonds.size()),
        static_cast<std::uint32_t>(soft_nodes.size()),
        static_cast<std::uint32_t>(surface_indices.size())};
    constexpr PmRopeCapacity rope_nodes{rope_capacity};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.softBodyCount = 1U;
    simulation_info.pSoftBodyCapacities = &soft_capacity;
    simulation_info.ropeCount = 1U;
    simulation_info.pRopeCapacities = &rope_nodes;
    simulation_info.couplingCapacity = 1U;
    FixtureObjects fixture{};
    PM_CHECK(pmCreateSimulation(context.device, &simulation_info,
                                &fixture.simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, fixture.simulation,
                                      &requirements);
    const PmSimulationMemoryBinding binding{
        context.allocate(requirements.persistent), 0U,
        context.allocate(requirements.scratch), 0U};
    PM_CHECK(pmBindSimulationMemory(context.device, fixture.simulation,
                                    &binding) == PM_SUCCESS);

    const PmSoftBodyMaterialState soft = soft_material();
    const PmRopeMaterialState rope = rope_material();
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.softBodyMaterialCount = 1U;
    pipeline_info.pSoftBodyMaterials = &soft;
    pipeline_info.ropeMaterialCount = 1U;
    pipeline_info.pRopeMaterials = &rope;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize soft_bytes = sizeof(PmDomainOutputHeader) +
        soft_nodes.size() * sizeof(PmSoftBodyNodeState);
    constexpr PmDeviceSize surface_bytes = sizeof(PmDomainOutputHeader) +
        soft_nodes.size() * sizeof(PmFloat3);
    constexpr PmDeviceSize rope_bytes = sizeof(PmDomainOutputHeader) +
        rope_capacity * sizeof(PmRopeNodeState);
    fixture.output = context.buffer(
        soft_bytes + surface_bytes + rope_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.soft_output = {fixture.output.handle, 0U, soft_bytes};
    fixture.surface_output = {fixture.output.handle, soft_bytes, surface_bytes};
    fixture.rope_output = {
        fixture.output.handle, soft_bytes + surface_bytes, rope_bytes};
    return fixture;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture) {
    const auto bindings = surface_bindings();
    const PmSoftBodyInfo soft{
        PM_TRUE, 0U, static_cast<std::uint32_t>(soft_nodes.size()),
        soft_nodes.data(), static_cast<std::uint32_t>(soft_bonds.size()),
        soft_bonds.data(), soft_inverse_masses.data(),
        static_cast<std::uint32_t>(soft_nodes.size()), soft_nodes.data(),
        static_cast<std::uint32_t>(surface_indices.size()),
        surface_indices.data(), bindings.data()};
    PmRopeInfo rope{};
    rope.active = PM_TRUE;
    rope.materialIndex = 0U;
    rope.pointCount = static_cast<std::uint32_t>(rope_centerline.size());
    rope.pCenterline = rope_centerline.data();
    const auto submit_stage = [&](const char *label, const auto &record) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        record();
        const PmResult submitted = context.submit();
        if (submitted != PM_SUCCESS)
            throw std::runtime_error(std::string("soft-rope ") + label +
                                     " failed: " +
                                     std::to_string(submitted));
    };
    submit_stage("soft-body initialization", [&] {
        PM_CHECK(pmCmdWriteSoftBodies(context.commands, fixture.simulation,
                                      0U, 1U, &soft) == PM_SUCCESS);
    });
    submit_stage("rope initialization", [&] {
        PM_CHECK(pmCmdWriteRopes(context.commands, fixture.simulation, 0U, 1U,
                                &rope) == PM_SUCCESS);
    });
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopySoftBodyState(context.commands, fixture.simulation, 0U,
                                   &fixture.soft_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodySurface(context.commands, fixture.simulation, 0U,
                                     &fixture.surface_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRopeState(context.commands, fixture.simulation, 0U,
                               &fixture.rope_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *soft_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.soft_output.offset);
    const auto *surface_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.surface_output.offset);
    const auto *rope_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.rope_output.offset);
    if (soft_header->activeCount != soft_nodes.size() ||
        surface_header->activeCount != soft_nodes.size() ||
        rope_header->activeCount < 2U ||
        rope_header->activeCount > rope_capacity)
        throw std::runtime_error("soft-rope public output count mismatch");
    Snapshot result{};
    const auto *soft = reinterpret_cast<const PmSoftBodyNodeState *>(
        soft_header + 1);
    const auto *surface = reinterpret_cast<const PmFloat3 *>(
        surface_header + 1);
    const auto *rope = reinterpret_cast<const PmRopeNodeState *>(
        rope_header + 1);
    std::copy_n(soft, result.soft.size(), result.soft.begin());
    std::copy_n(surface, result.surface.size(), result.surface.begin());
    result.rope.assign(rope, rope + rope_header->activeCount);
    return result;
}

[[nodiscard]] Snapshot capture(Context &context,
                               const FixtureObjects &fixture) {
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    return read_snapshot(context, fixture);
}

struct QualityRun {
    Json driver{};
    Snapshot initial{};
    Snapshot previous{};
    Snapshot final{};
};

[[nodiscard]] QualityRun run_quality_fixture() {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: soft-rope attachment features");
    require_attachment_capability(context);
    FixtureObjects fixture = create_fixture(context);
    initialize_endpoints(context, fixture);
    Snapshot initial = capture(context, fixture);
    const PmCouplingState coupling = attachment_edge();
    const PmStepInfo step{timestep, substeps, {0.0F, -9.81F, 0.0F}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &coupling) == PM_SUCCESS);
    for (std::uint32_t index = 0U; index + 1U < step_count; ++index)
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
    record_copies(context, fixture);
    PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("soft-rope quality submission failed: " +
                                 std::to_string(submitted));
    Snapshot previous = read_snapshot(context, fixture);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation, &step) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("soft-rope final submission failed: " +
                                 std::to_string(submitted));
    Snapshot final = read_snapshot(context, fixture);
    Json driver = driver_json(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(initial), std::move(previous),
            std::move(final)};
}

[[nodiscard]] Json vector_json(PmFloat3 value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    return result;
}

[[nodiscard]] Json soft_json(const PmSoftBodyNodeState &state,
                             std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(state.position);
    result["velocity_m_s"] = vector_json(state.velocity);
    result["rope_reaction_force_n"] = vector_json(state.ropeContactForce);
    return result;
}

[[nodiscard]] Json rope_json(const PmRopeNodeState &state,
                             std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(state.position);
    result["velocity_m_s"] = vector_json(state.velocity);
    result["soft_body_contact_force_n"] =
        vector_json(state.softBodyContactForce);
    return result;
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot) {
    Json soft = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.soft.size(); ++index)
        soft.push_back(soft_json(snapshot.soft[index], index));
    Json surface = Json::array();
    for (const auto value : snapshot.surface)
        surface.push_back(vector_json(value));
    Json rope = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.rope.size(); ++index)
        rope.push_back(rope_json(snapshot.rope[index], index));
    Json result = Json::object();
    result["soft_body"] = std::move(soft);
    result["soft_body_surface_m"] = std::move(surface);
    result["rope"] = std::move(rope);
    return result;
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}

} // namespace

std::string conformance_soft_rope_attachment_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "soft-rope-attachment-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/soft-rope-attachment-v1.json");
    try {
        QualityRun run = run_quality_fixture();
        Json fixture = Json::object();
        fixture["id"] = "supported-hanging-rope";
        fixture["timestep_seconds"] = static_cast<double>(timestep);
        fixture["step_count"] = static_cast<std::uint64_t>(step_count);
        fixture["substeps"] = static_cast<std::uint64_t>(substeps);
        fixture["gravity_m_s2"] = vector_json({0.0F, -9.81F, 0.0F});
        fixture["soft_node_mass_kg"] = static_cast<double>(soft_node_mass);
        fixture["soft_nodes_m"] = Json::array();
        for (const auto value : soft_nodes)
            fixture["soft_nodes_m"].push_back(vector_json(value));
        fixture["soft_inverse_masses_per_kg"] = Json::array();
        for (const auto value : soft_inverse_masses)
            fixture["soft_inverse_masses_per_kg"].push_back(
                static_cast<double>(value));
        Json bonds = Json::array();
        for (const auto &bond : soft_bonds) {
            Json item = Json::array();
            item.push_back(static_cast<std::uint64_t>(bond.first));
            item.push_back(static_cast<std::uint64_t>(bond.second));
            item.push_back(static_cast<double>(bond.restLength));
            bonds.push_back(std::move(item));
        }
        fixture["soft_bonds"] = std::move(bonds);
        fixture["surface_triangle_indices"] = Json::array();
        for (const auto index : surface_indices)
            fixture["surface_triangle_indices"].push_back(
                static_cast<std::uint64_t>(index));
        fixture["soft_node_radius_m"] = 0.025;
        fixture["soft_stretch_compliance_m_per_n"] = 1.0e-7;
        fixture["soft_velocity_damping_per_s"] = 0.0;
        fixture["soft_solver_iterations"] = static_cast<std::uint64_t>(8U);
        fixture["rope_centerline_m"] = Json::array();
        for (const auto point : rope_centerline)
            fixture["rope_centerline_m"].push_back(vector_json(point));
        fixture["rope_node_spacing_m"] = static_cast<double>(rope_spacing);
        fixture["rope_radius_m"] = 0.025;
        fixture["rope_total_mass_kg"] = static_cast<double>(rope_mass);
        fixture["rope_stretch_compliance_m_per_n"] = 1.0e-7;
        fixture["rope_velocity_damping_per_s"] = 0.0;
        fixture["rope_solver_iterations"] = static_cast<std::uint64_t>(8U);
        fixture["attached_rope_end"] = "first";
        fixture["attached_surface_triangle"] = Json::array();
        for (const auto index : {0U, 1U, 2U})
            fixture["attached_surface_triangle"].push_back(
                static_cast<std::uint64_t>(index));
        fixture["anchor_support_radius_scale"] = 0.0;
        fixture["anchor_contact_support_radius_scale"] = 0.0;
        fixture["maximum_soft_body_acceleration_m_s2"] = 500.0;
        fixture["oracle"] =
            "public-surface-drift-momentum-reaction-and-inactive-contact";
        fixture["initial"] = snapshot_json(run.initial);
        fixture["previous"] = snapshot_json(run.previous);
        fixture["final"] = snapshot_json(run.final);
        Json fixtures = Json::array();
        fixtures.push_back(std::move(fixture));
        result["execution_status"] = "completed";
        result["driver"] = std::move(run.driver);
        result["fixtures"] = std::move(fixtures);
        if (status != nullptr) *status = ExecutionStatus::completed;
    } catch (const std::exception &error) {
        const bool unsupported = unsupported_error(error);
        result["execution_status"] = unsupported ? "unsupported" : "failed";
        result["reason"] = error.what();
        result["driver"] = unavailable_driver();
        result["fixtures"] = Json::array();
        if (status != nullptr) *status = unsupported ?
            ExecutionStatus::unsupported : ExecutionStatus::failed;
    }
    return result.serialize();
}

} // namespace parallel_mater::conformance
