// SPDX-License-Identifier: MIT
// Public-SDK cloth/rope attachment quality and graph lifecycle fixtures.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
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
constexpr std::uint32_t substeps = 1U;
constexpr float cloth_vertex_mass = 0.25F;
constexpr float rope_mass = 0.5F;
constexpr float rope_spacing = 0.05F;
constexpr std::uint32_t rope_capacity = 8U;
constexpr std::array<PmFloat3, 3U> cloth_vertices{{
    {0.0F, 0.0F, 0.0F},
    {-0.1F, 0.1F, 0.0F},
    {0.1F, 0.1F, 0.0F},
}};
constexpr std::array<std::uint32_t, 3U> cloth_indices{{0U, 1U, 2U}};
constexpr std::array<float, 3U> cloth_inverse_masses{{
    1.0F / cloth_vertex_mass, 0.0F, 0.0F}};
constexpr std::array<PmFloat3, 2U> rope_centerline{{
    {0.0F, 0.0F, 0.0F},
    {0.0F, -0.2F, 0.0F},
}};

struct Snapshot {
    std::array<PmClothVertexState, 3U> cloth{};
    std::vector<PmRopeNodeState> rope{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange cloth_output{};
    PmBufferRange rope_output{};
};

[[nodiscard]] const char *native_driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult status = pmGetNativeDevice(device, &native);
    if (status == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (status != PM_SUCCESS)
        throw std::runtime_error("query cloth-rope attachment backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown cloth-rope attachment backend");
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
    needed.cloth = PM_TRUE;
    needed.rope = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

void require_attachment_capability(const Context &context) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_CLOTH,
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
        throw std::runtime_error("unsupported: cloth-rope attachment capability");
}

[[nodiscard]] PmClothMaterialState cloth_material() {
    PmClothMaterialState material{};
    material.vertexMass = cloth_vertex_mass;
    material.thickness = 0.005F;
    material.stretchCompliance = 1.0e-7F;
    material.bendingCompliance = 1.0e-6F;
    material.velocityDamping = 0.0F;
    material.contactFriction = 0.0F;
    material.solverIterations = 8U;
    material.fracturePersistenceSubsteps = 1U;
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

[[nodiscard]] PmCouplingState attachment_edge(float effective_mass) {
    PmCouplingState edge{};
    edge.type = PM_COUPLING_ROPE_CLOTH;
    edge.first = 0U;
    edge.second = 0U;
    edge.modeFlags = PM_INTERACTION_MODE_ATTACHMENT_BIT;
    edge.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    edge.parameters.attachment.firstVertex = 0U;
    edge.parameters.attachment.lastVertex =
        std::numeric_limits<std::uint32_t>::max();
    edge.parameters.attachment.effectiveMass = effective_mass;
    edge.parameters.attachment.maximumAcceleration = 500.0F;
    return edge;
}

[[nodiscard]] FixtureObjects create_fixture(Context &context,
                                            std::uint32_t coupling_capacity) {
    FixtureObjects fixture{};
    constexpr PmClothCapacity cloth_capacity{3U, 3U, 3U};
    constexpr PmRopeCapacity rope_nodes{rope_capacity};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.clothCount = 1U;
    simulation_info.pClothCapacities = &cloth_capacity;
    simulation_info.ropeCount = 1U;
    simulation_info.pRopeCapacities = &rope_nodes;
    simulation_info.couplingCapacity = coupling_capacity;
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

    const PmClothMaterialState cloth = cloth_material();
    const PmRopeMaterialState rope = rope_material();
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &cloth;
    pipeline_info.ropeMaterialCount = 1U;
    pipeline_info.pRopeMaterials = &rope;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize cloth_bytes = sizeof(PmDomainOutputHeader) +
        cloth_vertices.size() * sizeof(PmClothVertexState);
    constexpr PmDeviceSize rope_bytes = sizeof(PmDomainOutputHeader) +
        rope_capacity * sizeof(PmRopeNodeState);
    constexpr PmFlags usage = PM_BUFFER_SIMULATION_OUTPUT_BIT |
        PM_BUFFER_TRANSFER_DST_BIT;
    fixture.output = context.buffer(cloth_bytes + rope_bytes, usage);
    fixture.cloth_output = {fixture.output.handle, 0U, cloth_bytes};
    fixture.rope_output = {fixture.output.handle, cloth_bytes, rope_bytes};
    return fixture;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture) {
    const PmClothInfo cloth{
        PM_TRUE, 0U, static_cast<std::uint32_t>(cloth_vertices.size()),
        cloth_vertices.data(), static_cast<std::uint32_t>(cloth_indices.size()),
        cloth_indices.data(), cloth_inverse_masses.data()};
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
            throw std::runtime_error(std::string("cloth-rope ") + label +
                                     " failed: " +
                                     std::to_string(submitted));
    };
    submit_stage("cloth initialization", [&] {
        PM_CHECK(pmCmdWriteCloths(context.commands, fixture.simulation, 0U, 1U,
                                 &cloth) == PM_SUCCESS);
    });
    submit_stage("rope initialization", [&] {
        PM_CHECK(pmCmdWriteRopes(context.commands, fixture.simulation, 0U, 1U,
                                &rope) == PM_SUCCESS);
    });
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopyClothState(context.commands, fixture.simulation, 0U,
                                &fixture.cloth_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRopeState(context.commands, fixture.simulation, 0U,
                               &fixture.rope_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *cloth_header =
        reinterpret_cast<const PmDomainOutputHeader *>(base);
    const auto *rope_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.rope_output.offset);
    if (cloth_header->activeCount != cloth_vertices.size() ||
        rope_header->activeCount < 2U ||
        rope_header->activeCount > rope_capacity)
        throw std::runtime_error("cloth-rope public output count mismatch");
    Snapshot result{};
    const auto *cloth = reinterpret_cast<const PmClothVertexState *>(
        cloth_header + 1);
    for (std::size_t index = 0U; index < result.cloth.size(); ++index)
        result.cloth[index] = cloth[index];
    const auto *rope = reinterpret_cast<const PmRopeNodeState *>(rope_header + 1);
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
        throw std::runtime_error("unsupported: cloth-rope attachment features");
    require_attachment_capability(context);
    FixtureObjects fixture = create_fixture(context, 1U);
    initialize_endpoints(context, fixture);
    Snapshot initial = capture(context, fixture);
    const PmCouplingState coupling = attachment_edge(cloth_vertex_mass);
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
        throw std::runtime_error("cloth-rope quality submission failed: " +
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
        throw std::runtime_error("cloth-rope final submission failed: " +
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

[[nodiscard]] Json cloth_json(const PmClothVertexState &state,
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
    result["constraint_force_n"] = vector_json(state.constraintForce);
    return result;
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot) {
    Json cloth = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.cloth.size(); ++index)
        cloth.push_back(cloth_json(snapshot.cloth[index], index));
    Json rope = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.rope.size(); ++index)
        rope.push_back(rope_json(snapshot.rope[index], index));
    Json result = Json::object();
    result["cloth"] = std::move(cloth);
    result["rope"] = std::move(rope);
    return result;
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

void append_vector(std::ostringstream &stream, const float value[3]) {
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
        if (axis != 0U) stream << ',';
        append_float(stream, value[axis]);
    }
}

void append_vector(std::ostringstream &stream, PmFloat3 value) {
    const float components[3]{value.x, value.y, value.z};
    append_vector(stream, components);
}

[[nodiscard]] std::string graph_fingerprint(const PmCouplingState *edge) {
    if (edge == nullptr)
        return sha256("parallel-mater-cloth-rope-attachment-graph/v1;inactive");
    const auto &attachment = edge->parameters.attachment;
    std::ostringstream canonical;
    canonical << "parallel-mater-cloth-rope-attachment-graph/v1;type="
              << static_cast<unsigned>(edge->type)
              << ";first=" << edge->first << ";second=" << edge->second
              << ";modes=" << edge->modeFlags
              << ";direction=BIDIRECTIONAL;material="
              << static_cast<unsigned>(edge->materialProfile)
              << ";first-vertex=" << attachment.firstVertex
              << ";last-vertex=" << attachment.lastVertex
              << ";effective-mass=";
    append_float(canonical, attachment.effectiveMass);
    canonical << ";maximum-acceleration=";
    append_float(canonical, attachment.maximumAcceleration);
    return sha256(canonical.str());
}

[[nodiscard]] std::string endpoint_fingerprint(const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-cloth-rope-attachment-endpoints/v1";
    for (const auto &state : snapshot.cloth) {
        canonical << ";cloth-position=";
        append_vector(canonical, state.position);
        canonical << ",velocity=";
        append_vector(canonical, state.velocity);
    }
    for (const auto &state : snapshot.rope) {
        canonical << ";rope-position=";
        append_vector(canonical, state.position);
        canonical << ",velocity=";
        append_vector(canonical, state.velocity);
    }
    return sha256(canonical.str());
}

[[nodiscard]] Json resource_json(std::string id, bool active,
                                 std::uint64_t generation,
                                 std::string fingerprint) {
    Json result = Json::object();
    result["id"] = std::move(id);
    result["active"] = active;
    result["generation"] = generation;
    result["state_sha256"] = std::move(fingerprint);
    return result;
}

void add_checkpoint(Json &observations, std::string id, std::string outcome,
                    const PmCouplingState *edge,
                    std::uint64_t graph_generation,
                    const Snapshot &endpoints) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        "cloth-rope-attachment-graph", edge != nullptr, graph_generation,
        graph_fingerprint(edge)));
    resources.push_back(resource_json(
        "cloth-rope-endpoints", true, 1U,
        endpoint_fingerprint(endpoints)));
    Json checkpoint = Json::object();
    checkpoint["id"] = std::move(id);
    checkpoint["outcome"] = std::move(outcome);
    checkpoint["resources"] = std::move(resources);
    observations.push_back(std::move(checkpoint));
}

struct LifecycleRun {
    Json driver{};
    Json observations{};
};

[[nodiscard]] LifecycleRun run_lifecycle_fixture() {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: cloth-rope attachment features");
    require_attachment_capability(context);
    FixtureObjects fixture = create_fixture(context, 1U);
    initialize_endpoints(context, fixture);
    const PmCouplingState baseline = attachment_edge(0.25F);
    const PmCouplingState replacement = attachment_edge(0.30F);
    const PmCouplingState recovery = attachment_edge(0.35F);
    const std::array<PmCouplingState, 2U> overflow{{replacement, recovery}};
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    Json observations = Json::array();

    const auto mutate = [&](const PmCouplingState *edges,
                            std::uint32_t count, PmResult expected) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, count,
                                  edges) == expected);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        record_copies(context, fixture);
        PM_CHECK(context.submit() == PM_SUCCESS);
        return read_snapshot(context, fixture);
    };

    add_checkpoint(observations, "baseline", "PM_SUCCESS", &baseline, 1U,
                   mutate(&baseline, 1U, PM_SUCCESS));
    add_checkpoint(observations, "replacement", "PM_SUCCESS", &replacement,
                   2U, mutate(&replacement, 1U, PM_SUCCESS));
    add_checkpoint(observations, "rejected-overflow",
                   "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U,
                   mutate(overflow.data(),
                          static_cast<std::uint32_t>(overflow.size()),
                          PM_ERROR_CAPACITY_EXCEEDED));
    add_checkpoint(observations, "removal", "PM_SUCCESS", nullptr, 3U,
                   mutate(nullptr, 0U, PM_SUCCESS));
    add_checkpoint(observations, "recovery", "PM_SUCCESS", &recovery, 4U,
                   mutate(&recovery, 1U, PM_SUCCESS));

    Json driver = driver_json(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}

} // namespace

std::string conformance_cloth_rope_attachment_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "cloth-rope-attachment-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/cloth-rope-attachment-v1.json");
    try {
        QualityRun run = run_quality_fixture();
        Json fixture = Json::object();
        fixture["id"] = "supported-hanging-rope";
        fixture["timestep_seconds"] = static_cast<double>(timestep);
        fixture["step_count"] = static_cast<std::uint64_t>(step_count);
        fixture["substeps"] = static_cast<std::uint64_t>(substeps);
        fixture["gravity_m_s2"] = vector_json(PmFloat3{0.0F, -9.81F, 0.0F});
        fixture["cloth_vertex_mass_kg"] =
            static_cast<double>(cloth_vertex_mass);
        fixture["cloth_vertices_m"] = Json::array();
        for (const auto vertex : cloth_vertices)
            fixture["cloth_vertices_m"].push_back(vector_json(vertex));
        fixture["cloth_triangle_indices"] = Json::array();
        for (const auto index : cloth_indices)
            fixture["cloth_triangle_indices"].push_back(
                static_cast<std::uint64_t>(index));
        fixture["cloth_inverse_masses_per_kg"] = Json::array();
        for (const auto value : cloth_inverse_masses)
            fixture["cloth_inverse_masses_per_kg"].push_back(
                static_cast<double>(value));
        fixture["cloth_stretch_compliance_m_per_n"] = 1.0e-7;
        fixture["cloth_velocity_damping_per_s"] = 0.0;
        fixture["cloth_solver_iterations"] = static_cast<std::uint64_t>(8U);
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
        fixture["attached_cloth_vertex"] = static_cast<std::uint64_t>(0U);
        fixture["anchor_effective_mass_kg"] =
            static_cast<double>(cloth_vertex_mass);
        fixture["maximum_cloth_acceleration_m_s2"] = 500.0;
        fixture["oracle"] =
            "public-endpoint-drift-and-reciprocal-constraint-reaction";
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

std::string conformance_cloth_rope_attachment_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "cloth-rope-attachment-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/cloth-rope-attachment-lifecycle-v1.json");
    try {
        LifecycleRun run = run_lifecycle_fixture();
        Json families = Json::array();
        families.push_back("cloth");
        families.push_back("rope");
        Json unrelated = Json::array();
        unrelated.push_back("cloth-rope-endpoints");
        Json attempt = Json::object();
        attempt["fixture"] = "cloth-rope-attachment-public-lifecycle";
        attempt["status"] = "completed";
        Json item = Json::object();
        item["attempt"] = std::move(attempt);
        item["execution_status"] = "completed";
        item["families"] = std::move(families);
        item["id"] =
            "cloth-rope-attachment-capacity-replacement-removal-rollback";
        item["observations"] = std::move(run.observations);
        item["unrelated_resources"] = std::move(unrelated);
        Json cases = Json::array();
        cases.push_back(std::move(item));
        result["execution_status"] = "completed";
        result["driver"] = std::move(run.driver);
        result["cases"] = std::move(cases);
        if (status != nullptr) *status = ExecutionStatus::completed;
    } catch (const std::exception &error) {
        const bool unsupported = unsupported_error(error);
        result["execution_status"] = unsupported ? "unsupported" : "failed";
        result["reason"] = error.what();
        result["driver"] = unavailable_driver();
        result["cases"] = Json::array();
        if (status != nullptr) *status = unsupported ?
            ExecutionStatus::unsupported : ExecutionStatus::failed;
    }
    return result.serialize();
}

} // namespace parallel_mater::conformance
