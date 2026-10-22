// SPDX-License-Identifier: MIT
// Public-SDK soft-body/rope contact quality fixture.
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

constexpr float preparation_timestep = 0.01F;
constexpr PmFloat3 preparation_gravity{100.0F, -100.0F, 0.0F};
constexpr float contact_timestep = 0.01F;
constexpr float contact_distance = 0.1F;
constexpr float soft_node_mass = 2.0F;
constexpr float rope_mass = 3.0F;
constexpr float rope_spacing = 0.2F;
constexpr std::uint32_t rope_capacity = 8U;

constexpr std::array<PmFloat3, 2U> rope_centerline{{
    {-0.2F, 0.0F, 0.0F},
    {0.2F, 0.0F, 0.0F}}};

constexpr std::array<PmFloat3, 4U> soft_nodes{{
    {-0.01F, 0.02F, 0.0F},
    {0.49F, 0.5F, -0.5F},
    {-0.51F, 0.5F, -0.5F},
    {-0.01F, 0.5F, 0.5F}}};
constexpr std::array<PmFloat3, 4U> prepared_soft_nodes{{
    {0.0F, 0.01F, 0.0F},
    {0.5F, 0.49F, -0.5F},
    {-0.5F, 0.49F, -0.5F},
    {0.0F, 0.49F, 0.5F}}};
constexpr std::array<float, 4U> soft_inverse_masses{{
    0.5F, 0.5F, 0.5F, 0.5F}};
constexpr std::array<PmSoftBodyBond, 6U> soft_bonds{{
    {0U, 1U, 0.85463442476886F},
    {0U, 2U, 0.85463442476886F},
    {0U, 3U, 0.693108938046538F},
    {1U, 2U, 1.0F},
    {1U, 3U, 1.118033988749895F},
    {2U, 3U, 1.118033988749895F}}};
constexpr std::array<PmFloat3, 10U> surface_vertices{{
    soft_nodes[0], soft_nodes[1], soft_nodes[2], soft_nodes[3],
    {0.24F, 0.26F, -0.25F}, {-0.26F, 0.26F, -0.25F},
    {-0.01F, 0.26F, 0.25F}, {-0.01F, 0.5F, -0.5F},
    {0.24F, 0.5F, 0.0F}, {-0.26F, 0.5F, 0.0F}}};
constexpr std::array<std::uint32_t, 48U> surface_indices{{
    0U, 5U, 4U, 5U, 2U, 7U, 4U, 7U, 1U, 5U, 7U, 4U,
    0U, 4U, 6U, 4U, 1U, 8U, 6U, 8U, 3U, 4U, 8U, 6U,
    0U, 6U, 5U, 6U, 3U, 9U, 5U, 9U, 2U, 6U, 9U, 5U,
    1U, 7U, 8U, 7U, 2U, 9U, 8U, 9U, 3U, 7U, 9U, 8U}};

[[nodiscard]] std::array<PmSoftBodySurfaceBinding, 10U>
surface_bindings() {
    std::array<PmSoftBodySurfaceBinding, 10U> result{};
    for (std::uint32_t index = 0U; index < 4U; ++index) {
        result[index].nodes[0] = index;
        result[index].weights[0] = 1.0F;
    }
    constexpr std::array<std::array<std::uint32_t, 2U>, 6U> edges{{
        {{0U, 1U}}, {{0U, 2U}}, {{0U, 3U}},
        {{1U, 2U}}, {{1U, 3U}}, {{2U, 3U}}}};
    for (std::uint32_t index = 0U; index < edges.size(); ++index) {
        auto &binding = result[index + 4U];
        binding.nodes[0] = edges[index][0];
        binding.nodes[1] = edges[index][1];
        binding.weights[0] = binding.weights[1] = 0.5F;
    }
    return result;
}

struct Snapshot {
    std::vector<PmRopeNodeState> rope{};
    std::array<PmSoftBodyNodeState, 4U> soft{};
    std::array<PmFloat3, 10U> surface{};
};

struct SnapshotRanges {
    PmBufferRange rope{};
    PmBufferRange soft{};
    PmBufferRange surface{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    SnapshotRanges initial{};
    SnapshotRanges contacted{};
    SnapshotRanges released{};
};

[[nodiscard]] Json vector_json(PmFloat3 value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    return result;
}

template <std::size_t Size>
[[nodiscard]] Json vectors_json(const std::array<PmFloat3, Size> &values) {
    Json result = Json::array();
    for (const auto value : values) result.push_back(vector_json(value));
    return result;
}

template <std::size_t Size>
[[nodiscard]] Json floats_json(const std::array<float, Size> &values) {
    Json result = Json::array();
    for (const auto value : values)
        result.push_back(static_cast<double>(value));
    return result;
}

template <std::size_t Size>
[[nodiscard]] Json indices_json(const std::array<std::uint32_t, Size> &values) {
    Json result = Json::array();
    for (const auto value : values)
        result.push_back(static_cast<std::uint64_t>(value));
    return result;
}

[[nodiscard]] const char *native_driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult status = pmGetNativeDevice(device, &native);
    if (status == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (status != PM_SUCCESS)
        throw std::runtime_error("query soft-rope contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown soft-rope contact backend");
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
    needed.rope = PM_TRUE;
    needed.softBodies = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

[[nodiscard]] bool supports(PmPhysicalDevice physical) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_SOFT_BODY,
        PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
        PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE};
    PmInteractionCapability capability{};
    return pmGetPhysicalDeviceInteractionCapability(
               physical, &key, &capability) == PM_SUCCESS &&
        capability.maxGraphEdgesPerSimulation !=
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE;
}

[[nodiscard]] PmRopeMaterialState rope_material() {
    PmRopeMaterialState material{};
    material.nodeSpacing = rope_spacing;
    material.radius = 0.1F;
    material.mass = rope_mass;
    material.stretchCompliance = 1.0F;
    material.velocityDamping = 0.0F;
    material.maximumSubstepTimestep = contact_timestep;
    material.friction = 0.0F;
    material.maximumSpeed = 20.0F;
    material.solverIterations = 1U;
    material.selfCollision = PM_FALSE;
    return material;
}

[[nodiscard]] PmSoftBodyMaterialState soft_material() {
    PmSoftBodyMaterialState material{};
    material.nodeMass = soft_node_mass;
    material.nodeRadius = 0.05F;
    material.stretchCompliance = 1.0F;
    material.velocityDamping = 0.0F;
    material.springDamping = 0.0F;
    material.contactFriction = 0.0F;
    material.shapeMatchingStiffness = 0.0F;
    material.maximumProjectionFraction = 0.2F;
    material.constraintVelocityResponse = 0.0F;
    material.maximumSpeed = 10.0F;
    material.solverIterations = 1U;
    return material;
}

[[nodiscard]] PmCouplingState contact_coupling() {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_ROPE_SOFT_BODY;
    coupling.first = 0U;
    coupling.second = 0U;
    // The public rope/soft edge carries both supported mode bits. With both
    // endpoint flags false, this fixture executes contact without attachment.
    coupling.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_ATTACHMENT_BIT;
    coupling.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    coupling.parameters.mechanical.contactDistance = contact_distance;
    coupling.parameters.mechanical.friction = 0.0F;
    coupling.parameters.mechanical.twoWay = PM_TRUE;
    coupling.parameters.attachment.maximumAcceleration = 1000.0F;
    coupling.parameters.attachment.supportRadiusScale = 2.0F;
    coupling.parameters.attachment.contactSupportRadiusScale = 1.0F;
    coupling.parameters.attachment.attachFirst = PM_FALSE;
    coupling.parameters.attachment.attachLast = PM_FALSE;
    return coupling;
}

[[nodiscard]] SnapshotRanges make_ranges(PmBuffer buffer,
                                         PmDeviceSize offset) {
    constexpr PmDeviceSize rope_bytes = sizeof(PmDomainOutputHeader) +
        rope_capacity * sizeof(PmRopeNodeState);
    constexpr PmDeviceSize soft_bytes = sizeof(PmDomainOutputHeader) +
        soft_nodes.size() * sizeof(PmSoftBodyNodeState);
    constexpr PmDeviceSize surface_bytes = sizeof(PmDomainOutputHeader) +
        surface_vertices.size() * sizeof(PmFloat3);
    return {{buffer, offset, rope_bytes},
            {buffer, offset + rope_bytes, soft_bytes},
            {buffer, offset + rope_bytes + soft_bytes, surface_bytes}};
}

[[nodiscard]] FixtureObjects create_fixture(Context &context) {
    constexpr PmRopeCapacity rope_capacity_info{rope_capacity};
    constexpr PmSoftBodyCapacity soft_capacity{
        static_cast<std::uint32_t>(soft_nodes.size()),
        static_cast<std::uint32_t>(soft_bonds.size()),
        static_cast<std::uint32_t>(surface_vertices.size()),
        static_cast<std::uint32_t>(surface_indices.size())};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.ropeCount = 1U;
    simulation_info.pRopeCapacities = &rope_capacity_info;
    simulation_info.softBodyCount = 1U;
    simulation_info.pSoftBodyCapacities = &soft_capacity;
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

    const PmRopeMaterialState rope = rope_material();
    const PmSoftBodyMaterialState soft = soft_material();
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.ropeMaterialCount = 1U;
    pipeline_info.pRopeMaterials = &rope;
    pipeline_info.softBodyMaterialCount = 1U;
    pipeline_info.pSoftBodyMaterials = &soft;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize snapshot_bytes =
        3U * sizeof(PmDomainOutputHeader) +
        rope_capacity * sizeof(PmRopeNodeState) +
        soft_nodes.size() * sizeof(PmSoftBodyNodeState) +
        surface_vertices.size() * sizeof(PmFloat3);
    fixture.output = context.buffer(
        3U * snapshot_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.initial = make_ranges(fixture.output.handle, 0U);
    fixture.contacted = make_ranges(fixture.output.handle, snapshot_bytes);
    fixture.released = make_ranges(fixture.output.handle, 2U * snapshot_bytes);
    return fixture;
}

[[nodiscard]] PmSoftBodyInfo soft_info(
    const std::array<PmSoftBodySurfaceBinding, 10U> &bindings) {
    return {PM_TRUE, 0U, static_cast<std::uint32_t>(soft_nodes.size()),
            soft_nodes.data(), static_cast<std::uint32_t>(soft_bonds.size()),
            soft_bonds.data(), soft_inverse_masses.data(),
            static_cast<std::uint32_t>(surface_vertices.size()),
            surface_vertices.data(),
            static_cast<std::uint32_t>(surface_indices.size()),
            surface_indices.data(), bindings.data()};
}

[[nodiscard]] PmRopeInfo rope_info() {
    PmRopeInfo rope{};
    rope.active = PM_TRUE;
    rope.materialIndex = 0U;
    rope.pointCount = static_cast<std::uint32_t>(rope_centerline.size());
    rope.pCenterline = rope_centerline.data();
    return rope;
}

void record_copies(Context &context, const FixtureObjects &fixture,
                   const SnapshotRanges &ranges) {
    PM_CHECK(pmCmdCopyRopeState(context.commands, fixture.simulation, 0U,
                               &ranges.rope) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodyState(context.commands, fixture.simulation, 0U,
                                   &ranges.soft) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodySurface(context.commands, fixture.simulation, 0U,
                                     &ranges.surface) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture,
                                     const SnapshotRanges &ranges) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *rope_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + ranges.rope.offset);
    const auto *soft_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + ranges.soft.offset);
    const auto *surface_header =
        reinterpret_cast<const PmDomainOutputHeader *>(
            base + ranges.surface.offset);
    if (rope_header->activeCount < 2U ||
        rope_header->activeCount > rope_capacity ||
        soft_header->activeCount != soft_nodes.size() ||
        surface_header->activeCount != surface_vertices.size())
        throw std::runtime_error("soft-rope fixture output count mismatch");
    Snapshot result{};
    const auto *rope = reinterpret_cast<const PmRopeNodeState *>(
        rope_header + 1);
    result.rope.assign(rope, rope + rope_header->activeCount);
    const auto *soft = reinterpret_cast<const PmSoftBodyNodeState *>(
        soft_header + 1);
    const auto *surface = reinterpret_cast<const PmFloat3 *>(
        surface_header + 1);
    std::copy_n(soft, result.soft.size(), result.soft.begin());
    std::copy_n(surface, result.surface.size(), result.surface.begin());
    return result;
}

[[nodiscard]] Json rope_state_json(const PmRopeNodeState &state,
                                   std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(state.position);
    result["velocity_m_s"] = vector_json(state.velocity);
    result["soft_body_contact_force_n"] =
        vector_json(state.softBodyContactForce);
    return result;
}

[[nodiscard]] Json soft_state_json(const PmSoftBodyNodeState &state,
                                   std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(state.position);
    result["velocity_m_s"] = vector_json(state.velocity);
    result["rope_contact_force_n"] = vector_json(state.ropeContactForce);
    return result;
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot) {
    Json rope = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.rope.size(); ++index)
        rope.push_back(rope_state_json(snapshot.rope[index], index));
    Json soft = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.soft.size(); ++index)
        soft.push_back(soft_state_json(snapshot.soft[index], index));
    Json result = Json::object();
    result["rope"] = std::move(rope);
    result["soft_nodes"] = std::move(soft);
    result["surface_m"] = vectors_json(snapshot.surface);
    return result;
}

[[nodiscard]] Json settings_json(
    const std::array<PmSoftBodySurfaceBinding, 10U> &bindings) {
    const auto rope = rope_material();
    const auto soft = soft_material();
    Json bonds = Json::array();
    for (const auto &bond : soft_bonds) {
        Json item = Json::object();
        item["first"] = static_cast<std::uint64_t>(bond.first);
        item["second"] = static_cast<std::uint64_t>(bond.second);
        item["rest_length_m"] = static_cast<double>(bond.restLength);
        bonds.push_back(std::move(item));
    }
    Json bindings_json = Json::array();
    for (const auto &binding : bindings) {
        Json item = Json::object();
        Json nodes = Json::array();
        Json weights = Json::array();
        for (std::size_t slot = 0U; slot < 4U; ++slot) {
            nodes.push_back(static_cast<std::uint64_t>(binding.nodes[slot]));
            weights.push_back(static_cast<double>(binding.weights[slot]));
        }
        item["nodes"] = std::move(nodes);
        item["weights"] = std::move(weights);
        bindings_json.push_back(std::move(item));
    }
    Json result = Json::object();
    result["id"] = "soft-rope-bound-surface-normal-impact";
    result["preparation_timestep_seconds"] =
        static_cast<double>(preparation_timestep);
    result["preparation_gravity_m_s2"] = vector_json(preparation_gravity);
    result["timestep_seconds"] = static_cast<double>(contact_timestep);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["rope_centerline_m"] = vectors_json(rope_centerline);
    result["rope_node_spacing_m"] = static_cast<double>(rope.nodeSpacing);
    result["rope_radius_m"] = static_cast<double>(rope.radius);
    result["rope_total_mass_kg"] = static_cast<double>(rope.mass);
    result["rope_stretch_compliance_m_per_n"] =
        static_cast<double>(rope.stretchCompliance);
    result["rope_velocity_damping"] =
        static_cast<double>(rope.velocityDamping);
    result["rope_maximum_substep_timestep_seconds"] =
        static_cast<double>(rope.maximumSubstepTimestep);
    result["rope_friction"] = static_cast<double>(rope.friction);
    result["rope_maximum_speed_m_s"] =
        static_cast<double>(rope.maximumSpeed);
    result["rope_solver_iterations"] =
        static_cast<std::uint64_t>(rope.solverIterations);
    result["rope_self_collision"] = false;
    result["expected_rope_node_count"] = static_cast<std::uint64_t>(3U);
    result["soft_node_mass_kg"] = static_cast<double>(soft.nodeMass);
    result["soft_node_radius_m"] = static_cast<double>(soft.nodeRadius);
    result["soft_stretch_compliance_m_per_n"] =
        static_cast<double>(soft.stretchCompliance);
    result["soft_velocity_damping"] =
        static_cast<double>(soft.velocityDamping);
    result["soft_spring_damping"] = static_cast<double>(soft.springDamping);
    result["soft_contact_friction"] =
        static_cast<double>(soft.contactFriction);
    result["soft_shape_matching_stiffness"] =
        static_cast<double>(soft.shapeMatchingStiffness);
    result["soft_maximum_projection_fraction"] =
        static_cast<double>(soft.maximumProjectionFraction);
    result["soft_constraint_velocity_response"] =
        static_cast<double>(soft.constraintVelocityResponse);
    result["soft_maximum_speed_m_s"] =
        static_cast<double>(soft.maximumSpeed);
    result["soft_solver_iterations"] =
        static_cast<std::uint64_t>(soft.solverIterations);
    result["contact_distance_m"] = static_cast<double>(contact_distance);
    Json coupling_modes = Json::array();
    coupling_modes.push_back("CONTACT");
    coupling_modes.push_back("ATTACHMENT");
    result["coupling_mode_flags"] = std::move(coupling_modes);
    result["friction"] = 0.0;
    result["two_way"] = true;
    result["attach_first"] = false;
    result["attach_last"] = false;
    result["maximum_soft_body_acceleration_m_s2"] = 1000.0;
    result["anchor_support_radius_scale"] = 2.0;
    result["anchor_contact_support_radius_scale"] = 1.0;
    result["authored_soft_nodes_m"] = vectors_json(soft_nodes);
    result["prepared_soft_nodes_m"] = vectors_json(prepared_soft_nodes);
    result["prepared_soft_velocity_m_s"] =
        vector_json(PmFloat3{1.0F, -1.0F, 0.0F});
    result["soft_inverse_masses_kg_inverse"] =
        floats_json(soft_inverse_masses);
    result["soft_bonds"] = std::move(bonds);
    result["surface_vertices_m"] = vectors_json(surface_vertices);
    result["surface_triangle_indices"] = indices_json(surface_indices);
    result["surface_bindings"] = std::move(bindings_json);
    result["oracle"] =
        "public-state-rope-surface-clearance-reciprocal-contact-frictionless-slip-and-release";
    return result;
}

struct QualityRun {
    Json driver{};
    Json fixture{};
};

[[nodiscard]] QualityRun run_quality_fixture() {
    Context context;
    if (!context.open(required_features()) || !supports(context.physical))
        throw std::runtime_error("unsupported: soft-rope contact tuple");
    FixtureObjects fixture = create_fixture(context);
    const auto bindings = surface_bindings();
    const PmSoftBodyInfo soft = soft_info(bindings);
    const PmStepInfo preparation{
        preparation_timestep, 1U,
        {preparation_gravity.x, preparation_gravity.y,
         preparation_gravity.z}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSoftBodies(context.commands, fixture.simulation, 0U, 1U,
                                  &soft) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                &preparation) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmRopeInfo rope = rope_info();
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteRopes(context.commands, fixture.simulation, 0U, 1U,
                            &rope) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmCouplingState coupling = contact_coupling();
    const PmStepInfo step{contact_timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    record_copies(context, fixture, fixture.initial);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &coupling) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                &step) == PM_SUCCESS);
    record_copies(context, fixture, fixture.contacted);
    PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("soft-rope contact submission failed: " +
                                 std::to_string(submitted));

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 0U,
                              nullptr) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                &step) == PM_SUCCESS);
    record_copies(context, fixture, fixture.released);
    submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("soft-rope release submission failed: " +
                                 std::to_string(submitted));

    const Snapshot initial = read_snapshot(context, fixture, fixture.initial);
    const Snapshot contacted = read_snapshot(
        context, fixture, fixture.contacted);
    const Snapshot released = read_snapshot(context, fixture, fixture.released);
    Json result = settings_json(bindings);
    result["initial"] = snapshot_json(initial);
    result["contact"] = snapshot_json(contacted);
    result["released"] = snapshot_json(released);
    Json driver = driver_json(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(result)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}

} // namespace

std::string conformance_soft_rope_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "soft-rope-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/soft-rope-contact-v1.json");
    try {
        QualityRun run = run_quality_fixture();
        result["execution_status"] = "completed";
        result["driver"] = std::move(run.driver);
        result["fixture"] = std::move(run.fixture);
        if (status != nullptr) *status = ExecutionStatus::completed;
    } catch (const std::exception &error) {
        const bool unsupported = unsupported_error(error);
        result["execution_status"] = unsupported ? "unsupported" : "failed";
        result["reason"] = error.what();
        result["driver"] = unavailable_driver();
        result["fixture"] = Json::object();
        if (status != nullptr) *status = unsupported ?
            ExecutionStatus::unsupported : ExecutionStatus::failed;
    }
    return result.serialize();
}

} // namespace parallel_mater::conformance
