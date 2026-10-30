// SPDX-License-Identifier: MIT
// Public-SDK cloth/soft-body contact quality and graph lifecycle fixtures.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace parallel_mater::conformance {
namespace {
using parallel_mater::test::Buffer;
using parallel_mater::test::Context;

constexpr float preparation_timestep = 1.0F / 60.0F;
constexpr PmFloat3 preparation_gravity{60.0F, -60.0F, 0.0F};
constexpr float quality_timestep = 1.0F / 60.0F;
constexpr float cloth_vertex_mass = 1.0F;
constexpr float soft_node_mass = 2.0F;
constexpr float contact_distance = 0.1F;
constexpr std::array<PmFloat3, 3U> cloth_vertices{{
    {-0.8F, 0.0F, -0.8F},
    {0.0F, 0.0F, 0.8F},
    {0.8F, 0.0F, -0.8F}}};
constexpr std::array<std::uint32_t, 3U> cloth_indices{{0U, 1U, 2U}};
constexpr std::array<float, 3U> cloth_inverse_masses{{1.0F, 1.0F, 1.0F}};
constexpr std::array<float, 3U> pinned_cloth{{0.0F, 0.0F, 0.0F}};

constexpr std::array<PmFloat3, 4U> soft_nodes{{
    {-0.01666667F, 0.02666667F, 0.0F},
    {0.48333333F, 0.50666667F, -0.5F},
    {-0.51666667F, 0.50666667F, -0.5F},
    {-0.01666667F, 0.50666667F, 0.5F}}};
constexpr std::array<PmFloat3, 4U> prepared_soft_nodes{{
    {0.0F, 0.01F, 0.0F},
    {0.5F, 0.49F, -0.5F},
    {-0.5F, 0.49F, -0.5F},
    {0.0F, 0.49F, 0.5F}}};
constexpr std::array<PmFloat3, 4U> prepared_soft_velocities{{
    {1.0F, -1.0F, 0.0F},
    {1.0F, -1.0F, 0.0F},
    {1.0F, -1.0F, 0.0F},
    {1.0F, -1.0F, 0.0F}}};
constexpr std::array<float, 4U> soft_inverse_masses{{
    0.5F, 0.5F, 0.5F, 0.5F}};
constexpr std::array<float, 4U> lifecycle_soft_inverse_masses{{
    0.0F, 0.0F, 0.0F, 0.5F}};
constexpr std::array<PmSoftBodyBond, 6U> soft_bonds{{
    {0U, 1U, 0.85463442476886F},
    {0U, 2U, 0.85463442476886F},
    {0U, 3U, 0.693108938046538F},
    {1U, 2U, 1.0F},
    {1U, 3U, 1.118033988749895F},
    {2U, 3U, 1.118033988749895F}}};
constexpr std::array<PmFloat3, 10U> surface_vertices{{
    soft_nodes[0], soft_nodes[1], soft_nodes[2], soft_nodes[3],
    {0.23333333F, 0.26666667F, -0.25F},
    {-0.26666667F, 0.26666667F, -0.25F},
    {-0.01666667F, 0.26666667F, 0.25F},
    {-0.01666667F, 0.50666667F, -0.5F},
    {0.23333333F, 0.50666667F, 0.0F},
    {-0.26666667F, 0.50666667F, 0.0F}}};
constexpr std::array<std::uint32_t, 48U> surface_indices{{
    0U, 5U, 4U, 5U, 2U, 7U, 4U, 7U, 1U, 5U, 7U, 4U,
    0U, 4U, 6U, 4U, 1U, 8U, 6U, 8U, 3U, 4U, 8U, 6U,
    0U, 6U, 5U, 6U, 3U, 9U, 5U, 9U, 2U, 6U, 9U, 5U,
    1U, 7U, 8U, 7U, 2U, 9U, 8U, 9U, 3U, 7U, 9U, 8U}};
constexpr std::array<float, 3U> expected_cloth_weights{{
    0.25F, 0.5F, 0.25F}};

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
    std::array<PmClothVertexState, 3U> cloth{};
    std::array<PmSoftBodyNodeState, 4U> soft{};
    std::array<PmFloat3, 10U> surface{};
};

struct SnapshotRanges {
    PmBufferRange cloth{};
    PmBufferRange soft{};
    PmBufferRange surface{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    SnapshotRanges initial{};
    SnapshotRanges final{};
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
        throw std::runtime_error("query cloth-soft contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown cloth-soft contact backend");
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
    needed.softBodies = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

[[nodiscard]] bool supports(PmPhysicalDevice physical) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_CLOTH,
        PM_INTERACTION_FAMILY_SOFT_BODY,
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

[[nodiscard]] PmClothMaterialState cloth_material() {
    PmClothMaterialState material{};
    material.vertexMass = cloth_vertex_mass;
    material.thickness = 0.01F;
    material.stretchCompliance = 1.0e-3F;
    material.bendingCompliance = 1.0e-3F;
    material.velocityDamping = 0.0F;
    material.contactFriction = 0.0F;
    material.solverIterations = 1U;
    material.fracturePersistenceSubsteps = 1U;
    material.preserveVolume = PM_FALSE;
    material.volumeCompliance = 1.0e-7F;
    return material;
}

[[nodiscard]] PmSoftBodyMaterialState soft_material() {
    PmSoftBodyMaterialState material{};
    material.nodeMass = soft_node_mass;
    material.nodeRadius = 0.05F;
    material.stretchCompliance = 1.0e-3F;
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

[[nodiscard]] PmCouplingState contact_coupling(float distance,
                                                float friction = 0.0F) {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_SOFT_BODY_CLOTH;
    coupling.first = 0U;
    coupling.second = 0U;
    coupling.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    coupling.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    coupling.parameters.mechanical.contactDistance = distance;
    coupling.parameters.mechanical.friction = friction;
    coupling.parameters.mechanical.twoWay = PM_TRUE;
    return coupling;
}

[[nodiscard]] SnapshotRanges make_ranges(PmBuffer buffer,
                                         PmDeviceSize offset) {
    constexpr PmDeviceSize cloth_bytes = sizeof(PmDomainOutputHeader) +
        cloth_vertices.size() * sizeof(PmClothVertexState);
    constexpr PmDeviceSize soft_bytes = sizeof(PmDomainOutputHeader) +
        soft_nodes.size() * sizeof(PmSoftBodyNodeState);
    constexpr PmDeviceSize surface_bytes = sizeof(PmDomainOutputHeader) +
        surface_vertices.size() * sizeof(PmFloat3);
    return {{buffer, offset, cloth_bytes},
            {buffer, offset + cloth_bytes, soft_bytes},
            {buffer, offset + cloth_bytes + soft_bytes, surface_bytes}};
}

[[nodiscard]] FixtureObjects create_fixture(Context &context,
                                            std::uint32_t coupling_capacity) {
    constexpr PmClothCapacity cloth_capacity{
        static_cast<std::uint32_t>(cloth_vertices.size()),
        static_cast<std::uint32_t>(cloth_indices.size()),
        static_cast<std::uint32_t>(cloth_indices.size())};
    constexpr PmSoftBodyCapacity soft_capacity{
        static_cast<std::uint32_t>(soft_nodes.size()),
        static_cast<std::uint32_t>(soft_bonds.size()),
        static_cast<std::uint32_t>(surface_vertices.size()),
        static_cast<std::uint32_t>(surface_indices.size())};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.clothCount = 1U;
    simulation_info.pClothCapacities = &cloth_capacity;
    simulation_info.softBodyCount = 1U;
    simulation_info.pSoftBodyCapacities = &soft_capacity;
    simulation_info.couplingCapacity = coupling_capacity;
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

    const PmClothMaterialState cloth = cloth_material();
    const PmSoftBodyMaterialState soft = soft_material();
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &cloth;
    pipeline_info.softBodyMaterialCount = 1U;
    pipeline_info.pSoftBodyMaterials = &soft;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize snapshot_bytes =
        3U * sizeof(PmDomainOutputHeader) +
        cloth_vertices.size() * sizeof(PmClothVertexState) +
        soft_nodes.size() * sizeof(PmSoftBodyNodeState) +
        surface_vertices.size() * sizeof(PmFloat3);
    fixture.output = context.buffer(
        2U * snapshot_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.initial = make_ranges(fixture.output.handle, 0U);
    fixture.final = make_ranges(fixture.output.handle, snapshot_bytes);
    return fixture;
}

[[nodiscard]] PmClothInfo cloth_info(const float *inverse_masses) {
    return {PM_TRUE, 0U, static_cast<std::uint32_t>(cloth_vertices.size()),
            cloth_vertices.data(),
            static_cast<std::uint32_t>(cloth_indices.size()),
            cloth_indices.data(), inverse_masses};
}

[[nodiscard]] PmSoftBodyInfo soft_info(
    const float *inverse_masses,
    const std::array<PmSoftBodySurfaceBinding, 10U> &bindings) {
    return {PM_TRUE, 0U, static_cast<std::uint32_t>(soft_nodes.size()),
            soft_nodes.data(), static_cast<std::uint32_t>(soft_bonds.size()),
            soft_bonds.data(), inverse_masses,
            static_cast<std::uint32_t>(surface_vertices.size()),
            surface_vertices.data(),
            static_cast<std::uint32_t>(surface_indices.size()),
            surface_indices.data(), bindings.data()};
}

void record_copies(Context &context, const FixtureObjects &fixture,
                   const SnapshotRanges &ranges) {
    PM_CHECK(pmCmdCopyClothState(context.commands, fixture.simulation, 0U,
                                &ranges.cloth) == PM_SUCCESS);
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
    const auto *cloth_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + ranges.cloth.offset);
    const auto *soft_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + ranges.soft.offset);
    const auto *surface_header =
        reinterpret_cast<const PmDomainOutputHeader *>(
            base + ranges.surface.offset);
    if (cloth_header->activeCount != cloth_vertices.size() ||
        soft_header->activeCount != soft_nodes.size() ||
        surface_header->activeCount != surface_vertices.size())
        throw std::runtime_error("cloth-soft fixture output count mismatch");
    Snapshot result{};
    const auto *cloth =
        reinterpret_cast<const PmClothVertexState *>(cloth_header + 1);
    const auto *soft =
        reinterpret_cast<const PmSoftBodyNodeState *>(soft_header + 1);
    const auto *surface = reinterpret_cast<const PmFloat3 *>(surface_header + 1);
    std::copy_n(cloth, result.cloth.size(), result.cloth.begin());
    std::copy_n(soft, result.soft.size(), result.soft.begin());
    std::copy_n(surface, result.surface.size(), result.surface.begin());
    return result;
}

[[nodiscard]] Json cloth_state_json(const PmClothVertexState &state,
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
    result["cloth_contact_force_n"] = vector_json(state.clothContactForce);
    return result;
}

[[nodiscard]] Json snapshot_cloth_json(const Snapshot &snapshot) {
    Json result = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.cloth.size(); ++index)
        result.push_back(cloth_state_json(snapshot.cloth[index], index));
    return result;
}

[[nodiscard]] Json snapshot_soft_json(const Snapshot &snapshot) {
    Json result = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.soft.size(); ++index)
        result.push_back(soft_state_json(snapshot.soft[index], index));
    return result;
}

[[nodiscard]] Json snapshot_surface_json(const Snapshot &snapshot) {
    return vectors_json(snapshot.surface);
}

[[nodiscard]] Json settings_json(
    const std::array<PmSoftBodySurfaceBinding, 10U> &bindings) {
    const auto cloth = cloth_material();
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
    result["id"] = "cloth-soft-bound-surface-normal-impact";
    result["preparation_timestep_seconds"] =
        static_cast<double>(preparation_timestep);
    result["preparation_gravity_m_s2"] = vector_json(preparation_gravity);
    result["timestep_seconds"] = static_cast<double>(quality_timestep);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["cloth_vertices_m"] = vectors_json(cloth_vertices);
    result["cloth_triangle_indices"] = indices_json(cloth_indices);
    result["cloth_inverse_masses_kg_inverse"] =
        floats_json(cloth_inverse_masses);
    result["cloth_vertex_mass_kg"] = static_cast<double>(cloth.vertexMass);
    result["cloth_thickness_m"] = static_cast<double>(cloth.thickness);
    result["cloth_stretch_compliance_m_per_n"] =
        static_cast<double>(cloth.stretchCompliance);
    result["cloth_bending_compliance_m_per_n"] =
        static_cast<double>(cloth.bendingCompliance);
    result["cloth_velocity_damping"] =
        static_cast<double>(cloth.velocityDamping);
    result["cloth_contact_friction"] =
        static_cast<double>(cloth.contactFriction);
    result["cloth_solver_iterations"] =
        static_cast<std::uint64_t>(cloth.solverIterations);
    result["cloth_preserve_volume"] = false;
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
    result["coupling_solver_iterations"] = static_cast<std::uint64_t>(4U);
    result["contact_distance_m"] = static_cast<double>(contact_distance);
    result["friction"] = 0.0;
    result["two_way"] = true;
    result["authored_soft_nodes_m"] = vectors_json(soft_nodes);
    result["prepared_soft_nodes_m"] = vectors_json(prepared_soft_nodes);
    result["prepared_soft_velocities_m_s"] =
        vectors_json(prepared_soft_velocities);
    result["soft_inverse_masses_kg_inverse"] =
        floats_json(soft_inverse_masses);
    result["soft_bonds"] = std::move(bonds);
    result["surface_vertices_m"] = vectors_json(surface_vertices);
    result["surface_triangle_indices"] = indices_json(surface_indices);
    result["surface_bindings"] = std::move(bindings_json);
    result["minimum_binding_incidence_per_node"] =
        static_cast<std::uint64_t>(4U);
    result["contact_soft_node_index"] = static_cast<std::uint64_t>(0U);
    result["expected_cloth_contact_weights"] =
        floats_json(expected_cloth_weights);
    result["oracle"] =
        "public-state-triangle-clearance-reciprocal-impulse-frictionless-slip-and-unique-bound-node-mass";
    return result;
}

struct QualityRun {
    Json driver{};
    Json fixture{};
};

[[nodiscard]] QualityRun run_quality_fixture() {
    Context context;
    if (!context.open(required_features()) || !supports(context.physical))
        throw std::runtime_error("unsupported: cloth-soft contact tuple");
    FixtureObjects fixture = create_fixture(context, 1U);
    const auto bindings = surface_bindings();
    const PmClothInfo initial_cloth = cloth_info(pinned_cloth.data());
    const PmSoftBodyInfo initial_soft =
        soft_info(soft_inverse_masses.data(), bindings);
    const PmStepInfo preparation{
        preparation_timestep, 1U,
        {preparation_gravity.x, preparation_gravity.y,
         preparation_gravity.z}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, fixture.simulation, 0U, 1U,
                             &initial_cloth) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSoftBodies(context.commands, fixture.simulation, 0U, 1U,
                                  &initial_soft) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                &preparation) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmClothInfo dynamic_cloth = cloth_info(cloth_inverse_masses.data());
    const PmCouplingState coupling = contact_coupling(contact_distance);
    const PmStepInfo contact{quality_timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, fixture.simulation, 0U, 1U,
                             &dynamic_cloth) == PM_SUCCESS);
    record_copies(context, fixture, fixture.initial);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &coupling) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                &contact) == PM_SUCCESS);
    record_copies(context, fixture, fixture.final);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const Snapshot initial = read_snapshot(context, fixture, fixture.initial);
    const Snapshot final = read_snapshot(context, fixture, fixture.final);
    Json result = settings_json(bindings);
    result["initial_cloth_state"] = snapshot_cloth_json(initial);
    result["final_cloth_state"] = snapshot_cloth_json(final);
    result["initial_soft_node_state"] = snapshot_soft_json(initial);
    result["final_soft_node_state"] = snapshot_soft_json(final);
    result["initial_surface_state_m"] = snapshot_surface_json(initial);
    result["final_surface_state_m"] = snapshot_surface_json(final);
    Json driver = driver_json(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(result)};
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

void append_vector(std::ostringstream &stream, PmFloat3 value) {
    append_float(stream, value.x);
    stream << ',';
    append_float(stream, value.y);
    stream << ',';
    append_float(stream, value.z);
}

[[nodiscard]] std::string graph_fingerprint(
    const PmCouplingState *coupling) {
    if (coupling == nullptr)
        return sha256("parallel-mater-cloth-soft-contact-graph/v1;inactive");
    std::ostringstream canonical;
    canonical << "parallel-mater-cloth-soft-contact-graph/v1;type="
              << static_cast<unsigned>(coupling->type)
              << ";first=" << coupling->first
              << ";second=" << coupling->second
              << ";modes=" << coupling->modeFlags
              << ";direction=BIDIRECTIONAL"
              << ";material="
              << static_cast<unsigned>(coupling->materialProfile)
              << ";contact=";
    append_float(canonical,
                 coupling->parameters.mechanical.contactDistance);
    canonical << ";friction=";
    append_float(canonical, coupling->parameters.mechanical.friction);
    canonical << ";two-way=" << coupling->parameters.mechanical.twoWay;
    return sha256(canonical.str());
}

[[nodiscard]] std::string endpoint_fingerprint(const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-cloth-soft-contact-endpoints/v1";
    for (std::size_t index = 0U; index < snapshot.cloth.size(); ++index) {
        canonical << ";cloth=" << index << ",position=";
        append_vector(canonical, snapshot.cloth[index].position);
        canonical << ",velocity=";
        append_vector(canonical, snapshot.cloth[index].velocity);
    }
    for (std::size_t index = 0U; index < snapshot.soft.size(); ++index) {
        canonical << ";soft=" << index << ",position=";
        append_vector(canonical, snapshot.soft[index].position);
        canonical << ",velocity=";
        append_vector(canonical, snapshot.soft[index].velocity);
    }
    for (std::size_t index = 0U; index < snapshot.surface.size(); ++index) {
        canonical << ";surface=" << index << ',';
        append_vector(canonical, snapshot.surface[index]);
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
                    const PmCouplingState *coupling,
                    std::uint64_t graph_generation,
                    const Snapshot &endpoints) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        "cloth-soft-contact-graph", coupling != nullptr, graph_generation,
        graph_fingerprint(coupling)));
    resources.push_back(resource_json(
        "cloth-soft-endpoints", true, 1U,
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
    if (!context.open(required_features()) || !supports(context.physical))
        throw std::runtime_error("unsupported: cloth-soft contact tuple");
    FixtureObjects fixture = create_fixture(context, 1U);
    const auto bindings = surface_bindings();
    const PmClothInfo cloth = cloth_info(pinned_cloth.data());
    const PmSoftBodyInfo soft = soft_info(
        lifecycle_soft_inverse_masses.data(), bindings);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, fixture.simulation, 0U, 1U,
                             &cloth) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSoftBodies(context.commands, fixture.simulation, 0U, 1U,
                                  &soft) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmCouplingState baseline = contact_coupling(0.08F, 0.1F);
    const PmCouplingState replacement = contact_coupling(0.09F, 0.2F);
    const PmCouplingState recovery = contact_coupling(0.07F, 0.3F);
    const std::array<PmCouplingState, 2U> overflow{{replacement, recovery}};
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    Json observations = Json::array();

    const auto mutate = [&](const PmCouplingState *couplings,
                            std::uint32_t count, PmResult expected) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, count,
                                  couplings) == expected);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        record_copies(context, fixture, fixture.final);
        PM_CHECK(context.submit() == PM_SUCCESS);
        return read_snapshot(context, fixture, fixture.final);
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

std::string conformance_cloth_soft_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "cloth-soft-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/cloth-soft-contact-v1.json");
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

std::string conformance_cloth_soft_contact_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "cloth-soft-contact-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/cloth-soft-contact-lifecycle-v1.json");
    try {
        LifecycleRun run = run_lifecycle_fixture();
        Json families = Json::array();
        families.push_back("cloth");
        families.push_back("soft_body");
        Json unrelated = Json::array();
        unrelated.push_back("cloth-soft-endpoints");
        Json attempt = Json::object();
        attempt["fixture"] = "cloth-soft-contact-public-lifecycle";
        attempt["status"] = "completed";
        Json item = Json::object();
        item["attempt"] = std::move(attempt);
        item["execution_status"] = "completed";
        item["families"] = std::move(families);
        item["id"] =
            "cloth-soft-contact-capacity-replacement-removal-rollback";
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
