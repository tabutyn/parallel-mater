// SPDX-License-Identifier: MIT
// Public-SDK intrinsic soft-body/rigid contact quality and lifecycle fixtures.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

constexpr float timestep = 0.01F;
constexpr float preparation_gravity = 100.0F;
constexpr float node_mass = 2.0F;
constexpr float node_radius = 0.05F;
constexpr float rigid_mass = 2.0F;
constexpr float cube_half_extent = 0.1F;
constexpr float rigid_collision_margin = 0.005F;
constexpr float cube_inertia =
    (1.0F / 6.0F) * rigid_mass * (2.0F * cube_half_extent) *
    (2.0F * cube_half_extent);
constexpr float separated_translation = -0.145F;

constexpr std::array<PmFloat3, 8U> cube_vertices{{
    {-cube_half_extent, -cube_half_extent, -cube_half_extent},
    { cube_half_extent, -cube_half_extent, -cube_half_extent},
    { cube_half_extent,  cube_half_extent, -cube_half_extent},
    {-cube_half_extent,  cube_half_extent, -cube_half_extent},
    {-cube_half_extent, -cube_half_extent,  cube_half_extent},
    { cube_half_extent, -cube_half_extent,  cube_half_extent},
    { cube_half_extent,  cube_half_extent,  cube_half_extent},
    {-cube_half_extent,  cube_half_extent,  cube_half_extent},
}};

constexpr std::array<std::uint32_t, 36U> cube_indices{{
    0U, 3U, 2U, 0U, 2U, 1U, 4U, 5U, 6U, 4U, 6U, 7U,
    0U, 4U, 7U, 0U, 7U, 3U, 1U, 2U, 6U, 1U, 6U, 5U,
    0U, 1U, 5U, 0U, 5U, 4U, 3U, 7U, 6U, 3U, 6U, 2U,
}};

// Preparation advances every node +0.01 m in X and gives it +1 m/s before
// the rigid endpoint exists. The prepared apex is therefore at -0.155 m.
constexpr std::array<PmFloat3, 4U> authored_nodes{{
    {-0.165F, 0.0F, 0.0F},
    {-0.665F, 0.5F, -0.5F},
    {-0.665F, -0.5F, -0.5F},
    {-0.665F, 0.0F, 0.5F},
}};
constexpr std::array<PmFloat3, 4U> prepared_nodes{{
    {-0.155F, 0.0F, 0.0F},
    {-0.655F, 0.5F, -0.5F},
    {-0.655F, -0.5F, -0.5F},
    {-0.655F, 0.0F, 0.5F},
}};
constexpr std::array<float, 4U> inverse_masses{{0.5F, 0.5F, 0.5F, 0.5F}};
constexpr std::array<PmSoftBodyBond, 6U> bonds{{
    {0U, 1U, 0.866025404F}, {0U, 2U, 0.866025404F},
    {0U, 3U, 0.707106781F}, {1U, 2U, 1.0F},
    {1U, 3U, 1.118033989F}, {2U, 3U, 1.118033989F},
}};
constexpr std::array<std::uint32_t, 48U> surface_indices{{
    0U, 5U, 4U, 5U, 2U, 7U, 4U, 7U, 1U, 5U, 7U, 4U,
    0U, 4U, 6U, 4U, 1U, 8U, 6U, 8U, 3U, 4U, 8U, 6U,
    0U, 6U, 5U, 6U, 3U, 9U, 5U, 9U, 2U, 6U, 9U, 5U,
    1U, 7U, 8U, 7U, 2U, 9U, 8U, 9U, 3U, 7U, 9U, 8U,
}};

[[nodiscard]] std::array<PmFloat3, 10U> surface_vertices(
    float translation) {
    std::array<PmFloat3, 10U> result{};
    for (std::size_t index = 0U; index < authored_nodes.size(); ++index) {
        result[index] = authored_nodes[index];
        result[index].x += translation;
    }
    constexpr std::array<std::array<std::uint32_t, 2U>, 6U> edges{{
        {{0U, 1U}}, {{0U, 2U}}, {{0U, 3U}},
        {{1U, 2U}}, {{1U, 3U}}, {{2U, 3U}},
    }};
    for (std::size_t index = 0U; index < edges.size(); ++index) {
        const PmFloat3 first = result[edges[index][0]];
        const PmFloat3 second = result[edges[index][1]];
        result[index + 4U] = {
            0.5F * (first.x + second.x),
            0.5F * (first.y + second.y),
            0.5F * (first.z + second.z)};
    }
    return result;
}

[[nodiscard]] std::array<PmSoftBodySurfaceBinding, 10U> surface_bindings() {
    std::array<PmSoftBodySurfaceBinding, 10U> result{};
    for (std::uint32_t index = 0U; index < 4U; ++index) {
        result[index].nodes[0] = index;
        result[index].weights[0] = 1.0F;
    }
    constexpr std::array<std::array<std::uint32_t, 2U>, 6U> edges{{
        {{0U, 1U}}, {{0U, 2U}}, {{0U, 3U}},
        {{1U, 2U}}, {{1U, 3U}}, {{2U, 3U}},
    }};
    for (std::uint32_t index = 0U; index < edges.size(); ++index) {
        auto &binding = result[index + 4U];
        binding.nodes[0] = edges[index][0];
        binding.nodes[1] = edges[index][1];
        binding.weights[0] = 0.5F;
        binding.weights[1] = 0.5F;
    }
    return result;
}

struct Snapshot {
    std::array<PmSoftBodyNodeState, 4U> nodes{};
    std::array<PmFloat3, 10U> surface{};
    PmRigidState rigid{};
};

struct FixtureObjects {
    PmGeometry geometry{};
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange nodes{};
    PmBufferRange surface{};
    PmBufferRange rigid{};
};

[[nodiscard]] const char *native_driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult status = pmGetNativeDevice(device, &native);
    if (status == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (status != PM_SUCCESS)
        throw std::runtime_error("query soft-rigid contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown soft-rigid contact backend");
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
    needed.rigidBodies = PM_TRUE;
    needed.softBodies = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

void require_contact_capability(const Context &context) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_SOFT_BODY,
        PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
        PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation !=
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error("unsupported: soft-rigid contact capability");
}

[[nodiscard]] PmSoftBodyMaterialState soft_material() {
    PmSoftBodyMaterialState material{};
    material.nodeMass = node_mass;
    material.nodeRadius = node_radius;
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

[[nodiscard]] PmRigidBodyInfo rigid_body(PmGeometry geometry) {
    PmRigidBodyInfo body{};
    body.active = PM_TRUE;
    body.motionType = PM_MOTION_DYNAMIC;
    body.geometry = geometry;
    body.state.orientation[3] = 1.0F;
    body.mass = rigid_mass;
    body.inertiaDiagonal[0] = cube_inertia;
    body.inertiaDiagonal[1] = cube_inertia;
    body.inertiaDiagonal[2] = cube_inertia;
    body.friction = 0.0F;
    body.restitution = 0.0F;
    body.linearDamping = 0.0F;
    body.angularDamping = 0.0F;
    body.maximumLinearSpeed = 10.0F;
    body.maximumAngularSpeed = 10.0F;
    body.collisionMargin = rigid_collision_margin;
    return body;
}

[[nodiscard]] PmGeometry create_geometry(Context &context) {
    const PmDeviceSize vertex_bytes = sizeof(cube_vertices);
    const PmDeviceSize index_bytes = sizeof(cube_indices);
    const Buffer input = context.buffer(
        vertex_bytes + index_bytes, PM_BUFFER_GEOMETRY_INPUT_BIT);
    std::memcpy(input.mapped, cube_vertices.data(), vertex_bytes);
    std::memcpy(static_cast<std::byte *>(input.mapped) + vertex_bytes,
                cube_indices.data(), index_bytes);
    context.flush(input);
    PmGeometryCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    info.vertices = {{input.handle, 0U, vertex_bytes},
                     PM_FORMAT_R32G32B32_SFLOAT,
                     static_cast<std::uint32_t>(cube_vertices.size()),
                     sizeof(PmFloat3)};
    info.indices = {{input.handle, vertex_bytes, index_bytes},
                    PM_FORMAT_R32_UINT,
                    static_cast<std::uint32_t>(cube_indices.size()),
                    sizeof(std::uint32_t)};
    PmGeometry geometry{};
    PM_CHECK(pmCreateGeometry(context.device, &info, &geometry) == PM_SUCCESS);
    PmMemoryRequirements requirements{};
    pmGetGeometryMemoryRequirements(context.device, geometry, &requirements);
    PM_CHECK(pmBindGeometryMemory(context.device, geometry,
                                  context.allocate(requirements), 0U) ==
             PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdBuildGeometry(context.commands, geometry) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    return geometry;
}

[[nodiscard]] FixtureObjects create_fixture(Context &context) {
    FixtureObjects fixture{};
    fixture.geometry = create_geometry(context);
    constexpr PmSoftBodyCapacity soft_capacity{
        4U, 6U, 10U, static_cast<std::uint32_t>(surface_indices.size())};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.rigidBodyCapacity = 1U;
    create.softBodyCount = 1U;
    create.pSoftBodyCapacities = &soft_capacity;
    PM_CHECK(pmCreateSimulation(context.device, &create,
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
    const PmRigidSolverState rigid{4U, PM_FALSE, 0.0F, 0.0F, 0.0F};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid;
    pipeline_info.softBodyMaterialCount = 1U;
    pipeline_info.pSoftBodyMaterials = &soft;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize node_bytes = sizeof(PmDomainOutputHeader) +
        authored_nodes.size() * sizeof(PmSoftBodyNodeState);
    constexpr PmDeviceSize surface_bytes = sizeof(PmDomainOutputHeader) +
        10U * sizeof(PmFloat3);
    constexpr PmDeviceSize rigid_bytes = sizeof(PmRigidState);
    fixture.output = context.buffer(
        node_bytes + surface_bytes + rigid_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.nodes = {fixture.output.handle, 0U, node_bytes};
    fixture.surface = {fixture.output.handle, node_bytes, surface_bytes};
    fixture.rigid = {
        fixture.output.handle, node_bytes + surface_bytes, rigid_bytes};
    return fixture;
}

[[nodiscard]] PmSoftBodyInfo soft_info(
    const std::array<PmFloat3, 4U> &nodes,
    const std::array<PmFloat3, 10U> &surface,
    const std::array<PmSoftBodySurfaceBinding, 10U> &bindings) {
    return {PM_TRUE, 0U, static_cast<std::uint32_t>(nodes.size()), nodes.data(),
            static_cast<std::uint32_t>(bonds.size()), bonds.data(),
            inverse_masses.data(),
            static_cast<std::uint32_t>(surface.size()), surface.data(),
            static_cast<std::uint32_t>(surface_indices.size()),
            surface_indices.data(), bindings.data()};
}

[[nodiscard]] std::array<PmFloat3, 4U> translated_nodes(float translation) {
    auto result = authored_nodes;
    for (auto &node : result) node.x += translation;
    return result;
}

void record_copies(Context &context, const FixtureObjects &fixture,
                   bool include_soft = true) {
    if (include_soft) {
        PM_CHECK(pmCmdCopySoftBodyState(context.commands, fixture.simulation,
                                       0U, &fixture.nodes) == PM_SUCCESS);
        PM_CHECK(pmCmdCopySoftBodySurface(context.commands, fixture.simulation,
                                         0U, &fixture.surface) == PM_SUCCESS);
    }
    PM_CHECK(pmCmdCopyRigidStates(context.commands, fixture.simulation, 0U, 1U,
                                 &fixture.rigid) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *node_header =
        reinterpret_cast<const PmDomainOutputHeader *>(base);
    const auto *surface_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.surface.offset);
    if (node_header->activeCount != authored_nodes.size() ||
        surface_header->activeCount != 10U)
        throw std::runtime_error("soft-rigid output count mismatch");
    Snapshot result{};
    std::copy_n(reinterpret_cast<const PmSoftBodyNodeState *>(node_header + 1),
                result.nodes.size(), result.nodes.begin());
    std::copy_n(reinterpret_cast<const PmFloat3 *>(surface_header + 1),
                result.surface.size(), result.surface.begin());
    result.rigid = *reinterpret_cast<const PmRigidState *>(
        base + fixture.rigid.offset);
    return result;
}

[[nodiscard]] Json vector_json(PmFloat3 value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    return result;
}

[[nodiscard]] Json vector_json(const float value[3]) {
    return vector_json({value[0], value[1], value[2]});
}

[[nodiscard]] Json quaternion_json(const float value[4]) {
    Json result = Json::array();
    for (std::size_t axis = 0U; axis < 4U; ++axis)
        result.push_back(static_cast<double>(value[axis]));
    return result;
}

template <std::size_t Size>
[[nodiscard]] Json vectors_json(const std::array<PmFloat3, Size> &values) {
    Json result = Json::array();
    for (const auto value : values) result.push_back(vector_json(value));
    return result;
}

[[nodiscard]] Json nodes_json(
    const std::array<PmSoftBodyNodeState, 4U> &nodes) {
    Json result = Json::array();
    for (std::size_t index = 0U; index < nodes.size(); ++index) {
        Json node = Json::object();
        node["id"] = static_cast<std::uint64_t>(index);
        node["position_m"] = vector_json(nodes[index].position);
        node["velocity_m_s"] = vector_json(nodes[index].velocity);
        node["rigid_contact_force_n"] =
            vector_json(nodes[index].rigidContactForce);
        result.push_back(std::move(node));
    }
    return result;
}

[[nodiscard]] Json rigid_json(const PmRigidState &rigid) {
    Json result = Json::object();
    result["position_m"] = vector_json(rigid.position);
    result["orientation_xyzw"] = quaternion_json(rigid.orientation);
    result["linear_velocity_m_s"] = vector_json(rigid.linearVelocity);
    result["angular_velocity_rad_s"] = vector_json(rigid.angularVelocity);
    return result;
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot) {
    Json result = Json::object();
    result["soft_nodes"] = nodes_json(snapshot.nodes);
    result["surface_vertices_m"] = vectors_json(snapshot.surface);
    result["rigid"] = rigid_json(snapshot.rigid);
    return result;
}

[[nodiscard]] Json settings_json(float translation) {
    const auto nodes = translated_nodes(translation);
    const auto surface = surface_vertices(translation);
    const auto bindings = surface_bindings();
    Json result = Json::object();
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["step_count"] = static_cast<std::uint64_t>(1U);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["preparation_timestep_seconds"] = static_cast<double>(timestep);
    result["preparation_gravity_m_s2"] =
        vector_json({preparation_gravity, 0.0F, 0.0F});
    result["authored_soft_nodes_m"] = vectors_json(nodes);
    auto expected = prepared_nodes;
    for (auto &node : expected) node.x += translation;
    result["prepared_soft_nodes_m"] = vectors_json(expected);
    result["prepared_soft_velocity_m_s"] = vector_json({1.0F, 0.0F, 0.0F});
    Json masses = Json::array();
    for (const float value : inverse_masses)
        masses.push_back(static_cast<double>(value));
    result["soft_inverse_masses_kg_inverse"] = std::move(masses);
    result["soft_node_mass_kg"] = static_cast<double>(node_mass);
    result["soft_node_radius_m"] = static_cast<double>(node_radius);
    result["soft_bond_count"] = static_cast<std::uint64_t>(bonds.size());
    result["surface_vertices_m"] = vectors_json(surface);
    Json indices = Json::array();
    for (const auto index : surface_indices)
        indices.push_back(static_cast<std::uint64_t>(index));
    result["surface_triangle_indices"] = std::move(indices);
    Json binding_json = Json::array();
    for (const auto &binding : bindings) {
        Json item = Json::object();
        Json node_indices = Json::array();
        Json weights = Json::array();
        for (std::size_t slot = 0U; slot < 4U; ++slot) {
            node_indices.push_back(
                static_cast<std::uint64_t>(binding.nodes[slot]));
            weights.push_back(static_cast<double>(binding.weights[slot]));
        }
        item["nodes"] = std::move(node_indices);
        item["weights"] = std::move(weights);
        binding_json.push_back(std::move(item));
    }
    result["surface_bindings"] = std::move(binding_json);
    result["impact_node_index"] = static_cast<std::uint64_t>(0U);
    result["minimum_surface_binding_incidence_per_node"] =
        static_cast<std::uint64_t>(4U);
    result["rigid_geometry_vertices_m"] = vectors_json(cube_vertices);
    Json rigid_indices = Json::array();
    for (const auto index : cube_indices)
        rigid_indices.push_back(static_cast<std::uint64_t>(index));
    result["rigid_geometry_triangle_indices"] = std::move(rigid_indices);
    Json impact_face = Json::array();
    for (const std::uint64_t index : {0U, 4U, 7U, 3U})
        impact_face.push_back(index);
    result["impact_face_vertex_indices"] = std::move(impact_face);
    result["rigid_initial_position_m"] = vector_json(PmFloat3{});
    const float orientation[4]{0.0F, 0.0F, 0.0F, 1.0F};
    result["rigid_initial_orientation_xyzw"] = quaternion_json(orientation);
    result["rigid_mass_kg"] = static_cast<double>(rigid_mass);
    result["rigid_inertia_diagonal_kg_m2"] =
        vector_json({cube_inertia, cube_inertia, cube_inertia});
    result["rigid_collision_margin_m"] =
        static_cast<double>(rigid_collision_margin);
    result["rigid_friction"] = 0.0;
    result["rigid_restitution"] = 0.0;
    result["rigid_linear_damping_per_s"] = 0.0;
    result["rigid_angular_damping_per_s"] = 0.0;
    result["rigid_maximum_linear_speed_m_s"] = 10.0;
    result["rigid_maximum_angular_speed_rad_s"] = 10.0;
    result["oracle"] =
        "public-bound-surface-clearance-unique-node-mass-reciprocal-impulse";
    return result;
}

struct QualityRun {
    Json driver{};
    Json fixture{};
};

[[nodiscard]] QualityRun run_quality_fixture(float translation,
                                             const char *id) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: soft-rigid contact features");
    require_contact_capability(context);
    FixtureObjects fixture = create_fixture(context);
    const auto nodes = translated_nodes(translation);
    const auto surface = surface_vertices(translation);
    const auto bindings = surface_bindings();
    const PmSoftBodyInfo soft = soft_info(nodes, surface, bindings);
    const PmStepInfo preparation{
        timestep, 1U, {preparation_gravity, 0.0F, 0.0F}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSoftBodies(context.commands, fixture.simulation, 0U, 1U,
                                  &soft) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                &preparation) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmRigidBodyInfo rigid = rigid_body(fixture.geometry);
    const PmStepInfo impact{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteRigidBodies(context.commands, fixture.simulation, 0U,
                                   1U, &rigid) == PM_SUCCESS);
    record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    const Snapshot initial = read_snapshot(context, fixture);

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                &impact) == PM_SUCCESS);
    record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    const Snapshot final = read_snapshot(context, fixture);

    Json result = settings_json(translation);
    result["id"] = id;
    result["initial"] = snapshot_json(initial);
    result["final"] = snapshot_json(final);
    Json driver = driver_json(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    pmDestroyGeometry(context.device, fixture.geometry);
    return {std::move(driver), std::move(result)};
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

[[nodiscard]] std::string endpoint_fingerprint(
    const char *label, const Snapshot *snapshot) {
    if (snapshot == nullptr)
        return sha256("parallel-mater-soft-rigid-endpoints/v1;inactive");
    std::ostringstream canonical;
    canonical << "parallel-mater-soft-rigid-endpoints/v1;label=" << label;
    for (const auto &node : snapshot->nodes) {
        canonical << ";node=";
        append_vector(canonical, node.position);
        canonical << ',';
        append_vector(canonical, node.velocity);
    }
    for (const auto &vertex : snapshot->surface) {
        canonical << ";surface=";
        append_vector(canonical, vertex);
    }
    return sha256(canonical.str());
}

[[nodiscard]] std::string rigid_fingerprint(const PmRigidState &rigid) {
    std::ostringstream canonical;
    canonical << "parallel-mater-soft-rigid-rigid-endpoint/v1;position=";
    append_vector(canonical, rigid.position);
    canonical << ";linear=";
    append_vector(canonical, rigid.linearVelocity);
    canonical << ";angular=";
    append_vector(canonical, rigid.angularVelocity);
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

void add_checkpoint(Json &observations, const char *id, const char *outcome,
                    bool active, std::uint64_t generation, const char *label,
                    const Snapshot *snapshot, const PmRigidState &rigid) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        "soft-rigid-contact-endpoints", active, generation,
        endpoint_fingerprint(label, snapshot)));
    resources.push_back(resource_json(
        "rigid-endpoint", true, 1U, rigid_fingerprint(rigid)));
    Json checkpoint = Json::object();
    checkpoint["id"] = id;
    checkpoint["outcome"] = outcome;
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
        throw std::runtime_error("unsupported: soft-rigid contact features");
    require_contact_capability(context);
    FixtureObjects fixture = create_fixture(context);
    const auto bindings = surface_bindings();
    const PmRigidBodyInfo rigid = rigid_body(fixture.geometry);
    Json observations = Json::array();

    const auto mutate = [&](float translation, bool active,
                            bool rejected_over_capacity) {
        const auto nodes = translated_nodes(translation);
        const auto surface = surface_vertices(translation);
        PmSoftBodyInfo soft = soft_info(nodes, surface, bindings);
        if (!active) soft = {};
        soft.active = active ? PM_TRUE : PM_FALSE;
        if (rejected_over_capacity) soft.nodeCount = 5U;
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        if (active || rejected_over_capacity) {
            const PmResult expected = rejected_over_capacity
                ? PM_ERROR_INVALID_ARGUMENT : PM_SUCCESS;
            PM_CHECK(pmCmdWriteSoftBodies(context.commands, fixture.simulation,
                                          0U, 1U, &soft) == expected);
        } else {
            PM_CHECK(pmCmdWriteSoftBodies(context.commands, fixture.simulation,
                                          0U, 1U, &soft) == PM_SUCCESS);
        }
        record_copies(context, fixture, active);
        PM_CHECK(context.submit() == PM_SUCCESS);
        if (active) return read_snapshot(context, fixture);
        context.invalidate(fixture.output);
        Snapshot result{};
        result.rigid = *reinterpret_cast<const PmRigidState *>(
            static_cast<const std::byte *>(fixture.output.mapped) +
            fixture.rigid.offset);
        return result;
    };

    // Install the unrelated rigid endpoint exactly once.
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteRigidBodies(context.commands, fixture.simulation, 0U,
                                   1U, &rigid) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    Snapshot baseline = mutate(0.0F, true, false);
    add_checkpoint(observations, "baseline", "PM_SUCCESS", true, 1U,
                   "baseline", &baseline, baseline.rigid);
    Snapshot replacement = mutate(separated_translation, true, false);
    add_checkpoint(observations, "replacement", "PM_SUCCESS", true, 2U,
                   "replacement", &replacement, replacement.rigid);
    Snapshot rejected = mutate(separated_translation, true, true);
    add_checkpoint(observations, "rejected-over-capacity",
                   "PM_ERROR_INVALID_ARGUMENT", true, 2U,
                   "replacement", &rejected, rejected.rigid);
    Snapshot removed = mutate(0.0F, false, false);
    add_checkpoint(observations, "removal", "PM_SUCCESS", false, 3U,
                   "inactive", nullptr, removed.rigid);
    Snapshot recovery = mutate(0.0F, true, false);
    add_checkpoint(observations, "recovery", "PM_SUCCESS", true, 4U,
                   "baseline", &recovery, recovery.rigid);

    Json driver = driver_json(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    pmDestroyGeometry(context.device, fixture.geometry);
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}

} // namespace

std::string conformance_soft_rigid_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "soft-rigid-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/soft-rigid-contact-v1.json");
    try {
        QualityRun impact = run_quality_fixture(
            0.0F, "centered-bound-surface-impact");
        QualityRun control = run_quality_fixture(
            separated_translation, "separated-no-contact-control");
        Json fixtures = Json::array();
        fixtures.push_back(std::move(impact.fixture));
        fixtures.push_back(std::move(control.fixture));
        result["execution_status"] = "completed";
        result["driver"] = std::move(impact.driver);
        result["fixtures"] = std::move(fixtures);
        if (status != nullptr) *status = ExecutionStatus::completed;
    } catch (const std::exception &error) {
        const bool unsupported = unsupported_error(error);
        result["execution_status"] = unsupported ? "unsupported" : "failed";
        result["reason"] = error.what();
        result["driver"] = unavailable_driver();
        result["fixtures"] = Json::array();
        if (status != nullptr) *status = unsupported
            ? ExecutionStatus::unsupported : ExecutionStatus::failed;
    }
    return result.serialize();
}

std::string conformance_soft_rigid_contact_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "soft-rigid-contact-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/soft-rigid-contact-lifecycle-v1.json");
    try {
        LifecycleRun run = run_lifecycle_fixture();
        Json families = Json::array();
        families.push_back("soft_body");
        families.push_back("rigid_body");
        Json unrelated = Json::array();
        unrelated.push_back("rigid-endpoint");
        Json attempt = Json::object();
        attempt["fixture"] = "soft-rigid-intrinsic-endpoint-lifecycle";
        attempt["status"] = "completed";
        Json item = Json::object();
        item["attempt"] = std::move(attempt);
        item["execution_status"] = "completed";
        item["families"] = std::move(families);
        item["id"] =
            "soft-rigid-contact-capacity-replacement-removal-rollback";
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
        if (status != nullptr) *status = unsupported
            ? ExecutionStatus::unsupported : ExecutionStatus::failed;
    }
    return result.serialize();
}

} // namespace parallel_mater::conformance
