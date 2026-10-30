// SPDX-License-Identifier: MIT
// Public-SDK graph-free rope/rigid contact quality fixture.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace parallel_mater::conformance {
namespace {
using parallel_mater::test::Buffer;
using parallel_mater::test::Context;

constexpr float preparation_timestep = 1.0F / 60.0F;
constexpr PmFloat3 preparation_gravity{35.0F, 31.0F, 0.0F};
constexpr float quality_timestep = 1.0F / 60.0F;
constexpr float rope_node_spacing = 0.06F;
constexpr float rope_radius = 0.03F;
constexpr float rope_mass = 0.5F;
constexpr float rigid_mass = 100.0F;
constexpr float cube_half_extent = 0.1F;
constexpr float rigid_collision_margin = 0.005F;
constexpr float cube_inertia =
    (1.0F / 6.0F) * rigid_mass * (2.0F * cube_half_extent) *
    (2.0F * cube_half_extent);
constexpr float expected_impact_impulse =
    preparation_gravity.x * preparation_timestep /
    (1.0F / rope_mass + 1.0F / rigid_mass);

constexpr std::array<PmFloat3, 2U> impact_centerline{{
    {-0.144722222F, -0.038611111F, 0.0F},
    {-0.144722222F, 0.021388889F, 0.0F}}};
constexpr std::array<PmFloat3, 2U> impact_prepared{{
    {-0.135F, -0.03F, 0.0F}, {-0.135F, 0.03F, 0.0F}}};
constexpr std::array<PmFloat3, 2U> control_centerline{{
    {-0.299722222F, -0.038611111F, 0.0F},
    {-0.299722222F, 0.021388889F, 0.0F}}};
constexpr std::array<PmFloat3, 2U> control_prepared{{
    {-0.29F, -0.03F, 0.0F}, {-0.29F, 0.03F, 0.0F}}};

constexpr std::array<PmFloat3, 8U> cube_vertices{{
    {-cube_half_extent, -cube_half_extent, -cube_half_extent},
    { cube_half_extent, -cube_half_extent, -cube_half_extent},
    { cube_half_extent,  cube_half_extent, -cube_half_extent},
    {-cube_half_extent,  cube_half_extent, -cube_half_extent},
    {-cube_half_extent, -cube_half_extent,  cube_half_extent},
    { cube_half_extent, -cube_half_extent,  cube_half_extent},
    { cube_half_extent,  cube_half_extent,  cube_half_extent},
    {-cube_half_extent,  cube_half_extent,  cube_half_extent}}};
constexpr std::array<std::uint32_t, 36U> cube_indices{{
    0U, 3U, 2U, 0U, 2U, 1U, 4U, 5U, 6U, 4U, 6U, 7U,
    0U, 4U, 7U, 0U, 7U, 3U, 1U, 2U, 6U, 1U, 6U, 5U,
    0U, 1U, 5U, 0U, 5U, 4U, 3U, 7U, 6U, 3U, 6U, 2U}};

constexpr PmDeviceSize rope_snapshot_bytes = sizeof(PmDomainOutputHeader) +
    impact_centerline.size() * sizeof(PmRopeNodeState);
constexpr PmDeviceSize raw_snapshot_bytes =
    rope_snapshot_bytes + sizeof(PmRigidState);
constexpr PmDeviceSize snapshot_alignment = alignof(PmDomainOutputHeader);
constexpr PmDeviceSize snapshot_stride =
    ((raw_snapshot_bytes + snapshot_alignment - 1U) / snapshot_alignment) *
    snapshot_alignment;

struct Snapshot {
    std::uint32_t rope_count{};
    std::array<PmRopeNodeState, 2U> rope{};
    PmRigidState rigid{};
};

struct SnapshotRanges {
    PmBufferRange rope{};
    PmBufferRange rigid{};
};

struct FixtureObjects {
    PmGeometry geometry{};
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

[[nodiscard]] Json vector_json(const float value[3]) {
    return vector_json({value[0], value[1], value[2]});
}

[[nodiscard]] Json quaternion_json(const float value[4]) {
    Json result = Json::array();
    for (std::size_t index = 0U; index < 4U; ++index)
        result.push_back(static_cast<double>(value[index]));
    return result;
}

template <std::size_t Size>
[[nodiscard]] Json vectors_json(const std::array<PmFloat3, Size> &values) {
    Json result = Json::array();
    for (const auto value : values) result.push_back(vector_json(value));
    return result;
}

template <typename Value, std::size_t Size>
[[nodiscard]] Json numbers_json(const std::array<Value, Size> &values) {
    Json result = Json::array();
    for (const auto value : values) {
        if constexpr (std::is_floating_point_v<Value>)
            result.push_back(static_cast<double>(value));
        else
            result.push_back(static_cast<std::uint64_t>(value));
    }
    return result;
}

[[nodiscard]] const char *native_driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult status = pmGetNativeDevice(device, &native);
    if (status == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (status != PM_SUCCESS)
        throw std::runtime_error("query rope-rigid contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown rope-rigid contact backend");
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
    needed.rigidBodies = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

[[nodiscard]] bool supports(PmPhysicalDevice physical) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_ROPE,
        PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
        PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE};
    PmInteractionCapability capability{};
    return pmGetPhysicalDeviceInteractionCapability(
               physical, &key, &capability) == PM_SUCCESS &&
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE;
}

[[nodiscard]] PmRopeMaterialState rope_material() {
    PmRopeMaterialState material{};
    material.nodeSpacing = rope_node_spacing;
    material.radius = rope_radius;
    material.mass = rope_mass;
    material.stretchCompliance = 0.0F;
    material.velocityDamping = 0.0F;
    material.maximumSubstepTimestep = quality_timestep;
    material.friction = 0.0F;
    material.maximumSpeed = 1.75F;
    material.solverIterations = 2U;
    material.selfCollision = PM_FALSE;
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

[[nodiscard]] PmRopeInfo rope_info(
    const std::array<PmFloat3, 2U> &centerline) {
    PmRopeInfo rope{};
    rope.active = PM_TRUE;
    rope.materialIndex = 0U;
    rope.pointCount = static_cast<std::uint32_t>(centerline.size());
    rope.pCenterline = centerline.data();
    rope.first.enabled = PM_FALSE;
    rope.last.enabled = PM_FALSE;
    return rope;
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

[[nodiscard]] SnapshotRanges make_ranges(PmBuffer buffer,
                                         PmDeviceSize offset) {
    return {{buffer, offset, rope_snapshot_bytes},
            {buffer, offset + rope_snapshot_bytes, sizeof(PmRigidState)}};
}

[[nodiscard]] FixtureObjects create_fixture(Context &context) {
    constexpr PmRopeCapacity capacity{
        static_cast<std::uint32_t>(impact_centerline.size())};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    simulation_info.ropeCount = 1U;
    simulation_info.pRopeCapacities = &capacity;
    FixtureObjects fixture{};
    fixture.geometry = create_geometry(context);
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
    const PmRigidSolverState rigid{4U, PM_FALSE, 0.0F, 0.0F, 0.0F};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid;
    pipeline_info.ropeMaterialCount = 1U;
    pipeline_info.pRopeMaterials = &rope;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    fixture.output = context.buffer(
        2U * snapshot_stride,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.initial = make_ranges(fixture.output.handle, 0U);
    fixture.final = make_ranges(fixture.output.handle, snapshot_stride);
    return fixture;
}

void record_copies(Context &context, const FixtureObjects &fixture,
                   const SnapshotRanges &ranges) {
    PM_CHECK(pmCmdCopyRopeState(context.commands, fixture.simulation, 0U,
                               &ranges.rope) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRigidStates(context.commands, fixture.simulation, 0U, 1U,
                                 &ranges.rigid) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture,
                                     const SnapshotRanges &ranges) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + ranges.rope.offset);
    if (header->activeCount != impact_centerline.size())
        throw std::runtime_error("rope-rigid output count mismatch");
    Snapshot result{};
    result.rope_count = header->activeCount;
    const auto *rope = reinterpret_cast<const PmRopeNodeState *>(header + 1);
    std::copy_n(rope, result.rope_count, result.rope.begin());
    result.rigid = *reinterpret_cast<const PmRigidState *>(
        base + ranges.rigid.offset);
    return result;
}

[[nodiscard]] Json rope_state_json(const PmRopeNodeState &state,
                                   std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(state.position);
    result["velocity_m_s"] = vector_json(state.velocity);
    result["rigid_contact_force_n"] = vector_json(state.contactForce);
    return result;
}

[[nodiscard]] Json rigid_state_json(const PmRigidState &state) {
    Json result = Json::object();
    result["position_m"] = vector_json(state.position);
    result["orientation_xyzw"] = quaternion_json(state.orientation);
    result["linear_velocity_m_s"] = vector_json(state.linearVelocity);
    result["angular_velocity_rad_s"] = vector_json(state.angularVelocity);
    return result;
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot) {
    Json rope = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.rope_count; ++index)
        rope.push_back(rope_state_json(snapshot.rope[index], index));
    Json result = Json::object();
    result["rope"] = std::move(rope);
    result["rigid"] = rigid_state_json(snapshot.rigid);
    return result;
}

[[nodiscard]] Json settings_json(
    const char *id, const std::array<PmFloat3, 2U> &centerline,
    const std::array<PmFloat3, 2U> &prepared, bool expected_contact) {
    const auto rope = rope_material();
    Json impact_face = Json::array();
    for (const auto index : std::array<std::uint32_t, 4U>{0U, 4U, 7U, 3U})
        impact_face.push_back(static_cast<std::uint64_t>(index));
    constexpr std::array<float, 4U> identity{{0.0F, 0.0F, 0.0F, 1.0F}};
    Json result = Json::object();
    result["id"] = id;
    result["graph_policy"] = "none";
    result["expected_contact"] = expected_contact;
    result["preparation_timestep_seconds"] = preparation_timestep;
    result["preparation_gravity_m_s2"] = vector_json(preparation_gravity);
    result["timestep_seconds"] = quality_timestep;
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["authored_rope_centerline_m"] = vectors_json(centerline);
    result["prepared_rope_nodes_m"] = vectors_json(prepared);
    result["prepared_rope_velocity_m_s"] =
        vector_json(PmFloat3{
            preparation_gravity.x * preparation_timestep,
            preparation_gravity.y * preparation_timestep, 0.0F});
    result["rope_node_count"] =
        static_cast<std::uint64_t>(centerline.size());
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
    result["rope_first_attached"] = false;
    result["rope_last_attached"] = false;
    result["rigid_geometry_vertices_m"] = vectors_json(cube_vertices);
    result["rigid_geometry_triangle_indices"] = numbers_json(cube_indices);
    result["impact_face_vertex_indices"] = std::move(impact_face);
    result["rigid_initial_position_m"] = vector_json(PmFloat3{});
    result["rigid_initial_orientation_xyzw"] =
        quaternion_json(identity.data());
    result["rigid_mass_kg"] = rigid_mass;
    result["rigid_inertia_diagonal_kg_m2"] =
        vector_json(PmFloat3{cube_inertia, cube_inertia, cube_inertia});
    result["rigid_friction"] = 0.0;
    result["rigid_restitution"] = 0.0;
    result["rigid_linear_damping_per_s"] = 0.0;
    result["rigid_angular_damping_per_s"] = 0.0;
    result["rigid_maximum_linear_speed_m_s"] = 10.0;
    result["rigid_maximum_angular_speed_rad_s"] = 10.0;
    result["rigid_collision_margin_m"] = rigid_collision_margin;
    result["combined_clearance_m"] = rope_radius + rigid_collision_margin;
    result["contact_normal"] = vector_json(PmFloat3{-1.0F, 0.0F, 0.0F});
    result["expected_coherent_reduced_mass_impulse_n_s"] =
        expected_contact ? expected_impact_impulse : 0.0F;
    result["two_way"] = true;
    result["oracle"] = expected_contact
        ? "public-capsule-cube-clearance-coherent-reduced-mass-reciprocal-impulse-force-and-frictionless-slip"
        : "public-separated-control-free-flight-zero-force-and-stationary-rigid";
    return result;
}

[[nodiscard]] Json run_fixture(
    Context &context, const char *id,
    const std::array<PmFloat3, 2U> &centerline,
    const std::array<PmFloat3, 2U> &prepared, bool expected_contact) {
    FixtureObjects fixture = create_fixture(context);
    const PmRopeInfo rope = rope_info(centerline);
    const PmStepInfo preparation{
        preparation_timestep, 1U,
        {preparation_gravity.x, preparation_gravity.y,
         preparation_gravity.z}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteRopes(context.commands, fixture.simulation, 0U, 1U,
                            &rope) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                &preparation) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmRigidBodyInfo body = rigid_body(fixture.geometry);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteRigidBodies(context.commands, fixture.simulation, 0U,
                                  1U, &body) == PM_SUCCESS);
    record_copies(context, fixture, fixture.initial);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmStepInfo contact{quality_timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                &contact) == PM_SUCCESS);
    record_copies(context, fixture, fixture.final);
    const PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("rope-rigid contact submission failed: " +
                                 std::to_string(submitted));

    const Snapshot initial = read_snapshot(context, fixture, fixture.initial);
    const Snapshot final = read_snapshot(context, fixture, fixture.final);
    Json result = settings_json(id, centerline, prepared, expected_contact);
    result["initial"] = snapshot_json(initial);
    result["final"] = snapshot_json(final);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    pmDestroyGeometry(context.device, fixture.geometry);
    return result;
}

struct QualityRun {
    Json driver{};
    Json fixtures{};
};

[[nodiscard]] QualityRun run_quality_fixtures() {
    Context context;
    if (!context.open(required_features()) || !supports(context.physical))
        throw std::runtime_error("unsupported: rope-rigid contact tuple");
    Json fixtures = Json::array();
    fixtures.push_back(run_fixture(
        context, "rope-segment-centered-cube-impact", impact_centerline,
        impact_prepared, true));
    fixtures.push_back(run_fixture(
        context, "rope-segment-separated-cube-control", control_centerline,
        control_prepared, false));
    return {driver_json(context), std::move(fixtures)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}
} // namespace

std::string conformance_rope_rigid_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "rope-rigid-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/rope-rigid-contact-v1.json");
    try {
        QualityRun run = run_quality_fixtures();
        result["execution_status"] = "completed";
        result["driver"] = std::move(run.driver);
        result["fixtures"] = std::move(run.fixtures);
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
