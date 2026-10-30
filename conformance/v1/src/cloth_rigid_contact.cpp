// SPDX-License-Identifier: MIT
// Public-SDK graph-free cloth/rigid contact quality and endpoint lifecycle.
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
#include <type_traits>
#include <utility>

namespace parallel_mater::conformance {
namespace {
using parallel_mater::test::Buffer;
using parallel_mater::test::Context;

constexpr float preparation_timestep = 1.0F / 60.0F;
constexpr PmFloat3 preparation_gravity{60.0F, 60.0F, 0.0F};
constexpr float quality_timestep = 1.0F / 60.0F;
constexpr float cloth_vertex_mass = 1.0F;
constexpr float rigid_mass = 2.0F;
constexpr float cube_half_extent = 0.1F;
constexpr float rigid_collision_margin = 0.005F;
constexpr float cube_inertia =
    (1.0F / 6.0F) * rigid_mass * (2.0F * cube_half_extent) *
    (2.0F * cube_half_extent);

constexpr std::array<PmFloat3, 3U> cloth_vertices{{
    {-0.13166667F, -0.03333334F, 0.0F},
    {-1.00666667F, 0.98333333F, -1.0F},
    {-1.00666667F, -1.01666667F, -1.0F}}};
constexpr std::array<PmFloat3, 3U> prepared_cloth_vertices{{
    {-0.115F, -0.01666667F, 0.0F},
    {-0.99F, 1.0F, -1.0F},
    {-0.99F, -1.0F, -1.0F}}};
constexpr std::array<std::uint32_t, 3U> cloth_indices{{0U, 1U, 2U}};
constexpr std::array<float, 3U> cloth_inverse_masses{{1.0F, 1.0F, 1.0F}};
constexpr std::array<float, 3U> pinned_cloth{{0.0F, 0.0F, 0.0F}};
constexpr PmDeviceSize cloth_snapshot_bytes = sizeof(PmDomainOutputHeader) +
    cloth_vertices.size() * sizeof(PmClothVertexState);
constexpr PmDeviceSize raw_snapshot_bytes =
    cloth_snapshot_bytes + sizeof(PmRigidState);
constexpr PmDeviceSize snapshot_alignment = alignof(PmDomainOutputHeader);
constexpr PmDeviceSize snapshot_stride =
    ((raw_snapshot_bytes + snapshot_alignment - 1U) / snapshot_alignment) *
    snapshot_alignment;

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

constexpr std::array<PmFloat3, 3U> lifecycle_baseline{{
    {-10.0F, 0.0F, 0.0F}, {-10.0F, 1.0F, -1.0F},
    {-10.0F, -1.0F, -1.0F}}};
constexpr std::array<PmFloat3, 3U> lifecycle_replacement{{
    {-11.0F, 0.0F, 0.0F}, {-11.0F, 1.0F, -1.0F},
    {-11.0F, -1.0F, -1.0F}}};
constexpr std::array<PmFloat3, 3U> lifecycle_recovery{{
    {-12.0F, 0.0F, 0.0F}, {-12.0F, 1.0F, -1.0F},
    {-12.0F, -1.0F, -1.0F}}};

struct Snapshot {
    std::uint32_t cloth_count{};
    std::array<PmClothVertexState, 3U> cloth{};
    PmRigidState rigid{};
};

struct SnapshotRanges {
    PmBufferRange cloth{};
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
        throw std::runtime_error("query cloth-rigid contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown cloth-rigid contact backend");
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
    needed.rigidBodies = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

[[nodiscard]] bool supports(PmPhysicalDevice physical) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_CLOTH,
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
    body.maximumLinearSpeed = 5.0F;
    body.maximumAngularSpeed = 10.0F;
    body.collisionMargin = rigid_collision_margin;
    return body;
}

[[nodiscard]] PmClothInfo cloth_info(
    const std::array<PmFloat3, 3U> &vertices, const float *inverse_masses) {
    return {PM_TRUE, 0U, static_cast<std::uint32_t>(vertices.size()),
            vertices.data(), static_cast<std::uint32_t>(cloth_indices.size()),
            cloth_indices.data(), inverse_masses};
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
    constexpr PmDeviceSize rigid_bytes = sizeof(PmRigidState);
    return {{buffer, offset, cloth_snapshot_bytes},
            {buffer, offset + cloth_snapshot_bytes, rigid_bytes}};
}

[[nodiscard]] FixtureObjects create_fixture(Context &context) {
    constexpr PmClothCapacity capacity{
        static_cast<std::uint32_t>(cloth_vertices.size()),
        static_cast<std::uint32_t>(cloth_indices.size()),
        static_cast<std::uint32_t>(cloth_indices.size())};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    simulation_info.clothCount = 1U;
    simulation_info.pClothCapacities = &capacity;
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

    const PmClothMaterialState cloth = cloth_material();
    const PmRigidSolverState rigid{4U, PM_FALSE, 0.0F, 0.0F, 0.0F};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid;
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &cloth;
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
    PM_CHECK(pmCmdCopyClothState(context.commands, fixture.simulation, 0U,
                                &ranges.cloth) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRigidStates(context.commands, fixture.simulation, 0U, 1U,
                                 &ranges.rigid) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture,
                                     const SnapshotRanges &ranges,
                                     bool cloth_active = true) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    Snapshot result{};
    if (cloth_active) {
        const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(
            base + ranges.cloth.offset);
        if (header->activeCount > cloth_vertices.size())
            throw std::runtime_error("cloth-rigid output count mismatch");
        result.cloth_count = header->activeCount;
        const auto *cloth =
            reinterpret_cast<const PmClothVertexState *>(header + 1);
        std::copy_n(cloth, result.cloth_count, result.cloth.begin());
    }
    result.rigid = *reinterpret_cast<const PmRigidState *>(
        base + ranges.rigid.offset);
    return result;
}

[[nodiscard]] Json cloth_state_json(const PmClothVertexState &state,
                                    std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(state.position);
    result["velocity_m_s"] = vector_json(state.velocity);
    result["rigid_contact_force_n"] = vector_json(state.rigidContactForce);
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
    Json cloth = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.cloth_count; ++index)
        cloth.push_back(cloth_state_json(snapshot.cloth[index], index));
    Json result = Json::object();
    result["cloth"] = std::move(cloth);
    result["rigid"] = rigid_state_json(snapshot.rigid);
    return result;
}

[[nodiscard]] Json settings_json() {
    const auto cloth = cloth_material();
    Json impact_face = Json::array();
    for (const auto index : std::array<std::uint32_t, 4U>{0U, 4U, 7U, 3U})
        impact_face.push_back(static_cast<std::uint64_t>(index));
    Json result = Json::object();
    result["id"] = "cloth-vertex-centered-cube-impact";
    result["graph_policy"] = "none";
    result["preparation_timestep_seconds"] = preparation_timestep;
    result["preparation_gravity_m_s2"] = vector_json(preparation_gravity);
    result["timestep_seconds"] = quality_timestep;
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["authored_cloth_vertices_m"] = vectors_json(cloth_vertices);
    result["prepared_cloth_vertices_m"] =
        vectors_json(prepared_cloth_vertices);
    result["prepared_cloth_velocity_m_s"] =
        vector_json(PmFloat3{1.0F, 1.0F, 0.0F});
    result["cloth_triangle_indices"] = numbers_json(cloth_indices);
    result["cloth_inverse_masses_kg_inverse"] =
        numbers_json(cloth_inverse_masses);
    result["cloth_vertex_mass_kg"] = cloth.vertexMass;
    result["cloth_thickness_m"] = cloth.thickness;
    result["cloth_stretch_compliance_m_per_n"] = cloth.stretchCompliance;
    result["cloth_bending_compliance_m_per_n"] = cloth.bendingCompliance;
    result["cloth_velocity_damping"] = cloth.velocityDamping;
    result["cloth_contact_friction"] = cloth.contactFriction;
    result["cloth_solver_iterations"] =
        static_cast<std::uint64_t>(cloth.solverIterations);
    result["cloth_preserve_volume"] = false;
    result["contact_cloth_vertex_index"] = static_cast<std::uint64_t>(0U);
    result["rigid_geometry_vertices_m"] = vectors_json(cube_vertices);
    result["rigid_geometry_triangle_indices"] = numbers_json(cube_indices);
    result["impact_face_vertex_indices"] = std::move(impact_face);
    result["rigid_initial_position_m"] = vector_json(PmFloat3{});
    result["rigid_initial_orientation_xyzw"] =
        quaternion_json(std::array<float, 4U>{0.0F, 0.0F, 0.0F, 1.0F}.data());
    result["rigid_mass_kg"] = rigid_mass;
    result["rigid_inertia_diagonal_kg_m2"] =
        vector_json(PmFloat3{cube_inertia, cube_inertia, cube_inertia});
    result["rigid_friction"] = 0.0;
    result["rigid_restitution"] = 0.0;
    result["rigid_linear_damping_per_s"] = 0.0;
    result["rigid_angular_damping_per_s"] = 0.0;
    result["rigid_maximum_linear_speed_m_s"] = 5.0;
    result["rigid_maximum_angular_speed_rad_s"] = 10.0;
    result["rigid_collision_margin_m"] = rigid_collision_margin;
    result["two_way"] = true;
    result["oracle"] =
        "public-cube-face-clearance-reduced-mass-reciprocal-impulse-force-and-frictionless-slip";
    return result;
}

struct QualityRun {
    Json driver{};
    Json fixture{};
};

[[nodiscard]] QualityRun run_quality_fixture() {
    Context context;
    if (!context.open(required_features()) || !supports(context.physical))
        throw std::runtime_error("unsupported: cloth-rigid contact tuple");
    FixtureObjects fixture = create_fixture(context);
    const PmClothInfo cloth = cloth_info(cloth_vertices,
                                         cloth_inverse_masses.data());
    const PmStepInfo preparation{
        preparation_timestep, 1U,
        {preparation_gravity.x, preparation_gravity.y,
         preparation_gravity.z}, 0U};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, fixture.simulation, 0U, 1U,
                             &cloth) == PM_SUCCESS);
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
        throw std::runtime_error("cloth-rigid contact submission failed: " +
                                 std::to_string(submitted));

    const Snapshot initial = read_snapshot(context, fixture, fixture.initial);
    const Snapshot final = read_snapshot(context, fixture, fixture.final);
    Json result = settings_json();
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
    append_float(stream, value.x);
    stream << ',';
    append_float(stream, value.y);
    stream << ',';
    append_float(stream, value.z);
}

[[nodiscard]] std::string pair_fingerprint(bool active,
                                           const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-cloth-rigid-contact-pair/v1;active="
              << active << ";cloth-count=" << snapshot.cloth_count;
    for (std::uint32_t index = 0U; index < snapshot.cloth_count; ++index) {
        canonical << ";cloth=" << index << ",position=";
        append_vector(canonical, snapshot.cloth[index].position);
        canonical << ",velocity=";
        append_vector(canonical, snapshot.cloth[index].velocity);
    }
    canonical << ";rigid-position=";
    append_vector(canonical, snapshot.rigid.position);
    canonical << ";rigid-linear=";
    append_vector(canonical, snapshot.rigid.linearVelocity);
    return sha256(canonical.str());
}

[[nodiscard]] std::string rigid_fingerprint(const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-cloth-rigid-counterpart/v1;position=";
    append_vector(canonical, snapshot.rigid.position);
    canonical << ";linear=";
    append_vector(canonical, snapshot.rigid.linearVelocity);
    canonical << ";angular=";
    append_vector(canonical, snapshot.rigid.angularVelocity);
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
                    bool active, std::uint64_t generation,
                    const Snapshot &snapshot) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        "cloth-rigid-contact-pair", active, generation,
        pair_fingerprint(active, snapshot)));
    resources.push_back(resource_json(
        "cloth-rigid-rigid-counterpart", true, 1U,
        rigid_fingerprint(snapshot)));
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
        throw std::runtime_error("unsupported: cloth-rigid contact tuple");
    FixtureObjects fixture = create_fixture(context);
    const PmRigidBodyInfo body = rigid_body(fixture.geometry);
    const PmClothInfo baseline = cloth_info(lifecycle_baseline,
                                             pinned_cloth.data());
    const PmClothInfo replacement = cloth_info(lifecycle_replacement,
                                                pinned_cloth.data());
    const PmClothInfo recovery = cloth_info(lifecycle_recovery,
                                             pinned_cloth.data());
    PmClothInfo removed{};
    removed.active = PM_FALSE;
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    Json observations = Json::array();

    const auto mutate = [&](const auto &record, PmResult expected,
                            bool cloth_active) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(record() == expected);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        if (cloth_active)
            PM_CHECK(pmCmdCopyClothState(context.commands, fixture.simulation,
                                        0U, &fixture.final.cloth) == PM_SUCCESS);
        PM_CHECK(pmCmdCopyRigidStates(context.commands, fixture.simulation, 0U,
                                     1U, &fixture.final.rigid) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
        return read_snapshot(context, fixture, fixture.final, cloth_active);
    };

    add_checkpoint(observations, "baseline", "PM_SUCCESS", true, 1U,
        mutate([&] {
            const PmResult rigid_status = pmCmdWriteRigidBodies(
                context.commands, fixture.simulation, 0U, 1U, &body);
            return rigid_status == PM_SUCCESS ? pmCmdWriteCloths(
                context.commands, fixture.simulation, 0U, 1U, &baseline) :
                rigid_status;
        }, PM_SUCCESS, true));
    add_checkpoint(observations, "replacement", "PM_SUCCESS", true, 2U,
        mutate([&] {
            return pmCmdWriteCloths(context.commands, fixture.simulation, 0U,
                                    1U, &replacement);
        }, PM_SUCCESS, true));
    add_checkpoint(observations, "rejected-over-capacity",
        "PM_ERROR_INVALID_ARGUMENT", true, 2U,
        mutate([&] {
            return pmCmdWriteCloths(context.commands, fixture.simulation, 1U,
                                    1U, &baseline);
        }, PM_ERROR_INVALID_ARGUMENT, true));
    add_checkpoint(observations, "removal", "PM_SUCCESS", false, 3U,
        mutate([&] {
            return pmCmdWriteCloths(context.commands, fixture.simulation, 0U,
                                    1U, &removed);
        }, PM_SUCCESS, false));
    add_checkpoint(observations, "recovery", "PM_SUCCESS", true, 4U,
        mutate([&] {
            return pmCmdWriteCloths(context.commands, fixture.simulation, 0U,
                                    1U, &recovery);
        }, PM_SUCCESS, true));

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

std::string conformance_cloth_rigid_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "cloth-rigid-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/cloth-rigid-contact-v1.json");
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

std::string conformance_cloth_rigid_contact_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "cloth-rigid-contact-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/cloth-rigid-contact-lifecycle-v1.json");
    try {
        LifecycleRun run = run_lifecycle_fixture();
        Json families = Json::array();
        families.push_back("cloth");
        families.push_back("rigid_body");
        Json unrelated = Json::array();
        unrelated.push_back("cloth-rigid-rigid-counterpart");
        Json attempt = Json::object();
        attempt["fixture"] = "cloth-rigid-contact-public-lifecycle";
        attempt["status"] = "completed";
        Json item = Json::object();
        item["attempt"] = std::move(attempt);
        item["execution_status"] = "completed";
        item["families"] = std::move(families);
        item["id"] =
            "cloth-rigid-contact-capacity-replacement-removal-rollback";
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
