// SPDX-License-Identifier: MIT
// Public-SDK smoke/rigid contact quality and graph lifecycle fixtures.
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
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace parallel_mater::conformance {
namespace {
using parallel_mater::test::Buffer;
using parallel_mater::test::Context;

constexpr float timestep = 0.01F;
constexpr std::uint32_t warmup_steps = 2U;
constexpr std::uint32_t quality_steps = 1U;
constexpr std::uint32_t smoke_capacity = 1U;
constexpr std::uint32_t grid_resolution = 16U;
constexpr std::uint32_t grid_cells =
    grid_resolution * grid_resolution * grid_resolution;
constexpr float grid_edge_length = 0.16F;
constexpr float grid_spacing = grid_edge_length /
    static_cast<float>(grid_resolution);
constexpr float cube_half_extent = 0.02F;
constexpr float contact_distance = 0.015F;
constexpr float particle_radius = 0.005F;
constexpr float rigid_mass = 0.01F;
constexpr float rigid_collision_margin = 0.001F;
constexpr float cube_inertia = (1.0F / 6.0F) * rigid_mass *
    (2.0F * cube_half_extent) * (2.0F * cube_half_extent);
constexpr float air_density = 1'000.0F;
constexpr float drag_coefficient = 1.0F;
constexpr float quality_emission_rate = 60.0F;

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
    0U, 3U, 2U, 0U, 2U, 1U,
    4U, 5U, 6U, 4U, 6U, 7U,
    0U, 4U, 7U, 0U, 7U, 3U,
    1U, 2U, 6U, 1U, 6U, 5U,
    0U, 1U, 5U, 0U, 5U, 4U,
    3U, 7U, 6U, 3U, 6U, 2U,
}};

struct DriverInfo {
    std::string name{};
    std::string device{};
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t version{};
};

struct Snapshot {
    std::vector<PmSmokeParticleState> smoke{};
    PmRigidState rigid{};
};

struct FixtureObjects {
    PmGeometry geometry{};
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange smoke_output{};
    PmBufferRange rigid_output{};
};

struct QualityRun {
    DriverInfo driver{};
    Snapshot initial{};
    Snapshot final{};
};

[[nodiscard]] const char *native_driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult status = pmGetNativeDevice(device, &native);
    if (status == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (status != PM_SUCCESS)
        throw std::runtime_error("query smoke-rigid contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown smoke-rigid contact backend");
}

[[nodiscard]] DriverInfo driver_info(const Context &context) {
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    return {native_driver_name(context.device), properties.deviceName,
            properties.vendorId, properties.deviceId,
            properties.driverVersion};
}

[[nodiscard]] Json driver_json(const DriverInfo &driver) {
    Json result = Json::object();
    result["name"] = driver.name;
    result["device"] = driver.device;
    result["vendor_id"] = static_cast<std::uint64_t>(driver.vendor_id);
    result["device_id"] = static_cast<std::uint64_t>(driver.device_id);
    result["driver_version"] = std::to_string(driver.version);
    return result;
}

[[nodiscard]] Json unavailable_driver() {
    Json result = Json::object();
    result["name"] = "unavailable";
    result["device"] = "unavailable";
    return result;
}

[[nodiscard]] bool same_driver(const DriverInfo &first,
                               const DriverInfo &second) {
    return first.name == second.name && first.device == second.device &&
        first.vendor_id == second.vendor_id &&
        first.device_id == second.device_id && first.version == second.version;
}

[[nodiscard]] PmPhysicalDeviceFeatures required_features() {
    PmPhysicalDeviceFeatures needed{};
    needed.rigidBodies = PM_TRUE;
    needed.smoke = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

void require_contact_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_SMOKE, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(std::string("unsupported: smoke-rigid contact ") +
                                 (grid ? "SMOKE_GRID" : "SMOKE_PARTICLE"));
}

[[nodiscard]] PmSmokeMaterialState smoke_material(bool grid) {
    PmSmokeMaterialState material{};
    material.particleRadius = particle_radius;
    material.buoyancy = 0.0F;
    material.response = 0.0F;
    material.restNumberDensity = 1.0F;
    material.pressureStiffness = 0.0F;
    material.viscosity = 0.0F;
    material.vorticityConfinement = 0.0F;
    material.maximumSpeed = 2.0F;
    material.gridResolution = grid ? grid_resolution : 0U;
    material.gridVerticalResolution = grid ? grid_resolution : 0U;
    material.gridPressureIterations = grid ? 8U : 0U;
    material.gridPressureTolerance = 1.0e-5F;
    return material;
}

[[nodiscard]] PmSmokeInfo smoke_info(float emitter_x,
                                     PmFloat3 initial_velocity) {
    PmSmokeInfo info{};
    info.active = PM_TRUE;
    info.materialIndex = 0U;
    info.particleCapacity = smoke_capacity;
    info.emitterCenter = {emitter_x, 0.0F, 0.0F};
    info.emitterHalfExtents = {1.0e-6F, 1.0e-6F};
    info.initialVelocity = initial_velocity;
    info.wind = {};
    info.particlesPerSecond = quality_emission_rate;
    info.lifetime = 1.0F;
    info.gridMinimum = {-0.08F, -0.08F, -0.08F};
    info.gridEdgeLength = grid_edge_length;
    return info;
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
    body.maximumLinearSpeed = 2.0F;
    body.maximumAngularSpeed = 2.0F;
    body.collisionMargin = rigid_collision_margin;
    return body;
}

[[nodiscard]] PmCouplingState contact_coupling(bool grid, float distance) {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_SMOKE_RIGID;
    coupling.first = 0U;
    coupling.second = 0U;
    coupling.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_DRAG_BIT;
    coupling.materialProfile = grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
                                    : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    coupling.parameters.smoke.windDrag = 0.0F;
    coupling.parameters.smoke.maximumWindAcceleration = 0.0F;
    coupling.parameters.smoke.airDensity = air_density;
    coupling.parameters.smoke.dragCoefficient = drag_coefficient;
    coupling.parameters.smoke.contactDistance = distance;
    coupling.parameters.smoke.tracerContact = PM_TRUE;
    return coupling;
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

[[nodiscard]] FixtureObjects create_fixture(Context &context, bool grid,
                                            std::uint32_t coupling_capacity) {
    FixtureObjects fixture{};
    fixture.geometry = create_geometry(context);
    const PmSmokeCapacity smoke_domain{
        smoke_capacity, grid ? grid_cells : 1U};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.rigidBodyCapacity = 1U;
    create.smokeCount = 1U;
    create.pSmokeCapacities = &smoke_domain;
    create.couplingCapacity = coupling_capacity;
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
    const PmRigidSolverState rigid{4U, PM_FALSE, 0.0F, 0.0F, 0.0F};
    const PmSmokeMaterialState smoke = smoke_material(grid);
    PmPhysicsPipelineCreateInfo pipeline{};
    pipeline.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline.pRigidState = &rigid;
    pipeline.smokeMaterialCount = 1U;
    pipeline.pSmokeMaterials = &smoke;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline,
                                     &fixture.pipeline) == PM_SUCCESS);
    constexpr PmDeviceSize smoke_bytes = sizeof(PmDomainOutputHeader) +
        smoke_capacity * sizeof(PmSmokeParticleState);
    constexpr PmDeviceSize rigid_bytes = sizeof(PmRigidState);
    fixture.output = context.buffer(
        smoke_bytes + rigid_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.smoke_output = {fixture.output.handle, 0U, smoke_bytes};
    fixture.rigid_output = {fixture.output.handle, smoke_bytes, rigid_bytes};
    return fixture;
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopySmokeState(context.commands, fixture.simulation, 0U,
                                &fixture.smoke_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRigidStates(context.commands, fixture.simulation, 0U, 1U,
                                 &fixture.rigid_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(base);
    if (header->activeCount > smoke_capacity)
        throw std::runtime_error("smoke-rigid smoke output count mismatch");
    Snapshot result{};
    const auto *smoke = reinterpret_cast<const PmSmokeParticleState *>(
        header + 1);
    result.smoke.assign(smoke, smoke + header->activeCount);
    result.rigid = *reinterpret_cast<const PmRigidState *>(
        base + fixture.rigid_output.offset);
    return result;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture,
                          float emitter_x, PmFloat3 initial_velocity) {
    const PmRigidBodyInfo body = rigid_body(fixture.geometry);
    const PmSmokeInfo smoke = smoke_info(emitter_x, initial_velocity);
    const auto submit_stage = [&](const char *label, const auto &record) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        record();
        const PmResult submitted = context.submit();
        if (submitted != PM_SUCCESS)
            throw std::runtime_error(std::string("smoke-rigid ") + label +
                                     " failed: " +
                                     std::to_string(submitted));
    };
    submit_stage("rigid initialization", [&] {
        PM_CHECK(pmCmdWriteRigidBodies(context.commands, fixture.simulation,
                                       0U, 1U, &body) == PM_SUCCESS);
    });
    submit_stage("smoke initialization", [&] {
        PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                                &smoke) == PM_SUCCESS);
    });
}

[[nodiscard]] QualityRun run_quality_fixture(bool grid, bool coupled) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: smoke and rigid features");
    require_contact_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    initialize_endpoints(context, fixture, -0.043F,
                         {1.0F, 0.25F, 0.0F});
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    for (std::uint32_t index = 0U; index < warmup_steps; ++index) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    Snapshot initial = read_snapshot(context, fixture);
    if (initial.smoke.size() != 1U)
        throw std::runtime_error("smoke-rigid fixture did not emit one tracer");
    const PmCouplingState coupling = contact_coupling(grid, contact_distance);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation,
                              coupled ? 1U : 0U,
                              coupled ? &coupling : nullptr) == PM_SUCCESS);
    for (std::uint32_t index = 0U; index < quality_steps; ++index)
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
    record_copies(context, fixture);
    const PmResult status = context.submit();
    if (status != PM_SUCCESS)
        throw std::runtime_error("smoke-rigid contact submission failed: " +
                                 std::to_string(status));
    Snapshot final = read_snapshot(context, fixture);
    DriverInfo driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    pmDestroyGeometry(context.device, fixture.geometry);
    return {std::move(driver), std::move(initial), std::move(final)};
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
    for (std::size_t index = 0U; index < 4U; ++index)
        result.push_back(static_cast<double>(value[index]));
    return result;
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot) {
    Json smoke = Json::array();
    for (std::uint32_t slot = 0U; slot < snapshot.smoke.size(); ++slot) {
        const auto &particle = snapshot.smoke[slot];
        Json item = Json::object();
        item["slot"] = static_cast<std::uint64_t>(slot);
        item["position_m"] = vector_json(particle.position);
        item["velocity_m_s"] = vector_json(particle.velocity);
        item["age_seconds"] = static_cast<double>(particle.age);
        item["number_density"] = static_cast<double>(particle.numberDensity);
        item["pressure_m_s2"] = static_cast<double>(particle.pressure);
        item["vorticity_per_s"] = vector_json(particle.vorticity);
        smoke.push_back(std::move(item));
    }
    Json rigid = Json::object();
    rigid["position_m"] = vector_json(snapshot.rigid.position);
    rigid["orientation_xyzw"] = quaternion_json(snapshot.rigid.orientation);
    rigid["linear_velocity_m_s"] = vector_json(snapshot.rigid.linearVelocity);
    rigid["angular_velocity_rad_s"] = vector_json(snapshot.rigid.angularVelocity);
    Json result = Json::object();
    result["smoke_particles"] = std::move(smoke);
    result["rigid"] = std::move(rigid);
    return result;
}

[[nodiscard]] Json run_json(const char *id, const QualityRun &run) {
    Json result = Json::object();
    result["id"] = id;
    result["initial"] = snapshot_json(run.initial);
    result["final"] = snapshot_json(run.final);
    return result;
}

[[nodiscard]] Json vertices_json() {
    Json result = Json::array();
    for (const auto vertex : cube_vertices)
        result.push_back(vector_json(vertex));
    return result;
}

[[nodiscard]] Json indices_json() {
    Json result = Json::array();
    for (const auto index : cube_indices)
        result.push_back(static_cast<std::uint64_t>(index));
    return result;
}

[[nodiscard]] Json quality_fixture_json(bool grid, DriverInfo &driver) {
    QualityRun control = run_quality_fixture(grid, false);
    QualityRun coupled = run_quality_fixture(grid, true);
    if (!same_driver(control.driver, coupled.driver))
        throw std::runtime_error(
            "smoke-rigid paired runs selected different devices");
    driver = control.driver;
    Json modes = Json::array();
    modes.push_back("CONTACT");
    modes.push_back("DRAG");
    Json runs = Json::array();
    runs.push_back(run_json("control", control));
    runs.push_back(run_json("coupled", coupled));
    Json result = Json::object();
    result["id"] = grid ? "grid-smoke-rigid-contact" :
        "particle-smoke-rigid-contact";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["warmup_step_count"] = static_cast<std::uint64_t>(warmup_steps);
    result["step_count"] = static_cast<std::uint64_t>(quality_steps);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["declared_mode_flags"] = std::move(modes);
    result["exercised_mode"] = "CONTACT";
    result["rigid_geometry_vertices_m"] = vertices_json();
    result["rigid_geometry_triangle_indices"] = indices_json();
    Json face = Json::array();
    for (const std::uint64_t index : {0U, 4U, 7U, 3U})
        face.push_back(index);
    result["impact_face_vertex_indices"] = std::move(face);
    result["rigid_initial_position_m"] = vector_json(PmFloat3{});
    Json orientation = Json::array();
    for (const double value : {0.0, 0.0, 0.0, 1.0})
        orientation.push_back(value);
    result["rigid_initial_orientation_xyzw"] = std::move(orientation);
    result["rigid_mass_kg"] = static_cast<double>(rigid_mass);
    result["rigid_inertia_diagonal_kg_m2"] =
        vector_json(PmFloat3{cube_inertia, cube_inertia, cube_inertia});
    result["rigid_dynamic"] = true;
    result["rigid_friction"] = 0.0;
    result["rigid_restitution"] = 0.0;
    result["rigid_linear_damping_per_s"] = 0.0;
    result["rigid_angular_damping_per_s"] = 0.0;
    result["rigid_maximum_linear_speed_m_s"] = 2.0;
    result["rigid_maximum_angular_speed_rad_s"] = 2.0;
    result["rigid_collision_margin_m"] =
        static_cast<double>(rigid_collision_margin);
    result["smoke_particle_capacity"] =
        static_cast<std::uint64_t>(smoke_capacity);
    result["smoke_emitter_center_m"] =
        vector_json(PmFloat3{-0.043F, 0.0F, 0.0F});
    Json extents = Json::array();
    extents.push_back(1.0e-6);
    extents.push_back(1.0e-6);
    result["smoke_emitter_half_extents_m"] = std::move(extents);
    result["smoke_emission_rate_per_s"] =
        static_cast<double>(quality_emission_rate);
    result["smoke_initial_velocity_m_s"] =
        vector_json(PmFloat3{1.0F, 0.25F, 0.0F});
    result["smoke_wind_m_s"] = vector_json(PmFloat3{});
    result["smoke_lifetime_seconds"] = 1.0;
    result["smoke_particle_radius_m"] = static_cast<double>(particle_radius);
    result["smoke_rest_number_density"] = 1.0;
    result["smoke_pressure_stiffness_m2_s2"] = 0.0;
    result["smoke_viscosity_m2_s"] = 0.0;
    result["smoke_maximum_speed_m_s"] = 2.0;
    result["smoke_grid_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_resolution : 0U);
    result["smoke_grid_vertical_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_resolution : 0U);
    result["smoke_grid_minimum_m"] =
        vector_json(PmFloat3{-0.08F, -0.08F, -0.08F});
    result["smoke_grid_edge_length_m"] =
        static_cast<double>(grid_edge_length);
    result["smoke_grid_spacing_m"] =
        static_cast<double>(grid ? grid_spacing : 0.0F);
    result["coupling_contact_distance_m"] =
        static_cast<double>(contact_distance);
    result["coupling_tracer_contact"] = true;
    result["coupling_air_density_kg_m3"] = static_cast<double>(air_density);
    result["coupling_drag_coefficient"] =
        static_cast<double>(drag_coefficient);
    result["wind_drag_per_s"] = 0.0;
    result["maximum_wind_acceleration_m_s2"] = 0.0;
    result["oracle"] =
        "paired-public-cube-clearance-and-reciprocal-normal-response";
    result["runs"] = std::move(runs);
    return result;
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

[[nodiscard]] bool contact_separated(const Snapshot &snapshot,
                                     float distance) {
    if (snapshot.smoke.size() != 1U) return false;
    const float face = snapshot.rigid.position[0] - cube_half_extent;
    return snapshot.smoke[0].position.x <= face - distance + 0.0005F;
}

[[nodiscard]] bool rigid_responded(const Snapshot &snapshot) {
    return snapshot.rigid.linearVelocity[0] > 1.0e-7F;
}

[[nodiscard]] std::string graph_fingerprint(
    const PmCouplingState *coupling, const Snapshot &probe) {
    std::ostringstream canonical;
    canonical << "parallel-mater-smoke-rigid-contact-graph/v1"
              << ";contact-separated=" << contact_separated(probe, contact_distance)
              << ";rigid-responded=" << rigid_responded(probe);
    if (coupling == nullptr) {
        canonical << ";inactive";
        return sha256(canonical.str());
    }
    const auto &smoke = coupling->parameters.smoke;
    canonical << ";type=" << static_cast<unsigned>(coupling->type)
              << ";first=" << coupling->first
              << ";second=" << coupling->second
              << ";modes=" << coupling->modeFlags
              << ";qualified-mode=CONTACT;direction=BIDIRECTIONAL;material="
              << static_cast<unsigned>(coupling->materialProfile)
              << ";wind-drag=";
    append_float(canonical, smoke.windDrag);
    canonical << ";maximum-wind-acceleration=";
    append_float(canonical, smoke.maximumWindAcceleration);
    canonical << ";air-density=";
    append_float(canonical, smoke.airDensity);
    canonical << ";drag-coefficient=";
    append_float(canonical, smoke.dragCoefficient);
    canonical << ";contact-distance=";
    append_float(canonical, smoke.contactDistance);
    canonical << ";tracer-contact=" << smoke.tracerContact;
    return sha256(canonical.str());
}

[[nodiscard]] std::string endpoint_fingerprint(const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-smoke-rigid-contact-endpoints/v1"
              << ";smoke-count=" << snapshot.smoke.size()
              << ";rigid-count=1"
              << ";public-states-finite="
              << (snapshot.smoke.size() == 1U &&
                  std::isfinite(snapshot.smoke[0].position.x) &&
                  std::isfinite(snapshot.smoke[0].velocity.x) &&
                  std::isfinite(snapshot.rigid.position[0]) &&
                  std::isfinite(snapshot.rigid.linearVelocity[0]));
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
                    const Snapshot &endpoints, bool grid) {
    const std::string prefix = grid ? "grid" : "particle";
    Json resources = Json::array();
    resources.push_back(resource_json(
        prefix + "-smoke-rigid-contact-graph", coupling != nullptr,
        graph_generation, graph_fingerprint(coupling, endpoints)));
    resources.push_back(resource_json(
        prefix + "-smoke-rigid-endpoints", true, 1U,
        endpoint_fingerprint(endpoints)));
    Json checkpoint = Json::object();
    checkpoint["id"] = std::move(id);
    checkpoint["outcome"] = std::move(outcome);
    checkpoint["probe"] = snapshot_json(endpoints);
    checkpoint["resources"] = std::move(resources);
    observations.push_back(std::move(checkpoint));
}

struct LifecycleRun {
    DriverInfo driver{};
    Json observations{};
};

[[nodiscard]] LifecycleRun run_lifecycle_fixture(bool grid) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: smoke and rigid features");
    require_contact_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    initialize_endpoints(context, fixture, -0.034F, {1.0F, 0.0F, 0.0F});
    const PmCouplingState baseline = contact_coupling(grid, contact_distance);
    const PmCouplingState replacement = contact_coupling(
        grid, contact_distance + 0.001F);
    const PmCouplingState recovery = contact_coupling(
        grid, contact_distance + 0.002F);
    const std::array<PmCouplingState, 2U> overflow{{replacement, recovery}};
    const PmStepInfo warmup{
        timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    const PmStepInfo probe_step{
        1.0e-3F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    for (std::uint32_t index = 0U; index < warmup_steps; ++index) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &warmup) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }
    const auto submit_graph = [&](const PmCouplingState *couplings,
                                  std::uint32_t count,
                                  PmResult record_result) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, count,
                                  couplings) == record_result);
        if (record_result == PM_SUCCESS)
            PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                        &probe_step) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    };
    const auto run_probe = [&]() {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &probe_step) == PM_SUCCESS);
        record_copies(context, fixture);
        const PmResult status = context.submit();
        if (status != PM_SUCCESS)
            throw std::runtime_error(
                "smoke-rigid lifecycle probe failed: " +
                std::to_string(status));
        Snapshot snapshot = read_snapshot(context, fixture);
        if (snapshot.smoke.size() != 1U)
            throw std::runtime_error(
                "smoke-rigid lifecycle probe lost its tracer");
        return snapshot;
    };
    Json observations = Json::array();
    submit_graph(&baseline, 1U, PM_SUCCESS);
    add_checkpoint(observations, "baseline", "PM_SUCCESS", &baseline, 1U,
                   run_probe(), grid);
    submit_graph(&replacement, 1U, PM_SUCCESS);
    add_checkpoint(observations, "replacement", "PM_SUCCESS", &replacement,
                   2U, run_probe(), grid);
    submit_graph(overflow.data(),
                 static_cast<std::uint32_t>(overflow.size()),
                 PM_ERROR_CAPACITY_EXCEEDED);
    add_checkpoint(observations, "rejected-overflow",
                   "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U,
                   run_probe(), grid);
    submit_graph(nullptr, 0U, PM_SUCCESS);
    add_checkpoint(observations, "removal", "PM_SUCCESS", nullptr, 3U,
                   run_probe(), grid);
    submit_graph(&recovery, 1U, PM_SUCCESS);
    add_checkpoint(observations, "recovery", "PM_SUCCESS", &recovery, 4U,
                   run_probe(), grid);
    DriverInfo driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    pmDestroyGeometry(context.device, fixture.geometry);
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}
} // namespace

std::string conformance_smoke_rigid_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-rigid-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/smoke-rigid-contact-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json fixtures = Json::array();
        fixtures.push_back(quality_fixture_json(false, particle_driver));
        fixtures.push_back(quality_fixture_json(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "smoke-rigid representations selected different devices");
        result["execution_status"] = "completed";
        result["driver"] = driver_json(particle_driver);
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

std::string conformance_smoke_rigid_contact_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-rigid-contact-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/smoke-rigid-contact-lifecycle-v1.json");
    try {
        LifecycleRun particle = run_lifecycle_fixture(false);
        LifecycleRun grid = run_lifecycle_fixture(true);
        if (!same_driver(particle.driver, grid.driver))
            throw std::runtime_error(
                "smoke-rigid lifecycle representations selected different devices");
        Json cases = Json::array();
        const auto add_case = [&](const char *representation,
                                  const char *prefix, Json observations) {
            Json families = Json::array();
            families.push_back("smoke");
            families.push_back("rigid_body");
            Json unrelated = Json::array();
            unrelated.push_back(std::string(prefix) +
                                "-smoke-rigid-endpoints");
            Json attempt = Json::object();
            attempt["fixture"] = "smoke-rigid-contact-public-lifecycle";
            attempt["representation"] = representation;
            attempt["status"] = "completed";
            Json item = Json::object();
            item["attempt"] = std::move(attempt);
            item["execution_status"] = "completed";
            item["families"] = std::move(families);
            item["id"] = std::string(prefix) +
                "-smoke-rigid-contact-capacity-replacement-removal-rollback";
            item["observations"] = std::move(observations);
            item["unrelated_resources"] = std::move(unrelated);
            cases.push_back(std::move(item));
        };
        add_case("SMOKE_PARTICLE", "particle",
                 std::move(particle.observations));
        add_case("SMOKE_GRID", "grid", std::move(grid.observations));
        result["execution_status"] = "completed";
        result["driver"] = driver_json(particle.driver);
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
