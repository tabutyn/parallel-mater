// SPDX-License-Identifier: MIT
// Public-SDK bidirectional smoke/rigid drag quality and graph lifecycle fixtures.
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
#include <vector>

namespace parallel_mater::conformance {
namespace {
using parallel_mater::test::Buffer;
using parallel_mater::test::Context;

constexpr float timestep = 1.0F / 60.0F;
constexpr std::uint32_t warmup_steps = 5U;
[[nodiscard]] constexpr std::uint32_t drag_steps(bool grid) noexcept {
    // Grid tracers observe the projected Eulerian response on the next tick.
    return grid ? 2U : 1U;
}
constexpr std::uint32_t smoke_capacity = 64U;
constexpr std::uint32_t grid_resolution = 16U;
constexpr std::uint32_t grid_vertical_resolution = 8U;
constexpr std::uint32_t grid_cells =
    grid_resolution * grid_vertical_resolution * grid_resolution;
constexpr float carrier_speed = 1.0F;
constexpr float emission_rate = 50.0F;
constexpr float contact_distance = 0.01F;
constexpr float control_air_density = 1.0F;
constexpr float test_air_density = 1.0F;
constexpr float control_drag_coefficient = 0.0F;
constexpr float test_drag_coefficient = 20.0F;
constexpr float particle_radius = 0.1F;
constexpr float effective_particle_mass =
    test_air_density * (2.0F * particle_radius) *
    (2.0F * particle_radius) * (2.0F * particle_radius);
constexpr float cube_half_extent = 0.1F;
constexpr float rigid_mass = 1.0F;
constexpr float cube_inertia =
    (1.0F / 6.0F) * rigid_mass * (2.0F * cube_half_extent) *
    (2.0F * cube_half_extent);
constexpr float rigid_collision_margin = 0.005F;
constexpr PmFloat3 particle_rigid_position{0.25F, 0.25F, 0.34F};
constexpr PmFloat3 grid_rigid_position{0.25F, 0.25F, 0.265F};

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

struct DriverInfo {
    std::string name{};
    std::string device{};
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t version{};
};

struct Snapshot {
    PmRigidState rigid{};
    std::vector<PmSmokeParticleState> smoke{};
};

struct FixtureObjects {
    PmGeometry geometry{};
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange rigid_output{};
    PmBufferRange smoke_output{};
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
        throw std::runtime_error("query smoke-rigid drag backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown smoke-rigid drag backend");
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

void require_drag_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_SMOKE, PM_INTERACTION_FAMILY_RIGID_BODY,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_DRAG,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(std::string("unsupported: smoke-rigid drag ") +
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
    material.maximumSpeed = grid ? 3.7F : 4.0F;
    material.gridResolution = grid ? grid_resolution : 0U;
    material.gridVerticalResolution = grid ? grid_vertical_resolution : 0U;
    material.gridPressureIterations = grid ? 12U : 0U;
    material.gridKinematicViscosity = grid ? 0.1F : 0.0F;
    material.gridLesCoefficient = 0.0F;
    material.gridPressureTolerance = 1.0e-4F;
    return material;
}

[[nodiscard]] PmSmokeInfo smoke_info(float velocity,
                                     float particles_per_second) {
    PmSmokeInfo info{};
    info.active = PM_TRUE;
    info.materialIndex = 0U;
    info.particleCapacity = smoke_capacity;
    info.emitterCenter = {0.25F, 0.25F, 0.46F};
    info.emitterHalfExtents = {0.04F, 0.001F};
    info.initialVelocity = {velocity, 0.0F, 0.0F};
    info.wind = {velocity, 0.0F, 0.0F};
    info.particlesPerSecond = particles_per_second;
    info.lifetime = 10.0F;
    info.gridMinimum = {0.0F, 0.0F, 0.0F};
    info.gridEdgeLength = 1.0F;
    return info;
}

[[nodiscard]] PmRigidBodyInfo rigid_body(PmGeometry geometry, bool grid) {
    PmRigidBodyInfo body{};
    body.active = PM_TRUE;
    body.motionType = PM_MOTION_DYNAMIC;
    body.geometry = geometry;
    const PmFloat3 position = grid ? grid_rigid_position
                                   : particle_rigid_position;
    body.state.position[0] = position.x;
    body.state.position[1] = position.y;
    body.state.position[2] = position.z;
    body.state.orientation[3] = 1.0F;
    body.mass = rigid_mass;
    body.inertiaDiagonal[0] = cube_inertia;
    body.inertiaDiagonal[1] = cube_inertia;
    body.inertiaDiagonal[2] = cube_inertia;
    body.friction = 0.0F;
    body.restitution = 0.0F;
    body.linearDamping = 0.0F;
    body.angularDamping = 0.0F;
    body.maximumLinearSpeed = 4.0F;
    body.maximumAngularSpeed = 4.0F;
    body.collisionMargin = rigid_collision_margin;
    return body;
}

[[nodiscard]] PmCouplingState drag_edge(bool grid, float air_density,
                                        float drag_coefficient) {
    PmCouplingState edge{};
    edge.type = PM_COUPLING_SMOKE_RIGID;
    edge.first = 0U;
    edge.second = 0U;
    edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_DRAG_BIT;
    edge.materialProfile = grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
                                : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    edge.parameters.smoke.windDrag = 0.0F;
    edge.parameters.smoke.maximumWindAcceleration = 0.0F;
    edge.parameters.smoke.airDensity = air_density;
    edge.parameters.smoke.dragCoefficient = drag_coefficient;
    edge.parameters.smoke.contactDistance = contact_distance;
    // The public switch enables near-surface particle exchange; authored
    // clearance keeps every tracer outside the contact distance.
    edge.parameters.smoke.tracerContact = PM_TRUE;
    return edge;
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
    const PmSmokeCapacity smoke_domain{
        smoke_capacity, grid ? grid_cells : 1U};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.rigidBodyCapacity = 1U;
    create.smokeCount = 1U;
    create.pSmokeCapacities = &smoke_domain;
    create.couplingCapacity = coupling_capacity;
    FixtureObjects fixture{};
    fixture.geometry = create_geometry(context);
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

    const PmSmokeMaterialState smoke = smoke_material(grid);
    const PmRigidSolverState rigid{4U, PM_FALSE, 0.0F, 0.0F, 0.0F};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid;
    pipeline_info.smokeMaterialCount = 1U;
    pipeline_info.pSmokeMaterials = &smoke;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize rigid_bytes = sizeof(PmRigidState);
    constexpr PmDeviceSize smoke_bytes = sizeof(PmDomainOutputHeader) +
        smoke_capacity * sizeof(PmSmokeParticleState);
    fixture.output = context.buffer(
        rigid_bytes + smoke_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.smoke_output = {fixture.output.handle, 0U, smoke_bytes};
    fixture.rigid_output = {fixture.output.handle, smoke_bytes, rigid_bytes};
    return fixture;
}

void destroy_fixture(Context &context, const FixtureObjects &fixture) {
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    pmDestroyGeometry(context.device, fixture.geometry);
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture,
                          bool grid, float smoke_velocity,
                          float particles_per_second) {
    const PmRigidBodyInfo body = rigid_body(fixture.geometry, grid);
    const PmSmokeInfo smoke = smoke_info(smoke_velocity,
                                         particles_per_second);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteRigidBodies(context.commands, fixture.simulation, 0U,
                                  1U, &body) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopyRigidStates(context.commands, fixture.simulation, 0U, 1U,
                                 &fixture.rigid_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySmokeState(context.commands, fixture.simulation, 0U,
                                &fixture.smoke_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    Snapshot result{};
    result.rigid = *reinterpret_cast<const PmRigidState *>(
        base + fixture.rigid_output.offset);
    const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.smoke_output.offset);
    if (header->activeCount > smoke_capacity)
        throw std::runtime_error("smoke-rigid public output count mismatch");
    const auto *smoke = reinterpret_cast<const PmSmokeParticleState *>(
        header + 1);
    result.smoke.assign(smoke, smoke + header->activeCount);
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

[[nodiscard]] QualityRun run_quality_fixture(bool grid, float air_density,
                                             float drag_coefficient,
                                             float smoke_velocity) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: rigid and smoke features");
    require_drag_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    initialize_endpoints(context, fixture, grid, smoke_velocity, emission_rate);
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    for (std::uint32_t index = 0U; index < warmup_steps; ++index) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        const PmResult submitted = context.submit();
        if (submitted != PM_SUCCESS)
            throw std::runtime_error("smoke-rigid warmup failed: " +
                                     std::to_string(submitted));
    }
    Snapshot initial = capture(context, fixture);
    if (initial.smoke.empty())
        throw std::runtime_error("smoke-rigid carrier did not emit");

    const PmCouplingState edge = drag_edge(
        grid, air_density, drag_coefficient);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &edge) == PM_SUCCESS);
    for (std::uint32_t index = 0U; index < drag_steps(grid); ++index)
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
    record_copies(context, fixture);
    const PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("smoke-rigid drag submission failed: " +
                                 std::to_string(submitted));
    Snapshot final = read_snapshot(context, fixture);
    DriverInfo driver = driver_info(context);
    destroy_fixture(context, fixture);
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

[[nodiscard]] Json rigid_state_json(const PmRigidState &state) {
    Json result = Json::object();
    result["position_m"] = vector_json(state.position);
    result["orientation_xyzw"] = quaternion_json(state.orientation);
    result["linear_velocity_m_s"] = vector_json(state.linearVelocity);
    result["angular_velocity_rad_s"] = vector_json(state.angularVelocity);
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
    Json result = Json::object();
    result["rigid"] = rigid_state_json(snapshot.rigid);
    result["smoke_particles"] = std::move(smoke);
    return result;
}

[[nodiscard]] Json run_json(const char *id, const QualityRun &run) {
    Json result = Json::object();
    result["id"] = id;
    result["initial"] = snapshot_json(run.initial);
    result["final"] = snapshot_json(run.final);
    return result;
}

[[nodiscard]] Json quality_fixture_json(bool grid, DriverInfo &driver) {
    QualityRun forward_control = run_quality_fixture(
        grid, control_air_density, control_drag_coefficient, carrier_speed);
    QualityRun forward = run_quality_fixture(
        grid, test_air_density, test_drag_coefficient, carrier_speed);
    QualityRun stationary_control = run_quality_fixture(
        grid, control_air_density, control_drag_coefficient, 0.0F);
    QualityRun stationary = run_quality_fixture(
        grid, test_air_density, test_drag_coefficient, 0.0F);
    QualityRun reverse_control = run_quality_fixture(
        grid, control_air_density, control_drag_coefficient, -carrier_speed);
    QualityRun reverse = run_quality_fixture(
        grid, test_air_density, test_drag_coefficient, -carrier_speed);
    for (const auto *run : {&forward, &stationary_control, &stationary,
                            &reverse_control, &reverse})
        if (!same_driver(forward_control.driver, run->driver))
            throw std::runtime_error(
                "paired smoke-rigid drag runs selected different devices");
    driver = forward_control.driver;

    Json modes = Json::array();
    modes.push_back("CONTACT");
    modes.push_back("DRAG");
    Json result = Json::object();
    result["id"] = grid ? "grid-smoke-rigid-drag"
                         : "particle-smoke-rigid-drag";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["warmup_step_count"] = static_cast<std::uint64_t>(warmup_steps);
    result["drag_step_count"] =
        static_cast<std::uint64_t>(drag_steps(grid));
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["rigid_geometry_vertices_m"] = vectors_json(cube_vertices);
    result["rigid_geometry_triangle_indices"] = numbers_json(cube_indices);
    result["rigid_initial_position_m"] = vector_json(
        grid ? grid_rigid_position : particle_rigid_position);
    constexpr std::array<float, 4U> identity{{0.0F, 0.0F, 0.0F, 1.0F}};
    result["rigid_initial_orientation_xyzw"] =
        quaternion_json(identity.data());
    result["rigid_mass_kg"] = static_cast<double>(rigid_mass);
    result["rigid_inertia_diagonal_kg_m2"] = vector_json(
        PmFloat3{cube_inertia, cube_inertia, cube_inertia});
    result["rigid_friction"] = 0.0;
    result["rigid_restitution"] = 0.0;
    result["rigid_linear_damping_per_s"] = 0.0;
    result["rigid_angular_damping_per_s"] = 0.0;
    result["rigid_maximum_linear_speed_m_s"] = 4.0;
    result["rigid_maximum_angular_speed_rad_s"] = 4.0;
    result["rigid_collision_margin_m"] =
        static_cast<double>(rigid_collision_margin);
    result["smoke_particle_capacity"] =
        static_cast<std::uint64_t>(smoke_capacity);
    result["smoke_emitter_center_m"] =
        vector_json(PmFloat3{0.25F, 0.25F, 0.46F});
    Json half_extents = Json::array();
    half_extents.push_back(0.04);
    half_extents.push_back(0.001);
    result["smoke_emitter_half_extents_m"] = std::move(half_extents);
    result["smoke_emission_rate_per_s"] = static_cast<double>(emission_rate);
    result["smoke_lifetime_seconds"] = 10.0;
    result["smoke_particle_radius_m"] = static_cast<double>(particle_radius);
    result["effective_smoke_particle_mass_kg"] =
        static_cast<double>(effective_particle_mass);
    result["smoke_buoyancy_m_s2"] = 0.0;
    result["smoke_response_per_s"] = 0.0;
    result["smoke_rest_number_density"] = 1.0;
    result["smoke_pressure_stiffness_m2_s2"] = 0.0;
    result["smoke_viscosity_m2_s"] = 0.0;
    result["smoke_vorticity_confinement"] = 0.0;
    result["smoke_maximum_speed_m_s"] = grid ? 3.7 : 4.0;
    result["smoke_grid_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_resolution : 0U);
    result["smoke_grid_vertical_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_vertical_resolution : 0U);
    result["smoke_grid_pressure_iterations"] =
        static_cast<std::uint64_t>(grid ? 12U : 0U);
    result["smoke_grid_kinematic_viscosity_m2_s"] = grid ? 0.1 : 0.0;
    result["smoke_grid_les_coefficient"] = 0.0;
    result["smoke_grid_pressure_tolerance"] = 0.0001;
    result["smoke_grid_minimum_m"] = vector_json(PmFloat3{});
    result["smoke_grid_edge_length_m"] = 1.0;
    result["declared_mode_flags"] = std::move(modes);
    result["exercised_mode"] = "DRAG";
    result["contact_distance_m"] = static_cast<double>(contact_distance);
    result["tracer_contact"] = true;
    result["forward_carrier_velocity_m_s"] =
        vector_json(PmFloat3{carrier_speed, 0.0F, 0.0F});
    result["stationary_carrier_velocity_m_s"] = vector_json(PmFloat3{});
    result["reverse_carrier_velocity_m_s"] =
        vector_json(PmFloat3{-carrier_speed, 0.0F, 0.0F});
    result["control_air_density_kg_m3"] =
        static_cast<double>(control_air_density);
    result["test_air_density_kg_m3"] =
        static_cast<double>(test_air_density);
    result["control_drag_coefficient"] =
        static_cast<double>(control_drag_coefficient);
    result["test_drag_coefficient"] =
        static_cast<double>(test_drag_coefficient);
    result["oracle"] =
        "paired-public-noncontact-bidirectional-smoke-rigid-drag";
    Json runs = Json::array();
    runs.push_back(run_json("forward-control", forward_control));
    runs.push_back(run_json("forward", forward));
    runs.push_back(run_json("stationary-control", stationary_control));
    runs.push_back(run_json("stationary", stationary));
    runs.push_back(run_json("reverse-control", reverse_control));
    runs.push_back(run_json("reverse", reverse));
    result["runs"] = std::move(runs);
    return result;
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

void append_vector(std::ostringstream &stream, const float value[3]) {
    append_float(stream, value[0]);
    stream << ',';
    append_float(stream, value[1]);
    stream << ',';
    append_float(stream, value[2]);
}

[[nodiscard]] std::string graph_fingerprint(const PmCouplingState *edge) {
    if (edge == nullptr)
        return sha256("parallel-mater-smoke-rigid-drag-graph/v1;inactive");
    const auto &smoke = edge->parameters.smoke;
    std::ostringstream canonical;
    canonical << "parallel-mater-smoke-rigid-drag-graph/v1;type="
              << static_cast<unsigned>(edge->type)
              << ";first=" << edge->first << ";second=" << edge->second
              << ";modes=" << edge->modeFlags
              << ";direction=BIDIRECTIONAL;material="
              << static_cast<unsigned>(edge->materialProfile)
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
    canonical << "parallel-mater-smoke-rigid-drag-endpoints/v1;position=";
    append_vector(canonical, snapshot.rigid.position);
    canonical << ";linear-velocity=";
    append_vector(canonical, snapshot.rigid.linearVelocity);
    canonical << ";angular-velocity=";
    append_vector(canonical, snapshot.rigid.angularVelocity);
    canonical << ";smoke-count=" << snapshot.smoke.size();
    for (const auto &state : snapshot.smoke) {
        canonical << ";smoke-position=";
        append_float(canonical, state.position.x);
        canonical << ',';
        append_float(canonical, state.position.y);
        canonical << ',';
        append_float(canonical, state.position.z);
        canonical << ",velocity=";
        append_float(canonical, state.velocity.x);
        canonical << ',';
        append_float(canonical, state.velocity.y);
        canonical << ',';
        append_float(canonical, state.velocity.z);
        canonical << ",age=";
        append_float(canonical, state.age);
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

void add_checkpoint(Json &observations, const std::string &prefix,
                    std::string id, std::string outcome,
                    const PmCouplingState *edge,
                    std::uint64_t graph_generation,
                    const Snapshot &endpoints) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        prefix + "-smoke-rigid-drag-graph", edge != nullptr,
        graph_generation, graph_fingerprint(edge)));
    resources.push_back(resource_json(
        prefix + "-smoke-rigid-endpoints", true, 1U,
        endpoint_fingerprint(endpoints)));
    Json checkpoint = Json::object();
    checkpoint["id"] = std::move(id);
    checkpoint["outcome"] = std::move(outcome);
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
        throw std::runtime_error("unsupported: rigid and smoke features");
    require_drag_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    initialize_endpoints(context, fixture, grid, 0.0F, 1.0F);
    const PmCouplingState baseline = drag_edge(grid, 1.0F, 10.0F);
    const PmCouplingState replacement = drag_edge(grid, 1.0F, 20.0F);
    const PmCouplingState recovery = drag_edge(grid, 1.0F, 30.0F);
    const std::array<PmCouplingState, 2U> overflow{{replacement, recovery}};
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    Json observations = Json::array();
    const std::string prefix = grid ? "grid" : "particle";
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
    add_checkpoint(observations, prefix, "baseline", "PM_SUCCESS", &baseline,
                   1U, mutate(&baseline, 1U, PM_SUCCESS));
    add_checkpoint(observations, prefix, "replacement", "PM_SUCCESS",
                   &replacement, 2U, mutate(&replacement, 1U, PM_SUCCESS));
    add_checkpoint(observations, prefix, "rejected-overflow",
                   "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U,
                   mutate(overflow.data(),
                          static_cast<std::uint32_t>(overflow.size()),
                          PM_ERROR_CAPACITY_EXCEEDED));
    add_checkpoint(observations, prefix, "removal", "PM_SUCCESS", nullptr, 3U,
                   mutate(nullptr, 0U, PM_SUCCESS));
    add_checkpoint(observations, prefix, "recovery", "PM_SUCCESS", &recovery,
                   4U, mutate(&recovery, 1U, PM_SUCCESS));
    DriverInfo driver = driver_info(context);
    destroy_fixture(context, fixture);
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] Json lifecycle_case_json(bool grid, LifecycleRun run) {
    const std::string prefix = grid ? "grid" : "particle";
    Json families = Json::array();
    families.push_back("smoke");
    families.push_back("rigid-body");
    Json unrelated = Json::array();
    unrelated.push_back(prefix + "-smoke-rigid-endpoints");
    Json attempt = Json::object();
    attempt["fixture"] = "smoke-rigid-drag-public-lifecycle";
    attempt["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    attempt["status"] = "completed";
    Json item = Json::object();
    item["attempt"] = std::move(attempt);
    item["execution_status"] = "completed";
    item["families"] = std::move(families);
    item["id"] = prefix +
        "-smoke-rigid-drag-capacity-replacement-removal-rollback";
    item["observations"] = std::move(run.observations);
    item["unrelated_resources"] = std::move(unrelated);
    return item;
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}

} // namespace

std::string conformance_smoke_rigid_drag_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-rigid-drag-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/smoke-rigid-drag-v1.json");
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

std::string conformance_smoke_rigid_drag_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-rigid-drag-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/smoke-rigid-drag-lifecycle-v1.json");
    try {
        LifecycleRun particle = run_lifecycle_fixture(false);
        LifecycleRun grid = run_lifecycle_fixture(true);
        if (!same_driver(particle.driver, grid.driver))
            throw std::runtime_error(
                "smoke-rigid lifecycle representations selected different devices");
        const DriverInfo driver = particle.driver;
        Json cases = Json::array();
        cases.push_back(lifecycle_case_json(false, std::move(particle)));
        cases.push_back(lifecycle_case_json(true, std::move(grid)));
        result["execution_status"] = "completed";
        result["driver"] = driver_json(driver);
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
