// SPDX-License-Identifier: MIT
// Public-SDK smoke-to-soft drag quality and graph lifecycle fixtures.
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

constexpr float timestep = 1.0F / 60.0F;
constexpr std::uint32_t warmup_steps = 6U;
constexpr std::uint32_t smoke_capacity = 64U;
constexpr std::uint32_t grid_resolution = 16U;
constexpr std::uint32_t grid_vertical_resolution = 8U;
constexpr std::uint32_t grid_cells =
    grid_resolution * grid_vertical_resolution * grid_resolution;
constexpr float carrier_speed = 1.0F;
constexpr float test_drag = 20.0F;
constexpr float maximum_wind_acceleration = 10.0F;
constexpr float contact_distance = 0.01F;
constexpr float soft_surface_z = 0.35F;
constexpr float soft_node_mass = 0.25F;
constexpr float soft_node_radius = 0.005F;
constexpr std::array<PmFloat3, 4U> soft_nodes{{
    {0.15F, 0.15F, soft_surface_z},
    {0.35F, 0.15F, soft_surface_z},
    {0.25F, 0.35F, soft_surface_z},
    {0.25F, 0.216666667F, 0.25F},
}};
constexpr std::array<PmSoftBodyBond, 6U> soft_bonds{{
    {0U, 1U, 0.2F},
    {0U, 2U, 0.223606798F},
    {0U, 3U, 0.156347192F},
    {1U, 2U, 0.223606798F},
    {1U, 3U, 0.156347192F},
    {2U, 3U, 0.166666667F},
}};
constexpr std::array<PmFloat3, 3U> surface_vertices{{
    soft_nodes[0], soft_nodes[1], soft_nodes[2],
}};
constexpr std::array<std::uint32_t, 3U> surface_indices{{0U, 1U, 2U}};
constexpr std::array<float, soft_nodes.size()> soft_inverse_masses{{
    1.0F / soft_node_mass,
    1.0F / soft_node_mass,
    1.0F / soft_node_mass,
    1.0F / soft_node_mass,
}};

[[nodiscard]] std::array<PmSoftBodySurfaceBinding, surface_vertices.size()>
surface_bindings() {
    std::array<PmSoftBodySurfaceBinding, surface_vertices.size()> result{};
    for (std::uint32_t index = 0U; index < result.size(); ++index) {
        result[index].nodes[0] = index;
        result[index].weights[0] = 1.0F;
    }
    return result;
}

struct DriverInfo {
    std::string name{};
    std::string device{};
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t version{};
};

struct Snapshot {
    std::array<PmSoftBodyNodeState, soft_nodes.size()> soft{};
    std::array<PmFloat3, surface_vertices.size()> surface{};
    std::vector<PmSmokeParticleState> smoke{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange soft_output{};
    PmBufferRange surface_output{};
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
        throw std::runtime_error("query smoke-soft drag backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown smoke-soft drag backend");
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
    needed.softBodies = PM_TRUE;
    needed.smoke = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

void require_drag_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_SOFT_BODY, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_DRAG,
        PM_INTERACTION_DIRECTION_SECOND_TO_FIRST,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(std::string("unsupported: smoke-soft drag ") +
                                 (grid ? "SMOKE_GRID" : "SMOKE_PARTICLE"));
}

[[nodiscard]] PmSoftBodyMaterialState soft_material() {
    PmSoftBodyMaterialState material{};
    material.nodeMass = soft_node_mass;
    material.nodeRadius = soft_node_radius;
    material.stretchCompliance = 0.0F;
    material.velocityDamping = 0.0F;
    material.springDamping = 0.0F;
    material.contactFriction = 0.0F;
    material.shapeMatchingStiffness = 0.0F;
    material.maximumProjectionFraction = 0.2F;
    material.constraintVelocityResponse = 0.0F;
    material.maximumSpeed = 3.4F;
    material.solverIterations = 1U;
    return material;
}

[[nodiscard]] PmSmokeMaterialState smoke_material(bool grid) {
    PmSmokeMaterialState material{};
    material.particleRadius = 0.10F;
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
    info.emitterCenter = {0.25F, 0.25F, 0.50F};
    info.emitterHalfExtents = {0.05F, 0.05F};
    info.initialVelocity = {velocity, 0.0F, 0.0F};
    info.wind = {velocity, 0.0F, 0.0F};
    info.particlesPerSecond = particles_per_second;
    info.lifetime = 10.0F;
    info.gridMinimum = {0.0F, 0.0F, 0.0F};
    info.gridEdgeLength = 1.0F;
    return info;
}

[[nodiscard]] PmCouplingState drag_edge(bool grid, float wind_drag) {
    PmCouplingState edge{};
    edge.type = PM_COUPLING_SMOKE_SOFT_BODY;
    edge.first = 0U;
    edge.second = 0U;
    edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_DRAG_BIT;
    edge.materialProfile = grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
                                : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    edge.parameters.smoke.windDrag = wind_drag;
    edge.parameters.smoke.maximumWindAcceleration =
        maximum_wind_acceleration;
    edge.parameters.smoke.airDensity = 1.0F;
    edge.parameters.smoke.dragCoefficient = 1.0F;
    edge.parameters.smoke.contactDistance = contact_distance;
    edge.parameters.smoke.tracerContact = PM_FALSE;
    return edge;
}

[[nodiscard]] FixtureObjects create_fixture(Context &context, bool grid,
                                            std::uint32_t coupling_capacity) {
    constexpr PmSoftBodyCapacity soft_capacity{
        static_cast<std::uint32_t>(soft_nodes.size()),
        static_cast<std::uint32_t>(soft_bonds.size()),
        static_cast<std::uint32_t>(surface_vertices.size()),
        static_cast<std::uint32_t>(surface_indices.size())};
    const PmSmokeCapacity smoke_domain{
        smoke_capacity, grid ? grid_cells : 1U};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.softBodyCount = 1U;
    create.pSoftBodyCapacities = &soft_capacity;
    create.smokeCount = 1U;
    create.pSmokeCapacities = &smoke_domain;
    create.couplingCapacity = coupling_capacity;
    FixtureObjects fixture{};
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
    const PmSmokeMaterialState smoke = smoke_material(grid);
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.softBodyMaterialCount = 1U;
    pipeline_info.pSoftBodyMaterials = &soft;
    pipeline_info.smokeMaterialCount = 1U;
    pipeline_info.pSmokeMaterials = &smoke;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize soft_bytes = sizeof(PmDomainOutputHeader) +
        soft_nodes.size() * sizeof(PmSoftBodyNodeState);
    constexpr PmDeviceSize surface_bytes = sizeof(PmDomainOutputHeader) +
        surface_vertices.size() * sizeof(PmFloat3);
    constexpr PmDeviceSize smoke_bytes = sizeof(PmDomainOutputHeader) +
        smoke_capacity * sizeof(PmSmokeParticleState);
    fixture.output = context.buffer(
        smoke_bytes + soft_bytes + surface_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.smoke_output = {fixture.output.handle, 0U, smoke_bytes};
    fixture.soft_output = {fixture.output.handle, smoke_bytes, soft_bytes};
    fixture.surface_output = {
        fixture.output.handle, smoke_bytes + soft_bytes, surface_bytes};
    return fixture;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture,
                          float smoke_velocity,
                          float particles_per_second) {
    const auto bindings = surface_bindings();
    const PmSoftBodyInfo soft{
        PM_TRUE, 0U, static_cast<std::uint32_t>(soft_nodes.size()),
        soft_nodes.data(), static_cast<std::uint32_t>(soft_bonds.size()),
        soft_bonds.data(), soft_inverse_masses.data(),
        static_cast<std::uint32_t>(surface_vertices.size()),
        surface_vertices.data(),
        static_cast<std::uint32_t>(surface_indices.size()),
        surface_indices.data(), bindings.data()};
    const PmSmokeInfo smoke = smoke_info(smoke_velocity,
                                         particles_per_second);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSoftBodies(context.commands, fixture.simulation, 0U, 1U,
                                 &soft) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopySoftBodyState(context.commands, fixture.simulation, 0U,
                                &fixture.soft_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodySurface(context.commands, fixture.simulation, 0U,
                                     &fixture.surface_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySmokeState(context.commands, fixture.simulation, 0U,
                                &fixture.smoke_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *soft_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.soft_output.offset);
    const auto *surface_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.surface_output.offset);
    const auto *smoke_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.smoke_output.offset);
    if (soft_header->activeCount != soft_nodes.size() ||
        surface_header->activeCount != surface_vertices.size() ||
        smoke_header->activeCount > smoke_capacity)
        throw std::runtime_error("smoke-soft public output count mismatch");
    Snapshot result{};
    const auto *soft = reinterpret_cast<const PmSoftBodyNodeState *>(
        soft_header + 1);
    std::copy_n(soft, result.soft.size(), result.soft.begin());
    const auto *surface = reinterpret_cast<const PmFloat3 *>(
        surface_header + 1);
    std::copy_n(surface, result.surface.size(), result.surface.begin());
    const auto *smoke = reinterpret_cast<const PmSmokeParticleState *>(
        smoke_header + 1);
    result.smoke.assign(smoke, smoke + smoke_header->activeCount);
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

[[nodiscard]] QualityRun run_quality_fixture(bool grid, float wind_drag,
                                             float smoke_velocity) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: soft and smoke features");
    require_drag_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    initialize_endpoints(context, fixture, smoke_velocity, 1'000.0F);
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    for (std::uint32_t index = 0U; index < warmup_steps; ++index) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        const PmResult submitted = context.submit();
        if (submitted != PM_SUCCESS)
            throw std::runtime_error("smoke-soft warmup failed: " +
                                     std::to_string(submitted));
    }
    Snapshot initial = capture(context, fixture);
    if (initial.smoke.empty())
        throw std::runtime_error("smoke-soft carrier did not emit");

    const PmCouplingState edge = drag_edge(grid, wind_drag);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &edge) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation, &step) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    const PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("smoke-soft drag submission failed: " +
                                 std::to_string(submitted));
    Snapshot final = read_snapshot(context, fixture);
    DriverInfo driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(initial), std::move(final)};
}

[[nodiscard]] Json vector_json(PmFloat3 value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    return result;
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot) {
    Json soft = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.soft.size(); ++index) {
        Json item = Json::object();
        item["index"] = static_cast<std::uint64_t>(index);
        item["position_m"] = vector_json(snapshot.soft[index].position);
        item["velocity_m_s"] = vector_json(snapshot.soft[index].velocity);
        soft.push_back(std::move(item));
    }
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
    result["soft_body"] = std::move(soft);
    Json surface = Json::array();
    for (const auto vertex : snapshot.surface)
        surface.push_back(vector_json(vertex));
    result["soft_body_surface_vertices_m"] = std::move(surface);
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
        grid, 0.0F, carrier_speed);
    QualityRun forward = run_quality_fixture(grid, test_drag, carrier_speed);
    QualityRun stationary_control = run_quality_fixture(grid, 0.0F, 0.0F);
    QualityRun stationary = run_quality_fixture(grid, test_drag, 0.0F);
    QualityRun reverse_control = run_quality_fixture(
        grid, 0.0F, -carrier_speed);
    QualityRun reverse = run_quality_fixture(grid, test_drag, -carrier_speed);
    for (const auto *run : {&forward, &stationary_control, &stationary,
                            &reverse_control, &reverse})
        if (!same_driver(forward_control.driver, run->driver))
            throw std::runtime_error(
                "paired smoke-soft drag runs selected different devices");
    driver = forward_control.driver;
    Json mode_flags = Json::array();
    mode_flags.push_back("CONTACT");
    mode_flags.push_back("DRAG");
    Json result = Json::object();
    result["id"] = grid ? "grid-smoke-soft-drag"
                         : "particle-smoke-soft-drag";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["warmup_step_count"] = static_cast<std::uint64_t>(warmup_steps);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    Json vertices = Json::array();
    for (const auto vertex : soft_nodes)
        vertices.push_back(vector_json(vertex));
    result["soft_body_nodes_m"] = std::move(vertices);
    Json bonds = Json::array();
    for (const auto &bond : soft_bonds) {
        Json item = Json::array();
        item.push_back(static_cast<std::uint64_t>(bond.first));
        item.push_back(static_cast<std::uint64_t>(bond.second));
        item.push_back(static_cast<double>(bond.restLength));
        bonds.push_back(std::move(item));
    }
    result["soft_body_bonds"] = std::move(bonds);
    Json inverse_masses = Json::array();
    for (const auto value : soft_inverse_masses)
        inverse_masses.push_back(static_cast<double>(value));
    result["soft_body_inverse_masses_kg_inverse"] =
        std::move(inverse_masses);
    Json surface_vertices_json = Json::array();
    for (const auto vertex : surface_vertices)
        surface_vertices_json.push_back(vector_json(vertex));
    result["soft_body_surface_vertices_m"] =
        std::move(surface_vertices_json);
    Json indices = Json::array();
    for (const auto index : surface_indices)
        indices.push_back(static_cast<std::uint64_t>(index));
    result["soft_body_surface_triangle_indices"] = std::move(indices);
    Json bindings_json = Json::array();
    for (const auto &binding : surface_bindings()) {
        Json nodes = Json::array();
        Json weights = Json::array();
        for (std::uint32_t index = 0U; index < 4U; ++index) {
            nodes.push_back(static_cast<std::uint64_t>(binding.nodes[index]));
            weights.push_back(static_cast<double>(binding.weights[index]));
        }
        Json item = Json::object();
        item["nodes"] = std::move(nodes);
        item["weights"] = std::move(weights);
        bindings_json.push_back(std::move(item));
    }
    result["soft_body_surface_bindings"] = std::move(bindings_json);
    result["soft_body_node_mass_kg"] = static_cast<double>(soft_node_mass);
    result["soft_body_node_radius_m"] = static_cast<double>(soft_node_radius);
    result["soft_body_stretch_compliance_m_per_n"] = 0.0;
    result["soft_body_velocity_damping_per_s"] = 0.0;
    result["soft_body_spring_damping_per_s"] = 0.0;
    result["soft_body_contact_friction"] = 0.0;
    result["soft_body_shape_matching_stiffness"] = 0.0;
    result["soft_body_maximum_projection_fraction"] = 0.2;
    result["soft_body_constraint_velocity_response"] = 0.0;
    result["soft_body_maximum_speed_m_s"] = 3.4;
    result["soft_body_solver_iterations"] = static_cast<std::uint64_t>(1U);
    result["smoke_particle_capacity"] =
        static_cast<std::uint64_t>(smoke_capacity);
    result["smoke_emitter_center_m"] =
        vector_json(PmFloat3{0.25F, 0.25F, 0.50F});
    Json half_extents = Json::array();
    half_extents.push_back(0.05);
    half_extents.push_back(0.05);
    result["smoke_emitter_half_extents_m"] = std::move(half_extents);
    result["smoke_emission_rate_per_s"] = 1'000.0;
    result["smoke_lifetime_seconds"] = 10.0;
    result["smoke_particle_radius_m"] = 0.10;
    result["smoke_rest_number_density"] = 1.0;
    result["smoke_maximum_speed_m_s"] = grid ? 3.7 : 4.0;
    result["smoke_grid_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_resolution : 0U);
    result["smoke_grid_vertical_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_vertical_resolution : 0U);
    result["smoke_grid_kinematic_viscosity_m2_s"] = grid ? 0.1 : 0.0;
    result["smoke_grid_minimum_m"] = vector_json(PmFloat3{});
    result["smoke_grid_edge_length_m"] = 1.0;
    result["declared_mode_flags"] = std::move(mode_flags);
    result["exercised_mode"] = "DRAG";
    result["contact_distance_m"] = static_cast<double>(contact_distance);
    result["tracer_contact"] = false;
    result["forward_carrier_velocity_m_s"] =
        vector_json(PmFloat3{carrier_speed, 0.0F, 0.0F});
    result["stationary_carrier_velocity_m_s"] = vector_json(PmFloat3{});
    result["reverse_carrier_velocity_m_s"] =
        vector_json(PmFloat3{-carrier_speed, 0.0F, 0.0F});
    result["control_wind_drag_per_s"] = 0.0;
    result["test_wind_drag_per_s"] = static_cast<double>(test_drag);
    result["maximum_wind_acceleration_m_s2"] =
        static_cast<double>(maximum_wind_acceleration);
    result["oracle"] =
        "paired-public-relative-flow-soft-body-loading-and-smoke-invariance";
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

void append_vector(std::ostringstream &stream, PmFloat3 value) {
    append_float(stream, value.x);
    stream << ',';
    append_float(stream, value.y);
    stream << ',';
    append_float(stream, value.z);
}

[[nodiscard]] std::string graph_fingerprint(const PmCouplingState *edge) {
    if (edge == nullptr)
        return sha256("parallel-mater-smoke-soft-drag-graph/v1;inactive");
    const auto &smoke = edge->parameters.smoke;
    std::ostringstream canonical;
    canonical << "parallel-mater-smoke-soft-drag-graph/v1;type="
              << static_cast<unsigned>(edge->type)
              << ";first=" << edge->first << ";second=" << edge->second
              << ";modes=" << edge->modeFlags
              << ";direction=SECOND_TO_FIRST;material="
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
    canonical << "parallel-mater-smoke-soft-drag-endpoints/v1";
    for (const auto &state : snapshot.soft) {
        canonical << ";soft-position=";
        append_vector(canonical, state.position);
        canonical << ",velocity=";
        append_vector(canonical, state.velocity);
    }
    for (const auto &vertex : snapshot.surface) {
        canonical << ";soft-surface=";
        append_vector(canonical, vertex);
    }
    canonical << ";smoke-count=" << snapshot.smoke.size();
    for (const auto &state : snapshot.smoke) {
        canonical << ";smoke-position=";
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

void add_checkpoint(Json &observations, const std::string &prefix,
                    std::string id, std::string outcome,
                    const PmCouplingState *edge,
                    std::uint64_t graph_generation,
                    const Snapshot &endpoints) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        prefix + "-smoke-soft-drag-graph", edge != nullptr, graph_generation,
        graph_fingerprint(edge)));
    resources.push_back(resource_json(
        prefix + "-smoke-soft-endpoints", true, 1U,
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
        throw std::runtime_error("unsupported: soft and smoke features");
    require_drag_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    // A positive emitter rate is required by the public descriptor. The five
    // 0.1 ms lifecycle steps remain below the first-particle emission boundary.
    initialize_endpoints(context, fixture, 0.0F, 1.0F);
    const PmCouplingState baseline = drag_edge(grid, 10.0F);
    const PmCouplingState replacement = drag_edge(grid, 20.0F);
    const PmCouplingState recovery = drag_edge(grid, 30.0F);
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
    add_checkpoint(observations, prefix, "baseline", "PM_SUCCESS", &baseline, 1U,
                   mutate(&baseline, 1U, PM_SUCCESS));
    add_checkpoint(observations, prefix, "replacement", "PM_SUCCESS", &replacement,
                   2U, mutate(&replacement, 1U, PM_SUCCESS));
    add_checkpoint(observations, prefix, "rejected-overflow",
                   "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U,
                   mutate(overflow.data(),
                          static_cast<std::uint32_t>(overflow.size()),
                          PM_ERROR_CAPACITY_EXCEEDED));
    add_checkpoint(observations, prefix, "removal", "PM_SUCCESS", nullptr, 3U,
                   mutate(nullptr, 0U, PM_SUCCESS));
    add_checkpoint(observations, prefix, "recovery", "PM_SUCCESS", &recovery, 4U,
                   mutate(&recovery, 1U, PM_SUCCESS));
    DriverInfo driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] Json lifecycle_case_json(bool grid, LifecycleRun run) {
    const std::string prefix = grid ? "grid" : "particle";
    Json families = Json::array();
    families.push_back("soft-body");
    families.push_back("smoke");
    Json unrelated = Json::array();
    unrelated.push_back(prefix + "-smoke-soft-endpoints");
    Json attempt = Json::object();
    attempt["fixture"] = "smoke-soft-drag-public-lifecycle";
    attempt["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    attempt["status"] = "completed";
    Json item = Json::object();
    item["attempt"] = std::move(attempt);
    item["execution_status"] = "completed";
    item["families"] = std::move(families);
    item["id"] = prefix +
        "-smoke-soft-drag-capacity-replacement-removal-rollback";
    item["observations"] = std::move(run.observations);
    item["unrelated_resources"] = std::move(unrelated);
    return item;
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}

} // namespace

std::string conformance_smoke_soft_drag_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-soft-drag-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/smoke-soft-drag-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json fixtures = Json::array();
        fixtures.push_back(quality_fixture_json(false, particle_driver));
        fixtures.push_back(quality_fixture_json(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "smoke-soft representations selected different devices");
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

std::string conformance_smoke_soft_drag_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-soft-drag-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/smoke-soft-drag-lifecycle-v1.json");
    try {
        LifecycleRun particle = run_lifecycle_fixture(false);
        LifecycleRun grid = run_lifecycle_fixture(true);
        if (!same_driver(particle.driver, grid.driver))
            throw std::runtime_error(
                "smoke-soft lifecycle representations selected different devices");
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
