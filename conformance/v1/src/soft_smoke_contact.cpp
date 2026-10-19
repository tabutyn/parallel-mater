// SPDX-License-Identifier: MIT
// Public-SDK soft-to-smoke contact quality and grid graph lifecycle fixtures.
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
constexpr float contact_distance = 0.015F;
constexpr float quality_emission_rate = 60.0F;
constexpr float soft_node_mass = 0.25F;
constexpr float soft_node_radius = 0.002F;
constexpr std::array<PmFloat3, 5U> soft_nodes{{
    {0.0F, 0.02F, 0.02F},
    {0.0F, 0.14F, 0.02F},
    {0.0F, 0.14F, 0.14F},
    {0.0F, 0.02F, 0.14F},
    {0.08F, 0.08F, 0.08F}}};
constexpr std::array<PmSoftBodyBond, 10U> soft_bonds{{
    {0U, 1U, 0.12F}, {0U, 2U, 0.169705627F},
    {0U, 3U, 0.12F}, {0U, 4U, 0.116619038F},
    {1U, 2U, 0.12F}, {1U, 3U, 0.169705627F},
    {1U, 4U, 0.116619038F}, {2U, 3U, 0.12F},
    {2U, 4U, 0.116619038F}, {3U, 4U, 0.116619038F}}};
constexpr std::array<std::uint32_t, 18U> surface_indices{{
    0U, 1U, 2U, 0U, 2U, 3U,
    0U, 4U, 1U, 1U, 4U, 2U,
    2U, 4U, 3U, 3U, 4U, 0U}};
constexpr std::array<float, soft_nodes.size()> dynamic_inverse_masses{{
    1.0F / soft_node_mass, 1.0F / soft_node_mass,
    1.0F / soft_node_mass, 1.0F / soft_node_mass,
    1.0F / soft_node_mass}};

[[nodiscard]] std::array<PmSoftBodySurfaceBinding, soft_nodes.size()>
surface_bindings() {
    std::array<PmSoftBodySurfaceBinding, soft_nodes.size()> result{};
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
    std::vector<PmSmokeParticleState> smoke{};
    std::array<PmSoftBodyNodeState, soft_nodes.size()> nodes{};
    std::array<PmFloat3, soft_nodes.size()> surface{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange smoke_output{};
    PmBufferRange node_output{};
    PmBufferRange surface_output{};
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
        throw std::runtime_error("query soft-smoke contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown soft-smoke contact backend");
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

void require_contact_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_SOFT_BODY, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_FIRST_TO_SECOND,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(std::string("unsupported: soft-smoke contact ") +
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
    material.shapeMatchingStiffness = 1.0F;
    material.maximumProjectionFraction = 0.2F;
    material.constraintVelocityResponse = 0.0F;
    material.maximumSpeed = 10.0F;
    material.solverIterations = 2U;
    return material;
}

[[nodiscard]] PmSmokeMaterialState smoke_material(bool grid) {
    PmSmokeMaterialState material{};
    material.particleRadius = 0.005F;
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

[[nodiscard]] PmSmokeInfo smoke_info(float emission_rate,
                                     PmFloat3 initial_velocity,
                                     PmFloat3 wind) {
    PmSmokeInfo info{};
    info.active = PM_TRUE;
    info.materialIndex = 0U;
    info.particleCapacity = smoke_capacity;
    info.emitterCenter = {-0.005F, 0.08F, 0.08F};
    info.emitterHalfExtents = {0.001F, 0.001F};
    info.initialVelocity = initial_velocity;
    info.wind = wind;
    info.particlesPerSecond = emission_rate;
    info.lifetime = 1.0F;
    info.gridMinimum = {-0.08F, 0.0F, 0.0F};
    info.gridEdgeLength = grid_edge_length;
    return info;
}

[[nodiscard]] PmCouplingState contact_coupling(bool grid,
                                               float distance,
                                               float inert_drag_cap = 0.0F) {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_SMOKE_SOFT_BODY;
    // Public coupling storage names smoke first even though the advertised
    // semantic tuple is SOFT_BODY -> SMOKE.
    coupling.first = 0U;
    coupling.second = 0U;
    coupling.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_DRAG_BIT;
    coupling.materialProfile = grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
                                    : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    coupling.parameters.smoke.windDrag = 0.0F;
    coupling.parameters.smoke.maximumWindAcceleration = inert_drag_cap;
    coupling.parameters.smoke.contactDistance = distance;
    return coupling;
}

[[nodiscard]] FixtureObjects create_fixture(Context &context, bool grid,
                                            std::uint32_t coupling_capacity) {
    constexpr PmSoftBodyCapacity soft_capacity{
        static_cast<std::uint32_t>(soft_nodes.size()),
        static_cast<std::uint32_t>(soft_bonds.size()),
        static_cast<std::uint32_t>(soft_nodes.size()),
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

    constexpr PmDeviceSize smoke_size = sizeof(PmDomainOutputHeader) +
        smoke_capacity * sizeof(PmSmokeParticleState);
    constexpr PmDeviceSize node_size = sizeof(PmDomainOutputHeader) +
        soft_nodes.size() * sizeof(PmSoftBodyNodeState);
    constexpr PmDeviceSize surface_size = sizeof(PmDomainOutputHeader) +
        soft_nodes.size() * sizeof(PmFloat3);
    fixture.output = context.buffer(
        smoke_size + node_size + surface_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.smoke_output = {fixture.output.handle, 0U, smoke_size};
    fixture.node_output = {fixture.output.handle, smoke_size, node_size};
    fixture.surface_output = {
        fixture.output.handle, smoke_size + node_size, surface_size};
    return fixture;
}

[[nodiscard]] PmSoftBodyInfo soft_info(
    const float *inverse_masses,
    const std::array<PmSoftBodySurfaceBinding, soft_nodes.size()> &bindings) {
    return {PM_TRUE, 0U, static_cast<std::uint32_t>(soft_nodes.size()),
            soft_nodes.data(), static_cast<std::uint32_t>(soft_bonds.size()),
            soft_bonds.data(), inverse_masses,
            static_cast<std::uint32_t>(soft_nodes.size()), soft_nodes.data(),
            static_cast<std::uint32_t>(surface_indices.size()),
            surface_indices.data(), bindings.data()};
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopySmokeState(context.commands, fixture.simulation, 0U,
                                &fixture.smoke_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodyState(context.commands, fixture.simulation, 0U,
                                   &fixture.node_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySoftBodySurface(context.commands, fixture.simulation, 0U,
                                     &fixture.surface_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *smoke_header =
        reinterpret_cast<const PmDomainOutputHeader *>(base);
    const auto *node_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.node_output.offset);
    const auto *surface_header =
        reinterpret_cast<const PmDomainOutputHeader *>(
            base + fixture.surface_output.offset);
    if (smoke_header->activeCount > smoke_capacity ||
        node_header->activeCount != soft_nodes.size() ||
        surface_header->activeCount != soft_nodes.size())
        throw std::runtime_error("soft-smoke fixture output count mismatch");
    Snapshot result{};
    const auto *smoke =
        reinterpret_cast<const PmSmokeParticleState *>(smoke_header + 1);
    result.smoke.assign(smoke, smoke + smoke_header->activeCount);
    const auto *nodes =
        reinterpret_cast<const PmSoftBodyNodeState *>(node_header + 1);
    const auto *surface =
        reinterpret_cast<const PmFloat3 *>(surface_header + 1);
    std::copy_n(nodes, result.nodes.size(), result.nodes.begin());
    std::copy_n(surface, result.surface.size(), result.surface.begin());
    return result;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture,
                          const float *inverse_masses,
                          float emission_rate, PmFloat3 initial_velocity,
                          PmFloat3 wind) {
    const auto bindings = surface_bindings();
    const PmSoftBodyInfo soft = soft_info(inverse_masses, bindings);
    const PmSmokeInfo smoke = smoke_info(
        emission_rate, initial_velocity, wind);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSoftBodies(context.commands, fixture.simulation, 0U, 1U,
                                 &soft) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
}

[[nodiscard]] QualityRun run_quality_fixture(bool grid, bool coupled) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: soft and smoke features");
    require_contact_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    initialize_endpoints(context, fixture, dynamic_inverse_masses.data(),
                         quality_emission_rate,
                         {1.0F, 0.25F, 0.0F}, {});
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    for (std::uint32_t index = 0U; index < warmup_steps; ++index) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        const PmResult step_status = context.submit();
        if (step_status != PM_SUCCESS)
            throw std::runtime_error(
                "soft-smoke warmup failed: " +
                std::to_string(step_status));
    }
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    Snapshot initial = read_snapshot(context, fixture);
    if (initial.smoke.size() != 1U)
        throw std::runtime_error("soft-smoke fixture did not emit one tracer");

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
    const PmResult contact_status = context.submit();
    if (contact_status != PM_SUCCESS)
        throw std::runtime_error(
            "soft-smoke contact submission failed: " +
            std::to_string(contact_status));
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
    Json smoke = Json::array();
    for (std::uint32_t slot = 0U; slot < snapshot.smoke.size(); ++slot) {
        const auto &particle = snapshot.smoke[slot];
        Json item = Json::object();
        item["slot"] = static_cast<std::uint64_t>(slot);
        item["position_m"] = vector_json(particle.position);
        item["velocity_m_s"] = vector_json(particle.velocity);
        item["age_seconds"] = static_cast<double>(particle.age);
        item["number_density"] =
            static_cast<double>(particle.numberDensity);
        item["pressure_m_s2"] = static_cast<double>(particle.pressure);
        item["vorticity_per_s"] = vector_json(particle.vorticity);
        smoke.push_back(std::move(item));
    }
    Json vertices = Json::array();
    Json velocities = Json::array();
    for (std::size_t index = 0U; index < snapshot.surface.size(); ++index) {
        vertices.push_back(vector_json(snapshot.surface[index]));
        velocities.push_back(vector_json(snapshot.nodes[index].velocity));
    }
    Json result = Json::object();
    result["smoke_particles"] = std::move(smoke);
    result["soft_body_surface_vertices_m"] = std::move(vertices);
    result["soft_body_surface_velocities_m_s"] = std::move(velocities);
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
    for (const auto vertex : soft_nodes)
        result.push_back(vector_json(vertex));
    return result;
}

[[nodiscard]] Json indices_json() {
    Json result = Json::array();
    for (const auto index : surface_indices)
        result.push_back(static_cast<std::uint64_t>(index));
    return result;
}

[[nodiscard]] Json quality_fixture_json(bool grid, DriverInfo &driver) {
    QualityRun control = run_quality_fixture(grid, false);
    QualityRun coupled = run_quality_fixture(grid, true);
    if (!same_driver(control.driver, coupled.driver))
        throw std::runtime_error(
            "soft-smoke paired runs selected different devices");
    driver = control.driver;
    Json modes = Json::array();
    modes.push_back("CONTACT");
    modes.push_back("DRAG");
    Json runs = Json::array();
    runs.push_back(run_json("control", control));
    runs.push_back(run_json("coupled", coupled));
    Json result = Json::object();
    result["id"] = grid ? "grid-soft-contact" : "particle-soft-contact";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["warmup_step_count"] = static_cast<std::uint64_t>(warmup_steps);
    result["step_count"] = static_cast<std::uint64_t>(quality_steps);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["declared_mode_flags"] = std::move(modes);
    result["exercised_mode"] = "CONTACT";
    result["soft_body_surface_vertices_m"] = vertices_json();
    result["soft_body_surface_triangle_indices"] = indices_json();
    result["soft_body_node_mass_kg"] =
        static_cast<double>(soft_node_mass);
    result["soft_body_node_radius_m"] =
        static_cast<double>(soft_node_radius);
    result["soft_body_dynamic"] = true;
    result["smoke_particle_capacity"] =
        static_cast<std::uint64_t>(smoke_capacity);
    result["smoke_emitter_center_m"] =
        vector_json(PmFloat3{-0.005F, 0.08F, 0.08F});
    Json half_extents = Json::array();
    half_extents.push_back(0.001);
    half_extents.push_back(0.001);
    result["smoke_emitter_half_extents_m"] = std::move(half_extents);
    result["smoke_emission_rate_per_s"] =
        static_cast<double>(quality_emission_rate);
    result["smoke_initial_velocity_m_s"] =
        vector_json(PmFloat3{1.0F, 0.25F, 0.0F});
    result["smoke_wind_m_s"] = vector_json(PmFloat3{});
    result["smoke_lifetime_seconds"] = 1.0;
    result["smoke_particle_radius_m"] = 0.005;
    result["smoke_rest_number_density"] = 1.0;
    result["smoke_pressure_stiffness_m2_s2"] = 0.0;
    result["smoke_viscosity_m2_s"] = 0.0;
    result["smoke_maximum_speed_m_s"] = 2.0;
    result["smoke_grid_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_resolution : 0U);
    result["smoke_grid_vertical_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_resolution : 0U);
    result["smoke_grid_minimum_m"] =
        vector_json(PmFloat3{-0.08F, 0.0F, 0.0F});
    result["smoke_grid_edge_length_m"] =
        static_cast<double>(grid_edge_length);
    result["smoke_grid_spacing_m"] =
        static_cast<double>(grid ? grid_spacing : 0.0F);
    result["contact_distance_m"] = static_cast<double>(contact_distance);
    result["wind_drag_per_s"] = 0.0;
    result["maximum_wind_acceleration_m_s2"] = 0.0;
    result["oracle"] =
        "paired-public-surface-redirection-and-soft-body-invariance";
    result["runs"] = std::move(runs);
    return result;
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

[[nodiscard]] bool contact_redirected(const Snapshot &snapshot) {
    return snapshot.smoke.size() == 1U &&
        snapshot.smoke[0].position.x <= -0.0145F &&
        snapshot.smoke[0].velocity.x <= 1.0e-4F;
}

[[nodiscard]] bool soft_unchanged(const Snapshot &snapshot) {
    bool unchanged = snapshot.nodes.size() == soft_nodes.size();
    for (std::size_t index = 0U; index < snapshot.nodes.size(); ++index) {
        const auto &actual = snapshot.nodes[index];
        const auto &expected = soft_nodes[index];
        unchanged = unchanged &&
            std::abs(actual.position.x - expected.x) <= 1.0e-5F &&
            std::abs(actual.position.y - expected.y) <= 1.0e-5F &&
            std::abs(actual.position.z - expected.z) <= 1.0e-5F &&
            std::abs(actual.velocity.x) <= 1.0e-5F &&
            std::abs(actual.velocity.y) <= 1.0e-5F &&
            std::abs(actual.velocity.z) <= 1.0e-5F &&
            std::abs(snapshot.surface[index].x - expected.x) <= 1.0e-5F &&
            std::abs(snapshot.surface[index].y - expected.y) <= 1.0e-5F &&
            std::abs(snapshot.surface[index].z - expected.z) <= 1.0e-5F;
    }
    return unchanged;
}

[[nodiscard]] std::string graph_fingerprint(
    const PmCouplingState *coupling, const Snapshot &probe) {
    std::ostringstream canonical;
    canonical << "parallel-mater-soft-smoke-contact-graph/v2"
              << ";contact-redirected=" << contact_redirected(probe)
              << ";soft-unchanged=" << soft_unchanged(probe);
    if (coupling == nullptr) {
        canonical << ";inactive";
        return sha256(canonical.str());
    }
    canonical << ";type="
              << static_cast<unsigned>(coupling->type)
              << ";first=" << coupling->first
              << ";second=" << coupling->second
              << ";modes=" << coupling->modeFlags
              << ";direction=FIRST_TO_SECOND;material="
              << static_cast<unsigned>(coupling->materialProfile)
              << ";wind-drag=";
    append_float(canonical, coupling->parameters.smoke.windDrag);
    canonical << ";maximum-wind-acceleration=";
    append_float(canonical,
                 coupling->parameters.smoke.maximumWindAcceleration);
    canonical << ";contact-distance=";
    append_float(canonical, coupling->parameters.smoke.contactDistance);
    return sha256(canonical.str());
}

[[nodiscard]] std::string endpoint_fingerprint(const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-soft-smoke-contact-endpoints/v2"
              << ";smoke-count=" << snapshot.smoke.size()
              << ";contact-redirected=" << contact_redirected(snapshot)
              << ";soft-unchanged=" << soft_unchanged(snapshot);
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
        "grid-soft-smoke-contact-graph", coupling != nullptr,
        graph_generation, graph_fingerprint(coupling, endpoints)));
    resources.push_back(resource_json(
        "grid-soft-smoke-endpoints", true, 1U,
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

[[nodiscard]] LifecycleRun run_lifecycle_fixture() {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: soft and smoke features");
    require_contact_capability(context, true);
    FixtureObjects fixture = create_fixture(context, true, 1U);
    initialize_endpoints(context, fixture, dynamic_inverse_masses.data(),
                         quality_emission_rate, {1.0F, 0.0F, 0.0F},
                         {1.0F, 0.0F, 0.0F});
    // Replacement changes CONTACT itself. All clearances remain beyond the
    // shared -0.0145 m behavioral threshold. Bundled DRAG stays zero.
    const PmCouplingState baseline = contact_coupling(
        true, contact_distance);
    const PmCouplingState replacement = contact_coupling(
        true, contact_distance + 0.001F);
    const PmCouplingState recovery = contact_coupling(
        true, contact_distance + 0.002F);
    const std::array<PmCouplingState, 2U> overflow{{replacement, recovery}};
    const PmStepInfo step{1.0e-3F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    const PmStepInfo warmup_step{
        timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    Json observations = Json::array();
    for (std::uint32_t index = 0U; index < warmup_steps; ++index) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &warmup_step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }
    const auto submit_graph = [&](const PmCouplingState *couplings,
                                  std::uint32_t count,
                                  PmResult record_result,
                                  PmResult submit_result) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, count,
                                  couplings) == record_result);
        // Graph changes are consumed transactionally by a step. A failed
        // submit must roll back before the separate public probe below.
        if (record_result == PM_SUCCESS)
            PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                        &step) == PM_SUCCESS);
        PM_CHECK(context.submit() == submit_result);
    };
    const auto run_probe = [&]() {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        record_copies(context, fixture);
        const PmResult probe_status = context.submit();
        if (probe_status != PM_SUCCESS)
            throw std::runtime_error(
                "soft-smoke lifecycle probe submission failed: " +
                std::to_string(probe_status));
        Snapshot snapshot = read_snapshot(context, fixture);
        if (snapshot.smoke.size() != 1U)
            throw std::runtime_error(
                "soft-smoke lifecycle probe lost its tracer");
        return snapshot;
    };
    submit_graph(&baseline, 1U, PM_SUCCESS, PM_SUCCESS);
    add_checkpoint(observations, "baseline", "PM_SUCCESS", &baseline, 1U,
                   run_probe());
    submit_graph(&replacement, 1U, PM_SUCCESS, PM_SUCCESS);
    add_checkpoint(observations, "replacement", "PM_SUCCESS", &replacement,
                   2U, run_probe());
    submit_graph(overflow.data(),
                 static_cast<std::uint32_t>(overflow.size()),
                 PM_ERROR_CAPACITY_EXCEEDED, PM_SUCCESS);
    add_checkpoint(observations, "rejected-overflow",
                   "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U,
                   run_probe());
    submit_graph(nullptr, 0U, PM_SUCCESS, PM_SUCCESS);
    add_checkpoint(observations, "removal", "PM_SUCCESS", nullptr, 3U,
                   run_probe());
    submit_graph(&recovery, 1U, PM_SUCCESS, PM_SUCCESS);
    add_checkpoint(observations, "recovery", "PM_SUCCESS", &recovery, 4U,
                   run_probe());
    DriverInfo driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}
} // namespace

std::string conformance_soft_smoke_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "soft-smoke-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/soft-smoke-contact-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json fixtures = Json::array();
        fixtures.push_back(quality_fixture_json(false, particle_driver));
        fixtures.push_back(quality_fixture_json(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "soft-smoke representations selected different devices");
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

std::string conformance_soft_smoke_contact_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "soft-smoke-contact-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/soft-smoke-contact-lifecycle-v1.json");
    try {
        LifecycleRun run = run_lifecycle_fixture();
        Json families = Json::array();
        families.push_back("soft");
        families.push_back("smoke");
        Json unrelated = Json::array();
        Json attempt = Json::object();
        attempt["fixture"] = "soft-smoke-contact-public-lifecycle";
        attempt["representation"] = "SMOKE_GRID";
        attempt["status"] = "completed";
        Json item = Json::object();
        item["attempt"] = std::move(attempt);
        item["execution_status"] = "completed";
        item["families"] = std::move(families);
        item["id"] =
            "grid-soft-smoke-contact-capacity-replacement-removal-rollback";
        item["observations"] = std::move(run.observations);
        item["unrelated_resources"] = std::move(unrelated);
        Json cases = Json::array();
        cases.push_back(std::move(item));
        result["execution_status"] = "completed";
        result["driver"] = driver_json(run.driver);
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
