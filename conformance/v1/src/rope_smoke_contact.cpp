// SPDX-License-Identifier: MIT
// Public-SDK rope-to-smoke contact quality and grid graph lifecycle fixtures.
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

constexpr float timestep = 1.0F / 60.0F;
constexpr std::uint32_t warmup_steps = 2U;
constexpr std::uint32_t quality_steps = 1U;
constexpr std::uint32_t smoke_capacity = 1U;
constexpr std::uint32_t grid_resolution = 16U;
constexpr std::uint32_t grid_cells =
    grid_resolution * grid_resolution * grid_resolution;
constexpr float grid_edge_length = 0.16F;
constexpr float grid_spacing = grid_edge_length /
    static_cast<float>(grid_resolution);
constexpr float authored_contact_distance = 0.015F;
constexpr float contact_clearance = 0.015F;
// Preserve the historical cadence under the sole 60 Hz schedule: no tracer
// on warmup tick 1, one tracer on warmup tick 2, and no replacement on the
// contact probe tick. The former 60/s value encoded that cadence at 100 Hz.
constexpr float quality_emission_rate = 30.0F;
// Particle smoke advances a newly emitted tracer in its admission tick while
// grid smoke exposes the source point. Pre-offset the particle source by that
// one fixed-tick displacement so both representations reach the historical
// -0.020 m warmup boundary without changing production emission semantics.
constexpr float particle_quality_emitter_x = -0.0271144F;
constexpr float grid_quality_emitter_x = -0.020F;
constexpr float rope_mass = 0.5F;
constexpr float rope_radius = 0.015F;
constexpr float rope_node_spacing = 0.030F;
constexpr std::array<PmFloat3, 2U> rope_centerline{{
    {0.0F, 0.0F, 0.0F},
    {0.0F, 0.03F, 0.0F}}};
constexpr auto initial_rope_nodes = rope_centerline;

struct DriverInfo {
    std::string name{};
    std::string device{};
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t version{};
};

struct Snapshot {
    std::vector<PmSmokeParticleState> smoke{};
    std::array<PmRopeNodeState, initial_rope_nodes.size()> nodes{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange smoke_output{};
    PmBufferRange rope_output{};
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
        throw std::runtime_error("query rope-smoke contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown rope-smoke contact backend");
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
    needed.rope = PM_TRUE;
    needed.smoke = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

void require_contact_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_ROPE, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_FIRST_TO_SECOND,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(std::string("unsupported: rope-smoke contact ") +
                                 (grid ? "SMOKE_GRID" : "SMOKE_PARTICLE"));
}

[[nodiscard]] PmRopeMaterialState rope_material() {
    PmRopeMaterialState material{};
    material.nodeSpacing = rope_node_spacing;
    material.radius = rope_radius;
    material.mass = rope_mass;
    material.stretchCompliance = 0.0F;
    material.velocityDamping = 0.0F;
    material.maximumSubstepTimestep = timestep;
    material.friction = 0.0F;
    material.maximumSpeed = 0.89F;
    material.solverIterations = 2U;
    return material;
}

[[nodiscard]] PmSmokeMaterialState smoke_material(bool grid,
                                                  float wind_response) {
    PmSmokeMaterialState material{};
    material.particleRadius = 0.005F;
    material.buoyancy = 0.0F;
    material.response = wind_response;
    material.restNumberDensity = 1.0F;
    material.pressureStiffness = 0.0F;
    material.viscosity = 0.0F;
    material.vorticityConfinement = 0.0F;
    material.maximumSpeed = grid ? 0.59F : 0.44F;
    material.gridResolution = grid ? grid_resolution : 0U;
    material.gridVerticalResolution = grid ? grid_resolution : 0U;
    material.gridPressureIterations = grid ? 12U : 0U;
    material.gridPressureTolerance = 1.0e-5F;
    return material;
}

[[nodiscard]] PmSmokeInfo smoke_info(float emission_rate,
                                     PmFloat3 initial_velocity,
                                     PmFloat3 wind,
                                     float emitter_x) {
    PmSmokeInfo info{};
    info.active = PM_TRUE;
    info.materialIndex = 0U;
    info.particleCapacity = smoke_capacity;
    info.emitterCenter = {emitter_x, 0.015F, 0.0F};
    info.emitterHalfExtents = {0.001F, 0.001F};
    info.initialVelocity = initial_velocity;
    info.wind = wind;
    info.particlesPerSecond = emission_rate;
    info.lifetime = 1.0F;
    info.gridMinimum = {-0.08F, -0.08F, -0.08F};
    info.gridEdgeLength = grid_edge_length;
    return info;
}

[[nodiscard]] PmCouplingState contact_coupling(bool grid,
                                               float distance,
                                               float inert_drag_cap = 0.0F) {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_SMOKE_ROPE;
    // Public coupling storage names smoke first even though the advertised
    // semantic tuple is ROPE -> SMOKE.
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
                                            std::uint32_t coupling_capacity,
                                            float wind_response = 0.0F) {
    constexpr PmRopeCapacity rope_capacity{
        static_cast<std::uint32_t>(initial_rope_nodes.size())};
    const PmSmokeCapacity smoke_domain{
        smoke_capacity, grid ? grid_cells : 1U};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.ropeCount = 1U;
    create.pRopeCapacities = &rope_capacity;
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

    const PmRopeMaterialState rope = rope_material();
    const PmSmokeMaterialState smoke = smoke_material(grid, wind_response);
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.ropeMaterialCount = 1U;
    pipeline_info.pRopeMaterials = &rope;
    pipeline_info.smokeMaterialCount = 1U;
    pipeline_info.pSmokeMaterials = &smoke;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize smoke_size = sizeof(PmDomainOutputHeader) +
        smoke_capacity * sizeof(PmSmokeParticleState);
    constexpr PmDeviceSize rope_size = sizeof(PmDomainOutputHeader) +
        initial_rope_nodes.size() * sizeof(PmRopeNodeState);
    fixture.output = context.buffer(
        smoke_size + rope_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.smoke_output = {fixture.output.handle, 0U, smoke_size};
    fixture.rope_output = {fixture.output.handle, smoke_size, rope_size};
    return fixture;
}

[[nodiscard]] PmRopeInfo rope_info() {
    PmRopeInfo info{};
    info.active = PM_TRUE;
    info.materialIndex = 0U;
    info.pointCount = static_cast<std::uint32_t>(rope_centerline.size());
    info.pCenterline = rope_centerline.data();
    return info;
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopySmokeState(context.commands, fixture.simulation, 0U,
                                &fixture.smoke_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRopeState(context.commands, fixture.simulation, 0U,
                                &fixture.rope_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *smoke_header =
        reinterpret_cast<const PmDomainOutputHeader *>(base);
    const auto *rope_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.rope_output.offset);
    if (smoke_header->activeCount > smoke_capacity ||
        rope_header->activeCount != initial_rope_nodes.size())
        throw std::runtime_error("rope-smoke fixture output count mismatch");
    Snapshot result{};
    const auto *smoke =
        reinterpret_cast<const PmSmokeParticleState *>(smoke_header + 1);
    result.smoke.assign(smoke, smoke + smoke_header->activeCount);
    const auto *nodes =
        reinterpret_cast<const PmRopeNodeState *>(rope_header + 1);
    std::copy_n(nodes, result.nodes.size(), result.nodes.begin());
    return result;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture,
                          float emission_rate, PmFloat3 initial_velocity,
                          PmFloat3 wind, float emitter_x) {
    const PmRopeInfo rope = rope_info();
    const PmSmokeInfo smoke = smoke_info(
        emission_rate, initial_velocity, wind, emitter_x);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteRopes(context.commands, fixture.simulation, 0U, 1U,
                            &rope) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
}

[[nodiscard]] QualityRun run_quality_fixture(bool grid, bool coupled) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: rope and smoke features");
    require_contact_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    initialize_endpoints(context, fixture, quality_emission_rate,
                         {1.0F, 0.25F, 0.0F}, {},
                         grid ? grid_quality_emitter_x
                              : particle_quality_emitter_x);
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
                "rope-smoke warmup failed: " +
                std::to_string(step_status));
    }
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    Snapshot initial = read_snapshot(context, fixture);
    if (initial.smoke.size() != 1U)
        throw std::runtime_error("rope-smoke fixture did not emit one tracer");

    const PmCouplingState coupling = contact_coupling(
        grid, authored_contact_distance);
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
            "rope-smoke contact submission failed: " +
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
    Json nodes = Json::array();
    Json velocities = Json::array();
    for (std::size_t index = 0U; index < snapshot.nodes.size(); ++index) {
        nodes.push_back(vector_json(snapshot.nodes[index].position));
        velocities.push_back(vector_json(snapshot.nodes[index].velocity));
    }
    Json result = Json::object();
    result["smoke_particles"] = std::move(smoke);
    result["rope_nodes_m"] = std::move(nodes);
    result["rope_velocities_m_s"] = std::move(velocities);
    return result;
}

[[nodiscard]] Json run_json(const char *id, const QualityRun &run) {
    Json result = Json::object();
    result["id"] = id;
    result["initial"] = snapshot_json(run.initial);
    result["final"] = snapshot_json(run.final);
    return result;
}

[[nodiscard]] Json nodes_json() {
    Json result = Json::array();
    for (const auto node : rope_centerline)
        result.push_back(vector_json(node));
    return result;
}

[[nodiscard]] Json quality_fixture_json(bool grid, DriverInfo &driver) {
    QualityRun control = run_quality_fixture(grid, false);
    QualityRun coupled = run_quality_fixture(grid, true);
    if (!same_driver(control.driver, coupled.driver))
        throw std::runtime_error(
            "rope-smoke paired runs selected different devices");
    driver = control.driver;
    Json modes = Json::array();
    modes.push_back("CONTACT");
    modes.push_back("DRAG");
    Json runs = Json::array();
    runs.push_back(run_json("control", control));
    runs.push_back(run_json("coupled", coupled));
    Json result = Json::object();
    result["id"] = grid ? "grid-rope-contact" : "particle-rope-contact";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["warmup_step_count"] = static_cast<std::uint64_t>(warmup_steps);
    result["step_count"] = static_cast<std::uint64_t>(quality_steps);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["declared_mode_flags"] = std::move(modes);
    result["exercised_mode"] = "CONTACT";
    result["rope_centerline_m"] = nodes_json();
    result["rope_node_count"] =
        static_cast<std::uint64_t>(initial_rope_nodes.size());
    result["rope_node_spacing_m"] =
        static_cast<double>(rope_node_spacing);
    result["rope_radius_m"] = static_cast<double>(rope_radius);
    result["rope_total_mass_kg"] = static_cast<double>(rope_mass);
    result["rope_dynamic"] = true;
    result["smoke_particle_capacity"] =
        static_cast<std::uint64_t>(smoke_capacity);
    result["smoke_emitter_center_m"] =
        vector_json(PmFloat3{grid ? grid_quality_emitter_x
                                  : particle_quality_emitter_x,
                             0.015F, 0.0F});
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
    result["smoke_maximum_speed_m_s"] = grid ? 0.59 : 0.44;
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
    result["authored_contact_distance_m"] =
        static_cast<double>(authored_contact_distance);
    result["expected_clearance_m"] = static_cast<double>(contact_clearance);
    result["wind_drag_per_s"] = 0.0;
    result["maximum_wind_acceleration_m_s2"] = 0.0;
    result["oracle"] =
        "paired-public-capsule-redirection-and-rope-invariance";
    result["runs"] = std::move(runs);
    return result;
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

[[nodiscard]] bool contact_redirected(const Snapshot &snapshot) {
    if (snapshot.smoke.size() != 1U || snapshot.nodes.size() != 2U)
        return false;
    const auto &particle = snapshot.smoke[0];
    const auto &first = snapshot.nodes[0];
    const auto &second = snapshot.nodes[1];
    const PmFloat3 edge{second.position.x - first.position.x,
                        second.position.y - first.position.y,
                        second.position.z - first.position.z};
    const PmFloat3 offset{particle.position.x - first.position.x,
                          particle.position.y - first.position.y,
                          particle.position.z - first.position.z};
    const float edge_squared = edge.x * edge.x + edge.y * edge.y +
        edge.z * edge.z;
    const float fraction = std::clamp(
        (offset.x * edge.x + offset.y * edge.y + offset.z * edge.z) /
            edge_squared,
        0.0F, 1.0F);
    const PmFloat3 radial{
        offset.x - fraction * edge.x,
        offset.y - fraction * edge.y,
        offset.z - fraction * edge.z};
    const float distance = std::sqrt(radial.x * radial.x +
                                     radial.y * radial.y +
                                     radial.z * radial.z);
    if (distance < 1.0e-7F) return false;
    const PmFloat3 rope_velocity{
        first.velocity.x + fraction *
            (second.velocity.x - first.velocity.x),
        first.velocity.y + fraction *
            (second.velocity.y - first.velocity.y),
        first.velocity.z + fraction *
            (second.velocity.z - first.velocity.z)};
    const PmFloat3 relative{
        particle.velocity.x - rope_velocity.x,
        particle.velocity.y - rope_velocity.y,
        particle.velocity.z - rope_velocity.z};
    const float radial_speed = (relative.x * radial.x +
                                relative.y * radial.y +
                                relative.z * radial.z) / distance;
    return distance >= contact_clearance - 0.0005F &&
        std::abs(radial_speed) <= 0.001F;
}

[[nodiscard]] bool rope_unchanged(const Snapshot &snapshot) {
    bool unchanged = snapshot.nodes.size() == initial_rope_nodes.size();
    for (std::size_t index = 0U; index < snapshot.nodes.size(); ++index) {
        const auto &actual = snapshot.nodes[index];
        const auto &expected = initial_rope_nodes[index];
        unchanged = unchanged &&
            std::abs(actual.position.x - expected.x) <= 1.0e-5F &&
            std::abs(actual.position.y - expected.y) <= 1.0e-5F &&
            std::abs(actual.position.z - expected.z) <= 1.0e-5F &&
            std::abs(actual.velocity.x) <= 1.0e-5F &&
            std::abs(actual.velocity.y) <= 1.0e-5F &&
            std::abs(actual.velocity.z) <= 1.0e-5F;
    }
    return unchanged;
}

[[nodiscard]] std::string graph_fingerprint(
    const PmCouplingState *coupling, const Snapshot &probe) {
    std::ostringstream canonical;
    canonical << "parallel-mater-rope-smoke-contact-graph/v2"
              << ";contact-redirected=" << contact_redirected(probe)
              << ";rope-unchanged=" << rope_unchanged(probe);
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
    canonical << "parallel-mater-rope-smoke-contact-endpoints/v2"
              << ";smoke-count=" << snapshot.smoke.size()
              << ";contact-redirected=" << contact_redirected(snapshot)
              << ";rope-unchanged=" << rope_unchanged(snapshot);
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
        prefix + "-rope-smoke-contact-graph", coupling != nullptr,
        graph_generation, graph_fingerprint(coupling, endpoints)));
    resources.push_back(resource_json(
        prefix + "-rope-smoke-endpoints", true, 1U,
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
        throw std::runtime_error("unsupported: rope and smoke features");
    require_contact_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U, 10.0F);
    initialize_endpoints(context, fixture, quality_emission_rate,
                         {1.0F, 0.0F, 0.0F},
                         {1.0F, 0.0F, 0.0F}, -0.005F);
    // Replacement changes CONTACT itself. All clearances remain beyond the
    // shared -0.0145 m behavioral threshold. Bundled DRAG stays zero.
    const PmCouplingState baseline = contact_coupling(
        grid, authored_contact_distance);
    const PmCouplingState replacement = contact_coupling(
        grid, contact_clearance + 0.001F);
    const PmCouplingState recovery = contact_coupling(
        grid, contact_clearance + 0.002F);
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
                "rope-smoke lifecycle probe submission failed: " +
                std::to_string(probe_status));
        Snapshot snapshot = read_snapshot(context, fixture);
        if (snapshot.smoke.size() != 1U)
            throw std::runtime_error(
                "rope-smoke lifecycle probe lost its tracer");
        return snapshot;
    };
    submit_graph(&baseline, 1U, PM_SUCCESS, PM_SUCCESS);
    add_checkpoint(observations, "baseline", "PM_SUCCESS", &baseline, 1U,
                   run_probe(), grid);
    submit_graph(&replacement, 1U, PM_SUCCESS, PM_SUCCESS);
    add_checkpoint(observations, "replacement", "PM_SUCCESS", &replacement,
                   2U, run_probe(), grid);
    submit_graph(overflow.data(),
                 static_cast<std::uint32_t>(overflow.size()),
                 PM_ERROR_CAPACITY_EXCEEDED, PM_SUCCESS);
    add_checkpoint(observations, "rejected-overflow",
                   "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U,
                   run_probe(), grid);
    submit_graph(nullptr, 0U, PM_SUCCESS, PM_SUCCESS);
    add_checkpoint(observations, "removal", "PM_SUCCESS", nullptr, 3U,
                   run_probe(), grid);
    submit_graph(&recovery, 1U, PM_SUCCESS, PM_SUCCESS);
    add_checkpoint(observations, "recovery", "PM_SUCCESS", &recovery, 4U,
                   run_probe(), grid);
    DriverInfo driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}
} // namespace

std::string conformance_rope_smoke_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "rope-smoke-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/rope-smoke-contact-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json fixtures = Json::array();
        fixtures.push_back(quality_fixture_json(false, particle_driver));
        fixtures.push_back(quality_fixture_json(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "rope-smoke representations selected different devices");
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

std::string conformance_rope_smoke_contact_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "rope-smoke-contact-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/rope-smoke-contact-lifecycle-v1.json");
    try {
        LifecycleRun particle = run_lifecycle_fixture(false);
        LifecycleRun grid = run_lifecycle_fixture(true);
        if (!same_driver(particle.driver, grid.driver))
            throw std::runtime_error(
                "rope-smoke lifecycle representations selected different devices");
        Json cases = Json::array();
        const auto add_case = [&](const char *representation,
                                  const char *prefix, Json observations) {
            Json families = Json::array();
            families.push_back("rope");
            families.push_back("smoke");
            Json unrelated = Json::array();
            unrelated.push_back(std::string(prefix) +
                                "-rope-smoke-endpoints");
            Json attempt = Json::object();
            attempt["fixture"] = "rope-smoke-contact-public-lifecycle";
            attempt["representation"] = representation;
            attempt["status"] = "completed";
            Json item = Json::object();
            item["attempt"] = std::move(attempt);
            item["execution_status"] = "completed";
            item["families"] = std::move(families);
            item["id"] = std::string(prefix) +
                "-rope-smoke-contact-capacity-replacement-removal-rollback";
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
