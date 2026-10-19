// SPDX-License-Identifier: MIT
// Public-SDK cloth-to-smoke contact quality and grid graph lifecycle fixtures.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

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
constexpr float cloth_vertex_mass = 0.25F;
constexpr float cloth_thickness = 0.002F;
constexpr std::array<PmFloat3, 4U> cloth_vertices{{
    {0.0F, 0.02F, 0.02F},
    {0.0F, 0.14F, 0.02F},
    {0.0F, 0.14F, 0.14F},
    {0.0F, 0.02F, 0.14F}}};
constexpr std::array<std::uint32_t, 6U> cloth_indices{{
    0U, 1U, 2U, 0U, 2U, 3U}};
constexpr std::array<float, 4U> dynamic_inverse_masses{{
    1.0F / cloth_vertex_mass, 1.0F / cloth_vertex_mass,
    1.0F / cloth_vertex_mass, 1.0F / cloth_vertex_mass}};
constexpr std::array<float, 4U> pinned_inverse_masses{{
    0.0F, 0.0F, 0.0F, 0.0F}};

struct DriverInfo {
    std::string name{};
    std::string device{};
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t version{};
};

struct Snapshot {
    std::vector<PmSmokeParticleState> smoke{};
    std::array<PmClothVertexState, cloth_vertices.size()> cloth{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange smoke_output{};
    PmBufferRange cloth_output{};
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
        throw std::runtime_error("query cloth-smoke contact backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown cloth-smoke contact backend");
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
    needed.cloth = PM_TRUE;
    needed.smoke = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

void require_contact_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_FIRST_TO_SECOND,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(std::string("unsupported: cloth-smoke contact ") +
                                 (grid ? "SMOKE_GRID" : "SMOKE_PARTICLE"));
}

[[nodiscard]] PmClothMaterialState cloth_material() {
    PmClothMaterialState material{};
    material.vertexMass = cloth_vertex_mass;
    material.thickness = cloth_thickness;
    material.stretchCompliance = 0.0F;
    material.bendingCompliance = 0.0F;
    material.velocityDamping = 0.0F;
    material.contactFriction = 0.0F;
    material.solverIterations = 2U;
    material.fracturePersistenceSubsteps = 1U;
    material.preserveVolume = PM_FALSE;
    material.volumeCompliance = 1.0e-7F;
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
    coupling.type = PM_COUPLING_SMOKE_CLOTH;
    // Public coupling storage names smoke first even though the advertised
    // semantic tuple is CLOTH -> SMOKE.
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
    constexpr PmClothCapacity cloth_capacity{4U, 6U, 6U};
    const PmSmokeCapacity smoke_domain{
        smoke_capacity, grid ? grid_cells : 1U};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.clothCount = 1U;
    create.pClothCapacities = &cloth_capacity;
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

    const PmClothMaterialState cloth = cloth_material();
    const PmSmokeMaterialState smoke = smoke_material(grid);
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &cloth;
    pipeline_info.smokeMaterialCount = 1U;
    pipeline_info.pSmokeMaterials = &smoke;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize smoke_size = sizeof(PmDomainOutputHeader) +
        smoke_capacity * sizeof(PmSmokeParticleState);
    constexpr PmDeviceSize cloth_size = sizeof(PmDomainOutputHeader) +
        cloth_vertices.size() * sizeof(PmClothVertexState);
    fixture.output = context.buffer(
        smoke_size + cloth_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.smoke_output = {fixture.output.handle, 0U, smoke_size};
    fixture.cloth_output = {fixture.output.handle, smoke_size, cloth_size};
    return fixture;
}

[[nodiscard]] PmClothInfo cloth_info(const float *inverse_masses) {
    return {PM_TRUE, 0U, static_cast<std::uint32_t>(cloth_vertices.size()),
            cloth_vertices.data(),
            static_cast<std::uint32_t>(cloth_indices.size()),
            cloth_indices.data(), inverse_masses};
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopySmokeState(context.commands, fixture.simulation, 0U,
                                &fixture.smoke_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyClothState(context.commands, fixture.simulation, 0U,
                                &fixture.cloth_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *smoke_header =
        reinterpret_cast<const PmDomainOutputHeader *>(base);
    const auto *cloth_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.cloth_output.offset);
    if (smoke_header->activeCount > smoke_capacity ||
        cloth_header->activeCount != cloth_vertices.size())
        throw std::runtime_error("cloth-smoke fixture output count mismatch");
    Snapshot result{};
    const auto *smoke =
        reinterpret_cast<const PmSmokeParticleState *>(smoke_header + 1);
    result.smoke.assign(smoke, smoke + smoke_header->activeCount);
    const auto *cloth =
        reinterpret_cast<const PmClothVertexState *>(cloth_header + 1);
    for (std::size_t index = 0U; index < result.cloth.size(); ++index)
        result.cloth[index] = cloth[index];
    return result;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture,
                          const float *inverse_masses,
                          float emission_rate, PmFloat3 initial_velocity,
                          PmFloat3 wind) {
    const PmClothInfo cloth = cloth_info(inverse_masses);
    const PmSmokeInfo smoke = smoke_info(
        emission_rate, initial_velocity, wind);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, fixture.simulation, 0U, 1U,
                             &cloth) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
}

[[nodiscard]] QualityRun run_quality_fixture(bool grid, bool coupled) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: cloth and smoke features");
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
                "cloth-smoke warmup failed: " +
                std::to_string(step_status));
    }
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    PM_CHECK(context.submit() == PM_SUCCESS);
    Snapshot initial = read_snapshot(context, fixture);
    if (initial.smoke.size() != 1U)
        throw std::runtime_error("cloth-smoke fixture did not emit one tracer");

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
            "cloth-smoke contact submission failed: " +
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
    for (const auto &vertex : snapshot.cloth) {
        vertices.push_back(vector_json(vertex.position));
        velocities.push_back(vector_json(vertex.velocity));
    }
    Json result = Json::object();
    result["smoke_particles"] = std::move(smoke);
    result["cloth_vertices_m"] = std::move(vertices);
    result["cloth_velocities_m_s"] = std::move(velocities);
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
    for (const auto vertex : cloth_vertices)
        result.push_back(vector_json(vertex));
    return result;
}

[[nodiscard]] Json indices_json() {
    Json result = Json::array();
    for (const auto index : cloth_indices)
        result.push_back(static_cast<std::uint64_t>(index));
    return result;
}

[[nodiscard]] Json quality_fixture_json(bool grid, DriverInfo &driver) {
    QualityRun control = run_quality_fixture(grid, false);
    QualityRun coupled = run_quality_fixture(grid, true);
    if (!same_driver(control.driver, coupled.driver))
        throw std::runtime_error(
            "cloth-smoke paired runs selected different devices");
    driver = control.driver;
    Json modes = Json::array();
    modes.push_back("CONTACT");
    modes.push_back("DRAG");
    Json runs = Json::array();
    runs.push_back(run_json("control", control));
    runs.push_back(run_json("coupled", coupled));
    Json result = Json::object();
    result["id"] = grid ? "grid-cloth-contact" : "particle-cloth-contact";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["warmup_step_count"] = static_cast<std::uint64_t>(warmup_steps);
    result["step_count"] = static_cast<std::uint64_t>(quality_steps);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["declared_mode_flags"] = std::move(modes);
    result["exercised_mode"] = "CONTACT";
    result["cloth_vertices_m"] = vertices_json();
    result["cloth_triangle_indices"] = indices_json();
    result["cloth_vertex_mass_kg"] =
        static_cast<double>(cloth_vertex_mass);
    result["cloth_thickness_m"] = static_cast<double>(cloth_thickness);
    result["cloth_dynamic"] = true;
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
        "paired-public-swept-plane-redirection-and-cloth-invariance";
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

[[nodiscard]] bool cloth_unchanged(const Snapshot &snapshot) {
    bool unchanged = snapshot.cloth.size() == cloth_vertices.size();
    for (std::size_t index = 0U; index < snapshot.cloth.size(); ++index) {
        const auto &actual = snapshot.cloth[index];
        const auto &expected = cloth_vertices[index];
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
    canonical << "parallel-mater-cloth-smoke-contact-graph/v2"
              << ";contact-redirected=" << contact_redirected(probe)
              << ";cloth-unchanged=" << cloth_unchanged(probe);
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
    canonical << "parallel-mater-cloth-smoke-contact-endpoints/v2"
              << ";smoke-count=" << snapshot.smoke.size()
              << ";contact-redirected=" << contact_redirected(snapshot)
              << ";cloth-unchanged=" << cloth_unchanged(snapshot);
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
        "grid-cloth-smoke-contact-graph", coupling != nullptr,
        graph_generation, graph_fingerprint(coupling, endpoints)));
    resources.push_back(resource_json(
        "grid-cloth-smoke-endpoints", true, 1U,
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
        throw std::runtime_error("unsupported: cloth and smoke features");
    require_contact_capability(context, true);
    FixtureObjects fixture = create_fixture(context, true, 1U);
    initialize_endpoints(context, fixture, pinned_inverse_masses.data(),
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
                "cloth-smoke lifecycle probe submission failed: " +
                std::to_string(probe_status));
        Snapshot snapshot = read_snapshot(context, fixture);
        if (snapshot.smoke.size() != 1U)
            throw std::runtime_error(
                "cloth-smoke lifecycle probe lost its tracer");
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

std::string conformance_cloth_smoke_contact_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "cloth-smoke-contact-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/cloth-smoke-contact-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json fixtures = Json::array();
        fixtures.push_back(quality_fixture_json(false, particle_driver));
        fixtures.push_back(quality_fixture_json(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "cloth-smoke representations selected different devices");
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

std::string conformance_cloth_smoke_contact_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "cloth-smoke-contact-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/cloth-smoke-contact-lifecycle-v1.json");
    try {
        LifecycleRun run = run_lifecycle_fixture();
        Json families = Json::array();
        families.push_back("cloth");
        families.push_back("smoke");
        Json unrelated = Json::array();
        Json attempt = Json::object();
        attempt["fixture"] = "cloth-smoke-contact-public-lifecycle";
        attempt["representation"] = "SMOKE_GRID";
        attempt["status"] = "completed";
        Json item = Json::object();
        item["attempt"] = std::move(attempt);
        item["execution_status"] = "completed";
        item["families"] = std::move(families);
        item["id"] =
            "grid-cloth-smoke-contact-capacity-replacement-removal-rollback";
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
