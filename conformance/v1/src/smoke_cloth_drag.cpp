// SPDX-License-Identifier: MIT
// Public-SDK smoke-to-cloth drag quality and grid graph lifecycle fixtures.
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

constexpr float timestep = 0.01F;
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
constexpr float cloth_plane_z = 0.35F;
constexpr float cloth_vertex_mass = 0.25F;
constexpr std::array<PmFloat3, 3U> cloth_vertices{{
    {0.15F, 0.15F, cloth_plane_z},
    {0.35F, 0.15F, cloth_plane_z},
    {0.25F, 0.35F, cloth_plane_z},
}};
constexpr std::array<std::uint32_t, 3U> cloth_indices{{0U, 1U, 2U}};
constexpr std::array<float, 3U> cloth_inverse_masses{{
    1.0F / cloth_vertex_mass,
    1.0F / cloth_vertex_mass,
    1.0F / cloth_vertex_mass,
}};

struct DriverInfo {
    std::string name{};
    std::string device{};
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t version{};
};

struct Snapshot {
    std::array<PmClothVertexState, 3U> cloth{};
    std::vector<PmSmokeParticleState> smoke{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange cloth_output{};
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
        throw std::runtime_error("query smoke-cloth drag backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown smoke-cloth drag backend");
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

void require_drag_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_CLOTH, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_DRAG,
        PM_INTERACTION_DIRECTION_SECOND_TO_FIRST,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(std::string("unsupported: smoke-cloth drag ") +
                                 (grid ? "SMOKE_GRID" : "SMOKE_PARTICLE"));
}

[[nodiscard]] PmClothMaterialState cloth_material() {
    PmClothMaterialState material{};
    material.vertexMass = cloth_vertex_mass;
    material.thickness = 0.005F;
    material.stretchCompliance = 0.0F;
    material.bendingCompliance = 0.0F;
    material.velocityDamping = 0.0F;
    material.contactFriction = 0.0F;
    material.solverIterations = 1U;
    material.fracturePersistenceSubsteps = 1U;
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
    material.maximumSpeed = 4.0F;
    material.gridResolution = grid ? grid_resolution : 0U;
    material.gridVerticalResolution = grid ? grid_vertical_resolution : 0U;
    material.gridPressureIterations = grid ? 8U : 0U;
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
    edge.type = PM_COUPLING_SMOKE_CLOTH;
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
    constexpr PmClothCapacity cloth_capacity{3U, 3U, 3U};
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

    constexpr PmDeviceSize cloth_bytes = sizeof(PmDomainOutputHeader) +
        cloth_vertices.size() * sizeof(PmClothVertexState);
    constexpr PmDeviceSize smoke_bytes = sizeof(PmDomainOutputHeader) +
        smoke_capacity * sizeof(PmSmokeParticleState);
    fixture.output = context.buffer(
        cloth_bytes + smoke_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.cloth_output = {fixture.output.handle, 0U, cloth_bytes};
    fixture.smoke_output = {fixture.output.handle, cloth_bytes, smoke_bytes};
    return fixture;
}

void initialize_endpoints(Context &context, const FixtureObjects &fixture,
                          float smoke_velocity,
                          float particles_per_second) {
    const PmClothInfo cloth{
        PM_TRUE, 0U, static_cast<std::uint32_t>(cloth_vertices.size()),
        cloth_vertices.data(), static_cast<std::uint32_t>(cloth_indices.size()),
        cloth_indices.data(), cloth_inverse_masses.data()};
    const PmSmokeInfo smoke = smoke_info(smoke_velocity,
                                         particles_per_second);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, fixture.simulation, 0U, 1U,
                             &cloth) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopyClothState(context.commands, fixture.simulation, 0U,
                                &fixture.cloth_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySmokeState(context.commands, fixture.simulation, 0U,
                                &fixture.smoke_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *cloth_header =
        reinterpret_cast<const PmDomainOutputHeader *>(base);
    const auto *smoke_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.smoke_output.offset);
    if (cloth_header->activeCount != cloth_vertices.size() ||
        smoke_header->activeCount > smoke_capacity)
        throw std::runtime_error("smoke-cloth public output count mismatch");
    Snapshot result{};
    const auto *cloth = reinterpret_cast<const PmClothVertexState *>(
        cloth_header + 1);
    std::copy_n(cloth, result.cloth.size(), result.cloth.begin());
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
        throw std::runtime_error("unsupported: cloth and smoke features");
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
            throw std::runtime_error("smoke-cloth warmup failed: " +
                                     std::to_string(submitted));
    }
    Snapshot initial = capture(context, fixture);
    if (initial.smoke.empty())
        throw std::runtime_error("smoke-cloth carrier did not emit");

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
        throw std::runtime_error("smoke-cloth drag submission failed: " +
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
    Json cloth = Json::array();
    for (std::uint32_t index = 0U; index < snapshot.cloth.size(); ++index) {
        Json item = Json::object();
        item["index"] = static_cast<std::uint64_t>(index);
        item["position_m"] = vector_json(snapshot.cloth[index].position);
        item["velocity_m_s"] = vector_json(snapshot.cloth[index].velocity);
        cloth.push_back(std::move(item));
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
    result["cloth"] = std::move(cloth);
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
    QualityRun control = run_quality_fixture(grid, 0.0F, carrier_speed);
    QualityRun forward = run_quality_fixture(grid, test_drag, carrier_speed);
    QualityRun stationary = run_quality_fixture(grid, test_drag, 0.0F);
    QualityRun reverse = run_quality_fixture(grid, test_drag, -carrier_speed);
    for (const auto *run : {&forward, &stationary, &reverse})
        if (!same_driver(control.driver, run->driver))
            throw std::runtime_error(
                "paired smoke-cloth drag runs selected different devices");
    driver = control.driver;
    Json mode_flags = Json::array();
    mode_flags.push_back("CONTACT");
    mode_flags.push_back("DRAG");
    Json result = Json::object();
    result["id"] = grid ? "grid-smoke-cloth-drag"
                         : "particle-smoke-cloth-drag";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["warmup_step_count"] = static_cast<std::uint64_t>(warmup_steps);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    Json vertices = Json::array();
    for (const auto vertex : cloth_vertices)
        vertices.push_back(vector_json(vertex));
    result["cloth_vertices_m"] = std::move(vertices);
    Json indices = Json::array();
    for (const auto index : cloth_indices)
        indices.push_back(static_cast<std::uint64_t>(index));
    result["cloth_triangle_indices"] = std::move(indices);
    result["cloth_vertex_mass_kg"] = static_cast<double>(cloth_vertex_mass);
    result["cloth_thickness_m"] = 0.005;
    result["cloth_velocity_damping_per_s"] = 0.0;
    result["cloth_solver_iterations"] = static_cast<std::uint64_t>(1U);
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
    result["smoke_maximum_speed_m_s"] = 4.0;
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
        "paired-public-bidirectional-relative-flow-and-smoke-invariance";
    Json runs = Json::array();
    runs.push_back(run_json("control", control));
    runs.push_back(run_json("forward", forward));
    runs.push_back(run_json("stationary", stationary));
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
        return sha256("parallel-mater-grid-smoke-cloth-drag-graph/v1;inactive");
    const auto &smoke = edge->parameters.smoke;
    std::ostringstream canonical;
    canonical << "parallel-mater-grid-smoke-cloth-drag-graph/v1;type="
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
    canonical << "parallel-mater-grid-smoke-cloth-drag-endpoints/v1";
    for (const auto &state : snapshot.cloth) {
        canonical << ";cloth-position=";
        append_vector(canonical, state.position);
        canonical << ",velocity=";
        append_vector(canonical, state.velocity);
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

void add_checkpoint(Json &observations, std::string id, std::string outcome,
                    const PmCouplingState *edge,
                    std::uint64_t graph_generation,
                    const Snapshot &endpoints) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        "grid-smoke-cloth-drag-graph", edge != nullptr, graph_generation,
        graph_fingerprint(edge)));
    resources.push_back(resource_json(
        "grid-smoke-cloth-endpoints", true, 1U,
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

[[nodiscard]] LifecycleRun run_lifecycle_fixture() {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: cloth and smoke features");
    require_drag_capability(context, true);
    FixtureObjects fixture = create_fixture(context, true, 1U);
    // A positive emitter rate is required by the public descriptor. The five
    // 0.1 ms lifecycle steps remain below the first-particle emission boundary.
    initialize_endpoints(context, fixture, 0.0F, 1.0F);
    const PmCouplingState baseline = drag_edge(true, 10.0F);
    const PmCouplingState replacement = drag_edge(true, 20.0F);
    const PmCouplingState recovery = drag_edge(true, 30.0F);
    const std::array<PmCouplingState, 2U> overflow{{replacement, recovery}};
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    Json observations = Json::array();
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
    DriverInfo driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(observations)};
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}

} // namespace

std::string conformance_smoke_cloth_drag_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-cloth-drag-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/smoke-cloth-drag-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json fixtures = Json::array();
        fixtures.push_back(quality_fixture_json(false, particle_driver));
        fixtures.push_back(quality_fixture_json(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "smoke-cloth representations selected different devices");
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

std::string conformance_smoke_cloth_grid_drag_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id =
        "smoke-cloth-grid-drag-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/smoke-cloth-grid-drag-lifecycle-v1.json");
    try {
        LifecycleRun run = run_lifecycle_fixture();
        Json families = Json::array();
        families.push_back("cloth");
        families.push_back("smoke");
        Json unrelated = Json::array();
        unrelated.push_back("grid-smoke-cloth-endpoints");
        Json attempt = Json::object();
        attempt["fixture"] = "smoke-cloth-grid-drag-public-lifecycle";
        attempt["representation"] = "SMOKE_GRID";
        attempt["status"] = "completed";
        Json item = Json::object();
        item["attempt"] = std::move(attempt);
        item["execution_status"] = "completed";
        item["families"] = std::move(families);
        item["id"] =
            "grid-smoke-cloth-drag-capacity-replacement-removal-rollback";
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
