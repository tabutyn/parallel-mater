// Public-SDK fluid-smoke drag, heat, and graph lifecycle fixtures.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <array>
#include <bit>
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
constexpr std::uint32_t warmup_steps = 4U;
constexpr std::uint32_t smoke_capacity = 4U;
constexpr std::uint32_t grid_resolution = 16U;
constexpr std::uint32_t grid_vertical_resolution = 8U;
constexpr std::uint32_t grid_cells =
    grid_resolution * grid_vertical_resolution * grid_resolution;
constexpr float carrier_speed = 1.0F;
// Preserve the original 0.2 per-tick drag exponent after the fixed 60 Hz
// migration (20 /s at 0.01 s became 12 /s at 1/60 s).
constexpr float test_drag = 12.0F;
constexpr float heater_temperature = 120.0F;
constexpr float test_heat_transfer_rate = 20.0F;

struct DriverInfo {
    std::string name{};
    std::string device{};
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t version{};
};

struct Snapshot {
    PmFluidParticleOutput fluid{};
    std::vector<PmSmokeParticleState> smoke{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange fluid_output{};
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
        throw std::runtime_error("query smoke-fluid drag backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown smoke-fluid drag backend");
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

[[nodiscard]] PmPhysicalDeviceFeatures required_features() {
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.smoke = PM_TRUE;
    needed.domainCouplings = PM_TRUE;
    return needed;
}

void require_drag_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_DRAG,
        PM_INTERACTION_DIRECTION_SECOND_TO_FIRST,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(std::string("unsupported: smoke-fluid drag ") +
                                 (grid ? "SMOKE_GRID" : "SMOKE_PARTICLE"));
}

void require_thermal_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_THERMAL,
        PM_INTERACTION_DIRECTION_EXTERNAL_TO_FIRST,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(
            std::string("unsupported: externally heated fluid ") +
            (grid ? "SMOKE_GRID" : "SMOKE_PARTICLE"));
}

[[nodiscard]] PmLiquidMaterialState fluid_material() {
    PmLiquidMaterialState material{};
    material.restDensity = 1'000.0F;
    material.particleRadius = 0.025F;
    material.supportRadius = 0.10F;
    material.restParticleVolume = 0.001F;
    material.pressureStiffness = 4.0F;
    material.maximumSpeed = 2.9F;
    material.densityIterations = 1U;
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
    material.gridPressureTolerance = 1.0e-4F;
    return material;
}

[[nodiscard]] PmSmokeInfo smoke_info(float particles_per_second) {
    PmSmokeInfo info{};
    info.active = PM_TRUE;
    info.materialIndex = 0U;
    info.particleCapacity = smoke_capacity;
    info.emitterCenter = {0.25F, 0.25F, 0.5F};
    info.emitterHalfExtents = {0.01F, 0.01F};
    info.initialVelocity = {carrier_speed, 0.0F, 0.0F};
    info.wind = {carrier_speed, 0.0F, 0.0F};
    info.particlesPerSecond = particles_per_second;
    info.lifetime = 10.0F;
    info.gridMinimum = {0.0F, 0.0F, 0.0F};
    info.gridEdgeLength = 1.0F;
    return info;
}

[[nodiscard]] PmCouplingState drag_coupling(bool grid, float wind_drag) {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_LIQUID_SMOKE;
    coupling.first = 0U;
    coupling.second = 0U;
    // The current public edge descriptor groups the three independently
    // advertised fluid/smoke modes. Zero thermal and phase parameters isolate
    // the exact DRAG behavior qualified by this fixture.
    coupling.modeFlags = PM_INTERACTION_MODE_DRAG_BIT |
        PM_INTERACTION_MODE_THERMAL_BIT | PM_INTERACTION_MODE_PHASE_BIT;
    coupling.materialProfile = grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
                                    : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    coupling.parameters.thermal.heaterOrientation[3] = 1.0F;
    coupling.parameters.thermal.heaterHalfExtents[0] = 0.01F;
    coupling.parameters.thermal.heaterHalfExtents[1] = 0.01F;
    coupling.parameters.thermal.heaterTemperature = 20.0F;
    coupling.parameters.thermal.boilingTemperature = 1'000.0F;
    coupling.parameters.thermal.heatTransferRate = 0.0F;
    coupling.parameters.thermal.windDrag = wind_drag;
    coupling.parameters.thermal.steamRiseSpeed = 0.0F;
    return coupling;
}

[[nodiscard]] PmCouplingState thermal_coupling(
    bool grid, float heat_transfer_rate, float temperature) {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_LIQUID_SMOKE;
    coupling.first = 0U;
    coupling.second = 0U;
    // API 0.2 groups these independently advertised modes in one descriptor.
    // Zero drag and an unreachable boiling point isolate THERMAL.
    coupling.modeFlags = PM_INTERACTION_MODE_DRAG_BIT |
        PM_INTERACTION_MODE_THERMAL_BIT | PM_INTERACTION_MODE_PHASE_BIT;
    coupling.materialProfile = grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
                                    : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    coupling.parameters.thermal.heaterCenter[0] = 0.25F;
    coupling.parameters.thermal.heaterCenter[1] = 0.25F;
    coupling.parameters.thermal.heaterCenter[2] = 0.50F;
    coupling.parameters.thermal.heaterOrientation[3] = 1.0F;
    coupling.parameters.thermal.heaterHalfExtents[0] = 0.10F;
    coupling.parameters.thermal.heaterHalfExtents[1] = 0.10F;
    coupling.parameters.thermal.heaterTemperature = temperature;
    coupling.parameters.thermal.boilingTemperature = 1'000.0F;
    coupling.parameters.thermal.heatTransferRate = heat_transfer_rate;
    coupling.parameters.thermal.windDrag = 0.0F;
    coupling.parameters.thermal.steamRiseSpeed = 0.0F;
    return coupling;
}

[[nodiscard]] FixtureObjects create_fixture(Context &context, bool grid,
                                            std::uint32_t coupling_capacity) {
    constexpr PmFluidCapacity fluid_capacity{1U, 4U};
    const PmSmokeCapacity smoke_domain{
        smoke_capacity, grid ? grid_cells : 1U};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.fluidCount = 1U;
    create.pFluidCapacities = &fluid_capacity;
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

    const PmLiquidMaterialState liquid = fluid_material();
    const PmSmokeMaterialState smoke = smoke_material(grid);
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 1U;
    pipeline_info.pFluidMaterials = &liquid;
    pipeline_info.smokeMaterialCount = 1U;
    pipeline_info.pSmokeMaterials = &smoke;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &fixture.pipeline) == PM_SUCCESS);

    constexpr PmDeviceSize fluid_size = sizeof(PmDomainOutputHeader) +
        sizeof(PmFluidParticleOutput);
    constexpr PmDeviceSize smoke_size = sizeof(PmDomainOutputHeader) +
        smoke_capacity * sizeof(PmSmokeParticleState);
    fixture.output = context.buffer(
        fluid_size + smoke_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.fluid_output = {fixture.output.handle, 0U, fluid_size};
    fixture.smoke_output = {fixture.output.handle, fluid_size, smoke_size};
    return fixture;
}

void record_copies(Context &context, const FixtureObjects &fixture) {
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, fixture.simulation, 0U,
                                    &fixture.fluid_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySmokeState(context.commands, fixture.simulation, 0U,
                                &fixture.smoke_output) == PM_SUCCESS);
}

[[nodiscard]] Snapshot read_snapshot(Context &context,
                                     const FixtureObjects &fixture) {
    context.invalidate(fixture.output);
    const auto *base = static_cast<const std::byte *>(fixture.output.mapped);
    const auto *fluid_header =
        reinterpret_cast<const PmDomainOutputHeader *>(base);
    const auto *smoke_header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + fixture.smoke_output.offset);
    if (fluid_header->activeCount != 1U ||
        smoke_header->activeCount > smoke_capacity)
        throw std::runtime_error("smoke-fluid drag output count mismatch");
    Snapshot result{};
    result.fluid =
        reinterpret_cast<const PmFluidParticleOutput *>(fluid_header + 1)[0];
    const auto *smoke =
        reinterpret_cast<const PmSmokeParticleState *>(smoke_header + 1);
    result.smoke.assign(smoke, smoke + smoke_header->activeCount);
    return result;
}

[[nodiscard]] Buffer particle_input(Context &context, float velocity,
                                    float temperature,
                                    PmFloat3 position) {
    PmParticleState particle{};
    particle.position[0] = position.x;
    particle.position[1] = position.y;
    particle.position[2] = position.z;
    particle.velocity[0] = velocity;
    particle.temperatureCelsius = temperature;
    Buffer input = context.buffer(sizeof(particle), PM_BUFFER_PARTICLE_INPUT_BIT);
    *static_cast<PmParticleState *>(input.mapped) = particle;
    context.flush(input);
    return input;
}

[[nodiscard]] Buffer particle_input(Context &context, float velocity) {
    return particle_input(context, velocity, 20.0F,
                          {0.25F, 0.25F, 0.50F});
}

void set_fluid(Context &context, const FixtureObjects &fixture,
               const Buffer &input) {
    const PmBufferView view{{input.handle, 0U, sizeof(PmParticleState)},
                            PM_FORMAT_PARTICLE_STATE, 1U,
                            sizeof(PmParticleState)};
    PM_CHECK(pmCmdSetFluidParticles(context.commands, fixture.simulation, 0U,
                                    &view) == PM_SUCCESS);
}

[[nodiscard]] QualityRun run_quality_fixture(bool grid, float wind_drag,
                                             float fluid_velocity) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: fluid and smoke features");
    require_drag_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    Buffer input = particle_input(context, fluid_velocity);
    const PmSmokeInfo smoke = smoke_info(100.0F);
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input);
    PmResult warmup_status = context.submit();
    if (warmup_status != PM_SUCCESS)
        throw std::runtime_error("smoke-fluid initialization failed: " +
                                 std::to_string(warmup_status));
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    warmup_status = context.submit();
    if (warmup_status != PM_SUCCESS)
        throw std::runtime_error("smoke carrier initialization failed: " +
                                 std::to_string(warmup_status));
    for (std::uint32_t index = 0U; index < warmup_steps; ++index) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        warmup_status = context.submit();
        if (warmup_status != PM_SUCCESS)
            throw std::runtime_error(
                "smoke-fluid warmup step " + std::to_string(index) +
                " failed: " + std::to_string(warmup_status));
    }
    // Warm only the smoke carrier. Reset fluid to the immutable profile state.
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input);
    record_copies(context, fixture);
    warmup_status = context.submit();
    if (warmup_status != PM_SUCCESS)
        throw std::runtime_error("smoke-fluid warmup capture failed: " +
                                 std::to_string(warmup_status));
    Snapshot initial = read_snapshot(context, fixture);
    if (initial.smoke.empty())
        throw std::runtime_error("smoke-fluid drag carrier did not emit");

    const PmCouplingState coupling = drag_coupling(grid, wind_drag);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &coupling) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation, &step) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    const PmResult drag_status = context.submit();
    if (drag_status != PM_SUCCESS)
        throw std::runtime_error("smoke-fluid drag submission failed: " +
                                 std::to_string(drag_status));
    Snapshot final = read_snapshot(context, fixture);
    DriverInfo driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(initial), std::move(final)};
}

[[nodiscard]] QualityRun run_thermal_quality_fixture(
    bool grid, float heat_transfer_rate, float initial_temperature,
    PmFloat3 initial_position) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: fluid and smoke features");
    require_thermal_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    Buffer input = particle_input(context, 0.0F, initial_temperature,
                                  initial_position);
    const PmSmokeInfo smoke = smoke_info(100.0F);
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input);
    PmResult warmup_status = context.submit();
    if (warmup_status != PM_SUCCESS)
        throw std::runtime_error("thermal fluid initialization failed: " +
                                 std::to_string(warmup_status));
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    warmup_status = context.submit();
    if (warmup_status != PM_SUCCESS)
        throw std::runtime_error("thermal smoke initialization failed: " +
                                 std::to_string(warmup_status));
    for (std::uint32_t index = 0U; index < warmup_steps; ++index) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        warmup_status = context.submit();
        if (warmup_status != PM_SUCCESS)
            throw std::runtime_error(
                "thermal smoke warmup step " + std::to_string(index) +
                " failed: " + std::to_string(warmup_status));
    }
    // Warm only the retained smoke endpoint, then restore canonical fluid state.
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input);
    record_copies(context, fixture);
    warmup_status = context.submit();
    if (warmup_status != PM_SUCCESS)
        throw std::runtime_error("thermal warmup capture failed: " +
                                 std::to_string(warmup_status));
    Snapshot initial = read_snapshot(context, fixture);
    if (initial.smoke.empty())
        throw std::runtime_error("thermal fixture smoke endpoint did not emit");

    const PmCouplingState coupling = thermal_coupling(
        grid, heat_transfer_rate, heater_temperature);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &coupling) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation, &step) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    const PmResult thermal_status = context.submit();
    if (thermal_status != PM_SUCCESS)
        throw std::runtime_error("externally heated fluid submission failed: " +
                                 std::to_string(thermal_status));
    Snapshot final = read_snapshot(context, fixture);
    DriverInfo driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);
    return {std::move(driver), std::move(initial), std::move(final)};
}

[[nodiscard]] bool same_driver(const DriverInfo &first,
                               const DriverInfo &second) {
    return first.name == second.name && first.device == second.device &&
        first.vendor_id == second.vendor_id &&
        first.device_id == second.device_id && first.version == second.version;
}

[[nodiscard]] Json vector_json(const float value[3]) {
    Json result = Json::array();
    for (std::uint32_t axis = 0U; axis < 3U; ++axis)
        result.push_back(static_cast<double>(value[axis]));
    return result;
}

[[nodiscard]] Json vector_json(PmFloat3 value) {
    const float components[3]{value.x, value.y, value.z};
    return vector_json(components);
}

[[nodiscard]] Json snapshot_json(const Snapshot &snapshot) {
    Json fluid = Json::object();
    fluid["position_m"] = vector_json(snapshot.fluid.position);
    fluid["velocity_m_s"] = vector_json(snapshot.fluid.velocity);
    fluid["acceleration_m_s2"] = vector_json(snapshot.fluid.acceleration);
    fluid["temperature_celsius"] =
        static_cast<double>(snapshot.fluid.temperatureCelsius);
    fluid["stable_id"] = static_cast<std::uint64_t>(snapshot.fluid.stableId);
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
    result["fluid"] = std::move(fluid);
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
    QualityRun control = run_quality_fixture(grid, 0.0F, 0.0F);
    QualityRun drag = run_quality_fixture(grid, test_drag, 0.0F);
    QualityRun matched_control =
        run_quality_fixture(grid, 0.0F, carrier_speed);
    QualityRun matched_drag =
        run_quality_fixture(grid, test_drag, carrier_speed);
    for (const auto *run : {&drag, &matched_control, &matched_drag})
        if (!same_driver(control.driver, run->driver))
            throw std::runtime_error(
                "paired smoke-fluid drag runs selected different devices");
    driver = control.driver;
    Json mode_flags = Json::array();
    mode_flags.push_back("DRAG");
    mode_flags.push_back("THERMAL");
    mode_flags.push_back("PHASE");
    Json result = Json::object();
    result["id"] = grid ? "grid-carrier-drag" : "particle-carrier-drag";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["warmup_step_count"] = static_cast<std::uint64_t>(warmup_steps);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["fluid_initial_position_m"] =
        vector_json(PmFloat3{0.25F, 0.25F, 0.5F});
    result["fluid_initial_temperature_celsius"] = 20.0;
    result["fluid_particle_radius_m"] = 0.025;
    result["fluid_support_radius_m"] = 0.10;
    result["fluid_rest_density_kg_m3"] = 1'000.0;
    result["fluid_rest_particle_volume_m3"] = 0.001;
    result["smoke_particle_capacity"] =
        static_cast<std::uint64_t>(smoke_capacity);
    result["smoke_emitter_center_m"] =
        vector_json(PmFloat3{0.25F, 0.25F, 0.5F});
    Json emitter_half_extents = Json::array();
    emitter_half_extents.push_back(0.01);
    emitter_half_extents.push_back(0.01);
    result["smoke_emitter_half_extents_m"] =
        std::move(emitter_half_extents);
    result["smoke_emission_rate_per_s"] = 100.0;
    result["smoke_lifetime_seconds"] = 10.0;
    result["smoke_particle_radius_m"] = 0.10;
    result["smoke_rest_number_density"] = 1.0;
    result["smoke_maximum_speed_m_s"] = grid ? 3.7 : 4.0;
    result["smoke_grid_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_resolution : 0U);
    result["smoke_grid_vertical_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_vertical_resolution : 0U);
    result["smoke_grid_minimum_m"] = vector_json(PmFloat3{});
    result["smoke_grid_edge_length_m"] = 1.0;
    result["declared_mode_flags"] = std::move(mode_flags);
    result["exercised_mode"] = "DRAG";
    result["thermal_transfer_rate_per_s"] = 0.0;
    result["boiling_temperature_celsius"] = 1'000.0;
    result["carrier_velocity_m_s"] =
        vector_json(PmFloat3{carrier_speed, 0.0F, 0.0F});
    result["test_fluid_initial_velocity_m_s"] = vector_json(PmFloat3{});
    result["matched_fluid_initial_velocity_m_s"] =
        vector_json(PmFloat3{carrier_speed, 0.0F, 0.0F});
    result["control_wind_drag_per_s"] = 0.0;
    result["test_wind_drag_per_s"] = static_cast<double>(test_drag);
    result["oracle"] = "paired-public-relative-velocity-and-smoke-invariance";
    Json runs = Json::array();
    runs.push_back(run_json("control", control));
    runs.push_back(run_json("drag", drag));
    runs.push_back(run_json("matched-control", matched_control));
    runs.push_back(run_json("matched-drag", matched_drag));
    result["runs"] = std::move(runs);
    return result;
}

[[nodiscard]] Json thermal_quality_fixture_json(bool grid,
                                                DriverInfo &driver) {
    constexpr PmFloat3 inside{0.25F, 0.25F, 0.50F};
    constexpr PmFloat3 outside{0.50F, 0.25F, 0.50F};
    QualityRun control = run_thermal_quality_fixture(grid, 0.0F, 20.0F,
                                                     inside);
    QualityRun heated = run_thermal_quality_fixture(
        grid, test_heat_transfer_rate, 20.0F, inside);
    QualityRun equilibrium = run_thermal_quality_fixture(
        grid, test_heat_transfer_rate, heater_temperature, inside);
    QualityRun outside_run = run_thermal_quality_fixture(
        grid, test_heat_transfer_rate, 20.0F, outside);
    for (const auto *run : {&heated, &equilibrium, &outside_run})
        if (!same_driver(control.driver, run->driver))
            throw std::runtime_error(
                "paired externally heated fluid runs selected different devices");
    driver = control.driver;
    Json mode_flags = Json::array();
    mode_flags.push_back("DRAG");
    mode_flags.push_back("THERMAL");
    mode_flags.push_back("PHASE");
    Json result = Json::object();
    result["id"] = grid ? "grid-external-heating"
                        : "particle-external-heating";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["warmup_step_count"] = static_cast<std::uint64_t>(warmup_steps);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["fluid_initial_position_m"] = vector_json(inside);
    result["outside_fluid_position_m"] = vector_json(outside);
    result["fluid_initial_temperature_celsius"] = 20.0;
    result["equilibrium_temperature_celsius"] =
        static_cast<double>(heater_temperature);
    result["fluid_particle_radius_m"] = 0.025;
    result["fluid_support_radius_m"] = 0.10;
    result["fluid_rest_density_kg_m3"] = 1'000.0;
    result["fluid_rest_particle_volume_m3"] = 0.001;
    result["oracle_specific_heat_j_kg_k"] = 1.0;
    result["smoke_particle_capacity"] =
        static_cast<std::uint64_t>(smoke_capacity);
    result["smoke_emitter_center_m"] =
        vector_json(PmFloat3{0.25F, 0.25F, 0.50F});
    Json emitter_half_extents = Json::array();
    emitter_half_extents.push_back(0.01);
    emitter_half_extents.push_back(0.01);
    result["smoke_emitter_half_extents_m"] =
        std::move(emitter_half_extents);
    result["smoke_emission_rate_per_s"] = 100.0;
    result["smoke_lifetime_seconds"] = 10.0;
    result["smoke_particle_radius_m"] = 0.10;
    result["smoke_rest_number_density"] = 1.0;
    result["smoke_maximum_speed_m_s"] = grid ? 3.7 : 4.0;
    result["smoke_grid_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_resolution : 0U);
    result["smoke_grid_vertical_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_vertical_resolution : 0U);
    result["smoke_grid_minimum_m"] = vector_json(PmFloat3{});
    result["smoke_grid_edge_length_m"] = 1.0;
    result["declared_mode_flags"] = std::move(mode_flags);
    result["exercised_mode"] = "THERMAL";
    result["heater_center_m"] = vector_json(inside);
    Json heater_half_extents = Json::array();
    heater_half_extents.push_back(0.10);
    heater_half_extents.push_back(0.10);
    result["heater_half_extents_m"] = std::move(heater_half_extents);
    result["heater_temperature_celsius"] =
        static_cast<double>(heater_temperature);
    result["control_heat_transfer_rate_per_s"] = 0.0;
    result["test_heat_transfer_rate_per_s"] =
        static_cast<double>(test_heat_transfer_rate);
    result["wind_drag_per_s"] = 0.0;
    result["boiling_temperature_celsius"] = 1'000.0;
    result["oracle"] = "public-exponential-heater-and-endpoint-retention";
    Json runs = Json::array();
    runs.push_back(run_json("control", control));
    runs.push_back(run_json("heated", heated));
    runs.push_back(run_json("equilibrium", equilibrium));
    runs.push_back(run_json("outside", outside_run));
    result["runs"] = std::move(runs);
    return result;
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

void append_vector(std::ostringstream &stream, const float value[3]) {
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
        if (axis != 0U) stream << ',';
        append_float(stream, value[axis]);
    }
}

[[nodiscard]] std::string graph_fingerprint(
    const PmCouplingState *coupling) {
    if (coupling == nullptr)
        return sha256("parallel-mater-smoke-fluid-drag-graph/v1;inactive");
    std::ostringstream canonical;
    canonical << "parallel-mater-smoke-fluid-drag-graph/v1;type="
              << static_cast<unsigned>(coupling->type)
              << ";first=" << coupling->first
              << ";second=" << coupling->second
              << ";modes=" << coupling->modeFlags
              << ";direction=SECOND_TO_FIRST;material="
              << static_cast<unsigned>(coupling->materialProfile)
              << ";wind-drag=";
    append_float(canonical, coupling->parameters.thermal.windDrag);
    canonical << ";thermal-rate=";
    append_float(canonical, coupling->parameters.thermal.heatTransferRate);
    canonical << ";boiling=";
    append_float(canonical,
                 coupling->parameters.thermal.boilingTemperature);
    canonical << ";steam-rise=";
    append_float(canonical, coupling->parameters.thermal.steamRiseSpeed);
    return sha256(canonical.str());
}

[[nodiscard]] std::string thermal_graph_fingerprint(
    const PmCouplingState *coupling) {
    if (coupling == nullptr)
        return sha256(
            "parallel-mater-externally-heated-fluid-graph/v1;inactive");
    std::ostringstream canonical;
    canonical << "parallel-mater-externally-heated-fluid-graph/v1;type="
              << static_cast<unsigned>(coupling->type)
              << ";first=" << coupling->first
              << ";second=" << coupling->second
              << ";modes=" << coupling->modeFlags
              << ";direction=EXTERNAL_TO_FIRST;material="
              << static_cast<unsigned>(coupling->materialProfile)
              << ";heater-center=";
    append_vector(canonical, coupling->parameters.thermal.heaterCenter);
    canonical << ";heater-orientation=";
    for (std::uint32_t axis = 0U; axis < 4U; ++axis) {
        if (axis != 0U) canonical << ',';
        append_float(canonical,
                     coupling->parameters.thermal.heaterOrientation[axis]);
    }
    canonical << ";heater-half-extents=";
    for (std::uint32_t axis = 0U; axis < 2U; ++axis) {
        if (axis != 0U) canonical << ',';
        append_float(canonical,
                     coupling->parameters.thermal.heaterHalfExtents[axis]);
    }
    canonical << ";heater-temperature=";
    append_float(canonical,
                 coupling->parameters.thermal.heaterTemperature);
    canonical << ";thermal-rate=";
    append_float(canonical, coupling->parameters.thermal.heatTransferRate);
    canonical << ";boiling=";
    append_float(canonical,
                 coupling->parameters.thermal.boilingTemperature);
    canonical << ";wind-drag=";
    append_float(canonical, coupling->parameters.thermal.windDrag);
    canonical << ";steam-rise=";
    append_float(canonical, coupling->parameters.thermal.steamRiseSpeed);
    return sha256(canonical.str());
}

[[nodiscard]] std::string endpoint_fingerprint(const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-smoke-fluid-drag-endpoints/v1;fluid-position=";
    append_vector(canonical, snapshot.fluid.position);
    canonical << ";fluid-velocity=";
    append_vector(canonical, snapshot.fluid.velocity);
    canonical << ";fluid-temperature=";
    append_float(canonical, snapshot.fluid.temperatureCelsius);
    canonical << ";smoke-count=" << snapshot.smoke.size();
    for (std::size_t index = 0U; index < snapshot.smoke.size(); ++index) {
        canonical << ";smoke=" << index << ",position=";
        const float position[3]{snapshot.smoke[index].position.x,
                                snapshot.smoke[index].position.y,
                                snapshot.smoke[index].position.z};
        const float velocity[3]{snapshot.smoke[index].velocity.x,
                                snapshot.smoke[index].velocity.y,
                                snapshot.smoke[index].velocity.z};
        append_vector(canonical, position);
        canonical << ",velocity=";
        append_vector(canonical, velocity);
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
                    const PmCouplingState *coupling,
                    std::uint64_t graph_generation,
                    const Snapshot &endpoints) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        prefix + "-smoke-fluid-drag-graph", coupling != nullptr,
        graph_generation, graph_fingerprint(coupling)));
    resources.push_back(resource_json(
        prefix + "-fluid-smoke-endpoints", true, 1U,
        endpoint_fingerprint(endpoints)));
    Json checkpoint = Json::object();
    checkpoint["id"] = std::move(id);
    checkpoint["outcome"] = std::move(outcome);
    checkpoint["resources"] = std::move(resources);
    observations.push_back(std::move(checkpoint));
}

void add_thermal_checkpoint(Json &observations, const std::string &prefix,
                            std::string id, std::string outcome,
                            const PmCouplingState *coupling,
                            std::uint64_t graph_generation,
                            const Snapshot &endpoints) {
    Json resources = Json::array();
    resources.push_back(resource_json(
        prefix + "-externally-heated-fluid-graph", coupling != nullptr,
        graph_generation, thermal_graph_fingerprint(coupling)));
    resources.push_back(resource_json(
        prefix + "-fluid-smoke-endpoints", true, 1U,
        endpoint_fingerprint(endpoints)));
    Json checkpoint = Json::object();
    checkpoint["id"] = std::move(id);
    checkpoint["outcome"] = std::move(outcome);
    checkpoint["resources"] = std::move(resources);
    observations.push_back(std::move(checkpoint));
}

[[nodiscard]] Json lifecycle_case(bool grid, DriverInfo &driver) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: fluid and smoke features");
    require_drag_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    Buffer input = particle_input(context, 0.0F);
    const PmSmokeInfo smoke = smoke_info(1.0F);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmCouplingState baseline = drag_coupling(grid, 10.0F);
    const PmCouplingState replacement = drag_coupling(grid, 20.0F);
    const PmCouplingState recovery = drag_coupling(grid, 30.0F);
    const std::array<PmCouplingState, 2U> overflow{{replacement, recovery}};
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    Json observations = Json::array();
    const std::string prefix = grid ? "grid" : "particle";
    const auto mutate = [&](const PmCouplingState *couplings,
                            std::uint32_t count, PmResult expected) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, count,
                                  couplings) == expected);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        record_copies(context, fixture);
        PM_CHECK(context.submit() == PM_SUCCESS);
        return read_snapshot(context, fixture);
    };
    add_checkpoint(observations, prefix, "baseline", "PM_SUCCESS", &baseline,
                   1U, mutate(&baseline, 1U, PM_SUCCESS));
    add_checkpoint(observations, prefix, "replacement", "PM_SUCCESS",
                   &replacement, 2U,
                   mutate(&replacement, 1U, PM_SUCCESS));
    add_checkpoint(observations, prefix, "rejected-overflow",
                   "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U,
                   mutate(overflow.data(),
                          static_cast<std::uint32_t>(overflow.size()),
                          PM_ERROR_CAPACITY_EXCEEDED));
    add_checkpoint(observations, prefix, "removal", "PM_SUCCESS", nullptr,
                   3U, mutate(nullptr, 0U, PM_SUCCESS));
    add_checkpoint(observations, prefix, "recovery", "PM_SUCCESS", &recovery,
                   4U, mutate(&recovery, 1U, PM_SUCCESS));
    driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);

    Json families = Json::array();
    families.push_back("fluid");
    families.push_back("smoke");
    Json unrelated = Json::array();
    unrelated.push_back(prefix + "-fluid-smoke-endpoints");
    Json attempt = Json::object();
    attempt["fixture"] = "smoke-fluid-drag-public-lifecycle";
    attempt["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    attempt["status"] = "completed";
    Json result = Json::object();
    result["attempt"] = std::move(attempt);
    result["execution_status"] = "completed";
    result["families"] = std::move(families);
    result["id"] = prefix +
        "-smoke-fluid-drag-capacity-replacement-removal-rollback";
    result["observations"] = std::move(observations);
    result["unrelated_resources"] = std::move(unrelated);
    return result;
}

[[nodiscard]] Json thermal_lifecycle_case(bool grid, DriverInfo &driver) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: fluid and smoke features");
    require_thermal_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U);
    Buffer input = particle_input(context, 0.0F);
    const PmSmokeInfo smoke = smoke_info(1.0F);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    // Rate zero makes endpoint state an unrelated resource during graph
    // mutation. The analytic fixture independently exercises nonzero heating.
    const PmCouplingState baseline = thermal_coupling(grid, 0.0F, 80.0F);
    const PmCouplingState replacement = thermal_coupling(grid, 0.0F, 100.0F);
    const PmCouplingState recovery = thermal_coupling(grid, 0.0F, 120.0F);
    const std::array<PmCouplingState, 2U> overflow{{replacement, recovery}};
    const PmStepInfo step{1.0e-4F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    Json observations = Json::array();
    const std::string prefix = grid ? "grid" : "particle";
    const auto mutate = [&](const PmCouplingState *couplings,
                            std::uint32_t count, PmResult expected) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, count,
                                  couplings) == expected);
        PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation,
                                    &step) == PM_SUCCESS);
        record_copies(context, fixture);
        PM_CHECK(context.submit() == PM_SUCCESS);
        return read_snapshot(context, fixture);
    };
    add_thermal_checkpoint(observations, prefix, "baseline", "PM_SUCCESS",
                           &baseline, 1U,
                           mutate(&baseline, 1U, PM_SUCCESS));
    add_thermal_checkpoint(observations, prefix, "replacement", "PM_SUCCESS",
                           &replacement, 2U,
                           mutate(&replacement, 1U, PM_SUCCESS));
    add_thermal_checkpoint(
        observations, prefix, "rejected-overflow",
        "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U,
        mutate(overflow.data(), static_cast<std::uint32_t>(overflow.size()),
               PM_ERROR_CAPACITY_EXCEEDED));
    add_thermal_checkpoint(observations, prefix, "removal", "PM_SUCCESS",
                           nullptr, 3U, mutate(nullptr, 0U, PM_SUCCESS));
    add_thermal_checkpoint(observations, prefix, "recovery", "PM_SUCCESS",
                           &recovery, 4U,
                           mutate(&recovery, 1U, PM_SUCCESS));
    driver = driver_info(context);
    pmDestroySimulation(context.device, fixture.simulation);
    pmDestroyPhysicsPipeline(context.device, fixture.pipeline);

    Json families = Json::array();
    families.push_back("fluid");
    families.push_back("smoke");
    Json unrelated = Json::array();
    unrelated.push_back(prefix + "-fluid-smoke-endpoints");
    Json attempt = Json::object();
    attempt["fixture"] = "externally-heated-fluid-public-lifecycle";
    attempt["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    attempt["status"] = "completed";
    Json result = Json::object();
    result["attempt"] = std::move(attempt);
    result["execution_status"] = "completed";
    result["families"] = std::move(families);
    result["id"] = prefix +
        "-externally-heated-fluid-capacity-replacement-removal-rollback";
    result["observations"] = std::move(observations);
    result["unrelated_resources"] = std::move(unrelated);
    return result;
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}
} // namespace

std::string conformance_smoke_fluid_drag_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-fluid-drag-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/smoke-fluid-drag-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json fixtures = Json::array();
        fixtures.push_back(quality_fixture_json(false, particle_driver));
        fixtures.push_back(quality_fixture_json(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "smoke-fluid drag representations selected different devices");
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

std::string conformance_externally_heated_fluid_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "externally-heated-fluid-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/externally-heated-fluid-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json fixtures = Json::array();
        fixtures.push_back(thermal_quality_fixture_json(false,
                                                        particle_driver));
        fixtures.push_back(thermal_quality_fixture_json(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "thermal representations selected different devices");
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

std::string conformance_smoke_fluid_drag_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-fluid-drag-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/smoke-fluid-drag-lifecycle-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json cases = Json::array();
        cases.push_back(lifecycle_case(false, particle_driver));
        cases.push_back(lifecycle_case(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "smoke-fluid drag lifecycle selected different devices");
        result["execution_status"] = "completed";
        result["driver"] = driver_json(particle_driver);
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

std::string conformance_externally_heated_fluid_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id =
        "externally-heated-fluid-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/externally-heated-fluid-lifecycle-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json cases = Json::array();
        cases.push_back(thermal_lifecycle_case(false, particle_driver));
        cases.push_back(thermal_lifecycle_case(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "thermal lifecycle representations selected different devices");
        result["execution_status"] = "completed";
        result["driver"] = driver_json(particle_driver);
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
