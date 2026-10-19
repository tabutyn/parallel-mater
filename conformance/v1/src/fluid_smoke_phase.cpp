// Public-SDK fluid-to-smoke phase transfer quality and graph lifecycle fixtures.
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <algorithm>
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

constexpr float timestep = 0.01F;
constexpr std::uint32_t single_smoke_capacity = 1U;
constexpr std::uint32_t bounded_smoke_capacity = 2U;
constexpr std::uint32_t grid_resolution = 16U;
constexpr std::uint32_t grid_vertical_resolution = 8U;
constexpr std::uint32_t grid_cells =
    grid_resolution * grid_vertical_resolution * grid_resolution;
constexpr float source_temperature = 120.0F;
constexpr float control_boiling_temperature = 200.0F;
constexpr float transfer_boiling_temperature = 100.0F;

struct DriverInfo {
    std::string name{};
    std::string device{};
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t version{};
};

struct Snapshot {
    std::vector<PmFluidParticleOutput> fluid{};
    std::vector<PmSmokeParticleState> smoke{};
};

struct FixtureObjects {
    PmSimulation simulation{};
    PmPhysicsPipeline pipeline{};
    Buffer output{};
    PmBufferRange fluid_output{};
    PmBufferRange smoke_output{};
    std::uint32_t fluid_capacity{};
    std::uint32_t smoke_capacity{};
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
        throw std::runtime_error("query fluid-smoke phase backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unknown fluid-smoke phase backend");
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

void require_phase_capability(const Context &context, bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_DISTINCT_FAMILY, PM_INTERACTION_MODE_PHASE,
        PM_INTERACTION_DIRECTION_FIRST_TO_SECOND_TRANSFER,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            context.physical, &key, &capability) != PM_SUCCESS ||
        capability.maxGraphEdgesPerSimulation ==
            PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(std::string("unsupported: fluid-smoke phase ") +
                                 (grid ? "SMOKE_GRID" : "SMOKE_PARTICLE"));
}

[[nodiscard]] PmLiquidMaterialState fluid_material() {
    PmLiquidMaterialState material{};
    material.restDensity = 1'000.0F;
    material.particleRadius = 0.025F;
    material.supportRadius = 0.10F;
    material.restParticleVolume = 0.001F;
    material.pressureStiffness = 1.0F;
    material.maximumSpeed = 4.0F;
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
    material.maximumSpeed = 4.0F;
    material.gridResolution = grid ? grid_resolution : 0U;
    material.gridVerticalResolution = grid ? grid_vertical_resolution : 0U;
    material.gridPressureIterations = grid ? 8U : 0U;
    material.gridPressureTolerance = 1.0e-4F;
    return material;
}

[[nodiscard]] PmSmokeInfo smoke_info(std::uint32_t capacity) {
    PmSmokeInfo info{};
    info.active = PM_TRUE;
    info.materialIndex = 0U;
    info.particleCapacity = capacity;
    info.emitterCenter = {0.25F, 0.25F, 0.5F};
    info.emitterHalfExtents = {0.01F, 0.01F};
    info.initialVelocity = {};
    info.wind = {};
    // Public smoke domains require a positive emitter rate. This value is too
    // small to emit an unrelated particle during any one-step fixture run.
    info.particlesPerSecond = 0.001F;
    info.lifetime = 10.0F;
    info.gridMinimum = {0.0F, 0.0F, 0.0F};
    info.gridEdgeLength = 1.0F;
    return info;
}

[[nodiscard]] PmCouplingState phase_coupling(
    bool grid, float boiling_temperature, float steam_rise_speed = 0.0F) {
    PmCouplingState coupling{};
    coupling.type = PM_COUPLING_LIQUID_SMOKE;
    coupling.first = 0U;
    coupling.second = 0U;
    // The current public edge descriptor groups the three independently
    // advertised fluid/smoke modes. Zero heat transfer and drag isolate the
    // exact PHASE behavior qualified by this fixture.
    coupling.modeFlags = PM_INTERACTION_MODE_DRAG_BIT |
        PM_INTERACTION_MODE_THERMAL_BIT | PM_INTERACTION_MODE_PHASE_BIT;
    coupling.materialProfile = grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
                                    : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    coupling.parameters.thermal.heaterOrientation[3] = 1.0F;
    coupling.parameters.thermal.heaterHalfExtents[0] = 0.01F;
    coupling.parameters.thermal.heaterHalfExtents[1] = 0.01F;
    coupling.parameters.thermal.heaterTemperature = source_temperature;
    coupling.parameters.thermal.boilingTemperature = boiling_temperature;
    coupling.parameters.thermal.heatTransferRate = 0.0F;
    coupling.parameters.thermal.windDrag = 0.0F;
    coupling.parameters.thermal.steamRiseSpeed = steam_rise_speed;
    return coupling;
}

[[nodiscard]] FixtureObjects create_fixture(Context &context, bool grid,
                                            std::uint32_t fluid_particles,
                                            std::uint32_t smoke_particles,
                                            std::uint32_t coupling_capacity) {
    const PmFluidCapacity fluid_capacity{fluid_particles, 4U};
    const PmSmokeCapacity smoke_domain{
        smoke_particles, grid ? grid_cells : 1U};
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

    const PmDeviceSize fluid_size = sizeof(PmDomainOutputHeader) +
        fluid_particles * sizeof(PmFluidParticleOutput);
    const PmDeviceSize smoke_size = sizeof(PmDomainOutputHeader) +
        smoke_particles * sizeof(PmSmokeParticleState);
    fixture.output = context.buffer(
        fluid_size + smoke_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    fixture.fluid_output = {fixture.output.handle, 0U, fluid_size};
    fixture.smoke_output = {fixture.output.handle, fluid_size, smoke_size};
    fixture.fluid_capacity = fluid_particles;
    fixture.smoke_capacity = smoke_particles;
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
    if (fluid_header->activeCount > fixture.fluid_capacity ||
        smoke_header->activeCount > fixture.smoke_capacity)
        throw std::runtime_error("fluid-smoke phase output count mismatch");
    Snapshot result{};
    const auto *fluid =
        reinterpret_cast<const PmFluidParticleOutput *>(fluid_header + 1);
    result.fluid.assign(fluid, fluid + fluid_header->activeCount);
    const auto *smoke =
        reinterpret_cast<const PmSmokeParticleState *>(smoke_header + 1);
    result.smoke.assign(smoke, smoke + smoke_header->activeCount);
    return result;
}

[[nodiscard]] Buffer particle_input(
    Context &context, const std::vector<PmParticleState> &particles) {
    const PmDeviceSize size = particles.size() * sizeof(PmParticleState);
    Buffer input = context.buffer(size, PM_BUFFER_PARTICLE_INPUT_BIT);
    std::copy(particles.begin(), particles.end(),
              static_cast<PmParticleState *>(input.mapped));
    context.flush(input);
    return input;
}

void set_fluid(Context &context, const FixtureObjects &fixture,
               const Buffer &input, std::uint32_t count) {
    const PmBufferView view{{input.handle, 0U,
                             count * sizeof(PmParticleState)},
                            PM_FORMAT_PARTICLE_STATE, count,
                            sizeof(PmParticleState)};
    PM_CHECK(pmCmdSetFluidParticles(context.commands, fixture.simulation, 0U,
                                    &view) == PM_SUCCESS);
}

[[nodiscard]] QualityRun run_quality_fixture(
    bool grid, const std::vector<PmParticleState> &particles,
    std::uint32_t smoke_capacity, float boiling_temperature) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: fluid and smoke features");
    require_phase_capability(context, grid);
    FixtureObjects fixture = create_fixture(
        context, grid, static_cast<std::uint32_t>(particles.size()),
        smoke_capacity, 1U);
    Buffer input = particle_input(context, particles);
    const PmSmokeInfo smoke = smoke_info(smoke_capacity);
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input,
              static_cast<std::uint32_t>(particles.size()));
    PmResult submit_status = context.submit();
    if (submit_status != PM_SUCCESS)
        throw std::runtime_error("fluid phase source initialization failed: " +
                                 std::to_string(submit_status));
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    submit_status = context.submit();
    if (submit_status != PM_SUCCESS)
        throw std::runtime_error("smoke phase target initialization failed: " +
                                 std::to_string(submit_status));
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input,
              static_cast<std::uint32_t>(particles.size()));
    record_copies(context, fixture);
    submit_status = context.submit();
    if (submit_status != PM_SUCCESS)
        throw std::runtime_error("fluid-smoke phase initialization failed: " +
                                 std::to_string(submit_status));
    Snapshot initial = read_snapshot(context, fixture);

    const PmCouplingState coupling = phase_coupling(
        grid, boiling_temperature);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, fixture.simulation, 1U,
                              &coupling) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, fixture.simulation, &step) ==
             PM_SUCCESS);
    record_copies(context, fixture);
    submit_status = context.submit();
    if (submit_status != PM_SUCCESS)
        throw std::runtime_error("fluid-smoke phase submission failed: " +
                                 std::to_string(submit_status));
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
    Json fluid = Json::array();
    for (const auto &particle : snapshot.fluid) {
        Json item = Json::object();
        item["position_m"] = vector_json(particle.position);
        item["velocity_m_s"] = vector_json(particle.velocity);
        item["acceleration_m_s2"] = vector_json(particle.acceleration);
        item["temperature_celsius"] =
            static_cast<double>(particle.temperatureCelsius);
        item["stable_id"] = static_cast<std::uint64_t>(particle.stableId);
        fluid.push_back(std::move(item));
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
    PmParticleState single{};
    single.position[0] = 0.25F;
    single.position[1] = 0.25F;
    single.position[2] = 0.25F;
    single.velocity[0] = 0.375F;
    single.velocity[1] = -0.125F;
    single.velocity[2] = 0.25F;
    single.temperatureCelsius = source_temperature;
    const std::vector<PmParticleState> single_input{single};

    std::vector<PmParticleState> bounded_input(3U);
    constexpr std::array<float, 3U> positions{0.20F, 0.50F, 0.80F};
    for (std::uint32_t index = 0U; index < bounded_input.size(); ++index) {
        auto &particle = bounded_input[index];
        particle.position[0] = positions[index];
        particle.position[1] = 0.25F;
        particle.position[2] = 0.25F;
        particle.velocity[index] = 0.25F * static_cast<float>(index + 1U);
        particle.temperatureCelsius = source_temperature;
    }

    QualityRun control = run_quality_fixture(
        grid, single_input, single_smoke_capacity,
        control_boiling_temperature);
    QualityRun transfer = run_quality_fixture(
        grid, single_input, single_smoke_capacity,
        transfer_boiling_temperature);
    QualityRun bounded = run_quality_fixture(
        grid, bounded_input, bounded_smoke_capacity,
        transfer_boiling_temperature);
    for (const auto *run : {&transfer, &bounded})
        if (!same_driver(control.driver, run->driver))
            throw std::runtime_error(
                "fluid-smoke phase runs selected different devices");
    driver = control.driver;
    Json mode_flags = Json::array();
    mode_flags.push_back("DRAG");
    mode_flags.push_back("THERMAL");
    mode_flags.push_back("PHASE");
    Json result = Json::object();
    result["id"] = grid ? "grid-phase-transfer" : "particle-phase-transfer";
    result["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = vector_json(PmFloat3{});
    result["fluid_initial_temperature_celsius"] =
        static_cast<double>(source_temperature);
    result["fluid_particle_radius_m"] = 0.025;
    result["fluid_support_radius_m"] = 0.10;
    result["fluid_rest_density_kg_m3"] = 1'000.0;
    result["fluid_rest_particle_volume_m3"] = 0.001;
    result["fluid_particle_mass_kg"] = 1.0;
    result["single_source_position_m"] =
        vector_json(PmFloat3{0.25F, 0.25F, 0.25F});
    result["single_source_velocity_m_s"] =
        vector_json(PmFloat3{0.375F, -0.125F, 0.25F});
    result["single_smoke_particle_capacity"] =
        static_cast<std::uint64_t>(single_smoke_capacity);
    result["bounded_fluid_particle_count"] =
        static_cast<std::uint64_t>(bounded_input.size());
    result["bounded_smoke_particle_capacity"] =
        static_cast<std::uint64_t>(bounded_smoke_capacity);
    Json bounded_positions = Json::array();
    Json bounded_velocities = Json::array();
    for (const auto &particle : bounded_input) {
        bounded_positions.push_back(vector_json(particle.position));
        bounded_velocities.push_back(vector_json(particle.velocity));
    }
    result["bounded_source_positions_m"] = std::move(bounded_positions);
    result["bounded_source_velocities_m_s"] = std::move(bounded_velocities);
    result["smoke_emitter_center_m"] =
        vector_json(PmFloat3{0.25F, 0.25F, 0.5F});
    Json emitter_half_extents = Json::array();
    emitter_half_extents.push_back(0.01);
    emitter_half_extents.push_back(0.01);
    result["smoke_emitter_half_extents_m"] =
        std::move(emitter_half_extents);
    result["smoke_emission_rate_per_s"] = 0.001;
    result["smoke_lifetime_seconds"] = 10.0;
    result["smoke_particle_radius_m"] = 0.10;
    result["smoke_rest_number_density"] = 1.0;
    result["smoke_maximum_speed_m_s"] = 4.0;
    result["smoke_grid_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_resolution : 0U);
    result["smoke_grid_vertical_resolution"] =
        static_cast<std::uint64_t>(grid ? grid_vertical_resolution : 0U);
    result["smoke_grid_minimum_m"] = vector_json(PmFloat3{});
    result["smoke_grid_edge_length_m"] = 1.0;
    result["declared_mode_flags"] = std::move(mode_flags);
    result["exercised_mode"] = "PHASE";
    result["thermal_transfer_rate_per_s"] = 0.0;
    result["control_boiling_temperature_celsius"] =
        static_cast<double>(control_boiling_temperature);
    result["transfer_boiling_temperature_celsius"] =
        static_cast<double>(transfer_boiling_temperature);
    result["wind_drag_per_s"] = 0.0;
    result["steam_rise_speed_m_s"] = 0.0;
    result["oracle"] =
        "paired-public-count-stable-id-momentum-and-kinetic-energy";
    Json runs = Json::array();
    runs.push_back(run_json("below-threshold-control", control));
    runs.push_back(run_json("single-transfer", transfer));
    runs.push_back(run_json("capacity-bound", bounded));
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
        return sha256("parallel-mater-fluid-smoke-phase-graph/v1;inactive");
    std::ostringstream canonical;
    canonical << "parallel-mater-fluid-smoke-phase-graph/v1;type="
              << static_cast<unsigned>(coupling->type)
              << ";first=" << coupling->first
              << ";second=" << coupling->second
              << ";modes=" << coupling->modeFlags
              << ";direction=FIRST_TO_SECOND_TRANSFER;material="
              << static_cast<unsigned>(coupling->materialProfile)
              << ";thermal-rate=";
    append_float(canonical, coupling->parameters.thermal.heatTransferRate);
    canonical << ";boiling=";
    append_float(canonical,
                 coupling->parameters.thermal.boilingTemperature);
    canonical << ";steam-rise=";
    append_float(canonical, coupling->parameters.thermal.steamRiseSpeed);
    return sha256(canonical.str());
}

[[nodiscard]] std::string endpoint_fingerprint(const Snapshot &snapshot) {
    std::ostringstream canonical;
    canonical << "parallel-mater-fluid-smoke-phase-endpoints/v1;fluid-count="
              << snapshot.fluid.size();
    for (std::size_t index = 0U; index < snapshot.fluid.size(); ++index) {
        canonical << ";fluid=" << index << ",position=";
        append_vector(canonical, snapshot.fluid[index].position);
        canonical << ",velocity=";
        append_vector(canonical, snapshot.fluid[index].velocity);
        canonical << ",temperature=";
        append_float(canonical, snapshot.fluid[index].temperatureCelsius);
        canonical << ",stable-id=" << snapshot.fluid[index].stableId;
    }
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
        prefix + "-fluid-smoke-phase-graph", coupling != nullptr,
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

[[nodiscard]] Json lifecycle_case(bool grid, DriverInfo &driver) {
    Context context;
    if (!context.open(required_features()))
        throw std::runtime_error("unsupported: fluid and smoke features");
    require_phase_capability(context, grid);
    FixtureObjects fixture = create_fixture(context, grid, 1U, 1U, 1U);
    PmParticleState particle{};
    particle.position[0] = 0.25F;
    particle.position[1] = 0.25F;
    particle.position[2] = 0.25F;
    particle.temperatureCelsius = 20.0F;
    const std::vector<PmParticleState> particles{particle};
    Buffer input = particle_input(context, particles);
    const PmSmokeInfo smoke = smoke_info(1U);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input, 1U);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, fixture.simulation, 0U, 1U,
                            &smoke) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, fixture.pipeline) ==
             PM_SUCCESS);
    set_fluid(context, fixture, input, 1U);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmCouplingState baseline = phase_coupling(grid, 1'000.0F, 0.1F);
    const PmCouplingState replacement = phase_coupling(grid, 1'000.0F, 0.2F);
    const PmCouplingState recovery = phase_coupling(grid, 1'000.0F, 0.3F);
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
    attempt["fixture"] = "fluid-smoke-phase-public-lifecycle";
    attempt["representation"] = grid ? "SMOKE_GRID" : "SMOKE_PARTICLE";
    attempt["status"] = "completed";
    Json result = Json::object();
    result["attempt"] = std::move(attempt);
    result["execution_status"] = "completed";
    result["families"] = std::move(families);
    result["id"] = prefix +
        "-fluid-smoke-phase-capacity-replacement-removal-rollback";
    result["observations"] = std::move(observations);
    result["unrelated_resources"] = std::move(unrelated);
    return result;
}

[[nodiscard]] bool unsupported_error(const std::exception &error) {
    return std::string(error.what()).starts_with("unsupported:");
}
} // namespace

std::string conformance_fluid_smoke_phase_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "fluid-smoke-phase-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/fluid-smoke-phase-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json fixtures = Json::array();
        fixtures.push_back(quality_fixture_json(false, particle_driver));
        fixtures.push_back(quality_fixture_json(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "fluid-smoke phase representations selected different devices");
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

std::string conformance_fluid_smoke_phase_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "fluid-smoke-phase-lifecycle-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/fluid-smoke-phase-lifecycle-v1.json");
    try {
        DriverInfo particle_driver{};
        DriverInfo grid_driver{};
        Json cases = Json::array();
        cases.push_back(lifecycle_case(false, particle_driver));
        cases.push_back(lifecycle_case(true, grid_driver));
        if (!same_driver(particle_driver, grid_driver))
            throw std::runtime_error(
                "fluid-smoke phase lifecycle selected different devices");
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
