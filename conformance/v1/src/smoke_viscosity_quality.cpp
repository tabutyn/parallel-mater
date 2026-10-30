// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>
#include <parallel_mater_scene/command_scene.hpp>
#include <parallel_mater_scene/scene.hpp>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace parallel_mater::conformance {
namespace {
using namespace parallel_mater::gallery;

struct Fixture {
    const char *id{};
    const char *representation{};
    float timestep{};
    std::uint32_t initial_capture_step{};
    std::uint32_t final_capture_step{};
    SmokeOptions options{};
};

struct DriverInfo {
    std::string name{};
    std::string device{};
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t version{};
};

struct RunResult {
    DriverInfo driver{};
    std::vector<PmSmokeParticleState> initial{};
    std::vector<PmSmokeParticleState> final{};
};

[[nodiscard]] Json vector_json(PmFloat3 value) {
    Json result = Json::array();
    result.push_back(static_cast<double>(value.x));
    result.push_back(static_cast<double>(value.y));
    result.push_back(static_cast<double>(value.z));
    return result;
}

[[nodiscard]] Json vector_json(Vec3 value) {
    return vector_json(PmFloat3{value.x, value.y, value.z});
}

[[nodiscard]] Json particle_json(const PmSmokeParticleState &particle,
                                 std::uint32_t slot) {
    Json result = Json::object();
    result["slot"] = static_cast<std::uint64_t>(slot);
    result["position_m"] = vector_json(particle.position);
    result["velocity_m_s"] = vector_json(particle.velocity);
    result["age_seconds"] = static_cast<double>(particle.age);
    result["number_density"] = static_cast<double>(particle.numberDensity);
    result["pressure_m_s2"] = static_cast<double>(particle.pressure);
    result["vorticity_per_s"] = vector_json(particle.vorticity);
    return result;
}

[[nodiscard]] Json particles_json(
    const std::vector<PmSmokeParticleState> &particles) {
    Json result = Json::array();
    for (std::uint32_t slot = 0U; slot < particles.size(); ++slot)
        result.push_back(particle_json(particles[slot], slot));
    return result;
}

[[nodiscard]] const char *driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult result = pmGetNativeDevice(device, &native);
    if (result == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (result != PM_SUCCESS)
        throw std::runtime_error("query smoke quality backend");
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    throw std::runtime_error("unrecognized smoke quality backend");
}

void require_viscosity_capability(const CommandScene &simulation,
                                  bool grid) {
    const PmInteractionCapabilityKey key{
        PM_INTERACTION_FAMILY_SMOKE, PM_INTERACTION_FAMILY_SMOKE,
        PM_INTERACTION_SCOPE_SELF, PM_INTERACTION_MODE_VISCOSITY,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        grid ? PM_INTERACTION_MATERIAL_SMOKE_GRID
             : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE};
    PmInteractionCapability capability{};
    if (pmGetPhysicalDeviceInteractionCapability(
            simulation.physical_device(), &key, &capability) != PM_SUCCESS)
        throw std::runtime_error(
            std::string("advertised smoke self-viscosity capability is missing for ") +
            (grid ? "SMOKE_GRID" : "SMOKE_PARTICLE"));
    if (capability.maxGraphEdgesPerSimulation !=
        PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE)
        throw std::runtime_error(
            "intrinsic smoke self-viscosity unexpectedly consumes graph edges");
}

[[nodiscard]] RunResult run_fixture(const Fixture &fixture,
                                    bool viscous) {
    SceneDefinition scene{};
    scene.has_smoke = true;
    scene.gravity_scale = 0.0F;
    scene.smoke_options = fixture.options;
    if (fixture.options.grid_resolution == 0U)
        scene.smoke_options.viscosity = viscous ? 20.0F : 0.0F;
    else
        scene.smoke_options.grid_kinematic_viscosity =
            viscous ? 0.005F : 0.0F;

    CommandScene simulation{};
    CommandSceneInstance instance{};
    std::string error;
    if (!CommandScene::create(scene, 0U, simulation, instance, error))
        throw std::runtime_error("create smoke viscosity fixture: " + error);
    require_viscosity_capability(
        simulation, fixture.options.grid_resolution != 0U);

    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(simulation.physical_device(), &properties);
    RunResult result{};
    result.driver = {driver_name(simulation.device()), properties.deviceName,
                     properties.vendorId, properties.deviceId,
                     properties.driverVersion};
    for (std::uint32_t step = 1U; step <= fixture.final_capture_step; ++step) {
        CommandSceneFrame frame{};
        if (!simulation.step(fixture.timestep, 1U, Vec3{}, frame, error,
                             false, true, true))
            throw std::runtime_error("step smoke viscosity fixture: " + error);
        if (step == fixture.initial_capture_step)
            result.initial = frame.smoke_particles;
        if (step == fixture.final_capture_step)
            result.final = std::move(frame.smoke_particles);
    }
    if (result.initial.size() < 2U ||
        result.initial.size() != result.final.size())
        throw std::runtime_error(
            "smoke viscosity fixture did not retain its sampled particles");
    return result;
}

[[nodiscard]] SmokeOptions particle_options() {
    SmokeOptions options{};
    options.capacity = 2U;
    options.emitter_center = {};
    options.emitter_half_extents = {0.02F, 0.02F};
    options.initial_velocity = {};
    options.wind = {};
    options.particles_per_second = 10.0F;
    options.lifetime = 10.0F;
    options.particle_radius = 0.1F;
    options.buoyancy = 0.0F;
    options.response = 0.0F;
    options.rest_number_density = 1.0F;
    options.pressure_stiffness = 0.06F;
    options.viscosity = 0.0F;
    options.vorticity_confinement = 0.0F;
    options.maximum_speed = 8.9F;
    options.grid_resolution = 0U;
    options.grid_vertical_resolution = 8U;
    options.grid_pressure_iterations = 12U;
    options.grid_kinematic_viscosity = 0.0F;
    options.grid_les_coefficient = 0.0F;
    options.grid_pressure_tolerance = 0.001F;
    options.grid_minimum = {};
    options.grid_edge_length = 1.0F;
    return options;
}

[[nodiscard]] SmokeOptions grid_options() {
    SmokeOptions options{};
    options.capacity = 8U;
    options.emitter_center = {0.25F, 0.5F, 0.5F};
    options.emitter_half_extents = {0.2F, 0.2F};
    options.initial_velocity = {1.0F, 0.0F, 0.0F};
    options.wind = {};
    options.particles_per_second = 3.0F;
    options.lifetime = 10.0F;
    options.particle_radius = 0.03F;
    options.buoyancy = 0.0F;
    options.response = 0.0F;
    options.rest_number_density = 12.0F;
    options.pressure_stiffness = 0.0F;
    options.viscosity = 0.0F;
    options.vorticity_confinement = 0.0F;
    options.maximum_speed = 3.7F;
    options.grid_resolution = 16U;
    options.grid_vertical_resolution = 8U;
    options.grid_pressure_iterations = 12U;
    options.grid_kinematic_viscosity = 0.0F;
    options.grid_les_coefficient = 0.0F;
    options.grid_pressure_tolerance = 0.001F;
    options.grid_minimum = {};
    options.grid_edge_length = 1.0F;
    return options;
}

[[nodiscard]] Json fixture_inputs_json(const Fixture &fixture) {
    const SmokeOptions &options = fixture.options;
    Json result = Json::object();
    result["id"] = fixture.id;
    result["representation"] = fixture.representation;
    result["timestep_seconds"] = static_cast<double>(fixture.timestep);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["initial_capture_step"] =
        static_cast<std::uint64_t>(fixture.initial_capture_step);
    result["final_capture_step"] =
        static_cast<std::uint64_t>(fixture.final_capture_step);
    result["gravity_m_s2"] = vector_json(Vec3{});
    result["particle_capacity"] =
        static_cast<std::uint64_t>(options.capacity);
    result["expected_particle_count"] = static_cast<std::uint64_t>(
        options.grid_resolution == 0U ? 2U : 4U);
    result["emitter_center_m"] = vector_json(options.emitter_center);
    Json extents = Json::array();
    extents.push_back(static_cast<double>(options.emitter_half_extents.x));
    extents.push_back(static_cast<double>(options.emitter_half_extents.y));
    result["emitter_half_extents_m"] = std::move(extents);
    result["initial_velocity_m_s"] = vector_json(options.initial_velocity);
    result["wind_velocity_m_s"] = vector_json(options.wind);
    result["particles_per_second"] =
        static_cast<double>(options.particles_per_second);
    result["lifetime_seconds"] = static_cast<double>(options.lifetime);
    result["particle_radius_m"] =
        static_cast<double>(options.particle_radius);
    result["buoyancy_m_s2"] = static_cast<double>(options.buoyancy);
    result["wind_response_per_s"] = static_cast<double>(options.response);
    result["rest_number_density"] =
        static_cast<double>(options.rest_number_density);
    result["pressure_stiffness_m_s2"] =
        static_cast<double>(options.pressure_stiffness);
    result["control_viscosity_per_s"] = 0.0;
    result["test_viscosity_per_s"] =
        options.grid_resolution == 0U ? 20.0 : 0.0;
    result["vorticity_confinement_m_s2"] =
        static_cast<double>(options.vorticity_confinement);
    result["maximum_speed_m_s"] =
        static_cast<double>(options.maximum_speed);
    result["grid_resolution"] =
        static_cast<std::uint64_t>(options.grid_resolution);
    result["grid_vertical_resolution"] =
        static_cast<std::uint64_t>(options.grid_vertical_resolution);
    result["grid_pressure_iterations"] =
        static_cast<std::uint64_t>(options.grid_pressure_iterations);
    result["grid_control_kinematic_viscosity_m2_s"] = 0.0;
    result["grid_test_kinematic_viscosity_m2_s"] =
        options.grid_resolution == 0U ? 0.0 : 0.005;
    result["grid_les_coefficient"] =
        static_cast<double>(options.grid_les_coefficient);
    result["grid_pressure_tolerance_per_s"] =
        static_cast<double>(options.grid_pressure_tolerance);
    result["grid_minimum_m"] = vector_json(options.grid_minimum);
    result["grid_edge_length_m"] =
        static_cast<double>(options.grid_edge_length);
    result["oracle"] = options.grid_resolution == 0U
        ? "paired-public-smoke-particle-velocity-variance"
        : "paired-public-smoke-grid-tracer-velocity-variance";
    return result;
}

[[nodiscard]] Json run_json(const char *id, const RunResult &run) {
    Json result = Json::object();
    result["id"] = id;
    result["initial_particles"] = particles_json(run.initial);
    result["final_particles"] = particles_json(run.final);
    return result;
}

[[nodiscard]] Json fixture_json(const Fixture &fixture,
                                DriverInfo &driver) {
    RunResult control = run_fixture(fixture, false);
    RunResult viscous = run_fixture(fixture, true);
    if (control.driver.name != viscous.driver.name ||
        control.driver.device != viscous.driver.device ||
        control.driver.vendor_id != viscous.driver.vendor_id ||
        control.driver.device_id != viscous.driver.device_id ||
        control.driver.version != viscous.driver.version)
        throw std::runtime_error(
            "paired smoke viscosity runs selected different devices");
    driver = control.driver;
    Json result = fixture_inputs_json(fixture);
    Json runs = Json::array();
    runs.push_back(run_json("control", control));
    runs.push_back(run_json("viscous", viscous));
    result["runs"] = std::move(runs);
    return result;
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
} // namespace

std::string conformance_smoke_self_viscosity_quality(
    const std::filesystem::path &root) {
    const Fixture particle{"particle-pressure-seeded-relative-motion",
                           "SMOKE_PARTICLE", 1.0F / 60.0F, 13U, 17U,
                           particle_options()};
    const Fixture grid{"grid-driven-jet-relative-motion", "SMOKE_GRID",
                       1.0F / 60.0F, 81U, 99U, grid_options()};
    DriverInfo particle_driver{};
    DriverInfo grid_driver{};
    Json fixtures = Json::array();
    fixtures.push_back(fixture_json(particle, particle_driver));
    fixtures.push_back(fixture_json(grid, grid_driver));
    if (particle_driver.name != grid_driver.name ||
        particle_driver.device != grid_driver.device ||
        particle_driver.vendor_id != grid_driver.vendor_id ||
        particle_driver.device_id != grid_driver.device_id ||
        particle_driver.version != grid_driver.version)
        throw std::runtime_error(
            "smoke viscosity representations selected different devices");

    constexpr const char *profile_id = "smoke-self-viscosity-v1";
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality" /
        (std::string(profile_id) + ".json"));
    result["execution_status"] = "completed";
    result["driver"] = driver_json(particle_driver);
    result["fixtures"] = std::move(fixtures);
    return result.serialize();
}

} // namespace parallel_mater::conformance
