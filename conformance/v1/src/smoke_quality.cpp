// SPDX-License-Identifier: MIT
#include "test_context.hpp"

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

constexpr float timestep = 0.015625F;
constexpr std::uint32_t substeps = 1U;
constexpr std::uint32_t capacity = 128U;
constexpr float particles_per_second = 2'048.0F;

[[nodiscard]] const char *driver_name(const CommandScene &simulation) {
    PmNativeDeviceInfo native{};
    const PmResult result = pmGetNativeDevice(simulation.device(), &native);
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
                                 std::uint32_t index) {
    Json result = Json::object();
    result["index"] = static_cast<std::uint64_t>(index);
    result["position_m"] = vector_json(particle.position);
    result["velocity_m_s"] = vector_json(particle.velocity);
    result["age_seconds"] = static_cast<double>(particle.age);
    result["number_density"] = static_cast<double>(particle.numberDensity);
    result["pressure"] = static_cast<double>(particle.pressure);
    result["vorticity_per_s"] = vector_json(particle.vorticity);
    return result;
}

[[nodiscard]] Json particles_json(
    const std::vector<PmSmokeParticleState> &particles) {
    Json result = Json::array();
    for (std::uint32_t index = 0U; index < particles.size(); ++index)
        result.push_back(particle_json(particles[index], index));
    return result;
}

[[nodiscard]] Json driver_json(const CommandScene &simulation) {
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(simulation.physical_device(), &properties);
    Json result = Json::object();
    result["name"] = driver_name(simulation);
    result["device"] = properties.deviceName;
    result["vendor_id"] = static_cast<std::uint64_t>(properties.vendorId);
    result["device_id"] = static_cast<std::uint64_t>(properties.deviceId);
    result["driver_version"] = std::to_string(properties.driverVersion);
    return result;
}

[[nodiscard]] SmokeOptions fixture_options(bool grid) {
    SmokeOptions smoke{};
    smoke.capacity = capacity;
    smoke.emitter_center = {};
    smoke.emitter_half_extents = {0.01F, 0.01F};
    smoke.initial_velocity = {1.0F, 0.0F, 0.0F};
    smoke.wind = {};
    smoke.particles_per_second = particles_per_second;
    smoke.lifetime = 2.0F;
    smoke.particle_radius = 0.1F;
    smoke.buoyancy = 0.0F;
    smoke.response = 0.0F;
    smoke.rest_number_density = 1.0F;
    smoke.pressure_stiffness = 4.0F;
    smoke.viscosity = 0.0F;
    smoke.vorticity_confinement = 0.0F;
    smoke.maximum_speed = 4.0F;
    smoke.grid_resolution = grid ? 16U : 0U;
    smoke.grid_vertical_resolution = grid ? 8U : 0U;
    smoke.grid_pressure_iterations = grid ? 32U : 0U;
    smoke.grid_kinematic_viscosity = 0.0F;
    smoke.grid_les_coefficient = 0.0F;
    smoke.grid_pressure_tolerance = 1.0e-4F;
    smoke.grid_minimum = {-0.5F, -0.25F, -0.5F};
    smoke.grid_edge_length = grid ? 1.0F : 0.0F;
    return smoke;
}

struct FixtureResult {
    Json fixture{};
    Json driver{};
};

[[nodiscard]] FixtureResult run_fixture(bool grid) {
    SceneDefinition scene{};
    scene.has_smoke = true;
    scene.gravity_scale = 0.0F;
    scene.smoke_options = fixture_options(grid);

    CommandScene simulation{};
    CommandSceneInstance instance{};
    std::string error;
    if (!CommandScene::create(scene, 0U, simulation, instance, error))
        throw std::runtime_error("create smoke pressure fixture: " + error);

    CommandSceneFrame first{};
    if (!simulation.step(timestep, substeps, {}, first, error,
                         false, true, true))
        throw std::runtime_error("step smoke pressure fixture (first): " + error);
    CommandSceneFrame second{};
    if (!simulation.step(timestep, substeps, {}, second, error,
                         false, true, true))
        throw std::runtime_error("step smoke pressure fixture (second): " + error);
    CommandSceneFrame final{};
    if (!simulation.step(timestep, substeps, {}, final, error,
                         false, true, true))
        throw std::runtime_error("step smoke pressure fixture (final): " + error);

    const SmokeOptions &smoke = scene.smoke_options;
    Json fixture = Json::object();
    fixture["id"] = grid ? "smoke-grid-pressure" : "smoke-particle-pressure";
    fixture["representation"] = grid ? "grid" : "particle";
    fixture["timestep_seconds"] = static_cast<double>(timestep);
    fixture["step_count"] = static_cast<std::uint64_t>(3U);
    fixture["substeps"] = static_cast<std::uint64_t>(substeps);
    fixture["gravity_m_s2"] = vector_json(Vec3{});
    fixture["capacity"] = static_cast<std::uint64_t>(smoke.capacity);
    fixture["emitter_center_m"] = vector_json(smoke.emitter_center);
    Json extents = Json::array();
    extents.push_back(static_cast<double>(smoke.emitter_half_extents.x));
    extents.push_back(static_cast<double>(smoke.emitter_half_extents.y));
    fixture["emitter_half_extents_m"] = std::move(extents);
    fixture["initial_velocity_m_s"] = vector_json(smoke.initial_velocity);
    fixture["wind_m_s"] = vector_json(smoke.wind);
    fixture["particles_per_second"] =
        static_cast<double>(smoke.particles_per_second);
    fixture["lifetime_seconds"] = static_cast<double>(smoke.lifetime);
    fixture["particle_radius_m"] = static_cast<double>(smoke.particle_radius);
    fixture["rest_number_density"] =
        static_cast<double>(smoke.rest_number_density);
    fixture["pressure_stiffness"] =
        static_cast<double>(smoke.pressure_stiffness);
    fixture["maximum_speed_m_s"] = static_cast<double>(smoke.maximum_speed);
    fixture["grid_resolution"] =
        static_cast<std::uint64_t>(smoke.grid_resolution);
    fixture["grid_vertical_resolution"] =
        static_cast<std::uint64_t>(smoke.grid_vertical_resolution);
    fixture["grid_pressure_iterations"] =
        static_cast<std::uint64_t>(smoke.grid_pressure_iterations);
    fixture["grid_pressure_tolerance"] =
        static_cast<double>(smoke.grid_pressure_tolerance);
    fixture["grid_minimum_m"] = vector_json(smoke.grid_minimum);
    fixture["grid_edge_length_m"] =
        static_cast<double>(smoke.grid_edge_length);
    fixture["first_particles"] = particles_json(first.smoke_particles);
    fixture["second_particles"] = particles_json(second.smoke_particles);
    fixture["final_particles"] = particles_json(final.smoke_particles);
    fixture["oracle"] = grid
        ? "public-tracer-samples-of-projected-grid-pressure"
        : "public-number-density-pressure-equation";
    return {std::move(fixture), driver_json(simulation)};
}

[[nodiscard]] Json unavailable_driver() {
    Json driver = Json::object();
    driver["name"] = "unavailable";
    driver["device"] = "unavailable";
    return driver;
}
} // namespace

std::string conformance_smoke_pressure_quality(
    const std::filesystem::path &root, ExecutionStatus *status) {
    constexpr const char *profile_id = "smoke-self-pressure-v1";
    parallel_mater::test::Context feature_probe;
    PmPhysicalDeviceFeatures needed{};
    needed.smoke = PM_TRUE;
    if (!feature_probe.open(needed)) {
        if (status != nullptr) *status = ExecutionStatus::unsupported;
        Json result = Json::object();
        result["schema"] = "parallel-mater-conformance-quality-result/v1";
        result["profile_id"] = profile_id;
        result["profile_sha256"] = sha256_contract_file(
            root / "conformance/v1/quality/smoke-self-pressure-v1.json");
        result["execution_status"] = "unsupported";
        result["driver"] = unavailable_driver();
        result["fixtures"] = Json::array();
        return result.serialize();
    }

    FixtureResult particle = run_fixture(false);
    FixtureResult grid = run_fixture(true);
    Json fixtures = Json::array();
    fixtures.push_back(std::move(particle.fixture));
    fixtures.push_back(std::move(grid.fixture));
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/quality/smoke-self-pressure-v1.json");
    result["execution_status"] = "completed";
    result["driver"] = std::move(particle.driver);
    result["fixtures"] = std::move(fixtures);
    if (status != nullptr) *status = ExecutionStatus::completed;
    return result.serialize();
}

} // namespace parallel_mater::conformance
