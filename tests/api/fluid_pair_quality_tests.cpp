// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef PM_QUALITY_PROFILE_SHA256
#error PM_QUALITY_PROFILE_SHA256 is required
#endif
#ifndef PM_FLUID_SELF_CONTACT_PROFILE_SHA256
#error PM_FLUID_SELF_CONTACT_PROFILE_SHA256 is required
#endif
#ifndef PM_FLUID_SELF_PRESSURE_PROFILE_SHA256
#error PM_FLUID_SELF_PRESSURE_PROFILE_SHA256 is required
#endif

using parallel_mater::conformance::Json;
using parallel_mater::test::Context;

namespace {

enum class Fixture { pressure_contact, viscosity, self_contact, self_pressure };

[[nodiscard]] bool is_self_fixture(Fixture fixture) {
    return fixture == Fixture::self_contact ||
        fixture == Fixture::self_pressure;
}

struct RunResult {
    std::vector<PmFluidParticleOutput> first;
    std::vector<PmFluidParticleOutput> second;
};

[[nodiscard]] PmLiquidMaterialState material(Fixture fixture) {
    PmLiquidMaterialState result{};
    result.restDensity = 1000.0F;
    result.particleRadius = 0.1F;
    result.supportRadius = is_self_fixture(fixture) ? 0.4F : 1.0F;
    result.restParticleVolume = 0.008F;
    result.pressureStiffness =
        fixture != Fixture::viscosity ? 40.0F : 0.0F;
    result.kinematicViscosity = fixture == Fixture::viscosity ? 2.0F : 0.0F;
    result.normalDamping = fixture == Fixture::pressure_contact ? 2.0F : 0.0F;
    result.maximumSpeed = 100.0F;
    result.maximumPairAcceleration = 1000.0F;
    result.densityIterations = 1U;
    return result;
}

[[nodiscard]] std::vector<PmParticleState> particles(
    Fixture fixture, std::uint32_t ratio) {
    std::vector<PmParticleState> result(1U + ratio);
    result[0].position[0] =
        is_self_fixture(fixture) ? -0.05F : -0.4F;
    if (fixture == Fixture::viscosity) {
        result[0].velocity[0] = 1.0F;
    }
    for (std::uint32_t index = 0U; index < ratio; ++index) {
        auto &particle = result[1U + index];
        particle.position[0] =
            is_self_fixture(fixture)
                ? 0.05F
                : 0.15F + (fixture == Fixture::viscosity
                               ? 0.1F * static_cast<float>(index)
                               : 0.0F);
        particle.position[1] =
            fixture == Fixture::pressure_contact
                ? 0.05F * static_cast<float>(index)
                : 0.0F;
        if (fixture == Fixture::viscosity) {
            particle.velocity[0] = -1.0F / static_cast<float>(ratio);
        }
    }
    for (auto &particle : result) particle.temperatureCelsius = 20.0F;
    return result;
}

[[nodiscard]] std::vector<PmFluidParticleOutput> read_output(
    const void *mapped, std::uint32_t expected) {
    const auto *header = static_cast<const PmDomainOutputHeader *>(mapped);
    if (header->activeCount != expected)
        throw std::runtime_error("fluid output count mismatch");
    const auto *values = reinterpret_cast<const PmFluidParticleOutput *>(header + 1);
    std::vector<PmFluidParticleOutput> result(values, values + expected);
    std::sort(result.begin(), result.end(),
              [](const auto &left, const auto &right) {
                  return left.stableId < right.stableId;
              });
    for (std::uint32_t index = 0U; index < expected; ++index)
        if (result[index].stableId != index)
            throw std::runtime_error("fluid stable ID did not preserve input order");
    return result;
}

[[nodiscard]] RunResult run_once(Context &context, Fixture fixture,
                                 float timestep, std::uint32_t ratio,
                                 bool split, bool coupled = true) {
    const auto initial = particles(fixture, ratio);
    const auto source = context.buffer(
        initial.size() * sizeof(PmParticleState), PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(source.mapped, initial.data(),
                initial.size() * sizeof(PmParticleState));
    context.flush(source);

    const std::uint32_t fluid_count = split ? 2U : 1U;
    const PmFluidCapacity capacities[2]{
        {split ? 1U : 1U + ratio, 16U}, {ratio, 16U}};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.fluidCount = fluid_count;
    create.pFluidCapacities = capacities;
    create.couplingCapacity = split && coupled ? 1U : 0U;
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &create, &simulation) ==
             PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        context.allocate(requirements.persistent), 0U,
        context.allocate(requirements.scratch), 0U};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation, &binding) ==
             PM_SUCCESS);

    const auto fluid_material = material(fixture);
    const PmLiquidMaterialState materials[2]{fluid_material, fluid_material};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = fluid_count;
    pipeline_info.pFluidMaterials = materials;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const PmDeviceSize first_bytes = sizeof(PmDomainOutputHeader) +
        capacities[0].particleCapacity * sizeof(PmFluidParticleOutput);
    const PmDeviceSize second_bytes = split
        ? sizeof(PmDomainOutputHeader) +
              capacities[1].particleCapacity * sizeof(PmFluidParticleOutput)
        : 0U;
    const auto output = context.buffer(first_bytes + second_bytes,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferView first_view{
        {source.handle, 0U,
         (split ? 1U : 1U + ratio) * sizeof(PmParticleState)},
        PM_FORMAT_PARTICLE_STATE, split ? 1U : 1U + ratio,
        sizeof(PmParticleState)};
    const PmBufferView second_view{
        {source.handle, sizeof(PmParticleState),
         ratio * sizeof(PmParticleState)},
        PM_FORMAT_PARTICLE_STATE, ratio, sizeof(PmParticleState)};

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                    &first_view) == PM_SUCCESS);
    if (split) {
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 1U,
                                        &second_view) == PM_SUCCESS);
    }
    if (split && coupled) {
        PmCouplingState edge{};
        edge.type = PM_COUPLING_LIQUID_LIQUID;
        edge.first = 0U;
        edge.second = 1U;
        edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
            PM_INTERACTION_MODE_PRESSURE_BIT |
            PM_INTERACTION_MODE_VISCOSITY_BIT;
        edge.materialProfile = PM_INTERACTION_MATERIAL_EQUAL;
        edge.parameters.mechanical.twoWay = PM_TRUE;
        PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &edge) ==
                 PM_SUCCESS);
    }
    const PmStepInfo step{timestep, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    const PmBufferRange first_output{output.handle, 0U, first_bytes};
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                     &first_output) == PM_SUCCESS);
    if (split) {
        const PmBufferRange second_output{output.handle, first_bytes, second_bytes};
        PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 1U,
                                         &second_output) == PM_SUCCESS);
    }
    const PmResult submitted = context.submit();
    if (submitted != PM_SUCCESS)
        throw std::runtime_error("physical fluid-pair submission failed: " +
                                 std::to_string(submitted));

    RunResult result;
    context.invalidate(output);
    if (split) {
        result.first = read_output(output.mapped, 1U);
        result.second = read_output(
            static_cast<std::byte *>(output.mapped) + first_bytes, ratio);
    } else {
        const auto combined = read_output(output.mapped, 1U + ratio);
        result.first.assign(combined.begin(), combined.begin() + 1);
        result.second.assign(combined.begin() + 1, combined.end());
    }

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    return result;
}

[[nodiscard]] std::array<double, 3> vector(const float value[3]) {
    return {value[0], value[1], value[2]};
}

[[nodiscard]] Json vector_json(const std::array<double, 3> &value) {
    Json result = Json::array();
    for (double component : value) result.push_back(component);
    return result;
}

[[nodiscard]] Json zero_vector_json() {
    return vector_json({0.0, 0.0, 0.0});
}

[[nodiscard]] Json particle_json(const PmFluidParticleOutput &particle,
                                 std::string id) {
    Json result = Json::object();
    result["id"] = std::move(id);
    result["position"] = vector_json(vector(particle.position));
    result["velocity"] = vector_json(vector(particle.velocity));
    result["acceleration"] = vector_json(vector(particle.acceleration));
    result["temperature_celsius"] =
        static_cast<double>(particle.temperatureCelsius);
    result["foam"] = static_cast<double>(particle.foam);
    return result;
}

template <typename Particle>
[[nodiscard]] Json self_particle_json(const Particle &particle,
                                      std::uint32_t stable_id) {
    Json result = Json::object();
    result["stable_id"] = static_cast<std::uint64_t>(stable_id);
    result["position_m"] = vector_json(vector(particle.position));
    result["velocity_m_s"] = vector_json(vector(particle.velocity));
    return result;
}

[[nodiscard]] Json self_fixture(Context &context, Fixture fixture,
                                float timestep, const char *id,
                                const char *oracle) {
    const auto initial = particles(fixture, 1U);
    const auto run = run_once(context, fixture, timestep, 1U, false);
    const auto settings = material(fixture);

    Json initial_particles = Json::array();
    for (std::uint32_t index = 0U; index < initial.size(); ++index)
        initial_particles.push_back(
            self_particle_json(initial[index], index));
    Json final_particles = Json::array();
    final_particles.push_back(self_particle_json(run.first[0], 0U));
    final_particles.push_back(self_particle_json(run.second[0], 1U));

    Json result = Json::object();
    result["id"] = id;
    result["particle_count"] = static_cast<std::uint64_t>(initial.size());
    result["timestep_seconds"] = static_cast<double>(timestep);
    result["substeps"] = static_cast<std::uint64_t>(1U);
    result["gravity_m_s2"] = zero_vector_json();
    result["particle_radius_m"] =
        static_cast<double>(settings.particleRadius);
    result["support_radius_m"] =
        static_cast<double>(settings.supportRadius);
    result["rest_density_kg_m3"] =
        static_cast<double>(settings.restDensity);
    result["rest_particle_volume_m3"] =
        static_cast<double>(settings.restParticleVolume);
    result["pressure_stiffness"] =
        static_cast<double>(settings.pressureStiffness);
    result["kinematic_viscosity_m2_s"] =
        static_cast<double>(settings.kinematicViscosity);
    result["normal_damping"] =
        static_cast<double>(settings.normalDamping);
    result["maximum_speed_m_s"] =
        static_cast<double>(settings.maximumSpeed);
    result["maximum_pair_acceleration_m_s2"] =
        static_cast<double>(settings.maximumPairAcceleration);
    result["density_iterations"] =
        static_cast<std::uint64_t>(settings.densityIterations);
    result["initial_particles"] = std::move(initial_particles);
    result["final_particles"] = std::move(final_particles);
    result["oracle"] = oracle;
    if (fixture == Fixture::self_pressure)
        result["density_estimator"] =
            "particle-mass-per-isotropic-spacing-cubed";
    return result;
}

[[nodiscard]] Json self_contact_fixture(Context &context) {
    return self_fixture(context, Fixture::self_contact, 0.05F,
        "symmetric-two-particle-self-contact",
        "public-output-pair-distance-momentum-and-symmetry");
}

[[nodiscard]] Json self_pressure_fixture(Context &context) {
    return self_fixture(context, Fixture::self_pressure, 0.04F,
        "symmetric-compressed-pair-pressure-relaxation",
        "public-output-isotropic-packing-density-and-momentum");
}

[[nodiscard]] Json state_json(const RunResult &run) {
    Json result = Json::array();
    result.push_back(particle_json(run.first[0], "first/0"));
    for (std::size_t index = 0U; index < run.second.size(); ++index)
        result.push_back(particle_json(run.second[index],
                                      "second/" + std::to_string(index)));
    return result;
}

struct Aggregate {
    std::array<double, 3> linear{};
    std::array<double, 3> angular{};
    double energy{};
};

template <typename Particle, typename Position, typename Velocity>
[[nodiscard]] Aggregate aggregate(const std::vector<Particle> &values,
                                  double particle_mass,
                                  Position position, Velocity velocity) {
    Aggregate result;
    for (const auto &particle : values) {
        const auto p = position(particle);
        const auto v = velocity(particle);
        for (std::size_t axis = 0U; axis < 3U; ++axis)
            result.linear[axis] += particle_mass * v[axis];
        result.angular[0] += particle_mass * (p[1] * v[2] - p[2] * v[1]);
        result.angular[1] += particle_mass * (p[2] * v[0] - p[0] * v[2]);
        result.angular[2] += particle_mass * (p[0] * v[1] - p[1] * v[0]);
        result.energy += 0.5 * particle_mass *
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
    return result;
}

[[nodiscard]] Json system_json(const std::vector<PmParticleState> &before,
                               const std::vector<PmFluidParticleOutput> &after,
                               double particle_mass) {
    const auto initial = aggregate(before, particle_mass,
        [](const auto &value) { return vector(value.position); },
        [](const auto &value) { return vector(value.velocity); });
    const auto final = aggregate(after, particle_mass,
        [](const auto &value) { return vector(value.position); },
        [](const auto &value) { return vector(value.velocity); });
    Json result = Json::object();
    result["mass"] = particle_mass * static_cast<double>(before.size());
    result["linear_momentum_before"] = vector_json(initial.linear);
    result["linear_momentum_after"] = vector_json(final.linear);
    result["external_impulse"] = zero_vector_json();
    result["angular_momentum_before"] = vector_json(initial.angular);
    result["angular_momentum_after"] = vector_json(final.angular);
    result["external_angular_impulse"] = zero_vector_json();
    result["kinetic_energy_before"] = initial.energy;
    result["kinetic_energy_after"] = final.energy;
    result["external_work"] = 0.0;
    return result;
}

[[nodiscard]] Json sample(Context &context, Fixture fixture, double timestep,
                          std::uint32_t ratio) {
    const auto initial = particles(fixture, ratio);
    const auto split = run_once(context, fixture, static_cast<float>(timestep),
                                ratio, true);
    const auto isolated = run_once(context, fixture,
                                   static_cast<float>(timestep), ratio, true,
                                   false);
    const auto combined = run_once(context, fixture,
                                   static_cast<float>(timestep), ratio, false);
    double maximum_exchange = 0.0;
    const auto compare_exchange =
        [&maximum_exchange](const auto &coupled, const auto &control) {
        for (std::size_t index = 0U; index < coupled.size(); ++index)
            for (std::size_t axis = 0U; axis < 3U; ++axis)
                maximum_exchange = std::max(
                    maximum_exchange,
                    std::abs(
                        static_cast<double>(coupled[index].velocity[axis]) -
                        static_cast<double>(control[index].velocity[axis])));
        };
    compare_exchange(split.first, isolated.first);
    compare_exchange(split.second, isolated.second);
    if (maximum_exchange < 1.0e-3)
        throw std::runtime_error("fluid-pair fixture exchange is below 1e-3 m/s");
    const double particle_mass =
        static_cast<double>(material(fixture).restDensity) *
        static_cast<double>(material(fixture).restParticleVolume);

    Json systems = Json::array();
    systems.push_back(system_json(
        std::vector<PmParticleState>{initial[0]}, split.first, particle_mass));
    systems.push_back(system_json(
        std::vector<PmParticleState>(initial.begin() + 1, initial.end()),
        split.second, particle_mass));
    Json comparison = Json::object();
    comparison["support_radius_m"] =
        static_cast<double>(material(fixture).supportRadius);
    comparison["split"] = state_json(split);
    comparison["combined"] = state_json(combined);
    Json result = Json::object();
    result["timestep_seconds"] = timestep;
    result["mass_ratio"] = static_cast<double>(ratio);
    result["systems"] = std::move(systems);
    result["split_combined"] = std::move(comparison);
    return result;
}

[[nodiscard]] std::string driver_version(std::uint32_t version) {
    return std::to_string(version);
}

} // namespace

#ifdef PM_CONFORMANCE_EMBEDDED
int parallel_mater_fluid_pair_quality_fixture(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
    std::filesystem::path output;
    std::string driver_name = "selected-loadable-driver";
    bool self_contact = false;
    bool self_pressure = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--output" && index + 1 < argc)
            output = argv[++index];
        else if (argument == "--driver-name" && index + 1 < argc)
            driver_name = argv[++index];
        else if (argument == "--self-contact")
            self_contact = true;
        else if (argument == "--self-pressure")
            self_pressure = true;
        else {
            std::cerr << "usage: " << argv[0]
                      << " --output FILE [--driver-name NAME]"
                         " [--self-contact|--self-pressure]\n";
            return 2;
        }
    }
    if (output.empty() || (self_contact && self_pressure)) {
        std::cerr << "quality output path is required\n";
        return 2;
    }

    try {
        Context context;
        PmPhysicalDeviceFeatures needed{};
        needed.particleLiquids = PM_TRUE;
        const bool self_mode = self_contact || self_pressure;
        needed.liquidLiquidCoupling = self_mode ? PM_FALSE : PM_TRUE;
        if (!context.open(needed)) return 77;

        if (self_mode) {
            const PmInteractionCapabilityKey key{
                PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
                PM_INTERACTION_SCOPE_SELF,
                self_contact ? PM_INTERACTION_MODE_CONTACT
                             : PM_INTERACTION_MODE_PRESSURE,
                PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
                PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE};
            PmInteractionCapability capability{};
            if (pmGetPhysicalDeviceInteractionCapability(
                    context.physical, &key, &capability) != PM_SUCCESS)
                throw std::runtime_error(
                    "advertised fluid self mode is missing");
        } else {
            for (const auto mode : {PM_INTERACTION_MODE_CONTACT,
                                    PM_INTERACTION_MODE_PRESSURE,
                                    PM_INTERACTION_MODE_VISCOSITY}) {
                const PmInteractionCapabilityKey key{
                    PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
                    PM_INTERACTION_SCOPE_SEPARATE_SYSTEM, mode,
                    PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
                    PM_INTERACTION_MATERIAL_EQUAL};
                PmInteractionCapability capability{};
                if (pmGetPhysicalDeviceInteractionCapability(
                        context.physical, &key, &capability) != PM_SUCCESS)
                    throw std::runtime_error(
                        "advertised equal-fluid mode is missing");
            }
            for (const auto profile : {
                     PM_INTERACTION_MATERIAL_MISCIBLE,
                     PM_INTERACTION_MATERIAL_IMMISCIBLE_INTERFACE,
                     PM_INTERACTION_MATERIAL_THERMAL_PHASE}) {
                const PmInteractionCapabilityKey key{
                    PM_INTERACTION_FAMILY_FLUID,
                    PM_INTERACTION_FAMILY_FLUID,
                    PM_INTERACTION_SCOPE_SEPARATE_SYSTEM,
                    PM_INTERACTION_MODE_PRESSURE,
                    PM_INTERACTION_DIRECTION_BIDIRECTIONAL, profile};
                PmInteractionCapability capability{};
                if (pmGetPhysicalDeviceInteractionCapability(
                        context.physical, &key, &capability) !=
                    PM_ERROR_FEATURE_NOT_PRESENT)
                    throw std::runtime_error(
                        "unqualified fluid material profile was advertised");
            }
        }

        PmPhysicalDeviceProperties properties{};
        pmGetPhysicalDeviceProperties(context.physical, &properties);
        Json fixtures = Json::array();
        if (!self_mode) {
            for (const auto [fixture, id] : {
                     std::pair{Fixture::pressure_contact,
                               "equal-material-pressure-contact"},
                     std::pair{Fixture::viscosity,
                               "equal-material-viscosity"}}) {
                Json samples = Json::array();
                for (double timestep : {0.001, 0.01, 0.05})
                    for (std::uint32_t ratio : {1U, 2U, 4U})
                        samples.push_back(
                            sample(context, fixture, timestep, ratio));
                Json entry = Json::object();
                entry["id"] = id;
                entry["samples"] = std::move(samples);
                fixtures.push_back(std::move(entry));
            }
        }

        Json driver = Json::object();
        driver["name"] = driver_name;
        driver["device"] = properties.deviceName;
        driver["vendor_id"] = static_cast<std::uint64_t>(properties.vendorId);
        driver["device_id"] = static_cast<std::uint64_t>(properties.deviceId);
        driver["driver_version"] = driver_version(properties.driverVersion);
        Json result = Json::object();
        result["schema"] = "parallel-mater-conformance-quality-result/v1";
        result["profile_id"] = self_contact
            ? "fluid-self-contact-v1"
            : (self_pressure ? "fluid-self-pressure-v1"
                             : "core-conservation-stability-v1");
        result["profile_sha256"] = self_contact
            ? PM_FLUID_SELF_CONTACT_PROFILE_SHA256
            : (self_pressure ? PM_FLUID_SELF_PRESSURE_PROFILE_SHA256
                             : PM_QUALITY_PROFILE_SHA256);
        result["execution_status"] = "completed";
        result["driver"] = std::move(driver);
        if (self_contact)
            result["fixture"] = self_contact_fixture(context);
        else if (self_pressure)
            result["fixture"] = self_pressure_fixture(context);
        else
            result["fixtures"] = std::move(fixtures);

        if (!output.parent_path().empty())
            std::filesystem::create_directories(output.parent_path());
        std::ofstream stream(output, std::ios::binary);
        if (!stream) throw std::runtime_error("cannot open quality output");
        stream << result.serialize();
        if (!stream) throw std::runtime_error("cannot write quality output");
        std::cout << (self_contact
                          ? "PASS: public-API fluid self-contact sample on "
                          : (self_pressure
                                 ? "PASS: public-API fluid self-pressure sample on "
                                 : "PASS: 18 public-API equal-fluid quality samples on "))
                  << properties.deviceName << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "fluid-pair quality: " << error.what() << '\n';
        return 2;
    }
}
