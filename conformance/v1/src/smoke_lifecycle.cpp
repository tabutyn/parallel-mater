// SPDX-License-Identifier: MIT
// Public-SDK smoke replacement, capacity, removal, rollback, and recovery.
#include "lifecycle_fixture.hpp"
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using parallel_mater::conformance::Json;
using parallel_mater::conformance::sha256;
using parallel_mater::test::Buffer;
using parallel_mater::test::Context;
using parallel_mater::test::LifecycleCheckpointObservation;
using parallel_mater::test::LifecycleResourceObservation;
using parallel_mater::test::SmokeLifecycleResult;

namespace {

constexpr std::uint32_t slot_capacity = 4U;
constexpr std::uint32_t grid_resolution = 16U;
constexpr std::uint32_t grid_vertical_resolution = 8U;
constexpr std::uint32_t grid_cells =
    grid_resolution * grid_vertical_resolution * grid_resolution;

const PmDomainOutputHeader *output_header(const Buffer &buffer,
                                          PmDeviceSize offset) {
    return reinterpret_cast<const PmDomainOutputHeader *>(
        static_cast<const std::byte *>(buffer.mapped) + offset);
}

const PmSmokeParticleState *output_particles(
    const PmDomainOutputHeader *header) {
    return reinterpret_cast<const PmSmokeParticleState *>(header + 1);
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

std::string state_fingerprint(bool grid, bool active,
                              const PmSmokeInfo *info,
                              const PmDomainOutputHeader *header) {
    std::ostringstream canonical;
    canonical << "parallel-mater-smoke-public-state/v1;representation="
              << (grid ? "grid" : "particle")
              << ";active=" << (active ? 1 : 0);
    if (active && info != nullptr) {
        canonical << ";material=" << info->materialIndex
                  << ";particle-capacity=" << info->particleCapacity
                  << ";emitter=";
        append_vector(canonical, info->emitterCenter);
        canonical << ";emitter-half=";
        append_float(canonical, info->emitterHalfExtents.x);
        canonical << ',';
        append_float(canonical, info->emitterHalfExtents.y);
        canonical << ";initial-velocity=";
        append_vector(canonical, info->initialVelocity);
        canonical << ";wind=";
        append_vector(canonical, info->wind);
        canonical << ";rate=";
        append_float(canonical, info->particlesPerSecond);
        canonical << ";lifetime=";
        append_float(canonical, info->lifetime);
        canonical << ";grid-minimum=";
        append_vector(canonical, info->gridMinimum);
        canonical << ";grid-edge=";
        append_float(canonical, info->gridEdgeLength);
    }
    if (header != nullptr) {
        canonical << ";count=" << header->activeCount
                  << ";capacity=" << header->capacity;
        const auto *particles = output_particles(header);
        for (std::uint32_t index = 0U; index < header->activeCount; ++index) {
            const auto &particle = particles[index];
            canonical << ";particle=" << index << ",position=";
            append_vector(canonical, particle.position);
            canonical << ",velocity=";
            append_vector(canonical, particle.velocity);
            canonical << ",age=";
            append_float(canonical, particle.age);
            canonical << ",density=";
            append_float(canonical, particle.numberDensity);
            canonical << ",pressure=";
            append_float(canonical, particle.pressure);
            canonical << ",vorticity=";
            append_vector(canonical, particle.vorticity);
        }
    }
    return sha256(canonical.str());
}

LifecycleResourceObservation resource_observation(
    std::string id, std::uint64_t generation, bool grid, bool active,
    const PmSmokeInfo *info, const PmDomainOutputHeader *header) {
    return {std::move(id), active, generation,
            state_fingerprint(grid, active, info, header)};
}

PmSmokeInfo smoke_info(float center_x, std::uint32_t capacity) {
    PmSmokeInfo info{};
    info.active = PM_TRUE;
    info.materialIndex = 0U;
    info.particleCapacity = capacity;
    info.emitterCenter = {center_x, 0.0F, 0.0F};
    info.emitterHalfExtents = {0.1F, 0.1F};
    info.initialVelocity = {0.25F, 0.0F, 0.0F};
    info.wind = {0.25F, 0.0F, 0.0F};
    info.particlesPerSecond = 1.0F;
    info.lifetime = 2.0F;
    info.gridMinimum = {-1.0F, -1.0F, -1.0F};
    info.gridEdgeLength = 2.0F;
    return info;
}

void record_checkpoint(SmokeLifecycleResult *result, bool grid,
                       std::string id, std::string outcome,
                       const PmSmokeInfo *target_info,
                       std::uint64_t target_generation, bool target_active,
                       const PmDomainOutputHeader *target,
                       const PmSmokeInfo &control_info,
                       const PmDomainOutputHeader *control) {
    if (result == nullptr) return;
    const std::string prefix = grid ? "grid" : "particle";
    LifecycleCheckpointObservation observation;
    observation.id = std::move(id);
    observation.outcome = std::move(outcome);
    observation.resources.push_back(resource_observation(
        prefix + "-target-smoke", target_generation, grid, target_active,
        target_info, target));
    observation.resources.push_back(resource_observation(
        prefix + "-control-smoke", 1U, grid, true, &control_info, control));
    result->observations.push_back(std::move(observation));
}

const char *native_driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult status = pmGetNativeDevice(device, &native);
    if (status == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    PM_CHECK(status == PM_SUCCESS);
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    default: throw std::runtime_error("unknown smoke lifecycle backend");
    }
}

Json resource_json(const LifecycleResourceObservation &resource) {
    Json value = Json::object();
    value["id"] = resource.id;
    value["active"] = resource.active;
    value["generation"] = resource.generation;
    value["state_sha256"] = resource.state_sha256;
    return value;
}

Json case_json(bool grid, const SmokeLifecycleResult &fixture) {
    const std::string representation = grid ? "grid" : "particle";
    Json families = Json::array();
    families.push_back("smoke");
    Json unrelated = Json::array();
    unrelated.push_back(representation + "-control-smoke");
    Json observations = Json::array();
    for (const auto &checkpoint : fixture.observations) {
        Json resources = Json::array();
        for (const auto &resource : checkpoint.resources)
            resources.push_back(resource_json(resource));
        Json value = Json::object();
        value["id"] = checkpoint.id;
        value["outcome"] = checkpoint.outcome;
        value["resources"] = std::move(resources);
        observations.push_back(std::move(value));
    }
    Json attempt = Json::object();
    attempt["fixture"] = "smoke-public-lifecycle";
    attempt["representation"] = representation;
    attempt["status"] = fixture.status == 0 ? "completed" :
        fixture.status == 77 ? "unsupported" : "failed";
    if (!fixture.reason.empty()) attempt["reason"] = fixture.reason;
    Json result = Json::object();
    result["attempt"] = std::move(attempt);
    result["execution_status"] = fixture.status == 0 ? "completed" :
        fixture.status == 77 ? "unsupported" : "failed";
    result["families"] = std::move(families);
    result["id"] = "smoke-" + representation +
        "-capacity-replacement-removal-rollback";
    result["observations"] = std::move(observations);
    result["unrelated_resources"] = std::move(unrelated);
    return result;
}

} // namespace

int parallel_mater_smoke_lifecycle_fixture(bool grid,
                                           SmokeLifecycleResult *result) {
    if (result != nullptr) *result = {};
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.smoke = PM_TRUE;
    if (!context.open(needed)) {
        if (result != nullptr) result->status = 77;
        return 77;
    }

    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    if (result != nullptr) {
        result->device = std::string(native_driver_name(context.device)) + "|" +
            properties.deviceName;
    }

    const PmSmokeCapacity capacity{
        slot_capacity, grid ? grid_cells : 1U};
    const std::array<PmSmokeCapacity, 2U> capacities{capacity, capacity};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.smokeCount = static_cast<std::uint32_t>(capacities.size());
    simulation_info.pSmokeCapacities = capacities.data();
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &simulation_info, &simulation) ==
             PM_SUCCESS);
    PmSimulationMemoryRequirements memory_requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation,
                                      &memory_requirements);
    const PmSimulationMemoryBinding simulation_memory{
        context.allocate(memory_requirements.persistent), 0U,
        context.allocate(memory_requirements.scratch), 0U};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation,
                                    &simulation_memory) == PM_SUCCESS);

    PmSmokeMaterialState material{};
    material.particleRadius = 0.08F;
    material.restNumberDensity = 8.0F;
    material.pressureStiffness = 2.0F;
    material.maximumSpeed = 4.0F;
    material.gridResolution = grid ? grid_resolution : 0U;
    material.gridVerticalResolution = grid ? grid_vertical_resolution : 0U;
    material.gridPressureIterations = grid ? 8U : 0U;
    material.gridPressureTolerance = 1.0e-4F;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.smokeMaterialCount = 1U;
    pipeline_info.pSmokeMaterials = &material;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const PmDeviceSize output_stride = sizeof(PmDomainOutputHeader) +
        slot_capacity * sizeof(PmSmokeParticleState);
    const auto output = context.buffer(
        output_stride * 2U,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange target_output{output.handle, 0U, output_stride};
    const PmBufferRange control_output{
        output.handle, output_stride, output_stride};

    const PmSmokeInfo baseline = smoke_info(-0.25F, 2U);
    const PmSmokeInfo replacement = smoke_info(0.25F, 3U);
    const PmSmokeInfo overflowing = smoke_info(0.75F, slot_capacity + 1U);
    const PmSmokeInfo recovery = smoke_info(-0.75F, 4U);
    const PmSmokeInfo control = smoke_info(1.5F, 1U);

    const auto copy_both = [&] {
        PM_CHECK(pmCmdCopySmokeState(context.commands, simulation, 0U,
                                    &target_output) == PM_SUCCESS);
        PM_CHECK(pmCmdCopySmokeState(context.commands, simulation, 1U,
                                    &control_output) == PM_SUCCESS);
    };
    const auto refresh_output = [&] { context.invalidate(output); };

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    const std::array<PmSmokeInfo, 2U> initial{baseline, control};
    PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U,
                            static_cast<std::uint32_t>(initial.size()),
                            initial.data()) == PM_SUCCESS);
    copy_both();
    PM_CHECK(context.submit() == PM_SUCCESS);
    refresh_output();
    record_checkpoint(result, grid, "baseline", "PM_SUCCESS", &baseline, 1U,
                      true, output_header(output, 0U), control,
                      output_header(output, output_stride));

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U, 1U,
                            &replacement) == PM_SUCCESS);
    copy_both();
    PM_CHECK(context.submit() == PM_SUCCESS);
    refresh_output();
    record_checkpoint(result, grid, "replacement", "PM_SUCCESS", &replacement,
                      2U, true, output_header(output, 0U), control,
                      output_header(output, output_stride));

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U, 1U,
                            &overflowing) == PM_ERROR_CAPACITY_EXCEEDED);
    copy_both();
    PM_CHECK(context.submit() == PM_SUCCESS);
    refresh_output();
    record_checkpoint(result, grid, "rejected-overflow",
                      "PM_ERROR_CAPACITY_EXCEEDED", &replacement, 2U, true,
                      output_header(output, 0U), control,
                      output_header(output, output_stride));

    PmSmokeInfo removal{};
    removal.active = PM_FALSE;
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U, 1U,
                            &removal) == PM_SUCCESS);
    PM_CHECK(pmCmdCopySmokeState(context.commands, simulation, 1U,
                                &control_output) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    refresh_output();
    record_checkpoint(result, grid, "removal", "PM_SUCCESS", nullptr, 3U,
                      false, nullptr, control,
                      output_header(output, output_stride));

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U, 1U,
                            &recovery) == PM_SUCCESS);
    copy_both();
    PM_CHECK(context.submit() == PM_SUCCESS);
    refresh_output();
    record_checkpoint(result, grid, "recovery", "PM_SUCCESS", &recovery, 4U,
                      true, output_header(output, 0U), control,
                      output_header(output, output_stride));

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    if (result != nullptr) result->status = 0;
    return 0;
}

namespace parallel_mater::conformance {

std::string conformance_smoke_pressure_lifecycle(
    const std::filesystem::path &root, ExecutionStatus *status) {
    SmokeLifecycleResult particle;
    SmokeLifecycleResult grid;
    try {
        (void)parallel_mater_smoke_lifecycle_fixture(false, &particle);
    } catch (const std::exception &error) {
        particle.status = 2;
        particle.reason = error.what();
    }
    try {
        (void)parallel_mater_smoke_lifecycle_fixture(true, &grid);
    } catch (const std::exception &error) {
        grid.status = 2;
        grid.reason = error.what();
    }
    const bool failed = particle.status == 2 || grid.status == 2;
    const bool unsupported = particle.status == 77 || grid.status == 77;
    const ExecutionStatus execution = failed ? ExecutionStatus::failed :
        unsupported ? ExecutionStatus::unsupported : ExecutionStatus::completed;
    if (status != nullptr) *status = execution;

    std::string provenance = !particle.device.empty() ? particle.device :
        grid.device;
    std::string driver_name = "unavailable";
    std::string device_name = "unavailable";
    const auto separator = provenance.find('|');
    if (separator != std::string::npos) {
        driver_name = provenance.substr(0U, separator);
        device_name = provenance.substr(separator + 1U);
    }
    Json driver = Json::object();
    driver["name"] = driver_name;
    driver["device"] = device_name;
    Json cases = Json::array();
    cases.push_back(case_json(false, particle));
    cases.push_back(case_json(true, grid));
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = "smoke-self-pressure-lifecycle-v1";
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/smoke-self-pressure-lifecycle-v1.json");
    result["execution_status"] = execution == ExecutionStatus::completed ?
        "completed" : execution == ExecutionStatus::unsupported ?
        "unsupported" : "failed";
    result["driver"] = std::move(driver);
    result["cases"] = std::move(cases);
    return result.serialize();
}

} // namespace parallel_mater::conformance
