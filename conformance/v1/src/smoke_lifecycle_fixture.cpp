// SPDX-License-Identifier: MIT
// Public SDK only. Proves smoke particle/grid capacity, batched replacement,
// removal and deterministic recovery without inspecting driver-private state.
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
#include <string>
#include <utility>
#include <vector>

namespace parallel_mater::conformance {
namespace {
using test::Buffer;
using test::Context;

struct ResourceObservation {
    std::string id;
    bool active{};
    std::uint64_t generation{};
    std::string state_sha256;
};

struct CheckpointObservation {
    std::string id;
    std::string outcome;
    std::vector<ResourceObservation> resources;
};

struct SmokeLifecycleRun {
    ExecutionStatus status{ExecutionStatus::failed};
    std::string driver{"unavailable"};
    std::string device{"unavailable"};
    std::string reason{};
    std::vector<CheckpointObservation> observations;
};

[[nodiscard]] const char *driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult result = pmGetNativeDevice(device, &native);
    if (result == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (result != PM_SUCCESS) return "unknown";
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    }
    return "unknown";
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

void append_vector(std::ostringstream &stream, PmFloat3 value) {
    for (const float component : {value.x, value.y, value.z}) {
        stream << ',';
        append_float(stream, component);
    }
}

[[nodiscard]] std::string active_state_fingerprint(
    const PmDomainOutputHeader *header) {
    std::ostringstream canonical;
    canonical << "parallel-mater-smoke-public-state/v1;active=1;count="
              << header->activeCount << ";capacity=" << header->capacity;
    const auto *particles =
        reinterpret_cast<const PmSmokeParticleState *>(header + 1);
    for (std::uint32_t index = 0U; index < header->activeCount; ++index) {
        canonical << ";particle=" << index << ",position";
        append_vector(canonical, particles[index].position);
        canonical << ",velocity";
        append_vector(canonical, particles[index].velocity);
        canonical << ",age,";
        append_float(canonical, particles[index].age);
        canonical << ",density,";
        append_float(canonical, particles[index].numberDensity);
        canonical << ",pressure,";
        append_float(canonical, particles[index].pressure);
        canonical << ",vorticity";
        append_vector(canonical, particles[index].vorticity);
    }
    return sha256(canonical.str());
}

[[nodiscard]] std::string inactive_state_fingerprint() {
    return sha256("parallel-mater-smoke-public-state/v1;active=0");
}

[[nodiscard]] ResourceObservation active_resource(
    std::string id, std::uint64_t generation,
    const PmDomainOutputHeader *header) {
    return {std::move(id), true, generation,
            active_state_fingerprint(header)};
}

[[nodiscard]] PmSmokeMaterialState particle_material() {
    PmSmokeMaterialState material{};
    material.particleRadius = 0.1F;
    material.restNumberDensity = 1.0F;
    material.pressureStiffness = 20.0F;
    material.viscosity = 5.0F;
    material.maximumSpeed = 10.0F;
    return material;
}

[[nodiscard]] PmSmokeMaterialState grid_material(std::uint32_t resolution) {
    PmSmokeMaterialState material{};
    material.particleRadius = 0.03F;
    material.restNumberDensity = 12.0F;
    material.maximumSpeed = 4.0F;
    material.gridResolution = resolution;
    material.gridVerticalResolution = 8U;
    material.gridPressureIterations = 12U;
    material.gridKinematicViscosity = 0.005F;
    material.gridPressureTolerance = 0.001F;
    return material;
}

[[nodiscard]] PmSmokeInfo smoke_info(std::uint32_t material,
                                     PmFloat3 center,
                                     PmFloat3 velocity) {
    PmSmokeInfo smoke{};
    smoke.active = PM_TRUE;
    smoke.materialIndex = material;
    smoke.particleCapacity = 2U;
    smoke.emitterCenter = center;
    smoke.emitterHalfExtents = {0.08F, 0.08F};
    smoke.initialVelocity = velocity;
    smoke.particlesPerSecond = 200.0F;
    smoke.lifetime = 10.0F;
    smoke.gridMinimum = {0.0F, 0.0F, 0.0F};
    smoke.gridEdgeLength = 1.0F;
    return smoke;
}

[[nodiscard]] SmokeLifecycleRun execute_smoke_lifecycle() {
    SmokeLifecycleRun result;
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.smoke = PM_TRUE;
    if (!context.open(needed)) {
        result.status = ExecutionStatus::unsupported;
        return result;
    }
    result.driver = driver_name(context.device);
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    result.device = properties.deviceName;

    constexpr std::uint32_t grid_cells = 16U * 16U * 8U;
    constexpr std::array<PmSmokeCapacity, 2> capacities{{
        {2U, grid_cells}, {2U, grid_cells}}};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.smokeCount = static_cast<std::uint32_t>(capacities.size());
    simulation_info.pSmokeCapacities = capacities.data();
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &simulation_info, &simulation) ==
             PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        context.allocate(requirements.persistent), 0U,
        context.allocate(requirements.scratch), 0U};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation, &binding) ==
             PM_SUCCESS);

    const std::array<PmSmokeMaterialState, 3> materials{{
        particle_material(), grid_material(16U), grid_material(32U)}};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.smokeMaterialCount =
        static_cast<std::uint32_t>(materials.size());
    pipeline_info.pSmokeMaterials = materials.data();
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    constexpr PmDeviceSize smoke_output_size =
        sizeof(PmDomainOutputHeader) + 2U * sizeof(PmSmokeParticleState);
    const Buffer output = context.buffer(
        2U * smoke_output_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const std::array<PmBufferRange, 2> output_ranges{{
        {output.handle, 0U, smoke_output_size},
        {output.handle, smoke_output_size, smoke_output_size}}};
    const PmStepInfo step{0.01F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    const std::array<PmSmokeInfo, 2> baseline{{
        smoke_info(0U, {0.25F, 0.40F, 0.40F}, {0.5F, 0.0F, 0.0F}),
        smoke_info(1U, {0.25F, 0.60F, 0.60F}, {1.0F, 0.0F, 0.0F})}};
    const std::array<PmSmokeInfo, 2> replacement{{
        smoke_info(0U, {0.35F, 0.40F, 0.40F}, {-0.5F, 0.0F, 0.0F}),
        smoke_info(1U, {0.35F, 0.60F, 0.60F}, {0.0F, 0.5F, 0.0F})}};

    const auto copy_resources = [&](std::string id, std::string outcome,
                                    std::uint64_t generation) {
        context.begin();
        for (std::uint32_t index = 0U; index < output_ranges.size(); ++index)
            PM_CHECK(pmCmdCopySmokeState(context.commands, simulation, index,
                                         &output_ranges[index]) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
        context.invalidate(output);
        CheckpointObservation checkpoint{std::move(id), std::move(outcome), {}};
        for (std::uint32_t index = 0U; index < output_ranges.size(); ++index) {
            const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(
                static_cast<const std::byte *>(output.mapped) +
                index * smoke_output_size);
            PM_CHECK(header->activeCount == 1U);
            PM_CHECK(header->capacity == capacities[index].particleCapacity);
            checkpoint.resources.push_back(active_resource(
                index == 0U ? "particle-smoke" : "grid-smoke",
                generation, header));
        }
        result.observations.push_back(std::move(checkpoint));
    };
    const auto install_and_step = [&](const std::array<PmSmokeInfo, 2> &smokes,
                                      std::string id,
                                      std::uint64_t generation) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U,
                                 static_cast<std::uint32_t>(smokes.size()),
                                 smokes.data()) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
        copy_resources(std::move(id), "PM_SUCCESS", generation);
    };

    install_and_step(baseline, "baseline", 1U);
    install_and_step(replacement, "replacement", 2U);

    auto particle_overflow = replacement;
    particle_overflow[0].particleCapacity = 3U;
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U,
                             static_cast<std::uint32_t>(particle_overflow.size()),
                             particle_overflow.data()) ==
             PM_ERROR_CAPACITY_EXCEEDED);
    PM_CHECK(context.submit() == PM_SUCCESS);
    copy_resources("particle-capacity-rejection",
                   "PM_ERROR_CAPACITY_EXCEEDED", 2U);

    auto grid_overflow = replacement;
    grid_overflow[1].materialIndex = 2U;
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U,
                             static_cast<std::uint32_t>(grid_overflow.size()),
                             grid_overflow.data()) ==
             PM_ERROR_CAPACITY_EXCEEDED);
    PM_CHECK(context.submit() == PM_SUCCESS);
    copy_resources("grid-capacity-rejection",
                   "PM_ERROR_CAPACITY_EXCEEDED", 2U);

    std::array<PmSmokeInfo, 2> removed{};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0U,
                             static_cast<std::uint32_t>(removed.size()),
                             removed.data()) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    for (std::uint32_t index = 0U; index < output_ranges.size(); ++index) {
        context.begin();
        PM_CHECK(pmCmdCopySmokeState(context.commands, simulation, index,
                                     &output_ranges[index]) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_ERROR_INVALID_STATE);
    }
    CheckpointObservation removal{"removal", "PM_SUCCESS", {}};
    for (const char *id : {"particle-smoke", "grid-smoke"})
        removal.resources.push_back(
            {id, false, 3U, inactive_state_fingerprint()});
    result.observations.push_back(std::move(removal));

    install_and_step(baseline, "recovery", 4U);
    PM_CHECK(result.observations.size() == 6U);
    PM_CHECK(result.observations[0].resources[0].state_sha256 ==
             result.observations[5].resources[0].state_sha256);
    PM_CHECK(result.observations[0].resources[1].state_sha256 ==
             result.observations[5].resources[1].state_sha256);
    PM_CHECK(result.observations[0].resources[0].state_sha256 !=
             result.observations[1].resources[0].state_sha256);
    PM_CHECK(result.observations[0].resources[1].state_sha256 !=
             result.observations[1].resources[1].state_sha256);

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    result.status = ExecutionStatus::completed;
    return result;
}

[[nodiscard]] Json resource_json(const ResourceObservation &resource) {
    Json value = Json::object();
    value["id"] = resource.id;
    value["active"] = resource.active;
    value["generation"] = resource.generation;
    value["state_sha256"] = resource.state_sha256;
    return value;
}

[[nodiscard]] Json checkpoint_json(const CheckpointObservation &checkpoint) {
    Json resources = Json::array();
    for (const auto &resource : checkpoint.resources)
        resources.push_back(resource_json(resource));
    Json value = Json::object();
    value["id"] = checkpoint.id;
    value["outcome"] = checkpoint.outcome;
    value["resources"] = std::move(resources);
    return value;
}

[[nodiscard]] const char *status_name(ExecutionStatus status) {
    switch (status) {
    case ExecutionStatus::completed: return "completed";
    case ExecutionStatus::unsupported: return "unsupported";
    case ExecutionStatus::failed: return "failed";
    }
    return "failed";
}
} // namespace

std::string conformance_smoke_lifecycle(const std::filesystem::path &root,
                                        ExecutionStatus *status) {
    SmokeLifecycleRun run;
    try {
        run = execute_smoke_lifecycle();
    } catch (const std::exception &error) {
        run.status = ExecutionStatus::failed;
        run.reason = error.what();
    }
    if (status != nullptr) *status = run.status;

    Json resources = Json::array();
    for (const auto &checkpoint : run.observations)
        resources.push_back(checkpoint_json(checkpoint));
    Json families = Json::array();
    families.push_back("smoke");
    Json unrelated = Json::array();
    Json lifecycle_case = Json::object();
    lifecycle_case["id"] = "smoke-capacity-replacement-removal-rollback";
    lifecycle_case["execution_status"] = status_name(run.status);
    lifecycle_case["families"] = std::move(families);
    lifecycle_case["unrelated_resources"] = std::move(unrelated);
    lifecycle_case["observations"] = std::move(resources);
    Json cases = Json::array();
    cases.push_back(std::move(lifecycle_case));
    Json driver = Json::object();
    driver["name"] = run.driver;
    driver["device"] = run.device;
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = "smoke-self-lifecycle-v1";
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/smoke-self-lifecycle-v1.json");
    result["execution_status"] = status_name(run.status);
    if (!run.reason.empty()) result["reason"] = run.reason;
    result["driver"] = std::move(driver);
    result["cases"] = std::move(cases);
    return result.serialize();
}

} // namespace parallel_mater::conformance
