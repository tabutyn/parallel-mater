// SPDX-License-Identifier: MIT
// Public SDK only. The formal profile uses four colocated particles with a
// one-neighbor capacity, then proves selective recovery without disturbing a
// second population.
#include "lifecycle_fixture.hpp"
#include "test_context.hpp"

#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

using parallel_mater::conformance::Json;
using parallel_mater::conformance::sha256;
using parallel_mater::conformance::sha256_contract_file;
using parallel_mater::test::Buffer;
using parallel_mater::test::CapacityLifecycleResult;
using parallel_mater::test::Context;
using parallel_mater::test::LifecycleCheckpointObservation;
using parallel_mater::test::LifecycleResourceObservation;

namespace {

constexpr std::uint32_t overflowing_particles = 4U;
constexpr std::uint32_t neighbor_capacity = 1U;
constexpr std::uint8_t invalid_output_sentinel = 0xa5U;

PmBufferView particle_view(const Buffer &buffer, PmDeviceSize offset,
                           std::uint32_t count) {
    return {{buffer.handle, offset,
             static_cast<PmDeviceSize>(count) * sizeof(PmParticleState)},
            PM_FORMAT_PARTICLE_STATE, count, sizeof(PmParticleState)};
}

const PmDomainOutputHeader *output_header(const Buffer &buffer,
                                          PmDeviceSize offset) {
    return reinterpret_cast<const PmDomainOutputHeader *>(
        static_cast<const std::byte *>(buffer.mapped) + offset);
}

const PmFluidParticleOutput *output_particles(
    const PmDomainOutputHeader *header) {
    return reinterpret_cast<const PmFluidParticleOutput *>(header + 1);
}

void append_float(std::ostringstream &stream, float value) {
    stream << std::hex << std::setw(8) << std::setfill('0')
           << std::bit_cast<std::uint32_t>(value);
}

std::string state_fingerprint(const PmDomainOutputHeader *header) {
    std::ostringstream canonical;
    canonical << "parallel-mater-fluid-public-state/v1;"
              << "active=" << (header->activeCount != 0U ? 1 : 0)
              << ";count=" << header->activeCount
              << ";capacity=" << header->capacity;
    const auto *particles = output_particles(header);
    for (std::uint32_t index = 0U; index < header->activeCount; ++index) {
        canonical << ";particle=" << index << ",stable=" << particles[index].stableId;
        for (float value : particles[index].position) {
            canonical << ",p=";
            append_float(canonical, value);
        }
        for (float value : particles[index].velocity) {
            canonical << ",v=";
            append_float(canonical, value);
        }
        for (float value : particles[index].acceleration) {
            canonical << ",a=";
            append_float(canonical, value);
        }
        canonical << ",t=";
        append_float(canonical, particles[index].temperatureCelsius);
        canonical << ",f=";
        append_float(canonical, particles[index].foam);
    }
    return sha256(canonical.str());
}

LifecycleResourceObservation resource_observation(
    std::string id, std::uint64_t generation,
    const PmDomainOutputHeader *header) {
    return {std::move(id), header->activeCount != 0U, generation,
            state_fingerprint(header)};
}

LifecycleResourceObservation invalidated_resource_observation(
    std::string id, std::uint64_t generation, std::string_view outcome) {
    const std::string canonical =
        "parallel-mater-fluid-public-state/v1;active=0;outcome=" +
        std::string(outcome);
    return {std::move(id), false, generation, sha256(canonical)};
}

void record_checkpoint(CapacityLifecycleResult *result, std::string id,
                       std::string outcome,
                       const PmDomainOutputHeader *target,
                       std::uint64_t target_generation,
                       const PmDomainOutputHeader *control) {
    if (result == nullptr) return;
    LifecycleCheckpointObservation observation;
    observation.id = std::move(id);
    observation.outcome = std::move(outcome);
    observation.resources.push_back(resource_observation(
        "target-fluid", target_generation, target));
    observation.resources.push_back(resource_observation(
        "unrelated-fluid", 1U, control));
    result->observations.push_back(std::move(observation));
}

void record_overflow_checkpoint(CapacityLifecycleResult *result,
                                const PmDomainOutputHeader *control) {
    if (result == nullptr) return;
    LifecycleCheckpointObservation observation;
    observation.id = "overflow";
    observation.outcome = "PM_ERROR_CAPACITY_EXCEEDED";
    observation.resources.push_back(invalidated_resource_observation(
        "target-fluid", 3U, observation.outcome));
    observation.resources.push_back(resource_observation(
        "unrelated-fluid", 1U, control));
    result->observations.push_back(std::move(observation));
}

void copy_control(Context &context, PmSimulation simulation,
                  const PmBufferRange &control) {
    context.begin();
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 1U, &control) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
}

#ifndef PM_CONFORMANCE_EMBEDDED

Json resource_json(const LifecycleResourceObservation &resource) {
    Json value = Json::object();
    value["id"] = resource.id;
    value["active"] = resource.active;
    value["generation"] = resource.generation;
    value["state_sha256"] = resource.state_sha256;
    return value;
}

Json checkpoint_json(const LifecycleCheckpointObservation &checkpoint) {
    Json resources = Json::array();
    for (const auto &resource : checkpoint.resources)
        resources.push_back(resource_json(resource));
    Json value = Json::object();
    value["id"] = checkpoint.id;
    value["outcome"] = checkpoint.outcome;
    value["resources"] = std::move(resources);
    return value;
}

std::string lifecycle_result_json(const CapacityLifecycleResult &fixture,
                                  std::string_view driver_name,
                                  const std::filesystem::path &profile) {
    Json observations = Json::array();
    for (const auto &observation : fixture.observations)
        observations.push_back(checkpoint_json(observation));
    Json families = Json::array();
    families.push_back("liquid");
    Json unrelated = Json::array();
    unrelated.push_back("unrelated-fluid");
    Json capacity = Json::object();
    capacity["id"] = "capacity-replacement-recovery";
    capacity["execution_status"] =
        fixture.status == 0 ? "completed" :
        fixture.status == 77 ? "unsupported" : "failed";
    capacity["families"] = std::move(families);
    capacity["unrelated_resources"] = std::move(unrelated);
    capacity["observations"] = std::move(observations);
    Json cases = Json::array();
    cases.push_back(std::move(capacity));
    Json driver = Json::object();
    driver["name"] = std::string(driver_name);
    driver["device"] = fixture.device.empty() ? "unavailable" : fixture.device;
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = "adverse-capacity-three-family-v1";
    result["profile_sha256"] = sha256_contract_file(profile);
    result["execution_status"] =
        fixture.status == 0 ? "completed" :
        fixture.status == 77 ? "unsupported" : "failed";
    result["driver"] = std::move(driver);
    result["cases"] = std::move(cases);
    return result.serialize();
}

#endif

} // namespace

int parallel_mater_capacity_recovery_fixture(CapacityLifecycleResult *result) {
    if (result != nullptr) *result = {};
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.statisticsQueries = PM_TRUE;
    if (!context.open(needed)) {
        if (result != nullptr) result->status = 77;
        return 77;
    }

    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);
    if (result != nullptr) result->device = properties.deviceName;

    constexpr PmFluidCapacity capacities[2]{
        {overflowing_particles, neighbor_capacity}, {1U, neighbor_capacity}};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.contactCapacity = 16U;
    simulation_info.fluidCount = 2U;
    simulation_info.pFluidCapacities = capacities;
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

    PmLiquidMaterialState material{};
    material.restDensity = 1000.0F;
    material.particleRadius = 0.025F;
    material.supportRadius = 0.1F;
    material.restParticleVolume = 0.000125F;
    material.pressureStiffness = 50.0F;
    material.maximumSpeed = 2.9F;
    material.densityIterations = 1U;
    const PmLiquidMaterialState materials[2]{material, material};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 2U;
    pipeline_info.pFluidMaterials = materials;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    constexpr std::uint32_t control_index = overflowing_particles;
    constexpr std::uint32_t baseline_index = control_index + 1U;
    constexpr std::uint32_t replacement_index = baseline_index + 1U;
    constexpr std::uint32_t recovery_index = replacement_index + 1U;
    const auto input = context.buffer(
        (recovery_index + 1U) * sizeof(PmParticleState),
        PM_BUFFER_PARTICLE_INPUT_BIT);
    auto *states = static_cast<PmParticleState *>(input.mapped);
    for (std::uint32_t index = 0U; index < overflowing_particles; ++index)
        states[index].position[0] = 0.0F;
    states[control_index].position[0] = 10.0F;
    states[baseline_index].position[0] = 20.0F;
    states[replacement_index].position[0] = 30.0F;
    states[recovery_index].position[0] = 40.0F;
    context.flush(input);

    const PmDeviceSize target_output_size = sizeof(PmDomainOutputHeader) +
        overflowing_particles * sizeof(PmFluidParticleOutput);
    const PmDeviceSize control_output_size = sizeof(PmDomainOutputHeader) +
        sizeof(PmFluidParticleOutput);
    const auto output = context.buffer(
        target_output_size + control_output_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange target_output{output.handle, 0U, target_output_size};
    const PmBufferRange control_output{
        output.handle, target_output_size, control_output_size};

    PmQueryPoolCreateInfo query_info{};
    query_info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = PM_QUERY_STEP_STATISTICS;
    query_info.queryCount = 1U;
    PmQueryPool query{};
    PM_CHECK(pmCreateQueryPool(context.device, &query_info, &query) == PM_SUCCESS);

    const auto one_particle = [&](std::uint32_t index) {
        return particle_view(input, index * sizeof(PmParticleState), 1U);
    };
    const PmBufferView baseline = one_particle(baseline_index);
    const PmBufferView replacement = one_particle(replacement_index);
    const PmBufferView overflowing =
        particle_view(input, 0U, overflowing_particles);
    const PmBufferView control = one_particle(control_index);
    const PmBufferView recovered = one_particle(recovery_index);
    const PmStepInfo step{1.0F / 60.0F, 1U, {0.0F, 0.0F, 0.0F}, 0U};

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U, &baseline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 1U, &control) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                    &target_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 1U,
                                    &control_output) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    record_checkpoint(result, "baseline", "PM_SUCCESS",
                      output_header(output, 0U), 1U,
                      output_header(output, target_output_size));

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                    &replacement) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                    &target_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 1U,
                                    &control_output) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    record_checkpoint(result, "replacement", "PM_SUCCESS",
                      output_header(output, 0U), 2U,
                      output_header(output, target_output_size));

    std::memset(output.mapped, invalid_output_sentinel,
                static_cast<std::size_t>(output.size));
    context.flush(output);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                    &overflowing) == PM_SUCCESS);
    PM_CHECK(pmCmdResetQueryPool(context.commands, query, 0U, 1U) == PM_SUCCESS);
    PM_CHECK(pmCmdBeginStatisticsQuery(context.commands, query, 0U) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
    PM_CHECK(pmCmdEndStatisticsQuery(context.commands, query, 0U) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                    &target_output) == PM_SUCCESS);
    PM_CHECK(pmEndCommandBuffer(context.commands) == PM_SUCCESS);
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &context.commands;
    PM_CHECK(pmQueueSubmit(context.queue, &submit, context.fence) == PM_SUCCESS);
    const PmResult completion = pmWaitForFences(
        context.device, 1U, &context.fence, PM_TRUE, PM_TIMEOUT_INFINITE);
    PM_CHECK(completion == PM_ERROR_CAPACITY_EXCEEDED);

    PmResult execution{};
    PM_CHECK(pmGetSimulationStatus(context.device, simulation, &execution) ==
             PM_SUCCESS);
    PM_CHECK(execution == PM_ERROR_CAPACITY_EXCEEDED);
    PmStepStatistics failed_statistics{};
    PM_CHECK(pmGetQueryPoolResults(
                 context.device, query, 0U, 1U, sizeof(failed_statistics),
                 &failed_statistics, sizeof(failed_statistics),
                 PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
    PM_CHECK(failed_statistics.executionResult == PM_ERROR_CAPACITY_EXCEEDED);
    PM_CHECK(failed_statistics.overflowCount == overflowing_particles);
    PM_CHECK(failed_statistics.peakParticleCount == overflowing_particles + 1U);
    context.invalidate(output);
    const auto *output_bytes = static_cast<const std::uint8_t *>(output.mapped);
    for (PmDeviceSize index = 0U; index < output.size; ++index)
        PM_CHECK(output_bytes[index] == invalid_output_sentinel);

    copy_control(context, simulation, control_output);
    context.invalidate(output);
    PM_CHECK(output_header(output, target_output_size)->activeCount == 1U);
    record_overflow_checkpoint(result,
                               output_header(output, target_output_size));

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U, &recovered) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdResetQueryPool(context.commands, query, 0U, 1U) == PM_SUCCESS);
    PM_CHECK(pmCmdBeginStatisticsQuery(context.commands, query, 0U) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
    PM_CHECK(pmCmdEndStatisticsQuery(context.commands, query, 0U) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0U,
                                    &target_output) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 1U,
                                    &control_output) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    PM_CHECK(pmGetSimulationStatus(context.device, simulation, &execution) ==
             PM_SUCCESS);
    PM_CHECK(execution == PM_SUCCESS);
    PmStepStatistics recovered_statistics{};
    PM_CHECK(pmGetQueryPoolResults(
                 context.device, query, 0U, 1U, sizeof(recovered_statistics),
                 &recovered_statistics, sizeof(recovered_statistics),
                 PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
    PM_CHECK(recovered_statistics.executionResult == PM_SUCCESS);
    PM_CHECK(recovered_statistics.overflowCount == 0U);
    PM_CHECK(recovered_statistics.peakParticleCount == 2U);

    context.invalidate(output);
    const auto *target = output_header(output, 0U);
    const auto *control_after = output_header(output, target_output_size);
    PM_CHECK(target->activeCount == 1U);
    PM_CHECK(control_after->activeCount == 1U);
    PM_CHECK(std::fabs(output_particles(target)[0].position[0] - 40.0F) <
             1.0e-6F);
    PM_CHECK(std::fabs(output_particles(control_after)[0].position[0] - 10.0F) <
             1.0e-6F);
    record_checkpoint(result, "recovery", "PM_SUCCESS", target, 4U,
                      control_after);

    pmDestroyQueryPool(context.device, query);
    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    if (result != nullptr) result->status = 0;
    std::printf(
        "PASS: capacity replacement checkpoints and 4-particle/1-neighbor "
        "selective recovery on %s (vendor=%04x device=%04x)\n",
        properties.deviceName, properties.vendorId, properties.deviceId);
    return 0;
}

#ifndef PM_CONFORMANCE_EMBEDDED
int main(int argc, char **argv) {
    std::filesystem::path output;
    std::filesystem::path profile = PM_LIFECYCLE_PROFILE_PATH;
    std::string driver_name = "selected-driver";
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if ((option == "--output" || option == "--profile" ||
             option == "--driver-name") && index + 1 >= argc) {
            std::fprintf(stderr, "%.*s requires a value\n",
                         static_cast<int>(option.size()), option.data());
            return 2;
        }
        if (option == "--output") output = argv[++index];
        else if (option == "--profile") profile = argv[++index];
        else if (option == "--driver-name") driver_name = argv[++index];
        else {
            std::fprintf(stderr, "unknown option: %.*s\n",
                         static_cast<int>(option.size()), option.data());
            return 2;
        }
    }
    CapacityLifecycleResult result;
    const int status = parallel_mater_capacity_recovery_fixture(&result);
    if (!output.empty()) {
        if (!output.parent_path().empty())
            std::filesystem::create_directories(output.parent_path());
        std::ofstream stream(output, std::ios::binary);
        if (!stream) {
            std::fprintf(stderr, "cannot write lifecycle result: %s\n",
                         output.string().c_str());
            return 2;
        }
        stream << lifecycle_result_json(result, driver_name, profile);
    }
    return status;
}
#endif
