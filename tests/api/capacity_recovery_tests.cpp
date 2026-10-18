// SPDX-License-Identifier: MIT
// Public SDK only. Exact profile: four colocated particles with a one-neighbor
// capacity. Each particle observes three neighbors in the shared Slang solver.
#include "test_context.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

using parallel_mater::test::Context;

namespace {

constexpr std::uint32_t overflowing_particles = 4U;
constexpr std::uint32_t neighbor_capacity = 1U;
constexpr std::uint8_t invalid_output_sentinel = 0xa5U;

PmBufferView particle_view(const parallel_mater::test::Buffer &buffer,
                           PmDeviceSize offset, std::uint32_t count) {
    return {{buffer.handle, offset,
             static_cast<PmDeviceSize>(count) * sizeof(PmParticleState)},
            PM_FORMAT_PARTICLE_STATE, count, sizeof(PmParticleState)};
}

const PmDomainOutputHeader *output_header(
    const parallel_mater::test::Buffer &buffer, PmDeviceSize offset) {
    return reinterpret_cast<const PmDomainOutputHeader *>(
        static_cast<const std::byte *>(buffer.mapped) + offset);
}

const PmFluidParticleOutput *output_particles(
    const PmDomainOutputHeader *header) {
    return reinterpret_cast<const PmFluidParticleOutput *>(header + 1);
}

} // namespace

int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.statisticsQueries = PM_TRUE;
    if (!context.open(needed)) return 77;

    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(context.physical, &properties);

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
    material.maximumSpeed = 100.0F;
    material.densityIterations = 1U;
    const PmLiquidMaterialState materials[2]{material, material};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 2U;
    pipeline_info.pFluidMaterials = materials;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const auto input = context.buffer(
        (overflowing_particles + 2U) * sizeof(PmParticleState),
        PM_BUFFER_PARTICLE_INPUT_BIT);
    auto *states = static_cast<PmParticleState *>(input.mapped);
    for (std::uint32_t index = 0U; index < overflowing_particles; ++index)
        states[index].position[0] = 0.0F;
    states[overflowing_particles].position[0] = 10.0F;
    states[overflowing_particles + 1U].position[0] = 20.0F;
    context.flush(input);

    const PmDeviceSize target_output_size = sizeof(PmDomainOutputHeader) +
        overflowing_particles * sizeof(PmFluidParticleOutput);
    const PmDeviceSize control_output_size = sizeof(PmDomainOutputHeader) +
        sizeof(PmFluidParticleOutput);
    const auto output = context.buffer(
        target_output_size + control_output_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    std::memset(output.mapped, invalid_output_sentinel,
                static_cast<std::size_t>(output.size));
    context.flush(output);
    const PmBufferRange target_output{output.handle, 0U, target_output_size};
    const PmBufferRange control_output{
        output.handle, target_output_size, control_output_size};

    PmQueryPoolCreateInfo query_info{};
    query_info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = PM_QUERY_STEP_STATISTICS;
    query_info.queryCount = 1U;
    PmQueryPool query{};
    PM_CHECK(pmCreateQueryPool(context.device, &query_info, &query) == PM_SUCCESS);

    const PmBufferView overflowing = particle_view(input, 0U,
                                                    overflowing_particles);
    const PmBufferView control = particle_view(
        input, overflowing_particles * sizeof(PmParticleState), 1U);
    const PmStepInfo step{1.0F / 60.0F, 1U, {0.0F, 0.0F, 0.0F}, 0U};

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                    &overflowing) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 1U,
                                    &control) == PM_SUCCESS);
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

    // Recording and acceptance succeed. Generated overflow is deferred until
    // the accepted work completes.
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

    // The copy following the failed step was not executed: invalid output is
    // not accidentally published as a successful domain snapshot.
    context.invalidate(output);
    const auto *output_bytes = static_cast<const std::uint8_t *>(output.mapped);
    for (PmDeviceSize index = 0U; index < output.size; ++index)
        PM_CHECK(output_bytes[index] == invalid_output_sentinel);

    // Reinitialize only the failed population with one isolated particle.
    // The unrelated control population must remain live and unchanged.
    const PmBufferView recovered = particle_view(
        input, (overflowing_particles + 1U) * sizeof(PmParticleState), 1U);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                    &recovered) == PM_SUCCESS);
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
    PM_CHECK(std::fabs(output_particles(target)[0].position[0] - 20.0F) <
             1.0e-6F);
    PM_CHECK(std::fabs(output_particles(control_after)[0].position[0] - 10.0F) <
             1.0e-6F);

    pmDestroyQueryPool(context.device, query);
    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    std::printf(
        "PASS: 4 colocated particles / 1 neighbor overflow and selective "
        "reinitialization on %s (vendor=%04x device=%04x)\n",
        properties.deviceName, properties.vendorId, properties.deviceId);
}
