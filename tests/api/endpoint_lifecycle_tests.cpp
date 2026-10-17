// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <array>
#include <cstddef>

using parallel_mater::test::Context;

int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.particleEmitters = PM_TRUE;
    if (!context.open(needed)) return 77;

    constexpr std::uint32_t particle_capacity = 8U;
    const PmFluidCapacity capacities[2]{{particle_capacity, 16U},
                                        {particle_capacity, 16U}};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.fluidCount = 2U;
    create.pFluidCapacities = capacities;
    create.emitterCapacity = 2U;
    create.destroyerCapacity = 2U;
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &create, &simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        context.allocate(requirements.persistent), 0U,
        context.allocate(requirements.scratch), 0U};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation, &binding) == PM_SUCCESS);

    PmLiquidMaterialState material{};
    material.restDensity = 1000.0F;
    material.particleRadius = 0.025F;
    material.supportRadius = 0.1F;
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

    const PmFloat3 vertices[3]{{0.0F, 0.0F, 0.0F},
                               {0.2F, 0.0F, 0.0F},
                               {0.0F, 0.2F, 0.0F}};
    const std::uint32_t indices[3]{0U, 1U, 2U};
    PmParticleEmitterInfo emitters[2]{};
    PmParticleDestroyerInfo destroyers[2]{};
    for (std::uint32_t index = 0U; index < 2U; ++index) {
        emitters[index].active = PM_TRUE;
        emitters[index].fluidIndex = index;
        emitters[index].vertexCount = 3U;
        emitters[index].pVertices = vertices;
        emitters[index].triangleIndexCount = 3U;
        emitters[index].pTriangleIndices = indices;
        emitters[index].spacing = 0.1F;

        destroyers[index].active = PM_TRUE;
        destroyers[index].fluidIndex = index;
        destroyers[index].orientation[3] = 1.0F;
        destroyers[index].halfExtents = {1.0F, 1.0F};
        destroyers[index].crossing = PM_CROSSING_EITHER;
    }

    const PmBufferView empty{};
    constexpr PmDeviceSize output_size = sizeof(PmDomainOutputHeader) +
        particle_capacity * sizeof(PmFluidParticleOutput);
    const auto output = context.buffer(
        2U * output_size,
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange output_ranges[2]{{output.handle, 0U, output_size},
                                         {output.handle, output_size, output_size}};

    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    for (std::uint32_t fluid = 0U; fluid < 2U; ++fluid)
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, fluid,
                                       &empty) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteParticleEmitters(context.commands, simulation, 0U, 2U,
                                       emitters) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteParticleDestroyers(context.commands, simulation, 0U, 2U,
                                         destroyers) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    // Replacing a fluid retires only endpoints borrowing that public slot. The
    // same command list can bind fresh endpoint generations without reset.
    for (unsigned batch = 0U; batch < 4U; ++batch) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
        for (unsigned edit = 0U; edit < 64U; ++edit) {
            PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U,
                                           &empty) == PM_SUCCESS);
            PM_CHECK(pmCmdWriteParticleEmitters(context.commands, simulation,
                                               0U, 1U, &emitters[0]) == PM_SUCCESS);
            PM_CHECK(pmCmdWriteParticleDestroyers(context.commands, simulation,
                                                 0U, 1U, &destroyers[0]) == PM_SUCCESS);
        }
        for (std::uint32_t fluid = 0U; fluid < 2U; ++fluid)
            PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, fluid,
                                             &output_ranges[fluid]) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
        context.invalidate(output);
        for (std::uint32_t fluid = 0U; fluid < 2U; ++fluid) {
            const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(
                static_cast<const std::byte *>(output.mapped) +
                fluid * output_size);
            PM_CHECK(header->activeCount == 0U);
            PM_CHECK(header->capacity == particle_capacity);
        }
    }

    // Moving an endpoint to another fluid updates its ownership. Replacing the
    // old fluid must not orphan a still-live runtime generation.
    emitters[0].fluidIndex = 1U;
    destroyers[0].fluidIndex = 1U;
    context.begin();
    PM_CHECK(pmCmdWriteParticleEmitters(context.commands, simulation, 0U, 1U,
                                       &emitters[0]) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteParticleDestroyers(context.commands, simulation, 0U, 1U,
                                         &destroyers[0]) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0U, &empty) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdWriteParticleEmitters(context.commands, simulation, 0U, 1U,
                                       &emitters[0]) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteParticleDestroyers(context.commands, simulation, 0U, 1U,
                                         &destroyers[0]) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    // Execution-time construction failures retire the old target, reclaim its
    // storage, and leave unrelated fluids/endpoints available for later work.
    auto invalid_emitter = emitters[0];
    invalid_emitter.spacing = 0.01F;
    context.begin();
    PM_CHECK(pmCmdWriteParticleEmitters(context.commands, simulation, 0U, 1U,
                                       &invalid_emitter) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_ERROR_INVALID_ARGUMENT);

    context.begin();
    PM_CHECK(pmCmdWriteParticleEmitters(context.commands, simulation, 0U, 1U,
                                       &emitters[0]) == PM_SUCCESS);
    for (std::uint32_t fluid = 0U; fluid < 2U; ++fluid)
        PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, fluid,
                                         &output_ranges[fluid]) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    auto invalid_destroyer = destroyers[0];
    invalid_destroyer.orientation[3] = 0.0F;
    context.begin();
    PM_CHECK(pmCmdWriteParticleDestroyers(context.commands, simulation, 0U, 1U,
                                         &invalid_destroyer) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_ERROR_INVALID_ARGUMENT);

    context.begin();
    PM_CHECK(pmCmdWriteParticleDestroyers(context.commands, simulation, 0U, 1U,
                                         &destroyers[0]) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    PmParticleEmitterInfo inactive_emitter{};
    PmParticleDestroyerInfo inactive_destroyer{};
    context.begin();
    PM_CHECK(pmCmdWriteParticleEmitters(context.commands, simulation, 0U, 2U,
                                       std::array{inactive_emitter,
                                                  inactive_emitter}.data()) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteParticleDestroyers(context.commands, simulation, 0U, 2U,
                                         std::array{inactive_destroyer,
                                                    inactive_destroyer}.data()) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    // Reset must clear endpoint ownership along with private generations.
    emitters[0].fluidIndex = 0U;
    destroyers[0].fluidIndex = 0U;
    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    for (std::uint32_t fluid = 0U; fluid < 2U; ++fluid)
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, fluid,
                                       &empty) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteParticleEmitters(context.commands, simulation, 0U, 2U,
                                       emitters) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteParticleDestroyers(context.commands, simulation, 0U, 2U,
                                         destroyers) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    std::puts("PASS: 256 fluid endpoint replacements, failed rebind recovery and reset");
}
