// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <array>
#include <cstring>

using parallel_mater::test::Context;

namespace {

PmLiquidMaterialState material(float density = 1000.0F) {
    PmLiquidMaterialState result{};
    result.restDensity = density;
    result.particleRadius = 0.025F;
    result.supportRadius = 0.1F;
    result.restParticleVolume = 0.000125F;
    result.pressureStiffness = 1.0F;
    result.kinematicViscosity = 0.1F;
    result.normalDamping = 0.05F;
    result.maximumSpeed = 100.0F;
    result.maximumPairAcceleration = 1000.0F;
    result.densityIterations = 1U;
    return result;
}

PmCouplingState pair_edge() {
    PmCouplingState result{};
    result.type = PM_COUPLING_LIQUID_LIQUID;
    result.first = 0U;
    result.second = 1U;
    result.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_PRESSURE_BIT |
        PM_INTERACTION_MODE_VISCOSITY_BIT;
    result.materialProfile = PM_INTERACTION_MATERIAL_EQUAL;
    result.parameters.mechanical.twoWay = PM_TRUE;
    return result;
}

} // namespace

int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.liquidLiquidCoupling = PM_TRUE;
    if (!context.open(needed)) return 77;

    for (const auto mode : {PM_INTERACTION_MODE_CONTACT,
                            PM_INTERACTION_MODE_PRESSURE,
                            PM_INTERACTION_MODE_VISCOSITY}) {
        const PmInteractionCapabilityKey key{
            PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
            PM_INTERACTION_SCOPE_SEPARATE_SYSTEM, mode,
            PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
            PM_INTERACTION_MATERIAL_EQUAL};
        PmInteractionCapability capability{};
        PM_CHECK(pmGetPhysicalDeviceInteractionCapability(
                     context.physical, &key, &capability) == PM_SUCCESS);
        PM_CHECK(capability.maxGraphEdgesPerSimulation ==
                 PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY);
    }

    const PmFluidCapacity capacities[2]{{4U, 16U}, {4U, 16U}};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.fluidCount = 2U;
    create.pFluidCapacities = capacities;
    create.couplingCapacity = 1U;
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

    const std::array<PmParticleState, 2> particles{{
        {{0.0F, 0.0F, 0.0F}, {0.1F, 0.0F, 0.0F}, 20.0F},
        {{0.05F, 0.0F, 0.0F}, {-0.1F, 0.0F, 0.0F}, 20.0F},
    }};
    const auto input = context.buffer(sizeof(particles),
                                      PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(input.mapped, particles.data(), sizeof(particles));
    context.flush(input);
    const PmBufferView views[2]{
        {{input.handle, 0U, sizeof(PmParticleState)},
         PM_FORMAT_PARTICLE_STATE, 1U, sizeof(PmParticleState)},
        {{input.handle, sizeof(PmParticleState), sizeof(PmParticleState)},
         PM_FORMAT_PARTICLE_STATE, 1U, sizeof(PmParticleState)},
    };

    const PmLiquidMaterialState equal_materials[2]{material(), material()};
    const PmLiquidMaterialState unequal_materials[2]{material(), material(900.0F)};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 2U;
    pipeline_info.pFluidMaterials = equal_materials;
    PmPhysicsPipeline equal_pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &equal_pipeline) == PM_SUCCESS);
    pipeline_info.pFluidMaterials = unequal_materials;
    PmPhysicsPipeline unequal_pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &unequal_pipeline) == PM_SUCCESS);

    auto edge = pair_edge();
    auto invalid = edge;
    invalid.first = 1U;
    invalid.second = 0U;
    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &invalid) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, equal_pipeline) ==
             PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 2U; ++index)
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, index,
                                       &views[index]) == PM_SUCCESS);
    const PmStepInfo step{0.001F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_ERROR_INVALID_ARGUMENT);

    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &edge) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, equal_pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    // Replacing either endpoint clears the old private edge. The next step
    // recreates it with the new endpoint generation without growing capacity.
    for (std::uint32_t edit = 0U; edit < 32U; ++edit) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, equal_pipeline) ==
                 PM_SUCCESS);
        const auto endpoint = edit % 2U;
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, endpoint,
                                       &views[endpoint]) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }

    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 0U, nullptr) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, equal_pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, equal_pipeline) ==
             PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 2U; ++index)
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, index,
                                       &views[index]) == PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &edge) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    // Unequal endpoint materials reject the requested profile. Replacing the
    // endpoints under the equal pipeline makes the unchanged graph usable.
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, unequal_pipeline) ==
             PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 2U; ++index)
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, index,
                                       &views[index]) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_ERROR_FEATURE_NOT_PRESENT);

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, equal_pipeline) ==
             PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 2U; ++index)
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, index,
                                       &views[index]) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, unequal_pipeline);
    pmDestroyPhysicsPipeline(context.device, equal_pipeline);
    std::puts("PASS: bounded generation-safe fluid pair graph lifecycle");
}
