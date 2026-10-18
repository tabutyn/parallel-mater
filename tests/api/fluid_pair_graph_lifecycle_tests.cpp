// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <array>
#include <cstring>
#include <string_view>

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

PmCouplingState pair_edge(PmInteractionMaterialProfile profile) {
    PmCouplingState result{};
    result.type = PM_COUPLING_LIQUID_LIQUID;
    result.first = 0U;
    result.second = 1U;
    result.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_PRESSURE_BIT |
        PM_INTERACTION_MODE_VISCOSITY_BIT;
    result.materialProfile = profile;
    result.parameters.mechanical.twoWay = PM_TRUE;
    return result;
}

} // namespace

int main(int argc, char **argv) {
    const bool miscible = argc == 2 &&
        std::string_view(argv[1]) == "--miscible";
    PM_CHECK(argc == 1 || miscible);
    const auto profile = miscible ? PM_INTERACTION_MATERIAL_MISCIBLE
                                  : PM_INTERACTION_MATERIAL_EQUAL;
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
            profile};
        PmInteractionCapability capability{};
        PM_CHECK(pmGetPhysicalDeviceInteractionCapability(
                     context.physical, &key, &capability) == PM_SUCCESS);
        PM_CHECK(capability.maxGraphEdgesPerSimulation ==
                 PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY);
    }

    const PmFluidCapacity capacities[2]{{4U, 1U}, {4U, 1U}};
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

    const std::array<PmParticleState, 4> particles{{
        {{0.0F, 0.0F, 0.0F}, {0.1F, 0.0F, 0.0F}, 20.0F},
        {{0.01F, 0.0F, 0.0F}, {0.1F, 0.0F, 0.0F}, 20.0F},
        {{0.05F, 0.0F, 0.0F}, {-0.1F, 0.0F, 0.0F}, 20.0F},
        {{0.06F, 0.0F, 0.0F}, {-0.1F, 0.0F, 0.0F}, 20.0F},
    }};
    const auto input = context.buffer(sizeof(particles),
                                      PM_BUFFER_PARTICLE_INPUT_BIT);
    std::memcpy(input.mapped, particles.data(), sizeof(particles));
    context.flush(input);
    const PmBufferView views[2]{
        {{input.handle, 0U, sizeof(PmParticleState)},
         PM_FORMAT_PARTICLE_STATE, 1U, sizeof(PmParticleState)},
        {{input.handle, 2U * sizeof(PmParticleState), sizeof(PmParticleState)},
         PM_FORMAT_PARTICLE_STATE, 1U, sizeof(PmParticleState)},
    };
    const PmBufferView crowded_views[2]{
        {{input.handle, 0U, 2U * sizeof(PmParticleState)},
         PM_FORMAT_PARTICLE_STATE, 2U, sizeof(PmParticleState)},
        {{input.handle, 2U * sizeof(PmParticleState),
          2U * sizeof(PmParticleState)},
         PM_FORMAT_PARTICLE_STATE, 2U, sizeof(PmParticleState)},
    };

    const PmLiquidMaterialState equal_materials[2]{material(), material()};
    const PmLiquidMaterialState unequal_materials[2]{material(), material(900.0F)};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 2U;
    auto edge = pair_edge(profile);
    PmFluidPairMaterialState pipeline_materials{
        PM_STRUCTURE_TYPE_FLUID_PAIR_MATERIAL_STATE, nullptr, 0U, 1U};
    PmCouplingStateInfo pipeline_edge{};
    pipeline_edge.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
    pipeline_edge.pNext = &pipeline_materials;
    pipeline_edge.type = edge.type;
    pipeline_edge.first = edge.first;
    pipeline_edge.second = edge.second;
    pipeline_edge.modeFlags = edge.modeFlags;
    pipeline_edge.materialProfile = edge.materialProfile;
    pipeline_edge.parameters = edge.parameters;
    const PmPhysicsPipelineCouplingStateInfo pipeline_couplings{
        PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_COUPLING_STATE_INFO, nullptr, 1U,
        &pipeline_edge};
    pipeline_info.pNext = &pipeline_couplings;
    pipeline_info.pFluidMaterials = equal_materials;
    PmPhysicsPipeline equal_pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &equal_pipeline) == PM_SUCCESS);
    pipeline_info.pFluidMaterials = unequal_materials;
    PmPhysicsPipeline unequal_pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info,
                                     &unequal_pipeline) == PM_SUCCESS);
    // Creation owns a normalized copy of both descriptor layers.
    pipeline_materials.secondMaterialIndex = 0U;

    auto invalid = edge;
    invalid.first = 1U;
    invalid.second = 0U;
    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &invalid) ==
             PM_SUCCESS);
    const PmPhysicsPipeline working_pipeline = miscible
        ? unequal_pipeline : equal_pipeline;
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, working_pipeline) ==
             PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 2U; ++index)
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, index,
                                       &views[index]) == PM_SUCCESS);
    const PmStepInfo step{0.001F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_ERROR_INVALID_ARGUMENT);

    const PmFluidPairMaterialState command_materials{
        PM_STRUCTURE_TYPE_FLUID_PAIR_MATERIAL_STATE, nullptr, 0U, 1U};
    PmCouplingStateInfo command_edge{};
    command_edge.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
    command_edge.pNext = &command_materials;
    command_edge.type = edge.type;
    command_edge.first = edge.first;
    command_edge.second = edge.second;
    command_edge.modeFlags = edge.modeFlags;
    command_edge.materialProfile = edge.materialProfile;
    command_edge.parameters = edge.parameters;
    context.begin();
    PM_CHECK(pmCmdSetCouplings2(context.commands, simulation, 1U,
                                &command_edge) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, working_pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    // Replacing either endpoint clears the old private edge. The next step
    // recreates it with the new endpoint generation without growing capacity.
    for (std::uint32_t edit = 0U; edit < 32U; ++edit) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, working_pipeline) ==
                 PM_SUCCESS);
        const auto endpoint = edit % 2U;
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, endpoint,
                                       &views[endpoint]) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }

    context.begin();
    PM_CHECK(pmCmdSetCouplings2(context.commands, simulation, 0U, nullptr) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, working_pipeline) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, working_pipeline) ==
             PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 2U; ++index)
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, index,
                                       &views[index]) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    if (miscible) {
        // Installing an equal-material edge over unequal endpoints fails after
        // removal. The old miscible graph must be restored successfully.
        auto incompatible = command_edge;
        incompatible.pNext = nullptr;
        incompatible.materialProfile = PM_INTERACTION_MATERIAL_EQUAL;
        context.begin();
        PM_CHECK(pmCmdSetCouplings2(context.commands, simulation, 1U,
                                    &incompatible) == PM_SUCCESS);
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands,
                                          unequal_pipeline) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_ERROR_FEATURE_NOT_PRESENT);

        context.begin();
        PM_CHECK(pmCmdSetCouplings2(context.commands, simulation, 1U,
                                    &command_edge) == PM_SUCCESS);
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands,
                                          unequal_pipeline) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    } else {
        // Unequal endpoint materials reject the equal profile. Replacing the
        // endpoints under the equal pipeline makes the graph usable again.
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands,
                                          unequal_pipeline) == PM_SUCCESS);
        for (std::uint32_t index = 0U; index < 2U; ++index)
            PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation,
                                            index, &views[index]) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_ERROR_FEATURE_NOT_PRESENT);

        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands,
                                          equal_pipeline) == PM_SUCCESS);
        for (std::uint32_t index = 0U; index < 2U; ++index)
            PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation,
                                            index, &views[index]) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }

    // Pair-neighbor overflow is reported after asynchronous completion. A
    // smaller replacement recovers without rebuilding the simulation.
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, working_pipeline) ==
             PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 2U; ++index)
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, index,
                                       &crowded_views[index]) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_ERROR_CAPACITY_EXCEEDED);

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, working_pipeline) ==
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
    std::puts(miscible
        ? "PASS: bounded generation-safe miscible fluid pair graph lifecycle"
        : "PASS: bounded generation-safe equal fluid pair graph lifecycle");
}
