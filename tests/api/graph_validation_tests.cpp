// SPDX-License-Identifier: MIT
#include "test_context.hpp"

using parallel_mater::test::Context;

int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.rigidBodies = PM_TRUE;
    needed.particleLiquids = PM_TRUE;
    needed.liquidRigidCoupling = PM_TRUE;
    if (!context.open(needed)) return 77;

    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    simulation_info.couplingCapacity = 2U;
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

    PmCouplingState edge{};
    edge.type = PM_COUPLING_LIQUID_RIGID;
    edge.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_CONTAINMENT_BIT;
    edge.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    edge.parameters.mechanical.twoWay = PM_TRUE;
    const PmCouplingState duplicates[3]{edge, edge, edge};

    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    PmPhysicsPipeline empty{}, coupled{}, duplicate{}, excessive{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &empty) ==
             PM_SUCCESS);
    pipeline_info.couplingCount = 1U;
    pipeline_info.pCouplings = &edge;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &coupled) ==
             PM_SUCCESS);
    pipeline_info.couplingCount = 2U;
    pipeline_info.pCouplings = duplicates;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &duplicate) ==
             PM_SUCCESS);
    pipeline_info.couplingCount = 3U;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &excessive) ==
             PM_SUCCESS);

    const PmStepInfo step{1.0F / 60.0F, 1U, {0.0F, 0.0F, 0.0F}, 0U};
    const auto run = [&](PmPhysicsPipeline pipeline) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        return context.submit();
    };

    PM_CHECK(run(coupled) == PM_SUCCESS);
    PM_CHECK(run(duplicate) == PM_ERROR_INVALID_ARGUMENT);
    PM_CHECK(run(coupled) == PM_SUCCESS);
    PM_CHECK(run(excessive) == PM_ERROR_CAPACITY_EXCEEDED);
    PM_CHECK(run(coupled) == PM_SUCCESS);

    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &edge) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    PmCouplingState invalid = edge;
    invalid.modeFlags |= PM_INTERACTION_MODE_DRAG_BIT;
    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &invalid) ==
             PM_ERROR_FEATURE_NOT_PRESENT);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    invalid = edge;
    invalid.materialProfile = PM_INTERACTION_MATERIAL_SMOKE_GRID;
    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1U, &invalid) ==
             PM_ERROR_FEATURE_NOT_PRESENT);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 2U, duplicates) ==
             PM_ERROR_INVALID_ARGUMENT);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 3U, duplicates) ==
             PM_ERROR_CAPACITY_EXCEEDED);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    pmDestroySimulation(context.device, simulation);
    for (const auto pipeline : {empty, coupled, duplicate, excessive})
        pmDestroyPhysicsPipeline(context.device, pipeline);
    std::puts("PASS: rejected graph replacements preserve the active graph");
}
