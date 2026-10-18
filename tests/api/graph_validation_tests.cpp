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

    PmCouplingStateInfo pair{};
    pair.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
    pair.type = PM_COUPLING_LIQUID_LIQUID;
    pair.first = 0U;
    pair.second = 1U;
    pair.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_PRESSURE_BIT |
        PM_INTERACTION_MODE_VISCOSITY_BIT;
    pair.materialProfile = PM_INTERACTION_MATERIAL_MISCIBLE;
    pair.parameters.mechanical.twoWay = PM_TRUE;
    PmFluidPairMaterialState materials{
        PM_STRUCTURE_TYPE_FLUID_PAIR_MATERIAL_STATE, nullptr, 0U, 1U};

    const auto reject_pair = [&](const PmCouplingStateInfo &candidate,
                                 PmResult expected) {
        context.begin();
        PM_CHECK(pmCmdSetCouplings2(context.commands, simulation, 1U,
                                    &candidate) == expected);
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    };

    // Different-material profiles require one exact typed endpoint-material
    // state. Every rejected replacement leaves the prior liquid/rigid graph.
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);
    pair.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);
    pair.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;

    auto unknown_materials = materials;
    unknown_materials.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    pair.pNext = &unknown_materials;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);

    auto nested_materials = materials;
    nested_materials.pNext = &materials;
    pair.pNext = &nested_materials;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);

    auto mismatched_materials = materials;
    mismatched_materials.secondMaterialIndex = 0U;
    pair.pNext = &mismatched_materials;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);

    pair.pNext = &materials;
    reject_pair(pair, PM_ERROR_FEATURE_NOT_PRESENT);
    pair.materialProfile = PM_INTERACTION_MATERIAL_IMMISCIBLE_INTERFACE;
    pair.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_PRESSURE_BIT;
    reject_pair(pair, PM_ERROR_FEATURE_NOT_PRESENT);

    PmPhysicsPipelineCouplingStateInfo pipeline_couplings{
        PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_COUPLING_STATE_INFO, nullptr, 1U,
        &pair};
    PmPhysicsPipelineCreateInfo extensible_info{};
    extensible_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    extensible_info.pNext = &pipeline_couplings;
    PmPhysicsPipeline rejected = empty;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &extensible_info,
                                     &rejected) ==
             PM_ERROR_FEATURE_NOT_PRESENT);
    PM_CHECK(rejected == nullptr);

    // The legacy and extensible coupling arrays are mutually exclusive.
    extensible_info.couplingCount = 1U;
    extensible_info.pCouplings = &edge;
    rejected = empty;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &extensible_info,
                                     &rejected) ==
             PM_ERROR_INVALID_ARGUMENT);
    PM_CHECK(rejected == nullptr);

    pmDestroySimulation(context.device, simulation);
    for (const auto pipeline : {empty, coupled, duplicate, excessive})
        pmDestroyPhysicsPipeline(context.device, pipeline);
    std::puts("PASS: rejected graph replacements preserve the active graph");
}
