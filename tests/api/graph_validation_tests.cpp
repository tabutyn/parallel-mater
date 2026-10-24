// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <limits>

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

    const auto reject_smoke = [&](const PmCouplingState *candidate,
                                  std::uint32_t count, PmResult expected) {
        context.begin();
        PM_CHECK(pmCmdSetCouplings(context.commands, simulation, count,
                                   candidate) == expected);
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
                 PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    };

    PmCouplingState smoke_pair{};
    smoke_pair.type = PM_COUPLING_SMOKE_SMOKE;
    smoke_pair.first = 0U;
    smoke_pair.second = 1U;
    smoke_pair.modeFlags = PM_INTERACTION_MODE_PRESSURE_BIT;
    smoke_pair.materialProfile = PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    reject_smoke(&smoke_pair, 1U, PM_ERROR_FEATURE_NOT_PRESENT);

    auto reversed_smoke = smoke_pair;
    reversed_smoke.first = 1U;
    reversed_smoke.second = 0U;
    reject_smoke(&reversed_smoke, 1U, PM_ERROR_INVALID_ARGUMENT);

    const PmCouplingState duplicate_smoke[2]{smoke_pair, smoke_pair};
    reject_smoke(duplicate_smoke, 2U, PM_ERROR_INVALID_ARGUMENT);

    auto unsupported_smoke_bundle = smoke_pair;
    unsupported_smoke_bundle.modeFlags = PM_INTERACTION_MODE_DRAG_BIT;
    reject_smoke(&unsupported_smoke_bundle, 1U,
                 PM_ERROR_FEATURE_NOT_PRESENT);

    unsupported_smoke_bundle.materialProfile =
        PM_INTERACTION_MATERIAL_SMOKE_PARTICLE_GRID;
    unsupported_smoke_bundle.modeFlags = PM_INTERACTION_MODE_PRESSURE_BIT;
    reject_smoke(&unsupported_smoke_bundle, 1U,
                 PM_ERROR_FEATURE_NOT_PRESENT);

    unsupported_smoke_bundle.modeFlags = PM_INTERACTION_MODE_DRAG_BIT |
        PM_INTERACTION_MODE_THERMAL_BIT;
    reject_smoke(&unsupported_smoke_bundle, 1U,
                 PM_ERROR_FEATURE_NOT_PRESENT);

    PmCouplingState soft_pair{};
    soft_pair.type = PM_COUPLING_SOFT_BODY_SOFT_BODY;
    soft_pair.first = 0U;
    soft_pair.second = 1U;
    soft_pair.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    soft_pair.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    reject_smoke(&soft_pair, 1U, PM_ERROR_FEATURE_NOT_PRESENT);

    auto reversed_soft_pair = soft_pair;
    reversed_soft_pair.first = 1U;
    reversed_soft_pair.second = 0U;
    reject_smoke(&reversed_soft_pair, 1U, PM_ERROR_INVALID_ARGUMENT);

    auto self_soft_pair = soft_pair;
    self_soft_pair.second = 0U;
    reject_smoke(&self_soft_pair, 1U, PM_ERROR_INVALID_ARGUMENT);

    const PmCouplingState duplicate_soft_pairs[2]{soft_pair, soft_pair};
    reject_smoke(duplicate_soft_pairs, 2U, PM_ERROR_INVALID_ARGUMENT);

    auto invalid_soft_bundle = soft_pair;
    invalid_soft_bundle.modeFlags |= PM_INTERACTION_MODE_ATTACHMENT_BIT;
    reject_smoke(&invalid_soft_bundle, 1U,
                 PM_ERROR_FEATURE_NOT_PRESENT);

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

    PmCouplingStateInfo soft_pair_info{};
    soft_pair_info.sType = PM_STRUCTURE_TYPE_COUPLING_STATE_INFO;
    soft_pair_info.type = PM_COUPLING_SOFT_BODY_SOFT_BODY;
    soft_pair_info.first = 0U;
    soft_pair_info.second = 1U;
    soft_pair_info.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
    soft_pair_info.materialProfile =
        PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
    reject_pair(soft_pair_info, PM_ERROR_FEATURE_NOT_PRESENT);

    auto reversed_soft_pair_info = soft_pair_info;
    reversed_soft_pair_info.first = 1U;
    reversed_soft_pair_info.second = 0U;
    reject_pair(reversed_soft_pair_info, PM_ERROR_INVALID_ARGUMENT);

    const PmCouplingStateInfo duplicate_soft_pair_info[2]{
        soft_pair_info, soft_pair_info};
    context.begin();
    PM_CHECK(pmCmdSetCouplings2(context.commands, simulation, 2U,
                                duplicate_soft_pair_info) ==
             PM_ERROR_INVALID_ARGUMENT);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

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

    PmFluidInterfaceState interface_state{
        PM_STRUCTURE_TYPE_FLUID_INTERFACE_STATE, nullptr, 0.072F};
    materials.pNext = &interface_state;
    pair.pNext = &materials;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);

    pair.materialProfile = PM_INTERACTION_MATERIAL_EQUAL;
    pair.pNext = &interface_state;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);

    pair.materialProfile = PM_INTERACTION_MATERIAL_IMMISCIBLE_INTERFACE;
    pair.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
        PM_INTERACTION_MODE_PRESSURE_BIT |
        PM_INTERACTION_MODE_INTERFACE_BIT;

    pair.pNext = nullptr;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);
    materials.pNext = nullptr;
    pair.pNext = &materials;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);
    pair.pNext = &interface_state;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);

    auto invalid_interface = interface_state;
    invalid_interface.surfaceTension = -0.001F;
    materials.pNext = &invalid_interface;
    pair.pNext = &materials;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);
    invalid_interface.surfaceTension =
        std::numeric_limits<float>::infinity();
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);
    invalid_interface.surfaceTension =
        std::numeric_limits<float>::quiet_NaN();
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);

    auto duplicate_interface = interface_state;
    duplicate_interface.pNext = &interface_state;
    materials.pNext = &duplicate_interface;
    reject_pair(pair, PM_ERROR_INVALID_ARGUMENT);

    interface_state.surfaceTension = 0.0F;
    materials.pNext = &interface_state;
    pair.pNext = &materials;
    reject_pair(pair, PM_ERROR_FEATURE_NOT_PRESENT);

    interface_state.pNext = &materials;
    materials.pNext = nullptr;
    pair.pNext = &interface_state;
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
