// SPDX-License-Identifier: MIT
#include "test_context.hpp"
#include <cstring>
using parallel_mater::test::Context;
int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = needed.rope = needed.domainCouplings = PM_TRUE;
    if (!context.open(needed)) return 77;
    const auto input = context.buffer(sizeof(PmParticleState), PM_BUFFER_PARTICLE_INPUT_BIT);
    PmParticleState particle{}; particle.position[0] = 10;
    std::memcpy(input.mapped, &particle, sizeof(particle)); context.flush(input);
    const PmBufferView particles{{input.handle, 0, input.size}, PM_FORMAT_PARTICLE_STATE, 1, sizeof(particle)};
    const PmFluidCapacity fluid_capacity{1, 16};
    const PmRopeCapacity rope_capacity{16};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.fluidCount = create.ropeCount = 1;
    create.pFluidCapacities = &fluid_capacity; create.pRopeCapacities = &rope_capacity;
    create.couplingCapacity = 2;
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &create, &simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{context.allocate(requirements.persistent), 0,
        context.allocate(requirements.scratch), 0};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation, &binding) == PM_SUCCESS);
    PmLiquidMaterialState fluid{};
    fluid.restDensity = 1000; fluid.particleRadius = 0.025F; fluid.supportRadius = 0.1F;
    fluid.maximumSpeed = 100; fluid.densityIterations = 1;
    PmRopeMaterialState rope_material{};
    rope_material.nodeSpacing = 0.01F; rope_material.radius = 0.01F; rope_material.mass = 1;
    rope_material.maximumSubstepTimestep = 1.0F / 60; rope_material.maximumSpeed = 100;
    rope_material.solverIterations = 8;
    PmCouplingState edge{};
    edge.type = PM_COUPLING_LIQUID_ROPE;
    edge.parameters.mechanical.contactDistance = 0.01F;
    edge.parameters.mechanical.friction = 0.5F;
    edge.parameters.mechanical.maximumAcceleration = 10;
    edge.parameters.mechanical.twoWay = PM_TRUE;
    PmPhysicsPipelineCreateInfo policy{};
    policy.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    policy.fluidMaterialCount = policy.ropeMaterialCount = 1;
    policy.pFluidMaterials = &fluid; policy.pRopeMaterials = &rope_material;
    PmPhysicsPipeline empty{}, coupled{}, duplicate{}, excessive{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &policy, &empty) == PM_SUCCESS);
    policy.couplingCount = 1; policy.pCouplings = &edge;
    edge.parameters.mechanical.twoWay = PM_FALSE;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &policy, &coupled) == PM_ERROR_FEATURE_NOT_PRESENT);
    PM_CHECK(!coupled);
    edge.parameters.mechanical.twoWay = PM_TRUE;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &policy, &coupled) == PM_SUCCESS);
    const PmCouplingState duplicates[3]{edge, edge, edge};
    policy.couplingCount = 2; policy.pCouplings = duplicates;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &policy, &duplicate) == PM_SUCCESS);
    policy.couplingCount = 3;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &policy, &excessive) == PM_SUCCESS);
    const PmFloat3 line[2]{{2, 1, 0}, {2.08F, 1, 0}};
    PmRopeInfo rope{}; rope.active = PM_TRUE; rope.pointCount = 2; rope.pCenterline = line;
    const PmStepInfo step{1.0F / 60, 1, {0, 0, 0}, 0};
    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, coupled) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0, &particles) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteRopes(context.commands, simulation, 0, 1, &rope) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    for (unsigned edit = 0; edit < 32; ++edit) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, coupled) == PM_SUCCESS);
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0, &particles) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
        PM_CHECK(pmCmdWriteRopes(context.commands, simulation, 0, 1, &rope) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
        // Switching to empty policy retires edges. Switching back recreates
        // them with the current endpoint generations, within bound capacity.
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, coupled) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
        const auto result = context.submit();
        if (result != PM_SUCCESS) std::fprintf(stderr, "coupled replacement %u: %d\n", edit, result);
        PM_CHECK(result == PM_SUCCESS);
    }
    for (const auto policy_to_reject : {duplicate, excessive}) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, policy_to_reject) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
        PM_CHECK(context.submit() == (policy_to_reject == duplicate ? PM_ERROR_INVALID_ARGUMENT : PM_ERROR_CAPACITY_EXCEEDED));
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, coupled) == PM_SUCCESS);
        PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }
    // Explicit graph policy belongs to the simulation and overrides legacy
    // pipeline edge lists, including an explicitly empty graph.
    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 0, nullptr) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, duplicate) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 1, &edge) == PM_SUCCESS);
    edge.second = 99; // Recording owns a descriptor snapshot.
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    edge.second = 0;
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0, &particles) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 3, duplicates) == PM_ERROR_CAPACITY_EXCEEDED);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 2, duplicates) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, empty) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_ERROR_INVALID_ARGUMENT);
    context.begin();
    PM_CHECK(pmCmdSetCouplings(context.commands, simulation, 0, nullptr) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, coupled) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    pmDestroySimulation(context.device, simulation);
    for (auto pipeline : {empty, coupled, duplicate, excessive}) pmDestroyPhysicsPipeline(context.device, pipeline);
    std::puts("PASS: coupled endpoint replacement, policy changes, capacity and graph rollback");
}
