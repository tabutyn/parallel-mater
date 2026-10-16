// SPDX-License-Identifier: MIT
#include "test_context.hpp"
#include <array>
#include <cmath>
#include <cstddef>
using parallel_mater::test::Context;
int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.cloth = needed.rope = needed.softBodies = needed.smoke = PM_TRUE;
    if (!context.open(needed)) return 77;
    const auto output = context.buffer(16384, PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmClothCapacity cloth_capacity{6, 3, 3};
    const PmRopeCapacity rope_capacity{16};
    const PmSoftBodyCapacity soft_capacity{4, 6, 4, 12};
    const PmSmokeCapacity smoke_capacity{4, 16 * 16 * 8};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.clothCount = create.ropeCount = create.softBodyCount = create.smokeCount = 1;
    create.pClothCapacities = &cloth_capacity; create.pRopeCapacities = &rope_capacity;
    create.pSoftBodyCapacities = &soft_capacity; create.pSmokeCapacities = &smoke_capacity;
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &create, &simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{context.allocate(requirements.persistent), 0,
        context.allocate(requirements.scratch), 0};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation, &binding) == PM_SUCCESS);
    PmClothMaterialState cloth_material{};
    cloth_material.vertexMass = 1; cloth_material.thickness = 0.01F;
    cloth_material.solverIterations = 4; cloth_material.fracturePersistenceSubsteps = 1;
    cloth_material.breakStrain = 0.5F;
    PmRopeMaterialState rope_material{};
    rope_material.nodeSpacing = 0.01F; rope_material.radius = 0.01F; rope_material.mass = 1;
    rope_material.maximumSubstepTimestep = 1.0F / 60; rope_material.maximumSpeed = 100;
    rope_material.solverIterations = 8;
    PmSoftBodyMaterialState soft_material{};
    soft_material.nodeMass = 1; soft_material.nodeRadius = 0.01F;
    soft_material.maximumProjectionFraction = 0.2F; soft_material.constraintVelocityResponse = 1;
    soft_material.maximumSpeed = 100; soft_material.solverIterations = 4;
    PmSmokeMaterialState smoke_material{};
    smoke_material.particleRadius = 0.02F; smoke_material.restNumberDensity = 12;
    smoke_material.maximumSpeed = 100; smoke_material.gridResolution = 16;
    smoke_material.gridVerticalResolution = 8; smoke_material.gridPressureIterations = 8;
    smoke_material.gridPressureTolerance = 0.001F;
    PmPhysicsPipelineCreateInfo policy{};
    policy.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    policy.clothMaterialCount = policy.ropeMaterialCount = policy.softBodyMaterialCount = policy.smokeMaterialCount = 1;
    policy.pClothMaterials = &cloth_material; policy.pRopeMaterials = &rope_material;
    policy.pSoftBodyMaterials = &soft_material; policy.pSmokeMaterials = &smoke_material;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &policy, &pipeline) == PM_SUCCESS);
    // Alternate layouts within each declared slot: fracture/no fracture and
    // grid/particle smoke. Reserved regions must support both without reset.
    PmPhysicsPipeline compact_pipeline{};
    cloth_material.breakStrain = 0;
    smoke_material.gridResolution = 0;
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &policy, &compact_pipeline) == PM_SUCCESS);
    const PmFloat3 vertices[4]{{0, 1, 0}, {1, 1, 0}, {0, 2, 0}, {0, 1, 1}};
    const std::uint32_t indices[12]{0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};
    const float masses[4]{1, 1, 1, 1};
    const PmClothInfo cloth{PM_TRUE, 0, 3, vertices, 3, indices, masses};
    const PmFloat3 line[2]{{2, 1, 0}, {2.08F, 1, 0}};
    PmRopeInfo rope{}; rope.active = PM_TRUE; rope.pointCount = 2; rope.pCenterline = line;
    const PmFloat3 short_line[2]{{2, 1, 0}, {2.02F, 1, 0}};
    auto short_rope = rope; short_rope.pCenterline = short_line;
    std::array<PmSoftBodyBond, 6> bonds{};
    unsigned bond = 0;
    for (unsigned a = 0; a < 4; ++a) for (unsigned b = a + 1; b < 4; ++b) {
        const auto dx = vertices[a].x - vertices[b].x, dy = vertices[a].y - vertices[b].y,
                   dz = vertices[a].z - vertices[b].z;
        bonds[bond++] = {a, b, std::sqrt(dx * dx + dy * dy + dz * dz)};
    }
    std::array<PmSoftBodySurfaceBinding, 4> skin{};
    for (unsigned i = 0; i < 4; ++i) { skin[i].nodes[0] = i; skin[i].weights[0] = 1; }
    const PmSoftBodyInfo soft{PM_TRUE, 0, 4, vertices, 6, bonds.data(), masses,
        4, vertices, 12, indices, skin.data()};
    PmSmokeInfo smoke{}; smoke.active = PM_TRUE; smoke.particleCapacity = 4;
    smoke.emitterCenter = {4, 2, 0}; smoke.emitterHalfExtents = {0.1F, 0.1F};
    smoke.particlesPerSecond = 60; smoke.lifetime = 2;
    smoke.gridMinimum = {0, 0, -4}; smoke.gridEdgeLength = 8;
    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    for (unsigned batch = 0; batch < 4; ++batch) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
        for (unsigned edit = 0; edit < 64; ++edit) {
            PM_CHECK(pmCmdBindPhysicsPipeline(context.commands,
                edit % 2 ? pipeline : compact_pipeline) == PM_SUCCESS);
            PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0, 1, &cloth) == PM_SUCCESS);
            PM_CHECK(pmCmdWriteRopes(context.commands, simulation, 0, 1, edit % 2 ? &rope : &short_rope) == PM_SUCCESS);
            PM_CHECK(pmCmdWriteSoftBodies(context.commands, simulation, 0, 1, &soft) == PM_SUCCESS);
            PM_CHECK(pmCmdWriteSmoke(context.commands, simulation, 0, 1, &smoke) == PM_SUCCESS);
        }
        const PmBufferRange ranges[4]{{output.handle, 0, 4096}, {output.handle, 4096, 4096},
            {output.handle, 8192, 4096}, {output.handle, 12288, 4096}};
        PM_CHECK(pmCmdCopyClothState(context.commands, simulation, 0, &ranges[0]) == PM_SUCCESS);
        PM_CHECK(pmCmdCopyRopeState(context.commands, simulation, 0, &ranges[1]) == PM_SUCCESS);
        PM_CHECK(pmCmdCopySoftBodyState(context.commands, simulation, 0, &ranges[2]) == PM_SUCCESS);
        PM_CHECK(pmCmdCopySmokeState(context.commands, simulation, 0, &ranges[3]) == PM_SUCCESS);
        const auto result = context.submit();
        if (result != PM_SUCCESS) std::fprintf(stderr, "domain batch %u: %d\n", batch, result);
        PM_CHECK(result == PM_SUCCESS);
        context.invalidate(output);
        for (unsigned domain = 0; domain < 4; ++domain) {
            const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(
                static_cast<const std::byte *>(output.mapped) + 4096 * domain);
            PM_CHECK(header->activeCount <= header->capacity);
            if (domain == 0) {
                PM_CHECK(header->activeCount == 3);
                PM_CHECK(reinterpret_cast<const PmClothVertexState *>(header + 1)[0].position.y == 1);
            }
            if (domain == 1) PM_CHECK(header->activeCount >= 9 && header->activeCount <= 16);
            if (domain == 2) {
                PM_CHECK(header->activeCount == 4);
                PM_CHECK(reinterpret_cast<const PmSoftBodyNodeState *>(header + 1)[0].position.y == 1);
            }
            if (domain == 3) PM_CHECK(header->activeCount == 0);
        }
    }
    // Two authored points can expand beyond node capacity during resampling.
    const PmFloat3 long_line[2]{{2, 1, 0}, {3, 1, 0}};
    auto excessive = rope; excessive.pCenterline = long_line;
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteRopes(context.commands, simulation, 0, 1, &excessive) == PM_ERROR_CAPACITY_EXCEEDED);
    PM_CHECK(context.submit() == PM_SUCCESS);
    // Fracture capacity includes source vertices plus potential new corners.
    auto small_create = create;
    const PmClothCapacity small_cloth{3, 3, 3}; small_create.pClothCapacities = &small_cloth;
    PmSimulation small{};
    PM_CHECK(pmCreateSimulation(context.device, &small_create, &small) == PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, small, 0, 1, &cloth) == PM_ERROR_CAPACITY_EXCEEDED);
    PM_CHECK(context.submit() == PM_SUCCESS);
    pmDestroySimulation(context.device, small);
    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    pmDestroyPhysicsPipeline(context.device, compact_pipeline);
    std::puts("PASS: exact storage, 1024 mixed-domain writes and generated capacity bounds");
}
