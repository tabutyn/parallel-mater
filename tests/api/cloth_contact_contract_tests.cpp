// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

using parallel_mater::test::Context;

int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.cloth = PM_TRUE;
    if (!context.open(needed)) return 77;

    const PmClothCapacity capacity{4U, 6U, 6U};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.clothCount = 1U;
    simulation_info.pClothCapacities = &capacity;
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &simulation_info, &simulation) ==
             PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
    const PmSimulationMemoryBinding memory{
        context.allocate(requirements.persistent), 0U,
        context.allocate(requirements.scratch), 0U};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation, &memory) ==
             PM_SUCCESS);

    PmClothMaterialState material{};
    material.vertexMass = 1.0F;
    material.thickness = 0.05F;
    material.stretchCompliance = 0.001F;
    material.bendingCompliance = 0.001F;
    material.contactFriction = 0.5F;
    material.solverIterations = 4U;
    material.fracturePersistenceSubsteps = 1U;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &material;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const std::array<PmFloat3, 4> vertices{{
        {0.0F, 1.0F, 0.0F},
        {1.0F, 1.0F, 0.0F},
        {1.0F, 2.0F, 0.0F},
        {0.0F, 2.0F, 0.0F}}};
    const std::array<std::uint32_t, 6> indices{0U, 1U, 2U, 0U, 2U, 3U};
    const std::array<float, 4> inverse_masses{1.0F, 1.0F, 1.0F, 1.0F};
    const PmClothInfo initial{PM_TRUE, 0U, 4U, vertices.data(), 6U,
                              indices.data(), inverse_masses.data()};
    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U, &initial) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    PmClothSelfContactState self_contact{
        PM_STRUCTURE_TYPE_CLOTH_SELF_CONTACT_STATE, nullptr, PM_FALSE};
    PmClothStateInfo state{
        PM_STRUCTURE_TYPE_CLOTH_STATE_INFO,
        &self_contact,
        PM_TRUE,
        0U,
        4U,
        vertices.data(),
        6U,
        indices.data(),
        inverse_masses.data()};

    // The disabled extension is an ABI-compatible resource write.
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths2(context.commands, simulation, 0U, 1U, &state) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const auto output = context.buffer(
        4096U, PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange output_range{output.handle, 0U, output.size};
    const auto reject_without_replacement = [&](const PmClothStateInfo &value,
                                                PmResult expected) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdWriteCloths2(context.commands, simulation, 0U, 1U,
                                   &value) == expected);
        PM_CHECK(pmCmdCopyClothState(context.commands, simulation, 0U,
                                    &output_range) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
        context.invalidate(output);
        const auto *header =
            static_cast<const PmDomainOutputHeader *>(output.mapped);
        PM_CHECK(header->activeCount == 4U);
    };

    self_contact.enabled = PM_TRUE;
    reject_without_replacement(state, PM_ERROR_FEATURE_NOT_PRESENT);

    auto invalid_state = state;
    invalid_state.sType = PM_STRUCTURE_TYPE_CLOTH_SELF_CONTACT_STATE;
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto unknown_control = self_contact;
    unknown_control.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    invalid_state = state;
    invalid_state.pNext = &unknown_control;
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto nested_control = self_contact;
    nested_control.pNext = &self_contact;
    invalid_state.pNext = &nested_control;
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto invalid_control = self_contact;
    invalid_control.enabled = 2U;
    invalid_state.pNext = &invalid_control;
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    invalid_state = state;
    invalid_state.active = PM_FALSE;
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto invalid_indices = indices;
    invalid_indices[5] = 4U;
    invalid_state = state;
    invalid_state.pTriangleIndices = invalid_indices.data();
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto degenerate_indices = indices;
    degenerate_indices[1] = degenerate_indices[0];
    invalid_state.pTriangleIndices = degenerate_indices.data();
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    const std::array<std::uint32_t, 6> duplicate_indices{
        0U, 1U, 2U, 2U, 1U, 0U};
    invalid_state.pTriangleIndices = duplicate_indices.data();
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto invalid_vertices = vertices;
    invalid_vertices[0].x = std::numeric_limits<float>::infinity();
    invalid_state = state;
    invalid_state.pVertices = invalid_vertices.data();
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto invalid_masses = inverse_masses;
    invalid_masses[0] = -1.0F;
    invalid_state = state;
    invalid_state.pInverseMasses = invalid_masses.data();
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    pmDestroyPhysicsPipeline(context.device, pipeline);
    pmDestroySimulation(context.device, simulation);
    std::puts("PASS: cloth contact controls reject atomically");
}
