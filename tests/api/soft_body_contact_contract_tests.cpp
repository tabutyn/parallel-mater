// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

using parallel_mater::test::Context;

int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.softBodies = PM_TRUE;
    if (!context.open(needed)) return 77;

    const PmSoftBodyCapacity capacity{4U, 6U, 7U, 24U};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.softBodyCount = 1U;
    simulation_info.pSoftBodyCapacities = &capacity;
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

    PmSoftBodyMaterialState material{};
    material.nodeMass = 1.0F;
    material.nodeRadius = 0.05F;
    material.springDamping = 0.0F;
    material.contactFriction = 0.5F;
    material.maximumProjectionFraction = 0.2F;
    material.constraintVelocityResponse = 1.0F;
    material.maximumSpeed = 20.0F;
    material.solverIterations = 4U;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.softBodyMaterialCount = 1U;
    pipeline_info.pSoftBodyMaterials = &material;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const std::array<PmFloat3, 4> nodes{{
        {0.0F, 1.0F, 0.0F},
        {1.0F, 1.0F, 0.0F},
        {0.0F, 2.0F, 0.0F},
        {0.0F, 1.0F, 1.0F}}};
    const std::array<std::uint32_t, 12> closed_indices{
        0U, 2U, 1U, 0U, 1U, 3U, 0U, 3U, 2U, 1U, 2U, 3U};
    const std::array<float, 4> inverse_masses{1.0F, 1.0F, 1.0F, 1.0F};
    std::array<PmSoftBodyBond, 6> bonds{};
    std::uint32_t bond_count = 0U;
    for (std::uint32_t first = 0U; first < nodes.size(); ++first) {
        for (std::uint32_t second = first + 1U; second < nodes.size();
             ++second) {
            const float dx = nodes[first].x - nodes[second].x;
            const float dy = nodes[first].y - nodes[second].y;
            const float dz = nodes[first].z - nodes[second].z;
            bonds[bond_count++] = {
                first, second, std::sqrt(dx * dx + dy * dy + dz * dz)};
        }
    }
    std::array<PmSoftBodySurfaceBinding, 4> bindings{};
    for (std::uint32_t index = 0U; index < bindings.size(); ++index) {
        bindings[index].nodes[0] = index;
        bindings[index].weights[0] = 1.0F;
    }

    const PmSoftBodyInfo initial{
        PM_TRUE, 0U, 4U, nodes.data(), 6U, bonds.data(),
        inverse_masses.data(), 4U, nodes.data(), 12U,
        closed_indices.data(), bindings.data()};
    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSoftBodies(context.commands, simulation, 0U, 1U,
                                  &initial) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    PmSoftBodySelfContactState self_contact{
        PM_STRUCTURE_TYPE_SOFT_BODY_SELF_CONTACT_STATE, nullptr, PM_FALSE};
    PmSoftBodyStateInfo state{
        PM_STRUCTURE_TYPE_SOFT_BODY_STATE_INFO,
        &self_contact,
        PM_TRUE,
        0U,
        4U,
        nodes.data(),
        6U,
        bonds.data(),
        inverse_masses.data(),
        4U,
        nodes.data(),
        12U,
        closed_indices.data(),
        bindings.data()};

    // The disabled extension is an ABI-compatible resource write.
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteSoftBodies2(context.commands, simulation, 0U, 1U,
                                   &state) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const auto output = context.buffer(
        4096U, PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange output_range{output.handle, 0U, output.size};
    const auto reject_without_replacement = [&](const PmSoftBodyStateInfo &value,
                                                PmResult expected) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) ==
                 PM_SUCCESS);
        PM_CHECK(pmCmdWriteSoftBodies2(context.commands, simulation, 0U, 1U,
                                       &value) == expected);
        PM_CHECK(pmCmdCopySoftBodyState(context.commands, simulation, 0U,
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
    invalid_state.sType = PM_STRUCTURE_TYPE_SOFT_BODY_SELF_CONTACT_STATE;
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

    invalid_state = state;
    invalid_state.surfaceTriangleIndexCount = 9U;
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto inconsistent_indices = closed_indices;
    std::swap(inconsistent_indices[0], inconsistent_indices[1]);
    invalid_state = state;
    invalid_state.pSurfaceTriangleIndices = inconsistent_indices.data();
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto degenerate_indices = closed_indices;
    degenerate_indices[1] = degenerate_indices[0];
    invalid_state.pSurfaceTriangleIndices = degenerate_indices.data();
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    const std::array<PmFloat3, 7> pinched_vertices{{
        nodes[0], nodes[1], nodes[2], nodes[3],
        {-1.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, -1.0F}}};
    const std::array<std::uint32_t, 24> pinched_indices{
        0U, 2U, 1U, 0U, 1U, 3U, 0U, 3U, 2U, 1U, 2U, 3U,
        0U, 5U, 4U, 0U, 4U, 6U, 0U, 6U, 5U, 4U, 5U, 6U};
    std::array<PmSoftBodySurfaceBinding, 7> pinched_bindings{};
    for (auto &binding : pinched_bindings)
        binding.weights[0] = 1.0F;
    invalid_state = state;
    invalid_state.surfaceVertexCount =
        static_cast<std::uint32_t>(pinched_vertices.size());
    invalid_state.pSurfaceVertices = pinched_vertices.data();
    invalid_state.surfaceTriangleIndexCount =
        static_cast<std::uint32_t>(pinched_indices.size());
    invalid_state.pSurfaceTriangleIndices = pinched_indices.data();
    invalid_state.pSurfaceBindings = pinched_bindings.data();
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    auto invalid_bindings = bindings;
    invalid_bindings[0].weights[0] = 0.5F;
    invalid_state = state;
    invalid_state.pSurfaceBindings = invalid_bindings.data();
    reject_without_replacement(invalid_state, PM_ERROR_INVALID_ARGUMENT);

    pmDestroyPhysicsPipeline(context.device, pipeline);
    pmDestroySimulation(context.device, simulation);
    std::puts("PASS: soft-body contact controls reject atomically");
}
