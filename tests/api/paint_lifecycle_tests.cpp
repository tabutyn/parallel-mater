// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

using parallel_mater::test::Context;

namespace {

constexpr std::array<float, 24> cube_vertices{
    -0.5F, -0.5F, -0.5F,  0.5F, -0.5F, -0.5F,
     0.5F,  0.5F, -0.5F,  0.5F, -0.5F,
    -0.5F, -0.5F,  0.5F,  0.5F, -0.5F,  0.5F,
     0.5F,  0.5F,  0.5F, -0.5F,  0.5F,  0.5F};
constexpr std::array<std::uint32_t, 36> cube_indices{
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5};
constexpr std::array<PmFloat2, 8> paint_uvs{{
    {0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F},
    {0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}}};

PmRigidBodyInfo body(PmGeometry geometry, float x) {
    PmRigidBodyInfo result{};
    result.active = PM_TRUE;
    result.motionType = PM_MOTION_STATIC;
    result.geometry = geometry;
    result.state.position[0] = x;
    result.state.orientation[3] = 1.0F;
    result.friction = 0.5F;
    result.maximumLinearSpeed = 100.0F;
    result.maximumAngularSpeed = 100.0F;
    result.collisionMargin = 0.005F;
    return result;
}

PmPaintFieldInfo field(std::uint32_t body_index, PmGeometry geometry) {
    PmPaintFieldInfo result{};
    result.active = PM_TRUE;
    result.targetType = PM_PAINT_TARGET_RIGID;
    result.targetIndex = body_index;
    result.geometry = geometry;
    result.vertexCount = static_cast<std::uint32_t>(paint_uvs.size());
    result.pVertexUvs = paint_uvs.data();
    result.width = 2U;
    result.height = 2U;
    return result;
}

} // namespace

int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.rigidBodies = PM_TRUE;
    needed.paint = PM_TRUE;
    if (!context.open(needed)) return 77;

    const auto source = context.buffer(
        sizeof(cube_vertices) + sizeof(cube_indices),
        PM_BUFFER_GEOMETRY_INPUT_BIT);
    std::memcpy(source.mapped, cube_vertices.data(), sizeof(cube_vertices));
    std::memcpy(static_cast<std::byte *>(source.mapped) + sizeof(cube_vertices),
                cube_indices.data(), sizeof(cube_indices));
    context.flush(source);

    // The first 18 geometries are live together: 16 body meshes plus two
    // paint-only meshes. Later geometries exercise successful and failed reuse.
    constexpr std::size_t geometry_count = 48U;
    std::array<PmGeometry, geometry_count> geometries{};
    PmGeometryCreateInfo geometry_info{};
    geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    geometry_info.vertices = {
        {source.handle, 0U, sizeof(cube_vertices)},
        PM_FORMAT_R32G32B32_SFLOAT, 8U, 12U};
    geometry_info.indices = {
        {source.handle, sizeof(cube_vertices), sizeof(cube_indices)},
        PM_FORMAT_R32_UINT, 36U, 4U};
    for (auto &geometry : geometries) {
        PM_CHECK(pmCreateGeometry(context.device, &geometry_info, &geometry) ==
                 PM_SUCCESS);
        PmMemoryRequirements requirements{};
        pmGetGeometryMemoryRequirements(context.device, geometry, &requirements);
        PM_CHECK(pmBindGeometryMemory(
                     context.device, geometry, context.allocate(requirements), 0U) ==
                 PM_SUCCESS);
    }
    context.begin();
    for (const auto geometry : geometries)
        PM_CHECK(pmCmdBuildGeometry(context.commands, geometry) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    constexpr std::uint32_t rigid_capacity = 16U;
    const std::array<PmPaintFieldCapacity, 2> paint_capacities{{
        {8U, 2U, 2U}, {8U, 2U, 2U}}};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = rigid_capacity;
    simulation_info.contactCapacity = 4U;
    simulation_info.paintFieldCount =
        static_cast<std::uint32_t>(paint_capacities.size());
    simulation_info.pPaintFieldCapacities = paint_capacities.data();
    simulation_info.paintRuleCapacity = 1U;
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

    std::array<PmRigidBodyInfo, rigid_capacity> bodies{};
    for (std::uint32_t index = 0U; index < rigid_capacity; ++index)
        bodies[index] = body(geometries[index], static_cast<float>(index));
    std::array<PmPaintFieldInfo, 2> fields{
        field(0U, geometries[16]), field(1U, geometries[17])};
    context.begin();
    PM_CHECK(pmCmdWriteRigidBodies(
                 context.commands, simulation, 0U, rigid_capacity,
                 bodies.data()) == PM_SUCCESS);
    PM_CHECK(pmCmdWritePaintFields(
                 context.commands, simulation, 0U,
                 static_cast<std::uint32_t>(fields.size()), fields.data()) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const PmDeviceSize paint_bytes =
        sizeof(PmDomainOutputHeader) + 4U * sizeof(std::uint32_t);
    const auto output = context.buffer(
        256U, PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange paint_output{output.handle, 0U, paint_bytes};
    const PmBufferRange rigid_output{
        output.handle, 128U, sizeof(PmRigidState)};
    const auto expect_field = [&](std::uint32_t index) {
        context.begin();
        PM_CHECK(pmCmdCopyPaintField(context.commands, simulation, index,
                                     &paint_output) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
        context.invalidate(output);
        const auto *header =
            static_cast<const PmDomainOutputHeader *>(output.mapped);
        PM_CHECK(header->activeCount == 4U && header->capacity == 4U);
    };

    // Replacing one field repeatedly must recycle its private mesh while the
    // other field and all body meshes remain live.
    for (std::size_t index = 18U; index < 28U; ++index) {
        fields[0] = field(0U, geometries[index]);
        context.begin();
        PM_CHECK(pmCmdWritePaintFields(context.commands, simulation, 0U, 1U,
                                       fields.data()) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }
    expect_field(1U);

    // Runtime UV validation happens after geometry registration. Every failed
    // add must release that registration, or this loop exhausts mesh capacity.
    for (std::size_t index = 28U; index < geometries.size(); ++index) {
        auto invalid = field(0U, geometries[index]);
        invalid.vertexCount = 7U;
        context.begin();
        PM_CHECK(pmCmdWritePaintFields(context.commands, simulation, 0U, 1U,
                                       &invalid) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_ERROR_INVALID_ARGUMENT);
    }
    fields[0] = field(0U, geometries[16]);
    context.begin();
    PM_CHECK(pmCmdWritePaintFields(context.commands, simulation, 0U, 1U,
                                   fields.data()) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    expect_field(1U);

    // A live field pins its target generation. Replacing that body fails before
    // mutation, and both the old body and an unrelated field remain usable.
    const auto replacement = body(geometries[1], 71.0F);
    context.begin();
    PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 1U, 1U,
                                   &replacement) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_ERROR_INVALID_ARGUMENT);
    expect_field(1U);
    context.begin();
    PM_CHECK(pmCmdCopyRigidStates(context.commands, simulation, 1U, 1U,
                                  &rigid_output) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    PM_CHECK(std::abs(reinterpret_cast<const PmRigidState *>(
                          static_cast<const std::byte *>(output.mapped) +
                          rigid_output.offset)->position[0] - 1.0F) < 1.0e-6F);

    // A rule pins its target field generation. Remove the rule first, then the
    // field; the formerly pinned body can then be replaced normally.
    PmPaintRuleInfo rule{};
    rule.active = PM_TRUE;
    rule.sourceType = PM_PAINT_SOURCE_RIGID;
    rule.sourceIndex = 2U;
    rule.targetFieldIndex = 1U;
    rule.reach = 0.1F;
    rule.brushRadius = 0.05F;
    context.begin();
    PM_CHECK(pmCmdWritePaintRules(context.commands, simulation, 0U, 1U,
                                  &rule) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    const PmPaintFieldInfo inactive_field{};
    context.begin();
    PM_CHECK(pmCmdWritePaintFields(context.commands, simulation, 1U, 1U,
                                   &inactive_field) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_ERROR_INVALID_ARGUMENT);
    expect_field(1U);
    const PmPaintRuleInfo inactive_rule{};
    context.begin();
    PM_CHECK(pmCmdWritePaintRules(context.commands, simulation, 0U, 1U,
                                  &inactive_rule) == PM_SUCCESS);
    PM_CHECK(pmCmdWritePaintFields(context.commands, simulation, 1U, 1U,
                                   &inactive_field) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.begin();
    PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 1U, 1U,
                                   &replacement) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    // Reset retires all paint and body generations and clears the public mesh
    // registry. Re-register the same handles repeatedly in submission order.
    for (std::uint32_t iteration = 0U; iteration < 4U; ++iteration) {
        const auto reset_body =
            body(geometries[16], 80.0F + static_cast<float>(iteration));
        const auto reset_field = field(0U, geometries[16]);
        context.begin();
        PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
        PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 0U, 1U,
                                       &reset_body) == PM_SUCCESS);
        PM_CHECK(pmCmdWritePaintFields(context.commands, simulation, 0U, 1U,
                                       &reset_field) == PM_SUCCESS);
        PM_CHECK(pmCmdCopyPaintField(context.commands, simulation, 0U,
                                     &paint_output) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
        context.invalidate(output);
        const auto *header =
            static_cast<const PmDomainOutputHeader *>(output.mapped);
        PM_CHECK(header->activeCount == 4U && header->capacity == 4U);
    }

    pmDestroySimulation(context.device, simulation);
    for (const auto geometry : geometries)
        pmDestroyGeometry(context.device, geometry);
    std::puts("PASS: paint mesh capacity, replacement cleanup, endpoint guards and reset");
}
