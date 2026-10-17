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
     0.5F,  0.5F, -0.5F, -0.5F,  0.5F, -0.5F,
    -0.5F, -0.5F,  0.5F,  0.5F, -0.5F,  0.5F,
     0.5F,  0.5F,  0.5F, -0.5F,  0.5F,  0.5F};
constexpr std::array<std::uint32_t, 36> cube_indices{
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5};

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

} // namespace

int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.rigidBodies = PM_TRUE;
    if (!context.open(needed)) return 77;

    const auto source = context.buffer(
        sizeof(cube_vertices) + sizeof(cube_indices),
        PM_BUFFER_GEOMETRY_INPUT_BIT);
    std::memcpy(source.mapped, cube_vertices.data(), sizeof(cube_vertices));
    std::memcpy(static_cast<std::byte *>(source.mapped) + sizeof(cube_vertices),
                cube_indices.data(), sizeof(cube_indices));
    context.flush(source);

    // More distinct public geometries than the runtime's minimum 16 mesh slots.
    // A one-body replacement sequence can only finish if retired slots recycle.
    constexpr std::size_t geometry_count = 20U;
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

    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 2U;
    simulation_info.contactCapacity = 4U;
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

    // Both bodies initially share one mesh. Replacing slot zero must keep the
    // mesh alive for slot one while retiring each later slot-zero mesh.
    std::array<PmRigidBodyInfo, 2> bodies{
        body(geometries[0], 1.0F), body(geometries[0], 37.0F)};
    context.begin();
    PM_CHECK(pmCmdWriteRigidBodies(
                 context.commands, simulation, 0U,
                 static_cast<std::uint32_t>(bodies.size()), bodies.data()) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    for (std::size_t index = 1U; index < geometries.size(); ++index) {
        bodies[0] = body(geometries[index], static_cast<float>(index));
        context.begin();
        PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 0U, 1U,
                                       bodies.data()) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_SUCCESS);
    }
    // Re-register a previously retired public geometry. Its cached private ID
    // must have been erased so the reused runtime slot gets its new generation.
    bodies[0] = body(geometries[1], 21.0F);
    context.begin();
    PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 0U, 1U,
                                   bodies.data()) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);

    const auto output = context.buffer(
        sizeof(PmRigidState),
        PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    const PmBufferRange output_range{output.handle, 0U, sizeof(PmRigidState)};
    context.begin();
    PM_CHECK(pmCmdCopyRigidStates(context.commands, simulation, 1U, 1U,
                                  &output_range) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    PM_CHECK(std::abs(static_cast<const PmRigidState *>(output.mapped)
                          ->position[0] - 37.0F) < 1.0e-6F);

    // Explicit removal releases slot zero's last mesh reference and leaves the
    // unrelated shared-mesh body valid.
    const PmRigidBodyInfo inactive{};
    context.begin();
    PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 0U, 1U,
                                   &inactive) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRigidStates(context.commands, simulation, 1U, 1U,
                                  &output_range) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    PM_CHECK(std::abs(static_cast<const PmRigidState *>(output.mapped)
                          ->position[0] - 37.0F) < 1.0e-6F);

    // Reset retires every private mesh and reference. The same public geometry
    // can be registered again in the following command in submission order.
    bodies[0] = body(geometries[1], 73.0F);
    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteRigidBodies(context.commands, simulation, 0U, 1U,
                                   bodies.data()) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyRigidStates(context.commands, simulation, 0U, 1U,
                                  &output_range) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    PM_CHECK(std::abs(static_cast<const PmRigidState *>(output.mapped)
                          ->position[0] - 73.0F) < 1.0e-6F);

    pmDestroySimulation(context.device, simulation);
    for (const auto geometry : geometries)
        pmDestroyGeometry(context.device, geometry);
    std::puts("PASS: shared geometry, 20 replacements, stale generations, removal and reset");
}
