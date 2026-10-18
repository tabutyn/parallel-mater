// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

using parallel_mater::test::Buffer;
using parallel_mater::test::Context;

namespace {

constexpr std::uint32_t cloth_count = 2U;
constexpr std::uint32_t vertex_count = 3U;
constexpr std::uint32_t index_count = 3U;
constexpr PmDeviceSize cloth_output_stride = 512U;

struct ClothInput {
    std::unique_ptr<PmFloat3[]> vertices =
        std::make_unique<PmFloat3[]>(vertex_count);
    std::unique_ptr<std::uint32_t[]> indices =
        std::make_unique<std::uint32_t[]>(index_count);
    std::unique_ptr<float[]> inverse_masses =
        std::make_unique<float[]>(vertex_count);
    PmClothInfo info{};
};

ClothInput make_cloth(float x) {
    ClothInput input;
    input.vertices[0] = {x, 0.0F, 0.0F};
    input.vertices[1] = {x + 1.0F, 0.0F, 0.0F};
    input.vertices[2] = {x, 1.0F, 0.0F};
    input.indices[0] = 0U;
    input.indices[1] = 2U;
    input.indices[2] = 1U;
    for (std::uint32_t index = 0U; index < vertex_count; ++index)
        input.inverse_masses[index] = 1.0F;
    input.info.active = PM_TRUE;
    input.info.vertexCount = vertex_count;
    input.info.pVertices = input.vertices.get();
    input.info.triangleIndexCount = index_count;
    input.info.pTriangleIndices = input.indices.get();
    input.info.pInverseMasses = input.inverse_masses.get();
    return input;
}

void discard_caller_storage(ClothInput &input) {
    for (std::uint32_t index = 0U; index < vertex_count; ++index) {
        input.vertices[index] = {99.0F, 99.0F, 99.0F};
        input.inverse_masses[index] = 0.0F;
    }
    for (std::uint32_t index = 0U; index < index_count; ++index)
        input.indices[index] = 99U;
    input.info = {};
    input.vertices.reset();
    input.indices.reset();
    input.inverse_masses.reset();
}

void set_failure(const char *name, bool enabled) {
#if defined(_WIN32)
    PM_CHECK(_putenv_s(name, enabled ? "1" : "") == 0);
#else
    PM_CHECK(enabled ? setenv(name, "1", 1) == 0 : unsetenv(name) == 0);
#endif
}

void expect_transfer(const Buffer &destination, std::uint32_t first,
                     std::uint32_t second) {
    const auto *words = static_cast<const std::uint32_t *>(destination.mapped);
    PM_CHECK(words[0] == first);
    PM_CHECK(words[1] == second);
    for (std::size_t index = 2U; index < 8U; ++index)
        PM_CHECK(words[index] == 0xccccccccU);
}

void expect_cloth(const Buffer &output, std::uint32_t cloth, float x) {
    const auto *bytes = static_cast<const std::byte *>(output.mapped) +
        cloth * cloth_output_stride;
    const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(bytes);
    PM_CHECK(header->activeCount == vertex_count);
    PM_CHECK(header->capacity == vertex_count);
    const auto *states = reinterpret_cast<const PmClothVertexState *>(header + 1);
    const std::array<PmFloat3, vertex_count> expected{{
        {x, 0.0F, 0.0F}, {x + 1.0F, 0.0F, 0.0F}, {x, 1.0F, 0.0F}}};
    for (std::uint32_t index = 0U; index < vertex_count; ++index) {
        PM_CHECK(std::abs(states[index].position.x - expected[index].x) < 1.0e-6F);
        PM_CHECK(std::abs(states[index].position.y - expected[index].y) < 1.0e-6F);
        PM_CHECK(std::abs(states[index].position.z - expected[index].z) < 1.0e-6F);
    }
}

void copy_cloths(Context &context, PmSimulation simulation,
                 const Buffer &output) {
    for (std::uint32_t cloth = 0U; cloth < cloth_count; ++cloth) {
        const PmBufferRange range{
            output.handle, cloth * cloth_output_stride, cloth_output_stride};
        PM_CHECK(pmCmdCopyClothState(context.commands, simulation, cloth,
                                     &range) == PM_SUCCESS);
    }
}

} // namespace

int main(int argc, char **argv) {
    const bool failure_fixture =
        argc == 2 && std::strcmp(argv[1], "--failure-fixture") == 0;

    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.cloth = PM_TRUE;
    if (!context.open(needed)) return 77;

    // Flat transfer arrays are copied while recording. Mutating and freeing the
    // caller array before submission cannot redirect the copy.
    const std::array<std::uint32_t, 8> source_words{{
        0x11111111U, 0x22222222U, 0x33333333U, 0x44444444U,
        0x55555555U, 0x66666666U, 0x77777777U, 0x88888888U}};
    const auto usage = PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT;
    const Buffer source = context.buffer(sizeof(source_words), usage);
    const Buffer destination = context.buffer(sizeof(source_words), usage);
    std::memcpy(source.mapped, source_words.data(), sizeof(source_words));
    std::memset(destination.mapped, 0xcc, sizeof(source_words));
    context.flush(source);
    context.flush(destination);

    auto regions = std::make_unique<PmBufferCopy[]>(2U);
    regions[0] = {0U, 0U, sizeof(std::uint32_t)};
    regions[1] = {2U * sizeof(std::uint32_t), sizeof(std::uint32_t),
                  sizeof(std::uint32_t)};
    context.begin();
    PM_CHECK(pmCmdCopyBuffer(context.commands, source.handle, destination.handle,
                             2U, regions.get()) == PM_SUCCESS);
    regions[0].srcOffset = sizeof(std::uint32_t);
    regions[1].srcOffset = 3U * sizeof(std::uint32_t);
    regions.reset();
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(destination);
    expect_transfer(destination, source_words[0], source_words[2]);

    // Reset destroys an unsubmitted flat snapshot. Only the command recorded
    // after reset may execute.
    std::memset(destination.mapped, 0xcc, sizeof(source_words));
    context.flush(destination);
    regions = std::make_unique<PmBufferCopy[]>(1U);
    regions[0] = {0U, 0U, sizeof(std::uint32_t)};
    context.begin();
    PM_CHECK(pmCmdCopyBuffer(context.commands, source.handle, destination.handle,
                             1U, regions.get()) == PM_SUCCESS);
    regions.reset();
    context.begin();
    const PmBufferCopy surviving_region{
        sizeof(std::uint32_t), sizeof(std::uint32_t), sizeof(std::uint32_t)};
    PM_CHECK(pmCmdCopyBuffer(context.commands, source.handle, destination.handle,
                             1U, &surviving_region) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(destination);
    expect_transfer(destination, 0xccccccccU, source_words[1]);

    const std::array<PmClothCapacity, cloth_count> cloth_capacities{{
        {vertex_count, index_count, index_count},
        {vertex_count, index_count, index_count}}};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.clothCount = cloth_count;
    simulation_info.pClothCapacities = cloth_capacities.data();
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

    PmClothMaterialState cloth_material{};
    cloth_material.vertexMass = 1.0F;
    cloth_material.thickness = 0.01F;
    cloth_material.solverIterations = 1U;
    cloth_material.fracturePersistenceSubsteps = 1U;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.clothMaterialCount = 1U;
    pipeline_info.pClothMaterials = &cloth_material;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);

    const Buffer output = context.buffer(
        cloth_count * cloth_output_stride,
        PM_BUFFER_TRANSFER_DST_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT);

    // Each outer descriptor and every nested array are snapshots. Destroy all
    // caller storage before execution and verify the authored vertices survive.
    std::array<ClothInput, cloth_count> inputs{
        make_cloth(10.0F), make_cloth(20.0F)};
    auto descriptors = std::make_unique<PmClothInfo[]>(cloth_count);
    for (std::uint32_t index = 0U; index < cloth_count; ++index)
        descriptors[index] = inputs[index].info;
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, cloth_count,
                              descriptors.get()) == PM_SUCCESS);
    for (auto &input : inputs) discard_caller_storage(input);
    for (std::uint32_t index = 0U; index < cloth_count; ++index)
        descriptors[index] = {};
    descriptors.reset();
    copy_cloths(context, simulation, output);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    expect_cloth(output, 0U, 10.0F);
    expect_cloth(output, 1U, 20.0F);

    // Rejection must not enqueue a partial nested descriptor or retain any of
    // its caller-owned storage. The preceding live cloth remains unchanged.
    ClothInput rejected = make_cloth(30.0F);
    rejected.info.pTriangleIndices = nullptr;
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U,
                              &rejected.info) == PM_ERROR_INVALID_ARGUMENT);
    discard_caller_storage(rejected);
    copy_cloths(context, simulation, output);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    expect_cloth(output, 0U, 10.0F);
    expect_cloth(output, 1U, 20.0F);

    // Reset must release a successful nested snapshot without executing it.
    ClothInput reset_input = make_cloth(40.0F);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U,
                              &reset_input.info) == PM_SUCCESS);
    discard_caller_storage(reset_input);
    context.begin();
    copy_cloths(context, simulation, output);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    expect_cloth(output, 0U, 10.0F);

    if (failure_fixture) {
        // The fixture fails after nested arrays have been copied but before the
        // command is retained. The error is latched; reset releases temporary
        // snapshots and permits an immediate valid submission.
        ClothInput failed = make_cloth(50.0F);
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
        set_failure("PM_TEST_FAIL_RECORDING", true);
        PM_CHECK(pmCmdWriteCloths(context.commands, simulation, 0U, 1U,
                                  &failed.info) == PM_ERROR_OUT_OF_HOST_MEMORY);
        set_failure("PM_TEST_FAIL_RECORDING", false);
        discard_caller_storage(failed);
        PM_CHECK(pmEndCommandBuffer(context.commands) ==
                 PM_ERROR_OUT_OF_HOST_MEMORY);
        context.begin();
        copy_cloths(context, simulation, output);
        PM_CHECK(context.submit() == PM_SUCCESS);
        context.invalidate(output);
        expect_cloth(output, 0U, 10.0F);
    }

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    std::puts(failure_fixture
        ? "PASS: flat/nested snapshots, rejection, reset and recording OOM"
        : "PASS: flat/nested snapshots, rejection and reset");
}
