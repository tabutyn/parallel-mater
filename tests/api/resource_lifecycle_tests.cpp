// SPDX-License-Identifier: MIT
#include "test_context.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <utility>
#include <limits>

using parallel_mater::test::Context;
static int run(std::uint32_t rigid_capacity, std::uint32_t particle_capacity) {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.particleLiquids = PM_TRUE;
    needed.rigidBodies = PM_TRUE;
    if (!context.open(needed)) return 77;
    const auto source = context.buffer(512, PM_BUFFER_PARTICLE_INPUT_BIT);
    const auto destination = context.buffer(2 * (sizeof(PmDomainOutputHeader) + particle_capacity * sizeof(PmFluidParticleOutput)), PM_BUFFER_SIMULATION_OUTPUT_BIT | PM_BUFFER_TRANSFER_DST_BIT);
    std::array<PmParticleState, 2> particles{};
    particles[0].position[0] = -10;
    particles[1].position[0] = 10;
    std::memcpy(source.mapped, particles.data(), sizeof(particles));
    context.flush(source);
    const PmFluidCapacity capacities[2]{{particle_capacity, 16}, {particle_capacity, 16}};
    PmSimulationCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    create.rigidBodyCapacity = rigid_capacity;
    create.contactCapacity = 16;
    create.fluidCount = 2;
    create.pFluidCapacities = capacities;
    PmSimulation simulation{};
    PM_CHECK(pmCreateSimulation(context.device, &create, &simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(context.device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        context.allocate(requirements.persistent), 0,
        context.allocate(requirements.scratch), 0};
    PM_CHECK(pmBindSimulationMemory(context.device, simulation, &binding) == PM_SUCCESS);
    PmLiquidMaterialState material{};
    material.restDensity = 1000; material.particleRadius = 0.025F;
    material.supportRadius = 0.1F; material.maximumSpeed = 100;
    material.densityIterations = 1;
    const PmLiquidMaterialState materials[2]{material, material};
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.fluidMaterialCount = 2;
    pipeline_info.pFluidMaterials = materials;
    const PmRigidSolverState sleeping{10, PM_TRUE, 0.02F, 0.02F, 0.5F};
    auto unsupported_policy = pipeline_info; unsupported_policy.pRigidState = &sleeping;
    PmPhysicsPipeline unsupported{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &unsupported_policy, &unsupported) == PM_ERROR_FEATURE_NOT_PRESENT);
    PM_CHECK(!unsupported);
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) == PM_SUCCESS);
    const PmDeviceSize output_size = sizeof(PmDomainOutputHeader) + particle_capacity * sizeof(PmFluidParticleOutput);
    const PmBufferView view{{source.handle, 0, sizeof(particles)},
        PM_FORMAT_PARTICLE_STATE, 2, sizeof(PmParticleState)};
    context.begin();
    PM_CHECK(pmCmdResetSimulation(context.commands, simulation) == PM_SUCCESS);
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 1, &view) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    // Keep a second population live across replacements, separate submissions
    // and rejected writes. Exact queried memory must suffice for declared capacity.
    for (unsigned batch = 0; batch < 4; ++batch) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
        for (unsigned edit = 0; edit < 256; ++edit)
            PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0, &view) == PM_SUCCESS);
        auto oversized = view; oversized.count = particle_capacity + 1;
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0, &oversized) == PM_ERROR_INVALID_ARGUMENT);
        for (std::uint32_t slot = 0; slot < 2; ++slot) {
            const PmBufferRange output{destination.handle, slot * output_size, output_size};
            PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, slot, &output) == PM_SUCCESS);
        }
        const auto result = context.submit();
        if (result != PM_SUCCESS) std::fprintf(stderr, "replacement batch %u: %d\n", batch, result);
        PM_CHECK(result == PM_SUCCESS);
        context.invalidate(destination);
        for (std::uint32_t slot = 0; slot < 2; ++slot) {
            const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(
                static_cast<const std::byte *>(destination.mapped) + slot * output_size);
            PM_CHECK(header->activeCount == 2 && header->capacity == particle_capacity);
            const auto *output = reinterpret_cast<const PmFluidParticleOutput *>(header + 1);
            PM_CHECK(output[0].position[0] == -10 && output[1].position[0] == 10);
        }
    }
    // A zero count ignores all buffer-view fields and leaves a valid empty slot.
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    const PmBufferView empty{};
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0, &empty) == PM_SUCCESS);
    const PmBufferRange first_output{destination.handle, 0, output_size};
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0, &first_output) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(destination);
    PM_CHECK(static_cast<const PmDomainOutputHeader *>(destination.mapped)->activeCount == 0);
    // Recording-time rejection must not enqueue a partial write.
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    auto short_range = view; short_range.range.size = sizeof(PmParticleState);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0, &short_range) == PM_ERROR_INVALID_ARGUMENT);
    PM_CHECK(context.submit() == PM_SUCCESS);
    // Device-side validation happens after retirement. A failed replacement may
    // empty its target, but must reclaim partial storage and preserve other slots.
    particles[0].position[0] = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(source.mapped, particles.data(), sizeof(particles)); context.flush(source);
    for (unsigned failure = 0; failure < 32; ++failure) {
        context.begin();
        PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
        PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0, &view) == PM_SUCCESS);
        PM_CHECK(context.submit() == PM_ERROR_INVALID_ARGUMENT);
    }
    context.begin();
    const PmBufferRange second_output{destination.handle, output_size, output_size};
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 1, &second_output) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(destination);
    const auto *unrelated = reinterpret_cast<const PmDomainOutputHeader *>(
        static_cast<const std::byte *>(destination.mapped) + output_size);
    PM_CHECK(unrelated->activeCount == 2);
    PM_CHECK(reinterpret_cast<const PmFluidParticleOutput *>(unrelated + 1)[0].position[0] == -10);
    particles[0].position[0] = -10;
    std::memcpy(source.mapped, particles.data(), sizeof(particles)); context.flush(source);
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdSetFluidParticles(context.commands, simulation, 0, &view) == PM_SUCCESS);
    PM_CHECK(pmCmdCopyFluidParticles(context.commands, simulation, 0, &first_output) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(destination);
    PM_CHECK(static_cast<const PmDomainOutputHeader *>(destination.mapped)->activeCount == 2);
    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    return 0;
}

#ifdef PM_CONFORMANCE_EMBEDDED
int parallel_mater_resource_lifecycle_fixture() {
#else
int main() {
#endif
    // Include the rigid storage strategy boundary and radix tile boundary.
    for (const auto [rigid, particles] : std::array<std::pair<std::uint32_t, std::uint32_t>, 4>{{
            {1, 2}, {2, 2}, {17, 63}, {65, 257}}}) {
        if (run(rigid, particles) == 77) return 77;
    }
    std::puts("PASS: exact queried storage, capacity sweep and 4096 fluid replacements");
    return 0;
}
