// SPDX-License-Identifier: MIT
// Analytic rope translation and smoke transport through the public command API.
#include <parallel_mater/parallel_mater.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); \
} } while (false)

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_3;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t count = 1;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS && count == 1);
    const float priority = 1;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures.rope = PM_TRUE;
    device_info.enabledFeatures.smoke = PM_TRUE;
    PmDevice device{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    PmQueue queue{};
    pmGetDeviceQueue(device, 0, 0, &queue);
    std::vector<PmDeviceMemory> allocations;
    const auto allocate = [&](PmDeviceSize size) {
        PmMemoryAllocateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.allocationSize = size;
        info.memoryTypeIndex = 1;
        PmDeviceMemory memory{};
        CHECK(pmAllocateMemory(device, &info, &memory) == PM_SUCCESS);
        allocations.push_back(memory);
        return memory;
    };
    PmBufferCreateInfo buffer_info{};
    buffer_info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = 16384;
    buffer_info.usage = PM_BUFFER_TRANSFER_DST_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT;
    PmBuffer output{};
    CHECK(pmCreateBuffer(device, &buffer_info, &output) == PM_SUCCESS);
    const auto output_memory = allocate(buffer_info.size);
    CHECK(pmBindBufferMemory(device, output, output_memory, 0) == PM_SUCCESS);
    void *mapped{};
    CHECK(pmMapMemory(device, output_memory, 0, buffer_info.size, &mapped) == PM_SUCCESS);

    const PmRopeCapacity rope_capacity{32};
    const PmSmokeCapacity smoke_capacity{16, 16 * 16 * 8};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.ropeCount = simulation_info.smokeCount = 1;
    simulation_info.pRopeCapacities = &rope_capacity;
    simulation_info.pSmokeCapacities = &smoke_capacity;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(device, simulation, &requirements);
    const PmSimulationMemoryBinding binding{
        allocate(requirements.persistent.size), 0, allocate(requirements.scratch.size), 0};
    CHECK(pmBindSimulationMemory(device, simulation, &binding) == PM_SUCCESS);
    PmRopeMaterialState rope_material{};
    rope_material.nodeSpacing = 0.1F;
    rope_material.radius = 0.05F;
    rope_material.mass = 1;
    rope_material.maximumSubstepTimestep = 1.0F / 60.0F;
    rope_material.maximumSpeed = 2;
    rope_material.solverIterations = 8;
    PmSmokeMaterialState smoke_material{};
    smoke_material.particleRadius = 0.02F;
    smoke_material.restNumberDensity = 12;
    smoke_material.maximumSpeed = 1;
    smoke_material.gridVerticalResolution = 8;
    smoke_material.gridPressureIterations = 12;
    smoke_material.gridPressureTolerance = 0.001F;
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.ropeMaterialCount = pipeline_info.smokeMaterialCount = 1;
    pipeline_info.pRopeMaterials = &rope_material;
    pipeline_info.pSmokeMaterials = &smoke_material;
    const PmFloat3 line[2]{{0, 1, 0}, {0.8F, 1, 0}};
    PmRopeInfo rope{};
    rope.active = PM_TRUE;
    rope.pointCount = 2;
    rope.pCenterline = line;
    PmSmokeInfo smoke{};
    smoke.active = PM_TRUE;
    smoke.particleCapacity = 16;
    smoke.emitterCenter = {2, 2, 0};
    smoke.emitterHalfExtents = {0.1F, 0.1F};
    smoke.initialVelocity = smoke.wind = {1, 0, 0};
    smoke.particlesPerSecond = 60;
    smoke.lifetime = 2;
    smoke.gridMinimum = {0, 0, -4};
    smoke.gridEdgeLength = 8;
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PmCommandPool pool{};
    CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    PmCommandBuffer commands{};
    CHECK(pmAllocateCommandBuffers(device, pool, 1, &commands) == PM_SUCCESS);
    PmFence fence{};
    CHECK(pmCreateFence(device, 0, &fence) == PM_SUCCESS);
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commands;
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    for (bool grid : {false, true}) {
        smoke_material.gridResolution = grid ? 16 : 0;
        PmPhysicsPipeline pipeline{};
        CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) == PM_SUCCESS);
        CHECK(pmResetFences(device, 1, &fence) == PM_SUCCESS);
        CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
        CHECK(pmCmdWriteRopes(commands, simulation, 0, 1, &rope) == PM_SUCCESS);
        CHECK(pmCmdWriteSmoke(commands, simulation, 0, 1, &smoke) == PM_SUCCESS);
        const PmStepInfo step{1.0F / 60.0F, 1, {0, -9.81F, 0}, 0};
        for (int frame = 0; frame < 10; ++frame)
            CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
        const PmBufferRange rope_output{output, 0, 4096}, smoke_output{output, 4096, 4096};
        CHECK(pmCmdCopyRopeState(commands, simulation, 0, &rope_output) == PM_SUCCESS);
        CHECK(pmCmdCopySmokeState(commands, simulation, 0, &smoke_output) == PM_SUCCESS);
        CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
        CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
        const auto completed = pmWaitForFences(device, 1, &fence, PM_TRUE, PM_TIMEOUT_INFINITE);
        if (completed != PM_SUCCESS) std::fprintf(stderr, "transport grid=%d: %d\n", grid, int(completed));
        CHECK(completed == PM_SUCCESS);
        const auto *rope_header = static_cast<const PmDomainOutputHeader *>(mapped);
        const auto *smoke_header = reinterpret_cast<const PmDomainOutputHeader *>(static_cast<const std::byte *>(mapped) + 4096);
        CHECK(rope_header->activeCount == 9 && smoke_header->activeCount == 10);
        float expected_rope_y = 1.0F;
        float expected_rope_velocity = 0.0F;
        for (int frame = 0; frame < 10; ++frame) {
            expected_rope_velocity -= 9.81F / 60.0F;
            expected_rope_y += expected_rope_velocity / 60.0F;
            expected_rope_velocity *= 0.95F;
        }
        const auto *nodes = reinterpret_cast<const PmRopeNodeState *>(rope_header + 1);
        for (std::uint32_t i = 0; i < rope_header->activeCount; ++i) {
            CHECK(std::abs(nodes[i].position.x - 0.1F * i) < 1e-4F);
            CHECK(std::abs(nodes[i].position.y - expected_rope_y) < 1e-4F);
            CHECK(std::abs(nodes[i].velocity.y - expected_rope_velocity) < 1e-4F);
            CHECK(std::abs(nodes[i].position.z) < 1e-4F);
        }
        const auto *particles = reinterpret_cast<const PmSmokeParticleState *>(smoke_header + 1);
        float oldest = 0;
        for (std::uint32_t i = 0; i < smoke_header->activeCount; ++i) {
            const auto &p = particles[i];
            // Emission can occur after advection: the newest tracer has age zero.
            CHECK(p.age >= 0 && p.age <= 10.01F / 60.0F);
            oldest = std::max(oldest, p.age);
            // With no correction, the shared solver preserves the predicted
            // velocity for both particle and grid smoke transport.
            CHECK(std::abs(p.position.x - smoke.emitterCenter.x -
                           p.age) < 2e-4F);
            CHECK(std::abs(p.velocity.x - 1.0F) < 2e-4F);
            CHECK(std::abs(p.velocity.y) < 2e-4F && std::abs(p.velocity.z) < 2e-4F);
            CHECK(std::abs(p.position.y - 2) <= 0.101F && std::abs(p.position.z) <= 0.101F);
            CHECK(std::isfinite(p.numberDensity) && std::isfinite(p.pressure));
            CHECK(std::isfinite(p.vorticity.x) && std::isfinite(p.vorticity.y) && std::isfinite(p.vorticity.z));
        }
        CHECK(oldest > 0.13F);
        pmDestroyPhysicsPipeline(device, pipeline);
    }
    CHECK(pmDeviceWaitIdle(device) == PM_SUCCESS);
    pmDestroyFence(device, fence);
    pmFreeCommandBuffers(device, pool, 1, &commands);
    pmDestroyCommandPool(device, pool);
    pmDestroySimulation(device, simulation);
    pmDestroyBuffer(device, output);
    for (auto memory : allocations) pmFreeMemory(device, memory);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: rope free fall, smoke particle/grid transport, source counts and replacement");
}
