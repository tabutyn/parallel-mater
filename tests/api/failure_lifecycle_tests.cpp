// SPDX-License-Identifier: MIT
#include "test_context.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using parallel_mater::test::Buffer;
using parallel_mater::test::Context;

namespace {

void set_failure(const char *name, bool enabled) {
#if defined(_WIN32)
    PM_CHECK(_putenv_s(name, enabled ? "1" : "") == 0);
#else
    PM_CHECK(enabled ? setenv(name, "1", 1) == 0 : unsetenv(name) == 0);
#endif
}

bool is_guard_intact(const Buffer &guard) {
    const auto *bytes = static_cast<const std::uint8_t *>(guard.mapped);
    return std::all_of(bytes, bytes + guard.size,
                       [](std::uint8_t value) { return value == 0x3cU; });
}

PmSubmitInfo submit_info(const PmCommandBuffer *command) {
    PmSubmitInfo info{};
    info.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    info.commandBufferCount = 1U;
    info.pCommandBuffers = command;
    return info;
}

} // namespace

int main() {
    Context context;
    PmPhysicalDeviceFeatures needed{};
    needed.rigidBodies = PM_TRUE;
    needed.debugData = PM_TRUE;
    if (!context.open(needed)) return 77;

    const auto usage = PM_BUFFER_TRANSFER_SRC_BIT |
        PM_BUFFER_TRANSFER_DST_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT;
    const Buffer guard = context.buffer(256U, usage);
    const Buffer copied = context.buffer(256U, usage);
    const Buffer output = context.buffer(256U, usage);
    std::memset(guard.mapped, 0x3c, static_cast<std::size_t>(guard.size));
    std::memset(copied.mapped, 0, static_cast<std::size_t>(copied.size));
    context.flush(guard);
    context.flush(copied);

    // Device allocation failure publishes no handle, preserves existing
    // resources, and permits an immediate successful retry.
    PmMemoryAllocateInfo allocation_info{};
    allocation_info.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = 256U;
    allocation_info.memoryTypeIndex = 1U;
    PmDeviceMemory recovered{};
    set_failure("PM_TEST_FAIL_ALLOCATION", true);
    PM_CHECK(pmAllocateMemory(context.device, &allocation_info, &recovered) ==
             PM_ERROR_OUT_OF_DEVICE_MEMORY);
    PM_CHECK(recovered == nullptr);
    set_failure("PM_TEST_FAIL_ALLOCATION", false);
    PM_CHECK(pmAllocateMemory(context.device, &allocation_info, &recovered) ==
             PM_SUCCESS);
    context.memory.push_back(recovered);
    PM_CHECK(is_guard_intact(guard));

    // Recording allocation failure is synchronous and latched until reset.
    const PmBufferCopy region{0U, 0U, guard.size};
    context.begin();
    set_failure("PM_TEST_FAIL_RECORDING", true);
    PM_CHECK(pmCmdCopyBuffer(context.commands, guard.handle, copied.handle, 1U,
                             &region) == PM_ERROR_OUT_OF_HOST_MEMORY);
    set_failure("PM_TEST_FAIL_RECORDING", false);
    PM_CHECK(pmEndCommandBuffer(context.commands) ==
             PM_ERROR_OUT_OF_HOST_MEMORY);
    context.begin();
    PM_CHECK(pmCmdCopyBuffer(context.commands, guard.handle, copied.handle, 1U,
                             &region) == PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(copied);
    PM_CHECK(std::memcmp(guard.mapped, copied.mapped,
                         static_cast<std::size_t>(guard.size)) == 0);
    PM_CHECK(is_guard_intact(guard));

    // Submission allocation failure occurs before acceptance: neither the
    // executable command nor the unsignaled fence becomes pending.
    context.begin();
    PM_CHECK(pmCmdCopyBuffer(context.commands, guard.handle, copied.handle, 1U,
                             &region) == PM_SUCCESS);
    PM_CHECK(pmEndCommandBuffer(context.commands) == PM_SUCCESS);
    auto submit = submit_info(&context.commands);
    set_failure("PM_TEST_FAIL_SUBMISSION", true);
    PM_CHECK(pmQueueSubmit(context.queue, &submit, context.fence) ==
             PM_ERROR_OUT_OF_HOST_MEMORY);
    set_failure("PM_TEST_FAIL_SUBMISSION", false);
    PM_CHECK(pmGetFenceStatus(context.device, context.fence) == PM_NOT_READY);
    PM_CHECK(pmQueueSubmit(context.queue, &submit, context.fence) == PM_SUCCESS);
    PM_CHECK(pmWaitForFences(context.device, 1U, &context.fence, PM_TRUE,
                             PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    PM_CHECK(is_guard_intact(guard));

    // Debug output from an empty, bound simulation still uploads its header.
    // Its injected device failure is deferred until fence completion.
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    simulation_info.contactCapacity = 1U;
    simulation_info.debugFrameCapacity = 1U;
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
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    PmPhysicsPipeline pipeline{};
    PM_CHECK(pmCreatePhysicsPipeline(context.device, &pipeline_info, &pipeline) ==
             PM_SUCCESS);
    PmStepInfo step{1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F}, 0U};
    const PmBufferRange destination{output.handle, 0U, output.size};
    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdCopyDebugData(context.commands, simulation, &destination) ==
             PM_SUCCESS);
    PM_CHECK(pmEndCommandBuffer(context.commands) == PM_SUCCESS);
    submit = submit_info(&context.commands);
    set_failure("PM_TEST_FAIL_UPLOAD", true);
    PM_CHECK(pmQueueSubmit(context.queue, &submit, context.fence) == PM_SUCCESS);
    const PmResult deferred = pmWaitForFences(
        context.device, 1U, &context.fence, PM_TRUE, PM_TIMEOUT_INFINITE);
    if (deferred != PM_ERROR_OUT_OF_DEVICE_MEMORY)
        std::fprintf(stderr, "deferred upload result: %d\n", deferred);
    PM_CHECK(deferred == PM_ERROR_OUT_OF_DEVICE_MEMORY);
    set_failure("PM_TEST_FAIL_UPLOAD", false);
    PM_CHECK(is_guard_intact(guard));

    context.begin();
    PM_CHECK(pmCmdBindPhysicsPipeline(context.commands, pipeline) == PM_SUCCESS);
    PM_CHECK(pmCmdStepSimulation(context.commands, simulation, &step) ==
             PM_SUCCESS);
    PM_CHECK(pmCmdCopyDebugData(context.commands, simulation, &destination) ==
             PM_SUCCESS);
    PM_CHECK(context.submit() == PM_SUCCESS);
    context.invalidate(output);
    const auto *header = static_cast<const PmDomainOutputHeader *>(output.mapped);
    PM_CHECK(header->activeCount == 0U);
    PM_CHECK(is_guard_intact(guard));

    pmDestroySimulation(context.device, simulation);
    pmDestroyPhysicsPipeline(context.device, pipeline);
    std::puts("PASS: allocation, recording, submission and deferred failure recovery");
}
