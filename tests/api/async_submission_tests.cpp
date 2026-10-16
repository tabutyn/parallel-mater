// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); std::abort(); \
} } while (false)

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t count = 1U;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS);
    CHECK(count == 1U);
#if defined(PM_TEST_METAL_PREVIEW)
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(physical, &properties);
    CHECK(std::strstr(properties.deviceName, "Metal GPU preview") != nullptr);
    PmPhysicalDeviceFeatures supported{};
    pmGetPhysicalDeviceFeatures(physical, &supported);
    CHECK(supported.rigidBodies && supported.rigidConstraints);
    CHECK(supported.particleLiquids && supported.liquidRigidCoupling);
    CHECK(supported.cloth && supported.softBodies && supported.domainCouplings);
    CHECK(supported.rope && supported.smoke && supported.paint);
    CHECK(!supported.liquidLiquidCoupling);
#endif
    float priority = 1.0F;
    PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures.rigidBodies = PM_TRUE;
    device_info.enabledFeatures.timelineSemaphores = PM_TRUE;
    PmDevice device{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
#if defined(PM_TEST_METAL_PREVIEW)
    PmNativeDeviceInfo native{};
    CHECK(pmGetNativeDevice(device, &native) == PM_SUCCESS);
    CHECK(native.backend == PM_NATIVE_BACKEND_METAL && native.device != 0);
#endif
    PmQueue queue{};
    pmGetDeviceQueue(device, 0U, 0U, &queue);
    CHECK(queue != nullptr);
    std::vector<PmDeviceMemory> memory;
    const auto allocate = [&](PmDeviceSize size) {
        PmMemoryAllocateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.memoryTypeIndex = 1U;
        info.allocationSize = size;
        PmDeviceMemory result{};
        CHECK(pmAllocateMemory(device, &info, &result) == PM_SUCCESS);
        memory.push_back(result);
        return result;
    };
    const auto make_buffer = [&](PmDeviceSize size, void **mapped) {
        auto allocation = allocate(size);
        PmBufferCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size;
        info.usage = PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT |
            PM_BUFFER_GEOMETRY_INPUT_BIT | PM_BUFFER_SIMULATION_OUTPUT_BIT;
        PmBuffer buffer{};
        CHECK(pmCreateBuffer(device, &info, &buffer) == PM_SUCCESS);
        CHECK(pmBindBufferMemory(device, buffer, allocation, 0U) == PM_SUCCESS);
        CHECK(pmMapMemory(device, allocation, 0U, size, mapped) == PM_SUCCESS);
        return buffer;
    };
    void *input{}, *output{};
    auto source = make_buffer(512U, &input);
    auto destination = make_buffer(512U, &output);
    const float vertices[8][3] = {
        {-0.5F,-0.5F,-0.5F},{0.5F,-0.5F,-0.5F},{0.5F,0.5F,-0.5F},{-0.5F,0.5F,-0.5F},
        {-0.5F,-0.5F,0.5F},{0.5F,-0.5F,0.5F},{0.5F,0.5F,0.5F},{-0.5F,0.5F,0.5F}};
    const std::uint32_t indices[36] = {
        0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
    std::memcpy(input, vertices, sizeof vertices);
    std::memcpy(static_cast<std::byte *>(input) + sizeof vertices, indices, sizeof indices);
    PmGeometryCreateInfo geometry_info{};
    geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    geometry_info.vertices = {{source, 0U, sizeof vertices}, PM_FORMAT_R32G32B32_SFLOAT, 8U, 12U};
    geometry_info.indices = {{source, sizeof vertices, sizeof indices}, PM_FORMAT_R32_UINT, 36U, 4U};
    PmGeometry invalid_geometry{};
    auto invalid_geometry_info = geometry_info;
    invalid_geometry_info.vertices.count = 2U;
    CHECK(pmCreateGeometry(device, &invalid_geometry_info, &invalid_geometry) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(invalid_geometry == nullptr);
    PmGeometry geometry{};
    CHECK(pmCreateGeometry(device, &geometry_info, &geometry) == PM_SUCCESS);
    PmMemoryRequirements geometry_memory{};
    pmGetGeometryMemoryRequirements(device, geometry, &geometry_memory);
    CHECK(geometry_memory.size > 1U);
    CHECK(geometry_memory.alignment == 256U);
    CHECK(geometry_memory.size % geometry_memory.alignment == 0U);
    CHECK(pmBindGeometryMemory(
              device, geometry, allocate(geometry_memory.size - 1U), 0U) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(pmBindGeometryMemory(device, geometry, allocate(geometry_memory.size), 0U) == PM_SUCCESS);
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 2U;
    simulation_info.contactCapacity = 32U;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(device, simulation, &requirements);
    PmSimulationMemoryBinding binding{allocate(requirements.persistent.size), 0U,
                                      allocate(requirements.scratch.size), 0U};
    CHECK(pmBindSimulationMemory(device, simulation, &binding) == PM_SUCCESS);
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    PmPhysicsPipeline pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) == PM_SUCCESS);
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PmCommandPool pool{};
    CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    std::array<PmCommandBuffer, 5> commands{};
    CHECK(pmAllocateCommandBuffers(device, pool, commands.size(), commands.data()) == PM_SUCCESS);
    // Same driver is insufficient: dispatchable children belong to one device.
    PmDevice other_device{};
    CHECK(pmCreateDevice(physical, &device_info, &other_device) == PM_SUCCESS);
    PmQueue other_queue{};
    pmGetDeviceQueue(other_device, 0, 0, &other_queue);
    PmSubmitInfo foreign_submit{};
    foreign_submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    foreign_submit.commandBufferCount = 1;
    foreign_submit.pCommandBuffers = commands.data();
    CHECK(pmQueueSubmit(other_queue, &foreign_submit, nullptr) == PM_ERROR_INVALID_ARGUMENT);
    // A wrong-device free must leave the wrapper intact for its actual owner.
    pmFreeCommandBuffers(other_device, pool, 1, commands.data());
    pmDestroyDevice(other_device);
    std::array<PmFence, 5> fences{};
    for (auto &fence : fences) CHECK(pmCreateFence(device, 0U, &fence) == PM_SUCCESS);
    PmQueryPoolCreateInfo query_info{};
    query_info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = PM_QUERY_STEP_STATISTICS;
    query_info.queryCount = 2U;
    PmQueryPool query{};
    CHECK(pmCreateQueryPool(device, &query_info, &query) == PM_SUCCESS);
    PmSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = PM_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    PmSemaphore timeline{};
    CHECK(pmCreateSemaphore(device, &semaphore_info, &timeline) == PM_SUCCESS);
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CHECK(pmBeginCommandBuffer(commands[0], &begin) == PM_SUCCESS);
    CHECK(pmCmdBuildGeometry(commands[0], geometry) == PM_SUCCESS);
    PmRigidBodyInfo body{};
    body.active = PM_TRUE;
    body.geometry = geometry;
    body.motionType = PM_MOTION_DYNAMIC;
    body.mass = 1.0F;
    body.collisionMargin = 0.005F;
    body.state.orientation[3] = 1.0F;
    body.state.position[1] = 2.0F;
    body.inertiaDiagonal[0] = body.inertiaDiagonal[1] = body.inertiaDiagonal[2] = 1.0F / 6.0F;
    body.maximumLinearSpeed = body.maximumAngularSpeed = 100.0F;
    CHECK(pmCmdWriteRigidBodies(commands[0], simulation, 0U, 1U, &body) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands[0], pipeline) == PM_SUCCESS);
    CHECK(pmCmdBeginStatisticsQuery(commands[0], query, 0U) == PM_SUCCESS);
    PmStepInfo step{1.0F / 60.0F, 2U, {0.0F, -9.81F, 0.0F}, 0U};
    CHECK(pmCmdStepSimulation(commands[0], simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdEndStatisticsQuery(commands[0], query, 0U) == PM_SUCCESS);
    PmBufferRange range{destination, 0U, sizeof(PmRigidState)};
    CHECK(pmCmdCopyRigidStates(commands[0], simulation, 0U, 1U, &range) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands[0]) == PM_SUCCESS);
    PmSemaphoreSubmitInfo wait{timeline, 1U, PM_STAGE_ALL_COMMANDS_BIT};
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitCount = 1U;
    submit.pWaits = &wait;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &commands[0];
    {
        // Waiting for 2 while publishing 1 on the same timeline is also a
        // cycle: producer 2 cannot finish before the pending signal of 1.
        PmSemaphoreSubmitInfo future_wait{timeline, 2U, PM_STAGE_ALL_COMMANDS_BIT};
        PmSemaphoreSubmitInfo lower_signal{timeline, 1U, PM_STAGE_ALL_COMMANDS_BIT};
        PmSubmitInfo cyclic = submit;
        cyclic.pWaits = &future_wait;
        cyclic.signalCount = 1U;
        cyclic.pSignals = &lower_signal;
        CHECK(pmQueueSubmit(queue, &cyclic, fences[0]) == PM_ERROR_INVALID_ARGUMENT);
    }
    CHECK(pmQueueSubmit(queue, &submit, fences[0]) == PM_SUCCESS);
    CHECK(pmGetFenceStatus(device, fences[0]) == PM_NOT_READY);
    CHECK(pmResetCommandBuffer(commands[0]) == PM_ERROR_INVALID_STATE);
    CHECK(pmResetFences(device, 1U, &fences[0]) == PM_ERROR_INVALID_STATE);
    CHECK(pmWaitForFences(device, 1U, &fences[0], PM_TRUE, 0U) == PM_TIMEOUT);
    PmResult execution{};
    CHECK(pmGetSimulationStatus(device, simulation, &execution) == PM_NOT_READY);
    PmStepStatistics statistics{};
    CHECK(pmGetQueryPoolResults(device, query, 0U, 1U, sizeof statistics, &statistics,
                               sizeof statistics, 0U) == PM_NOT_READY);
    // Producer submitted after consumer: no caller-thread or worker deadlock.
    PmSemaphoreSubmitInfo signal{timeline, 1U, PM_STAGE_ALL_COMMANDS_BIT};
    PmSubmitInfo producer{};
    producer.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    producer.signalCount = 1U;
    producer.pSignals = &signal;
    CHECK(pmQueueSubmit(queue, &producer, nullptr) == PM_SUCCESS);
    const auto first_result = pmWaitForFences(device, 1U, &fences[0], PM_TRUE, PM_TIMEOUT_INFINITE);
    if (first_result != PM_SUCCESS) std::fprintf(stderr, "first submission result: %d\n", first_result);
    CHECK(first_result == PM_SUCCESS);
    CHECK(pmGetQueryPoolResults(device, query, 0U, 1U, sizeof statistics, &statistics,
                               sizeof statistics, PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
    CHECK(statistics.stepCount == 1U);
    CHECK(statistics.executionResult == PM_SUCCESS);
    CHECK(pmGetSimulationStatus(device, simulation, &execution) == PM_SUCCESS);
    CHECK(execution == PM_SUCCESS);
    // Several frames in flight retain distinct output slots and execute in order.
    for (std::uint32_t frame = 1U; frame < 3U; ++frame) {
        CHECK(pmBeginCommandBuffer(commands[frame], &begin) == PM_SUCCESS);
        CHECK(pmCmdBindPhysicsPipeline(commands[frame], pipeline) == PM_SUCCESS);
        CHECK(pmCmdStepSimulation(commands[frame], simulation, &step) == PM_SUCCESS);
        range.offset = frame * sizeof(PmRigidState);
        CHECK(pmCmdCopyRigidStates(commands[frame], simulation, 0U, 1U, &range) == PM_SUCCESS);
        CHECK(pmEndCommandBuffer(commands[frame]) == PM_SUCCESS);
        submit.waitCount = 0U;
        submit.pCommandBuffers = &commands[frame];
        CHECK(pmQueueSubmit(queue, &submit, fences[frame]) == PM_SUCCESS);
    }
    CHECK(pmWaitForFences(device, 3U, fences.data(), PM_TRUE, PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    const auto *states = static_cast<const PmRigidState *>(output);
    CHECK(states[0].position[1] < 2.0F);
    CHECK(states[1].position[1] < states[0].position[1]);
    CHECK(states[2].position[1] < states[1].position[1]);
    CHECK(std::abs(states[2].linearVelocity[1] - (-9.81F * 3.0F / 60.0F)) < 0.001F);
    // Runtime validation errors are deferred, stable, and do not strand handles.
    CHECK(pmBeginCommandBuffer(commands[3], &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands[3], pipeline) == PM_SUCCESS);
    CHECK(pmCmdBeginStatisticsQuery(commands[3], query, 0U) == PM_SUCCESS);
    step.substeps = 1025U;
    CHECK(pmCmdStepSimulation(commands[3], simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdEndStatisticsQuery(commands[3], query, 0U) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands[3]) == PM_SUCCESS);
    submit.pCommandBuffers = &commands[3];
    wait.value = 2U;
    submit.waitCount = 1U;
    CHECK(pmQueueSubmit(queue, &submit, fences[3]) == PM_SUCCESS);
    // Both query writes are accepted before either executes. Failure in slot
    // zero must not poison the successful frame recorded into slot one.
    CHECK(pmBeginCommandBuffer(commands[4], &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands[4], pipeline) == PM_SUCCESS);
    CHECK(pmCmdBeginStatisticsQuery(commands[4], query, 1U) == PM_SUCCESS);
    step.substeps = 2U;
    CHECK(pmCmdStepSimulation(commands[4], simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdEndStatisticsQuery(commands[4], query, 1U) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands[4]) == PM_SUCCESS);
    submit.pCommandBuffers = &commands[4];
    CHECK(pmQueueSubmit(queue, &submit, fences[4]) == PM_SUCCESS);
    signal.value = 2U;
    CHECK(pmQueueSubmit(queue, &producer, nullptr) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fences[3], PM_TRUE, PM_TIMEOUT_INFINITE) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(pmWaitForFences(device, 1U, &fences[4], PM_TRUE, PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    CHECK(pmGetSimulationStatus(device, simulation, &execution) == PM_SUCCESS);
    CHECK(execution == PM_SUCCESS);
    CHECK(pmGetQueryPoolResults(device, query, 1U, 1U, sizeof statistics, &statistics,
                               sizeof statistics, 0U) == PM_SUCCESS);
    CHECK(statistics.executionResult == PM_SUCCESS);
    CHECK(statistics.stepCount == 4U);
    CHECK(pmGetQueryPoolResults(device, query, 0U, 1U, sizeof statistics, &statistics,
                               sizeof statistics, 0U) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(pmQueueWaitIdle(queue) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(pmQueueWaitIdle(queue) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands[3]) == PM_SUCCESS);
    CHECK(pmResetFences(device, 1U, &fences[3]) == PM_SUCCESS);
    pmDestroyQueryPool(device, query);
    for (auto fence : fences) pmDestroyFence(device, fence);
    pmDestroySemaphore(device, timeline);
    pmDestroyCommandPool(device, pool);
    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroySimulation(device, simulation);
    pmDestroyGeometry(device, geometry);
    pmDestroyBuffer(device, destination);
    pmDestroyBuffer(device, source);
    for (auto allocation : memory) pmFreeMemory(device, allocation);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    return 0;
}
