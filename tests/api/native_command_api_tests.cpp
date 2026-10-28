// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#ifndef PM_TEST_NATIVE_BACKEND
#define PM_TEST_NATIVE_BACKEND PM_NATIVE_BACKEND_D3D12
#endif

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_3;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t physical_count{};
    CHECK(pmEnumeratePhysicalDevices(instance, &physical_count, nullptr) ==
          PM_SUCCESS);
    if (physical_count == 0U) {
        pmDestroyInstance(instance);
        std::puts("SKIP: no selected native GPU device");
        return 77;
    }
    std::vector<PmPhysicalDevice> physicals(physical_count);
    CHECK(pmEnumeratePhysicalDevices(
        instance, &physical_count, physicals.data()) == PM_SUCCESS);
    const PmPhysicalDevice physical = physicals.front();
    PmPhysicalDeviceFeatures features{};
    pmGetPhysicalDeviceFeatures(physical, &features);
    CHECK(features.rigidBodies == PM_TRUE);
    CHECK(features.rigidConstraints == PM_TRUE);
    CHECK(features.debugData == PM_TRUE);
    CHECK(features.memoryBudget == PM_TRUE);
    std::uint32_t family_count{};
    CHECK(pmGetPhysicalDeviceQueueFamilies(
        physical, &family_count, nullptr) == PM_SUCCESS);
    CHECK(family_count == 1U);
    PmQueueFamilyProperties family{};
    CHECK(pmGetPhysicalDeviceQueueFamilies(
        physical, &family_count, &family) == PM_SUCCESS);
    CHECK((family.queueFlags & PM_QUEUE_TRANSFER_BIT) != 0U);
    CHECK((family.queueFlags & PM_QUEUE_PHYSICS_BIT) != 0U);

    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures.rigidBodies = PM_TRUE;
    device_info.enabledFeatures.debugData = PM_TRUE;
    device_info.enabledFeatures.memoryBudget = PM_TRUE;
    CHECK(features.timestampQueries ==
        (PM_TEST_NATIVE_BACKEND == PM_NATIVE_BACKEND_METAL ? PM_FALSE : PM_TRUE));
    device_info.enabledFeatures.timestampQueries = features.timestampQueries;
    PmDevice device{};
    // The same public command fixture runs against each selected native driver.
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    PmQueue queue{};
    pmGetDeviceQueue(device, 0U, 0U, &queue);
    CHECK(queue != nullptr);

    constexpr PmDeviceSize byte_count = 256U;
    PmMemoryAllocateInfo allocation{};
    allocation.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = byte_count;
    allocation.memoryTypeIndex = 1U;
    PmDeviceMemory upload{};
#if defined(PM_TEST_METAL_PREVIEW)
    // An invalid allocation is not a device-lost error; subsequent work works.
    allocation.allocationSize = UINT64_MAX;
    CHECK(pmAllocateMemory(device, &allocation, &upload) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(upload == nullptr);
    allocation.allocationSize = byte_count;
#endif
    CHECK(pmAllocateMemory(device, &allocation, &upload) == PM_SUCCESS);
    allocation.memoryTypeIndex = 1U;
    PmDeviceMemory readback{};
    CHECK(pmAllocateMemory(device, &allocation, &readback) == PM_SUCCESS);

    PmBufferCreateInfo buffer_info{};
    buffer_info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = byte_count;
    buffer_info.usage = PM_BUFFER_TRANSFER_SRC_BIT |
        PM_BUFFER_GEOMETRY_INPUT_BIT;
    PmBuffer source{};
    CHECK(pmCreateBuffer(device, &buffer_info, &source) == PM_SUCCESS);
    buffer_info.usage = PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT |
        PM_BUFFER_SIMULATION_OUTPUT_BIT;
    PmBuffer destination{};
    CHECK(pmCreateBuffer(device, &buffer_info, &destination) == PM_SUCCESS);
    PmMemoryRequirements requirements{};
    pmGetBufferMemoryRequirements(device, source, &requirements);
    CHECK((requirements.memoryTypeBits & (1U << 1U)) != 0U);
    pmGetBufferMemoryRequirements(device, destination, &requirements);
    CHECK((requirements.memoryTypeBits & (1U << 1U)) != 0U);
    CHECK(pmBindBufferMemory(device, source, upload, 0U) == PM_SUCCESS);
    CHECK(pmBindBufferMemory(device, destination, readback, 0U) == PM_SUCCESS);

    void *mapped_upload{};
    void *mapped_readback{};
    CHECK(pmMapMemory(device, upload, 0U, PM_WHOLE_SIZE, &mapped_upload) ==
          PM_SUCCESS);
    CHECK(pmMapMemory(device, readback, 0U, PM_WHOLE_SIZE, &mapped_readback) ==
          PM_SUCCESS);
    for (std::uint32_t index = 0U; index < byte_count / sizeof(std::uint32_t);
         ++index)
        static_cast<std::uint32_t *>(mapped_upload)[index] =
            0xabc00000U + index;
    std::memset(mapped_readback, 0, byte_count);

    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PmCommandPool pool{};
    CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    PmCommandBuffer commands{};
    CHECK(pmAllocateCommandBuffers(device, pool, 1U, &commands) == PM_SUCCESS);
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    PmBufferMemoryBarrier visibility{};
    visibility.range = {destination, 0U, byte_count};
    visibility.srcQueueFamilyIndex = PM_QUEUE_FAMILY_IGNORED;
    visibility.dstQueueFamilyIndex = PM_QUEUE_FAMILY_IGNORED;
    CHECK(pmCmdPipelineBarrier(commands, 0U, nullptr, 1U, &visibility) ==
          PM_SUCCESS);
    const PmBufferCopy copy{0U, 0U, byte_count};
    CHECK(pmCmdCopyBuffer(commands, source, destination, 1U, &copy) ==
          PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    PmFence fence{};
    CHECK(pmCreateFence(device, 0U, &fence) == PM_SUCCESS);
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &commands;
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    for (std::uint32_t index = 0U; index < byte_count / sizeof(std::uint32_t);
         ++index)
        CHECK(static_cast<const std::uint32_t *>(mapped_readback)[index] ==
              0xabc00000U + index);

    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdCopyBuffer(commands, source, destination, 1U, &copy) ==
          PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, nullptr) == PM_SUCCESS);
    CHECK(pmQueueWaitIdle(queue) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);

    // Same-allocation GPU copies need a legal scratch path on D3D12. A later
    // narrow host write must not overwrite neighboring GPU-owned bytes.
    const PmBufferCopy self_copy{0U, 128U, 64U};
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdCopyBuffer(commands, destination, destination, 1U, &self_copy) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, nullptr) == PM_SUCCESS);
    CHECK(pmQueueWaitIdle(queue) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    for (std::uint32_t index = 0U; index < 16U; ++index)
        CHECK(static_cast<const std::uint32_t *>(mapped_readback)[32U + index] == 0xabc00000U + index);
    static_cast<std::uint32_t *>(mapped_upload)[1U] = 0x12345678U;
    const PmMappedMemoryRange narrow_upload{upload, 4U, 4U};
    CHECK(pmFlushMappedMemoryRanges(device, 1U, &narrow_upload) == PM_SUCCESS);
    pmUnmapMemory(device, readback);
    const PmBufferCopy narrow_copy{4U, 252U, 4U};
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdCopyBuffer(commands, source, destination, 1U, &narrow_copy) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    for (unsigned repeat = 0U; repeat < 8U; ++repeat) {
        CHECK(pmQueueSubmit(queue, &submit, nullptr) == PM_SUCCESS);
        CHECK(pmQueueWaitIdle(queue) == PM_SUCCESS);
    }
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmMapMemory(device, readback, 0U, PM_WHOLE_SIZE, &mapped_readback) == PM_SUCCESS);
    const PmMappedMemoryRange narrow_readback{readback, 252U, 4U};
    CHECK(pmInvalidateMappedMemoryRanges(device, 1U, &narrow_readback) == PM_SUCCESS);
    CHECK(static_cast<const std::uint32_t *>(mapped_readback)[63U] == 0x12345678U);
    for (std::uint32_t index = 0U; index < 16U; ++index)
        CHECK(static_cast<const std::uint32_t *>(mapped_readback)[32U + index] == 0xabc00000U + index);

    PmNativeDeviceInfo native_device{};
    PmNativeQueueInfo native_queue{};
    PmNativeBufferInfo native_buffer{};
    CHECK(pmGetNativeDevice(device, &native_device) == PM_SUCCESS);
    CHECK(native_device.backend == PM_TEST_NATIVE_BACKEND);
    CHECK(native_device.device != 0U);
    CHECK(pmGetNativeQueue(queue, &native_queue) == PM_SUCCESS);
    CHECK(native_queue.queue != 0U && native_queue.queueIndex == 0U);
    CHECK(pmGetNativeBuffer(device, destination, &native_buffer) == PM_SUCCESS);
    CHECK(native_buffer.buffer != 0U);
    CHECK(native_buffer.size == byte_count);
    PmMemoryBudget budget{};
    CHECK(pmGetDeviceMemoryBudget(device, &budget) == PM_SUCCESS);
    CHECK(budget.heapCount == 1U);
    const auto expected_budget_confidence =
        PM_TEST_NATIVE_BACKEND == PM_NATIVE_BACKEND_CUDA
        ? PM_MEMORY_BUDGET_CONFIDENCE_DRIVER_REPORTED
        : PM_TEST_NATIVE_BACKEND == PM_NATIVE_BACKEND_VULKAN
        ? PM_MEMORY_BUDGET_CONFIDENCE_IMPLEMENTATION_ESTIMATE
        : PM_MEMORY_BUDGET_CONFIDENCE_OS_REPORTED;
    CHECK(budget.confidence[0] == expected_budget_confidence);
    CHECK(budget.usage[0] <= budget.budget[0]);
    CHECK(budget.applicationAllocationBytes >= byte_count * 2U);

    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    // Exercise the no-leaf-cache schedule used by real galleries (>8 slots).
    simulation_info.rigidBodyCapacity = 12U;
    simulation_info.contactCapacity = 8U;
    simulation_info.debugFrameCapacity = 2U;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) == PM_SUCCESS);
    PmSimulationMemoryRequirements simulation_requirements{};
    pmGetSimulationMemoryRequirements(device, simulation,
                                      &simulation_requirements);
    allocation.memoryTypeIndex = 1U;
    allocation.allocationSize = simulation_requirements.persistent.size;
    PmDeviceMemory persistent{};
    CHECK(pmAllocateMemory(device, &allocation, &persistent) == PM_SUCCESS);
    allocation.allocationSize = simulation_requirements.scratch.size;
    PmDeviceMemory scratch{};
    CHECK(pmAllocateMemory(device, &allocation, &scratch) == PM_SUCCESS);
    const PmSimulationMemoryBinding simulation_binding{
        persistent, 0U, scratch, 0U};
    CHECK(pmBindSimulationMemory(device, simulation, &simulation_binding) ==
          PM_SUCCESS);

    const float cube_vertices[8][3] = {
        {-0.5F, -0.5F, -0.5F}, {0.5F, -0.5F, -0.5F},
        {0.5F, 0.5F, -0.5F}, {-0.5F, 0.5F, -0.5F},
        {-0.5F, -0.5F, 0.5F}, {0.5F, -0.5F, 0.5F},
        {0.5F, 0.5F, 0.5F}, {-0.5F, 0.5F, 0.5F}};
    const std::uint32_t cube_indices[36] = {
        0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
        3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
    std::memcpy(mapped_upload, cube_vertices, sizeof cube_vertices);
    std::memcpy(static_cast<std::byte *>(mapped_upload) + sizeof cube_vertices,
                cube_indices, sizeof cube_indices);
    PmGeometryCreateInfo geometry_info{};
    geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    geometry_info.vertices = {{source, 0U, sizeof cube_vertices},
                              PM_FORMAT_R32G32B32_SFLOAT, 8U, 12U};
    geometry_info.indices = {{source, sizeof cube_vertices, sizeof cube_indices},
                             PM_FORMAT_R32_UINT, 36U, 4U};
    PmGeometry geometry{};
    CHECK(pmCreateGeometry(device, &geometry_info, &geometry) == PM_SUCCESS);
    PmMemoryRequirements geometry_requirements{};
    pmGetGeometryMemoryRequirements(device, geometry, &geometry_requirements);
    allocation.allocationSize = geometry_requirements.size;
    PmDeviceMemory geometry_memory{};
    CHECK(pmAllocateMemory(device, &allocation, &geometry_memory) == PM_SUCCESS);
    CHECK(pmBindGeometryMemory(device, geometry, geometry_memory, 0U) == PM_SUCCESS);

    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    PmPhysicsPipeline pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) == PM_SUCCESS);
    const PmQueryPoolCreateInfo timing_info{PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        nullptr, PM_QUERY_TIMESTAMP, 2U};
    PmQueryPool timing{};
    if (features.timestampQueries)
        CHECK(pmCreateQueryPool(device, &timing_info, &timing) == PM_SUCCESS);
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBuildGeometry(commands, geometry) == PM_SUCCESS);
    PmRigidBodyInfo bodies[2]{};
    for (auto &body : bodies) {
        body.active = PM_TRUE;
        body.geometry = geometry;
        body.mass = 1.0F;
        body.inertiaDiagonal[0] = body.inertiaDiagonal[1] =
            body.inertiaDiagonal[2] = 1.0F / 6.0F;
        body.state.orientation[3] = 1.0F;
        body.maximumLinearSpeed = body.maximumAngularSpeed = 20.0F;
        body.collisionMargin = 0.005F;
    }
    bodies[0].motionType = PM_MOTION_STATIC;
    bodies[0].state.position[1] = -1.0F;
    bodies[1].motionType = PM_MOTION_DYNAMIC;
    bodies[1].state.position[1] = 1.0F;
    CHECK(pmCmdWriteRigidBodies(commands, simulation, 0U, 2U, bodies) ==
          PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    if (timing)
        CHECK(pmCmdWriteTimestamp(commands, PM_STAGE_ALL_COMMANDS_BIT, timing, 0U) == PM_SUCCESS);
    const PmStepInfo step{1.0F / 60.0F, 1U, {0.0F, -9.81F, 0.0F},
                          PM_STEP_COLLECT_CONTACT_EVENTS_BIT};
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    const PmBufferRange state_output{
        destination, 0U, 2U * sizeof(PmRigidState)};
    CHECK(pmCmdCopyRigidStates(commands, simulation, 0U, 2U, &state_output) ==
          PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    const auto *states = static_cast<const PmRigidState *>(mapped_readback);
    CHECK(states[0].position[1] == -1.0F);
    CHECK(std::isfinite(states[1].position[1]));
    const bool dynamic_body_advanced =
        std::abs(states[1].position[1] - bodies[1].state.position[1]) > 1.0e-4F ||
        std::abs(states[1].linearVelocity[1]) > 1.0e-4F;
    if (!dynamic_body_advanced)
        std::fprintf(stderr, "Native rigid body did not advance: y=%g vy=%g\n",
                     states[1].position[1], states[1].linearVelocity[1]);
    CHECK(dynamic_body_advanced);
    const float first_position = states[1].position[1];
    const float first_velocity = states[1].linearVelocity[1];

    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdCopyRigidStates(commands, simulation, 0U, 2U, &state_output) ==
          PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    if (!(states[1].position[1] < first_position - 1.0e-4F &&
          states[1].linearVelocity[1] < first_velocity - 1.0e-4F))
        std::fprintf(stderr,
                     "Native rigid body did not advance across submissions: "
                     "first y=%g vy=%g, second y=%g vy=%g\n",
                     first_position, first_velocity, states[1].position[1],
                     states[1].linearVelocity[1]);
    CHECK(states[1].position[1] < first_position - 1.0e-4F);
    CHECK(states[1].linearVelocity[1] < first_velocity - 1.0e-4F);
    // Continue through contact, then read a timestamp spanning submissions.
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    for (std::uint32_t frame = 0U; frame < 30U; ++frame)
        CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    if (timing)
        CHECK(pmCmdWriteTimestamp(commands, PM_STAGE_ALL_COMMANDS_BIT, timing, 1U) == PM_SUCCESS);
    CHECK(pmCmdCopyRigidStates(commands, simulation, 0U, 2U, &state_output) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE, PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    CHECK(std::isfinite(states[1].position[1]));
    CHECK(states[1].position[1] > -0.5F);
    if (timing) {
        std::uint64_t timestamps[2]{};
        CHECK(pmGetQueryPoolResults(device, timing, 0U, 2U, sizeof timestamps,
            timestamps, sizeof timestamps[0], PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
        CHECK(timestamps[1] > timestamps[0]);
    }

#if defined(PM_TEST_METAL_PREVIEW)
    // Separate contact case: preserve upstream free-fall/multiple-submission
    // checks above as well as the preview's independent settling check.
    bodies[1].state.position[1] = 0.02F;
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdWriteRigidBodies(commands, simulation, 0U, 2U, bodies) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    for (std::uint32_t frame = 0U; frame < 30U; ++frame)
        CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdCopyRigidStates(commands, simulation, 0U, 2U, &state_output) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE, PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    if (!(states[1].position[1] > -0.025F &&
          states[1].position[1] < 0.08F))
        std::fprintf(stderr,
            "Metal rigid settling outside tolerance: y=%g vy=%g\n",
            states[1].position[1], states[1].linearVelocity[1]);
    CHECK(states[1].position[1] > -0.025F && states[1].position[1] < 0.08F);
    CHECK(std::abs(states[1].linearVelocity[1]) < 0.15F);
#endif
    // Retire timestamp and multi-step commands before repeated submissions.
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdCopyRigidStates(commands, simulation, 0U, 2U, &state_output) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    if (timing) pmDestroyQueryPool(device, timing);

    // Reuse the same executable command buffer over six simulated seconds.
    // The cube must settle on the static cube, not merely remain finite.
    for (std::uint32_t frame = 32U; frame < 360U; ++frame) {
        CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
        CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
        CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                              PM_TIMEOUT_INFINITE) == PM_SUCCESS);
        CHECK(states[0].position[1] == -1.0F);
        CHECK(std::isfinite(states[1].position[1]));
        CHECK(states[1].position[1] > -0.025F);
    }
    CHECK(std::abs(states[1].position[1]) < 0.03F);
    CHECK(std::abs(states[1].linearVelocity[1]) < 0.1F);

    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroySimulation(device, simulation);
    pmDestroyGeometry(device, geometry);
    pmFreeMemory(device, geometry_memory);
    pmFreeMemory(device, scratch);
    pmFreeMemory(device, persistent);

    pmDestroyFence(device, fence);
    pmFreeCommandBuffers(device, pool, 1U, &commands);
    pmDestroyCommandPool(device, pool);
    pmUnmapMemory(device, readback);
    pmUnmapMemory(device, upload);
    pmDestroyBuffer(device, destination);
    pmDestroyBuffer(device, source);
    pmFreeMemory(device, readback);
    pmFreeMemory(device, upload);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: native transfer, handles and shared Slang rigid contact schedule");
    return 0;
}
