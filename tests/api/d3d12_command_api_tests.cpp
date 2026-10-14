// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t physical_count{};
    CHECK(pmEnumeratePhysicalDevices(instance, &physical_count, nullptr) ==
          PM_SUCCESS);
    if (physical_count == 0U) {
        pmDestroyInstance(instance);
        std::puts("SKIP: no D3D12 device with Shader Model 6.6");
        return 77;
    }
    std::vector<PmPhysicalDevice> physicals(physical_count);
    CHECK(pmEnumeratePhysicalDevices(
        instance, &physical_count, physicals.data()) == PM_SUCCESS);
    const PmPhysicalDevice physical = physicals.front();
    PmPhysicalDeviceFeatures features{};
    pmGetPhysicalDeviceFeatures(physical, &features);
    CHECK(features.rigidBodies == PM_FALSE);
    CHECK(features.memoryBudget == PM_TRUE);
    std::uint32_t family_count{};
    CHECK(pmGetPhysicalDeviceQueueFamilies(
        physical, &family_count, nullptr) == PM_SUCCESS);
    CHECK(family_count == 1U);
    PmQueueFamilyProperties family{};
    CHECK(pmGetPhysicalDeviceQueueFamilies(
        physical, &family_count, &family) == PM_SUCCESS);
    CHECK((family.queueFlags & PM_QUEUE_TRANSFER_BIT) != 0U);
    CHECK((family.queueFlags & PM_QUEUE_PHYSICS_BIT) == 0U);

    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures.memoryBudget = PM_TRUE;
    PmDevice device{};
    // Device creation compiles both embedded Slang DXIL entry points into
    // D3D12 PSOs, dispatches prepare/solve, fences, and validates the schedule.
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
    CHECK(pmAllocateMemory(device, &allocation, &upload) == PM_SUCCESS);
    allocation.memoryTypeIndex = 2U;
    PmDeviceMemory readback{};
    CHECK(pmAllocateMemory(device, &allocation, &readback) == PM_SUCCESS);

    PmBufferCreateInfo buffer_info{};
    buffer_info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = byte_count;
    buffer_info.usage = PM_BUFFER_TRANSFER_SRC_BIT;
    PmBuffer source{};
    CHECK(pmCreateBuffer(device, &buffer_info, &source) == PM_SUCCESS);
    buffer_info.usage = PM_BUFFER_TRANSFER_DST_BIT;
    PmBuffer destination{};
    CHECK(pmCreateBuffer(device, &buffer_info, &destination) == PM_SUCCESS);
    PmMemoryRequirements requirements{};
    pmGetBufferMemoryRequirements(device, source, &requirements);
    CHECK((requirements.memoryTypeBits & (1U << 1U)) != 0U);
    pmGetBufferMemoryRequirements(device, destination, &requirements);
    CHECK((requirements.memoryTypeBits & (1U << 2U)) != 0U);
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

    PmNativeDeviceInfo native_device{};
    PmNativeBufferInfo native_buffer{};
    CHECK(pmGetNativeDevice(device, &native_device) == PM_SUCCESS);
    CHECK(native_device.backend == PM_NATIVE_BACKEND_D3D12);
    CHECK(native_device.device != 0U);
    CHECK(pmGetNativeBuffer(device, destination, &native_buffer) == PM_SUCCESS);
    CHECK(native_buffer.buffer != 0U);
    CHECK(native_buffer.size == byte_count);
    PmMemoryBudget budget{};
    CHECK(pmGetDeviceMemoryBudget(device, &budget) == PM_SUCCESS);
    CHECK(budget.heapCount == 1U);
    CHECK(budget.applicationAllocationBytes >= byte_count * 2U);

    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(simulation == nullptr);

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
    std::puts("PASS: loader D3D12 transfer API and shared Slang AVBD DXIL dispatch");
    return 0;
}
