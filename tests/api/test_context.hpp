// SPDX-License-Identifier: MIT
#pragma once
// Public SDK only. Select the driver with PM_DRIVER_FILES; no gallery assets.
#include <parallel_mater/parallel_mater.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define PM_CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #value); std::abort(); \
} } while (false)

namespace parallel_mater::test {
struct Buffer {
    PmBuffer handle{};
    PmDeviceMemory memory{};
    void *mapped{};
    PmDeviceSize size{};
};
class Context {
public:
    PmInstance instance{};
    PmPhysicalDevice physical{};
    PmPhysicalDeviceFeatures supported{};
    PmDevice device{};
    PmQueue queue{};
    PmCommandPool pool{};
    PmCommandBuffer commands{};
    PmFence fence{};
    std::vector<PmDeviceMemory> memory;
    std::vector<Buffer> buffers;

    bool open(PmPhysicalDeviceFeatures needed) {
        PmInstanceCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.apiVersion = PM_API_VERSION_0_2;
        PM_CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_SUCCESS);
        std::uint32_t count{};
        PM_CHECK(pmEnumeratePhysicalDevices(instance, &count, nullptr) == PM_SUCCESS);
        std::vector<PmPhysicalDevice> devices(count);
        PM_CHECK(pmEnumeratePhysicalDevices(instance, &count, devices.data()) == PM_SUCCESS);
        for (auto candidate : devices) {
            PmPhysicalDeviceFeatures features{};
            pmGetPhysicalDeviceFeatures(candidate, &features);
            // The fixture deliberately checks the capabilities its tests use.
            if ((needed.rigidBodies && !features.rigidBodies) ||
                (needed.particleLiquids && !features.particleLiquids) ||
                (needed.cloth && !features.cloth) ||
                (needed.timelineSemaphores && !features.timelineSemaphores)) continue;
            physical = candidate; supported = features; break;
        }
        if (!physical) return false;
        const float priority = 1;
        const PmDeviceQueueCreateInfo queue_info{0, 1, &priority};
        PmDeviceCreateInfo create{};
        create.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create.queueCreateInfoCount = 1;
        create.pQueueInfos = &queue_info;
        create.enabledFeatures = needed;
        PM_CHECK(pmCreateDevice(physical, &create, &device) == PM_SUCCESS);
        pmGetDeviceQueue(device, 0, 0, &queue);
        PmCommandPoolCreateInfo pool_info{};
        pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
        PM_CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
        PM_CHECK(pmAllocateCommandBuffers(device, pool, 1, &commands) == PM_SUCCESS);
        PM_CHECK(pmCreateFence(device, 0, &fence) == PM_SUCCESS);
        return true;
    }
    PmDeviceMemory allocate(PmMemoryRequirements requirements) {
        PmMemoryProperties properties{};
        pmGetPhysicalDeviceMemoryProperties(physical, &properties);
        std::uint32_t type = PM_INVALID_INDEX;
        for (std::uint32_t i = 0; i < properties.typeCount; ++i)
            if ((requirements.memoryTypeBits & (1U << i)) &&
                (properties.types[i].propertyFlags & PM_MEMORY_HOST_VISIBLE_BIT)) { type = i; break; }
        PM_CHECK(type != PM_INVALID_INDEX);
        PmMemoryAllocateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.allocationSize = requirements.size;
        info.memoryTypeIndex = type;
        PmDeviceMemory allocation{};
        PM_CHECK(pmAllocateMemory(device, &info, &allocation) == PM_SUCCESS);
        memory.push_back(allocation);
        return allocation;
    }
    Buffer buffer(PmDeviceSize size, PmFlags usage) {
        PmBufferCreateInfo info{};
        info.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = size; info.usage = usage;
        Buffer result{};
        result.size = size;
        PM_CHECK(pmCreateBuffer(device, &info, &result.handle) == PM_SUCCESS);
        PmMemoryRequirements requirements{};
        pmGetBufferMemoryRequirements(device, result.handle, &requirements);
        result.memory = allocate(requirements);
        PM_CHECK(pmBindBufferMemory(device, result.handle, result.memory, 0) == PM_SUCCESS);
        PM_CHECK(pmMapMemory(device, result.memory, 0, size, &result.mapped) == PM_SUCCESS);
        buffers.push_back(result);
        return result;
    }
    void flush(Buffer buffer) {
        const PmMappedMemoryRange range{buffer.memory, 0, buffer.size};
        PM_CHECK(pmFlushMappedMemoryRanges(device, 1, &range) == PM_SUCCESS);
    }
    void invalidate(Buffer buffer) {
        const PmMappedMemoryRange range{buffer.memory, 0, buffer.size};
        PM_CHECK(pmInvalidateMappedMemoryRanges(device, 1, &range) == PM_SUCCESS);
    }
    void begin() {
        PM_CHECK(pmResetFences(device, 1, &fence) == PM_SUCCESS);
        PM_CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
        PmCommandBufferBeginInfo info{};
        info.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        PM_CHECK(pmBeginCommandBuffer(commands, &info) == PM_SUCCESS);
    }
    PmResult submit() {
        PM_CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
        PmSubmitInfo info{};
        info.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
        info.commandBufferCount = 1; info.pCommandBuffers = &commands;
        PM_CHECK(pmQueueSubmit(queue, &info, fence) == PM_SUCCESS);
        return pmWaitForFences(device, 1, &fence, PM_TRUE, PM_TIMEOUT_INFINITE);
    }
    ~Context() {
        if (device) {
            (void)pmDeviceWaitIdle(device);
            pmDestroyFence(device, fence);
            pmDestroyCommandPool(device, pool);
            for (const auto &buffer : buffers) {
                pmUnmapMemory(device, buffer.memory);
                pmDestroyBuffer(device, buffer.handle);
            }
            for (auto allocation : memory) pmFreeMemory(device, allocation);
            pmDestroyDevice(device);
        }
        if (instance) pmDestroyInstance(instance);
    }
};
} // namespace parallel_mater::test
