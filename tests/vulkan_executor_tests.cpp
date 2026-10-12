// SPDX-License-Identifier: MIT
#include "device/vulkan_executor.hpp"

#include <cstdio>
#include <memory>
#include <vector>

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

int main() {
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "Parallel Mater Vulkan executor test";
    application.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application;
    VkInstance instance{};
    const VkResult instance_result =
        vkCreateInstance(&instance_info, nullptr, &instance);
    if (instance_result == VK_ERROR_INCOMPATIBLE_DRIVER) return 77;
    CHECK(instance_result == VK_SUCCESS);

    std::uint32_t physical_count{};
    CHECK(vkEnumeratePhysicalDevices(instance, &physical_count, nullptr) ==
          VK_SUCCESS);
    if (physical_count == 0U) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    std::vector<VkPhysicalDevice> physicals(physical_count);
    CHECK(vkEnumeratePhysicalDevices(instance, &physical_count,
                                     physicals.data()) == VK_SUCCESS);
    std::uint32_t family_count{};
    vkGetPhysicalDeviceQueueFamilyProperties(physicals[0], &family_count,
                                              nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicals[0], &family_count,
                                              families.data());
    std::uint32_t family = UINT32_MAX;
    for (std::uint32_t index = 0U; index < family_count; ++index)
        if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U) {
            family = index;
            break;
        }
    if (family == UINT32_MAX) {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1U;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueCreateInfos = &queue_info;
    VkDevice device{};
    CHECK(vkCreateDevice(physicals[0], &device_info, nullptr, &device) ==
          VK_SUCCESS);
    VkQueue queue{};
    vkGetDeviceQueue(device, family, 0U, &queue);

    std::unique_ptr<parallel_mater::device::VulkanExecutor> executor;
    CHECK(parallel_mater::device::VulkanExecutor::create(
              device, queue, family, executor) == PM_SUCCESS);
    VkCommandBuffer command{};
    CHECK(executor->allocate_command_buffer(command) == PM_SUCCESS);
    CHECK(executor->begin_command_buffer(command) == PM_SUCCESS);
    executor->cmd_compute_barrier(command);
    CHECK(executor->end_command_buffer(command) == PM_SUCCESS);
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence{};
    CHECK(vkCreateFence(device, &fence_info, nullptr, &fence) == VK_SUCCESS);
    CHECK(executor->queue_submit(command, VK_NULL_HANDLE, 0U, VK_NULL_HANDLE,
                                 0U, fence) == PM_SUCCESS);
    CHECK(vkWaitForFences(device, 1U, &fence, VK_TRUE, UINT64_MAX) ==
          VK_SUCCESS);
    executor->free_command_buffer(command);
    vkDestroyFence(device, fence, nullptr);
    executor.reset();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::puts("PASS: Vulkan executor command lifecycle and submission");
    return 0;
}
