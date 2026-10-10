// SPDX-License-Identifier: MIT
#include <jni.h>
#include <parallel_mater/vulkan.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace parallel_mater;
using namespace parallel_mater::vulkan;

namespace {

struct NativeOwner {
    VkInstance instance{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    ~NativeOwner() {
        if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }
};

std::string status_message(const char *prefix, Status status) {
    return std::string(prefix) + ": " +
        (status.message ? status.message : "unknown") + " (VkResult " +
        std::to_string(status.vulkan_result) + ")";
}

std::string run_smoke() {
    NativeOwner owner;
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "ParallelMater Android smoke";
    application.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    VkResult result = vkCreateInstance(&instance_info, nullptr, &owner.instance);
    if (result != VK_SUCCESS)
        return "vkCreateInstance failed: " + std::to_string(result);

    std::uint32_t physical_count = 0U;
    result = vkEnumeratePhysicalDevices(owner.instance, &physical_count, nullptr);
    if (result != VK_SUCCESS || physical_count == 0U)
        return "No Vulkan physical device";
    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    result = vkEnumeratePhysicalDevices(owner.instance, &physical_count,
                                        physical_devices.data());
    if (result != VK_SUCCESS) return "Could not enumerate Vulkan devices";
    auto get_features = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        vkGetInstanceProcAddr(owner.instance, "vkGetPhysicalDeviceFeatures2"));
    if (!get_features) {
        get_features = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            vkGetInstanceProcAddr(owner.instance,
                                  "vkGetPhysicalDeviceFeatures2KHR"));
    }
    if (!get_features)
        return "Vulkan loader lacks physical-device feature queries";

    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::uint32_t family_index = UINT32_MAX;
    for (VkPhysicalDevice candidate : physical_devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.apiVersion < VK_API_VERSION_1_2 ||
            properties.limits.maxComputeWorkGroupInvocations < 64U ||
            properties.limits.maxComputeWorkGroupSize[0] < 64U ||
            properties.limits.maxPerStageDescriptorStorageBuffers < 9U ||
            properties.limits.maxDescriptorSetStorageBuffers < 9U)
            continue;
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        VkPhysicalDeviceFeatures2 features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &timeline;
        get_features(candidate, &features);
        if (!timeline.timelineSemaphore) continue;
        std::uint32_t family_count = 0U;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count,
                                                 nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &family_count,
                                                 families.data());
        for (std::uint32_t family = 0U; family < family_count; ++family) {
            if (families[family].queueCount != 0U &&
                (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                physical = candidate;
                family_index = family;
                break;
            }
        }
        if (physical != VK_NULL_HANDLE) break;
    }
    if (physical == VK_NULL_HANDLE)
        return "Device does not meet Vulkan 1.2 compute/timeline limits";

    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family_index;
    queue_info.queueCount = 1U;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timeline.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &timeline;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueCreateInfos = &queue_info;
    result = vkCreateDevice(physical, &device_info, nullptr, &owner.device);
    if (result != VK_SUCCESS)
        return "vkCreateDevice failed: " + std::to_string(result);
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(owner.device, family_index, 0U, &queue);

    {
        WorldOptions options{};
        options.rigid_body_capacity = 1U;
        options.triangle_mesh_capacity = 1U;
        World world;
        Status status = World::create(
            options,
            {owner.instance, physical, owner.device, queue, family_index, true},
            world);
        if (!status) return status_message("World::create", status);
        const std::array<Vec3, 3U> vertices{{
            {-1.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F},
            {0.0F, 1.0F, 0.0F},
        }};
        const std::array<std::uint32_t, 3U> indices{0U, 1U, 2U};
        TriangleMeshId mesh{};
        status = world.add_triangle_mesh({vertices.data(), vertices.size()},
                                         {indices.data(), indices.size()}, mesh);
        if (!status) return status_message("add_triangle_mesh", status);
        RigidBodyOptions rigid_options{};
        rigid_options.mesh = mesh;
        rigid_options.linear_damping = 0.0F;
        RigidBodyId body{};
        status = world.add_rigid_body(rigid_options, body);
        if (!status) return status_message("add_rigid_body", status);

        FrameToken token;
        StepOptions step{};
        step.timestep = 0.25F;
        step.substeps = 1U;
        step.gravity = {0.0F, -4.0F, 0.0F};
        status = world.step_async(step, token);
        if (!status) return status_message("step_async", status);
        const NativeCompletion completion = token.native_completion();
        if (completion.semaphore == VK_NULL_HANDLE || completion.value == 0U)
            return "Missing native timeline completion";

        VkTimelineSemaphoreSubmitInfo wait_timeline{
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        wait_timeline.waitSemaphoreValueCount = 1U;
        wait_timeline.pWaitSemaphoreValues = &completion.value;
        const VkPipelineStageFlags wait_stage =
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo wait_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        wait_submit.pNext = &wait_timeline;
        wait_submit.waitSemaphoreCount = 1U;
        wait_submit.pWaitSemaphores = &completion.semaphore;
        wait_submit.pWaitDstStageMask = &wait_stage;
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        result = vkCreateFence(owner.device, &fence_info, nullptr, &fence);
        if (result != VK_SUCCESS) return "Could not create GPU-wait fence";
        result = vkQueueSubmit(queue, 1U, &wait_submit, fence);
        if (result == VK_SUCCESS)
            result = vkWaitForFences(owner.device, 1U, &fence, VK_TRUE,
                                     UINT64_MAX);
        vkDestroyFence(owner.device, fence, nullptr);
        if (result != VK_SUCCESS) return "GPU-side timeline wait failed";

        RigidBodyState state{};
        status = world.read_rigid_body_state(body, state);
        if (!status) return status_message("read_rigid_body_state", status);
        if (std::abs(state.linear_velocity.y + 1.0F) > 2.0e-4F ||
            std::abs(state.position.y + 0.25F) > 2.0e-4F)
            return "Rigid integration result mismatch";
    }

    VkSemaphoreCreateInfo semaphore_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore probe = VK_NULL_HANDLE;
    result = vkCreateSemaphore(owner.device, &semaphore_info, nullptr, &probe);
    if (result != VK_SUCCESS)
        return "Backend destroyed caller-owned Vulkan handles";
    vkDestroySemaphore(owner.device, probe, nullptr);
    return {};
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_ballroller_parallelmater_vulkan_smoke_VulkanSmokeBridge_run(
    JNIEnv *environment, jclass) {
    const std::string result = run_smoke();
    return environment->NewStringUTF(result.c_str());
}
