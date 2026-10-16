// SPDX-License-Identifier: MIT
// Isolate driver shader compilation from the loader, scheduler and physics.
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
void check(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "%s: VkResult %d\n", operation, int(result));
        throw std::runtime_error(operation);
    }
}
struct Resources {
    VkInstance instance{};
    VkDevice device{};
    VkDescriptorSetLayout descriptors{};
    VkPipelineLayout layout{};
    VkShaderModule shader{};
    VkPipeline pipeline{};
    ~Resources() {
        if (device) {
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (shader) vkDestroyShaderModule(device, shader, nullptr);
            if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
            if (descriptors) vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};
} // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s shader.spv entry_point workgroup_size_x\n", argv[0]);
        return 2;
    }
    try {
        char *end{};
        const auto local = std::strtoul(argv[3], &end, 10);
        if (*end || local == 0 || local > 1024) throw std::runtime_error("Invalid workgroup size");
        const std::uint32_t block = static_cast<std::uint32_t>(local);
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("Cannot open SPIR-V file");
        const auto bytes = input.tellg();
        if (bytes < 20 || bytes > 64 * 1024 * 1024 || bytes % 4 != 0)
            throw std::runtime_error("Invalid SPIR-V file size");
        std::vector<std::uint32_t> code(static_cast<std::size_t>(bytes) / 4);
        input.seekg(0);
        if (!input.read(reinterpret_cast<char *>(code.data()), bytes) || code[0] != 0x07230203)
            throw std::runtime_error("Invalid SPIR-V file");
        Resources r;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Parallel Mater pipeline probe";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo instance{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance.pApplicationInfo = &app;
        const char *layer = "VK_LAYER_KHRONOS_validation";
        const char *validation = std::getenv("PM_VULKAN_VALIDATION");
        if (validation && std::strcmp(validation, "1") == 0) {
            instance.enabledLayerCount = 1;
            instance.ppEnabledLayerNames = &layer;
        }
        check(vkCreateInstance(&instance, nullptr, &r.instance), "vkCreateInstance");
        std::uint32_t count{};
        check(vkEnumeratePhysicalDevices(r.instance, &count, nullptr), "enumerate devices");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(r.instance, &count, devices.data()), "enumerate devices");
        for (auto physical : devices) {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(physical, &properties);
            if (properties.apiVersion < VK_API_VERSION_1_3 || properties.limits.maxPushConstantsSize < 256 ||
                properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
            VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12};
            vkGetPhysicalDeviceFeatures2(physical, &features);
            if (!f13.maintenance4 || !f12.bufferDeviceAddress || !f12.scalarBlockLayout ||
                !f12.storageBuffer8BitAccess || !f12.storagePushConstant8 || !f12.shaderInt8 ||
                !f12.shaderBufferInt64Atomics || !features.features.shaderInt64) continue;
            std::uint32_t families{};
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
            std::vector<VkQueueFamilyProperties> queues(families);
            vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, queues.data());
            std::uint32_t family{};
            while (family < families && !(queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
            if (family == families) continue;
            std::fprintf(stderr, "device=%s shader=%s entry=%s bytes=%zu block=%u\n",
                properties.deviceName, argv[1], argv[2], code.size() * 4, block);
            f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            f13.maintenance4 = VK_TRUE;
            f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
            f12.bufferDeviceAddress = f12.scalarBlockLayout = f12.storageBuffer8BitAccess = VK_TRUE;
            f12.storagePushConstant8 = f12.shaderInt8 = f12.shaderBufferInt64Atomics = VK_TRUE;
            VkPhysicalDeviceFeatures enabled{};
            enabled.shaderInt64 = VK_TRUE;
            float priority = 1;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family; queue.queueCount = 1; queue.pQueuePriorities = &priority;
            VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12};
            device.queueCreateInfoCount = 1; device.pQueueCreateInfos = &queue; device.pEnabledFeatures = &enabled;
            check(vkCreateDevice(physical, &device, nullptr, &r.device), "vkCreateDevice");
            break;
        }
        if (!r.device) throw std::runtime_error("No compatible Vulkan device");
        VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo descriptors{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptors.bindingCount = 1; descriptors.pBindings = &binding;
        check(vkCreateDescriptorSetLayout(r.device, &descriptors, nullptr, &r.descriptors), "descriptor layout");
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 256};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1; layout.pSetLayouts = &r.descriptors;
        layout.pushConstantRangeCount = 1; layout.pPushConstantRanges = &push;
        check(vkCreatePipelineLayout(r.device, &layout, nullptr, &r.layout), "pipeline layout");
        VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = code.size() * 4; shader.pCode = code.data();
        check(vkCreateShaderModule(r.device, &shader, nullptr, &r.shader), "shader module");
        VkSpecializationMapEntry entry{0, 0, sizeof(block)};
        VkSpecializationInfo specialization{1, &entry, sizeof(block), &block};
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.layout = r.layout;
        pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline.stage.module = r.shader; pipeline.stage.pName = argv[2];
        pipeline.stage.pSpecializationInfo = &specialization;
        check(vkCreateComputePipelines(r.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &r.pipeline), "compute pipeline");
        std::puts("PIPELINE_OK (compilation only; no physics dispatched)");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
