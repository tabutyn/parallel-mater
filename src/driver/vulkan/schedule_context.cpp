// SPDX-License-Identifier: MIT
#include "schedule_context.hpp"
#include "simulation/geometry_storage.hpp"
#include "simulation/paint_storage.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace parallel_mater::vulkan {
namespace {
thread_local ScheduleContext *active_context{};
void check(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "Parallel Mater Vulkan: %s failed (%d)\n", operation,
                     int(result));
        throw std::runtime_error(operation);
    }
}

std::vector<const char *> available_instance_extensions() {
    std::uint32_t count{};
    check(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
          "enumerate instance extensions");
    std::vector<VkExtensionProperties> available(count);
    check(vkEnumerateInstanceExtensionProperties(
              nullptr, &count, available.data()),
          "enumerate instance extensions");
    constexpr const char *candidates[]{
        "VK_KHR_surface", "VK_KHR_xcb_surface", "VK_KHR_xlib_surface",
        "VK_KHR_wayland_surface", "VK_KHR_win32_surface",
        "VK_KHR_android_surface", "VK_EXT_metal_surface",
        "VK_MVK_macos_surface"};
    std::vector<const char *> enabled;
    for (const char *candidate : candidates) {
        if (std::any_of(available.begin(), available.end(),
                [&](const VkExtensionProperties &extension) {
                    return std::strcmp(extension.extensionName, candidate) == 0;
                }))
            enabled.push_back(candidate);
    }
    return enabled;
}

std::vector<const char *> available_device_extensions(
    VkPhysicalDevice physical) {
    std::uint32_t count{};
    check(vkEnumerateDeviceExtensionProperties(
              physical, nullptr, &count, nullptr),
          "enumerate device extensions");
    std::vector<VkExtensionProperties> available(count);
    check(vkEnumerateDeviceExtensionProperties(
              physical, nullptr, &count, available.data()),
          "enumerate device extensions");
    std::vector<const char *> enabled;
    if (std::any_of(available.begin(), available.end(),
            [](const VkExtensionProperties &extension) {
                return std::strcmp(extension.extensionName,
                                   VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
            }))
        enabled.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    return enabled;
}

struct Instance {
    VkInstance handle{};
    Instance() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Parallel Mater";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &app;
        const auto extensions = available_instance_extensions();
        info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
        info.ppEnabledExtensionNames = extensions.data();
        const char *validation = "VK_LAYER_KHRONOS_validation";
        if (const char *enabled = std::getenv("PM_VULKAN_VALIDATION");
            enabled && std::strcmp(enabled, "1") == 0) {
            info.enabledLayerCount = 1;
            info.ppEnabledLayerNames = &validation;
        }
        check(vkCreateInstance(&info, nullptr, &handle), "vkCreateInstance");
    }
    ~Instance() {
        if (handle)
            vkDestroyInstance(handle, nullptr);
    }
};
struct Adapter {
    VkPhysicalDevice handle;
    VkPhysicalDeviceProperties properties;
    std::uint32_t family;
    std::uint32_t timestamp_bits;
    std::array<std::uint8_t, VK_UUID_SIZE> uuid;
};
std::vector<Adapter> adapters(VkInstance instance) {
    std::uint32_t count{};
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate devices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()),
          "enumerate devices");
    std::vector<Adapter> result;
    for (auto physical : devices) {
        VkPhysicalDeviceProperties properties{};
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
        vkGetPhysicalDeviceProperties2(physical, &properties2);
        properties = properties2.properties;
        std::array<std::uint8_t, VK_UUID_SIZE> uuid{};
        std::memcpy(uuid.data(), id.deviceUUID, uuid.size());
        if (properties.apiVersion < VK_API_VERSION_1_3 ||
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ||
            properties.limits.maxPushConstantsSize < 256U)
            continue;
        VkPhysicalDeviceVulkan12Features f12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan13Features f13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f12.pNext = &f13;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                           &f12};
        vkGetPhysicalDeviceFeatures2(physical, &features);
        if (!f13.maintenance4 || !f12.bufferDeviceAddress || !f12.scalarBlockLayout ||
            !f12.storageBuffer8BitAccess ||
            !f12.uniformAndStorageBuffer8BitAccess || !f12.storagePushConstant8 ||
            !f12.shaderInt8 ||
            !f12.shaderBufferInt64Atomics || !features.features.shaderInt64)
            continue;
        // Some installations expose the same adapter through multiple ICDs.
        if (std::any_of(result.begin(), result.end(), [&](const Adapter &a) {
                return a.properties.vendorID == properties.vendorID &&
                       a.properties.deviceID == properties.deviceID &&
                       a.uuid == uuid;
            }))
            continue;
        std::uint32_t families{};
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
        std::vector<VkQueueFamilyProperties> queues(families);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, queues.data());
        for (std::uint32_t i = 0; i < families; ++i)
            if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                queues[i].timestampValidBits) {
                result.push_back({physical, properties, i, queues[i].timestampValidBits, uuid});
                break;
            }
    }
    std::stable_sort(
        result.begin(), result.end(), [](const Adapter &a, const Adapter &b) {
            return a.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
                   b.properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        });
    return result;
}
} // namespace

struct ScheduleContext::Impl {
    struct Buffer {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        void *host{};
        VkDeviceAddress address{};
        std::size_t size{};
    };
    struct Pipeline {
        VkPipeline pipeline{};
    };
    Instance instance;
    Adapter adapter{};
    int ordinal{};
    VkDevice device{};
    VkQueue queue{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkCommandPool pool{};
    VkCommandBuffer commands{};
    VkFence fence{};
    VkDescriptorSetLayout descriptors[2]{};
    VkDescriptorPool descriptor_pool{};
    VkDescriptorSet descriptor_sets[2]{};
    VkPipelineLayout layouts[2]{};
    VkPipelineCache pipeline_cache{};
    std::filesystem::path cache_file;
    std::map<std::string, Pipeline> pipelines;
    std::vector<Buffer> allocations;
    std::vector<Buffer> temporary;
    std::map<const void *, std::pair<std::uint32_t, VkDeviceAddress>> meshes;
    std::map<const void *, std::uint32_t> mesh_capacities;
    std::map<const void *, std::pair<std::uint32_t, VkDeviceAddress>> paint_fields;
    Buffer globals{};
    Buffer arguments{};
    std::uint32_t global_offset{};
    std::uint32_t argument_offset{};
    std::uint32_t global_stride{256};
    std::uint32_t parameter_stride{256};
    static constexpr std::uint32_t parameter_range = 4096;
    bool open{}, pending{};
    std::uint64_t submitted{}, completed{};
    cudaError_t last_error{cudaSuccess};
    std::recursive_mutex mutex;

    void initialize_cache() {
        std::vector<std::byte> bytes;
        if (const char *directory = std::getenv("PM_VULKAN_CACHE_DIR")) {
            try {
                std::string name;
                constexpr char hex[] = "0123456789abcdef";
                for (auto byte : adapter.properties.pipelineCacheUUID) {
                    name += hex[byte >> 4];
                    name += hex[byte & 15];
                }
                cache_file = std::filesystem::path(directory) / (name + ".bin");
                std::ifstream input(cache_file, std::ios::binary | std::ios::ate);
                auto size = input.tellg();
                if (size >= std::streamoff(sizeof(VkPipelineCacheHeaderVersionOne)) &&
                    size <= 64 * 1024 * 1024) {
                    bytes.resize(static_cast<std::size_t>(size));
                    input.seekg(0);
                    input.read(reinterpret_cast<char *>(bytes.data()), size);
                    VkPipelineCacheHeaderVersionOne header{};
                    std::memcpy(&header, bytes.data(), sizeof header);
                    if (!input || header.headerSize != sizeof header ||
                        header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
                        header.vendorID != adapter.properties.vendorID ||
                        header.deviceID != adapter.properties.deviceID ||
                        std::memcmp(header.pipelineCacheUUID,
                                    adapter.properties.pipelineCacheUUID, VK_UUID_SIZE))
                        bytes.clear();
                }
            } catch (...) {
                bytes.clear();
            }
        }
        VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        info.initialDataSize = bytes.size();
        info.pInitialData = bytes.data();
        auto result = vkCreatePipelineCache(device, &info, nullptr, &pipeline_cache);
        if (result != VK_SUCCESS && !bytes.empty()) {
            info.initialDataSize = 0;
            info.pInitialData = nullptr;
            result = vkCreatePipelineCache(device, &info, nullptr, &pipeline_cache);
        }
        check(result, "create pipeline cache");
    }
    void save_cache() noexcept {
        if (cache_file.empty() || pipelines.empty() || !pipeline_cache)
            return;
        try {
            std::size_t size{};
            if (vkGetPipelineCacheData(device, pipeline_cache, &size, nullptr) !=
                    VK_SUCCESS ||
                size > 64 * 1024 * 1024)
                return;
            std::vector<std::byte> bytes(size);
            if (vkGetPipelineCacheData(device, pipeline_cache, &size, bytes.data()) !=
                VK_SUCCESS)
                return;
            std::filesystem::create_directories(cache_file.parent_path());
            auto temporary = cache_file.string() + "." +
                             std::to_string(reinterpret_cast<std::uintptr_t>(this)) +
                             ".tmp";
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char *>(bytes.data()), size);
            output.close();
            if (output)
                std::filesystem::rename(temporary, cache_file);
            else
                std::filesystem::remove(temporary);
        } catch (...) { /* An optional cache must never affect correctness. */
        }
    }

    ~Impl() {
        if (!device)
            return;
        vkDeviceWaitIdle(device);
        save_cache();
        for (auto &[key, p] : pipelines)
            vkDestroyPipeline(device, p.pipeline, nullptr);
        if (pipeline_cache)
            vkDestroyPipelineCache(device, pipeline_cache, nullptr);
        for (auto &b : temporary)
            destroy(b);
        for (auto &b : allocations)
            destroy(b);
        destroy(globals);
        destroy(arguments);
        for (auto layout : layouts)
            if (layout)
                vkDestroyPipelineLayout(device, layout, nullptr);
        if (descriptor_pool)
            vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        for (auto descriptor_layout : descriptors)
            if (descriptor_layout)
                vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
        if (pool)
            vkDestroyCommandPool(device, pool, nullptr);
        if (fence)
            vkDestroyFence(device, fence, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    void destroy(Buffer &b) {
        if (b.host)
            vkUnmapMemory(device, b.memory);
        if (b.buffer)
            vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory)
            vkFreeMemory(device, b.memory, nullptr);
        b = {};
    }
    Buffer make_buffer(std::size_t size) {
        Buffer b{};
        b.size = size;
        try {
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size = (std::max<std::size_t>(size, 4) + 3) & ~std::size_t(3);
            info.usage =
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            check(vkCreateBuffer(device, &info, nullptr, &b.buffer), "create buffer");
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device, b.buffer, &requirements);
            std::uint32_t type = UINT32_MAX;
            for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
                auto flags = memory.memoryTypes[i].propertyFlags;
                if ((requirements.memoryTypeBits & (1U << i)) &&
                    (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    type = i;
                    if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                        break;
                }
            }
            if (type == UINT32_MAX)
                throw std::runtime_error("no coherent host memory");
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
            allocate.allocationSize = requirements.size;
            allocate.memoryTypeIndex = type;
            check(vkAllocateMemory(device, &allocate, nullptr, &b.memory),
                  "allocate memory");
            if (std::getenv("PM_VULKAN_TRACE"))
                std::fprintf(stderr, "Parallel Mater Vulkan allocation: %llu bytes\n",
                             static_cast<unsigned long long>(requirements.size));
            check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "bind memory");
            check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.host),
                  "map memory");
            VkBufferDeviceAddressInfo address{
                VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            address.buffer = b.buffer;
            b.address = vkGetBufferDeviceAddress(device, &address);
            std::memset(b.host, 0, size);
            return b;
        } catch (...) {
            destroy(b);
            throw;
        }
    }
    Buffer *find(const void *p, std::size_t size = 0) {
        const auto value = reinterpret_cast<std::uintptr_t>(p);
        for (auto &b : allocations) {
            const auto start = reinterpret_cast<std::uintptr_t>(b.host);
            if (value >= start && value - start < b.size &&
                size <= b.size - (value - start))
                return &b;
        }
        return nullptr;
    }
    void wait_pending() {
        if (!pending)
            return;
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "wait fence");
        completed = submitted;
        pending = false;
    }
    void barrier() {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT |
                                VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(
            commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
            &barrier, 0, nullptr, 0, nullptr);
    }
    void begin() {
        if (open)
            return;
        wait_pending();
        check(vkResetCommandPool(device, pool, 0), "reset command pool");
        for (auto &b : temporary)
            destroy(b);
        temporary.clear();
        meshes.clear();
        paint_fields.clear();
        global_offset = 0;
        argument_offset = 0;
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(commands, &info), "begin commands");
        open = true;
        barrier();
    }
    void submit() {
        if (!open)
            return;
        barrier();
        check(vkEndCommandBuffer(commands), "end commands");
        check(vkResetFences(device, 1, &fence), "reset fence");
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        info.commandBufferCount = 1;
        info.pCommandBuffers = &commands;
        check(vkQueueSubmit(queue, 1, &info, fence), "submit");
        open = false;
        pending = true;
        ++submitted;
    }
    void sync() {
        submit();
        wait_pending();
    }
};

ScheduleContext::ScheduleContext() : impl_(std::make_unique<Impl>()) {}
ScheduleContext::~ScheduleContext() {
    if (active_context == this)
        active_context = nullptr;
}
ScheduleContext *context(cudaStream_t stream) noexcept {
    return stream ? static_cast<ScheduleContext *>(stream) : active_context;
}
void ScheduleContext::activate() noexcept { active_context = this; }
PmResult ScheduleContext::initialize(int ordinal) noexcept {
    try {
        auto available = adapters(impl_->instance.handle);
        if (ordinal < 0 || std::size_t(ordinal) >= available.size())
            return PM_ERROR_INVALID_ARGUMENT;
        auto &i = *impl_;
        i.adapter = available[ordinal];
        i.ordinal = ordinal;
        vkGetPhysicalDeviceMemoryProperties(i.adapter.handle, &i.memory);
        float priority = 1;
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex = i.adapter.family;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        VkPhysicalDeviceVulkan12Features f12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.bufferDeviceAddress = VK_TRUE;
        f12.scalarBlockLayout = VK_TRUE;
        f12.storageBuffer8BitAccess = VK_TRUE;
        f12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
        f12.storagePushConstant8 = VK_TRUE;
        f12.shaderInt8 = VK_TRUE;
        f12.shaderBufferInt64Atomics = VK_TRUE;
        VkPhysicalDeviceFeatures features{};
        features.shaderInt64 = VK_TRUE;
        VkPhysicalDeviceVulkan13Features f13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.maintenance4 = VK_TRUE;
        f12.pNext = &f13;
        VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12};
        create.queueCreateInfoCount = 1;
        create.pQueueCreateInfos = &queue;
        create.pEnabledFeatures = &features;
        const auto extensions = available_device_extensions(i.adapter.handle);
        create.enabledExtensionCount =
            static_cast<std::uint32_t>(extensions.size());
        create.ppEnabledExtensionNames = extensions.data();
        check(vkCreateDevice(i.adapter.handle, &create, nullptr, &i.device),
              "create device");
        i.initialize_cache();
        vkGetDeviceQueue(i.device, i.adapter.family, 0, &i.queue);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = i.adapter.family;
        check(vkCreateCommandPool(i.device, &pool, nullptr, &i.pool),
              "create command pool");
        VkCommandBufferAllocateInfo commands{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        commands.commandPool = i.pool;
        commands.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commands.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(i.device, &commands, &i.commands),
              "allocate command buffer");
        VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(i.device, &fence, nullptr, &i.fence), "create fence");
        const VkDescriptorSetLayoutBinding global_binding{
            0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
            VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo descriptor_layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptor_layout.bindingCount = 1;
        descriptor_layout.pBindings = &global_binding;
        check(vkCreateDescriptorSetLayout(i.device, &descriptor_layout, nullptr,
                                          &i.descriptors[0]),
              "create push descriptor layout");
        const VkDescriptorSetLayoutBinding parameter_bindings[]{
            global_binding,
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
             VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        descriptor_layout.bindingCount = std::size(parameter_bindings);
        descriptor_layout.pBindings = parameter_bindings;
        check(vkCreateDescriptorSetLayout(i.device, &descriptor_layout, nullptr,
                                          &i.descriptors[1]),
              "create parameter descriptor layout");
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 3};
        VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dp.maxSets = 2;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &pool_size;
        check(vkCreateDescriptorPool(i.device, &dp, nullptr, &i.descriptor_pool),
              "create descriptor pool");
        VkDescriptorSetAllocateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set.descriptorPool = i.descriptor_pool;
        set.descriptorSetCount = std::size(i.descriptors);
        set.pSetLayouts = i.descriptors;
        check(vkAllocateDescriptorSets(i.device, &set, i.descriptor_sets),
              "allocate descriptors");
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 256};
        for (std::uint32_t n = 0; n < std::size(i.layouts); ++n) {
            VkPipelineLayoutCreateInfo layout{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout.setLayoutCount = 1;
            layout.pSetLayouts = &i.descriptors[n];
            if (n == 0) {
                layout.pushConstantRangeCount = 1;
                layout.pPushConstantRanges = &range;
            }
            check(vkCreatePipelineLayout(i.device, &layout, nullptr, &i.layouts[n]),
                  "create pipeline layout");
        }
        i.global_stride = std::max(
            16U,
            std::uint32_t(i.adapter.properties.limits.minUniformBufferOffsetAlignment));
        i.parameter_stride = i.global_stride;
        i.globals = i.make_buffer(i.global_stride * 65536ULL);
        i.arguments = i.make_buffer(1024U * 1024U);
        const VkDescriptorBufferInfo buffers[]{
            {i.globals.buffer, 0, 16},
            {i.globals.buffer, 0, 16},
            {i.arguments.buffer, 0, Impl::parameter_range}};
        VkWriteDescriptorSet writes[3]{};
        for (std::uint32_t n = 0; n < std::size(writes); ++n) {
            writes[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[n].dstSet = i.descriptor_sets[n == 0 ? 0 : 1];
            writes[n].dstBinding = n == 2 ? 1 : 0;
            writes[n].descriptorCount = 1;
            writes[n].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            writes[n].pBufferInfo = &buffers[n];
        }
        vkUpdateDescriptorSets(i.device, std::size(writes), writes, 0, nullptr);
        activate();
        return PM_SUCCESS;
    } catch (...) {
        return PM_ERROR_INVALID_STATE;
    }
}
PmResult ScheduleContext::allocate(std::size_t size, void **output) noexcept {
    if (!output || !size)
        return PM_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    try {
        std::lock_guard lock(impl_->mutex);
        auto b = impl_->make_buffer(size);
        try {
            impl_->allocations.push_back(b);
        } catch (...) {
            impl_->destroy(b);
            throw;
        }
        *output = b.host;
        return PM_SUCCESS;
    } catch (...) {
        return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    }
}
void ScheduleContext::free(void *allocation) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (synchronize() != cudaSuccess)
        return;
    auto &a = impl_->allocations;
    auto found = std::find_if(a.begin(), a.end(),
                              [&](const auto &b) { return b.host == allocation; });
    if (found != a.end()) {
        impl_->destroy(*found);
        a.erase(found);
    }
    impl_->mesh_capacities.clear();
}
std::uint64_t ScheduleContext::address(const void *pointer) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (!pointer)
        return 0;
    auto b = impl_->find(pointer);
    if (!b) {
        const auto value = reinterpret_cast<std::uintptr_t>(pointer);
        const auto start = reinterpret_cast<std::uintptr_t>(impl_->arguments.host);
        if (value >= start && value - start < impl_->arguments.size)
            b = &impl_->arguments;
    }
    if (!b) {
        if (std::getenv("PM_VULKAN_TRACE"))
            std::fprintf(stderr,
                         "Parallel Mater Vulkan: cannot translate address %p\n",
                         pointer);
        impl_->last_error = cudaErrorInvalidValue;
        return 0;
    }
    return b->address + (reinterpret_cast<std::uintptr_t>(pointer) -
                         reinterpret_cast<std::uintptr_t>(b->host));
}
void *ScheduleContext::stage(std::span<const std::byte> bytes) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (bytes.empty() || bytes.size() > impl_->arguments.size) {
        impl_->last_error = cudaErrorInvalidValue;
        return nullptr;
    }
    try {
        auto &i = *impl_;
        i.begin();
        const auto size = (bytes.size() + 15U) & ~std::size_t(15U);
        if (i.argument_offset + size > i.arguments.size) {
            i.submit();
            i.begin();
        }
        auto *destination = static_cast<std::byte *>(i.arguments.host) +
                            i.argument_offset;
        std::memcpy(destination, bytes.data(), bytes.size());
        i.argument_offset += static_cast<std::uint32_t>(size);
        return destination;
    } catch (...) {
        impl_->last_error = cudaErrorLaunchFailure;
        return nullptr;
    }
}
std::uint64_t ScheduleContext::translated_meshes(const void *pointer,
                                                 const void *parameters,
                                                 std::uint32_t count,
                                                 std::uint32_t capacity,
                                                 const void *paint_fields,
                                                 std::uint32_t paint_field_capacity) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (!pointer)
        return 0;
    try {
        auto &i = *impl_;
        i.begin();
        if (count == 0 && capacity == 0) {
            auto prepared = i.mesh_capacities.find(pointer);
            if (prepared != i.mesh_capacities.end())
                capacity = prepared->second;
        }
        auto *bodies = static_cast<const simulation::BodyParameters *>(parameters);
        if (count && (!bodies || !i.find(bodies, std::size_t(count) * sizeof(*bodies))))
            throw std::runtime_error("mesh body range");
        for (std::uint32_t n = 0; n < count; ++n)
            capacity = std::max(capacity, bodies[n].mesh.index + 1);
        using Field = simulation::PaintFieldResource;
        const auto *fields = static_cast<const Field *>(paint_fields);
        if (paint_field_capacity &&
            (!fields || !i.find(fields, std::size_t(paint_field_capacity) *
                                     sizeof(*fields))))
            throw std::runtime_error("mesh paint field range");
        for (std::uint32_t n = 0; n < paint_field_capacity; ++n)
            if (fields[n].alive && fields[n].options.mesh.generation != 0U)
                capacity = std::max(capacity,
                                    fields[n].options.mesh.index + 1U);
        capacity = std::max(1U, capacity);
        i.mesh_capacities[pointer] = capacity;
        auto found = i.meshes.find(pointer);
        if (found != i.meshes.end() && found->second.first >= capacity)
            return found->second.second;
        using Mesh = simulation::TriangleMeshResource;
        if (!i.find(pointer, std::size_t(capacity) * sizeof(Mesh)))
            throw std::runtime_error("mesh range");
        auto b = i.make_buffer(std::size_t(capacity) * sizeof(Mesh));
        try {
            i.temporary.push_back(b);
        } catch (...) {
            i.destroy(b);
            throw;
        }
        auto *target = static_cast<Mesh *>(b.host);
        std::memcpy(target, pointer, b.size);
        for (std::uint32_t n = 0; n < capacity; ++n) {
            auto &m = target[n];
            m.vertices = reinterpret_cast<decltype(m.vertices)>(address(m.vertices));
            m.indices = reinterpret_cast<decltype(m.indices)>(address(m.indices));
            m.bvh_nodes = reinterpret_cast<decltype(m.bvh_nodes)>(address(m.bvh_nodes));
            m.bvh_leaves =
                reinterpret_cast<decltype(m.bvh_leaves)>(address(m.bvh_leaves));
            m.solid_planes =
                reinterpret_cast<decltype(m.solid_planes)>(address(m.solid_planes));
            m.shell_normals =
                reinterpret_cast<decltype(m.shell_normals)>(address(m.shell_normals));
        }
        i.meshes[pointer] = {capacity, b.address};
        return b.address;
    } catch (...) {
        impl_->last_error = cudaErrorInvalidValue;
        return 0;
    }
}
std::uint64_t ScheduleContext::translated_paint_fields(
    const void *pointer, std::uint32_t capacity) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (!pointer || capacity == 0U)
        return 0;
    try {
        auto &i = *impl_;
        i.begin();
        auto found = i.paint_fields.find(pointer);
        if (found != i.paint_fields.end() && found->second.first >= capacity)
            return found->second.second;
        using Field = simulation::PaintFieldResource;
        if (!i.find(pointer, std::size_t(capacity) * sizeof(Field)))
            throw std::runtime_error("paint field range");
        auto b = i.make_buffer(std::size_t(capacity) * sizeof(Field));
        try {
            i.temporary.push_back(b);
        } catch (...) {
            i.destroy(b);
            throw;
        }
        auto *target = static_cast<Field *>(b.host);
        std::memcpy(target, pointer, b.size);
        for (std::uint32_t n = 0; n < capacity; ++n) {
            auto &field = target[n];
            field.options.vertex_uvs.data =
                reinterpret_cast<decltype(field.options.vertex_uvs.data)>(
                    address(field.options.vertex_uvs.data));
            field.uvs = reinterpret_cast<decltype(field.uvs)>(address(field.uvs));
            field.pixels =
                reinterpret_cast<decltype(field.pixels)>(address(field.pixels));
        }
        i.paint_fields[pointer] = {capacity, b.address};
        return b.address;
    } catch (...) {
        impl_->last_error = cudaErrorInvalidValue;
        return 0;
    }
}
void ScheduleContext::dispatch(std::string_view module, std::string_view entry,
                               std::span<const std::uint32_t> shader, dim3 grid,
                               dim3 block, dim3 shader_block,
                               std::span<const std::byte> constants,
                               bool parameter_buffer) noexcept {
    activate();
    std::lock_guard lock(impl_->mutex);
    auto &i = *impl_;
    if (i.last_error != cudaSuccess) {
        if (std::getenv("PM_VULKAN_TRACE"))
            std::fprintf(stderr,
                         "Parallel Mater Vulkan: skipped %.*s/%.*s after error %d\n",
                         int(module.size()), module.data(), int(entry.size()),
                         entry.data(), static_cast<int>(i.last_error));
        return;
    }
    try {
        if (constants.size() > (parameter_buffer ? Impl::parameter_range : 256U) ||
            (shader_block.x != 0 && block.x != shader_block.x) ||
            block.y != shader_block.y || block.z != shader_block.z)
            throw std::runtime_error("dispatch shape or parameter-buffer mismatch");
        i.begin();
        const std::string key = std::string(module) + '/' + std::string(entry) + '/' +
                                std::to_string(block.x) +
                                (parameter_buffer ? "/buffer" : "/push");
        const std::uint32_t layout_index = parameter_buffer ? 1U : 0U;
        auto p = i.pipelines.find(key);
        if (p == i.pipelines.end()) {
            if (std::getenv("PM_VULKAN_TRACE"))
                std::fprintf(stderr, "Parallel Mater Vulkan pipeline: %s (%zu bytes; grid=%u,%u,%u)\n",
                             key.c_str(), shader.size_bytes(), grid.x, grid.y, grid.z);
            VkShaderModuleCreateInfo source{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            source.codeSize = shader.size_bytes();
            source.pCode = shader.data();
            VkShaderModule handle{};
            check(vkCreateShaderModule(i.device, &source, nullptr, &handle),
                  "create shader module");
            VkComputePipelineCreateInfo create{
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            create.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            create.stage.module = handle;
            std::string name(entry);
            create.stage.pName = name.c_str();
            create.layout = i.layouts[layout_index];
            VkSpecializationMapEntry specialization_entry{0, 0, sizeof(block.x)};
            VkSpecializationInfo specialization{1, &specialization_entry, sizeof(block.x),
                                                &block.x};
            create.stage.pSpecializationInfo = &specialization;
            VkPipeline pipeline{};
            const auto result = vkCreateComputePipelines(i.device, i.pipeline_cache, 1,
                                                         &create, nullptr, &pipeline);
            vkDestroyShaderModule(i.device, handle, nullptr);
            check(result, "create compute pipeline");
            try {
                p = i.pipelines.emplace(key, Impl::Pipeline{pipeline}).first;
            } catch (...) {
                vkDestroyPipeline(i.device, pipeline, nullptr);
                throw;
            }
        }
        std::uint32_t parameter_offset{};
        if (parameter_buffer) {
            i.argument_offset = (i.argument_offset + i.parameter_stride - 1U) &
                                ~(i.parameter_stride - 1U);
            if (i.argument_offset + Impl::parameter_range > i.arguments.size) {
                i.submit();
                i.begin();
            }
            parameter_offset = i.argument_offset;
            std::memcpy(static_cast<std::byte *>(i.arguments.host) + parameter_offset,
                        constants.data(), constants.size());
            i.argument_offset += static_cast<std::uint32_t>(constants.size());
        }
        const std::uint32_t globals[]{grid.x, grid.y, grid.z, block.x};
        std::memcpy(static_cast<std::byte *>(i.globals.host) + i.global_offset, globals,
                    sizeof globals);
        vkCmdBindPipeline(i.commands, VK_PIPELINE_BIND_POINT_COMPUTE, p->second.pipeline);
        const std::uint32_t offsets[]{i.global_offset, parameter_offset};
        const auto dynamic_offset_count = parameter_buffer ? 2U : 1U;
        vkCmdBindDescriptorSets(i.commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                i.layouts[layout_index], 0, 1,
                                &i.descriptor_sets[layout_index],
                                dynamic_offset_count, offsets);
        if (!parameter_buffer)
            vkCmdPushConstants(i.commands, i.layouts[layout_index],
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               std::uint32_t(constants.size()), constants.data());
        vkCmdDispatch(i.commands, grid.x, grid.y, grid.z);
        i.barrier();
        // Diagnostic mode: isolate a failing dispatch without changing normal
        // asynchronous submission behavior.
        if (std::getenv("PM_VULKAN_SYNCHRONOUS_DISPATCH"))
            i.sync();
        i.global_offset += i.global_stride;
        // Flush after recording: constants may contain a translated mesh address
        // prepared immediately before dispatch. Resetting the pool beforehand
        // would free that temporary while the pending dispatch still needs it.
        if (i.global_offset + i.global_stride > i.globals.size)
            i.submit();
    } catch (const std::exception &e) {
        std::fprintf(
            stderr, "Parallel Mater Vulkan: %.*s/%.*s: %s (block %u/%u; constants %zu)\n",
            int(module.size()), module.data(), int(entry.size()), entry.data(), e.what(),
            block.x, shader_block.x, constants.size());
        i.last_error = cudaErrorLaunchFailure;
    }
}
cudaError_t ScheduleContext::copy(void *dst, const void *src, std::size_t size,
                                  cudaMemcpyKind kind) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (!size)
        return cudaSuccess;
    if (!dst || !src)
        return cudaErrorInvalidValue;
    try {
        auto &i = *impl_;
        auto d = i.find(dst, size);
        auto s = i.find(src, size);
        if (kind == cudaMemcpyDeviceToDevice || (kind == cudaMemcpyDefault && d && s)) {
            if (!d || !s)
                return i.last_error = cudaErrorInvalidValue;
            i.begin();
            VkBufferCopy region{std::uint64_t(static_cast<const std::byte *>(src) -
                                              static_cast<std::byte *>(s->host)),
                                std::uint64_t(static_cast<std::byte *>(dst) -
                                              static_cast<std::byte *>(d->host)),
                                size};
            vkCmdCopyBuffer(i.commands, s->buffer, d->buffer, 1, &region);
            i.barrier();
        } else {
            i.sync();
            std::memmove(dst, src, size);
        }
        return cudaSuccess;
    } catch (...) {
        return impl_->last_error = cudaErrorLaunchFailure;
    }
}
cudaError_t ScheduleContext::clear(void *dst, int value, std::size_t size) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (!size)
        return cudaSuccess;
    try {
        auto &i = *impl_;
        auto b = i.find(dst, size);
        if (!b)
            return i.last_error = cudaErrorInvalidValue;
        auto offset = std::uint64_t(static_cast<std::byte *>(dst) -
                                    static_cast<std::byte *>(b->host));
        if (offset % 4 == 0 && size % 4 == 0) {
            i.begin();
            auto byte = std::uint32_t(value) & 255U;
            vkCmdFillBuffer(i.commands, b->buffer, offset, size, byte * 0x01010101U);
            i.barrier();
        } else {
            i.sync();
            std::memset(dst, value, size);
        }
        return cudaSuccess;
    } catch (...) {
        return impl_->last_error = cudaErrorLaunchFailure;
    }
}
cudaError_t ScheduleContext::synchronize() noexcept {
    try {
        std::lock_guard lock(impl_->mutex);
        impl_->sync();
        return impl_->last_error;
    } catch (...) {
        return impl_->last_error = cudaErrorLaunchFailure;
    }
}
cudaError_t ScheduleContext::record(cudaEvent_t event) noexcept {
    if (!event)
        return cudaErrorInvalidValue;
    try {
        std::lock_guard lock(impl_->mutex);
        if (event->stream && event->stream != this)
            return cudaErrorInvalidResourceHandle;
        if (event->timing) {
            impl_->begin();
            VkQueryPool pool = reinterpret_cast<VkQueryPool>(event->query_pool);
            if (!pool) {
                VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                info.queryType = VK_QUERY_TYPE_TIMESTAMP;
                info.queryCount = 1;
                check(vkCreateQueryPool(impl_->device, &info, nullptr, &pool),
                      "create timestamp query");
                event->query_pool = reinterpret_cast<std::uint64_t>(pool);
            }
            vkCmdResetQueryPool(impl_->commands, pool, 0, 1);
            vkCmdWriteTimestamp(impl_->commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                pool, 0);
        }
        impl_->submit();
        event->stream = this;
        event->fence_value = impl_->submitted;
        event->ready = false;
        event->recorded = std::chrono::steady_clock::now();
        return impl_->last_error;
    } catch (...) {
        return impl_->last_error = cudaErrorLaunchFailure;
    }
}
cudaError_t ScheduleContext::wait(cudaEvent_t event) noexcept {
    if (!event || event->stream != this)
        return cudaErrorInvalidResourceHandle;
    return synchronize();
}
cudaError_t ScheduleContext::query(cudaEvent_t event) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (!event || event->stream != this)
        return cudaErrorInvalidResourceHandle;
    if (event->fence_value <= impl_->completed)
        return cudaSuccess;
    auto result = vkGetFenceStatus(impl_->device, impl_->fence);
    if (result == VK_SUCCESS) {
        impl_->completed = impl_->submitted;
        return cudaSuccess;
    }
    return result == VK_NOT_READY ? cudaErrorNotReady : cudaErrorLaunchFailure;
}
cudaError_t ScheduleContext::error(bool take) noexcept {
    std::lock_guard lock(impl_->mutex);
    auto e = impl_->last_error;
    if (take)
        impl_->last_error = cudaSuccess;
    return e;
}
cudaError_t ScheduleContext::elapsed(cudaEvent_t begin, cudaEvent_t end,
                                     float *ms) noexcept {
    std::lock_guard lock(impl_->mutex);
    if (!ms || !begin || !end || begin->stream != this || end->stream != this ||
        !begin->query_pool || !end->query_pool)
        return cudaErrorInvalidValue;
    std::uint64_t ticks[2]{};
    const cudaEvent_t events[]{begin, end};
    for (unsigned n = 0; n < 2; ++n) {
        auto result = vkGetQueryPoolResults(
            impl_->device, reinterpret_cast<VkQueryPool>(events[n]->query_pool), 0, 1,
            sizeof(ticks[n]), &ticks[n], sizeof(ticks[n]), VK_QUERY_RESULT_64_BIT);
        if (result != VK_SUCCESS)
            return result == VK_NOT_READY ? cudaErrorNotReady : cudaErrorLaunchFailure;
    }
    const auto bits = impl_->adapter.timestamp_bits;
    const std::uint64_t mask = bits == 64 ? UINT64_MAX : ((std::uint64_t(1) << bits) - 1);
    *ms = float(double((ticks[1] - ticks[0]) & mask) *
                impl_->adapter.properties.limits.timestampPeriod / 1.0e6);
    return cudaSuccess;
}
VkInstance ScheduleContext::instance() const noexcept { return impl_->instance.handle; }
VkPhysicalDevice ScheduleContext::physical() const noexcept {
    return impl_->adapter.handle;
}
int ScheduleContext::ordinal() const noexcept { return impl_->ordinal; }
VkDevice ScheduleContext::device() const noexcept { return impl_->device; }
VkQueue ScheduleContext::queue() const noexcept { return impl_->queue; }
std::uint32_t ScheduleContext::queue_family() const noexcept {
    return impl_->adapter.family;
}
VkBuffer ScheduleContext::buffer(const void *pointer) const noexcept {
    std::lock_guard lock(impl_->mutex);
    auto b = impl_->find(pointer);
    return b ? b->buffer : VK_NULL_HANDLE;
}
PmResult ScheduleContext::memory_budget(PmDeviceSize &available,
                                        PmDeviceSize &total) noexcept {
    std::lock_guard lock(impl_->mutex);
    total = 0;
    for (std::uint32_t n = 0; n < impl_->memory.memoryHeapCount; ++n)
        if (impl_->memory.memoryHeaps[n].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total += impl_->memory.memoryHeaps[n].size;
    PmDeviceSize used{};
    for (auto &b : impl_->allocations)
        used += b.size;
    available = total > used ? total - used : 0;
    return PM_SUCCESS;
}
std::vector<device::Description> enumerate_devices() {
    Instance instance;
    auto list = adapters(instance.handle);
    std::vector<device::Description> result;
    for (std::size_t n = 0; n < list.size(); ++n) {
        device::Description d{};
        auto &a = list[n];
        d.ordinal = int(n);
        d.name = a.properties.deviceName;
        d.vendor_id = a.properties.vendorID;
        d.device_id = a.properties.deviceID;
        d.driver_version = a.properties.driverVersion;
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                      &id};
        vkGetPhysicalDeviceProperties2(a.handle, &p);
        std::memcpy(d.uuid, id.deviceUUID, sizeof d.uuid);
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(a.handle, &memory);
        for (std::uint32_t h = 0; h < memory.memoryHeapCount; ++h)
            if (memory.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                d.total_memory += memory.memoryHeaps[h].size;
        result.push_back(d);
    }
    return result;
}
} // namespace parallel_mater::vulkan

using parallel_mater::vulkan::context;
cudaError_t cudaMalloc(void **p, std::size_t size) noexcept {
    auto c = context();
    return c && c->allocate(size, p) == PM_SUCCESS ? cudaSuccess
                                                   : cudaErrorMemoryAllocation;
}
cudaError_t cudaFree(void *p) noexcept {
    auto c = context();
    if (!c)
        return cudaErrorInvalidResourceHandle;
    c->free(p);
    return cudaSuccess;
}
cudaError_t cudaMemcpy(void *d, const void *s, std::size_t n, cudaMemcpyKind k) noexcept {
    return pmVulkanMemcpyAsync(d, s, n, k, nullptr);
}
cudaError_t cudaMemset(void *d, int v, std::size_t n) noexcept {
    return pmVulkanMemsetAsync(d, v, n, nullptr);
}
cudaError_t pmVulkanMemcpyAsync(void *d, const void *s, std::size_t n, cudaMemcpyKind k,
                                cudaStream_t stream) noexcept {
    auto c = context(stream);
    return c ? c->copy(d, s, n, k) : cudaErrorInvalidResourceHandle;
}
cudaError_t pmVulkanMemsetAsync(void *d, int v, std::size_t n,
                                cudaStream_t stream) noexcept {
    auto c = context(stream);
    return c ? c->clear(d, v, n) : cudaErrorInvalidResourceHandle;
}
cudaError_t pmVulkanStreamSynchronize(cudaStream_t stream) noexcept {
    auto c = context(stream);
    return c ? c->synchronize() : cudaErrorInvalidResourceHandle;
}
cudaError_t pmVulkanRecordEvent(cudaEvent_t e, cudaStream_t stream) noexcept {
    auto c = context(stream);
    return c ? c->record(e) : cudaErrorInvalidResourceHandle;
}
cudaError_t pmVulkanWaitEvent(cudaEvent_t e) noexcept {
    auto c = e ? context(e->stream) : nullptr;
    return c ? c->wait(e) : cudaErrorInvalidResourceHandle;
}
cudaError_t pmVulkanQueryEvent(cudaEvent_t e) noexcept {
    auto c = e ? context(e->stream) : nullptr;
    return c ? c->query(e) : cudaErrorInvalidResourceHandle;
}
cudaError_t pmVulkanPeekError(cudaStream_t stream, bool take) noexcept {
    auto c = context(stream);
    return c ? c->error(take) : cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t e) noexcept {
    if (!e)
        return cudaSuccess;
    auto c = e->stream ? context(e->stream) : nullptr;
    if (c && e->query_pool) {
        (void)c->wait(e);
        vkDestroyQueryPool(c->device(), reinterpret_cast<VkQueryPool>(e->query_pool),
                           nullptr);
    }
    delete e;
    return cudaSuccess;
}
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t begin, cudaEvent_t end) noexcept {
    auto c = begin ? context(begin->stream) : nullptr;
    return c ? c->elapsed(begin, end, ms) : cudaErrorInvalidResourceHandle;
}

cudaError_t cudaGetDevice(int *device) noexcept {
    auto c = context();
    if (!device || !c) return cudaErrorInvalidValue;
    *device = c->ordinal();
    return cudaSuccess;
}
