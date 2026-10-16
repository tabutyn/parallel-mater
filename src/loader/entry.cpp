// SPDX-License-Identifier: MIT
#include "internal.hpp"
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
#include <parallel_mater/interop_vulkan.h>
#endif

#include <algorithm>
#include <cstring>
#include <new>

namespace {

template <typename T>
PmResult enumerate_values(const std::vector<T> &available, std::uint32_t *count,
                          T *values) {
    if (count == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    if (values == nullptr) {
        *count = static_cast<std::uint32_t>(available.size());
        return PM_SUCCESS;
    }
    const std::uint32_t written = std::min(
        *count, static_cast<std::uint32_t>(available.size()));
    std::copy_n(available.begin(), written, values);
    *count = written;
    return written == available.size() ? PM_SUCCESS : PM_INCOMPLETE;
}

bool validation_layer_name(const char *name) {
    // Keep the original 0.2 spelling as an unadvertised compatibility alias.
    return name != nullptr &&
        (std::strcmp(name, "PM_LAYER_PARALLEL_MATER_validation") == 0 ||
         std::strcmp(name, "PM_LAYER_KHRONOS_validation") == 0);
}

void copy_name(char (&destination)[256], const char *source) {
    std::strncpy(destination, source, sizeof destination - 1U);
    destination[sizeof destination - 1U] = '\0';
}

} // namespace

namespace parallel_mater::loader {

PFN_pmVoidFunction lookup_loader_proc(const char *name) noexcept {
    if (name == nullptr) return nullptr;
#define PM_LOADER_LOOKUP(command)                                             \
    if (std::strcmp(name, #command) == 0)                                    \
        return reinterpret_cast<PFN_pmVoidFunction>(&command)
#include "loader_lookup.inc"
#undef PM_LOADER_LOOKUP
    return nullptr;
}

} // namespace parallel_mater::loader

extern "C" PMAPI_ATTR PFN_pmVoidFunction PMAPI_CALL pmGetInstanceProcAddr(
    PmInstance instance, const char *name) {
    if (name == nullptr) return nullptr;
    if (instance == nullptr) {
        if (std::strcmp(name, "pmGetInstanceProcAddr") != 0 &&
            std::strcmp(name, "pmEnumerateInstanceVersion") != 0 &&
            std::strcmp(name, "pmEnumerateInstanceLayerProperties") != 0 &&
            std::strcmp(name, "pmEnumerateInstanceExtensionProperties") != 0 &&
            std::strcmp(name, "pmCreateInstance") != 0)
            return nullptr;
    } else if (instance->dispatch != &parallel_mater::loader::loader_dispatch) {
        return nullptr;
    }
    return parallel_mater::loader::lookup_loader_proc(name);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL
pmEnumerateInstanceLayerProperties(std::uint32_t *count,
                                   PmLayerProperties *properties) {
    PmLayerProperties validation{};
    copy_name(validation.layerName, "PM_LAYER_PARALLEL_MATER_validation");
    validation.specificationVersion = PM_API_VERSION_0_2;
    validation.implementationVersion = 1U;
    copy_name(validation.description, "Parallel Mater validation layer");
    return enumerate_values(std::vector<PmLayerProperties>{validation}, count,
                            properties);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL
pmEnumerateInstanceExtensionProperties(const char *layer_name,
                                       std::uint32_t *count,
                                       PmExtensionProperties *properties) {
    if (layer_name != nullptr &&
        !validation_layer_name(layer_name))
        return PM_ERROR_LAYER_NOT_PRESENT;
    return enumerate_values(std::vector<PmExtensionProperties>{}, count,
                            properties);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL
pmEnumerateDeviceExtensionProperties(PmPhysicalDevice physical,
                                     const char *layer_name,
                                     std::uint32_t *count,
                                     PmExtensionProperties *properties) {
    if (!parallel_mater::loader::route(physical))
        return PM_ERROR_INVALID_ARGUMENT;
    if (layer_name != nullptr) return PM_ERROR_LAYER_NOT_PRESENT;
    const auto route = parallel_mater::loader::route(physical);
    auto enumerate = reinterpret_cast<PFN_pmEnumerateDeviceExtensionProperties>(
        route.driver->resolve(route, "pmEnumerateDeviceExtensionProperties"));
    return enumerate == nullptr
        ? enumerate_values(std::vector<PmExtensionProperties>{}, count, properties)
        : enumerate(reinterpret_cast<PmPhysicalDevice>(route.native), nullptr,
                    count, properties);
}

extern "C" PMAPI_ATTR PFN_pmVoidFunction PMAPI_CALL pmGetDeviceProcAddr(
    PmDevice device, const char *name) {
    const auto route = parallel_mater::loader::route(device);
    if (!route || name == nullptr ||
        route.driver->device_proc(reinterpret_cast<PmDevice>(route.context), name) ==
            nullptr)
        return nullptr;
    return parallel_mater::loader::lookup_loader_proc(name);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmEnumerateInstanceVersion(
    std::uint32_t *version) {
    if (version == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *version = PM_API_VERSION_0_2;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateInstance(
    const PmInstanceCreateInfo *info, const PmAllocationCallbacks *allocator,
    PmInstance *instance) {
    if (instance == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *instance = nullptr;
    if (info == nullptr || info->sType != PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO ||
        (info->enabledLayerCount != 0U && info->ppEnabledLayerNames == nullptr) ||
        (info->enabledExtensionCount != 0U &&
         info->ppEnabledExtensionNames == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    if (info->apiVersion != 0U && info->apiVersion != PM_API_VERSION_0_2)
        return PM_ERROR_INCOMPATIBLE_VERSION;
    bool validation_enabled = info->enableValidation == PM_TRUE;
    for (std::uint32_t index = 0U; index < info->enabledLayerCount; ++index) {
        const char *name = info->ppEnabledLayerNames[index];
        if (!validation_layer_name(name))
            return PM_ERROR_LAYER_NOT_PRESENT;
        validation_enabled = true;
    }
    for (std::uint32_t index = 0U; index < info->enabledExtensionCount; ++index) {
        if (info->ppEnabledExtensionNames[index] == nullptr)
            return PM_ERROR_INVALID_ARGUMENT;
        // No base instance extensions are currently registered.  Device
        // extensions are selected at device creation after driver discovery.
        return PM_ERROR_EXTENSION_NOT_PRESENT;
    }
    try {
        auto created = std::make_unique<PmInstance_T>();
        created->dispatch = &parallel_mater::loader::loader_dispatch;
        created->layers =
            parallel_mater::loader::load_validation_layers(validation_enabled);
        if (validation_enabled && created->layers->empty())
            return PM_ERROR_LAYER_NOT_PRESENT;
        PmInstanceCreateInfo driver_info = *info;
        driver_info.enableValidation = PM_FALSE;
        driver_info.enabledLayerCount = 0U;
        driver_info.ppEnabledLayerNames = nullptr;
        for (auto &driver : parallel_mater::loader::discover_drivers()) {
            auto create = reinterpret_cast<PFN_pmCreateInstance>(
                driver->instance_proc(nullptr, "pmCreateInstance"));
            if (create == nullptr) continue;
            PmInstance native{};
            const PmResult result = create(&driver_info, allocator, &native);
            if (result == PM_SUCCESS && native != nullptr)
                created->children.push_back({driver, native});
        }
        if (created->children.empty()) return PM_ERROR_DRIVER_NOT_FOUND;

        for (const auto &child : created->children) {
            auto enumerate = reinterpret_cast<PFN_pmEnumeratePhysicalDevices>(
                child.driver->instance_proc(child.native,
                                             "pmEnumeratePhysicalDevices"));
            if (enumerate == nullptr) continue;
            std::uint32_t count{};
            if (enumerate(child.native, &count, nullptr) != PM_SUCCESS || count == 0U)
                continue;
            std::vector<PmPhysicalDevice> native(count);
            std::uint32_t written = count;
            const PmResult result = enumerate(child.native, &written, native.data());
            if (result != PM_SUCCESS && result != PM_INCOMPLETE) continue;
            for (std::uint32_t index = 0U; index < written; ++index) {
                auto physical = std::make_unique<PmPhysicalDevice_T>();
                physical->dispatch = &parallel_mater::loader::loader_dispatch;
                physical->driver = child.driver;
                physical->native = native[index];
                physical->native_instance = child.native;
                physical->layers = created->layers;
                created->physical_devices.push_back(std::move(physical));
            }
        }
        *instance = created.release();
        return PM_SUCCESS;
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyInstance(PmInstance instance) {
    if (instance == nullptr ||
        instance->dispatch != &parallel_mater::loader::loader_dispatch)
        return;
    instance->physical_devices.clear();
    for (const auto &child : instance->children) {
        auto destroy = reinterpret_cast<PFN_pmDestroyInstance>(
            child.driver->instance_proc(child.native, "pmDestroyInstance"));
        if (destroy != nullptr) destroy(child.native);
    }
    delete instance;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmEnumeratePhysicalDevices(
    PmInstance instance, std::uint32_t *count, PmPhysicalDevice *devices) {
    if (instance == nullptr || count == nullptr ||
        instance->dispatch != &parallel_mater::loader::loader_dispatch)
        return PM_ERROR_INVALID_ARGUMENT;
    const std::uint32_t available =
        static_cast<std::uint32_t>(instance->physical_devices.size());
    if (devices == nullptr) {
        *count = available;
        return PM_SUCCESS;
    }
    const std::uint32_t written = std::min(*count, available);
    for (std::uint32_t index = 0U; index < written; ++index)
        devices[index] = instance->physical_devices[index].get();
    *count = written;
    return written == available ? PM_SUCCESS : PM_INCOMPLETE;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateDevice(
    PmPhysicalDevice physical, const PmDeviceCreateInfo *info, PmDevice *device) {
    if (device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *device = nullptr;
    const auto route = parallel_mater::loader::route(physical);
    if (!route) return PM_ERROR_INVALID_ARGUMENT;
    if (info == nullptr ||
        (info->enabledExtensionCount != 0U &&
         info->ppEnabledExtensionNames == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    std::uint32_t extension_count{};
    PmResult extension_result = pmEnumerateDeviceExtensionProperties(
        physical, nullptr, &extension_count, nullptr);
    if (extension_result != PM_SUCCESS) return extension_result;
    std::vector<PmExtensionProperties> extensions(extension_count);
    if (extension_count != 0U) {
        extension_result = pmEnumerateDeviceExtensionProperties(
            physical, nullptr, &extension_count, extensions.data());
        if (extension_result != PM_SUCCESS) return extension_result;
    }
    for (std::uint32_t index = 0U; index < info->enabledExtensionCount; ++index) {
        const char *name = info->ppEnabledExtensionNames[index];
        if (name == nullptr) return PM_ERROR_INVALID_ARGUMENT;
        const bool supported = std::any_of(
            extensions.begin(), extensions.end(), [&](const auto &extension) {
                return std::strcmp(extension.extensionName, name) == 0;
            });
        if (!supported) return PM_ERROR_EXTENSION_NOT_PRESENT;
    }
    auto create = reinterpret_cast<PFN_pmCreateDevice>(
        route.driver->resolve(route, "pmCreateDevice"));
    if (create == nullptr) return PM_ERROR_FEATURE_NOT_PRESENT;
    PmDevice native{};
    const PmResult result = create(reinterpret_cast<PmPhysicalDevice>(route.native),
                                   info, &native);
    if (result != PM_SUCCESS) return result;
    try {
        auto wrapper = std::make_unique<PmDevice_T>();
        wrapper->dispatch = &parallel_mater::loader::loader_dispatch;
        wrapper->driver = route.driver;
        wrapper->native = native;
        wrapper->native_instance = reinterpret_cast<PmInstance>(route.context);
        wrapper->layers = physical->layers;
        *device = wrapper.release();
        return PM_SUCCESS;
    } catch (const std::bad_alloc &) {
        auto destroy = reinterpret_cast<PFN_pmDestroyDevice>(
            route.driver->device_proc(native, "pmDestroyDevice"));
        if (destroy != nullptr) destroy(native);
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyDevice(PmDevice device) {
    const auto route = parallel_mater::loader::route(device);
    if (!route) return;
    auto destroy = reinterpret_cast<PFN_pmDestroyDevice>(
        route.driver->device_proc(reinterpret_cast<PmDevice>(route.context),
                                  "pmDestroyDevice"));
    if (destroy != nullptr) destroy(reinterpret_cast<PmDevice>(route.native));
    delete device;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetDeviceQueue(
    PmDevice device, std::uint32_t family, std::uint32_t index, PmQueue *queue) {
    if (queue == nullptr) return;
    *queue = nullptr;
    const auto route = parallel_mater::loader::route(device);
    if (!route) return;
    const std::uint64_t key = (static_cast<std::uint64_t>(family) << 32U) | index;
    std::lock_guard lock(device->mutex);
    if (const auto found = device->queues.find(key); found != device->queues.end()) {
        *queue = found->second.get();
        return;
    }
    auto get = reinterpret_cast<PFN_pmGetDeviceQueue>(
        route.driver->device_proc(reinterpret_cast<PmDevice>(route.context),
                                  "pmGetDeviceQueue"));
    if (get == nullptr) return;
    PmQueue native{};
    get(reinterpret_cast<PmDevice>(route.native), family, index, &native);
    if (native == nullptr) return;
    try {
        auto wrapper = std::make_unique<PmQueue_T>();
        wrapper->dispatch = &parallel_mater::loader::loader_dispatch;
        wrapper->driver = route.driver;
        wrapper->native = native;
        wrapper->native_device = reinterpret_cast<PmDevice>(route.native);
        wrapper->layers = device->layers;
        *queue = wrapper.get();
        device->queues.emplace(key, std::move(wrapper));
    } catch (const std::bad_alloc &) {
        *queue = nullptr;
    }
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmAllocateCommandBuffers(
    PmDevice device, PmCommandPool pool, std::uint32_t count,
    PmCommandBuffer *buffers) {
    if (buffers == nullptr || count == 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    std::fill_n(buffers, count, nullptr);
    const auto route = parallel_mater::loader::route(device);
    if (!route) return PM_ERROR_INVALID_ARGUMENT;
    auto allocate = reinterpret_cast<PFN_pmAllocateCommandBuffers>(
        route.driver->device_proc(reinterpret_cast<PmDevice>(route.context),
                                  "pmAllocateCommandBuffers"));
    if (allocate == nullptr) return PM_ERROR_FEATURE_NOT_PRESENT;
    std::vector<PmCommandBuffer> native(count);
    const PmResult result = allocate(reinterpret_cast<PmDevice>(route.native),
                                     pool, count, native.data());
    if (result != PM_SUCCESS) return result;
    std::uint32_t made{};
    try {
        for (; made < count; ++made) {
            auto wrapper = std::make_unique<PmCommandBuffer_T>();
            wrapper->dispatch = &parallel_mater::loader::loader_dispatch;
            wrapper->driver = route.driver;
            wrapper->native = native[made];
            wrapper->native_device = reinterpret_cast<PmDevice>(route.native);
            wrapper->layers = device->layers;
            buffers[made] = wrapper.release();
        }
        return PM_SUCCESS;
    } catch (const std::bad_alloc &) {
        auto free = reinterpret_cast<PFN_pmFreeCommandBuffers>(
            route.driver->device_proc(reinterpret_cast<PmDevice>(route.context),
                                      "pmFreeCommandBuffers"));
        if (free != nullptr)
            free(reinterpret_cast<PmDevice>(route.native), pool, count,
                 native.data());
        for (std::uint32_t index = 0U; index < made; ++index) {
            delete buffers[index];
            buffers[index] = nullptr;
        }
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmFreeCommandBuffers(
    PmDevice device, PmCommandPool pool, std::uint32_t count,
    const PmCommandBuffer *buffers) {
    const auto route = parallel_mater::loader::route(device);
    if (!route || (count != 0U && buffers == nullptr)) return;
    std::vector<PmCommandBuffer> native(count);
    for (std::uint32_t index = 0U; index < count; ++index) {
        const auto command_route = parallel_mater::loader::route(buffers[index]);
        if (!command_route || command_route.driver != route.driver) return;
        native[index] = reinterpret_cast<PmCommandBuffer>(command_route.native);
    }
    auto free = reinterpret_cast<PFN_pmFreeCommandBuffers>(
        route.driver->device_proc(reinterpret_cast<PmDevice>(route.context),
                                  "pmFreeCommandBuffers"));
    if (free != nullptr)
        free(reinterpret_cast<PmDevice>(route.native), pool, count, native.data());
    for (std::uint32_t index = 0U; index < count; ++index) delete buffers[index];
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmQueueSubmit(
    PmQueue queue, const PmSubmitInfo *info, PmFence fence) {
    const auto route = parallel_mater::loader::route(queue);
    if (!route || info == nullptr ||
        (info->commandBufferCount != 0U && info->pCommandBuffers == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmCommandBuffer> native(info->commandBufferCount);
    for (std::uint32_t index = 0U; index < info->commandBufferCount; ++index) {
        const auto command_route =
            parallel_mater::loader::route(info->pCommandBuffers[index]);
        if (!command_route || command_route.driver != route.driver)
            return PM_ERROR_INVALID_ARGUMENT;
        native[index] = reinterpret_cast<PmCommandBuffer>(command_route.native);
    }
    PmSubmitInfo translated = *info;
    translated.pCommandBuffers = native.data();
    auto submit = reinterpret_cast<PFN_pmQueueSubmit>(
        route.driver->device_proc(reinterpret_cast<PmDevice>(route.context),
                                  "pmQueueSubmit"));
    return submit == nullptr
        ? PM_ERROR_FEATURE_NOT_PRESENT
        : submit(reinterpret_cast<PmQueue>(route.native), &translated, fence);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmImportNativeDevice(
    PmInstance instance, const PmNativeDeviceImportInfo *info,
    PmDevice *device) {
    if (instance == nullptr || device == nullptr ||
        instance->dispatch != &parallel_mater::loader::loader_dispatch)
        return PM_ERROR_INVALID_ARGUMENT;
    *device = nullptr;
    for (const auto &child : instance->children) {
        auto import = reinterpret_cast<PFN_pmImportNativeDevice>(
            child.driver->instance_proc(child.native, "pmImportNativeDevice"));
        if (import == nullptr) continue;
        PmDevice native{};
        const PmResult result = import(child.native, info, &native);
        if (result != PM_SUCCESS) continue;
        try {
            auto wrapper = std::make_unique<PmDevice_T>();
            wrapper->dispatch = &parallel_mater::loader::loader_dispatch;
            wrapper->driver = child.driver;
            wrapper->native = native;
            wrapper->native_instance = child.native;
            wrapper->layers = instance->layers;
            *device = wrapper.release();
            return PM_SUCCESS;
        } catch (const std::bad_alloc &) {
            return PM_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmImportVulkanDeviceEXT(
    PmInstance instance, const PmVulkanDeviceImportInfo *info,
    PmDevice *device) {
    if (instance == nullptr || device == nullptr ||
        instance->dispatch != &parallel_mater::loader::loader_dispatch)
        return PM_ERROR_INVALID_ARGUMENT;
    *device = nullptr;
    for (const auto &child : instance->children) {
        auto import = reinterpret_cast<PFN_pmImportVulkanDeviceEXT>(
            child.driver->instance_proc(child.native, "pmImportVulkanDeviceEXT"));
        if (import == nullptr) continue;
        PmDevice native{};
        const PmResult result = import(child.native, info, &native);
        if (result != PM_SUCCESS) continue;
        try {
            auto wrapper = std::make_unique<PmDevice_T>();
            wrapper->dispatch = &parallel_mater::loader::loader_dispatch;
            wrapper->driver = child.driver;
            wrapper->native = native;
            wrapper->native_instance = child.native;
            wrapper->layers = instance->layers;
            *device = wrapper.release();
            return PM_SUCCESS;
        } catch (const std::bad_alloc &) {
            return PM_ERROR_OUT_OF_HOST_MEMORY;
        }
    }
    return PM_ERROR_FEATURE_NOT_PRESENT;
}
#endif

#include "loader_forward.inc"
