// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_driver.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>

#if defined(_WIN32)
#define PM_MOCK_EXPORT extern "C" __declspec(dllexport)
#else
#define PM_MOCK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct MockInstance { std::uint32_t tag{0x504d4452U}; };
struct MockPhysical { MockInstance *instance{}; };

PmResult mock_create_instance(const PmInstanceCreateInfo *,
                              const PmAllocationCallbacks *,
                              PmInstance *output) {
    if (output == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *output = reinterpret_cast<PmInstance>(new (std::nothrow) MockInstance);
    return *output == nullptr ? PM_ERROR_OUT_OF_HOST_MEMORY : PM_SUCCESS;
}

void mock_destroy_instance(PmInstance instance) {
    delete reinterpret_cast<MockInstance *>(instance);
}

PmResult mock_enumerate_physical_devices(PmInstance instance,
                                         std::uint32_t *count,
                                         PmPhysicalDevice *devices) {
    if (instance == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (devices == nullptr) {
        *count = 1U;
        return PM_SUCCESS;
    }
    if (*count == 0U) return PM_INCOMPLETE;
    // The mock physical handle is the mock instance.  It remains valid for the
    // instance lifetime and no device can be created from it.
    devices[0] = reinterpret_cast<PmPhysicalDevice>(instance);
    *count = 1U;
    return PM_SUCCESS;
}

void mock_get_properties(PmPhysicalDevice physical,
                         PmPhysicalDeviceProperties *properties) {
    if (properties == nullptr) return;
    *properties = {};
    if (physical == nullptr) return;
    properties->apiVersion = PM_API_VERSION_0_2;
    properties->vendorId = 0x504dU;
    properties->deviceId = 1U;
    std::strncpy(properties->deviceName, "Parallel Mater mock vendor device",
                 sizeof properties->deviceName - 1U);
}

// Deliberately permissive device functions: the validation tests must prove
// rejection by the layer, rather than accidentally relying on the driver.
PmResult mock_create_device(PmPhysicalDevice, const PmDeviceCreateInfo *, PmDevice *out) {
    *out = reinterpret_cast<PmDevice>(new char);
    return PM_SUCCESS;
}
void mock_destroy_device(PmDevice device) { delete reinterpret_cast<char *>(device); }
PmResult mock_allocate(PmDevice, const PmMemoryAllocateInfo *info, PmDeviceMemory *out) {
    *out = nullptr;
    if (info->allocationSize == 13) return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    *out = reinterpret_cast<PmDeviceMemory>(new char);
    return PM_SUCCESS;
}
void mock_free(PmDevice, PmDeviceMemory memory) { delete reinterpret_cast<char *>(memory); }
PmResult mock_create_buffer(PmDevice, const PmBufferCreateInfo *, PmBuffer *out) {
    *out = reinterpret_cast<PmBuffer>(new char);
    return PM_SUCCESS;
}
void mock_destroy_buffer(PmDevice, PmBuffer buffer) { delete reinterpret_cast<char *>(buffer); }
void mock_requirements(PmDevice, PmBuffer, PmMemoryRequirements *out) {
    *out = {256, 256, 2, PM_FALSE};
}
PmResult mock_bind(PmDevice, PmBuffer, PmDeviceMemory, PmDeviceSize) { return PM_SUCCESS; }
PmResult mock_map(PmDevice, PmDeviceMemory, PmDeviceSize, PmDeviceSize, void **out) {
    static char storage;
    *out = &storage;
    return PM_SUCCESS;
}
void mock_unmap(PmDevice, PmDeviceMemory) {}
PmResult mock_ranges(PmDevice, uint32_t, const PmMappedMemoryRange *) { return PM_SUCCESS; }

PFN_pmVoidFunction lookup_instance(const char *name) {
    if (name == nullptr) return nullptr;
    if (std::strcmp(name, "pmCreateInstance") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_create_instance);
    if (std::strcmp(name, "pmDestroyInstance") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_destroy_instance);
    if (std::strcmp(name, "pmEnumeratePhysicalDevices") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(
            &mock_enumerate_physical_devices);
    return nullptr;
}

} // namespace

PM_MOCK_EXPORT PmResult PMAPI_CALL
pm_driverNegotiateLoaderInterfaceVersion(std::uint32_t *version) {
    if (version == nullptr || *version == 0U)
        return PM_ERROR_INVALID_ARGUMENT;
#if defined(PM_MOCK_INTERFACE_VERSION)
    *version = PM_MOCK_INTERFACE_VERSION;
#else
    *version = std::min(*version, PM_DRIVER_INTERFACE_VERSION);
#endif
    return PM_SUCCESS;
}

PM_MOCK_EXPORT PFN_pmVoidFunction PMAPI_CALL pm_driverGetInstanceProcAddr(
    PmInstance, const char *name) {
    return lookup_instance(name);
}

PM_MOCK_EXPORT PFN_pmVoidFunction PMAPI_CALL pm_driverGetDeviceProcAddr(
    PmDevice, const char *name) {
    if (!name) return nullptr;
#define MOCK(command, implementation) \
    if (std::strcmp(name, #command) == 0) \
        return reinterpret_cast<PFN_pmVoidFunction>(&implementation)
    MOCK(pmDestroyDevice, mock_destroy_device);
    MOCK(pmAllocateMemory, mock_allocate);
    MOCK(pmFreeMemory, mock_free);
    MOCK(pmCreateBuffer, mock_create_buffer);
    MOCK(pmDestroyBuffer, mock_destroy_buffer);
    MOCK(pmGetBufferMemoryRequirements, mock_requirements);
    MOCK(pmBindBufferMemory, mock_bind);
    MOCK(pmMapMemory, mock_map);
    MOCK(pmUnmapMemory, mock_unmap);
    MOCK(pmFlushMappedMemoryRanges, mock_ranges);
    MOCK(pmInvalidateMappedMemoryRanges, mock_ranges);
#undef MOCK
    return nullptr;
}

PM_MOCK_EXPORT PFN_pmVoidFunction PMAPI_CALL
pm_driverGetPhysicalDeviceProcAddr(PmInstance, const char *name) {
    if (name && std::strcmp(name, "pmCreateDevice") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_create_device);
    return name != nullptr &&
           std::strcmp(name, "pmGetPhysicalDeviceProperties") == 0
        ? reinterpret_cast<PFN_pmVoidFunction>(&mock_get_properties)
        : nullptr;
}
