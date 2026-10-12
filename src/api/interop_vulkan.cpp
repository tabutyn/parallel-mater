// SPDX-License-Identifier: MIT
#include <parallel_mater/interop_vulkan.h>

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmImportVulkanDeviceEXT(
    PmInstance instance, const PmVulkanDeviceImportInfo *info,
    PmDevice *device) {
    if (device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *device = nullptr;
    if (instance == nullptr || info == nullptr || info->device == VK_NULL_HANDLE ||
        info->physicalDevice == VK_NULL_HANDLE ||
        (info->queueCount != 0U && info->pQueues == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetVulkanDeviceEXT(
    PmDevice device, PmVulkanDeviceInfo *native_device) {
    if (native_device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *native_device = {};
    if (device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmImportVulkanBufferEXT(
    PmDevice device, const PmVulkanBufferImportInfo *info, PmBuffer *buffer) {
    if (buffer == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *buffer = nullptr;
    if (device == nullptr || info == nullptr || info->buffer == VK_NULL_HANDLE ||
        info->size == 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetVulkanBufferEXT(
    PmDevice device, PmBuffer buffer, VkBuffer *native_buffer) {
    if (native_buffer == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *native_buffer = VK_NULL_HANDLE;
    if (device == nullptr || buffer == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmImportVulkanSemaphoreEXT(
    PmDevice device, VkSemaphore native, PmSemaphore *semaphore) {
    if (semaphore == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *semaphore = nullptr;
    if (device == nullptr || native == VK_NULL_HANDLE)
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetVulkanSemaphoreEXT(
    PmDevice device, PmSemaphore semaphore, VkSemaphore *native) {
    if (native == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *native = VK_NULL_HANDLE;
    if (device == nullptr || semaphore == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_ERROR_FEATURE_NOT_PRESENT;
}
