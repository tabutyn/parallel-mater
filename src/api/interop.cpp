// SPDX-License-Identifier: MIT
#include "internal.hpp"

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetPhysicalDeviceInteropProperties(
    PmPhysicalDevice physical, PmNativeBackend backend,
    PmNativeInteropProperties *properties) {
    if (properties == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *properties = {};
    if (physical == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    if (backend != PM_NATIVE_BACKEND_CUDA)
        return PM_ERROR_FEATURE_NOT_PRESENT;
    properties->bufferImport = PM_TRUE;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmImportNativeDevice(
    PmInstance instance, const PmNativeDeviceImportInfo *info, PmDevice *device) {
    if (device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *device = nullptr;
    if (instance == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_NATIVE_DEVICE_IMPORT_INFO, info->pNext))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetNativeDevice(
    PmDevice device, PmNativeDeviceInfo *native_device) {
    if (native_device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *native_device = {};
    if (device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    native_device->backend = PM_NATIVE_BACKEND_CUDA;
    native_device->physicalDevice = static_cast<PmNativeHandle>(
        device->physical->description.ordinal);
    native_device->device = static_cast<PmNativeHandle>(
        reinterpret_cast<std::uintptr_t>(
            device->services->native_device_handle()));
    if (native_device->device == 0U) {
        *native_device = {};
        return PM_ERROR_INVALID_STATE;
    }
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmImportNativeBuffer(
    PmDevice device, const PmNativeBufferInfo *info, PmBuffer *buffer) {
    if (buffer == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *buffer = nullptr;
    if (device == nullptr || info == nullptr || info->buffer == 0U ||
        info->size == 0U ||
        (info->owningQueueFamilyIndex != 0U &&
         info->owningQueueFamilyIndex != PM_QUEUE_FAMILY_EXTERNAL) ||
        info->offset > UINTPTR_MAX || info->buffer > UINTPTR_MAX ||
        info->buffer > UINTPTR_MAX - info->offset ||
        info->offset > UINT64_MAX - info->size ||
        (info->buffer + info->offset) % 256U != 0U ||
        info->nativeUsage != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmBuffer_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->size = info->size;
    created->usage = info->usage;
    created->queue_family = info->owningQueueFamilyIndex;
    created->native_buffer = info->buffer;
    created->native_offset = info->offset;
    created->native_usage = info->nativeUsage;
    created->native_pointer = reinterpret_cast<void *>(
        static_cast<std::uintptr_t>(info->buffer + info->offset));
    created->imported = true;
    *buffer = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetNativeBuffer(
    PmDevice device, PmBuffer buffer, PmNativeBufferInfo *native_buffer) {
    if (device == nullptr || buffer == nullptr || native_buffer == nullptr ||
        buffer->device != device || buffer->native_pointer == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    *native_buffer = {};
    native_buffer->buffer = buffer->native_buffer;
    native_buffer->offset = buffer->native_offset;
    native_buffer->size = buffer->size;
    native_buffer->usage = buffer->usage;
    native_buffer->nativeUsage = buffer->native_usage;
    native_buffer->owningQueueFamilyIndex = buffer->queue_family;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmImportNativeSemaphore(
    PmDevice device, PmNativeHandle native, PmSemaphore *semaphore) {
    if (semaphore == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *semaphore = nullptr;
    if (device == nullptr || native == 0U) return PM_ERROR_INVALID_ARGUMENT;
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetNativeSemaphore(
    PmDevice device, PmSemaphore semaphore, PmNativeHandle *native) {
    if (native == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *native = 0U;
    if (device == nullptr || semaphore == nullptr ||
        semaphore->device != device)
        return PM_ERROR_INVALID_ARGUMENT;
    *native = semaphore->native;
    return semaphore->imported ? PM_SUCCESS : PM_ERROR_FEATURE_NOT_PRESENT;
}
