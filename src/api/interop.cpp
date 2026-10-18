// SPDX-License-Identifier: MIT
#include "internal.hpp"

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetPhysicalDeviceInteropProperties(
    PmPhysicalDevice physical, PmNativeBackend backend,
    PmNativeInteropProperties *properties) {
    if (properties == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *properties = {};
    if (physical == nullptr) return PM_ERROR_INVALID_ARGUMENT;
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
    (void)backend;
    return PM_ERROR_FEATURE_NOT_PRESENT;
#elif defined(PARALLEL_MATER_METAL_DRIVER)
    if (backend != PM_NATIVE_BACKEND_METAL) return PM_ERROR_FEATURE_NOT_PRESENT;
    return PM_SUCCESS; // Borrowed handles only; imports remain unsupported.
#elif defined(PARALLEL_MATER_VULKAN_DRIVER)
    if (backend != PM_NATIVE_BACKEND_VULKAN)
        return PM_ERROR_FEATURE_NOT_PRESENT;
    return PM_SUCCESS;
#elif defined(PARALLEL_MATER_D3D12_DRIVER)
    if (backend != PM_NATIVE_BACKEND_D3D12)
        return PM_ERROR_FEATURE_NOT_PRESENT;
    return PM_SUCCESS;
#else
    if (backend != PM_NATIVE_BACKEND_CUDA)
        return PM_ERROR_FEATURE_NOT_PRESENT;
    properties->bufferImport = PM_TRUE;
    return PM_SUCCESS;
#endif
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
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
    return PM_ERROR_FEATURE_NOT_PRESENT;
#else
#if defined(PARALLEL_MATER_METAL_DRIVER)
    native_device->backend = PM_NATIVE_BACKEND_METAL;
#elif defined(PARALLEL_MATER_VULKAN_DRIVER)
    native_device->backend = PM_NATIVE_BACKEND_VULKAN;
#elif defined(PARALLEL_MATER_D3D12_DRIVER)
    native_device->backend = PM_NATIVE_BACKEND_D3D12;
#else
    native_device->backend = PM_NATIVE_BACKEND_CUDA;
#endif
    native_device->physicalDevice = static_cast<PmNativeHandle>(
        device->physical->description.ordinal);
    native_device->device = static_cast<PmNativeHandle>(
        reinterpret_cast<std::uintptr_t>(
            device->services->native_device_handle()));
    device->services->describe_native_device(*native_device);
    if (native_device->device == 0U) {
        *native_device = {};
        return PM_ERROR_INVALID_STATE;
    }
    return PM_SUCCESS;
#endif
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetNativeQueue(
    PmQueue queue, PmNativeQueueInfo *native_queue) {
    if (native_queue == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *native_queue = {};
    if (queue == nullptr || queue->device == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
    return PM_ERROR_FEATURE_NOT_PRESENT;
#else
    queue->device->services->describe_native_queue(*native_queue);
    if (native_queue->queue == 0U) {
        *native_queue = {};
        return PM_ERROR_FEATURE_NOT_PRESENT;
    }
    return PM_SUCCESS;
#endif
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
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER) || \
    defined(PARALLEL_MATER_D3D12_DRIVER) || defined(PARALLEL_MATER_METAL_DRIVER) || \
    defined(PARALLEL_MATER_VULKAN_DRIVER)
    return PM_ERROR_FEATURE_NOT_PRESENT;
#else
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
    if (!parallel_mater::api::try_add_tracked_bytes(
            device->borrowed_buffer_bytes, created->size)) {
        parallel_mater::api::destroy_object(device, created);
        return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    *buffer = created;
    return PM_SUCCESS;
#endif
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetNativeBuffer(
    PmDevice device, PmBuffer buffer, PmNativeBufferInfo *native_buffer) {
    if (native_buffer == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *native_buffer = {};
    if (device == nullptr || buffer == nullptr || buffer->device != device ||
        buffer->native_pointer == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
    return PM_ERROR_FEATURE_NOT_PRESENT;
#else
    native_buffer->buffer = buffer->native_buffer;
    native_buffer->offset = buffer->native_offset;
    native_buffer->size = buffer->size;
    native_buffer->usage = buffer->usage;
    native_buffer->nativeUsage = buffer->native_usage;
    native_buffer->owningQueueFamilyIndex = buffer->queue_family;
    return PM_SUCCESS;
#endif
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
