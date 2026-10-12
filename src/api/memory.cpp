// SPDX-License-Identifier: MIT
#include "internal.hpp"

#include <cstring>

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmAllocateMemory(
    PmDevice device, const PmMemoryAllocateInfo *info, PmDeviceMemory *memory) {
    if (memory == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *memory = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, info->pNext) ||
        info->allocationSize == 0U || info->memoryTypeIndex > 1U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmDeviceMemory_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    const PmResult result = device->services->allocate(
        info->allocationSize, info->memoryTypeIndex == 1U, &created->allocation);
    if (result != PM_SUCCESS) {
        parallel_mater::api::destroy_object(device, created);
        return result;
    }
    created->device = device;
    created->size = info->allocationSize;
    created->memory_type = info->memoryTypeIndex;
    device->application_bytes.fetch_add(created->size, std::memory_order_relaxed);
    *memory = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmFreeMemory(
    PmDevice device, PmDeviceMemory memory) {
    if (device == nullptr || memory == nullptr || memory->device != device) return;
    device->services->free(memory->allocation);
    device->application_bytes.fetch_sub(memory->size, std::memory_order_relaxed);
    parallel_mater::api::destroy_object(device, memory);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmMapMemory(
    PmDevice device, PmDeviceMemory memory, PmDeviceSize offset,
    PmDeviceSize size, void **data) {
    if (data == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *data = nullptr;
    if (device == nullptr || memory == nullptr || memory->device != device ||
        memory->memory_type != 1U)
        return PM_ERROR_INVALID_ARGUMENT;
    size = parallel_mater::api::resolved_size(offset, size, memory->size);
    if (!parallel_mater::api::range_fits(offset, size, memory->size))
        return PM_ERROR_INVALID_ARGUMENT;
    memory->mapped = true;
    *data = static_cast<std::byte *>(memory->allocation) + offset;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmUnmapMemory(
    PmDevice device, PmDeviceMemory memory) {
    if (device != nullptr && memory != nullptr && memory->device == device)
        memory->mapped = false;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmFlushMappedMemoryRanges(
    PmDevice device, std::uint32_t count, const PmMappedMemoryRange *ranges) {
    if (device == nullptr || (count != 0U && ranges == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < count; ++index) {
        const auto &range = ranges[index];
        if (range.memory == nullptr || range.memory->device != device ||
            range.memory->memory_type != 1U ||
            !parallel_mater::api::range_fits(
                range.offset, parallel_mater::api::resolved_size(
                    range.offset, range.size, range.memory->size), range.memory->size))
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmInvalidateMappedMemoryRanges(
    PmDevice device, std::uint32_t count, const PmMappedMemoryRange *ranges) {
    return pmFlushMappedMemoryRanges(device, count, ranges);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateBuffer(
    PmDevice device, const PmBufferCreateInfo *info, PmBuffer *buffer) {
    if (buffer == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *buffer = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO, info->pNext) ||
        info->size == 0U || info->initialQueueFamilyIndex != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmBuffer_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->size = info->size;
    created->usage = info->usage;
    created->queue_family = info->initialQueueFamilyIndex;
    *buffer = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyBuffer(
    PmDevice device, PmBuffer buffer) {
    if (device == nullptr || buffer == nullptr || buffer->device != device) return;
    parallel_mater::api::destroy_object(device, buffer);
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetBufferMemoryRequirements(
    PmDevice device, PmBuffer buffer, PmMemoryRequirements *requirements) {
    if (requirements == nullptr) return;
    *requirements = {};
    if (device == nullptr || buffer == nullptr || buffer->device != device) return;
    requirements->size = buffer->size;
    requirements->alignment = 256U;
    requirements->memoryTypeBits = 0x3U;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmBindBufferMemory(
    PmDevice device, PmBuffer buffer, PmDeviceMemory memory,
    PmDeviceSize offset) {
    if (device == nullptr || buffer == nullptr || memory == nullptr ||
        buffer->device != device || memory->device != device ||
        offset % 256U != 0U ||
        !parallel_mater::api::range_fits(offset, buffer->size, memory->size))
        return PM_ERROR_INVALID_ARGUMENT;
    if (buffer->imported || buffer->memory != nullptr) return PM_ERROR_INVALID_STATE;
    buffer->memory = memory;
    buffer->memory_offset = offset;
    buffer->native_pointer = static_cast<std::byte *>(memory->allocation) + offset;
    return PM_SUCCESS;
}
