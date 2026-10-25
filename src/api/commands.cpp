// SPDX-License-Identifier: MIT
#include "internal.hpp"

#include <algorithm>

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateCommandPool(
    PmDevice device, const PmCommandPoolCreateInfo *info, PmCommandPool *pool) {
    if (pool == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *pool = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, info->pNext) ||
        info->queueFamilyIndex != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmCommandPool_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->flags = info->flags;
    created->queue_family = info->queueFamilyIndex;
    *pool = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyCommandPool(
    PmDevice device, PmCommandPool pool) {
    if (device == nullptr || pool == nullptr || pool->device != device) return;
    (void)pmDeviceWaitIdle(device);
    for (PmCommandBuffer buffer : pool->buffers)
        parallel_mater::api::destroy_object(device, buffer);
    pool->buffers.clear();
    parallel_mater::api::destroy_object(device, pool);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmResetCommandPool(
    PmDevice device, PmCommandPool pool) {
    if (device == nullptr || pool == nullptr || pool->device != device)
        return PM_ERROR_INVALID_ARGUMENT;
    for (PmCommandBuffer buffer : pool->buffers) {
        parallel_mater::api::refresh_command_state(buffer);
        if (buffer->state == PmCommandState::recording ||
            buffer->state == PmCommandState::pending)
            return PM_ERROR_INVALID_STATE;
    }
    for (PmCommandBuffer buffer : pool->buffers) {
        buffer->commands.clear();
        buffer->query_writes.clear();
        buffer->simulation_resources.clear();
        buffer->latched_error = PM_SUCCESS;
        buffer->bound_pipeline = nullptr;
        buffer->statistics_open = false;
        buffer->statistics_pool = nullptr;
        buffer->statistics_step_count = 0U;
        buffer->statistics_simulation = nullptr;
        buffer->statistics_execution.reset();
        buffer->state = PmCommandState::initial;
    }
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmAllocateCommandBuffers(
    PmDevice device, PmCommandPool pool, std::uint32_t count,
    PmCommandBuffer *buffers) {
    if (device == nullptr || pool == nullptr || pool->device != device ||
        count == 0U || buffers == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    std::fill_n(buffers, count, nullptr);
    const std::size_t previous = pool->buffers.size();
    try {
        pool->buffers.reserve(previous + count);
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    for (std::uint32_t index = 0U; index < count; ++index) {
        auto *created = parallel_mater::api::make_object<PmCommandBuffer_T>(device);
        if (created == nullptr) {
            for (std::size_t rollback = previous; rollback < pool->buffers.size(); ++rollback)
                parallel_mater::api::destroy_object(device, pool->buffers[rollback]);
            pool->buffers.resize(previous);
            return PM_ERROR_OUT_OF_HOST_MEMORY;
        }
        created->device = device;
        created->pool = pool;
        pool->buffers.push_back(created);
        buffers[index] = created;
    }
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmFreeCommandBuffers(
    PmDevice device, PmCommandPool pool, std::uint32_t count,
    const PmCommandBuffer *buffers) {
    if (device == nullptr || pool == nullptr || pool->device != device ||
        (count != 0U && buffers == nullptr))
        return;
    for (std::uint32_t index = 0U; index < count; ++index) {
        PmCommandBuffer buffer = buffers[index];
        parallel_mater::api::refresh_command_state(buffer);
        if (buffer == nullptr || buffer->pool != pool ||
            buffer->state == PmCommandState::pending)
            continue;
        const auto found = std::find(pool->buffers.begin(), pool->buffers.end(), buffer);
        if (found == pool->buffers.end()) continue;
        parallel_mater::api::destroy_object(device, buffer);
        pool->buffers.erase(found);
    }
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmBeginCommandBuffer(
    PmCommandBuffer buffer, const PmCommandBufferBeginInfo *info) {
    parallel_mater::api::refresh_command_state(buffer);
    if (buffer == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, info->pNext) ||
        buffer->state == PmCommandState::recording ||
        buffer->state == PmCommandState::pending)
        return PM_ERROR_INVALID_STATE;
    buffer->commands.clear();
    buffer->query_writes.clear();
    buffer->simulation_resources.clear();
    buffer->latched_error = PM_SUCCESS;
    buffer->bound_pipeline = nullptr;
    buffer->statistics_open = false;
    buffer->statistics_pool = nullptr;
    buffer->statistics_step_count = 0U;
    buffer->statistics_simulation = nullptr;
    buffer->statistics_execution.reset();
    buffer->usage = info->flags;
    buffer->state = PmCommandState::recording;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmEndCommandBuffer(
    PmCommandBuffer buffer) {
    if (buffer == nullptr || buffer->state != PmCommandState::recording)
        return PM_ERROR_INVALID_STATE;
    if (buffer->latched_error != PM_SUCCESS) return buffer->latched_error;
    if (buffer->statistics_open) return PM_ERROR_INVALID_STATE;
    buffer->state = PmCommandState::executable;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmResetCommandBuffer(
    PmCommandBuffer buffer) {
    parallel_mater::api::refresh_command_state(buffer);
    if (buffer == nullptr || buffer->pool == nullptr ||
        (buffer->pool->flags & PM_COMMAND_POOL_RESET_BUFFER_BIT) == 0U ||
        buffer->state == PmCommandState::pending)
        return PM_ERROR_INVALID_STATE;
    buffer->commands.clear();
    buffer->query_writes.clear();
    buffer->simulation_resources.clear();
    buffer->latched_error = PM_SUCCESS;
    buffer->bound_pipeline = nullptr;
    buffer->statistics_open = false;
    buffer->statistics_pool = nullptr;
    buffer->statistics_step_count = 0U;
    buffer->statistics_simulation = nullptr;
    buffer->statistics_execution.reset();
    buffer->state = PmCommandState::initial;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdCopyBuffer(
    PmCommandBuffer command_buffer, PmBuffer source, PmBuffer destination,
    std::uint32_t count, const PmBufferCopy *regions) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (source == nullptr || destination == nullptr ||
        source->device != command_buffer->device ||
        destination->device != command_buffer->device || count == 0U ||
        regions == nullptr || source->native_pointer == nullptr ||
        destination->native_pointer == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmBufferCopy> copied;
    try {
        copied.assign(regions, regions + count);
    } catch (const std::bad_alloc &) {
        command_buffer->latched_error = PM_ERROR_OUT_OF_HOST_MEMORY;
        return command_buffer->latched_error;
    }
    for (const PmBufferCopy &region : copied)
        if (!parallel_mater::api::range_fits(region.srcOffset, region.size,
                                             source->size) ||
            !parallel_mater::api::range_fits(region.dstOffset, region.size,
                                             destination->size))
            return PM_ERROR_INVALID_ARGUMENT;
    return parallel_mater::api::record(
        command_buffer, [source, destination, copied = std::move(copied)](
                            parallel_mater::device::Services &services) noexcept {
            if (source->queue_family != 0U || destination->queue_family != 0U)
                return PM_ERROR_INVALID_STATE;
            for (const PmBufferCopy &region : copied) {
                const PmResult result = services.copy(
                    parallel_mater::api::buffer_pointer(destination, region.dstOffset),
                    parallel_mater::api::buffer_pointer(source, region.srcOffset),
                    region.size);
                if (result != PM_SUCCESS) return result;
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdPipelineBarrier(
    PmCommandBuffer command_buffer, std::uint32_t memory_count,
    const PmMemoryBarrier *memory_barriers, std::uint32_t buffer_count,
    const PmBufferMemoryBarrier *buffer_barriers) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if ((memory_count != 0U && memory_barriers == nullptr) ||
        (buffer_count != 0U && buffer_barriers == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    std::vector<PmBufferMemoryBarrier> copied;
    try {
        if (buffer_count != 0U)
            copied.assign(buffer_barriers, buffer_barriers + buffer_count);
    } catch (const std::bad_alloc &) {
        command_buffer->latched_error = PM_ERROR_OUT_OF_HOST_MEMORY;
        return command_buffer->latched_error;
    }
    for (const auto &barrier : copied) {
        const bool ignored =
            barrier.srcQueueFamilyIndex == PM_QUEUE_FAMILY_IGNORED &&
            barrier.dstQueueFamilyIndex == PM_QUEUE_FAMILY_IGNORED;
        if (barrier.range.buffer == nullptr ||
            barrier.range.buffer->device != command_buffer->device ||
            !parallel_mater::api::range_fits(
                barrier.range.offset, barrier.range.size,
                barrier.range.buffer->size) ||
            (!ignored &&
             (barrier.srcQueueFamilyIndex == PM_QUEUE_FAMILY_IGNORED ||
              barrier.dstQueueFamilyIndex == PM_QUEUE_FAMILY_IGNORED)) ||
            (!ignored && barrier.srcQueueFamilyIndex != 0U &&
             barrier.srcQueueFamilyIndex != PM_QUEUE_FAMILY_EXTERNAL) ||
            (!ignored && barrier.dstQueueFamilyIndex != 0U &&
             barrier.dstQueueFamilyIndex != PM_QUEUE_FAMILY_EXTERNAL))
            return PM_ERROR_INVALID_ARGUMENT;
    }
    if (copied.empty()) return PM_SUCCESS;
    return parallel_mater::api::record(
        command_buffer, [copied = std::move(copied)](
            parallel_mater::device::Services &) noexcept {
            for (const auto &barrier : copied) {
                if (barrier.srcQueueFamilyIndex == PM_QUEUE_FAMILY_IGNORED)
                    continue;
                if (barrier.range.buffer->queue_family !=
                    barrier.srcQueueFamilyIndex)
                    return PM_ERROR_INVALID_STATE;
                barrier.range.buffer->queue_family =
                    barrier.dstQueueFamilyIndex;
            }
            return PM_SUCCESS;
        });
}
