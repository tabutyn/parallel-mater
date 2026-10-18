// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_layer.h>
#include <parallel_mater/pm_layer_args.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>

#if defined(_WIN32)
#define PM_LAYER_EXPORT extern "C" __declspec(dllexport)
#else
#define PM_LAYER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
struct Memory {
    PmDeviceSize size{};
    uint32_t type{};
    bool mapped{};
};
struct Buffer {
    PmDeviceSize size{};
    PmDeviceMemory memory{};
    bool bound{};
    PmMemoryRequirements requirements{};
};
enum class CommandState { initial, recording, executable, pending, invalid };
struct CommandPool {
    PmDevice device{};
    PmFlags flags{};
    std::uint32_t queue_family{};
    std::unordered_set<PmCommandBuffer> buffers;
};
struct CommandBuffer {
    PmDevice device{};
    PmCommandPool pool{};
    std::uint64_t pending_submission{};
    PmFlags usage{};
    CommandState state{CommandState::initial};
};
enum class FenceState { unsignaled, pending, signaled };
struct Fence {
    PmDevice device{};
    std::uint64_t pending_submission{};
    FenceState state{FenceState::unsignaled};
};
struct Submission {
    PmDevice device{};
    PmQueue queue{};
};
struct Semaphore {
    PmDevice device{};
    std::uint64_t value{};
    std::uint64_t greatest_value{};
    bool value_known{};
    std::map<std::uint64_t, std::uint64_t> signals;
};
struct Queue {
    PmDevice device{};
    std::uint32_t family{};
};
struct Device {
    std::unordered_map<PmDeviceMemory, Memory> memories;
    std::unordered_map<PmBuffer, Buffer> buffers;
};
std::mutex mutex;
std::unordered_map<PmDevice, Device> devices;
std::unordered_map<PmCommandPool, CommandPool> command_pools;
std::unordered_map<PmCommandBuffer, CommandBuffer> command_buffers;
std::unordered_map<PmQueue, Queue> queues;
std::unordered_map<PmFence, Fence> fences;
std::unordered_map<std::uint64_t, Submission> submissions;
std::unordered_map<PmSemaphore, Semaphore> semaphores;
std::uint64_t next_submission = 1U;
bool tracking_available = true;

bool fits(PmDeviceSize offset, PmDeviceSize size, PmDeviceSize capacity) {
    return offset <= capacity && size <= capacity - offset;
}
bool range_fits(PmDeviceSize offset, PmDeviceSize size, PmDeviceSize capacity) {
    return offset <= capacity && (size == PM_WHOLE_SIZE || fits(offset, size, capacity));
}
Memory *memory(PmDevice device, PmDeviceMemory value) {
    const auto d = devices.find(device);
    if (d == devices.end()) return nullptr;
    const auto m = d->second.memories.find(value);
    return m == d->second.memories.end() ? nullptr : &m->second;
}
Buffer *buffer(PmDevice device, PmBuffer value) {
    const auto d = devices.find(device);
    if (d == devices.end()) return nullptr;
    const auto b = d->second.buffers.find(value);
    return b == d->second.buffers.end() ? nullptr : &b->second;
}
CommandPool *command_pool(PmDevice device, PmCommandPool value) {
    const auto found = command_pools.find(value);
    return found != command_pools.end() && found->second.device == device
        ? &found->second : nullptr;
}
CommandBuffer *command_buffer(PmCommandBuffer value) {
    const auto found = command_buffers.find(value);
    return found == command_buffers.end() ? nullptr : &found->second;
}
Fence *fence(PmDevice device, PmFence value) {
    const auto found = fences.find(value);
    return found != fences.end() && found->second.device == device
        ? &found->second : nullptr;
}
Semaphore *semaphore(PmDevice device, PmSemaphore value) {
    const auto found = semaphores.find(value);
    return found != semaphores.end() && found->second.device == device
        ? &found->second : nullptr;
}
void complete(CommandBuffer &buffer) {
    buffer.state = (buffer.usage & PM_COMMAND_BUFFER_ONE_TIME_BIT) != 0U
        ? CommandState::invalid : CommandState::executable;
    buffer.pending_submission = 0U;
}
void finish_submission(std::uint64_t value, PmResult result) {
    if (!value) return;
    for (auto &[handle, buffer] : command_buffers) {
        (void)handle;
        if (buffer.state == CommandState::pending &&
            buffer.pending_submission == value)
            complete(buffer);
    }
    for (auto &[handle, tracked_fence] : fences) {
        (void)handle;
        if (tracked_fence.state == FenceState::pending &&
            tracked_fence.pending_submission == value) {
            tracked_fence.pending_submission = 0U;
            tracked_fence.state = FenceState::signaled;
        }
    }
    for (auto &[handle, timeline] : semaphores) {
        (void)handle;
        for (auto signal = timeline.signals.begin();
             signal != timeline.signals.end();) {
            if (signal->second != value) {
                ++signal;
                continue;
            }
            if (result == PM_SUCCESS) {
                timeline.value = std::max(timeline.value, signal->first);
                timeline.value_known = true;
            }
            signal = timeline.signals.erase(signal);
        }
    }
    submissions.erase(value);
}
void complete_fence(PmFence value, PmResult result = PM_SUCCESS) {
    const auto found = fences.find(value);
    if (found == fences.end()) return;
    finish_submission(found->second.pending_submission, result);
    found->second.pending_submission = 0U;
    found->second.state = FenceState::signaled;
}
void complete_queue(PmQueue value, PmResult result = PM_SUCCESS) {
    for (;;) {
        std::uint64_t pending{};
        for (const auto &[id, submission] : submissions)
            if (submission.queue == value) {
                pending = id;
                break;
            }
        if (!pending) break;
        finish_submission(pending, result);
    }
}
void complete_device(PmDevice device, PmResult result = PM_SUCCESS) {
    for (;;) {
        std::uint64_t pending{};
        for (const auto &[id, submission] : submissions)
            if (submission.device == device) {
                pending = id;
                break;
            }
        if (!pending) break;
        finish_submission(pending, result);
    }
}
void complete_timeline_through(PmDevice device, PmSemaphore handle,
                               std::uint64_t value, PmResult result) {
    auto *timeline = semaphore(device, handle);
    if (!timeline) return;
    const auto selected = timeline->signals.lower_bound(value);
    if (selected == timeline->signals.end()) {
        if (result == PM_SUCCESS) {
            timeline->value = std::max(timeline->value, value);
            timeline->greatest_value = std::max(
                timeline->greatest_value, timeline->value);
            timeline->value_known = true;
        }
        return;
    }
    const auto boundary = selected->first;
    for (;;) {
        timeline = semaphore(device, handle);
        if (!timeline || timeline->signals.empty() ||
            timeline->signals.begin()->first > boundary)
            break;
        const auto submission = timeline->signals.begin()->second;
        finish_submission(submission, result);
    }
    timeline = semaphore(device, handle);
    if (timeline && result == PM_SUCCESS) {
        timeline->value = std::max(timeline->value, value);
        timeline->greatest_value = std::max(
            timeline->greatest_value, timeline->value);
        timeline->value_known = true;
    }
}
void observe_timeline_value(PmDevice device, PmSemaphore handle,
                            std::uint64_t value) {
    auto *timeline = semaphore(device, handle);
    while (timeline && !timeline->signals.empty() &&
           timeline->signals.begin()->first <= value) {
        const auto submission = timeline->signals.begin()->second;
        finish_submission(submission, PM_SUCCESS);
        timeline = semaphore(device, handle);
    }
    if (timeline) {
        timeline->value = std::max(timeline->value, value);
        timeline->greatest_value = std::max(
            timeline->greatest_value, timeline->value);
        timeline->value_known = true;
    }
}
template<class T> PmResult validate(const T &) { return PM_SUCCESS; }
template<class T> void record(const T &) {}

void record(const PmLayerArgs_pmCreateDevice &a) {
    if (a.pDevice && *a.pDevice) devices.try_emplace(*a.pDevice);
}
void record(const PmLayerArgs_pmImportNativeDevice &a) {
    if (a.pDevice && *a.pDevice) devices.try_emplace(*a.pDevice);
}
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
void record(const PmLayerArgs_pmImportVulkanDeviceEXT &a) {
    if (a.pDevice && *a.pDevice) devices.try_emplace(*a.pDevice);
}
#endif
void record(const PmLayerArgs_pmGetDeviceQueue &a) {
    if (a.pQueue && *a.pQueue) queues[*a.pQueue] = {a.device, a.family};
}

PmResult validate(const PmLayerArgs_pmCreateCommandPool &a) {
    return a.pInfo && a.pPool &&
        a.pInfo->sType == PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO &&
        !a.pInfo->pNext &&
        (a.pInfo->flags & ~PM_COMMAND_POOL_RESET_BUFFER_BIT) == 0U
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCreateCommandPool &a) {
    if (a.pPool && *a.pPool)
        command_pools[*a.pPool] = {
            a.device, a.pInfo->flags, a.pInfo->queueFamilyIndex};
}
PmResult validate(const PmLayerArgs_pmDestroyCommandPool &a) {
    if (!tracking_available || !a.pool) return PM_SUCCESS;
    const auto *pool = command_pool(a.device, a.pool);
    if (!pool) return PM_ERROR_INVALID_ARGUMENT;
    for (const auto handle : pool->buffers)
        if (const auto *buffer = command_buffer(handle);
            buffer && buffer->state == CommandState::pending)
            return PM_ERROR_INVALID_STATE;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmDestroyCommandPool &a) {
    const auto found = command_pools.find(a.pool);
    if (found == command_pools.end()) return;
    for (const auto buffer : found->second.buffers)
        command_buffers.erase(buffer);
    command_pools.erase(found);
}
PmResult validate(const PmLayerArgs_pmResetCommandPool &a) {
    if (!tracking_available) return PM_SUCCESS;
    const auto *pool = command_pool(a.device, a.pool);
    if (!pool) return PM_ERROR_INVALID_ARGUMENT;
    for (const auto handle : pool->buffers) {
        const auto *buffer = command_buffer(handle);
        if (buffer && (buffer->state == CommandState::recording ||
                       buffer->state == CommandState::pending))
            return PM_ERROR_INVALID_STATE;
    }
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmResetCommandPool &a) {
    if (auto *pool = command_pool(a.device, a.pool))
        for (const auto handle : pool->buffers)
            if (auto *buffer = command_buffer(handle)) {
                buffer->state = CommandState::initial;
                buffer->usage = 0U;
            }
}
PmResult validate(const PmLayerArgs_pmAllocateCommandBuffers &a) {
    if (!a.count || !a.pBuffers) return PM_ERROR_INVALID_ARGUMENT;
    return !tracking_available || command_pool(a.device, a.pool)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmAllocateCommandBuffers &a) {
    auto *pool = command_pool(a.device, a.pool);
    if (!pool) return;
    for (std::uint32_t index = 0U; index < a.count; ++index)
        if (a.pBuffers[index]) {
            command_buffers[a.pBuffers[index]] = {a.device, a.pool};
            pool->buffers.insert(a.pBuffers[index]);
        }
}
PmResult validate(const PmLayerArgs_pmFreeCommandBuffers &a) {
    if (a.count && !a.pBuffers) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *pool = command_pool(a.device, a.pool);
    if (!pool) return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < a.count; ++index) {
        const auto *buffer = command_buffer(a.pBuffers[index]);
        if (!buffer || buffer->device != a.device || buffer->pool != a.pool)
            return PM_ERROR_INVALID_ARGUMENT;
        for (std::uint32_t previous = 0U; previous < index; ++previous)
            if (a.pBuffers[previous] == a.pBuffers[index])
                return PM_ERROR_INVALID_ARGUMENT;
        if (buffer->state == CommandState::pending)
            return PM_ERROR_INVALID_STATE;
    }
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmFreeCommandBuffers &a) {
    auto *pool = command_pool(a.device, a.pool);
    for (std::uint32_t index = 0U; index < a.count; ++index) {
        if (pool) pool->buffers.erase(a.pBuffers[index]);
        command_buffers.erase(a.pBuffers[index]);
    }
}
PmResult validate(const PmLayerArgs_pmBeginCommandBuffer &a) {
    if (!a.pInfo ||
        a.pInfo->sType != PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO ||
        a.pInfo->pNext ||
        (a.pInfo->flags & ~PM_COMMAND_BUFFER_ONE_TIME_BIT) != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *buffer = command_buffer(a.commandBuffer);
    if (!buffer) return PM_ERROR_INVALID_ARGUMENT;
    return buffer->state == CommandState::recording ||
           buffer->state == CommandState::pending
        ? PM_ERROR_INVALID_STATE : PM_SUCCESS;
}
void record(const PmLayerArgs_pmBeginCommandBuffer &a) {
    if (auto *buffer = command_buffer(a.commandBuffer)) {
        buffer->state = CommandState::recording;
        buffer->usage = a.pInfo->flags;
    }
}
PmResult validate(const PmLayerArgs_pmEndCommandBuffer &a) {
    if (!tracking_available) return PM_SUCCESS;
    const auto *buffer = command_buffer(a.commandBuffer);
    if (!buffer) return PM_ERROR_INVALID_ARGUMENT;
    return buffer->state == CommandState::recording
        ? PM_SUCCESS : PM_ERROR_INVALID_STATE;
}
void record(const PmLayerArgs_pmEndCommandBuffer &a) {
    if (auto *buffer = command_buffer(a.commandBuffer))
        buffer->state = CommandState::executable;
}
PmResult validate(const PmLayerArgs_pmResetCommandBuffer &a) {
    if (!tracking_available) return PM_SUCCESS;
    const auto *buffer = command_buffer(a.commandBuffer);
    if (!buffer) return PM_ERROR_INVALID_ARGUMENT;
    const auto *pool = command_pool(buffer->device, buffer->pool);
    if (!pool) return PM_ERROR_INVALID_ARGUMENT;
    if ((pool->flags & PM_COMMAND_POOL_RESET_BUFFER_BIT) == 0U ||
        buffer->state == CommandState::pending)
        return PM_ERROR_INVALID_STATE;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmResetCommandBuffer &a) {
    if (auto *buffer = command_buffer(a.commandBuffer)) {
        buffer->state = CommandState::initial;
        buffer->usage = 0U;
    }
}

PmResult validate(const PmLayerArgs_pmCreateSemaphore &a) {
    return a.pInfo && a.pSemaphore &&
        a.pInfo->sType == PM_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO &&
        !a.pInfo->pNext ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCreateSemaphore &a) {
    if (a.pSemaphore && *a.pSemaphore)
        semaphores[*a.pSemaphore] = {
            a.device, a.pInfo->initialValue, a.pInfo->initialValue, true};
}
PmResult validate(const PmLayerArgs_pmDestroySemaphore &a) {
    return !a.semaphore || !tracking_available ||
        semaphore(a.device, a.semaphore)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmDestroySemaphore &a) {
    for (;;) {
        const auto found = semaphores.find(a.semaphore);
        if (found == semaphores.end() || found->second.signals.empty()) break;
        finish_submission(found->second.signals.begin()->second,
                          PM_ERROR_DEVICE_LOST);
    }
    semaphores.erase(a.semaphore);
}
PmResult validate(const PmLayerArgs_pmGetSemaphoreCounterValue &a) {
    return a.pValue && (!tracking_available || semaphore(a.device, a.semaphore))
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
PmResult validate(const PmLayerArgs_pmWaitSemaphore &a) {
    return !tracking_available || semaphore(a.device, a.semaphore)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
PmResult validate(const PmLayerArgs_pmImportNativeSemaphore &a) {
    return a.nativeSemaphore && a.pSemaphore
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmImportNativeSemaphore &a) {
    if (a.pSemaphore && *a.pSemaphore)
        semaphores[*a.pSemaphore] = {a.device, 0U, 0U, false};
}
PmResult validate(const PmLayerArgs_pmGetNativeSemaphore &a) {
    return a.pNativeSemaphore &&
        (!tracking_available || semaphore(a.device, a.semaphore))
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}

PmResult validate(const PmLayerArgs_pmQueueSubmit &a) {
    if (!a.pInfo || a.pInfo->sType != PM_STRUCTURE_TYPE_SUBMIT_INFO ||
        a.pInfo->pNext ||
        (a.pInfo->waitCount && !a.pInfo->pWaits) ||
        (a.pInfo->commandBufferCount && !a.pInfo->pCommandBuffers) ||
        (a.pInfo->signalCount && !a.pInfo->pSignals))
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto queue = queues.find(a.queue);
    if (queue == queues.end()) return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < a.pInfo->waitCount; ++index) {
        const auto &wait = a.pInfo->pWaits[index];
        const auto *timeline = semaphore(queue->second.device, wait.semaphore);
        if (!timeline || wait.stages == 0U ||
            (wait.stages & ~PM_STAGE_ALL_COMMANDS_BIT) != 0U)
            return PM_ERROR_INVALID_ARGUMENT;
        const bool submitted = (timeline->value_known &&
                timeline->value >= wait.value) ||
            timeline->signals.lower_bound(wait.value) != timeline->signals.end();
        if (!submitted && timeline->value_known)
            for (std::uint32_t signal = 0U;
                 signal < a.pInfo->signalCount; ++signal)
                if (a.pInfo->pSignals[signal].semaphore == wait.semaphore)
                    return PM_ERROR_INVALID_ARGUMENT;
    }
    for (std::uint32_t index = 0U; index < a.pInfo->signalCount; ++index) {
        const auto &signal = a.pInfo->pSignals[index];
        const auto *timeline = semaphore(queue->second.device, signal.semaphore);
        if (!timeline || signal.stages != PM_STAGE_ALL_COMMANDS_BIT)
            return PM_ERROR_INVALID_ARGUMENT;
        auto greatest = timeline->greatest_value;
        for (std::uint32_t previous = 0U; previous < index; ++previous)
            if (a.pInfo->pSignals[previous].semaphore == signal.semaphore)
                greatest = std::max(greatest,
                    a.pInfo->pSignals[previous].value);
        if (signal.value <= greatest) return PM_ERROR_INVALID_ARGUMENT;
    }
    for (std::uint32_t index = 0U;
         index < a.pInfo->commandBufferCount; ++index) {
        const auto handle = a.pInfo->pCommandBuffers[index];
        const auto *buffer = command_buffer(handle);
        if (!buffer || buffer->device != queue->second.device)
            return PM_ERROR_INVALID_ARGUMENT;
        for (std::uint32_t previous = 0U; previous < index; ++previous)
            if (a.pInfo->pCommandBuffers[previous] == handle)
                return PM_ERROR_INVALID_ARGUMENT;
        if (buffer->state != CommandState::executable)
            return PM_ERROR_INVALID_STATE;
        const auto *pool = command_pool(buffer->device, buffer->pool);
        if (!pool || pool->queue_family != queue->second.family)
            return PM_ERROR_INVALID_ARGUMENT;
    }
    if (a.fence) {
        const auto *tracked_fence = fence(queue->second.device, a.fence);
        if (!tracked_fence) return PM_ERROR_INVALID_ARGUMENT;
        if (tracked_fence->state != FenceState::unsignaled)
            return PM_ERROR_INVALID_STATE;
    }
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmQueueSubmit &a) {
    const auto queue = queues.find(a.queue);
    if (queue == queues.end()) return;
    if (!a.fence && !a.pInfo->commandBufferCount && !a.pInfo->signalCount)
        return;
    auto submission = next_submission++;
    if (!submission) submission = next_submission++;
    submissions.emplace(submission, Submission{queue->second.device, a.queue});
    for (std::uint32_t index = 0U;
         index < a.pInfo->commandBufferCount; ++index)
        if (auto *buffer = command_buffer(a.pInfo->pCommandBuffers[index])) {
            buffer->state = CommandState::pending;
            buffer->pending_submission = submission;
        }
    for (std::uint32_t index = 0U;
         index < a.pInfo->signalCount; ++index) {
        const auto &signal = a.pInfo->pSignals[index];
        if (auto *timeline = semaphore(queue->second.device,
                                      signal.semaphore)) {
            timeline->greatest_value = std::max(
                timeline->greatest_value, signal.value);
            timeline->signals.emplace(signal.value, submission);
        }
    }
    if (a.fence) {
        if (auto *tracked_fence = fence(queue->second.device, a.fence)) {
            tracked_fence->state = FenceState::pending;
            tracked_fence->pending_submission = submission;
        }
    }
}
void record(const PmLayerArgs_pmQueueWaitIdle &a) { complete_queue(a.queue); }
void record(const PmLayerArgs_pmDeviceWaitIdle &a) { complete_device(a.device); }

PmResult validate(const PmLayerArgs_pmCreateFence &a) {
    return a.pFence && (a.flags & ~PM_FENCE_SIGNALED_BIT) == 0U
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCreateFence &a) {
    if (a.pFence && *a.pFence)
        fences[*a.pFence] = {a.device, 0U,
            (a.flags & PM_FENCE_SIGNALED_BIT) != 0U
                ? FenceState::signaled : FenceState::unsignaled};
}
PmResult validate(const PmLayerArgs_pmDestroyFence &a) {
    return !a.fence || !tracking_available || fence(a.device, a.fence)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmDestroyFence &a) {
    complete_fence(a.fence);
    fences.erase(a.fence);
}
PmResult validate(const PmLayerArgs_pmGetFenceStatus &a) {
    return !tracking_available || fence(a.device, a.fence)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
PmResult validate(const PmLayerArgs_pmResetFences &a) {
    if (!a.count || !a.pFences) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    for (std::uint32_t index = 0U; index < a.count; ++index) {
        const auto *tracked_fence = fence(a.device, a.pFences[index]);
        if (!tracked_fence) return PM_ERROR_INVALID_ARGUMENT;
        if (tracked_fence->state == FenceState::pending)
            return PM_ERROR_INVALID_STATE;
    }
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmResetFences &a) {
    for (std::uint32_t index = 0U; index < a.count; ++index)
        if (auto *tracked_fence = fence(a.device, a.pFences[index]))
            tracked_fence->state = FenceState::unsignaled;
}
PmResult validate(const PmLayerArgs_pmWaitForFences &a) {
    if (!a.count || !a.pFences ||
        (a.waitAll != PM_TRUE && a.waitAll != PM_FALSE))
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    for (std::uint32_t index = 0U; index < a.count; ++index)
        if (!fence(a.device, a.pFences[index]))
            return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}

PmResult validate(const PmLayerArgs_pmAllocateMemory &a) {
    return a.pInfo && a.pMemory &&
        a.pInfo->sType == PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO &&
        !a.pInfo->pNext && a.pInfo->allocationSize != 0
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmAllocateMemory &a) {
    if (*a.pMemory)
        devices[a.device].memories[*a.pMemory] = {a.pInfo->allocationSize, a.pInfo->memoryTypeIndex};
}
PmResult validate(const PmLayerArgs_pmCreateBuffer &a) {
    return a.pInfo && a.pBuffer &&
        a.pInfo->sType == PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO &&
        !a.pInfo->pNext && a.pInfo->size != 0
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCreateBuffer &a) {
    if (*a.pBuffer) devices[a.device].buffers[*a.pBuffer] = {a.pInfo->size};
}
void record(const PmLayerArgs_pmImportNativeBuffer &a) {
    if (a.pInfo && a.pBuffer && *a.pBuffer)
        devices[a.device].buffers[*a.pBuffer] = {a.pInfo->size, nullptr, true};
}
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
void record(const PmLayerArgs_pmImportVulkanBufferEXT &a) {
    if (a.pInfo && a.pBuffer && *a.pBuffer)
        devices[a.device].buffers[*a.pBuffer] = {a.pInfo->size, nullptr, true};
}
#endif
PmResult validate(const PmLayerArgs_pmGetBufferMemoryRequirements &a) {
    return a.pRequirements && (!tracking_available || buffer(a.device, a.buffer))
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmGetBufferMemoryRequirements &a) {
    if (auto *b = buffer(a.device, a.buffer)) b->requirements = *a.pRequirements;
}
PmResult validate(const PmLayerArgs_pmBindBufferMemory &a) {
    if (!tracking_available) return PM_SUCCESS;
    const auto *b = buffer(a.device, a.buffer);
    const auto *m = memory(a.device, a.memory);
    if (!b || !m) return PM_ERROR_INVALID_ARGUMENT;
    if (b->bound) return PM_ERROR_INVALID_STATE;
    const auto &r = b->requirements;
    const auto size = r.size ? r.size : b->size;
    if (!fits(a.offset, size, m->size) ||
        (r.alignment && a.offset % r.alignment != 0) ||
        (r.memoryTypeBits && (m->type >= 32 || !(r.memoryTypeBits & (1U << m->type)))))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmBindBufferMemory &a) {
    if (auto *b = buffer(a.device, a.buffer)) { b->memory = a.memory; b->bound = true; }
}
PmResult validate(const PmLayerArgs_pmMapMemory &a) {
    if (!a.ppData) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *m = memory(a.device, a.memory);
    if (!m || !range_fits(a.offset, a.size, m->size)) return PM_ERROR_INVALID_ARGUMENT;
    return m->mapped ? PM_ERROR_INVALID_STATE : PM_SUCCESS;
}
void record(const PmLayerArgs_pmMapMemory &a) {
    if (auto *m = memory(a.device, a.memory)) m->mapped = true;
}
PmResult validate(const PmLayerArgs_pmUnmapMemory &a) {
    return !tracking_available || memory(a.device, a.memory)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmUnmapMemory &a) {
    if (auto *m = memory(a.device, a.memory)) m->mapped = false;
}
PmResult ranges(PmDevice device, uint32_t count, const PmMappedMemoryRange *values,
                bool require_mapped) {
    if (count && !values) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    for (uint32_t i = 0; i < count; ++i) {
        const auto *m = memory(device, values[i].memory);
        if (!m || (require_mapped && !m->mapped) ||
            !range_fits(values[i].offset, values[i].size, m->size))
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}
PmResult validate(const PmLayerArgs_pmFlushMappedMemoryRanges &a) {
    return ranges(a.device, a.count, a.pRanges, false);
}
PmResult validate(const PmLayerArgs_pmInvalidateMappedMemoryRanges &a) {
    return ranges(a.device, a.count, a.pRanges, true);
}
PmResult validate(const PmLayerArgs_pmDestroyBuffer &a) {
    return !a.buffer || !tracking_available || buffer(a.device, a.buffer)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmDestroyBuffer &a) {
    if (const auto d = devices.find(a.device); d != devices.end())
        d->second.buffers.erase(a.buffer);
}
PmResult validate(const PmLayerArgs_pmFreeMemory &a) {
    if (!a.memory || !tracking_available) return PM_SUCCESS;
    if (!memory(a.device, a.memory)) return PM_ERROR_INVALID_ARGUMENT;
    for (const auto &[handle, b] : devices.at(a.device).buffers) {
        (void)handle;
        if (b.memory == a.memory) return PM_ERROR_INVALID_STATE;
    }
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmFreeMemory &a) {
    if (const auto d = devices.find(a.device); d != devices.end())
        d->second.memories.erase(a.memory);
}
void record(const PmLayerArgs_pmDestroyDevice &a) {
    complete_device(a.device, PM_ERROR_DEVICE_LOST);
    for (auto found = command_buffers.begin(); found != command_buffers.end();)
        if (found->second.device == a.device) found = command_buffers.erase(found);
        else ++found;
    for (auto found = command_pools.begin(); found != command_pools.end();)
        if (found->second.device == a.device) found = command_pools.erase(found);
        else ++found;
    for (auto found = queues.begin(); found != queues.end();)
        if (found->second.device == a.device) found = queues.erase(found);
        else ++found;
    for (auto found = fences.begin(); found != fences.end();)
        if (found->second.device == a.device) found = fences.erase(found);
        else ++found;
    for (auto found = semaphores.begin(); found != semaphores.end();)
        if (found->second.device == a.device) found = semaphores.erase(found);
        else ++found;
    devices.erase(a.device);
}

bool completion_result(PmResult result) {
    return result == PM_SUCCESS || result < PM_SUCCESS;
}
template<class T> void record_result(const T &a, PmResult result) {
    if (result == PM_SUCCESS) record(a);
}
void record_result(const PmLayerArgs_pmGetFenceStatus &a, PmResult result) {
    if (completion_result(result)) complete_fence(a.fence, result);
}
void record_result(const PmLayerArgs_pmWaitForFences &a, PmResult result) {
    if (!completion_result(result)) return;
    // A successful wait-all identifies every completed fence. A wait-any (or
    // failed multi-wait) does not identify which fence completed, so require a
    // later status query before allowing its command buffers to be reused.
    if ((result == PM_SUCCESS && a.waitAll == PM_TRUE) || a.count == 1U)
        for (std::uint32_t index = 0U; index < a.count; ++index)
            complete_fence(a.pFences[index], result);
}
void record_result(const PmLayerArgs_pmQueueWaitIdle &a, PmResult result) {
    if (completion_result(result)) complete_queue(a.queue, result);
}
void record_result(const PmLayerArgs_pmDeviceWaitIdle &a, PmResult result) {
    if (completion_result(result)) complete_device(a.device, result);
}
void record_result(const PmLayerArgs_pmGetSemaphoreCounterValue &a,
                   PmResult result) {
    if (result == PM_SUCCESS && a.pValue)
        observe_timeline_value(a.device, a.semaphore, *a.pValue);
}
void record_result(const PmLayerArgs_pmWaitSemaphore &a, PmResult result) {
    if (completion_result(result))
        complete_timeline_through(a.device, a.semaphore, a.value, result);
}

template<class Function>
PmResult dispatch(const PmLayerCallInfo &call, Function function) {
    if (!call.pCommandName || !call.pCommandName[0] || !call.primaryHandle || !call.pArguments)
        return PM_ERROR_INVALID_ARGUMENT;
#include "layer_dispatch.inc"
    // An older layer must allow commands added by a newer compatible loader.
    return PM_SUCCESS;
}
} // namespace

PM_LAYER_EXPORT PmResult PMAPI_CALL
pm_layerNegotiateLoaderInterfaceVersion(uint32_t *version) {
    if (!version) return PM_ERROR_INVALID_ARGUMENT;
    // A v1 loader cannot supply the typed payload required by this layer.
    if (*version < 2U) return PM_ERROR_INCOMPATIBLE_VERSION;
    *version = PM_LAYER_INTERFACE_VERSION;
    return PM_SUCCESS;
}

PM_LAYER_EXPORT PmResult PMAPI_CALL pm_layerPreCall(const PmLayerCallInfo *call) {
    if (!call) return PM_ERROR_INVALID_ARGUMENT;
    std::lock_guard lock(mutex);
    if (tracking_available && call->pCommandName &&
        std::strncmp(call->pCommandName, "pmCmd", 5U) == 0) {
        if (!call->pArguments || call->argumentSize < sizeof(PmCommandBuffer))
            return PM_ERROR_INVALID_ARGUMENT;
        PmCommandBuffer handle{};
        // Every typed pmCmd* payload begins with its application-visible
        // command buffer. primaryHandle is the loader's native dispatch key.
        std::memcpy(&handle, call->pArguments, sizeof handle);
        const auto *buffer = command_buffer(handle);
        if (!buffer) return PM_ERROR_INVALID_ARGUMENT;
        if (buffer->state != CommandState::recording)
            return PM_ERROR_INVALID_STATE;
    }
    return dispatch(*call, [](const auto &a) { return validate(a); });
}

PM_LAYER_EXPORT void PMAPI_CALL pm_layerPostCall(
    const PmLayerCallInfo *call, PmResult result) {
    if (!call) return;
    std::lock_guard lock(mutex);
    if (!tracking_available) return;
    try {
        (void)dispatch(*call, [result](const auto &a) {
            record_result(a, result);
            return PM_SUCCESS;
        });
    } catch (const std::bad_alloc &) {
        // A diagnostic must not unwind across the C ABI or turn a successful
        // resource creation into a failure that leaks the driver's resource.
        devices.clear();
        command_pools.clear();
        command_buffers.clear();
        queues.clear();
        fences.clear();
        submissions.clear();
        semaphores.clear();
        tracking_available = false;
        std::fputs("Parallel Mater validation: host memory exhausted; object tracking disabled\n", stderr);
    }
}
