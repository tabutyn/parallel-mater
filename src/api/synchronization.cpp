// SPDX-License-Identifier: MIT
#include "internal.hpp"

#include <algorithm>
#include <map>
#include <thread>
#include <vector>

namespace {

void refresh(PmSemaphore semaphore) noexcept {
    if (semaphore == nullptr) return;
    std::lock_guard lock(semaphore->mutex);
    for (const auto &[value, event] : semaphore->events) {
        const cudaError_t status = cudaEventQuery(event);
        if (status == cudaSuccess)
            semaphore->value = std::max(semaphore->value, value);
        else if (status != cudaErrorNotReady)
            (void)cudaGetLastError();
    }
}

PmResult wait_timeline(PmSemaphore semaphore, std::uint64_t value,
                       std::uint64_t timeout) noexcept {
    if (semaphore == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    const auto started = std::chrono::steady_clock::now();
    std::unique_lock lock(semaphore->mutex);
    for (;;) {
        if (semaphore->value >= value) return PM_SUCCESS;
        const auto found = semaphore->events.lower_bound(value);
        if (found != semaphore->events.end()) {
            cudaEvent_t event = found->second;
            lock.unlock();
            if (timeout == PM_TIMEOUT_INFINITE) {
                const PmResult result = parallel_mater::device::from_cuda(
                    cudaEventSynchronize(event));
                if (result != PM_SUCCESS) return result;
            } else {
                for (;;) {
                    const cudaError_t status = cudaEventQuery(event);
                    if (status == cudaSuccess) break;
                    if (status != cudaErrorNotReady)
                        return parallel_mater::device::from_cuda(status);
                    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count();
                    if (static_cast<std::uint64_t>(elapsed) >= timeout)
                        return PM_TIMEOUT;
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
            }
            lock.lock();
            semaphore->value = std::max(semaphore->value, found->first);
            return PM_SUCCESS;
        }
        if (timeout == 0U) return PM_TIMEOUT;
        if (timeout == PM_TIMEOUT_INFINITE) {
            semaphore->changed.wait(lock);
        } else {
            const auto deadline = started + std::chrono::nanoseconds(timeout);
            if (semaphore->changed.wait_until(lock, deadline) ==
                std::cv_status::timeout)
                return PM_TIMEOUT;
        }
    }
}

PmResult reset_fence(PmFence fence) noexcept {
    if (fence == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    if (fence->event != nullptr) (void)cudaEventDestroy(fence->event);
    const cudaError_t error = cudaEventCreateWithFlags(
        &fence->event, cudaEventDisableTiming);
    if (error != cudaSuccess) return parallel_mater::device::from_cuda(error);
    fence->submitted = false;
    fence->initially_signaled = false;
    return PM_SUCCESS;
}

[[nodiscard]] bool has_timeline_dependency(PmSemaphore semaphore,
                                           std::uint64_t value) noexcept {
    std::lock_guard lock(semaphore->mutex);
    return semaphore->value >= value ||
        semaphore->events.lower_bound(value) != semaphore->events.end();
}

[[nodiscard]] PmResult enqueue_timeline_wait(PmSemaphore semaphore,
                                             std::uint64_t value,
                                             cudaStream_t stream) noexcept {
    cudaEvent_t event{};
    {
        std::lock_guard lock(semaphore->mutex);
        if (semaphore->value >= value) return PM_SUCCESS;
        const auto found = semaphore->events.lower_bound(value);
        if (found == semaphore->events.end()) return PM_ERROR_INVALID_STATE;
        event = found->second;
    }
    return parallel_mater::device::from_cuda(
        cudaStreamWaitEvent(stream, event, 0U));
}

[[nodiscard]] bool submit_signals_dependency(
    const PmSubmitInfo &info, PmSemaphore semaphore,
    std::uint64_t value) noexcept {
    for (std::uint32_t index = 0U; index < info.signalCount; ++index)
        if (info.pSignals[index].semaphore == semaphore &&
            info.pSignals[index].value >= value)
            return true;
    return false;
}

struct PreparedSignal {
    PmSemaphore semaphore{};
    std::uint64_t value{};
    cudaEvent_t event{};
    std::map<std::uint64_t, cudaEvent_t>::node_type node;
};

void destroy_prepared_signals(std::vector<PreparedSignal> &signals) noexcept {
    for (PreparedSignal &signal : signals)
        if (signal.event != nullptr) (void)cudaEventDestroy(signal.event);
}

} // namespace

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateSemaphore(
    PmDevice device, const PmSemaphoreCreateInfo *info, PmSemaphore *semaphore) {
    if (semaphore == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *semaphore = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, info->pNext))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmSemaphore_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->value = info->initialValue;
    *semaphore = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroySemaphore(
    PmDevice device, PmSemaphore semaphore) {
    if (device == nullptr || semaphore == nullptr || semaphore->device != device) return;
    (void)device->services->wait_idle();
    for (const auto &[value, event] : semaphore->events) {
        (void)value;
        (void)cudaEventDestroy(event);
    }
    parallel_mater::api::destroy_object(device, semaphore);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetSemaphoreCounterValue(
    PmDevice device, PmSemaphore semaphore, std::uint64_t *value) {
    if (device == nullptr || semaphore == nullptr || value == nullptr ||
        semaphore->device != device)
        return PM_ERROR_INVALID_ARGUMENT;
    refresh(semaphore);
    std::lock_guard lock(semaphore->mutex);
    *value = semaphore->value;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmWaitSemaphore(
    PmDevice device, PmSemaphore semaphore, std::uint64_t value,
    std::uint64_t timeout) {
    if (device == nullptr || semaphore == nullptr || semaphore->device != device)
        return PM_ERROR_INVALID_ARGUMENT;
    return wait_timeline(semaphore, value, timeout);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmQueueSubmit(
    PmQueue queue, const PmSubmitInfo *info, PmFence fence) {
    if (queue == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_SUBMIT_INFO, info->pNext) ||
        (info->waitCount != 0U && info->pWaits == nullptr) ||
        (info->commandBufferCount != 0U && info->pCommandBuffers == nullptr) ||
        (info->signalCount != 0U && info->pSignals == nullptr) ||
        (fence != nullptr && fence->device != queue->device))
        return PM_ERROR_INVALID_ARGUMENT;

    // Dependencies that already have CUDA events are queued on the stream.
    // Only a value not submitted anywhere yet needs a host-side wait; do that
    // before taking the queue lock so another thread can submit its producer.
    for (std::uint32_t index = 0U; index < info->waitCount; ++index) {
        const auto &wait = info->pWaits[index];
        if (wait.semaphore == nullptr || wait.semaphore->device != queue->device)
            return PM_ERROR_INVALID_ARGUMENT;
        if (!has_timeline_dependency(wait.semaphore, wait.value)) {
            if (submit_signals_dependency(*info, wait.semaphore, wait.value))
                return PM_ERROR_INVALID_ARGUMENT;
            const PmResult result = wait_timeline(
                wait.semaphore, wait.value, PM_TIMEOUT_INFINITE);
            if (result != PM_SUCCESS) return result;
        }
    }
    for (std::uint32_t index = 0U; index < info->signalCount; ++index)
        if (info->pSignals[index].semaphore == nullptr ||
            info->pSignals[index].semaphore->device != queue->device)
            return PM_ERROR_INVALID_ARGUMENT;

    std::lock_guard queue_lock(queue->mutex);
    for (std::uint32_t index = 0U; index < info->commandBufferCount; ++index) {
        const PmCommandBuffer buffer = info->pCommandBuffers[index];
        parallel_mater::api::refresh_command_state(buffer);
        if (buffer == nullptr || buffer->device != queue->device ||
            buffer->state != PmCommandState::executable)
            return PM_ERROR_INVALID_STATE;
        for (std::uint32_t previous = 0U; previous < index; ++previous)
            if (info->pCommandBuffers[previous] == buffer)
                return PM_ERROR_INVALID_ARGUMENT;
    }
    if (fence != nullptr && (fence->submitted || fence->initially_signaled))
        return PM_ERROR_INVALID_STATE;

    // Validate strict timeline ordering against both completed and pending
    // values before any command in this submission can reach the GPU.
    for (std::uint32_t index = 0U; index < info->signalCount; ++index) {
        const auto &signal = info->pSignals[index];
        std::uint64_t greatest{};
        {
            std::lock_guard lock(signal.semaphore->mutex);
            greatest = signal.semaphore->value;
            if (!signal.semaphore->events.empty())
                greatest = std::max(greatest,
                                    signal.semaphore->events.rbegin()->first);
        }
        for (std::uint32_t previous = 0U; previous < index; ++previous)
            if (info->pSignals[previous].semaphore == signal.semaphore)
                greatest = std::max(greatest, info->pSignals[previous].value);
        if (signal.value <= greatest) return PM_ERROR_INVALID_ARGUMENT;
    }

    std::vector<PreparedSignal> prepared;
    try {
        prepared.reserve(info->signalCount);
        for (std::uint32_t index = 0U; index < info->signalCount; ++index) {
            const auto &signal = info->pSignals[index];
            std::map<std::uint64_t, cudaEvent_t> holder;
            holder.emplace(signal.value, nullptr);
            PreparedSignal item{};
            item.semaphore = signal.semaphore;
            item.value = signal.value;
            item.node = holder.extract(signal.value);
            const cudaError_t error = cudaEventCreateWithFlags(
                &item.event, cudaEventDisableTiming);
            if (error != cudaSuccess) {
                destroy_prepared_signals(prepared);
                return parallel_mater::device::from_cuda(error);
            }
            item.node.mapped() = item.event;
            prepared.push_back(std::move(item));
        }
    } catch (const std::bad_alloc &) {
        destroy_prepared_signals(prepared);
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }

    cudaStream_t stream = queue->device->services->stream();
    for (std::uint32_t index = 0U; index < info->waitCount; ++index) {
        const auto &wait = info->pWaits[index];
        const PmResult result = enqueue_timeline_wait(
            wait.semaphore, wait.value, stream);
        if (result != PM_SUCCESS) {
            destroy_prepared_signals(prepared);
            return result;
        }
    }
    for (std::uint32_t index = 0U; index < info->commandBufferCount; ++index) {
        PmCommandBuffer buffer = info->pCommandBuffers[index];
        buffer->state = PmCommandState::pending;
        for (auto &command : buffer->commands) {
            const PmResult result = command(stream);
            if (result != PM_SUCCESS) {
                // Some commands may already be in flight. Keep the buffer
                // pending until a completion marker reaches the stream.
                (void)cudaEventRecord(buffer->completion, stream);
                destroy_prepared_signals(prepared);
                return result;
            }
        }
        const cudaError_t completion = cudaEventRecord(buffer->completion, stream);
        if (completion != cudaSuccess) {
            destroy_prepared_signals(prepared);
            return parallel_mater::device::from_cuda(completion);
        }
    }
    for (PreparedSignal &signal : prepared) {
        const cudaError_t error = cudaEventRecord(signal.event, stream);
        if (error != cudaSuccess) {
            destroy_prepared_signals(prepared);
            return parallel_mater::device::from_cuda(error);
        }
    }
    if (fence != nullptr) {
        const cudaError_t error = cudaEventRecord(fence->event, stream);
        if (error != cudaSuccess) {
            destroy_prepared_signals(prepared);
            return parallel_mater::device::from_cuda(error);
        }
        fence->submitted = true;
    }
    for (PreparedSignal &signal : prepared) {
        {
            std::lock_guard lock(signal.semaphore->mutex);
            const auto inserted = signal.semaphore->events.insert(
                std::move(signal.node));
            if (!inserted.inserted) return PM_ERROR_INVALID_STATE;
        }
        signal.semaphore->changed.notify_all();
        signal.event = nullptr;
    }
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmQueueWaitIdle(PmQueue queue) {
    return queue == nullptr ? PM_ERROR_INVALID_ARGUMENT
                            : queue->device->services->wait_idle();
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmDeviceWaitIdle(PmDevice device) {
    return device == nullptr ? PM_ERROR_INVALID_ARGUMENT
                             : device->services->wait_idle();
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateFence(
    PmDevice device, PmFlags flags, PmFence *fence) {
    if (fence == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *fence = nullptr;
    if (device == nullptr || (flags & ~PM_FENCE_SIGNALED_BIT) != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmFence_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    const cudaError_t error = cudaEventCreateWithFlags(
        &created->event, cudaEventDisableTiming);
    if (error != cudaSuccess) {
        parallel_mater::api::destroy_object(device, created);
        return parallel_mater::device::from_cuda(error);
    }
    created->initially_signaled = (flags & PM_FENCE_SIGNALED_BIT) != 0U;
    *fence = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyFence(
    PmDevice device, PmFence fence) {
    if (device == nullptr || fence == nullptr || fence->device != device) return;
    if (fence->submitted) (void)cudaEventSynchronize(fence->event);
    (void)cudaEventDestroy(fence->event);
    parallel_mater::api::destroy_object(device, fence);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetFenceStatus(
    PmDevice device, PmFence fence) {
    if (device == nullptr || fence == nullptr || fence->device != device)
        return PM_ERROR_INVALID_ARGUMENT;
    if (fence->initially_signaled) return PM_SUCCESS;
    if (!fence->submitted) return PM_NOT_READY;
    const cudaError_t status = cudaEventQuery(fence->event);
    if (status == cudaSuccess) return PM_SUCCESS;
    if (status == cudaErrorNotReady) return PM_NOT_READY;
    return parallel_mater::device::from_cuda(status);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmResetFences(
    PmDevice device, std::uint32_t count, const PmFence *fences) {
    if (device == nullptr || count == 0U || fences == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < count; ++index)
        if (fences[index] == nullptr || fences[index]->device != device ||
            (fences[index]->submitted &&
             pmGetFenceStatus(device, fences[index]) == PM_NOT_READY))
            return PM_ERROR_INVALID_STATE;
    for (std::uint32_t index = 0U; index < count; ++index) {
        const PmResult result = reset_fence(fences[index]);
        if (result != PM_SUCCESS) return result;
    }
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmWaitForFences(
    PmDevice device, std::uint32_t count, const PmFence *fences,
    PmBool32 wait_all, std::uint64_t timeout) {
    if (device == nullptr || count == 0U || fences == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    const auto started = std::chrono::steady_clock::now();
    for (;;) {
        std::uint32_t ready = 0U;
        for (std::uint32_t index = 0U; index < count; ++index) {
            if (fences[index] == nullptr || fences[index]->device != device)
                return PM_ERROR_INVALID_ARGUMENT;
            const PmResult status = pmGetFenceStatus(device, fences[index]);
            if (status == PM_SUCCESS) ++ready;
            else if (status != PM_NOT_READY) return status;
        }
        if ((wait_all == PM_TRUE && ready == count) ||
            (wait_all == PM_FALSE && ready != 0U))
            return PM_SUCCESS;
        if (timeout == 0U) return PM_TIMEOUT;
        if (timeout != PM_TIMEOUT_INFINITE) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count();
            if (static_cast<std::uint64_t>(elapsed) >= timeout) return PM_TIMEOUT;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}
