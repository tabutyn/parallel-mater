// SPDX-License-Identifier: MIT
#include "internal.hpp"

#include <algorithm>
#include <map>
#include <thread>
#include <vector>

namespace {
using Completion = parallel_mater::device::SubmissionCompletion;
using CompletionPtr = std::shared_ptr<Completion>;

void refresh_timeline_locked(PmSemaphore semaphore) noexcept {
    while (!semaphore->signals.empty()) {
        const auto first = semaphore->signals.begin();
        if (first->second->status() != PM_SUCCESS) break;
        semaphore->value = std::max(semaphore->value, first->first);
        semaphore->signals.erase(first);
    }
}

PmResult timeline_status(PmSemaphore semaphore, std::uint64_t value) noexcept {
    std::lock_guard lock(semaphore->mutex);
    refresh_timeline_locked(semaphore);
    if (semaphore->value >= value) return PM_SUCCESS;
    const auto found = semaphore->signals.lower_bound(value);
    if (found == semaphore->signals.end()) return PM_NOT_READY;
    const PmResult result = found->second->status();
    if (result == PM_SUCCESS) semaphore->value = found->first;
    return result;
}

PmResult wait_timeline(PmSemaphore semaphore, std::uint64_t value,
                       std::uint64_t timeout) noexcept {
    const auto started = std::chrono::steady_clock::now();
    for (;;) {
        const PmResult result = timeline_status(semaphore, value);
        if (result != PM_NOT_READY) return result;
        if (timeout != PM_TIMEOUT_INFINITE) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count();
            if (static_cast<std::uint64_t>(elapsed) >= timeout) return PM_TIMEOUT;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

struct Submission {
    std::vector<std::function<PmResult(parallel_mater::device::Services &)>> commands;
    std::vector<PmSemaphoreSubmitInfo> waits;
    std::vector<CompletionPtr> signal_dependencies;
    std::vector<PmQueryWrite> queries;
    std::vector<PmSimulation> simulations;
#if defined(PARALLEL_MATER_TEST_FAILURE_DRIVER)
    bool delay_execution{};
#endif

    [[nodiscard]] bool ready() const noexcept {
        for (const auto &wait : waits)
            if (timeline_status(wait.semaphore, wait.value) == PM_NOT_READY)
                return false;
        for (const auto &dependency : signal_dependencies)
            if (dependency->status() == PM_NOT_READY) return false;
        return true;
    }

    [[nodiscard]] PmResult execute(parallel_mater::device::Services &services) noexcept {
#if defined(PARALLEL_MATER_TEST_FAILURE_DRIVER)
        if (delay_execution)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
#endif
        PmResult result = services.activate();
        for (const auto &wait : waits)
            if (result == PM_SUCCESS)
                result = timeline_status(wait.semaphore, wait.value);
        for (const auto &dependency : signal_dependencies)
            if (result == PM_SUCCESS) result = dependency->status();
        try {
            for (auto &command : commands) {
                if (result != PM_SUCCESS) break;
                result = command(services);
            }
        } catch (const std::bad_alloc &) {
            result = PM_ERROR_OUT_OF_HOST_MEMORY;
        } catch (...) {
            result = PM_ERROR_INVALID_STATE;
        }
        // Partial execution still owns all referenced storage until the
        // native queue drains. Never signal a failed submission prematurely.
        const PmResult drained = services.wait_idle();
        if (result == PM_SUCCESS) result = drained;
        for (PmSimulation simulation : simulations) {
            if (result != PM_SUCCESS) simulation->execution_result.store(result);
            simulation->pending_submissions.fetch_sub(1U, std::memory_order_release);
        }
        return result;
    }
};

struct PreparedSignal {
    PmSemaphore semaphore{};
    std::map<std::uint64_t, CompletionPtr>::node_type node;
};
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
    (void)pmDeviceWaitIdle(device);
    parallel_mater::api::destroy_object(device, semaphore);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetSemaphoreCounterValue(
    PmDevice device, PmSemaphore semaphore, std::uint64_t *value) {
    if (device == nullptr || semaphore == nullptr || value == nullptr ||
        semaphore->device != device)
        return PM_ERROR_INVALID_ARGUMENT;
    std::lock_guard lock(semaphore->mutex);
    refresh_timeline_locked(semaphore);
    if (!semaphore->signals.empty()) {
        const auto result = semaphore->signals.begin()->second->status();
        if (result < PM_SUCCESS) return result;
    }
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
    std::lock_guard queue_lock(queue->mutex);
    if (fence != nullptr && (fence->submitted || fence->initially_signaled))
        return PM_ERROR_INVALID_STATE;
    for (std::uint32_t index = 0U; index < info->commandBufferCount; ++index) {
        const auto buffer = info->pCommandBuffers[index];
        parallel_mater::api::refresh_command_state(buffer);
        if (buffer == nullptr || buffer->device != queue->device ||
            buffer->state != PmCommandState::executable)
            return PM_ERROR_INVALID_STATE;
        for (std::uint32_t previous = 0U; previous < index; ++previous)
            if (info->pCommandBuffers[previous] == buffer)
                return PM_ERROR_INVALID_ARGUMENT;
    }
    for (std::uint32_t index = 0U; index < info->waitCount; ++index) {
        const auto &wait = info->pWaits[index];
        if (wait.semaphore == nullptr || wait.semaphore->device != queue->device)
            return PM_ERROR_INVALID_ARGUMENT;
        // An unresolved wait cannot precede a signal on the same timeline.
        // Even a smaller signal value would prevent its future producer from
        // satisfying strict signal ordering.
        std::lock_guard lock(wait.semaphore->mutex);
        const bool submitted = wait.semaphore->value >= wait.value ||
            wait.semaphore->signals.lower_bound(wait.value) != wait.semaphore->signals.end();
        if (!submitted)
            for (std::uint32_t signal = 0U; signal < info->signalCount; ++signal)
                if (info->pSignals[signal].semaphore == wait.semaphore)
                    return PM_ERROR_INVALID_ARGUMENT;
    }
    for (std::uint32_t index = 0U; index < info->signalCount; ++index) {
        const auto &signal = info->pSignals[index];
        if (signal.semaphore == nullptr || signal.semaphore->device != queue->device ||
            signal.stages != PM_STAGE_ALL_COMMANDS_BIT)
            return PM_ERROR_INVALID_ARGUMENT;
        std::lock_guard lock(signal.semaphore->mutex);
        refresh_timeline_locked(signal.semaphore);
        auto greatest = signal.semaphore->value;
        if (!signal.semaphore->signals.empty())
            greatest = std::max(greatest, signal.semaphore->signals.rbegin()->first);
        for (std::uint32_t previous = 0U; previous < index; ++previous)
            if (info->pSignals[previous].semaphore == signal.semaphore)
                greatest = std::max(greatest, info->pSignals[previous].value);
        if (signal.value <= greatest) return PM_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto submission = std::make_shared<Submission>();
        auto completion = std::make_shared<Completion>();
#if defined(PARALLEL_MATER_TEST_FAILURE_DRIVER)
        submission->delay_execution =
            std::getenv("PM_TEST_DELAY_SUBMISSION") != nullptr;
#endif
        if (info->waitCount != 0U)
            submission->waits.assign(info->pWaits, info->pWaits + info->waitCount);
        for (std::uint32_t index = 0U; index < info->commandBufferCount; ++index) {
            const auto buffer = info->pCommandBuffers[index];
            submission->commands.insert(submission->commands.end(),
                buffer->commands.begin(), buffer->commands.end());
            submission->queries.insert(submission->queries.end(),
                buffer->query_writes.begin(), buffer->query_writes.end());
            for (auto simulation : buffer->simulation_resources)
                if (std::find(submission->simulations.begin(), submission->simulations.end(), simulation) ==
                    submission->simulations.end()) submission->simulations.push_back(simulation);
        }
        std::vector<PreparedSignal> prepared;
        prepared.reserve(info->signalCount);
        for (std::uint32_t index = 0U; index < info->signalCount; ++index) {
            const auto &signal = info->pSignals[index];
            std::map<std::uint64_t, CompletionPtr> holder;
            holder.emplace(signal.value, completion);
            prepared.push_back({signal.semaphore, holder.extract(signal.value)});
            std::lock_guard lock(signal.semaphore->mutex);
            if (!signal.semaphore->signals.empty())
                submission->signal_dependencies.push_back(signal.semaphore->signals.rbegin()->second);
        }
        return queue->submissions->enqueue(
            [submission, services = queue->device->services.get()]() noexcept {
                return submission->execute(*services);
            },
            [submission]() noexcept { return submission->ready(); }, completion,
            [&]() noexcept {
                for (std::uint32_t index = 0U; index < info->commandBufferCount; ++index) {
                    auto buffer = info->pCommandBuffers[index];
                    buffer->completion = completion;
                    buffer->state = PmCommandState::pending;
                }
                for (auto simulation : submission->simulations)
                    simulation->pending_submissions.fetch_add(1U);
                for (const auto &write : submission->queries) {
                    for (std::uint32_t slot = write.first; slot < write.first + write.count; ++slot)
                        std::atomic_store(&write.pool->slots[slot].completion, completion);
                }
                for (auto &signal : prepared) {
                    std::lock_guard lock(signal.semaphore->mutex);
                    signal.semaphore->signals.insert(std::move(signal.node));
                }
                if (fence != nullptr) {
                    fence->completion = completion;
                    fence->submitted = true;
                }
            });
    } catch (const std::bad_alloc &) {
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmQueueWaitIdle(PmQueue queue) {
    if (queue == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    const auto result = queue->submissions->wait_idle();
    const auto native = queue->device->services->wait_idle();
    return result == PM_SUCCESS ? native : result;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmDeviceWaitIdle(PmDevice device) {
    return device == nullptr ? PM_ERROR_INVALID_ARGUMENT : pmQueueWaitIdle(&device->queue);
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
    created->initially_signaled = (flags & PM_FENCE_SIGNALED_BIT) != 0U;
    *fence = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyFence(
    PmDevice device, PmFence fence) {
    if (device == nullptr || fence == nullptr || fence->device != device) return;
    if (fence->completion) (void)fence->completion->wait();
    parallel_mater::api::destroy_object(device, fence);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetFenceStatus(
    PmDevice device, PmFence fence) {
    if (device == nullptr || fence == nullptr || fence->device != device)
        return PM_ERROR_INVALID_ARGUMENT;
    if (fence->initially_signaled) return PM_SUCCESS;
    if (!fence->submitted) return PM_NOT_READY;
    return fence->completion->status();
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
        fences[index]->completion.reset();
        fences[index]->submitted = false;
        fences[index]->initially_signaled = false;
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
