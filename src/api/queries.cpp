// SPDX-License-Identifier: MIT
#include "internal.hpp"

#include <cstring>

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateQueryPool(
    PmDevice device, const PmQueryPoolCreateInfo *info, PmQueryPool *pool) {
    if (pool == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *pool = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, info->pNext) ||
        info->queryCount == 0U || info->queryCount > 65536U)
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmQueryPool_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->type = info->queryType;
    try {
        created->slots.resize(info->queryCount);
    } catch (const std::bad_alloc &) {
        parallel_mater::api::destroy_object(device, created);
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (info->queryType == PM_QUERY_TIMESTAMP) {
        for (PmQuerySlot &slot : created->slots) {
            const PmResult result = device->services->create_event(true, slot.event);
            if (result != PM_SUCCESS) {
                for (PmQuerySlot &rollback : created->slots)
                    if (rollback.event)
                        device->services->destroy_event(rollback.event);
                parallel_mater::api::destroy_object(device, created);
                return result;
            }
        }
    } else if (info->queryType != PM_QUERY_STEP_STATISTICS) {
        parallel_mater::api::destroy_object(device, created);
        return PM_ERROR_INVALID_ARGUMENT;
    }
    *pool = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyQueryPool(
    PmDevice device, PmQueryPool pool) {
    if (device == nullptr || pool == nullptr || pool->device != device) return;
    (void)pmDeviceWaitIdle(device);
    for (PmQuerySlot &slot : pool->slots)
        if (slot.event) device->services->destroy_event(slot.event);
    parallel_mater::api::destroy_object(device, pool);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdResetQueryPool(
    PmCommandBuffer command_buffer, PmQueryPool pool, std::uint32_t first,
    std::uint32_t count) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (pool == nullptr || pool->device != command_buffer->device || count == 0U ||
        first > pool->slots.size() || count > pool->slots.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    return parallel_mater::api::record_query(
        command_buffer, pool, first, count, [pool, first, count](
                            parallel_mater::device::Services &) noexcept {
            std::lock_guard lock(pool->mutex);
            for (std::uint32_t index = first; index < first + count; ++index) {
                pool->slots[index].available = false;
                pool->slots[index].statistics = {};
            }
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdWriteTimestamp(
    PmCommandBuffer command_buffer, PmStageFlags stage, PmQueryPool pool,
    std::uint32_t query) {
    (void)stage;
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (pool == nullptr || pool->device != command_buffer->device ||
        pool->type != PM_QUERY_TIMESTAMP || query >= pool->slots.size())
        return PM_ERROR_INVALID_ARGUMENT;
    return parallel_mater::api::record_query(
        command_buffer, pool, query, 1U, [pool, query](
                            parallel_mater::device::Services &services) noexcept {
            std::lock_guard lock(pool->mutex);
            const PmResult result = services.record_event(pool->slots[query].event);
            if (result != PM_SUCCESS) return result;
            pool->slots[query].available = true;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdBeginStatisticsQuery(
    PmCommandBuffer command_buffer, PmQueryPool pool, std::uint32_t query) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (pool == nullptr || pool->device != command_buffer->device ||
        pool->type != PM_QUERY_STEP_STATISTICS || query >= pool->slots.size() ||
        command_buffer->statistics_open)
        return PM_ERROR_INVALID_ARGUMENT;
    command_buffer->statistics_open = true;
    command_buffer->statistics_pool = pool;
    command_buffer->statistics_query = query;
    command_buffer->statistics_simulation = nullptr;
    try {
        command_buffer->statistics_execution = std::make_shared<bool>(false);
    } catch (const std::bad_alloc &) {
        command_buffer->statistics_open = false;
        command_buffer->statistics_pool = nullptr;
        command_buffer->latched_error = PM_ERROR_OUT_OF_HOST_MEMORY;
        return command_buffer->latched_error;
    }
    const auto execution = command_buffer->statistics_execution;
    return parallel_mater::api::record_query(
        command_buffer, pool, query, 1U, [pool, query, execution](
                            parallel_mater::device::Services &) noexcept {
            std::lock_guard lock(pool->mutex);
            pool->slots[query].statistics = {};
            pool->slots[query].available = false;
            *execution = true;
            return PM_SUCCESS;
        });
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCmdEndStatisticsQuery(
    PmCommandBuffer command_buffer, PmQueryPool pool, std::uint32_t query) {
    const PmResult state = parallel_mater::api::require_recording(command_buffer);
    if (state != PM_SUCCESS) return state;
    if (pool == nullptr || pool->device != command_buffer->device ||
        pool->type != PM_QUERY_STEP_STATISTICS || query >= pool->slots.size() ||
        !command_buffer->statistics_open ||
        command_buffer->statistics_pool != pool ||
        command_buffer->statistics_query != query ||
        command_buffer->statistics_simulation == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    const PmSimulation simulation = command_buffer->statistics_simulation;
    const auto execution = command_buffer->statistics_execution;
    command_buffer->statistics_open = false;
    command_buffer->statistics_pool = nullptr;
    command_buffer->statistics_simulation = nullptr;
    command_buffer->statistics_execution.reset();
    return parallel_mater::api::record_query(
        command_buffer, pool, query, 1U, [pool, query, simulation, execution](
                            parallel_mater::device::Services &services) noexcept {
            if (!*execution) return PM_SUCCESS;
            std::lock_guard lock(pool->mutex);
            const parallel_mater::device::ExecutionQueue queue = parallel_mater::device::execution_queue(services);
            parallel_mater::SimulationStatistics source{};
            const PmResult result = parallel_mater::api::from_status(
                simulation->runtime->collect_statistics(source, queue));
            if (result != PM_SUCCESS) return result;
            PmStepStatistics &statistics = pool->slots[query].statistics;
            statistics.stepCount = simulation->submitted_steps;
            statistics.substepsExecuted = simulation->last_substeps;
            statistics.poseIterations = source.rigid_contact_pose_passes;
            statistics.impactIterations = source.rigid_contact_impact_passes;
            statistics.peakContactCount = source.contact_count;
            statistics.peakActiveBodies = source.rigid_body_count;
            statistics.peakAwakeBodies = source.rigid_body_count -
                source.sleeping_rigid_body_count;
            statistics.peakParticleCount = source.particle_count;
            statistics.scratchHighWaterBytes = simulation->arena != nullptr
                ? simulation->arena->scratch_used() : 0U;
            statistics.overflowCount = source.contact_overflow_count +
                source.rigid_contact_overflow_pairs +
                source.fluid_neighbor_overflow_count;
            statistics.executionResult = simulation->execution_result;
            statistics.earlyExitCount =
                source.rigid_contact_early_exit_count;
            pool->slots[query].available = true;
            return PM_SUCCESS;
        }, true);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetQueryPoolResults(
    PmDevice device, PmQueryPool pool, std::uint32_t first,
    std::uint32_t count, std::size_t data_size, void *data,
    PmDeviceSize stride, PmFlags flags) {
    if (device == nullptr || pool == nullptr || pool->device != device ||
        count == 0U || data == nullptr || first > pool->slots.size() ||
        count > pool->slots.size() - first)
        return PM_ERROR_INVALID_ARGUMENT;
    const std::size_t value_size = pool->type == PM_QUERY_TIMESTAMP
        ? sizeof(std::uint64_t) : sizeof(PmStepStatistics);
    const std::size_t record_size = value_size +
        ((flags & PM_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0U
             ? sizeof(std::uint64_t) : 0U);
    if (stride < record_size || data_size < record_size + (count - 1U) * stride)
        return PM_ERROR_INVALID_ARGUMENT;
    const auto unavailable = [&]() noexcept {
        for (std::uint32_t offset = 0U; offset < count; ++offset)
            std::memset(static_cast<std::byte *>(data) + offset * stride, 0, record_size);
        return PM_NOT_READY;
    };
    std::unique_lock lock(pool->mutex, std::defer_lock);
    if ((flags & PM_QUERY_RESULT_WAIT_BIT) != 0U) {
        lock.lock();
    } else if (!lock.try_lock()) {
        return unavailable();
    }
    bool incomplete = false;
    for (std::uint32_t offset = 0U; offset < count; ++offset) {
        PmQuerySlot &slot = pool->slots[first + offset];
        // Completion belongs to each slot, so a failed or pending write to
        // another slot cannot hide this result. Recheck after an unlocked wait
        // in case a later submission replaced this slot's pending write.
        auto completion = std::atomic_load(&slot.completion);
        while (completion && completion->status() == PM_NOT_READY &&
               (flags & PM_QUERY_RESULT_WAIT_BIT) != 0U) {
            lock.unlock();
            (void)completion->wait();
            lock.lock();
            completion = std::atomic_load(&slot.completion);
        }
        const auto result = completion ? completion->status() : PM_SUCCESS;
        const bool failed_statistics_available =
            result == PM_ERROR_CAPACITY_EXCEEDED &&
            pool->type == PM_QUERY_STEP_STATISTICS && slot.available &&
            slot.statistics.executionResult == PM_ERROR_CAPACITY_EXCEEDED;
        if (result < PM_SUCCESS && !failed_statistics_available) return result;
        bool available = (result == PM_SUCCESS || failed_statistics_available) &&
            slot.available;
        if (available && pool->type == PM_QUERY_TIMESTAMP) {
            if ((flags & PM_QUERY_RESULT_WAIT_BIT) != 0U) {
                const PmResult result = device->services->wait_event(slot.event);
                if (result != PM_SUCCESS) return result;
            } else {
                bool ready = false;
                const PmResult result = device->services->query_event(
                    slot.event, ready);
                if (result != PM_SUCCESS) return result;
                if (!ready) available = false;
            }
        }
        auto *destination = static_cast<std::byte *>(data) + offset * stride;
        if (available) {
            if (pool->type == PM_QUERY_TIMESTAMP) {
                float milliseconds = 0.0F;
                const PmResult result = device->services->elapsed_time(
                    device->services->epoch(), slot.event, milliseconds);
                if (result != PM_SUCCESS) return result;
                const std::uint64_t nanoseconds = static_cast<std::uint64_t>(
                    milliseconds * 1'000'000.0F);
                std::memcpy(destination, &nanoseconds, sizeof nanoseconds);
            } else {
                std::memcpy(destination, &slot.statistics, sizeof slot.statistics);
            }
        } else {
            std::memset(destination, 0, value_size);
            incomplete = true;
        }
        if ((flags & PM_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0U) {
            const std::uint64_t availability = available ? 1U : 0U;
            std::memcpy(destination + value_size, &availability, sizeof availability);
        }
    }
    return incomplete ? PM_NOT_READY : PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetDeviceMemoryBudget(
    PmDevice device, PmMemoryBudget *budget) {
    if (device == nullptr || budget == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *budget = {};
    PmDeviceSize available = 0U;
    PmDeviceSize total = 0U;
    const PmResult result = device->services->memory_budget(available, total);
    if (result != PM_SUCCESS) return result;
    budget->heapCount = 1U;
    budget->budget[0] = total;
    budget->usage[0] = total - available;
    budget->applicationAllocationBytes =
        device->application_bytes.load(std::memory_order_relaxed);
    budget->libraryHostBytes =
        device->library_host_bytes.load(std::memory_order_relaxed);
    return PM_SUCCESS;
}
