// SPDX-License-Identifier: MIT
#include "internal.hpp"
#include "../simulation/realtime_schedule.hpp"
#include "../simulation/realtime_telemetry.hpp"

#include <cstring>

namespace {

[[nodiscard]] bool statistics_query(PmQueryType type) noexcept {
    return type == PM_QUERY_STEP_STATISTICS ||
        type == PM_QUERY_REALTIME_STEP_STATISTICS;
}

void copy_realtime_work(
    const parallel_mater::simulation::RealtimeWorkCounters &source,
    PmRealtimeWorkStatistics &destination) noexcept {
    destination.dispatches = source.dispatches;
    destination.workgroups = source.workgroups;
    destination.bytesMoved = source.bytes_moved;
    destination.spatialIndexBuilds = source.index_builds;
    destination.spatialIndexRefits = source.index_refits;
    destination.spatialIndexPublications = source.index_publications;
    destination.numericalIterations = source.numerical_iterations;
    destination.barriers = source.barriers;
    destination.queueSubmissions = source.submissions;
    destination.intraTickHostReadbacks = source.readbacks;
    destination.intraTickHostWaits = source.host_waits;
}

} // namespace

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateQueryPool(
    PmDevice device, const PmQueryPoolCreateInfo *info, PmQueryPool *pool) {
    if (pool == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *pool = nullptr;
    if (device == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, info->pNext) ||
        info->queryCount == 0U || info->queryCount > 65536U ||
        (info->queryType != PM_QUERY_TIMESTAMP &&
         info->queryType != PM_QUERY_STEP_STATISTICS &&
         info->queryType != PM_QUERY_REALTIME_STEP_STATISTICS))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmQueryPool_T>(device);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->device = device;
    created->type = info->queryType;
    try {
        created->slots.resize(info->queryCount);
        if (info->queryType == PM_QUERY_REALTIME_STEP_STATISTICS)
            created->realtime_statistics.resize(info->queryCount);
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
                if (pool->type == PM_QUERY_REALTIME_STEP_STATISTICS)
                    pool->realtime_statistics[index] = {};
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
        !statistics_query(pool->type) || query >= pool->slots.size() ||
        command_buffer->statistics_open)
        return PM_ERROR_INVALID_ARGUMENT;
    command_buffer->statistics_open = true;
    command_buffer->statistics_pool = pool;
    command_buffer->statistics_query = query;
    command_buffer->statistics_step_count = 0U;
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
            if (pool->type == PM_QUERY_REALTIME_STEP_STATISTICS)
                pool->realtime_statistics[query] = {};
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
        !statistics_query(pool->type) || query >= pool->slots.size() ||
        !command_buffer->statistics_open ||
        command_buffer->statistics_pool != pool ||
        command_buffer->statistics_query != query ||
        command_buffer->statistics_simulation == nullptr ||
        (pool->type == PM_QUERY_REALTIME_STEP_STATISTICS &&
         command_buffer->statistics_step_count != 1U))
        return PM_ERROR_INVALID_ARGUMENT;
    const PmSimulation simulation = command_buffer->statistics_simulation;
    const auto execution = command_buffer->statistics_execution;
    command_buffer->statistics_open = false;
    command_buffer->statistics_pool = nullptr;
    command_buffer->statistics_step_count = 0U;
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
            PmStepStatistics &statistics =
                pool->type == PM_QUERY_REALTIME_STEP_STATISTICS
                ? pool->realtime_statistics[query].step
                : pool->slots[query].statistics;
            // A realtime statistics query brackets exactly one step.  Its
            // public count is query-local, not the simulation's lifetime
            // submission counter (which legacy step statistics retain).
            statistics.stepCount =
                pool->type == PM_QUERY_REALTIME_STEP_STATISTICS
                ? 1U : simulation->submitted_steps;
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
            if (pool->type == PM_QUERY_REALTIME_STEP_STATISTICS) {
                parallel_mater::simulation::RealtimeTickTelemetry telemetry{};
                parallel_mater::simulation::RealtimeScheduleLedger ledger{};
                const PmResult realtime_result = parallel_mater::api::from_status(
                    simulation->runtime->collect_realtime_statistics(
                        telemetry, ledger, queue));
                if (realtime_result != PM_SUCCESS) return realtime_result;
                PmRealtimeStepStatistics &realtime =
                    pool->realtime_statistics[query];
                realtime.schemaVersion = telemetry.schema_version;
                realtime.profileVersion = telemetry.profile_version;
                realtime.budgetVersion = telemetry.budget_version;
                realtime.phaseCount = PM_REALTIME_WORK_PHASE_COUNT;
                realtime.tick = telemetry.tick;
                copy_realtime_work(telemetry.total, realtime.total);
                for (std::size_t index = 0U;
                     index < telemetry.phases.size(); ++index)
                    copy_realtime_work(
                        telemetry.phases[index].work,
                        realtime.phases[index]);
                realtime.timeAdvances = ledger.time_advances;
                for (std::size_t index = 0U;
                     index < ledger.domain_iterations.size(); ++index)
                    realtime.domainIterations[index] =
                        ledger.domain_iterations[index];
                for (std::size_t index = 0U;
                     index < ledger.coupling_sweeps.size(); ++index) {
                    const auto edges = ledger.coupling_edges[index];
                    realtime.couplingEdgeCount += edges;
                    // One algebraic sweep visits every registered edge in the
                    // family. Public work accounting is per edge, matching the
                    // frozen qualification contract rather than merely counting
                    // the batched dispatch loop.
                    realtime.couplingSweeps +=
                        ledger.coupling_sweeps[index] * edges;
                }
                // A rejected fixed tick has no published schedule work.  The
                // queue submission itself remains observable, but staged GPU
                // work, spatial updates, algebraic iterations, and the clock
                // advance are rolled back with the state.  Exposing the
                // aborted ledger here made a capacity failure look committed
                // even though readback correctly retained the prior state.
                if (statistics.executionResult != PM_SUCCESS) {
                    realtime.total = {};
                    realtime.total.queueSubmissions = 1U;
                    for (auto &phase : realtime.phases) phase = {};
                    realtime.timeAdvances = 0U;
                    for (auto &iterations : realtime.domainIterations)
                        iterations = 0U;
                    realtime.couplingEdgeCount = 0U;
                    realtime.couplingSweeps = 0U;
                }
            }
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
        ? sizeof(std::uint64_t)
        : pool->type == PM_QUERY_REALTIME_STEP_STATISTICS
            ? sizeof(PmRealtimeStepStatistics) : sizeof(PmStepStatistics);
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
            statistics_query(pool->type) && slot.available &&
            (pool->type == PM_QUERY_REALTIME_STEP_STATISTICS
                 ? pool->realtime_statistics[first + offset].step.executionResult
                 : slot.statistics.executionResult) ==
                PM_ERROR_CAPACITY_EXCEEDED;
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
            } else if (pool->type == PM_QUERY_REALTIME_STEP_STATISTICS) {
                std::memcpy(destination,
                            &pool->realtime_statistics[first + offset],
                            sizeof(PmRealtimeStepStatistics));
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
    parallel_mater::device::MemoryBudgetSnapshot snapshot{};
    const PmResult result = device->services->memory_budget(snapshot);
    if (result != PM_SUCCESS) return result;
    switch (snapshot.confidence) {
    case PM_MEMORY_BUDGET_CONFIDENCE_UNKNOWN:
        snapshot.available = 0U;
        snapshot.total = 0U;
        break;
    case PM_MEMORY_BUDGET_CONFIDENCE_IMPLEMENTATION_ESTIMATE:
    case PM_MEMORY_BUDGET_CONFIDENCE_DRIVER_REPORTED:
    case PM_MEMORY_BUDGET_CONFIDENCE_OS_REPORTED:
        if (snapshot.available > snapshot.total)
            return PM_ERROR_INVALID_STATE;
        break;
    default: return PM_ERROR_INVALID_STATE;
    }
    budget->heapCount = 1U;
    budget->budget[0] = snapshot.total;
    budget->usage[0] = snapshot.total - snapshot.available;
    budget->confidence[0] = snapshot.confidence;
    budget->applicationAllocationBytes =
        device->application_bytes.load(std::memory_order_relaxed);
    budget->libraryDeviceBytes =
        device->library_device_bytes.load(std::memory_order_relaxed);
    budget->libraryHostBytes =
        device->library_host_bytes.load(std::memory_order_relaxed);
    budget->borrowedBufferBytes =
        device->borrowed_buffer_bytes.load(std::memory_order_relaxed);
    return PM_SUCCESS;
}
