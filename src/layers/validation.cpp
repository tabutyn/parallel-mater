// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_layer.h>
#include <parallel_mater/pm_layer_args.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <vector>
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
struct Geometry {
    PmBuffer vertices{};
    PmBuffer indices{};
    PmDeviceMemory memory{};
    bool bound{};
    PmMemoryRequirements requirements{};
};
struct Simulation {
    PmSimulationMemoryRequirements requirements{};
    PmDeviceMemory persistent_memory{};
    PmDeviceMemory scratch_memory{};
    bool bound{};
    std::uint32_t rigid_body_capacity{};
    std::uint32_t constraint_capacity{};
    std::uint32_t fluid_count{};
    std::uint32_t cloth_count{};
    std::uint32_t rope_count{};
    std::uint32_t soft_body_count{};
    std::uint32_t smoke_count{};
    std::uint32_t emitter_capacity{};
    std::uint32_t destroyer_capacity{};
    std::uint32_t paint_field_count{};
    std::uint32_t paint_rule_capacity{};
    std::vector<PmGeometry> rigid_geometries;
    std::vector<PmGeometry> paint_geometries;
    std::unordered_set<PmGeometry> uncertain_geometries;
};
struct Pipeline {
    std::uint32_t fluid_material_count{};
    std::uint32_t cloth_material_count{};
    std::uint32_t rope_material_count{};
    std::uint32_t soft_body_material_count{};
    std::uint32_t smoke_material_count{};
};
enum class QueryState { initial, reset, written, consumed, pending };
struct QuerySlot {
    QueryState state{QueryState::initial};
    QueryState completion_state{QueryState::initial};
    std::uint64_t pending_submission{};
};
struct QueryPool {
    PmQueryType type{};
    std::vector<QuerySlot> slots;
};
enum class SimulationUpdateType { reset, rigid_geometry, paint_geometry };
struct SimulationUpdate {
    SimulationUpdateType type{};
    PmSimulation simulation{};
    std::uint32_t first{};
    std::vector<PmGeometry> geometries;
};
struct QueryOperation {
    PmQueryPool pool{};
    std::uint32_t first{};
    std::uint32_t count{};
    QueryState completion_state{QueryState::initial};
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
    PmPhysicsPipeline bound_pipeline{};
    bool statistics_open{};
    PmQueryPool statistics_pool{};
    std::uint32_t statistics_query{};
    PmSimulation statistics_simulation{};
    std::unordered_set<PmBuffer> buffers;
    std::unordered_set<PmGeometry> geometries;
    std::unordered_set<PmSimulation> simulations;
    std::unordered_set<PmPhysicsPipeline> pipelines;
    std::unordered_set<PmQueryPool> query_pools;
    std::vector<SimulationUpdate> simulation_updates;
    std::vector<QueryOperation> query_operations;
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
    std::vector<SimulationUpdate> simulation_updates;
    std::vector<QueryOperation> query_operations;
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
    std::unordered_map<PmGeometry, Geometry> geometries;
    std::unordered_map<PmSimulation, Simulation> simulations;
    std::unordered_map<PmPhysicsPipeline, Pipeline> pipelines;
    std::unordered_map<PmQueryPool, QueryPool> query_pools;
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
#if defined(PARALLEL_MATER_VALIDATION_TESTING)
std::atomic_bool fail_next_tracking_allocation{};
#endif

void emit_diagnostic(const PmLayerCallInfo &call, const char *rule,
                     PmValidationSeverity severity, PmResult result,
                     const char *message) noexcept {
    if (!call.pfnDiagnostic) return;
    const PmValidationDiagnostic diagnostic{
        rule, severity, result, call.pCommandName, call.primaryObjectType,
        call.primaryObject, call.argumentSize, call.pArguments, message};
    try {
        call.pfnDiagnostic(call.pDiagnosticUserData, &diagnostic);
    } catch (...) {
        // User callbacks must not unwind through the C ABI. Keep validation
        // and driver execution usable if a C++ callback violates that rule.
    }
}
const char *rule_for(PmResult result) noexcept {
    return result == PM_ERROR_INVALID_STATE
        ? PM_VALIDATION_RULE_INVALID_STATE
        : PM_VALIDATION_RULE_INVALID_ARGUMENT;
}
const char *message_for(PmResult result) noexcept {
    return result == PM_ERROR_INVALID_STATE
        ? "call is invalid for the tracked object state"
        : "call arguments or object ownership are invalid";
}

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
Geometry *geometry(PmDevice device, PmGeometry value) {
    const auto d = devices.find(device);
    if (d == devices.end()) return nullptr;
    const auto found = d->second.geometries.find(value);
    return found == d->second.geometries.end() ? nullptr : &found->second;
}
Simulation *simulation(PmDevice device, PmSimulation value) {
    const auto d = devices.find(device);
    if (d == devices.end()) return nullptr;
    const auto found = d->second.simulations.find(value);
    return found == d->second.simulations.end() ? nullptr : &found->second;
}
Pipeline *pipeline(PmDevice device, PmPhysicsPipeline value) {
    const auto d = devices.find(device);
    if (d == devices.end()) return nullptr;
    const auto found = d->second.pipelines.find(value);
    return found == d->second.pipelines.end() ? nullptr : &found->second;
}
QueryPool *query_pool(PmDevice device, PmQueryPool value) {
    const auto d = devices.find(device);
    if (d == devices.end()) return nullptr;
    const auto found = d->second.query_pools.find(value);
    return found == d->second.query_pools.end() ? nullptr : &found->second;
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
void clear_dependencies(CommandBuffer &buffer) {
    buffer.bound_pipeline = nullptr;
    buffer.statistics_open = false;
    buffer.statistics_pool = nullptr;
    buffer.statistics_query = 0U;
    buffer.statistics_simulation = nullptr;
    buffer.buffers.clear();
    buffer.geometries.clear();
    buffer.simulations.clear();
    buffer.pipelines.clear();
    buffer.query_pools.clear();
    buffer.simulation_updates.clear();
    buffer.query_operations.clear();
}
template<class Handle, class Member>
bool command_references(Handle handle, Member member) {
    for (const auto &[value, buffer] : command_buffers) {
        (void)value;
        if ((buffer.*member).contains(handle)) return true;
    }
    return false;
}
bool simulation_references(PmDevice device, PmGeometry handle) {
    const auto found = devices.find(device);
    if (found == devices.end()) return false;
    for (const auto &[value, tracked] : found->second.simulations) {
        (void)value;
        if (std::find(tracked.rigid_geometries.begin(),
                      tracked.rigid_geometries.end(), handle) !=
                tracked.rigid_geometries.end() ||
            std::find(tracked.paint_geometries.begin(),
                      tracked.paint_geometries.end(), handle) !=
                tracked.paint_geometries.end() ||
            tracked.uncertain_geometries.contains(handle))
            return true;
    }
    return false;
}
bool geometry_references(PmDevice device, PmBuffer handle) {
    const auto found = devices.find(device);
    if (found == devices.end()) return false;
    for (const auto &[value, tracked] : found->second.geometries) {
        (void)value;
        if (tracked.vertices == handle || tracked.indices == handle) return true;
    }
    return false;
}
PmResult validate_buffer_range(PmDevice device, const PmBufferRange *range,
                               PmDeviceSize minimum = 0U) {
    if (!range || !range->buffer || range->size < minimum)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *tracked = buffer(device, range->buffer);
    return tracked && range_fits(range->offset, range->size, tracked->size)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
PmResult validate_buffer_view(PmDevice device, const PmBufferView &view,
                              PmDeviceSize element_size) {
    if (view.count && (view.stride < element_size ||
        static_cast<PmDeviceSize>(view.count) >
            UINT64_MAX / static_cast<PmDeviceSize>(view.stride)))
        return PM_ERROR_INVALID_ARGUMENT;
    const auto required = static_cast<PmDeviceSize>(view.count) * view.stride;
    return validate_buffer_range(device, &view.range, required);
}
PmResult validate_simulation_command(PmCommandBuffer command,
                                     PmSimulation handle,
                                     Simulation **tracked = nullptr) {
    if (!handle) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *commands = command_buffer(command);
    if (!commands) return PM_ERROR_INVALID_ARGUMENT;
    auto *found = simulation(commands->device, handle);
    if (!found) return PM_ERROR_INVALID_ARGUMENT;
    if (tracked) *tracked = found;
    return PM_SUCCESS;
}
void record_simulation_dependency(PmCommandBuffer command,
                                  PmSimulation handle) {
    if (auto *tracked = command_buffer(command))
        tracked->simulations.insert(handle);
}
void record_buffer_dependency(PmCommandBuffer command, PmBuffer handle) {
    if (auto *tracked = command_buffer(command); tracked && handle)
        tracked->buffers.insert(handle);
}
QueryState effective_query_state(const CommandBuffer &commands,
                                 PmQueryPool handle, std::uint32_t query,
                                 const QueryPool &pool) {
    auto state = pool.slots[query].state;
    for (const auto &operation : commands.query_operations)
        if (operation.pool == handle && query >= operation.first &&
            query - operation.first < operation.count)
            state = operation.completion_state;
    return state;
}
QueryState submitted_query_state(const PmSubmitInfo &info,
                                 std::uint32_t current_buffer,
                                 std::size_t current_operation,
                                 PmQueryPool handle, std::uint32_t query,
                                 const QueryPool &pool) {
    auto state = pool.slots[query].state;
    for (std::uint32_t buffer_index = 0U;
         buffer_index <= current_buffer; ++buffer_index) {
        const auto *commands = command_buffer(
            info.pCommandBuffers[buffer_index]);
        if (!commands) continue;
        const auto limit = buffer_index == current_buffer
            ? current_operation : commands->query_operations.size();
        for (std::size_t operation_index = 0U;
             operation_index < limit; ++operation_index) {
            const auto &operation = commands->query_operations[operation_index];
            if (operation.pool == handle && query >= operation.first &&
                query - operation.first < operation.count)
                state = operation.completion_state;
        }
    }
    return state;
}
void apply_simulation_update(const SimulationUpdate &update,
                             bool successful) {
    PmDevice owner{};
    if (const auto commands = std::find_if(
            devices.begin(), devices.end(), [&](const auto &entry) {
                return entry.second.simulations.contains(update.simulation);
            }); commands != devices.end())
        owner = commands->first;
    if (!owner) return;
    auto *tracked = simulation(owner, update.simulation);
    if (!tracked) return;
    if (!successful) {
        for (const auto handle : update.geometries)
            if (handle) tracked->uncertain_geometries.insert(handle);
        return;
    }
    if (update.type == SimulationUpdateType::reset) {
        std::fill(tracked->rigid_geometries.begin(),
                  tracked->rigid_geometries.end(), nullptr);
        std::fill(tracked->paint_geometries.begin(),
                  tracked->paint_geometries.end(), nullptr);
        tracked->uncertain_geometries.clear();
        return;
    }
    auto &targets = update.type == SimulationUpdateType::rigid_geometry
        ? tracked->rigid_geometries : tracked->paint_geometries;
    for (std::size_t index = 0U; index < update.geometries.size(); ++index)
        if (update.first + index < targets.size())
            targets[update.first + index] = update.geometries[index];
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
    const auto submitted = submissions.find(value);
    if (submitted != submissions.end()) {
        for (const auto &update : submitted->second.simulation_updates)
            apply_simulation_update(update, result == PM_SUCCESS);
        if (const auto found = devices.find(submitted->second.device);
            found != devices.end())
            for (auto &[handle, pool] : found->second.query_pools) {
                (void)handle;
                for (auto &slot : pool.slots)
                    if (slot.state == QueryState::pending &&
                        slot.pending_submission == value) {
                        slot.state = result == PM_SUCCESS
                            ? slot.completion_state : QueryState::consumed;
                        slot.pending_submission = 0U;
                    }
            }
    }
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
                clear_dependencies(*buffer);
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
        clear_dependencies(*buffer);
        buffer->state = CommandState::recording;
        buffer->usage = a.pInfo->flags;
    }
}
PmResult validate(const PmLayerArgs_pmEndCommandBuffer &a) {
    if (!tracking_available) return PM_SUCCESS;
    const auto *buffer = command_buffer(a.commandBuffer);
    if (!buffer) return PM_ERROR_INVALID_ARGUMENT;
    if (buffer->statistics_open) return PM_ERROR_INVALID_STATE;
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
        clear_dependencies(*buffer);
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
    for (std::uint32_t buffer_index = 0U;
         buffer_index < a.pInfo->commandBufferCount; ++buffer_index) {
        const auto *commands = command_buffer(
            a.pInfo->pCommandBuffers[buffer_index]);
        for (std::size_t operation_index = 0U;
             operation_index < commands->query_operations.size();
             ++operation_index) {
            const auto &operation =
                commands->query_operations[operation_index];
            const auto *pool = query_pool(queue->second.device,
                                          operation.pool);
            if (!pool || operation.first > pool->slots.size() ||
                operation.count > pool->slots.size() - operation.first)
                return PM_ERROR_INVALID_ARGUMENT;
            for (std::uint32_t index = 0U; index < operation.count; ++index) {
                const auto state = submitted_query_state(
                    *a.pInfo, buffer_index, operation_index,
                    operation.pool, operation.first + index, *pool);
                if (operation.completion_state == QueryState::reset) {
                    if (state == QueryState::pending ||
                        state == QueryState::written)
                        return PM_ERROR_INVALID_STATE;
                } else if (state != QueryState::reset) {
                    return PM_ERROR_INVALID_STATE;
                }
            }
        }
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
    auto &submitted = submissions.at(submission);
    for (std::uint32_t index = 0U;
         index < a.pInfo->commandBufferCount; ++index)
        if (auto *buffer = command_buffer(a.pInfo->pCommandBuffers[index])) {
            submitted.simulation_updates.insert(
                submitted.simulation_updates.end(),
                buffer->simulation_updates.begin(),
                buffer->simulation_updates.end());
            submitted.query_operations.insert(
                submitted.query_operations.end(),
                buffer->query_operations.begin(),
                buffer->query_operations.end());
            buffer->state = CommandState::pending;
            buffer->pending_submission = submission;
        }
    for (const auto &operation : submitted.query_operations)
        if (auto *pool = query_pool(queue->second.device, operation.pool))
            for (std::uint32_t index = 0U; index < operation.count; ++index) {
                auto &slot = pool->slots[operation.first + index];
                slot.state = QueryState::pending;
                slot.completion_state = operation.completion_state;
                slot.pending_submission = submission;
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

PmResult validate(const PmLayerArgs_pmCreateGeometry &a) {
    if (!a.pInfo || !a.pGeometry ||
        a.pInfo->sType != PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO ||
        a.pInfo->pNext || a.pInfo->queueFamilyIndex != 0U ||
        a.pInfo->vertices.format != PM_FORMAT_R32G32B32_SFLOAT ||
        a.pInfo->vertices.count == 0U ||
        a.pInfo->vertices.stride != sizeof(float) * 3U ||
        a.pInfo->indices.format != PM_FORMAT_R32_UINT ||
        a.pInfo->indices.count == 0U ||
        a.pInfo->indices.count % 3U != 0U ||
        a.pInfo->indices.stride != sizeof(std::uint32_t))
        return PM_ERROR_INVALID_ARGUMENT;
    if (validate_buffer_view(a.device, a.pInfo->vertices,
                             sizeof(float) * 3U) != PM_SUCCESS ||
        validate_buffer_view(a.device, a.pInfo->indices,
                             sizeof(std::uint32_t)) != PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCreateGeometry &a) {
    if (a.pGeometry && *a.pGeometry)
        devices[a.device].geometries[*a.pGeometry] = {
            a.pInfo->vertices.range.buffer, a.pInfo->indices.range.buffer};
}
PmResult validate(const PmLayerArgs_pmDestroyGeometry &a) {
    if (!a.geometry || !tracking_available) return PM_SUCCESS;
    if (!geometry(a.device, a.geometry)) return PM_ERROR_INVALID_ARGUMENT;
    if (command_references(a.geometry, &CommandBuffer::geometries) ||
        simulation_references(a.device, a.geometry))
        return PM_ERROR_INVALID_STATE;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmDestroyGeometry &a) {
    if (const auto found = devices.find(a.device); found != devices.end())
        found->second.geometries.erase(a.geometry);
}
PmResult validate(const PmLayerArgs_pmGetGeometryMemoryRequirements &a) {
    return a.pRequirements &&
        (!tracking_available || geometry(a.device, a.geometry))
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmGetGeometryMemoryRequirements &a) {
    if (auto *tracked = geometry(a.device, a.geometry))
        tracked->requirements = *a.pRequirements;
}
PmResult validate(const PmLayerArgs_pmBindGeometryMemory &a) {
    if (!tracking_available) return PM_SUCCESS;
    const auto *tracked = geometry(a.device, a.geometry);
    const auto *allocation = memory(a.device, a.memory);
    if (!tracked || !allocation) return PM_ERROR_INVALID_ARGUMENT;
    if (tracked->bound) return PM_ERROR_INVALID_STATE;
    const auto &requirements = tracked->requirements;
    if ((requirements.size &&
         !fits(a.offset, requirements.size, allocation->size)) ||
        (requirements.alignment &&
         a.offset % requirements.alignment != 0U) ||
        (requirements.memoryTypeBits &&
         (allocation->type >= 32U ||
          !(requirements.memoryTypeBits & (1U << allocation->type)))))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmBindGeometryMemory &a) {
    if (auto *tracked = geometry(a.device, a.geometry)) {
        tracked->memory = a.memory;
        tracked->bound = true;
    }
}

PmResult validate(const PmLayerArgs_pmCreateSimulation &a) {
    if (!a.pInfo || !a.pSimulation ||
        a.pInfo->sType != PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO ||
        a.pInfo->pNext || a.pInfo->queueFamilyIndex != 0U ||
        (!a.pInfo->rigidBodyCapacity && !a.pInfo->fluidCount &&
         !a.pInfo->clothCount && !a.pInfo->ropeCount &&
         !a.pInfo->softBodyCount && !a.pInfo->smokeCount) ||
        (a.pInfo->fluidCount && !a.pInfo->pFluidCapacities) ||
        (a.pInfo->clothCount && !a.pInfo->pClothCapacities) ||
        (a.pInfo->ropeCount && !a.pInfo->pRopeCapacities) ||
        (a.pInfo->softBodyCount && !a.pInfo->pSoftBodyCapacities) ||
        (a.pInfo->smokeCount && !a.pInfo->pSmokeCapacities) ||
        (a.pInfo->paintFieldCount && !a.pInfo->pPaintFieldCapacities))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCreateSimulation &a) {
    if (!a.pSimulation || !*a.pSimulation) return;
    Simulation tracked{};
    tracked.rigid_body_capacity = a.pInfo->rigidBodyCapacity;
    tracked.constraint_capacity = a.pInfo->constraintCapacity;
    tracked.fluid_count = a.pInfo->fluidCount;
    tracked.cloth_count = a.pInfo->clothCount;
    tracked.rope_count = a.pInfo->ropeCount;
    tracked.soft_body_count = a.pInfo->softBodyCount;
    tracked.smoke_count = a.pInfo->smokeCount;
    tracked.emitter_capacity = a.pInfo->emitterCapacity;
    tracked.destroyer_capacity = a.pInfo->destroyerCapacity;
    tracked.paint_field_count = a.pInfo->paintFieldCount;
    tracked.paint_rule_capacity = a.pInfo->paintRuleCapacity;
    tracked.rigid_geometries.resize(a.pInfo->rigidBodyCapacity);
    tracked.paint_geometries.resize(a.pInfo->paintFieldCount);
    devices[a.device].simulations.emplace(*a.pSimulation,
                                          std::move(tracked));
}
PmResult validate(const PmLayerArgs_pmDestroySimulation &a) {
    if (!a.simulation || !tracking_available) return PM_SUCCESS;
    if (!simulation(a.device, a.simulation)) return PM_ERROR_INVALID_ARGUMENT;
    return command_references(a.simulation, &CommandBuffer::simulations)
        ? PM_ERROR_INVALID_STATE : PM_SUCCESS;
}
void record(const PmLayerArgs_pmDestroySimulation &a) {
    if (const auto found = devices.find(a.device); found != devices.end())
        found->second.simulations.erase(a.simulation);
}
PmResult validate(const PmLayerArgs_pmGetSimulationMemoryRequirements &a) {
    return a.pInfo &&
        (!tracking_available || simulation(a.device, a.simulation))
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmGetSimulationMemoryRequirements &a) {
    if (auto *tracked = simulation(a.device, a.simulation))
        tracked->requirements = *a.pInfo;
}
PmResult validate(const PmLayerArgs_pmBindSimulationMemory &a) {
    if (!a.pBinding) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *tracked = simulation(a.device, a.simulation);
    const auto *persistent = memory(a.device, a.pBinding->persistentMemory);
    const auto *scratch = memory(a.device, a.pBinding->scratchMemory);
    if (!tracked || !persistent || !scratch) return PM_ERROR_INVALID_ARGUMENT;
    if (tracked->bound) return PM_ERROR_INVALID_STATE;
    const auto valid_binding = [](const PmMemoryRequirements &requirements,
                                  const Memory &allocation,
                                  PmDeviceSize offset) {
        return (!requirements.size ||
                fits(offset, requirements.size, allocation.size)) &&
            (!requirements.alignment ||
             offset % requirements.alignment == 0U) &&
            (!requirements.memoryTypeBits ||
             (allocation.type < 32U &&
              (requirements.memoryTypeBits & (1U << allocation.type))));
    };
    if (!valid_binding(tracked->requirements.persistent, *persistent,
                       a.pBinding->persistentOffset) ||
        !valid_binding(tracked->requirements.scratch, *scratch,
                       a.pBinding->scratchOffset))
        return PM_ERROR_INVALID_ARGUMENT;
    if (a.pBinding->persistentMemory == a.pBinding->scratchMemory &&
        tracked->requirements.persistent.size &&
        tracked->requirements.scratch.size) {
        const auto persistent_end = a.pBinding->persistentOffset +
            tracked->requirements.persistent.size;
        const auto scratch_end = a.pBinding->scratchOffset +
            tracked->requirements.scratch.size;
        if (a.pBinding->persistentOffset < scratch_end &&
            a.pBinding->scratchOffset < persistent_end)
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmBindSimulationMemory &a) {
    if (auto *tracked = simulation(a.device, a.simulation)) {
        tracked->persistent_memory = a.pBinding->persistentMemory;
        tracked->scratch_memory = a.pBinding->scratchMemory;
        tracked->bound = true;
    }
}

PmResult validate(const PmLayerArgs_pmCreatePhysicsPipeline &a) {
    return a.pInfo && a.pPipeline &&
        a.pInfo->sType == PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO &&
        !a.pInfo->pNext &&
        (!a.pInfo->fluidMaterialCount || a.pInfo->pFluidMaterials) &&
        (!a.pInfo->couplingCount || a.pInfo->pCouplings) &&
        (!a.pInfo->clothMaterialCount || a.pInfo->pClothMaterials) &&
        (!a.pInfo->ropeMaterialCount || a.pInfo->pRopeMaterials) &&
        (!a.pInfo->softBodyMaterialCount || a.pInfo->pSoftBodyMaterials) &&
        (!a.pInfo->smokeMaterialCount || a.pInfo->pSmokeMaterials)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCreatePhysicsPipeline &a) {
    if (a.pPipeline && *a.pPipeline)
        devices[a.device].pipelines[*a.pPipeline] = {
            a.pInfo->fluidMaterialCount, a.pInfo->clothMaterialCount,
            a.pInfo->ropeMaterialCount, a.pInfo->softBodyMaterialCount,
            a.pInfo->smokeMaterialCount};
}
PmResult validate(const PmLayerArgs_pmDestroyPhysicsPipeline &a) {
    if (!a.pipeline || !tracking_available) return PM_SUCCESS;
    if (!pipeline(a.device, a.pipeline)) return PM_ERROR_INVALID_ARGUMENT;
    return command_references(a.pipeline, &CommandBuffer::pipelines)
        ? PM_ERROR_INVALID_STATE : PM_SUCCESS;
}
void record(const PmLayerArgs_pmDestroyPhysicsPipeline &a) {
    if (const auto found = devices.find(a.device); found != devices.end())
        found->second.pipelines.erase(a.pipeline);
}

Pipeline *bound_pipeline(PmCommandBuffer command) {
    const auto *commands = command_buffer(command);
    return commands && commands->bound_pipeline
        ? pipeline(commands->device, commands->bound_pipeline) : nullptr;
}
void record_material_dependency(PmCommandBuffer command) {
    if (auto *commands = command_buffer(command);
        commands && commands->bound_pipeline)
        commands->pipelines.insert(commands->bound_pipeline);
}
PmResult validate(const PmLayerArgs_pmCmdBuildGeometry &a) {
    if (!a.geometry) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *commands = command_buffer(a.commandBuffer);
    const auto *tracked = commands
        ? geometry(commands->device, a.geometry) : nullptr;
    return tracked && tracked->bound
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCmdBuildGeometry &a) {
    if (auto *commands = command_buffer(a.commandBuffer))
        commands->geometries.insert(a.geometry);
}
PmResult validate(const PmLayerArgs_pmCmdBindPhysicsPipeline &a) {
    if (!a.pipeline) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *commands = command_buffer(a.commandBuffer);
    return commands && pipeline(commands->device, a.pipeline)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCmdBindPhysicsPipeline &a) {
    if (auto *commands = command_buffer(a.commandBuffer)) {
        commands->bound_pipeline = a.pipeline;
        commands->pipelines.insert(a.pipeline);
    }
}
PmResult validate(const PmLayerArgs_pmCmdWriteRigidBodies &a) {
    Simulation *tracked{};
    if (!a.count || !a.pBodies ||
        validate_simulation_command(a.commandBuffer, a.simulation,
                                    &tracked) != PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    if (a.firstBody > tracked->rigid_body_capacity ||
        a.count > tracked->rigid_body_capacity - a.firstBody)
        return PM_ERROR_INVALID_ARGUMENT;
    const auto *commands = command_buffer(a.commandBuffer);
    for (std::uint32_t index = 0U; index < a.count; ++index) {
        const auto &body = a.pBodies[index];
        if (body.active != PM_FALSE && body.active != PM_TRUE)
            return PM_ERROR_INVALID_ARGUMENT;
        if (body.active == PM_TRUE &&
            !geometry(commands->device, body.geometry))
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdWriteRigidBodies &a) {
    auto *commands = command_buffer(a.commandBuffer);
    if (!commands) return;
    commands->simulations.insert(a.simulation);
    SimulationUpdate update{
        SimulationUpdateType::rigid_geometry, a.simulation, a.firstBody};
    update.geometries.reserve(a.count);
    for (std::uint32_t index = 0U; index < a.count; ++index) {
        const auto handle = a.pBodies[index].active == PM_TRUE
            ? a.pBodies[index].geometry : nullptr;
        update.geometries.push_back(handle);
        if (handle) commands->geometries.insert(handle);
    }
    commands->simulation_updates.push_back(std::move(update));
}
PmResult validate(const PmLayerArgs_pmCmdResetSimulation &a) {
    return validate_simulation_command(a.commandBuffer, a.simulation);
}
void record(const PmLayerArgs_pmCmdResetSimulation &a) {
    if (auto *commands = command_buffer(a.commandBuffer)) {
        commands->simulations.insert(a.simulation);
        commands->simulation_updates.push_back(
            {SimulationUpdateType::reset, a.simulation});
    }
}
PmResult validate(const PmLayerArgs_pmCmdSetFluidParticles &a) {
    Simulation *tracked{};
    if (!a.pParticles ||
        validate_simulation_command(a.commandBuffer, a.simulation,
                                    &tracked) != PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *materials = bound_pipeline(a.commandBuffer);
    if (!materials || a.fluidIndex >= tracked->fluid_count ||
        a.fluidIndex >= materials->fluid_material_count)
        return PM_ERROR_INVALID_ARGUMENT;
    if (a.pParticles->count &&
        validate_buffer_view(command_buffer(a.commandBuffer)->device,
                             *a.pParticles,
                             sizeof(PmParticleState)) != PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdSetFluidParticles &a) {
    record_simulation_dependency(a.commandBuffer, a.simulation);
    record_material_dependency(a.commandBuffer);
    if (a.pParticles->count)
        record_buffer_dependency(a.commandBuffer,
                                 a.pParticles->range.buffer);
}
PmResult validate(const PmLayerArgs_pmCmdStepSimulation &a) {
    if (!a.pStep || a.pStep->timestep <= 0.0F || !a.pStep->substeps ||
        (a.pStep->flags & ~PM_STEP_COLLECT_CONTACT_EVENTS_BIT) != 0U ||
        validate_simulation_command(a.commandBuffer, a.simulation) !=
            PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    return !tracking_available || bound_pipeline(a.commandBuffer)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCmdStepSimulation &a) {
    record_simulation_dependency(a.commandBuffer, a.simulation);
    record_material_dependency(a.commandBuffer);
    if (auto *commands = command_buffer(a.commandBuffer);
        commands && commands->statistics_open)
        commands->statistics_simulation = a.simulation;
}
PmResult validate(const PmLayerArgs_pmCmdWriteCloths &a) {
    Simulation *tracked{};
    if (!a.count || !a.pCloths ||
        validate_simulation_command(a.commandBuffer, a.simulation,
                                    &tracked) != PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *materials = bound_pipeline(a.commandBuffer);
    if (!materials || a.firstCloth > tracked->cloth_count ||
        a.count > tracked->cloth_count - a.firstCloth)
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < a.count; ++index)
        if (a.pCloths[index].active == PM_TRUE &&
            a.pCloths[index].materialIndex >=
                materials->cloth_material_count)
            return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdWriteCloths &a) {
    record_simulation_dependency(a.commandBuffer, a.simulation);
    record_material_dependency(a.commandBuffer);
}
PmResult validate(const PmLayerArgs_pmCmdWriteRopes &a) {
    Simulation *tracked{};
    if (!a.count || !a.pRopes ||
        validate_simulation_command(a.commandBuffer, a.simulation,
                                    &tracked) != PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *materials = bound_pipeline(a.commandBuffer);
    if (!materials || a.firstRope > tracked->rope_count ||
        a.count > tracked->rope_count - a.firstRope)
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < a.count; ++index)
        if (a.pRopes[index].active == PM_TRUE &&
            a.pRopes[index].materialIndex >= materials->rope_material_count)
            return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdWriteRopes &a) {
    record_simulation_dependency(a.commandBuffer, a.simulation);
    record_material_dependency(a.commandBuffer);
}
PmResult validate(const PmLayerArgs_pmCmdWriteSoftBodies &a) {
    Simulation *tracked{};
    if (!a.count || !a.pSoftBodies ||
        validate_simulation_command(a.commandBuffer, a.simulation,
                                    &tracked) != PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *materials = bound_pipeline(a.commandBuffer);
    if (!materials || a.firstSoftBody > tracked->soft_body_count ||
        a.count > tracked->soft_body_count - a.firstSoftBody)
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < a.count; ++index)
        if (a.pSoftBodies[index].active == PM_TRUE &&
            a.pSoftBodies[index].materialIndex >=
                materials->soft_body_material_count)
            return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdWriteSoftBodies &a) {
    record_simulation_dependency(a.commandBuffer, a.simulation);
    record_material_dependency(a.commandBuffer);
}
PmResult validate(const PmLayerArgs_pmCmdWriteSmoke &a) {
    Simulation *tracked{};
    if (!a.count || !a.pSmoke ||
        validate_simulation_command(a.commandBuffer, a.simulation,
                                    &tracked) != PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *materials = bound_pipeline(a.commandBuffer);
    if (!materials || a.firstSmoke > tracked->smoke_count ||
        a.count > tracked->smoke_count - a.firstSmoke)
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < a.count; ++index)
        if (a.pSmoke[index].active == PM_TRUE &&
            a.pSmoke[index].materialIndex >= materials->smoke_material_count)
            return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdWriteSmoke &a) {
    record_simulation_dependency(a.commandBuffer, a.simulation);
    record_material_dependency(a.commandBuffer);
}
PmResult validate(const PmLayerArgs_pmCmdWritePaintFields &a) {
    Simulation *tracked{};
    if (!a.count || !a.pFields ||
        validate_simulation_command(a.commandBuffer, a.simulation,
                                    &tracked) != PM_SUCCESS)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    if (a.firstField > tracked->paint_field_count ||
        a.count > tracked->paint_field_count - a.firstField)
        return PM_ERROR_INVALID_ARGUMENT;
    const auto *commands = command_buffer(a.commandBuffer);
    for (std::uint32_t index = 0U; index < a.count; ++index) {
        const auto &field = a.pFields[index];
        if (field.active != PM_FALSE && field.active != PM_TRUE)
            return PM_ERROR_INVALID_ARGUMENT;
        if (field.active == PM_TRUE &&
            field.targetType == PM_PAINT_TARGET_RIGID &&
            !geometry(commands->device, field.geometry))
            return PM_ERROR_INVALID_ARGUMENT;
    }
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdWritePaintFields &a) {
    auto *commands = command_buffer(a.commandBuffer);
    if (!commands) return;
    commands->simulations.insert(a.simulation);
    SimulationUpdate update{
        SimulationUpdateType::paint_geometry, a.simulation, a.firstField};
    update.geometries.reserve(a.count);
    for (std::uint32_t index = 0U; index < a.count; ++index) {
        const auto &field = a.pFields[index];
        const auto handle = field.active == PM_TRUE &&
                field.targetType == PM_PAINT_TARGET_RIGID
            ? field.geometry : nullptr;
        update.geometries.push_back(handle);
        if (handle) commands->geometries.insert(handle);
    }
    commands->simulation_updates.push_back(std::move(update));
}
PmResult validate(const PmLayerArgs_pmCmdCopyBuffer &a) {
    if (!a.src || !a.dst || !a.count || !a.pRegions)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *commands = command_buffer(a.commandBuffer);
    const auto *source = commands ? buffer(commands->device, a.src) : nullptr;
    const auto *destination = commands ? buffer(commands->device, a.dst) : nullptr;
    if (!source || !destination) return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < a.count; ++index)
        if (!fits(a.pRegions[index].srcOffset, a.pRegions[index].size,
                  source->size) ||
            !fits(a.pRegions[index].dstOffset, a.pRegions[index].size,
                  destination->size))
            return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdCopyBuffer &a) {
    record_buffer_dependency(a.commandBuffer, a.src);
    record_buffer_dependency(a.commandBuffer, a.dst);
}

#define PM_SIMULATION_DEPENDENCY(command_name) \
    PmResult validate(const PmLayerArgs_##command_name &a) { \
        return validate_simulation_command(a.commandBuffer, a.simulation); \
    } \
    void record(const PmLayerArgs_##command_name &a) { \
        record_simulation_dependency(a.commandBuffer, a.simulation); \
    }
PM_SIMULATION_DEPENDENCY(pmCmdApplyRigidInputs)
PM_SIMULATION_DEPENDENCY(pmCmdSetRigidStates)
PM_SIMULATION_DEPENDENCY(pmCmdSetKinematicTargets)
PM_SIMULATION_DEPENDENCY(pmCmdWriteRigidConstraints)
PM_SIMULATION_DEPENDENCY(pmCmdSetCouplings)
PM_SIMULATION_DEPENDENCY(pmCmdWriteParticleEmitters)
PM_SIMULATION_DEPENDENCY(pmCmdWriteParticleDestroyers)
PM_SIMULATION_DEPENDENCY(pmCmdWritePaintRules)
PM_SIMULATION_DEPENDENCY(pmCmdClearPaintFields)
#undef PM_SIMULATION_DEPENDENCY

#define PM_SIMULATION_DESTINATION(command_name) \
    PmResult validate(const PmLayerArgs_##command_name &a) { \
        if (validate_simulation_command(a.commandBuffer, a.simulation) != \
            PM_SUCCESS) return PM_ERROR_INVALID_ARGUMENT; \
        if (!tracking_available) \
            return a.pDestination ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT; \
        const auto *commands = command_buffer(a.commandBuffer); \
        return validate_buffer_range(commands->device, a.pDestination); \
    } \
    void record(const PmLayerArgs_##command_name &a) { \
        record_simulation_dependency(a.commandBuffer, a.simulation); \
        record_buffer_dependency(a.commandBuffer, a.pDestination->buffer); \
    }
PM_SIMULATION_DESTINATION(pmCmdQueryHitBox)
PM_SIMULATION_DESTINATION(pmCmdCopyClothState)
PM_SIMULATION_DESTINATION(pmCmdCopyClothSurface)
PM_SIMULATION_DESTINATION(pmCmdCopyRopeState)
PM_SIMULATION_DESTINATION(pmCmdCopySoftBodyState)
PM_SIMULATION_DESTINATION(pmCmdCopySoftBodySurface)
PM_SIMULATION_DESTINATION(pmCmdCopySmokeState)
PM_SIMULATION_DESTINATION(pmCmdCopyPaintField)
PM_SIMULATION_DESTINATION(pmCmdCopyConstraintStates)
PM_SIMULATION_DESTINATION(pmCmdCopyContactEvents)
PM_SIMULATION_DESTINATION(pmCmdCopyDebugData)
PM_SIMULATION_DESTINATION(pmCmdCopyRigidStates)
PM_SIMULATION_DESTINATION(pmCmdCopyFluidParticles)
#undef PM_SIMULATION_DESTINATION

PmResult validate(const PmLayerArgs_pmCmdPipelineBarrier &a) {
    if ((a.memoryBarrierCount && !a.pMemoryBarriers) ||
        (a.bufferBarrierCount && !a.pBufferBarriers))
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *commands = command_buffer(a.commandBuffer);
    for (std::uint32_t index = 0U; index < a.bufferBarrierCount; ++index)
        if (validate_buffer_range(commands->device,
                                  &a.pBufferBarriers[index].range) !=
            PM_SUCCESS)
            return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdPipelineBarrier &a) {
    for (std::uint32_t index = 0U; index < a.bufferBarrierCount; ++index)
        record_buffer_dependency(a.commandBuffer,
                                 a.pBufferBarriers[index].range.buffer);
}

PmResult validate(const PmLayerArgs_pmCreateQueryPool &a) {
    return a.pInfo && a.pPool &&
        a.pInfo->sType == PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO &&
        !a.pInfo->pNext && a.pInfo->queryCount &&
        a.pInfo->queryCount <= 65536U &&
        (a.pInfo->queryType == PM_QUERY_TIMESTAMP ||
         a.pInfo->queryType == PM_QUERY_STEP_STATISTICS)
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
void record(const PmLayerArgs_pmCreateQueryPool &a) {
    if (!a.pPool || !*a.pPool) return;
    QueryPool tracked{};
    tracked.type = a.pInfo->queryType;
    tracked.slots.resize(a.pInfo->queryCount);
    devices[a.device].query_pools.emplace(*a.pPool, std::move(tracked));
}
PmResult validate(const PmLayerArgs_pmDestroyQueryPool &a) {
    if (!a.pool || !tracking_available) return PM_SUCCESS;
    const auto *tracked = query_pool(a.device, a.pool);
    if (!tracked) return PM_ERROR_INVALID_ARGUMENT;
    if (command_references(a.pool, &CommandBuffer::query_pools))
        return PM_ERROR_INVALID_STATE;
    for (const auto &slot : tracked->slots)
        if (slot.state == QueryState::pending)
            return PM_ERROR_INVALID_STATE;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmDestroyQueryPool &a) {
    if (const auto found = devices.find(a.device); found != devices.end())
        found->second.query_pools.erase(a.pool);
}
PmResult validate(const PmLayerArgs_pmCmdResetQueryPool &a) {
    if (!a.pool || !a.count) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *commands = command_buffer(a.commandBuffer);
    const auto *pool = commands ? query_pool(commands->device, a.pool) : nullptr;
    if (!pool || a.first > pool->slots.size() ||
        a.count > pool->slots.size() - a.first)
        return PM_ERROR_INVALID_ARGUMENT;
    for (std::uint32_t index = 0U; index < a.count; ++index)
        if (const auto state = effective_query_state(
                *commands, a.pool, a.first + index, *pool);
            state == QueryState::pending || state == QueryState::written)
            return PM_ERROR_INVALID_STATE;
    if (commands->statistics_open && commands->statistics_pool == a.pool &&
        commands->statistics_query >= a.first &&
        commands->statistics_query - a.first < a.count)
        return PM_ERROR_INVALID_STATE;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdResetQueryPool &a) {
    if (auto *commands = command_buffer(a.commandBuffer)) {
        commands->query_pools.insert(a.pool);
        commands->query_operations.push_back(
            {a.pool, a.first, a.count, QueryState::reset});
    }
}
PmResult validate(const PmLayerArgs_pmCmdWriteTimestamp &a) {
    if (!a.pool || !a.stage ||
        (a.stage & ~PM_STAGE_ALL_COMMANDS_BIT) != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *commands = command_buffer(a.commandBuffer);
    const auto *pool = commands ? query_pool(commands->device, a.pool) : nullptr;
    if (!pool || pool->type != PM_QUERY_TIMESTAMP ||
        a.query >= pool->slots.size())
        return PM_ERROR_INVALID_ARGUMENT;
    return effective_query_state(*commands, a.pool, a.query, *pool) ==
            QueryState::reset
        ? PM_SUCCESS : PM_ERROR_INVALID_STATE;
}
void record(const PmLayerArgs_pmCmdWriteTimestamp &a) {
    if (auto *commands = command_buffer(a.commandBuffer)) {
        commands->query_pools.insert(a.pool);
        commands->query_operations.push_back(
            {a.pool, a.query, 1U, QueryState::written});
    }
}
PmResult validate(const PmLayerArgs_pmCmdBeginStatisticsQuery &a) {
    if (!a.pool) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *commands = command_buffer(a.commandBuffer);
    const auto *pool = commands ? query_pool(commands->device, a.pool) : nullptr;
    if (!pool || pool->type != PM_QUERY_STEP_STATISTICS ||
        a.query >= pool->slots.size())
        return PM_ERROR_INVALID_ARGUMENT;
    if (commands->statistics_open ||
        effective_query_state(*commands, a.pool, a.query, *pool) !=
            QueryState::reset)
        return PM_ERROR_INVALID_STATE;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmCmdBeginStatisticsQuery &a) {
    if (auto *commands = command_buffer(a.commandBuffer)) {
        commands->statistics_open = true;
        commands->statistics_pool = a.pool;
        commands->statistics_query = a.query;
        commands->statistics_simulation = nullptr;
        commands->query_pools.insert(a.pool);
    }
}
PmResult validate(const PmLayerArgs_pmCmdEndStatisticsQuery &a) {
    if (!a.pool) return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *commands = command_buffer(a.commandBuffer);
    const auto *pool = commands ? query_pool(commands->device, a.pool) : nullptr;
    if (!pool || pool->type != PM_QUERY_STEP_STATISTICS ||
        a.query >= pool->slots.size())
        return PM_ERROR_INVALID_ARGUMENT;
    return commands->statistics_open &&
            commands->statistics_pool == a.pool &&
            commands->statistics_query == a.query &&
            commands->statistics_simulation
        ? PM_SUCCESS : PM_ERROR_INVALID_STATE;
}
void record(const PmLayerArgs_pmCmdEndStatisticsQuery &a) {
    if (auto *commands = command_buffer(a.commandBuffer)) {
        commands->statistics_open = false;
        commands->statistics_pool = nullptr;
        commands->statistics_simulation = nullptr;
        commands->query_operations.push_back(
            {a.pool, a.query, 1U, QueryState::written});
    }
}
PmResult validate(const PmLayerArgs_pmGetSimulationStatus &a) {
    return a.pExecutionResult &&
        (!tracking_available || simulation(a.device, a.simulation))
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
PmResult validate(const PmLayerArgs_pmGetQueryPoolResults &a) {
    if (!a.pData || !a.count ||
        (a.flags & ~(PM_QUERY_RESULT_WAIT_BIT |
                     PM_QUERY_RESULT_WITH_AVAILABILITY_BIT)) != 0U ||
        a.stride % sizeof(std::uint64_t) != 0U ||
        reinterpret_cast<std::uintptr_t>(a.pData) %
                alignof(std::uint64_t) != 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    if (!tracking_available) return PM_SUCCESS;
    const auto *pool = query_pool(a.device, a.pool);
    if (!pool || a.first > pool->slots.size() ||
        a.count > pool->slots.size() - a.first)
        return PM_ERROR_INVALID_ARGUMENT;
    const std::size_t value_size = pool->type == PM_QUERY_TIMESTAMP
        ? sizeof(std::uint64_t) : sizeof(PmStepStatistics);
    const std::size_t record_size = value_size +
        ((a.flags & PM_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0U
             ? sizeof(std::uint64_t) : 0U);
    if (a.stride < record_size ||
        (a.count > 1U &&
         a.stride > (SIZE_MAX - record_size) / (a.count - 1U)) ||
        a.dataSize < record_size + (a.count - 1U) * a.stride)
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
void record(const PmLayerArgs_pmGetQueryPoolResults &a) {
    if (auto *pool = query_pool(a.device, a.pool))
        for (std::uint32_t index = 0U; index < a.count; ++index) {
            auto &slot = pool->slots[a.first + index];
            if (slot.state == QueryState::written)
                slot.state = QueryState::consumed;
        }
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
    if (!a.buffer || !tracking_available) return PM_SUCCESS;
    if (!buffer(a.device, a.buffer)) return PM_ERROR_INVALID_ARGUMENT;
    if (geometry_references(a.device, a.buffer) ||
        command_references(a.buffer, &CommandBuffer::buffers))
        return PM_ERROR_INVALID_STATE;
    return PM_SUCCESS;
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
    for (const auto &[handle, tracked] : devices.at(a.device).geometries) {
        (void)handle;
        if (tracked.memory == a.memory) return PM_ERROR_INVALID_STATE;
    }
    for (const auto &[handle, tracked] : devices.at(a.device).simulations) {
        (void)handle;
        if (tracked.persistent_memory == a.memory ||
            tracked.scratch_memory == a.memory)
            return PM_ERROR_INVALID_STATE;
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
    // A v1/v2 loader cannot supply diagnostic and application-object context.
    if (*version < 3U) return PM_ERROR_INCOMPATIBLE_VERSION;
    *version = PM_LAYER_INTERFACE_VERSION;
    return PM_SUCCESS;
}

#if defined(PARALLEL_MATER_VALIDATION_TESTING)
PM_LAYER_EXPORT void PMAPI_CALL pm_layerTestFailNextTrackingAllocation() {
    fail_next_tracking_allocation.store(true);
}
#endif

PM_LAYER_EXPORT PmResult PMAPI_CALL pm_layerPreCall(const PmLayerCallInfo *call) {
    if (!call) return PM_ERROR_INVALID_ARGUMENT;
    if (!call->pCommandName || !call->pCommandName[0] ||
        !call->primaryHandle || !call->pArguments) {
        emit_diagnostic(*call, PM_VALIDATION_RULE_INVALID_LAYER_CALL,
                        PM_VALIDATION_SEVERITY_ERROR,
                        PM_ERROR_INVALID_ARGUMENT,
                        "loader supplied an invalid layer call payload");
        return PM_ERROR_INVALID_ARGUMENT;
    }
    PmResult result = PM_SUCCESS;
    {
        std::lock_guard lock(mutex);
        if (tracking_available &&
            std::strncmp(call->pCommandName, "pmCmd", 5U) == 0) {
            if (call->argumentSize < sizeof(PmCommandBuffer)) {
                result = PM_ERROR_INVALID_ARGUMENT;
            } else {
                PmCommandBuffer handle{};
                // Every typed pmCmd* payload begins with its
                // application-visible command buffer.
                std::memcpy(&handle, call->pArguments, sizeof handle);
                const auto *buffer = command_buffer(handle);
                if (!buffer) result = PM_ERROR_INVALID_ARGUMENT;
                else if (buffer->state != CommandState::recording)
                    result = PM_ERROR_INVALID_STATE;
            }
        }
        if (result == PM_SUCCESS)
            result = dispatch(*call,
                [](const auto &a) { return validate(a); });
    }
    if (result != PM_SUCCESS)
        emit_diagnostic(*call, rule_for(result),
                        PM_VALIDATION_SEVERITY_ERROR, result,
                        message_for(result));
    return result;
}

PM_LAYER_EXPORT void PMAPI_CALL pm_layerPostCall(
    const PmLayerCallInfo *call, PmResult result) {
    if (!call) return;
    bool exhausted = false;
    {
        std::lock_guard lock(mutex);
        if (!tracking_available) return;
        try {
#if defined(PARALLEL_MATER_VALIDATION_TESTING)
            if (fail_next_tracking_allocation.exchange(false))
                throw std::bad_alloc{};
#endif
            (void)dispatch(*call, [result](const auto &a) {
                record_result(a, result);
                return PM_SUCCESS;
            });
        } catch (const std::bad_alloc &) {
            // Do not turn a successful driver call into a failure. Drop state
            // and retain stateless validation.
            devices.clear();
            command_pools.clear();
            command_buffers.clear();
            queues.clear();
            fences.clear();
            submissions.clear();
            semaphores.clear();
            tracking_available = false;
            exhausted = true;
        }
    }
    if (exhausted)
        emit_diagnostic(*call, PM_VALIDATION_RULE_TRACKING_UNAVAILABLE,
                        PM_VALIDATION_SEVERITY_WARNING,
                        PM_ERROR_OUT_OF_HOST_MEMORY,
                        "host bookkeeping exhausted; stateful validation disabled");
}
