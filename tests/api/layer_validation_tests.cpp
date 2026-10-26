// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>
#include <parallel_mater/pm_layer.h>
#include <parallel_mater/pm_layer_args.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

extern "C" PmResult PMAPI_CALL pm_layerNegotiateLoaderInterfaceVersion(uint32_t *);
extern "C" PmResult PMAPI_CALL pm_layerPreCall(const PmLayerCallInfo *);
extern "C" void PMAPI_CALL pm_layerTestFailNextTrackingAllocation();
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%d: %s\n", __LINE__, #x); return 1; } } while(false)

namespace {
struct DiagnosticState {
    std::mutex mutex;
    std::uint32_t count{};
    const char *rule{};
    const char *command{};
    const char *message{};
    PmValidationSeverity severity{};
    PmResult result{PM_SUCCESS};
    PmValidationObjectType object_type{};
    const void *object{};
    std::uint32_t argument_size{};
    PmDevice argument_device{};
    PmSemaphore argument_semaphore{};
    bool reenter{};
    bool reentered{};
    PmDevice reentry_device{};
    PmSemaphore reentry_semaphore{};
    PmResult reentry_result{PM_NOT_READY};
};

void PMAPI_CALL diagnostic_callback(
    void *user, const PmValidationDiagnostic *diagnostic) {
    auto &state = *static_cast<DiagnosticState *>(user);
    bool reenter = false;
    PmDevice device{};
    PmSemaphore semaphore{};
    {
        std::lock_guard lock(state.mutex);
        ++state.count;
        state.rule = diagnostic->pRuleId;
        state.command = diagnostic->pCommandName;
        state.message = diagnostic->pMessage;
        state.severity = diagnostic->severity;
        state.result = diagnostic->result;
        state.object_type = diagnostic->objectType;
        state.object = diagnostic->object;
        state.argument_size = diagnostic->argumentSize;
        if (diagnostic->pCommandName &&
            std::strcmp(diagnostic->pCommandName,
                        "pmGetSemaphoreCounterValue") == 0 &&
            diagnostic->argumentSize ==
                sizeof(PmLayerArgs_pmGetSemaphoreCounterValue)) {
            const auto &arguments = *static_cast<
                const PmLayerArgs_pmGetSemaphoreCounterValue *>(
                    diagnostic->pArguments);
            state.argument_device = arguments.device;
            state.argument_semaphore = arguments.semaphore;
        }
        if (state.reenter && !state.reentered) {
            state.reentered = true;
            reenter = true;
            device = state.reentry_device;
            semaphore = state.reentry_semaphore;
        }
    }
    if (reenter) {
        std::uint64_t value{};
        const auto result = pmGetSemaphoreCounterValue(
            device, semaphore, &value);
        std::lock_guard lock(state.mutex);
        state.reentry_result = result;
    }
}

void set_mock_failure(const char *name, bool enabled) {
#if defined(_WIN32)
    _putenv_s(name, enabled ? "1" : "");
#else
    if (enabled) setenv(name, "1", 1);
    else unsetenv(name);
#endif
}
} // namespace

int main(int argc, char **argv) {
    const bool validation = argc < 2 || std::strcmp(argv[1], "plain") != 0;
    DiagnosticState diagnostics{};
    uint32_t version = 1;
    CHECK(pm_layerNegotiateLoaderInterfaceVersion(&version) == PM_ERROR_INCOMPATIBLE_VERSION);
    version = 2;
    CHECK(pm_layerNegotiateLoaderInterfaceVersion(&version) == PM_ERROR_INCOMPATIBLE_VERSION);
    version = 999;
    CHECK(pm_layerNegotiateLoaderInterfaceVersion(&version) == PM_SUCCESS);
    CHECK(version == 3);
    PmLayerArgs_pmMapMemory bad_arguments{};
    PmLayerCallInfo bad_call{
        "pmMapMemory", &bad_arguments, 0U, nullptr,
        PM_VALIDATION_OBJECT_UNKNOWN, nullptr,
        diagnostic_callback, &diagnostics};
    CHECK(pm_layerPreCall(&bad_call) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(diagnostics.count == 1U);
    CHECK(std::strcmp(diagnostics.rule,
                      PM_VALIDATION_RULE_INVALID_LAYER_CALL) == 0);
    CHECK(diagnostics.severity == PM_VALIDATION_SEVERITY_ERROR);
    CHECK(diagnostics.result == PM_ERROR_INVALID_ARGUMENT);
    diagnostics.count = 0U;

    PmInstanceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = PM_API_VERSION_0_3;
    PmValidationDiagnosticsCreateInfo diagnostic_info{};
    diagnostic_info.sType =
        PM_STRUCTURE_TYPE_VALIDATION_DIAGNOSTICS_CREATE_INFO;
    diagnostic_info.pfnDiagnostic = diagnostic_callback;
    diagnostic_info.pUserData = &diagnostics;
    info.pNext = validation ? &diagnostic_info : nullptr;
    PmInstance instance{};
    if (validation) {
        diagnostic_info.pfnDiagnostic = nullptr;
        CHECK(pmCreateInstance(&info, nullptr, &instance) ==
              PM_ERROR_INVALID_ARGUMENT);
        CHECK(instance == nullptr);
        diagnostic_info.pfnDiagnostic = diagnostic_callback;
    }
    CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_SUCCESS);
    // The loader owns a copy; the create-info storage is no longer observed.
    diagnostic_info.pfnDiagnostic = nullptr;
    diagnostic_info.pUserData = nullptr;
    uint32_t count = 1;
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS);
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    PmDevice device{}, other{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    CHECK(pmCreateDevice(physical, &device_info, &other) == PM_SUCCESS);
    const auto invalid = validation ? PM_ERROR_INVALID_ARGUMENT : PM_SUCCESS;
    const auto invalid_state = validation ? PM_ERROR_INVALID_STATE : PM_SUCCESS;

    PmQueue queue{}, other_queue{};
    pmGetDeviceQueue(device, 0U, 0U, &queue);
    pmGetDeviceQueue(device, 1U, 0U, &other_queue);
    CHECK(queue);
    CHECK(other_queue);
    PmSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = PM_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    PmSemaphore timeline{}, foreign_timeline{};
    CHECK(pmCreateSemaphore(device, &semaphore_info, &timeline) == PM_SUCCESS);
    CHECK(pmCreateSemaphore(other, &semaphore_info, &foreign_timeline) == PM_SUCCESS);
    std::uint64_t counter{};
    diagnostics.reenter = validation;
    diagnostics.reentry_device = device;
    diagnostics.reentry_semaphore = timeline;
    CHECK(pmGetSemaphoreCounterValue(other, timeline, &counter) == invalid);
    if (validation) {
        CHECK(diagnostics.count == 1U);
        CHECK(std::strcmp(diagnostics.rule,
                          PM_VALIDATION_RULE_INVALID_ARGUMENT) == 0);
        CHECK(std::strcmp(diagnostics.command,
                          "pmGetSemaphoreCounterValue") == 0);
        CHECK(diagnostics.severity == PM_VALIDATION_SEVERITY_ERROR);
        CHECK(diagnostics.result == PM_ERROR_INVALID_ARGUMENT);
        CHECK(diagnostics.object_type == PM_VALIDATION_OBJECT_DEVICE);
        CHECK(diagnostics.object == other);
        CHECK(diagnostics.argument_size ==
              sizeof(PmLayerArgs_pmGetSemaphoreCounterValue));
        CHECK(diagnostics.argument_device == other);
        CHECK(diagnostics.argument_semaphore == timeline);
        CHECK(diagnostics.message && diagnostics.message[0]);
        CHECK(diagnostics.reentered);
        CHECK(diagnostics.reentry_result == PM_SUCCESS);
    }
    diagnostics.reenter = false;
    CHECK(pmWaitSemaphore(other, timeline, 0U, 0U) == invalid);

    // Delivery can occur concurrently; callback state remains application
    // synchronized, and no global callback registration is involved.
    const auto before_parallel = diagnostics.count;
    std::atomic_uint32_t thread_failures{};
    std::vector<std::thread> threads;
    for (std::uint32_t index = 0U; index < 8U; ++index)
        threads.emplace_back([&] {
            std::uint64_t value{};
            if (pmGetSemaphoreCounterValue(other, timeline, &value) != invalid)
                thread_failures.fetch_add(1U);
        });
    for (auto &thread : threads) thread.join();
    CHECK(thread_failures.load() == 0U);
    CHECK(diagnostics.count == before_parallel + (validation ? 8U : 0U));
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PmCommandPool pool{}, fixed_pool{};
    CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    pool_info.flags = 0U;
    CHECK(pmCreateCommandPool(device, &pool_info, &fixed_pool) == PM_SUCCESS);
    PmCommandBuffer commands{}, fixed_commands{};
    CHECK(pmAllocateCommandBuffers(device, pool, 1U, &commands) == PM_SUCCESS);
    CHECK(pmAllocateCommandBuffers(device, fixed_pool, 1U, &fixed_commands) == PM_SUCCESS);
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    // The mock driver accepts every transition. Differences in the validation
    // run therefore prove rejection happened before vendor dispatch.
    const auto before_state = diagnostics.count;
    CHECK(pmEndCommandBuffer(commands) == invalid_state);
    if (validation) {
        CHECK(diagnostics.count == before_state + 1U);
        CHECK(std::strcmp(diagnostics.rule,
                          PM_VALIDATION_RULE_INVALID_STATE) == 0);
        CHECK(std::strcmp(diagnostics.command, "pmEndCommandBuffer") == 0);
        CHECK(diagnostics.result == PM_ERROR_INVALID_STATE);
        CHECK(diagnostics.object_type ==
              PM_VALIDATION_OBJECT_COMMAND_BUFFER);
        CHECK(diagnostics.object == commands);
    }
    CHECK(pmCmdCopyBuffer(commands, nullptr, nullptr, 0U, nullptr) == invalid_state);
    CHECK(pmResetCommandBuffer(fixed_commands) == invalid_state);
    const auto before_valid = diagnostics.count;
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(diagnostics.count == before_valid);
    CHECK(pmCmdCopyBuffer(commands, nullptr, nullptr, 0U, nullptr) == invalid);
    CHECK(pmBeginCommandBuffer(commands, &begin) == invalid_state);
    CHECK(pmResetCommandPool(device, pool) == invalid_state);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == invalid_state);

    // Synchronization validation is deliberately tested against a driver
    // that accepts cross-device handles, invalid scopes and timeline cycles.
    PmSemaphoreSubmitInfo wait_scope{timeline, 1U, 0U};
    PmSubmitInfo sync_submit{};
    sync_submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    sync_submit.waitCount = 1U;
    sync_submit.pWaits = &wait_scope;
    CHECK(pmQueueSubmit(queue, &sync_submit, nullptr) == invalid);
    wait_scope.stages = PM_STAGE_HOST_BIT;
    CHECK(pmQueueSubmit(queue, &sync_submit, nullptr) == invalid);

    PmSemaphoreSubmitInfo foreign_signal{
        foreign_timeline, 1U, PM_STAGE_ALL_COMMANDS_BIT};
    sync_submit.waitCount = 0U;
    sync_submit.pWaits = nullptr;
    sync_submit.signalCount = 1U;
    sync_submit.pSignals = &foreign_signal;
    CHECK(pmQueueSubmit(queue, &sync_submit, nullptr) == invalid);

    PmSemaphoreSubmitInfo bad_signal{timeline, 1U, PM_STAGE_PHYSICS_BIT};
    sync_submit.pSignals = &bad_signal;
    CHECK(pmQueueSubmit(queue, &sync_submit, nullptr) == invalid);

    PmSemaphoreSubmitInfo future_wait{
        timeline, 2U, PM_STAGE_ALL_COMMANDS_BIT};
    PmSemaphoreSubmitInfo lower_signal{
        timeline, 1U, PM_STAGE_ALL_COMMANDS_BIT};
    sync_submit.waitCount = 1U;
    sync_submit.pWaits = &future_wait;
    sync_submit.pSignals = &lower_signal;
    CHECK(pmQueueSubmit(queue, &sync_submit, nullptr) == invalid);

    // A failed pre-acceptance submit must not reserve its timeline value.
    sync_submit.waitCount = 0U;
    sync_submit.pWaits = nullptr;
    set_mock_failure("PM_TEST_MOCK_SUBMIT_FAILURE", true);
    CHECK(pmQueueSubmit(queue, &sync_submit, nullptr) ==
          PM_ERROR_OUT_OF_HOST_MEMORY);
    set_mock_failure("PM_TEST_MOCK_SUBMIT_FAILURE", false);
    CHECK(pmQueueSubmit(queue, &sync_submit, nullptr) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &sync_submit, nullptr) == invalid);
    CHECK(pmWaitSemaphore(device, timeline, 1U,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);

    // Waiting work may be accepted before its producer. Timeline completion
    // releases only the producer; the consumer remains pending until its own
    // fence reports completion.
    PmCommandBuffer async_commands[2]{};
    CHECK(pmAllocateCommandBuffers(device, pool, 2U, async_commands) == PM_SUCCESS);
    for (auto command : async_commands) {
        CHECK(pmBeginCommandBuffer(command, &begin) == PM_SUCCESS);
        CHECK(pmEndCommandBuffer(command) == PM_SUCCESS);
    }
    PmFence async_fence{};
    CHECK(pmCreateFence(device, 0U, &async_fence) == PM_SUCCESS);
    PmSemaphoreSubmitInfo async_wait{
        timeline, 2U, PM_STAGE_PHYSICS_BIT};
    PmSubmitInfo consumer{};
    consumer.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    consumer.waitCount = 1U;
    consumer.pWaits = &async_wait;
    consumer.commandBufferCount = 1U;
    consumer.pCommandBuffers = &async_commands[0];
    CHECK(pmQueueSubmit(queue, &consumer, async_fence) == PM_SUCCESS);
    PmSemaphoreSubmitInfo async_signal{
        timeline, 2U, PM_STAGE_ALL_COMMANDS_BIT};
    PmSubmitInfo producer{};
    producer.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    producer.commandBufferCount = 1U;
    producer.pCommandBuffers = &async_commands[1];
    producer.signalCount = 1U;
    producer.pSignals = &async_signal;
    CHECK(pmQueueSubmit(queue, &producer, nullptr) == PM_SUCCESS);
    CHECK(pmWaitSemaphore(device, timeline, 2U,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(async_commands[1]) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(async_commands[0]) == invalid_state);
    CHECK(pmWaitForFences(device, 1U, &async_fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(async_commands[0]) == PM_SUCCESS);

    // Deferred timeline failure still drains and releases its submission.
    CHECK(pmBeginCommandBuffer(async_commands[1], &begin) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(async_commands[1]) == PM_SUCCESS);
    async_signal.value = 3U;
    CHECK(pmQueueSubmit(queue, &producer, nullptr) == PM_SUCCESS);
    set_mock_failure("PM_TEST_MOCK_DEFERRED_FAILURE", true);
    CHECK(pmWaitSemaphore(device, timeline, 3U,
                          PM_TIMEOUT_INFINITE) == PM_ERROR_DEVICE_LOST);
    set_mock_failure("PM_TEST_MOCK_DEFERRED_FAILURE", false);
    CHECK(pmResetCommandBuffer(async_commands[1]) == PM_SUCCESS);

    // Counter observation is also a precise submission completion boundary.
    CHECK(pmBeginCommandBuffer(async_commands[1], &begin) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(async_commands[1]) == PM_SUCCESS);
    async_signal.value = 4U;
    CHECK(pmQueueSubmit(queue, &producer, nullptr) == PM_SUCCESS);
    CHECK(pmGetSemaphoreCounterValue(device, timeline, &counter) == PM_SUCCESS);
    CHECK(counter == 4U);
    CHECK(pmResetCommandBuffer(async_commands[1]) == PM_SUCCESS);

    PmFence fence{};
    CHECK(pmCreateFence(device, 0U, &fence) == PM_SUCCESS);
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &commands;
    CHECK(pmQueueSubmit(other_queue, &submit, nullptr) == invalid);
    set_mock_failure("PM_TEST_MOCK_SUBMIT_FAILURE", true);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_ERROR_OUT_OF_HOST_MEMORY);
    set_mock_failure("PM_TEST_MOCK_SUBMIT_FAILURE", false);
    // A pre-acceptance failure leaves the command buffer and fence reusable.
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == invalid_state);
    CHECK(pmBeginCommandBuffer(commands, &begin) == invalid_state);
    CHECK(pmResetCommandBuffer(commands) == invalid_state);
    CHECK(pmResetCommandPool(device, pool) == invalid_state);

    // Completion errors are deferred but still drain native work. The layer
    // must release Pending state even though the wait itself reports failure.
    set_mock_failure("PM_TEST_MOCK_DEFERRED_FAILURE", true);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_ERROR_DEVICE_LOST);
    set_mock_failure("PM_TEST_MOCK_DEFERRED_FAILURE", false);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);

    // One-time completion becomes Invalid: it cannot be submitted again, but
    // Begin legally discards it and starts a new recording.
    begin.flags = PM_COMMAND_BUFFER_ONE_TIME_BIT;
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, nullptr) == invalid_state);
    begin.flags = 0U;
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmResetCommandPool(device, pool) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == invalid_state);

    // Queue/device idle are also completion boundaries, including errors.
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, nullptr) == PM_SUCCESS);
    CHECK(pmQueueWaitIdle(queue) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, nullptr) == PM_SUCCESS);
    set_mock_failure("PM_TEST_MOCK_DEFERRED_FAILURE", true);
    CHECK(pmDeviceWaitIdle(device) == PM_ERROR_DEVICE_LOST);
    set_mock_failure("PM_TEST_MOCK_DEFERRED_FAILURE", false);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);

    PmMemoryAllocateInfo allocation{PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, 512, 1};
    PmDeviceMemory memory{};
    CHECK(pmAllocateMemory(device, &allocation, &memory) == PM_SUCCESS);
    PmBufferCreateInfo buffer_info{PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 256,
                                  PM_BUFFER_TRANSFER_SRC_BIT, 0};
    PmBuffer buffer{};
    CHECK(pmCreateBuffer(device, &buffer_info, &buffer) == PM_SUCCESS);
    PmMemoryRequirements requirements{};
    pmGetBufferMemoryRequirements(device, buffer, &requirements);
    CHECK(requirements.size == 256);
    CHECK(pmBindBufferMemory(other, buffer, memory, 0) == invalid);
    CHECK(pmBindBufferMemory(device, buffer, memory, 1) == invalid);
    CHECK(pmBindBufferMemory(device, buffer, memory, 512) == invalid);
    CHECK(pmBindBufferMemory(device, buffer, memory, UINT64_MAX - 1) == invalid);
    CHECK(pmBindBufferMemory(device, buffer, memory, 256) == PM_SUCCESS);
    CHECK(pmBindBufferMemory(device, buffer, memory, 0) == invalid_state);

    void *mapped{};
    CHECK(pmMapMemory(other, memory, 0, 1, &mapped) == invalid);
    CHECK(pmMapMemory(device, memory, UINT64_MAX - 2, 8, &mapped) == invalid);
    CHECK(pmMapMemory(device, memory, 0, PM_WHOLE_SIZE, &mapped) == PM_SUCCESS);
    CHECK(pmMapMemory(device, memory, 0, 1, &mapped) == invalid_state);
    PmMappedMemoryRange range{memory, 513, 1};
    CHECK(pmFlushMappedMemoryRanges(device, 1, &range) == invalid);
    CHECK(pmInvalidateMappedMemoryRanges(device, 1, &range) == invalid);
    range = {memory, 0, PM_WHOLE_SIZE};
    CHECK(pmFlushMappedMemoryRanges(device, 1, &range) == PM_SUCCESS);
    CHECK(pmInvalidateMappedMemoryRanges(device, 1, &range) == PM_SUCCESS);
    pmUnmapMemory(device, memory);
    CHECK(pmInvalidateMappedMemoryRanges(device, 1, &range) == invalid);

    allocation.allocationSize = 13; // driver failure must not mutate layer state
    PmDeviceMemory failed{};
    CHECK(pmAllocateMemory(device, &allocation, &failed) == PM_ERROR_OUT_OF_DEVICE_MEMORY);
    CHECK(!failed);
    if (validation) {
        PmBuffer failed_buffer{};
        CHECK(pmCreateBuffer(device, nullptr, &failed_buffer) == PM_ERROR_INVALID_ARGUMENT);
        pmFreeMemory(device, memory); // rejected: live buffer still bound
        CHECK(pmMapMemory(device, memory, 0, 1, &mapped) == PM_SUCCESS);
        pmUnmapMemory(device, memory);
    }
    pmDestroyBuffer(device, buffer);
    CHECK(pmBindBufferMemory(device, buffer, memory, 0) == invalid);
    pmFreeMemory(device, memory);
    CHECK(pmMapMemory(device, memory, 0, 1, &mapped) == invalid);

    // Geometry, simulation, material, buffer and query dependencies are
    // validated against the same deliberately permissive mock driver.
    allocation.allocationSize = 2048U;
    PmDeviceMemory resource_memory{};
    CHECK(pmAllocateMemory(device, &allocation, &resource_memory) == PM_SUCCESS);
    buffer_info.size = 256U;
    buffer_info.usage = PM_BUFFER_GEOMETRY_INPUT_BIT |
                        PM_BUFFER_TRANSFER_SRC_BIT |
                        PM_BUFFER_TRANSFER_DST_BIT;
    PmBuffer geometry_input{};
    CHECK(pmCreateBuffer(device, &buffer_info, &geometry_input) == PM_SUCCESS);
    pmGetBufferMemoryRequirements(device, geometry_input, &requirements);
    CHECK(pmBindBufferMemory(device, geometry_input, resource_memory, 0U) ==
          PM_SUCCESS);

    PmGeometryCreateInfo geometry_info{};
    geometry_info.sType = PM_STRUCTURE_TYPE_GEOMETRY_CREATE_INFO;
    geometry_info.vertices = {{geometry_input, 0U, 36U},
                              PM_FORMAT_R32G32B32_SFLOAT, 3U, 12U};
    geometry_info.indices = {{geometry_input, 64U, 12U},
                             PM_FORMAT_R32_UINT, 3U, 4U};
    PmGeometry geometry{};
    CHECK(pmCreateGeometry(device, &geometry_info, &geometry) == PM_SUCCESS);
    PmMemoryRequirements geometry_requirements{};
    pmGetGeometryMemoryRequirements(other, geometry, &geometry_requirements);
    if (validation) CHECK(geometry_requirements.size == 0U);
    pmGetGeometryMemoryRequirements(device, geometry, &geometry_requirements);
    CHECK(geometry_requirements.size == 256U);
    CHECK(pmBindGeometryMemory(other, geometry, resource_memory, 256U) ==
          invalid);
    CHECK(pmBindGeometryMemory(device, geometry, resource_memory, 256U) ==
          PM_SUCCESS);
    CHECK(pmBindGeometryMemory(device, geometry, resource_memory, 512U) ==
          invalid_state);

    PmClothCapacity cloth_capacity{3U, 3U, 3U};
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    simulation_info.clothCount = 1U;
    simulation_info.pClothCapacities = &cloth_capacity;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_SUCCESS);
    PmSimulationMemoryRequirements simulation_requirements{};
    pmGetSimulationMemoryRequirements(device, simulation,
                                      &simulation_requirements);
    const PmSimulationMemoryBinding simulation_binding{
        resource_memory, 512U, resource_memory, 768U};
    CHECK(pmBindSimulationMemory(other, simulation, &simulation_binding) ==
          invalid);
    CHECK(pmBindSimulationMemory(device, simulation, &simulation_binding) ==
          PM_SUCCESS);
    CHECK(pmBindSimulationMemory(device, simulation, &simulation_binding) ==
          invalid_state);

    PmClothMaterialState cloth_material{};
    PmPhysicsPipelineCreateInfo pipeline_create{};
    pipeline_create.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_create.clothMaterialCount = 1U;
    pipeline_create.pClothMaterials = &cloth_material;
    PmPhysicsPipeline pipeline{}, replacement_pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_create, &pipeline) ==
          PM_SUCCESS);
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_create,
                                  &replacement_pipeline) == PM_SUCCESS);

    PmQueryPoolCreateInfo query_info{};
    query_info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = PM_QUERY_TIMESTAMP;
    query_info.queryCount = 13U;
    PmQueryPool failed_query{};
    CHECK(pmCreateQueryPool(device, &query_info, &failed_query) ==
          PM_ERROR_OUT_OF_HOST_MEMORY);
    CHECK(failed_query == nullptr);
    query_info.queryCount = 2U;
    PmQueryPool timestamps{};
    CHECK(pmCreateQueryPool(device, &query_info, &timestamps) == PM_SUCCESS);
    query_info.queryType = PM_QUERY_STEP_STATISTICS;
    query_info.queryCount = 1U;
    PmQueryPool statistics{};
    CHECK(pmCreateQueryPool(device, &query_info, &statistics) == PM_SUCCESS);

    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdBuildGeometry(commands, geometry) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, pipeline) == PM_SUCCESS);
    PmClothInfo invalid_cloth{};
    invalid_cloth.active = PM_TRUE;
    invalid_cloth.materialIndex = 1U;
    CHECK(pmCmdWriteCloths(commands, simulation, 0U, 1U,
                           &invalid_cloth) == invalid);
    invalid_cloth.materialIndex = 0U;
    CHECK(pmCmdWriteCloths(commands, simulation, 0U, 1U,
                           &invalid_cloth) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, replacement_pipeline) ==
          PM_SUCCESS);
    PmRigidBodyInfo body{};
    body.active = PM_TRUE;
    body.geometry = geometry;
    CHECK(pmCmdWriteRigidBodies(commands, simulation, 0U, 1U, &body) ==
          PM_SUCCESS);

    CHECK(pmCmdWriteTimestamp(commands, PM_STAGE_ALL_COMMANDS_BIT,
                              timestamps, 0U) == invalid_state);
    CHECK(pmCmdResetQueryPool(commands, timestamps, 1U, 2U) == invalid);
    CHECK(pmCmdResetQueryPool(commands, timestamps, 0U, 2U) == PM_SUCCESS);
    CHECK(pmCmdWriteTimestamp(commands, PM_STAGE_ALL_COMMANDS_BIT,
                              timestamps, 0U) == PM_SUCCESS);
    CHECK(pmCmdResetQueryPool(commands, statistics, 0U, 1U) == PM_SUCCESS);
    CHECK(pmCmdBeginStatisticsQuery(commands, statistics, 0U) == PM_SUCCESS);
    CHECK(pmCmdResetQueryPool(commands, statistics, 0U, 1U) ==
          invalid_state);
    CHECK(pmCmdBeginStatisticsQuery(commands, statistics, 0U) ==
          invalid_state);
    CHECK(pmCmdEndStatisticsQuery(commands, statistics, 0U) == invalid_state);
    PmStepInfo step{};
    step.timestep = 1.0F / 60.0F;
    step.substeps = 1U;
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdEndStatisticsQuery(commands, statistics, 0U) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);

    if (validation) {
        // Recorded references survive completion until command-buffer reset.
        pmDestroyGeometry(device, geometry);
        pmDestroyPhysicsPipeline(device, pipeline);
        pmDestroyPhysicsPipeline(device, replacement_pipeline);
        pmDestroyQueryPool(device, timestamps);
        pmDestroyBuffer(device, geometry_input);
        pmFreeMemory(device, resource_memory);
        PmResult execution{};
        CHECK(pmGetSimulationStatus(device, simulation, &execution) ==
              PM_SUCCESS);
    }
    submit.pCommandBuffers = &commands;
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    alignas(8) PmStepStatistics statistics_results{};
    CHECK(pmGetQueryPoolResults(device, statistics, 0U, 1U,
                                sizeof(statistics_results),
                                &statistics_results,
                                sizeof(statistics_results),
                                PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdResetQueryPool(commands, timestamps, 0U, 1U) ==
          invalid_state);
    CHECK(pmCmdResetQueryPool(commands, statistics, 0U, 1U) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    alignas(8) std::uint64_t timestamp_results[2]{};
    CHECK(pmGetQueryPoolResults(other, timestamps, 0U, 1U,
                                sizeof(timestamp_results), timestamp_results,
                                sizeof(std::uint64_t), 0U) == invalid);
    CHECK(pmGetQueryPoolResults(device, timestamps, 1U, 2U,
                                sizeof(timestamp_results), timestamp_results,
                                sizeof(std::uint64_t), 0U) == invalid);
    CHECK(pmGetQueryPoolResults(device, timestamps, 0U, 1U,
                                sizeof(std::uint64_t), timestamp_results,
                                4U, 0U) == invalid);
    CHECK(pmGetQueryPoolResults(device, timestamps, 0U, 2U,
                                sizeof(timestamp_results), timestamp_results,
                                sizeof(std::uint64_t), 0U) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdResetQueryPool(commands, timestamps, 0U, 2U) == PM_SUCCESS);
    CHECK(pmCmdResetQueryPool(commands, statistics, 0U, 1U) == PM_SUCCESS);
    CHECK(pmCmdBeginStatisticsQuery(commands, statistics, 0U) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(commands, replacement_pipeline) == PM_SUCCESS);
    CHECK(pmCmdStepSimulation(commands, simulation, &step) == PM_SUCCESS);
    CHECK(pmCmdEndStatisticsQuery(commands, statistics, 0U) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                          PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    CHECK(pmResetFences(device, 1U, &fence) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdResetQueryPool(commands, statistics, 0U, 1U) ==
          invalid_state);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmGetQueryPoolResults(device, statistics, 0U, 1U,
                                sizeof(statistics_results),
                                &statistics_results,
                                sizeof(statistics_results),
                                PM_QUERY_RESULT_WAIT_BIT) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdResetQueryPool(commands, statistics, 0U, 1U) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);

    if (validation) {
        // The submitted body keeps its geometry live after recorded
        // references are released. A submitted simulation reset releases it.
        pmDestroyGeometry(device, geometry);
        pmGetGeometryMemoryRequirements(device, geometry,
                                        &geometry_requirements);
        CHECK(geometry_requirements.size == 256U);
    }
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdResetSimulation(commands, simulation) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmQueueSubmit(queue, &submit, nullptr) == PM_SUCCESS);
    CHECK(pmQueueWaitIdle(queue) == PM_SUCCESS);
    CHECK(pmResetCommandBuffer(commands) == PM_SUCCESS);

    pmDestroyQueryPool(device, statistics);
    pmDestroyQueryPool(device, timestamps);
    pmDestroyPhysicsPipeline(device, replacement_pipeline);
    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroyGeometry(device, geometry);
    pmDestroyBuffer(device, geometry_input);
    pmDestroySimulation(device, simulation);
    pmFreeMemory(device, resource_memory);
    pmDestroyFence(device, fence);
    pmDestroyFence(device, async_fence);
    pmFreeCommandBuffers(device, pool, 2U, async_commands);
    pmDestroySemaphore(device, timeline);
    pmDestroySemaphore(other, foreign_timeline);
    pmFreeCommandBuffers(device, fixed_pool, 1U, &fixed_commands);
    pmFreeCommandBuffers(device, pool, 1U, &commands);
    pmDestroyCommandPool(device, fixed_pool);
    pmDestroyCommandPool(device, pool);

    // Device teardown drains accepted work and must remove every ownership
    // record before a driver can recycle the device handle.
    PmDevice teardown{};
    CHECK(pmCreateDevice(physical, &device_info, &teardown) == PM_SUCCESS);
    PmQueue teardown_queue{};
    pmGetDeviceQueue(teardown, 0U, 0U, &teardown_queue);
    PmSemaphore teardown_timeline{};
    CHECK(pmCreateSemaphore(teardown, &semaphore_info,
                            &teardown_timeline) == PM_SUCCESS);
    PmCommandPool teardown_pool{};
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    CHECK(pmCreateCommandPool(teardown, &pool_info, &teardown_pool) == PM_SUCCESS);
    PmCommandBuffer teardown_command{};
    CHECK(pmAllocateCommandBuffers(teardown, teardown_pool, 1U,
                                   &teardown_command) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(teardown_command, &begin) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(teardown_command) == PM_SUCCESS);
    PmSemaphoreSubmitInfo teardown_signal{
        teardown_timeline, 1U, PM_STAGE_ALL_COMMANDS_BIT};
    PmSubmitInfo teardown_submit{};
    teardown_submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    teardown_submit.commandBufferCount = 1U;
    teardown_submit.pCommandBuffers = &teardown_command;
    teardown_submit.signalCount = 1U;
    teardown_submit.pSignals = &teardown_signal;
    CHECK(pmQueueSubmit(teardown_queue, &teardown_submit, nullptr) == PM_SUCCESS);
    pmDestroyDevice(teardown);
    PmDevice replacement{};
    CHECK(pmCreateDevice(physical, &device_info, &replacement) == PM_SUCCESS);
    PmSemaphore replacement_timeline{};
    CHECK(pmCreateSemaphore(replacement, &semaphore_info,
                            &replacement_timeline) == PM_SUCCESS);
    CHECK(pmGetSemaphoreCounterValue(replacement, replacement_timeline,
                                     &counter) == PM_SUCCESS);
    CHECK(counter == 0U);
    pmDestroySemaphore(replacement, replacement_timeline);
    pmDestroyDevice(replacement);

    if (validation) {
        const auto before_oom = diagnostics.count;
        pm_layerTestFailNextTrackingAllocation();
        PmPhysicalDeviceProperties properties{};
        pmGetPhysicalDeviceProperties(physical, &properties);
        CHECK(diagnostics.count == before_oom + 1U);
        CHECK(std::strcmp(diagnostics.rule,
                          PM_VALIDATION_RULE_TRACKING_UNAVAILABLE) == 0);
        CHECK(std::strcmp(diagnostics.command,
                          "pmGetPhysicalDeviceProperties") == 0);
        CHECK(diagnostics.severity == PM_VALIDATION_SEVERITY_WARNING);
        CHECK(diagnostics.result == PM_ERROR_OUT_OF_HOST_MEMORY);
        CHECK(diagnostics.object_type ==
              PM_VALIDATION_OBJECT_PHYSICAL_DEVICE);
        CHECK(diagnostics.object == physical);
        pmGetPhysicalDeviceProperties(physical, &properties);
        CHECK(diagnostics.count == before_oom + 1U);
    }

    pmDestroyDevice(other);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts(validation
        ? "PASS: layer validates dependencies, queries, state and diagnostics"
        : "PASS: permissive driver accepts validation negative controls without diagnostics");
}
