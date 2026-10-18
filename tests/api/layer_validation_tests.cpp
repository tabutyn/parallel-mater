// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>
#include <parallel_mater/pm_layer.h>
#include <parallel_mater/pm_layer_args.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

extern "C" PmResult PMAPI_CALL pm_layerNegotiateLoaderInterfaceVersion(uint32_t *);
extern "C" PmResult PMAPI_CALL pm_layerPreCall(const PmLayerCallInfo *);
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%d: %s\n", __LINE__, #x); return 1; } } while(false)

namespace {
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
    uint32_t version = 1;
    CHECK(pm_layerNegotiateLoaderInterfaceVersion(&version) == PM_ERROR_INCOMPATIBLE_VERSION);
    version = 999;
    CHECK(pm_layerNegotiateLoaderInterfaceVersion(&version) == PM_SUCCESS);
    CHECK(version == 2);
    PmLayerArgs_pmMapMemory bad_arguments{};
    PmLayerCallInfo bad_call{"pmMapMemory", &bad_arguments, 0, &bad_arguments};
    CHECK(pm_layerPreCall(&bad_call) == PM_ERROR_INVALID_ARGUMENT);

    PmInstanceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = PM_API_VERSION_0_2;
    info.enableValidation = validation ? PM_TRUE : PM_FALSE;
    PmInstance instance{};
    CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_SUCCESS);
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
    CHECK(pmEndCommandBuffer(commands) == invalid_state);
    CHECK(pmCmdCopyBuffer(commands, nullptr, nullptr, 0U, nullptr) == invalid_state);
    CHECK(pmResetCommandBuffer(fixed_commands) == invalid_state);
    CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    CHECK(pmCmdCopyBuffer(commands, nullptr, nullptr, 0U, nullptr) == PM_SUCCESS);
    CHECK(pmBeginCommandBuffer(commands, &begin) == invalid_state);
    CHECK(pmResetCommandPool(device, pool) == invalid_state);
    CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    CHECK(pmEndCommandBuffer(commands) == invalid_state);

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
    pmDestroyFence(device, fence);
    pmFreeCommandBuffers(device, fixed_pool, 1U, &fixed_commands);
    pmFreeCommandBuffers(device, pool, 1U, &commands);
    pmDestroyCommandPool(device, fixed_pool);
    pmDestroyCommandPool(device, pool);
    pmDestroyDevice(other);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts(validation
        ? "PASS: layer rejects invalid resources and command transitions before permissive vendor driver"
        : "PASS: permissive driver accepts resource and command negative controls without layer");
}
