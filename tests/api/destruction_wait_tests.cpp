// SPDX-License-Identifier: MIT
// Public SDK only. The implicit profile intentionally exercises the in-tree
// reference-driver teardown extension; it is not portable lifetime guidance.
#include <parallel_mater/parallel_mater.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define PM_CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #value); \
    std::abort(); \
} } while (false)

namespace {

constexpr PmDeviceSize buffer_size = 1U << 20U;

void set_delay(bool enabled) {
#if defined(_WIN32)
    PM_CHECK(_putenv_s("PM_TEST_DELAY_SUBMISSION", enabled ? "1" : "") == 0);
#else
    PM_CHECK(enabled ? setenv("PM_TEST_DELAY_SUBMISSION", "1", 1) == 0
                     : unsetenv("PM_TEST_DELAY_SUBMISSION") == 0);
#endif
}

struct Work {
    PmDevice device{};
    PmQueue queue{};
    std::array<PmDeviceMemory, 2> memory{};
    std::array<PmBuffer, 2> buffers{};
    std::array<void *, 2> mapped{};
    PmCommandPool pool{};
    PmCommandBuffer commands{};
    PmFence fence{};
    PmSemaphore timeline{};
};

PmDevice create_device(PmPhysicalDevice physical,
                       const PmPhysicalDeviceFeatures &supported) {
    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1U;
    info.pQueueInfos = &queue_info;
    info.enabledFeatures.timelineSemaphores = supported.timelineSemaphores;
    PmDevice device{};
    PM_CHECK(pmCreateDevice(physical, &info, &device) == PM_SUCCESS);
    return device;
}

std::uint32_t host_visible_type(PmPhysicalDevice physical,
                                std::uint32_t type_bits) {
    PmMemoryProperties properties{};
    pmGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (std::uint32_t index = 0U; index < properties.typeCount; ++index)
        if ((type_bits & (1U << index)) != 0U &&
            (properties.types[index].propertyFlags &
             PM_MEMORY_HOST_VISIBLE_BIT) != 0U)
            return index;
    return PM_INVALID_INDEX;
}

Work submit_copy(PmPhysicalDevice physical,
                 const PmPhysicalDeviceFeatures &supported,
                 bool delay_producer) {
    Work work;
    work.device = create_device(physical, supported);
    pmGetDeviceQueue(work.device, 0U, 0U, &work.queue);
    PM_CHECK(work.queue != nullptr);

    for (std::size_t index = 0U; index < work.memory.size(); ++index) {
        PmBufferCreateInfo buffer{};
        buffer.sType = PM_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer.size = buffer_size;
        buffer.usage = PM_BUFFER_TRANSFER_SRC_BIT | PM_BUFFER_TRANSFER_DST_BIT;
        PM_CHECK(pmCreateBuffer(work.device, &buffer,
                                &work.buffers[index]) == PM_SUCCESS);
        PmMemoryRequirements requirements{};
        pmGetBufferMemoryRequirements(work.device, work.buffers[index],
                                      &requirements);
        const std::uint32_t memory_type =
            host_visible_type(physical, requirements.memoryTypeBits);
        PM_CHECK(memory_type != PM_INVALID_INDEX);
        PmMemoryAllocateInfo allocation{};
        allocation.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type;
        PM_CHECK(pmAllocateMemory(work.device, &allocation,
                                  &work.memory[index]) == PM_SUCCESS);
        PM_CHECK(pmBindBufferMemory(work.device, work.buffers[index],
                                    work.memory[index], 0U) == PM_SUCCESS);
        PM_CHECK(pmMapMemory(work.device, work.memory[index], 0U, buffer_size,
                             &work.mapped[index]) == PM_SUCCESS);
    }
    auto *source = static_cast<std::uint8_t *>(work.mapped[0]);
    auto *destination = static_cast<std::uint8_t *>(work.mapped[1]);
    for (std::size_t index = 0U; index < buffer_size; ++index)
        source[index] = static_cast<std::uint8_t>((index * 131U + 17U) & 0xffU);
    std::memset(destination, 0, static_cast<std::size_t>(buffer_size));
    const std::array<PmMappedMemoryRange, 2> mapped_ranges{{
        {work.memory[0], 0U, buffer_size},
        {work.memory[1], 0U, buffer_size}}};
    PM_CHECK(pmFlushMappedMemoryRanges(
                 work.device, static_cast<std::uint32_t>(mapped_ranges.size()),
                 mapped_ranges.data()) == PM_SUCCESS);

    PmCommandPoolCreateInfo pool{};
    pool.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PM_CHECK(pmCreateCommandPool(work.device, &pool, &work.pool) == PM_SUCCESS);
    PM_CHECK(pmAllocateCommandBuffers(work.device, work.pool, 1U,
                                      &work.commands) == PM_SUCCESS);
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    PM_CHECK(pmBeginCommandBuffer(work.commands, &begin) == PM_SUCCESS);
    const PmBufferCopy region{0U, 0U, buffer_size};
    PM_CHECK(pmCmdCopyBuffer(work.commands, work.buffers[0], work.buffers[1],
                             1U, &region) == PM_SUCCESS);
    PM_CHECK(pmEndCommandBuffer(work.commands) == PM_SUCCESS);
    PM_CHECK(pmCreateFence(work.device, 0U, &work.fence) == PM_SUCCESS);
    PmSemaphoreCreateInfo semaphore{};
    semaphore.sType = PM_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    PM_CHECK(pmCreateSemaphore(work.device, &semaphore, &work.timeline) ==
             PM_SUCCESS);

    // The consumer is accepted but cannot execute until its later producer.
    // This proves pending state without racing queue access against teardown.
    const PmSemaphoreSubmitInfo wait{
        work.timeline, 1U, PM_STAGE_ALL_COMMANDS_BIT};
    PmSubmitInfo consumer{};
    consumer.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    consumer.waitCount = 1U;
    consumer.pWaits = &wait;
    consumer.commandBufferCount = 1U;
    consumer.pCommandBuffers = &work.commands;
    PM_CHECK(pmQueueSubmit(work.queue, &consumer, work.fence) == PM_SUCCESS);
    PM_CHECK(pmGetFenceStatus(work.device, work.fence) == PM_NOT_READY);

    const PmSemaphoreSubmitInfo signal{
        work.timeline, 1U, PM_STAGE_ALL_COMMANDS_BIT};
    PmSubmitInfo producer{};
    producer.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    producer.signalCount = 1U;
    producer.pSignals = &signal;
    if (delay_producer) set_delay(true);
    PM_CHECK(pmQueueSubmit(work.queue, &producer, nullptr) == PM_SUCCESS);
    if (delay_producer) set_delay(false);
    return work;
}

void expect_copy(PmDevice device, const Work &work, bool invalidate) {
    if (invalidate) {
        const PmMappedMemoryRange range{work.memory[1], 0U, buffer_size};
        PM_CHECK(pmInvalidateMappedMemoryRanges(device, 1U, &range) ==
                 PM_SUCCESS);
    }
    PM_CHECK(std::memcmp(work.mapped[0], work.mapped[1],
                         static_cast<std::size_t>(buffer_size)) == 0);
}

void destroy_completed(Work &work) {
    pmDestroyFence(work.device, work.fence);
    pmDestroySemaphore(work.device, work.timeline);
    pmDestroyCommandPool(work.device, work.pool);
    for (std::size_t index = 0U; index < work.memory.size(); ++index) {
        pmUnmapMemory(work.device, work.memory[index]);
        pmDestroyBuffer(work.device, work.buffers[index]);
        pmFreeMemory(work.device, work.memory[index]);
    }
    pmDestroyDevice(work.device);
    work.device = nullptr;
}

void exercise_replacement(PmPhysicalDevice physical,
                          const PmPhysicalDeviceFeatures &supported) {
    PmDevice device = create_device(physical, supported);
    PmQueue queue{};
    pmGetDeviceQueue(device, 0U, 0U, &queue);
    PM_CHECK(queue != nullptr);
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    PmCommandPool pool{};
    PM_CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    PmCommandBuffer commands{};
    PM_CHECK(pmAllocateCommandBuffers(device, pool, 1U, &commands) == PM_SUCCESS);
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    PM_CHECK(pmBeginCommandBuffer(commands, &begin) == PM_SUCCESS);
    PM_CHECK(pmEndCommandBuffer(commands) == PM_SUCCESS);
    PmFence fence{};
    PM_CHECK(pmCreateFence(device, 0U, &fence) == PM_SUCCESS);
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1U;
    submit.pCommandBuffers = &commands;
    PM_CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    PM_CHECK(pmWaitForFences(device, 1U, &fence, PM_TRUE,
                             PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    pmDestroyFence(device, fence);
    pmDestroyCommandPool(device, pool);
    pmDestroyDevice(device);
}

} // namespace

int main(int argc, char **argv) {
    const bool failure_fixture =
        argc == 2 && std::strcmp(argv[1], "--failure-fixture") == 0;
    set_delay(false);

    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_3;
    instance_info.enableValidation =
        std::getenv("PM_TEST_VALIDATION") != nullptr ? PM_TRUE : PM_FALSE;
    PmInstance instance{};
    PM_CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t count{};
    PM_CHECK(pmEnumeratePhysicalDevices(instance, &count, nullptr) == PM_SUCCESS);
    std::vector<PmPhysicalDevice> devices(count);
    PM_CHECK(pmEnumeratePhysicalDevices(instance, &count, devices.data()) ==
             PM_SUCCESS);
    PmPhysicalDevice physical{};
    PmPhysicalDeviceFeatures supported{};
    for (const auto candidate : devices) {
        pmGetPhysicalDeviceFeatures(candidate, &supported);
        if (supported.timelineSemaphores != PM_FALSE) {
            physical = candidate;
            break;
        }
    }
    if (physical == nullptr) {
        pmDestroyInstance(instance);
        return 77;
    }
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(physical, &properties);

    // Portable path: explicit fence completion precedes child/device teardown.
    Work explicit_work = submit_copy(physical, supported, failure_fixture);
    PM_CHECK(pmWaitForFences(explicit_work.device, 1U, &explicit_work.fence,
                             PM_TRUE, PM_TIMEOUT_INFINITE) == PM_SUCCESS);
    expect_copy(explicit_work.device, explicit_work, true);
    destroy_completed(explicit_work);

    // Reference-only path: device teardown sees accepted pending work and must
    // drain it before deleting its queue/services. Child handles are purposely
    // abandoned and never reused; the process owns reclamation after this test.
    Work implicit_work = submit_copy(physical, supported, failure_fixture);
    pmDestroyDevice(implicit_work.device);
    if (failure_fixture)
        expect_copy(nullptr, implicit_work, false);

    // A fresh device and its children must not inherit validation ownership or
    // completion state from the retired device, even if an allocator recycles
    // an opaque handle value.
    exercise_replacement(physical, supported);
    pmDestroyInstance(instance);
    std::printf(
        "PASS: explicit completion and reference teardown drain on %s "
        "(vendor=%04x device=%04x, host-output=%s)\n",
        properties.deviceName, properties.vendorId, properties.deviceId,
        failure_fixture ? "verified" : "native-drain");
}
