// SPDX-License-Identifier: MIT
// Public SDK synchronization only; unsatisfied timelines hold work pending
// deterministically, without sleeps or a GPU-time threshold.
#include <parallel_mater/parallel_mater.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <latch>
#include <vector>

#define CHECK(value) do { if (!(value)) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #value); \
    std::abort(); \
} } while (false)

int main(int argc, char **argv) {
    const bool allow_no_device = argc == 2 &&
        std::strcmp(argv[1], "--allow-no-device") == 0;
    CHECK(argc == 1 || allow_no_device);
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_3;
    instance_info.enableValidation = std::getenv("PM_TEST_VALIDATION") ? PM_TRUE : PM_FALSE;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    std::uint32_t count{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, nullptr) == PM_SUCCESS);
    if (count == 0U && allow_no_device) {
        pmDestroyInstance(instance);
        std::puts("SKIP: requested native GPU is unavailable");
        return 77;
    }
    CHECK(count != 0U);
    count = 1U;
    PmPhysicalDevice physical{};
    const auto enumerated = pmEnumeratePhysicalDevices(instance, &count, &physical);
    CHECK((enumerated == PM_SUCCESS || enumerated == PM_INCOMPLETE) && count == 1U);
    PmPhysicalDeviceFeatures features{};
    pmGetPhysicalDeviceFeatures(physical, &features);
    CHECK(features.timelineSemaphores == PM_TRUE);
    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures.timelineSemaphores = PM_TRUE;
    PmDevice device{}, foreign_device{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    CHECK(pmCreateDevice(physical, &device_info, &foreign_device) == PM_SUCCESS);
    PmQueue queue{};
    pmGetDeviceQueue(device, 0U, 0U, &queue);
    CHECK(queue != nullptr);
    std::array<PmFence, 2> fences{};
    for (auto &fence : fences) CHECK(pmCreateFence(device, 0U, &fence) == PM_SUCCESS);
    PmFence signaled{}, foreign{};
    CHECK(pmCreateFence(device, PM_FENCE_SIGNALED_BIT, &signaled) == PM_SUCCESS);
    CHECK(pmCreateFence(foreign_device, PM_FENCE_SIGNALED_BIT, &foreign) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &signaled, PM_TRUE, 0U) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 1U, &fences[0], PM_TRUE, 0U) == PM_TIMEOUT);
    CHECK(pmWaitForFences(device, 0U, fences.data(), PM_TRUE, 0U) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(pmWaitForFences(device, 1U, nullptr, PM_TRUE, 0U) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(pmWaitForFences(nullptr, 1U, &signaled, PM_TRUE, 0U) == PM_ERROR_INVALID_ARGUMENT);
    const std::array<PmFence, 2> wrong_device{signaled, foreign};
    CHECK(pmWaitForFences(device, 2U, wrong_device.data(), PM_FALSE, 0U) == PM_ERROR_INVALID_ARGUMENT);

    PmSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = PM_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    PmSemaphore timeline{};
    CHECK(pmCreateSemaphore(device, &semaphore_info, &timeline) == PM_SUCCESS);
    CHECK(pmWaitSemaphore(device, timeline, 0U, 0U) == PM_SUCCESS);
    CHECK(pmWaitSemaphore(device, timeline, 1U, 0U) == PM_TIMEOUT);
    CHECK(pmWaitSemaphore(foreign_device, timeline, 0U, 0U) == PM_ERROR_INVALID_ARGUMENT);
    for (std::uint32_t index = 0U; index < fences.size(); ++index) {
        const PmSemaphoreSubmitInfo wait{timeline, index + 1U, PM_STAGE_ALL_COMMANDS_BIT};
        PmSubmitInfo submit{};
        submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitCount = 1U;
        submit.pWaits = &wait;
        CHECK(pmQueueSubmit(queue, &submit, fences[index]) == PM_SUCCESS);
        CHECK(pmGetFenceStatus(device, fences[index]) == PM_NOT_READY);
    }
    CHECK(pmResetFences(device, 2U, fences.data()) == PM_ERROR_INVALID_STATE);
    CHECK(pmWaitForFences(device, 2U, fences.data(), PM_TRUE, 1000U) == PM_TIMEOUT);
    CHECK(pmWaitForFences(device, 2U, fences.data(), PM_FALSE, 1000U) == PM_TIMEOUT);
    CHECK(pmWaitSemaphore(device, timeline, 1U, 1000U) == PM_TIMEOUT);
    const std::array<PmFence, 2> ready_and_pending{signaled, fences[0]};
    CHECK(pmWaitForFences(device, 2U, ready_and_pending.data(), PM_FALSE, 0U) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 2U, ready_and_pending.data(), PM_TRUE, 0U) == PM_TIMEOUT);

    std::latch entered{4};
    auto single = std::async(std::launch::async, [&] {
        entered.count_down();
        return pmWaitForFences(device, 1U, &fences[0], PM_TRUE, UINT64_MAX - 1U);
    });
    auto any = std::async(std::launch::async, [&] {
        entered.count_down();
        return pmWaitForFences(device, 2U, fences.data(), PM_FALSE, PM_TIMEOUT_INFINITE);
    });
    auto all = std::async(std::launch::async, [&] {
        entered.count_down();
        return pmWaitForFences(device, 2U, fences.data(), PM_TRUE, PM_TIMEOUT_INFINITE);
    });
    auto semaphore = std::async(std::launch::async, [&] {
        entered.count_down();
        return pmWaitSemaphore(device, timeline, 3U, PM_TIMEOUT_INFINITE);
    });
    entered.wait();
    const auto signal = [&](std::uint64_t value) {
        const PmSemaphoreSubmitInfo update{timeline, value, PM_STAGE_ALL_COMMANDS_BIT};
        PmSubmitInfo producer{};
        producer.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
        producer.signalCount = 1U;
        producer.pSignals = &update;
        CHECK(pmQueueSubmit(queue, &producer, nullptr) == PM_SUCCESS);
    };
    signal(1U);
    CHECK(single.get() == PM_SUCCESS);
    CHECK(any.get() == PM_SUCCESS);
    CHECK(pmGetFenceStatus(device, fences[1]) == PM_NOT_READY);
    CHECK(all.wait_for(std::chrono::seconds{0}) == std::future_status::timeout);
    CHECK(semaphore.wait_for(std::chrono::seconds{0}) == std::future_status::timeout);
    signal(2U);
    CHECK(all.get() == PM_SUCCESS);
    CHECK(semaphore.wait_for(std::chrono::seconds{0}) == std::future_status::timeout);
    signal(3U);
    CHECK(semaphore.get() == PM_SUCCESS);
    std::uint64_t value{};
    CHECK(pmGetSemaphoreCounterValue(device, timeline, &value) == PM_SUCCESS && value == 3U);
    CHECK(pmWaitSemaphore(device, timeline, 3U, 0U) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 2U, fences.data(), PM_TRUE, 0U) == PM_SUCCESS);
    CHECK(pmResetFences(device, 2U, fences.data()) == PM_SUCCESS);
    CHECK(pmWaitForFences(device, 2U, fences.data(), PM_FALSE, 0U) == PM_TIMEOUT);
    CHECK(pmQueueWaitIdle(queue) == PM_SUCCESS);
    for (const auto fence : fences) pmDestroyFence(device, fence);
    pmDestroyFence(device, signaled);
    pmDestroyFence(foreign_device, foreign);
    pmDestroySemaphore(device, timeline);
    pmDestroyDevice(foreign_device);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: completion waits preserve pending, ready, timeout, any/all and timeline ordering");
}
