// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(expression) do {                                              \
    if (!(expression)) {                                                    \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);     \
        return 1;                                                           \
    }                                                                       \
} while (false)

namespace {

constexpr std::uint32_t worker_count = 8U;
constexpr std::uint32_t query_iterations = 256U;
constexpr std::uint32_t recording_iterations = 256U;

struct Diagnostics {
    std::atomic_uint32_t count{};
};

void PMAPI_CALL diagnostic_callback(
    void *user, const PmValidationDiagnostic *diagnostic) {
    auto &state = *static_cast<Diagnostics *>(user);
    state.count.fetch_add(1U, std::memory_order_relaxed);
    std::fprintf(stderr, "validation %s in %s: %s\n",
        diagnostic && diagnostic->pRuleId ? diagnostic->pRuleId : "unknown",
        diagnostic && diagnostic->pCommandName ? diagnostic->pCommandName : "unknown",
        diagnostic && diagnostic->pMessage ? diagnostic->pMessage : "no message");
}

bool same_properties(
    const PmPhysicalDeviceProperties &left,
    const PmPhysicalDeviceProperties &right) {
    return left.apiVersion == right.apiVersion &&
        left.driverVersion == right.driverVersion &&
        left.vendorId == right.vendorId && left.deviceId == right.deviceId &&
        std::memcmp(left.deviceUuid, right.deviceUuid,
                    sizeof left.deviceUuid) == 0 &&
        std::strncmp(left.deviceName, right.deviceName,
                     sizeof left.deviceName) == 0;
}

bool same_features(
    const PmPhysicalDeviceFeatures &left,
    const PmPhysicalDeviceFeatures &right) {
    return left.rigidBodies == right.rigidBodies &&
        left.rigidConstraints == right.rigidConstraints &&
        left.particleLiquids == right.particleLiquids &&
        left.liquidRigidCoupling == right.liquidRigidCoupling &&
        left.liquidLiquidCoupling == right.liquidLiquidCoupling &&
        left.cloth == right.cloth && left.rope == right.rope &&
        left.softBodies == right.softBodies && left.smoke == right.smoke &&
        left.domainCouplings == right.domainCouplings &&
        left.particleEmitters == right.particleEmitters &&
        left.paint == right.paint &&
        left.contactEvents == right.contactEvents &&
        left.debugData == right.debugData &&
        left.timelineSemaphores == right.timelineSemaphores &&
        left.timestampQueries == right.timestampQueries &&
        left.statisticsQueries == right.statisticsQueries &&
        left.memoryBudget == right.memoryBudget &&
        left.vulkanInterop == right.vulkanInterop;
}

bool same_memory(
    const PmMemoryProperties &left, const PmMemoryProperties &right) {
    if (left.heapCount != right.heapCount || left.typeCount != right.typeCount ||
        left.heapCount > 16U || left.typeCount > 32U)
        return false;
    for (std::uint32_t index = 0U; index < left.heapCount; ++index)
        if (left.heaps[index].size != right.heaps[index].size ||
            left.heaps[index].flags != right.heaps[index].flags)
            return false;
    for (std::uint32_t index = 0U; index < left.typeCount; ++index)
        if (left.types[index].propertyFlags !=
                right.types[index].propertyFlags ||
            left.types[index].heapIndex != right.types[index].heapIndex)
            return false;
    return true;
}

bool same_queue_families(
    const std::vector<PmQueueFamilyProperties> &left,
    const std::vector<PmQueueFamilyProperties> &right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0U; index < left.size(); ++index)
        if (left[index].queueFlags != right[index].queueFlags ||
            left[index].queueCount != right[index].queueCount ||
            left[index].timestampValidBits != right[index].timestampValidBits ||
            left[index].timestampPeriodNanoseconds !=
                right[index].timestampPeriodNanoseconds)
            return false;
    return true;
}

bool same_extensions(
    const std::vector<PmExtensionProperties> &left,
    const std::vector<PmExtensionProperties> &right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0U; index < left.size(); ++index)
        if (left[index].specificationVersion !=
                right[index].specificationVersion ||
            std::strncmp(left[index].extensionName, right[index].extensionName,
                         sizeof left[index].extensionName) != 0)
            return false;
    return true;
}

struct PhysicalSnapshot {
    std::vector<PmPhysicalDevice> devices;
    PmPhysicalDeviceProperties properties{};
    PmPhysicalDeviceFeatures features{};
    PmMemoryProperties memory{};
    std::vector<PmQueueFamilyProperties> queue_families;
    std::vector<PmExtensionProperties> extensions;
};

bool capture_physical_snapshot(
    PmInstance instance, PmPhysicalDevice physical,
    PhysicalSnapshot &snapshot) {
    std::uint32_t count{};
    if (pmEnumeratePhysicalDevices(instance, &count, nullptr) != PM_SUCCESS ||
        count == 0U)
        return false;
    snapshot.devices.assign(count, nullptr);
    if (pmEnumeratePhysicalDevices(instance, &count, snapshot.devices.data()) !=
            PM_SUCCESS ||
        count != snapshot.devices.size() ||
        std::find(snapshot.devices.begin(), snapshot.devices.end(), physical) ==
            snapshot.devices.end())
        return false;

    pmGetPhysicalDeviceProperties(physical, &snapshot.properties);
    pmGetPhysicalDeviceFeatures(physical, &snapshot.features);
    pmGetPhysicalDeviceMemoryProperties(physical, &snapshot.memory);

    count = 0U;
    if (pmGetPhysicalDeviceQueueFamilies(physical, &count, nullptr) !=
            PM_SUCCESS ||
        count == 0U)
        return false;
    snapshot.queue_families.assign(count, {});
    if (pmGetPhysicalDeviceQueueFamilies(
            physical, &count, snapshot.queue_families.data()) != PM_SUCCESS ||
        count != snapshot.queue_families.size())
        return false;

    count = 0U;
    if (pmEnumerateDeviceExtensionProperties(
            physical, nullptr, &count, nullptr) != PM_SUCCESS)
        return false;
    snapshot.extensions.assign(count, {});
    if (count != 0U &&
        pmEnumerateDeviceExtensionProperties(
            physical, nullptr, &count, snapshot.extensions.data()) !=
            PM_SUCCESS)
        return false;
    return count == snapshot.extensions.size();
}

bool matches_snapshot(
    PmInstance instance, PmPhysicalDevice physical,
    const PhysicalSnapshot &expected) {
    PhysicalSnapshot actual;
    return capture_physical_snapshot(instance, physical, actual) &&
        actual.devices == expected.devices &&
        same_properties(actual.properties, expected.properties) &&
        same_features(actual.features, expected.features) &&
        same_memory(actual.memory, expected.memory) &&
        same_queue_families(actual.queue_families,
                            expected.queue_families) &&
        same_extensions(actual.extensions, expected.extensions);
}

template<class Work>
std::uint32_t run_concurrently(Work work) {
    std::atomic_uint32_t ready{};
    std::atomic_bool start{};
    std::atomic_uint32_t failures{};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::uint32_t worker = 0U; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            ready.fetch_add(1U, std::memory_order_release);
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            if (!work(worker))
                failures.fetch_add(1U, std::memory_order_relaxed);
        });
    }
    while (ready.load(std::memory_order_acquire) != worker_count)
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (auto &worker : workers) worker.join();
    return failures.load(std::memory_order_relaxed);
}

PmNativeBackend expected_backend(const char *profile) {
    if (std::strcmp(profile, "cuda") == 0) return PM_NATIVE_BACKEND_CUDA;
    if (std::strcmp(profile, "metal") == 0) return PM_NATIVE_BACKEND_METAL;
    if (std::strcmp(profile, "d3d12") == 0) return PM_NATIVE_BACKEND_D3D12;
    if (std::strcmp(profile, "vulkan") == 0) return PM_NATIVE_BACKEND_VULKAN;
    return static_cast<PmNativeBackend>(0);
}

} // namespace

int main(int argc, char **argv) {
    CHECK(argc == 2);
    const PmNativeBackend backend = expected_backend(argv[1]);
    CHECK(backend != static_cast<PmNativeBackend>(0));

    Diagnostics diagnostics;
    PmValidationDiagnosticsCreateInfo diagnostic_info{};
    diagnostic_info.sType =
        PM_STRUCTURE_TYPE_VALIDATION_DIAGNOSTICS_CREATE_INFO;
    diagnostic_info.pfnDiagnostic = diagnostic_callback;
    diagnostic_info.pUserData = &diagnostics;
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pNext = &diagnostic_info;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    instance_info.pApplicationName = "Parallel Mater host concurrency test";

    PmInstance instance{};
    const PmResult instance_result =
        pmCreateInstance(&instance_info, nullptr, &instance);
    if (instance_result == PM_ERROR_DRIVER_NOT_FOUND) return 77;
    CHECK(instance_result == PM_SUCCESS);

    std::uint32_t physical_count{};
    CHECK(pmEnumeratePhysicalDevices(instance, &physical_count, nullptr) ==
          PM_SUCCESS);
    if (physical_count == 0U) {
        pmDestroyInstance(instance);
        return 77;
    }
    std::vector<PmPhysicalDevice> physical_devices(physical_count);
    CHECK(pmEnumeratePhysicalDevices(
              instance, &physical_count, physical_devices.data()) == PM_SUCCESS);
    const PmPhysicalDevice physical = physical_devices.front();

    PhysicalSnapshot expected;
    CHECK(capture_physical_snapshot(instance, physical, expected));
    CHECK(run_concurrently([&](std::uint32_t) {
        for (std::uint32_t iteration = 0U;
             iteration < query_iterations; ++iteration)
            if (!matches_snapshot(instance, physical, expected)) return false;
        return true;
    }) == 0U);

    std::uint32_t queue_family = PM_INVALID_INDEX;
    for (std::uint32_t index = 0U;
         index < expected.queue_families.size(); ++index)
        if (expected.queue_families[index].queueCount != 0U &&
            (expected.queue_families[index].queueFlags &
             PM_QUEUE_PHYSICS_BIT) != 0U) {
            queue_family = index;
            break;
        }
    CHECK(queue_family != PM_INVALID_INDEX);

    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_info{queue_family, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    PmDevice device{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    PmNativeDeviceInfo native{};
    CHECK(pmGetNativeDevice(device, &native) == PM_SUCCESS);
    CHECK(native.backend == backend && native.device != 0U);

    std::vector<PmCommandPool> pools(worker_count);
    std::vector<PmCommandBuffer> command_buffers(worker_count);
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family;
    for (std::uint32_t worker = 0U; worker < worker_count; ++worker) {
        CHECK(pmCreateCommandPool(device, &pool_info, &pools[worker]) ==
              PM_SUCCESS);
        CHECK(pmAllocateCommandBuffers(
                  device, pools[worker], 1U, &command_buffers[worker]) ==
              PM_SUCCESS);
    }

    const PmCommandBufferBeginInfo begin{
        PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
        PM_COMMAND_BUFFER_ONE_TIME_BIT};
    CHECK(run_concurrently([&](std::uint32_t worker) {
        for (std::uint32_t iteration = 0U;
             iteration < recording_iterations; ++iteration) {
            if (pmBeginCommandBuffer(command_buffers[worker], &begin) !=
                    PM_SUCCESS ||
                pmEndCommandBuffer(command_buffers[worker]) != PM_SUCCESS)
                return false;
            const PmResult reset = (iteration & 1U) == 0U
                ? pmResetCommandBuffer(command_buffers[worker])
                : pmResetCommandPool(device, pools[worker]);
            if (reset != PM_SUCCESS) return false;
        }
        return true;
    }) == 0U);
    CHECK(diagnostics.count.load(std::memory_order_relaxed) == 0U);

    for (std::uint32_t worker = 0U; worker < worker_count; ++worker) {
        pmFreeCommandBuffers(
            device, pools[worker], 1U, &command_buffers[worker]);
        pmDestroyCommandPool(device, pools[worker]);
    }
    pmDestroyDevice(device);
    pmDestroyInstance(instance);

    std::printf(
        "PASS: profile=%s device=\"%s\" vendor=0x%04x device=0x%04x "
        "threads=%u query_iterations=%u recording_iterations=%u\n",
        argv[1], expected.properties.deviceName, expected.properties.vendorId,
        expected.properties.deviceId, worker_count, query_iterations,
        recording_iterations);
    return 0;
}
