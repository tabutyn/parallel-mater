#include "parallel_mater/command.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

// Do not use assert: this test must remain effective in Release/NDEBUG builds.
#define CHECK(expression) do { if (!(expression)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression); return 1; \
} } while (false)

int main()
{
    uint32_t version = UINT32_MAX;
    CHECK(pmEnumerateInstanceVersion(&version) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(version == 0); // No supported implementation version is invented.
    CHECK(pmEnumerateInstanceVersion(nullptr) == PM_ERROR_INVALID_ARGUMENT);

    auto instance = reinterpret_cast<PmInstance>(uintptr_t{1});
    PmInstanceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = PM_API_VERSION_0_2;
    CHECK(pmCreateInstance(&info, nullptr, &instance) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(instance == nullptr);
    CHECK(pmCreateInstance(&info, nullptr, nullptr) == PM_ERROR_INVALID_ARGUMENT);
    uint32_t count = 27;
    CHECK(pmEnumeratePhysicalDevices(nullptr, &count, nullptr) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(count == 0);

    auto loader = pmGetInstanceProcAddr(nullptr, "pmGetInstanceProcAddr");
    CHECK(loader == reinterpret_cast<PFN_pmVoidFunction>(pmGetInstanceProcAddr));
    auto create = reinterpret_cast<PFN_pmCreateInstance>(
        pmGetInstanceProcAddr(nullptr, "pmCreateInstance"));
    CHECK(create != nullptr);
    CHECK(create(&info, nullptr, &instance) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(instance == nullptr);
    CHECK(pmGetInstanceProcAddr(nullptr, "pmQueueSubmit") == nullptr);
    CHECK(pmGetInstanceProcAddr(nullptr, "pmcreateinstance") == nullptr);
    CHECK(pmGetInstanceProcAddr(nullptr, "") == nullptr);
    CHECK(pmGetInstanceProcAddr(nullptr, nullptr) == nullptr);
    CHECK(pmGetInstanceProcAddr(reinterpret_cast<PmInstance>(uintptr_t{1}),
                                "pmCreateInstance") == nullptr);
    CHECK(pmGetDeviceProcAddr(nullptr, "pmQueueSubmit") == nullptr);

    PmPhysicalDeviceFeatures features;
    std::memset(&features, 0x7f, sizeof features);
    pmGetPhysicalDeviceFeatures(nullptr, &features);
    CHECK(features.rigidBodies == PM_FALSE && features.vulkanInterop == PM_FALSE);
    PmMemoryRequirements requirements;
    std::memset(&requirements, 0x7f, sizeof requirements);
    pmGetBufferMemoryRequirements(nullptr, nullptr, &requirements);
    CHECK(requirements.size == 0 && requirements.alignment == 0 && requirements.memoryTypeBits == 0);
    void* mapping = &features;
    CHECK(pmMapMemory(nullptr, nullptr, 0, 64, &mapping) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(mapping == nullptr);

    PmCommandBuffer buffers[3] = {
        reinterpret_cast<PmCommandBuffer>(uintptr_t{1}),
        reinterpret_cast<PmCommandBuffer>(uintptr_t{2}),
        reinterpret_cast<PmCommandBuffer>(uintptr_t{3})};
    CHECK(pmAllocateCommandBuffers(nullptr, nullptr, 3, buffers) == PM_ERROR_FEATURE_NOT_PRESENT);
    for (auto buffer : buffers) CHECK(buffer == nullptr);
    CHECK(pmAllocateCommandBuffers(nullptr, nullptr, 1, nullptr) == PM_ERROR_INVALID_ARGUMENT);
    CHECK(pmQueueSubmit(nullptr, nullptr, nullptr) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(pmWaitForFences(nullptr, 0, nullptr, PM_TRUE, PM_TIMEOUT_INFINITE) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(pmWaitSemaphore(nullptr, nullptr, 1, PM_TIMEOUT_INFINITE) == PM_ERROR_FEATURE_NOT_PRESENT);

    for (auto backend : {PM_NATIVE_BACKEND_CUDA, PM_NATIVE_BACKEND_METAL,
                         PM_NATIVE_BACKEND_D3D12, PM_NATIVE_BACKEND_VULKAN}) {
        PmNativeInteropProperties properties{PM_TRUE, PM_TRUE, PM_TRUE};
        CHECK(pmGetPhysicalDeviceInteropProperties(nullptr, backend, &properties) == PM_ERROR_FEATURE_NOT_PRESENT);
        CHECK(properties.deviceImport == PM_FALSE && properties.bufferImport == PM_FALSE &&
              properties.timelineImport == PM_FALSE);
    }
    PmResult execution = PM_SUCCESS;
    CHECK(pmGetSimulationStatus(nullptr, nullptr, &execution) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(execution == PM_ERROR_FEATURE_NOT_PRESENT);
    uint64_t result = UINT64_MAX;
    CHECK(pmGetQueryPoolResults(nullptr, nullptr, 0, 1, sizeof result, &result,
                               sizeof result, 0) == PM_ERROR_FEATURE_NOT_PRESENT);
    CHECK(result == UINT64_MAX); // Failed readback never creates a valid sample.
    pmDestroyInstance(nullptr);
    pmDestroyDevice(nullptr);
    std::puts("PASS: command stubs fail closed; no resources, GPU work or waits");
}
