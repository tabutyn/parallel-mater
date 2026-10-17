// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
#include <parallel_mater/interop_vulkan.h>
#endif

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void PMAPI_CALL pm_testManualLayerReset();
extern "C" std::uint64_t PMAPI_CALL pm_testManualLayerPreMask();
extern "C" std::uint64_t PMAPI_CALL pm_testManualLayerPostMask();
extern "C" std::uint64_t PMAPI_CALL pm_testManualLayerExpectedMask();
extern "C" std::uint32_t PMAPI_CALL pm_testManualLayerErrors();

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

int main(int argc, char **argv) {
    const bool legacy = argc > 1 && std::strcmp(argv[1], "legacy") == 0;
    pm_testManualLayerReset();

    std::uint32_t version{};
    CHECK(pmEnumerateInstanceVersion(&version) == PM_SUCCESS);
    CHECK(version == PM_API_VERSION_0_2);
    CHECK(pmGetInstanceProcAddr(nullptr, "pmCreateInstance") != nullptr);
    std::uint32_t count{};
    CHECK(pmEnumerateInstanceLayerProperties(&count, nullptr) == PM_SUCCESS);
    CHECK(pmEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) ==
          PM_SUCCESS);

    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    instance_info.enableValidation = PM_TRUE;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);
    CHECK(pmGetInstanceProcAddr(instance, "pmEnumeratePhysicalDevices") !=
          nullptr);
    CHECK(pmEnumerateInstanceVersion(&version) == PM_SUCCESS);
    CHECK(pmEnumerateInstanceLayerProperties(&count, nullptr) == PM_SUCCESS);
    CHECK(pmEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) ==
          PM_SUCCESS);

    CHECK(pmEnumeratePhysicalDevices(instance, &count, nullptr) == PM_SUCCESS);
    CHECK(count != 0U);
    std::vector<PmPhysicalDevice> physicals(count);
    CHECK(pmEnumeratePhysicalDevices(instance, &count, physicals.data()) ==
          PM_SUCCESS);
    PmPhysicalDevice physical{};
    for (PmPhysicalDevice candidate : physicals) {
        PmPhysicalDeviceProperties properties{};
        pmGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.vendorId == 0x504dU) {
            physical = candidate;
            break;
        }
    }
    CHECK(physical != nullptr);

    std::uint32_t extension_count{};
    CHECK(pmEnumerateDeviceExtensionProperties(
              physical, nullptr, &extension_count, nullptr) == PM_SUCCESS);
    CHECK(extension_count == 0U);

    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    PmDevice device{};
    CHECK(pmCreateDevice(physical, &device_info, &device) == PM_SUCCESS);
    CHECK(pmGetDeviceProcAddr(device, "pmQueueSubmit") != nullptr);

    PmQueue queue{};
    pmGetDeviceQueue(device, 7U, 3U, &queue);
    CHECK(queue != nullptr);
    auto pool = reinterpret_cast<PmCommandPool>(std::uintptr_t{0x1234U});
    PmCommandBuffer buffers[2]{};
    CHECK(pmAllocateCommandBuffers(device, pool, 2U, buffers) == PM_SUCCESS);
    CHECK(buffers[0] != nullptr && buffers[1] != nullptr);
    PmSubmitInfo submit{};
    submit.sType = PM_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 2U;
    submit.pCommandBuffers = buffers;
    auto fence = reinterpret_cast<PmFence>(std::uintptr_t{0x5678U});
    CHECK(pmQueueSubmit(queue, &submit, fence) == PM_SUCCESS);
    pmFreeCommandBuffers(device, pool, 2U, buffers);

    PmNativeDeviceImportInfo native_info{};
    native_info.sType = PM_STRUCTURE_TYPE_NATIVE_DEVICE_IMPORT_INFO;
    PmDevice imported{};
    CHECK(pmImportNativeDevice(instance, &native_info, &imported) == PM_SUCCESS);
    pmDestroyDevice(imported);

#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
    PmVulkanDeviceImportInfo vulkan_info{};
    PmDevice vulkan_imported{};
    CHECK(pmImportVulkanDeviceEXT(instance, &vulkan_info, &vulkan_imported) ==
          PM_SUCCESS);
    pmDestroyDevice(vulkan_imported);
#endif

    pmDestroyDevice(device);
    pmDestroyInstance(instance);

    if (!legacy) {
        const std::uint64_t expected = pm_testManualLayerExpectedMask();
        CHECK(pm_testManualLayerErrors() == 0U);
        CHECK(pm_testManualLayerPreMask() == expected);
        CHECK(pm_testManualLayerPostMask() == expected);
    }
    std::puts(legacy
        ? "PASS: ABI v1 layer accepts every intercepted manual entrypoint"
        : "PASS: typed layer intercepts every inventoried manual entrypoint");
    return 0;
}
