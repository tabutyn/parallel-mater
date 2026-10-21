// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(expression) do { if (!(expression)) {                         \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expression);          \
    return 1;                                                               \
} } while (false)

namespace {

PmResult create_device(PmPhysicalDevice physical, bool enable_profile,
                       PmDevice *device) {
    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue{0U, 1U, &priority};
    const char *extension = PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME;
    PmDeviceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1U;
    info.pQueueInfos = &queue;
    pmGetPhysicalDeviceFeatures(physical, &info.enabledFeatures);
    info.enabledExtensionCount = enable_profile ? 1U : 0U;
    info.ppEnabledExtensionNames = enable_profile ? &extension : nullptr;
    return pmCreateDevice(physical, &info, device);
}

} // namespace

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    const PmResult instance_result =
        pmCreateInstance(&instance_info, nullptr, &instance);
    if (instance_result != PM_SUCCESS) {
        std::fprintf(stderr, "SKIP: pmCreateInstance returned %d\n",
                     static_cast<int>(instance_result));
        return 77;
    }
    std::uint32_t count{};
    if (pmEnumeratePhysicalDevices(instance, &count, nullptr) != PM_SUCCESS ||
        count == 0U) {
        std::fputs("SKIP: reference driver exposed no physical device\n",
                   stderr);
        pmDestroyInstance(instance);
        return 77;
    }
    std::vector<PmPhysicalDevice> physicals(count);
    CHECK(pmEnumeratePhysicalDevices(instance, &count, physicals.data()) ==
          PM_SUCCESS);
    const PmPhysicalDevice physical = physicals.front();

    count = 99U;
    CHECK(pmEnumerateExecutionProfiles(
              physical, &count, nullptr) == PM_SUCCESS);
    CHECK(count == 0U);

    PmSimulationProfileCreateInfo profile{};
    profile.sType = PM_STRUCTURE_TYPE_SIMULATION_PROFILE_CREATE_INFO;
    profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
    profile.versionMajor = 1U;
    profile.versionMinor = 0U;
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.pNext = &profile;
    simulation_info.rigidBodyCapacity = 1U;
    PmSimulation simulation{};

    PmDevice legacy_device{};
    const PmResult legacy_device_result =
        create_device(physical, false, &legacy_device);
    if (legacy_device_result != PM_SUCCESS) {
        std::fprintf(stderr, "SKIP: legacy device creation returned %d\n",
                     static_cast<int>(legacy_device_result));
        pmDestroyInstance(instance);
        return 77;
    }
    CHECK(pmCreateSimulation(legacy_device, &simulation_info, &simulation) ==
          PM_ERROR_EXTENSION_NOT_PRESENT);
    simulation_info.pNext = nullptr;
    CHECK(pmCreateSimulation(legacy_device, &simulation_info, &simulation) ==
          PM_SUCCESS);
    pmDestroySimulation(legacy_device, simulation);
    pmDestroyDevice(legacy_device);

    PmDevice device{};
    const PmResult profile_device_result =
        create_device(physical, true, &device);
    if (profile_device_result != PM_SUCCESS) {
        std::fprintf(stderr, "SKIP: profile device creation returned %d\n",
                     static_cast<int>(profile_device_result));
        pmDestroyInstance(instance);
        return 77;
    }
    simulation_info.pNext = &profile;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    profile.versionMinor = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_INCOMPATIBLE_VERSION);
    profile.versionMinor = 0U;
    profile.pProfileName = "unknown-profile";
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: reference driver advertises no unfinished realtime solver");
    return 0;
}
