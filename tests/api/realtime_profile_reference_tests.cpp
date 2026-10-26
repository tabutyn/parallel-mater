// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <cstdint>
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
    PmPhysicalDeviceFeatures features{};
    pmGetPhysicalDeviceFeatures(physical, &features);

    count = 0U;
    CHECK(pmEnumerateExecutionProfiles(
              physical, &count, nullptr) == PM_SUCCESS);
    CHECK(count == 1U);
    PmExecutionProfileProperties property{};
    CHECK(pmEnumerateExecutionProfiles(
        physical, &count, &property) == PM_SUCCESS);
    CHECK(count == 1U);
    CHECK(std::strcmp(property.profileName,
                      PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP) == 0);
    CHECK(property.versionMajor == 1U && property.versionMinor == 0U);

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
          PM_SUCCESS);
    pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.couplingCapacity = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_SUCCESS);
    pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.couplingCapacity = 0U;
    const PmFluidCapacity fluid{1U, 1U};
    simulation_info.fluidCount = 1U;
    simulation_info.pFluidCapacities = &fluid;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          (features.particleLiquids
               ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT));
    if (simulation != nullptr) pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.fluidCount = 0U;
    const PmClothCapacity cloth{1U, 3U, 1U};
    simulation_info.clothCount = 1U;
    simulation_info.pClothCapacities = &cloth;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          (features.cloth ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT));
    if (simulation != nullptr) pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.clothCount = 0U;
    const PmRopeCapacity rope{2U};
    simulation_info.ropeCount = 1U;
    simulation_info.pRopeCapacities = &rope;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          (features.rope ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT));
    if (simulation != nullptr) pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.ropeCount = 0U;
    const PmSoftBodyCapacity soft{1U, 1U, 1U, 3U};
    simulation_info.softBodyCount = 1U;
    simulation_info.pSoftBodyCapacities = &soft;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          (features.softBodies ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT));
    if (simulation != nullptr) pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.softBodyCount = 0U;
    const PmSmokeCapacity smoke{1U, 1U};
    simulation_info.smokeCount = 1U;
    simulation_info.pSmokeCapacities = &smoke;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          (features.smoke ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT));
    if (simulation != nullptr) pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.smokeCount = 0U;

    simulation_info.fluidCount = features.particleLiquids;
    simulation_info.clothCount = features.cloth;
    simulation_info.ropeCount = features.rope;
    simulation_info.softBodyCount = features.softBodies;
    simulation_info.smokeCount = features.smoke;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_SUCCESS);
    PmSimulationMemoryRequirements requirements{};
    pmGetSimulationMemoryRequirements(device, simulation, &requirements);
    CHECK(requirements.persistent.size != 0U);
    CHECK(requirements.scratch.size != 0U);
    pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.fluidCount = 0U;
    simulation_info.clothCount = 0U;
    simulation_info.ropeCount = 0U;
    simulation_info.softBodyCount = 0U;
    simulation_info.smokeCount = 0U;
    simulation_info.emitterCapacity = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          (features.particleEmitters
               ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT));
    if (simulation != nullptr) pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.emitterCapacity = 0U;
    simulation_info.destroyerCapacity = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_SUCCESS);
    pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.destroyerCapacity = 0U;
    const PmPaintFieldCapacity paint{1U, 1U, 1U};
    simulation_info.paintFieldCount = 1U;
    simulation_info.pPaintFieldCapacities = &paint;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    simulation_info.paintFieldCount = 0U;
    simulation_info.paintRuleCapacity = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    simulation_info.paintRuleCapacity = 0U;
    simulation_info.debugFrameCapacity = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    simulation_info.debugFrameCapacity = 0U;
    profile.versionMinor = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_INCOMPATIBLE_VERSION);
    profile.versionMinor = 0U;
    profile.pProfileName = "unknown-profile";
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: reference driver exposes executable realtime profile");
    return 0;
}
