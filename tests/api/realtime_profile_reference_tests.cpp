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

struct RetiredProfileCreateInfo {
    PmStructureType sType;
    const void *pNext;
    const char *pProfileName;
    std::uint32_t versionMajor;
    std::uint32_t versionMinor;
};

PmResult create_device(PmPhysicalDevice physical, PmDevice *device) {
    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue{0U, 1U, &priority};
    PmDeviceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1U;
    info.pQueueInfos = &queue;
    pmGetPhysicalDeviceFeatures(physical, &info.enabledFeatures);
    return pmCreateDevice(physical, &info, device);
}

} // namespace

int main() {
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_3;
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

    CHECK(pmGetInstanceProcAddr(instance, "pmEnumerateExecutionProfiles") ==
          nullptr);

    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    PmSimulation simulation{};

    PmDevice device{};
    const PmResult device_result = create_device(physical, &device);
    if (device_result != PM_SUCCESS) {
        std::fprintf(stderr, "SKIP: device creation returned %d\n",
                     static_cast<int>(device_result));
        pmDestroyInstance(instance);
        return 77;
    }
    RetiredProfileCreateInfo retired_profile{
        static_cast<PmStructureType>(19), nullptr, "realtime-single-step",
        1U, 0U};
    simulation_info.pNext = &retired_profile;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(simulation == nullptr);
    simulation_info.pNext = nullptr;
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
          (features.paint ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT));
    if (simulation != nullptr) pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.paintFieldCount = 0U;
    simulation_info.paintRuleCapacity = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          (features.paint ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT));
    if (simulation != nullptr) pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.paintRuleCapacity = 0U;
    simulation_info.debugFrameCapacity = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          (features.debugData ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT));
    if (simulation != nullptr) pmDestroySimulation(device, simulation);
    simulation = nullptr;
    simulation_info.debugFrameCapacity = 0U;
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts("PASS: reference driver exposes one fixed-tick runtime");
    return 0;
}
