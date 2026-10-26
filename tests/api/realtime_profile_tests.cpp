// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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
    return pmCreateDevice(physical, &info, device);
}

} // namespace

int main(int argc, char **argv) {
    const bool validation = argc == 2 &&
        std::strcmp(argv[1], "validation") == 0;
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_3;
    instance_info.enableValidation = validation ? PM_TRUE : PM_FALSE;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);

    std::uint32_t count{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, nullptr) == PM_SUCCESS);
    CHECK(count == 1U);
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS);

    CHECK(pmGetInstanceProcAddr(instance, "pmEnumerateExecutionProfiles") ==
          nullptr);

    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue{0U, 1U, &priority};
    const char *retired_extension = "PM_EXT_simulation_profile";
    PmDeviceCreateInfo retired_device_info{};
    retired_device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    retired_device_info.queueCreateInfoCount = 1U;
    retired_device_info.pQueueInfos = &queue;
    retired_device_info.enabledExtensionCount = 1U;
    retired_device_info.ppEnabledExtensionNames = &retired_extension;
    PmDevice retired_device{};
    CHECK(pmCreateDevice(physical, &retired_device_info, &retired_device) ==
          PM_ERROR_EXTENSION_NOT_PRESENT);
    CHECK(retired_device == nullptr);

    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    PmDevice device{};
    CHECK(create_device(physical, &device) == PM_SUCCESS);
    RetiredProfileCreateInfo retired_profile{
        static_cast<PmStructureType>(19), nullptr, "realtime-single-step",
        1U, 0U};
    simulation_info.pNext = &retired_profile;
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_INVALID_ARGUMENT);
    CHECK(simulation == nullptr);
    simulation_info.pNext = nullptr;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_SUCCESS);

    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    PmPhysicsPipeline pipeline{};
    CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) ==
          PM_SUCCESS);
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    PmCommandPool pool{};
    CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
    PmCommandBuffer command{};
    CHECK(pmAllocateCommandBuffers(device, pool, 1U, &command) == PM_SUCCESS);
    PmCommandBufferBeginInfo begin{};
    begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CHECK(pmBeginCommandBuffer(command, &begin) == PM_SUCCESS);
    CHECK(pmCmdBindPhysicsPipeline(command, pipeline) == PM_SUCCESS);

    constexpr float expected = 1.0F / 60.0F;
    PmStepInfo step{};
    step.timestep = std::nextafter(expected, 0.0F);
    step.substeps = 1U;
    CHECK(pmCmdStepSimulation(command, simulation, &step) == PM_SUCCESS);
    step.timestep = std::nextafter(
        expected, std::numeric_limits<float>::infinity());
    CHECK(pmCmdStepSimulation(command, simulation, &step) == PM_SUCCESS);
    step.timestep = std::nextafter(
        step.timestep, std::numeric_limits<float>::infinity());
    CHECK(pmCmdStepSimulation(command, simulation, &step) ==
          PM_ERROR_INVALID_ARGUMENT);
    step.timestep = expected;
    step.substeps = 2U;
    CHECK(pmCmdStepSimulation(command, simulation, &step) ==
          PM_ERROR_INVALID_ARGUMENT);
    step.substeps = 1U;
    step.timestep = std::numeric_limits<float>::quiet_NaN();
    CHECK(pmCmdStepSimulation(command, simulation, &step) ==
          PM_ERROR_INVALID_ARGUMENT);

    pmFreeCommandBuffers(device, pool, 1U, &command);
    pmDestroyCommandPool(device, pool);
    pmDestroyPhysicsPipeline(device, pipeline);
    pmDestroySimulation(device, simulation);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    std::puts(validation
        ? "PASS: validation enforces sole fixed-tick runtime"
        : "PASS: driver enforces sole fixed-tick runtime");
    return 0;
}
