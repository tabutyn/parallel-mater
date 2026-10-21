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

PmResult create_device(PmPhysicalDevice physical, bool enable_profile,
                       PmDevice *device) {
    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue{0U, 1U, &priority};
    const char *extension = PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME;
    PmDeviceCreateInfo info{};
    info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1U;
    info.pQueueInfos = &queue;
    info.enabledExtensionCount = enable_profile ? 1U : 0U;
    info.ppEnabledExtensionNames = enable_profile ? &extension : nullptr;
    return pmCreateDevice(physical, &info, device);
}

} // namespace

int main(int argc, char **argv) {
    const bool validation = argc == 2 &&
        std::strcmp(argv[1], "validation") == 0;
    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    instance_info.enableValidation = validation ? PM_TRUE : PM_FALSE;
    PmInstance instance{};
    CHECK(pmCreateInstance(&instance_info, nullptr, &instance) == PM_SUCCESS);

    std::uint32_t count{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, nullptr) == PM_SUCCESS);
    CHECK(count == 1U);
    PmPhysicalDevice physical{};
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS);

    count = 0U;
    CHECK(pmEnumerateExecutionProfiles(
              physical, &count, nullptr) == PM_SUCCESS);
    CHECK(count == 1U);
    PmExecutionProfileProperties property{};
    CHECK(pmEnumerateExecutionProfiles(
              physical, &count, &property) == PM_SUCCESS);
    CHECK(std::strcmp(property.profileName,
                      PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP) == 0);
    CHECK(property.versionMajor == 1U);
    CHECK(property.versionMinor == 0U);

    PmSimulationProfileCreateInfo profile{};
    profile.sType = PM_STRUCTURE_TYPE_SIMULATION_PROFILE_CREATE_INFO;
    profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
    profile.versionMajor = 1U;
    profile.versionMinor = 0U;
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.pNext = &profile;
    simulation_info.rigidBodyCapacity = 1U;

    PmDevice legacy_device{};
    CHECK(create_device(physical, false, &legacy_device) == PM_SUCCESS);
    PmSimulation simulation{};
    CHECK(pmCreateSimulation(legacy_device, &simulation_info, &simulation) ==
          PM_ERROR_EXTENSION_NOT_PRESENT);
    CHECK(simulation == nullptr);
    simulation_info.pNext = nullptr;
    CHECK(pmCreateSimulation(legacy_device, &simulation_info, &simulation) ==
          PM_SUCCESS);
    pmDestroySimulation(legacy_device, simulation);
    pmDestroyDevice(legacy_device);

    PmDevice device{};
    CHECK(create_device(physical, true, &device) == PM_SUCCESS);
    simulation_info.pNext = &profile;
    profile.pProfileName = nullptr;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_INVALID_ARGUMENT);
    profile.pProfileName = "";
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_INVALID_ARGUMENT);
    profile.pProfileName = "unknown-profile";
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_FEATURE_NOT_PRESENT);
    profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
    ++profile.versionMinor;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_INCOMPATIBLE_VERSION);
    --profile.versionMinor;
    PmSimulationProfileCreateInfo nested = profile;
    profile.pNext = &nested;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          PM_ERROR_INVALID_ARGUMENT);
    profile.pNext = nullptr;
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
        ? "PASS: validation enforces realtime profile negotiation"
        : "PASS: driver enforces realtime profile negotiation");
    return 0;
}
