/* SPDX-License-Identifier: MIT
 * Installed-SDK real-time profile negotiation checks. */
#include <parallel_mater/parallel_mater.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expression) do {                                              \
    if (!(expression)) {                                                    \
        fprintf(stderr, "line %d: %s\n", __LINE__, #expression);           \
        return 1;                                                           \
    }                                                                       \
} while (0)

static PmResult create_device(PmPhysicalDevice physical, int enable_profile,
                              PmDevice* device) {
    static const float priority = 1.0F;
    static const char* extension =
        PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME;
    PmDeviceQueueCreateInfo queue = {0U, 1U, &priority};
    PmDeviceCreateInfo info = {0};
    info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = 1U;
    info.pQueueInfos = &queue;
    info.enabledExtensionCount = enable_profile ? 1U : 0U;
    info.ppEnabledExtensionNames = enable_profile ? &extension : NULL;
    return pmCreateDevice(physical, &info, device);
}

int main(int argc, char** argv) {
    const int validation = argc == 2 && strcmp(argv[1], "validation") == 0;
    PmInstanceCreateInfo instance_info = {0};
    PmInstance instance = NULL;
    PmPhysicalDevice physical = NULL;
    uint32_t count = 1U;
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_2;
    instance_info.enableValidation = validation ? PM_TRUE : PM_FALSE;
    CHECK(pmCreateInstance(&instance_info, NULL, &instance) == PM_SUCCESS);
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS);
    CHECK(count == 1U && physical != NULL);

    count = 0U;
    CHECK(pmEnumerateExecutionProfiles(
              physical, &count, NULL) == PM_SUCCESS);
    CHECK(count == 1U);
    {
        PmExecutionProfileProperties property = {0};
        CHECK(pmEnumerateExecutionProfiles(
                  physical, &count, &property) == PM_SUCCESS);
        CHECK(strcmp(property.profileName,
                     PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP) == 0);
        CHECK(property.versionMajor == 1U && property.versionMinor == 0U);
    }

    {
        PmSimulationProfileCreateInfo profile = {0};
        PmSimulationCreateInfo simulation_info = {0};
        PmSimulation simulation = NULL;
        PmDevice device = NULL;
        profile.sType = PM_STRUCTURE_TYPE_SIMULATION_PROFILE_CREATE_INFO;
        profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
        profile.versionMajor = 1U;
        profile.versionMinor = 0U;
        simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
        simulation_info.pNext = &profile;
        simulation_info.rigidBodyCapacity = 1U;

        CHECK(create_device(physical, 0, &device) == PM_SUCCESS);
        CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
              PM_ERROR_EXTENSION_NOT_PRESENT);
        pmDestroyDevice(device);

        CHECK(create_device(physical, 1, &device) == PM_SUCCESS);
        profile.pProfileName = "unknown-profile";
        CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
              PM_ERROR_FEATURE_NOT_PRESENT);
        profile.pProfileName = PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP;
        profile.versionMinor = 1U;
        CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
              PM_ERROR_INCOMPATIBLE_VERSION);
        profile.versionMinor = 0U;
        CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
              PM_SUCCESS);

        {
            PmPhysicsPipelineCreateInfo pipeline_info = {0};
            PmPhysicsPipeline pipeline = NULL;
            PmCommandPoolCreateInfo pool_info = {0};
            PmCommandPool pool = NULL;
            PmCommandBuffer command = NULL;
            PmCommandBufferBeginInfo begin = {0};
            PmStepInfo step = {0};
            union { float value; uint32_t bits; } timestep;
            pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
            pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            CHECK(pmCreatePhysicsPipeline(device, &pipeline_info, &pipeline) ==
                  PM_SUCCESS);
            CHECK(pmCreateCommandPool(device, &pool_info, &pool) == PM_SUCCESS);
            CHECK(pmAllocateCommandBuffers(device, pool, 1U, &command) ==
                  PM_SUCCESS);
            CHECK(pmBeginCommandBuffer(command, &begin) == PM_SUCCESS);
            CHECK(pmCmdBindPhysicsPipeline(command, pipeline) == PM_SUCCESS);
            timestep.value = 1.0F / 60.0F;
            step.timestep = timestep.value;
            step.substeps = 1U;
            CHECK(pmCmdStepSimulation(command, simulation, &step) == PM_SUCCESS);
            timestep.bits += 2U;
            step.timestep = timestep.value;
            CHECK(pmCmdStepSimulation(command, simulation, &step) ==
                  PM_ERROR_INVALID_ARGUMENT);
            step.timestep = 1.0F / 60.0F;
            step.substeps = 2U;
            CHECK(pmCmdStepSimulation(command, simulation, &step) ==
                  PM_ERROR_INVALID_ARGUMENT);
            pmFreeCommandBuffers(device, pool, 1U, &command);
            pmDestroyCommandPool(device, pool);
            pmDestroyPhysicsPipeline(device, pipeline);
        }
        pmDestroySimulation(device, simulation);
        pmDestroyDevice(device);
    }
    pmDestroyInstance(instance);
    puts(validation
        ? "PASS: installed validation profile contract"
        : "PASS: installed driver profile contract");
    return 0;
}
