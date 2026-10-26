/* SPDX-License-Identifier: MIT */
#include <parallel_mater/parallel_mater.h>
#include <stdio.h>
#include <stdlib.h>

/* No graphics context, shader compiler, renderer or private driver API. */
int main(void) {
    PmInstanceCreateInfo info = {0};
    PmInstance instance = NULL;
    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = PM_API_VERSION_0_3;
    info.pApplicationName = "pm-devices";
    PmResult result = pmCreateInstance(&info, NULL, &instance);
    if (result != PM_SUCCESS) {
        fprintf(stderr, "Create instance: %d; select an installed driver with PM_DRIVER_FILES\n", result);
        return 1;
    }
    uint32_t count = 0;
    result = pmEnumeratePhysicalDevices(instance, &count, NULL);
    PmPhysicalDevice *devices = count ? calloc(count, sizeof(*devices)) : NULL;
    if (result != PM_SUCCESS || (count && !devices)) {
        pmDestroyInstance(instance);
        free(devices);
        return 1;
    }
    if (count) result = pmEnumeratePhysicalDevices(instance, &count, devices);
    if (result == PM_SUCCESS) {
        for (uint32_t index = 0; index < count; ++index) {
            PmPhysicalDeviceProperties properties = {0};
            pmGetPhysicalDeviceProperties(devices[index], &properties);
            printf("%s (vendor %u, device %u)\n", properties.deviceName,
                   properties.vendorId, properties.deviceId);
            uint32_t capability_count = 0;
            result = pmEnumeratePhysicalDeviceInteractionCapabilities(
                devices[index], &capability_count, NULL);
            PmInteractionCapability *capabilities = capability_count
                ? calloc(capability_count, sizeof(*capabilities)) : NULL;
            if (result != PM_SUCCESS ||
                (capability_count != 0 && capabilities == NULL)) {
                free(capabilities);
                break;
            }
            if (capability_count != 0)
                result = pmEnumeratePhysicalDeviceInteractionCapabilities(
                    devices[index], &capability_count, capabilities);
            for (uint32_t capability_index = 0;
                 result == PM_SUCCESS && capability_index < capability_count;
                 ++capability_index) {
                PmInteractionCapability queried = {0};
                result = pmGetPhysicalDeviceInteractionCapability(
                    devices[index], &capabilities[capability_index].key,
                    &queried);
                if (result == PM_SUCCESS &&
                    queried.maxGraphEdgesPerSimulation !=
                        capabilities[capability_index].maxGraphEdgesPerSimulation)
                    result = PM_ERROR_INVALID_STATE;
            }
            free(capabilities);
            if (result != PM_SUCCESS) break;
        }
    }
    free(devices);
    pmDestroyInstance(instance);
    return result == PM_SUCCESS ? 0 : 1;
}
