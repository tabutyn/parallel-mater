/* SPDX-License-Identifier: MIT
 * Installed-SDK compatibility checks. This fixture intentionally uses only
 * the public C application ABI. */
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

static PmResult create_instance(uint32_t version, int enable_layer,
                                PmInstance* instance) {
    static const char* validation = "PM_LAYER_PARALLEL_MATER_validation";
    PmInstanceCreateInfo info = {0};
    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = version;
    if (enable_layer) {
        info.enabledLayerCount = 1U;
        info.ppEnabledLayerNames = &validation;
    }
    return pmCreateInstance(&info, NULL, instance);
}

static int expect_application_version(uint32_t version, PmResult expected) {
    PmInstance instance = NULL;
    CHECK(create_instance(version, 0, &instance) == expected);
    if (expected == PM_SUCCESS) {
        CHECK(instance != NULL);
        pmDestroyInstance(instance);
    } else {
        CHECK(instance == NULL);
    }
    return 0;
}

static int check_instance_extensions(void) {
    uint32_t count = UINT32_MAX;
    PmInstance instance = NULL;
    const char* unknown = "PM_EXT_unknown";
    PmInstanceCreateInfo info = {0};

    CHECK(pmEnumerateInstanceExtensionProperties(NULL, &count, NULL) ==
          PM_SUCCESS);
    CHECK(count == 0U);
    CHECK(pmEnumerateInstanceExtensionProperties(
              "PM_LAYER_unknown", &count, NULL) == PM_ERROR_LAYER_NOT_PRESENT);

    info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.apiVersion = PM_API_VERSION_0_2;
    info.enabledExtensionCount = 1U;
    info.ppEnabledExtensionNames = &unknown;
    CHECK(pmCreateInstance(&info, NULL, &instance) ==
          PM_ERROR_EXTENSION_NOT_PRESENT);
    CHECK(instance == NULL);
    return 0;
}

static int check_device_extensions(void) {
    PmInstance instance = NULL;
    PmPhysicalDevice physical = NULL;
    PmDevice device = NULL;
    PmDeviceCreateInfo info = {0};
    PmExtensionProperties extension = {0};
    const char* unknown = "PM_EXT_unknown";
    uint32_t count = 1U;
    uint32_t extension_count = UINT32_MAX;

    CHECK(create_instance(PM_API_VERSION_0_2, 0, &instance) == PM_SUCCESS);
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) == PM_SUCCESS);
    CHECK(count == 1U && physical != NULL);
    CHECK(pmEnumerateDeviceExtensionProperties(
              physical, NULL, &extension_count, NULL) == PM_SUCCESS);
    CHECK(extension_count == 1U);
    CHECK(pmEnumerateDeviceExtensionProperties(
              physical, NULL, &extension_count, &extension) == PM_SUCCESS);
    CHECK(strcmp(extension.extensionName,
                 PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME) == 0);
    CHECK(extension.specificationVersion ==
          PM_EXT_SIMULATION_PROFILE_SPEC_VERSION);
    CHECK(pmEnumerateDeviceExtensionProperties(
              physical, "PM_LAYER_unknown", &extension_count, NULL) ==
          PM_ERROR_LAYER_NOT_PRESENT);

    info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.enabledExtensionCount = 1U;
    info.ppEnabledExtensionNames = &unknown;
    CHECK(pmCreateDevice(physical, &info, &device) ==
          PM_ERROR_EXTENSION_NOT_PRESENT);
    CHECK(device == NULL);
    pmDestroyInstance(instance);
    return 0;
}

int main(int argc, char** argv) {
    const char* mode = argc == 2 ? argv[1] : "";
    PmInstance instance = NULL;

    if (strcmp(mode, "app-default") == 0)
        return expect_application_version(0U, PM_SUCCESS);
    if (strcmp(mode, "app-current") == 0)
        return expect_application_version(PM_API_VERSION_0_2, PM_SUCCESS);
    if (strcmp(mode, "app-old") == 0)
        return expect_application_version(
            PM_MAKE_API_VERSION(0, 1, 0), PM_ERROR_INCOMPATIBLE_VERSION);
    if (strcmp(mode, "app-future") == 0)
        return expect_application_version(
            PM_MAKE_API_VERSION(0, 3, 0), PM_ERROR_INCOMPATIBLE_VERSION);
    if (strcmp(mode, "instance-extension") == 0)
        return check_instance_extensions();
    if (strcmp(mode, "device-extension") == 0)
        return check_device_extensions();
    if (strcmp(mode, "layer") == 0) {
        CHECK(create_instance(PM_API_VERSION_0_2, 1, &instance) == PM_SUCCESS);
        CHECK(instance != NULL);
        pmDestroyInstance(instance);
        return 0;
    }
    if (strcmp(mode, "reject-driver") == 0) {
        CHECK(create_instance(PM_API_VERSION_0_2, 0, &instance) ==
              PM_ERROR_DRIVER_NOT_FOUND);
        CHECK(instance == NULL);
        return 0;
    }
    if (strcmp(mode, "reject-layer") == 0) {
        CHECK(create_instance(PM_API_VERSION_0_2, 1, &instance) ==
              PM_ERROR_LAYER_NOT_PRESENT);
        CHECK(instance == NULL);
        return 0;
    }
    fprintf(stderr, "unknown compatibility mode: %s\n", mode);
    return 2;
}
