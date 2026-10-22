/* SPDX-License-Identifier: MIT
 * Frozen pre-profile API 0.2 declarations. This source deliberately does not
 * include the installed header, proving that an independently built old client
 * retains the legacy default against the new loader and driver. */
#include <stdint.h>
#include <stdio.h>

#if defined(_WIN32)
#define OLD_PMAPI __declspec(dllimport)
#else
#define OLD_PMAPI
#endif

typedef int32_t OldPmResult;
typedef uint32_t OldPmBool32;
typedef struct OldPmInstance_T* OldPmInstance;
typedef struct OldPmPhysicalDevice_T* OldPmPhysicalDevice;
typedef struct OldPmDevice_T* OldPmDevice;
typedef struct OldPmSimulation_T* OldPmSimulation;

enum {
    OLD_PM_SUCCESS = 0,
    OLD_PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    OLD_PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 2,
    OLD_PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO = 6,
    OLD_PM_API_VERSION_0_2 = (2U << 12U)
};

typedef struct OldPmInstanceCreateInfo {
    int32_t sType;
    const void* pNext;
    uint32_t apiVersion;
    const char* pApplicationName;
    OldPmBool32 enableValidation;
    uint32_t enabledLayerCount;
    const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} OldPmInstanceCreateInfo;

typedef struct OldPmDeviceCreateInfo {
    int32_t sType;
    const void* pNext;
    uint32_t queueCreateInfoCount;
    const void* pQueueInfos;
    OldPmBool32 enabledFeatures[19];
    uint32_t enabledExtensionCount;
    const char* const* ppEnabledExtensionNames;
} OldPmDeviceCreateInfo;

typedef struct OldPmSimulationCreateInfo {
    int32_t sType;
    const void* pNext;
    uint32_t queueFamilyIndex;
    uint32_t rigidBodyCapacity;
    uint32_t constraintCapacity;
    uint32_t contactCapacity;
    uint32_t fluidCount;
    const void* pFluidCapacities;
    uint32_t clothCount;
    const void* pClothCapacities;
    uint32_t ropeCount;
    const void* pRopeCapacities;
    uint32_t softBodyCount;
    const void* pSoftBodyCapacities;
    uint32_t smokeCount;
    const void* pSmokeCapacities;
    uint32_t couplingCapacity;
    uint32_t emitterCapacity;
    uint32_t destroyerCapacity;
    uint32_t paintFieldCount;
    const void* pPaintFieldCapacities;
    uint32_t paintRuleCapacity;
    uint32_t debugFrameCapacity;
} OldPmSimulationCreateInfo;

OLD_PMAPI OldPmResult pmCreateInstance(
    const OldPmInstanceCreateInfo*, const void*, OldPmInstance*);
OLD_PMAPI void pmDestroyInstance(OldPmInstance);
OLD_PMAPI OldPmResult pmEnumeratePhysicalDevices(
    OldPmInstance, uint32_t*, OldPmPhysicalDevice*);
OLD_PMAPI OldPmResult pmCreateDevice(
    OldPmPhysicalDevice, const OldPmDeviceCreateInfo*, OldPmDevice*);
OLD_PMAPI void pmDestroyDevice(OldPmDevice);
OLD_PMAPI OldPmResult pmCreateSimulation(
    OldPmDevice, const OldPmSimulationCreateInfo*, OldPmSimulation*);
OLD_PMAPI void pmDestroySimulation(OldPmDevice, OldPmSimulation);

#define CHECK(expression) do {                                              \
    if (!(expression)) {                                                    \
        fprintf(stderr, "line %d: %s\n", __LINE__, #expression);           \
        return 1;                                                           \
    }                                                                       \
} while (0)

int main(void) {
    OldPmInstanceCreateInfo instance_info = {0};
    OldPmInstance instance = 0;
    OldPmPhysicalDevice physical = 0;
    OldPmDeviceCreateInfo device_info = {0};
    OldPmDevice device = 0;
    OldPmSimulationCreateInfo simulation_info = {0};
    OldPmSimulation simulation = 0;
    uint32_t count = 1U;
    instance_info.sType = OLD_PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = OLD_PM_API_VERSION_0_2;
    CHECK(pmCreateInstance(&instance_info, 0, &instance) == OLD_PM_SUCCESS);
    CHECK(pmEnumeratePhysicalDevices(instance, &count, &physical) ==
          OLD_PM_SUCCESS);
    CHECK(count == 1U && physical != 0);
    device_info.sType = OLD_PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    CHECK(pmCreateDevice(physical, &device_info, &device) == OLD_PM_SUCCESS);
    simulation_info.sType = OLD_PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = 1U;
    CHECK(pmCreateSimulation(device, &simulation_info, &simulation) ==
          OLD_PM_SUCCESS);
    pmDestroySimulation(device, simulation);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    puts("PASS: independently built API 0.2 client retains legacy default");
    return 0;
}
