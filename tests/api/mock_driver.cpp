// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_driver.h>
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
#include <parallel_mater/interop_vulkan.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#if defined(_WIN32)
#define PM_MOCK_EXPORT extern "C" __declspec(dllexport)
#else
#define PM_MOCK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct MockInstance { std::uint32_t tag{0x504d4452U}; };
struct MockPhysical { MockInstance *instance{}; };
struct MockDevice { bool simulation_profile_extension{}; };
struct MockSimulation { bool realtime_single_step{}; };
struct MockSemaphore { std::uint64_t value{}; };

struct StructureHeader {
    PmStructureType sType;
    const void *pNext;
};

bool realtime_timestep(float value) {
    constexpr float expected = 1.0F / 60.0F;
    const float lower = std::nextafter(expected, 0.0F);
    const float upper = std::nextafter(
        expected, std::numeric_limits<float>::infinity());
    return std::isfinite(value) && value >= lower && value <= upper;
}

PmResult mock_create_instance(const PmInstanceCreateInfo *info,
                              const PmAllocationCallbacks *,
                              PmInstance *output) {
    if (info == nullptr || info->pNext != nullptr || output == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    *output = reinterpret_cast<PmInstance>(new (std::nothrow) MockInstance);
    return *output == nullptr ? PM_ERROR_OUT_OF_HOST_MEMORY : PM_SUCCESS;
}

void mock_destroy_instance(PmInstance instance) {
    delete reinterpret_cast<MockInstance *>(instance);
}

PmResult mock_enumerate_physical_devices(PmInstance instance,
                                         std::uint32_t *count,
                                         PmPhysicalDevice *devices) {
    if (instance == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (devices == nullptr) {
        *count = 1U;
        return PM_SUCCESS;
    }
    if (*count == 0U) return PM_INCOMPLETE;
    // The mock physical handle is the mock instance.  It remains valid for the
    // instance lifetime and no device can be created from it.
    devices[0] = reinterpret_cast<PmPhysicalDevice>(instance);
    *count = 1U;
    return PM_SUCCESS;
}

void mock_get_properties(PmPhysicalDevice physical,
                         PmPhysicalDeviceProperties *properties) {
    if (properties == nullptr) return;
    *properties = {};
    if (physical == nullptr) return;
    properties->apiVersion = PM_API_VERSION_0_2;
    properties->vendorId = 0x504dU;
    properties->deviceId = 1U;
    std::strncpy(properties->deviceName, "Parallel Mater mock vendor device",
                 sizeof properties->deviceName - 1U);
}

PmResult mock_enumerate_device_extensions(
    PmPhysicalDevice physical, const char *layer_name, std::uint32_t *count,
    PmExtensionProperties *properties) {
    if (physical == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (layer_name != nullptr) return PM_ERROR_LAYER_NOT_PRESENT;
    if (properties == nullptr) {
        *count = 1U;
        return PM_SUCCESS;
    }
    if (*count == 0U) return PM_INCOMPLETE;
    properties[0] = {};
    std::strncpy(properties[0].extensionName,
                 PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME,
                 sizeof properties[0].extensionName - 1U);
    properties[0].specificationVersion =
        PM_EXT_SIMULATION_PROFILE_SPEC_VERSION;
    *count = 1U;
    return PM_SUCCESS;
}

PmResult mock_enumerate_execution_profiles(
    PmPhysicalDevice physical, std::uint32_t *count,
    PmExecutionProfileProperties *properties) {
    if (physical == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (properties == nullptr) {
        *count = 1U;
        return PM_SUCCESS;
    }
    if (*count == 0U) return PM_INCOMPLETE;
    properties[0] = {};
    std::strncpy(properties[0].profileName,
                 PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP,
                 sizeof properties[0].profileName - 1U);
    properties[0].versionMajor = 1U;
    properties[0].versionMinor = 0U;
    *count = 1U;
    return PM_SUCCESS;
}

constexpr PmInteractionCapability mock_interaction{{
    PM_INTERACTION_FAMILY_FLUID,
    PM_INTERACTION_FAMILY_RIGID_BODY,
    PM_INTERACTION_SCOPE_DISTINCT_FAMILY,
    PM_INTERACTION_MODE_CONTACT,
    PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
    PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE}, 64U};

bool same_key(const PmInteractionCapabilityKey &left,
              const PmInteractionCapabilityKey &right) {
    return left.firstFamily == right.firstFamily &&
           left.secondFamily == right.secondFamily &&
           left.scope == right.scope && left.mode == right.mode &&
           left.direction == right.direction &&
           left.materialProfile == right.materialProfile;
}

bool valid_key(const PmInteractionCapabilityKey &key) {
    const auto first = static_cast<int>(key.firstFamily);
    const auto second = static_cast<int>(key.secondFamily);
    if (first < PM_INTERACTION_FAMILY_FLUID ||
        first > PM_INTERACTION_FAMILY_RIGID_BODY ||
        second < PM_INTERACTION_FAMILY_FLUID ||
        second > PM_INTERACTION_FAMILY_RIGID_BODY ||
        key.scope < PM_INTERACTION_SCOPE_SELF ||
        key.scope > PM_INTERACTION_SCOPE_DISTINCT_FAMILY ||
        key.mode < PM_INTERACTION_MODE_CONTACT ||
        key.mode > PM_INTERACTION_MODE_INTERFACE ||
        key.direction < PM_INTERACTION_DIRECTION_BIDIRECTIONAL ||
        key.direction > PM_INTERACTION_DIRECTION_EXTERNAL_TO_SECOND ||
        key.materialProfile < PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE ||
        key.materialProfile > PM_INTERACTION_MATERIAL_THERMAL_PHASE)
        return false;
    return key.scope == PM_INTERACTION_SCOPE_DISTINCT_FAMILY
        ? first < second : first == second;
}

PmResult mock_enumerate_interactions(PmPhysicalDevice physical,
                                     std::uint32_t *count,
                                     PmInteractionCapability *capabilities) {
    if (physical == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (capabilities == nullptr) {
        *count = 1U;
        return PM_SUCCESS;
    }
    if (*count == 0U) return PM_INCOMPLETE;
    capabilities[0] = mock_interaction;
    *count = 1U;
    return PM_SUCCESS;
}

PmResult mock_get_interaction(PmPhysicalDevice physical,
                              const PmInteractionCapabilityKey *key,
                              PmInteractionCapability *capability) {
    if (key == nullptr || capability == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    const PmInteractionCapabilityKey requested = *key;
    *capability = {};
    if (physical == nullptr || !valid_key(requested))
        return PM_ERROR_INVALID_ARGUMENT;
    if (!same_key(requested, mock_interaction.key))
        return PM_ERROR_FEATURE_NOT_PRESENT;
    *capability = mock_interaction;
    return PM_SUCCESS;
}

// Deliberately permissive device functions: the validation tests must prove
// rejection by the layer, rather than accidentally relying on the driver.
PmResult mock_create_device(PmPhysicalDevice, const PmDeviceCreateInfo *info,
                            PmDevice *out) {
    if (out == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    bool simulation_profile_extension = false;
    if (info != nullptr) {
        if (info->enabledExtensionCount != 0U &&
            info->ppEnabledExtensionNames == nullptr)
            return PM_ERROR_INVALID_ARGUMENT;
        for (std::uint32_t index = 0U;
             index < info->enabledExtensionCount; ++index) {
            const char *name = info->ppEnabledExtensionNames[index];
            if (name == nullptr) return PM_ERROR_INVALID_ARGUMENT;
            if (std::strcmp(name,
                            PM_EXT_SIMULATION_PROFILE_EXTENSION_NAME) == 0) {
                simulation_profile_extension = true;
                continue;
            }
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
            if (std::strcmp(name, "PM_EXT_vulkan_interop") == 0) continue;
#endif
            return PM_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    *out = reinterpret_cast<PmDevice>(
        new (std::nothrow) MockDevice{simulation_profile_extension});
    return *out ? PM_SUCCESS : PM_ERROR_OUT_OF_HOST_MEMORY;
}
void mock_destroy_device(PmDevice device) {
    delete reinterpret_cast<MockDevice *>(device);
}
void mock_get_queue(PmDevice device, std::uint32_t, std::uint32_t, PmQueue *out) {
    *out = reinterpret_cast<PmQueue>(device);
}
PmResult mock_create_command_pool(PmDevice, const PmCommandPoolCreateInfo *,
                                  PmCommandPool *out) {
    *out = reinterpret_cast<PmCommandPool>(new char);
    return PM_SUCCESS;
}
void mock_destroy_command_pool(PmDevice, PmCommandPool pool) {
    delete reinterpret_cast<char *>(pool);
}
PmResult mock_reset_command_pool(PmDevice, PmCommandPool) { return PM_SUCCESS; }
PmResult mock_allocate_commands(PmDevice, PmCommandPool, std::uint32_t count,
                                PmCommandBuffer *out) {
    for (std::uint32_t index = 0U; index < count; ++index)
        out[index] = reinterpret_cast<PmCommandBuffer>(new char);
    return PM_SUCCESS;
}
void mock_free_commands(PmDevice, PmCommandPool, std::uint32_t count,
                        const PmCommandBuffer *buffers) {
    for (std::uint32_t index = 0U; index < count; ++index)
        delete reinterpret_cast<char *>(buffers[index]);
}
PmResult mock_begin_commands(PmCommandBuffer,
                             const PmCommandBufferBeginInfo *) {
    return PM_SUCCESS;
}
PmResult mock_end_commands(PmCommandBuffer) { return PM_SUCCESS; }
PmResult mock_reset_commands(PmCommandBuffer) { return PM_SUCCESS; }
PmResult mock_cmd_copy_buffer(PmCommandBuffer, PmBuffer, PmBuffer,
                              std::uint32_t, const PmBufferCopy *) {
    return PM_SUCCESS;
}
PmResult mock_queue_submit(PmQueue, const PmSubmitInfo *info, PmFence) {
    if (std::getenv("PM_TEST_MOCK_SUBMIT_FAILURE"))
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    for (std::uint32_t index = 0U; index < info->signalCount; ++index) {
        auto *timeline = reinterpret_cast<MockSemaphore *>(
            info->pSignals[index].semaphore);
        timeline->value = std::max(timeline->value,
                                   info->pSignals[index].value);
    }
    return PM_SUCCESS;
}
PmResult mock_completion_result() {
    return std::getenv("PM_TEST_MOCK_DEFERRED_FAILURE")
        ? PM_ERROR_DEVICE_LOST : PM_SUCCESS;
}
PmResult mock_queue_wait_idle(PmQueue) { return mock_completion_result(); }
PmResult mock_device_wait_idle(PmDevice) { return mock_completion_result(); }
PmResult mock_create_fence(PmDevice, PmFlags, PmFence *out) {
    *out = reinterpret_cast<PmFence>(new char);
    return PM_SUCCESS;
}
void mock_destroy_fence(PmDevice, PmFence fence) {
    delete reinterpret_cast<char *>(fence);
}
PmResult mock_get_fence_status(PmDevice, PmFence) {
    return mock_completion_result();
}
PmResult mock_reset_fences(PmDevice, std::uint32_t, const PmFence *) {
    return PM_SUCCESS;
}
PmResult mock_wait_for_fences(PmDevice, std::uint32_t, const PmFence *,
                              PmBool32, std::uint64_t) {
    return mock_completion_result();
}
PmResult mock_create_semaphore(PmDevice, const PmSemaphoreCreateInfo *info,
                               PmSemaphore *out) {
    *out = reinterpret_cast<PmSemaphore>(
        new (std::nothrow) MockSemaphore{info->initialValue});
    return *out ? PM_SUCCESS : PM_ERROR_OUT_OF_HOST_MEMORY;
}
void mock_destroy_semaphore(PmDevice, PmSemaphore semaphore) {
    delete reinterpret_cast<MockSemaphore *>(semaphore);
}
PmResult mock_get_semaphore_value(PmDevice, PmSemaphore semaphore,
                                  std::uint64_t *value) {
    const auto result = mock_completion_result();
    if (result == PM_SUCCESS)
        *value = reinterpret_cast<MockSemaphore *>(semaphore)->value;
    return result;
}
PmResult mock_wait_semaphore(PmDevice, PmSemaphore, std::uint64_t,
                             std::uint64_t) {
    return mock_completion_result();
}
PmResult mock_get_memory_budget(PmDevice, PmMemoryBudget *budget) {
    if (budget == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *budget = {};
    const char *mode = std::getenv("PM_TEST_MOCK_MEMORY_BUDGET");
    if (mode != nullptr && std::strcmp(mode, "failure") == 0)
        return PM_ERROR_DEVICE_LOST;

    PmDeviceSize total = 0U;
    PmDeviceSize available = 0U;
    PmMemoryBudgetConfidence confidence =
        PM_MEMORY_BUDGET_CONFIDENCE_UNKNOWN;
    if (mode != nullptr && std::strcmp(mode, "estimated") == 0) {
        total = 1024U;
        available = 768U;
        confidence = PM_MEMORY_BUDGET_CONFIDENCE_IMPLEMENTATION_ESTIMATE;
    } else if (mode != nullptr && std::strcmp(mode, "overflow") == 0) {
        total = 1U;
        available = 2U;
        confidence = PM_MEMORY_BUDGET_CONFIDENCE_IMPLEMENTATION_ESTIMATE;
    }
    if (confidence == PM_MEMORY_BUDGET_CONFIDENCE_UNKNOWN) {
        total = 0U;
        available = 0U;
    } else if (available > total) {
        return PM_ERROR_INVALID_STATE;
    }

    budget->heapCount = 1U;
    budget->budget[0] = total;
    budget->usage[0] = total - available;
    budget->confidence[0] = confidence;
    if (confidence != PM_MEMORY_BUDGET_CONFIDENCE_UNKNOWN) {
        budget->applicationAllocationBytes = 64U;
        budget->libraryDeviceBytes = 32U;
        budget->libraryHostBytes = 16U;
        budget->borrowedBufferBytes = 128U;
    }
    return PM_SUCCESS;
}
PmResult mock_import_native_device(PmInstance, const PmNativeDeviceImportInfo *,
                                   PmDevice *out) {
    return mock_create_device(nullptr, nullptr, out);
}
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
PmResult mock_import_vulkan_device(PmInstance,
                                   const PmVulkanDeviceImportInfo *,
                                   PmDevice *out) {
    return mock_create_device(nullptr, nullptr, out);
}
#endif
PmResult mock_allocate(PmDevice, const PmMemoryAllocateInfo *info, PmDeviceMemory *out) {
    *out = nullptr;
    if (info->allocationSize == 13) return PM_ERROR_OUT_OF_DEVICE_MEMORY;
    *out = reinterpret_cast<PmDeviceMemory>(new char);
    return PM_SUCCESS;
}
void mock_free(PmDevice, PmDeviceMemory memory) { delete reinterpret_cast<char *>(memory); }
PmResult mock_create_buffer(PmDevice, const PmBufferCreateInfo *, PmBuffer *out) {
    *out = reinterpret_cast<PmBuffer>(new char);
    return PM_SUCCESS;
}
void mock_destroy_buffer(PmDevice, PmBuffer buffer) { delete reinterpret_cast<char *>(buffer); }
void mock_requirements(PmDevice, PmBuffer, PmMemoryRequirements *out) {
    *out = {256, 256, 2, PM_FALSE};
}
PmResult mock_bind(PmDevice, PmBuffer, PmDeviceMemory, PmDeviceSize) { return PM_SUCCESS; }
PmResult mock_map(PmDevice, PmDeviceMemory, PmDeviceSize, PmDeviceSize, void **out) {
    static char storage;
    *out = &storage;
    return PM_SUCCESS;
}
void mock_unmap(PmDevice, PmDeviceMemory) {}
PmResult mock_ranges(PmDevice, uint32_t, const PmMappedMemoryRange *) { return PM_SUCCESS; }
PmResult mock_create_geometry(PmDevice, const PmGeometryCreateInfo *,
                              PmGeometry *out) {
    *out = reinterpret_cast<PmGeometry>(new char);
    return PM_SUCCESS;
}
void mock_destroy_geometry(PmDevice, PmGeometry geometry) {
    delete reinterpret_cast<char *>(geometry);
}
void mock_geometry_requirements(PmDevice, PmGeometry,
                                PmMemoryRequirements *out) {
    *out = {256U, 256U, 2U, PM_FALSE};
}
PmResult mock_bind_geometry(PmDevice, PmGeometry, PmDeviceMemory,
                            PmDeviceSize) {
    return PM_SUCCESS;
}
PmResult mock_cmd_build_geometry(PmCommandBuffer, PmGeometry) {
    return PM_SUCCESS;
}
PmResult mock_create_simulation(PmDevice device, const PmSimulationCreateInfo *info,
                                PmSimulation *out) {
    if (device == nullptr || info == nullptr || out == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    bool realtime = false;
    if (info->pNext != nullptr) {
        const auto *header = static_cast<const StructureHeader *>(info->pNext);
        if (header->sType !=
                PM_STRUCTURE_TYPE_SIMULATION_PROFILE_CREATE_INFO ||
            header->pNext != nullptr)
            return PM_ERROR_INVALID_ARGUMENT;
        const auto *mock_device = reinterpret_cast<MockDevice *>(device);
        if (!mock_device->simulation_profile_extension)
            return PM_ERROR_EXTENSION_NOT_PRESENT;
        const auto *request = static_cast<
            const PmSimulationProfileCreateInfo *>(info->pNext);
        if (request->pProfileName == nullptr ||
            request->pProfileName[0] == '\0')
            return PM_ERROR_INVALID_ARGUMENT;
        if (std::strcmp(request->pProfileName,
                        PM_EXECUTION_PROFILE_REALTIME_SINGLE_STEP) != 0)
            return PM_ERROR_FEATURE_NOT_PRESENT;
        if (request->versionMajor != 1U || request->versionMinor != 0U)
            return PM_ERROR_INCOMPATIBLE_VERSION;
        realtime = true;
    }
    *out = reinterpret_cast<PmSimulation>(
        new (std::nothrow) MockSimulation{realtime});
    return *out ? PM_SUCCESS : PM_ERROR_OUT_OF_HOST_MEMORY;
}
void mock_destroy_simulation(PmDevice, PmSimulation simulation) {
    delete reinterpret_cast<MockSimulation *>(simulation);
}
void mock_simulation_requirements(PmDevice, PmSimulation,
                                  PmSimulationMemoryRequirements *out) {
    out->persistent = {256U, 256U, 2U, PM_FALSE};
    out->scratch = {256U, 256U, 2U, PM_FALSE};
}
PmResult mock_bind_simulation(PmDevice, PmSimulation,
                              const PmSimulationMemoryBinding *) {
    return PM_SUCCESS;
}
PmResult mock_cmd_write_rigid_bodies(PmCommandBuffer, PmSimulation,
                                     std::uint32_t, std::uint32_t,
                                     const PmRigidBodyInfo *) {
    return PM_SUCCESS;
}
PmResult mock_cmd_reset_simulation(PmCommandBuffer, PmSimulation) {
    return PM_SUCCESS;
}
PmResult mock_create_pipeline(PmDevice, const PmPhysicsPipelineCreateInfo *,
                              PmPhysicsPipeline *out) {
    *out = reinterpret_cast<PmPhysicsPipeline>(new char);
    return PM_SUCCESS;
}
void mock_destroy_pipeline(PmDevice, PmPhysicsPipeline pipeline) {
    delete reinterpret_cast<char *>(pipeline);
}
PmResult mock_cmd_bind_pipeline(PmCommandBuffer, PmPhysicsPipeline) {
    return PM_SUCCESS;
}
PmResult mock_cmd_write_cloths(PmCommandBuffer, PmSimulation,
                               std::uint32_t, std::uint32_t,
                               const PmClothInfo *) {
    return PM_SUCCESS;
}
PmResult mock_cmd_step(PmCommandBuffer, PmSimulation simulation,
                       const PmStepInfo *step) {
    if (simulation == nullptr || step == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    const auto *mock = reinterpret_cast<MockSimulation *>(simulation);
    if (mock->realtime_single_step &&
        (!realtime_timestep(step->timestep) || step->substeps != 1U))
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}
PmResult mock_create_query_pool(PmDevice, const PmQueryPoolCreateInfo *info,
                                PmQueryPool *out) {
    *out = nullptr;
    if (info->queryCount == 13U) return PM_ERROR_OUT_OF_HOST_MEMORY;
    *out = reinterpret_cast<PmQueryPool>(new char);
    return PM_SUCCESS;
}
void mock_destroy_query_pool(PmDevice, PmQueryPool pool) {
    delete reinterpret_cast<char *>(pool);
}
PmResult mock_cmd_reset_query_pool(PmCommandBuffer, PmQueryPool,
                                   std::uint32_t, std::uint32_t) {
    return PM_SUCCESS;
}
PmResult mock_cmd_write_timestamp(PmCommandBuffer, PmStageFlags,
                                  PmQueryPool, std::uint32_t) {
    return PM_SUCCESS;
}
PmResult mock_cmd_begin_statistics(PmCommandBuffer, PmQueryPool,
                                   std::uint32_t) {
    return PM_SUCCESS;
}
PmResult mock_cmd_end_statistics(PmCommandBuffer, PmQueryPool,
                                 std::uint32_t) {
    return PM_SUCCESS;
}
PmResult mock_get_simulation_status(PmDevice, PmSimulation,
                                    PmResult *out) {
    *out = PM_SUCCESS;
    return PM_SUCCESS;
}
PmResult mock_get_query_results(PmDevice, PmQueryPool, std::uint32_t,
                                std::uint32_t, std::size_t size, void *data,
                                PmDeviceSize, PmFlags) {
    if (data && size) std::memset(data, 0, size);
    return PM_SUCCESS;
}

PFN_pmVoidFunction lookup_instance(const char *name) {
    if (name == nullptr) return nullptr;
    if (std::strcmp(name, "pmCreateInstance") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_create_instance);
    if (std::strcmp(name, "pmDestroyInstance") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_destroy_instance);
    if (std::strcmp(name, "pmEnumeratePhysicalDevices") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(
            &mock_enumerate_physical_devices);
    if (std::strcmp(name, "pmImportNativeDevice") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_import_native_device);
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
    if (std::strcmp(name, "pmImportVulkanDeviceEXT") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_import_vulkan_device);
#endif
    return nullptr;
}

} // namespace

PM_MOCK_EXPORT PmResult PMAPI_CALL
pm_driverNegotiateLoaderInterfaceVersion(std::uint32_t *version) {
    if (version == nullptr || *version == 0U)
        return PM_ERROR_INVALID_ARGUMENT;
#if defined(PM_MOCK_INTERFACE_VERSION)
    *version = PM_MOCK_INTERFACE_VERSION;
#else
    *version = std::min(*version, PM_DRIVER_INTERFACE_VERSION);
#endif
    return PM_SUCCESS;
}

PM_MOCK_EXPORT PFN_pmVoidFunction PMAPI_CALL pm_driverGetInstanceProcAddr(
    PmInstance, const char *name) {
    return lookup_instance(name);
}

PM_MOCK_EXPORT PFN_pmVoidFunction PMAPI_CALL pm_driverGetDeviceProcAddr(
    PmDevice, const char *name) {
    if (!name) return nullptr;
#define MOCK(command, implementation) \
    if (std::strcmp(name, #command) == 0) \
        return reinterpret_cast<PFN_pmVoidFunction>(&implementation)
    MOCK(pmDestroyDevice, mock_destroy_device);
    MOCK(pmGetDeviceQueue, mock_get_queue);
    MOCK(pmCreateCommandPool, mock_create_command_pool);
    MOCK(pmDestroyCommandPool, mock_destroy_command_pool);
    MOCK(pmResetCommandPool, mock_reset_command_pool);
    MOCK(pmAllocateCommandBuffers, mock_allocate_commands);
    MOCK(pmFreeCommandBuffers, mock_free_commands);
    MOCK(pmBeginCommandBuffer, mock_begin_commands);
    MOCK(pmEndCommandBuffer, mock_end_commands);
    MOCK(pmResetCommandBuffer, mock_reset_commands);
    MOCK(pmCmdCopyBuffer, mock_cmd_copy_buffer);
    MOCK(pmQueueSubmit, mock_queue_submit);
    MOCK(pmQueueWaitIdle, mock_queue_wait_idle);
    MOCK(pmDeviceWaitIdle, mock_device_wait_idle);
    MOCK(pmCreateFence, mock_create_fence);
    MOCK(pmDestroyFence, mock_destroy_fence);
    MOCK(pmGetFenceStatus, mock_get_fence_status);
    MOCK(pmResetFences, mock_reset_fences);
    MOCK(pmWaitForFences, mock_wait_for_fences);
    MOCK(pmCreateSemaphore, mock_create_semaphore);
    MOCK(pmDestroySemaphore, mock_destroy_semaphore);
    MOCK(pmGetSemaphoreCounterValue, mock_get_semaphore_value);
    MOCK(pmWaitSemaphore, mock_wait_semaphore);
    MOCK(pmGetDeviceMemoryBudget, mock_get_memory_budget);
    MOCK(pmAllocateMemory, mock_allocate);
    MOCK(pmFreeMemory, mock_free);
    MOCK(pmCreateBuffer, mock_create_buffer);
    MOCK(pmDestroyBuffer, mock_destroy_buffer);
    MOCK(pmGetBufferMemoryRequirements, mock_requirements);
    MOCK(pmBindBufferMemory, mock_bind);
    MOCK(pmMapMemory, mock_map);
    MOCK(pmUnmapMemory, mock_unmap);
    MOCK(pmFlushMappedMemoryRanges, mock_ranges);
    MOCK(pmInvalidateMappedMemoryRanges, mock_ranges);
    MOCK(pmCreateGeometry, mock_create_geometry);
    MOCK(pmDestroyGeometry, mock_destroy_geometry);
    MOCK(pmGetGeometryMemoryRequirements, mock_geometry_requirements);
    MOCK(pmBindGeometryMemory, mock_bind_geometry);
    MOCK(pmCmdBuildGeometry, mock_cmd_build_geometry);
    MOCK(pmCreateSimulation, mock_create_simulation);
    MOCK(pmDestroySimulation, mock_destroy_simulation);
    MOCK(pmGetSimulationMemoryRequirements, mock_simulation_requirements);
    MOCK(pmBindSimulationMemory, mock_bind_simulation);
    MOCK(pmCmdWriteRigidBodies, mock_cmd_write_rigid_bodies);
    MOCK(pmCmdResetSimulation, mock_cmd_reset_simulation);
    MOCK(pmCreatePhysicsPipeline, mock_create_pipeline);
    MOCK(pmDestroyPhysicsPipeline, mock_destroy_pipeline);
    MOCK(pmCmdBindPhysicsPipeline, mock_cmd_bind_pipeline);
    MOCK(pmCmdWriteCloths, mock_cmd_write_cloths);
    MOCK(pmCmdStepSimulation, mock_cmd_step);
    MOCK(pmCreateQueryPool, mock_create_query_pool);
    MOCK(pmDestroyQueryPool, mock_destroy_query_pool);
    MOCK(pmCmdResetQueryPool, mock_cmd_reset_query_pool);
    MOCK(pmCmdWriteTimestamp, mock_cmd_write_timestamp);
    MOCK(pmCmdBeginStatisticsQuery, mock_cmd_begin_statistics);
    MOCK(pmCmdEndStatisticsQuery, mock_cmd_end_statistics);
    MOCK(pmGetSimulationStatus, mock_get_simulation_status);
    MOCK(pmGetQueryPoolResults, mock_get_query_results);
#undef MOCK
    return nullptr;
}

PM_MOCK_EXPORT PFN_pmVoidFunction PMAPI_CALL
pm_driverGetPhysicalDeviceProcAddr(PmInstance, const char *name) {
    if (name && std::strcmp(name, "pmCreateDevice") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_create_device);
    if (name && std::strcmp(
            name, "pmGetPhysicalDeviceProperties") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_get_properties);
    if (name && std::strcmp(
            name, "pmEnumerateDeviceExtensionProperties") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(
            &mock_enumerate_device_extensions);
    if (name && std::strcmp(
            name,
            "pmEnumerateExecutionProfiles") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(
            &mock_enumerate_execution_profiles);
    if (name && std::strcmp(
            name, "pmEnumeratePhysicalDeviceInteractionCapabilities") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_enumerate_interactions);
    if (name && std::strcmp(
            name, "pmGetPhysicalDeviceInteractionCapability") == 0)
        return reinterpret_cast<PFN_pmVoidFunction>(&mock_get_interaction);
    return nullptr;
}
