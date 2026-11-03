// SPDX-License-Identifier: MIT
#include "internal.hpp"
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
#include <parallel_mater/interop_vulkan.h>
#endif

#include <algorithm>
#include <cstring>
#include <limits>

namespace parallel_mater::api {

namespace {

enum InteractionRequirement : std::uint32_t {
    require_rigidBodies = 1U << 0U,
    require_rigidConstraints = 1U << 1U,
    require_particleLiquids = 1U << 2U,
    require_liquidRigidCoupling = 1U << 3U,
    require_liquidLiquidCoupling = 1U << 4U,
    require_cloth = 1U << 5U,
    require_rope = 1U << 6U,
    require_softBodies = 1U << 7U,
    require_smoke = 1U << 8U,
    require_domainCouplings = 1U << 9U,
};

struct InteractionCandidate {
    PmInteractionCapability capability;
    std::uint32_t requirements;
};

constexpr InteractionCandidate interaction_candidates[] = {
#define PM_INTERACTION_CAPABILITY(first, second, scope, mode, direction, profile, maximum, requirements) \
    {{{first, second, scope, mode, direction, profile}, maximum}, requirements},
#include "interaction_capabilities.inc"
#if defined(PARALLEL_MATER_TEST_RESOURCE_DRIVER)
    PM_INTERACTION_CAPABILITY(
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_SCOPE_SEPARATE_SYSTEM, PM_INTERACTION_MODE_CONTACT,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_MISCIBLE,
        PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY,
        require_particleLiquids | require_liquidLiquidCoupling)
    PM_INTERACTION_CAPABILITY(
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_SCOPE_SEPARATE_SYSTEM, PM_INTERACTION_MODE_PRESSURE,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_MISCIBLE,
        PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY,
        require_particleLiquids | require_liquidLiquidCoupling)
    PM_INTERACTION_CAPABILITY(
        PM_INTERACTION_FAMILY_FLUID, PM_INTERACTION_FAMILY_FLUID,
        PM_INTERACTION_SCOPE_SEPARATE_SYSTEM, PM_INTERACTION_MODE_VISCOSITY,
        PM_INTERACTION_DIRECTION_BIDIRECTIONAL,
        PM_INTERACTION_MATERIAL_MISCIBLE,
        PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY,
        require_particleLiquids | require_liquidLiquidCoupling)
#endif
#undef PM_INTERACTION_CAPABILITY
};

bool requirements_supported(std::uint32_t requirements,
                            const PmPhysicalDeviceFeatures &features) noexcept {
    return (!(requirements & require_rigidBodies) || features.rigidBodies) &&
           (!(requirements & require_rigidConstraints) || features.rigidConstraints) &&
           (!(requirements & require_particleLiquids) || features.particleLiquids) &&
           (!(requirements & require_liquidRigidCoupling) || features.liquidRigidCoupling) &&
           (!(requirements & require_liquidLiquidCoupling) || features.liquidLiquidCoupling) &&
           (!(requirements & require_cloth) || features.cloth) &&
           (!(requirements & require_rope) || features.rope) &&
           (!(requirements & require_softBodies) || features.softBodies) &&
           (!(requirements & require_smoke) || features.smoke) &&
           (!(requirements & require_domainCouplings) || features.domainCouplings);
}

bool driver_supports_interaction(
    const PmInteractionCapabilityKey &key) noexcept {
    if (key.materialProfile == PM_INTERACTION_MATERIAL_MISCIBLE) {
#if defined(PARALLEL_MATER_TEST_RESOURCE_DRIVER) || \
    defined(PARALLEL_MATER_QUALIFIED_MISCIBLE_FLUID)
        return true;
#else
        // The registry identity exists for private lifecycle coverage. P1-09
        // will enable it only on adapters that pass physical qualification.
        return false;
#endif
    }
    return true;
}

bool interaction_supported(
    const InteractionCandidate &candidate,
    const PmPhysicalDeviceFeatures &features) noexcept {
    return requirements_supported(candidate.requirements, features) &&
           driver_supports_interaction(candidate.capability.key);
}

bool valid_interaction_key(const PmInteractionCapabilityKey &key) noexcept {
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
        key.materialProfile > PM_INTERACTION_MATERIAL_ROPE_CAPSULE)
        return false;
    return key.scope == PM_INTERACTION_SCOPE_DISTINCT_FAMILY
        ? first < second : first == second;
}

bool same_interaction_key(const PmInteractionCapabilityKey &left,
                          const PmInteractionCapabilityKey &right) noexcept {
    return left.firstFamily == right.firstFamily &&
           left.secondFamily == right.secondFamily &&
           left.scope == right.scope && left.mode == right.mode &&
           left.direction == right.direction &&
           left.materialProfile == right.materialProfile;
}

} // namespace

PmResult find_interaction_capability(
    const PmPhysicalDeviceFeatures &features,
    const PmInteractionCapabilityKey &key,
    PmInteractionCapability &capability) noexcept {
    capability = {};
    if (!valid_interaction_key(key)) return PM_ERROR_INVALID_ARGUMENT;
    for (const auto &candidate : interaction_candidates) {
        if (!same_interaction_key(key, candidate.capability.key)) continue;
        if (!interaction_supported(candidate, features))
            return PM_ERROR_FEATURE_NOT_PRESENT;
        capability = candidate.capability;
        return PM_SUCCESS;
    }
    return PM_ERROR_FEATURE_NOT_PRESENT;
}

PmPhysicalDeviceFeatures supported_features() noexcept {
    PmPhysicalDeviceFeatures features{};
    features.rigidBodies = PM_TRUE;
    features.rigidConstraints = PM_TRUE;
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER) || \
    defined(PARALLEL_MATER_D3D12_DRIVER) || defined(PARALLEL_MATER_METAL_DRIVER) || defined(PARALLEL_MATER_VULKAN_DRIVER)
#if defined(PARALLEL_MATER_SLANG_CPU_LIQUIDS) || defined(PARALLEL_MATER_VULKAN_DRIVER)
    features.particleLiquids = PM_TRUE;
    features.liquidRigidCoupling = PM_TRUE;
    features.particleEmitters = PM_TRUE;
#endif
#if defined(PARALLEL_MATER_VULKAN_DRIVER)
    features.cloth = PM_TRUE;
    features.rope = PM_TRUE;
    features.softBodies = PM_TRUE;
    features.smoke = PM_TRUE;
    features.domainCouplings = PM_TRUE;
    features.paint = PM_TRUE;
#endif
    features.contactEvents = PM_TRUE;
    features.debugData = PM_TRUE;
    features.timelineSemaphores = PM_TRUE;
    features.timestampQueries = PM_TRUE;
#if defined(PARALLEL_MATER_METAL_DRIVER)
    features.particleLiquids = PM_TRUE;
    features.liquidRigidCoupling = PM_TRUE;
    features.particleEmitters = PM_TRUE;
    features.cloth = PM_TRUE;
    features.softBodies = PM_TRUE;
    features.domainCouplings = PM_TRUE;
    features.rope = PM_TRUE;
    features.smoke = PM_TRUE;
    features.paint = PM_TRUE;
#endif
#if defined(PARALLEL_MATER_TEST_RESOURCE_DRIVER)
    // Private host fixture: exercise resource lifetime without claiming
    // production Slang CPU simulation support for these domains.
    features.particleLiquids = PM_TRUE;
    features.liquidRigidCoupling = PM_TRUE;
    features.liquidLiquidCoupling = PM_TRUE;
    features.particleEmitters = PM_TRUE;
    features.cloth = PM_TRUE;
    features.softBodies = PM_TRUE;
    features.paint = PM_TRUE;
#endif
#if defined(PARALLEL_MATER_QUALIFIED_EQUAL_FLUID)
    features.liquidLiquidCoupling = PM_TRUE;
#endif
    features.statisticsQueries = PM_TRUE;
    features.memoryBudget = PM_TRUE;
#if defined(PARALLEL_MATER_D3D12_DRIVER)
    features.particleLiquids = PM_TRUE;
    features.liquidRigidCoupling = PM_TRUE;
    features.cloth = PM_TRUE;
    features.rope = PM_TRUE;
    features.softBodies = PM_TRUE;
    features.smoke = PM_TRUE;
    features.domainCouplings = PM_TRUE;
    features.particleEmitters = PM_TRUE;
    features.paint = PM_TRUE;
#endif
#else
    features.particleLiquids = PM_TRUE;
    features.liquidRigidCoupling = PM_TRUE;
#if defined(PARALLEL_MATER_QUALIFIED_EQUAL_FLUID)
    features.liquidLiquidCoupling = PM_TRUE;
#else
    features.liquidLiquidCoupling = PM_FALSE;
#endif
    features.cloth = PM_TRUE;
    features.rope = PM_TRUE;
    features.softBodies = PM_TRUE;
    features.smoke = PM_TRUE;
    features.domainCouplings = PM_TRUE;
    features.particleEmitters = PM_TRUE;
    features.paint = PM_TRUE;
    features.contactEvents = PM_TRUE;
    features.debugData = PM_TRUE;
    features.timelineSemaphores = PM_TRUE;
    features.timestampQueries = PM_TRUE;
    features.statisticsQueries = PM_TRUE;
    features.memoryBudget = PM_TRUE;
#endif
    return features;
}

#define PM_LOOKUP(name) \
    if (std::strcmp(value, #name) == 0) \
        return reinterpret_cast<PFN_pmVoidFunction>(name)

PFN_pmVoidFunction lookup_proc(const char *value) noexcept {
    if (value == nullptr || *value == '\0') return nullptr;
    PM_LOOKUP(pmGetInstanceProcAddr);
    PM_LOOKUP(pmGetDeviceProcAddr);
    PM_LOOKUP(pmEnumerateInstanceVersion);
    PM_LOOKUP(pmCreateInstance);
    PM_LOOKUP(pmDestroyInstance);
    PM_LOOKUP(pmEnumeratePhysicalDevices);
    PM_LOOKUP(pmGetPhysicalDeviceProperties);
    PM_LOOKUP(pmGetPhysicalDeviceQueueFamilies);
    PM_LOOKUP(pmGetPhysicalDeviceFeatures);
    PM_LOOKUP(pmEnumeratePhysicalDeviceInteractionCapabilities);
    PM_LOOKUP(pmGetPhysicalDeviceInteractionCapability);
    PM_LOOKUP(pmCreateDevice);
    PM_LOOKUP(pmDestroyDevice);
    PM_LOOKUP(pmGetDeviceQueue);
    PM_LOOKUP(pmGetPhysicalDeviceMemoryProperties);
    PM_LOOKUP(pmAllocateMemory);
    PM_LOOKUP(pmFreeMemory);
    PM_LOOKUP(pmMapMemory);
    PM_LOOKUP(pmUnmapMemory);
    PM_LOOKUP(pmFlushMappedMemoryRanges);
    PM_LOOKUP(pmInvalidateMappedMemoryRanges);
    PM_LOOKUP(pmCreateBuffer);
    PM_LOOKUP(pmDestroyBuffer);
    PM_LOOKUP(pmGetBufferMemoryRequirements);
    PM_LOOKUP(pmBindBufferMemory);
    PM_LOOKUP(pmCreateGeometry);
    PM_LOOKUP(pmDestroyGeometry);
    PM_LOOKUP(pmGetGeometryMemoryRequirements);
    PM_LOOKUP(pmBindGeometryMemory);
    PM_LOOKUP(pmCmdBuildGeometry);
    PM_LOOKUP(pmCreateSimulation);
    PM_LOOKUP(pmDestroySimulation);
    PM_LOOKUP(pmGetSimulationMemoryRequirements);
    PM_LOOKUP(pmBindSimulationMemory);
    PM_LOOKUP(pmCmdWriteRigidBodies);
    PM_LOOKUP(pmCmdApplyRigidInputs);
    PM_LOOKUP(pmCmdSetRigidStates);
    PM_LOOKUP(pmCmdSetKinematicTargets);
    PM_LOOKUP(pmCmdWriteRigidConstraints);
    PM_LOOKUP(pmCreatePhysicsPipeline);
    PM_LOOKUP(pmDestroyPhysicsPipeline);
    PM_LOOKUP(pmCreateCommandPool);
    PM_LOOKUP(pmDestroyCommandPool);
    PM_LOOKUP(pmResetCommandPool);
    PM_LOOKUP(pmAllocateCommandBuffers);
    PM_LOOKUP(pmFreeCommandBuffers);
    PM_LOOKUP(pmBeginCommandBuffer);
    PM_LOOKUP(pmEndCommandBuffer);
    PM_LOOKUP(pmResetCommandBuffer);
    PM_LOOKUP(pmCmdResetSimulation);
    PM_LOOKUP(pmCmdSetFluidParticles);
    PM_LOOKUP(pmCmdSetCouplings);
    PM_LOOKUP(pmCmdSetCouplings2);
    PM_LOOKUP(pmCmdWriteCloths);
    PM_LOOKUP(pmCmdWriteCloths2);
    PM_LOOKUP(pmCmdWriteRopes);
    PM_LOOKUP(pmCmdWriteSoftBodies);
    PM_LOOKUP(pmCmdWriteSoftBodies2);
    PM_LOOKUP(pmCmdWriteSmoke);
    PM_LOOKUP(pmCmdWriteParticleEmitters);
    PM_LOOKUP(pmCmdWriteParticleDestroyers);
    PM_LOOKUP(pmCmdWritePaintFields);
    PM_LOOKUP(pmCmdWritePaintRules);
    PM_LOOKUP(pmCmdClearPaintFields);
    PM_LOOKUP(pmCmdBindPhysicsPipeline);
    PM_LOOKUP(pmCmdStepSimulation);
    PM_LOOKUP(pmCmdCopyBuffer);
    PM_LOOKUP(pmCmdCopyRigidStates);
    PM_LOOKUP(pmCmdCopyFluidParticles);
    PM_LOOKUP(pmCmdCopyClothState);
    PM_LOOKUP(pmCmdCopyClothSurface);
    PM_LOOKUP(pmCmdCopyRopeState);
    PM_LOOKUP(pmCmdCopySoftBodyState);
    PM_LOOKUP(pmCmdCopySoftBodySurface);
    PM_LOOKUP(pmCmdCopySmokeState);
    PM_LOOKUP(pmCmdCopyPaintField);
    PM_LOOKUP(pmCmdCopyConstraintStates);
    PM_LOOKUP(pmCmdQueryHitBox);
    PM_LOOKUP(pmCmdCopyContactEvents);
    PM_LOOKUP(pmCmdCopyDebugData);
    PM_LOOKUP(pmCmdPipelineBarrier);
    PM_LOOKUP(pmCreateSemaphore);
    PM_LOOKUP(pmDestroySemaphore);
    PM_LOOKUP(pmGetSemaphoreCounterValue);
    PM_LOOKUP(pmWaitSemaphore);
    PM_LOOKUP(pmQueueSubmit);
    PM_LOOKUP(pmQueueWaitIdle);
    PM_LOOKUP(pmDeviceWaitIdle);
    PM_LOOKUP(pmCreateFence);
    PM_LOOKUP(pmDestroyFence);
    PM_LOOKUP(pmGetFenceStatus);
    PM_LOOKUP(pmResetFences);
    PM_LOOKUP(pmWaitForFences);
    PM_LOOKUP(pmGetPhysicalDeviceInteropProperties);
    PM_LOOKUP(pmImportNativeDevice);
    PM_LOOKUP(pmGetNativeDevice);
    PM_LOOKUP(pmGetNativeQueue);
    PM_LOOKUP(pmImportNativeBuffer);
    PM_LOOKUP(pmGetNativeBuffer);
    PM_LOOKUP(pmImportNativeSemaphore);
    PM_LOOKUP(pmGetNativeSemaphore);
    PM_LOOKUP(pmCreateQueryPool);
    PM_LOOKUP(pmDestroyQueryPool);
    PM_LOOKUP(pmCmdResetQueryPool);
    PM_LOOKUP(pmCmdWriteTimestamp);
    PM_LOOKUP(pmCmdBeginStatisticsQuery);
    PM_LOOKUP(pmCmdEndStatisticsQuery);
    PM_LOOKUP(pmGetSimulationStatus);
    PM_LOOKUP(pmGetQueryPoolResults);
    PM_LOOKUP(pmGetDeviceMemoryBudget);
    PM_LOOKUP(pmGetPhysicalDeviceLimits);
    PM_LOOKUP(pmGetPhysicalDeviceFormatProperties);
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
    PM_LOOKUP(pmImportVulkanDeviceEXT);
    PM_LOOKUP(pmGetVulkanDeviceEXT);
    PM_LOOKUP(pmImportVulkanBufferEXT);
    PM_LOOKUP(pmGetVulkanBufferEXT);
    PM_LOOKUP(pmImportVulkanSemaphoreEXT);
    PM_LOOKUP(pmGetVulkanSemaphoreEXT);
#endif
    return nullptr;
}

#undef PM_LOOKUP

} // namespace parallel_mater::api

extern "C" PMAPI_ATTR PFN_pmVoidFunction PMAPI_CALL pmGetInstanceProcAddr(
    PmInstance instance, const char *name) {
    if (name == nullptr) return nullptr;
    if (instance == nullptr) {
        if (std::strcmp(name, "pmGetInstanceProcAddr") != 0 &&
            std::strcmp(name, "pmEnumerateInstanceVersion") != 0 &&
            std::strcmp(name, "pmCreateInstance") != 0)
            return nullptr;
    }
    return parallel_mater::api::lookup_proc(name);
}

extern "C" PMAPI_ATTR PFN_pmVoidFunction PMAPI_CALL pmGetDeviceProcAddr(
    PmDevice device, const char *name) {
    if (device == nullptr || name == nullptr) return nullptr;
    return parallel_mater::api::lookup_proc(name);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmEnumerateInstanceVersion(
    std::uint32_t *version) {
    if (version == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *version = PM_API_VERSION_0_3;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL
pmEnumerateDeviceExtensionProperties(PmPhysicalDevice physical,
                                     const char *layer_name,
                                     std::uint32_t *count,
                                     PmExtensionProperties *properties) {
    if (physical == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    if (layer_name != nullptr) return PM_ERROR_LAYER_NOT_PRESENT;
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
    constexpr std::uint32_t available = 1U;
#else
    constexpr std::uint32_t available = 0U;
#endif
    if (properties == nullptr) {
        *count = available;
        return PM_SUCCESS;
    }
    const std::uint32_t capacity = *count;
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
    if (capacity > 0U) {
        properties[0] = {};
        std::strncpy(properties[0].extensionName, "PM_EXT_vulkan_interop",
                     sizeof properties[0].extensionName - 1U);
        properties[0].specificationVersion = 1U;
    }
#endif
    *count = std::min(capacity, available);
    return capacity >= available ? PM_SUCCESS : PM_INCOMPLETE;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateInstance(
    const PmInstanceCreateInfo *info, const PmAllocationCallbacks *allocator,
    PmInstance *instance) {
    if (instance == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *instance = nullptr;
    if (info == nullptr || !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, info->pNext))
        return PM_ERROR_INVALID_ARGUMENT;
    if (info->apiVersion != 0U && info->apiVersion != PM_API_VERSION_0_3)
        return PM_ERROR_INCOMPATIBLE_VERSION;
    if (allocator != nullptr &&
        (allocator->pfnAllocate == nullptr || allocator->pfnFree == nullptr))
        return PM_ERROR_INVALID_ARGUMENT;
    auto *created = parallel_mater::api::make_object<PmInstance_T>(nullptr);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    if (allocator != nullptr) {
        created->allocator = *allocator;
        created->has_allocator = true;
    }
    created->validation = info->enableValidation == PM_TRUE;
    std::uint32_t count = 0U;
    PmResult result = parallel_mater::device::enumerate(&count, nullptr);
    if (result != PM_SUCCESS) {
        delete created;
        return result;
    }
    try {
        std::vector<parallel_mater::device::Description> descriptions(count);
        if (count != 0U) {
            std::uint32_t requested = count;
            result = parallel_mater::device::enumerate(&requested, descriptions.data());
            if (result != PM_SUCCESS || requested != count) {
                delete created;
                return result == PM_SUCCESS ? PM_ERROR_INVALID_STATE : result;
            }
        }
        created->physical_devices.reserve(count);
        for (auto &description : descriptions) {
            auto physical = std::make_unique<PmPhysicalDevice_T>();
            physical->instance = created;
            physical->description = std::move(description);
            created->physical_devices.push_back(std::move(physical));
        }
    } catch (const std::bad_alloc &) {
        delete created;
        return PM_ERROR_OUT_OF_HOST_MEMORY;
    }
    *instance = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyInstance(PmInstance instance) {
    delete instance;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmEnumeratePhysicalDevices(
    PmInstance instance, std::uint32_t *count, PmPhysicalDevice *devices) {
    if (instance == nullptr || count == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    const std::uint32_t available = static_cast<std::uint32_t>(
        instance->physical_devices.size());
    if (devices == nullptr) {
        *count = available;
        return PM_SUCCESS;
    }
    const std::uint32_t written = std::min(*count, available);
    for (std::uint32_t index = 0U; index < written; ++index)
        devices[index] = instance->physical_devices[index].get();
    *count = written;
    return written == available ? PM_SUCCESS : PM_INCOMPLETE;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetPhysicalDeviceProperties(
    PmPhysicalDevice physical, PmPhysicalDeviceProperties *properties) {
    if (properties == nullptr) return;
    *properties = {};
    if (physical == nullptr) return;
    properties->apiVersion = PM_API_VERSION_0_3;
    properties->driverVersion = physical->description.driver_version;
    properties->vendorId = physical->description.vendor_id;
    properties->deviceId = physical->description.device_id;
    std::memcpy(properties->deviceUuid, physical->description.uuid,
                sizeof properties->deviceUuid);
    std::strncpy(properties->deviceName, physical->description.name.c_str(),
                 sizeof properties->deviceName - 1U);
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmGetPhysicalDeviceQueueFamilies(
    PmPhysicalDevice physical, std::uint32_t *count,
    PmQueueFamilyProperties *properties) {
    if (physical == nullptr || count == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    if (properties == nullptr) {
        *count = 1U;
        return PM_SUCCESS;
    }
    if (*count == 0U) return PM_INCOMPLETE;
    properties[0] = {PM_QUEUE_PHYSICS_BIT | PM_QUEUE_TRANSFER_BIT, 1U, 64U, 1.0F};
    const bool complete = *count >= 1U;
    *count = 1U;
    return complete ? PM_SUCCESS : PM_INCOMPLETE;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetPhysicalDeviceFeatures(
    PmPhysicalDevice physical, PmPhysicalDeviceFeatures *features) {
    if (features == nullptr) return;
    *features = physical == nullptr ? PmPhysicalDeviceFeatures{}
                                    : parallel_mater::api::supported_features();
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL
pmEnumeratePhysicalDeviceInteractionCapabilities(
    PmPhysicalDevice physical, std::uint32_t *count,
    PmInteractionCapability *capabilities) {
    if (physical == nullptr || count == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    const auto features = parallel_mater::api::supported_features();
    std::uint32_t available = 0U;
    for (const auto &candidate : parallel_mater::api::interaction_candidates)
        if (parallel_mater::api::interaction_supported(candidate, features))
            ++available;
    if (capabilities == nullptr) {
        *count = available;
        return PM_SUCCESS;
    }
    const std::uint32_t capacity = *count;
    std::uint32_t written = 0U;
    for (const auto &candidate : parallel_mater::api::interaction_candidates) {
        if (!parallel_mater::api::interaction_supported(candidate, features))
            continue;
        if (written == capacity) break;
        capabilities[written++] = candidate.capability;
    }
    *count = written;
    return written == available ? PM_SUCCESS : PM_INCOMPLETE;
}

extern "C" PMAPI_ATTR PmResult PMAPI_CALL
pmGetPhysicalDeviceInteractionCapability(
    PmPhysicalDevice physical, const PmInteractionCapabilityKey *key,
    PmInteractionCapability *capability) {
    if (key == nullptr || capability == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    const PmInteractionCapabilityKey requested = *key;
    *capability = {};
    if (physical == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    return parallel_mater::api::find_interaction_capability(
        parallel_mater::api::supported_features(), requested, *capability);
}
