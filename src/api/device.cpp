// SPDX-License-Identifier: MIT
#include "internal.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {

[[nodiscard]] bool subset(PmPhysicalDeviceFeatures requested,
                          PmPhysicalDeviceFeatures supported) noexcept {
    const auto *left = reinterpret_cast<const PmBool32 *>(&requested);
    const auto *right = reinterpret_cast<const PmBool32 *>(&supported);
    constexpr std::size_t count = sizeof(PmPhysicalDeviceFeatures) / sizeof(PmBool32);
    static_assert(sizeof(PmPhysicalDeviceFeatures) % sizeof(PmBool32) == 0U);
    for (std::size_t index = 0U; index < count; ++index)
        if (left[index] != PM_FALSE && right[index] == PM_FALSE) return false;
    return true;
}

} // namespace

extern "C" PMAPI_ATTR PmResult PMAPI_CALL pmCreateDevice(
    PmPhysicalDevice physical, const PmDeviceCreateInfo *info, PmDevice *device) {
    if (device == nullptr) return PM_ERROR_INVALID_ARGUMENT;
    *device = nullptr;
    if (physical == nullptr || info == nullptr ||
        !parallel_mater::api::valid_structure(
            info->sType, PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO, info->pNext) ||
        info->queueCreateInfoCount != 1U || info->pQueueInfos == nullptr ||
        info->pQueueInfos[0].queueFamilyIndex != 0U ||
        info->pQueueInfos[0].queueCount != 1U ||
        !subset(info->enabledFeatures, parallel_mater::api::supported_features()))
        return PM_ERROR_INVALID_ARGUMENT;
    PmResult result = PM_SUCCESS;
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
    auto services = parallel_mater::device::create_cpu(
        physical->description.ordinal, result);
#else
    auto services = parallel_mater::device::create_cuda(
        physical->description.ordinal, result);
#endif
    if (!services) return result;
    auto *created = parallel_mater::api::make_object<PmDevice_T>(nullptr);
    if (created == nullptr) return PM_ERROR_OUT_OF_HOST_MEMORY;
    created->physical = physical;
    created->enabled = info->enabledFeatures;
    created->services = std::move(services);
    created->queue.device = created;
    created->queue.family = 0U;
    created->queue.index = 0U;
    created->library_host_bytes.store(sizeof(PmDevice_T), std::memory_order_relaxed);
    *device = created;
    return PM_SUCCESS;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmDestroyDevice(PmDevice device) {
    if (device == nullptr) return;
    (void)device->services->wait_idle();
    delete device;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetDeviceQueue(
    PmDevice device, std::uint32_t family, std::uint32_t index, PmQueue *queue) {
    if (queue == nullptr) return;
    *queue = device != nullptr && family == 0U && index == 0U
        ? &device->queue : nullptr;
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetPhysicalDeviceMemoryProperties(
    PmPhysicalDevice physical, PmMemoryProperties *properties) {
    if (properties == nullptr) return;
    *properties = {};
    if (physical == nullptr) return;
    properties->heapCount = 1U;
    properties->heaps[0] = {physical->description.total_memory,
                            PM_MEMORY_DEVICE_LOCAL_BIT};
    properties->typeCount = 2U;
    properties->types[0] = {PM_MEMORY_DEVICE_LOCAL_BIT, 0U};
    properties->types[1] = {PM_MEMORY_DEVICE_LOCAL_BIT |
                            PM_MEMORY_HOST_VISIBLE_BIT |
                            PM_MEMORY_HOST_COHERENT_BIT, 0U};
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetPhysicalDeviceLimits(
    PmPhysicalDevice physical, PmPhysicalDeviceLimits *limits) {
    if (limits == nullptr) return;
    *limits = {};
    if (physical == nullptr) return;
    limits->maxAllocationSize = physical->description.total_memory;
    limits->maxBufferSize = physical->description.total_memory;
    limits->nonCoherentAtomSize = 1U;
    limits->maxMemoryAllocations = 1U << 20U;
    limits->maxRigidBodies = 1U << 20U;
    limits->maxRigidConstraints = 1U << 20U;
    limits->maxContacts = 1U << 24U;
    limits->maxFluidSystems = 1024U;
    limits->maxParticlesPerFluid = 1U << 24U;
    limits->maxNeighborsPerParticle = 1024U;
    limits->maxRigidIterations = 64U;
    limits->maxLiquidIterations = 64U;
    limits->maxSubsteps = 64U;
    limits->maxQueriesPerPool = 65536U;
    limits->maxTimelineValueDifference = std::numeric_limits<std::uint64_t>::max();
}

extern "C" PMAPI_ATTR void PMAPI_CALL pmGetPhysicalDeviceFormatProperties(
    PmPhysicalDevice physical, PmFormat format, PmFormatProperties *properties) {
    if (properties == nullptr) return;
    *properties = {};
    if (physical == nullptr) return;
    switch (format) {
    case PM_FORMAT_R32_UINT:
    case PM_FORMAT_R32_SFLOAT:
        properties->elementSize = properties->elementAlignment = 4U;
        properties->supportedUsages = PM_FORMAT_GEOMETRY_INDEX_BIT |
            PM_FORMAT_PARTICLE_INPUT_BIT | PM_FORMAT_SIMULATION_OUTPUT_BIT;
        break;
    case PM_FORMAT_R32G32B32_SFLOAT:
        properties->elementSize = 12U;
        properties->elementAlignment = 4U;
        properties->supportedUsages = PM_FORMAT_GEOMETRY_VERTEX_BIT |
            PM_FORMAT_PARTICLE_INPUT_BIT | PM_FORMAT_SIMULATION_OUTPUT_BIT;
        break;
    case PM_FORMAT_R32G32B32A32_SFLOAT:
        properties->elementSize = properties->elementAlignment = 16U;
        properties->supportedUsages = PM_FORMAT_PARTICLE_INPUT_BIT |
            PM_FORMAT_SIMULATION_OUTPUT_BIT;
        break;
    case PM_FORMAT_PARTICLE_STATE:
        properties->elementSize = sizeof(PmParticleState);
        properties->elementAlignment = alignof(PmParticleState);
        properties->supportedUsages = PM_FORMAT_PARTICLE_INPUT_BIT |
            PM_FORMAT_SIMULATION_OUTPUT_BIT;
        break;
    case PM_FORMAT_RIGID_STATE:
        properties->elementSize = sizeof(PmRigidState);
        properties->elementAlignment = alignof(PmRigidState);
        properties->supportedUsages = PM_FORMAT_SIMULATION_OUTPUT_BIT;
        break;
    default: break;
    }
}
