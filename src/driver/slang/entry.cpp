// SPDX-License-Identifier: MIT
#include "driver_remap.h"
#include <parallel_mater/parallel_mater.h>
#if defined(PARALLEL_MATER_HAS_VULKAN_INTEROP)
#include <parallel_mater/interop_vulkan.h>
#endif
#include "driver_unmap.h"
#include <parallel_mater/pm_driver.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#define PM_DRIVER_EXPORT extern "C" PM_DRIVER_API_ATTR

namespace {

PFN_pmVoidFunction lookup(const char *name) noexcept {
    if (name == nullptr) return nullptr;
#define PM_DRIVER_LOOKUP(public_name, implementation_name)                    \
    if (std::strcmp(name, #public_name) == 0)                                 \
        return reinterpret_cast<PFN_pmVoidFunction>(&implementation_name)
#include "driver_lookup.inc"
#undef PM_DRIVER_LOOKUP
    return nullptr;
}

} // namespace

PM_DRIVER_EXPORT PmResult PMAPI_CALL
pm_driverNegotiateLoaderInterfaceVersion(std::uint32_t *version) {
    if (version == nullptr || *version == 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    *version = std::min(*version, PM_DRIVER_INTERFACE_VERSION);
    return PM_SUCCESS;
}

PM_DRIVER_EXPORT PFN_pmVoidFunction PMAPI_CALL pm_driverGetInstanceProcAddr(
    PmInstance instance, const char *name) {
    if (instance == nullptr && name != nullptr &&
        std::strcmp(name, "pmGetInstanceProcAddr") != 0 &&
        std::strcmp(name, "pmEnumerateInstanceVersion") != 0 &&
        std::strcmp(name, "pmCreateInstance") != 0)
        return nullptr;
    return lookup(name);
}

PM_DRIVER_EXPORT PFN_pmVoidFunction PMAPI_CALL pm_driverGetDeviceProcAddr(
    PmDevice device, const char *name) {
    return device == nullptr ? nullptr : lookup(name);
}

PM_DRIVER_EXPORT PFN_pmVoidFunction PMAPI_CALL
pm_driverGetPhysicalDeviceProcAddr(PmInstance instance, const char *name) {
    return instance == nullptr ? nullptr : lookup(name);
}
