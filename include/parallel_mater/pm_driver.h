/* SPDX-License-Identifier: MIT
 * Private loader-to-driver ABI. Applications include parallel_mater.h. */
#ifndef PARALLEL_MATER_DRIVER_H
#define PARALLEL_MATER_DRIVER_H

#include "parallel_mater.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PM_DRIVER_API_ATTR
#if defined(_WIN32) && defined(PARALLEL_MATER_DRIVER_IMPLEMENTATION)
#define PM_DRIVER_API_ATTR __declspec(dllexport)
#elif defined(PARALLEL_MATER_DRIVER_IMPLEMENTATION) && defined(__GNUC__)
#define PM_DRIVER_API_ATTR __attribute__((visibility("default")))
#else
#define PM_DRIVER_API_ATTR
#endif
#endif

#define PM_DRIVER_INTERFACE_VERSION 2u
#define PM_DRIVER_MANIFEST_FILE_VERSION "1.0.0"

typedef PmResult (PMAPI_CALL *PFN_pmDriverNegotiateLoaderInterfaceVersion)(
    uint32_t* pVersion);
typedef PFN_pmVoidFunction (PMAPI_CALL *PFN_pmDriverGetInstanceProcAddr)(
    PmInstance instance, const char* pName);
typedef PFN_pmVoidFunction (PMAPI_CALL *PFN_pmDriverGetDeviceProcAddr)(
    PmDevice device, const char* pName);
typedef PFN_pmVoidFunction (PMAPI_CALL *PFN_pmDriverGetPhysicalDeviceProcAddr)(
    PmInstance instance, const char* pName);

PM_DRIVER_API_ATTR PmResult PMAPI_CALL pm_driverNegotiateLoaderInterfaceVersion(
    uint32_t* pVersion);
PM_DRIVER_API_ATTR PFN_pmVoidFunction PMAPI_CALL pm_driverGetInstanceProcAddr(
    PmInstance instance, const char* pName);
PM_DRIVER_API_ATTR PFN_pmVoidFunction PMAPI_CALL pm_driverGetDeviceProcAddr(
    PmDevice device, const char* pName);
PM_DRIVER_API_ATTR PFN_pmVoidFunction PMAPI_CALL pm_driverGetPhysicalDeviceProcAddr(
    PmInstance instance, const char* pName);

#ifdef __cplusplus
}
#endif
#endif
