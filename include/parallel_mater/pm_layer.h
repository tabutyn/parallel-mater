/* SPDX-License-Identifier: MIT
 * Private loader-to-layer ABI. Applications include parallel_mater.h. */
#ifndef PARALLEL_MATER_LAYER_H
#define PARALLEL_MATER_LAYER_H

#include "parallel_mater.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PM_LAYER_INTERFACE_VERSION 3u

typedef struct PmLayerCallInfo {
    const char* pCommandName;
    const void* primaryHandle;
    /* Version 2: application-visible arguments, valid only during the callback.
     * Cast pArguments to PmLayerArgs_pmCommand from pm_layer_args.h after
     * checking argumentSize. The version 1 prefix remains unchanged. */
    uint32_t argumentSize;
    const void* pArguments;
    /* Version 3: application-visible dispatch context and the immutable
     * per-instance diagnostic sink. These pointers are callback-scoped. */
    PmValidationObjectType primaryObjectType;
    const void* primaryObject;
    PmValidationDiagnosticFn pfnDiagnostic;
    void* pDiagnosticUserData;
} PmLayerCallInfo;

typedef PmResult (PMAPI_CALL *PFN_pmLayerNegotiateLoaderInterfaceVersion)(
    uint32_t* pVersion);
typedef PmResult (PMAPI_CALL *PFN_pmLayerPreCall)(
    const PmLayerCallInfo* pCall);
typedef void (PMAPI_CALL *PFN_pmLayerPostCall)(
    const PmLayerCallInfo* pCall, PmResult result);

#ifdef __cplusplus
}
#endif
#endif
