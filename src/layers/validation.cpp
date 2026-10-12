// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_layer.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#define PM_LAYER_EXPORT extern "C" __declspec(dllexport)
#else
#define PM_LAYER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

PM_LAYER_EXPORT PmResult PMAPI_CALL
pm_layerNegotiateLoaderInterfaceVersion(std::uint32_t *version) {
    if (version == nullptr || *version == 0U)
        return PM_ERROR_INVALID_ARGUMENT;
    *version = std::min(*version, PM_LAYER_INTERFACE_VERSION);
    return PM_SUCCESS;
}

PM_LAYER_EXPORT PmResult PMAPI_CALL pm_layerPreCall(
    const PmLayerCallInfo *call) {
    if (call == nullptr || call->pCommandName == nullptr ||
        call->pCommandName[0] == '\0' || call->primaryHandle == nullptr)
        return PM_ERROR_INVALID_ARGUMENT;
    return PM_SUCCESS;
}

PM_LAYER_EXPORT void PMAPI_CALL pm_layerPostCall(
    const PmLayerCallInfo *, PmResult) {}
