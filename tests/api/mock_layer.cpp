// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_layer.h>
#if defined(_WIN32)
#define PM_MOCK_EXPORT extern "C" __declspec(dllexport)
#else
#define PM_MOCK_EXPORT extern "C" __attribute__((visibility("default")))
#endif
PM_MOCK_EXPORT PmResult PMAPI_CALL
pm_layerNegotiateLoaderInterfaceVersion(uint32_t *version) {
    *version = PM_MOCK_INTERFACE_VERSION;
    return PM_SUCCESS;
}
PM_MOCK_EXPORT PmResult PMAPI_CALL pm_layerPreCall(const PmLayerCallInfo *) {
    return PM_SUCCESS;
}
PM_MOCK_EXPORT void PMAPI_CALL pm_layerPostCall(const PmLayerCallInfo *, PmResult) {}
