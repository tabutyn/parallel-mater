// SPDX-License-Identifier: MIT
// Deliberately independent of pm_layer.h: model an already installed v1 layer.
#include <parallel_mater/parallel_mater.h>
#if defined(_WIN32)
#define PM_MOCK_EXPORT extern "C" __declspec(dllexport)
#else
#define PM_MOCK_EXPORT extern "C" __attribute__((visibility("default")))
#endif
struct LegacyCall { const char *name; const void *primary; };
PM_MOCK_EXPORT PmResult PMAPI_CALL pm_layerNegotiateLoaderInterfaceVersion(uint32_t *version) {
    if (!version || !*version) return PM_ERROR_INCOMPATIBLE_VERSION;
    *version = 1;
    return PM_SUCCESS;
}
PM_MOCK_EXPORT PmResult PMAPI_CALL pm_layerPreCall(const LegacyCall *call) {
    return call && call->name && call->primary ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}
PM_MOCK_EXPORT void PMAPI_CALL pm_layerPostCall(const LegacyCall *, PmResult) {}
