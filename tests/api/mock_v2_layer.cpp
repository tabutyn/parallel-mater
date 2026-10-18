// SPDX-License-Identifier: MIT
// Deliberately independent of pm_layer.h: model an installed ABI v2 layer.
#include <parallel_mater/parallel_mater.h>

#if defined(_WIN32)
#define PM_MOCK_EXPORT extern "C" __declspec(dllexport)
#else
#define PM_MOCK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct V2Call {
    const char *name;
    const void *primary;
    uint32_t argument_size;
    const void *arguments;
};

PM_MOCK_EXPORT PmResult PMAPI_CALL
pm_layerNegotiateLoaderInterfaceVersion(uint32_t *version) {
    if (!version || *version < 2U) return PM_ERROR_INCOMPATIBLE_VERSION;
    *version = 2U;
    return PM_SUCCESS;
}

PM_MOCK_EXPORT PmResult PMAPI_CALL pm_layerPreCall(const V2Call *call) {
    return call && call->name && call->primary && call->argument_size &&
            call->arguments
        ? PM_SUCCESS : PM_ERROR_INVALID_ARGUMENT;
}

PM_MOCK_EXPORT void PMAPI_CALL pm_layerPostCall(const V2Call *, PmResult) {}
