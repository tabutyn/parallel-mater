/* SPDX-License-Identifier: MIT
 * Independently compiled layer ABI fixture. Keep frozen prefixes here rather
 * than including pm_layer.h so old installed layers cannot inherit new fields. */
#include <parallel_mater/parallel_mater.h>

#include <stdint.h>

#ifndef PM_TEST_LAYER_VERSION
#error PM_TEST_LAYER_VERSION must name the negotiated fixture ABI
#endif

#if defined(_WIN32)
#define PM_SDK_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define PM_SDK_EXPORT __attribute__((visibility("default")))
#else
#define PM_SDK_EXPORT
#endif

typedef struct PmSdkLayerCallV1 {
    const char* command_name;
    const void* primary_handle;
} PmSdkLayerCallV1;

typedef struct PmSdkLayerCallV2 {
    const char* command_name;
    const void* primary_handle;
    uint32_t argument_size;
    const void* arguments;
} PmSdkLayerCallV2;

typedef struct PmSdkLayerCallV3 {
    const char* command_name;
    const void* primary_handle;
    uint32_t argument_size;
    const void* arguments;
    PmValidationObjectType primary_object_type;
    const void* primary_object;
    PmValidationDiagnosticFn diagnostic;
    void* diagnostic_user_data;
} PmSdkLayerCallV3;

PM_SDK_EXPORT PmResult PMAPI_CALL
pm_layerNegotiateLoaderInterfaceVersion(uint32_t* version) {
    if (version == NULL || *version == 0U)
        return PM_ERROR_INCOMPATIBLE_VERSION;
    /* Versions 0 and 999 deliberately return success for negative fixtures. */
    *version = PM_TEST_LAYER_VERSION;
    return PM_SUCCESS;
}

PM_SDK_EXPORT PmResult PMAPI_CALL pm_layerPreCall(const void* opaque) {
    const PmSdkLayerCallV1* v1 = (const PmSdkLayerCallV1*)opaque;
    if (v1 == NULL || v1->command_name == NULL || v1->primary_handle == NULL)
        return PM_ERROR_INVALID_ARGUMENT;
#if PM_TEST_LAYER_VERSION >= 2
    {
        const PmSdkLayerCallV2* v2 = (const PmSdkLayerCallV2*)opaque;
        if (v2->argument_size == 0U || v2->arguments == NULL)
            return PM_ERROR_INVALID_ARGUMENT;
    }
#endif
#if PM_TEST_LAYER_VERSION >= 3 && PM_TEST_LAYER_VERSION < 999
    {
        const PmSdkLayerCallV3* v3 = (const PmSdkLayerCallV3*)opaque;
        if (v3->primary_object_type == PM_VALIDATION_OBJECT_UNKNOWN ||
            v3->primary_object == NULL)
            return PM_ERROR_INVALID_ARGUMENT;
    }
#endif
    return PM_SUCCESS;
}

PM_SDK_EXPORT void PMAPI_CALL
pm_layerPostCall(const void* opaque, PmResult result) {
    (void)opaque;
    (void)result;
}
