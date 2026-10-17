// SPDX-License-Identifier: MIT
#include <parallel_mater/pm_layer.h>
#include <parallel_mater/pm_layer_args.h>

#include <atomic>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
#define PM_MOCK_EXPORT extern "C" __declspec(dllexport)
#else
#define PM_MOCK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
std::atomic<std::uint64_t> pre_mask{};
std::atomic<std::uint64_t> post_mask{};
std::atomic<std::uint32_t> errors{};

PmResult observe(const PmLayerCallInfo *call,
                 std::atomic<std::uint64_t> &mask) {
    if (call == nullptr || call->pCommandName == nullptr ||
        call->primaryHandle == nullptr || call->pArguments == nullptr) {
        errors.fetch_add(1U);
        return PM_ERROR_INVALID_ARGUMENT;
    }
#define PM_MANUAL_LAYER_INTERCEPTED(command, index, reason)                  \
    if (std::strcmp(call->pCommandName, #command) == 0) {                   \
        if (call->argumentSize != sizeof(PmLayerArgs_##command)) {          \
            errors.fetch_add(1U);                                           \
            return PM_ERROR_INVALID_ARGUMENT;                               \
        }                                                                   \
        mask.fetch_or(std::uint64_t{1} << index);                           \
        return PM_SUCCESS;                                                   \
    }
#define PM_MANUAL_LAYER_EXCLUDED(command, reason)                            \
    if (std::strcmp(call->pCommandName, #command) == 0) {                   \
        errors.fetch_add(1U);                                                \
        return PM_ERROR_INVALID_ARGUMENT;                                   \
    }
#include "manual_layer_inventory.inc"
#undef PM_MANUAL_LAYER_EXCLUDED
#undef PM_MANUAL_LAYER_INTERCEPTED
    return PM_SUCCESS;
}

std::uint64_t expected_mask() {
    std::uint64_t result{};
#define PM_MANUAL_LAYER_INTERCEPTED(command, index, reason) \
    result |= std::uint64_t{1} << index;
#define PM_MANUAL_LAYER_EXCLUDED(command, reason)
#include "manual_layer_inventory.inc"
#undef PM_MANUAL_LAYER_EXCLUDED
#undef PM_MANUAL_LAYER_INTERCEPTED
    return result;
}
} // namespace

PM_MOCK_EXPORT PmResult PMAPI_CALL
pm_layerNegotiateLoaderInterfaceVersion(std::uint32_t *version) {
    if (version == nullptr || *version < PM_LAYER_INTERFACE_VERSION)
        return PM_ERROR_INCOMPATIBLE_VERSION;
    *version = PM_LAYER_INTERFACE_VERSION;
    return PM_SUCCESS;
}

PM_MOCK_EXPORT PmResult PMAPI_CALL pm_layerPreCall(
    const PmLayerCallInfo *call) {
    return observe(call, pre_mask);
}

PM_MOCK_EXPORT void PMAPI_CALL pm_layerPostCall(
    const PmLayerCallInfo *call, PmResult) {
    (void)observe(call, post_mask);
}

PM_MOCK_EXPORT void PMAPI_CALL pm_testManualLayerReset() {
    pre_mask.store(0U);
    post_mask.store(0U);
    errors.store(0U);
}

PM_MOCK_EXPORT std::uint64_t PMAPI_CALL pm_testManualLayerPreMask() {
    return pre_mask.load();
}

PM_MOCK_EXPORT std::uint64_t PMAPI_CALL pm_testManualLayerPostMask() {
    return post_mask.load();
}

PM_MOCK_EXPORT std::uint64_t PMAPI_CALL pm_testManualLayerExpectedMask() {
    return expected_mask();
}

PM_MOCK_EXPORT std::uint32_t PMAPI_CALL pm_testManualLayerErrors() {
    return errors.load();
}
