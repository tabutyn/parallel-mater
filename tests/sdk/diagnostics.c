/* SPDX-License-Identifier: MIT */
#include <parallel_mater/parallel_mater.h>
#include <parallel_mater/pm_layer_args.h>

#include <stdint.h>
#include <string.h>

typedef struct DiagnosticState {
    uint32_t count;
    int matched;
    PmDevice expectedDevice;
    PmSemaphore expectedSemaphore;
} DiagnosticState;

static void PMAPI_CALL receive_diagnostic(
    void* user, const PmValidationDiagnostic* diagnostic) {
    DiagnosticState* state = (DiagnosticState*)user;
    const PmLayerArgs_pmGetSemaphoreCounterValue* arguments;
    ++state->count;
    if (diagnostic == NULL || diagnostic->pRuleId == NULL ||
        diagnostic->pCommandName == NULL || diagnostic->pArguments == NULL ||
        strcmp(diagnostic->pRuleId, PM_VALIDATION_RULE_INVALID_ARGUMENT) != 0 ||
        strcmp(diagnostic->pCommandName, "pmGetSemaphoreCounterValue") != 0 ||
        diagnostic->severity != PM_VALIDATION_SEVERITY_ERROR ||
        diagnostic->result != PM_ERROR_INVALID_ARGUMENT ||
        diagnostic->objectType != PM_VALIDATION_OBJECT_DEVICE ||
        diagnostic->object != state->expectedDevice ||
        diagnostic->argumentSize !=
            sizeof(PmLayerArgs_pmGetSemaphoreCounterValue))
        return;
    arguments = (const PmLayerArgs_pmGetSemaphoreCounterValue*)
        diagnostic->pArguments;
    state->matched = arguments->device == state->expectedDevice &&
        arguments->semaphore == state->expectedSemaphore;
}

int main(void) {
    DiagnosticState state = {0};
    PmValidationDiagnosticsCreateInfo diagnostics = {0};
    PmInstanceCreateInfo instanceInfo = {0};
    PmInstance instance = NULL;
    PmPhysicalDevice physical = NULL;
    PmDeviceCreateInfo deviceInfo = {0};
    PmDevice device = NULL;
    PmDevice other = NULL;
    PmSemaphoreCreateInfo semaphoreInfo = {0};
    PmSemaphore semaphore = NULL;
    uint64_t value = 0;
    uint32_t count = 1;
    PmResult result;

    diagnostics.sType = PM_STRUCTURE_TYPE_VALIDATION_DIAGNOSTICS_CREATE_INFO;
    diagnostics.pfnDiagnostic = receive_diagnostic;
    diagnostics.pUserData = &state;
    instanceInfo.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pNext = &diagnostics;
    instanceInfo.apiVersion = PM_API_VERSION_0_2;
    result = pmCreateInstance(&instanceInfo, NULL, &instance);
    if (result != PM_SUCCESS) return 1;
    result = pmEnumeratePhysicalDevices(instance, &count, &physical);
    if (result != PM_SUCCESS || count != 1 || physical == NULL) return 1;

    deviceInfo.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    if (pmCreateDevice(physical, &deviceInfo, &device) != PM_SUCCESS ||
        pmCreateDevice(physical, &deviceInfo, &other) != PM_SUCCESS)
        return 1;
    semaphoreInfo.sType = PM_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (pmCreateSemaphore(device, &semaphoreInfo, &semaphore) != PM_SUCCESS)
        return 1;

    state.expectedDevice = other;
    state.expectedSemaphore = semaphore;
    result = pmGetSemaphoreCounterValue(other, semaphore, &value);

    pmDestroySemaphore(device, semaphore);
    pmDestroyDevice(other);
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    return result == PM_ERROR_INVALID_ARGUMENT && state.count == 1 &&
        state.matched ? 0 : 1;
}
