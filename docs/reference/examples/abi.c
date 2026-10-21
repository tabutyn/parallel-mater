/* Compile-only checks for the portable buffer layouts. */
#include "../api/parallel_mater.h"
#ifdef __cplusplus
#define CHECK(condition) static_assert(condition, #condition)
#else
#define CHECK(condition) _Static_assert(condition, #condition)
#endif

CHECK(sizeof(float) == 4);
CHECK(sizeof(PmBool32) == 4);
CHECK(sizeof(PmDeviceSize) == 8);
CHECK(sizeof(PmRigidState) == 52);
CHECK(offsetof(PmRigidState, position) == 0);
CHECK(offsetof(PmRigidState, orientation) == 12);
CHECK(offsetof(PmRigidState, linearVelocity) == 28);
CHECK(offsetof(PmRigidState, angularVelocity) == 40);
CHECK(sizeof(PmParticleState) == 28);
CHECK(offsetof(PmParticleState, velocity) == 12);
CHECK(offsetof(PmParticleState, temperatureCelsius) == 24);
CHECK(sizeof(PmStepStatistics) == 88);
CHECK(offsetof(PmStepStatistics, executionResult) == 80);
CHECK(offsetof(PmStepStatistics, earlyExitCount) == 84);

/* Function pointer assignment checks the complete submission signature. */
PmResult (PMAPI_CALL *const checked_submit)(
    PmQueue, const PmSubmitInfo*, PmFence) = pmQueueSubmit;

void PMAPI_CALL checked_diagnostic_callback(
    void* user, const PmValidationDiagnostic* diagnostic) {
    (void)user;
    (void)diagnostic;
}

const PmValidationDiagnosticsCreateInfo checked_diagnostics = {
    PM_STRUCTURE_TYPE_VALIDATION_DIAGNOSTICS_CREATE_INFO,
    NULL,
    checked_diagnostic_callback,
    NULL
};

void checked_abi_symbols(void) {
    (void)checked_submit;
    (void)checked_diagnostics;
}
