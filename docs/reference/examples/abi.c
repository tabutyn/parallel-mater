/* Compile-only checks for the draft's portable buffer layouts. */
#include "../api/parallel_mater.h"
#ifdef __cplusplus
#define CHECK(condition) static_assert(condition, #condition)
#else
#define CHECK(condition) _Static_assert(condition, #condition)
#endif

CHECK(sizeof(float) == 4);
CHECK(sizeof(PmBool32) == 4);
CHECK(sizeof(PmDeviceSize) == 8);
CHECK(sizeof(PmRigidState) == 64);
CHECK(offsetof(PmRigidState, position) == 0);
CHECK(offsetof(PmRigidState, orientation) == 16);
CHECK(offsetof(PmRigidState, linearVelocity) == 32);
CHECK(offsetof(PmRigidState, angularVelocity) == 48);
CHECK(sizeof(PmParticleState) == 32);
CHECK(offsetof(PmParticleState, temperatureKelvin) == 12);
CHECK(offsetof(PmParticleState, velocity) == 16);
CHECK(offsetof(PmParticleState, reserved) == 28);
CHECK(sizeof(PmStepStatistics) == 88);
CHECK(offsetof(PmStepStatistics, executionResult) == 80);
CHECK(offsetof(PmStepStatistics, reserved) == 84);

/* Function pointer assignment checks the complete submission signature. */
PmResult (PMAPI_CALL *const checked_submit)(
    PmQueue, const PmSubmitInfo*, PmFence) = pmQueueSubmit;
