# F8-23 frozen-candidate rejection

Status: `realtime-single-step-v1-dea47c1` is not executable through the
production public API and cannot receive physical qualification evidence.
The machine-readable rejection is
[`realtime-candidate-v1-rejection.json`](../../../conformance/v1/qualification/realtime-candidate-v1-rejection.json);
it preserves the old manifest hash, invalidates all eight physical rows and
forbids union with replacement evidence.

F8-23 built the frozen source closure on the registered Linux CPU host before
collecting any quality or timing samples. The clean `sdk-cpu` build completed
258/258 edges with GCC 12.2.0 and Slang 2026.18. Its production driver is
`build-sdk-cpu/libparallel_mater_driver_slang_cpu.so`, SHA-256
`648e702ac02e7dfbc62a0f6a083b17c141629d5fc82bd357158b236719a8758e`.

The physical-driver public-profile test passed only because it asserts the
unfinished behavior:

```text
PASS: reference driver advertises no unfinished realtime solver
```

At the protected candidate revision `dea47c178c3f1308abb54d46942f965b25e87eb6`,
`pmEnumerateExecutionProfiles` unconditionally returns zero records and
`select_execution_profile` unconditionally returns
`PM_ERROR_FEATURE_NOT_PRESENT` for the valid v1.0 request. The positive profile
tests use the mock driver, not the Slang CPU production binary.

Source inspection found a second independent blocker. Production
`SimulationRuntime::step_async` always calls the 156 KiB legacy
`simulation::schedule_frame`; it never calls `assemble_realtime_tick` or
`assemble_realtime_tick_measured`. The F8-20/F8-21 schedule and telemetry tests
therefore prove their model contracts, not native candidate execution. No
honest F8-23 observation can report one shared spatial publication, fixed native
dispatch/synchronization work, or realtime quality/timing from this binary.

Reproduction:

```sh
/home/thomas/Android/Sdk/cmake/4.1.2/bin/cmake --preset sdk-cpu \
  -DPARALLEL_MATER_SLANGC_EXECUTABLE=<Slang-2026.18>/bin/slangc
/home/thomas/Android/Sdk/cmake/4.1.2/bin/cmake --build --preset sdk-cpu --parallel 4
ctest --test-dir build-sdk-cpu --output-on-failure \
  -R '^parallel-mater-cpu-realtime-profile-reference-tests$' -V
ctest --test-dir build-sdk-cpu --output-on-failure \
  -R '^parallel-mater-realtime-profile-(plain|validation)-tests$'
```

The first CTest selects the production Slang CPU manifest and confirms zero
profiles. The second selects the mock driver and is contract-only. This failure
is preserved instead of being relabeled as unsupported success. F8-33--36 own
production spatial integration, native schedule/telemetry binding, truthful
public exposure and a replacement freeze. F8-23--30 must qualify only
`realtime-single-step-v1-fd78fa8` from F8-36.
