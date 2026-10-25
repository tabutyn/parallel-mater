# P3-02 Android Vulkan schedule optimization

Result: completed on physical Pixel 7 (`panther`, Mali-G710), Android 17/API
37, with Slang 2026.18 and Android NDK 29.0.14206865. Both source revisions
were clean, the runner installed the named APK, and the local/installed APK and
driver bytes matched. Thermal status remained 0.

The baseline is commit `0eba746cfe3de95beb2407c042f8dac355da0d30`.
The candidate is commit `9424029633d5b08734391cf8e1b63ba38ef9958f`.
Every timing window retained the legacy 30 Hz fixed timestep, eight solver
substeps, 60 scheduled ticks, the fixed scene workload, direct Vulkan
presentation, and opt-in schedule telemetry.

| Workload | Baseline physics | Candidate physics | Speedup | Baseline / candidate render | Dispatches |
|---|---:|---:|---:|---:|---:|
| Rigid Body | 106.607 ms | 53.2045 ms, mean of two 60-tick candidate windows | 2.0037x | 1.617 / 1.2495 ms | 138 / 122 |
| Fluid | 113.340 ms | 50.743 ms | 2.2336x | 1.948 / 1.780 ms | 1,163 / 1,155 |
| Smoke Soft Body | 600.346 ms | 301.128 ms | 1.9937x | 1.799 / 1.448 ms | 6,841 / 6,825 |

The two final-source Rigid Body windows measured 52.364 and 54.045 ms. Their
120-tick aggregate is reported instead of selecting only the faster repeat.
Render ratios are 0.7727, 0.9138 and 0.8049 respectively, all below the allowed
1.20 ratio. Live/capacity workload counts are retained in the profile files.

The implementation adds bounded, opt-in host/GPU/dispatch phase telemetry;
packs the symmetric AVBD matrix; specializes the small Vulkan AVBD solve and
leaf reduction; caches small-solve pair/Jacobian data; narrows barriers by
producer/consumer domain; and folds the identical speed-limit operation into
the small solver tail. The latter removes eight launches without dropping the
clamp or changing its equations. The general solver remains selected for the
sensitive four-body legacy coupling workload.

Validation:

- `cmake --build build-p3-02-vulkan --parallel 2`: pass.
- `Test-AndroidVulkanSchedule.py`: deployment/provenance pass; all recorded
  workload windows pass runner checks; no thermal throttling or fatal runtime
  diagnostics.
- `Test-AndroidGallery.py`: 28/28 scenes pass at 120 frames each; direct
  presentation, pause/reset, background/foreground, resize, rotation, failure
  recovery, invalid-scene recovery and teardown pass.
- `Test-AndroidVulkan.py`: 25/25 device executions pass. Command/API,
  asynchronous submission, liquid/reset, corpus execution, twelve analytic
  quality profiles, five adverse lifecycle profiles and Vulkan device contracts
  pass. Independent evaluators pass 17/18.

The one failed evaluator is preserved separately: CUDA-golden trajectory
comparison passes 16 cases and reports 14 numerical mismatches. It is not
converted into a physical-qualification or performance pass. Exact per-case
differences are in `reference-report.json`; all cases executed successfully on
device.

Artifacts:

- `baseline-profile.json`: complete current-main 60-tick windows and phase data.
- `candidate-profile.json`: complete candidate Rigid/Fluid/Smoke windows.
- `candidate-rigid-repeat.json`: second candidate Rigid window used by the
  aggregate.
- `gallery-lifecycle.json`: 28-scene presentation and lifecycle result.
- `native-capture.json` and `native-results.json`: device/toolchain identity,
  hashes, 25 native invocation results and 18 evaluator results.
- `reference-report.json`: explicit CUDA-golden comparison result.
