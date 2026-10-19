# Android Vulkan qualification — 2026-10-19

Device: Google Pixel 7 (`panther`), Android 17/API 37, September 2026
security patch, Mali-G710, Vulkan 1.4.343. The APK uses the loadable Parallel
Mater Vulkan driver, common Slang physics and the native Vulkan Gallery
renderer. Ordinary frames render directly to the Android swapchain without CPU
readback or software presentation.

Tested source revision: `973cc228d1170938092f3c815d8a2fa591d0c50c`.
APK SHA-256: `714fab7cff53b3a5b2ea5f14f309a2aac380f31a9680b4dc5ec547c496adcdee`.
Embedded Vulkan driver SHA-256:
`ad47917345a563b448e1e86a0eebbda2cb2cc88790a0c9a08f307bb9c6ec0a71`.

## Physics correctness

F5-15 separately qualified the public API on this device. All 25 native device
executions, 30 public trajectories, 12 analytic profiles and five adverse
lifecycle profiles ran successfully. Fifteen trajectories differ numerically
from CUDA golden output, and several capability tuples still lack exact
analytic or lifecycle contracts. Those remain explicit qualification failures;
Gallery execution does not broaden the physics claim.

Mali previously hung when divergent 16-lane soft-body branches reached a
workgroup barrier that had been used as a masked warp barrier. Vulkan and D3D12
now select scalar shared-Slang soft-body kernels. A separate Mali SPIR-V
compiler crash was fixed by ordering acyclic pointer declarations before use.

## Complete Gallery sweep

All 28 capability-driven catalog scenes were available and passed 120
diagnostic frames. The shared qualification exercised scene selection,
pause/resume, camera, directional, count and primary actions. Every scene also
performed pause, inactive-frame suppression, foreground resume and reset.

The Android shell then passed real background/foreground recovery, swapchain
surface recreation, resize, rotation and native teardown. The report retained
`vulkan-direct`, zero ordinary readback pixels and zero software presentations.
Logs contain no Vulkan validation error, fatal process signal or failed native
teardown.

The UI sweep checked named, reachable controls for gravity, authored force,
motor/steering, release/restore, dump-bed and camera/help behavior. User-facing
text contains no roadmap or qualification terms. Tracked screenshots show the
large landscape controls and scene information on the physical phone.

## Ordinary presentation and thermal result

Diagnostic rendering intentionally reads back pixels, so its timing is not a
display-performance claim. A separate ordinary-presentation run measured two
60-frame windows for every scene: 3,360 direct Vulkan frames over 797.1 seconds.
The second-window observations were:

| Scene | FPS | Mean work (ms) | p95 work (ms) |
| --- | ---: | ---: | ---: |
| Rigid body | 8.36 | 106.880 | 119.108 |
| Constraint: fixed | 4.27 | 220.954 | 231.210 |
| Constraint: point | 4.53 | 207.356 | 260.146 |
| Constraint: hinge + slider | 2.25 | 430.383 | 525.517 |
| Constraint: piston | 18.02 | 47.523 | 57.231 |
| Constraint: generic | 11.92 | 75.412 | 89.482 |
| Constraint: motor + spring | 5.19 | 178.636 | 187.483 |
| Dump | 1.94 | 499.152 | 507.971 |
| Fluid | 7.77 | 116.463 | 143.838 |
| Fluid rigid | 5.12 | 182.524 | 216.524 |
| Peg paint | 7.21 | 127.103 | 152.083 |
| Cloth | 11.85 | 76.188 | 86.744 |
| Cloth tear | 11.40 | 80.697 | 98.730 |
| Cloth paint | 11.51 | 79.088 | 93.111 |
| Water cloth | 6.37 | 146.089 | 167.090 |
| Soft body | 9.30 | 97.678 | 112.727 |
| Soft body rigid | 9.39 | 97.904 | 114.378 |
| Soft body cloth | 3.65 | 261.388 | 275.560 |
| Soft body fluid | 2.31 | 418.586 | 450.101 |
| Rope | 3.96 | 240.866 | 369.868 |
| Rope fluid | 3.45 | 276.861 | 316.342 |
| Rope soft body | 2.73 | 352.092 | 374.485 |
| Rope cloth | 4.94 | 190.175 | 210.783 |
| Smoke | 11.29 | 80.384 | 92.059 |
| Smoke water | 4.19 | 226.667 | 262.663 |
| Smoke soft body | 1.66 | 585.746 | 620.789 |
| Smoke cloth | 6.37 | 145.491 | 169.582 |
| Smoke rope | 4.49 | 209.926 | 232.163 |

Rates ranged from 1.66 to 18.02 FPS with a 5.16 FPS median. No scene achieved
30 FPS. Second-window rate divided by first-window rate had a 0.68 minimum and
0.98 median. Android thermal status remained 0 after every scene. G3D
temperature started at 28 °C, reached 43 °C and ended at 40 °C. Maximum public
physics-buffer footprint in the diagnostic report was 219,759,476 bytes.

The `pixel-7-preview` profile therefore remains a conservative functional
profile, not a verified performance profile. Its 640×288 landscape render is
visibly coarse, but increasing resolution is not justified until physics frame
work improves. The 30 updates/s value controls the fixed simulation clock; the
device measurements do not claim that display rate.

## Reproduction and artifacts

Build, install and public-API commands are in the
[Android README](../examples/android/README.md). Run the common Gallery check,
UI audit and ordinary-presentation measurement in that order. Revisioned raw
reports, logs, hashes and screenshots are tracked in
[`docs/roadmap/evidence/F7-18-android-gallery/`](roadmap/evidence/F7-18-android-gallery/).

This bounded sweep establishes scene availability, controls, recovery,
rendering, memory observation and absence of thermal throttling. It does not
establish hours-long battery endurance or acceptable performance on other
Android devices.
