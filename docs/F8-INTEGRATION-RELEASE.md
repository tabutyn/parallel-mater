# F8 single-path integration delivery

This is the release note for the F8 development integration delivery. It ships
the SDK and Gallery architecture in which normal simulation creation has one
fixed-tick production path. It is **not** a claim that every driver is physically
correct, every Gallery scene is convincing, or every target sustains 60 Hz.

The machine-readable delivery record is
[`F8-31-INTEGRATED-DELIVERY.json`](roadmap/evidence/F8-31-INTEGRATED-DELIVERY.json).
It binds the source, packages, installed applications, device results, retained
failures, and evidence-file hashes described below.

## Delivered identity

| Field | Value |
|---|---|
| Candidate | `realtime-single-step-v1-5bacbe2` |
| Source revision | `5bacbe2b1e0fbfad565fb9477bb6873396e31d5d` |
| Source tree | `67f5088c2e9f50938c55856e3683326e42675310` |
| Candidate manifest SHA-256 | `29e051d94a89a39d51c2128b212165d60da6da7de4b2c107a72ae7cc5b77f20a` |
| Public API | 0.3.0 |
| Driver interface | 2 |
| Gallery execution contract | `fixed-tick-60hz-v1` |

Every successful tick advances once by 1/60 second, publishes one shared
dynamic spatial epoch, submits bounded work, and commits once. Normal
`pmCreateSimulation` selects that path for every supported domain and
interaction. There is no application, Gallery, environment, backend, or build
switch for a legacy scheduler. Unsupported capabilities fail explicitly; a GPU
driver never silently changes to CPU physics.

Any material runtime, kernel, public API, driver, or Gallery application change
invalidates this delivery identity. Such a change needs a new candidate and
fresh qualification for affected targets. Documentation and immutable evidence
added after the freeze do not change the candidate executable identity.

## API 0.3 migration

API 0.3 is intentionally incompatible with API 0.2. Applications must:

1. request `PM_API_VERSION_0_3`;
2. remove `PM_EXT_simulation_profile` from extension lists;
3. remove `PmSimulationProfileCreateInfo` from simulation creation and pass a
   null `pNext`;
4. stop calling `pmEnumerateExecutionProfiles`; and
5. record `pmCmdStepSimulation` with one substep and a timestep within one
   binary32 ULP of 1/60 second.

The loader rejects API 0.2 and incompatible driver/layer interfaces. It does
not accept old variable-timestep or substep semantics and reinterpret them.
The application-facing resources, commands, submit, synchronization, and output
model remains the integration boundary.

## Device and driver result

“SDK/Gallery integration passed” below means the sole schedule, installed
package or application, lifecycle, and fixed-work contracts passed. It does not
override a separate frozen correctness or performance failure.

| Physical target | Integration | Frozen correctness | Performance | Disclosed limit |
|---|---|---|---|---|
| Windows GTX 1080, CUDA | Pass | Pass | Not qualified | Six-family physics p95: 16.7733 ms plain, 22.2918 ms validation. PEG PAINT deposits 7/100 required texels. |
| Windows GTX 1080, D3D12 | Pass | **Fail** | Not adjudicated | Advertises 46/49 required rows; cannot select six-family mixed; 27/28 Gallery scenes pass. |
| MacBook Air M4, Metal | Pass | Pass | Not qualified | Six-family physics p95: 12.111333 ms plain, 11.656125 ms validation versus 10 ms. |
| iPad13,8 M1, Metal | Pass | Pass with scoped scene debt | Pass for its frozen tier | Worst physics p95: 5.025583 ms versus 8.5 ms. Fluid-rope reciprocity/distribution checks fail. |
| Pixel 7, Vulkan | Pass | **Fail** | Not qualified | Three lifecycle and four quality evaluations fail; exact frozen workload capture is missing. |
| Linux laptop, Slang CPU | Pass | **Fail** | Pass for its frozen tier | All 12 applicable public cases differ from the frozen CUDA reference. |
| Linux RTX 3050 Ti, CUDA | Pass | **Fail** | Pass with stalls retained | All 30 public cases differ from the frozen reference; cloth-water repeatability fails. |
| Linux RTX 3050 Ti, Vulkan | Pass | **Fail** | Pass with stalls retained | All 30 public cases differ; three named repeatability cases fail. |

These failures are release information, not unsupported-hardware labels and not
passes. Windows D3D12 and Android Vulkan are installed, executable integration
targets whose required correctness rows fail. The Linux package/runtime path
passes while its frozen reference comparison fails. Candidate-v9 observations
are historical and are not combined with these candidate-v10 rows.

## Packages and installed applications

Relocated external-consumer checks pass for the Windows CUDA and D3D12 packages
and Linux CPU, CUDA, and Vulkan packages. Those checks install, move, discover,
and consume the public SDK, build the installed conformance runner, distinguish
missing-driver and unsupported-capability errors, and reject 0.2 and 0.4 package
requests. Their complete file hashes remain in the device evidence archives
referenced by the delivery record.

The installed Apple applications match the frozen candidate:

- `/Applications/ParallelMater Gallery.app` is signed, is the canonical Dock
  target, reports revision `5bacbe2`, selects Metal, and completed a six-frame
  direct-Metal smoke. Its build-info and executable hashes match the F8-28
  qualified installation.
- The connected iPad app freshly reported revision `5bacbe2`, backend `metal`,
  renderer `Metal GPU renderer`, and `fixed-tick-60hz-v1`. F8-29 binds its app,
  embedded driver, and runtime SDK hashes and exercised all 28 scenes for 120
  frames each.
- The Pixel 7 APK and Vulkan driver hashes are retained in F8-30. Installation,
  Gallery catalog, UI, lifecycle, and sustained runner contracts pass, but the
  correctness failures above prevent qualification.

Build and deployment commands remain platform-specific packaging operations;
they do not select different simulation schedules. Use the documented CMake
presets, Android build script, and iPad deployment wrapper. On macOS the normal
Metal preset refreshes the canonical `/Applications` app and Dock item.

## Gallery presentation, lifecycle, and memory

Shared Gallery pacing tests cover 30, 60, 90, 120, and 144 Hz. A render callback
submits at most one fixed tick. At 30 Hz the simulation deliberately runs in
slow motion; at refresh rates above 60 Hz rendering interpolates without adding
physics steps. Pause, reset, scene change, backgrounding, and long stalls clear
display debt, so foregrounding cannot trigger a catch-up burst. Headless clients
may issue fixed ticks as fast as they can without a display clock.

The candidate iPad sweep passed attach, resize, background/foreground, failure
recovery, invalid-scene recovery, pause/resume, camera, scene selection, count,
directional, and primary-action checks. Its 28 scene footprints ranged from
47,530,728 bytes to a retained peak of 361,923,352 bytes for Cloth Tear. Normal
mobile rendering used the public GPU output buffer; CPU readback was diagnostic
only. Android retained two sustained 60-frame windows per scene with no thermal
throttling, but its slowest Smoke Soft Body window reached 173.605 ms frame p95
and does not establish 60 Hz.

## Native dispatch reduction

F8 reduced actual native work; it did not merely rename one simulation step:

| Workload | Before | Delivered | Reduction / ceiling |
|---|---:|---:|---:|
| Metal rigid public fixture, total dispatches | 112 | 35 | 68.75% |
| Metal rigid public fixture, domain dispatches | 88 | 11 | 87.5% |
| Metal cooperative rigid solve | up to 3,102 staged solve dispatches at 386 bodies | 1 solve plus 1 prepare dispatch | Fixed independent ceiling |
| Shared coupling public fixture | Unbounded per-record/exhaustive paths | 467 CUDA / 457 D3D12 total dispatches | 512 total, 288 coupling, 16 candidate ceiling |
| Apple Smoke Soft Body | 1,277 dispatches and 1,712 dependency barriers | 269 dispatches and 632 barriers | 78.935% dispatch reduction |

The Metal rigid path has zero intra-tick host readbacks and host waits in the
measured fixture. Couplings use one spatial build and publication. Apple Smoke
Soft Body GPU p50 improved 26.292% on macOS and 31.603% on iPadOS, but record
relocation remained dominant and total frame throughput did not materially
improve. This evidence supports a dispatch-reduction claim, not a universal 60-FPS claim.

## Remaining work

- F9-01 through F9-06 repair visible rigid, liquid, cloth, soft-body, rope, and
  smoke behavior one system at a time. The solver work limits remain unchanged.
- F9-09 repairs the retained PEG PAINT deposition failure.
- F9-07 and F9-08 own the unchanged physics/frame/ratio targets and a fresh
  broad device-performance candidate after scene repairs.
- The Android, D3D12, and Linux frozen-correctness failures in the table remain
  explicit input to the F8 exit review. Shipping this integration does not
  convert them into passes.

Use [qualification](QUALIFICATION.md) for capability-specific evidence and
[the candidate freeze](roadmap/F8-74-CANDIDATE-FREEZE.md) for immutable source
closure. Run `python3 tools/check_f8_31_delivery.py` to verify this delivery
record, evidence hashes, retained verdicts, and API migration statements.
