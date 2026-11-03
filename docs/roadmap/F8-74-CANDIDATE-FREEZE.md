# F8-74 integrated repair candidate freeze

F8-74 freezes replacement candidate `realtime-single-step-v1-5bacbe2` at
source revision `5bacbe2b1e0fbfad565fb9477bb6873396e31d5d` and tree
`67f5088c2e9f50938c55856e3683326e42675310`. Its machine-readable manifest is
[`realtime-candidate-v10.json`](../../conformance/v1/qualification/realtime-candidate-v10.json),
SHA-256 `29e051d94a89a39d51c2128b212165d60da6da7de4b2c107a72ae7cc5b77f20a`.

This is the exact clean source physically exercised by the integrated F8-78
Windows run after the accepted F8-73 Metal/shared-Gallery repair and the
F8-75/F8-76/F8-77 Windows fixture, ownership and asynchronous-completion
repairs. F8-74 changes freeze metadata and task routing only; it does not change
the executable source identity.

## Separate verdicts

- **F8 integration/correctness:** passed for the frozen source, with replacement
  device rows pending. Exact-source CI is green, and the accepted repair
  evidence passes correctness, physical assertions, resource recovery and
  independent work budgets.
- **Final performance:** not qualified. The 10 ms physics p95, 16.666666666666668
  ms frame p95 and 0.99 simulated/wall targets remain unchanged. Windows
  six-family physics p95 is 18.5305 ms plain and 20.6715 ms with validation;
  its ratios are 0.989253 and 0.837211. Validation deformable physics p95 is
  11.1551 ms. F9-07 and F9-08 own these retained timing-only misses.

This split does not weaken a solver, workload, case, tolerance, timeout or
capability. It prevents timing debt from blocking F8 integration while also
preventing the candidate from claiming a performance pass it has not earned.

## Protected closure

`tools/generate_realtime_candidate_v10.py` hashes the immutable source tree and
emits candidate v10. `tools/check_realtime_candidate.py` independently extracts
that revision and verifies every closure, contract input, exact-source CI
artifact, accepted repair, retained timing miss, historical manifest and
replacement task.

| Category | Files | Closure SHA-256 |
|---|---:|---|
| API and ABI | 18 | `f360354da65e282c915245f4316841ad5fd631c3f2123dc70749b18d037527c5` |
| Shared Slang kernels | 39 | `dfe3ef649f8cfd363be331280218e533783750f82e080d1dd7503664171bb5b1` |
| Adapter binding | 49 | `4cd9ad476cdc1bd54f84fe17f053f810f3c620bf2250a7bc3f59fce71f13e6d9` |
| Runtime | 127 | `d7fbd9df07da2a41169f6a0b5accd77e0b7622a063e551f623a15e224a0cbd3e` |
| Schedule and telemetry | 6 | `32b8e4de98281e9fcb07174ef227ee26599e6ed35b1c9f8cd7f5c8a14b8c16d6` |
| Public contract | 6 | `2306546edbe897e760b4c08703994db48838fb949d33075a00dda03e4d16edd7` |
| Public oracle | 543 | `b42e6d7c12ef27231b666dbbb284492cf3c5d03678f1be6cac65dbcf221f339f` |
| Work budget | 7 | `f3af25d62dc3ecc6d4b443b2d6004bea45e2371322da2a8704bf7a6d55127b7f` |
| Shared Gallery | 160 | `186f826501c6ac2f092d23f98983af0871b67f0d10745052964855f737d6f867` |

The frozen surface has six public domains, eight advertised interaction modes,
49 capabilities, 30 public cases, 12 scenarios, 37 measurements, 26 canonical
Slang modules, 234 entries, 32 kernel groups, five runtime variants and eight
physical targets. API 0.3 and driver ABI 2 are unchanged.

## Exact-source evidence

[GitHub Actions run 41708083151](https://github.com/tabutyn/parallel-mater/actions/runs/41708083151)
checked out the exact source and passed all 17 required jobs. Its ten artifacts
have downloaded archive hashes and independently regenerated extracted-tree
hashes in
[`F8-74-PROTECTED-CI.json`](evidence/F8-74-PROTECTED-CI.json), SHA-256
`bc7103710434ab9636e2a4a2f72bfec5820669130185333bd6bbf71506f58cfe`.
Linux, macOS, Windows and iPad records all report the exact clean revision.

F8-78 used that source on a physical GTX 1080. The clean Release suite passed
146/146 with no skips, and installed SDK relocation passed 12/12. Plain and
validation modes each passed all 30 public cases, 12 physical scenarios, five
workloads and five independent negative controls. Each workload retained 600
warmup and 7200 measured ticks. In total the evidence retains 72,000 measured
ticks and 288,000 raw timing values, including stalls and failed timing targets.

## Installed-public reproduction

Build and install the exact source with the normal platform preset and driver.
Then run the installed conformance producer and frozen validator; do not edit
the manifest, profile, fixtures or capture between the two commands:

```sh
git show -s --format='%H %T' 5bacbe2b1e0fbfad565fb9477bb6873396e31d5d

python3 <installed-prefix>/share/parallel-mater/conformance/v1/realtime_device_capture.py \
  --runner <installed-prefix>/bin/parallel-mater-conformance \
  --driver-manifest <installed-prefix>/share/parallel-mater/drivers.d/<driver>.json \
  --validation-layer-manifest <installed-prefix>/share/parallel-mater/layers.d/validation.json \
  --runtime-variant <cpu-runtime|cuda-runtime|vulkan-runtime|d3d12-runtime|metal-runtime> \
  --target-id <registered-physical-target-id> \
  --output <capture-directory>

python3 <installed-prefix>/share/parallel-mater/conformance/v1/realtime_device_suite.py \
  --capture <capture-directory>/capture.json \
  --candidate-manifest conformance/v1/qualification/realtime-candidate-v10.json \
  --output <capture-directory>/report.json
```

The producer uses only installed public API resources, commands, submission and
readback. Device cards F8-26 through F8-30 and F8-85 through F8-87 record their
platform-specific build/install command, deployment hash and toolchain without
changing the frozen command contract.

Regenerate and independently validate the freeze with:

```sh
python3 tools/generate_realtime_candidate_v10.py \
  --ci-evidence docs/roadmap/evidence/F8-74-PROTECTED-CI.json --route
sha256sum conformance/v1/qualification/realtime-candidate-v10.json
python3 tools/check_realtime_candidate.py
python3 tools/roadmap.py check
```

## History and routing

Candidate v9 remains byte-immutable at SHA-256
`4ed4ed7a097928944ac5667e8120e2447affda962a258e56c01dfd0d90d9dd4b`.
Its macOS Metal and Windows CUDA failures remain immutable at SHA-256
`e244ba5b67dd7c6584e822b57df26a9b2c870098854c08b7b51efa9400936da7`
and `b0858b7ff180c76bd1ee15af09c3fd3cb59f229d2ff655852773404c7e4b35f8`.
Its completed Linux CPU, CUDA and Vulkan observations are history only.

F8-26 through F8-30 bind unfinished rows to candidate v10. F8-85, F8-86 and
F8-87 are bounded replacement leaves for the completed candidate-v9 Linux CPU,
CUDA and Vulkan rows. F8-31 and F8-99 depend on all three. Evidence from
candidate v9 or any earlier candidate cannot be unioned with candidate v10.
Any later protected-file change creates another candidate rather than mutating
this freeze.
