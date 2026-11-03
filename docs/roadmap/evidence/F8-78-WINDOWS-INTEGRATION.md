# F8-78 integrated Windows validation

Accepted under the revised integration contract from PR #657, with timing debt
assigned to F9-07/F9-08. Claim PR #658 reserved the exact lease
`426c0a4a-1f2a-4a23-9faf-a5bcdf25f531`.

The clean Release build and all executions use source
`5bacbe2b1e0fbfad565fb9477bb6873396e31d5d` on Windows GTX 1080, with accepted
F8-73/F8-76/F8-77 repairs integrated. CUDA uses normal asynchronous execution.
The report records the source tree, compiler, GPU/driver, managed-memory
attributes, installed SDK hashes and compiled artifact identities.

## Passing integration evidence

- All 146 registered tests pass with zero skips. Every previous F8-77 test is
  retained, including ownership, failure recovery and completion-wait coverage.
- All 12 relocated-SDK checks pass, including independently built consumers,
  conformance inputs, incompatible versions and unsupported/missing-driver cases.
- Both plain and validation modes complete all 30 public cases and all 12
  physical scenarios. Frozen physical/numerical assertions, capability coverage,
  one fixed advance, one spatial publication and independent work/recovery
  checks pass.
- All five workloads complete in both modes at frozen live/capacity counts,
  600 warmup and 7,200 measured ticks. All 72,000 measured ticks succeed.
  All 288,000 CPU/GPU/physics/frame timing values remain in the archive. The
  verifier recomputes every reported percentile and maximum from those samples.

## Retained timing debt

| Workload | Plain physics p95 ms | Validation physics p95 ms |
|---|---:|---:|
| rigid-wall | 3.5879 | 3.7471 |
| fluid-rigid | 3.2716 | 5.1158 |
| deformable-mixed | 9.0004 | 11.1551 |
| smoke-coupled | 2.3684 | 5.0230 |
| six-family-mixed | 18.5305 | 20.6715 |

Six-family misses the original 10 ms physics and 16.667 ms frame targets in both
modes. Its simulated/wall ratios are 0.989253 plain and 0.837211 validation,
against 0.99. Validation deformable also misses the 10 ms physics target.
The original oracle retains eight failures in plain mode and nine in validation.
Those comprise three/four timing target misses and five timing-contaminated
canonical negative-control baselines per mode.

Each of the five injected defects adds its expected rejection in both modes.
The report retains the full unmodified baseline and injected reports plus their
failure differences. These demonstrate capability omission, rejected-step
atomicity, hidden-retry, retained-stall data and failed-tick rejection separately.
Canonical performance-control qualification remains pending under F9-07/F9-08.

The historical F8-76 comparison retains its original source and all measurements.
That capture predates both F8-77 completion notifications and F8-73 shared-runtime
changes. The report identifies source differences and classifies all nine
protected closures. Frozen API/ABI and contract closures are byte-identical.
The work-budget closure differs through added managed-auxiliary coverage and
validation-mode enabling, with independent budget values and assertions preserved.

## Review and reproduction

The JSON report binds the archive SHA-256 and verified member hashes. The archive
contains complete raw captures, original oracle failures, differential control
results, build/test/SDK logs and the exact execution/evaluation scripts.
Run the recorded commands from the named source on the specified Windows CUDA
host. Reproduction scripts record the original local paths for toolchain, build,
SDK and evidence directories.

- [Structured acceptance and provenance](F8-78-WINDOWS-INTEGRATION.json)
- [Raw evidence and reproduction scripts](F8-78-WINDOWS-INTEGRATION.zip)

F8-74 still owns the replacement-candidate freeze and fresh device routing.
This capture uses the historical v9 oracle contract with its actual executed
source identified separately. Scene-quality improvements remain assigned to
F9-01 through F9-06, and broad 60 Hz qualification awaits later passing evidence.
