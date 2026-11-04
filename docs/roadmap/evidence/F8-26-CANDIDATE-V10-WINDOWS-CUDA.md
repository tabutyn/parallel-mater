# F8-26 candidate v10 Windows CUDA integration

Accepted under the scoped integration contract from PR #667 and fresh claim
PR #668. The original execution claim and release remain recorded as PR #663
and PR #665. PEG PAINT scene quality and timing have separate failed/deferred
verdicts.

Candidate `realtime-single-step-v1-5bacbe2` uses exact source
`5bacbe2b1e0fbfad565fb9477bb6873396e31d5d`, tree
`67f5088c2e9f50938c55856e3683326e42675310`, and manifest SHA-256
`29e051d94a89a39d51c2128b212165d60da6da7de4b2c107a72ae7cc5b77f20a`.
The physical target is Windows GTX 1080 CUDA, NVIDIA driver 582.66.

## Integration results

- All 46 retained Release artifacts match the exact-source F8-78 build hashes.
  All nine frozen source closures match v10. Fresh execution passes all 146
  registered tests, including 58 CUDA-labeled tests, with zero skips.
- Fresh SDK relocation and independent consumer/conformance builds pass all 12
  checks. The immutable v10 manifest and routing were installed before capture.
  Producer, validator, fixtures, contracts and native binaries retain their
  exact-source contents.
- Plain and validation modes each complete all 30 public cases, 49 capability
  rows and 12 physical scenarios. Frozen physical/numerical, work-budget and
  atomic-recovery assertions pass. Independent validation agrees with the
  original capture report.
- Five workloads per mode retain frozen counts, 600 warmup and 7,200 measured
  ticks. All 72,000 measured ticks succeed. All 288,000 timing values remain
  available, and percentile/maximum summaries were recomputed from them.
- All five injected violations add their expected rejection in each mode.
  Their timing-contaminated canonical baseline failures remain recorded.

## Installed Gallery

The managed Windows installation and Start-menu shortcut point to candidate
source `5bacbe2b`, reporting Gallery 0.3.0. Every installed file hash was checked.
The frozen Ninja build places portable manifests beneath `Release/manifests`.
A hash-verified staging directory joins those generated manifests with the root
executables for the existing installer. Its source mapping is retained.

All 28 Gallery scenes execute the normal 240-frame window with one physics step
per frame. Twenty-seven built-in scene checks pass. PEG PAINT completes all 240
ticks with zero capacity overflows, then fails its deposition check with **7
painted texels against 100 required**. Liquid Y bounds reach -11.9014 to
-11.4821. F9-09 owns this preserved scene-quality failure after the F8 exit.
The original threshold and workload remain part of its repair acceptance.

A six-frame hidden interactive startup passes, produces its image and CSV, and
keeps at most one physics step per displayed frame. The retained image was
visually inspected for rendered scene content and the timing overlay. Initial
short-window diagnostics and the normal-window failed result remain in the
archive. [Recorded scene debt](F8-26-CANDIDATE-V10-GALLERY-DEBT.json)

## Timing debt

| Workload | Plain physics p95 ms | Validation physics p95 ms |
|---|---:|---:|
| rigid-wall | 3.7887 | 3.0285 |
| fluid-rigid | 3.1156 | 5.1332 |
| deformable-mixed | 8.6777 | 11.1380 |
| smoke-coupled | 2.0890 | 4.6169 |
| six-family-mixed | 16.7733 | 22.2918 |

Original latency/rate limits and every failure remain in the report. F9-07 and
F9-08 own timing optimization and canonical performance-control qualification.

## Evidence

The structured report separates the original execution lease from the later
scope plan, fresh claim and acceptance. Every physical observation uses the
same frozen candidate. The archive contains complete captures, original failed
reports, differential controls, Gallery images and frame records, build/SDK
identities, deployment staging and exact reproduction scripts.

- [Structured qualification report](F8-26-CANDIDATE-V10-WINDOWS-CUDA.json)
- [Raw evidence archive](F8-26-CANDIDATE-V10-WINDOWS-CUDA.zip)

Archive SHA-256 is
`b780ff46f01be7be38fd43d075e99af683134ba1e939068f3168eba591603ef6`.
All 295 member hashes and archive CRC checks pass. Archive size is 11,913,098 bytes.
