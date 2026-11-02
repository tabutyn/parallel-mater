# F8-76: Windows CUDA managed metadata repair

Windows GTX 1080 normal execution now passes all registered tests and the focused
ownership regressions. The complete public capture still fails frozen performance
acceptance, so **F8-76 remains incomplete; do not merge**. Actual executing source is
`68e7d8f3af24c47d8d4d1da374d7ef600b9f04e2`, tree `ff988c80e7ae52285197b6fd9fb23e0c55c93c0e`. This report does not qualify F8-26.

## Ownership diagnosis and repair

The device reports managed memory support with `concurrentManagedAccess=0`.
Retiring public frames does not retire earlier asynchronous native uploads.
The CPU therefore cannot inspect managed arenas merely because the public
runtime is idle.

PEG PAINT faults in `native_realtime_limits` while reading managed paint
liveness (baseline line 437). The installed `soft-body-fluid` case faults in
`add_fluid_soft_body_coupling` while traversing managed rest geometry (line
2137). Full regression and lifecycle execution also exposed rope/soft-body,
rope/cloth, smoke body/mesh identity, and public hit-box count accesses. Exact
symbols, source lines and exception parameters are in the JSON report and raw
archive; every retained failing-before control reports `0xc0000006` with
underlying status `0xc0000022` and no `CUDA_LAUNCH_BLOCKING` override.

Paint descriptors and immutable mesh/body identities now have ordinary host
owners. Device descriptors are published at resource edits, and retirement
completes before payload release. Three geometry/attachment constructors wait
at their CPU resource-edit boundary. The public hit-box query snapshots live
counts together before launching queries. Its host-result path gains one
batched completion boundary; no fixed-tick wait or readback is added.

The entire production realtime backend matches the original file except five
host metadata substitutions. Kernels, numerical equations, dispatches,
iterations, timestep, spatial publication and budget formulas remain unchanged.
The three constructor waits use `Services.wait_idle`, which may be device-wide;
they are outside fixed ticks and are not per-kernel synchronization.

## Acceptance evidence

- Fresh Release: **144/144 tests pass, zero skips**, retaining all original 130
  tests and adding 14 strict plain/validation ownership regressions.
- All 14 corresponding controls fail against the pinned earlier drivers.
- Relocated SDK: all 12 packaging/public ABI checks pass.
- Public SDK lifecycle probe, both modes: painted-fluid creation, five repeated
  ticks, rejected edit with identical public bytes, six-pixel field replacement,
  clear/reset; soft-body/fluid construction twice, ten ticks, two atomic rejected
  edits, reset/replacement and recovery. Each successful probe tick checks one
  advance, one publication and zero intra-tick host readbacks.
- Installed public capture retained in full, both modes: 30 cases, 12 scenarios,
  five desktop workloads and five negative controls, with 600 warmup and 7,200
  measured ticks per workload at frozen counts and limits. **The evaluator fails**;
  the JSON report contains every failure and the archive retains every sample.
- Candidate v9 checker passes; original manifest SHA-256 remains
  `4ed4ed7a097928944ac5667e8120e2447affda962a258e56c01dfd0d90d9dd4b`.

Reviewable evidence: [F8-76-WINDOWS-CUDA-METADATA-PROGRESS.json](F8-76-WINDOWS-CUDA-METADATA-PROGRESS.json) and
[F8-76-WINDOWS-CUDA-METADATA-PROGRESS.zip](F8-76-WINDOWS-CUDA-METADATA-PROGRESS.zip). Archive SHA-256:
`6c7c472f0d7066b0e15a1787f46e2b5c9212c73740852dfbc911a3dd38e5d222`. Every archived member is hash-verified.
The archive includes exact commands, source diff, binary/SDK hashes, JUnit,
raw public samples, lifecycle source/results, before controls and diagnosis.

## Remaining performance acceptance

The unchanged desktop physics p95 limit is 10 ms and frame budget is 16.667 ms.
The public workload critical path includes completion waits. The existing
`pmWaitForFences` loop sleeps for 50 microseconds between checks; the observations
show Windows timer-sized latency plateaus. This is a source/measurement lead,
not a proved causal attribution. Six-family GPU execution itself also exceeds
10 ms in this capture. Read-only GPU samples observed low clocks; no device power
or clock configuration was changed. These observations do not justify discarding
stalls, modifying measurement boundaries or increasing limits.

Performance-only implementation beyond the leased managed-metadata repair scope
requires release/replan/claim before editing. The five negative controls require
a passing unmodified baseline, so failed baseline acceptance also prevents those
controls from establishing success. Their exact emitted statuses are retained.

## Refreeze boundary

The closure classification records each changed file and each protected hash.
API/ABI, kernel, adapter binding and contract categories are unchanged by this
repair. Runtime and schedule/telemetry metadata ownership changes require
refreeze. The public oracle and work-budget closure hashes also change because
the native realtime test file gains validation and strict-selector routing;
its existing fixtures, assertions, tolerances and workloads are byte-identical.

The capture uses v9 as an immutable qualification input and explicitly sets
`execution_identity_claimed=false`. F8-74 must refreeze after F8-73 and F8-76;
final Windows qualification needs a fresh F8-26 lease. No other physical device
is qualified by this Windows run.

Baseline provenance is explicit: the original retained Release identity record
describes the evidence harness checkout. Its separate, preserved provenance
annotation identifies the actual compiled source from the authoritative F8-26
failure report. Neither the original raw record nor candidate v9 was rewritten.
