# F8-21 measured schedule telemetry

F8-21 adds a bounded, opt-in measurement contract around the F8-20 one-tick
schedule and extends the production Vulkan profiler to report actual emitted
work. It does not advertise or silently select the realtime profile.

## Per-tick contract

`realtime_telemetry.hpp` groups work into boundary, predict, index, candidate,
domain, coupling and commit phases. Every phase and the tick total report actual
dispatches, workgroups, bytes moved, index builds/refits/publications, numerical
iterations, barriers, submissions, readbacks and host waits. Profile, budget and
telemetry schema versions travel with every record.

The measured assembler asks the backend for counter deltas after each schedule
visit. It rejects overflow, a second build/refit/publication, an extra numerical
iteration, an unbudgeted dispatch/barrier/submission, or a mandatory readback or
host wait. These checks occur before commit whenever the offending work is
observed. The ordinary assembler remains a separate template instantiation: it
does not require a telemetry method, allocate query storage, take a branch,
issue a query/readback or wait.

CPU encode/submit/wait nanoseconds and GPU nanoseconds are represented per
phase. GPU timing is explicitly `unsupported`, `pending` or `valid`; CPU time is
never substituted for unavailable GPU time. Instrumentation cost is separate
from simulation work and records query resets/writes, asynchronous result reads,
result bytes, forced waits and measured CPU overhead.

`RealtimeTelemetryQueryRing` owns fixed-size generation-tagged slots. A full
ring or an incomplete result returns `PM_NOT_READY`; it never evicts an in-flight
tick or waits. Unsupported adapters resolve zero GPU nanoseconds with an
explicit unsupported status and create no timestamp-query traffic.

## Vulkan adapter evidence

The existing opt-in Vulkan schedule profiler now emits schema v2. Its phase and
submission records contain actual dispatch/workgroup, barrier, transfer,
readback and byte counts; CPU record/submit/wait durations; GPU timestamps; and
instrumentation traffic. Timestamp results are read only after the normal fence
has completed. With telemetry disabled, the query pool is not created.

A clean build from implementation commit `453a238` ran on a physical Pixel 7
with Mali-G710. After 10 warmup ticks, 30 measured legacy ticks per scene
reported:

| Legacy workload | Dispatches | Workgroups | Barriers | Readbacks | GPU p50 / p95 / p99 (ms) | Sim/wall |
|---|---:|---:|---:|---:|---:|---:|
| Rigid Body | 122 | 962 | 125 | 0 | 47.420 / 52.346 / 61.038 | 0.539 |
| Fluid | 1,155 | 1,155 | 1,330 | 2 | 26.362 / 28.565 / 31.586 | 0.572 |
| Smoke Soft Body | 6,825 | 153,416 | 10,822 | 2 | 234.409 / 246.968 / 248.022 | 0.111 |

The report binds the clean source revision, APK and installed driver hashes,
package identity, a fresh run nonce, workload capacities/live counts, warmup,
p50/p95/p99 CPU and GPU distributions, thermals and simulation-to-wall ratio.
All instrumentation records reported zero forced waits. A separately built
AArch64 telemetry-contract binary also passed on the Pixel 7.

## Regression coverage and evidence boundary

Negative controls inject an extra numerical iteration, duplicate spatial build
and excess dispatch budget. Query-capacity and not-ready cases prove no wait or
eviction. An unmeasured backend proves the ordinary schedule has zero telemetry
calls. The host F8 suite passed 16/16; Android report parsing passed 20/20.

The physical timings above measure the unchanged legacy 30 Hz/eight-substep
production path and validate the Vulkan telemetry implementation only. They are
not a realtime-single-step performance claim. The new schedule remains
unadvertised while F8-22 freezes and completes one native candidate; F8-23
through F8-30 perform candidate qualification.

Exact identities, distributions and commands are in
[`evidence/F8-21-realtime-telemetry.json`](evidence/F8-21-realtime-telemetry.json).
