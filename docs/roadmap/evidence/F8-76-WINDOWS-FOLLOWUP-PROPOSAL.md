# Proposed Windows follow-up split after F8-76

This is a review proposal. It changes no canonical task card, acceptance limit
or current lease. Draft repair PR [#643](https://github.com/tabutyn/parallel-mater/pull/643)
remains incomplete under the current F8-76 acceptance contract.

## Evidence requiring a scope decision

The managed-memory repair passes all 144 registered Release tests with no skip,
all 14 pinned failing-before controls, the relocated SDK, both public lifecycle
probes and all 30 public cases / 12 physical scenarios in both modes. The complete
frozen workload capture has zero failed ticks, but five physics p95 values exceed
10 ms; six-family also misses the 60 Hz frame/rate gates. All five negative
controls remain failed because their unmodified baseline is not passing.

An isolated public-consumer diagnostic uses the unchanged production driver and
the same six-family workload/counts. Replacing the copied consumer's public fence
wait with a status/yield loop lowers p95 from 32.0458 ms to 19.9639 ms, with GPU
p95 19.881984 ms in that second run. This shorter 600/1200-tick exploratory pair
does not qualify performance, but shows that removing host polling delay alone
will not satisfy the unchanged 10 ms gate. Busy polling is not the proposed fix.

The current F8-76 scope authorizes managed metadata ownership repair; it does not
authorize a general public synchronization rewrite or shared GPU optimization.
Repository `AGENTS.md` requires: “Freeze scope/locks during a lease; release and
replan if the task must expand.”

## Proposed planning changes

1. Keep F8-76 as the bounded managed metadata repair. Preserve its source,
   before/after reproductions, lifetime/atomic recovery requirements, all original
   tests, new regressions, SDK checks, physical quality/work/recovery checks and
   closure classification. Its third acceptance item would require the complete
   unmodified workload capture and explicit failure classification, while moving
   successful performance and negative-control acceptance to the new performance
   task below. This is an explicit proposed acceptance split, not a claim that
   the current card has passed.
2. Add a leaf task for public completion waiting on Windows, depending on the
   accepted metadata repair. Use completion notifications rather than timer-sized
   polling sleeps, preserving finite/infinite timeout, wait-any/wait-all, error,
   lifetime and validation semantics across backends. Require focused deterministic
   synchronization tests, affected portable/reference checks and retained physical
   Windows measurements. Do not introduce global timer/power configuration or a
   busy-spin workaround.
3. Add a leaf task for complete Windows GTX 1080 performance acceptance, depending
   on the repair and completion-wait task. Profile the remaining native GPU and
   command overhead, coordinate shared files with the live Metal work, and retain
   one fixed tick/publication and every existing work budget. Require all original
   tests plus ownership regressions, relocated SDK, both 30-case / 12-scenario
   captures, all five workloads and five negative controls, 600 warmup / 7200
   measured ticks, **10 ms physics p95**, **16.667 ms frame budget** and **0.99
   minimum simulated/wall ratio**. No smaller workload, discarded stall, raised
   budget, weakened quality or backend physics fork is allowed.
4. Make F8-74 depend on both new tasks as well as the existing repairs. Candidate
   v9 remains immutable failed history. Final Windows qualification still needs
   the refrozen candidate and a fresh F8-26 lease.

This proposal permits the reviewed ownership fix to merge separately while
keeping the complete original physical quality and performance gates mandatory
before refreeze/qualification. No canonical split or performance implementation
will occur until the planning change is accepted and its task lease merges.
