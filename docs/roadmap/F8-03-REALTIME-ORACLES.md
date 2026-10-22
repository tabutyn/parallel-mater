# F8-03: independent single-step oracles

Status: portable oracle and test contract implemented; no runtime or physical
device is qualified by this document.

The normative numbers remain in
`conformance/v1/qualification/realtime-profile-v1.json`. The executable oracle
is `conformance/v1/realtime_oracle.py`, and
`conformance/v1/realtime/oracle-suite-v1.json` defines the public test sequence,
fixture dimensions, accounting fields and evidence separation. The oracle reads
those sources directly. It does not copy the advertised interaction matrix or
the fixed device workloads into code.

## What is independent

- Domain cases come from every `domain_quality` entry in the profile contract.
- Interaction cases are the Cartesian expansion of the authoritative rules in
  `capability-obligations-v1.json`; there is no second pair/mode list.
- Backend execution case names and applicable runtime variants come from the
  F8-01 kernel/device inventory.
- Fixed workload counts and device-tier timing limits come directly from the
  F8-00 profile contract.
- Numerical metrics, continuous swept-pair completeness, boundary mutation,
  per-tick work accounting and host timing use portable reference code. They do
  not inspect a driver's solver memory or accept a backend-specific equation.

The public execution contract starts by enumerating the exact profile and then
uses only installed API operations to create resources, submit one 1/60-second
step, wait, query status/statistics and copy public state. A driver that does not
enumerate `realtime-single-step` 1.0 is recorded as unsupported. It is never run
through the legacy schedule and called a profile pass.

`validate_public_observation` is the executable result boundary. It accepts only
`parallel-mater-realtime-observation/v1` records identified as installed-public-
API output, verifies exact profile identity, rejects private-driver observations,
checks each step against the one-binary32-ULP timestep rule and one substep,
selects the assertions for that derived case, and evaluates its per-tick work.
Missing measurements fail; a driver cannot omit an inconvenient metric.

The F8 registry also closes the four public-trajectory gaps identified by F8-01:
distinct-fluid indexing, miscible fluid pairs, immiscible fluid pairs and paint
contact stamping. Their fixed small fixtures live in `oracle-suite-v1.json`, and
the kernel inventory points back to those case IDs. These cases remain
unsupported until a driver enumerates the exact profile. F8-20 runs them against
the integrated schedule instead of silently exercising legacy.

## Named limits and negative controls

Every numeric or Boolean limit in the following F8-00 sections becomes one
stable named assertion:

- profile clock/update rules and interpolation latency;
- the complete input envelope and exact algebraic iteration counts;
- measurement windows and every domain/mode quality threshold;
- atomic failure flags;
- timing allocations and every frozen workload count;
- legacy/result-separation flags.

The portable test suite evaluates both the boundary value and a value just
outside every assertion. Non-finite observations always fail. This makes adding
a contract limit without a negative control a test failure.

The compliance oracle applies the contract formula
`ratio = compliance / (dt^2 * effective_inverse_mass)` and
`response = 1 / (1 + ratio)`. RMS, nearest-rank percentiles and reciprocal
exchange residuals use the definitions in F8-00.

## Candidate completeness

Small scenes use continuously swept spheres with a support radius and maximum
solver correction. The independent oracle computes closest approach for every
unordered stable-ID pair. A runtime candidate set may contain false positives,
but omitting any brute-force swept overlap fails. Tests include empty, sparse,
crossing, correction-bound and deterministic randomized scenes. F8-06/F8-07
will apply this oracle to the production shared-index entrypoints.

## Boundary and failure behavior

The reference mutation model tests deterministic identity-priority births,
next-boundary removals, replacement generations and atomic phase transfer.
Destination-capacity rejection retains the source. A GPU-discovered capacity or
invalid-state failure preserves committed state, clock and pending inputs,
publishes no epoch and advances no time. Recovery is a later caller-controlled
submission; hidden retry substeps and second index updates fail.

## Work and timing records

Work is reported per tick. Required counters are dispatches, workgroups,
barriers, queue submissions, intra-tick host readbacks, bytes moved, dynamic
index updates/publications, time advances and algebraic iterations. Dispatches
must equal the emitted phase sum. Iterations equal the versioned budget exactly,
and coupling work equals four sweeps per registered edge. Successful ticks have
one publication/time advance and at most one index update. There are no
intra-tick host readbacks.

F8-03 deliberately does not invent final device dispatch ceilings; F8-21 owns
measured dispatch, synchronization and GPU-phase budgets after the schedule
exists. The fields and per-tick invariants are fixed here so F8-21 cannot replace
actual emitted work with hand-maintained estimates.

Frozen workload observations require the exact count-key set and exact integer
values. Missing, reduced, non-integral or invented counts fail, so a smaller
scene cannot inherit a tier's timing result.

Timing uses an injected monotonic clock. Exactly 600 warmup ticks are excluded,
all 7,200 measured samples (including stalls) are retained, and nearest-rank
p50/p95/p99/max plus simulated-to-wall ratio are checked against the selected
tier. Missing GPU timestamps remain explicitly unavailable; host time is never
relabeled as GPU phase time.

## Legacy and baseline evidence

Legacy goldens, schedules and results keep the `legacy-default` identity. A
one-substep legacy baseline may be compared with these realtime assertions and
may fail them; the failed assertion IDs are retained without changing the
thresholds. Such a baseline is diagnostic only and cannot satisfy a realtime
execution or physical qualification row.

Run the portable gate with:

```sh
python3 conformance/v1/realtime_oracle.py --check-contract
python3 -m unittest discover -s conformance/v1/tests -p 'test_*.py'
```

Later implementation and device tasks supply public-API observations to this
contract. F8-22 freezes their candidate revision; F8-23 through F8-30 provide
physical evidence. Until then, the profile remains runtime-unqualified.
