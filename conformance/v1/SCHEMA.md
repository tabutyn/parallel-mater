# Conformance schemas

## Case input: `parallel-mater-conformance-case/v1`

Each case contains a stable `id`, title, `analytic` or `integrated_glb` kind,
complete world settings, named resources, ordered per-frame commands,
checkpoint frames, tolerance values, invariants, and a coverage list.
Integrated resources contain a repository-relative GLB path whose SHA-256 is
pinned in the manifest. All quantities use metres, kilograms, seconds, and
radians.

Commands execute immediately before the step whose zero-based index is
`frame`. Checkpoint zero is the initial state; checkpoint N is the state after
N complete steps. Resource names, not allocation handles, are the portable
identity contract.

## Result: `parallel-mater-conformance-result/v1`

Every result contains:

- `backend`: diagnostic backend name;
- `case_id` and `case_sha256`: exact input identity and tolerance provenance;
- `checkpoints`: ordered frames containing resources, canonical contacts, and
  invariant values;
- `diagnostics`: non-gating timings and backend-specific state hashes;
- `provenance`: non-gating run details.

Each resource declares a `type`, stable `id`, and `tolerance_class`. Rigid
states carry position, orientation, and linear/angular velocity. Constraint
states carry exact presence, enabled, and broken values. Particle/deformable
resources carry exact counts, stable IDs where available, complete or sampled
state, aggregate bounds, center of mass, momentum, kinetic energy, maximum
speed, and exact topology invariants. Smoke additionally carries projection
residual and divergence; rope carries maximum strain; fluids carry foam.

## Report: `parallel-mater-conformance-report/v1`

The comparator emits per-case `passed` status and structured differences with
JSON paths, messages, expected values, and actual values. Human-readable output
uses the same differences. Diagnostic fields are deliberately ignored.

## Execution outcome metadata

Completed v1 result documents may add `execution_status: "completed"` and
`quality_status: "not_evaluated"`; older results omit both. Incomplete execution
uses `parallel-mater-conformance-execution/v1` with `case_id`, `case_sha256`,
`execution_status` (`unsupported` or `failed`), `quality_status`, and `reason`.
These documents contain no physics result and cannot pass comparison.
Report v1 adds per-case `status` and `reason`, an `unsupported` total, and
`scope: "reference_regression"`. The existing `passed` boolean is false for
unsupported cases. `failed` excludes unsupported cases; qualification callers
must require both totals to be zero.

## Quality profile: `parallel-mater-conformance-quality-profile/v1`

The quality profile is an immutable, hashed contract independent of trajectory
goldens. It declares the required fixture IDs, exact timestep/mass-ratio matrix
and numerical limits. Each threshold names its specification source. The core
profile covers equal-material pressure/contact and viscosity with linear and
angular momentum, dissipative kinetic energy and split-versus-combined state.

## Quality result: `parallel-mater-conformance-quality-result/v1`

A result names the profile ID and exact SHA-256, reports completed execution,
and supplies every required fixture/sample. Each sample contains two endpoint
aggregates: positive mass, linear/angular momentum before and after, external
impulse/angular impulse, kinetic energy before and after, and external work.
The evaluator derives internal exchange and residuals from those values.

Split/combined state contains the support radius and two particle arrays keyed
by input identity. Position, velocity and acceleration errors are normalized by
`h`, `h/dt` and `h/dt^2`; temperature is exact and foam uses the profile's
absolute tolerance. Missing identities or sweep points fail.

## Quality report: `parallel-mater-conformance-quality-report/v1`

The report has `scope: "analytic_invariants"`, the profile ID/hash, driver and
device identity, fixture and sample counts, a pass flag, and structured
failures with path, message, actual value and limit. Unsupported/incomplete
execution is a failure. This report is separate from
`parallel-mater-conformance-report/v1` reference regression.

## Lifecycle profile: `parallel-mater-conformance-lifecycle-profile/v1`

The immutable profile declares exact checkpoint sequences for destructive
capacity recovery and a three-family interaction graph with an unrelated edge.
Each checkpoint names its expected `PmResult`, tracked resources, active state,
generation/state relations to earlier checkpoints, and graph-fingerprint
relations. Expected command errors belong inside a completed lifecycle case.

## Lifecycle result: `parallel-mater-conformance-lifecycle-result/v1`

A result pins the profile ID/hash and driver/device identity. Each completed
case lists its participating families, unrelated resources and observations.
Every observation contains a symbolic public command outcome and every tracked
resource's harness write ordinal, active state and canonical public-state
SHA-256. The ordinal records accepted public writes; it is not a private driver
generation or handle. Commands complete and state is observed before stepping.
Graph cases also contain the active public graph descriptor SHA-256. Hashes
exclude private handles, timings and diagnostic state.

## Lifecycle report: `parallel-mater-conformance-lifecycle-report/v1`

The report has `scope: "lifecycle_invariants"`, case/checkpoint totals and
structured failures. Missing, extra, malformed, failed or unsupported cases
fail the report. The report is independent of trajectory regression and
analytic numerical quality.

## Capability obligations: `parallel-mater-conformance-obligations/v1`

The hashed profile maps exact interaction capability keys to scoped evidence
contracts. Rules may name a trajectory contract, which creates separate
execution and reference-regression obligations, plus explicit lifecycle and
analytic-invariant contracts. A `missing` scope is an intentional failing gap,
not a waiver. Canonical registry tests require every API 0.2 interaction record
to match exactly one rule.

## Qualification result: `parallel-mater-conformance-qualification-result/v1`

A revisioned result records a clean source commit, driver name/version/binary
SHA-256, device identity and driver version, compiler identity, exact advertised
capability keys and scoped evidence. Every evidence record names its scope,
contract ID/hash, source commit, artifact SHA-256 and one of `passed`, `failed`,
`unsupported` or `missing`. The same artifact may support distinct scopes only
through separate records; one scope never substitutes for another.

## Qualification report: `parallel-mater-conformance-qualification-report/v1`

The report has `scope: "capability_qualification"`, one result per advertised
record, per-scope status, provenance, unused evidence and structured failures.
Unknown records, stale hashes, dirty or mismatched revisions, missing contracts,
absent evidence and non-passing required evidence leave the capability
unqualified. Optional timing remains visible without certifying correctness.

## Evidence sources: `parallel-mater-evidence-sources/v1`

The source registry lists expected production drivers, the exact qualification
profile, raw qualification results and supplemental historical reports. Every
input path carries a SHA-256. Changing or replacing any result requires an
explicit registry and generated-index update.

## Driver evidence index: `parallel-mater-driver-evidence-index/v1`

The generated index preserves each result's source, driver binary, device and
toolchain tuple; exact capability key and per-scope status; known failures; and
input hashes. Only a raw result that passes the current qualification evaluator
contributes qualified capability records. Missing results, stale inputs,
failed/unsupported evidence and supplemental reports remain visible without
certifying support. The generated block in `docs/QUALIFICATION.md` is checked
byte-for-byte against this index.

## Benchmark measurement: `parallel-mater-benchmark-measurement/v1`

The headless measurement records clean source, driver binary, device, toolchain
and workload identity; an exact quality profile/result hash; and raw timing and
public memory samples. At least 20 samples are required. Submitted percentile
or throughput summaries are not accepted.

## Benchmark report: `parallel-mater-benchmark-qualification-report/v1`

The report contains computed operations/second, nearest-rank median/p95 time per
operation and peak memory fields. Each metric repeats the quality-profile hash.
Only a passing raw quality result from the exact benchmark tuple makes the
report qualified; stale, missing, failed or unsupported quality results leave
the report unqualified. Benchmark-bound quality results must include matching
clean `source`, `toolchain`, and driver version/binary provenance in addition to
the quality result's base driver name and device fields.
