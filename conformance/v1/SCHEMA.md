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
