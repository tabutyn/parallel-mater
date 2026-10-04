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
