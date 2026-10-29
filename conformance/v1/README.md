# Parallel Mater physics conformance v1

This directory defines versioned reference-regression and physical-quality
inputs for independently loaded drivers. Existing captures do not qualify every
advertised capability. It is separate from physics
capture v1, which remains a debugging/replay format.

For shared implementation code and small host/device equation fixtures, see
the [rigid-contact solver contract](../../docs/CONTACT_SOLVER.md). Full-engine
conformance remains required; passing the portable row tests is not sufficient.

## Layout

- `cases/` contains canonical, byte-stable JSON inputs in SI units. Analytic
  inputs are serialized from the CUDA-free C++ case registry. Integrated
  inputs reference committed GLBs; `manifest.json` pins every asset by SHA-256.
- `golden/cuda/` contains one CUDA result per case. CUDA is the v1 reference,
  but matching is tolerance-based rather than bit-identical.
- `compare.py` compares a driver's result directory against a reference and
  writes an optional machine-readable report.
- `quality.py` evaluates analytic momentum, angular-momentum, dissipative-energy,
  split-population and rigid surface-separation invariants without a reference
  backend. Immutable profiles live in `quality/`.
- `lifecycle.py` evaluates adverse capacity recovery, mixed-family graph
  rollback and representation-specific smoke replacement/removal without a
  reference backend. Immutable profiles live in `lifecycle/`.
- `qualification.py` maps each advertised interaction record to revisioned
  evidence obligations in `qualification/capability-obligations-v1.json`.
- `realtime/scenario-fixtures-v1.json` is the single executable definition of
  the twelve fixed-tick physical scenarios. It fixes public scene inputs,
  events, measurements, tick outcomes and diagnostics without selecting a
  backend. `realtime_scenario_fixtures.py` validates it and emits the identical
  runner plan for CPU, CUDA, Vulkan, D3D12 or Metal.
- `realtime_device_suite.py` accepts physical results only when every scenario
  names the exact shared fixture SHA-256, was run both plain and validation-enabled,
  used one 1/60-second public step per tick and retained the atomic failed tick
  in the capacity/recovery fixture.
- `evidence_index.py` revalidates registered qualification results, retains
  failures and provenance, and checks the generated per-driver documentation.
- `evidence/index-v1.json` is the published driver/capability matrix. Its source
  registry pins every input report by SHA-256.
- `manifest.json` records the exact source revision, deterministic settings,
  toolchain/device provenance, and GLB hashes used for the reviewed goldens.

The 30-case trajectory registry covers direct rigid integration and lifecycle changes, compound
weld release/rebuild, breaking, all eight constraint types, fluids, cloth,
soft bodies, ropes, smoke, and every implemented pairwise coupling. The
required integrated scenes (`PassiveActive`, `ConstraintFixed`, `FluidRigid`,
`ClothWater`, `SoftbodyRigidBody`, `SmokeWater`, and `SmokeRope`) are included
alongside the remaining coupling and subsystem scenes.

The separate two-case lifecycle profile covers destructive replacement at exact
capacity and a three-family liquid/rope/soft-body graph beside an unrelated
smoke/cloth edge. These operation-sequence cases compare state within one run;
they do not use CUDA trajectory goldens.

The smoke lifecycle profile independently exercises particle and grid smoke at
their exact declared capacities. A single public-API sequence proves
replacement, atomic particle/grid capacity rejection, removal, and
deterministic recovery. The smoke self-viscosity quality profile pairs control
and viscous public outputs for both representations and checks relative-motion
dissipation, bounded energy, impulse residual, and authored speed limits.

## Validation tools

`parallel-mater-conformance` loads every canonical case through the shared
scene loader and executes it through the public `pm*` command API. It does not
link a backend-specific simulation interface. Validate the canonical inputs
and comparator with:

```text
ctest --test-dir <build> -R parallel-mater-conformance
```

When a file already named by `manifest.json` changes, refresh all pinned hashes
with the same JSON/XML line-ending normalization used by the installed runner:

```text
python3 conformance/v1/update_manifest.py
python3 conformance/v1/update_manifest.py --check
```

Review the manifest diff with the input change. The tool does not add files,
change provenance, regenerate the evidence index or update trajectory goldens;
those remain explicit review steps. A second refresh must produce no diff.

Record the selected driver's public identity and exact advertised interaction
keys with `--provenance` and `--capabilities`. The capability output uses the
same canonical names and graph-limit representation as qualification results.

Validate the shared realtime scenario definitions, or inspect the exact plan a
device runner receives, with:

```text
python3 conformance/v1/realtime_scenario_fixtures.py --check
python3 conformance/v1/realtime_scenario_fixtures.py \
  --runtime-variant metal-runtime
```

The runtime selector is the only backend-specific value in that plan. The
fixture definition and capture validator are part of the F8-61 protected
candidate closure. The integrity-checked
`qualification/active-realtime-candidate.json` pointer selects the one manifest
used by default; changing that pointer never permits evidence from two
candidates to be combined.

```text
PM_DRIVER_FILES=<build>/manifests/slang-metal.json \
  <build>/parallel-mater-conformance --provenance
PM_DRIVER_FILES=<build>/manifests/slang-metal.json \
  <build>/parallel-mater-conformance --capabilities
```

Run one case or the complete corpus and compare its
`parallel-mater-conformance-result/v1` output with:

```text
PM_DRIVER_FILES=<build>/manifests/slang-cuda.json \
  <build>/parallel-mater-conformance \
  --case all --output /path/to/driver-results

python3 conformance/v1/compare.py \
  --cases conformance/v1/cases \
  --expected conformance/v1/golden/cuda \
  --actual /path/to/driver-results \
  --report /path/to/report.json
```

Every driver result keeps the canonical case SHA-256. Driver identity,
timings, and implementation-specific hashes are diagnostic and never gate
correctness.

## Independent SDK build

Enable `PARALLEL_MATER_BUILD_CONFORMANCE=ON` with gallery and native drivers
OFF to build the runner and `ParallelMater::scene` / `ParallelMater::preparation`
static tooling libraries. No Slang compiler or graphics SDK is required when
shader compiler gates and drivers are also OFF. The scene headers contain host
geometry and authored inputs, not private solver handles. These C++ tooling
libraries are source-level helpers; only the `pm*` C API is the driver boundary.

After installing the SDK, build this directory as an independent project:

```bash
cmake -S conformance/v1 -B build-conformance-sdk \
  -DCMAKE_PREFIX_PATH=/path/to/installed-sdk
cmake --build build-conformance-sdk --parallel 1
build-conformance-sdk/parallel-mater-conformance --check-inputs --root /path/to/corpus-root
```

The corpus root contains `conformance/v1` and `examples/assets`. `--root` avoids
embedding a developer checkout as a runtime requirement. Select a separately
installed driver with `PM_DRIVER_FILES`; the runner links only the loader and
host scene helpers, never a reference driver.

Execution emits `completed`, `unsupported`, or `failed` for each selected case
and continues the corpus after a case failure. Unsupported means advertised
features cannot satisfy scene requirements; errors after capability selection
remain failures. No installed driver is a setup failure. Exit codes: 0 for a
run with completed cases and no failures, 2 for execution/setup failure, 77
when every selected case is unsupported. Mixed completed/unsupported runs must
still go through the comparator; execution exit zero is not qualification.

The reference-independent quality gate consumes aggregate endpoint state and
split/combined particle state produced through the public API. The evaluator
can account for profile-permitted external impulse, angular impulse and work;
the core closed-system fixtures require all three to be zero. Run it with:

```text
python3 conformance/v1/quality.py \
  --actual /path/to/core-conservation-stability-v1.json \
  --report /path/to/quality-report.json
```

The profile requires pressure/contact and viscosity fixtures at timesteps
0.001, 0.01 and 0.05 seconds and endpoint mass ratios 1:1, 1:2 and 1:4. A
nonzero exchange floor prevents an inactive operator from passing conservation.
Momentum and split/combined limits come from `docs/INTERACTION-MODES.md`; the
fixed sweep is the reviewed F2-08 matrix. The quality report embeds the exact
profile SHA-256. Missing, duplicate, unsupported, malformed or nonfinite
samples fail. This gate never reads CUDA goldens and never treats better
reference agreement as proof of conservation or stability.

Rigid contact uses the separate `rigid-brick-penetration-v1` profile. Its
result repeats the fixed `RigidBody.glb`, eight-brick, 300-frame configuration
at one 1/60-second step per frame and reports the maximum pairwise penetration
observed by the independent closed-mesh sampler in
`tests/gallery/rigid_mesh_checks.hpp`.
Select one installed driver and capture the public-API result, then evaluate it:

```text
parallel-mater-conformance --quality rigid-brick-penetration-v1 \
  --output /path/to/rigid-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/rigid-brick-penetration-v1.json \
  --actual /path/to/rigid-quality-results/rigid-brick-penetration-v1.json \
  --report /path/to/rigid-brick-report.json
```

The existing 2 mm threshold is unchanged. A driver-contact depth is diagnostic;
it cannot replace the independent mesh measurement. The original CUDA/D3D12
failures and corrected observations remain in the revisioned Windows report.

Rigid attachment has its own `rigid-attachment-v1` profile. It runs the
accepted six-body, five-constraint `ConstraintHinge.glb` settling fixture for
2,400 frames and derives hinge-anchor, hinge-axis, slider-drift, slider-axis and
slider-travel measurements from authored local frames and public rigid output.
It also requires every authored constraint to remain enabled and unbroken.
The existing 2 mm, 0.014, 2 mm, 0.001 and 0.01--1.0 m gates are unchanged.

```text
parallel-mater-conformance --quality rigid-attachment-v1 \
  --output /path/to/rigid-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/rigid-attachment-v1.json \
  --actual /path/to/rigid-quality-results/rigid-attachment-v1.json \
  --report /path/to/rigid-attachment-report.json
```

Both profiles pin their scene SHA-256, schedule, physical units and oracle.
They are separate analytic obligations for the advertised same-domain rigid
contact and attachment capabilities; neither substitutes for lifecycle or
reference-regression evidence.

Cloth-rope endpoint attachment uses the immutable
`cloth-rope-attachment-v1` profile. Its public-API fixture suspends a rope from
a supported cloth vertex, bounds endpoint and pinned-support drift, and checks
reciprocity by comparing the public cloth reaction with an independent rope
momentum balance. The paired
`cloth-rope-attachment-lifecycle-v1` profile checks exact graph capacity,
replacement, rejected-overflow rollback, removal and deterministic recovery:

```text
parallel-mater-conformance --quality cloth-rope-attachment-v1 \
  --output /path/to/cloth-rope-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/cloth-rope-attachment-v1.json \
  --actual /path/to/cloth-rope-quality-results/cloth-rope-attachment-v1.json \
  --report /path/to/cloth-rope-quality-report.json

parallel-mater-conformance \
  --lifecycle-profile cloth-rope-attachment-lifecycle-v1 \
  --output /path/to/cloth-rope-lifecycle-result.json
python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/cloth-rope-attachment-lifecycle-v1.json \
  --actual /path/to/cloth-rope-lifecycle-result.json \
  --report /path/to/cloth-rope-lifecycle-report.json
```

Soft-body-rope contact uses the immutable `soft-rope-contact-v1` profile. Its
public-API fixture starts a resampled rope inside a moving bound tetrahedral
surface, reconstructs every public surface vertex from its node bindings, and
checks separation, reciprocal endpoint impulse, public reaction-force
consistency, and frictionless tangential slip. A second step removes the graph
and requires contact force to clear while relative slip remains, so a hidden
attachment cannot satisfy the contact contract. Capacity, replacement and
rollback remain a separate obligation supplied by the existing
`adverse-capacity-three-family-v1#three-family-partial-rollback` case:

```text
parallel-mater-conformance --quality soft-rope-contact-v1 \
  --output /path/to/soft-rope-contact-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/soft-rope-contact-v1.json \
  --actual /path/to/soft-rope-contact-quality-results/soft-rope-contact-v1.json \
  --report /path/to/soft-rope-contact-quality-report.json
```

Soft-body-rope endpoint attachment uses the immutable
`soft-rope-attachment-v1` profile. Its public-API fixture attaches the first
rope endpoint to a supported tetrahedral surface, bounds endpoint and pinned
support drift, and compares the public soft-body reaction with an independent
rope momentum balance. The public coupling representation coauthors contact;
the profile requires that public contact-force channel to remain inactive so
contact cannot substitute for attachment. Capacity, replacement and rollback
remain a separate obligation supplied by the existing
`adverse-capacity-three-family-v1#three-family-partial-rollback` case:

```text
parallel-mater-conformance --quality soft-rope-attachment-v1 \
  --output /path/to/soft-rope-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/soft-rope-attachment-v1.json \
  --actual /path/to/soft-rope-quality-results/soft-rope-attachment-v1.json \
  --report /path/to/soft-rope-quality-report.json
```

Intrinsic soft-body/rigid CONTACT uses `soft-rigid-contact-v1`. A prepared
four-node tetrahedron exposes a ten-vertex bound surface and strikes a centered
dynamic cube beside a separated control. The public-state oracle reconstructs
the surface from its bindings, counts each physical node mass once, checks
exterior clearance, reciprocal endpoint momentum, reduced-mass impulse,
closing speed, transverse/angular drift, and energy. This exact capability has
no `PmCouplingState`; its graph limit is `not-applicable`. The paired
`soft-rigid-contact-lifecycle-v1` profile therefore exercises bounded soft
endpoint replacement, record-time over-capacity rollback, removal, and
deterministic recovery while a public rigid endpoint stays unchanged:

```text
parallel-mater-conformance --quality soft-rigid-contact-v1 \
  --output /path/to/soft-rigid-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/soft-rigid-contact-v1.json \
  --actual /path/to/soft-rigid-quality-results/soft-rigid-contact-v1.json \
  --report /path/to/soft-rigid-quality-report.json

parallel-mater-conformance \
  --lifecycle-profile soft-rigid-contact-lifecycle-v1 \
  --output /path/to/soft-rigid-lifecycle-result.json
python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/soft-rigid-contact-lifecycle-v1.json \
  --actual /path/to/soft-rigid-lifecycle-result.json \
  --report /path/to/soft-rigid-lifecycle-report.json
```

Intrinsic rope/rigid CONTACT uses `rope-rigid-contact-v1`. A free two-node
rope segment receives a public preparation impulse, then strikes the centered
face of a dynamic cube beside a separated control. The public-state oracle
reconstructs capsule-to-mesh clearance and checks coherent reduced-mass
impulse, reciprocal rope/body momentum, public contact-force consistency,
frictionless tangential slip, closing speed, angular drift, energy and speed
bounds. Both endpoint attachments are disabled and this exact capability has
no `PmCouplingState`; its graph limit is `not-applicable`:

```text
parallel-mater-conformance --quality rope-rigid-contact-v1 \
  --output /path/to/rope-rigid-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/rope-rigid-contact-v1.json \
  --actual /path/to/rope-rigid-quality-results/rope-rigid-contact-v1.json \
  --report /path/to/rope-rigid-quality-report.json
```

The existing `case:rope-core` contract remains the independent trajectory and
lifecycle obligation. A checked physical aggregate binds both scopes to the
same completed `rope-core` result and passing comparison report; the analytic
fixture cannot substitute for its capacity, replacement or rollback coverage.

Intrinsic rope/rigid ATTACHMENT uses `rope-rigid-attachment-v1`. A public
force drives a dynamic cube through an off-center native rope endpoint while
the other endpoint remains free. The public-state oracle reconstructs the
rotated local anchor, bounds endpoint drift, and checks reciprocal rigid/free-
rope linear impulse, endpoint constraint-force consistency, off-center angular
reaction, inactive contact force, segment length, and speed limits. No
`PmCouplingState` exists for this graph-free capability:

```text
parallel-mater-conformance --quality rope-rigid-attachment-v1 \
  --output /path/to/rope-rigid-attachment-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/rope-rigid-attachment-v1.json \
  --actual /path/to/rope-rigid-attachment-quality-results/rope-rigid-attachment-v1.json \
  --report /path/to/rope-rigid-attachment-quality-report.json
```

`case:rope-core` remains independently required for rope-rigid attachment
trajectory and lifecycle evidence. The attachment fixture does not broaden or
replace that binding.

Soft-body-to-smoke CONTACT uses the immutable `soft-smoke-contact-v1`
profile for particle and projected-grid smoke. Paired no-edge and CONTACT-edge
runs derive crossing, surface clearance, normal rejection, tangential motion,
and one-way soft-body invariance from public node, bound-surface, and smoke
state. DRAG is present in the API descriptor but is made inert and is
qualified by the separate profile below. Grid graph capacity, replacement, rejected-overflow
rollback, removal, and recovery use `soft-smoke-contact-lifecycle-v1`:

```text
parallel-mater-conformance --quality soft-smoke-contact-v1 \
  --output /path/to/soft-smoke-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/soft-smoke-contact-v1.json \
  --actual /path/to/soft-smoke-quality-results/soft-smoke-contact-v1.json \
  --report /path/to/soft-smoke-quality-report.json

parallel-mater-conformance \
  --lifecycle-profile soft-smoke-contact-lifecycle-v1 \
  --output /path/to/soft-smoke-lifecycle-result.json
python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/soft-smoke-contact-lifecycle-v1.json \
  --actual /path/to/soft-smoke-lifecycle-result.json \
  --report /path/to/soft-smoke-lifecycle-report.json
```

Rope-to-smoke CONTACT uses `rope-smoke-contact-v1` for both particle and
projected-grid smoke. Paired no-edge and CONTACT-edge runs derive capsule
entry, declared-radius clearance, relative radial rejection, retained
tangential motion, speed bounds, and one-way dynamic-rope invariance from
public rope-node and smoke snapshots. The explicit `0.015 m` contact distance
equals the rope radius; bundled DRAG is inert. Two exact graph cases cover
capacity, replacement, rejected-overflow rollback, removal, and recovery:

```text
parallel-mater-conformance --quality rope-smoke-contact-v1 \
  --output /path/to/rope-smoke-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/rope-smoke-contact-v1.json \
  --actual /path/to/rope-smoke-quality-results/rope-smoke-contact-v1.json \
  --report /path/to/rope-smoke-quality-report.json

parallel-mater-conformance \
  --lifecycle-profile rope-smoke-contact-lifecycle-v1 \
  --output /path/to/rope-smoke-lifecycle-result.json
python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/rope-smoke-contact-lifecycle-v1.json \
  --actual /path/to/rope-smoke-lifecycle-result.json \
  --report /path/to/rope-smoke-lifecycle-report.json
```

Smoke-to-rigid-body CONTACT uses `smoke-rigid-contact-v1` for particle and
projected-grid smoke. Paired no-edge and CONTACT-edge runs send one public
smoke tracer through the negative face of a dynamic cube. The independent
oracle reconstructs that face from public rigid position and orientation,
then checks shell clearance, relative normal rejection, reciprocal rigid
impulse, finite bounded state, and a stationary no-edge rigid control. Zero
wind drag isolates CONTACT even though API 0.2 bundles CONTACT and DRAG in one
descriptor. Two exact graph cases cover capacity, replacement, rejected-
overflow rollback, removal, and recovery while retaining both endpoints:

```text
parallel-mater-conformance --quality smoke-rigid-contact-v1 \
  --output /path/to/smoke-rigid-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/smoke-rigid-contact-v1.json \
  --actual /path/to/smoke-rigid-quality-results/smoke-rigid-contact-v1.json \
  --report /path/to/smoke-rigid-quality-report.json

parallel-mater-conformance \
  --lifecycle-profile smoke-rigid-contact-lifecycle-v1 \
  --output /path/to/smoke-rigid-lifecycle-result.json
python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/smoke-rigid-contact-lifecycle-v1.json \
  --actual /path/to/smoke-rigid-lifecycle-result.json \
  --report /path/to/smoke-rigid-lifecycle-report.json
```

Smoke-to-rigid-body DRAG uses `smoke-rigid-drag-v1` for particle and
projected-grid smoke. Six paired public-state runs isolate zero-drag,
forward-flow, stationary-flow, and reverse-flow behavior. The oracle requires
non-vacuous velocity-dependent rigid and tracer response, forward/reverse
symmetry, geometric separation beyond the contact distance, and bounded
speed. Particle smoke additionally proves reciprocal momentum, force, and
dissipative energy from its public effective mass; grid tracer pressure stays
descriptive because it samples the private Eulerian projection and does not
stand in for that field's mass, force, or energy.
`smoke-rigid-drag-lifecycle-v1` supplies separate exact adverse graph cases
for particle and grid representations. These contracts make DRAG evidence
disjoint from the neighboring smoke-rigid CONTACT contracts; they do not by
themselves register a physical driver qualification result.

```text
parallel-mater-conformance --quality smoke-rigid-drag-v1 \
  --output /path/to/smoke-rigid-drag-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/smoke-rigid-drag-v1.json \
  --actual /path/to/smoke-rigid-drag-quality-results/smoke-rigid-drag-v1.json \
  --report /path/to/smoke-rigid-drag-quality-report.json

parallel-mater-conformance \
  --lifecycle-profile smoke-rigid-drag-lifecycle-v1 \
  --output /path/to/smoke-rigid-drag-lifecycle-result.json
python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/smoke-rigid-drag-lifecycle-v1.json \
  --actual /path/to/smoke-rigid-drag-lifecycle-result.json \
  --report /path/to/smoke-rigid-drag-lifecycle-report.json
```

Smoke-to-rope DRAG uses `smoke-rope-drag-v1` for particle and projected-grid
smoke. Six public-state runs pair zero-drag controls with forward, stationary,
and reverse carrier flow around a free two-node rope. The oracle bounds signed
mean rope-node response, acceleration, transverse drift, forward/reverse
symmetry, segment-length preservation, closest-segment tracer clearance, node
speed, and one-way smoke invariance. `smoke-rope-drag-lifecycle-v1` carries an
exact five-checkpoint graph case for each representation:

```text
parallel-mater-conformance --quality smoke-rope-drag-v1 \
  --output /path/to/smoke-rope-drag-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/smoke-rope-drag-v1.json \
  --actual /path/to/smoke-rope-drag-quality-results/smoke-rope-drag-v1.json \
  --report /path/to/smoke-rope-drag-quality-report.json

parallel-mater-conformance \
  --lifecycle-profile smoke-rope-drag-lifecycle-v1 \
  --output /path/to/smoke-rope-drag-lifecycle-result.json
python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/smoke-rope-drag-lifecycle-v1.json \
  --actual /path/to/smoke-rope-drag-lifecycle-result.json \
  --report /path/to/smoke-rope-drag-lifecycle-report.json
```

Smoke-to-soft-body DRAG uses `smoke-soft-drag-v1` for particle and projected-
grid smoke. Six public-state runs pair zero-drag controls with forward,
stationary, and reverse carrier flow. The oracle bounds signed soft-node
response, acceleration, transverse drift, forward/reverse symmetry, public
surface reconstruction, closest-triangle tracer clearance, node speed, and
one-way smoke invariance. `smoke-soft-drag-lifecycle-v1` carries an exact
five-checkpoint graph case for each representation:

```text
parallel-mater-conformance --quality smoke-soft-drag-v1 \
  --output /path/to/smoke-soft-drag-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/smoke-soft-drag-v1.json \
  --actual /path/to/smoke-soft-drag-quality-results/smoke-soft-drag-v1.json \
  --report /path/to/smoke-soft-drag-quality-report.json

parallel-mater-conformance \
  --lifecycle-profile smoke-soft-drag-lifecycle-v1 \
  --output /path/to/smoke-soft-drag-lifecycle-result.json
python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/smoke-soft-drag-lifecycle-v1.json \
  --actual /path/to/smoke-soft-drag-lifecycle-result.json \
  --report /path/to/smoke-soft-drag-lifecycle-report.json
```

Smoke self-pressure has one immutable profile with particle and projected-grid
variants. It derives particle pressure from public density state, checks public
grid tracer pressure activity, and bounds particle mean-velocity drift:

```text
parallel-mater-conformance --quality smoke-self-pressure-v1 \
  --output /path/to/smoke-quality-results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/smoke-self-pressure-v1.json \
  --actual /path/to/smoke-quality-results/smoke-self-pressure-v1.json \
  --report /path/to/smoke-pressure-report.json
```

The `rigid-kinematic-sweep-v1` profile independently adjudicates the
`rigid-direct` frame-21 kinematic impact. Capture all frames, then evaluate the
raw trajectory directly:

```text
parallel-mater-conformance --case rigid-direct --every-frame \
  --output /path/to/results
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/rigid-kinematic-sweep-v1.json \
  --actual /path/to/results/rigid-direct.json \
  --report /path/to/rigid-kinematic-sweep-report.json
```

The profile pins the unchanged case hash and requires checkpoints 0 through
60. It derives expected contact direction and torque sign from the declared
0.2 m-half-extent collider and off-centre sweep, enforces the Coulomb cone, and
bounds out-of-plane leakage from the fixture's mirror symmetry. It reads no
trajectory golden and does not weaken the `direct` comparison tolerance.

Capture the complete capacity and graph-lifecycle suite through the same
installed command used for trajectory conformance, then run its independent
gate:

```text
PM_DRIVER_FILES=<build>/manifests/slang-cuda.json \
  <build>/parallel-mater-conformance --lifecycle \
  --output /path/to/adverse-capacity-three-family-v1.json

python3 conformance/v1/lifecycle.py \
  --actual /path/to/adverse-capacity-three-family-v1.json \
  --report /path/to/lifecycle-report.json
```

Select the smoke-specific contract without changing platform code:

```text
PM_DRIVER_FILES=<build>/manifests/slang-metal.json \
  <build>/parallel-mater-conformance \
  --lifecycle-profile smoke-self-lifecycle-v1 \
  --output /path/to/smoke-self-lifecycle-v1.json

python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/smoke-self-lifecycle-v1.json \
  --actual /path/to/smoke-self-lifecycle-v1.json \
  --report /path/to/smoke-self-lifecycle-report.json
```

For an installed SDK, pass its installed corpus root explicitly:

```text
PM_DRIVER_FILES=/path/to/slang-cuda.json \
  /path/to/sdk/bin/parallel-mater-conformance --lifecycle \
  --root /path/to/sdk/share/parallel-mater \
  --output /path/to/adverse-capacity-three-family-v1.json
```

Select the smoke-specific lifecycle profile explicitly to capture bounded
particle and grid capacity, replacement, rejected-overflow rollback, removal
and recovery in one public-API run:

```text
parallel-mater-conformance --lifecycle smoke-self-pressure-lifecycle-v1 \
  --output /path/to/smoke-self-pressure-lifecycle-v1.json
python3 conformance/v1/lifecycle.py \
  --profile conformance/v1/lifecycle/smoke-self-pressure-lifecycle-v1.json \
  --actual /path/to/smoke-self-pressure-lifecycle-v1.json \
  --report /path/to/smoke-lifecycle-report.json
```

An individual producer can be checked while the other case is still absent by
adding `--case capacity-replacement-recovery` or
`--case three-family-partial-rollback`. Omitting `--case` always requires the
complete suite. The lower-level standalone capacity fixture writes its revisioned input
with `parallel-mater-capacity-recovery-tests --output <path> --driver-name
<driver>`. The graph fixture uses the same options on
`parallel-mater-interaction-graph-lifecycle-tests`. The installed CLI and
embedded qualification compile the same operation sequences and one lifecycle
result serializer; they do not merge standalone result files.
The CLI writes the result before returning 2 for a failed fixture or 77 when a
required fixture is unsupported, so either outcome remains reviewable.

The result records canonical public resource-state SHA-256 fingerprints and
harness write ordinals at every required checkpoint; ordinals are never private
driver handles. Each checkpoint completes and state is read before any
following physics step. Replacement must advance the target ordinal and change
its observable state. Graph resource fingerprints cover active count, capacity,
stable liquid identities and public positions canonicalized to 0.1 mm; this
keeps the lifecycle identity independent of no-force solver roundoff. Active
graph fingerprints cover every public coupling descriptor field without that
position canonicalization. `active: false` at the capacity checkpoint means the
failed step has no valid published target output; it does not infer destruction
of driver-private storage or forbid retrying the prior input. The unrelated
fluid must remain
publicly readable and unchanged before recovery. The mixed graph contains a
liquid/rope/soft-body component plus an unrelated smoke/cloth edge; a rejected
partial update must restore the prior complete graph and preserve every
endpoint. Expected command errors are observations inside a completed case.
Suite or case `failed` and `unsupported` outcomes fail this lifecycle gate and
never count as physical passes.

Validate a driver's advertised interaction records with:

```text
python3 conformance/v1/qualification.py \
  --actual /path/to/driver-qualification.json \
  --report /path/to/qualification-report.json
```

Do not hand-author a passing qualification result from a physical capture.
Derive it with the same evaluator after supplying every hashed raw artifact:

```text
python3 conformance/v1/qualification.py \
  --from-evidence conformance/v1/evidence/<physical-evidence>.json \
  --artifact-path conformance/v1/evidence/<physical-evidence>.json \
  --driver-binary <driver-library> \
  --provenance <provenance.json> --capabilities <capabilities.json> \
  --trajectory-result <trajectory.json> --trajectory-report <comparison.json> \
  --quality-result <quality.json> --quality-report <quality-report.json> \
  --lifecycle-result <lifecycle.json> --lifecycle-report <lifecycle-report.json> \
  --output conformance/v1/evidence/<qualification-result>.json \
  --report <qualification-report.json>
```

This path verifies the driver binary and every capture hash, checks provenance,
capability identity, contract hashes and raw pass reports, then writes the
result only after the ordinary qualification evaluator accepts it.

The obligation profile covers all 49 API 0.2 interaction records. Evidence is
keyed by both scope and immutable contract hash: compilation, execution,
reference regression, lifecycle invariants, analytic invariants and timing are
never interchangeable. Every record must match exactly one rule. Required
evidence must be `passed`, from the same clean source revision, and retain
driver binary, device and toolchain provenance. Known gaps are explicit
`missing_contract` failures. Timing is reported separately and does not make a
functional capability pass.

Regenerate the checked-in per-driver matrix and its documentation block with:

```text
python3 conformance/v1/evidence_index.py
```

CI and release review use `--check`. Raw qualification results are evaluated
again against the pinned obligations rather than trusted as summaries. Failed,
unsupported, incomplete or stale evidence stays in `known_failures` and yields
no qualified capability row. Older reports may be registered as supplemental
evidence; their source, binary, device and toolchain tuple remains visible, but
they cannot certify a current capability.

Text-contract hashes are checkout-stable: JSON and XML use UTF-8 bytes with
CRLF normalized to repository-canonical LF. Other bytes remain significant,
and binary artifacts are always hashed exactly. This permits the same pinned
contract to validate on Windows without making semantic text or binary changes
invisible.

Performance publication has a separate headless gate:

```text
python3 benchmarks/qualification.py \
  --measurement /path/to/benchmark-measurement.json \
  --quality-result /path/to/core-quality-result.json
```

It re-evaluates the raw quality result and rejects qualified scores unless the
profile hash and complete source/driver/device/toolchain tuple match. See
`docs/PERFORMANCE.md` for the measurement and report schemas.

Completed trajectory results explicitly say `quality_status: not_evaluated`.
The comparator reports reference-regression `passed`, `failed`, and
`unsupported` separately. A separate successful `quality.py` report is needed
to establish analytic physical quality.
Historical captures without execution metadata remain readable. Comparator exit
codes are 0 for all passed, 1 for any failure, 77 for incomplete unsupported
coverage without failures. Unsupported results never count as passes; malformed
case identity/hash still fails. Analytic/invariant quality gates remain separate
from agreement with CUDA captures and must gate performance qualification.

2026-10-16 evidence: gallery-free CPU build passed 34 checks, skipped two
unsupported domain tests. An installed and relocated SDK built this runner and
validated all 30 canonical inputs. CPU `rigid-direct` execution completed but
failed three final angular-velocity comparisons against the historical CUDA
capture (maximum absolute difference about 0.000256 versus about 0.0000102
allowed). CPU `cloth-core` produced unsupported/exit 77. No tolerance or golden
was changed by this separation.

The tracked [F5-04 isolation](../../docs/roadmap/evidence/F5-04-rigid-direct-divergence.json)
reproduces that result at its exact `603fee5` revision and retains same-revision
CPU/CUDA outputs for all 61 frames. Frame 20 is identical and contact-free; the
first contract failure appears on frame 21, in the first kinematic-box contact
submission, including dynamic-box angular velocity. Current `3acb413` CPU and
CUDA outputs pass the unchanged every-frame comparison, while both drift from
the old CUDA golden. F5-17 therefore tracks independent adjudication of the
shared response/reference instead of treating the remaining issue as CPU-only.

Subsequent [Windows GTX 1080 evidence](evidence/windows-gtx1080-physics-2026-10-16.json)
resolves the ten failing comparisons: the same-build SDK runner passes all 17
selected CUDA/D3D12 cases with zero unsupported results. At that historical F5
revision, independent eight-brick, four-substep, 300-frame mesh checks pass on
both drivers at 0.205278 mm against the unchanged 2 mm limit. The report retains
original divergences, source/case/
binary hashes and reproduction commands. These corrections do not qualify the
remaining corpus, CPU historical regression, other devices or performance.

## Matching rules

- IDs, counts, presence, enabled/broken state, and topology are exact.
- Finite floating values pass when
  `abs(error) <= abs_tolerance + rel_tolerance * max(abs(expected), abs(actual))`.
- Direct integration uses `1e-5`; constraint/contact positions use `2e-3 m`
  and velocities `2e-2 m/s`; deformable samples use `5e-3 m` and `5e-2 m/s`.
  Quaternion angular error uses `2e-3 rad`, with opposite signs equivalent.
- Contact lists are canonicalized and rigid manifolds aligned by geometric
  cost within body pairs. Final comparison requires a one-to-one match within
  stable body/particle identities, with every matched field satisfying its
  existing tolerance. Floating-point noise cannot reorder the correspondence.
  NaN, infinity, missing entities, changed topology, and changed case hashes fail.
- Systems of at most 256 elements emit complete state. Larger systems emit
  stable-ID samples plus counts, bounds, center of mass, momentum, energy,
  maximum speed, divergence, strain, clearance/contact, and topology data.
- Cases marked `chaotic_envelope` retain particle samples, rigid trajectories,
  and raw contact records for inspection, but those individual trajectories do
  not gate. Aggregate bounds/centres gate at `0.1 m + 5%`, momentum at
  `25 + 25%`, and energy at `25 + 25%`. Contact counts use a symmetric
  `0.05 + 10%` tolerance. Divergence, pressure residual, and strain are
  upper-bound envelopes; minimum clearance is a lower-bound envelope, with the
  same tolerance. Improving those safety metrics never fails conformance.
  Identities, lifecycle counts, and topology remain exact.

CTest byte-checks registry serialization and GLB hashes and exercises
comparator edge cases without rewriting the committed goldens.

## NVIDIA reference handoff

Available captures:
[2026-10-06 CUDA diagnostic package](references/cuda-20261006/README.md).
Its strict repeatability limitations are recorded alongside the complete data.

`capture_reference.py` accepts an external command-API conformance runner and
retains ten complete runs, raw timings, device and source provenance, build
options, per-file hashes, and the first divergent record. It refuses dirty
source or a runner configured for another commit. Output must be a new
directory outside the source worktree. It never updates goldens or relaxes
tolerances.

```bash
python3 conformance/v1/capture_reference.py \
  --runner /path/to/build/parallel-mater-conformance \
  --output /path/to/new-cuda-reference
```

An independently maintained diagnostic capture tool may be supplied with
`--parity-runner`; the retired in-tree capture harness is no longer built.

Only `diagnostics.timings` is excluded from exact same-device repeatability;
every checkpoint, discrete value, ordering, state hash and provenance field
remains exact. All raw runs survive a failure. `--allow-dirty` is available
for investigation only and always produces an unqualified package.

Golden updates require a reviewed command-API runner and an explicit complete
corpus capture. Inspect all numerical diffs and rerun the complete CTest suite.
Retain its log and lossless gallery images alongside the package. Do not call
a non-repeatable capture a golden.

A future adapter must run this same case registry through the shared command
API and compare the complete result directory. First-divergence instrumentation
belongs in that driver path; backend-specific physics APIs, capture executables,
and duplicate scene runners are not part of the canonical conformance contract.
