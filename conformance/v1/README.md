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
- `lifecycle.py` evaluates adverse capacity recovery and mixed-family graph
  rollback without a reference backend. The immutable profile is in
  `lifecycle/adverse-capacity-three-family-v1.json`.
- `qualification.py` maps each advertised interaction record to revisioned
  evidence obligations in `qualification/capability-obligations-v1.json`.
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

## Validation tools

`parallel-mater-conformance` loads every canonical case through the shared
scene loader and executes it through the public `pm*` command API. It does not
link a backend-specific simulation interface. Validate the canonical inputs
and comparator with:

```text
ctest --test-dir <build> -R parallel-mater-conformance
```

Record the selected driver's public identity and exact advertised interaction
keys with `--provenance` and `--capabilities`. The capability output uses the
same canonical names and graph-limit representation as qualification results.

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
result repeats the fixed `RigidBody.glb`, eight-brick, 300-frame, four-substep
configuration and reports the maximum pairwise penetration observed by the
independent closed-mesh sampler in `tests/gallery/rigid_mesh_checks.hpp`.
Evaluate a captured result with:

```text
python3 conformance/v1/quality.py \
  --profile conformance/v1/quality/rigid-brick-penetration-v1.json \
  --actual /path/to/rigid-brick-result.json \
  --report /path/to/rigid-brick-report.json
```

The existing 2 mm threshold is unchanged. A driver-contact depth is diagnostic;
it cannot replace the independent mesh measurement. The original CUDA/D3D12
failures and corrected observations remain in the revisioned Windows report.

Run the capacity and graph-lifecycle gate with:

```text
python3 conformance/v1/lifecycle.py \
  --actual /path/to/adverse-capacity-three-family-v1.json \
  --report /path/to/lifecycle-report.json
```

The result records canonical public resource-state SHA-256 fingerprints and
harness write ordinals at every required checkpoint; ordinals are never private
driver handles. Each command completes and state is read before any physics
step. Replacement must advance the target ordinal and change its observable
state. A capacity failure may retire its target, but
must preserve an unrelated fluid before recovery. The mixed graph contains a
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
selected CUDA/D3D12 cases with zero unsupported results. Independent eight-brick,
four-substep, 300-frame mesh checks pass on both drivers at 0.205278 mm against
the unchanged 2 mm limit. The report retains original divergences, source/case/
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
