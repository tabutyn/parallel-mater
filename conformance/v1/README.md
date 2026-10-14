# Parallel Mater physics conformance v1

This directory is the backend-neutral correctness contract for the qualified
CUDA driver and future Slang device adapters. It is separate from physics
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
- `manifest.json` records the exact source revision, deterministic settings,
  toolchain/device provenance, and GLB hashes used for the reviewed goldens.

The 30-case registry covers direct rigid integration and lifecycle changes, compound
weld release/rebuild, breaking, all eight constraint types, fluids, cloth,
soft bodies, ropes, smoke, and every implemented pairwise coupling. The
required integrated scenes (`PassiveActive`, `ConstraintFixed`, `FluidRigid`,
`ClothWater`, `SoftbodyRigidBody`, `SmokeWater`, and `SmokeRope`) are included
alongside the remaining coupling and subsystem scenes.

## Validation tools

The former CUDA runner directly called the removed C++ `World` interface and
is not part of the command-only runtime. The repository currently validates
the canonical inputs and comparator with:

```text
ctest --test-dir <build> -R parallel-mater-conformance
```

Driver qualification must execute these cases through the public `pm*`
command API and emit `parallel-mater-conformance-result/v1`. Compare that
output with:

```text
python3 conformance/v1/compare.py \
  --cases conformance/v1/cases \
  --expected conformance/v1/golden/cuda \
  --actual /path/to/driver-results \
  --report /path/to/report.json
```

Every driver result keeps the canonical case SHA-256. Driver identity,
timings, and implementation-specific hashes are diagnostic and never gate
correctness.

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
