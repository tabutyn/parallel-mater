# Parallel Mater physics conformance v1

This directory is the backend-neutral correctness contract for CUDA, Metal,
and future implementations. It is separate from physics capture v1, which
remains a debugging/replay format.

## Layout

- `cases/` contains canonical, byte-stable JSON inputs in SI units. Analytic
  inputs are serialized from the CUDA-free C++ case registry. Integrated
  inputs reference committed GLBs; `manifest.json` pins every asset by SHA-256.
- `golden/cuda/` contains one CUDA result per case. CUDA is the v1 reference,
  but matching is tolerance-based rather than bit-identical.
- `compare.py` compares CUDA, Metal, or another backend's result directory and
  writes an optional machine-readable report.
- `manifest.json` records the exact source revision, deterministic settings,
  toolchain/device provenance, and GLB hashes used for the reviewed goldens.

The 30-case registry covers direct rigid integration and lifecycle changes, compound
weld release/rebuild, breaking, all eight constraint types, fluids, cloth,
soft bodies, ropes, smoke, and every implemented pairwise coupling. The
required integrated scenes (`PassiveActive`, `ConstraintFixed`, `FluidRigid`,
`ClothWater`, `SoftbodyRigidBody`, `SmokeWater`, and `SmokeRope`) are included
alongside the remaining coupling and subsystem scenes.

## CUDA runner

Build with `PARALLEL_MATER_BUILD_OPTIX_GALLERY=ON`, then use:

```text
parallel-mater-conformance --list
parallel-mater-conformance --check-inputs
parallel-mater-conformance --provenance
parallel-mater-conformance --case <id|all> --output <directory>
parallel-mater-conformance --update-goldens
```

`--update-goldens` is the only command that writes `golden/cuda/` or the
manifest. It always refreshes the complete corpus. Reconfigure CMake after the
implementation commit and before refreshing so `source_commit` identifies the
code that produced the results. Golden and provenance diffs must be reviewed;
the command never commits them or changes tolerances.

To compare another backend:

```text
python3 conformance/v1/compare.py \
  --cases conformance/v1/cases \
  --expected conformance/v1/golden/cuda \
  --actual /path/to/metal-results \
  --report /path/to/report.json
```

Metal emits the same `parallel-mater-conformance-result/v1` schema and keeps
the canonical case SHA-256 in each result. Backend identity, timings, and
backend-specific hashes are diagnostic and never gate correctness.

## Matching rules

- IDs, counts, presence, enabled/broken state, and topology are exact.
- Finite floating values pass when
  `abs(error) <= abs_tolerance + rel_tolerance * max(abs(expected), abs(actual))`.
- Direct integration uses `1e-5`; constraint/contact positions use `2e-3 m`
  and velocities `2e-2 m/s`; deformable samples use `5e-3 m` and `5e-2 m/s`.
  Quaternion angular error uses `2e-3 rad`, with opposite signs equivalent.
- Contact lists are canonicalized. Rigid manifold samples are grouped by body
  pair and matched by minimum geometric cost so harmless floating-point noise
  cannot reorder otherwise equivalent contacts. NaN, infinity, missing
  entities, changed topology, and a changed case hash fail.
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

CTest byte-checks registry serialization and GLB hashes, exercises comparator
edge cases, and reproduces CUDA results in a temporary directory without
rewriting the committed goldens.

## NVIDIA reference handoff

Available captures:
[2026-10-06 CUDA diagnostic package](references/cuda-20261006/README.md).
Its strict repeatability limitations are recorded alongside the complete data.

`capture_reference.py` retains ten complete runs, raw timings, device and
source provenance, build options, per-file hashes, and the first divergent
record. It refuses dirty source or a runner configured for another commit.
Output must be a new directory outside the source worktree. It never updates
goldens or relaxes tolerances.

```bash
python3 conformance/v1/capture_reference.py \
  --runner /path/to/build/parallel-mater-conformance \
  --parity-runner /path/to/build/parallel-mater-cuda-parity-capture \
  --output /path/to/new-cuda-reference
```

Only `diagnostics.timings` is excluded from exact same-device repeatability;
every checkpoint, discrete value, ordering, state hash and provenance field
remains exact. All raw runs survive a failure. `--allow-dirty` is available
for investigation only and always produces an unqualified package.

After qualification, explicitly run `--update-goldens`, inspect all numerical
diffs, and rerun the complete CTest suite. Retain its log and lossless gallery
images alongside the package. Do not call a non-repeatable capture a golden.

The CUDA and Metal all-system capture executables share
`tests/parity/capture_scenario.hpp`. This is Metal's detailed public-view
diagnostic format, not a replacement for the 30-case conformance contract:

```bash
build-metal/parallel-mater-metal-parity-capture /tmp/metal.capture
build-metal/parallel-mater-parity-compare \
  /path/to/new-cuda-reference/all-systems-01.capture /tmp/metal.capture
build-metal/parallel-mater-metal-conformance --case all --output /tmp/metal-results
python3 conformance/v1/compare.py --cases conformance/v1/cases \
  --expected conformance/v1/golden/cuda --actual /tmp/metal-results \
  --report /tmp/metal-report.json
```

See [CUDA reference requirements](../../docs/CUDA_REFERENCE_REQUIREMENTS.md)
for conditional first-divergent-phase instrumentation and the separate visual
and performance handoffs. A CUDA reference alone does not establish Metal
parity.
