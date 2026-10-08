# CUDA physics handoff — 2026-10-06

These are **diagnostic captures, not newly approved goldens**. They contain
real CUDA kernel results for every v1 case, produced from clean source commit
`cc2a2e40656c2a850176d97220d7c32d6da6bf2d` after merging the Metal backend.
That commit also fixes inward contact recovery against concave rigid surfaces.

Device: NVIDIA GeForce RTX 3050 Ti Laptop GPU, compute capability 8.6;
CUDA compiler 13.1.80, runtime/driver API version 13010; GCC 12.2.0;
Release, CUDA ON, Metal OFF, OptiX gallery ON, testing ON, architecture 86.
The archive includes the actual CMake cache and NVIDIA driver report.

## Use on the Mac

Check out this branch (or its merged descendant), build the Metal backend, then:

```bash
mkdir -p /tmp/cuda-reference-20261006
tar -xzf conformance/v1/references/cuda-20261006/physics-runs.tar.gz \
  -C /tmp/cuda-reference-20261006
build-metal/parallel-mater-metal-conformance --check-inputs
build-metal/parallel-mater-metal-conformance --case all --output /tmp/metal-results
python3 conformance/v1/compare.py --cases conformance/v1/cases \
  --expected /tmp/cuda-reference-20261006/run-01 --actual /tmp/metal-results \
  --report /tmp/metal-vs-cuda.json
build-metal/parallel-mater-metal-parity-capture /tmp/metal.capture
build-metal/parallel-mater-parity-compare \
  /tmp/cuda-reference-20261006/all-systems-01.capture /tmp/metal.capture
```

`run-01` is simply the first capture, **not a selected best result**. For any
of the five non-repeatable cases, compare against all ten retained runs before
classifying a discrepancy as Metal-specific. Do not loosen tolerances to match.

## Repeatability findings

The archive contains ten complete 30-case runs, their raw timings and hashes,
ten all-system captures, canonical inputs, GLB SHA-256 pins, input validation,
and comparisons of runs 02–10 with run 01. Its diagnostic `manifest.json`
records the actual capture revision, independently of the historical golden
manifest. See the reviewable `repeatability.json` beside the archive.

- 25/30 cases are exactly repeatable after excluding only diagnostic timings.
- Five differ: `cloth-water`, `smoke-cloth`, `smoke-rope`,
  `smoke-soft-body`, and `smoke-water`.
- All nine cross-run comparisons pass all 30 existing tolerance profiles.
- All-system CUDA captures contain 23,347 records and are byte-identical over
  ten runs. Both backends now use the same scenario and serializer.
- First recorded corpus divergence: cloth-water frame 24, combined X momentum,
  -3.045713424682617 versus -3.3850932121276855 kg m/s.
- Clean source and unchanged source revision were verified before and after
  capture. No inputs, tolerances, or golden results were rewritten.

The five varying cases prevent the strict reference qualification requested in
`docs/CUDA_REFERENCE_REQUIREMENTS.md`. Floating atomic reductions in
`src/fluid_cloth.cuh` and `src/smoke_grid.cuh` are investigation candidates,
not a proven complete diagnosis. Preserve this evidence while isolating their
first divergent phase; do not attribute these variations to Metal.

The historical golden corpus is still historical: 18 cases match this capture
and 12 differ under today's comparator. The old Hinge/Piston results were
already documented as stale. The historical comparison is retained for review;
it is not justification for silently blessing all changed results.

A separate fresh capture of unchanged `origin/main` (`597626b`) matches
27/30 final cases. The contact fix changes `constraint-point`,
`passive-active`, and `smoke-water` beyond their previous checkpoints'
tolerances; `main-comparison.json` records every difference. This is additional
review evidence, not an assertion that unchanged main is an approved golden.

## Contact regression

The new authored-tray fixture fails against unchanged main at frame 18 with
25.7 mm measured penetration. With the final fix, all six combinations of
body ordering and 4/8/16 substeps have zero sampled penetration and zero later
energy gain. Hinge/slider, the eight-right-left-cycle piston test, and rope
settling/winding checks pass. The fix is limited to intersecting triangles;
separated surface contacts retain their established behavior.

The Metal physics kernels were not changed here. Port the intersecting
convex/concave contact logic in `src/geometry_constraints.cuh` before
classifying those changed contact trajectories as unrelated Metal defects.
The essential inputs are both bodies' previous transforms, the concave face,
and its signed separation from the incident convex triangle. Do not substitute
the concave body's origin for the face's entry side.

## Scope

Final full Release CUDA/OptiX CTest run: **99/101 passed**, no skipped tests.
Failures are `parallel-mater-conformance-cuda-goldens` (historical reference
drift described above) and `parallel-mater-rigid-wall-tests` (also fails on
unchanged main). All headless/render tests pass; 40 lossless PPM images were
included in `gallery-ppm.tar.gz`. `validation/` contains the full-suite CTest log,
JUnit report, exact test commands, image hashes, build output, baseline checks,
and the failing-before/passing-after Generic regression evidence.

Verify archive checksums with `shasum -a 256 -c SHA256SUMS` on macOS.
The gallery archive covers the existing CTest commands, not the complete
requested matrix of all scene/debug/control combinations. It is supplemental
evidence, not visual-parity approval.

This package unblocks numerical inspection and tolerance comparisons.
It does not assert Metal parity, exact CUDA repeatability for all systems, or
approval of the historical goldens. Engine-internal phase tracing is the next
step for failures that public checkpoints cannot isolate. Gallery debug-view,
OptiX GPU-capture, and performance handoffs remain separate requirements.
