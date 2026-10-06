# CUDA reference requirements for Metal parity

The CUDA implementation is the behavioral reference for the Metal port. The
Metal work can continue on Apple Silicon, but the remaining engine differences
cannot be classified reliably without a CUDA capture produced from the same
source and assets. This document defines the reference package needed from an
NVIDIA machine.

## Available NVIDIA captures

The [2026-10-06 CUDA physics package](../conformance/v1/references/cuda-20261006/README.md)
contains all 30 cases from ten runs of clean source, plus the shared all-system
capture, toolchain provenance, and first-divergence reports. It is usable for
Metal numerical investigation now, but is explicitly **not a golden refresh**:
five coupled cases fail exact same-device repeatability, although every repeat
passes the existing cross-backend tolerance comparator. Do not treat that
known CUDA variability as a Metal-only failure.

## Required baseline

Build and capture the exact commit under review after it is available on the
NVIDIA machine. Do not reuse results from another source revision, modified
GLBs, or a dirty worktree. Record:

- the full Git commit and confirmation that the worktree is clean;
- GPU name, compute capability, CUDA driver/runtime, host compiler, CMake, and
  build type;
- every non-default simulation, gallery, and deterministic-mode option;
- the complete CTest log and any CUDA, OptiX, or sanitizer validation output.

The committed `conformance/v1/manifest.json` currently identifies results made
at source commit `4fa2b7e`. Its Hinge and Piston asset pins have been advanced
to the current GLBs so input validation remains useful, but those two committed
result files were not produced from those assets. Treat the present CUDA corpus
as historical until it is regenerated as one reviewed set.

## First priority: fresh conformance corpus

On the NVIDIA host:

```bash
git checkout <metal-port-review-commit>
git status --short
cmake -S . -B build-cuda-reference \
  -DCMAKE_BUILD_TYPE=Release \
  -DPARALLEL_MATER_BUILD_CUDA=ON \
  -DPARALLEL_MATER_BUILD_METAL=OFF \
  -DPARALLEL_MATER_BUILD_OPTIX_GALLERY=ON \
  -DBUILD_TESTING=ON
cmake --build build-cuda-reference --parallel
./build-cuda-reference/parallel-mater-conformance --check-inputs
./build-cuda-reference/parallel-mater-conformance --update-goldens
ctest --test-dir build-cuda-reference --output-on-failure
```

Return the reviewed diff for all of `conformance/v1/golden/cuda/` and
`conformance/v1/manifest.json`, not a subset of passing cases. The manifest
must name the capture commit and actual device/toolchain. The current 30 cases
cover the engine systems and pairwise couplings; refreshing the entire corpus
keeps case hashes, lifecycle counts, topology, and floating checkpoints tied to
one source revision.

Run the corpus at least ten times on the same GPU and toolchain. Counts, IDs,
flags, generations, topology, stable ordering, and deterministic CUDA output
must be byte-identical. If CUDA itself is not repeatable, retain all differing
captures and the first divergent record rather than selecting one as golden.

## Differential checkpoints needed

For a case that still differs after the refresh, the most useful CUDA data is a
stable-ID dump immediately before and after the first divergent engine phase.
Add temporary reference instrumentation only when the public conformance
checkpoint does not isolate that phase. Capture, in execution order:

1. frame/substep inputs and old/current transforms;
2. integration and swept bounds;
3. broad-phase pairs, BVH leaf pairs, manifold candidates, and selected
   contacts;
4. contact colors and every solver iteration's accumulated impulse/correction;
5. constraint Jacobians, effective masses, limits, motors, breakage, and
   compound membership;
6. particle sort keys/ranges, neighbor or surface contacts, contribution
   buffers, and reduced reactions;
7. deformable projection passes, tear decisions, rebuilt topology, and
   lifecycle compaction;
8. final public views, events, statistics, and debug contacts.

Each record needs the frame, substep, phase, stable object IDs, source indices,
units, and exact CUDA values. Buffer-layout dumps (size, alignment, field
offsets, and element stride) are also required when the first difference is at
upload or readback. A final-state screenshot is not enough to locate an engine
difference.

Work the refreshed failures in this order:

1. rigid contacts and constraints, including the revised Hinge/Piston scenes;
2. cloth tearing, especially the frame of each topology/event transition;
3. rope core and rope/soft-body endpoint/contact coupling;
4. remaining fluid, smoke, paint, and cross-system differences;
5. rendering after the underlying simulation checkpoints agree.

## Gallery reference package

The CUDA OptiX gallery remains the visual and control reference. From the same
commit, run its headless CTest coverage and retain the lossless PPM outputs:

```bash
ctest --test-dir build-cuda-reference --output-on-failure \
  -R 'parallel-mater-.*(headless|render).*test'
```

For every entry in the shared 29-scene registry, provide the exact selector,
frame number, resolution, camera state, particle-count override, control input
timeline, debug mode, and output image. Include reference captures for:

- the normal beauty view and applicable `V/Z/X/C/B/N` debug views;
- pre-action and post-action constraint states;
- authored gravity/steering and count-dialog rebuild paths;
- implicit water, refraction/reflection, transparent skins, paint, foam,
  particle/grid smoke, deformable normals, contacts, and force overlays.

Also retain the OptiX validation log and a GPU capture for representative rigid,
fluid-rigid, cloth-tear, rope-soft-body, and grid-smoke scenes. These images are
the inputs for the planned SSIM and mean-channel-error gates; Metal smoke-test
captures are not approved visual references.

## Performance reference

Record CUDA per-phase timings, launch counts, scene settings, warm-up length,
and sample count on the named NVIDIA system. Use these to find unexpected work
or phase-count differences, not as an FPS target for Apple hardware. Metal
regressions will be gated against per-chip Metal baselines once correctness is
established.

## What can proceed without the NVIDIA host

Metal API/layout tests, Metal validation, same-device deterministic replay,
gallery controls, and isolated solver tests can continue on the Mac. Without
the package above, the port cannot authoritatively refresh CUDA goldens, find
the first divergent CUDA phase, approve visual references, or claim numerical
or rendering parity.
