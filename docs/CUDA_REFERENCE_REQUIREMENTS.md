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

## Current Metal comparison status

The 2026-10-06 package has been verified and compared on Apple Silicon against
`run-01`. On unmodified merged source, Metal passes 13 of the 30 v1 cases. The
current Metal conformance branch ports the CUDA convex/concave entry-side
contact rule, small closed-convex face manifolds, the missing piston alignment
row, reduced-coordinate slider/piston integration, and CUDA-style persistent
contact initialization, warm starting, accumulated impulses, and cache
write-back. Guided static mechanisms now use CUDA's entry-only sweep, stable
face projection, 0.1 mm rest offset, guide-space normal selection, and
projected conservative advancement. Dense mesh pairs also honor CUDA's
512-entry leaf-pair cache limit before switching to serial BVH traversal.
Motor-driven contacts also resolve nearly parallel triangle normals
to the authored collider plane, removing backend-local tangent noise without
changing the manifold. Convex face patches discard numerically near-collinear
triangle-seam samples while retaining CUDA's full eight-contact capacity. With
those changes, these 20 cases pass:

- `cloth-core` and `cloth-water`;
- `constraint-breaking`, `constraint-motor`, `constraint-piston`, and
  `constraint-slider`;
- `compound-weld-lifecycle`;
- `fluid-lifecycle`, `fluid-rigid`, `rope-cloth`, and `rope-fluid`;
- `smoke-cloth`, `smoke-grid`, `smoke-rope`, `smoke-soft-body`, and
  `smoke-water`;
- `soft-body-cloth`, `soft-body-core`, `soft-body-fluid`, and
  `soft-body-rigid`.

The 10 remaining cases and their current comparator difference counts are:

| Area | Cases |
| --- | --- |
| Rigid lifecycle/contact | `passive-active` (49), `rigid-direct` (13) |
| Constraints | `constraint-fixed` (7), `constraint-generic-spring` (17), `constraint-generic` (11), `constraint-hinge` (2), `constraint-point` (22) |
| Cloth/rope | `cloth-tear` (62), `rope-core` (177), `rope-soft-body` (12) |

These counts describe comparison records, not necessarily independent bugs.
For example, one earlier contact-manifold difference can alter every later
state and event record. Before accepting a change, rerun all 30 cases so a
local improvement does not hide a cross-system regression.

## CUDA engine traces still needed

The public checkpoints identify which scenarios differ, but they do not expose
the first engine phase that differs. The next CUDA handoff should be generated
from the exact source commit under review and include the following focused
traces. A small machine-readable JSON or binary dump is preferable to console
logging; retain the dumping code or patch so Metal can emit the identical
schema.

### Rigid contacts and lifecycle

For `passive-active`, `rigid-direct`, and the first failing frame of each
constraint case, capture:

- previous and predicted body transforms and velocities;
- broad-phase body pairs and BVH leaf/triangle pairs in stable order;
- convex solid planes, selected reference and incident faces, clipped polygon
  vertices, separation/depth, normal, and the final reduced manifold;
- contact keys/colors and each solver iteration's normal/friction impulses;
- compound root/member remapping before and after lifecycle compaction.

This is the first priority because Metal now generates CUDA-style small-convex
face manifolds and implements persistent contact state plus substep-local cache
reuse, but the public checkpoints still cannot distinguish geometry drift from
solver drift. The cache work makes `constraint-breaking` exact, and the
combined contact ports reduce `constraint-generic` from 33 to 11 records and
`constraint-hinge` from 27 records to one contact-count/list mismatch.
`compound-weld-lifecycle` is now exact: Metal removes only near-collinear
triangle-seam samples from a clipped convex patch, yielding CUDA's four patch
corners without imposing a blanket four-contact cap. No additional compound
trace is required unless a future CUDA capture changes that case.

The remaining `constraint-hinge` mismatch is isolated more narrowly than the
public count suggests. At frame 90, the `Gear.001`/`Gear.002` pair has 697
overlapping BVH leaf pairs, so both backends cross CUDA's 512-entry cache limit.
CUDA's final manifold contains contacts at z = 2.02543545, 2.40554595, and
2.34251475 m; Metal's serial fallback retains the first two. The omitted CUDA
contact has zero normal/friction impulse, while all rigid states pass tolerance.
For this pair, capture the cache-overflow flag, every serial BVH stack push and
pop, leaf and reordered triangle IDs, closest points, acceptance/rejection
reason, and the manifold immediately after every `add_manifold_contact` call.
The trace must cover the final non-empty substep at frames 45 and 90. This is
the shortest reference artifact that can identify the remaining geometry-order
difference without perturbing solver behavior.

`constraint-motor` is now exact. For motor-driven bodies, Metal replaces only
a triangle contact normal already parallel to an authored convex collider plane
with that plane's transformed normal. This preserves the contact position,
depth, and ordering while eliminating a roughly 0.002 tangent component caused
by backend-local triangle arithmetic. No additional motor trace is currently
required unless a future CUDA capture changes that case.

For `constraint-generic`, `passive-active`, and `rigid-direct`, capture each
contact's cache key, cache hit/miss, impact fraction, initial normal speed,
persistent/face-patch flags, accumulated normal and friction impulses,
response-patch membership, and cache contents before load and after store.
Include the state immediately before contact
initialization, after warm start, after every velocity and position sweep, and
after cache write-back. The trace must say whether the face path, triangle
path, or swept path selected the final contact; final contact events alone
cannot distinguish them.

Cross-frame cache reuse is not enabled yet. A direct lifetime experiment
reopened `fluid-rigid` and increased `constraint-generic` from 13 to 22
records, while closing no case. The trace therefore needs to identify CUDA's
first cross-frame cache match and the resulting warm-start impulse, not merely
the final cached values.

For a guided or constrained body, include the complete contact frame used by
CUDA: fixed, axial, axial-rotation, fixed-member, and static-body flags; body
reference point; axis; projected point velocity; directional inverse mass;
and the exact impulse applied to every aggregate member. Metal now mirrors
those contact-frame fields, so these records are needed to determine whether
the first mismatch is geometry, frame construction, or constrained impulse
response.

### Constraint solve

For each failing constraint case, dump the prepared world anchors and frames,
all linear/angular rows, effective mass, bias/error, limits, motor target,
accumulated impulse, break decision, and body velocities after every solver
iteration. For `constraint-hinge`, also include its alignment error and final
hinge correction. Slider and piston traces are no longer required unless a
future CUDA capture reopens those exact cases. Stable body and constraint IDs
are required.

### Particle and deformable systems

- `cloth-tear`: record the constraint strain values, eligible tear keys,
  deterministic selection order, emitted events, vertex duplication, rebuilt
  indices, and generations at the first topology change.
- `rope-core` and `rope-soft-body`: record endpoint constraints, segment
  projection rows, BVH triangle visitation order, collision candidates,
  selected triangle/fraction, per-iteration corrections, and the
  contribution/reduction inputs that update the rope and soft body. Metal now
  mirrors CUDA's right-first BVH leaf order and exact rigid orientation update;
  the trace should begin before the first frame-36 `rope-core` orientation and
  velocity mismatch and before the frame-24 `rope-soft-body` maximum-speed
  mismatch.

### Trace contract

Every record must name the case, frame, substep, phase, solver iteration,
stable object IDs, source buffer indices, units, and exact bit pattern of each
floating value. Include count/capacity, element stride, structure size,
alignment, and field offsets for every dumped buffer. Sort only by the same
stable key used by the engine; do not sort dumps afterward to conceal ordering
differences.

No wider tolerance or additional final-state screenshot is needed at this
stage. The useful deliverable is one CUDA trace immediately before and after
the first divergent phase, plus the same-schema Metal trace. Once those agree,
the existing public checkpoint comparator remains the acceptance gate.

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
