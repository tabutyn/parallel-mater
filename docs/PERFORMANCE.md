# Rigid contact performance, 2026-09-21

## Soft Body Fluid (PR 15, 2026-09-28)

### Initial coarse surface (superseded by refinement below)

Local Release measurements on the RTX 3050 Ti Laptop GPU, 1/60 s frames,
four substeps and the shared fluid's two iterations per substep:

- The original thin-axis export resolution generated 16,189 nodes and
  273,227 bonds for the authored slab. Aspect-ratio-aware sampling produces
  883 nodes and 11,884 bonds, retaining all four Goal pins and the authored
  triangle surface. The `.blend` is unchanged; spacing remains overridable.
- An early independent-reaction prototype reached 43.36 m/s at soft nodes.
  Rejected. Shared contact-degree relaxation and a symmetric velocity-budget
  bound keep the final implementation at its configured 2 m/s ceiling without
  discarding the opposing water impulse.
- 600 frames, 4,000 particle capacity: approximately 10.04 ms GPU/frame,
  3.89 ms coupling, versus 21.14 ms / 6.27 ms with the oversampled lattice.
  Four pins remain exact, no inside-skin particles in the every-ten-frame
  winding checks, and 21,426 particles leave through the authored outflow.
- 1,200 frames, 30,000 capacity, alternating 45-degree gravity tilt after
  frame 240: 11.85 ms GPU/frame, 4.90 ms coupling, 12,023 live particles at
  the end, 35,977 outflow removals, zero pin drift and zero sampled inside
  particles. This measures physics, not rendering/capture/readback wall time.
  The finite outflow only removes particles crossing its authored rectangle;
  water spilling outside it under tilt remains live until reset/capacity.
- API fixtures test a fast particle crossing a thin closed surface, reversed
  winding, pinned and free nodes, duplicated binding influences, and 64
  simultaneous impacts. Maximum paired-impulse error was 1.42e-11 in the dense
  case. A zero-gravity fixture verifies that water itself moves the soft body.

All 49 CTest cases passed, including the real Blender exporter and headless
scene. Focused API tests passed CUDA memcheck with zero errors. This is not a
full-scene sanitizer claim. The gallery reuses its existing renderer and foam;
the default-capacity 240-frame render and opt-in physics capture were inspected.
Contact diagnostics report proposal counts and maximum pre-recovery depth;
they are not measurements of residual penetration.

### Lattice-resolution surface support

The earlier surface only simulated the 16 authored corners. The shared API
now refines oversized faces, preserving the closed shape and rendering seams,
and connects all new surface nodes into the volume. Surface edges are limited
to `1.5 * spacing` to allow triangle diagonals; existing lattice-scale spheres
are not unnecessarily subdivided. Sorted spring insertion preserves their
previous accumulation order. Before material tuning, the slab had 2,104 nodes, 41,990 bonds,
2,488 skin triangles (formerly 28), and 27 exact pins covering the authored
attachment face.

An initially over-dense 3,090-node version measured 161.04 ms/frame at 120 frames,
149.12 ms in fluid coupling: every nearby particle scanned every triangle.
A refitted swept triangle BVH and accelerated winding query reduced that same
mesh to 32.09 ms/frame, 20.25 ms coupling, with zero sampled inside particles
and zero pin drift. The final resolution avoids that unnecessary over-density.
These are short, 4,000-capacity physics-only measurements, not rendering times.
At the final resolution, the same 120-frame run measured 20.42 ms/frame,
12.69 ms coupling. The firmer material below measured 40.86 ms/frame,
17.55 ms coupling; both retained exact pins and zero sampled inside particles.

Stiffness is independent of mesh support: the final slab's maximum displacement
after four dry seconds was 1.79 m at defaults, 0.65 m with shape matching 0.35
and 32 iterations, and 0.38 m with shape matching 0.5 and 64 iterations. The
source material settings were unchanged in that comparison. See `BLENDER_SCENES.md` for controls
that are actually exported; native Blender Pull/Push/Bending are not mapped.

All 50 CTest cases pass, including unchanged rigid-containment and cloth
tear-through limits. CUDA memcheck reports zero errors in the coupling API
fixtures, including a 1,536-triangle pinned-skin recovery case. This is not a
full-gallery sanitizer result.

### Coarser, firmer authored material

`SoftbodyFluid.blend` now authors spacing 0.12, shape recovery 1.0, 64 graph
iterations, projection fraction 0.5 and zero stretch compliance. Its original
geometry, mass, Goal weights and gravity are preserved; no solver specialization
is involved. The export produces 777 nodes, 14,400 bonds, 944 surface triangles
and 19 exact pins (63% fewer nodes than the 2,104-node mesh).

Ten-second dry comparisons at recovery 0.8 and 32 iterations gave 11.8 cm sag
at spacing 0.12 (777 nodes), 19.0 cm at 0.16 (420 nodes), and 34.6 cm at 0.20
(219 nodes). The larger spacings lost too much thickness/attachment support.
At spacing 0.12 the selected firmer settings reduced final dry sag to 8.0 cm.
The 600-frame water run measured 45.7 cm maximum displacement, zero pin drift,
zero sampled inside particles, 21,286 outflow removals and 30.46 ms GPU/frame
(10.60 ms coupling), with 4,000 particle capacity. It still bends under load;
it is not made static. Timings are physics only on the same RTX 3050 Ti.

All six material-related regressions pass (geometry, coupling API, Blender
export, 600-frame dry sag, 600-frame water loading, and headless rendering).
The dry case includes the initial oscillation: peak sag about 14.5 cm, settling
to about 8 cm. A 1,200-frame alternating 45-degree gravity stress run also kept
pins exact and found no sampled inside particles (maximum displacement 46 cm).

## Occupancy-driven mesh inflow (PR 15)

The authored 0.7303 m-square SoftbodyFluid emitter at initial velocity -1 m/s
previously injected 2,400 particles/s regardless of occupancy. At frame 180,
298 particles were above its surface. API surface sampling now produces 20
sites at the default 0.18 m clearance. Over the same three seconds it emitted
520 particles, retained 129, and had zero particles above the source at every
frame. Capacity was 4,000; gravity, pressure, and collision kernels were unchanged.

A force-free tilted-triangle regression emitted 21 stationary particles and
stopped, including with an overlapping second source. At 1 m/s it emitted 75
particles over one second; at 2 m/s, 132. This confirms velocity-driven
throughput without a fixed rate or accumulated backlog. The per-step occupancy
index is GPU sorted; the deterministic commit checks only earlier sources, not
every pair within one already-separated source. All storage is preallocated.

## Original rigid-contact investigation

These are local engineering measurements, not general CUDA or hardware
claims. The objective was to remove the observed 20+ ms rigid-contact frame
without weakening determinism, containment, or authored geometry contracts.

## Environment and method

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, compute capability 8.6
- CUDA toolkit: 13.1
- build: `Release`, `CMAKE_CUDA_ARCHITECTURES=86`
- scene: committed `examples/assets/PassiveActive.glb`, five rigid bodies
- frame: `1/60 s`, four substeps
- sampling: settle 360 frames, warm 10, measure 120
- timing: CUDA events requested through `WorldStepTimings`; no external profiler

Build and reproduce the retained scene benchmark with:

```bash
cmake -S . -B build-gallery \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPARALLEL_MATER_BUILD_OPTIX_GALLERY=ON \
  -DPARALLEL_MATER_BUILD_BENCHMARKS=ON \
  -DBUILD_TESTING=ON
cmake --build build-gallery -j
./build-gallery/parallel-mater-rigid-scene-benchmark \
  examples/assets/PassiveActive.glb
```

## Result

The original serial BVH/contact path measured **24.100 ms median GPU time**
and **24.132 ms median wall time**. The retained pipeline measures **1.797 ms
median GPU time** and **1.806 ms median wall time**: about **13.4× faster** for
this scene. All bodies remained in the bowl and repeated settled poses were
bit-identical on the tested GPU.

Final 120-frame distribution:

| Stage | Median | p5 | p95 |
|---|---:|---:|---:|
| Integration | 0.017 ms | 0.014 ms | 0.030 ms |
| World bounds | 0.015 ms | 0.012 ms | 0.020 ms |
| Pair filter | 0.013 ms | 0.011 ms | 0.019 ms |
| Deterministic pair compaction | 0.022 ms | 0.019 ms | 0.032 ms |
| Leaf-pair generation | 0.636 ms | 0.634 ms | 0.641 ms |
| Triangle contact evaluation | 0.905 ms | 0.532 ms | 0.960 ms |
| Contact solve | 0.171 ms | 0.120 ms | 0.196 ms |
| Input clear | 0.003 ms | 0.002 ms | 0.004 ms |
| **GPU total** | **1.797 ms** | **1.365 ms** | **1.873 ms** |
| **Frame wall** | **1.806 ms** | **1.375 ms** | **1.886 ms** |

The live `F` overlay reports these same stages. Rendering is deliberately not
included in `WorldStepTimings`.

## Experiments kept or removed

| Hypothesis | Isolated measurement | Decision |
|---|---:|---|
| Parallel leaf/triangle candidates with fixed-order reduction | Default scene 24.100 → 2.997 ms before later changes; cooperative leaf enumeration subsequently reduced the final pipeline further | Keep |
| Refit a world-space BVH every substep | 2.9973 → 2.9980 ms and added state/kernel work | Remove completely |
| GPU broad phase plus stable active-pair compaction | 64 separated dynamic bodies 15.351 → 0.241 ms; primary scene was neutral | Keep |
| Blender collision proxies | Dynamic-body proxies 3.006 → 2.312 ms while detailed render meshes remained unchanged | Keep |
| Decimate the static bowl collision surface | 3.006 → 4.946 ms and altered support behavior | Remove that proxy; keep the detailed bowl |
| Velocity-gated swept triangle pairs | 120 m/s fixture required 32 discrete substeps at 1.587 ms; conservative sweep contained it in one substep at 0.810 ms | Keep |

The sparse-world broad-phase fixture used 64 identical triangle boxes on a
10-metre grid with zero gravity, four substeps, 10 warmups, and 120 samples.
Its original cost came from the serial solver scanning the complete 64×64
manifold cache eight times. Pair filtering alone was not sufficient; the
solver had to consume the stable compacted pair list.

The swept regression is checked into `tests/rigid_tests.cpp`. A 0.2-metre box
moves 2 metres through a two-sided open plane in one `1/60 s` step. The
conservative triangle-pair path must stop it above the plane and reproduce the
same state across 20 resets.

## Current bottleneck and memory tradeoff

Leaf-pair generation and triangle evaluation now dominate. Candidate order is
deterministic: each body pair uses a cooperative fixed-order leaf scan and a
fixed thread-order manifold reduction. The 512-candidate fast cache costs about
15.8 MiB at the default 64-body capacity. For larger worlds it caches eight
compacted active-pair slots per body rather than every possible pair. Additional active
pairs retain swept-contact correctness through the serial path but can be slower.

## DUMP stress follow-up, 2026-09-23

The 1,000-sphere DUMP scene exposed two correctness bugs and a different
performance profile: the solver ran on one GPU thread, and sphere pairs spent
most of their time in triangle contact evaluation. Contact generation wrote
manifolds at capacity stride while the solver read them at live-count stride;
with spare capacity, bodies could fall through the receiver. Leaf-cache overflow
also lost swept contacts. Both paths now have GPU regressions.

Retained changes: pair-relative swept gating, compact active-pair manifold
indexing, a deterministic conflict-free parallel solver for worlds with at least
128 bodies, conservative swept bounding-sphere rejection, swept-triangle bounds
rejection, and a contact allocation sized for the maximum number of eligible
pairs rather than all directed pairs. Exact authored triangles remain the
collision representation. Parallel diagnostics use stable pair-order event
offsets and the same solver path, so enabling them does not change body states.

Reproduce the stress run with
`./build-gallery/parallel-mater-dump-benchmark 1000 480`. It rotates the hopper
at 45 degrees per simulated second, uses `1/60 s` frames and four substeps, and
prints 30-frame stage averages, memory, and final containment. Rendering is
excluded. This local run used the same RTX 3050 Ti Laptop GPU; an interactive
gallery was also open on that GPU, so times include contention and are not an
isolated throughput claim.

| Configuration | Frames | Triangle evaluation | Solve | GPU total | Allocated |
|---|---:|---:|---:|---:|---:|
| Before stress pass | 31–60 | 319 ms | 673 ms | 997 ms | 340.0 MB |
| Retained changes | 31–60 | 43.2 ms | 12.4 ms | 57.0 ms | 211.9 MB |
| Retained changes | 451–480 | 23.9 ms | 9.7 ms | 34.9 ms | 211.9 MB |

After 480 frames, all 1,000 sphere positions and velocities were finite; none
was below the receiver, and maximum sphere speed was 0.581 m/s. The checked-in
gallery test repeats the 128-sphere parallel solver bit-identically. The rigid
tests exercise spare capacity, a high-speed leaf-cache overflow, and a
high-degree contact graph that exceeds the solver's 32 parallel colors.

An accumulated-impulse variant was rejected: it raised allocation to 258.0 MB,
slightly increased frame time, and ended the same 480-frame fixture at 0.674
m/s maximum speed. Sleeping and a spatial broad phase remain candidates for
separate measured changes; this scene deliberately keeps triangle contacts.

## DUMP triangle-pipeline follow-up, 2026-09-23

With the gallery closed, both configurations were measured on the same RTX
3050 Ti Laptop GPU: 1,000 authored triangle spheres, 480 frames, four
substeps, no rendering. Times below are GPU-event averages over each 30-frame
window. The new configuration was run twice; its final 13-component-per-sphere
state hash matched bit for bit (`4a016b8258d02a31`).

| Frames | Previous GPU total | Retained GPU total | Triangle evaluation, before → after | Contact solve, before → after |
|---|---:|---:|---:|---:|
| 31–60 | 56.99 ms | 20.71 ms (−64%) | 43.11 → 13.12 ms | 12.41 → 6.07 ms |
| 211–240 | 86.29 ms | 24.86 ms (−71%) | 75.86 → 18.52 ms | 9.25 → 5.16 ms |
| 451–480 | 35.34 ms | 15.11 ms (−57%) | 24.20 → 9.25 ms | 9.87 → 4.65 ms |

An intermediate isolated build with the triangle-pipeline changes but the
prior serial greedy color assignment measured 26.65, 28.37, and 20.46 ms in
the same three windows. The parallel matcher reduced those to 20.71, 24.86,
and 15.11 ms; the remaining reduction comes from the triangle-pipeline bundle.
Individual contact-kernel changes were also A/B tested, but several of those
earlier measurements had gallery GPU contention, so they are not used to
assign isolated speedup percentages to each kernel change.

The existing five-body `PassiveActive.glb` benchmark also improved: median GPU
time measured 1.506 ms here versus the previous documented 1.797 ms, with
triangle evaluation 0.652 versus 0.905 ms. This checks that the shared
triangle changes did not trade away the smaller rigid-scene path.

The retained changes split the rare BVH-overflow fallback out of the main
contact-evaluation kernel, use 64-thread contact blocks, tighten the
conservative pair-relative swept-triangle speed bound, and eliminate repeated
segment/triangle distance work. Triangle-pair distance now checks segment
intersections, vertex/triangle distances, and edge/edge distances. A
100,000-pair deterministic random differential check, including near and
degenerate triangles, found a maximum squared-distance discrepancy of
`1.43e-6` versus the prior routine. No analytic sphere or convex contact path
was substituted.

At the time of this measurement, worlds of at least 256 bodies used
deterministic parallel matching; 128–255 bodies used serial greedy coloring,
and smaller worlds used serial contact resolution. These size branches were
subsequently removed in the unified-solver follow-up below. A shuffled,
unique per-pair priority prevented the high-degree graph from collapsing into
the serial overflow fallback. The 480-frame DUMP run ended with zero invalid
states and zero sphere centers outside the receiver; linear RMS speed was
0.161 m/s and maximum speed 0.582 m/s. Allocation remained 211,851,112
bytes. All five tests passed, including high-degree graphs at 128 and 256
bodies, a 256-sphere diagnostic-neutrality test, and high-speed BVH-cache
overflow containment. An additional 900-frame run also ended with zero invalid
states and zero centers outside the receiver; its final maximum speed was
0.508 m/s.

Rejected measured variants: marking collision helpers `__noinline__` raised
stack usage and did not help; shared per-pair transformed vertices did not
improve timing; a 16-color limit offered a small isolated speed gain but risked
serial fallback on higher-degree graphs; ordered-priority parallel matching
left 1,408 of 1,850 contacts in the serial overflow at frame 60; compact
per-color solver worklists increased early total from 20.7 to 22.1 ms, peak
from 25.0 to 25.8 ms, and allocation by about 2 MB. Only the faster and stable
variants were retained.

## Unified parallel rigid solver, 2026-09-23

All non-empty rigid worlds now use the same matching/coloring and contact
resolution kernels. The serial small-world solver and serial greedy colorer
were removed. For fewer than nine bodies, the maximum color-round count is
the number of possible unordered body pairs (at least one); each round
colors the lowest-priority remaining pair, so this limit cannot discard a
contact. Larger worlds retain the 32-round cap and serial overflow fallback.
Contact physics and triangle geometry are unchanged.

With the gallery closed, the five-body `PassiveActive.glb` benchmark measured
2.116 ms median GPU time, versus 1.506 ms with the previous serial small-world
path. A fixed 32-round parallel schedule had measured 3.423 ms; bounding the
small-world rounds recovered most of that overhead. A 10-sphere DUMP run
averaged 2.942 ms over frames 451–480 and finished with all spheres inside the
receiver. The 128-sphere run averaged 6.651 ms over frames 31–60; this is an
absolute measurement, not a before/after claim for that size.

The 1,000-sphere path is unchanged: frames 31–60, 211–240, and 451–480
averaged 20.588, 24.909, and 15.100 ms. Its 480-frame final state hash was
again `4a016b8258d02a31`, with no invalid or escaped spheres. All five tests
passed, including the two-body timing/diagnostic contract, deterministic
contact fixtures (including eight overlapping dynamic bodies), and both
gallery headless scenes. The unified path also
preserves contact events from earlier substeps when a later substep has none.

## Fluid + rigid coupling: 64 spheres, 30,000 particles

The `FluidRigid.blend` scene exports 64 separate ACTIVE triangle spheres from
three Array modifiers, plus one passive terrain body. All spheres reuse one
20-triangle mesh. The benchmark can be reproduced with
`./build-gallery/parallel-mater-fluid-rigid-benchmark 2000` on the local RTX
3050 Ti. Every tenth frame collects CUDA stage timings and optional contact
events; each 100-frame row reports the mean of those ten samples. The other
frames run without diagnostic collection.

At frame 2,000, the scene had 29,993 live particles, zero non-finite body or
particle states, zero particles below the terrain's global bottom, and zero
contact-event overflows in all sampled windows. Frames 1,901–2,000 averaged
16.49 ms/frame wall time and 16.76 ms/frame GPU time. Moving-triangle contacts
averaged 1.45 ms, body indexing 0.012 ms, static triangle contacts 1.49 ms,
neighbor forces 6.51 ms, and rigid contact solving 4.23 ms. Event collection
averaged 0.025 ms per sampled frame. The diagnostic stream retains at most one
contact per surviving particle per frame, prioritizing moving bodies; this
prevents ordinary resting floor contacts from saturating the default 65,536
event buffer at 30,000 particles.

An independent 1,000-frame gallery run with `--fluid-rigid
--fluid-particle-view --trace-fluid-escapes` found zero particles below the
authored terrain, both by its global bottom and by the local floor-triangle
check. The post-rotation-bound 1,000-frame benchmark also finished with zero
invalid states and zero event overflows.

The first unindexed moving-body implementation approximately doubled the
120-frame wall time (2.9 s versus 1.4 s uncoupled). The spatial body index
reduced the coupled 120-frame run to about 1.42 s. These are local project
measurements, not general hardware guarantees.

## Peg Paint: one-shot fill and sinking body

`Pegs.blend` exports six passive triangle bodies, one dynamic Icosphere, and
one Liquid Flow/Geometry cylinder. The first implementation used 2,947
cubic-grid particles, 1g, repulsion 8, tangential viscosity, and a sphere
starting almost on the bowl bottom. It stayed contained, but formed a shallow,
contact-heavy layer with no visible sinking path. At authored mass 1 kg, the
sphere floated; the old Water example used a 3,500-unit sphere against
roughly 2,500 displaced unit-mass particles.

The old Water physics also used HCP packing at 0.045 m, pair repulsion plus
radial damping, a 55-unit force cap, 0.4/s velocity damping, a 3 m/s speed
cap, two-g gravity, and overlap correction reacting on both particle and
sphere. The reusable API now supports HCP-consistent particle rest volume,
radial pair damping, a bounded pair-acceleration cap, and shared
fluid/triangle-body overlap reaction. The Blender Geometry flow authors
0.052 m spacing and 2× gravity; its 5,707 particles retain the intended
fluid density. At the smaller bowl scale, repulsion 30 kept the pack
contained; 50 ejected 15–18 particles over the open rim. The sphere is
authored at 125 kg and Y 0.4 m, above the bowl, so it visibly falls and
settles near Y −0.72.

In `parallel-mater-pegs-benchmark 2000` on the local RTX 3050 Ti, all 5,707
particles remained live and inside the bowl at every 100-frame sample;
contact overflow stayed zero. Frames 1,901–2,000 averaged 15.55 ms per
physics step. The old example's analytic sphere collision, analytic bowl
projection, and bowl-specific inward hydrostatic correction were not copied:
this scene still uses authored triangle meshes and the shared solver.

### Old Water comparison, 2026-09-24

After 300 warmup frames, 60 profiled frames of the old Water course with
20,000 particles, four iterations, 320×240 particle rendering, and foam off
measured 4.04 ms median wall time (3.91 ms GPU). The revised Peg scene with
5,707 particles averaged 10.31 ms physics wall time over frames 201–300,
excluding rendering. These are **different scenes and workloads**, not an
algorithmic speedup ratio. At Peg frame 300, GPU stages were 3.39 ms rigid
triangle contact, 3.24 ms fluid neighbors, 2.40 ms static triangle contact,
0.73 ms moving-triangle contact, and 0.59 ms cell sorting.

The old course projects particles against an analytic hemisphere and capped
cylinders in its integration kernel and couples one analytic sphere. Peg uses
generic BVH triangle contacts for six passive bodies, one dynamic body, and
the shared rigid solver. The old course runs four fluid iterations per frame;
Peg currently runs four substeps × two solver iterations. The old neighbor
kernel schedules threads in sorted-cell order and sorts only active particles;
the reusable API schedules in particle-ID order and sorts reserved capacity
(30,000 slots here). The old bowl also has a bounded, bowl-specific
hydrostatic equalization force that reduces its raised outer ring. That
heuristic is part of its appearance but is not general triangle-water physics.

### Wall drainage and one-texel paint, 2026-09-25

The old bowl cancels inward normal velocity but retains tangential wall motion.
The shared triangle solver applies authored material friction to fluid too, so
the default Blender friction `0.5` removed 75% of tangential velocity on each
impact. Zero friction caused a transient neighbor overflow at repulsion 30;
raising repulsion to 50 avoided overflow but made the wall layer surge before
settling. Keeping the shared solver and setting only the bowl and invisible
containment cylinder to friction `0.05` drained the wall without that surge.

At 5,707 particles and repulsion 30, the number above Y −0.45 m in the outer
R > 0.8 m annulus fell from 459 to 18 at frame 100 and from 266 to zero at
frame 300. No particles were below the bowl's lower bound at the sampled
frames, and physics timing remained near 11 ms per frame. Paint now sets a
single texel per particle contact, as in the old bowl; a 64×64 mask gives the
smaller bowl 64×32 used texels, reconstructed with a smooth cubic B-spline
filter to avoid the coarse mask's square edges.

The Peg arrow-key limit is now 50° from vertical, above the original lab's
20° and below the excessive 85° setting. Static fluid contact previously
removed 7.5% of tangential velocity on each solver pass against the bowl's
0.05-friction material. Eight passes per frame could halve wall momentum.
The static contact now uses impulse-limited Coulomb friction, matching the
dynamic fluid contact and retaining tangential motion when normal impulse is
small. Bulk velocity damping (0.4), normal damping (2), and speed cap (3 m/s)
already matched the old lab and were left unchanged.

In a 50° gravity-reversal probe, mean horizontal particle velocity ten frames
after reversing changed from −1.27 m/s to −2.29 m/s. At frame 190, the old
contact still moved left at −0.21 m/s; the revised contact had rebounded right
at +0.21 m/s. The slosh regression guards both behaviors. At sustained 50°
tilt, the highest outer particle reached Y 0.52 m at frame 600, with no
lower-bound escapes or contact overflow. The invisible containment cylinder
still holds a raised sheet under prolonged strong tilt; spilling over an open
rim would need separate scene authoring.

### Shared foam visualization

The render-only `FoamVisuals` module now owns stable-ID patch lifetime,
bubble drawing, and depth occlusion for Fluid, Fluid + Rigid, and Peg Paint.
The two stream scenes have 0.045 m particles and a wider camera than Peg's
0.03 m particles, so patch and minimum screen-space radius scale with particle
and support radius. This does not change the fluid solver or foam signal.
At 14,400 Fluid particles, the bounded 2,048-patch renderer took 4.7 ms CPU
on the local test frame; Fluid + Rigid took 4.1 ms. Headless scene checks
require nonzero patches, and a CPU test covers scaling and rigid occlusion.

## Soft body passive collision, 2026-09-27

The supplied `Softbody.blend` exports one closed Icosphere and one passive
triangle arena. At 0.222 m authored node spacing, the shared volume sampler and
graph builder produce 673 physical nodes and 10,032 unique spring bonds. The
solver uses four substeps and 16 bounded Jacobi spring iterations per frame.

On the local RTX 3050 Ti, one opt-in timed frame after a 600-frame settling run
measured about 4.0 ms total GPU physics time. Maximum settled bond strain was
25.3%, peak residual node speed was 0.033 m/s, and no node crossed below the
passive mesh; the minimum measured node clearance was 0.083 m. A 45-degree
traction run stays under the authored 2 m/s node cap, averages 0.90 of the
no-slip angular/linear rolling ratio, rebounds at 0.72 m/s, remains below 31.1%
bond strain through impact, and records no node beyond the arena walls. The
test also checks finite state, all 10,032 API-visible bonds,
generation-invalidated handles, API capture, and the real Blender export.
These are local acceptance measurements rather than a cross-hardware
guarantee.

## Soft body dynamic rigid coupling, 2026-09-28

`SoftbodyRigidBody.blend` reuses the 673-node, 10,032-bond soft lattice and
adds two 100 kg active triangle-mesh spheres. Dynamic contacts run through the
same interleaved soft-body contact pass as passive geometry. Per-node contact
impulses are deterministically reduced into the existing rigid states, while a
post-constraint lattice momentum correction preserves the matching soft-body
impulse. No second solver or gallery-only coupling is involved.

On the local RTX 3050 Ti, a timed frame after 1,200 settling frames measured
about 5.6 ms total GPU physics time. Maximum settled bond strain was 44.5%,
maximum rigid speed was 0.076 m/s, maximum soft-node speed was 0.019 m/s, and no
node crossed the passive floor. In the focused 1.5 m/s control impact, a 1 kg
sphere produced 30 transfer steps with 4.3% mean and 15.6% worst per-step
momentum imbalance and moved the soft-body center 0.757 m. Under the same
conditions the authored 100 kg sphere moved it 1.160 m, a 53% increase, and
retained 1.410 m/s forward speed. A separate 4 m/s heavy impact moved the
soft-body center 2.412 m, remained finite and non-tunneling, and duplicate
heavy runs matched byte-for-byte.

The reusable co-rotated shape constraint maps the authored Blender Goal weight
and stiffness to a 0.35 API stiffness. Two symmetric 100 kg impacts produced a
15.5% peak radial shape error. Shape restoration yields while those dynamic
contacts are active instead of pushing through them. A quarter second after
both loads were removed, the normalized radial error was 0.00032%, versus
0.0064% for the same spring lattice with shape matching disabled. A 300-frame
camera-relative gravity-steering regression left zero nodes beyond the passive
arena bounds. Recovery preserves the body's center and best-fit rotation, so
the existing rolling regression remains unchanged.
These values are local acceptance measurements rather than a cross-hardware
guarantee.

### Contact audit and containment regression, 2026-09-28

The revised user asset contains two passive objects and has Blender Goal
disabled. A new stress test steers gravity through eight directions at 45°,
180 frames per direction, using four substeps at 60 Hz. It checks every frame,
including physical nodes and skin-triangle centroids and edge midpoints, rather
than only checking final bounds. The original solver put nodes as much as
0.4395 units inside a rigid sphere; 1,356 of 1,440 frames exceeded 0.02 units.
The final frame still had zero escaped nodes, so the old check missed this.

The shared solver now recovers interior nodes toward the outside of verified
closed convex triangle meshes, shares positional reaction by inverse mass,
and constrains actual skin triangles through their API bindings. A final
geometric cleanup follows friction. Its small recovery skin avoids contradictory
full-node-radius margins in narrow gaps, while the main contact/friction solve
retains the authored node radius. Keeping only one adjacent face correction
left brief face-interior penetration; combining contact normals resolved it.
Increasing cleanup iterations alone, with the conflicting full-radius margins,
did not resolve the problem and was not retained.

With Goal disabled, the final 1,440-frame run measured 0.00281334 units maximum
sampled penetration. With shape-matching stiffness 0.35, the corresponding run
measured zero sampled penetration. Both had zero wall escapes and zero frames
above the 0.005-unit collision-margin tolerance. These are sampled, discrete
contact measurements, not a guarantee of continuous triangle collision at
arbitrary speeds or timesteps. The checks run as
`parallel-mater-soft-body-contact-stress` and
`parallel-mater-soft-body-shape-contact-stress` in CTest.

On the local RTX 3050 Ti, the revised asset's timed frame after 1,200 settling
frames measured 6.68 ms GPU physics time, with 41.3% maximum bond strain,
0.055 m/s maximum rigid speed, and 0.018 m/s maximum soft-node speed. This
single-frame timing is not a like-for-like speedup comparison with the earlier
one-passive-object asset. The focused momentum-transfer, heavy-impact,
deterministic-replay, shape-recovery, and passive rolling checks still pass.

The unfinished surface experiment also exceeded the kernel-timing event budget
and mixed cross-block position reads and writes. The retained implementation
uses separate detection and deterministic gather kernels, budgets the cleanup
timing boundary, and checks event capacity before recording it.

Validation: all 41 CTest cases passed. A targeted CUDA memcheck of 2,048
collision/surface/reduction kernel launches after skipping 10,880 matching
launches completed with zero errors in the 180-frame `--contact-smoke`
reproducer. This covers the initial collision window, including the former
frame-87 failure; it is not a full-run memory-check claim. The long instrumented
regression did not complete, so the short reproducer is available for repeatable
memory checking without shortening either full CTest stress case.

## Soft body and cloth coupling, 2026-09-28

`SoftbodyCloth.blend` contains one soft sphere, two 289-vertex/512-triangle
cloths, and two passive triangle meshes. Each cloth uses its own material:
the pinned bridge has fracture disabled and 48 spring iterations; the vertical
curtain has 10% break strain, four-substep persistence, and 24 iterations.
The shared API uses four contact passes per substep, triangle barycentric
reactions, Coulomb friction, and deterministic gathers. Gravity starts down;
the regression begins a 45-degree roll toward the curtain after 240 frames.

The first eight-iteration bridge sagged 0.825 units and trapped the sphere.
Increasing authored spring convergence let it roll out. A stronger clearance
test then found 0.0875 units of local impact penetration despite successful
support. Dividing both contact responses by the full cloth contact degree
over-relaxed the independent soft nodes. Scaling the effective inverse mass
of the shared cloth instead removed that penetration without extra passes.
Both bodies still receive the same opposing impulse.

The initial 900-frame run at 60 Hz/four substeps, before the fragment fix below, measured:

- Minimum bridge clearance 0.0414 units over the initial 240 frames, sampling
  every soft node, skin vertex, triangle centroid, and edge midpoint against
  the deformed bridge triangles.
- Zero pin displacement and zero broken bridge bonds. Maximum bridge sag was
  0.516 units; maximum live bridge-bond strain over the full run was 31.3%.
- 237 broken curtain bonds, none before contact, with all 512 triangles
  retained. The sphere passed through and finished at z = -4.936.
- Maximum soft-node speed 2 m/s; normalized contact-force imbalance below
  3.8e-7, including reactions at pinned vertices.

Controls remove all rigid geometry and disable the API coupling: after 180
frames the sphere falls to y = -4.883 instead of being supported at y = 0.082.
Two enabled runs match node positions exactly. Disabling curtain fracture
keeps its bonds intact and blocks the sphere from both sides, with 160 contact
frames in each 180-frame run. Lifecycle, capture-force, and timing checks are
part of the same test.

On the local RTX 3050 Ti, the full scene's opt-in timed frame measured 16.36 ms
GPU physics, including 7.67 ms in soft/cloth coupling and cleanup. In the
one-cloth/no-rigid support control, unprofiled stepping with capture disabled
averaged 8.97 ms/frame; disabled coupling averaged 5.22 ms/frame. These are
different workloads, not a full-scene speedup claim. Contacts currently scan
cloth triangles per soft node; these measurements do not establish scaling
to large collections of deformables or continuous collision guarantees.

### Independent cloth fragments

The later tear audit exposed a missing test: fitted surface triangles could
follow original vertices belonging to different physical components. In a
1,200-frame replay, up to four torn faces had all three corners and their
centroid inside the soft skin; one stayed inside for 172 frames. Such faces
occurred in 191 frames. Surface corners jumped at an apparent 100.35 units/s,
although sampled physical cloth nodes peaked at 6.37 units/s.

The API now splits physical vertex fans at broken seams, disables spanning
bends, and uses those same vertices for rendering and collision. It reserves
node/link capacity up front and rebuilds connectivity only after a bond-state
change at an idle frame boundary. Mass and velocity are inherited through
incident-face shares; authored source indices remain stable for painting.

The extended 1,200-frame regression (240 down, 660 tilted, 300 down) measured
zero fully-inside sampled faces and zero surface/physical-position mismatch.
The sphere passed through the curtain and finished at z = -4.477. The bridge
retained all bonds and pins, with 0.0414 minimum sampled clearance; normalized
force imbalance stayed below 4.1e-7. All 512 curtain triangles remain present.
The two-fragment isolation test preserves mass and inherited velocity, then
kicks only one fragment with gravity/contact disabled: maximum error from the
expected independent trajectories was 6.35e-5 units over 120 frames.
A separate 2,400-frame replay, reversing gravity every 180 frames after frame
900, also found zero fully-inside sampled faces. Physical/rendered corner
motion now agrees; maximum frame-to-frame corner speed was 13.25 units/s,
instead of the previous fitted-surface jumps. The complete 46-test suite and
the fragment-isolation CUDA memory check passed (zero reported memory errors).
The focused memory-check regression also allocates a soft/cloth coupling
before splitting, then drives contact into appended cloth vertices: peak
split-node force 4.674 N, exactly zero force on the remote fragment, and zero
memory errors. An instrumented full-scene replay exceeded its 240-second bound
before the selected tear window; no full-scene sanitizer coverage is claimed.

A global strain clamp was rejected: it prevented continued tearing and could
push attached faces inside the soft body. Only isolated, unpinned triangles
receive the additional 10% strain projection, within the contact solve.
Attached material still follows its authored compliance/break threshold.
The settled pre-tear frame measured 16.88 ms GPU time (7.77 ms coupling),
versus the earlier 16.36 ms sample. This is not a speedup claim or a measurement
of host-side topology rebuilding during fracture.

Validation: all 43 CTest cases passed, including the real Blender exporter and
headless scene. Targeted CUDA memcheck checked 2,048 `soft_cloth_` launches after
skipping 3,456 matching launches (the bridge-impact window) with zero errors in
the 240-frame `--smoke` run. This is a bounded new-kernel check, not a claim of
full-run sanitizer coverage.

## Rope settling (in-progress PR 16)

The original rest curve crossed an invisible enclosure wall: one segment
reached over 100 times its rest length. That is an invalid starting topology,
not a material-stiffness problem. Only the enclosure was widened to the
existing floor bounds. Curve geometry, Hook endpoints, post, ball, and material
values were preserved. `World::add_rope` rejects centerlines crossing rigid
triangles before creating a resource, with a failed-creation regression.

The corrected scene exposed an independent solver problem. Contact normals
were released after building the reduced inverse-mass matrix, but corrections
used the newly released masses. Rebuilding after release prevents that energy
injection; a surviving second support plane remains active. A full-mass
velocity-level distance solve removes axial relative motion, with balanced
impulses at dynamic attachments. Contact-reduced velocity solves were rejected:
incoming normal velocities can be incompatible with the reduced mass matrix.
The API also enforces a 1/480 s maximum integration step, shared with the rigid
attachments. It preserves uniform free-fall velocity in the API regression.
Active-set flags are synchronized between passes; otherwise a fast warp can
clear the flag before another warp reads it and split the block's control flow.
Nominal solves exit below 0.1% segment strain. Contacts still above 0.5% after
the nominal iteration budget receive bounded recovery passes (up to four
times that budget, capped at 32). A capped contact solve finishes with eight
length projections after releasing stale support planes. Each projection
moves at most one rope radius. The looser early-exit tolerance avoids
spending every iteration chasing micrometre-scale residuals, while the final
projection prevents large length errors from becoming velocity spikes.

The 1,200-frame release test uses 72 nodes, 1/60 s frames and 24 nominal
iterations. The initial solver used four substeps; the corrected API uses eight.
Gravity is down initially, tilted 45 degrees during
frames 120–239, then down again. The final 120 frames measure settling. RTX
3050 Ti timings include opt-in CUDA stage events, not rendering:

| Corrected scene, steering release | Initial solver | Contact + velocity + timestep fixes |
|---|---:|---:|
| Mean GPU physics / frame | 24.88 ms | 6.68 ms |
| Peak segment strain | 2.88% | 0.32% |
| Late RMS node speed | 13.13 mm/s | 1.99 mm/s |
| Maximum late displacement | 18.38 mm | 4.44 mm |

At that stage, the straight-down control also improved settling: RMS speed 2.73 to 1.62 mm/s,
late displacement 4.38 to 2.39 mm, and strain 2.88% to 0.32%. Its mean GPU cost
is similar (5.58 versus 5.51 ms). The two runs have 40 and 38 of 72 nodes,
respectively, within 25 mm of the floor, hook error below
0.13 micrometres, and zero measured node penetration into the active sphere.
The section rising to the elevated post is intentionally suspended. Increasing
velocity damping from 0.1 to 0.5 /s did not improve the controlled release test
and was rejected; the material damping remains unchanged. Internal bend damping
and moving static friction entirely to the velocity stage also regressed
settling or impact stretch and were removed. Continuous circular steering
passes at 0.32% peak strain and 7.66 ms/frame. A pre-wrapped three-turn fixture
separately checks segment samples around the triangle post: 0.47% peak strain,
at least 2.85 retained turns, and 22.29 ms/frame. Its longer curve is a test
fixture, not a change to the authored rest curve.

All eight targeted rope/export/render regressions pass. The final 12-frame
wrapped-contact smoke test reports zero Compute Sanitizer memory errors and
zero racecheck hazards; the normal wrapped regression runs 600 frames.

### Tight winding and bounce

A controlled 500-frame test steers the authored 72-node rope around its post
with tangential and inward gravity after frame 120. It reaches 2.94 turns,
tightens, bounces, and begins unwinding. The original solver spent 428 ms in
the rope kernel at the third wrap and later stretched one segment by 68%.
Renderer time at the hitch was under 4 ms; the stall was in physics.

On the RTX 3050 Ti, the same physics-only steering sequence measured:

| 500-frame winding run | Original | This change |
|---|---:|---:|
| Peak rope GPU / frame | 428.07 ms | 90.98 ms |
| 95th percentile rope GPU / frame | 125.09 ms | 41.75 ms |
| Peak segment strain | 68.46% | 0.497% |

The shared triangle solver now checks individual triangle bounds before the
expensive capsule-to-triangle distance query. A verified separating-plane hint
skips repeated convex plane tests, including whole-body queries when a static
collider is provably beyond the rope radius. Bounded contact recovery and the
final length projection keep the loaded Hook from entering a long solve loop.
The winding regression checks peak frame time relative to its mean, rope
strain, Hook drift, active-ball penetration, and node/segment post clearance.
The 1,200-frame quiet and steering-release tests, 600-frame pre-wrapped test,
and headless render still pass. A wider triangle candidate cache was rejected
after it reduced post clearance; warming support planes and applying friction
only on the first iteration also regressed stability or settling.
Current quiet and steering-release mean GPU physics times are 5.86 and
7.69 ms/frame, with late RMS speeds of 1.61 and 3.28 mm/s respectively;
the pre-wrapped fixture averages 17.06 ms/frame. The steering-release test
therefore costs slightly more than the earlier 6.68 ms snapshot, while the
tight-winding peak and pre-wrapped throughput improve substantially.

### Soft post and shared solver profiling (RTX 3050 Ti)

The authored RopeSoftbody post has 912 simulation nodes, 130,033 bonds, and
33 pinned base nodes. A rope-only pull test verifies that Hook tension bends
the post without ball/post contact. Distributed anchor loads and swept
triangle contact keep a 500-frame, three-wrap winding run at 0.50% peak rope
strain and 0.321 m maximum post bend. The same steering followed by downward
gravity after frame 500 is a 1,000-frame regression and also passed a manual
5,000-frame endurance run: strain stayed at 0.50%, the base did not drift,
and the post ended at about 2.5 mm bend. Continuing the
tangential drive indefinitely can still overload the post after frame 700;
that is a stress limit, not covered by the release regression.

Measured hypotheses on the same GPU:

| Workload/change | Before | After | Decision |
|---|---:|---:|---|
| 500-frame rope/post, ordered parallel shape projection | 81.45 ms/frame | 80.22 ms/frame | Keep; state metrics identical |
| Shape projection kernel, 30-frame profile | 497 µs/call | 380 µs/call | Keep |
| ClothWater, vertex-centric closed-volume projection | 12.47 ms/step | 8.37–8.55 ms/step | Keep; 5,013 particles, zero escapes |
| Closed-volume kernel, 3,600 calls | 105 µs/call | 11 µs/call | Keep; volume ratio 0.999999–1 |
| 1,000-body DUMP, 32-thread leaf evaluation | 14.97 ms/frame | 13.74 ms/frame | Keep; identical state hash, settled interval |
| FluidRigid, 65 bodies, fused small-scene coloring/solve | 9.39 ms/frame | 6.66 ms/frame | Keep; 100 profiled frames, same contacts and final speed |
| 100-sphere DUMP, fused small-scene coloring/solve | 4.14 ms/frame | 2.72 ms/frame | Keep; identical state hash, last 30 frames |

The cloth volume constraint precomputes incident triangle corners in face
order. Independent triangle volumes and per-vertex gradients run in parallel;
the two global scalar sums retain their original order. Soft-body shape
matching likewise keeps center, transform, and momentum sums ordered while
projecting nodes in parallel. These avoid unordered floating-point atomics
and preserve the tested trajectories.
The old water lab's node-centric spring projection confirmed that the dense
post should stay parallel across nodes. An ELL-transposed neighbor layout
coalesces those loads on the current near-uniform lattice while preserving
each node's spring accumulation order; sparse graphs still use CSR.

Rigid contact coloring at 24 rounds saved roughly 0.3–0.5 ms/frame on the
1,000-body DUMP versus 32 rounds without changing the state hash. Fewer
rounds spilled contacts to the serial fallback and were rejected. A 128-thread
leaf evaluator and an adaptive dual-launch version were slower. Soft graph
launches at 32 or 64 threads produced no repeatable gain over 128 and were
reverted. Extra rope recovery passes and narrower post force support worsened
either runtime or long-run stability and were also rejected.

For at most 256 rigid bodies, color rounds and the eight contact passes now
run inside single synchronized GPU blocks. Disjoint pairs remain parallel
within each color; a barrier preserves color order. Larger worlds retain the
multi-block path. This removes many tiny launches without changing the 100-
sphere DUMP state hash. At 250 spheres, the last 30-frame GPU mean was 7.32 ms
with the fused path versus 7.88 ms with the multi-block path; hashes matched.
Replacing the rope movement maximum's shared atomic
with another warp reduction was slower (80.63 versus 80.22 ms/frame) and was
reverted.
The large-world path also ran a 1,000-sphere DUMP for 10,000 frames. Its final
state had zero invalid, below-floor, or outside-receiver spheres; the final
30-frame window averaged 14.16 ms/frame GPU physics.
The 65-body FluidRigid scene also ran for 10,000 frames. Its final state had
29,996 live particles and zero invalid or below-mesh particles; the last 100
frames had zero contact-event overflow and averaged 16.60 ms/frame of GPU
physics. The higher late cost
reflects the fluid approaching its 30,000-particle cap; the small rigid solve
remained about 1.77 ms/frame.
