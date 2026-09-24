# Rigid contact performance, 2026-09-21

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
