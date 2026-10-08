# Physics performance

## Metal post-regression solver audit (Apple M4, 2026-10-08)

The opening scene continues to use its authored indexed triangle meshes. No
box primitive, simplified collision shape, welded brick, reduced substep, or
reduced solver budget is used in this follow-up.

An exact rebuild A/B closed the interactive gallery, ran the public-API impact
replay twice on `0fac9c4`, rebuilt the candidate, then ran the same replay
twice. Each replay settles for 120 frames and measures 120 frames at 1/60 s
with four substeps and sleeping enabled:

| Build | Wall median | Wall p95 | GPU median |
| --- | ---: | ---: | ---: |
| `0fac9c4` control, average of two | 61.66 ms | 76.21 ms | 61.27 ms |
| Retained candidate, average of two | 61.33 ms | 75.95 ms | 60.90 ms |

The retained change is deliberately small: about 0.5% wall time and 0.6% GPU
time in this final paired sample. Both candidate runs improved their paired
control run, but the difference is near normal run-to-run variation and should
not be extrapolated. The impact trajectory remains unchanged at 384 displaced
bricks, peak speed 10.1393 m/s, and peak mechanical-energy ratio 1.0.

Retained implementation changes:

- Declare the actual maximum threadgroup sizes used by contact generation,
  cooperative triangle contact collection, cache matching, contact reduction,
  and stack solving. This gives the Metal compiler a tighter register-allocation
  bound without changing dispatch sizes.
- Do not initialize or update the convergence maximum during passes 0–15,
  because the existing solver cannot exit before pass 16. Pass 16 onward keeps
  the same residual, threshold, overflow rule, contact order, and iteration
  budget.

Measured and rejected experiments:

| Experiment | Result | Decision |
| --- | --- | --- |
| Exact 128-thread compiler cap | 62.19–62.83 ms median | Reject; slower than 256-thread cap |
| 64-thread solver group | 72.89–74.91 ms median | Reject; 128 threads remains faster |
| SIMD convergence reduction | 61.29 ms average median | Reject; whole step regressed despite cheaper solve |
| Two local pair sweeps per color visit | 63.94 ms median, 91.73 ms p95 | Reject; slower and changed trajectory |
| Algebraic friction normalization removal | 60.76 ms average median, 382 displaced bricks | Reject; marginal gain changed trajectory |
| Cached friction coefficient | 61.98 ms average median | Reject; added manifold bandwidth cost more |
| Explicit residual-result gating | 60.98 ms average median | Reject; no gain over compiler optimization |

The two-sweep experiment remained within the energy bound over six seconds,
but moving dependent work inside each pair reduced propagation through the
connected wall. Earlier prepared-Jacobian, cross-threadgroup block, transformed
leaf-cache, and permissive cached-anchor experiments remain rejected for the
stability or performance reasons recorded below. A CPU/GPU hybrid was not
implemented: the current step encodes all four substeps in one GPU command
buffer, so inserting a CPU solve requires four execution/read/write boundaries
and a different step architecture rather than a local solver optimization.

These results still do not establish a hardware limit. They show that local
shader tuning is now yielding sub-millisecond changes while the connected
Gauss-Seidel solve remains serial across contact colors inside one GPU
threadgroup. A material improvement requires a stable solver that exposes
more independent work or converges in fewer dependent sweeps.

## Metal impact regression correction (Apple M4, 2026-10-08)

The 47.53 ms impact result reported at `20a1072` is invalid. The wall could
explode while all positions and velocities remained finite. The old benchmark
only checked finite final states and displaced bricks, and the quiet-wall
and short wake tests did not detect this impact failure.

Two unsafe changes have been removed:

- Batched normal impulses used the same stale pair velocity for every contact
  on a face, then summed the impulses without accounting for their coupling.
  The solver again updates both body velocities after each contact row before
  evaluating the next row, including sleeping-enabled worlds.
- The discrete convex-face shortcut replaced required swept collision tests.
  Its success return can describe an empty manifold at separated endpoints,
  so a fast body could pass completely through another body. Swept pairs
  again use the existing CCD path.

The compact geometry-only cooperative scratch optimization remains. It stores
32-byte point, normal, penetration, and impact records, reducing the 32-lane
scratch allocation from 17,664 to 8,320 bytes. Lane zero reconstructs the
unchanged solver records in the same deterministic merge order. This retains
the measured improvement without changing contact response or CCD semantics.

Corrected runs use the public Metal API, the authored 386-body scene, a 1/60 s
timestep, four substeps, sleeping enabled, and 120 measured impact frames after
120 settling frames. The interactive gallery was closed. The averages of three
earlier baseline runs and two corrected runs are:

| Impact metric | `ea276e5` | Corrected result |
| --- | ---: | ---: |
| Wall median | 73.12 ms | 63.32 ms |
| Wall p95 | 93.52 ms | 79.88 ms |
| GPU median | 72.63 ms | 62.63 ms |
| Contact solve median | 46.50 ms | 44.58 ms |
| Contact evaluation median | 17.37 ms | 12.88 ms |

Individual corrected impact medians are 63.43 and 63.22 ms; p95 is 80.26 and
79.50 ms. The median is about 13% below the earlier valid baseline, not the
previously claimed 35%. These are separate runs subject to thermal and host
scheduling variation, not a new interleaved comparison. All 384 bricks moved
independently by more than 5 cm. Solve work remains the largest cost.

The new regression launches the authored 100 kg ball at 10 m/s and checks every
frame for six seconds with sleeping both enabled and disabled. It rejects
translational kinetic energy plus signed gravitational potential above 110%
of the starting budget. Signed potential accounts for bricks falling off the
finite floor. This is a lower-bound energy check, not a full rotational-energy
or trajectory-parity proof. The benchmark runs the same check outside its
physics timer and refuses to report successful impact timings on failure.

Both guards reject the old build at frame 18: 581,367 J kinetic plus 10,478 J
potential energy versus a 15,071 J starting budget, with a peak speed of
203.6 m/s. The corrected six-second runs peak at 0.99951 of the starting
mechanical-energy budget. A separate regression checks a 90 m/s cube crossing
a static cube in both two-body and 32-body worlds; it fails the unsafe shortcut
and passes the restored CCD path.

The strict quiet-wall gate retains its unchanged thresholds: 0.00197983 m
maximum drop, 0.00305891 m displacement, 0.000976562 rad rotation, 0.103747 m/s
peak speed, 0.00441963 m/s late speed, 1.00007 maximum energy ratio, and
0.000647023 m minimum clearance.

Rejected experiments:

- Caching a world inverse-inertia matrix per pair/pass raised impact median to
  about 86.2 ms. Lower occupancy was suspected but not measured directly.
- Caching the four transformed triangles in each BVH leaf raised median to
  about 81.1 ms. Its cause was not isolated.
- Merging restored cached anchors into newly discovered swept manifolds raised
  median from 63.8 to 65.4 ms and changed the displaced-brick count to 382.
- Allowing convergence after pass 8 at a `1e-8` squared velocity-change
  threshold raised median to about 67.9 ms. The changed trajectory created
  more downstream work.
- A four-partition, multi-threadgroup block preconditioner produced a
  misleading 12.6 ms median by destabilizing the wall. The correctness gate
  measured 310 m drop, 412 m displacement, and extreme energy growth, so the
  implementation was removed.

No public API signature or layout changed, and the CUDA implementation remains
untouched.

## Expanded authored wall status (2026-10-05)

The current `RigidBody.blend` and GLB contain 384 independent 1 kg bricks
(192 from each Array source) plus the sphere and ground. Both Array sources
export `pm_gravity_tilt = false`. Export and scene-loader regressions pass
with the expanded counts, and the full Release build succeeds.

The CUDA reference does **not** pass the unchanged physical acceptance limits
for this expanded wall: its ten-second vertical-gravity regression measures
0.126318 m maximum drop, 0.286006 m displacement, and 0.524741 rad rotation.
At 386 bodies CUDA uses its larger-world contact solver, outside the optimized
32–256-body stack path described below. CUDA larger-stack support remains
unresolved; the Metal result is recorded separately below.

## Metal collision scheduling and bounds follow-up (Apple M4, 2026-10-08)

A further audit of commit `f9540fa` found avoidable serial work inside the
API's aggregate contact-solve timer. Cache matching ran in a single
threadgroup; cached-color validation and iteration selection scanned every
pair serially; island labeling scanned every body for every island. The
ordinary solve itself remains the largest remaining cost.

The authored impact replay below was measured twice with interleaved baseline
and candidate binaries, the interactive gallery closed, and no concurrent GPU
tests. Scene, timestep, four substeps, solver budgets, and physical acceptance
thresholds are unchanged.

| Impact run | Before median | After median | Before p95 | After p95 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 89.42 ms | 73.51 ms | 101.94 ms | 95.46 ms |
| 2 | 89.25 ms | 73.85 ms | 102.39 ms | 92.65 ms |

This is about 18% less median physics time, or roughly 13.6 simulation steps/s
during heavy impact. It is still above the 16.7 ms budget for 60 Hz. Both
candidate runs displaced all 384 bricks independently by more than 5 cm,
with finite positions, velocities, and orientations.

The paired quiet run improved from 24.96 to 20.18 ms with sleeping disabled
(p95 25.82 to 21.75 ms). Sleeping rest measured 1.14 ms with all 385 dynamic
bodies asleep. Held steering measured 1.70 ms with 384 bricks asleep; its GPU
median was 0.82 ms. Small wall-time differences in sleeping workloads include
host scheduling latency and should not be treated as fixed frame costs.

Retained backend changes:

- Match cached contacts in an indirect pair-parallel dispatch before contact
  reduction, preserving the same matching and warm-start formulas.
- Validate cached colors and select the maximum 8/32/64-pass budget with
  parallel atomic reductions. The existing all-or-nothing reuse decision,
  deterministic recoloring, overflow behavior, and convergence threshold stay
  unchanged.
- Skip the serial contact-event prefix when events were not requested.
  Requested events keep their existing ordering and capacity behavior.
- Label island roots and members in parallel around a linear root-ID scan,
  preserving ascending-root island IDs and the same contact graph.
- Reject separated swept oriented bounding boxes before expensive leaf
  contacts. The union of endpoint projections encloses the same linear vertex
  motion as the existing triangle sweep; margins and a coordinate-scaled
  roundoff guard make the rejection conservative. Larger leaf products, deep
  static sweeps, and constrained/coupled paths keep their existing fallback.
- Use 128-thread solver groups for ordinary worlds up to 512 bodies; larger
  worlds keep 256. This adjusts scheduling without reducing solver passes.
- Report all contact-generation and solve dispatches in the existing API
  timing counters; the wall regression checks the ordinary-stack counts.

A diagnostic run computed the proposed bounds rejection but still ran the
original triangle evaluation for all 480 measured impact substeps. It found
166,019 rejectable pair/substep instances and zero rejected pairs with actual
contacts. The retained guard additionally includes unprojected coordinates to
cover cancellation at large coordinates. Diagnostic kernels and readback were
removed from the final build.

The strict ten-second wall and sleep/wake gates retain the previous measured
stability, including 0.00197983 m maximum drop, 0.00305891 m displacement,
0.000976562 rad rotation, 0.00441963 m/s late speed, and 0.000647023 m minimum
clearance. No rendered geometry, body mass, timestep, collision margin,
friction, or restitution was simplified.

Rejected probes: threadgroup-local body state and compact per-island work
lists produced no reliable gain; distributing every color across GPU groups
increased impact median to 104 ms because dispatch overhead outweighed
parallelism. Those implementations were removed. Temporary stage probes
isolated about 9–13 ms of baseline contact preparation, falling to roughly
3–4 ms after scheduling changes. The final aggregate solve median is
44–46 ms and contact evaluation about 18 ms; stage medians do not sum to the
frame median. Further progress needs cheaper contact response or a scheduling
scheme that spreads a connected solve without thousands of dispatches. These
results do not establish a hardware limit or full CUDA trajectory parity.

## Metal steering and impact follow-up (Apple M4, 2026-10-08)

The next audit reproduced the reported slowdown through the public Metal API,
without rendering. Baseline is commit `353ae2b`; all runs use the same authored
386-body scene, 1/60 s timestep, four substeps, and kernel timestamps. The
interactive gallery was closed during these sequential runs.

| Workload | Before wall median | After wall median | Before p95 | After p95 |
| --- | ---: | ---: | ---: | ---: |
| Settled, sleeping disabled | 29.66 ms* | 25.01 ms | 31.16 ms* | 25.95 ms |
| Held steering, sleeping enabled | 31.10 ms | 1.42 ms | 39.43 ms | 6.94 ms |
| Ball impact, sleeping enabled | 159.25 ms | 89.30 ms | 191.40 ms | 102.94 ms |

\* Previous recorded quiet-scene run on the same baseline commit. Steering
and impact were measured again before and after this change. These are sample
timings on a fanless M4, not frame-rate guarantees or rendered-frame timings.
The final sleeping rest median is 1.23 ms. Impact contact generation fell from
83.20 to 19.72 ms and contact solving from 73.86 to 54.23 ms. Stage medians do
not sum to the whole-frame median.

The impact replay settles for 120 frames, places the authored 100 kg ball at
`(0, 1.05, 2.5)` moving toward the wall at 10 m/s, and measures the following
120 frames. It requires finite final positions and at least 16 independently
displaced bricks. The final run displaced all 384 bricks by more than 5 cm;
the solver refactor changes floating-point trajectories, so this is a physical
acceptance check rather than a claim of bitwise CUDA parity.

Retained engine changes:

- **Wake on changed net loads.** Repeated gravity-compensation forces no longer
  reset the quiet timer for every brick. The API tracks each body's effective
  acceleration and torque; impulses, changed loads, and state/resource edits
  still wake affected bodies. All 384 bricks stay asleep while the ball follows
  the steering gravity.
- **Cooperative narrow phase.** For unconstrained rigid worlds, bounded leaf
  products are evaluated by a 32-lane threadgroup. Each leaf pair uses the
  existing contact and continuous-collision routines. Lane zero merges the
  results in the original CUDA lane-grouped order. Products larger than 512
  candidates, missing leaf data, deep static sweeps, constrained worlds, and
  coupled systems retain the existing fallback.
- **Compact ordinary solver.** Persistent and transient contacts hold both
  body states locally and bypass hinge, compound, and constraint traversal.
  Contact budgets, friction/restitution formulas, timestep, geometry, and wall
  acceptance tolerances are retained. Overflow contacts cannot trigger the
  convergence shortcut without being included in its residual.

The unchanged wall gate passes with 0.00197983 m maximum drop, 0.00305891 m
maximum displacement, 0.000976562 rad maximum angle, and 0.000647023 m minimum
clearance. It now also checks sleeping under compensated steering and waking
under an impulse, changed gravity, or gradually changing force. The load
reference stays fixed during sleep, so small changes cannot drift unnoticed.
Memory with sleeping is 87,215,452 bytes.

Rejected probes: reducing the solve threadgroup from 256 to 32 lanes increased
impact median to 199 ms; independent per-island budgets/colors did not improve
the retained 89 ms result. Both were removed. The remaining dominant work is
the contact solve inside the connected impact island (54 ms median), followed
by narrow-phase evaluation (20 ms). Future work should improve utilization of
that connected solve and reuse contact response calculations while continuing
to pass the strict wall gate; previous prepared-response probes did not.

Reproduce all three workloads:

```bash
cmake --build build-metal-gallery --target parallel-mater-metal-rigid-scene-benchmark
./build-metal-gallery/parallel-mater-metal-rigid-scene-benchmark
./build-metal-gallery/parallel-mater-metal-rigid-scene-benchmark --scenario steering --mode sleep
./build-metal-gallery/parallel-mater-metal-rigid-scene-benchmark --scenario impact --mode sleep
```

## Metal expanded-wall contact scheduling (Apple M4, 2026-10-08)

The 386-body Metal opening scene previously invalidated its rigid-contact cache
at every frame boundary and scanned every active pair for every contact color
and solver pass. The retained path now preserves cache epochs across adjacent
ordinary-stack frames, invalidates them on public state/resource mutation,
and compacts colored work into adjacent lanes in the broad-phase flag scratch.
It also keeps the maximum requested cold-patch iteration budget instead of
allowing a later cached patch to reduce 64 passes back to 32.

A physics-only Apple M4 run measured 120 frames after 10 warm-up frames at
1/60 s, four substeps, with kernel timing enabled and no rendering/readback:

| Stage | Before | After |
| --- | ---: | ---: |
| Contact solve median | 96.07 ms | 29.56 ms |
| Contact evaluation median | 4.29 ms | 2.64 ms |
| Total GPU median | 113.01 ms | 46.08 ms |
| Step wall median | 113.43 ms | 46.57 ms |

The contact solve is 69% lower and total GPU time is 59% lower in this sample.
That was the intermediate retained-cache result. The completed path now also:

- stores manifolds and caches in triangular pair space and initializes only
  compacted active manifolds;
- generates contacts and saves caches through indirect active-pair dispatches;
- reconstructs validated convex face geometry and persistent colors from the
  preceding substep;
- caches ordinary-stack inertia inputs inside the two-body solver;
- partitions the contact graph into independent islands and dispatches one
  solver threadgroup per island;
- stops converged island iterations and optionally sleeps supported quiet
  islands; and
- renders rigid meshes from static vertex buffers with aligned 32-byte
  per-body instances.

The repeatable `parallel-mater-metal-rigid-scene-benchmark` measures the first
scene with kernel timing enabled. A final Apple M4 run produced:

| Phase | Wall median | GPU median | Solve median |
| --- | ---: | ---: | ---: |
| Historical retained-cache result | 46.57 ms | 46.08 ms | 29.56 ms |
| Current, sleeping disabled, settled | 29.66 ms | 29.22 ms | 25.56 ms |
| Current, sleeping enabled, startup | 29.37 ms | 28.97 ms | 25.95 ms |
| Current, sleeping enabled, settled | 1.25 ms | 0.88 ms | 0.27 ms |

All 385 dynamic bricks sleep in the settled sample. Allocated bytes fell from
144,485,608 in the historical run to 87,140,640 with sleeping enabled. The
strict wall gate now passes with 0.00197983 m maximum drop, 0.00305353 m
maximum displacement, 0.000976562 rad maximum angle, and 0.000646994 m minimum
clearance. Prepared response and first-fit coloring remain excluded because
the earlier probes destabilized this expanded wall.

The following audit, timing numbers, and prior passing wall replays concern
the earlier 96-brick asset. They are not acceptance results for the current
384-brick wall. CUDA golden reproduction also retains the six differences
listed below; no goldens or numerical tolerances have been changed.

## Rigid brick wall audit (RTX 3050 Ti Laptop GPU, 2026-10-04)

The 96 independent bricks in `RigidBody.glb` exposed the cost of the new
zero-gap, warm-started convex contact solver. The pre-optimization baseline
spent 91.8% of GPU kernel time in the small-world contact solver (Nsight
Systems), with a 44.25 ms physics median and 40.64 ms contact-solve median.
This baseline already includes the wall-support correctness repair; it is
not a comparison against the earlier collapsing-wall solver.

Retained changes:

- Pack body-disjoint contact colors into adjacent lanes. Ordinary convex
  stacks with 32–256 bodies use deterministic first-fit colors to reduce
  dependent batches; small, constrained, and coupled worlds retain their
  previous coloring and contact response.
- Prepare contact arms, inertia responses, effective masses, and material
  terms once per substep. Keep each pair's states local while processing its
  rows. Friction uses the same sliding-direction response expressed in its
  two-dimensional tangent plane. Contact Jacobians stay fixed during the
  substep's velocity solve; positional residuals still update every pass.
  This prepared-response path has the same ordinary-stack restriction;
  applying it globally caused out-of-tolerance numerical changes in analytic
  and coupled cases and was rejected.
- Run 64 passes when an ordinary-stack face patch has no matching cached
  contact, then return to the normal 32-pass steady-state budget. This gives a
  newly formed deep stack enough time to converge without doubling every
  settled frame. The authored wall stays below 1.2 mm displacement and 0.01
  radians of rotation for ten seconds; impact still moves a brick.
- Generate eligible convex face patches before triangle evaluation, skipping
  duplicate triangle work for handled pairs. Swept, curved, concave, and
  overflow contacts retain their general path.
- Publish prepared patches' accumulated support/friction impulses once after
  solving, including warm-start impulses. Other cases retain their diagnostic
  summation order. Gather capture samples on the GPU and copy them in a
  batch, avoiding CPU migration of live rigid-state pages every frame.
- Bound interactive catch-up by estimated work as well as step count: at most
  four steps, with an 8 ms budget for extra physics work. A first due step
  always runs; overload drops excess backlog, so this is a responsiveness
  guard, not a claim that over-budget physics remains real-time.
- Apply the Rigid Body scene's camera-relative tilt to its controllable sphere,
  while one batched central-acceleration command keeps all 96 authored Array
  bricks under vertical gravity. This changes only the gallery control field;
  bricks remain dynamic and the contact solver still handles sphere impacts.

The stable-wall profiled physics median is 9.28 ms, with 7.69 ms in contact
solving and 1.20 ms in narrow-phase evaluation. Both before and after
measurements had another interactive gallery process on the same GPU. These are useful
audit observations, not isolated throughput or p99 acceptance measurements.
These runs do not meet the 8 ms physics p99 / 16.7 ms complete-frame p99
targets; isolated acceptance remains pending.
No sleeping, welded bricks, lowered substeps, reduced contact passes, altered
authored geometry, or relaxed regression tolerances produce these gains.

Reproduce physics-only stage timings with
`parallel-mater-rigid-scene-benchmark examples/assets/RigidBody.glb`.
It settles 360 frames, warms 10, then measures 120 frames at 1/60 s with four
substeps and kernel profiling enabled. For gallery-like frame measurements:

```sh
./build-gallery/parallel-mater-rigid-frame-benchmark examples/assets/RigidBody.glb --csv /tmp/wall-rest.csv
./build-gallery/parallel-mater-rigid-frame-benchmark examples/assets/RigidBody.glb --impact --csv /tmp/wall-impact.csv
./build-gallery/parallel-mater-rigid-frame-benchmark examples/assets/RigidBody.glb --tilt --csv /tmp/wall-tilt.csv
```

This second benchmark disables kernel profiling, enables the gallery's
30-frame capture ring, and renders/readbacks at 960×720. It reports separate
physics, render, and total median/p95/p99/worst times over 240 frames after
120 settling and 10 warmup frames. Window upload/presentation and debug
overlays are excluded. The impact workload must record an actual sphere–brick
impulse; every measured rigid state must remain finite. Close other GPU
applications before using either benchmark for acceptance.

Observed full-frame results with that other gallery still running (milliseconds;
diagnostic only, not an isolated acceptance pass):

| Workload | Physics median | Render median | Total median | Total p95 | Total p99 | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rest | 10.91 | 0.98 | 11.94 | 12.29 | 24.92 | 25.97 |
| Sphere impact | 12.22 | 0.93 | 13.16 | 14.58 | 26.16 | 27.83 |
| 20° screen-space tilt | 11.72 | 0.93 | 12.69 | 19.16 | 24.50 | 28.81 |

The sphere-impact check confirmed a nonzero sphere–brick contact impulse.
All three runs retained finite rigid states. Raw local CSVs are
`/tmp/pm-wall-stable-{rest,impact,tilt}.csv`; repeat the commands above to
regenerate them. The 12 ms medians must not be presented as sustained 60 FPS:
the observed tails exceed the frame budget, and presentation is excluded.

After adding the authored vertical-gravity override for the 96 bricks, an
isolated rerun of the corrected 20-degree ball-only tilt workload measured
11.06 ms physics median / 12.19 ms p99 and 12.33 ms complete-frame median /
13.49 ms p99, with a 13.55 ms worst frame. Its raw CSV is
`/tmp/pm-wall-untilted-bricks.csv`. The complete-frame target passed in this
run; the 8 ms physics p99 target did not.

`parallel-mater-rigid-wall-tests` checks first-frame coplanar support, ten
seconds of all-brick authored wall stability, a 30-degree control tilt that
moves the sphere without rotating or translating the wall, floor clearance,
energy, independent impact response, cache invalidation, and support-impulse
diagnostics. `parallel-mater-physics-frame-budget-tests`
checks catch-up behavior without CUDA. Golden results and conformance
tolerances are never refreshed by these benchmarks.

Full Release build and CTest: 101/102 passed. The sole failure is CUDA golden
reproduction, with the same six case IDs already differing after the earlier
contact-correctness repair: `compound-weld-lifecycle`, `constraint-breaking`,
`constraint-generic-spring`, `constraint-generic`, `fluid-rigid`, and
`rigid-direct`. These numerical/contact differences require explicit review;
golden JSON and tolerances remain unchanged.
After restricting the fast path, all 30 cases match the pre-performance,
wall-correctness candidate using the existing comparator and tolerances.
That comparison does not approve or replace the committed goldens: their
six pre-existing failures still need review. The latest local comparison
report is `/tmp/pm-wall-stable-vs-correctness.json`.
All ten affected rigid, wall, hit-box, collector, joint, color-limit,
fluid–rigid, capture, and frame-budget tests passed again on the final scoped
build (`/tmp/pm-wall-performance-final-regressions.log`).
Compute Sanitizer `memcheck` and `racecheck` also pass a one-snapshot wall
replay (four substeps, warmup plus measured replay): zero memory errors and
zero shared-memory hazards. This is a targeted kernel check, not exhaustive
coverage of every scene or global-memory race.

## Dense soft-body and rope coupling (RTX 3050 Ti Laptop GPU, 2026-10-01)

Three isolated Release runs of
`parallel-mater-rope-soft-body-tests 180 --pull` measured a median 30.39
ms/frame before the dense-graph work and 9.53 ms/frame after it, a 68.6%
reduction (3.19x throughput). The soft-body constraint stage fell from 24.35
to 4.39 ms/frame, an 82.0% reduction. The fixture has 912 soft nodes, 130,033
bonds, 1,792 skin triangles, 33 pins, and a 72-node attached rope. Kernel
timings include its eight effective substeps and exclude rendering.

The retained dense path stores an 8-byte CSR neighbor descriptor and one
precomputed minimum rest length per node. Sixteen-thread warp subgroups
evaluate spring terms concurrently; each subgroup leader folds shared-memory
batches in original CSR order. The same ordered scheme accelerates spring
damping. Shape matching coalesces raw node data into shared-memory batches,
then preserves the original serial floating-point reduction. Short-run output
remained identical across the final scheduling and staging changes, and the
1,000-frame rope-release stability test passed.

Rejected measurements include a parallel cyclic-reduction rope solve (+8.3%),
distributed rope contact scans (+26.8%), eight warp threads per soft node
(+3.3%), eight warps per block (+2.9%), and a full shared node cache (+2.9%).
Unordered tree reductions were faster but failed the long release test, so
they were removed. Warp-aggregated rope atomics, adaptive rope block sizing,
and shared rope self-collision positions were neutral or slower and were also
removed. The final source has no rope-kernel change; its useful gain in this
fixture comes from the attached soft-body work. All 38 rope/soft-body CTest
cases passed in 326.72 seconds.

## Staggered hybrid smoke solver (RTX 3050 Ti Laptop GPU, 2026-10-01)

The current implementation replaces the prototype with a 128×32×128 MAC
grid, RK2 monotonic MacCormack face advection, thin triangle cut faces,
Smagorinsky LES viscosity, bounded curl restoration, and a four-level
geometric multigrid projection. GPU regression output now includes pressure
relative residual, normalized post-projection divergence, front/rear pressure,
side speed, lee recirculation, and lee enstrophy against an unobstructed
control. Three isolated 120-frame runs on the RTX 3050 Ti measured 5.73,
5.74, and 5.61 ms/frame, meeting the 6 ms target. These are physics-only wall
times and exclude rendering.

The final 524,288-cell field had an 8.00e-4 relative pressure residual and
1.40e-4 normalized post-projection divergence. Around the triangle sphere,
mean front pressure was 0.365 versus -0.323 behind it, mean sampled side speed
was 1.63 m/s, 470 lee cells recirculated, and mean lee enstrophy was 41.16
versus 0.238 in the unobstructed control. No sphere-specific flow kernel or
prescribed wake participates in those measurements. Fixed-point quadratic
B-spline deposition made the grid and tracer replay bit deterministic without
moving the solver over budget.

The same GPU acceptance run verifies zero pre-arrival reaction and local
density scaling: the dense plume moved the 1 kg rigid body 0.206 m relative to
its containment-only control, versus 0.0452 m for the sparse plume. The
smoke-water, 20-post soft-body, cloth, and rope coupling regressions all pass
with grid-mode stress and tracer containment enabled.

## Former cell-centered hybrid baseline (RTX 3050 Ti Laptop GPU, 2026-10-01)

The former prototype used 128×32×128 cells (524,288) with uniform cell
spacing. In the sphere scene, 120 frames with 24 Jacobi pressure passes
averaged about 2.8 ms per physics frame; 692 cells represented the rigid triangle
surface, 2,941 carried visible-density smoke, and projected lateral airflow
was measurable before, beside, and behind the sphere (0.334, 0.179, and
0.172 m/s in the sampled regions). The isolated soft-body comparison averaged
7.8 ms per pair of coupled/reference frames, produced 3.79 m of summed
X-position difference over its 176 nodes after 120 frames, and had 1.28%
maximum bond stretch. The authored 20-post scene measured 33.6 m of summed
X-position difference across 3,520 nodes at 180 frames, with 21.4% maximum
bond stretch versus 23.9% for its uncoupled reference. These timings exclude
rendering. The smoke sphere,
water, soft-body, cloth, and rope GPU regressions pass with the gallery grid
enabled. The particle-only API remains available by setting grid resolution
to zero. This is a cell-centered projection prototype, not a verified
high-fidelity incompressible solver.

Sizing prototypes on the same GPU measured 116³ (1.56 million cells) at about
7.8 ms/frame and 256³ (16.78 million cells) at about 91 ms/frame for the
sphere scene. The 128×32×128 choice preserves roughly the horizontal spacing
of the larger grid while avoiding air cells far above and below the scene.

## Smoke plume and contact tuning (RTX 3050 Ti Laptop GPU, 2026-09-30)

With the 4,500-slot, 900-particle/s inlet and the sphere uncoupled, mature
smoke now averages 1.60 m/s forward and 0.165 m/s transverse at frame 300.
The prior calibration measured 1.49 and 1.33 m/s respectively: pressure was
scattering the unobstructed plume. Raising the rest number-density threshold
to 12, reducing the pressure stiffness to 2 and vorticity confinement to 0.1,
and relaxing particles toward the inlet flow at 0.5/s preserved forward
motion while leaving pressure active at contacts.

Near the sphere, mean particle speed is 1.00 m/s versus 1.35 m/s farther
downstream; the previous wall response almost froze particles at 0.0037 m/s.
At full emission rate, a dynamic sphere moved 0.389 m forward and -0.002 m
vertically relative to a contact-only reference over 90 zero-gravity frames.
With the authored floor and 1 kg active ball, the 180-frame smoke reaction
changed ball height by -0.003 m. Contact pressure now comes from blocked
normal speed; the former direct pressure-release impulse on the rigid body
was removed.

For the two-sided cloth, the pressure-aware finite-sheet boundary clears
797 particles laterally downstream at frame 300 with 461 in the broad
windward region. Isolated coupled physics averaged about 4.2 ms/frame versus
3.8 ms/frame for the reference. The 20-post soft-body test averaged
29.9 ms/frame in isolation with 45.3% maximum bond stretch. All ten smoke
GPU/headless regressions pass. This remains a weakly compressible particle
approximation with a finite-sheet boundary rule, not an incompressible
pressure projection.

## Previous local smoke calibration (RTX 3050 Ti Laptop GPU, 2026-09-30)

The 4,500-slot Smoke regression now uses sorted particle neighbors, local
number-density pressure, viscosity, and measured-vorticity confinement.
Compared with identical particle flow without obstacle contact, 2,414 lee
particles differ by 0.094 m/s in mean transverse velocity. Maximum sampled
pressure was 21.7 in solver units. Mean lee-region curl was 0.358 versus
0.222 for the unobstructed control, measured from neighboring velocities.
Of 170 particles
within 0.1 m of the sphere surface, mean speed was 0.0037 m/s versus the
1.6 m/s inlet speed. A 90-frame dynamic-sphere test moved 0.232 m farther
with particle reaction than its contact-only zero-mass reference; a lower
particle emission rate produced less push. Pre-emission and distant-body
controls measured no smoke force.

The 20-post SmokeSoftbody test averaged 27.9 ms/frame at 180 frames, with
maximum bond stretch 28.8%; this improves on the immediately preceding
triangle-obstacle prototype at about 30 ms/frame but remains slower than the
older prescribed-field baseline at about 21 ms/frame. The 300-frame
SmokeCloth test averaged 3.37 ms/frame coupled and 3.03 ms/frame for its
reference after removing the old scripted edgeward contact speed, excluding
rendering. The new gas model is weakly compressible,
not a pressure-projected incompressible solve; deformable contacts still lack
equal-and-opposite particle reactions.

## Smoke triangle-obstacle update (RTX 3050 Ti Laptop GPU, 2026-09-30)

The smoke obstacle now uses a triangle-mesh carrier deflection and swept
tracer/triangle contact instead of analytic sphere flow and collision. A
non-spherical box regression confirms both contact and wake. The 20-post
SmokeSoftbody scene remains stable, but its measured physics step increased
from about 21 to 30 ms/frame at 180 frames; this is a known cost of the
former mesh-guided path and needed profiling before further optimization.
The local-particle update above supersedes that prescribed-field path.

## Shared smoke force and soft-body wind (RTX 3050 Ti Laptop GPU, 2026-09-30)

The closed 1 kg sphere moves 1.45 m farther along the carrier wind than an
identical uncoupled sphere after 90 frames with no gravity. In SmokeRope,
the suspended panel differs by 0.31 m after 300 frames while final rope
strain remains below 0.1%; coupled physics costs about 7 ms/frame versus
2.8 ms/frame without smoke couplings. For SmokeSoftbody's 20 posts, reducing
the shared soft-body wind default from 2.0 to 0.5 inverse seconds and capping
wind acceleration at 2 m/s² lowered peak bond stretch from 85% to 41% at
180 frames. Wind still displaces the posts, and the physics step remains about
21 ms/frame. These measurements exclude rendering.

## Smoke–rope suspended panel (RTX 3050 Ti Laptop GPU, 2026-09-30)

`SmokeRope.blend` exports four roughly 40-node ropes between an active panel
and two passive posts. At 300 frames of 1/60 s with four requested substeps
(the ropes raise the shared step count to 8), the coupled scene averaged
4.43 ms/frame versus 2.98 ms for the same scene with smoke couplings removed.
The panel center differed by 0.145 m, summed rope-node positions by 12.2 m,
and summed tracer positions by 1,749 m. Peak rope strain at the final frame
was 0.04%. A focused emitter aimed at a rope produced 88.1 m of summed
tracer-path divergence; the authored narrow plume instead passes between the
four corner ropes and interacts mainly with the suspended panel. These are
physics step wall times, excluding rendering and readback.

## Smoke–cloth sheet (RTX 3050 Ti Laptop GPU, 2026-09-30)

`SmokeCloth.blend` exports one 289-vertex, 512-triangle sheet with 34 pinned
vertices. With 1/60 s frames, four substeps, and zero gravity to isolate wind,
the 180-frame GPU comparison averaged 1.57 ms/step coupled versus 1.51 ms
without the API coupling. All pins stayed exact; maximum cloth speed was
0.21 m/s and maximum bond strain 2.5%. The moving cloth differed from the
uncoupled reference by 27.8 m of summed vertex displacement; tracer paths
differed by 149 m summed across the occupied smoke slots. These are physics
step wall times, not render FPS or conserved two-way momentum measurements.
A separate 6 m/s tracer test confirms that swept triangle contact prevents
one-frame tunneling through the thin sheet.

In the five-second follow-up (300 frames), the original normal-only contact
left 599 live tracers in the upstream surface region and none beyond the
sheet. Tangential edge flow reduced that region to 479 and placed 1,002
tracers beyond the sheet, at 1.74 ms coupled versus 1.66 ms uncoupled per
frame. A wider pre-contact steering zone increased the upstream count to
653, so it was discarded. Counts use fixed regions around the authored sheet
and measure distribution, not total mass flux or render time.

## Smoke–soft-body grid (RTX 3050 Ti Laptop GPU, 2026-09-30)

The 20 authored cylinders originally expanded to 56,280 nodes, 4,791,300
springs, and 105,680 skin triangles. Rebuilding their mesh at 16 radial sides
and nine axial rings, with 0.12 m node spacing, yields 3,520 nodes, 57,720
springs, and 5,760 skin triangles while retaining 340 exact Goal pins.
No solver specialization was added; the shared smoke and soft-body APIs own
all forces and contacts.

Release GPU measurements use 1/60 s frames and four substeps, excluding
rendering. The first 30 frames took 42.3 ms/frame at the default 16 graph
iterations, 22.7 ms at six, and 18.8 ms at four. At four iterations, the
180-frame test averaged 18.0 ms/frame, retained every pin, and measured
11.8% maximum spring strain and 0.50 m/s maximum soft-node speed at the end.
An uncoupled reference diverged by 474 m of summed smoke-tracer positions
after 180 frames, confirming that the authored plume actually meets the
posts. A focused one-post comparison also verifies both wind bending and
tracer deflection. These figures are physics step wall time, not render FPS.

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

## Hinge gear capture latency, 2026-10-03

The historical four-body ConstraintHinge capture covering frames 743–772, made
before the authored scene expanded to three gears, reproduced the
reported roughly 50 ms physics frames. Dense gear leaf pairs exceeded the
512-entry contact cache and used the serial BVH fallback. The other costs
were repeated leaf-bound transforms, 64-bit division in the Cartesian leaf
scan, and geometry recalculation in all 64 hinge solver iterations.

On the RTX 3050 Ti Laptop GPU, Release build, a sequential before/after run
repeated those 30 snapshots three times after GPU warmup. Each sample restores
the captured post-step body states and advances one frame with eight substeps,
recorded gravity, and zero new input forces. Rendering is excluded; this is
a snapshot workload, not a reconstruction of the user's input history.

| GPU stage/statistic | Before | After |
|---|---:|---:|
| Total, median | 15.20 ms | 5.73 ms |
| Total, p95 | 44.00 ms | 6.90 ms |
| Total, maximum | 46.95 ms | 7.41 ms |
| Leaf candidates, median | 4.07 ms | 1.47 ms |
| Triangle contacts, median | 6.66 ms | 1.44 ms |
| Contact/constraint solve, median | 3.90 ms | 2.63 ms |

Worlds with rigid-body capacity at most eight now reserve 4,096 leaf candidates
per pair and parallelize each pair across 16 blocks. Per-candidate manifolds
are reduced in their original order. Shared bounds avoid repeated transforms;
a strided cursor eliminates per-candidate 64-bit division without changing
candidate order. Constraint axes, anchors, and effective masses are prepared
once per substep, with local velocity states published in constraint order.
The eight substeps, 64 hinge iterations, exact triangles, swept collision
checks, rest offset, and contact correction rules are unchanged.

The four-body scene uses approximately 6.4 MiB more cache/scratch; the extra
allocation is bounded at approximately 30 MiB for eight-body capacity. Larger
worlds retain the smaller cache and one-block-per-pair evaluation. Both paths
retain swept BVH fallback if their cache overflows. Expanding the cache changes
which reduction path handles previously overflowing pairs; the subsequent
parallel and solver optimizations preserved the expanded-cache replay's body
states bit-for-bit (`8fcc8220dd54e5f5` for the 90-sample workload).

The standard PassiveActive benchmark measured 1.40 ms median GPU time, versus
2.43 ms after the preceding rigid-contact repair. Rigid tests, both cache-size
overflow fixtures, 1,200 natural hinge frames plus driven full rotation,
1,200-frame DUMP containment, and 1,000-frame FluidRigid containment passed.

New captures made with `m` can be profiled with:

```sh
./build-gallery/parallel-mater-rigid-capture-benchmark \
  examples/assets/ConstraintHinge.glb capture.log 8 5
```

The last arguments select substeps and measured repetitions. One additional
unmeasured capture pass warms the GPU. The benchmark accepts matching rigid-only
version 1 captures and reports stage percentiles, worst snapshot, and a body-state
hash. Different repetition counts produce different hashes.

## Fixed collector ground-contact capture, 2026-10-03

A 30-frame Fixed capture at frames 4,380–4,409 exposed a three-frame support
cycle. `Small.022`, fixed into the large collector and resting on Ground,
alternated between 2–5 contact points. Intersecting triangle pairs reported the
entire `0.01201 m` combined search margin as penetration. Independent position
correction moved that small sphere away from its fixed joint; the joint then
pulled it back at up to `2.25 m/s`, while asymmetric ground friction kept the
cluster rotating near `0.118 rad/s`.

The first repair estimated intersection depth from body vertices behind the
contacted triangle plane and changed fixed members to velocity recovery. A
100:1-mass regression fixed one large body to three ground supports and checked
settling speeds. That removed the visible oscillation but did not verify floor
clearance.

The later capture at frames 373–402 exposed the missing load transfer: body 43
had its center at `y=-0.0516 m`, while loose marbles rested near `y=0.0904 m`.
Ground impulses were resolved before joint impulses, so the heavy parent could
pull its light supports downward again. The speed-only regression also passed
with support corners `0.3993 m` below its plane.

Fixed contacts and joint rows now iterate together through eight contact
sweeps, and penetration recovery uses measured depth instead of a 1 mm cap.
Free welded groups receive residual position correction as a shared translation
against immovable surfaces; member offsets and velocities are preserved. Groups
anchored by other joint types are excluded from that translation.

The strengthened 100:1 regression checks actual rotated-box clearance as well
as settling speeds. A separate 620-frame gallery regression rolls, releases
input, turns, and settles while checking every collision-mesh vertex against an
extended test floor. Eight bodies were collected, with minimum clearance
`0.000394 m` above the floor and no escapes. The rigid-body, API contract,
collector-clearance, and Fixed headless tests pass on the RTX 3050 Ti.
