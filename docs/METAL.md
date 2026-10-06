# Metal 4 port status

The Metal backend is additive. The CUDA API, `ParallelMater::parallel_mater`
target, OptiX gallery, and CUDA implementation remain unchanged when
`PARALLEL_MATER_BUILD_CUDA=ON`.

## Implemented foundation

- CMake starts as C++ and enables CUDA or Objective-C++ only for requested
  backends. Apple builds default to Metal; non-Apple builds default to CUDA.
- `<parallel_mater/types.hpp>` is the framework-free layout boundary for
  vectors, handles, backend-neutral options, events, timings, statistics, and
  host geometry/debug values. `<parallel_mater/metal.hpp>` aliases those exact
  types and compiles without CUDA headers.
- `ParallelMater::metal` embeds its MSL 4 metallib into the static archive, so
  installed consumers have no runtime shader path.
- `World` owns a Metal 4 queue, reusable command allocator and command buffer,
  argument table, residency set, shared completion event, and commit feedback.
- `FrameToken` provides asynchronous completion and propagates Metal command
  errors. One frame per world may be in flight. Completion state and commit
  options are allocated with the token, completed tokens are reusable, and
  synchronous stepping uses a token owned by the world.
- Public rigid and particle buffers have fixed capacity and shared storage.
  Stepping performs no application allocation or CPU wait. Tearable-cloth
  topology scratch and every enabled physics-debug ring payload are reserved
  at world/system creation, including worst-case physical cloth splits.
- Rigid, fluid, smoke, and paint views expose the same world-wide revision
  contract as CUDA. Successful topology/resource mutations and completed
  frames advance it once; borrowed views reject access while a frame is in
  flight and must be reacquired afterward.
- Host or shared/private `MTLBuffer` triangle meshes and generation-checked
  rigid bodies support create, remove, state/target updates, forces, impulses,
  views, readback, statistics, and GPU integration.
- The first rigid-context gate has deterministic swept world bounds,
  dynamic-body-first pair filtering, row-parallel stable broad-phase
  compaction, CPU-built four-triangle-leaf BVHs, pair-parallel exact triangle
  manifolds, and swept conservative advancement. Contacts use up to 24
  deterministic conflict-free colors for parallel eight-pass solving, with a
  stable serial overflow path.
  It runs the committed `RigidBody.glb` wall scene on Apple Silicon.
- Static, kinematic, and dynamic integration follows the CUDA phase semantics:
  static velocities clear, kinematic targets interpolate across substeps with
  derived velocity, damping uses the same rational form, and torque/impulse
  response transforms local inverse inertia into world space.
- Generation-checked fixed, point, hinge, slider, piston, generic,
  generic-spring, and motor constraints use the CUDA Jacobian, world-inertia,
  Baumgarte, limit, spring, damping, bounded-motor, accumulated-breakage, and
  per-constraint iteration equations in stable order. Their Metal resource
  contract also includes runtime updates and collision suppression.
- Correctness-first Metal kernels now cover deterministic particle-fluid
  density/neighbor forces and foam state through a stable eight-pass 64-bit
  cell-key sort and 27-cell traversal. The sort generates keys in parallel and
  uses fixed 256-element blocks, per-block histograms, parallel per-bucket
  scans, and stable block scatter; only the 256 bucket totals are scanned by a
  single lane. CUDA-style per-fluid solver iterations preserve
  sort/force/coupling/integration order; cloth force and containment projection,
  soft-body passes, rope/rigid contacts, and stable outflow compaction execute
  in each fluid microstep. GPU-spawned particles participate
  in those phases in the same submitted frame. A GPU-side count snapshot taken
  immediately before emission identifies that frame's spawned range without a
  CPU wait or a shared-memory race. The kernels also cover source
  emission once per frame (independent of world substeps), with a hard
  capacity-bounded commit. Matching CUDA's frame scheduler, the rigid and
  deformable substeps finish before source emission and the contiguous fluid
  block; each fluid runs `world substeps * fluid solver iterations` microsteps
  over the full frame timestep instead of being interleaved with deformables.
  Source sites retain authored commit order while each occupancy search runs
  cooperatively across 64 lanes. Destroy planes compact every particle field
  in stable 64-element chunks without scratch; a 257-particle gate verifies
  that retained stable IDs and lifetime counts survive chunk boundaries.
  The kernels also cover cloth stretch/bending
  constraints with deterministic per-node Jacobi gathers, persistent strain
  and per-bond rigid-impact fracture, idle-boundary physical fan splitting,
  source mapping, and split-mass preservation. Cloth volume and soft-body
  best-fit shape matching use deterministic cooperative threadgroup
  reductions; soft-body spring projection is per node. Direct rope stretch
  solves include dynamic rigid endpoint
  reaction, and CUDA-style pre-sampled cloth/soft endpoint anchors included in
  the tridiagonal endpoint mass solve before reactions are scattered back.
  The two endpoint samples execute independently, and rope-soft packed surface,
  binding, velocity, inverse-mass, and index preparation is striped across 64
  lanes. Same-target endpoint application and the tridiagonal solve retain
  CUDA's dependency order.
  Rope-rigid swept-node and alternating capsule-segment contacts now execute
  between nonlinear stretch projections, use contact-reduced mass directions,
  release unilateral contact planes through CUDA's four-pass active set,
  preserve pre-integration rigid-anchor positions for kinematic sweeps, use
  the eight-pass contact-free stretch recovery when nonlinear iterations are
  exhausted, accumulate body movement per phase, and retain frame-scoped
  constraint and contact forces. Up to two rope-soft targets are packed into fixed coupling
  storage before the solve; their swept-node and capsule contacts execute in
  the same nonlinear phases, accumulate deterministic lattice reactions, and
  apply the bounded soft-body load afterward without a CPU wait.
  Rope anchor snapshots, contact-plane reset, gravity/damping integration, and
  free-node prediction run per node in parallel before the deterministic
  tridiagonal/contact solve.
  Particle-smoke density, vorticity, force, and integration phases run one
  thread per tracer with deterministic phase barriers; projected-grid tracer
  advection is likewise particle-parallel. Particle-smoke wind on cloth,
  soft-body, and rope nodes is target-parallel, while swept tracer contact
  with those surfaces is tracer-parallel. The kernels also cover projected
  grid smoke.
- Every exposed fluid/smoke/cloth/soft-body/rope pair handle has validated
  lifecycle and an encoded coupling phase. This includes heated liquid-to-smoke
  conversion, particle or projected-grid smoke wind applied to nearby liquid,
  two-way fluid-rope capsule response using the fluid's physical particle mass
  and frame-rate force diagnostics, smoke-rope wind and tracer deflection
  with swept capsule contact, bidirectional rope-cloth joints, and rope-soft
  endpoint anchors.
  Dedicated surface kernels enforce closed-triangle fluid-cloth containment,
  barycentrically distribute fluid-soft reactions over the skinned lattice,
  solve soft-node contact against current (including tear-safe) cloth
  triangles, and sweep smoke tracers against cloth and soft skins while local
  smoke velocity bends their free nodes. Rope-soft contact now tests capsule
  segments against the current skinned triangle surface; optional endpoints
  retain their authored rest-surface offsets and distribute reactions over
  nearby lattice nodes with a barycentric fallback. Fluid, cloth, soft-body,
  and rope nodes also contact live rigid triangle meshes. Shared particle-rigid
  contact now follows CUDA's old/current body transforms for swept open-triangle
  hits, uses authored fluid particle mass and angular effective mass, batches
  simultaneous reactions against one body, and applies rigid momentum only
  after deterministic reduction. Late fluid and smoke sweeps read a dedicated
  frame-start rigid transform snapshot, while deformable contacts retain the
  previous transform for their own rigid substep. First-iteration fluid sweep
  constants also have separate fixed storage, so later encoded microsteps
  cannot overwrite the command before GPU execution. Contact then resolves
  every static boundary in stable body order, so a moving wall cannot suppress
  a simultaneous floor or containment contact; moving-body contact events
  retain CUDA's priority.
  Fluid-rope, fluid-cloth, fluid-soft-body, and soft-body-cloth reactions use
  particle-owned contribution records followed by stable node-order gathers;
  their floating-point results never depend on atomic arrival order. Rope-soft
  lattice scatter and endpoint refresh are node/surface parallel after the
  ordered rope solve. Projected-grid smoke loads gather independently per
  deformable node. Liquid-smoke wind and heating run per liquid particle before
  a cooperative boiling/compaction commit. Stable hot-particle ranks choose
  smoke-ring slots; excess hot water stays live in original stable-ID order. A
  257-particle, 80-slot gate covers multi-chunk conversion and retention.
  Closed-convex rigid planes are uploaded with the flattened mesh data so
  trapped soft nodes recover along an outward face instead of selecting an
  inward nearest-triangle normal. Dynamic soft-rigid overlap correction is
  split by effective inverse mass and reduced back into rigid translation.
  Rigid-cloth contact also covers the inverse geometry direction: each dynamic
  body's swept conservative mesh sphere tests cloth triangle interiors, shares
  positional correction with the contacted face, and applies bounded normal
  and friction response to the surrounding movable cloth nodes. Fracturing
  cloth retains CUDA's node-contact-only response while its public split
  surface is refreshed after rigid contact.
  Smoke-rigid tracer contact now sweeps previous/current tracer positions
  against old/current body transforms, preserves near-wall drag and pressure,
  and reduces equal-and-opposite reaction using the CUDA tracer-volume mass;
  grid-only coupling remains independent of tracer contact. Frame-scoped public
  coupling diagnostics
  accumulate rigid, fluid, cloth, soft-body, and rope reactions across all
  substeps instead of exposing only the final substep.
- Fluid neighbor limits use M1-safe 32-bit GPU reductions. The exact maximum
  reached by the solver feeds statistics/debug capture, and overflow is
  snapshotted per asynchronous frame and returned by `FrameToken::wait()` as
  `capacity_exceeded`. Fluid-soft, fluid-rope, and rope-soft contact proposal
  counts and maximum pre-correction penetration now come directly from their
  coupling kernels.
- Fluid lifetime statistics match CUDA accounting: emitted, destroyed
  (including boiled), boiled, and source-capacity-miss totals survive fluid
  removal without GPU 64-bit atomics.
- Grid smoke owns a persistent staggered face-velocity field, performs RK2
  MacCormack self-advection with a local extrema clamp, and applies the exposed
  kinematic-viscosity, LES, vorticity-confinement, and gravity-relative
  buoyancy coefficients before projection. Boiled fluid contributes decaying
  thermal lift through the same density-weighted grid field. Particle smoke
  uses the configured gravity-relative up direction as well. The public
  pressure residual is the actual infinity-norm Poisson residual relative to
  the pre-projection RHS. Pressure uses CUDA's four-level aperture-weighted
  V-cycle: two weighted-Jacobi pre/post sweeps, twelve coarsest-level sweeps,
  residual restriction, damped prolongation, and tolerance-based early exit.
  Its fixed scratch aliases the completed face-advection ping-pong storage.
  Public cell velocities/divergence are rebuilt from the projected face field;
  additional authored work is gated to reduce both residual and divergence,
  and identical smoke runs produce byte-identical grid captures. Matching
  CUDA's frame phases, obstacle rasterization and grid projection run once at
  the start of a frame before deformable substeps. Cloth, soft-body, and rope
  wind then sample that projected field on every deformable substep. Fluid-to-
  smoke phase transfer runs once after fluid work, tracer advection runs once
  after that transfer, and tracer contacts run against the final deformable
  state after advection. Authored smoke emission is last: fresh particles
  retain age zero and their emitter position until the following frame.
  Fractional emission uses CUDA's host-side double-intermediate schedule,
  skips zero-count GPU dispatches, and preserves the same slot/serial hash
  sequence. One cooperative group emits requested slots in parallel, then
  commits ring metadata after a device barrier.
  Grid cell/face initialization and obstacle-state clearing run in parallel.
  Particle deposition emits the fixed 27-point quadratic stencil in parallel,
  stable-sorts contributions by cell with four radix passes, then performs one
  local 64-bit reduction per cell. This preserves CUDA's 24-bit fixed-point
  contribution semantics without requiring unavailable 64-bit Metal atomics.
  A 257-particle oracle spans multiple radix blocks and checks every cell.
  Rigid, cloth, and soft-body obstacle triangles rasterize in parallel. A
  first pass atomically minimizes positive cut-face apertures and stable global
  triangle IDs; ordered selection/resolution passes then write exactly one
  wall velocity and normal per occupied cell/face. This removes overlapping
  triangle write races while retaining fixed storage and deterministic
  tie-breaking. The four-level pressure V-cycle dispatches aperture
  restriction, weighted-Jacobi smoothing, residual restriction, coarse-grid
  clearing, damped prolongation, and final residual measurement across all
  cells/faces in parallel. A 32-bit GPU convergence flag skips later authored
  cycles without a CPU readback; residual maxima use positive-float atomic
  ordering and remain deterministic on M1/M2-class devices. The remaining
  staggered-grid phases are parallel as well: domain marking, forward/reverse
  RK2 MacCormack advection, limiter correction, CUDA-shaped cell diagnostics,
  subgrid/buoyancy/viscosity forces, boundary blending, divergence, pressure
  projection, and public cell reconstruction each dispatch one thread per
  face or cell. Fixed advection storage becomes pressure scratch only after
  its final force consumer, so the parallel schedule adds no stepping-time
  allocation or CPU wait. Fully static rigid-only boundaries retain their
  nearest-triangle face metadata between frames while rebuilding apertures;
  this skips the repeated selection pass and is invalidated by coupling or
  explicit rigid-state changes, matching CUDA's cache contract.
  Enabled smoke-rigid, smoke-cloth, and smoke-soft-body couplings voxelize
  current world-space triangle shells once per frame, retain fractional
  cut-face apertures and interpolated wall-normal velocities, and use an
  aperture-weighted Neumann pressure stencil. Projected grid pressure and
  viscosity are integrated over coupled rigid, cloth, and soft-body triangles.
  Dynamic rigid bodies receive bounded linear and angular reaction; cloth and
  skinned soft bodies receive deterministic, bounded per-node loads before
  prediction, and rope nodes sample projected grid velocity/density before
  their direct stretch solve. Grid tracers no longer add a duplicate
  particle-mass reaction. Coupling removal clears the obstacle on the next
  submitted frame.
- Generation-checked paint fields accept host, shared, or private Metal UV
  buffers. Deterministic contact paint covers fluid-to-rigid meshes and
  rigid-to-cloth brushes, including two-sided pixel bits and authored UV
  remapping after physical cloth splits. Fluid particles stamp concurrently
  with atomic side-bit ORs; rigid-to-cloth stamping reduces the deepest
  triangle with stable tie-breaking and rasterizes triangles cooperatively. A
  65-particle gate verifies simultaneous front/back bits across a lane boundary.
- Opt-in rigid and fluid contact views retain stable, capacity-bounded events
  for completed frames. Rigid events are prepared in stable compacted-pair
  order and retain the exact normal and friction impulses applied by the
  colored solver. Fluid events retain
  the strongest exact particle-rigid impulse per particle across substeps and
  preserve that solver sample through source, boiling, and destroy-plane
  compaction.
- Opt-in physics-debug capture records a stride-controlled chronological ring
  of completed rigid, fluid, cloth, soft-body, and rope states, frame-scoped
  force diagnostics, and retained contacts. Latest-frame views borrow ring
  storage; full capture copies are independent.
- Opt-in Metal 4 counter-heap profiling reports total GPU time plus launch
  counts and accumulated GPU time for rigid integration, swept bounds, pair
  filtering/compaction, contact phases, fluid
  forces/integration/lifecycle/couplings, separate cloth and soft-body
  prediction/constraint phases, deformable contacts, rope solves, and the
  separate smoke-grid, smoke-advection, and smoke-emission phases. Profiling
  uses fixed storage and never allocates while stepping.
- Backend-neutral rope, fluid-volume, inflow-surface, and soft-body geometry
  preparation is compiled into the Metal target without CUDA headers.
- Cloth and soft-body prediction, bond projection, post-projection velocity
  finalization, and surface updates run per node/vertex in parallel. Cloth
  fracture runs per bond; volume and shape matching use a single cooperative
  group with stable-order reductions. Soft-body spring damping is a parallel
  gather/apply pair.
- Particle-rigid response uses one cooperative group: particles detect and
  solve against an immutable rigid snapshot, then each body reduces
  per-particle impulses and position corrections in stable particle order.
  Smoke-rigid tracer contact uses the same cooperative pattern, while its
  projected-grid surface load is applied before the tracer phase.
- Registration follows CUDA's material and topology guards for smoke-grid cell
  spacing and automatic wind-travel bounds, movable volumetric shape-matching
  lattices, rope node spacing, rigid-anchor coincidence, rest-centerline
  collider rejection, and duplicate or conflicting particle-system couplings.
- `parallel-mater-metal-gallery` reuses the shared 29-entry registry, GLB
  parser, and Metal scene instantiator. A native GLFW `CAMetalLayer` window
  draws rigid meshes, live cloth and soft-body surfaces, rebuilt rope tubes,
  fluid particles, and smoke particles with embedded MSL shaders. Tab opens the
  CUDA-style scene page; Up/Down selects; Enter loads; arrows steer gravity and
  kinematic bodies; Space performs the scene action; `R` reloads; and Escape
  closes the page or exits. Mouse orbit, Shift-pan, and wheel zoom use the same
  controller as CUDA. Interactive stepping uses a fixed timestep and the
  renderer keeps three frames in flight. The same renderer has an offscreen
  path for PPM captures.
  `--all-scenes --frames 1 --validate` instantiates, steps, renders, reads back,
  and validates every current entry; this is also a separately named CTest
  gate.
- The renderer comparison, known visual gaps, and implementation order are
  maintained in [METAL_GALLERY_PARITY.md](METAL_GALLERY_PARITY.md).
- MSL contains deterministic cooperative-threadgroup scan, stable 64-bit radix
  pass, flagged compaction, segmented float reduction, and min/max kernels.
  One group advances through 256-element chunks in order; radix buckets and
  segments execute in parallel while retaining original order within each
  bucket or segment. Their gate covers 1,025 elements, duplicate-key stability,
  cross-chunk segments, and empty inputs under Metal shader validation.
- Rigid constraints plus fluid, cloth, soft-body, rope, smoke, coupling, paint,
  and debug behavior each have a separately named CTest gate in addition to
  the API/lifecycle integration test. Fluid gates include shuffled signed-cell
  CPU oracles spanning both one and multiple radix blocks, solver-iteration
  equivalence, exact coupling phase ratios,
  same-frame source/contact coverage, rope-anchor participation in the same
  solve, fluid-rope reaction scaling with authored particle volume, capped
  pair acceleration, resting-overlap momentum exchange, full contact-event
  identity, and authored curved-triangle collision. Rigid gates also cover
  two-sided open surfaces, dynamic-pair momentum, off-center angular response,
  BVH leaf-cache overflow, and 36-body color overflow. Ten 120-frame
  rigid-context replays are byte-identical on the current Apple Silicon test
  machine. Ten 257-particle smoke replays spanning four threadgroups are also
  byte-identical across position, velocity, age, density, pressure, and
  vorticity captures. The current build passes all 24 non-golden CTest gates.
- `parallel-mater-metal-parity-capture` runs one scenario containing rigid
  bodies, a fixed constraint, all five particle systems, and every exposed
  pairwise coupling. It serializes exact IDs, counts, ordering, topology,
  flags, states, force diagnostics, contacts, statistics, constraint state,
  smoke tracers, and the projected smoke grid into the versioned
  `parallel-mater-parity-v1` text contract. Its CTest gate requires ten
  byte-identical Metal runs and currently checks 23,347 records per run.
  `parallel-mater-parity-compare` rejects schema/order/non-finite differences,
  compares integer records exactly, and applies the acceptance tolerances of
  `1e-5` absolute plus `1e-3` relative to floating records. Generate and
  compare checkpoints with:

  ```bash
  ./build-metal/parallel-mater-metal-parity-capture metal.capture
  ./build-metal/parallel-mater-parity-compare cuda.capture metal.capture
  ```
- `parallel-mater-metal-conformance` runs the backend-neutral `conformance/v1`
  registry through the same runner and checkpoint serializer as CUDA. The 30
  canonical cases, source assets, committed CUDA goldens, and comparator are
  therefore the primary cross-backend gate; the smaller parity capture remains
  the byte-identical same-backend replay gate. Host geometry is compiled with
  floating-point contraction disabled so exact topology fields do not depend
  on the workstation CPU architecture. Run the full comparison with:

  ```bash
  ./build-metal/parallel-mater-metal-conformance \
    --case all --output build-metal/conformance-metal
  python3 conformance/v1/compare.py \
    --cases conformance/v1/cases \
    --expected conformance/v1/golden/cuda \
    --actual build-metal/conformance-metal
  ```

  The current Apple Silicon checkpoint executes all 30 cases without Metal
  validation failures. The committed CUDA corpus predates the revised hinge
  and piston assets and must be refreshed on NVIDIA hardware before those two
  cases are meaningful again. The comparison continues to expose remaining
  constraint, cloth-tear, and rope-soft-body parity work; it is intentionally
  not weakened or treated as a gallery acceptance gate. The exact capture,
  provenance, phase-checkpoint, and gallery artifacts needed from that machine
  are listed in
  [CUDA_REFERENCE_REQUIREMENTS.md](CUDA_REFERENCE_REQUIREMENTS.md).
  Individual particle/rigid trajectories and raw contact records in chaotic
  cases remain finite-checked diagnostics. Fluid-contact records in
  non-chaotic cases
  are compared in their CUDA contract order of stable particle ID then rigid
  body, independent of backend-local contact-buffer insertion order.
  Rigid manifold samples are grouped by body pair and assigned by minimum
  geometric cost before tolerance checks; this prevents sub-micrometre normal
  noise from changing their comparison order.
  Rigid contact generation uploads the CUDA-style BVH leaf order and reduces
  leaf-pair manifolds in the same stable 128-lane candidate order; this is
  required for symmetric impacts to make the same deterministic choice.
  Rigid manifolds now use CUDA's standard triangle-pair closest points first.
  Metal's segment/triangle query is only a numerical fallback for an empty
  deep-sweep manifold, a body-reference plane crossing with no approaching
  standard contact, or inconsistent depths across an otherwise coplanar deep
  sweep. A moving, unconstrained pair whose standard manifold contains only
  separating contacts switches to swept-only evaluation once its relative
  motion exceeds the collision margin; separating swept samples are then
  discarded. Fixed clusters retain the CUDA-ordered standard manifold. Mixed
  approaching/separating pruning is likewise limited to pairs eligible for
  that robust fallback, so kinematic pairs retain CUDA's complete ordered
  manifold. This keeps slow falling bodies and the 120 m/s tunnelling
  regression above two-sided surfaces without rewriting contact normals,
  closes `fluid-rigid`. In the current package, `rigid-direct` first diverges
  during the authored kinematic impact at frame 40 and retains 13 reported
  differences; its post-replacement frame agrees again.
  Small closed-convex rigid pairs now use CUDA's supporting-face selection and
  clipped incident-face manifold instead of retaining redundant triangle-pair
  contacts. This closes `fluid-rigid`; after the persistent solver port,
  `compound-weld-lifecycle` differs only at frame 48, where micrometre-scale
  pose drift clips each adjacent box patch to six points instead of CUDA's
  four. A blanket four-point cap is intentionally not used because CUDA's
  manifold contract permits up to eight contacts.
  Persistent rigid contacts now carry CUDA's impact fraction, initial normal
  speed, accumulated normal/friction impulse, warm-start state, initial
  relative position, and face-patch classification. Metal loads and stores a
  fixed-capacity, generation-checked cache between substeps without allocating
  or waiting in a step, applies all warm starts before velocity sweeps, uses CUDA's
  translational projection, and raises face patches to the CUDA 32-sweep
  budget. This makes `constraint-breaking` exact, reduces
  `constraint-generic` from 33 to 13 reported fields, reduces
  `constraint-hinge` from 27 fields to one contact-count/list mismatch, and
  brings `rigid-direct` to sub-millimetre positional drift.
  Motor-constrained body contacts stabilize a triangle normal that is already
  parallel to an authored convex collider plane by using the transformed plane
  normal. This removes backend-local tangent noise while retaining contact
  position, depth, and ordering, and makes `constraint-motor` exact.
  Cross-frame cache reuse remains gated: enabling it directly reopens
  `fluid-rigid` and increases `constraint-generic` from 13 to 22 differences,
  so the next CUDA handoff requests the first cache match and warm-start delta.
  Against `run-01` of the reviewed 2026-10-06 CUDA package, this branch passes
  19 of 30 cases. The 11 outstanding cases are `cloth-tear`,
  `compound-weld-lifecycle`, `constraint-fixed`,
  `constraint-generic-spring`, `constraint-generic`, `constraint-hinge`,
  `constraint-point`, `passive-active`, `rigid-direct`, `rope-core`, and
  `rope-soft-body`. The package also exposes five CUDA cases that are not
  byte-repeatable but remain inside the tolerance comparator; those captures
  are retained rather than being misclassified as Metal-only failures.
  Fracturing cloth now rebuilds CUDA's per-face CSR constraint graph at idle
  frame boundaries, preserves active bending links, and runs the detached-face
  strain limiter before sampling impact and strain damage. Impact fracture
  reads the per-node contact impulse directly, matching CUDA instead of
  reconstructing it from a force and timestep.
  Soft-body rigid contact now follows CUDA's position-solve cadence: before
  graph projection and after every two spring passes. Best-fit orientation is
  persistent across substeps, its projection uses the global minimum-bond cap,
  and closed-solid sweeps do not pull an already recovered interior node back
  through the exit face. Contact momentum, normal displacement, and friction
  are accumulated across the whole substep. Static and dynamic contacts apply
  CUDA-style support and tangential traction after each contact pass, finish
  equal-and-opposite rigid reactions deterministically, suppress shape matching
  after dynamic contact, and restore the predicted soft-body momentum after the
  final pass. This makes both `soft-body-core` (formerly 216 reported field
  differences) and `soft-body-rigid` (formerly 236) exact. Rope/soft endpoint
  refresh reduced `rope-soft-body` from 177 to 95 differences and now
  matches CUDA's substep boundary: the post-reaction skin updates the stored
  anchor for the next solve without rewriting the already-solved rope endpoint.
  This closes both maximum-strain differences in that case (about 2.18% and
  1.64% on the prior Metal path). The coupling now also uses CUDA's cumulative
  frame contact count, 256-lane anchor-weight reduction order, and closed-skin
  nearest/swept point query. Those changes reduce the remaining report from 95
  differences to 8.
  Fluid/rope reactions now gather particle impulses per rope node before
  applying CUDA's single acceleration cap, exclude rigid-attached endpoints,
  and leave fluid acceleration diagnostics unchanged by the post-integration
  contact projection. This moves `rope-fluid` inside its documented chaotic
  envelope.
  Shared cloth, fluid, and soft-body reactions against rigid bodies now use
  CUDA's exact 128-lane strided accumulation and binary-tree reduction order
  instead of a Metal-only serial particle sum. This preserves deterministic
  reaction ordering without adding allocations or waits inside stepping.
  Fluid/soft-body coupling transports the previous and current deforming skin,
  uses swept face/edge/vertex nearest-point classification with winding-based
  inside tests, performs CUDA's position-only final recovery after graph
  projection, then repeats current-skin recovery three times after rigid-fluid
  contacts. Together with stable contact-degree projection this reduces
  `soft-body-fluid` from 19 reported differences to only a 0.013 final
  maximum-strain delta. The conformance comparator now applies the documented
  chaotic scalar tolerance to strain, clearance, and divergence metrics.
  Soft-body/cloth coupling now uses CUDA's global maximum pass count across all
  enabled pairs and interleaves soft contact, cloth application/fracture,
  cloth projection/volume/strain work, and soft projection in CUDA phase
  order. Deterministic 32-bit contact-degree counts drive the exact CUDA
  relaxation formula; floating reactions still use stable contribution
  gathers. This makes `soft-body-cloth` exact (111 reported differences to
  zero).
  Safety-quality fields are directional envelopes: divergence, pressure
  residual, and strain may improve below CUDA's upper bound, while clearance
  may improve above its lower bound. This admits Metal's lower final
  `smoke-rope` divergence without relaxing topology, lifecycle, finite-state,
  aggregate-motion, energy, or momentum checks.
  Fixed joints now rebuild CUDA-style aggregate rigid compounds every
  substep, reconstruct member poses, conserve aggregate linear and angular
  momentum, use aggregate mass and inertia during contact coloring/response,
  absorb eligible weld constraints, interleave general weld/contact solves,
  and project movable fixed components out of static contacts as a unit. This
  makes `compound-weld-lifecycle` exact and reduces `constraint-fixed` from 41
  to 7 differences; the latter's body state and momentum now pass, leaving
  only contact-debug sample deltas. Removing the final fixed constraint also
  clears its derived compound state immediately so released members re-enter
  broad-phase collision filtering on the next step.
  Contact manifold selection now follows CUDA's stable insertion rule instead
  of retaining an extra Metal-only deepest-contact subset. Coincident triangle
  normals use the rigid origin or hinge anchor as CUDA does. Static-hinge
  contacts use the hinge's single rotational degree of freedom for effective
  mass, point velocity, friction, impulse application, and position recovery.
  This made `constraint-hinge` exact against the older committed corpus and
  preserves the rigid and constraint validation gates. The revised CUDA
  package still reports 27 hinge differences, so it remains in the explicit
  outstanding list above. The fast authored kinematic sweep in
  `passive-active` likewise remains an outstanding manifold-parity case.
  Rope contact now follows CUDA's closed-convex-mesh semantics: nodes already
  inside a solid recover through the nearest or swept-entry plane, capsule
  segments use outward solid-plane normals, and zero-thickness triangle meshes
  retain their two-sided behavior. Each rope node also retains CUDA's cached
  solid-plane hint across substeps, and segment/triangle intersection uses the
  same CUDA thresholds and barycentric acceptance. The current `rope-core`
  capture does not exercise an interior-solid recovery and therefore remains
  one of the outstanding trajectory-parity cases.

## Gated work remaining

Optional Metal captures return `StatusCode::not_supported` when they were not
enabled at World creation. The particle implementation is a stable,
deterministic correctness baseline, not a claim of CUDA numerical or
performance parity.

1. Establish large-pile correctness and performance baselines for the colored
   rigid solver and row-parallel broad phase.
2. Establish production-capacity performance baselines for the cooperative
   lifecycle, paint, and endpoint-preparation paths. The rope tridiagonal solve
   and same-target endpoint application intentionally retain CUDA ordering.
3. Replace the complete native raster gallery with Metal ray-traced visual
   parity, including implicit fluid surfaces, paint textures, transparent
   skins, foam, smoke-grid debug maps, CUDA-gallery overlays, and the remaining
   advanced camera/debug controls.
4. Close the remaining canonical CUDA/Metal physics differences reported by
   `parallel-mater-metal-conformance-cuda-goldens`; then add render references,
   validation captures, and named-chip performance baselines.

The Metal tests require access to a physical Metal device. Sandboxed runners
that hide the GPU can compile the target and API contract but cannot run the
foundation or rigid runtime tests.

Build and run the gallery with:

```bash
cmake -S . -B build-metal-gallery \
  -DPARALLEL_MATER_BUILD_CUDA=OFF \
  -DPARALLEL_MATER_BUILD_METAL=ON \
  -DPARALLEL_MATER_BUILD_METAL_GALLERY=ON \
  -DBUILD_TESTING=ON
cmake --build build-metal-gallery
./build-metal-gallery/parallel-mater-metal-gallery
./build-metal-gallery/parallel-mater-metal-gallery \
  --rope-soft-body
./build-metal-gallery/parallel-mater-metal-gallery \
  --all-scenes --frames 1 --headless --validate \
  --output build-metal-gallery/captures
```
