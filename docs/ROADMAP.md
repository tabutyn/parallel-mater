# Implementation roadmap

Every stage is a separate pull request. A stage is merged only after its public
example, deterministic tests, sanitizer checks, and unprofiled measurements
pass.

## Blender interface direction

- Keep one exporter, `tools/blender/export_parallel_mater_scene.py`, for every
  supported physics system. CLI, Blender File → Export, and automation call
  the same `export_scene()` function and write the same versioned schema.
- Keep example source authoring helpers under `examples/assets/tools/`; they
  do not export or implement runtime physics.
- Test real Blender exports, failure cleanup, source preservation, and the
  runtime loader contract. Ship the exporter with the installed package.
- Next steps: Blender property panels and a reusable runtime scene importer
  outside gallery support. Extend this interface instead of adding per-scene
  scripts or moving simulation behavior into Blender export code.

## PR 1 — API review RFC (merged)

- Agree on ownership, handles, stepping, views, contacts, and scope.
- No physics implementation and no copied legacy source.

## PR 2 — Core and initial rigid bodies (merged)

- Implement status, token, world lifetime, generation-checked handles, and
  CUDA stream behavior.
- Establish static, kinematic, and dynamic rigid integration.
- Add CPU-reference integration tests.

## PR 3 — OptiX gallery and Blender-authored scenes

- Add an optional OptiX renderer under `examples/`; OptiX must not become a
  dependency of the installed physics target.
- Load `.glb` geometry, materials, transforms, and ParallelMater node metadata.
- Define a versioned Blender custom-property convention for static, kinematic,
  and dynamic rigid bodies.
- Add a Blender Python authoring/export script and concise scene-authoring guide.
- Replace the analytic shape family with one indexed-triangle representation
  for all static, kinematic, and dynamic rigid bodies.
- Export one Blender-authored scene containing a PASSIVE ground plus ACTIVE
  cube, icosphere, and Suzanne. Apply scale and triangulation in a
  non-destructive exporter; the gallery must not reconstruct the scene.
- Complete deterministic BVH-accelerated mesh contact, dynamic–dynamic
  response, and two-sided open-surface collision before fluid work begins.
- Add a headless image test and an interactive orbit-camera example.

## PR 4 — Rigid observability and gallery navigation

- Remove the procedural rigid sandbox so the Blender-authored gallery remains
  the only example surface.
- Add opt-in per-kernel GPU timings and rigid contact diagnostics.
- Add the examples-only `Tab` selector with grey Rigid Body and blue Fluid
  identities. Fluid remains unavailable until its acceptance scene exists.

## PR 5 — Rigid performance pass (in review)

- Parallelize deterministic BVH leaf-pair contact evaluation.
- Compact GPU broad-phase results so the solver visits only potentially
  overlapping body pairs.
- Support explicit Blender collision proxies without changing detailed render
  geometry.
- Add velocity-gated conservative swept triangle contacts and a high-speed
  tunneling regression.
- Publish retained and rejected hypotheses with reproducible timing settings in
  [the performance report](PERFORMANCE.md).

## PR 6 — DUMP rigid stress scene

- Add DUMP between Rigid Body and Fluid in the gallery selector.
- Instance 10–1,000 cube-projected triangle spheres inside a kinematic hopper.
- Rotate the hopper into a larger receiver and rebuild the scene when its
  sphere-count dialog changes.
- Bound rigid contact caches by active-pair demand so 1,000-body worlds do not
  reserve a leaf-pair buffer for every possible pair.
- Repair spare-capacity manifold indexing and preserve swept contacts when a
  leaf cache overflows.
- Add pair-relative swept gating, conservative shape and triangle bounds,
  compact manifolds, and a deterministic parallel contact solver.
- Reduce exact triangle-pair distance work, isolate rare BVH overflow, tighten
  conservative sweep bounds, and color contacts at all body counts with
  deterministic parallel matching; keep authored triangles as contacts.
- Retain only measured stress-scene gains and publish a reproducible DUMP
  benchmark with 1,000-sphere containment, repeatable state hashes, and
  128/256-body coloring regressions. Record rejected variants and isolated
  before/after timings in [the performance report](PERFORMANCE.md).

## PR 7 — Blender-authored liquid flow and passive collision

- Use the supplied `examples/assets/Fluid.blend` as the acceptance scene. Its
  native Liquid Inflow and Outflow planes define particle lifecycle, and its
  passive triangle mesh defines the collision surface. No Blender Fluid Domain
  is needed by ParallelMater.
- Implement owned particles, deterministic sorted-cell repulsion and viscosity,
  and explicit neighbor-overflow errors.
- Implement deterministic device-side inflow, swept outflow, and stable
  compaction without CUDA allocations during stepping.
- Collide particles with passive authored triangles, including swept crossing
  of open surfaces. Keep reaction forces on dynamic rigid bodies in PR 8.
- Show blue debug particles and short-lived white agitation foam in the gallery;
  reconstruct an example-only OptiX water surface from the particles.
- Test a brute-force neighbor pair, lifecycle, passive high-speed impact,
  authored-scene loading, and a headless fluid image.

## PR 8 — Dynamic fluid–rigid coupling (merged)

- Extend the PR 7 passive particle/triangle path to moving and dynamic bodies,
  including friction, restitution, projection, and balanced reactions.
- Retain bounded, deterministic per-particle contact events and add the
  Blender-authored `FluidRigid.blend` scene: three Array modifiers detach into
  64 independently simulated, shared-mesh dynamic spheres.
- Validate momentum exchange, containment, high-speed impact, and overflow.

## Peg Paint — implemented in this branch

- Export Blender Liquid Flow/Geometry as a one-shot, closed-mesh particle fill.
- Paint authored rigid UVs persistently from fluid–rigid contacts through
  opt-in `World` paint fields and transfer rules; the gallery owns color and
  filtering, not paint state.
- Tune the single active sphere to sink while retaining fluid in the bowl;
  validate paint UV seams, the authored scene, containment, and timing.

## Cloth — Blender-authored pinned sheet and rigid coupling

- Export `Cloth.blend` as an open, indexed cloth mesh with its
  `FixedVertices` Shape Pin Group and stiffness 1.0. Preserve the two pinned
  rows through triangulation and glTF vertex splitting.
- Add world-owned cloth particles and links, exact fixed vertices, reusable
  cloth handles/views, and two-way contact with authored rigid triangles.
- Add a Cloth gallery entry with straight-down gravity by default and arrow
  steering up to 45 degrees, plus a dynamic sphere; update its OptiX mesh as
  vertices deform.
- Test authored pin count, zero pin drift, free-vertex movement, rigid
  containment, and headless rendering.

## PR 10 — Cloth tear and cloth paint variations

- Add API-owned persistent bond fracture from sustained strain or rigid
  contact impulses. Preserve every triangle with face-local surface corners
  that separate when their supporting bonds fail.
- Derive `ClothTear.blend` from the original-sized pinned sheet with a heavier
  ball at its original position and friction. Gravity starts down in every
  cloth scene: verify the ball lands without breaking bonds, then rolls under
  steered gravity through the fractured sheet without a prolonged stop. A
  no-impact control does not fracture, and surviving surface faces stay close
  to their rest shapes.
- Derive `ClothPaint.blend` from the authored `Cloth.blend`: retain its pinned
  rows and active rigid sphere, enable cloth UV painting, and keep the cloth
  intact. Rigid–cloth contact and filled-disk texel stamping belong to `World`.
- Add headless and GPU scene tests for both variations.

## PR 11 — Water inside pressure cloth

- Export Blender's native Cloth Pressure settings and accept closed unpinned
  cloth. Initial authored volume is the default pressure target.
- Add reusable API volume preservation and an explicit fluid–cloth coupling
  resource. Particle containment transfers equal-and-opposite reactions to
  the cloth; the gallery only instantiates exported relationships.
- Add the Blender-authored `ClothWater` gallery context with common fluid,
  camera, reset, timing, and gravity controls.
- Render contained-fluid cloth as a transparent water skin and expose shared
  Z/X/C/V normal, coupling-force, particle, and wireframe diagnostics.
- Add opt-in API-owned rolling physics capture for rigid, fluid, cloth, and
  contact state. Share Z/X/C/V/B/N visualization and M log capture across all
  gallery contexts while keeping final drawing and persistence outside `World`.
- Treat `maximum_neighbors` as an explicit diagnostic ceiling, report the
  observed peak, and cover left-steered pressure-cloth compression without
  terminating the gallery on a valid 268-neighbor transient.
- Measure volume drift, escaped particles, and frame cost in a GPU regression
  test, then verify the scene through the common exporter and headless render.

## PR 12 — Soft body and passive rigid collision

- Export Blender's native Soft Body modifier through the single scene exporter.
  Convert each closed authored surface into a deterministic volumetric spring
  lattice while retaining the authored surface shape for rendering.
- Add world-owned, generation-checked soft-body nodes, bonds, surface bindings,
  device views, timings, statistics, and opt-in physics capture to the public API.
- Follow the old lab's stable substep order: predict nodes, resolve passive
  triangle contacts, project the fixed spring graph, reconstruct velocity,
  damp bond-relative motion, and delta-skin the surface.
- Add the Blender-authored `Softbody` gallery context with common reset, camera,
  gravity steering, timing, capture, normal, wireframe, bond, force, and velocity
  diagnostics.
- Test connected volume generation, finite deformation, bounded bond strain,
  passive-mesh containment, stale handles, real Blender re-export, and headless
  rendering.

## PR 13 — Soft body and dynamic rigid coupling

- Reuse PR 12 soft-body resources and surface bindings; add balanced impulses
  against dynamic rigid triangle bodies without a second soft-body solver.
- Map Blender Goal strength to rotation-invariant rest-shape matching so a
  crushed soft body recovers without pinning its translation or rotation or
  projecting through an active collider.
- Validate momentum transfer, high-speed contact, settling, and deterministic
  replay, crush recovery, and free rolling in a focused Blender-authored
  gallery scene.
- Recover nodes from closed convex triangle solids, constrain the skinned
  triangles through API bindings, and resolve final passive barriers after
  traction. Test eight gravity directions with Goal both disabled and enabled,
  checking node and sampled face penetration throughout 1,440 frames per mode.

## PR 14 — Soft body and cloth coupling

- Couple soft-body nodes to current cloth triangles through an explicit API
  resource, with mass-weighted reactions, contact friction, and force capture.
- Sample fracture strain before projection, preserving the existing triangle
  surface and each cloth's independent tear settings.
- Add `SoftbodyCloth.blend`: an intact pinned bridge over a pit and a tearable
  vertical curtain. Reuse gravity steering, camera, timing, reset, and debug.
- Validate sampled bridge clearance, fixed pins, force balance, selective
  tearing, passage, deterministic replay, disabled-coupling and two-sided
  intact-curtain controls, stale handles, Blender export, and headless rendering.
- Split physical vertex fans when seams tear; preserve triangle topology,
  source/paint mappings, mass, and velocity. Test independent fragment motion
  and exclude torn triangles trapped within the deforming soft-body skin.

## PR 15 — Soft body and fluid coupling

- Couple existing fluid particles to the PR 12 soft lattice/surface through an
  explicit API resource with balanced forces and bounded contact diagnostics.
- Reuse the common fluid renderer, foam, particle cap, and capture facilities.
- Added `SoftbodyFluid.blend` and the Soft Body Fluid gallery entry. Gravity
  starts down; arrows tilt 45 degrees. Goal-group full-weight vertices remain
  fixed through the existing API inverse-mass contract.
- External closed-triangle contact maps reactions through skin bindings.
  Shared contact-degree relaxation and symmetric speed-bounded impulses avoid
  launching lightweight nodes under dense water. Current-skin recovery runs
  after rigid boundaries; no scene-specific collision shape or force exists.
- Added contact counters, penetration diagnostics, timings, and fluid reactions
  to soft-body views and opt-in captures. Reused flow source/sink resources,
  renderer, foam, particle cap, reset, camera, and debug controls.
- Aspect-ratio-aware export sampling avoids excessive thin-slab lattices.
  Blender pin export, API impulse balance/lifecycle, dense impacts, and the
  authored inflow/outflow scene have automated regressions.
- Surface preparation now lives in the API: conforming subdivision gives large
  authored faces physical support at interior-lattice resolution, with exact
  pinned-region inheritance. Refitted triangle acceleration keeps fluid contact
  practical on the denser skin. Documented stiffness controls distinguish
  exported custom properties from Blender-only spring settings.
- Authored a coarser, firm soft-fluid material in Blender (777 physical nodes),
  retaining the same mass, Goal attachment and shared API. Dry sag and water-load
  regressions guard against the slab collapsing; no gallery-only support force.
- Replaced rate-based inflow rectangles with API-sampled triangle mesh sources.
  Coarse sites emit only into unoccupied water space; initial velocity controls
  clearance and throughput. Shared spatial lookup, seam thinning, overlapping
  source checks, capacity limits, and lifecycle tests cover every fluid gallery.

## PR 16 — Rope and rigid attachments (review)

- Extend the single Blender exporter with open Bézier curves and native Hook
  attachments, preserving evaluated endpoints without baking Soft Body motion.
- Add API-owned centerline sampling, rope handles/views, whole-chain tension,
  rigid-local endpoint constraints, triangle contacts, friction, and self-contact.
- Reuse gallery gravity, camera, reset, timing, API force capture, and segment
  visualization; keep tube rendering outside the physics library.
- Validate three retained wraps around the authored triangle post, bounded
  strain, hook drift, finite state, and GPU memory safety.
- The initial hidden enclosure crossed the rest rope, producing segments over
  100 times their rest length. Widened only that enclosure to the floor bounds;
  the rest curve, Hook endpoints, ball, post, and materials are unchanged.
  `add_rope` now rejects rest centerlines crossing rigid triangles before
  allocating rope storage. Added creation-failure and settling regressions.
- Contact release now rebuilds the reduced constraint matrix, and a final
  full-mass velocity constraint suppresses axial jitter without increasing drag.
  The API enforces a configurable 1/480 s maximum shared integration step so
  ropes and rigid attachments advance together, independent of gallery code. The
  steering-release regression checks late speed/drift, ground contact, strain,
  hook drift, and sphere/post clearance; a free-fall control preserves bulk
  velocity. Gallery headless coverage includes the settled rope.
  High-strain contacts receive bounded extra nonlinear recovery passes rather
  than carrying a large unresolved correction into the next frame's velocity.
- The pre-wrapped regression uses a longer rest curve around the authored
  post; the steering regression also winds nearly three turns. A later PR 16
  optimization reduced the measured third-wrap rope GPU peak from roughly
  428 ms to 91 ms while retaining the wrap and release checks.

## PR 17 — Rope and fluid (review)

- `RopeFluid.blend` adds a one-shot liquid volume over the rope, active ball,
  passive post, and ground. The single exporter emits the same rope hooks and
  standard fluid initial volume; no gallery-only physics or exporter branch.
- The API owns an optional fluid/rope coupling with particle-to-segment capsule
  contacts, a bounded reaction on free rope nodes, handle lifecycle, force
  views, diagnostics, and per-stage timing. Existing fluid/rigid triangle
  contact handles the ball and post.
- Restored the post Hook geometry to the dry rope scene. Blender's active ball
  mass is 20 kg so it winds through this water volume instead of floating away.
- Tests cover two-way water/rope response, Blender export, 600-frame wet winding
  (2.91 peak turns, <0.5% strain), three-system headless rendering, and CUDA
  memory/race checks.

## PR 18 — Rope and soft body

- `RopeSoftbody.blend` replaces the central rigid post with a soft-body post.
  Its bottom Goal ring is pinned. The ball is an active 1 kg rigid body as in
  the original dry-rope scene, while the post Hook targets the soft body.
- The API adds two-way rope/soft-body triangle-skin contact and surface-bound
  Hook endpoints. Gallery registration has no scene-specific contact physics.
- Validation covers export, attachment lifecycle, force/penetration diagnostics,
  three-wrap winding stability, and a headless render. An 0.08 m generated post
  lattice, eight soft-body iterations, four rope iterations, and 0.9 shape
  matching retain the wrap with 1.2% peak rope strain and roughly 70 ms/frame
  in the 320-frame GPU stress test; four soft-body iterations were unstable.

## PR 19 — Rope and cloth bridge

- `RopeCloth.blend` has four passive posts, four open Poly ropes, one subdivided
  cloth sheet, and an active rigid sphere. Coincident rope/cloth vertices and
  rope endpoints inside passive posts author the joints without Hooks.
- The exporter preserves pre-Cloth Simple subdivision and emits inferred target
  names. The gallery resolves those names to explicit rope/cloth vertex joints
  in the shared physics API; no gallery-specific constraint solver is used.
- The rope solver includes the cloth endpoint's effective supported-patch mass
  and applies a bounded, reciprocal position/velocity correction to the sheet.
  A 1,200-frame GPU bridge check holds all four corners with zero endpoint gap
  and less than 0.003% measured peak rope strain; disabling the joints lets the
  cloth fall. The new gallery entry also has a headless render regression.

## PR 20 — Smoke around a sphere

- `Smoke.blend` authors a Blender Smoke Inflow plane and passive sphere. The
  single exporter maps them to a separate gas resource, not liquid particles.
- The API emits bounded GPU tracer slots, diverts them around a spherical
  obstacle with a no-through-flow field, sweeps contacts to prevent tunneling,
  and sheds alternating vortices. The gallery only composites smoke visuals.
- The later mesh-obstacle update supersedes that original analytic sphere
  flow and collision path; the same API now accepts arbitrary rigid triangles.
- The local particle-gas update supersedes the prescribed wake and global
  rigid wind. Neighbor density, pressure, viscosity, and measured curl govern
  particle motion; near-wall particles alone push rigid meshes.
- The 300-frame GPU comparison against zero wake measures about 0.19 m/s mean
  transverse wake difference for 2,887 downstream particles. Exporter, API,
  smoke physics, headless rendering, gallery, and rope–cloth regressions pass.
  This is an analytic first gas flow, not full Navier–Stokes pressure projection.

## PR 21 — Smoke–water boiling coupling (merged)

- `SmokeWater.blend` authors a liquid volume at 80°C and a separate 500°C
  finite thermal surface that is also a visible passive collision plate.
  Its passive container is open above the heater so
  water can actually reach the plate.
- The API stores Celsius temperature per water particle, carries it through
  source emission and compaction, applies smoke-carrier drag, and transfers
  particles at 100°C into smoke with buoyant thermal lift. The gallery only
  configures the resources and renders their borrowed views.
- GPU tests check temperature metadata, conservation across phase transfer,
  retained-particle identity, wind-driven water motion, and existing smoke
  and fluid regressions. Full-scene 180-frame smoke/water render is stable.

## PR 24 — Smoke–soft-body coupling (review)

- `SmokeSoftbody.blend` authors 20 Goal-pinned posts and the same smoke inlet.
  The gallery registers reusable `SmokeSoftBodyCouplingOptions` for each post;
  smoke drag and tracer contact run in the physics API, not in gallery code.
- Authoring a 16-sided, nine-ring surface and 0.12 m lattice spacing reduces
  the scene from about 56,000 nodes / 4.8 million bonds to 3,520 nodes /
  57,720 bonds. Four graph iterations keep the 180-frame GPU test stable,
  with exact pins and at most 12% observed bond strain.

## PR 25 — Smoke–cloth coupling (review)

- `SmokeCloth.blend` authors one 17×17 sheet with 34 Pin vertices, the smoke
  inlet, an active vortex sphere, and a passive ground. The gallery registers
  a reusable smoke/cloth API coupling without custom scene physics.
- The carrier wind bends free cloth vertices; massless tracers collide on both
  sides of its current triangles, including updated tear topology. A 180-frame
  GPU comparison checks pin stability, spring strain, cloth displacement,
  and smoke-path divergence against the uncoupled scene. A fast-tracer sweep
  regression prevents one-frame tunneling through the thin sheet.
- Follow-up: contact now redirects blocked smoke tangentially toward cloth
  edges. A 300-frame regression checks that the plume reaches the far side
  without collecting against the windward surface.
- The local-particle update removed the fixed edgeward speed. A later surface
  tuning converts measured impact pressure into speed along finite cloth
  toward an open edge, capped by the smoke speed limit. The 300-frame
  regression now counts 797 lateral escapes and 461 particles in the broad
  windward region, without scattering an unobstructed plume.

## PR 26 — Smoke–rope and suspended panel coupling (merged)

- `SmokeRope.blend` supplies four Poly ropes, an active panel, two passive
  posts, the vortex sphere, and a smoke inlet. The single exporter infers
  rigid endpoints on both active and passive meshes.
- Reusable API couplings bend free rope nodes with carrier wind, deflect
  tracers from swept rope capsules, and opt a moving rigid panel into
  swept triangle contact plus bounded carrier pressure.
- GPU regression compares 300 frames against uncoupled motion, checks rope
  strain and a focused rope-hit plume, and rejects fast tracer tunneling
  through the thin panel.

## PR 27 — Smoke gravity, rigid drag, and soft-body cleanup (review)

- All five smoke gallery contexts reuse the 45° camera-relative arrow gravity
  controller. Smoke buoyancy and steam lift follow the resulting vector.
- Projected triangle area replaces signed-area cancellation, so the shared
  smoke/rigid API can push closed dynamic spheres as well as open panels.
  The designated obstacle also uses the shared tracer/triangle contact path.
- Soft bodies, cloth, and ropes share one bounded wind-response calculation;
  smoke/soft-body options now expose the acceleration cap. Tuning the shared
  defaults reduced the 20-post scene's maximum bond stretch from 85% to 41%
  while retaining visible wind response. GPU regressions cover the sphere,
  panel, soft bodies, cloth, rope, and tilt.
- The subsequent local-particle smoke update replaces projected-area rigid
  drag with near-wall particle reaction and samples deformable wind locally.
- The next hybrid update adds a shallow, API-owned 128×32×128 Eulerian air
  velocity/pressure/density field and retains particles as smoke tracers.
  Coupled triangle surfaces obstruct the grid; soft-body surface triangles
  sample its airflow and distribute force to their bound nodes.
- The current hybrid replaces that cell-centered prototype with staggered MAC
  faces, RK2 monotonic MacCormack transport, moving triangle cut faces,
  density/temperature B-spline deposition, LES viscosity and bounded curl
  restoration, and a four-level residual-terminated multigrid projection.
  Grid-mode tracers no longer run a second particle pressure solver. Rigid and
  deformable loads come from local grid pressure and tangential surface stress;
  swept tracer contacts remain containment-only. The shared API exposes grid
  curl, divergence, and relative pressure residual for verification.
- RTX 3050 Ti acceptance is complete: three 128×32×128 runs measured
  5.61–5.74 ms/frame, relative residual 8.00e-4, and normalized divergence
  1.40e-4. Bit-deterministic replay and all smoke coupling regressions pass.
- Shared smoke inspection maps bind `Z/X/C/V/B/N` to grid/cut cells, velocity,
  pressure, density/thermal loading, vorticity, and divergence. RGB mapping,
  slice selection, and legends remain gallery concerns; the API additionally
  exposes the deposited thermal field through `SmokeDeviceView`.

## PR 28 — Rigid body constraints

- Add world-owned, generation-checked Fixed, Point, Hinge, Slider, Piston,
  Generic, Generic Spring, and Motor resources with local frames, limits,
  springs, motors, collision suppression, breaking, and live updates.
- Export Blender's native Rigid Body Constraint settings through schema 2 and
  resolve their body references in the shared gallery loader.
- Add eight reproducibly authored `.blend`/`.glb` gallery scenes. `Space`
  toggles Fixed and Point at the bodies' current poses; arrow keys provide tank
  control for a four-wheel motor car.
- Validate API lifecycle and all solver types on GPU, real Blender re-export,
  authored metadata, scene instantiation, and eight OptiX headless renders.

## PR 29 — Physics source layout refactor

- Split the private CUDA implementation out of the monolithic `world.cu` and
  name each unit for the system or pairwise coupling it implements:

```text
geometry_constraints.cuh
geometry_fluid.cuh
geometry_soft_body.cuh
geometry_cloth.cuh
geometry_rope.cuh
geometry_smoke.cuh
fluid.cuh
fluid_soft_body.cuh
fluid_cloth.cuh
fluid_rope.cuh
fluid_smoke.cuh
soft_body.cuh
soft_body_cloth.cuh
soft_body_rope.cuh
soft_body_smoke.cuh
cloth.cuh
cloth_rope.cuh
cloth_smoke.cuh
rope_smoke.cuh
world.cu
```

- Keep the C++ geometry and Blender realization helpers as dedicated source
  files. This is a source-organization refactor; it does not change the public
  API or simulation behavior.

## Later — Gallery game shell

- Reuse the exact gallery scenes in a progression application.
- Add objectives, scene selection, controls, and save data outside the library.
- Add the obstacle-bowl scene.

## Later — Water rendering portability

- Build a raster fallback for the PR 7 OptiX surface renderer, without changing
  the installed physics API.
- Select by GPU capability without changing or conditionally compiling the
  installed physics API.

## Later, one solver at a time

Measure the local-particle smoke baseline and investigate equal-and-opposite
momentum transfer for the remaining soft-body, cloth, and rope couplings.
No campaign or presentation concept is promoted into the installed library.
