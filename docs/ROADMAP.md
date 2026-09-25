# Implementation roadmap

Every stage is a separate pull request. A stage is merged only after its public
example, deterministic tests, sanitizer checks, and unprofiled measurements
pass.

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

## PR 9 — Gallery game shell

- Reuse the exact gallery scenes in a progression application.
- Add objectives, scene selection, controls, and save data outside the library.
- Add the obstacle-bowl scene.

## PR 10 — Water rendering portability

- Build a raster fallback for the PR 7 OptiX surface renderer, without changing
  the installed physics API.
- Select by GPU capability without changing or conditionally compiling the
  installed physics API.

## Later, one solver at a time

Soft body, rope, and smoke each require an approved API extension, one
focused gallery scene, two-system coupling tests, and performance evidence.
No campaign or presentation concept is promoted into the installed library.
