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

## Rigid performance pass — before PR 5

- Parallelize deterministic BVH leaf-pair contact evaluation.
- Compact GPU broad-phase results so the solver visits only potentially
  overlapping body pairs.
- Support explicit Blender collision proxies without changing detailed render
  geometry.
- Add velocity-gated conservative swept triangle contacts and a high-speed
  tunneling regression.
- Publish retained and rejected hypotheses with reproducible timing settings in
  [the performance report](PERFORMANCE.md).

## PR 5 — Isolated fluid and particle lifecycle

- Do not implement or execute this stage until its Blender-authored acceptance
  scene has been supplied and reviewed. The scene, rather than procedural C++,
  defines the feature demonstration.
- Implement owned particle storage and deterministic sorted-cell neighbors.
- Implement fluid forces/constraints without rigid coupling.
- Implement deterministic device-side spawn planes, swept destroy planes, and
  stable compaction without allocations during stepping.
- Add the fluid-tank scene and brute-force neighbor reference tests.

## PR 6 — Fluid–rigid coupling

- Add particle/triangle contacts, friction, restitution, projection, and balanced
  reactions on dynamic bodies.
- Add deterministic contact events and the heavy-sphere scene.
- Validate momentum exchange, containment, high-speed impact, and overflow.

## PR 7 — Gallery game shell

- Reuse the exact gallery scenes in a progression application.
- Add objectives, scene selection, controls, and save data outside the library.
- Add the obstacle-bowl scene.

## PR 8 — Water rendering

- Add debug particle rendering first.
- Evaluate reconstructed raster water and OptiX water as example-only renderer
  modules.
- Select by GPU capability without changing or conditionally compiling the
  installed physics API.

## Later, one solver at a time

Cloth, soft body, rope, and smoke each require an approved API extension, one
focused gallery scene, two-system coupling tests, and performance evidence.
No campaign or presentation concept is promoted into the installed library.
