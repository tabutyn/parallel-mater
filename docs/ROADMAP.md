# Implementation roadmap

Every stage is a separate pull request. A stage is merged only after its public
example, deterministic tests, sanitizer checks, and unprofiled measurements
pass.

## PR 1 — API review RFC (merged)

- Agree on ownership, handles, stepping, views, contacts, and scope.
- No physics implementation and no copied legacy source.

## PR 2 — Core and rigid bodies (merged)

- Implement status, token, world lifetime, generation-checked handles, and
  CUDA stream behavior.
- Implement static, kinematic, and dynamic sphere/box/capsule/plane bodies.
- Add the rigid-sandbox gallery scene and CPU-reference integration tests.

## PR 3 — OptiX gallery and Blender-authored scenes

- Add an optional OptiX renderer under `examples/`; OptiX must not become a
  dependency of the installed physics target.
- Load `.glb` geometry, materials, transforms, and ParallelMater node metadata.
- Define a versioned Blender custom-property convention for static, kinematic,
  and dynamic rigid bodies.
- Add a Blender Python authoring/export script and concise scene-authoring guide.
- Render one Blender-authored scene containing a static plane plus dynamic
  sphere, box, and capsule. The gallery source must not reconstruct that scene
  procedurally.
- Add a headless image test and an interactive orbit-camera example.

## PR 4 — Isolated fluid and particle lifecycle

- Implement owned particle storage and deterministic sorted-cell neighbors.
- Implement fluid forces/constraints without rigid coupling.
- Implement deterministic device-side spawn planes, swept destroy planes, and
  stable compaction without allocations during stepping.
- Add the fluid-tank scene and brute-force neighbor reference tests.

## PR 5 — Fluid–rigid coupling

- Add analytic contacts, friction, restitution, projection, and balanced
  reactions on dynamic bodies.
- Add deterministic contact events and the heavy-sphere scene.
- Validate momentum exchange, containment, high-speed impact, and overflow.

## PR 6 — Gallery game shell

- Reuse the exact gallery scenes in a progression application.
- Add objectives, scene selection, controls, and save data outside the library.
- Add the obstacle-bowl scene.

## PR 7 — Water rendering

- Add debug particle rendering first.
- Evaluate reconstructed raster water and OptiX water as example-only renderer
  modules.
- Select by GPU capability without changing or conditionally compiling the
  installed physics API.

## Later, one solver at a time

Cloth, soft body, rope, and smoke each require an approved API extension, one
focused gallery scene, two-system coupling tests, and performance evidence.
No campaign or presentation concept is promoted into the installed library.
