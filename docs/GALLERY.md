# Gallery and progression game

The gallery is an executable examples package built on the public API. Product
acceptance scenes use Blender-authored `.glb` geometry and declarative physics
metadata. Focused stress scenes may build small procedural triangle meshes when
the construction itself is part of the example contract. Each scene must be
understandable in isolation and reusable by the progression game, but none of
its types are installed with the physics library.

## Boundary

```text
parallel_mater (installed library)
  World, FluidId, RigidBodyId, geometry sampling, inflow/outflow,
  paint fields and rules, device views, contacts

examples/gallery (not installed)
  .glb assets, scene loader, renderer adapters, controls, objectives

apps/gallery_game (not installed)
  scene selection, progression, save data, menus
```

The renderer consumes borrowed device views. The current optional OptiX module
ray traces Blender-authored rigid render meshes without making OptiX a
dependency of the installed physics package. Its first implementation performs
a synchronous device-to-host image readback for a deliberately small and clear
example; CUDA/OpenGL interop is deferred until measurements justify it.
Headless physics builds remain free of OpenGL and OptiX.

## First gallery sequence

1. **Rigid body** — Blender-authored static, kinematic, and dynamic triangle
   meshes collide inside a concave bowl.
2. **DUMP** — 10–1,000 shared-mesh spheres pour from a kinematic open hopper
   into a larger static receiver.
3. **Fluid flow** — Blender Inflow emits repelling particles over a passive
   triangle surface; Outflow removes them and impact agitation shows as foam.
4. **Fluid + rigid** — 64 independently simulated dynamic spheres from a
   Blender 4×4×4 array fall into the fluid and receive two-way impulses.
5. **Peg Paint** — a Blender Geometry flow seeds water once into a bowl with
   four static pegs and one dynamic sphere; fluid contacts persistently paint
   the authored UV surfaces blue.

The visible `parallel-mater-gallery` loads `examples/assets/PassiveActive.glb`,
instantiates its passive ground, kinematic Cube, and dynamic Icosphere and
Suzanne through `World`, and ray traces their authored render triangles. The
three detailed dynamic meshes use separate Blender-authored collision proxies;
the bowl retains its detailed collision surface. Arrow input moves the Cube
and tilts gravity for the dynamic bodies. C++ does not restate that scene's
body list or transforms.

`Tab` opens an examples-only context selector ordered Rigid Body, DUMP, Fluid,
Fluid + Rigid, Peg Paint.
Up/Down changes selection and Enter activates an available scene. A shared
camera controller works in all scenes: left-drag orbits,
Shift+left-drag pans, and the wheel zooms. Switching scenes resets the pan to
the new scene's target while preserving orbit and zoom. The Fluid camera can
also move beneath the level for inspection. DUMP uses one
procedural sphere mesh—eight cube corners plus six face centers, projected to a
radius and joined as four triangles per face—and instances it for every body.
Its hopper omits top and right faces; Left Arrow rotates the hopper clockwise.
`P` opens a 10–1,000 sphere-count editor and applying a value rebuilds the DUMP
runtime. Fluid loads the supplied Blender-authored `Fluid.blend` via its GLB
export. In Fluid, `P` edits the particle cap (100–100,000; default 30,000)
and restarts the scene. `--fluid-particles N` sets the cap for a headless run.
`R` rebuilds the active scene from its initial state, clearing particles and
emitter history. In Fluid, `V` toggles between the continuous surface and
individual particles; `--fluid-particle-view` selects particles in headless
mode. In other scenes, `V` retains the rigid-contact debug view.
For containment debugging, run the headless Fluid scene with
`--trace-fluid-escapes`. It reports the first particle below the passive
mesh's overall bottom or more than 1 mm below its local floor surface,
including its stable ID and recent positions, and exits nonzero on penetration.
`F` shows Fluid solver stages, surface-build GPU time, OptiX/render wall time,
and live/emitted/outflow/capacity-miss counts. It shows rigid solver stages in
the other scenes. Fluid + Rigid uses the same `P`, `R`, `V`, and `F` controls as
Fluid, or `--fluid-rigid` for a headless run.
Peg Paint uses the same fluid controls and can be selected with `--peg-paint`.
In Peg Paint, the arrow keys or WASD tilt the scene's authored 2g gravity up
to 50 degrees relative to the camera. The tilt eases in and returns to
vertical when released. Static fluid contacts use impulse-limited friction,
so reversing direction can build a wave instead of stopping at the wall.
Unlike Inflow, its Blender Flow/Geometry cylinder is sampled into particles
on an HCP lattice once at scene creation; it does not keep emitting. The
authored sphere starts above the bowl and falls through the water. The gallery
registers the bowl's render triangles and UVs as a `World` paint field; the
physics contact path stamps its persistent two-sided mask. The renderer only
samples that mask and applies blue color with cubic filtering. The sphere and
pegs remain unpainted. `World` owns no renderer or paint color.
The shared water shader reflects authored geometry and uses the original
course's lighter absorption and haze. All three fluid scenes use the shared
render-only foam module: foam signals seed bounded, short-lived multi-bubble
patches that follow stable water-particle IDs. Patch size follows particle
scale so Fluid and Fluid + Rigid remain visible from their wider camera. Foam
uses the rigid-body depth buffer for occlusion, so splashes behind the thin
bowl wall do not appear on its exterior.

Each scene adds one capability and becomes its regression example. The game
can present the same scenes in order and layer objectives on top.

## Scene contract

An example scene owns only application state:

- create resources in a `World`;
- translate input into gravity, forces, impulses, or kinematic targets;
- invoke one world step per fixed frame;
- render borrowed device views;
- consume contact events for effects and objectives;
- decide success and request the next scene.

No scene reaches into solver memory or calls private kernels. A feature is not
considered reusable until a scene can express it through the installed API.

## Visual quality

Physics particles and rendering are separate. The gallery ray traces rigid
meshes and reconstructs a continuous fluid surface from a weighted local
particle-center field on an adaptive GPU grid. OptiX traces that field and
shades water with Fresnel reflection, refraction, absorption, and glints.
Sparse depth-tested foam flecks come from the particle foam signal. The
surface renderer now uses the solver's particle-neighbor reach and caps its
lower envelope near the particle radius, preventing the reconstructed water
from bulging through the underside of the terrain. Stronger absorption avoids
a noisy checkerboard transmission over the test terrain. Meshing is
example-only; fluid stepping stays in the public API. The adaptive surface
grid ignores isolated distant particle
outliers so escaped droplets cannot consume the grid resolution needed by the
main stream; `F` reports how many particles fell outside its render bounds.
