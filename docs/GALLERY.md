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
  World, FluidId, RigidBodyId, ClothId, SoftBodyId, RopeId, geometry sampling, inflow/outflow,
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
6. **Cloth** — a Blender-authored subdivided sheet pins its top and bottom
   vertex rows while a dynamic sphere can press it when arrow keys tilt
   gravity up to 45 degrees; the passive box supports both.
7. **Cloth Tear** — a heavy sphere first falls to the floor; arrow-key gravity
   rolls it into the sheet, where overloaded bonds break and triangle-local
   faces separate without being deleted. The API owns fracture and topology.
8. **Cloth Paint** — an active rigid sphere interacts with an intact pinned
   cloth sheet and persistently paints its UVs at contact through an API rule.
9. **Water Cloth** — one closed, unpinned pressure cloth contains a one-shot
   Geometry-flow water volume. Fluid pressure deforms the cloth and the moving
   cloth surface contains and accelerates the particles through a public
   two-way coupling resource.
10. **Soft Body** — Blender's native Soft Body modifier marks a closed
    Icosphere. The loader converts it into a volumetric spring lattice and the
    public API deforms it against the authored passive triangle arena.
11. **Soft Body Rigid** — the same API lattice collides with two Blender-authored
    active spheres, transferring equal-and-opposite linear and angular impulses
    without a second soft-body solver.
12. **Soft Body Cloth** — a soft sphere lands on an intact pinned bridge over a
    pit, then arrow-key gravity rolls it into a tearable curtain. Both sheets
    share the API coupling and use independently authored fracture settings.
13. **Soft Body Fluid** — Goal-pinned soft bodies deflect under mesh-source
    inflow, with shared fluid rendering and outflow lifecycle.
14. **Rope** — a Bézier rest curve connects passive and active rigid bodies
    through native Blender Hooks. The API owns sampling, tension and contacts;
    the gallery builds an orange tube from its node view.
15. **Rope Fluid** — the same hooked rope, post, and ball under a one-shot
    Blender liquid volume. The API handles fluid/rope segment contact and
    bounded two-way reaction, while existing fluid/rigid triangle contact
    handles the ball and post. The Blender ball has 20 kg mass so it can wind
    under the authored water volume without floating away.
16. **Rope Soft Body** — a hooked rope wraps a Goal-pinned soft post and
    transfers contact and anchor forces through the shared API.
17. **Rope Cloth** — four post-to-corner ropes suspend a subdivided cloth sheet
    and catch a falling rigid sphere through explicit rope/cloth API joints.
18. **Smoke** — a Blender Smoke Inflow plane emits dilute GPU tracers toward
    a passive sphere mesh. A shared staggered air grid, triangle cut faces,
    pressure projection, boundary shear, and resolved curl produce the wake;
    the gallery composites translucent tracer splats.
19. **Smoke Water** — `SmokeWater.blend` starts water at 80°C above an open
    container. The finite `Hot` mesh is a visible passive collision plate and
    a 500°C thermal source. Smoke blows across the falling liquid; contact
    with the plate warms particles to the 100°C boiling point, transfers them
    into the smoke solver, and gives the steam buoyant lift. Select with
    `--smoke-water`; `P` changes water capacity and `R`
    restarts, and headless output reports the boiled-particle count.
20. **Smoke Soft Body** — `SmokeSoftbody.blend` contains 20 Goal-pinned soft
    posts in the smoke plume. The shared API integrates air force sampled on
    their exterior triangles and deflects tracers at their live surfaces.
    Select with `--smoke-softbody`; `R` restarts.
21. **Smoke Cloth** — `SmokeCloth.blend` pins two edges of a 17×17 sheet in
    front of the smoke flow. The API's wind coupling bends the free cloth and
    two-sided triangle contact deflects smoke tracers. Select with
    `--smoke-cloth`; `R` restarts.
22. **Smoke Rope** — `SmokeRope.blend` suspends an active rigid panel from four
    ropes attached to two passive posts. Local smoke-particle motion bends the
    ropes and loads the panel; live capsule and panel-triangle contacts divert
    smoke tracers. Select with `--smoke-rope`; `R` restarts.

All five smoke scenes use the arrow keys to ease gravity up to 45° from down;
releasing the arrows returns it to straight down. Buoyancy and steam rise
follow the tilt. Only locally deposited smoke loads dynamic rigid meshes and
deformable surfaces; Blender-passive bodies remain fixed.
Their shared shallow air field has 128×128 horizontal cells and 32 vertical
cells; Blender emitter properties can adjust these counts, viscosity, LES
coefficient, pressure work budget, and convergence tolerance.

All smoke contexts share solver-field inspection. Each key selects one mode;
pressing the active key again returns to tracer rendering. `Z` draws the grid
bounds, three orthogonal slices, and orange solid cut cells. `X` draws
cell-centered velocity as signed XYZ RGB plus sparse vectors. `C` draws a
blue-to-red signed pressure map. `V` shows deposited density with cold loading
in blue and hot loading in red. `B` draws signed XYZ RGB vorticity, and `N`
draws blue-negative/red-positive divergence. Field maps use three center
slices except density/heat, which draws occupied cells throughout the volume.
The legend reports the per-frame normalization maximum. These visualizers read
`SmokeDeviceView`; no rendering state enters the physics solver.

The visible `parallel-mater-gallery` loads `examples/assets/PassiveActive.glb`,
instantiates its passive ground, kinematic Cube, and dynamic Icosphere and
Suzanne through `World`, and ray traces their authored render triangles. The
three detailed dynamic meshes use separate Blender-authored collision proxies;
the bowl retains its detailed collision surface. Arrow input moves the Cube
and tilts gravity for the dynamic bodies. C++ does not restate that scene's
body list or transforms.

`Tab` opens an examples-only context selector ordered Rigid Body, DUMP, Fluid,
Fluid + Rigid, Peg Paint, Cloth, Cloth Tear, Cloth Paint, Water Cloth, Soft Body,
Soft Body Rigid, Soft Body Cloth, Soft Body Fluid, Rope, Rope Fluid.
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
mode.
For containment debugging, run the headless Fluid scene with
`--trace-fluid-escapes`. It reports the first particle below the passive
mesh's overall bottom or more than 1 mm below its local floor surface,
including its stable ID and recent positions, and exits nonzero on penetration.
`F` shows Fluid solver stages, surface-build GPU time, OptiX/render wall time,
and live/emitted/outflow/capacity-miss counts. It shows rigid solver stages in
the other scenes. Fluid + Rigid uses the same `P`, `R`, `V`, and `F` controls as
Fluid, or `--fluid-rigid` for a headless run.
Peg Paint uses the same fluid controls and can be selected with `--peg-paint`.
Cloth can be selected with `--cloth`; `R` restarts the pinned sheet and
`F` reports its constraint and contact timings. Cloth, Cloth Tear, and Cloth
Paint all start with straight-down gravity. Arrow keys steer it relative to
the camera, up to 45 degrees from vertical; releasing them eases it back to
straight down. `--cloth-tear` and `--cloth-paint` also support `R` reset and
`F` timings. Headless runs can opt into `--cloth-tilt-degrees 1..45`.
Water Cloth can be selected with `--water-cloth`; it also accepts the common
fluid particle cap. Its pressure cloth is a transparent refractive render
layer.
Soft Body can be selected with `--soft-body`. It starts with straight-down
gravity; arrow keys steer gravity through the shared deformable control path,
`R` rebuilds the API resource, and `F` reports node prediction, spring
projection, and passive triangle contact timings. `V` draws every generated
internal API spring; `B` draws the authored surface wireframe.
Soft Body Rigid can be selected with `--soft-body-rigid`. It reuses all of those
controls and diagnostics while two active spheres collide with the lattice.
Rope can be selected with `--rope`. Gravity starts down; arrows use the shared
45-degree tilt controller. `R` resets, `F` includes rope solve time, `V`/`B`
draw physical segments, `X` shows constraint/contact forces, `N` velocities,
and `M` writes the common opt-in API capture.
Rope Fluid can be selected with `--rope-fluid` and uses the same gravity,
camera, reset, and rope debug controls. `P` changes the water-particle cap;
`V` toggles water particles; `F` includes fluid/rope contact time and depth.
Soft Body Cloth can be selected with `--soft-body-cloth`. It starts with
gravity down and reuses those controls. `F` also reports cloth stepping and
soft/cloth coupling. The bridge's Blender `pm_break_strain` is zero; the
curtain's is 0.10. Both use the common exporter and API fracture implementation.

Soft Body Fluid (`--soft-body-fluid`) uses `SoftbodyFluid.blend`. Full-weight
Goal vertices remain fixed while falling water loads the soft body's current
triangles and exits through the authored outflow. Gravity starts straight down;
arrows tilt it up to 45 degrees. `P` changes particle capacity and resets;
`R` resets, `F` includes fluid/soft contact timing, and the shared water surface
and foam renderer are unchanged. `V` exposes both water particles and internal
soft springs; `B` shows the soft surface and `C` shows water reactions.

All gallery entries create `World` with opt-in rolling physics capture.
Outside the smoke contexts, `Z` shows available contact and deformable-surface normals, `X` rigid inputs and
rigid/deformable contact forces,
`C` fluid accelerations and reactions, and `N` rigid/fluid/cloth/soft velocities.
`V` switches to available particle/deformable structure, `B` shows active cloth bonds
or the soft-body surface, and
`M` copies the common API capture to a self-describing log in `/tmp`. Missing
systems simply contribute no vectors. Rendering and log persistence stay in
gallery support; state, forces, contacts, and the chronological capture come
from the public API. `--physics-capture file.log` exercises the same writer in
headless runs. `--cloth-debug` draws overlays for every cloth in the selected
scene; the legacy `--water-cloth-debug` also selects Water Cloth. Wireframe and
normals follow the API's separated triangle surface after tearing; broken
bonds are omitted rather than drawn across detached fragments. Force vectors
remain attached to physical nodes.
The gallery supplies cloth UVs to `World::add_paint_field` and registers a
rigid-to-cloth paint rule. Collision and mask stamping happen in `World`; the OptiX
adapter merely filters and displays the borrowed two-sided mask.
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
