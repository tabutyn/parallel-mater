# Blender-authored triangle scenes

Gallery scenes are `.glb` assets, not C++ recipes. Render triangles drive
collision by default, while an optional Blender-authored lower-resolution mesh
can be selected per body as its collision proxy. ParallelMater has no separate
sphere, box, capsule, or plane collider types.

## Authoring contract

1. Model every rigid object as a Blender mesh. Open, disconnected,
   non-manifold, and inconsistently wound surfaces are accepted; every triangle
   collides from both sides.
2. Use one Blender unit as one metre and keep simulated objects at the scene
   root.
3. Select each object and choose **Object → Rigid Body → Add Active** or
   **Add Passive**. Blender `ACTIVE` maps to `MotionType::dynamic`; an ACTIVE
   body with **Animated** enabled maps to `MotionType::kinematic`; `PASSIVE`
   maps to `MotionType::static_body`.
4. Set mass, friction, restitution, linear damping, angular damping, and an
   optional collision margin in Blender's Rigid Body panel.
5. Put the object origin near its intended center of mass. The current loader
   recenters local geometry from its AABB and preserves the resulting world
   pose. A zero API inertia uses the mesh AABB approximation; an application
   can supply a measured inertia diagonal later.
6. For a detailed object, optionally create a separate low-resolution mesh and
   set the rigid object's custom string property `pm_collision_proxy` to that
   object's Blender name. The proxy may have its own transform and modifiers;
   the exporter evaluates it into the rigid body's local frame. Do not add a
   Blender Rigid Body to the proxy. Keep silhouettes and support surfaces close
   enough that the physical approximation remains intentional.
7. To author a grid of independent identical rigid bodies, add Array modifiers
   to one ACTIVE mesh and keep the evaluated copies disconnected. The exporter
   splits each copy into its own rigid node with a shared triangle mesh. It
   rejects arrays with overlapping or non-identical copies rather than
   silently simulating them as one compound body.

The exporter rejects parented rigid bodies for now. Continuous collision,
compound bodies, and automatic convex decomposition are not part of this
milestone.

For an invisible collider, assign a material and set its **Principled BSDF →
Alpha** to `0`. Blender exports the material alpha; the gallery hides those
faces from camera and depth rays while keeping the rigid body and all its
collision triangles. This works per material slot, not by object name. Constant
glTF `MASK` alpha respects its cutoff; `BLEND` alpha zero is invisible. Partial
alpha blending and alpha textures are not supported yet; glTF `OPAQUE`
materials remain visible regardless of their alpha value.

## Smoke flow

`Smoke.blend` is reproducible with `examples/assets/tools/make_smoke_scene.py`.
It has a passive rigid Icosphere and one mesh with **Fluid → Flow**, **Flow
Type: Smoke**, **Flow Behavior: Inflow**, and initial velocity along +X. The
flow mesh is a world-YZ plane. Its `pm_smoke_obstacle` custom string property
names the rigid sphere. Exporting creates `pm_system=smoke_emitter`, separate
from Liquid Inflow; no Blender domain cache is needed. `pm_smoke_capacity`,
`pm_smoke_rate`, `pm_smoke_lifetime`, `pm_smoke_radius`, `pm_smoke_buoyancy`,
`pm_smoke_response`, and `pm_smoke_wake_strength` are optional emitter
properties. The gallery resolves the obstacle name and radius and registers a
`SmokeOptions` resource in the physics API. The initial implementation requires
one centered passive sphere and one smoke inlet per scene.

## Smoke and water boiling

`SmokeWater.blend` combines a Smoke Inflow with a Liquid Flow/Geometry volume.
Put a `pm_temperature` custom property (degrees Celsius) on the liquid object
to set its initial particle temperature; liquid Inflow objects can set it too.
Add a scene-root plane with no Fluid modifier or rigid body and set
`pm_temperature` to make a heated surface. `Hot` uses 500°C. The single
exporter writes `pm_system=thermal_surface`; the gallery imports its triangles
as a visible passive collision plate and configures the API coupling from its
finite plane and temperature. Optional plane properties are
`pm_heat_transfer_rate`, `pm_smoke_drag`, and `pm_steam_rise_speed`.
Water particles transfer to the separate smoke system at 100°C, so keep the
passive container open above the heater. Blender's own fluid cache is unused.

## Rope curves and attachments

Use one open Bézier spline at the scene root. Select an endpoint control point
in Edit Mode and use a Hook modifier targeting a native passive or active rigid
body, or a mesh with Soft Body physics. Use strength 1 and no distance falloff (a zero falloff radius also works).
Each Hook must contain exactly one endpoint control point, not interior points.
The endpoints may target distinct rigid or soft bodies, or an endpoint may be free.

For a rope-to-cloth joint, use an open **Poly** curve with a Soft Body modifier
and place its endpoint exactly on an authored cloth vertex. The exporter infers
that endpoint joint without a Hook. It also infers a rigid attachment when an
unhooked endpoint lies on or inside one passive rigid mesh. If several targets
overlap, the nearest surface wins only when clearly closer; ambiguous targets
are rejected. A Hook, where present, takes precedence over inference. Poly
curves with Hooks are not supported; use Bézier for explicit Hook attachments.
The geometry decides *which* objects connect; `World::add_rope_cloth_coupling`
implements the physical joint after export.

`RopeCloth.blend` has four such post-to-corner curves. Its Simple Subdivision
modifier sits before Cloth and becomes a 17×17 physical sheet during export;
the original four corner positions remain exact for rope binding. Only Simple
pre-Cloth subdivision, up to five viewport levels, is currently mapped.

The exporter evaluates Blender's actual Hook deformation, including bind
matrices and moved targets. It removes Soft Body only from a temporary copy so
cached simulation does not replace the rest curve. Sampling and runtime
constraints are implemented by the physics API; there is no Blender bake.

Optional curve custom properties:

| Property | Meaning / default |
|---|---|
| `pm_rope_radius` | Physical radius; bevel depth, or 0.01 m when unbevelled |
| `pm_rope_spacing` | Maximum node spacing; default twice the radius |
| `pm_rope_mass` | Total rope mass; native Soft Body mass, or 0.1 kg |
| `pm_rope_compliance` | Stretch compliance; 0 requests a stiff rope |
| `pm_rope_friction` | Coulomb contact friction; 0.4 |
| `pm_rope_damping` | Velocity damping per second; 0.1 |
| `pm_rope_maximum_substep_timestep` | Maximum shared integration step in seconds; 1/480 |
| `pm_rope_iterations` | Nominal constraint/contact budget; 24. High-strain recovery allows up to 4×, capped at 32 |

Keep the rest curve outside collision geometry, except its attachment
neighborhoods. Material alpha zero hides a collider but does not disable it.
Author enough curve length for intended wraps: three turns need at least
`3 * 2 * pi * (post radius + rope radius + margin)`, plus the remaining lengths
between the wraps and the two attachments. A stiff rope cannot create slack.

The schema stores `pm_system = "rope"`, `pm_rope_points` as world-space Y-up
polyline samples, and `pm_rope_first_body` / `pm_rope_last_body`,
`pm_rope_first_soft_body` / `pm_rope_last_soft_body`, or
`pm_rope_first_cloth` / `pm_rope_last_cloth` as target names. A soft Hook
follows the closest rest-surface triangle, while all other rope segments collide
with its deformed skin through the shared rope/soft-body API. To keep a soft post
rooted while the rope winds, assign its bottom vertices to a Blender Soft Body
Goal group with full effective weight; an ungrouped Goal restores shape but
does not pin the post in world space.
Ambiguous instanced Hook targets and unsupported curve modifiers are rejected.

## One Blender–ParallelMater export interface

`tools/blender/export_parallel_mater_scene.py` is the single exporter for rigid
bodies, collision proxies, Arrays, soft bodies, cloth/pins/fracture, liquid
Inflow/Outflow/Geometry, and paint metadata. New physics systems extend this
script and the versioned scene contract, not a per-example exporter. It has no
gallery scene names or scene-specific physics settings.

In Blender 4.5+, install that one `.py` file as an add-on, or open it in the
Scripting workspace and run it once. Use **File → Export → ParallelMater Scene
(.glb)** in Object Mode. The file browser uses the same `export_scene()` entry
point as headless export:

```bash
blender --background examples/assets/PassiveActive.blend \
  --python tools/blender/export_parallel_mater_scene.py -- \
  --output examples/assets/PassiveActive.glb
```

When `--output` is omitted, the script writes a `.glb` beside the currently
open `.blend` using the same filename stem. The script is non-destructive:
it exports evaluated rigid copies and undeformed soft/cloth rest copies, bakes scale
into their vertices, triangulates all polygons, writes schema-2 glTF extras, and removes
the temporary data. The source `.blend` is not saved or changed.
Saving a `.blend` does not update the gallery by itself; re-export its `.glb`
after changing physics properties such as mass.
Selection and the active object are restored on success and failure. Soft-body,
cloth-only, and fluid-only scenes are supported; a dummy rigid body is not required.

Other Blender automation can import this file and call
`export_scene(filepath)` (or omit `filepath` to use the open blend's stem).
Importing the module does not export, register a UI, or edit the scene.
`register()` / `unregister()` manage the optional menu. Installation of the
CMake package also ships this same file under
`share/parallel-mater/blender/`; it does not add a Blender dependency to the
physics library.

The exporter is the Blender-facing boundary. GLB schema 2 is the interchange
contract; the current gallery loader consumes it and constructs public API
resources. A reusable runtime scene importer and Blender property panels can
grow around this boundary without introducing another export implementation.

The generated metadata is:

| Property | Source |
|---|---|
| `pm_schema = 2` | Exporter version |
| `pm_system = "rigid_body"` | Exporter |
| `pm_motion` | Blender ACTIVE/PASSIVE setting |
| `pm_mass` | Blender rigid-body mass |
| `pm_friction`, `pm_restitution` | Blender rigid-body material |
| `pm_linear_damping`, `pm_angular_damping` | Blender rigid-body damping |
| `pm_collision_margin` | Blender margin when enabled, otherwise `0.005 m` |
| `pm_initial_velocity` | Optional 3-component Blender-space custom property for a rigid body's initial linear velocity |
| `pm_checkerboard` | Optional source custom property; defaults on for passive objects in the example exporter |
| `pm_paintable` | Optional Boolean source custom property; gallery registers a persistent API paint field and fluid-to-rigid rule for that body |
| `pm_paint_resolution` | Optional integer 32–2048; square mask resolution (default 512) for a paintable body |
| `pm_collision_proxy` | Optional source custom property naming a lower-resolution Blender mesh |

When a proxy is selected, the exporter adds a non-rendered
`pm_system = "collision_mesh"` glTF node and references it from the rigid-body
node. The gallery still renders the detailed mesh. Proxies are explicit
authored data: the loader does not decimate or invent collision geometry.

Built-in rigid-body settings are exported automatically. The optional `pm_*`
properties configure features without a standard Blender panel yet. Names are
labels except for explicit references such as collision proxies and paint
sources; they do not select scene-specific physics.

## Validate the result

With Blender installed, the gallery build adds
`parallel-mater-blender-export-tests` to CTest. This exports all committed
source scenes into temporary files and checks them with the runtime loader,
plus soft-body-only/cloth-only/fluid-only scenes, the menu operator, CLI, and error cleanup.
No committed assets are rewritten.

```bash
ctest --test-dir build-gallery --output-on-failure -R blender-export
```

To run the Blender checks without building the gallery loader:

```bash
blender --background --factory-startup --threads 1 --python-exit-code 1 \
  --python tests/blender/export_scene_tests.py
```

To render an exported scene:

```bash
./build-gallery/parallel-mater-gallery \
  --scene examples/assets/PassiveActive.glb \
  --headless /tmp/parallel-mater-scene-check.ppm \
  --frames 180
```

The committed `PassiveActive.blend` contains a scaled passive bowl, an ACTIVE
Animated cube, and dynamic ACTIVE icosphere and Suzanne meshes. The detailed
bowl remains its own collision mesh; the three detailed dynamic objects carry
authored decimated proxies. Regression checks cover scale baking,
triangulation, proxy selection, kinematic targets, tilted gravity, toppling,
and containment.

## Collision behavior

`World::add_triangle_mesh` copies device vertices and indices, validates them,
and builds a deterministic private BVH. Mesh–mesh narrow phase uses exact
edge/triangle closest features with a small two-sided shell. When motion over a
substep exceeds that shell, conservative swept triangle-pair tests cover the
linearized previous-to-current vertex paths before the discrete solve. This
handles fast translation, rotation, open surfaces, and contacts from either
side, but it is not an exact analytic time-of-impact solution for curved
rotational trajectories. Substeps remain the accuracy control for extreme
angular motion and multiple impacts. The collision margin remains numerical
thickness rather than visible geometry.

Future rope and smoke schemas will be introduced only
with their reviewed public APIs. Unknown systems and schema versions fail
explicitly rather than silently changing scene meaning.

## Soft Body closed volume

Add Blender's **Physics → Soft Body** modifier to one closed, scene-root mesh.
The shared exporter writes its undeformed triangle surface as
`pm_system = "soft_body"`, including Blender mass, damping, and friction. The
runtime loader calls the API's `build_soft_body_geometry`: it subdivides long
surface edges to the same spacing as the HCP interior, then connects surface
and interior nodes into one spring graph. Every refined surface vertex is a
physical node, not interpolation between distant authored corners. Refinement
preserves the authored piecewise-planar shape, closed seams, normals and UVs;
it does not round or inflate the object. The installed API builds and advances
the lattice; the gallery only transfers rendering attributes.

Optional object properties tune conversion and the reusable solver:
`pm_node_spacing`, `pm_node_radius`, `pm_stretch_compliance`,
`pm_velocity_damping`, `pm_spring_damping`, `pm_contact_friction`,
`pm_shape_matching_stiffness`,
`pm_maximum_projection_fraction`, `pm_constraint_velocity_response`,
`pm_maximum_speed`, and `pm_solver_iterations`. Smaller spacing creates more
nodes and bonds. The gallery export defaults to the old lab's `2 m/s` soft-body
speed cap, `0.2` per-pass projection bound, `0.7` projection velocity response,
and 16 graph iterations. The first stage supports passive rigid triangle
collision. Active rigid triangle bodies use the same collision pass and receive
balanced reaction impulses automatically; no extra Blender property or
scene-specific force is needed. `SoftbodyRigidBody.blend` demonstrates two
active spheres contacting one soft body. Authored cloth and fluid systems can
also register the API's explicit soft-body coupling resources.

When Blender **Soft Body → Goal** is enabled **without a vertex group**, the exporter maps Default Weight
times Stiffness to `pm_shape_matching_stiffness`. ParallelMater interprets that
signal as co-rotated rest-shape matching rather than a world-space pin: the
body can translate and roll, compresses under load, and restores its authored
shape after the load leaves. Restoration yields during any substep with active
rigid contact, preventing the Goal projection from rebuilding through a
collider. A custom `pm_shape_matching_stiffness` overrides the Blender-derived
value; zero disables restoration.

With a Goal vertex group, full effective weight (`Min + weight * (Max - Min)`
equal to 1) instead exports an exact fixed node. This uses API inverse mass
zero, not a gallery force. Matching is position-based so triangulation and
glTF normal/UV seams do not lose pins. Refined vertices on an edge or face whose
original vertices are all pinned also stay fixed, preserving the attached area
rather than only its original corners. Partial Goal weights remain movable;
animated targets and weighted attachment springs are not implemented. A group
does not implicitly enable whole-body shape matching; the explicit custom
property remains available. See Blender's [Goal settings](https://docs.blender.org/manual/en/5.0/physics/soft_body/settings/goal.html).

`SoftbodyFluid.blend` demonstrates four pinned Goal vertices, one liquid inflow,
one liquid outflow, and passive rigid boundaries. Export it with the same
`export_parallel_mater_scene.py` script. The exporter does not modify the source blend.
Default lattice spacing retains the existing nine-sample thin-axis rule for
roughly isotropic bodies; for slabs it targets 18 samples along the long axis
while retaining at least three through the thickness. `pm_node_spacing`
overrides this resolution choice.

The fixture now explicitly authors a **coarser, firm material** on its soft
object: `pm_node_spacing = 0.12`, `pm_shape_matching_stiffness = 1.0`,
`pm_solver_iterations = 64`, `pm_maximum_projection_fraction = 0.5`, and
`pm_stretch_compliance = 0.0`. It uses 777 nodes rather than 2,104, with 944
supported surface triangles and 19 fixed nodes across the same Goal attachment.
Mass, gravity, source geometry and Goal weights are unchanged. This is material
configuration passed through the exporter and shared API, not a gallery force.

### Stiffening the exported soft body

On the soft object, use **Object Properties → Custom Properties** and re-export:

- `pm_shape_matching_stiffness = 0.35` restores the co-rotated rest shape while
  retaining the fixed Goal region. Increase toward 1 for stronger restoration;
  this is whole-body form recovery, not a local bending modulus.
- `pm_solver_iterations = 32` gives the spring graph more time to transmit load
  from the fixed region. Up to 64 is supported, at additional simulation cost.
- `pm_stretch_compliance` controls spring softness: smaller is stiffer, zero is
  the hard-constraint limit. The default `1e-7` is already nearly hard; simply
  reducing it cannot compensate for insufficient solver convergence.

Before the coarser preset, on the 2,104-node `SoftbodyFluid` fixture, after four seconds of gravity without
water, maximum node displacement was 1.79 m at defaults and 0.65 m with shape
matching 0.35 plus 32 graph iterations.
Shape matching 0.5 plus 64 iterations reduced it further to 0.38 m.
That firmer preset roughly doubled physics time in a short water-loaded run
(20.4 to 40.9 ms/frame on the test GPU); use 0.35/32 as a cheaper starting point.
These are displacement measurements (including rotation), not vertical sag or
material guarantees. The current 777-node preset holds the slab much closer to
its authored position: about 8 cm of unloaded sag after ten seconds, while
still flexing under the water stream. The stronger correction bound is a tested
setting for this material, not a new global solver default.

Blender's native **Edges → Pull, Push, Bending, Stiff Quads** affect Blender's
own solver but are **not currently exported** to ParallelMater. Do not expect
those controls to change this runtime. For Blender-only simulation, Pull/Push
resist stretching/compression and Bending resists angular deformation; damping
reduces motion, not static sag. See the [Blender Edges reference](https://docs.blender.org/manual/en/5.0/physics/soft_body/settings/edges.html).

## Cloth Shape Pin Group

`examples/assets/Cloth.blend` has a 33×33 subdivided sheet. Its top and
bottom rows belong to `FixedVertices`, selected in **Cloth → Shape → Pin
Group**, with pin stiffness `1.0`. The exporter writes the undeformed rest
mesh, not Blender's evaluated Cloth modifier, and records each weighted pin
coordinate in schema-2 `pm_system = "cloth"` metadata. Pin coordinates are
matched to glTF positions after triangulation, including vertices split by
UVs or normals. Weight 1 becomes zero inverse mass and is exactly fixed;
partial weights retain proportionate motion. Optional `pm_vertex_mass` and
`pm_thickness` object properties set the cloth's physical scale.
Keep the exported surface topologically connected: separate coincident
vertices are distinct solver particles even when they receive the same pin.

## Closed pressure cloth and contained fluid

For an air-filled closed cloth such as `ClothWater.blend`, enable **Cloth →
Physical Properties → Pressure**. Leave **Custom Volume** off to preserve the
authored initial volume, and leave Fluid Density at zero for air. Pressure
Scale maps to inverse volume compliance. A pressure cloth may be unpinned; the
exporter welds glTF vertices split only by render seams before building its
closed physics topology.

Set the cloth object's Boolean custom property `pm_contains_fluid` when a
Geometry-flow liquid volume belongs inside that cloth. The gallery translates
this relationship into `World::add_fluid_cloth_coupling`; containment,
reaction forces, and volume preservation live in the installed physics API,
not in scene-specific gallery code. Uniform Pressure Force is currently
required to remain zero because the API implements target-volume pressure,
not a separate constant inflation force.

The gallery creates the cloth through `World::add_cloth` and updates its
OptiX triangles each frame. The scene also contains the authored passive box
and active sphere. All four Cloth gallery entries start with gravity straight
down. Arrow keys steer it camera-relatively within a 45-degree tilt, returning
to straight down when released.
The API advances rigid bodies and cloth together at each substep. Cloth vertex
velocity damping (`ClothOptions::velocity_damping`, default 5/s) and
tangential contact friction (`contact_friction`, default 0.4) are configurable.
Rigid and cloth exchange friction impulses while touching, but separating
bodies shed tangential friction and are free to escape; cloth-side impulses
are limited to avoid local vertex pops.
Run `--cloth` for headless output or select Cloth with `Tab`; `R` resets it and
`F` shows rigid/cloth timings.

`ClothTear.blend` derives from the same sheet, authoring
`pm_break_strain = 0.96`, 16 persistent substeps,
`pm_impact_break_impulse = 0.001`, zero stretch compliance, and 24 solver
iterations. The sheet retains its original size. The 35 kg rigid sphere keeps
its original position and ground friction, so it falls first and can then be
rolled into the sheet with arrow-key gravity. The API breaks loaded bonds and
separates triangle-local surface faces without deleting them; the gallery does
not decide which bonds fail.

`SoftbodyCloth.blend` contains two independent cloths. Select a cloth object and
open **Object Properties → Custom Properties** to edit its ParallelMater settings:

- `Plane.002`, the bridge: `pm_break_strain = 0` keeps all bonds intact.
- `Plane.003`, the curtain: `pm_break_strain = 0.10` permits a bond to tear after
  extending 10% beyond its rest length for `pm_fracture_persistence_substeps = 4`
  consecutive substeps. It uses no impact-triggered fracture.
- Both retain their authored `PinEdge` groups and pin stiffness 1.0.
- `pm_solver_iterations` is 48 for the loaded bridge and 24 for the curtain.
  Increasing this improves spring convergence under load, at additional cost.

Blender's Cloth modifier does not supply a native tearing toggle for this
export contract; the custom properties travel through the same exporter used
by all scenes. Zero break strain and zero impact threshold mean unbreakable.
Change these properties, save, then export through **ParallelMater Scene**.
The gallery starts with gravity down; arrow steering tilts it up to 45°.
`X` includes cloth/soft-body reactions, `V` and `B` show the two systems' spring
and surface structures, and `M` includes both systems' contact forces.

`ClothPaint.blend` derives from the supplied `Cloth.blend`. It retains the pinned
top and bottom rows and the active rigid sphere, authors `pm_paintable = true`,
a 128×128 paint mask, `pm_paint_source` naming that sphere, and a 0.18-unit
world-space paint brush radius. The sphere has an authored forward velocity
so it can contact the cloth while gravity remains straight down. The gallery
binds the cloth UVs to a `World` paint field and registers a rigid-to-cloth
paint rule. Rigid–cloth collision and disk-shaped texel stamping are in the API;
the gallery chooses blue and cubic filtering, while the variant has a neutral
dry material so contact marks are visible. The cloth remains intact.
The example authoring helper `examples/assets/tools/make_cloth_variants.py`
preserves the original `.blend` file. It only creates source `.blend` variants;
export both through the common exporter above.

## Liquid Flow scene

`examples/assets/Fluid.blend` is the PR 7 source. It contains `Inflow` with
Blender **Fluid → Flow → Liquid → Inflow**, `Outflow` with Liquid Outflow, and a
sculpted `Plane` with **Rigid Body → Passive**. The exporter maps the Flow
modifiers to `pm_system = "fluid_inflow"` and `"fluid_outflow"` glTF nodes.
Plane geometry, transform, initial velocity, and flow behavior come from the
authored file. Blender's fluid solver and Domain are not used at runtime.

An inflow exports its actual triangle surface. The API subdivides it into
coarse sites; a site emits only after nearby water clears. Blender's enabled
Initial Velocity controls the new particles and naturally changes throughput.
Optional `pm_source_spacing` sets site spacing/clearance in metres; zero uses
the fluid support radius. The old `pm_particles_per_second` property is ignored.
No rectangular approximation is used for inflow. Outflow remains a rectangle.
The gallery uses a 30,000-particle capacity, 0.045 m particle radius, and 0.18 m
support radius; those are example settings, not hidden physics-world defaults.
The passive surface is exported as its authored triangles and collides on both
sides. Bright agitated surface particles visualize foam; the gallery also
reconstructs a continuous, example-only water surface.

Export with:

```bash
blender --background examples/assets/Fluid.blend \
  --python tools/blender/export_parallel_mater_scene.py -- \
  --output examples/assets/Fluid.glb
./build-gallery/parallel-mater-gallery --fluid
```

`examples/assets/FluidRigid.blend` adds a 4×4×4 Array of ACTIVE icospheres to
the same inflow, outflow, and passive triangle terrain. Export it with the
same script to `FluidRigid.glb`, then run the gallery with `--fluid-rigid`.
The 64 spheres are separate dynamic bodies, not one compound mesh; they share
one uploaded triangle mesh. Particle impacts change their linear and angular
velocity, while their moving triangles push particles and produce opt-in
contact events.

## One-shot Geometry flow and paint

`examples/assets/Pegs.blend` contains a closed cylinder with **Fluid → Flow →
Liquid → Geometry**, a passive bowl, four passive pegs, an invisible passive
containment cylinder, and one ACTIVE icosphere. The exporter marks the flow
cylinder as
`pm_system = "fluid_initial_volume"`. The gallery uses the public geometry
sampler to fill its authored mesh on a deterministic HCP lattice, then creates
those particles once when the scene starts. No Blender Domain or baked cache
is needed, and the flow does not emit on later frames. `P` changes the maximum
particle count and restarts; if lower than the authored fill, the gallery
selects an evenly distributed subset.

The Geometry-flow object can author `pm_particle_spacing` (metres) and
`pm_gravity_scale` as Blender custom properties. The Peg source uses 0.052 m
spacing and 2× standard gravity. Particle rest volume follows the HCP cell
volume, so denser sampling does not silently increase fluid mass. These are
general flow settings, not Peg-specific solver branches.

The Peg Paint example paints persistent blue masks from qualifying fluid
contacts with the bodies' exported UV surfaces. `World` owns the two-sided
paint field; the gallery renderer chooses blue and cubic filtering. The
exporter retains each render mesh's UV coordinates. Only the bowl has
`pm_paintable = true`; the active sphere and four pegs remain unpainted. Other
scenes do not allocate paint masks unless their authors opt in. Front and back
faces maintain separate paint channels, so
water touching the inside of a thin bowl does not tint its outside. Run with
`./build-gallery/parallel-mater-gallery --peg-paint`.
The four peg caps have a slight authored crown, giving resting droplets a
downhill path off the posts without peg-specific fluid forces.
The bowl and invisible containment cylinder use Blender rigid-body friction
`0.05`, while the pegs retain `0.5`. This lets water slide down the curved
wall instead of accumulating as a thin raised layer. Contact paint records
one texel per qualifying contact and reconstructs the mask with a smooth cubic
B-spline filter.
The bowl authors `pm_paint_resolution = 64`; its half-height UV chart uses
64×32 texels. Larger future paintable surfaces retain the default resolution.

The ACTIVE Icosphere is authored at 125 kg and starts above the bowl at
Y 0.4 m. At its 0.268 m radius, a 1 kg sphere floats in a water-density
fill; the heavier value falls through the water and settles against the bowl.
Keep mass and initial position as authored Rigid Body properties rather than
hidden scene-specific runtime overrides.
