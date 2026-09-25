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

## Export the open `.blend`

Run this inside Blender's Scripting workspace, or from a shell:

```bash
blender --background examples/assets/PassiveActive.blend \
  --python tools/blender/export_parallel_mater_scene.py -- \
  --output examples/assets/PassiveActive.glb
```

When `--output` is omitted, the script writes a `.glb` beside the currently
open `.blend` using the same filename stem. The script is non-destructive:
it exports evaluated rigid copies and undeformed cloth rest copies, bakes scale
into their vertices, triangulates all polygons, writes schema-2 glTF extras, and removes
the temporary data. The source `.blend` is not saved or changed.

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
| `pm_checkerboard` | Optional source custom property; defaults on for passive objects in the example exporter |
| `pm_paintable` | Optional Boolean source custom property; gallery registers a persistent API paint field and fluid-to-rigid rule for that body |
| `pm_paint_resolution` | Optional integer 32–2048; square mask resolution (default 512) for a paintable body |
| `pm_collision_proxy` | Optional source custom property naming a lower-resolution Blender mesh |

When a proxy is selected, the exporter adds a non-rendered
`pm_system = "collision_mesh"` glTF node and references it from the rigid-body
node. The gallery still renders the detailed mesh. Proxies are explicit
authored data: the loader does not decimate or invent collision geometry.

Users do not need to type these properties by hand. Names are labels only;
physics behavior comes from Blender's Rigid Body settings.

## Validate the result

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

Future soft-body, rope, and smoke schemas will be introduced only
with their reviewed public APIs. Unknown systems and schema versions fail
explicitly rather than silently changing scene meaning.

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

The gallery creates the cloth through `World::add_cloth` and updates its
OptiX triangles each frame. The scene also contains the authored passive box
and active sphere. In the Cloth gallery entry, gravity points 45 degrees
downward and toward the sheet. Run `--cloth` for headless output or select
Cloth with `Tab`; `R` resets it and `F` shows rigid/cloth timings.

## Liquid Flow scene

`examples/assets/Fluid.blend` is the PR 7 source. It contains `Inflow` with
Blender **Fluid → Flow → Liquid → Inflow**, `Outflow` with Liquid Outflow, and a
sculpted `Plane` with **Rigid Body → Passive**. The exporter maps the Flow
modifiers to `pm_system = "fluid_inflow"` and `"fluid_outflow"` glTF nodes.
Plane geometry, transform, initial velocity, and flow behavior come from the
authored file. Blender's fluid solver and Domain are not used at runtime.

By default an inflow emits 2,400 particles/s. Set the optional Blender custom
property `pm_particles_per_second` on the inflow object to change it. The
gallery uses a 30,000-particle capacity, 0.045 m particle radius, and 0.18 m
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
