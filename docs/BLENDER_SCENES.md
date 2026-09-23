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

The exporter rejects parented rigid bodies for now. Continuous collision,
compound bodies, and automatic convex decomposition are not part of this
milestone.

## Export the open `.blend`

Run this inside Blender's Scripting workspace, or from a shell:

```bash
blender --background examples/assets/PassiveActive.blend \
  --python tools/blender/export_parallel_mater_scene.py -- \
  --output examples/assets/PassiveActive.glb
```

When `--output` is omitted, the script writes a `.glb` beside the currently
open `.blend` using the same filename stem. The script is non-destructive:
it exports evaluated temporary copies, bakes each object's scale into its
vertices, triangulates all polygons, writes schema-2 glTF extras, and removes
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

Future fluid, cloth, soft-body, rope, and smoke schemas will be introduced only
with their reviewed public APIs. Unknown systems and schema versions fail
explicitly rather than silently changing scene meaning.

## Gate for the isolated-fluid milestone

PR 6 does not start—and no fluid kernels are run—until the authored source file
`examples/assets/FluidLifecycle.blend` is supplied. That file should contain:

- one mesh object named `FluidSeed` whose vertices, with no faces required,
  define the initial particle positions;
- one rectangular mesh named `SpawnPlane` defining the finite emission area;
- one rectangular mesh named `DestroyPlane` below it, positioned so particles
  must cross it during the example;
- a visible but non-colliding backdrop that makes the particle motion readable;
- enough separation between the two planes to observe neighbor rebuilding,
  fluid advancement, deterministic spawning, and swept destruction.

The initial review will agree on fluid capacity, particle radius, emission
rate, initial velocity, destroy direction, and custom-property names before the
exporter or loader is changed. Do not manually add provisional `pm_*` fluid
properties: the schema must follow the reviewed public API rather than define
it accidentally. Rigid containment belongs to the following coupling
milestone, so this first scene should demonstrate a free particle stream, not a
tank that depends on unimplemented fluid–rigid contacts.
