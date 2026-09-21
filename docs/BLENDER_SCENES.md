# Blender-authored triangle scenes

Gallery scenes are `.glb` assets, not C++ recipes. The same exported triangles
drive OptiX rendering and rigid collision; ParallelMater has no separate
sphere, box, capsule, or plane collider types.

## Authoring contract

1. Model every rigid object as a Blender mesh. Open, disconnected,
   non-manifold, and inconsistently wound surfaces are accepted; every triangle
   collides from both sides.
2. Use one Blender unit as one metre and keep simulated objects at the scene
   root.
3. Select each object and choose **Object → Rigid Body → Add Active** or
   **Add Passive**. Blender `ACTIVE` maps to `MotionType::dynamic`; `PASSIVE`
   maps to `MotionType::static_body`.
4. Set mass, friction, restitution, linear damping, angular damping, and an
   optional collision margin in Blender's Rigid Body panel.
5. Put the object origin near its intended center of mass. The current loader
   recenters local geometry from its AABB and preserves the resulting world
   pose. A zero API inertia uses the mesh AABB approximation; an application
   can supply a measured inertia diagonal later.

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

Users do not need to type these properties by hand. Names are labels only;
physics behavior comes from Blender's Rigid Body settings.

## Validate the result

```bash
./build-gallery/parallel-mater-gallery \
  --scene examples/assets/PassiveActive.glb \
  --headless /tmp/parallel-mater-scene-check.ppm \
  --frames 180
```

The committed `PassiveActive.blend` contains a scaled passive ground plane and
active cube, icosphere, and Suzanne. Its regression checks that scale was
baked, polygons became triangles, all four objects use World-owned collision
meshes, gravity advances every ACTIVE object, and Suzanne topples and remains
supported.

## Collision behavior

`World::add_triangle_mesh` copies device vertices and indices, validates them,
and builds a deterministic private BVH. Mesh–mesh narrow phase uses exact
edge/triangle closest features with a small two-sided shell. This is a discrete
solver: sufficiently fast motion can tunnel, so applications must choose a
substep count appropriate for speed and triangle scale. The collision margin
is numerical thickness, not a replacement for continuous collision detection.

Future fluid, cloth, soft-body, rope, and smoke schemas will be introduced only
with their reviewed public APIs. Unknown systems and schema versions fail
explicitly rather than silently changing scene meaning.
