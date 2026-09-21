# Blender-authored gallery scenes

Gallery scenes are `.glb` assets, not C++ recipes. The gallery loader translates
authored node metadata into public `parallel_mater::World` calls, while rendering
geometry and materials remain example-owned data.

## Coordinate and modeling rules

- Use Blender's default metric convention: one Blender unit is one metre.
- Keep every simulated object at the scene root and give it a centered origin.
- Apply mesh edits in Edit Mode. Object translation and rotation become the
  initial rigid-body pose; object scale is baked into render geometry and the
  collider dimensions by the loader.
- Use uniform scale for spheres. Capsules use local Z as their long axis in
  Blender, which the glTF exporter converts to local Y.
- A plane collider is mathematically infinite. Its mesh should therefore be
  large enough that the visible boundary is not reached in the example.
- Use triangle geometry. The loader rejects unsupported primitive modes and
  malformed or non-finite data instead of guessing.

## Required custom properties

Select an object and use **Object Properties → Custom Properties**. Blender's
glTF exporter writes these properties to the node's standard `extras` object
when **Include → Custom Properties** is enabled.

| Property | Values | Meaning |
|---|---|---|
| `pm_schema` | integer `1` | Metadata schema version. |
| `pm_system` | `rigid_body` | Solver that owns this object. |
| `pm_motion` | `static`, `kinematic`, `dynamic` | Rigid-body motion type. |
| `pm_collider` | `sphere`, `box`, `capsule`, `plane`, `triangles` | Collision representation. |
| `pm_mass` | positive number | Required for dynamic objects; kilograms. |
| `pm_friction` | nonnegative number | Optional; defaults to `0.5`. |
| `pm_restitution` | number from `0` to `1` | Optional; defaults to `0`. |
| `pm_checkerboard` | Boolean | Optional example-renderer material effect. |

Names are labels only. Physics behavior never depends on an object being named
"Floor" or "Sphere".

## Scripted example

The repository script creates and exports the canonical rigid scene:

```bash
blender --background \
  --python tools/blender/create_rigid_shapes_scene.py -- \
  --output examples/assets/rigid_shapes.glb
```

It produces a static checkerboard plane, dynamic sphere, box, and capsule, and
a static Suzanne using her actual triangles as the collider.
The generated `.glb` is committed so users can run the gallery without Blender;
the Python script is its reviewable source of truth.

For a hand-authored scene, set the same properties, then choose **File → Export
→ glTF 2.0**, select **glTF Binary (.glb)**, enable **Custom Properties**, and
export. Use the repository validator before adding the asset to the gallery.

Validate an exported scene by loading and rendering it without opening a
window:

```bash
./build-gallery/parallel-mater-gallery \
  --scene path/to/scene.glb \
  --headless /tmp/parallel-mater-scene-check.ppm \
  --frames 1
```

The command fails on malformed geometry, unsupported metadata, invalid
colliders, world-capacity errors, OptiX setup errors, or a black/uniform frame.

## Future systems

The `pm_system` discriminator reserves a clean extension point for `fluid`,
`cloth`, `soft_body`, `rope`, and `smoke`. Each system will receive a reviewed
schema only when its public physics API exists. Unknown systems or schema
versions are errors; the loader does not silently create a different scene.

## Triangle colliders

`pm_collider = "triangles"` uploads every triangle primitive on that node into
one World-owned collision resource. It is a two-sided triangle soup: meshes may
be open, disconnected, non-manifold, or inconsistently wound. Degenerate
triangles, invalid indices, and non-finite vertices are rejected. Triangle
colliders are static or kinematic in this release; dynamic triangle meshes need
defined mass properties and mesh–mesh collision and are intentionally rejected.

Triangle collision is currently exact but brute force. This favors a small,
reviewable correctness baseline for authored obstacles such as Suzanne. A
deterministic acceleration structure is required before large environment
meshes become a supported performance target.
