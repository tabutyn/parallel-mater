# Gallery and progression game

The gallery is an executable examples package built on the public API. Scene
geometry, materials, transforms, and declarative physics metadata come from
Blender-authored `.glb` files; C++ provides reusable loading, stepping, and
rendering rather than rebuilding each scene procedurally. Each scene must be
understandable in isolation and reusable by the progression game, but none of
its types are installed with the physics library.

## Boundary

```text
parallel_mater (installed library)
  World, FluidId, RigidBodyId, device views, contacts

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

1. **Rigid sandbox** — a dynamic triangle mesh falls onto a triangle floor.
2. **Fluid tank** — particle water settles inside static rigid boundaries.
3. **Heavy sphere** — a dynamic sphere enters the fluid and receives visible
   two-way reaction forces.
4. **Obstacle bowl** — gravity steering rolls the sphere through pegs while
   fluid remains contained.

The procedural `parallel-mater-rigid-sandbox` remains a minimal headless API
example. The visible `parallel-mater-gallery` instead loads
`examples/assets/PassiveActive.glb`, instantiates its passive ground, kinematic
Cube, and dynamic Icosphere and Suzanne through `World`, and ray traces the same
authored triangles. Arrow input moves the Cube and tilts gravity for the two
dynamic bodies. C++ does not restate that scene's body list or transforms.

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

Physics particles and rendering are separate. Rigid meshes are rendered now;
continuous water is a later renderer milestone with two implementations
evaluated independently:

- screen-space or scalar-field reconstruction for broad compatibility;
- OptiX ray tracing on supported NVIDIA desktop/server GPUs.

The gallery chooses a renderer at runtime. Neither implementation changes
fluid stepping or the public physics API.
