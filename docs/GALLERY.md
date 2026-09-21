# Gallery and progression game

The gallery is an executable examples package built on the public API. Each
scene must be understandable in isolation and reusable by the progression
game, but none of its types are installed with the physics library.

## Boundary

```text
parallel_mater (installed library)
  World, FluidId, RigidBodyId, device views, contacts

examples/gallery (not installed)
  scene setup, assets, renderer adapters, controls, objectives

apps/gallery_game (not installed)
  scene selection, progression, save data, menus
```

The renderer consumes borrowed device views. The initial native renderer may
use CUDA/OpenGL interop for debug particles. A later OptiX renderer belongs in
`examples/support/optix_renderer`; it can ray trace reconstructed water and
rigid render meshes without making OptiX a dependency of the physics package.
Headless server simulations build without OpenGL or OptiX.

## First gallery sequence

1. **Rigid sandbox** — a dynamic sphere rolls and collides with static boxes.
2. **Fluid tank** — particle water settles inside static rigid boundaries.
3. **Heavy sphere** — a dynamic sphere enters the fluid and receives visible
   two-way reaction forces.
4. **Obstacle bowl** — gravity steering rolls the sphere through pegs while
   fluid remains contained.

The first scene is available now as the headless
`parallel-mater-rigid-sandbox` example. It launches a sphere across a
high-friction plane into a static box and prints sampled device-simulated state.
It intentionally has no renderer; rendering is a later gallery milestone.

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

Physics particles and rendering are separate. Debug particle spheres are the
first visualization. Continuous water is a later renderer milestone with two
implementations evaluated independently:

- screen-space or scalar-field reconstruction for broad compatibility;
- OptiX ray tracing on supported NVIDIA desktop/server GPUs.

The gallery chooses a renderer at runtime. Neither implementation changes
fluid stepping or the public physics API.
