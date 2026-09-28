# ParallelMater

ParallelMater is an MIT-licensed CUDA C++ physics library. The first complete
milestone couples particle fluid with triangle rigid bodies; later solvers
will be added only after the small public API is proven by gallery examples.

The current implementation provides the `World` lifecycle and GPU rigid-body
integration for static, kinematic, and dynamic indexed triangle meshes. Every
surface is two-sided; open and disconnected meshes are accepted. A
deterministic private BVH accelerates mesh contact. Particle fluid, Blender
Liquid Inflow/Outflow, moving and passive triangle collision, two-way dynamic
momentum exchange, and agitation foam are also available through the same
`World`.
One-shot Blender Geometry flows use the public volume sampler. Opt-in
fluid-to-rigid and rigid-to-cloth contact paint fields are owned by `World`; the gallery
supplies UVs and chooses their display color and filtering.

The current rigid pipeline reduced the measured five-body Blender scene from
24.10 ms to 1.81 ms median GPU time on the local RTX 3050 Ti. The retained and
rejected experiments, sparse-world result, and high-speed fixture are recorded
in [the rigid performance report](docs/PERFORMANCE.md); these are project
measurements, not general hardware claims.

## Design goals

- One owning `World` coordinates simulation and cross-system coupling.
- Fluid, cloth, soft-body, and rigid-body resources use stable, generation-checked handles.
- One `step` call advances a complete fixed frame; applications do not invoke
  solver-internal phases.
- CUDA allocations remain owned by the library while renderers borrow explicit
  device views.
- Fluids accept device-resident initial particles or one-shot host geometry
  volumes, and can own deterministic, capacity-bounded inflow/outflow planes.
- Synchronous convenience calls and stream-ordered asynchronous calls share the
  same semantics.
- The gallery is simultaneously the example suite, visual regression surface,
  and basis of a small progression game.
- Rendering, OptiX, scene objectives, authored controls, and campaign state are
  application code—not physics API concepts.

Start with [the proposed API](docs/API.md), then review
[the gallery boundary](docs/GALLERY.md) and [the staged implementation
plan](docs/ROADMAP.md).

## Build and test

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The runtime test skips with code 77 when no CUDA device is available. Compute
capability `86` is the local RTX 3050 Ti setting; consumers should select the
architectures they ship.

## Blender and OptiX gallery

The optional gallery loads a committed Blender-authored `.glb`, creates its
rigid bodies through the public API, and ray traces its render meshes with
OptiX. It is deliberately separate from the installed physics library.

All Blender scenes use one exporter,
`tools/blender/export_parallel_mater_scene.py`: as a Blender File → Export menu
entry, a headless script, or its `export_scene()` Python function. See the
[Blender interface guide](docs/BLENDER_SCENES.md). If Blender is installed when
configuring the gallery, CTest also checks fresh exports against the runtime
loader without changing the source assets.

```bash
cmake -S . -B build-gallery \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPARALLEL_MATER_BUILD_OPTIX_GALLERY=ON \
  -DBUILD_TESTING=ON
cmake --build build-gallery
ctest --test-dir build-gallery --output-on-failure
./build-gallery/parallel-mater-gallery
```

Left-drag orbits, Shift+left-drag pans, the wheel zooms, and `R` resets the active scene. `Tab` opens
the selector for Rigid Body, DUMP, Fluid, Fluid + Rigid, Peg Paint, Cloth,
Cloth Tear, Cloth Paint, Water Cloth, and Soft Body; use
Up/Down and Enter to switch.
In Rigid Body, arrow keys move the authored kinematic Cube and tilt gravity. In
DUMP, hold Left Arrow to rotate the hopper clockwise and press `P` to edit its
10–1,000 sphere count; applying a count restarts DUMP. `F` toggles per-kernel
GPU timings. Fluid uses the supplied
`Fluid.blend` scene and displays blue particles with white surface foam. Escape
quits. In Peg Paint, arrows or WASD tilt the authored gravity relative to the
camera; release them to return it smoothly to vertical. Display-free scene
renders are available through CLI flags. Cloth uses `Cloth.blend`, keeps its
two `FixedVertices` rows pinned. All four Cloth scenes start with gravity
straight down; arrow keys steer it relative to the camera within a 45-degree
tilt, and releasing them restores straight-down gravity.
`--cloth-tear` lets a heavy sphere land, then roll through bond-fractured cloth
when gravity is steered toward the sheet. No cloth triangles are deleted.
`--cloth-paint` keeps the
cloth intact while the active sphere presses and paints it through the public
paint-field API.
`--water-cloth` loads `ClothWater.blend`: Blender Cloth Pressure preserves the
closed unpinned sphere's authored volume while an explicit API coupling keeps
its Geometry-flow water inside and transfers equal-and-opposite forces back to
the cloth. Its containing skin is rendered as refractive transparent water.
`--soft-body` loads `Softbody.blend`: the exporter marks its native Blender Soft
Body surface, the loader builds a volumetric spring lattice, and the public API
deforms it against the authored passive triangle arena. It starts with gravity
straight down and shares arrow steering, reset, timing, capture, and deformable
debug controls.
Every entry uses the same physics diagnostics: `Z` toggles contact/deformable
normals, `X` rigid and rigid-contact forces, `C` fluid acceleration/reaction
forces, `N` velocities, and `M` writes the rolling physics capture to `/tmp`.
`V` exposes particles, cloth structure, or every soft-body spring where those
systems exist; `B` shows cloth bonds or the soft-body surface wireframe. The
gallery opts into capture when it creates each `World`;
ordinary library consumers pay no capture/readback cost unless they do the
same. `P`, `F`, arrow-gravity steering, and `R` use the common controls.

```bash
./build-gallery/parallel-mater-gallery \
  --headless /tmp/parallel-mater-gallery.ppm --frames 120

./build-gallery/parallel-mater-gallery \
  --dump-spheres 100 \
  --headless /tmp/parallel-mater-dump.ppm --frames 120

./build-gallery/parallel-mater-gallery \
  --fluid --headless /tmp/parallel-mater-fluid.ppm --frames 120

./build-gallery/parallel-mater-gallery \
  --cloth --headless /tmp/parallel-mater-cloth.ppm --frames 180

./build-gallery/parallel-mater-gallery \
  --water-cloth --cloth-tilt-left --frames 180 \
  --physics-capture /tmp/water-cloth-physics.log \
  --headless /tmp/water-cloth.ppm
```

The gallery currently requires an NVIDIA driver supported by OptiX 9.1,
OpenGL, and GLFW. CMake fetches pinned cgltf and OptiX header revisions only
when the optional gallery is enabled. See [the Blender scene contract](docs/BLENDER_SCENES.md)
before authoring or exporting another scene.
