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

The current rigid pipeline reduced the measured five-body Blender scene from
24.10 ms to 1.81 ms median GPU time on the local RTX 3050 Ti. The retained and
rejected experiments, sparse-world result, and high-speed fixture are recorded
in [the rigid performance report](docs/PERFORMANCE.md); these are project
measurements, not general hardware claims.

## Design goals

- One owning `World` coordinates simulation and cross-system coupling.
- Fluid and rigid-body resources use stable, generation-checked handles.
- One `step` call advances a complete fixed frame; applications do not invoke
  solver-internal phases.
- CUDA allocations remain owned by the library while renderers borrow explicit
  device views.
- Fluids accept device-resident initial particles and can own deterministic,
  capacity-bounded particle spawn and destroy planes.
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
the selector for Rigid Body, DUMP, and Fluid; use Up/Down and Enter to switch.
In Rigid Body, arrow keys move the authored kinematic Cube and tilt gravity. In
DUMP, hold Left Arrow to rotate the hopper clockwise and press `P` to edit its
10–1,000 sphere count; applying a count restarts DUMP. `F` toggles per-kernel
GPU timings and `V` toggles rigid contact diagnostics. Fluid uses the supplied
`Fluid.blend` scene and displays blue particles with white surface foam. Escape
quits. Display-free DUMP and Fluid renders are available through CLI flags:

```bash
./build-gallery/parallel-mater-gallery \
  --headless /tmp/parallel-mater-gallery.ppm --frames 120

./build-gallery/parallel-mater-gallery \
  --dump-spheres 100 \
  --headless /tmp/parallel-mater-dump.ppm --frames 120

./build-gallery/parallel-mater-gallery \
  --fluid --headless /tmp/parallel-mater-fluid.ppm --frames 120
```

The gallery currently requires an NVIDIA driver supported by OptiX 9.1,
OpenGL, and GLFW. CMake fetches pinned cgltf and OptiX header revisions only
when the optional gallery is enabled. See [the Blender scene contract](docs/BLENDER_SCENES.md)
before authoring or exporting another scene.
