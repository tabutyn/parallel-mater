# ParallelMater

ParallelMater is an MIT-licensed CUDA C++ physics library. The first complete
milestone will couple particle fluid with analytic rigid bodies; later solvers
will be added only after the small public API is proven by gallery examples.

The current implementation provides the `World` lifecycle and GPU rigid-body
integration for static, kinematic, and dynamic spheres, boxes, capsules, and
planes. Fluid declarations are present for API review but are implemented in
the next milestone.

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
./build/parallel-mater-rigid-sandbox
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

Left-drag orbits, the wheel zooms, `R` restores the authored poses, and Escape
quits. A display-free render is also available:

```bash
./build-gallery/parallel-mater-gallery \
  --headless /tmp/parallel-mater-gallery.ppm --frames 120
```

The gallery currently requires an NVIDIA driver supported by OptiX 9.1,
OpenGL, and GLFW. CMake fetches pinned cgltf and OptiX header revisions only
when the optional gallery is enabled. See [the Blender scene contract](docs/BLENDER_SCENES.md)
before authoring or exporting another scene.
