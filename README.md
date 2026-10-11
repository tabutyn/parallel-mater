# ParallelMater

ParallelMater is an MIT-licensed GPU physics library. CUDA is the complete
reference backend; Apple Metal 4 and Windows D3D12 backends are under active
development. The
first complete milestone couples particle fluid with triangle rigid bodies; later solvers
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
Generation-checked rigid constraints provide fixed, point, hinge, slider,
piston, generic, generic-spring, and motor joints with runtime updates,
breaking thresholds, limits, springs, and collision suppression.

CUDA and Metal rigid contacts and joints use a shared [AVBD body-block solver](docs/CONTACT_SOLVER.md).
Its production numerical core is generated from one Slang module for both
backends and checked against the portable CPU reference. There is no
stack-only or joint-triggered fallback solver. Impact velocity correction uses
the same body blocks and is included in the reported iteration count.

The [Slang implementation](docs/SLANG.md) also contains a same-basename,
same-entry-point mirror for every CUDA physics kernel: rigid AVBD and contact
geometry, fluid, cloth, soft bodies, ropes, smoke, and every coupling. A CMake
gate compiles all 120 mirrors to CUDA and PTX and a coverage test prevents a
native CUDA entry point from being added without its Slang counterpart.

The newly integrated D3D12 backend retains its existing solver; it has not yet
been migrated to AVBD.

Measured performance, validation limits and earlier solver experiments are
recorded in [the rigid performance report](docs/PERFORMANCE.md). Historical
timings in that report are not measurements of the current AVBD implementation.

## Design goals

- One owning `World` coordinates simulation and cross-system coupling.
- Fluid, cloth, soft-body, rigid-body, and rigid-constraint resources use
  stable, generation-checked handles.
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

CUDA and Metal compile the standalone fluid solver and AVBD numerical core
from `src/slang/fluid_shared.slang` and `src/slang/avbd.slang`. The complete
CUDA-reference mirror lives beside them under `src/slang/`; build target
`parallel-mater-slang-kernel-mirrors` validates every entry. CMake resolves one
exact Slang 2026.18 compiler from `PATH` or downloads the matching
checksum-pinned official package. Set
`PARALLEL_MATER_SLANGC_EXECUTABLE` to an exact compiler path for offline builds.
The native Apple Metal compiler and NVCC still validate and compile Slang's
generated source; no Slang runtime is shipped. See the [Slang toolchain and
source layout](docs/SLANG.md).

### Windows D3D12 rigid-body gate

Windows defaults to the D3D12 backend and does not require CUDA, NVIDIA
drivers, OptiX, DXR, or the CUDA toolkit. It requires the Windows 10 SDK's FXC,
Shader Model 5.1, and 64 UAV slots. Resource Binding Tier 2+ qualifies at any
D3D12 feature level; Tier 1 qualifies at feature level 11_1 or newer. The
installed target is `ParallelMater::d3d12`, with its public API in
`<parallel_mater/d3d12.hpp>`.

```powershell
cmake -S . -B build-d3d12 -A x64 `
  -DPARALLEL_MATER_BUILD_CUDA=OFF `
  -DPARALLEL_MATER_BUILD_D3D12=ON `
  -DPARALLEL_MATER_BUILD_D3D12_GALLERY=ON `
  -DBUILD_TESTING=ON
cmake --build build-d3d12 --config Release
ctest --test-dir build-d3d12 -C Release --output-on-failure
.\build-d3d12\Release\parallel-mater-d3d12-gallery.exe `
  --all-rigid-scenes --headless --validate `
  --output build-d3d12\captures
```

The first gate covers rigid integration, contacts, all eight constraint types,
the eight current rigid gallery entries, adapter diagnostics, headless PPM
capture, and repeated deterministic validation. Non-rigid entry points already
exist and return transactional `not_supported` until their staged solver gate.
The backend remains experimental: full CUDA conformance and WARP validation
are not complete. In the native viewer, press `F` for FPS and GPU physics
timings, `Tab` for the scene catalog, and arrows/Enter or click to select a
supported scene. Gray catalog entries require unimplemented subsystems.
See [the D3D12 backend guide](docs/D3D12.md).

### Metal 4 foundation

On Apple Silicon with macOS 26 and Xcode 26, CMake defaults to the Metal target
and does not require a CUDA toolkit:

```bash
cmake -S . -B build-metal \
  -DPARALLEL_MATER_BUILD_CUDA=OFF \
  -DPARALLEL_MATER_BUILD_METAL=ON \
  -DBUILD_TESTING=ON
cmake --build build-metal
ctest --test-dir build-metal --output-on-failure
```

The installed target is `ParallelMater::metal`, with its public API in
`<parallel_mater/metal.hpp>`. The current correctness milestone implements the
Metal 4 queue/frame lifecycle, embedded MSL 4 shaders (including the shared
Slang fluid solver), rigid bodies and all
eight constraint types, fluid, cloth, soft bodies, ropes, particle/grid smoke,
fluid sources/outflow, rigid contact for every particle family, and the exposed
pairwise coupling handles. The native Metal gallery loads the shared 29-entry
registry and renders rigid bodies, live cloth/soft-body surfaces, rebuilt rope
tubes, fluid particles, and smoke through embedded MSL shaders. CUDA/Metal
numerical parity, production parallel kernels, and ray-traced visual parity
remain gated; see [Metal port status](docs/METAL.md), the [gallery parity
audit](docs/METAL_GALLERY_PARITY.md), and the [CUDA reference package needed
for engine parity](docs/CUDA_REFERENCE_REQUIREMENTS.md).

Run the gallery on Apple Silicon with:

```bash
cmake -S . -B build-metal-gallery \
  -DPARALLEL_MATER_BUILD_CUDA=OFF \
  -DPARALLEL_MATER_BUILD_METAL=ON \
  -DPARALLEL_MATER_BUILD_METAL_GALLERY=ON \
  -DBUILD_TESTING=ON
cmake --build build-metal-gallery
./build-metal-gallery/parallel-mater-metal-gallery
./build-metal-gallery/parallel-mater-metal-gallery \
  --cloth-tear
./build-metal-gallery/parallel-mater-metal-gallery \
  --all-scenes --frames 1 --headless --validate \
  --output build-metal-gallery/captures
```

Build and atomically replace the Dock-pinned development app with:

```bash
cmake --build build-metal-gallery --target deploy-metal-gallery
```

The deployed window title and `--version` identify the exact Git commit,
working-tree state, and build time. Quit and reopen an already-running gallery
after deployment because macOS keeps its current executable mapped in memory.
The application bundle carries its gallery GLBs in `Contents/Resources`, so a
deployed build does not read scene assets from the source checkout or request
access to its external drive.

Measure the opening brick scene using:

```bash
cmake --build build-metal-gallery \
  --target parallel-mater-metal-rigid-scene-benchmark
./build-metal-gallery/parallel-mater-metal-rigid-scene-benchmark
./build-metal-gallery/parallel-mater-metal-rigid-scene-benchmark --scenario steering --mode sleep
./build-metal-gallery/parallel-mater-metal-rigid-scene-benchmark --scenario impact --mode sleep
```

The sleeping flags remain accepted, but AVBD currently keeps rigid bodies
awake. Native Metal results from before this migration do not validate the new
solver; see [the current solver contract](docs/CONTACT_SOLVER.md).

Use `--list-scenes` to list every scene selector. In the interactive gallery,
Tab opens the CUDA-style scene page, Up/Down changes its selection, Enter loads
the highlighted scene, the mouse orbits/pans/zooms, arrows run the scene's
control policy, Space performs its scene action, `R` reloads, and Escape closes
the scene page or exits. Headless rendering writes PPM captures and uses the
same embedded Metal pipelines as the windowed path.

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

For AVBD-budget experiments, use `--contact-passes 1` in the CUDA gallery,
or `--passes 1` with `parallel-mater-rigid-scene-benchmark examples/assets/RigidBody.glb`.
The shared CUDA/Metal API is `StepOptions::rigid_contact_pass_limit`: zero keeps
the automatic budget, 1–64 selects iterations per pose/impact phase for both
contacts and joints. Impacted islands may therefore report twice that value;
integration substeps are unchanged. Low budgets can destabilize stacks. Check
quality with `parallel-mater-rigid-wall-tests --passes 1`, not only timing.
The overlay labels `X` as GPU launches and reports total AVBD iterations in
the last substep separately.

Left-drag orbits, Shift+left-drag pans, the wheel zooms, and `R` resets the active scene. `Tab` opens
the selector for Rigid Body, six Constraint scenes, DUMP, Fluid, Fluid + Rigid, Peg Paint, Cloth,
Cloth Tear, Cloth Paint, Water Cloth, Soft Body, and Soft Body Rigid; use
Up/Down and Enter to switch.
Fixed collects loose spheres on contact. Point starts with four spheres
orbiting a shared anchor in two perpendicular pairs; `Space` releases or
reattaches all four. Arrow keys tilt gravity relative to the current camera in
Constraint scenes except Motor + Spring. Up/Down drives all four Motor + Spring
wheels together. Left/Right yaws each hub and wheel around
the suspension's up axis, with rear steering opposite the front. Hinge launches a sphere into a panel merged with the first of three
hinged gears; contact propagates through both interfaces with alternating
rotation. Slider, Piston, and Generic launch an
authored sphere impact automatically.
Rigid Body loads `RigidBody.blend`: two evaluated Array stacks become 384
independent bricks sharing two meshes. Arrows tilt the sphere's gravity relative
to the current camera to drive it through the wall while the bricks retain
vertical gravity. The authored `LoadBox`
is available as a non-colliding hit-box query volume. In
DUMP, arrows drive and steer the suspended dump truck; Space tips its open-top
rear bucket 100 degrees, and another press lowers it. Press `P` to edit its
10–1,000 sphere load centered on the Blender Empty `SphereCluster`; applying a
count restarts DUMP. `F` toggles per-kernel
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
deforms it against the authored passive triangle arena. Blender Goal settings
drive rotation-invariant shape restoration without preventing rolling and
yield while a dynamic collider is actively loading the lattice. It starts with
gravity straight down and shares arrow steering, reset, timing, capture, and
deformable debug controls. `--soft-body-rigid` loads
`SoftbodyRigidBody.blend` and uses the
same API solver while two active spheres exchange balanced linear and angular
contact impulses with the lattice.
`--soft-body-cloth` loads `SoftbodyCloth.blend`: a soft sphere lands on an intact
cloth bridge over a pit and can be rolled into a separate tearable curtain.
Each cloth carries its own `pm_break_strain` Blender custom property. The shared
API handles contacts, friction, and fracture, with gravity initially down.
`--soft-body-fluid` loads `SoftbodyFluid.blend`: the Soft Body Goal group fixes
the authored attachment while an inflow loads its triangle surface and water
drains through the outflow. Fluid/soft-body contact is an explicit API resource;
the gallery reuses its normal water surface, foam, controls, and diagnostics.
Every entry uses the same physics diagnostics: `Z` toggles contact/deformable
normals, `X` rigid and deformable contact forces, `C` fluid acceleration/reaction
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
