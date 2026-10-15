# ParallelMater

ParallelMater is an MIT-licensed GPU physics API. Applications create an
instance, enumerate and create a device, bind memory, create resources and a
physics pipeline, record commands, submit them to a queue, synchronize, and
read timing or memory queries through `parallel_mater.h`.

Applications link the `parallel-mater` loader. The loader discovers installable
Parallel Mater drivers, aggregates their physical devices, loads optional
layers, and dispatches commands to the driver that owns each device. It contains
no GPU API, allocator, shader compiler, simulation schedule, or physics code.

One set of Slang shaders implements rigid bodies, particle fluid, cloth, rope,
soft bodies, smoke, and their couplings. CUDA is the first runtime device
adapter; it owns allocation, transfer, dispatch, synchronization, and native
interop, but no physics equations or alternate simulation schedule. Unsupported
targets are reported explicitly and never fall back to an older solver.
Generation-checked rigid constraints provide fixed, point, hinge, slider,
piston, generic, generic-spring, and motor joints with runtime updates,
breaking thresholds, limits, springs, and collision suppression.

Rigid contacts and joints use a shared [AVBD body-block solver](docs/CONTACT_SOLVER.md).
Its production numerical core and runtime schedule are generated from
`geometry_constraints.slang` and checked against the portable CPU reference. There is no
stack-only or joint-triggered fallback solver. Impact velocity correction uses
the same body blocks and is included in the reported iteration count.

The [Slang implementation](docs/SLANG.md) is the production source for rigid
AVBD and triangle/BVH contact geometry, fluid, cloth, soft bodies, ropes,
smoke, and every coupling. A CMake gate compiles the shared entries to CUDA and
PTX; an architecture test rejects native physics kernels and direct launches.
Current qualification evidence and target acceptance are recorded in the
[Slang unification validation report](docs/slang-unification-validation.md).

Measured performance, validation limits and earlier solver experiments are
recorded in [the rigid performance report](docs/PERFORMANCE.md). Historical
timings in that report are not measurements of the current AVBD implementation.

## Design goals

- One public C command API and one Slang implementation serve every physics
  domain.
- Explicit devices, queues, memory bindings, pipelines, command buffers,
  barriers, semaphores, fences, and queries compose with a larger GPU pipeline.
- Resource capacity and ownership are explicit; command submission remains
  asynchronous.
- Device adapters provide platform services only. Renderers draw common frame
  data and cannot step physics or select scenes.
- One gallery catalog, input handler, scene builder, and controller path drives
  interactive, headless, benchmark, and replay execution.

Start with [the API](docs/API.md), then review [the gallery
boundary](docs/GALLERY.md) and [the migration plan](docs/ROADMAP.md).

The [command API reference PDF](docs/reference/parallel-mater-reference.pdf)
and [API specification](docs/reference/parallel-mater-specification.pdf)
describe the Vulkan-style API: explicit devices, allocation, physics pipelines,
command buffers and synchronization. The reference lists declarations;
its purple section markers point to explanations and rules in the specification.
The `parallel-mater` loader exposes the public C command API in
`parallel_mater/parallel_mater.h`. The first installable driver,
`parallel_mater_driver_slang_cuda`, implements device, memory, command,
synchronization, query, native-buffer, and physics services. Driver discovery,
the private ABI, manifests, and vendor requirements are documented in the
[driver SDK](docs/DRIVER_SDK.md). Features are advertised only when their
shared Slang schedule is present.

On Windows, `parallel_mater_driver_slang_d3d12` is also loader-visible. It
implements D3D12 memory, buffers, transfers, barriers, fences, native handles,
and memory budgets, and qualifies device creation by dispatching the embedded
shared AVBD DXIL prepare/solve schedule. It deliberately does not advertise
physics features until the remaining shared domain schedules are wired through
the command API.
The [coverage audit](docs/reference/API-AUDIT.md) maps every declaration to its
runtime definition.

## Build and test

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

CUDA runtime tests skip with code 77 when no CUDA device is available. The
Windows D3D12 command test also skips when no Shader Model 6.6 adapter is
available. Compute capability `86` is the local RTX 3050 Ti setting; consumers
should select the architectures they ship.

CUDA compiles the shared physics modules under `src/slang/`; build target
`parallel-mater-slang-kernel-mirrors` validates every entry. CMake resolves one
exact Slang 2026.18 compiler from `PATH` or downloads the matching
checksum-pinned official package. Set
`PARALLEL_MATER_SLANGC_EXECUTABLE` to an exact compiler path for offline builds.
NVCC compiles Slang's generated CUDA source; no Slang runtime is shipped. See
the [Slang toolchain and source layout](docs/SLANG.md).

### Target and driver support

The Slang/CUDA driver is the first fully qualified physics implementation.
macOS also builds a loader-native Slang CPU driver: its serial host adapter
executes the shared generated rigid-body, constraint, contact, debug, and query
kernels. It is a correctness and development path, not yet a performance
replacement for a native Apple GPU adapter, and it does not advertise fluid,
cloth, rope, soft-body, smoke, or paint capabilities. All shared Slang modules
are also compiled to Vulkan SPIR-V as a portability check, but there is no
Vulkan driver until a complete shared schedule passes the same conformance and
gallery gates. The D3D12 driver currently qualifies the AVBD DXIL schedule and
core transfer API but advertises no physics domains; Metal and WebGPU have no
drivers. The backend-neutral AVBD numerical core is also compiled to WGSL
as a WebGPU portability gate; the complete pointer-based domain schedules do
not yet have WebGPU resource wrappers. Retired backend-specific physics targets
are absent rather than available as fallbacks. Native
API names remain in
interoperability enums because applications may query support and receive
`PM_ERROR_FEATURE_NOT_PRESENT` explicitly.

## Blender gallery and renderer adapters

The optional gallery loads committed Blender-authored `.glb` files and creates
every scene through the public command API. The default renderer is a portable
software rasterizer feeding a thin native window/presentation adapter (D3D12 on
Windows, OpenGL elsewhere). OptiX is an optional renderer adapter and is
deliberately separate from the common catalog, scene, input, and application
targets.

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
  -DPARALLEL_MATER_BUILD_CUDA=ON \
  -DPARALLEL_MATER_BUILD_GALLERY=ON \
  -DPARALLEL_MATER_GALLERY_RENDERER=SOFTWARE \
  -DBUILD_TESTING=ON
cmake --build build-gallery
ctest --test-dir build-gallery --output-on-failure
./build-gallery/parallel-mater-gallery
```

### macOS Gallery application

The default macOS build creates `ParallelMater Gallery.app`, bundles the Slang
CPU driver and portable manifest, and loads physics only through the public
loader. It contains none of the retired Metal physics implementation. Rigid and
rigid-constraint galleries run through shared Slang kernels; unadvertised
simulation domains remain unavailable.

For a stable development Dock item, enable continuous deployment in one build
directory. Each successful default build ad-hoc signs and atomically refreshes
`/Applications/ParallelMater Gallery.app`:

```bash
cmake -S . -B build-macos-gallery \
  -DPARALLEL_MATER_BUILD_GALLERY=ON \
  -DPARALLEL_MATER_BUILD_SLANG_CPU_DRIVER=ON \
  -DPARALLEL_MATER_MACOS_GALLERY_AUTO_DEPLOY=ON
cmake --build build-macos-gallery
```

The installed app searches only its bundled `Resources/drivers.d`; it never
scans the source or build volume. The explicit `deploy-macos-gallery` target
performs the same refresh when automatic deployment is disabled.

For AVBD-budget experiments, use `--contact-passes 1` in the gallery.
The command API field is `PmRigidSolverState::rigidIterations`; 1–64 selects
iterations per pose/impact phase for both contacts and joints. Impacted islands
may therefore report twice that value;
integration substeps are unchanged. Low budgets can destabilize stacks. Check
quality with `parallel-mater-command-gallery-tests`, not only timing.
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
forces, `N` velocities, and `M` writes the rolling physics capture to the
platform temporary directory (`/tmp` on Linux and the per-user temporary
directory on macOS). The interactive gallery shows the resolved path after a
successful capture.
`V` exposes particles, cloth structure, or every soft-body spring where those
systems exist; `B` shows cloth bonds or the soft-body surface wireframe. The
gallery opts into debug-data output when it creates the simulation;
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

The gallery executable has no compile-time CUDA or OptiX dependency. Running a
scene still requires a loader driver advertising all physics features used by
that scene; today that is the Slang/CUDA driver. Select `OPTIX` instead of
`SOFTWARE` to build the optional NVIDIA renderer; only that selection fetches
the pinned OptiX headers. See [the Blender scene contract](docs/BLENDER_SCENES.md)
before authoring or exporting another scene.
