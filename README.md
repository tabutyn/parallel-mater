# ParallelMater

ParallelMater is an MIT-licensed physics library with a C API for running
simulations on GPUs. It covers rigid bodies and joints, particle fluids, cloth,
ropes, soft bodies, smoke, and interactions between these systems. Applications
control devices, memory, simulation resources, command submission, and
synchronization through one API.

The repository includes the loader and SDK, reference drivers for CUDA, Metal,
Direct3D 12, and Vulkan, and the Gallery example application for desktop,
Android, and iPad. The reference drivers compile shared Slang physics shaders
for their target GPU; Metal physics executes on the Apple GPU.

- **Try the examples:** [build and run Gallery](#try-the-gallery).
- **Integrate physics into an application:** [use the C API](#use-the-c-api).
- **Implement a driver:** [driver development](#write-a-driver).

The project is in development on API version **0.3**. Before 1.0, minor API
versions may introduce incompatible changes. Drivers have different feature
coverage, and complete physical and performance qualification is still open.
Use the [qualification index](docs/QUALIFICATION.md) to check evidence for the
specific driver, device, revision, and interaction you intend to use.

## How it works

Applications link the `parallel-mater` loader and include
`<parallel_mater/parallel_mater.h>`. The loader discovers installed drivers,
enumerates their physical devices, and routes calls to the driver that owns the
selected device. Optional layers provide validation and diagnostics.

| Component | Responsibility |
|---|---|
| Application or Gallery | Choose a device, create the simulation, submit steps, consume output, and render or otherwise use the results. |
| Loader | Discover drivers and layers, negotiate interfaces, enumerate devices, and dispatch API calls. |
| Reference runtime and Slang shaders | Share simulation scheduling, resource planning, collision detection, solver equations, and coupling operators across GPU backends. |
| Native device adapter | Allocate GPU memory, transfer and bind resources, dispatch shaders, synchronize work, and provide supported native interoperability. |

An **installable driver** implements the public API behind the loader. A
**device adapter** is the native GPU service layer inside a reference driver.
The shared C++ runtime schedules work on the host; the GPU drivers execute
physics kernels on the GPU. A separately selected Slang CPU driver exists for
development and reference checks.

Rigid contacts and joints use a shared Augmented Vertex Block Descent (AVBD)
solver. Other domains and coupling operators also live in shared Slang. See
[physics source and compilation](docs/SLANG.md) and the
[rigid solver](docs/CONTACT_SOLVER.md) for implementation details.

## Drivers and feature coverage

These are implemented reference drivers and their advertised domain scope,
not a claim that every interaction or device has passed qualification:

| Driver | Platform / hardware | Physics execution | Domain scope |
|---|---|---|---|
| CUDA | Linux and Windows with a compatible NVIDIA GPU | Slang-generated CUDA / PTX on the GPU | Rigid bodies, joints, fluids, cloth, ropes, soft bodies, smoke |
| Metal | macOS and iPadOS with Apple GPU family 7 or newer | Slang-generated Metal libraries on the GPU | Rigid bodies, joints, fluids, cloth, ropes, soft bodies, smoke |
| D3D12 | Windows with a Shader Model 6.6-capable GPU | Slang-generated DXIL on the GPU | Rigid bodies, joints, fluids, cloth |
| Vulkan | Linux and Android with the [required Vulkan features](docs/VULKAN.md) | Slang-generated SPIR-V on the GPU | Rigid bodies, joints, fluids, cloth, ropes, soft bodies, smoke |
| Slang CPU | Linux and macOS development hosts | Slang-generated C++ executed serially | Rigid bodies and joints; experimental liquids require an explicit build option |

There is no WebGPU runtime driver; WGSL compilation checks cover the AVBD
numerical core only. CUDA is one backend choice, not a universal default.
Platform presets select which drivers are built, and applications select a
physical device from the installed drivers.

Domain support does not imply every interaction between domains. Query
`pmGetPhysicalDeviceFeatures` and
`pmEnumeratePhysicalDeviceInteractionCapabilities` before creating resources.
Interaction records identify the exact pair, scope, mode, direction, and
material profile. For example, equal-material separate-fluid interaction is
advertised by the CUDA build; this does not imply support for miscible,
immiscible, or phase-transfer fluid pairs on every driver. See the
[interaction contract](docs/INTERACTION-MODES.md) for those distinctions.

An unavailable capability returns `PM_ERROR_FEATURE_NOT_PRESENT`; selecting a
GPU driver does not silently switch physics to the CPU. Runtime capability
queries describe what the driver exposes. The revisioned
[qualification results](docs/QUALIFICATION.md#per-driver-evidence-index)
separately report whether its required execution, lifecycle, regression, and
analytic physical checks have passed. A working Gallery scene or successful
shader compilation alone does not establish that qualification.

## Use the C API

The API uses explicit resources and command buffers, similar to modern graphics
APIs. A typical application:

1. Creates an instance, enumerates physical devices, and queries capabilities.
2. Creates a device and queue, queries memory requirements, and allocates and
   binds memory for geometry, buffers, and simulation resources.
3. Creates a physics pipeline containing solver and material policy, then
   records initial state and supported interactions into command buffers.
4. Records simulation steps and output commands, submits them to the queue,
   and uses fences or timeline semaphores to observe completion.
5. Uses completed output for rendering or application logic, keeping resources
   alive until their submitted work finishes.

The library does not require a window or renderer. Native buffer sharing is
capability-gated; check the selected driver's interoperability support before
designing an engine integration around it.

Start with the [API guide](docs/API.md) and the small
[C device-discovery example](examples/sdk/devices.c). The
[API specification](docs/reference/parallel-mater-specification.md) defines
behavior and ownership; the [API reference PDF](docs/reference/parallel-mater-reference.pdf)
lists declarations. The [specification PDF](docs/reference/parallel-mater-specification.pdf)
provides the same behavioral rules in printable form.

### Build and install the SDK

Requirements: CMake 3.25+, Ninja, Python 3, and a C++20 compiler. The C example
also needs a C11 compiler. On Windows, use an x64 Visual Studio Developer
Command Prompt with Ninja on `PATH`.

Run from the repository root:

```sh
cmake --preset sdk-minimal
cmake --build --preset sdk-minimal
cmake --install build-sdk-minimal --prefix "$PWD/out/sdk"

cmake -S examples/sdk -B build-sdk-example -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/out/sdk"
cmake --build build-sdk-example
```

These are POSIX-shell commands; in a Windows Developer Command Prompt, replace
`$PWD/out/sdk` with an absolute installation path and put each command on one
line. This profile installs the loader, public headers, and validation layer.
It needs no Slang compiler or GPU SDK and includes **no physics driver**.
Build or install a driver separately before running a simulation.

Reference builds generate manifests under `<build>/<configuration>/manifests`;
for example, the Metal preset below produces
`build-macos-metal-preview/Release/manifests/slang-metal.json`. Installed driver
packages place manifests under `share/parallel-mater/drivers.d`.
Point the example at the chosen manifest (replace the path):

```sh
PM_DRIVER_FILES=/absolute/path/to/slang-metal.json ./build-sdk-example/pm-devices
```

On Windows, set `PM_DRIVER_FILES` with `set` or `$env:PM_DRIVER_FILES`, then run
`build-sdk-example\pm-devices.exe`. `PM_DRIVER_FILES` lists manifest files;
`PM_DRIVER_PATH` lists directories to search. Lists use `:` on Unix and `;` on
Windows. Manifest library paths may be relative to their manifest directory.
See [driver discovery](docs/DRIVER_SDK.md#driver-manifest) for packaging details.

In a CMake consumer:

```cmake
find_package(ParallelMater 0.3 CONFIG REQUIRED COMPONENTS loader)
target_link_libraries(your_application PRIVATE ParallelMater::api)
```

For portable SDK tests, use `cmake --preset sdk-ci`,
`cmake --build --preset sdk-ci`, and `ctest --preset sdk-ci`.
[SDK builds and releases](docs/SDK-RELEASES.md) covers optional components,
installed examples, compatibility, and release verification.

## Try the Gallery

Gallery loads Blender-authored `.glb` scenes through the public API. Desktop,
Android, and iPad share the catalog, scene/session logic, controls, and frame
timer. The picker reports which scenes the selected driver can run. Selecting
a physics backend and choosing a rendering method are separate concerns;
native Metal, D3D12, and Vulkan paths can draw public GPU output buffers directly.
Software rendering and optional NVIDIA OptiX rendering are also available.

GPU builds use the repository-pinned Slang compiler. CMake finds the exact
version on `PATH` or downloads its checksum-pinned package. For offline builds,
set `PARALLEL_MATER_SLANGC_EXECUTABLE` to that compiler. See
[toolchain requirements](docs/SLANG.md#compiler-toolchain).

### macOS: Apple GPU

Install Xcode with the Metal toolchain and `dockutil` (`brew install dockutil`),
then use the explicit Metal preset:

```sh
cmake --preset macos-metal-preview
cmake --build --preset macos-metal-preview
```

The successful build atomically replaces the canonical
`/Applications/ParallelMater Gallery.app`, verifies its signature and exact
build identity, and replaces the Dock item with that installed path. Launch the
Gallery from the Dock or `/Applications`; do not launch a stale build-tree copy.
This app bundles the Metal physics driver. A plain macOS configuration without
this preset defaults to the CPU development driver.

CI, packaging, or a deliberately non-installing build can opt out explicitly:

```sh
cmake --preset macos-metal-preview -DPARALLEL_MATER_MACOS_GALLERY_AUTO_DEPLOY=OFF
```

For an additional native rendering smoke test and launch verification, run
`python3 scripts/Deploy-Gallery.py macos`. The wrapper requires a clean checkout
and matching build by default; `--allow-dirty` permits an explicit development
iteration.

### Windows: D3D12

Use Visual Studio 2022 with its C++ tools and Windows SDK, plus the pinned Slang
compiler:

```powershell
cmake --preset windows-d3d12-preview
cmake --build --preset windows-d3d12-preview
.\build-windows-d3d12-preview\Release\Parallel-Mater.exe --backend d3d12
```

The preview preset leaves the installed application unchanged. To refresh the
canonical Start Menu app, run
`cmake --build --preset windows-d3d12-preview --target install-windows-gallery`.
The [Gallery guide](docs/GALLERY.md#backend-selection) explains installation
and backend selection.

### Linux: Vulkan or CUDA

For Vulkan, install a Vulkan SDK, a compatible GPU driver, Ninja, and desktop
window-system development libraries. Follow the exact hardware and validation
requirements in the [Vulkan guide](docs/VULKAN.md).

```sh
cmake --preset linux-vulkan-preview
cmake --build --preset linux-vulkan-preview
./build-linux-vulkan-preview/parallel-mater-gallery --backend vulkan
```

For NVIDIA CUDA, install the CUDA toolkit and use a compatible host compiler:

```sh
cmake -S . -B build-gallery -DCMAKE_BUILD_TYPE=Release \
  -DPARALLEL_MATER_BUILD_CUDA=ON -DPARALLEL_MATER_BUILD_GALLERY=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-gallery --parallel 2
./build-gallery/parallel-mater-gallery --backend cuda
```

`native` targets the build machine's GPU. Set explicit CUDA architectures when
building for other machines. The interactive backend picker offers compatible
installed drivers; `--backend` selects one explicitly for scripted runs.

### Android and iPad

- [Android build, installation, and controls](examples/android/README.md):
  Vulkan GPU physics and rendering; the default ARM64 build requires Android
  13+ and the specified Vulkan features. The guide lists the pinned SDK/NDK
  dependencies and an explicit CPU development alternative.
- [iPad build, installation, and controls](examples/ios/README.md): Metal GPU
  physics and rendering; requires iPadOS 26+, Apple GPU family 7+, Xcode
  development signing, and a paired device with Developer Mode enabled.

Both mobile pickers use the shared 28-scene catalog and runtime capability
checks. Device profiles set conservative starting workloads; these are not
measured sustained-performance guarantees.

### Controls, capture, and timing

On desktop, `Tab` opens the scene picker, `R` resets, left-drag orbits,
Shift+left-drag pans, and the wheel zooms. Arrow keys and `Space` perform the
actions shown for the current scene. `F` toggles the shared frame-time/FPS
panel. `M` saves a rolling physics capture and reports the output path.

For a headless image and per-frame timing CSV, using the Linux CUDA build above:

```sh
./build-gallery/parallel-mater-gallery --backend cuda --constraint-point \
  --headless point.ppm --frames 120 --frame-timings point.csv
```

Use the corresponding executable and backend for other desktop builds. With
timing enabled, headless runs simulate and render every frame. Without it,
they simulate the requested steps and render the final image once. Measurements
are host wall-clock costs; interactive presentation includes VSync wait. See
[Gallery timing](docs/GALLERY.md#frame-timing) for the measurement boundaries.
Run the executable with `--help` for scene and diagnostic options, or
`--version` to check its embedded build identity.

See the [Gallery guide](docs/GALLERY.md) for scene-specific controls and examples.
To author scenes, use the common
[Blender exporter and scene contract](docs/BLENDER_SCENES.md).

## Write a driver

Start with the [driver SDK](docs/DRIVER_SDK.md), the installed
`parallel_mater/pm_driver.h`, and the API specification. A driver supplies a
shared library and JSON manifest, negotiates the loader/driver interface, and
exports the four `pm_driver*` entry points. Applications continue to use the
loader's public `pm*` API.

Choose the integration level that fits your implementation:

- **Add a backend to the reference implementation:** reuse the Slang physics
  modules and shared runtime; implement the native memory, dispatch,
  synchronization, query, and interoperability services. See
  [shared Slang](docs/SLANG.md) and the
  [common resource boundary](docs/COMMON-RESOURCE-BOUNDARY.md).
- **Implement an independent driver:** implement the versioned driver ABI and
  public API behavior. The loader does not require Slang or a particular native
  GPU API. A hardware vendor can use its own compiler and execution services.

Report only capabilities your driver implements, honor resource ownership and
asynchronous completion rules, and return defined errors for unsupported
requests. Validate ABI and lifecycle behavior with the SDK tests, then run the
independent [conformance suite](conformance/v1/README.md). Physical quality and
performance claims need evidence for the exact hardware and build being
distributed; see [qualification](docs/QUALIFICATION.md) and
[release requirements](docs/SDK-RELEASES.md#required-release-evidence).

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules and
[the roadmap](docs/ROADMAP.md) for remaining work and task ownership. The
code and specification are [MIT licensed](LICENSE); dependency notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
