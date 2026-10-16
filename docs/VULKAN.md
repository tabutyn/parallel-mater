# Shared Slang Vulkan driver

`PARALLEL_MATER_BUILD_VULKAN=ON` builds the loadable
`parallel_mater_driver_slang_vulkan` library and its `slang-vulkan.json`
manifest. The public loader and Gallery remain backend independent. CUDA,
Vulkan, and D3D12 are selected by manifest; the desktop Gallery also accepts
`--backend cuda|vulkan|d3d12|metal` for discovered drivers.

The initial Vulkan qualification scope is rigid bodies, rigid constraints,
liquids, liquid–rigid coupling, emitters, queries, and asynchronous public
submission. Other particle domains remain unadvertised. The older standalone
Vulkan `World` prototype is not used by this driver.

## Architecture

The driver runs the same host preparation, portable schedule, and Slang
physics entry points as CUDA. Slang emits SPIR-V plus reflection. Generated
transport wrappers reuse the existing typed host argument lists and check
buffer structure sizes and offsets against reflection. Constants are packed
explicitly, including host/Vulkan boolean differences. Shader debug metadata is
emitted separately so older Vulkan tooling can read the executable SPIR-V.
Production SPIR-V uses Slang optimization with precise floating-point mode;
`spirv-val`, when installed, checks every generated module in CTest. Runtime
tests also fail on Vulkan validation errors, even if the API call succeeds.
No physics equations are generated or maintained in the Vulkan adapter.

Buffers use coherent mapped Vulkan memory in this first implementation.
Mesh preparation takes one contiguous copy into cached host memory before
building acceleration data, avoiding repeated CPU reads across device mappings.
Shader pointers are translated to `VkDeviceAddress`; nested mesh pointers use
a typed, bounded translated table. Transfers and compute dispatches use an
owned compute queue, barriers, and fences. Public submission uses the shared
worker queue. GPU timestamps use Vulkan timestamp queries.

Vulkan specializes workgroup size to match each scheduler launch. The contact
reduction uses eight workers and the AVBD preparation uses smaller caches to
fit a 32 KiB workgroup memory budget. The AVBD solve uses its existing single
workgroup path because Vulkan has no cooperative grid barrier.

Requirements: Vulkan 1.3, maintenance4, buffer device address, scalar block
layout, 8-bit buffer/push-constant storage and arithmetic, 64-bit integer and
buffer atomics, timestamp support, coherent host-visible memory, and 256-byte
push constants.
The first enumerated supported discrete GPU is preferred; software adapters
are excluded. Native import is not advertised. Memory-budget reporting currently
uses total device-local heap capacity minus this context's allocations; it does
not yet account for other applications' GPU memory use.

This is a correctness and integration preview. Mapped host preparation,
single-workgroup solving, and synchronization boundaries still need performance
work. First-use pipeline compilation can delay scene loading. Passing desktop
tests does not qualify sustained phone performance or the remaining domains.

## Desktop testing

Configure a Release build named `build-desktop-switch` with CUDA, Vulkan, and
Gallery enabled. Use `cmake --build build-desktop-switch --target
parallel-mater-gallery --parallel 1` on memory-constrained workstations.

```sh
./scripts/Run-DesktopGallery.sh
./scripts/Run-DesktopGallery.sh --backend vulkan
./scripts/Run-DesktopGallery.sh --backend cuda
./scripts/Install-DesktopTest.sh
```

Linux uses the same in-app backend picker as Windows. Use Up/Down, then Enter.
The application menu entry is **Parallel Mater Backend Test**. It launches the
built executable without triggering a build.
`PM_GALLERY_BUILD_DIR` can override the script's build directory.
The launcher opts into a Vulkan pipeline cache under that build directory.
Library users may set `PM_VULKAN_CACHE_DIR` to their own cache directory; the
driver does not write a cache unless that variable is set. Cached data is
checked against the adapter's Vulkan cache identity before use.

`desktop-vulkan-preview` in `config/device-profiles.json` supplies conservative
interactive startup limits. Command-line counts override startup defaults.
Headless tests use explicit counts and do not apply interactive profiles.

Use a recent Vulkan SDK validator. On this laptop the system 1.3.239 layer
crashes while inspecting the liquid shader; the isolated SDK 1.4.341.0 layer
passes. No system driver or validation package was replaced.

```sh
export VULKAN_SDK=/path/to/sdk/x86_64
VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d" \
LD_LIBRARY_PATH="$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ctest --test-dir build-desktop-switch -R vulkan --output-on-failure -j1
```

Tests exercise transfers, rigid-body motion and contact, GPU timestamps,
asynchronous completion, analytic liquid free fall across dispatch boundaries,
bounded Fluid and FluidRigid gallery scenes, and reset.

## Desktop evidence (2026-10-16)

On the RTX 3050 Ti laptop (NVIDIA 590.44.01), the combined CUDA/Vulkan build
passes six SPIR-V module checks, CUDA command-API checks, Vulkan command/API
and asynchronous-submission checks, the bounded liquid/coupling/reset test,
and shared input/workload checks. Vulkan validation used SDK 1.4.341.0.
The architecture check confirms no native physics kernels or direct launches
were added to the shared scheduler.

All ten advertised preview scenes also rendered ten headless frames with
validation enabled: rigid body, Fixed/Collector, Point, Hinge/Slider, Piston,
Generic, Motor/Spring, Dump, Fluid, and FluidRigid. These used four bricks,
128 fluid particles, and ten dump spheres where applicable. This short sweep
is a startup/rendering check, not long-run stability qualification.

Interactive checks selected CUDA and Vulkan through the same picker and
verified Vulkan rigid rendering, liquid–rigid scene switching, reset, and
unavailable-scene labels. The desktop profile uses four moving liquid-coupled
bodies and 256 particles. Both physics drivers are separate loadable libraries;
the Vulkan library has no CUDA runtime dependency.

Android packaging and GPU runtime qualification remain the next step. The
installed Android application still uses the previously tested CPU preview.
