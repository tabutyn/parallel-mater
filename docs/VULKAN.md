# Shared Slang Vulkan driver

`PARALLEL_MATER_BUILD_VULKAN=ON` builds the loadable
`parallel_mater_driver_slang_vulkan` library and its `slang-vulkan.json`
manifest. The public loader and Gallery remain backend independent. CUDA,
Vulkan, and D3D12 are selected by manifest; the desktop Gallery also accepts
`--backend cuda|vulkan|d3d12|cpu` for drivers built beside it.

The initial Vulkan qualification scope is rigid bodies, rigid constraints,
liquids, liquid–rigid coupling, emitters, queries, and asynchronous public
submission. Other particle domains remain unadvertised. The older standalone
Vulkan `World` prototype is not used by this driver.

## Architecture

The driver runs the same host preparation, portable schedule, and Slang
physics entry points as CUDA. Slang emits SPIR-V plus reflection. Generated
transport wrappers reuse the existing typed host argument lists and check
buffer structure sizes and offsets against reflection. Constants are packed
explicitly, including host/Vulkan boolean differences. No physics equations
are generated or maintained in the Vulkan adapter.

Buffers use coherent mapped Vulkan memory in this first implementation.
Shader pointers are translated to `VkDeviceAddress`; nested mesh pointers use
a typed, bounded translated table. Transfers and compute dispatches use an
owned compute queue, barriers, and fences. Public submission uses the shared
worker queue. GPU timestamps use Vulkan timestamp queries.

Vulkan specializes workgroup size to match each scheduler launch. The contact
reduction uses eight workers and the AVBD preparation uses smaller caches to
fit a 32 KiB workgroup memory budget. The AVBD solve uses its existing single
workgroup path because Vulkan has no cooperative grid barrier.

Requirements: Vulkan 1.3, maintenance4, buffer device address, scalar block
layout, 8-bit storage and arithmetic, 64-bit integer and buffer atomics,
timestamp support, coherent host-visible memory, and 256-byte push constants.
The first enumerated supported discrete GPU is preferred; software adapters
are excluded. Native import is not advertised.

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

The Linux picker uses Zenity. The application menu entry is **Parallel Mater
Backend Test**. It launches the built executable without triggering a build.
`PM_GALLERY_BUILD_DIR` can override the script's build directory.
The launcher opts into a Vulkan pipeline cache under that build directory.
Library users may set `PM_VULKAN_CACHE_DIR` to their own cache directory; the
driver does not write a cache unless that variable is set. Cached data is
checked against the adapter's Vulkan cache identity before use.

`desktop-vulkan-preview` in `config/device-profiles.json` supplies conservative
interactive startup limits. Command-line counts override startup defaults.
Headless tests use explicit counts and do not apply interactive profiles.

```sh
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  ctest --test-dir build-desktop-switch -R vulkan --output-on-failure -j1
```

Tests exercise transfers, overlapping-body contact, GPU timestamps,
asynchronous completion, analytic liquid free fall across dispatch boundaries,
bounded Fluid and FluidRigid gallery scenes, and reset.
