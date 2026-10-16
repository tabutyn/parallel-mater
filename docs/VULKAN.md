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
./scripts/Install-DesktopGallery.sh
```

Linux uses the same in-app backend picker as Windows. Use Up/Down, then Enter.
The application menu entry is **Parallel Mater**. It launches the
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

The Linux validation scripts pin SDK 1.4.341.0 and its archive SHA-256 in
`config/vulkan-validation-sdk.json`. Setup extracts only the validation layer,
its manifest, and `spirv-val` into ignored `out/toolchains/`, verifies installed
file hashes on reuse, and leaves system packages untouched. The test wrapper
sets both the layer search path and library search path to that installation.
It validates all six production SPIR-V modules before running CTest serially.

```sh
./scripts/Setup-VulkanValidation.sh
./scripts/Test-Vulkan.sh
```

`PM_VULKAN_SDK_ARCHIVE` can supply an already downloaded archive; the same
pinned checksum is required. `PM_GALLERY_BUILD_DIR` selects an existing build.
The scripts do not start a build. Use `--parallel 1` for builds on this laptop.


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

Android Vulkan packaging builds, including native public-API qualification
executables. Pixel 7 now passes command/API, asynchronous submission and bounded
liquid/coupling/reset tests after the pointer declaration workaround below.
Gallery scene/lifecycle and sustained performance qualification remain open. See
[Android status and reproduction](../examples/android/README.md#vulkan-status-2026-10-16).

For opt-in diagnostics, `PM_VULKAN_TRACE=1` prints each newly created pipeline's
module, entry point, workgroup size and SPIR-V size. `PM_VULKAN_VALIDATION=1`
explicitly requests `VK_LAYER_KHRONOS_validation`; it must already be available
to the platform loader. A missing requested layer fails instance creation.

### Mali pointer declaration workaround

Slang 2026.18 emits forward pointer declarations for acyclic structures. On the
Pixel 7 driver above, even a 1496-byte `pm_capture_rigid_inputs` shader crashes
in `spir2lir::SPIR_MBS2Helper::apply_decorations`. Defining its pointer types
before use removes the crash; the full geometry module also passes pipeline
creation after the same transformation.

`tools/normalize_vulkan_spirv.py` now orders type/constant dependencies before
use and removes those redundant declarations before embedding each module.
IDs, decorations, layouts and executable instructions are preserved. Recursive
types or unfamiliar dependencies fail the build with a diagnostic. CMake keeps
the compiler output as `generated/vulkan/*.raw.spv` and embeds the normalized
`*.spv`; validate the latter. All six modules pass SDK 1.4.341.0 validation.

`tools/vulkan_pipeline_probe.cpp` isolates pipeline compilation without the
Parallel Mater loader, scheduler or physics dispatch. Build with a host or
Android NDK C++ compiler using `-std=c++20 -O2 -lvulkan` (add
`-static-libstdc++` for a standalone Android executable), then run:

```sh
./probe geometry.spv pm_capture_rigid_inputs 128
```

It requests the same device features, bindings and specialization constant as
the Vulkan transport. `PIPELINE_OK` proves compilation only, not simulation
correctness or phone qualification.
