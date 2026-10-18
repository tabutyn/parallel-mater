# Shared Slang Vulkan driver

`PARALLEL_MATER_BUILD_VULKAN=ON` builds the loadable
`parallel_mater_driver_slang_vulkan` library and its `slang-vulkan.json`
manifest. The public loader and Gallery remain backend independent. CUDA,
Vulkan, and D3D12 are selected by manifest; the desktop Gallery also accepts
`--backend cuda|vulkan|d3d12|metal` for discovered drivers.

The Vulkan qualification scope is rigid bodies, rigid constraints, liquids,
cloth, ropes, soft bodies, smoke, all existing cross-domain Gallery couplings,
paint, emitters, queries, and asynchronous public submission. Exact interaction
capability records cover liquid–rigid, liquid–cloth, liquid–rope,
liquid–soft-body, liquid–smoke, rigid–deformable/smoke, cloth–soft-body,
cloth–rope, soft-body–rope, cloth–smoke, soft-body–smoke and rope–smoke. The
older standalone Vulkan `World` prototype is not used by this driver.

## Architecture

The driver runs the same host preparation, portable schedule, and Slang
physics entry points as CUDA. Slang emits SPIR-V plus reflection. Generated
transport wrappers reuse the existing typed host argument lists and check
buffer structure sizes and offsets against reflection. Constants are packed
explicitly, including host/Vulkan boolean differences. Modules whose reflected
entry block exceeds 256 bytes use a bounded dynamic uniform-buffer path;
smaller modules retain their qualified push-constant pipeline layout. Large
rope records use a bounded mapped staging arena and enter the shader by device
address, keeping the actual push-constant block at 88 bytes. Nested buffer
fields are translated to `VkDeviceAddress` before staging. Shader debug metadata
is emitted separately so older Vulkan tooling can read the executable SPIR-V.
Production SPIR-V uses Slang optimization with precise floating-point mode;
`spirv-val`, when installed, checks every generated module in CTest. Runtime
tests also fail on Vulkan validation errors, even if the API call succeeds.
No physics equations are generated or maintained in the Vulkan adapter.

Buffers use coherent mapped Vulkan memory in this first implementation.
Mesh preparation takes one contiguous copy into cached host memory before
building acceleration data, avoiding repeated CPU reads across device mappings.
Shader pointers are translated to `VkDeviceAddress`; nested mesh and paint-field
pointers use typed, bounded translated tables. Paint-referenced render meshes
also contribute to the translated mesh-table bound, even when no collision body
references them. Transfers and compute dispatches use an owned compute queue,
barriers, and fences. Public submission uses the shared worker queue. GPU
timestamps use Vulkan timestamp queries.

Vulkan specializes workgroup size to match each scheduler launch. The contact
reduction uses eight workers and the AVBD preparation uses smaller caches to
fit a 32 KiB workgroup memory budget. The AVBD solve uses its existing single
workgroup path because Vulkan has no cooperative grid barrier.

Requirements: Vulkan 1.3, maintenance4, buffer device address, scalar block
layout, 8-bit uniform/storage-buffer and push-constant storage and arithmetic,
64-bit integer and buffer atomics, timestamp support, coherent host-visible
memory, and 256-byte push constants.
The first enumerated supported discrete GPU is preferred; software adapters
are excluded. Native import is not advertised. Memory-budget reporting currently
uses total device-local heap capacity minus this context's allocations; it does
not yet account for other applications' GPU memory use.

This is a correctness and integration preview. Mapped host preparation,
single-workgroup solving, and synchronization boundaries still need performance
work. First-use pipeline compilation can delay scene loading. Passing desktop
tests does not qualify sustained phone performance.

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

### Native Gallery rendering

The desktop Gallery compiles `examples/gallery/render/gallery.slang` to SPIR-V
and renders the public physics output buffer directly on the Vulkan device. The
renderer obtains the borrowed instance, physical device, logical device, queue,
and output buffer only through Parallel Mater's public native-interoperability
records. Ordinary frames perform no physics-state or image readback. Explicit
capture and qualification requests render offscreen and copy only the final
RGBA image.

Desktop GLFW and Android `ANativeWindow` shells own their `VkSurfaceKHR`; the
common renderer owns its swapchain. Target
detach releases the swapchain before the shell destroys or replaces the
surface, while the physics driver keeps the borrowed Vulkan instance alive.
The renderer handles resize, out-of-date/suboptimal swapchains and reported
surface loss. Android presents through this same renderer without ordinary-frame
image readback.

Use a recent Vulkan SDK validator. On this laptop the system 1.3.239 layer
crashes in `vkCreateShaderModule` while inspecting the liquid shader; the
isolated SDK 1.4.341.0 layer passes the same standalone probe and unchanged
lifecycle case. No system driver or validation package was replaced.

The Linux validation scripts pin SDK 1.4.341.0 and its archive SHA-256 in
`config/vulkan-validation-sdk.json`. Setup extracts only the validation layer,
its manifest, and `spirv-val` into ignored `out/toolchains/`, verifies installed
file hashes on reuse, and leaves system packages untouched. The test wrapper
sets both the layer search path and library search path to that installation.
It validates all 23 production SPIR-V modules before running CTest serially.

```sh
./scripts/Setup-VulkanValidation.sh
./scripts/Test-Vulkan.sh
./scripts/Probe-VulkanValidationLayer.sh --pinned
```

`./scripts/Probe-VulkanValidationLayer.sh --system` selects only the Debian
layer and reproduces exit 139 on affected installations. The probe compiles no
physics adapter and dispatches no work; it isolates instance/device setup,
descriptor and pipeline layouts, shader-module inspection and pipeline
compilation. Its printed hashes identify the layer, shader and probe binaries.

`PM_VULKAN_SDK_ARCHIVE` can supply an already downloaded archive; the same
pinned checksum is required. `PM_GALLERY_BUILD_DIR` selects an existing build.
The scripts do not start a build. Use `--parallel 1` for builds on this laptop.


Tests exercise transfers, rigid-body motion and contact, GPU timestamps,
asynchronous completion, analytic liquid free fall across dispatch boundaries,
bounded Fluid and FluidRigid scenes, all four cloth Gallery scenes, the soft
body, soft-body/rigid and soft-body/fluid scenes, Rope, RopeFluid, Smoke,
SmokeWater, all six cross-deformable Gallery scenes, paint, and reset. Native
Gallery tests additionally cover device/queue identity, direct
output, frames in flight, oriented diagnostic readback, resize, surface detach and
replacement,
offscreen fallback, and validation-clean teardown.

## Native Gallery evidence (2026-10-18)

On the NVIDIA GeForce RTX 3050 Ti Laptop GPU, the validation-enabled renderer
contract passed direct rendering without ordinary-frame readback, repeated
frames, offscreen image checks, resize, and teardown. A physical GLFW test
passed surface attach, detach, offscreen fallback, replacement, resize, and
teardown. A six-frame interactive Point-constraint smoke presented through the
Vulkan swapchain and captured an upright 1280×720 image without a validation
message.

## Desktop cloth evidence (2026-10-17)

On the RTX 3050 Ti laptop, SDK 1.4.341.0 validation accepts all nine production
SPIR-V modules. The common-session rendered-image contract and 120-frame public
command lifecycle tests pass for Cloth, Cloth Tear, Cloth Paint, and Water
Cloth. Headless Gallery checks also report 2,288 painted texels for Cloth Paint
and 777 painted texels for Peg Paint. The latter exposed and now guards the
separate render-mesh bound required by paint fields.

This desktop evidence enables exact `cloth`, `domainCouplings`, and `paint`
capability advertising in the loadable Vulkan driver. Android uses the same
modules and capability query. The revision-pinned Pixel catalog sweep passes
the four cloth scenes; sustained phone qualification remains F7-18.

## Desktop soft-body evidence (2026-10-18)

On the RTX 3050 Ti laptop, the shared Slang soft-body, geometry–soft-body and
fluid–soft-body modules compile to normalized SPIR-V and pass validation. The
common-session rendered-image contract passes Soft Body, Soft Body Rigid and
Soft Body Fluid. Separate 120-frame public-command lifecycle checks pass all
three scenes, including reset and no-host-output recovery.

Vulkan advertises `softBodies`, fluid–soft-body, soft-body–rigid,
cloth–soft-body and rope–soft-body interactions. The Pixel catalog sweep at the
F7-13 revision passed the three then-available soft-body scenes.

## Desktop rope evidence (2026-10-18)

On the RTX 3050 Ti laptop, the shared Slang rope and fluid–rope modules compile
to normalized SPIR-V and pass SDK 1.4.341.0 validation. Separate 120-frame
public-command lifecycle checks pass Rope and RopeFluid, including finite output
and reset. The common Gallery session contract advertises, selects, qualifies,
and renders both scenes through the same backend-independent session used by
desktop and mobile shells.

Vulkan advertises `rope`, rope self-contact, fluid–rope, rope–rigid,
cloth–rope, soft-body–rope and rope–smoke interactions. The Vulkan transport
stages reflected rope records because the complete 612-byte by-value entry block
cannot fit the required 256-byte push-constant baseline. Android packages the
same modules; the Pixel catalog sweep at the F7-13 revision passed Rope and
RopeFluid.

## Desktop and Android smoke evidence (2026-10-18)

The shared Slang geometry-smoke, smoke-grid and fluid-smoke modules compile to
normalized SPIR-V. All 23 production modules pass SDK 1.4.341.0 static
validation. Separate 120-frame public-command lifecycle checks pass Smoke and
SmokeWater, including finite visible output and reset. The common Gallery
session qualifies both scenes through the backend-independent runtime.

Vulkan advertises smoke self pressure/viscosity, smoke–rigid contact/drag,
fluid–smoke drag/thermal/phase interactions and cloth/soft-body/rope–smoke
contact and drag for the particle and grid profiles.

A revision-pinned F7-13 Pixel 7 run passed Smoke and SmokeWater within the full
28-scene catalog sweep. That run reported 22 supported scenes, direct Vulkan
presentation without ordinary-frame readback, background recovery, resize,
rotation and native teardown. This short physical run establishes correctness
and visible output; sustained thermal/performance qualification remains F7-18.

## Desktop cross-domain evidence (2026-10-18)

The shared Slang soft-body–cloth, cloth–rope, soft-body–rope, cloth–smoke,
soft-body–smoke and rope–smoke modules compile to normalized SPIR-V and expose
their exact interaction records. Separate 120-frame public-command tests pass
all six Gallery scenes. The complete 17-scene Vulkan domain sweep and the common
Gallery session contract also pass on the RTX 3050 Ti.

The rope–smoke wind entry passes only the three smoke fields used by the common
Slang wind function. Its reflected block is 188 bytes and stays on the portable
push-constant path; passing the full 128-byte smoke options record made the
module use the dynamic parameter-buffer path and stalled this workload on the
desktop driver. CUDA compilation and Android ARM64 packaging pass with the same
entry ABI. The revision-pinned Android APK contains all 23 modules; a fresh
physical Pixel sweep is still required before claiming those six scenes on that
device.

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
The earlier eight-scene mobile subset completes short runtime checks;
performance remains below target. See [the qualification record](ANDROID-QUALIFICATION.md) and
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
IDs and executable instructions are preserved. Oversized reflected entry blocks
are additionally rebound from push constants to a scalar-layout uniform buffer;
smaller modules remain byte-identical after pointer ordering. Recursive types or
unfamiliar dependencies fail the build with a diagnostic. CMake keeps
the compiler output as `generated/vulkan/*.raw.spv` and embeds the normalized
`*.spv`; validate the latter. All 23 modules pass SDK 1.4.341.0 validation on
Linux; the previous Pixel run covered the original six-module subset.

`tools/vulkan_pipeline_probe.cpp` isolates shader inspection and pipeline
compilation without the
Parallel Mater loader, scheduler or physics dispatch. Build with a host or
Android NDK C++ compiler using `-std=c++20 -O2 -lvulkan` (add
`-static-libstdc++` for a standalone Android executable), then run:

```sh
./probe geometry.spv pm_capture_rigid_inputs 128
```

It requests the same device features, bindings and specialization constant as
the Vulkan transport. `PIPELINE_OK` proves compilation only, not simulation
correctness or phone qualification.
