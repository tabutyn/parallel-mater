# Vulkan backend status

The Vulkan backend is an opt-in Vulkan 1.2 compute backend for Linux and
Android. Its first vertical slice implements rigid-body resource lifecycle and
direct integration. It is not a conformance-v1 backend yet because even the
smallest rigid conformance case includes contacts.

## Implemented

- Headless owned instance/device/compute-queue creation.
- Borrowed `VkInstance`, `VkPhysicalDevice`, `VkDevice`, and compute queue.
- Timeline-semaphore `FrameToken` completion, including a native semaphore and
  value suitable for GPU-side renderer waits.
- Host triangle-mesh registration for mass-property bounds and stable,
  generation-checked rigid-body handles.
- Static, kinematic, and dynamic integration; substeps; previous state;
  damping; force, acceleration, and impulse consumption; angular response; and
  linear/angular speed clamps.
- Stable fixed-capacity device buffers, synchronous state readback, statistics,
  and optional timestamp timings.
- Direct Slang-to-SPIR-V 1.5 compilation, Vulkan 1.2 validation, and generated
  static-library embedding. No shader compiler or shader files ship at runtime.

Resource additions, removals, and state edits require an idle `World`. One
frame may be in flight. Reacquire `RigidBodyDeviceView` after completion to
observe current counts and revision. Cross-queue-family ownership transfer is
not implemented.

## Unsupported in this milestone

Rigid contacts, constraints, sleeping, contact collection, physics-debug
capture, fluids, cloth, soft bodies, ropes, smoke, pairwise couplings, gallery
integration, and product integration are intentionally absent. Unrelated
capacity fields in `WorldOptions` are inert. Requests exposed by shared option
types, such as contact collection or rigid sleeping, return `not_supported`.

## Build on Linux

Vulkan mode requires Vulkan 1.2 headers/loader, C++20, host Slang 2026.19 or
newer, and host SPIRV-Tools 2026 or newer. Older `spirv-val` releases do not
recognize the standardized Slang `OpSource` enumerant.

The repository bootstrap script pins and verifies the official Slang 2026.19
Linux archive:

```bash
chmod +x tools/bootstrap-slang.sh
tools/bootstrap-slang.sh /tmp/parallel-mater-slang

cmake -S . -B build-vulkan \
  -DPARALLEL_MATER_BUILD_CUDA=OFF \
  -DPARALLEL_MATER_BUILD_METAL=OFF \
  -DPARALLEL_MATER_BUILD_VULKAN=ON \
  -DPARALLEL_MATER_SLANGC_EXECUTABLE=/tmp/parallel-mater-slang/bin/slangc \
  -DPARALLEL_MATER_SPIRV_VAL_EXECUTABLE=/path/to/spirv-val \
  -DPARALLEL_MATER_SPIRV_OPT_EXECUTABLE=/path/to/spirv-opt \
  -DBUILD_TESTING=ON
cmake --build build-vulkan
ctest --test-dir build-vulkan --output-on-failure
```

The installed target is `ParallelMater::vulkan`; include
`<parallel_mater/vulkan.hpp>`. Package configuration conditionally resolves
`Vulkan::Vulkan`.

## Borrowed-context rules

Every `NativeContext` handle remains caller-owned. The caller must keep all
handles alive until the `World` and its submitted `FrameToken` objects are
destroyed, declare that `timelineSemaphore` was enabled, and externally
serialize queue access while a backend call submits work. All handles must
refer to the declared physical/logical device and queue family. Vulkan does not
provide a query that can prove those handle relationships after device
creation.

Destroying a borrowed `World` waits only for work submitted by that backend.
It does not call `vkDeviceWaitIdle` and does not destroy caller handles. Device
loss permanently faults the `World` when observed.

## Android smoke test

`tests/android-smoke` is a standalone API-26/target-36 instrumentation app for
`arm64-v8a`. Its manifest requires Vulkan compute level 0 and Vulkan 1.2. JNI
repeats version, timeline, compute, descriptor, and workgroup checks; creates an
app-owned device/queue; passes a borrowed context; runs an asynchronous rigid
fixture; submits a GPU-side timeline wait; validates readback; and proves the
backend left caller handles alive.

Using Gradle 9.1 and an Android SDK containing platform 36, NDK
28.2.13676358, and CMake 4.1.2:

```bash
gradle -p tests/android-smoke assembleDebugAndroidTest \
  -PparallelMaterSlangc=/absolute/path/to/slangc \
  -PparallelMaterSpirvVal=/absolute/path/to/spirv-val \
  -PparallelMaterSpirvOpt=/absolute/path/to/spirv-opt

gradle -p tests/android-smoke connectedDebugAndroidTest \
  -PparallelMaterSlangc=/absolute/path/to/slangc \
  -PparallelMaterSpirvVal=/absolute/path/to/spirv-val \
  -PparallelMaterSpirvOpt=/absolute/path/to/spirv-opt
```

The assembly task runs host shader tools only; it never executes an Android
binary. Milestone closure still requires `connectedDebugAndroidTest` on at
least one physical Vulkan 1.2 device and the opt-in real-GPU Linux validation
job.
