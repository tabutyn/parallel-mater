# Shared Slang physics

Slang is the source language for GPU physics. File names describe physics
domains rather than native APIs or solver experiments.

- `types.slang` contains shared resource and value layouts.
- `math.slang` contains numerical helpers.
- `device_intrinsics.slang` contains portable dispatch, barrier, and atomic
  operations.
- `geometry.slang` contains rigid triangle/BVH contact generation.
- `geometry_constraints.slang` contains AVBD contact and joint solving.
- `fluid.slang`, `cloth.slang`, `rope.slang`, `soft_body.slang`, and the smoke
  modules contain their domain kernels.
- Domain-pair files contain coupling kernels.
- `queries.slang` contains scene-query kernels.

The command-API CUDA driver dispatches generated Slang kernels. The Linux/macOS Slang
CPU driver compiles the same rigid-body, constraint, contact, debug, and query
kernels to C++ and dispatches them serially as a correctness-first development
path. Their device and simulation adapters allocate, transfer, synchronize,
and dispatch; they do not contain alternate physics equations.

## Reference scheduling and submission

[`reference_schedule.hpp`](../src/simulation/reference_schedule.hpp) contains
the shared domain and coupling order as ordinary C++. Its state parameter owns
resource descriptions and bookkeeping; its backend parameter supplies typed
Slang dispatch, copy/clear, timing, readback, error checks, and queue waits.
The header compiles without a native SDK. The private
[`native_schedule_backend.cu.inc`](../src/simulation/native_schedule_backend.cu.inc)
binds those operations to the existing CUDA and serial Slang CPU adapters.
Dispatches on the backend queue must preserve write/read ordering and insert
the native barriers required by that backend.

The command API copies each accepted submission into a queue-owned worker.
The worker activates the device, runs ready submissions, retains frame tokens
through completion, drains partially issued work on failure, and publishes a
completion shared by fences, command buffers, and timeline signals. Unresolved
timeline waits leave the worker free to run their producers. Ready submissions
remain FIFO; explicit dependencies must order access to shared resources.

This removes physics execution and readbacks from the submitting thread.
The reference implementation still waits on its worker between frames and
performs explicit readbacks for emission, phase transfer, diagnostics, and
some resource metadata. It does not yet overlap independent GPU schedules.
Current storage remains pointer-oriented; descriptor resource mapping and
full native adapters for other graphics APIs remain separate work. Vulkan
still has no loadable physics driver.

The Windows D3D12 driver embeds DXIL generated from the common kernel manifest.
Its public command path binds and dispatches the shared rigid-body,
rigid-constraint, contact, debug, and query schedules and advertises those
tested capabilities. Fluid, cloth, rope, soft-body, smoke, and their couplings
remain unadvertised.

The opt-in Metal preview connects the same public API and reference scheduler
to native Apple-GPU dispatch, advertising only rigid bodies and constraints.
It bundles generated Slang Metal libraries, not generated CPU physics. The CPU
driver similarly advertises only its implemented rigid capabilities.
All existing domain and
coupling entries now compile separately to Metal. Native kernel tests use actual
Apple GPU buffers and dispatch through the reusable asynchronous C++ compute
boundary. AVBD now has shared staged dispatches for adapters without cooperative
grid barriers; the original cooperative entry uses the same stage equations.
The preview Gallery uses the Metal driver for physics and software rendering
for images. It still synchronizes for prepared solver counts and frame output;
other domains, native import and GPU rendering handoff remain unfinished. See the
[Metal qualification checklist](METAL-QUALIFICATION.md) for the release gate.

The AVBD numerical core also compiles to WGSL and its reflection contract is
tested. This qualifies the shared equations for WebGPU code generation only;
the full domain modules still need graphics-resource entry wrappers before a
WebGPU driver can execute the common schedule.

## Compiler toolchain

CMake resolves Slang through `cmake/Slang.cmake` and pins version 2026.18. An
offline build can provide the compiler explicitly:

```sh
cmake -S . -B build \
  -DPARALLEL_MATER_SLANGC_EXECUTABLE=/opt/slang-2026.18/bin/slangc
```

Generated CUDA, C++, PTX, MSL, HLSL, DXIL, SPIR-V, WGSL, and reflection
records remain in the build tree.

## Validation

The portable AVBD test executes 105 fixtures against the host reference. The
full mirror target compiles every shared command-runtime entry point to CUDA,
PTX when available, SPIR-V, and Metal. The architecture test rejects native CUDA
physics kernels, direct physics launches, legacy command-runtime `World`
dependencies, and dispatch adapters that are not backed by generated Slang.

```sh
cmake --build build --target parallel-mater-slang-geometry-constraints
cmake --build build --target parallel-mater-slang-kernel-mirrors
ctest --test-dir build --output-on-failure \
  -R 'parallel-mater-(avbd|slang)'
```

The [Shader portability workflow](../.github/workflows/shader-portability.yml)
runs without path filters on pull requests, branch pushes, and merge queues.
Its `Shader portability required` check fails unless every platform job passes:

- Linux, Windows, and macOS compile every manifest module to CUDA source and
  SPIR-V, every entry separately to Metal, plus the AVBD numerical core to CUDA,
  Metal, HLSL, and WGSL. CPU tests and reflection checks run on each platform.
- Windows also compiles every manifest entry to DXIL and builds the D3D12
  driver against those outputs. Device-dependent execution tests are excluded
  from this compiler workflow and remain separate runtime qualification.
- macOS additionally compiles and links every generated MSL entry with Apple's
  native toolchain. Native pipeline/layout and GPU execution tests run when an
  Apple GPU is available; lack of hardware produces an explicit skip, never a
  CPU fallback. Xcode's Metal toolchain must be installed.
- Linux and macOS build the Slang CPU driver and execute public-API asynchronous
  submission tests, including actual rigid integration. All platforms test
  submission lifetime, dependency scheduling, and deferred failures without a GPU.

All gates use the pinned Slang toolchain and treat shader compiler warnings as
errors. Native MSL compilation exempts only unused-variable diagnostics because
Slang retains unused atomic fetch results. Compiler/reflection gates require
neither a CUDA toolkit nor a GPU; CUDA source generation
does not replace NVCC/PTX compilation or device execution in CUDA qualification.
The existing optional PTX gate still runs when NVRTC is available. A complete
Metal execution schedule and command-API driver remain unimplemented. Reflection
and test logs are retained as CI artifacts.
Repository branch rules should require the stable `Shader portability required`
check; the workflow itself does not modify repository protection settings.

Compilation alone does not qualify a command-API target. Its schedule must use
the shared Slang kernels, pass API and architecture checks, and complete gallery
review on that target. Retired backend-specific adapters are rejected by the
architecture test.
