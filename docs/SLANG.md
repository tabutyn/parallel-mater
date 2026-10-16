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
full native adapters for other graphics APIs remain separate work. The
[Vulkan preview](VULKAN.md) now binds this schedule for rigid bodies, constraints,
liquids, and liquid–rigid coupling. Metal remains unavailable.

The Windows D3D12 driver embeds DXIL generated from the common kernel manifest.
Its public command path binds and dispatches the shared rigid-body,
rigid-constraint, contact, debug, and query schedules and advertises those
tested capabilities. Fluid, cloth, rope, soft-body, smoke, and their couplings
remain unadvertised.

No Metal device adapter or Metal physics implementation is shipped. The CPU
driver advertises only the rigid capabilities that its shared Slang schedule
currently executes, with experimental liquids explicitly enabled for the Android
preview; other domains remain unavailable. Cross-target generation
of the shared AVBD module remains a compiler and reflection portability check;
it does not expose a backend-specific runtime or advertise Metal execution.

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
PTX when available, and SPIR-V. The architecture test rejects native CUDA
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
  SPIR-V, plus the AVBD numerical core to CUDA, Metal, HLSL, and WGSL. CPU tests
  and reflection checks run on each platform.
- Windows also compiles every manifest entry to DXIL and builds the D3D12
  driver against those outputs. Device-dependent execution tests are excluded
  from this compiler workflow and remain separate runtime qualification.
- macOS additionally compiles the generated AVBD MSL with Apple's `metal`
  compiler and links a `.metallib`. Xcode's Metal toolchain must be installed.
- Linux and macOS build the Slang CPU driver and execute public-API asynchronous
  submission tests, including actual rigid integration. All platforms test
  submission lifetime, dependency scheduling, and deferred failures without a GPU.

All gates use the pinned Slang toolchain and treat shader compiler warnings as
errors. They require neither a CUDA toolkit nor a GPU; CUDA source generation
does not replace NVCC/PTX compilation or device execution in CUDA qualification.
The existing optional PTX gate still runs when NVRTC is available. Full Metal
domain schedules and a Metal driver remain unimplemented; the native Metal gate
covers only the AVBD core. Reflection and test logs are retained as CI artifacts.
Repository branch rules should require the stable `Shader portability required`
check; the workflow itself does not modify repository protection settings.

Compilation alone does not qualify a command-API target. Its schedule must use
the shared Slang kernels, pass API and architecture checks, and complete gallery
review on that target. Retired backend-specific adapters are rejected by the
architecture test.
