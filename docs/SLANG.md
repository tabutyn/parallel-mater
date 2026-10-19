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

The CUDA, Metal, D3D12, and Vulkan reference drivers compile these sources to
their GPU targets and dispatch the generated kernels through the public command
API. Their native adapters allocate, transfer, bind, synchronize, and dispatch;
they do not contain alternate physics equations. The Linux/macOS Slang CPU
driver separately compiles supported kernels to C++ and dispatches them
serially for development and reference checks.

## Reference scheduling and submission

[`reference_schedule.hpp`](../src/simulation/reference_schedule.hpp) contains
the shared domain and coupling order as ordinary C++. Its state parameter owns
resource descriptions and bookkeeping; its backend parameter supplies typed
Slang dispatch, copy/clear, timing, readback, error checks, and queue waits.
The header compiles without a native SDK. Backend bindings connect its
operations to CUDA, Metal, D3D12, Vulkan, or the serial Slang CPU driver.
Dispatches must preserve resource dependencies with the native barriers
required by that backend. The [common resource boundary](COMMON-RESOURCE-BOUNDARY.md)
describes reflected bindings, range-based dependencies, retained resources,
and completion ownership, including which adapters use that interface.

The command API copies each accepted submission into a queue-owned worker.
The worker activates the device, runs ready submissions, retains frame tokens
through completion, drains partially issued work on failure, and publishes a
completion shared by fences, command buffers, and timeline signals. Unresolved
timeline waits leave the worker free to run their producers. Ready submissions
remain FIFO; explicit dependencies must order access to shared resources.

The reference implementation still has worker waits and explicit readbacks for
some preparation, emission, phase transfer, diagnostics, and resource metadata.
Asynchronous API submission does not promise that every backend overlaps
independent GPU schedules or eliminates all host waits. Storage remains
pointer-oriented, and adoption of the common command/completion interface
differs by adapter.

## Execution targets

| Driver | Generated target | Advertised domain scope |
|---|---|---|
| CUDA | CUDA source compiled by NVCC / PTX | Rigid bodies/constraints, liquids, cloth, ropes, soft bodies, smoke, paint, and implemented couplings |
| Metal | Metal libraries compiled with Apple's toolchain | Rigid bodies/constraints, liquids, cloth, ropes, soft bodies, smoke, paint, and implemented couplings |
| D3D12 | DXIL embedded in the driver | Rigid bodies/constraints, liquids, cloth, and their implemented couplings |
| Vulkan | SPIR-V with reflected argument bindings | Rigid bodies/constraints, liquids, cloth, ropes, soft bodies, smoke, paint, and implemented couplings |
| Slang CPU | C++ host code | Rigid bodies/constraints; experimental liquids are opt-in |

This table describes implementation scope. Exact interactions and native
interoperability must be queried from the selected device; the
[qualification index](QUALIFICATION.md) records the evidence and unresolved
failures separately. See [Vulkan requirements](VULKAN.md) and
[Metal qualification](METAL-QUALIFICATION.md) for platform details.

The Metal driver uses native Apple GPU buffers and dispatch through the common
asynchronous C++ compute boundary. AVBD uses shared staged dispatches on
adapters without cooperative grid barriers; the cooperative entry uses the
same stage equations. macOS and iPad Gallery draw directly from public GPU
output buffers through the shared Metal renderer. Physics/graphics handoff is
fenced, and diagnostics and some controllers retain explicit readbacks. This
does not establish imported engine-device support or full physics/performance
qualification.

Android defaults to the Vulkan driver. Its explicitly selected CPU development
build enables experimental liquids; Vulkan failure does not select CPU physics.

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

The portable AVBD test executes equation fixtures against the host reference. The
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
The existing optional PTX gate still runs when NVRTC is available. The
Metal command-API preview now executes the existing domain schedule; a fully
asynchronous GPU-only schedule and all-domain qualification remain incomplete. Reflection
and test logs are retained as CI artifacts.
Repository branch rules should require the stable `Shader portability required`
check; the workflow itself does not modify repository protection settings.

Compilation alone does not qualify a command-API target. Its schedule must use
the shared Slang kernels, pass API and architecture checks, and complete gallery
review on that target. Retired backend-specific adapters are rejected by the
architecture test.
