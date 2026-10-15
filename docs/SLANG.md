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

The command-API CUDA driver dispatches generated Slang kernels. The macOS Slang
CPU driver compiles the same rigid-body, constraint, contact, debug, and query
kernels to C++ and dispatches them serially as a correctness-first development
path. Their device and simulation adapters allocate, transfer, synchronize,
and dispatch; they do not contain alternate physics equations.

The Windows D3D12 driver embeds the DXIL generated from the same AVBD prepare
and solve entries. Device creation builds both compute pipelines, binds every
reflected UAV/constant resource, dispatches a zero-body schedule, inserts a UAV
barrier, fences, reads the schedule back, and validates its counters. Its public
command path currently exposes only core transfer services, so it does not
advertise rigid-body or other physics features yet.

No Metal device adapter or Metal physics implementation is shipped. The CPU
driver advertises only the rigid capabilities that its shared Slang schedule
currently executes; other domains remain unavailable. Cross-target generation
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
  -R 'parallel-mater-(avbd|slang-kernel)'
```

Compilation alone does not qualify a command-API target. Its schedule must use
the shared Slang kernels, pass API and architecture checks, and complete gallery
review on that target. Retired backend-specific adapters are rejected by the
architecture test.
