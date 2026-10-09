# Shared Slang kernels

Slang is the source language for backend-neutral GPU physics. The repository
has two complementary layers:

- `src/slang/fluid_shared.slang` owns production standalone fluid cell indexing, neighbor
  force/foam evaluation, and integration.
- `src/slang/fluid.slang` is the 1:1 CUDA-reference mirror, including the
  lifecycle kernels that are not yet shared by production backends.
- `src/slang/avbd.slang` owns the numerical core used by the production CUDA
  and Metal rigid-body/brick solvers: dual updates, force projection, friction,
  six-degree-of-freedom block assembly and solve, and constraint-row updates.
- Every CUDA physics source has a same-basename mirror under `src/slang/`.
  Those files preserve all 120 CUDA entry-point names across rigid bodies,
  fluid, cloth, soft bodies, ropes, smoke, and every coupling. The rigid
  adapter is split between `geometry_constraints.slang` and
  `avbd_cuda.slang`, while the portable numerical core remains
  `avbd.slang`.

The production CUDA and Metal adapters still own backend dispatch, allocation,
sorting, and resource lifetime. D3D12 still uses its existing rigid solver.
The full mirror gate targets CUDA and PTX; the smaller portable AVBD core also
emits Metal and HLSL for cross-backend conformance.

## Source layout

The source tree follows the CUDA subsystem boundary exactly. For example,
`src/fluid_soft_body.cuh` maps to `src/slang/fluid_soft_body.slang`, and
`src/world.cu` maps to `src/slang/world.slang`. Entry-point names do not change;
only the folder and extension do. Shared ABI and CUDA-target intrinsics live in
`cuda_common.slang`. `fluid_shared.slang` is intentionally separate because it
is the compact cross-backend production pilot rather than the 1:1 reference
mirror.

## Compiler toolchain

The build resolves Slang once through `cmake/Slang.cmake` and pins version
2026.18. If an exact `slangc` is not on `PATH` or under `SLANG_DIR`, CMake
downloads the matching official host package and verifies its SHA-256 digest.
macOS, Linux, and Windows packages are mapped for arm64 and x86-64 hosts. An
offline or managed build can provide the compiler explicitly:

```sh
cmake -S . -B build \
  -DPARALLEL_MATER_SLANGC_EXECUTABLE=/opt/slang-2026.18/bin/slangc
```

CMake rejects another compiler version so generated source and ABI names do
not drift silently between machines.

For Metal, Slang emits MSL for both modules. The AVBD core is included by the
native geometry/storage adapter, while standalone fluid entry points compile
to a second AIR object. Apple's `metal` compiler validates both generated
sources, and `metallib` links them with the remaining native kernels. The build
uses Slang's `metallib_3_2` capability because Metal output remains
experimental upstream; Apple compiles the result as Metal 4.

For CUDA, Slang emits CUDA C++ for both modules. `world.cu` includes the
generated AVBD core and standalone fluid kernels, then launches them through
the existing cooperative-grid, stream, and CUB schedules. NVCC remains the
final compiler. No Slang runtime library is linked or shipped.

## AVBD conformance

`PARALLEL_MATER_BUILD_SLANG_AVBD` follows `BUILD_TESTING` by default. The
conformance entry emits precise-float CUDA, Metal, and Shader Model 5.1 HLSL.
Reflection tests lock its structured-buffer ABI at 260-byte inputs, 36-byte
outputs, and 64 threads per group. CUDA builds also generate PTX with NVRTC and
execute 38 fixtures against the portable host reference.

```sh
cmake --build build --target parallel-mater-slang-avbd
ctest --test-dir build --output-on-failure \
  -R 'parallel-mater-avbd-(core|slang)'
```

Generated source, reflection records, PTX, AIR, and metallibs stay below the
build tree. A migration is complete only when both target generators succeed,
the native compilers accept their output, and runtime and conformance tests
pass on the target hardware.

## Full CUDA mirror gate

`PARALLEL_MATER_BUILD_SLANG_MIRRORS` follows `BUILD_TESTING` by default. It
builds `parallel-mater-slang-kernel-mirrors`, compiling every mirrored module
to CUDA and, when NVRTC is available, PTX with warnings treated as errors.
The `parallel-mater-slang-kernel-mirror-coverage` test independently checks
same-basename files and exact entry names.

```sh
cmake --build build --target parallel-mater-slang-kernel-mirrors
ctest --test-dir build -R slang-kernel-mirror-coverage --output-on-failure
```
