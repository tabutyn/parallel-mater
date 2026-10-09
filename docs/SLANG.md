# Shared Slang kernels

Slang is the source language for backend-neutral GPU physics. CUDA and Metal
currently use two shared modules:

- `src/slang/fluid.slang` owns standalone fluid cell indexing, neighbor
  force/foam evaluation, and integration.
- `src/slang/avbd.slang` owns the numerical core used by the production CUDA
  and Metal rigid-body/brick solvers: dual updates, force projection, friction,
  six-degree-of-freedom block assembly and solve, and constraint-row updates.

The AVBD collision geometry, contact graph, graph coloring, body storage,
scheduling, and dispatch adapters remain backend-specific. D3D12 still uses
its existing rigid solver. The Slang AVBD conformance entry also emits HLSL so
the shared numerical contract is ready for a later D3D12 adapter migration.

## Source layout

The source tree follows the CUDA subsystem boundary. Each particle system gets
one top-level file (`fluid.slang`, then `cloth.slang`, `soft_body.slang`,
`rope.slang`, and `smoke.slang`). Rigid-body numerical constraints live in
`avbd.slang`; other shared constraints belong in `constraints.slang`. Pairwise
interactions belong under `src/slang/couplings/` and migrate only after both
standalone systems pass their backend conformance gates.

Particle sources, destroy planes, contact generation, rigid/deformable/smoke
couplings, CUDA CUB sorting, and Metal radix sorting remain native. Do not put
couplings into a particle-system file.

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
