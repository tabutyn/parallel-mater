# Slang solver direction

Parallel Mater is moving its GPU numerical kernels toward
[Slang](https://shader-slang.org/) so one implementation can target CUDA,
Metal, and D3D12-compatible HLSL. The first migration slice is the AVBD
numerical core in `src/slang/avbd.slang`.

## Current scope

The Slang module implements the complete backend-independent AVBD math:

- augmented-Lagrangian dual warm start and penalty updates;
- unilateral and bounded force projection;
- radial Coulomb friction and its positive-semidefinite stiffness scales;
- coupled six-degree-of-freedom block accumulation and conditioned LDLT solve;
- bilateral, spring, motor, and one-sided limit row evaluation.

The conformance entry point is deliberately thin. Slang emits precise-float
CUDA, Metal, and Shader Model 5.1 HLSL from the same module. Reflection tests
lock the structured-buffer ABI at 260-byte inputs, 36-byte outputs, and 64
threads per group. On CUDA, Slang also uses NVRTC to produce PTX; the test loads
that PTX through the CUDA driver and executes 38 fixtures against the existing
portable C++ contract.

This PR does **not** switch the production worlds to the conformance entry
point. Collision generation, graph coloring, body storage, scheduling, and
runtime dispatch remain in the CUDA and Metal adapters. D3D12 still uses its
existing rigid solver. There is therefore no runtime performance claim in this
slice. The next migration can import this module from production kernels while
keeping these ABI and numerical tests as the gate.

## Reproducible compiler

`PARALLEL_MATER_BUILD_SLANG_AVBD` follows `BUILD_TESTING` by default. The build
requires Slang 2026.18 exactly. If `slangc` is not on `PATH`, CMake downloads an
official release archive selected by host OS and architecture, verified with
the SHA-256 digest published on the Slang GitHub release. Supported bootstrap
hosts are Linux x86-64/AArch64, macOS x86-64/Apple Silicon, and Windows
x86-64/AArch64.

Use an existing installation with:

```sh
cmake -S . -B build \
  -DPARALLEL_MATER_SLANGC_EXECUTABLE=/path/to/slangc
cmake --build build --target parallel-mater-slang-avbd
```

The generated sources, reflection records, and PTX are written below
`build/generated/slang`; generated files are not checked in. CUDA builds run
the PTX conformance test. Native Metal builds additionally compile the emitted
MSL to AIR, and native D3D12 builds compile the emitted HLSL with FXC.

Run the focused tests with:

```sh
ctest --test-dir build --output-on-failure \
  -R 'parallel-mater-avbd-(core|slang)'
```
