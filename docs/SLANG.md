# Shared Slang physics

Slang is the source language for GPU physics. File names describe physics
domains rather than native APIs or solver experiments.

- `types.slang` contains shared resource and value layouts.
- `math.slang` contains numerical helpers.
- `device_intrinsics.slang` contains dispatch, barrier, and atomic operations.
- `geometry.slang` contains rigid triangle/BVH contact generation.
- `geometry_constraints.slang` contains AVBD contact and joint solving.
- `fluid.slang`, `cloth.slang`, `rope.slang`, `soft_body.slang`, and the smoke
  modules contain their domain kernels.
- Domain-pair files contain coupling kernels.
- `queries.slang` contains scene-query kernels.

`fluid.slang` is the single fluid implementation. Device adapters allocate,
transfer, synchronize, and dispatch generated kernels; they contain no fluid,
collision, constraint, or coupling equations.

## Compiler toolchain

CMake resolves Slang through `cmake/Slang.cmake` and pins version 2026.18. An
offline build can provide the compiler explicitly:

```sh
cmake -S . -B build \
  -DPARALLEL_MATER_SLANGC_EXECUTABLE=/opt/slang-2026.18/bin/slangc
```

Generated CUDA, PTX, MSL, HLSL, DXIL, SPIR-V, and reflection records remain
in the build tree.

## Validation

The portable AVBD test executes 38 fixtures against the host reference. The
full mirror target compiles every shared entry point to CUDA and PTX. The
architecture test rejects native physics kernels, direct physics launches,
legacy consumer `World` dependencies, and dispatch adapters that are not backed
by generated Slang.

```sh
cmake --build build --target parallel-mater-slang-geometry-constraints
cmake --build build --target parallel-mater-slang-kernel-mirrors
ctest --test-dir build --output-on-failure \
  -R 'parallel-mater-(avbd|slang-kernel)'
```

Compilation alone does not qualify runtime dispatch. A target becomes an
advertised command-API capability after its schedule uses the shared Slang
kernels, passes API and architecture checks, and completes gallery review on
that target. Historical backend goldens and performance comparisons remain
diagnostic data, not target-acceptance gates.
