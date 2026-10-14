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

The command-API CUDA driver dispatches generated Slang kernels. Its device and
simulation adapters allocate, transfer, synchronize, and dispatch; they do not
contain alternate physics equations.

The Metal compatibility `World` builds every physics kernel from Slang,
including rigid broad phase, narrow phase, AVBD constraints, particles,
cloth, rope, soft bodies, fluids, smoke, paint, and all coupling passes.
Metal-specific entry points use the packed runtime ABI in `metal_common.slang`
and the domain adapter modules. No native Metal physics source remains. The
Metal compatibility target is not yet an advertised command-API device driver.

## Compiler toolchain

CMake resolves Slang through `cmake/Slang.cmake` and pins version 2026.18. An
offline build can provide the compiler explicitly:

```sh
cmake -S . -B build \
  -DPARALLEL_MATER_SLANGC_EXECUTABLE=/opt/slang-2026.18/bin/slangc
```

Generated CUDA, PTX, SPIR-V, MSL, reflection records, AIR, and metallibs remain
in the build tree.

For Metal, Slang emits one MSL module per production subsystem or coupling so
each module preserves the compatibility runtime's argument-table bindings.
Apple's `metal` compiler validates every generated source independently, and
`metallib` links the generated AIR objects. The build uses Slang's
`metallib_3_2` capability and Apple compiles the result as Metal 4.

## Validation

The portable AVBD test executes 38 fixtures against the host reference. The
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
review on that target. The Metal compatibility suite separately validates the
generated MSL/AIR/metallib pipeline and its retained `World` behavior.
