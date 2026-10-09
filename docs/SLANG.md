# Shared Slang kernels

Slang is the source language for backend-neutral physics kernels. The first
pilot owns standalone particle-fluid cell indexing, neighbor force and foam
evaluation, and integration in `src/slang/fluid.slang`. CUDA and Metal execute
code generated from that file; particle sources, destroy planes, contacts, and
all couplings remain in their existing backend implementations during the
pilot.

The source tree follows the CUDA subsystem boundary. Each particle system gets
one top-level file (`fluid.slang`, then `cloth.slang`, `soft_body.slang`,
`rope.slang`, and `smoke.slang`). Shared rigid constraints belong in
`constraints.slang`. Pairwise interactions belong under `src/slang/couplings/`
and are migrated only after both participating standalone systems pass their
backend conformance gates. Do not put couplings into a particle-system file.

## Compiler toolchain

The build pins Slang 2026.18. If an exact `slangc` is not on `PATH`, CMake
downloads the matching official release package for the host and verifies its
SHA-256 digest. macOS arm64/x86-64, Linux arm64/x86-64, and Windows
arm64/x86-64 packages are mapped in `cmake/Slang.cmake`. An offline or managed
build can provide the same compiler explicitly:

```bash
cmake -S . -B build \
  -DPARALLEL_MATER_SLANGC_EXECUTABLE=/opt/slang-2026.18/bin/slangc
```

CMake rejects another compiler version so generated source does not change
silently between machines.

For Metal, `slangc` emits one MSL module containing the three entry points.
Explicit D3D register annotations preserve the existing Metal argument-table
indices. The Apple `metal` compiler then validates that MSL and combines its
AIR with the remaining native Metal kernels before `metallib` and embedding.
The Slang target uses the `metallib_3_2` capability because upstream still
marks Metal generation experimental and newer inferred capabilities have had
toolchain-version issues; Apple still compiles the emitted source as Metal 4.

For CUDA, `slangc` emits CUDA C++ with compact kernel parameters. `world.cu`
includes that generated file and launches the kernels through the existing
stream and CUB sort. NVCC remains the final compiler. No Slang runtime library
is linked or shipped on either backend.

The generated source is a build artifact under `generated/`; only `.slang`
sources are tracked. A migration is complete only when both target generators
succeed, each native compiler accepts the result, and the subsystem's runtime
and conformance tests pass on its hardware.
