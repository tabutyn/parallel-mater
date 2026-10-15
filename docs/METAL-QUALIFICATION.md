# Metal all-domain qualification

Work branch: `codex/slang-metal-all-domains`. This work must not merge into
`main` or replace the installed Gallery until all domains and interactions in
[the roadmap](ROADMAP.md) pass, including currently missing interaction modes.
Compiling existing shaders is necessary but does not meet that release gate.

## Execution boundary

Physics equations remain in shared Slang and compile to GPU Metal libraries.
The device adapter owns allocation, device addresses, binding, residency,
dispatch, dependencies, synchronization, and native interoperability. It must
not execute CPU-generated Slang as a hidden fallback. Host scheduling is not
host physics execution.

The current macOS Gallery still uses the existing serial Slang CPU driver.
This branch does not install a Metal driver, change feature reporting, or
deploy a replacement app.

## Shader foundation

`PARALLEL_MATER_BUILD_SLANG_MIRRORS=ON` builds every canonical kernel entry to
Metal on all host platforms and native `.metallib` files on macOS. The manifest
currently contains 135 production entries across 23 modules. Tests add one
device-operation contract kernel. Generated files remain in the build tree.

- Compile one entry per invocation. Slang 2026.18 multi-entry Metal output can
  omit builtin thread inputs from later kernels without a compilation error.
  Portable reflection/source checks guard this regression.
- Put dispatch dimensions in a scalar-field constant buffer. Free global
  uniforms trigger an internal error in the pinned Metal compiler target.
- Encode `MTLBuffer.gpuAddress`, never the CPU `contents` pointer. Declare
  indirect-buffer residency and validate native versus Slang argument offsets.
  Nested resource-record pointers still need relocation in the runtime driver.
- 64-bit mask/counter reductions use two atomic 32-bit words on Metal. They
  have no fetch return value and permit reads only after producer completion;
  they are not general atomic 64-bit load/store operations. Device tests cover
  contention, carry and wraparound.
- Group barriers cover device and threadgroup memory. Metal soft-body neighbor
  batches use a full 32-lane SIMD group (four nodes per 128-thread dispatch), not
  CUDA's masked 16-lane half-warps. The adapter must validate SIMD width.
- Use the existing smaller AVBD metadata caches with device-memory tails.
  Leaf contact evaluation requires per-leaf device scratch followed by the
  existing reduction kernel, avoiding oversized inline threadgroup manifolds.

## Required before merge

- [ ] Backend-neutral asynchronous schedule and resource lifetime management.
- [ ] Correct multi-dispatch replacement for AVBD's CUDA grid barrier. The
      current graphics group barrier is **not** a cross-threadgroup barrier;
      compiling the solve entry does not qualify multi-group execution.
- [ ] All domain dispatch bindings, nested address relocation, transfers,
      deferred failures, capacity reporting, queries and native GPU interop.
- [ ] Mutable coupling graph, pipeline rebind behavior, parameter units,
      working one-way/two-way settings, and per-pair/per-mode capabilities.
- [ ] Fluid-fluid pressure, contact, viscosity, balanced momentum, differing
      material properties, miscible/immiscible interfaces and thermal modes.
- [ ] Cloth-cloth, soft-body/soft-body, rope-rope, smoke-smoke, self-contact,
      and explicit smoke overlap/exchange semantics.
- [ ] All 15 distinct-family pairs, including general interactions beyond the
      current attachment-only demonstrations, qualified mode by mode.
- [ ] Independent public-API tests: lifecycle, endpoint replacement, limits,
      timestep/mass-ratio sweeps, fast impacts, containment, long runs, momentum,
      energy and mixed scenes containing three or more families.
- [ ] Gallery/engine GPU handoff and native controls, no mandatory frame
      readback, physical quality and performance evidence on identified hardware.

The present GPU tests create every native pipeline and verify its argument
layout and resource limits. They execute dispatch/atomic contracts, multi-group
rigid integration, and soft-body projection/damping comparisons between shared
GPU entries. These are focused kernel tests, not all-domain physics conformance.

## Local evidence (2026-10-15)

Apple M4, Slang 2026.18, native Metal 3.0 libraries:

- 135 production entries plus one device-contract entry compile and link.
- All 136 native pipeline creations and argument layouts pass on the M4.
- The isolated host/reflection/GPU suite passes 18/18 tests.
- The existing CPU Gallery configuration passes 24/24 tests, including its
  headless timing regression and the new Metal tests. Auto-deploy stays off.
- The six executed kernels pass with Metal API and GPU shader validation
  enabled, using the runner's `--execution-only` mode. Full-manifest shader
  instrumentation is not qualified: instrumentation raises AVBD prepare's
  threadgroup storage to 47,104 bytes, above the M4's 32,768-byte limit.

Native shader compilation disables fast math and implicit floating-point
contraction, preserving Slang's precise mode and explicit `fma` operations.
Linux/Windows CI and complete simulation/interaction qualification have not run
for this branch. Nothing here establishes release readiness.

## Reproduce

```sh
cmake -S . -B build-metal-qualification \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPARALLEL_MATER_BUILD_CUDA=OFF \
  -DPARALLEL_MATER_BUILD_D3D12=OFF \
  -DPARALLEL_MATER_BUILD_GALLERY=OFF \
  -DPARALLEL_MATER_BUILD_SLANG_CPU_DRIVER=OFF \
  -DPARALLEL_MATER_BUILD_SLANG_MIRRORS=ON
cmake --build build-metal-qualification --parallel 4
ctest --test-dir build-metal-qualification --output-on-failure
```

Hardware-dependent tests exit 77 when Metal is unavailable. A skipped test is
not runtime qualification; GPU access must be enabled for local hardware runs.

On macOS, instrument all kernels exercised by the focused execution tests:

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  build-metal-qualification/tests/solver/parallel-mater-slang-metal-kernel-tests \
  build-metal-qualification/generated/slang/metal/manifest.txt --execution-only
```

Run the default test too: `--execution-only` does not replace the complete
manifest's pipeline, layout and resource-limit checks.
