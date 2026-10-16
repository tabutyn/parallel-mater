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

## Runtime foundation implemented

`src/device/compute.hpp` defines a private C++ execution boundary with buffers,
reflected arguments, resource-record relocation, command lists and asynchronous
completion tokens. `src/device/metal_compute.mm` implements it with native Metal:

- Shared and private GPU allocations, upload snapshots, copies and fills.
- Separate tracked-resource encoders establish device-wide dependencies.
- Submission does not wait; callers explicitly wait or query completion.
  Native execution errors are reported by completion operations.
- Referenced allocations survive until GPU completion, even if the caller
  drops both the command list and its completion token.
- Embedded pointers, including arrays of pointers, are explicitly relocated
  to GPU addresses. Unrelocated records and raw host-pointer arguments are
  rejected. Host records remain unchanged. Consumers must declare every
  indirectly referenced allocation for residency and hazard tracking.
- Native Metal reflection supplies record layouts. Slang 2026.18 reports
  incorrect four-byte storage for some nested booleans; its pointee sizes
  must not determine native buffer strides. Native pointer-array reflection
  also needs normalization of `metal::array` wrappers and zero pointer strides.

This is reusable device code, not a test-only dispatcher. The public command
API and existing domain resource manager are **not connected to it yet**.
Full runtime migration, ABI error translation, native interop/import, and
Gallery packaging remain pending. There is still no installable Metal driver.

### AVBD dispatch schedule

The shared Slang solver now exposes `solve_avbd_stage_kernel`. Its equations
are also used by the original cooperative entry: there is no Metal solver copy.
`src/simulation/avbd_schedule.hpp` orders impact preparation, body colors,
dual updates, island translation correction, state application and finishing.
Cross-workgroup dependencies are dispatch boundaries, never group barriers.

The encoder accepts proven upper bounds for iteration/color counts and guards
unused work on the GPU, avoiding a readback between prepare and solve. It is a
correctness foundation, **not a performance-qualified schedule**: using body
count as the color bound can enqueue many empty stages. Optimized indirect work
and integration into the complete runtime remain required. The cooperative
`solve_avbd_kernel` entry is still invalid for multi-workgroup Metal dispatch;
Metal must use the staged entry instead.

## Shader foundation

`PARALLEL_MATER_BUILD_SLANG_MIRRORS=ON` builds every canonical kernel entry to
Metal on all host platforms and native `.metallib` files on macOS. The manifest
currently contains 136 production entries across 23 modules. Tests add two
device-operation/record contract kernels. Generated files remain in the build tree.

- Compile one entry per invocation. Slang 2026.18 multi-entry Metal output can
  omit builtin thread inputs from later kernels without a compilation error.
  Portable reflection/source checks guard this regression.
- Put dispatch dimensions in a scalar-field constant buffer. Free global
  uniforms trigger an internal error in the pinned Metal compiler target.
- Encode `MTLBuffer.gpuAddress`, never the CPU `contents` pointer. Declare
  indirect-buffer residency and validate native versus Slang argument offsets.
  Record relocation is implemented and GPU-tested; domain resource management
  still needs to use it throughout the runtime.
- 64-bit mask/counter reductions use two atomic 32-bit words on Metal. They
  have no fetch return value and permit reads only after producer completion;
  they are not general atomic 64-bit load/store operations. Device tests cover
  contention, carry and wraparound.
- Group barriers cover device and threadgroup memory. Metal soft-body neighbor
  batches use a full 32-lane SIMD group (four nodes per 128-thread dispatch), not
  CUDA's masked 16-lane half-warps. The adapter must validate SIMD width.
- Use 256-entry AVBD body/contact metadata caches with device-memory tails,
  leaving room for native shader-validation instrumentation.
  Leaf contact evaluation requires per-leaf device scratch followed by the
  existing reduction kernel, avoiding oversized inline threadgroup manifolds.

## Required before merge

- [ ] Backend-neutral asynchronous schedule and resource lifetime management.
- [x] Shared multi-dispatch AVBD replacement, tested across workgroups against
      the original single-workgroup solve, including pre-refactor binaries.
- [ ] Integrate and performance-qualify staged AVBD in the complete runtime.
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
layout and resource limits. They execute through the reusable device adapter:
dispatch/atomic contracts; private-memory transfers and resource lifetimes;
nested pointer records; multi-group rigid integration; soft-body scalar/SIMD
projection and damping; and staged AVBD contacts/joints. These are focused
kernel/runtime-service tests, not all-domain public-API physics conformance.

## Local evidence (2026-10-15)

Apple M4, Slang 2026.18, native Metal 3.0 libraries:

- 136 production entries plus two device-contract entries compile and link.
- All 138 native pipeline creations and argument layouts pass on the M4.
- The isolated host/reflection/GPU suite passes 19/19 tests, with no GPU skip.
- The existing CPU Gallery configuration passes 25/25 tests, including its
  headless timing regression and the new Metal tests. Auto-deploy stays off.
- All ten executed kernels pass with Metal API and GPU shader validation
  enabled, using the runner's `--execution-only` mode. This now includes AVBD
  prepare, cooperative and staged solving, plus nested-pointer records.
- AVBD tests use 97/545 bodies, cross-workgroup contacts, all eight joint types
  with active rows, limits, a motor, springs, disabled/broken/breaking joints,
  both impact phases, device-memory cache tails and over-dispatched bounds.
  States, impulses and joint outputs match within the test tolerances. The
  same cases also pass against the original solver compiled from `d9ba088`.
- Full-manifest shader instrumentation remains unqualified: instrumented
  `pm_generate_rigid_leaf_pairs` exceeds the M4's 32,768-byte threadgroup limit.
  AVBD prepare's previous instrumentation limit has been fixed. This limitation
  does not affect normal native pipeline creation, and that leaf kernel is not
  among the currently executed physics tests.

Native shader compilation disables fast math and implicit floating-point
contraction, preserving Slang's precise mode and explicit `fma` operations.
Linux/Windows CI and complete simulation/interaction qualification have not run
for this branch. Nothing here establishes release readiness.

## Upstream integration (2026-10-16)

Merged `origin/main` at `54d99ed`, including the shared reference schedule,
asynchronous public submissions and D3D12 adapter work. Preserved staged AVBD
while incorporating upstream's 32-bit resource flags and explicit flag tests.
The Metal compute boundary still needs to be connected to that reference
schedule and the public driver API; this merge alone does not supply it.

On Apple M4, the rebuilt Gallery test configuration passes all 27 tests,
including asynchronous submission, headless Gallery timing, all 138 native
Metal pipeline/layout checks and the focused GPU execution fixtures. All
canonical CUDA-source, SPIR-V and native Metal compiler gates rebuild cleanly.
Auto-deployment remains disabled. No installed app or main-branch release is
qualified by these development tests.

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
