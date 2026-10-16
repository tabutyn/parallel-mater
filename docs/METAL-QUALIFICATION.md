# Metal all-domain qualification

Work branch: `codex/slang-metal-all-domains`. A fully qualified release still
requires all domains and interactions in
[the roadmap](ROADMAP.md) to pass, including currently missing interaction modes.
Compiling existing shaders is necessary but does not meet that release gate.
On 2026-10-16 the user explicitly authorized a limited, labeled Apple-GPU
Gallery preview before full qualification. That exception is not completion
of the all-domain roadmap or authorization to claim production qualification.

## Execution boundary

Physics equations remain in shared Slang and compile to GPU Metal libraries.
The device adapter owns allocation, device addresses, binding, residency,
dispatch, dependencies, synchronization, and native interoperability. It must
not execute CPU-generated Slang as a hidden fallback. Host scheduling is not
host physics execution.

The opt-in `PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW=ON` Gallery bundles a
loader-discovered Metal driver for rigid bodies/constraints, liquids, cloth,
soft bodies, rope, particle/grid smoke, implemented cross-domain couplings and
paint. Its device name and window title identify the unqualified preview.
Liquid–liquid coupling remains unadvertised and rejected, with no CPU fallback.
The default build still uses
the serial CPU driver. The preview starts on Point; its uncalibrated brick
default is 16 rather than a production performance claim.

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

The preview connects the public command API and existing pointer-oriented
resource manager to this compute boundary through `src/driver/metal` and a
transitional scheduler vocabulary in `src/device/metal_compat`. All physics,
including collision, compaction and constraints, dispatches shared Slang on
the GPU. Mesh resource records are relocated explicitly; shared host addresses
are never substituted for GPU addresses. Native borrowed device/buffer handles
are exposed, but imports and timestamp queries remain unsupported.

This is not the final asynchronous GPU-only architecture: the submission
worker waits between frames, shared-memory transfers can synchronize, and
AVBD reads prepared work counts before staged solving. The Mac interactive and
iPad Gallery now render directly from the public Metal output buffer using one
graphics adapter. Physics completion precedes graphics, and graphics completion
precedes buffer reuse/destruction. Headless/reference rendering, explicit debug
captures and CPU scene controls retain readback. Existing domain dispatch integration is
present; missing roadmap modes, complete physics qualification, native engine
handoff and performance qualification remain pending.

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
  Metal uses four contact workers in the existing shared inline reduction,
  or the existing per-leaf scratch/reduction path for small scenes. A 256-entry
  transformed-leaf cache retains the existing uncached tail path without
  changing the 128-lane candidate traversal. CUDA defaults are unchanged.

## Required for full all-domain release

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
At that merge, the Metal compute boundary was not yet connected to the reference
schedule and public driver API; the subsequent preview integration supplies it.

On Apple M4, the rebuilt Gallery test configuration passes all 27 tests,
including asynchronous submission, headless Gallery timing, all 138 native
Metal pipeline/layout checks and the focused GPU execution fixtures. All
canonical CUDA-source, SPIR-V and native Metal compiler gates rebuild cleanly.
Auto-deployment remains disabled. No installed app or main-branch release is
qualified by these development tests.

## Initial rigid Gallery preview (2026-10-16)

On Apple M4, the opt-in preview configuration passes 30/30 CTest tests without
GPU skips. This includes the public native-command/contact fixture, asynchronous
submissions, shared frame timing, all 138 pipeline/layout checks, focused kernel
execution and the new seven-scene Gallery preview smoke test.

- Fixed, Point, Hinge/Slider, Piston, Generic, Motor/Spring and 16-brick scenes
  each complete 120 GPU steps, with finite states, normalized rotations and a
  rendered final image. At three steps their maximum component differences
  from the serial Slang reference are below 0.00027 (test tolerance 0.005).
- A separate two-cube public-API test checks contact settling position and
  vertical speed, not just finiteness. It also verifies invalid allocation
  recovery, transfers, native handles and memory budget reporting.
- Public command and asynchronous tests, plus five-step Point, Hinge and
  16-brick Gallery runs, pass Metal API and GPU shader validation. The smaller
  transformed-leaf cache fixes the earlier instrumentation limit for these
  exercised rigid paths; full-manifest instrumentation has not been rerun.
- That initial preview rejected unsupported fluid selection without CPU fallback.
  Its bundle contains only the Metal driver manifest and relative shader
  resource paths. The CPU reference is built separately for comparison tests.
- Long-horizon numerical agreement remains unqualified: an exploratory
  120-step Point comparison had a maximum state-component difference of about
  1.4 from the CPU reference. The smoke test does not hide this behind a
  long-run trajectory-equivalence claim or establish its cause.

The installed preview is a development exception, not completion of the
all-domain release gate. No Windows/Linux runtime qualification was performed.
Unsupported scene selection retains the current scene and reports a capability
error. At that initial milestone native rendering was still pending; the later
shared Gallery graphics path removes rendering readback, not solver host waits.

## Current-domain Gallery expansion (2026-10-16)

Merged upstream `6f7af12` into the work branch at `818514d`, preserving upstream
backend selection, version reporting and D3D12 dispatch parameters alongside
shared staged AVBD. The expanded preview replaces every generated unsupported
domain dispatch stub with native bindings to the canonical Slang kernels.
No physics equations were copied into Metal or CPU implementation files.

- All six existing domains and their implemented interactions are connected to
  the public API, including emitters, destroyers, exports, queries and paint.
  This does **not** implement the missing roadmap interactions listed above.
- Rope/soft-target records, smoke grids (including pointer arrays), meshes and
  nested paint tables validate host/native layouts and relocate GPU addresses.
  Indirect allocations remain resident and retained through completion.
  Paint-only UV meshes are included even when not used for body collisions.
- Soft-body SIMD dispatches use four nodes per native 128-thread workgroup,
  rather than the CUDA scheduler's eight-node grouping.
- Public-API regressions check 513 liquid particles with unused capacity,
  analytic free fall, sorting, queries, pair momentum, static contact and
  two-way rigid impulse balance; cloth/soft-body free fall, pinned invariance,
  bond lengths, surface export and replacement; rope free fall and uniform-wind
  smoke transport with both particle and grid solvers. These fixtures also run
  with Metal API and shader validation enabled on Apple M4.
- Gallery integration revealed and fixed two software-rendering bugs: torn
  cloth requires per-corner positions, and near-plane crossing triangles must
  be clipped before projection. Renderer tests cover both; depth interpolation
  is also perspective-correct. These changes do not make rendering GPU-native.

The new Gallery domain matrix checks populated domain output, finite exported
fluid/debug values, existing paint/cloth visibility checks, final image output,
and the actual Metal device label. It is execution smoke, not comprehensive
physical conformance, long-horizon stability or performance qualification.

On Apple M4, all 34 CTest tests pass without GPU skips. The domain matrix runs
20 liquid/deformable/transport/mixed scenes for 30 steps each (120 for both
paint scenes), alongside the seven existing 120-step rigid-scene checks.
All 20 domain cases also pass at those step counts with Metal API and shader
validation enabled, alongside the five public native API fixtures and renderer
regressions (7/7 selected CTest tests, including the scene matrix).

Build, test and explicitly deploy the preview on macOS:

```sh
cmake -S . -B build-macos-metal-preview \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPARALLEL_MATER_BUILD_CUDA=OFF \
  -DPARALLEL_MATER_BUILD_D3D12=OFF \
  -DPARALLEL_MATER_BUILD_GALLERY=ON \
  -DPARALLEL_MATER_BUILD_SLANG_CPU_DRIVER=ON \
  -DPARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW=ON \
  -DPARALLEL_MATER_MACOS_GALLERY_AUTO_DEPLOY=OFF
cmake --build build-macos-metal-preview --parallel 6
ctest --test-dir build-macos-metal-preview --output-on-failure
cmake --build build-macos-metal-preview --target deploy-macos-gallery
```

The CPU-driver option supplies test comparisons, not an installed fallback.
Normal full builds refresh bundled runtime libraries even when only the driver
changes; the deploy target also refreshes them before atomically replacing and
signing `/Applications/ParallelMater Gallery.app`. Quit an already-running
Gallery before reopening it to load the new driver. Disable auto-deploy in
older CPU build directories so they cannot silently replace the preview.

## Shared Gallery delivery and GPU graphics (2026-10-16)

The newer user instruction permits merging the labeled development preview to
main; it does not waive the production gates above. Engine integration is
explicitly deferred. Use `CMakePresets.json` for backend-specific builds and
`scripts/Deploy-Gallery.py` for revision/backend-verified Apple deployment.
The iPad app now packages the approved logo; Android and iPad consume one shared
session and timer, including the same bounded qualification routine.

`CommandScene::gpu_frame` exports only public output-buffer regions after the
physics fence. The shared graphics adapter retains the native buffer while its
command reads it and finishes before the next simulation step can overwrite it.
Cloth (including torn per-corner surfaces), soft surfaces, rigid transforms,
liquid/smoke/rope particles and paint are read by graphics shaders. UIKit and
the Mac window supply a Metal layer; neither implements physics. Mac overlays
remain CPU-generated UI textures, not CPU-generated scene geometry or images.

This is a **fenced direct-rendering preview**, not asynchronous frames-in-flight
or complete external-engine interoperability. Native buffer/device/semaphore
imports, general queue-sharing and timestamp queries remain unadvertised.
Those public API capabilities and the deferred engine slice require their own
lifetime/error qualification. The full all-domain checklist remains open.

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
