# Parallel Mater roadmap

## Vision and acceptance contract

Parallel Mater will provide stable, fast simulation of five particle-system
families (fluid, cloth, soft bodies, rope, and smoke), rigid bodies, and their
interactions. The reference GPU physics implementation uses Slang, with CUDA,
D3D12 compute/DXIL, Vulkan/SPIR-V, and Metal execution adapters. Unity, Unreal,
and Godot integrations consume the same public C API.

The specification and reference implementation are intended to remain MIT
licensed. The long-term goal is an independently implementable standard with
competing vendor drivers and eventual Khronos standardization. Khronos adoption
requires an industry proposal and participation; it is not an existing project
affiliation or an outcome guaranteed by the API design.

The specification defines observable physics, units, accuracy requirements,
resource ownership, synchronization, errors, and capability negotiation. The
shared Slang schedule is the reference implementation. Vendor implementations
may optimize layouts, kernels, and schedules while satisfying the same
behavioral and quality requirements. Reference-solver regression tests and
vendor-neutral conformance are distinct obligations.

## Architecture to preserve and complete

- Keep one versioned C API generated from `registry/pm.xml`, a small independent
  loader, installable drivers, and validation layers.
- Keep simulation resources and the mutable coupling graph in the simulation.
  Physics pipelines own reusable solver/material policy. Changing a pipeline or
  replacing an endpoint must not silently retain stale coupling behavior.
- Put scheduling, dependency tracking, substeps, resource descriptions, and
  memory planning in a backend-neutral reference runtime.
- Keep domain equations, collision, coupling operators, and GPU primitives in
  shared Slang. Reference device adapters provide allocation, binding,
  dispatch, barriers, transfers, synchronization, queries, and native interop.
- Keep engine adapters, scene authoring, rendering, input, and gallery controls
  outside the physics implementation. Public output must support GPU rendering
  without mandatory per-frame CPU readback.
- Build conformance and benchmarks as independent public-API consumers. Every
  advertised feature needs evidence from its own driver and hardware.

## Interaction coverage

"All couplings" includes all 15 distinct-family pairs among fluid, cloth,
soft bodies, rope, smoke, and rigid bodies, plus meaningful same-family and
self interactions. An attachment demo does not qualify general collision,
friction, drag, containment, thermal exchange, or phase change for that pair.
Track and advertise these modes separately, including one-way/two-way behavior.

**Fluid-fluid system interactions are required.** Distinct fluid populations
must be able to interact across system boundaries, including pressure, contact,
viscosity, and balanced momentum exchange. Define behavior for equal and
different material properties, miscible/immiscible liquids, interfaces, and
optional thermal transfer before advertising those capabilities. Test separate
populations against an equivalent combined population where the material model
makes that equivalence valid. `liquidLiquidCoupling` is currently unsupported;
multiple isolated fluid resources do not satisfy this milestone.

Also specify and qualify cloth-cloth, soft-body/soft-body, rope-rope, and
smoke-smoke system interactions, relevant self-contact, and the existing
rigid-rigid path. Resolve smoke-field overlap and exchange semantics explicitly.
Capability queries must distinguish supported modes, limits, and material
models rather than rely on one broad `domainCouplings` flag.

Coupling qualification includes creation, update, removal, endpoint replacement,
capacity failure, and combinations of three or more families. Test timestep and
mass-ratio sweeps, fast impacts, containment, long-run stability, momentum
balance, and energy behavior with external forces and dissipation accounted for.

## Ordered implementation milestones

### 1. Restore shader portability and require compiler gates

Status: complete (2026-10-15).

The atomic target guards were supplied by upstream `7a724fe` before this
milestone's implementation. The complete 23-module CUDA-source/SPIR-V mirror
build and 15 host/reflection tests pass locally with Slang 2026.18. The
[shader portability workflow](../.github/workflows/shader-portability.yml)
now runs Linux, Windows, and macOS builds, including full Windows DXIL schedules
and native compilation/linking of the AVBD Metal core. All three platform jobs
and the aggregate gate passed in [CI run 38763192659](https://github.com/tabutyn/parallel-mater/actions/runs/38763192659),
with 15 host/reflection tests passing per platform. The Windows run also exposed
and verified a fix for locale-dependent source decoding in the architecture
checker. Runtime physics qualification remains separate.

- Fix the SPIR-V atomic-overload regression introduced by the D3D12 resource
  wrappers. Pointer and descriptor-backed operations must compile only for
  their applicable targets.
- Compile every supported domain/coupling entry through the canonical kernel
  manifest. Require CUDA-source and SPIR-V generation, the full Windows DXIL
  schedule, and existing portable AVBD core targets, including Metal, in CI.
  Use the pinned Slang compiler and fail on compilation/reflection errors.
- Run these gates on pull requests and changes to `main`; do not substitute
  architecture text scanning for shader compilation.
- State exact coverage: compiling the AVBD core to Metal is not a full Metal
  domain schedule or a runtime driver. Complete Metal resource wrappers and
  native runtime qualification remain part of milestones 4 and 6.

Exit: the reproduced SPIR-V failure is fixed, supported compiler targets pass,
and CI exercises the target boundaries that allowed the regression.

### 2. Repair the public physics and coupling contract

Status: pending.

- Separate mutable coupling resources from pipeline policy; define rebind and
  update semantics. Remove one-time initialization that ignores new pipelines.
- Correct parameter mappings and units, including fluid/cloth interaction
  radius versus maximum acceleration. Implement or explicitly reject exposed
  controls such as `twoWay` and sleeping; no silently ignored settings.
- Define per-pair/per-mode capabilities and the same-family interaction scope
  above. Implement fluid-fluid interactions and qualify them before enabling
  `liquidLiquidCoupling`; follow with the other missing interaction modes.
- Reconcile specification, runtime, feature reporting, and negative tests.
  Keep algorithm-specific profiles explicit instead of accidentally freezing
  reference implementation details into the universal contract.

Exit: command behavior matches the specification, supported interaction modes
are queryable, and fluid-fluid lifecycle and physics tests pass.

### 3. Complete portable, asynchronous execution

Status: scheduler extraction and asynchronous submission implemented (2026-10-15);
GPU-only scheduling and performance qualification remain open.

The domain/coupling schedule now lives in backend-parameterized ordinary C++
(`reference_schedule.hpp`), compiled through CUDA and serial Slang CPU adapters.
Public submission owns command snapshots and returns after enqueue. Completion
includes deferred validation and partial-work draining; fences, timelines,
simulation status, and queries account for queued host work as well as GPU work.
CPU tests exercise real rigid integration, multiple submitted frames, pending
resource reuse, wait-before-signal dependencies, and deferred failures. Native
CUDA compilation passes; GPU execution still requires working hardware.

Remaining work includes descriptor resource mapping for graphics adapters,
removing inner-frame readbacks, overlapping independent GPU work, and measured
scan/sort/collision/coupling improvements. The worker currently serializes frame
completion; asynchronous application submission alone does not qualify those
performance milestones.

- Extract the CUDA-dependent host schedule behind backend-neutral execution
  and resource interfaces, including dispatch, indirect work, and barriers.
- Submit physics without waiting for the whole GPU step on the calling thread.
  Retain execution state until completion; expose deferred errors and overflow
  through the documented status/query paths.
- Remove required inner-frame host readbacks, coordinate coupled substeps, and
  express dependencies so independent work can overlap safely.
- Benchmark scalable scan, sort, compaction, collision, and coupling workloads;
  replace serial bottlenecks where measurements justify changes.

Exit: the same reference schedule runs through adapters, submission remains
asynchronous, and correctness survives resource reuse and frames in flight.

### 4. Prove a non-CUDA engine-facing vertical slice

Status: pending.

- Qualify rigid bodies, fluid, cloth, and their interactions through CUDA and
  one non-CUDA driver using the same public commands and cases.
- Implement native device/queue, buffer, and synchronization interoperability,
  including resource ownership, transitions, and lifetime.
- Simulate and render shared GPU output in an engine integration without a
  required per-frame CPU readback. Define authority when engine physics and
  Parallel Mater coexist.

Exit: physical results, lifecycle, native handoff, and quality-controlled
performance pass on both drivers. Shader compilation alone is insufficient.

### 5. Establish independent conformance and performance qualification

Status: pending.

- Build the public-API conformance runner without a CUDA or gallery dependency;
  select any installed driver at runtime.
- Cover every advertised pair and same-family interaction, including
  fluid-fluid, plus mixed scenes and adverse numerical/capacity conditions.
- Combine analytic cases, physical invariants, tolerance-based comparisons,
  and long-run envelopes. Define determinism guarantees separately from
  cross-vendor numerical equivalence.
- Benchmark fixed quality requirements: penetration, density/divergence,
  strain, drift, throughput, memory, and median/p95 latency. Vendors must not
  improve scores by weakening physical quality.
- Give validation layers sufficient command arguments and object state to
  check usage and synchronization. Use Parallel Mater's own layer namespace.

Exit: an independent implementation can determine compliance without private
runtime code or copying the reference solver's exact trajectories.

### 6. Complete backends, engine packages, and adoption work

Status: pending.

- Qualify all declared domains and interaction modes on CUDA, D3D12, Vulkan,
  and Metal; add full shader/resource-wrapper coverage as adapters mature.
- Ship Unity native-plugin/C# integration, Unreal RHI/RDG integration, and
  Godot GDExtension/RenderingDevice integration with common ownership rules.
- Publish the driver SDK, versioned specification, conformance corpus,
  reproducible benchmark procedures, and contribution/governance process.
- Seek independent vendor implementations and engine partners before pursuing
  Khronos standardization through its new-initiative process.

Exit: each advertised driver passes its build, conformance, engine-handoff,
gallery-control, stability, and performance gates on identified hardware.

## Audit baseline (2026-10-15)

The source audit covered `deb158a` and local gallery/hinge changes. A fresh
CPU-only build passed 15 tests. The shader mirror build failed on Vulkan
SPIR-V because descriptor atomic overloads leaked into the pointer target.
GPU runtime and performance were not revalidated during that audit.

CUDA contains the full reference physics path. D3D12 advertises its tested
shared DXIL rigid-body and rigid-constraint schedules plus core transfer
services; the other physics domains remain unadvertised. Vulkan has no runtime
driver; Metal has no current driver. Existing shader generation, historical
galleries, and older worktrees do not establish current backend qualification.
Completion evidence belongs beside each milestone.

## Desktop Vulkan integration progress (2026-10-16)

The loadable Vulkan preview now uses the existing public loader, portable
scheduler, and common Slang kernels for rigid bodies, constraints, liquids, and
liquid–rigid coupling. The shared Windows backend picker also selects CUDA or
Vulkan on Linux. Backend capabilities control scene availability, and the shared
hardware-profile resource limits preview workloads. Details and checks are in
[VULKAN.md](VULKAN.md).

This does not complete the full-domain qualification milestone. Android still
ships the CPU preview; packaging and qualifying this Vulkan driver on the phone,
remaining particle domains/couplings, engine handoff, and sustained performance
remain open.

## Windows shared-runtime cleanup (2026-10-16)

Implementation now uses `src/simulation/runtime.cpp`, the common schedule,
opaque execution queues, dispatch dimensions and device-service events/copies.
The Windows CUDA-shaped compatibility headers and the include-`runtime.cu`
driver shims are removed. CUDA/CPU/Vulkan dispatch interfaces use the same
types; native API interpretation stays in each adapter. CPU and Vulkan runtime
qualification still belongs on their native machines.

D3D12 binds liquid, cloth and their rigid/liquid interactions to the shared
DXIL schedules. Unsupported stubs for those paths are removed. Persistent GPU
allocations no longer receive full readback/re-upload on synchronization:
host edits upload dirty ranges, explicit mappings read requested ranges, and
same-allocation copies use GPU scratch. Tear topology currently still needs
selected host reads at frame boundaries; this is not GPU-only scheduling.

The native drawing adapter consumes public GPU output and presents a shared
texture. Rigid interpolation, invisible authored meshes, expanded tearable
cloth surfaces, and liquid particle/surface/wireframe drawing are implemented.
The liquid presentation field is shared Slang, including the CUDA renderer's
field kernels. Normal interactive D3D12 frames do not read fluid/cloth records
back to the CPU. Controllers, diagnostics and explicit captures retain scoped
readback. Queue overlap, native GPU timings, OptiX lighting/foam equivalence and
performance qualification remain separate work.

Validation distinguishes working execution from parity. The real-window test
has exercised all 13 advertised scenes, Tab/Enter switching, F timing, Point
release and resets, with at least five seconds per scene/load. The regression
suite now requires 300 frames per supported scene, reset/backend retention,
finite/contact/static-body/pinned-cloth invariants, GPU-only output followed by
explicit diagnostic reads, 360 repeated API steps and narrow-range/self-copy
checks. The long hinge/slider fixture independently checks drift, contacts and
penetration. These gates do not replace canonical conformance.

Strict same-build CUDA/D3D12 comparisons remain unqualified: the final 17-case
capture completed on both drivers, with 7 comparisons passing and 10 failing.
Passing cases are cloth-core, cloth-water, compound-weld-lifecycle,
constraint-breaking, constraint-piston, constraint-slider and fluid-lifecycle.
Failures concern other rigid contact/state cases and cloth-tear topology.
Existing tolerances and goldens
are unchanged. Windows CUDA's managed-memory host-read ordering fault in the
cloth-tear run was fixed in the device adapter and the case now completes;
completion does not erase its cross-driver differences. Milestones 4 and 6
remain open until those numerical failures are resolved.

The additional independent brick/brick mesh test also exposes a shared quality
failure: with eight bricks, four substeps and a 2 mm separation gate, D3D12
reaches 7.21844 mm penetration at frame 102 and CUDA reaches 7.02056 mm at
frame 101 (zero-based). Earlier finite-state/300-frame execution passes must
not be described as nonpenetration or full gallery parity. Keep this test red
until the common solver is corrected. An isolated relaxed floating-point
compilation experiment did not qualify rigid-direct, hinge or generic; the
production precise settings remain unchanged.

The final real-window run used installed build `8d5cfc6`, whose SHA-256 matched
the build-tree executable. All 13 supported scenes passed load/reset dwell,
including GPU-only rendering, scoped diagnostics, fluid surface/particle modes
and cloth structure display. Captures and the unchanged-tolerance comparison
report are in `build-cuda-slang-vs2022/d3d12-installed-final/` and
`build-cuda-slang-vs2022/conformance-final-report.json`. Those local artifacts
are evidence, not committed goldens or a claim that the failing cases pass.

The final 2,400-step hinge/slider fixture and 360-step repeated-submission API
test passed. The 20-test execution/architecture/reflection run passed before
the new brick-mesh assertion was exercised; rerunning the strengthened
`parallel-mater-d3d12-gallery-RigidBody` test fails at frame 102 as above. Do
not report that older 20/20 result as a passing current physical-quality suite.

The completed pre-command-API implementation plan is retained in
[the legacy roadmap](history/ROADMAP-LEGACY.md).
