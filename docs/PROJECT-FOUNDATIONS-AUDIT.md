# Project foundations audit — 2026-10-16

Initial runtime probes used `e9e0144`. During review, `origin/main` advanced to
`a137042`; those changes were merged and the affected findings rechecked against
source. This audit covers the library, reference runtime, loader, specification,
conformance and SDK boundaries. Platform expansion and shader optimization are
deferred behind the foundation work in
[ROADMAP.md](ROADMAP.md#current-priority-project-foundations).

## Assessment

Keep the loadable-driver architecture, registry-generated C API, shared Slang
physics and extracted reference scheduler. An installed C application can
already use the loader without building a solver. The missing work is making
resource lifetime, coupling behavior, validation and qualification dependable
for applications and independently developed drivers.

The next deliverable should be a headless public-API lifecycle test suite and
reusable simulation storage. Adding more adapters first would spread the same
lifecycle and contract defects across more implementations.

## 1. High priority: declared capacity does not survive repeated replacement

**Runtime reproduction:** the existing command API test passes with one fluid
write. Replacing that write with 256 identical `pmCmdSetFluidParticles` commands
before the step fails at fence completion with
`PM_ERROR_OUT_OF_DEVICE_MEMORY` (`-4`). Live capacity remains two particles in
one fluid slot; the queried persistent allocation is 9,589,504 bytes. Both runs
use the existing CUDA driver and otherwise identical test setup.

The cause is visible in the shared implementation:

- [resources.cpp](../src/api/resources.cpp), `pmCmdSetFluidParticles`, removes
  the old fluid and allocates a new fluid even when its capacity is unchanged.
- [runtime.cpp](../src/simulation/runtime.cpp), `release_managed`, does not reclaim
  bound arena allocations.
- [memory_arena.cpp](../src/simulation/memory_arena.cpp) advances allocation
  offsets and only reclaims them on a whole-arena reset.

This is exhaustion within application-bound storage, not evidence that the GPU
ran out of physical memory. The mechanism is shared; this audit reproduced the
failure on CUDA only before the final upstream merge. The merged source keeps
the same remove/reallocate and non-reclaiming arena behavior; its new device
services do not add reclamation. Other domain replacements and failed partial
allocations need the same lifecycle audit.

**Required outcome:** reserve reusable storage per declared resource capacity,
or implement bounded reclamation with explicit completion rules. Derive memory
requirements and allocations from one resource plan instead of maintaining
separate byte estimates. Define replacement failure behavior and rollback.
Repeated create/write/remove/recreate operations within live capacity must not
consume unbounded additional arena space or disturb unrelated populations.

## 2. High priority: coupling graph is still pipeline initialization state

[domains.cpp](../src/api/domains.cpp), `initialize_couplings`, returns immediately
once `simulation->couplings_initialized` is true. Binding another pipeline does
not reconcile its edges. Created private coupling IDs are discarded, leaving
the public API without an edge update/removal mechanism. Replacing a coupled
fluid can fail because the private runtime still references its old handle.

There are also observable parameter-contract gaps:

- Fluid/cloth `interaction_radius` is populated from `maximumAcceleration`.
- Public `twoWay` is not read by the API implementation.
- Public sleeping settings are not applied; simulation creation sets
  `rigid_sleeping = true` independently of pipeline settings.
- Liquid/liquid is explicitly rejected, while `domainCouplings` is too broad
  to distinguish contact, attachment, drag, thermal and directional modes.

**Required outcome:** simulation-owned, generation-safe coupling resources;
pipeline-owned reusable material/solver policy; explicit creation, mutation,
removal and endpoint replacement semantics. Query capabilities per pair, mode
and material model. Unsupported settings must fail explicitly. Implement and
qualify distinct fluid–fluid population interaction after this contract is
stable; retain all 15 distinct-family pairs and same-family interactions in the
coverage matrix. Do not infer general collision support from attachment demos.

## 3. High priority: normative behavior and validation must agree

The generated specification says destruction does not wait
([content.py](../tools/reference/content.py), `units`), while
[API.md](API.md#synchronization) promises draining pending submissions and
resource destructors call `pmDeviceWaitIdle`. This matters for engine threading,
shutdown and latency. The specification requires
`PM_ERROR_INCOMPATIBLE_VERSION` for unsupported API versions, but
[loader/entry.cpp](../src/loader/entry.cpp), `pmCreateInstance`, returns
`PM_ERROR_INVALID_ARGUMENT` for a newer version.

The current [layer ABI](../include/parallel_mater/pm_layer.h) exposes only a
command name and primary handle. [validation.cpp](../src/layers/validation.cpp)
checks their presence; it cannot inspect buffer ranges, resource dependencies,
feature enablement or semaphore values. Passing with this layer enabled is not
evidence of synchronization or lifetime validation.

**Required outcome:** one normative contract for lifetime, host threading,
command snapshots, deferred failure, reset, overflow and memory visibility.
Give rules stable identifiers and map each to positive and negative public-API
tests. Generate typed layer interception from the registry, track object state,
and use a project-owned layer name instead of `PM_LAYER_KHRONOS_validation`.
Document API, loader-driver ABI, layer ABI and extension compatibility
separately; test compatible and incompatible independently built drivers.

The spec also prescribes one AVBD path. Keep algorithm-specific requirements in
an explicit reference-solver profile; vendor-neutral compliance should state
observable behavior and physical quality. Header generation and dispatch-route
coverage are valuable, but do not test these behavioral promises.

## 4. Architectural follow-up: finish the neutral resource boundary

[reference_schedule.hpp](../src/simulation/reference_schedule.hpp) is ordinary
C++ and parameterized by a backend. The newly merged Windows work also moves
the host runtime into [runtime.cpp](../src/simulation/runtime.cpp), removes the
include-`runtime.cu` shims and introduces common queue, dispatch and device-service
types. Do not repeat that completed extraction. The remaining
[generate_vulkan_dispatch.py](../tools/generate_vulkan_dispatch.py) derives host
signatures from generated CUDA text and existing CUDA wrappers. This does not
introduce a CUDA runtime dependency, but couples adapter maintenance to CUDA
source shapes and private pointer layouts.

**Required outcome:** incrementally define common resource identities, typed
dispatch arguments, allocation plans, access dependencies and completion
ownership. Generate bindings from a canonical schema plus Slang reflection.
Retain the existing kernel manifest and schedule; avoid a wholesale scheduler
rewrite. Test the boundary with a recording/mock adapter before adding another
platform. Native API details should remain inside adapters.

The merged D3D12 gallery has a renderer consuming public GPU output. Use that
vertical slice to test the common engine handoff: device identity, borrowed
device/queue lifetime, buffer ownership, synchronization and output formats.
Engine-owned device/queue integration and portable handoff qualification remain
distinct from rendering on a device created by the reference driver. Implement
Unity, Unreal and Godot packages against the resulting common contract.

Memory reporting belongs to this boundary too. For example, Vulkan currently
reports device-local heap size minus tracked allocations as available memory
([schedule_context.cpp](../src/driver/vulkan/schedule_context.cpp),
`memory_budget`). Define the distinction between physical heap size, usable
budget, application allocations and driver-private/unknown usage before using
the query as a reliable workload admission decision.

## 5. Qualification exists, but is not independent yet

The 30-case runner uses public `pm*` commands, with existing analytic cases,
invariants and tolerance/envelope comparisons. Preserve that work. However,
[CMakeLists.txt](../CMakeLists.txt) still creates `parallel-mater-conformance`
inside gallery + testing conditions and links gallery scene support. The
upstream merge removed the CUDA-only creation condition; preserve that progress.
Its scene model still includes private `simulation/types.hpp`. A loader-only
build has the case registry and comparator tests, but no executable physics
conformance runner.

**Required outcome:** build an installable headless runner against the public
SDK only, with driver selection at runtime. Separate portable case data from
gallery UI/rendering and private solver types. Capability-aware results must
distinguish pass, fail and unsupported; advertised features require their full
test set. Keep reference-trajectory regression separate from vendor-neutral
physical compliance and quality-constrained benchmarks.

Extend lifecycle, capacity, failure-recovery and multi-system tests before
growing platform matrices. Then qualify all advertised interaction modes,
mixed three-or-more-family scenes and fluid–fluid behavior. Record source,
driver, hardware, compiler, test revision and quality thresholds with every
qualification result. Historical goldens are evidence for their recorded
revision, not automatic qualification of a refactored driver.

The merged [Windows qualification record](ROADMAP.md#windows-shared-runtime-cleanup-2026-10-16)
reports only 7/17 same-build CUDA/D3D12 case comparisons passing, and a new
independent brick test failing on both drivers: approximately 7 mm penetration
against a 2 mm gate. Those are upstream-reported results, not reruns on this
laptop. They strengthen the need for independent physical-quality gates:
agreement between implementations alone cannot establish correctness. Keep
these failures visible and fix common behavior without relaxing the gates.

## 6. SDK and repository cleanup need release gates

A clean loader-only build and install succeeded. An external C consumer built
and ran against a relocated install, reporting API 0.2. However,
`find_package(ParallelMater 0.2 CONFIG REQUIRED)` fails: the CMake package is
version 0.1.0 while headers and driver manifests advertise API 0.2.

**Required outcome:** define package/API version relationships and pre-1.0
compatibility policy. Add clean-install, relocation, C/C++ consumer and external
sample-driver CI tests using installed files only. Ship reproducible SDK
components, documented discovery paths and minimal headless examples. Split
loader, reference-driver, gallery and conformance build definitions by their
actual dependencies, not by the platform that first hosted them.

Add contribution, release, extension-allocation and qualification-review rules
alongside the MIT license. Keep current capability documentation generated or
checked against a machine-readable evidence matrix, without equating successful
shader compilation with runtime support.

Local cleanup remains necessary: reconcile the archived unmerged changes in
the two older worktrees and move active builds off their cached toolchain
directories before retiring them. Preserve the existing snapshots and source;
do not discard old physics work merely to make worktree status clean.

## Verification and scope

- Fresh loader-only Release build, serial: **16/16 CTest checks passed**.
  This was repeated after the upstream merge. No reference physics driver was
  built in this configuration.
- Clean install and relocated external C consumer: **passed** without a
  requested package version; requesting version **0.2 failed** as described.
- Existing CUDA command API test adapted in a temporary file: **one write
  passed; 256 identical replacements failed with deferred result -4** on the
  pre-merge driver. The newly merged GPU drivers were not rebuilt or runtime
  qualified in this audit.
- Coupling, layer, specification and resource-boundary findings are source
  inspection findings, not claims of new runtime qualification on all drivers.

The temporary lifecycle reproducer uses `tests/api/command_api_tests.cpp` as its
base. Replace its single pre-step `pmCmdSetFluidParticles` call with a loop of
256 identical calls, keeping setup, capacity and input unchanged. Build against
the public header and loader, select the CUDA manifest, and inspect fence
completion. Run once with a loop count of one as the control. No production
source or shader changes were made for this audit.
