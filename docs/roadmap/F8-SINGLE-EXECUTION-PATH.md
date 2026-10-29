# F8 is the simulation architecture

Maintainer direction, 2026-10-25: F8 replaces the production simulation
architecture. It is not an optional performance mode, a second solver product,
or a per-platform feature switch. Every supported simulation uses the same
fixed-tick architecture. This decision supersedes the opt-in/legacy-default
selection policy in the original F8-00 contract and F8-32 implementation.

This is the required destination, not a claim that current binaries implement
it. Task cards remain the ownership and completion record.

## One execution contract

- Every successful physics tick advances exactly once by 1/60 second, publishes
  one shared dynamic spatial epoch, executes bounded algebraic iterations and
  commits once. Invalid input, unsupported capabilities and resource exhaustion
  return explicit errors; they never select another schedule or smaller steps.
- Rigid bodies, liquids, cloth, soft bodies, ropes, both smoke representations,
  and their currently supported interactions use that same production path.
  Empty scenes, mixed scenes, cold starts, topology changes, reset, headless
  execution, debug capture and recovery must obey it too.
- Shared Slang supplies physics on every GPU. CUDA, Vulkan, D3D12 and Metal
  supply native resource/dispatch/synchronization services. CPU reference
  execution and independent numerical oracles do not constitute selectable
  alternative GPU physics or permission for platform-specific equations.
- Keep the application's create/resources/commands/submit/output model. Remove
  the requirement to discover or opt into an F8 execution profile. Normal
  simulation creation selects the sole implementation. No mode flag, environment
  switch, build option, adapter fallback or compatibility branch may resurrect
  the legacy production scheduler in the released SDK.
- API/ABI versioning describes compatibility, not a choice of physics engines.
  If old timestep/substep semantics cannot be honored, publish an explicitly
  incompatible API/SDK version and reject old requests clearly. A compatibility
  shim is acceptable only if it unambiguously uses the same single schedule;
  silently accepting incompatible semantics is not acceptable.
- Library ticks do not depend on monitor refresh or sleep to enforce a wall
  clock. Offline/headless clients may run successive fixed ticks as fast as they
  can. Gallery owns real-time pacing: at most one tick per render callback,
  interpolation above 60 Hz, deliberate slow motion below 60 Hz, no catch-up
  bursts, and reset of accumulated display debt after stalls/backgrounding.
- Gallery selects this behavior once in shared support. Desktop, Android and
  iPad shells supply input, monotonic timestamps, lifecycle and presentation.
  No user-facing enable switch is required. Correct platform presentation caps
  as part of rollout; the current iPad 30 FPS request cannot demonstrate 60 Hz.
- The sole schedule must substantially reduce actual native dispatch overhead.
  F8-54 batches Metal rigid work; F8-55 replaces remaining exhaustive coupling
  searches with shared-epoch candidates and batches compatible edges. F8-56
  proves reductions against independent ceilings in installed Mac/iPad Gallery.
  These are completion gates, as specified in
  [dispatch reduction requirements](F8-DISPATCH-REDUCTION.md).

"Always" applies to every supported execution and recovery path. It does not
promise that arbitrary scenes meet 16.67 ms on every device, or make previously
unimplemented P-series interactions supported. Preserve the existing quality
envelopes, workload counts, capability matrix and measured performance gates.

## Observed gaps at 8d77e5b5

| Source | Current limitation | Completion owner |
|---|---|---|
| `src/api/resources.cpp`, `select_execution_profile` | Omitted creation extension selects `legacy_default` | F8-50 |
| `src/api/resources.cpp`, `pmCreateSimulation` | Real-time creation rejects non-rigid domains, couplings, sources, paint and debug capacities | F8-39 through F8-50 |
| `src/simulation/production_realtime_backend.cu.inc` | Native prediction/solve binding is restricted to rigid bodies | F8-40 through F8-49 |
| `src/simulation/runtime_schedule.cu.inc` | Profile condition selects one of two production schedules | F8-50 |
| `examples/support/include/parallel_mater_gallery/session.hpp` and shared scene creation | Gallery defaults to legacy clock and does not request the F8 execution path | F8-51 |
| `examples/ios/gallery_main.mm` | Display requests a maximum of 30 FPS | F8-51 |

F8-11 through F8-21 contain reusable domain/operator/schedule work. Their
completion is not proof those operators execute through the production public
API. F8-34 and F8-35 explicitly retained a rigid/query subset. Preserve these
completed cards and evidence; new tasks close the end-to-end gaps without
rewriting their history or treating unsupported results as passes.

## Migration and qualification order

1. F8-39 establishes common storage, staging, rollback and admission for all
   domain families. F8-40 through F8-44 connect each domain's existing shared
   operators. F8-45 through F8-47 connect the existing interaction matrix.
2. F8-48 completes source/removal/phase/topology lifecycle behavior. F8-49
   integrates paint, debug capture and public outputs/queries. Each task needs
   actual production execution evidence, not only mock-schedule tests or shader
   compilation, and must preserve the one-tick invariants.
3. F8-50 removes the legacy production schedule and profile-selection API,
   makes the complete schedule the normal execution contract, and updates
   registry/bindings/reference documentation and installed consumer tests.
   F8-51 makes shared Gallery use it with appropriate presentation on every
   platform. These are release requirements, not optional follow-ups.
4. F8-54 and F8-55 complete Metal batching and coupling search/batching repairs.
   After those fixes and F8-51, F8-56 verifies actual dispatch reductions and
   physical quality on installed Mac/iPad Gallery. F8-52 then freezes the
   complete source/API/Gallery candidate and independent work-budget inputs.
   F8-24 through F8-30 qualify that candidate on their registered GPU
   targets; F8-53 supplies final CPU/public-contract qualification. F8-31 packages
   and deploys verified builds; F8-99 audits the sole production path and all
   required results.

F8-38 completed the narrowly scoped rigid-path refreeze. F8-23 completed its
interim qualification; its history and evidence remain intact. That candidate
cannot satisfy the final single-path release. F8-52 freezes its replacement as
`realtime-single-step-v1-5a811c4`, manifest SHA-256
`f0f15a65fed178625b263c6462e7e0d9397fd4880cfc2a3a73582b98584876ae`.
F8-24 through F8-30 and F8-53 depend on F8-52 and accept only that exact
candidate. Historical candidate references remain provenance; evidence from
different candidates may not be combined into a pass. See the
[F8-52 freeze record](F8-52-CANDIDATE-FREEZE.md).

The frozen `realtime-profile-v1.json` and earlier candidate records remain
unchanged historical inputs during this planning correction. F8-50 updates the
executable contract, including machine-readable negotiation rules and tests;
F8-52 captures the resulting hashes. Numeric quality limits may not be weakened
to make this migration pass.

Existing branches may temporarily contain migration scaffolding, but "F8 done"
requires its removal: one public creation path, one production scheduler, no
legacy selection, complete supported-domain coverage, shared Gallery adoption,
measured dispatch reduction under independent ceilings, and matching installed
device evidence.
