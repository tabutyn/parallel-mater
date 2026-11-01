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
- Put scheduling, dependency tracking, fixed ticks, resource descriptions, and
  memory planning in a backend-neutral reference runtime.
- F8 replaces the production simulation architecture for all supported domains
  and interactions. Ship one fixed-tick schedule and normal public creation,
  with no legacy/F8 mode choice. Gallery must use it through shared support on
  every platform. See the [single execution path decision](roadmap/F8-SINGLE-EXECUTION-PATH.md).
- F8 completion also requires Metal rigid dispatch batching, shared-BVH coupling
  searches and batched edge work, plus independently bounded native dispatch
  reductions in installed Mac/iPad Gallery. F8-54--56 gate the final candidate
  freeze; see [dispatch reduction requirements](roadmap/F8-DISPATCH-REDUCTION.md).
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

## Leasing work across machines

This is the current work queue. Stable leaf IDs below are the only leaseable
units; F-series categories are work areas, not whole-area reservations. Choose the
smallest ready task matching the machine's capabilities. Each task links to a
canonical JSON card with scope, acceptance checks, dependencies, shared-area
locks, evidence and lease history. Completed cards record accepted work and
must not be implemented again; an ID ending in `-00` does not itself mean done.

Read [the lease workflow](roadmap/WORKFLOW.md) before changing a task. Push a
**claim-only PR** first. Ownership begins only when that PR merges into `main`;
a pushed branch or open PR alone is not a lease. Then branch from the merged
claim, implement the task and submit a completion PR with evidence. Update task
cards with `tools/roadmap.py`; the index below is generated from those cards.

Leases last at most 24 hours, use UTC and can be renewed. Expired leases remain
reserved until an explicit reclaim/release PR merges. A unique lease token
prevents a previous owner from completing a reclaimed task. Shared-area locks
prevent different task IDs from reserving the same common implementation area.

F8 is the next architecture/performance priority, alongside finishing existing
foundation reviews and honoring current leases. Read the
[F8 design and cross-machine gates](roadmap/F8-REALTIME-60HZ.md) before claiming.
Its target is one fixed 1/60-second physics step and one shared spatial-index
update per simulation tick; faster rendering interpolates and overload slows
simulation instead of adding catch-up steps. A versioned real-time profile may
trade accuracy for stability and speed, with explicit new bounds and evidence.
This is a redesign, not a change from eight substeps to one without solver tests.

Independent ready tasks may proceed in parallel when dependencies and locks
permit. Entirely completed categories are collapsed below. Completed F5 is
archived out of the live index; its canonical cards, evidence and history stay
under `docs/roadmap/`. Unfinished or deferred categories stay visible.
Other platform expansion, engine plugins and optimization remain subject to
their existing activation/dependency gates; F8 does not activate them implicitly.
Platform-neutral planning may start when its own API, interaction, resource and
package prerequisites have closed; each resulting implementation and
qualification leaf still requires separate activation and lease review.

The lease check must be required with up-to-date branches for race-safe
automated merges. Each merge candidate permits one lease operation. Until configured, serialize claim/renew/reclaim/
completion merges against current `main` and rerun the check immediately before
merging. The workflow does not change GitHub branch-protection settings.

<!-- BEGIN TASK INDEX -->
<!-- Generated by tools/roadmap.py render. Edit task cards, not this index. -->

### F8. 60 Hz single-step simulation and shared spatial indexing

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [F8-00](roadmap/tasks/F8-00.json) | Specify the versioned 60 Hz real-time quality contract | done | — | — |
| [F8-01](roadmap/tasks/F8-01.json) | Inventory every kernel, capability and physical test target | done | — | — |
| [F8-02](roadmap/tasks/F8-02.json) | Verify deployed binaries and capture trustworthy Android baselines | done | F8-00, F8-01 | — |
| [F8-03](roadmap/tasks/F8-03.json) | Build independent single-step quality and work-budget oracles | done | F8-00, F8-01, F8-32 | — |
| [F8-04](roadmap/tasks/F8-04.json) | Implement fixed 60 Hz presentation and overload clock policy | done | F8-00, F8-03, F8-32 | — |
| [F8-05](roadmap/tasks/F8-05.json) | Design one shared spatial-index epoch for all consumers | done | — | — |
| [F8-06](roadmap/tasks/F8-06.json) | Implement low-dispatch shared spatial build primitives | done | F8-00, F8-01, F8-03, F8-05 | — |
| [F8-07](roadmap/tasks/F8-07.json) | Implement the once-per-tick shared BVH resource service | done | F8-00, F8-03, F8-05, F8-06, F8-32 | — |
| [F8-08](roadmap/tasks/F8-08.json) | Move rigid broad phase onto the shared BVH | done | F8-03, F8-07 | — |
| [F8-09](roadmap/tasks/F8-09.json) | Move liquid populations and sources onto shared BVH queries | done | F8-03, F8-07 | — |
| [F8-10](roadmap/tasks/F8-10.json) | Move deformable and smoke collision lookup onto shared indexing | done | F8-03, F8-07 | — |
| [F8-11](roadmap/tasks/F8-11.json) | Stabilize rigid contacts and joints at one 60 Hz step | done | F8-00, F8-03, F8-08 | — |
| [F8-12](roadmap/tasks/F8-12.json) | Stabilize liquid dynamics at one 60 Hz step | done | F8-00, F8-03, F8-09 | — |
| [F8-13](roadmap/tasks/F8-13.json) | Stabilize cloth at one 60 Hz step | done | F8-00, F8-03, F8-10 | — |
| [F8-14](roadmap/tasks/F8-14.json) | Stabilize soft bodies at one 60 Hz step | done | F8-00, F8-03, F8-10 | — |
| [F8-15](roadmap/tasks/F8-15.json) | Stabilize ropes at one 60 Hz step | done | F8-00, F8-03, F8-10 | — |
| [F8-16](roadmap/tasks/F8-16.json) | Stabilize smoke transport and projection at one 60 Hz step | done | F8-00, F8-03, F8-10 | — |
| [F8-17](roadmap/tasks/F8-17.json) | Integrate rigid two-way couplings into the single-step solve | done | F8-08, F8-10, F8-11, F8-13, F8-14, F8-15, F8-16 | — |
| [F8-18](roadmap/tasks/F8-18.json) | Integrate liquid cross-system couplings into the single-step solve | done | F8-09, F8-10, F8-11, F8-12, F8-13, F8-14, F8-15, F8-16 | — |
| [F8-19](roadmap/tasks/F8-19.json) | Integrate remaining deformable and smoke coupling operators | done | F8-10, F8-13, F8-14, F8-15, F8-16 | — |
| [F8-20](roadmap/tasks/F8-20.json) | Assemble the one-tick backend-neutral schedule | done | F8-04, F8-17, F8-18, F8-19 | — |
| [F8-21](roadmap/tasks/F8-21.json) | Add measured dispatch, GPU phase and synchronization budgets | done | F8-01, F8-02, F8-20 | — |
| [F8-22](roadmap/tasks/F8-22.json) | Freeze the cross-machine candidate and complete kernel coverage | done | F8-01, F8-03, F8-20, F8-21 | — |
| [F8-23](roadmap/tasks/F8-23.json) | Qualify independent CPU and public-contract cases | done | F8-22, F8-36, F8-38 | — |
| [F8-24](roadmap/tasks/F8-24.json) | Qualify the candidate on Linux CUDA devices | done | F8-22, F8-36, F8-38, F8-52, F8-61, F8-63, F8-65 | — |
| [F8-25](roadmap/tasks/F8-25.json) | Qualify the candidate on Linux Vulkan devices | done | F8-22, F8-36, F8-38, F8-52, F8-61, F8-63, F8-65, F8-67 | — |
| [F8-26](roadmap/tasks/F8-26.json) | Qualify the candidate on Windows CUDA devices | waiting on dependencies | F8-22, F8-36, F8-38, F8-52, F8-61, F8-63, F8-65, F8-67, F8-70 | — |
| [F8-27](roadmap/tasks/F8-27.json) | Qualify the candidate on Windows D3D12 devices | waiting on dependencies | F8-22, F8-36, F8-38, P1-59, F8-52, F8-61, F8-63, F8-65, F8-67, F8-70 | — |
| [F8-28](roadmap/tasks/F8-28.json) | Qualify the candidate on macOS Metal devices | waiting on dependencies | F8-22, F8-36, F8-38, F8-52, F8-61, F8-63, F8-65, F8-67, F8-70 | — |
| [F8-29](roadmap/tasks/F8-29.json) | Qualify the candidate on physical iOS Metal devices | waiting on dependencies | F8-22, F8-36, F8-38, F8-52, F8-61, F8-63, F8-65, F8-67, F8-70 | — |
| [F8-30](roadmap/tasks/F8-30.json) | Qualify the candidate and sustained performance on Android | waiting on dependencies | F8-02, F8-22, F8-36, F8-38, F8-52, F8-61, F8-63, F8-65, F8-67, F8-70 | — |
| [F8-31](roadmap/tasks/F8-31.json) | Ship and document the qualified single-path SDK and Gallery | waiting on dependencies | F8-23, F8-24, F8-25, F8-26, F8-27, F8-28, F8-29, F8-30, F8-51, F8-52, F8-53, F8-56, F8-58, F8-59, F8-60, F8-61 | — |
| [F8-32](roadmap/tasks/F8-32.json) | Expose the real-time profile through the registry and SDK | done | F8-00 | — |
| [F8-33](roadmap/tasks/F8-33.json) | Integrate the shared spatial epoch into production runtime | done | F8-22 | — |
| [F8-34](roadmap/tasks/F8-34.json) | Bind fixed realtime phases and telemetry to native adapters | done | F8-33 | — |
| [F8-35](roadmap/tasks/F8-35.json) | Expose executable realtime profile through installed API | done | F8-34 | — |
| [F8-36](roadmap/tasks/F8-36.json) | Freeze replacement executable realtime candidate | done | F8-35 | — |
| [F8-37](roadmap/tasks/F8-37.json) | Repair native realtime telemetry envelope | done | F8-36 | — |
| [F8-38](roadmap/tasks/F8-38.json) | Refreeze realtime candidate after telemetry repair | done | F8-37 | — |
| [F8-39](roadmap/tasks/F8-39.json) | Prepare common storage and atomic staging for all domains | done | F8-38 | — |
| [F8-40](roadmap/tasks/F8-40.json) | Connect liquid operators to production fixed ticks | done | F8-39 | — |
| [F8-41](roadmap/tasks/F8-41.json) | Connect cloth operators to production fixed ticks | done | F8-39 | — |
| [F8-42](roadmap/tasks/F8-42.json) | Connect soft-body operators to production fixed ticks | done | F8-39 | — |
| [F8-43](roadmap/tasks/F8-43.json) | Connect rope operators to production fixed ticks | done | F8-39 | — |
| [F8-44](roadmap/tasks/F8-44.json) | Connect both smoke representations to production fixed ticks | done | F8-39 | — |
| [F8-45](roadmap/tasks/F8-45.json) | Connect rigid and deformable interactions to production fixed ticks | done | F8-41, F8-42, F8-43 | — |
| [F8-46](roadmap/tasks/F8-46.json) | Connect liquid interactions to production fixed ticks | done | F8-40, F8-41, F8-42, F8-43, F8-44 | — |
| [F8-47](roadmap/tasks/F8-47.json) | Connect remaining smoke interactions to production fixed ticks | done | F8-41, F8-42, F8-43, F8-44 | — |
| [F8-48](roadmap/tasks/F8-48.json) | Complete source and topology lifecycle in the common tick | done | F8-45, F8-46, F8-47 | — |
| [F8-49](roadmap/tasks/F8-49.json) | Integrate paint, capture and public outputs with the common tick | done | F8-48 | — |
| [F8-50](roadmap/tasks/F8-50.json) | Make the fixed-tick runtime the sole public execution path | done | F8-49 | — |
| [F8-51](roadmap/tasks/F8-51.json) | Make shared Gallery use the sole fixed-tick architecture | done | F8-50 | — |
| [F8-52](roadmap/tasks/F8-52.json) | Freeze the complete single-path SDK and Gallery candidate | done | F8-51, F8-54, F8-55, F8-56 | — |
| [F8-53](roadmap/tasks/F8-53.json) | Requalify CPU and public contracts on the sole-path candidate | waiting on dependencies | F8-23, F8-52, F8-61, F8-63, F8-65, F8-67, F8-70 | — |
| [F8-54](roadmap/tasks/F8-54.json) | Batch Metal rigid solver dispatches | done | F8-34, F8-45 | — |
| [F8-55](roadmap/tasks/F8-55.json) | Use shared BVH candidates and batch coupling work | done | F8-45, F8-46, F8-47 | — |
| [F8-56](roadmap/tasks/F8-56.json) | Verify Gallery dispatch reductions on Mac and iPad | done | F8-51, F8-54, F8-55, F8-57 | — |
| [F8-57](roadmap/tasks/F8-57.json) | Keep installed macOS Gallery synchronized | done | F8-51 | — |
| [F8-58](roadmap/tasks/F8-58.json) | Validate fixed-tick physical device captures | done | F8-52 | — |
| [F8-59](roadmap/tasks/F8-59.json) | Define executable public fixed-tick scenario fixtures | done | F8-52, F8-58 | — |
| [F8-60](roadmap/tasks/F8-60.json) | Reconcile the final capability and public-case matrix | done | F8-52, F8-58 | — |
| [F8-61](roadmap/tasks/F8-61.json) | Refreeze the repaired single-step candidate | done | F8-59, F8-60 | — |
| [F8-62](roadmap/tasks/F8-62.json) | Repair Linux CUDA Gallery candidate failures | done | F8-61 | — |
| [F8-63](roadmap/tasks/F8-63.json) | Refreeze candidate after Linux CUDA repairs | done | F8-62 | — |
| [F8-64](roadmap/tasks/F8-64.json) | Repair installed-public CUDA candidate failures | done | F8-63 | — |
| [F8-65](roadmap/tasks/F8-65.json) | Refreeze candidate after public CUDA repairs | done | F8-64 | — |
| [F8-66](roadmap/tasks/F8-66.json) | Repair Linux Vulkan Gallery candidate failures | done | F8-65 | — |
| [F8-67](roadmap/tasks/F8-67.json) | Refreeze candidate after Linux Vulkan repairs | done | F8-66 | — |
| [F8-68](roadmap/tasks/F8-68.json) | Requalify the replacement candidate on Linux CUDA devices | waiting on dependencies | F8-24, F8-67, F8-70 | — |
| [F8-69](roadmap/tasks/F8-69.json) | Repair shared Gallery CPU session recreation | leased | F8-67 | codex-linux-thomas-laptop / ThomasLaptop / 2026-11-02T00:02:03Z |
| [F8-70](roadmap/tasks/F8-70.json) | Refreeze candidate after CPU Gallery lifecycle repair | waiting on dependencies | F8-69 | — |
| [F8-99](roadmap/tasks/F8-99.json) | Review F8 single-step quality, performance and device exit evidence | waiting on dependencies | F8-00, F8-01, F8-02, F8-03, F8-04, F8-05, F8-06, F8-07, F8-08, F8-09, F8-10, F8-11, F8-12, F8-13, F8-14, F8-15, F8-16, F8-17, F8-18, F8-19, F8-20, F8-21, F8-22, F8-23, F8-24, F8-25, F8-26, F8-27, F8-28, F8-29, F8-30, F8-31, F8-32, F8-33, F8-34, F8-35, F8-36, F8-37, F8-38, F8-39, F8-40, F8-41, F8-42, F8-43, F8-44, F8-45, F8-46, F8-47, F8-48, F8-49, F8-50, F8-51, F8-52, F8-53, F8-54, F8-55, F8-56, F8-57, F8-58, F8-59, F8-60, F8-61, F8-62, F8-63, F8-64, F8-65, F8-66, F8-67, F8-68, F8-69, F8-70 | — |

### P1. Deferred domain and interaction coverage

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [P1-01](roadmap/tasks/P1-01.json) | Plan remaining driver/domain qualification | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99 | — |
| [P1-02](roadmap/tasks/P1-02.json) | Plan remaining same-family operators | done | F1-99, F2-99, F3-99, F4-99, F6-99 | — |
| [P1-03](roadmap/tasks/P1-03.json) | Plan multi-material fluid operators | done | F1-99, F2-99, F3-99, F4-99, F6-99 | — |
| [P1-04](roadmap/tasks/P1-04.json) | Expose multi-material fluid pair state | done | P1-03 | — |
| [P1-05](roadmap/tasks/P1-05.json) | Build heterogeneous fluid pair sampling | done | P1-04 | — |
| [P1-06](roadmap/tasks/P1-06.json) | Exchange miscible pressure and contact | done | P1-05 | — |
| [P1-07](roadmap/tasks/P1-07.json) | Exchange miscible viscosity | done | P1-06 | — |
| [P1-08](roadmap/tasks/P1-08.json) | Schedule miscible fluid pair graphs | done | P1-07 | — |
| [P1-09](roadmap/tasks/P1-09.json) | Qualify and advertise miscible fluids | deferred | P1-08, F5-02, F5-03 | — |
| [P1-10](roadmap/tasks/P1-10.json) | Reconstruct immiscible fluid interfaces | done | P1-05, P1-56 | — |
| [P1-11](roadmap/tasks/P1-11.json) | Apply immiscible pressure and capillarity | done | P1-10 | — |
| [P1-12](roadmap/tasks/P1-12.json) | Apply immiscible interfacial viscosity | done | P1-11 | — |
| [P1-13](roadmap/tasks/P1-13.json) | Schedule immiscible fluid pair graphs | deferred | P1-12 | — |
| [P1-14](roadmap/tasks/P1-14.json) | Qualify and advertise immiscible fluids | deferred | P1-13, F5-02, F5-03 | — |
| [P1-15](roadmap/tasks/P1-15.json) | Add fluid caloric material state | deferred | P1-04 | — |
| [P1-16](roadmap/tasks/P1-16.json) | Exchange conservative fluid heat | deferred | P1-05, P1-15 | — |
| [P1-17](roadmap/tasks/P1-17.json) | Schedule fluid thermal pair graphs | deferred | P1-08, P1-13, P1-16, P1-57 | — |
| [P1-18](roadmap/tasks/P1-18.json) | Qualify and advertise fluid heat exchange | deferred | P1-17, F5-02, F5-03 | — |
| [P1-19](roadmap/tasks/P1-19.json) | Expose cloth contact scopes and controls | done | P1-02 | — |
| [P1-20](roadmap/tasks/P1-20.json) | Build bounded cloth contact generation | deferred | P1-19 | — |
| [P1-21](roadmap/tasks/P1-21.json) | Solve cloth self-contact and friction | deferred | P1-20 | — |
| [P1-22](roadmap/tasks/P1-22.json) | Solve contact between cloth systems | deferred | P1-20 | — |
| [P1-23](roadmap/tasks/P1-23.json) | Schedule cloth contact scopes | deferred | P1-21, P1-22 | — |
| [P1-24](roadmap/tasks/P1-24.json) | Qualify and advertise cloth contact | deferred | P1-23, F5-02, F5-03 | — |
| [P1-25](roadmap/tasks/P1-25.json) | Expose soft-body contact scopes | done | P1-02 | — |
| [P1-26](roadmap/tasks/P1-26.json) | Build bounded soft-surface contacts | deferred | P1-25 | — |
| [P1-27](roadmap/tasks/P1-27.json) | Solve soft-body self-contact | deferred | P1-26 | — |
| [P1-28](roadmap/tasks/P1-28.json) | Solve contact between soft bodies | deferred | P1-26 | — |
| [P1-29](roadmap/tasks/P1-29.json) | Schedule soft-body contact scopes | deferred | P1-27, P1-28 | — |
| [P1-30](roadmap/tasks/P1-30.json) | Qualify and advertise soft-body contact | deferred | P1-29, F5-02, F5-03 | — |
| [P1-31](roadmap/tasks/P1-31.json) | Audit sampled-node rope self-contact | done | P1-02, F5-02 | — |
| [P1-32](roadmap/tasks/P1-32.json) | Expose rope capsule pairs and attachments | done | P1-02 | — |
| [P1-33](roadmap/tasks/P1-33.json) | Build bounded rope segment contacts | deferred | P1-32 | — |
| [P1-34](roadmap/tasks/P1-34.json) | Solve rope capsule self-contact | deferred | P1-33 | — |
| [P1-35](roadmap/tasks/P1-35.json) | Solve contact between rope systems | deferred | P1-33 | — |
| [P1-36](roadmap/tasks/P1-36.json) | Solve rope endpoint attachments | deferred | P1-32 | — |
| [P1-37](roadmap/tasks/P1-37.json) | Schedule rope pair modes | deferred | P1-34, P1-35, P1-36 | — |
| [P1-38](roadmap/tasks/P1-38.json) | Qualify and advertise rope pair modes | deferred | P1-37, F5-02, F5-03 | — |
| [P1-39](roadmap/tasks/P1-39.json) | Expose smoke pair profiles and edges | done | P1-02 | — |
| [P1-40](roadmap/tasks/P1-40.json) | Expose conserved smoke material state | deferred | P1-39 | — |
| [P1-41](roadmap/tasks/P1-41.json) | Build bounded particle-smoke cross neighbors | deferred | P1-40 | — |
| [P1-42](roadmap/tasks/P1-42.json) | Solve particle-smoke mechanical exchange | deferred | P1-41 | — |
| [P1-43](roadmap/tasks/P1-43.json) | Solve particle-smoke thermal exchange | deferred | P1-40 | — |
| [P1-44](roadmap/tasks/P1-44.json) | Schedule particle-smoke pair modes | deferred | P1-42, P1-43 | — |
| [P1-45](roadmap/tasks/P1-45.json) | Qualify particle-smoke pair modes | deferred | P1-44, F5-02, F5-03 | — |
| [P1-46](roadmap/tasks/P1-46.json) | Build conservative smoke-grid overlap maps | deferred | P1-40 | — |
| [P1-47](roadmap/tasks/P1-47.json) | Solve coupled smoke-grid pressure | deferred | P1-46 | — |
| [P1-48](roadmap/tasks/P1-48.json) | Solve smoke-grid viscous exchange | deferred | P1-46 | — |
| [P1-49](roadmap/tasks/P1-49.json) | Solve smoke-grid thermal exchange | deferred | P1-46 | — |
| [P1-50](roadmap/tasks/P1-50.json) | Schedule smoke-grid pair modes | deferred | P1-47, P1-48, P1-49 | — |
| [P1-51](roadmap/tasks/P1-51.json) | Qualify smoke-grid pair modes | deferred | P1-50, F5-02, F5-03 | — |
| [P1-52](roadmap/tasks/P1-52.json) | Solve mixed smoke reciprocal drag | deferred | P1-40 | — |
| [P1-53](roadmap/tasks/P1-53.json) | Solve mixed smoke thermal exchange | deferred | P1-40 | — |
| [P1-54](roadmap/tasks/P1-54.json) | Schedule mixed smoke pair modes | deferred | P1-52, P1-53 | — |
| [P1-55](roadmap/tasks/P1-55.json) | Qualify mixed smoke pair modes | deferred | P1-54, F5-02, F5-03 | — |
| [P1-56](roadmap/tasks/P1-56.json) | Expose immiscible interface state | done | P1-04 | — |
| [P1-57](roadmap/tasks/P1-57.json) | Expose interface thermal conductance | deferred | P1-15, P1-56 | — |
| [P1-58](roadmap/tasks/P1-58.json) | Release fluid-pair interface storage | done | P1-10 | — |
| [P1-59](roadmap/tasks/P1-59.json) | Complete D3D12 common-domain parity | done | F4-04, F5-10, F7-06 | — |

### P2. Deferred engine integration

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [P2-01](roadmap/tasks/P2-01.json) | Qualify one native engine-handoff slice | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99 | — |
| [P2-02](roadmap/tasks/P2-02.json) | Build Unity minimal SDK integration | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99, P2-01 | — |
| [P2-03](roadmap/tasks/P2-03.json) | Build Unreal minimal SDK integration | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99, P2-01 | — |
| [P2-04](roadmap/tasks/P2-04.json) | Build Godot minimal SDK integration | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99, P2-01 | — |

### P3. Deferred measured optimization

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [P3-01](roadmap/tasks/P3-01.json) | Plan measured platform and shader optimization | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99 | — |
| [P3-02](roadmap/tasks/P3-02.json) | Profile and optimize Android Vulkan stepping | done | F5-07, F5-15, F7-18, F7-99 | — |

### P4. Deferred vendor and standards adoption

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [P4-01](roadmap/tasks/P4-01.json) | Prepare independent vendor-driver handoff | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99 | — |
| [P4-02](roadmap/tasks/P4-02.json) | Prepare standards proposal prerequisites | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99, P4-01 | — |

### Completed categories

All cards, evidence and lease history remain available; expand a category to inspect them.

<details>
<summary>F1. Resource lifetime and capacity — 9 tasks complete</summary>

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [F1-00](roadmap/tasks/F1-00.json) | Storage reclamation and domain capacity baseline | done | — | — |
| [F1-01](roadmap/tasks/F1-01.json) | Plan triangle geometry allocations once | done | — | — |
| [F1-02](roadmap/tasks/F1-02.json) | Plan paint allocations once | done | — | — |
| [F1-03](roadmap/tasks/F1-03.json) | Retire geometry dependencies safely | done | F1-01 | — |
| [F1-04](roadmap/tasks/F1-04.json) | Retire paint dependencies safely | done | F1-02, F1-03 | — |
| [F1-05](roadmap/tasks/F1-05.json) | Rebind emitters and destroy planes | done | — | — |
| [F1-06](roadmap/tasks/F1-06.json) | Qualify allocation failure and pending lifetime | done | F1-03, F1-04, F1-05 | — |
| [F1-07](roadmap/tasks/F1-07.json) | Bound host auxiliary metadata | done | — | — |
| [F1-99](roadmap/tasks/F1-99.json) | Review F1 exit evidence | done | F1-01, F1-02, F1-03, F1-04, F1-05, F1-06, F1-07 | — |

</details>

<details>
<summary>F2. Interaction graph and system coupling — 17 tasks complete</summary>

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [F2-00](roadmap/tasks/F2-00.json) | Simulation-owned graph baseline | done | — | — |
| [F2-01](roadmap/tasks/F2-01.json) | Specify the pair and mode matrix | done | — | — |
| [F2-02](roadmap/tasks/F2-02.json) | Expose pair and material capability queries | done | F2-01 | — |
| [F2-03](roadmap/tasks/F2-03.json) | Validate graph modes against capabilities | done | F2-02 | — |
| [F2-04](roadmap/tasks/F2-04.json) | Qualify graph rebinding across implemented pairs | done | F1-05, F2-03 | — |
| [F2-05](roadmap/tasks/F2-05.json) | Specify equal-material fluid–fluid behavior | done | F2-01 | — |
| [F2-06](roadmap/tasks/F2-06.json) | Build cross-population neighbor indexing | done | F2-05, F1-00 | — |
| [F2-07](roadmap/tasks/F2-07.json) | Exchange fluid pressure and contact forces | done | F2-06 | — |
| [F2-08](roadmap/tasks/F2-08.json) | Exchange fluid viscosity forces | done | F2-07 | — |
| [F2-09](roadmap/tasks/F2-09.json) | Integrate fluid–fluid graph storage and scheduling | done | F2-03, F2-08 | — |
| [F2-10](roadmap/tasks/F2-10.json) | Qualify and advertise equal-material fluid–fluid | done | F2-04, F2-09, F5-02 | — |
| [F2-11](roadmap/tasks/F2-11.json) | Specify different-material and interface modes | done | F2-05 | — |
| [F2-12](roadmap/tasks/F2-12.json) | Specify cloth–cloth system modes | done | F2-01 | — |
| [F2-13](roadmap/tasks/F2-13.json) | Specify soft-body/soft-body system modes | done | F2-01 | — |
| [F2-14](roadmap/tasks/F2-14.json) | Specify rope–rope system modes | done | F2-01 | — |
| [F2-15](roadmap/tasks/F2-15.json) | Specify smoke–smoke system modes | done | F2-01 | — |
| [F2-99](roadmap/tasks/F2-99.json) | Review F2 exit evidence | done | F2-01, F2-02, F2-03, F2-04, F2-05, F2-06, F2-07, F2-08, F2-09, F2-10, F2-11, F2-12, F2-13, F2-14, F2-15 | — |

</details>

<details>
<summary>F3. Specification, ABI and validation — 14 tasks complete</summary>

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [F3-00](roadmap/tasks/F3-00.json) | Version and typed validation baseline | done | — | — |
| [F3-01](roadmap/tasks/F3-01.json) | Finish manual command interception | done | — | — |
| [F3-02](roadmap/tasks/F3-02.json) | Validate command recording state | done | F3-01 | — |
| [F3-03](roadmap/tasks/F3-03.json) | Validate object dependencies and query state | done | F3-01, F1-03, F1-04 | — |
| [F3-04](roadmap/tasks/F3-04.json) | Validate synchronization state | done | F3-02 | — |
| [F3-05](roadmap/tasks/F3-05.json) | Expose structured validation diagnostics | done | F3-01 | — |
| [F3-06](roadmap/tasks/F3-06.json) | Close threading and deferred-error contract gaps | done | F1-06, F3-04 | — |
| [F3-07](roadmap/tasks/F3-07.json) | Test independent compatibility combinations | done | F3-05 | — |
| [F3-08](roadmap/tasks/F3-08.json) | Restore repeated statistics-query validation | done | F3-03 | — |
| [F3-09](roadmap/tasks/F3-09.json) | Qualify portable host concurrency | done | F3-06 | — |
| [F3-10](roadmap/tasks/F3-10.json) | Qualify descriptor snapshot coverage | done | F3-06 | — |
| [F3-11](roadmap/tasks/F3-11.json) | Qualify destruction wait profiles | done | F3-06 | — |
| [F3-12](roadmap/tasks/F3-12.json) | Qualify runtime capacity recovery | done | F3-06 | — |
| [F3-99](roadmap/tasks/F3-99.json) | Review F3 exit evidence | done | F3-01, F3-02, F3-03, F3-04, F3-05, F3-06, F3-07, F3-08, F3-09, F3-10, F3-11, F3-12 | — |

</details>

<details>
<summary>F4. Common resources and engine ownership — 10 tasks complete</summary>

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [F4-00](roadmap/tasks/F4-00.json) | Portable schedule and neutral reflection baseline | done | — | — |
| [F4-01](roadmap/tasks/F4-01.json) | Describe resources and access ranges | done | — | — |
| [F4-02](roadmap/tasks/F4-02.json) | Retain resources through completion | done | F4-01 | — |
| [F4-03](roadmap/tasks/F4-03.json) | Schedule common resource dependencies | done | F4-01, F4-02 | — |
| [F4-04](roadmap/tasks/F4-04.json) | Remove generated-source ABI parsing | done | F4-01 | — |
| [F4-05](roadmap/tasks/F4-05.json) | Specify memory-budget confidence | done | — | — |
| [F4-06](roadmap/tasks/F4-06.json) | Apply conservative workload admission | done | F4-05 | — |
| [F4-07](roadmap/tasks/F4-07.json) | Specify borrowed device and queue handoff | done | F4-02, F3-04 | — |
| [F4-08](roadmap/tasks/F4-08.json) | Qualify engine handoff with a mock adapter | done | F4-03, F4-07 | — |
| [F4-99](roadmap/tasks/F4-99.json) | Review F4 exit evidence | done | F4-01, F4-02, F4-03, F4-04, F4-05, F4-06, F4-07, F4-08 | — |

</details>

<details>
<summary>F6. SDK releases and repository ownership — 18 tasks complete</summary>

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [F6-00](roadmap/tasks/F6-00.json) | Versioned SDK and release baseline | done | — | — |
| [F6-01](roadmap/tasks/F6-01.json) | Qualify Windows SDK installation | done | — | — |
| [F6-02](roadmap/tasks/F6-02.json) | Qualify macOS SDK installation | done | — | — |
| [F6-03](roadmap/tasks/F6-03.json) | Separate optional package exports | done | F5-07 | — |
| [F6-04](roadmap/tasks/F6-04.json) | Record reproducible release inputs | done | — | — |
| [F6-05](roadmap/tasks/F6-05.json) | Regenerate and verify release reference artifacts | done | F3-06 | — |
| [F6-06](roadmap/tasks/F6-06.json) | Reconcile legacy launcher and branding patches | done | — | — |
| [F6-07](roadmap/tasks/F6-07.json) | Reconcile legacy hinge and contact patches | done | — | — |
| [F6-08](roadmap/tasks/F6-08.json) | Retire reconciled old worktrees | done | F6-06, F6-07, F6-09, F6-10, F6-11 | — |
| [F6-09](roadmap/tasks/F6-09.json) | Qualify the retained hinge fixture candidate | done | F6-07 | — |
| [F6-10](roadmap/tasks/F6-10.json) | Productize physics capture and substep replay | done | F6-07 | — |
| [F6-11](roadmap/tasks/F6-11.json) | Adjudicate the retained rigid contact patch | done | F6-09, F6-10 | — |
| [F6-12](roadmap/tasks/F6-12.json) | Keep installed corpus manifest synchronized | done | F5-24, F6-04 | — |
| [F6-13](roadmap/tasks/F6-13.json) | Package the conformance runner source closure | done | F6-12 | — |
| [F6-14](roadmap/tasks/F6-14.json) | Remove workstation paths from installed evidence | done | F6-04, F6-12 | — |
| [F6-15](roadmap/tasks/F6-15.json) | Orient project documentation around users | done | — | — |
| [F6-16](roadmap/tasks/F6-16.json) | Make retained evidence hashes checkout-portable | done | F6-07, F6-09, F6-10 | — |
| [F6-99](roadmap/tasks/F6-99.json) | Review F6 exit evidence | done | F6-01, F6-02, F6-03, F6-04, F6-05, F6-06, F6-07, F6-08, F6-09, F6-10, F6-11, F6-12, F6-13, F6-14 | — |

</details>

<details>
<summary>F7. Shared Gallery application — 24 tasks complete</summary>

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [F7-01](roadmap/tasks/F7-01.json) | Specify one common Gallery runtime | done | — | — |
| [F7-02](roadmap/tasks/F7-02.json) | Unify Gallery catalog and availability | done | F7-01 | — |
| [F7-03](roadmap/tasks/F7-03.json) | Extract common Gallery scene lifecycle | done | F7-01 | — |
| [F7-04](roadmap/tasks/F7-04.json) | Move Gallery controllers into common runtime | done | F7-03 | — |
| [F7-05](roadmap/tasks/F7-05.json) | Unify Gallery clock and workload policy | done | F7-03 | — |
| [F7-06](roadmap/tasks/F7-06.json) | Adopt common runtime in desktop Gallery | done | F7-02, F7-03, F7-04, F7-05 | — |
| [F7-07](roadmap/tasks/F7-07.json) | Make mobile Gallery shells thin adapters | done | F7-02, F7-03, F7-04, F7-05 | — |
| [F7-08](roadmap/tasks/F7-08.json) | Define common Gallery GPU renderer contract | done | F7-01 | — |
| [F7-09](roadmap/tasks/F7-09.json) | Implement native Vulkan Gallery rendering | done | F7-08 | — |
| [F7-10](roadmap/tasks/F7-10.json) | Enable Vulkan cloth Gallery domains | done | F7-03 | — |
| [F7-11](roadmap/tasks/F7-11.json) | Enable Vulkan soft-body Gallery domains | done | F7-03 | — |
| [F7-12](roadmap/tasks/F7-12.json) | Enable Vulkan rope Gallery domains | done | F7-03 | — |
| [F7-13](roadmap/tasks/F7-13.json) | Enable Vulkan smoke Gallery domains | done | F7-03 | — |
| [F7-14](roadmap/tasks/F7-14.json) | Enable Vulkan cross-deformable Gallery couplings | done | F7-10, F7-11, F7-12, F7-13 | — |
| [F7-15](roadmap/tasks/F7-15.json) | Expose capability-driven Android catalog | done | F7-02, F7-07 | — |
| [F7-16](roadmap/tasks/F7-16.json) | Use native Vulkan rendering on Android | done | F7-07, F7-09 | — |
| [F7-17](roadmap/tasks/F7-17.json) | Complete Android Gallery controls and help | done | F7-04, F7-07, F7-15, F7-20, F7-21 | — |
| [F7-18](roadmap/tasks/F7-18.json) | Qualify complete Android Gallery | done | F5-15, F7-05, F7-10, F7-11, F7-12, F7-13, F7-14, F7-15, F7-16, F7-17, F7-22 | — |
| [F7-19](roadmap/tasks/F7-19.json) | Close common Gallery runtime boundary | done | F7-06, F7-08 | — |
| [F7-20](roadmap/tasks/F7-20.json) | Consolidate the common Gallery scene descriptor | done | F7-12, F7-19 | — |
| [F7-21](roadmap/tasks/F7-21.json) | Make Gallery presentation state backend-neutral | done | F7-19, F7-20 | — |
| [F7-22](roadmap/tasks/F7-22.json) | Unify mobile Gallery qualification orchestration | done | F7-07, F7-19, F7-21 | — |
| [F7-23](roadmap/tasks/F7-23.json) | Normalize Gallery renderer adapter ownership | done | F7-08, F7-09, F7-19 | — |
| [F7-99](roadmap/tasks/F7-99.json) | Review shared Gallery exit evidence | done | F7-01, F7-02, F7-03, F7-04, F7-05, F7-06, F7-07, F7-08, F7-09, F7-10, F7-11, F7-12, F7-13, F7-14, F7-15, F7-16, F7-17, F7-18, F7-19, F7-20, F7-21, F7-22, F7-23 | — |

</details>
<!-- END TASK INDEX -->

## Scope and evidence

Foundation exit reviews preserve the original requirements: bounded resource
lifetime (F1), simulation-owned interaction graphs and qualified fluid–fluid
behavior (F2), agreeing specification/runtime/validation (F3), common resource
and engine-ownership contracts (F4), independent physical qualification (F5),
relocatable reproducible SDK releases (F6), and one cross-platform Gallery
runtime with qualified platform adapters (F7). A completed task closes only its
listed acceptance checks, not its entire area or a driver's physical compliance.
F8 adds a separately versioned single-step real-time contract and requires new
kernel, domain, coupling and performance evidence on every registered target.
F1–F7 evidence is historical coverage, not proof that the F8 solver qualifies.

The [qualification index](QUALIFICATION.md) keeps known failures visible.
Windows F5-10 evidence passes the selected 17 CUDA/D3D12 comparisons and the
independent shared-brick penetration gate without changing inputs, tolerances
or historical goldens. Current limits include shared rigid-direct reference
drift pending independent adjudication, remaining driver/domain qualification,
and unqualified engine handoff. Native Vulkan validation uses the isolated,
pinned F5-09 setup. Do not promote
execution to physical correctness to complete a lease.

The detailed [2026-10-16 roadmap snapshot](history/ROADMAP-2026-10-16.md) preserves
prior implementation checkpoints, original exits and platform evidence. It is
historical context, not a second live work queue. The
[pre-command-API roadmap](history/ROADMAP-LEGACY.md) is also historical.
