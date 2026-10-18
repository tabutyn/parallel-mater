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

## Leasing work across machines

This is the current work queue. Stable leaf IDs below are the only leaseable
units; F1–F7 are foundation areas, not whole-area reservations. Choose the
smallest ready task matching the machine's capabilities. Each task links to a
canonical JSON card with scope, acceptance checks, dependencies, shared-area
locks, evidence and lease history. Completed `*-00` cards record existing work
and must not be implemented again.

Read [the lease workflow](roadmap/WORKFLOW.md) before changing a task. Push a
**claim-only PR** first. Ownership begins only when that PR merges into `main`;
a pushed branch or open PR alone is not a lease. Then branch from the merged
claim, implement the task and submit a completion PR with evidence. Update task
cards with `tools/roadmap.py`; the index below is generated from those cards.

Leases last at most 24 hours, use UTC and can be renewed. Expired leases remain
reserved until an explicit reclaim/release PR merges. A unique lease token
prevents a previous owner from completing a reclaimed task. Shared-area locks
prevent different task IDs from reserving the same common implementation area.

F1 has the highest priority. Independent ready tasks in F2–F7 may proceed in
parallel when dependencies and locks permit. Platform expansion, engine plugins
and shader performance work remain deferred until the seven `*-99` foundation
review gates close and a planning PR activates the selected follow-up task.

The lease check must be required with up-to-date branches for race-safe
automated merges. Each merge candidate permits one lease operation. Until configured, serialize claim/renew/reclaim/
completion merges against current `main` and rerun the check immediately before
merging. The workflow does not change GitHub branch-protection settings.

<!-- BEGIN TASK INDEX -->
<!-- Generated by tools/roadmap.py render. Edit task cards, not this index. -->

### F1. Resource lifetime and capacity

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

### F2. Interaction graph and system coupling

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

### F3. Specification, ABI and validation

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

### F4. Common resources and engine ownership

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [F4-00](roadmap/tasks/F4-00.json) | Portable schedule and neutral reflection baseline | done | — | — |
| [F4-01](roadmap/tasks/F4-01.json) | Describe resources and access ranges | done | — | — |
| [F4-02](roadmap/tasks/F4-02.json) | Retain resources through completion | done | F4-01 | — |
| [F4-03](roadmap/tasks/F4-03.json) | Schedule common resource dependencies | done | F4-01, F4-02 | — |
| [F4-04](roadmap/tasks/F4-04.json) | Remove generated-source ABI parsing | done | F4-01 | — |
| [F4-05](roadmap/tasks/F4-05.json) | Specify memory-budget confidence | waiting on lock | — | — |
| [F4-06](roadmap/tasks/F4-06.json) | Apply conservative workload admission | waiting on dependencies | F4-05 | — |
| [F4-07](roadmap/tasks/F4-07.json) | Specify borrowed device and queue handoff | done | F4-02, F3-04 | — |
| [F4-08](roadmap/tasks/F4-08.json) | Qualify engine handoff with a mock adapter | leased | F4-03, F4-07 | codex-windows-sdk / windows-gtx1080 / 2026-10-19T17:52:24Z |
| [F4-99](roadmap/tasks/F4-99.json) | Review F4 exit evidence | waiting on dependencies | F4-01, F4-02, F4-03, F4-04, F4-05, F4-06, F4-07, F4-08 | — |

### F5. Independent conformance and quality

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [F5-00](roadmap/tasks/F5-00.json) | Independent headless conformance baseline | done | — | — |
| [F5-01](roadmap/tasks/F5-01.json) | Map advertised modes to conformance cases | done | F2-02 | — |
| [F5-02](roadmap/tasks/F5-02.json) | Add vendor-neutral conservation and stability cases | done | — | — |
| [F5-03](roadmap/tasks/F5-03.json) | Add capacity and three-family corpus cases | done | F1-06, F2-04 | — |
| [F5-04](roadmap/tasks/F5-04.json) | Isolate CPU rigid-direct divergence | done | — | — |
| [F5-05](roadmap/tasks/F5-05.json) | Isolate shared brick penetration failure | done | — | — |
| [F5-06](roadmap/tasks/F5-06.json) | Classify CUDA/D3D12 comparison failures | done | — | — |
| [F5-07](roadmap/tasks/F5-07.json) | Gate performance on physical quality | done | F5-01, F5-02 | — |
| [F5-08](roadmap/tasks/F5-08.json) | Publish per-driver evidence indexes | done | F5-01 | — |
| [F5-09](roadmap/tasks/F5-09.json) | Isolate native Vulkan validation crash | done | — | — |
| [F5-10](roadmap/tasks/F5-10.json) | Correct shared contacts and CUDA/D3D12 drift | done | — | — |
| [F5-11](roadmap/tasks/F5-11.json) | Publish current Windows driver evidence | done | F5-01, F5-02, F5-03, F5-05, F5-06, F5-08, F5-10 | — |
| [F5-12](roadmap/tasks/F5-12.json) | Publish current macOS Metal driver evidence | done | F5-01, F5-02, F5-03, F5-05, F5-06, F5-08, F5-10, F5-11 | — |
| [F5-13](roadmap/tasks/F5-13.json) | Expose capability qualification on iPadOS | done | F5-01, F5-08 | — |
| [F5-14](roadmap/tasks/F5-14.json) | Publish current iPadOS Metal driver evidence | done | F5-02, F5-03, F5-05, F5-06, F5-10, F5-11, F5-13 | — |
| [F5-15](roadmap/tasks/F5-15.json) | Publish current Android Vulkan driver evidence | ready | F5-01, F5-02, F5-03, F5-05, F5-06, F5-08, F5-09, F5-10, F5-11 | — |
| [F5-16](roadmap/tasks/F5-16.json) | Publish current Linux driver evidence | ready | F5-01, F5-02, F5-03, F5-04, F5-05, F5-06, F5-08, F5-09, F5-10, F5-11 | — |
| [F5-17](roadmap/tasks/F5-17.json) | Adjudicate rigid-direct reference drift | done | F5-04, F5-10 | — |
| [F5-18](roadmap/tasks/F5-18.json) | Stabilize rigid-direct planar symmetry | done | F5-17 | — |
| [F5-99](roadmap/tasks/F5-99.json) | Review F5 exit evidence | waiting on dependencies | F5-01, F5-02, F5-03, F5-04, F5-05, F5-06, F5-07, F5-08, F5-09, F5-10, F5-11, F5-12, F5-13, F5-14, F5-15, F5-16, F5-17, F5-18 | — |

### F6. SDK releases and repository ownership

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
| [F6-08](roadmap/tasks/F6-08.json) | Retire reconciled old worktrees | waiting on dependencies | F6-06, F6-07, F6-09, F6-10, F6-11 | — |
| [F6-09](roadmap/tasks/F6-09.json) | Qualify the retained hinge fixture candidate | leased | F6-07 | codex-linux-hinge / thomas-laptop / 2026-10-19T17:38:57Z |
| [F6-10](roadmap/tasks/F6-10.json) | Productize physics capture and substep replay | ready | F6-07 | — |
| [F6-11](roadmap/tasks/F6-11.json) | Adjudicate the retained rigid contact patch | waiting on dependencies | F6-09, F6-10 | — |
| [F6-99](roadmap/tasks/F6-99.json) | Review F6 exit evidence | waiting on dependencies | F6-01, F6-02, F6-03, F6-04, F6-05, F6-06, F6-07, F6-08, F6-09, F6-10, F6-11 | — |

### F7. Shared Gallery application

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
| [F7-18](roadmap/tasks/F7-18.json) | Qualify complete Android Gallery | waiting on dependencies | F5-15, F7-05, F7-10, F7-11, F7-12, F7-13, F7-14, F7-15, F7-16, F7-17, F7-22 | — |
| [F7-19](roadmap/tasks/F7-19.json) | Close common Gallery runtime boundary | done | F7-06, F7-08 | — |
| [F7-20](roadmap/tasks/F7-20.json) | Consolidate the common Gallery scene descriptor | done | F7-12, F7-19 | — |
| [F7-21](roadmap/tasks/F7-21.json) | Make Gallery presentation state backend-neutral | done | F7-19, F7-20 | — |
| [F7-22](roadmap/tasks/F7-22.json) | Unify mobile Gallery qualification orchestration | done | F7-07, F7-19, F7-21 | — |
| [F7-23](roadmap/tasks/F7-23.json) | Normalize Gallery renderer adapter ownership | done | F7-08, F7-09, F7-19 | — |
| [F7-99](roadmap/tasks/F7-99.json) | Review shared Gallery exit evidence | waiting on dependencies | F7-01, F7-02, F7-03, F7-04, F7-05, F7-06, F7-07, F7-08, F7-09, F7-10, F7-11, F7-12, F7-13, F7-14, F7-15, F7-16, F7-17, F7-18, F7-19, F7-20, F7-21, F7-22, F7-23 | — |

### P1. Deferred domain and interaction coverage

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [P1-01](roadmap/tasks/P1-01.json) | Plan remaining driver/domain qualification | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99 | — |
| [P1-02](roadmap/tasks/P1-02.json) | Plan remaining same-family operators | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99 | — |
| [P1-03](roadmap/tasks/P1-03.json) | Plan multi-material fluid operators | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99 | — |
| [P1-04](roadmap/tasks/P1-04.json) | Expose multi-material fluid pair state | deferred | P1-03 | — |
| [P1-05](roadmap/tasks/P1-05.json) | Build heterogeneous fluid pair sampling | deferred | P1-04 | — |
| [P1-06](roadmap/tasks/P1-06.json) | Exchange miscible pressure and contact | deferred | P1-05 | — |
| [P1-07](roadmap/tasks/P1-07.json) | Exchange miscible viscosity | deferred | P1-06 | — |
| [P1-08](roadmap/tasks/P1-08.json) | Schedule miscible fluid pair graphs | deferred | P1-07 | — |
| [P1-09](roadmap/tasks/P1-09.json) | Qualify and advertise miscible fluids | deferred | P1-08, F5-02, F5-03 | — |
| [P1-10](roadmap/tasks/P1-10.json) | Reconstruct immiscible fluid interfaces | deferred | P1-05 | — |
| [P1-11](roadmap/tasks/P1-11.json) | Apply immiscible pressure and capillarity | deferred | P1-10 | — |
| [P1-12](roadmap/tasks/P1-12.json) | Apply immiscible interfacial viscosity | deferred | P1-11 | — |
| [P1-13](roadmap/tasks/P1-13.json) | Schedule immiscible fluid pair graphs | deferred | P1-12 | — |
| [P1-14](roadmap/tasks/P1-14.json) | Qualify and advertise immiscible fluids | deferred | P1-13, F5-02, F5-03 | — |
| [P1-15](roadmap/tasks/P1-15.json) | Add fluid caloric material state | deferred | P1-04 | — |
| [P1-16](roadmap/tasks/P1-16.json) | Exchange conservative fluid heat | deferred | P1-05, P1-15 | — |
| [P1-17](roadmap/tasks/P1-17.json) | Schedule fluid thermal pair graphs | deferred | P1-08, P1-13, P1-16 | — |
| [P1-18](roadmap/tasks/P1-18.json) | Qualify and advertise fluid heat exchange | deferred | P1-17, F5-02, F5-03 | — |
| [P1-19](roadmap/tasks/P1-19.json) | Expose cloth contact scopes and controls | deferred | P1-02 | — |
| [P1-20](roadmap/tasks/P1-20.json) | Build bounded cloth contact generation | deferred | P1-19 | — |
| [P1-21](roadmap/tasks/P1-21.json) | Solve cloth self-contact and friction | deferred | P1-20 | — |
| [P1-22](roadmap/tasks/P1-22.json) | Solve contact between cloth systems | deferred | P1-20 | — |
| [P1-23](roadmap/tasks/P1-23.json) | Schedule cloth contact scopes | deferred | P1-21, P1-22 | — |
| [P1-24](roadmap/tasks/P1-24.json) | Qualify and advertise cloth contact | deferred | P1-23, F5-02, F5-03 | — |
| [P1-25](roadmap/tasks/P1-25.json) | Expose soft-body contact scopes | deferred | P1-02 | — |
| [P1-26](roadmap/tasks/P1-26.json) | Build bounded soft-surface contacts | deferred | P1-25 | — |
| [P1-27](roadmap/tasks/P1-27.json) | Solve soft-body self-contact | deferred | P1-26 | — |
| [P1-28](roadmap/tasks/P1-28.json) | Solve contact between soft bodies | deferred | P1-26 | — |
| [P1-29](roadmap/tasks/P1-29.json) | Schedule soft-body contact scopes | deferred | P1-27, P1-28 | — |
| [P1-30](roadmap/tasks/P1-30.json) | Qualify and advertise soft-body contact | deferred | P1-29, F5-02, F5-03 | — |
| [P1-31](roadmap/tasks/P1-31.json) | Audit sampled-node rope self-contact | deferred | P1-02, F5-02 | — |
| [P1-32](roadmap/tasks/P1-32.json) | Expose rope capsule pairs and attachments | deferred | P1-02 | — |
| [P1-33](roadmap/tasks/P1-33.json) | Build bounded rope segment contacts | deferred | P1-32 | — |
| [P1-34](roadmap/tasks/P1-34.json) | Solve rope capsule self-contact | deferred | P1-33 | — |
| [P1-35](roadmap/tasks/P1-35.json) | Solve contact between rope systems | deferred | P1-33 | — |
| [P1-36](roadmap/tasks/P1-36.json) | Solve rope endpoint attachments | deferred | P1-32 | — |
| [P1-37](roadmap/tasks/P1-37.json) | Schedule rope pair modes | deferred | P1-34, P1-35, P1-36 | — |
| [P1-38](roadmap/tasks/P1-38.json) | Qualify and advertise rope pair modes | deferred | P1-37, F5-02, F5-03 | — |
| [P1-39](roadmap/tasks/P1-39.json) | Expose smoke pair profiles and edges | deferred | P1-02 | — |
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

### P4. Deferred vendor and standards adoption

| Task | Work unit | State | Depends on | Owner / machine / expiry (UTC) |
|---|---|---|---|---|
| [P4-01](roadmap/tasks/P4-01.json) | Prepare independent vendor-driver handoff | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99 | — |
| [P4-02](roadmap/tasks/P4-02.json) | Prepare standards proposal prerequisites | deferred | F1-99, F2-99, F3-99, F4-99, F5-99, F6-99, F7-99, P4-01 | — |
<!-- END TASK INDEX -->

## Scope and evidence

Foundation exit reviews preserve the original requirements: bounded resource
lifetime (F1), simulation-owned interaction graphs and qualified fluid–fluid
behavior (F2), agreeing specification/runtime/validation (F3), common resource
and engine-ownership contracts (F4), independent physical qualification (F5),
relocatable reproducible SDK releases (F6), and one cross-platform Gallery
runtime with qualified platform adapters (F7). A completed task closes only its
listed acceptance checks, not its entire area or a driver's physical compliance.

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
