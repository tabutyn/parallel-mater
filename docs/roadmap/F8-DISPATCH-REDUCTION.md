# F8 dispatch reduction is a completion requirement

Maintainer direction, 2026-10-25: F8 must substantially reduce the GPU work
submission overhead that motivated the redesign. One time advance and one
dynamic BVH build/publication are necessary, but do not alone establish success.
Metal dispatch batching and removal of the remaining inefficient coupling
search/dispatch paths are mandatory F8 work, before candidate freeze and final
qualification. They are not deferred P3 optimization.

## Audit evidence and scope

The source audit at `75584181` found these concrete gaps:

- `src/simulation/metal_geometry_constraints_dispatch.cpp` calls
  `encode_avbd_schedule(rigid_realtime_sweeps, body_count, ..., 1U)`.
  `src/simulation/avbd_schedule.hpp` issues one solve-color command per possible
  body/color per sweep. The fixed rigid solve therefore encodes
  `8 * (body_count + 1) + 6` dispatches: 526 at 64 bodies, 1,038 at 128 and
  3,102 at the 386-body desktop workload, before other physics work. The Slang
  guard returns for unused colors after their commands have already been issued.
  These are reproduced encoder counts, not measured GPU timings.
- `src/simulation/production_realtime_backend.cu.inc` invokes
  `detect_soft_body_cloth_contacts` inside the four coupling sweeps.
  `src/slang/soft_body_cloth.slang` scans every cloth triangle for each soft-body
  node. Other production consumers require the same audit: publishing a common
  BVH does not prove that contact search consumes its candidates.
- The same production backend calls the generic attachment projection/apply
  kernels for individual records. Compatible edges must be batched to avoid
  multiplying host dispatch sequences by graph size.
- `native_realtime_limits` includes the current adapter's dispatch formula in
  its permitted work. This is useful admission/accounting protection, but an
  implementation-derived allowance cannot independently establish low dispatch
  cost or reject the inefficient implementation it describes.

Retain the genuine F8 gains: one fixed time advance, one common dynamic spatial
epoch and bounded algebraic iterations. Liquid's six correction sweeps reuse
the published neighbors. One timestep need not be one kernel launch; batching
must preserve required device dependencies and the existing physical contract.

## Required implementation

**F8-54** replaces the Metal rigid body's worst-case color launch graph with
batched device work. It must eliminate issued empty-color commands, preserve
the eight shared Slang solver sweeps, and avoid a phase-local CPU count readback
or GPU-wide wait. Merely moving thousands of commands into an indirect command
buffer does not establish a dispatch reduction. Any change to the shared solve
ordering still requires independent physical quality checks.

**F8-55** audits and connects the supported coupling searches to the one shared
spatial epoch and batches compatible records. Conservative candidates must
cover motion and all permitted corrections for the full tick. Small randomized
oracle cases compare candidates with brute force; brute force is acceptable in
the independent oracle, not as the unconditional production search per sweep.
Explicit attachments already know their endpoints and require no spatial
search. Grid pressure stencils retain their appropriate grid algorithm.
Capacity exhaustion or out-of-envelope inputs remain explicit errors with
atomic recovery; truncating candidates is not an optimization.

Both tasks preserve existing interactions, workload tiers, numerical limits,
one-way/two-way semantics, one commit, and the single production architecture.
New tasks have their own leases. Completed cards and current owners' scopes
remain intact.

## Independent acceptance and installed evidence

**F8-56** verifies the complete public runtime and shared Gallery on a physical
Mac and iPad. Before dispatch/search optimization, F8-54 and F8-55 must retain
reviewed numerical ceilings for named fixtures, with rationale based on the
intended batched phases and measured baseline. Ceilings must require substantial
absolute and relative reductions and reject the old scaling; setting a ceiling
just below the old count is insufficient. Keep these expectations outside
runtime cost-estimation code. F8-56 integrates them into regression tests and
the final evidence inputs; it cannot raise ceilings to fit the result.

The installed comparisons must:

- use identical scene assets, live/reserved counts and material inputs on each
  named device, preserving its declared workload tier;
- retain the legacy Gallery baseline and pre-fix F8 baseline where applicable,
  plus source, binary, toolchain, driver and deployment identities for every run;
- count actual issued native dispatches, including indirect and no-op commands,
  workgroups, barriers, candidate/narrow-phase work, BVH builds/publications,
  time advances and solver/coupling iterations; reconcile public counters with
  native captures rather than trusting a manually incremented ledger alone;
- report per physics tick, per displayed frame and per simulated second, plus
  simulation/wall-time ratio, so a smaller timestep or overload slow motion
  cannot masquerade as a performance gain;
- report CPU/GPU and frame p50/p95/p99, cold starts, warmup, sustained operation,
  thermal state and independent stability/quality; account separately for
  instrumentation, rendering and screenshots;
- include sparse/dense scaling, the 386-body desktop workload, liquid,
  soft-body/cloth and mixed couplings; negative controls must detect the old
  per-body/per-edge command loops and unconditional exhaustive searches.

Unsupported timers, absent hardware, quality failures and missed ceilings are
recorded as such and cannot pass the gate. No workload reduction, accuracy
relaxation, private benchmark path or platform-local physics substitute is
allowed. Historical baseline binaries are measurement inputs, not a selectable
legacy solver in the released SDK.

## Completion ordering

F8-54 can follow the completed native rigid/mechanical integration. F8-55 follows
F8-45--47 so all currently supported interaction consumers are covered.
F8-56 follows both fixes and F8-51's shared Gallery adoption.

F8-52 depends on all three and freezes their tested source closure, workload
definitions and independent budget artifacts. Protected changes require fresh
affected captures. F8-24--30 and F8-53 still qualify the exact final candidate;
the pre-freeze Apple comparison does not replace those device obligations.
F8-31 publishes the measured reductions with the installed candidate identity,
and F8-99 explicitly reviews them before F8 can close.
