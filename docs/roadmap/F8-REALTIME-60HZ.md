# F8: 60 Hz single-step simulation and shared spatial indexing

Status: planning contract; no F8 implementation or qualification claimed.
Task cards in `tasks/F8-*.json` remain the canonical queue. This document records
the user-directed architecture change of 2026-10-21, not a second task tracker.

## Target contract

1. Advance simulation exactly once by `dt = 1/60 s` per physics tick. No eight
   global substeps, hidden per-domain smaller timesteps, adaptive time subdivision,
   or catch-up bursts in the new real-time profile. A render callback executes
   at most one physics tick. At 120/144 Hz, render between completed states;
   rendering does not increase the physics rate. At 30 Hz under load, advance
   30 ticks of 1/60 s per wall-clock second: half-speed simulation, not `dt=1/30`.
2. Update the shared dynamic bounding-volume hierarchy once per tick, then reuse
   its snapshot for all domains and couplings through the solve. One update
   phase may contain several kernels; it must not hide repeated builds for each
   solver iteration, fluid population, or coupling edge. Immutable mesh-local
   BVHs stay cached. Build versus refit is a measured implementation choice, not
   permission for a second dynamic update in the same tick.
3. Prefer stable, visually useful, bounded behavior at this timestep over the old
   level of numerical accuracy. Permit damping, compliance, bounded correction
   and other approximations only with explicit material semantics, limits and
   separately versioned quality tests. Do not choose a solver merely by name
   or claim unconditional stability for arbitrary inputs.
4. Bounded algebraic pressure/contact/constraint iterations within that *same*
   timestep are allowed only with an approved fixed budget. They must not advance
   time again or rebuild the index. Count them, their dispatches and barriers.
   Test that the iteration cap actually meets the new quality envelope; one
   physics step does not mean one kernel launch.
5. Shared Slang equations and spatial primitives serve CUDA, Vulkan, D3D12 and
   Metal. Native adapters own resource/command/synchronization services, not
   different Android physics. CPU reference/oracle evidence stays independent.
6. Physics must leave room for rendering/input in the 16.67 ms frame budget on
   nominated 60 Hz reference workloads. F8-00 locks per-device workload tiers,
   physics allocations and p95 limits before solver changes; F8-01 confirms the
   actual device inventory. Overload slow-motion is correct fallback behavior,
   not a way to pass the performance gate. Do not reduce scene counts silently.

F8 does not add new advertised interaction modes. In particular, sharing an
index does not establish pressure, viscosity or any other coupling capability.
Existing modes must retain their semantics or explicitly negotiate a new profile.

## Current architecture is not yet shared

Source inventory at main `4f9c43c` (reconfirm in F8-05):

| Consumer | Current path | F8 direction |
|---|---|---|
| Rigid bodies | World AABBs, `count * count` pair flags/compaction, then mesh-local BVH collision queries | Shared dynamic BVH candidates; retain immutable mesh-local trees |
| Liquid particles | Per-population sorted cell keys, plus separate cross-population indices, rebuilt inside fluid iterations | Shared particle leaves and radius queries, identities independent of storage order |
| Deformable surfaces/couplings | Domain-specific traversal and repeated surface BVH refits | Common bounds/build/refit/traversal primitives and one published tick snapshot |
| Smoke fields | Structured-grid sampling/advection/pressure kernels | Keep grid field operators; share collision/obstacle indexing and tick schedule |

Relevant sources: `src/simulation/reference_schedule.hpp`,
`src/slang/geometry.slang` (`pm_broad_phase_rigid_pairs`),
`src/simulation/primitives.cu`, `src/slang/primitives/radix_sort.slang`,
`src/slang/fluid.slang`, and `src/slang/smoke_grid.slang`.
Do not force a BVH into a regular-grid pressure stencil; common collision lookup
does not require identical domain equations or identical query kernels.

The observed 1,163 fluid dispatches were for a complete old tick, not a single
particle. Its eight global substeps times two fluid iterations produce sixteen
neighbor rebuilds. The current 64-bit radix sort uses sixteen four-bit rounds,
each with histogram/prefix/scatter dispatches: 48 sort dispatches plus key
generation per rebuild, or 784 dispatches before the other fluid work. This is
the structural motivation, not a validated F8 speedup or a literal BVH-build cost.

## Tick design and invariants

`tick-boundary mutations -> predict/conservative bounds -> one index publication
-> candidates + bounded domain/coupling solve -> commit -> render snapshots`

- Gather emitters, removals, topology/endpoint edits and kinematic targets at a
  documented tick boundary. No extra source-specific neighbor rebuild followed
  by another post-emission rebuild. F8-05 specifies admission/overlap behavior
  for new particles and when removals become visible; test those semantics.
- Spatial keys/leaves retain generation-tagged identities across populations,
  compaction and slot reuse. One managed hierarchy may retain static geometry
  subtrees, but per-edge/per-domain trees must not multiply dynamic rebuilds.
- Swept bounds cover translation, rotation, support radius and the maximum
  permitted solver correction through the tick. Prove candidate completeness
  against brute force on small randomized scenes. High speeds, impulses,
  teleports and corrections escaping that envelope require explicit bounded
  handling (for example rejecting an unsupported tick before commit), not
  undetected missed contacts or a hidden rebuild/substep fallback.
- Collision margins, support radii, maximum speeds/corrections, mass ratios,
  stiffness ranges and capacity limits belong in the profile contract. Bounds
  that are so broad that everything pairs with everything fail scaling budgets.
- Process active elements rather than reserved capacity where practical, with
  GPU-resident counts/indirect work. Empty and sparse large-capacity scenes are
  mandatory tests. Do not introduce a CPU count readback on every phase.
- Reuse buffers, contacts and warm starts with generation-aware invalidation.
  Fixed small solve budgets must cover cold starts, spawn/reset and impacts too.
- One coherent state epoch for two-way coupling, with accounted equal/opposite
  exchanges and documented splitting error; do not integrate endpoints twice.
- Synchronize actual producer/consumer hazards. GPU barriers can remain necessary;
  remove routine host waits/readbacks between phases. Asynchronous telemetry must
  not insert the very stalls it measures. Explicit application waits are distinct.
- Interpolation uses previous/current completed poses and domain outputs, including
  orientation and changing topology. Specify its latency, spawn/removal/reset
  behavior and input sampling. No extrapolation or stale-generation interpolation.
  Reset/clamp presentation time after stalls/backgrounding; never repay a backlog.

## Quality and measurement gates

F8-00 defines a named/versioned real-time profile before implementation. It must
set quantitative limits for contact penetration/tunneling, resting stability,
joint/stretch/volume error, liquid compression/free-surface behavior, smoke
divergence, coupling imbalance and long-run finite-state/energy behavior over
explicit input ranges. Lower accuracy is allowed and reported, not disguised as
legacy conformance. Keep old-profile tests/results separate. Capability/version
negotiation and unsupported behavior must be explicit if public behavior changes.
This does not require shipping two solvers forever: an explicitly versioned
replacement may retire the old implementation. Preserve its historical evidence
and reject incompatible requests clearly instead of silently reinterpreting them.

F8-03 implements independent small-scene oracles, fixed workloads and adversarial
cases: cold/warm starts, long runs, high mass ratios/stiffness, fast rotation and
impact, thin containment, all registered coupling modes, three-plus-family scenes,
spawn/outflow, resource replacement, capacity failure and recovery. Include
negative controls that actually violate bounds/budgets. Test domains individually
before integrated mixed scenes. A visually stable demo alone cannot pass.

Capture CPU encode/submit/wait time, GPU index/candidate/domain/coupling/commit
times, dispatches, workgroups, barriers, queue submissions, readbacks, bytes moved,
index updates and numerical iteration counts. Count actual emitted work, not only
hand-maintained estimates. Report per-tick and per-render-frame units separately,
physics-to-wall-clock rate and presentation latency. Record warmup, cold-start,
p50/p95/p99, workload/capacity, sustained thermals/power state and instrumentation
overhead. Timer-unavailable is explicit; never substitute CPU time as GPU time.

Android provenance is a prerequisite: the old profiling script's `--apk` path
recorded a local APK hash without installing or verifying the installed package.
Local O3/rank-one/cooperative experiment names therefore do not prove those
binaries ran. F8-02 must bind build revision/dirty patch, artifact hash, installed
package identity and run nonce together, reject stale deployment, and quarantine
unverified comparisons. Preserve original reports; remeasure instead of inferring
which uninstalled optimization did or did not work.

## Cross-machine work and exact coverage

| Lane | Initial responsibility | Final evidence |
|---|---|---|
| Any host | F8-00 contract, F8-01 inventory, F8-05 design; independent planning leaves | Reviewed contracts and exact obligations, not performance claims |
| Shared-runtime/GPU development | F8-03–21 plus F8-32, dependency-ordered | Shared kernels, profile API, fixed-step tests, scheduling and dispatch budgets |
| Linux CPU/CUDA/Vulkan | F8-23/24/25 | Independent oracle plus physical GPU runs for each registered device |
| Windows CUDA/D3D12 | F8-26/27 | Same candidate/workloads on both drivers; honor P1-59 ownership |
| macOS/iOS Metal | F8-28/29 | macOS and physical iOS/iPadOS runs, not simulator substitution |
| Android Vulkan | F8-02 and F8-30 | Verified installed build; Pixel 7 first plus every other registered target |
| Integration/release | F8-22, F8-31, F8-99 | Frozen candidate matrix, installed SDK/Gallery and honest exit review |

F8-01 records actual OS/device/GPU/driver/toolchain owners; this table does not
claim all those machines are currently available. Cover every registered physical
target, not every consumer GPU ever made. Add a bounded qualification leaf for
each additional device/backend tuple before freezing the candidate. Unavailable
hardware is missing evidence, not a skip that becomes a pass. All kernel variants
from the build manifest must map to executed tests on each applicable target,
including spatial primitives, paint/auxiliary kernels and every advertised mode.
An explicitly unsupported mode is not qualified; compile-only is not execution.

F8-22 freezes one source revision, profile version, kernel-manifest hash, workloads
and tolerance/budget manifest. Qualification leaves are test/report work, not
permission for private platform physics. A failure gets a separately leased fix.
Any later physics/kernel/schedule/profile change invalidates affected candidate
rows. Freeze a new candidate and rerun affected rows (new follow-up cards if an
old qualification card already completed). F8-99 audits the *current* candidate,
not a collection of green reports from different revisions. Artifacts must be
tracked or attached at durable reviewable locations, not only on an owner's disk.

## Dependencies, locks and migration

Start with F8-00, F8-01 and F8-05; then the independent harness/index/domain lanes
open as prerequisites merge. Claim one leaf, implement/test, merge completion,
then claim the next. The phase overview here never reserves a whole category.
Common scheduler/API/spatial locks intentionally serialize overlapping edits;
domain equation work may run on different machines when its shared changes have
landed. Replan an overlarge leaf before claiming rather than widening its lease.

P3-02's eight-substep/30 Hz optimization was released in PR #437 and is deferred
as superseded by F8. Its unfinished branch `work/P3-02/thomas-laptop` and dirty
experiments remain preserved; it was not completed or physically qualified.
Do not reactivate it to perform the single-step redesign. F8-02/F8-30 retain
the Android profiling lock, as do the clock and rigid-equation changes, so these
tasks cannot race deployment or overlapping experiments. P1-59 D3D12 parity
retains its existing owner/contract. The completed F5-99 review remains evidence
for the legacy implementation; new equations need new qualification.

F8-99 closes only when all dependencies, device rows, quantitative quality and
frame budgets pass on the final candidate, the once-per-tick invariants are
enforced, installed clients interpolate/slow correctly, and remaining unsupported
features are documented. Failed or missing evidence keeps the gate open.
