# F8-05: one shared spatial-index epoch

Design for [F8](F8-REALTIME-60HZ.md), based on main
`f8f7c52c8db9c0ed94485d9e34da547b3f0ebb16`. This is a migration contract and
executable mathematical model, **not an implemented GPU BVH or a performance /
physical-qualification claim**. F8-00 owns numerical limits; F8-01 owns the device
matrix. Limits below are symbolic until those tasks approve concrete values.

## 1. Source inventory and migration

Paths are relative to the repository root. Symbols identify the inspected paths
without depending on line numbers that move as other machines merge changes.

| Consumer | Current source/path | Shared replacement / owner |
|---|---|---|
| Rigid body pairs | `src/slang/geometry.slang::pm_broad_phase_rigid_pairs`: `count * count` flags; `reference_schedule.hpp` compacts them every substep | Body-proxy overlap queries, canonical candidate pairs; F8-08 |
| Immutable triangle meshes | `src/simulation/runtime_deformables.cu.inc`: host centroid-split BVH at mesh creation; `geometry_storage.hpp::TriangleMeshResource` owns tree/leaves | Keep cached local trees; one world proxy per body, never duplicate triangles per instance; F8-07/08 |
| Liquid self / distinct populations | `src/slang/fluid.slang::fluid_emit_cells`, `fluid_emit_pair_cells`; `reference_schedule.hpp` sorts inside `substeps * solver_iterations`; `resource_plan.hpp::FluidPairNeighborIndexPlan` owns per-edge indices | Particle leaves from all populations in one tree; owner/material filters replace pair-private indices; F8-09 |
| Liquid sources | `fluid_source_vacancies`, `fluid_source_emit`; scheduler builds an extra pre-emission sorted index and reads counts | Provisional birth admission using the sole private build, then publish once; F8-09 |
| Liquid / soft surface; rope / soft surface | `fluid_soft_body.slang::fluid_soft_refit`; scheduler calls `refit_fluid_soft_surface` in coupling loops | One conservative triangle-leaf envelope for the whole tick, shared by every edge; F8-10 |
| Liquid / cloth | `fluid_cloth.slang::fluid_cloth_containment_forces`, `fluid_project_inside_cloth` | Nearest-surface and containment queries; retain orientation/interior tests, not just radius neighbors; F8-09/10 |
| Liquid / rope | `fluid_rope.slang::fluid_rope_contacts` scans rope segments for each particle | Capsule/segment candidates, same nearest-contact and reaction semantics; F8-09/10 |
| Rope self / geometry | `rope.slang` samples self nodes with a nested node loop; `rope_contact` traverses rigid mesh trees | Point leaves for existing sampled self-contact, segment leaves only for modes already using segments; body proxies plus cached mesh trees; F8-10/15 |
| Particle smoke | `geometry_smoke.slang::smoke_emit_cells`, `smoke_density_pressure`, `smoke_pair_forces`; scheduler sorts before coupling, dirty phase transfer and pressure | Point/radius queries on the shared tree; no separate smoke sorting epoch; F8-10/16 |
| Eulerian smoke | `smoke_grid.slang::smoke_grid_raster_rigid`, `smoke_grid_raster_deformable`, pressure/advection stencils | Use shared obstacle candidates where useful; retain structured field grids and their numerical operators; F8-10/16 |
| Scene hit-box queries | `queries.slang::query_rigid_hit_box_kernel`, `query_particle_hit_box_kernel` scan current bodies/particles | Optional consumer of the committed epoch; no query-triggered dynamic rebuild; visibility/version semantics belong to F8-32 |
| Shared primitives | `src/simulation/primitives.cu::radix_sort_pairs_u64_u32`: sixteen histogram/prefix/scatter rounds and final copies | Build-policy benchmark in F8-06; do not blindly transplant this 48-dispatch sort into the new tree |

Same query infrastructure does not mean identical narrow phase or force laws.
No new self-contact, fluid coupling, pressure or viscosity capability is enabled
by this migration. Grid pressure stencils and fixed topology constraint adjacency
are not spatial indices to replace with a BVH.

## 2. One world hierarchy, cached mesh geometry

Use one binary world BVH containing typed leaves: rigid-body proxies, liquid and
particle-smoke points, deformable triangles, and rope points/segments required
by existing modes. Only consumers with an enabled spatial interaction register
leaves. A cloth surface used by three edges is registered once, not three times.
Static body proxies can live in this world tree; their immutable mesh-local
trees remain cached. No separately rebuilt tree per population, pair or domain.

F8-06 compares a shared linear-BVH build (Morton ordering with full identity
tie-breaking) against other bounded policies on the actual workload tiers.
Morton codes organize leaves; they are **not** identities or correctness bounds.
Duplicate codes, coincident points and zero scene extents must be valid. Select
key width/pass count by measured cost and precision, not the old 64-bit layout.

F8-07 initially permits one full build per tick. A measured refit alternative
may replace that build when topology and leaf identity set are unchanged; never
refit and then decide to rebuild in the same tick. Choose the policy before the
update using previous quality metrics and staged mutations. Empty scenes publish
an empty epoch with zero build dispatches. Static mesh creation is outside the
tick build count, but its work must not be hidden on the per-frame hot path.

## 3. Identity, ownership and resource layout

Conceptual private records (not new public ABI allocations):

- `SpatialIdentity = (family, owner_slot, owner_generation, topology_revision,
  primitive_kind, primitive_slot, primitive_generation)`. It distinguishes equal
  particle indices in different populations and equal slots after reuse.
- `Leaf = (identity, conservative_aabb, payload_ref, query_mask, admitted)`.
  Payload references resolve through the epoch's immutable identity-to-storage
  mapping. Dense storage offsets and Morton ranks never become persistent IDs.
- `Epoch = (tick, profile_version, graph_revision, resource_revision,
  root, leaf_count, node_count, counts, mappings, status)` plus a retirement fence.
  Capture material/support/filter data or pin their exact resource versions.
- `Query = (epoch, query_kind, geometry, owner/edge filter, capacity)`.
  Results contain identities and required primitive data, not unversioned offsets.

An epoch pins its nodes, mappings, resources and graph until all submitted
consumers retire. Slot reuse, compaction or material replacement cannot alter an
in-flight view. Overflow of any generation/revision counter requires explicit
retirement/error handling; wrapping into a live old identity is forbidden.
Warm-start keys include both identities, profile and relevant material/graph
revisions; invalidate on teleport, topology/material edits and endpoint changes.

Allocate bounded storage at resource/capacity changes, not each tick. Compute
capacity sums and byte sizes in checked wide arithmetic; a binary tree requires
at most `2*N - 1` nodes for `N > 0` leaves. Include provisional births and every
required primitive representation in `N`; do not equate particles with all leaves.
Capacity reserves memory; GPU active counts determine ordinary work sizes.
Traversal scratch, candidate buffers and adjacency/filter work have explicit
caps. Never silently truncate candidates or skip an overflowing traversal stack.

## 4. Full-tick conservative bounds

For each point/vertex, require a predictor path contained in the componentwise
box between `x0` and `xp`, expanded by a declared path-deviation bound `D`.
The **total** solver displacement from that path is bounded componentwise by
`C` for the entire tick, not `C` per iteration. All coupling corrections share
that budget. Contact radius/margin is `r`, and outward floating-point padding
is `epsilon`:

`Bpoint = box(x0, xp) expanded by (D + C + r + epsilon)`.

`D` must include any acceleration/curved-motion departure from the chosen path;
it is zero only when that path model actually applies. Testing current endpoints
alone cannot prove this precondition. Solvers enforce the approved path/correction
envelope before publishing output; numerical limits come from F8-00.
Check every interaction-evaluation state, not only the final state: leaving an
envelope and returning before commit must still invalidate the tick.

- Segment/capsule: union endpoint boxes, with the segment radius included.
- Triangle: union its three vertex boxes, with contact thickness included.
  Convex combinations of enclosed vertices remain enclosed. For skinned vertices
  with signed binding weights, propagate interval bounds using each weight's
  sign (and absolute weights for error budgets); a convex-hull shortcut is invalid.
- Rigid body: a simple safe baseline is the translation-path box expanded by
  `R + D + C + margin + epsilon`, where `R` encloses every mesh point about the
  **rotation origin**. If a cached sphere has an offset center, use at least
  `length(center) + radius`. This encloses all orientations, including full turns.
- A tighter rigid option starts with the initial oriented AABB translated along
  the translation-path box and expands by `min(R * Theta, 2 * R)` plus the other
  bounds. `Theta` bounds total angular travel including solver corrections;
  shortest-angle difference between endpoint quaternions is not sufficient.
- Parent node bounds are outward-rounded unions of children. Quantized bounds
  must round outward too. Broadphase overlap is inclusive at touching boundaries.

For a neighbor query, intersect the query point's full-tick center envelope
expanded by the applicable support radius with target center envelopes. For
pair supports that differ, use the actual operator's conservative maximum (for
example `max(h_a, h_b)` where that is the existing rule), then apply its exact
distance/material filter at the current solver state. Do not shrink bounds
because an interaction's final position happens to be far away.

Completeness argument: an actual point/primitive contact lies inside both
conservative expanded envelopes; their AABBs therefore overlap. Parent unions
preserve that overlap to the root, so exhaustive non-truncating traversal reaches
the candidate. This proves **candidate inclusion only**, not narrow-phase CCD,
correct force integration, or a measured speedup. Those remain domain gates.

Large bounds can degenerate into near-all-pairs work. Report leaf/node visits,
candidate/live ratios and bytes alongside launch counts. A correctness-preserving
but excessively loose tree fails the performance budget; it does not justify
secret extra updates or unreported pruning.

## 5. Tick transaction, births and phase changes

1. Freeze graph/material revisions and stage external mutations: removals,
   topology changes, teleport/kinematic targets and a bounded set of source
   proposals. Old in-flight epochs keep their original resources.
2. Predict from that staged boundary state and emit conservative leaves for all
   surviving primitives **and provisional births**. Unconditional removals are
   absent; conditional phase-transfer removals retain fallback source leaves.
3. Perform the sole private world-tree build/refit. Resolve source vacancies
   against this tree at the declared birth state, before public epoch publication.
4. Admit a proposal only if it conflicts with no retained live primitive and no
   lower-identity conflicting proposal. Compare against all lower proposals,
   even ones themselves rejected: this deterministic one-pass rule can underfill,
   but avoids an iterative greedy admission loop. F8-00 must approve the changed
   spawn semantics. Retain rejected provisional leaves as inactive tombstones
   until next tick; their bounds are harmless false positives. Do not refit them
   away. Accepted particle counts and ID allocation are GPU-resident.
5. Publish the epoch once with its immutable admission mask/mappings. Execute
   bounded domain/coupling iterations against it; evaluate current geometry in
   narrow phase. Published leaves, support limits and identities never change.
6. Validate envelopes and all capacity/work errors; commit the state once.
   Outputs/events refer to the same committed epoch. Retire old resources only
   after their last consumer fence.

This is one boundary-admission transaction ending in one publication; using the
private completed tree for admission is **not** a second index build. Proposal
counts and inactive-leaf overhead are budgeted, including an all-rejected burst.
Rejected births do not overwrite live slots or emit successful-spawn events.

Outflow/phase-transfer requests discovered during a solve apply at the **next**
boundary using committed source state. The source remains physically present
until atomic removal/admission succeeds; do not remove mass now and create its
destination later. Destination-capacity failure leaves the source intact and
reports the deferred/failed transfer. Cross-family timing is an explicit new
profile semantic for F8-00/F8-18 review, not a claim of legacy equivalence.
If admission can still fail after the build, reserve both the source's full-tick
fallback envelope and the destination's full-tick envelope in that one build.
An atomic transfer-group admission selects source OR destination, never both or
neither. Exclude the group's own source from its destination vacancy test; other
conditional sources remain conservative obstacles until admission completes.
Count this possible underfill and both reservations in proposal/capacity budgets.
For multi-particle transfers, reserve and select the entire group atomically.
Do not remove a source before a later query could reject its destination.
No new active leaf may appear after publication. If immediate same-tick phase
births are later required, replan their conservative pre-reservation contract;
do not add a mid-solve rebuild as a fallback.

## 6. Queries, filtering and deterministic work

Shared Slang helpers provide inclusive AABB overlap, point-radius neighbors,
swept primitive candidates, and branch-and-bound nearest-surface traversal.
Narrow phase supplies actual geometry and mode-specific support rules.
Containment can additionally need sidedness, closed-solid or winding tests;
nearest triangle alone does not prove a point is inside. Keep these operations.

Capture existing graph/self-contact/adjacency filters in the epoch. Filters may
remove only semantically forbidden pairs, not physically plausible candidates
to meet a budget. Candidate output order is canonical identity order where
observable reductions require determinism. Pair-list consumers emit each unordered
pair once; symmetric gather consumers may query both directions but apply only
their own endpoint's contribution. Do not double-apply a two-way force.

Neighbor lists must cover the full envelope, not just initial-distance neighbors,
if cached for all iterations. Alternatively repeat traversal of the **same**
tree with conservative queries; traversal is allowed, rebuilding is not.
Nearest queries use a deterministic tie-break by identity and preserve all
eligible owners. Queries without a finite radius must not invent one for speed.

Explicit scene queries can consume the latest committed epoch and pinned mesh
trees. Visibility of staged mutations must be specified through F8-32; a query
cannot implicitly build another dynamic index. No hidden change to existing
command semantics is authorized by this design.

## 7. Failure, synchronization and instrumentation

Reject non-finite geometry, unsupported motion envelopes, invalid generations,
capacity arithmetic overflow, traversal/candidate overflow and exhausted work
budgets explicitly. A solve that escapes its bound invalidates the tick; never
commit incomplete contacts, silently clamp outside the approved profile, enlarge
`dt`, or retry with extra substeps/builds. Recovery is a later caller-controlled
tick with valid parameters/capacity; report its cost and failure reason.

Use staged output state and a GPU-visible validity/commit token. All downstream
mutations and emitted events are conditional on a valid tick; failure preserves
the prior committed state, clock and pending inputs. F8-07/20 must implement and
test this atomicity, not merely set an error flag after writing public state.
Surface asynchronous errors through the existing completion/explicit-wait model,
without a mandatory host wait between bounds, tree, queries and solve.

Native adapters bind buffers, dispatch and enforce producer/consumer hazards.
Required GPU barriers remain; shader publication/atomics cannot substitute for
the appropriate queue/device memory ordering. Resource destruction/reuse waits
for all relevant fences. No per-domain CPU live-count round-trip is required.

Record separate counters for world builds, refits and publications; enforce
`builds + refits <= 1` and `publications == 1` for a successful non-aborted tick
(including empty publication). Also record key bits/sort passes, actual dispatches,
active/reserved/provisional leaves, visited nodes, emitted/filtered candidates,
query overflows, bytes, barriers, host waits and GPU phase times. A full build
includes every topology/bounds pass, not only its final publication kernel.

## 8. Implementation handoff and executable evidence

| Task | Interface/result to implement |
|---|---|
| F8-06 | Checked buffer plan; active compaction; bounds reduction/key ordering; binary topology and conservative union kernels; duplicate/empty/sparse benchmarks |
| F8-07 | `stage_boundary`, `emit_envelopes`, `build_or_refit_once`, `resolve_admission`, `publish_epoch`, query helpers, validity/retirement lifecycle |
| F8-08 | Rigid proxies, canonical body pairs, cached local mesh traversal and query visibility |
| F8-09 | Self/cross-population radius queries, identity-preserving storage mapping, source admission; remove pair-private sorts |
| F8-10 | Deformable triangle/capsule/point envelopes and particle-smoke queries; remove repeated surface refits; retain grid stencils |
| F8-11–19 | Enforce total motion/correction envelopes, preserve supported mode filters and paired exchanges; define phase-transfer boundary behavior |
| F8-20/21 | Atomic tick/clock commit, hazard schedule and actual-work telemetry assertions |

`tests/roadmap/test_f8_spatial_design.py` is a small independent CPU model of
the proposed contracts. It checks randomized full-path particle neighbors,
rotating bodies (including the endpoint-only negative control), segment/triangle
enclosure, signed skin bindings, touching/empty/duplicate cases, identity/compaction,
birth admission and atomic publication/failure. It does not call production
kernels or measure hardware. The production implementations must port these
obligations into F8-03's public tests and the F8-01 device matrix; passing this
model alone cannot qualify them.

The model rejects duplicate update/publication API calls before they can do
work; that does not invalidate otherwise valid state. In contrast, a solve
envelope escape or query overflow latches tick failure and prevents commit.
Ordinary admission can reject only provisional leaves; an atomic phase group
may instead retire its source when its destination is selected. Queries mask
inactive leaves. The executable transfer model covers one-to-one groups;
multi-particle conversion and physical conservation remain F8-18 obligations.

Run: `python3 -m unittest discover -s tests/roadmap -p 'test_*.py'`.
F8-00 must resolve the named symbolic limits and admission/phase timing tradeoffs
before dependent implementation freezes them. No device-specific equation fork,
missing-device pass, or retroactive relaxation of existing evidence is permitted.
