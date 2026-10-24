# F8-09 liquid consumers on the shared spatial epoch

F8-09 supplies the liquid leaf, source-admission and radius-query consumer path
for the once-per-tick spatial service from F8-07. It does not switch the public
reference schedule or advertise the real-time profile: F8-12 owns the new
single-step liquid equations, F8-18 owns cross-system force integration, and
F8-20 owns final schedule activation. The legacy profile and its evidence are
unchanged.

## One flattened identity space

Every retained particle and bounded source proposal is staged in the same world
leaf array. A particle identity is:

`(fluid family, population slot, population generation, material revision,
particle primitive, stable particle ID, primitive generation)`.

The epoch metadata maps that full identity to the current dense storage slot.
Dense slots, query slots and Morton ranks are not identities. Therefore equal
local or stable particle numbers in two populations remain distinct, compaction
does not change identity, and population-slot reuse cannot alias an in-flight
epoch.

`fluid_emit_spatial_particles` emits retained leaves, full-tick query envelopes,
inactive capacity tombstones and query-to-leaf mappings. All populations write
different ranges of one snapshot. `fluid_emit_spatial_source_proposals` writes
provisional leaves into that same snapshot. Neither kernel emits cell keys or
owns a tree.

## Boundary transaction

The new profile uses this exact order:

1. Apply unconditional outflow and compact each population once. Stable IDs
   survive the dense compaction.
2. Reserve bounded source proposals. Their prospective stable IDs and full
   identities are fixed before the private build; rejected proposals may leave
   ID gaps, but an ID is never reused.
3. Emit retained and provisional leaves, then perform the sole shared
   build/refit and the F8-07 deterministic admission pass.
4. `fluid_apply_spatial_source_admission` commits accepted proposals to dense
   storage, rewrites only their pre-publication payload mapping, disables
   rejected/capacity-limited query slots, and publishes accepted/rejected/
   capacity-miss counts. Rejected leaves remain harmless tombstones in the
   already-built tree. No source-specific tree or post-emission rebuild exists.
5. Publish once. Self and distinct-population consumers query that same epoch.

Retained leaves use half the maximum active source spacing as their admission
radius; proposals use half their own spacing. AABB admission is conservative
and may intentionally underfill, as allowed by the F8-05 priority rule. The
exact spherical distance test remains in the neighbor consumer. Zero-particle
populations emit only inactive capacity records and are valid.

## Query and solve reuse

The service returns source indices in canonical full-identity order.
`fluid_filter_spatial_candidates` validates the pinned query identity and
stable-ID mapping, then selects a requested population slot/generation. It is
the same kernel for self and cross-population views; no pair-private key array,
radix workspace or hierarchy is allocated. Accepted source proposals use the
same mapping after admission.

`fluid_count_spatial_neighbors` resolves current dense positions and performs
the exact radius test. Self queries use that population's support; an existing
cross-population operator passes its documented support, including
`max(h_a, h_b)` where required. The broad list covers the full-tick envelope,
so six density sweeps or four coupling sweeps may reevaluate current positions
without another query build or sort. Candidate/query overflow is all-or-none;
identity drift, invalid bounds and capacity errors latch the shared tick status.

`FluidSpatialSolveEpoch` rejects any build, refit, publication or query-count
change during its six fixed density sweeps. This makes “one update” executable,
not a scheduler convention.

## Conservative motion and measured work

Particle center bounds are the componentwise box between boundary and predicted
positions, expanded by the declared full-tick path-deviation and total solver-
correction budgets. Query bounds add the applicable support radius and outward
epsilon. The independent host oracle samples states throughout those envelopes
and proves that every exact self/cross neighbor overlaps the staged query/leaf
bounds for heterogeneous supports.

The checked consumer plan owns only query-to-leaf mappings and a reusable
owner-filtered candidate view. The shared service owns the unfiltered candidates
and hierarchy. The desktop fluid workload measurements are:

| Case | Live / reserved particles | Proposals | 64-thread live / capacity query workgroups | Filtered candidate bytes | New-profile sort/build dispatches inside solve |
|---|---:|---:|---:|---:|---:|
| Sparse diagnostic | 128 / 30,000 | 64 | 3 / 470 | 15,392,768 | 0 |
| `fluid-rigid` full tier | 30,000 / 30,000 | 0 | 469 / 469 | 15,360,000 | 0 |

The sparse case has two populations, one undirected pair (two directed views),
128 candidate slots per query and nine boundary consumer dispatches. Removing
just that pair's two legacy key and two identity buffers avoids at least 720,000
bytes; radix scratch and the pair's density/interface arrays make the real
legacy allocation larger. Candidate storage is fixed at capacity and reused,
while dispatch sizing follows GPU-resident live/proposal counts in the assembled
F8-20 schedule. F8-21 will freeze final per-workload limits and timing.

## Evidence and limits

The generated Slang kernels execute on a physical RTX 3050 Ti and compare
self/cross counts and overflow with brute force, including full-tick motion,
heterogeneous supports, accepted/rejected sources, inactive capacity and stale
identity negatives. CPU/CUDA/Vulkan generated bindings compile; SPIR-V validates;
the five entries compile to Metal and HLSL SM 6.6 source. Exact commands and
device/toolchain records are in
[`evidence/F8-09-fluid-shared-bvh.json`](evidence/F8-09-fluid-shared-bvh.json).

This task does not claim new interaction modes, final liquid stability at
`dt=1/60`, schedule speedup, D3D12/Metal physical execution, or public real-time
profile support.
