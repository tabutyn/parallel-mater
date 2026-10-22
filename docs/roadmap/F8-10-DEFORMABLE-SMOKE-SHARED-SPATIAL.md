# F8-10 deformable and smoke shared-spatial consumers

F8-10 moves cloth triangles, rope segments, soft-body surface triangles,
stationary geometry and smoke obstacle lookup onto the F8-07 spatial epoch. It
does not replace the structured smoke field or activate the new public
schedule. F8-13 through F8-16 own one-step domain stability, F8-17/F8-18 own
cross-system exchange, and F8-20 owns schedule activation. Legacy-profile
behavior and advertised capability modes remain unchanged.

## Flattened primitive identity

`pm_emit_deformable_spatial_primitives` stages two-vertex segments and
three-vertex triangles from all supported deformable owners. The full identity
is:

`(cloth/rope/soft/static family, owner slot, owner generation, topology
revision, segment/triangle kind, stable primitive slot, primitive generation)`.

Dense leaf position, Morton order and query position are never identities.
Owner-slot reuse, topology replacement and individual primitive replacement
therefore cannot alias an in-flight snapshot. Cloth, rope and soft surfaces use
families 3, 4 and 5; family 6 represents stationary surface primitives. Rigid
family 1 and fluid family 2 remain owned by F8-08 and F8-09.

Each owner writes a non-overlapping range in one flattened leaf array. The
generic emitter writes service metadata, retained/inactive state, conservative
bounds, query masks, excluded self keys and query-to-leaf mappings. It never
builds, refits or sorts a private hierarchy. Static surfaces use the same
emitter with equal previous/predicted positions and zero motion budget.

## Conservative bounds and exclusions

A primitive bound is the union of every endpoint/vertex box over its boundary
and predicted position, expanded componentwise by:

`path deviation + total correction budget + thickness + outward epsilon`.

This retains thin contacts and intermediate segment/triangle crossings for the
whole tick. The independent host oracle samples 4,096 segment/triangle states
inside randomized motion/correction envelopes and checks containment. A
deterministic thin segment crossing a stationary triangle is also retained.

`pm_filter_deformable_spatial_candidates` consumes the service's canonical
source list. It validates both current owner generation and topology revision,
then maps stable primitive generation to the current payload. Self identity is
removed. For a same-owner self-contact view, segments/triangles sharing any
vertex are also removed, covering rope neighbors and cloth/soft surface
adjacency. Cross-owner and cross-family views reuse the same candidate storage
without applying those local topology exclusions. Any stale mapping or
overflow clears that query's output and latches the tick status.

The boundary transaction is therefore one owner emission phase, one shared
build or refit, one publication, and bounded service queries. Four collision
passes pin the exact epoch token and query count through
`DeformableSpatialSolveEpoch`; another build/refit/publication/query is an
invalid state. There is no per-edge, per-triangle or per-coupling refit.

## Smoke remains a structured-grid solver

`pm_emit_smoke_spatial_queries` emits bounded obstacle-query boxes from smoke
cell/sample centers. `pm_filter_smoke_spatial_obstacles` selects admitted rigid,
cloth, rope, soft or static leaves through an explicit family mask and rejects
stale identities. These entries replace only obstacle/collision lookup.

Pressure smoothing, residual restriction, prolongation, face projection,
advection, splatting and boundary stencils remain in `smoke_grid.slang` and its
regular grid. No BVH is introduced for a numerical field stencil, and no smoke
capability or coupling mode is added.

## Bounded work

The checked consumer plan owns query-to-leaf mappings plus one reusable
filtered candidate view. The service owns immutable snapshots, unfiltered
candidates and build scratch. Measurements use 64 candidate slots per query:

| Case | Live / reserved primitives | Smoke obstacle queries | Live / capacity 64-thread query workgroups | Filtered candidate bytes | Boundary consumer dispatches | Rebuilds in solve |
|---|---:|---:|---:|---:|---:|---:|
| Sparse mixed scene | 512 / 8,192 | 2,048 | 40 / 160 | 2,621,440 | 7 | 0 |
| Full deformable tier | 8,192 / 8,192 | 0 | 128 / 128 | 2,097,152 | 5 | 0 |

For three deformable domains, removing repeated consumer-local bounds avoids at
least 1,048,576 bytes in the sparse mixed case and 786,432 bytes in the full
deformable case. Those figures exclude private BVH nodes and sort/refit scratch,
so they are conservative. Allocation is fixed at capacity; assembled live
dispatch sizing belongs to F8-20, and final budget freezing belongs to F8-21.

## Evidence and limits

Generated production Slang runs on a physical RTX 3050 Ti for cloth, rope and
static emission, self-adjacency exclusion, cross-family reuse, smoke obstacle
queries, stale-generation rejection and overflow. The shared F8-07 service also
passes physical empty, full and sparse-capacity build/query cases. CUDA, CPU
stub and Vulkan bindings compile; normalized SPIR-V validates; all four entries
compile to Metal and HLSL SM 6.6 source. Commands, toolchain and budget records
are in
[`evidence/F8-10-deformable-smoke-shared-spatial.json`](evidence/F8-10-deformable-smoke-shared-spatial.json).

This task does not claim 60 Hz domain stability, final timing, physical D3D12
or Metal execution, schedule speedup, new interaction modes, or public
real-time-profile support.
