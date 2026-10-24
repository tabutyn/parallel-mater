# F8-07 once-per-tick spatial service

F8-07 turns the F8-05 contract and F8-06 build primitives into a bounded,
backend-neutral epoch service. It does not migrate a domain consumer or
advertise the real-time profile; F8-08 through F8-10 consume this service and
F8-20 owns final schedule integration.

## Resource and epoch contract

`make_spatial_service_plan` checks every capacity and byte calculation and lays
out two to four 16-byte-aligned immutable snapshot slots. Each slot owns an
epoch header, full generation-bearing leaf metadata, conservative bounds,
active flags, admission bytes, BVH output, and bounded query output/status.
Only compaction/build scratch is shared while producing the next unpublished
slot. Published queries read the leaf count from their pinned epoch header, so
a later build cannot redirect an old in-flight traversal. Allocation therefore
occurs at capacity changes, not on the tick hot path.

The full identity is `(family, owner slot, owner generation, topology revision,
primitive kind, primitive slot, primitive generation)`. Dense staging slots,
compacted indices, and Morton rank are never identity. Removed entries are
inactive tombstones; the GPU compactor preserves stable source mapping for the
remaining leaves. An epoch pins graph/resource revisions and a monotonically
generated slot token until all acquired users release it. Slot-generation
exhaustion is an error rather than wraparound.

The boundary transaction is:

1. Stage the next tick and its frozen profile, graph, and resource revisions.
2. Emit conservative full-tick envelopes and active flags.
3. Select a full build, a refit, or the zero-leaf path before dispatch. The
   staging call must explicitly mark refit eligibility after proving topology
   and the full leaf identity set unchanged; otherwise refit is rejected. The
   prior topology is copied device-to-device into the unpublished slot before
   its leaves are refit, preserving the older immutable snapshot.
4. Compact active leaves and perform the sole build/refit. Counts and scratch
   remain GPU resident; no service kernel requires an intervening host wait.
5. Resolve deterministic provisional admission against the private completed
   tree, including source/destination transfer XOR semantics.
6. Publish one immutable epoch, execute any number of bounded queries against
   it, validate identities/envelopes/status, and atomically commit or abort.

`SpatialEpochLifecycle` rejects a second update or publication, non-consecutive
ticks, stale graph/resource revisions, stale slot generations, and use after
retirement. Acquired older snapshots remain valid while a new epoch commits.
Reset retires current state without making old tokens alias reused storage.
Build, refit, publication, query, candidate, and overflow counters are attached
to the exact epoch. A successful non-empty epoch has exactly one build or
refit; an empty epoch has neither and still has exactly one publication.

## Bounds, admission, and queries

The host helpers generate outward-rounded point envelopes from the complete
translation path, declared path deviation, total solve correction, and radius.
A rotation-origin radius encloses full rigid turns. Signed weighted interval
bounds use each weight's sign. Segment and triangle consumers can union those
point envelopes without losing conservatism.

Retained leaves are admitted. A provisional leaf conflicts with retained
geometry or every lower-identity overlapping proposal, including a lower proposal
that was itself rejected; deterministic underfill is intentional. Inactive
tombstones never participate. A transfer destination excludes its own source,
then finalization keeps exactly the source or destination. Malformed identity,
duplicate keys, missing/duplicate transfer destinations, or out-of-range
compaction mappings fail explicitly.

AABB traversal is inclusive and returns admitted candidates in canonical full
identity order. Candidate or traversal overflow is all-or-none and, during the
published pre-commit solve, latches tick failure. A later explicit query reports
its own overflow without retroactively invalidating committed state. Current
geometry leaving a published envelope, or an identity/count
change during validation, also prevents commit. Narrow phase remains the
consumer's job. The existing immutable per-mesh BVHs in `geometry_storage.hpp`
remain cached and outside the dynamic tick update.

## Kernel and device evidence

Eight service kernels initialize/stage epochs, refit leaves, resolve admission
and transfers, publish, query, and validate. They share the canonical Slang
manifest with the F8-06 compaction/build kernels. The refit path is two
dispatches (leaf scatter/arrival clear, then bottom-up union); it never falls
back to a build in the same tick.

The independent host tests cover checked plans, randomized motion/correction
envelopes, full rotation, signed weights, tombstone compaction, spawn/removal,
transfer admission, randomized BVH candidates against brute force, overflow,
topology/revision edits, snapshot pinning, reset, and generation reuse. The
physical CUDA test executes production generated Slang for sparse build,
admission, query, overflow, stale/envelope validation, duplicate publication,
and refit on an RTX 3050 Ti. The complete Vulkan driver and SPIR-V validation
pass, as does a physical Vulkan command-API run; changed entries also compile
to Metal source. Exact commands and device identities are recorded in
`evidence/F8-07-spatial-service.json`.
