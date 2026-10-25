# F8-20 backend-neutral one-tick schedule

F8-20 adds the allocation-free orchestration contract for
`realtime-single-step` version 1.0. `realtime_schedule.hpp` is the sole ordering
authority for a profile tick. It composes the spatial, domain, and coupling
budgets landed by F8-07 through F8-19 without changing the legacy reference
schedule.

## Fixed tick order

`assemble_realtime_tick` records this sequence on one ordered backend queue:

1. sample the input snapshot and apply boundary mutations once;
2. predict every present domain once at `dt = 1/60`, `substeps = 1`;
3. build or refit the dynamic spatial service at most once;
4. publish exactly one spatial epoch, including an empty epoch;
5. execute the exact domain iteration budgets;
6. execute four batched sweeps for every present coupling family;
7. atomically commit state, clock, pending inputs, and the render snapshot.

Domain presence is host graph metadata, not a live element count. The backend
retains fluid, smoke, topology, and candidate counts in GPU buffers. No count
readback, host wait, per-domain timestep, per-edge dispatch loop, retry, or
second index update exists in the scheduler interface. A backend may encode
multiple kernels inside one phase visit, but ordered queue hazards must remain
between producers and consumers.

The version-1 domain iteration vector is rigid `8`, liquid `6`, cloth `8`, soft
body `8`, rope `8`, and particle smoke `6` or grid smoke `12`. Each of the
rigid, liquid, and deformable/smoke coupling families receives four sweeps when
it has registered edges. Edges are passed as one batched family count, so the
schedule cannot subdivide time or rebuild spatial data per edge.

## Atomic ownership

`RealtimeSnapshotOwnership` pins one input generation while work is pending and
rotates previous/current render generations only on commit. Abort discards the
pending generations and preserves completed input and render ownership. The
backend contract similarly exposes one `commit_tick` and one `abort_tick`; no
domain can publish a partial commit.

Scene, budget, and ledger records are fixed-size trivially copyable values.
The assembler contains no allocation path. Its ledger rejects an incomplete
tick unless it observes exactly one boundary and time advance, one publication,
zero or one spatial update, every fixed domain/coupling iteration, zero host
waits/readbacks, and one final commit.

## Integration tests

`parallel-mater-realtime-schedule-tests` covers:

- an empty tick with one empty publication and no update;
- each domain independently, including twelve grid-smoke cycles;
- a six-domain mixed tick with all three coupling families;
- invalid timestep, substep, edge-capacity, and missing-update requests;
- failure before begin and during solve, with no commit;
- input/render generation commit and abort ownership;
- 600 warmup plus 7,200 measured mixed orchestration ticks.

The x86-64 host suite passed 28/28 tests. A physical Pixel 7 ran the final
AArch64 scheduler binary (SHA-256
`3ea32e5290a810928e96e373ef862b7050a1aea98b6679186d1f2103fd5ca43f`)
at exact `dt = 1/60`: 7,200 orchestration ticks took `0.003437866 s`, a
simulated-to-wall ratio of `34905.3744`. This is schedule overhead only, not a
GPU physics or F8-21 frame-time result.

The provenance-bound Vulkan public-API regression artifact from implementation
commit `23e7ea6c05d1e6e1b237b3a458d5c0d27f6d7d74` passed 25/25 device
invocations on the Pixel 7 Mali-G710 with Slang 2026.18 and Android NDK 29.
It preserves the existing 17/18 independent legacy evaluators and the explicit
CUDA-golden numerical mismatch; it is not relabeled as realtime qualification.

## Evidence boundary

F8-20 supplies the backend-neutral schedule law and its integration seam. The
reference driver still enumerates no realtime profile: native adapters must
bind these phase visits to emitted production work and expose measured
dispatch, barrier, submission, readback, and phase-time records in F8-21 before
advertisement. F8-22 then freezes one cross-backend candidate. Keeping the
profile unadvertised here prevents an orchestration test or legacy Vulkan run
from becoming a false public execution claim.

Exact commands, hashes, counts, and device identity are in
[`evidence/F8-20-realtime-schedule.json`](evidence/F8-20-realtime-schedule.json).
