# API contract rules and evidence

These identifiers describe portable API behavior. They do not qualify a solver,
material, backend, or physical tolerance. A passing lifecycle test is not a
physics accuracy result.

| Rule | Required behavior | Executable evidence |
|---|---|---|
| PM-API-001 | Accept API 0.2 and zero (default 0.2); reject other versions with INCOMPATIBLE_VERSION and null output. | loader discovery tests; driver ABI tests (future version) |
| PM-MEM-001 | Buffer/memory handles belong to one device. Binding and mapped ranges fit the allocation without overflow. Binding twice or mapping twice is invalid state; live bound buffers prevent freeing their memory. | permissive-driver validation tests, paired with no-layer negative controls |
| PM-ABI-001 | A successful driver/layer negotiation returns a nonzero version no greater than requested. | loader rejection tests with independent zero/999 mock libraries |
| PM-LIFE-001 | Retired domain storage is reusable within the queried allocation; unrelated live slots survive replacement. | SDK resource/domain/coupling lifecycle tests; arena tests |
| PM-CMD-001 | Rejected argument validation enqueues no command. Recording can continue. Host descriptors are snapshots; referenced resources remain borrowed. | SDK resource lifecycle (rejected capacity/range then valid work); coupling lifecycle (descriptor mutation) |
| PM-SYNC-001 | Submission is asynchronous. Completion reports deferred errors. Failed submissions drain issued work; a later valid submission can recover. | async submission and submission queue tests; resource lifecycle NaN failure/recovery |
| PM-GRAPH-001 | Explicit simulation graph survives pipeline changes, rebinds replaced endpoints, and can be cleared. Graph failure retires partial edges. | SDK coupling lifecycle tests |

## Lifetime and error boundaries

Destroy children before parents and release bound memory last. Destruction may
wait; the reference resource destructors drain pending device work. Explicitly
waiting for completion avoids an implicit wait. Destruction does not repair
recorded references: reset or free dependent command buffers before reuse.
Host calls accessing mutable objects require external synchronization.

Recording validates arguments and snapshots host descriptors. A rejected
argument is not recorded. A latched allocation error makes `pmEndCommandBuffer`
fail. Execution-time validation can fail after earlier commands ran; there is
no transactional rollback of an entire submission. Replacing a domain may
retire its prior population before a deferred failure. Other domain slots
remain valid and a later replacement can recover. See
[resource lifetime](RESOURCE-LIFETIME.md) for exact tested scope.

## Remaining qualification

Layer ABI 2 supplies registry-generated typed arguments to generated forwarding
commands and device destruction. Memory/buffer validation tracks allocation,
queried requirements, binding, mapping, ownership and retirement. It includes
native buffer import tracking. Old v1 layers receive their unchanged prefix;
this v2 layer rejects old loaders that cannot supply arguments.

Coverage is intentionally explicit: command-buffer transitions, submit/fence
state, geometry/simulation memory dependencies, query state and interop handoff
are not yet validated by this layer. Most hand-written loader entrypoints do
not yet invoke layer callbacks. Driver validation remains required. Invalid
void calls are suppressed; the layer has no structured diagnostic callback yet.
Host bookkeeping exhaustion reports a diagnostic on stderr and disables state
tracking, preserving already-created driver objects and stateless checks. The table above maps implemented tests, not comprehensive proof of
all specification clauses. Geometry/paint ownership, every synchronization
interleaving, all coupling modes and physical accuracy remain separate gates.
