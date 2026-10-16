# API contract rules and evidence

These identifiers describe portable API behavior. They do not qualify a solver,
material, backend, or physical tolerance. A passing lifecycle test is not a
physics accuracy result.

| Rule | Required behavior | Executable evidence |
|---|---|---|
| PM-API-001 | Accept API 0.2 and zero (default 0.2); reject other versions with INCOMPATIBLE_VERSION and null output. | loader discovery tests; driver ABI tests (future version) |
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

The validation layer still needs typed command arguments and object-state
coverage. The table above maps implemented tests, not comprehensive proof of
all specification clauses. Geometry/paint ownership, every synchronization
interleaving, all coupling modes and physical accuracy remain separate gates.
