# API contract rules and evidence

These identifiers describe portable API behavior. They do not qualify a solver,
material, backend, or physical tolerance. A passing lifecycle test is not a
physics accuracy result.

Evidence is scoped to the profile named by the test. `portable API` means the
rule applies to every conforming implementation. `validation layer` means the
optional diagnostic layer is being qualified, including a paired no-layer
control. `host fixture` is an injectable test driver, not an advertised solver.
`reference runtime` is the in-tree object/submission implementation, while
`reference solver` additionally executes an advertised physics domain. A pass
in one profile never silently qualifies another profile or backend.

| Rule | Required behavior | Executable evidence |
|---|---|---|
| PM-API-001 | Accept API 0.2 and zero (default 0.2); reject other versions with INCOMPATIBLE_VERSION and null output. | installed-SDK compatibility matrix; loader discovery tests; driver ABI tests (future version) |
| PM-MEM-001 | Buffer/memory handles belong to one device. Binding and mapped ranges fit the allocation without overflow. Binding twice or mapping twice is invalid state; live bound buffers prevent freeing their memory. | permissive-driver validation tests, paired with no-layer negative controls |
| PM-DEP-001 | Geometry, simulation, pipeline, buffer and backing-memory dependencies remain live while recorded or simulation-owned. Command reset/free releases recorded references; successful simulation replacement/reset updates retained geometry without preserving stale bindings. | permissive-driver validation tests, paired with no-layer negative controls |
| PM-QUERY-001 | Query pools belong to one device. Slots are reset before use, statistics scopes are matched around a step, in-flight or unread results are not reset, and result ranges, alignment, stride and flags are validated without overflow. Failed creation publishes no tracked object. | permissive-driver validation tests, paired with no-layer negative controls |
| PM-ABI-001 | A successful driver/layer negotiation returns a nonzero version no greater than requested. | installed C ABI 1/2/3 layer fixtures and independent driver/layer zero/999 rejection libraries |
| PM-LIFE-001 | Retired domain storage is reusable within the queried allocation; unrelated live slots survive replacement. | SDK resource/domain/coupling lifecycle tests; arena tests |
| PM-CMD-001 | Rejected argument validation enqueues no command. Recording can continue. Host descriptors are snapshots; referenced resources remain borrowed. | SDK descriptor-snapshot qualification (flat and nested mutation/free, rejection, reset and recording OOM); resource and coupling lifecycle tests |
| PM-SYNC-001 | Submission is asynchronous. Completion reports deferred errors. Failed submissions drain issued work; a later valid submission can recover. | async submission and submission queue tests; resource lifecycle NaN failure/recovery |
| PM-GRAPH-001 | Explicit simulation graph survives pipeline changes, rebinds replaced endpoints, and can be cleared. Graph failure retires partial edges. | SDK coupling lifecycle tests |
| PM-DIAG-001 | Validation failures deliver stable rule IDs, application-object context and callback-scoped typed arguments. Delivery is concurrent, independently reentrant and allocation-free; bookkeeping OOM disables stateful tracking without changing the API result. | permissive-driver validation diagnostics tests, paired with no-layer controls |
| PM-THREAD-001 | Physical-device queries and distinct command pools may be used concurrently. Host access to the same mutable queue, pool, fence, mapped range or resource, including destruction versus queries, is externally synchronized. | native-driver public host-concurrency test through the loader and validation layer; concurrent validation callback test |
| PM-DESTROY-001 | Applications complete work and destroy children before parents. In-tree reference drivers drain accepted work during device teardown; this reference-runtime behavior is not permission to violate portable dependency lifetime. | SDK destruction-wait qualification on the host fixture and advertised native drivers; submission queue destructor and validation device-teardown tests |
| PM-CAP-001 | Authored data exceeding declared capacity is rejected without disturbing unrelated state. Runtime-generated overflow is deferred through statistics/status, invalidates that step output and requires reinitialization. | SDK lifecycle capacity tests; runtime-overflow qualification on the host fixture and advertised native drivers |
| PM-RESET-001 | Pool, command-buffer, fence, query and simulation reset obey pending/state restrictions. Reset clears the state named by the command without silently changing declared capacity or unrelated objects. | async submission, failure lifecycle, resource lifecycle and validation-layer tests |

## F3-06 clause audit

The normative source, executable positive/negative cases and evidence profile
are kept together here. Source inspection is useful audit evidence but is not a
substitute for an executable case. Deliberately racing an externally
synchronized object is undefined application behavior, so negative threading
coverage must validate safe API boundaries rather than manufacture a data race.

| Clause / rule | Normative anchors | Profile and positive case | Negative / recovery case | Disposition |
|---|---|---|---|---|
| Threading · PM-THREAD-001 | [diagnostic callback](reference/parallel-mater-specification.md#spec-validation_diagnostics), [command pools](reference/parallel-mater-specification.md#spec-command_pool), [threading](reference/parallel-mater-specification.md#spec-threading) | advertised native driver + validation layer: [`host_concurrency_tests.cpp`](../tests/api/host_concurrency_tests.cpp) repeats physical-device snapshots and records/resets one distinct pool per thread; [`layer_validation_tests.cpp`](../tests/api/layer_validation_tests.cpp) delivers concurrent callbacks outside bookkeeping locks | same-object races remain externally synchronized and are deliberately absent; teardown begins only after all worker threads join | covered |
| Host snapshots · PM-CMD-001 | [calling conventions](reference/parallel-mater-specification.md#spec-conventions), [command buffers](reference/parallel-mater-specification.md#spec-command_buffers) | host fixture + advertised native driver: [`descriptor_snapshot_tests.cpp`](../tests/api/descriptor_snapshot_tests.cpp) mutates and frees flat transfer arrays plus nested cloth descriptor/vertex/index/mass arrays before execution | rejected nested recording leaves existing state unchanged; reset discards successful flat and nested snapshots; deterministic recording OOM is latched, retains no caller storage and recovers after reset | covered |
| Destruction / wait · PM-DESTROY-001 | [lifetime](reference/parallel-mater-specification.md#spec-units), [command lifetime](reference/parallel-mater-specification.md#spec-lifecycle), [submission](reference/parallel-mater-specification.md#spec-submission) | host fixture + advertised native driver: [`destruction_wait_tests.cpp`](../tests/api/destruction_wait_tests.cpp) distinguishes explicit fence completion from reference-driver device teardown with accepted pending work | deterministic host delay verifies copied output before teardown returns; a fresh device submission verifies no stale completion or validation ownership after retirement | covered |
| Deferred completion · PM-SYNC-001 | [return codes](reference/parallel-mater-specification.md#spec-results), [command lifetime](reference/parallel-mater-specification.md#spec-lifecycle), [submission](reference/parallel-mater-specification.md#spec-submission), [statistics](reference/parallel-mater-specification.md#spec-step_stats) | reference solver: [`async_submission_tests.cpp`](../tests/api/async_submission_tests.cpp) covers wait-before-signal, pending state, ordered submissions and completion channels | host fixture: [`failure_lifecycle_tests.cpp`](../tests/api/failure_lifecycle_tests.cpp) separates rejection before acceptance from deferred failure and recovery; [`submission_queue_tests.cpp`](../tests/api/submission_queue_tests.cpp) proves a consumed error does not poison later waits | covered |
| Capacity · PM-CAP-001 | [return codes](reference/parallel-mater-specification.md#spec-results), [pipeline bounds](reference/parallel-mater-specification.md#spec-pipeline_validity), [statistics](reference/parallel-mater-specification.md#spec-step_stats) | host fixture / advertised reference solver: [`resource_lifecycle_fixture.cpp`](../conformance/v1/src/resource_lifecycle_fixture.cpp), [`domain_lifecycle_tests.cpp`](../tests/api/domain_lifecycle_tests.cpp), [`coupling_lifecycle_tests.cpp`](../tests/api/coupling_lifecycle_tests.cpp), [`owner_slot_tests.cpp`](../tests/solver/owner_slot_tests.cpp) cover declared-capacity rejection; [`capacity_recovery_fixture.cpp`](../conformance/v1/src/capacity_recovery_fixture.cpp) generates exact neighbor overflow and proves deferred completion/statistics/status plus selective reinitialization | failed-step output remains unpublished; replacing only the overflowing population preserves an unrelated live population and the next step succeeds | covered |
| Reset · PM-RESET-001 | [command pools](reference/parallel-mater-specification.md#spec-command_pool), [command buffers](reference/parallel-mater-specification.md#spec-command_buffers), [simulation reset](reference/parallel-mater-specification.md#spec-simulation_init), [statistics](reference/parallel-mater-specification.md#spec-step_stats) | portable API / reference runtime: [`resource_lifecycle_fixture.cpp`](../conformance/v1/src/resource_lifecycle_fixture.cpp) reuses exact queried storage after reset; [`layer_validation_tests.cpp`](../tests/api/layer_validation_tests.cpp) releases tracked dependencies at completion; [`capacity_recovery_fixture.cpp`](../conformance/v1/src/capacity_recovery_fixture.cpp) reinitializes only the overflowing population | [`failure_lifecycle_tests.cpp`](../tests/api/failure_lifecycle_tests.cpp) clears a latched recording error; [`async_submission_tests.cpp`](../tests/api/async_submission_tests.cpp) rejects command/fence reset while pending | covered |

### F3-09 host-concurrency profile

The public-only host-concurrency fixture selects exactly one advertised native
driver manifest and requires the validation layer. It reports the native backend,
device name and PCI vendor/device identifiers obtained through the public API.
Eight workers repeat 256 complete physical-device query snapshots each. A second
eight-worker phase records, ends and resets 256 empty command buffers per worker,
with one command pool and buffer owned by each thread. The fixture fails on any
query instability, API error or validation diagnostic.

The test is registered for each built CUDA, D3D12, Metal and Vulkan Slang driver.
It does not race a queue, pool, command buffer, fence, mapped range, resource or
destruction against another access. Such same-object cases require application
synchronization and remain outside executable negative testing.

### F3-10 descriptor-snapshot profile

The public-only descriptor fixture covers the two record-time host-array shapes.
It records a flat `PmBufferCopy[]`, then mutates and frees the caller array before
submission. It also records an outer `PmClothInfo[]` whose vertex, triangle-index
and inverse-mass members each point to separate nested arrays, then mutates and
frees every caller allocation before copying the resulting cloth state. Execution
must reproduce the original recorded values in both cases.

Separate cases reject an invalid nested descriptor, reset successful flat and
nested recordings before submission, and use the deterministic host fixture to
fail command retention after the nested arrays have been copied. Rejection leaves
the prior cloth untouched; reset discards owned snapshots; recording OOM is
latched until reset and the next valid submission succeeds. The same success,
rejection and reset cases run on every built native Slang driver that advertises
cloth; the allocation-failure case runs only on the private host fixture.

### F3-11 destruction-wait profile

The public-only destruction fixture submits a copy behind an unsignaled timeline
value and confirms its fence is pending. Its portable profile then submits the
producer, explicitly waits for the consumer fence, verifies the copied bytes and
destroys children before the device. This is the required application lifetime
sequence and avoids relying on an implicit device wait.

The reference-runtime profile instead destroys the device with the accepted
consumer and producer still pending. A private host-fixture delay makes that
state deterministic and verifies that the copy completed before device teardown
returned. The advertised native-driver case verifies teardown completes without
a hang, and both profiles create and submit through a fresh device afterward to
exclude stale completion or validation ownership after handle retirement.

Implicit device-teardown draining is an in-tree reference-driver extension, not
portable guidance for external drivers. Applications must still wait explicitly
and destroy dependent child objects before their parent device.

### F3-12 runtime-capacity profile

The public-only capacity fixture declares two fluid populations. Its target
contains four particles at one position with `neighborCapacityPerParticle = 1`
and `supportRadius = 0.1`; the shared Slang neighbor operator therefore counts
three neighbors for each target particle and reports exactly four overflows. An
isolated one-particle population at x = 10 is the unrelated live control.

Command recording and `pmQueueSubmit` succeed. Fence completion,
`pmGetSimulationStatus` and the closed statistics query all expose
`PM_ERROR_CAPACITY_EXCEEDED`; the query reports `overflowCount = 4` and the copy
recorded after the failed step remains unpublished. The fixture then replaces
only the target with one isolated particle at x = 20. The next step, statistics
query and both population copies succeed with zero overflow, while the control
particle remains at x = 10.

The same source runs through the validation layer on the private Slang C++ host
fixture and each built advertised native Slang driver. Windows D3D12 compiles
the schedule to DXIL and passed 20/20 runs on the NVIDIA GeForce GTX 1080; the
WSL GCC 13.3 host fixture passed 20/20. This qualifies the named profile and
runtime recovery contract, not arbitrary capacity choices or physical quality.

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

Layer ABI 3 supplies registry-generated typed arguments, application-visible
dispatch-object context, and the immutable per-instance diagnostic sink to
generated forwarding commands and the handwritten loader paths listed below. The canonical
classification in `tools/registry/generate.py` emits
`manual_layer_inventory.inc`, which makes additions to the handwritten set an
explicit generated/tested decision.

| Manual entrypoint | Layer interception | Reason |
|---|---|---|
| `pmEnumerateDeviceExtensionProperties` | pre/post | physical-device extension dispatch |
| `pmDestroyInstance` | pre/post | aggregate instance teardown |
| `pmEnumeratePhysicalDevices` | pre/post | aggregate physical-device enumeration |
| `pmCreateDevice` | pre/post | device wrapper creation |
| `pmDestroyDevice` | pre/post | device wrapper teardown |
| `pmGetDeviceQueue` | pre/post | queue wrapper lookup and creation |
| `pmAllocateCommandBuffers` | pre/post | command-buffer wrapper creation |
| `pmFreeCommandBuffers` | pre/post | command-buffer translation and teardown |
| `pmQueueSubmit` | pre/post | command-buffer translation before submission |
| `pmImportNativeDevice` | pre/post | native-device wrapper creation |
| `pmImportVulkanDeviceEXT` | pre/post when built | Vulkan-device wrapper creation |
| `pmGetInstanceProcAddr` | excluded | loader symbol discovery, not command execution |
| `pmGetDeviceProcAddr` | excluded | loader symbol discovery, not command execution |
| `pmEnumerateInstanceVersion` | excluded | global call with no enabled layer chain |
| `pmEnumerateInstanceLayerProperties` | excluded | global call with no enabled layer chain |
| `pmEnumerateInstanceExtensionProperties` | excluded | global call with no enabled layer chain |
| `pmCreateInstance` | excluded | selects the layer chain that later calls use |

Memory/buffer validation tracks allocation, queried requirements, binding,
mapping, ownership and retirement. Geometry input buffers, geometry/simulation
bindings, submitted simulation geometry, pipeline material tables and every
recorded resource reference participate in lifetime checks. Command reset/free
releases its recorded dependency set; successful simulation reset/replacement
updates the separate live dependency set. Query pools track ownership, typed
slots, reset/write/consume/pending state and matched statistics scopes.
Command-buffer, fence, submission and timeline state are tracked through precise
completion boundaries. Native buffer and semaphore imports participate in
ownership checks. Old v1/v2 layers receive their unchanged prefix; this v3
validation layer rejects old loaders that cannot supply diagnostics and
application-object context.

Structured diagnostics use stable IDs. `PM-VAL-0001` reports a malformed
loader/layer payload, `PM-VAL-0002` invalid arguments or ownership,
`PM-VAL-0003` invalid tracked state, and `PM-VAL-0004` loss of stateful tracking
after host-bookkeeping exhaustion. Each callback receives the API result,
severity, command, application-visible dispatch object, and callback-scoped
typed arguments. The create-info callback is copied per instance. Delivery may
be concurrent, occurs outside the validation state mutex, and therefore permits
independent reentrant API reads. The application must keep its callback target
and user data alive through instance destruction and synchronize its own state.

Coverage is intentionally explicit: interop handoff is not yet validated by
this layer, and driver validation remains required. Invalid void calls are
suppressed after delivering a diagnostic. Host-bookkeeping exhaustion delivers
`PM-VAL-0004` without allocation, disables stateful tracking, and preserves
already-created driver objects and stateless checks. The table above maps
implemented tests, not comprehensive proof of all specification clauses.
Private geometry/paint generations, all coupling modes and physical accuracy
remain separate gates.
