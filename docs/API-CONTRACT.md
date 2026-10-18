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
| PM-DIAG-001 | Validation failures deliver stable rule IDs, application-object context and callback-scoped typed arguments. Delivery is concurrent, independently reentrant and allocation-free; bookkeeping OOM disables stateful tracking without changing the API result. | permissive-driver validation diagnostics tests, paired with no-layer controls |

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
mapping, ownership and retirement. Command-buffer, fence, submission and
timeline state are tracked through precise completion boundaries. Native buffer
and semaphore imports participate in ownership checks. Old v1/v2 layers receive
their unchanged prefix; this v3 validation layer rejects old loaders that cannot
supply diagnostics and application-object context.

Structured diagnostics use stable IDs. `PM-VAL-0001` reports a malformed
loader/layer payload, `PM-VAL-0002` invalid arguments or ownership,
`PM-VAL-0003` invalid tracked state, and `PM-VAL-0004` loss of stateful tracking
after host-bookkeeping exhaustion. Each callback receives the API result,
severity, command, application-visible dispatch object, and callback-scoped
typed arguments. The create-info callback is copied per instance. Delivery may
be concurrent, occurs outside the validation state mutex, and therefore permits
independent reentrant API reads. The application must keep its callback target
and user data alive through instance destruction and synchronize its own state.

Coverage is intentionally explicit: geometry/simulation memory dependencies,
query state and interop handoff are not yet validated by this layer. Driver
validation remains required. Invalid void calls are suppressed after delivering
a diagnostic. Host-bookkeeping exhaustion delivers `PM-VAL-0004` without
allocation, disables stateful tracking, and preserves already-created driver
objects and stateless checks. The table above maps implemented tests, not
comprehensive proof of all specification clauses. Geometry/paint ownership,
all coupling modes and physical accuracy remain separate gates.
