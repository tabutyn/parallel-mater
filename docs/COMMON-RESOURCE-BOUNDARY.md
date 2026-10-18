# Common resource and binding boundary

The reference driver keeps physics in shared Slang. Its host scheduler uses
`device::ExecutionQueue`, `device::DispatchSize` and `device::Services` from
`src/device/device.hpp`. `compute.hpp` additionally defines reflected records,
typed arguments, retained buffer references, command lists and completion tokens;
the Metal adapter currently uses that newer interface. These two execution
surfaces have not yet been consolidated across every adapter.

## Reflection-derived bindings

`tools/slang_abi.py` reads Slang JSON reflection and emits ordinary C++ records
and function signatures using source names. It checks supported types, record
size/alignment, member offsets, array stride and duplicate definitions. Generated
C++ asserts every record's size, alignment and member offsets. Unknown shapes
fail generation rather than silently choosing an ABI.

Vulkan's 23 supported shader modules use this model through
`generate_vulkan_dispatch.py`. They no longer compile Slang to CUDA or inspect
compiler-generated CUDA declarations. The kernel inventory remains in
`cmake/SlangKernelManifest.cmake`; the SPIR-V reflection supplies parameter order
and push-constant layout. Pointer translation and native dispatch stay in the
Vulkan adapter. Shared Slang equations are unchanged.

D3D12's 141 isolated DXIL entries use the same model. The Python embedder now
validates each reflected signature and emits the bytecode table, thread-group
size, total constant size and named parameter offsets/sizes. Direct rigid,
AVBD and primitive dispatches refer to those generated symbols; they contain no
literal DXIL offsets. The domain wrapper generator validates every reflected
type and parameter before translating host launch control. The former CMake
JSON layout parser has been removed.

This is an incremental boundary: handwritten `.cu` launch wrappers still
provide transitional host control flow and launch geometry, including legacy
mangled field/type references. They provide no ABI declarations or offsets.
Migrating those wrappers to common typed command lists remains open. A
reflection match alone does not prove synchronization or physical correctness.

## DXIL binding qualification — 2026-10-18

- Pinned Slang 2026.18 generated and validated all 141 DXIL entry reflections;
  the eight D3D12 domain sources produced 52 host functions.
- Existing Vulkan artifacts validated 84 entries in 14 available modules with
  the same common model, including padded arrays and target-specific resources.
- Visual Studio 2022/MSVC 19.44 built the Release D3D12 driver on Windows 11
  x64. Common ABI, DXIL generator, architecture and native D3D12 command API
  tests passed 4/4 on the GTX 1080.
- Negative tests reject duplicate/mismatched entries, signature-count changes,
  overlapping layouts and unsupported uniform shapes before C++ compilation.

These results qualify binding generation and native adapter execution only.
They do not replace physical-quality, lifecycle or performance evidence.

## Resource identities and dispatch access

The private `device::compute` boundary identifies each allocation with an
opaque `(device, allocation)` pair. Allocation values from different devices do
not alias. Two buffer wrappers that report the same pair describe the same
allocation and must agree on allocation size and device address; disagreement
is rejected rather than guessed.

Every non-null direct parameter, pointer inside a by-value record, and resource
reached indirectly through a device-memory record produces a named binding with
all of the following metadata:

- binding kind: parameter, record pointer, or indirect reference;
- allocation identity and retained buffer reference;
- a non-empty half-open byte range `[offset, offset + size)`; and
- read, write, or read-write access.

Omitted sizes conservatively cover the remainder of the allocation, and the
current pointer-oriented scheduler uses that default where the launch ABI does
not yet carry a more precise extent. Explicit ranges must be inside the
allocation and at least large enough for the reflected pointee. Null pointers do
not create resource bindings and cannot carry offsets or ranges.

Overlapping bindings for the same allocation are legal and remain separate, so
an adapter can observe every shader-visible use and combine access modes without
losing their source names. Adjacent half-open ranges do not overlap. Access
metadata describes a dispatch and the common scheduler below derives ordering
from it. Completion ownership remains a separate lifetime responsibility.

## Range dependency scheduling

`compute::CommandList` normalizes the resource ranges of each dispatch, copy,
fill and upload before native encoding. Overlapping aliases within one command
combine their access modes and never create a dependency on that same command.
The list retains the most recent access state for each byte interval; replacing
one interval does not erase hazards on disjoint portions of the allocation.

An exact intersected range is ordered when either its prior or next access
writes. This covers read-after-write, write-after-read and write-after-write.
Read-after-read, adjacent or disjoint ranges, and different allocation
identities do not create a dependency. Inconsistent wrappers for one allocation
remain validation errors and do not discard earlier valid commands.

The common scheduler passes each required transition to
`CommandEncoder::barrier` immediately before the consuming command. The callback
contains only buffer/range/access data. It records native GPU ordering and must
not submit or wait on the CPU. Metal already orders the separate compute/blit
encoders in one command buffer, so its thin adapter validates the transition and
uses that native encoder ordering without adding a host wait. Other adapters can
translate the same callback to their barrier model as they adopt this command
interface.

Dependency history exists only while a command list records. Submission clears
that history while the completion owner retains the actual resources, so a
submitted `CommandList` cannot extend allocation lifetime. Resource hazards
inside one command list are distinct from the prior/next queue handoff below.

Portable mock coverage is in `compute_dependency_tests.cpp`. It checks exact
RAW, WAR and WAW intersections, partial-range history, overlapping aliases in a
single dispatch, read/read and independent-resource elision, callback ordering,
validation recovery and the absence of implicit waits. The ordinary
shader-portability matrix builds and runs it on Windows, Linux and macOS.

## Command and completion ownership

`compute::CommandList` retains kernels, direct and record-pointer bindings,
explicit indirect references, transfer buffers and upload staging allocations.
Native `CommandEncoder` implementations encode commands and return native
completion tokens; they no longer implement their own C++ retention policy.
The Metal adapter uses this shared layer. Other adapter execution surfaces have
not yet migrated, so this is not a claim of universal backend interoperability.

A device owns one `SubmissionQueue`. Its pending list is independent of client
completion tokens: dropping a token or command list neither cancels accepted
work nor waits on the GPU. Observing terminal completion releases the retained
owners, including on deferred execution failure. Dropped-token completions are
reclaimed nonblockingly when recording/submitting again or calling `collect()`;
an idle queue may retain them until its next collection or destruction.

The queue allocates its completion wrapper and pending slot before native
acceptance. Native submission must either enqueue nothing and throw, or return
a valid token without throwing after acceptance. Native `ready()` may throw only
for terminal failures, and `wait()` must drain before returning or throwing.
These are adapter contracts, exercised by a deliberately non-owning mock.

Common transfer/binding validation errors preserve prior recorded commands.
Encoding or submit failure cancels the private command list and releases its
owners after destroying the unsubmitted native encoder. This private behavior
does not change public `pmQueueSubmit` retry/error semantics. Native upload
encoding snapshots caller bytes and returns the staging owner to the common
list; later host edits do not change queued data.

The Metal host scheduler discards canceled lists after reporting an encoding or
submission error, allowing the next operation to start a fresh batch. It preserves
still-recording lists after ordinary validation errors. Its own pending-token
slot is also allocated before native acceptance, so allocation failure cannot
make `wait()` miss an accepted submission. The scheduler's injected-failure test
runs without GPU hardware on macOS CI.

Device teardown drains accepted work before releasing native device/queue
members, continues draining after a deferred error, and does not throw. A
surviving unsubmitted command list cannot enqueue after teardown. Each command
list is single-writer; shared queue bookkeeping and completion observations are
serialized. F4-02 changed no physics equations, synchronization policy,
borrowed-device contract or public API; the range scheduler above subsequently
added the common GPU-ordering policy without changing physics or public API.

Portable evidence is in `tests/solver/compute_lifetime_tests.cpp`; the ordinary
shader-portability CI executes it on Windows, Linux and macOS. Native Metal
tests additionally verify dropped-token transfers and device teardown with
actual GPU work. These tests establish ownership/execution behavior, not
physical solver quality or an engine integration.

## Borrowed objects and queue handoff

`device::compute::NativeObject` is the type-erased ownership primitive for thin
native adapters. A borrowed object contains a kind and opaque nonzero handle but
cannot carry a release callback. Reset, move-assignment and destruction therefore
never destroy an imported device, queue, buffer or completion. An owned object
requires a release callback and calls it exactly once. Native API types and
release functions remain inside the adapter.

The common `Device::native_device()` and `native_queue()` accessors return these
borrowed wrappers rather than raw owning-looking pointers. Metal is the first
adapter on this boundary; its exported wrappers carry the Metal API domain and
cannot release the `MTLDevice` or `MTLCommandQueue` retained by the adapter.

Every command encoder and completion also exposes an opaque `(API, device, queue)`
identity. `SubmissionQueue::commands(encoder, prior)` validates the complete
prior list before asking the adapter to encode waits:

- null, unidentified, different-API and different-device completions are rejected;
- same-device/same-queue work may use native queue ordering;
- same-device/different-queue work is accepted only when the adapter can encode
  a native fence, event or timeline wait; unsupported handoff is an explicit
  error; and
- `CommandEncoder::wait` may record GPU ordering but must not poll, block the
  host or submit the consumer command list.

The completion returned by `CommandList::submit` is the explicit **next**
completion and reports the consumer queue identity. It retains every prior
completion transitively until that next work reaches a terminal state, even if
the caller drops either token. This keeps all resources owned by prior physics
or rendering work alive without a whole-device or CPU synchronization point.
An adapter must allocate the next native token before accepting work, as for
ordinary submissions. Validation or adapter-wait failure accepts no consumer
submission and destroys only the unsubmitted wrapper state.

The public API maps onto this neutral contract without another backend-specific
physics API. `pmImportNativeDevice`, `pmImportNativeBuffer` and
`pmImportNativeSemaphore` borrow application objects; destroying a PM wrapper
destroys its internal children but never the imported native object. Conversely,
objects returned by `pmGetNativeDevice`, `pmGetNativeQueue`,
`pmGetNativeBuffer` and `pmGetNativeSemaphore` are borrowed views and do not
transfer ownership to the application. The application must keep imported
objects alive until PM children and pending work have been destroyed or drained.

Import support is advertised separately through `PmNativeInteropProperties`.
An unadvertised representation returns `PM_ERROR_FEATURE_NOT_PRESENT`; null,
foreign-device, wrong-backend, unsupported-feature and incompatible usage/state
inputs are errors and enqueue no work. The same native queue requires serialized
host submission. Different queues use imported timeline values in
`PmSubmitInfo`: the producer signal is the consumer's prior wait, and the
consumer signal is the renderer's next wait. Queue ordering does not replace
buffer ownership or visibility. Imported buffers are acquired with
`PM_QUEUE_FAMILY_EXTERNAL -> 0` and released with `0 ->
PM_QUEUE_FAMILY_EXTERNAL`; D3D12 buffers enter and leave PM in
`D3D12_RESOURCE_STATE_COMMON`.

The portable handoff contract test covers borrowed/owned destruction, move and
exactly-once release, prior-before-consumer ordering, same-device cross-queue
handoff, invalid and foreign completion rejection, explicit next identity,
transitive prior retention and absence of CPU waits.

## Engine handoff mock qualification

`engine_handoff_mock_tests.cpp` supplies a native-API-independent engine and
thin adapter around the common objects. The engine owns one device, separate
physics and renderer queues, and shared scene/output buffers. The adapter stores
borrowed `NativeObject` views, creates only PM buffer wrappers, and uses the
common `SubmissionQueue` for both queue identities.

Three frames are recorded before any mock GPU work completes. Each physics
submission waits for the previous renderer completion, and each renderer copy
waits for that frame's physics completion. This produces six simultaneous live
submissions and five cross-queue GPU dependencies in strict
physics/render/physics/render order. All caller completion tokens and buffer
handles are dropped before adapter teardown; queue-owned and transitive
completion references keep them valid without a host wait.

The failure path rejects a queue from another device, a wrong-API object, a
queue passed as a buffer, and a buffer from another device before creating a
wrapper or accepting work. Teardown drains each accepted native job and destroys
each completion and imported wrapper exactly once. It then proves that no native
device, queue or buffer was destroyed; only the engine's later explicit teardown
destroys those objects.

This portable mock qualifies the common ownership/dependency protocol before a
native engine plugin exists. It does not advertise device/timeline import in a
driver whose `PmNativeInteropProperties` says it is unsupported, and it does not
replace native D3D12 fence, Metal shared-event, Vulkan timeline or CUDA external
semaphore qualification.

## Vulkan binding and validation isolation — 2026-10-18

- All 23 reflected Vulkan modules compile with storage-layout assertions and
  pass SDK 1.4.341.0 static validation.
- Neutral ABI tests reject unknown types, overlapping/out-of-bounds records,
  inconsistent duplicate definitions, padded array strides and by-value cycles.
- Vulkan public-command API, asynchronous submission and resource lifecycle
  tests pass with the pinned native Vulkan validation layer.
- Exact-query lifecycle tests pass with Parallel Mater validation: 4,096 fluid
  replacements across capacity cases, preserved unrelated slots, and 128 deferred
  failure/recovery cycles.

The installed `libVkLayer_khronos_validation.so` from Debian package
`1.3.239.0-2` still crashes on that lifecycle test. The standalone
`tools/vulkan_pipeline_probe.cpp` reduction shows descriptor and pipeline-layout
creation complete, then the layer segfaults inside `vkCreateShaderModule` while
inspecting `fluid_emit_cells`. The same shader and probe complete with the
pinned SDK 1.4.341.0 layer, and the unchanged lifecycle case passes afterward.

The prior CUDA-derived binding generator and current reflection-derived
generator were controlled at commits `a973248054a79582f274e8485c792f8d6537e4b0`
and `5adeee1bf0178210f40ce1628d8e78cfca60f10a`. That refactor changed no Slang
source or SPIR-V normalization source, and the standalone probe loads the SPIR-V
without either generated host wrapper. This excludes binding generation from
the crash. The supported setup selects the pinned layer without replacing the
system package. See [F5-09 evidence](roadmap/evidence/F5-09-vulkan-validation-layer.json).

## Remaining common contracts

Resources now carry explicit owning-device/allocation identities, typed access
ranges, range-derived dependencies and common completion ownership at the Metal
command-list boundary. Borrowed-object destruction and prior/next queue handoff
are now explicit common contracts, and the engine-independent mock qualifies
alternation, simultaneous frames, caller-token release and teardown. Migration
of the other adapters and native engine plugins remains incomplete. Native
packages must still prove their API-specific imports and synchronization before
advertising interoperability.

Physical heap capacity, usable OS/driver budget, library/application allocation
accounting and borrowed bytes are different quantities. Current budget queries
do not provide a confidence/unknown flag; Vulkan's heap-minus-tracked-allocation
estimate is not an OS admission guarantee. That distinction still needs an API
contract and conservative workload admission. Device and timeline imports remain
unadvertised by the current reference drivers; rendering on a library-created
device does not qualify engine-owned device import.
