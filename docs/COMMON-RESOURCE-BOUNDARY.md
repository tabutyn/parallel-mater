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

This is an incremental boundary: handwritten `.cu` launch wrappers are still
translated, including their legacy mangled field/type references. Migrating
those wrappers to common typed command lists and using this ABI model in every
adapter remain open. A reflection match alone does not prove synchronization or
physical correctness.

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
metadata describes a dispatch; it does not itself insert a barrier or establish
completion ownership. Those scheduling and lifetime rules remain separate
follow-up work.

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

Resources now carry explicit owning-device/allocation identities and typed
access ranges at the Metal command-list boundary. Completion ownership and
dependency scheduling are still incomplete. A borrowed engine
device/queue/buffer must never be destroyed by the library. Queue handoff must
identify both prior completion and the completion the renderer waits for; all
indirect pointer resources must survive that completion, even if the caller
drops its token. The common mock-adapter tests must exercise those rules before
engine-specific packages advertise interoperability.

Physical heap capacity, usable OS/driver budget, library/application allocation
accounting and borrowed bytes are different quantities. Current budget queries
do not provide a confidence/unknown flag; Vulkan's heap-minus-tracked-allocation
estimate is not an OS admission guarantee. That distinction still needs an API
contract and conservative workload admission. Device and timeline imports remain
unadvertised by the current reference drivers; rendering on a library-created
device does not qualify engine-owned device import.
