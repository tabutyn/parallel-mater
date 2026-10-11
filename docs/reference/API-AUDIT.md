# API coverage audit

## Result

The revised command contract has **90 functions and 88 named public types**.
Every function is declared in a header, printed in both consumer documents,
has a matching `PFN_pm*` typedef, and has a linkable placeholder definition.
Every named type is indexed in both documents and compiles as a complete
public value type. Opaque handle targets (`PmDevice_T`, etc.) deliberately
remain incomplete; consumers use the handle, not its private representation.

These are **unsupported stubs, not 90 working physics functions**. The gallery
still uses the existing C++ `World` backends. No solver or gallery behavior was
changed by this work.

## Omissions repaired

| Finding | Change |
| --- | --- |
| All 83 previously proposed C entry points lacked definitions | Added generated, opt-in stub translation units with exact C linkage |
| Basic aliases, opaque handles, constants and `PmStructureType` were absent from the quick reference | Added Fundamental declarations to both documents from the same header source |
| Allocation callback aliases were absent from the symbol index | Index now includes `PmAllocateFn`, `PmFreeFn`, and all other public type aliases |
| Header/document checks covered command names but not complete types | Added public-type inventory, field-token coverage, C/C++ compilation and installed/reference header equality checks |
| Backend integration only specified Vulkan | Added seven common native capability/import/get functions and native object mappings for CUDA, Metal, DirectX 12 and Vulkan |
| Category headings repeated across columns and linked their whole title | Print each category once; link only its purple specification marker |

The seven additional commands are `pmGetPhysicalDeviceInteropProperties`,
`pmImportNativeDevice`, `pmGetNativeDevice`, `pmImportNativeBuffer`,
`pmGetNativeBuffer`, `pmImportNativeSemaphore`, and `pmGetNativeSemaphore`.
They have the same unsupported-stub coverage as the existing commands.
The six typed Vulkan adapter declarations are retained under the general
interoperability chapter, not presented as the only integration model.

## Coverage evidence

- [api-coverage.json](api-coverage.json) lists every command/type, its specification
  section, header, and stub source. Both PDFs have the same 178 indexed names.
- The 90 derived `PFN_pm*` types follow the signature rule in §3.1/§17.1 and the
  reference's opening legend. Repeating all 90 signatures a second time in the
  guide would add no information. C11 and C++17 pointer assignments verify each.
- The checker links and calls every command from **both C and C++**, compares
  unmangled exported symbols with the declaration inventory, and compiles all
  88 named public types. Full type fields are checked in their reference panels.
- The runtime stub test checks failure outputs, bootstrap name lookup, native
  capability rejection, untouched failed-query payloads, and immediate return
  from infinite-timeout waits. It remains active under `NDEBUG`.
- Regeneration checks cover headers, stubs, inventories, reference PDF,
  specification PDF, and specification Markdown.

Generated files are not a second source of API truth. Edit
`tools/reference/content.py`, review output-parameter policy in
`tools/reference/stub_outputs.json`, and rebuild. The checker rejects stale files.

## Placeholder behavior

`parallel-mater-command-stubs` is a separate CMake target, OFF by default. It
must not be linked alongside a real implementation of the same entry points.
The base target defines 84 SDK-independent commands; enabling the optional
Vulkan stub source adds the six typed adapter commands. Native CUDA/Metal/D3D12/
Vulkan handle carriers in the common header do not require native SDK headers.

Result-returning operations report `PM_ERROR_FEATURE_NOT_PRESENT`, or
`PM_ERROR_INVALID_ARGUMENT` for a missing required output pointer. No operation
reports successful physics, successful submission, or a completed wait. Singular
outputs are cleared; enumeration counts become zero without writing array
records; allocated command-buffer output slots become null. Failed query result
payloads stay untouched. Void property queries clear valid outputs; cleanup calls
are no-ops because these placeholders never create resources.

The null-instance bootstrap lookup recognizes exactly itself, the version query,
and instance creation. All other lookups return null; no live instance/device is
manufactured. The version query writes zero and reports unsupported, rather
than advertising an implemented version. A simulation-status output reports
unsupported, never a zero that could be misread as successful execution.

This is a linkable migration scaffold, not a conforming instance/device runtime.
Native import, synchronization, allocation, recording and physics conformance
still require the implementation gates in [MIGRATION.md](MIGRATION.md).

## Older C++ API: remaining documentation/design gaps

The new command guide is not a complete reference for the older `World` API.
[legacy-api-inventory.json](legacy-api-inventory.json) records the public headers,
85 shared type names, backend-specific types, and source locations for 82 unique
World method names per backend, three FrameToken methods, and four host utilities.
The following capabilities still need command-API designs or a separate legacy
reference; they were **not silently declared implemented or removed**:

| Existing C++ surface | Command API status |
| --- | --- |
| `World`, `WorldOptions`, `StepOptions`, `FrameToken` | Reorganized conceptually into instance/device, simulation, pipeline, submission and fences; not a source-compatible wrapper |
| `RigidBodyOptions`, rigid state and inputs, triangle meshes | Covered conceptually by geometry, body descriptors and recorded edits/copies; field-level migration and runtime parity remain |
| `RigidConstraintOptions`, limits, springs and motors | Descriptor model exists; `RigidConstraintState` / `read_rigid_constraint_state` has no corresponding command readback |
| `FluidOptions`, `FluidParticle`, fluid GPU views | Liquid material, particle buffers and output copies exist; no drop-in view or option-parity guarantee |
| `ClothOptions`, `ClothBond`, `ClothDeviceView` | No C cloth resource/pipeline/command declarations yet |
| `RopeOptions`, `RopeAttachment`, `RopeDeviceView` | No C rope resource/pipeline/command declarations yet |
| `SoftBodyOptions`, bonds, geometry, surface bindings and views | No C soft-body resource/pipeline/command declarations yet |
| `SmokeOptions`, `SmokeDeviceView` | No C smoke grid/resource/pipeline/command declarations yet |
| Fluid–cloth, fluid–rope, fluid–soft-body, fluid–smoke, smoke–rigid/deformable and deformable–deformable coupling options/IDs | Not represented by the current liquid–rigid/liquid–liquid coupling model |
| `ParticleSource*`, `ParticleDestroyPlane*` | No C emitter or destroy-plane commands yet |
| `PaintField*`, `PaintRule*` | No C paint resources/commands yet |
| `HitBox*`, `ContactEvent`, `RigidContactEvent`, contact views | No equivalent spatial query/contact-event readback API yet |
| `PhysicsDebug*`, `KernelTiming`, detailed `WorldStepTimings` | Aggregate statistics/timestamps exist; debug captures and per-kernel breakdown do not |
| `sample_rope_centerline`, `sample_fluid_source`, `sample_fluid_geometry`, `build_soft_body_geometry` | Existing host utilities; not in the C command contract |
| `DeviceSpan`, `BufferSpan`, `NativeContext`, backend GPU views | Replaced conceptually by explicit buffers and native import interfaces, not ABI-compatible aliases |

The source-level scan found definition or forwarding-macro candidates for every
audited World/FrameToken method and host utility. D3D12 already has **58 World
method names** routed to unsupported stubs in `src/d3d12/stubs.cpp`; those were
preserved, not mistaken for working physics. Metal forwarding macros resolve
through `MetalSystems`, not missing-method stubs.

This legacy scan is name-level evidence, not proof of overload completeness or
backend conformance. Constructors/operators and internal solver helper APIs are
not part of that inventory. CUDA, Metal and D3D12 execution were not rerun for a
documentation/stub change. The stronger compile/link/export checks above apply
to the entire **new C contract**, including its optional typed Vulkan adapter.
