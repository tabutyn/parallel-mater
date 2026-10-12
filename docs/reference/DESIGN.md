# Reference design study

## Source material

The user's three photographs show pages 1–3 of the Khronos Vulkan 1.1
Reference Guide. The [official PDF](https://www.khronos.org/assets/uploads/developers/presentations/vulkan11-reference-guide.pdf)
provides the broader reference structure. The local glTF reference is
`/home/thomas/Downloads/gltf20-reference-guide.pdf` (eight pages, revision
0922). It is an input to this design study, not a redistributed project asset.

The Vulkan guide groups declarations into colored topic regions whose
boundaries step around neighboring topics across columns. It distinguishes
functions from types with color and uses short page references. Its first
pages establish instance/device ownership before execution
and synchronization. The compact format is useful during implementation,
without the opening pages of a tutorial. Our guide adopts these information
design choices with original content and its own visual identity.

The glTF guide puts relationship diagrams close to the structures they
explain. Its object graph, buffer/accessor explanation, and animation flow
connect data layout to use. Parallel Mater similarly puts its ownership map,
allocation sequence, physics pipeline, and command lifecycle beside the
relevant declarations. Those drawings remain searchable vector content.

This is an independent Parallel Mater document. Khronos has not endorsed
the API or this reference. No Vulkan/Khronos/glTF logo or page artwork is
reused. The PDF is not a Vulkan specification or a compatibility claim.

## Visual system

- Compact portrait sheets with 8-point outer margins, no cover-only page or running banner.
- A first-page, full-width overview bar holds the wordmark, introduction, and color key.
- Return codes come next, then Function Pointers and Instancing, then devices and queues.
- Continuous pastel topic regions, joined across columns with rounded, stepped outlines.
- Category names print once, not again at every column/page continuation.
- A distinct background for each topic, muted burgundy borders, teal diagram accents.
- Blue functions; green structures/types (including function-pointer types); plum constants and subsection headings. The same name colors apply in prose, tables, declarations, and the opening legend.
- Declarations, compact tables, and diagrams only; no explanatory body prose except the return-code caption.
- Purple `[n.n]` markers identify sections of the long specification, not pages of the guide; only the marker, never the title, links to that specification page.
- A tiny folio and PDF outline; the symbol index uses local guide-page numbers.
- Three-column reading order; long declarations flow between columns, with continuation labels.
- Compact vector figures near the declarations they explain; no separate diagram pages.
- Embedded text fonts: 6.7-point declarations (6.4 for the longest unsplit identifier) and an 8-point linked symbol index.
- Full declarations generated into a C header from the same source as the PDF.
- Typed `PFN_pm*` declarations are generated from each command signature; C/C++ checks verify assignments and loader usage.

The source includes explicit overflow checks. Text is not rasterized into
screenshots, shrunk silently, or clipped to make a panel fit. A generated
layout manifest records text fragments, topic contours, and page occupancy
for verification. The checker also rejects orphan headings and author-only
notes in the reference.

The quick reference and the API specification serve different reading tasks.
The guide is a declaration lookup sheet. The long specification contains all
behavioral prose, validity and lifetime rules, failure handling, synchronization
requirements, and worked examples. It uses numbered chapters and subsections,
a linked contents page, PDF bookmarks, and a symbol index. Body text is 10.2
points and code is 8.2 points. A generated Markdown edition has matching anchors.

Both editions and the C headers use the same source; the checker verifies
declaration coverage, preservation of every explanatory paragraph, consistent
section numbers, and clickable cross-document targets. No runtime behavior
was changed by separating the documents.

Build instructions, source-study notes, implementation progress, and migration
gates stay in companion project documents, outside both consumer PDFs.

## API design basis

Vulkan is a useful model for explicit ownership and scheduling, not a reason
to expose every internal physics dispatch. `pmCmdStepSimulation` describes
one simulation operation; a pipeline selects its material configuration and
iteration policy. Slang remains an implementation language, not a second
public physics API.

The [Vulkan command-buffer chapter](https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html)
informs the record/execute/reset states and the requirement that a pending
buffer and its referenced objects remain alive. Parallel Mater's first
contract intentionally has primary buffers only, with no simultaneous reuse.

The [Vulkan memory chapter](https://docs.vulkan.org/spec/latest/chapters/memory.html)
informs explicit resource requirements, memory-type selection, allocation,
binding, and suballocation. Host allocation callbacks are a separate concern
from device memory. We expose persistent and scratch simulation storage;
wrapping a `World` that secretly allocates these does not satisfy the contract.

The [Vulkan synchronization guide](https://docs.vulkan.org/guide/latest/synchronization.html)
informs the separation between device-to-device dependencies and host
completion. A physics/render handoff needs visibility and ownership as well
as an execution signal. External sharing is a queried capability, not an
assumption based on two APIs using the same vendor's GPU.

The interoperability chapter is API-neutral. Its native mappings draw on
[CUDA external-resource synchronization](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXTRES__INTEROP.html),
[Metal shared events](https://developer.apple.com/documentation/metal/about-synchronization-events),
[DirectX 12 queue waits](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandqueue-wait),
and [Vulkan synchronization and ownership](https://docs.vulkan.org/spec/latest/chapters/synchronization.html).
The common import structs and handoff rules are Parallel Mater's proposed
contract, not native API guarantees. In particular, a CUDA event is not a
timeline counter, a D3D12 fence does not perform resource-state transitions,
and a Vulkan semaphore does not replace queue-family ownership transfers.

## Scope decisions

The first contract covers rigid triangle geometry, rigid constraints,
particle liquids, liquid/rigid and liquid/liquid coupling, and renderer
interop. Cloth, rope, soft bodies, smoke grids, emitters, and paint remain
available in the existing API; they require additional contract extensions
before replacing that API. Nothing silently reinterprets them as liquids.

There is one rigid geometry family. Spheres and boxes are mesh assets, not
alternate body simulators. The pipeline's rigid state configures the unified
AVBD solve for contacts and joints. Solver pass counts describe work, not a
choice between competing rigid-body implementations.

We deliberately do not promise cross-device bitwise determinism, automatic
substeps, zero-copy sharing on all devices, backend feature parity, or
universal fluid material equivalence. Capability queries and migration gates
make those boundaries visible.
