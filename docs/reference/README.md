# Parallel Mater command API reference

This is the **0.2 design draft**, not a description of a released C API.
The current executable library still exposes the C++ `World` interfaces.
The C contract is available as `parallel_mater/command.h`, with an optional
`parallel_mater/command_vulkan.h` typed adapter. `api/` contains identical
reference copies. Every declared entry point now has a linkable placeholder
in the **opt-in `parallel-mater-command-stubs` library**, not a working solver.
It reports unsupported operations; it never creates a device or submits work.
Nothing here switches the gallery to a new backend.

The documentation has two distinct roles: a compact declaration reference
and a long API specification containing explanations and requirements.
Availability and migration status live in this README and `MIGRATION.md`,
not in either consumer document. The reference's three-column sheets use continuous colored
topic regions with stepped boundaries, compact diagrams, and a linked index.
The opening bar contains the wordmark, API overview, and color key.
Return codes follow, then Function Pointers and Instancing. Purple `[n.n]`
markers identify numbered sections in the specification and link directly
to their pages. Keep both PDFs in the same directory for those links; a PDF
viewer may ask before opening another local document. The numbers also work
as printed cross-references.
Only the purple markers are clickable, not their surrounding headings.
Each category title appears once; its colored region continues across columns.
The reference contains no explanatory body paragraphs except its one-line
return-code caption. Validity, lifetime, synchronization, failure behavior,
and examples belong in the specification, also generated as searchable Markdown.
Every command has a generated `PFN_pm*` typedef; the instance/device lookup
functions and version query are covered by the same contract and stub audit.
Interoperability covers CUDA, Metal, DirectX 12, and Vulkan through common
native-object imports and optional typed Vulkan adapters. These declarations
are requirements for future implementations, not claims of backend availability.

The proposed API separates device discovery, explicitly bound simulation
storage, immutable physics pipelines, command recording, queue submission,
and synchronization. Rigid contacts and joints use one AVBD geometry solver;
an application does not choose CUDA/Metal variants or a second rigid model.

## Read and rebuild

- [Reference guide](parallel-mater-reference.pdf): declarations, tables, and section markers.
- [API specification](parallel-mater-specification.pdf): numbered behavior, rules, and examples.
- [Specification text](parallel-mater-specification.md): the same content in Markdown.
- [C contract](api/parallel_mater.h): generated from the same declarations.
- [Design study](DESIGN.md): reference analysis and design decisions.
- [Runtime migration](MIGRATION.md): implementation boundaries and test gates.
- [API audit](API-AUDIT.md): repaired omissions, stub coverage, and remaining legacy gaps.
- [Symbol inventory](api-coverage.json): every C function/type and its specification section.
- [Legacy inventory](legacy-api-inventory.json): C++ types, methods, and definition locations.
- [Wordmark](assets/parallel-mater-wordmark.png): transparent text logo.

From the repository root:

```sh
python3 -m venv /tmp/pm-reference-venv
/tmp/pm-reference-venv/bin/pip install -r tools/reference/requirements.txt
/tmp/pm-reference-venv/bin/python tools/reference/build.py
/tmp/pm-reference-venv/bin/python tools/reference/check.py
```

`tools/reference/content.py` is the editable source for declarations and API
behavior. `model.py` defines common section numbering and the compact-reference
selection. The builder emits both PDFs, the Markdown specification, the C headers,
stub translation units, and audit inventories. `stub_outputs.json` records the
reviewed output parameters; generation fails when a signature changes until
its failure-output policy is reviewed. PDF
diagrams are vector drawing instructions, not generated pictures. Only the
wordmark is an image-generation asset. Do not edit generated files directly.

The builder uses Lato and DejaVu Sans Mono from system font directories; pass
`--font-dir` for a directory containing those four font files on other hosts.
The checker compiles examples as C11 and C++17, checks declaration coverage,
consumer-facing PDF text, links, fonts, page bounds, section geometry,
page occupancy, heading placement, and deterministic regeneration. It also
checks that every explanation survives in the specification, that none leak
back into the reference body, and that purple markers target the correct pages.
It compiles **and links** C/C++ callers of every entry point, verifies exported
symbol coverage, compiles every public type, and executes fail-closed stub tests.
Those are contract/stub checks, not physics or native-interoperability conformance.

## Build the placeholder library

The target is separate from all production backends and defaults OFF:

```sh
cmake -S . -B build-command-stubs \
  -DPARALLEL_MATER_BUILD_CUDA=OFF \
  -DPARALLEL_MATER_BUILD_METAL=OFF \
  -DPARALLEL_MATER_BUILD_D3D12=OFF \
  -DPARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS=OFF \
  -DPARALLEL_MATER_BUILD_SLANG_MIRRORS=OFF \
  -DPARALLEL_MATER_BUILD_COMMAND_STUBS=ON \
  -DBUILD_TESTING=ON
cmake --build build-command-stubs --target parallel-mater-command-stubs-test
ctest --test-dir build-command-stubs -R command-api-unsupported-stubs --output-on-failure
```

The common API needs no GPU SDK. Add
`-DPARALLEL_MATER_BUILD_VULKAN_COMMAND_STUBS=ON` to include the six typed Vulkan
adapter functions; that option needs Vulkan headers but neither a loader nor GPU.
The installed CMake target is `ParallelMater::parallel-mater-command-stubs`.
Do not link it alongside a future real implementation of the same C entry points.
