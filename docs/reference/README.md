# Parallel Mater API reference

The public C API is installed as `parallel_mater/parallel_mater.h`. Optional
SDK-typed Vulkan interoperability is declared in
`parallel_mater/interop_vulkan.h`.

The documentation has two forms:

- [Reference guide](parallel-mater-reference.pdf): compact declarations,
  tables, diagrams, and specification section markers.
- [API specification](parallel-mater-specification.pdf): required behavior,
  validity, lifetime, synchronization, and failure rules.
- [Specification text](parallel-mater-specification.md): searchable Markdown
  generated from the same source.
- [API coverage](api-coverage.json): every function and type, its source
  location, implementation state, and specification section.

The three-column reference uses continuous colored topic regions with stepped
boundaries. The opening bar contains the wordmark, API overview, and color key.
Return codes follow, then Function Pointers and Instancing. Purple section
markers refer to the numbered specification. Explanatory prose belongs in the
specification rather than the compact reference.

## Generate and verify

From the repository root:

```sh
python3 -m venv /tmp/pm-reference-venv
/tmp/pm-reference-venv/bin/pip install -r tools/reference/requirements.txt
/tmp/pm-reference-venv/bin/python tools/reference/build.py
/tmp/pm-reference-venv/bin/python tools/reference/check.py
```

Header and runtime coverage generation does not require the PDF dependencies:

```sh
python3 tools/reference/build.py --headers-only
```

`tools/reference/content.py` is the declaration and specification source.
`tools/reference/model.py` owns section numbering. Generation writes headers,
the two documents, Markdown, and coverage inventories; it never writes runtime
implementation files. Diagrams are vector drawing instructions. The transparent
wordmark is the only raster asset.
