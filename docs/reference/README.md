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
- [API routing coverage](api-coverage.json): every function and type, its
  loader source or forwarding route, and specification section. Device support
  is queried from the selected driver at runtime.

The three-column reference uses continuous colored topic regions with stepped
boundaries. The opening bar contains the wordmark, API overview, and color key.
Return codes follow, then Function Pointers and Instancing. Purple section
markers refer to the numbered specification. Explanatory prose belongs in the
specification rather than the compact reference.

## Generate and verify

The release toolchain is pinned by `tools/reference/toolchain.json`: CPython,
Python packages, immutable font sources, extracted font hashes, and licenses.
The font files are cached below ignored `out/reference-toolchain/`; they are not
read from the host operating system.

Install CPython 3.13.15, then regenerate and independently verify the complete
artifact set from the repository root with one command:

```sh
python tools/reference/bootstrap.py --build --check
```

Verification also requires a C11/C++17 compiler. On Windows, run the command
from a Visual Studio Developer PowerShell so `cl.exe` is available.

`bootstrap.py` creates an ignored virtual environment, installs only the exact
package versions in `requirements.txt`, downloads immutable font inputs, checks
every hash, and runs generation/checking with that environment. Use
`--skip-vulkan-header` only on a host without Vulkan SDK headers; this skips the
optional header compile probe, not artifact generation or comparison.

For verification without updating tracked artifacts:

```sh
python tools/reference/bootstrap.py --check --skip-vulkan-header
```

The checker generates the full set below a clean temporary root and byte
compares both header copies, searchable text, API coverage, both PDFs, both
layout manifests, and `release-artifacts.json`. A mismatch names every changed
artifact. The release manifest binds those files and the locked toolchain to
one SHA-256 revision of `registry/pm.xml`; `known_nondeterminism` records any
accepted exception and is currently empty.

Header and runtime coverage generation does not require the PDF dependencies:

```sh
python3 tools/reference/build.py --headers-only
python3 tools/reference/build.py --text-only
```

`registry/pm.xml` is the declaration and ABI source of truth, including stable
structure-type values. `tools/reference/content.py` owns reference layout text,
and `tools/reference/model.py` owns section numbering. Generation writes
headers, the two documents, Markdown, and routing inventories; it never writes
runtime implementation files. Diagrams are vector drawing instructions. The
transparent wordmark is the only raster asset.

`--text-only` updates searchable specification text and coverage as well as the
headers. CI performs a clean full regeneration, including the PDF and layout
artifacts, on every pull request and main-branch update.
