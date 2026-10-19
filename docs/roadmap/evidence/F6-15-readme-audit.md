# F6-15: README and project-state audit

Audit date: 2026-10-19. Source baseline:
`9093b811ddade1b266939a36bf339212a72a7f18`, including the merged P1-58 memory
cleanup and F5-35 CUDA smoke/fluid qualification. This is documentation and
integration-entry-point evidence, not new physical qualification.

## Findings and disposition

The entire previous README was reviewed, including its introduction, design
goals, API links, build commands, driver coverage, rendering description,
platform installation sections, controls, scene descriptions, and CLI examples.

| Previous guidance | Source-backed finding | Disposition |
|---|---|---|
| CUDA described as the first runtime adapter and first installable driver | Four GPU drivers exist; physical-device selection chooses a driver. Chronology does not describe default selection. | Explain loader, installable driver, native adapter, and shared Slang execution separately. |
| Metal had no driver; macOS instructions built CPU physics | `ParallelMaterSlangMetal.cmake` builds a loadable Metal driver. `macos-metal-preview` bundles it; plain macOS CMake still defaults to CPU. | Provide the explicit Metal build, executable, and verified deployment path; retain the distinction from plain CMake defaults. |
| Vulkan covered only rigid/fluid scenes; Android exposed eight scenes | `supported_features()` exposes all six domains on Vulkan. Android uses the common 28-scene catalog and capability-based admission. | Update coverage and link to mobile hardware/build requirements without treating scene availability as qualification. |
| D3D12 summaries disagreed across documents | `supported_features()` exposes rigid bodies/constraints, liquids, and cloth. Rope, soft bodies, smoke, and paint remain unadvertised. | Align README, driver SDK, and Slang summaries with the code. |
| All distinct fluid systems were unsupported | CUDA enables `PARALLEL_MATER_QUALIFIED_EQUAL_FLUID`; exact equal-material records exist. Different-material operators and qualification are separate work. | Describe exact capability queries and avoid generalizing equal-material support to arbitrary fluid pairs. |
| Software presentation was described as the general Gallery path; Slang guide said Metal GPU handoff was unfinished | Metal, D3D12, and Vulkan have native drawing paths using public GPU output. Metal still fences handoff and some controllers/diagnostics read back. | Distinguish physics selection, rendering, native output, and remaining synchronization/import limits. |
| Build example hard-coded a developer laptop's CUDA architecture | The repository has SDK, Metal, D3D12, Vulkan, and iPad presets. | Use named presets, explain `native` for a local CUDA build, and identify cross-machine architecture selection. |
| SDK and driver-author entry points were buried in implementation prose | The minimal SDK installs without a GPU SDK or Slang; an independent C consumer links `ParallelMater::api`. Driver ABI and Slang reference adapter are separate integration levels. | Add an installed SDK quick start, manifest discovery, application lifecycle, and a dedicated driver-author section. |
| Windows installation description implied every preview build replaces the installed app | `gallery-preview-base` disables automatic deployment. The explicit installer target remains available. | Document the preset's actual behavior and exact deployment target. |
| Long scene-by-scene notes, solver experiments, historical failures, and retired-backend statements dominated the landing page | Scene controls, solver details, qualification records, and historical reports already have dedicated documents. | Keep common controls and one timing example in README; link detailed guides and evidence instead of duplicating their narratives. |
| SDK release summary said layer ABI 2 | Installed `pm_layer.h` defines interface version 3; driver interface remains 1 and application API remains 0.2. | Correct the linked SDK summary and add the Metal build option. |

Primary sources inspected:

- `src/api/instance.cpp`, `src/driver/slang/entry.cpp`, and
  `include/parallel_mater/{parallel_mater,pm_driver,pm_layer}.h`.
- `CMakeLists.txt`, `CMakePresets.json`, `cmake/Slang.cmake`, and native-driver,
  loader, macOS, and iOS CMake modules.
- `src/device/metal_compute.mm`, `src/simulation/reference_schedule.hpp`,
  `docs/COMMON-RESOURCE-BOUNDARY.md`, and `docs/INTERACTION-MODES.md`.
- Gallery CLI parsing, common catalog metadata, platform app documentation,
  `scripts/Deploy-Gallery.py`, and `examples/sdk`.
- The generated per-driver block in `docs/QUALIFICATION.md` and its checked
  evidence index, rather than historical narrative below that block.

## Verification

Host: Apple M4 macOS; AppleClang 21.0.0.21000101, CMake 4.3.4, Ninja 1.13.2,
Python 3.14.7. The README's SDK quick start was run in an isolated checkout:

```sh
cmake --preset sdk-minimal
cmake --build --preset sdk-minimal
cmake --install build-sdk-minimal --prefix "$PWD/out/sdk"
cmake -S examples/sdk -B build-sdk-example -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/out/sdk"
cmake --build build-sdk-example
```

All commands passed. The C example links only the installed public SDK. It was
then run against an existing local Metal build's
`Release/manifests/slang-metal.json`, with native device access enabled:

```text
Apple M4 - Metal GPU preview (qualification incomplete) (vendor 4203, device 0)
```

This verifies driver discovery and capability enumeration, not newly built
Metal physics. The sandboxed run exposed no GPU and was not counted as a
physical-device check. An initial invocation pointed at a nonexistent manifest
outside `Release` and returned the expected missing-driver error; the README
now documents the configuration directory explicitly.

Additional checks:

- 53 local Markdown links and heading anchors passed across README and the
  four edited supporting guides. Every README configure/build/test preset name
  resolves in `CMakePresets.json`.
- Gallery flags were checked against the current parser/common catalog and
  the existing Mac executable's `--help`. Windows executable spelling and
  installer commands were checked against CMake definitions.
- `python3 tests/solver/check_slang_architecture.py` passed.
- `python3 conformance/v1/evidence_index.py --check` passed: 2/259 indexed
  capability results qualified across 17 report records at this baseline.
  The two qualified records concern CUDA smoke-to-fluid drag; this does not
  qualify a whole driver. README links the maintained index instead of copying
  its changing counts.
- `python3 -m unittest discover -s tests/roadmap -p 'test_*.py'` passed 28 tests.
- `git diff --check` passed. Task/index validation is also required when
  recording completion.

No new Windows, Linux, Android, iPad, whole-corpus physics, or performance run
is claimed. The changes affect documentation only.

## Remaining project work exposed by the audit

- Physical qualification remains incomplete. Exact lifecycle, regression,
  analytic, and hardware evidence must agree; F5-34 fluid/rope is still open.
- The GPU drivers do not have equal domain/interaction coverage. Broader
  common resource-interface adoption and missing interaction work remain in
  the P1 roadmap. Existing operators do not automatically establish
  qualification or advertisement.
- Native engine-owned device/timeline handoff remains a separate deferred
  P2 scope; Gallery's direct GPU rendering is not that integration.
- Complete Android device and sustained-runtime qualification remains F7-18.
- Detailed guides such as `docs/GALLERY.md` still interleave user instructions
  and dated implementation narratives. A later documentation pass can separate
  those narratives without turning this README back into a development log.

These are retained limits and follow-ups, not features completed by F6-15.
