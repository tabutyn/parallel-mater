# F8-22 realtime candidate freeze

Historical status: this candidate was rejected during F8-23 because it was not
executable through the production public API. Its manifest remains immutable;
the durable rejection record and active replacement are documented in
[F8-36](F8-36-CANDIDATE-FREEZE.md).

F8-22 freezes candidate `realtime-single-step-v1-dea47c1` at source revision
`dea47c178c3f1308abb54d46942f965b25e87eb6`. Its machine-readable identity is
[`realtime-candidate-v1.json`](../../conformance/v1/qualification/realtime-candidate-v1.json).
The candidate was frozen but never qualified. Its original physical rows are
invalidated and F8-23 through F8-30 may report only the F8-36 replacement.

## Frozen closure

The candidate binds realtime profile 1.0, its workloads/tolerances/budgets,
the public oracle suite, capability obligations, one-tick schedule, telemetry,
the canonical Slang manifest and all 38 Slang sources. The manifest records the
source/tree revisions and SHA-256 for every protected contract plus the complete
Slang closure. F8-22 itself changes only validation, documentation and roadmap
metadata outside that closure.

The canonical build manifest contains 26 modules and 225 entries. F8-01 maps
every entry exactly once into 32 kernel groups. Every group has an existing
execution test and either a legacy public case set or an F8 trajectory case.
The paint auxiliary path is explicitly covered by `f8-paint-contact-stamping`.
All five runtime variants map to public F8 execution cases. CUDA source, PTX and
SPIR-V mirrors remain compile-only and cannot qualify a device.

## Compile matrix

A clean local Release build used CMake 4.1.2, GCC 12.2.0, CUDA 13.1.80 and
Slang 2026.18. All 615 build edges passed, including the production CUDA
adapter, generated CUDA sources for all manifest entries, runtime test binaries,
reflection, and CUDA/PTX/SPIR-V mirrors. The NVIDIA driver was unavailable, so
this is compilation evidence only.

Successful GitHub Actions run
[`40177972578`](https://github.com/tabutyn/parallel-mater/actions/runs/40177972578)
adds native D3D12 on Windows Server 2025/MSVC x64, native Metal on macOS 26 and
iPadOS/Xcode 26, Vulkan in the Android 36/NDK 29 APK, CPU bindings, all mirrors,
reflection and package checks. Durable artifact IDs and extracted-tree hashes
are in the candidate manifest. Its protected closure is identical to this
candidate; intervening commits contain completion/merge/lease metadata only.
Compile success, hosted tests and package creation qualify no physical row.

## Physical matrix

Eight durable device/backend tuples remain, each with one bounded owner:

| Target | Owner | Availability at freeze |
|---|---|---|
| Linux CPU | F8-23 | Available |
| Linux RTX 3050 Ti CUDA | F8-24 | Driver unavailable |
| Linux RTX 3050 Ti Vulkan | F8-25 | No PM Vulkan device exposed |
| Windows GTX 1080 CUDA | F8-26 | Not reachable from freeze host |
| Windows GTX 1080 D3D12 | F8-27 | Not reachable from freeze host |
| macOS M4 Metal | F8-28 | Current access unverified |
| iPadOS M1 Metal | F8-29 | Current paired access unverified |
| Pixel 7 Vulkan | F8-30 | Connected during F8-21; candidate not run |

All eight owners depend on F8-22 and are already F8-99 dependencies. The F8-01
device audit found no additional durable tuple, so no extra task was added. A
later discovery must create a bounded task depending on F8-22 and add it to
F8-99 before exit.

## Invalidation

Any change to shared kernels/entrypoints, native binding or dispatch paths,
physics/coupling equations, schedule/work budgets/telemetry, profile workloads
or tolerances, or public oracle contracts invalidates affected rows. Freeze a
new candidate and rerun them. If an old device card is already complete, create
a separate fix or requalification card. Preserve old evidence as history; never
union green reports from incompatible candidates. F8-99 accepts one internally
consistent candidate only.
