# F4-03: common resource dependency evidence

Date: 2026-10-18. Host: Windows 11 x64, Visual Studio 2022/MSVC
19.44.35228, Windows SDK 10.0.26100.0. Claim: PR #243, token
`391a8b64-fe19-47df-a436-f94266f79dc9`.

## Contract

The common command list now derives resource ordering from the typed access
ranges introduced by F4-01. Each command is normalized into non-overlapping byte
intervals per device/allocation identity. A native barrier callback is emitted
immediately before a consuming command only when an overlapping prior/next pair
contains a writer. The callback carries no native type and is forbidden from
submitting or waiting on the CPU.

The interval history is replaced only over ranges touched by the new command.
This preserves unresolved hazards on disjoint portions of an allocation while
avoiding repeated dependencies once an access has been ordered. Overlapping
bindings within one dispatch combine into read, write or read-write intervals;
they do not depend on themselves. Dispatch, copy, fill and upload accesses all
use the same planner.

Metal remains a thin adapter: separate compute and blit encoders are already
ordered within its command buffer, so it validates the common dependency and
uses native encoder ordering without a CPU wait. No Slang physics, schedule,
public API or backend-specific physics implementation changed.

## Portable acceptance evidence

Configured and built the standalone solver tests without a GPU or shader
toolchain:

```powershell
cmake -S tests/solver -B build-f4-03-portable -G 'Visual Studio 17 2022' -A x64
cmake --build build-f4-03-portable --config Release --target `
  parallel-mater-compute-dependency-tests `
  parallel-mater-compute-contract-tests `
  parallel-mater-compute-lifetime-tests --parallel 4
ctest --test-dir build-f4-03-portable -C Release --output-on-failure `
  -R '^parallel-mater-compute-(dependency|contract|lifetime)-tests$'
```

PASS: 3/3. The dedicated dependency mock verifies exact RAW, WAR and WAW
intersections; read/read, different-allocation and disjoint-range elision;
partial-range history; same-command alias normalization; barrier-before-consumer
ordering; validation recovery; explicit copy/fill/upload access semantics; and
zero native completion waits during barrier planning, submission and nonblocking
collection.

The first lifetime run exposed that dependency history still retained buffers
after submission. Clearing that recording-only history during command-list
handoff fixed the failure; the unchanged lifetime assertion now passes and
prevents regression.

## Windows D3D12 integration evidence

Configured the repository build with D3D12 enabled, CUDA and the gallery
disabled, and the pinned Slang 2026.18 compiler:

```powershell
cmake -S . -B build-f4-03-root -G 'Visual Studio 17 2022' -A x64 `
  -DBUILD_TESTING=ON -DPARALLEL_MATER_BUILD_CUDA=OFF `
  -DPARALLEL_MATER_BUILD_D3D12=ON -DPARALLEL_MATER_BUILD_GALLERY=OFF `
  -DPARALLEL_MATER_BUILD_CONFORMANCE=OFF `
  -DPARALLEL_MATER_BUILD_SCENE_SDK=OFF `
  -DPARALLEL_MATER_SLANGC_EXECUTABLE=<pinned-slangc>
cmake --build build-f4-03-root --config Release --parallel 4 --target `
  parallel-mater-compute-dependency-tests `
  parallel-mater-compute-contract-tests `
  parallel-mater-compute-lifetime-tests `
  parallel-mater-driver-slang-d3d12 `
  parallel-mater-d3d12-command-api-tests
ctest --test-dir build-f4-03-root -C Release --output-on-failure `
  -R 'parallel-mater-(d3d12-command-api|compute-(contract|dependency|lifetime)|slang-architecture)-tests'
```

PASS: 5/5. This compiles the shared resource contract into the D3D12 driver and
checks the D3D12 command path alongside the portable contract, dependency and
lifetime suites and the Slang architecture boundary.

Repeated the dependency and lifetime suites with
`--repeat until-fail:100`: PASS, 200/200 executions. After adding the explicit
copy/fill/upload checks, the dependency suite separately passed another 100/100
executions.

## Strict-warning evidence

Configured the standalone suite with MSVC `/W4 /WX /wd4458 /EHsc`, where
`/wd4458` suppresses the pre-existing `Kernel::name` shadow warning. Built the
dependency, contract and lifetime targets and ran their focused CTest set.

PASS: 3/3. The dependency implementation introduces no warning under the
repository's existing exception and shadow-warning policy.

## Limits

This evidence qualifies dependency derivation within the common command-list
boundary. It does not qualify borrowed devices/queues, cross-queue handoff,
engine-provided prior/next completions, physical solver quality or performance.
Those remain separate roadmap gates, principally F4-07 and F4-08.
