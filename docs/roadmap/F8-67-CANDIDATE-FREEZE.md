# F8-67 Linux Vulkan repair candidate freeze

F8-67 freezes replacement candidate `realtime-single-step-v1-c05d320` at
source revision `c05d3207138c6e969e1347558c08601aeaf1cc07` and tree
`340ab85997363f328e25f1c549114c5b44571bc3`. Its machine-readable manifest is
[`realtime-candidate-v8.json`](../../conformance/v1/qualification/realtime-candidate-v8.json),
SHA-256 `7ce34cfa8e6258dc529ddeae8ea24e73427b8214f75ea49ea9e38ae42994ec33`.

This is the exact clean final F8-66 repair source. It repairs candidate v7's
mandatory physical Linux Vulkan Gallery, installed-public, smoke, paint and
performance failures while preserving API 0.3, driver ABI 2, the sole fixed
tick, independent work budgets and quality tolerances.

## Protected closure

`tools/generate_realtime_candidate.py` hashes the immutable source tree and
emits candidate v8. `tools/check_realtime_candidate.py` independently extracts
that revision, validates every contract input, verifies the exact CI and
physical artifacts, preserves candidate-v7 pass/failure history, and enforces
replacement qualification routing.

| Category | Files | Closure SHA-256 |
|---|---:|---|
| API and ABI | 18 | `f360354da65e282c915245f4316841ad5fd631c3f2123dc70749b18d037527c5` |
| Shared Slang kernels | 39 | `2b36eeaa5c4a3905fa2c6754b94d02008ea58ffc8a8d8d6b8feb77081ee32993` |
| Adapter binding | 49 | `67cc1c19c3c60d3289f3d3c73027f56590e4f086739e847100e02b679200bc67` |
| Runtime | 127 | `396ce6fee3fbe2cffa8009a47ddaf7baf21074da3e41f1cf5827c987d3c938c7` |
| Schedule and telemetry | 6 | `f8bf6420481cab111fcae91a08f3feb4158eb964b9b4feb2715c8fca1713e037` |
| Public contract | 6 | `2306546edbe897e760b4c08703994db48838fb949d33075a00dda03e4d16edd7` |
| Public oracle | 539 | `d1a4aead1f13062adbf7a743edc62051883c814c46bd9091c972e277eb4ac369` |
| Work budget | 7 | `46e33b1c55fc28891368ee991e1f0572ff02319b644ab0f47594a3e573c4d405` |
| Shared Gallery | 160 | `26c26be570dc57af1b3d56d8685957d9c645ca899acd0a3e77b6a924acc42f38` |

The frozen surface still contains six public domains, eight advertised
interaction modes, 49 capabilities, 30 public cases, 12 scenarios, 37 named
measurements, 26 canonical Slang modules, 234 entries, 32 kernel groups, five
runtime variants and eight physical target rows.

## Exact-source evidence

Manual GitHub Actions run
[`41246222115`](https://github.com/tabutyn/parallel-mater/actions/runs/41246222115)
checked out the exact frozen revision and passed all 17 required jobs. The ten
artifacts have repository-API archive hashes and independently regenerated
extracted-tree hashes in
[`F8-67-PROTECTED-CI.json`](evidence/F8-67-PROTECTED-CI.json), SHA-256
`70aa6ae39fbff25826a5f9f22e32527c9c4fa96ea5eaa5cd600edea5ed0334b7`. Linux, macOS, Windows and iPad evidence report the
exact revision with a clean source.

F8-66 rebuilt the same exact source on ThomasLaptop's physical RTX 3050 Ti
using Vulkan SDK 1.4.341.0 validation. It passed 157/157 registered tests,
including 41 Vulkan-labelled tests and 27 SPIR-V modules, plus all 12 relocated
SDK checks. The shared Gallery session completed in 364.69 seconds under the
unchanged 600-second limit; ClothPaint and all five smoke scenes passed their
120-frame gates. The installed public capture passed 49 capability rows, 30
cases, 12 scenarios, 37 measurements, five negative controls and five workloads
across 72,000 measured ticks with zero failed ticks. The exact report and raw
artifact archive are
[`F8-66-LINUX-VULKAN-REPAIR.json`](evidence/F8-66-LINUX-VULKAN-REPAIR.json)
and
[`F8-66-LINUX-VULKAN-REPAIR.tar.zst`](evidence/F8-66-LINUX-VULKAN-REPAIR.tar.zst).

## Reproduction

```sh
git show -s --format='%H %T' c05d3207138c6e969e1347558c08601aeaf1cc07
python3 tools/generate_realtime_candidate.py \
  --ci-evidence docs/roadmap/evidence/F8-67-PROTECTED-CI.json --route
sha256sum conformance/v1/qualification/realtime-candidate-v8.json
python3 tools/check_realtime_candidate.py
python3 tools/roadmap.py check
```

Any later protected-file change requires another candidate; candidate v8 must
never be edited in place.

## Rejected history and routing

Candidate v7 remains byte-immutable at SHA-256
`15f168583cdd16afbf0fa3d7a26a86349dab206e6aab87e9b3f1b18bdcbb8b8f`.
Its passing F8-24 Linux CUDA evidence remains immutable at SHA-256
`49a2e57dd43af1a5735e3b9838d6280de80d4f688ff61ca56035ecc173881764`;
its failing F8-25 Linux Vulkan evidence remains immutable at SHA-256
`2adbc20a1ef42d2d6f2c0f3f0e4d977aeeb14d65f72219fafac42cad9efbf6db`.
[`realtime-candidate-v7-rejection.json`](../../conformance/v1/qualification/realtime-candidate-v7-rejection.json)
preserves both outcomes and forbids evidence union.

The active pointer and F8-25 through F8-30 plus F8-53 bind candidate v8's exact
ID, manifest hash, source, tree and CI run. F8-24 remains completed historical
candidate-v7 qualification; F8-68 owns replacement-candidate Linux CUDA
requalification. Candidates v1 through v7 and all prior device observations
remain historical provenance only.
