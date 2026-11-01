# F8-70 CPU Gallery repair candidate freeze

F8-70 freezes replacement candidate `realtime-single-step-v1-ba67db3` at
source revision `ba67db35447b1f9022be4c3d8aae59dbe6d514d5` and tree
`2dfc4b680fac97519aee1c67526bddffdd01c474`. Its machine-readable manifest is
[`realtime-candidate-v9.json`](../../conformance/v1/qualification/realtime-candidate-v9.json),
SHA-256 `4ed4ed7a097928944ac5667e8120e2447affda962a258e56c01dfd0d90d9dd4b`.

This is the exact clean final F8-69 repair source. It makes the shared Gallery
contract derive its parameter scene from the device catalog, preserves the live
runtime when an unsupported Fluid replacement is attempted, and names the
missing `particleLiquids` capability. It does not add CPU capability, weaken a
scene, or remove any lifecycle operation or required test.

## Protected closure

`tools/generate_realtime_candidate.py` hashes the immutable source tree and
emits candidate v9. `tools/check_realtime_candidate.py` independently extracts
that revision, verifies every contract input, exact-source CI and CPU repair
artifact, preserves candidate-v8 pass/failure history, and enforces replacement
qualification routing.

| Category | Files | Closure SHA-256 |
|---|---:|---|
| API and ABI | 18 | `f360354da65e282c915245f4316841ad5fd631c3f2123dc70749b18d037527c5` |
| Shared Slang kernels | 39 | `2b36eeaa5c4a3905fa2c6754b94d02008ea58ffc8a8d8d6b8feb77081ee32993` |
| Adapter binding | 49 | `67cc1c19c3c60d3289f3d3c73027f56590e4f086739e847100e02b679200bc67` |
| Runtime | 127 | `396ce6fee3fbe2cffa8009a47ddaf7baf21074da3e41f1cf5827c987d3c938c7` |
| Schedule and telemetry | 6 | `f8bf6420481cab111fcae91a08f3feb4158eb964b9b4feb2715c8fca1713e037` |
| Public contract | 6 | `2306546edbe897e760b4c08703994db48838fb949d33075a00dda03e4d16edd7` |
| Public oracle | 541 | `234ff210221d3fc30e5800d41cb9cc3b0509b9ce99273f02f06bae8122cd48cb` |
| Work budget | 7 | `46e33b1c55fc28891368ee991e1f0572ff02319b644ab0f47594a3e573c4d405` |
| Shared Gallery | 160 | `78d552ba178f5acf969c09c662f63f8fac2d07bb9ed3dc38f21757da0d8f6c80` |

The frozen surface still contains six public domains, eight advertised
interaction modes, 49 capabilities, 30 public cases, 12 scenarios, 37 named
measurements, 26 canonical Slang modules, 234 entries, 32 kernel groups, five
runtime variants and eight physical target rows.

## Exact-source evidence

Manual GitHub Actions run
[`41257022787`](https://github.com/tabutyn/parallel-mater/actions/runs/41257022787)
checked out the exact frozen revision and passed all 17 required jobs. The ten
artifacts have repository-API archive hashes and independently regenerated
extracted-tree hashes in
[`F8-70-PROTECTED-CI.json`](evidence/F8-70-PROTECTED-CI.json), SHA-256
`9f716eeb970ccd27a48e04ede9657862fda2df84b0d23d79d021a9f8c1599eb1`.
Linux, macOS, Windows and iPad evidence report the exact revision with a clean
source.

F8-69 rebuilt the same exact source on ThomasLaptop with the Slang CPU driver.
The clean Release build completed 610 targets. The complete registered CPU
suite passed 112/112 in CTest semantics: 97 passes, the same 15 explicit
unsupported skips, and zero failures. Isolated discovery/launcher passed 3/3;
installed SDK relocation passed 14/14 with the realtime driver required; and
plain plus validation-enabled repeated Gallery create/reset/background/failure
recovery passed. The exact report and raw archive are
[`F8-69-CPU-GALLERY-LIFECYCLE-REPAIR.json`](evidence/F8-69-CPU-GALLERY-LIFECYCLE-REPAIR.json)
and
[`F8-69-CPU-GALLERY-LIFECYCLE-REPAIR.tar.zst`](evidence/F8-69-CPU-GALLERY-LIFECYCLE-REPAIR.tar.zst).

## Reproduction

```sh
git show -s --format='%H %T' ba67db35447b1f9022be4c3d8aae59dbe6d514d5
python3 tools/generate_realtime_candidate.py \
  --ci-evidence docs/roadmap/evidence/F8-70-PROTECTED-CI.json --route
sha256sum conformance/v1/qualification/realtime-candidate-v9.json
python3 tools/check_realtime_candidate.py
python3 tools/roadmap.py check
```

Any later protected-file change requires another candidate; candidate v9 must
never be edited in place.

## Rejected history and routing

Candidate v8 remains byte-immutable at SHA-256
`7ce34cfa8e6258dc529ddeae8ea24e73427b8214f75ea49ea9e38ae42994ec33`.
Its passing F8-25 Linux Vulkan evidence remains immutable at SHA-256
`429b929744ca7245aa31021e2a0850017cbe69dc0e232fa724f1e395c2a7a33b`;
its failing F8-53 CPU evidence remains immutable at SHA-256
`e5e9a172c3bb2874e05b256f31827e7d1e174d9c52e4a4089fd78ba1d21d4741`.
[`realtime-candidate-v8-rejection.json`](../../conformance/v1/qualification/realtime-candidate-v8-rejection.json)
preserves both outcomes and forbids evidence union.

The active pointer and F8-26 through F8-30, F8-53 and F8-68 bind candidate
v9's exact ID, manifest hash, source, tree and CI run. F8-25 remains completed
candidate-v8 history; F8-71 owns replacement-candidate Linux Vulkan
requalification. Candidates v1 through v8 and all their device observations
remain historical provenance only.
