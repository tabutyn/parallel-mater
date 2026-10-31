# F8-63 post-CUDA-repair candidate freeze

> Historical record: F8-24 later rejected candidate v6 on the mandatory
> installed-public CUDA suite. F8-64 repaired that path and
> [F8-65](F8-65-CANDIDATE-FREEZE.md) froze replacement candidate v7. Candidate
> v6 and this record remain immutable provenance and cannot supply a pass.

F8-63 freezes replacement candidate `realtime-single-step-v1-fc2ffba` at
source revision `fc2ffbafdb81631c4c7393336970cefc3d65f55e` and tree
`fccd57eabd07dd4ae40cd38fdce8020d079a09da`. Its machine-readable manifest is
[`realtime-candidate-v6.json`](../../conformance/v1/qualification/realtime-candidate-v6.json),
SHA-256 `6d418e610601337ed4aa25a3c09d9791bcc4ccbb3fa281a69f45e4c7775fe2c9`.

This is the exact clean F8-62 repair commit. It repairs candidate-v5's three
required Linux CUDA Gallery failures without changing public API/ABI, shared
Slang kernels, runtime physics, fixed-tick scheduling, scenario fixtures,
capture validation, capabilities, public cases, independent work budgets or
quality tolerances.

## Protected closure

`tools/generate_realtime_candidate.py` hashes the immutable source tree and
emits candidate v6. `tools/check_realtime_candidate.py` independently extracts
and hashes the source revision, validates every contract input, verifies the
full CI job/artifact set and enforces exact qualification routing.

| Category | Files | Closure SHA-256 |
|---|---:|---|
| API and ABI | 10 | `80b96574e37d84ab5d3cfaf8c0c1b66b81403ea71975ebf44e78b7989088502d` |
| Shared Slang kernels | 39 | `b53c371b1bae224625fe4d9fd60f3c85b6f3eeb209de75f2a4639d55c1857ec7` |
| Adapter binding | 45 | `3618762ef2849cfdad85a226a49eea98101ab5597986ebc27eec1db9f6b83cae` |
| Runtime | 127 | `2f90a8f0d12f5ee737f9228bc954333cc9cbed74897074cc49ff4649905b91ec` |
| Schedule and telemetry | 6 | `2d7ffc3b4a59fe7df72c4449487b3529062cfd23da0217b89a36811714e2e1e2` |
| Public contract | 6 | `54a9c59f7f815b15c7c55363f3b86be9dcec18a51231affea4491a60c0129192` |
| Public oracle | 11 | `a7820e9ce7b109cb48606ea59898326db3c9b63937079dc1858ab6827197685f` |
| Work budget | 7 | `a503b5c4cb98f2e6912e24f83ffbf567c34d507df0b46839929b7f6fd3c6d8ea` |
| Shared Gallery | 160 | `fb31aebed76d89a1e9806ac0825cd2ba456eac3f16b191ae473e1037cc7ce65f` |

Only five files in the Gallery closure differ from candidate v5. The other
eight closure identities are byte-for-byte unchanged. The frozen surface still
contains six public domains, eight advertised interaction modes, 49
interaction capabilities, 30 public cases, 12 scenarios, 37 measurements, 26
canonical Slang modules, 225 entries, 32 kernel groups, five runtime variants
and eight physical target rows.

## Exact-source execution evidence

Manual GitHub Actions run
[`40982968850`](https://github.com/tabutyn/parallel-mater/actions/runs/40982968850)
checked out the exact frozen revision and passed all 17 required jobs: Linux,
macOS and Windows installed SDK/reproducibility lanes; Linux, macOS and Windows
shader portability; Android and iPadOS Gallery builds; Android UI; portable
timing; Windows loader; and reference reproduction. All ten artifacts have
repository-API archive hashes and independently regenerated extracted-tree
hashes in
[`F8-63-PROTECTED-CI.json`](evidence/F8-63-PROTECTED-CI.json). Linux, macOS,
Windows and iPad evidence all report the exact revision with a clean source.

F8-62 built the same exact source on ThomasLaptop's physical RTX 3050 Ti and
passed the complete 130/130 registered suite, including 42 CUDA-labeled tests,
plus all 12 relocated-SDK checks. The unchanged 0.03 brick stability limit,
100-degree dump-truck target and headless manifest discovery contract all pass.
That repair evidence makes the source eligible to freeze; it does not complete
F8-24 or any other final qualification row.

## Reproduction

From a clone containing the immutable source and evidence commits:

```sh
git show -s --format='%H %T' fc2ffbafdb81631c4c7393336970cefc3d65f55e
python3 tools/generate_realtime_candidate.py \
  --ci-evidence docs/roadmap/evidence/F8-63-PROTECTED-CI.json --route
sha256sum conformance/v1/qualification/realtime-candidate-v6.json
python3 tools/check_realtime_candidate.py
python3 tools/roadmap.py check
```

The F8-62 evidence records the physical configure, build, CTest and relocated
SDK commands verbatim. Any later protected-file change requires another
candidate; candidate v6 must never be edited in place.

## Rejected history and routing

Candidate v5 remains byte-immutable at SHA-256
`3435be0a4d4caa2d014d82faba7272aaadeba546a555a2e35a77ff18d0d8a000`.
Its F8-24 failure record remains byte-immutable at SHA-256
`529f43bf26c36cd989c8bd72c936ecdd1306b372d70d194071b71f5c0e9453dd`.
[`realtime-candidate-v5-rejection.json`](../../conformance/v1/qualification/realtime-candidate-v5-rejection.json)
preserves the failure and explicitly forbids evidence union.

The active pointer and F8-24 through F8-30 plus F8-53 bind candidate v6's
exact ID, manifest hash, source, tree and CI run before those tasks become
ready. Candidates v1 through v5 and all prior device observations remain
historical provenance only.
