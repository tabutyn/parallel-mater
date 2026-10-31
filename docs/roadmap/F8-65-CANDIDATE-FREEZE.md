# F8-65 installed-public CUDA repair candidate freeze

F8-65 freezes replacement candidate `realtime-single-step-v1-05f8b04` at
source revision `05f8b04a14fc34cfb1ba322f363ea7165fcf3d48` and tree
`e99f19d6af7ddca416b68b0a9a59755a738345f0`. Its machine-readable manifest is
[`realtime-candidate-v7.json`](../../conformance/v1/qualification/realtime-candidate-v7.json),
SHA-256 `15f168583cdd16afbf0fa3d7a26a86349dab206e6aab87e9b3f1b18bdcbb8b8f`.

This is the exact clean final F8-64 source. It repairs candidate v6's mandatory
installed-public Linux CUDA failures and the final generated D3D12 binding
coverage needed by protected CI. Candidate v7 preserves API 0.3, driver ABI 2,
the sole fixed-tick schedule, independent budgets and quality tolerances.

## Protected closure

`tools/generate_realtime_candidate.py` hashes the immutable source tree and
emits candidate v7. `tools/check_realtime_candidate.py` independently extracts
that revision, validates every contract input, verifies the exact CI and
physical artifacts, and enforces qualification routing.

| Category | Files | Closure SHA-256 |
|---|---:|---|
| API and ABI | 18 | `f360354da65e282c915245f4316841ad5fd631c3f2123dc70749b18d037527c5` |
| Shared Slang kernels | 39 | `b5b592446a1bbfd7cc7ba3d834d04e59fc5cde65aee8c7731cb548a3be7b127c` |
| Adapter binding | 49 | `a71570018a172f5f0f9b9947026c2268c5e5879540d165cc5a0c544b2feeba50` |
| Runtime | 127 | `396ce6fee3fbe2cffa8009a47ddaf7baf21074da3e41f1cf5827c987d3c938c7` |
| Schedule and telemetry | 6 | `f8bf6420481cab111fcae91a08f3feb4158eb964b9b4feb2715c8fca1713e037` |
| Public contract | 6 | `2306546edbe897e760b4c08703994db48838fb949d33075a00dda03e4d16edd7` |
| Public oracle | 538 | `06d68b457d87fb2f1f98e049295e9899d82def6d965a0d70bf35ae7a026229f9` |
| Work budget | 7 | `46e33b1c55fc28891368ee991e1f0572ff02319b644ab0f47594a3e573c4d405` |
| Shared Gallery | 160 | `b8d3b5381fd7a4061da4a93fedcebbafcf0286aaeb3037c16b86885e4f66702d` |

The closure now explicitly includes public scene headers, dispatch generators,
all conformance cases and goldens, the installed capture producer and the
validator. The frozen surface contains six public domains, eight advertised
interaction modes, 49 capabilities, 30 public cases, 12 scenarios, 37 named
measurements, 26 canonical Slang modules, 234 entries, 32 kernel groups, five
runtime variants and eight physical target rows.

## Exact-source evidence

Manual GitHub Actions run
[`41213026819`](https://github.com/tabutyn/parallel-mater/actions/runs/41213026819)
checked out the exact frozen revision and passed all 17 required jobs. The ten
artifacts have repository-API archive hashes and independently regenerated
extracted-tree hashes in
[`F8-65-PROTECTED-CI.json`](evidence/F8-65-PROTECTED-CI.json). Linux, macOS,
Windows and iPad evidence report the exact revision with a clean source.

The same source was rebuilt on ThomasLaptop's physical RTX 3050 Ti. It passed
131/131 registered tests, including all 42 CUDA-labelled tests, and all 12
relocated-SDK checks. The installed public capture passed 49 capability rows,
30 cases, 12 scenarios, 37 measurements, five negative controls and five
workloads across 72,000 measured fixed ticks with zero failed ticks. The raw
capture, reports and streams are preserved in
[`F8-65-installed-public-capture.tar.zst`](evidence/F8-65-installed-public-capture.tar.zst)
and bound by
[`F8-65-EXACT-SOURCE-LINUX-CUDA.json`](evidence/F8-65-EXACT-SOURCE-LINUX-CUDA.json).

## Reproduction

```sh
git show -s --format='%H %T' 05f8b04a14fc34cfb1ba322f363ea7165fcf3d48
python3 tools/generate_realtime_candidate.py \
  --ci-evidence docs/roadmap/evidence/F8-65-PROTECTED-CI.json --route
sha256sum conformance/v1/qualification/realtime-candidate-v7.json
python3 tools/check_realtime_candidate.py
python3 tools/roadmap.py check
```

Any later protected-file change requires another candidate; candidate v7 must
never be edited in place.

## Rejected history and routing

Candidate v6 remains byte-immutable at SHA-256
`6d418e610601337ed4aa25a3c09d9791bcc4ccbb3fa281a69f45e4c7775fe2c9`.
Its mandatory installed-public CUDA failure record remains byte-immutable at
SHA-256
`1be323331ac05533be95ec448c5be121604d734c7b3f43185c6c7f58f5465d07`.
The raw F8-64 capture preserves its pre-freeze v6 qualification-input label
with `execution_identity_claimed=false`; F8-65 binds the whole observation set
to one exact v7 source and binary set. It does not relabel or union a v6 pass.

The active pointer and F8-24 through F8-30 plus F8-53 bind candidate v7's exact
ID, manifest hash, source, tree and CI run before those tasks become ready.
Candidates v1 through v6 and all prior device observations remain historical
provenance only.
