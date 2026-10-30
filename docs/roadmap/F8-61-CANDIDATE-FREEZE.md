# F8-61 repaired single-step candidate freeze

> Historical candidate record. Candidate v5 failed the required physical Linux
> CUDA suite recorded by F8-24 and is durably rejected by F8-63. Final
> qualification must use the replacement named by the
> [F8-63 freeze record](F8-63-CANDIDATE-FREEZE.md); candidate-v5 observations
> cannot be unioned with replacement evidence.

F8-61 freezes final qualification candidate
`realtime-single-step-v1-cfb104b` at source revision
`cfb104b1c530f859d9a1c3549d2e67c7de31c289` and tree
`e6838f954384363e79a4485f3c2cfb6ff5e0ea79`. Its machine-readable manifest is
[`realtime-candidate-v5.json`](../../conformance/v1/qualification/realtime-candidate-v5.json),
SHA-256 `3435be0a4d4caa2d014d82faba7272aaadeba546a555a2e35a77ff18d0d8a000`.

This freeze follows the executable F8-59 scenario fixtures and F8-60 repair of
the complete 49-capability, 30-public-case matrix. It is not physical device
qualification. F8-24 through F8-30 own the seven GPU rows; F8-53 owns the final
CPU/public-contract row.

## Protected closure

The manifest hashes every protected file from the immutable source tree. The
scenario fixtures and device-capture validator are part of the public-oracle
closure. Adding, removing or changing any matching file requires a new
candidate and requalification of affected rows.

| Category | Files | Closure SHA-256 |
|---|---:|---|
| API and ABI | 10 | `80b96574e37d84ab5d3cfaf8c0c1b66b81403ea71975ebf44e78b7989088502d` |
| Shared Slang kernels | 39 | `b53c371b1bae224625fe4d9fd60f3c85b6f3eeb209de75f2a4639d55c1857ec7` |
| Adapter binding | 45 | `3618762ef2849cfdad85a226a49eea98101ab5597986ebc27eec1db9f6b83cae` |
| Runtime | 127 | `2f90a8f0d12f5ee737f9228bc954333cc9cbed74897074cc49ff4649905b91ec` |
| Schedule and telemetry | 6 | `2d7ffc3b4a59fe7df72c4449487b3529062cfd23da0217b89a36811714e2e1e2` |
| Public contract | 6 | `54a9c59f7f815c7c55363f3b86be9dcec18a51231affea4491a60c0129192` |
| Public oracle | 11 | `a7820e9ce7b109cb48606ea59898326db3c9b63937079dc1858ab6827197685f` |
| Work budget | 7 | `a503b5c4cb98f2e6912e24f83ffbf567c34d507df0b46839929b7f6fd3c6d8ea` |
| Shared Gallery | 160 | `d5b61b13af014116e2f1c507fcf058bfc158b59b8bdec857c0458a6614913d21` |

The surface contains all six public domains, eight advertised interaction
modes, 49 interaction capabilities, 30 public cases, 12 executable scenarios
and 37 independent measurements. It binds 26 canonical Slang modules, 225
entries, 32 kernel groups, five runtime variants and eight physical target
rows. Capability reduction is not allowed.

## Evidence identity

Manual GitHub Actions run
[`40906474584`](https://github.com/tabutyn/parallel-mater/actions/runs/40906474584)
executes the exact protected source on Linux, macOS, Windows, Android and
iPadOS. The manifest binds each artifact archive and independently regenerated
extracted-tree hash. Compile and hosted-runner results do not claim physical
device qualification.

The installed Linux CPU preflight and F8-58 contract acceptance are recorded
in [`F8-61-CPU-PREFLIGHT.json`](evidence/F8-61-CPU-PREFLIGHT.json). They prove
that the frozen contract is executable through the installed public API; F8-53
still owns final physical CPU qualification.

## Routing and history

`active-realtime-candidate.json` integrity-checks and selects this one
manifest. F8-24 through F8-30 and F8-53 bind its exact identity before
qualification. Candidate v4 is preserved as superseded history, F8-23 and
candidate v3 remain interim provenance, and candidates v1 and v2 remain
rejected. Evidence from different candidates may never be combined into a
pass. `tools/check_realtime_candidate.py` enforces the source closure, contract
inputs, CI artifacts, preflight, history and task routing.
