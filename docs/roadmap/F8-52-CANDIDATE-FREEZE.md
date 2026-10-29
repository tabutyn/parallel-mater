# F8-52 complete single-path candidate freeze

> Historical record: F8-61 supersedes this candidate after freezing the
> executable scenario contract and repaired 49-capability/30-case matrix. See
> the [F8-61 freeze record](F8-61-CANDIDATE-FREEZE.md). Evidence from this
> candidate cannot be combined with F8-61 qualification rows.

F8-52 freezes final qualification candidate
`realtime-single-step-v1-5a811c4` at source revision
`5a811c4cba0e09fa9f6faaa2ca1a6f9e6cd59855` and tree
`ce80fdeb6bcff12c418165795bb6a7a42d4de000`. Its machine-readable manifest is
[`realtime-candidate-v4.json`](../../conformance/v1/qualification/realtime-candidate-v4.json),
SHA-256 `f0f15a65fed178625b263c6462e7e0d9397fd4880cfc2a3a73582b98584876ae`.
It is the API 0.3/driver ABI 2 sole fixed-tick SDK and Gallery candidate after
F8-50, F8-51, F8-54, F8-55 and F8-56.

This freeze is not physical qualification. F8-24 through F8-30 own the seven
GPU target rows; F8-53 owns the final CPU/public-contract row.

## Protected closure

The manifest hashes every protected file from the immutable candidate Git tree.
Adding, removing or changing a matching file invalidates the candidate.

| Category | Files | Closure SHA-256 |
|---|---:|---|
| API and ABI | 10 | `80b96574e37d84ab5d3cfaf8c0c1b66b81403ea71975ebf44e78b7989088502d` |
| Shared Slang kernels | 39 | `f1e497658fdd1f30a7e93befa5370011a46e691a9891719ce47f790322145be3` |
| Adapter binding | 45 | `60fea2c2bf3d777bb49c941a9f72ba30e45313aeb2b3d2bb90b40bd26a6f1f79` |
| Runtime | 127 | `4c822d3a4f3adee41d4049cd7cef2ca2df125185325dacb2e472b96cc63d7eac` |
| Schedule and telemetry | 6 | `0998a0778a31f759d568155c2ba9153abaf4ca23be1d2be52259510ab90af6cd` |
| Public contract | 6 | `00f1d73ed2fc64ed77d82a8c1be272187e83adb5d2f9ab510407f6e9dc7f9f5b` |
| Public oracle | 6 | `be2d24f768441a5dfe909602064b0bf921e00353db971a8dda531a1abfd142f3` |
| Work budget | 7 | `5a31208572f609e965b54ff99775c01b26c270680548d098dfff31f667ce5d73` |
| Shared Gallery | 160 | `55374d4a228703313a9b9a2a1b81510b21a9d012977e5f191e3cb0198f1a8bb9` |

The frozen surface contains all six public domains, eight advertised
interaction modes, 49 interaction capabilities and 30 public cases. It binds
26 canonical Slang modules, 225 entries, 32 kernel groups, five runtime
variants and eight physical qualification rows. Capability reduction is not
allowed.

## Evidence identity

GitHub Actions run
[`40787851423`](https://github.com/tabutyn/parallel-mater/actions/runs/40787851423)
passed the cross-platform SDK, shader/reflection, Android Vulkan and iPadOS
Metal build gates. The manifest binds all ten artifact archive and extracted
tree hashes, plus exact macOS, iPadOS and Android Gallery application hashes.
It also binds the merged F8-50, F8-51, F8-54, F8-55 and F8-56 evidence hashes.

The clean installed macOS CPU preflight passed all 14 relocated-SDK,
public-profile and negative checks. The report hash is
`a905ebb71296db229302e7e33c9436bf8085a12ef2d2658cefc424462a732061`;
the production CPU driver hash is
`1be55fa5885fc5127da8f7d0d0a50ffe032419de69132780f7880812c47cc416`.
Plain and validation observation hashes are
`c75b0345824e86ba75c5c9f9acbdefa4d7a9a30c7e809a68cb4c124793d7b4c7`
and `2d3775af5a8ce8f90fd2c1ef74e230baaf1d2c54caa10602522a1f32544fdd42`.
The tracked summary is
[`F8-52-INSTALLED-PREFLIGHT.json`](evidence/F8-52-INSTALLED-PREFLIGHT.json).

## Routing and history

Coordination PR #567 routes F8-24 through F8-30 and F8-53 to this exact
candidate. Evidence union across revisions is forbidden. Candidate v3 and the
completed F8-23 CPU work remain immutable interim provenance; candidates v1
and v2 remain rejected. `tools/check_realtime_candidate.py` enforces all four
candidate identities, protected closures, exact task routing, application and
CI artifacts, prerequisite evidence, capability surface and installed
preflight.
