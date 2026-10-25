# F8-38 repaired realtime candidate freeze

F8-38 freezes candidate `realtime-single-step-v1-3671f18` from merged F8-37
repair revision `3671f180ed951d8a62cb34e5ec4ce1dab52cd560` and tree
`785513efa475e8b3b1010a79cb2142e33463089a`. The machine-readable manifest is
[`realtime-candidate-v3.json`](../../conformance/v1/qualification/realtime-candidate-v3.json),
SHA-256 `d6d84f85a87af71fcdb0ee5c335f6c3979a741a151b3fbb4151096dd588d4a04`.
It is frozen but not physically qualified; all eight device rows remain pending
F8-23 through F8-30.

## Protected closure

The manifest freezes every executable and public-oracle category using SHA-256
over each sorted repository-relative path, NUL, file contents and NUL. Candidate
selection in the oracle now names v3; it changes no runtime, kernel, profile or
API behavior.

| Category | Files | Closure SHA-256 |
|---|---:|---|
| Kernel manifest and Slang sources | 39 | `dd23174e4ace56067257446a16b14466066f77b2d29ddd1c4ca7f8bdef8e3bf2` |
| Runtime and physics/spatial implementation | 17 | `c2e07b9f00eeb4b428610f76ebc2629b9fafad4d916825cc48c1ebfc9f33bc0b` |
| Public/native adapter binding | 8 | `46107f17d2f9772175684f244d1b96d097d4136b9f599fd386137fce59af150d` |
| Device and dispatch paths | 104 | `15930916e0b0f9d588c0267dd89162be8223d44c56ca39e6550ae43ca9256fda` |
| Realtime schedule | 3 | `9023c3ca413ce39203d307845743696296ddcab4d2d9f352ff7ec1cd8d964b1b` |
| Realtime telemetry | 2 | `a4940cff8657e4f4bbd45cb9abab4bc4566486d523c04776c87877edd931763f` |
| Profile and obligations | 3 | `dd0e8beea70031006525dabf9fb8421f3848b58b2da6a65ae2681ce7d8f7a75e` |
| Public oracle and installed execution case | 4 | `0e2c7413836247a5a82d1e417cfd2040bd6e7c0515dbf92b5707ab2ee234e0bc` |

`tools/check_realtime_candidate.py` verifies the exact source identity, closure
hashes, 26 modules, 225 entries, 32 kernel groups, five runtime variants, three
compile-only variants, eight physical rows, both immutable rejected predecessors
and F8-23--30 routing. Negative controls alter every closure hash and the source
revision/tree, proving drift rejection.

## Build and installed execution evidence

GitHub Actions run
[`40200762911`](https://github.com/tabutyn/parallel-mater/actions/runs/40200762911)
passed Linux, Windows and macOS SDK and shader/reflection gates, Android Vulkan,
iPadOS Metal, Windows loader, mobile UI and timing jobs. Candidate v3 records the
eight relevant artifact IDs, GitHub SHA-256 digests, extracted file counts and
extracted-tree hashes. It also binds rebuilt local CPU, CUDA and Vulkan drivers,
the CI CPU driver, Android APK and both iPadOS framework binaries.

The relocated installed public SDK passed all 14 gate stages and 22/22 external
consumer tests. Its exact regression uses eight live/capacity rigid bodies,
eight live/capacity constraints and contact capacity 2,048. The formerly
overflowing tick now completes with 20 dispatches, 1,421 workgroups, all eight
rigid contact/joint sweeps, one spatial build/publication, one submission, one
time advance and no intra-tick host readback. All copied public states are finite.
The tracked record is
[`F8-38-INSTALLED-CPU.json`](evidence/F8-38-INSTALLED-CPU.json).

This is candidate executability and freeze evidence, not F8-23 physical quality
or performance qualification. GPU timing and physical device rows remain owned
by their separate qualification tasks.

## Rejection and routing

Candidate v2 remains byte-immutable at SHA-256
`f4f1c7807dd50b07857c568b9df3e8e2ccd6fd3870907238dc9dcfab45b493af`.
Its separate
[`realtime-candidate-v2-rejection.json`](../../conformance/v1/qualification/realtime-candidate-v2-rejection.json)
binds the first-tick capacity failure, invalidates all eight physical rows and
forbids evidence union. Candidate v1 and its rejection record also remain
immutable history; the validator checks the complete v1-to-v2-to-v3 chain.

F8-23 through F8-30 depend on F8-38 and name only
`realtime-single-step-v1-3671f18` as accepted. Every qualification report must
bind its source revision, manifest hash, native/deployment hash and protected
closure hashes. F8-99 depends on both F8-37 and F8-38; any later protected change
requires another repair/refreeze leaf and requalification of affected rows.
