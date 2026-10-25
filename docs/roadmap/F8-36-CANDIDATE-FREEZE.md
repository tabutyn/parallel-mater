# F8-36 executable realtime candidate freeze

Historical status: F8-23 rejected this candidate after the exact CPU row
overflowed its telemetry envelope. The immutable manifest remains as evidence;
the rejection record and active repaired replacement are documented in
[F8-38](F8-38-CANDIDATE-FREEZE.md).

F8-36 freezes replacement candidate `realtime-single-step-v1-fd78fa8` at
source revision `fd78fa83048a57ba5ad2e62827daf76cb94d62c1` and tree
`f4b06abaf775c2016b6938bac3a9d8cbb0f2c4ba`. The F8-35 merge commit has the
same tree. The machine-readable manifest is
[`realtime-candidate-v2.json`](../../conformance/v1/qualification/realtime-candidate-v2.json),
SHA-256 `f4f1c7807dd50b07857c568b9df3e8e2ccd6fd3870907238dc9dcfab45b493af`.
It was frozen but never physically qualified; all eight device rows were
invalidated and rerouted to the F8-38 replacement.

## Protected closure

The manifest records the sorted path-and-content SHA-256 identity of every
protected category. Patterns are part of the immutable manifest, so a new file
matching a protected pattern also invalidates the candidate.

| Category | Files | Closure SHA-256 |
|---|---:|---|
| Kernel manifest and Slang sources | 39 | `dd23174e4ace56067257446a16b14466066f77b2d29ddd1c4ca7f8bdef8e3bf2` |
| Runtime and physics/spatial implementation | 17 | `a22dacdaf23645ffd6fdb7aed07d44590e042014d047aed19706414c4c6c36b2` |
| Public/native adapter binding | 8 | `2da5dbdf4c260dab642c251a2af2e4507dcdc65d62a7a449b7307f616a300ed5` |
| Device and dispatch paths | 104 | `15930916e0b0f9d588c0267dd89162be8223d44c56ca39e6550ae43ca9256fda` |
| Realtime schedule | 3 | `9023c3ca413ce39203d307845743696296ddcab4d2d9f352ff7ec1cd8d964b1b` |
| Realtime telemetry | 2 | `a4940cff8657e4f4bbd45cb9abab4bc4566486d523c04776c87877edd931763f` |
| Profile and obligations | 3 | `dd0e8beea70031006525dabf9fb8421f3848b58b2da6a65ae2681ce7d8f7a75e` |
| Public oracle and installed execution case | 4 | `4916b56e34e89d5fbe65a3d7b6555df2bc6d22d2ffb1ff3491f4c7218552b097` |

`tools/check_realtime_candidate.py` recalculates every closure and validates
the profile, all 26 modules, 225 entries, 32 kernel groups, five runtime
variants, three compile-only variants, eight physical rows and exact task
routing. Its negative-control test changes each expected closure hash in turn
and proves runtime, adapter, dispatch, schedule, telemetry, profile and oracle
drift are rejected.

## Executable and cross-platform evidence

GitHub Actions run
[`40192899733`](https://github.com/tabutyn/parallel-mater/actions/runs/40192899733)
passed Linux, Windows and macOS shader compilation/reflection and relocated
component checks, Linux/macOS/Windows SDK install checks, Android Vulkan APK and
iPadOS Metal builds. Every retained archive ID, GitHub archive digest, extracted
file count and extracted-tree digest is in the candidate manifest.

The final immutable manifest was then used to install the production Slang CPU
driver and execute the profile through the relocated public C API. All 14
packaging/execution/negative checks passed. Plain and validation observations
passed the public oracle with SHA-256
`55b4e6c457e9ff2d89ded37494c5ec0f466fe6a892ce7e41eaca002d78fe80e8`
and `0fb6a4ee7c51ac727fed739c41d1ea0f8d77e95766fb71ec9f394aaca70ddc3e`.
Both observations bind the final candidate manifest hash. The installed
production driver SHA-256 is
`4a457b2288c3fe4474a73acc0f602b0eda7bb7153dbcdee3c7f66e9646622a47`;
the tracked execution record is
[`F8-36-INSTALLED-CPU.json`](evidence/F8-36-INSTALLED-CPU.json).

The manifest also binds reviewed native/deployment hashes for local Slang CUDA
and Vulkan drivers, the Android Vulkan APK, and iPadOS ParallelMater and Metal
framework binaries. Windows D3D12 and macOS Metal compiler/reflection/package
success remain build evidence; no hosted build or compile-only mirror is
physical qualification.

## Rejection and routing

Candidate `realtime-single-step-v1-dea47c1` remains byte-immutable at SHA-256
`c25c010be3a0e908537e3c6542c1759b28b10cfb4a4169b51bb2834233411d43`.
Its separate
[`realtime-candidate-v1-rejection.json`](../../conformance/v1/qualification/realtime-candidate-v1-rejection.json)
records the production public-profile and schedule failures, marks all eight
physical rows invalid, forbids evidence union, and routes them to this
replacement. The original manifest and F8-23 rejection report remain history.

F8-23 through F8-30 retain F8-36 as history but also depend on F8-38 and accept
only `realtime-single-step-v1-3671f18`. Evidence from this rejected candidate
cannot be unioned with the replacement. F8-99 depends on both repair/refreeze
leaves and all eight qualification tasks.
