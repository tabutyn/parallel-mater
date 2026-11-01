# F8-26 Windows CUDA qualification failure

The 2026-11-01 Windows GTX 1080 run **does not qualify candidate v9**.
The clean Release build used CUDA 12.9.86, Slang 2026.18, MSVC 19.44.35228.0
and NVIDIA driver 582.66 on Windows 11 build 26200.

The tested harness is `bf3ba752792cd54f144e0ddf525a05911d5f65d8`, including
the merged F8-72 test-only repair. The production closure and manifest of
`realtime-single-step-v1-ba67db3` remain exact. The manifest SHA-256 is
`4ed4ed7a097928944ac5667e8120e2447affda962a258e56c01dfd0d90d9dd4b`.

| Check | Result |
|---|---|
| Clean Release build | Pass |
| Registered physical/portable suite | 128/130 pass; two crashes; zero skips |
| Installed SDK relocation | 12/12 pass |
| Installed-public capture | Crashes at `soft-body-fluid` after 19 completed cases |
| Validation-mode capture, scenarios, controls and sustained workloads | Not reached |

The registered failures are `parallel-mater-command-gallery-tests` at PEG PAINT
and `parallel-mater-fluid-realtime-cuda-tests`. Both and the installed-public
failure reproduce independently with exception `0xc0000006` and underlying
access status `0xc0000022` on CPU reads of managed-address ranges.

The device reports `concurrentManagedAccess=0`. The fluid fixture reads managed
`*count` after launching a kernel; NVIDIA documents the corresponding
[managed-memory concurrency restriction](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html).
`CUDA_LAUNCH_BLOCKING=1` makes the unchanged fluid fixture pass, but PEG PAINT
still crashes. This control is diagnostic only. Production paint metadata reads
after asynchronous clears are an investigation lead; the exact paint and
soft-body/fluid access sites still need diagnosis.

The [report](evidence/F8-26-CANDIDATE-V9-WINDOWS-CUDA-FAILURE.json) contains
commands, source/device/binary hashes and limitations. The
[raw archive](evidence/F8-26-CANDIDATE-V9-WINDOWS-CUDA-FAILURE.zip) retains full
build/test/SDK logs, partial capture, native exception records and reproduction
helpers. Its SHA-256 is
`0e04f4cd40fa2d762830ed625edc0b0a1acc35e2412d9b0c52cc8161628ae4a2`;
every member is independently listed in `SHA256SUMS.json`.

[Claim #633](https://github.com/tabutyn/parallel-mater/pull/633) was released by
[#637](https://github.com/tabutyn/parallel-mater/pull/637), without completion.
[F8-75](tasks/F8-75.json) owns the fixture repair;
[F8-76](tasks/F8-76.json) owns production repair. Both feed the existing
[F8-74](tasks/F8-74.json) refreeze alongside the Mac repair. A fresh F8-26 lease
must execute the complete replacement candidate; partial observations and
forced-synchronization controls cannot be combined into a pass.
