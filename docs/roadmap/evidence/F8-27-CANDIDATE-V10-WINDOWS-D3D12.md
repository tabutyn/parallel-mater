# F8-27 candidate v10 Windows D3D12 qualification

Candidate `realtime-single-step-v1-5bacbe2` fails the Windows D3D12 integration gate. The result uses exact source `5bacbe2b1e0fbfad565fb9477bb6873396e31d5d`, tree `67f5088c2e9f50938c55856e3683326e42675310`, and manifest SHA-256 `29e051d94a89a39d51c2128b212165d60da6da7de4b2c107a72ae7cc5b77f20a` on the registered GTX 1080 with NVIDIA driver 582.66.

## Decisive failure

The frozen contract requires 49 interaction capability rows and forbids capability reduction. D3D12 advertises 46. It omits separate-system equal-material fluid-fluid contact, pressure, and viscosity. The same missing `liquidLiquidCoupling` feature causes two registered tests to skip and rejects the required `six-family-mixed` workload before its first tick.

The original plain capture exits 2 after four workloads complete. Targeted plain and validation executions both reproduce exit 2 with `no physical device supports workload six-family-mixed`. This is a capability and workload-coverage failure. It is not a numeric timing miss covered by F9-07 or F9-08.

## Completed evidence

- Fresh Visual Studio 2022 Release build produced 573 hashed binaries, manifests, reflection records, and embedded DXIL artifacts. All nine frozen source closures match.
- Registered execution passed 135 of 137 tests. The fluid-pair graph lifecycle and fluid-pair quality tests returned the configured skip code 77.
- SDK relocation and independent consumer checks passed 12 of 12.
- Plain mode completed all 30 public cases and executed all 12 scenario fixtures. Aggregate oracle evaluation did not run because sustained workload execution failed.
- `rigid-wall`, `fluid-rigid`, `deformable-mixed`, and `smoke-coupled` each completed 600 warmup and 7,200 measured ticks with zero failed ticks. All 115,200 raw timing values remain in the archive.

| Completed workload | Physics p50 ms | Physics p95 ms | Physics p99 ms |
|---|---:|---:|---:|
| rigid-wall | 44.8260 | 54.9738 | 62.8047 |
| fluid-rigid | 41.2764 | 44.9370 | 48.5544 |
| deformable-mixed | 47.0698 | 49.3522 | 52.1612 |
| smoke-coupled | 33.0553 | 40.9871 | 46.3987 |

These timing samples are retained as partial evidence. They do not qualify performance after the non-timing integration failure.

## Installed Gallery

The managed candidate-v10 installation contains the fresh D3D12 runtime and the hash-verified candidate-v10 CUDA runtime from F8-26. All installed hashes and the Start-menu target were checked.

Twenty-seven of 28 D3D12 headless scenes passed 240 frames with one physics step per frame. PEG PAINT completed 240 frames with zero capacity overflows, then failed with 0 painted texels against 100 required. That scene-quality class remains deferred to F9. Six-frame D3D12 interactive startup completed. The retained interactive image shows the timing overlay on a black scene frame, while the headless rigid image contains rendered scene geometry. A six-frame CUDA check also passed after the combined install.

## Evidence

- [Structured failed qualification report](F8-27-CANDIDATE-V10-WINDOWS-D3D12.json)
- [Raw evidence archive](F8-27-CANDIDATE-V10-WINDOWS-D3D12.zip)

Archive SHA-256 is `0038963bc78cabb1e948ca37ce270e0d878dd5bde3fa4311899774d2732de3dd`. All 204 member hashes and archive CRC checks pass. Archive size is 6,148,107 bytes.
