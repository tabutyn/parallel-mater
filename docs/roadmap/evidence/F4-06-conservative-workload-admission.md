# F4-06: conservative workload-admission evidence

Date: 2026-10-18. Host: Windows 11 x64, NVIDIA GeForce GTX 1080,
Visual Studio 2022/MSVC 19.44.35228, Windows SDK 10.0.26100.0. Claim token
`63dba482-70a6-4630-aaa8-b85e36f10c7b`.

## Shared policy

The device-profile library now evaluates the complete public
`PmMemoryBudget` snapshot. Unknown snapshots deny admission. Known heaps must
have valid confidence and `usage <= budget`; the policy chooses the largest
known headroom, subtracts `borrowedBufferBytes` with saturation, and limits a
workload to 70% of the remainder without overflowing the calculation.

The desktop Gallery queries a fresh snapshot for every rigid-scene decision.
The result therefore follows a falling or recovering live budget and never
reuses a cached available-byte value. Implementation estimates and driver/OS
reports retain their provenance but use the same conservative reserve because
the confidence enum describes source, not a quality ordering.

## Fake-resource qualification

`parallel-mater-gallery-workload-tests` uses constructed hardware snapshots;
it has no native backend, phone model or shader dependency. It proves:

- unknown confidence denies even when numeric fields are nonzero;
- implementation-estimated and driver-reported snapshots are admitted only
  inside the 70% limit;
- every borrowed byte reduces headroom, saturating at zero;
- falling, then recovering, snapshots produce deny, then admit decisions from
  the current observation; and
- over-budget usage and invalid confidence deny as malformed input.

## Windows integration validation

```text
PASS cmake --build build-f4-03-root --config Release --target parallel-mater-gallery parallel-mater-gallery-workload-tests
PASS ctest --test-dir build-f4-03-root -C Release -R parallel-mater-gallery-workload-tests --output-on-failure
PASS Parallel-Mater.exe --backend d3d12 --brick-count 8 --frames 1 --headless <temporary.ppm>
PASS strict MSVC /W4 /WX portable suite (12/12)
```

The final run selected the D3D12 driver, created the shared rigid scene,
reported the NVIDIA GeForce GTX 1080, rendered one headless frame and exited
successfully. It validates that the OS-reported D3D12 budget reaches the common
admission policy. This is an execution/admission result, not new physical or
performance qualification.
