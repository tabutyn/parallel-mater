# F4-05: memory-budget confidence evidence

Date: 2026-10-18. Host: Windows 11 x64, NVIDIA GeForce GTX 1080,
Visual Studio 2022/MSVC 19.44.35228, Windows SDK 10.0.26100.0. Claim token
`806182dd-19d9-4b66-aeb5-d8eeff23997a`.

## Public contract

`PmMemoryBudget` now reports a confidence source for every heap snapshot:

| Driver source | Public confidence |
|---|---|
| no usable report | `UNKNOWN` |
| implementation-derived value | `IMPLEMENTATION_ESTIMATE` |
| native driver report | `DRIVER_REPORTED` |
| operating-system budget report | `OS_REPORTED` |

These labels identify provenance, not a guarantee or quality ordering. Unknown
entries are normalized to zero. Reported and estimated entries are rejected if
available bytes exceed total bytes, so `usage = total - available` cannot wrap.
Physical heap size and resource allocation requirements remain separate API
queries. Driver/OS snapshots remain live observations rather than reservations.

Owned application allocations, implementation-tracked device/host allocations,
and borrowed buffer ranges are separate counters. CUDA import tests prove that a
borrowed range is counted while its wrapper is live and removed at destruction.
Borrowed aliases may overlap, and neither a zero library counter nor heap usage
proves the absence of driver-private/untracked allocations. Checked additions
fail an allocation/import instead of wrapping a public counter.

## Mock SDK qualification

The independently built vendor mock exercises the installed public header and
loader ABI. Its budget modes verify:

- unknown confidence produces zero budget and usage;
- an implementation estimate preserves owned, library and borrowed counters;
- an inconsistent `available > total` snapshot returns
  `PM_ERROR_INVALID_STATE` with cleared output; and
- a backend failure propagates `PM_ERROR_DEVICE_LOST` with cleared output.

The relocation harness was also made generator-independent on Windows: it can
reuse Visual Studio when Ninja is absent, records the SDK from generated project
metadata when CMake caches bare `rc`, and resolves multi-configuration outputs.
The clean install, relocated external driver/consumers, version rejection,
standalone conformance build, installed input check, unsupported case and
missing-driver case all passed.

## Native and portable validation

```text
PASS full Windows Release build
PASS generated API/reference consistency
PASS reflected ABI
PASS installed/relocated SDK contract (all 12 stages)
PASS independent loader/mock budget confidence and failure test
PASS D3D12 command API on GTX 1080 (61.38 s)
PASS strict MSVC /W4 /WX portable suite (12/12)
PASS Python syntax and git diff checks
```

The D3D12 runtime reported `OS_REPORTED`, non-wrapping heap usage and the live
owned allocation total. CUDA is classified `DRIVER_REPORTED`; Vulkan and the
serial CPU adapter are explicitly `IMPLEMENTATION_ESTIMATE`; Metal is
`OS_REPORTED`. Cross-platform compilation and execution remain the corresponding
platform CI responsibility.

## Follow-on

F4-06 may now consume this provenance in the shared workload admission policy.
It must reject unknown snapshots, reserve for borrowed bytes, and handle a
budget changing between decisions; this task intentionally does not choose that
policy.
