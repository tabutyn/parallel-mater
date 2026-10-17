# Qualification scope and open gates

Updated 2026-10-16. This is an evidence index, not blanket driver certification.
Each release must attach a revisioned machine-readable evidence package with
the exact source, driver, toolchain, device, case hashes and settings. Repeat the
affected scope after changes; historical passes do not automatically qualify a
new build.

| Scope | Evidence and host scope | Unqualified or failed scope |
|---|---|---|
| SDK/loader/host contracts | 30 shader-free `sdk-ci` CTests; relocated C/C++ consumers and separately built mock driver; standalone conformance checks all 30 inputs | New remote workflow not yet verified; Windows/macOS installed packages not run here; physics not evaluated by this gate |
| Slang CPU | Foundation suite: 34 passed, two unsupported domain skips; exact queried fluid lifecycle/recovery | Mixed particle domains unsupported; `rigid-direct` final angular velocity fails historical CUDA comparison |
| CUDA | Linux public rigid/fluid/cloth command API after scene SDK extraction and earlier exact-query lifecycle checks; Windows GTX 1080 passes 17/17 same-build CUDA/D3D12 comparisons and independent brick/contact checks | Full current 30-case physical qualification and performance targets incomplete |
| Vulkan | Six reflected modules and SPIR-V checks; public API and async tests with native validation; fluid lifecycle with PM validation | Fluid lifecycle crashes inside installed native Vulkan validation layer; reproduced with previous binding generator too |
| D3D12 | Windows GTX 1080: 17/17 selected CUDA comparisons, zero unsupported; independent eight-brick/300-frame penetration 0.205278 mm against the unchanged 2 mm limit, matching CUDA | Remaining domains, devices and performance targets unqualified; comparison agreement alone is not physical quality |
| Metal | Existing preview/compiler evidence documented separately | Not runnable on this host; see Metal qualification document |
| Distinct fluid–fluid systems | Unsupported explicitly | Material semantics, per-pair capabilities and physical/lifecycle qualification still required |
| Engine-owned devices/queues | Public contract and import declarations exist | Common dependency/completion handoff and imported-device tests still required before engine adoption claims |

Detailed results: [lifetime](RESOURCE-LIFETIME.md), [API rules](API-CONTRACT.md),
[common resources](COMMON-RESOURCE-BOUNDARY.md), [conformance](../conformance/v1/README.md),
[Metal](METAL-QUALIFICATION.md), [Android](ANDROID-QUALIFICATION.md).

The [Windows physics report](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json)
records tested source/binary/case hashes, original first divergences, comparison
results, independent physical metrics and reproduction commands. See also the
[shared correction and regression scope](slang-unification-validation.md#current-cross-driver-qualification-2026-10-16).

Reports must distinguish compilation, execution, reference regression, analytic
or invariant physical quality, and performance. An unsupported case counts as
neither a pass nor a numerical failure, and leaves its qualification scope open.
Performance comparisons require both implementations to pass the same fixed
quality profile first. Raw case outputs and failures survive the comparison.
