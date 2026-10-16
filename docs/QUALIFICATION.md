# Qualification scope and open gates

Updated 2026-10-16. This is an evidence index, not blanket driver certification.
Each release must attach a revisioned machine-readable evidence package with
the exact source, driver, toolchain, device, case hashes and settings. Repeat the
affected scope after changes; historical passes do not automatically qualify a
new build.

| Scope | Evidence on this laptop | Unqualified or failed scope |
|---|---|---|
| SDK/loader/host contracts | 30 shader-free `sdk-ci` CTests; relocated C/C++ consumers and separately built mock driver; standalone conformance checks all 30 inputs | New remote workflow not yet verified; Windows/macOS installed packages not run here; physics not evaluated by this gate |
| Slang CPU | Foundation suite: 34 passed, two unsupported domain skips; exact queried fluid lifecycle/recovery | Mixed particle domains unsupported; `rigid-direct` final angular velocity fails historical CUDA comparison |
| CUDA | Public rigid/fluid/cloth command API after scene SDK extraction; earlier exact-query fluid/domain/coupling lifecycle regressions | Full current 30-case physical qualification incomplete; upstream CUDA/D3D12 comparisons pass 7/17 and fail 10/17 |
| Vulkan | Six reflected modules and SPIR-V checks; public API and async tests with native validation; fluid lifecycle with PM validation | Fluid lifecycle crashes inside installed native Vulkan validation layer; reproduced with previous binding generator too |
| D3D12 | Upstream runtime/renderer and comparison evidence retained in project history | Not runnable on this Linux host; shared brick penetration about 7 mm exceeds 2 mm gate |
| Metal | Existing preview/compiler evidence documented separately | Not runnable on this host; see Metal qualification document |
| Distinct fluid–fluid systems | Unsupported explicitly | Material semantics, per-pair capabilities and physical/lifecycle qualification still required |
| Engine-owned devices/queues | Public contract and import declarations exist | Common dependency/completion handoff and imported-device tests still required before engine adoption claims |

Detailed results: [lifetime](RESOURCE-LIFETIME.md), [API rules](API-CONTRACT.md),
[common resources](COMMON-RESOURCE-BOUNDARY.md), [conformance](../conformance/v1/README.md),
[Metal](METAL-QUALIFICATION.md), [Android](ANDROID-QUALIFICATION.md).

Reports must distinguish compilation, execution, reference regression, analytic
or invariant physical quality, and performance. An unsupported case counts as
neither a pass nor a numerical failure, and leaves its qualification scope open.
Performance comparisons require both implementations to pass the same fixed
quality profile first. Raw case outputs and failures survive the comparison.
