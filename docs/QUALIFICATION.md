# Qualification scope and open gates

Updated 2026-10-17. This is an evidence index, not blanket driver certification.
Each release must attach a revisioned machine-readable evidence package with
the exact source, driver, toolchain, device, case hashes and settings. Repeat the
affected scope after changes; historical passes do not automatically qualify a
new build.

| Scope | Evidence and host scope | Unqualified or failed scope |
|---|---|---|
| Resource lifetime and capacity (F1) | Reference-lifecycle foundation closed by F1-01–F1-07: exact-query fluid churn/recovery; CUDA mixed-domain churn; geometry, paint and borrowed-endpoint replacement; failure/pending lifetime; fixed host owner metadata | Does not qualify numerical physics, every driver/domain pair, all implemented coupling pairs, host/OS budget accuracy or the adverse three-family corpus; see F2-04, F4-05, F5-03 and P1-01 |
| SDK/loader/host contracts | 33 shader-free `sdk-ci` CTests; relocated C/C++ consumers and separately built mock driver; standalone conformance checks all 30 inputs | New remote workflow not yet verified; Windows/macOS installed packages not run here; physics not evaluated by this gate |
| Slang CPU | Current foundation suite: 48/48 passed as configured with four expected unsupported-domain skips; exact queried fluid lifecycle/recovery; private host fixtures cover paint, endpoint and injected-failure paths without advertising solver support | Mixed particle domains unsupported; fixture execution is lifecycle evidence only; `rigid-direct` final angular velocity fails historical CUDA comparison |
| CUDA | Linux public rigid/fluid/cloth command API after scene SDK extraction and earlier exact-query lifecycle checks; Windows GTX 1080 passes 17/17 same-build CUDA/D3D12 comparisons and independent brick/contact checks | Full current 30-case physical qualification and performance targets incomplete |
| Vulkan | Six reflected modules and SPIR-V checks; public API and async tests with native validation; fluid lifecycle with PM validation | Fluid lifecycle crashes inside installed native Vulkan validation layer; reproduced with previous binding generator too |
| D3D12 | Windows GTX 1080: 17/17 selected CUDA comparisons, zero unsupported; independent eight-brick/300-frame penetration 0.205278 mm against the unchanged 2 mm limit, matching CUDA | Remaining domains, devices and performance targets unqualified; comparison agreement alone is not physical quality |
| Metal | Existing preview/compiler evidence documented separately | Not runnable on this host; see Metal qualification document |
| Interaction graph lifecycle (F2-04) | Linux RTX 3050 Ti CUDA executes all 12 implemented pairs through pipeline and explicit create/update/clear, six-family endpoint replacement, three-family mutation and partial-create rollback | Production CPU and Vulkan report the complete-domain case unsupported; D3D12 and Metal not run; lifecycle execution does not qualify numerical physics |
| Distinct fluid–fluid systems | F2-05 specifies the equal-material edge, combined-population tolerances, balanced-momentum limit, stable-ID ownership and capacity failures | Still explicitly unsupported; neighbor indexing, force exchange, graph scheduling, lifecycle and physical qualification remain F2-06–F2-10; different-material interfaces remain F2-11 |
| Engine-owned devices/queues | Public contract and import declarations exist | Common dependency/completion handoff and imported-device tests still required before engine adoption claims |

## F1 resource-lifetime exit review

The original F1 exit required repeated churn to remain bounded without a whole
simulation reset, unrelated live systems to survive replacement and failure,
and exact-capacity/recovery cases to execute through the public API. The merged
evidence in task cards F1-01 through F1-07 satisfies that reference-lifecycle
scope:

- Public fluid lifecycle binds the exact queried memory, performs 4,096
  replacements and 128 deferred-failure recoveries, rejects undersized views,
  and preserves a second population. Slang CPU, CUDA and Vulkan passes are
  recorded in [resource lifetime](RESOURCE-LIFETIME.md).
- CUDA performs 1,024 exact-query interleaved cloth, rope, soft-body and smoke
  writes while changing layouts. Production CPU and Vulkan do not advertise
  that full domain set, so their skips are retained as unsupported outcomes.
- Public geometry, paint and emitter/destroy-plane tests cover shared-resource
  preservation, stale generations, failed replacement, endpoint moves and
  reset reuse. Private non-installed CPU fixtures exercise host lifecycle paths
  without claiming production solver support.
- Public failure tests distinguish allocation, recording, pre-acceptance submit
  and deferred execution failures. The production Slang CPU test holds reset
  work behind a timeline dependency and verifies pending objects cannot be
  reused until completion.
- Arena tests verify reclamation/coalescing and owner-slot tests perform 4,096
  retire/recreate cycles with unchanged table storage, immediate payload
  destruction, explicit exhaustion and stale-handle rejection.

The current Linux foundation build passes 45/45 configured tests with three
expected production CPU feature skips. The shared runtime also compiles in the
current CUDA and Vulkan desktop targets. Local GPU lifecycle cases skipped in
this review because no eligible physical device was exposed to the test loader;
that is not a new execution pass.

The review found no uncovered F1 leaf. F2-04 now records the separate coupling
rebinding lifecycle evidence below. Memory-budget confidence is F4-05, adverse
capacity and three-family corpus coverage is F5-03, and the remaining
driver/domain matrix is P1-01. The native Vulkan validation failure and
production CPU domain limits remain open in the table above. Closing F1 does
not qualify physical accuracy or release readiness.

## F2 interaction graph lifecycle

The public SDK-only
[interaction graph lifecycle suite](../tests/api/interaction_graph_lifecycle_tests.cpp)
constructs one endpoint from every implemented family and installs all 12
implemented non-fluid-fluid pair types. It exercises pipeline and explicit
graph create, parameter update, clear and recreation. It then replaces fluid,
rigid, cloth, rope, soft-body and smoke endpoints while the complete graph is
requested, forcing all incident edges to bind the new private generations.

The same suite mutates a fluid/rope/soft-body graph while retaining an unrelated
smoke-cloth edge. A descriptor with an invalid runtime parameter fails after
the first new edge is installed, exercising partial-create cleanup and prior
graph restoration; a valid replacement and smoke endpoint rebind then succeed.
On this Linux laptop, the CUDA case passed on an NVIDIA GeForce RTX 3050 Ti
Laptop GPU with driver 590.44.01, CUDA compiler 13.1, Slang 2026.18 and GCC
12.2. Production CPU and Vulkan retained exit 77 as unsupported for the full
six-family feature request. These are lifecycle results; they do not establish
physical accuracy, cross-driver agreement or performance.

Reproduction commands:

```text
cmake --build build-desktop-switch --target parallel-mater-interaction-graph-lifecycle-tests -j2
ctest --test-dir build-desktop-switch -R '(cuda|vulkan)-interaction-graph-lifecycle' --output-on-failure
cmake --build build-foundations -j2
ctest --test-dir build-foundations -j2 --output-on-failure
python3 tools/check_sdk_install.py --build-dir build-sdk-ci --output out/sdk-evidence/f2-04
```

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
