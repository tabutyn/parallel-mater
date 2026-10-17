# Qualification scope and open gates

Updated 2026-10-17. This is an evidence index, not blanket driver certification.
Each release must attach a revisioned machine-readable evidence package with
the exact source, driver, toolchain, device, case hashes and settings. Repeat the
affected scope after changes; historical passes do not automatically qualify a
new build.

## Per-driver evidence index

<!-- BEGIN GENERATED DRIVER EVIDENCE -->
| Driver | Current capability result | Qualified records | Open evidence |
|---|---|---:|---|
| Slang CPU | None | 0 | 1 missing qualification result |
| CUDA | None | 0 | Supplemental: [`windows-gtx1080-physics-2026-10-16`](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json); 10 historical baseline failure, 3 limitation, 1 missing qualification result |
| D3D12 | None | 0 | Supplemental: [`windows-gtx1080-physics-2026-10-16`](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json); 10 historical baseline failure, 3 limitation, 1 missing qualification result, 2 unsupported |
| Vulkan | None | 0 | 1 missing qualification result |
| Metal | None | 0 | 1 missing qualification result |

Index status: **0 qualified capability records from 2 revisioned report records**. Supplemental and historical evidence remains visible but never certifies a capability.
<!-- END GENERATED DRIVER EVIDENCE -->

Only the generated block states current per-driver capability qualification.
The scope table below records project-wide contracts and other evidence without
promoting those results to capability certification.

| Scope | Evidence and host scope | Unqualified or failed scope |
|---|---|---|
| Resource lifetime and capacity (F1) | Reference-lifecycle foundation closed by F1-01–F1-07: exact-query fluid churn/recovery; CUDA mixed-domain churn; geometry, paint and borrowed-endpoint replacement; failure/pending lifetime; fixed host owner metadata. F5-03 adds the immutable adverse-capacity lifecycle profile and evaluator. | No driver is newly qualified by the profile alone; each advertising driver must emit a complete passing result. Numerical physics, remaining driver/domain pairs and host/OS budget accuracy remain separate; see F2-04, F4-05 and P1-01. |
| SDK/loader/host contracts | 33 shader-free `sdk-ci` CTests; relocated C/C++ consumers and separately built mock driver; standalone conformance checks all 30 inputs | New remote workflow not yet verified; Windows/macOS installed packages not run here; physics not evaluated by this gate |
| Interaction graph lifecycle (F2-04) | Linux RTX 3050 Ti CUDA executes all 12 implemented pairs through pipeline and explicit create/update/clear, six-family endpoint replacement, three-family mutation and partial-create rollback. F5-03 makes the adverse sequence a hashed two-case lifecycle contract with exact preservation checks. | Existing runtime evidence predates the F5-03 result schema. Production CPU and Vulkan report the complete-domain case unsupported; D3D12 and Metal were not run. Unsupported never passes the lifecycle evaluator; lifecycle execution does not qualify numerical accuracy. |
| Vendor-neutral physical quality (F5-02) | Hashed `core-conservation-stability-v1` profile and independent evaluator cover pressure/contact and viscosity across 18 timestep/mass-ratio samples; negative tests reject momentum, angular-momentum, energy, split/combined, sweep, provenance and inactive-exchange faults | This defines the gate but does not qualify a driver; F2-10 must produce public-API results for each advertising driver, and later profiles still need their own fixtures |
| Capability evidence obligations (F5-01) | Hashed `capability-obligations-v1` rules cover all 49 API 0.2 interaction records exactly once and keep compilation, execution, regression, lifecycle, analytic and timing evidence separate. Revision, contract, binary, device and toolchain provenance are mandatory. | This defines validation, not new driver evidence. Known missing lifecycle/analytic contracts fail explicitly; F2-10 and later qualification leaves must supply complete per-driver results. |
| Quality-gated performance (F5-07) | Independent headless gate computes throughput, median/p95 timing and public memory peaks from raw samples; every metric carries the fixed quality-profile hash and exact source/binary/device/toolchain tuple | This defines score admission but publishes no production benchmark. Missing, failed, unsupported, stale or mismatched physical-quality evidence leaves measurements unqualified. |
| Distinct fluid–fluid systems | F2-05–F2-09 specify and schedule equal-material contact/pressure/viscosity with bounded pair storage, combined-population tolerances, balanced momentum, stable-ID ownership, lifecycle recovery and CPU Slang physical fixtures. F2-11 specifies miscible, immiscible-interface and optional conservative thermal contracts and splits their deferred API/operator/schedule/qualification work into P1-04–P1-18. | Every production driver still reports the separate-fluid records unsupported pending F2-10 qualification. Different-material enums remain reservations only; their typed state, shared operators and per-driver evidence remain P1-04–P1-18, and phase transfer remains undefined. |
| Cloth self and cloth–cloth systems | F2-12 specifies separate intrinsic self-contact and graph-owned inter-system contact identities, swept double-sided vertex-triangle/edge-edge behavior, topology exclusions, thickness, friction, conservation, tunnelling and overflow criteria. | API 0.2 exposes neither capability; typed controls, bounded contact generation, self/inter operators, scheduling and per-driver qualification remain P1-19–P1-24. |
| Soft-body self and soft-body systems | F2-13 specifies separate intrinsic and graph-owned surface-contact identities, bound closed-surface sampling, swept primitive contact, topology exclusions, node-radius separation, friction, conservation and bounded failure behavior. | API 0.2 exposes neither capability; typed controls, bounded surface contacts, self/inter operators, scheduling and per-driver qualification remain P1-25–P1-30. Open/non-manifold skins, volume inversion and merging remain unsupported. |
| Rope self and rope–rope systems | F2-14 narrows the API 0.2 self-contact record to sampled-node exclusion and specifies a separate future `rope-capsule` profile for continuous self/inter segment contact plus an independent endpoint attachment mode. | The advertised node-only record still needs the P1-31 physical audit. Capsule contact, pair edges and rope-rope attachment remain unsupported pending P1-32–P1-38; knots, splices, sliding and interior attachments are outside that profile. |
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
rebinding lifecycle evidence below. Memory-budget confidence remains F4-05.
Per-driver results for the F5-03 capacity and three-family profile remain open;
the remaining driver/domain matrix is P1-01. The native Vulkan validation failure and
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
