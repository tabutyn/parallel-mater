# Qualification scope and open gates

Updated 2026-10-21. This is an evidence index, not blanket driver certification.
Each release must attach a revisioned machine-readable evidence package with
the exact source, driver, toolchain, device, case hashes and settings. Repeat the
affected scope after changes; historical passes do not automatically qualify a
new build.

## Per-driver evidence index

<!-- BEGIN GENERATED DRIVER EVIDENCE -->
| Driver | Current capability result | Qualified records | Open evidence |
|---|---|---:|---|
| Slang CPU | [`linux-thomas-laptop-slang-cpu-eed669e`](../conformance/v1/evidence/linux-thomas-laptop-slang-cpu-eed669e-qualification.json) | 0 | 8 qualification failure |
| CUDA | [`linux-rtx3050ti-cuda-equal-fluid-qualification-2026-10-17`](../conformance/v1/evidence/linux-rtx3050ti-cuda-equal-fluid-qualification-2026-10-17.json), [`linux-rtx3050ti-slang-cuda-eed669e`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-eed669e-qualification.json), [`linux-rtx3050ti-slang-cuda-e3256bc-fluid-rigid-containment`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-e3256bc-fluid-rigid-containment-qualification.json), [`linux-rtx3050ti-slang-cuda-7934b3f-cloth-smoke-contact-grid`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-7934b3f-cloth-smoke-contact-grid-qualification.json), [`linux-rtx3050ti-slang-cuda-7934b3f-cloth-smoke-contact-particle`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-7934b3f-cloth-smoke-contact-particle-qualification.json), [`linux-rtx3050ti-slang-cuda-9a71fd6-soft-rope-attachment`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-9a71fd6-soft-rope-attachment-qualification.json), [`linux-rtx3050ti-slang-cuda-cc2b9c0-soft-smoke-contact-grid`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-cc2b9c0-soft-smoke-contact-grid-qualification.json), [`linux-rtx3050ti-slang-cuda-cc2b9c0-soft-smoke-contact-particle`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-cc2b9c0-soft-smoke-contact-particle-qualification.json), [`linux-rtx3050ti-slang-cuda-8bbbeb8-smoke-soft-drag-grid`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-8bbbeb8-smoke-soft-drag-grid-qualification.json), [`linux-rtx3050ti-slang-cuda-8bbbeb8-smoke-soft-drag-particle`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-8bbbeb8-smoke-soft-drag-particle-qualification.json), [`linux-rtx3050ti-slang-cuda-7396577-soft-rigid-contact`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-7396577-soft-rigid-contact-qualification.json), [`linux-rtx3050ti-slang-cuda-84a475e-rope-smoke-contact-grid`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-84a475e-rope-smoke-contact-grid-qualification.json), [`linux-rtx3050ti-slang-cuda-84a475e-rope-smoke-contact-particle`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-84a475e-rope-smoke-contact-particle-qualification.json), [`linux-rtx3050ti-slang-cuda-0dc9fbf-smoke-rope-drag-grid`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-0dc9fbf-smoke-rope-drag-grid-qualification.json), [`linux-rtx3050ti-slang-cuda-0dc9fbf-smoke-rope-drag-particle`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-0dc9fbf-smoke-rope-drag-particle-qualification.json), [`linux-rtx3050ti-slang-cuda-b8bc7ee-smoke-rigid-drag-grid`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-b8bc7ee-smoke-rigid-drag-grid-qualification.json), [`linux-rtx3050ti-slang-cuda-b8bc7ee-smoke-rigid-drag-particle`](../conformance/v1/evidence/linux-rtx3050ti-slang-cuda-b8bc7ee-smoke-rigid-drag-particle-qualification.json), [`windows-gtx1080-slang-cuda-03c78b1-smoke-fluid-drag-grid`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-03c78b1-smoke-fluid-drag-grid-qualification.json), [`windows-gtx1080-slang-cuda-03c78b1-smoke-fluid-drag-particle`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-03c78b1-smoke-fluid-drag-particle-qualification.json), [`windows-gtx1080-slang-cuda-2b04e55`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-2b04e55-qualification.json), [`windows-gtx1080-slang-cuda-5001123-fluid-cloth-contact`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-5001123-fluid-cloth-contact-qualification.json), [`windows-gtx1080-slang-cuda-59eb0ce-cloth-rigid-contact`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-59eb0ce-cloth-rigid-contact-qualification.json), [`windows-gtx1080-slang-cuda-b191454-cloth-soft-contact`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-b191454-cloth-soft-contact-qualification.json), [`windows-gtx1080-slang-cuda-dac1c20-externally-heated-fluid-grid`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-dac1c20-externally-heated-fluid-grid-qualification.json), [`windows-gtx1080-slang-cuda-dac1c20-externally-heated-fluid-particle`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-dac1c20-externally-heated-fluid-particle-qualification.json), [`windows-gtx1080-slang-cuda-ec024d7-smoke-rigid-contact-grid`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-ec024d7-smoke-rigid-contact-grid-qualification.json), [`windows-gtx1080-slang-cuda-ec024d7-smoke-rigid-contact-particle`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-ec024d7-smoke-rigid-contact-particle-qualification.json) | 0 | Supplemental: [`windows-gtx1080-physics-2026-10-16`](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json); 1 case validation, 10 historical baseline failure, 3 limitation, 371 qualification failure |
| D3D12 | [`windows-gtx1080-slang-d3d12-2b04e55`](../conformance/v1/evidence/windows-gtx1080-slang-d3d12-2b04e55-qualification.json), [`windows-gtx1080-slang-d3d12-5001123-fluid-cloth-contact`](../conformance/v1/evidence/windows-gtx1080-slang-d3d12-5001123-fluid-cloth-contact-qualification.json), [`windows-gtx1080-slang-d3d12-59eb0ce-cloth-rigid-contact`](../conformance/v1/evidence/windows-gtx1080-slang-d3d12-59eb0ce-cloth-rigid-contact-qualification.json) | 0 | Supplemental: [`windows-gtx1080-physics-2026-10-16`](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json); 1 case validation, 10 historical baseline failure, 3 limitation, 40 qualification failure, 2 unsupported |
| Vulkan | [`android-pixel7-slang-vulkan-3187375`](../conformance/v1/evidence/android-pixel7-slang-vulkan-3187375-qualification.json), [`linux-rtx3050ti-slang-vulkan-eed669e`](../conformance/v1/evidence/linux-rtx3050ti-slang-vulkan-eed669e-qualification.json) | 0 | 224 qualification failure |
| Metal | [`ipados-m1-slang-metal-9f7e72e`](../conformance/v1/evidence/ipados-m1-slang-metal-9f7e72e-qualification.json), [`macos-m4-slang-metal-16169d5-smoke-cloth-grid`](../conformance/v1/evidence/macos-m4-slang-metal-16169d5-smoke-cloth-grid-qualification.json), [`macos-m4-slang-metal-16169d5-smoke-cloth-particle`](../conformance/v1/evidence/macos-m4-slang-metal-16169d5-smoke-cloth-particle-qualification.json), [`macos-m4-slang-metal-4e09ecb-fluid-rigid-contact`](../conformance/v1/evidence/macos-m4-slang-metal-4e09ecb-fluid-rigid-contact-qualification.json), [`macos-m4-slang-metal-e09edbe-cloth-rope-attachment`](../conformance/v1/evidence/macos-m4-slang-metal-e09edbe-cloth-rope-attachment-qualification.json), [`macos-m4-slang-metal-7161cca-fluid-smoke-phase-grid`](../conformance/v1/evidence/macos-m4-slang-metal-7161cca-fluid-smoke-phase-grid-qualification.json), [`macos-m4-slang-metal-7161cca-fluid-smoke-phase-particle`](../conformance/v1/evidence/macos-m4-slang-metal-7161cca-fluid-smoke-phase-particle-qualification.json), [`macos-m4-slang-metal-86f10b2-fluid-cloth-containment`](../conformance/v1/evidence/macos-m4-slang-metal-86f10b2-fluid-cloth-containment-qualification.json), [`macos-m4-slang-metal-86f10b2-fluid-soft-contact`](../conformance/v1/evidence/macos-m4-slang-metal-86f10b2-fluid-soft-contact-qualification.json), [`macos-m4-slang-metal-86f10b2-smoke-viscosity`](../conformance/v1/evidence/macos-m4-slang-metal-86f10b2-smoke-viscosity-qualification.json), [`macos-m4-slang-metal-aac6d59`](../conformance/v1/evidence/macos-m4-slang-metal-aac6d59-qualification.json), [`macos-m4-slang-metal-cffde48-soft-rope-contact`](../conformance/v1/evidence/macos-m4-slang-metal-cffde48-soft-rope-contact-qualification.json) | 1 | 317 qualification failure |

Index status: **1 qualified capability records from 47 revisioned report records**. Supplemental and historical evidence remains visible but never certifies a capability.
<!-- END GENERATED DRIVER EVIDENCE -->

Only the generated block states current per-driver capability qualification.
The scope table below records project-wide contracts and other evidence without
promoting those results to capability certification.

| Scope | Evidence and host scope | Unqualified or failed scope |
|---|---|---|
| Resource lifetime and capacity (F1) | Reference-lifecycle foundation closed by F1-01–F1-07: exact-query fluid churn/recovery; CUDA mixed-domain churn; geometry, paint and borrowed-endpoint replacement; failure/pending lifetime; fixed host owner metadata. Current Windows CUDA/D3D12 and Apple M4 Metal console tests pass 4,096 fluid replacements and exact-capacity recovery. | The console tests do not emit F5-03 checkpoint state hashes/write ordinals, so current results retain explicit `missing` profile evidence. Numerical physics and remaining driver/domain pairs remain separate; see F2-04, F4-05 and P1-01. |
| Specification, ABI and validation (F3) | F3-01–F3-12 reconcile the generated API/specification, loader and layer negotiation, typed validation diagnostics, command/object/query/synchronization state, host snapshots, deferred errors, teardown and capacity recovery. The F3-99 review passes 23 Windows D3D12/validation tests, 6 independent WSL host tests and all 18 relocated public SDK consumer/ABI cases. | This is portable contract and reference-runtime evidence, not exhaustive proof of every specification clause, physical correctness or every driver. Native interop handoff remains F4-07/F4-08; driver/domain qualification remains F5/P1. SDK installation is qualified separately by F6-01/F6-02. |
| Runtime-generated capacity recovery (F3-12) | Public API profile fixes four colocated particles, one allowed neighbor and one isolated control population. D3D12 on the Windows GTX 1080 and the independent WSL Slang C++ host fixture each pass 20/20: acceptance succeeds; completion, status and statistics report exactly four generated overflows; invalid output is not published; selective reinitialization recovers while preserving the control. | This qualifies only the named shared-Slang neighbor-overflow profile and lifecycle contract. It is not physical-quality, performance or blanket driver-capacity evidence. |
| SDK/loader/host contracts (F6) | API/package 0.2.0. Clean Linux source `3407259` passes 47/47 `sdk-ci` CTests, all 12 relocated-install stages, 44/44 component checks and 87/87 conformance metadata tests. Windows F6-01 passes 38/38 tests plus 18/18 external C/C++/ABI cases and all 30 then-current conformance inputs; macOS ARM64 and Linux CI pass the same 12-stage relocation gate. F6-03 separates loader, validation, preparation, scene, driver, conformance and benchmark exports. Latest tracked two-build Linux evidence has matching 173-file installed artifact sets and no required workstation paths. | This qualifies packaging and public ABI behavior only. Physical quality and the optional C++ preparation/scene ABI are not certified. Bit-identical output is not established: one installed ELF and ten build-tree manifests differ between the recorded isolated Linux builds. Gallery is an opt-in public-API example application under F7, not an installed SDK component. |
| Interaction graph lifecycle (F2-04, P1-59) | Linux RTX 3050 Ti and Windows GTX 1080 CUDA, Apple M4 Metal and current Windows GTX 1080 D3D12 execute the implemented common-domain graph through create/update/clear, endpoint replacement, three-family mutation and partial-create rollback. P1-59 passes all 17 applicable D3D12 lifecycle profiles, including the adverse three-family case. | Historical captures without F5-03 checkpoint fingerprints remain unqualified. P1-59 supplies exact lifecycle evaluation for the newly advertised D3D12 records, but reference-regression failures still prevent blanket capability qualification. |
| Native Vulkan validation (F5-09) | A standalone shader-module/pipeline probe pins the layer, fluid SPIR-V and probe hashes. Debian layer 1.3.239 exits 139 inside `vkCreateShaderModule`; pinned SDK layer 1.4.341 completes the same probe and the unchanged 4,096-replacement validated lifecycle case on the RTX 3050 Ti. | This isolates a host validation-tool defect and supplies a working validation setup. It does not qualify Vulkan physics, other drivers or imported engine devices. |
| Vendor-neutral physical quality (F5-02, F5-05, F5-25–F5-29, P1-59) | Hashed profiles independently evaluate fluid conservation/stability, rigid surface separation, fluid self contact/pressure/viscosity, sampled-node rope separation and particle/grid smoke behavior. Current Windows CUDA passes the earlier gates. P1-59 adds current GTX 1080 D3D12 passes for all 22 applicable rope, soft-body, smoke and cross-domain analytic profiles plus all 17 applicable lifecycle profiles. Apple M4 Metal passes the brick gate. | D3D12 still does not advertise the three separate-system equal-fluid records, and Metal does not advertise separate-system equal-fluid. P1-59 retains 12 pinned-reference failures and one same-revision CUDA/D3D12 `cloth-tear` mismatch. Analytic quality never overrides failed reference regression, so the new execution evidence grants no blanket qualification. |
| Capability evidence obligations (F5-01, P1-59) | Hashed `capability-obligations-v1` rules cover all 49 API 0.2 interaction records exactly once and keep compilation, execution, regression, lifecycle, analytic and timing evidence separate. Current P1-59 physical results enumerate 49 CUDA and 46 D3D12 records on the GTX 1080; the three absent D3D12 records are the deliberately unsupported equal-fluid modes. Apple M4 Metal also reports 46 records. | Current P1-59 D3D12 qualification remains 0/46: 18/30 pinned regressions pass, 12 fail, while 22/22 analytic and 17/17 lifecycle profiles pass. CUDA produces 27/30 same-revision cases because three coupled-deformable cases crash; 26/27 available CUDA/D3D12 comparisons pass. Failures remain explicit and unqualified. |
| Independent conformance and quality exit review (F5-99) | All 56 dependencies are merged `done` with reviewable evidence. The installed runner, 46 exact obligation rules, revisioned index and separate execution/regression/lifecycle/analytic/performance gates establish the independent qualification mechanism. Candidate Apple M4 verification passes 112/112 macOS Metal preset tests and the unchanged brick-penetration gate. | The original every-advertised-mode current-evidence exit is not met: the generated index qualifies 1/336 report-scoped capability occurrences and retains 990 known failures. Historical passes from different source revisions cannot be composed. P1-01 owns the current driver/domain inventory and same-revision recapture split. |
| iPadOS development qualification runner (F5-13) | The opt-in, development-only app channel reuses the canonical 30-case public-API runner, rigid penetration oracle, equal-fluid fixture and adverse lifecycle operations. Collection pins a fresh nonce, source, app/driver hashes, public device/OS/Metal and toolchain metadata and rejects stale, partial, mismatched and Gallery-only reports. | This runner is not enabled by default in ordinary iOS builds. Equal-fluid remains unsupported on Metal and the existing lifecycle fixtures still lack F5-03 checkpoint fingerprints; device execution and evidence publication remain separate gates. |
| Quality-gated performance (F5-07) | Independent headless gate computes throughput, median/p95 timing and public memory peaks from raw samples; every metric carries the fixed quality-profile hash and exact source/binary/device/toolchain tuple | This defines score admission but publishes no production benchmark. Missing, failed, unsupported, stale or mismatched physical-quality evidence leaves measurements unqualified. |
| Distinct fluid–fluid systems | F2-05–F2-10 specify, schedule and qualify equal-material contact/pressure/viscosity with bounded pair storage, combined-population tolerances, balanced momentum, stable-ID ownership and lifecycle recovery. Linux RTX 3050 Ti CUDA qualifies all three records. Current Windows GTX 1080 CUDA independently passes the same 18/18 physical samples and fluid-pair graph lifecycle console test. | Windows CUDA lacks the exact F5-03 checkpoint artifact, so its three records remain unqualified; D3D12 and Metal do not advertise these records, and their tests skip without fallback. Different-material modes remain P1-04–P1-18, and phase transfer remains undefined. |
| Cloth self and cloth–cloth systems | F2-12 specifies separate intrinsic self-contact and graph-owned inter-system contact identities, swept double-sided vertex-triangle/edge-edge behavior, topology exclusions, thickness, friction, conservation, tunnelling and overflow criteria. | API 0.2 exposes neither capability; typed controls, bounded contact generation, self/inter operators, scheduling and per-driver qualification remain P1-19–P1-24. |
| Soft-body self and soft-body systems | F2-13 specifies separate intrinsic and graph-owned surface-contact identities, bound closed-surface sampling, swept primitive contact, topology exclusions, node-radius separation, friction, conservation and bounded failure behavior. | API 0.2 exposes neither capability; typed controls, bounded surface contacts, self/inter operators, scheduling and per-driver qualification remain P1-25–P1-30. Open/non-manifold skins, volume inversion and merging remain unsupported. |
| Rope self and rope–rope systems | F2-14 narrows the API 0.2 self-contact record to sampled-node exclusion and specifies a separate future `rope-capsule` profile for continuous self/inter segment contact plus an independent endpoint attachment mode. F5-28 binds the exact sampled-node gate; P1-59 now advertises the existing rope record on D3D12 and passes `rope-self-contact-v1`, rope coupling quality/lifecycle profiles and all four 300-frame rope Gallery scenes. | P1-59 retains pinned `rope-core` and `rope-fluid` reference failures, so D3D12 rope is implemented but not blanket-qualified. Capsule contact, pair edges and rope-rope attachment remain unsupported pending P1-32–P1-38; knots, splices, sliding and interior attachments are outside this profile. |
| Smoke self and smoke–smoke systems | F2-15 specifies particle-particle, grid-grid and particle-grid profiles with separate mechanical, thermal and projection behavior; bounded overlap/neighborhood construction; fixed-point and conservation criteria; and graph lifecycle rules. F5-29 binds the exact particle/grid self-pressure records. P1-59 now advertises existing particle/grid smoke records on D3D12 and passes self-pressure, self-viscosity, cross-domain quality/lifecycle and all five 300-frame smoke Gallery scenes. | The P1-59 D3D12 smoke executions and analytic gates do not override the retained pinned-reference failures elsewhere in the 46-record driver. Broader same-family smoke-system work remains P1-39–P1-55. Combustion, chemistry, phase transfer and topology-changing guarantees remain unsupported. |
| Common resources and engine ownership (F4, P1-59) | F4-01–F4-08 define device/allocation identities, typed access ranges, reflection-derived bindings, exact RAW/WAR/WAW ordering, common completion ownership, confidence-aware memory admission, borrowed native objects and prior/next queue handoff. P1-59 routes all existing common rope, soft-body, smoke, paint and coupling schedules through reflection-generated DXIL and the thin native D3D12 command/resource adapter; the 20-source architecture inventory rejects restored dispatch stubs. | No driver yet claims qualified imported-device/timeline interoperability. P2-01 owns the first native handoff fixture and P2-02–P2-04 own engine plugins. Adapter execution does not establish physical quality, performance or native engine adoption; those claims remain tied to their independent evidence gates. |

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
the remaining driver/domain matrix is P1-01. Native Vulkan validation now uses
the F5-09 pinned-layer setup; production CPU domain limits remain open. Closing F1 does
not qualify physical accuracy or release readiness.

## F2 interaction graph lifecycle

The public SDK-only
[interaction graph lifecycle suite](../conformance/v1/src/interaction_graph_lifecycle_fixture.cpp)
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

### F2-99 interaction and coupling exit review

The original F2 exit required pipeline changes and endpoint replacement to
avoid stale edges, exact discovery of supported pair/mode/material behavior,
and lifecycle plus physics cases for the scope a driver declares. Task cards
F2-01 through F2-15 are merged `done` with reviewable evidence. Together they
close this foundation scope as follows:

- The normative interaction matrix enumerates all 15 unordered
  distinct-family pairs and the six same-family rows. Capability records name
  scope, mode, direction and material profile; reference inventory, proposed
  behavior and unsupported behavior remain separate.
- Public graph replacement validates the complete requested bundle before it
  becomes live. Negative tests cover malformed and unsupported records,
  duplicates, capacity, partial creation and restoration of the prior graph.
  The all-domain lifecycle suite covers create, update, clear, pipeline
  changes and endpoint generations for all 12 implemented non-fluid-fluid pair
  types, including a three-family mutation that preserves an unrelated edge.
- Equal-material separate fluid systems have bounded cross-population neighbor
  storage and shared Slang contact, pressure and viscosity operators. Public
  lifecycle and split-versus-combined quality tests sweep two fixtures, three
  timesteps and three population mass ratios. The revisioned Linux RTX 3050 Ti
  CUDA result qualifies exactly those three records; no other driver or
  material profile is inferred from it.
- Different-material fluid behavior and cloth, soft-body, rope and smoke
  same-family behavior are specified but remain unadvertised. Their typed API,
  bounded generation, operators, scheduling and qualification work is already
  split into P1-04 through P1-55. P1-31 separately audits the narrower
  advertised rope sampled-node self-contact record.

The F2 exit review reran its then-current Windows CUDA/D3D12 slice on a GTX 1080:
five applicable tests passed and three D3D12 tests reported their defined
unsupported result. CUDA passed complete interaction-graph lifecycle, graph
validation, fluid-pair lifecycle and all 18 equal-fluid quality samples; D3D12
passed graph validation and did not advertise the unsupported six-family or
fluid-pair scopes. Roadmap tests (19), conformance tests (67), and the generated
public-header/specification check also passed.

This closes the shared F2 graph/coupling foundation, not blanket physics or
driver qualification. Current per-driver failures, crashes, missing immutable
F5-03 lifecycle captures and analytic gaps remain visible in the generated
index. F5-12 through F5-16 own current hardware evidence, while P1-01 must split
every remaining driver/domain implementation and evidence slice after the six
foundation reviews. No uncovered F2 implementation leaf was found.

### F3-99 specification, ABI and validation exit review

The original F3 exit required documentation, runtime and validation to agree,
with executable evidence beyond header generation and command routing. Task
cards F3-01 through F3-12 are merged `done` with nonempty evidence and completed
lease history. Together they close that foundation scope as follows:

- The registry remains the canonical API source. Generated headers,
  specification text and reflected ABI agree, while independent fixtures accept
  API 0.2/default requests and reject incompatible driver and layer negotiation.
  Installed public C/C++ headers compile consumers and layer ABI 1, 2 and 3
  fixtures without importing implementation types.
- Generated and handwritten loader calls have an explicit interception
  inventory. The validation layer tracks memory, buffers, geometry, simulations,
  pipelines, recorded resources, command state, fences, timelines and query
  scopes. Stable `PM-VAL-*` diagnostics carry typed arguments and application
  object context through a copied, concurrent and reentrant callback contract.
- `PM-API-001` through `PM-RESET-001` separate portable API behavior,
  validation-layer behavior, private host fixtures, reference-runtime
  extensions and reference-solver profiles. Positive, negative and recovery
  cases cover command snapshots, deferred failures, query reuse, independent
  host concurrency, explicit and implicit teardown profiles, runtime-generated
  overflow and selective recovery.

The exit review rebuilt the current Visual Studio 18 D3D12 and validation
targets and passed 23/23 focused tests on the Windows GTX 1080. An independent
GCC 13.3 WSL Slang C++ host build passed 6/6 asynchronous submission, queue,
descriptor snapshot, deferred-failure, destruction-wait and capacity-recovery
tests. A relocated current SDK passed all 18 public consumer and compatibility
cases, including application/extension modes, current and legacy layer ABIs and
invalid negotiation controls. The generated header/specification check, all 67
conformance metadata tests and all 19 roadmap workflow tests also passed.

The review preserves these boundaries:

- Same-object host races remain externally synchronized and are deliberately
  not manufactured as negative tests. Implicit device-teardown draining is an
  in-tree reference-runtime extension, and runtime overflow is qualified only
  for the named shared-Slang neighbor profile on D3D12 and the host fixture.
- Validation does not yet qualify imported native-device/queue handoff; F4-07
  and F4-08 own that contract and mock-adapter evidence. Physical accuracy,
  every driver/domain combination and production evidence remain F5 and P1
  work rather than being inferred from F3 lifecycle passes.
- The broader Windows SDK verifier originally exposed a Debug/Release mismatch
  in its independent conformance consumer. F6-01 now applies the requested
  configuration consistently and separately qualifies the complete relocated
  package; that packaging result does not broaden F3's specification claims.

No uncovered F3-specific leaf was found. Closing F3 establishes the reviewed
specification/ABI/validation foundation only; it does not certify a solver,
driver, engine integration, package release or physical-quality profile.

### F6-99 SDK release and repository ownership exit review

The original F6 exit required a clean checkout to produce a relocatable,
versioned SDK and independently built consumer and driver packages without
private-source or workstation-path dependencies. F6-01 through F6-14 are
merged `done`, carry nonempty reviewable evidence and have completed lease
history. Together they satisfy that scope:

- Package and API version 0.2.0, compatibility rejection, minimal/headless
  presets and independent installed C/C++ consumers are exercised on Linux,
  Windows and Apple Silicon macOS. The external mock driver proves the public
  driver ABI without importing implementation types.
- Loader, validation, preparation, scene, driver, conformance and benchmark
  exports have independent positive and negative component checks. The
  Gallery remains separately gated by `PARALLEL_MATER_BUILD_GALLERY` as an
  example application built on the public API; its types are deliberately not
  installed with the physics SDK.
- Standalone conformance owns its complete source closure and validates the
  relocated installed corpus. One deterministic updater maintains every
  manifest hash, and installed text evidence rejects capture-host paths.
- Two isolated release builds record source, toolchain, dependency and option
  identity, compare matching artifact sets and audit installed runtime
  metadata for workstation paths. The later F6-14 Linux audit passes for 173
  installed files after the stale corpus hashes and captured absolute paths
  found by the first two exit-review attempts were fixed.
- Legacy launcher, branding, hinge, capture/replay and rigid-contact work was
  archived and adjudicated before old worktrees were retired. The retained
  hinge candidate was rejected on measured penetration, and no failing asset,
  solver change or golden was promoted merely to close the foundation.

Current clean Linux source `34072593338c19bc5d3eb37e03d70d16e6e75c9c`
passes 47/47 `sdk-ci` CTests, all 12 relocated-install stages, 44/44 supported
and negative component checks, and 87/87 conformance metadata/unit tests. The
tracked [F6-99 evidence](roadmap/evidence/F6-99-exit-review.json) ties those
results to the prior Windows, macOS and two-build release records.

This closes package reproducibility in the recorded input sense, not
byte-for-byte reproducible binaries: the current two-build Linux record has one
differing installed conformance ELF and ten generated test manifests containing
their distinct build paths. Cross-host and cross-toolchain binary identity is
also unclaimed. The installed preparation and scene libraries do not promise a
stable C++ ABI. Driver exports establish package integrity only; the generated
capability index still has 0 qualified records, so physical correctness,
production vendor-driver qualification and performance remain F5/P1/P3 work.
F7-18 and F7-99 retain full Android Gallery/device qualification, while P2 owns
Unity, Unreal and Godot integrations and P4 owns independent vendor and standards
handoff. No uncovered F6 leaf remains.

### F6-01 Windows SDK installation qualification

Clean source `6994b6c658ca81dd62234e93b3c79df5aa8af0cc` built the Release
`sdk-ci` profile on Windows 11 x64 with Visual Studio Build Tools 18.9.1,
MSVC 19.51.36256.0, Windows SDK 10.0.26100.0, CMake 4.4.2 and Ninja 1.12.1.
All 38 headless SDK CTests passed. The installed-SDK verifier then installed and
moved 154 files, passed all 12 stages, passed all 18 external C/C++ and ABI
cases, rejected package lines 0.1 and 0.3, rebuilt conformance against only the
relocated public package, validated all 30 inputs and preserved the unsupported
and missing-driver classifications.

The [tracked F6-01 evidence](roadmap/evidence/F6-01-windows-sdk-install.json)
pins the source, toolchain, binary/report hashes and reproduction commands. The
Windows x64 job in `sdk-install.yml` repeats the same minimal build, `sdk-ci`
build/CTest and relocated-install verifier in CI and retains its raw report and
logs. This qualifies packaging and public ABI behavior only; no physical quality
or native GPU-driver claim follows from it.

### F6-03 optional SDK package exports

Clean source `f859ef46813f45b72ce14fa590226c3b6ea4b8e9` separates the former
combined export into loader, validation, preparation, scene, driver,
conformance and benchmark install components. Relocated Windows `sdk-ci` passes
44/44 supported and negative component checks. A full D3D12 build passes 49/49,
including an explicit `ParallelMater::slang_d3d12_driver` assertion. The
existing full-install verifier remains green at 12/12 and all 38 `sdk-ci` CTests
pass.

The [tracked F6-03 evidence](roadmap/evidence/F6-03-package-components.json)
pins the clean reports and reproduction commands. SDK CI repeats the driver-free
matrix on Linux, Apple Silicon macOS and Windows; shader portability repeats it
for built CPU, D3D12 and Metal drivers and requires their exact imported target
names. Absent components and a scene install without preparation must fail at
configuration. These checks qualify packaging boundaries only, not physical
quality, capability coverage or a stable C++ ABI.

### F6-05 release reference artifacts

The reference release uses a tracked CPython 3.13.15/package/font lock instead
of host-installed fonts. All four generated header copies, searchable
specification text, API coverage, two PDFs and two layout manifests identify or
are hash-bound to registry revision
`13c217d668d5c007a3aa7d5e6b4d580753f65bafbad64004848716c6c12030d7`.
The clean staging check reproduced all 10 files byte-for-byte with no accepted
nondeterminism. It also passed 6 MSVC C11/C++17 ABI probes, checked all 118
functions and 143 non-Vulkan public types, and validated 52 specification pages,
9 reference pages, embedded fonts, links, page bounds and non-overlap.

The [tracked F6-05 evidence](roadmap/evidence/F6-05-reference-artifacts.json)
and [release artifact manifest](reference/release-artifacts.json) pin the exact
inputs, output hashes and reproduction command. Reference-artifact CI repeats a
clean Ubuntu regeneration. The optional Vulkan header compile probe was skipped
locally because this host lacks Vulkan SDK headers; generation and byte
comparison still include that header. This qualifies document release
artifacts, not driver execution, physics quality or performance.

The [Windows physics report](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json)
records tested source/binary/case hashes, original first divergences, comparison
results, independent physical metrics and reproduction commands. See also the
[shared correction and regression scope](slang-unification-validation.md#current-cross-driver-qualification-2026-10-16).

### F5-99 independent conformance and quality exit review

The F5-99 review is complete, but the original F5 exit is not a blanket pass.
F5-01 through F5-56 are all merged `done`, carry nonempty evidence and have
completed lease history. Together they establish an installed, standalone
conformance runner; immutable case, lifecycle and analytic profiles; 46 exact
capability-obligation rules; quality-gated performance admission; and a
revisioned evidence index. Compilation, execution, reference regression,
adverse lifecycle, analytic quality and timing remain separate evidence.

The current generated index is the controlling boundary: it evaluates 47
reports from five drivers, has zero index-validation failures and qualifies
only 1 of 336 report-scoped capability occurrences. The other 335 occurrences
remain unqualified and produce 990 retained failures. `qualification_complete`
means that every indexed report was evaluated; it does not mean every
capability passed. The one exact current pass cannot certify Metal generally,
and historical per-feature passes from different source revisions cannot be
combined to certify CUDA, D3D12, Vulkan, Metal or Slang CPU on this candidate.

Apple M4 verification of candidate `92486e4a5b8054089d20d60942176721fde473a8`
builds the shared Slang Metal kernels, driver, standalone conformance runner
and Gallery. All 112 macOS Metal preset tests pass; 15 unsupported cases remain
skips rather than passes. The review also made the immutable
`rigid-brick-penetration-v1` profile a required Metal CTest gate. Metal passes
the unchanged 2 mm bound at 0.178732 mm maximum penetration; a supplemental
Slang CPU run passes at 0.604257 mm. Raw state identity is retained for the
stable constraint comparisons but is not used as a false oracle for a chaotic
stacked-contact trajectory.

Accordingly, the installed-suite half of the original exit is met, while the
requirement that every advertised mode have current correctness and quality
evidence is not. P1-01 becomes the owner of the exact current driver/domain
inventory and same-revision implementation/evidence split after foundation
reviews. Existing P1 same-family and material cards plus P1-59 retain known
implementation work. F5-23 already split and completed the missing-contract
leaves, so this review creates no duplicate aggregate task. The machine-readable
[F5-99 evidence](roadmap/evidence/F5-99-exit-review.json) records the decision,
counts, physical scope, repairs and unresolved limits.

### F5-11 current Windows driver evidence

Clean source `2b04e55ef580c305e960c98134b3c5d4b651a179` was rebuilt with
MSVC 19.44.35228 and Slang 2026.18 on a GTX 1080 with NVIDIA driver
582.66; CUDA additionally used toolkit 12.9.86. Public capability queries
returned 49 CUDA and 10 D3D12 records. Revisioned
[CUDA](../conformance/v1/evidence/windows-gtx1080-slang-cuda-2b04e55-evidence.json)
and [D3D12](../conformance/v1/evidence/windows-gtx1080-slang-d3d12-2b04e55-evidence.json)
bundles retain capability keys, binary hashes, all comparison differences,
quality results, lifecycle attempts and raw CTest output.

CUDA reference regression records 11 passed and 19 failed cases. Five failures
(`rope-cloth`, `rope-fluid`, `rope-soft-body`, `smoke-water`,
`soft-body-fluid`) repeatably terminate with Windows status `0xc0000006` and
emit no case result. D3D12 records 5 passed, 12 failed and 13 unsupported
cases. Neither tolerance nor golden changed. CUDA passes the 18-sample
equal-fluid profile; D3D12 reports it unsupported. Both drivers pass
`rigid-brick-penetration-v1` at 0.205278 mm on frame 196.

Capacity console tests pass both drivers. CUDA's complete interaction graph and
fluid-pair graph console tests pass; D3D12 reports both unsupported. Those
tests do not emit the immutable F5-03 checkpoint fingerprints, so exact adverse
lifecycle evidence remains missing instead of being promoted to a pass.
Qualification therefore reports 0/49 CUDA and 0/10 D3D12 records for this
Windows revision while retaining Linux's independent 3-record CUDA result.

### P1-59 current Windows D3D12 common-domain parity

Clean implementation source `cbba6a467bef657feb38fa90767371be9c409ff5`
was rebuilt on the physical GTX 1080 with NVIDIA driver 582.66, Slang 2026.18
and MSVC 19.51.36256. The D3D12 public capability query now returns 46 API
0.2 records and advertises the existing rope, soft-body, smoke, paint and
cross-domain schedules. CUDA returns 49 records at the same source; the only
D3D12 omissions are the three separate-system equal-fluid modes.
Reflection-generated DXIL bindings execute the common Slang schedules through
the thin D3D12 command/resource adapter; no Windows-specific physics or unsupported
dispatch stub supplies the newly enabled domains.

The exact D3D12 build executes all 30 conformance cases with zero unsupported
or failed executions. All 28 canonical Gallery scenes pass 300 frames
(1,345.85 seconds total). A separate real-window run observes every scene and
reset for at least five seconds and exercises startup, scene switching, F/Tab
paths, point interaction/release, GPU-only rendering and the applicable
surface, particle, diagnostic and cloth-structure controls without a crash.
All 22 applicable analytic profiles and all 17 applicable lifecycle profiles
pass capture and independent evaluation.

Those execution results are not blanket qualification. Against the unchanged
pinned CUDA goldens, D3D12 passes 18/30 and fails 12/30. Of the 27 cases for
which same-revision CUDA emits output, D3D12 matches 26 and differs on
`cloth-tear`; CUDA still crashes in `soft-body-fluid`, `rope-soft-body` and
`rope-cloth` with Windows status `0xc0000006`. No tolerance, case, golden,
quality profile or lifecycle profile changed. The
[P1-59 evidence](roadmap/evidence/P1-59-d3d12-common-domain-parity.json) pins
the hashes, exact failure lists, profile inventory and D:-resident raw
artifacts. The current D3D12 result therefore remains 0/46 qualified records.

### F5-12 current macOS Metal driver evidence

Clean source `aac6d595e4821bace3ed77cbbf47f1cad794e548` was rebuilt with
Apple Clang 21.0.0, Apple Metal 32023.883 and Slang 2026.18 on a 16 GB
MacBook Air `Mac16,12` with Apple M4 and macOS 26.5.2. The exact native driver
binary SHA-256 is retained in the revisioned
[Metal evidence bundle](../conformance/v1/evidence/macos-m4-slang-metal-aac6d59-evidence.json),
without serial numbers or other private device identifiers. A new shared
`parallel-mater-conformance --capabilities` mode records the 46 keys returned
by the public API rather than inferring them from source or Gallery scenes.

All 30 public-API trajectories complete without a crash or unsupported result.
Against the unchanged CUDA v1 reference, 14 pass and 16 fail; the bundle keeps
every structured difference. The independent eight-brick profile passes at
0.146627 mm on frame 5 against the unchanged 2 mm limit. Capacity and complete
interaction-graph console tests pass, while the immutable F5-03 checkpoint
results remain explicitly missing. Metal correctly does not advertise the
three separate-system equal-fluid records, so those tests skip without a CPU
fallback. The current obligation profile therefore reports 0/46 qualified
Metal records rather than converting partial evidence into certification.

The full 80-test build also exposes a supplemental Gallery packaging defect:
five launch tests cannot find the bundled `libparallel-mater.0.2.dylib` soname
link. Native Metal API, kernel, async-submission and lifecycle tests pass. That
bundle failure is retained in the evidence but is not treated as physics proof
or a physics failure.

### F5-14 current iPadOS Metal driver evidence

Clean source `9f7e72e7f9d4f44f10527dea7d3f1f7bc7d8d148` ran on physical
iPad model `iPad13,8` with Apple M1 GPU, iPadOS 26.6.2, Metal 4 and Apple7.
The [qualification record](../conformance/v1/evidence/ipados-m1-slang-metal-9f7e72e-qualification.json)
pins AppleClang 21.0.0.21000101, Xcode 26.6, iPhoneOS SDK 26.5 and Slang
2026.18, along with exact application and embedded-driver SHA-256 identities.
The [capture index](../conformance/v1/evidence/ipados-m1-slang-metal-9f7e72e/capture-index.json)
preserves and hashes all 42 raw device and evaluator files without private
device identifiers.

All 30 public-API trajectory executions completed. The shared reference check
passes 15 and fails 15, preserving every structured difference. The independent
rigid profile passes at 0.146627 mm against the unchanged 2 mm limit. The
separate-system equal-fluid fixture remains unsupported because Metal does not
advertise those capabilities. Both adverse lifecycle cases retain 11 missing
checkpoint failures even though their public-API operations completed. The
obligation profile therefore reports 0/46 qualified iPad Metal records; no
failure or missing contract is promoted to support.

This is public-API physics evidence only. Gallery controls, rendering, thermal
behavior and timing remain separate claims and are explicitly excluded from
the capture index.

### F5-05 brick penetration classification

The retained eight-brick reproduction failed before F5-10 on both adapters at
7.02056 mm (CUDA, frame 101) and 7.21844 mm (D3D12, frame 102), against the
existing 2 mm limit. The authored scene, brick generator and independent
closed-mesh sampling oracle were unchanged by the correction, so the fixture
was not the source of the failure. Focused contact tests isolated missing convex
face support and false concave/open seam normals in contact generation. Separate
free-rotation and tangent-response tests isolated cancellation in relative
rotation/anchor updates and backend-sensitive radial projection in solver
response. F5-10 applied those bounded shared-Slang corrections; neither adapter
received a backend-specific physics fork and the global iteration budget did
not change.

The corrected run measures 0.205278 mm on both adapters. The immutable
`rigid-brick-penetration-v1` profile now fixes the scene, body count, frame and
substep counts, timestep, independent oracle and 2 mm limit. Its evaluator
rejects altered fixtures, incomplete execution, invalid frames, nonfinite or
negative depths and excessive penetration. This closes the F5-05 investigation.
F5-11 now registers current-build CUDA and D3D12 measurements without changing
the profile, tolerance, fixture or shared solver.

### F5-06 CUDA/D3D12 divergence classification

The revisioned Windows report retains every original first difference, all 17
case hashes and both qualified driver hashes. The ten baseline failures reduce
to these primary causes:

| Primary reduced cause | Cases |
|---|---|
| Relative pose/anchor cancellation and solver response | `constraint-fixed`, `constraint-generic-spring`, `constraint-generic`, `constraint-hinge`, `constraint-motor`, `constraint-point`, `passive-active` |
| Contact feature and manifold classification | `cloth-tear`, `fluid-rigid` |
| Cross-target floating-point compilation contract | `rigid-direct` |

These groups identify the earliest independent reproduction, not mutually
exclusive later symptoms: small pose or arithmetic drift can change a later
contact branch. The report records this qualification explicitly and preserves
unchanged inputs, tolerances and historical goldens. JSON/XML contract hashes
now normalize only Git's CRLF checkout conversion to canonical LF in both the
Python and native C++ checkers; binary hashes and every other byte remain exact.

Reports must distinguish compilation, execution, reference regression, analytic
or invariant physical quality, and performance. An unsupported case counts as
neither a pass nor a numerical failure, and leaves its qualification scope open.
Performance comparisons require both implementations to pass the same fixed
quality profile first. Raw case outputs and failures survive the comparison.

### F5-17 rigid-direct reference adjudication

Clean source `d6310bf9b65d81f8c30b2d45e415c43b82396249` exposed that the
command runner did not execute the geometry declared by the hashed
`rigid-direct` case: its kinematic body reused the dynamic box's 0.25 m
half-extent mesh instead of the declared 0.2 m mesh. The retained F5-04 CPU and
CUDA raws confirm the mismatch through frame-21 contact points at approximately
`z = +/-0.25 m`. The runner now creates the declared kinematic mesh; the case,
shared solver, backend adapters and `direct` tolerances are unchanged.

The independent
[`rigid-kinematic-sweep-v1`](../conformance/v1/quality/rigid-kinematic-sweep-v1.json)
profile pins the case hash and all 61 checkpoints. It checks collider bounds,
contact direction, the Coulomb cone, non-vacuous linear/angular response and
planar symmetry without reading a trajectory golden. Corrected CPU and CUDA
captures both pass with zero failures; the retained pre-correction captures
each fail five collider-bound checks. The
[versioned adjudication](roadmap/evidence/F5-17-rigid-direct-adjudication.json)
therefore replaces only the stale CUDA `rigid-direct` golden, from SHA-256
`c79c9a768c1b3f0d60d7ec5af7e1a8a69977d3e5e7c7c5558c5352c65617208d`
to `6ebfbcc7be7cbfb5363a4fc29029bc2ba9d1f96ce03c4ab7334af9ade45ac5ba`.

CUDA passes the F5-17 replacement at all five canonical checkpoints. At that
revision CPU still failed one unchanged-tolerance field at checkpoint 40:
`dynamic-box.angular_velocity[1]` differed by `2.4093271349556744e-05 rad/s`
against an allowed `1.000034577675251e-05 rad/s`. The retained all-frame report
preserves the same component on frames 21 through 40. F5-18 closes that bounded
divergence below; the five known `0xc0000006` process failures and all other
F5-11 limitations remain. This evidence grants no broader CPU, CUDA, Gallery,
performance or driver qualification.

### F5-18 rigid-direct planar symmetry

Clean source `f33707a649399f5630b687a50eabfae7f6cef79e` reproduced the
F5-17 failure from the same case and profile hashes. The frame-21 face manifold
contained the four rectangular patch endpoints plus a fifth collinear interior
point at `z = -0.082687892 m`, contributed by the incident face's internal
triangle diagonal. That unpaired point made the AVBD block depend on face
tessellation. The shared Slang face-manifold path now removes only points lying
strictly inside a segment between two other patch contacts. It does not change
the fixture, tolerance, backend adapters or output values after solving.

The corrected CPU and CUDA captures each contain 61 checkpoints and a mirrored
four-contact frame-21 patch at `z = +/-0.2 m`. They compare with zero differences
at the unchanged `direct` tolerances, and both pass all 61 samples of the pinned
`rigid-kinematic-sweep-v1` oracle. Maximum reported planar leakage is
`1.1697111e-06` on CPU and `6.6892048e-06` on CUDA. A focused CUDA/D3D12
regression additionally produces `1.21739 rad/s` from a real `0.02 rad` tilted
contact, proving that the reduction does not discard three-dimensional angular
response. Both backends pass 300 rigid Gallery frames, reset, torque-free,
convex-sweep, face-normal, brick-penetration, seam and open-surface gates.

Because the correction changes the physical post-impact trajectory, only the
CUDA `rigid-direct` golden is replaced, from LF-normalized SHA-256
`6ebfbcc7be7cbfb5363a4fc29029bc2ba9d1f96ce03c4ab7334af9ade45ac5ba`
to `ab9fd6f8deef860cb415d0b43633d367678f3acb6993d1007ab4d97fce80147a`.
The [versioned F5-18 evidence](roadmap/evidence/F5-18-rigid-direct-planar-symmetry.json)
retains the pre-correction failure, current raw captures, independent quality
reports and both reference comparisons. It closes only the bounded F5-17
planar-symmetry limitation and grants no broader driver or performance claim.

### F5-25 fluid self-contact quality

Clean source `990531ccc09c82013909842750d6d83e1c5701bd` adds the immutable
`fluid-self-contact-v1` profile without adding a solver or backend path. Its
public-API fixture places two equal particles inside their declared exclusion
distance with zero gravity, viscosity and damping. The independent evaluator
uses only the emitted public states to require stable identity and ordering,
95 percent of the two-radius exclusion distance, non-vacuous outward response,
balanced internal impulse, fixed center of mass and bounded transverse leakage.

Physical GTX 1080 CUDA and D3D12 runs each pass the single sample at
`0.21250000596046448 m` separation against the `0.19 m` minimum. Both report
`18 kg*m/s` exchange activity and zero net internal impulse, center-of-mass
drift and transverse leakage. The [versioned F5-25 evidence](roadmap/evidence/F5-25-fluid-self-contact.json)
pins the profile, source, runner, driver binaries, raw public output and reports.

The obligation matrix binds the analytic profile only to
`FLUID/FLUID/SELF/CONTACT/BIDIRECTIONAL/ENDPOINT_NATIVE`. The existing fluid
lifecycle trajectory and adverse capacity/replacement contract remain separate
requirements. Pressure and viscosity keep separate exact rules and are
qualified independently by F5-26 and F5-27. The rebuilt index retains all
historical failures and leaves every unrecaptured driver unqualified.

### F5-26 fluid self-pressure quality

Clean source `c6deaa06461eefe22cf9e165374ad8166e7e7037` adds the immutable
`fluid-self-pressure-v1` profile without adding a solver or backend path. Its
public-API fixture places two equal particles at half their isotropic rest
spacing with zero gravity, viscosity and damping. Because the API deliberately
does not expose driver-private density, the independent evaluator derives SI
density as authored particle mass divided by the cube of public pair spacing.
It also requires stable identity and ordering, non-vacuous outward response,
balanced internal impulse, fixed center of mass and bounded transverse leakage.

Physical GTX 1080 CUDA and D3D12 runs each pass the single sample. Both move the
independent density estimate from `7999.999642372142 kg/m^3` to
`1572.1888607632552 kg/m^3` toward the authored `1000 kg/m^3` rest density,
reducing relative density error by `0.9182587300005414`. Both report zero net
internal impulse, center-of-mass drift and transverse leakage. The
[versioned F5-26 evidence](roadmap/evidence/F5-26-fluid-self-pressure.json)
pins the profile, source, runner, driver binaries, raw public output and reports.

The obligation matrix binds the analytic profile only to
`FLUID/FLUID/SELF/PRESSURE/BIDIRECTIONAL/ENDPOINT_NATIVE`. The exact fluid
lifecycle trajectory and adverse capacity/replacement/rollback contract remain
independent requirements; this analytic fixture cannot substitute for them.
Self-viscosity retains its separate exact analytic contract under F5-27, and
historical or unrecaptured drivers remain unqualified.

### F5-27 fluid self-viscosity quality

Clean source `d08ca42c2f1e67ca02ac8f1bfe20f3d7c123706f` adds the immutable
`fluid-self-viscosity-v1` profile without adding a solver or backend path. Its
public-API fixture gives two equal particles symmetric separating velocities
with zero gravity, pressure and damping. The independent evaluator derives
relative speed, kinetic energy and impulse only from public particle states and
authored mass. It requires non-vacuous relative-motion and energy dissipation,
no energy increase, balanced internal impulse, fixed center of mass, stable
identity and ordering, and bounded transverse leakage.

Physical GTX 1080 CUDA and D3D12 runs each pass the single sample. Both reduce
relative speed from `2.0 m/s` to `1.7000000476837158 m/s` and kinetic energy
from `8.0 J` to `5.780000324249272 J`, reductions of `0.1499999761581421` and
`0.277499959468841`. Both report zero net internal impulse, center-of-mass drift
and transverse leakage. The
[versioned F5-27 evidence](roadmap/evidence/F5-27-fluid-self-viscosity.json)
pins the profile, source, runner, driver binaries, raw public output and reports.

The obligation matrix binds the analytic profile only to
`FLUID/FLUID/SELF/VISCOSITY/BIDIRECTIONAL/ENDPOINT_NATIVE`. The exact fluid
lifecycle trajectory and adverse capacity/replacement/rollback contract remain
independent requirements; this analytic fixture cannot substitute for them.
Self-contact and self-pressure retain their separate exact analytic contracts,
and historical or unrecaptured drivers remain unqualified.

### F5-28 sampled rope self-contact quality

Clean source `f08b130193b0f0982bc9e1384afd5cd66acede82` adds the immutable
`rope-self-contact-v1` profile without adding or changing a solver/backend path.
Its canonical public-API fixture folds four sampled nodes so indices zero and
three start at `0.019999999552965164 m`, well inside their combined `0.12 m`
material radius, while the topology-neighbor pair one and three remains inside
the same radius. The independent evaluator consumes only the public input and
queried rope output. It checks exact node ordering, eligible pair separation,
the index-distance-two exclusion, center-of-mass drift, transverse leakage and
the authored speed bound.

The physical GTX 1080 CUDA run passes the single sample: the eligible pair ends
at `0.11999999507617162 m` against the `0.114 m` retained minimum, with
`9.315374889959498e-10 m` center-of-mass drift and zero transverse leakage.
At the F5-28 revision D3D12 did not advertise rope, so its failed unsupported
result remains retained and grants no qualification. The [versioned F5-28 evidence](roadmap/evidence/F5-28-rope-self-contact.json)
pins the profile, source, runner, validation layer, driver binaries, raw public
output, passing CUDA report and D3D12 failure.

The obligation matrix binds the analytic profile only to
`ROPE/ROPE/SELF/CONTACT/BIDIRECTIONAL/ENDPOINT_NATIVE`. The unchanged
`case:rope-core` trajectory/lifecycle obligation remains independently required;
this analytic fixture cannot substitute for capacity, replacement or rollback
evidence. Nodes separated by two topology indices or less remain excluded, and
no continuous segment, knot, splice or inter-rope contact behavior is claimed.
Historical and unrecaptured drivers remain unqualified.

### F5-29 smoke self-pressure quality

Clean source `7356e02212d83c9bc61f0ca88d3f7f27f41e92ff` adds immutable
particle and projected-grid fixtures to `smoke-self-pressure-v1` plus the exact
`smoke-self-pressure-lifecycle-v1` operation sequence. Both fixtures use only
public SDK inputs and public smoke particle output. The particle oracle checks
the public density-pressure equation, bounded speed and mean-velocity drift;
the grid oracle checks projected pressure activity through public tracer
samples under its explicitly authored emitter and boundary field.

Physical GTX 1080 CUDA passes all 96 pressure samples, the canonical
`smoke-grid` golden comparison and all 10 lifecycle checkpoints. Particle
pressure reaches the authored clamp of `100.0` with zero equation error and
`0.047486430593880344 m/s` mean-velocity drift under the retained `0.05 m/s`
limit. Projected-grid tracer pressure reaches `0.003504612948745489`. At the
F5-29 revision D3D12 did not advertise smoke; its two exit-77 results remain
retained and grant no qualification. The [versioned F5-29 evidence](roadmap/evidence/F5-29-smoke-self-pressure.json)
pins both profiles, the clean source, runner, validation layers, production
drivers, raw public outputs and evaluator reports.

The obligation matrix binds only
`SMOKE/SMOKE/SELF/PRESSURE/BIDIRECTIONAL/SMOKE_PARTICLE` and
`SMOKE/SMOKE/SELF/PRESSURE/BIDIRECTIONAL/SMOKE_GRID`. The `smoke-grid`
trajectory, both representation-specific lifecycle cases and the analytic
profile remain independent requirements. Smoke self-viscosity retains its
separate missing contract, and historical or unrecaptured drivers remain
unqualified.

### F5-50 rope-to-smoke contact quality

Clean source `84a475e8a17102f9b38697bcaeb784bd68a41a47` adds the immutable
`rope-smoke-contact-v1` profile and two exact cases in
`rope-smoke-contact-lifecycle-v1`, without changing a solver or backend path.
The paired public-API fixture drives particle and projected-grid smoke from
outside a dynamic two-node rope capsule into its explicit `0.015 m` radius.
Its independent oracle derives closest-segment distance, interpolated rope
velocity, relative radial rejection, tangential retention, speed bounds and
rope invariance only from public rope-node and smoke snapshots.

Physical RTX 3050 Ti CUDA passes both analytic samples, the canonical
`smoke-rope` reference comparison and all ten lifecycle checkpoints. Both
representations move from `0.020010764323243238 m` outside the rope to a
control distance of `0.010021511972323802 m`, while CONTACT holds them at
`0.015000000072806125 m`. Relative radial velocity is
`-1.849310753490596e-08 m/s`, tangential speed is
`0.2584348960133991 m/s`, and every public rope position and velocity
difference is zero. The [versioned F5-50 evidence](roadmap/evidence/F5-50-rope-smoke-contact.json)
pins the clean source, production driver, raw public output and reports.

The obligation matrix binds only the exact particle and grid
`ROPE/SMOKE/DISTINCT_FAMILY/CONTACT/FIRST_TO_SECOND` records. Each exact
representation-specific lifecycle contract remains required alongside the
shared analytic profile and `smoke-rope` trajectory. The two inverse-direction
DRAG records keep separate explicit missing rules for F5-51; CONTACT evidence
cannot qualify them. Historical and unrecaptured drivers remain visible and
unqualified.

### F5-51 smoke-to-rope drag quality

Clean source `0dc9fbf2a35181b29b6b151d87289d173158a971` adds the immutable
`smoke-rope-drag-v1` profile and two exact cases in
`smoke-rope-drag-lifecycle-v1`, without changing a solver or backend path.
Its paired public-API fixture places a free dynamic two-node rope outside all
smoke tracers, then compares zero-drag controls with forward, stationary and
reverse carrier flow. The independent oracle derives signed rope response,
acceleration, symmetry, transverse drift, segment preservation, tracer
clearance and one-way smoke invariance only from public snapshots.

Physical RTX 3050 Ti CUDA passes all six analytic samples, the canonical
`smoke-rope` reference comparison and all ten lifecycle checkpoints. Particle
forward and reverse response fractions are `0.09999871253967285` and
`0.10000020265579224`; projected-grid response is
`0.009642541408538818` in both directions. Maximum accelerations remain at
`10.000020265579224 m/s^2` and `1.0490417491416375 m/s^2`, stationary
response and every paired public smoke-state delta are zero, and the minimum
tracer-to-segment clearance is `0.10411272626082052 m`. The
[versioned F5-51 evidence](roadmap/evidence/F5-51-smoke-rope-drag.json)
pins the clean source, production driver, raw public output and reports.

The obligation matrix binds only the exact particle and grid
`ROPE/SMOKE/DISTINCT_FAMILY/DRAG/SECOND_TO_FIRST` records. Each exact
representation-specific lifecycle contract remains required alongside the
shared analytic profile and `smoke-rope` trajectory. The two opposite
rope-to-smoke CONTACT records retain their separate evidence; CONTACT cannot
substitute for DRAG. Historical and unrecaptured drivers remain visible and
unqualified.

### F5-52 rope-rigid contact quality

Clean source `6ca716beb0c21e51a78a4d509ab9608b2449fd4f` adds the immutable
`rope-rigid-contact-v1` profile without changing a solver or backend path. A
public preparation step drives a free two-node rope segment into a centered
dynamic cube beside a separated control. The independent oracle reconstructs
capsule-to-mesh clearance and checks coherent reduced-mass impulse, reciprocal
rope/body momentum, public contact-force consistency, frictionless slip,
closing speed, rotation, energy and speed only from public snapshots.

Physical RTX 3050 Ti CUDA passes both analytic samples. The impact retains
`0.02002456114350934 m` capsule clearance, exchanges
`0.49873446114361286 N s`, and has reciprocal and force-impulse residuals of
`6.146728992462158e-08 N s` and `1.1265277866101542e-07 N s`. Tangential
impulse, final closing speed, angular speed, control momentum and control
contact force are all zero. The first physical attempt is retained: an exactly
one-spacing segment could round to three sampled nodes and was rejected by the
two-node capacity; the corrected `0.02 m` segment removes that ambiguity.

The existing `rope-core` case completes all three checkpoints and repeats
byte-identically, but its current physical output retains 242 differences from
the historical CUDA golden. The first divergence is rigid-only contact
topology outside this exact oracle. No tolerance or golden changes here;
F5-56 owns the required same-revision CUDA/Metal adjudication. The exact CUDA
record therefore remains unqualified, while its advertisement is retained
because the independent physical contract passes. The
[versioned F5-52 evidence](roadmap/evidence/F5-52-rope-rigid-contact.json)
pins both attempts, binaries, raw outputs, reports and hashes.

The obligation matrix binds the analytic profile only to
`ROPE/RIGID_BODY/DISTINCT_FAMILY/CONTACT/BIDIRECTIONAL/ENDPOINT_NATIVE` and
retains `case:rope-core` independently for trajectory and lifecycle evidence.
Rope-rigid ATTACHMENT remains a separate exact rule; its F5-53 analytic
contract and physical result are documented below. Historical and unrecaptured
drivers remain visible and unqualified.

### F5-53 rope-rigid attachment quality

Clean source `1b362ad54f8842a503d98bacd8c30f625477e368` adds the immutable
`rope-rigid-attachment-v1` profile without changing a solver or backend path.
Its public-API fixture attaches the first endpoint of a five-node rope directly
to an off-center `0.2 m` local anchor on a force-driven dynamic cube. Over four
zero-gravity steps, the independent oracle reconstructs the transformed anchor
from public rigid state and checks endpoint drift, free-rope and rigid momentum,
public constraint-force impulse, angular reaction, rope length and inactive
contact state without reading driver-private data.

Physical RTX 3050 Ti CUDA passes the analytic sample twice with byte-identical
raw output. Final attachment drift is `2.250640657308347e-09 m`. The body
attachment impulse is `[0.0002832858881447464, -0.01702868938446045, 0] N s`;
reciprocal linear and public constraint-force residuals are
`1.6535544015115795e-07 N s` and `1.144409181158083e-07 N s`. The off-center
anchor produces `[0, 0, -0.003386819362640381] N m s` angular reaction with
`2.78533134179422e-06 N m s` residual and `4.021114826202393 rad/s` rigid
angular speed. Contact force remains zero and maximum rope-length error is
`2.126754256731589e-08 m`.

The existing `rope-core` case again completes all three checkpoints, but the
comparison retains the same 242 differences and byte-identical failed report
from F5-52 against the historical CUDA golden. No tolerance or golden changes
here; F5-56 owns the required same-revision CUDA/Metal adjudication. The exact
CUDA record therefore remains unqualified, no passing result is registered in
the evidence-source registry, and its advertisement is retained because the
independent physical contract passes repeatably. The
[versioned F5-53 evidence](roadmap/evidence/F5-53-rope-rigid-attachment.json)
pins the clean source, production driver, binaries, raw outputs and reports.

The obligation matrix binds the analytic profile only to
`ROPE/RIGID_BODY/DISTINCT_FAMILY/ATTACHMENT/BIDIRECTIONAL/ENDPOINT_NATIVE` and
retains `case:rope-core` independently for trajectory and lifecycle evidence.
The analytic fixture cannot substitute for that exact binding, and the separate
rope-rigid CONTACT profile cannot qualify ATTACHMENT. Historical and
unrecaptured drivers remain visible and unqualified.

### F5-54 smoke-to-rigid contact quality

Clean source `ec024d7ea0010f25654dab170d4ad8bc607aa6e9` adds the immutable
`smoke-rigid-contact-v1` profile and two exact cases in
`smoke-rigid-contact-lifecycle-v1`, without changing a solver or backend path.
Its paired public-API fixture drives particle and projected-grid smoke into a
dynamic rigid cuboid, then compares each coupled run with a zero-contact
control. The independent oracle derives face clearance, relative normal
velocity, outward smoke response, reciprocal rigid impulse, angular response
and speed bounds only from public smoke and rigid snapshots.

Physical GTX 1080 CUDA passes both analytic samples, the canonical
`smoke-grid` reference comparison and all ten lifecycle checkpoints. Both
representations finish at `0.015000000149011612 m` face clearance with only
`1.4901161207725444e-10 m` error. Particle and grid smoke produce outward
responses of `1.0 m/s` and `0.994413879700005 m/s`; reciprocal rigid impulses
are `0.0009999999403953552 N s` and `5.4725231602787975e-05 N s`, with
non-zero angular responses. At the F5-54 revision D3D12 did not advertise
smoke and rigid support; its two exit-77 results remain retained and grant no
qualification. The
[versioned F5-54 evidence](roadmap/evidence/F5-54-smoke-rigid-contact.json)
pins both profiles, the clean source, runner, validation layer, production
driver, raw public outputs and evaluator reports.

The obligation matrix binds only the exact particle and grid
`SMOKE/RIGID_BODY/DISTINCT_FAMILY/CONTACT/BIDIRECTIONAL` records. Each exact
representation-specific lifecycle contract remains required alongside the
shared analytic profile and `smoke-grid` trajectory. The separate smoke-to-
rigid DRAG records are now qualified independently by F5-55 evidence. CONTACT
cannot substitute for DRAG, or DRAG for CONTACT. Historical and unrecaptured
drivers remain visible and unqualified.

### F5-55 smoke-to-rigid drag quality

Clean source `b8bc7ee292262d0a26c3532834df7d339ecf2bc0` pins the immutable
`smoke-rigid-drag-v1` profile and two exact cases in
`smoke-rigid-drag-lifecycle-v1`. Six public-state runs pair zero-drag controls
with forward, stationary and reverse carrier velocities for particle and grid
smoke. The independent oracle derives rigid and smoke response, forward/reverse
symmetry, transverse drift, non-contact clearance, stationary null behavior and
speed bounds only from public smoke and rigid snapshots. Particle smoke also
supports reciprocal momentum, force and kinetic-energy checks; public grid
tracers are not the Eulerian field mass, so the grid oracle intentionally makes
no reciprocal-momentum or energy claim.

Physical RTX 3050 Ti CUDA passes all six analytic samples, the canonical
`smoke-grid` reference comparison and all ten lifecycle checkpoints. Particle
forward and reverse rigid response fractions are both
`0.0011470927856862545`; smoke response fractions are both
`0.03584665060043335`. Reciprocal residuals remain below
`3.352761914534979e-11 kg m/s` and `1.6763809572674894e-09 N`, with no
kinetic-energy increase. Grid forward/reverse rigid response fractions are
`0.006723881234397` and `0.007830076848351696`; paired public-tracer response
magnitudes are `0.00016030669212341309` and `0.0002097785472869873`.
Minimum particle and grid clearances are `0.01993628740310671 m` and
`0.09471385176949018 m`, both above the `0.012 m` non-contact floor. A repeated
physical quality capture is byte-identical. The
[versioned F5-55 evidence](roadmap/evidence/F5-55-smoke-rigid-drag.json) pins
the clean source, profiles, runner, validation layer, production driver, raw
public outputs and evaluator reports.

The obligation matrix binds only the exact particle and grid
`SMOKE/RIGID_BODY/DISTINCT_FAMILY/DRAG/BIDIRECTIONAL` records. Each exact
representation-specific lifecycle contract remains required alongside the
shared analytic profile and `smoke-grid` trajectory. Both qualification
results pass `1/1`, while CONTACT evidence remains disjoint and cannot
substitute. Historical and unrecaptured drivers remain visible and
unqualified.
