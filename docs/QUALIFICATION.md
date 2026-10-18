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
| CUDA | [`linux-rtx3050ti-cuda-equal-fluid-qualification-2026-10-17`](../conformance/v1/evidence/linux-rtx3050ti-cuda-equal-fluid-qualification-2026-10-17.json), [`windows-gtx1080-slang-cuda-2b04e55`](../conformance/v1/evidence/windows-gtx1080-slang-cuda-2b04e55-qualification.json) | 0 | Supplemental: [`windows-gtx1080-physics-2026-10-16`](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json); 10 historical baseline failure, 3 limitation, 179 qualification failure |
| D3D12 | [`windows-gtx1080-slang-d3d12-2b04e55`](../conformance/v1/evidence/windows-gtx1080-slang-d3d12-2b04e55-qualification.json) | 0 | Supplemental: [`windows-gtx1080-physics-2026-10-16`](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json); 10 historical baseline failure, 3 limitation, 34 qualification failure, 2 unsupported |
| Vulkan | None | 0 | 1 missing qualification result |
| Metal | [`ipados-m1-slang-metal-9f7e72e`](../conformance/v1/evidence/ipados-m1-slang-metal-9f7e72e-qualification.json), [`macos-m4-slang-metal-aac6d59`](../conformance/v1/evidence/macos-m4-slang-metal-aac6d59-qualification.json) | 0 | 291 qualification failure |

Index status: **0 qualified capability records from 7 revisioned report records**. Supplemental and historical evidence remains visible but never certifies a capability.
<!-- END GENERATED DRIVER EVIDENCE -->

Only the generated block states current per-driver capability qualification.
The scope table below records project-wide contracts and other evidence without
promoting those results to capability certification.

| Scope | Evidence and host scope | Unqualified or failed scope |
|---|---|---|
| Resource lifetime and capacity (F1) | Reference-lifecycle foundation closed by F1-01–F1-07: exact-query fluid churn/recovery; CUDA mixed-domain churn; geometry, paint and borrowed-endpoint replacement; failure/pending lifetime; fixed host owner metadata. Current Windows CUDA/D3D12 and Apple M4 Metal console tests pass 4,096 fluid replacements and exact-capacity recovery. | The console tests do not emit F5-03 checkpoint state hashes/write ordinals, so current results retain explicit `missing` profile evidence. Numerical physics and remaining driver/domain pairs remain separate; see F2-04, F4-05 and P1-01. |
| SDK/loader/host contracts | 34 shader-free `sdk-ci` CTests; relocated C/C++ consumers and separately built mock driver; standalone conformance checks all 30 inputs; Linux and Apple Silicon macOS CI preserve revisioned compiler/architecture reports | Windows installed packages are not run here; physics is not evaluated by this gate |
| Interaction graph lifecycle (F2-04) | Linux RTX 3050 Ti and current Windows GTX 1080 CUDA plus Apple M4 Metal execute all 12 implemented pairs through create/update/clear, endpoint replacement, three-family mutation and partial-create rollback. | Current CUDA and Metal console evidence lacks required F5-03 checkpoint fingerprints; D3D12 reports the complete three-family case unsupported because it does not advertise rope, soft-body or smoke. Unsupported/missing never passes the lifecycle evaluator. |
| Vendor-neutral physical quality (F5-02, F5-05) | Hashed `core-conservation-stability-v1` and `rigid-brick-penetration-v1` profiles independently evaluate fluid conservation/stability and rigid surface separation. Current Windows CUDA passes all 18 equal-fluid samples; CUDA and D3D12 pass the fixed eight-brick gate at 0.205278 mm and Apple M4 Metal passes at 0.146627 mm, all against 2 mm. | D3D12 and Metal do not advertise separate-system equal-fluid modes. Brick quality does not override failed reference regression or missing lifecycle/interaction-specific analytic evidence; remaining modes still need fixtures. |
| Capability evidence obligations (F5-01) | Hashed `capability-obligations-v1` rules cover all 49 API 0.2 interaction records exactly once and keep compilation, execution, regression, lifecycle, analytic and timing evidence separate. Current physical results register 49 CUDA and 10 D3D12 records on GTX 1080 plus 46 Metal records on Apple M4. | Qualification remains 0/49 for current Windows CUDA, 0/10 D3D12 and 0/46 Metal. Reference failures, process crashes, missing exact lifecycle captures and rule-level analytic gaps stay explicit. |
| iPadOS development qualification runner (F5-13) | The opt-in, development-only app channel reuses the canonical 30-case public-API runner, rigid penetration oracle, equal-fluid fixture and adverse lifecycle operations. Collection pins a fresh nonce, source, app/driver hashes, public device/OS/Metal and toolchain metadata and rejects stale, partial, mismatched and Gallery-only reports. | This runner is not enabled by default in ordinary iOS builds. Equal-fluid remains unsupported on Metal and the existing lifecycle fixtures still lack F5-03 checkpoint fingerprints; device execution and evidence publication remain separate gates. |
| Quality-gated performance (F5-07) | Independent headless gate computes throughput, median/p95 timing and public memory peaks from raw samples; every metric carries the fixed quality-profile hash and exact source/binary/device/toolchain tuple | This defines score admission but publishes no production benchmark. Missing, failed, unsupported, stale or mismatched physical-quality evidence leaves measurements unqualified. |
| Distinct fluid–fluid systems | F2-05–F2-10 specify, schedule and qualify equal-material contact/pressure/viscosity with bounded pair storage, combined-population tolerances, balanced momentum, stable-ID ownership and lifecycle recovery. Linux RTX 3050 Ti CUDA qualifies all three records. Current Windows GTX 1080 CUDA independently passes the same 18/18 physical samples and fluid-pair graph lifecycle console test. | Windows CUDA lacks the exact F5-03 checkpoint artifact, so its three records remain unqualified; D3D12 and Metal do not advertise these records, and their tests skip without fallback. Different-material modes remain P1-04–P1-18, and phase transfer remains undefined. |
| Cloth self and cloth–cloth systems | F2-12 specifies separate intrinsic self-contact and graph-owned inter-system contact identities, swept double-sided vertex-triangle/edge-edge behavior, topology exclusions, thickness, friction, conservation, tunnelling and overflow criteria. | API 0.2 exposes neither capability; typed controls, bounded contact generation, self/inter operators, scheduling and per-driver qualification remain P1-19–P1-24. |
| Soft-body self and soft-body systems | F2-13 specifies separate intrinsic and graph-owned surface-contact identities, bound closed-surface sampling, swept primitive contact, topology exclusions, node-radius separation, friction, conservation and bounded failure behavior. | API 0.2 exposes neither capability; typed controls, bounded surface contacts, self/inter operators, scheduling and per-driver qualification remain P1-25–P1-30. Open/non-manifold skins, volume inversion and merging remain unsupported. |
| Rope self and rope–rope systems | F2-14 narrows the API 0.2 self-contact record to sampled-node exclusion and specifies a separate future `rope-capsule` profile for continuous self/inter segment contact plus an independent endpoint attachment mode. | The advertised node-only record still needs the P1-31 physical audit. Capsule contact, pair edges and rope-rope attachment remain unsupported pending P1-32–P1-38; knots, splices, sliding and interior attachments are outside that profile. |
| Smoke self and smoke–smoke systems | F2-15 specifies particle-particle, grid-grid and particle-grid profiles with separate mechanical, thermal and projection behavior; bounded overlap/neighborhood construction; fixed-point and conservation criteria; and graph lifecycle rules. | API 0.2 exposes no smoke–smoke record. Typed profiles/material state, bounded maps, shared Slang operators, scheduling and per-driver qualification remain P1-39–P1-55; combustion, chemistry, phase transfer and topology-changing guarantees remain unsupported. |
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

The exit review reran the current Windows CUDA/D3D12 F2 slice on a GTX 1080:
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

The [Windows physics report](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json)
records tested source/binary/case hashes, original first divergences, comparison
results, independent physical metrics and reproduction commands. See also the
[shared correction and regression scope](slang-unification-validation.md#current-cross-driver-qualification-2026-10-16).

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
