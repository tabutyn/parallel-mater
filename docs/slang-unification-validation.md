# Slang unification validation

This page records the current validation contract for the unified runtime.

## Runtime architecture

- `parallel-mater` exposes the C command API in `parallel_mater.h`.
- The production runtime dispatches physics generated from `src/slang`.
- `src/simulation/runtime.cpp` is ordinary C++ using common device services;
  it contains no CUDA runtime dependencies, GPU kernels or direct launches.
- Device and simulation dispatch adapters contain platform services and
  generated-kernel launches only.
- DirectCompute and Metal backend-specific implementations and galleries have
  been removed. The loadable Vulkan preview runs shared rigid and liquid
  schedules; WebGPU physics remains unavailable.
- The retired hand-written CUDA physics kernels have been deleted. CUDA source
  produced by Slang, the thin CUDA device adapter, CUDA interoperability, and
  OptiX drawing remain.

`parallel-mater-slang-architecture` passes and reports zero native physics
kernels or direct launches in the runtime.

## Completed validation

- Release `parallel-mater` library build: pass.
- Shared Slang CUDA source generation: pass.
- Shared Slang direct PTX generation: pass.
- AVBD CUDA/MSL/HLSL/WGSL/DXIL bundle and reflection validation: pass.
- `parallel-mater-gallery` link through the command API: pass.
- API declaration/runtime coverage generation: pass.
- CPU contract, catalog, input, conformance registry/comparator, contact-friction,
  AVBD-core, architecture, and reflection tests: pass.
- Gallery catalog contains 28 canonical entries; shared input tests verify Tab
  opens the complete catalog and dismisses parameter editing.
- Command API, API contract, gallery catalog/input, Slang architecture, and
  Slang reflection tests pass on the CUDA host.
- All 105 AVBD fixtures and the CPU/CUDA/Slang AVBD-core tests pass.
- The loader-visible D3D12 preview executes shared rigid, constraint, liquid,
  cloth and rigid/liquid/cloth interaction schedules compiled to DXIL. The
  installed gallery's native D3D12 renderer consumes public GPU output, including
  liquid surface/particle drawing and cloth surfaces. It does not upload a CPU
  scene image. Explicit captures and diagnostics still read selected outputs.
- The 2026-10-16 installed-gallery test passed all 13 supported D3D12 scenes,
  switching, resets, F/Tab controls and fluid/cloth display modes, observing at
  least five seconds per scene/load. The installed executable's SHA-256 matched
  the tested build. These are execution/input gates, not physical parity.
- All 42 CUDA gallery headless, action, render, debug, containment, and coupled
  scene tests pass through the shared gallery application path. This includes
  Point release/restore, Hinge, Piston, Motor + Spring, dump tip/drive, rope,
  smoke, cloth tearing/paint, fluid, and soft-body couplings.
- The rigid solver now has one scene-wide authored AVBD budget. The obsolete
  per-body budget and its dormant alternate scheduling path were removed from
  both the Slang layout and the CUDA dispatch mirror.

## CUDA acceptance

The CUDA target was accepted by interactive review of the unified gallery on
the RTX 3050 Ti on 2026-10-12. Exact comparisons with the retired CUDA solver,
its performance profile, and its trajectory goldens are no longer release or
source-deletion gates. Canonical conformance inputs, historical goldens, and
the comparator remain; a replacement runner must execute through the public
command API.

## Current cross-driver qualification (2026-10-16)

The same-build SDK/public-command runner completes all 17 selected cases on
both GTX 1080 backends, with 17/17 CUDA/D3D12 comparisons passing and no
unsupported cases. All ten former failures, including cloth-tear topology,
are resolved. Canonical inputs, numerical tolerances and historical goldens
are unchanged; execution remains separate from physical qualification.

At that historical F5 revision, the independent closed-mesh brick-separation
check passes on both drivers: eight bricks, four substeps and 300 frames produce
0.205278 mm maximum sampled penetration against the unchanged 2 mm gate.
Additional regressions check
torque-free rotation, floor clearance, closed/open concave seams, two-sided
surfaces, real open-edge impacts across a triangulation diagonal, and centered
fast convex impacts (including an initially touching complete crossing).
These results qualify the selected cases, not all domains or performance
targets. The [revisioned Windows report](../conformance/v1/evidence/windows-gtx1080-physics-2026-10-16.json)
retains source, binary and case hashes, the original first divergences,
comparison results, independent physical metrics and reproduction commands.
Raw captures remain in `build-cuda-slang-vs2022/physics-qualified-{cuda,d3d12}/`;
they are not replacement goldens.

The correction addresses three interacting causes:

- Nearly equal world-space rotations and anchors lost small differences to
  cancellation, creating spin/friction and target-dependent contact branches.
  Shared relative-rotation arithmetic and identity-relative solver corrections
  preserve those differences, including exactly zero free-spin correction.
- Convex face manifolds, signed exterior separation and concave/open feature
  classification prevent missing face support and false lateral seam normals.
  Two-sided sheets and genuine open boundary impacts remain active; negative
  controls caught a candidate fix that incorrectly discarded diagonal-floor
  edge contacts. The independent mesh oracle is unchanged.
- CUDA's explicit multiply/add, reciprocal and square-root compilation contract
  aligns with the graphics target. Slang precise mode remains enabled; the
  Coulomb projection and curvature use one common radial scale. No global
  iteration-budget increase or backend-specific physics implementation is used.

The exact numerical and contact contracts are in [CONTACT_SOLVER.md](CONTACT_SOLVER.md).
The tested physics revision is `2a6819f14c0cd48dced13d5c1c80d6695326d6bf`;
later roadmap/lease-only merges do not change its shader or runtime trees.

The broader run passes 34 distinct checks: all 13 D3D12 300-frame scene/reset
gates, the 2,400-frame hinge/slider fixture, 360 repeated API steps, resource
lifecycle with/without validation on both drivers, CUDA domain/coupling
lifecycle, numerical fixtures, reflection, architecture and conformance/layer
contracts. The long fixture records zero sampled gear/slider penetration and
0.0181449 mm maximum slider sideways drift. Two additional D3D12 lifecycle
tests explicitly skip unsupported rope/soft-body/smoke requirements; those
skips are not passes. An initial lifecycle startup timeout was traced to the
new SDK test directory missing the loader DLL search path; CTest now supplies
the selected loader directory, with unchanged assertions and 120-second limit.

The installed `2a6819f` gallery then passes all 13 supported scenes with at
least five seconds for each load/reset, plus F/Tab/point-release and fluid/cloth
display modes (43 observed dwell periods). The installed executable, loader
and both driver DLLs match the tested build byte-for-byte. Captures are in
`build-cuda-slang-vs2022/physics-qualified-window/`; hashes and results are in
the committed report. This window result is execution/input evidence, separate
from the comparison and physical-quality gates above.

## Target status

| Target | Status |
|---|---|
| CUDA | Accepted shared Slang runtime; command API, architecture checks, 105 AVBD fixtures, automated gallery gates, and interactive gallery review pass |
| Metal | Unavailable; no qualified Slang device adapter |
| DirectX 12 | Shared rigid/constraint/liquid/cloth preview and native GPU drawing; 17 targeted CUDA/D3D12 comparisons and independent brick/contact gates pass; remaining domains and performance unqualified |
| Vulkan | Shared rigid/constraint/liquid/cloth/rope/soft-body preview with qualified rigid and liquid couplings; focused GPU and shader validation, not full backend qualification. See [Vulkan](VULKAN.md) |
| WebGPU | AVBD numerical core compiles to WGSL; full schedule and driver unavailable |
