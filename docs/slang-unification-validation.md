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

The same-build public-command runner completed all 17 selected cases on both
GTX 1080 backends. Using unchanged canonical tolerances, CUDA/D3D12 comparison
passes 7 cases: cloth-core, cloth-water, compound-weld-lifecycle,
constraint-breaking, constraint-piston, constraint-slider and fluid-lifecycle.
The other 10 fail rigid state/contact or cloth-tear topology comparisons.
The Windows CUDA managed-memory ordering fault in cloth-tear is fixed; the
case completes but does not pass cross-driver topology comparison.

An independent closed-mesh brick-separation check also fails on both drivers:
with eight bricks and four substeps, the 2 mm gate observes 7.21844 mm on D3D12
at frame 102 and 7.02056 mm on CUDA at frame 101 (zero-based). Earlier finite
state and interactive acceptance do not establish nonpenetration. Neither this
gate nor canonical tolerances/goldens have been relaxed. Full backend parity
and the shared brick solver's physical quality remain unqualified; see the
[Windows roadmap evidence](ROADMAP.md#windows-shared-runtime-cleanup-2026-10-16).

## Target status

| Target | Status |
|---|---|
| CUDA | Accepted shared Slang runtime; command API, architecture checks, 105 AVBD fixtures, automated gallery gates, and interactive gallery review pass |
| Metal | Unavailable; no qualified Slang device adapter |
| DirectX 12 | Shared rigid/constraint/liquid/cloth preview and native GPU drawing; 13-scene window gate passes, strict physical parity remains unqualified |
| Vulkan | Shared rigid/constraint/liquid preview; focused GPU and shader validation, not full backend qualification. See [Vulkan](VULKAN.md) |
| WebGPU | AVBD numerical core compiles to WGSL; full schedule and driver unavailable |
