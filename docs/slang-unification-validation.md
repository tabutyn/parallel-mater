# Slang unification validation

This page records the current validation contract for the unified runtime.

## Runtime architecture

- `parallel-mater` exposes the C command API in `parallel_mater.h`.
- The production runtime dispatches physics generated from `src/slang`.
- `runtime.cu` contains no GPU kernels or direct physics launches.
- Device and simulation dispatch adapters contain platform services and
  generated-kernel launches only.
- DirectCompute and Metal backend-specific implementations and galleries have
  been removed; Vulkan and WebGPU physics are not advertised.
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
- The loader-visible D3D12 adapter creates both shared AVBD DXIL pipelines,
  binds and dispatches their schedule, validates readback, and passes public
  memory/buffer/barrier/fence/native-handle tests. It advertises no physics
  feature until the remaining domain schedules are command-wired.
- The common gallery builds with OptiX disabled, uses the software renderer and
  thin D3D12 presentation window, and passes the real-window F/Tab/scene-switch
  harness while physics is supplied by the selected full-feature driver.
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

## Target status

| Target | Status |
|---|---|
| CUDA | Accepted shared Slang runtime; command API, architecture checks, 105 AVBD fixtures, automated gallery gates, and interactive gallery review pass |
| Metal | Unavailable; no qualified Slang device adapter |
| DirectX 12 | Core loader driver and AVBD DXIL schedule qualified; physics features intentionally unavailable |
| Vulkan | Interoperability declarations only; physics unavailable |
| WebGPU | AVBD numerical core compiles to WGSL; full schedule and driver unavailable |
