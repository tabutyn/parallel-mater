# Slang unification validation

> Historical CUDA migration snapshot. Metal was subsequently qualified as a
> Slang-backed compatibility adapter; see `METAL.md` for current status.

Status updated 2026-10-12 on source baseline
`b3cb3d0fac068890771c5effe29dc30f916c298d` plus the uncommitted command-runtime
migration.

## Runtime architecture

- `parallel-mater` exposes the C command API in `parallel_mater.h`.
- The production runtime dispatches physics generated from `src/slang`.
- `world.cu` contains no GPU kernels or direct physics launches.
- Device and simulation dispatch adapters contain platform services and
  generated-kernel launches only.
- The DirectCompute implementation and its platform gallery have been removed.
  The retired Metal physics build option still fails at configure time; Vulkan
  and WebGPU physics are not advertised.
- The retired hand-written CUDA physics kernels have been deleted. CUDA source
  produced by Slang, the thin CUDA device adapter, CUDA interoperability, and
  OptiX drawing remain.

`parallel-mater-slang-architecture` passes and reports zero native physics
kernels or direct launches in the runtime.

## Completed validation

- Release `parallel-mater` library build: pass.
- Shared Slang CUDA source generation: pass.
- Shared Slang direct PTX generation: pass.
- AVBD cross-target bundle and reflection validation: pass.
- `parallel-mater-gallery` link through the command API: pass.
- API declaration/runtime coverage generation: pass.
- CPU contract, catalog, input, conformance registry/comparator, contact-core,
  AVBD-core, architecture, and reflection tests: pass.
- Gallery catalog contains 28 canonical entries; shared input tests verify Tab
  opens the complete catalog and dismisses parameter editing.
- Command API, API contract, gallery catalog/input, Slang architecture, and
  Slang reflection tests pass on the CUDA host.
- All 38 AVBD fixtures and the CPU/CUDA/Slang AVBD-core tests pass.
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
source-deletion gates. The conformance runner remains available for diagnostic
captures, while the obsolete CUDA-golden comparison is no longer registered as
a required CTest.

## Target status

| Target | Status |
|---|---|
| CUDA | Accepted shared Slang runtime; command API, architecture checks, 38 AVBD fixtures, 42 automated gallery gates, and interactive gallery review pass |
| Metal | Unavailable; no qualified Slang device adapter |
| DirectX 12 | Unavailable; no qualified Slang device adapter |
| Vulkan | Interoperability declarations only; physics unavailable |
| WebGPU | Unavailable |
