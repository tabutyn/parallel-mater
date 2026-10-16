# Android Vulkan qualification — 2026-10-16

Device: Pixel 7, Android 17, Mali-G710, Vulkan 1.4.343. The gallery uses the
loadable Parallel Mater Vulkan driver and common Slang physics; rendering uses
the existing software renderer. Builds and workstation tests run serially.

## Correctness and startup

The three native suites pass: public command/API behavior, asynchronous
submission, and analytic liquid free fall plus bounded Fluid/FluidRigid scenes
and reset. Desktop regression checks pass 13/13 with pinned SDK 1.4.341.0
validation; all six SPIR-V modules validate. Android native tests enable the
Parallel Mater API validation layer, not the Khronos Vulkan layer.

Slang's redundant forward pointer declarations initially crashed Mali's compiler.
The build now normalizes acyclic type dependencies before embedding the modules,
without changing IDs, layouts or executable instructions. The standalone
pipeline probe and regression tests preserve that reproducer.

The first gallery scene replacement was killed by Android for low memory. Its
old and new GPU simulations were resident together. Android now releases the
old scene before allocating its replacement; failure leaves Reset/scene selection
available without dereferencing a missing scene.

## Performance limitations

The Pixel 7 startup resource declares four bricks, 128 fluid particles, four
moving liquid-coupled bodies, a maximum 640×360 render surface and a 30 FPS target.
The phone's landscape aspect uses 640×288. These are provisional workload limits,
not a verified performance profile. Authored constraint graphs are retained;
Collector, for example, still contains 50 moving bodies.

Early brick-scene samples ran around 6 FPS and consumed approximately 3.2 GiB
PSS. Public physics buffers account for about 148 MiB; Android reports most of
the process footprint as graphics memory. Mali's executable statistics report
17,664 bytes of thread-local storage for `pm_finalize_rigid_contact_manifolds`
(including 6,536 bytes of spills). This is evidence of costly shader scratch,
not a complete attribution of every driver allocation.

Smaller submission batches, command-pool resource release and generic SPIR-V
optimization did not materially reduce the observed footprint. Those experiments
are not enabled in the shipping build. Per-thread contact kernels now avoid
launching groups beyond the maximum possible pair count, but that alone does not
resolve Mali's private-memory cost. No physics iterations or formulas were
removed to manufacture a faster result.

The remaining performance work is to profile and reduce private shader storage,
prepare smaller representative constraint scenes through the hardware-profile
mechanism, and improve rendering cost. Sustained device measurements must precede
promotion to a verified profile.

## Eight-scene sweep

Every scene completed two 60-frame measurement windows in one app process,
with scene replacement between tests. This checks startup, rendering and short
runtime continuity; it is not a general physics conformance or long-run result.
The table shows the second window and the subsequent process-memory snapshot.

| Scene | Observed FPS | Mean work (ms) | p95 work (ms) | PSS (GiB) |
| --- | ---: | ---: | ---: | ---: |
| RigidBody.glb | 5.87 | 167.760 | 182.349 | 3.21 |
| ConstraintFixed.glb | 3.27 | 303.728 | 327.861 | 3.12 |
| Celestial.glb | 4.45 | 221.712 | 262.673 | 3.23 |
| ConstraintHinge.glb | 1.96 | 507.882 | 609.808 | 3.23 |
| ConstraintPiston.glb | 15.60 | 62.118 | 76.275 | 3.13 |
| ConstraintGeneric.glb | 4.62 | 214.497 | 246.550 | 3.15 |
| Fluid.glb | 6.32 | 156.317 | 200.115 | 3.19 |
| FluidRigid.glb | 4.10 | 241.076 | 281.636 | 3.28 |

All eight thermal snapshots reported status 0 (no throttling indication).
The phone was connected over USB at full battery; these are not battery-life
measurements. The 30 FPS target was not achieved. Verified profiles stay empty.

## Reproduction and artifacts

Build and native test commands are in [the Android README](../examples/android/README.md).
Debug scene/control extras use the same commands as the gallery controls. Local
logs, screenshots and memory/thermal snapshots are under ignored
`out/android/gallery-vulkan/`; native results are under
`out/android/native-vulkan-tests/`.

The pipeline probe accepts `PM_VULKAN_PROBE_STATISTICS=1` on devices supporting
`VK_KHR_pipeline_executable_properties`. `PM_VULKAN_PROBE_HOLD_SECONDS` (1–60)
keeps a compiled pipeline alive briefly for external memory inspection. These
measure compilation and compiler statistics; the probe never dispatches physics.
