# Metal gallery parity audit

Audit baseline: `main` commit `e005be8`. The CUDA gallery is the behavioral
and visual reference. Physics conformance is tracked separately under
`conformance/v1`; this document covers the gallery application and renderer.

## Current comparison

| Area | CUDA gallery | Metal gallery | Status |
| --- | --- | --- | --- |
| Scene registry and GLB assets | Shared 29-entry registry and authored GLBs | Same registry, parser, and GLBs | Matching |
| Physics scene construction | CUDA `World` instantiator | Metal `World` instantiator from the same definitions | Matching by subsystem gates |
| Camera | Left-drag orbit, Shift-left-drag pan, wheel zoom, per-scene presets | Uses the same `CameraController` and callbacks | Matching |
| Sky | View-dependent blue gradient from the primary ray | Same camera-ray sky equation in MSL | Matching |
| Opaque materials | Authored base color, exact checker colors, one-sided key light, ambient and rim light | Same equations and constants in MSL | Matching |
| Contact paint | Per-fragment bicubic UV lookup with separate front/back bits | Live Metal field sampled bicubically at vertices; front/back bits currently combined | Partial |
| Rigid geometry | OptiX instances over authored render meshes | Rasterized authored render meshes transformed from live states | Close |
| Cloth and soft-body surfaces | Refit deforming OptiX geometry with render bindings | Live surface buffers rebuilt as raster triangles | Partial; smooth binding normals differ |
| Rope | Rebuilt tube rendered through OptiX | Same shared tube builder rendered through Metal | Close |
| Fluid | Implicit scalar surface with reflection, refraction, absorption, Fresnel, depth, and optional particle/wireframe modes | Depth-correct particle impostors only | Major gap |
| Containing skins | Separate transparent visibility layer with thickness and absorption | Opaque cloth surface | Major gap |
| Foam | Screen-space foam patches driven by agitation and depth | Foam only changes particle color | Major gap |
| Smoke | Depth-aware screen-space smoke plus six grid debug views | Transparent particle impostors plus live `Z/X/C/V/B/N` grid/particle field views | Partial; compositing still differs |
| Scene controls | Policy-specific gravity, fixed contact collector, tank motors, dump rotation, constraint action, count dialog | Same policies, collector welding, action, and bounded `P` dialog | Matching for interactive behavior |
| Debug and timings | `F/V/Z/X/C/B/N/M`, force vectors, contacts, topology, grids, timing panels | `F` uses Metal counter timing; `V` switches particles/structure/contacts; smoke maps are live; vector/topology overlays and `M` capture remain | Partial |
| Camera/physics interpolation | Fixed-step accumulator with rigid interpolation | One asynchronous fixed step in flight; render/input stay responsive on the last completed state | Partial |
| Capture | Headless PPM plus interactive debug capture | Headless PPM; no interactive `M` capture | Partial |
| Renderer | OptiX ray tracing with dynamic acceleration structures | Metal raster pipelines | Intentional temporary divergence |

## Work completed from this audit

- Replaced the fixed Metal camera with the shared CUDA camera controller.
- Added the CUDA sky ray equation and copied its opaque shading, checkerboard,
  gamma, light, and rim constants into MSL.
- Added live paint-field visualization for rigid and cloth meshes.
- Ported CUDA control policies for tilted gravity, cloth/peg steering, tank
  motors, dump rotation, and constraint enable/disable actions.
- Updated rigid and collector steering to use the shared camera-relative 30°
  gravity model, including authored bodies that retain vertical gravity.
- Ported the combined hinge/slider scene, revised piston, 384-brick wall, and
  the shared 29-entry selector; `Tab` opens the selector exactly as on CUDA.
- Added Metal hit-box polling for rigid collision meshes and stable fluid
  particle IDs, with the same deterministic ordering and validation contract.
- Made rigid coloring and solve passes traverse the compacted active-pair list;
  the 120-frame wall gate improved from 113.6 seconds to 13.4 seconds on the
  development Mac.
- Ported the fixed-scene contact collector: loose bodies retain downward
  gravity, contacts with the large body create fixed constraints, and the
  collector cluster follows the arrow-controlled gravity.
- Added the bounded `P` count dialog and rebuild flow shared by dump, fluid,
  paint, cloth-water, soft-fluid, rope-fluid, and smoke-water scenes.
- Restored `F/V/Z/X/C/B/N` input mapping, Metal GPU-counter feedback,
  particle/structure/contact modes, and smoke grid field visualization.
- Retained asynchronous Metal stepping and three buffered render frames so
  camera and gallery UI remain responsive while a physics frame is running.

## Ordered parity work

1. Port the CUDA implicit-fluid field and ray-marched water material to Metal.
2. Add the transparent containing-skin pass and depth/thickness handling.
3. Port foam and smoke compositing using the same depth conventions.
4. Share backend-neutral overlay drawing, then complete normal, force, bond,
   velocity, and interactive `M` capture views.
5. Add rigid interpolation and render-binding normals for deforming surfaces.
6. Capture approved CUDA images on NVIDIA hardware and compare fixed Metal
   frames with SSIM and mean-channel-error gates.

The current Metal captures are smoke tests, not visual-parity references. The
required NVIDIA capture matrix and provenance are defined in
[CUDA_REFERENCE_REQUIREMENTS.md](CUDA_REFERENCE_REQUIREMENTS.md).
