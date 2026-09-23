# Rigid contact performance, 2026-09-21

These are local engineering measurements, not general CUDA or hardware
claims. The objective was to remove the observed 20+ ms rigid-contact frame
without weakening determinism, containment, or authored geometry contracts.

## Environment and method

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, compute capability 8.6
- CUDA toolkit: 13.1
- build: `Release`, `CMAKE_CUDA_ARCHITECTURES=86`
- scene: committed `examples/assets/PassiveActive.glb`, five rigid bodies
- frame: `1/60 s`, four substeps
- sampling: settle 360 frames, warm 10, measure 120
- timing: CUDA events requested through `WorldStepTimings`; no external profiler

Build and reproduce the retained scene benchmark with:

```bash
cmake -S . -B build-gallery \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPARALLEL_MATER_BUILD_OPTIX_GALLERY=ON \
  -DPARALLEL_MATER_BUILD_BENCHMARKS=ON \
  -DBUILD_TESTING=ON
cmake --build build-gallery -j
./build-gallery/parallel-mater-rigid-scene-benchmark \
  examples/assets/PassiveActive.glb
```

## Result

The original serial BVH/contact path measured **24.100 ms median GPU time**
and **24.132 ms median wall time**. The retained pipeline measures **1.797 ms
median GPU time** and **1.806 ms median wall time**: about **13.4× faster** for
this scene. All bodies remained in the bowl and repeated settled poses were
bit-identical on the tested GPU.

Final 120-frame distribution:

| Stage | Median | p5 | p95 |
|---|---:|---:|---:|
| Integration | 0.017 ms | 0.014 ms | 0.030 ms |
| World bounds | 0.015 ms | 0.012 ms | 0.020 ms |
| Pair filter | 0.013 ms | 0.011 ms | 0.019 ms |
| Deterministic pair compaction | 0.022 ms | 0.019 ms | 0.032 ms |
| Leaf-pair generation | 0.636 ms | 0.634 ms | 0.641 ms |
| Triangle contact evaluation | 0.905 ms | 0.532 ms | 0.960 ms |
| Contact solve | 0.171 ms | 0.120 ms | 0.196 ms |
| Input clear | 0.003 ms | 0.002 ms | 0.004 ms |
| **GPU total** | **1.797 ms** | **1.365 ms** | **1.873 ms** |
| **Frame wall** | **1.806 ms** | **1.375 ms** | **1.886 ms** |

The live `F` overlay reports these same stages. Rendering is deliberately not
included in `WorldStepTimings`.

## Experiments kept or removed

| Hypothesis | Isolated measurement | Decision |
|---|---:|---|
| Parallel leaf/triangle candidates with fixed-order reduction | Default scene 24.100 → 2.997 ms before later changes; cooperative leaf enumeration subsequently reduced the final pipeline further | Keep |
| Refit a world-space BVH every substep | 2.9973 → 2.9980 ms and added state/kernel work | Remove completely |
| GPU broad phase plus stable active-pair compaction | 64 separated dynamic bodies 15.351 → 0.241 ms; primary scene was neutral | Keep |
| Blender collision proxies | Dynamic-body proxies 3.006 → 2.312 ms while detailed render meshes remained unchanged | Keep |
| Decimate the static bowl collision surface | 3.006 → 4.946 ms and altered support behavior | Remove that proxy; keep the detailed bowl |
| Velocity-gated swept triangle pairs | 120 m/s fixture required 32 discrete substeps at 1.587 ms; conservative sweep contained it in one substep at 0.810 ms | Keep |

The sparse-world broad-phase fixture used 64 identical triangle boxes on a
10-metre grid with zero gravity, four substeps, 10 warmups, and 120 samples.
Its original cost came from the serial solver scanning the complete 64×64
manifold cache eight times. Pair filtering alone was not sufficient; the
solver had to consume the stable compacted pair list.

The swept regression is checked into `tests/rigid_tests.cpp`. A 0.2-metre box
moves 2 metres through a two-sided open plane in one `1/60 s` step. The
conservative triangle-pair path must stop it above the plane and reproduce the
same state across 20 resets.

## Current bottleneck and memory tradeoff

Leaf-pair generation and triangle evaluation now dominate. Candidate order is
deterministic: each body pair uses a cooperative fixed-order leaf scan and a
fixed thread-order manifold reduction. The 512-candidate fallback buffer costs
32 MiB at the default 64-body capacity; overflow retains correctness through
the exact serial path but can be slower. Future work should reduce that storage
or improve overflow handling only with a new measured fixture—neither is a
fluid-milestone prerequisite.
