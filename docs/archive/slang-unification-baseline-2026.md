# Archived Slang unification baseline (2026)

> Historical pre-migration measurement record. This is not a current build,
> validation, performance, or release checklist.

Captured before command-runtime physics dispatch changed. Goldens were not
updated and scene counts or solver quality were not reduced.

## Provenance

- Date: 2026-10-11 (America/St_Johns)
- Source: `b3cb3d0fac068890771c5effe29dc30f916c298d`
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB
- GPU UUID: `GPU-2d010adb-af2c-99d1-bf34-a89a8551e301`
- PCI bus: `00000000:01:00.0`
- Driver: 590.44.01
- CUDA compiler: 13.1.80; architecture 75
- Slang: 2026.18
- Host C++ compiler: GCC 12.2.0
- Build: Release, CUDA and OptiX gallery enabled, benchmarks enabled, all
  Slang mirrors enabled

## Conformance and gallery baseline

The complete build succeeded. CTest reported 112/124 tests passing (90%). The
following failures were present before runtime-dispatch work:

- conformance input and CUDA-golden asset hashes no longer match the manifest;
- Generic/GenericSpring checkerboard metadata and authored arrow-force value
  differ from their tests;
- rigid-wall static support, fixed-collector, dump-truck, soft-body-cloth, and
  rope-soft-body-release regressions;
- Hinge/Slider anchor behavior and Piston tooth indexing regressions;
- rope-winding GPU budget regression.

All built headless gallery scene tests completed, including the mixed-domain
scene set. The failing goldens remain evidence to resolve; they are not accepted
as replacement output.

## Rigid scene timing baseline

Command:

```sh
parallel-mater-rigid-scene-benchmark examples/assets/RigidBody.glb \
  --settle 120 --frames 120
```

Three independent runs on the otherwise idle GPU:

| Run | GPU median | GPU p95 | Wall median | Wall p95 | Solve median | Solve p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.3188 ms | 19.4836 ms | 19.3719 ms | 19.5349 ms | 16.4024 ms | 16.5233 ms |
| 2 | 19.2632 ms | 19.4058 ms | 19.3283 ms | 19.4624 ms | 16.3994 ms | 16.5181 ms |
| 3 | 19.1601 ms | 19.3167 ms | 19.2228 ms | 19.3779 ms | 16.2796 ms | 16.3983 ms |

Each run used 24 solve launches, three islands, ten maximum contact passes,
four colors, 1043 candidate/live pairs, and no pair overflow. All runs produced
the same quality result: maximum drop 0.0383096, maximum displacement 0.359672,
and final maximum speed 0.168359.

The migration performance gate compares the same asset, counts, resolution,
settings, warm-up, and measurement intervals. A median or p95 regression over
10% across three runs must be investigated and resolved rather than hidden by
lower counts or solver settings.
