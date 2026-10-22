# F8-06 low-dispatch spatial build primitives

F8-06 implements the allocation-free primitive layer selected by F8-05. It
does not publish epochs or schedule one build per tick; that lifecycle remains
F8-07's responsibility.

## Contract and layout

`make_spatial_build_plan` computes checked, 16-byte-aligned scratch and output
ranges from a reserved leaf capacity. The caller allocates those ranges when
capacity changes and reuses them. The hot path performs no allocation and its
canonical active count remains GPU-resident. Input flags, bounds, and identities
remain caller-owned.

The pipeline is:

1. Stable active-index compaction: block count, one 256-lane exclusive scan,
   and scatter.
2. Two-dispatch conservative AABB reduction and Morton-key emission.
3. One-workgroup bitonic sort through 1,024 live leaves, otherwise eight
   stable 8-bit radix passes (24 dispatches).
4. One linear-BVH topology dispatch and one atomic bottom-up bounds dispatch.

Each 64-bit ordering key is a 30-bit Morton code followed by the complete
32-bit epoch-local identity. F8-07 must assign collision-free epoch-local IDs
from its full generation-bearing identity map; an exact duplicate is reported,
not silently accepted. Duplicate Morton codes and zero scene extents are valid.
Invalid bounds, duplicate complete keys, active-count overflow, and a wrong
small-sort policy set distinct status bits. Counts above capacity are clamped
before buffer access; unsafe signed topology ranges are rejected by the plan.

The portable kernels assume 256-thread workgroups, shared memory, 32-bit atomics,
and a device-scope publication barrier. They do not depend on subgroup width.
The topology prefix comparison deliberately uses integer bit tests rather than
backend `firstbithigh` lowering, which produced incorrect CUDA results on the
GTX 1080 at the 30,065-leaf tier.

## Dispatch policy

An already-known empty scene has zero build dispatches. A non-empty small build
uses 9 dispatches; a large build uses 32. The previous sixteen-round radix sort
alone used 48 dispatches, so the large complete pipeline now costs 32 instead
of 56 with that sort substituted. The 1,024 crossover is the maximum fixed
shared-memory sorting-network capacity and is supported by physical timings:
the one-dispatch path remained about 0.03-0.04 ms at 128, 386, and 1,024 live
leaves. Larger inputs use the bounded 24-dispatch radix path.

Compaction necessarily visits reserved flag capacity. Bounds, keying, sorting,
topology, and union work are sized from the live count supplied to the schedule.
F8-07 may use its known registration count or backend indirect scheduling; no
primitive requires a routine host readback of the GPU count.

## Correctness and physical evidence

The independent CPU oracle checks randomized BVH overlap results against brute
force, inclusive touching, sparse compaction, duplicate Morton codes, a
zero-extent scene, empty input, checked arithmetic, range non-overlap, and the
30,065-leaf desktop fixture. The CUDA test executes the generated production
Slang kernels and verifies every sorted key/value, source identity, topology
edge, parent edge, internal union, and root bound. It also exercises NaN bounds,
duplicate complete keys, and count overflow. F8-07 owns collision-free assignment
of the epoch-local identity itself; this primitive detects equal composite keys.

On the physical NVIDIA GeForce GTX 1080 (compute 6.1, driver 582.66, CUDA 12.9),
40 measured samples followed 5 warmups; full builds used 24 samples after 3
warmups. Seed copies and allocations were outside sort event timing.

| Capacity / live | Policy | Sort dispatches | Full dispatches | Workgroups | Scratch bytes | Output bytes | Sort p50 / p95 ms | Legacy sort p50 / p95 ms | Full build p50 / p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65,536 / 128 | bitonic | 1 | 9 | 519 | 2,631,732 | 5,504,992 | 0.032768 / 0.047104 | 0.380928 / 0.421888 | 0.140288 / 0.162816 |
| 386 / 386 | bitonic | 1 | 9 | 15 | 16,628 | 32,416 | 0.043008 / 0.050176 | 0.398336 / 0.493568 | 0.158720 / 0.164864 |
| 1,024 / 1,024 | bitonic | 1 | 9 | 27 | 41,172 | 85,984 | 0.035840 / 0.047104 | 0.389120 / 0.457728 | 0.168800 / 0.207872 |
| 4,500 / 4,500 | radix-8 | 24 | 32 | 406 | 181,652 | 377,968 | 0.250880 / 0.272384 | 0.943104 / 0.974848 | 0.378880 / 0.407552 |
| 30,065 / 30,065 | radix-8 | 24 | 32 | 2,606 | 1,208,580 | 2,525,440 | 1.019904 / 1.185792 | 4.568064 / 4.930560 | 1.050624 / 1.224704 |

The sparse row demonstrates the intentional boundary: compaction work follows
capacity, while all later phases follow 128 live leaves. The largest fixture
contains repeated positions/Morton codes with distinct identities.

DXIL schedules and reflection, CUDA source/runtime, Vulkan SPIR-V mirrors, and
per-entry Metal source/reflection all compile from the canonical manifest. This
is compile coverage for non-CUDA backends, not physical runtime qualification;
F8-22 through F8-30 retain those device obligations.
