# F8-08 rigid shared-BVH consumer

F8-08 provides the rigid candidate path for the new real-time profile. It
replaces unconditional `body_count * body_count` flag production with bounded
queries against the F8-07 tick snapshot. The legacy reference schedule and
`pm_broad_phase_rigid_pairs` remain unchanged; F8-20 owns final profile
integration and advertisement.

## Candidate pipeline

The new path has three rigid-specific Slang kernels around the shared service:

1. `pm_emit_rigid_spatial_proxies` emits one generation-bearing leaf and query
   per body. Its swept proxy covers start-to-predicted translation, the full
   rotation-origin reach of the cached mesh sphere, collision margin, declared
   path deviation, the complete solve-correction budget, and epsilon.
2. `pm_spatial_query_aabbs` traverses the one published F8-07 hierarchy into a
   fixed `body_capacity * candidates_per_body` allocation. Its existing
   all-or-none overflow status prevents a partial candidate list from reaching
   narrow phase.
3. `pm_filter_rigid_spatial_candidates` validates current full identities,
   keeps each dynamic/dynamic pair once in identity order, and removes self,
   shared fixed-group, and collision-disabled constraint pairs. Stable flag
   compaction followed by `pm_gather_rigid_spatial_pairs` produces the same
   `body * body_count + collider` codes consumed by the cached mesh-local
   narrow phase.

Non-rigid shared leaves are ignored by the rigid filter. Dense indices and
sort keys are never treated as lifetime identities. Owner, mesh, and topology
generations are carried by the shared metadata and revalidated immediately
before candidate consumption.

## Fixed storage and work budget

`make_rigid_spatial_plan` checks multiplication and lays out aligned scratch
and output ranges. Capacity is explicit; no hot-path allocation, growth, or
retry is permitted. The 386-body `rigid-wall` profile count is unchanged. Mesh
resource count and the immutable per-mesh BVHs are also unchanged: the new path
only changes the dynamic world-level candidate source.

The host contract test emits these exact worst-case consumer budgets:

| Case | Bodies | Candidates/body | Candidate slots | Consumer dispatches | Worst-case workgroups | Query candidate bytes | Pair scratch bytes | Pair output bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sparse operating example | 386 | 64 | 24,704 | 7 | 592 | 98,816 | 223,140 | 98,820 |
| Dense oracle reserve | 386 | 385 | 148,610 | 7 | 3,498 | 594,440 | 1,342,196 | 594,452 |

The removed legacy flag allocation is 148,996 bytes at 386 bodies. Sparse work
therefore follows `N * K`; the dense reserve exposes the inherently dense
candidate/output cost rather than hiding it. Both cases are fixed before
submission. A query count above `K`, arithmetic overflow, or an output count
above capacity latches an error and produces no accepted partial tick. F8-21
owns final per-workload profile capacities; the values here demonstrate and
test the bounded policy rather than freezing those later profile choices.

The seven consumer dispatches are proxy emission, shared query, pair filter,
stable compaction count/scan/scatter, and pair gather. Workgroup totals use the
full candidate allocation as a conservative upper bound; GPU-resident active
counts may reduce indirect work when F8-20 records the final schedule.

## One snapshot through the solve

`RigidSpatialSolveEpoch` pins the exact F8-07 epoch token, its single selected
build/refit, and its publication count. Every one of the profile's eight
contact/joint sweeps must present the same counters and token. A second build or
refit, publication change, overflow, stale token, extra sweep, or early finish
is rejected. The guard does not advance time and cannot request a recovery
substep or second spatial update.

## Correctness and portability

Independent host tests compare shared-tree query results with brute force for
fast translation, a fully rotating offset/thin bound, margins, disabled
constraint pairs, fixed groups, and generation replacement. The physical CUDA
test executes generated production Slang for proxy emission, filtering,
compaction handoff, overflow, and stale generation rejection. Existing contact,
friction, AVBD, compute-lifetime, and shared spatial-service tests remain green,
so cached narrow-phase behavior and resource lifetime stay in their established
owners.

The kernels and native dispatch bindings are in the canonical CPU, CUDA,
Vulkan, D3D12, and Metal manifests. Linux qualification includes physical CUDA,
generated CPU/CUDA and Vulkan driver builds, SPIR-V validation, Metal source
compilation, and HLSL SM 6.6 compilation. Local DXIL execution is unavailable
and remains a completion-PR CI/platform gate. Exact commands and limitations
are recorded in `evidence/F8-08-rigid-shared-bvh.json`.
