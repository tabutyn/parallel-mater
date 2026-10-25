# F8-39 common realtime storage and atomic staging

F8-39 extends the native fixed-tick boundary to every supported domain family
without adding another scheduler. Public simulation creation now reserves one
shared spatial arena for the declared rigid, liquid, cloth, soft-body, rope and
smoke capacities. Domain allocations remain in their existing fixed-capacity
arenas; `ProductionRealtimeResources` is the allocation-free ownership registry
that pins those allocations and generations for one tick.

## Admission and ownership

At `begin_tick`, the production backend derives a fixed-size registration for
each live resource. Registrations include live slot count, reserved element
count, handle-generation identity and the operator phases currently bound for
that family. Rigid bodies identify their public handle slots. Other domains
identify their owner slot and generation. Particle and grid smoke have distinct
registrations but share the public smoke-slot limit.

The registry admits the complete set before snapshots or device work begin.
Slot, element, generation and shared-smoke overflow fail without creating a
pending generation. A successfully staged generation remains private until all
of these conditions hold:

1. the family has predict, spatial and domain bindings;
2. the shared spatial service publishes exactly once;
3. the schedule and telemetry preflight succeed; and
4. spatial, input/render snapshot and resource commits all complete.

Any failure calls the same abort boundary for spatial state, snapshot ownership
and resource registration. Completed public state, clock, pending inputs and
arena capacities remain unchanged. Reset and replacement retire owner handles;
the next registration carries the new generation, while an explicitly
invalidated pending generation cannot be reused.

Only rigid operators are marked executable in F8-39. A live liquid,
deformable, rope or smoke registration is therefore admitted and then returns
`PM_ERROR_FEATURE_NOT_PRESENT` before spatial/device work. F8-40 through F8-44
replace that explicit rejection family by family.

## Phase bindings

The compile-time `realtime_phase_bindings` table is the ownership contract.
No phase permits a host readback or global wait.

| Phase | Reads | Writes | Publication | Follow-up owner |
|---|---|---|---|---|
| boundary | completed public resources | private staged state | none | F8-48 lifecycle inputs |
| predict | public and staged resources | private predicted state | none | F8-40--44 domains |
| spatial | staged state and the common spatial arena | one staged index epoch | none | F8-40--44 emit/query adapters |
| domain | staged state and the published spatial epoch | private domain state | none | F8-40--44 solvers |
| coupling | staged state and the same spatial epoch | private coupling deltas | none | F8-45--47 interactions |
| lifecycle | staged state and the same spatial epoch | private add/remove/topology state | none | F8-48 lifecycle |
| output | staged state and the same spatial epoch | private output/capture state | none | F8-49 paint/debug/output |
| commit | private staged state | no further physics state | public state, clock and snapshots | F8-49 output publication |

All domain adapters consume the same staged generation and spatial token. They
may encode multiple ordered kernels inside their phase visit, but cannot build,
refit or publish a second index, allocate per tick, read a live count to the
host, wait the global device, or expose a partial domain commit.

## Capacity model

The public creation plan aggregates declared spatial capacity with checked
32-bit admission:

- rigid bodies and liquid/particle-smoke particles contribute leaves and
  queries;
- cloth triangles contribute leaves and cloth vertices contribute queries;
- rope nodes and segments contribute leaves, with nodes as queries;
- soft nodes and surface triangles contribute leaves, while nodes and surface
  vertices contribute queries;
- grid-smoke cells remain in the existing grid arena and use the common index
  only for later obstacle/collision bindings.

Rigid-only plans preserve their existing candidates-per-query value. Plans
containing liquid or particle smoke reserve 128 candidates per query; plans
containing deformables reserve at least 64. Checked aggregation and the spatial
planner reject overflow before a simulation object is returned.

## Verification boundary

`parallel-mater-realtime-resources-tests` covers empty, sparse and mixed
registrations, every family, both smoke representations, slot/element/shared
smoke overflow, unsupported binding rejection, abort/retry, capacity reuse and
generation invalidation. The native runtime test additionally binds a mixed
public simulation, executes an empty common tick, proves a live unbound liquid
fails before commit, resets it, and reuses the same fixed arenas successfully.

The Android qualification target builds that native test against the production
Vulkan driver and runs it with failure injection disabled. This is resource and
lifecycle evidence only: F8-39 does not claim liquid, deformable, rope or smoke
physics execution, quality, performance, or final F8 qualification.
