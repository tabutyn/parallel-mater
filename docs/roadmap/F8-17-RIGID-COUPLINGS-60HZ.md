# F8-17 rigid two-way couplings at one 60 Hz step

F8-17 adds the cross-domain exchange phase for execution profile
`realtime-single-step` version 1.0. It covers only API 0.2 modes already
advertised between rigid bodies and cloth, soft bodies, ropes, and particle or
grid smoke. Rigid-liquid remains F8-18. The public profile remains
unadvertised until F8-20 assembles the complete schedule.

## Registered modes

| Endpoint | Modes |
|---|---|
| Cloth | Contact |
| Soft body | Contact |
| Rope | Contact, attachment |
| Particle smoke | Contact, drag |
| Grid smoke | Contact, drag |

The host and device validators reject every other family/mode pair. In
particular, this task does not infer rigid-smoke pressure or viscosity modes
from smoke's self-dynamics, and it does not add containment or rigid-liquid
support.

## Same-epoch exchange

F8-10 supplies generation-tagged constraint records from one published spatial
epoch. `RigidCouplingRealtimeTickContract` pins that token, its build/refit and
publication counters, graph order, topology revision, and endpoint generations
for the complete coupling pass. Registrations use canonical family, mode, then
stable edge identity order. A replacement generation cannot consume an old
record; the stale proposal is zero until the boundary installs a newly bound
record.

Every registered edge receives exactly four inner algebraic sweeps. Each sweep
uses three batched shared-Slang entries:

1. `rigid_coupling_project` computes endpoint and equal-opposite rigid
   proposals from the immutable constraint set.
2. `rigid_coupling_apply_endpoints` reduces proposals in stable constraint
   order and updates each non-rigid endpoint once.
3. `rigid_coupling_apply_bodies` reduces the opposite linear and angular
   exchanges and updates each dynamic rigid body once.

These entries do not predict, integrate, commit, advance time, query the host,
or build/refit a BVH. Domain predictors and final commits remain owned by the
domain tasks and the assembled F8-20 schedule. The coupling pass therefore has
zero time advances and cannot double-integrate either endpoint.

## Equations

For endpoint position `x`, rigid contact point `b`, outward normal `n`, contact
distance `d`, endpoint inverse mass `w_e`, and rigid translational inverse mass
`w_b`, contact uses

`C = max(d - dot(x - b, n), 0)`

`delta_x = n C w_e / (w_e + w_b)`

`delta_b = -n C w_b / (w_e + w_b)`

The current and previous endpoint/contact-point separations share this path, so
a thin crossing is resolved without another spatial query. Normal impulse uses
rigid angular effective mass at the contact arm. Coulomb tangent impulse is
bounded by `friction * normal_impulse`. Rigid linear impulse is exactly the
negative endpoint impulse, and rigid angular impulse is
`cross(contact_arm, -endpoint_impulse)`.

Attachment applies the same mass split to the point-to-anchor error with its
versioned response. Drag applies the response to relative point velocity and
does not move either position. Four fixed sweeps replace adaptive convergence.
Endpoint and rigid correction-consumption buffers enforce F8-00's cumulative
`0.25 L` limit.

## Bounded work

The frozen six-family workload contains 16 registered coupling edges. The
checked budget uses 6,062 non-rigid endpoints, 32 rigid bodies, and 16 active
constraints:

| Item | Budget |
|---|---:|
| Coupling sweeps per registered edge | 4 |
| Edge sweeps | 64 |
| Constraint visits | 64 |
| Batched dispatches | 12 |
| Workgroups at 128 threads | 200 |
| Solver scratch | 25,144 bytes |
| Time advances | 0 |
| Spatial updates/rebuilds during solve | 0 |
| Intra-tick host readbacks | 0 |

Dispatch count is independent of edge count because all registered constraints
share each proposal and reduction phase. Scratch is four `Vec3` proposal
streams plus one correction-consumption scalar per endpoint and rigid body.
There is no per-edge candidate or BVH allocation.

## Physical quality

Generated production Slang ran on an RTX 3050 Ti with all eight supported
family/mode combinations active in one ordered pass. A swept thin contact at
the exact 100:1 dynamic mass-ratio boundary ended with zero measured
penetration, zero closing speed, and zero tunnels. Reciprocal linear and angular
exchange residuals were `1.0e-7` and `8.0e-8`. Rope attachment separation was
0.5 mm. Particle- and grid-smoke drag each reduced unit relative speed to
0.31640625 without energy growth. A stale generation produced no mutation; a
rebound generation became active; invalid profile execution remained atomic.

The eight-edge, eight-constraint equation path ran 120 warmup ticks and 600
timed samples. CUDA-event timing was p50 0.052224 ms, p95 0.052992 ms, p99
0.053248 ms, and maximum 0.065248 ms. This is an operator measurement, not an
F8-21 integrated frame-budget result.

## Accuracy tradeoff and legacy separation

Version 1 accepts operator splitting, four fixed coupling sweeps, point-contact
linearization, batched deterministic reductions, and response-based attachment
and drag. Those losses remain bounded by F8-00's contact, attachment, and drag
gates. It does not accept tunneling, generation aliasing, extra time advances,
unbalanced exchange, silent correction overflow, adaptive retries, or a second
spatial update.

Legacy kernels, schedules, tolerances, and goldens are unchanged. Physical
legacy CUDA execution completed for `cloth-core`, `soft-body-rigid`,
`rope-core`, and `smoke-grid`. `cloth-core` and `smoke-grid` pass their pinned
goldens. `rope-core` retains its documented 242-difference historical reference
failure, and `soft-body-rigid` retains 25 historical-golden differences; neither
result is relabeled as realtime evidence.

Commands, provenance, exact budgets, and results are recorded in
[`evidence/F8-17-rigid-couplings-60hz.json`](evidence/F8-17-rigid-couplings-60hz.json).
