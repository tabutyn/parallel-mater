# F8-19 deformable and smoke couplings at one 60 Hz step

F8-19 adds the remaining non-rigid, non-liquid exchange phase for execution
profile `realtime-single-step` version 1.0. It covers only API 0.2 modes already
advertised between cloth, soft bodies, ropes, and particle or grid smoke. The
public profile remains unadvertised until F8-20 assembles the complete schedule.

## Registered combinations

| Pair | Representation | Modes and direction |
|---|---|---|
| Cloth–soft body | Surface endpoints | Contact, cloth ↔ soft body |
| Cloth–rope | Surface/node endpoints | Attachment, cloth ↔ rope |
| Cloth–smoke | Particle and grid | Contact, cloth → smoke; drag, smoke → cloth |
| Soft body–rope | Surface/node endpoints | Contact and attachment, soft body ↔ rope |
| Soft body–smoke | Particle and grid | Contact, soft body → smoke; drag, smoke → soft body |
| Rope–smoke | Particle and grid | Contact, rope → smoke; drag, smoke → rope |

The six semantic family pairs contain ten advertised mode combinations. The
particle/grid distinction produces nine compiled pair variants and sixteen
registered operators in the frozen workload. Every other mode is rejected. In
particular, this task does not add cloth–rope contact, deformable–smoke
pressure/viscosity, cloth/soft self-contact, distinct-system same-family
contact, or smoke-field exchange.

## One shared epoch and bounded exchange

F8-10 supplies generation-tagged constraint records from one published spatial
epoch. `DeformableSmokeCouplingTickContract` pins that token, its build/refit
and publication counters, graph order, topology revision, and both endpoint
generations for the full coupling pass. Registrations use canonical pair, mode,
then stable-edge identity order. Retired generations produce zero proposals;
only a boundary-installed record for the replacement generation can reactivate.

Every registered edge receives exactly four algebraic sweeps. Each sweep uses
three batched shared-Slang entries:

1. `deformable_smoke_coupling_project` computes position and impulse proposals
   from the immutable shared candidate set.
2. `deformable_smoke_coupling_apply_first` reduces proposals in stable
   constraint order and updates each canonical first endpoint once.
3. `deformable_smoke_coupling_apply_second` does the same for canonical second
   endpoints.

The entries do not predict, integrate, commit, advance time, query the host, or
build/refit a hierarchy. Smoke grid advection and projection remain F8-16 grid
operations. Domain prediction and final commit remain owned by the domain tasks
and the assembled F8-20 schedule. Thus the coupling pass has zero time advances,
zero solve-time spatial updates, and cannot integrate either endpoint twice.

## Equations and direction

For canonical endpoint positions `x_a`, `x_b`, normal `n` from B toward A,
contact distance `d`, and inverse masses `w_a`, `w_b`, two-way deformable
contact uses

`C = max(d - dot(x_a - x_b, n), 0)`

`delta_a = n C w_a / (w_a + w_b)`

`delta_b = -n C w_b / (w_a + w_b)`.

Normal impulse cancels closing speed and bounded Coulomb tangent impulse cannot
increase relative tangential energy. The impulse applied to B is exactly the
negative of A. Attachment applies the same reciprocal mass split to the
point-to-point error and relative velocity, with the versioned response. Four
fixed sweeps replace adaptive convergence.

Smoke modes preserve the API's two separate one-way claims. Contact treats the
deformable as the obstacle and changes only the smoke endpoint; drag samples
the smoke velocity and changes only the deformable endpoint. Scheduling both
does not turn either into a reciprocal mode. Current and previous separations
share the contact path, so swept thin crossings resolve without another query.
Each endpoint's cumulative positional correction is capped at F8-00's `0.25 L`
limit; overflow is explicit instead of silently widening bounds or rebuilding.

## Bounded work

The frozen workload contains all sixteen compiled pair/mode variants, 6,144
canonical first endpoints, 6,144 canonical second endpoints, and sixteen active
constraints:

| Item | Budget |
|---|---:|
| Coupling sweeps per registered edge | 4 |
| Edge sweeps | 64 |
| Constraint visits | 64 |
| Batched dispatches | 12 |
| Workgroups at 128 threads | 388 |
| Solver scratch | 49,920 bytes |
| Time advances | 0 |
| Spatial updates/rebuilds during solve | 0 |
| Intra-tick host readbacks | 0 |

Dispatch count is independent of edge count because every active constraint
shares each proposal and pair of reduction phases. Scratch is four `Vec3`
proposal streams plus one correction-consumption scalar per endpoint. Candidate
and hierarchy storage remain owned by the shared epoch; there is no per-edge BVH,
pair sort, adaptive retry, or hidden substep.

## Physical quality

Generated production Slang ran on a physical GTX 1080 with all sixteen variants
active in one ordered pass. At the exact 100:1 dynamic mass-ratio boundary,
swept contact ended with zero measured penetration, zero closing speed, zero
tunnels, and reciprocal momentum residual `0.00000010`. Cloth–rope and
soft-body–rope attachment separation was at most 0.5 mm. Particle- and grid-
smoke drag reduced unit relative speed to `0.31640625`, with no mutation of the
smoke sample. Smoke contact moved only smoke and left the deformable obstacle
unchanged.

A stale generation caused no mutation; rebinding the exact generation restored
the operator; invalid profile execution was atomic. A saturated correction
budget reported overflow without positional mutation, and boundary reset made
the exact workload retryable. The sixteen-constraint
equation path ran 120 warmup ticks and 600 CUDA-event samples: p50 0.516256 ms,
p95 1.239392 ms, p99 1.667232 ms, maximum 1.822400 ms. This is operator timing,
not an F8-21 integrated frame-budget result.

## Accuracy tradeoff and legacy separation

Version 1 accepts operator splitting, four fixed sweeps, point-contact
linearization, response-based attachment/drag, deterministic batched reductions,
and prescribed one-way smoke samples. These losses remain bounded by F8-00's
contact, attachment, correction, dissipation, and direction gates. It does not
accept tunneling, generation aliasing, unsupported-mode fallback, extra time
advances, unbalanced two-way exchange, silent correction overflow, or a second
spatial update.

Legacy pair kernels, schedules, tolerances, capability records, and goldens are
unchanged. The new entries cannot be selected implicitly; F8-20 owns public
schedule activation. Commands, provenance, budgets, results, and remaining
cross-device qualification limits are recorded in
[`evidence/F8-19-deformable-smoke-couplings-60hz.json`](evidence/F8-19-deformable-smoke-couplings-60hz.json).
