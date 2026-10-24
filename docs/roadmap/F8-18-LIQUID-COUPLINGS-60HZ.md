# F8-18 liquid cross-system couplings at one 60 Hz step

F8-18 adds the liquid exchange phase for execution profile
`realtime-single-step` version 1.0. It covers only API 0.2 combinations already
advertised for equal-material liquid pairs and liquid interaction with rigid,
cloth, soft-body, rope, particle-smoke, and grid-smoke endpoints. The public
profile remains unadvertised until F8-20 assembles the complete schedule.

## Registered combinations

| Partner | Modes |
|---|---|
| Equal-material liquid | Contact, pressure, viscosity |
| Rigid body | Contact, containment |
| Cloth | Contact, containment |
| Soft body | Contact |
| Rope | Contact |
| Particle smoke | Drag, thermal, phase |
| Grid smoke | Drag, thermal, phase |

These are 15 existing family/mode combinations. The host and device validators
reject every other pair. In particular, typed miscible and immiscible liquid
profiles remain unadvertised and cannot fall back to the equal-material path.
No new capability record or legacy interaction mode is added.

## One shared epoch and one advance

F8-10 supplies generation-tagged constraint records from one published
particle/geometry hierarchy. `LiquidCouplingRealtimeTickContract` pins that
epoch token, build/refit and publication counters, graph order, topology
revision, and both endpoint generations for the full liquid exchange. Edges
are visited in canonical family, mode, then stable-edge order. A retired
endpoint generation produces no proposal; only a boundary-installed record
for the replacement generation can become active.

Every registered edge receives exactly four algebraic sweeps. Each sweep uses
three batched shared-Slang entries:

1. `liquid_coupling_project` computes per-constraint position, impulse,
   temperature, and phase-transfer proposals from the shared snapshot.
2. `liquid_coupling_apply_particles` reduces proposals in stable constraint
   order and updates each liquid endpoint once.
3. `liquid_coupling_apply_partners` applies the opposite position, linear, and
   rigid angular exchange to dynamic partner endpoints once.

The entries never predict, integrate, commit, advance time, sort pairs,
subdivide time, build/refit a hierarchy, or read back to the host. Domain
prediction and final commit remain owned by the domain tasks and F8-20. Thus
the coupling phase has zero time advances and cannot double-integrate either
endpoint or replace the frozen candidate set mid-solve.

## Mechanical, thermal, and phase laws

For liquid point `x`, partner point `b`, normal `n`, interaction distance `d`,
and inverse masses `w_l` and `w_p`, contact and containment use

`C = max(d - dot(x - b, n), 0)`

`delta_x = n C w_l / (w_l + w_p)`

`delta_b = -n C w_p / (w_l + w_p)`.

The previous and current separations share this path, so a thin crossing is
resolved without another spatial query. Normal and bounded Coulomb tangent
impulses are reciprocal. Rigid contacts also apply
`cross(contact_arm, -liquid_impulse)` to angular velocity. Equal-material
pressure uses the same reciprocal mass split with a versioned response;
viscosity reduces relative velocity without increasing kinetic energy. The
equal-material signature must be nonzero and identical at both endpoints.

Smoke drag is intentionally one-way toward the liquid endpoint, matching the
existing API direction. Thermal response is monotonic. If particle and grid
heaters target one liquid in the same sweep, the strongest admissible proposal
wins instead of summing individually valid responses past the hottest
endpoint. Phase work is only a transfer proposal during the solve; it never
changes source or destination storage in place.

`LiquidCouplingBoundaryContract` stages authored sources, outflow, and phase
transfers by stable identity. They become visible atomically at exactly the
next boundary. Capacity or generation failure leaves counts and the complete
pending set unchanged, so the caller can enlarge capacity or restore the
endpoint and retry without source loss or duplicate IDs.

## Bounded work

The frozen desktop workload contains 16 registered coupling edges. Fifteen
cover the complete advertised matrix and one repeats equal-material contact.
The checked budget uses 4,096 liquid particles, 4,128 partner endpoints, and
16 active constraints:

| Item | Budget |
|---|---:|
| Coupling sweeps per registered edge | 4 |
| Edge sweeps | 64 |
| Constraint visits | 64 |
| Batched dispatches | 12 |
| Workgroups at 128 threads | 264 |
| Solver scratch | 33,792 bytes |
| Time advances | 0 |
| Spatial updates/rebuilds during solve | 0 |
| Intra-tick host readbacks | 0 |

Dispatch count is independent of edge count because all constraints share the
proposal and two reduction phases. Scratch is four `Vec3` and two scalar
proposal streams per constraint plus one correction-consumption scalar per
endpoint. There is no per-pair sort, candidate hierarchy, adaptive loop,
retry, or hidden substep.

## Physical quality

Generated production Slang ran on an RTX 3050 Ti with all seven partner
families and all 15 supported family/mode combinations active. At the exact
100:1 dynamic mass-ratio boundary, swept contact and containment ended with
zero measured penetration, 0.00000475 m/s maximum closing speed, and zero
tunnels. Reciprocal linear exchange residual was 0.00000012. The
equal-material pressure result matched the four-sweep analytic
combined-population reference exactly, while viscosity and both smoke drag
representations reduced unit relative speed to 0.31640625 without energy
growth. Two simultaneous heaters remained monotonic and below the hottest
endpoint.

Stale generations produced no mutation; rebound generations became active;
unequal liquid material identity and invalid profile execution were atomic.
Phase capacity failure emitted no transfer and retained the source-side state.
The 16-edge equation path ran 120 warmup ticks and 600 CUDA-event samples;
exact timing is retained in the evidence record. This is operator timing, not
the F8-21 integrated frame budget.

## Accuracy tradeoff and legacy separation

Version 1 accepts operator splitting, four fixed sweeps, point-contact
linearization, response-based pressure/viscosity/drag/thermal exchange,
strongest-proposal thermal reduction, and deterministic batched reductions.
Those losses are bounded by F8-00 correction, contact, conservation,
dissipation, thermal, and phase gates. It does not accept tunneling,
generation aliasing, unequal-material fallback, extra time advances,
unbalanced two-way exchange, silent correction overflow, source loss, or a
second spatial update.

Legacy kernels, schedules, tolerances, goldens, capability records, and profile
selection are unchanged. The new entries cannot be selected implicitly;
F8-20 owns public schedule activation. Commands, provenance, exact budgets,
and results are recorded in
[`evidence/F8-18-liquid-couplings-60hz.json`](evidence/F8-18-liquid-couplings-60hz.json).
