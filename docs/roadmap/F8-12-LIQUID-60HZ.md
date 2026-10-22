# F8-12 liquid dynamics at one 60 Hz step

F8-12 adds the liquid self-dynamics equations for execution profile
`realtime-single-step` version 1.0. The profile advances particle state once at
exactly `dt = 1/60 s`, consumes the owner-filtered F8-09 spatial snapshot, and
performs exactly six density sweeps. It does not activate the complete public
profile: F8-18 owns cross-system liquid coupling and F8-20 owns assembled
schedule activation. The legacy cell-sort/force/integrate path is unchanged.

## Equation and schedule

At the tick boundary, F8-09 applies outflow compaction and deterministic source
admission before the sole spatial publication. The liquid predictor then saves
`x_n` and performs the only time advance:

`v* = clamp(exp(-d_v dt) (v_n + g dt), min(v_max, 0.5 h / dt))`

`x* = x_n + dt v*`

where `h` is support radius and `d_v` is authored velocity damping. Every later
stage is algebraic and uses the same filtered candidate arrays. Density uses the
normalized three-dimensional poly6 kernel and explicit particle volume:

`rho_i = m (W(0,h) + sum_j W(|x_i-x_j|,h))`, `m = rho_0 V_0`

`C_i = max(rho_i / rho_0 - 1, 0)`

The one-sided constraint removes compression without tensile attraction at a
free surface. Its Jacobi multiplier and correction are:

`lambda_i = -C_i / (sum_k |grad C_i,k|^2 + 1e-6/h^2)`

`delta x_i = s sum_j (lambda_i + lambda_j) (m/rho_0) grad W_ij`

`s = clamp(1 - exp(-repulsion dt), 0.05, 0.995)`

Each sweep computes all multipliers, computes all deltas, then applies all
deltas. Total vector correction is clamped to `0.25 h` for the tick. There are
exactly six sweeps: no authored iteration override, cold-start pass, convergence
exit, retry, neighbor rebuild, sort or second time advance.

After the sixth sweep, viscosity uses a symmetric XSPH weight:

`delta v_i = (1-exp(-nu dt)) sum_j 2m W_ij/(rho_i+rho_j) (v_j-v_i)`

Authored normal damping applies the same symmetric weight to the normal
relative velocity. Commit reconstructs velocity from corrected displacement,
adds the Jacobi velocity delta and applies one fixed 5% solver damping:

`v_(n+1) = 0.95 ((x_(n+1)-x_n)/dt + delta v_i)`

The result is clamped to the authored maximum speed. The solver-added 5% is
below the profile's 12% limit. All intermediate position, density, multiplier,
correction and velocity results must remain finite.

For version 1, `rest_particle_volume` is mandatory rather than using the legacy
cubic-diameter fallback. `repulsion`, viscosity, normal damping and velocity
damping are nonnegative inverse-second response rates. Active density response
must map into `[0.05, 0.995]`; zero explicitly disables pressure. Maximum speed
must satisfy both 20 m/s and `0.5 h` travel per tick. Positive compression error
is the density metric for constrained particles; free-surface under-support is
instead covered by particle-loss, finite-state and center-drift checks and does
not create a tensile correction.

Every new entry point receives profile version `1` and `dt` explicitly and
rejects another version, timestep or material envelope. The common Slang source
compiles for CPU, CUDA, Vulkan, Metal and HLSL. One-substep legacy execution
does not select it accidentally. Public creation remains unsupported and
unadvertised until F8-20 can activate every required domain together.

## Bounded work

For the frozen desktop fluid workload of 30,000 live/capacity particles and
128 candidate slots per query, the checked upper bound is:

| Item | Bound |
|---|---:|
| Time advances | 1 |
| Density sweeps | 6 |
| Spatial updates/sorts during solve | 0 |
| Domain dispatches | 21 |
| Domain workgroups at 128 threads | 4,935 |
| Candidate visits upper bound | 49,920,000 |
| Solver scratch bytes | 1,320,008 |
| F8-09 filtered-candidate bytes | 15,360,000 |

The 21 domain dispatches are one predictor, six repetitions of
density/lambda + correction + apply, one viscosity pass and one commit.
`FluidRealtimeTickContract` pins the published epoch counters, accepts one
1/60 advance, and accepts exactly six sweeps. Any extra advance/sweep or changed
build/refit/publication/query counter is invalid.

## Quality and intentional losses

Generated production Slang ran on a physical RTX 3050 Ti. A 64-particle pack
started with 38.56% p95 compression and reached 0% positive compression error
after one advance and six sweeps; its largest correction was 22.918 mm, or
0.2292 support radius. A separate viscosity fixture retained 55.34% of initial
kinetic energy after its first tick. Across a 720-tick cold/settled window with
one outflow and two boundary source admissions, it recorded no particle loss,
no leak from the one-metre closed test envelope, no energy growth and center
drift of 0.02962 support radius. Its 720 equation-path samples were p50 0.850
ms, p95 1.092 ms, p99 1.345 ms and maximum 1.608 ms. This is small-fixture
domain-kernel evidence, not the F8-21 full-workload frame-budget claim.

Version 1 intentionally drops tensile pressure/surface tension, adaptive
convergence and the legacy option of buying accuracy with more substeps or
iterations. Free surfaces can have a deficient raw SPH sample but receive no
inward artificial pressure. Containment/contact detection and reciprocal rigid,
cloth, soft-body, rope and smoke exchange remain F8-18 work; this task bounds
the self-solver motion consumed by those operators. Inputs outside the declared
envelope reject instead of silently substepping.

Legacy evidence remains separate. The old fluid entry points and reference
schedule were not changed. The three known CPU legacy self-quality tests still
produce their baseline stable-ID fixture failure, while the unsupported-public-
profile test passes. No legacy failure is counted as new-profile evidence.

Commands, exact outputs and toolchain provenance are recorded in
[`evidence/F8-12-liquid-60hz.json`](evidence/F8-12-liquid-60hz.json).
