# F8-16 smoke dynamics at one 60 Hz step

F8-16 adds particle- and grid-smoke dynamics for execution profile
`realtime-single-step` version 1.0. Each representation advances exactly once
at `dt = 1/60 s`, uses a fixed algebraic budget, and consumes one pinned F8-10
spatial publication. It does not activate the public profile: F8-17 and F8-19
own cross-system coupling, and F8-20 owns the assembled schedule. All legacy
smoke entries, equations, schedules, tolerances, and goldens remain unchanged.

## Particle equations and schedule

Particle support is `h = 3 radius`. Source emission happens at the tick
boundary, before the sole predictor. For wind rate `r_w`, buoyancy `b`, thermal
lift `T`, normalized up direction `u`, and input velocity `v_n`:

`v* = clamp(v_n + (b + T) u dt + (1 - exp(-r_w dt))(wind - v_n), min(v_max, 0.5 h / dt))`

`x* = x_n + dt v*`

The shared spatial publication supplies at most 64 fixed candidates per
particle. It is neither rebuilt nor mutated during six Jacobi sweeps. With
compact weight `w_ij = max(0, 1 - |x_i-x_j|/h)^3`:

`rho_i = 1 + sum_j w_ij`

`q_i = max(rho_i/rho_rest - 1, 0)`

`p_i = (1 - exp(-pressure_stiffness)) q_i`

Each Jacobi proposal averages the pair-pressure displacement
`h w_ij (p_i+p_j)/2` along the separation normal and adds an XSPH-like
viscosity displacement with response `1-exp(-viscosity dt)`. Proposal and
application are separate dispatches. The accumulated six-sweep correction is
clamped to `0.25 h`; there is no convergence exit, retry, hidden substep, or
candidate rebuild. Coincident particles use a stable identity-ordered axis.

Commit performs the only velocity reconstruction and applies fixed 5% solver
damping:

`v_(n+1) = clamp(0.95 (x_(n+1)-x_n)/dt, v_max)`

Age advances once. Thermal lift decays with a three-second time constant.

## Grid equations and schedule

The structured MAC grid uses one cell as its characteristic length and rejects
requests with CFL `v_max dt / dx > 1`. Nonnegative density and temperature
sources are rates per second and are integrated once. Density, temperature,
and each staggered face component use midpoint (RK2-backtrace)
semi-Lagrangian advection. Scalar interpolation is clamped nonnegative and
face velocity is clamped to the material speed limit.

For effective kinematic viscosity
`nu_eff = nu + C_LES^2 dx^2 |strain|`, one bounded neighbor-average update
replaces an unstable explicit Laplacian step:

`a = clamp(1 - exp(-6 nu_eff dt/dx^2), 0, 0.995)`

`u_diffused = u + a (mean_6(u) - u)`

Buoyancy and supplied subgrid force are then integrated once. Closed obstacle
faces take their wall velocity exactly; open domain faces copy the nearest
interior face. The pressure right-hand side is `div(u*)/dt`. Exactly 12
weighted-Jacobi cycles with weight `2/3` solve
`laplacian(p) = div(u*)/dt`, after which the face projection is
`u_(n+1) = u* - dt grad(p)`. Tracers use an RK2 forward sample of the projected
field and advance once. There is no MacCormack correction, multigrid cycle,
residual early exit, adaptive pass, or per-device equation.

Every entry receives profile version `1` and `dt` explicitly. The grid also
requires the authored projection count to equal 12. Unsupported versions,
timesteps, material values, nonfinite state, excessive speed/travel, and
out-of-envelope corrections report status instead of silently changing the
physics.

## Bounded work

The checked desktop smoke workload has 4,500 live/capacity particles, at most
64 candidates per particle, and—when using the grid representation—a
64-by-32-by-64 grid with 131,072 cells and 401,408 staggered faces.

| Item | Particle representation | Grid representation |
|---|---:|---:|
| Time advances | 1 | 1 |
| Algebraic iterations | 6 pressure sweeps | 12 projection cycles |
| Spatial passes | 6 fixed-candidate reads | 1 obstacle-mask read |
| Spatial updates/rebuilds during solve | 0 | 0 |
| Domain dispatches | 21 | 30 |
| Domain workgroups at 128 threads | 756 | 29,988 |
| Candidate/cell-face visits | 3,456,000 | 3,838,356 |
| Solver scratch bytes | 198,008 | 4,281,080 |

The particle schedule is source, predictor, six repetitions of density +
proposal + apply, then commit. The grid schedule is begin, source, two scalar
advections, three face advections, three face-force passes, three boundary
passes, divergence, 12 pressure cycles, three projections, and tracer commit.
`SmokeRealtimeTickContract` pins the spatial epoch, build/refit/publication
counters, exact time advance, exact iterations, exact spatial reads, dispatch
count, and zero solve-time spatial updates.

## Quality and intentional losses

Generated production Slang ran on a physical RTX 3050 Ti. Particle cold-start
emission was finite, advanced once, rejected an invalid profile without
mutation, and consumed at most 40.828 mm of its 63.75 mm correction allowance.
A 4,500-particle high-speed fixture ran 3,600 ticks with 0% density-error p95,
maximum speed 0.220721 m/s, and no nonfinite state.

A compact pressure impulse reduced dimensionless divergence RMS from 0.312500
to 0.006558 and maximum divergence to 0.025056 after exactly 12 cycles. Moving
obstacle velocity and buoyancy checks passed. A nonnegative source followed by
analytic constant-flow transport had relative mass error 0.000048% and centroid
error 0.00000008 cell. The desktop 64-by-32-by-64 grid with 4,500 tracers ran
3,600 ticks with maximum tracer speed 1.000086 m/s and no nonfinite state.

Across 600 uncontended timed ticks after 120 warmup ticks, the complete
particle schedule measured p50/p95 0.059392 ms, p99 0.061440 ms, and maximum
0.063488 ms. The complete grid schedule measured p50 0.710656 ms, p95
0.775168 ms, p99 0.793600 ms, and maximum 0.795648 ms. These are domain-kernel
measurements, not the F8-21 integrated frame-budget result.

Version 1 intentionally accepts first-order semi-Lagrangian numerical
diffusion, one bounded viscosity update, 12 fixed pressure cycles, 64 fixed
particle candidates, six fixed particle pressure sweeps, and 5% particle
velocity damping. It gives up legacy MacCormack/multigrid accuracy, adaptive
convergence, and hidden smaller timesteps to obtain bounded work at one 60 Hz
advance. Cross-family impulses and thermal exchange remain F8-17/F8-19 work.

The unchanged physical CUDA legacy `smoke-grid` case still matches its existing
golden. Shared source compiles for CPU, CUDA, Vulkan, Metal, PTX, and D3D12 HLSL
SM 6.6. Commands, outputs, budgets, and provenance are recorded in
[`evidence/F8-16-smoke-60hz.json`](evidence/F8-16-smoke-60hz.json).
