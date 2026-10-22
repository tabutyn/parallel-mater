# F8-13 cloth dynamics at one 60 Hz step

F8-13 adds cloth self-dynamics for execution profile
`realtime-single-step` version 1.0. It advances every vertex exactly once at
`dt = 1/60 s`, executes eight bounded algebraic sweeps and consumes one pinned
F8-10 deformable-spatial publication. It does not activate the complete public
profile: F8-18 owns cross-system contact generation/reciprocity and F8-20 owns
assembled schedule activation. The legacy deformable predictor, authored
iteration count, strain limiter, fracture and contact schedule are unchanged.

## Equation and schedule

The predictor saves `x_n`, clears scalar correction consumption and performs
the only time advance:

`v* = clamp(exp(-d_v dt) (v_n + g dt), min(v_max, 0.5 h / dt))`

`x* = x_n + dt v*`

Here `h` is the vertex's shortest active rest length. Inverse mass zero is an
exact pin. Dynamic mass, coordinates, speed, acceleration displacement,
thickness and material values must remain in the profile envelope.

Each of eight Jacobi sweeps visits the fixed directed-link graph. For link
`i,j`, rest length `L_0`, compliance `c`, and inverse masses `w_i,w_j`:

`C = |x_i-x_j| - L_0`, `alpha = c/dt^2`

`delta x_i = -w_i C (x_i-x_j) / (|x_i-x_j|(w_i+w_j+alpha))`

Per-vertex proposals are averaged over active incident constraints, then
applied in a separate dispatch. Compliance zero means rigid. Positive
stretch/bend compliance must give
`r = c/(dt^2(w_i+w_j))` in `[0.0050251256, 19]`, equivalent to response
`1/(1+r)` in `[0.05, 0.995]`. There is no convergence exit, retry, extra
high-strain pass or authored iteration override.

Closed manifolds use the existing signed-volume gradient with a versioned XPBD
multiplier:

`C_V = orientation V(x) - V_target`

`delta lambda = (-C_V - (c_V/dt^2) lambda) /
                 (sum_i w_i |grad_i V|^2 + c_V/dt^2)`

Positive volume compliance is checked against the same response interval using
the current volume-gradient effective inverse mass. The multiplier resets on
the first sweep of every tick; it is not a cross-tick warm start.

Four contact projections execute after sweeps 2, 4, 6 and 8. F8-18 will write
the canonical, prefiltered per-vertex constraint buffers from the pinned F8-10
snapshot. F8-13 solves the supplied half-space `n dot x >= o + thickness`,
then applies bounded Coulomb tangential displacement. Structural and volume
work may consume at most `0.20 h`, reserving `0.05 h` for safety contact; every
correction together remains at most `0.25 h`. A contact that cannot be resolved
inside that declared allowance sets an error bit so the enclosing atomic tick
can roll back instead of publishing tunneling.

Commit reconstructs velocity and applies one fixed 5% solver damping:

`v_(n+1) = 0.95 (x_(n+1)-x_n)/dt`

At an active contact, inward normal velocity is removed and authored contact
friction damps the tangent. The solver-added damping is below the profile's 12%
limit. All intermediate values must remain finite.

Every new entry receives profile version `1` and `dt` explicitly. Another
version, timestep or material envelope is rejected. The common Slang source
compiles for CPU, CUDA, Vulkan, Metal and D3D12 HLSL. No adapter has a private
physics equation or tolerance.

## Bounded work

For the frozen desktop deformable workload's two 17-by-17 cloth grids (578
vertices, 1,024 triangles, 6,144 directed stretch/bend links and at most one
canonical contact constraint per vertex), the checked domain bound is:

| Item | Bound |
|---|---:|
| Time advances | 1 |
| Constraint sweeps | 8 |
| Contact passes | 4 |
| Spatial updates/rebuilds during solve | 0 |
| Domain dispatches | 31 |
| Domain workgroups at 128 threads | 126 |
| Directed-link visits | 49,152 |
| Contact-constraint visits | 2,312 |
| Solver scratch bytes | 23,132 |

The 31 dispatches are one predictor, eight repetitions of link proposal +
proposal apply + volume projection, four contact passes, one velocity commit
and one surface update. `ClothRealtimeTickContract` pins the spatial token,
build/refit/publication/query counters and topology revision across the solve.
It accepts one 1/60 advance, exactly eight sweeps and the fixed four-pass
contact cadence. Topology mutation or reset is accepted only at a later tick
boundary with a new revision/token.

## Quality and intentional losses

Generated production Slang ran on a physical RTX 3050 Ti. A stiff 20%-strained
link returned to zero measured strain in eight sweeps with zero pin drift; a
fast downward vertex stopped at the 10 mm contact plane with zero inward speed.
A tetrahedron starting at 50% volume error ended at 0.1155% error. An open
64-vertex pinned cloth ran 3,600 ticks with stretch RMS 2.6326%, maximum stretch
7.2607%, bend RMS 0 degrees, zero pin drift, zero tunneling, zero final kinetic
energy and maximum correction 22.615 mm. All are inside the frozen cloth gates.

Across 600 timed ticks after 120 warmup ticks, the 64-vertex equation fixture
recorded p50 0.120832 ms, p95/p99 0.121856 ms and maximum 0.129024 ms. This is domain-kernel
evidence, not the F8-21 full `desktop-full-v1` frame-budget result.

Version 1 intentionally loses adaptive iteration, legacy 32-pass strain
recovery, cross-tick volume warm starts and the option to buy accuracy with
smaller hidden timesteps. Contact storage selects one canonical constraint per
vertex rather than an unbounded manifold. Cloth self-contact is not an
advertised capability in the current kernel/capability inventory, so this task
does not fabricate it; F8-18 owns advertised distinct-family contacts and F8-20
owns profile activation.

Legacy `cloth-core` still matches the CUDA golden. `cloth-tear` retains the
already recorded reference mismatch from P1-59; no legacy function, tolerance,
golden or schedule changed. Commands, outputs and provenance are recorded in
[`evidence/F8-13-cloth-60hz.json`](evidence/F8-13-cloth-60hz.json).
