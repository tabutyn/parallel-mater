# F8-14 soft-body dynamics at one 60 Hz step

F8-14 adds soft-body self-dynamics for execution profile
`realtime-single-step` version 1.0. It advances every node exactly once at
`dt = 1/60 s`, executes eight bounded algebraic sweeps and consumes one pinned
F8-10 deformable-spatial publication. It does not activate the complete public
profile: F8-18 owns cross-system contact generation/reciprocity and F8-20 owns
assembled schedule activation. The legacy predictor, authored iteration count,
shape matcher, spring damping and contact schedule are unchanged.

## Equation and schedule

The predictor saves `x_n`, clears scalar correction consumption and performs
the only time advance:

`v* = clamp(exp(-d_v dt) (v_n + g dt), min(v_max, 0.5 h / dt))`

`x* = x_n + dt v*`

Here `h` is the node's shortest rest bond. Inverse mass zero remains an exact
pin. Dynamic mass, coordinates, speed, acceleration displacement, node radius
and material values must remain in the profile envelope. Interacting dynamic
masses may differ by at most 100:1.

Each of eight sweeps first evaluates the rest-shape mapping. This ordering makes
an inverted input observable before bond projection could conceal it. For
movable mass `m_i`, current and rest centers `c` and `c_0`, the deformation is:

`A = (sum_i m_i (x_i-c)(x_i^0-c_0)^T) A_0^-1`

The polar rotation `R` is found by the existing fixed 12-iteration quaternion
solve. A non-finite or non-positive deformation determinant sets the inversion
status; the enclosing atomic tick must roll back instead of publishing an
inverted state. Otherwise shape matching proposes:

`delta x_i = s (c + R(x_i^0-c_0) - x_i)`

where `s = 0` disables shape matching and active `s` is in `[0.05, 0.995]`.
The mass-weighted mean correction is removed to preserve translation. The
existing public soft-body API describes a rest-shape matrix rather than a
tetrahedral mesh, so version 1 deliberately uses this shape constraint for
volume retention; it does not invent an unexposed per-tetrahedron material.

The sweep then visits the fixed directed-bond graph. For bond `i,j`, rest
length `L_0`, uniform compliance `c_s`, and inverse masses `w_i,w_j`:

`C = |x_i-x_j| - L_0`, `alpha = c_s/dt^2`

`delta x_i = -w_i C (x_i-x_j) / (|x_i-x_j|(w_i+w_j+alpha))`

Per-node proposals are averaged over incident bonds and applied in a separate
dispatch. Compliance zero means rigid. Positive compliance must give
`r = c_s/(dt^2(w_i+w_j))` in `[0.0050251256, 19]`, equivalent to response
`1/(1+r)` in `[0.05, 0.995]`. There is no convergence exit, retry, extra
high-strain pass or authored iteration override.

Shape and bond work together may consume at most `0.20 h` per tick, reserving
`0.05 h` for safety contact. Four contact projections run after sweeps 2, 4, 6
and 8. F8-18 will write the canonical, prefiltered per-node half-space buffers
from the pinned F8-10 snapshot. F8-14 solves
`n dot x >= plane_offset + node_radius` and applies bounded Coulomb tangential
displacement. Total correction remains at most `0.25 h`; unresolved contact is
an error, never accepted tunneling.

Commit reconstructs velocity and applies one fixed 5% solver damping:

`v_(n+1) = 0.95 (x_(n+1)-x_n)/dt`

At active contact, inward normal velocity is removed and authored friction
damps the tangent. Version 1 does not run the legacy neighbor-relative spring
damping pass; `springDamping` must therefore be zero. This intentional loss
keeps the fixed work budget explicit instead of hiding another traversal.
Authored velocity damping remains the predictor's exponential coefficient.

Every new entry receives profile version `1` and `dt` explicitly. Another
version, timestep, material range or mass ratio is rejected. One
`soft_body.slang` source produces CUDA, Vulkan SPIR-V, Metal and D3D12 HLSL;
no adapter owns a private soft-body equation or tolerance.

## Bounded work

For the frozen `desktop-full-v1/deformable-mixed` soft body (912 live/capacity
nodes, 130,033 directed bonds and at most one canonical contact constraint per
node), the checked self-domain bound is:

| Item | Bound |
|---|---:|
| Time advances | 1 |
| Constraint sweeps | 8 |
| Contact passes | 4 |
| Spatial updates/rebuilds during solve | 0 |
| Domain dispatches | 30 |
| Domain workgroups at 128 threads | 184 |
| Directed-bond visits | 1,040,264 |
| Contact-constraint visits | 3,648 |
| Solver scratch bytes | 25,560 |

The 30 dispatches are one predictor, eight repetitions of shape projection +
bond proposal + proposal apply, four contact passes and one velocity commit.
Surface skinning is a render-output update and is not included in this
self-dynamics bound. `SoftBodyRealtimeTickContract` pins the spatial token,
build/refit/publication/query counters and topology revision across the solve.
It accepts one 1/60 advance, exactly eight sweeps and the fixed four-pass
contact cadence. Topology mutation or reset is accepted only at a later tick
boundary with a new revision/token.

## Quality and intentional losses

Generated production Slang ran through native Metal on a physical 8-core Apple
M4. A 100:1 mass-ratio tetrahedron began with 79.4% volume error, 22.2816% bond
RMS error and 25%-bounded deformation. One tick reduced these to 0.895625%
volume error, 1.02437% bond RMS, 1.99945% maximum bond error and 1.7224% of
characteristic length shape RMS. The same fixture cold-started under gravity,
impacted a plane, and ran 3,600 ticks with zero tunneling. Its settled metrics
were 0.384992% volume error, 0.112162% bond RMS, 0.259012% maximum bond error,
0.132958% shape RMS and `1.32945e-9 J` kinetic energy.

Across the final 600 ticks, the four-node equation fixture recorded p50
0.126 ms, p95 0.12625 ms, p99 0.1265 ms and maximum 0.202417 ms. This is
domain-kernel evidence, not the F8-21 full `desktop-full-v1` frame-budget result.

Version 1 intentionally loses adaptive iteration, legacy authored solver
iterations, neighbor-relative spring damping, cross-tick shape warm starts and
the option to buy accuracy with smaller hidden timesteps. Contact storage
selects one canonical constraint per node rather than an unbounded manifold.
Soft-body self-contact is not an advertised capability in the current
inventory, so this task does not fabricate it.

The unchanged legacy `soft-body-core` path still passes its CUDA-reference
golden on the same Metal device. No legacy entry point, schedule, tolerance or
golden changed. Commands, outputs and provenance are recorded in
[`evidence/F8-14-soft-body-60hz.json`](evidence/F8-14-soft-body-60hz.json).
