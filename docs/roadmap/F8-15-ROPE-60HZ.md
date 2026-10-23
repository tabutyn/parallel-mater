# F8-15 rope dynamics at one 60 Hz step

F8-15 adds rope self-dynamics for execution profile
`realtime-single-step` version 1. It advances every node exactly once at
`dt = 1/60 s`, performs eight bounded algebraic sweeps and consumes one pinned
F8-10 deformable-spatial publication. It does not activate the complete public
profile: F8-17 owns reciprocal cross-system rope interactions and F8-20 owns
assembled schedule activation. Legacy `rope_advance` and
`rope_advance_coupled` remain profile 0 and are unchanged.

## Equation and schedule

The predictor saves `x_n`, clears correction consumption and performs the only
time advance:

`v* = clamp(exp(-d_v dt) (v_n + g dt), min(v_max, 0.5 h / dt))`

`x* = x_n + dt v*`

Here `h` is node spacing. An attached endpoint is assigned its explicit target
and target velocity; an unattached node with zero inverse mass is fixed.
Coordinates, node count, spacing, radius, per-node mass, interacting mass
ratio, speed, acceleration displacement, strain, compliance, friction and
solver correction must remain in the versioned profile envelope.

For adjacent segment `e=(i,i+1)`, direction `d_e`, rest length `L_e`, inverse
masses `w_i,w_(i+1)` and `alpha=c/dt^2`, one cooperative group forms the
tridiagonal linearized stretch system:

`A[e,e] = w_i + w_(i+1) + alpha`

`A[e,e+1] = -w_(i+1) dot(d_e,d_(e+1))`

`b[e] = -(|x_(i+1)-x_i| - L_e)`

After a bounded Thomas solve `A delta_lambda = b`, node correction is
`delta x = W J^T delta_lambda`. This carries attachment tension across the
complete chain without an adaptive convergence loop. The chain has at most
1,024 nodes and uses one 128-thread cooperative group; lanes cover rows in
strides while the forward/back substitution is deliberately serial and
bounded.

Bending uses the existing rope stretch-compliance scalar as an explicit v1
material compromise. Each node projects the rest length of its two-edge chord
against nodes `i-2` and `i+2` with the same mass-weighted compliant distance
equation. The two parity-chain Jacobi proposals are half-relaxed. This avoids
inventing an unversioned public bend material while adding resistance to sharp
folds. Compliance zero means rigid. Positive compliance must give
`r = c/(dt^2(w_i+w_j))` in `[0.0050251256, 19]`, equivalent to a per-equation
response `1/(1+r)` in `[0.05, 0.995]`.

Structural work may consume at most `0.20 h` per tick, reserving `0.05 h` for
safety contact; all correction together remains at most `0.25 h`. Four contact
projections run after sweeps 2, 4, 6 and 8. They consume the canonical,
prefiltered per-node half-space constraints from the pinned F8-10 snapshot,
including same-owner non-adjacent rope contact, and apply bounded Coulomb
tangential displacement. A contact that cannot be resolved inside the declared
allowance latches an error so the enclosing atomic tick can roll back.

Commit reconstructs velocity with one fixed 5% solver damping:

`v_(n+1) = 0.95 (x_(n+1)-x_n)/dt`

At active contact, inward normal velocity is removed and authored friction
damps the tangent. All new entry points receive profile version `1` and `dt`
explicitly. Another version, timestep or material envelope is rejected. The
same Slang equations compile for CPU C++, CUDA, Vulkan SPIR-V, Metal and D3D12
DXIL; no adapter has a private rope equation or tolerance.

## Bounded work

For the frozen desktop rope workload of 72 nodes, 71 segments, 70 bend chords
and at most one canonical contact constraint per node, the checked domain bound
is:

| Item | Bound |
|---|---:|
| Time advances | 1 |
| Constraint sweeps | 8 |
| Contact passes | 4 |
| Spatial updates/rebuilds during solve | 0 |
| Domain dispatches | 22 |
| Domain workgroups | 22 |
| Stretch-row visits | 568 |
| Directed bend-chord visits | 1,120 |
| Contact-constraint visits | 288 |
| Solver scratch bytes | 26,600 |

The 22 dispatches are one predictor, eight repetitions of chain proposal plus
proposal apply, four contact passes and one velocity commit. Scratch includes
the fixed 1,024-row, 24,576-byte cooperative-group tridiagonal workspace.
`RopeRealtimeTickContract` pins the spatial token, build/refit/publication/query
counters and topology revision across the solve. Reset or topology mutation is
accepted only at a later tick boundary with a new revision and rest buffers.

## Quality and intentional losses

Generated production Slang ran on a physical GTX 1080. A stiff 20%-strained
segment returned to 0.000007% measured error in eight sweeps with zero endpoint
drift. A target moving at 0.6 m/s stayed exact. A fast downward rope stopped at
the 50 mm contact plane with zero inward speed, and a canonical non-adjacent
self-contact reached the same plane.

A 72-node, two-end attached chain ran 3,600 ticks under gravity. Stretch RMS
was 0.6382%, maximum stretch/compression error was 0.9274%, bend RMS was zero,
attachment drift and tunneling were zero, final kinetic energy was 0.000571 J,
and maximum correction was 15.192 mm. Across 600 timed ticks after 120 warmup
ticks, the equation fixture recorded p50 0.507904 ms, p95 1.319936 ms, p99
1.352704 ms and maximum 1.565696 ms. This is domain-kernel evidence, not the
F8-21 complete `desktop-full-v1` frame-budget result.

Version 1 intentionally loses adaptive iteration, hidden substeps, the legacy
active-set contact search and arbitrary authored iteration counts. Bending is a
two-edge chord approximation sharing stretch compliance. One canonical contact
is stored per node, and the cooperative chain solve caps ropes at 1,024 nodes.
These losses are explicit admission constraints rather than device-specific
fallbacks. Cross-system constraint generation and reciprocal impulse exchange
remain F8-17 work; public schedule negotiation remains F8-20 work.

Legacy entry points, profile marker, schedule, tolerances and goldens were not
changed. Commands, outputs and provenance are recorded in
[`evidence/F8-15-rope-60hz.json`](evidence/F8-15-rope-60hz.json).
