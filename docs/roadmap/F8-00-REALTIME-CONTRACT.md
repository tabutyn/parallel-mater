# F8-00: real-time single-step profile contract

Status: normative design contract for `realtime-single-step` version 1.0. It is
not a claim that any current driver implements or qualifies the profile. The
machine-readable source of the numbers below is
`conformance/v1/qualification/realtime-profile-v1.json`.

This contract resolves the accuracy/performance tradeoff authorized for F8:
prefer bounded, stable and useful 60 Hz behavior over legacy substep accuracy,
without changing equations per backend or hiding reduced workloads. F8-03 turns
these limits into independent tests, F8-32 implements negotiation, F8-06–21
implement the schedule and solvers, and F8-22 freezes a candidate before any
physical device can qualify it.

## Selection and compatibility

The semantic profile identifier is `realtime-single-step`, version 1.0. F8-32
will reserve the corresponding registry symbols and generated C structures. A
simulation requests the profile and exact major/minor version in its creation
extension chain. Selection is immutable for that simulation.

- Omitting the request retains the existing legacy schedule and `PmStepInfo`
  meaning. The new profile is never a default or silent fallback.
- A missing extension returns `PM_ERROR_EXTENSION_NOT_PRESENT`; an unknown
  profile returns `PM_ERROR_FEATURE_NOT_PRESENT`; an unsupported version returns
  `PM_ERROR_INCOMPATIBLE_VERSION`; an invalid profile parameter returns
  `PM_ERROR_INVALID_ARGUMENT`.
- Under version 1.0, `PmStepInfo.timestep` must be 1/60 second within one
  binary32 ULP and `substeps` must be one. The runtime rejects other values; it
  does not reinterpret them. These checks remain active when validation is off.
- Execution support and physical qualification are separate device properties.
  Compiling a shader or accepting profile creation does not qualify a driver.
- A new major may change schedule or material semantics. A minor may add
  optional observability, but cannot relax the 1.0 envelope. Any numerical,
  workload or tolerance change invalidates affected candidate evidence.

Keeping two solver implementations forever is not required. An eventual legacy
retirement must be an explicit incompatible-version release. Until then, legacy
goldens, quality profiles and results retain their original names and hashes and
cannot be relabeled as version 1.0 evidence.

## Tick and presentation semantics

One successful tick advances time exactly once by 1/60 second and publishes
exactly one dynamic spatial epoch. A full build or refit may contain several
kernels, but `builds + refits <= 1`; an empty successful tick still publishes
one empty epoch. No domain, interaction, recovery path or telemetry path may
advance time again or rebuild the dynamic index.

Each render callback requests at most one physics tick. At display rates above
60 Hz, clients interpolate the previous and current committed snapshots with
alpha in `[0, 1)`; they never extrapolate. Interpolation covers positions,
orientations, domain outputs and generation-tagged topology, and adds at most
one completed tick of latency. Inputs and kinematic targets are sampled once at
the tick boundary.

At sustained 30 Hz, the client executes 30 ticks and advances 0.5 simulated
seconds per wall-clock second. This deliberate slow motion is the overload
behavior. A stall, reset or background transition discards presentation debt,
sets alpha to zero and never triggers catch-up. Reset makes both interpolation
snapshots the reset state.

The fixed algebraic budgets per tick are:

| Solve | Iterations |
|---|---:|
| Rigid contact and joints | 8 |
| Liquid density | 6 |
| Cloth constraints | 8 |
| Soft-body constraints | 8 |
| Rope constraints | 8 |
| Particle-smoke pressure | 6 |
| Grid-smoke projection | 12 |
| Each registered coupling edge | 4 |

These are exact version-1.0 counts, not data-dependent maxima. There are no
extra high-strain, convergence or cold-start iterations. An algebraic iteration
does not advance time or republish the index.

## Supported input envelope

All inputs and intermediate state must be finite. The characteristic length
`L` is the minimum relevant contact diameter, rest edge, support radius or grid
cell width named by a fixture. Both the absolute and normalized limits apply:

| Quantity | Version-1.0 range |
|---|---:|
| Absolute world coordinate | at most 10,000 m |
| Characteristic length `L` | 0.005–2 m |
| Dynamic endpoint mass | 0.001–10,000 kg |
| Interacting dynamic mass ratio | at most 100:1 |
| Linear speed | at most 20 m/s and 0.5 `L` travel/tick |
| Angular speed | at most 10π rad/s and π/6 travel/tick |
| Acceleration displacement | at most 0.25 `L`/tick |
| Total solver correction | at most 0.25 `L`/tick |
| Friction coefficient | 0–2 |
| Restitution | 0–0.8 |
| Dimensionless constraint stiffness | 0 disables; active range 0.05–0.995 |
| Dimensionless compliance ratio | 0.005025–19 |
| Added solver velocity damping | at most 12%/tick |
| Fluid travel | at most 0.5 support radius/tick |
| Smoke-grid CFL | at most 1 |
| Authored deformable strain | at most 25% |

Static and kinematic endpoints are exempt from the dynamic mass-ratio limit,
but their motion and corrections remain bounded. A teleport is a boundary
mutation: the target becomes the staged starting state and must satisfy overlap
and correction limits. It is not a reason to sweep from an old pose, rebuild
mid-tick or accept an already-overlapping invalid state.

For compliant constraints, implementations expose material values in their SI
units. With `w` as effective generalized inverse mass and `c` as compliance,
the dimensionless compliance ratio is `c / (dt² w)` and response is
`1 / (1 + ratio)`. The active ratio range 0.005025–19 corresponds to response
0.995–0.05; zero response disables the constraint. The same generalized-mass
conversion applies to linear, angular and volume constraints. F8-03 includes the
conversion in its oracle; backend-specific remapping is forbidden. User-authored
damping remains physical input. Any additional solver damping must fit the 12%
per-tick allowance and appear in the profile result.

## Quantitative quality gates

Metrics use SI units. Relative exchange residual means
`norm(residual) / max(norm(applied exchange), 1e-6 SI)`. Cold, settled and
long-run windows are 120, 600 and 3,600 ticks. Every sampled value must remain
finite, including intermediate states inspected by an oracle.

### Domains

| Domain | Required limits |
|---|---|
| Rigid | penetration ≤ 5 mm and 0.1 `L`; zero tunneling; 600-tick rest drift ≤ 1 cm / 0.035 rad; joint error ≤ 1 cm / 3°; unforced energy growth ≤ 5% |
| Liquid | density error p95 ≤ 15%, max ≤ 30%; zero free-surface particle loss; closed-container leaks ≤ 1%; closed-system centre drift ≤ 0.05 support radius; unforced energy growth ≤ 5% |
| Cloth | stretch RMS ≤ 3%, max ≤ 10%; bend RMS ≤ 10°; closed-volume error ≤ 15%; pin drift ≤ 2 mm; zero tunneling |
| Soft body | bond stretch RMS ≤ 4%, max ≤ 12%; volume error ≤ 15%; shape RMS ≤ 0.1 `L`; zero inverted volume elements and tunneling |
| Rope | segment stretch RMS ≤ 3%, max ≤ 8%; bend RMS ≤ 15°; attachment error ≤ 1 cm; zero tunneling |
| Smoke | `divergence * dt` RMS ≤ 0.025, max ≤ 0.1; closed tracer-mass error ≤ 10%; analytic transport centroid error ≤ one cell; particle-density error p95 ≤ 25%; speed overshoot ≤ 0.1 mm/s |

These are not legacy tolerances. In particular, larger deformation, density and
volume errors are accepted in exchange for one time advance and bounded work.
No limit implies unconditional stability outside the input envelope.

### Couplings

Every registered capability must satisfy both endpoint-domain gates and every
gate for each enabled mode. F8-03 derives the exact list from the F8-01
capability inventory; it must not maintain a second hand-written pair list.

| Mode | Required limits |
|---|---|
| Contact | penetration ≤ 5 mm and 0.15 contact distance; final closing speed ≤ 0.05 m/s; reciprocal linear/angular impulse residual ≤ 2%/5%; zero tunneling |
| Attachment | endpoint separation ≤ 1 cm and 0.05 support radius; reciprocal linear-impulse residual ≤ 2% |
| Containment | leaked elements ≤ 1%; surface penetration ≤ 0.15 `L`; reciprocal linear-impulse residual ≤ 2% |
| Drag | reciprocal linear-impulse residual ≤ 5%; closed-system energy growth ≤ 1%; relative speed reduction ≥ 5% |
| Pressure | reciprocal linear-impulse residual ≤ 2%; split-versus-combined normalized state error ≤ 5%; density error p95 ≤ 15% |
| Viscosity | reciprocal linear-impulse residual ≤ 2%; closed-system energy growth ≤ 1%; relative speed reduction ≥ 5% |
| Thermal | thermal-energy residual ≤ 5%; overshoot ≤ 2 °C; zero monotonic-heating violations |
| Phase | mass error ≤ 2%; exactly one boundary of latency; no duplicate/missing stable IDs; no source loss when destination admission fails |

There are no pair-specific or backend-specific tolerance overrides. Unsupported
modes remain unsupported rather than receiving weaker gates.

## One-boundary mutation policy

Birth proposals are staged before the single build, represented in conservative
bounds, and admitted with deterministic identity priority. A proposal rejected
for spacing/capacity remains observable. Earlier rejected proposals still
exclude later proposals, so deterministic underfill is an accepted accuracy
tradeoff; retrying admission with another build is forbidden.

Unconditional removals apply at the next boundary. A phase transfer discovered
during a solve is staged for the following boundary. Source removal and
destination admission are atomic; destination failure leaves the source alive.
All coupling endpoints consume one epoch and commit once. Operator splitting is
allowed only when its residual stays inside the mode gates. Topology changes
invalidate affected warm starts and interpolation generations at a boundary.

## Failure and recovery

Known invalid input is rejected before recording or submission. GPU-discovered
motion-envelope escape or non-finite state reports `PM_ERROR_INVALID_STATE` at
completion/explicit wait. Capacity, traversal or candidate overflow reports
`PM_ERROR_CAPACITY_EXCEEDED`.

A failed tick is atomic: previous committed state, simulation clock and pending
inputs are preserved. There is no partial commit, silent clamp, extra iteration,
smaller retry substep or second spatial update. Recovery is a later
caller-controlled tick after changing input or capacity, and its cost is
measured separately.

## Frozen workload tiers and frame allocation

F8-01 confirms eight physical backend tuples. Version 1.0 maps each to exactly
one tier; the mapping is asserted against its machine-readable inventory.

| Tier | Physical targets | Physics p95 | Render/input reserve |
|---|---|---:|---:|
| `desktop-full-v1` | GTX 1080 CUDA/D3D12; RTX 3050 Ti CUDA/Vulkan; M4 Metal | 10.0 ms | 6.67 ms |
| `ipad-gallery-v1` | M1 iPad Metal | 8.5 ms | 8.17 ms |
| `pixel7-gallery-v1` | Pixel 7 Vulkan | 8.5 ms | 8.17 ms |
| `cpu-reference-v1` | i7-11370H Slang CPU | 10.0 ms | 6.67 ms |

The machine contract freezes every workload's live/object/capacity counts. The
desktop tier retains the authored 386-body wall, the 65-body/30,000-particle
fluid-rigid load, the 578-cloth-vertex/912-soft-node/72-rope-node deformable
load, the 4,500-particle smoke load, and a fixed six-family integration load.
The iPad tier retains 4 authored bricks, 256 fluid particles, 4 coupled dynamic
bodies and 128 smoke particles. Pixel 7 retains the same counts except its
existing 128-particle fluid workload. CPU runs the eight-body rigid oracle and
does not acquire unsupported GPU-domain claims.

Those are separate, named product tiers—not adaptive downscaling. A result with
fewer active objects, simplified collision geometry, smaller live count, fewer
interactions or reduced reserved capacity is a different workload and cannot
pass. A device runs every workload applicable to capabilities it advertises;
unsupported or missing work is not a pass.

Physics timing is the host-monotonic critical path from physics encoding start
through physics completion, including required waits and excluding rendering
and presentation. Each sustained run uses 600 warmup ticks followed by 7,200
measured ticks, retains stalls, fails on any aborted tick, reports p50/p95/p99/max
and advances at least 0.99 simulated seconds per wall second. GPU phase timings
are asynchronous where available; unavailable GPU timing is labeled unavailable,
not replaced with CPU timing.

The installed Gallery must separately demonstrate that rendering and input fit
inside the reserved portion of 16.67 ms. A headless p95 result alone never
qualifies the 60 Hz client.

## Intentional losses and evidence boundary

Version 1.0 explicitly accepts less numerical convergence than the legacy
multi-substep profile: bounded penetration/stretch/volume/density error, false
positive spatial candidates, deterministic source underfill, one-boundary phase
latency, overload slow motion and at most one tick of interpolation latency.
It does not accept tunneling, non-finite state, silent overflow, missing reciprocal
exchange or backend-specific equations.

F8-03 must implement failing negative controls for every envelope and threshold.
F8-22 freezes the source revision, profile hash, kernel manifest, workload counts
and budgets. F8-23–30 then measure the exact candidate on the exact F8-01 devices.
No current legacy result, compile-only result or visually stable Gallery run is
evidence that this contract passes.
