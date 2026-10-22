# F8-11 rigid contacts and joints at one 60 Hz step

F8-11 adds the rigid-domain equations for execution profile
`realtime-single-step` version 1.0. The profile advances rigid state once at
exactly `dt = 1/60 s`, reuses the published F8 spatial epoch, and performs
exactly eight contact/joint sweeps. It does not activate the complete public
profile: F8-12 through F8-19 still own other domains and coupling, and F8-20
owns assembled schedule activation. Legacy-profile behavior remains separate.

## Equation and schedule

The ordinary rigid predictor advances position, orientation and velocity once.
AVBD then operates on that predicted pose. For profile version 1, preparation
ignores both the pipeline's authored rigid-iteration count and per-joint
iteration overrides and writes an exact budget of eight. Every sweep visits
each graph color once and advances contact and joint duals once. Pose commit is
followed by hinge-anchor projection and schedule accounting. There is no second
impact phase, adaptive exit, time advance, spatial rebuild or spatial refit.

The contact penalty continues to use generalized mass and the physical step:

`penalty = 1 / (effective_inverse_mass * dt^2)`

Persistent contacts retain the existing penetration, friction-anchor and warm
start equations. A new cold contact closing faster than `0.1 m/s` folds its
normal velocity response into the pose error:

`impact_displacement = -(1 + min(restitution_a, restitution_b)) * v_n * dt`

where `v_n < 0`. This cancels the incoming normal displacement and requests the
bounded restitution displacement in the same eight sweeps. It replaces the
legacy solver's second velocity phase only for profile version 1. After pose
correction, velocity receives `delta_position / dt` and the quaternion-log
angular correction divided by `dt`. A fixed 5% solver-added linear and angular
velocity damping is then applied once per tick, below the profile's 12% cap.

Locked, limited, spring and motor joint rows keep the common Slang AVBD
generalized-mass equations. Active compliant rows use the profile conversion

`response = 1 / (1 + compliance / (dt^2 * effective_inverse_mass))`

and must remain in `[0.05, 0.995]`. Rigid inputs are finite and bounded by the
versioned F8-00 envelope: 5 mm to 2 m characteristic length, 1 g to 10,000 kg
dynamic mass, 100:1 interacting dynamic mass ratio, 20 m/s absolute and 0.5
characteristic-length linear travel, `10*pi rad/s` absolute and `pi/6` angular
travel, 0.25 characteristic-length acceleration and solver correction,
friction `[0,2]`, restitution `[0,0.8]`, and damping `[0,0.12]`. Teleports and
resets are tick-boundary target states, not swept motion.

`PhysicsStepProfile::realtime_single_step_v1` is passed explicitly from the
simulation through every CPU, CUDA, Vulkan/D3D12 and Metal preparation adapter.
One substep never selects it accidentally. Runtime validation accepts only one
substep and one-float-ULP tolerance around `1/60`; rope settings that would add
a hidden substep reject the tick. Public creation remains unsupported and
unadvertised until F8-20 can activate all required domains together.

## Bounded work

For the frozen desktop rigid-wall capacity (386 live bodies, 65,536 contact
slots, 32 joint slots, eight graph colors, 13 workgroups), the checked budget
is:

| Item | Bound |
|---|---:|
| Time advances | 1 |
| Contact/joint sweeps | 8 |
| Spatial updates during solve | 0 |
| Cooperative prepare/solve dispatches | 2 |
| Staged Metal dispatches | 79 |
| Staged workgroups | 1,027 |
| AVBD persistent bytes | 151,168,728 |

The staged count is one preparation, eight repetitions of eight colors plus
dual advance, and six final pose stages. `RigidRealtimeTickContract` pins one
published epoch token and its build/refit/publication counters across all eight
sweeps. Any extra sweep or changed epoch is invalid.

## Quality and intentional losses

Generated production Slang ran for 720 physical-CUDA ticks on an RTX 3050 Ti.
The two-body resting stack stayed below 1.229 mm penetration, drifted 0.042 mm
over the 600-tick settled window, gained no mechanical energy and recorded no
tunneling. The cold-impact fixture ended at `+1.539 m/s` with 26.32% of incoming
kinetic energy; its fixed joint ended at 2.1 micrometres and 0.000043 radians.
Fast rotation, a 100:1 contact and staged teleport/reset remained finite and
inside their gates. The 720-sample small-stack equation-path timing was p50
0.355 ms, p95 0.429 ms, p99 0.860 ms and maximum 0.908 ms. This is domain-kernel
evidence, not the F8-21 full-workload frame-budget claim.

Version 1 intentionally trades accuracy for bounded work: restitution is a
single pose-space approximation, every dynamic rigid receives fixed damping,
there is no adaptive convergence or second velocity solve, and authored
iteration requests cannot buy more accuracy. Inputs outside the declared
envelope are rejected rather than silently substepped. Cross-domain exchange,
full-workload timing and platform qualification remain owned by later F8 tasks.

Commands, exact outputs and toolchain provenance are recorded in
[`evidence/F8-11-rigid-60hz.json`](evidence/F8-11-rigid-60hz.json).
