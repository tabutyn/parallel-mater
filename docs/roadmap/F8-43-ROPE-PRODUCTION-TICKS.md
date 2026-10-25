# F8-43 rope operators in production fixed ticks

F8-43 connects the frozen F8-10 deformable spatial emitter and F8-15 rope
self-dynamics to the public native `realtime-single-step` execution path. A
simulation with a live rope is no longer rejected merely because it contains
rope state. CPU, CUDA, Vulkan and Metal adapters call the same Slang operators;
no adapter contains a private or serial CPU rope solve.

## Fixed tick

At the tick boundary, every live rope copies its completed public positions and
velocities into fixed-capacity staging. The production schedule then:

1. Runs `rope_realtime_predict` once at `dt = 1/60`, preserving explicit pinned
   endpoint targets and recording the sole time advance.
2. Emits swept F8-10 segments into a disjoint range of the common spatial
   arena, builds and queries that arena once, and publishes one immutable epoch
   shared with live rigid, liquid, cloth and soft-body resources.
3. Clears the reserved canonical contact view, ready for the interaction
   operators that F8-45 will bind without changing the frozen rope ABI.
4. Executes exactly eight chain-constraint/apply sweeps. The existing contact
   consumer runs only after sweeps 2, 4, 6 and 8. The authored legacy iteration
   count and `maximum_substep_timestep` cannot add a pass, hidden substep, retry
   or spatial rebuild.
5. Reconstructs velocity once and atomically swaps staged positions and
   velocities into public view.

The rope portion records one prediction dispatch, one spatial-emission
dispatch, 20 solver dispatches and one commit dispatch. Every sweep consumes
the F8-10 epoch pinned by `RopeRealtimeTickContract`.

## Storage, validation and failure behavior

Creation reserves public and staged position/velocity arrays, previous and
delta positions, inverse masses, rest bend lengths, explicit endpoint targets,
segment identity/generation arrays, correction/contact scratch and the
time-advance counter. A tick performs no allocation, capacity growth,
intra-tick host readback or global device wait.

Material and geometry inputs are checked against the existing F8-15 envelope
before device work: node scale, radius, total mass, speed, damping, friction,
gravity displacement, correction distance, compliance response and the 100:1
dynamic mass-ratio ceiling. Whole-rope replacement is accepted only at a tick
boundary and receives a new generation/revision.

Exact-limit preflight failure occurs before staged state can publish. The
public failure fixture lowers the workgroup limit by one, observes
`PM_ERROR_CAPACITY_EXCEEDED`, verifies byte-for-byte unchanged output and then
reuses the same capacity. Reset retires the rope and its cached identity;
rewriting the slot and stepping again succeeds from the reserved arena.

## Public and independent checks

The installed public fixture covers a six-node rope, an exact rigid endpoint
pin, finite state and stretch bounds, exact telemetry, failed-tick rollback,
whole-rope replacement from six to five nodes and reset/reuse. It ran through
production Slang CUDA on an RTX 3050 Ti and production Slang Vulkan on a Pixel
7. The Pixel trace names predictor, constraint, apply, contact-consumer, commit
and public-export pipelines plus the common spatial build/publication pipeline.

The independent frozen F8-15 physical CUDA oracle remains the quality judge for
the same five Slang operators. Its stiff one-tick fixture measured
0.00000007 m stretch error, zero attachment drift and a 0.60000002 m/s moving
anchor. Its 3,600-tick cold/impulse/high-mass-ratio run measured 0.00638221 m
stretch RMS, 0.00927449 m maximum stretch error, zero tunneling and 0.00057085 J
final energy; p95 tick time was 0.149504 ms. Protected-closure checking confirms
F8-43 did not change those frozen kernels; the physical CUDA and Vulkan public
runs independently prove production dispatch reaches them.

F8-43 binds rope self-dynamics, segment emission and the frozen explicit-target
and contact consumers. Moving rigid-target generation, rigid/rope contact
generation and reciprocal response belong to F8-45. The installed fixture
therefore uses an exact static endpoint target and an empty canonical contact
view. Unsupported P-series capsule and pair interactions remain unadvertised.

Exact commands, device identities and artifact hashes are recorded in
[`evidence/F8-43-rope-production-ticks.json`](evidence/F8-43-rope-production-ticks.json).
