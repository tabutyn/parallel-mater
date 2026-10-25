# F8-42 soft-body operators in production fixed ticks

F8-42 connects the frozen F8-10 deformable spatial emitter and F8-14
soft-body self-dynamics to the public native `realtime-single-step` execution
path. A simulation with a live soft body is no longer rejected merely because
it contains soft-body state. CPU, CUDA, Vulkan and Metal adapters call the same
Slang operators; no adapter contains a private or serial CPU soft-body solve.

## Fixed tick

At the tick boundary, every live soft body copies its completed public node
positions and velocities into fixed-capacity staging. The production schedule
then:

1. Runs `soft_body_realtime_predict` once at `dt = 1/60`, preserving exact
   pinned nodes and recording the sole time advance.
2. Skins the predicted surface once, emits swept F8-10 triangles into a
   disjoint range of the common spatial arena, builds and queries that arena
   once, and publishes one immutable epoch shared with live rigid, liquid and
   cloth resources.
3. Clears the reserved canonical contact view, ready for the interaction
   operators that F8-45 will bind without changing the frozen soft-body ABI.
4. Executes exactly eight shape/link/apply sweeps. Shape matching is the
   existing public soft-body representation's volume-retention operator. The
   existing contact consumer runs only after sweeps 2, 4, 6 and 8. The authored
   legacy iteration count cannot add a pass, hidden substep, retry or spatial
   rebuild.
5. Reconstructs velocity once, skins the completed surface once and atomically
   swaps staged node and surface state into public view.

The soft-body portion records two prediction dispatches, one spatial-emission
dispatch, 28 solver dispatches and two commit dispatches. Every sweep consumes
the F8-10 epoch pinned by `SoftBodyRealtimeTickContract`.

## Storage, validation and failure behavior

Creation reserves public and staged node position/velocity arrays, staged
surface positions, the frozen directed-neighbor representation, per-node
characteristic lengths, correction/contact scratch, surface triangle
generations, shape orientation scratch and the time-advance counter. A tick
performs no allocation, capacity growth, intra-tick host readback or global
device wait.

Material and geometry inputs are checked against the existing F8-14 envelope
before device work: coordinate and bond scale, node radius, velocity and spring
damping, friction, maximum speed, shape stiffness, compliance response and the
100:1 dynamic mass-ratio ceiling. Whole-body replacement is accepted only at a
tick boundary and receives a new generation/revision.

Exact-limit preflight failure occurs before staged state can publish. The
public failure fixture lowers the workgroup limit by one, observes
`PM_ERROR_CAPACITY_EXCEEDED`, verifies byte-for-byte unchanged output and then
reuses the same capacity. Reset retires the body and its cached identity;
rewriting the slot and stepping again succeeds from the reserved arena.

## Public and independent checks

The installed public fixture covers a sparse-capacity five-node soft body, an
exact pin, all ten pair bonds, surface export, exact telemetry, failed-tick
rollback, whole-body replacement and reset/reuse. It ran through production
Slang CUDA on an RTX 3050 Ti and production Slang Vulkan on a Pixel 7. The
Pixel trace names predictor, surface, shape, link, apply, contact-consumer,
commit and public-export pipelines plus the common spatial build/publication
pipeline.

The independent frozen F8-14 physical Metal oracle remains the quality judge
for the same six Slang operators. Its one-tick 100:1 mass-ratio deformation
reduced volume error from 79.4% to 0.895625%, bond RMS error from 22.2816% to
1.02437%, and shape RMS error to 1.7224% of characteristic length. It rejects
an inverted mapping before publication. Its cold-start impact run completed
3,600 ticks with zero tunneling and settled at 0.384992% volume error,
0.112162% bond RMS error and 0.132958% shape RMS error. Protected-closure
checking confirms F8-42 did not change those frozen kernels; the physical CUDA
and Vulkan public runs independently prove production dispatch reaches them.

F8-42 binds soft-body self-dynamics, surface emission and the frozen contact
consumer. Rigid/soft contact generation and reciprocal response belong to
F8-45, other cross-domain couplings retain their later leaves, and no
unavailable interaction mode is advertised.

Exact commands, device identities and artifact hashes are recorded in
[`evidence/F8-42-soft-body-production-ticks.json`](evidence/F8-42-soft-body-production-ticks.json).
