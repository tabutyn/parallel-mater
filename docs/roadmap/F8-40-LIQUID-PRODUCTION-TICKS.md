# F8-40 liquid operators in production fixed ticks

F8-40 connects the frozen F8-09 spatial consumers and F8-12 liquid
self-dynamics to the public native `realtime-single-step` execution path. A
simulation with live liquid populations is no longer rejected merely because
it contains liquid state. The same production schedule used by rigid bodies
now owns liquid prediction, one common spatial epoch, six density sweeps,
viscosity, commit and public output.

## Fixed tick

At the tick boundary, every live liquid population copies its completed public
position and velocity into fixed-capacity staging. The production schedule then
performs this order:

1. Run `fluid_realtime_predict` once for each population with `dt = 1/60` and
   profile version 1.
2. Emit all rigid and liquid leaves into disjoint ranges of the common spatial
   arena, build it once, query every declared liquid slot once and publish one
   epoch.
3. Filter each population's candidates by owner identity. Empty live counts
   retain their reserved ranges as tombstones and execute safely.
4. Repeat density/lambda, position-delta and apply-delta exactly six times.
   Every repetition reads the same published epoch; none builds, refits, sorts
   or queries the index again.
5. Run viscosity once, reconstruct velocity once, and atomically publish the
   staged position/velocity pair with the other tick state.

The 21 liquid operator dispatches remain the frozen F8-12 sequence: one
predictor, eighteen density-sweep dispatches, one viscosity dispatch and one
commit. Their native CPU, CUDA, Vulkan and Metal bindings all call the common
Slang entry points; there is no serial CPU physics path in a GPU driver.

## Ownership and bounded work

Creation allocates accumulated corrections, position deltas, velocity deltas
and the time-advance counter for every declared particle capacity. It also
allocates the common query bounds, masks, excluded keys and owner-filtered
candidate views from the checked spatial plan. A tick performs no allocation,
capacity growth, host readback, global device wait, adaptive pass or hidden
substep.

All liquid registrations must provide predict, spatial, domain and commit
bindings before the schedule begins. Material validation enforces the F8-12
profile envelope before device work. Work counters include the liquid
predictor, one emit/filter pair, six density iterations, viscosity and commit;
the same exact-limit preflight used by the other domains rejects work before a
partial publication. The reference failure driver proves that a one-below
workgroup limit leaves both liquid outputs byte-for-byte unchanged and permits
the next tick to reuse the same capacities.

## Public and independent checks

The public runtime test creates two populations with capacities four and seven.
It executes a one-particle free surface beside an empty population, replaces
them with unequal four- and three-particle populations, and advances cold and
warm ticks. Public output must stay finite and inside the declared one-step
envelope. Statistics must report one time advance, six liquid density
iterations, one index build, one publication and zero intra-tick host
readbacks. Reset then reuses the same fixed allocations.

The installed native test ran through the production Slang Vulkan driver on a
Pixel 7. The separate mobile liquid suite passed on the same device. Those
public-path results are combined with the independent F8-09 brute-force
neighbor/overflow oracle and F8-12 compression, free-surface, closed-envelope,
viscosity and 720-tick stability oracle; private kernel tests are not used as a
substitute for the public run.

This task binds liquid self-dynamics only. Cross-population material operators
and rigid/deformable containment remain explicit F8-46 coupling work; source,
outflow and other boundary mutation remain F8-48. Their absence does not cause
a liquid-only simulation to fall back to the legacy scheduler.

Exact commands, device identity and artifact hashes are recorded in
[`evidence/F8-40-liquid-production-ticks.json`](evidence/F8-40-liquid-production-ticks.json).
