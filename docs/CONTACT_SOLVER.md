# Shared rigid-contact solver

The CUDA backend now has one rigid-contact solve loop and one velocity-row
implementation. There is no ordinary-stack mode, triangle-only impulse solver,
or world-wide opt-out when a joint, fluid, cloth, rope or soft body exists.

## Backend-neutral core

`include/parallel_mater/solver/contact.hpp` is C++17, header-only and independent
of the CUDA runtime and public `World` API. It owns target velocity (speculative
separation, restitution and fixed-member recovery), nonnegative normal impulse,
Coulomb friction, persistent/transient impulse history, iteration budgets and
the `ContactConvergence` stopping policy.
CUDA invokes this header directly; it is not a second reference implementation.

A backend supplies a vector aggregate with float `x`, `y`, `z`, the shared
`ContactRow<Vector>` (or a compatible row), and a response adapter:

- `relative_velocity()` returns A's contact-point velocity minus B's.
- `normal_response()` returns the normal direction and its summed inverse
  effective mass. Cached values must be invalidated when either pose changes.
- `response(direction)` returns that direction and summed inverse effective
  mass **quadratic in direction length**, `dᵀ M d`. The normal is unit length;
  persistent friction uses the unnormalized sliding tangent, cancelling
  normalization and its square root. Transient impacts retain their unit-tangent
  normalization and scalar clamp order. Returning a unit-direction mass for an
  arbitrary persistent tangent is wrong.
- `apply(response, magnitude)` applies positive impulse to A and negative to B,
  including allowed angular motion and every member of an eligible welded body.
- `apply_friction(impulse)` applies an already-computed friction impulse without
  needlessly evaluating its effective mass or evicting a cached unit tangent.

Material mixing uses minimum restitution and geometric-mean friction. Normal
points from B toward A; positive penetration means overlap; impulses are N·s.
Backend storage, collision generation, guided-body response, compound mass,
position projection and thread synchronization remain backend responsibilities.
Do not implement a free-body approximation for hinge/slider/compound responses.

CPU C++ implementations can include the header directly. Metal shader code
still needs a platform adaptation (including address spaces/math intrinsics);
the header and shared fixtures specify the equations rather than claiming a
tested Metal port. No other backend checkout was available during this change.

## History, cache and scheduling rules

1. Generate contacts and identify stable features. Triangle/open-mesh contacts
   can remain transient; history eligibility is not a separate solver.
2. Load persistent impulses only for matching body generations, cache epoch,
   timestep and nearby local feature/normal. Project old friction onto the new
   tangent plane. Invalidate after world revisions, handle reuse or missed steps.
3. Color contacts so no dynamic body **or welded root** is written by two
   simultaneous pairs. Static/kinematic state is shared read-only. Overflow
   pairs run serially with the exact same row solver.
4. Apply **all** warm starts within each island before any of its solve
   iterations. Keep position projection interleaved with velocity solving;
   transient impacts retain their one-time correction on the first pass.
   Retain the existing positional and guided-body equations. The CUDA response
   cache checks both positions and orientations **after** projection; cache
   overflow prepares the same response locally, never switches equations.
5. Use the maximum `contact_iteration_budget` over live patches: eight normal
   passes or 32 for face patches. Warm-starting is an additional pass. The old
   stack-only 64-pass cold-cache escalation is removed: cache availability,
   world size and unrelated systems cannot select a different solve budget.
   Within that cap, connected contact/joint islands may stop independently:
   at least eight velocity passes and two consecutive complete quiet sweeps
   (maximum linear/angular velocity change <= 1e-7, position change <= 1e-8,
   normal/friction impulse change <= 1e-7 N·s). Check pair activity and net
   whole-sweep body changes: opposing updates must not mask unconverged rows.
   Reduce over the entire island after each sweep, not individual rows. Shared
   static/kinematic bodies do not connect islands; live dynamic joint edges do.
   Reset convergence every substep. This is not sleeping: impulses, new contacts
   and changed joint settings remain active on the next substep.
6. Converge Fixed/Point joints with their contacts. Other joints do not scan
   every contact looking for nonexistent Fixed/Point membership. Finish speed
   limits, guided-body finalization and persistent-cache publication as before.
7. Publish diagnostic impulses after the contact and joint solves. Persistent
   rows already contain the final accumulated support/friction impulse,
   including warm starts; transient rows require a separate within-step sum.
   Respect event capacity and preserve earlier-substep events if later substeps
   contain no contacts.

CUDA retains two *color construction* launch schedules (one-block versus
multi-block), both using the same mutual-minimum ordering, and only one contact
solve kernel. Work is compacted into flat color lists, then distributed across a
resident cooperative grid, including contacts within a single connected stack.
Grid barriers preserve color order. Islands can stop independently, but all
active islands use the same warm-start/position/velocity schedule. Island IDs
gate convergence, not a serial outer loop that would serialize independent
contacts within a warp. Eight-thread
blocks and one resident block per multiprocessor avoid over-subscribing the
barriers on the measured GPU. Worlds with at most 256 bodies, and devices
without cooperative launch, run the **same kernel** in one 128-thread block
with block barriers. Launch geometry never changes the equations, pair order,
iteration budget or convergence test. No per-color CPU launch loop or graph
cache is involved. Temporary launch-tuning environment variables were removed.

Prepared data includes material mixing, contact arms, normal effective mass,
and (for retained free-body features) normal/tangent angular responses and the 2-D tangent
response matrix. World-space inertia tensors are invalidated on rotation;
translations refresh arms and Jacobians without rebuilding those tensors.
Full rows enter the cache only after the same pose is observed twice, avoiding
writes that neighboring position projections would immediately invalidate.
Moving rows are prepared locally with current poses. Guided/compound responses
retain the generalized body adapter. Its normal masses occupy a compact scalar
cache, without copying unused free-body response columns. Transient impacts retain their direct quaternion response and
global state-update arithmetic, because changing it perturbs coupled ropes.
Preparation uses each body's actual free, guided or compound response, never a free-body
approximation for a constrained body. Refresh after translation or rotation,
including joint projections. Keep a row's prepared data
local while evaluating it; global cache references in the inner adapter force
costly reloads after state writes. No Jacobian is frozen across changing
poses. Cache slots index live contact patches, not empty broad-phase candidates;
the bounded-cache overflow uses identical row preparation without retention.
The maximum budgets are unchanged, but precomputing response
changes floating-point trajectories: integrated tests
are required, not just scalar equation fixtures.
Persistent patches may keep pair state local; transient patches preserve their
global-state store order. Prepared response and simplified friction arithmetic
can still change floating-point trajectories. Both storage policies invoke the same
shared row solver. The free-body adapter excludes global world pointers from
its velocity arithmetic; it does not duplicate normal/friction equations or
select another iteration budget. Welded compounds retain aggregate application.

`WorldStatistics` exposes last-substep island/early-exit counts, maximum velocity
passes, colors, overflow pairs, GPU block count, candidate pairs and live pairs.
These are diagnostics, not switches that select physics.
Convergence reductions group same-island lanes within each warp and skip zero
atomic updates; opposing contact impulses still prevent a false early stop.

## Port and regression checks

Run the scalar fixtures without any CUDA toolkit:

```sh
cmake -S tests/solver -B /tmp/parallel-mater-contact-core-build
cmake --build /tmp/parallel-mater-contact-core-build
ctest --test-dir /tmp/parallel-mater-contact-core-build --output-on-failure
```

The regular build also runs identical fixtures on CUDA and compares host/device
outputs, including rebound, speculative separation, Coulomb clamping, removal
of stale support, immovable pairs and penetration recovery. Rigid tests exercise
isolated support with/without an unrelated Point joint at 4/31/32/256/257 bodies,
static-ground island isolation, joint merging, impact reactivation and deferred
warm-start support, and empty-candidate overflow beyond 4,096 pairs. The same
25 equation/convergence fixtures run on CPU and CUDA, including anisotropic
tangent response and opposing impulses hidden behind unchanged body velocities.

For full-engine parity use [conformance v1](../conformance/v1/README.md) and the
same canonical inputs/assets. Keep row fixtures and integrated conformance
separate: scalar parity cannot prove collision detection, coupled-system
behavior, joint projection or full-scene determinism. Never regenerate goldens
or increase tolerances merely to bless a solver refactor.

The comparator now matches contacts one-to-one by stable identity and all
existing field tolerances. Previously sorting entire JSON records could let a
tiny friction rounding difference reorder unrelated contact points and report
false backend failures. Tests cover ambiguous matching, duplicate/missing
contacts, changed identities and real impulse violations; none are ignored.
