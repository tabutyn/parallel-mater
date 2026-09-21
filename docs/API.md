# Initial API proposal: fluid and rigid bodies

## The central decision

`parallel_mater::World` owns every simulated fluid and rigid body and advances
their interactions in one call. This replaces the former design where an
application manually called `begin_frame`, `prepare_substep`, contact helpers,
solver-specific completion functions, and telemetry readbacks in the correct
order.

The installed API initially consists of one header:
[`parallel_mater.hpp`](../include/parallel_mater/parallel_mater.hpp).

## Current implementation status

The core and rigid-body milestone implements world ownership, asynchronous
completion, generation-checked rigid handles, forces, impulses, kinematic
targets, device views, and GPU integration. Rigid contact in this milestone is
discrete and resolves dynamic bodies against static or kinematic bodies.
Dynamic–dynamic and continuous rigid collision remain deferred. Fluid and
particle-lifecycle declarations currently return `StatusCode::not_supported`
status and are implemented in PR 3.

## Minimal use

```cpp
using namespace parallel_mater;

World world;
Status status = World::create({.fluid_capacity = 1, .rigid_body_capacity = 8}, world);
if (!status) return report(status);

RigidBodyId floor;
status = world.add_rigid_body(
    {.motion = MotionType::static_body,
     .shape = CollisionShape::plane(),
     .friction = 0.6F},
    floor);
if (!status) return report(status);

RigidBodyId ball;
status = world.add_rigid_body(
    {.motion = MotionType::dynamic,
     .shape = CollisionShape::sphere(0.35F),
     .initial_state = {.position = {0.0F, 2.0F, 0.0F}},
     .mass = 12.0F},
    ball);
if (!status) return report(status);

DeviceSpan<const FluidParticle> particles = make_device_particles();
FluidId water;
status = world.add_fluid(
    {.capacity = 20'000, .particle_radius = 0.0225F}, particles, water);
if (!status) return report(status);

ParticleSpawnPlaneId inlet;
status = world.add_particle_spawn_plane(
    {.fluid = water,
     .plane = {.center = {0.0F, 2.0F, 0.0F},
               .half_extents = {0.4F, 0.4F}},
     .particles_per_second = 2'000.0F,
     .initial_velocity = {0.0F, -1.0F, 0.0F}},
    inlet);
if (!status) return report(status);

ParticleDestroyPlaneId drain;
status = world.add_particle_destroy_plane(
    {.fluid = water,
     .plane = {.center = {0.0F, -2.0F, 0.0F},
               .half_extents = {1.0F, 1.0F}},
     .crossing = CrossingDirection::against_normal},
    drain);
if (!status) return report(status);

status = world.step({.timestep = 1.0F / 60.0F, .substeps = 4});
if (!status) return report(status);

FluidDeviceView water_view;
status = world.fluid_view(water, water_view);
// A CUDA/OptiX renderer can consume water_view.positions directly.
```

## Ownership and handles

- `World` owns all CPU and CUDA allocations.
- `FluidId` and `RigidBodyId` contain an index and generation. Removing an
  object invalidates its old handle; reusing the slot cannot make the old
  handle valid again.
- Initial particles are supplied as a device span. `add_fluid` enqueues a
  device-to-device copy on the supplied stream; the source must remain valid
  until that stream reaches the copy. An empty span creates an emitter-only
  fluid.
- Device views borrow library memory and are never host-dereferenceable.
- A view must be reacquired after a step, add/remove operation, or capacity
  change. `revision` makes accidental caching detectable.
- A world is bound to the CUDA device current during `World::create`.
- A world is movable, not copyable, and externally synchronized.

## Stepping and CUDA streams

`step_async` enqueues a complete frame on the caller's CUDA stream and records
a `FrameToken`. The initial release permits one frame in flight per world.
`ready` polls without blocking; `wait` establishes completion and reports
deferred CUDA failures. A ready token must still be acknowledged with `wait`
before mutating or stepping that world again. Destroying an unacknowledged
token waits automatically so the world cannot retain an unreachable pending
frame. `step` is the convenience wrapper that enqueues and waits.

The fixed frame duration and substep count are explicit. A slow application
lags physical time; the physics layer never invents, drops, or catches up
ticks. Every substep performs fluid neighborhood construction, fluid solve,
rigid integration, fluid–rigid contacts, and equal-and-opposite dynamic-body
reactions in a documented fixed order.

`apply_force` and `apply_impulse` queue contributions for the next submitted
frame and consume them exactly once. `set_kinematic_target` replaces the target
for the next frame. `read_rigid_body_state` and `collect_statistics` are
explicit synchronous readbacks; ordinary stepping performs no telemetry
readback. Device views and contacts describe the most recently completed
frame.

## Fluid contract

The first fluid is a fixed-radius particle fluid with deterministic neighbor
ordering. `FluidOptions` deliberately exposes physical/solver quantities—not
gallery presets or render settings. Capacity is separate from initial count so
emitters can add particles without allocating during a step.

Initial implementation requirements:

- finite positions and velocities;
- no duplicate neighbor IDs or self-neighbors;
- bounded neighbor overflow reported as an error, never silently truncated;
- identical same-GPU particle/contact ordering for identical input;
- no non-finite state;
- collision projection plus velocity response against every supported rigid
  shape;
- reaction impulses applied to dynamic bodies.

Cross-fluid interaction, phase changes, foam, and surface reconstruction are
deferred.

## Spawn and destroy planes

A spawn plane emits into one existing fluid at a rate measured in particles
per second. It is a finite oriented rectangle with an initial world-space
velocity. A deterministic fractional accumulator carries the un-emitted part
of the rate between frames, and a seeded sequence distributes new particles
over the rectangle. New stable particle IDs increase monotonically and never
alias a surviving particle. Emission happens before neighbor construction.

A destroy plane removes a particle when its swept path crosses the finite
rectangle in the selected normal direction. Using the swept path avoids
missing a plane when a fast particle moves from one side to the other in one
substep. Compaction is stable, so surviving particles retain deterministic
order and IDs.

Spawn and destroy capacity is fixed in `WorldOptions`; neither feature may
allocate during stepping. If a fluid is full, emission pauses rather than
overwriting particles, and `spawn_capacity_miss_count` reports how many
particles could not be created. Planes are generation-checked resources that
can be enabled, moved, updated, and removed without rebuilding the fluid.

## Rigid-body contract

The first release supports spheres, boxes, local-Y capsules, and local +Y
planes. A body is static, kinematic, or dynamic:

- static bodies never move;
- kinematic bodies follow explicit targets and impart their velocity to fluid;
- dynamic bodies integrate gravity, forces, impulses, damping, and contact
  reactions.

Dynamic mass must be positive. A zero inertia diagonal requests an analytic
value derived from mass and shape. Plane bodies must be static or kinematic.
Triangle meshes, compound shapes, joints, sleeping, and continuous rigid–rigid
collision are deferred. Dynamic–dynamic rigid collision is also deferred from
the core milestone; fluid–dynamic-body momentum exchange arrives with the
fluid-coupling milestone.

## Contacts are the application extension point

The optional contact buffer reports fluid-particle/rigid-body contacts in a
stable order. These are the physics-side input for painting: stable particle
and body IDs, contact position, normal, and impulse let gallery code update its
own color fields or textures. The physics API does not own paint pixels,
materials, UVs, or textures. The same records can drive sound, objectives,
foam emission, or debugging. Overflow is explicit in `ContactDeviceView` and
statistics.

## Errors and validation

Public calls are `noexcept` and return `Status`. Invalid values and stale
handles fail before work is submitted. `Status::message` points to
library-owned static text; callers do not free it. A CUDA failure carries its
original `cudaError_t`.

## Deliberately absent

- renderer, camera, lights, materials, meshes, textures, or OptiX objects;
- gallery recipes, level order, victory conditions, input bindings, or UI;
- public hierarchy, neighbor, scratch-allocation, or constraint-batch types;
- cloth, rope, soft body, smoke, foam, paint storage, or fracture;
- serialization and network replication;
- CPU fallback or non-CUDA backend.

These omissions are the main defense against another application-shaped API.

## Decisions from the first review

- `World` is the only stepping interface in v0.1.
- Initial fluid data is device-resident.
- A capacity-bounded contact stream is retained as the input to painting and
  other application effects.
- One frame may be in flight per world.
- Sphere, box, capsule, and plane are the initial rigid shapes.
- Deterministic particle spawn and destroy planes are part of the initial
  resource model.
- The gallery boundary and staged roadmap are approved.
