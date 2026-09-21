# Initial API proposal: fluid and rigid bodies

## The central decision

`parallel_mater::World` owns every simulated fluid and rigid body and advances
their interactions in one call. This replaces the former design where an
application manually called `begin_frame`, `prepare_substep`, contact helpers,
solver-specific completion functions, and telemetry readbacks in the correct
order.

The installed API initially consists of one header:
[`parallel_mater.hpp`](../include/parallel_mater/parallel_mater.hpp).

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

std::vector<FluidParticle> particles = make_particles();
FluidId water;
status = world.add_fluid(
    {.capacity = 20'000, .particle_radius = 0.0225F}, particles, water);
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
- Input particle spans are host memory copied during `add_fluid` in the initial
  release and may be released when that call returns.
- Device views borrow library memory and are never host-dereferenceable.
- A view must be reacquired after a step, add/remove operation, or capacity
  change. `revision` makes accidental caching detectable.
- A world is bound to the CUDA device current during `World::create`.
- A world is movable, not copyable, and externally synchronized.

## Stepping and CUDA streams

`step_async` enqueues a complete frame on the caller's CUDA stream and records
a `FrameToken`. The initial release permits one frame in flight per world.
`ready` polls without blocking; `wait` establishes completion and reports
deferred CUDA failures. `step` is the convenience wrapper that enqueues and
waits.

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
runtime emission/removal can be added without changing the resource model.

Initial implementation requirements:

- finite positions and velocities;
- no duplicate neighbor IDs or self-neighbors;
- bounded neighbor overflow reported as an error, never silently truncated;
- identical same-GPU particle/contact ordering for identical input;
- no non-finite state;
- collision projection plus velocity response against every supported rigid
  shape;
- reaction impulses applied to dynamic bodies.

Cross-fluid interaction, phase changes, foam, surface reconstruction, and
particle emission/removal are deferred.

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
collision are deferred.

## Contacts are the application extension point

The optional contact buffer reports fluid-particle/rigid-body contacts in a
stable order. Gallery code can use it for paint, sound, objectives, foam
emission, or debugging without putting those concepts into the physics API.
Overflow is explicit in `ContactDeviceView` and statistics.

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

## Review questions

1. Should `World` remain the only stepping interface, or is independent solver
   advancement a real requirement for the first release?
2. Are host particle spans sufficient initially, or must device-side creation
   be part of v0.1?
3. Should contacts be enabled by capacity as proposed, or omitted until a
   gallery scene needs them?
4. Is one in-flight frame per world acceptable?
5. Are sphere, box, capsule, and plane the right initial rigid shapes?
