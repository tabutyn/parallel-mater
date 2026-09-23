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
completion, generation-checked rigid and triangle-mesh handles, forces,
impulses, kinematic targets, device views, GPU integration, and deterministic
triangle-mesh contact. Every rigid body uses indexed triangles; dynamic,
kinematic, static, open, and two-sided meshes share one code path. Continuous
rigid contact is velocity-gated through conservative swept triangle-pair
tests. Fluid and particle-lifecycle declarations currently return
`StatusCode::not_supported` and are implemented in PR 5.

## Minimal use

```cpp
using namespace parallel_mater;

World world;
Status status = World::create({.rigid_body_capacity = 8,
                               .triangle_mesh_capacity = 8}, world);
if (!status) return report(status);

// These spans point to CUDA memory. Geometry is body-local.
TriangleMeshId floor_mesh;
status = world.add_triangle_mesh(floor_vertices, floor_triangle_indices,
                                 floor_mesh);
if (!status) return report(status);

RigidBodyId floor;
status = world.add_rigid_body(
    {.motion = MotionType::static_body,
     .mesh = floor_mesh,
     .friction = 0.6F},
    floor);
if (!status) return report(status);

TriangleMeshId object_mesh;
status = world.add_triangle_mesh(object_vertices, object_triangle_indices,
                                 object_mesh);
if (!status) return report(status);

RigidBodyId object;
status = world.add_rigid_body(
    {.motion = MotionType::dynamic,
     .mesh = object_mesh,
     .initial_state = {.position = {0.0F, 2.0F, 0.0F}},
     .mass = 12.0F},
    object);
if (!status) return report(status);

status = world.step({.timestep = 1.0F / 60.0F, .substeps = 4});
if (!status) return report(status);
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
ticks. Every implemented rigid substep performs integration followed by
deterministic triangle-mesh contact resolution. Fluid ordering will be
documented when that solver is implemented.

`apply_force` and `apply_impulse` queue contributions for the next submitted
frame and consume them exactly once. `set_kinematic_target` replaces the target
for the next frame. `read_rigid_body_state` and `collect_statistics` are
explicit synchronous readbacks; ordinary stepping performs no telemetry
readback. Device views and contacts describe the most recently completed
frame.

Kernel timing is opt-in per `StepOptions`. When requested, CUDA events measure
rigid integration, world bounds, GPU pair filtering, deterministic pair
compaction, leaf-pair generation, triangle contact evaluation, contact solving,
and input clearing. `rigid_contact_generation` remains the sum of the five
broad/narrow-phase fields. `collect_step_timings` reads those events after
frame completion. Timing is diagnostic data rather than solver input and is
unavailable for frames that did not request it.

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

The first release has one rigid representation: a World-owned indexed triangle
mesh. Spheres, boxes, capsules, planes, Suzanne, and arbitrary Blender meshes
are all ordinary triangle data. A body is static, kinematic, or dynamic:

- static bodies never move;
- kinematic bodies follow explicit targets and impart their velocity to fluid;
- dynamic bodies integrate gravity, forces, impulses, damping, and contact
  reactions.

Dynamic mass must be positive. A zero inertia diagonal requests a box-inertia
approximation derived from the mesh's local AABB; callers with known mass
properties can provide an exact diagonal. Triangle meshes are two-sided and
need not be closed, connected, manifold, or consistently wound. Their vertex
and index data is copied from device spans and organized into a deterministic
private BVH. Degenerate triangles are rejected. Motion beyond the collision
shell activates conservative swept triangle-pair testing over linearized
vertex paths; this prevents the tested fast-body tunneling case without making
ordinary resting contacts pay the full cost. Compound bodies, joints, and
sleeping are deferred. Fluid–dynamic-body momentum exchange arrives with the
fluid-coupling milestone.

```cpp
TriangleMeshId terrain_mesh;
status = world.add_triangle_mesh(device_vertices, device_triangle_indices,
                                 terrain_mesh);
if (!status) return report(status);

RigidBodyId terrain;
status = world.add_rigid_body(
    {.motion = MotionType::static_body,
     .mesh = terrain_mesh},
    terrain);
```

The World copies both device spans, so callers may release their upload buffers
after `add_triangle_mesh` returns. A mesh cannot be removed while a body still
references it.

## Contacts are the application extension point

The optional contact buffer reports fluid-particle/rigid-body contacts in a
stable order. These are the physics-side input for painting: stable particle
and body IDs, contact position, normal, and impulse let gallery code update its
own color fields or textures. The physics API does not own paint pixels,
materials, UVs, or textures. The same records can drive sound, objectives,
foam emission, or debugging. Overflow is explicit in `ContactDeviceView` and
statistics.

Rigid–rigid diagnostics are a separate opt-in stream. `RigidContactEvent`
reports the two handles, contact point, normal, penetration, accumulated normal
impulse, and accumulated tangential friction impulse. Requesting the stream in
`StepOptions` makes `rigid_contacts` describe the most recently completed
frame. It exists for visualization and analysis; applications must not feed it
back into the solver.

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
- Indexed triangles are the only rigid representation; Blender primitives are
  triangulated during export instead of creating parallel collider types.
- Deterministic particle spawn and destroy planes are part of the initial
  resource model.
- The gallery boundary and staged roadmap are approved.
