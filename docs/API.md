# Physics API: rigid bodies, fluid, cloth, soft bodies, and ropes

## The central decision

`parallel_mater::World` owns every simulated fluid, cloth, soft body, rope, and rigid body and advances
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
tests. Fluid and particle-lifecycle calls are implemented in PR 7, including
passive triangle contacts and balanced dynamic-rigid reactions. Soft bodies use
world-owned volumetric spring lattices with passive/dynamic rigid contacts and
explicit cloth/fluid coupling resources.

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
- `FluidId`, `ClothId`, `SoftBodyId`, `RopeId`, and `RigidBodyId` contain an index and generation. Removing an
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

## Rope centerlines and attachments

`World::add_rope` copies an open world-space polyline and resamples its arc
length through `sample_rope_centerline`. `RopeOptions` specifies total mass,
radius, node spacing (no larger than the diameter), stretch compliance,
velocity damping, contact friction, speed limit, and iteration budget. At most
1,024 nodes are supported per rope. Zero compliance requests an inextensible
chain; the finite-iteration solve still has a measurable tolerance.
The API rejects rest centerlines crossing rigid triangles, apart from the
immediate Hook attachment neighborhoods. This check runs at creation, not per
frame: no amount of stiffness can repair a rope initially threaded through an
unrelated wall. Correct the rest curve or collider before retrying.

Each endpoint may have a `RopeAttachment` to a rigid body with a body-local
anchor. The initial endpoint must match that anchor. Passive/kinematic bodies
drive attachments; dynamic bodies receive tension and contact reactions,
including torque. Alternatively, `RopeSoftBodyCouplingOptions::attach_first`
or `attach_last` binds an endpoint to the closest rest-surface triangle of a
soft body. The attachment follows its skinned triangle and transfers tension
through the surface bindings to physical soft-body nodes. An endpoint cannot
have both attachment types. Attached bodies cannot be removed before their ropes.
The current open-chain solver requires distinct targets when both endpoints
are attached. Unattached endpoints move freely.

Rope stepping belongs to `World::step`, not the gallery. A tridiagonal distance
solve propagates tension along the chain, interleaved with swept node and
segment/triangle contacts, friction, and non-neighbor self-contact. Contact
planes constrain the solve's inverse masses; nearly parallel triangle normals
must not create an indefinite matrix. Releasing a contact rebuilds the matrix
before applying corrections, retaining any second support plane. A final
velocity solve removes axial stretch rates and transfers endpoint impulses
to the attached bodies without damping uniform translation or swing. It uses
the full mass matrix: positional contact projectors cannot safely resolve
arbitrary incoming normal velocities.
`maximum_substep_timestep` defaults to 1/480 s. `World::step` raises the requested
substep count as needed, advancing rigid attachments and all other systems on
the same smaller steps. Worlds without ropes are unchanged. The smallest live
rope limit wins; a request needing more than 1,024 substeps fails before stepping.
Sharp contacts can use up to eight times the nominal `solver_iterations` budget
(capped at 128), stopping recovery below 0.5% segment strain. This avoids feeding
an unresolved contact/stretch correction back as a large velocity on the next
step without paying the recovery cost for already settled chains.
No analytic post collider is used.
Ropes collide with rigid triangle bodies. Fluid contact is optional: register a
`FluidRopeCouplingOptions` pair after creating a fluid and rope. Water is kept
outside each moving rope segment capsule and transmits a bounded reaction to
its free nodes. The acceleration limit controls dense-splash stability; it does
not change fluid/rigid contacts. `fluid_rope_contacts` timing and contact/depth
statistics expose its cost and activity. A `RopeSoftBodyCouplingOptions` pair
adds two-way node and segment contacts against the soft body's current triangle
skin, not its render-only rest mesh or an analytic post. Its independent
contact/force views, penetration statistics, and timing stage expose the
interaction. Different soft bodies can bind opposite rope ends; one rope
supports up to two soft-body contact targets. Cloth and other ropes are not
coupled to ropes.

`rope_view` exposes node positions, velocities, segment rest lengths, and
constraint/contact/fluid-contact/soft-body-contact forces. `RopeId` is generation checked; the usual borrowed
view lifetime applies. `WorldOptions::rope_capacity` bounds resources, and
statistics/timings include rope nodes, storage, and solve time. Opt-in physics
capture includes the fluid reaction in each rope sample. Tube construction and final debug drawing stay
outside the installed physics library.

## Cloth meshes and pinning

`World::add_cloth` copies host positions, triangle indices, and optional
per-vertex inverse masses. Zero inverse mass pins a vertex exactly; omitted
masses default to `1 / vertex_mass`. Triangles create stretch and bending
links, solved by compliant Jacobi projection over each substep. World-owned
cloth positions and triangles are borrowed through `cloth_view` and reacquired
after stepping. `ClothId` is generation checked like other resource handles.
The cloth contact stage resolves vertices against rigid triangle BVHs and
transfers equal-and-opposite impulses to dynamic bodies. A conservative
triangle-side body constraint prevents fast bodies from crossing intact,
nonfracturing sheets. Fracturing cloth instead uses node contacts to let the
body keep its incoming momentum while bonds fail; the conservative
triangle-radius constraint otherwise holds it against the separating faces.
The broad-phase rigid radius can overestimate non-spherical shapes, so a
future exact deforming-mesh narrow phase remains possible without changing
the API.
Optional `break_strain > 0` enables persistent bond fracture: each stretch,
shear, or bending bond has a stable ID, rest length, active state, and damage
counter. A bond breaks after its extension exceeds `break_strain` for
`fracture_persistence_substeps` consecutive substeps. Optional
`impact_break_impulse > 0` also breaks bonds whose endpoint contact impulses
exceed that threshold. Fracture never deletes a triangle. At the next idle
frame boundary the API splits vertex fans across failed shared edges and
removes bending links across those seams. Every triangle retains its own
material edge constraints. Split nodes inherit position, velocity, and pins;
incident-face mass shares preserve the original mass and momentum. Isolated,
unpinned triangles have an additional 10% physical stretch projection during
contact solving. Attached material keeps its authored compliance and break
strain: globally clamping it would erase the strain needed to continue tearing.
For tearable cloth, `cloth_view` exposes triangle-local `surface_positions`,
stable `surface_triangle_indices`, `surface_source_indices` for UV lookup,
and `bonds`/`active_bonds` for diagnostics. Surface corners are exact copies
of their physical vertices, including after tearing; no rest-shape fitting
to disconnected vertices is performed. `triangle_indices` contains current
physical connectivity and may change after a tear. Reacquire `cloth_view`
each frame: its physical vertex count can grow. `vertex_source_indices` maps
physical nodes to authored vertices, and `inverse_masses` exposes their split
masses. `surface_source_indices` remains stable for UV lookup. Storage for
split nodes and links is reserved at creation; graph rebuilding happens only
when bond states change, outside the asynchronous GPU frame.
The gallery renders that API-owned surface; rigid-body response on fracturing
cloth uses physical node contacts, not a triangle-radius barrier. The gallery
does not choose a cut shape.
`ClothDeviceView` also exposes the last-substep per-node
`rigid_contact_forces`, `fluid_contact_forces`, and `soft_body_contact_forces`.
They are ordinary borrowed device spans and are available without enabling
the bounded contact-event
streams, so an application can build force diagnostics without reaching into
solver storage.

`FluidDeviceView::accelerations`, `ClothDeviceView::velocities`, and the
opt-in `RigidBodyDeviceView::applied_forces`/`applied_torques` provide the
remaining live force and motion inputs needed by client visualizers. The
library never draws these spans.

## Volumetric soft bodies

`build_soft_body_geometry` is a host-only, transactional preparation API for a
closed, consistently wound triangle surface. It refines long edges to the
requested lattice spacing, welds rendering seams into physical nodes, fills
the interior with HCP nodes, and connects both sets with springs. Capacity
limits bound generated nodes and triangles. Fully pinned source edges/faces
stay pinned after refinement. `SoftBodyGeometry::surface_sources` supplies
three source indices/weights per refined vertex so callers can interpolate
their own UVs, normals or other attributes without putting rendering in physics.

`World::add_soft_body` copies host nodes, fixed-topology bonds, optional inverse
masses, an indexed render surface, and four-node delta-skinning bindings.
`SoftBodyId` is generation checked, and `soft_body_view` exposes borrowed device
spans for physical nodes, velocities, bonds, the deforming surface, and rigid
contact forces.

`SoftBodyOptions::shape_matching_stiffness` optionally restores the best-fit
rest shape after spring projection. The constraint solves the body's current
center and rotation before applying a bounded correction, so it removes crush
deformation without tethering translation or rolling to the original world
pose. A substep that records dynamic rigid contact suppresses restoration so
the goal cannot project nodes through the active collider; restoration resumes
on the next contact-free substep. Zero disables it; values through one increase
per-substep restoration.

The first solver stage follows the proven lab model: integrate nodes under
gravity, jointly solve node-sphere/passive-triangle contacts and bounded
compliant Jacobi spring constraints, blend projected motion back into velocity,
damp bond-relative velocity, apply Coulomb traction bounded by accumulated
normal constraint work, and update the authored surface. Interleaving a contact
pass after every two graph passes prevents a later spring projection from
stranding a node beyond a wall. Contact restitution comes from the contacted
rigid material rather than penetration depth. Static and dynamic triangle
bodies share this contact pass. A dynamic contact applies equal-and-opposite
linear and angular impulses to the existing rigid-body state; the soft solver
restores the corresponding total lattice momentum after spring projection so
constraints cannot silently erase the exchanged impulse. The passive-only path
retains its original damping behavior. Swept deformable contact transforms the
start and end samples by the corresponding previous and current rigid poses,
so contact queries include relative collider motion.

Mesh upload identifies closed convex solids from welded triangle topology and
supporting face planes. Soft nodes inside those solids recover outward; leaving
a solid is not mistaken for entering the back of a two-sided triangle. Dynamic
contacts also share positional corrections according to inverse mass. Open and
concave meshes retain the two-sided swept triangle path.

After traction, a bounded cleanup solve checks the actual soft surface triangles
against closed convex rigid meshes and transfers corrections through their
four-node bindings. Adjacent triangle corrections are gathered deterministically
per node, combining different contact normals instead of discarding all but the
largest correction. Passive boundaries are resolved last. This final recovery
uses a small geometric skin (1% of node radius plus the rigid collision margin),
because full node-radius safety margins can overlap when heavy colliders squeeze
a soft body.
The normal contact/friction solve continues to use the configured node radius.
Recovery adds no artificial second velocity impulse and does not replace rigid
triangle geometry with bounding spheres. It is a discrete overlap cleanup, not
continuous triangle/triangle collision detection for arbitrary timesteps.

Node radius,
mass/inverse masses, compliance, projection bound/velocity response, global
and spring damping, contact friction, shape-matching stiffness, maximum speed,
and iteration count are API configuration rather than gallery constants.
Fluid coupling uses the same lattice and independently bound triangle surface.

`add_soft_body_cloth_coupling` binds one soft body to one cloth through
`SoftBodyClothCouplingOptions`. Contact uses the cloth's current triangles on
either side, including the separated surface after fracture. Relative previous
positions guard crossings; inverse masses and triangle barycentric weights
distribute contact corrections and opposing normal/friction impulses. Separate
detection and gather kernels avoid concurrent position reads and writes.
Shared contact relaxation applies the same scale to both sides, including
the reported reaction at pinned vertices. Its effective-mass adjustment
accounts for shared cloth vertices without weakening each independent soft
node by the same contact count. Speed caps may limit momentum at extreme
impacts. This is node-to-triangle contact with a crossing guard, not continuous
triangle-to-triangle collision for arbitrary timesteps or deformation.

The API steps both deformables at each substep, interleaves their graph
constraints with contact, and samples cloth fracture once before those graph
projections remove the impact strain. `ClothOptions::break_strain = 0` and
`impact_break_impulse = 0` keep that cloth intact. A positive break strain
enables its existing persistent, triangle-preserving fracture independently
of every other cloth. There is no scene-dependent tear rule.

Coupling options expose contact distance, friction, iteration count, and an
enable switch. Iteration requests range from 1 to 16; the highest enabled
request sets the shared pass count, keeping every pair constrained while
shared bodies continue projecting. Endpoints stay fixed; remove and add a
resource to bind a new pair. Remove couplings before their cloth or soft body;
handles become stale
after removal. Reserve pairs with `WorldOptions::soft_body_cloth_coupling_capacity`.
Device views and opt-in physics captures expose `cloth_contact_forces` on soft
bodies and `soft_body_contact_forces` on cloth. Timings include
`soft_body_cloth_contacts`. The gallery connects each authored soft body to
each authored cloth using this API.

## Fluid sources and contact paint

### Fluid and soft bodies

`World::add_fluid_soft_body_coupling({.fluid = water, .soft_body = body}, id)`
adds external contact with a closed, consistently wound soft surface. Reversed
winding is accepted; open/nonmanifold surfaces are rejected. Reserve pairs with
`WorldOptions::fluid_soft_body_coupling_capacity`. Updates can change contact
distance (zero selects particle radius), friction, 1–16 contact iterations, or
enable state. Endpoints are immutable; remove/re-add to change them. Duplicate
pairs are rejected. Remove the resource before either endpoint, even if disabled.

Contact follows current triangles and merges their barycentric skin influences
onto physical nodes. Each fluid iteration alternates contact and soft graph
projection. A fixed-topology triangle BVH refits swept bounds after deformation;
nearest-face and signed-ray winding queries traverse it. Ambiguous shared-edge
ray hits fall back to solid-angle winding. No coarse collision proxy replaces
the refined skin. Shared-node contact-degree relaxation and a symmetric impulse
bound keep node velocities within the configured soft-body speed ceiling.
Fluid and node impulses are equal and opposite; forces on fixed nodes represent
reactions absorbed by their external support. Separate non-energetic position
recovery handles crossings and embedded particles, then rechecks the skin after
rigid boundaries. This is a finite-step triangle method with a relative crossing
guard, not exact continuous collision detection for arbitrary deformation.

`SoftBodyDeviceView::fluid_contact_forces` and the matching debug sample expose
frame-average contact reactions. `WorldStatistics` exposes contact proposal
count (including repeated solver passes) and maximum **pre-correction**
penetration, not residual overlap. `WorldStepTimings::fluid_soft_body_contacts`
includes detection, reactions, constraint cleanup, and geometric recovery.
Capture remains opt-in through `WorldOptions::physics_debug`; drawing stays
outside the API. Scratch/contact buffers are reserved at coupling creation.

The current broad phase rejects against deformed body bounds, then searches
its triangles. This is appropriate for the authored low-poly obstacle; a
refittable triangle hierarchy remains a future optimization for dense skins.

### Sources and paint

Continuous inflow is configured with `ParticleSourceOptions` and
`World::add_particle_source`; `ParticleDestroyPlaneOptions` supplies the
matching outflow. These are physics-owned sources and sinks, so the gallery
only translates Blender flow-plane metadata into their options.

For a one-time Flow/Geometry volume, `sample_fluid_geometry` expects a closed,
indexed host triangle mesh, transform, velocity, and particle spacing. It
appends an HCP particle lattice to a host vector without requiring a GPU.
`World::add_fluid_geometry` performs that sampling and creates a fluid directly,
deterministically selecting particles if the requested volume exceeds the
configured fluid capacity. Gallery scene loading uses the same public sampler
to combine authored geometry volumes before instantiation.

Contact paint is opt-in. `World::add_paint_field` attaches a persistent,
two-sided UV mask to one rigid-body instance or one cloth. Rigid paint meshes
and UVs may differ from the body's collision proxy; cloth UVs correspond to
its deforming vertices.
`World::add_paint_rule` selects exactly one source: a fluid for a rigid target,
or a rigid body for a cloth target. Fluid rules can set extra reach beyond the
particle radius. Fluid–rigid contacts stamp one texel. Rigid–cloth contacts
fill a world-space disk of UV texels using `brush_radius`, producing continuous
marks as the body moves across deforming triangles. Paint does not depend on
diagnostic contact
collection or render frequency. `paint_field_view` exposes the device mask;
bit 1 is the front side and bit 2 is the back side. `clear_paint_field` resets
the mask. Render color and cubic filtering remain application choices.

Remove rules before their source fluid or rigid body or target field, and
remove fields before their target rigid body, paint mesh, or cloth. Rigid bodies
and deforming cloth exchange contact impulses while paint remains owned by the
world; the gallery only selects color and filtering. Painting does not require
or enable cloth tearing.

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
deterministic triangle-mesh contact resolution. Fluids then run their
configured neighbor-force and contact iterations; cloth and
soft-body stages advance once per rigid substep.

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
broad/narrow-phase fields. Fluid frames additionally measure spawn, cell
sorting, neighbor forces, integration, static triangle contacts, and outflow
compaction. `total_gpu_milliseconds` covers all active solvers in the frame.
Cloth frames also report prediction, link projection, and contact stages.
Soft-body frames report node prediction, spring projection, and passive
triangle contact stages.
`collect_step_timings` reads those events after
frame completion. Timing is diagnostic data rather than solver input and is
unavailable for frames that did not request it.

Every non-empty rigid world uses the same deterministic parallel contact
coloring and resolution path. Small worlds launch no more color rounds than
their possible unordered body pairs; large worlds cap coloring at 32 rounds
and resolve any remaining conflicting contacts serially. This scheduling
choice does not change the triangle-mesh contact API.

Rigid contact diagnostics retain at most `WorldOptions::contact_capacity`
events. Contact solving remains complete when diagnostic storage is capped.

## Opt-in physics capture

Set `WorldOptions::physics_debug.frame_capacity` before `World::create` to
retain a rolling host history. Zero, the default, allocates no capture force
buffers, performs no state readback, and leaves contact collection controlled
solely by `StepOptions`. A nonzero capacity retains rigid state and applied
inputs, fluid position/velocity/solver acceleration/foam, cloth state and both
coupling forces, contact events, gravity, timestep, and the measured peak
fluid-neighbor count. `frame_stride` can reduce capture frequency.

`physics_debug_frame` borrows the newest immutable host frame until the next
completed step. `copy_physics_debug_capture` deep-copies the ring in
chronological order for logging. Enabling capture intentionally also retains
contact events and synchronously assembles a host frame when completion is
acknowledged; consumers should enable it only in tools or diagnostic builds.
File formats and vector drawing remain outside the physics library.

## Fluid contract

The first fluid is a fixed-radius particle fluid with deterministic neighbor
ordering. `FluidOptions` deliberately exposes physical/solver quantities—not
gallery presets or render settings. Capacity is separate from initial count so
emitters can add particles without allocating during a step.

Initial implementation requirements:

- finite positions and velocities;
- no duplicate neighbor IDs or self-neighbors;
- neighbor overflow reported as `capacity_exceeded`, without truncating forces;
- identical same-GPU particle/contact ordering for identical input;
- no non-finite state;
- collision projection plus velocity response against passive triangle meshes;
- reaction impulses on dynamic bodies are deferred to PR 8.

Cross-fluid interaction and phase changes are deferred. Continuous surface
reconstruction stays outside the public API in the example renderer.
`FluidDeviceView::foam` exposes a short-lived impact/surface signal for the
examples-only renderer; it is not a separate foam fluid.

## Mesh sources and destroy planes

`World::add_particle_source(mesh, options, id)` copies a world-space triangle
surface and subdivides/thins it into deterministic, separated emission sites.
Open, tilted, disconnected, and closed surfaces are supported. Sampling belongs
to the API; `sample_fluid_source` also exposes the host-only sampler.
`ParticleSourceMesh::spacing` defaults to the fluid support radius and cannot
be smaller than its particle diameter. Each step, sites query a GPU spatial
index and emit one particle only if no particle of the destination fluid lies
within that spacing. Earlier emissions, including overlapping sources, also
block occupied sites. Faster initial velocity clears sites sooner, increasing
flow naturally; there is no particles-per-second setting or emission backlog.
Checks occur once per `World::step`, so callers should use a fixed step small
enough that water travels less than the site spacing per step.
Stable particle IDs increase monotonically. All source buffers are preallocated.
`update_particle_source` changes velocity/enabled state; changing mesh, spacing,
or destination requires removal and registration. The old rate-based
`ParticleSpawnPlaneOptions`/`add_particle_spawn_plane` API is removed.

A destroy plane removes a particle when its swept path crosses the finite
rectangle in the selected normal direction. Using the swept path avoids
missing a plane when a fast particle moves from one side to the other in one
substep. Compaction is stable, so surviving particles retain deterministic
order and IDs.

Source and destroy capacity is fixed in `WorldOptions`; neither feature may
allocate during stepping. If a fluid is full, emission pauses rather than
overwriting particles, and `spawn_capacity_miss_count` reports how many
vacant sites could not emit. Handles are generation-checked. Destroy planes can
be enabled, moved, updated, and removed without rebuilding the fluid.

## Rigid-body contract

The first release has one rigid representation: a World-owned indexed triangle
mesh. Spheres, boxes, capsules, planes, Suzanne, and arbitrary Blender meshes
are all ordinary triangle data. A body is static, kinematic, or dynamic:

- static bodies never move;
- kinematic bodies follow explicit targets and transfer momentum to fluid
  particles on moving triangle contacts;
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
sleeping are deferred. Fluid particles collide with static, kinematic, and
dynamic triangles. Dynamic impacts exchange equal-and-opposite linear and
angular impulse with the body; the moving-body path uses swept triangle
contacts and a spatial body index.

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

Set `StepOptions::collect_fluid_contacts` to retain one representative
fluid-particle/rigid-body contact per surviving particle per frame. Moving
body contacts take priority over static contacts, and the strongest normal
impulse wins within each class. Events appear in fluid order, then stable
particle order. Collection is disabled by default. These diagnostics can drive
sound, objectives, or debugging, but contact paint does not depend on this
bounded representative stream: it stamps directly from each qualifying
fluid–rigid collision. Overflow is explicit in `ContactDeviceView` and
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
- smoke or a separate foam-particle simulation;
- serialization and network replication;
- CPU fallback or non-CUDA backend.

These omissions are the main defense against another application-shaped API.

## Decisions from the first review

- `World` is the only stepping interface in v0.1.
- Initial fluid data may be device-resident or sampled from host geometry.
- A capacity-bounded contact stream remains available for diagnostic and
  application effects; contact paint uses the collision path directly.
- One frame may be in flight per world.
- Indexed triangles are the only rigid representation; Blender primitives are
  triangulated during export instead of creating parallel collider types.
- Deterministic particle spawn and destroy planes are part of the initial
  resource model.
- The gallery boundary and staged roadmap are approved.
