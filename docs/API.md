# Physics API: rigid bodies, constraints, fluid, cloth, soft bodies, and ropes

## The central decision

`parallel_mater::World` owns every simulated fluid, cloth, soft body, rope,
rigid body, and rigid constraint and advances
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
tests. Rigid constraints cover fixed, point, hinge, slider, piston, generic,
generic spring, and motor joints. Fluid and particle-lifecycle calls are implemented in PR 7, including
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
- `FluidId`, `ClothId`, `SoftBodyId`, `RopeId`, `RigidBodyId`, and
  `RigidConstraintId` contain an index and generation. Removing an
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

## Hit-box queries

`World::query_hit_box` synchronously polls the latest completed state using a
non-colliding oriented `HitBox`. A rigid body is returned when its collision
triangles touch or enter the box; this is an exact triangle/box test rather
than a center or broad-phase-bounds check. A fluid particle is returned when
its center is inside the box, including its boundary.

```cpp
HitBoxResult hits;
status = world.query_hit_box(
    {.center = {0.0F, 1.0F, -8.0F},
     .half_extents = {2.0F, 1.0F, 2.0F}},
    hits);
if (!status) return report(status);

const bool reached_goal =
    std::find(hits.rigid_bodies.begin(), hits.rigid_bodies.end(), object) !=
    hits.rigid_bodies.end();
```

`HitBoxResult::rigid_bodies` contains generation-checked handles.
`HitBoxResult::particles` pairs each `FluidId` with its stable particle ID, so
clients can preserve coloring or gameplay state even when the solver reorders
particle storage. Both arrays have deterministic ordering. The query covers
all live rigid bodies and fluids; callers filter for the target handle or
fluid. Invalid boxes leave the previous output unchanged.

## Rigid constraints

`World::add_rigid_constraint` connects two existing rigid bodies through
body-local anchor and orientation frames. At least one body must be dynamic.
`RigidConstraintType` provides Blender-compatible fixed, point, hinge, slider,
piston, generic, generic-spring, and motor behavior. Hinge rotation uses local
Z; slider translation and piston translation/rotation use local X. Generic
limits and springs use explicit X/Y/Z bit masks. Motor targets act on local X
and can drive linear or angular relative velocity.

Constraints have generation-checked add/update/remove lifecycle. Updating can
toggle `enabled`, move the local frames, or change limits, springs, and motor
targets without recreating the resource. A nonzero breaking threshold disables
the joint when one active substep exceeds that accumulated impulse;
`read_rigid_constraint_state` reports enabled/broken state and the latest
active impulse. `disable_collisions` suppresses rigid contact only while the
joint is enabled and intact. Fixed joints also suppress contacts between all
bodies connected through fixed joints with this flag, so overlapping welded
siblings cannot fight their constraints. Other joint types suppress only their
direct pair. External contacts, including ground support, remain enabled.
The component is rebuilt each substep after edits, removal, or breaking.
Referenced bodies cannot be removed first.

Constraint capacity is fixed by `WorldOptions::rigid_constraint_capacity`, and
`WorldStatistics::rigid_constraint_count` reports live resources. The solver
runs inside every rigid substep. A connected group of dynamic, unbreakable
fixed joints with collision suppression is rebuilt as one compound rigid body:
member meshes remain separate collision surfaces, while contacts use the
group's combined mass and inertia and all member transforms share one rigid
motion. Adding, removing, disabling, or breaking a joint changes the grouping
on the next substep. Groups touching a breakable, articulated, kinematic, or
static joint keep the general constraint solver. On that path, fixed-member
contact impulses and joint rows iterate together through eight contact sweeps,
each using the authored joint iteration budget. Motor impulse limits remain per
substep across those sweeps. Applications still call only `World::step`.

## Smoke tracer gas

`World::add_smoke` creates smoke tracers, separate from liquid. The optional
Eulerian air field (`grid_resolution > 0`) stores pressure, deposited smoke
density and temperature at cell centers and velocity on staggered MAC faces.
X and Z use `grid_resolution` cells; Y is up and uses
`grid_vertical_resolution` cells. The gallery uses 128×32×128, or 524,288
cells. A zero `grid_edge_length` sizes a shallow, uniform-cell domain around
the emitter's horizontal travel; explicit `grid_minimum` and
`grid_edge_length` override it. Each step deposits live tracers with a
quadratic B-spline, rasterizes coupled triangles into thin cut-face apertures
and moving-wall velocities, RK2/MacCormack-advects momentum, applies buoyancy,
physical and Smagorinsky LES viscosity, and restrained vorticity confinement,
then projects with a four-level geometric multigrid solve. Wind is an inlet
and far-field condition; downstream faces are open outflow rather than a
whole-domain relaxation. Tracers follow the projected field with RK2. Swept
triangle contact remains only as a containment safeguard and does not apply a
second rigid impulse. No sphere-specific flow rule or prescribed wake exists.

`grid_pressure_iterations` is a fine-grid-equivalent maximum work budget;
`grid_pressure_tolerance` (default `1e-3`) lets scheduled multigrid work stop
on the GPU when the infinity-norm relative residual converges.
`grid_kinematic_viscosity` defaults to `1.5e-5 m^2/s`, and
`grid_les_coefficient` defaults to `0.12`. `vorticity_confinement` is a bounded
correction for curl lost to grid transport, not a wake generator.

`grid_resolution = 0` retains the older particle-only solver. It uses a
sorted spatial grid to estimate local number density, particle pressure,
viscosity, and measured vorticity. The default rest number density is 12,
pressure stiffness 2, vorticity confinement 0.1, and ambient-flow response
0.5/s. Grid mode skips particle sorting, pair pressure, pair viscosity, and
particle-vorticity force kernels; those diagnostic particle fields sample the
projected grid instead.
`SmokeDeviceView` exposes positions, velocities, ages, number densities,
pressures, measured vorticity vectors, cell-centered reconstructed grid
velocity, grid pressure, density, density-weighted thermal loading, vorticity,
divergence, dimensions, and the final relative pressure residual. Divide
`grid_temperature` by nonzero `grid_density` to recover the local mean thermal
acceleration. The cell-centered velocity remains a
compatibility/debug view; the solver owns the face velocities. A slot whose
age reaches `lifetime` is ignored until
reused by emission. `remove_smoke` invalidates its generation-tagged handle.
`WorldStepTimings` reports the air grid, tracer advection, and emission separately, while
`WorldStatistics` reports occupied slots and total emitted smoke particles.
The bounded GPU ring recycles expired particle slots.
Migration from the prescribed-flow API: remove `SmokeOptions::obstacle` and
`wake_strength`; register `SmokeRigidCouplingOptions` for each mesh that should
interact with smoke. `response` only controls particle-only relaxation toward
ambient wind; grid mode does not relax the domain toward it.

`World::add_fluid_smoke_coupling` links existing `FluidId` and `SmokeId`
resources to a finite `ParticlePlane` heater. `FluidParticle::temperature`
and `ParticleSourceOptions::initial_temperature` are Celsius (20°C by
default); `FluidDeviceView::temperatures` exposes the live values. Near the
heater, water temperature approaches `heater_temperature` at the configured
`heat_transfer_rate`. At `boiling_temperature` (100°C by default) a water
particle is removed from the liquid solver and inserted into bounded smoke
storage with velocity and decaying thermal lift opposite gravity. This is a phase
transfer, not a second copy of the water particle. Grid mode samples local
deposited density and projected air velocity for configurable water drag;
particle-only mode uses the sorted smoke neighbor grid.
`WorldStatistics::boiled_particle_count` tracks transfers; source
temperature survives fluid compaction. Remove the coupling before removing
either system. This first thermal model has no latent heat, condensation, or
two-way gas momentum solve.

`World::add_smoke_soft_body_coupling` links any existing smoke and soft-body
resources. In grid mode, surface triangles sample air velocity, density, and
pressure on both sides and distribute bounded aerodynamic force through their
node bindings; this reaches the exterior even when interior nodes lie inside
solid grid cells. In particle-only mode, nearby tracer velocity bends movable
nodes. Exact Goal pins remain fixed. Smoke tracers collide with
the current skinned soft-body surface samples, using a broad-phase bound so
posts outside the plume are cheap to skip. Contact follows the moving skin
and its node velocity. This deformable coupling is still one-way: its contact
does not return equal-and-opposite impulses to smoke particles. The
generation-checked coupling must be removed before either resource. The
gallery registers the same API coupling for every soft body in a smoke scene;
no scene-specific physics kernel is involved.

`World::add_smoke_cloth_coupling` links a smoke system to any cloth, including
an open or tearing sheet. In grid mode, pressure and tangential surface stress
on its live triangles add bounded acceleration to movable cloth vertices,
leaving authored pins exact. Particle-only mode retains local tracer drag. Smoke tracers make
two-sided swept contact with the cloth's current triangles; the contact normal
uses the tracer's incoming side so a thin sheet does not flip particles through
it. Triangle barycentric weights transfer the local cloth velocity to the
tracer response. As with smoke/soft body, this deformable coupling remains
one-way. The coupling has a
generation-tagged handle and must be removed before the smoke or cloth.
Impact pressure and existing tangential velocity carry smoke around the
finite sheet's edges. The boundary converts measured impact pressure into
bounded tangential motion toward an open edge; it has no fixed edge speed.

`World::add_smoke_rope_coupling` applies bounded local-particle drag to free rope
nodes before the shared rope solve. Anchored endpoints stay governed by their
rigid or deformable attachment. Smoke tracers use swept capsule contact with
the rope's live segments; this deformable coupling does not yet return
reaction impulses to particles. The
generation-checked coupling must be removed before its smoke or rope.

`World::add_smoke_rigid_coupling` works with closed bodies and open panels.
In grid mode, pressure and tangential stress are integrated over the body's
actual triangles, including torque about its center of mass. Deposited smoke
density scales the traction, so an unreached body receives no smoke force and
a denser local plume produces more force. The triangle raster uses rigid wall
velocity `v + omega × r`. Particle-only mode retains equal-and-opposite local
contact impulses whose mass derives from `air_density` and particle size.
Set `tracer_contact=false` only when containment is intentionally disabled or
handled elsewhere.
The coupling must be removed before its smoke or rigid
body. The gallery couples all dynamic rigid bodies automatically; the Blender
`pm_smoke_collider` property additionally opts in a static or kinematic mesh.

Smoke buoyancy, including heated steam, points opposite the current step
gravity; zero gravity retains world-up buoyancy. Soft-body, cloth, and rope
wind use one bounded local-flow response calculation. Soft-body couplings expose
`maximum_wind_acceleration` to keep stronger smoke from injecting an
unbounded node velocity change.

## Rope centerlines and attachments

`World::add_rope` copies an open world-space polyline and resamples its arc
length through `sample_rope_centerline`. `RopeOptions` specifies total mass,
radius, node spacing (no larger than the diameter), stretch compliance,
velocity damping, contact friction, speed limit, and iteration budget. At most
1,024 nodes are supported per rope. Zero compliance requests an inextensible
chain; the finite-iteration solve still has a measurable tolerance.
The API rejects rest centerlines crossing rigid triangles, apart from the
short attachment path that starts inside its own rigid collider. This check runs at creation, not per
frame: no amount of stiffness can repair a rope initially threaded through an
unrelated wall. Correct the rest curve or collider before retrying.

Each endpoint may have a `RopeAttachment` to a rigid body with a body-local
anchor. The initial endpoint must match that anchor. Passive/kinematic bodies
drive attachments; dynamic bodies receive tension and contact reactions,
including torque. Alternatively, `RopeSoftBodyCouplingOptions::attach_first`
or `attach_last` binds an endpoint to the closest rest-surface triangle of a
soft body. The attachment follows its skinned triangle and transfers tension
through the surface bindings to physical soft-body nodes. An endpoint cannot
have both attachment types. `RopeClothCouplingOptions` instead binds an endpoint
to a cloth vertex by index. That vertex participates in the rope's distance
solve with an effective supported-patch mass, and its position/velocity receive
the same bounded correction. Rope tension appears in
`ClothDeviceView::rope_contact_forces`. The coupling has add/update/remove
lifecycle and prevents removal of its rope or cloth while active. Attached
bodies cannot be removed before their ropes.
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
Sharp contacts can use up to four times the nominal `solver_iterations` budget
(capped at 32), stopping recovery below 0.5% segment strain. This avoids feeding
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
interaction. Attached endpoints distribute load over nearby lattice nodes.
`anchor_support_radius_scale` sets the free-rope radius; as soft-skin contacts
share a wrap's load, it fades toward `anchor_contact_support_radius_scale`.
Both are multiples of soft-body node radius; set both to zero for a point
attachment. Different soft bodies can bind opposite rope ends; one rope
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
transfers equal-and-opposite impulses to dynamic bodies. A second constraint
prevents fast bodies from crossing intact, nonfracturing sheets, but treats the
rigid body's bounding sphere as its contact shape against cloth triangles.
This can overestimate non-spherical bodies. Fracturing cloth instead uses node
contacts to let the body keep its incoming momentum while bonds fail; the
bounding-sphere constraint otherwise holds it against the separating faces.
An exact rigid-mesh/cloth-triangle narrow phase remains future work.
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
frame and consume them exactly once. `apply_central_acceleration` batches one
mass-independent acceleration across selected dynamic bodies without reading
their states. `set_kinematic_target` replaces the target for the next frame.
`read_rigid_body_state` and `collect_statistics` are explicit synchronous
readbacks; ordinary stepping performs no telemetry readback. Device views and
contacts describe the most recently completed frame.
`RigidBodyDeviceView::previous_states` exposes the state at the start of that
physics tick so fixed-timestep render loops can interpolate toward `states`.

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
- equal-and-opposite reaction impulses on dynamic triangle bodies.

Other cross-fluid interactions remain deferred. Continuous surface
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
contacts and a spatial body index. Simultaneous particle contacts use a
mass-ratio-weighted batched effective body mass before equal-and-opposite
impulses are applied. This prevents a light rigid body from receiving one
full body reaction per particle without weakening heavy-body contacts.

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

`RigidBodyOptions::collision_margin` is primarily a contact-search distance.
The solver uses at most `0.001 m` of the combined margin as a rest offset, then
uses the remaining margin only for speculative detection. This small gap keeps
triangle surfaces from numerically crossing without making bodies float by the
full authored margin. Small closed convex meshes additionally use clipped face
patches: their normals remain defined at zero separation, so coplanar authored
faces require no artificial gap. These convex contact impulses accumulate
across solver passes and warm-start from matching contact points in the previous
substep. Body edits, stale handles, and timestep changes invalidate that history.
Curved/concave meshes retain their triangle contact path. Hinge and fixed-member
contact impulses retain their established response rules.
Friction remains active within the numerical skin only while normal support
exists, and cached friction is removed when contact opens.
For eligible convex bodies and fixed members, intersection recovery depth comes
from the body vertices behind the contacted triangle plane rather than from the
search margin. A body in an enabled fixed constraint recovers through contact velocity.
Recovery uses the measured penetration, and contact and joint rows iterate
together so light supports receive the heavy cluster's load before the next
integration step. This avoids separating a member from its joint or letting a
later joint solve undo its ground-support response.
Free groups joined exclusively by fixed constraints also recover residual
overlap against static or kinematic surfaces with a shared translation. Every
member moves by the same amount, preserving joint offsets without changing
velocities. Groups containing an immovable body or another joint type retain
the velocity solve rather than translating away from an external anchor.

## Errors and validation

Public calls are `noexcept` and return `Status`. Invalid values and stale
handles fail before work is submitted. `Status::message` points to
library-owned static text; callers do not free it. A CUDA failure carries its
original `cudaError_t`.

## Deliberately absent

- renderer, camera, lights, materials, meshes, textures, or OptiX objects;
- gallery recipes, level order, victory conditions, input bindings, or UI;
- public hierarchy, neighbor, scratch-allocation, or constraint-batch types;
- a separate foam-particle simulation;
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
