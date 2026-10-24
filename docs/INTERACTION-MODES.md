# Interaction pair and mode contract

This document is the semantic source for per-pair capability discovery. It
defines what an interaction claim means; it does not advertise a driver. A
driver may report a mode only when that exact pair, mode, direction and material
profile is implemented and qualified. The broad API 0.2 `domainCouplings` bit
does not establish any row in this matrix.

The six families are fluid (called `LIQUID` by the API 0.2 symbols), cloth,
soft body, rope, smoke and rigid body. Pair order is canonical in this document;
direction arrows define behavior and do not depend on `first`/`second` storage
order.

## Mode vocabulary

| Mode key | Required observable behavior |
|---|---|
| `contact` | Prevent overlap at a declared distance and define normal response plus any friction. Contact alone does not claim closed-volume containment. |
| `attachment` | Maintain a persistent positional or kinematic relationship until it is removed or broken. |
| `drag` | Exchange velocity-dependent force without requiring contact. Contact friction is not a drag capability. |
| `containment` | Keep one family on the declared side of a closed or consistently oriented boundary, with a measurable leakage/penetration limit. |
| `pressure` | Exchange pressure forces across a system boundary. A penalty contact stiffness is not a pressure capability. |
| `viscosity` | Exchange dissipative shear forces across a system boundary. Contact friction is not a viscosity capability. |
| `thermal` | Exchange heat while retaining both endpoint families and account for supplied, removed and transferred energy. |
| `phase` | Transfer mass or membership between families or material phases while defining identity, capacity, momentum and energy handling. |
| `interface` | Preserve a material boundary and exchange capillary traction. Support is queried independently from contact, pressure and thermal exchange. |

`contact` parameters may include friction. Pressure and viscosity remain
separate modes because a vendor can implement collision without a continuum
interface. A phase mode can depend on thermal input, but support for one never
implies support for the other.

## Status and direction

| Mark | Meaning |
|---|---|
| `R` | A reference implementation path exists in API 0.2. This is an implementation inventory, not qualification of every driver or device. |
| `P` | Proposed by the roadmap; API 0.2 must report it unsupported until its implementation and qualification tasks close. |
| `U` | Undefined/unsupported. An explicit request must fail rather than silently selecting another mode. |
| `A → B` | A affects B; A receives no reaction from this operator. |
| `A ↔ B` | Both endpoints participate. Mechanical exchange must apply equal and opposite impulse within the mode's tolerance unless an identified external source or sink is present. |
| `A ⇒ B` | A phase/membership transfer consumes from A and creates in B; the material profile defines conserved quantities and stable-ID behavior. |

A static or kinematic endpoint can be immovable while the mode remains
two-way: the reaction is still observable through force, impulse or constraint
output. Two one-way modes in opposite directions are not automatically one
two-way mode; they can use different sampling and conservation rules.

## Distinct-family matrix

The table is exhaustive: six families produce 15 unordered distinct-family
pairs. Text after `R` states the fixed API 0.2 direction. Proposed behavior is
shown only where an existing roadmap task defines it.

| Pair | Contact | Attachment | Drag | Containment | Pressure | Viscosity | Thermal | Phase |
|---|---|---|---|---|---|---|---|---|
| Fluid–cloth | `R fluid ↔ cloth` | `U` | `U` | `R cloth ↔ fluid` | `U` | `U` | `U` | `U` |
| Fluid–soft body | `R fluid ↔ soft` | `U` | `U` | `U` | `U` | `U` | `U` | `U` |
| Fluid–rope | `R fluid ↔ rope` | `U` | `U` | `U` | `U` | `U` | `U` | `U` |
| Fluid–smoke | `U` | `U` | `R smoke → fluid` | `U` | `U` | `U` | `R heater → fluid` | `R fluid ⇒ smoke` |
| Fluid–rigid | `R fluid ↔ rigid` | `U` | `U` | `R rigid ↔ fluid` | `U` | `U` | `U` | `U` |
| Cloth–soft body | `R cloth ↔ soft` | `U` | `U` | `U` | `U` | `U` | `U` | `U` |
| Cloth–rope | `U` | `R cloth ↔ rope` | `U` | `U` | `U` | `U` | `U` | `U` |
| Cloth–smoke | `R cloth → smoke` | `U` | `R smoke → cloth` | `U` | `U` | `U` | `U` | `U` |
| Cloth–rigid | `R cloth ↔ rigid` | `U` | `U` | `U` | `U` | `U` | `U` | `U` |
| Soft body–rope | `R soft ↔ rope` | `R soft ↔ rope` | `U` | `U` | `U` | `U` | `U` | `U` |
| Soft body–smoke | `R soft → smoke` | `U` | `R smoke → soft` | `U` | `U` | `U` | `U` | `U` |
| Soft body–rigid | `R soft ↔ rigid` | `U` | `U` | `U` | `U` | `U` | `U` | `U` |
| Rope–smoke | `R rope → smoke` | `U` | `R smoke → rope` | `U` | `U` | `U` | `U` | `U` |
| Rope–rigid | `R rope ↔ rigid` | `R rope ↔ rigid` | `U` | `U` | `U` | `U` | `U` | `U` |
| Smoke–rigid | `R smoke ↔ rigid` | `U` | `R smoke ↔ rigid` | `U` | `U` | `U` | `U` | `U` |

Fluid–cloth containment is an oriented-surface operator. It does not claim
that any open or inconsistently wound cloth is a closed container. Fluid–rigid
containment similarly requires closed collision geometry. The exact leakage,
penetration and reaction tolerances remain qualification properties.

The smoke/deformable rows contain two independently directed operators: the
deformable surface redirects smoke tracers, while smoke flow loads the
deformable. A driver must report them separately even if it schedules both.
Fluid–smoke heating currently comes from the coupling's external heater rather
than heat stored by the smoke endpoint. Its drag is one-way, and its phase path
consumes fluid particles and emits smoke particles. Those paths cannot be
advertised as bidirectional thermal equilibrium or conservative drag.

API 0.2 has enum edges for the particle-family pairs and smoke/rigid. Its
fluid/rigid enum selects parameters/capability while the reference contact path
remains intrinsic even with an empty graph. Cloth/rigid and soft-body/rigid
contact, rigid-body contact/constraints, and rope/rigid contact/endpoint
attachments are also intrinsic rather than graph edges. Intrinsic scheduling
does not exempt a mode from future per-pair capability reporting.

## Same-family and self interaction

"Same family" has three distinct scopes: self interaction within one object or
population, interaction between objects in one domain, and interaction between
separately owned systems. Capability discovery must identify the scope.

| Family | Current reference scope | Separate-system roadmap scope |
|---|---|---|
| Fluid | `R` contact, pressure and viscosity within one population. | `P fluid A ↔ fluid B` contact, pressure and viscosity for equal material on qualified drivers; miscible/immiscible requests have typed state but remain unadvertised, while thermal and phase profiles remain later work. |
| Cloth | No advertised self or cloth–cloth contact mode. | `P cloth A ↔ cloth B` contact/friction; self-contact and distinct sheets are specified by F2-12. |
| Soft body | No advertised self or soft-body/soft-body surface-contact mode. | `P soft A ↔ soft B` surface contact; self and distinct volumes are specified by F2-13. |
| Rope | `R` sampled-node self-exclusion within one rope when `selfCollision` is true; no segment-continuity or self-friction claim. `P` continuous capsule-segment self-contact/friction. | `P rope A ↔ rope B` capsule-segment contact and endpoint attachment; specified by F2-14. |
| Smoke | `R` pressure, viscosity and transport within one particle population or projected grid. | `P smoke A ↔ smoke B` overlapping-field exchange; pressure, viscosity and thermal semantics are specified by F2-15. |
| Rigid body | `R` body–body contact/friction and persistent constraints between bodies in one simulation. | There is no separate rigid-system resource in API 0.2; a future one must not inherit the intra-simulation claim implicitly. |

The proposed marks reserve semantic work, not support. Until their leaf tasks
finish, the corresponding query result is unsupported and edge creation must
fail without mutating the usable graph.

## Cloth contact scopes

Cloth contact has two distinct future capability identities. Self-contact uses
`(cloth, cloth, self, contact, bidirectional, endpoint-native)` and is intrinsic
to one live cloth slot; it consumes no graph edge. Contact between two live
cloth slots uses `(cloth, cloth, separate-system, contact, bidirectional,
endpoint-native)` and a canonical graph edge with `first < second`. API 0.2
exposes both exact identities, the typed self-contact control and the canonical
cloth pair edge, but advertises neither capability. Current drivers omit both
records and reject enabled controls and edge requests with
`PM_ERROR_FEATURE_NOT_PRESENT` without mutating live state or the graph. The
`same-domain` scope remains unsupported because the API has no aggregate cloth
domain distinct from its separately owned slots.

Both scopes treat cloth triangles as double-sided swept surfaces. Let `t_a` and
`t_b` be the finite positive collision thicknesses owned by the two endpoint
materials. A valid primitive pair has target separation

```text
d_ab = t_a + t_b
```

and self-contact uses `d_aa = 2 * t_a`. A conforming implementation detects
vertex-triangle and edge-edge contact over the complete substep from previous
to predicted positions; a discrete end-position overlap test alone is
insufficient. Contact may use any bounded broad phase, but it must not miss a
candidate because the surfaces reside in different storage allocations.

Self-contact excludes topological neighbors that would fight the sheet's own
stretch and bending constraints. A vertex-triangle candidate is excluded when
the vertex belongs to the triangle or has an active structural bond to one of
its vertices. An edge-edge candidate is excluded when the edges share a vertex
or their triangles share an active edge. The exclusion uses the current active
topology after tearing; retired triangles and inactive bonds cannot create
ghost exclusions or contacts. Distinct cloth systems have no topology-based
exclusion, even when their numeric vertex indices match.

### Normal response and ownership

At the end of a successful substep, every nonexcluded detected pair must have
separation at least `d_ab - epsilon`, where

```text
epsilon = max(1e-5 m, 0.05 * d_ab).
```

A high-speed qualification case moves at least `4 * d_ab` in one substep and
must not tunnel through the other sheet. Corrections and impulses are
distributed to triangle vertices by contact barycentrics and inverse mass.
Zero-inverse-mass vertices remain fixed but still form collision geometry; the
reaction must be included in contact diagnostics so a kinematic sink is not
reported as conserved dynamic momentum.

With no fixed vertices or external impulse, contact applies equal and opposite
impulse. Let `delta_v_i` be each vertex velocity change, `m_i` be
`1 / inverseMass_i` for each dynamic input vertex,
`I = sum(m_i * ||delta_v_i||2)`, and
`R = ||sum(m_i * delta_v_i)||2`. A fixture with `I >= 1e-4 kg*m/s` must satisfy

```text
R <= max(1e-7 kg*m/s, 5e-5 * I).
```

The analogous angular-momentum residual about the fixture centre uses each
vertex's `r_i cross (m_i * delta_v_i)` and the same absolute/relative limit in
kg*m^2/s. Endpoint order and triangle winding cannot change the double-sided
contact result beyond these tolerances.

### Friction and unsupported behavior

Friction is a parameter of `contact`, not a separate capability. For distinct
materials the Coulomb coefficient is

```text
mu_ab = sqrt(contactFriction_a * contactFriction_b).
```

For self-contact it is the material's `contactFriction`. Coefficients are
finite and nonnegative. Tangential impulse obeys `||J_t|| <= mu_ab * J_n` and,
without external work, cannot increase relative tangential kinetic energy.
Zero friction preserves tangential velocity within numerical tolerance. Static
and dynamic friction are not separate claims in this initial profile.

Contact does not imply attachment, sewing, adhesion, welding, cutting,
one-sided collision, pressure containment, aerodynamic drag, thermal exchange
or material transfer. It does not make one cloth inherit the other's rigid,
fluid, smoke, paint or attachment edges. A torn cloth retains the same system
identity and self-contact setting; newly separated active primitives become
eligible according to the topology rule above.

Candidate, contact and diagnostic storage must be part of the queried memory
plan. Overflow returns `PM_ERROR_CAPACITY_EXCEEDED`, invalidates the affected
step output and preserves graph/configuration usability for a later retry; it
must not drop a subset of contacts. Empty live cloths are no-ops. Removing an
inter-system edge isolates the sheets on the next step, and endpoint
replacement observes the normal generation and rollback rules.

`PmClothStateInfo` chains `PmClothSelfContactState` to opt one cloth into
intrinsic contact; `PM_COUPLING_CLOTH_CLOTH` represents a canonical
`first < second` graph edge through `PmCouplingStateInfo`. Enabled self-contact
validates finite vertices, nonnegative inverse masses and distinct, in-range,
nondegenerate, nonduplicated triangles before feature availability. Invalid,
duplicate or reversed requests cannot replace usable state. Separate leaves
add bounded broad-phase/narrow-phase data, shared Slang contact/friction
operators, portable scheduling and public-API qualification. Advertisement is
per scope and driver after folded-self, two-sheet, high-speed, friction,
momentum, topology-change, capacity and lifecycle cases pass.

## Soft-body surface-contact scopes

Soft-body contact likewise has two future identities. Intrinsic self-contact
uses `(soft body, soft body, self, contact, bidirectional, endpoint-native)`
within one live soft-body slot. Contact between two separately owned soft-body
slots uses `(soft body, soft body, separate-system, contact, bidirectional,
endpoint-native)` and a canonical graph edge with `first < second`. API 0.2
exposes both exact identities, the typed self-contact control and the canonical
soft-body pair edge, but advertises neither capability. Drivers omit both
records and reject enabled controls and edge requests with
`PM_ERROR_FEATURE_NOT_PRESENT` without changing the live graph. There is no
aggregate soft-body domain, so `same-domain` remains unsupported.

These modes operate on each endpoint's current bound surface, not on every
pair of interior nodes. Surface vertex positions and velocities are evaluated
from `PmSoftBodySurfaceBinding`; contact corrections and impulses return to the
bound nodes with the same normalized weights and their actual input inverse
masses. Numeric surface, node and stable indices are local to an endpoint and
cannot alias the other endpoint. The contact mode requires a finite,
consistently wound, closed two-manifold surface. Open or non-manifold input may
still render, but is an unsupported material/geometry combination for this
capability and must fail before replacing a usable graph or self-contact state.
Every nonzero binding weight is finite and nonnegative, references an in-range
node and the four weights sum to one within `1e-5`; repeated node entries are
combined before contact mass is evaluated. Surface triangles use three
distinct in-range vertices and have nonzero rest area.

Let `r_a` and `r_b` be the positive `nodeRadius` values of the endpoint
materials. Surface contact uses the symmetric target separation

```text
d_ab = r_a + r_b
```

and self-contact uses `d_aa = 2 * r_a`. Candidate generation covers swept
vertex-triangle and edge-edge pairs from previous to predicted surface
positions over the whole substep. It must detect one closed surface entering
another even when no end-position vertex lies inside. Shape matching, spring
projection and contact are interleaved or followed by a final cleanup so a
later internal constraint pass cannot invalidate the reported contact result.

Self-contact excludes local surface topology. A vertex-triangle pair is
excluded when the vertex belongs to the triangle or any node with nonzero
binding weight for that vertex is also bound to a triangle corner or connected
to one by an active material bond. Edge-edge pairs sharing a surface vertex,
surface edge or bound node are excluded. All other nonadjacent surface pairs
remain eligible. Distinct systems have no topology exclusion even when local
indices or binding-node numbers match.

### Surface response and invariants

At successful substep completion, every detected nonexcluded pair has
separation at least `d_ab - epsilon`, with

```text
epsilon = max(1e-5 m, 0.05 * d_ab).
```

A swept qualification case moves one surface at least `4 * d_ab` in a substep
and must not tunnel. Corrections use contact barycentrics, binding weights and
inverse mass. Nodes with zero inverse mass do not move; their reaction is an
external/kinematic impulse and must be reported separately from conserved
dynamic momentum.

With no fixed nodes or external impulse, the total node impulse and torque from
self or inter-system contact are equal and opposite. For dynamic node mass
`m_i = 1 / inverseMass_i`, velocity change `delta_v_i`,
`I = sum(m_i * ||delta_v_i||2)` and
`R = ||sum(m_i * delta_v_i)||2`, a fixture with
`I >= 1e-4 kg*m/s` satisfies

```text
R <= max(1e-7 kg*m/s, 5e-5 * I).
```

Angular impulse about the fixture centre obeys the same relative tolerance and
an absolute `1e-7 kg*m^2/s` floor. Reversing graph endpoints cannot change the
result beyond those limits. Surface binding resolution may change convergence,
but cannot multiply physical mass or impulse because several surface vertices
share one node.

Friction remains part of contact. Self-contact uses the endpoint
`contactFriction`; distinct materials use
`mu_ab = sqrt(contactFriction_a * contactFriction_b)`. The Coulomb bound is
`||J_t|| <= mu_ab * J_n`. Without external work, friction cannot increase
relative tangential kinetic energy; zero friction preserves tangential motion
within tolerance.

This capability does not claim attachment, adhesion, merging, fracture,
cutting, plasticity, volume preservation or containment by an open skin.
`shapeMatchingStiffness`, bonds and material damping keep their existing
ownership. Contact with A does not give B any of A's rigid, cloth, rope, fluid
or smoke edges. Self-contact does not replace inversion prevention or certify
that every internal tetrahedron retains orientation.

Candidate, contact, gather and diagnostic arrays are planned and bounded.
Overflow returns `PM_ERROR_CAPACITY_EXCEEDED`, invalidates the affected output
and permits retry without rebuilding otherwise valid state; silent contact
dropping is forbidden. Empty live bodies are no-ops. Graph removal isolates
distinct bodies on the next step, while replacement/reset follow normal
generation and rollback rules.

`PmSoftBodyStateInfo` chains `PmSoftBodySelfContactState` to opt one body into
intrinsic contact; `PM_COUPLING_SOFT_BODY_SOFT_BODY` represents a canonical
`first < second` graph edge through `PmCouplingStateInfo`. The control validates
the closed bound-surface topology before feature availability, and invalid,
duplicate or reversed requests cannot replace usable state. Separate leaves add
bounded surface candidate generation, shared Slang self and inter-system
operators, portable scheduling and public-API qualification. Advertisement is
per scope and driver after closed-surface, folded-self, two-volume, high-speed,
friction, conservation, binding-resolution, capacity and lifecycle cases pass.

## Rope self, segment-contact and attachment scopes

API 0.2 has one narrow intrinsic rope capability. When `selfCollision` is
true, nonadjacent sampled nodes in one rope are separated as spheres of the
material `radius`; node pairs whose indices differ by at most two are excluded.
This is `(rope, rope, self, contact, bidirectional, endpoint-native)`. It does
not claim continuous segment collision, protection from two segments crossing
between nodes, high-speed continuous detection or self-friction. The current
capability record must be qualified against this exact sampled-node behavior or
withdrawn; it cannot be cited as evidence for the stronger profile below.
CUDA, D3D12 and Vulkan advertise this record after passing the expanded
`rope-self-contact-v2` audit. Metal retains general rope support but does not
advertise this record until equivalent current device evidence exists.

Complete rope contact uses the future `rope-capsule` material profile. Each
centreline edge from node `i` to `i + 1` is a swept capsule with the endpoint
material radius. Intrinsic self-contact has identity
`(rope, rope, self, contact, bidirectional, rope-capsule)` and consumes no graph
edge. Contact between different live rope slots uses
`(rope, rope, separate-system, contact, bidirectional, rope-capsule)` and a
canonical edge with `first < second`. API 0.2 advertises neither capsule
identity and has no rope-rope edge; requests must fail with
`PM_ERROR_FEATURE_NOT_PRESENT` without mutating the graph. `same-domain`
remains unsupported because each rope slot is independently owned.

For segments with radii `r_a` and `r_b`, the target centreline separation is

```text
d_ab = r_a + r_b.
```

Self-contact excludes the same segment, segments sharing a node and segment
pairs separated by only one intervening centreline edge. Every other pair is
eligible, including geometrically close portions with distant node indices.
Distinct ropes have no index-based exclusion. Candidate generation finds the
closest points on both swept segments over the full substep, including an
interior-interior crossing where all four endpoint spheres remain separated.
A high-speed case moving at least `4 * d_ab` per substep must not tunnel.

Contact distributes correction and impulse with the closest-point barycentric
weights to the four segment nodes. A free rope's node mass is its material
`mass` divided by the runtime sampled-node count. A rigid- or soft-bound rope
endpoint uses its resolved effective inverse mass; a fixed endpoint does not
move and its reaction is an external/kinematic impulse. Stretch projection and
contact must be interleaved or followed by cleanup so the reported final
separation is at least

```text
d_ab - max(1e-5 m, 0.05 * d_ab).
```

With no fixed endpoints or external impulse, normal contact conserves total
linear and angular momentum. Using all participating node masses,
`I = sum(m_i * ||delta_v_i||2)` and
`R = ||sum(m_i * delta_v_i)||2`, a fixture with
`I >= 1e-4 kg*m/s` satisfies

```text
R <= max(1e-7 kg*m/s, 5e-5 * I).
```

Angular impulse about the fixture centre uses the same relative tolerance and
an absolute `1e-7 kg*m^2/s` floor. Reversing graph endpoints cannot change the
result beyond these limits. Contact friction uses the endpoint material value
for self-contact and `sqrt(friction_a * friction_b)` between ropes. It obeys
`||J_t|| <= mu * J_n`, does not increase relative tangential kinetic energy,
and becomes tangentially neutral when `mu = 0`.

### Rope-rope endpoint attachment

Attachment is independent from contact. The initial rope-rope attachment joins
one selected endpoint (`first` or `last`) of each rope as a point constraint;
arbitrary interior knots, sliding rings, splices and orientation constraints
are unsupported. The future typed edge state contains the two endpoint
selectors, nonnegative compliance in m/N, nonnegative damping in N*s/m, a
nonnegative capture distance in m and a nonnegative breaking impulse threshold
in N*s. Zero compliance is rigid, and a zero breaking threshold is
unbreakable. The attachment target is coincident at creation; creation fails
rather than teleporting endpoints whose initial gap exceeds the capture
distance.

An unbroken attachment keeps endpoint separation below
`max(1e-5 m, 0.02 * min(nodeSpacing_a, nodeSpacing_b))` after each completed
step. Constraint impulse is equal and opposite after resolving any rigid or
soft endpoint owners. When accumulated impulse first exceeds a positive break
threshold during one completed step, the attachment becomes inactive exactly
once, reports that event and applies no attachment impulse on later steps.
Explicit edge removal is effective on the next step. Attachment alone does not
enable collision, friction, knotting, shared stable IDs or transfer of another
coupling.

Contact and attachment may coexist in one canonical rope-pair edge but remain
separately queryable mode records and separately capacity-accounted operators.
Candidate, contact, constraint and diagnostic storage is declared in the
simulation memory plan. Overflow returns `PM_ERROR_CAPACITY_EXCEEDED`,
invalidates the affected output and permits retry; silently dropping contacts
or attachments is forbidden. Empty ropes, endpoint replacement, removal and
reset follow normal generation and rollback rules.

A future API leaf adds the capsule profile, rope-pair edge and typed endpoint
attachment state. Separate implementation leaves cover bounded segment
generation, capsule self-contact, inter-system contact, attachments and
portable scheduling. Qualification first audits the legacy node-only record,
then advertises capsule contact and attachment per driver only after crossing,
friction, conservation, stretch, break, capacity and lifecycle cases pass.

## Smoke self and overlapping-system scopes

Smoke has two independently advertised representations. `smoke-particle` is
the particle pressure/viscosity model. `smoke-grid` is the projected Eulerian
MAC grid; its particles are tracers and scalar-deposition sources, not a second
particle pressure solve. The API 0.2 `self` records describe the internal solve
of one live smoke slot only. They do not allow one smoke slot to read another
slot's particles, grid cells, velocity faces, density or temperature.

Separate-system smoke is a graph edge between two different live smoke
slots. It has three representation identities: particle-particle uses
`smoke-particle`, grid-grid uses `smoke-grid`, and particle-grid uses
`smoke-particle-grid`. API 0.2 reserves these identities and the canonical
`PM_COUPLING_SMOKE_SMOKE` edge without advertising any separate-system record.
They remain unsupported until their individual implementation and qualification
leaves finish. A driver cannot infer one from another or from a distinct-family
smoke capability.

The edge never merges ownership. Each endpoint retains its emitter, particle
count and capacity, stable-ID namespace, grid origin and spacing, output range,
material and generation. Numeric particle IDs and grid coordinates are local
to their endpoint. There is no smoke mass, particle, species or material
transfer in these profiles. Empty endpoints are no-ops. Edge removal isolates
the systems on the next step; endpoint replacement follows normal generation
and rollback rules.

### Required material state and shared invariants

`PmSmokeMaterialState` currently contains number-density and numerical solver
parameters but no mass density, particle mass, heat capacity or thermal
conductivity. Its grid temperature is an internal buoyancy scalar, and the
particle public state exposes no temperature. Those values cannot support a
portable conservation claim. Before any separate-system record is advertised,
a typed versioned extension must expose:

- positive mass density in kg/m^3 and, for the particle representation, a
  positive particle rest volume in m^3;
- positive specific heat capacity in J/(kg*K) for a `thermal` record and
  nonnegative thermal conductivity in W/(m*K); and
- explicit particle/grid representation selection and a canonical smoke-pair
  edge with `first < second`.

The particle mass is `massDensity * particleRestVolume`. Grid cell mass is
`massDensity * cellVolume`; scalar smoke density is a dimensionless transported
concentration and must not be mistaken for material mass density. The future
absolute endpoint temperature is in degrees Celsius, while temperature
differences used by the operator are also valid in kelvins.

All profiles use bidirectional exchange. In a closed fixture without buoyancy,
wind, emitters, obstacles, decay or speed caps, pressure and viscous exchange
must conserve total linear momentum. Let `delta_p_i` be the momentum change of
each owned particle or control volume, `I = sum(||delta_p_i||2)`, and
`R = ||sum(delta_p_i)||2`. A case with `I >= 1e-4 kg*m/s` must satisfy

```text
R <= max(1e-7 kg*m/s, 5e-5 * I).
```

The analogous angular-momentum residual about the fixture centre uses
`r_i cross delta_p_i` and the same absolute/relative limit in kg*m^2/s.
Viscosity and drag cannot increase kinetic energy beyond a relative `5e-5`
integration tolerance. Pressure produces no net impulse. With thermal exchange
and no external source, total sensible energy

```text
E = sum_i mass_i * specificHeat_i * (temperature_i - T_reference)
```

is conserved to `max(1e-6 J, 5e-5 * sum_i |delta_E_i|)`, independently of the
reference temperature. A stable conductive step keeps temperatures within the
pre-step endpoint extrema. Endpoint order cannot change the result beyond the
same tolerances.

### Particle-particle profile

A separate-system `smoke-particle` edge may advertise `pressure`, `viscosity`
and `thermal` independently. Pressure and viscosity use one complete symmetric
neighborhood over both populations. Every unordered cross pair contributes
once and applies equal and opposite force; solving A against B and B against A
as independent full-strength passes is forbidden. Cross neighbors count
against each endpoint's declared neighbor capacity, and overflow is an explicit
failed step rather than a partial interaction.

When endpoint material states are equal, a split run with the edge must match a
combined-population run on the same driver. Both runs contain the same particle
records and use the same step, substeps, external inputs and representable
capacities. Mapping by input record, the maximum position and velocity errors
after one step are at most `5e-4 * particleRadius` and
`5e-4 * particleRadius / dt`; particle counts remain exact. Unequal materials
use their own pressure law, particle mass and kinematic viscosity. Cross-pair
dynamic viscosity uses the harmonic combination of
`massDensity * kinematicViscosity`, as for miscible fluids.

Thermal exchange transfers equal energy in opposite directions using the
harmonic conductivity. It does not imply buoyancy, reaction, radiation, phase
change or material transfer. Pressure alone does not enable thermal exchange,
and thermal alone does not enable mechanical exchange.

### Grid-grid overlap profile

The active overlap is the world-space intersection of two grid volumes. Grid
origins, cell sizes and resolutions may differ. Conservative overlap weights
are nonnegative, proportional to intersected cell/face measure and sum to one
within `1e-6` for every covered destination sample. A uniform scalar or
velocity field is a fixed point of remapping; equal overlapping fields cannot
double density, temperature, pressure or momentum merely because two owners
cover the same location.

Each endpoint transports its own scalar concentration outside the overlap. In
the overlap, a `pressure` edge builds one composite flux/projection problem, or
an equivalent conservative interface operator, across both grids. Two
independent projections followed by velocity blending are insufficient because
they double pressure work and need not balance interface flux. After projection,
the volume-weighted divergence norm over the coupled overlap is no greater than
the pre-projection norm and meets the stricter advertised endpoint tolerance.
The pressure exchange has zero net impulse within the shared invariant above.

`viscosity` exchanges tangential momentum on the overlap using harmonic dynamic
viscosity. `thermal` exchanges sensible energy using harmonic conductivity.
Both use one conservative interface flux per geometric overlap and its equal
opposite contribution; neither overwrites endpoint-owned state with the other
grid's sample. A translating uniform field and a uniform temperature field are
fixed points. Nonoverlapping grid pairs produce bitwise-identical endpoint
outputs to the isolated control.

Overlap-cell, overlap-face, reduction and diagnostic storage is declared by the
simulation memory plan. Capacity overflow returns
`PM_ERROR_CAPACITY_EXCEEDED`, invalidates affected output and allows retry. It
cannot truncate the overlap or choose only one endpoint's resolution.

### Particle-grid profile

`smoke-particle-grid` initially defines only bidirectional `drag` and `thermal`
exchange. Particles sample the grid velocity and temperature at their world
position; their momentum and energy changes are deposited back to the same
covered grid faces/cells with normalized conservative weights. Drag is
reciprocal and dissipative. Thermal exchange is reciprocal and satisfies the
shared sensible-energy invariant. Particles outside the grid volume are
unchanged by the edge.

This profile does not claim a shared pressure solve, viscosity equivalence,
particle-to-grid conversion, grid-to-particle reseeding or phase change.
Pressure and viscosity remain unsupported for mixed representation until a
later contract defines a single conservative discretization rather than
blending two unrelated solvers.

### Scheduling and qualification boundary

Each edge consumes one declared coupling slot. Cross-neighbor, overlap-map,
exchange and diagnostic resources are bounded by the queried memory plan.
Portable scheduling must place exchange between the endpoint transport and
final integration/projection stages, use canonical endpoints, and avoid
reading a retired generation. Failure leaves the previously usable graph and
configuration intact.

Qualification is per driver, representation pair and mode. Particle cases
cover split-versus-combined equivalence, unequal masses, pressure, viscosity,
thermal exchange, capacity and lifecycle. Grid cases cover aligned and
misaligned origins, unequal resolution, partial/no overlap, uniform fixed
points, projection, conservation, capacity and lifecycle. Mixed cases cover
inside/outside sampling, reciprocal drag, thermal exchange and lifecycle.
Timestep sweeps and long-run energy/divergence cases are required before any
record is exposed. Passing one profile never advertises another.

## Materials and combination profiles

Every endpoint keeps its pipeline material. An edge selects a pair mode and a
material-combination profile; it never replaces either endpoint material.
Capability identity is therefore:

`(first family, second family, scope, mode, direction, material profile)`.

`scope` distinguishes self, same-domain inter-object, separate-system and
distinct-family interaction.

The initial profiles are:

| Profile key | Meaning | API 0.2 state |
|---|---|---|
| `endpoint-native` | Use both endpoint materials plus the pair's parameters. No unspecified averaging is permitted. | Reference profile for implemented distinct-family modes. |
| `smoke-particle` | Smoke is the particle tracer model. | Reference profile for implemented smoke pairs. |
| `smoke-grid` | Smoke is the projected Eulerian grid model. | Reference profile for implemented smoke pairs; support is independent from `smoke-particle`. |
| `smoke-particle-grid` | A particle smoke endpoint exchanges with a projected-grid smoke endpoint. | Reserved and unsupported; only reciprocal drag and thermal exchange are defined. |
| `equal-material` | Distinct fluid systems use numerically equal material state and a symmetric cross-population operator. | CUDA qualification complete for `contact`, `pressure` and `viscosity`; other production drivers remain unsupported. |
| `miscible` | Different Newtonian fluid materials share a mechanical neighborhood without a material interface. | Typed mechanical request exists; all drivers remain unadvertised pending implementation and qualification. |
| `immiscible-interface` | Different Newtonian fluids retain a material boundary with defined pressure, shear and capillary behavior. | Typed mechanical and capillary request exists; all drivers remain unadvertised pending implementation and qualification. |
| `thermal-phase` | Reserved for a future material-changing thermal or phase law. Thermal exchange that retains both materials uses the mechanical material profile plus the `thermal` mode. | Unsupported; F2-11 does not define phase transfer. |
| `rope-capsule` | Rope centreline segments are swept capsules; contact covers their full length rather than sampled nodes only. | Future profile specified by F2-14; unsupported in API 0.2. |

A driver must expose the supported smoke representation and fluid material
profile. Reporting only `domainCouplings`, or only a pair, is insufficient.
The fixed API 0.2 fluid–smoke law is identified by its `thermal` or `phase`
mode while its material profile remains `smoke-particle` or `smoke-grid`; this
preserves independent discovery of the smoke representation.

## Equal-material separate fluid systems

This section defines the `equal-material` profile reserved by F2-05. CUDA API
0.2 advertises `contact`, `pressure` and `viscosity` for a
`PM_COUPLING_LIQUID_LIQUID` edge after passing the F2-10 public-API quality and
lifecycle gates. A production driver without equivalent qualification must
omit these separate-system records and reject the edge as unsupported.

### Edge and material validity

An equal-material edge joins two different live fluid slots in one simulation.
`first` must be less than `second`; equal or reversed slots are invalid. The
edge uses this exact bundle:

- `modeFlags` is `CONTACT | PRESSURE | VISCOSITY`;
- `materialProfile` is `EQUAL_MATERIAL`;
- `mechanical.twoWay` is `PM_TRUE`; and
- every other coupling parameter is reserved and zero.

All fields of the two referenced `PmLiquidMaterialState` values must be
numerically equal after API validation: `restDensity`, `particleRadius`,
`supportRadius`, `restParticleVolume`, `pressureStiffness`,
`kinematicViscosity`, `normalDamping`, `velocityDamping`, `maximumSpeed`,
`maximumPairAcceleration`, and `densityIterations`. Material indices and fluid
slot indices need not be equal. A well-formed edge with unequal material state
is an unsupported material combination and returns
`PM_ERROR_FEATURE_NOT_PRESENT`; it must not average values or replace the live
graph with a partial edge. F2-11 defines different-material profiles.

The current pipeline supplies the endpoint material values. An existing graph
is rechecked when a pipeline or endpoint generation changes. A step whose edge
no longer resolves to two live, equal-material endpoints fails instead of using
stale material or silently isolating the populations. Restoring compatible
endpoints makes the unchanged graph usable again.

### Physical behavior

The ownership label is invisible to the mechanical solve. For every particle,
same-system and cross-system neighbors participate in the same support kernel,
iteration budget, ordering rules and material law:

- `contact` applies the same short-range exclusion response used inside one
  population; it does not create a membrane at the ownership boundary;
- `pressure` includes cross-system neighbors in density/pressure estimation
  and applies the same pressure-gradient response used for same-system
  neighbors; a surface penalty alone is insufficient; and
- `viscosity` includes cross-system relative velocity in the same Newtonian
  dissipative operator used inside one population. With no other work, it must
  not increase relative kinetic energy beyond the numerical tolerance below.

Each unordered cross-particle pair contributes once to the physical operator.
Its impulses on the two particles are equal and opposite within tolerance.
Implementations may fuse or reorder work, but must not solve A against B and
then B against A as two independent full-strength interactions. There is no
surface tension, thermal exchange, material mixing rule, phase change or
particle transfer in this profile.

### Numerical acceptance

Qualification compares two runs on the same device and driver. The split run
uses fluid A, fluid B and the edge above. The combined run concatenates the same
particle records into one fluid with the same material. Both runs use identical
time step, one substep, pipeline, external inputs and initially representable
capacities. Emitters, destroyers, other domains and other couplings are absent;
gravity and velocity damping are zero; speed and pair-acceleration caps do not
activate. The corpus contains both a zero-relative-velocity pressure/contact
case and a nonzero-relative-velocity viscosity case. Each case has cross-system
neighbors and changes at least one particle velocity by `1e-3 m/s` versus the
isolated split control.

Map particles by their input record, not by their numeric stable ID. Let `h` be
`supportRadius`, `dt` the step time, and `||.||inf` the largest absolute vector
component. After one step, the split and combined outputs must satisfy:

```text
max particle ||split.position     - combined.position||inf / h          <= 5e-4
max particle ||split.velocity     - combined.velocity||inf / (h / dt)   <= 5e-4
max particle ||split.acceleration - combined.acceleration||inf / (h/dt²) <= 5e-4
max particle |split.foam - combined.foam|                                <= 5e-4
```

Particle counts and temperatures are exact. These are same-driver
split-versus-combined criteria, not cross-vendor bitwise-output requirements.
Qualification sweeps timesteps `0.001`, `0.01` and `0.05` seconds and
population mass ratios `1`, `2` and `4` for both fixtures before a driver may
advertise the profile.

For the momentum fixture, particle mass is
`m = restDensity * restParticleVolume` when `restParticleVolume > 0`, otherwise
`m = restDensity * (2 * particleRadius)^3`. With no external force or boundary
impulse, let `delta_v_i` be each particle's velocity change,
`I = m * sum(||delta_v_i||2)`, and
`R = ||m * sum(delta_v_i)||2`. The fixture must produce `I >= 1e-4 kg*m/s` and
must satisfy:

```text
R <= max(1e-7 kg*m/s, 5e-5 * I)
```

The same residual limit applies to the pressure/contact and viscosity fixtures.
Energy lost by viscosity is dissipation, not a momentum sink.

### Ownership and capacity boundaries

The edge never merges storage. A and B retain independent particle counts,
declared capacities, output ranges, temperature/foam state and stable-ID
namespaces. Stable IDs may have the same numeric value in both systems; the
portable identity is `(fluid slot, stable ID)`. Coupling must not renumber a
live particle. Replacing a population creates that slot's new generation under
the normal resource-lifetime rules and must not retain particles or IDs from
the retired generation.

One fluid–fluid edge consumes one `couplingCapacity` entry. Each live count must
fit its own `PmFluidCapacity::particleCapacity`; the sum need not fit one
fluid's capacity. A particle's `neighborCapacityPerParticle` covers all of its
same-system and cross-system neighbors. Overflow returns
`PM_ERROR_CAPACITY_EXCEEDED` for the step and invalidates its output; silently
dropping cross neighbors is forbidden. Queried persistent and scratch memory
requirements must cover the declared pair without hidden allocation during a
step.

An edge with an empty but live population is a no-op for that endpoint and does
not move particles from the other population. Removing the edge isolates the
systems on the next step. Other graph edges and physical boundaries remain
owned per system: coupling A to a rigid, cloth or world boundary does not make
B inherit that interaction. A split-versus-combined comparison is valid only
when corresponding external boundary behavior is applied to both split
systems.

## Different-material separate fluid systems

The `miscible` and `immiscible-interface` profiles describe two live fluid
systems whose `PmLiquidMaterialState` values differ. The public contract can
represent these requests, but no driver advertises them yet. Every driver must
omit their capability records and reject a structurally valid graph edge with
`PM_ERROR_FEATURE_NOT_PRESENT`. It must not fall back to the `equal-material`
operator, average endpoint materials silently, or mutate the live graph after
rejecting a replacement.

`PmCouplingStateInfo` is the extensible coupling descriptor. For either
different-material profile its `pNext` chain contains exactly one
`PmFluidPairMaterialState`; duplicate and unknown structures are invalid.
An `immiscible-interface` request also contains exactly one
`PmFluidInterfaceState`. Recognized structures may appear in either chain
order. `firstMaterialIndex` and `secondMaterialIndex` equal the edge's
`first` and `second` fluid slots and select those endpoints' immutable
`PmLiquidMaterialState` entries. This explicit mapping prevents a loader or
driver from swapping, copying or synthesizing material ownership. Equal-
material fluid pairs may use the same typed state for source compatibility,
but retain the API 0.2 `PmCouplingState` path.

`pmCmdSetCouplings2` records a complete simulation-owned array of extensible
descriptors. `PmPhysicsPipelineCouplingStateInfo` supplies the corresponding
complete pipeline array through `PmPhysicsPipelineCreateInfo::pNext`; the
legacy base count and pointer are zero and null when that extension is present.
Both entry points validate and deep-copy all descriptor layers before changing
live state. A rejected replacement leaves the previous usable graph intact.

Both profiles are bidirectional. The edge keeps independent endpoint storage,
counts, capacities, stable-ID namespaces, temperature and foam fields. It
never changes a particle's owner or material. Particle mass for endpoint `i`
is

```text
m_i = restDensity_i * restParticleVolume_i
```

when `restParticleVolume_i > 0`; otherwise it is
`restDensity_i * (2 * particleRadius_i)^3`. A cross pair uses the symmetric
support `h_ij = max(supportRadius_i, supportRadius_j)` and contact separation
`d_ij = particleRadius_i + particleRadius_j`. The same pair support and kernel
sample must be used from both endpoints. A pair force is accumulated once as
`F_ij`, then applied as `F_ij / m_i` and `-F_ij / m_j`; acceleration caps limit
the common force before those divisions. Thus unequal masses do not permit
unequal-and-opposite acceleration to masquerade as momentum conservation.

Every cross-neighbor counts against the receiving endpoint's declared
`neighborCapacityPerParticle`. Pair storage and scratch remain bounded by the
simulation plan. Empty endpoints, removal, replacement, graph rollback after
failure, and external-boundary ownership follow the equal-material rules.

### Miscible mechanical profile

A miscible mechanical edge has the exact mode bundle bidirectional `contact`,
`pressure` and `viscosity` with material profile `miscible`. The populations may interpenetrate
and intermingle without an ownership seam. `contact` supplies the same
short-range particle exclusion used within each endpoint; it does not create a
membrane. Cross neighbors contribute their own mass to density. Each particle
uses its endpoint rest density and pressure stiffness for its equation of
state, while the pressure-gradient pair operator is symmetric in both endpoint
pressures and densities.

For the initial `miscible` profile, density uses the normalized three-
dimensional poly6 sample from the preceding pair-sampling contract. For a
cross pair at distance `r < h_ij`, endpoint pressure and the positive radial
kernel-gradient magnitude are

```text
p_i = pressureStiffness_i * max(rho_i - restDensity_i, 0)
G_ij = 9.40008882636 * r / h_ij^5 * (1 - r^2 / h_ij^2)^2
```

With `n_ij = (x_i - x_j) / r`, the common pressure force on endpoint `i` is

```text
F_pressure = n_ij * m_i * m_j *
             (p_i / rho_i^2 + p_j / rho_j^2) * G_ij
```

Short-range contact applies only for `r < d_ij`. Define

```text
m_pair = 2 * m_i * m_j / (m_i + m_j)
a_contact = 0.5 * (pressureStiffness_i * 1000 / restDensity_i +
                   pressureStiffness_j * 1000 / restDensity_j) *
                   (1 - r / d_ij)^2 +
            0.5 * (normalDamping_i + normalDamping_j) *
                   dot(v_j - v_i, n_ij)
F_contact = n_ij * m_pair * a_contact
```

The factor in `m_pair` preserves the established per-particle contact response
when the endpoint masses and material values are equal. At coincident
positions, `n_ij` is the normalized relative velocity `v_j - v_i`; it is zero
when that velocity is also zero. This rule is independent of endpoint
ownership and avoids choosing a physical direction from local array indices.
The implementation sums `F_pressure + F_contact`, limits that common force to
the smaller enabled endpoint bound `m_i * maximumPairAcceleration_i`, and only
then divides by endpoint mass.

Let `nu_i` be kinematic viscosity and `mu_i = restDensity_i * nu_i` be dynamic
viscosity. The cross-pair viscosity is the harmonic combination

```text
mu_ij = 2 * mu_i * mu_j / (mu_i + mu_j)
```

with `mu_ij = 0` when both inputs are zero. The discrete shear force must be
equal and opposite and, without external work, must not increase kinetic
energy. Pressure/contact must also conserve linear and angular momentum within
the qualification tolerances. Endpoint order cannot affect any result.

Let `V_i = m_i / restDensity_i` be the explicit or fallback rest particle
volume defined above. The initial miscible shear operator uses

```text
V_ij = 2 * V_i * V_j / (V_i + V_j)
q_ij = max(1 - ||x_i - x_j|| / h_ij, 0)
F_viscosity = mu_ij * V_ij * q_ij * (v_j - v_i)
```

`F_viscosity / m_i` and `-F_viscosity / m_j` are accumulated at the two
endpoints. This reduces exactly to the established equal-material acceleration
`kinematicViscosity * q_ij * (v_j - v_i)` when material and particle volume
match. If either endpoint viscosity is zero, harmonic `mu_ij` makes the cross
viscous force zero. Enabled endpoint acceleration limits clamp the common force
before either mass division.

This profile is a mechanical mixture, not a species solver. Particles retain
their material and stable identity as they mingle. It does not claim Fickian
diffusion, concentration fields, reactions, temperature-dependent properties,
mass transfer, phase change, or a homogeneous effective material. A driver
must report those behaviors unsupported rather than infer them from
`miscible`.

### Immiscible interface profile

An immiscible mechanical request has the exact bundle bidirectional `contact`,
`pressure` and `interface` with material profile `immiscible-interface`.
These are three independently queryable capability identities even though the
initial graph descriptor requires their complete bundle. Interfacial
`viscosity` and `thermal` remain independent optional records. Contact prevents
particles crossing the represented boundary, pressure transfers normal
traction, and interface force represents capillarity. No one of these records
implies another.

`PmFluidInterfaceState` owns a finite, nonnegative surface-tension coefficient
`surfaceTension`, written as `sigma` below, in N/m. The state appears exactly
once in an immiscible edge's `pNext` chain and is invalid for every other
material profile. At resolved curvature its equilibrium pressure jump obeys the
Young-Laplace relation

```text
pressure_first - pressure_second = sigma * (1 / R1 + 1 / R2)
```

for principal radii `R1` and `R2`. The normal always points from the edge's
first fluid endpoint to its second fluid endpoint; swapping endpoints reverses
the normal and pressure-jump sign without changing the physical state. A
planar interface therefore has zero capillary pressure jump. The
discrete interface force is internal: absent external work it has zero net
linear force and torque within tolerance. It may exchange kinetic and surface
energy but must not create unbounded energy in the long-run stability corpus.
`sigma = 0` removes capillary traction while retaining the no-mixing boundary.

When viscosity is requested, the tangential traction uses the same harmonic
dynamic-viscosity rule as the miscible profile and is dissipative. Surface
tension gradients, surfactants, Marangoni stress, wetting/contact angles,
triple-line behavior, species diffusion, reactions, breakup guarantees and
phase transfer are unsupported by this initial profile. Fluid-solid wetting
requires a separate three-material contract and cannot be inferred from a
fluid-fluid edge.

### Optional heat exchange without phase transfer

`thermal` may be advertised independently for either the `miscible` or
`immiscible-interface` profile after the API exposes endpoint caloric data.
Each endpoint material then owns positive specific heat capacity `c_p` in
J/(kg*K) and nonnegative thermal conductivity `k` in W/(m*K). Miscible
conduction uses the harmonic pair conductivity

```text
k_ij = 2 * k_i * k_j / (k_i + k_j)
```

with zero when both inputs are zero. An immiscible edge additionally owns a
nonnegative interfacial conductance in W/(m^2*K); zero means an insulating
interface. Conductive pair work transfers equal energy in opposite directions.
With no external source, the total sensible energy

```text
E = sum_i m_i * c_p_i * (temperatureCelsius_i - T_reference)
```

is conserved independently of the arbitrary reference temperature. A stable
step with nonnegative coefficients keeps particle temperatures within the
pre-step extrema. Endpoint order cannot change heat flow, and graph removal
stops it on the next step.

Thermal expansion, buoyancy, radiation, temperature-dependent coefficients,
latent heat, boiling, freezing, evaporation, condensation and particle
transfer are unsupported here. Those need explicit phase identities, energy
and capacity accounting, so the `thermal-phase` profile remains reserved.

### API separation and remaining state

The public registry has typed, versioned mechanical pair and capillary state.
Thermal work must add caloric fields without reusing the API 0.2 heater rate.
Mechanical, interface, thermal and phase records remain separately queryable.
Implementation leaves add shared Slang operators and backend-neutral
scheduling; qualification
leaves advertise one exact `(scope, mode, direction, material profile)` record
at a time on each tested driver.

## Capability query contract

`pmEnumeratePhysicalDeviceInteractionCapabilities` returns only supported
records. Each record is the exact six-part identity above plus
`maxGraphEdgesPerSimulation`. `PM_INTERACTION_GRAPH_EDGES_NOT_APPLICABLE`
means the mode is intrinsic and is not expressed as a graph edge.
`PM_INTERACTION_GRAPH_EDGES_SIMULATION_CAPACITY` means the device imposes no
narrower limit than `PmSimulationCreateInfo.couplingCapacity`. Any other value
is an additional per-simulation ceiling for that exact record.

`pmGetPhysicalDeviceInteractionCapability` queries one identity. Families use
the enum's canonical order: equal families for self, same-domain and
separate-system scopes; increasing family values for a distinct-family pair.
Unknown enum values, reversed distinct pairs and a scope/pair shape mismatch
return `PM_ERROR_INVALID_ARGUMENT`. A valid but unadvertised pair, mode,
direction or material combination returns `PM_ERROR_FEATURE_NOT_PRESENT`.
Applications must not infer support from a related record or a broad feature
bit.

Graph edges name their complete interaction bundle with
`PmCouplingState::modeFlags` and select its material representation with
`materialProfile`. Each mode bit maps to the corresponding
`PmInteractionMode` record used by the capability query. API 0.2 implements a
fixed bundle for each `PmCouplingType`; missing and additional known modes are
unsupported rather than silently enabled or ignored. Endpoint-native pairs
require `PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE`; smoke pairs select the
advertised particle or grid profile. The implementation validates every record,
duplicate endpoint pair, endpoint slot and graph limit before replacing the
live graph. A rejected replacement leaves the previous graph usable.

## Units and parameter ownership

Parallel Mater uses metres, seconds, kilograms and radians. API 0.2 particle
temperature fields explicitly use degrees Celsius; `heaterTemperature` and
`boilingTemperature` use the same Celsius scale. Temperature differences have
the same magnitude in kelvin. Future absolute thermodynamic fields must state
their scale rather than inheriting a global assumption.

| Parameter | Unit / rule |
|---|---|
| `contactDistance`, heater centre/half extents | m |
| mechanical `stiffness` | N/m |
| mechanical `damping`, `tangentialDrag` | N·s/m |
| `friction`, `dragCoefficient`, radius scales | dimensionless |
| `maximumForce` | N |
| `maximumAcceleration`, `maximumWindAcceleration` | m/s² |
| `windDrag`, `heatTransferRate` | s⁻¹ |
| `airDensity` | kg/m³ |
| attachment `effectiveMass` | kg |
| `steamRiseSpeed` | m/s |
| heater and boiling temperatures | °C |
| vertex indices and Boolean selectors | dimensionless |

The edge owns contact/attachment/transfer parameters. Endpoint materials own
mass, density, radius, compliance, viscosity and internal solver policy. A zero
parameter means a documented pair-specific default only where the pair contract
says so; it never selects another interaction mode.

## API 0.2 direction rules

`PmMechanicalCouplingState.twoWay` is meaningful only for the mechanical edge
types that read it. The reference implementation accepts `PM_TRUE` and rejects
`PM_FALSE` for those types. It must not silently turn a requested one-way mode
into two-way contact. Rope attachments and smoke/thermal modes have the fixed
directions shown above; graph validation checks those directions through their
capability records.

These exact pair/mode/profile queries replace broad discovery. F2-03 validates
every graph edge against them before changing the live graph. Later
implementation tasks may change `P` to a reference path only after their public
behavior and negative cases are reviewable; driver advertisement still waits
for qualification evidence.
