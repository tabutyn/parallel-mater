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
| Fluid | `R` contact, pressure and viscosity within one population. | `P fluid A ↔ fluid B` contact, pressure and viscosity for equal material; later miscible/immiscible, thermal and phase profiles. API 0.2 `LIQUID_LIQUID` is explicitly unsupported. |
| Cloth | No advertised self or cloth–cloth contact mode. | `P cloth A ↔ cloth B` contact/friction; self-contact and distinct sheets are specified by F2-12. |
| Soft body | No advertised self or soft-body/soft-body surface-contact mode. | `P soft A ↔ soft B` surface contact; self and distinct volumes are specified by F2-13. |
| Rope | `R` segment self-contact/friction within one rope when `selfCollision` is true. | `P rope A ↔ rope B` segment contact and attachment; specified by F2-14. |
| Smoke | `R` pressure, viscosity and transport within one particle population or projected grid. | `P smoke A ↔ smoke B` overlapping-field exchange; pressure, viscosity and thermal semantics are specified by F2-15. |
| Rigid body | `R` body–body contact/friction and persistent constraints between bodies in one simulation. | There is no separate rigid-system resource in API 0.2; a future one must not inherit the intra-simulation claim implicitly. |

The proposed marks reserve semantic work, not support. Until their leaf tasks
finish, the corresponding query result is unsupported and edge creation must
fail without mutating the usable graph.

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
| `equal-material` | Distinct fluid systems use numerically equal material state and a symmetric cross-population operator. | Proposed by F2-05 through F2-10. |
| `miscible` | Different fluid materials mix under a specified combination law. | Proposed specification in F2-11; unsupported in API 0.2. |
| `immiscible-interface` | Different fluids retain an interface with defined density, viscosity and surface behavior. | Proposed specification in F2-11; unsupported in API 0.2. |
| `thermal-phase` | The material combination itself defines heat and/or phase transfer, including external sources and capacity behavior. | Reserved for broader combinations proposed in F2-11. |

A driver must expose the supported smoke representation and fluid material
profile. Reporting only `domainCouplings`, or only a pair, is insufficient.
The fixed API 0.2 fluid–smoke law is identified by its `thermal` or `phase`
mode while its material profile remains `smoke-particle` or `smoke-grid`; this
preserves independent discovery of the smoke representation.

## Equal-material separate fluid systems

This section defines the `equal-material` profile reserved by F2-05. It is the
contract for a future `PM_COUPLING_LIQUID_LIQUID` edge; it does not make that
edge supported in API 0.2. A driver must continue to report the separate-system
records unsupported until the implementation and qualification tasks close.

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
F2-10 adds timestep, population-ratio and long-run cases before a driver may
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
