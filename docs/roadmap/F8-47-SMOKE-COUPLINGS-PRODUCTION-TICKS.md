# F8-47 smoke couplings in production fixed ticks

F8-47 connects every implemented smoke/rigid and smoke/deformable mechanical
tuple to the installed public `realtime-single-step` schedule. Both particle
and grid smoke now execute contact and drag with rigid bodies, cloth, soft
bodies and ropes. Liquid/smoke remains the F8-46 path, and unavailable
smoke/smoke modes remain outside this task.

## Fixed coupling pass

The native boundary stages each live edge in canonical pair/mode order and
pins its endpoint generations and graph revision. Smoke coupling then executes
four fixed `dt / 4` sweeps after the domain finalizers have produced staged
positions and velocities, but before any public output or domain state is
swapped. This placement prevents the domain commit from overwriting coupling
velocity changes and preserves one time advance and one atomic publication.

Particle-smoke deformable drag builds and sorts its cell keys once before the
four sweeps. Every later drag visit reuses that order. Grid smoke reads the
staged density, pressure and face-velocity buffers. Rigid contact accumulates
and applies reciprocal impulses; grid drag applies the existing grid force.
Deformable contact changes smoke as specified by its one-way capability, while
drag changes the cloth, soft-body or rope endpoint. The implementation does
not claim reciprocity for those one-way modes.

The physical Vulkan trace records the production kernels directly:

- `smoke_rigid_contact`, `smoke_apply_rigid_impulses` and
  `smoke_grid_rigid_force`;
- `smoke_cloth_contact`, `smoke_cloth_wind`, `smoke_grid_cloth_force` and
  `smoke_grid_apply_cloth_force`;
- `smoke_soft_body_contact`, `smoke_soft_body_wind` and
  `smoke_grid_soft_body_force`; and
- `smoke_rope_contact` and `smoke_rope_wind`, plus the shared particle-cell
  emission and radix-sort entries.

There is no platform-local physics or CPU fallback. The protected Slang
closure and routing hash remains exact.

## Public graph and atomic failure

Realtime simulations now accept a nonzero fixed `couplingCapacity`; graph
installation previously failed before any production coupling could run.
Dynamic emitter, destroyer, paint and debug-frame capacities remain rejected
by this profile. The public fixture declares capacity 13, installs the four
smoke edges, and exercises particle and grid smoke through
`pmCmdSetCouplings2` and `pmCmdStepSimulation`.

Both smoke representations report 15 active mode records, 12 family sweeps,
one time advance, one spatial build, one publication and zero intra-tick host
readbacks in the focused mixed fixture. One-below workgroup capacity fails
before commit and leaves cloth, soft-body, rope, liquid and smoke output bytes
unchanged. Complete graph removal, particle-to-grid replacement and reset then
succeed. Registration rejects retired generations and invalid endpoint or mesh
identity before device mutation.

## Qualification

An exact-revision Android arm64 Vulkan build ran on a Pixel 7 with Mali-G710.
Command, async, liquid and focused production smoke runtime tests all passed;
the runtime log contains the particle and grid kernels above and no Vulkan
validation error. The same staged build then ran all eight frozen smoke quality
profiles: 30 samples passed without a tolerance change. The eight matching
lifecycle profiles passed 13 cases and 65 checkpoints, covering graph removal,
endpoint retirement/rebinding and reset behavior.

The complete 102-test host suite passed with 87 executed cases and 15
explicitly skipped unadvertised CPU-domain cases. Failure-injection and
production CPU fixtures passed, as did 418 conformance tests, 87 roadmap tests,
the protected realtime closure check, and current CPU, CUDA, desktop Vulkan
and Android Vulkan builds.

Exact commands, hashes, counters and profile lists are in
[`evidence/F8-47-smoke-couplings-production-ticks.json`](evidence/F8-47-smoke-couplings-production-ticks.json).
