# Unified rigid AVBD solver

The production CUDA adapter consumes the numerical core generated from
[one Slang implementation](SLANG.md); the same source is compiled for DXIL and
MSL validation. The portable C++ header is the host reference used by equation
fixtures. Collision geometry, graph construction, storage, synchronization,
and dispatch remain device-adapter responsibilities.

Rigid contacts and fixed, point, hinge, slider, piston, generic, spring and
motor joints participate in the same Augmented Vertex Block Descent solver.
There is no stack-only solver, joint-triggered fallback, welded-compound
velocity solve, or guided-body projection in the production stepping path.
Collision filtering still groups welded siblings; that does not merge their
physical mass or replace their joints.

The retired DirectCompute backend is no longer present. The equations,
iteration semantics and validation below describe the common Slang solver;
native APIs provide execution and presentation rather than separate physics.

## Shared numerical core

[geometry_constraints.slang](../src/slang/geometry_constraints.slang) is
compiled into the production CUDA runtime and cross-target validation outputs.
It owns both the shared
AVBD numerical core and the migrating rigid contact/joint solver entry points.
[geometry_constraints.hpp](../include/parallel_mater/solver/geometry_constraints.hpp)
preserves
the matching C++17 reference contract and thin generated-code adapters. The
shared numerical core owns:

- Coupled six-degree-of-freedom body blocks and an SPD LDLᵀ solve.
- Augmented-Lagrangian forces, penalty growth and temporal warm starting.
- Finite spring stiffness, implicit damping and bounded motor forces.
- Nonnegative contact forces and a radial Coulomb friction cone.
- Corrective-phase reference forces, so total motor/friction bounds are not
  accidentally applied twice.

Device adapters own geometry, persistent state, adjacency, vertex coloring and
GPU synchronization. Static/kinematic endpoints are read-only.
Dynamic contact and joint edges connect islands; sharing a floor does not.
CPU fixtures exercise the portable reference, while Slang reflection and PTX
fixtures exercise generated code from the production source.

Well-conditioned blocks use ordinary LDLᵀ. If floating-point cancellation
destroys an inertial pivot, the same block is equilibrated and retried with a
roundoff-sized modified-Cholesky floor. Materially indefinite or nonfinite
input is still rejected; this does not cap physical forces or add solve sweeps.

The method follows [Giles, Diaz and Yuksel, SIGGRAPH 2025](https://graphics.cs.utah.edu/research/projects/avbd/Augmented_VBD-SIGGRAPH25.pdf).
The authors' [reference implementation](https://github.com/savant117/avbd-demo3d)
also informed the rigid-body formulation. This engine's collision geometry,
unit scaling, API integration and impact treatment are adaptations; the
paper's hardware timings are not performance promises for this library.

## One body solver, two explicit physical phases

1. Integrate external forces, impulses and damping into inertial target poses.
2. Prepare contacts/joints, load matching force/penalty history, and color the
   body graph. Neighboring dynamic bodies have different colors.
3. For each iteration, minimize each body's local six-dimensional energy, then
   update constraint forces and penalties.
4. Reconstruct velocities. Impacted islands additionally run the same AVBD
   block/dual machinery in velocity coordinates `u = h*v`. This accounts for
   post-impact velocity and restitution; whole-step pose differences alone
   would report average travel velocity.
5. Apply speed limits and publish total impulses and cache history.

The impact phase includes supporting contacts and hard joints in the affected
island. It retains total force bounds, excludes a second application of spring
forces, and leaves conservative collision-corrected poses unchanged. It is
**not** an exact within-substep time-of-impact rebound trajectory.

Inactive unilateral joint rows have zero tangent stiffness. Bounded motors
retain a conservative stiffness estimate to prevent Newton steps from cycling
between opposite force caps. In either phase, if a proposed body update
would activate an inactive normal contact, its stiffness is added and that
local block is solved once more. Separating contacts retain zero curvature,
so geometric overlap correction cannot leave artificial ejection velocity.
Zero friction still means zero tangential stiffness. These are local Hessian
safeguards, not additional global constraint sweeps.

After each phase, a completely free dynamic island also minimizes the common
translation mode of that same energy: subtract its mass-weighted mean pose or
velocity-coordinate correction. Relative constraints are unchanged, while
finite-iteration ordering cannot introduce net linear momentum. Islands with
any static/kinematic contact or joint endpoint are excluded so external
reaction impulses are preserved. The reduction uses deterministic body order.

Unilateral joint limits target recovery of 20% of a pre-existing violation
per substep; their unviolated interior gap is unchanged. Equal-bound locks
and other hard joints retain the shared stabilization parameter (a 1% recovery
target). Finite iteration budgets do not guarantee either target is reached.
Force warm-start decay retains the shared AVBD parameters.

Linear multipliers are forces in N; angular multipliers are torques in N·m.
Diagnostic impulses multiply these by the substep duration, yielding N·s or
N·m·s respectively.
Normals point from B to A; reported impulses act on A. Restitution mixes by
minimum, friction by geometric mean. Persistent tangent forces are reprojected
into the new contact basis. Sphere and constrained-body contacts use the same
history rules as boxes.

The contact slop band follows the collision adapter's rest offset, including
the sharper offset used by guided swept contacts. Collision skin is not a
request to push an already resting body outward.

Contact cache matching checks body generations, nearby local features, normal,
epoch and timestep. Teleports/topology edits invalidate history. Updating only
a joint target/frame must not invalidate every unrelated contact in the world.
Joint row history is reset when its authored definition changes or its slot is
reused.

## Iterations and diagnostics

The default is ten primal/dual iterations per phase, raised by active joint
`solver_iterations`. `StepOptions::rigid_contact_pass_limit` keeps its existing
API name but now selects **iterations per AVBD phase**, for contacts and joints
together: zero means automatic; 1–64 is explicit. This replaces its former
contact-only hard-cap semantics. An impacted island can therefore use twice
the requested number. Substeps are separate and unchanged.

The overlay's AVBD iteration count reports actual pose plus impact work, not
GPU launches. The usual default is 10 for ordinary stepping and 20 for a
substep with impact correction. A smaller budget is not proof of convergence.

The CUDA driver distributes colored vertices over a resident cooperative grid; devices
without cooperative launch use the same equations in one thread block.
Graph preparation stages compact adjacency, parent and color metadata in a
bounded shared-memory cache. CUDA caches the first 1,024 bodies/contacts and
128 joints (44,032 bytes). The DXIL compile contract uses 512/512/64 to fit its
threadgroup budget. Entries outside those prefixes use global storage through
the same accessors. Union order, coloring order and physical equations do not
change at a cache boundary.

## Validation

Standalone numerical tests need no GPU toolkit:

```sh
cmake -S tests/solver -B /tmp/parallel-mater-avbd-core-build
cmake --build /tmp/parallel-mater-avbd-core-build
ctest --test-dir /tmp/parallel-mater-avbd-core-build --output-on-failure
```

The regular build also runs identical fixtures on CUDA, including an independent
double-precision block reference, anisotropic inertia, off-center coupling,
force bounds, finite stiffness, damping, cone projection, clamped contact
curvature and corrective reference forces.
Ordinary fixtures require component-wise CPU/CUDA parity. Deliberately
ill-conditioned contact blocks instead require finite bounded descent and
small diagonally scaled backward error on both platforms: float rounding can
remove inertia from the matrix, so near-null forward components are not a
meaningful parity contract. Materially indefinite or nonfinite blocks must
still be rejected.
The same fixture runners also validate a two-halfspace projection used by
CUDA soft-surface contact cleanup. For two colliders, cleanup searches their convex
face pairs for the smallest total squared corner correction, keeping one face
pair for the whole triangle. This avoids both alternating opposing corrections
and unnecessarily large escapes from extrapolated local planes. A seeded
upper bound prunes deeper faces; single-collider arithmetic is unchanged.
Mesh upload removes exact duplicate planes from this search's private candidate
list, preserving the original per-triangle planes for every other consumer.
Worst-case search cost remains quadratic in the number of distinct faces.
It adds no cleanup passes. This is a coupled-surface repair, not another rigid
solver. Fixtures cover nearly opposing thin boxes, sphere pinching and a
triangle whose corners must not escape through inconsistent faces.
`parallel-mater-command-api-tests` executes a Slang AVBD step through the
public command API and reads the result through an output buffer.
`parallel-mater-command-gallery-tests` builds and advances all 28 gallery
scenes through the same API, covering rigid constraints and every coupled
domain. `parallel-mater-contact-friction-tests` retains the coupled tangent
descent fixtures used by the runtime; the obsolete standalone
sequential-impulse solver and its self-tests have been removed.

Run production physics tests serially so independent GPU workloads do not
contend with each other or distort performance comparisons:

```sh
# Full configured suite, including gallery and backend contracts.
ctest --test-dir build-gallery -j1 --output-on-failure
# Command API and complete gallery scene path.
ctest --test-dir build-gallery -j1 --output-on-failure \
  -R '^parallel-mater-command-(api|gallery)-tests$'
# Narrow equation fixtures; these are not a substitute for the full suite.
ctest --test-dir build-gallery -R '^parallel-mater-avbd-core-tests$' \
  --output-on-failure
```

Full gallery regressions and driver conformance remain separate requirements:
scalar parity does not prove collision generation, scene stability or coupled
fluid/rope/cloth behavior. Do not regenerate goldens or loosen physical
tolerances to conceal a solver regression. Future D3D12, Metal, Vulkan, and
WebGPU adapters must execute these same cases through the loader; source
cross-compilation alone is not device qualification.

The migration also repairs two coupled-contact cases exposed by changed rigid
trajectories: opposing soft-surface planes use the two-halfspace projection
above, and rigid-to-rope positional contacts honor existing static support in
both effective mass and displacement. Shared `contact_friction.hpp` eliminates
the normal row before the friction correction, accounting for the resulting
anisotropic mass and enforcing the friction bound against the corrected normal
force. This is one coupled tangent descent step, not a claim of fully converged
sliding friction. Rope supports remain unilateral; an
upward-gravity lift-off regression checks that settled nodes can still leave
the floor. These changes do not increase the global rigid iteration budget.
