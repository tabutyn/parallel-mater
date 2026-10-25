# F8-41 cloth operators in production fixed ticks

F8-41 connects the frozen F8-10 deformable spatial emitter and F8-13 cloth
self-dynamics to the public native `realtime-single-step` execution path. A
simulation with live cloth is no longer rejected merely because it contains
cloth state. CPU, CUDA, Vulkan and Metal adapters call the same Slang
operators; GPU drivers contain no private or serial CPU cloth physics.

## Fixed tick

At the tick boundary, every live cloth copies its completed public positions
and velocities into fixed-capacity staging. The production schedule then:

1. Runs `cloth_realtime_predict` once at `dt = 1/60`, preserving exact pinned
   vertices and recording the sole time advance.
2. Emits swept cloth triangles into disjoint ranges of the common spatial
   arena, builds and queries that arena once, and publishes one immutable
   epoch shared with any live rigid or liquid resources.
3. Clears the reserved canonical contact view, ready for the existing
   interaction operators that F8-45 will bind without changing cloth's frozen
   solver ABI.
4. Executes exactly eight link/apply/volume sweeps. The existing contact
   consumer runs only after sweeps 2, 4, 6 and 8. The authored legacy iteration
   count cannot
   add a pass, hidden substep, retry or spatial rebuild.
5. Reconstructs velocity once, updates the surface once and atomically swaps
   staged position, velocity and applicable surface state into public view.

The schedule records 31 cloth-domain dispatches: one predictor, 24 structural
and volume dispatches, four contact passes, one commit and one surface update.
Every sweep consumes the F8-10 epoch pinned by `ClothRealtimeTickContract`.

## Storage, validation and failure behavior

Creation reserves both public and staged position/velocity arrays, per-vertex
characteristic lengths and correction/contact scratch, triangle generations,
the time-advance counter and deformable-spatial scratch. A tick performs no
allocation, capacity growth, intra-tick host readback or global device wait.

Material and geometry inputs are checked against the existing F8-13 envelope
before device work. The maximum speed is rounded conservatively below the
half-characteristic-length boundary so valid meshes cannot fail from a
floating-point boundary product. Fracture remains unsupported in this profile
until F8-48 owns its topology lifecycle; ordinary whole-cloth replacement is
accepted only at a tick boundary and receives a new generation/revision.

Exact-limit preflight failure occurs before staging can publish. The public
failure fixture lowers the workgroup limit by one, observes
`PM_ERROR_CAPACITY_EXCEEDED`, verifies byte-for-byte unchanged cloth output,
and then reuses the same capacity. Reset retires the cloth and its cached
identity; rewriting the slot and stepping again succeeds from the reserved
arena.

## Public and independent checks

The installed public fixture covers a pinned cloth-only tick, exact telemetry,
failed-tick rollback, topology replacement, cloth/static-rigid resource
coexistence and reset/reuse. It ran through production Slang CUDA on an RTX
3050 Ti and production Slang Vulkan on a Pixel 7. The Pixel trace names every
operator above and the common spatial build/publication pipeline.

The independent F8-13 physical CUDA oracle remains separate from the public
fixture. It reports zero stiff-link error and pin drift, 0.115461% final volume
error from a 50% initial error, zero tunneling over 3,600 ticks, bounded stretch
and zero final kinetic energy. The complete host suite, CUDA adapter build,
Vulkan device build and Metal binding syntax also pass.

F8-41 binds cloth self-dynamics and its frozen contact consumer. Rigid/cloth
contact generation and bidirectional response belong to F8-45, other domain
couplings retain their later leaves, and fracture/topology mutation belongs to
F8-48. No unavailable interaction mode is advertised.

Exact commands, device identities and artifact hashes are recorded in
[`evidence/F8-41-cloth-production-ticks.json`](evidence/F8-41-cloth-production-ticks.json).
