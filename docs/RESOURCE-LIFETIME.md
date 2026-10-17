# Reference resource lifetime and recovery

This records the reference driver's lifecycle behavior and qualification scope.
The public specification remains authoritative; unresolved differences are
tracked in ROADMAP F1–F3 rather than treated as conformance evidence.

## Bound storage

Simulation requirements and allocation share typed traversals for global,
fluid, cloth, rope, soft-body, smoke, emitter and major coupling storage.
Alignment padding and integer overflow are included in planning. Each particle
system has its own disjoint region within the application's binding; resizing
one slot cannot fragment another slot's promised capacity. Smoke planning bounds
staggered faces and multigrid levels for any supported shape within cell capacity.
Coupling storage is bounded using the largest declared endpoint capacities.
Emitter storage covers the reference sampler's maximum of 65,536 sites per
emitter. Geometry and paint allocations use the same typed planning traversal.
Memory requirements describe application-bound storage; they do not include
every host allocation or native-driver cost.

Rope capacity applies to the resampled nodes, not just the authored centerline.
Cloth fracture reserves source vertices plus triangle corners. Writes exceeding
those generated capacities fail before enqueueing; the gallery declares the
full fracture reserve.

The arena tracks live allocations and reuses freed gaps. It never writes host
allocator metadata into device memory. Retiring an allocation does not reset
unrelated resources. Simulation reset first destroys runtime-owned allocations,
then reinitializes within the same binding. Geometry allocations owned by other
simulations are not reset.

## Replacement and deferred failure

A recording-time validation failure does not enqueue that write. Previously
recorded commands remain executable unless a recording allocation error latches
a command-buffer error. Nonempty fluid input needs PARTICLE_INPUT usage and a
view whose declared range covers every element. A zero particle count ignores
buffer, offset, range, format and stride and creates a valid empty population.

The reference driver retires the target population before validating device
particle contents. If execution fails, the target may be empty. The submission
stops at the first failure, drains already issued native work, then reports the
error through its fence/timeline. Partial allocation is reclaimed. Successful
commands earlier in the submission are not rolled back. Other populations
retain their data. A subsequent valid write can recreate the target; reset is
required after a failed physics step where output validity is unknown.

Applications must keep referenced objects and memory alive until completion.
Pending storage is not reusable simply because pmQueueSubmit returned.
Generation changes invalidate private runtime handles when a slot is retired;
public commands continue to address their declared slot indices. Active coupling
handles are retired before endpoint writes. Graph descriptors retain slot indices
and are resolved against current generations on the next step. A failed graph
creation removes its partial edges; recovery can install a valid graph. Broader
auxiliary references, including emitters and paint, still need lifecycle coverage.

Each rigid slot also records the public geometry used to create its private mesh
handle. Executing a body replacement or removal retires the old private mesh only
after the final rigid reference disappears; a shared mesh and unrelated bodies
remain live. The runtime refuses body-driven retirement while a paint field still
owns the mesh; F1-04 covers releasing that final paint dependency. Freed mesh
slots advance their generation before reuse, and reset clears the full
per-simulation geometry registry. Geometry objects, their bound memory, and input
buffers remain application-owned and must outlive recorded and pending commands
plus every live simulation reference.

## Current evidence

`tests/api/resource_lifecycle_tests.cpp` uses only installed public SDK headers
and loader symbols. It can be built separately with `cmake -S tests/api` and an
installed `ParallelMater_DIR`; select an installed driver using PM_DRIVER_FILES.

The test binds exactly queried storage, crosses rigid and radix tile capacity
boundaries, performs 4,096 replacements, clears populations, rejects undersized
views, exercises 128 deferred failures and recovers while preserving another
population. Slang CPU, CUDA and Vulkan pass on this laptop.

`tests/api/domain_lifecycle_tests.cpp` performs 1,024 interleaved cloth, rope,
soft-body and smoke writes using exact queried memory. It alternates fracture
storage, rope lengths and smoke grid layouts and rejects generated capacity
overflow. CUDA passes. CPU and Vulkan do not advertise all required domains,
so this case is unsupported on those drivers, not a passing physics result.
`tests/api/geometry_lifecycle_tests.cpp` instances one mesh across two bodies,
then cycles 20 public geometries through one body slot. It covers final-reference
removal, shared-body preservation, stale-generation reuse and reset using exact
queried bindings. Paint and emitter lifecycle cases remain qualification
obligations; these tests do not establish numerical solver quality.

`tests/solver/memory_arena_tests.cpp` additionally verifies coalescing, live-data
isolation, exact-capacity allocation, alignment, rejected oversized allocations,
and rejection of interior/double frees.

`tests/api/coupling_lifecycle_tests.cpp` qualifies graph lifetime using a liquid
and rope without contact forces: repeated endpoint replacement, pipeline changes,
explicit simulation graph changes, descriptor snapshots, capacity rejection and
partial-creation rollback pass on CUDA. This is lifecycle evidence, not a claim
of physical accuracy for every coupling pair. Retiring fluid/smoke owners and
smoke coupling edges no longer allocates a replacement host tombstone.
