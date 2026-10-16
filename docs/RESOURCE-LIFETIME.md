# Reference resource lifetime and recovery

This records the reference driver's lifecycle behavior and qualification scope.
The public specification remains authoritative; unresolved differences are
tracked in ROADMAP F1–F3 rather than treated as conformance evidence.

## Bound storage

Simulation requirements and allocation share typed traversals for global and
fluid storage. Alignment padding and integer overflow are included in planning.
Other domains, geometry and coupling budgets still need conversion from
conservative estimates. Memory requirements describe application-bound storage;
they do not include every host allocation or native-driver cost.

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
public commands continue to address their declared slot indices. Coupled endpoint
replacement and graph rollback remain F2 work.

## Current evidence

`tests/api/resource_lifecycle_tests.cpp` uses only installed public SDK headers
and loader symbols. It can be built separately with `cmake -S tests/api` and an
installed `ParallelMater_DIR`; select an installed driver using PM_DRIVER_FILES.

The test binds exactly queried storage, crosses rigid and radix tile capacity
boundaries, performs 4,096 replacements, clears populations, rejects undersized
views, exercises 128 deferred failures and recovers while preserving another
population. Slang CPU and freshly rebuilt CUDA pass on this laptop. Equivalent
deformable, smoke, geometry and coupling lifecycle coverage remains a
qualification obligation.

`tests/solver/memory_arena_tests.cpp` additionally verifies coalescing, live-data
isolation, exact-capacity allocation, alignment, rejected oversized allocations,
and rejection of interior/double frees.
