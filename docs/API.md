# Parallel Mater command API

Parallel Mater exposes one C API from
[`parallel_mater.h`](../include/parallel_mater/parallel_mater.h). Applications
link the `parallel-mater` loader, not a device implementation. The normative
rules and complete declarations are in the generated
[API specification](reference/parallel-mater-specification.pdf) and
[reference guide](reference/parallel-mater-reference.pdf).

## Object and submission model

```text
PmInstance
  └─ PmPhysicalDevice ── selects one installable driver
       └─ PmDevice
            ├─ PmQueue
            ├─ memory + buffers + geometry
            ├─ physics pipelines + simulations
            ├─ command pools ── PmCommandBuffer
            ├─ fences + timeline semaphores
            └─ query pools
```

The loader discovers drivers, aggregates their physical devices, and routes
each call to the driver that owns the first dispatchable handle. No backend
enum chooses a solver. Unsupported capabilities are unadvertised and return
the documented error; the loader never falls back to another physics
implementation.

## Startup

1. Call `pmEnumerateInstanceVersion`.
2. Enumerate instance layers and extensions.
3. Enable required names in `PmInstanceCreateInfo` and call
   `pmCreateInstance`.
4. Enumerate `PmPhysicalDevice` handles and query their properties, queue
   families, features, limits, formats, memory properties, and device
   extensions.
5. Enable only supported features and extensions in `PmDeviceCreateInfo`, then
   call `pmCreateDevice` and `pmGetDeviceQueue`.

Commands may be linked directly through the loader or resolved through
`pmGetInstanceProcAddr` and `pmGetDeviceProcAddr`. An enabled extension must
still have non-null function pointers for the selected device.

## Memory and resources

Applications allocate `PmDeviceMemory` explicitly and bind it to buffers,
geometry, and simulations. Query memory requirements before allocation and
respect size, alignment, and memory-type bits. Mapping, flushing, and
invalidating operate on host-visible allocations.

Geometry describes indexed triangles through typed buffer views. A simulation
reserves explicit capacities for every enabled physics domain, coupling,
event stream, and debug stream. A physics pipeline contains solver and material
state for rigid bodies, fluid, cloth, rope, soft bodies, smoke, and their
couplings. Resource creation does not execute simulation work.

## Command buffers

Allocate command buffers from a command pool. Their lifecycle is:

```text
initial → recording → executable → pending → executable
   ↑          │             │                    │
   └──────────┴──────── reset command buffer ────┘
```

Record resource uploads, geometry builds, simulation writes, pipeline binds,
steps, barriers, queries, and output copies between begin/end calls. Submit an
executable buffer to a compatible queue. A pending buffer and every resource it
references remain alive until submission completion.

## Synchronization

`PmFence` reports completion to the host. Timeline `PmSemaphore` values order
work between queues and external pipelines. `PmSubmitInfo` carries command
buffers, waits, and signals. `pmCmdPipelineBarrier` expresses in-command-buffer
memory dependencies. Queue and device idle calls are available for teardown,
not required for ordinary frame scheduling.

Native interoperability is capability- and extension-gated. Import/export
calls retain explicit ownership; applications must obey both APIs' memory and
synchronization rules.

## Queries and output

Timestamp and statistics queries are recorded with simulation commands and
read after completion. Memory-budget queries report driver and application
usage. Simulation status reports deferred execution errors. State, contacts,
constraint state, paint, and debug data are copied into application buffers by
recorded commands; drivers do not expose private solver pointers.

## Drivers and layers

Driver manifests, the private loader/driver ABI, dispatch ownership, vendor
responsibilities, and qualification requirements are documented in the
[driver SDK](DRIVER_SDK.md). The optional validation layer is enabled by name
and observes the same command stream without becoming a physics
implementation.

The legacy C++ `World` implementation remains a private implementation detail
inside the first Slang/CUDA driver while its scheduling code is decomposed. It
is not installed, exported, or a supported application interface.
