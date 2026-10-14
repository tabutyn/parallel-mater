# Current architecture and remaining work

Parallel Mater exposes one Vulkan-style C command API through
`include/parallel_mater/parallel_mater.h`. Applications discover a driver,
enumerate physical devices, create a logical device and queues, allocate and
bind resources, record physics commands, submit work, synchronize, and query
results. The gallery and conformance runner use this API rather than a private
simulation interface.

Physics equations, couplings, GPU primitives, and scheduling are shared Slang
and common simulation code. Device adapters provide allocation, dispatch,
transfer, synchronization, queries, and native interoperability only. Renderer
adapters draw public output and do not own simulation behavior or input.

Current priorities are to qualify each device adapter against the same command
API conformance corpus and gallery catalog, finish moving backend-neutral
resource descriptions out of device code, and keep platform rendering thin.
An adapter is advertised only after its supported features pass its own build,
conformance, gallery-control, and performance gates.

The completed pre-command-API implementation plan is retained in
[the legacy roadmap](history/ROADMAP-LEGACY.md).
